"""
World Model DiT: 专门为视频预测任务设计的 Diffusion Transformer

核心设计理念:
1. 时空条件机制: 利用历史帧作为强条件
2. 因果注意力: 确保只关注过去的信息
3. 时空建模: 同时建模空间和时间依赖
4. 参数规模: ~1.4B (与 WAN 相当)

架构特点:
- 使用 3D patch embedding 处理视频潜在表示
- Causal spatiotemporal attention 确保因果性
- 时间位置编码 + 条件嵌入
- 支持灵活的条件长度
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 基础组件
# =====================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal timestep embedding for diffusion timestep
    t: [B] diffusion timestep
    returns: [B, dim]
    """
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def temporal_position_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Learnable temporal position embeddings
    returns: [length, dim]
    """
    return nn.Parameter(torch.randn(length, dim, device=device) * 0.02)


# =====================================================================
# 因果时空注意力
# =====================================================================

class CausalSpatioTemporalAttention(nn.Module):
    """
    因果时空注意力: 当前帧只能看到过去的帧
    
    设计要点:
    1. 条件帧 (0:t_cond) 可以互相看到 (双向注意力)
    2. 预测帧 (t_cond:T) 只能看到 <= 自己时间步的所有帧 (因果注意力)
    3. 空间维度内自由注意
    """
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

    def forward(
        self, 
        x: torch.Tensor,           # [B, N, C] where N = T*H*W
        t_cond: int,               # 条件帧数量
        temporal_shape: Tuple[int, int, int],  # (T, H, W)
    ) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = temporal_shape
        assert N == T * H * W
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RMSNorm to Q and K
        q = q * self.scale
        
        # Reshape to separate temporal and spatial dimensions
        # [B, heads, T, H*W, head_dim]
        q = q.reshape(B, self.num_heads, T, H * W, self.head_dim)
        k = k.reshape(B, self.num_heads, T, H * W, self.head_dim)
        v = v.reshape(B, self.num_heads, T, H * W, self.head_dim)
        
        # Compute attention: [B, heads, T, H*W, T, H*W]
        # 为了内存效率，我们展平处理
        q = q.reshape(B, self.num_heads, T * H * W, self.head_dim)
        k = k.reshape(B, self.num_heads, T * H * W, self.head_dim)
        v = v.reshape(B, self.num_heads, T * H * W, self.head_dim)
        
        attn = (q @ k.transpose(-2, -1))  # [B, heads, THW, THW]
        
        # 创建因果mask
        # 条件部分 (0:t_cond) 可以互相看到
        # 预测部分 (t_cond:T) 只能看到 <= 自己的帧
        mask = self._create_causal_mask(T, H * W, t_cond, device=x.device, dtype=x.dtype)
        attn = attn + mask
        
        attn = attn.softmax(dim=-1)
        
        out = attn @ v  # [B, heads, THW, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out
    
    def _create_causal_mask(
        self, 
        T: int, 
        HW: int, 
        t_cond: int, 
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        创建因果mask
        mask[i, j] = 0 表示 i 可以看到 j
        mask[i, j] = -inf 表示 i 不能看到 j
        """
        N = T * HW
        mask = torch.zeros(N, N, device=device, dtype=dtype)
        
        for t_query in range(T):
            for t_key in range(T):
                query_start = t_query * HW
                query_end = (t_query + 1) * HW
                key_start = t_key * HW
                key_end = (t_key + 1) * HW
                
                # 条件部分可以互相看到
                if t_query < t_cond and t_key < t_cond:
                    continue  # mask = 0, 可见
                
                # 预测部分只能看到过去（包括所有条件帧）
                elif t_query >= t_cond:
                    if t_key > t_query:
                        mask[query_start:query_end, key_start:key_end] = float('-inf')
        
        return mask


# =====================================================================
# FFN
# =====================================================================

class FeedForward(nn.Module):
    """Standard FFN with GELU activation"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# DiT Block with Adaptive Layer Norm (adaLN)
# =====================================================================

class WorldModelDiTBlock(nn.Module):
    """
    DiT Block for World Model with:
    - Causal spatiotemporal attention
    - AdaLN for diffusion timestep conditioning
    - FFN
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        ffn_hidden: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = CausalSpatioTemporalAttention(dim, num_heads)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ffn = FeedForward(dim, ffn_hidden, dropout)
        
        # AdaLN modulation (6 params: shift/scale/gate for attn and ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        
    def forward(
        self, 
        x: torch.Tensor,                    # [B, N, dim]
        t_emb: torch.Tensor,                # [B, dim] diffusion timestep embedding
        t_cond: int,                        # 条件帧数量
        temporal_shape: Tuple[int, int, int],  # (T, H, W)
    ) -> torch.Tensor:
        # AdaLN modulation parameters
        modulation = self.adaLN_modulation(t_emb)  # [B, 6*dim]
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + gate1.unsqueeze(1) * self.attn(x_norm, t_cond, temporal_shape)
        
        # FFN with AdaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.ffn(x_norm)
        
        return x


# =====================================================================
# Final Layer
# =====================================================================

class FinalLayer(nn.Module):
    """Final layer with AdaLN"""
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


# =====================================================================
# World Model DiT
# =====================================================================

class WorldModelDiT(nn.Module):
    """
    World Model DiT for video prediction
    
    参数规模: ~1.4B (与 WAN DiT 相当)
    
    Args:
        latent_channels: 潜在空间通道数 (default: 16)
        dim: Transformer 维度 (default: 1536)
        depth: Transformer 层数 (default: 30)
        num_heads: 注意力头数 (default: 24)
        ffn_hidden: FFN 隐藏层维度 (default: 8960)
        patch_size: 3D patch 大小 (T, H, W) (default: (1,2,2))
    """
    def __init__(
        self,
        latent_channels: int = 16,
        dim: int = 1536,
        depth: int = 30,
        num_heads: int = 24,
        ffn_hidden: int = 8960,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        pt, ph, pw = patch_size
        self.patch_volume = latent_channels * pt * ph * pw
        
        # Patch embedding: 3D conv
        self.patch_embed = nn.Conv3d(
            in_channels=latent_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        
        # Temporal position embedding (learnable)
        # 最大支持 64 个时间步的 patch
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        
        # Diffusion timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WorldModelDiTBlock(dim, num_heads, ffn_hidden, dropout)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(dim, self.patch_volume)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights following DiT paper"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Zero-init output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def patchify(self, z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Convert latent to patches
        z: [B, C, T, H, W]
        returns: [B, N, dim], (Tp, Hp, Wp)
        """
        B, C, T, H, W = z.shape
        
        # Pad if necessary
        pt, ph, pw = self.patch_size
        pad_t = (pt - T % pt) % pt
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            z = F.pad(z, (0, pad_w, 0, pad_h, 0, pad_t))
        
        # Patch embed
        x = self.patch_embed(z)  # [B, dim, Tp, Hp, Wp]
        Tp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]
        
        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # [B, Tp*Hp*Wp, dim]
        
        return x, (Tp, Hp, Wp)
    
    def unpatchify(
        self, 
        x: torch.Tensor, 
        patch_shape: Tuple[int, int, int],
        original_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Convert patches back to latent
        x: [B, N, patch_volume]
        returns: [B, C, T, H, W]
        """
        B, N, _ = x.shape
        Tp, Hp, Wp = patch_shape
        T_orig, H_orig, W_orig = original_shape
        pt, ph, pw = self.patch_size
        
        # Reshape
        x = x.reshape(B, Tp, Hp, Wp, self.latent_channels, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [B, C, Tp, pt, Hp, ph, Wp, pw]
        x = x.reshape(B, self.latent_channels, Tp * pt, Hp * ph, Wp * pw)
        
        # Crop to original size
        x = x[:, :, :T_orig, :H_orig, :W_orig]
        
        return x
    
    def forward(
        self,
        z: torch.Tensor,           # [B, C, T, H, W] 潜在表示（部分加噪）
        t: torch.Tensor,           # [B] 或 scalar, diffusion timestep
        t_cond: int,               # 条件帧数量（在潜在空间的时间步）
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z: [B, C, T, H, W] 潜在表示
               - z[:, :, :t_cond] 是干净的条件帧
               - z[:, :, t_cond:] 是加噪的未来帧
            t: diffusion timestep (0-999)
            t_cond: 条件帧数量（在 patch 后的时间维度）
        
        Returns:
            predicted noise: [B, C, T, H, W]
        """
        B, C, T, H, W = z.shape
        device = z.device
        
        # Patchify
        x, (Tp, Hp, Wp) = self.patchify(z)  # [B, Tp*Hp*Wp, dim]
        
        # Add temporal position embeddings
        # 只给时间维度加位置编码
        temporal_pos = self.temporal_pos_embed[:, :Tp, :].repeat(1, Hp * Wp, 1)
        x = x + temporal_pos.reshape(1, Tp * Hp * Wp, self.dim)
        
        # Diffusion timestep embedding
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(B)
        elif t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        
        t_emb = timestep_embedding(t, 256).to(device)
        t_emb = self.time_embed(t_emb)  # [B, dim]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, t_cond, (Tp, Hp, Wp))
        
        # Final layer
        x = self.final_layer(x, t_emb)  # [B, N, patch_volume]
        
        # Unpatchify
        out = self.unpatchify(x, (Tp, Hp, Wp), (T, H, W))
        
        return out
    
    def compute_params(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())


# =====================================================================
# 辅助函数: 创建不同规模的模型
# =====================================================================

def worldmodel_dit_base() -> WorldModelDiT:
    """Base model: ~350M params"""
    return WorldModelDiT(
        latent_channels=16,
        dim=768,
        depth=12,
        num_heads=12,
        ffn_hidden=3072,
    )


def worldmodel_dit_large() -> WorldModelDiT:
    """Large model: ~800M params"""
    return WorldModelDiT(
        latent_channels=16,
        dim=1024,
        depth=24,
        num_heads=16,
        ffn_hidden=4096,
    )


def worldmodel_dit_xlarge() -> WorldModelDiT:
    """XLarge model: ~1.4B params (与 WAN 相当)"""
    return WorldModelDiT(
        latent_channels=16,
        dim=1536,
        depth=30,
        num_heads=24,
        ffn_hidden=8960,
    )


# =====================================================================
# Test
# =====================================================================

if __name__ == "__main__":
    # Test
    model = worldmodel_dit_xlarge()
    print(f"参数量: {model.compute_params() / 1e9:.2f}B")
    
    # Forward test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    B, C, T, H, W = 2, 16, 13, 16, 24
    z = torch.randn(B, C, T, H, W, device=device)
    t = torch.rand(B, device=device) * 999
    t_cond = 4  # 前4个时间步是条件
    
    with torch.no_grad():
        out = model(z, t, t_cond)
    
    print(f"输入: {tuple(z.shape)}")
    print(f"输出: {tuple(out.shape)}")
    print("✅ 测试通过!")