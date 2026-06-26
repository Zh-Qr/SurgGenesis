"""
World Model DiT with Model Parallelism (Pipeline Parallel)

将 30 层 Transformer 分片到多张卡上：
- 4 卡: 每张卡 7-8 层
- 3 卡: 每张卡 10 层  
- 2 卡: 每张卡 15 层

使用方法:
    model = WorldModelDiTMP(
        devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
        ...
    )
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldmodel_dit import (
    RMSNorm, timestep_embedding, CausalSpatioTemporalAttention,
    FeedForward, WorldModelDiTBlock, FinalLayer
)


class WorldModelDiTMP(nn.Module):
    """
    World Model DiT with Model Parallelism
    
    将 Transformer blocks 均匀分配到多个 GPU 上
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
        devices: List[str] = None,  # ⭐ 例如 ['cuda:0', 'cuda:1', 'cuda:2']
    ):
        super().__init__()
        
        if devices is None:
            devices = ['cuda:0']
        
        self.devices = [torch.device(d) for d in devices]
        self.num_devices = len(self.devices)
        self.latent_channels = latent_channels
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        pt, ph, pw = patch_size
        self.patch_volume = latent_channels * pt * ph * pw
        
        # =====================================================================
        # 输入层放在第一个设备
        # =====================================================================
        device_0 = self.devices[0]
        
        self.patch_embed = nn.Conv3d(
            in_channels=latent_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        ).to(device_0)
        
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, 64, dim, device=device_0) * 0.02
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        ).to(device_0)
        
        # =====================================================================
        # Transformer blocks 分片到不同设备
        # =====================================================================
        self.blocks = nn.ModuleList()
        self.block_devices = []  # 记录每个 block 在哪个设备
        
        # 计算每个设备分配多少层
        blocks_per_device = [depth // self.num_devices] * self.num_devices
        for i in range(depth % self.num_devices):
            blocks_per_device[i] += 1
        
        print(f"[Model Parallel] Splitting {depth} blocks across {self.num_devices} devices:")
        
        block_idx = 0
        for device_idx, device in enumerate(self.devices):
            num_blocks = blocks_per_device[device_idx]
            print(f"  Device {device_idx} ({device}): Blocks {block_idx} - {block_idx + num_blocks - 1}")
            
            for _ in range(num_blocks):
                block = WorldModelDiTBlock(dim, num_heads, ffn_hidden, dropout).to(device)
                self.blocks.append(block)
                self.block_devices.append(device)
                block_idx += 1
        
        # =====================================================================
        # 输出层放在最后一个设备
        # =====================================================================
        device_last = self.devices[-1]
        
        self.final_layer = FinalLayer(dim, self.patch_volume).to(device_last)
        
        # Initialize
        self._initialize_weights()
        
        print(f"[Model Parallel] Model created with {self.compute_params() / 1e9:.2f}B params")
    
    def _initialize_weights(self):
        """Initialize weights"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def patchify(self, z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Patchify on device 0
        z: [B, C, T, H, W] on any device
        returns: [B, N, dim] on device 0, (Tp, Hp, Wp)
        """
        device_0 = self.devices[0]
        z = z.to(device_0)
        
        B, C, T, H, W = z.shape
        pt, ph, pw = self.patch_size
        
        # Pad
        pad_t = (pt - T % pt) % pt
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            z = F.pad(z, (0, pad_w, 0, pad_h, 0, pad_t))
        
        # Patch embed
        x = self.patch_embed(z)  # [B, dim, Tp, Hp, Wp]
        Tp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]
        
        # Reshape
        x = x.flatten(2).transpose(1, 2)  # [B, Tp*Hp*Wp, dim]
        
        return x, (Tp, Hp, Wp)
    
    def unpatchify(
        self, 
        x: torch.Tensor, 
        patch_shape: Tuple[int, int, int],
        original_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Unpatchify on last device
        x: [B, N, patch_volume] on last device
        returns: [B, C, T, H, W] on last device
        """
        B, N, _ = x.shape
        Tp, Hp, Wp = patch_shape
        T_orig, H_orig, W_orig = original_shape
        pt, ph, pw = self.patch_size
        
        x = x.reshape(B, Tp, Hp, Wp, self.latent_channels, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.reshape(B, self.latent_channels, Tp * pt, Hp * ph, Wp * pw)
        
        # Crop
        x = x[:, :, :T_orig, :H_orig, :W_orig]
        
        return x
    
    def forward(
        self,
        z: torch.Tensor,           # [B, C, T, H, W] on any device
        t: torch.Tensor,           # [B] on any device
        t_cond: int,
    ) -> torch.Tensor:
        """
        Forward with model parallelism
        
        Pipeline:
        1. Input on device 0
        2. Blocks distributed across devices
        3. Output on last device
        """
        B, C, T, H, W = z.shape
        device_0 = self.devices[0]
        device_last = self.devices[-1]
        
        # =====================================================================
        # Stage 1: Patchify on device 0
        # =====================================================================
        x, (Tp, Hp, Wp) = self.patchify(z)  # x on device_0
        
        # Add position embeddings
        temporal_pos = self.temporal_pos_embed[:, :Tp, :].repeat(1, Hp * Wp, 1)
        x = x + temporal_pos.reshape(1, Tp * Hp * Wp, self.dim)
        
        # Timestep embedding
        t = t.to(device_0)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(B)
        elif t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        
        t_emb = timestep_embedding(t, 256)
        t_emb = self.time_embed(t_emb)  # [B, dim] on device_0
        
        # =====================================================================
        # Stage 2: Transformer blocks (with pipeline)
        # =====================================================================
        for i, (block, block_device) in enumerate(zip(self.blocks, self.block_devices)):
            # Move data to current block's device
            if x.device != block_device:
                x = x.to(block_device)
                t_emb = t_emb.to(block_device)
            
            # Forward through block
            x = block(x, t_emb, t_cond, (Tp, Hp, Wp))
        
        # =====================================================================
        # Stage 3: Final layer on last device
        # =====================================================================
        if x.device != device_last:
            x = x.to(device_last)
            t_emb = t_emb.to(device_last)
        
        x = self.final_layer(x, t_emb)
        
        # =====================================================================
        # Stage 4: Unpatchify
        # =====================================================================
        out = self.unpatchify(x, (Tp, Hp, Wp), (T, H, W))
        
        return out
    
    def compute_params(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())


# =====================================================================
# 辅助函数: 自动检测可用 GPU 并创建模型
# =====================================================================

def create_worldmodel_mp_auto(
    model_size: str = 'xlarge',
    max_gpus: int = None,
) -> WorldModelDiTMP:
    """
    自动检测可用 GPU 并创建模型
    
    Args:
        model_size: 'base' | 'large' | 'xlarge'
        max_gpus: 最多使用多少张卡（None = 全部）
    """
    # 检测可用 GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")
    
    if max_gpus is not None:
        num_gpus = min(num_gpus, max_gpus)
    
    devices = [f'cuda:{i}' for i in range(num_gpus)]
    
    print(f"[Auto] Detected {torch.cuda.device_count()} GPUs, using {num_gpus}")
    
    # 根据模型大小选择配置
    configs = {
        'base': dict(
            latent_channels=16,
            dim=768,
            depth=12,
            num_heads=12,
            ffn_hidden=3072,
        ),
        'large': dict(
            latent_channels=16,
            dim=1024,
            depth=24,
            num_heads=16,
            ffn_hidden=4096,
        ),
        'xlarge': dict(
            latent_channels=16,
            dim=1536,
            depth=30,
            num_heads=24,
            ffn_hidden=8960,
        ),
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config = configs[model_size]
    
    return WorldModelDiTMP(
        **config,
        patch_size=(1, 2, 2),
        dropout=0.0,
        devices=devices,
    )


# =====================================================================
# Gradient Checkpointing (可选，进一步节省显存)
# =====================================================================

class WorldModelDiTMPWithCheckpoint(WorldModelDiTMP):
    """
    带 Gradient Checkpointing 的版本
    进一步节省显存，但会略微增加训练时间
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpoint = True
    
    def forward(self, z, t, t_cond):
        B, C, T, H, W = z.shape
        device_0 = self.devices[0]
        device_last = self.devices[-1]
        
        # Patchify
        x, (Tp, Hp, Wp) = self.patchify(z)
        
        # Position + time embedding
        temporal_pos = self.temporal_pos_embed[:, :Tp, :].repeat(1, Hp * Wp, 1)
        x = x + temporal_pos.reshape(1, Tp * Hp * Wp, self.dim)
        
        t = t.to(device_0)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(B)
        elif t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        
        t_emb = timestep_embedding(t, 256)
        t_emb = self.time_embed(t_emb)
        
        # Transformer with checkpoint
        for i, (block, block_device) in enumerate(zip(self.blocks, self.block_devices)):
            if x.device != block_device:
                x = x.to(block_device)
                t_emb = t_emb.to(block_device)
            
            # ⭐ Use checkpoint for backward pass
            if self.training and self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, t_emb, t_cond, (Tp, Hp, Wp), use_reentrant=False
                )
            else:
                x = block(x, t_emb, t_cond, (Tp, Hp, Wp))
        
        # Final layer
        if x.device != device_last:
            x = x.to(device_last)
            t_emb = t_emb.to(device_last)
        
        x = self.final_layer(x, t_emb)
        
        # Unpatchify
        out = self.unpatchify(x, (Tp, Hp, Wp), (T, H, W))
        
        return out


# =====================================================================
# Test
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Model Parallel World Model DiT")
    print("=" * 60)
    
    # Test with available GPUs
    try:
        model = create_worldmodel_mp_auto(model_size='xlarge', max_gpus=4)
        
        # Forward test
        B, C, T, H, W = 1, 16, 13, 16, 24
        z = torch.randn(B, C, T, H, W, device='cuda:0')
        t = torch.tensor([500], device='cuda:0')
        t_cond = 4
        
        print(f"\nInput: {tuple(z.shape)} on {z.device}")
        
        with torch.no_grad():
            out = model(z, t, t_cond)
        
        print(f"Output: {tuple(out.shape)} on {out.device}")
        print(f"\n✅ Model parallel test passed!")
        
        # Memory usage
        for i, device in enumerate(model.devices):
            mem_allocated = torch.cuda.memory_allocated(device) / 1e9
            mem_reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"GPU {i}: Allocated {mem_allocated:.2f}GB, Reserved {mem_reserved:.2f}GB")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()