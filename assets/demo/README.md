# Demo Media

Curated qualitative results for the [main README](../../README.md).

## `gifs/`

Labeled, side-by-side comparison strips (README-embeddable). Each shows the **same conditioning frames** rolled forward by the three curriculum stages:

```
Stage-1 | Stage-2 | Stage-3 (ours)
```

The base model and ground-truth future are intentionally **omitted from the GIFs** (they are qualitative teasers of the progressive refinement); the full `base` / `ground_truth_future` clips and the quantitative comparison against both remain available in `videos/` and in the [results tables](../tables/).

| File | Notes |
|------|-------|
| `hero_comparison.gif` | headline case (= `case_03`) |
| `case_03_comparison.gif`, `case_05_comparison.gif` | shown in the main README |
| `case_09_comparison.gif`, `case_13_comparison.gif` | additional held-out Cholec80 cases |

## `videos/`

Full-resolution `.mp4` clips (832×480, 15 FPS) for each curated case, plus `case_metadata.json` (source clip + prompt):

```
videos/case_XX/
├── condition.mp4              # 41 conditioning frames (input)
├── ground_truth_future.mp4    # real surgical continuation (target)
├── base_generated.mp4         # WAN2.2 base
├── stage1_generated.mp4
├── stage2_generated.mp4
├── stage3_generated.mp4       # ours
└── case_metadata.json
```

These clips were taken from a `scripts/40_eval_three_stage_wan22ti2v.sh` run. The GIFs were generated from these `.mp4`s by down-sampling to ~9 FPS and tiling the panels with labels.
