# Research 149: FlashAR — Diagonal-Step Parallel Decoding

**Paper:** [arXiv:2605.09430](https://arxiv.org/pdf/2605.09430) — FlashAR: Efficient Post-Training Acceleration for Autoregressive Image Generation
**Date:** 2025-06
**Status:** Distilled — **GAIN** (Tri-Mode Enhancement)

---

## Implementation Reference

Source: [lxazjk/Emu3.5-FlashAR](https://github.com/lxazjk/Emu3.5-FlashAR) (`flashar/model/modeling_emu_flashar.py`)

Key code patterns:

### Fusion Gate (actual)
```python
# g_pq = σ(MLP([h_feat_left || v_feat_up]))
gate_proj_dim = max(64, hidden_size // 8)
hv_gate_mlp = nn.Sequential(
    nn.Linear(2 * hidden_size, gate_proj_dim, bias=False),  # concat both hidden states
    nn.SiLU(),
    nn.Linear(gate_proj_dim, 1, bias=True),                  # scalar gate
)
# Init zeros → sigmoid(0) = 0.5 (symmetric start)

# Boundary rules:
#   First row → horizontal-only (from left neighbor)
#   First col → vertical-only (from up neighbor)
#   Interior  → g * h_logits + (1-g) * v_logits
```

### Vertical Branch (actual)
```python
# Clone decoder layers from backbone at split point
vertical_start_layer = backbone_num_layers - vertical_layers
vertical_block = copy.deepcopy(backbone_layers[vertical_start_layer:])
# Force non-causal attention — key!
for layer in vertical_block:
    layer.self_attn.is_causal = False
```

### Diagonal Step Loop (actual)
```python
step_id = row + col  # for each flattened position
max_step = height + width - 2
for step in range(max_step + 1):
    positions = (step_id == step).nonzero()  # all (h,w) where h+w==step
    logits = compute_from_prev_diagonal(prev_h_hidden, prev_v_hidden)
    tokens = sample(logits)
    grid[positions] = tokens
    # Append to BOTH KV caches (backbone + vertical)
    backbone_cache = append_backbone_kv(tokens, positions)
    vertical_cache = append_vertical_kv(hidden, positions)
    prev_h_hidden = current_h_hidden
    prev_v_hidden = current_v_hidden
```

### Proximity Mask
```text
allow[q, k] = True iff (row_k + col_k) <= (row_q + col_q)
= All prior diagonals are visible + full text prefix
```

---

## Paper Summary

FlashAR adapts a pre-trained raster-scan AR model into a **diagonal-step parallel generator**:

1. **Diagonal-step factorization**: Partition 2D token grid into anti-diagonals D_t = {y_{p,q} | p+q=t}. Tokens on the same diagonal are conditionally independent → generate in parallel. Reduces iterations from O(HW) → O(H+W). **22.9× wall-clock speedup**.
2. **Dual-head intermediate branching**: Original AR head → horizontal (row-wise). New vertical head branches from intermediate layer m (not final layer — final layer is too specialized to the original objective). Both share trunk, branch at depth m.
3. **Learnable fusion gate**: σ(MLP([h_H; h_V])) per position, dynamically blends the two complementary predictions. Outperforms fixed averaging (FID 4.12 vs 4.36). Key insight: different positions have different optimal direction weights.
4. **Two-stage post-training**: Stage 1 freezes backbone, trains vertical head + gate only. Stage 2 unfreezes all, joint fine-tune. Uses 0.05% of original data.
5. **Hardware-aware inference**: FlexAttention sparse 2D proximity masks + batched KV-cache updates.

**Results**: LlamaGen-L FID 3.16 (vs 3.80 AR), 224.7 img/s (vs 47.1). Emu3.5-34B 5.68s vs 130.1s (22.9×). Only 0.05% training data.

---

## Abstract Principles (Domain-Independent)

Strip away the 2D image specifics. What are the *abstract* principles?

1. **Multi-direction prediction**: Two prediction pathways with complementary bias → fused per-position. Reduces single-path blind spots.
2. **Intermediate branching**: New prediction head branches from intermediate layer (not final). Final layer is over-specialized to original objective; intermediate layers retain richer representations.
3. **Per-position adaptive fusion**: Static combination (averaging) is suboptimal. A learned gate that varies per-position captures directional anisotropy.
4. **Consensus enables parallel acceptance**: If two independent pathways agree at a position, confidence is high → can skip verification at that position.
5. **Stratified generation order**: Generate "easy" positions first (sparse anchors), then fill remaining with bidirectional context from anchors.

---

## Distillation to Our Stack

### Where the Gain Lives: Tri-Mode D2F+AR Pipeline

Our `tri_mode` (Plan 089) already mixes AR + diffusion:
- D2F drafts B tokens in parallel via iterative denoising
- AR verifies sequentially via prefix-match acceptance
- Accepts longest prefix + bonus token

**The bottleneck**: prefix-match acceptance wastes good predictions at non-contiguous positions. If D2F is correct at positions 2,4,6 but wrong at 1,3,5, current tri_mode only accepts position 0. All the correct non-adjacent predictions are thrown away.

FlashAR's fusion gate solves exactly this problem: **per-position acceptance** instead of prefix-match.

### Four Distillation Ideas

#### Idea 1: Dual-Path Consensus Draft with Ternary Thermal Paths (Highest Impact)

**FlashAR analog**: Horizontal head + Vertical head → fusion gate → per-position acceptance.

**Our mapping**:
- **Path H** (horizontal/AR): MTP drafter or `dflash_predict_ar_with` — left-to-right autoregressive, strong at local next-token
- **Path V** (vertical/diffusion): D2F block decode — bidirectional within block, strong at non-local position prediction
- **Fusion**: Per-position confidence-weighted selection instead of prefix-match

**Ternary thermal path encoding**:
```text
Per-position consensus encoded as ternary {-1, 0, +1}:
  +1 → Path H (MTP) wins   — H confidence > V confidence, they disagree
   0  → CONSENSUS           — h_i == v_i, both agree → SKIP VERIFICATION
  -1 → Path V (D2F) wins   — V confidence > H confidence, they disagree

Thermal paths (derived from ternary consensus):
  PLASMA path  (hottest): consensus + both high conf  → accept, zero compute
  HOT path:     ternary ±1 + winner high conf        → accept winner, quick spot-check
  WARM path:    ternary ±1 + moderate conf           → AR verify only this position
  COLD path:    both low confidence                   → fallback to prefix-match
```

**Why ternary?** The consensus result is *naturally* ternary. And our existing `TernaryWeights` + `simd_ternary_matvec` infrastructure (Plan 148) already provides SIMD-accelerated ternary operations with zero multiplication. The fusion gate decision becomes:

```text
  // FlashAR uses: σ(MLP([h_feat || v_feat])) → 2*hidden_size → gate
  // Our ternary analog:
  // Concatenate DiffusionSampler features from BOTH paths (6+6 = 12 dims)
  // Ternary SIMD gate: 12 features × ternary weights → thermal path score
  // Same kernel as plasma_path: acc += (pos_bits & x) - (neg_bits & x)
  // Zero multiplication — pure SIMD add/subtract
  simd_ternary_matvec(&gate_weights, &dual_path_features_12d, &mut fusion_scores)
```

This means the fusion gate is **multiplication-free** — the same optimization that makes plasma_path fast for matmul applies to the consensus fusion decision. The ternary encoding maps directly:
- `pos_bits[i] = 1` when position i favors Path H
- `neg_bits[i] = 1` when position i favors Path V
- Both 0 = agreement (plasma path)

The 12-dim input mirrors FlashAR's `[h_feat || v_feat]` concatenation pattern — using DiffusionSampler features (top1_prob, margin, top3_mass, entropy, step_norm, pos_norm) from each path.

**Why this helps**: FlashAR shows that independent predictions from different pathways have different failure modes. When they agree, error rate drops dramatically. Our MTP and D2F are architecturally different:
- MTP: autoregressive, forward-only context, biased toward local patterns
- D2F: bidirectional within block, iterative refinement, better at "seeing the whole block"

Their agreement is a much stronger signal than either alone. The ternary thermal path encoding lets us route each position to the cheapest safe verification level — plasma (zero compute) for consensus positions, warm/cold only for disputed ones.

**Expected gain**: 1.5-2× throughput improvement over current tri_mode by reducing AR verification load. Additional gain from ternary SIMD fusion gate (zero-multiply decision).

#### Idea 2: Strided Anchor-Then-Fill D2F

**FlashAR analog**: Diagonal-step generation — generate sparse anchors first (every k-th token), then fill gaps with bidirectional context.

**Our mapping**:
```text
Current D2F: block 0 → block 1 → block 2 → block 3 (sequential blocks)

Strided D2F:
  Round 1 (anchors): Predict positions [0, S, 2S, 3S, ...] via AR (independent, batched)
  Round 2 (fill): D2F decode remaining positions with bidirectional context from anchors
  → Fewer denoising iterations needed because anchors provide structural guidance
```

This is the direct analog of diagonal-step: the "anchors" are the first diagonal, the "fill" is subsequent diagonals that use previous diagonal context.

**Why this helps**: D2F currently starts each block with no future context (all masked). With pre-placed anchors, the denoising has a "skeleton" to fill around. FlashAR shows this converges in fewer iterations.

**Expected gain**: 20-40% reduction in D2F denoising steps when anchors are available.

#### Idea 3: Intermediate-Layer MTP Branch (Model-Based, riir-ai)

**FlashAR analog**: Vertical head branches from layer m, not final layer, because final layer is over-specialized.

**Our mapping**: MTP projection currently uses the final layer's hidden state. FlashAR shows intermediate layers retain richer representations for alternative prediction tasks.

In riir-ai's LoRA training, the MTP LoRA adapter could:
- Project from layer L-1 or L-2 instead of final layer
- Use a separate LoRA adapter for the "vertical" (multi-position) prediction
- This follows FlashAR's finding: branching at depth m gives better results than m=L

**This requires model training changes → riir-ai domain.**

---

## Verdict by 003 Commercial Strategy

### GOAT Criteria

| Criterion | Assessment |
|---|---|
| Improves core inference pipeline? | ✅ Yes — tri_mode acceptance rate directly improves |
| Replaces existing functionality? | No — enhances tri_mode, doesn't replace |
| Requires model training? | Ideas 1-2: No (modelless). Idea 3: Yes (model-based) |
| Performance risk? | Low — opt-in feature flag, falls back to prefix-match |
| Commercial moat impact? | ✅ Higher acceptance rate → faster RIIR → better product |

### Alignment with optimization.md

| Principle | Compliance |
|---|---|
| Profile first | ✅ Will benchmark vs current tri_mode |
| Pre-allocated buffers | ✅ Reuse SpeculativeContext, D2fContext |
| Fixed-size arrays | ✅ Block size bounded by draft_width |
| No allocation in hot loops | ✅ Stack arrays for consensus bitfield |
| Benchmark before/after | ✅ GOAT proof required |

### Modelless vs Model-Based Split

| Idea | Project | Feature Gate | Training Required |
|---|---|---|---|
| Dual-Path Consensus + Ternary Thermal Paths | katgpt-rs | `flashar_consensus` (under `tri_mode`) | No — uses MTP + D2F + DiffusionSampler + `simd_ternary_matvec` |
| Strided Anchor-Then-Fill | katgpt-rs | `flashar_anchor` (under `dllm`) | No — modelless decode strategy |
| Intermediate-Layer Branch | riir-ai | `flashar_mtp_branch` | Yes — LoRA training change |
| Ternary Fusion Gate Training | riir-ai | `flashar_gate_ternary` | Yes — ternary weight distillation for gate |

---

## Decision: **GAIN**

FlashAR's abstract principles — dual-path consensus, per-position fusion, intermediate branching, and stratified generation order — transfer meaningfully to our tri_mode pipeline. The core insight is:

**Consensus from two architecturally different prediction pathways (AR + D2F) is a much stronger signal than either alone, enabling per-position acceptance instead of prefix-match. The ternary {-1, 0, +1} consensus encoding maps directly to our plasma_path SIMD infrastructure, making the fusion gate multiplication-free.**

This directly addresses tri_mode's biggest bottleneck: wasted good predictions at non-contiguous positions.

### Concrete Gain Estimate

Current tri_mode acceptance: ~40-60% of drafted tokens (prefix-match limited).
FlashAR-inspired consensus acceptance: ~70-90% (per-position, non-contiguous).

That's potentially **1.5-2× more tokens accepted per draft cycle** → direct decode throughput improvement without any model changes.

### Bonus: Plasma Path Synergy

The ternary consensus encoding isn't just an encoding trick — it reuses our existing SIMD ternary matvec infrastructure (`simd_ternary_matvec` from Plan 148). This means:
1. **Zero new SIMD kernels** — the same Neon/AVX2 paths that accelerate plasma_path matmul accelerate the consensus fusion gate
2. **Zero multiplication** — the fusion decision is pure add/subtract on bitmask, same as plasma_path
3. **Thermal path routing** — ternary magnitude naturally maps to plasma/hot/warm/cold verification paths
4. **Binary-bloat free** — reuses existing compiled SIMD paths, no new code in the icache

### Scorecard

| Criterion | Rating | Notes |
|---|---|---|
| Core idea transferability | ⭐⭐⭐⭐☆ | Multi-path consensus is domain-independent |
| Supporting technique overlap | ⭐⭐⭐⭐⭐ | All prerequisites exist (MTP, D2F, DiffusionSampler) |
| Implementation cost | ⭐⭐⭐⭐☆ | Moderate — extends tri_mode, no new models |
| Performance impact | ⭐⭐⭐⭐☆ | 1.5-2× acceptance improvement estimated |
| Commercial relevance | ⭐⭐⭐⭐☆ | Faster RIIR inference = better product |
| **Overall** | **GAIN** | **Plan 166: FlashAR Consensus Tri-Mode** |
