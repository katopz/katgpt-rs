# Research 233: Fast KV Compaction via Attention Matching — Modelless Distillation

**Date:** 2026-06-14
**Source:** [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284) (Zweiger, Fu, Guo, Kim — MIT, ICML 2026)
**Repo:** [adamzweiger/compaction](https://github.com/adamzweiger/compaction) ( vendored under `katgpt-rs/.raw/compaction/`)
**Target:** `katgpt-rs` modelless inference engine — new `src/attn_match/` module
**Status:** Verdict = **GAIN** — direct port + novel CPU/SIMD/GPU/ANE adaptive solver routing. Plan 265.

---

## Paper Summary

### The Problem

Long-context LM inference is bottlenecked by KV cache size. Existing approaches:
- **Token eviction** (H2O, SnapKV, PyramidKV, KVzip) — degrade rapidly past ~10× compaction
- **Token merging** (KVMerger) — local averaging, loses global structure
- **Summarization** (Claude Code, OpenAI Codex) — lossy, fails on dense retrieval
- **Cartridges** (Eyuboglu et al., 2025) — latent-space compaction via gradient-based prefix-tuning. Quality is excellent but **5+ GPU-hours per context** to train a Cartridge.

### The Insight: Mass-Preserving Attention Matching

When we replace `(K, V)` with compacted `(Ck, β, Cv)`, attention over `[K; Kfixed]` decomposes into a mixture whose weights are determined by unnormalized attention **mass**:

```
Attn(q; [K; Kfixed], [V; Vfixed]) =
    Mass(q; K) / (Mass(q; K) + Mass(q; Kfixed)) · Attn(q; K, V) +
    Mass(q; Kfixed) / (Mass(q; K) + Mass(q; Kfixed)) · Attn(q; Kfixed, Vfixed)
```

Therefore, to preserve attention behavior under concatenation with **any** future `(Kfixed, Vfixed)`, it suffices to match two quantities on a set of reference queries `Qref`:

1. **Attention output** (Eq. 1): `softmax(qK^T/√d) V ≈ softmax((qCk^T + β)/√d) Cv`
2. **Attention mass** (Eq. 2): `Σ_j exp(qK_j^T/√d) ≈ Σ_j exp((qCk_j^T + β_j)/√d)`

### Why β is Critical (Showstopper if Missing)

With `t < T` and no bias, `Mass(q; Ck) ≤ Mass(q; K)` always — compaction systematically underestimates attention contribution. β introduces multiplicative weights `exp(β_j)` so each retained key can represent the mass of many removed keys.

This is the same β that StillKV (Research 213, Plan 245) needs. **StillKV's heuristic β (log(T/t), vortex flow) is APPROXIMATE. AM's β is OPTIMAL via NNLS.**

### The Method (Closed-Form, No Gradient Descent)

Given reference queries `Qref = [q_1; ...; q_n]` and compact keys `Ck`:

**Step 1: Fit β via NNLS.**
- Target: `m_i = Σ_k exp(q_i K_k^T / √d)` (original mass per query)
- Feature matrix: `A_ij = exp(q_i (Ck)_j^T / √d)`, with `w_j = exp(β_j)`
- Solve: `min_{w ≥ 0} ||A w - m||_2^2` via projected gradient descent
- Recover: `β_j = log(w_j)`

**Step 2: Fit Cv via Ordinary Least Squares.**
- Target: `Y_i = softmax(q_i K^T / √d) V` (original attention output per query)
- Design: `X_i = softmax((q_i Ck^T + β) / √d)`
- Solve: `Cv = (X^T X)^{-1} X^T Y` via normal equations + Cholesky

**Step 3: Select Ck** (the only non-closed-form piece). Two families:
- **HighestAttnKeys**: rank keys by RMS attention score across `Qref`, take top-t
- **OMP** (Orthogonal Matching Pursuit): greedily select keys whose `exp(q·k_j)` maximally reduces mass residual; refit `w` via NNLS every τ steps

### Reference Query Generation

- **Context-prefill**: cheapest, run prefill on `C` alone, extract query vectors
- **Repeat-prefill**: prompt `{C} Repeat the previous context. {C}`, extract queries during reconstruction
- **Self-study**: synthetic interactions (4 fixed prompts) — best quality, ~139s on 60k-token context
- **Random vectors**: `N(0, I_d)` — works, but lags
- **On-policy**: per-layer `Qref` extracted after layers `<ℓ` already compacted (slight gain)

### Nonuniform Head Budgets

Different attention heads have different sensitivity to compaction ratio (Figure 3 in paper). Key findings:
- Sensitivity curves are **input-invariant** — same ranking across contexts and datasets
- Therefore: **compute per-head budget schedule ONCE per model**, reuse across all contexts
- Algorithm 4: greedy swap — start uniform, iteratively move budget units between heads to minimize predicted loss

### Chunked Compaction

For long contexts:
- **KV-based**: prefill full context, slice per-chunk KV, compact independently, concatenate
- **Text-based**: prefill+compact each chunk in isolation, apply uniform RoPE phase shift to align to global positions
- KV-based is more faithful (preserves cross-chunk interactions in the prefilled state)

### Online Compaction (Appendix F.3)

Compact mid-trajectory to support arbitrarily long generation under fixed physical memory:
- Cap physical KV cache to budget `B`
- When model reaches `B`, compact entire context except most recent 20 tokens
- Continue decoding
- AIME proof-of-concept: 6 consecutive compactions preserve reasoning quality (13/30 → 13/30 with 4× more decoded tokens)

### Disaggregated Compaction (Appendix G.3)

- Compaction is a pure function of an existing KV snapshot — no gradient updates, no model modification
- A separate **compaction worker** reads prefix snapshot, computes compact cache
- Serving engine continues from original prefix, swaps at token boundary (e.g., between turns), reclaims memory
- Zero user-facing latency; cost is background compute + KV transfer

### Headline Results

| Metric | Value |
|--------|-------|
| Compaction ratio | Up to **50×** with minimal quality loss |
| Compaction time | **Seconds** (vs hours for Cartridges) |
| Speedup vs Cartridges | **~2 orders of magnitude** |
| QuALITY accuracy at 50× | AM exceeds Cartridges |
| RULER (kvpress) | AM-fast > KVzip (prior SOTA) and all leaderboard methods |
| AIME online compaction | Reasoning preserved across 6 mid-trajectory compactions |

---

## Distillation Constraint Check

| Constraint | Status |
|------------|--------|
| 1. Modelless first (inference-time only) | ✅ Pure post-hoc procedure on any pretrained model, no LLM training |
| 2. riir-ai engine/fuel split | ✅ AM is generic framework → public katgpt-rs; specific fusion recipes → private riir-ai |
| 3. LoRA only for training | ✅ No training in core AM (LoRA β predictor is an optional fusion — Research 121) |
| 4. Self-learning adaptive CoT welcome | ✅ Online compaction enables adaptive CoT trace compaction (no LLM training) |
| 5. SOLID, DRY | ✅ Reuses StillKV's `BetaBias` + `CompactKVCache`; trait-based selectors and fitters |
| 6. Tests/examples before/after | ✅ Required by Plan 265 — perplexity proxy + reconstruction tests |
| 7. CPU/GPU/ANE auto-route | ✅ Novel fusion: adaptive solver routing by tensor size (see below) |
| 8. Plasma/hot/warm/cold/freeze path | ✅ Tier-mapped in Research 121 (riir-ai fusion) |
| 9. Threshold-based CPU/SIMD/GPU/ANE | ✅ Size-thresholded router in this research |

---

## Novel Fusion Ideas (Modelless, katgpt-rs)

### Fusion A: Adaptive Solver Routing (CPU/SIMD/GPU/ANE) — Constraint 7, 9

The AM pipeline has three solver hotspots with very different compute profiles:
1. **QK^T score matrix** — `O(n·T·d)`, embarrassingly parallel
2. **NNLS for β** — projected gradient descent on `t × n` system
3. **Cholesky for Cv** — `O(t²·d + t³/3)`, dense

**Routing rules** (per AGENTS.md optimization rules — pre-allocate, no allocs in hot loops, SIMD chunked loops):

| Solver stage | Tensor regime | Backend | Rationale |
|---|---|---|---|
| Score matrix `QK^T` | `T < 4_096` | CPU f32 loop | < 5μs rayon overhead — not worth parallelizing |
| Score matrix `QK^T` | `4_096 ≤ T < 65_536` | Wide SIMD (8-wide f32, AVX2/NEON) | Auto-vectorizable chunked loop, no allocation |
| Score matrix `QK^T` | `T ≥ 65_536` | Rayon parallel blocked | Per-block 4 KiB tiles, L2-resident |
| Score matrix `QK^T` | `T ≥ 1_048_576` AND GPU available | `gpu_backend` dispatch | Game trace replay scale |
| NNLS β | `t < 64` | Pure Rust loop | Projected GD converges in <10 iters |
| NNLS β | `64 ≤ t < 512` | SIMD-batched matvec | 8 keys per iter, power iteration for `L ≈ ||M||²` |
| NNLS β | `t ≥ 512` | Rayon + blocked Cholesky | Solve normal equations `M^T M w = M^T y` |
| Cholesky Cv | `t < 128` | Pure Rust (LDL^T no-pivot) | Small enough for branch-free unrolled |
| Cholesky Cv | `128 ≤ t` | Blocked Cholesky + SIMD triangular solve | Cache-aware 32×32 blocks |

**ANE path** (constraint 9, macOS only): when `t < 64` AND feature `ane` enabled, the β projection can be compiled to a CoreML matvec + sigmoid — this is the bridge function from AGENTS.md (raw → latent projection, sigmoid bounded).

**Threshold tunability**: thresholds exposed as `SolverRouterConfig { cpu_max_t, simd_max_t, gpu_min_t, ane_max_t }`. Per-load (constraint 7: "when load changes"), the router can demote GPU→SIMD if GPU saturated.

**GOAT gate**: router must pick the same backend at the same load level deterministically — no flapping between adjacent regimes. Hysteresis window of ±10% around each threshold.

### Fusion B: Self-Learning Adaptive CoT Compaction (Constraint 4)

The paper's online compaction (Appendix F.3) is the perfect primitive for **adaptive CoT trace compaction**:

```
While generating thinking trace:
  1. Monitor per-token entropy H_t of the next-token distribution
  2. If trace length > PHYS_BUDGET AND H_t < θ_low:
       - The model is converging — safe to compact
       - Run AM compaction on trace KV (except most recent 32 tokens)
       - Continue generation against compacted trace
  3. If H_t > θ_high:
       - The model is exploring — DO NOT compact, preserve all tokens
  4. Track compaction count; if > MAX_COMPACTS, force early-stop
```

**Self-learning (no LLM training)**: the thresholds `(θ_low, θ_high, PHYS_BUDGET, MAX_COMPACTS)` are tuned online via existing `FreqBandit` (Plan 189) — bandit observes quality metric (e.g., final answer correctness on AIME-style validation) and adjusts thresholds. This is "self-learning adaptive CoT" within constraint 4.

**GOAT gate**:
- G1: compacted-trace generation quality ≥ non-compacted at same effective length
- G2: compaction triggers only when entropy is low (not blindly at budget)
- G3: bandit converges to non-degenerate thresholds

**Integration**: extends existing `ThoughtFold` (Plan 195) with a new `AdaptiveTraceCompactor` trait. The `FoldCache` is reused for KV rollback during compaction.

### Fusion C: AM × VortexFlow Entmax-Regularized OMP

VortexFlow (Plan 196) uses α-entmax routing to produce sparse attention gates. Fuse with OMP:

**Standard OMP** selects key `j*` maximizing `(m - Φw)^T Φ_:,j`. Replace with:

```
j* = argmax_j [ (m - Φw)^T Φ_:,j · entmax_gate_j^α ]
```

where `entmax_gate_j` is VortexFlow's sparsity score for key `j`. Keys with low entmax weight are deprioritized even if their mass correlation is high — they're "sparse-routing-irrelevant".

**Gain**: better key selection when the model has known sparse attention patterns (common in long-context retrieval heads — Wu et al. 2025, Xiao et al. 2025).

**GOAT gate**: entmax-regularized OMP ≥ vanilla OMP at same compaction ratio on retrieval-heavy tasks (RULER).

### Fusion D: AM × SpectralQuant Eigenbasis Pre-Selection

SpectralQuant (Plan 077) projects KV cache to a low-rank eigenbasis. Fuse:

1. Project `(K, V)` to eigenbasis `(K̃, Ṽ)` of rank `r ≪ T`
2. Run HighestAttnKeys selection in eigenbasis space — `O(n · r · d)` instead of `O(n · T · d)`
3. Map selected eigenvectors back to original keys (top-contributing tokens per eigenvector)
4. Run standard AM (β, Cv fitting) on the original-space subset

**Gain**: faster key selection for very long contexts (T > 100k). Quality cost: the eigenvector → token mapping is approximate.

**GOAT gate**: spectral-pre-select + AM ≥ direct AM at same wall-clock on T > 100k contexts.

### Fusion E: Nonuniform Head Budget × Per-Region Allocation (BFCF)

Paper computes per-head budgets via global sensitivity curves. For **game NPC memory** (where BFCF partitions space into regions), extend to per-region-per-head:

```
budget[layer ℓ, head h, region r] = global_budget[ℓ, h] · region_importance[r] · region_attention_density[ℓ, h, r]
```

Region importance comes from BFCF (Plan 218/220). Spatial heads in frequently-visited regions get more budget.

**Gain**: better compaction quality for game NPC long-horizon memory where attention is spatially biased.

**Note**: this is mostly relevant for riir-ai (private) — the per-region allocation logic depends on game domain. The generic per-head budget solver is public.

---

## Verdict: GOAT/Gain Analysis

### Fusion A: Adaptive Solver Routing (CPU/SIMD/GPU/ANE)
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ Pure inference-time routing |
| Gain vs existing | ✅ Strong — no existing size-adaptive AM solver in Rust; performance-critical for game hot path |
| Novel fusion | ✅ AM paper does not address adaptive routing; novel application of AGENTS.md rules |
| Complexity | 🟡 Moderate — three solvers × 4 backends, but each is a well-known algorithm |
| Hot path impact | ✅ Positive — routing PREVENTS bad choices (e.g., rayon overhead on tiny matrices) |
| Repo proof | ✅ Strong — paper demonstrates seconds-scale compaction; we mirror algorithms exactly |

**Verdict: GAIN** — Implement as part of Plan 265. Promote CPU/SIMD paths to default; gate GPU/ANE behind features.

### Fusion B: Adaptive CoT Compaction
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ Inference-time only |
| Gain vs existing | ✅ ThoughtFold is selection-based 78% reduction; AM with closed-form β preserves more |
| Novel fusion | ✅ Online compaction applied to CoT traces is novel |
| Complexity | 🟡 Moderate — reuses ThoughtFold infra, adds bandit-tuned thresholds |
| Hot path impact | 🟡 Acceptable — compaction deferred to entropy trough |
| Repo proof | 🟡 Paper shows AIME preservation across 6 compactions; CoT-specific untested |

**Verdict: GAIN** — Implement behind `adaptive_cot_compaction` feature. GOAT gate required before promotion.

### Fusion C: Entmax-Regularized OMP
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ |
| Gain vs existing | 🟡 Unknown — depends on VortexFlow gate quality |
| Novel fusion | ✅ |
| Complexity | 🟡 Low — single-line modification to OMP selection score |
| Hot path impact | ✅ Negligible |
| Repo proof | 🔴 Untested — pure theoretical fusion |

**Verdict: GATE** — Implement as opt-in variant behind `entmax_omp` feature. GOAT gate required.

### Fusion D: Spectral Pre-Selection
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ |
| Gain vs existing | 🟡 Speed gain on T > 100k; quality cost unclear |
| Novel fusion | ✅ |
| Complexity | 🔴 Higher — eigenbasis → token mapping is non-trivial |
| Hot path impact | ✅ Positive for very long contexts |
| Repo proof | 🔴 Untested |

**Verdict: DEFER** — Implement only if AM direct selection is too slow on production-scale contexts.

### Fusion E: Per-Region Head Budget
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ |
| Gain vs existing | ✅ For game NPC memory specifically |
| Novel fusion | ✅ |
| Complexity | 🟡 Moderate |
| Hot path impact | ✅ |
| Repo proof | 🔴 Untested |

**Verdict: DEFER to riir-ai** — Game-domain-specific. Public katgpt-rs exposes only the per-head budget solver; riir-ai adds per-region allocation (Research 121).

---

## GOAT Gate Matrix (Core AM Module)

| Gate | Criterion | Measurement |
|------|-----------|-------------|
| G1 | β recovery within paper tolerances | `||β_NNLS − β_ref||_∞ < 0.1` on synthetic test |
| G2 | Cv reconstruction | `||X Cv − Y||_F / ||Y||_F < 0.05` on synthetic test |
| G3 | OMP mass coverage | After t OMP iters, residual < 5% of initial mass |
| G4 | HighestAttnKeys RMS | Top-t keys cover > 80% of RMS attention mass |
| G5 | Reconstruction perplexity | Compacted perplexity within 5% of original on test contexts |
| G6 | Routing determinism | Same load level → same backend (no flapping within ±10% hysteresis) |
| G7 | Memory bound | No allocation inside NNLS / Cholesky hot loops (scratch buffers reused) |
| G8 | SIMD speedup | 8-wide SIMD ≥ 4× over scalar on score-matrix kernel |

**Promotion rule**: pass G1–G7 → promote `attention_matching` to default. Pass G8 → promote `am_simd` to default.

---

## Commercial Strategy Alignment (per Research 003)

| Item | Public (katgpt-rs) | Private (riir-ai) |
|------|---------------------|-------------------|
| AM algorithm (OMP, HighestAttn, NNLS β, LS Cv) | ✅ Generic framework — adoption funnel | — |
| Nonuniform head budget solver (greedy swap) | ✅ Generic algorithm | — |
| Chunked + online compaction | ✅ Generic primitives | — |
| Adaptive CPU/SIMD/GPU/ANE router | ✅ Generic engine mechanic | — |
| `BetaBias`, `CompactKVCache` types | ✅ Already public (StillKV) | — |
| Per-region head budget allocation (BFCF × AM) | — | ✅ Game-domain-specific |
| AM × ThoughtFold adaptive CoT recipe | — | ✅ Fusion know-how |
| AM × chain SyncBlock boundary swap | — | ✅ Chain internals |
| Trained LoRA β predictor (replaces NNLS) | — | ✅ Trained weight asset |
| Head budget schedules for trained game LoRA models | — | ✅ Data asset |
| GOAT benchmark numbers on game contexts | — | ✅ Implementation-level detail |

**Rule of thumb**: WHAT = public (the AM algorithm exists, here's the math). HOW = private (we use AM with this specific fusion recipe for NPC memory).

---

## Related Work in Our Stack

| Existing Module | Relationship |
|---|---|
| `still_kv/` (Plan 245, Research 213) | **Sibling** — synthesis-based compaction (Perceiver). AM is selection-based with optimal β. StillKV's `BetaBias` and `CompactKVCache` types are **reused** by AM. |
| `mux_latent/` (Plan 238) | **Composable** — MUX-Latent provides vocabulary superposition encoder; AM can compact MUX-encoded latents |
| `spectralquant/` (Plan 077) | **Composable** — eigenbasis projection for Fusion D |
| `octopus/` (Plan 099) | **Alternative** — octahedral KV cache compression; AM is a different family |
| `kvarn/` (Plan 179) | **Composable** — variance-normalized KV cache; AM selection can use variance as score |
| `sp_kv/` (Plan 070) | **Predecessor** — self-pruned attention; AM generalizes with optimal β |
| `turboquant/` (Plan 043) | **Orthogonal** — quantization; AM operates on selection, can stack on quantized cache |
| `shard_kv/` (Plan 147) | **Composable** — sharded asymmetric KV cache; AM can compact per-shard |
| `rt_turbo/` (Plan 126) | **Composable** — retrieval-head sparse decode; AM selects keys, RT-Turbo routes decode |
| `fold/` (Plan 195 ThoughtFold) | **Extends** — Fusion B builds on ThoughtFold's `ChainFolder` trait |
| `breakeven/` (Plan 250) | **Router** — Breakeven Complexity Router can decide AM vs MUX-Latent vs summarization |
| `sense_composition/` | **Consumer** — NPC sense data can be AM-compacted for long-horizon memory |

---

## Algorithmic Implementation Notes

### Numerical Stability

Following the paper (Appendix A.1, C.2):
- All scores computed with **max-shift**: `exp(s - max(s))` for stability
- Compute in **f32**, cast to **bf16/f16** for storage
- NNLS uses **projected gradient descent** with `η = 1/L`, `L ≈ ||M||²` via power iteration
- Cholesky with diagonal jitter `λI` (λ=1e-6) for fallback when rank-deficient

### β Stability Constraints

From Appendix C.2:
- For HighestAttnKeys: bounded NNLS with `w_j ∈ [e^{-3}, e^{3}]` (β ∈ [-3, 3])
- For OMP: discard keys with `β < -7` after selection; cap `w_j ≤ e^7`
- These prevent degenerate solutions where one key absorbs all mass

### OMP Speedups (Algorithm 2)

- Select `k` keys per greedy iteration (default k=4)
- Refit NNLS every `τ` iterations (default τ=2)
- 4–8× speedup with little degradation

### Cholesky vs pinv vs lstsq

Paper Appendix C.2 ranks quality: `lstsq > cholesky > pinv`. We choose:
- **Default**: Cholesky with jitter fallback (fastest in Rust, no LAPACK dep)
- **Optional**: QR-based lstsq if `ndarray-linalg` or `nalgebra` feature enabled

---

## Key Reference Equations

### Mass-Preserving Attention Matching

For compacted `(Ck, β, Cv)` with reference queries `Qref = [q_1; ...; q_n]`:

```
Eq. 1 (Attention output):
  softmax(qK^T/√d) V  ≈  softmax((qCk^T + β)/√d) Cv

Eq. 2 (Attention mass):
  Σ_j exp(qK_j^T/√d)  ≈  Σ_j exp((qCk_j^T + β_j)/√d)
```

### β Fitting (NNLS)

```
A ∈ ℝ^{n×t},  A_ij = exp(q_i (Ck)_j^T / √d)
m ∈ ℝ^n,      m_i   = Σ_k exp(q_i K_k^T / √d)
w* = argmin_{w ≥ 0} ||A w − m||²       // projected gradient descent
β  = log(w*)
```

### Cv Fitting (Ordinary Least Squares)

```
Y_i = softmax(q_i K^T / √d) V                          ∈ ℝ^d
X_i = softmax((q_i Ck^T + β) / √d)                     ∈ ℝ^t
Cv* = argmin_Cv ||X Cv − Y||²_F  =  (X^T X)^{-1} X^T Y
```

### Highest Attention Keys (RMS aggregation)

```
a_i = softmax(q_i K^T / √d)         ∈ ℝ^T            (per-query attention)
s_j = sqrt( (1/n) Σ_i a_{i,j}² )    ∈ ℝ^T            (RMS score per key)
S   = argsort(s, descending)[:t]                       (top-t indices)
Ck  = K[S, :]
```

### OMP Key Selection (Algorithm 1)

```
Φ_ij = exp(q_i K_j^T / √d)                            // mass feature matrix
m_i  = Σ_j Φ_ij                                        // target mass
r    ← m,  S ← ∅
for k = 1 to t:
    j* = argmax_{j ∉ S} (r^T Φ_:,j)
    S ← S ∪ {j*}
    w  = argmin_{w ≥ 0} ||Φ_:,S w − m||²              // NNLS refit
    r  ← m − Φ_:,S w
return S, w  (with β = log w)
```

### Nonuniform Head Budget (Algorithm 4)

```
Given per-head sensitivity curves J_h(ρ) and reference ratio r_0:
Initialize: p_h ← 1/H  for all heads
Repeat:
    ρ_h ← p_h · H · r_0                                // map share → ratio
    g+_h ← J_h(ρ_h) − J_h(ρ_h + η')                   // marginal gain of adding
    g−_h ← J_h(ρ_h − η') − J_h(ρ_h)                   // marginal loss of removing
    b* ← argmax_h g+_h,  a* ← argmin_h g−_h  (a* ≠ b*)
    if g+_b* > g−_a*:  p_a* −= η, p_b* += η
    else: break
return {p_h}
```

---

## Audit Trail

- **2026-06-14**: Initial research from arxiv 2602.16284. Verdict GAIN for Fusion A (router) and Fusion B (adaptive CoT). Fusion C/D/E deferred or gated.

---

## TL;DR

**Paper**: Fast KV Compaction via Attention Matching (MIT, ICML 2026). Replaces `(K, V)` with `(Ck, β, Cv)` that match both attention output AND mass. 50× compaction in seconds, 2 orders of magnitude faster than Cartridges, often better quality.

**Distillation**: Pure modelless port to `katgpt-rs/src/attn_match/`. Reuses StillKV's `BetaBias` + `CompactKVCache` types but **replaces heuristic β with NNLS-optimal β** — direct upgrade path.

**Novel fusions**:
- **A (GAIN)**: Adaptive CPU/SIMD/GPU/ANE solver routing by tensor size — no existing size-adaptive AM solver in Rust.
- **B (GAIN, gated)**: Adaptive CoT trace compaction via online AM + entropy-thresholded bandit.
- **C (GATE)**: Entmax-regularized OMP via VortexFlow gates.
- **D (DEFER)**: SpectralQuant pre-selection for T > 100k.
- **E (riir-ai)**: Per-region head budgets via BFCF.

**Verdict**: GAIN — promote core AM + CPU/SIMD router to default after GOAT gate. Plan 265.
