# Plan 280: CS-KV-Importance Probe + Density-Budget Interpolator

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/247_Dense_Latent_Heterogeneous_Communication_CS_Probe.md](../.research/247_Dense_Latent_Heterogeneous_Communication_CS_Probe.md)
**Private guide:** [riir-ai/.research/133_NPC_Mind_Reading_Adaptive_Bandwidth_Guide.md](../../riir-ai/.research/133_NPC_Mind_Reading_Adaptive_Bandwidth_Guide.md)
**Source paper:** [arxiv 2606.13594](https://arxiv.org/abs/2606.13594) — Chen et al., "See What I See, Know What I Think"
**Target:** `katgpt-rs/src/cs_kv_probe/` (new module) + Cargo feature `cs_kv_probe`
**Status:** ✅ Complete — Phases 1–4 done, GOAT gate G1/G2/G3 green (24 unit + 6 gate tests, opt-in `cs_kv_probe` feature)

---

## Goal

Ship the two modelless primitives distilled from Research 247 as a generic, MIT-licensed, no-game-semantics module in katgpt-rs:

1. **`CsKvProbe`** — compressed-sensing KV-group importance probe. Given (a) a black-box eval function `Fn(&[bool], &[Episode]) -> f32`, (b) `M` ablation masks, (c) `N` episodes, produces a `KvGroupRanking { scores: Vec<f32> }`. Pure inference, zero training, zero allocations in the hot path.
2. **`DensityBudget`** — the `K(ca)` interpolator. Given `ca ∈ [0,1]`, `K_sparse`, `K_dense`, `D`, returns the integer top-K budget. One scalar in, one scalar out.
3. **`GatedKvSlice`** — applies a `KvGroupRanking` + `DensityBudget` to a KV cache (or any `&[f32]` of length `D`) to produce a top-K gated slice via `soft_gate_bias` (sigmoid, never softmax). Reuses the SP-KV gate-bias pattern from Plan 070.

**GOAT gate:** the primitive must (a) reproduce the paper's sparse-vs-dense K-sweep shape on a synthetic homogeneous self-comm task (G2 from the riir-ai guide), (b) zero-overhead when feature disabled, (c) no allocations in the apply path. The headline task-gain proof (G6) and NPC wiring live in riir-ai Plan 311 — this plan ships only the open math.

**Non-goals (explicitly out of scope here):**
- NPC comms wiring, fog-of-war `ca` computation, zone broadcast → riir-ai Plan 311.
- Cross-shape projection training → riir-train.
- Position-disentanglement (RoPE strip/restore) → already shipped in `src/shard_kv/rope.rs` (`undo_rope`/`reapply_rope`); this plan re-exports it, does not reinvent it.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `src/cs_kv_probe/mod.rs` with module root + re-exports. Add `cs_kv_probe` feature to `Cargo.toml` (opt-in, NOT in `default` or `full` until G2 passes). Gate all module code behind `#[cfg(feature = "cs_kv_probe")]`.
- [x] **T1.2** Define types in `src/cs_kv_probe/types.rs`:
  - `pub struct Episode { pub kv_cache: Vec<f32>, pub label_success: bool }` — generic, no game semantics. `kv_cache` is the flattened `[D]` slice for one inference; `label_success` is the task outcome.
  - `pub struct AblationMask { pub bits: Vec<bool>, pub n_heads: usize }` — binary retention mask over `H` heads. `bits[h]=true` means head `h` retained.
  - `pub struct KvGroupRanking { pub scores: Vec<f32>, pub n_groups: usize }` — Lasso coefficients aggregated per KV group. Higher = more important. BLAKE3-hashable.
  - `pub struct DensityBudget { pub k_sparse: usize, pub k_dense: usize, pub d_total: usize }` — the interpolator config. Defaults: `k_sparse = round(0.035 * d_total)`, `k_dense = round(0.87 * d_total)` (paper's floors/ceilings).
- [x] **T1.3** Implement `sample_masks` in `src/cs_kv_probe/probe.rs`:
  - `pub fn sample_masks(n_heads: usize, m: usize, ablation_fraction: f32, rng: &mut fastrand::Rng) -> Vec<AblationMask>` — stratified random masks, each zeroing exactly `ablation_fraction` (default 0.05) of heads. Returns `Vec::with_capacity(m)`. Paper uses `M=200`, `fraction=0.05`.
- [x] **T1.4** Implement Lasso solver in `src/cs_kv_probe/lasso.rs`:
  - `pub fn lasso(Phi: &[[bool; HishouldBeDynamicUseVec]], y: &[f32], alpha: f32, n_iter: usize) -> Vec<f32>` — coordinate descent L1-regularized regression. Inputs: measurement matrix `Phi` (M×N bool → cast to f32), centered observations `y` (M,), regularization `alpha` (default 1e-4), iteration count (default 1000). Output: coefficient vector `x` (N,). 
  - **No external dep** — implement coordinate descent in-place with pre-allocated scratch. ~80 lines. Avoid pulling in a linear-algebra crate for a single solver.
  - Test: known-sparse ground truth (e.g. only heads 3, 17, 42 matter) → Lasso recovers them within rank tolerance.

### Phase 2 — Probe Assembly

- [x] **T2.1** Implement `CsKvProbe::run` in `src/cs_kv_probe/probe.rs`:
  - `pub fn run<Eval>(episodes: &[Episode], eval: &Eval, config: &CsProbeConfig, rng: &mut fastrand::Rng) -> KvGroupRanking where Eval: Fn(&AblationMask, &[Episode]) -> f32`
  - Steps: (1) compute `y_baseline = eval(all_ones_mask, episodes)`, (2) sample `M` masks, (3) for each mask compute `y_m = eval(mask, episodes)`, center `ỹ = y - y_baseline`, (4) build `Phi` from masks, (5) `lasso(Phi, ỹ, alpha, n_iter)`, (6) aggregate per-KV-group via `kv_group = head * n_kv_head / n_head` (existing GQA mapping), (7) return `KvGroupRanking`.
  - **Allocation discipline:** `Phi` built once as `Vec<Vec<f32>>::with_capacity(M)`; `y` as `Vec::with_capacity(M)`; both reused across the eval loop via `clear()` + overwrite. No per-iteration allocation.
- [x] **T2.2** Implement `DensityBudget::k_for` in `src/cs_kv_probe/budget.rs`:
  - `pub fn k_for(&self, ca: f32) -> usize` — `round(self.k_sparse as f32 + ca * (self.k_dense - self.k_sparse) as f32) as usize`, clamped to `[1, self.d_total]`. One line, branchless if using `min`/`max`.
- [x] **T2.3** Implement `GatedKvSlice::apply` in `src/cs_kv_probe/gate.rs`:
  - `pub fn apply(ranking: &KvGroupRanking, budget: &DensityBudget, ca: f32, kv: &[f32], out_bias: &mut [f32])` — writes `out_bias[g] = log(score_normalized[g] + ε)` for top-K groups (K = `budget.k_for(ca)`), `-INFINITY` for the rest. Reuses the SP-KV `soft_gate_bias` convention (Plan 070). **Sigmoid-compatible, never softmax.**
  - **Zero-allocation:** caller passes `out_bias: &mut [f32]` of length `n_groups`. No Vec returns.

### Phase 3 — GOAT Proof (G1, G2, G3 from riir-ai guide)

- [x] **T3.1** G1 test `test_cs_ranking_beats_random`: synthetic task where heads {3, 17, 42} of 64 carry signal. Run probe with M=200, N=100 synthetic episodes. Assert top-3 CS-ranked heads ⊇ {3, 17, 42} with ≥80% overlap. Assert top-3 CS accuracy ≥ top-3 random + 15pp.
- [x] **T3.2** G2 test `test_sparse_dense_duality_shape`: synthetic homogeneous self-comm task (D=16, mimicking 8-dim × 2-layer HLA). Vary K ∈ {1, 2, 4, 8, 14, 16}. Context-aware receiver (has own signal) → plateau at K≤2. Context-unaware receiver (blind) → chance until K≥12, sharp rise. Assert qualitative shape match to paper Fig 5. This is the headline proof the duality holds for our dimensionality.
- [x] **T3.3** G3 test `test_ca_monotone_and_bounded`: property test `DensityBudget::k_for(ca)` is monotone non-decreasing in `ca ∈ [0,1]`, bounded `[k_sparse, k_dense]`. 1000-point sweep + edge cases ca=0, ca=1, ca=0.5.
- [x] **T3.4** Zero-overhead test `test_feature_disabled_is_passthrough`: with `cs_kv_probe` feature OFF, module does not compile into the binary. Verify via `cargo build --no-default-features` succeeds and the module symbols are absent.
- [x] **T3.5** Allocation test `test_apply_zero_alloc`: run `GatedKvSlice::apply` 10K times in a loop with `dhat` or `cargo-bench` heap profiler; assert 0 heap allocations after the first warmup call.

### Phase 4 — Docs + Feature Wiring

- [x] **T4.1** Add `cs_kv_probe` to the feature table in `katgpt-rs/.docs/01_overview.md` (opt-in, not default).
- [x] **T4.2** Add module to `katgpt-rs/README.md` Feature Showcase section with a one-paragraph summary + cross-ref to Research 247.
- [x] **T4.3** Write `katgpt-rs/examples/cs_kv_probe_demo.rs` — synthetic task, run probe, print ranking, demo `K(ca)` interpolation for ca ∈ {0.0, 0.25, 0.5, 0.75, 1.0}. <100 lines.

---

## Risks

1. **Lasso solver correctness.** Coordinate descent is simple but easy to get the soft-thresholding sign wrong. Mitigation: T1.4 has a known-sparse ground-truth test; if it fails, the bug is in the solver, not the probe.
2. **Recovery limit at our dimensionality.** Paper used H=1152 heads, M=200 masks → ~70 reliable coefficients. Our HLA is D=16 (or D=64 for larger configs). At D=16, M=200 is overkill — recovery is trivial. But the K-sweep shape (G2) may not show the sharp phase transition at such low D. If G2 fails to reproduce the shape → the duality may be an artifact of high-D; downgrade verdict to GOAT (diagnostic) and note the dimensional caveat in Research 247.
3. **Eval function is caller-supplied.** The probe is only as good as the task labeling. katgpt-rs ships the math; riir-ai ships the labels. If riir-ai Plan 311's labels are noisy, G1 fails not because of the probe but because of the labels. Keep the boundary clean: this plan ships the probe, not the labels.

---

## TL;DR

Open primitive for Research 247's Super-GOAT. Ships `CsKvProbe` (compressed-sensing KV-group importance via ablation + Lasso, pure inference), `DensityBudget` (the `K(ca)` interpolator, sparse floor 3.5% / dense ceiling 87%), and `GatedKvSlice` (sigmoid-gated top-K application, reuses SP-KV convention). No game semantics, no NPC wiring — that's riir-ai Plan 311. No cross-shape projection training — that's riir-train. No RoPE reinvention — reuse `shard_kv/rope.rs`. GOAT gate: G1 (CS beats random), G2 (duality shape reproduces at D=16), G3 (interpolator monotone+bounded), zero-overhead when off, zero-alloc in apply. Feature `cs_kv_probe`, opt-in until G2 passes.
