# Plan 305: Algorithmic-Probability Sampler + Coincidence Gate (Open Primitive)

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/284_Simplicity_Bias_Sampler_Coincidence_Extrema.md](../.research/284_Simplicity_Bias_Sampler_Coincidence_Extrema.md)
**Private guide:** [riir-ai/.research/150_Algorithmic_Probability_Sampler_NPC_Guide.md](../../riir-ai/.research/150_Algorithmic_Probability_Sampler_NPC_Guide.md) — **PRIVATE, do not export**
**Source paper:** [Dingle & Hutter, *Entropy* 28(2):226, 2026](https://www.mdpi.com/1099-4300/28/2/226) — Simplicity and Complexity in Combinatorial Optimization
**Target:** `katgpt-rs/src/screening/complexity_prior.rs` + `katgpt-rs/src/screening/coincidence_gate.rs`
**Feature:** `complexity_prior_sampler` (off by default until GOAT gate passes)
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Implement two open primitives distilled from Dingle–Hutter 2026 (Research 284):

1. **`CompressionPriorSampler<K: ComplexityProxy>`** — replaces uniform candidate sampling in MCTS / bandits / DDTree / speculative drafters with `sigmoid(-α·K̃(x) - β)`-weighted sampling. Pluggable `K̃`: RLE ratio (R188), Shannon entropy (R188), `‖θ‖_1` (R125), lz4 length (R256). **Safety guarantee:** never worse than uniform sampling; exponentially faster when the optimum is low-K.

2. **`CoincidenceGate`** — given a found optimum `x*` for one simple objective `f1`, probe `x*` against all other simple objectives `f2_k`. Theorem-backed hit rate: `r / |X_O(1)|` per probe vs `r / |X|` from random candidates (exponential lift).

**GOAT gate (G1 + G2):** sampler is never-worse-than-uniform on 5 game types (G1), and exponential speedup on a synthetic low-K optimum (G2). Pass → promote to default. Fail → keep opt-in, create issue.

**Latent reframing:** the public primitive operates on `&[u8]` and `&[f32]` (byte-quantized latents); the riir-ai side wires it to HLA / functor / shard vectors (private, Plan 331 TBD).

---

## Phase 1 — Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/src/screening/complexity_prior.rs` with:
  - `pub trait ComplexityProxy { fn k_tilde<T: AsRef<[u8]>>(&self, candidate: T) -> f32; }`
  - `pub struct RleComplexity;` — re-export `rle_compress` from `ruliology/irreducibility.rs`, compute `compressed_len / raw_len`
  - `pub struct EntropyComplexity;` — re-export Shannon entropy kernel from `ruliology/irreducibility.rs` (already SIMD-friendly)
  - `pub struct L1Complexity;` — sum of `|x|` over the byte slice (R125 sandwich bound proxy for fixed-precision latents)
  - `pub struct Lz4Complexity;` — lazily-initialized lz4 encoder (Warm tier; behind sub-feature `lz4_proxy` to keep the default zero-dep)
  - All proxies `#[inline]`, zero-allocation, `const fn new()` where possible

- [ ] **T1.2** Implement `CompressionPriorSampler<K: ComplexityProxy>`:
  - Fields: `proxy: K`, `alpha: f32`, `beta: f32`
  - `pub fn log_prob<T: AsRef<[u8]>>(&self, candidate: T) -> f32` — returns `-α·K̃(x) - β` (log-sigmoid input; **never softmax**)
  - `pub fn sample_ix(&self, candidates: &[&[u8]], scratch: &mut [f32], rng: &mut impl Rng) -> usize` — fills `scratch` with log-probs, computes softmax-free categorical sample via cumulative-sum + binary search (sigmoid per candidate, then normalize for sampling only — never as the public API)
  - `pub fn top_k(&self, candidates: &[&[u8]], k: usize, out: &mut [usize])` — partial sort by log-prob, in-place
  - `pub const fn default() -> Self` — `alpha = 1.0, beta = 0.0`
  - All methods `#[inline]`, zero heap allocation in the hot path

- [ ] **T1.3** Implement latent variant `LatentCompressionPriorSampler<K>`:
  - Operates on `&[f32]` via byte-quantization (`fn quantize_latent(v: &[f32], scratch: &mut [u8])` — min-max scale to `[0, 255]`)
  - Reuses the same `ComplexityProxy` trait (over `&[u8]`)
  - Same `log_prob` / `sample_ix` / `top_k` API
  - Quantization is zero-allocation: caller provides scratch buffer

- [ ] **T1.4** Create `katgpt-rs/src/screening/coincidence_gate.rs` with:
  - `pub struct CoincidenceGate { simple_set_size_estimate: f32 }` — threshold τ on `|X_O(1)|`; above τ → optimistic transfer probe, below τ → skip
  - `pub fn probe_transfer<F, I>(&self, x_star: &[u8], objectives: I, rank_threshold_r: usize) -> Vec<usize>` where `F: Fn(&[u8]) -> f32`, `I: IntoIterator<Item = F>` — returns indices of `f2_k` where `x*` ranks in top-r
  - `pub fn should_probe(&self, k_tilde_of_f2: f32) -> bool` — skip if `f2` is complex (high-K reward function)
  - Zero allocation in the hot path; returns `SmallVec<[usize; 8]>` or pre-allocated `&mut [usize]` slice

- [ ] **T1.5** Feature-gate both modules behind `complexity_prior_sampler` in `katgpt-rs/Cargo.toml`:
  - Default: off (zero-dep baseline preserved)
  - Sub-feature `lz4_proxy` adds `lz4` dep (for `Lz4Complexity`)
  - Sub-feature `blake3_proxy` adds `blake3` dep (for `Blake3CanonicalLengthComplexity`, used by `riir-neuron-db`)

- [ ] **T1.6** Re-export from `katgpt-rs/src/screening/mod.rs` and `katgpt-rs/src/lib.rs`:
  - `pub use complexity_prior::{ComplexityProxy, CompressionPriorSampler, LatentCompressionPriorSampler, RleComplexity, EntropyComplexity, L1Complexity};`
  - `pub use coincidence_gate::CoincidenceGate;`
  - Gated by `#[cfg(feature = "complexity_prior_sampler")]`

- [ ] **T1.7** Unit tests (in-module `#[cfg(test)] mod tests`):
  - `test_rle_complexity_all_same` — `[42u8; 100]` → K̃ near 0
  - `test_rle_complexity_random` — pseudo-random bytes → K̃ near 1
  - `test_entropy_complexity_uniform` — uniform byte distribution → max entropy
  - `test_entropy_complexity_degenerate` — all-same → zero entropy
  - `test_l1_complexity` — `[1.0, -2.0, 3.0]` bytes → sum of abs = 6
  - `test_sampler_log_prob_monotone` — lower K̃ → higher log_prob
  - `test_sampler_sample_ix_distribution` — over 10000 samples, empirical distribution correlates > 0.9 with theoretical
  - `test_sampler_top_k_correct` — top-K indices match argsort
  - `test_sampler_never_worse_than_uniform` — on a synthetic uniform-reward candidate set, sampler's expected rank ≤ uniform's expected rank ± 5% (safety)
  - `test_coincidence_gate_probe_transfer` — given `x_star` and 3 objectives where `x_star` is top-1 in 2 of them, returns the 2 indices
  - `test_coincidence_gate_should_probe_skips_complex_f2` — high-K `f2` → skip

- [ ] **T1.8** Add `examples/algorithmic_probability_sampler_demo.rs`:
  - Build a 16-bit action space (65536 candidates)
  - Define a simple objective `f1(x) = -popcount(x XOR 0xFFFF)` (optimum = `0xFFFF`, K = O(1))
  - Show: uniform sampling needs ~32768 samples avg to find optimum; K-prior sampler (RLE) needs ~10
  - Print speedup ratio

---

## Phase 2 — GOAT Gate (G1 + G2)

### Tasks

- [ ] **T2.1** **G1 — Sampler safety benchmark.** Create `katgpt-rs/.benchmarks/305_complexity_prior_sampler_goat.md`:
  - Replace uniform child expansion in `mcts.rs` with `CompressionPriorSampler<RleComplexity>` (behind a sub-feature `mcts_k_prior` for isolation)
  - Run 1000 rollouts × 5 game types (Go 9×9, FFTactics, Bomber, Civ-sim, Bomberman-arena — reuse existing test harnesses)
  - Record: win/draw rate vs uniform baseline, p99 rollout time, K̃ distribution of sampled candidates
  - **Pass criterion:** win/draw ≥ 50% on every game type (never significantly worse)
  - **Stretch:** ≥ 5% win-rate improvement on ≥ 3 of 5 game types
  - Document honest result (positive or negative)

- [ ] **T2.2** **G2 — Exponential speedup benchmark.**
  - Synthetic game with provably low-K optimum: "always pick the lexicographically smallest action" (K = O(1))
  - 16-bit action space, measure time-to-optimum (samples until first hit)
  - Compare: uniform sampler vs `CompressionPriorSampler<RleComplexity>` vs `EntropyComplexity` vs `L1Complexity`
  - **Pass criterion:** K-prior sampler reaches optimum in ≤ `2^K(x*)` samples; uniform needs `≈ |X|`. Speedup ≥ 100×.
  - **Stretch:** ≥ 1000× speedup.

- [ ] **T2.3** Document results in `.benchmarks/305_*.md` with honest verdict:
  - If G1 + G2 pass → recommend promotion to default
  - If G1 fails → mis-scaled `K̃`, document fix needed (tune `α`, swap proxy), keep opt-in
  - If G2 fails → the `K̃` proxy doesn't track true `K(x)` for this domain, document alternative proxies, keep opt-in

- [ ] **T2.4** If G1 + G2 pass → flip `complexity_prior_sampler` to default in `katgpt-rs/Cargo.toml`. Update README "GOAT-Proved Additions" section. Run `./scripts/ci_feature_guard.sh` to confirm no combo regression.

---

## Phase 3 — Integration Hooks (post-promotion)

### Tasks

- [ ] **T3.1** Add adapter trait impl for `katgpt-rs/src/mcts.rs`:
  - `MctsExpansionPrior` trait with default impl `UniformExpansion`
  - New impl `KPriorExpansion<K: ComplexityProxy>` gated by `mcts_k_prior` sub-feature
  - Zero-cost when feature is off (existing `UniformExpansion` unchanged)

- [ ] **T3.2** Add integration for `katgpt-rs/src/bandit.rs`:
  - New bandit variant `KPriorBandit<K>` that biases arm selection by `sigmoid(-α·K̃(arm) - β)`
  - Gated by `bandit_k_prior` sub-feature

- [ ] **T3.3** Add speculative drafter hook in `katgpt-rs/src/speculative/`:
  - `KPriorDrafter<K>` wraps an existing drafter, re-ranking drafts by K-prior
  - Composes cleanly with `CompressionDrafter` (R256) and `DendriticGate` (R260)
  - Gated by `spec_k_prior` sub-feature

- [ ] **T3.4** Documentation: README section "🧠 Algorithmic-Probability Sampler: Safe Prior for Inference-Time Search (Plan 305, Research 284)" under Feature Showcase. Honest framing: "Levin-Search variant applied to modelless inference; never worse than uniform, exponentially better on simple optima; theorem-backed cross-task transfer via CoincidenceGate."

---

## Phase 4 — riir-ai Hand-off (reference, executed in riir-ai Plan 331)

This phase is *referenced* here for traceability but executed in `riir-ai/.plans/331_*.md` (TBD).

- [ ] **T4.1 (riir-ai)** Wire `LatentCompressionPriorSampler` into `riir-engine/src/hla/` for per-NPC `(α_i, β_i)` K-prior on candidate affect vectors. (Private, gated by riir-ai feature `hla_k_prior`.)
- [ ] **T4.2 (riir-ai)** Wire `dirichlet_energy` K-prior into `riir-engine/src/latent_functor/` for functor `C` matrix sampling. (Private.)
- [ ] **T4.3 (riir-ai)** Unify curiosity pulse with K-prior deviation: `curiosity = KL(p_sampled || p_K_prior)` in `riir-engine/src/cgsp_runtime/`. (Private.)
- [ ] **T4.4 (riir-ai)** Online `(α, β)` calibration via curiosity signal. (Private — the moat.)
- [ ] **T4.5 (riir-ai)** `CoincidenceGate` wiring to KG triple emission in `riir-engine/src/kg_*.rs` + `riir-games/src/social/`. Free zone-transfer of KG patterns. (Private.)
- [ ] **T4.6 (riir-ai)** G3 + G4 + G5 GOAT gate on the runtime side. Promote riir-ai features if pass.

---

## Phase 5 — Chain + Shard Bridges (reference, executed in respective repos)

- [ ] **T5.1 (riir-chain)** Add `latcal_fixed::to_fixed(α)` and `to_fixed(β)` commitment of per-NPC K-prior scalars in `riir-chain/src/encoding/latcal_fixed.rs`. Update `MerkleFrozenEnvelope` schema. (Private, gated by `k_prior_commitment`.)
- [ ] **T5.2 (riir-neuron-db)** Extend `NeuronShard` with K-prior signature field `(α, β)` alongside `style_weights[64]`. Audit ALL constructors (`new`, `new_unchecked`, `new_spectral`, `from_bytes`) — per the `merkle_root` lesson. (Private, gated by `k_prior_signature`.)
- [ ] **T5.3 (riir-chain + riir-neuron-db)** CI guard: `cargo check --all-features` across both repos to catch combo-only regressions on the new shard field.

---

## GOAT Gate Summary

| Gate | Metric | Pass | Stretch | Repo |
|------|--------|------|---------|------|
| **G1** Sampler safety | Win/draw vs uniform on 5 games | ≥ 50% each | ≥ 5% improvement on ≥ 3/5 | katgpt-rs |
| **G2** Exponential speedup | Time-to-optimum on low-K synthetic | ≤ `2^K(x*)` samples; ≥ 100× speedup | ≥ 1000× | katgpt-rs |
| **G3** Coincidence transfer | Hit rate vs random baseline | ≥ 10× | ≥ 100× | riir-ai |
| **G4** Latent ranking | Spearman correlation between `K̃` proxies | ≥ 0.9 | ≥ 0.95 | riir-ai |
| **G5** Tick latency | p99 tick time at 1000-NPC scale | ≤ 50ms (20Hz) | ≤ 25ms | riir-ai |

G1 + G2 pass → promote `complexity_prior_sampler` to default in katgpt-rs.
G3 + G4 + G5 pass → promote riir-ai HLA/functor/cgsp wiring to default.

---

## Constraints Check

- [x] Modelless first — inference-time only, no LLM training, no backprop through weights
- [x] Lands in katgpt-rs domain (Phase 1–3) — generic math primitives, no game/chain/shard IP
- [x] SOLID, DRY — `ComplexityProxy` trait decouples the prior from the proxy; reuses R188/R256/R125 shipped kernels
- [x] CPU/GPU auto-route — RLE/Entropy/L1 are Plasma-tier (CPU SIMD); Lz4 is Warm-tier
- [x] Plasma/hot/warm/cold — proxies map to tiers (RLE=Plasma, Lz4=Warm, BLAKE3=Cold)
- [x] Threshold-based — sigmoid thresholds for sample selection (not softmax)
- [x] Feature-gated — behind `complexity_prior_sampler`, off by default until GOAT
- [x] Zero-allocation hot paths — scratch buffers passed by caller, all proxies `#[inline]`
- [x] Sigmoid not softmax — `p(x) = sigmoid(-α·K̃(x) - β)` per project rule
- [x] Latent-to-latent primary — `LatentCompressionPriorSampler` operates on `&[f32]` via byte-quantization

---

## TL;DR

Two open primitives from Dingle–Hutter 2026 (Research 284): `CompressionPriorSampler<K>` (universal algorithmic-probability sampler with pluggable `K̃` — RLE, entropy, L1, lz4) and `CoincidenceGate` (theorem-backed cross-task transfer). Feature `complexity_prior_sampler`, off by default. Phase 1: skeleton + tests + demo (8 tasks). Phase 2: GOAT gate G1 (sampler safety on 5 games) + G2 (exponential speedup on low-K synthetic). Pass → promote to default. Phase 3: MCTS / bandit / speculative integration hooks. Phase 4 (riir-ai): HLA / functor / cgsp / KG-triple wiring. Phase 5 (riir-chain + riir-neuron-db): LatCal commitment + NeuronShard K-prior signature storage. **The safest improvement in the stack: never hurts, sometimes exponentially helps, with a free cross-task transfer theorem on top.**
