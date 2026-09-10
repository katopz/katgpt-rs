# Plan 589: QSG Gossip Primitive — Quantized Simplex Belief Dynamics Kernel

**Date:** 2026-09-10
**Status:** Active — Phase 1 not started
**Research:** [katgpt-rs/.research/542_QSG_Memetic_Drift_Simplex_Gossip.md](../.research/542_QSG_Memetic_Drift_Simplex_Gossip.md)
**Source paper:** [arXiv:2603.24676](https://arxiv.org/abs/2603.24676) — Tanaka, "When Is Collective
Intelligence a Lottery? Multi-Agent Scaling Laws for Memetic Drift in LLMs" (Mar 2026)
**Target:** `crates/katgpt-core/src/qsg_gossip.rs` (new module) + Cargo feature `qsg_gossip`
**Consumer (later, separate repos):** riir-ai Issue 907 / Research 369 guide

---

## Goal

Ship the modelless QSG kernel behind opt-in feature `qsg_gossip`: N agents holding K-option simplex
beliefs, sampled Top-m/Hard/Soft messages, α-rate listener blending, order parameters (U, V, S), plus
a physics-validation harness that proves the shipped kernel reproduces the paper's scaling laws
(`t_cons ∝ N²`, early drift `∝ 1/(mN²)`, 1/m bandwidth law, logistic fixation in `Γh = mNh/α`).
GOAT gate = the kernel IS the law (G1 physics identity + scaling reproduction, G2 perf at
`signed_coupling` class, G3 no default-path regression, G4 alloc-free hot path). Promotion to default
only after the gate passes AND a production consumer lands (riir-ai Issue 907 class).

Sibling-slot note: `signed_coupling` owns the crowd **temperature** axis; this module owns the
**bandwidth/adaptation** axes (m, α) on persistent simplex state. Neither subsumes the other; cross-doc
both.

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** `crates/katgpt-core/src/qsg_gossip.rs`: `QsgConfig { n, k, alpha, m: MessageMode }` with `MessageMode::{Hard, TopM(u32), Soft}`; const-generic-friendly `K` via runtime `k` + fixed-cap scratch ([f32; MAX_K] pattern, MAX_K = 16) — no Vec in state.
- [ ] **T1.2** `qsg_init_uniform_into(x: &mut [f32], n, k)` + `qsg_normalize_into` (L1 simplex projection with clamp-to-ε, renormalize — drift-safe).
- [ ] **T1.3** `qsg_sample_message_into(msg: &mut [f32], belief: &[f32], k, mode, uniforms: &[f32])` — categorical draw from caller-supplied uniforms (RNG-free kernel pattern, mirrors `sample_states_into`); TopM = m iid draws → empirical freq; Soft = copy.
- [ ] **T1.4** `qsg_gossip_step_into(beliefs: &mut [f32], cfg, s: usize, l: usize, msg_scratch: &mut [f32], uniforms: &[f32])` — listener blend `x_L ← (1−α)x_L + α·y` in place; zero alloc.
- [ ] **T1.5** Batch driver `qsg_gossip_run_into(..., interactions: u32, pair_rng: impl FnMut() -> (usize, usize))` or uniform pair sampling from uniforms slice — one interaction = O(k).
- [ ] **T1.6** Reducers: `qsg_mean_into`, `qsg_polarization_u(mean,k)`, `qsg_disagreement_v`, `qsg_coordination_s`; plus per-speaker fuel `qsg_uncertainty(x) = 1 − ‖x‖²`.
- [ ] **T1.7** Unit tests (G1 physics identities, the load-bearing part):
  - [ ] Thm 1 one-step identity: `E[ΔU]_hard − E[ΔU]_soft = (α²/N²)·E[1−‖x_S‖²]` — Monte Carlo over ≥10⁵ shared-snapshot draws, tol 1e-3 relative.
  - [ ] Thm 2: same for TopM(m), and the 1/m corollary at symmetry `(1−1/K)/m` (paper Fig 6c, R² ≥ 0.999).
  - [ ] Soft martingale: `E[x̄'|X] = x̄` and `E[ΔV|X]_soft ≤ 0` (Eq. 8 factor `−2α/(N−1)(1−α+αN)`).
  - [ ] α=1 voter reduction: neutral init → winner uniform over K across ≥10⁴ seeds (martingale/optional-stopping property).
  - [ ] Soft from symmetric init never breaks symmetry (exact invariance test).
- [ ] **T1.8** `Cargo.toml`: feature `qsg_gossip = []` (opt-in, not default); module `#[cfg(feature = "qsg_gossip")]`; re-export from lib.rs docs.

## Phase 2 — Physics-Validation Harness (the GOAT gate bench)

### Tasks

- [ ] **T2.1** `crates/katgpt-core/tests/bench_589_qsg_scaling_laws.rs` (required-features = qsg_gossip): sweep N ∈ {2,4,8,16,32,64,128}, m ∈ {1,2,3,5,10}, α ∈ {0.1,0.3,1.0}; assert early drift `ΔU/step ∝ 1/(mN²)` and `t_cons(U⋆=0.9) ∝ N²` (log-log slope fit, R² ≥ 0.98) and the mean-field curve `U(t) = 1−(1−1/K)exp(−α²t/(mN²))` within the paper's Jensen/α=1 correction bands (small α: sim below curve; α=1: above).
- [ ] **T2.2** Γh crossover test: K=2, speaker bias h ∈ {0.02, 0.05, 0.1}; measured Pr(fix) vs logistic `1/(1+exp(−mNh/α))`; assert |Γh| ≪ 1 → ≈0.5, |Γh| ≫ 1 → ≈1; crossover Nc scaling `α/(m|h|)` monotone in all three.
- [ ] **T2.3** Tempered sampling `g_T(x) ∝ x^{1/T}` + `ΓT = (mN/α)|1/T−1|` crossover at ΓT ≈ 1 (T < 1 amplifies, T > 1 damps — linearization Eq. 40).
- [ ] **T2.4** Perf (G2): interaction cost ≤ 20 ns/interaction at K=8, N=1024 (signed_coupling class, Bench 672 precedent 1.8 ns/edge); 10⁶-interaction sweep < 100 ms release. Record in `.benchmarks/589_qsg_gossip_goat.md` with wall/CPU + exclusivity note.
- [ ] **T2.5** G4: alloc-count assertion on the hot path (no allocation per interaction; state is caller-owned slices).

## Phase 3 — Corpus Integration (thin, cross-doc)

### Tasks

- [ ] **T3.1** `mean_field` (Plan 371) cross-doc row: U/V as optional additional reducers — implement only if the reducer pass accepts a pluggable projection WITHOUT touching the default path; otherwise defer to the riir-ai consumer issue (note which).
- [ ] **T3.2** Cross-doc: R497 `signed_coupling` (sibling dial table: T axis vs m/α axes; K=2 QSG ≈ voter-model limit of signed_coupling at α=1 — state the precise relationship, do not blur them).
- [ ] **T3.3** Cross-doc: DEC `belief_mass_divergence` conservation validator for K ≤ 3 gossip flows (explicit curse-of-dimension caveat: not for high-K).
- [ ] **T3.4** README/docs: crowd-dynamics family table row (signed_coupling / qsg_gossip / mean_field) + feature catalog entry.

## Phase 4 — GOAT Gate Verdict & Promotion

### Tasks

- [ ] **T4.1** Run full gate: `cargo clippy -p katgpt-core --features qsg_gossip --all-targets -- -D warnings` + `cargo test -p katgpt-core --features qsg_gossip --lib` (count floor: record, never bare exit-0 — the `#![cfg]` green-zero rule; the test file needs required-features AND the count pinned in the bench doc).
- [ ] **T4.2** G1–G4 verdict table in `.benchmarks/589_qsg_gossip_goat.md`; verdict GOAT requires ALL of: physics identities (T1.7), scaling reproduction (T2.1–T2.3), perf (T2.4), alloc-free (T2.5).
- [ ] **T4.3** Promotion decision: stay opt-in until riir-ai Issue 907 (or equivalent) lands a production consumer + its own physics gate; then revisit default promotion with demote/keep verdict for the crowd-family slot.

## Constraints

- Modelless only (paper is closed-form math; no training anywhere).
- RNG stays OUT of the kernel (caller-supplied uniforms) — determinism + replay friendliness.
- Sigmoid/softmax discipline: categorical sampling over the simplex is a distribution draw (required by the math); blend gates remain sigmoid; no softmax projections added anywhere.
- Latent/raw: kernel state is local latent; nothing here touches sync paths.
- `#![cfg]` discipline: T1.7/T2.x tests gated by `qsg_gossip` must appear in a count-pinned row (T4.1), never a green-zero.
