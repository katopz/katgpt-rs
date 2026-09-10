# Plan 589: QSG Gossip Primitive — Quantized Simplex Belief Dynamics Kernel

**Date:** 2026-09-10
**Status:** Done — Phases 1–4 complete (Bench 703 GOAT G1–G4 ALL PASS); consumer LANDED 2026-09-10 (riir-ai Plan 577 / Issue 907, `riir-games/src/swarm/qsg_crowd.rs`, Bench 902 GOAT ALL PASS) — stays opt-in while the consumer is itself default-off (promotion is a two-step chain: a shipped zone consumer adopts qsg_crowd → qsg_crowd promotes → this follows, the signed_coupling rule)
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

- [x] **T1.1** `crates/katgpt-core/src/qsg_gossip.rs`: `QsgConfig { n, k, alpha, m: MessageMode }` with `MessageMode::{Hard, TopM(u32), Soft}`; const-generic-friendly `K` via runtime `k` + fixed-cap scratch ([f32; MAX_K] pattern, MAX_K = 16) — no Vec in state.
- [x] **T1.2** `qsg_init_uniform_into(x: &mut [f32], n, k)` + `qsg_normalize_into` (L1 simplex projection with clamp-to-ε, renormalize — drift-safe).
- [x] **T1.3** `qsg_sample_message_into(msg: &mut [f32], belief: &[f32], k, mode, uniforms: &[f32])` — categorical draw from caller-supplied uniforms (RNG-free kernel pattern, mirrors `sample_states_into`); TopM = m iid draws → empirical freq; Soft = copy.
- [x] **T1.4** `qsg_gossip_step_into(beliefs: &mut [f32], cfg, s: usize, l: usize, msg_scratch: &mut [f32], uniforms: &mut &[f32])` — listener blend `x_L ← (1−α)x_L + α·y` in place; zero alloc. (+ `qsg_blend_listener_into` — the blend half alone, for caller-authored messages.)
- [x] **T1.5** Batch driver `qsg_gossip_run_into(..., interactions: u32, pair_rng: impl FnMut() -> (usize, usize))` or uniform pair sampling from uniforms slice — one interaction = O(k). (+ `uniform_ordered_pair` — the canonical uniform pairing as a pure fn.)
- [x] **T1.6** Reducers: `qsg_mean_into`, `qsg_polarization_u(mean,k)`, `qsg_disagreement_v`, `qsg_coordination_s`; plus per-speaker fuel `qsg_uncertainty(x) = 1 − ‖x‖²`.
- [x] **T1.7** Unit tests (G1 physics identities, the load-bearing part):
  - [x] Thm 1 one-step identity: `E[ΔU]_hard − E[ΔU]_soft = (α²/N²)·E[1−‖x_S‖²]` — paired estimator (per-pair analytic vs Q-message-averaged kernel), tol 1%.
  - [x] Thm 2: same for TopM(m), and the 1/m corollary at symmetry `(1−1/K)/m` (paper Fig 6c, ≤3%).
  - [x] Soft martingale: `E[x̄'|X] = x̄` and `E[ΔV|X]_soft = −(2α/(N−1))(1−α+α/N)V` (corrected form — the scraped Eq. 8 renders α/N as αN; the α→1 voter limit disambiguates).
  - [x] α=1 voter reduction: neutral init → winner uniform over K across 1500 seeds.
  - [x] Soft from symmetric init never breaks symmetry (exact invariance test).
- [x] **T1.8** `Cargo.toml`: feature `qsg_gossip = []` (opt-in, not default); module `#[cfg(feature = "qsg_gossip")]`; re-export from lib.rs docs.

## Phase 2 — Physics-Validation Harness (the GOAT gate bench)

### Tasks

- [x] **T2.1** `crates/katgpt-core/tests/bench_703_qsg_scaling_laws.rs` (required-features = qsg_gossip): N-scaling + 1/m + mean-field gates — landed as: per-step drift via reset-per-draw (≤1% per N ∈ {8..64}, ratio 16±0.32), 1/m ratios, and the trajectory gate against the EXACT two-moment (U,V) flow (ensemble of 48 runs vs E[U]/E[V] at 3 checkpoints to 2.5·t_char) — a strictly tighter reference than the V≈0 homogeneous closure the plan originally named.
- [x] **T2.2** Γh crossover: fixation collapse onto Γh = mNh/α verified across (N,h) ∈ 8 points; same-Γh pairs agree ≤0.043 across an 8× population range; **honest deviation recorded**: the measured logistic is σ(Γh/2) — the first-order diffusion drops the pair-choice heterogeneity variance, halving the effective slope. Nc direction (bias more decisive at larger N) gated.
- [x] **T2.3** Tempered sampling `g_T(x) ∝ x^{1/T}` + `ΓT = (mN/α)|1/T−1|` crossover: T=0.5 amplifies (U > 0.9), T=2.0 damps (U < 0.45), gap > 0.35.
- [x] **T2.4** Perf (G2): **20.5–20.8 ns/interaction** at K=8, N=1024 (gate ≤100 ms/10⁶; measured 20.6 ms release). Recorded in `.benchmarks/703_qsg_gossip_goat.md`.
- [x] **T2.5** G4: alloc-count gate — 0 allocs over 1000 TopM(3) steps + reducers (`qsg_gossip_alloc_check`, counting_allocator canary per Issue 714).

## Phase 3 — Corpus Integration (thin, cross-doc)

### Tasks

- [-] **T3.1** `mean_field` (Plan 371) cross-doc row: U/V as optional additional reducers — DEFERRED: the `MeanFieldOverlap` aggregation pass has no pluggable-projection seam (adding one would touch the DEFAULT path for an opt-in consumer that does not exist yet); the reducer composition moves to riir-ai Issue 907's plan, where the consumer justifies the seam.
- [x] **T3.2** Cross-doc: R497 `signed_coupling` — sibling dial table in the qsg_gossip module docs (T axis vs m/α axes; K=2 QSG at α=1 ≡ voter-model limit — stated, not blurred) + README family narrative.
- [x] **T3.3** Cross-doc: DEC `belief_mass_divergence` conservation validator for K ≤ 3 gossip flows (curse-of-dimension caveat stated) — module docs + feature comment.
- [x] **T3.4** README/docs: crowd-dynamics family row + feature catalog entry (README §qsg_gossip).

## Phase 4 — GOAT Gate Verdict & Promotion

### Tasks

- [x] **T4.1** Run full gate: `cargo clippy -p katgpt-core --features qsg_gossip --all-targets -- -D warnings` (clean at BOTH feature states) + `cargo test -p katgpt-core --features qsg_gossip --lib qsg` — **21 passed** (count floor recorded in Bench 703; never a bare exit-0).
- [x] **T4.2** G1–G4 verdict table in `.benchmarks/703_qsg_gossip_goat.md` — GOAT: physics identities (T1.7, 21 tests), scaling reproduction (G1a–G1e incl. the Γh collapse + the σ(Γh/2) deviation), perf 20.6 ns/interaction (T2.4), alloc-free (T2.5).
- [x] **T4.3** Promotion decision: **stay opt-in** until riir-ai Issue 907 lands a production consumer + its own physics gate; no crowd-family slot change (signed_coupling keeps T axis; no demotion — QSG opens the m/α axes).

## Constraints

- Modelless only (paper is closed-form math; no training anywhere).
- RNG stays OUT of the kernel (caller-supplied uniforms) — determinism + replay friendliness.
- Sigmoid/softmax discipline: categorical sampling over the simplex is a distribution draw (required by the math); blend gates remain sigmoid; no softmax projections added anywhere.
- Latent/raw: kernel state is local latent; nothing here touches sync paths.
- `#![cfg]` discipline: T1.7/T2.x tests gated by `qsg_gossip` must appear in a count-pinned row (T4.1), never a green-zero.
