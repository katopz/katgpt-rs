# Plan 592: Nonergodic Belief Kernel (`nonergodic_belief`)

**Status:** Active — Phase 0 (not started)
**Date:** 2026-09-10
**Research:** [katgpt-rs/.research/545_Nonergodic_Belief_Decomposition.md](../.research/545_Nonergodic_Belief_Decomposition.md)
**Source:** Simplex blog "The geometry of nonergodic composition" (simplex.pub/nonergodic-geometry/, 2026-09-09); classical ancestry MHT (Reid 1979) / IMM (Blom & Bar-Shalom 1988) — cited, not claimed.
**Target:** `crates/katgpt-micro-belief/src/nonergodic.rs` (new) + Cargo feature `nonergodic_belief`
**Cross-ref:** riir-ai `.research/373` (game-runtime guide — wiring is OUT of scope here)

---

## Goal

Ship the two-level nonergodic Bayes filter as a generic, modelless, zero-alloc runtime primitive: K per-generator inner beliefs `η_n` (dim D each) + online posterior `w_n` over generators, updated per-block by each generator's own likelihood and coupled only by one scalar normalizer, with the telescoping readout `w_n·η_n`. K≤16, D≤64, fixed-size arrays, SIMD dots. GOAT gate: exact-posterior correctness (G1), sub-µs/tick (G2), no regression on `bom_sampling` (G3), alloc-free (G4), plus an identification-quality bench vs single-belief and BoM-K baselines.

**Substrate-first note:** substrate audit done 2026-09-10 (Research 545 §2/§3): closest substrate is `katgpt-micro-belief/src/bom.rs::sample_k_states` (right `[K·D]` slot shape, noise-copy hypotheses) + `katgpt-core/src/distributional_steering.rs::bom::hypothesis_weights_into` (one-shot tilt weights). Zero TWO-LEVEL surfaces workspace-wide under 6 name variants. This plan CONSUMES that substrate's layout conventions; it does not duplicate a filter that exists.

## Phase 1 — Core kernel

### Tasks

- [ ] **T1.1** `ComponentModel` trait (public, generic — NO game/domain vocabulary, this repo is public MIT): `fn likelihood(&self, eta: &[f32], token: u8) -> f32` + `fn update_into(&self, eta: &[f32], token: u8, out: &mut [f32])` (block Bayes step: `η' ∝ η·T^(token)`, normalized). Reference impls for tests: `BernoulliPair` (two coins) + `Mess3Block` (transition matrices from the blog appendix, parameterized α/x).
- [ ] **T1.2** `NonergodicFilter<const K: usize, const D: usize>`: `eta: [[f32; D]; K]`, `log_w: [f32; K]`, constructor from `ComponentModel` array + uniform/prior weights. `tick(&mut self, token: u8)`: per-block `ℓ_n = likelihood(...)`, `update_into` + normalize; `log_w[n] += ln(ℓ_n)`; `Z = LSE(log_w)`; renormalize `log_w` in place (numerically stable per-tick form; Research 545 §2).
- [ ] **T1.3** Telescoping readout: `telescope_into(&self, out: &mut [f32; K*D])` writes `w_n·η_n` in place (the paper's `(w₁η₁,…,w_Nη_N)` object; BoM slot layout). Normalized-view accessors: `weights() -> &[f32; K]`, `component(&self, n) -> &[f32; D]`.
- [ ] **T1.4** Behavior semantics: `committed() -> (usize, f32)` (argmax + `1 − max w` uncertainty) and sigmoid-gated revive probe `revive_margin(&self) -> f32` (gap between top-2 in log space; consumer decides thresholds — the filter reports, never gates internally).
- [ ] **T1.5** Zero-alloc + SIMD: reuse `katgpt-core` simd dot helpers; `#[inline]` hot path; no Vec/Box in `tick`/`telescope_into`. Const-generic loop bounds; `match` dispatch, early returns.
- [ ] **T1.6** Feature wiring: `nonergodic_belief = ["dep:katgpt-core"]`-style opt-in in `katgpt-micro-belief/Cargo.toml`; NOT in default features. Cargo comment + `.docs` feature-catalog row per repo convention.

## Phase 2 — Tests + GOAT gate

### Tasks

- [ ] **T2.1** G1 exactness: property tests vs brute-force posterior (enumerate all length-L token sequences, compute exact `w_n` by definition) on random BernoulliPair + Mess3Block instances — agree to 1e-5 for L ≤ 12, K ∈ {2,3,5}.
- [ ] **T2.2** G1 identities: Σw = 1 after every tick; telescope view consistency (`telescope_into` block sums == w_n); collapse monotonicity (consistent-evidence streams → `max w` non-decreasing in expectation); revival test (contradiction stream re-inflates a collapsed w above its floor).
- [ ] **T2.3** G2/G4 bench: ns/tick + alloc count at K∈{2,8,16} × D∈{8,32,64} (criterion, release). Bars: tick < 1µs at K=8/D=8; 0 allocs after construction. Record in `.benchmarks/` per numbering discipline.
- [ ] **T2.4** G3 no-regression: existing `bom_sampling` tests green with and without the new feature; confirm no default-feature surface change (`cargo clippy --no-default-features` + `--features nonergodic_belief`).
- [ ] **T2.5** Identification bench: streams from K generators (coins + Mess3 mixtures) — accuracy / log-loss / calibration for (a) `NonergodicFilter`, (b) single-belief leaky integrator baseline, (c) BoM K-particle + `select_best` baseline. Gate: (a) beats both on log-loss at K=2..4; honest record if not (demotion rule in Research 545 §5).
- [ ] **T2.6** Report-the-Floor decision (documented, binding only if triggered): the filter exposes a hypothesis posterior, not forecast intervals — the conformal-naive floor rule (Research 322) binds ONLY if a predictive-interval surface is added on top. Write the one-paragraph decision into the bench doc; if any future interval accessor lands, the floor becomes a gate.

## Phase 3 — Promotion decision + docs

### Tasks

- [ ] **T3.1** Bench doc in `.benchmarks/` (next free number): G1–G4 table + T2.5 quality table + machine/GPU-exclusivity note (CPU-only — no GPU gate needed).
- [ ] **T3.2** Promotion ruling per AGENTS.md discipline: default-on ONLY with a runtime consumer GOAT (conformal precedent: primitive-level pass + consumer gates). Expected ruling on current evidence: stays opt-in until riir-ai P1/P2 consumers land (Guide 373).
- [ ] **T3.3** `README.md` feature-catalog row + `.docs` updates; PASS-Redirects lines added to closest cousins (R248, R302, R545 already cross-links) — one-line references so future greps hit.
- [ ] **T3.4** Commit on `develop` (`feat: nonergodic_belief kernel`), highwater bumps in same commit, push.

## Non-goals

- Game semantics (archetypes/agendas) — riir-ai Guide 373 owns that mapping; `ComponentModel` stays abstract.
- Unnormalized-accumulation variant (underflow-prone) — per-tick LSE renormalization is the shipped form; the telescoping object is formed at readout.
- Healer domain-router change — separate follow-up if P3 ever triggers.
