# Plan 623: Event-State Window Primitive — Event-Anchored Sigmoid Segmentation + Identity-Free Population Summary

**Date:** 2026-10-08
**Status:** Active — Phase 1 pending
**Research:** [katgpt-rs/.research/611_TWS_Event_State_Windows_Identity_Free_Population_Tokens.md](../.research/611_TWS_Event_State_Windows_Identity_Free_Population_Tokens.md) · [riir-ai/.research/396_crowd_regime_tokens_event_state_guide.md](../../riir-ai/.research/396_crowd_regime_tokens_event_state_guide.md)
**Source paper:** [arXiv:2610.03001](https://arxiv.org/abs/2610.03001) — Bae & Cha, "Neural Data Needs Semantic Tokenization" (TWS)
**Target:** `katgpt-rs/crates/katgpt-core/src/event_state_window.rs` (new module) + Cargo feature `event_state_windows`
**GOAT decision:** opt-in until G1–G4 pass; promote to default only if the gain is modelless (it is, by construction — zero trained parameters).

---

## Goal

Ship the two modelless primitives distilled from TWS so downstream crowd/shard systems can segment aggregates at semantic events instead of fixed windows:

1. `event_state_window(t, b_prev, b_next, delta, gamma)` — the two-sided sigmoid product window with the settle transition `[b, b+Δ)` excluded from both states (TWS Eq. 1).
2. `population_state_summary(values, scratch) -> StateSummary` — identity-free order statistics (mean, spread, 10/25/50/75/90% quantiles) over an unordered member set, zero-alloc; all statistics over the sorted buffer.

Note on Δ: **do not write a `derive_settle_offset` wrapper** — call `settling_ticks(β, ε)` directly (R586). β is the update rate of the crowd statistic being windowed — named at wiring time, measured by the Phase-2 PoC (tick-horizon until the per-tick manifold axes stop rotating past ε after an event); the primitive takes Δ as a parameter and stays agnostic.

The paper's evidence this is worth a feature flag: event-anchored boundaries vs fixed/random = +166–233% decode (0.565 vs 0.188–0.215), wrong boundaries cost −62–70% and training cannot repair them; identity-free summaries transfer to never-seen populations where per-member models score 0.000.

## Phase 1 — Primitive + correctness gates

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/event_state_window.rs` with the two functions above; feature `event_state_windows = []` in katgpt-core `Cargo.toml`; module gated `#[cfg(feature = "event_state_windows")]`.
- [ ] **T1.2** G1 correctness tests (closed-form):
  - partition-of-unity: for K adjacent states with shared boundaries, `Σ_k w_k(t) ≈ 1` *inside states* and `< 1` exactly on the excluded transitions (the transition belongs to no state — pin the exclusion with a literal test at `t ∈ [b, b+Δ)`).
  - permutation invariance, **bit-identical f32**: compute ALL statistics (mean, spread, quantiles) over the **sorted** scratch buffer — the sort is already paid for the quantiles, and summation over a fixed (sorted) order is permutation-invariant by construction. (Summing in input order is NOT: f32 addition is non-associative.)
  - γ→0 limit: window → hard indicator on `[b_prev+Δ, b_next)`; γ→∞ → flat (degenerate) — both pinned.
- [ ] **T1.3** G2 latency bench (`--release` mandatory), **with a baseline arm**: fixed-window aggregation (the incumbent shape — same statistics over a fixed `[t−W, t)` window, no event anchoring) as the comparison, so the claim is relative (event-anchored ≤ fixed-window + ε) and does not drift with box load. Absolute bars: per-call window eval ≤ 20 ns; summary over N=10k members ≤ 100 µs (the R311 shape). Zero heap allocation (G4 counting allocator canary).
- [ ] **T1.4** Property test: fuzz boundaries for `b_next + Δ_next > b_prev + Δ_prev` violations — malformed boundary sequences must return the zero window, never panic.
- [ ] **T1.5** Run `scripts/full_gate.sh` posture relevant to the crate (`cargo clippy -p katgpt-core --all-targets --features event_state_windows`); `cargo refine` before manual lint fixes.

## Phase 2 — Consumer PoC (riir-ai side; runs after Phase 1 lands — the go/no-go gate, not deferrable)

- [ ] **T2.1** riir-ai P1 PoC per Research 396 §5 (cross-half Type-A index on crowd latents; boundary-law cliff incl. the event-time-only control; measures β for Δ). Dependency: Phase 1 merged. Promotion of the feature to default is gated on this — the paper's domain evidence does NOT auto-transfer (Research 611 §6).

## GOAT gate

- G1 partition/exclusion/invariance closed-form tests green.
- G2/G4 per T1.3 (release + alloc canary).
- G3 no-regression: feature is additive, default-off; nothing existing changes.
- Promotion to default: only after riir-ai T2.1 PoC confirms the Type-A index on crowd latents (the paper's domain evidence does NOT auto-transfer — Research 611 §6).
