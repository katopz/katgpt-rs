# Plan 623: Event-State Window Primitive — Event-Anchored Sigmoid Segmentation + Identity-Free Population Summary

**Date:** 2026-10-08
**Status:** COMPLETE — Phase 1 LANDED 2026-10-10 (G1–G4 ALL PASS, bench 931); Phase 2 LANDED 2026-10-10 — T2.1 PoC **GO** (riir-ai bench 978: Type-A cross-half index 1.000, Type-B control 0.2917 ≈ chance, β̂ 0.9% off λ_true, Δ̂=settling_ticks validated 34.08/35.00); **`event_state_windows` PROMOTED TO DEFAULT 2026-10-10** per the GOAT gate below (katgpt-core default list + README counts 205→206 + feature-def/lib.rs comments; default lib 2151/0 — the 10 module tests joined the default run; count_features + cargo_comment_audit + bench_doc_audit green; docs gate 35/35 with the standing seal-std partial-clone deferral; G2b default-posture re-read 103.8 µs @10k vs the 100 µs bar = the recorded trainer-load caveat, quiet re-run owed when the 4090 trainer exits — bench 931's PASS stands, code byte-identical, promotion adds only the default-list line)
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

- [x] **T1.1** Create `crates/katgpt-core/src/event_state_window.rs` with the two functions above; feature `event_state_windows = []` in katgpt-core `Cargo.toml`; module gated `#[cfg(feature = "event_state_windows")]`.
  → LANDED 2026-10-10: flat module (the state_option_scoring shape), root-feature forward added; consumes ONLY always-on substrate (simd::fast_sigmoid ±40-saturation + stats::nearest_rank — the percentile-of-record, tail supports carried in `StateSummary.supports`); Δ stays a parameter per line 19 (no settling_ticks dep, no feature implication).
- [x] **T1.2** G1 correctness tests (closed-form):
  → 10/10 inline tests (bench 931 G1 section): partition-of-unity + exclusion pin + γ limits (exact 0/1 via fast_sigmoid saturation; the σ(0)=0.5 edge points pinned as the ambiguous class) + permutation BIT-identity (sorted-order fold) + closed form + NaN/empty/single shapes.
- [x] **T1.3** G2 latency bench (`--release` mandatory), **with a baseline arm**:
  → bench 931: window 6 ns p99 (bar 20); summary N=10k 98.3 µs p50 (bar 100 µs) with the fixed-window baseline arm; anchoring delta +500 ns — PASS under documented trainer load (re-run quiet at Phase 2). First run's per-call Instant pairs read the timer quantum (p50 0 ns/p99 100 ns) — batched 256/sample.
- [x] **T1.4** Property test: fuzz boundaries for `b_next + Δ_next > b_prev + Δ_prev` violations — malformed boundary sequences must return the zero window, never panic.
  → 20k-iteration LCG fuzz (exact-zero on every malformed combo incl. NaN/inf/γ≤0/inverted support; >15k live-path samples bounded [0,1]). Note: b_next=+inf is the VALID terminal shape, not malformed.
- [x] **T1.5** Run `scripts/full_gate.sh` posture relevant to the crate (`cargo clippy -p katgpt-core --all-targets --features event_state_windows`); `cargo refine` before manual lint fixes.
  → clippy clean at the feature posture (all-targets); default lib 2141 passed 0 failed; refine 0 edits (compile-gated).

## Phase 2 — Consumer PoC (riir-ai side; runs after Phase 1 lands — the go/no-go gate, not deferrable)

- [x] **T2.1** riir-ai P1 PoC per Research 396 §5 (cross-half Type-A index on crowd latents; boundary-law cliff incl. the event-time-only control; measures β for Δ). Dependency: Phase 1 merged. Promotion of the feature to default is gated on this — the paper's domain evidence does NOT auto-transfer (Research 611 §6).
  → **GO, LANDED 2026-10-10** (riir-ai bench 978 + `riir-poc/tests/event_state_windows_poc.rs`, feature `event_state_windows_poc`): Type-A cross-half index **1.000** (bar 0.40), Type-B control **0.2917** ≈ chance (bar ≤ 0.50), β̂ = 0.08415 vs λ_true 0.08338 (0.9%), `settling_ticks(β̂, 0.05)` = 34.08 vs direct 35.00 ticks, event-anchored ≥ fixed/random, settle-geometry 9.9× (Δ=0 vs Δ̂ own-centroid dist), clock control 0.50, shuffled 0.000, bit-identical debug/release. **Recorded traps:** derangement private-maps leak 1.000 (index-exclusion bias), deterministic shift-cycle leaks 0.750 (cross-half-aligned pairing); the control needs random BALANCED shift decks per half. **Recorded finding:** with baseline heterogeneity ON the control lifts to 0.750 (second-order center×baseline pairing read by spread/quantiles — the CLEAN posture is the instrument; the read is a P2-wiring input). **Recorded negative-magnitude:** the paper's −62–70% accuracy cliff does NOT transfer at this SNR (saturation); the Δ law's contribution shows in geometry (9.9×), not accuracy. Held-out episode-split 0.333 — episode-level transfer is P3's business, not established here.

## GOAT gate

- G1 partition/exclusion/invariance closed-form tests green.
- G2/G4 per T1.3 (release + alloc canary).
- G3 no-regression: feature is additive, default-off; nothing existing changes.
- Promotion to default: only after riir-ai T2.1 PoC confirms the Type-A index on crowd latents (the paper's domain evidence does NOT auto-transfer — Research 611 §6). → **Condition MET + PROMOTED 2026-10-10** (default-list flip + the five count sites 205→206 + comment updates; verification: default lib 2151/0, `count_features.py`/`cargo_comment_audit.py`/`bench_doc_audit.py` green, docs gate 35/35 at the standing partial-clone deferral).
