# Benchmark 710: Hebbian KARC Readout GOAT (G2 perf + G4 alloc)

**Date:** 2026-09-11
**Plan:** [riir-ai Plan 584](../../riir-ai/.plans/584_karc_hebbian_wout_unification.md) T3 (closes Phase 1; ndb Plan 322 T3.2/T4.3 design execution)
**Bench:** [`crates/katgpt-core/benches/bench_584_hebbian_karc_readout_goat.rs`](../crates/katgpt-core/benches/bench_584_hebbian_karc_readout_goat.rs)
**Primitive:** `karc::hebbian_readout` (`HebbianKarcReadout`, feature `karc_hebbian_readout`, opt-in)
**Upstream:** Plan 559 (the Hebbian primitive, DEFAULT-ON) + ndb Bench 463 (bridge seam) + riir-ai Bench 908 (runtime seam — the load-tolerant gate methodology this bench reuses)
**Hardware:** M3 Max, macOS — shared box (concurrent sibling cargo; ratios + wrapper-Δs are the load-independent claims)

## Goal

Validate the `w_out`-fit-Hebbian-flavored seam: `HebbianKarcReadout::fit` (training pairs → facts via inner-product-exact `pad64`) + `forecast_into` (zero-alloc head-slice over `forward_into`), measured beside the arm it stands in for — `KarcForecaster::fit_ridge`.

## G2 perf — PASS ✅ (3/3 stable runs)

| Gate | Measured | Target | Verdict |
|---|---|---|---|
| G2a fit per-fact ratio | hebbian **10.6 µs/fact** (F=128, m=128) vs `fit_ridge` **15.7–15.9 µs/fact** (F=512 > d_h=256) — **ratio 0.7×** | ≤ 10× | ✅ |
| G2b forecast latency | **1.00 µs/query** · direct `forward_into` 0.50 · **wrapper Δ +0.50 µs** | Δ ≤ 1 µs | ✅ |

**Headline:** the Hebbian arm is *faster per fact* than the ridge arm at runtime scale — the m=128 f32 closed form beats the d_h=256 f64 Cholesky — while carrying strictly more (the F-forward margin audit + fact semantics). The forecast wrapper costs +0.5 µs over the raw kernel (the pad64 copy + head-slice + stack zero-fills; ≤1 µs gate).

## G4 alloc-free — PASS ✅

| Kernel | Allocs / 100 calls |
|---|---|
| `HebbianKarcReadout::forecast_into` | **0** |

## GOAT gate summary

| Gate | Status | Detail |
|---|---|---|
| G1 correctness | ✅ PASS | 5 in-module gates (observed-pair retrieval at belief scale, bit-identical determinism, pad64 inner-product exactness, F=1 infinite margin, error shape); feature-on 1992 = default 1987 + exactly 5 |
| G2 perf | ✅ PASS | Table above (ratio + wrapper-Δ gates; absolutes reported with load context) |
| G3 no-regression | ✅ PASS | Default clippy `-D warnings` clean; feature-gated module compiles to nothing at default |
| G4 alloc | ✅ PASS | 0 allocs / 100 forecast calls |
| G5 modelless | ✅ structural | Closed-form construction + BLAKE3 seed; no GD |
| G6 freeze | — (Phase 2) | The ndb envelope leg (T4/T5) carries it |

## Honest caveats

1. **Cross-arm data differs** — the ridge arm is fed via the real `observe_and_maybe_pair` path (consecutive delay-window → next-belief pairs, F=512 for Gram PD-ness at d_h=256); the Hebbian arm fits synthetic spike-seeded pairs (F=128). Timing is shape-driven (per-fact, same widths), which the ratio gate needs; a same-data comparison is the T8 demo's job (accuracy axis, not this perf gate).
2. **The ridge arm's F=512 requirement** — `fit_ridge` is structurally rank-deficient at `n_samples < d_h` (the first run's Cholesky panic at F=128 was exactly this); its natural regime is long accumulation, the Hebbian arm's is selective journaling. The per-fact ratio is the honest comparison.
3. **Absolutes are load-dependent** (shared box, sibling cargo) — ratios, wrapper-Δs, and alloc counts are the load-independent claims (the Bench-908 methodology).
4. **f32 vs f64** — the Hebbian closed form is f32 throughout vs `fit_ridge`'s f64 solve; at belief-scale targets this is the documented precision; the accuracy-axis demo (T8) must state its tolerance.

## What this closes

- **Plan 584 Phase 1 (T1+T2+T3)** — the public primitive + G1 gates + this GOAT record. Stays opt-in (`karc_hebbian_readout`); promotion follows the Plan 584 T9 consumer gate.

## What remains (Plan 584)

- Phase 2 (T4/T5): the ndb `karc_hebbian_envelope` commitment leg + G6 gates.
- Phase 3 (T6–T9): the riir-ai consumer (observe-time pair capture), the value-table graduation, the demo, the promotion decision.

## References

- riir-ai Plan 584: [`riir-ai/.plans/584_karc_hebbian_wout_unification.md`](../../riir-ai/.plans/584_karc_hebbian_wout_unification.md)
- ndb Plan 322 (T3.2 scoping note — the "Hebbian closed form AS the w_out fit" thesis): [`riir-neuron-db/.plans/322_hebbian_fact_storing_shard_bridge.md`](../../riir-neuron-db/.plans/322_hebbian_fact_storing_shard_bridge.md)
- Paper: [arXiv:2607.10034](https://arxiv.org/abs/2607.10034) — Garcia et al., "MLPs are Hebbians" §5.2
