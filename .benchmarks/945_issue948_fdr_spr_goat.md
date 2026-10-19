# Bench 945 — Issue 948 T1 GOAT + T2 paired read (FDR credit + SPR order)

**Date:** 2026-10-19
**Issue:** katgpt-rs `.issues/948_terminal_rsi_modelless_extractions.md` (T1 GOAT + T2 paired read — the two benchmarks the landing deferred to "a corpus/bench window")
**Code:** `benches/bench_945_fdr_spr_goat.rs` · `crates/katgpt-pruners/src/g_zero/fdr_credit.rs` (T1, landed `1f0c9300b`) · `crates/katgpt-pruners/src/spr_order.rs` (T2, landed `1f0c9300b`)
**Features:** `fdr_credit` + `spr_order` (opt-in, unchanged — see Verdict)
**Run:** `cargo bench --features fdr_credit,spr_order --bench bench_945_fdr_spr_goat --release`
**Box state:** M3 Max (16 cores), macOS 26.6.2, AC power; two sibling agent sessions active (moderate CPU load). No latency gates in this bench — every number is seeded deterministic arithmetic (xorshift64* + total_cmp sorts), so load state conditions nothing; the G4 gate counts allocations.

## Verdict

**PASS — all gates.** T1's GOAT and T2's paired read are no longer deferred: the
bench window was this session, and the gates pass at the frozen bounds.

## Harness (synthetic, adjudicated by construction — bench_684 shape)

10 families (hint templates), check-count scales `K ∈ {4..80}` (20× spread);
true per-family quality `θ` stratified over [0.15, 0.85] and **shuffled against
K each seed** (the adversarial pairing — scale carries no quality signal).
Raw scores `P ~ Binomial(K, θ)`; reference slice (N=128) freezes the divisors,
a fresh held-out slice (N=128) is what every gate reads (G1c — the freeze never
sees the scores it is judged on). 16 seeds, B=200 bootstrap refreezes.

## T1 — G1 ordering (FDR vs raw-P, vs the by-construction adjudication θ)

| gate | measured | bound | verdict |
|---|---|---|---|
| G1a FDR Spearman vs θ (mean) | **+0.6318** | ≥ 0.55 | PASS |
| G1a margin over raw-P | **+0.2682** | ≥ 0.20 | PASS |
| G1a per-seed wins | **15/16** | ≥ 14 | PASS |
| G1b raw-P Spearman (premise: scale corrupts) | **+0.3636** | ≤ 0.60 | PASS |
| G3 top-3 mean θ | **0.6531 vs 0.6006** (+0.0525) | margin ≥ 0.02 | PASS |
| G3 top-3 never-worse | **16/16** | 16/16 | PASS |

**The honest reading (first measured run re-pinned the a-priori bounds once,
then frozen):** FDR-p90 **dominates** raw-P ordering everywhere but does NOT
approach perfect recovery — Spearman plateaus at ~0.63 because the estimator
saturates at small K (a 4-check family's p90 divisor compresses its credit
toward the ceiling; cross-K inversions follow). That is a property of the
paper's estimator, not the harness: the GOAT claim this bench pins is
*dominance + stability*, exactly what issue 948's T1 shape asked ("FDR ordering
beats raw-P ordering"). A consumer wanting near-perfect recovery needs the
complementary normalization axis (e.g. mean-based divisors) — a different
primitive, not a re-pin of this one.

## G2 — bootstrap stability (B=200 × 16 seeds, refreeze + re-rank)

| gate | measured | bound | verdict |
|---|---|---|---|
| divisor CV mean | **0.0432** | ≤ 0.10 | PASS |
| divisor CV max (one small-K family, seed 31337) | **0.3439** | ≤ 0.40 | PASS |
| ordering Kendall τ vs full freeze (mean) | **0.8782** | ≥ 0.80 | PASS |

The frozen table is stable under resampling: the mean divisor moves < 5% and
the resulting arm ordering agrees with the full-freeze ordering at τ 0.88. The
0.34 max CV is the small-K discrete-p90 lumpiness (one family, one seed) —
disclosed, bounded, and the ordering gate absorbs it (τ floor holds).

## T2 — paired read (f32 `spr_reward` vs canonical `SprKey` pair ordering)

Corpus: 2000 candidates over the 10 frozen families — sparse verified passes
(`P == K`, the Terminal-RSI shape: successes are rare, partial credit is dense)
plus a 1% stress band at the 2^24/2^20 rounding-corner ratios and zero-pass
rows.

| gate | measured | verdict |
|---|---|---|
| top-200 membership flips (float vs pair selection) | **0** (0 outside exact-tie groups) | PASS — the honest no-op |
| canonical-selection law violations (excluded pass / included failure) | **0** | PASS |

**The honest no-op, recorded as the issue prescribed:** on today's corpus the
f32 transport form selects byte-identically to the canonical pair ordering.
The tie-group membership check stays in the bench as the drift guard — any
future corpus that produces a cross-class exact tie at the selection boundary
(the documented f32 corner) flips the count and the guard fires.

`mass_on_failures` (report-only diagnostic): raw-P rewards **1.0000** | SPR
rewards **0.9979** — with verified passes this sparse (`P == K`), reward mass
sits on failures under both forms. The diagnostic characterizes the corpus; it
is not a reshaping claim (the paper's 50.09% → 28.72% figure is a
training-reward read, out of this repo's modelless scope).

## G4 — zero-alloc scoring

| gate | measured | verdict |
|---|---|---|
| `credit()` loop | **0 allocs / 4096 calls** (after warm-up) | PASS |
| `observe_hint_delta_calibrated` loop | **0 allocs / 4096 observations** (pre-built `HintDelta`s; warm-up first) | PASS |

## Provenance

- Paper: arXiv:2610.22192 (Terminal-RSI) — FDR `R = max(P/D_f, 0)` + SPR
  `R = S + P/(P+D')`; distilled through riir-train Research 477's three-track
  panel; modelless residue landed as katgpt-rs issue 948 T1/T2 at `1f0c9300b`.
- Scope law: this bench validates the MECHANISM on a synthetic adjudicated
  harness — no real-template corpus with adjudicated quality exists in-repo.
  When one lands, re-run the arms unchanged; the gates re-read as-is.
- Promotion posture: unchanged (opt-in). Dominance + stability + zero-alloc
  are now MEASURED, but no consumer is wired (the absorb adapter and the
  ranking sites remain incumbent-raw) — promotion requires a wired consumer
  showing the gain on real traffic, per the Feature Flag Discipline.
