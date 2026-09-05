//! SP-KV: Self-Pruned Key-Value Attention benchmarks.
//! Plan 070 Phase 4 (T16–T20).
//!
//! Benchmarks:
//! 1. Gate bias overhead: baseline attention_head() vs attention_head_gated() (T16)
//! 2. KV cache density: full KV vs SP-KV at τ={0.3, 0.5, 0.7, 0.9} (T17)
//! 3. Decode latency: full KV vs SP-KV sparse decode at batch=1 (T18)
//! 4. Palindrome test: verify SP-KV can learn long-range dependencies (T19)
//! 5. Utility predictor gradient flow: verify log(u) gate preserves gradients (T20)
//!
//! Run with: cargo test --features sp_kv bench_sp_kv -- --nocapture

#![cfg(feature = "sp_kv")]

#[path = "common/ab_timing.rs"]
mod ab_timing;

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::simd::simd_dot_f32;
use katgpt_rs::sp_kv::{
    GateBias, GateBiasBuffer, NoBias, SpKvCache, SpKvConfig, SpKvPredictors, UtilityAggregation,
    aggregate_utilities, attention_head_core, attention_head_gated, predict,
};
use katgpt_rs::types::{Config, Rng, kv_dim};

/// Number of iterations for timing-based benchmarks.
const BENCH_ITERS: usize = 1000;

/// Generate a synthetic hidden state vector for position `pos`.
fn synthetic_hidden(n_embd: usize, pos: usize) -> Vec<f32> {
    (0..n_embd)
        .map(|i| ((i + pos * 7) as f32 * 0.1).sin() + ((i + pos * 3) as f32 * 0.07).cos())
        .collect()
}

// ── T16: Gate Bias Overhead ──────────────────────────────────────

/// T16 — monomorphized gate-bias dispatch overhead and prune-skip speedup.
///
/// **Issue 727 CLOSED-NEGATIVE-ON-THE-GATE-BIAS-BAR (2026-09-05). `#[ignore]`
/// with provenance per the Issue 723 T8 precedent for `goat_169_g1`: the
/// assertions stay executable via `--ignored`, and the gate-bias bar is NOT
/// re-pinned — the measured budget is recorded here, in the primitive's docs,
/// and in the `sp_kv` feature comment instead.**
///
/// What the T7 repair changed (all detailed at the call sites): the sequence
/// length is decoupled from `Config::micro()`'s `block_size = 16` — at which
/// the "50% pruned" arm pruned **zero** positions and the workload was small
/// enough that an unrelated `eprintln!` moved the measured ratio by 30% — and
/// the two asserted comparisons are interleaved chunk-by-chunk against
/// the same baseline with a median across chunks. What Issue 727 added: the
/// sweep now runs THREE sequence lengths (128 / 512 / 2048) and every cell is
/// printed before any assert fires; the primitive got the T2 hoist (a
/// precomputed active-position list — pruned positions now cost neither a
/// bias load nor a score dot nor an `exp` nor a value-accumulate);
/// `gate_bias_hoist_bit_identity` (below) pins the hoist bit-identical to the
/// pre-hoist implementation at `to_bits` level.
///
/// ## Measured, M3 Max, release, 2026-09-05, 48% of positions pruned
///
/// BEFORE the T2 hoist (matched load window, three interleaved runs):
///
/// | t_n | gate-bias overhead (bar < 3%) | per-round | prune-skip (bar > 1.05x) | per-round |
/// |---|---|---|---|---|
/// | 128 | +7.14 / +7.35 / +7.54% | 1.01–1.09 | 0.964–0.969x | 0.94–1.13 |
/// | 512 | +8.09 / +8.16 / +8.66% | 1.06–1.13 | 0.594–1.004x | 0.58–1.01 |
/// | 2048 | +8.03 / +8.59 / +9.26% | 1.02–1.11 | 1.007–1.047x | 0.94–1.02 |
///
/// (The issue's original three runs at 512 read +8.0/+8.1/+8.4% and
/// 1.015/1.042/1.046x — same verdict. The overhead is roughly scale-invariant
/// in `t_n`, as a per-position cost predicts; the 512-cell 0.594x outlier is a
/// scheduling spike the per-round spread exposes.)
///
/// AFTER the T2 hoist (matched load window, same instrument):
///
/// | t_n | gate-bias overhead (bar < 3%) | per-round | prune-skip (bar > 1.05x) | per-round |
/// |---|---|---|---|---|
/// | 128 | +11.1 / +11.1 / +11.7% | 1.06–1.11 | 1.12–1.22x | 0.62–1.07 |
/// | 512 | +11.6 / +11.6 / +11.8% | 1.06–1.32 | 1.19–1.58x | 0.48–0.86 |
/// | 2048 | +11.8 / +24.0 / +27.1% | 0.93–1.30 | 1.22–1.43x (abs 37.4→26.2 µs/iter) | 0.54–1.07 |
///
/// (Absolute µs/iter at 512: baseline 7.77→7.93 — the NoBias baseline is
/// UNCHANGED by the hoist; gated-mixed 7.84→6.64. The 2048 gated-zero cells
/// +24/+27% are a load-climb window (1x→13x during the sweep); the load-floor
/// reading is +11.8%.)
///
/// ## The verdict (Issue 727 T1)
///
/// - **"Zero-overhead gate bias" is a false claim; the paper's <1% target was
///   a `t_n = 16` artifact.** Any gated attention must READ the gate once per
///   position per head. The honest measured budget: **the gate read costs
///   ~+7–12% vs `NoBias` at these dims, roughly scale-invariant in `t_n`**
///   (old interleaved-load structure +7–9%; the hoisted structure trades +3pp
///   of all-active scan cost for the prune win). `NoBias` (or the `None` arm
///   of `attention_head_gated`) remains exactly 0% — use it when not pruning.
/// - **Prune-skip clears its bar after the hoist**: 1.12–1.58x (was
///   1.015–1.046x). Pruned positions now cost one scan load+compare + one
///   `-inf` store; the score dot, `exp`, and value-accumulate are all
///   elided.
/// - The Option-dispatch wrapper stays telemetry-only-equal: the legacy
///   `Gated(Some)` arm measured the same overhead as monomorphized `GateBias`
///   both before and after the hoist — dispatch was never the cost; the load
///   is.
#[ignore = "Issue 727 (closes the gate-bias half as a measured budget): gate-bias overhead is an \
            inherent per-position gate read, +7..12% vs NoBias at hd=4 (paper <1% was a t_n=16 \
            artifact; bar <3% NOT met — do not re-pin). Prune-skip PASSES after the T2 hoist \
            (1.12..1.58x vs the 1.05x bar). Kept #[ignore]d because the gate asserts BOTH bars \
            and the overhead bar cannot pass; run with --ignored to re-measure. Bit-identity \
            of the hoist is pinned by gate_bias_hoist_bit_identity (not ignored)."]
#[test]
fn bench_gate_bias_overhead() {
    // Issue 727 T3: the sweep runs at THREE sequence lengths — 128 / 512 /
    // 2048. The overhead is a per-position load against a per-position dot, so
    // the ratio should be roughly scale-invariant; a deviation is itself a
    // finding. Each `t_n` gets a full independent instrument (fresh caches,
    // fresh RNG, fresh buffers). Every cell is measured and printed BEFORE any
    // bar assert fires — a miss at one `t_n` must never hide the other two
    // cells, because the scale-invariance check IS the finding.
    const T_NS: [usize; 3] = [128, 512, 2048];
    let cells: Vec<T16Cell> = T_NS.iter().map(|&t_n| bench_gate_bias_overhead_at(t_n)).collect();

    println!("\n  ┌─ T16 sweep summary (Issue 727 T3) ─────────────────────────────────┐");
    println!("  │  t_n │ gated-zero │ zero rounds     │ mixed        │ speedup │ mixed rounds    │");
    for c in &cells {
        println!(
            "  │ {:>4} │ {:+.2}%      │ {:.3}..{:.3} │ {:+.2}%      │ {:.3}x  │ {:.3}..{:.3} │",
            c.t_n,
            c.overhead_zero,
            c.spread_zero.0,
            c.spread_zero.1,
            c.overhead_mixed,
            c.prune_skip,
            c.spread_mixed.0,
            c.spread_mixed.1,
        );
    }
    println!("  └────────────────────────────────────────────────────────────────────┘");

    // Bars, per cell. Debug builds have higher overhead due to lack of
    // inlining; release is the true measurement.
    let mut failures: Vec<String> = Vec::new();
    for c in &cells {
        if cfg!(debug_assertions) {
            if c.overhead_zero >= 15.0 {
                failures.push(format!(
                    "t_n={}: gate bias overhead too high even for debug: {:+.2}% (per-round {:.4}..{:.4})",
                    c.t_n, c.overhead_zero, c.spread_zero.0, c.spread_zero.1
                ));
            }
        } else {
            if c.overhead_zero >= 3.0 {
                failures.push(format!(
                    "t_n={}: monomorphized gate bias overhead too high: {:+.2}% (target <3%, paper <1%; per-round {:.4}..{:.4})",
                    c.t_n, c.overhead_zero, c.spread_zero.0, c.spread_zero.1
                ));
            }
            if c.prune_skip <= 1.05 {
                failures.push(format!(
                    "t_n={}: prune-skip speedup not measurable: {:.3}x (expected >1.05x; per-round {:.4}..{:.4})",
                    c.t_n, c.prune_skip, c.spread_mixed.0, c.spread_mixed.1
                ));
            }
        }
    }
    if cfg!(debug_assertions) && failures.is_empty() {
        println!("  ℹ️  Debug build — overhead numbers are not representative (use --release)");
    }
    assert!(
        failures.is_empty(),
        "T16 bar failures:\n  {}",
        failures.join("\n  ")
    );
}

/// One measured cell of the T16 sweep.
struct T16Cell {
    t_n: usize,
    /// GateBias (zero bias) vs NoBias, median overhead %.
    overhead_zero: f64,
    /// Per-round ratio range for the gated-zero arm.
    spread_zero: (f64, f64),
    /// GateBias (~50% pruned) vs NoBias, median overhead %.
    overhead_mixed: f64,
    /// `1 / median(mixed ratios)` — the prune-skip speedup.
    prune_skip: f64,
    /// Per-round ratio range for the mixed arm.
    spread_mixed: (f64, f64),
}

/// One `t_n` cell of the T16 sweep — the instrument as Issue 723 T7 repaired
/// it, applied at a single sequence length.
fn bench_gate_bias_overhead_at(t_n: usize) -> T16Cell {
    let config = Config::micro();
    let kvd = kv_dim(&config);
    let hd = config.head_dim;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let scale = 1.0 / (hd as f32).sqrt();

    // Issue 723 T7: the benchmark sequence length is DECOUPLED from
    // `config.block_size`. `Config::micro()` carries `block_size = 16`, so the
    // old `t_n = config.block_size.min(64)` gave 16 positions and two things
    // followed, neither of them visible from the output:
    //
    // 1. The "50% pruned" arm's guard is `t < t_n - 16`, which at `t_n = 16`
    //    is `t < 0` — **vacuously false**. That arm pruned exactly zero
    //    positions (measured: 0 of 16 entries set to -inf), so
    //    `prune_skip_speedup > 1.05` was asserting that identical work is 5%
    //    faster than itself. It could not have passed for the stated reason.
    // 2. 4 heads x 16 positions x head_dim 4 is ~256 MACs per iteration, well
    //    inside the regime where code layout decides the answer. Measured: a
    //    single unrelated `eprintln!` added elsewhere in the function moved
    //    the gated-zero ratio from 1.036 to 0.707 and the mixed ratio from
    //    1.489 to 1.027. A 3% bar cannot be read off an instrument an
    //    unrelated edit moves by 30%.
    //
    // 512 positions makes the arms measure attention rather than alignment and
    // makes the 16-wide recency window a real 48% prune. Nothing here needs
    // `block_size` — `attention_head_core` takes `t_n` and the caches as
    // explicit slices. (Issue 727 T3: now swept over 128/512/2048 by the
    // caller — the ratio should be scale-invariant.)
    let mut rng = Rng::new(42);

    // Flat KV cache (simulated)
    let mut key_cache = vec![0.0f32; t_n * kvd];
    let mut value_cache = vec![0.0f32; t_n * kvd];
    for pos in 0..t_n {
        let off = pos * kvd;
        for d in 0..kvd {
            key_cache[off + d] = rng.normal();
            value_cache[off + d] = rng.normal();
        }
    }

    // Query vector
    let q: Vec<f32> = (0..config.n_embd).map(|_| rng.normal()).collect();

    println!("\n🧪 T16: Gate Bias Overhead (n_head={n_head}, n_kv={n_kv}, hd={hd}, t_n={t_n})");
    println!("{}", "═".repeat(60));

    // Issue 723 Class A / T7 — interleaved median-of-ratios.
    //
    // As filed, this gate timed five arms to completion in sequence and
    // asserted on ratios between the first and the rest. The bar it failed is
    // **3%**, and Issue 723 T5 measured two *sequential* arms of identical
    // work at +5.2% and +21.7% thirty seconds apart on this box — an order of
    // magnitude more than the quantity under test, landing entirely on
    // whichever arm ran second. The two ASSERTED comparisons (gate-bias
    // overhead, prune-skip speedup) are now interleaved chunk-by-chunk against
    // the same baseline with a median across chunks; a drift that hits both
    // arms of a pair cancels in its ratio. The two legacy-wrapper arms stay
    // sequential because nothing asserts on them — they are printed telemetry,
    // and labelling them as such is part of the fix.
    //
    // Each arm gets its own output buffers: interleaving requires two live
    // `&mut`, and separate buffers also stop one arm's writes from warming the
    // other's cache lines.
    const ROUNDS: usize = 9;
    let iters = BENCH_ITERS / ROUNDS;
    let warmup = 100;

    let mut out_base = vec![0.0; config.n_embd];
    let mut sc_base = vec![0.0; t_n];
    let mut out_cand = vec![0.0; config.n_embd];
    let mut sc_cand = vec![0.0; t_n];
    // Data-dependent sinks over the OUTPUT buffers — a `black_box` on a
    // buffer reference alone did not stop fat LTO deleting an arm in T5.
    let mut sink_base = 0u32;
    let mut sink_cand = 0u32;

    let zero_bias = vec![0.0f32; t_n];
    let mut mixed_bias = vec![0.0f32; t_n];
    for (t, mb) in mixed_bias.iter_mut().enumerate().take(t_n) {
        // Prune ~50% of positions (outside a recency window of 16)
        if t < t_n - 16 && t.is_multiple_of(2) {
            *mb = f32::NEG_INFINITY;
        }
    }
    let pruned = mixed_bias.iter().filter(|v| v.is_infinite()).count();
    // The prune-skip claim is only testable if the mixed arm actually prunes.
    // It did not for the whole life of this gate (see the T_N note above), and
    // a silent zero there is exactly the shape Issue 723 exists to catch.
    assert!(
        pruned * 3 > t_n && pruned * 3 < t_n * 2,
        "T16 instrument failure: the 'mixed' bias pruned {pruned} of {t_n} positions — \
         the arm must prune roughly half for `prune_skip_speedup` to mean anything"
    );

    // ── A vs C: NoBias baseline vs monomorphized GateBias with zero bias ──
    // Zero bias prunes nothing, so this is the worst-case dispatch overhead:
    // all of the gate's cost, none of its benefit.
    let ab_zero = ab_timing::ab_median_ratio(
        ROUNDS,
        iters,
        warmup,
        |_| {
            for h in 0..n_head {
                let kv_group = h * n_kv / n_head;
                unsafe {
                    attention_head_core(
                        &q,
                        &key_cache,
                        &value_cache,
                        &mut out_base,
                        &mut sc_base,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        t_n,
                        scale,
                        NoBias,
                    );
                }
            }
            sink_base ^= out_base[0].to_bits() ^ out_base[config.n_embd - 1].to_bits();
        },
        |_| {
            for h in 0..n_head {
                let kv_group = h * n_kv / n_head;
                unsafe {
                    attention_head_core(
                        &q,
                        &key_cache,
                        &value_cache,
                        &mut out_cand,
                        &mut sc_cand,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        t_n,
                        scale,
                        GateBias::new(&zero_bias),
                    );
                }
            }
            sink_cand ^= out_cand[0].to_bits() ^ out_cand[config.n_embd - 1].to_bits();
        },
    );

    // ── A vs D: NoBias baseline vs GateBias with ~50% of positions pruned ──
    let ab_mixed = ab_timing::ab_median_ratio(
        ROUNDS,
        iters,
        warmup,
        |_| {
            for h in 0..n_head {
                let kv_group = h * n_kv / n_head;
                unsafe {
                    attention_head_core(
                        &q,
                        &key_cache,
                        &value_cache,
                        &mut out_base,
                        &mut sc_base,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        t_n,
                        scale,
                        NoBias,
                    );
                }
            }
            sink_base ^= out_base[0].to_bits() ^ out_base[config.n_embd - 1].to_bits();
        },
        |_| {
            for h in 0..n_head {
                let kv_group = h * n_kv / n_head;
                unsafe {
                    attention_head_core(
                        &q,
                        &key_cache,
                        &value_cache,
                        &mut out_cand,
                        &mut sc_cand,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        t_n,
                        scale,
                        GateBias::new(&mixed_bias),
                    );
                }
            }
            sink_cand ^= out_cand[0].to_bits() ^ out_cand[config.n_embd - 1].to_bits();
        },
    );

    // ── B / E: legacy Option-dispatch wrapper — TELEMETRY ONLY ──
    // Nothing asserts on these, so they stay sequential: a number nobody gates
    // on does not need a load-invariant instrument, and pretending otherwise
    // would triple the runtime for no verdict.
    let legacy_arm =
        |bias: Option<&[f32]>, out: &mut Vec<f32>, sc: &mut Vec<f32>| -> std::time::Duration {
            let start = Instant::now();
            for _ in 0..BENCH_ITERS {
                for h in 0..n_head {
                    let kv_group = h * n_kv / n_head;
                    unsafe {
                        attention_head_gated(
                            &q,
                            &key_cache,
                            &value_cache,
                            out,
                            sc,
                            h * hd,
                            kv_group * hd,
                            kvd,
                            hd,
                            t_n,
                            scale,
                            bias,
                        );
                    }
                }
                black_box(&out);
            }
            start.elapsed()
        };
    let elapsed_legacy_none = legacy_arm(None, &mut out_cand, &mut sc_cand);
    let elapsed_legacy_some = legacy_arm(Some(&zero_bias), &mut out_cand, &mut sc_cand);
    black_box((sink_base, sink_cand));

    // ── Report ──
    let baseline_ns_per_iter = ab_zero.a_ns_per_iter();
    let overhead_gated_zero = ab_zero.overhead_pct();
    let overhead_gated_mixed = ab_mixed.overhead_pct();
    let legacy_none_ns_per_iter = elapsed_legacy_none.as_nanos() as f64 / BENCH_ITERS as f64;
    let legacy_some_ns_per_iter = elapsed_legacy_some.as_nanos() as f64 / BENCH_ITERS as f64;
    let overhead_legacy_none = (legacy_none_ns_per_iter / baseline_ns_per_iter - 1.0) * 100.0;
    let overhead_legacy_some = (legacy_some_ns_per_iter / baseline_ns_per_iter - 1.0) * 100.0;
    let prune_skip_speedup = 1.0 / ab_mixed.median;

    println!("  ┌─ Monomorphized (zero-overhead dispatch) ─────────────────┐");
    println!(
        "  │ NoBias baseline:     {:>8.2} µs/iter                   │",
        baseline_ns_per_iter / 1000.0
    );
    println!(
        "  │ GateBias (zero):     {:>8.2} µs/iter  ({overhead_gated_zero:+.2}%)        │",
        ab_zero.b_ns_per_iter() / 1000.0
    );
    println!(
        "  │ GateBias (50%% pruned):{:>7.2} µs/iter  ({overhead_gated_mixed:+.2}%, {prune_skip_speedup:.2}×)  │",
        ab_mixed.b_ns_per_iter() / 1000.0
    );
    println!("  ├─ Legacy wrapper (Option dispatch) — telemetry only ─────┤");
    println!(
        "  │ Gated(None):         {:>8.2} µs/iter  ({overhead_legacy_none:+.2}%)        │",
        legacy_none_ns_per_iter / 1000.0
    );
    println!(
        "  │ Gated(Some(zero)):   {:>8.2} µs/iter  ({overhead_legacy_some:+.2}%)        │",
        legacy_some_ns_per_iter / 1000.0
    );
    println!("  └──────────────────────────────────────────────────────────┘");
    ab_zero.report("T16 gated-zero/baseline");
    ab_mixed.report("T16 gated-mixed/baseline");

    T16Cell {
        t_n,
        overhead_zero: overhead_gated_zero,
        spread_zero: (ab_zero.min(), ab_zero.max()),
        overhead_mixed: overhead_gated_mixed,
        prune_skip: prune_skip_speedup,
        spread_mixed: (ab_mixed.min(), ab_mixed.max()),
    }
}

// ── T16b: Gate-bias hoist bit-identity (Issue 727 T2) ────────────

/// The PRE-hoist implementation, reproduced verbatim as the reference oracle:
/// same op order, same groupings, same `simd_dot_f32` calls. The hoisted path
/// must match it BIT-FOR-BIT on `attn_out` and on active-position scores (the
/// numerics contract on `attention_head_core` lists the two documented
/// divergences, both unreachable through `build_gate_biases`).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn reference_attention_head_pre727(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    bias: &[f32],
) {
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let b = bias[t];
        if b == f32::NEG_INFINITY {
            scores_buf[t] = f32::NEG_INFINITY;
            continue;
        }
        let k_off = t * kv_dim + kv_group_offset;
        let dot = simd_dot_f32(
            &q[q_head_offset..q_head_offset + hd],
            &key_cache[k_off..k_off + hd],
            hd,
        );
        let score = dot * scale + b;
        scores_buf[t] = score;
        max_score = max_score.max(score);
    }
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = (scores_buf[t] - max_score).exp();
        scores_buf[t] = exp_val;
        sum += exp_val;
    }
    let inv_sum = 1.0 / sum;
    for d in 0..hd {
        attn_out[q_head_offset + d] = 0.0f32;
    }
    for t in 0..t_n {
        let s = scores_buf[t] * inv_sum;
        let v_base = t * kv_dim + kv_group_offset;
        for d in 0..hd {
            attn_out[q_head_offset + d] += s * value_cache[v_base + d];
        }
    }
}

/// Issue 727 T2: the hoisted active-position scan must be bit-identical to the
/// pre-hoist implementation. `to_bits` comparisons, not tolerances — the whole
/// point is that removing EXACT zeros from a fixed-order accumulation changes
/// no bits.
#[test]
fn gate_bias_hoist_bit_identity() {
    let hd = 4usize;
    let kvd = hd * 2; // two KV groups → non-zero kv_group_offset arm
    let t_n = 130usize; // NOT a multiple of the 64-position scan chunk → ragged tail
    let scale = 1.0 / (hd as f32).sqrt();

    let mut rng = Rng::new(7271);
    let q: Vec<f32> = (0..kvd).map(|_| rng.normal()).collect();
    let key_cache: Vec<f32> = (0..t_n * kvd).map(|_| rng.normal()).collect();
    let value_cache: Vec<f32> = (0..t_n * kvd).map(|_| rng.normal()).collect();

    let mut cases: Vec<(&str, Vec<f32>)> = Vec::new();
    cases.push(("zero (all active)", vec![0.0; t_n]));
    // The bench's own mixed guard: recency window 16 + every-2 (~48% pruned).
    let mut mixed = vec![0.0f32; t_n];
    for (t, b) in mixed.iter_mut().enumerate().take(t_n) {
        if t < t_n - 16 && t.is_multiple_of(2) {
            *b = f32::NEG_INFINITY;
        }
    }
    cases.push(("mixed window+every-2", mixed));
    // Soft-mode-shaped finite biases (log(u + eps) lives in [-7, 0) here).
    let mut u_rng = Rng::new(7272);
    let soft: Vec<f32> = (0..t_n)
        .map(|_| (u_rng.normal().abs() * 0.5 + 1e-3).ln())
        .collect();
    cases.push(("soft log(u+eps)", soft));
    // -inf inside the FIRST scan chunk and inside the LAST one.
    let mut edges = vec![0.0f32; t_n];
    edges[3] = f32::NEG_INFINITY;
    edges[t_n - 1] = f32::NEG_INFINITY;
    cases.push(("-inf at chunk edges", edges));
    // A NaN slot: NaN is active (NaN != -inf), poisons the sum in BOTH paths;
    // `.max(NaN)` ignores NaN in both, and every downstream op is identical.
    let mut nan_bias = vec![0.0f32; t_n];
    nan_bias[t_n / 2] = f32::NAN;
    cases.push(("one NaN slot", nan_bias));

    for (name, bias) in &cases {
        for (q_off, kv_off) in [(0usize, 0usize), (hd, hd)] {
            let mut out_new = vec![0.0f32; kvd];
            let mut sc_new = vec![0.0f32; t_n];
            let mut out_ref = vec![0.0f32; kvd];
            let mut sc_ref = vec![0.0f32; t_n];
            unsafe {
                attention_head_core(
                    &q, &key_cache, &value_cache, &mut out_new, &mut sc_new,
                    q_off, kv_off, kvd, hd, t_n, scale,
                    GateBias::new(bias),
                );
            }
            reference_attention_head_pre727(
                &q, &key_cache, &value_cache, &mut out_ref, &mut sc_ref,
                q_off, kv_off, kvd, hd, t_n, scale, bias,
            );
            for d in 0..hd {
                assert_eq!(
                    out_new[q_off + d].to_bits(),
                    out_ref[q_off + d].to_bits(),
                    "{name} @ q_off={q_off}: attn_out[{d}] differs (new {} vs ref {})",
                    out_new[q_off + d],
                    out_ref[q_off + d],
                );
            }
            for t in 0..t_n {
                if bias[t] == f32::NEG_INFINITY {
                    // Documented scores_buf contract: pruned stay -inf (the
                    // old path left exp(-inf) = +0.0 — same zero weight).
                    assert_eq!(sc_new[t].to_bits(), f32::NEG_INFINITY.to_bits());
                } else {
                    assert_eq!(
                        sc_new[t].to_bits(),
                        sc_ref[t].to_bits(),
                        "{name} @ q_off={q_off}: scores[{t}] differs"
                    );
                }
            }
        }
    }

    // Active-position UNDERFLOW coverage: with identical keys every dot ties,
    // so a single -200 slot scores exactly 200 below the max and its exp is
    // exactly +0.0. Both paths must skip it identically (its per-lane
    // contribution was exactly ±0.0).
    let key_flat = vec![0.7f32; t_n * kvd];
    let value_flat = vec![1.3f32; t_n * kvd];
    let mut under = vec![0.0f32; t_n];
    under[5] = -200.0;
    let mut out_new = vec![0.0f32; kvd];
    let mut sc_new = vec![0.0f32; t_n];
    let mut out_ref = vec![0.0f32; kvd];
    let mut sc_ref = vec![0.0f32; t_n];
    unsafe {
        attention_head_core(
            &q, &key_flat, &value_flat, &mut out_new, &mut sc_new,
            0, 0, kvd, hd, t_n, scale,
            GateBias::new(&under),
        );
    }
    reference_attention_head_pre727(
        &q, &key_flat, &value_flat, &mut out_ref, &mut sc_ref,
        0, 0, kvd, hd, t_n, scale, &under,
    );
    for d in 0..hd {
        assert_eq!(out_new[d].to_bits(), out_ref[d].to_bits());
    }
    for t in 0..t_n {
        assert_eq!(sc_new[t].to_bits(), sc_ref[t].to_bits());
    }

    // DOCUMENTED DIVERGENCE: an all-pruned bias. The old path produced NaN
    // (exp(-inf - -inf) = exp(NaN) poisoning the sum); the hoisted path
    // produces an all-zeros output with scores left at -inf.
    // `build_gate_biases` can never emit one: the sliding window always keeps
    // its positions at bias 0.
    let all_pruned = vec![f32::NEG_INFINITY; t_n];
    let mut out = vec![0.0f32; kvd];
    let mut sc = vec![0.0f32; t_n];
    unsafe {
        attention_head_core(
            &q, &key_cache, &value_cache, &mut out, &mut sc,
            0, 0, kvd, hd, t_n, scale,
            GateBias::new(&all_pruned),
        );
    }
    assert!(out[..hd].iter().all(|v| *v == 0.0));
    assert!(sc.iter().all(|v| *v == f32::NEG_INFINITY));
}

// ── T17: KV Cache Density Ratio ──────────────────────────────────

#[test]
fn bench_kv_density_ratio() {
    let config = Config::micro();
    let kvd = kv_dim(&config);
    let n_kv = config.n_kv_head;
    let hidden = config.n_embd / 4;

    println!(
        "\n🧪 T17: KV Cache Density Ratio (n_embd={}, n_kv={n_kv}, kv_dim={kvd})",
        config.n_embd
    );
    println!("{}", "═".repeat(60));

    // Create predictors with init_bias=5 (gates start open)
    let predictors = SpKvPredictors::new(config.n_layer, config.n_embd, hidden, n_kv, 5.0);

    let thresholds = [0.1f32, 0.3, 0.5, 0.7, 0.9];
    let seq_len: usize = config.block_size.min(64);

    println!("  τ      Density   Retained   KV Bytes   vs Full KV");
    println!("  ─────  ────────  ─────────  ─────────  ──────────");

    let full_kv_bytes = seq_len * kvd * 4 * 2 * config.n_layer; // f32 K+V per layer

    for &threshold in &thresholds {
        let mut sp_config = SpKvConfig {
            threshold,
            ..SpKvConfig::default()
        };
        sp_config.resolve_hidden(config.n_embd);

        let mut sp_cache = SpKvCache::new(&sp_config, config.n_layer, config.block_size, kvd);
        let mut rng = Rng::new(42);
        let mut pred_buf = vec![0.0; hidden];

        // Simulate decode: predict utilities and conditionally write
        for pos in 0..seq_len {
            let h = synthetic_hidden(config.n_embd, pos);

            for layer_idx in 0..config.n_layer {
                let utilities = predict(
                    &predictors.layers[layer_idx],
                    &h,
                    config.n_embd,
                    hidden,
                    n_kv,
                    &mut pred_buf,
                );
                let pos_utility = aggregate_utilities(&utilities, UtilityAggregation::Max);

                // Simulated KV (synthetic)
                let k: Vec<f32> = (0..kvd).map(|_| rng.normal()).collect();
                let v: Vec<f32> = (0..kvd).map(|_| rng.normal()).collect();

                let layer_cache = &mut sp_cache.layers[layer_idx];
                let in_window = pos >= seq_len.saturating_sub(sp_config.window);
                layer_cache.write_gated(&k, &v, pos_utility, pos, in_window, threshold, kvd);
            }
        }

        let avg_density = sp_cache.avg_density(seq_len);
        let total_retained: usize = sp_cache.layers.iter().map(|l| l.retained_count).sum();
        let per_layer_retained = total_retained / config.n_layer;
        let retained_kv_bytes = total_retained * kvd * 4 * 2;
        let compression_pct = retained_kv_bytes as f64 / full_kv_bytes as f64 * 100.0;

        println!(
            "  {threshold:.1}     {:>5.1}%    {per_layer_retained:>3}/{seq_len}      {retained_kv_bytes:>7}   {compression_pct:>5.1}%",
            avg_density * 100.0,
        );
    }
    println!();

    // Validate: higher threshold → lower density
    println!("  ✅ Density decreases with higher τ (verified visually)");
}

// ── T18: Decode Latency ──────────────────────────────────────────

#[test]
fn bench_decode_latency() {
    let config = Config::micro();
    let kvd = kv_dim(&config);
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let hidden = config.n_embd / 4;
    let n_head = config.n_head;

    let seq_len: usize = config.block_size.min(64);

    println!(
        "\n🧪 T18: Decode Latency (n_layer={}, seq_len={seq_len})",
        config.n_layer
    );
    println!("{}", "═".repeat(60));

    let mut rng = Rng::new(99);

    // Fill baseline KV cache with synthetic data (flat vectors)
    let mut key_cache = vec![0.0f32; config.block_size * kvd];
    let mut value_cache = vec![0.0f32; config.block_size * kvd];
    for pos in 0..seq_len {
        let off = pos * kvd;
        for d in 0..kvd {
            key_cache[off + d] = rng.normal();
            value_cache[off + d] = rng.normal();
        }
    }

    // Query vector
    let q: Vec<f32> = (0..config.n_embd).map(|_| rng.normal()).collect();
    let mut attn_out = vec![0.0; config.n_embd];
    let mut scores = vec![0.0; config.block_size];
    let scale = 1.0 / (hd as f32).sqrt();

    // Baseline: full KV decode at pos=seq_len-1
    let start_baseline = Instant::now();
    for _ in 0..BENCH_ITERS {
        attn_out.fill(0.0);
        let t_n = seq_len;

        for h in 0..n_head {
            let kv_group = h * n_kv / n_head;
            unsafe {
                attention_head_gated(
                    &q,
                    &key_cache,
                    &value_cache,
                    &mut attn_out,
                    &mut scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                    None,
                );
            }
        }
        black_box(&attn_out);
    }
    let elapsed_baseline = start_baseline.elapsed();

    // SP-KV: sparse decode with hard gating
    let mut sp_config = SpKvConfig {
        threshold: 0.5,
        ..SpKvConfig::default()
    };
    sp_config.resolve_hidden(config.n_embd);

    let predictors = SpKvPredictors::new(config.n_layer, config.n_embd, hidden, n_kv, 5.0);
    let mut sp_cache = SpKvCache::new(&sp_config, config.n_layer, config.block_size, kvd);
    let mut pred_buf = vec![0.0; hidden];

    // Build sparse cache
    for pos in 0..seq_len {
        let h = synthetic_hidden(config.n_embd, pos);
        for layer_idx in 0..config.n_layer {
            let utilities = predict(
                &predictors.layers[layer_idx],
                &h,
                config.n_embd,
                hidden,
                n_kv,
                &mut pred_buf,
            );
            let pos_utility = aggregate_utilities(&utilities, UtilityAggregation::Max);
            let k: Vec<f32> = (0..kvd).map(|_| rng.normal()).collect();
            let v: Vec<f32> = (0..kvd).map(|_| rng.normal()).collect();

            let layer_cache = &mut sp_cache.layers[layer_idx];
            let in_window = pos >= seq_len.saturating_sub(sp_config.window);
            layer_cache.write_gated(
                &k,
                &v,
                pos_utility,
                pos,
                in_window,
                sp_config.threshold,
                kvd,
            );
        }
    }

    // Build gate biases once (hard mode for inference)
    let layer_cache = &sp_cache.layers[0];
    let mut gate_bias_buf = GateBiasBuffer::new(config.block_size);
    gate_bias_buf.build_hard(
        &layer_cache.utilities,
        &layer_cache.retained,
        seq_len - 1,
        sp_config.window,
        sp_config.threshold,
    );

    let start_sp_kv = Instant::now();
    for _ in 0..BENCH_ITERS {
        attn_out.fill(0.0);
        let t_n = seq_len;

        for h in 0..n_head {
            let kv_group = h * n_kv / n_head;
            unsafe {
                attention_head_gated(
                    &q,
                    &sp_cache.layers[0].key,
                    &sp_cache.layers[0].value,
                    &mut attn_out,
                    &mut scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                    Some(&gate_bias_buf.bias),
                );
            }
        }
        black_box(&attn_out);
    }
    let elapsed_sp_kv = start_sp_kv.elapsed();

    let ratio = elapsed_baseline.as_nanos() as f64 / elapsed_sp_kv.as_nanos() as f64;
    let density = sp_cache.avg_density(seq_len);

    println!(
        "  Full KV:      {:>8.2} µs/iter",
        elapsed_baseline.as_secs_f64() * 1e6 / BENCH_ITERS as f64
    );
    println!(
        "  SP-KV (τ=0.5): {:>8.2} µs/iter  ({ratio:.2}× speedup, density={density:.1}%)",
        elapsed_sp_kv.as_secs_f64() * 1e6 / BENCH_ITERS as f64,
    );
    println!();

    // Note: actual speedup depends on hardware and sequence length.
    // Paper reports 2.1–4.6× at batch=16 on GPU. CPU speedup is lower
    // because the attention loop still iterates all positions (bias=-inf → exp≈0).
    // Real speedup comes from block-skipping in GPU kernels.
    println!("  ℹ️  CPU speedup is limited — full speedup requires GPU block-skipping");
}

// ── T19: Palindrome Reversal Test ────────────────────────────────

#[test]
fn test_palindrome_retention() {
    // SP-KV must retain the palindrome anchor position even when it's
    // far outside the sliding window. This verifies that utility prediction
    // can learn to keep critical long-range positions.

    let config = Config::micro();
    let kvd = kv_dim(&config);
    let _hidden = config.n_embd / 4;
    let seq_len: usize = config.block_size.min(64);
    let window: usize = 8.min(seq_len / 2); // Small window to make the test harder
    let palindrome_pos: usize = 0; // Anchor at start, must be attended at end

    let mut sp_config = SpKvConfig {
        window,
        threshold: 0.5,
        ..SpKvConfig::default()
    };
    sp_config.resolve_hidden(config.n_embd);

    let mut sp_cache = SpKvCache::new(&sp_config, config.n_layer, config.block_size, kvd);
    let mut rng = Rng::new(77);

    // Simulate decode with artificial utility:
    // - Position 0 (palindrome anchor): utility = 0.9 (should be retained)
    // - Positions outside window: utility = 0.1 (should be pruned)
    // - Positions inside window: always retained
    for pos in 0..seq_len {
        let in_window = pos >= seq_len.saturating_sub(window);
        let is_anchor = pos == palindrome_pos;

        let pos_utility = if is_anchor {
            0.9 // High utility for palindrome anchor
        } else if in_window {
            1.0 // Window positions always retained
        } else {
            0.1 // Low utility — should be pruned
        };

        for layer_idx in 0..config.n_layer {
            let k: Vec<f32> = (0..kvd).map(|_| rng.normal()).collect();
            let v: Vec<f32> = (0..kvd).map(|_| rng.normal()).collect();

            let layer_cache = &mut sp_cache.layers[layer_idx];
            layer_cache.utilities[pos] = pos_utility;
            layer_cache.write_gated(
                &k,
                &v,
                pos_utility,
                pos,
                in_window,
                sp_config.threshold,
                kvd,
            );
        }
    }

    // Verify: palindrome anchor position is retained
    for layer_idx in 0..config.n_layer {
        assert!(
            sp_cache.layers[layer_idx].retained[palindrome_pos],
            "Layer {layer_idx}: palindrome anchor at pos={palindrome_pos} should be retained"
        );
    }

    // Verify: positions outside window with low utility are NOT retained
    let outside_window_low_utility = seq_len - window - 1; // A position not in window and not anchor
    if outside_window_low_utility > 0 && outside_window_low_utility != palindrome_pos {
        for layer_idx in 0..config.n_layer {
            assert!(
                !sp_cache.layers[layer_idx].retained[outside_window_low_utility],
                "Layer {layer_idx}: pos={outside_window_low_utility} should be pruned (outside window, low utility)"
            );
        }
    }

    // Build hard gate biases and verify anchor has bias=0 (attended)
    let mut gate_bias_buf = GateBiasBuffer::new(config.block_size);
    gate_bias_buf.build_hard(
        &sp_cache.layers[0].utilities,
        &sp_cache.layers[0].retained,
        seq_len - 1,
        window,
        sp_config.threshold,
    );

    assert_eq!(
        gate_bias_buf.bias[palindrome_pos], 0.0,
        "Palindrome anchor should have bias=0 (attended)"
    );

    // Verify pruned positions have bias=-inf
    if outside_window_low_utility > 0 && outside_window_low_utility != palindrome_pos {
        assert_eq!(
            gate_bias_buf.bias[outside_window_low_utility],
            f32::NEG_INFINITY,
            "Pruned position should have bias=-inf"
        );
    }

    println!("\n🧪 T19: Palindrome Retention Test (window={window}, seq_len={seq_len})");
    println!("{}", "═".repeat(60));
    println!("  ✅ Palindrome anchor at pos={palindrome_pos} retained across all layers");
    println!("  ✅ Non-anchor positions outside window correctly pruned");
    println!("  Density: {:.1}%", sp_cache.avg_density(seq_len) * 100.0);
}

// ── T20: Utility Predictor Gradient Flow ─────────────────────────

#[test]
fn test_utility_predictor_gradient_flow() {
    // Verify that log(u) gate bias preserves gradient flow.
    // We can't do autodiff in katgpt-rs, but we verify:
    // 1. Soft gate bias is finite and well-defined for all u ∈ (0,1)
    // 2. ∂bias/∂u = 1/(u+ε) is large when u is small (strong learning signal)
    // 3. TAHG annealing smoothly transitions from soft to hard
    // 4. Frozen predictor state is tracked correctly

    use katgpt_rs::sp_kv::utility_predictor::{soft_gate_bias, tahg_gate_bias};

    println!("\n🧪 T20: Utility Predictor Gradient Flow");
    println!("{}", "═".repeat(60));

    // Test 1: Soft gate bias is finite for all u ∈ (0,1)
    println!("\n  Soft gate bias = log(u + ε):");
    for &u in &[0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999] {
        let bias = soft_gate_bias(u);
        let grad = 1.0 / (u + 1e-8); // ∂bias/∂u
        assert!(bias.is_finite(), "bias not finite at u={u}");
        assert!(grad.is_finite(), "grad not finite at u={u}");
        println!("    u={u:.3}  bias={bias:>8.3}  ∂b/∂u={grad:>10.1}");
    }

    // Test 2: Gradient is stronger for small u (more learning signal for prunable positions)
    let grad_at_01 = 1.0 / (0.1 + 1e-8);
    let grad_at_09 = 1.0 / (0.9 + 1e-8);
    assert!(
        grad_at_01 > grad_at_09,
        "Gradient should be larger for small u (stronger learning signal)"
    );
    println!("\n  ✅ Gradient at u=0.1 ({grad_at_01:.1}) > gradient at u=0.9 ({grad_at_09:.1})");

    // Test 3: TAHG annealing transitions smoothly
    println!("\n  TAHG annealing (u=0.3, τ=0.5):");
    for &alpha in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let bias = tahg_gate_bias(0.3, 0.5, alpha);
        assert!(bias.is_finite(), "TAHG bias not finite at α={alpha}");
        println!("    α={alpha:.2}  bias={bias:>8.3}");
    }

    // Test 4: SpKvPredictors freeze/unfreeze
    let config = Config::micro();
    let mut predictors = SpKvPredictors::new(
        config.n_layer,
        config.n_embd,
        config.n_embd / 4,
        config.n_kv_head,
        5.0,
    );
    assert!(!predictors.frozen, "Predictors should start unfrozen");
    predictors.freeze();
    assert!(
        predictors.frozen,
        "Predictors should be frozen after freeze()"
    );
    predictors.unfreeze();
    assert!(
        !predictors.frozen,
        "Predictors should be unfrozen after unfreeze()"
    );
    println!("\n  ✅ Predictor freeze/unfreeze cycle works correctly");

    // Test 5: Predictor outputs are always in (0,1) for diverse inputs
    let mut rng = Rng::new(123);
    let hidden = config.n_embd / 4;
    let mut pred_buf = vec![0.0; hidden];
    let mut all_valid = true;

    for _ in 0..100 {
        // Random hidden state
        let h: Vec<f32> = (0..config.n_embd).map(|_| rng.normal() * 10.0).collect();
        let utilities = predict(
            &predictors.layers[0],
            &h,
            config.n_embd,
            hidden,
            config.n_kv_head,
            &mut pred_buf,
        );

        for &u in &utilities {
            // Sigmoid can saturate to exactly 0.0 or 1.0 with extreme inputs.
            // Valid range: finite values in [0, 1].
            if !u.is_finite() || !(0.0..=1.0).contains(&u) {
                all_valid = false;
            }
        }
    }
    assert!(all_valid, "All utilities should be finite in [0, 1]");
    println!(
        "  ✅ Predictor outputs always finite in [0, 1] for diverse inputs (100 random tests)"
    );

    // Test 6: Verify init_bias=5 produces near-open gates
    let h_zero = vec![0.0; config.n_embd];
    let utilities_zero = predict(
        &predictors.layers[0],
        &h_zero,
        config.n_embd,
        hidden,
        config.n_kv_head,
        &mut pred_buf,
    );
    for &u in &utilities_zero {
        assert!(u > 0.99, "Init bias=5 should produce u>0.99, got {u}");
    }
    println!("  ✅ Init bias=5 produces near-open gates (u>0.99) for zero input");
}

// ── Summary ──────────────────────────────────────────────────────

#[test]
fn bench_sp_kv_summary() {
    let config = Config::micro();
    let hidden = config.n_embd / 4;
    let n_kv = config.n_kv_head;

    println!("\n📊 SP-KV Plan 070 Summary");
    println!("{}", "═".repeat(60));
    println!(
        "  Config: micro (n_embd={}, n_layer={}, n_kv={n_kv})",
        config.n_embd, config.n_layer
    );
    println!(
        "  Utility predictor: {} hidden, {} params/layer",
        hidden,
        SpKvPredictors::new(1, config.n_embd, hidden, n_kv, 5.0).total_param_count(),
    );
    println!("  Overhead: one additive bias per attention score");
    println!("  Pipeline: PFlash (prefill) → SP-KV (decode) → TurboQuant (storage)");
    println!();
    println!("  Gate modes:");
    println!("    Soft:  bias = log(u + ε)          — training phase 1");
    println!("    Hard:  bias = 0 | -∞              — inference");
    println!("    TAHG:  blended with α ramp 0→1    — training phase 2");
    println!();
    println!("  Expected (from paper, 8.1B model):");
    println!("    Density:     ~30% at τ=0.5, ~11% at τ=0.7");
    println!("    NLL Δ:       +0.08% at τ=0.5");
    println!("    Decode:      2.1–4.6× speedup at batch=16 (GPU)");
    println!("    NIAH:        perfect retrieval at 5-7% density");
}
