//! Bench 871 — strict IEEE ordered-reduction dot vs Rust 1.98 `algebraic_*`
//! reassociated dot (Issue 871): does per-op reassociation permission earn a
//! lane in this repo?
//!
//! Three measurements, everything printed, **no perf bar asserted** — this is
//! the record bench for the Issue 871 verdict; promotion is a separate owner
//! act:
//!
//! 1. **G2 A/B** — interleaved median-of-ratios (`tests/common/ab_timing.rs`,
//!    the Issue 723 Class A treatment) at d ∈ {16, 64, 256, 1024, 4096}.
//!    `ratio = b/a` with **a = strict** (the `dot_8wide` shape verbatim: zip
//!    single-accumulator, strict IEEE) and **b = algebraic**
//!    (`algebraic_add`/`algebraic_mul` on the same loop). Median < 1.0 ⇒
//!    algebraic is FASTER. Each round's per-round range is printed beside the
//!    median (a median inside a 0.9–1.1 band and one inside 0.3–3.0 are not
//!    the same claim).
//! 2. **G1 numerics** — max ulp delta strict-vs-algebraic, plus the relative
//!    error of BOTH against a strict f64 reference. Reassociation can be MORE
//!    accurate (multiple accumulators reduce rounding), so both sides are
//!    reported, never pooled.
//! 3. **G1 argmax retention** — 64-key logit fixtures (d = 64) with a forced
//!    near-tie pair on half the trials; counts argmax flips strict-vs-
//!    algebraic. The Issue-750-T3 per-family retention hazard in miniature:
//!    aggregate agreement can look flat while near-tie argmax flips.
//!
//! Deterministic end to end: a fixed local LCG (the global_rng_gate class —
//! no unseeded global RNG), fixed buffers, no runtime RNG. The ALGEBRAIC
//! numbers are deterministic per binary but NOT across target-features or
//! compiler versions — that is the property under test, not a bug.
//!
//! The run prints its compile-time target features so each invocation
//! self-documents which ISA arm executed:
//!
//! ```sh
//! # baseline (ordinary x86_64 build — what every consumer gets)
//! cargo run --release --bench bench_871_algebraic_dot_ab
//! # AVX2+FMA (x86-64-v3 class)
//! RUSTFLAGS="-C target-feature=+avx2,+fma" \
//!   cargo run --release --target-dir /tmp/alg871_avx2 --bench bench_871_algebraic_dot_ab
//! ```

use std::hint::black_box;

use katgpt_types::approx::ulp_distance;

#[path = "../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;

/// Rounds per dim for the interleaved A/B. 13 medians discard preemption
/// spikes; the printed per-round range says whether the box was quiet.
const ROUNDS: usize = 13;
/// ~2M FMA per arm per round, clamped — keeps every chunk comfortably above
/// timer resolution at every dim.
const WORK_PER_CHUNK: usize = 2_000_000;
/// Buffer pairs rotated by the iteration index (constant input is what lets
/// the optimiser hoist the arm out of the loop — Issue 723 Class A2).
const PAIRS: usize = 8;

/// Numerical-Recipes 64-bit LCG — deterministic, zero deps, zero global RNG.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Finite f32 with mixed magnitudes (exponent 2⁻⁷..2⁸, random sign and
    /// mantissa) so the dots see real cancellation, not uniform-scale sums.
    fn next_f32(&mut self) -> f32 {
        let r = self.next_u32();
        let r2 = self.next_u32();
        let exp = 120 + (r % 16);
        f32::from_bits((exp << 23) | (r2 & 0x007F_FFFF) | (r2 & 0x8000_0000))
    }
}

/// The strict baseline — the `dot_8wide` shape verbatim
/// (`katgpt-attn-match/src/score_matrix_simd.rs`): zip single-accumulator,
/// strict IEEE. Under strict FP this ordered reduction preserves the scalar
/// left-assoc order regardless of SIMD width — bit-identical cross-arch.
#[inline]
fn dot_strict(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
    }
    dot
}

/// The candidate — identical loop, per-op reassociation + contraction
/// permission (`reassoc contract arcp nsz`, no `nnan`/`ninf`, so no poison).
/// LLVM may now use multiple independent vector accumulators and a tree
/// horizontal reduce — at the cost of cross-arch/cross-build bit-equality.
#[inline]
fn dot_algebraic(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot = dot.algebraic_add(x.algebraic_mul(y));
    }
    dot
}

/// Strict f64 reference, same order — the ground truth both f32 sides are
/// compared against.
fn dot_strict_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += f64::from(x) * f64::from(y);
    }
    dot
}

/// Integer bit-distance — the shared ULP metric (`katgpt_types::approx`,
/// Issue 944), with the signed-zero and non-finite laws stated there. The
/// old local copy's "same-sign finite" precondition is subsumed: every gate
/// domain here is finite dot outputs, and the metric now refuses (via
/// `expect`) instead of measuring garbage if that ever changes.
fn ulp_delta(x: f32, y: f32) -> u32 {
    ulp_distance(x, y).expect("finite dot outputs")
}

fn rel_err(x: f32, reference: f64) -> f64 {
    let scale = reference.abs().max(1e-30);
    (f64::from(x) - reference).abs() / scale
}

fn argmax_idx(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// ulp gap between the top-2 entries (the near-tie detector).
fn top2_ulp_gap(v: &[f32]) -> u32 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    ulp_delta(sorted[n - 1], sorted[n - 2])
}

const DIMS: [usize; 5] = [16, 64, 256, 1024, 4096];

fn make_pair(d: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Lcg(seed);
    let a: Vec<Vec<f32>> = (0..PAIRS)
        .map(|_| (0..d).map(|_| rng.next_f32()).collect())
        .collect();
    let b: Vec<Vec<f32>> = (0..PAIRS)
        .map(|_| (0..d).map(|_| rng.next_f32()).collect())
        .collect();
    (a, b)
}

/// G2: interleaved strict-vs-algebraic A/B at one dim. Returns the median
/// ratio (b/a, < 1.0 ⇒ algebraic faster) for the verdict hint.
fn run_dim(d: usize) -> f64 {
    let (a, b) = make_pair(d, 0x0871_0000 + d as u64);
    let iters = (WORK_PER_CHUNK / d).clamp(128, 32_768);

    let mut sink_strict = 0.0f32;
    let mut sink_alg = 0.0f32;
    let ratio = ab_median_ratio(
        ROUNDS,
        iters,
        96,
        |i| {
            let p = i % PAIRS;
            sink_strict += dot_strict(black_box(&a[p]), black_box(&b[p]));
        },
        |i| {
            let p = i % PAIRS;
            sink_alg += dot_algebraic(black_box(&a[p]), black_box(&b[p]));
        },
    );

    println!("\nd = {d:5}  ({iters} iters/arm/round)");
    ratio.report("strict(a) vs algebraic(b)");
    println!("         ratio b/a: < 1.0 ⇒ algebraic FASTER; overhead_pct < 0 ⇒ same");

    // G1 numerics for this dim: ulp delta + both sides vs the f64 reference.
    let mut max_ulp = 0u32;
    let mut max_rel_strict = 0.0f64;
    let mut max_rel_alg = 0.0f64;
    for p in 0..PAIRS {
        let s = dot_strict(&a[p], &b[p]);
        let al = dot_algebraic(&a[p], &b[p]);
        let reference = dot_strict_f64(&a[p], &b[p]);
        max_ulp = max_ulp.max(ulp_delta(s, al));
        max_rel_strict = max_rel_strict.max(rel_err(s, reference));
        max_rel_alg = max_rel_alg.max(rel_err(al, reference));
    }
    println!(
        "   G1: ulp(strict↔algebraic) ≤ {max_ulp}; rel-err vs f64 — strict {max_rel_strict:.3e}, algebraic {max_rel_alg:.3e}"
    );

    black_box((&sink_strict, &sink_alg));
    ratio.median
}

/// G1 retention: near-tie argmax flips. Returns (trials, near_tie_trials,
/// flips_total, flips_inside_near_ties).
fn run_argmax_retention() -> (usize, usize, usize, usize) {
    const KEYS: usize = 64;
    const D: usize = 64;
    const TRIALS: usize = 1024;

    let mut rng = Lcg(0x0871_07E7);
    let mut near_ties = 0;
    let mut flips = 0;
    let mut flips_near = 0;

    for t in 0..TRIALS {
        let q: Vec<f32> = (0..D).map(|_| rng.next_f32()).collect();
        let mut keys: Vec<Vec<f32>> = (0..KEYS)
            .map(|_| (0..D).map(|_| rng.next_f32()).collect())
            .collect();

        // Half the trials carry a FORCED near-tie: key 1 is key 0 with one
        // weight nudged a single ulp — the logits land within a few ulp.
        if t % 2 == 0 {
            keys[1] = keys[0].clone();
            let bit = keys[1][7].to_bits().wrapping_add(1);
            keys[1][7] = f32::from_bits(bit);
        }

        let logits_strict: Vec<f32> = keys.iter().map(|k| dot_strict(&q, k)).collect();
        let logits_alg: Vec<f32> = keys.iter().map(|k| dot_algebraic(&q, k)).collect();

        let gap = top2_ulp_gap(&logits_strict);
        let near = gap <= 8;
        if near {
            near_ties += 1;
        }
        let (am_s, am_a) = (argmax_idx(&logits_strict), argmax_idx(&logits_alg));
        if am_s != am_a {
            flips += 1;
            if near {
                flips_near += 1;
            }
        }
    }

    (TRIALS, near_ties, flips, flips_near)
}

/// G1 ill-conditioned arm (Issue 871 T4 (a′) landing condition — the
/// f978a20b class): near-zero dots and rank-deficient structure, where the
/// mixed-magnitude LCG "algebraic is MORE accurate" result does not
/// automatically transfer. Printed, NEVER asserted — recorded either way;
/// consumer selection reads this before adopting anywhere.
///
/// Two classes, 512 trials each, d = 512:
/// - **ortho** (catastrophic cancellation): `b` orthogonalized against `a`
///   in f64, so the true dot of the STORED f32 values sits at rounding
///   scale. The pivot-flip analogue: count SIGN disagreements of each f32
///   kernel vs the f64 truth.
/// - **dup** (rank-deficient, the Cholesky duplicate-column shape): `a`
///   repeats 8-element blocks exactly, `b` alternates ±v in matching blocks
///   — exact-cancellation structure with large partial sums.
///
/// Metric: scale-relative error `|kernel − f64 truth| / (‖a‖·‖b‖)` (the
/// plain relative error degenerates when truth ≈ 0).
fn run_ill_conditioned() {
    const D: usize = 512;
    const TRIALS: usize = 512;

    /// Fixture factory: (rng, d) → (a, b) pair for one trial.
    type MkPair<'a> = &'a dyn Fn(&mut Lcg, usize) -> (Vec<f32>, Vec<f32>);

    let worst = |name: &str, mk: MkPair<'_>| {
        let mut max_s = 0.0f64;
        let mut max_g = 0.0f64;
        let mut sum_s = 0.0f64;
        let mut sum_g = 0.0f64;
        let mut sign_s = 0usize;
        let mut sign_g = 0usize;
        for t in 0..TRIALS {
            let mut rng = Lcg(0x0871_1CC0 + t as u64);
            let (a, b) = mk(&mut rng, D);
            let truth = dot_strict_f64(&a, &b);
            let s = dot_strict(&a, &b);
            let g = dot_algebraic(&a, &b);
            let na: f64 = a
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = b
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            let scale = (na * nb).max(1e-30);
            let es = (f64::from(s) - truth).abs() / scale;
            let eg = (f64::from(g) - truth).abs() / scale;
            max_s = max_s.max(es);
            max_g = max_g.max(eg);
            sum_s += es;
            sum_g += eg;
            // Sign disagreement vs the f64 truth — only meaningful when the
            // truth is small relative to scale (near-cancellation trials).
            if truth.abs() < 1e-6 * scale {
                if s != 0.0 && (s < 0.0) != (truth < 0.0) {
                    sign_s += 1;
                }
                if g != 0.0 && (g < 0.0) != (truth < 0.0) {
                    sign_g += 1;
                }
            }
        }
        println!(
            "  {name}: scale-rel err — strict mean {:.3e} max {:.3e} | algebraic mean {:.3e} max {:.3e} | sign-flips vs f64 truth: strict {sign_s}, algebraic {sign_g} (of {TRIALS} trials)",
            sum_s / TRIALS as f64,
            max_s,
            sum_g / TRIALS as f64,
            max_g,
        );
    };

    println!(
        "\nG1 ill-conditioned arm (d={D}, {TRIALS} trials/class; recorded either way, no bar):"
    );
    worst("ortho (near-zero dot)", &|rng, d| {
        let a: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let y: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let xy = dot_strict_f64(&a, &y);
        let xx = dot_strict_f64(&a, &a);
        let b: Vec<f32> = y
            .iter()
            .zip(&a)
            .map(|(&yy, &xx_)| (f64::from(yy) - (xy / xx) * f64::from(xx_)) as f32)
            .collect();
        (a, b)
    });
    worst("dup   (rank-deficient)", &|rng, d| {
        let block: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();
        let v: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();
        let a: Vec<f32> = (0..d).map(|i| block[i % 8]).collect();
        let b: Vec<f32> = (0..d)
            .map(|i| {
                if (i / 8) % 2 == 0 {
                    v[i % 8]
                } else {
                    -v[i % 8]
                }
            })
            .collect();
        (a, b)
    });
}

fn main() {
    println!("bench_871 — strict IEEE ordered-reduction dot vs Rust 1.98 algebraic_* (Issue 871)");
    println!(
        "compile-time target features: avx2={} fma={} avx512f={} (self-documenting ISA arm)",
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "fma"),
        cfg!(target_feature = "avx512f"),
    );

    let mut medians = Vec::with_capacity(DIMS.len());
    for &d in &DIMS {
        medians.push(run_dim(d));
    }

    let faster = medians.iter().filter(|&&m| m < 0.9).count();
    let slower = medians.iter().filter(|&&m| m > 1.1).count();
    println!(
        "\nG2 summary: {} of {} dims ≥1.10× faster for algebraic, {slower} slower; medians: {}",
        medians.len() - slower,
        medians.len(),
        medians
            .iter()
            .map(|m| format!("{m:.3}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    // The 0.9/1.1 band reading, printed not asserted — the Issue 871 record
    // carries the verdict; a promotion bar is a separate act.
    if faster > 0 {
        println!(
            "VERDICT HINT: algebraic clears ≥1.10× on {faster} dim(s) — promotion candidate (needs retention walk + owner call)"
        );
    } else {
        println!(
            "VERDICT HINT: no dim clears ≥1.10× — negative-leaning; strict bit-identity stays"
        );
    }

    let (trials, near, flips, flips_near) = run_argmax_retention();
    println!(
        "\nG1 retention: {trials} trials × {near_} near-tie (top-2 gap ≤ 8 ulp): \
         {flips} argmax flip(s) total, {flips_near} inside near-ties",
        near_ = near,
    );
    if flips_near > 0 {
        println!(
            "  ⚠ reassociation flips near-tie argmax — the Issue-750-T3 retention walk is MANDATORY before any logits-lane adoption"
        );
    } else if flips > 0 {
        println!(
            "  flips occurred only outside near-ties (data-scale divergences — check ulp table)"
        );
    } else {
        println!(
            "  zero flips on this fixture (64-key, forced near-ties) — retention clean at this scale"
        );
    }

    run_ill_conditioned();
}
