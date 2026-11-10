//! Bench 847 — Issue 847: are `simd_lut_dequant`'s and `bf16_convert`'s AVX2
//! arms REACHED on an ordinary x86_64 build?
//!
//! ```text
//! # (1) the default build — the configuration everybody ships
//! cargo test -p katgpt-core --release --features bf16_simd \
//!     --test bench_847_avx2_arm_reachability -- --nocapture
//!
//! # (2) the same thing with the arm's compile-time predicate satisfied
//! RUSTFLAGS="-C target-feature=+avx2" cargo test -p katgpt-core --release \
//!     --features simd_lut_dequant,bf16_simd \
//!     --test bench_847_avx2_arm_reachability -- --nocapture
//! ```
//!
//! # The instrument, and why its DEFAULT reading is the proof
//!
//! Each dispatcher (`dequant_via_lut`, `dequant_dot_via_lut`,
//! `bf16_bits_to_f32_into`) selects an AVX2 kernel behind
//! `#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]` and otherwise
//! falls through to a scalar one. `target_feature = "avx2"` is a **compile-time**
//! predicate and is OFF by default on x86_64, so on an ordinary build the AVX2
//! arm compiles to nothing and the dispatcher calls the scalar path.
//!
//! This target compares each **dispatcher** against its own **public scalar**
//! function, and prints BOTH absolute figures. The reading rule is not the
//! ratio:
//!
//! ⛔ **Compare a row's ABSOLUTE ns across the two builds; the within-build
//! ratio is confounded and the first run of this target was read wrongly
//! because of it.** Under `+avx2` the *scalar* arm is compiled with AVX2
//! available too, so LLVM may autovectorise it — measured: `bf16` scalar goes
//! 200 ns → 97 ns at n=4096 while the intrinsic arm goes 201 → 169. Both arms
//! move between builds, so the ratio column mixes two effects. The
//! **dispatcher's own** absolute time, default vs `+avx2`, is what isolates
//! "what the compile-time gate costs".
//!
//! ⚠ And a within-build ratio near 1.00 is evidence only when the two arms are
//! otherwise the same code. `dequant_via_lut` reads 1.5–1.7× on a build where
//! its AVX2 arm cannot exist — a generic `#[inline]` dispatcher gets its
//! `shift`/`mask` constants propagated into the inlined scalar body while a
//! direct out-of-line call to the same scalar function does not. That row is
//! confounded in BOTH builds and only its cross-build delta means anything.
//! `dequant_dot_via_lut` and `bf16_bits_to_f32_into` read 1.00–1.01, which is
//! the clean form of the evidence.
//!
//! # It is a GATE now, and it bars exactly ONE row
//!
//! The defect is repaired (both dispatchers take a runtime `simd_level()` probe),
//! so the measurement above is history and what a run can still assert is the
//! property the repair established: **on x86_64 the dispatcher must be
//! meaningfully faster than its own scalar reference.** A compile-time gate
//! re-introduced anywhere in that dispatch collapses it back to ~1.00 and reds.
//!
//! ⛔ **Only `dequant_dot_via_lut` is barred, and the reason is the confound
//! above.** That row measured **1.00–1.01** while its AVX2 arm did not compile
//! and **5.0×** after the repair, so a bar between those is a real wall.
//! `dequant_via_lut` read **1.5–1.7× even while broken** (const-propagation into
//! an inlined scalar body), so the same bar would have passed the defect — a
//! guard that greens on the thing it guards against is worse than none.
//! `bf16_bits_to_f32_into` is **deliberately unbarred**: its arm is still
//! compile-gated and its intrinsics lose to LLVM's autovectorised scalar under
//! `+avx2` (169 vs 97 ns), which is Issue 847 T2 and not a reachability
//! question.
//!
//! The bar is 1.5× against a measured 5.0×: over 3× of slack, because this is a
//! REACHABILITY assertion and not a perf budget.
//!
#![cfg(feature = "bf16_simd")]

use katgpt_core::bf16_convert::{bf16_bits_to_f32_into, bf16_bits_to_f32_scalar_into};
use katgpt_core::simd_lut_dequant::{
    QuantLut, UInt4Lut, dequant_dot_via_lut, dequant_dot_via_lut_scalar, dequant_via_lut,
    dequant_via_lut_scalar,
};
use std::hint::black_box;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;
use ab_timing::ab_median_ratio;

/// Element counts. 256 is a GGUF Q4_K super-block; 4096 and 32768 are row-ish
/// and tensor-ish, where a dead vector kernel costs the most.
const NS: [(usize, usize); 3] = [(256, 3000), (4096, 400), (32768, 60)];

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 40) as u8
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        2.0 * (((self.0 >> 33) as f32) / (1u64 << 31) as f32) - 1.0
    }
}

fn banner() {
    println!("\n   Issue 847 — is the AVX2 arm REACHED on this build?");
    println!(
        "   arch = {}, cfg!(target_feature=\"avx2\") = {}",
        std::env::consts::ARCH,
        cfg!(target_feature = "avx2")
    );
    if cfg!(target_arch = "x86_64") && !cfg!(target_feature = "avx2") {
        println!(
            "   ⇒ every `cfg(all(x86_64, target_feature=\"avx2\"))` arm compiled to NOTHING \
             here. A ratio of ~1.00 below means the dispatcher IS its scalar fallback."
        );
    }
    println!(
        "   {:>8}  {:>26}  {:>14}  {:>14}  {:>11}",
        "n", "kernel", "dispatch ns", "scalar ns", "scalar/disp"
    );
}

/// Print one row and assert instrument health. `ratio` is scalar/dispatcher, so
/// `> 1` means the dispatcher is faster, i.e. the vector arm is live.
fn row(n: usize, label: &str, disp_ns: f64, scalar_ns: f64, ratio: f64, survived: (usize, usize)) {
    println!("   {n:>8}  {label:>26}  {disp_ns:>14.0}  {scalar_ns:>14.0}  {ratio:>11.2}");
    // A grep-able row so the two builds can be diffed mechanically rather than
    // by eye — the cross-build delta IS the measurement (see the module doc).
    println!(
        "   ROW847 {label} n={n} disp_ns={disp_ns:.0} scalar_ns={scalar_ns:.0} avx2_cfg={}",
        cfg!(target_feature = "avx2")
    );
    assert!(
        ratio.is_finite() && ratio > 0.0,
        "instrument FAIL for {label} at n={n}: ratio {ratio} is not a measurement"
    );
    assert_eq!(
        survived.0, survived.1,
        "instrument FAIL for {label} at n={n}: {} of {} rounds survived",
        survived.0, survived.1
    );
}

#[test]
fn avx2_arms_reachability_and_price() {
    banner();
    let mut live = Vec::with_capacity(NS.len());

    for &(n, iters) in &NS {
        let mut rng = Lcg::new(0x0846_0000 + n as u64);
        let codes: Vec<u8> = (0..n).map(|_| rng.next_u8()).collect();
        let lut = UInt4Lut::build(0.5, 8.0);
        // NOT `.to_vec()`: `as_f32_slice()` coerces a `[f32; 16]`, so LLVM knows
        // the length is 16 and folds the scalar kernel's bounds checks. A heap
        // `Vec` hides that, and the scalar arm then measures 1.5x slower for a
        // reason that has nothing to do with AVX2 — which is exactly what the
        // first run of this target reported, and what its own fail-safe branch
        // caught. Same provenance on both arms or the comparison is not one.
        let lut_slice = lut.as_f32_slice();
        let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let bits: Vec<u16> = (0..n)
            .map(|_| ((rng.next_u8() as u16) << 8) | 0x3f)
            .collect();

        // ── dequant_via_lut ────────────────────────────────────────────────
        let mut out_d = vec![0.0f32; n];
        let mut out_s = vec![0.0f32; n];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out_d);
        dequant_via_lut_scalar(&codes, lut_slice, 0, 0x0F, &mut out_s);
        assert_eq!(
            out_d, out_s,
            "dequant_via_lut and its scalar reference must agree BIT-EXACTLY (n={n}) — \
             a LUT lookup is a table read, not an arithmetic reassociation"
        );
        let r = ab_median_ratio(
            11,
            iters,
            iters / 2,
            |_| {
                dequant_via_lut(black_box(&codes), black_box(&lut), 0, 0x0F, &mut out_d);
                black_box(out_d[0]);
            },
            |_| {
                dequant_via_lut_scalar(
                    black_box(&codes),
                    black_box(lut_slice),
                    0,
                    0x0F,
                    &mut out_s,
                );
                black_box(out_s[0]);
            },
        );
        row(
            n,
            "dequant_via_lut",
            r.a_ns_per_iter(),
            r.b_ns_per_iter(),
            r.median,
            (r.ratios.len(), r.rounds),
        );
        live.push(("dequant_via_lut", n, r.median));

        // ── dequant_dot_via_lut (the fused one) ───────────────────────────
        let d_disp = dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F);
        let d_scal = dequant_dot_via_lut_scalar(&codes, lut_slice, &x, 0, 0x0F);
        let scale = d_scal.abs().max(1.0);
        assert!(
            (d_disp - d_scal).abs() / scale <= 1e-5,
            "dequant_dot_via_lut disagrees with its scalar reference at n={n}: \
             {d_disp} vs {d_scal} (a fused dot DOES reassociate, hence a tolerance)"
        );
        let r = ab_median_ratio(
            11,
            iters,
            iters / 2,
            |_| {
                black_box(dequant_dot_via_lut(
                    black_box(&codes),
                    black_box(&lut),
                    black_box(&x),
                    0,
                    0x0F,
                ));
            },
            |_| {
                black_box(dequant_dot_via_lut_scalar(
                    black_box(&codes),
                    black_box(lut_slice),
                    black_box(&x),
                    0,
                    0x0F,
                ));
            },
        );
        row(
            n,
            "dequant_dot_via_lut",
            r.a_ns_per_iter(),
            r.b_ns_per_iter(),
            r.median,
            (r.ratios.len(), r.rounds),
        );
        live.push(("dequant_dot_via_lut", n, r.median));

        // ── bf16_bits_to_f32_into ─────────────────────────────────────────
        let mut bf_d = vec![0.0f32; n];
        let mut bf_s = vec![0.0f32; n];
        bf16_bits_to_f32_into(&bits, &mut bf_d);
        bf16_bits_to_f32_scalar_into(&bits, &mut bf_s);
        assert_eq!(
            bf_d, bf_s,
            "bf16 widening is LOSSLESS (u16 -> u32 << 16 -> f32), so the arms must be \
             bit-identical at n={n}"
        );
        let r = ab_median_ratio(
            11,
            iters,
            iters / 2,
            |_| {
                bf16_bits_to_f32_into(black_box(&bits), &mut bf_d);
                black_box(bf_d[0]);
            },
            |_| {
                bf16_bits_to_f32_scalar_into(black_box(&bits), &mut bf_s);
                black_box(bf_s[0]);
            },
        );
        row(
            n,
            "bf16_bits_to_f32_into",
            r.a_ns_per_iter(),
            r.b_ns_per_iter(),
            r.median,
            (r.ratios.len(), r.rounds),
        );
        live.push(("bf16_bits_to_f32_into", n, r.median));
    }

    // ── the verdict ───────────────────────────────────────────────────────
    //
    // NOT read off the ratio column in general (see the module doc: both arms
    // move between builds). The ONE row whose two arms are the same code, and
    // which measured 1.00 while broken, is the one that can carry a wall.
    println!();
    let barred: Vec<&(&str, usize, f64)> = live
        .iter()
        .filter(|(label, n, _)| *label == "dequant_dot_via_lut" && *n >= 4096)
        .collect();
    assert!(
        barred.len() >= 2,
        "instrument FAIL: the barred row is missing from the sweep — {} of \
         an expected 2+ (did NS or the label change?)",
        barred.len()
    );
    for (label, n, ratio) in &barred {
        if cfg!(target_arch = "x86_64") {
            assert!(
                *ratio >= 1.5,
                "GATE FAIL: {label} at n={n} measures only {ratio:.2}x its own \
                 scalar reference on x86_64. It measured 1.00-1.01x while its \
                 AVX2 arm was gated on a COMPILE-time `target_feature` and \
                 5.0x after Issue 847 replaced that with a runtime \
                 `simd_level()` probe — so a value this low means the vector \
                 arm is UNREACHABLE again on an ordinary build."
            );
        }
    }
    if cfg!(target_arch = "x86_64") {
        println!(
            "   PASS: the vector arms are REACHED on this build — \
             dequant_dot_via_lut at {:.2}x / {:.2}x its scalar reference \
             (bar 1.5x). Before Issue 847 this read 1.01x on a default build.",
            barred[0].2, barred[1].2
        );
    } else {
        println!(
            "   x86_64 bar NOT APPLICABLE on {} — the aarch64 arm was never \
             compile-feature-gated (NEON is implied by the arch), so this \
             class does not exist here and the rows above are reported, not \
             barred.",
            std::env::consts::ARCH
        );
    }
    println!(
        "   ⚠ bf16_bits_to_f32_into is UNREPAIRED and unbarred (Issue 847 T2): \
         its arm is still compile-gated, and under +avx2 its intrinsics LOSE \
         to the autovectorised scalar loop — a different question from \
         reachability.
"
    );
}

/// Issue 847 T2 — the THREE-arm question the issue refuses to answer by
/// reflex: is the hand-written AVX2 arm worth keeping at all?
///
/// The repair 847 applied to `simd_lut_dequant` was "runtime-probe the
/// intrinsics", and applying it here by reflex would be a ~1.2x win
/// (198 -> 169 ns at n=4096). But under `+avx2` the PLAIN SCALAR loop
/// measured **97 ns** against the intrinsics' 169 — LLVM beating the
/// intrinsics at their own job — and that arm is reachable on a DEFAULT
/// build through `#[target_feature(enable = ..)]`, whose whole purpose is to
/// compile a body for an ISA the build does not target. Three arms:
///
///   A  the shipped dispatcher        (compile-time cfg == the scalar path
///                                     on a default build)
///   B  the hand-written intrinsics   (visible only under +avx2, where the
///                                     dispatcher column BECOMES them)
///   C  the scalar body compiled WITH avx2 (`*_autovec`)
///
/// A REPORT: it bars nothing. A bar written before the sweep would be a bar
/// written from a hypothesis, which is `bench_843_ternary_size_sweep`'s
/// recorded rule — and the hypothesis here is exactly what is in doubt.
///
/// What it DOES assert is bit-identity: A and C are the same body with
/// different codegen, so they must agree exactly, and that is an assertion
/// no box state can invalidate.
#[test]
fn t2_bf16_autovec_vs_intrinsics() {
    // Two arms of ONE cfg, not a runtime test plus an early return: the
    // `#[cfg]` block below compiles to NOTHING off x86_64, which made the
    // `return` the function's last statement there and the whole arm a
    // `needless_return` visible only on aarch64 — this file's own subject,
    // one level up.
    #[cfg(not(target_arch = "x86_64"))]
    println!("   Issue 847 T2 is an x86_64 question — SKIPPED on this arch");
    #[cfg(target_arch = "x86_64")]
    {
        use katgpt_core::bf16_convert::{
            bf16_bits_to_f32_autovec, f32_to_bf16_rne_autovec, f32_to_bf16_rne_into,
            f32_to_bf16_trunc_autovec, f32_to_bf16_trunc_into,
        };
        println!("\n   Issue 847 T2 — is the hand-written AVX2 arm worth keeping?");
        println!(
            "   arch = x86_64, cfg!(target_feature=\"avx2\") = {}",
            cfg!(target_feature = "avx2")
        );
        println!(
            "   {:>8}  {:>20}  {:>14}  {:>14}  {:>11}",
            "n", "kernel", "dispatch ns", "autovec ns", "disp/auto"
        );

        for &(n, iters) in &NS {
            let mut rng = Lcg::new(0x0847_1111 + n as u64);

            // widen
            let bits: Vec<u16> = (0..n)
                .map(|_| u16::from(rng.next_u8()) << 8 | u16::from(rng.next_u8()))
                .collect();
            let mut d = vec![0.0f32; n];
            let mut a = vec![0.0f32; n];
            bf16_bits_to_f32_into(&bits, &mut d);
            unsafe { bf16_bits_to_f32_autovec(&bits, &mut a) };
            // ⛔ BITS, not values. `assert_eq!` on `Vec<f32>` is `PartialEq`,
            // under which NaN != NaN — and a random u16 fixture widens to NaN
            // bit patterns, so a value comparison FAILS on two byte-identical
            // outputs. Measured here: two 256-element vectors that printed
            // identically, differing only at one `NaN`. The claim is
            // bit-exactness, which a value comparison could not express even
            // on the runs where it passed.
            let db: Vec<u32> = d.iter().map(|x| x.to_bits()).collect();
            let ab: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
            assert_eq!(
                db, ab,
                "the autovec arm is the scalar body with other codegen — \
                 bit-identical or the experiment is meaningless (n={n})"
            );
            let r = ab_median_ratio(
                11,
                iters,
                iters / 2,
                |_| {
                    bf16_bits_to_f32_into(black_box(&bits), &mut d);
                    black_box(d[0]);
                },
                |_| {
                    unsafe { bf16_bits_to_f32_autovec(black_box(&bits), &mut a) };
                    black_box(a[0]);
                },
            );
            println!(
                "   {n:>8}  {:>20}  {:>14.0}  {:>14.0}  {:>11.2}",
                "bf16_bits_to_f32",
                r.a_ns_per_iter(),
                r.b_ns_per_iter(),
                r.median
            );
            println!(
                "   ROW847T2 widen n={n} disp_ns={:.0} autovec_ns={:.0} ratio={:.2} avx2_cfg={}",
                r.a_ns_per_iter(),
                r.b_ns_per_iter(),
                r.median,
                cfg!(target_feature = "avx2")
            );
            assert_eq!(
                r.ratios.len(),
                r.rounds,
                "instrument FAIL: {} of {} rounds survived (widen, n={n})",
                r.ratios.len(),
                r.rounds
            );

            // narrow (RNE) — the harder kernel, and the one whose scalar body
            // is a branchy NaN predicate rather than a shift
            let src: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let mut nd = vec![0u16; n];
            let mut na = vec![0u16; n];
            f32_to_bf16_rne_into(&src, &mut nd);
            unsafe { f32_to_bf16_rne_autovec(&src, &mut na) };
            assert_eq!(nd, na, "RNE narrowing: same body, so bit-identical (n={n})");
            let r2 = ab_median_ratio(
                11,
                iters,
                iters / 2,
                |_| {
                    f32_to_bf16_rne_into(black_box(&src), &mut nd);
                    black_box(nd[0]);
                },
                |_| {
                    unsafe { f32_to_bf16_rne_autovec(black_box(&src), &mut na) };
                    black_box(na[0]);
                },
            );
            println!(
                "   {n:>8}  {:>20}  {:>14.0}  {:>14.0}  {:>11.2}",
                "f32_to_bf16_rne",
                r2.a_ns_per_iter(),
                r2.b_ns_per_iter(),
                r2.median
            );
            println!(
                "   ROW847T2 rne n={n} disp_ns={:.0} autovec_ns={:.0} ratio={:.2} avx2_cfg={}",
                r2.a_ns_per_iter(),
                r2.b_ns_per_iter(),
                r2.median,
                cfg!(target_feature = "avx2")
            );
            assert_eq!(
                r2.ratios.len(),
                r2.rounds,
                "instrument FAIL: {} of {} rounds survived (rne, n={n})",
                r2.ratios.len(),
                r2.rounds
            );

            // narrow (TRUNC) — the third dispatcher. Measured rather than
            // assumed to behave like RNE: its scalar body is a bare shift
            // where RNE's is a branchy NaN predicate, so the two have no
            // reason to autovectorise alike.
            let mut td = vec![0u16; n];
            let mut ta = vec![0u16; n];
            f32_to_bf16_trunc_into(&src, &mut td);
            unsafe { f32_to_bf16_trunc_autovec(&src, &mut ta) };
            assert_eq!(
                td, ta,
                "trunc narrowing: same body, so bit-identical (n={n})"
            );
            let r3 = ab_median_ratio(
                11,
                iters,
                iters / 2,
                |_| {
                    f32_to_bf16_trunc_into(black_box(&src), &mut td);
                    black_box(td[0]);
                },
                |_| {
                    unsafe { f32_to_bf16_trunc_autovec(black_box(&src), &mut ta) };
                    black_box(ta[0]);
                },
            );
            println!(
                "   {n:>8}  {:>20}  {:>14.0}  {:>14.0}  {:>11.2}",
                "f32_to_bf16_trunc",
                r3.a_ns_per_iter(),
                r3.b_ns_per_iter(),
                r3.median
            );
            println!(
                "   ROW847T2 trunc n={n} disp_ns={:.0} autovec_ns={:.0} ratio={:.2} avx2_cfg={}",
                r3.a_ns_per_iter(),
                r3.b_ns_per_iter(),
                r3.median,
                cfg!(target_feature = "avx2")
            );
            assert_eq!(
                r3.ratios.len(),
                r3.rounds,
                "instrument FAIL: {} of {} rounds survived (trunc, n={n})",
                r3.ratios.len(),
                r3.rounds
            );
        }
        println!(
            "   ⚠ REPORT, no bar. `disp/auto` > 1 means the AUTOVEC arm wins on \
             this build. The intrinsics' own number is visible only under \
             RUSTFLAGS=\"-C target-feature=+avx2\", where the dispatcher column \
             becomes them — so BOTH builds are needed before 847 T2 can be \
             answered."
        );
    }
}

/// Issue 847 T4 — the aarch64 half of the figures. Every number in this
/// issue was measured on x86_64/shikuwa; the NEON arms were never
/// compile-feature-gated (NEON is implied by the arch), so the *defect*
/// does not exist here — but "the vector arm is N× the scalar" is an
/// x86_64 measurement, and the gate's bar is skipped off x86_64 rather
/// than assumed. This test measures the three bf16 families
/// (widen/rne/trunc) dispatcher-vs-scalar-reference on aarch64 with the
/// shared interleaved harness, and is also T5's precondition: the delete
/// decision for `f32_to_bf16_trunc_avx2` needs to know whether its NEON
/// sibling wins where the AVX2 one does not.
///
/// A REPORT, no bar (the `bench_843_ternary_size_sweep` rule: a bar written
/// before the sweep is a bar written from a hypothesis). What it DOES
/// assert, on every arch it compiles on, is bit-identity between the
/// dispatcher and the scalar reference — an assertion no box state can
/// invalidate, and the same law t2 states for its A/C arms.
///
/// ⚠ Reading rule (the module doc's confound, one arch over): on aarch64
/// the SCALAR arm is compiled with NEON available too, so LLVM may
/// autovectorise it — the ratio column is `autovectorised-scalar / NEON
/// intrinsics`, the same comparison t2 runs deliberately under `+avx2`.
/// Both absolute figures print beside it; the ratio alone never decides.
#[test]
fn t4_neon_vs_scalar_aarch64() {
    use katgpt_core::bf16_convert::{
        bf16_bits_to_f32_into, bf16_bits_to_f32_scalar_into, f32_to_bf16_rne_into,
        f32_to_bf16_rne_scalar_into, f32_to_bf16_trunc_into, f32_to_bf16_trunc_scalar_into,
    };

    if !cfg!(target_arch = "aarch64") {
        println!(
            "   t4 SKIPPED on {} — the aarch64 figures are measured on the M3; \
             this is the loud skip, never a green zero.",
            std::env::consts::ARCH
        );
        return;
    }

    println!("\n   Issue 847 T4 — NEON dispatcher vs scalar reference (aarch64)");
    println!(
        "   {:>8}  {:>22}  {:>14}  {:>14}  {:>11}",
        "n", "kernel", "NEON disp ns", "scalar ns", "scalar/disp"
    );

    for &(n, iters) in &NS {
        // widen — fixture: full-range u16 bit patterns (NaNs included; the
        // bit-identity assert below compares BITS, not f32 values, the t2 law).
        let mut bits = vec![0u16; n];
        let mut lcg = Lcg::new(0x8470_0001);
        for b in &mut bits {
            *b = ((lcg.next_u8() as u16) << 8) | lcg.next_u8() as u16;
        }
        let mut wd = vec![0f32; n];
        let mut ws = vec![0f32; n];
        bf16_bits_to_f32_into(&bits, &mut wd);
        bf16_bits_to_f32_scalar_into(&bits, &mut ws);
        assert_eq!(
            wd.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            ws.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "widen: dispatcher vs scalar must be BIT-identical (n={n})"
        );
        let r1 = ab_median_ratio(
            11,
            iters,
            iters / 2,
            |_| {
                bf16_bits_to_f32_into(black_box(&bits), &mut wd);
                black_box(wd[0]);
            },
            |_| {
                bf16_bits_to_f32_scalar_into(black_box(&bits), &mut ws);
                black_box(ws[0]);
            },
        );
        println!(
            "   {n:>8}  {:>22}  {:>14.0}  {:>14.0}  {:>11.2}",
            "bf16_bits_to_f32",
            r1.a_ns_per_iter(),
            r1.b_ns_per_iter(),
            r1.median
        );
        println!(
            "   ROW847T4 widen n={n} neon_ns={:.0} scalar_ns={:.0} ratio={:.2}",
            r1.a_ns_per_iter(),
            r1.b_ns_per_iter(),
            r1.median
        );
        assert_eq!(
            r1.ratios.len(),
            r1.rounds,
            "instrument FAIL: {} of {} rounds survived (widen, n={n})",
            r1.ratios.len(),
            r1.rounds
        );

        // narrow (RNE) — fixture in [-1, 1) plus a few interesting classes.
        let mut src = vec![0f32; n];
        let mut lcg = Lcg::new(0x8470_0002);
        for (i, v) in src.iter_mut().enumerate() {
            *v = if i % 97 == 0 {
                [f32::NAN, f32::INFINITY, 0.0, -0.0, 1e-40][i / 97 % 5]
            } else {
                lcg.next_f32()
            };
        }
        let mut rd = vec![0u16; n];
        let mut rs = vec![0u16; n];
        f32_to_bf16_rne_into(&src, &mut rd);
        f32_to_bf16_rne_scalar_into(&src, &mut rs);
        assert_eq!(
            rd, rs,
            "rne narrowing: bit-exact for every input class (n={n})"
        );
        let r2 = ab_median_ratio(
            11,
            iters,
            iters / 2,
            |_| {
                f32_to_bf16_rne_into(black_box(&src), &mut rd);
                black_box(rd[0]);
            },
            |_| {
                f32_to_bf16_rne_scalar_into(black_box(&src), &mut rs);
                black_box(rs[0]);
            },
        );
        println!(
            "   {n:>8}  {:>22}  {:>14.0}  {:>14.0}  {:>11.2}",
            "f32_to_bf16_rne",
            r2.a_ns_per_iter(),
            r2.b_ns_per_iter(),
            r2.median
        );
        println!(
            "   ROW847T4 rne n={n} neon_ns={:.0} scalar_ns={:.0} ratio={:.2}",
            r2.a_ns_per_iter(),
            r2.b_ns_per_iter(),
            r2.median
        );
        assert_eq!(
            r2.ratios.len(),
            r2.rounds,
            "instrument FAIL: {} of {} rounds survived (rne, n={n})",
            r2.ratios.len(),
            r2.rounds
        );

        // narrow (TRUNC) — T5's precondition. Its scalar body is a bare shift,
        // the easiest thing in this file for LLVM to autovectorise, so the
        // ratio here is the closest aarch64 analogue of the x86_64 finding
        // that the trunc intrinsics lose to the compiler's own loop.
        let mut td = vec![0u16; n];
        let mut ts = vec![0u16; n];
        f32_to_bf16_trunc_into(&src, &mut td);
        f32_to_bf16_trunc_scalar_into(&src, &mut ts);
        assert_eq!(td, ts, "trunc narrowing: exact by construction (n={n})");
        let r3 = ab_median_ratio(
            11,
            iters,
            iters / 2,
            |_| {
                f32_to_bf16_trunc_into(black_box(&src), &mut td);
                black_box(td[0]);
            },
            |_| {
                f32_to_bf16_trunc_scalar_into(black_box(&src), &mut ts);
                black_box(ts[0]);
            },
        );
        println!(
            "   {n:>8}  {:>22}  {:>14.0}  {:>14.0}  {:>11.2}",
            "f32_to_bf16_trunc",
            r3.a_ns_per_iter(),
            r3.b_ns_per_iter(),
            r3.median
        );
        println!(
            "   ROW847T4 trunc n={n} neon_ns={:.0} scalar_ns={:.0} ratio={:.2}",
            r3.a_ns_per_iter(),
            r3.b_ns_per_iter(),
            r3.median
        );
        assert_eq!(
            r3.ratios.len(),
            r3.rounds,
            "instrument FAIL: {} of {} rounds survived (trunc, n={n})",
            r3.ratios.len(),
            r3.rounds
        );
    }
    println!(
        "   ⚠ REPORT, no bar (T4). `scalar/disp` > 1 means the NEON dispatcher \
         wins. The scalar arm is compiled with NEON available on this arch, so \
         LLVM may autovectorise it — read the absolute pair, not the ratio \
         alone (the module-doc confound, one arch over). T5 reads the trunc \
         rows: if the NEON trunc dispatcher does not beat its scalar reference \
         anywhere, the aarch64 sibling of the x86_64 finding is confirmed and \
         the delete case for the AVX2 kernel is measured on both arches."
    );
}

/// Issue 847 T6 — the widen "crossover" is a BIMODALITY, and the axis is
/// BUFFER ALIGNMENT rather than box load.
///
/// T6 was filed as *"measure the crossover on a QUIET box"*, on the reading
/// that a cell swinging 97 ↔ 173 ns between two runs of one binary was a
/// loaded box. Measured on a quiet one (23.6 GB free, commit 27.8 / 62.8 GB,
/// nothing heavy in the top five), five runs per build:
///
/// - the SCALAR dispatcher is stable to **under 1%** — 1692, 1696, 1696,
///   1698, 1704 ns at n=32768;
/// - every AVX2-bearing arm is **BIMODAL**, ~98 or ~170 ns at n=4096, with
///   the two clusters separated by 1.75x and nothing in between;
/// - the mode is picked per PROCESS and per ARM — one run had the dispatcher
///   fast (97) and `_autovec` slow (175) in the same measurement.
///
/// A loaded box produces a SPREAD. Two clusters with a gap, while the scalar
/// arm beside them does not move, is a property of the run rather than of
/// the machine — so T6's premise is refuted and its instruction ("run it
/// somewhere quiet") could never have settled the question.
///
/// This arm tests the leading hypothesis DIRECTLY instead of running the
/// same benchmark again: `vec![…]` guarantees only the element's alignment,
/// so a 256-bit `loadu`/`storeu` sequence starts at whatever the allocator
/// returned, and a ~50/50 split is what a 32-byte-alignment coin flip looks
/// like. It measures the SAME kernel over sub-slices whose start address is
/// forced to a known residue mod 32, and prints the residue next to the
/// time.
///
/// ⛔ A REPORT, and deliberately not a bar. The asserted part is the one no
/// box state can touch: every alignment must produce BIT-IDENTICAL output,
/// because an alignment cannot change a conversion. If the timings turn out
/// flat across residues the hypothesis is refuted and the bimodality needs a
/// different explanation — which is a result, and the reason this prints a
/// table rather than a verdict.
#[test]
fn t6_widen_bimodality_vs_buffer_alignment() {
    // Two arms of ONE cfg, not a runtime test plus an early return: the
    // `#[cfg]` block below compiles to NOTHING off x86_64, which made the
    // `return` the function's last statement there and the whole arm a
    // `needless_return` visible only on aarch64 — this file's own subject,
    // one level up.
    #[cfg(not(target_arch = "x86_64"))]
    println!("   Issue 847 T6 is an x86_64 question — SKIPPED on this arch");
    #[cfg(target_arch = "x86_64")]
    {
        use katgpt_core::bf16_convert::{bf16_bits_to_f32_autovec, bf16_bits_to_f32_into};
        println!("\n   Issue 847 T6 — is the widen bimodality BUFFER ALIGNMENT?");
        println!(
            "   cfg!(target_feature=\"avx2\") = {}",
            cfg!(target_feature = "avx2")
        );
        println!(
            "   {:>8}  {:>9}  {:>9}  {:>13}  {:>13}",
            "n", "src%32", "dst%32", "dispatch ns", "autovec ns"
        );

        // One oversized allocation per buffer, sliced at a controlled offset.
        // Taking sub-slices of ONE allocation is what makes the residue the
        // only thing that varies: a fresh `vec!` per residue would re-roll
        // the allocator and measure two things at once.
        for &(n, iters) in &NS {
            let mut rng = Lcg::new(0x0847_6666 + n as u64);
            let pad = 32usize;
            let src_all: Vec<u16> = (0..n + pad)
                .map(|_| u16::from(rng.next_u8()) << 8 | u16::from(rng.next_u8()))
                .collect();
            let mut d_all = vec![0.0f32; n + pad];
            let mut a_all = vec![0.0f32; n + pad];

            // The oracle is computed ONCE, from the aligned case, and every
            // residue is compared against it — so a shifted window cannot
            // silently compare against itself.
            let mut oracle = vec![0.0f32; n];
            bf16_bits_to_f32_into(&src_all[..n], &mut oracle);

            for &(so, doff) in &[(0usize, 0usize), (1, 1), (8, 4), (9, 5)] {
                let src = &src_all[so..so + n];
                let sres = (src.as_ptr() as usize) % 32;
                let dres = (unsafe { d_all.as_ptr().add(doff) } as usize) % 32;
                {
                    let d = &mut d_all[doff..doff + n];
                    bf16_bits_to_f32_into(&src_all[..n], d);
                    // ⛔ BITS, not values. `Vec<f32>` compares with
                    // `PartialEq`, under which `NaN != NaN`, and a random
                    // `u16` fixture widens to NaN — so a value comparison
                    // fails on two byte-identical vectors. The module doc
                    // records this trap for T2’s arm and the first draft of
                    // THIS arm reproduced it anyway, which is the argument
                    // for writing it at the line.
                    let got: Vec<u32> = d.iter().map(|x| x.to_bits()).collect();
                    let want: Vec<u32> = oracle.iter().map(|x| x.to_bits()).collect();
                    assert_eq!(
                        got, want,
                        "alignment changed the RESULT (n={n}, src%32={sres}, \
                                dst%32={dres}) — an alignment cannot change a \
                                conversion, so the instrument is wrong"
                    );
                }
                let r = {
                    let (d_chunk, a_chunk) = (doff, doff);
                    let _ = (d_chunk, a_chunk);
                    ab_median_ratio(
                        11,
                        iters,
                        iters / 2,
                        |_| {
                            let d = &mut d_all[doff..doff + n];
                            bf16_bits_to_f32_into(black_box(src), d);
                            black_box(d[0]);
                        },
                        |_| {
                            let a = &mut a_all[doff..doff + n];
                            unsafe { bf16_bits_to_f32_autovec(black_box(src), a) };
                            black_box(a[0]);
                        },
                    )
                };
                println!(
                    "   {n:>8}  {sres:>9}  {dres:>9}  {:>13.0}  {:>13.0}",
                    r.a_ns_per_iter(),
                    r.b_ns_per_iter()
                );
                println!(
                    "   ROW847T6 widen n={n} src_res={sres} dst_res={dres} \
                          disp_ns={:.0} autovec_ns={:.0} avx2_cfg={}",
                    r.a_ns_per_iter(),
                    r.b_ns_per_iter(),
                    cfg!(target_feature = "avx2")
                );
                assert_eq!(
                    r.ratios.len(),
                    r.rounds,
                    "instrument FAIL: {} of {} rounds survived \
                            (T6, n={n}, src%32={sres})",
                    r.ratios.len(),
                    r.rounds
                );
            }
        }
        println!(
            "   ⚠ REPORT, no bar. If the times track the residue columns, the \
             bimodality is alignment and the earlier 97-vs-173 'crossover' was \
             one coin flip per process. If they are FLAT, the hypothesis is \
             refuted and T6 needs a different axis — record which, because a \
             refuted hypothesis measured on a quiet box is still an answer."
        );
    }
}
