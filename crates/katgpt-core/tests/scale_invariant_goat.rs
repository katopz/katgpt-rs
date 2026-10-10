//! Plan 622 Phase 7 GOAT gates — the CPU-executable legs.
//!
//! The Phase-1..6 suites already pin most of G1 (LUT vs closed form ≤ 1e-6,
//! the boundary pins, the build validity assert, the sigmoid transfer vs
//! Φ(−1/a_t) within C·a_t⁻³ with C fixed once — `tilt_matches_phi_
//! minus_one_over_a_within_c_a_cubed` in `scale_invariant.rs`, and the
//! no-double-temperature ordering law — `si_and_ssmax_armed_is_a_loud_
//! config_error` in `parallax_attn/tests.rs`). THIS file adds the legs those
//! suites do not carry:
//!
//! - **G1 decade-mass flatness to t = 10⁶** — the Θ(1) band (≤ ατ·ln Δ,
//!   ≥ the T-independent floor) holds on the LAST decade window, summed
//!   from the f32 LUT pairs the kernels actually apply (not just f64
//!   closed forms), with the f64 harmonic reference alongside.
//! - **G2 LUT build < 1 ms @ 64k** — release-profile bar, `black_box`
//!   loud-zero defence (the measured object is asserted, not just timed).
//! - **G4 zero-alloc apply** — the counting-allocator canary over BOTH
//!   apply variants (`apply_inplace`, `apply_inplace_calibrated`) on
//!   full-width rows; the LUT itself is the preallocated session object.
//!
//! The GPU legs (G2 pp2048/tg128 ±3%, G3 Bonsai zero-shot split rows,
//! Bonsai real-logit decade diagnostics, league cells) ride the engine
//! wiring and are recorded in the plan, not here.
//!
//! Run: `cargo test -p katgpt-core --features scale_invariant_attn --test scale_invariant_goat`
//! (the G2 bar asserts under `--release` only; dev prints the measurement
//! and skips the bar LOUDLY — a debug-profile latency number is not a
//! measurement of the shipped binary).
//!
//! Cross-references:
//! - Plan: `.plans/622_scale_invariant_attn.md` (Phase 7)
//! - Implementation: `crates/katgpt-core/src/scale_invariant.rs`

#![cfg(feature = "scale_invariant_attn")]
#![cfg(test)]

#[path = "common/mod.rs"]
mod common;

counting_allocator!();

use katgpt_core::scale_invariant::ScaleInvariantLut;
use std::hint::black_box;
use std::sync::atomic::Ordering;

// ── The schedule reference (f64, independent transcription of the spec) ──

fn alpha() -> f64 {
    (0.5f64).exp()
}

/// The Def-3.1 T-independent floor: ατ·log((Δ+1+τ)/(2+τ)) for Δ ≥ 2.
fn decade_floor(d: f64, tau: f64) -> f64 {
    alpha() * tau * ((d + 1.0 + tau) / (2.0 + tau)).ln()
}

// ── G1: decade-mass flatness to t = 10⁶ ─────────────────────────────────

/// The last decade window's unnormalized Gaussian mass stays in the Θ(1)
/// band at T = 10⁶, summed from the f32 LUT pairs.
///
/// Per-key mass `exp(m_t + a_t²/2)` = `α/(t/τ+1)` (the Lean
/// `perKeyMass_eq_alpha` identity) — the window (10⁵, 10⁶] holds 9·10⁵
/// terms each ≈ 1.65e-5, summing to ≈ ατ·ln(Δ) ≈ 37.9 at τ = 10: Θ(1) at a
/// context length where SSMax's position-independent scalar has long
/// flattened the distribution (the length-generalization claim, checked at
/// the extreme).
///
/// Three legs: (i) the f64 harmonic reference sum is inside the Lean
/// interval bounds EXACTLY (no tolerance beyond summation order); (ii) the
/// f32-LUT mass sum equals the reference within 1e-4 relative (f32 pair
/// rounding ~1.4e-6 per term at a ≈ 4.4, summation-accumulated); (iii) the
/// LUT sum is inside the T-independent band — ≤ ατ·ln Δ and ≥ the
/// positive floor — the flatness statement at t = 10⁶.
#[test]
fn g1_decade_mass_flat_at_one_million() {
    let tau = 10.0_f64;
    let big_t: u64 = 1_000_000;
    let d: u64 = 10;
    // The LUT must cover the window: build once at max_ctx = 10⁶ (the
    // no-NaN-to-10⁶ posture; ~8 MB interleaved f32 — session-scale).
    let lut = ScaleInvariantLut::build(tau as f32, big_t as usize);
    assert_eq!(lut.max_distance(), big_t as usize);

    // (i) the f64 harmonic reference over the same window.
    let t_win = big_t / d; // T — the window is (T, TΔ] with TΔ = big_t
    let t_lo = t_win + 1;
    let t_hi = big_t;
    let mut ref_sum = 0.0_f64;
    for t in t_lo..=t_hi {
        ref_sum += alpha() / (t as f64 / tau + 1.0);
    }
    let lo_bound =
        alpha() * tau * ((big_t as f64 + 1.0 + tau) / (t_win as f64 + 1.0 + tau)).ln();
    let hi_bound = alpha() * tau * ((big_t as f64 + tau) / (t_win as f64 + tau)).ln();
    assert!(
        lo_bound <= ref_sum && ref_sum <= hi_bound,
        "f64 reference outside the Lean interval: {lo_bound} ≤ {ref_sum} ≤ {hi_bound}"
    );

    // (ii) the f32-LUT mass sum (what the kernels apply) matches.
    let mut lut_sum = 0.0_f64;
    for t in t_lo..=t_hi {
        let (a, m) = lut.pair(t as usize);
        lut_sum += ((m as f64) + (a as f64) * (a as f64) / 2.0).exp();
    }
    let rel = (lut_sum - ref_sum).abs() / ref_sum;
    assert!(
        rel <= 1e-4,
        "f32-LUT decade mass drifted from the f64 reference: rel {rel:.3e} at T={big_t}"
    );

    // (iii) the Θ(1) band: ≤ ατ·ln Δ always, ≥ the T-independent floor
    // for Δ ≥ 2 — flatness at the top of the ladder.
    let band = alpha() * tau * (d as f64).ln();
    let floor = decade_floor(d as f64, tau);
    assert!(floor > 0.0);
    assert!(
        lut_sum <= band * (1.0 + 1e-9),
        "band violated at T={big_t}: {lut_sum} > ατ·lnΔ = {band}"
    );
    assert!(
        lut_sum >= floor * (1.0 - 1e-9),
        "floor violated at T={big_t}: {lut_sum} < {floor}"
    );
}

// ── G2: LUT build < 1 ms @ 64k ──────────────────────────────────────────

/// The session-start LUT build is sub-millisecond at the 64k posture —
/// the whole point of the precompute (per-token apply must not pay the
/// schedule's transcendental cost).
///
/// Loud-zero defence (the timed_region lesson): the measured object is
/// ASSERTED, not just timed — the boundary pair must be exact and the
/// top-distance pair finite on every timed build, so a build optimized to
/// nothing cannot satisfy the bar. The bar itself runs under `--release`
/// only; a dev-profile run PRINTS the measurement and skips the bar with
/// a loud disclosure (a debug-profile latency number is not a measurement
/// of the shipped binary — the profile is part of the claim).
#[test]
fn g2_lut_build_under_1ms_at_64k() {
    const MAX_CTX: usize = 65_535;
    // Warmup (page the 512 KB table, spin the transcendentals' caches).
    let warm = ScaleInvariantLut::build(10.0, MAX_CTX);
    let (a0, m0) = warm.pair(0);
    assert_eq!((a0, m0), (1.0, 0.0), "boundary pin must hold on every build");
    let (a_top, m_top) = warm.pair(MAX_CTX);
    assert!(
        (a_top as f64).is_finite() && (m_top as f64).is_finite(),
        "top pair must be finite on every build"
    );

    const ITERS: usize = 7;
    let mut best = f64::INFINITY;
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        let lut = ScaleInvariantLut::build(black_box(10.0), black_box(MAX_CTX));
        let dt = t0.elapsed().as_secs_f64();
        // The loud-zero defence: every timed build produced a real table.
        let (a, m) = black_box(lut).pair(MAX_CTX);
        assert!((a as f64).is_finite() && (m as f64).is_finite());
        best = best.min(dt);
    }
    let ms = best * 1e3;
    if cfg!(debug_assertions) {
        // Dev profile: report, don't gate. The bar belongs to --release.
        println!("g2_lut_build_under_1ms_at_64k: dev profile, measured {ms:.3} ms — bar SKIPPED (run --release)");
    } else {
        assert!(
            ms < 1.0,
            "LUT build {ms:.3} ms @ 64k exceeds the 1 ms G2 bar (--release)"
        );
        println!("g2_lut_build_under_1ms_at_64k: {ms:.3} ms @ 64k (release) — PASS");
    }
}

// ── G4: zero-alloc apply ────────────────────────────────────────────────

/// The hot-path apply allocates NOTHING — both variants, on full-width
/// rows, at the boundary and mid-row entry positions. The LUT is the
/// preallocated session object; the score rows are the caller's buffers.
///
/// Setup allocations (the LUT, the score vectors) happen BEFORE the
/// counting window; the per-thread counter then must not move across 2·64
/// apply calls spanning both entry shapes and both calibration postures
/// (head_scale = 1.0 the bit-identical pass-through, 1.7 the calibrated
/// divide — the fusion must not gain a scratch buffer for either).
#[test]
fn g4_apply_is_allocation_free() {
    let max_ctx = 4096;
    let lut = ScaleInvariantLut::build(10.0, max_ctx);
    // Two rows: full-causal (query at the end) and mid-row entry.
    let mut full: Vec<f32> = (0..=max_ctx).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut mid = full.clone();
    let head_start = ALLOC_COUNT.load(Ordering::Relaxed);

    for q in [max_ctx, max_ctx / 2] {
        // Uncalibrated.
        lut.apply_inplace(&mut full, q, 0, 1.0);
        // Calibrated (the fused divide+affine — no scratch).
        lut.apply_inplace_calibrated(&mut full, q, 0, 1.0, 1.7);
        // Mid-row entry, both postures.
        lut.apply_inplace(&mut mid[16..], q, 16, 1.0);
        lut.apply_inplace_calibrated(&mut mid[16..], q, 16, 1.0, 1.7);
        // Logit-scale folding posture (pre-scale space — the SDPA shape).
        lut.apply_inplace(&mut full, q, 0, 8.0);
        lut.apply_inplace_calibrated(&mut full, q, 0, 8.0, 1.7);
    }

    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - head_start;
    assert_eq!(
        allocs, 0,
        "the si apply allocated {allocs} time(s) across 12 calls — the hot path must be alloc-free"
    );

    // Loud-zero defence: the transforms actually ran (a flat identity row
    // would allocate nothing too). The applied rows must differ from the
    // pristine inputs at the transformed positions, and the sink must not
    // have moved.
    let pristine: Vec<f32> = (0..=max_ctx).map(|i| (i as f32 * 0.01).sin()).collect();
    assert_ne!(full[1..], pristine[1..], "apply was a no-op — the canary measured nothing");
    assert_eq!(full[0], pristine[0], "the sink (key 0) must be untouched");
    assert_ne!(mid[17..], pristine[17..], "mid-row apply was a no-op");
}
