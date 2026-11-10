//! Spec-match test for the SSMax Lean 4 dilution-bound proof.
//!
//! **Plan 411 S3.** This test asserts that the Rust dilution-bound formula
//! (documented in `crates/katgpt-core/src/ssmax.rs` lines 11–17) and
//! `SsmaxMode::multiplier` match the Lean 4 spec at
//! `katgpt-rs/.proofs/KatgptProof/Ssmax/Basic.lean` + `DilutionBound.lean`.
//!
//! If this test fails, the Lean proof in `.proofs/` is invalid (Rust drifted
//! from spec). The proof must be updated to match before merging.
//!
//! Run: `cargo test --features ssmax_temperature --test ssmax_spec_match`
//! (now runs by default since `ssmax_temperature` is DEFAULT-ON, Phase 13.)
//!
//! ## What the Lean proof establishes
//!
//! 1. **`alphaGold_strictMono_in_c`** — for `N > 1`, the gold mass
//!    `α_gold(N, c) = 1 / (1 + (N−1)·N^(−c))` is strictly increasing in `c`
//!    (the effective sharpening exponent). This is WHY SSMax works: rescaling
//!    logits to increase `c` provably pushes more mass onto the gold document.
//!
//! 2. **`ssmax_dominates_base`** — for `s_L · log(N) ≥ 1` and `c_base > 0`:
//!    SSMax (replacing `c_base` with `s_L · log(N) · c_base`) does not decrease
//!    the gold mass. The hypothesis `s_L · log(N) ≥ 1` is **tight**: at
//!    `s_L = 1, N = 2`, `log(2) ≈ 0.693 < 1`, so SSMax is *milder* than base.
//!
//! The formal proof **sharpened the plan's informal statement** — Plan 411 S3
//! sketched "`s_L = 1, N ≥ 2 ⇒ SSMax ≥ base`", but the correct threshold is
//! `s_L · log(N) ≥ 1` (i.e. `N ≥ 3` for `s_L = 1`). This test guards both the
//! dominance (for `N ≥ 3`) and the reversal (at `N = 2`).
//!
//! Cross-references:
//! - Plan: `.plans/411_ssmax_goldshare.md` (Stretch S3)
//! - Research: `.research/392_*`
//! - Lean proof: `.proofs/KatgptProof/Ssmax/DilutionBound.lean`
//! - Rust implementation: `crates/katgpt-core/src/ssmax.rs`

#![cfg(test)]

use katgpt_core::ssmax::{SsmaxConfig, SsmaxMode, apply_ssmax_inplace};

// ── Spec: the dilution bound formula ────────────────────────────────────

/// The post-normalization gold attention mass under the paper's dilution bound
/// (arXiv:2607.01538 §2). Mirrors the Lean definition at
/// `.proofs/KatgptProof/Ssmax/Basic.lean::alphaGold`:
///
/// ```text
/// alphaGold N c = 1 / (1 + (N − 1) · N^(−c))
/// ```
///
/// where `N` is the corpus size and `c` is the effective sharpening exponent
/// (base: `c = s · Δ`; SSMax: `c = s_L · log(N) · s · Δ`).
///
/// This is the Rust re-implementation of the Lean spec. The tests below
/// assert this function (and the Rust `apply_ssmax_inplace` semantics) agree
/// with the Lean theorems at f32 precision.
fn alpha_gold(n: f64, c: f64) -> f64 {
    // Mirrors `alphaGold N c = 1 / (1 + (N − 1) · N^(−c))` from Lean.
    // f64 (not f32) for the spec-match math — the Lean proof is over ℝ,
    // and f64 is closer to the idealised real contract than f32.
    1.0 / (1.0 + (n - 1.0) * n.powf(-c))
}

// ── Spec-match tests ───────────────────────────────────────────────────

/// The dilution bound formula matches the paper's headline numbers.
///
/// Lean spec: `alphaGold N c = 1 / (1 + (N−1) · N^(−c))` (Basic.lean).
/// At `N = 10_000, c = 0` (no sharpening): `alphaGold = 1 / (1 + 9999) = 1e-4`
/// — the gold mass is uniformly distributed, exactly `1/N`. At `c → ∞`:
/// `alphaGold → 1` (all mass on gold). This is the extreme-behavior check.
#[test]
fn spec_dilution_bound_extremes() {
    // c = 0: uniform distribution, alphaGold = 1/N.
    let n = 10_000.0_f64;
    let uniform = alpha_gold(n, 0.0);
    assert!(
        (uniform - 1.0 / n).abs() < 1e-12,
        "alphaGold(N, 0) must be 1/N (uniform), got {uniform} vs {}",
        1.0 / n
    );

    // c → large: gold mass → 1.
    let sharp = alpha_gold(n, 100.0);
    assert!(
        sharp > 0.9999,
        "alphaGold(N, large c) must approach 1, got {sharp}"
    );

    // The paper's dilution: as N grows at fixed c, gold mass drops.
    // c = 1 (baseline exponent): alphaGold(10, 1) > alphaGold(1000, 1) > alphaGold(100_000, 1).
    let m10 = alpha_gold(10.0, 1.0);
    let m1k = alpha_gold(1000.0, 1.0);
    let m100k = alpha_gold(100_000.0, 1.0);
    assert!(m10 > m1k, "gold mass must drop as N grows (10 vs 1000)");
    assert!(m1k > m100k, "gold mass must drop as N grows (1000 vs 100k)");
}

/// **Lean theorem `alphaGold_strictMono_in_c` (empirical ∃-check).**
///
/// The Lean proof establishes that `alphaGold N c` is strictly increasing in
/// `c` for `N > 1` — this is the monotonicity that makes SSMax work. This test
/// samples random `(N, c₁, c₂)` triples and confirms the *observed* f64
/// ordering matches the Lean theorem.
///
/// The Lean theorem holds for **every** triple over ℝ; this test guards against
/// f64-precision regressions and is the complement to the ∀-theorem.
///
/// **Saturation ties:** at large `c` (large sharpening), `N^(−c)` underflows
/// to `0.0` in f64, so `alphaGold` saturates to exactly `1.0`. Two distinct
/// `c` values that both saturate produce `m₁ = m₂ = 1.0` — a tie, not a
/// monotonicity violation. This mirrors the bridge spec-match's handling of
/// f32 sigmoid saturation. The only real regression is an ordering **flip**
/// (larger `c` → smaller `m`), which this test catches.
#[test]
fn spec_alpha_gold_strict_mono_in_c() {
    let mut rng = fastrand::Rng::with_seed(411);
    let mut checked = 0usize;
    let mut ties = 0usize;

    for _ in 0..10_000 {
        // N in (2, 100_000): the dilution regime.
        let n = 2.0 + rng.f64() * 99_998.0;
        // c₁, c₂ in (0, 3): keep below the saturation regime (at large c,
        // N^(-c) underflows to 0 and alphaGold saturates to 1.0 — a tie, not
        // a violation). Cap at 3.0 to stay in the strict-ordering band for
        // most sampled N; the bridge spec-match uses the same cap-and-skip
        // pattern for its sigmoid monotonicity check.
        let c1 = rng.f64() * 3.0;
        let c2 = rng.f64() * 3.0;

        if (c1 - c2).abs() < 1e-10 {
            continue; // skip near-ties in the input.
        }

        let m1 = alpha_gold(n, c1);
        let m2 = alpha_gold(n, c2);

        // Allowable outcomes:
        //   - c₁ < c₂  AND  m₁ < m₂   (strict preservation)
        //   - c₁ < c₂  AND  m₁ = m₂   (f64-saturation tie — ok)
        //   - c₁ > c₂  AND  m₁ > m₂   (strict preservation)
        //   - c₁ > c₂  AND  m₁ = m₂   (f64-saturation tie — ok)
        // Forbidden (would be a real monotonicity regression):
        //   - c₁ < c₂  AND  m₁ > m₂   (FLIP)
        //   - c₁ > c₂  AND  m₁ < m₂   (FLIP)
        let c_lt = c1 < c2;
        let m_lt = m1 < m2;
        let m_eq = m1 == m2;
        if m_eq {
            ties += 1;
            continue;
        }
        // Not a tie → ordering must match exactly (no flip).
        assert_eq!(
            c_lt, m_lt,
            "alphaGold monotonicity FLIP: N={n}, c1={c1} c2={c2}, \
             m1={m1} m2={m2} — Lean alphaGold_strictMono_in_c assumes strict increase",
        );
        checked += 1;
    }
    assert!(
        checked > 8000,
        "must have checked >8000 strict pairs (got {checked})"
    );
    // Ties are expected but should be a minority.
    assert!(
        ties < 2000,
        "too many f64-saturation ties ({ties}) — c-range too wide?"
    );
}

/// **Lean theorem `ssmax_dominates_base` (empirical ∃-check).**
///
/// For `s_L · log(N) ≥ 1` and `c_base > 0`: applying SSMax (replacing `c_base`
/// with `s_L · log(N) · c_base`) does not decrease the gold mass. This test
/// checks the dominance across a grid of `(N, s_L, c_base)` values satisfying
/// the hypothesis.
#[test]
fn spec_ssmax_dominates_base_when_threshold_met() {
    for &n in &[3.0_f64, 5.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0] {
        for &s_l in &[1.0_f64, 1.5, 2.0, 5.0] {
            let log_n = n.ln();
            let threshold = s_l * log_n;
            if threshold < 1.0 {
                continue; // skip: hypothesis not met (need s_L · log(N) ≥ 1).
            }
            for &c_base in &[0.1_f64, 0.5, 1.0, 2.0, 5.0] {
                let c_ssmax = s_l * log_n * c_base;
                let m_base = alpha_gold(n, c_base);
                let m_ssmax = alpha_gold(n, c_ssmax);
                assert!(
                    m_ssmax >= m_base - 1e-12,
                    "ssmax_dominates_base violated: N={n}, s_L={s_l}, c_base={c_base} \
                     (s_L·log(N)={threshold:.4} ≥ 1), m_base={m_base:.6}, m_ssmax={m_ssmax:.6} — \
                     Lean ssmax_dominates_base says SSMax should not decrease gold mass"
                );
            }
        }
    }
}

/// **The sharpened threshold (the formal-verification value-add).**
///
/// The Lean proof revealed that Plan 411 S3's informal statement
/// "`s_L = 1, N ≥ 2 ⇒ SSMax ≥ base`" is **false at N = 2**: `log(2) ≈ 0.693 < 1`,
/// so SSMax at `s_L = 1, N = 2` is *milder* than base. The correct threshold is
/// `s_L · log(N) ≥ 1` (i.e. `N ≥ 3` for `s_L = 1`).
///
/// This test documents the reversal at N = 2 and confirms the correction at
/// N = 3, guarding the precise threshold the Lean theorem establishes.
#[test]
fn spec_threshold_is_s_l_times_log_n_geq_one_not_n_geq_two() {
    let c_base = 1.0_f64;

    // N = 2, s_L = 1: s_L · log(N) = log(2) ≈ 0.693 < 1. SSMax is MILDER.
    // The reversal: SSMax gold mass < base gold mass.
    let n2: f64 = 2.0;
    let s_l = 1.0;
    let log_n2 = n2.ln();
    assert!(
        s_l * log_n2 < 1.0,
        "log(2) must be < 1 (got {}) — this is why the plan's N≥2 threshold is wrong",
        s_l * log_n2
    );
    let m_base_n2 = alpha_gold(n2, c_base);
    let m_ssmax_n2 = alpha_gold(n2, s_l * log_n2 * c_base);
    assert!(
        m_ssmax_n2 < m_base_n2,
        "At N=2, s_L=1: SSMax must be MILDER than base (m_ssmax={m_ssmax_n2} < m_base={m_base_n2}) \
         — log(2) < 1 so the effective exponent DECREASES"
    );

    // N = 3, s_L = 1: s_L · log(N) = log(3) ≈ 1.099 > 1. SSMax is STRONGER.
    let n3: f64 = 3.0;
    let log_n3 = n3.ln();
    assert!(
        s_l * log_n3 > 1.0,
        "log(3) must be > 1 (got {}) — this is the corrected threshold",
        s_l * log_n3
    );
    let m_base_n3 = alpha_gold(n3, c_base);
    let m_ssmax_n3 = alpha_gold(n3, s_l * log_n3 * c_base);
    assert!(
        m_ssmax_n3 >= m_base_n3,
        "At N=3, s_L=1: SSMax must be ≥ base (m_ssmax={m_ssmax_n3} ≥ m_base={m_base_n3}) \
         — log(3) > 1 so ssmax_dominates_base applies"
    );
}

// ── `apply_ssmax_inplace` semantics match the multiplier ─────────────────

/// The Rust `SsmaxMode::multiplier(log_n)` must return `s_L · log_n`.
///
/// Lean spec: the SSMax intervention replaces `c_base` with
/// `c_SSMax = s_L · log(N) · c_base`. The multiplier applied to each logit is
/// `s_L · log(N)`. If the Rust multiplier drifts (e.g. `s_L · log(N) / √d`
/// accidentally folded in), the Lean theorem's mapping from logit-scale to
/// exponent-scale breaks.
#[test]
fn spec_multiplier_is_s_l_times_log_n() {
    let log_n = 10_000.0_f32.ln(); // ≈ 9.21

    // Fixed mode: multiplier = s_l · log_n.
    for &s_l in &[0.5_f32, 1.0, 2.0, 5.0] {
        let mode = SsmaxMode::Fixed { s_l };
        let got = mode.multiplier(log_n);
        let expected = s_l * log_n;
        assert!(
            (got - expected).abs() < 1e-5,
            "SsmaxMode::Fixed{{s_l={s_l}}}.multiplier({log_n}) = {got}, expected {expected} \
             — Lean spec assumes multiplier = s_L · log(N)"
        );
    }

    // Adaptive mode: s_l = 1/rolling_delta clamped to [0.1, 10.0].
    // multiplier = resolve_s_l() · log_n.
    let mode = SsmaxMode::Adaptive { rolling_delta: 0.5 };
    let s_l_resolved = mode.resolve_s_l();
    let got = mode.multiplier(log_n);
    let expected = s_l_resolved * log_n;
    assert!(
        (got - expected).abs() < 1e-5,
        "SsmaxMode::Adaptive multiplier = {got}, expected {s_l_resolved} · {log_n} = {expected}"
    );
}

/// `apply_ssmax_inplace` multiplies each logit by `s_L · log(N)`.
///
/// This is the SSMax intervention. The Lean dilution bound
/// (`alphaGold N (s_L · log N · c)`) assumes the logit gap scales by exactly
/// `s_L · log(N)`; if `apply_ssmax_inplace` applies a different factor, the
/// bound's exponent is wrong and the theorem doesn't apply.
#[test]
fn spec_apply_ssmax_scales_by_multiplier() {
    let log_n = 5.0_f32; // arbitrary.
    let mode = SsmaxMode::Fixed { s_l: 2.0 };
    let mult = mode.multiplier(log_n); // 2.0 · 5.0 = 10.0

    let original = [-1.0_f32, 0.5, 2.0, -0.3, 7.0, -9.0, 0.0, 1e-5];
    let mut logits = original;
    apply_ssmax_inplace(&mut logits, &mode, log_n);

    for (i, (&orig, &scaled)) in original.iter().zip(logits.iter()).enumerate() {
        let expected = orig * mult;
        assert!(
            (scaled - expected).abs() < 1e-5,
            "logits[{i}] = {scaled}, expected {orig} · {mult} = {expected} \
             — apply_ssmax_inplace must scale by s_L · log(N) exactly"
        );
    }
}

/// `SsmaxConfig` caches `log(N)` correctly: `from_mode(N).log_n == N.ln()`.
///
/// The Lean proof's dominance theorem uses `Real.log N`; the Rust
/// `SsmaxConfig::from_mode(n).log_n` must agree. A drift here would break the
/// `s_L · log(N) ≥ 1` threshold check silently.
#[test]
fn spec_config_caches_log_n_correctly() {
    let mode = SsmaxMode::default(); // Fixed { s_l: 1.0 }

    for &n in &[2_usize, 3, 10, 100, 1_000, 10_000, 100_000] {
        let cfg = SsmaxConfig::from_mode(&mode, n);
        let expected = (n as f32).ln();
        assert!(
            (cfg.log_n - expected).abs() < 1e-5,
            "SsmaxConfig::from_mode(_, {n}).log_n = {}, expected ln({n}) = {expected}",
            cfg.log_n
        );
    }

    // Edge: N ≤ 1 → log_n = 0.0 (no-op, by convention).
    let cfg0 = SsmaxConfig::from_mode(&mode, 0);
    assert_eq!(cfg0.log_n, 0.0, "log_n at N=0 must be 0.0 (convention)");
    let cfg1 = SsmaxConfig::from_mode(&mode, 1);
    assert_eq!(cfg1.log_n, 0.0, "log_n at N=1 must be 0.0 (convention)");
}

// ── Sentinel: .proofs/KatgptProof/Ssmax/ directory integrity ─────────────

/// Sentinel test: the Lean 4 proof directory for SSMax exists. If the Ssmax
/// proof files are ever deleted, this test fails and reminds maintainers to
/// restore them — the dilution-bound monotonicity and dominance would
/// otherwise rely only on the empirical G1/G2 tests rather than the Lean
/// ∀-theorems.
#[test]
fn proofs_directory_exists() {
    // CARGO_MANIFEST_DIR is `crates/katgpt-core` — the `.proofs/` dir is at
    // the repo root, two levels up.
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../");
    let proofs_dir = repo_root.join(".proofs");
    assert!(
        proofs_dir.exists(),
        ".proofs/ directory is missing at {} — Lean 4 dilution-bound proof is gone. \
         See .plans/411_ssmax_goldshare.md (Stretch S3)",
        proofs_dir.display()
    );
    assert!(
        proofs_dir.join("KatgptProof/Ssmax/Basic.lean").exists(),
        "Lean spec file missing: .proofs/KatgptProof/Ssmax/Basic.lean"
    );
    assert!(
        proofs_dir
            .join("KatgptProof/Ssmax/DilutionBound.lean")
            .exists(),
        "Lean theorem file missing: .proofs/KatgptProof/Ssmax/DilutionBound.lean"
    );
}
