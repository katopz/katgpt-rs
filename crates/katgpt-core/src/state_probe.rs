//! state_probe — the FlyBy decision-layer probe kernel (Plan 621 Phase 1).
//!
//! Given a plain ensemble outcome over one decision state — `n` sampled
//! continuations, `pass_count` of which reached a verified answer, and the
//! histogram of the answers they produced — this computes the statistics the
//! Phase-2 bottleneck classifier consumes:
//!
//! * `v_hat = pass_count / n` — the pass-fraction point estimate `V(s)`
//!   (FlyBy: "productive reasoning moves toward `(V, H) = (1, 0)`");
//! * `h_mm` — the answer entropy `H(s)` in nats with the Miller–Madow bias
//!   correction `(K′ − 1) / (2N)`, where `K′` is the number of NONZERO bins
//!   (the observed-support reading of Miller 1955; on a fully populated
//!   histogram `K′` equals the alphabet size, which is Research 609's
//!   `(K − 1)/(2N)` form and the fixture's case);
//! * the Wilson score interval on `v_hat` — **CONSUMED from
//!   [`crate::hint_regret::wilson_score_ci`]**, never a third copy
//!   (Research 609's DRY pin; the byte-identical `speculative::qmc`
//!   bootstrap twin is recorded there as a pre-existing observation).
//!
//! NOT calibrated UQ: the interval is a binomial CI on the ensemble
//! pass-rate, nothing more. If a consumer treats it as calibrated
//! uncertainty, the Report-the-Floor extension binds at that consumer's
//! re-gate (Plan 621 GOAT summary). Reflex-free by construction: the kernel
//! takes plain counts and emits plain floats — no prompt, no model, no
//! sampling.
//!
//! Zero-alloc: borrows the caller's histogram slice; no collections.

use crate::hint_regret::wilson_score_ci;

/// The ensemble outcome over one decision state (plain counts — the caller
/// owns the sampling; the kernel never generates continuations).
#[derive(Debug, Clone, Copy)]
pub struct ProbeInput<'a> {
    /// Ensemble members whose continuation reached a verified answer.
    pub pass_count: u32,
    /// Total ensemble members (`n` in `V(s) = k/n`).
    pub n: u32,
    /// Answer histogram over the ensemble (index = answer class). Should
    /// summarize the SAME ensemble as `n` when answers are tracked; an empty
    /// slice means answers-not-tracked and yields `h_mm = 0.0`.
    pub histogram: &'a [u16],
}

/// The probe's read over one state — plain floats, no policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeEstimate {
    /// `V̂ = pass_count / n` (0.0 at `n = 0`).
    pub v_hat: f64,
    /// Miller–Madow answer entropy `Ĥ = H_emp + (K′−1)/(2N)` in nats
    /// (0.0 at `N = 0`).
    pub h_mm: f64,
    /// Wilson score interval on `v_hat` (substrate-computed; `(0.0, 1.0)`
    /// at `n = 0`).
    pub wilson_lo: f64,
    /// Wilson score interval upper bound.
    pub wilson_hi: f64,
}

/// Probe one state. `z` is the two-sided Wilson critical value
/// (`1.959963984540054` for 95%) — passed straight through to the substrate.
#[inline]
pub fn probe(input: &ProbeInput<'_>, z: f64) -> ProbeEstimate {
    let v_hat = if input.n == 0 {
        0.0
    } else {
        f64::from(input.pass_count) / f64::from(input.n)
    };
    let (wilson_lo, wilson_hi) = wilson_score_ci(v_hat, u64::from(input.n), z);
    ProbeEstimate {
        v_hat,
        h_mm: miller_madow_entropy(input.histogram),
        wilson_lo,
        wilson_hi,
    }
}

/// Miller–Madow entropy over a count histogram, nats.
///
/// `H_emp = ln N − (Σ cᵢ ln cᵢ)/N` over nonzero bins (algebraically the
/// plug-in `−Σ pᵢ ln pᵢ`, one division fewer), plus the Miller–Madow
/// correction `(K′ − 1)/(2N)` with `K′` = number of NONZERO bins. `N = 0`
/// (empty histogram) → 0.0.
///
/// Why nonzero bins and not the raw alphabet length: the correction
/// compensates for classes the sample did NOT observe; counting empty bins
/// overstates it (Grassberger 2003 §3.2). On a fully populated histogram the
/// two readings coincide, and Research 609's uniform fixture is that case.
#[inline]
pub fn miller_madow_entropy(histogram: &[u16]) -> f64 {
    let mut total: u64 = 0;
    let mut k_nonzero: u64 = 0;
    let mut sum_c_ln_c: f64 = 0.0;
    for &c in histogram {
        if c == 0 {
            continue;
        }
        let c = u64::from(c);
        total += c;
        k_nonzero += 1;
        let cf = c as f64;
        sum_c_ln_c += cf * cf.ln();
    }
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let h_emp = n.ln() - sum_c_ln_c / n;
    h_emp + (k_nonzero as f64 - 1.0) / (2.0 * n)
}
