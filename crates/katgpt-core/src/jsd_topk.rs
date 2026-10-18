//! jsd_topk — NaN-safe bounded top-K Jensen–Shannon divergence kernel
//! (Issue 802 item 2, from Research 561 / arXiv:2609.15177 distill).
//!
//! `JSD = H(M) − ½H(P) − ½H(Q)` with `M = ½(P+Q)`, computed over the top-K
//! slice of each distribution RENORMALIZED to sum 1. The naive alternative —
//! KL over top-K zero-padded vectors — is +∞ on any disjoint support and one
//! NaN poisons every downstream gate; this kernel is bounded by construction:
//! disjoint top-K supports return EXACTLY ln 2, identical inputs return
//! exactly 0, and no input class produces NaN or ±∞.
//!
//! Prerequisite for Issue 802's commitment-gap calibration tables (1) and
//! the tri_mode graded verdicts (4). OPT-IN behind the `jsd_topk` feature.
//!
//! # Restriction semantics (per-vector top-K, union support)
//!
//! The top-K entries of P are taken by P's OWN values and the top-K of Q by
//! Q's OWN values; the union of the two index sets is the effective support.
//! Each restricted slice is renormalized to sum 1 (divide by the sum of the
//! selected K values); an index outside a side's selection contributes
//! exactly 0 restricted mass for that side, even if its raw entry is
//! positive. The kernel therefore measures the divergence of the RESTRICTED
//! objects — the Issue 802 gate semantics, where only the head of the
//! distribution matters and tail agreement is out of scope by construction.
//!
//! Ties at the K-th value are broken by the selection algorithm
//! (`select_nth_unstable_by` — deterministic for a fixed input, no
//! user-visible ordering guarantee among equal values). A tie only moves
//! which equal-valued filler entries join the union; fillers carry
//! restricted mass 0 on the selecting side and contribute nothing to the
//! divergence sum.
//!
//! # Exactness contract
//!
//! - **Disjoint restricted supports** (no union index carrying both
//!   restricted masses positive) → exactly `f32` ln 2. That predicate is the
//!   exact condition under which `H(M) = ½H(P) + ½H(Q) + ln 2` holds for
//!   normalized restricted masses, so the constant IS the mathematical value
//!   by construction of the predicate; the accumulated f32 sum only
//!   approaches it within rounding and is discarded when the predicate
//!   fires (the Issue 802 G1 test pins the bitwise equality against
//!   `f32::ln(2.0)`).
//! - **Identical inputs** → exactly `0.0`: each union term cancels exactly
//!   in f32 (see `jsd_term` — the ½-scalings are powers of two, so the
//!   shared-mass term is an exact `x − ½x − ½x`).
//! - **Both sides all-zero** → exactly `0.0` (no signal — documented
//!   convention: there is nothing to diverge).
//! - **One side all-zero** → exactly ln 2 via the same disjointness
//!   predicate: the surviving side's support shares nothing with the empty
//!   support. Honest caveat: the degenerate side is not a probability
//!   measure, so the textbook ln-2 bound does not strictly apply (the
//!   unnormalized reading would be `½H(Q) + ln 2`, which is unbounded and
//!   useless as a gate). The kernel pins the boundary constant because the
//!   Issue 802 gate consumer needs a bounded "no shared evidence" signal.
//!
//! # NaN safety
//!
//! Never returns NaN or ±∞ for finite non-negative inputs where at least
//! one restricted side has positive mass. Precondition: callers sanitize;
//! `debug_assert!` gates finiteness/non-negativity — the kernel does not
//! silently absorb garbage, and there is NO panic path in release beyond
//! the debug asserts: length mismatches truncate to the common prefix and
//! the index-encoding cap short-circuits to `0.0` (both documented below).
//! Release hardening: a non-positive or NaN restricted sum classifies as
//! the degenerate branch rather than propagating.
//!
//! # Zero-alloc contract
//!
//! [`jsd_topk_into`] is the hot path: caller-owned scratch, no heap
//! allocation. Scratch is used as an f32-ENCODED INDEX WORKSPACE (iota fill
//! → partial selection → ascending sort → two-pointer union merge); the
//! kernel never copies the input values, it reads masses by encoded index,
//! so scratch contents are clobbered and the scratch-size contract is
//! `>= input len`. Indices up to [`MAX_LEN`] = 2²⁴ are exactly representable
//! in f32; beyond the cap the kernel short-circuits to `0.0` (debug builds
//! assert instead). [`jsd_topk`] is the convenience form: it allocates two
//! `Vec`s per call, honestly and by design — allocation outside hot loops is
//! acceptable there; the G4 gate is on the INTO form.
//!
//! jsd_topk — NaN-safe bounded top-K Jensen–Shannon divergence kernel
//! (Issue 802 item 2): JSD = H(M) − ½H(P) − ½H(Q) over renormalized top-K
//! slices; disjoint supports → EXACTLY ln 2 (never the KL +∞/NaN trap that
//! poisons gates). Known-answer gates + throughput bench. Prerequisite for
//! Issue 802's commitment-gap calibration tables and graded tri_mode
//! verdicts. OPT-IN.

use core::f32::consts::LN_2;
use std::cmp::Ordering;

/// Largest input length the INTO form supports: indices must encode exactly
/// in f32 (every integer up to 2²⁴ is exact; beyond it the iota encoding
/// would round). Modern vocabularies are ≤ 2¹⁸ — this is a wide margin, not
/// a tuning knob. Debug builds assert; release short-circuits to `0.0`.
const MAX_LEN: usize = 1 << 24;

/// One union-support entry of the restricted JSD sum:
/// `½·pm·ln(pm) + ½·qm·ln(qm) − m·ln(m)` with `m = ½(pm+qm)` and the
/// `0·ln0 = 0` convention (positive guards below).
///
/// Sign note (Issue 802 spec precision): `JSD = H(M) − ½H(P) − ½H(Q)` and
/// `H(x) = −Σ x·ln x`, so the per-entry expansion is
/// `½ pm ln pm + ½ qm ln qm − m ln m` — the NEGATIVE of the naive
/// `m ln m − ½(pm ln pm + qm ln qm)` reading. Each term is ≥ 0 by convexity
/// of `x·ln x` (Jensen: `f(½pm+½qm) ≤ ½f(pm)+½f(qm)`), so a negative
/// computed term is pure f32 rounding noise and is clamped to `0.0` — that
/// clamp is a provably-safe rounding repair, not a semantic mask. For
/// identical inputs the term is an exact `x − ½x − ½x = 0` (½-scalings are
/// powers of two), which is what makes the identical-inputs → exactly-0.0
/// contract hold bitwise.
#[inline]
fn jsd_term(pm: f32, qm: f32) -> f32 {
    let m = 0.5 * (pm + qm);
    let mut t = 0.0f32;
    if pm > 0.0 {
        t += 0.5 * pm * pm.ln();
    }
    if qm > 0.0 {
        t += 0.5 * qm * qm.ln();
    }
    if m > 0.0 {
        t -= m * m.ln();
    }
    if t < 0.0 { 0.0 } else { t }
}

/// Fills `scratch[..src.len()]` with the encoded indices of `src`'s top-K,
/// sorted ascending, and returns the sum of the selected values.
///
/// Scratch doubles as the workspace: iota fill (index `i` encoded as `i as
/// f32` — exact up to [`MAX_LEN`]), `select_nth_unstable_by` partitions the
/// K largest to the front (descending-by-value comparator over the SOURCE
/// values at the encoded indices — the `mux::top_k` approach), then the
/// K-side is sorted ascending so the caller can two-pointer-merge the two
/// sides' index sets. The returned sum runs over ascending indices
/// (deterministic for a fixed input).
fn select_topk(src: &[f32], k: usize, scratch: &mut [f32]) -> f32 {
    let n = src.len();
    for (i, s) in scratch[..n].iter_mut().enumerate() {
        *s = i as f32;
    }
    // Partition so that scratch[..k] holds the K largest src-values
    // (unordered) as encoded indices.
    scratch[..n]
        .select_nth_unstable_by(k - 1, |a, b| src[*b as usize].total_cmp(&src[*a as usize]));
    let head = &mut scratch[..k];
    head.sort_unstable_by(|a, b| a.total_cmp(b));
    head.iter().map(|e| src[*e as usize]).sum()
}

/// NaN-safe bounded top-K JSD — zero-alloc INTO form (the hot path).
///
/// Writes the divergence to `*out` on every return path. Contract:
///
/// - `p.len() == q.len() == n`, `scratch_p.len() >= n`,
///   `scratch_q.len() >= n` (debug-asserted; release truncates to the
///   common prefix of all four lengths rather than panicking).
/// - `k` is clamped to `[1, n]` — `k = 0` means `k = 1`, `k > n` means
///   full support.
/// - Scratch contents are CLOBBERED (f32-encoded index workspace, NOT a
///   value copy). Inputs `p`/`q` are read-only.
/// - `n <= MAX_LEN` (debug-asserted; release short-circuits to `0.0`).
/// - Inputs must be finite and non-negative (debug-asserted; the kernel
///   does not silently absorb garbage, but nothing panics in release).
/// - No heap allocation anywhere on this path (the Issue 802 G4 gate).
///
/// Degenerate results (all documented in the module docs): both sides
/// carry no restricted mass → `0.0`; exactly one side does → ln 2;
/// disjoint restricted supports → ln 2 (bitwise `f32::ln(2.0)`); identical
/// inputs → `0.0` (bitwise).
pub fn jsd_topk_into(
    p: &[f32],
    q: &[f32],
    k: usize,
    scratch_p: &mut [f32],
    scratch_q: &mut [f32],
    out: &mut f32,
) {
    debug_assert_eq!(p.len(), q.len(), "jsd_topk_into: p/q length mismatch");
    debug_assert!(
        scratch_p.len() >= p.len() && scratch_q.len() >= q.len(),
        "jsd_topk_into: scratch smaller than input"
    );
    debug_assert!(
        p.len() <= MAX_LEN,
        "jsd_topk_into: length exceeds the f32 index-encoding cap (2^24)"
    );
    debug_assert!(
        p.iter().all(|&x| x.is_finite() && x >= 0.0)
            && q.iter().all(|&x| x.is_finite() && x >= 0.0),
        "jsd_topk_into: inputs must be finite and non-negative (sanitize upstream)"
    );

    // Release hardening: the misuses above must not panic — truncate to the
    // common prefix, refuse beyond the encoding cap. Both are documented.
    let n = p
        .len()
        .min(q.len())
        .min(scratch_p.len())
        .min(scratch_q.len());
    if n == 0 || n > MAX_LEN {
        *out = 0.0;
        return;
    }
    let k = k.clamp(1, n);

    let p_sum = select_topk(p, k, scratch_p);
    let q_sum = select_topk(q, k, scratch_q);

    // Degenerate restricted mass. `!(s > 0.0)` (not `s <= 0.0`) so a NaN
    // sum from garbage input classifies here instead of propagating.
    let p_ok = p_sum > 0.0;
    let q_ok = q_sum > 0.0;
    match (p_ok, q_ok) {
        // No signal on either side — nothing to diverge.
        (false, false) => {
            *out = 0.0;
            return;
        }
        // One-sided zero: the surviving support shares nothing with the
        // empty support — the disjointness predicate, pinned to the exact
        // boundary constant (module docs: honest-caveat note).
        (false, true) | (true, false) => {
            *out = LN_2;
            return;
        }
        (true, true) => {}
    }

    // Two-pointer merge over the two ascending encoded-index sets: the
    // union support has ≤ 2k entries, which is what bounds the SUM phase.
    // A P-only or Q-only index contributes 0 restricted mass on the other
    // side by definition of the restriction, so only a SHARED index can
    // carry both masses positive and break disjointness.
    let sel_p = &scratch_p[..k];
    let sel_q = &scratch_q[..k];
    let mut acc = 0.0f32;
    let mut disjoint = true;
    let mut i = 0usize;
    let mut j = 0usize;
    while i < k && j < k {
        let a = sel_p[i] as usize;
        let b = sel_q[j] as usize;
        match a.cmp(&b) {
            Ordering::Less => {
                acc += jsd_term(p[a] / p_sum, 0.0);
                i += 1;
            }
            Ordering::Greater => {
                acc += jsd_term(0.0, q[b] / q_sum);
                j += 1;
            }
            Ordering::Equal => {
                let pm = p[a] / p_sum;
                let qm = q[b] / q_sum;
                if pm > 0.0 && qm > 0.0 {
                    disjoint = false;
                }
                acc += jsd_term(pm, qm);
                i += 1;
                j += 1;
            }
        }
    }
    // Remainders are single-sided by construction (their index cannot be in
    // the other side's set — the merge exhausted one side); they can never
    // affect disjointness.
    while i < k {
        let a = sel_p[i] as usize;
        acc += jsd_term(p[a] / p_sum, 0.0);
        i += 1;
    }
    while j < k {
        let b = sel_q[j] as usize;
        acc += jsd_term(0.0, q[b] / q_sum);
        j += 1;
    }

    // Disjoint restricted supports: the mathematical value IS ln 2 by
    // construction of the predicate (H(M) = ½H(P) + ½H(Q) + ln 2 for
    // normalized masses) — pin the exactly-representable constant instead
    // of a float sum that merely approaches it (Issue 802 G1 pins bitwise
    // equality; the accumulated `acc` is discarded).
    *out = if disjoint { LN_2 } else { acc };
}

/// NaN-safe bounded top-K JSD — convenience form.
///
/// Allocates two `Vec<f32>` scratch buffers per call (honest allocation —
/// this form exists for cold paths, callers, and tests; allocation outside
/// hot loops is acceptable). The zero-alloc contract is [`jsd_topk_into`]:
/// the hot path must reuse caller-owned scratch. Results are bitwise
/// identical to the INTO form (same code path, same buffers).
pub fn jsd_topk(p: &[f32], q: &[f32], k: usize) -> f32 {
    let mut scratch_p = vec![0.0f32; p.len()];
    let mut scratch_q = vec![0.0f32; q.len()];
    let mut out = 0.0f32;
    jsd_topk_into(p, q, k, &mut scratch_p, &mut scratch_q, &mut out);
    out
}

/// Sets-form of the restricted-JSD kernel: both top-K sets arrive directly
/// as `(index, mass)` pairs — the caller already tracks each side's set
/// across steps (the Issue 802 item-3 inter-step drift shape), so the
/// selection half of [`jsd_topk_into`] is the caller's and only the merge
/// half runs here. Same bounded law and constants as the full forms:
/// each side is renormalized over its own set (the restricted-mass step);
/// disjoint restricted supports → bitwise [`LN_2`]; one-sided → [`LN_2`];
/// both sides massless → `0.0`; identical restricted mass → bitwise `0.0`
/// (the same ½-power-of-two exactness as [`jsd_topk_into`]).
///
/// # Contract
///
/// - `p`/`q` must be **sorted ascending by index, indices unique** — the
///   two-pointer merge relies on it (debug-asserted; a fixed-size tracker
///   sorts its sets before calling).
/// - Masses finite and non-negative (debug-asserted). Nothing panics in
///   release: a degenerate side sum reads as the one-sided/massless case
///   via `!(s > 0.0)`, the same NaN-classifying guard as the full form.
/// - Zero-alloc: writes `*out` on every return path, no heap anywhere.
pub fn jsd_topk_sets(p: &[(u32, f32)], q: &[(u32, f32)], out: &mut f32) {
    debug_assert!(
        p.windows(2).all(|w| w[0].0 < w[1].0) && q.windows(2).all(|w| w[0].0 < w[1].0),
        "jsd_topk_sets: index sets must be sorted ascending and unique"
    );
    debug_assert!(
        p.iter()
            .chain(q.iter())
            .all(|t| t.1.is_finite() && t.1 >= 0.0),
        "jsd_topk_sets: masses must be finite and non-negative"
    );

    let p_sum: f32 = p.iter().map(|t| t.1).sum();
    let q_sum: f32 = q.iter().map(|t| t.1).sum();
    let p_ok = p_sum > 0.0;
    let q_ok = q_sum > 0.0;
    match (p_ok, q_ok) {
        (false, false) => {
            *out = 0.0;
            return;
        }
        (false, true) | (true, false) => {
            *out = LN_2;
            return;
        }
        (true, true) => {}
    }

    // Two-pointer merge over the ascending index sets — the same union
    // walk as [`jsd_topk_into`]'s merge phase, with the sets given
    // directly instead of selected.
    let mut acc = 0.0f32;
    let mut disjoint = true;
    let (mut i, mut j) = (0usize, 0usize);
    while i < p.len() && j < q.len() {
        match p[i].0.cmp(&q[j].0) {
            Ordering::Less => {
                acc += jsd_term(p[i].1 / p_sum, 0.0);
                i += 1;
            }
            Ordering::Greater => {
                acc += jsd_term(0.0, q[j].1 / q_sum);
                j += 1;
            }
            Ordering::Equal => {
                let pm = p[i].1 / p_sum;
                let qm = q[j].1 / q_sum;
                if pm > 0.0 && qm > 0.0 {
                    disjoint = false;
                }
                acc += jsd_term(pm, qm);
                i += 1;
                j += 1;
            }
        }
    }
    while i < p.len() {
        acc += jsd_term(p[i].1 / p_sum, 0.0);
        i += 1;
    }
    while j < q.len() {
        acc += jsd_term(0.0, q[j].1 / q_sum);
        j += 1;
    }
    *out = if disjoint { LN_2 } else { acc };
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    fn one_hot(len: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        v[idx] = 1.0;
        v
    }

    fn normalize(v: &mut [f32]) {
        let s: f32 = v.iter().sum();
        assert!(s > 0.0);
        for x in v.iter_mut() {
            *x /= s;
        }
    }

    // ── 1. Disjoint supports → EXACTLY ln 2 (bitwise) ───────────────────────

    #[test]
    fn g1_disjoint_supports_return_exact_ln2() {
        let ln2 = f32::ln(2.0);
        // One-hot disjoint pair, several K (incl. k = len: the zero-filler
        // entries both sides select carry restricted mass 0 and must NOT
        // break disjointness).
        for k in [1usize, 4, 16] {
            let p = one_hot(16, 3);
            let q = one_hot(16, 7);
            let v = jsd_topk(&p, &q, k);
            assert_eq!(v, ln2, "one-hot disjoint k={k}: {v} != {ln2}");
            assert_eq!(v, LN_2, "kernel constant must equal f32::ln(2.0)");
        }
        // Vocab-scale layout: disjoint spiky heads at opposite ends.
        let n = 4096;
        let mut p = vec![0.0f32; n];
        let mut q = vec![0.0f32; n];
        let mut rng = fastrand::Rng::with_seed(0x802_0001);
        for i in 0..32 {
            p[i] = 1.0 + rng.f32();
            q[n - 32 + i] = 1.0 + rng.f32();
        }
        let v = jsd_topk(&p, &q, 64);
        assert_eq!(v, ln2, "vocab-scale disjoint k=64: {v} != {ln2}");
        // Same pair through the INTO form — identical constant.
        let mut sp = vec![0.0f32; n];
        let mut sq = vec![0.0f32; n];
        let mut out = 0.0f32;
        jsd_topk_into(&p, &q, 64, &mut sp, &mut sq, &mut out);
        assert_eq!(out, ln2);
    }

    // ── 2. Identical vectors → exactly 0.0 ──────────────────────────────────

    #[test]
    fn g1_identical_vectors_return_exact_zero() {
        let mut v = vec![0.0f32; 32];
        let mut rng = fastrand::Rng::with_seed(0x802_0002);
        for x in v.iter_mut() {
            *x = rng.f32() + 0.01;
        }
        normalize(&mut v);
        for k in [1usize, 8, 32] {
            assert_eq!(jsd_topk(&v, &v, k), 0.0, "identical k={k}");
        }
        // k-clamp semantics: 0 → 1, over-length → len (bitwise equal).
        assert_eq!(jsd_topk(&v, &v, 0), jsd_topk(&v, &v, 1));
        assert_eq!(jsd_topk(&v, &v, 10_000), jsd_topk(&v, &v, 32));
    }

    // ── 3. The KL trap: disjoint one-hots must NOT NaN/+∞ ───────────────────

    #[test]
    fn g1_kl_trap_disjoint_one_hots_are_bounded() {
        // This is the test that justifies the kernel: KL over zero-padded
        // top-K vectors is +∞ here and any NaN poisons the downstream gate
        // (Research 561 §2.2 — JSD's ln-2 bound is the whole reason TSD
        // picked it over reverse-KL, whose blowup is the NFE-saturation
        // signature).
        let ln2 = f32::ln(2.0);
        let n = 1024;
        for (i, j) in [(0usize, 1usize), (5, 900), (1023, 0)] {
            for k in [1usize, 8, n] {
                let p = one_hot(n, i);
                let q = one_hot(n, j);
                let v = jsd_topk(&p, &q, k);
                assert!(!v.is_nan(), "NaN at i={i} j={j} k={k}");
                assert!(!v.is_infinite(), "+inf at i={i} j={j} k={k}");
                assert_eq!(v, ln2, "disjoint one-hots must pin ln 2 exactly");
            }
        }
    }

    // ── 4. Symmetry within 1e-7 ─────────────────────────────────────────────

    #[test]
    fn g1_symmetry_within_1e_7() {
        let n = 512;
        let mut p = vec![0.0f32; n];
        let mut q = vec![0.0f32; n];
        let mut rng = fastrand::Rng::with_seed(0x802_0004);
        for (side, v) in [(0, &mut p), (1, &mut q)] {
            let _ = side;
            // Spiky: a few large heads over small noise.
            for x in v.iter_mut() {
                *x = rng.f32() * 0.01;
            }
            for _ in 0..5 {
                v[rng.usize(..n)] += 1.0 + rng.f32();
            }
            normalize(v);
        }
        for k in [1usize, 8, 64, 512] {
            let fw = jsd_topk(&p, &q, k);
            let bw = jsd_topk(&q, &p, k);
            assert!((fw - bw).abs() <= 1e-7, "asymmetry at k={k}: {fw} vs {bw}");
        }
    }

    // ── 5. Bound 0 ≤ JSD ≤ ln 2 over ~10⁴ seeded random pairs ───────────────

    fn fill_class(kind: usize, rng: &mut fastrand::Rng, v: &mut [f32]) {
        let n = v.len();
        match kind {
            0 => {
                // dense softmax-ish
                let mut s = 0.0f32;
                for x in v.iter_mut() {
                    let e = (rng.f32() * 6.0 - 3.0).exp();
                    *x = e;
                    s += e;
                }
                for x in v.iter_mut() {
                    *x /= s;
                }
            }
            1 => {
                // sparse: 8 nonzero entries
                for x in v.iter_mut() {
                    *x = 0.0;
                }
                let mut s = 0.0f32;
                for _ in 0..8 {
                    let e = 0.1 + rng.f32();
                    v[rng.usize(..n)] += e;
                    s += e;
                }
                assert!(s > 0.0);
            }
            2 => {
                // dirac
                for x in v.iter_mut() {
                    *x = 0.0;
                }
                v[rng.usize(..n)] = 1.0;
            }
            3 => {
                // spiky: 3 heads over noise
                for x in v.iter_mut() {
                    *x = rng.f32() * 0.01;
                }
                for _ in 0..3 {
                    v[rng.usize(..n)] += 5.0 + rng.f32() * 20.0;
                }
                normalize(v);
            }
            _ => {
                // uniform
                let u = 1.0 / n as f32;
                for x in v.iter_mut() {
                    *x = u;
                }
            }
        }
    }

    #[test]
    fn g2_bounded_over_seeded_random_classes() {
        let ln2 = f32::ln(2.0);
        let n = 1024usize;
        let ks = [1usize, 8, 64, 4096]; // 4096 > n: exercises the k-clamp
        let pairs_per_cell = 625usize; // 5 classes × 4 Ks × 625 = 12.5k calls
        let mut rng = fastrand::Rng::with_seed(0x802_0005);
        let mut p = vec![0.0f32; n];
        let mut q = vec![0.0f32; n];
        for kind in 0..5 {
            for &k in &ks {
                for _ in 0..pairs_per_cell {
                    fill_class(kind, &mut rng, &mut p);
                    fill_class(kind, &mut rng, &mut q);
                    let v = jsd_topk(&p, &q, k);
                    assert!(
                        !v.is_nan() && v >= 0.0 && v <= ln2 + TOL,
                        "class={kind} k={k}: out of [0, ln2]: {v}"
                    );
                }
            }
        }
    }

    // ── 6. Monotone sanity: one-hot > two-hot vs uniform (NOT a theorem) ────

    #[test]
    fn sanity_one_hot_more_divergent_than_two_hot() {
        // Sanity, not a theorem (Issue 802): a sharper head against the same
        // uniform must read as MORE divergent at the same K. The uniform's
        // top entries are pinned deterministically (indices 0/1 elevated to
        // 2/64, the other 60 at 1/64 — sums to exactly 1) so the union
        // support cannot become disjoint by tie-break luck.
        let n = 64usize;
        let k = 8usize;
        let mut u = vec![1.0f32 / 64.0; n];
        u[0] = 2.0 / 64.0;
        u[1] = 2.0 / 64.0;
        let p1 = one_hot(n, 0);
        let mut p2 = vec![0.0f32; n];
        p2[0] = 0.5;
        p2[1] = 0.5;
        let d1 = jsd_topk(&p1, &u, k);
        let d2 = jsd_topk(&p2, &u, k);
        assert!(d1 > d2, "one-hot {d1} must exceed two-hot {d2}");
        assert!(!d1.is_nan() && !d2.is_nan());
    }

    // ── 7. Degenerate sides ─────────────────────────────────────────────────

    #[test]
    fn g1_degenerate_sides() {
        let ln2 = f32::ln(2.0);
        let mut q = vec![0.0f32; 64];
        q[7] = 0.7;
        q[9] = 0.3;
        let zeros = vec![0.0f32; 64];
        // Both all-zero → 0.0 (no signal).
        assert_eq!(jsd_topk(&zeros, &zeros, 8), 0.0);
        // P all-zero, Q positive: restricted P carries no mass → supports
        // trivially disjoint → exactly ln 2 (the predicate path, both Ks).
        for k in [1usize, 8, 64] {
            assert_eq!(jsd_topk(&zeros, &q, k), ln2, "P-zero k={k}");
            assert_eq!(jsd_topk(&q, &zeros, k), ln2, "Q-zero k={k}");
        }
        // Empty inputs → 0.0 (release-hardened, no panic).
        let e: Vec<f32> = Vec::new();
        let mut scratch_a: Vec<f32> = Vec::new();
        let mut scratch_b: Vec<f32> = Vec::new();
        let mut out = f32::NAN;
        jsd_topk_into(&e, &e, 4, &mut scratch_a, &mut scratch_b, &mut out);
        assert_eq!(out, 0.0);
    }

    // ── 8. Renormalization: scale-invariance ────────────────────────────────

    #[test]
    fn g1_scale_invariance() {
        let n = 256usize;
        let mut p = vec![0.0f32; n];
        let mut q = vec![0.0f32; n];
        let mut rng = fastrand::Rng::with_seed(0x802_0008);
        for v in [&mut p, &mut q] {
            for x in v.iter_mut() {
                *x = rng.f32() * 0.1;
            }
            v[rng.usize(..n)] += 1.0;
            normalize(v);
        }
        for &k in &[1usize, 8, 64] {
            let base = jsd_topk(&p, &q, k);
            for &s in &[1e-3f32, 0.5, 2.0, 1e3] {
                let sp: Vec<f32> = p.iter().map(|&x| x * s).collect();
                let sq: Vec<f32> = q.iter().map(|&x| x * s).collect();
                let v = jsd_topk(&sp, &sq, k);
                assert!((v - base).abs() <= TOL, "scale {s} k={k}: {v} vs {base}");
            }
        }
    }

    // ── 9. G4 witness: INTO form on caller buffers ──────────────────────────

    #[test]
    fn g4_into_form_caller_buffers_zero_alloc() {
        // Stack buffers only — the code-review witness is that jsd_topk_into
        // takes caller-owned scratch and heap-allocates nothing; this test
        // exercises exactly that shape at K=64, len=1024.
        let n = 1024usize;
        let k = 64usize;
        let mut p = vec![0.0f32; n];
        let mut q = vec![0.0f32; n];
        let mut rng = fastrand::Rng::with_seed(0x802_0009);
        for v in [&mut p, &mut q] {
            for x in v.iter_mut() {
                *x = rng.f32() * 0.05;
            }
            for _ in 0..6 {
                v[rng.usize(..n)] += 1.0 + rng.f32();
            }
            normalize(v);
        }
        let mut sp = [0.0f32; 1024];
        let mut sq = [0.0f32; 1024];
        let mut out = f32::NAN;
        jsd_topk_into(&p, &q, k, &mut sp, &mut sq, &mut out);
        // Bitwise identical to the convenience form (same code path).
        assert_eq!(out, jsd_topk(&p, &q, k));
        // Determinism: a second run over re-clobbered scratch is bit-stable.
        let mut out2 = f32::NAN;
        jsd_topk_into(&p, &q, k, &mut sp, &mut sq, &mut out2);
        assert_eq!(out, out2);
        // Disjoint pair through the INTO form → exact constant.
        let a = one_hot(n, 11);
        let b = one_hot(n, 900);
        let mut out3 = f32::NAN;
        jsd_topk_into(&a, &b, k, &mut sp, &mut sq, &mut out3);
        assert_eq!(out3, f32::ln(2.0));
    }

    // ── 10. Sets-form: the Issue-802 item-3 inter-step drift shape ────────

    #[test]
    fn g1_sets_disjoint_one_sided_massless_identical() {
        let ln2 = f32::ln(2.0);
        let p = [(3u32, 0.7f32), (5, 0.2), (9, 0.1)];
        let q = [(1u32, 0.5f32), (4, 0.3), (7, 0.2)];
        let mut out = f32::NAN;
        // Disjoint supports → bitwise ln 2.
        jsd_topk_sets(&p, &q, &mut out);
        assert_eq!(out, ln2);
        // Identical restricted mass → bitwise 0.
        jsd_topk_sets(&p, &p, &mut out);
        assert_eq!(out, 0.0);
        // One-sided → ln 2 (empty set carries no restricted mass).
        jsd_topk_sets(&p, &[], &mut out);
        assert_eq!(out, ln2);
        jsd_topk_sets(&[], &q, &mut out);
        assert_eq!(out, ln2);
        // Both sides massless → 0 (nothing to diverge).
        jsd_topk_sets(&[], &[], &mut out);
        assert_eq!(out, 0.0);
        // Zero masses on a non-empty set read as massless too.
        let zero = [(3u32, 0.0f32), (5, 0.0)];
        jsd_topk_sets(&zero, &q, &mut out);
        assert_eq!(out, ln2);
    }

    #[test]
    fn g1_sets_agree_with_full_kernel_on_the_same_restriction() {
        // A spiky pair whose top-3 the full kernel selects is exactly the
        // hand set: both forms measure the same restricted JSD (the
        // renormalization sums associate differently → tolerance, not
        // bitwise).
        let n = 64;
        let mut p = vec![0.0f32; n];
        let mut q = vec![0.0f32; n];
        p[3] = 0.7;
        p[5] = 0.2;
        p[9] = 0.1;
        q[3] = 0.5;
        q[5] = 0.25;
        q[11] = 0.25;
        let full = jsd_topk(&p, &q, 3);
        let sp = [(3u32, 0.7f32), (5, 0.2), (9, 0.1)];
        let sq = [(3u32, 0.5f32), (5, 0.25), (11, 0.25)];
        let mut sets = f32::NAN;
        jsd_topk_sets(&sp, &sq, &mut sets);
        assert!(
            (full - sets).abs() <= TOL,
            "sets form {sets} vs full kernel {full}"
        );
        // Not the disjoint constant: the sets share indices 3 and 5.
        assert!(sets < f32::ln(2.0));
    }

    #[test]
    fn g4_sets_form_is_deterministic_and_never_nan() {
        let p = [(3u32, 0.7f32), (5, 0.2), (9, 0.1)];
        let q = [(3u32, 0.5f32), (5, 0.25), (11, 0.25)];
        let mut a = f32::NAN;
        let mut b = f32::NAN;
        jsd_topk_sets(&p, &q, &mut a);
        jsd_topk_sets(&p, &q, &mut b);
        assert_eq!(a, b, "deterministic");
        assert!(a.is_finite() && a >= 0.0 && a <= f32::ln(2.0));
    }
}
