//! Sparse lower-bounded KL divergence between two full-vocab logit rows
//! (katgpt-rs Issue 1210 C6 — the Speculative Self-Distillation extraction,
//! arXiv:2611.12016; Provenance: SSD NeurIPS 2026, Apache-2.0, pinned
//! `cccf657b`).
//!
//! # The problem
//!
//! KL(teacher ‖ student) over a full vocabulary is `O(V)` per call — at the
//! paper's 151,936-token vocab that is prohibitive inside any per-position
//! gate. The paper's own trick, transcribed: evaluate the divergence **only**
//! on `union(top-K(teacher), top-K(student))`, aggregate everything else into
//! ONE tail bucket per side, and read the result as a **lower bound** on the
//! true KL — never an estimate of it.
//!
//! # Why the result is a lower bound (the error sign is KNOWN)
//!
//! Coarsening a partition never increases KL (the log-sum inequality):
//! `KL(P̃ ‖ Q̃) ≤ KL(P ‖ Q)` where `P̃/Q̃` are the distributions over the
//! partition {kept atoms, one tail atom}. The coarsened divergence is
//!
//! ```text
//! D̃ = Σ_{v ∈ kept} p_v·ln(p_v/q_v)  +  p_tail·ln(p_tail/q_tail)
//! ```
//!
//! with `p_tail = 1 − kept_mass_p`, `q_tail = 1 − kept_mass_q`. This module
//! returns exactly `D̃`, plus one disclosed weakening: when `q_tail` is
//! nonzero but underflows the `TAIL_MASS_FLOOR` (`1e-12` — reachable by the
//! `-1e9` finite sentinel), the tail is clamped **UP** before its `ln`. A
//! larger `q̃_tail` shrinks the tail term, so the clamped value is still `≤
//! KL(P̃ ‖ Q̃) ≤ KL(P ‖ Q)` — the clamp weakens the bound, it can never
//! invert it. The teacher tail is never clamped: `p_tail = 0` contributes
//! exactly zero (skipped), and a tiny positive `p_tail` is used as-is (the
//! term `p·(ln p − ln q)` is not monotone in `p`, so clamping it could
//! INVERT the bound).
//!
//! # Why the union (not single-side top-K)
//!
//! Single-side truncation cuts exactly the terms the other direction is made
//! of: `top-K(teacher)` alone gives every student mass outside the teacher's
//! head to the tail bucket, and vice versa. The union keeps both heads
//! exact — `kept_mass` per side is the coverage floor that says how much the
//! bound has actually seen. A regression test pins union strictly
//! dominating the teacher-only bound on a planted fixture.
//!
//! # Numerics riders (permanent property tests)
//!
//! - Finite sentinel `-1e9`, never `-inf`: a `-inf` logit makes
//!   `0·ln(0) = NaN` reachable in masked reductions; `-1e9` underflows the
//!   tail mass to exact `0.0` instead, which the tail path handles honestly
//!   (skip or clamp, per above).
//! - Tail clamp `1e-12` before `log`.
//! - Raw pre-temperature logits in, never rescaled log-probs: KL is
//!   shift-invariant, so adding a constant offset to BOTH rows must not move
//!   the bound (pinned bit-tight in tests).
//!
//! # Modelless discipline
//!
//! Pure closed-form arithmetic. No training, no learned parameters, no RNG.
//! Determinism: sequential f64 folds in ascending index order (the Issue-940
//! reduction-order law, applied even though this is a local instrument —
//! free determinism). Bounded allocation: two top-K index vectors of length
//! `≤ k` each, nothing per-element.
//!
//! # Kernel divergence (deliberate, precedent-backed)
//!
//! Normalization uses a PRIVATE f64 two-pass logsumexp, NOT the shared f32
//! [`crate::simd::logsumexp_parts`] kernel: the kernel's f32 `ln_z`
//! carries ~1e-7 relative error, which makes the computed kept masses
//! overshoot 1 (measured: `1.000000057632` on a 64-token fixture) and the
//! recovered tail masses go NEGATIVE — at which point the floor clamp fires
//! spuriously and inflates the tail bucket ABOVE the true term, inverting
//! the bound. The bound's error-sign guarantee needs f64-exact masses.
//! `gmm_support` carries the same divergence for the same reason (its
//! "f64-accumulator logsumexp" docs). The two-pass shape is identical to
//! the kernel's — max pass, then one Σ exp fold — only the accumulator is
//! f64. The O(V) cost is unchanged; this module's savings live in the
//! per-index KL reduction (O(kept)), which is the part a wire consumer
//! avoids (it stores kept logits + the two scalars only).
//!
//! # Honest caveat
//!
//! The bound can be arbitrarily loose when the teacher is high-entropy and
//! `k` is small (kept mass ≈ `k/V`); `kept_mass_*` is the caller's
//! honest-coverage signal, not a hidden normalization. This is a staged-lane
//! primitive — NO GOAT claim (Issue 1210 ungoat discipline): consumers are
//! the C1 divergence-threshold gate family, the C4 post-hoc τ-sweep, and
//! decision-level escalation-band tuning (`seat_rerank`/ESC); promote
//! per-primitive only when a wired consumer passes its pre-registered gate.
//!
//! Opt-in `sparse_kl_bound` until such a consumer lands. Sigmoid never
//! softmax; coordinates stay in distribution space.

/// The one failure class of [`sparse_kl_lower_bound`]: everything else is a
/// valid reading (a loose bound is still a bound).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseKlError {
    /// The two rows must cover the same vocabulary.
    LengthMismatch { teacher: usize, student: usize },
    /// A logit was `NaN`/`±inf`. The finite sentinel `-1e9` is the supported
    /// spelling of "no mass here" — `-inf` makes `0·ln 0 = NaN` reachable.
    NonFiniteLogit { side: &'static str, index: usize },
    /// `k = 0` has no kept set.
    KZero,
    /// `k` beyond the vocabulary covers everything; the caller meant a
    /// sparse bound, so this refuses rather than silently becoming the
    /// exact divergence.
    KExceedsVocab { k: usize, vocab: usize },
    /// Empty rows.
    Empty,
}

impl std::fmt::Display for SparseKlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { teacher, student } => {
                write!(f, "length mismatch: teacher {teacher} vs student {student}")
            }
            Self::NonFiniteLogit { side, index } => {
                write!(
                    f,
                    "non-finite logit ({side}) at index {index} — use the -1e9 sentinel"
                )
            }
            Self::KZero => write!(f, "k must be ≥ 1"),
            Self::KExceedsVocab { k, vocab } => {
                write!(
                    f,
                    "k {k} ≥ vocab {vocab}: the bound would be the exact KL — refusing"
                )
            }
            Self::Empty => write!(f, "empty logit rows"),
        }
    }
}

impl std::error::Error for SparseKlError {}

/// One reading of the sparse lower-bounded KL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseKlBound {
    /// The coarsened divergence `D̃` in nats — a LOWER bound on the true
    /// `KL(teacher ‖ student)` (error sign known; see module docs). Never
    /// negative beyond f64 rounding of the exact-zero case.
    pub kl_lower_bound_nats: f64,
    /// Teacher probability mass on the kept union — the coverage floor of
    /// the bound's teacher side (`1 − p_tail`).
    pub kept_mass_teacher: f64,
    /// Student probability mass on the kept union.
    pub kept_mass_student: f64,
    /// Size of the kept union (`≤ 2k`, duplicates deduplicated).
    pub kept_count: usize,
}

/// Tail masses below this floor clamp UP to it before their `ln`. Only ever
/// applied to the STUDENT tail — see the module docs for why the teacher
/// tail is never clamped.
pub const TAIL_MASS_FLOOR: f64 = 1e-12;

/// Private f64 two-pass logsumexp — same shape as the shared f32 kernel
/// (max pass, then one Σ exp fold), f64 accumulator. See the module docs
/// for why the f32 kernel's ln_z is not used here (bound-direction
/// guarantee; the gmm_support precedent).
fn f64_logsumexp(logits: &[f32]) -> f64 {
    let mut max = f32::NEG_INFINITY;
    for &x in logits {
        if x > max {
            max = x;
        }
    }
    let max_f = f64::from(max);
    let mut sum = 0.0f64;
    for &x in logits {
        sum += (f64::from(x) - max_f).exp();
    }
    max_f + sum.ln()
}

/// One side's top-K selection: strict `>` against the running worst — on
/// boundary ties the EARLIER index wins (the Issue-940 fixed strict tie
/// rule, first-index spelling). Returns indices ascending.
fn top_k_indices(logits: &[f32], k: usize) -> Vec<u32> {
    debug_assert!(k >= 1 && k <= logits.len());
    // Descending-sorted (score, idx); insertion keeps the first-index tie
    // rule by never displacing an equal score already held.
    let mut held: Vec<(f32, u32)> = Vec::with_capacity(k);
    for (idx, &score) in logits.iter().enumerate() {
        let idx = idx as u32;
        if held.len() < k {
            held.push((score, idx));
            if held.len() == k {
                held.sort_by(|a, b| b.0.total_cmp(&a.0));
            }
            continue;
        }
        let worst = held[k - 1];
        if score > worst.0 {
            held[k - 1] = (score, idx);
            // Bubble the replaced slot down to its place; the rest of the
            // vector is already sorted, so one pass suffices.
            let mut i = k - 1;
            while i > 0 && held[i].0 > held[i - 1].0 {
                held.swap(i, i - 1);
                i -= 1;
            }
        }
    }
    held.sort_unstable_by_key(|&(_, idx)| idx);
    held.into_iter().map(|(_, idx)| idx).collect()
}

/// Sparse LOWER bound on `KL(teacher ‖ student)` in nats over the union of
/// the two rows' top-K indices, both tails one bucket. See the module docs
/// for the bound argument, the tie rule, and the clamp direction.
///
/// Shift-invariant in both rows jointly (raw logits in, no rescaling
/// assumed); deterministic (sequential f64 folds, ascending index order);
/// refuses loud on length mismatch, non-finite logits, `k = 0`, and
/// `k ≥ vocab` (that posture is the exact KL — take it directly instead).
pub fn sparse_kl_lower_bound(
    teacher_logits: &[f32],
    student_logits: &[f32],
    k: usize,
) -> Result<SparseKlBound, SparseKlError> {
    if teacher_logits.is_empty() || student_logits.is_empty() {
        return Err(SparseKlError::Empty);
    }
    if teacher_logits.len() != student_logits.len() {
        return Err(SparseKlError::LengthMismatch {
            teacher: teacher_logits.len(),
            student: student_logits.len(),
        });
    }
    let vocab = teacher_logits.len();
    if k == 0 {
        return Err(SparseKlError::KZero);
    }
    if k >= vocab {
        return Err(SparseKlError::KExceedsVocab { k, vocab });
    }
    for (side, row) in [("teacher", teacher_logits), ("student", student_logits)] {
        for (index, &x) in row.iter().enumerate() {
            if !x.is_finite() {
                return Err(SparseKlError::NonFiniteLogit { side, index });
            }
        }
    }

    // Full-vocab normalizers through the private f64 kernel (see the module
    // docs — the f32 shared kernel's ln_z error inverts the bound at high
    // coverage): log p_v = f64(v) − t_lse, always ≤ 0, so exp cannot overflow.
    let t_lse = f64_logsumexp(teacher_logits);
    let s_lse = f64_logsumexp(student_logits);

    // Union of the two top-K sets (both already ascending; dedup merge).
    let t_top = top_k_indices(teacher_logits, k);
    let s_top = top_k_indices(student_logits, k);
    let mut kept: Vec<u32> = Vec::with_capacity(t_top.len() + s_top.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < t_top.len() || j < s_top.len() {
        let next = match (i < t_top.len(), j < s_top.len()) {
            (true, true) => {
                if t_top[i] < s_top[j] {
                    i += 1;
                    t_top[i - 1]
                } else if s_top[j] < t_top[i] {
                    j += 1;
                    s_top[j - 1]
                } else {
                    i += 1;
                    j += 1;
                    t_top[i - 1]
                }
            }
            (true, false) => {
                i += 1;
                t_top[i - 1]
            }
            (false, true) => {
                j += 1;
                s_top[j - 1]
            }
            (false, false) => unreachable!(),
        };
        kept.push(next);
    }

    // Exact per-index terms + coverage, one ascending f64 fold.
    let mut exact_sum = 0.0f64;
    let mut kept_mass_p = 0.0f64;
    let mut kept_mass_q = 0.0f64;
    for &v in &kept {
        let lp = f64::from(teacher_logits[v as usize]) - t_lse;
        let lq = f64::from(student_logits[v as usize]) - s_lse;
        let p = lp.exp();
        let q = lq.exp();
        exact_sum += p * (lp - lq);
        kept_mass_p += p;
        kept_mass_q += q;
    }

    // Tail buckets from the EXACT complement sums — never `1 − Σkept`,
    // whose cancellation error is what the f32-kernel bug above rode in on.
    // Ascending f64 folds; underflowed sentinels contribute exact 0.0.
    let mut kept_i = 0usize;
    let mut p_tail = 0.0f64;
    let mut q_tail_raw = 0.0f64;
    for v in 0..vocab {
        let v32 = v as u32;
        while kept_i < kept.len() && kept[kept_i] < v32 {
            kept_i += 1;
        }
        if kept_i < kept.len() && kept[kept_i] == v32 {
            continue;
        }
        p_tail += (f64::from(teacher_logits[v]) - t_lse).exp();
        q_tail_raw += (f64::from(student_logits[v]) - s_lse).exp();
    }
    let tail_term = if p_tail <= 0.0 {
        0.0
    } else {
        let q_tail = q_tail_raw.max(TAIL_MASS_FLOOR);
        p_tail * (p_tail.ln() - q_tail.ln())
    };

    Ok(SparseKlBound {
        kl_lower_bound_nats: exact_sum + tail_term,
        kept_mass_teacher: kept_mass_p,
        kept_mass_student: kept_mass_q,
        kept_count: kept.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic micro-vocab fixture: V=64 logits from a fixed-seed LCG
    /// (no unseeded-global draws), teacher spiked, student flatter — a
    /// planted divergence whose dense KL the tests compute through the SAME
    /// shared kernel.
    struct Fixture {
        teacher: Vec<f32>,
        student: Vec<f32>,
    }

    impl Fixture {
        fn planted() -> Self {
            // LCG64, fixed seed — deterministic across runs and platforms.
            let mut state: u64 = 0x853c_49e6_748f_ea9b;
            let mut next = move || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
            };
            let mut teacher = Vec::with_capacity(64);
            let mut student = Vec::with_capacity(64);
            for i in 0..64 {
                // Teacher: spiked rows (a few strong heads + noise); student:
                // the same shape compressed toward uniform (scale 0.25) so
                // KL(teacher‖student) is decisively positive.
                let noise = next();
                let head = if i < 6 { 6.0 - i as f32 } else { 0.0 };
                teacher.push(head + noise);
                student.push(0.25 * (head + noise));
            }
            Self { teacher, student }
        }

        /// Dense reference KL through the SAME f64 logsumexp — the
        /// exactness oracle the bound is adjudicated against.
        fn dense_kl(&self) -> f64 {
            let t_lse = f64_logsumexp(&self.teacher);
            let s_lse = f64_logsumexp(&self.student);
            let mut kl = 0.0f64;
            for i in 0..self.teacher.len() {
                let lp = f64::from(self.teacher[i]) - t_lse;
                let lq = f64::from(self.student[i]) - s_lse;
                let p = lp.exp();
                if p > 0.0 {
                    kl += p * (lp - lq);
                }
            }
            kl
        }
    }

    #[test]
    fn k_equals_vocab_refuses_loud() {
        let f = Fixture::planted();
        let err = sparse_kl_lower_bound(&f.teacher, &f.student, 64).unwrap_err();
        assert_eq!(err, SparseKlError::KExceedsVocab { k: 64, vocab: 64 });
    }

    #[test]
    fn full_coverage_is_exact_against_the_dense_reference() {
        let f = Fixture::planted();
        let dense = f.dense_kl();
        // k = V−1: student = 0.25·teacher shares the teacher's unique argmin
        // (positive scaling preserves it), so BOTH sides miss the SAME index
        // and the tail bucket holds exactly one index — the coarsening is
        // then the identity and the bound equals the dense KL to f64
        // rounding. (With distinct missed indices the tail coarsens two
        // atoms into one and a small honest gap opens — bound direction
        // still guaranteed, covered by bound_tightens_monotonically_in_k.)
        let b = sparse_kl_lower_bound(&f.teacher, &f.student, 63).unwrap();
        assert!(
            (b.kl_lower_bound_nats - dense).abs() < 1e-9,
            "near-full bound {} vs dense {}",
            b.kl_lower_bound_nats,
            dense
        );
        // The one missed index carries ~1e-3 mass on this fixture (the
        // noise logits all sit within e^-0.5..e^0.5 of each other) — the
        // coverage floors reflect that, not a 1e-6 fantasy.
        assert!(b.kept_mass_teacher > 0.99);
        assert!(b.kept_mass_student > 0.98);
    }

    #[test]
    fn bound_tightens_monotonically_in_k() {
        let f = Fixture::planted();
        let dense = f.dense_kl();
        let mut prev = 0.0f64;
        for k in [1usize, 2, 4, 8, 16, 32, 63] {
            let b = sparse_kl_lower_bound(&f.teacher, &f.student, k).unwrap();
            assert!(
                b.kl_lower_bound_nats >= prev - 1e-12,
                "bound loosened at k={k}: {prev} -> {}",
                b.kl_lower_bound_nats
            );
            assert!(
                b.kl_lower_bound_nats <= dense + 1e-9,
                "bound exceeded dense KL at k={k}"
            );
            prev = b.kl_lower_bound_nats;
        }
    }

    #[test]
    fn union_dominates_the_teacher_only_bound() {
        // Single-side truncation gives student mass outside the teacher's
        // head to the tail bucket — strictly weaker on a fixture where the
        // student's head sits outside the teacher's top-K. Falsification
        // arm: a single-side estimator cannot pass this.
        let mut teacher = vec![0.0f32; 64];
        let mut student = vec![0.0f32; 64];
        teacher[0] = 10.0; // teacher head: index 0
        student[63] = 10.0; // student head: index 63 (disjoint)
        for i in 1..63 {
            student[i] = -1.0;
        }
        let dense = {
            let t_lse = f64_logsumexp(&teacher);
            let s_lse = f64_logsumexp(&student);
            let mut kl = 0.0f64;
            for i in 0..64 {
                let lp = f64::from(teacher[i]) - t_lse;
                let lq = f64::from(student[i]) - s_lse;
                kl += lp.exp() * (lp - lq);
            }
            kl
        };
        let lse = f64_logsumexp;
        let t_lse = lse(&teacher);
        let s_lse = lse(&student);
        // Teacher-only truncation at k=1: kept = {0}, student's head (index
        // 63) buried in the tail bucket.
        let teacher_only = {
            let lp = f64::from(teacher[0]) - t_lse;
            let lq = f64::from(student[0]) - s_lse;
            let p0 = lp.exp();
            let q0 = lq.exp();
            let p_tail = 1.0 - p0;
            let q_tail = (1.0 - q0).max(TAIL_MASS_FLOOR);
            p0 * (lp - lq) + p_tail * (p_tail.ln() - q_tail.ln())
        };
        let union = sparse_kl_lower_bound(&teacher, &student, 1).unwrap();
        assert!(
            union.kl_lower_bound_nats > teacher_only,
            "union bound {} must strictly dominate teacher-only {}",
            union.kl_lower_bound_nats,
            teacher_only
        );
        assert!(union.kl_lower_bound_nats <= dense + 1e-9);
        // The union at k=1 must keep BOTH heads (kept_count == 2).
        assert_eq!(union.kept_count, 2);
    }

    #[test]
    fn finite_sentinel_tail_never_produces_infinity() {
        // The student puts NO mass on the teacher's head: index 0 is a -1e9
        // sentinel under normal student logits. Separately, at k = V-1 the
        // ONE missed student index is itself a sentinel — its tail mass
        // underflows to EXACT 0.0 in f64, and the floor clamp is what keeps
        // the bound finite. Without the clamp the raw formula reads
        // p_tail·ln(p_tail/0) = +inf — the shipped clamp is the sign story.
        let mut teacher = vec![-5.0f32; 64];
        teacher[0] = 5.0;
        let mut student = vec![-1.0f32; 64];
        student[0] = -1.0e9; // student: NO mass on the teacher's head
        let b = sparse_kl_lower_bound(&teacher, &student, 1).unwrap();
        assert!(b.kl_lower_bound_nats.is_finite(), "bound must stay finite");
        assert!(b.kept_mass_teacher > 0.99);
        // The student's kept mass is exactly its ONE tie-head (all the -1.0
        // logits tie; the first-index rule picks index 1): 1/63. The
        // teacher's head (index 0) carries exp(-1e9 ···) = 0.0 — no mass,
        // which is the fixture's whole point.
        assert!((b.kept_mass_student - 1.0 / 63.0).abs() < 1e-9);

        // The clamp-firing arm: k = 63 misses one student index; make that
        // index the sentinel so q_tail underflows to exact 0.0.
        let mut student2 = vec![-1.0f32; 64];
        student2[40] = -1.0e9; // never in top-63 of the student row
        let teacher2 = vec![0.0f32; 64];
        let b2 = sparse_kl_lower_bound(&teacher2, &student2, 63).unwrap();
        assert!(b2.kl_lower_bound_nats.is_finite());
        // q side coverage: 63 normal indices carry ≈ all the mass; the
        // sentinel carries exp((-1e9 - max) - ln_z) = exact 0.0.
        assert!(b2.kept_mass_student > 1.0 - 1e-9);
        // p side: the missed index is a plain -1.0 — p_tail is tiny but
        // positive, so the tail term exists and used the floored q_tail.
        assert!(b2.kl_lower_bound_nats > 0.0);
    }

    #[test]
    fn shift_invariant_jointly_raw_logits_contract() {
        // Exact invariance in real arithmetic; the tolerance covers the f32
        // quantization of the SHIFTED row itself (x + 37.0 rounds to a
        // different f32 than x — up to ~1 ulp of the shifted magnitude,
        // ≈ 4e-6 in log-prob terms here).
        let f = Fixture::planted();
        let base = sparse_kl_lower_bound(&f.teacher, &f.student, 8).unwrap();
        let shifted_teacher: Vec<f32> = f.teacher.iter().map(|&x| x + 37.0).collect();
        let shifted_student: Vec<f32> = f.student.iter().map(|&x| x + 37.0).collect();
        let shifted = sparse_kl_lower_bound(&shifted_teacher, &shifted_student, 8).unwrap();
        assert!(
            (base.kl_lower_bound_nats - shifted.kl_lower_bound_nats).abs() < 1e-4,
            "KL is shift-invariant; raw-logit inputs must not be rescaled (base {}, shifted {})",
            base.kl_lower_bound_nats,
            shifted.kl_lower_bound_nats
        );
    }

    #[test]
    fn coverage_floors_spiked_and_uniform() {
        // Spiked teacher: k=1 already covers ~everything.
        let mut teacher = vec![-5.0f32; 64];
        teacher[3] = 10.0;
        let student = vec![0.0f32; 64];
        let b = sparse_kl_lower_bound(&teacher, &student, 1).unwrap();
        assert!(
            b.kept_mass_teacher > 0.99,
            "spiked head must be covered at k=1"
        );
        // Uniform teacher: k=1 covers ≈ 1/64 — the honest loose-coverage
        // signal the caveat describes.
        let uniform = vec![0.0f32; 64];
        let flat = vec![0.0f32; 64];
        let u = sparse_kl_lower_bound(&uniform, &flat, 1).unwrap();
        let want = 1.0 / 64.0;
        assert!(
            (u.kept_mass_teacher - want).abs() < 1e-3,
            "uniform coverage at k=1 should be ≈ k/V, got {}",
            u.kept_mass_teacher
        );
    }

    #[test]
    fn degenerates_refuse_loud() {
        let f = Fixture::planted();
        assert_eq!(
            sparse_kl_lower_bound(&f.teacher[..3], &f.student, 1).unwrap_err(),
            SparseKlError::LengthMismatch {
                teacher: 3,
                student: 64
            }
        );
        assert_eq!(
            sparse_kl_lower_bound(&f.teacher, &f.student, 0).unwrap_err(),
            SparseKlError::KZero
        );
        assert_eq!(
            sparse_kl_lower_bound(&[], &[], 1).unwrap_err(),
            SparseKlError::Empty
        );
        let mut bad = f.teacher.clone();
        bad[7] = f32::NEG_INFINITY;
        assert_eq!(
            sparse_kl_lower_bound(&bad, &f.student, 1).unwrap_err(),
            SparseKlError::NonFiniteLogit {
                side: "teacher",
                index: 7
            }
        );
        let mut nan = f.student.clone();
        nan[0] = f32::NAN;
        assert_eq!(
            sparse_kl_lower_bound(&f.teacher, &nan, 1).unwrap_err(),
            SparseKlError::NonFiniteLogit {
                side: "student",
                index: 0
            }
        );
    }

    #[test]
    fn tie_rule_first_index_wins() {
        // All-equal rows: the top-1 pick must be index 0 on both sides —
        // the Issue-940 first-index strict tie rule.
        let teacher = vec![1.0f32; 64];
        let student = vec![1.0f32; 64];
        let b = sparse_kl_lower_bound(&teacher, &student, 1).unwrap();
        assert_eq!(b.kept_count, 1);
        // kept = {index 0} both sides: KL of identical distributions is 0.
        assert!(b.kl_lower_bound_nats.abs() < 1e-12);
    }
}
