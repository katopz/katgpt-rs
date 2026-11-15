//! `katgpt-core::compounding` — the coverage-factor compound-failure bound
//! substrate (Issue 1222; arXiv:2609.19465, Lemma 5.1 / Prop B.2 / §5.2 —
//! distilled modelless in katgpt-rs Research 674).
//!
//! Every input is a COUNT histogram the consumers already persist; zero
//! gradients involved. Three closed-form primitives:
//!
//! ```text
//! C_T    = max_c (Σ_t μ_t(c)) / q(c)     Lemma 5.1's coverage factor
//! bound  = clamp(ε_atom · C_T, 0, 1)     atomic → composed (Lemma 5.1)
//! ε_atom ≤ ε_comp / c1                   composed → atomic (§5.2 reverse, T = 1)
//! ```
//!
//! # The `Vacuous` refusal IS the theorem
//!
//! A rollout context with no q-mass is the paper's Level-III structural gap AS
//! A TYPE: [`CoverageFactor::Vacuous`] names the uncovered keys and
//! [`compound_failure_bound`] refuses — it never folds the gap into 0.0, and a
//! perfect atomic accuracy (ε_atom = 0) buys no Level-III guarantee. Thin
//! support (`q(c) < min_support` on a touched cell) refuses
//! [`CoverageFactor::Undecidable`] — decline-first, the esc_audit precedent.
//!
//! # Determinism (Issue 940 law)
//!
//! All arithmetic is f64 over u64 counts; every multi-key reduction walks keys
//! in ascending order and folds sequentially; no HashMap iteration anywhere.
//! [`direct_bound`]'s reduction order is (steps in input order, keys ascending
//! within a step) — pinned by the API, so repeat calls on the same input are
//! bit-identical.
//!
//! # `direct_bound`'s normalization regime — read before comparing to `C_T · ε`
//!
//! `direct_bound` reads each step's counts as a per-step probability
//! (count / step total) and returns the rollout's expected per-step error.
//! The comparison against `C_T · ε_atom` is a THEOREM only under matched
//! per-step mass: with M = per-step count total and Q = Σ_c q(c),
//!
//! ```text
//! Σ_c (Σ_t μ_t(c)) = T·M ≤ C_T · Σ_c q(c) = C_T · Q   ⟹   C_T ≥ T·M/Q
//! ```
//!
//! so M = Q gives C_T ≥ T, and direct ≤ T·ε_atom ≤ C_T·ε_atom. Under-sampled
//! rollouts (M < Q) can read C_T < T and the inequality may not hold — that is
//! a property of the regime, not of the code. The G1 property test generates
//! matched-mass rollouts only.
//!
//! # Estimator-design notes (read before wiring a consumer — issue T6)
//!
//! - **Context binning is a modeling choice** (the `contrastive_scope`
//!   disclosure precedent): the keys here are opaque u64 signatures; the
//!   caller's binning decides what "coverage" even means.
//! - **μ_t is survivorship-conditioned** — mistake-free prefixes only, unless
//!   the caller records context-at-step on failing rollouts too. Disclose the
//!   conditioning beside any bound the consumer reports.
//! - **The consumer-side audit is a pre-registered TIGHTNESS readout, never a
//!   dominance check**: report (a) bound/measured-failure ratio per campaign
//!   class (a ratio ≫ 1 means the bound is loose — a finding, never a pass)
//!   and (b) the non-vacuous fraction of campaigns (a low fraction means the
//!   binning is broken — the C_T instrument is unusable on that surface, said
//!   loud). "Bound ≥ measured failure" alone is NOT an audit — any loose bound
//!   passes it.
//! - **`Finite(0.0)` means "no rollout mass observed"** — the bound reads 0;
//!   treat that as *no data*, never as *safe* ([`error_budget`] refuses it).
//!
//! # Feature
//!
//! Gated under the `estimators` family umbrella — the Plan 635 ratchet holds
//! katgpt-core at ceiling + grants exactly (zero key headroom), so the module
//! lands under the parent cfg instead of a new `[features]` key; a consumer
//! needing selective enablement escalates via a reasoned grant row in
//! `scripts/feature_count_pins.txt`.

use std::collections::BTreeMap;

/// A rollout-context signature. Opaque u64 — the consumer folds its context
/// record (the advisory wiring names blake3 over `{code}:{strategy}:{shape}`
/// class signatures); the substrate never inspects it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextKey(u64);

impl From<u64> for ContextKey {
    fn from(k: u64) -> Self {
        Self(k)
    }
}

impl From<ContextKey> for u64 {
    fn from(k: ContextKey) -> u64 {
        k.0
    }
}

/// COUNT histogram context → count (integers, never probabilities), canonical
/// form: entries sorted by key ascending, duplicate keys pre-summed. Explicit
/// zero counts are legal (an uncovered witness); absent keys read 0.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CtxHist(Vec<(u64, u64)>);

impl CtxHist {
    /// Panics only if summing duplicate keys overflows u64 — a caller bug,
    /// loud by design.
    pub fn from_pairs(pairs: &[(u64, u64)]) -> Self {
        Self(canonicalize_count_pairs(pairs))
    }

    /// The count at `key`; 0 for absent keys.
    pub fn get(&self, key: u64) -> u64 {
        self.entries()
            .binary_search_by(|(k, _)| k.cmp(&key))
            .ok()
            .map(|i| self.0[i].1)
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The canonical entries — key-ascending, duplicates pre-summed.
    pub fn entries(&self) -> &[(u64, u64)] {
        &self.0
    }
}

impl From<&BTreeMap<u64, u64>> for CtxHist {
    fn from(m: &BTreeMap<u64, u64>) -> Self {
        Self(m.iter().map(|(&k, &c)| (k, c)).collect())
    }
}

/// One rollout step's context visitation COUNTS — same canonical form as
/// [`CtxHist`]. The per-step probability [`direct_bound`] reads out of a step
/// is count / step-total (see the module doc's normalization regime).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepHist(Vec<(u64, u64)>);

impl StepHist {
    /// Panics only if summing duplicate keys overflows u64 — a caller bug,
    /// loud by design.
    pub fn from_pairs(pairs: &[(u64, u64)]) -> Self {
        Self(canonicalize_count_pairs(pairs))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The canonical entries — key-ascending, duplicates pre-summed.
    pub fn entries(&self) -> &[(u64, u64)] {
        &self.0
    }
}

impl From<&BTreeMap<u64, u64>> for StepHist {
    fn from(m: &BTreeMap<u64, u64>) -> Self {
        Self(m.iter().map(|(&k, &c)| (k, c)).collect())
    }
}

/// Per-context error counters: (errors, trials). A rollout context with NO
/// row here refuses [`BoundRefusal::Undecidable`] in [`direct_bound`] — never
/// treated as error-free.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CtxErr(Vec<(u64, (u64, u64))>);

impl CtxErr {
    /// Panics only if summing duplicate keys overflows u64 — a caller bug,
    /// loud by design.
    pub fn from_pairs(pairs: &[(u64, (u64, u64))]) -> Self {
        let mut v = pairs.to_vec();
        v.sort_unstable_by_key(|&(k, _)| k);
        let mut out: Vec<(u64, (u64, u64))> = Vec::with_capacity(v.len());
        for &(k, (e, t)) in &v {
            match out.last_mut() {
                Some(last) if last.0 == k => {
                    last.1.0 = last
                        .1
                        .0
                        .checked_add(e)
                        .expect("error count overflow summing duplicate keys");
                    last.1.1 = last
                        .1
                        .1
                        .checked_add(t)
                        .expect("trial count overflow summing duplicate keys");
                }
                _ => out.push((k, (e, t))),
            }
        }
        Self(out)
    }

    /// The (errors, trials) counter at `key`, if a row exists.
    pub fn get_pair(&self, key: u64) -> Option<(u64, u64)> {
        self.0
            .binary_search_by(|(k, _)| k.cmp(&key))
            .ok()
            .map(|i| self.0[i].1)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The canonical entries — key-ascending, duplicates pre-summed.
    pub fn entries(&self) -> &[(u64, (u64, u64))] {
        &self.0
    }
}

impl From<&BTreeMap<u64, (u64, u64)>> for CtxErr {
    fn from(m: &BTreeMap<u64, (u64, u64)>) -> Self {
        Self(m.iter().map(|(&k, &v)| (k, v)).collect())
    }
}

/// Lemma 5.1's C_T as a decision, never a guess: the finite factor, the
/// structural-gap refusal (naming the uncovered keys), or the thin-support
/// abstention.
#[derive(Clone, Debug, PartialEq)]
pub enum CoverageFactor {
    /// The sup ratio. `Finite(0.0)` means *no rollout mass observed* — no
    /// data, never safety.
    Finite(f64),
    /// Level-III structural gap: rollout contexts with zero q-mass, ascending.
    Vacuous { uncovered: Vec<ContextKey> },
    /// Thin support below `min_support` on a touched cell — decline-first.
    Undecidable,
}

impl CoverageFactor {
    /// The factor when decidable.
    pub fn factor(&self) -> Option<f64> {
        match self {
            Self::Finite(x) => Some(*x),
            _ => None,
        }
    }

    pub fn is_vacuous(&self) -> bool {
        matches!(self, Self::Vacuous { .. })
    }
}

/// Why a bound or budget refuses. `Vacuous` carries the uncovered keys —
/// the structural gap travels with the refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundRefusal {
    Vacuous(Vec<ContextKey>),
    Undecidable,
}

/// Reusable scratch for the zero-alloc steady fold. Size it once with
/// [`scratch_capacity_hint`] per histogram family and reuse across calls —
/// the fold clears and refills in place.
#[derive(Clone, Debug, Default)]
pub struct FoldScratch {
    sums: Vec<(u64, u64)>,
    merge: Vec<(u64, u64)>,
}

impl FoldScratch {
    pub fn with_capacity(hint: usize) -> Self {
        Self {
            sums: Vec::with_capacity(hint),
            merge: Vec::with_capacity(hint),
        }
    }

    pub fn clear(&mut self) {
        self.sums.clear();
        self.merge.clear();
    }
}

/// Capacity that guarantees the steady fold performs zero allocations:
/// distinct touched keys ≤ Σ|steps|, and the merge pass needs
/// |sums| + max|step| ≤ Σ|steps| + max|step|. An under-provisioned scratch
/// still produces correct results — it just allocates (the reserve fallback).
pub fn scratch_capacity_hint(mu: &[StepHist]) -> usize {
    let total: usize = mu.iter().map(StepHist::len).sum();
    let max = mu.iter().map(StepHist::len).max().unwrap_or(0);
    total + max
}

/// Overflow summing one context's counts across steps — mapped to
/// `Undecidable` by every deciding caller (the ratio is genuinely
/// undecidable past u64).
struct MergeOverflow;

/// Merge one sorted step into the sorted running sums (two-pointer, in place
/// through the swap buffer). Zero-alloc at the hinted capacity.
fn merge_step(
    sums: &mut Vec<(u64, u64)>,
    step: &[(u64, u64)],
    merge: &mut Vec<(u64, u64)>,
) -> Result<(), MergeOverflow> {
    // No-op at the hinted capacity; the fallback allocates on under-provisioning.
    // (After the swap below this buffer holds the previous sums — cleared here.)
    merge.reserve(sums.len() + step.len());
    merge.clear();
    let (mut i, mut j) = (0usize, 0usize);
    while i < sums.len() && j < step.len() {
        let (sk, sv) = sums[i];
        let (tk, tv) = step[j];
        if sk < tk {
            merge.push((sk, sv));
            i += 1;
        } else if sk > tk {
            merge.push((tk, tv));
            j += 1;
        } else {
            let total = sv.checked_add(tv).ok_or(MergeOverflow)?;
            merge.push((sk, total));
            i += 1;
            j += 1;
        }
    }
    if i < sums.len() {
        merge.extend_from_slice(&sums[i..]);
    }
    if j < step.len() {
        merge.extend_from_slice(&step[j..]);
    }
    std::mem::swap(sums, merge);
    Ok(())
}

/// Accumulate Σ_t μ_t(c) per key across all steps — the streaming fold
/// (O(|C|) memory, never a T×|C| materialization).
fn fold_sums(mu: &[StepHist], scratch: &mut FoldScratch) -> Result<(), MergeOverflow> {
    scratch.clear();
    for step in mu {
        merge_step(&mut scratch.sums, step.entries(), &mut scratch.merge)?;
    }
    Ok(())
}

/// Lemma 5.1's coverage factor C_T = max over touched c of
/// (Σ_t μ_t(c)) / q(c). Convenience form — allocates its own scratch; see
/// [`coverage_factor_with`] for the zero-alloc steady path.
pub fn coverage_factor(mu: &[StepHist], q: &CtxHist, min_support: u32) -> CoverageFactor {
    let mut scratch = FoldScratch::with_capacity(scratch_capacity_hint(mu));
    coverage_factor_with(mu, q, min_support, &mut scratch)
}

/// [`coverage_factor`] with caller-owned scratch: zero allocations in the
/// steady path at the [`scratch_capacity_hint`] capacity. The `Vacuous`
/// answer may allocate — reporting the gap names it.
///
/// Precedence: the structural gap (`Vacuous`) outranks thin support
/// (`Undecidable`) — checked in that order, globally.
pub fn coverage_factor_with(
    mu: &[StepHist],
    q: &CtxHist,
    min_support: u32,
    scratch: &mut FoldScratch,
) -> CoverageFactor {
    if fold_sums(mu, scratch).is_err() {
        return CoverageFactor::Undecidable;
    }
    // Pass 1 — the structural gap, key-ascending as the walk.
    let mut uncovered: Vec<ContextKey> = Vec::new();
    for &(key, sum) in &scratch.sums {
        if sum > 0 && q.get(key) == 0 {
            uncovered.push(ContextKey(key));
        }
    }
    if !uncovered.is_empty() {
        return CoverageFactor::Vacuous { uncovered };
    }
    // Pass 2 — thin support, then the ratio sup.
    let mut best = 0.0f64;
    for &(key, sum) in &scratch.sums {
        if sum == 0 {
            continue; // zero-mass keys decide nothing
        }
        let qc = q.get(key);
        if qc < u64::from(min_support) {
            return CoverageFactor::Undecidable;
        }
        let ratio = sum as f64 / qc as f64;
        if ratio > best {
            best = ratio;
        }
    }
    CoverageFactor::Finite(best)
}

/// Lemma 5.1's composed-failure bound, CLAMPED to [0, 1] (a loose C_T can
/// push the raw product past 1 — the clamp is disclosed here, never silent).
/// Refuses the structural gap (naming the keys) and the thin-support
/// abstention; a non-finite or out-of-[0,1] `eps_atom` refuses
/// [`BoundRefusal::Undecidable`] (decline-first on invalid input).
pub fn compound_failure_bound(eps_atom: f64, c: &CoverageFactor) -> Result<f64, BoundRefusal> {
    let ct = match c {
        CoverageFactor::Finite(x) => *x,
        CoverageFactor::Vacuous { uncovered } => {
            return Err(BoundRefusal::Vacuous(uncovered.clone()));
        }
        CoverageFactor::Undecidable => return Err(BoundRefusal::Undecidable),
    };
    if !eps_atom.is_finite() || !(0.0..=1.0).contains(&eps_atom) {
        return Err(BoundRefusal::Undecidable);
    }
    Ok((eps_atom * ct).clamp(0.0, 1.0))
}

/// The measured composed failure under per-context counters:
/// Σ_t Σ_c (count_t(c) / step-total_t) · e(c), e(c) = errors/trials — the
/// rollout's expected per-step error (see the module doc's normalization
/// regime). A context in any step with NO counter row refuses — never treated
/// as error-free; so does a 0-trials or errors > trials row. Always
/// ≤ C_T·ε_atom under matched per-step mass (C_T ≥ T there — the module doc
/// carries the derivation).
pub fn direct_bound(mu: &[StepHist], err: &CtxErr) -> Result<f64, BoundRefusal> {
    let mut acc = 0.0f64;
    for step in mu {
        let mut total = 0u64;
        for &(_, c) in step.entries() {
            total = total.checked_add(c).ok_or(BoundRefusal::Undecidable)?;
        }
        if total == 0 {
            continue; // an empty rollout step contributes nothing
        }
        let inv = 1.0f64 / total as f64;
        for &(key, c) in step.entries() {
            let (errors, trials) = err.get_pair(key).ok_or(BoundRefusal::Undecidable)?;
            if trials == 0 || errors > trials {
                return Err(BoundRefusal::Undecidable);
            }
            let rate = errors as f64 / trials as f64;
            acc += c as f64 * inv * rate;
        }
    }
    Ok(acc)
}

/// §5.2's reverse bound at T = 1: the exact inverse of Lemma 5.1's forward
/// direction (ε_comp = c1·ε_atom there), so composed-slice calibration
/// transfers down for free. c1 ≥ 1 tightens below `eps_comp`; `c1 = 0.0`
/// yields +inf — honest no-information (an uncovered first step attributes
/// nothing); invalid inputs surface as NaN, decline-first at the caller.
pub fn first_step_bound(eps_comp: f64, c1: f64) -> f64 {
    debug_assert!(
        c1.is_finite() && c1 >= 0.0,
        "c1 comes from CoverageFactor::Finite — never negative or NaN"
    );
    eps_comp / c1
}

/// The horizon floor C_T ≥ T: under matched per-step sampling (each step's
/// context counts proportional to q) C_T = T exactly; over-sampled rollouts
/// exceed it, under-sampled ones can read below it (the general inequality is
/// C_T ≥ T·M/Q — see the module doc).
pub fn horizon_floor(t_steps: usize) -> f64 {
    t_steps as f64
}

/// §5.2's error budget: the largest ε_atom keeping the composed failure ≤
/// 1 − success_target, i.e. (1 − S) / C_T. Refuses the structural gap and the
/// thin-support abstention like [`compound_failure_bound`]; refuses
/// [`BoundRefusal::Undecidable`] for an out-of-[0,1] or non-finite
/// `success_target` and for `Finite(0.0)` (no rollout mass — no budget is
/// decidable). No horizon argument: C_T already encodes it.
pub fn error_budget(success_target: f64, c: &CoverageFactor) -> Result<f64, BoundRefusal> {
    let ct = match c {
        CoverageFactor::Finite(x) => *x,
        CoverageFactor::Vacuous { uncovered } => {
            return Err(BoundRefusal::Vacuous(uncovered.clone()));
        }
        CoverageFactor::Undecidable => return Err(BoundRefusal::Undecidable),
    };
    if !success_target.is_finite() || !(0.0..=1.0).contains(&success_target) {
        return Err(BoundRefusal::Undecidable);
    }
    if ct <= 0.0 {
        return Err(BoundRefusal::Undecidable);
    }
    Ok((1.0 - success_target) / ct)
}

/// The structural gap as a list: rollout contexts with μ-mass > 0 and zero
/// q-mass, key-ascending.
pub fn support_gap(mu: &[StepHist], q: &CtxHist) -> Vec<ContextKey> {
    let mut scratch = FoldScratch::with_capacity(scratch_capacity_hint(mu));
    // Overflow is ignored here on purpose: gap detection needs presence, not
    // the exact count (documented divergence from the deciding callers).
    let _ = fold_sums(mu, &mut scratch);
    scratch
        .sums
        .iter()
        .filter(|&&(key, sum)| sum > 0 && q.get(key) == 0)
        .map(|&(key, _)| ContextKey(key))
        .collect()
}

/// Sort by key, sum duplicates (loud on u64 overflow).
fn canonicalize_count_pairs(pairs: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut v = pairs.to_vec();
    v.sort_unstable_by_key(|&(k, _)| k);
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(v.len());
    for &(k, c) in &v {
        match out.last_mut() {
            Some(last) if last.0 == k => {
                last.1 = last
                    .1
                    .checked_add(c)
                    .expect("context count overflow summing duplicate keys");
            }
            _ => out.push((k, c)),
        }
    }
    out.shrink_to_fit();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The issue's G1 fixture (COUNT-scaled, asserted to 1e-12): q Σ=10,
    /// per-key μ sums 16 / 9 / 5 → ratios 3.2 / 3.0 / 2.5 → C_T = 3.2;
    /// ε_atom = 0.05 → bound = 0.16; direct = 0.15 ≤ 0.16.
    fn fixture() -> (CtxHist, Vec<StepHist>, CtxErr) {
        let q = CtxHist::from_pairs(&[(1, 5), (2, 3), (3, 2)]);
        let mu = vec![
            StepHist::from_pairs(&[(1, 10)]),
            StepHist::from_pairs(&[(1, 4), (2, 6)]),
            StepHist::from_pairs(&[(1, 2), (2, 3), (3, 5)]),
        ];
        let err = CtxErr::from_pairs(&[(1, (5, 100)), (2, (5, 100)), (3, (5, 100))]);
        (q, mu, err)
    }

    #[test]
    fn g1_fixture_ct_bound_and_direct() {
        let (q, mu, err) = fixture();
        let cf = coverage_factor(&mu, &q, 0);
        let ct = cf.factor().expect("fixture decides");
        assert!((ct - 3.2).abs() < 1e-12, "C_T {ct} != 3.2");
        let bound = compound_failure_bound(0.05, &cf).expect("finite factor");
        assert!((bound - 0.16).abs() < 1e-12, "bound {bound} != 0.16");
        let direct = direct_bound(&mu, &err).expect("all counters present");
        assert!((direct - 0.15).abs() < 1e-12, "direct {direct} != 0.15");
        assert!(
            direct <= bound,
            "direct {direct} must sit under C_T·ε {bound}"
        );
    }

    #[test]
    fn t3_structural_gap_refuses_vacuous_even_at_zero_epsilon() {
        let q = CtxHist::from_pairs(&[(1, 5), (2, 3), (3, 2)]);
        let mu = vec![
            StepHist::from_pairs(&[(1, 10)]),
            StepHist::from_pairs(&[(1, 4), (2, 6), (4, 2)]), // c4 injected, q[c4] = 0
            StepHist::from_pairs(&[(1, 2), (2, 3), (3, 5)]),
        ];
        let cf = coverage_factor(&mu, &q, 0);
        match &cf {
            CoverageFactor::Vacuous { uncovered } => {
                assert_eq!(uncovered, &[ContextKey(4)]);
            }
            other => panic!("expected Vacuous, got {other:?}"),
        }
        assert_eq!(
            compound_failure_bound(0.05, &cf),
            Err(BoundRefusal::Vacuous(vec![ContextKey(4)]))
        );
        // Perfect atomic accuracy buys no Level-III guarantee:
        assert!(matches!(
            compound_failure_bound(0.0, &cf),
            Err(BoundRefusal::Vacuous(_))
        ));
        // support_gap reports the same gap independently:
        assert_eq!(support_gap(&mu, &q), vec![ContextKey(4)]);
    }

    #[test]
    fn clamp_discloses_the_loose_bound() {
        let (q, mu, _) = fixture();
        let cf = coverage_factor(&mu, &q, 0);
        // 0.4 · 3.2 = 1.28 → clamped to 1.0.
        let bound = compound_failure_bound(0.4, &cf).expect("finite factor");
        assert_eq!(bound, 1.0);
    }

    #[test]
    fn bound_is_monotone_in_epsilon_atom() {
        let (q, mu, _) = fixture();
        let cf = coverage_factor(&mu, &q, 0);
        let mut prev = -1.0f64;
        for eps in [0.01, 0.05, 0.1, 0.3] {
            let b = compound_failure_bound(eps, &cf).expect("finite factor");
            assert!(b > prev, "bound not strictly increasing at eps {eps}");
            prev = b;
        }
    }

    #[test]
    fn bound_is_monotone_in_horizon_t() {
        let (q, mu, _) = fixture();
        let first = &mu[0];
        let mut prev = -1.0f64;
        for t in 1..=3usize {
            let steps: Vec<StepHist> = std::iter::repeat_n(first.clone(), t).collect();
            let cf = coverage_factor(&steps, &q, 0);
            let b = compound_failure_bound(0.05, &cf).expect("finite factor");
            assert!(b > prev, "bound not strictly increasing at T {t}");
            prev = b;
        }
    }

    #[test]
    fn horizon_floor_holds_under_matched_sampling() {
        // q uniform, μ_t = q per step (matched) → C_T = T exactly.
        let q = CtxHist::from_pairs(&[(1, 10), (2, 10), (3, 10)]);
        let step = StepHist::from_pairs(&[(1, 10), (2, 10), (3, 10)]);
        let mu = vec![step.clone(), step.clone(), step];
        let cf = coverage_factor(&mu, &q, 0);
        let ct = cf.factor().expect("fixture decides");
        assert!((ct - 3.0).abs() < 1e-12, "matched C_T {ct} != 3.0");
        assert!(ct + 1e-12 >= horizon_floor(3), "C_T {ct} < horizon floor");
    }

    #[test]
    fn thin_support_abstains_and_untouched_cells_do_not_fire() {
        let (q, mu, _) = fixture();
        // q = {c1: 5, c2: 3, c3: 2}: min_support 3 touches c3 (count 2).
        assert_eq!(coverage_factor(&mu, &q, 3), CoverageFactor::Undecidable);
        // min_support 2 admits every touched cell.
        assert!(coverage_factor(&mu, &q, 2).factor().is_some());
        // A thin cell μ never touches decides nothing: q4 count 1, no μ;
        // min_support 2 admits every touched cell (5, 3, 2) and only the
        // untouched cell (count 1) is thin.
        let q4 = CtxHist::from_pairs(&[(1, 5), (2, 3), (3, 2), (9, 1)]);
        assert!(
            coverage_factor(&mu, &q4, 2).factor().is_some(),
            "untouched thin cell must not fire Undecidable"
        );
    }

    #[test]
    fn direct_bound_refuses_missing_and_invalid_counters() {
        let (q, mu, _) = fixture();
        // Missing counter row for c3:
        let partial = CtxErr::from_pairs(&[(1, (5, 100)), (2, (5, 100))]);
        assert_eq!(
            direct_bound(&mu, &partial),
            Err(BoundRefusal::Undecidable),
            "a context with no counter must refuse, never read as error-free"
        );
        // 0 trials:
        let zero_trials = CtxErr::from_pairs(&[(1, (5, 100)), (2, (5, 100)), (3, (0, 0))]);
        assert_eq!(
            direct_bound(&mu, &zero_trials),
            Err(BoundRefusal::Undecidable)
        );
        // errors > trials:
        let invalid = CtxErr::from_pairs(&[(1, (5, 100)), (2, (5, 100)), (3, (9, 5))]);
        assert_eq!(direct_bound(&mu, &invalid), Err(BoundRefusal::Undecidable));
        let _ = q;
    }

    #[test]
    fn first_step_bound_is_the_t1_inverse() {
        let b = first_step_bound(0.16, 3.2);
        assert!((b - 0.05).abs() < 1e-12, "reverse bound {b} != 0.05");
        // c1 = 1 is the identity:
        assert!((first_step_bound(0.07, 1.0) - 0.07).abs() < 1e-15);
        // c1 = 0 is honest no-information:
        let inf = first_step_bound(0.16, 0.0);
        assert!(inf.is_infinite() && inf > 0.0);
    }

    #[test]
    fn error_budget_refuses_and_divides() {
        let (q, mu, _) = fixture();
        let cf = coverage_factor(&mu, &q, 0);
        let budget = error_budget(0.8, &cf).expect("finite factor");
        assert!(
            (budget - (1.0 - 0.8) / 3.2).abs() < 1e-12,
            "budget {budget}"
        );
        // Out-of-range success target refuses:
        assert_eq!(error_budget(1.2, &cf), Err(BoundRefusal::Undecidable));
        assert_eq!(error_budget(f64::NAN, &cf), Err(BoundRefusal::Undecidable));
        // Finite(0.0) — no rollout mass — refuses too:
        let empty = coverage_factor(&[], &q, 0);
        assert_eq!(empty, CoverageFactor::Finite(0.0));
        assert_eq!(error_budget(0.8, &empty), Err(BoundRefusal::Undecidable));
        // Vacuous propagates with its keys:
        let q2 = CtxHist::from_pairs(&[(1, 5)]);
        let gapped = vec![StepHist::from_pairs(&[(1, 4), (7, 3)])];
        let vac = coverage_factor(&gapped, &q2, 0);
        assert_eq!(
            error_budget(0.8, &vac),
            Err(BoundRefusal::Vacuous(vec![ContextKey(7)]))
        );
    }

    #[test]
    fn compound_bound_validates_epsilon_atom() {
        let (q, mu, _) = fixture();
        let cf = coverage_factor(&mu, &q, 0);
        assert_eq!(
            compound_failure_bound(-0.1, &cf),
            Err(BoundRefusal::Undecidable)
        );
        assert_eq!(
            compound_failure_bound(1.5, &cf),
            Err(BoundRefusal::Undecidable)
        );
        assert_eq!(
            compound_failure_bound(f64::NAN, &cf),
            Err(BoundRefusal::Undecidable)
        );
    }

    #[test]
    fn support_gap_is_empty_when_every_context_is_supported() {
        let (q, mu, _) = fixture();
        assert!(support_gap(&mu, &q).is_empty());
    }

    #[test]
    fn fold_is_bit_deterministic_across_repeats() {
        let (q, mu, err) = fixture();
        let a = coverage_factor(&mu, &q, 1);
        let b = coverage_factor(&mu, &q, 1);
        match (&a, &b) {
            (CoverageFactor::Finite(x), CoverageFactor::Finite(y)) => {
                assert_eq!(x.to_bits(), y.to_bits(), "C_T must be bit-identical");
            }
            _ => panic!("fixture must decide both times"),
        }
        let d1 = direct_bound(&mu, &err).expect("counters");
        let d2 = direct_bound(&mu, &err).expect("counters");
        assert_eq!(d1.to_bits(), d2.to_bits(), "direct must be bit-identical");
    }

    /// Deterministic xorshift64 — the property test needs reproducible
    /// pseudo-randomness, not cryptography; zero new deps.
    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1) // never zero
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// The paper's inequality pinned on randomized histograms: per-step mass
    /// matched to Σq (the regime where C_T ≥ T is a theorem — module doc),
    /// uniform e(c) = 0.05 → direct ≤ C_T·ε_atom, plus the horizon floor,
    /// over 100 seeds.
    #[test]
    fn property_direct_leq_compound_bound_100_seeds() {
        for seed in 0u64..100 {
            let mut rng = Xorshift64::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
            let n_ctx = 4 + (rng.next() % 8) as usize; // 4..=11 contexts
            let ctx: Vec<u64> = (0..n_ctx as u64).map(|i| 100 + i * 7).collect();
            let q_pairs: Vec<(u64, u64)> = ctx.iter().map(|&c| (c, 1 + rng.next() % 40)).collect();
            let q = CtxHist::from_pairs(&q_pairs);
            let total_q: u64 = q_pairs.iter().map(|&(_, c)| c).sum();
            let n_steps = 2 + (rng.next() % 5) as usize; // 2..=6 steps
            let mut steps = Vec::with_capacity(n_steps);
            for _ in 0..n_steps {
                // Split the step mass (== total_q) across the contexts at random.
                let mut remaining = total_q;
                let mut pairs = Vec::with_capacity(n_ctx);
                for (i, &c) in ctx.iter().enumerate() {
                    let take = if i + 1 == ctx.len() {
                        remaining
                    } else {
                        rng.next() % (remaining + 1)
                    };
                    pairs.push((c, take));
                    remaining -= take;
                }
                steps.push(StepHist::from_pairs(&pairs));
            }
            let cf = coverage_factor(&steps, &q, 0);
            let ct = match &cf {
                CoverageFactor::Finite(x) => *x,
                other => panic!("seed {seed}: expected Finite, got {other:?}"),
            };
            // The matched-mass theorem: C_T ≥ T·M/Q = T.
            assert!(
                ct + 1e-9 >= horizon_floor(n_steps),
                "seed {seed}: C_T {ct} < T {n_steps}"
            );
            let err =
                CtxErr::from_pairs(&ctx.iter().map(|&c| (c, (5u64, 100u64))).collect::<Vec<_>>());
            let direct = direct_bound(&steps, &err).expect("all counters present");
            let bound = compound_failure_bound(0.05, &cf).expect("finite factor");
            // 1e-9: f64 hygiene slack on a theorem with a wide margin
            // (C_T ≥ T strict except at exact proportionality).
            assert!(
                direct <= bound + 1e-9,
                "seed {seed}: direct {direct} > C_T·ε {bound}"
            );
        }
    }

    // ── G4: allocation-free steady fold ──

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn g4_zero_alloc_steady_fold() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};
        let (q, mu, _) = fixture();
        let mut scratch = FoldScratch::with_capacity(scratch_capacity_hint(&mu));
        reset_alloc_stats();
        let first = coverage_factor_with(&mu, &q, 0, &mut scratch);
        // Reuse the SAME scratch — the second fold must stay allocation-free.
        let second = coverage_factor_with(&mu, &q, 1, &mut scratch);
        let (count, _bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "steady fold must be allocation-free at the hinted capacity, got {count} allocs"
        );
        assert_eq!(first, second);
    }

    // ── G2: the 10^4 × 10^3 perf bar (release-only measurement posture) ──

    /// **#[ignore]** — release-only: `cargo test -p katgpt-core --features
    /// estimators --release --lib -- --ignored g2_perf`. Bar: 10^7 adds
    /// (10^4 contexts × 10^3 steps, full coverage per step) fold ≤ 25 ms
    /// single-thread release. Quote box state beside any recorded number.
    #[test]
    #[ignore]
    #[cfg(not(debug_assertions))] // release-only posture; never a dev-profile assert
    fn g2_perf_fold_10k_contexts_x_1k_steps() {
        const KEYS: u64 = 10_000;
        const STEPS: usize = 1_000;
        let q = CtxHist::from_pairs(&(0..KEYS).map(|k| (k, 3)).collect::<Vec<_>>());
        let steps: Vec<StepHist> = (0..STEPS)
            .map(|t| {
                StepHist::from_pairs(
                    &(0..KEYS)
                        .map(|k| (k, 1 + ((k + t as u64) % 3)))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut scratch = FoldScratch::with_capacity(scratch_capacity_hint(&steps));
        let t0 = std::time::Instant::now();
        let r = coverage_factor_with(&steps, &q, 0, &mut scratch);
        let dt = t0.elapsed();
        let ms = dt.as_secs_f64() * 1e3;
        assert!(
            matches!(r, CoverageFactor::Finite(_)),
            "full-support fold must decide, got {r:?}"
        );
        println!(
            "g2 fold: {KEYS} contexts x {STEPS} steps = {} adds in {ms:.2} ms (bar 25 ms)",
            STEPS * KEYS as usize
        );
        assert!(
            ms <= 25.0,
            "10^7-add fold took {ms:.2} ms > the 25 ms G2 bar"
        );

        // The T4 paired arm: the naive HashMap-accumulate-then-sort fold —
        // LEGAL under the Issue-940 law (the sort pins the read order), so
        // the sorted-key determinism tax is measured, not asserted from a
        // forbidden baseline. Interleaved 5x each; medians + ratio printed.
        let naive = |steps: &[StepHist]| -> Vec<(u64, u64)> {
            let mut m: std::collections::HashMap<u64, u64> =
                std::collections::HashMap::with_capacity(KEYS as usize);
            for step in steps {
                for &(k, c) in step.entries() {
                    *m.entry(k).or_insert(0) += c;
                }
            }
            let mut v: Vec<(u64, u64)> = m.into_iter().collect();
            v.sort_unstable_by_key(|&(k, _)| k);
            v
        };
        let mut ours = Vec::with_capacity(5);
        let mut theirs = Vec::with_capacity(5);
        for i in 0..10 {
            if i % 2 == 0 {
                let t = std::time::Instant::now();
                let _ = coverage_factor_with(&steps, &q, 0, &mut scratch);
                ours.push(t.elapsed().as_secs_f64() * 1e3);
            } else {
                let t = std::time::Instant::now();
                let _ = naive(&steps);
                theirs.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        ours.sort_by(|a, b| a.total_cmp(b));
        theirs.sort_by(|a, b| a.total_cmp(b));
        let (om, tm) = (ours[2], theirs[2]);
        println!(
            "g2 A/B medians: merge-fold {om:.2} ms vs naive-fold {tm:.2} ms (ratio {:.2}x)",
            om / tm
        );
    }
}
