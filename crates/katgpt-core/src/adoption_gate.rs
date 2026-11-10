//! Adoption/retention regression gate — AGCL form (Plan 633 T0; issue 987
//! priority-1, Research 657 §A5).
//!
//! A challenger's outcome stream is scored against the incumbent's gate
//! window `T` — how much of what the incumbent accepted it still accepts
//! (`retained`), and how much of what it outputs was already rejected under
//! the incumbent (`repeated`) — and adopted only when it does not regress
//! the incumbent on the same window. Accepted outcomes are decontaminated
//! first: every `stride`-th one files to a held-out audit set `A` that the
//! gate never reads, so the score is never computed over evidence the
//! challenger was fitted on. Filing to both `A` and `T` is
//! unrepresentable: [`Decontaminator::file`] returns ONE
//! [`FileDestination`] per outcome, and the sinks are separate.
//!
//! Modelless throughout: BLAKE3 digests, integer counts, ONE linear α mix.
//! **No softmax anywhere** — the score is `(1 − α)·retained + α·(1 −
//! repeated)`, a fixed-order f64 expression (the AGENTS.md Issue-940
//! determinism law: sequential folds in one pinned order, one strict tie
//! rule). Zero allocation: the rings are fixed `[T; 32]` arrays and
//! [`score_adoption`] takes slices and rings only. Zero new deps (blake3
//! is already non-optional).
//!
//! The `#[cfg(test)]` fixtures below are **SYNTHETIC GATE-MECHANICS
//! fixtures** — they prove the gate's arithmetic, dedup and decontamination
//! invariants, never serving quality. No number in them is quotable as a
//! model-quality result (the 976-corpus lesson).

use core::fmt;

/// Ring capacity — fixed, never grown (the plan's fixed `[T; 32]` law).
pub const RING_CAPACITY: usize = 32;

/// Default α — the linear mix weight on the repeated-rejected term.
pub const DEFAULT_ADOPTION_ALPHA: f64 = 0.5;

/// The ONE tie rule, pinned as a value: an exact score tie goes to the
/// INCUMBENT (katgpt-rs issue 994, landed 2026-11-10 — an uninformative
/// window declines, the decline-first law; measured in riir-refine Bench
/// 148: the old tie-adopts-challenger rule demoted a rule carrying an
/// accepted fix on a no-evidence window at the default α and demoted the
/// only productive rule at the α=1 endpoint). [`AdoptionGate::decide`]
/// selects its comparison BY this constant — `>` while it is `false`
/// (ties Keep; strict improvement still adopts); `true` restores the old
/// `>=` tie-adopts-challenger behavior.
pub const TIE_GOES_TO_CHALLENGER: bool = false;

/// Every `stride`-th ACCEPTED outcome files to the held-out audit set `A`;
/// every other outcome (accepted or rejected) files to the gate set `T`.
pub const DEFAULT_DECONTAMINATION_STRIDE: usize = 5;

/// A BLAKE3 digest over an outcome's bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutcomeDigest([u8; 32]);

impl OutcomeDigest {
    /// Wrap raw 32 bytes (round-trip partner of [`Self::as_bytes`]).
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The wrapped bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// BLAKE3 over the outcome's bytes.
    pub fn of(data: &[u8]) -> Self {
        Self(blake3::hash(data).into())
    }
}

/// Fixed 32-slot FIFO membership set over [`OutcomeDigest`]s.
///
/// Layout: a `usize` write cursor plus two parallel arrays —
/// `occupied: [bool; 32]` (32 B) and `slots: [[u8; 32]; 32]` (1024 B) —
/// chosen over `[Option<OutcomeDigest>; 32]` because `Option` costs a
/// discriminant byte per slot (a plain `[u8; 32]` has no niche) and splits
/// occupancy away from the contiguous mask. The arrays are align-1, so the
/// total is exact: 8 + 32 + 1024 = **1064 bytes**, no padding — pinned by
/// the `ring_layout_size_is_pinned` fixture. Zero allocation in every
/// operation; `contains` is a linear scan over the ≤ 32 slots (REQUIRED
/// shape, not a perf bug — the plan pins the scan).
#[derive(Clone)]
pub struct DedupRing {
    /// Next slot to overwrite (the oldest slot once the ring is full).
    next: usize,
    occupied: [bool; RING_CAPACITY],
    slots: [[u8; 32]; RING_CAPACITY],
}

impl DedupRing {
    pub const fn new() -> Self {
        Self {
            next: 0,
            occupied: [false; RING_CAPACITY],
            slots: [[0; 32]; RING_CAPACITY],
        }
    }

    /// Insert `digest`. Returns `true` when it was NEW (inserted), `false`
    /// when already present (seen — the ring is left untouched). At
    /// capacity the OLDEST slot is evicted.
    pub fn insert(&mut self, digest: OutcomeDigest) -> bool {
        if self.contains(&digest) {
            return false;
        }
        let slot = self.next;
        self.occupied[slot] = true;
        self.slots[slot] = *digest.as_bytes();
        self.next = (self.next + 1) % RING_CAPACITY;
        true
    }

    /// Linear membership scan over the ≤ 32 occupied slots.
    pub fn contains(&self, digest: &OutcomeDigest) -> bool {
        let bytes = digest.as_bytes();
        self.occupied
            .iter()
            .zip(self.slots.iter())
            .any(|(occupied, slot)| *occupied && slot == bytes)
    }

    /// Number of occupied slots (≤ [`RING_CAPACITY`]) — O(32) count, in
    /// slot-index order.
    pub fn len(&self) -> usize {
        self.occupied.iter().filter(|&&occupied| occupied).count()
    }

    /// True when no slot is occupied.
    pub fn is_empty(&self) -> bool {
        !self.occupied.iter().any(|occupied| *occupied)
    }

    /// Digests in slot-index order — a FIXED order (insertion order until
    /// the ring wraps; the wrap keeps slot-index order deterministic). The
    /// scorer's folds are exact integer counts, so the arithmetic never
    /// depends on this order; the order is pinned for reproducibility.
    pub fn iter(&self) -> impl Iterator<Item = OutcomeDigest> + '_ {
        self.occupied
            .iter()
            .zip(self.slots.iter())
            .filter(|&(occupied, _)| *occupied)
            .map(|(_, slot)| OutcomeDigest(*slot))
    }
}

impl fmt::Debug for DedupRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DedupRing")
            .field("len", &self.len())
            .field("next", &self.next)
            .finish()
    }
}

impl Default for DedupRing {
    fn default() -> Self {
        Self::new()
    }
}

/// Where the decontaminator files an outcome: the held-out audit set `A` or
/// the gate set `T`. One destination per outcome, by construction — there
/// is no API that files to both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileDestination {
    /// Held out — never enters the gate's T rings; `decide` never reads it.
    Audit,
    /// Gate window — enters the accepted/rejected T ring by its accepted bit.
    Gate,
}

/// Never-double-file splitter over an outcome stream.
///
/// Counts ACCEPTED outcomes only: every `stride`-th accepted outcome files
/// to [`FileDestination::Audit`], every other outcome (accepted or
/// rejected) files to [`FileDestination::Gate`]. Index-deterministic — a
/// pure count, no RNG — so the same stream always decontaminates
/// identically.
#[derive(Clone, Debug)]
pub struct Decontaminator {
    stride: usize,
    accepted_count: usize,
}

impl Decontaminator {
    /// `stride` must be ≥ 1 (a stride of 0 would hold out nothing and
    /// divide by zero on the modulo — `new` refuses it loud).
    pub fn new(stride: usize) -> Self {
        assert!(stride >= 1, "decontamination stride must be >= 1");
        Self {
            stride,
            accepted_count: 0,
        }
    }

    /// File one outcome. Returns the ONE destination it goes to.
    pub fn file(&mut self, accepted: bool) -> FileDestination {
        if !accepted {
            return FileDestination::Gate;
        }
        self.accepted_count += 1;
        if self.accepted_count.is_multiple_of(self.stride) {
            FileDestination::Audit
        } else {
            FileDestination::Gate
        }
    }
}

/// The pure scorer's output — three f64 numbers, no decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdoptionScore {
    /// Share of the previously-accepted window the candidate still accepts.
    pub retained: f64,
    /// Share of the candidate's outputs that were already rejected.
    pub repeated: f64,
    /// `(1 − α)·retained + α·(1 − repeated)` — higher is better.
    pub score: f64,
}

/// Pure adoption scorer — slices and rings in, three f64s out, ZERO
/// allocation, no Vec parameters.
///
/// - `retained = |candidate_accepted ∩ prev_accepted| / max(1, |prev_accepted|)`
/// - `repeated = |candidate_outputs ∩ prev_rejected| / max(1, |candidate_outputs|)`
/// - `score = (1 − α)·retained + α·(1 − repeated)` (linear mix — never a
///   softmax; `alpha` is the caller's weight, [`DEFAULT_ADOPTION_ALPHA`]
///   when in doubt)
///
/// Determinism (AGENTS.md Issue-940 law): intersections are sequential
/// counts over the candidate slices in slice order — exact integer
/// arithmetic, so the count value is order-independent by construction —
/// followed by ONE fixed f64 expression in the written order. The
/// incumbent's own score under the same formula is derived from the rings
/// by [`AdoptionGate::decide`] (always-from-base — no incumbent internals
/// are consumed).
pub fn score_adoption(
    candidate_accepted: &[OutcomeDigest],
    candidate_outputs: &[OutcomeDigest],
    prev_accepted: &DedupRing,
    prev_rejected: &DedupRing,
    alpha: f64,
) -> AdoptionScore {
    let mut retained_hits: usize = 0;
    for digest in candidate_accepted {
        if prev_accepted.contains(digest) {
            retained_hits += 1;
        }
    }
    let mut repeated_hits: usize = 0;
    for digest in candidate_outputs {
        if prev_rejected.contains(digest) {
            repeated_hits += 1;
        }
    }
    let retained = retained_hits as f64 / prev_accepted.len().max(1) as f64;
    let repeated = repeated_hits as f64 / candidate_outputs.len().max(1) as f64;
    let score = (1.0 - alpha) * retained + alpha * (1.0 - repeated);
    AdoptionScore {
        retained,
        repeated,
        score,
    }
}

/// The incumbent's own score over its T window, from the rings alone — the
/// allocation-free equivalent of running [`score_adoption`] on the
/// ring-derived slices (its accepted ring IS its accepted set; its outputs
/// on the window are accepted ∪ rejected — the rings are disjoint by the
/// ingest dedup). Equivalence is pinned bit-for-bit by the
/// `incumbent_score_matches_scorer_over_ring_slices` fixture.
fn incumbent_score(accepted: &DedupRing, rejected: &DedupRing, alpha: f64) -> AdoptionScore {
    let accepted_n = accepted.len();
    let rejected_n = rejected.len();
    let retained = accepted_n as f64 / accepted_n.max(1) as f64;
    let outputs_n = accepted_n + rejected_n;
    let repeated = rejected_n as f64 / outputs_n.max(1) as f64;
    let score = (1.0 - alpha) * retained + alpha * (1.0 - repeated);
    AdoptionScore {
        retained,
        repeated,
        score,
    }
}

/// The gate's decision over a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdoptionDecision {
    /// Challenger scores strictly above the incumbent on the same T
    /// window; an exact tie adopts only while [`TIE_GOES_TO_CHALLENGER`]
    /// is `true` (the pinned value is `false` — ties Keep, issue 994).
    Adopt,
    /// Challenger regresses the incumbent — keep what is running.
    Keep,
    /// Both T rings are empty — no evidence either way. The caller keeps
    /// the incumbent (decline-first law).
    NoEvidence,
}

/// The verdict: the candidate's three score numbers plus the decision.
/// Under [`AdoptionDecision::NoEvidence`] the score numbers are
/// informational only (the empty-window arithmetic).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdoptionVerdict {
    pub retained: f64,
    pub repeated: f64,
    pub score: f64,
    pub decision: AdoptionDecision,
}

/// The gate: the incumbent's two T-window rings (accepted, rejected), the
/// held-out audit sink, and the stream decontaminator.
///
/// [`Self::ingest`] files each NEW outcome to exactly one sink — audit
/// digests never enter the T rings; T-filed digests land in the accepted
/// or rejected ring by their accepted bit. A digest already present in
/// EITHER T ring is dedup'd: the gate updates nothing (not even the stride
/// counter) and reports it seen. [`Self::decide`] takes `&self` — pure
/// over the current window.
#[derive(Clone, Debug)]
pub struct AdoptionGate {
    accepted_ring: DedupRing,
    rejected_ring: DedupRing,
    audit_ring: DedupRing,
    decontaminator: Decontaminator,
}

impl AdoptionGate {
    pub fn new() -> Self {
        Self::with_stride(DEFAULT_DECONTAMINATION_STRIDE)
    }

    /// A gate whose decontaminator holds out every `stride`-th accepted
    /// outcome instead of [`DEFAULT_DECONTAMINATION_STRIDE`].
    pub fn with_stride(stride: usize) -> Self {
        Self {
            accepted_ring: DedupRing::new(),
            rejected_ring: DedupRing::new(),
            audit_ring: DedupRing::new(),
            decontaminator: Decontaminator::new(stride),
        }
    }

    /// File one stream outcome. Returns `true` when the digest was NEW
    /// (filed into exactly one sink); `false` when already present in a T
    /// ring (dedup — nothing updated, stride counter included).
    pub fn ingest(&mut self, accepted: bool, digest: OutcomeDigest) -> bool {
        if self.accepted_ring.contains(&digest) || self.rejected_ring.contains(&digest) {
            return false;
        }
        match self.decontaminator.file(accepted) {
            FileDestination::Audit => self.audit_ring.insert(digest),
            FileDestination::Gate => {
                if accepted {
                    self.accepted_ring.insert(digest)
                } else {
                    self.rejected_ring.insert(digest)
                }
            }
        }
    }

    /// The incumbent's T-accepted ring (gate set).
    pub fn accepted(&self) -> &DedupRing {
        &self.accepted_ring
    }

    /// The incumbent's T-rejected ring (gate set).
    pub fn rejected(&self) -> &DedupRing {
        &self.rejected_ring
    }

    /// The held-out audit sink `A` — never consulted by [`Self::decide`].
    pub fn audit(&self) -> &DedupRing {
        &self.audit_ring
    }

    /// Decide a challenger against the incumbent's current T window.
    ///
    /// - Both T rings empty → [`AdoptionDecision::NoEvidence`] → the caller
    ///   keeps the incumbent (decline-first law).
    /// - Otherwise the incumbent's own score under the same formula is
    ///   derived from the rings (always-from-base), and the challenger is
    ///   ADOPTED iff its score beats the incumbent's under the pinned tie
    ///   rule ([`TIE_GOES_TO_CHALLENGER`] — `false` since issue 994: an
    ///   exact tie is a no-evidence state and KEEPS the incumbent,
    ///   decline-first; strict improvement adopts).
    pub fn decide(
        &self,
        candidate_accepted: &[OutcomeDigest],
        candidate_outputs: &[OutcomeDigest],
        alpha: f64,
    ) -> AdoptionVerdict {
        let candidate = score_adoption(
            candidate_accepted,
            candidate_outputs,
            &self.accepted_ring,
            &self.rejected_ring,
            alpha,
        );
        if self.accepted_ring.is_empty() && self.rejected_ring.is_empty() {
            return AdoptionVerdict {
                retained: candidate.retained,
                repeated: candidate.repeated,
                score: candidate.score,
                decision: AdoptionDecision::NoEvidence,
            };
        }
        let incumbent = incumbent_score(&self.accepted_ring, &self.rejected_ring, alpha);
        // The pinned tie rule SELECTS the comparison (`>=` vs `>`); it must
        // never gate the whole adoption — a plain `&&` here would turn every
        // consult into Keep once the constant is false (issue 994 verdict
        // correction). Pinned `false`: ties Keep (a tie is a no-evidence
        // state, decline-first); strict improvement adopts.
        let beats = if TIE_GOES_TO_CHALLENGER {
            candidate.score >= incumbent.score
        } else {
            candidate.score > incumbent.score
        };
        let decision = if beats {
            AdoptionDecision::Adopt
        } else {
            AdoptionDecision::Keep
        };
        AdoptionVerdict {
            retained: candidate.retained,
            repeated: candidate.repeated,
            score: candidate.score,
            decision,
        }
    }
}

impl Default for AdoptionGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    /// Deterministic distinct digest per index — SYNTHETIC, gate mechanics
    /// only.
    fn digest_of(index: u64) -> OutcomeDigest {
        OutcomeDigest::of(&index.to_le_bytes())
    }

    /// A fresh default-stride gate ingesting `accepted_n` accepted outcomes
    /// (digests 0..n) then `rejected_m` rejected ones (digests 1000..).
    /// At stride 5 the 5th/10th/…th ACCEPTED outcome files to the audit
    /// sink; all rejected outcomes file to the gate set.
    fn gate_from_accepted_then_rejected(accepted_n: usize, rejected_m: usize) -> AdoptionGate {
        let mut gate = AdoptionGate::new();
        for i in 0..accepted_n as u64 {
            gate.ingest(true, digest_of(i));
        }
        for j in 0..rejected_m as u64 {
            gate.ingest(false, digest_of(1000 + j));
        }
        gate
    }

    #[test]
    fn ring_layout_size_is_pinned() {
        // next: usize (8 B) + occupied [bool; 32] (32 B) + slots
        // [[u8; 32]; 32] (1024 B) — align-1 arrays behind the cursor, no
        // padding: 1064 exactly. A layout regression would silently change
        // the ring's memory contract.
        assert_eq!(core::mem::size_of::<DedupRing>(), 1064);
        assert_eq!(core::mem::size_of::<OutcomeDigest>(), 32);
    }

    /// SYNTHETIC GATE-MECHANICS fixture 1 — the AGCL failure class: a
    /// challenger that drops a third of the previously-accepted outcomes
    /// and repeats a previously-rejected one scores BELOW the incumbent and
    /// the gate KEEPS the incumbent. Never a serving-quality number.
    #[test]
    fn degraded_challenger_keeps_incumbent() {
        // 15 accepted ingests at stride 5 → audit {4, 9, 14}, 12 T-accepted;
        // 2 rejected ingests → 2 T-rejected.
        let gate = gate_from_accepted_then_rejected(15, 2);
        assert_eq!(gate.accepted().len(), 12);
        assert_eq!(gate.rejected().len(), 2);
        assert_eq!(gate.audit().len(), 3);

        let accepted_ring: Vec<OutcomeDigest> = gate.accepted().iter().collect();
        let candidate_accepted: Vec<OutcomeDigest> = accepted_ring[..8].to_vec();
        let mut candidate_outputs = candidate_accepted.clone();
        candidate_outputs.push(digest_of(1000)); // repeats a previously-rejected output
        candidate_outputs.push(digest_of(7000)); // novel output

        let verdict = gate.decide(
            &candidate_accepted,
            &candidate_outputs,
            DEFAULT_ADOPTION_ALPHA,
        );
        assert_eq!(verdict.decision, AdoptionDecision::Keep);
        // Dropped 4 of the 12 previously-accepted (a third), repeated 1 of 10.
        assert_eq!(verdict.retained, 8.0 / 12.0);
        assert_eq!(verdict.repeated, 1.0 / 10.0);
        assert!(verdict.score < 1.0);
        // The incumbent, scored by the same formula over the same window, wins.
        let incumbent = incumbent_score(gate.accepted(), gate.rejected(), DEFAULT_ADOPTION_ALPHA);
        assert!(verdict.score < incumbent.score);
    }

    /// SYNTHETIC GATE-MECHANICS fixture 2 — a challenger that retains every
    /// previously-accepted outcome and never repeats a rejected one ADOPTS
    /// (with [`TIE_GOES_TO_CHALLENGER`] pinned `false`: the guard that
    /// adoption stays live — a plain `&&` on the constant would turn this
    /// arm Keep too); an exact score tie at the default α KEEPS the
    /// incumbent, and the α=1 zero-rejection tie KEEPS as well (issue 994's
    /// measured endpoint). Never a serving-quality number.
    #[test]
    fn genuine_improvement_adopts_and_ties_keep() {
        let gate = gate_from_accepted_then_rejected(15, 2);
        let candidate_accepted: Vec<OutcomeDigest> = gate.accepted().iter().collect();
        let candidate_outputs = candidate_accepted.clone();

        let verdict = gate.decide(
            &candidate_accepted,
            &candidate_outputs,
            DEFAULT_ADOPTION_ALPHA,
        );
        assert_eq!(verdict.decision, AdoptionDecision::Adopt);
        assert_eq!(verdict.retained, 1.0);
        assert_eq!(verdict.repeated, 0.0);
        assert_eq!(verdict.score, 1.0);

        // Exact-tie arm (issue 994): an incumbent with no rejected outcomes
        // scores 1.0 at the default α; the identical challenger also scores
        // 1.0 — the tie KEEPS the incumbent. Bit-exact tie: both sides run
        // the identical expression.
        let tie_gate = gate_from_accepted_then_rejected(15, 0);
        assert_eq!(tie_gate.rejected().len(), 0);
        let tie_accepted: Vec<OutcomeDigest> = tie_gate.accepted().iter().collect();
        let tie_verdict = tie_gate.decide(&tie_accepted, &tie_accepted, DEFAULT_ADOPTION_ALPHA);
        assert_eq!(tie_verdict.score, 1.0);
        assert_eq!(tie_verdict.decision, AdoptionDecision::Keep);

        // The α=1 endpoint (issue 994's measured degeneracy): on the same
        // zero-rejection window both sides score exactly 1.0 at α=1 — the
        // tie KEEPS the incumbent instead of demoting it.
        let endpoint_verdict = tie_gate.decide(&tie_accepted, &tie_accepted, 1.0);
        assert_eq!(endpoint_verdict.score, 1.0);
        assert_eq!(endpoint_verdict.decision, AdoptionDecision::Keep);
    }

    /// SYNTHETIC GATE-MECHANICS fixture 3 — a 600-outcome stream at stride
    /// 5 never double-files: the audit set and the gate set are disjoint
    /// (exactly one destination per outcome), and the stride counts
    /// ACCEPTED outcomes only (rejected outcomes never advance it).
    #[test]
    fn never_double_file_over_long_stream() {
        const TOTAL: usize = 600;
        let mut decon = Decontaminator::new(DEFAULT_DECONTAMINATION_STRIDE);
        let mut audit_set = HashSet::new();
        let mut gate_set = HashSet::new();
        let mut audit_filed = 0usize;
        let mut gate_accepted = 0usize;
        let mut gate_rejected = 0usize;
        for i in 0..TOTAL as u64 {
            let accepted = i % 6 != 5; // every 6th outcome is rejected
            let digest = digest_of(i);
            match decon.file(accepted) {
                FileDestination::Audit => {
                    audit_set.insert(digest);
                    audit_filed += 1;
                }
                FileDestination::Gate => {
                    gate_set.insert(digest);
                    if accepted {
                        gate_accepted += 1;
                    } else {
                        gate_rejected += 1;
                    }
                }
            }
        }
        assert_eq!(audit_filed, 100); // 500 accepted / stride 5
        assert_eq!(gate_accepted, 400);
        assert_eq!(gate_rejected, 100); // stride counts ACCEPTED only
        assert!(audit_set.is_disjoint(&gate_set));

        // The gate's own ingest path agrees: its audit sink and its T rings
        // stay disjoint, every audit digest it still holds came from the
        // audit set, and every T digest came from the gate set.
        let mut gate = AdoptionGate::new();
        for i in 0..TOTAL as u64 {
            gate.ingest(i % 6 != 5, digest_of(i));
        }
        assert_eq!(gate.audit().len(), RING_CAPACITY); // 100 audited, 32 held
        for digest in gate.audit().iter() {
            assert!(audit_set.contains(&digest));
            assert!(!gate.accepted().contains(&digest));
            assert!(!gate.rejected().contains(&digest));
        }
        for digest in gate.accepted().iter().chain(gate.rejected().iter()) {
            assert!(!gate.audit().contains(&digest));
            assert!(gate_set.contains(&digest));
        }
    }

    /// SYNTHETIC GATE-MECHANICS fixture 4 — the ring bound: 40 distinct
    /// digests evict the oldest 8, keep the recent 32; a duplicate insert
    /// reports seen; an evicted digest is NEW again.
    #[test]
    fn ring_evicts_oldest_at_capacity() {
        let mut ring = DedupRing::new();
        for i in 0..40u64 {
            assert!(
                ring.insert(digest_of(i)),
                "distinct digest {i} must insert as NEW"
            );
        }
        assert_eq!(ring.len(), RING_CAPACITY);
        for i in 0..8u64 {
            assert!(!ring.contains(&digest_of(i)), "oldest 8 must be evicted");
        }
        for i in 8..40u64 {
            assert!(ring.contains(&digest_of(i)));
        }
        assert!(!ring.insert(digest_of(39)), "duplicate insert reports seen");
        assert!(ring.insert(digest_of(0)), "an evicted digest is NEW again");
        assert_eq!(ring.len(), RING_CAPACITY);
    }

    /// SYNTHETIC GATE-MECHANICS fixture 5 — purity/determinism: the same
    /// inputs replayed twice give bit-identical verdicts (f64::to_bits
    /// compared, not ==) and the same decisions.
    #[test]
    fn scorer_is_pure_and_deterministic() {
        let gate = gate_from_accepted_then_rejected(15, 2);
        let candidate_accepted: Vec<OutcomeDigest> = gate.accepted().iter().take(8).collect();
        let mut candidate_outputs = candidate_accepted.clone();
        candidate_outputs.push(digest_of(1000));
        candidate_outputs.push(digest_of(7000));

        let first = gate.decide(
            &candidate_accepted,
            &candidate_outputs,
            DEFAULT_ADOPTION_ALPHA,
        );
        let second = gate.decide(
            &candidate_accepted,
            &candidate_outputs,
            DEFAULT_ADOPTION_ALPHA,
        );
        assert_eq!(first.retained.to_bits(), second.retained.to_bits());
        assert_eq!(first.repeated.to_bits(), second.repeated.to_bits());
        assert_eq!(first.score.to_bits(), second.score.to_bits());
        assert_eq!(first.decision, second.decision);

        let score_a = score_adoption(
            &candidate_accepted,
            &candidate_outputs,
            gate.accepted(),
            gate.rejected(),
            DEFAULT_ADOPTION_ALPHA,
        );
        let score_b = score_adoption(
            &candidate_accepted,
            &candidate_outputs,
            gate.accepted(),
            gate.rejected(),
            DEFAULT_ADOPTION_ALPHA,
        );
        assert_eq!(score_a.retained.to_bits(), score_b.retained.to_bits());
        assert_eq!(score_a.repeated.to_bits(), score_b.repeated.to_bits());
        assert_eq!(score_a.score.to_bits(), score_b.score.to_bits());
    }

    /// The ring-derived incumbent score is the allocation-free equivalent
    /// of the pure scorer over the ring-derived slices — pinned
    /// bit-for-bit on all three numbers.
    #[test]
    fn incumbent_score_matches_scorer_over_ring_slices() {
        let gate = gate_from_accepted_then_rejected(15, 3);
        let accepted_slice: Vec<OutcomeDigest> = gate.accepted().iter().collect();
        let outputs_slice: Vec<OutcomeDigest> = gate
            .accepted()
            .iter()
            .chain(gate.rejected().iter())
            .collect();
        let via_scorer = score_adoption(
            &accepted_slice,
            &outputs_slice,
            gate.accepted(),
            gate.rejected(),
            DEFAULT_ADOPTION_ALPHA,
        );
        let via_rings = incumbent_score(gate.accepted(), gate.rejected(), DEFAULT_ADOPTION_ALPHA);
        assert_eq!(via_scorer.retained.to_bits(), via_rings.retained.to_bits());
        assert_eq!(via_scorer.repeated.to_bits(), via_rings.repeated.to_bits());
        assert_eq!(via_scorer.score.to_bits(), via_rings.score.to_bits());
    }

    /// Ingest dedup: a digest present in either T ring reports seen and
    /// updates nothing; audit-filed digests never enter the T rings.
    #[test]
    fn ingest_dedups_and_reports_seen() {
        let mut gate = AdoptionGate::new();
        assert!(gate.ingest(true, digest_of(0)));
        assert!(gate.ingest(false, digest_of(1)));
        assert!(!gate.ingest(true, digest_of(0)), "same T ring reports seen");
        assert!(
            !gate.ingest(false, digest_of(0)),
            "the OTHER T ring reports seen too"
        );
        assert_eq!(gate.accepted().len(), 1);
        assert_eq!(gate.rejected().len(), 1);

        // Stride hold-out: at stride 5 the 5th accepted outcome files to
        // the audit sink, never to the accepted ring.
        let mut held_out = AdoptionGate::new();
        for i in 0..5u64 {
            held_out.ingest(true, digest_of(i));
        }
        assert_eq!(held_out.audit().len(), 1);
        assert!(held_out.audit().contains(&digest_of(4)));
        assert!(!held_out.accepted().contains(&digest_of(4)));
        assert_eq!(held_out.accepted().len(), 4);
    }

    /// The decline-first law: with both T rings empty the gate reports
    /// NoEvidence (the caller keeps the incumbent); the score numbers are
    /// the empty-window arithmetic, informational only.
    #[test]
    fn no_evidence_declines() {
        let gate = AdoptionGate::new();
        let candidate = [digest_of(0), digest_of(1)];
        let verdict = gate.decide(&candidate, &candidate, DEFAULT_ADOPTION_ALPHA);
        assert_eq!(verdict.decision, AdoptionDecision::NoEvidence);
        assert_eq!(verdict.retained, 0.0);
        assert_eq!(verdict.repeated, 0.0);
        assert_eq!(verdict.score, DEFAULT_ADOPTION_ALPHA);
    }
}
