//! Pool admission — hysteresis admission policy for a fixed-capacity resident
//! set (Issue 873 primitive A; Research 581 mechanism rows #6/#7/#8, distilled
//! from volotat/mini-AGI `paged.py:569-714` @ `96784b7`, MIT).
//!
//! # The policy
//!
//! A candidate displaces the weakest **evictable** resident only when its
//! want-score beats the victim's by the multiplicative [`DEFAULT_MARGIN`]
//! (×1.10); a newcomer is un-evictable for `dwell` ticks measured from its
//! **admission tick, never a last-use tick**; during cold start exactly one
//! never-admitted item receives a fair turn per cycle (terminating sweep,
//! re-arms only on universe growth); the accepted set is read out by
//! identity so the caller's load count is the true set delta.
//!
//! The three hysteresis legs and what each refuses:
//!
//! | Leg | Refuses |
//! |---|---|
//! | margin ×1.10 | churn on noise — a candidate within 10% of the weakest resident moves nothing (the set is stable under a flat distribution) |
//! | dwell-from-admission | eviction cascades over newborns — a freshly admitted resident gets `dwell` ticks to earn usage before it can be displaced |
//! | fair-turn sweep | starvation of never-demanded items — without it, a zero-observed-want item can never enter, so the system can never discover it is wanted |
//!
//! # The clock law (measured the hard way at mini-AGI `paged.py:242-247`)
//!
//! Residents are touched every cycle (want-scores refresh). A dwell clock
//! reading a **last-use** tick would therefore read every resident as
//! permanently young and the set could never move. [`ResidentRow`] carries
//! **no last-use field at all** — the invariant holds by construction (the
//! `calibration_staleness` posture), and
//! [`since_admission_not_last_use_is_the_dwell_clock`] pins the behavior:
//! a resident touched every single cycle becomes evictable at exactly
//! `admission_tick + dwell`.
//!
//! # Leaf-clean (the `suspect_indices` house pattern)
//!
//! Want-scores are CALLER-SUPPLIED — this module never produces a signal,
//! it only governs over one. The natural producer in this crate is
//! [`crate::kv_eviction::score`] (usage-rate `cum_mass/max(1,age)`, Plan 585):
//! feed its output as `want`. The resident STORAGE is also caller-shaped:
//! the free function [`admission_decision`] decides over caller-owned rows;
//! [`AdmissionSet`] is the convenience layer that owns rows + the fair-turn
//! sweep state. Slot allocation itself is not re-implemented here —
//! index-stable pools are [`graph_stable_pool`](crate::GraphStablePool)
//! territory (Issue 800 Arm C); this module is the *policy* over a resident
//! set, not its allocator.
//!
//! # Prior art (Issue 873 A4)
//!
//! Nearest published cousin is **TinyLFU / W-TinyLFU** (Caffeine; admission
//! by frequency-sketch candidate-vs-victim compare). Deltas, per Research
//! 581 §4: the dwell immunity window, the admission-tick clock (vs sketch
//! recency), the **terminating** fair-turn sweep (vs W-TinyLFU's permanent
//! admission window), and a multiplicative margin over arbitrary
//! caller-supplied want-scores (no sketch, no frequency assumption — works
//! over attention mass, demand predictions, or any bounded score).
//!
//! # Sync boundary
//!
//! None — pure policy over caller-owned state; no sync surfaces are touched
//! (the [`crate::kv_eviction`] posture). Decisions are deterministic in
//! `(rows, candidate_want, tick, config)`; the only floats are the
//! caller-supplied want-scores, compared with `>` against
//! `victim_want * margin` — never hashed, never committed, never synced.
//!
//! # Allocation
//!
//! [`AdmissionSet::new`] allocates once; the steady-state paths
//! (`consider`, `consider_fair`, `update_want`, `fair_turn_next`,
//! `resident_ids_into` into a caller-owned buffer) never allocate. The
//! fair-turn `had_turn` vector grows ONLY on universe growth — a cold
//! event, amortized, never on the per-cycle path.
//!
//! pool_admission — hysteresis admission policy for a fixed-capacity resident
//! set (Issue 873 primitive A; Research 581 rows #6/#7/#8): margin ×1.10 over
//! caller-supplied want-scores, dwell-from-ADMISSION-tick newborn immunity
//! (no last-use field by construction — the clock law), terminating
//! fair-turn sweep, apply-by-identity readout. The admission complement of
//! kv_eviction (which scores whom to KEEP); want-scores caller-supplied, slot
//! allocation stays graph_stable_pool's concern. Zero steady-state allocs,
//! pure std. OPT-IN pending first consumer GOAT (riir-ai working sets /
//! PagedKVCache, riir-neuron-db zone_cache).

use crate::float_order;

/// Source-pinned admission margin: a candidate must beat the weakest evictable
/// resident by ×1.10 (mini-AGI `paged.py`, `MARGIN = 1.10`). 10% hysteresis —
/// small enough that a real distribution shift turns the set over in a few
/// cycles, large enough that a flat distribution moves nothing.
pub const DEFAULT_MARGIN: f32 = 1.10;

/// Admission policy knobs. Knobs are config, NEVER adaptive (the Issue-033
/// adaptive-blend negative law): the caller pins them at construction and the
/// policy never mutates its own config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolAdmissionConfig {
    /// Multiplicative admission margin: admit only when
    /// `candidate_want > victim_want * margin`. [`DEFAULT_MARGIN`] is the
    /// source-pinned value.
    pub margin: f32,
    /// Newborn immunity in ticks, measured from the ADMISSION tick. A
    /// resident is evictable iff `tick - admission_tick >= dwell`. Scale to
    /// the caller's tick cadence (mini-AGI measures this in its own
    /// admission passes — there is no source-pinned universal value).
    pub dwell: u64,
}

impl PoolAdmissionConfig {
    /// Config with the source-pinned margin and an explicit dwell. The dwell
    /// is deliberately a required argument — there is no honest default for
    /// an unknown tick cadence.
    pub const fn with_dwell(dwell: u64) -> Self {
        Self {
            margin: DEFAULT_MARGIN,
            dwell,
        }
    }
}

/// One resident's admission bookkeeping. The caller owns the rows when using
/// the free functions; [`AdmissionSet`] owns them in the convenience layer.
///
/// Deliberately carries **no last-use field**: the dwell clock must read the
/// admission tick (see the module doc clock law). Adding a last-use tick here
/// is the exact bug this type exists to make unrepresentable.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResidentRow {
    /// Caller identity. The accepted set is applied by identity match, so
    /// the caller's load count is the true set delta.
    pub id: u64,
    /// Last-known want-score (caller-supplied signal; refreshed as often as
    /// the caller observes — refreshing does NOT touch the dwell clock).
    pub want: f32,
    /// Tick at which this resident was admitted. The dwell clock reads THIS.
    pub admission_tick: u64,
}

/// Pure admission decision over caller-owned rows. Answers only the full-set
/// question — a caller with a free slot inserts without asking.
///
/// `rows[..live]` are the residents; `live <= rows.len()`. O(live): a single
/// scan for the weakest evictable resident. NaN-safe by contract: resident
/// want-scores are finite (callers refresh via finite-supplying paths; the
/// [`AdmissionSet`] refresh drops non-finite loudly), and a non-finite
/// `candidate_want` rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// Admit the candidate; `evict` names the resident ROW INDEX to
    /// displace. The newcomer's `admission_tick` must be stamped with the
    /// decision tick (its own dwell window starts now).
    Admit { evict: usize },
    /// Reject: no evictable resident (all newborns), or the candidate did
    /// not beat the weakest evictable resident by `margin`. Either way the
    /// set does not move — hysteresis.
    Reject,
}

/// Scan `rows[..live]` for the weakest evictable resident at `tick`.
/// Evictable = `tick - admission_tick >= dwell`. Weakest = minimum `want`
/// (ties: lowest index — deterministic). O(live), zero-alloc.
pub fn weakest_evictable(
    rows: &[ResidentRow],
    live: usize,
    tick: u64,
    dwell: u64,
) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for (i, row) in rows[..live.min(rows.len())].iter().enumerate() {
        if tick.saturating_sub(row.admission_tick) < dwell {
            continue; // newborn — un-evictable regardless of want
        }
        // float_order::asc: a TOTAL order; NaN sorts above every real, so a
        // poisoned want can never look weakest-to-evict (the "corrupt must
        // never look cheap" law). Ties keep the first (lowest) index.
        match best {
            Some((w, _)) if !matches!(float_order::asc(row.want, w), core::cmp::Ordering::Less) => {
            }
            _ => best = Some((row.want, i)),
        }
    }
    best.map(|(_, i)| i)
}

/// Pure margin test against a named victim. `true` iff
/// `candidate_want > victim_want * margin` (non-finite candidate rejects;
/// victim wants are finite by construction — a poisoned victim makes the
/// product NaN and `> NaN` is `false`, so poison can only ever refuse to
/// move the set, never corrupt it).
fn beats_margin(candidate_want: f32, victim_want: f32, margin: f32) -> bool {
    candidate_want.is_finite() && candidate_want > victim_want * margin
}

/// Pure decision: which evictable resident (if any) does `candidate_want`
/// displace at `tick`? See [`AdmissionDecision`]. Leaf-clean core over
/// caller-owned state — the [`AdmissionSet::consider`] convenience layer
/// wraps exactly this logic.
pub fn admission_decision(
    rows: &[ResidentRow],
    live: usize,
    candidate_want: f32,
    tick: u64,
    config: &PoolAdmissionConfig,
) -> AdmissionDecision {
    let Some(victim) = weakest_evictable(rows, live, tick, config.dwell) else {
        return AdmissionDecision::Reject;
    };
    if beats_margin(candidate_want, rows[victim].want, config.margin) {
        AdmissionDecision::Admit { evict: victim }
    } else {
        AdmissionDecision::Reject
    }
}

/// Outcome of [`AdmissionSet::consider`] / [`AdmissionSet::consider_fair`].
/// `displaced` carries the evicted resident's ID (not slot) — the caller
/// needs identity, not layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionOutcome {
    /// Admitted. `displaced: Some(victim_id)` = a resident was evicted;
    /// `None` = a free slot was filled. The newcomer's dwell window starts
    /// at the decision tick.
    Admitted { displaced: Option<u64> },
    /// Rejected — the set did not move. (Margin test failed, everything
    /// newborn, or a non-finite want: never poison the set.)
    Rejected,
    /// The candidate is already resident (identity match). Its want-score
    /// was refreshed; no slot moved, nothing will load twice. This is the
    /// apply-by-identity contract: re-considering a resident is a no-op
    /// beyond the refresh.
    AlreadyResident,
}

/// Stateful admission set: capacity-bounded resident rows + the fair-turn
/// sweep state. Allocation happens at [`AdmissionSet::new`] and on universe
/// growth only — every steady-state method is zero-alloc (G4-pinned).
///
/// Fair-turn sweep contract: the sweep universe is the DENSE id range
/// `0..universe_len` (mini-AGI's expert indices; callers with sparse ids map
/// their own space and use only the admission half). Each id receives AT
/// MOST ONE designation ever, in index order, one per
/// [`fair_turn_next`](Self::fair_turn_next) call; once every id has had its
/// turn the sweep terminates for good; only
/// [`note_universe`](Self::note_universe) growth re-arms it (new ids are
/// never-designated by construction).
pub struct AdmissionSet {
    config: PoolAdmissionConfig,
    capacity: usize,
    rows: Vec<ResidentRow>,
    live: usize,
    // Fair-turn sweep (cold-start anti-starvation). `had_turn` grows ONLY on
    // universe growth — a cold, amortized event, never per-cycle. `pending`
    // counts never-designated ids so a TERMINATED sweep rejects in O(1)
    // instead of rescanning the universe every cycle.
    had_turn: Vec<bool>,
    pending: usize,
    sweep_cursor: usize,
    universe_len: usize,
}

impl AdmissionSet {
    /// Allocate once for `capacity` residents. The set starts empty;
    /// fill free slots first ([`Self::consider`] admits into free slots
    /// unconditionally — the margin policy governs displacement only).
    pub fn new(capacity: usize, config: PoolAdmissionConfig) -> Self {
        Self {
            config,
            capacity,
            rows: Vec::with_capacity(capacity),
            live: 0,
            had_turn: Vec::new(),
            pending: 0,
            sweep_cursor: 0,
            universe_len: 0,
        }
    }

    /// Resident capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Live resident count.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Config readout (the policy never mutates its own config).
    pub fn config(&self) -> &PoolAdmissionConfig {
        &self.config
    }

    /// Is `id` currently resident? O(live).
    pub fn contains(&self, id: u64) -> bool {
        self.rows[..self.live].iter().any(|r| r.id == id)
    }

    /// Read the resident row for `id`. O(live).
    pub fn row_for(&self, id: u64) -> Option<&ResidentRow> {
        self.rows[..self.live].iter().find(|r| r.id == id)
    }

    /// Refresh a resident's want-score. This is the every-cycle touch — it
    /// does NOT touch the dwell clock (the row has no last-use field).
    /// Non-finite wants are dropped `debug_assert!`-loud (the
    /// [`crate::kv_eviction::observe`] posture: never poison; a violation is
    /// a caller bug). Unknown ids are ignored the same way. O(live).
    pub fn update_want(&mut self, id: u64, want: f32) {
        if !want.is_finite() {
            debug_assert!(
                false,
                "pool_admission::update_want: want must be finite, got {want} for id {id}"
            );
            return;
        }
        if let Some(row) = self.rows[..self.live].iter_mut().find(|r| r.id == id) {
            row.want = want;
        }
    }

    /// Consider `id` for admission at `tick` with caller-supplied `want`.
    ///
    /// - Already resident → refresh want, [`AdmissionOutcome::AlreadyResident`].
    /// - Free slot exists → admit unconditionally (the margin policy governs
    ///   displacement, not filling an empty set).
    /// - Full → [`admission_decision`]: displace the weakest evictable
    ///   resident iff `want > victim_want * margin`; the newcomer's
    ///   `admission_tick` is stamped `tick`.
    ///
    /// Non-finite `want` rejects `debug_assert!`-loud. O(live), zero-alloc.
    pub fn consider(&mut self, id: u64, want: f32, tick: u64) -> AdmissionOutcome {
        if let Some(outcome) = self.consider_inner(id, want, tick, false) {
            return outcome;
        }
        // margin path: full set — the pure core decides.
        match admission_decision(&self.rows, self.live, want, tick, &self.config) {
            AdmissionDecision::Admit { evict } => {
                let victim_id = self.rows[evict].id;
                self.rows[evict] = ResidentRow {
                    id,
                    want,
                    admission_tick: tick,
                };
                AdmissionOutcome::Admitted {
                    displaced: Some(victim_id),
                }
            }
            AdmissionDecision::Reject => AdmissionOutcome::Rejected,
        }
    }

    /// Fair-turn admission: the margin test is WAIVED (the whole point of a
    /// fair turn — a never-resident item has no observed usage, so its
    /// want-score cannot clear any margin; it must be resident to earn
    /// evidence). Dwell is still honored: only an EVICTABLE resident is
    /// displaced, and a full set of newborns still rejects. Call this only
    /// for ids designated by
    /// [`fair_turn_next`](Self::fair_turn_next) — the caller owns that
    /// discipline (one fair admission per cycle).
    pub fn consider_fair(&mut self, id: u64, want: f32, tick: u64) -> AdmissionOutcome {
        if let Some(outcome) = self.consider_inner(id, want, tick, true) {
            return outcome;
        }
        match weakest_evictable(&self.rows, self.live, tick, self.config.dwell) {
            Some(slot) => {
                let victim_id = self.rows[slot].id;
                self.rows[slot] = ResidentRow {
                    id,
                    want,
                    admission_tick: tick,
                };
                AdmissionOutcome::Admitted {
                    displaced: Some(victim_id),
                }
            }
            None => AdmissionOutcome::Rejected,
        }
    }

    /// Shared prefix of both consider paths: identity check, non-finite
    /// guard, free-slot fill. Returns `None` to fall through to the
    /// full-set displacement policy.
    fn consider_inner(
        &mut self,
        id: u64,
        want: f32,
        tick: u64,
        _fair: bool,
    ) -> Option<AdmissionOutcome> {
        if !want.is_finite() {
            debug_assert!(
                false,
                "pool_admission::consider: want must be finite, got {want} for id {id}"
            );
            return Some(AdmissionOutcome::Rejected);
        }
        if let Some(row) = self.rows[..self.live].iter_mut().find(|r| r.id == id) {
            row.want = want; // identity refresh — no slot moves
            return Some(AdmissionOutcome::AlreadyResident);
        }
        if self.live < self.capacity {
            self.rows.push(ResidentRow {
                id,
                want,
                admission_tick: tick,
            });
            self.live += 1;
            return Some(AdmissionOutcome::Admitted { displaced: None });
        }
        None // full set — displacement policy decides
    }

    /// Read the accepted set into a caller-owned buffer (zero-alloc with a
    /// reused `out`). THE apply-by-identity readout: diff this against the
    /// caller's loaded set; the load count is the true set delta — evictions
    /// and admissions both surface here, by identity, never by slot.
    pub fn resident_ids_into(&self, out: &mut Vec<u64>) {
        out.clear();
        out.extend(self.rows[..self.live].iter().map(|r| r.id));
    }

    /// Grow the fair-turn universe to `universe_len` (never shrinks; a
    /// smaller value is ignored). New ids `old_len..universe_len` are
    /// never-designated by construction, which re-arms the sweep — this is
    /// the ONLY re-arm. Cold path: may grow the `had_turn` vector.
    pub fn note_universe(&mut self, universe_len: usize) {
        if universe_len > self.universe_len {
            self.pending += universe_len - self.universe_len;
            self.universe_len = universe_len;
            if self.had_turn.len() < universe_len {
                self.had_turn.resize(universe_len, false);
            }
        }
    }

    /// Current fair-turn universe length.
    pub fn universe_len(&self) -> usize {
        self.universe_len
    }

    /// Number of universe ids that have NOT yet had their fair turn (the
    /// pending sweep backlog — 0 means the sweep has terminated). O(1).
    pub fn fair_turn_pending(&self) -> usize {
        self.pending
    }

    /// Designate the next fair-turn id: the lowest-then-wrapping index that
    /// has never had a turn, ONE per call (one per cycle is the caller's
    /// cadence). Marks the id as turned at designation time — the turn is
    /// the designation, not its outcome (a fair admission that rejects on a
    /// full-newborn set does not un-turn the id; the sweep still terminates).
    /// Returns `None` once every id has had its turn; only
    /// [`note_universe`](Self::note_universe) growth re-arms it.
    /// Zero-alloc, O(universe_len) worst case (typical O(1) from cursor).
    pub fn fair_turn_next(&mut self) -> Option<u64> {
        if self.pending == 0 {
            return None; // terminated — O(1), no universe rescan
        }
        debug_assert!(
            self.universe_len > 0,
            "pending>0 with empty universe is corrupt"
        );
        for step in 0..self.universe_len {
            let idx = (self.sweep_cursor + step) % self.universe_len;
            if !self.had_turn[idx] {
                self.had_turn[idx] = true;
                self.sweep_cursor = (idx + 1) % self.universe_len;
                self.pending -= 1;
                return Some(idx as u64);
            }
        }
        // Unreachable while pending is maintained correctly (pending>0 ⇒ an
        // undesignated id exists); kept as a fail-loud invariant, not a
        // silent pass.
        debug_assert!(
            false,
            "fair_turn pending={}>0 but sweep found none",
            self.pending
        );
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic resident table helper: `id` = index, want = `wants[i]`.
    fn full_set(wants: &[f32], admission_tick: u64) -> AdmissionSet {
        let mut s = AdmissionSet::new(wants.len(), PoolAdmissionConfig::with_dwell(8));
        for (i, &w) in wants.iter().enumerate() {
            match s.consider(i as u64, w, admission_tick) {
                AdmissionOutcome::Admitted { displaced: None } => {}
                other => panic!("fixture fill expected free-slot admit, got {other:?}"),
            }
        }
        s
    }

    #[test]
    fn no_movement_on_noise() {
        // A2 arm 1 — hysteresis: a candidate within the 10% margin of the
        // weakest evictable resident moves nothing.
        let mut s = full_set(&[1.0, 2.0, 5.0, 9.0], 100);
        // weakest = 1.0 (id 0), margin 1.10 → threshold 1.10.
        // 1.05 < 1.10 → Reject; set unchanged.
        let out = s.consider(99, 1.05, 200);
        assert_eq!(out, AdmissionOutcome::Rejected);
        let mut ids = Vec::new();
        s.resident_ids_into(&mut ids);
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn admits_exactly_at_margin_boundary_plus_epsilon() {
        // Threshold is victim*margin; strictly above admits, exactly-at
        // rejects (the `>` contract — a tie is noise, not a shift).
        let mut s = full_set(&[1.0, 5.0], 100);
        assert_eq!(s.consider(99, 1.10, 200), AdmissionOutcome::Rejected);
        assert_eq!(
            s.consider(99, 1.11, 200),
            AdmissionOutcome::Admitted { displaced: Some(0) }
        );
    }

    #[test]
    fn burst_on_real_shift_turns_the_set_over() {
        // A2 arm 2 — a real distribution shift cascades: each strong
        // candidate displaces the (new) weakest, and the set converges to
        // the shift in K admissions — margin does not lock the old set in.
        let mut s = full_set(&[1.0, 2.0, 3.0, 4.0], 100);
        // Four shift candidates, each beating everything resident.
        for want in [10.0, 11.0, 12.0, 13.0] {
            match s.consider(50 + want as u64, want, 200) {
                AdmissionOutcome::Admitted { displaced: Some(_) } => {}
                other => panic!("shift candidate must admit, got {other:?}"),
            }
        }
        let mut ids = Vec::new();
        s.resident_ids_into(&mut ids);
        assert_eq!(ids, vec![60, 61, 62, 63]);
    }

    #[test]
    fn newborn_immunity_blocks_displacement() {
        // A2 arm 3 — dwell: a full set of newborns (age < dwell) rejects
        // even a huge candidate; the weakest-by-want newborn is NOT
        // evictable. At exactly admission+dwell the set becomes movable.
        let mut s = full_set(&[1.0, 2.0, 3.0], 100); // admitted at 100
        assert_eq!(s.consider(99, 1e9, 107), AdmissionOutcome::Rejected); // age 7 < 8
        assert_eq!(
            s.consider(99, 1e9, 108),                          // age 8 — evictable now
            AdmissionOutcome::Admitted { displaced: Some(0) }  // weakest want = 1.0
        );
    }

    #[test]
    fn weakest_among_evictable_not_weakest_overall() {
        // The victim is the weakest EVICTABLE resident: an old weak
        // resident is displaced even when a stronger NEWBORN carries a
        // larger want — age gates the candidate pool before want ranks it.
        let mut s = AdmissionSet::new(2, PoolAdmissionConfig::with_dwell(8));
        s.consider(0, 0.1, 100); // old, weak — admitted at 100
        s.consider(1, 5.0, 105); // newborn, strong — admitted at 105
        // At tick 110: id 0 evictable (age 10), id 1 newborn (age 5).
        // Candidate want 1.0 loses to id 1's 5.0 on margin (1.0 < 5.5), and
        // id 1 is immune anyway; id 0 (0.1 × 1.10 = 0.11) is the weakest
        // EVICTABLE → admit, evict 0. A weakest-OVERALL ranking that
        // ignored evictability would have to reject (id 1 undefeatable).
        assert_eq!(
            s.consider(42, 1.0, 110),
            AdmissionOutcome::Admitted { displaced: Some(0) }
        );
        // And the direct newborn case: the weaker NEWBORN is never the
        // victim while an older stronger resident is evictable.
        let mut s2 = AdmissionSet::new(2, PoolAdmissionConfig::with_dwell(8));
        s2.consider(0, 5.0, 100); // old, strong
        s2.consider(1, 0.1, 105); // newborn, WEAK
        assert_eq!(
            s2.consider(42, 100.0, 110), // both considered; victim must be id 0 (only evictable)
            AdmissionOutcome::Admitted { displaced: Some(0) }
        );
    }

    #[test]
    fn since_admission_not_last_use_is_the_dwell_clock() {
        // A2 arm 4 — THE clock law. A resident touched (want refreshed)
        // EVERY cycle from admission through dwell must become evictable at
        // exactly admission_tick + dwell. A last-use clock would read the
        // touch and keep it immune forever — this arm reds under that bug.
        let mut s = AdmissionSet::new(1, PoolAdmissionConfig::with_dwell(8));
        s.consider(7, 1.0, 1000);
        for t in 1001..1008 {
            s.update_want(7, 1.0 + (t % 3) as f32 * 0.001); // touched every cycle
            assert_eq!(
                s.consider(8, 1e6, t),
                AdmissionOutcome::Rejected,
                "immune at tick {t} (age {} < 8)",
                t - 1000
            );
        }
        s.update_want(7, 1.002); // still touched
        assert_eq!(
            s.consider(8, 1e6, 1008),
            AdmissionOutcome::Admitted { displaced: Some(7) },
            "evictable at exactly admission+despite constant touching"
        );
    }

    #[test]
    fn fair_turn_sweep_terminates_and_rearms_only_on_growth() {
        // A2 arm 5 — the sweep: one designation per item ever, index order,
        // terminates for good, re-arms only via note_universe growth.
        let mut s = AdmissionSet::new(4, PoolAdmissionConfig::with_dwell(8));
        s.note_universe(3);
        assert_eq!(s.fair_turn_pending(), 3);
        assert_eq!(s.fair_turn_next(), Some(0));
        assert_eq!(s.fair_turn_next(), Some(1));
        assert_eq!(s.fair_turn_next(), Some(2));
        // Terminated for good — repeated calls return None.
        assert_eq!(s.fair_turn_next(), None);
        assert_eq!(s.fair_turn_next(), None);
        assert_eq!(s.fair_turn_pending(), 0);
        // Shrinking is ignored; no re-arm.
        s.note_universe(1);
        assert_eq!(s.fair_turn_next(), None);
        // Growth re-arms — ONLY the new ids.
        s.note_universe(5);
        assert_eq!(s.fair_turn_pending(), 2);
        assert_eq!(s.fair_turn_next(), Some(3));
        assert_eq!(s.fair_turn_next(), Some(4));
        assert_eq!(s.fair_turn_next(), None);
    }

    #[test]
    fn fair_turn_resumes_from_cursor_in_index_order() {
        // The sweep wraps from the cursor: after designating the last index,
        // the next sweep continues at 0 (index order is per-sweep, the
        // cursor just keeps the O(1) typical case).
        let mut s = AdmissionSet::new(4, PoolAdmissionConfig::with_dwell(8));
        s.note_universe(4);
        assert_eq!(s.fair_turn_next(), Some(0));
        assert_eq!(s.fair_turn_next(), Some(1));
        assert_eq!(s.fair_turn_next(), Some(2));
        s.note_universe(6); // growth mid-sweep
        assert_eq!(s.fair_turn_next(), Some(3));
        assert_eq!(s.fair_turn_next(), Some(4));
        assert_eq!(s.fair_turn_next(), Some(5));
        assert_eq!(s.fair_turn_next(), None);
    }

    #[test]
    fn fair_admission_waives_margin_but_honors_dwell() {
        // The fair turn exists BECAUSE a never-resident item cannot clear a
        // margin on observed usage — the margin is waived, dwell is not.
        let mut s = full_set(&[100.0, 200.0, 300.0], 100); // all evictable at 108+
        assert_eq!(
            s.consider_fair(99, 0.0001, 200),
            AdmissionOutcome::Admitted { displaced: Some(0) }, // weakest evictable = 100.0
            "fair admission displaces the weakest evictable regardless of margin"
        );
        // Full newborn set still rejects: dwell outranks the fair turn.
        let mut s2 = full_set(&[1.0, 2.0], 100);
        assert_eq!(s2.consider_fair(99, 1e9, 107), AdmissionOutcome::Rejected);
    }

    #[test]
    fn identity_apply_no_double_load() {
        // A2 arm 6 — apply by identity: re-considering a resident refreshes
        // its want and moves nothing; the readout is exactly the live ids.
        let mut s = full_set(&[1.0, 2.0, 3.0], 100);
        assert_eq!(s.consider(1, 99.0, 200), AdmissionOutcome::AlreadyResident);
        assert_eq!(s.row_for(1).unwrap().want, 99.0);
        let mut ids = Vec::new();
        s.resident_ids_into(&mut ids);
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(s.len(), 3);
        assert!(s.contains(2));
        assert!(!s.contains(9));
    }

    #[test]
    fn free_slots_fill_without_displacement() {
        let mut s = AdmissionSet::new(3, PoolAdmissionConfig::with_dwell(8));
        assert_eq!(
            s.consider(0, 0.0, 1),
            AdmissionOutcome::Admitted { displaced: None }
        );
        assert_eq!(
            s.consider(1, 0.0, 1),
            AdmissionOutcome::Admitted { displaced: None }
        );
        assert_eq!(s.len(), 2);
        // A zero-want candidate fills the LAST free slot unconditionally —
        // filling an empty set is not the margin policy's question.
        assert_eq!(
            s.consider(2, 0.0, 1),
            AdmissionOutcome::Admitted { displaced: None }
        );
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn zero_want_is_a_valid_signal() {
        // Want-scores are caller-supplied and may legitimately be 0 (no
        // observed usage yet). 0 is finite: it admits into free slots, and
        // on a full set a 0 candidate only displaces a 0-want victim when
        // margin×0 = 0 < anything — i.e. only a strictly positive candidate
        // ever displaces a 0-want resident. The hysteresis stays total.
        let mut s = full_set(&[0.0, 5.0], 100);
        assert_eq!(s.consider(9, 0.0, 200), AdmissionOutcome::Rejected); // 0 > 0*1.10 is false
        assert_eq!(
            s.consider(9, 0.01, 200),
            AdmissionOutcome::Admitted { displaced: Some(0) }
        );
    }

    #[test]
    fn non_finite_never_poisons() {
        // The kv_eviction::observe posture: non-finite inputs are rejected
        // debug_assert!-loud in debug (a violation is a caller bug) and
        // behaviorally dropped in every profile. The intentional debug
        // fires are caught so the arm runs in both profiles.
        let mut s = full_set(&[1.0, 2.0], 100);
        let nan_candidate = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.consider(9, f32::NAN, 200)
        }));
        if cfg!(debug_assertions) {
            assert!(
                nan_candidate.is_err(),
                "NaN candidate must debug_assert in debug"
            );
        } else {
            assert_eq!(nan_candidate.unwrap(), AdmissionOutcome::Rejected);
        }
        let inf_refresh = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.update_want(0, f32::INFINITY)
        }));
        if !cfg!(debug_assertions) {
            assert!(inf_refresh.is_ok());
            assert_eq!(
                s.row_for(0).unwrap().want,
                1.0,
                "inf refresh dropped, want intact"
            );
        }
        let _ = inf_refresh;
    }

    #[test]
    fn weakest_evictable_tie_breaks_lowest_index() {
        let wants = [3.0, 1.0, 1.0, 2.0];
        let rows: Vec<ResidentRow> = wants
            .iter()
            .enumerate()
            .map(|(i, &w)| ResidentRow {
                id: i as u64,
                want: w,
                admission_tick: 0,
            })
            .collect();
        assert_eq!(weakest_evictable(&rows, 4, 100, 8), Some(1));
    }

    #[test]
    fn pure_core_matches_stateful() {
        // The free function and the convenience layer implement ONE policy.
        let mut s = full_set(&[1.0, 4.0, 2.0], 100);
        let mut next_id = 5000u64;
        for (want, tick) in [(2.0, 200u64), (10.0, 200), (0.5, 200), (2.2, 200)] {
            let rows_snapshot = s.rows.clone();
            let pure = admission_decision(
                &rows_snapshot,
                s.len(),
                want,
                tick,
                &PoolAdmissionConfig::with_dwell(8),
            );
            next_id += 1; // unique id, never resident — no identity shortcut
            let stateful = s.consider(next_id, want, tick);
            match (pure, stateful) {
                (
                    AdmissionDecision::Admit { .. },
                    AdmissionOutcome::Admitted { displaced: Some(_) },
                ) => {}
                (AdmissionDecision::Reject, AdmissionOutcome::Rejected) => {}
                other => panic!("pure {other:?}: core and set disagree at want={want}"),
            }
        }
    }

    /// G4 — every steady-state path is zero-alloc (counters via the crate's
    /// own test TrackingAllocator, the convergence_cadence pattern; counters
    /// are debug-only by design, so the test gates to match).
    #[cfg(debug_assertions)]
    #[test]
    fn g4_alloc_free_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        const K: usize = 32;
        let mut s = AdmissionSet::new(K, PoolAdmissionConfig::with_dwell(8));
        // Cold-path setup OUTSIDE the measurement: fill the set, grow the
        // universe (had_turn allocation is a growth event, never per-cycle).
        for i in 0..K as u64 {
            let _ = s.consider(i, 1.0 + (i % 7) as f32 * 0.1, 1000);
        }
        s.note_universe(4096);
        let mut ids_buf = Vec::with_capacity(K);
        reset_alloc_stats();
        for cycle in 0..1000u64 {
            let t = 2000 + cycle;
            // want refreshes (the every-cycle touch)
            for i in 0..K as u64 {
                s.update_want(i, 1.0 + ((i + cycle) % 11) as f32 * 0.05);
            }
            // one fair-turn designation + fair admission per cycle
            if let Some(idx) = s.fair_turn_next() {
                let _ = s.consider_fair(idx, 0.5 + (cycle % 5) as f32 * 0.1, t);
            }
            // ordinary consideration (mostly rejects — hysteresis)
            let cand = 5000 + cycle;
            let _ = s.consider(cand, 1.2 + (cycle % 9) as f32 * 0.03, t);
            // the identity readout into the caller-owned buffer
            s.resident_ids_into(&mut ids_buf);
            let _ = std::hint::black_box(&ids_buf);
        }
        let (count, _bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "steady-state pool_admission must be zero-alloc, saw {count} allocs"
        );
    }
}
