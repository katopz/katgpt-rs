//! InfoRegistry — ARG Step 9 (Info) + Step C (Collection) dedup primitive.
//!
//! Distilled from ARG §2.9 (Info IO) + §3.4 (Validation). The registry is the
//! **dedup layer** that prevents the same information unit from being committed
//! twice under different `InfoKey`s, and flags near-duplicates for human review.
//!
//! ## Two-phase dedup
//!
//! 1. **Primary index** (`by_key`): `HashMap<InfoKey, InfoUnit>`. Exact match
//!    on `InfoKey` (signature + type + scope). A hit here is a `StrongMatch`
//!    (same semantic address) or a `GreyZone` (same key, different payload —
//!    possible corruption / version drift).
//! 2. **Secondary index** (`by_payload`): `HashMap<PayloadHash, Vec<InfoKey>>`.
//!    Detects cross-key content collisions (different semantic address,
//!    identical payload — possible cross-scope duplicate). A hit here without a
//!    primary hit is a `GreyZone`.
//!
//! A miss on both is `NoMatch` — genuinely new information.
//!
//! ## CompareFn extension point
//!
//! The default comparison is payload-hash equality. The `CompareFn` trait is the
//! Phase 2 slot for pluggable semantic / vector comparison (bounded recall on
//! Top-K). It is NOT used in the v1 hot path — the two indexes above cover the
//! ARG-mandated exact + collision cases without learned similarity.
//!
//! ## Concurrency
//!
//! Both indexes are `papaya::HashMap` — lock-free reads, fine-grained writes.
//! The registry is safe to share across threads (`Send + Sync` via papaya).

use super::scorer::InfoOutcomeStatus;
use papaya::HashMap;

/// Controlled info category (caller-defined enum projected to `u8`).
pub type InfoType = u8;

/// Tenant / workspace isolation boundary. Units with different scopes never
/// collide — cross-scope dedup is a separate, opt-in operation.
pub type AccessScope = u64;

/// BLAKE3 hash of `L_final_ids` (the validated label set). 32 bytes, fixed.
pub type LabelSignature = [u8; 32];

/// BLAKE3 hash of the info payload (the grounded response content). 32 bytes.
pub type PayloadHash = [u8; 32];

/// The composite address of an info unit — `(signature, info_type, scope)`.
///
/// Two units with the same `InfoKey` occupy the same semantic address; whether
/// they are *the same* information depends on payload comparison. `Ord` is
/// lexicographic on `(signature, info_type, scope)` so registry iteration is
/// deterministic (replay-safe).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InfoKey {
    pub signature: LabelSignature,
    pub info_type: InfoType,
    pub scope: AccessScope,
}

impl InfoKey {
    /// Construct a key from its parts.
    #[inline]
    pub const fn new(signature: LabelSignature, info_type: InfoType, scope: AccessScope) -> Self {
        InfoKey {
            signature,
            info_type,
            scope,
        }
    }
}

// Manual Ord: lexicographic on (signature bytes, info_type, scope). Arrays
// already derive lexicographic Ord, so we delegate — but spell it out for
// clarity and to guarantee the ordering is byte-deterministic (replay-safe
// across nodes).
impl PartialOrd for InfoKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InfoKey {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.signature
            .cmp(&other.signature)
            .then(self.info_type.cmp(&other.info_type))
            .then(self.scope.cmp(&other.scope))
    }
}

/// Provenance — where the info unit came from. Minimal for v1 (the caller can
/// encode richer provenance in `source_id` + `flags`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Provenance {
    /// Source identifier (e.g. shard id, agent id, external feed id).
    pub source_id: u32,
    /// Caller-defined provenance flags (e.g. 0=internal, 1=external, 2=synthetic).
    pub flags: u8,
}

/// A single info unit — the registry's value type. `Copy` (88 bytes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InfoUnit {
    pub key: InfoKey,
    pub payload_hash: PayloadHash,
    /// Groundedness / confidence in `[0, 1]` (ARG §2.9 `C_info`).
    pub c_info: f32,
    /// Outcome tag — reused from the scorer for consistency.
    pub outcome: InfoOutcomeStatus,
    pub provenance: Provenance,
    /// Monotonic timestamp (caller-defined epoch; e.g. chain tick or unix ms).
    pub ts: u64,
}

impl InfoUnit {
    /// Construct a unit with minimal fields (provenance defaulted).
    #[inline]
    pub fn new(key: InfoKey, payload_hash: PayloadHash, c_info: f32, outcome: InfoOutcomeStatus, ts: u64) -> Self {
        InfoUnit {
            key,
            payload_hash,
            c_info,
            outcome,
            provenance: Provenance::default(),
            ts,
        }
    }
}

/// Grey-zone comparison result (ARG §3.4 Validation). The `CompareFn` trait
/// returns this; the registry uses it to decide Strong vs Grey vs No.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareResult {
    /// Definitely the same information — collapse to canonical.
    Same,
    /// Definitely different — keep separate.
    Different,
    /// Unsure — flag for human review (GreyZone).
    Unsure,
}

/// Extension point for pluggable semantic / vector comparison (Phase 2 bounded
/// recall slot). The default impl compares payload hashes only.
pub trait CompareFn: Send + Sync {
    fn compare(&self, a: &InfoUnit, b: &InfoUnit) -> CompareResult;
}

/// Default comparator — payload-hash equality. `Same` iff payload hashes match.
#[derive(Clone, Copy, Debug, Default)]
pub struct PayloadHashCompare;

impl CompareFn for PayloadHashCompare {
    #[inline]
    fn compare(&self, a: &InfoUnit, b: &InfoUnit) -> CompareResult {
        if a.payload_hash == b.payload_hash {
            CompareResult::Same
        } else {
            CompareResult::Different
        }
    }
}

/// Result of [`InfoRegistry::canonicalize`].
#[derive(Clone, Debug)]
pub enum MatchResult {
    /// Exact match — `unit` is the same information as an existing canonical.
    /// The wrapped unit is the canonical (lowest-ts) representative.
    StrongMatch(InfoUnit),
    /// Possible duplicate — same key with different payload, OR different key
    /// with same payload. Needs human / CompareFn review before commit.
    /// The Vec holds the candidate(s); the caller decides.
    GreyZone(Vec<InfoUnit>),
    /// Genuinely new — no key match and no payload collision.
    NoMatch,
}

impl MatchResult {
    /// Returns `true` if this is a [`MatchResult::StrongMatch`].
    #[inline]
    pub fn is_strong(&self) -> bool {
        matches!(self, MatchResult::StrongMatch(_))
    }

    /// Returns `true` if this is a [`MatchResult::GreyZone`].
    #[inline]
    pub fn is_grey(&self) -> bool {
        matches!(self, MatchResult::GreyZone(_))
    }

    /// Returns `true` if this is a [`MatchResult::NoMatch`].
    #[inline]
    pub fn is_none(&self) -> bool {
        matches!(self, MatchResult::NoMatch)
    }
}

/// Scratch buffer for zero-alloc canonicalize on the common (no-collision) path.
/// Pre-allocate once and reuse across calls. The GreyZone path still allocates
/// a `Vec` (it is an offline-only, rare path).
#[derive(Clone, Debug, Default)]
pub struct MatchScratch {
    /// Reusable buffer for collecting grey-zone candidates.
    pub grey: Vec<InfoUnit>,
}

impl MatchScratch {
    /// Construct with a hint capacity.
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        MatchScratch {
            grey: Vec::with_capacity(n),
        }
    }

    /// Clear without freeing the backing buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.grey.clear();
    }
}

/// The two-index dedup registry. Lock-free reads via papaya.
#[derive(Debug, Default)]
pub struct InfoRegistry {
    /// Primary: `InfoKey -> canonical InfoUnit`.
    by_key: HashMap<InfoKey, InfoUnit>,
    /// Secondary: `PayloadHash -> keys sharing that payload`. For collision detect.
    by_payload: HashMap<PayloadHash, Vec<InfoKey>>,
}

impl InfoRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        InfoRegistry {
            by_key: HashMap::new(),
            by_payload: HashMap::new(),
        }
    }

    /// Number of canonical units (primary index size).
    pub fn len(&self) -> usize {
        self.by_key.pin().len()
    }

    /// Returns `true` if the registry holds no units.
    pub fn is_empty(&self) -> bool {
        self.by_key.pin().is_empty()
    }

    /// Insert `unit` as a canonical unit under its key. Overwrites any existing
    /// unit at the same key. Also updates the payload secondary index.
    ///
    /// This is the *commit* path — the caller has already decided (via
    /// [`canonicalize`](Self::canonicalize)) that this unit is genuinely new or
    /// is the chosen canonical after grey-zone resolution.
    pub fn insert(&self, unit: InfoUnit) {
        // Update payload index: add this key to the payload's key list (dedup).
        // papaya 0.2's update_or_insert_with takes Fn(&V) -> V (copy-on-write),
        // not Fn(&mut V). Read existing, clone+push, return new vec.
        let new_key = unit.key;
        self.by_payload.pin().update_or_insert_with(
            unit.payload_hash,
            |keys: &Vec<InfoKey>| {
                if keys.contains(&new_key) {
                    // Already present — return a clone (no-op update).
                    keys.clone()
                } else {
                    let mut next = keys.clone();
                    next.push(new_key);
                    next
                }
            },
            || vec![new_key],
        );
        self.by_key.pin().insert(unit.key, unit);
    }

    /// Look up the canonical unit for `key`, if any.
    pub fn get(&self, key: &InfoKey) -> Option<InfoUnit> {
        self.by_key.pin().get(key).copied()
    }

    /// Returns `true` if `key` has a canonical unit.
    pub fn contains_key(&self, key: &InfoKey) -> bool {
        self.by_key.pin().contains_key(key)
    }

    /// Canonicalize `unit` against the registry using the default
    /// [`PayloadHashCompare`]. See [`canonicalize_with`](Self::canonicalize_with)
    /// for the pluggable-comparator variant.
    pub fn canonicalize(&self, unit: &InfoUnit, scratch: &mut MatchScratch) -> MatchResult {
        self.canonicalize_with(unit, scratch, &PayloadHashCompare)
    }

    /// Canonicalize `unit` against the registry using a custom `CompareFn`.
    ///
    /// Decision tree:
    /// 1. **Primary hit** (`by_key` has `unit.key`):
    ///    - `compare(unit, existing) == Same` → `StrongMatch(existing)`
    ///    - `compare(unit, existing) == Different | Unsure` → `GreyZone([existing])`
    ///      (same semantic address, different content — version drift / corruption)
    /// 2. **Primary miss, secondary hit** (`by_payload` has `unit.payload_hash`):
    ///    → `GreyZone(candidates)` (cross-key content collision — possible dup)
    /// 3. **Both miss** → `NoMatch`
    ///
    /// The GreyZone Vec is built in `scratch.grey` then cloned out (the scratch
    /// is reused across calls; the clone is the one allocation on the grey path).
    pub fn canonicalize_with(
        &self,
        unit: &InfoUnit,
        scratch: &mut MatchScratch,
        cmp: &dyn CompareFn,
    ) -> MatchResult {
        scratch.clear();

        // 1. Primary index hit?
        let by_key = self.by_key.pin();
        if let Some(existing) = by_key.get(&unit.key) {
            match cmp.compare(unit, existing) {
                CompareResult::Same => {
                    // Same semantic address + same content → strong match.
                    // Return the canonical (lowest-ts) representative.
                    let canonical = if existing.ts <= unit.ts {
                        *existing
                    } else {
                        *unit
                    };
                    return MatchResult::StrongMatch(canonical);
                }
                CompareResult::Different | CompareResult::Unsure => {
                    // Same key, different content — grey zone (version drift).
                    scratch.grey.push(*existing);
                    return MatchResult::GreyZone(scratch.grey.clone());
                }
            }
        }
        drop(by_key);

        // 2. Secondary index hit (payload collision with a different key)?
        let by_payload = self.by_payload.pin();
        if let Some(keys) = by_payload.get(&unit.payload_hash) {
            // Collect the canonical units for each colliding key.
            let by_key = self.by_key.pin();
            for &k in keys.iter() {
                if k != unit.key {
                    if let Some(u) = by_key.get(&k) {
                        scratch.grey.push(*u);
                    }
                }
            }
            if !scratch.grey.is_empty() {
                return MatchResult::GreyZone(scratch.grey.clone());
            }
        }

        // 3. Both miss — genuinely new.
        MatchResult::NoMatch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(n: u8) -> LabelSignature {
        // Distinct signatures: fill with n so they differ and are easy to reason about.
        let mut s = [0u8; 32];
        s[0] = n;
        s
    }

    fn payload(n: u8) -> PayloadHash {
        let mut p = [0u8; 32];
        p[0] = n;
        p
    }

    fn key(n: u8, scope: AccessScope) -> InfoKey {
        InfoKey::new(sig(n), 0u8, scope)
    }

    fn unit(key_n: u8, payload_n: u8, ts: u64) -> InfoUnit {
        InfoUnit::new(key(key_n, 1), payload(payload_n), 0.9, InfoOutcomeStatus::InfoConfirmedSuccess, ts)
    }

    fn fresh_scratch() -> MatchScratch {
        MatchScratch::with_capacity(8)
    }

    // --------------------------------------------------------------------
    // T3.2 property tests (the spec).
    // --------------------------------------------------------------------

    #[test]
    fn same_info_key_yields_strong_match() {
        let reg = InfoRegistry::new();
        let u1 = unit(1, 100, 10);
        reg.insert(u1);
        // Same key, same payload → StrongMatch.
        let u2 = unit(1, 100, 20);
        let mut scratch = fresh_scratch();
        let result = reg.canonicalize(&u2, &mut scratch);
        assert!(result.is_strong(), "same key + same payload must be StrongMatch");
        if let MatchResult::StrongMatch(canonical) = result {
            // Canonical is the lowest-ts representative.
            assert_eq!(canonical.ts, 10);
        }
    }

    #[test]
    fn same_key_different_payload_yields_grey_zone() {
        let reg = InfoRegistry::new();
        let u1 = unit(1, 100, 10); // key=1, payload=100
        reg.insert(u1);
        // Same key, different payload → GreyZone (version drift).
        let u2 = InfoUnit::new(key(1, 1), payload(200), 0.5, InfoOutcomeStatus::InfoLowConfidence, 20);
        let mut scratch = fresh_scratch();
        let result = reg.canonicalize(&u2, &mut scratch);
        assert!(result.is_grey(), "same key + different payload must be GreyZone");
    }

    #[test]
    fn different_key_same_payload_yields_grey_zone() {
        let reg = InfoRegistry::new();
        let u1 = unit(1, 100, 10); // key=1, payload=100
        reg.insert(u1);
        // Different key, same payload → GreyZone (cross-key collision).
        let u2 = InfoUnit::new(key(2, 1), payload(100), 0.9, InfoOutcomeStatus::InfoConfirmedSuccess, 20);
        let mut scratch = fresh_scratch();
        let result = reg.canonicalize(&u2, &mut scratch);
        assert!(result.is_grey(), "different key + same payload must be GreyZone");
        if let MatchResult::GreyZone(cands) = result {
            assert!(!cands.is_empty());
            // The candidate is the existing unit (key=1).
            assert_eq!(cands[0].key, key(1, 1));
        }
    }

    #[test]
    fn different_key_different_payload_yields_no_match() {
        let reg = InfoRegistry::new();
        let u1 = unit(1, 100, 10);
        reg.insert(u1);
        // Different key, different payload → NoMatch.
        let u2 = unit(2, 200, 20);
        let mut scratch = fresh_scratch();
        let result = reg.canonicalize(&u2, &mut scratch);
        assert!(result.is_none(), "different key + different payload must be NoMatch");
    }

    #[test]
    fn empty_registry_yields_no_match() {
        let reg = InfoRegistry::new();
        let u = unit(1, 100, 10);
        let mut scratch = fresh_scratch();
        let result = reg.canonicalize(&u, &mut scratch);
        assert!(result.is_none());
        assert!(reg.is_empty());
    }

    // --------------------------------------------------------------------
    // InfoKey ordering (deterministic, replay-safe).
    // --------------------------------------------------------------------

    #[test]
    fn info_key_order_is_deterministic() {
        // Ordering is lexicographic on (signature, info_type, scope).
        let k1 = InfoKey::new(sig(1), 0, 100);
        let k2 = InfoKey::new(sig(1), 0, 200); // same sig+type, higher scope
        let k3 = InfoKey::new(sig(1), 1, 100); // same sig, higher type
        let k4 = InfoKey::new(sig(2), 0, 100); // higher sig
        assert!(k1 < k2); // scope breaks tie
        assert!(k2 < k3); // type breaks tie (both sig(1))
        assert!(k3 < k4); // sig breaks tie
        // Equal keys compare equal.
        assert_eq!(k1.cmp(&k1), std::cmp::Ordering::Equal);
    }

    #[test]
    fn info_key_ord_is_total_and_transitive() {
        let keys = [
            InfoKey::new(sig(5), 2, 3),
            InfoKey::new(sig(1), 9, 9),
            InfoKey::new(sig(1), 1, 1),
            InfoKey::new(sig(9), 0, 0),
            InfoKey::new(sig(1), 1, 2),
        ];
        let mut sorted = keys;
        sorted.sort();
        // Verify sorted ascending.
        for w in sorted.windows(2) {
            assert!(w[0] <= w[1], "not ascending: {:?} > {:?}", w[0], w[1]);
        }
        // First should be sig(1) (lowest signature byte).
        assert_eq!(sorted[0].signature, sig(1));
        // Among sig(1), lowest type first.
        assert_eq!(sorted[0].info_type, 1);
    }

    // --------------------------------------------------------------------
    // Insert / get / contains semantics.
    // --------------------------------------------------------------------

    #[test]
    fn insert_and_get_round_trip() {
        let reg = InfoRegistry::new();
        let u = unit(7, 42, 100);
        assert!(!reg.contains_key(&u.key));
        reg.insert(u);
        assert!(reg.contains_key(&u.key));
        assert_eq!(reg.len(), 1);
        let got = reg.get(&u.key).expect("must be present");
        assert_eq!(got, u);
    }

    #[test]
    fn insert_overwrites_same_key() {
        let reg = InfoRegistry::new();
        let u1 = unit(1, 100, 10);
        reg.insert(u1);
        let u2 = InfoUnit::new(key(1, 1), payload(100), 0.95, InfoOutcomeStatus::InfoConfirmedSuccess, 20);
        reg.insert(u2); // same key, same payload, higher c_info + ts
        assert_eq!(reg.len(), 1); // still one canonical
        let got = reg.get(&u1.key).expect("must be present");
        assert_eq!(got.ts, 20); // overwritten with the newer unit
    }

    #[test]
    fn multiple_distinct_keys_coexist() {
        let reg = InfoRegistry::new();
        for n in 1..=5u8 {
            reg.insert(unit(n, n, n as u64));
        }
        assert_eq!(reg.len(), 5);
        for n in 1..=5u8 {
            assert!(reg.contains_key(&key(n, 1)));
        }
    }

    // --------------------------------------------------------------------
    // CompareFn extension point.
    // --------------------------------------------------------------------

    #[test]
    fn custom_compare_fn_unsure_forces_grey_zone_on_same_key() {
        // A CompareFn that always returns Unsure → same-key hits always GreyZone.
        struct AlwaysUnsure;
        impl CompareFn for AlwaysUnsure {
            fn compare(&self, _a: &InfoUnit, _b: &InfoUnit) -> CompareResult {
                CompareResult::Unsure
            }
        }
        let reg = InfoRegistry::new();
        let u1 = unit(1, 100, 10);
        reg.insert(u1);
        let u2 = unit(1, 100, 20); // same key + payload
        let mut scratch = fresh_scratch();
        // With PayloadHashCompare this would be StrongMatch; with AlwaysUnsure it's GreyZone.
        let result = reg.canonicalize_with(&u2, &mut scratch, &AlwaysUnsure);
        assert!(result.is_grey());
    }

    #[test]
    fn payload_hash_compare_same_only_on_equal_hash() {
        let cmp = PayloadHashCompare;
        let u1 = unit(1, 100, 10);
        let u2 = unit(1, 100, 20); // same payload
        let u3 = unit(1, 200, 30); // different payload
        assert_eq!(cmp.compare(&u1, &u2), CompareResult::Same);
        assert_eq!(cmp.compare(&u1, &u3), CompareResult::Different);
    }

    // --------------------------------------------------------------------
    // Scratch reuse (no allocation growth on common path).
    // --------------------------------------------------------------------

    #[test]
    fn scratch_clear_resets_len_without_freeing_capacity() {
        let mut scratch = MatchScratch::with_capacity(16);
        scratch.grey.push(unit(1, 1, 1));
        scratch.grey.push(unit(2, 2, 2));
        assert_eq!(scratch.grey.len(), 2);
        scratch.clear();
        assert_eq!(scratch.grey.len(), 0);
        assert!(scratch.grey.capacity() >= 16); // capacity preserved
    }

    #[test]
    fn canonicalize_no_match_does_not_consume_scratch_capacity() {
        // Repeated NoMatch calls reuse the scratch without growing it.
        let reg = InfoRegistry::new();
        let mut scratch = fresh_scratch();
        let cap_before = scratch.grey.capacity();
        for n in 1..=20u8 {
            let u = unit(n, n, n as u64);
            let result = reg.canonicalize(&u, &mut scratch);
            assert!(result.is_none(), "iter {} must be NoMatch", n);
            reg.insert(u);
        }
        // Scratch is cleared each call; capacity unchanged by the no-match path.
        assert_eq!(scratch.grey.capacity(), cap_before);
        assert_eq!(scratch.grey.len(), 0);
    }

    // --------------------------------------------------------------------
    // Cross-thread safety (papaya lock-free).
    // --------------------------------------------------------------------

    #[test]
    fn registry_supports_concurrent_reads() {
        let reg = std::sync::Arc::new(InfoRegistry::new());
        reg.insert(unit(1, 100, 10));
        let reg_clone = reg.clone();
        let handle = std::thread::spawn(move || {
            let mut scratch = fresh_scratch();
            let u = unit(1, 100, 20);
            reg_clone.canonicalize(&u, &mut scratch)
        });
        let result = handle.join().unwrap();
        assert!(result.is_strong());
    }
}
