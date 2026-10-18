//! LtHash — incremental homomorphic multiset hash (lattice hash).
//!
//! Issue 807, mined from the Agave validator snapshot (riir-clippy Research
//! 172; the eprint 2019/227 / Facebook LtHash instantiation Agave ships as
//! `lattice-hash/src/lt_hash.rs`). Consumers (proposal-gated): riir-chain
//! Proposal 010 D1 — `CpiAccountStore::commitment_root` becomes an O(1) read
//! (executing the fix `dispatch.rs` + Bench 028 #16 already specify); riir-
//! dapps Proposal 005 D1 — the `kat:statehash:v1` per-batch tamper-evidence
//! trail.
//!
//! # Construction
//!
//! The state is `[u16; N]` limbs (default `N = 1024`, the battle-tested
//! instantiation). Each multiset member is lifted to a group element via a
//! domain-separated BLAKE3 XOF; the aggregate is the **componentwise modular
//! sum** (wrapping add mod 2¹⁶):
//!
//! - `insert(e)` = add e's limbs,
//! - `remove(e)` = subtract e's limbs,
//! - `merge(other)` = sum — the parallel-fold combiner,
//! - `checksum()` = one BLAKE3 over the limb bytes.
//!
//! Because the group is commutative, the aggregate is **order-independent by
//! construction** — deterministic under any insertion order or thread
//! schedule (the property a sorted-fold Merkle has to work for and this gets
//! for free). Insert + remove make the state **incremental**: an update is
//! `remove(element(key, old)); insert(element(key, new))`, O(1) in the number
//! of accounts — independent of how many members the multiset holds.
//!
//! # Element derivation (ambiguity-free)
//!
//! [`Element::derive`] hashes a **length-prefixed part list** under a
//! caller-supplied domain (BLAKE3 `new_derive_key`): each part is fed as
//! `(u32-LE length ‖ bytes)`, so `("ab", "c")` and `("a", "bc")` derive to
//! different elements. Callers MUST bind the member's identity (its key)
//! into the element — not just its value — or two members with equal values
//! would collide as multiset duplicates. The canonical two-part form is
//! `derive(domain, &[key, value])`.
//!
//! # Security posture (read before choosing N)
//!
//! Mod-2¹⁶ wrapping limbs follow eprint 2019/227 / Agave (the original
//! Bellare–Micciancio MSet-Add-Hash is mod-prime). Collision resistance
//! degrades with lane count: `N = 1024` is the default and the conservative
//! choice; smaller `N` (e.g. 128) is an explicit consumer trade of state
//! width against security margin — record the decision at the consumer.
//! This is an **audit / drift primitive**: it provides tamper-evidence and
//! O(Δ) checkpoints; it does NOT provide per-member inclusion proofs —
//! that remains Merkle territory (see riir-dapps Proposal 005 D1: the
//! LtHash trail augments, never replaces, the Merkle anchor).
//!
//! # Determinism + perf contract
//!
//! Pure integer arithmetic over fixed arrays: zero allocations in every op
//! path, no `unsafe`, no platform-dependent float behavior, wasm32-clean.
//! The from-scratch vs incremental equivalence (the drift-gate property both
//! consumers pin) is exact — bit-identical limbs — because both paths are
//! the same modular sums evaluated in different orders.
//!
//! lthash — incremental homomorphic multiset hash (Issue 807): LtHash
//! [u16; N] (default 1024 lanes, wrapping add mod 2^16) with insert=add,
//! remove=subtract, merge=sum, checksum=BLAKE3(state). Order-independent
//! aggregate by construction (commutative monoid) → deterministic under any
//! thread schedule. Element = domain-separated BLAKE3 XOF over a
//! length-prefixed part list (no concatenation-ambiguity collisions). Mined
//! from the Agave validator snapshot (eprint 2019/227 instantiation) for
//! riir-chain Proposal 010 D1 (commitment_root → O(1)) + riir-dapps
//! Proposal 005 D1 (kat:statehash trail). Pure integer arithmetic, zero
//! deps, zero allocs, wasm32-clean. OPT-IN pending the Bench 771 GOAT +
//! first consumer wiring.

use blake3::Hasher;

/// Default lane count — the eprint 2019/227 / Agave instantiation (2 KB state).
pub const DEFAULT_LANES: usize = 1024;

/// A lifted group element: the BLAKE3-XOF expansion of one multiset member.
///
/// Deliberately NOT `Copy` — at the default width this is 2 KB; accidental
/// copies would be silent perf cliffs. `Clone` is explicit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Element<const N: usize = DEFAULT_LANES> {
    limbs: [u16; N],
}

impl<const N: usize> Element<N> {
    /// Derives the element for one multiset member from a length-prefixed
    /// part list under a domain-separation string.
    ///
    /// The canonical member encoding is `derive(domain, &[key, value])` —
    /// the member's identity and its value both bound into the element.
    pub fn derive(domain: &str, parts: &[&[u8]]) -> Self {
        let mut hasher = Hasher::new_derive_key(domain);
        for part in parts {
            hasher.update(&(part.len() as u32).to_le_bytes());
            hasher.update(part);
        }
        let mut limbs = [0u16; N];
        // Stable const generics cannot size an array as `2 * N`; fill through a
        // fixed stack chunk instead (XOF output advances across `fill` calls) —
        // zero alloc for any N.
        let mut reader = hasher.finalize_xof();
        let mut chunk = [0u8; 1024]; // 512 limbs per fill
        let mut filled = 0;
        while filled < N {
            let take = (N - filled).min(chunk.len() / 2);
            reader.fill(&mut chunk[..2 * take]);
            for (j, limb) in limbs[filled..filled + take].iter_mut().enumerate() {
                *limb = u16::from_le_bytes([chunk[2 * j], chunk[2 * j + 1]]);
            }
            filled += take;
        }
        Self { limbs }
    }

    /// The element's limbs (inspector for tests + custom folds).
    pub fn as_limbs(&self) -> &[u16; N] {
        &self.limbs
    }
}

/// The multiset accumulator: componentwise modular sum of member elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LtHash<const N: usize = DEFAULT_LANES> {
    limbs: [u16; N],
}

impl<const N: usize> Default for LtHash<N> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const N: usize> LtHash<N> {
    /// The additive identity — the hash of the empty multiset.
    pub const fn identity() -> Self {
        Self { limbs: [0u16; N] }
    }

    /// True when the state equals the identity (empty multiset).
    pub fn is_identity(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    /// Inserts a member: `state += element` (O(N) lane adds, no alloc).
    pub fn insert(&mut self, element: &Element<N>) {
        for (s, e) in self.limbs.iter_mut().zip(element.limbs.iter()) {
            *s = s.wrapping_add(*e);
        }
    }

    /// Removes a member: `state -= element`. Removing a member that is not
    /// in the multiset is the group inverse of inserting it — well-defined
    /// arithmetic; guarding membership is the caller's drift gate.
    pub fn remove(&mut self, element: &Element<N>) {
        for (s, e) in self.limbs.iter_mut().zip(element.limbs.iter()) {
            *s = s.wrapping_sub(*e);
        }
    }

    /// Replaces a member's contribution in one pass: the riir-dapps
    /// `state − element(key, old) + element(key, new)` update shape.
    pub fn replace(&mut self, old: &Element<N>, new: &Element<N>) {
        for ((s, o), n) in self
            .limbs
            .iter_mut()
            .zip(old.limbs.iter())
            .zip(new.limbs.iter())
        {
            *s = s.wrapping_sub(*o).wrapping_add(*n);
        }
    }

    /// Merges another accumulator into this one (`self += other`) — the
    /// commutative combiner for deterministic parallel folds: chunk results
    /// combined in any order yield the identical aggregate.
    pub fn merge(&mut self, other: &Self) {
        for (s, o) in self.limbs.iter_mut().zip(other.limbs.iter()) {
            *s = s.wrapping_add(*o);
        }
    }

    /// The 32-byte state tag: one BLAKE3 over the limb bytes (LE).
    pub fn checksum(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        for limb in &self.limbs {
            hasher.update(&limb.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    /// The raw limb state (persist/restore without re-derivation).
    pub fn limbs(&self) -> &[u16; N] {
        &self.limbs
    }

    /// Restores state from limbs (the persist counterpart of [`Self::limbs`]).
    pub fn from_limbs(limbs: &[u16; N]) -> Self {
        Self { limbs: *limbs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const DOMAIN: &str = "katgpt-lthash-test-v1";

    fn elem(key: &[u8], value: &[u8]) -> Element {
        Element::derive(DOMAIN, &[key, value])
    }

    /// G1: the aggregate is invariant under insertion order — three seeded
    /// permutations of the same member multiset agree on limbs + checksum.
    #[test]
    fn order_invariance() {
        let members: Vec<Element> = (0u16..64)
            .map(|i| elem(&i.to_le_bytes(), &[i as u8; 8]))
            .collect();
        let mut perms: Vec<Vec<usize>> = Vec::new();
        for seed in [1u64, 42, 1337] {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut idx: Vec<usize> = (0..members.len()).collect();
            rng.shuffle(&mut idx);
            perms.push(idx);
        }
        let mut checksums = Vec::with_capacity(perms.len());
        for perm in &perms {
            let mut h: LtHash = LtHash::identity();
            for &i in perm {
                h.insert(&members[i]);
            }
            checksums.push(h.checksum());
        }
        assert_eq!(checksums[0], checksums[1]);
        assert_eq!(checksums[1], checksums[2]);
    }

    /// G1: insert-then-remove returns exactly to identity; `replace` equals
    /// remove+insert.
    #[test]
    fn remove_and_replace_roundtrip() {
        let a = elem(b"k1", b"v1");
        let b = elem(b"k1", b"v2");
        let mut h: LtHash = LtHash::identity();
        h.insert(&a);
        assert!(!h.is_identity());
        h.remove(&a);
        assert!(h.is_identity());

        h.insert(&a);
        let mut via_pair = h.clone();
        via_pair.remove(&a);
        via_pair.insert(&b);
        h.replace(&a, &b);
        assert_eq!(h, via_pair);

        h.remove(&b);
        assert!(h.is_identity());
    }

    /// G1: genuine multiset semantics — duplicates count ({A,A,B} ≠
    /// {A,B,B}; {A,A} minus one A ≠ the empty multiset).
    #[test]
    fn multiset_semantics() {
        let a = elem(b"a", b"va");
        let b = elem(b"b", b"vb");
        let mut aab: LtHash = LtHash::identity();
        aab.insert(&a);
        aab.insert(&a);
        aab.insert(&b);
        let mut abb: LtHash = LtHash::identity();
        abb.insert(&a);
        abb.insert(&b);
        abb.insert(&b);
        assert_ne!(aab, abb);
        assert_ne!(aab.checksum(), abb.checksum());

        let mut aa: LtHash = LtHash::identity();
        aa.insert(&a);
        aa.insert(&a);
        aa.remove(&a);
        let single: LtHash = {
            let mut h: LtHash = LtHash::identity();
            h.insert(&a);
            h
        };
        assert_eq!(aa, single);
    }

    /// G1: merge ≡ incremental insertion — the deterministic-parallel-fold
    /// property. Chunk boundaries and combination order must not matter.
    #[test]
    fn merge_equals_incremental() {
        let members: Vec<Element> = (0u16..100)
            .map(|i| elem(&i.to_le_bytes(), &[i as u8; 4]))
            .collect();
        let mut sequential: LtHash = LtHash::identity();
        for m in &members {
            sequential.insert(m);
        }
        // Three chunkings, folded in different orders.
        let chunked = |chunk_size: usize, reverse_fold: bool| {
            let mut chunks: Vec<LtHash> = members
                .chunks(chunk_size)
                .map(|c| {
                    let mut h: LtHash = LtHash::identity();
                    for m in c {
                        h.insert(m);
                    }
                    h
                })
                .collect();
            if reverse_fold {
                chunks.reverse();
            }
            let mut acc: LtHash = LtHash::identity();
            for c in &chunks {
                acc.merge(c);
            }
            acc
        };
        assert_eq!(sequential, chunked(7, false));
        assert_eq!(sequential, chunked(10, true));
        assert_eq!(sequential, chunked(1, false));
    }

    /// G1: the length-prefixed part encoding is unambiguous — part-boundary
    /// shifts change the element.
    #[test]
    fn encoding_is_unambiguous() {
        let ab_c: Element = Element::derive(DOMAIN, &[b"ab", b"c"]);
        let a_bc: Element = Element::derive(DOMAIN, &[b"a", b"bc"]);
        assert_ne!(ab_c, a_bc);
        // And an actual (key,value) collision across boundaries is impossible.
        assert_ne!(elem(b"ab", b"c"), elem(b"a", b"bc"));
    }

    /// G1: domain separation — the same parts under different domains
    /// derive to different elements.
    #[test]
    fn domain_separation() {
        let d1: Element = Element::derive("domain-one", &[b"k", b"v"]);
        let d2: Element = Element::derive("domain-two", &[b"k", b"v"]);
        assert_ne!(d1, d2);
    }

    /// G1 drift property: interleaved incremental updates (insert / remove /
    /// replace) land bit-identically on the from-scratch rebuild of the
    /// final multiset. This is the property both consumers' drift gates pin.
    #[test]
    fn drift_gate_incremental_equals_rebuild() {
        let mut rng = fastrand::Rng::with_seed(20260916);
        let mut model: BTreeMap<u16, u8> = BTreeMap::new(); // key -> value
        let mut incremental: LtHash = LtHash::identity();
        for step in 0..500 {
            let key = rng.u16(0..32);
            let value = rng.u8(1..=255); // 0 = absent sentinel in the model
            match step % 3 {
                0 => {
                    // insert-or-replace
                    let new = elem(&key.to_le_bytes(), &[value]);
                    match model.get(&key).copied() {
                        Some(old_v) if old_v != 0 => {
                            incremental.replace(&elem(&key.to_le_bytes(), &[old_v]), &new);
                        }
                        _ => incremental.insert(&new),
                    }
                    model.insert(key, value);
                }
                1 => {
                    // remove if present
                    if let Some(old_v) = model.get(&key).copied()
                        && old_v != 0
                    {
                        incremental.remove(&elem(&key.to_le_bytes(), &[old_v]));
                        model.remove(&key);
                    }
                }
                _ => {
                    // pure insert of a fresh key value pair (duplicate values allowed)
                    let new = elem(&key.to_le_bytes(), &[value]);
                    match model.get(&key).copied() {
                        Some(old_v) if old_v != 0 => {
                            incremental.replace(&elem(&key.to_le_bytes(), &[old_v]), &new);
                        }
                        _ => incremental.insert(&new),
                    }
                    model.insert(key, value);
                }
            }
        }
        let mut rebuild: LtHash = LtHash::identity();
        for (k, v) in &model {
            rebuild.insert(&elem(&k.to_le_bytes(), &[*v]));
        }
        assert_eq!(
            incremental, rebuild,
            "incremental state drifted from rebuild"
        );
        assert_eq!(incremental.checksum(), rebuild.checksum());
    }

    /// G1: narrower lane widths compile and hold the same algebra (the
    /// consumer-side width trade, e.g. riir-chain's 128-lane sketch).
    #[test]
    fn narrow_lane_width_holds() {
        let a = Element::<128>::derive(DOMAIN, &[b"key", b"value"]);
        let b = Element::<128>::derive(DOMAIN, &[b"key", b"value2"]);
        let mut h = LtHash::<128>::identity();
        h.insert(&a);
        h.replace(&a, &b);
        h.remove(&b);
        assert!(h.is_identity());
    }

    /// G1: limbs persist/restore round-trip (the riir-dapps durable-row shape).
    #[test]
    fn limbs_roundtrip() {
        let mut h: LtHash = LtHash::identity();
        h.insert(&elem(b"k", b"v"));
        let restored = LtHash::from_limbs(h.limbs());
        assert_eq!(h, restored);
        assert_eq!(h.checksum(), restored.checksum());
    }

    /// Known-answer vector: pins the construction (BLAKE3 derive_key XOF →
    /// LE u16 limbs → wrapping sum → BLAKE3 checksum) against drift. If this
    /// test fails after an edit, the WIRE/AUDIT VALUE of every persisted
    /// state hash changes — do not update it casually.
    #[test]
    fn known_answer_vector() {
        let mut h: LtHash = LtHash::identity();
        let e1: Element = Element::derive("katgpt-lthash-kat-v1", &[b"account-1", b"1000"]);
        h.insert(&e1);
        let e2: Element = Element::derive("katgpt-lthash-kat-v1", &[b"account-2", b"2049"]);
        h.insert(&e2);
        let checksum = h.checksum();
        let hex: String = checksum.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex, "dc6a921442ede14db66c2c56a80999deaacc6627b37e57f1596ee395975c4963",
            "construction changed — every persisted LtHash value changes with it"
        );
    }
}
