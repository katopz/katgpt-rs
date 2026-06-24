//! The `Rubric` trait + verdict / predicate types.
//!
//! A rubric evaluates a trajectory prefix and returns a fixed-size array of
//! predicate results. The arity `N` is a const generic on both the trait and
//! the verdict, so the whole structure is stack-allocated and zero-allocation
//! on the hot path.
//!
//! See [`crate::compaction`] for the module-level doc and paper citation.

/// A predicate verdict for one position in the rubric.
///
/// `Yes` carries the trajectory span `[quote_start, quote_start+quote_len]`
/// that grounded the decision — the audit-trail obligation from the paper's
/// verbatim-quote requirement. We preserve the *obligation* (cite the span
/// that grounded the decision) while computing the predicate from latent
/// features (per Research 300 §2.4). This is what keeps the gate auditable
/// without leaking latent embeddings across the sync boundary.
///
/// `No` carries a reason so the caller / audit log can explain *why* the
/// rubric declined to fire. Cheap debugging surface; never affects the
/// fire-rule Boolean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredicateResult {
    /// Predicate fired. `quote_start`/`quote_len` cite the trajectory span
    /// where the latent feature crossed threshold (paper's verbatim quote,
    /// latent-reframed per Research 300 §2.4).
    Yes { quote_start: u32, quote_len: u16 },
    /// Predicate did not fire, with a short reason.
    No { reason: PredicateReason },
}

impl PredicateResult {
    /// Returns `true` iff this is [`PredicateResult::Yes`].
    #[inline]
    #[must_use]
    pub const fn is_yes(&self) -> bool {
        matches!(self, Self::Yes { .. })
    }

    /// Returns `true` iff this is [`PredicateResult::No`].
    #[inline]
    #[must_use]
    pub const fn is_no(&self) -> bool {
        matches!(self, Self::No { .. })
    }
}

impl Default for PredicateResult {
    /// Default is `No { reason: Unset }` — a rubric that has not been
    /// evaluated yet. This makes `[PredicateResult; N]` default-constructible
    /// for scratch initialization without `unsafe`.
    #[inline]
    fn default() -> Self {
        Self::No {
            reason: PredicateReason::Unset,
        }
    }
}

/// Why a predicate returned `No`. Compact (`u8`-discriminant-backed) so
/// the audit record stays small and deterministic.
///
/// The variants cover the four paper predicates plus a generic `Unset` for
/// default-constructed scratch. `Custom` lets a rubric surface a domain
/// reason without bloating the enum; the byte payload is opaque to the
/// fire-rule logic (which only reads `Yes`/`No`).
///
/// Note: `Custom(u8)` carries a payload, so the enum is *not* field-less and
/// cannot be cast with `as u8`. Use [`PredicateReason::discriminant_byte`]
/// to read the discriminant, or [`PredicateReason::from_byte`] to reconstruct
/// (the `Custom` payload is lost in the round-trip — only the discriminant
/// survives the sync boundary, which is the contract).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PredicateReason {
    /// Scratch slot never written — default before a rubric evaluation.
    #[default]
    Unset = 0,
    /// C1-style: trajectory is not yet a closed unit (open sub-goal).
    NotClosedUnit = 1,
    /// C2-style: intrinsic rank too high — not yet summarizable.
    TooHighRank = 2,
    /// C3-style: no positive divergence since last summary — no progress.
    NoProgress = 3,
    /// N1-style: novelty rate above threshold — agent is *not* stuck, so the
    /// fire rule's negation does not hold.
    StillNovel = 4,
    /// Generic / domain-specific reason. The byte payload is opaque to the
    /// fire rule. Note: the payload does NOT survive a `discriminant_byte` →
    /// `from_byte` round-trip — only the discriminant crosses the sync
    /// boundary, by design.
    Custom(u8) = 5,
}

impl PredicateReason {
    /// Returns the discriminant byte (without any `Custom` payload). This is
    /// what gets written into the [`super::audit::PredicateAudit`] POD and
    /// crosses the sync boundary.
    #[inline]
    #[must_use]
    pub const fn discriminant_byte(self) -> u8 {
        match self {
            Self::Unset => 0,
            Self::NotClosedUnit => 1,
            Self::TooHighRank => 2,
            Self::NoProgress => 3,
            Self::StillNovel => 4,
            // For Custom, only the discriminant survives; the payload is
            // opaque to the fire rule and to the sync contract.
            Self::Custom(_) => 5,
        }
    }

    /// Reconstruct a reason from its discriminant byte. `Custom` is
    /// reconstructed with a zero payload (the original payload is lost across
    /// the sync boundary — only the discriminant is meaningful there).
    #[inline]
    #[must_use]
    pub const fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::Unset,
            1 => Self::NotClosedUnit,
            2 => Self::TooHighRank,
            3 => Self::NoProgress,
            4 => Self::StillNovel,
            // 5 (Custom) and any unknown byte → Custom(0). Unknown bytes are
            // a forward-compat path: a future rubric adding a variant will
            // not break old audit readers (they see it as a generic Custom).
            _ => Self::Custom(0),
        }
    }
}

/// Fixed-size verdict returned by [`Rubric::evaluate`].
///
/// `N` is the rubric arity (e.g. `4` for the paper's search rubric
/// C1/C2/C3/N1, `3` for the math rubric Q1/Q2/Q3, `2` for the shard-freeze
/// rubric P0/P1). Stored as a stack array — zero heap allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RubricVerdict<const N: usize> {
    /// Per-predicate results, in the rubric's canonical order.
    pub predicates: [PredicateResult; N],
}

impl<const N: usize> RubricVerdict<N> {
    /// Construct a verdict from its predicate array.
    #[inline]
    #[must_use]
    pub const fn new(predicates: [PredicateResult; N]) -> Self {
        Self { predicates }
    }

    /// Construct an all-`No` verdict (every predicate defaulted to
    /// `No { reason: Unset }`). Useful for unit tests and as a baseline.
    #[inline]
    #[must_use]
    pub fn all_no() -> Self {
        Self {
            predicates: [PredicateResult::No {
                reason: PredicateReason::Unset,
            }; N],
        }
    }

    /// Returns `true` iff predicate `i` is [`PredicateResult::Yes`].
    ///
    /// Debug-asserts `i < N`; out-of-bounds is a programmer bug, not a
    /// runtime condition.
    #[inline]
    #[must_use]
    pub fn is_yes(&self, i: usize) -> bool {
        debug_assert!(i < N, "predicate index {i} out of bounds (arity {N})");
        self.predicates[i].is_yes()
    }

    /// Pack the `Yes`/`No` pattern into a bitmask, bit `i` set iff predicate
    /// `i` is `Yes`. This is the representation the [`crate::compaction::FireRule`]
    /// combinators operate on.
    ///
    /// `N` is capped at 8 by the `FireRule` bitmask type (`u8`); debug-asserts
    /// this.
    #[inline]
    #[must_use]
    pub fn yes_mask(&self) -> u8 {
        debug_assert!(
            N <= 8,
            "RubricVerdict::yes_mask requires N <= 8 (FireRule bitmask width), got {N}"
        );
        let mut mask = 0u8;
        let mut bit = 1u8;
        for p in &self.predicates {
            if p.is_yes() {
                mask |= bit;
            }
            bit <<= 1;
        }
        mask
    }
}

impl<const N: usize> Default for RubricVerdict<N> {
    #[inline]
    fn default() -> Self {
        Self::all_no()
    }
}

/// Reusable scratch buffer passed into [`Rubric::evaluate`].
///
/// The rubric implementation may use this for any temporary computation
/// (e.g. a divergence accumulation, a novelty-rate EMA). The contract is:
/// **the scratch is caller-owned and reused across calls**, so the hot path
/// performs zero heap allocation. A rubric that needs owned storage should
/// pre-size it in `Default` / `new` and `clear()` + reuse on each call.
///
/// The default scratch is empty — a rubric that needs scratch state defines
/// its own scratch type and passes `&mut` to a custom evaluate entry point.
/// This base type is the minimal hook for the generic trait.
#[derive(Clone, Debug, Default)]
pub struct RubricScratch {
    /// Generic f32 scratch (e.g. for divergence / coherence accumulators).
    pub f32_buf: Vec<f32>,
    /// Generic usize scratch (e.g. for span bookkeeping).
    pub usize_buf: Vec<usize>,
}

impl RubricScratch {
    /// Construct an empty scratch.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a scratch with pre-allocated capacity for `f` floats and
    /// `u` usizes. Use this once at setup so the hot path never allocates.
    #[inline]
    #[must_use]
    pub fn with_capacity(f: usize, u: usize) -> Self {
        Self {
            f32_buf: Vec::with_capacity(f),
            usize_buf: Vec::with_capacity(u),
        }
    }

    /// Clear both buffers without freeing capacity. Call at the start of each
    /// `evaluate` to reuse the storage.
    #[inline]
    pub fn clear(&mut self) {
        self.f32_buf.clear();
        self.usize_buf.clear();
    }
}

/// The rubric trait — evaluates a trajectory prefix into a fixed-size
/// verdict of `N` predicate results.
///
/// Generic over the arity `N` (const generic) so the verdict is a stack
/// array and the fire-rule combinators can pack the `Yes`/`No` pattern into
/// a `u8` bitmask (requires `N <= 8`).
///
/// The trajectory is passed as `&[u8]` — the gate is agnostic to whether
/// those bytes are UTF-8 tokens, latent-feature scalars (reinterpreted), or
/// raw event bytes. Each rubric implementation documents its expected input
/// encoding. The latent-reframed rubrics (Research 300 §2.4) compute their
/// predicates from scalar features the caller supplies, not from literal
/// byte inspection.
///
/// **Zero-allocation contract**: implementations MUST NOT allocate inside
/// `evaluate`. Use the provided [`RubricScratch`] for any temporary storage.
pub trait Rubric<const N: usize> {
    /// Evaluate the rubric against `trajectory_prefix` (the `y_{1:t}` prefix
    /// up to the probe point), writing per-predicate results into the
    /// returned verdict.
    ///
    /// `scratch` is caller-owned and reused across calls — do not allocate
    /// inside this method; clear and reuse `scratch` instead.
    fn evaluate(&self, trajectory_prefix: &[u8], scratch: &mut RubricScratch) -> RubricVerdict<N>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_result_is_yes_no_helpers() {
        let yes = PredicateResult::Yes {
            quote_start: 10,
            quote_len: 3,
        };
        let no = PredicateResult::No {
            reason: PredicateReason::NotClosedUnit,
        };
        assert!(yes.is_yes());
        assert!(!yes.is_no());
        assert!(no.is_no());
        assert!(!no.is_yes());
    }

    #[test]
    fn predicate_result_default_is_no_unset() {
        let d = PredicateResult::default();
        assert_eq!(
            d,
            PredicateResult::No {
                reason: PredicateReason::Unset
            }
        );
        assert!(d.is_no());
    }

    #[test]
    fn verdict_yes_mask_packs_bits_lsb_first() {
        // N = 4: predicates [Yes, No, Yes, No] → mask 0b0101 = 0x05.
        let v = RubricVerdict::<4>::new([
            PredicateResult::Yes {
                quote_start: 0,
                quote_len: 1,
            },
            PredicateResult::No {
                reason: PredicateReason::TooHighRank,
            },
            PredicateResult::Yes {
                quote_start: 2,
                quote_len: 1,
            },
            PredicateResult::No {
                reason: PredicateReason::StillNovel,
            },
        ]);
        assert_eq!(v.yes_mask(), 0b0101);
        assert!(v.is_yes(0));
        assert!(!v.is_yes(1));
        assert!(v.is_yes(2));
        assert!(!v.is_yes(3));
    }

    #[test]
    fn verdict_all_no_has_zero_mask() {
        let v = RubricVerdict::<3>::all_no();
        assert_eq!(v.yes_mask(), 0);
        for i in 0..3 {
            assert!(!v.is_yes(i));
        }
    }

    #[test]
    fn verdict_default_equals_all_no() {
        let v = RubricVerdict::<2>::default();
        assert_eq!(v, RubricVerdict::<2>::all_no());
    }

    #[test]
    fn verdict_yes_mask_full_arity_8() {
        // All 8 predicates Yes → mask 0xFF.
        let v = RubricVerdict::<8>::new(
            [PredicateResult::Yes {
                quote_start: 0,
                quote_len: 1,
            }; 8],
        );
        assert_eq!(v.yes_mask(), 0xFF);
    }

    #[test]
    fn scratch_clear_keeps_capacity() {
        let mut s = RubricScratch::with_capacity(16, 8);
        s.f32_buf.extend_from_slice(&[1.0; 16]);
        s.usize_buf.extend(0..8);
        assert_eq!(s.f32_buf.len(), 16);
        assert_eq!(s.usize_buf.len(), 8);
        s.clear();
        assert!(s.f32_buf.is_empty());
        assert!(s.usize_buf.is_empty());
        // Capacity preserved — hot path reuse invariant.
        assert!(s.f32_buf.capacity() >= 16);
        assert!(s.usize_buf.capacity() >= 8);
    }

    /// Minimal rubric for trait exercise: arity 2, Yes iff the trajectory
    /// contains byte `b'A'` (predicate 0) or `b'B'` (predicate 1).
    struct DemoRubric;
    impl Rubric<2> for DemoRubric {
        fn evaluate(&self, traj: &[u8], _scratch: &mut RubricScratch) -> RubricVerdict<2> {
            let p0 = if let Some(i) = traj.iter().position(|&b| b == b'A') {
                PredicateResult::Yes {
                    quote_start: i as u32,
                    quote_len: 1,
                }
            } else {
                PredicateResult::No {
                    reason: PredicateReason::Custom(0),
                }
            };
            let p1 = if let Some(i) = traj.iter().position(|&b| b == b'B') {
                PredicateResult::Yes {
                    quote_start: i as u32,
                    quote_len: 1,
                }
            } else {
                PredicateResult::No {
                    reason: PredicateReason::Custom(1),
                }
            };
            RubricVerdict::new([p0, p1])
        }
    }

    #[test]
    fn rubric_trait_object_can_be_invoked() {
        let r = DemoRubric;
        let mut s = RubricScratch::new();
        let v = r.evaluate(b"xxAyB", &mut s);
        assert!(v.is_yes(0));
        assert!(v.is_yes(1));
        assert_eq!(v.yes_mask(), 0b11);

        let v2 = r.evaluate(b"no match here", &mut s);
        assert!(!v2.is_yes(0));
        assert!(!v2.is_yes(1));
        assert_eq!(v2.yes_mask(), 0);
    }
}
