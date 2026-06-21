//! [`FaithfulnessProbe`] trait + [`DefaultFaithfulnessProbe`] implementation.
//!
//! The probe runs the causal intervention suite from Research 244 §4 / Plan 278:
//! for each [`Intervention`](super::types::Intervention), it perturbs a clone of
//! the memory, queries the consumer's behavior, and measures the delta from
//! baseline (no-memory). The aggregated deltas form a [`FaithfulnessProfile`]
//! whose `is_faithfully_used(threshold)` gives the verdict.
//!
//! # Plan 298 — Smear-aware audit (Phase 2)
//!
//! When the `smear_classifier` feature is on AND the probe has been wired with
//! an optional [`SmearClassifier`](super::smear::SmearClassifier) via
//! [`DefaultFaithfulnessProbe::with_smear_classifier`], the new
//! [`DefaultFaithfulnessProbe::probe_intervention_full`] method additionally
//! classifies the consumer's current latent-mass distribution and emits a
//! [`SmearReport`](super::smear::SmearReport) alongside the binary `Delta`.
//!
//! **The report is a diagnostic** — it does NOT add a sync dependency, does NOT
//! emit a chain commit, and does NOT change the existing
//! [`TriggeredInjectionGate`](super::gate::TriggeredInjectionGate) decision
//! (Plan 278, default-on). The gate remains the source of truth for
//! inject/skip; the report only enriches the audit stream for downstream
//! consumers (e.g. riir-ai Cognitive Integrity Layer, see `.research/129`).
//!
//! Backwards compatibility: when no classifier is attached, or when the
//! consumer does not expose a [`SmearSource`],
//! [`DefaultFaithfulnessProbe::probe_intervention_full`] returns
//! `InterventionOutcome { smear: None, .. }` — i.e. zero behavior change.

use fastrand::Rng;

use super::perturb;
use super::types::{ConsumerContext, FaithfulnessProfile, Intervention, MemorySlice};

// SmearClassifier integration is gated by the `smear_classifier` feature, which
// itself depends on `faithfulness_probe`. When the feature is off, *all*
// smear-aware surface area disappears from `DefaultFaithfulnessProbe` so the
// zero-overhead-off property (Plan 278 G8) is preserved.
#[cfg(feature = "smear_classifier")]
use super::smear::{SmearClassifier, SmearReport};

/// Optional surface that lets a [`ConsumerContext`] expose its current
/// latent-mass distribution for smear classification (Plan 298 Phase 2).
///
/// Implement this on consumers that carry a multi-hypothesis superposition —
/// typically:
/// - MUX superposition generators (Plan 178) — the K parallel token streams.
/// - BoM K-hypothesis samplers (Plan 281) — the K belief states.
///
/// **Do NOT implement this on consumers without superposition** (e.g. plain
/// autoregressive decoders, deterministic retrieval). The probe falls back to
/// `smear: None` when the source is absent, which is the correct behavior for
/// single-hypothesis consumers (they are always `CoherentSingle` by
/// construction; classification would be a no-op).
///
/// The returned slice MUST be `[k * d]` row-major (k rows of d elements).
/// `k` is capped at 16 by the [`SmearClassifier`] trait contract; callers that
/// carry more hypotheses must subsample (e.g. top-k by norm).
#[cfg(feature = "smear_classifier")]
pub trait SmearSource {
    /// Returns `(weights_slice, k, d)` for the current latent-mass distribution.
    /// The slice is borrowed for the duration of the call; implementations
    /// should expose a snapshot of the live state without cloning when possible.
    fn latent_mass_distribution(&self) -> (&[f32], usize, usize);
}

/// Outcome of a smear-aware causal intervention (Plan 298 Phase 2).
///
/// Wraps the existing binary `Delta` from [`FaithfulnessProbe::probe_intervention`]
/// with an optional [`SmearReport`] describing how the consumer's latent mass
/// was distributed at the audit site.
///
/// `smear` is `None` when:
/// - the probe has no `SmearClassifier` attached (default), OR
/// - the consumer does not implement `SmearSource`.
///
/// In both cases the binary verdict is unchanged — `smear` is purely additive
/// diagnostic signal.
#[cfg(feature = "smear_classifier")]
#[derive(Debug, Clone, Copy)]
pub struct InterventionOutcome<D> {
    /// Existing binary behavioral delta from the causal intervention.
    pub delta: D,
    /// Optional smear classification of the consumer's latent mass at audit
    /// time. `None` when no classifier is attached or the consumer does not
    /// expose a `SmearSource`.
    pub smear: Option<SmearReport>,
}

/// Causal-intervention faithfulness probe for injected memory segments.
///
/// Verifies that a consumer's behavior is causally bound to an injected
/// memory segment. Modelless: zero training, zero backprop through base weights.
///
/// Based on Zhao et al. 2026 (arxiv 2601.22436), Research 244 §4.
pub trait FaithfulnessProbe {
    /// Memory representation under audit.
    type Memory;
    /// Behavior representation produced by the consumer.
    type Behavior;
    /// Delta metric (must support ordering, be `Copy`).
    type Delta: PartialOrd + Copy + Default;

    /// Run a single causal intervention and report the behavioral delta
    /// between baseline (no memory) and behavior with the perturbed memory.
    fn probe_intervention(
        &mut self,
        memory: &Self::Memory,
        intervention: Intervention,
        rng: &mut Rng,
    ) -> Self::Delta;

    /// Run the full intervention suite {Empty, Shuffle, Corrupt, Irrelevant,
    /// Filler} and aggregate into a [`FaithfulnessProfile`].
    fn faithfulness_profile(
        &mut self,
        memory: &Self::Memory,
        rng: &mut Rng,
    ) -> FaithfulnessProfile<Self::Delta>;
}

/// Default probe that runs the full intervention suite using the
/// [`perturb`] strategies on a slice view of memory.
///
/// Generic over `C: ConsumerContext`. Requires `C::Memory: MemorySlice + Clone`
/// so the probe can clone-and-perturb.
///
/// # Fields
///
/// - `consumer` — the behavioral surface under audit.
/// - `irrelevant_pool` — source of unrelated content for the `Irrelevant` intervention.
/// - `filler_elem` — placeholder for the `Filler` intervention.
/// - `smear` *(only when `smear_classifier` feature is on)* — optional
///   [`SmearClassifier`] for ternary latent-mass classification. `None` by
///   default → zero behavior change. Attach via
///   [`with_smear_classifier`](Self::with_smear_classifier).
///
/// Runs at **audit cadence** (every N ticks), not per-tick. May allocate
/// (clones memory per intervention). See Plan 278 ADR-2.
pub struct DefaultFaithfulnessProbe<C>
where
    C: ConsumerContext,
    C::Memory: MemorySlice,
{
    pub consumer: C,
    pub irrelevant_pool: Vec<<C::Memory as MemorySlice>::Elem>,
    pub filler_elem: <C::Memory as MemorySlice>::Elem,
    /// Optional smear classifier (Plan 298 Phase 2). `None` by default — the
    /// probe behaves exactly like Plan 278's binary probe unless the caller
    /// explicitly attaches a classifier via `with_smear_classifier`.
    #[cfg(feature = "smear_classifier")]
    pub smear: Option<Box<dyn SmearClassifier>>,
}

impl<C> DefaultFaithfulnessProbe<C>
where
    C: ConsumerContext,
    C::Memory: MemorySlice + Clone,
{
    /// Create a probe with the given consumer, irrelevant pool, and filler element.
    pub fn new(
        consumer: C,
        irrelevant_pool: Vec<<C::Memory as MemorySlice>::Elem>,
        filler_elem: <C::Memory as MemorySlice>::Elem,
    ) -> Self {
        Self {
            consumer,
            irrelevant_pool,
            filler_elem,
            #[cfg(feature = "smear_classifier")]
            smear: None,
        }
    }

    /// Attach a [`SmearClassifier`] to this probe (Plan 298 Phase 2).
    ///
    /// After this call, [`probe_intervention_full`](Self::probe_intervention_full)
    /// will additionally classify the consumer's latent mass (when the
    /// consumer implements [`SmearSource`]) and emit a [`SmearReport`] in the
    /// returned [`InterventionOutcome`].
    ///
    /// The existing [`probe_intervention`](FaithfulnessProbe::probe_intervention)
    /// method is **unaffected** — it continues to return only the binary `Delta`.
    /// The classifier is a diagnostic; it does NOT change the inject/skip
    /// decision of [`TriggeredInjectionGate`](super::gate::TriggeredInjectionGate).
    ///
    /// Only available when the `smear_classifier` feature is on (which implies
    /// `faithfulness_probe`).
    #[cfg(feature = "smear_classifier")]
    pub fn with_smear_classifier(mut self, classifier: Box<dyn SmearClassifier>) -> Self {
        self.smear = Some(classifier);
        self
    }
}

impl<C> FaithfulnessProbe for DefaultFaithfulnessProbe<C>
where
    C: ConsumerContext,
    C::Memory: MemorySlice + Clone,
{
    type Memory = C::Memory;
    type Behavior = C::Behavior;
    type Delta = C::Delta;

    fn probe_intervention(
        &mut self,
        memory: &Self::Memory,
        intervention: Intervention,
        rng: &mut Rng,
    ) -> Self::Delta {
        let baseline = self.consumer.baseline_behavior();

        // Clone + perturb in-place. The mutable borrow from mem_as_mut_slice
        // ends before we hand &perturbed to behavior_with_memory (NLL).
        let mut perturbed = memory.clone();
        {
            let slice = perturbed.mem_as_mut_slice();
            match intervention {
                Intervention::Empty => perturb::perturb_empty(slice),
                Intervention::Shuffle => perturb::perturb_shuffle(slice, rng),
                Intervention::Corrupt => perturb::perturb_corrupt(slice, rng),
                Intervention::Irrelevant => {
                    perturb::perturb_irrelevant(slice, rng, &self.irrelevant_pool)
                }
                Intervention::Filler => perturb::perturb_filler(slice, &self.filler_elem),
            }
        }

        let behavior = self.consumer.behavior_with_memory(&perturbed);
        self.consumer.behavior_delta(&baseline, &behavior)
    }

    fn faithfulness_profile(
        &mut self,
        memory: &Self::Memory,
        rng: &mut Rng,
    ) -> FaithfulnessProfile<Self::Delta> {
        // Each intervention clones from the ORIGINAL memory — perturbations
        // do not compound. 5 clones total (audit cadence, acceptable).
        let empty_delta = self.probe_intervention(memory, Intervention::Empty, rng);
        let shuffle_delta = self.probe_intervention(memory, Intervention::Shuffle, rng);
        let corrupt_delta = self.probe_intervention(memory, Intervention::Corrupt, rng);
        let irrelevant_delta = self.probe_intervention(memory, Intervention::Irrelevant, rng);
        let filler_delta = self.probe_intervention(memory, Intervention::Filler, rng);

        // Aggregate shuffle + corrupt into the max (whichever disrupted more).
        let shuffle_or_corrupt_delta = if shuffle_delta >= corrupt_delta {
            shuffle_delta
        } else {
            corrupt_delta
        };

        FaithfulnessProfile {
            empty_delta,
            shuffle_or_corrupt_delta,
            irrelevant_delta,
            filler_delta,
        }
    }
}

// ---------------------------------------------------------------------------
// Plan 298 Phase 2 — smear-aware audit surface (feature-gated).
// Only available when `smear_classifier` is on. When off, *all* of these
// symbols vanish from the public API, preserving Plan 278 G8 (zero-overhead
// when off). The existing `probe_intervention` / `faithfulness_profile`
// methods above are unchanged and remain available regardless of feature.
// ---------------------------------------------------------------------------

#[cfg(feature = "smear_classifier")]
impl<C> DefaultFaithfulnessProbe<C>
where
    C: ConsumerContext,
    C::Memory: MemorySlice + Clone,
{
    /// Run a causal intervention AND, when a [`SmearClassifier`] is attached
    /// and the consumer exposes a [`SmearSource`], classify the consumer's
    /// current latent-mass distribution. Returns an [`InterventionOutcome`]
    /// wrapping both the binary delta and the optional [`SmearReport`].
    ///
    /// # Behavior
    ///
    /// - If `self.smear` is `None` → returns `InterventionOutcome { delta, smear: None }`.
    ///   Equivalent to [`probe_intervention`](FaithfulnessProbe::probe_intervention)
    ///   but boxed in the outcome struct.
    /// - If `self.smear` is `Some(_)` but the consumer does NOT implement
    ///   [`SmearSource`] (caller passed `source: None`) → still returns
    ///   `smear: None`. Plain-autoregressive consumers fall here, which is the
    ///   correct behavior (they are always `CoherentSingle` by construction).
    /// - If both are present → classifies `source.latent_mass_distribution()`
    ///   and emits the [`SmearReport`] alongside the delta.
    ///
    /// # Diagnostics only
    ///
    /// The report does NOT change the inject/skip decision of
    /// [`TriggeredInjectionGate`](super::gate::TriggeredInjectionGate). It is
    /// surfaced for downstream consumers (e.g. riir-ai Cognitive Integrity
    /// Layer, see `.research/129`) to react differently to benign positional
    /// uncertainty vs potentially-unfaithful multi-hypothesis superposition.
    ///
    /// # Arguments
    ///
    /// - `memory` — injected memory segment under audit.
    /// - `intervention` — which perturbation to apply.
    /// - `rng` — RNG for stochastic perturbations.
    /// - `source` — optional `&dyn SmearSource` view of the consumer. Pass
    ///   `Some(&self.consumer as &dyn SmearSource)` when `C: SmearSource`, or
    ///   `None` to skip classification for this call.
    /// - `scratch` — caller-allocated scratch buffer for the classifier.
    ///   Length MUST be `>= k + k*(k-1)/2` for the worst-case `k` the consumer
    ///   exposes. Reused across calls; zero-allocation in the audit hot path.
    pub fn probe_intervention_full(
        &mut self,
        memory: &C::Memory,
        intervention: Intervention,
        rng: &mut Rng,
        source: Option<&dyn SmearSource>,
        scratch: &mut [f32],
    ) -> InterventionOutcome<C::Delta> {
        // Compute the binary delta exactly as `probe_intervention` does.
        let delta = self.probe_intervention(memory, intervention, rng);

        // Smear classification is gated by BOTH the classifier being attached
        // AND the caller supplying a `SmearSource` view for this audit.
        // Either missing → `None` (no diagnostic).
        let smear = match (&self.smear, source) {
            (Some(classifier), Some(src)) => {
                let (weights, k, d) = src.latent_mass_distribution();
                Some(classifier.classify(weights, k, d, scratch))
            }
            _ => None,
        };

        InterventionOutcome { delta, smear }
    }

    /// Smear-aware variant of [`faithfulness_profile`](FaithfulnessProbe::faithfulness_profile).
    ///
    /// Runs the full intervention suite as `faithfulness_profile` does, but
    /// additionally classifies the consumer's latent mass once per audit call
    /// (not per intervention — the distribution is the consumer's state, not
    /// the memory's). The single [`SmearReport`] is attached to the returned
    /// [`FaithfulnessProfileFull`].
    ///
    /// # Arguments
    ///
    /// Same as [`probe_intervention_full`](Self::probe_intervention_full),
    /// minus the per-call `Intervention`.
    pub fn faithfulness_profile_full(
        &mut self,
        memory: &C::Memory,
        rng: &mut Rng,
        source: Option<&dyn SmearSource>,
        scratch: &mut [f32],
    ) -> FaithfulnessProfileFull<C::Delta> {
        let empty_delta = self.probe_intervention(memory, Intervention::Empty, rng);
        let shuffle_delta = self.probe_intervention(memory, Intervention::Shuffle, rng);
        let corrupt_delta = self.probe_intervention(memory, Intervention::Corrupt, rng);
        let irrelevant_delta = self.probe_intervention(memory, Intervention::Irrelevant, rng);
        let filler_delta = self.probe_intervention(memory, Intervention::Filler, rng);

        let shuffle_or_corrupt_delta = if shuffle_delta >= corrupt_delta {
            shuffle_delta
        } else {
            corrupt_delta
        };

        // One classification per audit — the distribution is the consumer's
        // state, not the memory's. See probe.rs docs.
        let smear = match (&self.smear, source) {
            (Some(classifier), Some(src)) => {
                let (weights, k, d) = src.latent_mass_distribution();
                Some(classifier.classify(weights, k, d, scratch))
            }
            _ => None,
        };

        FaithfulnessProfileFull {
            profile: FaithfulnessProfile {
                empty_delta,
                shuffle_or_corrupt_delta,
                irrelevant_delta,
                filler_delta,
            },
            smear,
        }
    }
}

/// Audit-cadence outcome bundling the binary [`FaithfulnessProfile`] with an
/// optional smear classification (Plan 298 Phase 2).
///
/// `smear` is `None` when no classifier is attached or the consumer does not
/// expose a [`SmearSource`]. See
/// [`DefaultFaithfulnessProbe::faithfulness_profile_full`].
#[cfg(feature = "smear_classifier")]
#[derive(Debug, Clone, Copy)]
pub struct FaithfulnessProfileFull<D> {
    /// Existing binary faithfulness profile (Plan 278).
    pub profile: FaithfulnessProfile<D>,
    /// Optional smear classification of the consumer's latent mass at audit
    /// time. `None` when no classifier is attached or the consumer does not
    /// expose a `SmearSource`.
    pub smear: Option<SmearReport>,
}

// ---------------------------------------------------------------------------
// Unit tests — Plan 278 T1.8 (G1 + G1b synthetic consumers)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faithfulness::types::ConsumerContext;

    /// Faithful consumer: behavior = weighted dot product with position-dependent
    /// weights. Empty memory → behavior 0 (= baseline). Meaningful perturbations
    /// → non-zero behavior.
    struct FaithfulConsumer {
        weights: Vec<f32>,
    }

    impl ConsumerContext for FaithfulConsumer {
        type Behavior = f32;
        type Delta = f32;
        type Memory = Vec<f32>;

        fn baseline_behavior(&self) -> f32 {
            0.0
        }

        fn behavior_with_memory(&self, memory: &Vec<f32>) -> f32 {
            memory
                .iter()
                .zip(self.weights.iter())
                .map(|(&v, &w)| v * w)
                .sum()
        }

        fn behavior_delta(&self, a: &f32, b: &f32) -> f32 {
            (a - b).abs()
        }
    }

    /// Unfaithful consumer: ignores memory entirely, always returns a constant.
    struct UnfaithfulConsumer;

    impl ConsumerContext for UnfaithfulConsumer {
        type Behavior = f32;
        type Delta = f32;
        type Memory = Vec<f32>;

        fn baseline_behavior(&self) -> f32 {
            42.0
        }

        fn behavior_with_memory(&self, _memory: &Vec<f32>) -> f32 {
            42.0 // ignores memory
        }

        fn behavior_delta(&self, a: &f32, b: &f32) -> f32 {
            (a - b).abs()
        }
    }

    #[test]
    fn test_faithful_consumer_detected() {
        // Position-dependent weights so shuffle matters.
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let consumer = FaithfulConsumer { weights };
        // Non-zero pool so irrelevant produces non-zero behavior.
        let irrelevant_pool = vec![0.1_f32, 0.2, 0.3, 0.4];
        // Non-zero filler so filler behavior = sum(weights) != 0.
        let filler = 1.0_f32;

        let mut probe = DefaultFaithfulnessProbe::new(consumer, irrelevant_pool, filler);

        // Distinct values so shuffle/corrupt change the dot product.
        let memory = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut rng = Rng::with_seed(42);

        let profile = probe.faithfulness_profile(&memory, &mut rng);

        // empty_delta should be ~0 (zeros -> behavior 0 = baseline).
        assert!(
            profile.empty_delta < 0.001,
            "empty_delta should be ~0, got {}",
            profile.empty_delta
        );

        let threshold = 0.5;
        assert!(
            profile.is_faithfully_used(threshold),
            "faithful consumer should be detected as faithfully used: {:?}",
            profile
        );
    }

    #[test]
    fn test_unfaithful_consumer_detected() {
        let consumer = UnfaithfulConsumer;
        let irrelevant_pool = vec![1.0_f32, 2.0, 3.0];
        let filler = 1.0_f32;

        let mut probe = DefaultFaithfulnessProbe::new(consumer, irrelevant_pool, filler);

        let memory = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut rng = Rng::with_seed(42);

        let profile = probe.faithfulness_profile(&memory, &mut rng);

        // All deltas should be 0 (behavior always 42.0 = baseline).
        assert_eq!(profile.empty_delta, 0.0);
        assert_eq!(profile.shuffle_or_corrupt_delta, 0.0);
        assert_eq!(profile.irrelevant_delta, 0.0);
        assert_eq!(profile.filler_delta, 0.0);

        let threshold = 0.5;
        assert!(
            !profile.is_faithfully_used(threshold),
            "unfaithful consumer should NOT be detected as faithfully used: {:?}",
            profile
        );
    }
}
