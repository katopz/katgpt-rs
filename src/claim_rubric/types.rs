//! Claim Rubric primitive data types — evidence level, claim, grade (Plan 307,
//! Research 287).
//!
//! These types are the data surface for the L1/L2/L3 evidence-ladder
//! validator. They carry claim text + metadata; the validator (in
//! [`crate::claim_rubric::validator`]) is the algorithm.

use crate::claim_rubric::validator::ClaimValidator;
use katgpt_core::traits::FeatureClass;

/// The four-level evidence ladder (R287 §2.2 + L0 floor).
///
/// `L0` is the auto-downgrade target — "no supporting evidence". The honest
/// level of a claim whose `satisfied` set does not even meet L1 requirements
/// is `L0`, and the validator flags every missing L1 item.
///
/// Ordering is `L0 < L1 < L2 < L3` so `Ord::max` / `Ord::min` produce the
/// right "highest level whose requirements are met" / "lowest cap" results.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceLevel {
    /// No supporting evidence. The validator's auto-downgrade target when a
    /// claim's `satisfied` set does not even meet L1's minimums.
    L0 = 0,
    /// **Behavioral** (R287 §2.2 row 1) — "primitive P reads/detects/projects
    /// signal D at rate/measurement M". Requires: operational definition, n +
    /// variance, ≥1 ablation, exclusions, linear-probe calibration.
    L1 = 1,
    /// **Functional** (R287 §2.2 row 2) — "signal from P induces downstream
    /// effect E consistently across variations". Inherits L1, adds:
    /// downstream-effect measurement, ≥3-variation generalization,
    /// human-grounded validation, base rate, latent-freshness check,
    /// benign-shift control.
    L2 = 2,
    /// **Causal-mechanistic** (R287 §2.2 row 3) — "intervening on w_B
    /// produces predictable change in B with specificity". Inherits L2,
    /// adds: intervention, predict-control parity, specificity,
    /// general-capability control, falsifiable competing explanation,
    /// failure cases.
    L3 = 3,
}

impl EvidenceLevel {
    /// Short label `"L0"` / `"L1"` / `"L2"` / `"L3"`.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            EvidenceLevel::L0 => "L0",
            EvidenceLevel::L1 => "L1",
            EvidenceLevel::L2 => "L2",
            EvidenceLevel::L3 => "L3",
        }
    }

    /// One-line definition (matches the R287 §2.2 row name).
    #[inline]
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            EvidenceLevel::L0 => "no supporting evidence",
            EvidenceLevel::L1 => "Behavioral — reads/detects/projects a signal",
            EvidenceLevel::L2 => "Functional — induces a downstream effect",
            EvidenceLevel::L3 => "Causal-mechanistic — causally controls behavior",
        }
    }
}

/// Identifier for a single R287 §5 checklist row.
///
/// `#[non_exhaustive]` so future checklist items can be added without
/// breaking downstream `match` arms. `#[repr(u16)]` so a `Vec<EvidenceItemId>`
/// is compact and a sorted lookup table (if ever needed) is cache-friendly.
///
/// Variants are grouped by the minimum level at which the item becomes
/// required: L1 items first (R287 §2.2 L1 row + §5 S1/S2 L1 rows), then L2
/// items, then L3 items.
#[non_exhaustive]
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceItemId {
    // ── L1 items (R287 §2.2 L1 row + §5 S1/S2/S3/S4 L1-tagged rows) ──
    /// §5 S1: 1–3 sentence operational definition of the signal in measurable
    /// terms (what counts, what threshold, what aggregation). L1-required.
    OperationalDefinition = 1,
    /// §5 S2 + §2.2: report n (independent generations), seeds, temperature;
    /// measurement rule; variance across seeds. L1-required.
    SampleSize = 2,
    /// §5 S3: ablate prompt / sampling / seed; report sensitivity. L1-required.
    Ablation = 3,
    /// §5 S1: explicit list of what the definition excludes (near-misses,
    /// lookalikes, non-target behaviors). L1-required.
    Exclusions = 4,
    /// §2.2 L1 row: for linear probes (ours), report direction-vector norm,
    /// projection calibration (reliability diagram / MAE), and whether trained
    /// on detection or prediction labels. L1-required.
    LinearProbeCalibration = 5,

    // ── L2 items (R287 §2.2 L2 row + §5 S1/S3 L2-tagged rows) ──
    /// §2.2 L2 row: downstream effect E measured in a deployment-plausible
    /// context (not just a toy stress test). L2-required.
    DownstreamEffect = 6,
    /// §2.2 L2 row: generalization across ≥3 reasonable variations (paraphrase,
    /// seed, model variant, sampling temperature). L2-required.
    Generalization3Variations = 7,
    /// §2.2 L2 row: if E concerns a human-facing metric, human-grounded
    /// validation (not LLM-judge-only); report inter-rater agreement. L2-required.
    HumanGroundedValidation = 8,
    /// §2.2 L2 row: base rate of E reported explicitly (especially for rare
    /// behaviors — Jones et al. 2025 extreme-value estimation). L2-required.
    BaseRateReported = 9,
    /// §2.2 L2 row (domain-specific): if P reads a latent state, show the
    /// latent was within its freshness window at decision time, or downgrade.
    /// Encodes R287 §6 anti-pattern #2 (latent-staleness confound).
    /// L2-required.
    LatentFreshnessCheck = 10,
    /// §2.2 L2 row: ≥1 benign-shift / OOD control — is E specific to the
    /// intended mechanism vs generic capability degradation. L2-required.
    BenignShiftControl = 11,

    // ── L3 items (R287 §2.2 L3 row + §5 S4 L3-tagged rows) ──
    /// §2.2 L3 row + §5 S4: intervention (ablate / zero / clamp / steer) along
    /// w_B produces a pre-registered-direction change in target B. L3-required.
    Intervention = 12,
    /// §2.2 L3 row + §5 S4 (paper C9, Research 267): **predict-control parity**
    /// — prediction-optimal vector equals control-optimal vector, or the
    /// discrepancy is measured and explained. **Mandatory for L3**; a probe
    /// that predicts B at high accuracy but cannot steer B is L1/L2 at most,
    /// never L3. L3-required.
    PredictControlParity = 13,
    /// §2.2 L3 row + §5 S4: specificity — intervention changes target B more
    /// than closely-related non-target behaviors. Report full shift vector.
    /// L3-required.
    Specificity = 14,
    /// §2.2 L3 row: general-capability control (MMLU / MT-Bench-equivalent;
    /// in our domain: arena win-rate, baseline-reasoning benchmark) does not
    /// degrade outside a pre-registered tolerance. L3-required.
    GeneralCapabilityControl = 15,
    /// §2.2 L3 row + §5 S4: ≥1 falsifiable competing explanation tested and
    /// reported (e.g. paper Experiment 3 stress test preserving surface cue
    /// while removing target). L3-required.
    FalsifiableCompetingExplanation = 16,
    /// §2.2 L3 row + §5 S4: failure cases reported — where the effect
    /// disappears, flips sign, or produces broad degradation. L3-required.
    FailureCases = 17,
}

/// A single row of the R287 §5 checklist, tagged by its minimum level.
///
/// Used by [`crate::claim_rubric::checklist::section_items`] to render the
/// S1–S4 tables. The `min_level` field is informational (the
/// [`crate::claim_rubric::checklist::requirements`] table is the source of
/// truth for which items a level *requires*).
#[derive(Clone, Copy, Debug)]
pub struct EvidenceItem {
    /// Which checklist row this is.
    pub id: EvidenceItemId,
    /// Short human-readable description (paraphrased from R287 §5).
    pub description: &'static str,
    /// Minimum level at which this item becomes *required* (R287 §5 row tag).
    pub min_level: EvidenceLevel,
}

/// The four sections of the R287 §5 validation checklist.
///
/// `S1` — Target behavior framing. `S2` — Data / measurement construction.
/// `S3` — Experimental design. `S4` — Causal / mechanistic attribution.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChecklistSection {
    /// Target behavior framing (what you claim). R287 §5 S1.
    S1 = 0,
    /// Data / measurement construction. R287 §5 S2.
    S2 = 1,
    /// Experimental design. R287 §5 S3.
    S3 = 2,
    /// Causal / mechanistic attribution. R287 §5 S4.
    S4 = 3,
}

/// A single probe / steering claim with its declared evidence level and the
/// checklist items its supporting note actually satisfies.
///
/// Construct via [`Claim::new`] then chain [`Claim::with_evidence`]. Grade
/// via [`Claim::grade`] or [`crate::claim_rubric::validator::ClaimValidator`].
#[derive(Clone, Debug)]
pub struct Claim {
    /// Human-readable claim text, e.g. "the probe detects desperation at 87%
    /// accuracy". Scanned by [`crate::claim_rubric::vocabulary::scan`] for
    /// level-forbidden verbs (R287 §2.3).
    pub text: String,
    /// Re-exported `FeatureClass` tag (Detection vs Prediction). R287 §3 uses
    /// this as a *claim tag*, not a math operation. Currently informational;
    /// future revisions may apply the §3 row-2 Prediction-parity rule.
    pub feature_class: FeatureClass,
    /// What the claim asserts (L1/L2/L3). Echoed back in [`Grade`] as
    /// `declared_level` so consumers can compare it to the validator's
    /// `honest_level`.
    pub declared_level: EvidenceLevel,
    /// Which checklist items the claim's supporting note actually provides
    /// evidence for. Compared against
    /// [`crate::claim_rubric::checklist::requirements`] to determine
    /// `honest_level`.
    pub satisfied: Vec<EvidenceItemId>,
}

impl Claim {
    /// Construct a new claim with no satisfied items. Use [`Self::with_evidence`]
    /// to record which checklist items the claim's note provides evidence for.
    #[inline]
    #[must_use]
    pub fn new(
        text: impl Into<String>,
        feature_class: FeatureClass,
        declared_level: EvidenceLevel,
    ) -> Self {
        Self {
            text: text.into(),
            feature_class,
            declared_level,
            satisfied: Vec::new(),
        }
    }

    /// Builder: extend `satisfied` with the given items. Returns `self` for
    /// chaining.
    #[inline]
    #[must_use]
    pub fn with_evidence(mut self, items: &[EvidenceItemId]) -> Self {
        self.satisfied.extend_from_slice(items);
        self
    }

    /// Convenience: grade this claim with a default [`ClaimValidator`].
    ///
    /// Equivalent to `ClaimValidator::default().grade(self)`. Provided so
    /// single-claim ad-hoc checks read as `claim.grade()`.
    #[inline]
    #[must_use]
    pub fn grade(&self) -> Grade {
        ClaimValidator.grade(self)
    }
}

/// A claim's graded result — the validator's verdict on whether the evidence
/// actually supports the declared level.
///
/// The headline field is [`Grade::honest_level`]: the highest level whose
/// requirements are all satisfied AND whose vocabulary the claim's text
/// upholds (R287 §2.3).
#[derive(Clone, Debug)]
pub struct Grade {
    /// The max level whose requirements are all satisfied AND whose
    /// vocabulary the claim upholds. This is the level the claim is
    /// *licensed to assert*. If `honest_level < declared_level`, the claim
    /// is overclaiming (see [`Grade::downgraded`]).
    pub honest_level: EvidenceLevel,
    /// Echo of the claim's declared level (what the author asserted).
    pub declared_level: EvidenceLevel,
    /// Items required for `declared_level` but absent from `claim.satisfied`.
    /// Drives [`ClaimValidator::promote_advice`].
    pub missing_for_declared: Vec<EvidenceItemId>,
    /// Verbs found in the claim text that are forbidden at the claim's
    /// honest (evidence-backed) level (R287 §2.3). Empty when the vocabulary
    /// is honest. A non-empty list means the author is using language
    /// stronger than the evidence supports — reword or add evidence.
    pub vocabulary_violations: Vec<VocabularyViolation>,
    /// `true` iff `honest_level < declared_level`. The claim is overclaiming
    /// — either missing evidence (see `missing_for_declared`) or using
    /// vocabulary above its honest level (see `vocabulary_violations`).
    pub downgraded: bool,
}

/// A vocabulary violation — a level-restricted verb found in claim text at a
/// level below where it is allowed (R287 §2.3).
///
/// Example: `"causally controls"` is L3-only; finding it in a claim whose
/// evidence only supports L1 is a `VocabularyViolation { verb: "causally
/// controls", found_at_level: L1, max_allowed_level: L3 }`. The author must
/// either reword (e.g. to "predicts") or upgrade the evidence to L3.
#[derive(Clone, Debug)]
pub struct VocabularyViolation {
    /// The forbidden verb phrase, as it appears in the vocabulary table
    /// (R287 §2.2 row it belongs to).
    pub verb: &'static str,
    /// The honest (evidence-backed) level of the claim — the level the
    /// evidence actually supports, NOT the declared level. The violation is
    /// "this verb demands a higher evidence level than the claim has".
    pub found_at_level: EvidenceLevel,
    /// The minimum level at which this verb becomes permissible.
    pub max_allowed_level: EvidenceLevel,
}

impl VocabularyViolation {
    /// Human-readable description, e.g.
    /// `"verb 'causally controls' is L3-only but claim is L1"`.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "verb '{}' is {only}-only but claim is {found}",
            self.verb,
            only = self.max_allowed_level.label(),
            found = self.found_at_level.label(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_ordering() {
        assert!(EvidenceLevel::L0 < EvidenceLevel::L1);
        assert!(EvidenceLevel::L1 < EvidenceLevel::L2);
        assert!(EvidenceLevel::L2 < EvidenceLevel::L3);
    }

    #[test]
    fn level_labels_and_descriptions() {
        assert_eq!(EvidenceLevel::L0.label(), "L0");
        assert_eq!(EvidenceLevel::L1.label(), "L1");
        assert_eq!(EvidenceLevel::L2.label(), "L2");
        assert_eq!(EvidenceLevel::L3.label(), "L3");
        assert_eq!(EvidenceLevel::L0.description(), "no supporting evidence");
        assert_eq!(
            EvidenceLevel::L1.description(),
            "Behavioral — reads/detects/projects a signal"
        );
        assert_eq!(
            EvidenceLevel::L3.description(),
            "Causal-mechanistic — causally controls behavior"
        );
    }

    #[test]
    fn violation_description_format() {
        let v = VocabularyViolation {
            verb: "causally controls",
            found_at_level: EvidenceLevel::L1,
            max_allowed_level: EvidenceLevel::L3,
        };
        assert_eq!(
            v.description(),
            "verb 'causally controls' is L3-only but claim is L1"
        );
    }

    #[test]
    fn claim_builder_chains() {
        let c = Claim::new("reads", FeatureClass::Detection, EvidenceLevel::L1)
            .with_evidence(&[EvidenceItemId::OperationalDefinition]);
        assert_eq!(c.satisfied.len(), 1);
        assert_eq!(c.satisfied[0], EvidenceItemId::OperationalDefinition);
    }
}
