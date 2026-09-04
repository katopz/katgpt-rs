//! R287 §2.2 requirements table + §5 S1–S4 checklist as static data.
//!
//! Two parallel encodings of the rubric:
//!
//! - [`requirements`] — the R287 §2.2 per-level minimum-evidence rule. Used
//!   by the validator to compute `evidence_level` from `claim.satisfied`.
//! - [`S1_ITEMS`] / [`S2_ITEMS`] / [`S3_ITEMS`] / [`S4_ITEMS`] — the R287 §5
//!   validation checklist rows, grouped by section, tagged by minimum level.
//!   Used by [`section_items`] / [`full_checklist`] to render UI / CI reports.
//!
//! Both encodings reference [`EvidenceItemId`] variants by name — no string
//! duplication, no schema drift.

use crate::claim_rubric::types::{ChecklistSection, EvidenceItem, EvidenceItemId, EvidenceLevel};

// ──────────────────────────────────────────────────────────────────────────
// R287 §2.2 requirements table (per-level minimums)
// ──────────────────────────────────────────────────────────────────────────

/// L1-required items (R287 §2.2 row 1 + §5 S1/S2/S3 L1-tagged rows).
pub const L1_REQUIREMENTS: &[EvidenceItemId] = &[
    EvidenceItemId::OperationalDefinition,
    EvidenceItemId::SampleSize,
    EvidenceItemId::Ablation,
    EvidenceItemId::Exclusions,
    EvidenceItemId::LinearProbeCalibration,
];

/// L2-required items (R287 §2.2 row 2 + §5 S1/S3 L2-tagged rows). L2 inherits
/// all L1 items and adds these.
pub const L2_ADDITIONAL: &[EvidenceItemId] = &[
    EvidenceItemId::DownstreamEffect,
    EvidenceItemId::Generalization3Variations,
    EvidenceItemId::HumanGroundedValidation,
    EvidenceItemId::BaseRateReported,
    EvidenceItemId::LatentFreshnessCheck,
    EvidenceItemId::BenignShiftControl,
];

/// L3-required items (R287 §2.2 row 3 + §5 S4 L3-tagged rows). L3 inherits
/// all L2 items (and therefore all L1 items) and adds these.
pub const L3_ADDITIONAL: &[EvidenceItemId] = &[
    EvidenceItemId::Intervention,
    EvidenceItemId::PredictControlParity,
    EvidenceItemId::Specificity,
    EvidenceItemId::GeneralCapabilityControl,
    EvidenceItemId::FalsifiableCompetingExplanation,
    EvidenceItemId::FailureCases,
];

/// Static "all L1 items" slice (returned by [`requirements`] for L1).
pub const L1_FULL: &[EvidenceItemId] = L1_REQUIREMENTS;

/// Static "all items required at L2 (L1 ∪ L2-additional)" slice. Computed at
/// compile time via `const`-context concatenation.
pub const L2_FULL: &[EvidenceItemId] = &{
    let l1 = L1_REQUIREMENTS;
    let l2 = L2_ADDITIONAL;
    let mut out: [EvidenceItemId; L1_REQUIREMENTS.len() + L2_ADDITIONAL.len()] =
        [EvidenceItemId::OperationalDefinition; L1_REQUIREMENTS.len() + L2_ADDITIONAL.len()];
    let mut i = 0;
    while i < l1.len() {
        out[i] = l1[i];
        i += 1;
    }
    let mut j = 0;
    while j < l2.len() {
        out[i + j] = l2[j];
        j += 1;
    }
    out
};

/// Static "all items required at L3 (L1 ∪ L2 ∪ L3-additional)" slice.
pub const L3_FULL: &[EvidenceItemId] = &{
    let l1 = L1_REQUIREMENTS;
    let l2 = L2_ADDITIONAL;
    let l3 = L3_ADDITIONAL;
    let total: usize = L1_REQUIREMENTS.len() + L2_ADDITIONAL.len() + L3_ADDITIONAL.len();
    let mut out: [EvidenceItemId; 17] = [EvidenceItemId::OperationalDefinition; 17];
    let _ = total; // silence unused; total == out.len() by construction
    let mut i = 0;
    while i < l1.len() {
        out[i] = l1[i];
        i += 1;
    }
    let mut j = 0;
    while j < l2.len() {
        out[i + j] = l2[j];
        j += 1;
    }
    let i2 = i + j;
    let mut k = 0;
    while k < l3.len() {
        out[i2 + k] = l3[k];
        k += 1;
    }
    out
};

/// Empty slice for L0 (no requirements — the auto-downgrade floor).
pub const L0_FULL: &[EvidenceItemId] = &[];

/// R287 §2.2 per-level requirements table.
///
/// Returns the items that must **all** be satisfied to support a claim *at*
/// `level`. L1 returns the L1 row; L2 returns L1 ∪ L2-additional; L3 returns
/// L1 ∪ L2 ∪ L3-additional; L0 returns an empty slice (no requirements).
///
/// Zero allocations — returns a `&'static` slice built at compile time.
#[must_use]
pub const fn requirements(level: EvidenceLevel) -> &'static [EvidenceItemId] {
    match level {
        EvidenceLevel::L0 => L0_FULL,
        EvidenceLevel::L1 => L1_FULL,
        EvidenceLevel::L2 => L2_FULL,
        EvidenceLevel::L3 => L3_FULL,
    }
}

// ──────────────────────────────────────────────────────────────────────────
// R287 §5 S1–S4 checklist tables
// ──────────────────────────────────────────────────────────────────────────

/// S1 — Target behavior framing (R287 §5 S1).
pub const S1_ITEMS: &[EvidenceItem] = &[
    EvidenceItem {
        id: EvidenceItemId::OperationalDefinition,
        description: "1–3 sentence operational definition in measurable terms (threshold, aggregation).",
        min_level: EvidenceLevel::L1,
    },
    EvidenceItem {
        id: EvidenceItemId::Exclusions,
        description: "List what the definition excludes (near-misses, lookalikes, non-targets).",
        min_level: EvidenceLevel::L1,
    },
    EvidenceItem {
        id: EvidenceItemId::DownstreamEffect,
        description: "State deployment-plausible context (arena / game system / sync tier).",
        min_level: EvidenceLevel::L2,
    },
];

/// S2 — Data / measurement construction (R287 §5 S2).
pub const S2_ITEMS: &[EvidenceItem] = &[
    EvidenceItem {
        id: EvidenceItemId::SampleSize,
        description: "Report n (independent generations), seeds, temperature, top-p; justify vs effect size.",
        min_level: EvidenceLevel::L1,
    },
    EvidenceItem {
        id: EvidenceItemId::Ablation,
        description: "Diversity across paraphrases / domains / formats; ablate prompts and sampling.",
        min_level: EvidenceLevel::L1,
    },
    EvidenceItem {
        id: EvidenceItemId::LinearProbeCalibration,
        description: "Direction-vector norm, projection calibration (reliability diagram / MAE), label source.",
        min_level: EvidenceLevel::L1,
    },
];

/// S3 — Experimental design (R287 §5 S3).
pub const S3_ITEMS: &[EvidenceItem] = &[
    EvidenceItem {
        id: EvidenceItemId::Generalization3Variations,
        description: "≥3 variations: paraphrase, seed, model variant, temperature; report sensitivity.",
        min_level: EvidenceLevel::L2,
    },
    EvidenceItem {
        id: EvidenceItemId::HumanGroundedValidation,
        description: "If LLM judge: report model/prompt/n, validate vs human subset, agreement, errors.",
        min_level: EvidenceLevel::L2,
    },
    EvidenceItem {
        id: EvidenceItemId::BaseRateReported,
        description: "Report base rate of effect (esp. rare behaviors — extreme-value estimation).",
        min_level: EvidenceLevel::L2,
    },
    EvidenceItem {
        id: EvidenceItemId::LatentFreshnessCheck,
        description: "If reading a latent state, confirm freshness window (R286 drift / fog-of-war) or downgrade.",
        min_level: EvidenceLevel::L2,
    },
    EvidenceItem {
        id: EvidenceItemId::BenignShiftControl,
        description: "≥1 OOD / benign-shift control: is effect specific to intended mechanism vs distribution shift?",
        min_level: EvidenceLevel::L2,
    },
    EvidenceItem {
        id: EvidenceItemId::GeneralCapabilityControl,
        description: "General-capability pre/post measurement (arena win-rate, baseline benchmark) within tolerance.",
        min_level: EvidenceLevel::L2,
    },
];

/// S4 — Causal / mechanistic attribution (R287 §5 S4).
pub const S4_ITEMS: &[EvidenceItem] = &[
    EvidenceItem {
        id: EvidenceItemId::Intervention,
        description: "Intervention (ablate / zero / clamp / steer) along w_B produces pre-registered change in B.",
        min_level: EvidenceLevel::L3,
    },
    EvidenceItem {
        id: EvidenceItemId::PredictControlParity,
        description: "Predict-control parity: predict-optimal vector = control-optimal, or discrepancy explained.",
        min_level: EvidenceLevel::L3,
    },
    EvidenceItem {
        id: EvidenceItemId::Specificity,
        description: "Specificity: target changes more than non-target behaviors; report full shift vector.",
        min_level: EvidenceLevel::L3,
    },
    EvidenceItem {
        id: EvidenceItemId::FalsifiableCompetingExplanation,
        description: "≥1 falsifiable competing explanation tested (paper Experiment 3 stress test style).",
        min_level: EvidenceLevel::L3,
    },
    EvidenceItem {
        id: EvidenceItemId::FailureCases,
        description: "Failure cases reported — where effect disappears, flips, or broadly degrades.",
        min_level: EvidenceLevel::L3,
    },
];

/// R287 §5 table accessor: returns the items in a given section.
///
/// Zero allocations — returns a `&'static` slice.
#[must_use]
pub const fn section_items(section: ChecklistSection) -> &'static [EvidenceItem] {
    match section {
        ChecklistSection::S1 => S1_ITEMS,
        ChecklistSection::S2 => S2_ITEMS,
        ChecklistSection::S3 => S3_ITEMS,
        ChecklistSection::S4 => S4_ITEMS,
    }
}

/// R287 §5 full checklist as a 4-tuple, in S1→S2→S3→S4 order.
///
/// Useful for iterating the entire checklist when rendering a CI report or
/// looking up an item's description by id (see
/// [`crate::claim_rubric::validator::ClaimValidator::promote_advice`]).
#[must_use]
pub const fn full_checklist() -> [(ChecklistSection, &'static [EvidenceItem]); 4] {
    [
        (ChecklistSection::S1, S1_ITEMS),
        (ChecklistSection::S2, S2_ITEMS),
        (ChecklistSection::S3, S3_ITEMS),
        (ChecklistSection::S4, S4_ITEMS),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_requirements_are_the_five_l1_items() {
        let r = requirements(EvidenceLevel::L1);
        assert_eq!(r.len(), 5);
        assert!(r.contains(&EvidenceItemId::OperationalDefinition));
        assert!(r.contains(&EvidenceItemId::SampleSize));
        assert!(r.contains(&EvidenceItemId::Ablation));
        assert!(r.contains(&EvidenceItemId::Exclusions));
        assert!(r.contains(&EvidenceItemId::LinearProbeCalibration));
    }

    #[test]
    fn l2_inherits_l1_and_adds_six() {
        let r = requirements(EvidenceLevel::L2);
        assert_eq!(r.len(), 11, "5 L1 + 6 L2-additional");
        // All L1 items still present.
        assert!(r.contains(&EvidenceItemId::OperationalDefinition));
        // L2-only items present.
        assert!(r.contains(&EvidenceItemId::DownstreamEffect));
        assert!(r.contains(&EvidenceItemId::Generalization3Variations));
        assert!(r.contains(&EvidenceItemId::LatentFreshnessCheck));
        assert!(r.contains(&EvidenceItemId::BenignShiftControl));
    }

    #[test]
    fn l3_inherits_l2_and_adds_six() {
        let r = requirements(EvidenceLevel::L3);
        assert_eq!(r.len(), 17, "5 L1 + 6 L2 + 6 L3");
        // All L1 + L2 items still present.
        assert!(r.contains(&EvidenceItemId::OperationalDefinition));
        assert!(r.contains(&EvidenceItemId::DownstreamEffect));
        // L3-only items present.
        assert!(r.contains(&EvidenceItemId::Intervention));
        assert!(r.contains(&EvidenceItemId::PredictControlParity));
        assert!(r.contains(&EvidenceItemId::Specificity));
        assert!(r.contains(&EvidenceItemId::FailureCases));
    }

    #[test]
    fn l0_requirements_empty() {
        assert!(requirements(EvidenceLevel::L0).is_empty());
    }

    #[test]
    fn section_items_match_constants() {
        assert_eq!(section_items(ChecklistSection::S1).len(), S1_ITEMS.len());
        assert_eq!(section_items(ChecklistSection::S4).len(), 5);
    }

    #[test]
    fn full_checklist_has_four_sections_in_order() {
        let fc = full_checklist();
        assert_eq!(fc.len(), 4);
        assert_eq!(fc[0].0, ChecklistSection::S1);
        assert_eq!(fc[1].0, ChecklistSection::S2);
        assert_eq!(fc[2].0, ChecklistSection::S3);
        assert_eq!(fc[3].0, ChecklistSection::S4);
    }
}
