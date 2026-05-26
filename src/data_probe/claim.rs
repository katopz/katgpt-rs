//! Claim card infrastructure for formal C1–C4 validation.
//!
//! Each claim card captures a testable hypothesis about the data-probe pipeline,
//! with an intervention (knob, baseline, treatment, expected direction) and
//! a falsification condition. Verdict logic distinguishes:
//!
//! - **TransferAccepted**: both internal and external validity confirmed
//! - **ProbeLocal**: internal validity confirmed but external validity failed
//! - **Rejected**: internal validity failed

/// A structured claim card for information-theoretic hypothesis validation.
pub struct ClaimCard {
    /// Human-readable claim statement (e.g., "Perplexity tracks entropy rate").
    pub claim: String,
    /// Description of the validation process.
    pub process_description: String,
    /// The intervention being tested.
    pub intervention: Intervention,
    /// Diagnostic metric or test used.
    pub diagnostic: String,
    /// Condition under which the claim is falsified.
    pub falsification_condition: String,
    /// Internal validity verdict (probe-local experiment).
    pub internal_validity: Option<ValidityVerdict>,
    /// External validity verdict (generalization beyond probe).
    pub external_validity: Option<ValidityVerdict>,
}

/// An intervention with a single knob varied between baseline and treatment.
pub struct Intervention {
    /// Name of the knob being varied (e.g., "entropy_rate", "sequence_length").
    pub knob: String,
    /// Baseline value of the knob.
    pub baseline: String,
    /// Treatment value of the knob.
    pub treatment: String,
    /// Expected direction of effect: +1 for increase, -1 for decrease, 0 for no change.
    pub expected_direction: i8,
}

/// Verdict on claim validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidityVerdict {
    /// Claim confirmed with evidence of external generalization.
    TransferAccepted,
    /// Claim confirmed within the probe but not externally.
    ProbeLocal,
    /// Claim rejected — evidence contradicts it.
    Rejected,
}

impl ClaimCard {
    /// Compute the overall verdict from internal and external validity.
    ///
    /// Logic:
    /// - IV = TransferAccepted + EV = TransferAccepted → TransferAccepted
    /// - IV = TransferAccepted + EV ≠ TransferAccepted → ProbeLocal
    /// - IV ≠ TransferAccepted → Rejected
    pub fn verdict(&self) -> ValidityVerdict {
        match self.internal_validity {
            Some(ValidityVerdict::TransferAccepted) => match self.external_validity {
                Some(ValidityVerdict::TransferAccepted) => ValidityVerdict::TransferAccepted,
                _ => ValidityVerdict::ProbeLocal,
            },
            _ => ValidityVerdict::Rejected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_card(iv: Option<ValidityVerdict>, ev: Option<ValidityVerdict>) -> ClaimCard {
        ClaimCard {
            claim: "Test claim".into(),
            process_description: "Test process".into(),
            intervention: Intervention {
                knob: "test_knob".into(),
                baseline: "0".into(),
                treatment: "1".into(),
                expected_direction: 1,
            },
            diagnostic: "Test diagnostic".into(),
            falsification_condition: "Never".into(),
            internal_validity: iv,
            external_validity: ev,
        }
    }

    #[test]
    fn test_verdict_transfer_accepted() {
        let card = make_card(
            Some(ValidityVerdict::TransferAccepted),
            Some(ValidityVerdict::TransferAccepted),
        );
        assert_eq!(card.verdict(), ValidityVerdict::TransferAccepted);
    }

    #[test]
    fn test_verdict_probe_local_ev_none() {
        let card = make_card(Some(ValidityVerdict::TransferAccepted), None);
        assert_eq!(card.verdict(), ValidityVerdict::ProbeLocal);
    }

    #[test]
    fn test_verdict_probe_local_ev_rejected() {
        let card = make_card(
            Some(ValidityVerdict::TransferAccepted),
            Some(ValidityVerdict::Rejected),
        );
        assert_eq!(card.verdict(), ValidityVerdict::ProbeLocal);
    }

    #[test]
    fn test_verdict_probe_local_ev_probe_local() {
        let card = make_card(
            Some(ValidityVerdict::TransferAccepted),
            Some(ValidityVerdict::ProbeLocal),
        );
        assert_eq!(card.verdict(), ValidityVerdict::ProbeLocal);
    }

    #[test]
    fn test_verdict_rejected_iv_rejected() {
        let card = make_card(Some(ValidityVerdict::Rejected), None);
        assert_eq!(card.verdict(), ValidityVerdict::Rejected);
    }

    #[test]
    fn test_verdict_rejected_iv_probe_local() {
        let card = make_card(Some(ValidityVerdict::ProbeLocal), None);
        assert_eq!(card.verdict(), ValidityVerdict::Rejected);
    }

    #[test]
    fn test_verdict_rejected_iv_none() {
        let card = make_card(None, Some(ValidityVerdict::TransferAccepted));
        assert_eq!(card.verdict(), ValidityVerdict::Rejected);
    }

    // GOAT proof G6: Claim card round-trip — IV+EV=TransferAccepted → verdict is TransferAccepted.
    #[test]
    fn goat_g6_claim_card_round_trip() {
        // Construct a claim card with both IV and EV set to TransferAccepted.
        let card = ClaimCard {
            claim: "Perplexity tracks entropy rate".into(),
            process_description: "GOAT proof formal validation".into(),
            intervention: Intervention {
                knob: "entropy_rate".into(),
                baseline: "1.0".into(),
                treatment: "2.0".into(),
                expected_direction: 1,
            },
            diagnostic: "NLL convergence".into(),
            falsification_condition: "|NLL - H| > ε".into(),
            internal_validity: Some(ValidityVerdict::TransferAccepted),
            external_validity: Some(ValidityVerdict::TransferAccepted),
        };
        assert_eq!(
            card.verdict(),
            ValidityVerdict::TransferAccepted,
            "GOAT G6 FAIL: IV+EV=TransferAccepted should yield TransferAccepted"
        );
    }
}
