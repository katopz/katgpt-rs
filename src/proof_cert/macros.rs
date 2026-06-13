/// Macro for declaring conditional GOAT proofs.
#[macro_export]
macro_rules! conditional_proof {
    (
        $id:expr,
        property = $prop:expr,
        value = $val:expr,
        threshold = $thresh:expr,
        conditions = [$( $cond:expr ),* $(,)?],
        implies = [$( $imp:expr ),* $(,)?]
    ) => {{
        $crate::proof_cert::ProofCertificate {
            id: $id.into(),
            property: $prop,
            result: $crate::proof_cert::ProofResult::Conditional {
                value: $val,
                threshold: $thresh,
                conditions: vec![$( $cond.into() ),*],
            },
            prerequisites: vec![],
            implies: vec![$( $imp.into() ),*],
            explanation: String::new(),
            evidence: $crate::proof_cert::ProofEvidence::Custom {
                data: Vec::new(),
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }};
}
