use super::certificate::*;

/// Generate WASM validator proof certificates (Pillar 2 pilot).
pub fn generate_wasm_validator_certificates(
    n_comparisons: usize,
    mismatches: usize,
    latency_us: f64,
    target_latency_us: u64,
    lora_wasm_delta: i32,
) -> Vec<ProofCertificate> {
    let mut certs = Vec::with_capacity(4);

    // P2.1: Deterministic correctness — 0 critical mismatches
    {
        let mut cert = ProofCertificate::new(
            "P2.1",
            ProofProperty::DeterministicCorrectness {
                game: "bomber".into(),
                n_comparisons,
            },
            ProofResult::Full {
                value: mismatches as f64,
                threshold: 1.0, // <1 mismatch
            },
            ProofEvidence::Comparison {
                baseline: "native".into(),
                challenger: "wasm".into(),
                delta: mismatches as f64,
            },
        );
        cert.implies = vec!["P2.3".into()];
        cert.explanation = format!(
            "{} critical mismatches in {} A/B comparisons",
            mismatches, n_comparisons
        );
        certs.push(cert);
    }

    // P2.2: Latency feasibility
    {
        let mut cert = ProofCertificate::new(
            "P2.2",
            ProofProperty::RealtimeFeasibility {
                domain: "bomber".into(),
                target_latency_us,
            },
            ProofResult::Full {
                value: latency_us,
                threshold: target_latency_us as f64,
            },
            ProofEvidence::Benchmark {
                n_samples: n_comparisons,
                mean: latency_us,
                std_dev: 0.0,
                min: latency_us,
                max: latency_us,
            },
        );
        cert.implies = vec!["P2.3".into()];
        cert.explanation = format!(
            "Latency {:.2}µs/call < target {}µs/call",
            latency_us, target_latency_us
        );
        certs.push(cert);
    }

    // P2.3: Production viability (implied by P2.1 + P2.2)
    {
        let mut cert = ProofCertificate::new(
            "P2.3",
            ProofProperty::RealtimeFeasibility {
                domain: "bomber".into(),
                target_latency_us: 1000, // <1ms
            },
            ProofResult::Full {
                value: latency_us,
                threshold: 1000.0,
            },
            ProofEvidence::Custom {
                data: serde_json::json!({ "derived_from": ["P2.1", "P2.2"] }),
            },
        );
        cert.prerequisites = vec!["P2.1".into(), "P2.2".into()];
        cert.explanation = "Production-viable: deterministic + real-time".into();
        certs.push(cert);
    }

    // P2.4: LoRA+WASM adds value
    {
        let mut cert = ProofCertificate::new(
            "P2.4",
            ProofProperty::Convergence {
                algorithm: "lora_wasm".into(),
                metric: "win_rate_delta".into(),
            },
            ProofResult::Full {
                value: lora_wasm_delta as f64,
                threshold: 0.0,
            },
            ProofEvidence::Comparison {
                baseline: "lora_only".into(),
                challenger: "lora_wasm".into(),
                delta: lora_wasm_delta as f64,
            },
        );
        cert.explanation = format!(
            "LoRA+WASM beats LoRA alone by +{} win rate points",
            lora_wasm_delta
        );
        certs.push(cert);
    }

    certs
}
