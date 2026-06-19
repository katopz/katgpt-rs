//! Plan 292 Phase 4 — FPCG GOAT Gate (achievable subset).
//!
//! Phase 4 gates G1–G4 require a *trained* `FutureBehaviorProbe` direction
//! vector (T4.2 — offline logistic regression on mid-layer activations) and a
//! *test corpus* with ground-truth behavior labels (T4.1 — paper's resampling
//! recipe S=10 × M=10). Both are out of scope for the public `katgpt-rs`
//! engine: offline training lives in `riir-train` per `AGENTS.md`, and a
//! behavior-labeled corpus is external data. G1–G4 are therefore **BLOCKED** —
//! see `.issues/032_fpcg_phase4_training_blocker.md` and the methodology
//! section of `.benchmarks/292_fpcg_goat.md`. No fabricated numbers.
//!
//! This file covers the **achievable-in-pure-Rust** subset of the gate:
//!
//! | Gate | What | How measured | Status |
//! |------|------|--------------|--------|
//! | G5 | Zero-alloc hot path across 1000 selector steps | `Vec::capacity` stable before/after | runnable here |
//! | G7 | BLAKE3 commitment — tampered bytes refuse to serve | `load_from_bytes` returns `Err` on tamper, `Ok` + serves on clean | runnable here |
//!
//! (G6 — `forecast()` latency < 200ns — lives in `benches/fpcg_probe_forecast_bench.rs`
//! because it needs `harness = false` + `fn main()` per the repo bench convention.)
//!
//! **Run:**
//! ```text
//! cargo test -p katgpt-rs --features fpcg_selector --test fpcg_goat_gate -- --nocapture
//! ```

#![cfg(feature = "fpcg_selector")]

use std::collections::HashMap;
use std::sync::Arc;

use fastrand::Rng;

use katgpt_rs::pruners::fpcg_selector::{
    ActivationExtractor, FpcgSelector, SentenceCandidateGenerator, SteeringDirection,
};
use katgpt_rs::pruners::future_probe::{FutureBehaviorProbe, ProbeLoadError};

// ── Minimal public-API stubs (the private stubs in fpcg_selector::tests are
//    not visible from an integration test, so we define fresh minimal ones). ──

/// Generator that cycles through a fixed candidate pool, producing `n` strings
/// per call (deterministic given `call_count`). Mirrors the Phase 3 unit test's
/// `CyclingGenerator` but lives here so the GOAT gate is self-contained.
struct CyclingGenerator {
    pool: Vec<String>,
    call_count: usize,
}

impl SentenceCandidateGenerator for CyclingGenerator {
    fn generate_candidates(&mut self, _prefix: &str, n: usize, _rng: &mut Rng) -> Vec<String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = self.call_count % self.pool.len();
            out.push(self.pool[idx].clone());
            self.call_count += 1;
        }
        out
    }
}

/// Extractor that maps each candidate string to a pre-registered activation
/// vector, copied into the caller-provided scratch buffer (zero extra alloc
/// beyond the lookup). Unknown candidates get the zero activation.
struct LookupExtractor {
    activations: HashMap<String, Vec<f32>>,
}

impl ActivationExtractor for LookupExtractor {
    fn extract_activation<'a>(
        &'a mut self,
        _prefix: &str,
        candidate: &str,
        _layer: usize,
        scratch: &'a mut [f32],
    ) -> &'a [f32] {
        match self.activations.get(candidate) {
            Some(act) => {
                let len = act.len().min(scratch.len());
                scratch[..len].copy_from_slice(&act[..len]);
                // Zero any trailing scratch the activation doesn't cover so the
                // probe reads deterministic bytes.
                for v in scratch[len..].iter_mut() {
                    *v = 0.0;
                }
                scratch
            }
            None => {
                for v in scratch.iter_mut() {
                    *v = 0.0;
                }
                scratch
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// G5 — Zero-alloc hot path across 1000 selector steps
// ──────────────────────────────────────────────────────────────────────────

/// Plan 292 T4.5 G5 / T3.6: the selector's pre-allocated `candidates_buf`
/// MUST NOT grow across 1000 `step()` calls. We record `candidates_buf_capacity()`
/// before the first step and after the 1000th and assert equality — no growth
/// means no reallocation, which is the zero-alloc hot-path contract.
///
/// The selector is constructed with `num_candidates = 10` (paper default). The
/// cycling generator always returns exactly `num_candidates` candidates per
/// call, so the buffer is filled to capacity every step — the worst case for
/// reallocation. If capacity stays stable here, it stays stable everywhere.
#[test]
fn g5_zero_alloc_hot_path_across_1000_steps() {
    let d_model: usize = 4;
    let pool: Vec<String> = (0..5).map(|i| format!("candidate_{i}.")).collect();
    let activations: HashMap<String, Vec<f32>> = pool
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), vec![i as f32 - 2.0, 0.0, 0.0, 0.0]))
        .collect();

    // Probe where direction[0] is the only nonzero element, so forecast is
    // monotone in activation[0] — gives the selector a real signal to argmax.
    let probe = Arc::new(FutureBehaviorProbe::new(
        vec![1.0, 0.0, 0.0, 0.0],
        0.0,
        0,
        "g5_stub",
    ));
    let generator = CyclingGenerator {
        pool,
        call_count: 0,
    };
    let extractor = LookupExtractor { activations };

    let mut sel = FpcgSelector::new(
        generator,
        extractor,
        probe,
        SteeringDirection::Positive,
        10, // num_candidates — paper default
        d_model,
    );

    let capacity_before = sel.candidates_buf_capacity();
    let mut rng = Rng::with_seed(42);
    let mut last_chosen = String::new();
    for _ in 0..1000 {
        last_chosen = sel.step("irrelevant prefix", &mut rng);
    }
    let capacity_after = sel.candidates_buf_capacity();

    // Sanity: the loop actually ran and produced candidates.
    assert!(
        !last_chosen.is_empty(),
        "G5 setup sanity: selector should have produced a candidate"
    );
    assert_eq!(
        capacity_before, capacity_after,
        "G5 FAIL: candidates_buf capacity grew across 1000 steps \
         (before={capacity_before}, after={capacity_after}) — hot path is not zero-alloc"
    );
    println!(
        "G5 PASS: candidates_buf capacity stable at {capacity_before} across 1000 steps \
         (last_chosen={last_chosen:?})"
    );
}

// ──────────────────────────────────────────────────────────────────────────
// G7 — BLAKE3 commitment: clean load serves, tampered load refuses
// ──────────────────────────────────────────────────────────────────────────

/// Plan 292 T4.5 G7 / T2.4: a probe reloaded from its own serialized bytes
/// MUST serve forecasts identical to the original, and a probe reloaded from
/// *tampered* bytes MUST refuse to serve (return `ProbeLoadError::HashMismatch`).
///
/// This test exercises both halves of the contract in one place so the G7
/// gate is self-documenting:
///
/// (a) Clean round-trip: `save_to_bytes` → `load_from_bytes` → `Ok`, and the
///     loaded probe's `forecast()` returns a finite probability matching the
///     original to within f32 epsilon.
/// (b) Tampered reload: flip one direction byte post-serialization →
///     `load_from_bytes` returns `Err(HashMismatch)`.
#[test]
fn g7_blake3_commitment_clean_loads_and_tamper_refusal() {
    // ── (a) Clean round-trip serves and forecasts ──────────────────────
    let original = FutureBehaviorProbe::new(
        vec![0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        0.42,
        13,
        "refusal",
    );
    let clean_bytes = original.save_to_bytes();
    let loaded = FutureBehaviorProbe::load_from_bytes(&clean_bytes).expect(
        "G7 (a): clean round-trip load must succeed — if this fails the BLAKE3 \
                 commitment is rejecting untampered bytes, which is a bug",
    );
    assert_eq!(
        loaded.artifact_hash(),
        original.artifact_hash(),
        "G7 (a): loaded probe hash must match original"
    );
    let activation = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let f_orig = original.forecast(&activation).probability;
    let f_loaded = loaded.forecast(&activation).probability;
    assert!(
        f_orig.is_finite() && f_loaded.is_finite(),
        "G7 (a): forecasts must be finite (orig={f_orig}, loaded={f_loaded})"
    );
    assert!(
        (f_orig - f_loaded).abs() < 1e-6,
        "G7 (a): clean-loaded probe must forecast identically to original \
         (orig={f_orig}, loaded={f_loaded})"
    );
    println!(
        "G7 (a) PASS: clean round-trip load serves, forecast probability = {f_loaded:.6}"
    );

    // ── (b) Tampered reload refuses ─────────────────────────────────────
    // The commitment bytes are `layer (LE u64) || bias (LE f32) || direction
    // (LE f32 × d)`. The direction bytes start at offset 4 (magic) + 1 (ver) +
    // 8 (layer) + 4 (bias) + 4 (d_model) = 21. Flip the first direction byte —
    // this changes the numerical content without tripping the header checks,
    // so the only thing that can catch it is the BLAKE3 recomputation.
    let mut tampered = clean_bytes.clone();
    assert!(
        tampered.len() > 21,
        "G7 (b) setup: serialized probe must be long enough to have direction bytes"
    );
    tampered[21] ^= 0xFF;

    match FutureBehaviorProbe::load_from_bytes(&tampered) {
        Err(ProbeLoadError::HashMismatch { embedded, computed }) => {
            assert_ne!(
                embedded, computed,
                "G7 (b): HashMismatch must report genuinely different hashes"
            );
            println!(
                "G7 (b) PASS: tampered reload refused with HashMismatch \
                 (embedded={:#x?}, computed={:#x?})",
                &embedded[..4],
                &computed[..4],
            );
        }
        Ok(_) => panic!(
            "G7 (b) FAIL: tampered probe bytes loaded successfully — the BLAKE3 \
             commitment is NOT enforcing hash-check-on-load (Plan 292 T2.4 violation)"
        ),
        Err(other) => panic!(
            "G7 (b) FAIL: tampered reload returned {other:?}, expected HashMismatch \
             (a different error means the tamper tripped a header check before reaching \
             the hash comparison — the test must tamper *direction* bytes, not headers)"
        ),
    }
}

// ── Sanity: feature-class tag from the public API (Phase 1 contract) ──────

/// Lightweight confirmation that the prediction-side vocabulary tag is visible
/// from the integration-test boundary. Phase 1 is the always-on output; this
/// just guards against an accidental re-export regression.
#[test]
fn prediction_feature_class_visible_from_public_api() {
    use katgpt_core::traits::{FeatureClass, ScreeningPruner};
    let probe = FutureBehaviorProbe::new(vec![1.0, 0.0], 0.0, 0, "sanity");
    assert_eq!(probe.feature_class(), FeatureClass::Prediction);
}
