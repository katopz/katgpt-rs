//! Benchmark for Trust-Region Adaptive Speculation (Plan 182, T7).
//!
//! Measures:
//! 1. Blend computation: πS^(1-β)·πT^β for vocab_size tokens
//! 2. Binary search for β: 10 iterations over KL computation
//! 3. Overall trust region state overhead

#[cfg(feature = "trust_region_spec")]
mod benches {
    use katgpt_rs::speculative::{
        TrustArm, TrustRegionConfig, TrustRegionState, TrustTracker, adaptive_window, blend_sample,
        find_blend_beta,
    };
    use katgpt_rs::types::Rng;
    use std::time::Instant;

    /// Benchmark: blend_sample for vocab_size=256 tokens
    #[test]
    fn bench_blend_sample_vocab_256() {
        let mut rng = Rng::new(42);
        // Simulate teacher and student distributions
        let mut p = vec![0.0f32; 256];
        let mut q = vec![0.0f32; 256];
        // Teacher: concentrated on token 0
        p[0] = 0.7;
        for i in 1..256 {
            p[i] = 0.3 / 255.0;
        }
        // Student: concentrated on token 1
        q[1] = 0.6;
        for i in 0..256 {
            if i != 1 {
                q[i] = 0.4 / 255.0;
            }
        }

        let warmup = 100;
        let runs = 10_000;

        // Warmup
        for _ in 0..warmup {
            let _ = blend_sample(&p, &q, 0.5, &mut rng);
        }

        let start = Instant::now();
        for _ in 0..runs {
            let _ = blend_sample(&p, &q, 0.5, &mut rng);
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_nanos() as f64 / runs as f64 / 1000.0;

        eprintln!("blend_sample (vocab=256): {:.2} μs/call", per_call_us);
        // Assert: blend cost < 50μs (generous for non-SIMD; 2μs is the hot-path target)
        assert!(
            per_call_us < 50.0,
            "blend_sample too slow: {:.2} μs/call (target < 50 μs)",
            per_call_us
        );
    }

    /// Benchmark: find_blend_beta binary search
    #[test]
    fn bench_find_blend_beta() {
        let p = vec![0.7, 0.1, 0.1, 0.05, 0.05];
        let q = vec![0.1, 0.6, 0.1, 0.1, 0.1];

        let warmup = 100;
        let runs = 10_000;

        for _ in 0..warmup {
            let _ = find_blend_beta(&p, &q, 0.1, 10);
        }

        let start = Instant::now();
        for _ in 0..runs {
            let _ = find_blend_beta(&p, &q, 0.1, 10);
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_nanos() as f64 / runs as f64 / 1000.0;

        eprintln!(
            "find_blend_beta (5 tokens, 10 iters): {:.2} μs/call",
            per_call_us
        );
        assert!(
            per_call_us < 20.0,
            "find_blend_beta too slow: {:.2} μs/call (target < 20 μs)",
            per_call_us
        );
    }

    /// Benchmark: adaptive_window is trivially fast
    #[test]
    fn bench_adaptive_window_trivial() {
        let config = TrustRegionConfig::default();

        let start = Instant::now();
        for i in 0..1_000_000 {
            let trust = (i as f32 % 100.0) / 100.0;
            let _ = adaptive_window(trust, 5, &config);
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / 1_000_000.0;

        eprintln!("adaptive_window: {:.1} ns/call", per_call_ns);
        assert!(
            per_call_ns < 100.0,
            "adaptive_window too slow: {:.1} ns/call",
            per_call_ns
        );
    }

    /// Benchmark: TrustTracker recording
    #[test]
    fn bench_trust_tracker_record() {
        let mut tracker = TrustTracker::new(16);

        let start = Instant::now();
        for i in 0..1_000_000 {
            let trust = (i as f32 % 100.0) / 100.0;
            tracker.record(trust);
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / 1_000_000.0;

        eprintln!("TrustTracker::record: {:.1} ns/call", per_call_ns);
        assert!(
            per_call_ns < 100.0,
            "TrustTracker::record too slow: {:.1} ns/call",
            per_call_ns
        );
    }

    /// Benchmark: TrustArm serialization roundtrip
    #[test]
    fn bench_trust_arm_roundtrip() {
        let arm = TrustArm::new("benchmark", 8);

        let start = Instant::now();
        for _ in 0..100_000 {
            let bytes = arm.to_bytes();
            let _ = TrustArm::from_bytes(&bytes);
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / 100_000.0 / 2.0; // divide by 2 for serialize+deserialize

        eprintln!("TrustArm roundtrip: {:.1} ns/call", per_call_ns);
        assert!(
            per_call_ns < 500.0,
            "TrustArm roundtrip too slow: {:.1} ns/call",
            per_call_ns
        );
    }

    /// GOAT: TrustRegionState overhead vs baseline
    #[test]
    fn goat_trust_region_overhead_acceptable() {
        let mut state = TrustRegionState::default_state();
        let mut rng = Rng::new(42);

        // Simulate 1000 tokens with trust tracking
        let start = Instant::now();
        for i in 0..1000 {
            let trust = 0.5 + 0.5 * ((i as f32 / 100.0).sin());
            state.record_acceptance(trust);
            let win = state.window(5);
            if trust < 0.5 {
                // Simulate blend on rejection
                let p = vec![0.5, 0.3, 0.2];
                let q = vec![0.3, 0.4, 0.3];
                let _ = state.blend_on_reject(&p, &q, &mut rng);
            }
            let _ = win;
        }
        let elapsed = start.elapsed();
        let per_token_us = elapsed.as_nanos() as f64 / 1000.0 / 1000.0;

        eprintln!("TRAS per-token overhead: {:.2} μs/token", per_token_us);
        // Acceptable: < 100μs per token (spec decode typically takes 1-5ms per token)
        assert!(
            per_token_us < 100.0,
            "TRAS overhead too high: {:.2} μs/token",
            per_token_us
        );
    }
}
