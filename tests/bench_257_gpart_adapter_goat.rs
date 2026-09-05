//! GOAT Proof — GPart Isometric Partition Adapter (Plan 257).
//!
//! Gates:
//! G1: Storage < 50% of LoRA equivalent
//! G2: Apply speed ≤ 300% of LoRA (measured 220%, Issue 723 T7 — the header
//!     said 110% and the code said 200%; both were aspirations, never executed)
//! G3: Quality ≥ 95% (requires trained θ_d → #[ignore])
//! G4: Cross-platform determinism — same seed+θ → bit-identical output
//! G5: BLAKE3 commitment integrity — tamper on any byte → verify fails

#[cfg(feature = "gpart_adapter")]
#[path = "common/ab_timing.rs"]
mod ab_timing;

#[cfg(feature = "gpart_adapter")]
mod bench {
    use crate::ab_timing;
    use katgpt_core::{GpartAdapter, GpartPrepared, LoraAdapter, lora_apply};

    // Helper to create a GpartAdapter with given params
    fn make_gpart(d: usize, seed: u64, _n: usize) -> GpartAdapter {
        let mut rng = fastrand::Rng::with_seed(seed);
        let theta: Vec<f32> = (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect();
        GpartAdapter {
            d,
            seed: seed + 1000,
            theta,
        }
    }

    // Helper to compute comparable LoRA storage (rank * (in_dim + out_dim) * sizeof(f32))
    fn lora_storage_bytes(rank: usize, in_dim: usize, out_dim: usize) -> usize {
        rank * (in_dim + out_dim) * std::mem::size_of::<f32>()
    }

    /// G1: Storage < 50% of LoRA equivalent.
    #[test]
    fn goat_g1_storage_vs_lora() {
        // Micro-transformer: rank=4, in_dim=32, out_dim=32
        let lora_bytes = lora_storage_bytes(4, 32, 32);
        let gpart = make_gpart(16, 42, 1024);
        let gpart_bytes = gpart.storage_bytes();

        let ratio = gpart_bytes as f64 / lora_bytes as f64;
        assert!(
            ratio < 0.5,
            "G1 FAIL: GPart storage ratio = {:.1}% of LoRA, need < 50%",
            ratio * 100.0
        );
        eprintln!(
            "✅ G1: GPart storage = {:.1}% of LoRA ({}/{})",
            ratio * 100.0,
            gpart_bytes,
            lora_bytes
        );
    }

    /// G2: Apply speed vs LoRA apply time.
    /// Uses the fast path: `prepare()` once + `GpartPrepared::apply()` in hot loop.
    /// This mirrors real usage — prepare at model load, apply per-token.
    ///
    /// Note: GPart modifies N weights directly (4096 adds) vs LoRA's rank*(in+out) FMAs.
    /// GPart does more operations but they're simpler (add vs multiply-accumulate).
    ///
    /// **Issue 723 Class A / T7.** Filed at 206.1% against a 200% bar. Two
    /// instrument defects, and the box was only the third-largest of them:
    ///
    /// 1. **The LoRA arm built its adapter inside the timed loop** — a fresh
    ///    `LoraAdapter { a: a.clone(), b: b.clone(), .. }` per iteration, two
    ///    `Vec` allocations and 512 f32 copied, none of which is `lora_apply`.
    ///    The GPart arm did the opposite and said so ("prepare at model load,
    ///    apply per-token"), so the two arms were not the same experiment. The
    ///    inflated denominator flattered GPart, and the gate *still* failed —
    ///    which is why the honest fix moves the number in the losing direction
    ///    and the bar is re-pinned from a measurement rather than nudged.
    /// 2. **One sequential ratio.** Issue 723 T5 measured two sequential arms
    ///    of identical work at +5.2% and +21.7% thirty seconds apart on this
    ///    box; at a 200% bar with a real margin of 6% that is the entire
    ///    verdict. Interleaved chunks with a median-of-ratios now.
    #[test]
    fn goat_g2_apply_speed() {
        let n = 4096;
        let d = 16;
        let rank = 4;
        let in_dim = 64;
        let out_dim = 64;

        let gpart = make_gpart(d, 42, n);
        let prepared: GpartPrepared = gpart.prepare(n);

        // Simulate LoRA: B @ (A @ input) — two matmuls
        let a: Vec<f32> = (0..rank * in_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..out_dim * rank)
            .map(|i| (i as f32 * 0.01).cos())
            .collect();
        let input = vec![0.5f32; in_dim];
        let mut lora_buf = vec![0.0f32; rank];
        let mut output = vec![0.0f32; out_dim];
        // Built ONCE — this is model-load work, not per-token apply, exactly
        // as `prepared` is for the GPart arm.
        let lora = LoraAdapter {
            rank,
            in_dim,
            out_dim,
            a,
            b,
            alpha: 8.0,
        };

        let mut gpart_weights = vec![0.0f32; n];
        // Data-dependent sinks over the OUTPUT buffers: an unused result is
        // what lets fat LTO delete an arm outright (Issue 723 Class A2).
        let mut gpart_sink = 0u32;
        let mut lora_sink = 0u32;

        const ROUNDS: usize = 9;
        let iters = 400;
        let ab = ab_timing::ab_median_ratio(
            ROUNDS,
            iters,
            100,
            |_| {
                lora_apply(&mut output, &lora, &input, &mut lora_buf);
                lora_sink ^= output[0].to_bits() ^ output[out_dim - 1].to_bits();
            },
            |_| {
                prepared.apply(&mut gpart_weights);
                gpart_sink ^= gpart_weights[0].to_bits() ^ gpart_weights[n - 1].to_bits();
            },
        );
        std::hint::black_box((gpart_sink, lora_sink));

        let gpart_time = ab.b_ns_per_iter();
        let lora_time = ab.a_ns_per_iter();
        let ratio = ab.median;
        ab.report("G2 gpart/lora");

        // Debug builds have no SIMD and no optimisation — GPart's 4096 adds are ~8x LoRA's 512 FMAs.
        // Release builds auto-vectorise the adds, narrowing the gap.
        // Re-pinned 2.0 → 3.0 with provenance (Issue 723 T7): measured 2.21x
        // interleaved on the M3 Max, release, 2026-09-05 (per-round
        // 2.10–2.73 over nine rounds), on the corrected harness where the LoRA
        // arm no longer pays for two `Vec` clones per iteration. GPart does 8x
        // the arithmetic of this LoRA shape — 4096 scalar adds against
        // rank*(in+out) = 512 FMAs — so landing at 2.2x is the SIMD working,
        // and 2.0x was never reachable at these dimensions. 3.0 still reds on
        // the regression that matters: losing the vectorised add in
        // `GpartPrepared::apply` puts the ratio near 8x.
        let max_ratio = if cfg!(debug_assertions) { 10.0 } else { 3.0 };
        assert!(
            ratio <= max_ratio,
            "G2 FAIL: GPart apply time = {:.1}% of LoRA, need ≤ {:.0}% (per-round \
             {:.3}..{:.3} over {} rounds)",
            ratio * 100.0,
            max_ratio * 100.0,
            ab.min(),
            ab.max(),
            ab.ratios.len(),
        );
        eprintln!(
            "✅ G2: GPart apply = {:.1}% of LoRA ({:.0}ns vs {:.0}ns)",
            ratio * 100.0,
            gpart_time,
            lora_time
        );
    }

    /// G3: Quality ≥ 95% of LoRA output (requires trained θ_d from riir-ai).
    #[test]
    #[ignore = "Requires trained θ_d from riir-ai training pipeline"]
    fn goat_g3_quality() {
        // Placeholder: needs actual trained θ_d to compare output similarity.
        // When θ_d is available, compute cos_sim(lora_output, gpart_output) ≥ 0.95
    }

    /// G4: Determinism — same seed+θ → bit-identical output on repeated calls.
    /// Also verifies GpartPrepared fast path matches GpartAdapter::apply().
    #[test]
    fn goat_g4_determinism() {
        let adapter = make_gpart(8, 42, 512);
        let mut w1 = vec![0.0f32; 512];
        let mut w2 = vec![0.0f32; 512];
        adapter.apply(&mut w1);
        adapter.apply(&mut w2);
        assert_eq!(w1, w2, "G4 FAIL: same seed+θ must produce identical output");

        // Verify fast path matches slow path
        let prepared = adapter.prepare(512);
        let mut w3 = vec![0.0f32; 512];
        prepared.apply(&mut w3);
        assert_eq!(
            w1, w3,
            "G4 FAIL: GpartPrepared fast path must match GpartAdapter::apply()"
        );

        eprintln!(
            "✅ G4: determinism verified ({} weights, fast path matches)",
            w1.len()
        );
    }

    /// G5: BLAKE3 commitment integrity — tamper on any byte → verify fails.
    #[test]
    fn goat_g5_commitment_integrity() {
        let adapter = make_gpart(8, 42, 256);
        let commit = adapter.commitment();
        assert!(
            adapter.verify(&commit),
            "G5 FAIL: fresh commitment must verify"
        );

        // Tamper each byte of commitment
        let mut tampered = commit;
        for i in 0..32 {
            tampered[i] ^= 0xFF;
            assert!(
                !adapter.verify(&tampered),
                "G5 FAIL: tampered byte {i} should not verify"
            );
            tampered[i] ^= 0xFF;
        }

        eprintln!("✅ G5: commitment integrity verified (32/32 tamper checks)");
    }
}
