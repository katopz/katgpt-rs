//! Cross-Resolution Spectral Transport — G4 zero-allocation steady-state gate
//! (Plan 310 T2.4).
//!
//! Pattern mirrors `tests/funcattn_g5_zero_alloc.rs`:
//! - Debug-only: `katgpt_rs::alloc::TrackingAllocator` is the global allocator
//!   only under `#[cfg(debug_assertions)]`. The thread-local counters give
//!   per-test isolation.
//! - Release: just exercises the hot path with a timing sanity check (no
//!   allocation audit possible — TrackingAllocator is debug-only).
//!
//! ## Run
//!
//! ```bash
//! # Debug — formal allocation audit.
//! cargo test --features cross_resolution_transport \
//!   --test cross_res_g4_zero_alloc -- --nocapture
//!
//! # Release — timing sanity.
//! cargo test --features cross_resolution_transport --release \
//!   --test cross_res_g4_zero_alloc -- --nocapture
//! ```

#![cfg(feature = "cross_resolution_transport")]

use katgpt_core::cross_resolution::{
    CrossResolutionBases, CrossResScratch, transport_cross_resolution_into,
};

const D_SRC: usize = 64;
const D_DST: usize = 256;
const K: usize = 16;

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }
    fn next_f32(&mut self) -> f32 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        let bits = (self.s >> 11) as u32;
        let u01 = bits as f32 / u32::MAX as f32;
        u01 * 2.0 - 1.0
    }
}

fn random_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.next_f32()).collect()
}

#[test]
fn g4_zero_alloc_steady_state() {
    let phi_src = random_vec(0xC400_0001u64, D_SRC * K);
    let psi_dst = random_vec(0xC400_0002u64, D_DST * K);
    let bases = CrossResolutionBases::new(phi_src, psi_dst, D_SRC, D_DST, K)
        .expect("bases should construct");
    let src = random_vec(0xC400_0003u64, D_SRC);
    let mut dst = vec![0.0f32; D_DST];
    let mut scratch = CrossResScratch::new(K);

    // ── Debug: allocation audit ───────────────────────────────────────────
    #[cfg(debug_assertions)]
    {
        // Warmup — absorbs any one-time `ensure_capacity` resize.
        for _ in 0..50 {
            transport_cross_resolution_into(&src, &bases, &mut scratch, &mut dst);
        }
        std::hint::black_box(&dst);

        katgpt_rs::alloc::reset_alloc_stats();
        const MEASURED_ITERS: usize = 1000;
        for _ in 0..MEASURED_ITERS {
            transport_cross_resolution_into(&src, &bases, &mut scratch, &mut dst);
        }
        std::hint::black_box(&dst);

        let (count, bytes) = katgpt_rs::alloc::get_alloc_stats();
        println!(
            "G4 cross_res: {count} allocations, {bytes} bytes over {MEASURED_ITERS} transports \
             (d_src={D_SRC}, d_dst={D_DST}, k={K})"
        );
        assert!(
            count == 0,
            "G4 FAIL: transport_cross_resolution_into allocated {count} times ({bytes} bytes) \
             over {MEASURED_ITERS} calls. Expected zero — the hot path must be allocation-free \
             after warmup (all scratch is pre-allocated, ensure_capacity is a no-op when cached \
             dimensions match)."
        );
        assert!(bytes == 0);
        println!("G4 PASS: zero allocations on the steady-state hot path.");
    }

    // ── Release: timing sanity (TrackingAllocator is debug-only) ──────────
    #[cfg(not(debug_assertions))]
    {
        use std::time::Instant;
        for _ in 0..50 {
            transport_cross_resolution_into(&src, &bases, &mut scratch, &mut dst);
        }
        std::hint::black_box(&dst);

        const N: usize = 100_000;
        let t0 = Instant::now();
        for _ in 0..N {
            transport_cross_resolution_into(&src, &bases, &mut scratch, &mut dst);
        }
        let elapsed = t0.elapsed();
        std::hint::black_box(&dst);
        let ns_per = elapsed.as_nanos() as f64 / N as f64;
        println!(
            "G4 release timing: {ns_per:.1} ns/transport (d_src={D_SRC}, d_dst={D_DST}, k={K}); \
             {N} iters in {elapsed:?}"
        );
        // Sanity: should be well under 1ms even on a slow build. We don't
        // assert a hard gate here — the formal perf gate would be Phase 3
        // (SIMD). This is just "the transport isn't pathologically slow".
        assert!(
            ns_per < 100_000.0,
            "G4 timing FAIL: {ns_per:.1} ns/transport is pathologically slow — investigate"
        );
        println!("G4 release timing OK: {ns_per:.1} ns/transport.");
    }
}
