//! Issue 1238 G4 — `branch_parallel_forward` allocates nothing once its
//! scratch is sized (K=2 with and without a shared basis, both fusions).
//!
//! Own test binary: a global counting allocator cannot share a binary with
//! tests that allocate in parallel (the `bench_612_alloc_check` precedent).

#![cfg(feature = "funcattn_branch_parallel")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use katgpt_attn::funcattn_compose::branch_parallel::{
    BranchFusion, BranchParallelConfig, BranchParallelScratch, BranchWeights,
    branch_parallel_forward,
};
use katgpt_core::funcattn::FuncAttnBasis;

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn ramp(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + salt * 11) % 29) as f32 / 29.0) - 0.5)
        .collect()
}

fn eye(d: usize) -> Vec<f32> {
    let mut w = vec![0.0; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    w
}

#[test]
fn g4_branch_parallel_forward_is_alloc_free() {
    const N: usize = 64;
    const M: usize = 8;
    const D: usize = 5;
    const K: usize = 8;
    let x = ramp(N * M, 1);
    let (p0, p1, pb) = (ramp(D * M, 2), ramp(D * M, 3), ramp(D * M, 4));
    let (wb, id) = (ramp(K * D, 5), eye(D));
    let (r0, r1) = (ramp(M * D, 6), ramp(M * D, 7));
    let b0 = BranchWeights {
        in_proj: &p0,
        basis_in_proj: None,
        w_basis: &wb,
        w_q: &id,
        w_k: &id,
        w_v: &id,
        out_proj: &r0,
    };
    let b1 = BranchWeights {
        in_proj: &p1,
        out_proj: &r1,
        ..b0
    };
    let shared = [
        BranchWeights {
            basis_in_proj: Some(&pb),
            ..b0
        },
        BranchWeights {
            basis_in_proj: Some(&pb),
            ..b1
        },
    ];
    let mut out = vec![0.0f32; N * M];
    for fusion in [BranchFusion::SumInvSqrtK, BranchFusion::Mean] {
        let cfg = BranchParallelConfig {
            d_model: M,
            d_branch: D,
            k: K,
            basis: FuncAttnBasis::Sigmoid,
            alpha: 0.01,
            temperature: 0.1,
            cholesky_jitter: 1e-6,
            fusion,
        };
        let mut s = BranchParallelScratch::new(N, &cfg);
        // warm-up
        branch_parallel_forward(&x, &[b0, b1], &cfg, &mut s, &mut out).unwrap();
        let before = ALLOC_COUNT.load(Ordering::Relaxed);
        for _ in 0..16 {
            branch_parallel_forward(&x, &[b0, b1], &cfg, &mut s, &mut out).unwrap();
            branch_parallel_forward(&x, &shared, &cfg, &mut s, &mut out).unwrap();
        }
        let after = ALLOC_COUNT.load(Ordering::Relaxed);
        assert_eq!(after - before, 0, "G4: hot path allocated ({fusion:?})");
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            out.iter().any(|v| *v != 0.0),
            "loud zero: forward produced nothing"
        );
    }
}
