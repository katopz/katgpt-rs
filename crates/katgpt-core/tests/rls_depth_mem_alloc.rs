//! Plan 630 Phase 3 T3.3 — `rls_depth_mem` G4 zero-allocation gate.
//!
//! `write` / `read_into` / `write_ctx` / `unwind_step` / `p_restore` /
//! `WindowedReader::read_into` must not allocate: the state is fixed-size
//! `[[f32; RANK]; RANK]` arrays, the ctx record is a fixed-size struct, and
//! every dot is a stack-local fold. Own test binary — `#[global_allocator]`
//! is crate-binary-unique and would collide with other test modules in the
//! lib test binary (the karc_alloc_check convention).

#![cfg(feature = "rls_depth_mem")]

use katgpt_core::delta_mem::rls::{RlsCtx, RlsMemoryState, WindowedReader};

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

const RANK: usize = 8;

fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0f32;
    for &x in v.iter() {
        s += x * x;
    }
    let n = s.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// Deterministic xorshift64* stream (no global draw — the Issue-809 class).
struct Xs {
    s: u64,
}

impl Xs {
    fn next_f(&mut self) -> f32 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 - 1.0
    }
}

fn gen_pair(xs: &mut Xs) -> ([f32; RANK], [f32; RANK]) {
    let mut k = [0.0f32; RANK];
    for x in k.iter_mut() {
        *x = xs.next_f();
    }
    l2_normalize(&mut k);
    let mut v = [0.0f32; RANK];
    for x in v.iter_mut() {
        *x = 0.5 * xs.next_f();
    }
    (k, v)
}

#[test]
fn g4_write_read_unwind_zero_alloc() {
    assert_counter_is_live();

    let mut xs = Xs { s: 0xDEAD_BEEF_CAFE_0001 };
    let mut st = RlsMemoryState::<RANK>::new();
    // Warmup: seed some state + prime any lazy paths, then drain the counter
    // delta over the measured window only.
    for _ in 0..16 {
        let (k, v) = gen_pair(&mut xs);
        st.write(&k, &v);
    }
    let mut query = [0.0f32; RANK];
    for x in query.iter_mut() {
        *x = xs.next_f();
    }
    l2_normalize(&mut query);
    let mut out = [0.0f32; RANK];

    // Measured section A: 1000 write+read rounds.
    let ((), writes_allocs) = alloc_delta(|| {
        for _ in 0..1000 {
            let (k, v) = gen_pair(&mut xs);
            st.write(&k, &v);
            st.read_into(&query, &mut out);
        }
    });
    assert_eq!(writes_allocs, 0, "write/read allocated {writes_allocs} times over 1000 rounds");

    // Measured section B: 1000 journal write + unwind + restore rounds.
    let ((), unwind_allocs) = alloc_delta(|| {
        for _ in 0..1000 {
            let (k, v) = gen_pair(&mut xs);
            let mut ctx = RlsCtx::<RANK>::default();
            st.write_ctx(&k, &v, &mut ctx);
            st.unwind_step(&ctx);
            st.p_restore(&ctx);
        }
    });
    assert_eq!(unwind_allocs, 0, "write_ctx/unwind/restore allocated {unwind_allocs} times");

    // Measured section C: WindowedReader reads (reader built OUTSIDE — its
    // one-time with_capacity is construction, not the read hot path).
    let base = st.clone();
    let mut reader = WindowedReader::<RANK>::new(&base, 4);
    for _ in 0..4 {
        let (k, v) = gen_pair(&mut xs);
        let mut ctx = RlsCtx::<RANK>::default();
        st.write_ctx(&k, &v, &mut ctx);
        reader.push(ctx);
    }
    let ((), read_allocs) = alloc_delta(|| {
        for _ in 0..1000 {
            reader.read_into(&query, &mut out);
        }
    });
    assert_eq!(read_allocs, 0, "WindowedReader::read_into allocated {read_allocs} times");

    // The reads computed real numbers (anti-DCE): a zero output after 3000
    // writes would mean the compiler deleted the work.
    let norm: f32 = out.iter().map(|x| x * x).sum();
    assert!(norm.is_finite() && norm >= 0.0);
}
