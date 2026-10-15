//! G2 bench — the ordered-fold cost at HLA decode-loop scale (Issue 940, HL-1/2/3).
//!
//! ```bash
//! cargo bench -p katgpt-hla --bench bench_939_hla_ordered_dot_g2
//! ```
//!
//! The fix unit swaps the state-feeding dots in `hla_state_update` /
//! `hla_per_head_update` / `ahla_step` / `ahla_per_head_step` from
//! `simd_dot_f32` (platform lane order, different bits per arch) to
//! `dot_f32_ordered` (sequential fold) because they feed `h`/`n`/`mQ` —
//! state that lands in `hla_moments`, the Cold-tier Pod BLAKE3-committed on
//! freeze. This bench prices the swap:
//!
//! - **raw-dot arms**: `simd_dot_f32` vs `dot_f32_ordered` alone at
//!   hd ∈ {64, 128, 256} — the per-call delta the fix introduces;
//! - **kernel arms**: `hla_state_update` / `ahla_step` at hd=128 — the
//!   whole-step cost. Run BEFORE the landing this prints the platform
//!   posture; run AFTER it prints the ordered posture; the two runs are the
//!   A/B (the bench code does not change);
//! - **layer arm**: 32 heads × `hla_state_update` at hd=128 — one decode
//!   token's per-layer symmetric-HLA update (decode-loop scale).
//!
//! G2 is REPORTED, not barred (the bench_844 convention): best-of-N minimum
//! ns/op, the load-robust statistic. Quote every number beside its box
//! state (power source, concurrent load, commit).

use std::hint::black_box;
use std::time::Instant;

use katgpt_hla::{ahla_step, hla_state_update, AhlaQHeadState, HlaQHeadState};
use katgpt_types::simd::{dot_f32_ordered, simd_dot_f32};

/// Deterministic xorshift — seeded, local, never the global RNG.
struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

const ROUNDS: usize = 5;
const RAW_ITERS: usize = 200_000;
const KERNEL_ITERS: usize = 20_000;
const LAYER_ITERS: usize = 4_000;
const N_HEAD: usize = 32;

fn best_of_ns(rounds: usize, iters: usize, mut op: impl FnMut(usize)) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        for i in 0..iters {
            op(i);
        }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        best = best.min(ns);
    }
    best
}

fn vec_of(rng: &mut Xorshift, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32()).collect()
}

fn main() {
    println!("bench_939_hla_ordered_dot_g2 — G2 report (ns/op, best of {ROUNDS})");

    // ── Raw dot arms: the per-call delta the fix introduces ──
    for &hd in &[64usize, 128, 256] {
        let mut rng = Xorshift(0x9400 + hd as u64);
        let a = vec_of(&mut rng, hd);
        let b = vec_of(&mut rng, hd);

        let mut acc_simd = 0.0f32;
        let ns_simd = best_of_ns(ROUNDS, RAW_ITERS, |_| {
            acc_simd += simd_dot_f32(&a, &b, hd);
        });
        black_box(acc_simd);

        let mut acc_ord = 0.0f32;
        let ns_ord = best_of_ns(ROUNDS, RAW_ITERS, |_| {
            acc_ord += dot_f32_ordered(&a, &b);
        });
        black_box(acc_ord);

        println!(
            "  raw dot      hd={hd:<4} simd {ns_simd:8.1}  ordered {ns_ord:8.1}  ratio {:.2}x",
            ns_ord / ns_simd
        );
    }

    // ── Kernel arm: hla_state_update at hd=128 ──
    let hd = 128usize;
    {
        let mut rng = Xorshift(0x9401);
        let q = vec_of(&mut rng, hd);
        let k = vec_of(&mut rng, hd);
        let v = vec_of(&mut rng, hd);
        let mut sk = vec![0.0f32; hd * hd];
        let mut q_head = HlaQHeadState::new(hd);
        let mut tmp_k_cqv = vec![0.0f32; hd];
        let gamma = 1.0f32;

        let ns = best_of_ns(ROUNDS, KERNEL_ITERS, |_| {
            hla_state_update(
                black_box(&mut sk),
                &mut q_head,
                &q,
                &k,
                &v,
                hd,
                gamma,
                &mut tmp_k_cqv,
            );
        });
        black_box((&sk, &q_head));
        println!("  hla_state_update hd={hd}      {ns:8.1}");
    }

    // ── Kernel arm: ahla_step at hd=128 ──
    {
        let mut rng = Xorshift(0x9402);
        let q = vec_of(&mut rng, hd);
        let k = vec_of(&mut rng, hd);
        let v = vec_of(&mut rng, hd);
        let mut pkv = vec![0.0f32; hd * hd];
        let mut mk = vec![0.0f32; hd];
        let mut q_head = AhlaQHeadState::new(hd);
        let mut out = vec![0.0f32; hd];
        let mut tmp_r = vec![0.0f32; hd];
        let gamma = 1.0f32;

        let ns = best_of_ns(ROUNDS, KERNEL_ITERS, |_| {
            ahla_step(
                black_box(&mut pkv),
                &mut mk,
                &mut q_head,
                &q,
                &k,
                &v,
                hd,
                gamma,
                &mut out,
                &mut tmp_r,
            );
        });
        black_box((&pkv, &mk, &q_head, &out));
        println!("  ahla_step        hd={hd}      {ns:8.1}");
    }

    // ── Layer arm: 32 heads × hla_state_update at hd=128 (decode-loop scale) ──
    {
        let mut rng = Xorshift(0x9403);
        let q = vec_of(&mut rng, N_HEAD * hd);
        let k = vec_of(&mut rng, hd);
        let v = vec_of(&mut rng, hd);
        let mut sk = vec![0.0f32; hd * hd];
        let mut heads: Vec<HlaQHeadState> = (0..N_HEAD).map(|_| HlaQHeadState::new(hd)).collect();
        let mut tmp_k_cqv = vec![0.0f32; hd];
        let gamma = 1.0f32;

        let ns = best_of_ns(ROUNDS, LAYER_ITERS, |_| {
            for h in heads.iter_mut() {
                hla_state_update(
                    black_box(&mut sk),
                    h,
                    &q,
                    &k,
                    &v,
                    hd,
                    gamma,
                    &mut tmp_k_cqv,
                );
            }
        });
        black_box((&sk, &heads));
        println!("  layer (32 heads x hla_state_update)  {ns:8.1}  (per token per layer)");
    }
}
