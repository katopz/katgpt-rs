//! Issue 949 T6/T9 — the `path_kernel_sim` GOAT gate: G2 latency bars over
//! the fingerprint hot path at the POC envelope (p = 64 knobs, 256 anchors)
//! plus the T5c/T5d battery re-run at `--release` (the T9 promotion
//! verdict's measured record).
//!
//! # Gates
//!
//! - **G2 (latency)** — `fingerprint_into` ≤ 5 µs/call and
//!   `similarity` ≤ 1 µs/pair at p=64 (µs-class per-pair bars, the issue's
//!   T6 sizing); `black_box` on args AND receiver so LLVM cannot delete the
//!   measured work (the timed_region law).
//! - **T9 (the promotion verdict)** — the planted-class battery at release:
//!   the fingerprint must beat BOTH the input-cosine and the output-vector
//!   baselines for promotion; a tie is a KEEP-OFF (issue 949 T5d's own law).
//!   The verdict prints on the final line; the feature ships opt-in either
//!   way, and the record lands in .benchmarks + the research note.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features path_kernel_sim \
//!     --bench path_kernel_sim_goat -- --nocapture
//! ```

#![cfg(feature = "path_kernel_sim")]

use katgpt_core::path_kernel_sim::{
    FingerprintKernel, KnobScorer, SensitivityFingerprint, fingerprint_into, similarity,
};
use katgpt_core::simd::{dot_f32_ordered, fast_sigmoid};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── the fixture scorer (the battery's, lifted to the bench) ────────────────

/// Exponent is PURE cross-term — `f = σ(w1·w2·u·v + w3·w4·s·t)` — so ONLY
/// the nonlinear cross-term structure carries any class signal (the issue's
/// T5c validity clause). Closed-form gradient at `w0 = 1`:
/// direction `(uv, uv, st, st)`.
struct CrossGateScorer;

impl KnobScorer for CrossGateScorer {
    fn knobs(&self) -> usize {
        4
    }
    fn eval(&self, x: &[f32], w: &[f32]) -> f32 {
        let (u, v, s, t) = (x[0], x[1], x[2], x[3]);
        fast_sigmoid(w[0] * w[1] * u * v + w[2] * w[3] * s * t)
    }
    fn grad_knobs_closed_form(&self, x: &[f32], w: &[f32], out: &mut [f32]) -> bool {
        let (u, v, s, t) = (x[0], x[1], x[2], x[3]);
        let z = w[0] * w[1] * u * v + w[2] * w[3] * s * t;
        let sgm = fast_sigmoid(z);
        let sp = sgm * (1.0 - sgm);
        out[0] = sp * w[1] * u * v;
        out[1] = sp * w[0] * u * v;
        out[2] = sp * w[3] * s * t;
        out[3] = sp * w[2] * s * t;
        true
    }
}

/// The T5a shape at p = 64 — the G2 latency envelope.
struct SigmoidGate64 {
    phi: [f32; 64],
}

impl SigmoidGate64 {
    fn new(x: &[f32; 64]) -> Self {
        let mut phi = [0.0f32; 64];
        phi.copy_from_slice(x);
        Self { phi }
    }
}

impl KnobScorer for SigmoidGate64 {
    fn knobs(&self) -> usize {
        64
    }
    fn eval(&self, _x: &[f32], w: &[f32]) -> f32 {
        fast_sigmoid(dot_f32_ordered(w, &self.phi))
    }
    fn grad_knobs_closed_form(&self, _x: &[f32], w: &[f32], out: &mut [f32]) -> bool {
        let s = fast_sigmoid(dot_f32_ordered(w, &self.phi));
        let sp = s * (1.0 - s);
        for (o, p) in out.iter_mut().zip(self.phi.iter()) {
            *o = sp * p;
        }
        true
    }
}

/// Deterministic LCG (the unseeded-global-RNG gate).
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f32) / ((1u64 << 53) as f32)
    }
    fn next_signed(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Issue 949 — path_kernel_sim GOAT gate (T6 latency + T9 verdict)");
    println!("═══════════════════════════════════════════════════════════");

    let mut failures = 0u32;

    // ── G2: fingerprint latency at the p=64 envelope ────────────────────
    {
        let kernel = FingerprintKernel::default();
        let mut x = [0.0f32; 64];
        let mut w0 = [0.0f32; 64];
        let mut rng = Lcg(0x949_0064);
        for i in 0..64 {
            x[i] = rng.next_signed();
            w0[i] = rng.next_signed() * 0.5;
        }
        let scorer = SigmoidGate64::new(&x);
        let mut fp = [0.0f32; 64];
        let mut w_scratch = w0;

        // Warm-up.
        for _ in 0..1000 {
            black_box(fingerprint_into(
                black_box(&kernel),
                black_box(&scorer),
                black_box(&x),
                &mut w_scratch,
                &mut fp,
            ));
        }
        const N: usize = 100_000;
        let t0 = Instant::now();
        for _ in 0..N {
            // Args AND receiver black-boxed — the timed_region law.
            let n = fingerprint_into(
                black_box(&kernel),
                black_box(&scorer),
                black_box(&x),
                &mut w_scratch,
                &mut fp,
            );
            black_box(n);
        }
        let per = t0.elapsed().as_nanos() as f64 / N as f64;
        println!("[G2] fingerprint_into p=64: {per:.0} ns/call");
        if per > 5_000.0 {
            eprintln!("[G2] FAIL: fingerprint bar is 5 µs/call");
            failures += 1;
        }

        let fp2 = fp.map(|v| v * 0.9 + 0.01);
        for _ in 0..1000 {
            black_box(similarity(
                black_box(&kernel),
                black_box(&fp),
                black_box(&fp2),
            ));
        }
        let t0 = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..N {
            acc += similarity(black_box(&kernel), black_box(&fp), black_box(&fp2)).unwrap_or(0.0);
        }
        black_box(acc);
        let per_pair = t0.elapsed().as_nanos() as f64 / N as f64;
        println!("[G2] similarity p=64: {per_pair:.0} ns/pair");
        if per_pair > 1_000.0 {
            eprintln!("[G2] FAIL: similarity bar is 1 µs/pair");
            failures += 1;
        }
    }

    // ── T9: the battery at release ──────────────────────────────
    {
        let kernel = FingerprintKernel::default();
        let scorer = CrossGateScorer;
        let w0 = [1.0f32; 4];

        let probes: [[f32; 4]; 4] = [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
        ];

        let n_per_class = 40;
        let mut rng = Lcg(0x949_C1A5);
        let mut xs: Vec<[f32; 4]> = Vec::with_capacity(n_per_class * 2);
        let mut labels: Vec<u8> = Vec::with_capacity(n_per_class * 2);
        for _ in 0..(n_per_class * 2) {
            let z = (rng.next_unit() < 0.5) as u8;
            let a = 0.8 + 0.4 * rng.next_unit();
            let c = if z == 0 { 1.0 } else { 1.8 } + 0.15 * rng.next_signed();
            let mut j = || 0.08 * rng.next_signed();
            xs.push([
                a * (1.0 + j()),
                a * (1.0 + j()),
                a * (c + j()),
                -a * (c + j()),
            ]);
            labels.push(z);
        }

        let mut fps: Vec<SensitivityFingerprint> = Vec::with_capacity(xs.len());
        let mut outs: Vec<[f32; 4]> = Vec::with_capacity(xs.len());
        for x in &xs {
            let mut grad = vec![0.0f32; 4];
            let mut w_scratch = w0;
            let norm = fingerprint_into(&kernel, &scorer, x, &mut w_scratch, &mut grad);
            fps.push(SensitivityFingerprint { grad, norm });
            let mut o = [0.0f32; 4];
            for (slot, pw) in o.iter_mut().zip(probes.iter()) {
                *slot = scorer.eval(x, pw);
            }
            outs.push(o);
        }

        // kNN(1) by mean-similarity-to-class (nearest-class-centroid in
        // similarity space): classify each anchor by its mean similarity to
        // the same-class vs other-class anchors (leave-one-out).
        let n = xs.len();
        let loocv = |sims: &dyn Fn(usize, usize) -> Option<f32>| -> f32 {
            let mut hits = 0u32;
            for (i, label_i) in labels.iter().enumerate() {
                let mut s0 = (0.0f32, 0u32);
                let mut s1 = (0.0f32, 0u32);
                for (j, label_j) in labels.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let s = sims(i, j).unwrap_or(0.0);
                    if *label_j == 0 {
                        s0.0 += s;
                        s0.1 += 1;
                    } else {
                        s1.0 += s;
                        s1.1 += 1;
                    }
                }
                let m0 = if s0.1 > 0 { s0.0 / s0.1 as f32 } else { 0.0 };
                let m1 = if s1.1 > 0 { s1.0 / s1.1 as f32 } else { 0.0 };
                let guess = if m0 >= m1 { 0u8 } else { 1u8 };
                if guess == *label_i {
                    hits += 1;
                }
            }
            hits as f32 / n as f32
        };

        let acc_fp = loocv(&|i, j| similarity(&kernel, &fps[i].grad, &fps[j].grad));
        let acc_in = loocv(&|i, j| {
            let (a, b) = (&xs[i], &xs[j]);
            let dot: f32 = a.iter().zip(b).map(|(p, q)| p * q).sum();
            let na: f32 = a.iter().map(|p| p * p).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|p| p * p).sum::<f32>().sqrt();
            Some(dot / (na * nb))
        });
        let acc_out = loocv(&|i, j| {
            let (a, b) = (&outs[i], &outs[j]);
            let dot: f32 = a.iter().zip(b).map(|(p, q)| p * q).sum();
            let na: f32 = a.iter().map(|p| p * p).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|p| p * p).sum::<f32>().sqrt();
            Some(dot / (na * nb))
        });

        println!(
            "[T9] battery (release) — fingerprint {acc_fp:.3} · input-cosine {acc_in:.3} · output-vector {acc_out:.3}"
        );
        let promotes = acc_fp > acc_in && acc_fp > acc_out;
        if promotes {
            println!(
                "[T9] VERDICT: fingerprint beats BOTH baselines — promotion QUESTION opens (owner-gated; consumers unwired)"
            );
        } else {
            println!(
                "[T9] VERDICT: KEEP-OFF — the fingerprint does not strictly beat both baselines (T5d's own law: a tie is a keep-off). The feature stays opt-in, promoted nowhere."
            );
        }
        // The verdict itself is the record — either branch is a valid gate
        // outcome (decline is a correct answer); nothing asserts `promotes`.
        let _ = promotes;
    }

    // ── gate roll-up ────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════");
    if failures == 0 {
        println!("  GOAT G2 latency: PASS · T9 verdict: recorded above");
    } else {
        println!("  {failures} gate(s) FAILED");
        std::process::exit(1);
    }
    println!("═══════════════════════════════════════════════════════════");
}
