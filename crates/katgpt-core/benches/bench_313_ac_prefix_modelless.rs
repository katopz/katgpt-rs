//! AC-GPT Arbitrary-Conditional Prefix — §3.5 Modelless Unblock Investigation
//! (Issue 003 Phase 0, Path 2: raw/lora hot-swap / deterministic mask correction).
//!
//! Tests the central claim of the modelless unblock protocol: can a
//! **deterministically constructed** mask variant (no gradient descent) make
//! AC-GPT single-pass conditional logprob match iterative-MLM conditional
//! logprob on an untrained micro-GPT?
//!
//! # The doubled-signal bias
//!
//! The original AC-GPT mask (`AcPrefix::attends`) lets an eval token at
//! original position `k` attend to an in-place `xc` token at original position
//! `p <= k` **twice**: once via its r0 copy, once via its r1 in-place slot. On
//! an untrained model both appearances contribute real signal, biasing the
//! conditional likelihood by ~7.5e-4 vs iterative-MLM.
//!
//! # The modelless fix (Path 2)
//!
//! `AcPrefix::attends_dedup` zeroes eval→in-place-xc attention, forcing all
//! conditioning through r0 copies. On a single-layer model this makes the
//! attended (token, original_position) set identical to iterative-MLM's →
//! same K/V → same softmax → same logprobs. The fix is a pure attention-pattern
//! modification (no weights, no training).
//!
//! # Gates
//!
//! - **G1-modelless (correctness):** `|dedup_logprob - iterative_logprob| < 1e-4`.
//!   Expected to be ~0.0 on single-layer (bit-identical attended sets).
//! - **G1-bias (negative control):** `|original_logprob - iterative_logprob| > 1e-4`.
//!   Confirms the original mask IS biased (the known 7.5e-4 mismatch).
//! - **G1-dedup-vs-original:** `|dedup_logprob - original_logprob| > 0`.
//!   Confirms the correction actually changes the output.
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features ac_prefix --bench bench_313_ac_prefix_modelless --release -- --nocapture
//! ```

#![cfg(feature = "ac_prefix")]

use katgpt_core::ac_prefix::AcPrefix;
use katgpt_core::{Config, matmul, matmul_relu, rmsnorm, softmax};

// ─── Micro-GPT (identical to bench_313_ac_prefix_goat.rs) ───────────────────

struct MicroGpt {
    wte: Vec<f32>,
    lm_head: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_fc: Vec<f32>,
    w_proj: Vec<f32>,
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    vocab: usize,
    mlp_hidden: usize,
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f32 {
        let bits = ((self.next() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }
}

fn rand_vec(n: usize, rng: &mut SimpleRng, scale: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.uniform() * 2.0 - 1.0) * scale).collect()
}

impl MicroGpt {
    fn new(cfg: &Config, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let n_embd = cfg.n_embd;
        let vocab = cfg.vocab_size;
        let mlp_hidden = cfg.mlp_hidden;
        Self {
            wte: rand_vec(vocab * n_embd, &mut rng, 0.05),
            lm_head: rand_vec(vocab * n_embd, &mut rng, 0.05),
            w_q: rand_vec(n_embd * n_embd, &mut rng, 0.1),
            w_k: rand_vec(n_embd * n_embd, &mut rng, 0.1),
            w_v: rand_vec(n_embd * n_embd, &mut rng, 0.1),
            w_o: rand_vec(n_embd * n_embd, &mut rng, 0.1),
            w_fc: rand_vec(mlp_hidden * n_embd, &mut rng, 0.1),
            w_proj: rand_vec(n_embd * mlp_hidden, &mut rng, 0.1),
            n_embd,
            n_head: cfg.n_head,
            head_dim: cfg.head_dim,
            vocab,
            mlp_hidden,
        }
    }
}

/// General masked forward (identical to bench_313_ac_prefix_goat.rs).
fn forward_masked(
    model: &MicroGpt,
    tokens: &[u32],
    positions: &[usize],
    attends_fn: &dyn Fn(usize, usize) -> bool,
) -> Vec<f32> {
    let seq = tokens.len();
    let n_embd = model.n_embd;
    let n_head = model.n_head;
    let head_dim = model.head_dim;
    let vocab = model.vocab;

    let mut hidden = vec![0.0f32; seq * n_embd];
    for i in 0..seq {
        let tok = tokens[i] as usize;
        for d in 0..n_embd {
            hidden[i * n_embd + d] = model.wte[tok * n_embd + d];
        }
    }
    let mut q_all = vec![0.0f32; seq * n_embd];
    let mut k_all = vec![0.0f32; seq * n_embd];
    let mut v_all = vec![0.0f32; seq * n_embd];
    for i in 0..seq {
        let h_in = &hidden[i * n_embd..(i + 1) * n_embd];
        matmul(&mut q_all[i * n_embd..(i + 1) * n_embd], &model.w_q, h_in, n_embd, n_embd);
        matmul(&mut k_all[i * n_embd..(i + 1) * n_embd], &model.w_k, h_in, n_embd, n_embd);
        matmul(&mut v_all[i * n_embd..(i + 1) * n_embd], &model.w_v, h_in, n_embd, n_embd);
    }
    let mut attn_out = vec![0.0f32; seq * n_embd];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let pos_phase = |pi: usize, pj: usize| -> f32 {
        ((pi.max(pj) - pi.min(pj)) as f32 * 0.1).cos()
    };
    for i in 0..seq {
        for h in 0..n_head {
            let off = h * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq];
            let mut max_score = f32::NEG_INFINITY;
            let pi = positions[i];
            for j in 0..seq {
                if !attends_fn(i, j) {
                    continue;
                }
                let pj = positions[j];
                let phase = pos_phase(pi, pj);
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q_all[i * n_embd + off + d] * k_all[j * n_embd + off + d];
                }
                s = s * scale * phase;
                scores[j] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut().take(seq) {
                if s.is_finite() {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                } else {
                    *s = 0.0;
                }
            }
            let inv = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq {
                    if scores[j] > 0.0 {
                        acc += scores[j] * v_all[j * n_embd + off + d];
                    }
                }
                attn_out[i * n_embd + off + d] = acc * inv;
            }
        }
    }
    for i in 0..seq {
        let mut o = vec![0.0f32; n_embd];
        matmul(&mut o, &model.w_o, &attn_out[i * n_embd..(i + 1) * n_embd], n_embd, n_embd);
        for d in 0..n_embd {
            hidden[i * n_embd + d] += o[d];
        }
        rmsnorm(&mut hidden[i * n_embd..(i + 1) * n_embd]);
    }
    let mut hidden2 = vec![0.0f32; seq * n_embd];
    for i in 0..seq {
        let mut h = vec![0.0f32; model.mlp_hidden];
        matmul_relu(
            &mut h,
            &model.w_fc,
            &hidden[i * n_embd..(i + 1) * n_embd],
            model.mlp_hidden,
            n_embd,
        );
        let mut out = vec![0.0f32; n_embd];
        matmul(&mut out, &model.w_proj, &h, n_embd, model.mlp_hidden);
        for d in 0..n_embd {
            hidden2[i * n_embd + d] = hidden[i * n_embd + d] + out[d];
        }
        rmsnorm(&mut hidden2[i * n_embd..(i + 1) * n_embd]);
    }
    let mut logprobs = vec![0.0f32; seq];
    for i in 0..seq {
        let mut logits = vec![0.0f32; vocab];
        matmul(
            &mut logits,
            &model.lm_head,
            &hidden2[i * n_embd..(i + 1) * n_embd],
            vocab,
            n_embd,
        );
        softmax(&mut logits);
        let actual = tokens[i] as usize;
        logprobs[i] = logits[actual].max(1e-30).ln();
    }
    logprobs
}

// ─── Phase 0 modelless unblock: three-way comparison ────────────────────────

/// Compute AC-GPT conditional logprob using the ORIGINAL mask (doubled signal).
fn ac_gpt_original_logprob(
    model: &MicroGpt,
    base_tokens: &[u32],
    xc_positions: &[usize],
) -> f32 {
    let prefix = AcPrefix::new(base_tokens, xc_positions);
    prefix.conditional_logprob(|tokens, positions, mask, _loss_mask| {
        let n = tokens.len();
        forward_masked(model, tokens, positions, &|i, j| mask.get(i, j, n))
    })
}

/// Compute AC-GPT conditional logprob using the DEDUPLICATED mask (modelless fix).
fn ac_gpt_dedup_logprob(
    model: &MicroGpt,
    base_tokens: &[u32],
    xc_positions: &[usize],
) -> f32 {
    let prefix = AcPrefix::new(base_tokens, xc_positions);
    prefix.conditional_logprob_dedup(|tokens, positions, mask, _loss_mask| {
        let n = tokens.len();
        forward_masked(model, tokens, positions, &|i, j| mask.get(i, j, n))
    })
}

/// Compute iterative-MLM conditional logprob (the reference).
///
/// For each eval position `p` (left to right), forward the full original
/// sequence where position `p` attends to all xc (any position) + positions
/// <= p. Read the logprob at position p. Sum over all eval positions.
/// This is `|xe|` forward passes.
fn iterative_mlm_logprob(
    model: &MicroGpt,
    base_tokens: &[u32],
    xc_positions: &[usize],
) -> f32 {
    let positions: Vec<usize> = (0..base_tokens.len()).collect();
    let xc_set = xc_positions;
    let mut total = 0.0f32;
    for p in 0..base_tokens.len() {
        if xc_set.binary_search(&p).is_ok() {
            continue; // skip conditioning positions
        }
        let logprobs = forward_masked(model, base_tokens, &positions, &|i, j| {
            if i != p {
                return j <= i; // standard causal for non-target positions
            }
            // Eval position p: attend to all xc (any position) + positions <= p.
            xc_set.binary_search(&j).is_ok() || j <= p
        });
        total += logprobs[p];
    }
    total
}

fn run_phase0() {
    let cfg = Config::micro();
    let model = MicroGpt::new(&cfg, 0xC0FFEE);

    // 32-token base, 16 conditioning (every other).
    let base_tokens: Vec<u32> = (0..32)
        .map(|i| (i * 7 + 3) as u32 % cfg.vocab_size as u32)
        .collect();
    let xc_positions: Vec<usize> = (0..32).filter(|i| i % 2 == 0).collect();

    let original = ac_gpt_original_logprob(&model, &base_tokens, &xc_positions);
    let dedup = ac_gpt_dedup_logprob(&model, &base_tokens, &xc_positions);
    let iterative = iterative_mlm_logprob(&model, &base_tokens, &xc_positions);

    let diff_original = (original - iterative).abs();
    let diff_dedup = (dedup - iterative).abs();
    let diff_dedup_vs_original = (dedup - original).abs();

    println!("── §3.5 Modelless Unblock Investigation (Issue 003 Phase 0, Path 2) ──");
    println!("   Config: 32-token base, 16 xc (every other), single-layer micro-GPT");
    println!();
    println!("   AC-GPT original logprob:    {original:.6}");
    println!("   AC-GPT deduplicated logprob:{dedup:.6}");
    println!("   Iterative-MLM logprob:      {iterative:.6}");
    println!();
    println!("── G1-bias (negative control): original should NOT match iterative ──");
    println!("   |original - iterative|:     {diff_original:.6}");
    println!("   Threshold:                  > 1e-4 (confirms the doubled-signal bias)");
    println!("   Result:                     {}", if diff_original > 1e-4 { "CONFIRMED ✓ (bias present)" } else { "UNEXPECTED (no bias detected)" });
    println!();
    println!("── G1-modelless (correctness): dedup should match iterative ──");
    println!("   |dedup - iterative|:        {diff_dedup:.6}");
    println!("   Threshold:                  < 1e-4 (modelless correction eliminates bias)");
    println!("   Result:                     {}", if diff_dedup < 1e-4 { "PASS ✓ (MODELLESS-VALIDABLE)" } else { "FAIL ✗" });
    println!();
    println!("── G1-dedup-vs-original: correction must change output ──");
    println!("   |dedup - original|:         {diff_dedup_vs_original:.6}");
    println!("   Threshold:                  > 0.0 (correction is non-trivial)");
    println!("   Result:                     {}", if diff_dedup_vs_original > 0.0 { "CONFIRMED ✓" } else { "UNEXPECTED (no change)" });
    println!();
    println!("═══ Phase 0 Path 2 verdict ─══");
    if diff_dedup < 1e-4 && diff_original > 1e-4 {
        println!("   ✓ MODELLESS-VALIDABLE: the deduplicated mask eliminates the");
        println!("     doubled-signal bias on single-layer micro-GPT without gradient");
        println!("     descent. Per §3.5, this unblocks G1 modellessly.");
        println!("   → ACTION: re-promote `ac_prefix` to default-on with deduplicated");
        println!("     mask as the recommended default (or document as alternative).");
        println!("   → CAVEAT: multi-layer equivalence (r0 copy representation divergence)");
        println!("     remains a riir-train question, documented in Issue 003.");
    } else if diff_dedup < 1e-4 && diff_original <= 1e-4 {
        println!("   ⚠ UNEXPECTED: both original and dedup match iterative. The bias");
        println!("     may not manifest at this scale or config. Try larger configs.");
    } else {
        println!("   ✗ FAIL: deduplicated mask does NOT match iterative-MLM on");
        println!("     single-layer. The doubling is not the sole source of divergence.");
        println!("   → ACTION: check Path 3 (latent correction) or document genuine");
        println!("     riir-train dependency per §3.5.");
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 003 Phase 0 — AC-Prefix §3.5 Modelless Unblock (Path 2)  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    run_phase0();
}
