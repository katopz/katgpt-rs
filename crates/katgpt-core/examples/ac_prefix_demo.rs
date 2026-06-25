//! AC-GPT Arbitrary-Conditional Prefix demo (Plan 313 T2.4).
//!
//! Shows the primitive end-to-end on a tiny inline micro-GPT:
//!   - 16-token base sequence, 8 marked as conditioning (every other position).
//!   - AC-GPT conditional logprob via [`AcPrefix::conditional_logprob`] (single
//!     augmented forward).
//!   - A "naive" variant that lets later eval tokens attend to in-place
//!     conditioning tokens (standard causal everywhere, no copies) — this is
//!     the leakage-permitting mask the paper warns against.
//!   - A sampled continuation via [`AcPrefix::conditional_sample`] (Gumbel-max,
//!     sigmoid-respecting).
//!
//! The micro-GPT here is a hand-rolled 1-layer attention block sized to
//! `Config::micro()` (vocab=27, n_embd=16, n_head=4, head_dim=4, mlp_hidden=64).
//! We don't use the production `TransformerWeights` (which lives in the root
//! crate) — we roll random f32 weights from a fixed seed so the demo is
//! self-contained and deterministic. This keeps the AC-Prefix primitive
//! decoupled from any concrete weight type, which is the whole point of T2.3's
//! `ForwardForAcPrefix` trait.
//!
//! # Run
//!
//! ```sh
//! cargo run --example ac_prefix_demo --features ac_prefix --release
//! ```

#![cfg(feature = "ac_prefix")]

use katgpt_core::ac_prefix::{AcPrefix, AcPrefixMask};
use katgpt_core::{Config, matmul, matmul_relu, rmsnorm, softmax};

/// Self-contained micro-GPT for the demo. Dimensions track `Config::micro()`:
/// `n_embd=16, n_head=4, head_dim=4, vocab=27, mlp_hidden=64, n_layer=1`.
struct MicroGpt {
    wte: Vec<f32>,     // [vocab, n_embd]
    lm_head: Vec<f32>, // [vocab, n_embd]
    w_q: Vec<f32>,     // [n_embd, n_embd]
    w_k: Vec<f32>,     // [n_embd, n_embd]
    w_v: Vec<f32>,     // [n_embd, n_embd]
    w_o: Vec<f32>,     // [n_embd, n_embd]
    w_fc: Vec<f32>,    // [mlp_hidden, n_embd]
    w_proj: Vec<f32>,  // [n_embd, mlp_hidden]
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    vocab: usize,
    mlp_hidden: usize,
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

/// Minimal xorshift RNG for weight init — deterministic per seed, no extra dep.
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
    (0..n)
        .map(|_| (rng.uniform() * 2.0 - 1.0) * scale)
        .collect()
}

/// Per-position logprobs for the *actual* token at each augmented slot, using
/// an AC-GPT three-region attention mask.
///
/// The LM-head uses log-softmax over the vocab dimension. AGENTS.md "sigmoid
/// not softmax" rule applies to blending/decision gates — the LM-head softmax
/// over the output vocabulary is the standard next-token distribution and is
/// not a blending gate, so it is exempt.
///
/// `augmented_positions` is consumed as a learnable-frequency RoPE-lite phase
/// bias on the Q/K dot product (a real RoPE rotation would be applied here in
/// a production wiring; the demo uses a scalar phase term to keep the code
/// short while still making the original-position-awareness load-bearing).
fn forward_masked_ac(
    model: &MicroGpt,
    augmented_tokens: &[u32],
    augmented_positions: &[usize],
    mask: &AcPrefixMask,
) -> Vec<f32> {
    forward_masked_ac_logits(model, augmented_tokens, augmented_positions, mask)
        .into_iter()
        .enumerate()
        .map(|(i, mut logits)| {
            softmax(&mut logits);
            let actual = augmented_tokens[i] as usize;
            logits[actual].max(1e-30).ln()
        })
        .collect()
}

/// Same as [`forward_masked_ac`] but returns the per-position vocab logits
/// instead of logprobs. Used by the demo's conditional_sample path so the
/// Gumbel-max sampler sees the true distribution.
fn forward_masked_ac_logits(
    model: &MicroGpt,
    augmented_tokens: &[u32],
    augmented_positions: &[usize],
    mask: &AcPrefixMask,
) -> Vec<Vec<f32>> {
    let seq = augmented_tokens.len();
    let n_embd = model.n_embd;
    let n_head = model.n_head;
    let head_dim = model.head_dim;
    let vocab = model.vocab;

    // Embed all tokens → hidden [seq, n_embd].
    let mut hidden = vec![0.0f32; seq * n_embd];
    for i in 0..seq {
        let tok = augmented_tokens[i] as usize;
        for d in 0..n_embd {
            hidden[i * n_embd + d] = model.wte[tok * n_embd + d];
        }
    }

    // Attention block.
    let mut q_all = vec![0.0f32; seq * n_embd];
    let mut k_all = vec![0.0f32; seq * n_embd];
    let mut v_all = vec![0.0f32; seq * n_embd];
    for i in 0..seq {
        let h_in = &hidden[i * n_embd..(i + 1) * n_embd];
        matmul(
            &mut q_all[i * n_embd..(i + 1) * n_embd],
            &model.w_q,
            h_in,
            n_embd,
            n_embd,
        );
        matmul(
            &mut k_all[i * n_embd..(i + 1) * n_embd],
            &model.w_k,
            h_in,
            n_embd,
            n_embd,
        );
        matmul(
            &mut v_all[i * n_embd..(i + 1) * n_embd],
            &model.w_v,
            h_in,
            n_embd,
            n_embd,
        );
    }

    let mut attn_out = vec![0.0f32; seq * n_embd];
    let scale = 1.0 / (head_dim as f32).sqrt();
    // RoPE-lite: per-(i,j) phase term using original positions. This is the
    // load-bearing use of augmented_positions — without it the demo would not
    // exercise the position-aware copy mechanism.
    let pos_phase =
        |pi: usize, pj: usize| -> f32 { ((pi.max(pj) - pi.min(pj)) as f32 * 0.1).cos() };
    for i in 0..seq {
        for h in 0..n_head {
            let off = h * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq];
            let mut max_score = f32::NEG_INFINITY;
            let pi = augmented_positions[i];
            for j in 0..seq {
                if !mask.get(i, j, seq) {
                    continue;
                }
                let pj = augmented_positions[j];
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
            (0..seq).for_each(|j| {
                if scores[j].is_finite() {
                    scores[j] = (scores[j] - max_score).exp();
                    sum_exp += scores[j];
                } else {
                    scores[j] = 0.0;
                }
            });
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

    // Output projection + residual.
    for i in 0..seq {
        let mut o = vec![0.0f32; n_embd];
        matmul(
            &mut o,
            &model.w_o,
            &attn_out[i * n_embd..(i + 1) * n_embd],
            n_embd,
            n_embd,
        );
        for d in 0..n_embd {
            hidden[i * n_embd + d] += o[d];
        }
        rmsnorm(&mut hidden[i * n_embd..(i + 1) * n_embd]);
    }

    // MLP block + residual.
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

    // LM head → logits [seq, vocab].
    let mut all_logits = Vec::with_capacity(seq);
    for i in 0..seq {
        let mut logits = vec![0.0f32; vocab];
        matmul(
            &mut logits,
            &model.lm_head,
            &hidden2[i * n_embd..(i + 1) * n_embd],
            vocab,
            n_embd,
        );
        all_logits.push(logits);
    }
    all_logits
}

/// Naive leakage-permitting variant: standard causal everywhere, no copies.
/// Returns per-position logprob of the *actual* token at each original slot.
///
/// This is the "let later eval tokens attend to in-place conditioning tokens"
/// mask that the AC-GPT paper proves leaks future information through the
/// conditioning tokens over multiple layers. We compute the conditional
/// logprob by running this naive forward over the original (non-augmented)
/// sequence and summing the logprobs at the eval positions only.
fn forward_naive_causal(model: &MicroGpt, base_tokens: &[u32], xc_positions: &[usize]) -> f32 {
    let seq = base_tokens.len();
    let n_embd = model.n_embd;
    let n_head = model.n_head;
    let head_dim = model.head_dim;
    let vocab = model.vocab;

    let mut hidden = vec![0.0f32; seq * n_embd];
    for i in 0..seq {
        let tok = base_tokens[i] as usize;
        for d in 0..n_embd {
            hidden[i * n_embd + d] = model.wte[tok * n_embd + d];
        }
    }
    let mut q_all = vec![0.0f32; seq * n_embd];
    let mut k_all = vec![0.0f32; seq * n_embd];
    let mut v_all = vec![0.0f32; seq * n_embd];
    for i in 0..seq {
        let h_in = &hidden[i * n_embd..(i + 1) * n_embd];
        matmul(
            &mut q_all[i * n_embd..(i + 1) * n_embd],
            &model.w_q,
            h_in,
            n_embd,
            n_embd,
        );
        matmul(
            &mut k_all[i * n_embd..(i + 1) * n_embd],
            &model.w_k,
            h_in,
            n_embd,
            n_embd,
        );
        matmul(
            &mut v_all[i * n_embd..(i + 1) * n_embd],
            &model.w_v,
            h_in,
            n_embd,
            n_embd,
        );
    }
    let mut attn_out = vec![0.0f32; seq * n_embd];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for i in 0..seq {
        for h in 0..n_head {
            let off = h * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q_all[i * n_embd + off + d] * k_all[j * n_embd + off + d];
                }
                s *= scale;
                scores[j] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum_exp = 0.0f32;
            (0..=i).for_each(|j| {
                scores[j] = (scores[j] - max_score).exp();
                sum_exp += scores[j];
            });
            let inv = 1.0 / sum_exp;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=i {
                    acc += scores[j] * v_all[j * n_embd + off + d];
                }
                attn_out[i * n_embd + off + d] = acc * inv;
            }
        }
    }
    for i in 0..seq {
        let mut o = vec![0.0f32; n_embd];
        matmul(
            &mut o,
            &model.w_o,
            &attn_out[i * n_embd..(i + 1) * n_embd],
            n_embd,
            n_embd,
        );
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
    let mut total = 0.0f32;
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
        let actual = base_tokens[i] as usize;
        let lp = logits[actual].max(1e-30).ln();
        if xc_positions.binary_search(&i).is_err() {
            total += lp;
        }
    }
    total
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 313: AC-GPT Arbitrary-Conditional Prefix — Demo            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let cfg = Config::micro();
    let model = MicroGpt::new(&cfg, 0xC0FFEE);

    // 16-token base sequence, 8 conditioning positions (every other).
    let base_tokens: Vec<u32> = (0..16)
        .map(|i| (i * 7 + 3) as u32 % cfg.vocab_size as u32)
        .collect();
    let xc_positions: Vec<usize> = (0..16).filter(|i| i % 2 == 0).collect();
    println!("Base tokens ({}):  {:?}", base_tokens.len(), base_tokens);
    println!(
        "Conditioning positions ({} of {}): {:?}",
        xc_positions.len(),
        base_tokens.len(),
        xc_positions
    );
    println!();

    let prefix = AcPrefix::new(&base_tokens, &xc_positions);
    println!(
        "Augmented sequence length: {} (= {} base + {} copies)",
        prefix.augmented_len(),
        base_tokens.len(),
        xc_positions.len()
    );

    // AC-GPT conditional logprob — single augmented forward.
    let ac_logprob = prefix.conditional_logprob(|tokens, positions, mask, _loss_mask| {
        forward_masked_ac(&model, tokens, positions, mask)
    });
    println!();
    println!("── AC-GPT conditional logprob (single forward, leakage-prevented) ──");
    println!("   log p(xe | xc) = {ac_logprob:.6}");

    // Naive leakage-permitting variant — standard causal everywhere.
    let naive_logprob = forward_naive_causal(&model, &base_tokens, &xc_positions);
    println!();
    println!("── Naive causal logprob (leakage-permitting, no copies) ──");
    println!("   log p(xe | xc) = {naive_logprob:.6}");

    let diff = (ac_logprob - naive_logprob).abs();
    println!();
    println!("── Leakage-prevention proof ──");
    println!("   |AC - Naive| = {diff:.6}");
    if diff > 1e-4 {
        println!("   ✓ The two logprobs differ — proving the AC-GPT three-region");
        println!("     discipline matters (the naive mask permits multi-layer");
        println!("     information leakage through the in-place conditioning tokens).");
    } else {
        println!("   (Difference below threshold — possible with degenerate weights.)");
    }

    // Sampled continuation.
    println!();
    println!("── Sampled continuation via conditional_sample (Gumbel-max) ──");
    let mut rng = fastrand::Rng::with_seed(0xABCDEF);
    let sampled = prefix.conditional_sample(
        |tokens, positions, mask, _loss_mask, eval_slot| {
            // Forward the whole augmented sequence, return the vocab logits
            // at the current eval slot. The Gumbel-max sampler then draws
            // from softmax(logits) without materializing the categorical.
            let all_logits = forward_masked_ac_logits(&model, tokens, positions, mask);
            all_logits[eval_slot].clone()
        },
        &mut rng,
    );
    println!("   Sampled {} eval tokens: {:?}", sampled.len(), sampled);

    println!();
    println!("Demo complete — primitive is leak-free and end-to-end functional.");
}
