//! Plan 337 Phase A+C+D — Train the DualEncoderIndexer on real Kimi-K3-0.40B
//! attention patterns and compare against the modelless FlashMemorySelector.
//!
//! **CAVEAT (user insight 2026-08-13):** Kimi-K3-0.40B (395M, hybrid 6-KDA/2-MLA)
//! has near-uniform attention patterns at this scale — the model is too small to
//! have learned strong retrieval-focused attention. The golden labels extracted
//! from its attention are low-signal. This bench proves the TRAINING PIPELINE works
//! end-to-end (Phase A extraction → Phase C training loop → Phase D GOAT gate),
//! but the quality result is NOT meaningful on this model. The real validation
//! needs Bonsai (27B, all full-attention layers, strong learned patterns) on the 4090.
//!
//! **Phase C upgrade (2026-08-13):** the IndexerTrainer now uses Adam
//! (β1=0.9, β2=0.999, ε=1e-8) instead of SGD+momentum. This resolves the
//! bilinear σ(q·k) vanishing gradient issue that prevented convergence.
//! See bench_026 for the synthetic convergence proof (100% accuracy with
//! Adam + bias init on clear-pattern data).
//!
//! **Training convergence note:** Even with Adam, the training may not
//! converge on Kimi-K3-0.40B because the golden labels are low-signal
//! (near-uniform attention at 395M scale). The real validation needs
//! Bonsai (27B, strong learned patterns) on the 4090.
//!
//! This bench runs ENTIRELY on M3 Metal (no GPU needed):
//! 1. Load real Kimi-K3-0.40B model.safetensors
//! 2. Run dense MLA forward on tokenized input → populate KV cache
//! 3. Extract per-query per-block attention mass (the training labels)
//! 4. Train DualEncoderIndexer MLPs with manual gradient descent + BCE loss
//! 5. GOAT gate: compare trained indexer vs modelless selector on recall/precision
//!
//! # Run
//!
//! ```bash
//! cargo bench --features "kimi_k3_loader trained_indexer" \
//!     --bench bench_025_flashmemory_trained_indexer -- --nocapture
//! ```

#![cfg(feature = "trained_indexer")]
#![allow(clippy::needless_range_loop)]

use std::time::Instant;

use katgpt_attn::dash_attn::flashmemory_sparse::{
    DualEncoderIndexer, FlashMemoryBlockCache, FlashMemoryConfig, FlashMemorySelector,
    mla_forward_token_flashmemory,
};
use katgpt_attn::mla::{MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token};
use katgpt_core::simd::{simd_dot_f32, simd_matmul_rows};
use katgpt_kv::shard_kv::rope::RopeFreqs;

use katgpt_rs::kimi_k3::loader::{KimiK3ModelWeights, load_kimi_k3};

const MLA_LAYER_IDX: usize = 3;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot = simd_dot_f32(a, b, a.len());
    let na = simd_dot_f32(a, a, a.len()).sqrt();
    let nb = simd_dot_f32(b, b, a.len()).sqrt();
    if na < 1e-12 || nb < 1e-12 { 0.0 } else { dot / (na * nb) }
}

/// A single training triple: (query projections, block key centroid, label).
struct TrainingTriple {
    /// `[n_heads * d_h]` — query content projections (q_c).
    q_c: Vec<f32>,
    /// `[n_heads * d_h]` — block key centroid (up-projected from latent).
    k_centroid: Vec<f32>,
    /// Golden label: 1.0 if the block was attended (mass > threshold), else 0.0.
    label: f32,
}

/// Adam trainer for the DualEncoderIndexer MLPs (Phase C upgrade from SGD).
///
/// Manual backprop for Linear(d, hidden) → ReLU → Linear(hidden, 1).
/// Uses BCE loss: L = -(y·log(σ(z)) + (1-y)·log(1-σ(z))) where z = q_score · k_score.
///
/// Upgraded from SGD+momentum to Adam (Phase C, Plan 337). Adam handles the
/// bilinear σ(q·k) dynamics far better — the vanishing gradient that plagued
/// SGD (`dq = dz * k_score ≈ 0` at random init) is resolved by Adam's
/// per-parameter adaptive learning rates. See bench_026 for the synthetic
/// convergence proof.
struct IndexerTrainer {
    d_h: usize,
    hidden: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: usize,

    // Q-Indexer weights
    q_w1: Vec<f32>, q_b1: Vec<f32>, q_w2: Vec<f32>, q_b2: f32,
    // K-Indexer weights
    k_w1: Vec<f32>, k_b1: Vec<f32>, k_w2: Vec<f32>, k_b2: f32,

    // Adam first moment (m) + second moment (v) buffers
    q_w1_m: Vec<f32>, q_w1_v: Vec<f32>,
    q_b1_m: Vec<f32>, q_b1_v: Vec<f32>,
    q_w2_m: Vec<f32>, q_w2_v: Vec<f32>,
    q_b2_m: f32, q_b2_v: f32,

    k_w1_m: Vec<f32>, k_w1_v: Vec<f32>,
    k_b1_m: Vec<f32>, k_b1_v: Vec<f32>,
    k_w2_m: Vec<f32>, k_w2_v: Vec<f32>,
    k_b2_m: f32, k_b2_v: f32,

    // Forward scratch
    q_hidden: Vec<f32>,
    k_hidden: Vec<f32>,
}

impl IndexerTrainer {
    fn from_indexer(indexer: &DualEncoderIndexer, lr: f32) -> Self {
        // Extract weights from the indexer via to_bytes → from_bytes roundtrip
        // (the indexer owns the weights; we copy them out for training).
        let d_h = indexer.d_h_dim();
        let hidden = indexer.hidden_dim();
        let (qw1, qb1, qw2, qb2, kw1, kb1, kw2, kb2) = indexer.extract_weights();

        Self {
            d_h, hidden, lr,
            beta1: 0.9, beta2: 0.999, epsilon: 1e-8, timestep: 0,
            q_w1: qw1.clone(), q_b1: qb1.clone(), q_w2: qw2.clone(), q_b2: qb2,
            k_w1: kw1.clone(), k_b1: kb1.clone(), k_w2: kw2.clone(), k_b2: kb2,
            q_w1_m: vec![0.0; hidden * d_h], q_w1_v: vec![0.0; hidden * d_h],
            q_b1_m: vec![0.0; hidden], q_b1_v: vec![0.0; hidden],
            q_w2_m: vec![0.0; hidden], q_w2_v: vec![0.0; hidden],
            q_b2_m: 0.0, q_b2_v: 0.0,
            k_w1_m: vec![0.0; hidden * d_h], k_w1_v: vec![0.0; hidden * d_h],
            k_b1_m: vec![0.0; hidden], k_b1_v: vec![0.0; hidden],
            k_w2_m: vec![0.0; hidden], k_w2_v: vec![0.0; hidden],
            k_b2_m: 0.0, k_b2_v: 0.0,
            q_hidden: vec![0.0; hidden],
            k_hidden: vec![0.0; hidden],
        }
    }

    /// Forward + backward + Adam update on a single triple.
    /// Returns the BCE loss for this sample.
    fn train_step(&mut self, q_c_h: &[f32], k_centroid_h: &[f32], label: f32) -> f32 {
        let d = self.d_h;
        let h = self.hidden;
        self.timestep += 1;
        let t = self.timestep as f32;

        // ── Forward: Q-Indexer ──
        simd_matmul_rows(&mut self.q_hidden, &self.q_w1, q_c_h, h, d);
        for i in 0..h { self.q_hidden[i] = (self.q_hidden[i] + self.q_b1[i]).max(0.0); }
        let mut q_score = self.q_b2;
        for i in 0..h { q_score += self.q_w2[i] * self.q_hidden[i]; }

        // ── Forward: K-Indexer ──
        simd_matmul_rows(&mut self.k_hidden, &self.k_w1, k_centroid_h, h, d);
        for i in 0..h { self.k_hidden[i] = (self.k_hidden[i] + self.k_b1[i]).max(0.0); }
        let mut k_score = self.k_b2;
        for i in 0..h { k_score += self.k_w2[i] * self.k_hidden[i]; }

        // ── Prediction: σ(q_score · k_score) ──
        let z = (q_score * k_score).clamp(-30.0, 30.0);
        let p = katgpt_core::sigmoid(z);
        let p_clamped = p.clamp(1e-7, 1.0 - 1e-7);

        // ── BCE loss ──
        // Asymmetric BCE: w+ = 8 penalizes false-elimination 8× harder.
        let w_pos = 8.0f32;
        let w_neg = 1.0f32;
        let loss = -(w_pos * label * p_clamped.ln() + w_neg * (1.0 - label) * (1.0 - p_clamped).ln());

        // ── Backward: dL/dz for asymmetric BCE ──
        let mut dz = w_neg * p + label * (p * (w_pos - w_neg) - w_pos);
        dz = dz.clamp(-5.0, 5.0);

        let dq_score = dz * k_score;
        let dk_score = dz * q_score;

        // ── Backward Q-Indexer + Adam update ──
        let mut dq_hidden = vec![0.0f32; h];
        for i in 0..h {
            dq_hidden[i] = dq_score * self.q_w2[i];
            if self.q_hidden[i] <= 0.0 { dq_hidden[i] = 0.0; }
        }

        for i in 0..h {
            let grad = dq_score * self.q_hidden[i];
            Self::adam_vec(self.beta1, self.beta2, self.epsilon, self.lr,
                &mut self.q_w2, &mut self.q_w2_m, &mut self.q_w2_v, i, grad, t);
        }
        Self::adam_scalar(self.beta1, self.beta2, self.epsilon, self.lr,
            &mut self.q_b2, &mut self.q_b2_m, &mut self.q_b2_v, dq_score, t);

        for i in 0..h {
            Self::adam_vec(self.beta1, self.beta2, self.epsilon, self.lr,
                &mut self.q_b1, &mut self.q_b1_m, &mut self.q_b1_v, i, dq_hidden[i], t);
            let row_off = i * d;
            for j in 0..d {
                let grad = dq_hidden[i] * q_c_h[j];
                Self::adam_vec(self.beta1, self.beta2, self.epsilon, self.lr,
                    &mut self.q_w1, &mut self.q_w1_m, &mut self.q_w1_v, row_off + j, grad, t);
            }
        }

        // ── Backward K-Indexer + Adam update ──
        let mut dk_hidden = vec![0.0f32; h];
        for i in 0..h {
            dk_hidden[i] = dk_score * self.k_w2[i];
            if self.k_hidden[i] <= 0.0 { dk_hidden[i] = 0.0; }
        }

        for i in 0..h {
            let grad = dk_score * self.k_hidden[i];
            Self::adam_vec(self.beta1, self.beta2, self.epsilon, self.lr,
                &mut self.k_w2, &mut self.k_w2_m, &mut self.k_w2_v, i, grad, t);
        }
        Self::adam_scalar(self.beta1, self.beta2, self.epsilon, self.lr,
            &mut self.k_b2, &mut self.k_b2_m, &mut self.k_b2_v, dk_score, t);

        for i in 0..h {
            Self::adam_vec(self.beta1, self.beta2, self.epsilon, self.lr,
                &mut self.k_b1, &mut self.k_b1_m, &mut self.k_b1_v, i, dk_hidden[i], t);
            let row_off = i * d;
            for j in 0..d {
                let grad = dk_hidden[i] * k_centroid_h[j];
                Self::adam_vec(self.beta1, self.beta2, self.epsilon, self.lr,
                    &mut self.k_w1, &mut self.k_w1_m, &mut self.k_w1_v, row_off + j, grad, t);
            }
        }

        loss
    }

    /// Adam update for a single element in a slice. Free function pattern
    /// to avoid `&mut self` borrow conflicts.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn adam_vec(
        beta1: f32, beta2: f32, epsilon: f32, lr: f32,
        param: &mut [f32], m: &mut [f32], v: &mut [f32],
        idx: usize, grad: f32, t: f32,
    ) {
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad * grad;
        let m_hat = m[idx] / (1.0 - beta1.powf(t));
        let v_hat = v[idx] / (1.0 - beta2.powf(t));
        param[idx] -= lr * m_hat / (v_hat.sqrt() + epsilon);
    }

    /// Adam update for a scalar parameter.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn adam_scalar(
        beta1: f32, beta2: f32, epsilon: f32, lr: f32,
        param: &mut f32, m: &mut f32, v: &mut f32,
        grad: f32, t: f32,
    ) {
        *m = beta1 * *m + (1.0 - beta1) * grad;
        *v = beta2 * *v + (1.0 - beta2) * grad * grad;
        let m_hat = *m / (1.0 - beta1.powf(t));
        let v_hat = *v / (1.0 - beta2.powf(t));
        *param -= lr * m_hat / (v_hat.sqrt() + epsilon);
    }

    /// Build a trained DualEncoderIndexer from the current weights.
    fn to_indexer(&self, config: FlashMemoryConfig, n_heads: usize, max_blocks: usize) -> DualEncoderIndexer {
        DualEncoderIndexer::from_weights(
            config, self.d_h, n_heads, max_blocks,
            self.q_w1.clone(), self.q_b1.clone(), self.q_w2.clone(), self.q_b2,
            self.k_w1.clone(), self.k_b1.clone(), self.k_w2.clone(), self.k_b2,
        )
    }
}

fn run_bench() {
    let config = katgpt_rs::kimi_k3::model::KimiK3ModelConfig::kimi_k3_0_40b();
    let mla_config = config.mla_config.clone();
    let d = config.hidden_size;
    let d_h = mla_config.d_h();
    let n_heads = mla_config.n_heads;

    // ── Load model ──
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        format!("{}/data/kimi-k3-0.40b", env!("CARGO_MANIFEST_DIR"))
    });
    let model_path = format!("{model_dir}/model.safetensors");
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: requires model.safetensors at {model_path}");
        std::process::exit(1);
    }
    print!("Loading model ... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t0 = Instant::now();
    let weights: KimiK3ModelWeights = load_kimi_k3(&model_path).unwrap();
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── Extract MLA weights ──
    use katgpt_rs::kimi_k3::decoder_layer::KimiAttentionWeights;
    let KimiAttentionWeights::Mla(mla_w) = &weights.layers[MLA_LAYER_IDX].attention else {
        eprintln!("ERROR: layer {MLA_LAYER_IDX} is not MLA");
        std::process::exit(1);
    };
    let mla_weights: MlaWeights = mla_w.clone();

    // ── Build hidden states from embeddings ──
    let seq_len: usize = 256; // short context for speed on M3
    let hay_cycle = 128.min(config.vocab_size);
    let token_ids: Vec<u32> = (0..seq_len)
        .map(|i| (i % hay_cycle) as u32)
        .collect();
    let hidden_states: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&tid| weights.embed_weight[(tid as usize) * d..(tid as usize) * d + d].to_vec())
        .collect();

    // ── Phase A: Run dense forward to populate KV cache + extract attention labels ──
    println!("\n=== Phase A: Dense attention extraction ({seq_len} tokens) ===");
    let mut cache = MlaKVCache::new(&mla_config, seq_len + 1);
    let mut scratch = MlaForwardScratch::new(&mla_config, seq_len + 1);
    let mut rope = RopeFreqs::new_with_theta(mla_config.qk_rope_head_dim, mla_config.rope_theta);

    // Run dense forward on ALL tokens to populate the KV cache.
    for h in &hidden_states {
        mla_forward_token(&mla_config, &mla_weights, &mut cache, &mut scratch, &mut rope, h);
    }
    println!("KV cache populated: {seq_len} tokens");

    // Build block cache for centroid extraction.
    let block_size = 16usize;
    let fm_config = FlashMemoryConfig { block_size, refresh_period: 1000, threshold: 0.5 };
    let max_blocks = seq_len.div_ceil(block_size);
    let mut block_cache = FlashMemoryBlockCache::new(&mla_config, &fm_config, seq_len + 1);
    block_cache.rebuild_from_cache(&cache, &mla_weights);

    // Extract training triples: for each query position, compute attention mass per block.
    let scale = mla_config.attn_scale();
    let d_c = mla_config.kv_lora_rank;

    // Sample query positions (every 8th token to keep dataset manageable).
    let query_positions: Vec<usize> = (block_size..seq_len).step_by(8).collect();
    println!("Query positions: {} (every 8th token from {block_size} to {seq_len})", query_positions.len());

    let mut triples: Vec<TrainingTriple> = Vec::new();
    let label_threshold = 1.5 / max_blocks as f32; // attended if mass > 1.5× uniform

    for &q_pos in &query_positions {
        // Recompute query projections for this position.
        let h_q = &hidden_states[q_pos];

        // c_q = W_DQ · h, then RMSNorm
        let mut c_q = vec![0.0; mla_config.q_lora_rank];
        simd_matmul_rows(&mut c_q, &mla_weights.w_dq, h_q, mla_config.q_lora_rank, d);
        // RMSNorm c_q
        let mut ss = 0.0f32;
        for &v in &c_q { ss += v * v; }
        let rms = (ss / c_q.len() as f32 + 1e-5).sqrt();
        for i in 0..c_q.len() { c_q[i] = c_q[i] / rms * mla_weights.q_a_norm_weight[i]; }

        // q_c = W_UQ · c_q (n_heads * d_h)
        let mut q_c = vec![0.0; n_heads * d_h];
        simd_matmul_rows(&mut q_c, &mla_weights.w_uq, &c_q, n_heads * d_h, mla_config.q_lora_rank);

        for head in 0..n_heads {
            let q_c_h = &q_c[head * d_h..(head + 1) * d_h];

            // Compute attention scores for all tokens up to q_pos.
            let mut scores = vec![0.0f32; q_pos + 1];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..=q_pos {
                let c_kv_j = cache.latent_kv_at(j);
                // k_c_j = W_UK[head] · c_kv_j
                let mut k_c = vec![0.0; d_h];
                simd_matmul_rows(
                    &mut k_c, &mla_weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                    c_kv_j, d_h, d_c,
                );
                let content = simd_dot_f32(q_c_h, &k_c, d_h);
                // Rope term intentionally skipped — needs q_r, kept simple.
                scores[j] = content * scale;
                if scores[j] > max_s { max_s = scores[j]; }
            }

            // Softmax → attention weights.
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - max_s).exp(); sum_exp += *s; }
            let inv = 1.0 / sum_exp;
            for s in scores.iter_mut() { *s *= inv; }

            // Sum per block → block attention mass.
            for block_idx in 0..max_blocks {
                let (bs, be) = (block_idx * block_size, ((block_idx + 1) * block_size).min(q_pos + 1));
                if bs >= be { continue; }
                let mut mass = 0.0f32;
                for j in bs..be { mass += scores[j]; }

                let label = if mass > label_threshold { 1.0 } else { 0.0 };

                // Store the block's key centroid for this head.
                let k_centroid_h = block_cache.key_centroid(block_idx, head).to_vec();
                let mut q_c_all = vec![0.0; n_heads * d_h];
                q_c_all.copy_from_slice(&q_c);

                triples.push(TrainingTriple {
                    q_c: q_c_h.to_vec(),
                    k_centroid: k_centroid_h,
                    label,
                });
            }
        }
    }

    let n_positive = triples.iter().filter(|t| t.label > 0.5).count();
    println!("Training triples: {} ({} positive = {:.1}%)",
        triples.len(), n_positive, 100.0 * n_positive as f32 / triples.len() as f32);

    // ── Phase C: Train the DualEncoderIndexer ──
    println!("\n=== Phase C: Training DualEncoderIndexer ===");
    let init_indexer = DualEncoderIndexer::new_random(
        fm_config.clone(), d_h, n_heads, max_blocks, 42,
    );
    println!("Indexer params: {} (d_h={d_h}, hidden={})",
        init_indexer.param_count(), init_indexer.hidden_dim());

    let mut trainer = IndexerTrainer::from_indexer(&init_indexer, 0.001);

    let n_epochs = 100;
    for epoch in 0..n_epochs {
        let mut total_loss = 0.0f32;
        for t in &triples {
            let loss = trainer.train_step(&t.q_c, &t.k_centroid, t.label);
            total_loss += loss;
        }
        let avg_loss = total_loss / triples.len() as f32;
        if epoch % 10 == 0 || epoch == n_epochs - 1 {
            println!("  Epoch {epoch:2}/{n_epochs}: avg_loss = {avg_loss:.4}");
        }
    }

    let trained_indexer = trainer.to_indexer(fm_config.clone(), n_heads, max_blocks);

    // ── Phase D: GOAT gate — trained vs modelless vs dense ──
    println!("\n=== Phase D: GOAT gate (trained vs modelless vs dense) ===");

    // We need fresh caches for each path (dense / modelless / trained).
    // Run all three on the same hidden states + compare outputs.

    let mut cos_modelless = Vec::new();
    let mut dense_outputs: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

    // Dense baseline.
    let mut cache_d = MlaKVCache::new(&mla_config, seq_len + 1);
    let mut scratch_d = MlaForwardScratch::new(&mla_config, seq_len + 1);
    let mut rope_d = RopeFreqs::new_with_theta(mla_config.qk_rope_head_dim, mla_config.rope_theta);
    for h in &hidden_states {
        let out = mla_forward_token(&mla_config, &mla_weights, &mut cache_d, &mut scratch_d, &mut rope_d, h);
        dense_outputs.push(out.to_vec());
    }

    // Modelless sparse.
    let mut cache_m = MlaKVCache::new(&mla_config, seq_len + 1);
    let mut scratch_m = MlaForwardScratch::new(&mla_config, seq_len + 1);
    let mut rope_m = RopeFreqs::new_with_theta(mla_config.qk_rope_head_dim, mla_config.rope_theta);
    let mut bc_m = FlashMemoryBlockCache::new(&mla_config, &fm_config, seq_len + 1);
    let mut sel_m = FlashMemorySelector::new(fm_config.clone(), n_heads, max_blocks);
    for (step, h) in hidden_states.iter().enumerate() {
        let out = mla_forward_token_flashmemory(
            &mla_config, &mla_weights, &mut cache_m, &mut scratch_m, &mut rope_m,
            h, &mut bc_m, &mut sel_m, step,
        );
        cos_modelless.push(cosine_sim(&dense_outputs[step], out));
    }

    // Trained sparse — we need a forward function that uses the trained indexer.
    // Since mla_forward_token_flashmemory takes &mut FlashMemorySelector, and
    // DualEncoderIndexer has a compatible select() interface, we manually
    // replicate the forward here using the trained indexer's selection.
    //
    // For simplicity, we measure the indexer's selection QUALITY directly:
    // what fraction of golden blocks does it select? (recall)
    let mut trained_recall = 0.0f32;
    let mut trained_precision = 0.0f32;
    let mut modelless_recall = 0.0f32;
    let mut n_queries = 0usize;

    for &q_pos in &query_positions {
        let h_q = &hidden_states[q_pos];
        let mut c_q = vec![0.0; mla_config.q_lora_rank];
        simd_matmul_rows(&mut c_q, &mla_weights.w_dq, h_q, mla_config.q_lora_rank, d);
        let mut ss = 0.0f32;
        for &v in &c_q { ss += v * v; }
        let rms = (ss / c_q.len() as f32 + 1e-5).sqrt();
        for i in 0..c_q.len() { c_q[i] = c_q[i] / rms * mla_weights.q_a_norm_weight[i]; }
        let mut q_c = vec![0.0; n_heads * d_h];
        simd_matmul_rows(&mut q_c, &mla_weights.w_uq, &c_q, n_heads * d_h, mla_config.q_lora_rank);

        // Build a fresh block cache for this position.
        // We need a cache populated up to q_pos. Use the dense cache.
        // Actually, rebuild from the dense cache up to q_pos.
        // For simplicity, use the full block_cache we already built.
        let n_active = (q_pos + 1).div_ceil(block_size).min(max_blocks);

        // Trained indexer selection.
        let mut idx = trained_indexer.clone_for_eval();
        idx.force_refresh();
        let sel_t = idx.select(&q_c, &block_cache, 0).clone();

        // Modelless selection.
        let mut sel_m = FlashMemorySelector::new(fm_config.clone(), n_heads, max_blocks);
        sel_m.force_refresh();
        let sel_m_result = sel_m.select(&q_c, &block_cache, mla_config.attn_scale(), 0).clone();

        // Golden blocks: compute which blocks have above-threshold attention for this query.
        for head in 0..n_heads {
            let q_c_h = &q_c[head * d_h..(head + 1) * d_h];
            let mut scores = vec![0.0f32; q_pos + 1];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..=q_pos {
                let c_kv_j = cache_d.latent_kv_at(j);
                let mut k_c = vec![0.0; d_h];
                simd_matmul_rows(
                    &mut k_c, &mla_weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                    c_kv_j, d_h, d_c,
                );
                scores[j] = simd_dot_f32(q_c_h, &k_c, d_h) * scale;
                if scores[j] > max_s { max_s = scores[j]; }
            }
            let mut se = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - max_s).exp(); se += *s; }
            let inv = 1.0 / se;
            for s in scores.iter_mut() { *s *= inv; }

            // Golden blocks.
            let mut golden: Vec<usize> = Vec::new();
            for block_idx in 0..n_active {
                let (bs, be) = (block_idx * block_size, ((block_idx + 1) * block_size).min(q_pos + 1));
                let mut mass = 0.0f32;
                for j in bs..be { mass += scores[j]; }
                if mass > label_threshold { golden.push(block_idx); }
            }
            if golden.is_empty() { continue; }
            n_queries += 1;

            // Trained recall.
            let sel_t_h = &sel_t.blocks_per_head[head];
            let recalled_t = golden.iter().filter(|g| sel_t_h.contains(g)).count();
            trained_recall += recalled_t as f32 / golden.len() as f32;
            let precision_t = if sel_t_h.is_empty() { 0.0 } else {
                sel_t_h.iter().filter(|s| golden.contains(s)).count() as f32 / sel_t_h.len() as f32
            };
            trained_precision += precision_t;

            // Modelless recall.
            let sel_m_h = &sel_m_result.blocks_per_head[head];
            let recalled_m = golden.iter().filter(|g| sel_m_h.contains(g)).count();
            modelless_recall += recalled_m as f32 / golden.len() as f32;
        }
    }

    if n_queries > 0 {
        trained_recall /= n_queries as f32;
        trained_precision /= n_queries as f32;
        modelless_recall /= n_queries as f32;
    }

    // ── Report ──
    println!("\n=== Results ===");
    println!("Modelless sparse vs dense (cosine sim):");
    let cos_m_sorted = sort_median(&cos_modelless);
    println!("  median cos: {cos_m_sorted:.4}");

    println!("\nBlock selection recall (fraction of golden blocks selected):");
    println!("  Trained indexer:   recall={trained_recall:.4}  precision={trained_precision:.4}");
    println!("  Modelless selector: recall={modelless_recall:.4}");

    // GOAT verdict
    println!("\n=== GOAT Gate ===");
    if n_queries > 0 {
        let trained_beats = trained_recall > modelless_recall;
        println!("D3 (recall gate): trained recall {trained_recall:.4} vs modelless {modelless_recall:.4}");
        println!("  → {}", if trained_beats { "✅ TRAINED BEATS MODELESS" } else { "⚠️  TRAINED DOES NOT BEAT MODELESS" });
    } else {
        println!("D3: no golden blocks found (all attention uniform) — inconclusive");
    }

    let cos_m_med = sort_median(&cos_modelless);
    println!("G1 (modelless correctness): median cos = {cos_m_med:.4} {}", if cos_m_med >= 0.90 { "✅ PASS" } else { "❌ FAIL" });
}

fn sort_median(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

fn main() {
    run_bench();
}
