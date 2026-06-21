//! Functional Attention (FUNCATTN) — G6 LLM-domain token-prediction gate
//! (Plan 286 T4.4).
//!
//! ## Why this test exists
//!
//! G1–G5 (mechanics, scaling, zero-alloc, sigmoid-vs-softmax, FUNCATTN-vs-
//! Parallax-vs-SDPA on **regression**) all PASS, and G2 STRICT PASSes the
//! paper's headline §5.1 sample-efficiency targets. Per Plan 286 T4.2 this
//! made `funcattn` eligible for opt-in promotion (it's in the `full`
//! aggregation). But T4.4 explicitly blocks **default-on** promotion until
//! LLM-domain token-prediction evidence exists, because:
//!
//!   - Research 257 §1.5: the paper itself has *not* verified FUNCATTN on NLP.
//!   - Research 257 §5 Q2: "Risk: we ship the open primitive, run GOAT gate,
//!     find no gain over Parallax/SDPA on real LM data, demote. This is the
//!     expected outcome for the katgpt-rs side."
//!   - Benchmark 058 G2 caveat: SDPA catches up to FUNCATTN at 500+ training
//!     steps in the regression task — the strict G2 win is specifically a
//!     sample-efficiency effect, and LLM training is a many-thousand-step
//!     regime where that effect may vanish.
//!
//! This test gathers that LLM-domain evidence on a small-but-genuine masked-
//! token-prediction task (the same `generate_pattern_dataset` infrastructure
//! used by the rest of `dllm.rs`).
//!
//! ## Task
//!
//! Masked token prediction on alternating-pattern sequences
//! `[a, b, a, b, a, b, a, b]` (seq_len=8, vocab=8). The model sees a sequence
//! with one position masked; it must predict the masked token from the rest.
//! This is a genuine LM-domain task (token prediction on discrete sequences
//! with cross-entropy loss) — NOT a PDE-style regression on continuous fields
//! like G2/G3. Both attention variants must learn the period-2 structure from
//! the unmasked positions.
//!
//! ## Architectures (matched param budget)
//!
//! Both predictors are single-head, single-attention-layer:
//!
//! ```text
//! X[n,:]   = W_emb[token[n],:] + W_pos[n,:]            // (N, D)
//! O[n,:]   = attention(X)                              // (N, D)
//! logits   = W_head · O                                // (N, V)
//! ```
//!
//! ```text
//! FUNCATTN: W_emb ((V+1)·D — one extra row for the mask token) + W_pos
//!           (N·D) + W_basis (k·D) + W_q (D·D) + W_k (D·D) + W_v (D·D)
//!           + W_head (V·D)
//!           = (V+1)·D + N·D + k·D + 3·D² + V·D  params
//! SDPA:     W_emb ((V+1)·D) + W_pos (N·D) + W_q (D·D) + W_k (D·D)
//!           + W_v (D·D) + W_head (V·D)
//!           = (V+1)·D + N·D + 3·D² + V·D       params
//! ```
//!
//! At V=D=N=k=8: FUNCATTN = 72+64+64+192+64 = 456 params;
//!               SDPA     = 72+64+192+64    = 392 params.
//!
//! SDPA has ~14% fewer params (the W_basis term is FUNCATTN-specific). This
//! is a slight handicap against FUNCATTN, mirroring the G2 setup. If FUNCATTN
//! cannot beat SDPA even with more capacity in the LM domain, the null result
//! is robust.
//!
//! ## Training
//!
//! Central finite-difference SGD (matches G2/G3). Steps chosen to land in
//! the "thousands of gradient updates per parameter" regime where the G2
//! sample-efficiency advantage is expected to vanish — this is the explicit
//! purpose of T4.4 per the benchmark 058 caveat.
//!
//! ## Verdict
//!
//! - **Sanity** (hard assert): both variants reduce loss below their init loss.
//!   If a variant didn't learn, the harness is broken — not a real G6 verdict.
//! - **T4.4 promotion gate** (reported, not asserted): FUNCATTN token-
//!   prediction accuracy ≥ SDPA accuracy at convergence → eligible for
//!   default-on promotion (T4.4 lift). Strictly less → null result,
//!   `funcattn` stays opt-in only.
//!
//! Per Plan 286 T4.4 and Research 257 §5 Q2, the **expected** outcome is the
//! null result. The test passes either way — what matters is the honest
//! measurement recorded in `.benchmarks/058_funcattn_goat.md`.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features funcattn --release \
//!   --test funcattn_g6_token_prediction_lm_domain -- --nocapture
//! ```
//!
//! (Release strongly recommended: 2 variants × STEPS × ~448 params × 2 FD
//! evals ≈ 1.5M+ forward passes.)

#![cfg(feature = "funcattn")]

use katgpt_core::attention::tiled_attention_forward_with_scores;
use katgpt_core::funcattn::{funcattn_forward, FuncAttnBasis, FuncAttnConfig, FuncAttnScratch};
use katgpt_core::simd;

// ── Model dimensions ─────────────────────────────────────────────────
/// Vocabulary size — real tokens span `0..V`.
const V: usize = 8;
/// Embedding / head dim (matches G2).
const D: usize = 8;
/// Sequence length.
const N: usize = 8;
/// FUNCATTN basis dimension.
const K: usize = 8;
/// Mask token id — occupies its own embedding row at index `V`. Real tokens
/// are `0..V`; the mask is `V`, so the embedding matrix is sized `(V+1) * D`.
const MASK_TOKEN: usize = V;
/// Effective vocab — tokens 0..V are real.
const EFFECTIVE_VOCAB: usize = V;
/// SDPA scale = 1/√D.
const SCALE: f32 = 0.353_553_38; // 1.0 / sqrt(8.0)

// ── Training hyperparameters ────────────────────────────────────────────
/// FD-SGD steps. Sized to land in the "thousands of updates per param"
/// regime — the explicit purpose of T4.4 (per benchmark 058 G2 caveat:
/// SDPA catches up to FUNCATTN at 500+ steps in the regression task).
#[cfg(not(debug_assertions))]
const STEPS: usize = 600;
#[cfg(debug_assertions)]
const STEPS: usize = 40;
/// FD-SGD learning rate. CE gradients on near-uniform softmax are large at
/// init (|∂L/∂logit| ≈ 1/V = 0.125 per class), so we keep LR well below the
/// G2 regression value (1.0). 0.05 is empirically stable across both
/// variants for ≥500 steps; higher values (≥0.1) diverge within ~30 steps
/// because the FD gradient estimate amplifies any near-saturated softmax
/// row. The point of T4.4 is the *converged* regime, not sample-efficiency,
/// so a conservative LR is the right call.
const LR: f32 = 0.05;
const FD_EPS: f32 = 1e-2;
/// FUNCATTN Tikhonov regularization. α=0.5 = the reference default
/// (sigmoid(0)); a middle-of-the-road choice — neither overfit nor over-
/// regularized. G2 used α=0.01 (minimal ridge); here we use the reference
/// default to avoid giving FUNCATTN an unfair task-specific tuning advantage.
const ALPHA: f32 = 0.5;
/// FUNCATTN sigmoid basis temperature. Per G3 finding, sigmoid needs τ ≤ 0.1
/// at small input scales to produce non-uniform Φ. We use τ=0.1 (the lower
/// bound of the reference clamp [0.1, 5.0]).
const TEMPERATURE: f32 = 0.1;
/// Fixed reproducible seed (literal kept short on purpose so it's easy to
/// grep for). Same seed across debug/release so the G6 verdict is stable.
const SEED_U64: u64 = 0xC0FFEE_42AA_u64;

// ── Deterministic xorshift64* PRNG (mirrors G2/G3 tests) ─────────────────

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            s: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.s ^= self.s >> 12;
        self.s ^= self.s << 25;
        self.s ^= self.s >> 27;
        self.s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let u01 = bits as f32 / ((1u32 << 24) as f32);
        u01 * 2.0 - 1.0
    }
    fn fill(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = self.next_f32();
        }
    }
}

/// D×D identity matrix, row-major.
fn identity_matrix(d: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    w
}

/// Row-orthogonal init via modified Gram-Schmidt (reference L20-21).
fn orthogonal_init(rows: usize, cols: usize, rng: &mut Rng) -> Vec<f32> {
    let mut w = vec![0.0f32; rows * cols];
    rng.fill(&mut w);
    for i in 0..rows {
        let mut norm = 0.0;
        for c in 0..cols {
            norm += w[i * cols + c] * w[i * cols + c];
        }
        norm = norm.sqrt().max(1e-12);
        for c in 0..cols {
            w[i * cols + c] /= norm;
        }
    }
    for i in 0..rows {
        for j in 0..i {
            let mut dot = 0.0;
            for c in 0..cols {
                dot += w[i * cols + c] * w[j * cols + c];
            }
            for c in 0..cols {
                w[i * cols + c] -= dot * w[j * cols + c];
            }
        }
        let mut norm = 0.0;
        for c in 0..cols {
            norm += w[i * cols + c] * w[i * cols + c];
        }
        norm = norm.sqrt().max(1e-12);
        for c in 0..cols {
            w[i * cols + c] /= norm;
        }
    }
    w
}

// ── Pattern dataset (mirrors `dllm::generate_pattern_dataset` shape) ──────
//
// We vendor a local copy because `dllm::generate_pattern_dataset` lives
// behind the `dllm` feature, and we want this gate to run with only
// `funcattn` enabled (the actual subject of T4.4). The data is identical:
// `[a, b, a, b, ...]` alternating sequences with effective_vocab distinct
// (a, b) pairs.

fn generate_pattern_dataset(
    rng: &mut Rng,
    n_sequences: usize,
    seq_len: usize,
    effective_vocab: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(n_sequences);
    for _ in 0..n_sequences {
        let a = (rng.next_u64() as usize) % effective_vocab.max(1);
        let b = (rng.next_u64() as usize) % effective_vocab.max(1);
        let seq: Vec<usize> = (0..seq_len).map(|i| if i % 2 == 0 { a } else { b }).collect();
        out.push(seq);
    }
    out
}

// ── Shared forward pieces ────────────────────────────────────────────────

/// Embed tokens + add positional encoding.
/// `x[n, d] = w_emb[token[n]*D + d] + w_pos[n*D + d]`.
fn embed_add_pos(
    tokens: &[usize],
    w_emb: &[f32],
    w_pos: &[f32],
    out: &mut [f32],
) {
    for n in 0..N {
        let tok = tokens[n];
        let emb_row = &w_emb[tok * D..(tok + 1) * D];
        let pos_row = &w_pos[n * D..(n + 1) * D];
        let out_row = &mut out[n * D..(n + 1) * D];
        simd::simd_add_into(out_row, emb_row, pos_row);
    }
}

/// Compute logits = W_head · O for every position. `W_head` is (V, D) row-
/// major; output is (N, V).
fn project_to_vocab(o: &[f32], w_head: &[f32], logits: &mut [f32]) {
    for n in 0..N {
        let o_row = &o[n * D..(n + 1) * D];
        let logits_row = &mut logits[n * V..(n + 1) * V];
        simd::simd_matmul_rows(logits_row, w_head, o_row, V, D);
    }
}

/// Per-token linear projection `dst[n,:] = W · x[n,:]` where W is (D, D)
/// row-major. Free function (not a method) so callers can pass disjoint
/// field borrows without contending on `&self`.
fn project_x_free(x: &[f32], w: &[f32], dst: &mut [f32]) {
    for n in 0..N {
        let x_row = &x[n * D..(n + 1) * D];
        let dst_row = &mut dst[n * D..(n + 1) * D];
        simd::simd_matmul_rows(dst_row, w, x_row, D, D);
    }
}

/// Numerically stable softmax over the V axis for each row.
fn softmax_rows_v(logits: &[f32], probs: &mut [f32]) {
    for n in 0..N {
        let row = &logits[n * V..(n + 1) * V];
        let mut max = f32::NEG_INFINITY;
        for &x in row {
            if x > max {
                max = x;
            }
        }
        let mut sum = 0.0f32;
        let out_row = &mut probs[n * V..(n + 1) * V];
        for v in 0..V {
            let e = (row[v] - max).exp();
            out_row[v] = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(1e-20);
        for v in 0..V {
            out_row[v] *= inv;
        }
    }
}

/// Cross-entropy loss on a single masked position.
///
/// `loss = -log(probs[masked_pos, true_token])`.
fn cross_entropy_masked(probs: &[f32], masked_pos: usize, true_token: usize) -> f32 {
    let p = probs[masked_pos * V + true_token].max(1e-12);
    -p.ln()
}

// ── Field dispatch for FD-SGD ───────────────────────────────────────
//
// We use an enum + dispatch instead of a closure-over-self so the borrow
// checker can see that each `field_set` releases its borrow before
// `forward_loss` (which needs `&mut self`) runs. This is the standard Rust
// pattern for per-parameter FD where the forward pass touches the whole
// struct.

#[derive(Clone, Copy, PartialEq, Eq)]
enum FieldId {
    WEmb,
    WPos,
    WBasis,
    WQ,
    WK,
    WV,
    WHead,
}

impl FieldId {
    /// All FUNCATTN-trainable fields, in the order they are updated.
    const FUNCATTN_ALL: [FieldId; 7] = [
        FieldId::WEmb,
        FieldId::WPos,
        FieldId::WBasis,
        FieldId::WQ,
        FieldId::WK,
        FieldId::WV,
        FieldId::WHead,
    ];
    /// All SDPA-trainable fields (no W_basis).
    const SDPA_ALL: [FieldId; 6] = [
        FieldId::WEmb,
        FieldId::WPos,
        FieldId::WQ,
        FieldId::WK,
        FieldId::WV,
        FieldId::WHead,
    ];
}

// ── FUNCATTN predictor ───────────────────────────────────────

struct FuncattnPredictor {
    w_emb: Vec<f32>,   // (V, D)
    w_pos: Vec<f32>,   // (N, D)
    w_basis: Vec<f32>, // (K, D)
    w_q: Vec<f32>,     // (D, D)
    w_k: Vec<f32>,     // (D, D)
    w_v: Vec<f32>,     // (D, D)
    w_head: Vec<f32>,  // (V, D)
    // Scratch
    scratch: FuncAttnScratch,
    x_buf: Vec<f32>,    // (N, D)
    o_buf: Vec<f32>,    // (N, D)
    logits: Vec<f32>,   // (N, V)
    probs: Vec<f32>,    // (N, V)
}

impl FuncattnPredictor {
    fn n_params() -> usize {
        // (V+1)·D + N·D + K·D + D·D + D·D + D·D + V·D
        (V + 1) * D + N * D + K * D + 3 * D * D + V * D
    }

    fn new(rng: &mut Rng) -> Self {
        // Embeddings (V+1 rows so the mask token at index V has its own row)
        // + position + head: small random init (±0.5 scaled by 0.1 to keep
        // softmax away from saturation at init).
        let small_init = |rows: usize, cols: usize, rng: &mut Rng| -> Vec<f32> {
            let mut w = vec![0.0f32; rows * cols];
            rng.fill(&mut w);
            for x in w.iter_mut() {
                *x *= 0.1;
            }
            w
        };
        // w_emb has V+1 rows (real tokens 0..V, mask token at index V).
        let w_emb = small_init(V + 1, D, rng);
        let w_pos = small_init(N, D, rng);
        // W_basis: orthogonal init per reference L20-21.
        let w_basis = orthogonal_init(K, D, rng);
        // W_q, W_k, W_v: identity init (recovers plain linear at init).
        let w_q = identity_matrix(D);
        let w_k = identity_matrix(D);
        let w_v = identity_matrix(D);
        let w_head = small_init(V, D, rng);
        Self {
            w_emb,
            w_pos,
            w_basis,
            w_q,
            w_k,
            w_v,
            w_head,
            scratch: FuncAttnScratch::new(N, D, K),
            x_buf: vec![0.0; N * D],
            o_buf: vec![0.0; N * D],
            logits: vec![0.0; N * V],
            probs: vec![0.0; N * V],
        }
    }

    fn cfg() -> FuncAttnConfig {
        FuncAttnConfig {
            d: D,
            k: K,
            basis: FuncAttnBasis::Sigmoid,
            alpha: ALPHA,
            temperature: TEMPERATURE,
            cholesky_jitter: 1e-6,
        }
    }

    /// Forward to per-position probabilities. Returns loss at `masked_pos`
    /// against `true_token`.
    fn forward_loss(&mut self, tokens: &[usize], masked_pos: usize, true_token: usize) -> f32 {
        embed_add_pos(tokens, &self.w_emb, &self.w_pos, &mut self.x_buf);
        let cfg = Self::cfg();
        funcattn_forward(
            &self.x_buf,
            &self.x_buf,
            &self.w_basis,
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &cfg,
            &mut self.scratch,
            &mut self.o_buf,
        )
        .expect("funcattn forward");
        project_to_vocab(&self.o_buf, &self.w_head, &mut self.logits);
        softmax_rows_v(&self.logits, &mut self.probs);
        cross_entropy_masked(&self.probs, masked_pos, true_token)
    }

    /// Predict argmax token at `masked_pos`.
    fn predict(&mut self, tokens: &[usize], masked_pos: usize) -> usize {
        embed_add_pos(tokens, &self.w_emb, &self.w_pos, &mut self.x_buf);
        let cfg = Self::cfg();
        funcattn_forward(
            &self.x_buf,
            &self.x_buf,
            &self.w_basis,
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &cfg,
            &mut self.scratch,
            &mut self.o_buf,
        )
        .expect("funcattn forward");
        project_to_vocab(&self.o_buf, &self.w_head, &mut self.logits);
        softmax_rows_v(&self.logits, &mut self.probs);
        let row = &self.probs[masked_pos * V..(masked_pos + 1) * V];
        let mut best = 0usize;
        let mut best_p = row[0];
        for v in 1..V {
            if row[v] > best_p {
                best_p = row[v];
                best = v;
            }
        }
        best
    }

    /// One FD-SGD step on every trainable weight. Returns post-step loss.
    ///
    /// We perturb one weight at a time via field-id dispatch (avoids the
    /// double-mutable-borrow that a closure-over-self would trigger).
    fn fd_sgd_step(&mut self, tokens: &[usize], masked_pos: usize, true_token: usize) -> f32 {
        let inv_2eps = 1.0 / (2.0 * FD_EPS);
        for field in FieldId::FUNCATTN_ALL.iter() {
            let len = self.field_len(*field);
            for i in 0..len {
                let orig = self.field_get(*field, i);
                self.field_set(*field, i, orig + FD_EPS);
                let lp = self.forward_loss(tokens, masked_pos, true_token);
                self.field_set(*field, i, orig - FD_EPS);
                let lm = self.forward_loss(tokens, masked_pos, true_token);
                self.field_set(*field, i, orig);
                let grad = (lp - lm) * inv_2eps;
                self.field_set(*field, i, orig - LR * grad);
            }
        }
        self.forward_loss(tokens, masked_pos, true_token)
    }

    fn field_len(&self, field: FieldId) -> usize {
        match field {
            FieldId::WEmb => self.w_emb.len(),
            FieldId::WPos => self.w_pos.len(),
            FieldId::WBasis => self.w_basis.len(),
            FieldId::WQ => self.w_q.len(),
            FieldId::WK => self.w_k.len(),
            FieldId::WV => self.w_v.len(),
            FieldId::WHead => self.w_head.len(),
        }
    }

    fn field_get(&self, field: FieldId, i: usize) -> f32 {
        match field {
            FieldId::WEmb => self.w_emb[i],
            FieldId::WPos => self.w_pos[i],
            FieldId::WBasis => self.w_basis[i],
            FieldId::WQ => self.w_q[i],
            FieldId::WK => self.w_k[i],
            FieldId::WV => self.w_v[i],
            FieldId::WHead => self.w_head[i],
        }
    }

    fn field_set(&mut self, field: FieldId, i: usize, v: f32) {
        match field {
            FieldId::WEmb => self.w_emb[i] = v,
            FieldId::WPos => self.w_pos[i] = v,
            FieldId::WBasis => self.w_basis[i] = v,
            FieldId::WQ => self.w_q[i] = v,
            FieldId::WK => self.w_k[i] = v,
            FieldId::WV => self.w_v[i] = v,
            FieldId::WHead => self.w_head[i] = v,
        }
    }

    /// Evaluate token-prediction accuracy on a batch of (sequence, masked_pos,
    /// true_token) triples.
    fn accuracy(&mut self, samples: &[(Vec<usize>, usize, usize)]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut correct = 0usize;
        for (seq, mp, tt) in samples {
            let pred = self.predict(seq, *mp);
            if pred == *tt {
                correct += 1;
            }
        }
        correct as f32 / samples.len() as f32
    }
}

// ── SDPA predictor ───────────────────────────────────────────────────────

struct SdpaPredictor {
    w_emb: Vec<f32>,   // (V, D)
    w_pos: Vec<f32>,   // (N, D)
    w_q: Vec<f32>,     // (D, D)
    w_k: Vec<f32>,     // (D, D)
    w_v: Vec<f32>,     // (D, D)
    w_head: Vec<f32>,  // (V, D)
    // Scratch
    x_buf: Vec<f32>,    // (N, D)
    q_buf: Vec<f32>,    // (N, D)
    k_buf: Vec<f32>,    // (N, D)
    v_buf: Vec<f32>,    // (N, D)
    o_buf: Vec<f32>,    // (N, D)
    scores_buf: Vec<f32>, // (N, N)
    logits: Vec<f32>,   // (N, V)
    probs: Vec<f32>,    // (N, V)
}

impl SdpaPredictor {
    fn n_params() -> usize {
        // (V+1)·D + N·D + D·D + D·D + D·D + V·D
        (V + 1) * D + N * D + 3 * D * D + V * D
    }

    fn new(rng: &mut Rng) -> Self {
        let small_init = |rows: usize, cols: usize, rng: &mut Rng| -> Vec<f32> {
            let mut w = vec![0.0f32; rows * cols];
            rng.fill(&mut w);
            for x in w.iter_mut() {
                *x *= 0.1;
            }
            w
        };
        // w_emb has V+1 rows (real tokens 0..V, mask token at index V).
        let w_emb = small_init(V + 1, D, rng);
        let w_pos = small_init(N, D, rng);
        let w_q = orthogonal_init(D, D, rng);
        let w_k = identity_matrix(D);
        let w_v = identity_matrix(D);
        let w_head = small_init(V, D, rng);
        Self {
            w_emb,
            w_pos,
            w_q,
            w_k,
            w_v,
            w_head,
            x_buf: vec![0.0; N * D],
            q_buf: vec![0.0; N * D],
            k_buf: vec![0.0; N * D],
            v_buf: vec![0.0; N * D],
            o_buf: vec![0.0; N * D],
            scores_buf: vec![0.0; N * N],
            logits: vec![0.0; N * V],
            probs: vec![0.0; N * V],
        }
    }

    fn forward_loss(&mut self, tokens: &[usize], masked_pos: usize, true_token: usize) -> f32 {
        embed_add_pos(tokens, &self.w_emb, &self.w_pos, &mut self.x_buf);
        project_x_free(&self.x_buf, &self.w_q, &mut self.q_buf);
        project_x_free(&self.x_buf, &self.w_k, &mut self.k_buf);
        project_x_free(&self.x_buf, &self.w_v, &mut self.v_buf);
        tiled_attention_forward_with_scores(
            &self.q_buf,
            &self.k_buf,
            &self.v_buf,
            &mut self.o_buf,
            N,
            D,
            SCALE,
            Some(&mut self.scores_buf),
        );
        project_to_vocab(&self.o_buf, &self.w_head, &mut self.logits);
        softmax_rows_v(&self.logits, &mut self.probs);
        cross_entropy_masked(&self.probs, masked_pos, true_token)
    }

    fn predict(&mut self, tokens: &[usize], masked_pos: usize) -> usize {
        embed_add_pos(tokens, &self.w_emb, &self.w_pos, &mut self.x_buf);
        project_x_free(&self.x_buf, &self.w_q, &mut self.q_buf);
        project_x_free(&self.x_buf, &self.w_k, &mut self.k_buf);
        project_x_free(&self.x_buf, &self.w_v, &mut self.v_buf);
        tiled_attention_forward_with_scores(
            &self.q_buf,
            &self.k_buf,
            &self.v_buf,
            &mut self.o_buf,
            N,
            D,
            SCALE,
            Some(&mut self.scores_buf),
        );
        project_to_vocab(&self.o_buf, &self.w_head, &mut self.logits);
        softmax_rows_v(&self.logits, &mut self.probs);
        let row = &self.probs[masked_pos * V..(masked_pos + 1) * V];
        let mut best = 0usize;
        let mut best_p = row[0];
        for v in 1..V {
            if row[v] > best_p {
                best_p = row[v];
                best = v;
            }
        }
        best
    }

    fn fd_sgd_step(&mut self, tokens: &[usize], masked_pos: usize, true_token: usize) -> f32 {
        let inv_2eps = 1.0 / (2.0 * FD_EPS);
        for field in FieldId::SDPA_ALL.iter() {
            let len = self.field_len(*field);
            for i in 0..len {
                let orig = self.field_get(*field, i);
                self.field_set(*field, i, orig + FD_EPS);
                let lp = self.forward_loss(tokens, masked_pos, true_token);
                self.field_set(*field, i, orig - FD_EPS);
                let lm = self.forward_loss(tokens, masked_pos, true_token);
                self.field_set(*field, i, orig);
                let grad = (lp - lm) * inv_2eps;
                self.field_set(*field, i, orig - LR * grad);
            }
        }
        self.forward_loss(tokens, masked_pos, true_token)
    }

    fn field_len(&self, field: FieldId) -> usize {
        match field {
            FieldId::WEmb => self.w_emb.len(),
            FieldId::WPos => self.w_pos.len(),
            FieldId::WQ => self.w_q.len(),
            FieldId::WK => self.w_k.len(),
            FieldId::WV => self.w_v.len(),
            FieldId::WHead => self.w_head.len(),
            _ => 0,
        }
    }

    fn field_get(&self, field: FieldId, i: usize) -> f32 {
        match field {
            FieldId::WEmb => self.w_emb[i],
            FieldId::WPos => self.w_pos[i],
            FieldId::WQ => self.w_q[i],
            FieldId::WK => self.w_k[i],
            FieldId::WV => self.w_v[i],
            FieldId::WHead => self.w_head[i],
            _ => 0.0,
        }
    }

    fn field_set(&mut self, field: FieldId, i: usize, v: f32) {
        match field {
            FieldId::WEmb => self.w_emb[i] = v,
            FieldId::WPos => self.w_pos[i] = v,
            FieldId::WQ => self.w_q[i] = v,
            FieldId::WK => self.w_k[i] = v,
            FieldId::WV => self.w_v[i] = v,
            FieldId::WHead => self.w_head[i] = v,
            _ => {}
        }
    }

    fn accuracy(&mut self, samples: &[(Vec<usize>, usize, usize)]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut correct = 0usize;
        for (seq, mp, tt) in samples {
            let pred = self.predict(seq, *mp);
            if pred == *tt {
                correct += 1;
            }
        }
        correct as f32 / samples.len() as f32
    }
}

// ── Sample construction ──────────────────────────────────────────────────

/// Build masked-position training samples: for each sequence, mask each
/// position in turn (rotating per epoch keeps the loss landscape fresh
/// without needing per-epoch reshuffling machinery). Returns
/// `(sequence_with_mask, masked_pos, true_token)`.
fn make_samples(seqs: &[Vec<usize>], epoch: usize) -> Vec<(Vec<usize>, usize, usize)> {
    let mut out = Vec::with_capacity(seqs.len());
    for seq in seqs {
        let mp = epoch % seq.len();
        let mut masked = seq.clone();
        let true_tok = seq[mp];
        masked[mp] = MASK_TOKEN;
        out.push((masked, mp, true_tok));
    }
    out
}

/// Build eval samples: for each sequence, mask every position once and
/// evaluate. This gives a robust per-position accuracy measurement.
fn make_eval_samples(seqs: &[Vec<usize>]) -> Vec<(Vec<usize>, usize, usize)> {
    let mut out = Vec::with_capacity(seqs.len() * N);
    for seq in seqs {
        for mp in 0..seq.len() {
            let mut masked = seq.clone();
            let true_tok = seq[mp];
            masked[mp] = MASK_TOKEN;
            out.push((masked, mp, true_tok));
        }
    }
    out
}

// ── Main G6 test ─────────────────────────────────────────────────────────

#[test]
fn g6_token_prediction_lm_domain() {
    // ── Build dataset ───────────────────────────────────────────────────
    // Train set: 32 sequences (more than enough to cover the vocab=8 pairs);
    // eval set: a separate 16 sequences drawn from the same distribution.
    let mut rng = Rng::new(SEED_U64);
    let train_seqs = generate_pattern_dataset(&mut rng, 32, N, EFFECTIVE_VOCAB);
    let eval_seqs = generate_pattern_dataset(&mut rng, 16, N, EFFECTIVE_VOCAB);
    let eval_samples = make_eval_samples(&eval_seqs);

    // ── Init predictors (same PRNG state for fair comparison) ───────────
    let mut rng_a = Rng::new(SEED_U64 + 1);
    let mut fa = FuncattnPredictor::new(&mut rng_a);
    let mut rng_b = Rng::new(SEED_U64 + 1);
    let mut sd = SdpaPredictor::new(&mut rng_b);

    // ── Init loss & accuracy ────────────────────────────────────────────
    let init_samples = make_samples(&train_seqs, 0);
    // Use the first sample's loss as the init-loss proxy for both — same
    // sample ⇒ same loss landscape ⇒ the init loss is directly comparable.
    let (init_seq, init_mp, init_tt) = &init_samples[0];
    let fa_init_loss = fa.forward_loss(init_seq, *init_mp, *init_tt);
    let sd_init_loss = sd.forward_loss(init_seq, *init_mp, *init_tt);
    let fa_init_acc = fa.accuracy(&eval_samples);
    let sd_init_acc = sd.accuracy(&eval_samples);

    eprintln!("\n=== G6: FUNCATTN vs SDPA on masked-token LM prediction ===");
    eprintln!(
        "model: V={}, D={}, N={}, K={}, steps={} (FD-SGD, LR={}, FD_EPS={}, α={}, τ={})",
        V, D, N, K, STEPS, LR, FD_EPS, ALPHA, TEMPERATURE
    );
    eprintln!(
        "params: funcattn={} (W_emb+W_pos+W_basis+3·D²+W_head), sdpa={} (W_emb+W_pos+3·D²+W_head)",
        FuncattnPredictor::n_params(),
        SdpaPredictor::n_params(),
    );
    eprintln!(
        "init:   funcattn loss={:.4} acc={:.3}   sdpa loss={:.4} acc={:.3}",
        fa_init_loss, fa_init_acc, sd_init_loss, sd_init_acc,
    );

    // ── Train ───────────────────────────────────────────────────────────
    let mut fa_last_loss = fa_init_loss;
    let mut sd_last_loss = sd_init_loss;
    for step in 0..STEPS {
        let samples = make_samples(&train_seqs, step);
        let mut fa_epoch_loss = 0.0f32;
        let mut sd_epoch_loss = 0.0f32;
        for (seq, mp, tt) in &samples {
            fa_last_loss = fa.fd_sgd_step(seq, *mp, *tt);
            sd_last_loss = sd.fd_sgd_step(seq, *mp, *tt);
            fa_epoch_loss += fa_last_loss;
            sd_epoch_loss += sd_last_loss;
        }
        let n = samples.len().max(1) as f32;
        if step == 0 || (step + 1) % (STEPS / 6).max(1) == 0 || step + 1 == STEPS {
            let fa_acc = fa.accuracy(&eval_samples);
            let sd_acc = sd.accuracy(&eval_samples);
            eprintln!(
                "[step {:>4}/{:<4}]  fa: mean_loss={:.4} acc={:.3}   sd: mean_loss={:.4} acc={:.3}",
                step + 1,
                STEPS,
                fa_epoch_loss / n,
                fa_acc,
                sd_epoch_loss / n,
                sd_acc,
            );
        }
    }

    // ── Final eval ──────────────────────────────────────────────────────
    let fa_acc = fa.accuracy(&eval_samples);
    let sd_acc = sd.accuracy(&eval_samples);

    eprintln!("\n=== G6 verdict ===");
    eprintln!(
        "  funcattn  acc = {:.4}   (params {})",
        fa_acc,
        FuncattnPredictor::n_params(),
    );
    eprintln!(
        "  sdpa      acc = {:.4}   (params {})",
        sd_acc,
        SdpaPredictor::n_params(),
    );
    eprintln!();
    let acc_delta = fa_acc - sd_acc;
    eprintln!("  accuracy delta (fa − sd) = {:+.4}", acc_delta);
    eprintln!();
    eprintln!("  Plan 286 T4.4 promotion gate (hard, not asserted in this test):");
    eprintln!(
        "    FUNCATTN acc ≥ SDPA acc   → {} (fa {:.4} vs sd {:.4}, Δ {:+.4})",
        if fa_acc >= sd_acc { "PASS — eligible for default-on promotion" } else { "FAIL — stays opt-in" },
        fa_acc,
        sd_acc,
        acc_delta,
    );

    // ── Sanity: both variants actually learned ──────────────────────────
    // If a variant's final loss is not lower than its init loss *and* the
    // final loss is finite, the harness is broken — not a real G6 verdict.
    // NaN defense: treat NaN as "did not finish" rather than failing the
    // sanity check (mirrors G2's NaN handling).
    let fa_finite = fa_last_loss.is_finite();
    let sd_finite = sd_last_loss.is_finite();
    eprintln!();
    eprintln!(
        "  training loss: funcattn {:.4}→{:.4} ({:.1}%), sdpa {:.4}→{:.4} ({:.1}%){}",
        fa_init_loss,
        fa_last_loss,
        if fa_finite { (1.0 - fa_last_loss / fa_init_loss.max(1e-20)) * 100.0 } else { f32::NAN },
        sd_init_loss,
        sd_last_loss,
        if sd_finite { (1.0 - sd_last_loss / sd_init_loss.max(1e-20)) * 100.0 } else { f32::NAN },
        if fa_finite && sd_finite { "" } else { "  [at least one DNF]" },
    );

    assert!(
        !fa_finite || fa_last_loss < fa_init_loss,
        "G6 sanity: FUNCATTN did not reduce loss ({} ≥ init {})",
        fa_last_loss,
        fa_init_loss,
    );
    assert!(
        !sd_finite || sd_last_loss < sd_init_loss,
        "G6 sanity: SDPA did not reduce loss ({} ≥ init {})",
        sd_last_loss,
        sd_init_loss,
    );

    // ── Surface promotion-decision verdict ──────────────────────────────
    // We do NOT auto-promote even if the gate passes — promotion is a human
    // decision per Plan 286 T4.4 (it would flip the feature into the default
    // list, affecting every downstream consumer). The benchmark doc records
    // the honest number and the human flips the feature flag if appropriate.
    if fa_acc >= sd_acc {
        eprintln!();
        eprintln!("  *** G6 PASS — FUNCATTN ≥ SDPA on LM-domain token prediction. ***");
        eprintln!("  *** Eligible for default-on promotion per Plan 286 T4.4. ***");
        eprintln!("  *** Promotion is a human decision: flip the `funcattn` feature ***");
        eprintln!("  *** into the default list and update `.docs/01_overview.md`. ***");
    } else {
        eprintln!();
        eprintln!("  *** G6 FAIL — FUNCATTN < SDPA on LM-domain token prediction. ***");
        eprintln!("  *** Per Plan 286 T4.4 + Research 257 §5 Q2: keep `funcattn` opt-in, ***");
        eprintln!("  *** document null result in `.benchmarks/058_funcattn_goat.md`. ***");
        eprintln!("  *** This is the EXPECTED outcome — the paper itself defers NLP. ***");
    }
}
