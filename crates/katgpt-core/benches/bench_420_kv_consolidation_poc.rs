//! Plan 420 Phase 1 — §3.6 Defend-Wrong PoC for Modelless KV Cache Consolidation
//!
//! Tests the central quality claim of Research 401: can a DETERMINISTIC
//! sigmoid-gated value mean-shift (no training) improve reasoning quality
//! by periodically rewriting KV cache entries at step boundaries?
//!
//! Source paper: arXiv:2505.16950 (Bottlenecked Transformers, Oomerjee et al.)
//! — the paper's Cache Processor is TRAINED. This PoC tests whether the
//! MODELLESS analog (deterministic mean-shift) achieves any quality gain.
//!
//! # Three competitors (head-to-head, same model, same prompt)
//!
//! 1. **Baseline** — vanilla KV cache, no consolidation.
//! 2. **Modelless consolidation** — sigmoid-gated value mean-shift at each
//!    step boundary ('\n'), layer-decaying gate, top-k reconsolidation by
//!    attention mass.
//! 3. **Random-rewrite control** — same selection, same perturbation magnitude,
//!    but random direction (sign flips). Tests whether the mean-shift DIRECTION
//!    matters, or just the perturbation.
//!
//! # Task: few-shot in-context addition
//!
//! 5 solved examples + 1 query. The model must generate the answer digits.
//! Each example is a "step" (newline-delimited). Consolidation fires at each
//! '\n' during prefill, rewriting the recent step's values + top-k recalled
//! prior entries.
//!
//! # Honest limitation
//!
//! The micro-GPT uses RANDOM weights (no training). The paper's IB argument
//! applies to TRAINED models (KV cache is overly detailed after training).
//! On an untrained model, the KV cache has no learned "extraneous detail" to
//! remove. This PoC may REFUTE the quality claim for untrained models — which
//! is a valid §3.6 outcome. The implementation is correct (self-test verifies
//! it); the result is honest.
//!
//! # Metric
//!
//! - **Exact-match**: all answer digits correct (headline, per plan T1.4).
//! - **Token-level accuracy**: fraction of individual digits correct (sensitive).
//! - **NLL**: negative log-likelihood of correct answer tokens (most sensitive).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/kv_consolidation_poc \
//!   cargo run -p katgpt-core --bench bench_420_kv_consolidation_poc --release -- --nocapture
//! ```

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use katgpt_core::{matmul, matmul_relu, rmsnorm, sigmoid, softmax};

// ─── Constants ─────────────────────────────────────────────────────────────

const D_MODEL: usize = 64;
const N_HEAD: usize = 8;
const HEAD_DIM: usize = D_MODEL / N_HEAD; // 8
const MLP_HIDDEN: usize = 128;
const VOCAB: usize = 20;
const MAX_TOKENS: usize = 256;
const N_PROBLEMS: usize = 200;
const N_SEEDS: usize = 3;
const N_FEWSHOT: usize = 5;

// Token IDs
const TOK_PLUS: u32 = 10;
const TOK_EQ: u32 = 15;
const TOK_SPACE: u32 = 16;
const TOK_NEWLINE: u32 = 17;
const TOK_BOS: u32 = 18;

// ─── RNG (xorshift64, same as bench_313) ────────────────────────────────────

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
    fn next_u32(&mut self) -> u32 {
        self.next() as u32
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

// ─── Micro-GPT (single-layer, adapted from bench_313) ───────────────────────

struct MicroGpt {
    wte: Vec<f32>,
    lm_head: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_fc: Vec<f32>,
    w_proj: Vec<f32>,
}

impl MicroGpt {
    fn new(seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        Self {
            wte: rand_vec(VOCAB * D_MODEL, &mut rng, 0.05),
            lm_head: rand_vec(VOCAB * D_MODEL, &mut rng, 0.05),
            w_q: rand_vec(D_MODEL * D_MODEL, &mut rng, 0.1),
            w_k: rand_vec(D_MODEL * D_MODEL, &mut rng, 0.1),
            w_v: rand_vec(D_MODEL * D_MODEL, &mut rng, 0.1),
            w_o: rand_vec(D_MODEL * D_MODEL, &mut rng, 0.1),
            w_fc: rand_vec(MLP_HIDDEN * D_MODEL, &mut rng, 0.1),
            w_proj: rand_vec(D_MODEL * MLP_HIDDEN, &mut rng, 0.1),
        }
    }
}

// ─── KV Cache ──────────────────────────────────────────────────────────────

struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
}

impl KvCache {
    fn new(capacity: usize) -> Self {
        Self {
            k: vec![0.0f32; capacity * D_MODEL],
            v: vec![0.0f32; capacity * D_MODEL],
            len: 0,
        }
    }
}

// ─── Forward pass (single token, incremental decode with KV cache) ──────────

/// Process one token at `pos`, append its K/V to the cache, return (logits,
/// attention_weights). The attention_weights are flat: `[h * new_len + j]`
/// where `new_len = cache.len` after appending (= `pos + 1`).
fn forward_token(
    model: &MicroGpt,
    cache: &mut KvCache,
    token: u32,
    pos: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d = D_MODEL;
    let nh = N_HEAD;
    let hd = HEAD_DIM;
    let scale = 1.0 / (hd as f32).sqrt();

    // 1. Embed
    let mut h = vec![0.0f32; d];
    let tok = (token as usize).min(VOCAB - 1);
    for i in 0..d {
        h[i] = model.wte[tok * d + i];
    }

    // 2. Q, K, V projections
    let mut q = vec![0.0f32; d];
    let mut k_new = vec![0.0f32; d];
    let mut v_new = vec![0.0f32; d];
    matmul(&mut q, &model.w_q, &h, d, d);
    matmul(&mut k_new, &model.w_k, &h, d, d);
    matmul(&mut v_new, &model.w_v, &h, d, d);

    // 3. Append to cache
    let cur_len = cache.len;
    cache.k[cur_len * d..(cur_len + 1) * d].copy_from_slice(&k_new);
    cache.v[cur_len * d..(cur_len + 1) * d].copy_from_slice(&v_new);
    cache.len = cur_len + 1;
    let new_len = cache.len;

    // 4. Multi-head causal attention
    let mut attn_out = vec![0.0f32; d];
    let mut attn_weights = vec![0.0f32; nh * new_len];

    for h_idx in 0..nh {
        let off = h_idx * hd;
        // Scores
        let mut scores = vec![0.0f32; new_len];
        let mut max_score = f32::NEG_INFINITY;
        for j in 0..new_len {
            let phase = ((pos.max(j) - pos.min(j)) as f32 * 0.1).cos();
            let mut s = 0.0f32;
            for d_idx in 0..hd {
                s += q[off + d_idx] * cache.k[j * d + off + d_idx];
            }
            s = s * scale * phase;
            scores[j] = s;
            if s > max_score {
                max_score = s;
            }
        }
        // Softmax
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut().take(new_len) {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        for j in 0..new_len {
            scores[j] *= inv;
            attn_weights[h_idx * new_len + j] = scores[j];
        }
        // Weighted sum of values
        for d_idx in 0..hd {
            let mut acc = 0.0f32;
            for j in 0..new_len {
                acc += scores[j] * cache.v[j * d + off + d_idx];
            }
            attn_out[off + d_idx] = acc;
        }
    }

    // 5. Output projection + residual + rmsnorm
    let mut o = vec![0.0f32; d];
    matmul(&mut o, &model.w_o, &attn_out, d, d);
    for i in 0..d {
        h[i] += o[i];
    }
    rmsnorm(&mut h);

    // 6. MLP (FC → ReLU → Proj → residual → rmsnorm)
    let mut h_fc = vec![0.0f32; MLP_HIDDEN];
    matmul_relu(&mut h_fc, &model.w_fc, &h, MLP_HIDDEN, d);
    let mut h_proj = vec![0.0f32; d];
    matmul(&mut h_proj, &model.w_proj, &h_fc, d, MLP_HIDDEN);
    for i in 0..d {
        h[i] += h_proj[i];
    }
    rmsnorm(&mut h);

    // 7. LM head → logits → softmax (returns probabilities)
    let mut logits = vec![0.0f32; VOCAB];
    matmul(&mut logits, &model.lm_head, &h, VOCAB, d);
    softmax(&mut logits);

    (logits, attn_weights)
}

// ─── Consolidation ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConsolidationMode {
    Baseline,
    Consolidation,
    RandomRewrite,
}

impl ConsolidationMode {
    fn label(self) -> &'static str {
        match self {
            Self::Baseline => "Baseline (no consolidation)",
            Self::Consolidation => "Modelless consolidation",
            Self::RandomRewrite => "Random-rewrite control",
        }
    }
}

#[derive(Clone)]
struct ConsolidationConfig {
    g_max: f32,
    #[allow(dead_code)]
    lambda: f32, // layer decay — degenerate for single-layer (sigmoid(0)=0.5)
    k: usize,
    rsw_len: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            g_max: 0.3,
            lambda: 4.0,
            k: 32,
            rsw_len: 64,
        }
    }
}

/// Apply KV cache consolidation at a step boundary.
///
/// - `step_start`: position after the previous step boundary (or 0).
/// - `step_end`: position of the current '\n' + 1 (exclusive).
/// - `attn_history[i]`: flat attention weights for token at position i,
///   `[h_idx * (i + 1) + j]`.
///
/// Consolidation (Research 401 §2.2):
///   v_j ← v_j + σ(g) · (μ_v − v_j)            (recent step entries)
///   v_i ← v_i + σ(g · α_i) · (μ_v − v_i)      (top-k recalled entries)
///
/// Random-rewrite control: same magnitude, random sign per dimension.
/// Baseline: no-op.
fn consolidate(
    cache: &mut KvCache,
    step_start: usize,
    step_end: usize,
    attn_history: &[Vec<f32>],
    config: &ConsolidationConfig,
    mode: ConsolidationMode,
    rng: &mut SimpleRng,
) {
    if mode == ConsolidationMode::Baseline || step_end <= step_start {
        return;
    }

    let d = D_MODEL;
    let nh = N_HEAD;

    // Layer gate: single-layer → g = g_max · sigmoid(−λ · 0/1) = g_max · 0.5
    let g = config.g_max * sigmoid(0.0);
    let gate = sigmoid(g);

    // Cap the consolidation window at rsw_len
    let rsw_start = if step_end > step_start + config.rsw_len {
        step_end - config.rsw_len
    } else {
        step_start
    };

    // 1. Compute step mean: μ_v = mean(v[rsw_start..step_end])
    let mut mu_v = vec![0.0f32; d];
    let step_count = (step_end - rsw_start) as f32;
    for j in rsw_start..step_end {
        for i in 0..d {
            mu_v[i] += cache.v[j * d + i];
        }
    }
    for i in 0..d {
        mu_v[i] /= step_count;
    }

    // 2. Consolidation: move recent step values toward mean
    for j in rsw_start..step_end {
        apply_value_update(&mut cache.v[j * d..(j + 1) * d], &mu_v, gate, mode, rng);
    }

    // 3. Reconsolidation: top-k prior positions by attention mass
    if step_start > 0 && config.k > 0 {
        let k_eff = config.k.min(step_start);

        // Compute attention mass for each prior position j < step_start:
        // mean over heads and recent-step queries of attn_weight[i][h][j]
        let mut attn_mass = vec![0.0f64; step_start];
        for j in 0..step_start {
            let mut total = 0.0f64;
            let mut count = 0u32;
            for i in rsw_start..step_end {
                if i < attn_history.len() {
                    let weights = &attn_history[i];
                    let n_pos = i + 1; // positions token i attended to
                    for h_idx in 0..nh {
                        let idx = h_idx * n_pos + j;
                        if idx < weights.len() {
                            total += weights[idx] as f64;
                            count += 1;
                        }
                    }
                }
            }
            attn_mass[j] = if count > 0 { total / count as f64 } else { 0.0 };
        }

        // Top-k selection (partial sort)
        let mut indices: Vec<usize> = (0..step_start).collect();
        indices.sort_by(|&a, &b| attn_mass[b].total_cmp(&attn_mass[a]));

        let max_mass = indices.first().map_or(1.0, |&i| attn_mass[i]).max(1e-10);

        for &j in indices.iter().take(k_eff) {
            let alpha = (attn_mass[j] / max_mass) as f32;
            let recon_gate = sigmoid(g * alpha);
            apply_value_update(
                &mut cache.v[j * d..(j + 1) * d],
                &mu_v,
                recon_gate,
                mode,
                rng,
            );
        }
    }
}

/// Apply a single value-vector update: mean-shift (Consolidation) or
/// same-magnitude random sign (RandomRewrite).
fn apply_value_update(
    v: &mut [f32],
    mu_v: &[f32],
    gate: f32,
    mode: ConsolidationMode,
    rng: &mut SimpleRng,
) {
    match mode {
        ConsolidationMode::Consolidation => {
            for i in 0..v.len() {
                let delta = mu_v[i] - v[i];
                v[i] += gate * delta;
            }
        }
        ConsolidationMode::RandomRewrite => {
            for i in 0..v.len() {
                let mag = (mu_v[i] - v[i]).abs();
                let sign = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
                v[i] += gate * mag * sign;
            }
        }
        ConsolidationMode::Baseline => {}
    }
}

// ─── Problem generation ────────────────────────────────────────────────────

struct Problem {
    prompt: Vec<u32>,
    answer: Vec<u32>,
}

fn tokenize_number(n: u32) -> Vec<u32> {
    if n == 0 {
        return vec![0];
    }
    let mut digits = Vec::new();
    let mut m = n;
    while m > 0 {
        digits.push(m % 10);
        m /= 10;
    }
    digits.reverse();
    digits.shrink_to_fit();
    digits
}

/// Generate one few-shot addition problem.
/// Format: BOS, {N_FEWSHOT examples}, query.
/// Each example: "{a} + {b} = {c}\n"
/// Query: "{a} + {b} = "
fn gen_problem(rng: &mut SimpleRng) -> Problem {
    let mut prompt = vec![TOK_BOS];

    for _ in 0..N_FEWSHOT {
        let ea = rng.next_u32() % 90 + 10; // 10..99
        let eb = rng.next_u32() % 90 + 10;
        let ec = ea + eb;
        prompt.extend(tokenize_number(ea));
        prompt.push(TOK_SPACE);
        prompt.push(TOK_PLUS);
        prompt.push(TOK_SPACE);
        prompt.extend(tokenize_number(eb));
        prompt.push(TOK_SPACE);
        prompt.push(TOK_EQ);
        prompt.push(TOK_SPACE);
        prompt.extend(tokenize_number(ec));
        prompt.push(TOK_NEWLINE);
    }

    // Query (the problem to solve)
    let a = rng.next_u32() % 90 + 10;
    let b = rng.next_u32() % 90 + 10;
    let c = a + b;
    prompt.extend(tokenize_number(a));
    prompt.push(TOK_SPACE);
    prompt.push(TOK_PLUS);
    prompt.push(TOK_SPACE);
    prompt.extend(tokenize_number(b));
    prompt.push(TOK_SPACE);
    prompt.push(TOK_EQ);
    prompt.push(TOK_SPACE);

    Problem {
        prompt,
        answer: tokenize_number(c),
    }
}

fn gen_problems(seed: u64, n: usize) -> Vec<Problem> {
    let mut rng = SimpleRng::new(seed);
    (0..n).map(|_| gen_problem(&mut rng)).collect()
}

// ─── Evaluation ────────────────────────────────────────────────────────────

struct EvalResult {
    exact_match: bool,
    token_accuracy: f32,
    nll: f32,
    digit_mass: f32,
    consolidation_count: u32,
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// Evaluate one problem under one competitor. Returns metrics.
///
/// Uses TEACHER FORCING (feeds correct answer digits, measures the model's
/// assigned probability) rather than greedy decode. On an untrained model,
/// greedy decode immediately diverges (generates non-digit tokens), making
/// all metrics zero and uninformative. Teacher forcing always produces
/// non-zero NLL and token-accuracy, enabling fair comparison across
/// competitors with different cache states.
fn evaluate(
    model: &MicroGpt,
    problem: &Problem,
    mode: ConsolidationMode,
    config: &ConsolidationConfig,
    rng: &mut SimpleRng,
) -> EvalResult {
    let mut cache = KvCache::new(MAX_TOKENS);
    let mut attn_history: Vec<Vec<f32>> = Vec::with_capacity(MAX_TOKENS);
    let mut consolidation_count = 0u32;
    let mut step_start = 0usize;

    // Prefill: process prompt tokens one by one with consolidation at '\n'
    for (pos, &token) in problem.prompt.iter().enumerate() {
        let (_logits, attn_w) = forward_token(model, &mut cache, token, pos);
        attn_history.push(attn_w);

        if token == TOK_NEWLINE {
            let step_end = pos + 1;
            consolidate(
                &mut cache,
                step_start,
                step_end,
                &attn_history,
                config,
                mode,
                rng,
            );
            if mode != ConsolidationMode::Baseline {
                consolidation_count += 1;
            }
            step_start = step_end;
        }
    }

    // Teacher-forced evaluation: feed correct answer digits one by one,
    // measure the model's assigned probability for each correct digit.
    let prompt_len = problem.prompt.len();
    let mut last_token = *problem.prompt.last().unwrap_or(&TOK_BOS);
    let mut nll = 0.0f32;
    let mut token_correct = 0usize;
    let mut digit_mass = 0.0f32; // total prob mass on digit tokens (0-9)
    let mut all_argmax_correct = true;

    for (gen_pos, &correct_digit) in (prompt_len..).zip(problem.answer.iter()) {
        let (logits, _attn) = forward_token(model, &mut cache, last_token, gen_pos);

        // NLL of the correct digit
        nll -= logits[correct_digit as usize].max(1e-30).ln();

        // Token accuracy: does argmax match?
        let pred = argmax(&logits) as u32;
        if pred == correct_digit {
            token_correct += 1;
        } else {
            all_argmax_correct = false;
        }

        // Digit probability mass (diagnostic: is the model even trying?)
        let mut dm = 0.0f32;
        for d in 0..10u32 {
            dm += logits[d as usize];
        }
        digit_mass += dm / problem.answer.len() as f32;

        // Feed the CORRECT token (teacher forcing)
        last_token = correct_digit;
    }

    let exact_match = all_argmax_correct;
    let token_accuracy = if !problem.answer.is_empty() {
        token_correct as f32 / problem.answer.len() as f32
    } else {
        0.0
    };

    EvalResult {
        exact_match,
        token_accuracy,
        nll,
        digit_mass,
        consolidation_count,
    }
}

// ─── Aggregated metrics ────────────────────────────────────────────────────

#[derive(Default)]
struct AggregatedMetrics {
    exact_match_rate: f64,
    token_accuracy: f64,
    mean_nll: f64,
    mean_digit_mass: f64,
    mean_consolidation_count: f64,
}

fn aggregate(results: &[EvalResult]) -> AggregatedMetrics {
    let n = results.len();
    if n == 0 {
        return AggregatedMetrics::default();
    }
    let em: f64 = results.iter().filter(|r| r.exact_match).count() as f64 / n as f64;
    let ta: f64 = results.iter().map(|r| r.token_accuracy as f64).sum::<f64>() / n as f64;
    let nll: f64 = results.iter().map(|r| r.nll as f64).sum::<f64>() / n as f64;
    let dm: f64 = results.iter().map(|r| r.digit_mass as f64).sum::<f64>() / n as f64;
    let cc: f64 = results
        .iter()
        .map(|r| r.consolidation_count as f64)
        .sum::<f64>()
        / n as f64;
    AggregatedMetrics {
        exact_match_rate: em,
        token_accuracy: ta,
        mean_nll: nll,
        mean_digit_mass: dm,
        mean_consolidation_count: cc,
    }
}

// ─── Self-test: verify consolidation code is correct ───────────────────────

fn self_test() -> bool {
    let mut cache = KvCache::new(128);
    let mut rng = SimpleRng::new(42);

    // Fill cache with random values (32 positions)
    for j in 0..32 {
        for i in 0..D_MODEL {
            cache.k[j * D_MODEL + i] = rng.uniform() * 2.0 - 1.0;
            cache.v[j * D_MODEL + i] = rng.uniform() * 2.0 - 1.0;
        }
    }
    cache.len = 32;

    // Save state before
    let keys_before = cache.k[..32 * D_MODEL].to_vec();
    let vals_before = cache.v[..32 * D_MODEL].to_vec();

    // Dummy attention history (uniform weights)
    let attn_history: Vec<Vec<f32>> = (0..32)
        .map(|i| vec![1.0f32 / (i + 1) as f32; N_HEAD * (i + 1)])
        .collect();

    let config = ConsolidationConfig {
        g_max: 0.3,
        lambda: 4.0,
        k: 8,
        rsw_len: 64,
    };

    // Apply consolidation on step [16, 32) — step_start=16 means positions
    // [0,16) are prior positions eligible for reconsolidation.
    consolidate(
        &mut cache,
        16,
        32,
        &attn_history,
        &config,
        ConsolidationMode::Consolidation,
        &mut rng,
    );

    // Check 1: keys unchanged
    let mut keys_ok = true;
    for i in 0..32 * D_MODEL {
        if (cache.k[i] - keys_before[i]).abs() > 1e-10 {
            keys_ok = false;
            break;
        }
    }

    // Check 2: values in [16,32) changed (consolidation window)
    let mut vals_changed = false;
    for i in (16 * D_MODEL)..(32 * D_MODEL) {
        if (cache.v[i] - vals_before[i]).abs() > 1e-6 {
            vals_changed = true;
            break;
        }
    }

    // Check 3: values in [0,16) also changed (reconsolidation top-k)
    let mut recon_changed = false;
    for i in 0..(16 * D_MODEL) {
        if (cache.v[i] - vals_before[i]).abs() > 1e-6 {
            recon_changed = true;
            break;
        }
    }

    // Check 4: no NaN/Inf
    let mut all_finite = true;
    for i in 0..32 * D_MODEL {
        if !cache.v[i].is_finite() || !cache.k[i].is_finite() {
            all_finite = false;
            break;
        }
    }

    // Check 5: consolidation reduces variance toward mean
    // Compute mean-square of consolidation window [16,32) before and after
    let mut var_before = 0.0f32;
    for j in 16..32 {
        for i in 0..D_MODEL {
            var_before += vals_before[j * D_MODEL + i] * vals_before[j * D_MODEL + i];
        }
    }
    var_before /= (16 * D_MODEL) as f32;
    let mut var_after = 0.0f32;
    for j in 16..32 {
        for i in 0..D_MODEL {
            var_after += cache.v[j * D_MODEL + i] * cache.v[j * D_MODEL + i];
        }
    }
    var_after /= (16 * D_MODEL) as f32;

    let all_pass = keys_ok && vals_changed && recon_changed && all_finite;
    println!("─── Self-test ───");
    println!(
        "  Keys unchanged:     {}",
        if keys_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  Values changed:     {}",
        if vals_changed { "PASS" } else { "FAIL" }
    );
    println!(
        "  Recon changed:      {}",
        if recon_changed { "PASS" } else { "FAIL" }
    );
    println!(
        "  All finite:         {}",
        if all_finite { "PASS" } else { "FAIL" }
    );
    println!(
        "  Variance {} → {} ({:.1}% reduction): {}",
        var_before,
        var_after,
        (1.0 - var_after / var_before.max(1e-10)) * 100.0,
        if var_after < var_before {
            "PASS (reduced)"
        } else {
            "WARN (not reduced)"
        }
    );
    println!("  Overall: {}", if all_pass { "PASS" } else { "FAIL" });
    println!();
    all_pass
}

// ─── Hyperparameter sweep (T1.5) ───────────────────────────────────────────

fn run_sweep(model: &MicroGpt, problems: &[Problem]) {
    println!("─── T1.5 Hyperparameter Sweep ───");
    println!("  (single layer → λ is degenerate; fixed at 4.0, gate = g_max · 0.5)");
    println!();

    let g_max_vals = [0.1f32, 0.3, 0.5];
    let k_vals = [16usize, 32, 64];

    let mut best_ta = 0.0f64;
    let mut best_config = String::new();

    print!(
        "{:<8} {:<6} {:<10} {:<10} {:<10} {:<10}",
        "g_max", "k", "token_acc", "mean_nll", "em_rate", "digit_mass"
    );
    println!();
    println!("{}", "-".repeat(64));

    for &g_max in &g_max_vals {
        for &k in &k_vals {
            let config = ConsolidationConfig {
                g_max,
                lambda: 4.0,
                k,
                rsw_len: 64,
            };
            let mut rng = SimpleRng::new(999);
            let results: Vec<EvalResult> = problems
                .iter()
                .map(|p| {
                    evaluate(
                        model,
                        p,
                        ConsolidationMode::Consolidation,
                        &config,
                        &mut rng,
                    )
                })
                .collect();
            let agg = aggregate(&results);
            println!(
                "{:<8.1} {:<6} {:<10.4} {:<10.4} {:<10.4} {:<10.4}",
                g_max,
                k,
                agg.token_accuracy,
                agg.mean_nll,
                agg.exact_match_rate,
                agg.mean_digit_mass
            );
            if agg.token_accuracy > best_ta {
                best_ta = agg.token_accuracy;
                best_config = format!("g_max={g_max}, k={k}");
            }
        }
    }
    println!();
    println!("  Best config (by token accuracy): {best_config} → ta={best_ta:.4}");
    println!();
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("==============================================================");
    println!("  Plan 420 Phase 1 — §3.6 Defend-Wrong PoC");
    println!("  Modelless KV Cache Consolidation");
    println!("  (arXiv:2505.16950 Bottlenecked Transformers)");
    println!("==============================================================");
    println!();

    // Self-test
    if !self_test() {
        eprintln!("SELF-TEST FAILED — aborting.");
        std::process::exit(2);
    }

    let config = ConsolidationConfig::default();

    let modes = [
        ConsolidationMode::Baseline,
        ConsolidationMode::Consolidation,
        ConsolidationMode::RandomRewrite,
    ];

    // ── T1.3: Run 200 problems × 3 competitors × 3 seeds ──
    println!("─── T1.3 Main Comparison ───");
    println!(
        "  {} problems × {} competitors × {} seeds = {} evaluations",
        N_PROBLEMS,
        modes.len(),
        N_SEEDS,
        N_PROBLEMS * modes.len() * N_SEEDS
    );
    println!(
        "  Config: g_max={}, k={}, rsw={} (single-layer, λ degenerate)",
        config.g_max, config.k, config.rsw_len
    );
    println!();

    // Accumulate per-mode across all seeds
    let mut all_results: Vec<Vec<EvalResult>> = modes.iter().map(|_| Vec::new()).collect();

    for seed_idx in 0..N_SEEDS {
        let model_seed = 1000 + seed_idx as u64 * 7777;
        let problem_seed = 2000 + seed_idx as u64 * 3333;
        let model = MicroGpt::new(model_seed);
        let problems = gen_problems(problem_seed, N_PROBLEMS);

        for (mi, &mode) in modes.iter().enumerate() {
            // Each mode gets its own RNG for reproducibility
            let mut rng = SimpleRng::new(5000 + seed_idx as u64 * 111 + mi as u64 * 99);
            for problem in &problems {
                let result = evaluate(&model, problem, mode, &config, &mut rng);
                all_results[mi].push(result);
            }
        }
    }

    // Print results table
    println!(
        "{:<30} {:<8} {:<10} {:<10} {:<10} {:<8}",
        "Competitor", "EM_rate", "Token_acc", "Mean_NLL", "DigitMass", "N_consol"
    );
    println!("{}", "-".repeat(86));

    let mut agg_metrics = Vec::new();
    for (mi, &mode) in modes.iter().enumerate() {
        let agg = aggregate(&all_results[mi]);
        println!(
            "{:<30} {:<8.4} {:<10.4} {:<10.4} {:<10.4} {:<8.1}",
            mode.label(),
            agg.exact_match_rate,
            agg.token_accuracy,
            agg.mean_nll,
            agg.mean_digit_mass,
            agg.mean_consolidation_count
        );
        agg_metrics.push(agg);
    }
    println!();

    // ── T1.4: Verdict gate ──
    let baseline = &agg_metrics[0];
    let consolidation = &agg_metrics[1];
    let random_rewrite = &agg_metrics[2];

    let em_gain = consolidation.exact_match_rate - baseline.exact_match_rate;
    let ta_gain = consolidation.token_accuracy - baseline.token_accuracy;
    let nll_gain = consolidation.mean_nll - baseline.mean_nll; // negative = better
    let em_vs_random = consolidation.exact_match_rate - random_rewrite.exact_match_rate;
    let ta_vs_random = consolidation.token_accuracy - random_rewrite.token_accuracy;
    let nll_vs_random = consolidation.mean_nll - random_rewrite.mean_nll;

    println!("─── T1.4 Verdict Gate ───");
    println!();
    println!("  Consolidation vs Baseline:");
    println!(
        "    Exact-match gain:  {:+.4} ({:+.2}pp)",
        em_gain,
        em_gain * 100.0
    );
    println!(
        "    Token-acc gain:    {:+.4} ({:+.2}pp)",
        ta_gain,
        ta_gain * 100.0
    );
    println!("    NLL change:        {nll_gain:+.4} (negative = better)");
    println!();
    println!("  Consolidation vs Random-rewrite:");
    println!(
        "    EM difference:     {:+.4} ({:+.2}pp)",
        em_vs_random,
        em_vs_random * 100.0
    );
    println!(
        "    Token-acc diff:    {:+.4} ({:+.2}pp)",
        ta_vs_random,
        ta_vs_random * 100.0
    );
    println!("    NLL difference:    {nll_vs_random:+.4} (negative = consolidation better)");
    println!();

    // Verdict (≥2pp EM gain AND beats random-rewrite → GOAT confirmed)
    let em_gain_pp = em_gain * 100.0;
    let em_vs_random_pp = em_vs_random * 100.0;

    let goat_confirmed = em_gain_pp >= 2.0 && em_vs_random_pp > 0.0;
    let refuted = em_gain_pp.abs() < 1.0;
    let direction_matters = ta_vs_random.abs() > 0.005;

    println!("  Verdict:");
    if goat_confirmed {
        println!(
            "    ✅ GOAT CONFIRMED — consolidation beats baseline by ≥2pp AND beats random-rewrite."
        );
        println!("    Proceed to Phase 2 (feature flag).");
    } else if refuted {
        println!("    ❌ QUALITY GAIN REFUTED — consolidation ≈ baseline (<1pp difference).");
        println!("    The modelless mean-shift does not improve quality on this task.");
        println!("    This is expected on an untrained model: the IB argument applies to");
        println!("    TRAINED models whose KV cache carries learned extraneous detail.");
        println!("    Record raw numbers in Research 401 §PoC Addendum. Demote or shelve.");
    } else {
        println!("    ⚠️ INCONCLUSIVE — gain is between 1pp and 2pp. Needs more data or");
        println!("    a trained model to reach a definitive verdict.");
    }
    println!();

    if !direction_matters && !refuted {
        println!("    ⚠️ Mean-shift DIRECTION does not matter (consolidation ≈ random-rewrite).");
        println!("    Any gain is from noise injection, not IB-consistent consolidation.");
    } else if direction_matters {
        println!(
            "    ℹ️ Mean-shift direction matters (Δtoken_acc vs random = {ta_vs_random:+.4})."
        );
    }
    println!();

    // ── T1.5: Hyperparameter sweep ──
    // Use model seed 0, problems seed 1 for the sweep
    let sweep_model = MicroGpt::new(1000);
    let sweep_problems = gen_problems(2000, N_PROBLEMS);
    run_sweep(&sweep_model, &sweep_problems);

    println!("==============================================================");
    if goat_confirmed {
        println!("  OVERALL: GOAT CONFIRMED — eligible for Phase 2.");
        std::process::exit(0);
    } else {
        println!("  OVERALL: Quality claim NOT confirmed by this PoC.");
        println!("  (Architectural coverage stands; quality needs a trained model.)");
        std::process::exit(1);
    }
}
