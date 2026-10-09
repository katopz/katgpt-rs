#![cfg(all(feature = "kimi_k3_loader", feature = "loop_alignment_probe"))]
//! Issue 929 T1.2/T1.3 — real-checkpoint loop-alignment probe runner.
//!
//! Runs the DiscoLoop alignment probe (Research 614, arXiv:2607.00341) over
//! the REAL kimi-k3-0.40B checkpoint through a caller-side loop: prefill the
//! fixture prompt, then at the answer position re-enter the decoder stack
//! (`kimi_k3_forward_token_hidden` — the Ouro-recurrence surface, training
//! twin in `src/kimi_k3/backward.rs` Plan 324 A4), probing
//! `cos(H, W[·])` + margins at every loop.
//!
//! Two loop arms per query, both recorded (neither changes any shipped
//! path — this binary is measurement-only):
//! - arm 0 `reentry` (α=0): the state re-enters unchanged; the KDA
//!   recurrent state still advances, so loop 2 differs from loop 1.
//! - arm 1 `reinject` (α=0.5): the paper's hard re-anchor Eq. 3 —
//!   `H ← (1−α)·H + α·RMSNorm(W[v̂])`.
//!
//! The checkpoint is UNTIED (`language_model.lm_head.weight` is a required
//! separate tensor — the loader refuses without it), so head-row reads are
//! the true readout operator (the issue's tied-embeddings precondition).
//!
//! Report-only: prints the per-arm/per-split AUROC table + one JSON line.
//! The T1.4 kill-criterion adjudication is recorded in the issue, not
//! enforced here.
//!
//! Run (release mandatory — debug is ~10× slower per forward):
//! ```text
//! cargo run --release --features kimi_k3_loader,loop_alignment_probe \
//!   --example issue929_kimi_loop_alignment_probe
//! ```
//! Checkpoint dir: `KIMI_K3_MODEL_DIR` env, else
//! `{CARGO_MANIFEST_DIR}/data/kimi-k3-0.40b`.

use katgpt_core::loop_alignment_probe::{
    bootstrap_auroc_ci, generate_two_hop_fixture, probe_alignment, FixtureSpec,
};
use katgpt_rs::kimi_k3::loader::{load_kimi_k3, KimiK3ModelWeights};
use katgpt_rs::kimi_k3::model::{
    kimi_k3_forward_token, kimi_k3_forward_token_hidden, KimiK3ModelConfig, KimiK3Runtime,
};
use katgpt_rs::kimi_k3::tiktoken::{load_tiktoken_bpe, TiktokenTokenizer};
use katgpt_core::simd::{simd_dot_f32, simd_sum_sq};

/// Prompt facts per query (gold facts + same-pool distractors). Trimmed from
/// the default 24 to keep the prefill bill at ~150 tokens/query.
const FACTS_PER_PROMPT: usize = 12;
/// Two-hop queries per pool (the hard class).
const TWO_HOP_PER_POOL: usize = 60;
/// One-hop controls per pool (the easy class — AUROC needs both).
const ONE_HOP_PER_POOL: usize = 30;
/// Caller-side loop depth at the answer position (loop 0 = the prefill read).
const LOOPS: usize = 2;
/// Hard re-anchor strength (the paper's α≈0.5 arm).
const ALPHA: f32 = 0.5;
/// Greedy continuation tokens generated for the answer check.
const ANSWER_TOKENS: usize = 8;
/// Bootstrap iterations per reported CI.
const BOOTSTRAP_ITERS: usize = 400;

/// One probe row at one loop of one query.
#[derive(Clone, Copy, Debug)]
struct Row {
    correct: bool,
    cos_argmax: f32,
    margin: f32,
    cos_bridge: f32,
    bridge_top1: bool,
    /// The loop's argmax token — feeds the next loop's reinject.
    argmax_row: usize,
}

/// One fixture query with its tokenized prompt.
struct Query {
    is_two_hop: bool,
    is_ood: bool,
    tokens: Vec<u32>,
    bridge_first: u32,
    answer: String,
}

type ArmRows = Vec<Vec<Vec<Row>>>;
type SplitFilter = fn(&Query) -> bool;

fn main() {
    let t0 = std::time::Instant::now();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR")
        .unwrap_or_else(|_| format!("{manifest_dir}/data/kimi-k3-0.40b"));

    // ── Load tokenizer + weights ─────────────────────────────────────────
    let tiktoken_bytes =
        std::fs::read(format!("{model_dir}/tiktoken.model")).expect("tiktoken.model");
    let ranks = load_tiktoken_bpe(&tiktoken_bytes).expect("tiktoken ranks");
    let tokenizer = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);
    eprintln!(
        "[load] tokenizer vocab={} bos={} eos={}",
        tokenizer.vocab_size(),
        tokenizer.bos_id(),
        tokenizer.eos_id()
    );

    let weights = load_kimi_k3(&format!("{model_dir}/model.safetensors"))
        .expect("model.safetensors");
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let dim = config.hidden_size;
    eprintln!(
        "[load] weights in {:.1}s (untied head: {} rows × {dim})",
        t0.elapsed().as_secs_f32(),
        weights.lm_head_weight.len() / dim,
    );

    // ── Fixture (paper §2 shape; deterministic) ──────────────────────────
    let spec = FixtureSpec {
        facts_per_prompt: FACTS_PER_PROMPT,
        two_hop_queries: TWO_HOP_PER_POOL,
        one_hop_queries: ONE_HOP_PER_POOL,
        ..FixtureSpec::default()
    };
    let fixture = generate_two_hop_fixture(spec);
    let two_hop_n = fixture.items.iter().filter(|i| i.is_two_hop).count();
    eprintln!(
        "[fixture] {} queries ({two_hop_n} two-hop; unique comps/pool={})",
        fixture.items.len(),
        fixture.unique_compositions_per_pool,
    );

    let queries: Vec<Query> = fixture
        .items
        .iter()
        .map(|item| Query {
            is_two_hop: item.is_two_hop,
            is_ood: item.is_ood,
            tokens: {
                let mut t: Vec<u32> = tokenizer
                    .encode(&item.prompt)
                    .into_iter()
                    .map(|x| x as u32)
                    .collect();
                t.insert(0, 1); // bos — the checkpoint is completion-style
                t
            },
            bridge_first: if item.bridge.is_empty() {
                0
            } else {
                tokenizer.encode(&item.bridge).first().copied().unwrap_or(0) as u32
            },
            answer: item.answer.clone(),
        })
        .collect();
    let queries_max_prompt = queries
        .iter()
        .map(|q| q.tokens.len())
        .max()
        .unwrap_or(0);
    eprintln!("[fixture] max prompt {queries_max_prompt} tokens");

    let max_extra = LOOPS + ANSWER_TOKENS; // per-arm appends: last token +
    // (LOOPS-1) re-entries + ANSWER_TOKENS continuation, with slack.

    // ── Probe loop ───────────────────────────────────────────────────
    // rows[arm][loop] — one Row per query, query order preserved.
    let mut rows: ArmRows = vec![vec![Vec::new(); LOOPS]; 2];

    for (qi, q) in queries.iter().enumerate() {
        // Each (query, arm) pair runs on a FRESH runtime: the KDA recurrent
        // state and the MLA KV cache must not carry across arms (arm 1
        // inheriting arm 0's continuation positions would contaminate the
        // comparison). The prefill is redone per arm — honest, and cheap
        // relative to the loop passes.
        let runtime_len = q.tokens.len() + max_extra;
        for arm in 0..2usize {
            let alpha = [0.0f32, ALPHA][arm];
            let mut runtime = KimiK3Runtime::new(&config, runtime_len);
            // Prefill all but the last prompt token (no probing — the paper
            // probes the ANSWER position).
            for &t in &q.tokens[..q.tokens.len() - 1] {
                kimi_k3_forward_token(&config, &weights, &mut runtime, t);
            }
            // Loop 0 at the answer position: the last prompt token.
            let last = *q.tokens.last().expect("non-empty prompt");
            kimi_k3_forward_token(&config, &weights, &mut runtime, last);
            for k in 0..LOOPS {
                if k > 0 {
                    // H ← (1−α)·H + α·RMSNorm(W[v̂]) from the previous loop's
                    // argmax, then re-enter the stack.
                    let v = rows[arm][k - 1][qi].argmax_row;
                    reinject_hidden(&mut runtime, &weights, v, alpha, dim);
                    kimi_k3_forward_token_hidden(&config, &weights, &mut runtime);
                }
                let probe =
                    probe_alignment(&runtime.hidden, &runtime.logits, &weights.lm_head_weight, dim);
                let cos_bridge =
                    cosine_rows(&runtime.hidden, &weights.lm_head_weight, q.bridge_first, dim);
                // Correctness: greedy-continue THROUGH the runtime, then
                // decode. The continuation mutates the runtime state — fine,
                // this is the loop's LAST observation for this arm.
                let correct = if k == LOOPS - 1 {
                    let logits_owned = runtime.logits.clone();
                    let text = greedy_continue(
                        &config, &weights, &mut runtime, &tokenizer, &logits_owned,
                    );
                if qi < 3 && arm == 0 {
                        eprintln!(
                            "[dbg] q{qi} answer={:?} gen={:?}",
                            q.answer,
                            text.replace('\n', " ")
                        );
                    }
                    text.contains(&q.answer)
                } else {
                    false
                };
                rows[arm][k].push(Row {
                    correct,
                    cos_argmax: probe.cos_alignment,
                    margin: probe.margin,
                    cos_bridge,
                    bridge_top1: probe.argmax == q.bridge_first as usize,
                    argmax_row: probe.argmax,
                });
            }
        }
        if (qi + 1) % 20 == 0 {
            eprintln!(
                "[probe] {}/{} queries ({:.1}s)",
                qi + 1,
                queries.len(),
                t0.elapsed().as_secs_f32()
            );
        }
    }

    // ── Report ───────────────────────────────────────────────────────────
    println!("# issue929 kimi-k3-0.40B loop-alignment probe");
    println!(
        "# fixture: {} queries, facts_per_prompt={FACTS_PER_PROMPT}, loops={LOOPS}, alpha_reinject={ALPHA}",
        queries.len()
    );
    println!("# arms: 0=reentry(alpha=0) 1=reinject(alpha={ALPHA}); score cells are point/lb95");
    println!("arm,loop,split,n,acc,bridge_top1,auroc_cos_argmax,auroc_margin,auroc_cos_bridge");
    let splits: [(&str, SplitFilter); 3] = [
        ("all", |_| true),
        ("two_hop", |q: &Query| q.is_two_hop),
        ("ood", |q: &Query| q.is_ood),
    ];
    for (arm, arm_rows) in rows.iter().enumerate() {
        for (k, loop_rows) in arm_rows.iter().enumerate() {
            for (name, keep) in splits {
                let idx: Vec<usize> = queries
                    .iter()
                    .enumerate()
                    .filter(|(_, q)| keep(q))
                    .map(|(i, _)| i)
                    .collect();
                let rs = loop_rows;
                let scores_cos: Vec<f32> = idx.iter().map(|&i| rs[i].cos_argmax).collect();
                let scores_mg: Vec<f32> = idx.iter().map(|&i| rs[i].margin).collect();
                let scores_cb: Vec<f32> = idx.iter().map(|&i| rs[i].cos_bridge).collect();
                let labels: Vec<bool> = idx.iter().map(|&i| rs[i].correct).collect();
                let acc = labels.iter().filter(|&&l| l).count();
                let bt1 = idx.iter().filter(|&&i| rs[i].bridge_top1).count();
                println!(
                    "{arm},{k},{name},{},{},{},{},{},{}",
                    idx.len(),
                    fmt_frac(acc, idx.len()),
                    fmt_frac(bt1, idx.len()),
                    auroc_cell(&scores_cos, &labels),
                    auroc_cell(&scores_mg, &labels),
                    auroc_cell(&scores_cb, &labels),
                );
            }
        }
    }
    let acc = |arm: usize| {
        let last = &rows[arm][LOOPS - 1];
        let c = last.iter().filter(|r| r.correct).count();
        fmt_frac(c, last.len())
    };
    println!(
        "json:{{\"queries\":{},\"loops\":{LOOPS},\"alpha\":{ALPHA},\"acc_reentry\":{},\"acc_reinject\":{},\"elapsed_s\":{:.1}}}",
        queries.len(),
        acc(0),
        acc(1),
        t0.elapsed().as_secs_f32()
    );
}

/// `k/n` as a 4-decimal fraction (0.0 when the set is empty).
fn fmt_frac(k: usize, n: usize) -> String {
    if n == 0 {
        "0.0".to_string()
    } else {
        format!("{:.4}", k as f64 / n as f64)
    }
}

/// "point/lb95" or "nan" when either class is empty.
fn auroc_cell(scores: &[f32], labels: &[bool]) -> String {
    let ci = bootstrap_auroc_ci(scores, labels, BOOTSTRAP_ITERS, 929);
    if ci.point.is_nan() {
        "nan".to_string()
    } else {
        format!("{:.4}/{:.4}", ci.point, ci.ci_lo)
    }
}

/// `H ← (1−α)·H + α·RMSNorm(W[v])` on `runtime.hidden` (post-norm space: H
/// is already the final-norm readout, so the anchor is norm-matched to it).
fn reinject_hidden(
    runtime: &mut KimiK3Runtime,
    weights: &KimiK3ModelWeights,
    v: usize,
    alpha: f32,
    dim: usize,
) {
    if alpha <= 0.0 || dim == 0 {
        return;
    }
    let start = v * dim;
    if weights.lm_head_weight.len() < start + dim {
        return;
    }
    let row = &weights.lm_head_weight[start..start + dim];
    let sum_sq = simd_sum_sq(row, dim);
    let rms = (sum_sq / dim as f32).sqrt();
    let inv = if rms > 0.0 { 1.0 / rms } else { 0.0 };
    for (h, &w) in runtime.hidden.iter_mut().zip(row.iter()) {
        *h = (1.0 - alpha) * *h + alpha * (w * inv);
    }
}

/// cosine(H, W[row]) — the true-bridge-row alignment (the paper's quantity).
fn cosine_rows(state: &[f32], lm_head: &[f32], row_idx: u32, dim: usize) -> f32 {
    let start = row_idx as usize * dim;
    if dim == 0 || lm_head.len() < start + dim {
        return 0.0;
    }
    let row = &lm_head[start..start + dim];
    let dot = simd_dot_f32(state, row, dim);
    let denom = (simd_sum_sq(state, dim) * simd_sum_sq(row, dim)).sqrt();
    if denom > 0.0 && denom.is_finite() {
        dot / denom
    } else {
        0.0
    }
}

/// Greedy-continue `ANSWER_TOKENS` tokens (feeding each argmax through the
/// runtime), decode, return the text. Deterministic (strict `>` argmax →
/// lowest index on ties).
fn greedy_continue(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    tokenizer: &TiktokenTokenizer,
    logits: &[f32],
) -> String {
    let mut generated: Vec<usize> = Vec::with_capacity(ANSWER_TOKENS);
    let mut current = argmax_of(logits);
    for i in 0..ANSWER_TOKENS {
        if i > 0 {
            let l = kimi_k3_forward_token(config, weights, runtime, current as u32);
            current = argmax_of(l);
        }
        generated.push(current);
    }
    tokenizer.decode(&generated)
}

fn argmax_of(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
