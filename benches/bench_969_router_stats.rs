//! Issue 969 — routing-statistics diagnostic suite (Nestor App C.6) on
//! kimi_k3's shipped MoE.
//!
//! Port of arXiv:2610.31818 App C.6's four routing statistics as a
//! zero-training, offline instrument over FROZEN weights + a labeled corpus
//! (the reflex seated suites — 4 labeled domains), with an engagement proof
//! that the forced-routing counterfactual actually applies (the green-zero
//! trap: an instrument whose counterfactual never engages reads every
//! subject as "not load-bearing").
//!
//! # The four statistics (per MoE router/layer — layers 1..7)
//!
//! 1. **Effective choices** — `exp(H(p))/E` of the aggregate slot-marginal
//!    assignment distribution (load balance).
//! 2. **Corpus MI** — mutual information (bits) between the suite label and
//!    the unordered top-2 assignment pair.
//! 3. **Assignment drift** — % of assignments whose unordered pair changes
//!    under seeded router-weight noise (two levels, σ = c·std(w_router),
//!    all routers perturbed simultaneously, measured per layer).
//! 4. **Shuffled / random counterfactuals** — re-route every token at ONE
//!    layer with (a) another token's ordered pair (shuffled; the pair
//!    multiset is unchanged, so expert load is exactly preserved) or (b) two
//!    independent slot permutations (random equal-size; per-slot load exactly
//!    preserved), weights otherwise unchanged; measure the final-hidden delta
//!    (cosine + relative L2) at every position and the last-position argmax
//!    flip through the LM head. The gate weights stay the receiver token's
//!    OWN sigmoid scores over the forced experts — assignment-only
//!    counterfactual ("re-routed at unchanged load", Nestor App C.6).
//!
//! # Scope caveats (from the issue, both carried to the verdict)
//!
//! - K3 routes at ONE level (flat top-2; the shared expert is never routed),
//!   so the paper's two-level column-vs-expert MI split is NOT runnable here.
//! - The 0.40B checkpoint is test-arch quality (owner's standing note) — the
//!   verdict row describes THAT checkpoint's router and is never evidence
//!   about Nestor, MoE in general, or production-MoE regimes. Provenance
//!   (upstream Moonshot release) is recorded in the stats meta.
//!
//! # Instrument liveness (the K3-arm half of the issue's G1)
//!
//! The reflex `pick_domain` arm (T3) is the issue's liveness CONTROL. This
//! bench adds its own cheaper engagement proof: (a) after arming a forced
//! plan the read-back assignments must EQUAL the plan (the seam applied);
//! (b) at least one layer must show a nonzero shuffle delta (the
//! counterfactual reached the output). Either violation refuses the run
//! loud — a near-zero delta would otherwise publish "not load-bearing" over
//! a blind instrument.
//!
//! # Byte-stability
//!
//! The whole stats computation runs twice (identical seeds, identical corpus
//! order); the two serializations must be byte-equal.
//!
//! # Run
//!
//! ```bash
//! KIMI_K3_MODEL_DIR=../riir-train/data/kimi-k3-0.40b \
//! cargo bench --manifest-path Cargo.toml \
//!     --features kimi_k3_loader \
//!     --bench bench_969_router_stats -- --nocapture
//! ```
//!
//! Report-only: no serving-path change, no feature promotion. Skips loud
//! (exit 0, named reason) when the model or the datasets are absent.

#![cfg(feature = "kimi_k3_loader")]
#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use katgpt_core::simd::simd_matmul_rows;
use katgpt_rs::kimi_k3::decoder_layer::{KimiFfnConfig, KimiFfnWeights};
use katgpt_rs::kimi_k3::loader::{KimiK3ModelWeights, load_kimi_k3};
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token_traced};
use katgpt_rs::kimi_k3::tiktoken::{TiktokenTokenizer, load_tiktoken_bpe};

// ─── Instrument constants (recorded in the stats meta) ─────────────────────

/// Seated suites = the 4 labeled domains (the issue's named T2 corpus).
const SUITES: [&str; 4] = ["ag_news", "banking77", "emotion", "sst5"];
const DATASETS_DIR: &str = "../riir-reflex/.raw/datasets";
const N_SEQS_PER_SUITE: usize = 16;
const MAX_TOKENS: usize = 48;
/// Router-noise drift levels as fractions of the router weight std.
const DRIFT_LEVELS: [f32; 2] = [0.05, 0.20];
/// Verdict thresholds (instrument constants, recorded):
/// corpus-DEPENDENT regime (Nestor expert level ≈ 0.28 bits) at or above this MI.
const MI_DEPENDENT_BITS: f64 = 0.10;
/// load-bearing shuffle delta at or above this relative L2 ...
const SHUFFLE_LOAD_REL_L2: f64 = 0.05;
/// ... or this last-position argmax flip rate.
const SHUFFLE_LOAD_FLIP: f64 = 0.10;
const SEED_BASE: u64 = 0x9690_0000;

// ─── Deterministic RNG (no external dep — the bench_012 LCG pattern) ───────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(SEED_BASE))
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    /// Uniform f32 in [0, 1).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (1u64 << 32) as f64) as f32
    }
    /// Standard normal via Box-Muller (stateless — two uniforms per draw, so
    /// the value depends only on the stream position, never on call sharing).
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (std::f32::consts::TAU * u2).cos()
    }
}

/// Fisher-Yates shuffle, seeded (deterministic).
fn seeded_permutation(n: usize, rng: &mut Lcg) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        v.swap(i, j);
    }
    v
}

// ─── Corpus loading (reflex seated datasets — HF datasets-server pages) ────

/// The texts of one suite, in sorted-file + row order (test-000.json first,
/// rows in page order) — deterministic, byte-stable.
fn load_suite_texts(suite: &str, n: usize) -> Option<Vec<String>> {
    let dir = Path::new(DATASETS_DIR).join(suite);
    let mut pages: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.starts_with("test-") && f.ends_with(".json"))
        })
        .collect();
    pages.sort();
    let mut texts = Vec::new();
    for page in pages {
        let raw = std::fs::read_to_string(&page).ok()?;
        let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let rows = v.get("rows")?.as_array()?;
        for r in rows {
            let text = r.get("row")?.get("text")?.as_str()?.trim().to_string();
            if !text.is_empty() {
                texts.push(text);
            }
        }
        if texts.len() >= n {
            break;
        }
    }
    texts.truncate(n);
    Some(texts)
}

struct Corpus {
    /// token ids per sequence
    tokens: Vec<Vec<u32>>,
    /// suite index per sequence (the MI "corpus label")
    suite_of_seq: Vec<usize>,
    total_tokens: usize,
}

// ─── The pass runner ────────────────────────────────────────────────────────

/// Per-sequence counterfactual plan at one counterfactual layer row:
/// per-pos ordered expert pair. `None` = keep the router's own choice.
type SeqPlan = Option<Vec<(usize, usize)>>;

/// One full-corpus forward pass.
struct PassOut {
    /// ordered (slot0, slot1) pair per [layer_row][seq][pos] — captured from
    /// the MoE scratch INSIDE the per-token loop (the scratch holds only the
    /// current token's assignment).
    ordered_pairs: Vec<Vec<Vec<(usize, usize)>>>,
    /// final normalized hidden per [seq][pos][d]
    hidden: Vec<Vec<Vec<f32>>>,
    /// last-position argmax per seq (one LM head projection per sequence)
    top1: Vec<u32>,
    /// engagement witness: forced reads that did NOT match the plan
    plan_mismatches: usize,
    /// count of forced applications (plan rows actually applied)
    plan_applied: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_pass(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    corpus: &Corpus,
    moe_layers: &[usize],
    // plans[seq][row_i] — per-sequence rows over the moe_layers order
    plans: Option<&[Vec<SeqPlan>]>,
    logits_scratch: &mut Vec<f32>,
    traj_scratch: &mut Vec<Vec<f32>>,
) -> PassOut {
    let d = config.hidden_size;
    let n_layers = config.num_layers;
    let n_moe = moe_layers.len();
    let n_seqs = corpus.tokens.len();

    let mut out = PassOut {
        ordered_pairs: vec![vec![Vec::new(); n_seqs]; n_moe],
        hidden: Vec::with_capacity(n_seqs),
        top1: vec![0; n_seqs],
        plan_mismatches: 0,
        plan_applied: 0,
    };

    for (seq, toks) in corpus.tokens.iter().enumerate() {
        runtime.reset();
        let mut seq_hidden: Vec<Vec<f32>> = Vec::with_capacity(toks.len());

        for (pos, &tok) in toks.iter().enumerate() {
            // Arm the per-token plan: one row per decoder layer.
            match plans {
                Some(rows) => {
                    let mut plan: Vec<Option<Vec<usize>>> = vec![None; n_layers];
                    for (row_i, &layer) in moe_layers.iter().enumerate() {
                        let forced_pair = rows
                            .get(seq)
                            .and_then(|r| r.get(row_i))
                            .and_then(|p| p.as_ref())
                            .and_then(|pairs| pairs.get(pos));
                        if let Some(&(a, b)) = forced_pair {
                            plan[layer] = Some(vec![a, b]);
                            out.plan_applied += 1;
                        }
                    }
                    runtime.forced_router_plan = Some(plan);
                }
                None => runtime.forced_router_plan = None,
            }

            kimi_k3_forward_token_traced(config, weights, runtime, tok, traj_scratch);
            seq_hidden.push(runtime.hidden.clone());

            // Capture this token's assignments + engagement witness NOW (the
            // scratch holds only the current token's assignment).
            for (row_i, &layer) in moe_layers.iter().enumerate() {
                let idx = &runtime.layers[layer].ffn_scratch.moe.topk_indices;
                let pair = (idx[0], idx[1]);
                let want_pair = plans
                    .and_then(|rows| rows.get(seq))
                    .and_then(|r| r.get(row_i))
                    .and_then(|p| p.as_ref())
                    .and_then(|pairs| pairs.get(pos));
                if let Some(&(a, b)) = want_pair {
                    let mut want = [a, b];
                    let mut got = [idx[0], idx[1]];
                    want.sort_unstable();
                    got.sort_unstable();
                    if want != got {
                        out.plan_mismatches += 1;
                    }
                }
                out.ordered_pairs[row_i][seq].push(pair);
            }
        }

        runtime.forced_router_plan = None;
        out.hidden.push(seq_hidden);

        // Last-position argmax through the LM head (one projection/sequence).
        if let Some(h) = out.hidden.last().and_then(|s| s.last()) {
            logits_scratch.clear();
            logits_scratch.resize(config.vocab_size, 0.0);
            simd_matmul_rows(
                logits_scratch,
                &weights.lm_head_weight,
                h,
                config.vocab_size,
                d,
            );
            out.top1[seq] = logits_scratch
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i as u32);
        }
    }

    out
}

#[inline]
fn unordered_key(pair: (usize, usize), e: usize) -> usize {
    let a = pair.0.min(pair.1);
    let b = pair.0.max(pair.1);
    a * e + b
}

// ─── The stats computation (run twice; byte-equal serializations gate) ─────

struct LayerStats {
    layer: usize,
    n_tokens: usize,
    effective_choices: f64,
    max_load_frac: f64,
    mi_bits: f64,
    drift: [f64; DRIFT_LEVELS.len()],
    shuffle_cos: f64,
    shuffle_rel_l2: f64,
    shuffle_flip: f64,
    random_cos: f64,
    random_rel_l2: f64,
    random_flip: f64,
}

struct Stats {
    layers: Vec<LayerStats>,
    total_tokens: usize,
    plan_applied: usize,
    plan_mismatches: usize,
    baseline_s: f64,
    total_s: f64,
}

fn moe_layer_indices(config: &KimiK3ModelConfig) -> Vec<usize> {
    (0..config.num_layers)
        .filter(|&l| matches!(config.layer_config(l).ffn, KimiFfnConfig::Moe(_)))
        .collect()
}

fn compute_stats(
    config: &KimiK3ModelConfig,
    weights: &mut KimiK3ModelWeights,
    corpus: &Corpus,
    moe_layers: &[usize],
) -> Stats {
    let e = config.moe_config.num_experts;
    let d = config.hidden_size;
    let n_moe = moe_layers.len();
    let n_seqs = corpus.tokens.len();
    let t_start = Instant::now();

    let mut runtime = KimiK3Runtime::new(config, MAX_TOKENS + 2);
    let mut logits = Vec::new();
    let mut traj: Vec<Vec<f32>> = Vec::new();

    // ── Baseline pass ──────────────────────────────────────────────────────
    let t0 = Instant::now();
    let base = run_pass(
        config,
        weights,
        &mut runtime,
        corpus,
        moe_layers,
        None,
        &mut logits,
        &mut traj,
    );
    let baseline_s = t0.elapsed().as_secs_f64();
    println!(
        "    baseline: {} tokens in {:.1}s ({:.2} ms/token)",
        corpus.total_tokens,
        baseline_s,
        baseline_s * 1000.0 / corpus.total_tokens.max(1) as f64
    );

    // ── Stat 1: effective choices (slot-marginal, per layer) ──────────────
    let mut effective = Vec::with_capacity(n_moe);
    for row in 0..n_moe {
        let mut count = vec![0u64; e];
        let mut total = 0u64;
        for seq in &base.ordered_pairs[row] {
            for &pair in seq {
                count[pair.0] += 1;
                count[pair.1] += 1;
                total += 2;
            }
        }
        let mut h = 0.0f64;
        let mut max_frac = 0.0f64;
        for &c in &count {
            let p = c as f64 / total.max(1) as f64;
            max_frac = max_frac.max(p);
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        effective.push((h.exp() / e as f64, max_frac));
    }

    // ── Stat 2: corpus MI(label; unordered pair), per layer ───────────────
    let mut mi = Vec::with_capacity(n_moe);
    for row in 0..n_moe {
        let n_pairs = e * e;
        let mut joint = vec![0u64; SUITES.len() * n_pairs];
        let mut label_marg = vec![0u64; SUITES.len()];
        let mut pair_marg = vec![0u64; n_pairs];
        let mut total = 0u64;
        for (seq, pairs) in base.ordered_pairs[row].iter().enumerate() {
            let s = corpus.suite_of_seq[seq];
            for &pair in pairs {
                let key = unordered_key(pair, e);
                joint[s * n_pairs + key] += 1;
                label_marg[s] += 1;
                pair_marg[key] += 1;
                total += 1;
            }
        }
        let mut m = 0.0f64;
        for s in 0..SUITES.len() {
            for k in 0..n_pairs {
                let j = joint[s * n_pairs + k] as f64;
                if j > 0.0 {
                    let p_joint = j / total as f64;
                    let p_s = label_marg[s] as f64 / total as f64;
                    let p_k = pair_marg[k] as f64 / total as f64;
                    m += p_joint * (p_joint / (p_s * p_k)).log2();
                }
            }
        }
        mi.push(m);
    }

    // ── Stat 3: assignment drift under seeded router-weight noise ─────────
    let mut routers: Vec<Vec<f32>> = Vec::with_capacity(n_moe);
    for &layer in moe_layers {
        if let KimiFfnWeights::Moe(w) = &weights.layers[layer].ffn {
            routers.push(w.router_weight.clone());
        }
    }
    let mut drift = vec![[0.0f64; DRIFT_LEVELS.len()]; n_moe];
    for (level_i, &c) in DRIFT_LEVELS.iter().enumerate() {
        for (l, &layer) in moe_layers.iter().enumerate() {
            if let KimiFfnWeights::Moe(w) = &mut weights.layers[layer].ffn {
                let n = w.router_weight.len() as f64;
                let mean = w.router_weight.iter().map(|&x| x as f64).sum::<f64>() / n;
                let var = w
                    .router_weight
                    .iter()
                    .map(|&x| (x as f64 - mean) * (x as f64 - mean))
                    .sum::<f64>()
                    / n;
                let std = var.sqrt() as f32;
                let mut rng = Lcg::new(0x100 + l as u64 * 17 + level_i as u64 * 911);
                for x in w.router_weight.iter_mut() {
                    *x += c * std * rng.next_gaussian();
                }
            }
        }
        let noisy = run_pass(
            config,
            weights,
            &mut runtime,
            corpus,
            moe_layers,
            None,
            &mut logits,
            &mut traj,
        );
        for row in 0..n_moe {
            let mut changed = 0u64;
            let mut seen = 0u64;
            for seq in 0..n_seqs {
                for (pos, &pair) in base.ordered_pairs[row][seq].iter().enumerate() {
                    let np = noisy.ordered_pairs[row][seq][pos];
                    if unordered_key(pair, e) != unordered_key(np, e) {
                        changed += 1;
                    }
                    seen += 1;
                }
            }
            drift[row][level_i] = changed as f64 / seen.max(1) as f64;
        }
        // Restore routers.
        for (l, &layer) in moe_layers.iter().enumerate() {
            if let KimiFfnWeights::Moe(w) = &mut weights.layers[layer].ffn {
                w.router_weight.copy_from_slice(&routers[l]);
            }
        }
    }

    // ── Stat 4: shuffled / random counterfactuals, per layer ──────────────
    // Global (seq,pos) index space over the corpus.
    let mut global_base: Vec<(usize, usize)> = Vec::with_capacity(corpus.total_tokens);
    for (seq, toks) in corpus.tokens.iter().enumerate() {
        for pos in 0..toks.len() {
            global_base.push((seq, pos));
        }
    }
    let n_global = global_base.len();

    let mut shuffle_stats = Vec::with_capacity(n_moe);
    let mut random_stats = Vec::with_capacity(n_moe);
    let mut plan_applied = 0usize;
    let mut plan_mismatches = 0usize;

    for (row, &layer) in moe_layers.iter().enumerate() {
        // Baseline ordered pairs flattened in the same global order.
        let mut flat: Vec<(usize, usize)> = Vec::with_capacity(n_global);
        for &(seq, pos) in &global_base {
            flat.push(base.ordered_pairs[row][seq][pos]);
        }

        // (a) SHUFFLE — whole-pair permutation (multiset unchanged ⇒ load exact).
        let perm = seeded_permutation(n_global, &mut Lcg::new(0xF00 + layer as u64));
        let shuffled: Vec<(usize, usize)> = perm.iter().map(|&i| flat[i]).collect();

        // (b) RANDOM equal-size — two independent slot permutations.
        let mut firsts: Vec<usize> = flat.iter().map(|p| p.0).collect();
        let mut seconds: Vec<usize> = flat.iter().map(|p| p.1).collect();
        let p1 = seeded_permutation(n_global, &mut Lcg::new(0xF10 + layer as u64));
        let p2 = seeded_permutation(n_global, &mut Lcg::new(0xF20 + layer as u64));
        let f2: Vec<usize> = p1.iter().map(|&i| firsts[i]).collect();
        let s2: Vec<usize> = p2.iter().map(|&i| seconds[i]).collect();
        firsts = f2;
        seconds = s2;

        for mode in 0..2 {
            // Build the per-sequence plan for THIS layer. The flat arrays are
            // seq-major over (seq,pos), so a running counter walks them in
            // the same order — no lookup needed.
            let mut plans: Vec<Vec<SeqPlan>> = Vec::with_capacity(n_seqs);
            let mut g = 0usize;
            for toks in corpus.tokens.iter() {
                let mut rows: Vec<SeqPlan> = vec![None; n_moe];
                let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(toks.len());
                for _pos in 0..toks.len() {
                    pairs.push(if mode == 0 {
                        shuffled[g]
                    } else {
                        (firsts[g], seconds[g])
                    });
                    g += 1;
                }
                rows[row] = Some(pairs);
                plans.push(rows);
            }
            let cf = run_pass(
                config,
                weights,
                &mut runtime,
                corpus,
                moe_layers,
                Some(&plans),
                &mut logits,
                &mut traj,
            );
            plan_applied += cf.plan_applied;
            plan_mismatches += cf.plan_mismatches;

            // Deltas vs baseline.
            let mut cos_sum = 0.0f64;
            let mut rel_sum = 0.0f64;
            let mut n_h = 0u64;
            for seq in 0..n_seqs {
                for pos in 0..corpus.tokens[seq].len() {
                    let b = &base.hidden[seq][pos];
                    let f = &cf.hidden[seq][pos];
                    let (mut dot, mut nb, mut nf) = (0.0f64, 0.0f64, 0.0f64);
                    let (mut diff2, mut base2) = (0.0f64, 0.0f64);
                    for i in 0..d {
                        let (bv, fv) = (b[i] as f64, f[i] as f64);
                        dot += bv * fv;
                        nb += bv * bv;
                        nf += fv * fv;
                        let diff = bv - fv;
                        diff2 += diff * diff;
                        base2 += bv * bv;
                    }
                    let denom = nb.sqrt() * nf.sqrt();
                    cos_sum += if denom > 0.0 { dot / denom } else { 1.0 };
                    rel_sum += if base2 > 0.0 {
                        diff2.sqrt() / base2.sqrt()
                    } else {
                        0.0
                    };
                    n_h += 1;
                }
            }
            let flips = (0..n_seqs)
                .filter(|&seq| base.top1[seq] != cf.top1[seq])
                .count();
            let record = (
                cos_sum / n_h.max(1) as f64,
                rel_sum / n_h.max(1) as f64,
                flips as f64 / n_seqs.max(1) as f64,
            );
            if mode == 0 {
                shuffle_stats.push(record);
            } else {
                random_stats.push(record);
            }
        }
    }

    let mut layers = Vec::with_capacity(n_moe);
    for row in 0..n_moe {
        let n_tokens: usize = base.ordered_pairs[row].iter().map(|s| s.len()).sum();
        layers.push(LayerStats {
            layer: moe_layers[row],
            n_tokens,
            effective_choices: effective[row].0,
            max_load_frac: effective[row].1,
            mi_bits: mi[row],
            drift: drift[row],
            shuffle_cos: shuffle_stats[row].0,
            shuffle_rel_l2: shuffle_stats[row].1,
            shuffle_flip: shuffle_stats[row].2,
            random_cos: random_stats[row].0,
            random_rel_l2: random_stats[row].1,
            random_flip: random_stats[row].2,
        });
    }

    Stats {
        layers,
        total_tokens: corpus.total_tokens,
        plan_applied,
        plan_mismatches,
        baseline_s,
        total_s: t_start.elapsed().as_secs_f64(),
    }
}

// ─── Deterministic serialization (hand-rolled, fixed precision + key order) ─

/// The compared form: layers + engagement ONLY. Wall-clock timings are
/// meta — they differ between two runs by construction and must never
/// participate in the byte-stability gate (the founding run refused exactly
/// there).
fn fmt_layers(s: &Stats) -> String {
    let mut out = String::with_capacity(8192);
    out.push_str("{\n");
    out.push_str(&format!(
        "  \"corpus\": {{\"suites\": {:?}, \"n_seqs_per_suite\": {}, \"max_tokens\": {}, \"total_tokens\": {}, \"drift_levels\": {:?}}},\n",
        SUITES, N_SEQS_PER_SUITE, MAX_TOKENS, s.total_tokens, DRIFT_LEVELS
    ));
    out.push_str("  \"layers\": [\n");
    for (i, l) in s.layers.iter().enumerate() {
        let comma = if i + 1 < s.layers.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\"layer\": {}, \"n_tokens\": {}, \"effective_choices\": {:.6}, \"max_load_frac\": {:.6}, \"mi_bits\": {:.6}, \"drift\": [{:.6}, {:.6}], \"shuffle\": {{\"mean_cos\": {:.6}, \"mean_rel_l2\": {:.3e}, \"flip_rate\": {:.6}}}, \"random\": {{\"mean_cos\": {:.6}, \"mean_rel_l2\": {:.3e}, \"flip_rate\": {:.6}}}}}{}\n",
            l.layer,
            l.n_tokens,
            l.effective_choices,
            l.max_load_frac,
            l.mi_bits,
            l.drift[0],
            l.drift[1],
            l.shuffle_cos,
            l.shuffle_rel_l2,
            l.shuffle_flip,
            l.random_cos,
            l.random_rel_l2,
            l.random_flip,
            comma
        ));
    }
    out.push_str("  ],\n");
    out.push_str(&format!(
        "  \"engagement\": {{\"plan_applied\": {}, \"plan_mismatches\": {}}}\n",
        s.plan_applied, s.plan_mismatches
    ));
    out.push_str("}\n");
    out
}

fn fmt_stats(s: &Stats, verdict: Option<&str>) -> String {
    let mut out = fmt_layers(s);
    // Splice the meta (timings included) after the opening line.
    let meta = format!(
        "  \"meta\": {{\"baseline_s\": {:.3}, \"total_s\": {:.3}}},\n",
        s.baseline_s, s.total_s
    );
    out.insert_str(2, &meta);
    if let Some(v) = verdict {
        let vline = format!("  \"verdict_hint\": \"{}\"\n", v);
        let close = out.len() - 2;
        out.insert_str(close, &vline);
    }
    out
}

// ─── Verdict (T4, K3 half — instrument hint; the record md owns judgment) ──

fn verdict_of(s: &Stats) -> &'static str {
    let mi_mean: f64 =
        s.layers.iter().map(|l| l.mi_bits).sum::<f64>() / s.layers.len().max(1) as f64;
    let max_rel_l2 = s
        .layers
        .iter()
        .map(|l| l.shuffle_rel_l2)
        .fold(0.0f64, f64::max);
    let max_flip = s
        .layers
        .iter()
        .map(|l| l.shuffle_flip)
        .fold(0.0f64, f64::max);
    if mi_mean < 0.02 && max_rel_l2 < 0.01 {
        "NOT_LOAD_BEARING (both near zero on every layer)"
    } else if mi_mean >= MI_DEPENDENT_BITS {
        "CORPUS_DEPENDENT (Nestor expert-level regime: high corpus MI)"
    } else if max_rel_l2 >= SHUFFLE_LOAD_REL_L2 || max_flip >= SHUFFLE_LOAD_FLIP {
        "CORPUS_INDEPENDENT_BUT_LOAD_BEARING (Nestor column-level regime: low MI, large shuffle cost)"
    } else {
        "MIXED_LOW_SIGNAL (below both regime bars — quote the numbers, do not force a label)"
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

fn model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("KIMI_K3_MODEL_DIR") {
        return PathBuf::from(d);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let local = manifest.join("data/kimi-k3-0.40b");
    if local.join("model.safetensors").exists() {
        return local;
    }
    // Documented fallback: the shared data lane (riir-train local-keep pin).
    manifest
        .join("../riir-train/data/kimi-k3-0.40b")
        .components()
        .collect::<PathBuf>()
}

fn main() {
    println!("== Issue 969 — kimi_k3 router-statistics diagnostic (Nestor App C.6) ==");

    // ── Material checks: skip LOUD, never a green zero ────────────────────
    let dir = model_dir();
    let model_p = dir.join("model.safetensors");
    let tik_p = dir.join("tiktoken.model");
    if !model_p.exists() || !tik_p.exists() {
        println!(
            "SKIP: model material missing at {} (set KIMI_K3_MODEL_DIR) — instrument not run, not a pass",
            dir.display()
        );
        return;
    }
    for s in SUITES {
        if !Path::new(DATASETS_DIR).join(s).exists() {
            println!(
                "SKIP: dataset suite missing: {DATASETS_DIR}/{s} — instrument not run, not a pass"
            );
            return;
        }
    }

    // ── Tokenizer + weights ────────────────────────────────────────────────
    let tik_bytes = std::fs::read(&tik_p).unwrap_or_else(|e| panic!("read tiktoken: {e}"));
    let ranks = load_tiktoken_bpe(&tik_bytes).unwrap_or_else(|e| panic!("tiktoken parse: {e:?}"));
    let tokenizer = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);
    let t_load = Instant::now();
    let mut weights =
        load_kimi_k3(model_p.to_str().unwrap()).unwrap_or_else(|e| panic!("load failed: {e}"));
    println!(
        "    weights: {} ({:.1}s)",
        model_p.display(),
        t_load.elapsed().as_secs_f64()
    );

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let moe_layers = moe_layer_indices(&config);
    println!(
        "    config : {} layers, hidden={}, vocab={}, MoE layers {:?} (top-{} of {})",
        config.num_layers,
        config.hidden_size,
        config.vocab_size,
        moe_layers,
        config.moe_config.num_experts_per_token,
        config.moe_config.num_experts,
    );

    // ── Corpus ─────────────────────────────────────────────────────────────
    let mut tokens: Vec<Vec<u32>> = Vec::new();
    let mut suite_of_seq: Vec<usize> = Vec::new();
    for (s_i, suite) in SUITES.iter().enumerate() {
        let texts = load_suite_texts(suite, N_SEQS_PER_SUITE)
            .unwrap_or_else(|| panic!("suite {suite} failed to parse"));
        assert!(
            texts.len() == N_SEQS_PER_SUITE,
            "suite {suite}: short corpus"
        );
        for t in texts {
            let toks: Vec<u32> = tokenizer
                .encode(&t)
                .into_iter()
                .take(MAX_TOKENS)
                .map(|x| x as u32)
                .collect();
            if toks.is_empty() {
                continue;
            }
            tokens.push(toks);
            suite_of_seq.push(s_i);
        }
    }
    let total_tokens: usize = tokens.iter().map(|t| t.len()).sum();
    let corpus = Corpus {
        tokens,
        suite_of_seq,
        total_tokens,
    };
    println!(
        "    corpus : {} seqs, {} tokens ({} suites × ≤{} tokens)",
        corpus.tokens.len(),
        total_tokens,
        SUITES.len(),
        MAX_TOKENS
    );

    // ── Stats: run TWICE; the STAT serializations must be byte-equal ─────
    // (timings are meta and excluded — they differ between runs by design).
    let t_all = Instant::now();
    let s1 = compute_stats(&config, &mut weights, &corpus, &moe_layers);
    let j1 = fmt_layers(&s1);
    let s2 = compute_stats(&config, &mut weights, &corpus, &moe_layers);
    let j2 = fmt_layers(&s2);
    println!(
        "    two full stat computations: {:.1}s",
        t_all.elapsed().as_secs_f64()
    );
    if j1 != j2 {
        eprintln!("REFUSE: stats NOT byte-stable across two runs (G1)");
        std::process::exit(1);
    }
    println!("    byte-stability: OK (two runs serialize identically)");

    // ── Engagement (instrument liveness, K3-arm half of G1) ────────────────
    if s1.plan_mismatches != 0 {
        eprintln!(
            "REFUSE: forced plan did not engage — {} read-back mismatches (instrument blind)",
            s1.plan_mismatches
        );
        std::process::exit(1);
    }
    let max_rel_l2 = s1
        .layers
        .iter()
        .map(|l| l.shuffle_rel_l2)
        .fold(0.0f64, f64::max);
    if max_rel_l2 <= 0.0 {
        eprintln!(
            "REFUSE: zero shuffle delta on EVERY layer — counterfactual never reached the output"
        );
        std::process::exit(1);
    }
    println!(
        "    engagement: {} forced applications, 0 mismatches, max shuffle rel_l2 {:.3e} — instrument live",
        s1.plan_applied, max_rel_l2
    );

    // ── Table ──────────────────────────────────────────────────────────────
    println!(
        "\n    {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} | {:>24} | {:>24}",
        "layer",
        "eff",
        "maxld",
        "MI bits",
        "dr.05",
        "dr.20",
        "shuffle cos / relL2 / flip",
        "random cos / relL2 / flip"
    );
    for l in &s1.layers {
        println!(
            "    {:>5} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} | {:>7.4} / {:>7.2e} / {:.4} | {:>7.4} / {:>7.2e} / {:.4}",
            l.layer,
            l.effective_choices,
            l.max_load_frac,
            l.mi_bits,
            l.drift[0],
            l.drift[1],
            l.shuffle_cos,
            l.shuffle_rel_l2,
            l.shuffle_flip,
            l.random_cos,
            l.random_rel_l2,
            l.random_flip,
        );
    }

    // ── Write the artifact ─────────────────────────────────────────────────
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".benchmarks/969_router_stats");
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("mkdir: {e}"));
    let verdict = verdict_of(&s1);
    let record = fmt_stats(&s1, Some(verdict));
    std::fs::write(out_dir.join("stats.json"), &record)
        .unwrap_or_else(|e| panic!("write stats.json: {e}"));
    println!(
        "\n    stats.json → {}\n    verdict hint: {}",
        out_dir.join("stats.json").display(),
        verdict
    );
}
