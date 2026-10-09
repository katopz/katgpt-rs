//! Issue 929 T1.2-T1.4 runner — the loop-alignment probe on a real checkpoint.
//!
//! Runs the DiscoLoop alignment probe (arXiv:2607.00341, Research 614) over
//! the two-hop composition fixture on the REAL Kimi-K3-0.40B checkpoint
//! (the arch-test model — this measures whether the signal EXISTS on the
//! loop-recurrence surface, never output quality).
//!
//! ## What it measures
//!
//! The T1.2 checkpoint leg is the **caller-side Ouro recurrence** over
//! `kimi_k3_forward_token` / `kimi_k3_forward_token_hidden` (the
//! `forward_looped` leg is arch-impossible: learned-`wpe` GPT-2-style vs
//! RoPE+qk-norm — see the issue Status). After prefilling the prompt, each
//! loop iteration re-injects the previous post-norm hidden as the decoder
//! input — one Ouro step (KDA update + MLA cache append, no token consumed;
//! the same path Plan 324 A4's backward twin trains through). At every step
//! k = 0..=L the runner records:
//!
//! - `cos(k)` = `cos(H^(k), W[v̂^(k)])` — state-vs-decode-manifold alignment,
//! - `margin(k)` = top1 − top2 of the readout,
//!
//! where the logits ARE `W·H^(k)` (the probe adds no GEMV of its own).
//!
//! After the K loops the model decodes the answer greedily (normal token
//! feeding). Labels:
//!
//! - **full** (primary) — the decoded text starts with the gold entity name,
//! - **first** — the first decoded piece already starts the gold name,
//! - **pre** — same, from the pre-loop readout (k=0 control, JSON only).
//!
//! T1.3 reads AUROC(cos(k)) and AUROC(margin(k)) against the primary label.
//! T1.4 is the kill bar: **max 95% bootstrap CI lower bound < 0.6 over all
//! (k, signal) rows ⇒ the signal is absent/weak on non-loop-trained
//! weights ⇒ record negative, close the issue (reopen trigger = riir-train
//! Plan 451's loop-trained artifact).**
//!
//! ## Preconditions (verified at boot, loud exit otherwise)
//!
//! - checkpoint on disk + **untied LM head** (`lm_head_weight` is the true
//!   readout operator the probe reads; kimi-k3-0.40B is untied by loader
//!   contract — the runner re-asserts the row count).
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --features kimi_k3_loader,loop_alignment_probe \
//!   --example kimi_k3_loop_alignment_probe
//! K929_SMOKE=1 cargo run ...    # ~1 min posture (tiny fixture, L=4)
//! ```
//!
//! CPU-only, single thread — no GPU exclusivity needed; safe beside a GPU
//! trainer.
//!
//! ## Environment knobs (all optional)
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `KIMI_K3_MODEL_DIR` | `<repo>/data/kimi-k3-0.40b` | checkpoint + tiktoken dir |
//! | `K929_SMOKE` | off | `1` = tiny fast posture |
//! | `K929_ENTITIES` | 64 | entities per pool (paper: 250) |
//! | `K929_RELATIONS` | 50 | shared role count |
//! | `K929_OUT_DEGREE` | 6 | facts per entity |
//! | `K929_FACTS` | 8 | in-context fact window per prompt |
//! | `K929_TWO_HOP` | 12 | two-hop queries PER POOL |
//! | `K929_ONE_HOP` | 16 | one-hop controls PER POOL |
//! | `K929_SEED` | 2345 | fixture seed (deterministic) |
//! | `K929_LOOPS` | 8 | loop iterations L |
//! | `K929_ANSWER_MAX_TOKENS` | 12 | greedy decode budget |
//! | `K929_BOS` | off | `1` = prepend the BOS token to each sequence |
//! | `K929_EXEMPLARS` | 0 | solved in-context exemplars prepended (few-shot elicitation; the exemplar items are excluded from the eval) |
//! | `K929_BOOTSTRAP_ITERS` | 2000 | stratified bootstrap draws |
//! | `K929_MAX_ITEMS` | none | cap items (smoke/debug) |
//! | `K929_OUT` | none | write the full JSON record here |

#![cfg(all(feature = "kimi_k3_loader", feature = "loop_alignment_probe"))]

use std::path::Path;
use std::time::Instant;

use katgpt_core::loop_alignment_probe::{
    FixtureSpec, TwoHopItem, bootstrap_auroc_ci, generate_two_hop_fixture, probe_alignment,
};
use katgpt_rs::kimi_k3::loader::{KimiK3ModelWeights, load_kimi_k3};
use katgpt_rs::kimi_k3::model::{
    KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token, kimi_k3_forward_token_hidden,
};
use katgpt_rs::kimi_k3::tiktoken::{TiktokenTokenizer, load_tiktoken_bpe};

/// T1.4 kill bar: the signal must clear this 95% CI lower bound to live.
const KILL_BAR: f64 = 0.6;

// ─── Posture ────────────────────────────────────────────────────────────────

struct RunConfig {
    entities: usize,
    relations: usize,
    out_degree: usize,
    facts: usize,
    two_hop_per_pool: usize,
    one_hop_per_pool: usize,
    seed: u64,
    loops: usize,
    answer_max_tokens: usize,
    bos: bool,
    exemplars: usize,
    bootstrap_iters: usize,
    max_items: Option<usize>,
    smoke: bool,
}

impl RunConfig {
    fn from_env() -> Self {
        let smoke = matches!(std::env::var("K929_SMOKE").as_deref(), Ok("1") | Ok("true"));
        let (entities, relations, out_degree, facts, two_hop, one_hop, loops, answer_max, boot) =
            if smoke {
                (32, 8, 3, 6, 3, 5, 4, 8, 500)
            } else {
                (64, 50, 6, 8, 12, 16, 8, 12, 2000)
            };
        Self {
            entities: env_or("K929_ENTITIES", entities),
            relations: env_or("K929_RELATIONS", relations),
            out_degree: env_or("K929_OUT_DEGREE", out_degree),
            facts: env_or("K929_FACTS", facts),
            two_hop_per_pool: env_or("K929_TWO_HOP", two_hop),
            one_hop_per_pool: env_or("K929_ONE_HOP", one_hop),
            seed: env_or("K929_SEED", 0x929),
            loops: env_or("K929_LOOPS", loops),
            answer_max_tokens: env_or("K929_ANSWER_MAX_TOKENS", answer_max),
            bos: matches!(std::env::var("K929_BOS").as_deref(), Ok("1") | Ok("true")),
            exemplars: env_or("K929_EXEMPLARS", 0),
            bootstrap_iters: env_or("K929_BOOTSTRAP_ITERS", boot),
            max_items: std::env::var("K929_MAX_ITEMS").ok().and_then(|s| s.parse().ok()),
            smoke,
        }
    }
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(s) => s.parse().unwrap_or(default),
        Err(_) => default,
    }
}

// ─── Per-item run ───────────────────────────────────────────────────────────

struct ProbeRun<'a> {
    config: &'a KimiK3ModelConfig,
    weights: &'a KimiK3ModelWeights,
    tokenizer: &'a TiktokenTokenizer,
    loops: usize,
    answer_max_tokens: usize,
    bos: bool,
}

struct ItemOutcome {
    cos: Vec<f32>,
    margin: Vec<f32>,
    pre_first_ok: bool,
    first_ok: bool,
    full_ok: bool,
    generated: String,
}

/// Prefill + K loop iterations + the loop-refined greedy decode.
///
/// `prompt_tokens` must be non-empty (the caller checks). Borrow shape: the
/// forward calls return logits tied to the `&mut runtime` borrow, so each
/// step copies the logits into a reused scratch before `runtime.hidden` is
/// snapshotted — the probe reads the post-forward state H^(k), which is what
/// the returned logits were computed from.
fn run_item(run: &ProbeRun, item: &TwoHopItem, prompt_tokens: &[usize]) -> ItemOutcome {
    let d = run.config.hidden_size;
    let seq_cap = prompt_tokens.len() + run.loops + run.answer_max_tokens + 4;
    let mut runtime = KimiK3Runtime::new(run.config, seq_cap);
    let mut logits_buf: Vec<f32> = Vec::with_capacity(run.config.vocab_size);

    if run.bos {
        let lg = kimi_k3_forward_token(
            run.config,
            run.weights,
            &mut runtime,
            run.tokenizer.bos_id() as u32,
        );
        logits_buf.clear();
        logits_buf.extend_from_slice(lg);
    }
    for &t in prompt_tokens {
        let lg = kimi_k3_forward_token(run.config, run.weights, &mut runtime, t as u32);
        logits_buf.clear();
        logits_buf.extend_from_slice(lg);
    }

    // k = 0: the pre-loop state H^(0) (post final RMSNorm) and its readout.
    let h0 = runtime.hidden.clone();
    let probe0 = probe_alignment(&h0, &logits_buf, &run.weights.lm_head_weight, d);
    let pre_first_ok = id_piece_ok(run.tokenizer, probe0.argmax, &item.answer);

    let mut cos = Vec::with_capacity(run.loops + 1);
    let mut margin = Vec::with_capacity(run.loops + 1);
    cos.push(probe0.cos_alignment);
    margin.push(probe0.margin);
    let mut last_argmax = probe0.argmax;

    for _ in 0..run.loops {
        let lg = kimi_k3_forward_token_hidden(run.config, run.weights, &mut runtime);
        logits_buf.clear();
        logits_buf.extend_from_slice(lg);
        let h = runtime.hidden.clone();
        let probe = probe_alignment(&h, &logits_buf, &run.weights.lm_head_weight, d);
        cos.push(probe.cos_alignment);
        margin.push(probe.margin);
        last_argmax = probe.argmax;
    }

    // Loop-refined greedy decode from the looped state (normal token feeding).
    let mut generated: Vec<usize> = Vec::with_capacity(run.answer_max_tokens);
    let mut next = last_argmax;
    for _ in 0..run.answer_max_tokens {
        generated.push(next);
        if next == run.tokenizer.eos_id() {
            break;
        }
        let lg = kimi_k3_forward_token(run.config, run.weights, &mut runtime, next as u32);
        next = argmax(lg);
    }

    let gen_text = run.tokenizer.decode(&generated);
    let full_ok = gen_text.trim_start().starts_with(&item.answer);
    let first_ok = generated
        .first()
        .is_some_and(|&g| id_piece_ok(run.tokenizer, g, &item.answer));

    ItemOutcome {
        cos,
        margin,
        pre_first_ok,
        first_ok,
        full_ok,
        generated: gen_text,
    }
}

fn argmax(scores: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in scores.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// The decoded piece (leading whitespace stripped) must be a prefix of the
/// gold name — entity names render as "adj noun", so a leading-space piece
/// is the expected continuation after the bare `Answer:` cue.
fn piece_ok(piece: &str, answer: &str) -> bool {
    let trimmed = piece.trim_start();
    !trimmed.is_empty() && answer.starts_with(trimmed)
}

fn id_piece_ok(tokenizer: &TiktokenTokenizer, id: usize, answer: &str) -> bool {
    piece_ok(&tokenizer.decode(std::slice::from_ref(&id)), answer)
}

// ─── Records + analysis ─────────────────────────────────────────────────────

struct ItemRecord {
    kind: &'static str,
    ood: bool,
    answer: String,
    cos: Vec<f32>,
    margin: Vec<f32>,
    pre_first_ok: bool,
    first_ok: bool,
    full_ok: bool,
    generated: String,
}

fn class_counts(labels: &[bool]) -> (usize, usize) {
    let pos = labels.iter().filter(|&&l| l).count();
    (pos, labels.len() - pos)
}

fn mean_of(vals: &[f32]) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64
}

/// Print the AUROC grid for one label over one item subset; returns
/// `(max finite CI lower bound, "signal@k" where it was seen)`.
fn auroc_table(
    title: &str,
    picked: &[&ItemRecord],
    label_of: fn(&ItemRecord) -> bool,
    loops: usize,
    boot_iters: usize,
    seed: u64,
) -> (f64, String) {
    let labels: Vec<bool> = picked.iter().map(|r| label_of(r)).collect();
    let (pos, neg) = class_counts(&labels);
    println!("\n== AUROC vs {title} (n={} pos={pos} neg={neg}) ==", picked.len());
    if pos == 0 || neg == 0 {
        println!("   UNDECIDABLE - a class is empty; AUROC is undefined (never a pass).");
        return (f64::NEG_INFINITY, String::from("undecidable"));
    }
    let mut best = (f64::NEG_INFINITY, String::from("none"));
    for k in 0..=loops {
        let mut line = format!(" k={k:<2}");
        for (name, use_cos, salt) in [("cos", true, 0u64), ("margin", false, 1_000_003u64)] {
            let scores: Vec<f32> = picked
                .iter()
                .map(|r| if use_cos { r.cos[k] } else { r.margin[k] })
                .collect();
            let row_seed = seed.wrapping_add(salt).wrapping_add(k as u64);
            let b = bootstrap_auroc_ci(&scores, &labels, boot_iters, row_seed);
            line.push_str(&format!(
                "  {name} {:.3} [{:.3},{:.3}]",
                b.point, b.ci_lo, b.ci_hi
            ));
            if b.ci_lo.is_finite() && b.ci_lo > best.0 {
                best = (b.ci_lo, format!("{name}@k={k}"));
            }
        }
        println!("{line}");
    }
    best
}

fn subset(records: &[ItemRecord], keep: fn(&ItemRecord) -> bool) -> Vec<&ItemRecord> {
    records.iter().filter(|r| keep(r)).collect()
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let t_start = Instant::now();
    let cfg = RunConfig::from_env();

    let model_dir = std::env::var("KIMI_K3_MODEL_DIR")
        .unwrap_or_else(|_| format!("{}/data/kimi-k3-0.40b", env!("CARGO_MANIFEST_DIR")));
    let model_p = format!("{model_dir}/model.safetensors");
    let tik_p = format!("{model_dir}/tiktoken.model");
    if !Path::new(&model_p).exists() {
        eprintln!("FAIL: model.safetensors not found at {model_p}");
        eprintln!("      download instructions in the kimi_k3_hello_world example header");
        std::process::exit(1);
    }
    if !Path::new(&tik_p).exists() {
        eprintln!("FAIL: tiktoken.model not found at {tik_p}");
        std::process::exit(1);
    }

    println!("issue 929 loop-alignment probe runner (DiscoLoop, arXiv:2607.00341)");
    println!(
        "  posture : smoke={} entities/pool={} relations={} out_degree={} facts={} 2hop/pool={} 1hop/pool={} seed={:#x}",
        cfg.smoke as u8,
        cfg.entities,
        cfg.relations,
        cfg.out_degree,
        cfg.facts,
        cfg.two_hop_per_pool,
        cfg.one_hop_per_pool,
        cfg.seed
    );
    println!(
        "  loops L={} answer_max={} bos={} exemplars={} bootstrap_iters={} kill_bar={KILL_BAR}",
        cfg.loops, cfg.answer_max_tokens, cfg.bos as u8, cfg.exemplars, cfg.bootstrap_iters
    );
    println!("  cpu-only single thread - safe beside a GPU trainer (no GPU exclusivity needed)");

    // Tokenizer.
    let tik_bytes = match std::fs::read(&tik_p) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL: read {tik_p}: {e}");
            std::process::exit(1);
        }
    };
    let ranks = match load_tiktoken_bpe(&tik_bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FAIL: tiktoken parse: {e:?}");
            std::process::exit(1);
        }
    };
    let tokenizer = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);
    println!("  tokenizer: {} mergeable ranks", tokenizer.vocab_size());

    // Weights.
    let t_load = Instant::now();
    let weights = match load_kimi_k3(&model_p) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("FAIL: load {model_p}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "  weights  : {:.2} GB in {:.1}s",
        std::fs::metadata(&model_p).map_or(0.0, |m| m.len() as f64 / 1e9),
        t_load.elapsed().as_secs_f64()
    );

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let d = config.hidden_size;
    let vocab = config.vocab_size;
    // T1.2 precondition: the probe reads the TRUE readout operator; a tied
    // checkpoint would make head-row reads an embedding-table alias instead.
    if weights.lm_head_weight.len() != vocab * d {
        eprintln!(
            "FAIL: untied-head precondition violated: lm_head rows = {} != vocab*d = {}",
            weights.lm_head_weight.len(),
            vocab * d
        );
        std::process::exit(2);
    }
    println!("  untied head verified: lm_head [{vocab} x {d}] is the readout operator");

    // Fixture.
    let spec = FixtureSpec {
        entities_per_pool: cfg.entities,
        relations: cfg.relations,
        out_degree: cfg.out_degree,
        facts_per_prompt: cfg.facts,
        two_hop_queries: cfg.two_hop_per_pool,
        one_hop_queries: cfg.one_hop_per_pool,
        seed: cfg.seed,
    };
    let fixture = generate_two_hop_fixture(spec);
    let n_two = fixture.items.iter().filter(|i| i.is_two_hop).count();
    let n_one = fixture.items.len() - n_two;
    println!(
        "  fixture  : {} items (two-hop {n_two}, one-hop {n_one}), unique compositions/pool = {}",
        fixture.items.len(),
        fixture.unique_compositions_per_pool
    );
    if fixture.unique_compositions_per_pool == 0 && n_two == 0 {
        eprintln!("WARN: KG shape admits no unique two-hop compositions - one-hop controls only");
    }

    // Few-shot elicitation: the last `exemplars` items are rendered as solved
    // in-context examples (prompt + gold answer + blank line) and EXCLUDED
    // from the eval. Item prompts end on the bare `Answer:` cue, so appending
    // the gold name completes an exemplar exactly.
    let n_exemplars = cfg
        .exemplars
        .min(fixture.items.len().saturating_sub(2));
    let exemplar_text: String = fixture
        .items
        .iter()
        .skip(fixture.items.len() - n_exemplars)
        .map(|e| format!("{}{}\n\n", e.prompt, e.answer))
        .collect();
    let eval_len = fixture.items.len() - n_exemplars;
    if n_exemplars > 0 {
        println!(
            "  elicitation: {n_exemplars} solved exemplars prepended ({eval_len} eval items), bos={}" ,
            cfg.bos as u8
        );
    }

    let run = ProbeRun {
        config: &config,
        weights: &weights,
        tokenizer: &tokenizer,
        loops: cfg.loops,
        answer_max_tokens: cfg.answer_max_tokens,
        bos: cfg.bos,
    };

    let mut records: Vec<ItemRecord> = Vec::new();
    let mut forwards = 0usize;
    let mut tokens = 0usize;
    for item in fixture.items.iter().take(eval_len) {
        if item.answer.is_empty() {
            continue;
        }
        if let Some(cap) = cfg.max_items
            && records.len() >= cap
        {
            break;
        }
        let full_prompt = format!("{exemplar_text}{}", item.prompt);
        let prompt_tokens = tokenizer.encode(&full_prompt);
        if prompt_tokens.is_empty() {
            continue;
        }
        tokens += prompt_tokens.len();
        let outcome = run_item(&run, item, &prompt_tokens);
        forwards += prompt_tokens.len() + run.loops + outcome.generated.chars().count();
        let kind = if item.is_two_hop { "two-hop" } else { "one-hop" };
        println!(
            "[{:>3}] {:<7} ood={} pre={} first={} full={} gen={:?}",
            records.len() + 1,
            kind,
            item.is_ood as u8,
            outcome.pre_first_ok as u8,
            outcome.first_ok as u8,
            outcome.full_ok as u8,
            truncate(outcome.generated.trim_start(), 36),
        );
        records.push(ItemRecord {
            kind,
            ood: item.is_ood,
            answer: item.answer.clone(),
            cos: outcome.cos,
            margin: outcome.margin,
            pre_first_ok: outcome.pre_first_ok,
            first_ok: outcome.first_ok,
            full_ok: outcome.full_ok,
            generated: outcome.generated,
        });
    }

    if records.is_empty() {
        eprintln!("FAIL: no runnable items - fixture degenerate at this posture");
        std::process::exit(2);
    }

    let n_full = records.iter().filter(|r| r.full_ok).count();
    let n_first = records.iter().filter(|r| r.first_ok).count();
    let n_pre = records.iter().filter(|r| r.pre_first_ok).count();
    println!(
        "\nlabels: full-match {}/{}  first-token {}/{}  pre-loop-first {}/{}",
        n_full,
        records.len(),
        n_first,
        records.len(),
        n_pre,
        records.len()
    );

    // ── Analysis: AUROC grids ──────────────────────────────────────────────
    let all = subset(&records, |_| true);
    let id = subset(&records, |r| !r.ood);
    let ood = subset(&records, |r| r.ood);
    let (best_full, where_full) = auroc_table(
        "final-answer match (PRIMARY, all items)",
        &all,
        |r| r.full_ok,
        cfg.loops,
        cfg.bootstrap_iters,
        cfg.seed,
    );
    let (_best_first, _where_first) = auroc_table(
        "first-token match (secondary, all items)",
        &all,
        |r| r.first_ok,
        cfg.loops,
        cfg.bootstrap_iters,
        cfg.seed,
    );
    if !id.is_empty() {
        let _ = auroc_table("final-answer match (ID pool only)", &id, |r| r.full_ok, cfg.loops, cfg.bootstrap_iters, cfg.seed);
    }
    if !ood.is_empty() {
        let _ = auroc_table("final-answer match (OOD pool only)", &ood, |r| r.full_ok, cfg.loops, cfg.bootstrap_iters, cfg.seed);
    }

    // Mean trajectories, ID vs OOD (the cos-trajectory record T1.3 asks for).
    println!("\n== mean trajectories (ID vs OOD) ==");
    for k in 0..=cfg.loops {
        let grab = |rs: &[&ItemRecord], cos: bool| -> Vec<f32> {
            rs.iter().map(|r| if cos { r.cos[k] } else { r.margin[k] }).collect()
        };
        println!(
            " k={k:<2} cos id={:.4} ood={:.4}   margin id={:.4} ood={:.4}",
            mean_of(&grab(&id, true)),
            mean_of(&grab(&ood, true)),
            mean_of(&grab(&id, false)),
            mean_of(&grab(&ood, false)),
        );
    }

    // ── T1.4 verdict ───────────────────────────────────────────────────────
    let labels_full: Vec<bool> = records.iter().map(|r| r.full_ok).collect();
    let (pos, neg) = class_counts(&labels_full);
    println!("\n== T1.4 kill bar (bar = {KILL_BAR} on the 95% CI lower bound) ==");
    if pos == 0 || neg == 0 {
        println!(
            "  UNDECIDABLE: pos={pos} neg={neg} - widen the fixture (more one-hop controls / bigger pools) before reading a verdict"
        );
    } else {
        println!("  best row: {where_full} ci_lo={best_full:.3}");
        if best_full >= KILL_BAR {
            println!("  verdict: SIGNAL PRESENT - the alignment signal separates correct from incorrect");
            println!("           on non-loop-trained weights; Phase 2 injection arms may be drafted.");
        } else {
            println!("  verdict: SIGNAL ABSENT/WEAK - the DiscoLoop misalignment premise does not transfer");
            println!("           to the kimi-k3-0.40B caller-side loop surface; record the negative beside");
            println!("           Benches 847/850/906; reopen trigger = riir-train Plan 451 loop-trained artifact.");
        }
    }

    println!(
        "\nwall {:.1}s | prefill tokens {tokens} | ~{forwards} forwards",
        t_start.elapsed().as_secs_f64()
    );

    // ── Optional JSON record ───────────────────────────────────────────────
    if let Ok(out_p) = std::env::var("K929_OUT") {
        let doc = serde_json::json!({
            "issue": 929,
            "paper": "arXiv:2607.00341",
            "checkpoint": model_p,
            "untied_head_verified": true,
            "smoke": cfg.smoke,
            "fixture": {
                "entities_per_pool": cfg.entities,
                "relations": cfg.relations,
                "out_degree": cfg.out_degree,
                "facts_per_prompt": cfg.facts,
                "two_hop_per_pool": cfg.two_hop_per_pool,
                "one_hop_per_pool": cfg.one_hop_per_pool,
                "seed": cfg.seed,
                "unique_compositions_per_pool": fixture.unique_compositions_per_pool,
            },
            "loops": cfg.loops,
            "answer_max_tokens": cfg.answer_max_tokens,
            "bos": cfg.bos,
            "exemplars": cfg.exemplars,
            "bootstrap_iters": cfg.bootstrap_iters,
            "kill_bar": KILL_BAR,
            "labels": { "full": n_full, "first": n_first, "pre_first": n_pre, "n": records.len() },
            "verdict": {
                "best_ci_lo": best_full,
                "best_row": where_full,
                "undecidable": pos == 0 || neg == 0,
            },
            "items": records.iter().map(|r| serde_json::json!({
                "kind": r.kind,
                "ood": r.ood,
                "answer": r.answer,
                "cos": r.cos,
                "margin": r.margin,
                "pre_first_ok": r.pre_first_ok,
                "first_ok": r.first_ok,
                "full_ok": r.full_ok,
                "generated": r.generated,
            })).collect::<Vec<_>>(),
        });
        match serde_json::to_string_pretty(&doc)
            .map_err(|e| e.to_string())
            .and_then(|s| std::fs::write(&out_p, s).map_err(|e| e.to_string()))
        {
            Ok(()) => println!("json record: {out_p}"),
            Err(e) => eprintln!("WARN: json write failed ({out_p}): {e}"),
        }
    }
}
