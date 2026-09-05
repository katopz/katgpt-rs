//! Issue 691 — Stale-Residual Speculative Layer Pipelining POC
//! (arXiv:2608.23841 §6.3 Approach A + B — the paper's own UNTESTED
//! hypotheses; first measured verdict).
//!
//! Real Kimi-K3-0.40B weights (`data/kimi-k3-0.40b/model.safetensors`),
//! real tiktoken prompts, pure CPU. Cross-repo traces (Bonsai-27B /
//! Gemma-2-2B, dumped by riir-ai's `stale_residual_trace_dump` example)
//! extend T1 to the other model classes when present under
//! `$STALE_RESIDUAL_TRACES` (default `data/stale_residual_traces/`).
//!
//! Sections:
//! - G1 anchors (determinism, capture bit-identity)
//! - T1: per-layer residual-dominance ratios + the paper's viability bar
//!   (>50% of layers, median ratio < 0.05), K3 + external SRTR traces
//! - T2: accept-rate θ-sweep × delay-layer grid, conditional top-1
//!   preservation + KL among accepted; persistent-hazard arm (stale-written
//!   KV/KDA state survives accepts — the compounding-corruption measurement)
//! - T3: Approach-B closed-form δ-predictors (router-logit, x_in-linear),
//!   held-out R² + corrected-replay quality lift
//! - Latency: `(C+IO)/max(C,IO)` overlap model at OUR stream ratios, three
//!   regimes (RAM-resident shared-bus / disk-resident hideable / GPU H2D),
//!   accept-rate adjusted
//!
//! Run:
//! ```bash
//! cargo bench --features "kimi_k3_loader stale_residual" \
//!     --bench bench_683_stale_residual_poc -- --nocapture
//! ```

#![cfg(feature = "stale_residual")]

use katgpt_core::stale_residual::{
    AcceptGate, OverlapLatency, PAPER_RATIO_THRESHOLD, PAPER_VIABILITY_BAR, SpecOutcome,
    fraction_layers_under_paper_bar, layer_ratio_stats, sweep_cell,
};
use katgpt_rs::kimi_k3::decoder_layer::{KimiAttentionWeights, KimiFfnWeights};
use katgpt_rs::kimi_k3::loader::{KimiK3ModelWeights, load_kimi_k3};
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token};
use katgpt_rs::kimi_k3::stale_residual::{
    DeltaPredictor, StaleResidualSim, TokenCapture, capture_forward_token,
    router_logit_features,
};
use katgpt_rs::kimi_k3::tiktoken::{TiktokenRanks, TiktokenTokenizer, load_tiktoken_bpe};

const PROMPT_TOKENS: usize = 48;
const GEN_TOKENS: usize = 16;

// ─── Prompt corpus ─────────────────────────────────────────────────────────
//
// Varied register (narrative / technical / dialogue / code-adjacent) so the
// residual statistics are not one-mode. Self-authored.

const CORPUS: &[&str] = &[
    "The lighthouse keeper counted seventeen steps down to the tide pool every morning, and every morning the count came out different. He attributed this to the erosion of the cliff path, though the true cause was that he never started counting from the same step. The village below had long stopped listening to his tide reports, which was a shame, because on the morning of the storm they were entirely accurate.",
    "A hash map resolves collisions by chaining or open addressing. Chaining keeps a bucket list per slot; open addressing probes forward until an empty slot appears. Linear probing is cache friendly but suffers primary clustering. Quadratic probing reduces clustering. Robin Hood hashing equalizes probe distances by stealing from rich buckets. The load factor threshold determines when to grow the table.",
    "fn main() { let mut total: f64 = 0.0; for i in 0..1000 { total += (i as f64).sqrt() * 0.001; } println!(\"sum is {}\", total); } The program accumulates a thousandth of each square root into a running total and prints it with six decimal places of precision.",
    "\"You keep saying the machine dreams,\" the inspector said. \"I keep saying it produces output,\" the engineer answered. \"The distinction matters legally, not philosophically. If it dreams, shutting it down is cruelty. If it produces output, shutting it down is maintenance.\" The inspector wrote neither word in his notebook. Outside, the machine produced output.",
    "Photosynthesis converts light energy into chemical energy stored in glucose. The light-dependent reactions occur in the thylakoid membrane, splitting water to release oxygen. The Calvin cycle fixes carbon dioxide in the stroma. RuBisCO catalyzes the primary fixation step and is remarkably inefficient, processing only a few reactions per second, which plants compensate for with sheer abundance.",
    "The trade route crossed three deserts and one mountain range that appeared on no official map. Caravans measured distance in water, not miles: a two-skin crossing, a five-skin crossing. Maps made by surveyors were ignored in favor of the older hydrography, which is why the border treaty failed and why the town of Kesmet exists in two countries at once, depending on which well you drink from.",
    "In the beginning the committee archived everything. Then storage costs rose and they archived only summaries. Then summarization was automated and the summaries of summaries drifted until the archive described a university that had never existed, complete with a founder nobody remembered inventing. The founding date was the drift artifact: 1847, chosen by an interpolation error in 2031.",
    "Consider a queue where arrivals follow a Poisson process and service times are exponential. The M/M/1 queue has a simple closed form: mean wait is rho over mu times one minus rho, where rho is the utilization. Little's law holds regardless of distribution: the average number in the system equals the arrival rate times the average time in system, which is why the line looks longer than it feels.",
    "The instrument was tuned to the resonance of the room rather than to itself. Musicians who visited complained it sounded wrong until they stayed a week, after which every other room sounded wrong. The builder called this the second ear: the one you grow for a space. Recording engineers removed the resonance with filters and the recordings sounded dead, which proved nothing either way.",
    "Memory consolidation during sleep replays the day's hippocampal sequences to the neocortex at accelerated speed. The replay is not literal: segments are skipped, reordered, and occasionally stitched into counterfactual routes. The stitching is believed to serve credit assignment, though this remains contested and the counterfactual routes sometimes appear in later behavior.",
    "The migration tool supported three strategies: lift and shift, replatform, and rewrite. Each failed differently. Lift and shift reproduced the outages. Replatform reproduced them with new names. The rewrite was cancelled twice and finished once, by which time the business had changed shape and the requirements described a product nobody sold anymore, to customers who had retired.",
    "Salt marshes sit between the tide and the land, accumulating sediment until they sit above the tide, at which point they become meadow unless the sea rises to reclaim them. The accretion loop is tight: more grass traps more sediment, raising the marsh, which grows more grass. Engineers who channelized the creeks broke the loop and the marsh subsided within a decade.",
    "def quicksort(xs): if len(xs) <= 1: return xs; pivot = xs[len(xs) // 2]; lo = [x for x in xs if x < pivot]; eq = [x for x in xs if x == pivot]; hi = [x for x in xs if x > pivot]; return quicksort(lo) + eq + quicksort(hi). The implementation is not stable and not in place but it is short and it is correct.",
    "The negotiator's rule was never to accept the first version of anything: the first price, the first apology, the first draft of the ceasefire line. Counterintuitively, this also applied to her own proposals. When she caught herself satisfied, she assumed the satisfaction was information about fatigue, not about quality, and revised once more before sending anything to the other side.",
    "A monad is a monoid in the category of endofunctors. Practically: it is an interface with two operations that must satisfy three laws. The laws guarantee that sequencing is associative and that the identity operation does nothing, which is precisely what lets you ignore the plumbing when reasoning about the effects that the plumbing exists to sequence.",
    "Fog on the harbor swallowed the ferry schedule whole. The schedule had never been more than a suggestion, but fog made the suggestion honest. Commuters who relied on the printed times waited on principle; those who watched the water simply left when the boat appeared. Two epistemologies, one dock, and the boat came when the boat came regardless.",
];

// ─── SRTR trace loading (cross-repo handoff) ───────────────────────────────

struct SrtrTrace {
    name: String,
    n_layer: usize,
    dim: usize,
    n_pos: usize,
    embeddings: Vec<f32>,
    per_layer: Vec<Vec<f32>>, // [layer] flat [pos*dim]
}

fn load_srtr(path: &std::path::Path) -> Option<SrtrTrace> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 20 || &bytes[0..4] != b"SRTR" {
        eprintln!("[trace] {path:?}: not an SRTR file");
        return None;
    }
    let mut off = 4usize;
    let u32_at = |off: &mut usize| -> u32 {
        let v = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
        *off += 4;
        v
    };
    let version = u32_at(&mut off);
    if version != 1 {
        eprintln!("[trace] {path:?}: unsupported version {version}");
        return None;
    }
    let name_len = u32_at(&mut off) as usize;
    let name = String::from_utf8_lossy(&bytes[off..off + name_len]).into_owned();
    off += name_len;
    let n_layer = u32_at(&mut off) as usize;
    let dim = u32_at(&mut off) as usize;
    let n_pos = u32_at(&mut off) as usize;
    let vec_at = |off: &mut usize, n: usize| -> Vec<f32> {
        let n_bytes = n * 4;
        let v = bytes[*off..*off + n_bytes]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        *off += n_bytes;
        v
    };
    let embeddings = vec_at(&mut off, n_pos * dim);
    let mut per_layer = Vec::with_capacity(n_layer);
    for _ in 0..n_layer {
        per_layer.push(vec_at(&mut off, n_pos * dim));
    }
    if off != bytes.len() {
        eprintln!("[trace] {path:?}: trailing bytes (off {off}, len {})", bytes.len());
    }
    Some(SrtrTrace {
        name,
        n_layer,
        dim,
        n_pos,
        embeddings,
        per_layer,
    })
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    let t_start = std::time::Instant::now();
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Issue 691 — Stale-Residual Speculative Layer Pipelining POC");
    println!("  Kimi-K3-0.40B real weights | pure CPU | arXiv:2608.23841 §6.3");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "Config: D={}, L={}, vocab={}, MLA@{:?}, MoE experts={}",
        config.hidden_size,
        config.num_layers,
        config.vocab_size,
        config.mla_layer_indices,
        config.moe_config.num_experts
    );

    // ── Load model + tokenizer ─────────────────────────────────────────
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    });
    let model_path = format!("{model_dir}/model.safetensors");
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: requires real model.safetensors at {model_path}");
        std::process::exit(1);
    }
    print!("Loading real model.safetensors ... ");
    let t0 = std::time::Instant::now();
    let weights = load_kimi_k3(&model_path).unwrap_or_else(|e| {
        eprintln!("\n  load failed: {e}");
        std::process::exit(1);
    });
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());

    let tok_bytes = std::fs::read(format!("{model_dir}/tiktoken.model")).expect("tiktoken.model");
    let ranks: TiktokenRanks = load_tiktoken_bpe(&tok_bytes).expect("tiktoken parse");
    let tok = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);

    let prompts: Vec<Vec<u32>> = CORPUS
        .iter()
        .map(|text| {
            let mut ids = tok.encode(text);
            ids.truncate(PROMPT_TOKENS - 1);
            let mut prompt = vec![tok.bos_id() as u32];
            prompt.extend(ids.iter().map(|&i| i as u32));
            prompt
        })
        .collect();
    let n_prompts = prompts.len();
    let prompt_len = prompts[0].len();
    println!("Prompts: {n_prompts} × {prompt_len} tokens (real tiktoken)");

    let mut runtime = KimiK3Runtime::new(&config, prompt_len + GEN_TOKENS + 8);

    // ═══ G1 anchors ════════════════════════════════════════════════════
    println!("\n── G1 anchors ──");
    {
        let mut rt_a = KimiK3Runtime::new(&config, prompt_len + GEN_TOKENS + 8);
        let mut rt_b = KimiK3Runtime::new(&config, prompt_len + GEN_TOKENS + 8);
        let mut caps_a = Vec::new();
        let mut caps_b = Vec::new();
        for &t in &prompts[0] {
            caps_a.push(capture_forward_token(&config, &weights, &mut rt_a, t));
        }
        for &t in &prompts[0] {
            caps_b.push(capture_forward_token(&config, &weights, &mut rt_b, t));
        }
        let identical = caps_a
            .iter()
            .zip(caps_b.iter())
            .all(|(a, b)| a.logits == b.logits && a.x_in == b.x_in);
        println!(
            "  G1a determinism (double run):         {}",
            if identical { "PASS" } else { "FAIL" }
        );
        assert!(identical, "G1a determinism failed");

        let mut rt_ref = KimiK3Runtime::new(&config, prompt_len + GEN_TOKENS + 8);
        let ok = prompts[0]
            .iter()
            .zip(caps_a.iter())
            .all(|(&t, cap)| {
                kimi_k3_forward_token(&config, &weights, &mut rt_ref, t) == cap.logits.as_slice()
            });
        println!(
            "  G1b capture ≡ kimi_k3_forward_token:  {}",
            if ok { "PASS" } else { "FAIL" }
        );
        assert!(ok, "G1b capture bit-identity failed");
    }

    // ═══ Collect the full capture corpus ═══════════════════════════════
    print!("Collecting captures ({n_prompts}×{prompt_len} positions) ... ");
    let t0 = std::time::Instant::now();
    let mut caps: Vec<Vec<TokenCapture>> = Vec::with_capacity(n_prompts);
    for prompt in &prompts {
        runtime.reset(); // each prompt is an independent sequence
        let mut row = Vec::with_capacity(prompt.len());
        for &t in prompt {
            row.push(capture_forward_token(&config, &weights, &mut runtime, t));
        }
        caps.push(row);
    }
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());

    // ═══ T1: residual-dominance table (K3) ═════════════════════════════
    let d = config.hidden_size;
    let n_layer = config.num_layers;
    println!(
        "\n── T1: residual dominance (K3-0.40B, {} positions) ──",
        n_prompts * prompt_len
    );
    let mut streams = vec![Vec::new(); n_layer];
    for row in &caps {
        for cap in row {
            for (layer, stream) in streams.iter_mut().enumerate().take(n_layer - 1) {
                stream.extend_from_slice(&cap.x_in[layer]);
                stream.extend_from_slice(&cap.x_in[layer + 1]);
            }
            // Last layer: x_out^7 = pre-output-attn-res hidden (captured).
            streams[n_layer - 1].extend_from_slice(&cap.x_in[n_layer - 1]);
            streams[n_layer - 1].extend_from_slice(&cap.x_out_final);
        }
    }
    let stats = layer_ratio_stats(&streams, d);
    println!(
        "  {:>5} {:>9} {:>9} {:>9} {:>9} {:>5}  bar(<{})",
        "layer", "median", "mean", "min", "max", "n", PAPER_RATIO_THRESHOLD
    );
    for s in &stats {
        println!(
            "  {:>5} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>5}  {}",
            s.layer,
            s.median,
            s.mean,
            s.min,
            s.max,
            s.n,
            if s.passes_paper_bar() { "PASS" } else { "-" }
        );
    }
    let frac_k3 = fraction_layers_under_paper_bar(&stats);
    println!(
        "  K3 verdict: {:.1}% of layers under the bar (paper bar: >{:.0}%)",
        100.0 * frac_k3,
        100.0 * PAPER_VIABILITY_BAR
    );

    // ── T1 cross-model traces ──
    let trace_dir = std::env::var("STALE_RESIDUAL_TRACES").unwrap_or_else(|_| {
        format!("{}/data/stale_residual_traces", env!("CARGO_MANIFEST_DIR"))
    });
    for (file, label) in [("bonsai.srtr", "Bonsai-27B"), ("gemma2.srtr", "Gemma-2-2B")] {
        let p = std::path::Path::new(&trace_dir).join(file);
        if !p.exists() {
            println!("  [trace] {label}: absent ({}), skipped", p.display());
            continue;
        }
        match load_srtr(&p) {
            Some(tr) => {
                let rows = katgpt_core::stale_residual::residual_dominance_from_trace(
                    &tr.per_layer,
                    &tr.embeddings,
                    tr.dim,
                );
                let frac = fraction_layers_under_paper_bar(&rows);
                let medians: Vec<String> = rows
                    .iter()
                    .filter(|r| r.n > 0)
                    .map(|r| format!("L{}={:.4}", r.layer, r.median))
                    .collect();
                println!(
                    "  [trace] {} ({}): {} layers × {} pos — {:.1}% under bar",
                    tr.name,
                    label,
                    tr.n_layer,
                    tr.n_pos,
                    100.0 * frac
                );
                println!("           medians: {}", medians.join(" "));
            }
            None => println!("  [trace] {label}: FAILED to load"),
        }
    }

    // ═══ T2: θ-sweep × delay-layer grid ════════════════════════════════
    println!("\n── T2: speculative replay sweep (stale x_in^ℓ → layers ℓ+1..L) ──");
    let mut sim = StaleResidualSim::new(&config, &weights, &mut runtime);
    let mut outcomes_by_delay: Vec<Vec<SpecOutcome>> = vec![Vec::new(); n_layer - 1];
    let t0 = std::time::Instant::now();
    for row in &caps {
        for cap in row {
            for (delay, bucket) in outcomes_by_delay.iter_mut().enumerate() {
                let stale = cap.x_in[delay].clone();
                let out = sim.replay_stale(cap, delay, &stale);
                bucket.push(out.core);
            }
        }
    }
    println!(
        "  replayed {} executions in {:.1}s",
        outcomes_by_delay.iter().map(|v| v.len()).sum::<usize>(),
        t0.elapsed().as_secs_f64()
    );

    let thetas = [0.01f32, 0.02, 0.05, 0.10, 0.20, 0.50];
    println!("  {:>6} | {:>7} {:>11} {:>8}", "theta", "accept", "top1|acc", "meanKL");
    let mut cells_at_005: Vec<katgpt_core::stale_residual::SweepCell> = Vec::new();
    for &theta in &thetas {
        let mut accepts = Vec::new();
        let mut preserves = Vec::new();
        let mut kls = Vec::new();
        let mut cells = Vec::new();
        for (delay, outcomes) in outcomes_by_delay.iter().enumerate() {
            let cell = sweep_cell(outcomes, theta, delay);
            accepts.push(cell.accept_rate);
            preserves.push(cell.top1_preserve_given_accept);
            kls.push(cell.mean_kl_given_accept);
            if (theta - PAPER_RATIO_THRESHOLD).abs() < 1e-6 {
                cells_at_005.push(cell);
            }
            cells.push(cell);
        }
        println!(
            "  {:>6.2} | {:>6.1} {:>10.1} {:>8.3}",
            theta,
            100.0 * mean(&accepts),
            100.0 * mean(&preserves),
            mean(&kls)
        );
    }
    println!("  per-delay @ theta=0.05:");
    for cell in &cells_at_005 {
        println!(
            "    delay {:>2}: accept={:>5.1} top1|accept={:>5.1} KL={:>7.4} n={}",
            cell.layer,
            100.0 * cell.accept_rate,
            100.0 * cell.top1_preserve_given_accept,
            cell.mean_kl_given_accept,
            cell.n
        );
    }

    // ═══ T2b: persistent-hazard arm ═══════════════════════════════════
    println!("\n── T2b: persistent-hazard trajectory (stale KV/KDA persist on accept) ──");
    {
        let best = cells_at_005
            .iter()
            .copied()
            .max_by(|a, b| {
                let sa = a.accept_rate * a.top1_preserve_given_accept;
                let sb = b.accept_rate * b.top1_preserve_given_accept;
                sa.partial_cmp(&sb).unwrap()
            })
            .expect("cells at 0.05");
        let delay = best.layer;
        println!(
            "  delay layer {delay} (accept {:.1}, top1|accept {:.1} @ theta=0.05), {} generated tokens/prompt",
            100.0 * best.accept_rate,
            100.0 * best.top1_preserve_given_accept,
            GEN_TOKENS
        );

        let gate = AcceptGate::default();
        let run_hazard = |theta: f32, tag: &str| {
            let gate = AcceptGate { threshold: theta };
            let mut agree_total = 0usize;
            let mut compared_total = 0usize;
            let mut first_div: Vec<usize> = Vec::new();
            let mut accepted_count = 0usize;
            let mut rejected_count = 0usize;

            for prompt in &prompts {
                // True greedy trajectory.
                let mut rt_t = KimiK3Runtime::new(&config, prompt.len() + GEN_TOKENS + 4);
                for &t in prompt {
                    kimi_k3_forward_token(&config, &weights, &mut rt_t, t);
                }
                let mut true_tokens: Vec<u32> = Vec::new();
                let mut logits = rt_t.logits.clone();
                for _ in 0..GEN_TOKENS {
                    let t_next = argmax(&logits);
                    true_tokens.push(t_next);
                    logits = kimi_k3_forward_token(&config, &weights, &mut rt_t, t_next).to_vec();
                }

                // Speculative persistent trajectory.
                let mut rt_s = KimiK3Runtime::new(&config, prompt.len() + GEN_TOKENS + 4);
                for &t in prompt {
                    kimi_k3_forward_token(&config, &weights, &mut rt_s, t);
                }
                let mut spec_tokens: Vec<u32> = Vec::new();
                let mut cur_logits = rt_s.logits.clone();
                for _ in 0..GEN_TOKENS {
                    let t_next = argmax(&cur_logits);
                    spec_tokens.push(t_next);
                    let cap = capture_forward_token(&config, &weights, &mut rt_s, t_next);
                    let ratio = ratio_of(&cap, delay);
                    if gate.accepts(ratio) {
                        accepted_count += 1;
                        let mut sim_s = StaleResidualSim::new(&config, &weights, &mut rt_s);
                        cur_logits = sim_s.replay_raw(&cap, delay, &cap.x_in[delay]);
                    } else {
                        rejected_count += 1;
                        cur_logits = cap.logits.clone();
                    }
                }

                let mut fd = GEN_TOKENS;
                for (i, (a, b)) in true_tokens.iter().zip(spec_tokens.iter()).enumerate() {
                    compared_total += 1;
                    if a == b {
                        agree_total += 1;
                    } else if fd == GEN_TOKENS {
                        fd = i;
                    }
                }
                first_div.push(fd);
            }
            let agree = agree_total as f32 / compared_total.max(1) as f32;
            let n_undiv = first_div.iter().filter(|&&f| f == GEN_TOKENS).count();
            println!(
                "  [{tag}] agreement: {:.1}% over {} tokens; accepts={} rejects={}; clean trajectories {n_undiv}/{}; first-div [{}]",
                100.0 * agree,
                compared_total,
                accepted_count,
                rejected_count,
                prompts.len(),
                first_div
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        };
        run_hazard(gate.threshold, "theta=0.05 paper");
        run_hazard(0.5, "theta=0.50 loose ");
    }

    // ═══ T3: Approach B — closed-form predictors ══════════════════════
    println!("\n── T3: Approach-B delta-predictors (closed-form OLS) ──");
    {
        // Train/test split BY PROMPT (no leakage). Delay layers 1..L-2
        // (need x_in[delay+1]; MoE for router features).
        let n_train = n_prompts / 2;
        println!("  {} train / {} test prompts", n_train, n_prompts - n_train);
        let mut phi = Vec::new();

        let mut router_preds: Vec<Option<DeltaPredictor>> = vec![None; n_layer];
        let mut lin_preds: Vec<Option<DeltaPredictor>> = vec![None; n_layer];
        let t0 = std::time::Instant::now();
        for delay in 1..n_layer - 1 {
            // Router-logit predictor (paper's Approach B).
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut n = 0usize;
            for (pi, row) in caps.iter().enumerate() {
                if pi >= n_train {
                    continue;
                }
                for cap in row {
                    if router_logit_features(&config, &weights, delay, &cap.x_in[delay], &mut phi)
                    {
                        xs.extend_from_slice(&phi);
                        ys.extend_from_slice(&delta_of(cap, delay));
                        n += 1;
                    }
                }
            }
            if n > 32 {
                let t = 1 + config.moe_config.num_experts;
                router_preds[delay] = Some(DeltaPredictor::fit(&xs, &ys, n, t, d, 1e-3));
            }
            // x_in-linear predictor (linear-predictability ceiling; ridge —
            // the n < d regime is real here).
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut n = 0usize;
            for (pi, row) in caps.iter().enumerate() {
                if pi >= n_train {
                    continue;
                }
                for cap in row {
                    xs.extend_from_slice(&cap.x_in[delay]);
                    ys.extend_from_slice(&delta_of(cap, delay));
                    n += 1;
                }
            }
            if n > 32 {
                lin_preds[delay] = Some(DeltaPredictor::fit(&xs, &ys, n, d, d, 1.0));
            }
        }
        println!("  fit done ({:.1}s)", t0.elapsed().as_secs_f64());

        println!("  {:>5} {:>12} {:>14} {:>12}", "delay", "router R²(ho)", "x_in-lin R²(ho)", "router R²(is)");
        for delay in 1..n_layer - 1 {
            let r_router = heldout_r2(&caps, n_train, delay, true, router_preds[delay].as_ref(), &config, &weights);
            let r_lin = heldout_r2(&caps, n_train, delay, false, lin_preds[delay].as_ref(), &config, &weights);
            let is_r = router_preds[delay].as_ref().map(|p| p.r_squared);
            println!(
                "  {:>5} {:>12.4} {:>14.4} {:>12.4}",
                delay,
                r_router.unwrap_or(f32::NAN),
                r_lin.unwrap_or(f32::NAN),
                is_r.unwrap_or(f32::NAN)
            );
        }

        // Corrected-replay lift at the layer with the best held-out router R².
        let mut ho_r2: Vec<f32> = vec![f32::NAN; n_layer];
        for delay in 1..n_layer - 1 {
            ho_r2[delay] = heldout_r2(
                &caps,
                n_train,
                delay,
                true,
                router_preds[delay].as_ref(),
                &config,
                &weights,
            )
            .unwrap_or(f32::NEG_INFINITY);
        }
        let best_delay = (1..n_layer - 1)
            .max_by(|&a, &b| ho_r2[a].partial_cmp(&ho_r2[b]).unwrap())
            .unwrap_or(1);
        if let Some(pred) = router_preds[best_delay].as_ref().filter(|_| ho_r2[best_delay].is_finite()) {
            println!("  corrected replay @ delay {best_delay} (router predictor):");
            let mut phi = Vec::new();
            let mut corrected = Vec::new();
            let mut stale_outcomes = Vec::new();
            let mut corrected_outcomes = Vec::new();
            for row in caps.iter().skip(n_train) {
                for cap in row {
                    let stale = cap.x_in[best_delay].clone();
                    let out_s = sim.replay_stale(cap, best_delay, &stale);
                    stale_outcomes.push(out_s.core);
                    router_logit_features(&config, &weights, best_delay, &cap.x_in[best_delay], &mut phi);
                    pred.predict_into(&phi, &mut corrected);
                    for (c, s) in corrected.iter_mut().zip(cap.x_in[best_delay].iter()) {
                        *c += s;
                    }
                    let out_c = sim.replay_stale(cap, best_delay, &corrected);
                    corrected_outcomes.push(out_c.core);
                }
            }
            let cell_s = sweep_cell(&stale_outcomes, PAPER_RATIO_THRESHOLD, best_delay);
            let cell_c = sweep_cell(&corrected_outcomes, PAPER_RATIO_THRESHOLD, best_delay);
            // Unconditional aggregates — the informative comparison when the
            // gate accepts nothing at the paper's θ (then the conditional
            // cells are vacuous and say so).
            let uncond = |outs: &[SpecOutcome]| {
                let top1 = outs.iter().filter(|o| o.top1_match).count() as f32 / outs.len() as f32;
                let kl = outs.iter().map(|o| o.kl_true_given_spec).sum::<f32>() / outs.len() as f32;
                (100.0 * top1, kl)
            };
            let (t1s, kls) = uncond(&stale_outcomes);
            let (t1c, klc) = uncond(&corrected_outcomes);
            println!(
                "    gate @0.05:   stale accept={:>5.1} | corrected accept={:>5.1}  (conditional cells {})",
                100.0 * cell_s.accept_rate,
                100.0 * cell_c.accept_rate,
                if cell_s.accept_rate == 0.0 && cell_c.accept_rate == 0.0 {
                    "vacuous — nothing accepted"
                } else {
                    "above"
                }
            );
            println!(
                "    UNCONDITIONAL (all {} test execs): stale top1={:.1}% KL={:.4} | corrected top1={:.1}% KL={:.4}",
                stale_outcomes.len(),
                t1s,
                kls,
                t1c,
                klc
            );
        } else {
            println!("  no router predictor fit (insufficient samples)");
        }
        let _ = &ho_r2;
    }

    // ═══ Latency model at our stream ratios ════════════════════════════
    println!("\n── Latency model: (C+IO)/max(C,IO) at measured accept rate ──");
    {
        let accept_005 = mean(
            &cells_at_005
                .iter()
                .map(|c| c.accept_rate)
                .collect::<Vec<_>>(),
        ) as f64;
        let per_layer_params: Vec<u64> = weights
            .layers
            .iter()
            .map(|lw| {
                let mut n: u64 =
                    (lw.input_layernorm_weight.len() + lw.post_attention_layernorm_weight.len())
                        as u64;
                match &lw.attention {
                    KimiAttentionWeights::Mla(w) => {
                        n += (w.w_dkv.len()
                            + w.w_dq.len()
                            + w.w_uq.len()
                            + w.w_qr.len()
                            + w.w_uk.len()
                            + w.w_uv.len()
                            + w.w_kr.len()
                            + w.w_o.len()
                            + w.q_a_norm_weight.len()
                            + w.kv_a_norm_weight.len()) as u64;
                        if let Some(wg) = &w.w_g {
                            n += wg.len() as u64;
                        }
                    }
                    KimiAttentionWeights::Kda(w) => {
                        n += (w.q_proj.len()
                            + w.k_proj.len()
                            + w.v_proj.len()
                            + w.o_proj.len()
                            + w.q_conv_weight.len()
                            + w.k_conv_weight.len()
                            + w.v_conv_weight.len()
                            + w.f_a_proj.len()
                            + w.f_b_proj.len()
                            + w.dt_bias.len()
                            + w.a_log.len()
                            + w.beta_proj.len()
                            + w.g_proj.len()
                            + w.o_norm_weight.len()) as u64;
                    }
                }
                match &lw.ffn {
                    KimiFfnWeights::Dense(e) => {
                        n += (e.gate_proj.len() + e.up_proj.len() + e.down_proj.len()) as u64;
                    }
                    KimiFfnWeights::Moe(m) => {
                        n += m.router_weight.len() as u64;
                        for e in &m.experts {
                            n += (e.gate_proj.len() + e.up_proj.len() + e.down_proj.len()) as u64;
                        }
                        for e in &m.shared_experts {
                            n += (e.gate_proj.len() + e.up_proj.len() + e.down_proj.len()) as u64;
                        }
                        if let Some(p) = &m.routed_expert_down_proj {
                            n += p.len() as u64;
                        }
                        if let Some(p) = &m.routed_expert_up_proj {
                            n += p.len() as u64;
                        }
                    }
                }
                n
            })
            .collect();
        let avg_params =
            per_layer_params.iter().sum::<u64>() / per_layer_params.len().max(1) as u64;
        println!(
            "  avg layer params: {:.0}M | measured accept rate @0.05: {:.1}%",
            avg_params as f64 / 1e6,
            100.0 * accept_005
        );

        // (label, compute FLOP/s, bandwidth B/s, bits/weight, shared_bus)
        let regimes: &[(&str, f64, f64, f64, bool)] = &[
            ("M3 Max RAM-resident f32 (shared bus)", 80e9, 300e9, 32.0, true),
            ("Disk-resident Q4 (NVMe, hideable IO)", 80e9, 6e9, 4.6, false),
            ("Disk-resident ternary 1.58b/w (hideable)", 80e9, 6e9, 1.58, false),
            ("GPU H2D cold-thaw (hideable IO)", 200e9, 20e9, 8.0, false),
        ];
        println!("  {:>38} {:>18} {:>14}", "regime", "paper (C+IO)/max", "pair speedup");
        for (label, compute, bw, bits, shared) in regimes {
            let m = OverlapLatency {
                compute_rate: *compute,
                bandwidth: *bw,
                accept_rate: accept_005,
                rollback_factor: 1.0,
            };
            let (c, io) = m.layer_span(avg_params, *bits, 2.0);
            let sp_paper = OverlapLatency::overlap_speedup(c, io);
            let sp_pair = m.pair_speedup((c, io), (c, io), *shared);
            println!("  {label:>38} {sp_paper:>17.2}x {sp_pair:>13.2}x");
        }
    }

    println!("\nPOC complete in {:.1}s", t_start.elapsed().as_secs_f64());
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn ratio_of(cap: &TokenCapture, delay: usize) -> f32 {
    let x_in = &cap.x_in[delay];
    let x_out = &cap.x_in[delay + 1];
    let mut s = 0.0f32;
    for i in 0..x_in.len() {
        let e = x_out[i] - x_in[i];
        s += e * e;
    }
    let mut q = 0.0f32;
    for &v in x_in {
        q += v * v;
    }
    if q > 0.0 {
        (s / q).sqrt()
    } else {
        f32::INFINITY
    }
}

fn delta_of(cap: &TokenCapture, delay: usize) -> Vec<f32> {
    cap.x_in[delay + 1]
        .iter()
        .zip(cap.x_in[delay].iter())
        .map(|(o, i)| o - i)
        .collect()
}

fn heldout_r2(
    caps: &[Vec<TokenCapture>],
    n_train: usize,
    delay: usize,
    router: bool,
    pred: Option<&DeltaPredictor>,
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
) -> Option<f32> {
    let pred = pred?;
    let d = config.hidden_size;
    let mut phi = Vec::new();
    let mut samples: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for row in caps.iter().skip(n_train) {
        for cap in row {
            let ok = if router {
                router_logit_features(config, weights, delay, &cap.x_in[delay], &mut phi)
            } else {
                phi.clone_from(&cap.x_in[delay]);
                true
            };
            if ok {
                samples.push((phi.clone(), delta_of(cap, delay)));
            }
        }
    }
    if samples.is_empty() {
        return None;
    }
    let mut mean_y = vec![0.0f64; d];
    for (_, y) in &samples {
        for j in 0..d {
            mean_y[j] += y[j] as f64;
        }
    }
    for v in mean_y.iter_mut() {
        *v /= samples.len() as f64;
    }
    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    let mut out = Vec::new();
    for (phi, y) in &samples {
        pred.predict_into(phi, &mut out);
        for j in 0..d {
            let res = y[j] - out[j];
            ss_res += (res * res) as f64;
            let dev = y[j] as f64 - mean_y[j];
            ss_tot += dev * dev;
        }
    }
    if ss_tot > 0.0 {
        Some((1.0 - ss_res / ss_tot) as f32)
    } else {
        None
    }
}
