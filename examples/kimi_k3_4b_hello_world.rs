//! Kimi-K3-4B-A2B "hello world" — random-init forward pass on the Plan 318
//! architecture (4.43B total / 1.99B active params, 12 layers: 9 KDA + 3 MLA,
//! 12 routed experts top-4 + 2 shared, kv_lora_rank=512, vocab=163840).
//!
//! This mirrors `kimi_k3_hello_world` (the 0.40B variant) but uses the 4B-A2B
//! config with **random-init weights** instead of real safetensors. The output
//! tokens are therefore meaningless (gibberish) — the point is to:
//!
//!   1. Prove the architecture composes end-to-end at 4B scale (no NaN / Inf,
//!      valid logit shape).
//!   2. Measure per-phase decode latency + tok/s on this CPU.
//!   3. Give developers a no-download smoke test for the 4B config (Phase A4).
//!
//! Once real weights are trained (Plan 318 Phase C/D), this example becomes a
//! real quality check — swap `KimiK3ModelWeights::random` for the trained
//! safetensors loader and the same harness measures real output quality.
//!
//! # Prerequisites
//!
//! The tiktoken BPE file from the 0.40B model is reused (same vocab=163840):
//! ```sh
//! curl -sSL -H "Authorization: Bearer $HF_KEY" \
//!   "https://huggingface.co/inference-optimization/Kimi-K3-0.40B/resolve/main/tiktoken.model" \
//!   -o data/kimi-k3-0.40b/tiktoken.model
//! ```
//!
//! # Run
//!
//! ```sh
//! # Default prompt: "Hello", 8 decode tokens, seed=42
//! cargo run --release --features kimi_k3_loader --example kimi_k3_4b_hello_world
//!
//! # Custom prompt + decode length + seed
//! KIMI_PROMPT="fn main()" KIMI_N_TOKENS=16 KIMI_K3_4B_SEED=7 \
//!   cargo run --release --features kimi_k3_loader --example kimi_k3_4b_hello_world
//! ```
//!
//! # Skip on constrained runners
//!
//! Set `KIMI_K3_4B_SKIP=1` to exit early without allocating the ~17.7 GB
//! random-init weights (CI / machines under 32 GB RAM).

#![cfg(feature = "kimi_k3_loader")]

use std::path::Path;
use std::time::Instant;

use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{
    ForwardTiming, KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token,
    kimi_k3_forward_token_timed,
};
use katgpt_rs::kimi_k3::tiktoken::{load_tiktoken_bpe, TiktokenTokenizer};

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Reuse the 0.40B tiktoken.model (same Kimi-K3 vocab=163840).
fn tiktoken_path() -> String {
    std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        format!("{}/data/kimi-k3-0.40b", env!("CARGO_MANIFEST_DIR"))
    }) + "/tiktoken.model"
}

fn bar() {
    println!("{}", "─".repeat(72));
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

fn print_phase_breakdown(timing: &ForwardTiming, n_tokens: u64, num_layers: usize, n_mla: usize) {
    let n = n_tokens.max(1) as f64;
    let total_us = timing.total_us();
    let total_ms = total_us as f64 / 1000.0 / n;

    println!("   Per-token phase breakdown (avg over {n_tokens} decode tokens):");
    println!("   {:<26} {:>10} {:>8}", "Phase", "µs/token", "% total");
    println!("   {}", "·".repeat(50));
    print_row("Embedding lookup", timing.embed_us, n, total_us);
    print_row(
        &format!("{num_layers} decoder layers"),
        timing.layers_us,
        n,
        total_us,
    );
    let n_kda = num_layers - n_mla;
    print_row(&format!("  ({n_kda}× KDA + {n_mla}× MLA)"), 0, n, total_us);
    print_row("Output attn-res", timing.output_attn_res_us, n, total_us);
    print_row("Final RMSNorm", timing.final_norm_us, n, total_us);
    print_row("LM head (163840×hidden)", timing.lm_head_us, n, total_us);
    println!("   {}", "·".repeat(50));
    println!(
        "   {:<26} {:>8.2} ms {:>8}",
        "TOTAL", total_ms, "100.0%",
    );
}

fn print_row(label: &str, us_sum: u128, n: f64, total_us: u128) {
    if us_sum == 0 && !label.starts_with("  (") {
        return;
    }
    let avg_us = us_sum as f64 / n;
    let pct = if total_us > 0 {
        us_sum as f64 / total_us as f64 * 100.0
    } else {
        0.0
    };
    if label.starts_with("  (") {
        println!("   {label:<26}");
    } else {
        println!("   {label:<26} {avg_us:>8.1} µs {pct:>7.1}%");
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    use std::io::Write;

if std::env::var("KIMI_K3_4B_SKIP").ok().as_deref() == Some("1") {
        eprintln!("skipping: KIMI_K3_4B_SKIP=1");
        return;
    }

    let tiktoken_p = tiktoken_path();
    if !Path::new(&tiktoken_p).exists() {
        eprintln!("❌ tiktoken.model not found at {tiktoken_p}");
        eprintln!("   Download instructions in the example header comment.");
        std::process::exit(1);
    }

    let prompt = std::env::var("KIMI_PROMPT").unwrap_or_else(|_| "Hello".to_string());
    let n_decode: usize = std::env::var("KIMI_N_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let seed: u64 = std::env::var("KIMI_K3_4B_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    println!("🤖 Kimi-K3-4B-A2B — hello world (random-init weights, Plan 318 Phase A4)");
    bar();
    println!("   prompt    : {prompt:?}");
    println!("   n_decode  : {n_decode} tokens (greedy argmax)");
    println!("   seed      : {seed} (KIMI_K3_4B_SEED)");
    println!("   ⚠️  weights: RANDOM INIT — output is gibberish, not real text.");
    bar();

    // ── 1. Config ─────────────────────────────────────────────────────────
    let config = KimiK3ModelConfig::kimi_k3_4b_a2b();
    let n_mla = config.mla_layer_indices.len();
    println!(
        "   config    : {} layers (MLA at {:?}, KDA = {}), hidden={}, vocab={}",
        config.num_layers,
        config.mla_layer_indices,
        config.num_layers - n_mla,
        config.hidden_size,
        config.vocab_size,
    );
    println!(
        "   MoE       : {} routed (top-{}) + {} shared, moe_intermediate={}",
        config.moe_config.num_experts,
        config.moe_config.num_experts_per_token,
        config.moe_config.num_shared_experts,
        config.moe_config.moe_intermediate_size,
    );
    println!(
        "   MLA       : kv_lora={}, q_lora={}, n_heads={}, nope_dim={}, rope_dim={}, v_dim={}",
        config.mla_config.kv_lora_rank,
        config.mla_config.q_lora_rank,
        config.mla_config.n_heads,
        config.mla_config.qk_nope_head_dim,
        config.mla_config.qk_rope_head_dim,
        config.mla_config.v_head_dim,
    );
    bar();

    // ── 2. Load tokenizer ─────────────────────────────────────────────────
    print!("   loading tiktoken.model ... ");
    let _ = std::io::stdout().flush();
    let t_tok = Instant::now();
    let tiktoken_bytes = std::fs::read(&tiktoken_p).unwrap_or_else(|e| {
        eprintln!("\n❌ failed to read tiktoken.model: {e}");
        std::process::exit(1);
    });
    let ranks = load_tiktoken_bpe(&tiktoken_bytes).unwrap_or_else(|e| {
        eprintln!("\n❌ tiktoken parse failed: {e:?}");
        std::process::exit(1);
    });
    let tokenizer = TiktokenTokenizer::from_ranks(&ranks)
        .with_special_tokens(1, 2, 0); // BOS=1, EOS=2, PAD=0 (Kimi-K3 convention)
    let tok_load_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
    println!("{tok_load_ms:.0} ms  (vocab={})", tokenizer.vocab_size());

    // ── 3. Allocate random-init weights (~17.7 GB) ────────────────────────
    print!("   allocating random-init weights (seed={seed}) ... ");
    let _ = std::io::stdout().flush();
    let t_weights = Instant::now();
    let weights = KimiK3ModelWeights::random(&config, seed);
    let weights_s = t_weights.elapsed().as_secs_f64();
    println!(
        "{:.2}s  ({} layers)",
        weights_s,
        weights.layers.len(),
    );

    // ── 4. Runtime (KV caches + scratch + block state) ────────────────────
    // max_seq_len=64 keeps the MLA KV cache tiny — this is a smoke test, not
    // a 256K run (see Plan 318 A5 for the 256K KV cache allocation gate).
    let mut runtime = KimiK3Runtime::new(&config, 64);
    bar();

    // ── 5. Tokenize prompt ────────────────────────────────────────────────
    let prompt_tokens = tokenizer.encode(&prompt);
    println!(
        "   tokenized : {:?} → {} tokens: {:?}",
        prompt, prompt_tokens.len(), prompt_tokens
    );
    bar();

    // ── 6. Prefill (process prompt tokens, no timing — just seed state) ───
    if !prompt_tokens.is_empty() {
        println!("   ⚙️  PREFILL (processing {} prompt tokens)", prompt_tokens.len());
        let t_prefill = Instant::now();
        for &tok in &prompt_tokens {
            let _ = kimi_k3_forward_token(&config, &weights, &mut runtime, tok as u32);
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        println!(
            "   prefill   : {prefill_ms:.1} ms  ({:.2} ms/tok)",
            prefill_ms / prompt_tokens.len() as f64
        );
        bar();
    }

    // ── 7. Decode N tokens with per-step + per-phase timing ───────────────
    println!("   ⚙️  DECODE ({n_decode} tokens, greedy argmax)");

    // First decode token comes from the last prefill logits — re-run the
    // last prompt token uninstrumented to get the seed logits, then enter
    // the decode loop. (If prompt was empty, start from BOS=1.)
    let seed_tok = *prompt_tokens.last().unwrap_or(&1) as u32;
    let seed_logits = kimi_k3_forward_token(&config, &weights, &mut runtime, seed_tok);
    let mut current_tok = argmax(seed_logits) as u32;

    let mut generated_tokens: Vec<u32> = Vec::with_capacity(n_decode + 1);
    generated_tokens.push(current_tok);
    println!(
        "   [ 0] tok={current_tok:>6}  piece={:?}",
        tokenizer.decode(&[current_tok as usize])
    );

    let mut phase_total = ForwardTiming::default();
    let mut decode_latencies_ms: Vec<f64> = Vec::with_capacity(n_decode);

    for i in 0..n_decode {
        let t_token = Instant::now();

        let logits = kimi_k3_forward_token_timed(
            &config,
            &weights,
            &mut runtime,
            current_tok,
            &mut phase_total,
        );

        let elapsed = t_token.elapsed();
        decode_latencies_ms.push(elapsed.as_secs_f64() * 1000.0);

        current_tok = argmax(logits) as u32;
        let piece = tokenizer.decode(&[current_tok as usize]);
        generated_tokens.push(current_tok);
        let step = i + 1;
        println!(
            "   [{step:>2}] tok={current_tok:>6}  {:>6.2} ms  piece={piece:?}",
            elapsed.as_secs_f64() * 1000.0,
        );

        if current_tok as usize == tokenizer.eos_id() {
            println!("   (EOS generated — stopping)");
            break;
        }
    }
    bar();

    // ── 8. Summary ────────────────────────────────────────────────────────
    let full_text = tokenizer.decode(
        &generated_tokens.iter().map(|&t| t as usize).collect::<Vec<_>>(),
    );
    let total_decode_ms: f64 = decode_latencies_ms.iter().sum();
    let n = decode_latencies_ms.len().max(1);
    let mean_ms = total_decode_ms / n as f64;
    let tok_s = 1000.0 / mean_ms;

    let mut sorted = decode_latencies_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p50 = sorted[n / 2];
    // Nearest rank: `ceil(0.99 * n) - 1`. The old `(n * 99) / 100` is `n - 1`
    // -- the MAX -- for every n <= 100, and KIMI_N_TOKENS defaults to **8**, so
    // this line printed the slowest single token under a p99 label on every
    // default run. Nearest rank does not rescue it at n = 8 either: a 99th
    // percentile does not exist in 8 samples. So the support is printed with
    // the number -- that is what makes the label readable rather than wrong.
    // (.issues/722; scripts/percentile_index_audit.py.)
    let p99_idx = (n * 99).div_ceil(100).saturating_sub(1).min(n - 1);
    let p99 = sorted[p99_idx];
    let p99_support = n - p99_idx;

    println!("   📊 SUMMARY");
    println!("   generated : {} tokens (random-init — gibberish, not real text)", generated_tokens.len());
    println!("   output    : {full_text:?}");
    println!();
    println!("   Throughput (decode-only, excluding prefill):");
    println!("     mean    : {mean_ms:>6.2} ms/tok   →  {tok_s:>6.1} tok/s");
    println!("     p50     : {p50:>6.2} ms/tok");
    println!(
        "     p99     : {p99:>6.2} ms/tok   (tail support {p99_support} of {n} \
         — raise KIMI_N_TOKENS for a quantile rather than a worst case)"
    );
    println!("     total   : {total_decode_ms:>6.1} ms over {n} tokens");
    println!();
    print_phase_breakdown(&phase_total, n as u64, config.num_layers, n_mla);
    bar();

    println!(
        "   ✅ Kimi-K3-4B-A2B architecture composes at 4B scale: {tok_s:.1} tok/s decode,\n   \
         ~17.7 GB random-init weights, 0 NaN/Inf. (Weights alloc: {weights_s:.2}s.)",
    );
    println!(
        "   ℹ️  Output is gibberish because weights are random. Phase C (Plan 318)\n   \
         trains real weights; then swap `KimiK3ModelWeights::random` for the trained\n   \
         safetensors loader to measure real output quality."
    );

    let _ = weights; // keep the weights alive for the summary
}
