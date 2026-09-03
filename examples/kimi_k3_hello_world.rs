//! Kimi-K3-0.40B "hello world" — load real weights, tokenize a prompt, decode
//! N tokens, and report detailed per-phase latency + tok/s.
//!
//! This is the developer entry point for Kimi-K3 native support (Proposal 032,
//! Phase 6 GOAT COMPLETE). It mirrors the `percepta_phase0` example shape:
//! build → run → measure, with the three numbers that matter:
//!   1. weight load time (1.5 GB safetensors),
//!   2. per-token decode latency broken down by phase (embed / 8 layers /
//!      output-res / final-norm / lm_head),
//!   3. end-to-end tok/s at the configured sequence length.
//!
//! # Prerequisites
//!
//! Model files must be downloaded to `data/kimi-k3-0.40b/`:
//! ```sh
//! curl -sSL -H "Authorization: Bearer $HF_KEY" \
//!   "https://huggingface.co/inference-optimization/Kimi-K3-0.40B/resolve/main/model.safetensors" \
//!   -o data/kimi-k3-0.40b/model.safetensors
//! curl -sSL -H "Authorization: Bearer $HF_KEY" \
//!   "https://huggingface.co/inference-optimization/Kimi-K3-0.40B/resolve/main/tiktoken.model" \
//!   -o data/kimi-k3-0.40b/tiktoken.model
//! ```
//!
//! # Run
//!
//! ```sh
//! # Default prompt: "Hello"
//! cargo run --release --features kimi_k3_loader --example kimi_k3_hello_world
//!
//! # Custom prompt + decode length
//! KIMI_PROMPT="The meaning of life is" KIMI_N_TOKENS=16 \
//!   cargo run --release --features kimi_k3_loader --example kimi_k3_hello_world
//! ```

#![cfg(feature = "kimi_k3_loader")]

use std::path::Path;
use std::time::Instant;

use katgpt_rs::kimi_k3::loader::load_kimi_k3;
use katgpt_rs::kimi_k3::model::{
    ForwardTiming, KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token,
    kimi_k3_forward_token_timed,
};
use katgpt_rs::kimi_k3::tiktoken::{load_tiktoken_bpe, TiktokenTokenizer};

// ─── Helpers ────────────────────────────────────────────────────────────────

fn model_dir() -> String {
    std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        format!("{}/data/kimi-k3-0.40b", env!("CARGO_MANIFEST_DIR"))
    })
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

/// Print the per-phase timing breakdown averaged over N tokens.
fn print_phase_breakdown(timing: &ForwardTiming, n_tokens: u64) {
    let n = n_tokens.max(1) as f64;
    let total_us = timing.total_us();
    let total_ms = total_us as f64 / 1000.0 / n;

    println!("   Per-token phase breakdown (avg over {n_tokens} decode tokens):");
    println!("   {:<26} {:>10} {:>8}", "Phase", "µs/token", "% total");
    println!("   {}", "·".repeat(50));
    print_row("Embedding lookup", timing.embed_us, n, total_us);
    print_row("8 decoder layers", timing.layers_us, n, total_us);
    print_row("  (6× KDA + 2× MLA)", 0, n, total_us); // sub-label, no value
    print_row("Output attn-res", timing.output_attn_res_us, n, total_us);
    print_row("Final RMSNorm", timing.final_norm_us, n, total_us);
    print_row("LM head (163840×1024)", timing.lm_head_us, n, total_us);
    println!("   {}", "·".repeat(50));
    println!(
        "   {:<26} {:>8.2} ms {:>8}",
        "TOTAL", total_ms, "100.0%",
    );
}

fn print_row(label: &str, us_sum: u128, n: f64, total_us: u128) {
    if us_sum == 0 && !label.starts_with("  (") {
        // Skip zero rows except the sub-label
        return;
    }
    let avg_us = us_sum as f64 / n;
    let pct = if total_us > 0 {
        us_sum as f64 / total_us as f64 * 100.0
    } else {
        0.0
    };
    if label.starts_with("  (") {
        // Sub-label (architectural annotation, no value)
        println!("   {label:<26}");
    } else {
        println!("   {label:<26} {avg_us:>8.1} µs {pct:>7.1}%");
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    use std::io::Write;

let model_p = format!("{}/model.safetensors", model_dir());
    let tiktoken_p = format!("{}/tiktoken.model", model_dir());

    if !Path::new(&model_p).exists() {
        eprintln!("❌ model.safetensors not found at {model_p}");
        eprintln!("   Download instructions in the example header comment.");
        std::process::exit(1);
    }

    let prompt = std::env::var("KIMI_PROMPT").unwrap_or_else(|_| "Hello".to_string());
    let n_decode: usize = std::env::var("KIMI_N_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    println!("🤖 Kimi-K3-0.40B — hello world (native Rust forward pass)");
    bar();
    println!("   prompt    : {prompt:?}");
    println!("   n_decode  : {n_decode} tokens (greedy argmax)");
    println!("   model_dir : {}", model_dir());
    bar();

    // ── 1. Load tokenizer ──────────────────────────────────────────────────
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
        .with_special_tokens(1, 2, 0); // BOS=1, EOS=2, PAD=0 (from config.json)
    let tok_load_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
    println!("{tok_load_ms:.0} ms  (vocab={})", tokenizer.vocab_size());

    // ── 2. Load model weights (1.5 GB) ────────────────────────────────────
    print!("   loading model.safetensors ... ");
    let _ = std::io::stdout().flush();
    let t_load = Instant::now();
    let weights = load_kimi_k3(&model_p).unwrap_or_else(|e| {
        eprintln!("\n❌ load failed: {e}");
        std::process::exit(1);
    });
    let load_s = t_load.elapsed().as_secs_f64();
    let load_ms = load_s * 1000.0;
    // Use the on-disk safetensors file size for the bandwidth calc — it's the
    // true cost of mmap'ing + deserializing the whole artifact.
    let weight_bytes = std::fs::metadata(&model_p).map_or(0, |m| m.len() as usize);
    println!(
        "{load_ms:.0} ms  ({:.2} GB, {:.1} GB/s)",
        weight_bytes as f64 / 1e9,
        weight_bytes as f64 / 1e9 / load_s,
    );

    // ── 3. Config + runtime ───────────────────────────────────────────────
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let mut runtime = KimiK3Runtime::new(&config, 64);
    println!(
        "   config    : {} layers, hidden={}, vocab={}, MoE(top-2 of {}+1 shared)",
        config.num_layers,
        config.hidden_size,
        config.vocab_size,
        config.moe_config.num_experts,
    );
    bar();

    // ── 4. Tokenize prompt ────────────────────────────────────────────────
    let prompt_tokens = tokenizer.encode(&prompt);
    println!(
        "   tokenized : {:?} → {} tokens: {:?}",
        prompt, prompt_tokens.len(), prompt_tokens
    );
    bar();

    // ── 5. Prefill (process prompt tokens, no timing — just seed state) ───
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

    // ── 6. Decode N tokens with per-step + per-phase timing ───────────────
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

    // ── Alloc-free verification (debug builds only) ───────────────────────
    // The `TrackingAllocator` is installed under `cfg(debug_assertions)` in
    // `katgpt-rs/src/lib.rs`. In release builds, the alloc counters are
    // unavailable — the zero-alloc claim is verified by code review of the
    // hot path instead (no `to_vec()`, no `Vec::with_capacity` inside loops).
    //
    // We measure ONLY the `kimi_k3_forward_token_timed` call itself — the
    // surrounding `argmax` + `tokenizer.decode` + `generated_tokens.push` are
    // example bookkeeping (String allocs, Vec growth) and are NOT part of the
    // forward hot path. The dedicated `tests/kimi_k3_g4_alloc_free.rs` test
    // provides the formal gate (it runs the forward loop with zero bookkeeping).
    #[cfg(debug_assertions)]
    let mut forward_alloc_count: usize = 0;
    #[cfg(debug_assertions)]
    let mut forward_alloc_bytes: usize = 0;

    for i in 0..n_decode {
        let t_token = Instant::now();

        #[cfg(debug_assertions)]
        katgpt_core::alloc::reset_alloc_stats();

        let logits = kimi_k3_forward_token_timed(
            &config,
            &weights,
            &mut runtime,
            current_tok,
            &mut phase_total,
        );

        #[cfg(debug_assertions)]
        {
            let (c, b) = katgpt_core::alloc::get_alloc_stats();
            forward_alloc_count += c;
            forward_alloc_bytes += b;
        }

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

    // ── 7. Summary ────────────────────────────────────────────────────────
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
    println!("   generated : {} tokens", generated_tokens.len());
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
    print_phase_breakdown(&phase_total, n as u64);
    bar();

    println!(
        "   ✅ Kimi-K3-0.40B runs natively at {tok_s:.1} tok/s on this CPU.\n   \
         Model load: {load_ms:.0} ms ({:.2} GB/s effective).",
        weight_bytes as f64 / 1e9 / load_s,
    );

    // ── Alloc-free report ─────────────────────────────────────────────────
    #[cfg(debug_assertions)]
    {
        if forward_alloc_count == 0 {
            println!(
                "   ✅ ZERO-ALLOC decode hot path: 0 allocations across {n} forward calls (debug build verification)."
            );
        } else {
            println!(
                "   ⚠️  forward hot path allocated {forward_alloc_count} times ({forward_alloc_bytes} bytes) across {n} forward calls."
            );
            println!(
                "      → {:.1} allocs/forward (target: 0). Review `to_vec()` / `Vec::new()` in `kimi_k3_forward_token`.",
                forward_alloc_count as f64 / n as f64
            );
        }
    }
    #[cfg(not(debug_assertions))]
    {
        println!();
        println!("   ℹ️  Alloc count not measured in release build (TrackingAllocator is debug-only).");
        println!("      Run `cargo run --features kimi_k3_loader --example kimi_k3_hello_world` (debug)");
        println!("      to verify the zero-alloc decode hot path.");
    }

    let _ = weights; // keep the weights alive for the summary
}
