//! Issue 883 P0 — the Kimi-K3 dashboard-only second fixture: the MLA/KDA
//! R² dashboard on real weights (the gemma-2 half is riir-infer Bench 004;
//! this is the fixture-class-null territory twin, 883 trap 4).
//!
//! **Tap definition (the MLA wrinkle).** Kimi-K3-0.40B is a hybrid: 6 KDA
//! layers (linear/delta attention — NO KV cache; the recurrent state is the
//! whole mechanism) + 2 MLA layers (0-indexed [3, 7], from
//! `full_attn_layers: [4, 8]` 1-indexed). On an MLA layer the production
//! cache stores the **128-d normed latent `c_kv`** (+ the 32-d shared rope
//! key) — K/V are NEVER materialized in the cache; they are up-projected
//! per cached token at attention time (`k_c = W_UK·c`, `v_c = W_UV·c`). So
//! the dashboard's tap — "where the cache path would consume K/V" — is the
//! up-projection of the cached latent, replayed **bit-identically** from
//! `MlaKVCache::latent_kv_at(j)` after each chunk with the same
//! `simd_matmul_rows` call the forward itself uses. Zero production-forward
//! changes; the replay IS the step-3 arithmetic on the step-5-cached input.
//!
//! **What the numbers mean here (read before quoting):** on this fixture
//! K and V are both linear functions of ONE shared latent — the
//! independent-W_K/W_V mechanism 883's products retrofit does not exist.
//! - KDA layers: no KV cache ⇒ P1 (V-quant) / P2 (retrofit) / P3 (cache
//!   halving) have no substrate to act on — **fixture-class null** (trap 4).
//! - MLA layers: the "V cache" IS the latent (160 floats/token vs 1056
//!   full-attention-equivalents — 6.6× structural compression already
//!   built in). P1/P3 are null-class for the same trap-4 reason. The ONE
//!   live question is P2 **on the explicit-up-projection path**
//!   (`mla_forward_token` up-projects per cached token per step — the
//!   V-side is 50% of that work): ρ_l(V−K) is the fit-ability read for
//!   replacing the V up-projection with `K + E_l[s]`.
//!
//! **MEASUREMENT-ONLY (the 883 P0 law): no quality claim is made.**
//!
//! # Run
//! ```sh
//! cargo bench --features "kimi_k3_loader fitted_anchor_tables" \
//!   --bench bench_889_kimi_k3_vk_dashboard -- --nocapture
//! # env: KIMI_K3_MODEL_DIR (default <repo>/data/kimi-k3-0.40b — needs the
//! # public model.safetensors from inference-optimization/Kimi-K3-0.40B);
//! # argv: [corpus-dir-or-txt] [--top-k N] [--max-tokens N] [--seq-len N]
//! #       [--report PATH]
//! ```

#![cfg(all(feature = "kimi_k3_loader", feature = "fitted_anchor_tables"))]
#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use katgpt_core::fitted_anchor_table::LayeredVkCalibration;
use katgpt_core::simd::simd_matmul_rows;
use katgpt_rs::kimi_k3::decoder_layer::{KimiAttentionState, KimiAttentionWeights};
use katgpt_rs::kimi_k3::loader::load_kimi_k3;
use katgpt_rs::kimi_k3::model::{
    KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token_traced,
};
use katgpt_rs::kimi_k3::tiktoken::{load_tiktoken_bpe, TiktokenTokenizer};

fn main() {
    // ── Args ────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut corpus_arg: Option<PathBuf> = None;
    let mut top_k: usize = 8192;
    let mut max_tokens: usize = 120_000;
    let mut seq_len: usize = 1024;
    let mut report_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--top-k" => {
                top_k = parse_arg(&args, &mut i, "--top-k");
            }
            "--max-tokens" => {
                max_tokens = parse_arg(&args, &mut i, "--max-tokens");
            }
            "--seq-len" => {
                seq_len = parse_arg(&args, &mut i, "--seq-len");
            }
            "--report" => {
                report_path = Some(PathBuf::from(next_arg(&args, &mut i, "--report")));
            }
            other if !other.starts_with("--") => {
                corpus_arg = Some(PathBuf::from(other));
                i += 1;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    if seq_len > 4096 {
        eprintln!("--seq-len must stay <= 4096 (kimi-k3 max_position_embeddings)");
        std::process::exit(2);
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR")
        .unwrap_or_else(|_| format!("{manifest_dir}/data/kimi-k3-0.40b"));
    let corpus_path = corpus_arg.unwrap_or_else(|| {
        PathBuf::from(format!("{manifest_dir}/../riir-train/data/chat_probe"))
    });

    // ── Model (REAL weights required — random weights measure Xavier noise,
    //    not the architecture's trained behavior; the dashboard would be a
    //    hollow number. Refuse loudly instead.) ─────────────────────────────
    let model_path = format!("{model_dir}/model.safetensors");
    if !Path::new(&model_path).is_file() {
        eprintln!("model weights missing: {model_path}");
        eprintln!(
            "download (public, no key):\n  curl -sL -o {model_path} \\\n    \
             https://huggingface.co/inference-optimization/Kimi-K3-0.40B/resolve/main/model.safetensors"
        );
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let weights = load_kimi_k3(&model_path).unwrap_or_else(|e| {
        eprintln!("load_kimi_k3 failed: {e}");
        std::process::exit(1);
    });
    println!(
        "# model: Kimi-K3-0.40B real weights | {:.2} GiB loaded in {:.1}s",
        std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0) as f64 / (1 << 30) as f64,
        t0.elapsed().as_secs_f32()
    );

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let mla_layers: Vec<usize> = config.mla_layer_indices.clone();
    let mla = &config.mla_config;
    let d_c = mla.kv_lora_rank;
    let d_h = mla.qk_nope_head_dim;
    let v_h = mla.v_head_dim;
    let n_h = mla.n_heads;
    if d_h != v_h {
        eprintln!(
            "tap ill-defined: qk_nope_head_dim ({d_h}) != v_head_dim ({v_h}) \
             — the elementwise V−K residual requires equal per-head dims"
        );
        std::process::exit(2);
    }
    let tap_width = d_h * n_h; // == v_h * n_h
    println!(
        "# arch: {} layers | MLA@{:?} (tap here) | KDA@{:?} (fixture-class null — no KV cache) | \
         latent d_c={d_c} d_h={d_h} d_r={} n_h={n_h}",
        config.num_layers,
        mla_layers,
        (0..config.num_layers).filter(|l| !mla_layers.contains(l)).collect::<Vec<_>>(),
        mla.qk_rope_head_dim
    );

    // ── Tokenizer + corpus ──────────────────────────────────────────────────
    let tiktoken_path = format!("{model_dir}/tiktoken.model");
    let tiktoken_bytes = std::fs::read(&tiktoken_path).unwrap_or_else(|e| {
        eprintln!("read {tiktoken_path} failed: {e}");
        std::process::exit(1);
    });
    let ranks = load_tiktoken_bpe(&tiktoken_bytes).unwrap_or_else(|e| {
        eprintln!("tiktoken parse failed: {e}");
        std::process::exit(1);
    });
    let tokenizer = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);

    let t1 = Instant::now();
    let text = load_corpus_text(&corpus_path).unwrap_or_else(|e| {
        eprintln!("corpus load failed: {e}");
        std::process::exit(1);
    });
    let all_tokens = tokenizer.encode(&text);
    println!(
        "# corpus: {} chars → {} tokens ({:.1}s) from {}",
        text.len(),
        all_tokens.len(),
        t1.elapsed().as_secs_f32(),
        corpus_path.display()
    );
    let take = all_tokens.len().min(max_tokens);
    let tokens: Vec<usize> = all_tokens[..take].to_vec();

    let mut counts = vec![0u64; tokenizer.vocab_size().max(1)];
    for &t in &tokens {
        if t < counts.len() {
            counts[t] += 1;
        }
    }

    // ── Tables: one triplet per TAPPED (MLA) layer ──────────────────────────
    let mut tables = LayeredVkCalibration::from_counts(mla_layers.len(), tap_width, counts, top_k);
    let table_bytes = mla_layers.len() * 3 * (tables.top_k + 1) * tap_width * 4;
    let total_n: u64 = tables.token_counts.iter().sum();
    let tracked_preview: u64 = {
        let mut c: Vec<u64> = tables.token_counts.clone();
        c.sort_unstable_by(|a, b| b.cmp(a));
        c.iter().take(tables.top_k).sum()
    };
    println!(
        "# tables: top_k={} of {} seen | {:.2} GiB | coverage preview: top-{} hold {:.1}% of {} tokens",
        tables.top_k,
        tables.token_counts.iter().filter(|&&c| c > 0).count(),
        table_bytes as f64 / (1 << 30) as f64,
        tables.top_k,
        100.0 * tracked_preview as f64 / total_n.max(1) as f64,
        total_n
    );

    // ── The calibration pass (chunked; taps replayed from cached latents) ──
    let mut runtime = KimiK3Runtime::new(&config, seq_len);
    let mut traj_scratch: Vec<Vec<f32>> = Vec::new();
    let mut k_c = vec![0.0f32; tap_width];
    let mut v_c = vec![0.0f32; tap_width];
    let t2 = Instant::now();
    let mut done = 0usize;
    let mut next_report = 0usize;
    for chunk in tokens.chunks(seq_len) {
        reset_runtime_caches(&config, &mut runtime);
        for &token in chunk {
            kimi_k3_forward_token_traced(&config, &weights, &mut runtime, token as u32, &mut traj_scratch);
        }
        // Tap replay — bit-identical to the forward's step-3 up-projections
        // (same simd_matmul_rows, same cached normed latents).
        for (table_idx, &model_layer) in mla_layers.iter().enumerate() {
            let KimiAttentionState::Mla(cache) = &runtime.layers[model_layer].attn_state else {
                eprintln!("layer {model_layer} is not MLA — config/weights mismatch");
                std::process::exit(2);
            };
            let KimiAttentionWeights::Mla(w) = &weights.layers[model_layer].attention else {
                eprintln!("layer {model_layer} weights are not MLA — config/weights mismatch");
                std::process::exit(2);
            };
            debug_assert_eq!(cache.seq_len, chunk.len());
            for j in 0..cache.seq_len {
                let c = cache.latent_kv_at(j);
                simd_matmul_rows(&mut k_c, &w.w_uk, c, tap_width, d_c);
                simd_matmul_rows(&mut v_c, &w.w_uv, c, v_h * n_h, d_c);
                tables.observe_layer(table_idx, chunk[j], &k_c, &v_c);
            }
        }
        done += chunk.len();
        if done >= next_report {
            let el = t2.elapsed().as_secs_f32();
            let rate = done as f32 / el.max(1e-6);
            println!(
                "# progress {}/{} tokens | {:.0} tok/s | eta {:.0} min",
                done,
                tokens.len(),
                rate,
                (tokens.len() - done) as f32 / (rate + 1e-6) / 60.0
            );
            next_report = (done + tokens.len() / 20).max(done + 1);
        }
    }
    let pass_s = t2.elapsed().as_secs_f32();
    println!(
        "# pass done: {} tokens in {:.0}s ({:.0} tok/s) — box: 4090 workstation i7-13700K, CPU lane, AC power",
        tokens.len(),
        pass_s,
        tokens.len() as f32 / pass_s.max(1e-6)
    );

    // ── The dashboard ───────────────────────────────────────────────────────
    let mut out = String::new();
    out.push_str(&format!(
        "# Issue 883 P0 — R² dashboard (second fixture): Kimi-K3-0.40B real weights, \
         MLA layers {mla_layers:?} (tap: k = W_UK·c vs v = W_UV·c, replayed from cached latents)\n\n"
    ));
    out.push_str(&format!(
        "slice: {} tokens | top_k={} | seq_len={} | tap width {} (= d_h·n_h = v_h·n_h)\n\n",
        tokens.len(),
        tables.top_k,
        seq_len,
        tap_width
    ));
    out.push_str("| model layer | ρ_l(V) | ρ_l(K) | ρ_l(V−K) | V mass | K mass | V−K mass |\n");
    out.push_str("|---|---|---|---|---|---|---|\n");
    let mut r_v_all = Vec::new();
    let mut r_k_all = Vec::new();
    let mut r_vk_all = Vec::new();
    for lt in tables.layers.iter() {
        r_v_all.push(lt.v.r_squared());
        r_k_all.push(lt.k.r_squared());
        r_vk_all.push(lt.vk.r_squared());
    }
    for ((li, ((r_v, r_k), r_vk)), model_layer) in r_v_all
        .iter()
        .zip(r_k_all.iter())
        .zip(r_vk_all.iter())
        .enumerate()
        .zip(mla_layers.iter())
    {
        if r_v.empty || r_k.empty || r_vk.empty {
            eprintln!("table {li} (model layer {model_layer}) is EMPTY — calibration wiring bug");
            std::process::exit(1);
        }
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.3} | {:.3} | {:.3} |\n",
            model_layer,
            r_v.aggregate,
            r_k.aggregate,
            r_vk.aggregate,
            r_v.tracked_mass,
            r_k.tracked_mass,
            r_vk.tracked_mass
        ));
    }
    out.push_str("\n## per-head ρ_l (all tapped layers — per-head slice d_h dims)\n\n");
    for (li, &model_layer) in mla_layers.iter().enumerate() {
        let hd = d_h;
        let row_v: Vec<String> = (0..n_h)
            .map(|h| format!("{:.4}", r_v_all[li].aggregate_over(h * hd, (h + 1) * hd)))
            .collect();
        let row_vk: Vec<String> = (0..n_h)
            .map(|h| format!("{:.4}", r_vk_all[li].aggregate_over(h * hd, (h + 1) * hd)))
            .collect();
        out.push_str(&format!("| layer {model_layer} ρ(V) | {} |\n", row_v.join(" | ")));
        out.push_str(&format!("| layer {model_layer} ρ(V−K) | {} |\n", row_vk.join(" | ")));
    }
    // Zipf coverage (the storage dial read).
    let cov = tables.layers[0].v.coverage_curve();
    out.push_str("\n## coverage(K) — first MLA layer V table (storage dial P(K)=b_w·L·K·d_v)\n\n");
    for &k in &[16usize, 64, 256, 1024, 4096, tables.top_k.min(8192)] {
        if k <= cov.len() && k > 0 {
            out.push_str(&format!("- K={k}: {:.4}\n", cov[k - 1]));
        }
    }
    let sorted = tables.layers[0].v.sorted_counts_desc();
    out.push_str("\n## top-10 n_s (first MLA layer)\n\n");
    out.push_str(&format!("- {:?}\n", &sorted[..sorted.len().min(10)]));

    let mean_rho = |rs: &[katgpt_core::fitted_anchor_table::R2Report]| {
        rs.iter().map(|r| r.aggregate as f64).sum::<f64>() / rs.len().max(1) as f64
    };
    out.push_str(&format!(
        "\nmean over the {} tapped MLA layers: ρ(V)={:.4} ρ(K)={:.4} ρ(V−K)={:.4}\n",
        mla_layers.len(),
        mean_rho(&r_v_all),
        mean_rho(&r_k_all),
        mean_rho(&r_vk_all)
    ));

    // The fixture-class analysis (883 trap 4) — the reason this fixture is
    // dashboard-only.
    out.push_str("\n## fixture-class analysis (trap 4 — read before quoting)\n\n");
    out.push_str(&format!(
        "- KDA layers {:?}: NO KV cache (fixed-size recurrent state). P1 (V-quant), \
         P2 (K=V+ retrofit), P3 (cache halving) have no substrate to act on — \
         fixture-class NULL by construction, independent of any ρ value.\n",
        (0..config.num_layers).filter(|l| !mla_layers.contains(l)).collect::<Vec<_>>()
    ));
    out.push_str(&format!(
        "- MLA layers {mla_layers:?}: the cache stores the {d_c}-d latent + {}-d shared rope \
         key ({} floats/token) — K/V are up-projected at attention time, never cached. \
         vs {} full-attention-equivalent floats/token = {:.1}× structural compression ALREADY \
         built in: P1 (separate V quant) and P3 (halving) are null-class here for the same \
         trap-4 reason.\n",
        mla.qk_rope_head_dim,
        d_c + mla.qk_rope_head_dim,
        d_h * n_h + v_h * n_h + mla.qk_rope_head_dim,
        (d_h * n_h + v_h * n_h + mla.qk_rope_head_dim) as f64 / (d_c + mla.qk_rope_head_dim) as f64
    ));
    out.push_str(
        "- The ONE live question on this fixture: P2 on the explicit-up-projection path \
         (mla_forward_token up-projects W_UK and W_UV per cached token per decode step — the \
         V-side is 50% of that work). ρ_l(V−K) above is the fit-ability read for serving \
         V := K + E_l[s] with the W_UV up-projection deleted. Under weight absorption \
         (production MLA serving), even that question dissolves — the absorbed path never \
         materializes K/V either.\n",
    );
    out.push_str("\nMEASUREMENT-ONLY (P0 law): no quality claim. The nulls above are fixture-class (trap 4), never model-class; gemma-2-2b-it (Bench 004, riir-infer) is the mechanism-bearing fixture for P1/P2/P3.\n");

    print!("{out}");
    if let Some(p) = report_path {
        std::fs::write(&p, &out).unwrap_or_else(|e| {
            eprintln!("write report {} failed: {e}", p.display());
            std::process::exit(1);
        });
        eprintln!("# report written: {}", p.display());
    }
}

/// Reset every layer's attention cache between chunks (MLA latent cache +
/// KDA recurrent state — each chunk is an independent sequence).
fn reset_runtime_caches(config: &KimiK3ModelConfig, runtime: &mut KimiK3Runtime) {
    for (layer_idx, layer_rt) in runtime.layers.iter_mut().enumerate() {
        if config.is_mla_layer(layer_idx) {
            let KimiAttentionState::Mla(cache) = &mut layer_rt.attn_state else {
                unreachable!("config says MLA but state is KDA");
            };
            cache.reset();
        } else {
            let KimiAttentionState::Kda(cache) = &mut layer_rt.attn_state else {
                unreachable!("config says KDA but state is MLA");
            };
            cache.reset();
        }
    }
}

fn next_arg(args: &[String], i: &mut usize, name: &str) -> String {
    *i += 1;
    if *i >= args.len() {
        eprintln!("{name} needs a value");
        std::process::exit(2);
    }
    let v = args[*i].clone();
    *i += 1; // step PAST the value — the caller's loop must not re-read it
    v
}

fn parse_arg(args: &[String], i: &mut usize, name: &str) -> usize {
    let v = next_arg(args, i, name);
    v.parse().unwrap_or_else(|_| {
        eprintln!("{name} needs a number, got {v}");
        std::process::exit(2);
    })
}

/// Mirror of riir-infer vk_calibration's loader: a `.txt`/`.md` file
/// directly, or a directory of HF datasets-server `page_*.json` rows
/// (`rows[].row.messages[].content` + `rows[].row.prompt`) — the sibling
/// riir-train `chat_probe` shape, the SAME corpus family as the gemma-2
/// dashboard (comparability across fixtures).
fn load_corpus_text(path: &Path) -> Result<String, String> {
    if path.is_file() {
        return std::fs::read_to_string(path).map_err(|e| e.to_string());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| format!("read_dir {}: {e}", path.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("page_") && n.ends_with(".json"))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no page_*.json under {}", path.display()));
    }
    let mut text = String::new();
    for f in &files {
        let raw = std::fs::read_to_string(f).map_err(|e| e.to_string())?;
        let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        if let Some(rows) = v.get("rows").and_then(|r| r.as_array()) {
            for row in rows {
                if let Some(msgs) = row.pointer("/row/messages").and_then(|m| m.as_array()) {
                    for m in msgs {
                        if let Some(c) = m.get("content").and_then(|c| c.as_str()) {
                            text.push_str(c);
                            text.push('\n');
                        }
                    }
                }
                if let Some(p) = row.pointer("/row/prompt").and_then(|p| p.as_str()) {
                    text.push_str(p);
                    text.push('\n');
                }
            }
        }
    }
    if text.is_empty() {
        return Err(format!("corpus text empty from {}", path.display()));
    }
    Ok(text)
}
