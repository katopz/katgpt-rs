//! Spike-census validation (katgpt-rs Issue 919 T2) — compares the
//! weight-derived spike census (T1 sidecars: `tests/fixtures/spike_census/
//! <model>.json`) against MEASURED per-channel activations (the riir-ai
//! `spike_census_calib_dump` SPCM sidecars, same directory) and scores the
//! census:
//!
//! - **channel precision/recall @ K** per block: do the census's top-K
//!   `(k, i)` channels (score `s`) land in the measured top-K channels by
//!   per-channel max |activation| at the FFN down-projection input (the
//!   space the census scores)?
//! - **block-level rank agreement**: Spearman between the census's
//!   `max_s` and the measured spike ratio (max/median of per-channel max)
//!   across all blocks.
//! - **magnitude context**: measured top-channel value against the block's
//!   global RMS at the same tap (the paper's "massive activation" scale
//!   check; the census's own 10×/2× thresholds live on its SCORE scale and
//!   are never re-applied to activations here).
//!
//! Measurement-only (P0 law): nothing is promoted by this read; a low
//! agreement score is a VALID result (the census stat is an approximation
//! — the dominant-rank-1 argument — and T2's job is to price it).
//!
//! Usage:
//! ```bash
//! cargo run --release -p katgpt-attn --example spike_census_validate -- \
//!     --model gemma-2-2b-it-f16
//! cargo run -p katgpt-attn --example spike_census_validate -- --self-test
//! ```

use std::path::{Path, PathBuf};

use serde_json::Value;

const FIXTURE_DIR: &str = "crates/katgpt-attn/tests/fixtures/spike_census";

// ─── SPCM v1 reader ───────────────────────────────────────────────────────

struct Spcm {
    model: String,
    n_layers: usize,
    n_taps: usize,
    widths: [usize; 4],
    observed: u64,
    /// max_abs[tap][layer][channel]
    max_abs: Vec<Vec<Vec<f32>>>,
    /// rms[tap][layer][channel]
    rms: Vec<Vec<Vec<f32>>>,
}

fn read_spcm(path: &Path) -> Result<Spcm, String> {
    let buf = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut cur = Cursor { buf: &buf, pos: 0 };
    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    if &magic != b"SPCM" {
        return Err(format!("bad magic {magic:?} in {}", path.display()));
    }
    let version: u32 = cur.read_u32()?;
    if version != 1 {
        return Err(format!("unsupported SPCM v{version}"));
    }
    let name_len: usize = cur.read_u32()? as usize;
    let model = cur.read_str(name_len)?;
    let n_layers = cur.read_u32()? as usize;
    let n_taps = cur.read_u32()? as usize;
    if n_taps != 4 && n_taps != 1 {
        // 4 = the gemma dump (FFN tap 3); 1 = the ternary dump (FFN tap 0).
        return Err(format!("expected 4 or 1 taps, got {n_taps}"));
    }
    let mut widths = [0usize; 4];
    for w in widths.iter_mut().take(n_taps) {
        *w = cur.read_u32()? as usize;
    }
    let observed = cur.read_u64()?;
    let mut max_abs = Vec::with_capacity(n_taps);
    for &w in widths.iter().take(n_taps) {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(cur.read_f32s(w)?);
        }
        max_abs.push(layers);
    }
    let mut rms = Vec::with_capacity(n_taps);
    for &w in widths.iter().take(n_taps) {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(cur.read_f32s(w)?);
        }
        rms.push(layers);
    }
    if cur.pos != buf.len() {
        return Err(format!(
            "trailing bytes in {}: {} extra",
            path.display(),
            buf.len() - cur.pos
        ));
    }
    Ok(Spcm {
        model,
        n_layers,
        n_taps,
        widths,
        observed,
        max_abs,
        rms,
    })
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.buf.len() {
            return Err("truncated SPCM".into());
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_exact(&mut self, out: &mut [u8]) -> Result<(), String> {
        out.copy_from_slice(self.take(out.len())?);
        Ok(())
    }
    fn read_u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn read_u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn read_str(&mut self, n: usize) -> Result<String, String> {
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| format!("bad name: {e}"))
    }
    fn read_f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let bytes = self.take(n * 4)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}

// ─── Statistics ───────────────────────────────────────────────────────────

/// Spearman rank correlation (ties get average ranks; guard n < 3 and
/// zero-variance inputs → None).
fn spearman(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 3 {
        return None;
    }
    let rx = ranks(xs);
    let ry = ranks(ys);
    let n = xs.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..xs.len() {
        let a = rx[i] - mx;
        let b = ry[i] - my;
        num += a * b;
        dx2 += a * a;
        dy2 += b * b;
    }
    if dx2 <= 0.0 || dy2 <= 0.0 {
        return None;
    }
    Some(num / (dx2.sqrt() * dy2.sqrt()))
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            out[idx[k]] = avg;
        }
        i = j + 1;
    }
    out
}

// ─── The comparison ───────────────────────────────────────────────────────

struct BlockRead {
    idx: usize,
    census_top: Vec<usize>,
    census_max_s: f64,
    measured_top: Vec<usize>,
    measured_top_vals: Vec<f32>,
    ratio: f64,
    p_at_k: f64,
    p_at_k_rms: f64,
    top_over_global_rms: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--self-test") {
        self_test();
        println!("# self-test PASS");
        return;
    }
    let mut model = String::from("gemma-2-2b-it-f16");
    let mut dir = PathBuf::from(FIXTURE_DIR);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = args[i + 1].clone();
                i += 2;
            }
            "--dir" => {
                dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    if let Err(e) = run(&model, &dir) {
        eprintln!("⛔ {e}");
        std::process::exit(1);
    }
}

fn run(model: &str, dir: &Path) -> Result<(), String> {
    // ── Census sidecar ──
    let census_path = dir.join(format!("{model}.json"));
    let census_raw = std::fs::read_to_string(&census_path)
        .map_err(|e| format!("read {}: {e}", census_path.display()))?;
    let census: Value = serde_json::from_str(&census_raw)
        .map_err(|e| format!("parse {}: {e}", census_path.display()))?;
    if census["schema"].as_str() != Some("spike_census_v1") {
        return Err(format!(
            "{}: not a spike_census_v1 sidecar",
            census_path.display()
        ));
    }

    // ── SPCM + blake3 verification (b3sum line shape) ──
    let spcm_path = dir.join(format!("{model}.calib.spcm"));
    let spcm = read_spcm(&spcm_path)?;
    if spcm.model != model {
        return Err(format!(
            "SPCM model '{}' != requested '{model}'",
            spcm.model
        ));
    }
    let blake_path = dir.join(format!("{model}.calib.spcm.blake3"));
    let blake_line = std::fs::read_to_string(&blake_path)
        .map_err(|e| format!("read {}: {e}", blake_path.display()))?;
    let expected_hex = blake_line
        .split_whitespace()
        .next()
        .ok_or_else(|| format!("empty blake3 line: {}", blake_path.display()))?;
    let bytes = std::fs::read(&spcm_path).map_err(|e| format!("re-read spcm: {e}"))?;
    let actual = blake3::hash(&bytes);
    if actual.to_hex().as_str() != expected_hex {
        return Err(format!(
            "BLAKE3 mismatch on {}: expected {expected_hex}, got {}",
            spcm_path.display(),
            actual.to_hex()
        ));
    }

    // ── Per-block reads (FFN down-input tap = LAST tap: index 3 in the
    // gemma dump's 4-tap files, index 0 in the ternary dump's 1-tap files) ──
    let ffn_tap = spcm.n_taps - 1;
    let blocks = census["blocks"]
        .as_array()
        .ok_or_else(|| "census: missing blocks[]".to_string())?;
    let top_k = census["top_k"].as_u64().unwrap_or(4) as usize;
    let mut reads: Vec<BlockRead> = Vec::with_capacity(blocks.len());
    for b in blocks {
        let idx = b["idx"].as_u64().ok_or("census block missing idx")? as usize;
        if idx >= spcm.n_layers {
            return Err(format!(
                "census block {idx} >= SPCM layers {}",
                spcm.n_layers
            ));
        }
        let chans = b["channels"]
            .as_array()
            .ok_or_else(|| format!("census block {idx}: missing channels[]"))?;
        let mut census_top = Vec::with_capacity(chans.len());
        for c in chans {
            let k = c["k"]
                .as_u64()
                .ok_or_else(|| format!("block {idx}: channel missing k"))?
                as usize;
            if k >= spcm.widths[ffn_tap] {
                return Err(format!(
                    "block {idx}: census channel k={k} >= tap width {}",
                    spcm.widths[ffn_tap]
                ));
            }
            census_top.push(k);
        }
        let census_max_s = b["max_s"].as_f64().unwrap_or(0.0);

        // Measured ranking at this block (descending by max_abs).
        let mut order: Vec<usize> = (0..spcm.widths[ffn_tap]).collect();
        order.sort_by(|&a, &b| {
            spcm.max_abs[ffn_tap][idx][b]
                .partial_cmp(&spcm.max_abs[ffn_tap][idx][a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sorted_vals: Vec<f32> = spcm.max_abs[ffn_tap][idx].clone();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted_vals[sorted_vals.len() / 2];
        let top1 = spcm.max_abs[ffn_tap][idx][order[0]];
        let ratio = if median > 0.0 {
            (top1 / median) as f64
        } else {
            f64::INFINITY
        };
        // Channel-level mean-square over the block → global RMS context.
        let mean_sq = spcm.rms[ffn_tap][idx]
            .iter()
            .map(|r| (*r as f64) * (*r as f64))
            .sum::<f64>()
            / spcm.widths[ffn_tap] as f64;
        let global_rms = mean_sq.sqrt() as f32;
        let hits: usize = census_top
            .iter()
            .take(top_k)
            .filter(|k| order[..top_k].contains(k))
            .count();
        let denom = census_top.len().min(top_k) as f64;
        let p_at_k = if denom > 0.0 {
            hits as f64 / denom
        } else {
            0.0
        };
        // RMS-ranking leg (closes the single-token-max confound): same P@K
        // against the per-channel RMS order.
        let mut order_rms: Vec<usize> = (0..spcm.widths[ffn_tap]).collect();
        order_rms.sort_by(|&a, &b| {
            spcm.rms[ffn_tap][idx][b]
                .partial_cmp(&spcm.rms[ffn_tap][idx][a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let hits_rms: usize = census_top
            .iter()
            .take(top_k)
            .filter(|k| order_rms[..top_k].contains(k))
            .count();
        let p_at_k_rms = if denom > 0.0 {
            hits_rms as f64 / denom
        } else {
            0.0
        };
        reads.push(BlockRead {
            idx,
            census_top,
            census_max_s,
            measured_top: order[..top_k].to_vec(),
            measured_top_vals: order[..top_k]
                .iter()
                .map(|&k| spcm.max_abs[ffn_tap][idx][k])
                .collect(),
            ratio,
            p_at_k,
            p_at_k_rms,
            top_over_global_rms: if global_rms > 0.0 {
                (top1 / global_rms) as f64
            } else {
                f64::INFINITY
            },
        });
    }

    // ── Flagged sets (the census's own vocabulary) ──
    let summary = &census["summary"];
    let flagged: Vec<usize> = summary["spike_blocks"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default();
    let emerging: Vec<usize> = summary["emerging_blocks_2x"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default();
    let locality = summary["locality"].as_str().unwrap_or("?");

    // ── Report ──
    println!(
        "# spike_census_validate: {model} | census locality={locality} spike_blocks={flagged:?} emerging={emerging:?} | SPCM observed={} tokens",
        spcm.observed
    );
    println!(
        "# block | census max_s | measured max/med | top/globalRMS | P@K | census top-K (k) | measured top-K (k, max_abs)"
    );
    for r in &reads {
        let ct: Vec<String> = r.census_top.iter().map(|k| k.to_string()).collect();
        let mt: Vec<String> = r
            .measured_top
            .iter()
            .zip(&r.measured_top_vals)
            .map(|(k, v)| format!("{k}:{v:.1}"))
            .collect();
        println!(
            "#{:<2} | {:>8.4} | {:>8.1}x | {:>6.1} | {:.2} | [{}] | [{}]",
            r.idx,
            r.census_max_s,
            r.ratio,
            r.top_over_global_rms,
            r.p_at_k,
            ct.join(", "),
            mt.join(", ")
        );
    }

    let mean_p_flagged = {
        let sel: Vec<&BlockRead> = reads
            .iter()
            .filter(|r| flagged.contains(&r.idx) || emerging.contains(&r.idx))
            .collect();
        if sel.is_empty() {
            None
        } else {
            Some(sel.iter().map(|r| r.p_at_k).sum::<f64>() / sel.len() as f64)
        }
    };
    let mean_p_rest = {
        let sel: Vec<&BlockRead> = reads
            .iter()
            .filter(|r| !flagged.contains(&r.idx) && !emerging.contains(&r.idx))
            .collect();
        if sel.is_empty() {
            None
        } else {
            Some(sel.iter().map(|r| r.p_at_k).sum::<f64>() / sel.len() as f64)
        }
    };
    let rho_s = spearman(
        &reads.iter().map(|r| r.census_max_s).collect::<Vec<_>>(),
        &reads.iter().map(|r| r.ratio).collect::<Vec<_>>(),
    );
    let rho_t = spearman(
        &reads.iter().map(|r| r.census_max_s).collect::<Vec<_>>(),
        &reads
            .iter()
            .map(|r| r.measured_top_vals[0] as f64)
            .collect::<Vec<_>>(),
    );
    println!("# ----");
    println!(
        "# P@{top_k}: flagged∪emerging mean = {} | rest mean = {} | all-blocks mean = {:.3} | RMS-ranked leg = {:.3}",
        mean_p_flagged
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a (none flagged)".into()),
        mean_p_rest
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".into()),
        reads.iter().map(|r| r.p_at_k).sum::<f64>() / reads.len() as f64,
        reads.iter().map(|r| r.p_at_k_rms).sum::<f64>() / reads.len() as f64
    );
    println!(
        "# Spearman(census max_s, measured max/median) = {}",
        rho_s
            .map(|v| format!("{v:+.3}"))
            .unwrap_or_else(|| "undefined".into())
    );
    println!(
        "# Spearman(census max_s, measured top1 max_abs) = {}",
        rho_t
            .map(|v| format!("{v:+.3}"))
            .unwrap_or_else(|| "undefined".into())
    );
    println!(
        "# magnitude context: max top/globalRMS = {:.1} (paper-class massive is O(10^2-10^3))",
        reads
            .iter()
            .map(|r| r.top_over_global_rms)
            .fold(0.0, f64::max)
    );
    Ok(())
}

// ─── Self-test (synthetic census + SPCM with a known answer) ──────────────

fn self_test() {
    let dir = std::env::temp_dir().join(format!("spike_census_validate_st_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("selftest mkdir");
    let model = "selftest";
    let n_layers = 3usize;
    let width = 64usize;

    // SPCM: block 0's measured top-4 channels are {10, 20, 30, 40}; block 1
    // shares channel 10; block 2 is flat.
    let mut max_abs = vec![vec![vec![0.5f32; width]; n_layers]; 4];
    for (li, ks) in [(0usize, [10usize, 20, 30, 40]), (1, [10, 11, 12, 13])] {
        for (rank, &k) in ks.iter().enumerate() {
            max_abs[3][li][k] = 100.0 - rank as f32 * 10.0;
        }
    }
    let rms = vec![vec![vec![1.0f32; width]; n_layers]; 4];

    // Census JSON: block 0 predicts {10, 21, 30, 40} → 3/4 in measured top-4
    // (21 is NOT in {10,20,30,40}); block 1 predicts {10, 11, 12, 13} → 4/4;
    // block 2 predicts arbitrary low channels → 0/4.
    let census_json = r#"{"schema":"spike_census_v1","arch":"x","model_file":"x.gguf","top_k":4,
"blocks":[
{"idx":0,"channels":[{"k":10,"i":0,"s":1.0},{"k":21,"i":0,"s":0.9},{"k":30,"i":0,"s":0.8},{"k":40,"i":0,"s":0.7}],"max_s":1.0},
{"idx":1,"channels":[{"k":10,"i":0,"s":1.0},{"k":11,"i":0,"s":0.9},{"k":12,"i":0,"s":0.8},{"k":13,"i":0,"s":0.7}],"max_s":0.9},
{"idx":2,"channels":[{"k":1,"i":0,"s":1.0},{"k":2,"i":0,"s":0.9},{"k":3,"i":0,"s":0.8},{"k":4,"i":0,"s":0.7}],"max_s":0.5}],
"summary":{"spike_blocks":[0],"emerging_blocks_2x":[1],"locality":"selftest"}}"#;

    // SPCM bytes (SPCM v1) + blake3 line, then run the same read path.
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"SPCM");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&(model.len() as u32).to_le_bytes());
    buf.extend_from_slice(model.as_bytes());
    buf.extend_from_slice(&(n_layers as u32).to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    for _ in 0..4 {
        buf.extend_from_slice(&(width as u32).to_le_bytes());
    }
    buf.extend_from_slice(&42u64.to_le_bytes());
    for tap in &max_abs {
        for layer in tap {
            for &v in layer {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    for tap in &rms {
        for layer in tap {
            for &v in layer {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let spcm_path = dir.join(format!("{model}.calib.spcm"));
    std::fs::write(&spcm_path, &buf).unwrap();
    std::fs::write(
        dir.join(format!("{model}.calib.spcm.blake3")),
        format!("{}  {model}.calib.spcm\n", blake3::hash(&buf).to_hex()),
    )
    .unwrap();
    let census_path = dir.join(format!("{model}.json"));
    std::fs::write(&census_path, census_json).unwrap();

    // The measured top-4 of block 0 by construction: {10, 20, 30, 40}.
    let spcm = read_spcm(&spcm_path).expect("selftest spcm");
    assert_eq!(spcm.observed, 42);
    assert_eq!(spcm.widths, [64; 4]);
    let mut order: Vec<usize> = (0..width).collect();
    order.sort_by(|&a, &b| {
        spcm.max_abs[3][0][b]
            .partial_cmp(&spcm.max_abs[3][0][a])
            .unwrap()
    });
    assert_eq!(&order[..4], &[10, 20, 30, 40], "measured ranking");
    // P@4 for the census's {10,21,30,40} against that set = 3/4.
    let hits = [10usize, 21, 30, 40]
        .iter()
        .filter(|k| order[..4].contains(k))
        .count();
    assert_eq!(hits, 3, "selftest P@4");
    // Block 1 census is exact → 4/4.
    let mut order1: Vec<usize> = (0..width).collect();
    order1.sort_by(|&a, &b| {
        spcm.max_abs[3][1][b]
            .partial_cmp(&spcm.max_abs[3][1][a])
            .unwrap()
    });
    let hits1 = [10usize, 11, 12, 13]
        .iter()
        .filter(|k| order1[..4].contains(k))
        .count();
    assert_eq!(hits1, 4, "selftest exact block");
    // Spearman sanity: perfect monotone pair.
    assert!(
        (spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 25.0, 40.0]).unwrap() - 1.0).abs() < 1e-9
    );
    // Zero-variance → None.
    assert!(spearman(&[1.0, 1.0, 1.0], &[2.0, 3.0, 4.0]).is_none());
    let _ = &rms;
    let _ = std::fs::remove_dir_all(&dir);
}
