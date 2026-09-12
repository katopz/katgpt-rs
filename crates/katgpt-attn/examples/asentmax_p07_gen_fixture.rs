//! Issue 747 P0.7 — real-model ASEntmax routing-fixture GENERATOR.
//!
//! Captures the exact inputs `EntmaxRouter` consumes on a production hot
//! path — per-head post-RoPE query rows and per-block mean-pooled post-RoPE
//! key summaries — from an **actual model prefill over real text**, plus the
//! ground-truth per-block softmax attention mass (the oracle the router is
//! scored against in the G2 re-gate).
//!
//! # Model
//!
//! `Ternary-Bonsai-8B-Q2_0.gguf` — the ternary Bonsai family's 8B member
//! (`general.architecture = qwen3`: 36 layers, ALL standard full attention —
//! unlike the 27B's qwen35 hybrid, whose GDN layers need the riir-ai engine).
//! 32 q-heads / 8 kv-heads (GQA 4:1), head_dim 128, per-head QK-RMSNorm,
//! SwiGLU FFN, YaRN rope (base 1e6, factor 4.0, orig ctx 16384). Weights are
//! the custom GGUF type 42 = `Q2_0_g128`: 34 B per 128 weights (f16 group
//! scale + 32 B of 2-bit codes, LSB-first, 4/byte; decode `(q-1)·d` — the
//! layout verified against the PrismML llama.cpp fork in riir-infer-core
//! `quant/q2_0.rs`, Plan 333 / katgpt-rs Issue 578).
//!
//! # Faithfulness validation (run BEFORE trusting the fixture)
//!
//! `--validate-ppl` reproduces llama-perplexity's exact chunk semantics
//! (n_ctx=1024, non-overlapping chunks, positions [512, 1024) scored, 511
//! NLL terms per chunk) in this crate's own f32 CPU forward. Reference
//! (PrismML fork build/bin/llama-perplexity, Metal, same GGUF, same prompt
//! file): **PPL = 16.9778** over the same 2×511 scored positions. A wrong
//! dequant / rope convention / QK-norm order / tokenizer shifts PPL by whole
//! points — the match is the evidence the captured rows are the real
//! model's routing inputs, not a lookalike.
//!
//! # Capture protocol (deterministic)
//!
//! - Prompt: `tests/data/asentmax_p07_prompt.txt` (~2.1k tokens of ASCII
//!   prose, `include_str!` — byte-identical to the llama.cpp reference run)
//!   with a planted needle sentence mid-document (Bench 032 pattern).
//! - Prefill all tokens, full causal attention, f32, fixed thread
//!   partition (row-parallel GEMM: every output element sums in one fixed
//!   order — reproducible on this machine class).
//! - Sampled surface: layers {1,5,10,14,19,23,28,33} × q-heads
//!   {0,4,9,13,17,22,26,31} (one per kv group → all 8 kv-heads covered).
//! - Rows are captured at block-end steps only (t = k·64 − 1): the first k
//!   64-token blocks are complete, the router would see exactly their
//!   frozen mean-pooled summaries, and full attention at t covers exactly
//!   those k blocks — the oracle mass is exact, not extrapolated.
//! - Per row: post-RoPE query (f16), the kv-head's block summaries (f16,
//!   stored once per (layer, kv-head) stream), and the ground-truth mass
//!   of each block (f32) under full softmax attention at t.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p katgpt-attn --features asentmax_schedule \
//!   --example asentmax_p07_gen_fixture -- --validate-ppl \
//!   --model /Users/katopz/git/riir-train/data/Ternary-Bonsai-8B-Q2_0.gguf
//! cargo run --release -p katgpt-attn --features asentmax_schedule \
//!   --example asentmax_p07_gen_fixture -- --capture \
//!   --model ... --out tests/data/asentmax_p07_bonsai8b.fixture
//! ```
//!
//! Issue 747 P0.7; the replay gate lives in
//! `tests/asentmax_p07_realmodel_regate.rs`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use katgpt_attn::dash_attn::entmax_router::{EntmaxCache, EntmaxRouter};
use katgpt_attn::dash_attn::vortex_flow::{VortexFlow, VortexScratch};
use katgpt_core::simd::simd_dot_f32;

// ──────────────────────────────────────────────────────────────────────────
// f16 (no new deps: bit-exact software conversion)
// ──────────────────────────────────────────────────────────────────────────

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal f16 → normalized f32
            let mut e: i32 = -1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            (sign << 31) | (((113 - e) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

fn f32_to_f16(v: f32) -> u16 {
    let sign = if v.is_sign_negative() { 1u16 << 15 } else { 0 };
    let x = v.abs();
    if x.is_nan() {
        return 0x7e00;
    }
    if x == 0.0 {
        return sign;
    }
    if x > 65504.0 {
        return sign | 0x7c00; // saturate (matches cast semantics closely enough)
    }
    if x < 6.10352e-5 {
        // subnormal range: round to nearest half-ULP of 2^-24
        let sub = x * 16777216.0;
        return sign | if sub >= 0.5 { 1 } else { 0 };
    }
    let mut e: i32 = x.log2().floor() as i32;
    let mut frac = x / 2.0f32.powi(e) - 1.0;
    if frac < 0.0 {
        e -= 1;
        frac = x / 2.0f32.powi(e) - 1.0;
    }
    let mut m = (frac * 1024.0 + 0.5).floor() as u32;
    if m >= 1024 {
        e += 1;
        m = 0;
    }
    sign | (((e + 14) as u16) << 10) | (m as u16)
}

// ──────────────────────────────────────────────────────────────────────────
// Minimal GGUF v3 reader (read-only, the tensors this generator needs)
// ──────────────────────────────────────────────────────────────────────────

struct GgufTensorInfo {
    ne: [u64; 2], // [ne0 = cols = input dim, ne1 = rows = output dim]
    ty: u32,
    offset: u64, // relative to the data section
}

struct Gguf {
    data: Vec<u8>,
    data_start: usize,
    tensors: HashMap<String, GgufTensorInfo>,
    tokens: Vec<String>,
    merges: Vec<(String, String)>,
    u32_meta: HashMap<String, u32>,
    f32_meta: HashMap<String, f32>,
}

impl Gguf {
    fn meta_u32(&self, key: &str) -> Option<u32> {
        self.u32_meta.get(key).copied()
    }
    fn meta_f32(&self, key: &str) -> Option<f32> {
        self.f32_meta.get(key).copied()
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl Cursor<'_> {
    #[allow(dead_code)] // u8/u16 readers kept for format completeness
    fn bytes(&mut self, n: usize) -> &[u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.bytes(8).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn f64(&mut self) -> f64 {
        f64::from_bits(self.u64())
    }
    fn string(&mut self) -> String {
        let n = self.u64() as usize;
        String::from_utf8_lossy(self.bytes(n)).into_owned()
    }
}

fn read_gguf(path: &std::path::Path) -> Result<Gguf, String> {
    let t0 = Instant::now();
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    eprintln!("[gguf] read {} bytes in {:.1}s", data.len(), t0.elapsed().as_secs_f32());
    let mut c = Cursor { b: &data, p: 0 };
    if c.u32() != 0x46554747 {
        return Err("not a GGUF file".into());
    }
    let ver = c.u32();
    if ver != 3 {
        return Err(format!("gguf version {ver} != 3"));
    }
    let n_tensors = c.u64() as usize;
    let n_kv = c.u64() as usize;

    let mut gg = Gguf {
        data: Vec::new(),
        data_start: 0,
        tensors: HashMap::with_capacity(n_tensors),
        tokens: Vec::new(),
        merges: Vec::new(),
        u32_meta: HashMap::new(),
        f32_meta: HashMap::new(),
    };
    let mut tokens: Option<Vec<String>> = None;
    let mut merges: Option<Vec<(String, String)>> = None;

    for _ in 0..n_kv {
        let key = c.string();
        let ty = c.u32();
        match ty {
            0 | 1 => {
                c.bytes(1);
            }
            2 | 3 => {
                c.bytes(2);
            }
            4 | 5 => {
                gg.u32_meta.insert(key, c.u32());
            }
            6 => {
                gg.f32_meta.insert(key, c.f32());
            }
            7 => {
                c.bytes(1);
            }
            8 => {
                c.string();
            }
            10 | 11 => {
                c.bytes(8);
            }
            12 => {
                c.f64();
            }
            9 => {
                let et = c.u32();
                let n = c.u64() as usize;
                match (key.as_str(), et) {
                    ("tokenizer.ggml.tokens", 8) => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(c.string());
                        }
                        tokens = Some(v);
                    }
                    ("tokenizer.ggml.merges", 8) => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            let s = c.string();
                            // merges are stored as "a b" pairs (byte-level
                            // tokens never contain a literal space)
                            if let Some((a, b)) = s.split_once(' ') {
                                v.push((a.to_string(), b.to_string()));
                            } else {
                                v.push((s, String::new()));
                            }
                        }
                        merges = Some(v);
                    }
                    (_, 8) => {
                        for _ in 0..n {
                            c.string();
                        }
                    }
                    (_, 0 | 1 | 7) => {
                        c.bytes(n);
                    }
                    (_, 2 | 3) => {
                        c.bytes(n * 2);
                    }
                    (_, 4..=6) => {
                        c.bytes(n * 4);
                    }
                    (_, 10..=12) => {
                        c.bytes(n * 8);
                    }
                    _ => return Err(format!("array elem type {et} in {key}")),
                }
            }
            _ => return Err(format!("metadata type {ty} in {key}")),
        }
    }

    for _ in 0..n_tensors {
        let name = c.string();
        let n_dims = c.u32() as usize;
        let mut ne = [1u64, 1];
        for d in ne.iter_mut().take(n_dims) {
            *d = c.u64();
        }
        if n_dims > 2 {
            return Err(format!("tensor {name}: {n_dims} dims unsupported"));
        }
        let ty = c.u32();
        let offset = c.u64();
        gg.tensors.insert(name, GgufTensorInfo { ne, ty, offset });
    }
    gg.data_start = (c.p + 31) & !31; // llama.cpp default alignment = 32
    gg.tokens = tokens.ok_or("missing tokenizer.ggml.tokens")?;
    gg.merges = merges.ok_or("missing tokenizer.ggml.merges")?;
    gg.data = data;
    Ok(gg)
}

impl Gguf {
    /// Dequantize a tensor to row-major f32 `[rows][cols]` (GGUF ne0 = cols =
    /// fastest-varying = input dim). Supports f32 (0), f16 (1), Q2_0_g128
    /// (42/142: 34 B per 128 weights, `(q-1)·d`, LSB-first 2-bit codes).
    fn tensor_f32(&self, name: &str) -> Result<(usize, usize, Vec<f32>), String> {
        let info = self.tensors.get(name).ok_or(format!("tensor {name} missing"))?;
        let cols = info.ne[0] as usize;
        let rows = info.ne[1] as usize;
        let base = self.data_start + info.offset as usize;
        match info.ty {
            0 => {
                if base + rows * cols * 4 > self.data.len() {
                    return Err(format!("tensor {name}: truncated"));
                }
                let mut v = vec![0f32; rows * cols];
                for (i, slot) in v.iter_mut().enumerate() {
                    *slot = f32::from_le_bytes(self.data[base + i * 4..base + i * 4 + 4].try_into().unwrap());
                }
                Ok((rows, cols, v))
            }
            1 => {
                let mut v = vec![0f32; rows * cols];
                for (i, slot) in v.iter_mut().enumerate() {
                    *slot = f16_to_f32(u16::from_le_bytes(
                        self.data[base + i * 2..base + i * 2 + 2].try_into().unwrap(),
                    ));
                }
                Ok((rows, cols, v))
            }
            42 | 142 => {
                if !cols.is_multiple_of(128) {
                    return Err(format!("tensor {name}: cols {cols} not a multiple of 128"));
                }
                let bpr = cols / 128;
                let need = rows * bpr * 34;
                if base + need > self.data.len() {
                    return Err(format!("tensor {name}: need {need} B, truncated"));
                }
                let mut v = vec![0f32; rows * cols];
                for r in 0..rows {
                    for g in 0..bpr {
                        let off = base + (r * bpr + g) * 34;
                        let d = f16_to_f32(u16::from_le_bytes([self.data[off], self.data[off + 1]]));
                        let qs = &self.data[off + 2..off + 34];
                        let dst = &mut v[r * cols + g * 128..r * cols + g * 128 + 128];
                        for (j, slot) in dst.iter_mut().enumerate() {
                            let q = ((qs[j / 4] >> ((j % 4) * 2)) & 0x03) as i32;
                            *slot = (q - 1) as f32 * d;
                        }
                    }
                }
                Ok((rows, cols, v))
            }
            other => Err(format!("tensor {name}: type {other} unsupported")),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Qwen2 BPE (gpt2 byte-level + the llama.cpp qwen2 pre-tokenizer, ASCII arm)
// ──────────────────────────────────────────────────────────────────────────

/// GPT-2 byte→unicode table: printable bytes identity, the rest sequential
/// codepoints from 256 (deterministic, no unsafe).
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for (i, slot) in table.iter_mut().enumerate() {
        let b = i as u8;
        if (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b) {
            *slot = b as char;
        } else {
            *slot = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

struct Bpe {
    token_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), usize>,
    byte_char: [char; 256],
}

impl Bpe {
    fn from_gguf(gg: &Gguf) -> Self {
        let byte_char = bytes_to_unicode();
        let mut token_id = HashMap::with_capacity(gg.tokens.len());
        for (i, t) in gg.tokens.iter().enumerate() {
            token_id.insert(t.clone(), i as u32);
        }
        let mut merge_rank = HashMap::with_capacity(gg.merges.len());
        for (r, pair) in gg.merges.iter().enumerate() {
            merge_rank.insert(pair.clone(), r);
        }
        Self { token_id, merge_rank, byte_char }
    }

    /// ASCII arm of llama.cpp's qwen2 pre-tokenizer:
    /// `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
    ///   | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
    ///   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+`
    /// The prompt is ASCII by construction (documented in the fixture).
    fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let b = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        let alpha = |c: u8| c.is_ascii_alphabetic();
        let digit = |c: u8| c.is_ascii_digit();
        let space = |c: u8| matches!(c, b' ' | b'\t' | b'\n' | b'\r');
        while i < b.len() {
            let c = b[i];
            // contraction arm
            if c == b'\'' && i + 1 < b.len() {
                let n1 = b[i + 1];
                let three = i + 2 < b.len()
                    && matches!(n1, b'r' | b'R' | b'v' | b'V')
                    && matches!(b[i + 2], b'e' | b'E');
                let three_ll = i + 2 < b.len()
                    && matches!(n1, b'l' | b'L')
                    && matches!(b[i + 2], b'l' | b'L');
                if matches!(n1, b's' | b'S' | b't' | b'T' | b'm' | b'M' | b'd' | b'D') || three || three_ll {
                    let end = if three || three_ll { i + 3 } else { i + 2 };
                    out.push(&text[i..end]);
                    i = end;
                    continue;
                }
            }
            // `[^\r\n L N]? L+` — one optional non-letter/digit prefix + letters
            let mut k = i;
            if k < b.len() && !b[k].is_ascii_alphanumeric() && b[k] != b'\r' && b[k] != b'\n' {
                k += 1;
            }
            if k < b.len() && alpha(b[k]) {
                let mut e = k;
                while e < b.len() && alpha(b[e]) {
                    e += 1;
                }
                out.push(&text[i..e]);
                i = e;
                continue;
            }
            // `\p{N}` — single digit
            if digit(c) {
                out.push(&text[i..=i]);
                i += 1;
                continue;
            }
            // ` ?[^\s L N]+[\r\n]*` — optional space + punctuation run
            let mut p = i;
            if b[p] == b' ' && p + 1 < b.len() && !b[p + 1].is_ascii_alphanumeric() && !space(b[p + 1]) {
                p += 1;
            }
            if p < b.len() && !b[p].is_ascii_alphanumeric() && !space(b[p]) {
                let mut e = p;
                while e < b.len() && !b[e].is_ascii_alphanumeric() && !space(b[e]) {
                    e += 1;
                }
                while e < b.len() && (b[e] == b'\r' || b[e] == b'\n') {
                    e += 1;
                }
                out.push(&text[i..e]);
                i = e;
                continue;
            }
            // `\s*[\r\n]+` | `\s+(?!\S)` | `\s+`
            if space(c) {
                let mut e = i;
                while e < b.len() && space(b[e]) {
                    e += 1;
                }
                let run = &text[i..e];
                let has_nl = run.contains('\n') || run.contains('\r');
                if !has_nl && e < b.len() && e - i > 1 {
                    e -= 1; // leave the last space to prefix the next token
                }
                out.push(&text[i..e]);
                i = e;
                continue;
            }
            out.push(&text[i..=i]);
            i += 1;
        }
        out.shrink_to_fit();
        out
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for pre in self.pretokenize(text) {
            let mut sym: Vec<String> = pre.bytes().map(|b| self.byte_char[b as usize].to_string()).collect();
            loop {
                let mut best: Option<(usize, usize)> = None;
                for w in 0..sym.len().saturating_sub(1) {
                    if let Some(r) = self.merge_rank.get(&(sym[w].clone(), sym[w + 1].clone()))
                        && (best.is_none() || *r < best.unwrap().1)
                    {
                        best = Some((w, *r));
                    }
                }
                let Some((w, _)) = best else { break };
                let merged = format!("{}{}", sym[w], sym[w + 1]);
                sym.splice(w..w + 2, [merged]);
            }
            for s in sym {
                if let Some(id) = self.token_id.get(&s) {
                    ids.push(*id);
                }
            }
        }
        ids
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Qwen3 model (the subset the generator needs)
// ──────────────────────────────────────────────────────────────────────────

struct Q3Layer {
    attn_norm: Vec<f32>,
    q_w: Vec<f32>,   // [hidden, hidden] row-major, rows = out
    q_norm: Vec<f32>, // [head_dim]
    k_w: Vec<f32>,   // [kv_dim, hidden]
    k_norm: Vec<f32>,
    v_w: Vec<f32>,   // [kv_dim, hidden]
    o_w: Vec<f32>,   // [hidden, hidden]
    post_norm: Vec<f32>,
    gate_w: Vec<f32>, // [ffn, hidden]
    up_w: Vec<f32>,  // [ffn, hidden]
    down_w: Vec<f32>, // [hidden, ffn]
}

struct Qwen3 {
    embed: Vec<f32>, // [vocab, hidden]
    layers: Vec<Q3Layer>,
    out_norm: Vec<f32>,
    head: Vec<f32>, // [vocab, hidden]
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    hidden: usize,
    ffn: usize,
    rms_eps: f32,
}

fn load_qwen3(gg: &Gguf) -> Result<Qwen3, String> {
    let n_layer = gg.meta_u32("qwen3.block_count").ok_or("block_count")? as usize;
    let hidden = gg.meta_u32("qwen3.embedding_length").ok_or("emb")? as usize;
    let ffn = gg.meta_u32("qwen3.feed_forward_length").ok_or("ffn")? as usize;
    let n_head = gg.meta_u32("qwen3.attention.head_count").ok_or("heads")? as usize;
    let n_kv = gg.meta_u32("qwen3.attention.head_count_kv").unwrap_or(n_head as u32) as usize;
    let head_dim = gg.meta_u32("qwen3.attention.key_length").unwrap_or((hidden / n_head) as u32) as usize;
    let rms_eps = gg.meta_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
    eprintln!("[model] layers={n_layer} hidden={hidden} ffn={ffn} heads={n_head} kv={n_kv} hd={head_dim} eps={rms_eps}");

    let t0 = Instant::now();
    let (vr, _vc, embed) = gg.tensor_f32("token_embd.weight")?;
    let vocab = vr;
    let g = |n: &str| -> Result<Vec<f32>, String> { gg.tensor_f32(n).map(|(_, _, v)| v) };
    let mut layers = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        layers.push(Q3Layer {
            attn_norm: g(&format!("blk.{l}.attn_norm.weight"))?,
            q_w: g(&format!("blk.{l}.attn_q.weight"))?,
            q_norm: g(&format!("blk.{l}.attn_q_norm.weight"))?,
            k_w: g(&format!("blk.{l}.attn_k.weight"))?,
            k_norm: g(&format!("blk.{l}.attn_k_norm.weight"))?,
            v_w: g(&format!("blk.{l}.attn_v.weight"))?,
            o_w: g(&format!("blk.{l}.attn_output.weight"))?,
            post_norm: g(&format!("blk.{l}.ffn_norm.weight"))?,
            gate_w: g(&format!("blk.{l}.ffn_gate.weight"))?,
            up_w: g(&format!("blk.{l}.ffn_up.weight"))?,
            down_w: g(&format!("blk.{l}.ffn_down.weight"))?,
        });
        if l % 8 == 0 {
            eprintln!("[model] layer {l}/{n_layer} ({:.1}s)", t0.elapsed().as_secs_f32());
        }
    }
    let out_norm = g("output_norm.weight")?;
    let head = if gg.tensors.contains_key("output.weight") {
        g("output.weight")?
    } else {
        embed.clone()
    };
    eprintln!("[model] vocab={vocab}, loaded in {:.1}s", t0.elapsed().as_secs_f32());
    Ok(Qwen3 {
        embed,
        layers,
        out_norm,
        head,
        n_head,
        n_kv,
        head_dim,
        hidden,
        ffn,
        rms_eps,
    })
}

// ── YaRN rope (llama.cpp rope_yarn semantics, half-split = NEOX pairs) ──

struct RopeTable {
    cs: Vec<(f32, f32)>, // [T][hd/2]
}

fn rope_table(m: &Qwen3, gg_meta: &Gguf, t_len: usize) -> RopeTable {
    let d = m.head_dim;
    let base = gg_meta.meta_f32("qwen3.rope.freq_base").unwrap_or(10000.0);
    let factor = gg_meta.meta_f32("qwen3.rope.scaling.factor").unwrap_or(1.0);
    let orig = gg_meta.meta_u32("qwen3.rope.scaling.original_context_length").unwrap_or(0) as usize;
    // Debug toggles (P0.7 bring-up only; the validated combo is the default):
    //   ASE_P07_NO_YARN=1   plain rope (no remap, no mscale)
    //   ASE_P07_NO_MSCALE=1 yarn remap without the 1+0.1·ln(factor) magnitude
    //   ASE_P07_INTERLEAVE=1 GPT-J-style adjacent pairs instead of half-split
    //   ASE_P07_NO_QKNORM=1  skip the per-head QK RMSNorm
    let no_yarn = std::env::var("ASE_P07_NO_YARN").is_ok();
    let no_mscale = std::env::var("ASE_P07_NO_MSCALE").is_ok();
    let mscale_only = std::env::var("ASE_P07_MSCALE_ONLY").is_ok();
    let interleave = std::env::var("ASE_P07_INTERLEAVE").is_ok();
    debug_flags::set_interleave(interleave);
    debug_flags::set_no_qknorm(std::env::var("ASE_P07_NO_QKNORM").is_ok());
    let freq_scale = 1.0 / factor;
    // corr_dims (beta_fast = 32, beta_slow = 1) — llama.cpp corr_dim
    let corr_dim = |n_rot: f32| {
        (d as f32) * ((orig as f32) / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
    };
    let start = corr_dim(32.0).floor().max(0.0);
    let end = corr_dim(1.0).ceil().min(d as f32 - 1.0);
    let mscale = if factor > 1.0 { 1.0 + 0.1 * factor.ln() } else { 1.0 };
    let mut cs = Vec::with_capacity(t_len * d / 2);
    for t in 0..t_len {
        for p in 0..d / 2 {
            let theta_extrap = (t as f32) * base.powf(-(2.0 * p as f32) / d as f32);
            // llama.cpp rope_yarn_ramp (CPU ops.cpp + Metal ggml-metal.metal,
            // read from the PrismML fork): `1 − clamp((i0/2 − start)/(end − start))`
            // — PAIR index, denominator (end − start), and the ramp runs DOWN
            // with dimension (high-freq pairs ≤ start keep θ; low-freq pairs
            // ≥ end take θ·freq_scale).
            let (theta, ms) = if factor > 1.0 && !no_yarn && !mscale_only {
                let y = ((p as f32 - start) / (end - start).max(0.001)).clamp(0.0, 1.0);
                let ramp = 1.0 - y;
                (freq_scale * theta_extrap * (1.0 - ramp) + theta_extrap * ramp, if no_mscale { 1.0 } else { mscale })
            } else if mscale_only {
                (theta_extrap, mscale)
            } else {
                (theta_extrap, 1.0)
            };
            cs.push((theta.cos() * ms, theta.sin() * ms));
        }
    }
    RopeTable { cs }
}

mod debug_flags {
    use std::sync::atomic::{AtomicBool, Ordering};
    static INTERLEAVE: AtomicBool = AtomicBool::new(false);
    static NO_QKNORM: AtomicBool = AtomicBool::new(false);
    pub fn set_interleave(v: bool) {
        INTERLEAVE.store(v, Ordering::Relaxed);
    }
    pub fn interleave() -> bool {
        INTERLEAVE.load(Ordering::Relaxed)
    }
    pub fn set_no_qknorm(v: bool) {
        NO_QKNORM.store(v, Ordering::Relaxed);
    }
    pub fn no_qknorm() -> bool {
        NO_QKNORM.load(Ordering::Relaxed)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Forward (row-parallel, deterministic) + capture
// ──────────────────────────────────────────────────────────────────────────

/// Row-parallel GEMM: `Y[T×out] = X[T×in] · W[out×in]^T`. Each thread owns
/// disjoint output-column ranges; every output element sums in one fixed
/// order regardless of scheduling — reproducible.
fn gemm(x: &[f32], t_len: usize, w: &[f32], out_dim: usize, in_dim: usize, n_threads: usize) -> Vec<f32> {
    let mut y = vec![0f32; t_len * out_dim];
    let chunk = out_dim.div_ceil(n_threads);
    let n_parts = out_dim.div_ceil(chunk);
    let mut parts: Vec<Vec<f32>> = Vec::with_capacity(n_parts);
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_parts);
        for ti in 0..n_parts {
            let o0 = ti * chunk;
            let o1 = (o0 + chunk).min(out_dim);
            let wrows = &w[o0 * in_dim..o1 * in_dim];
            handles.push(s.spawn(move || {
                let width = o1 - o0;
                let mut part = vec![0f32; t_len * width];
                for (di, wrow) in wrows.chunks_exact(in_dim).enumerate() {
                    for t in 0..t_len {
                        part[t * width + di] = simd_dot_f32(&x[t * in_dim..(t + 1) * in_dim], wrow, in_dim);
                    }
                }
                part
            }));
        }
        for h in handles {
            parts.push(h.join().expect("gemm thread"));
        }
    });
    for (ti, part) in parts.iter().enumerate() {
        let o0 = ti * chunk;
        let width = part.len() / t_len;
        for t in 0..t_len {
            y[t * out_dim + o0..t * out_dim + o0 + width].copy_from_slice(&part[t * width..(t + 1) * width]);
        }
    }
    y
}

fn rmsnorm_into(x: &[f32], g: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv * g[i];
    }
}

/// Half-split (NEOX) rope in place over one head's slice. The interleave
/// debug arm applies GPT-J-style adjacent pairs instead.
fn apply_rope_half_split(sl: &mut [f32], rt: &[(f32, f32)]) {
    let hd = sl.len();
    if debug_flags::interleave() {
        for p in 0..hd / 2 {
            let (ci, si) = rt[p];
            let (a, b) = (sl[2 * p], sl[2 * p + 1]);
            sl[2 * p] = a * ci - b * si;
            sl[2 * p + 1] = a * si + b * ci;
        }
    } else {
        for p in 0..hd / 2 {
            let (ci, si) = rt[p];
            let (a, b) = (sl[p], sl[p + hd / 2]);
            sl[p] = a * ci - b * si;
            sl[p + hd / 2] = a * si + b * ci;
        }
    }
}

const BLOCK: usize = 64; // DashAttnConfig::default().chunk_size

struct CaptureSpec {
    layers: Vec<usize>,
    q_heads: Vec<usize>,
    k_ends: std::collections::HashSet<usize>,
}

struct Summaries {
    layer: usize,
    kv_head: usize,
    n_blocks: usize,
    sums: Vec<f32>, // [n_blocks][head_dim]
}

struct CapturedRow {
    layer: usize,
    q_head: usize,
    kv_head: usize,
    n_blocks: usize,
    query: Vec<f32>,  // [head_dim] post-rope query at t
    masses: Vec<f32>, // [n_blocks] ground-truth softmax mass per block
}

/// One full prefill. Returns (summary streams, captured rows, final hidden).
fn prefill_capture(
    m: &Qwen3,
    rope: &RopeTable,
    tokens: &[u32],
    cap: &CaptureSpec,
    n_threads: usize,
) -> (Vec<Summaries>, Vec<CapturedRow>, Vec<f32>) {
    let t_len = tokens.len();
    let hidden = m.hidden;
    let hd = m.head_dim;
    let kvd = m.n_kv * hd;

    let mut x = vec![0f32; t_len * hidden];
    for (t, &tok) in tokens.iter().enumerate() {
        x[t * hidden..(t + 1) * hidden]
            .copy_from_slice(&m.embed[tok as usize * hidden..(tok as usize + 1) * hidden]);
    }

    let mut summaries: Vec<Summaries> = Vec::new();
    let mut rows: Vec<CapturedRow> = Vec::new();
    let mut xn = vec![0f32; t_len * hidden];

    for (li, layer) in m.layers.iter().enumerate() {
        let t0 = Instant::now();
        let sampled = cap.layers.contains(&li);
        for t in 0..t_len {
            rmsnorm_into(&x[t * hidden..(t + 1) * hidden], &layer.attn_norm, m.rms_eps, &mut xn[t * hidden..(t + 1) * hidden]);
        }
        let q = gemm(&xn, t_len, &layer.q_w, hidden, hidden, n_threads);
        let k = gemm(&xn, t_len, &layer.k_w, kvd, hidden, n_threads);
        let v = gemm(&xn, t_len, &layer.v_w, kvd, hidden, n_threads);

        // per-head QK RMSNorm then rope (half-split pairs)
        let mut qr = q;
        let mut kr = k;
        for t in 0..t_len {
            let rt = &rope.cs[t * hd / 2..(t + 1) * hd / 2];
            for h in 0..m.n_head {
                let s = t * hidden + h * hd;
                if debug_flags::no_qknorm() {
                    // skip QK norm (debug arm)
                } else {
                    let tmp = qr[s..s + hd].to_vec();
                    rmsnorm_into(&tmp, &layer.q_norm, m.rms_eps, &mut qr[s..s + hd]);
                }
                apply_rope_half_split(&mut qr[s..s + hd], rt);
            }
            for h in 0..m.n_kv {
                let s = t * kvd + h * hd;
                if debug_flags::no_qknorm() {
                    // skip QK norm (debug arm)
                } else {
                    let tmp = kr[s..s + hd].to_vec();
                    rmsnorm_into(&tmp, &layer.k_norm, m.rms_eps, &mut kr[s..s + hd]);
                }
                apply_rope_half_split(&mut kr[s..s + hd], rt);
            }
        }

        // causal attention per q-head (f32, max-subtracted softmax)
        let mut attn_out = vec![0f32; t_len * hidden];
        let scale = 1.0 / (hd as f32).sqrt();
        let gqa = m.n_head / m.n_kv;
        for h in 0..m.n_head {
            let kvh = h / gqa;
            let mut scores = Vec::with_capacity(t_len);
            for t in 0..t_len {
                let qv = &qr[t * hidden + h * hd..t * hidden + (h + 1) * hd];
                scores.clear();
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=t {
                    let kv = &kr[j * kvd + kvh * hd..j * kvd + (kvh + 1) * hd];
                    let sc = simd_dot_f32(qv, kv, hd) * scale;
                    scores.push(sc);
                    if sc > mx {
                        mx = sc;
                    }
                }
                let mut z = 0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - mx).exp();
                    z += *sc;
                }
                let invz = 1.0 / z;
                let want_capture = sampled
                    && cap.q_heads.contains(&h)
                    && {
                        let n_done = (t + 1) / BLOCK;
                        t + 1 == n_done * BLOCK && cap.k_ends.contains(&n_done)
                    };
                if want_capture {
                    let n_blocks = (t + 1) / BLOCK;
                    let mut masses = vec![0f32; n_blocks];
                    for (j, &e) in scores.iter().enumerate() {
                        masses[j / BLOCK] += e * invz;
                    }
                    rows.push(CapturedRow {
                        layer: li,
                        q_head: h,
                        kv_head: kvh,
                        n_blocks,
                        query: qv.to_vec(),
                        masses,
                    });
                }
                for (j, &e) in scores.iter().enumerate() {
                    if e <= 0.0 {
                        continue;
                    }
                    let wgt = e * invz;
                    for (d, ov) in v[j * kvd + kvh * hd..j * kvd + (kvh + 1) * hd].iter().enumerate() {
                        attn_out[t * hidden + h * hd + d] += wgt * ov;
                    }
                }
            }
        }

        if sampled {
            let n_blocks = t_len / BLOCK;
            for kvh in 0..m.n_kv {
                let mut sums = vec![0f32; n_blocks * hd];
                for b in 0..n_blocks {
                    for t in b * BLOCK..(b + 1) * BLOCK {
                        for (d, sv) in kr[t * kvd + kvh * hd..t * kvd + (kvh + 1) * hd].iter().enumerate() {
                            sums[b * hd + d] += sv;
                        }
                    }
                    for d in 0..hd {
                        sums[b * hd + d] /= BLOCK as f32;
                    }
                }
                summaries.push(Summaries { layer: li, kv_head: kvh, n_blocks, sums });
            }
        }

        let o = gemm(&attn_out, t_len, &layer.o_w, hidden, hidden, n_threads);
        for (xo, oo) in x.iter_mut().zip(o.iter()) {
            *xo += oo;
        }
        for t in 0..t_len {
            rmsnorm_into(&x[t * hidden..(t + 1) * hidden], &layer.post_norm, m.rms_eps, &mut xn[t * hidden..(t + 1) * hidden]);
        }
        let g = gemm(&xn, t_len, &layer.gate_w, m.ffn, hidden, n_threads);
        let u = gemm(&xn, t_len, &layer.up_w, m.ffn, hidden, n_threads);
        let mut act = vec![0f32; t_len * m.ffn];
        for (a, (gv, uv)) in act.iter_mut().zip(g.iter().zip(u.iter())) {
            let silu = gv / (1.0 + (-gv).exp());
            *a = silu * uv;
        }
        let d = gemm(&act, t_len, &layer.down_w, hidden, m.ffn, n_threads);
        for (xo, dv) in x.iter_mut().zip(d.iter()) {
            *xo += dv;
        }
        let last_norm: f32 = x[(t_len - 1) * hidden..t_len * hidden]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        eprintln!(
            "[prefill] layer {}/{} ({:.1}s, rows {}, |x_last|={last_norm:.3})",
            li + 1,
            m.layers.len(),
            t0.elapsed().as_secs_f32(),
            rows.len()
        );
    }
    (summaries, rows, x)
}

// ──────────────────────────────────────────────────────────────────────────
// Fixture writer
// ──────────────────────────────────────────────────────────────────────────

/// FNV-1a (provenance drift sentinel; not crypto).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn write_fixture(
    path: &std::path::Path,
    head_dim: usize,
    summaries: &[Summaries],
    rows: &[CapturedRow],
    prompt: &str,
    n_tokens: usize,
    needle_block: usize,
) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 20);
    buf.extend_from_slice(b"ASEP07\0\0");
    buf.extend_from_slice(&1u32.to_le_bytes()); // format version
    buf.extend_from_slice(&(head_dim as u32).to_le_bytes());
    buf.extend_from_slice(&(BLOCK as u32).to_le_bytes());
    buf.extend_from_slice(&(summaries.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(n_tokens as u32).to_le_bytes());
    buf.extend_from_slice(&fnv1a(prompt.as_bytes()).to_le_bytes());
    // reserved → needle block index (the planted-fact sentence's 64-token
    // block — the Bench 032 needle-retention axis)
    buf.extend_from_slice(&(needle_block as u64).to_le_bytes());
    for s in summaries {
        buf.extend_from_slice(&(s.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(s.kv_head as u32).to_le_bytes());
        buf.extend_from_slice(&(s.n_blocks as u32).to_le_bytes());
        for &f in &s.sums {
            buf.extend_from_slice(&f32_to_f16(f).to_le_bytes());
        }
    }
    for r in rows {
        buf.extend_from_slice(&(r.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(r.q_head as u32).to_le_bytes());
        buf.extend_from_slice(&(r.kv_head as u32).to_le_bytes());
        buf.extend_from_slice(&(r.n_blocks as u32).to_le_bytes());
        for &f in &r.query {
            buf.extend_from_slice(&f32_to_f16(f).to_le_bytes());
        }
        for &f in &r.masses {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    std::fs::write(path, &buf)?;
    eprintln!(
        "[fixture] {} rows / {} streams / {} bytes → {}",
        rows.len(),
        summaries.len(),
        buf.len(),
        path.display()
    );
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Validation: llama-perplexity chunk semantics (n_ctx=1024, first=512)
// ──────────────────────────────────────────────────────────────────────────

fn validate_ppl(m: &Qwen3, rope_for: &impl Fn(usize) -> RopeTable, tokens: &[u32], n_ctx: usize, n_threads: usize) -> f64 {
    let first = n_ctx / 2;
    let n_chunk = tokens.len() / n_ctx;
    let mut nll = 0f64;
    let mut count = 0usize;
    let vocab = m.head.len() / m.hidden;
    for ci in 0..n_chunk {
        let chunk = &tokens[ci * n_ctx..(ci + 1) * n_ctx];
        let rope = rope_for(n_ctx);
        let cap = CaptureSpec {
            layers: vec![],
            q_heads: vec![],
            k_ends: std::collections::HashSet::new(),
        };
        let (_, _, hidden) = prefill_capture(m, &rope, chunk, &cap, n_threads);
        // final norm + lm_head
        let mut xn = vec![0f32; n_ctx * m.hidden];
        for t in 0..n_ctx {
            rmsnorm_into(&hidden[t * m.hidden..(t + 1) * m.hidden], &m.out_norm, m.rms_eps, &mut xn[t * m.hidden..(t + 1) * m.hidden]);
        }
        let logits = gemm(&xn, n_ctx, &m.head, vocab, m.hidden, n_threads);
        for t in first..n_ctx - 1 {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let target = chunk[t + 1] as usize;
            let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let z: f32 = row.iter().map(|&v| (v - mx).exp()).sum();
            nll -= ((row[target] - mx).exp() / z).ln() as f64;
        }
        count += n_ctx - 1 - first;
        eprintln!("[validate] chunk {ci} done ({count} scored)");
    }
    let ppl = (nll / count as f64).exp();
    eprintln!("[validate] PPL = {ppl:.4} over {count} positions ({n_chunk} chunks × {n_ctx})");
    ppl
}

// ──────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────

fn main() {
    let prompt: &str = include_str!("../tests/data/asentmax_p07_prompt.txt");
    let mut model_path = PathBuf::from("/Users/katopz/git/riir-train/data/Ternary-Bonsai-8B-Q2_0.gguf");
    let mut out_path = PathBuf::from("tests/data/asentmax_p07_bonsai8b.fixture");
    let mut mode_validate = false;
    let mut mode_capture = false;
    let mut n_threads = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = PathBuf::from(args.next().expect("--model PATH")),
            "--out" => out_path = PathBuf::from(args.next().expect("--out PATH")),
            "--threads" => n_threads = args.next().expect("--threads N").parse().expect("int"),
            "--validate-ppl" => mode_validate = true,
            "--capture" => mode_capture = true,
            "--head-tokens" => {
                let n: usize = args.next().expect("N").parse().expect("int");
                let gg2 = read_gguf(&model_path).expect("gguf");
                let tok2 = Bpe::from_gguf(&gg2);
                let ids = tok2.encode(prompt);
                println!("{}", ids[..n.min(ids.len())].iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
                return;
            }
            "--dump-activations" => {
                // Debug: print per-layer residual norms for the first N tokens,
                // comparable against llama-dump-layer-activations.
                let n: usize = args.next().expect("N").parse().expect("int");
                let gg2 = read_gguf(&model_path).expect("gguf");
                let tok2 = Bpe::from_gguf(&gg2);
                let ids = tok2.encode(prompt);
                let ids = &ids[..n.min(ids.len())];
                let m2 = load_qwen3(&gg2).expect("model");
                let rope = rope_table(&m2, &gg2, ids.len());
                let cap = CaptureSpec { layers: vec![], q_heads: vec![], k_ends: Default::default() };
                eprintln!("tokens: {ids:?}");
                let (_s, _r, _h) = prefill_capture(&m2, &rope, ids, &cap, n_threads);
                return;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    if !mode_validate && !mode_capture {
        mode_validate = true;
        mode_capture = true;
    }
    let gg = read_gguf(&model_path).expect("gguf");
    let tok = Bpe::from_gguf(&gg);
    let tokens = tok.encode(prompt);
    eprintln!("[tokenizer] {} bytes → {} tokens", prompt.len(), tokens.len());

    let m = load_qwen3(&gg).expect("model");

    if mode_validate {
        let rope_for = |t: usize| rope_table(&m, &gg, t);
        let ppl = validate_ppl(&m, &rope_for, &tokens, 1024, n_threads);
        eprintln!("[validate] final PPL = {ppl:.4} (llama.cpp reference: 16.9778)");
    }
    if mode_capture {
        let cap = CaptureSpec {
            layers: vec![1, 5, 10, 14, 19, 23, 28, 33],
            q_heads: vec![0, 4, 9, 13, 17, 22, 26, 31],
            k_ends: (4..=tokens.len() / BLOCK).collect(),
        };
        let rope = rope_table(&m, &gg, tokens.len());
        let t0 = Instant::now();
        let (summaries, rows, _hidden) = prefill_capture(&m, &rope, &tokens, &cap, n_threads);
        eprintln!(
            "[capture] {} rows / {} streams in {:.1}s",
            rows.len(),
            summaries.len(),
            t0.elapsed().as_secs_f32()
        );
        // Needle block: the planted-fact sentence's 64-token block. Prefix
        // re-encode at the sentence's char offset (pretoken-safe cut: the
        // sentence starts after ". ") — token count of the prefix = the
        // needle's first token position.
        let needle_sent = "The secret access code for the observatory is 7F-39-KE";
        let co = prompt.find(needle_sent).expect("needle sentence present");
        let n_needle_tok = tok.encode(&prompt[..co]).len();
        let needle_block = n_needle_tok / BLOCK;
        eprintln!(
            "[capture] needle at char {co} → token {n_needle_tok} → block {needle_block}"
        );
        write_fixture(&out_path, m.head_dim, &summaries, &rows, prompt, tokens.len(), needle_block)
            .expect("write fixture");

        // Replay smoke: prove the fixture replays through EntmaxRouter.
        let router = EntmaxRouter::default_router();
        let s0 = summaries.first().expect("stream");
        let mut cache = EntmaxCache::with_capacity(s0.n_blocks, s0.sums.len());
        for b in 0..s0.n_blocks {
            cache.summaries.push(s0.sums[b * m.head_dim..(b + 1) * m.head_dim].to_vec());
        }
        let r0 = rows.iter().find(|r| r.layer == s0.layer && r.kv_head == s0.kv_head).expect("row");
        let mut scratch = VortexScratch::new(s0.n_blocks);
        let dec = router.forward_indexer(&r0.query, &cache, s0.n_blocks, s0.n_blocks, &mut scratch);
        eprintln!(
            "[smoke] layer {} kvh {} n={} → support {} blocks",
            s0.layer,
            s0.kv_head,
            s0.n_blocks,
            dec.blocks.len()
        );
    }
}
