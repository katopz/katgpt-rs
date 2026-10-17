//! Spike census — weight-derived massive-activation screening (Issue 919 T1).
//!
//! Research 605 (arXiv:2603.05498, "The Spike, the Sparse and the Sink",
//! ICML 2026): massive activations are injected by 1–2 early FFN "step-up"
//! blocks through a directional quadratic amplifier and cancelled by late
//! "step-down" blocks. Spike channels correspond to large `‖U_k‖_F` where
//! `U_k = Σ_i W_down(k,i)·W_gate(i)·W_up(i)ᵀ`; the dominant rank-1 term gives
//! the screening stat `s(k,i) = |W_down(k,i)| · ‖γ⊙W_gate(i)‖ · ‖γ⊙W_up(i)‖`,
//! with all spike channels sharing one trigger direction
//! `s⋆ ≈ γ⊙W_gate(i)/‖γ⊙W_gate(i)‖`.
//!
//! γ MUST be folded in: GGUF stores the RMSNorm scale separately (Gemma uses
//! the `(1+γ)` convention) — scoring raw gate/up rows reads the wrong
//! matrices. Ternary packs (GGML type 42, Q2_0_g128 — the Bonsai family) need
//! the scale-aware variant: entry magnitudes collapse to group scales, so the
//! |W_down| z-screen is skipped and ranking runs on the raw score (the
//! {−1,0,+1}×d dequant still carries the structure).
//!
//! Offline and deterministic: reads the GGUF weights alone, no activation
//! data (the delta vs AWQ-class calibration pipelines). Output: a
//! BLAKE3-committable JSON sidecar (blocks × channels × trigger direction) +
//! a Table-1-style block-locality verdict (1–2 early step-up + 1–2 late
//! step-down). The screening stat is an approximation (dominant-rank-1
//! argument; the paper measures the exact Frobenius norm) — T2's
//! calibration-forward validation gates any use.
//!
//! Reader provenance: a minimal GGUF v3 reader following the
//! `asentmax_p07_gen_fixture` example-local precedent (PoC-grade; hoisting a
//! shared reader into a lib is a T3-graduation decision, not taken here).
//! Extended with Q8_0 (type 8), Q4_K (12), and Q6_K (14 — per the
//! fixture-gate-proven riir-infer `src/quant/q6k.rs` port, copied not
//! re-derived) dequant for the Qwen3.8-class packs. Q2_0_g128 covers ids 42
//! AND 142 (the fork-tip relabel is byte-identical — the league
//! Ternary-Bonsai-27B ships as 142). f16 conversion is a local bit-twiddle
//! with a subnormal arm the precedent's own version got wrong by 2× (its
//! gates never exercise subnormals; this example's test pins 0x0001 →
//! 2^-24).
//!
//! Usage:
//! ```text
//! cargo run --release -p katgpt-attn --example spike_census -- \
//!     --model /path/model.gguf [--out census.json] [--top-k 4] \
//!     [--entry-z 6.0] [--gamma-plus-one on|off|auto] [--blocks 0,1,20-25]
//! cargo run -p katgpt-attn --example spike_census -- --self-test
//! ```
//!
//! `--self-test` plants synthetic spikes in tiny in-memory FFN weights and
//! asserts the census recovers the planted channel, its intermediate index,
//! and the trigger direction (plus the (1+γ) score-scaling invariant) —
//! exit 1 on any miss.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// f16 → f32 (exact bit twiddle; asentmax_p07_gen_fixture precedent)
// ──────────────────────────────────────────────────────────────────────────

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal f16 → normalized f32. Shift count s brings the
            // leading bit to bit 10; value = (1+m/1024)·2^(−14−s), so the
            // exponent field is 113−s = 112−e in this loop's counting.
            // ⚠ Deliberate divergence from the asentmax_p07_gen_fixture
            // precedent (113−e there): off by one — every f16 subnormal
            // came back 2× too large. Its fixture gates never exercise
            // subnormals; this example's test pins 0x0001 → 2^-24.
            let mut e: i32 = -1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            (sign << 31) | (((112 - e) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

// ──────────────────────────────────────────────────────────────────────────
// Minimal GGUF v3 reader (read-only; precedent copy + Q8_0/Q4_K, str_meta)
// ──────────────────────────────────────────────────────────────────────────

const TY_F32: u32 = 0;
const TY_F16: u32 = 1;
const TY_Q8_0: u32 = 8;
const TY_Q4_K: u32 = 12;
const TY_Q6_K: u32 = 14;
const TY_Q2_0_G128: u32 = 42; // fork-tip relabels the same payload as 142

struct GgufTensorInfo {
    ne: [u64; 3], // [ne0 = cols = input dim, ne1 = rows, ne2 = 1 for 2-D]
    ty: u32,
    offset: u64, // relative to the data section
}

struct Gguf {
    data: Vec<u8>,
    data_start: usize,
    tensors: HashMap<String, GgufTensorInfo>,
    u32_meta: HashMap<String, u32>,
    f32_meta: HashMap<String, f32>,
    str_meta: HashMap<String, String>,
}

impl Gguf {
    fn str_meta(&self, key: &str) -> Option<&str> {
        self.str_meta.get(key).map(String::as_str)
    }

    fn tensor_type(&self, name: &str) -> Option<u32> {
        self.tensors.get(name).map(|t| t.ty)
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl Cursor<'_> {
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

fn read_gguf(path: &Path) -> Result<Gguf, String> {
    let t0 = Instant::now();
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    eprintln!(
        "[gguf] read {} bytes in {:.1}s",
        data.len(),
        t0.elapsed().as_secs_f32()
    );
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
        u32_meta: HashMap::new(),
        f32_meta: HashMap::new(),
        str_meta: HashMap::new(),
    };

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
            4 => {
                gg.u32_meta.insert(key, c.u32());
            }
            5 => {
                c.u32();
            }
            6 => {
                gg.f32_meta.insert(key, c.f32());
            }
            7 => {
                c.bytes(1);
            }
            8 => {
                let v = c.string();
                gg.str_meta.insert(key, v);
            }
            10 => {
                c.u64();
            }
            11 => {
                c.bytes(8);
            }
            12 => {
                c.f64();
            }
            9 => {
                let et = c.u32();
                let n = c.u64() as usize;
                match et {
                    8 => {
                        for _ in 0..n {
                            c.string();
                        }
                    }
                    0 | 1 | 7 => {
                        c.bytes(n);
                    }
                    2 | 3 => {
                        c.bytes(n * 2);
                    }
                    4..=6 => {
                        c.bytes(n * 4);
                    }
                    10..=12 => {
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
        let mut ne = [1u64; 3];
        for d in ne.iter_mut().take(n_dims) {
            *d = c.u64();
        }
        // >2 dims are fine in the HEADER (the DFlash2 fork carries 3-dim
        // conv-state tensors the census never reads) — dequant is where the
        // 2-D limit is enforced, so the data section still parses.
        let ty = c.u32();
        let offset = c.u64();
        gg.tensors.insert(name, GgufTensorInfo { ne, ty, offset });
    }
    gg.data_start = (c.p + 31) & !31; // llama.cpp default alignment = 32
    gg.data = data;
    Ok(gg)
}

/// Q4_K scale packing (ggml get_scale_min_k4): 8 (sc, min) pairs of 6 bits
/// over 12 bytes. Returns (scale, min) for sub-block j.
fn scale_min_k4(j: usize, q: &[u8]) -> (u32, u32) {
    if j < 4 {
        (u32::from(q[j] & 63), u32::from(q[j + 4] & 63))
    } else {
        let d = u32::from(q[j + 4] & 0xF) | (u32::from(q[j - 4] & 0xC0) >> 2);
        let m = u32::from(q[j + 4] >> 4) | (u32::from(q[j - 3] & 0xC0) >> 2);
        (d, m)
    }
}

impl Gguf {
    /// Dequantize a tensor to row-major f32 `[rows][cols]` (GGUF ne0 = cols =
    /// fastest-varying = input dim). Supports f32 (0), f16 (1), Q8_0 (8),
    /// Q4_K (12), Q6_K (14), Q2_0_g128 (42 — and the fork-tip relabel 142:
    /// byte-identical payload per riir-infer gguf_loader.rs; the league
    /// Ternary-Bonsai-27B ships as 142).
    fn tensor_f32(&self, name: &str) -> Result<(usize, usize, Vec<f32>), String> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} missing"))?;
        if info.ne[2] != 1 {
            return Err(format!("tensor {name}: >2 dims unsupported"));
        }
        let cols = info.ne[0] as usize;
        let rows = info.ne[1] as usize;
        let base = self.data_start + info.offset as usize;
        match info.ty {
            TY_F32 => {
                let need = rows
                    .checked_mul(cols)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or_else(|| format!("tensor {name}: size overflow"))?;
                if base + need > self.data.len() {
                    return Err(format!("tensor {name}: truncated"));
                }
                let mut v = vec![0f32; rows * cols];
                for (i, slot) in v.iter_mut().enumerate() {
                    *slot = f32::from_le_bytes(
                        self.data[base + i * 4..base + i * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                }
                Ok((rows, cols, v))
            }
            TY_F16 => {
                let mut v = vec![0f32; rows * cols];
                for (i, slot) in v.iter_mut().enumerate() {
                    *slot = f16_to_f32(u16::from_le_bytes(
                        self.data[base + i * 2..base + i * 2 + 2]
                            .try_into()
                            .unwrap(),
                    ));
                }
                Ok((rows, cols, v))
            }
            TY_Q8_0 => {
                if !cols.is_multiple_of(32) {
                    return Err(format!("tensor {name}: cols {cols} not a multiple of 32"));
                }
                let bpr = cols / 32;
                let need = rows.checked_mul(bpr * 34).ok_or("size overflow")?;
                if base + need > self.data.len() {
                    return Err(format!("tensor {name}: truncated"));
                }
                let mut v = vec![0f32; rows * cols];
                for r in 0..rows {
                    for g in 0..bpr {
                        let off = base + (r * bpr + g) * 34;
                        let d =
                            f16_to_f32(u16::from_le_bytes([self.data[off], self.data[off + 1]]));
                        let dst = &mut v[r * cols + g * 32..r * cols + g * 32 + 32];
                        for (j, slot) in dst.iter_mut().enumerate() {
                            let q = self.data[off + 2 + j] as i8;
                            *slot = d * f32::from(q);
                        }
                    }
                }
                Ok((rows, cols, v))
            }
            TY_Q4_K => {
                if !cols.is_multiple_of(256) {
                    return Err(format!("tensor {name}: cols {cols} not a multiple of 256"));
                }
                let bpr = cols / 256;
                let need = rows.checked_mul(bpr * 144).ok_or("size overflow")?;
                if base + need > self.data.len() {
                    return Err(format!("tensor {name}: truncated"));
                }
                let mut v = vec![0f32; rows * cols];
                for r in 0..rows {
                    for g in 0..bpr {
                        let off = base + (r * bpr + g) * 144;
                        let d =
                            f16_to_f32(u16::from_le_bytes([self.data[off], self.data[off + 1]]));
                        let dmin = f16_to_f32(u16::from_le_bytes([
                            self.data[off + 2],
                            self.data[off + 3],
                        ]));
                        let scales = &self.data[off + 4..off + 16];
                        let qs = &self.data[off + 16..off + 144];
                        let dst = &mut v[r * cols + g * 256..r * cols + g * 256 + 256];
                        for j in 0..8 {
                            let (sc, mn) = scale_min_k4(j, scales);
                            let d1 = d * sc as f32;
                            let m1 = dmin * mn as f32;
                            let q = &qs[j * 16..j * 16 + 16];
                            let out = &mut dst[j * 32..j * 32 + 32];
                            for (l, slot_pair) in out.chunks_mut(2).enumerate() {
                                slot_pair[0] = d1 * f32::from(q[l] & 0xF) - m1;
                                slot_pair[1] = d1 * f32::from(q[l] >> 4) - m1;
                            }
                        }
                    }
                }
                Ok((rows, cols, v))
            }
            TY_Q6_K => {
                // 210 B per 256 weights: ql[128], qh[64], scales[16] (i8),
                // d (f16). Dequant per riir-infer src/quant/q6k.rs (the
                // fixture-gate-proven port of ggml dequantize_row_q6_K —
                // copied, not re-derived; katgpt-rs cannot depend on
                // riir-infer, the upstream direction is forbidden).
                if !cols.is_multiple_of(256) {
                    return Err(format!("tensor {name}: cols {cols} not a multiple of 256"));
                }
                let bpr = cols / 256;
                let need = rows.checked_mul(bpr * 210).ok_or("size overflow")?;
                if base + need > self.data.len() {
                    return Err(format!("tensor {name}: truncated"));
                }
                let mut v = vec![0f32; rows * cols];
                for r in 0..rows {
                    for g in 0..bpr {
                        let off = base + (r * bpr + g) * 210;
                        let d = f16_to_f32(u16::from_le_bytes([
                            self.data[off + 208],
                            self.data[off + 209],
                        ]));
                        let dst = &mut v[r * cols + g * 256..r * cols + g * 256 + 256];
                        let (mut ql_off, mut qh_off, mut sc_off) = (0usize, 0usize, 0usize);
                        for half in 0..2 {
                            let ql = &self.data[off + ql_off..off + ql_off + 64];
                            let qh = &self.data[off + 128 + qh_off..off + 128 + qh_off + 32];
                            let out = &mut dst[half * 128..half * 128 + 128];
                            for l in 0..32 {
                                let is = l / 16;
                                let sc = self.data[off + 192 + sc_off + is] as i8 as f32;
                                let sc2 = self.data[off + 192 + sc_off + is + 2] as i8 as f32;
                                let sc3 = self.data[off + 192 + sc_off + is + 4] as i8 as f32;
                                let sc4 = self.data[off + 192 + sc_off + is + 6] as i8 as f32;
                                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                                let q2 =
                                    ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                                let q4 =
                                    ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                                out[l] = d * sc * q1 as f32;
                                out[l + 32] = d * sc2 * q2 as f32;
                                out[l + 64] = d * sc3 * q3 as f32;
                                out[l + 96] = d * sc4 * q4 as f32;
                            }
                            ql_off += 64;
                            qh_off += 32;
                            sc_off += 8;
                        }
                    }
                }
                Ok((rows, cols, v))
            }
            TY_Q2_0_G128 | 142 => {
                if !cols.is_multiple_of(128) {
                    return Err(format!("tensor {name}: cols {cols} not a multiple of 128"));
                }
                let bpr = cols / 128;
                let need = rows.checked_mul(bpr * 34).ok_or("size overflow")?;
                if base + need > self.data.len() {
                    return Err(format!("tensor {name}: truncated"));
                }
                let mut v = vec![0f32; rows * cols];
                for r in 0..rows {
                    for g in 0..bpr {
                        let off = base + (r * bpr + g) * 34;
                        let d =
                            f16_to_f32(u16::from_le_bytes([self.data[off], self.data[off + 1]]));
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
// Census core (pure, testable — no GGUF, no I/O)
// ──────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CensusOpts {
    top_k: usize,
    entry_z: f64,
    gamma_plus_one: bool,
    scale_aware: bool,
}

struct SpikeChannel {
    k: usize,
    i: usize,
    s: f64,
    cos_gate_up: f64,
    z_entry: f64,
}

struct BlockCensus {
    idx: usize,
    z_screen_fired: bool,
    n_candidates: usize,
    channels: Vec<SpikeChannel>,
    /// Normalized folded gate row at the top channel's intermediate index
    /// (the paper's trigger direction s⋆). Empty when no channel found.
    s_star: Vec<f32>,
    max_s: f64,
    gamma_found: bool,
}

/// One FFN block's census inputs (bundled — the pure census fn takes this
/// plus the options and nothing else).
struct BlockTensors<'a> {
    idx: usize,
    wdown: &'a [f32], // [n_embd][n_ff] row-major
    wgate: &'a [f32], // [n_ff][d_model] row-major
    wup: &'a [f32],   // [n_ff][d_model] row-major
    gamma: &'a [f32], // [d_model] raw (the (1+γ) convention is applied in census_block)
    n_embd: usize,
    n_ff: usize,
    d_model: usize,
    gamma_found: bool,
}

/// Screening census for one FFN block (Research 605 recipe steps 1–3).
/// Deterministic: ties keep the lowest index.
fn census_block(t: &BlockTensors, opts: CensusOpts) -> Result<BlockCensus, String> {
    let (idx, wdown, wgate, wup, gamma) = (t.idx, t.wdown, t.wgate, t.wup, t.gamma);
    let (n_embd, n_ff, d_model) = (t.n_embd, t.n_ff, t.d_model);
    if wdown.len() != n_embd * n_ff || wgate.len() != n_ff * d_model || wup.len() != n_ff * d_model
    {
        return Err(format!("blk {idx}: tensor shape mismatch"));
    }
    if gamma.len() != d_model {
        return Err(format!(
            "blk {idx}: gamma len {} != d_model {d_model}",
            gamma.len()
        ));
    }

    // Fold the norm scale into gate/up rows on the fly: row norms and the
    // gate/up collinearity per intermediate index i.
    let mut ng = vec![0f64; n_ff];
    let mut nu = vec![0f64; n_ff];
    let mut cos = vec![0f64; n_ff];
    for i in 0..n_ff {
        let g = &wgate[i * d_model..(i + 1) * d_model];
        let u = &wup[i * d_model..(i + 1) * d_model];
        let (mut ag, mut au, mut gu) = (0f64, 0f64, 0f64);
        for j in 0..d_model {
            let cj = if opts.gamma_plus_one {
                1.0 + f64::from(gamma[j])
            } else {
                f64::from(gamma[j])
            };
            let gj = cj * f64::from(g[j]);
            let cu = cj * f64::from(u[j]);
            ag += gj * gj;
            au += cu * cu;
            gu += gj * cu;
        }
        ng[i] = ag.sqrt();
        nu[i] = au.sqrt();
        cos[i] = if ng[i] > 0.0 && nu[i] > 0.0 {
            gu / (ng[i] * nu[i])
        } else {
            0.0
        };
    }

    // Step 1: anomalous |W_down| entries (log-z screen). Skipped in
    // scale-aware mode (ternary magnitudes collapse to group scales) and
    // when the screen finds nothing (weak-spike models — rank everything).
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    let mut z_screen_fired = false;
    if !opts.scale_aware {
        let mut sum = 0f64;
        let mut n = 0usize;
        for &w in wdown {
            if w != 0.0 {
                sum += f64::from(w).abs().ln();
                n += 1;
            }
        }
        if n >= 256 {
            let mean = sum / n as f64;
            let mut var = 0f64;
            for &w in wdown {
                if w != 0.0 {
                    let d = f64::from(w).abs().ln() - mean;
                    var += d * d;
                }
            }
            let std = (var / n as f64).sqrt();
            if std > 0.0 {
                for (j, &w) in wdown.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    let z = (f64::from(w).abs().ln() - mean) / std;
                    if z > opts.entry_z {
                        candidates.push((j, z));
                    }
                }
                z_screen_fired = !candidates.is_empty();
            }
        }
    }

    // Step 2: per-output-channel best score s(k,·) — over the screened
    // candidates when the screen fired, over all entries otherwise.
    let mut best = vec![0f64; n_embd];
    let mut bi = vec![0usize; n_embd];
    let mut bz = vec![0f64; n_embd];
    if z_screen_fired {
        for &(j, z) in &candidates {
            let k = j / n_ff;
            let i = j % n_ff;
            let s = f64::from(wdown[j]).abs() * ng[i] * nu[i];
            if s > best[k] {
                best[k] = s;
                bi[k] = i;
                bz[k] = z;
            }
        }
    } else {
        for i in 0..n_ff {
            let w = ng[i] * nu[i];
            for k in 0..n_embd {
                let s = f64::from(wdown[k * n_ff + i]).abs() * w;
                if s > best[k] {
                    best[k] = s;
                    bi[k] = i;
                }
            }
        }
    }

    // Step 3: top-K spike channels (stable: lowest k wins ties).
    let mut order: Vec<usize> = (0..n_embd).collect();
    order.sort_by(|&a, &b| best[b].total_cmp(&best[a]));
    let channels: Vec<SpikeChannel> = order
        .into_iter()
        .filter(|&k| best[k] > 0.0)
        .take(opts.top_k)
        .map(|k| SpikeChannel {
            k,
            i: bi[k],
            s: best[k],
            cos_gate_up: cos[bi[k]],
            z_entry: if z_screen_fired { bz[k] } else { 0.0 },
        })
        .collect();

    let mut s_star = Vec::new();
    if let Some(top) = channels.first() {
        let i = top.i;
        let mut acc = 0f64;
        s_star.reserve(d_model);
        for j in 0..d_model {
            let cj = if opts.gamma_plus_one {
                1.0 + f64::from(gamma[j])
            } else {
                f64::from(gamma[j])
            };
            let v = (cj * f64::from(wgate[i * d_model + j])) as f32;
            acc += f64::from(v) * f64::from(v);
            s_star.push(v);
        }
        let nrm = acc.sqrt();
        if nrm > 0.0 {
            let inv = (1.0 / nrm) as f32;
            for v in &mut s_star {
                *v *= inv;
            }
        } else {
            s_star.clear();
        }
    }

    Ok(BlockCensus {
        idx,
        z_screen_fired,
        n_candidates: candidates.len(),
        max_s: channels.first().map_or(0.0, |c| c.s),
        channels,
        s_star,
        gamma_found: t.gamma_found,
    })
}

/// Cosine between two equal-length trigger-direction vectors (0.0 if either
/// is degenerate).
fn cos_dirs(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa > 0.0 && bb > 0.0 {
        ab / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    }
}

// ──────────────────────────────────────────────────────────────────────────
// CLI
// ──────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum GammaArg {
    Auto,
    On,
    Off,
}

struct Args {
    model: Option<PathBuf>,
    out: Option<PathBuf>,
    top_k: usize,
    entry_z: f64,
    gamma: GammaArg,
    blocks: Option<String>,
    self_test: bool,
}

impl Args {
    fn parse() -> Result<Args, String> {
        let mut a = Args {
            model: None,
            out: None,
            top_k: 4,
            entry_z: 6.0,
            gamma: GammaArg::Auto,
            blocks: None,
            self_test: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut val_for = |name: &str| -> Result<String, String> {
                it.next().ok_or_else(|| format!("{name}: missing value"))
            };
            match arg.as_str() {
                "--model" => a.model = Some(PathBuf::from(val_for("--model")?)),
                "--out" => a.out = Some(PathBuf::from(val_for("--out")?)),
                "--top-k" => {
                    a.top_k = val_for("--top-k")?
                        .parse()
                        .map_err(|e| format!("--top-k: {e}"))?
                }
                "--entry-z" => {
                    a.entry_z = val_for("--entry-z")?
                        .parse()
                        .map_err(|e| format!("--entry-z: {e}"))?
                }
                "--gamma-plus-one" => match val_for("--gamma-plus-one")?.as_str() {
                    "on" => a.gamma = GammaArg::On,
                    "off" => a.gamma = GammaArg::Off,
                    "auto" => a.gamma = GammaArg::Auto,
                    other => {
                        return Err(format!("--gamma-plus-one: unknown {other} (on|off|auto)"));
                    }
                },
                "--blocks" => a.blocks = Some(val_for("--blocks")?),
                "--self-test" => a.self_test = true,
                other => return Err(format!("unknown arg {other}")),
            }
        }
        if !a.self_test && a.model.is_none() {
            return Err("--model <path> required (or --self-test)".into());
        }
        Ok(a)
    }
}

/// Parse a block filter like "0,1,20-25" into a sorted, deduped list.
fn parse_blocks(spec: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.trim().parse().map_err(|e| format!("--blocks: {e}"))?;
            let hi: usize = hi.trim().parse().map_err(|e| format!("--blocks: {e}"))?;
            if hi < lo {
                return Err(format!("--blocks: reversed range {part}"));
            }
            out.extend(lo..=hi);
        } else {
            out.push(part.parse().map_err(|e| format!("--blocks: {e}"))?);
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────
// Run
// ──────────────────────────────────────────────────────────────────────────

fn run(args: &Args) -> Result<(), String> {
    let model = args.model.as_ref().expect("model checked by Args::parse");
    let gg = read_gguf(model)?;
    let arch = gg.str_meta("general.architecture").unwrap_or("unknown");
    let gamma_plus_one = match args.gamma {
        GammaArg::On => true,
        GammaArg::Off => false,
        GammaArg::Auto => arch.contains("gemma"),
    };
    eprintln!(
        "[census] arch={arch} gamma_plus_one={gamma_plus_one} top_k={} entry_z={}",
        args.top_k, args.entry_z
    );

    // Discover FFN blocks: every blk.N.ffn_down.weight with gate+up present.
    let mut blocks: Vec<usize> = gg
        .tensors
        .keys()
        .filter_map(|name| {
            name.strip_prefix("blk.")?
                .strip_suffix(".ffn_down.weight")?
                .parse()
                .ok()
        })
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    if blocks.is_empty() {
        return Err("no blk.N.ffn_down.weight tensors found".into());
    }
    if let Some(spec) = &args.blocks {
        let keep = parse_blocks(spec)?;
        blocks.retain(|b| keep.contains(b));
        if blocks.is_empty() {
            return Err(format!("--blocks {spec}: no matching blocks in the file"));
        }
    }

    let opts = CensusOpts {
        top_k: args.top_k,
        entry_z: args.entry_z,
        gamma_plus_one,
        scale_aware: false, // per block, from the W_down tensor type
    };

    let mut censuses: Vec<BlockCensus> = Vec::with_capacity(blocks.len());
    let mut skipped: Vec<(usize, String)> = Vec::new();
    for b in blocks {
        let down_name = format!("blk.{b}.ffn_down.weight");
        let gate_name = format!("blk.{b}.ffn_gate.weight");
        let up_name = format!("blk.{b}.ffn_up.weight");
        let scale_aware = matches!(gg.tensor_type(&down_name), Some(TY_Q2_0_G128) | Some(142));
        let (nr, nc, wdown) = gg.tensor_f32(&down_name)?;
        if !gg.tensors.contains_key(&gate_name) || !gg.tensors.contains_key(&up_name) {
            skipped.push((b, "missing ffn_gate/ffn_up".into()));
            continue;
        }
        let (gr, gc, wgate) = gg.tensor_f32(&gate_name)?;
        let (ur, uc, wup) = gg.tensor_f32(&up_name)?;
        if gr != nc || ur != nc || gc != uc {
            return Err(format!(
                "blk {b}: shape mismatch down {nr}x{nc} gate {gr}x{gc} up {ur}x{uc}"
            ));
        }
        let (n_embd, n_ff, d_model) = (nr, nc, gc);
        // γ = the RMSNorm scale on the FFN's INPUT, by arch:
        // ffn_norm (llama/qwen3/gemma — the pre-FFN norm), else
        // post_attention_norm (qwen3.5/GDN-family: attn_norm → attention →
        // post_attention_norm → FFN, no separate pre-FFN norm exists), else
        // γ=1 with the honest gamma_found=false (a constant γ preserves the
        // per-block argmax over k, it only unscales the reported score).
        let gamma_name = [
            "ffn_norm.gamma",
            "ffn_norm.weight",
            "post_attention_norm.weight",
        ]
        .iter()
        .find_map(|s| {
            gg.tensors
                .contains_key(&format!("blk.{b}.{s}"))
                .then(|| format!("blk.{b}.{s}"))
        });
        let (gamma, gamma_found) = match &gamma_name {
            Some(name) => {
                let (_, _, g) = gg.tensor_f32(name)?;
                (g, true)
            }
            None => (vec![1.0f32; d_model], false),
        };
        let opts_b = CensusOpts {
            scale_aware,
            ..opts
        };
        let t = BlockTensors {
            idx: b,
            wdown: &wdown,
            wgate: &wgate,
            wup: &wup,
            gamma: &gamma,
            n_embd,
            n_ff,
            d_model,
            gamma_found,
        };
        let c = census_block(&t, opts_b)?;
        censuses.push(c);
        if censuses.len().is_multiple_of(8) {
            eprintln!("[census] {} blocks done", censuses.len());
        }
    }
    if censuses.is_empty() {
        return Err("no blocks censused".into());
    }

    // Locality summary (report-only — T2's calibration forward is the gate).
    // Strict spike blocks clear 10× the median block max; the soft tier
    // (2×) names "emerging" blocks so an end-concentrated shape stays
    // visible even when no block clears the strict bar (the 27B league pack
    // reads soft ends, 2–3× the middle — the paper's step-up/step-down
    // shape under a weaker gain).
    let n_total = censuses.len();
    let mut maxima: Vec<f64> = censuses.iter().map(|c| c.max_s).collect();
    maxima.sort_by(|a, b| a.total_cmp(b));
    let median = maxima[n_total / 2];
    let (spike_blocks, emerging): (Vec<&BlockCensus>, Vec<&BlockCensus>) = if median <= 0.0 {
        (
            censuses.iter().filter(|c| c.max_s > 0.0).collect(),
            Vec::new(),
        )
    } else {
        (
            censuses
                .iter()
                .filter(|c| c.max_s >= median * 10.0)
                .collect(),
            censuses
                .iter()
                .filter(|c| c.max_s >= median * 2.0 && c.max_s < median * 10.0)
                .collect(),
        )
    };
    let classified: &[&BlockCensus] = if spike_blocks.is_empty() {
        &emerging
    } else {
        &spike_blocks
    };
    let (early_end, late_start) = (n_total / 3, n_total - n_total / 3);
    let mut early = Vec::new();
    let mut mid = Vec::new();
    let mut late = Vec::new();
    for c in classified {
        let idx = c.idx;
        let is_mid = n_total > 2 && idx >= early_end && idx < late_start;
        if is_mid {
            mid.push(idx);
        } else if n_total <= 2 || idx < early_end {
            early.push(idx);
        } else {
            late.push(idx);
        }
    }
    let locality = if args.blocks.is_some() {
        "filtered"
    } else if spike_blocks.is_empty() && emerging.is_empty() {
        "no_spike_blocks"
    } else if mid.is_empty() && early.len() <= 2 && late.len() <= 2 {
        "matches_table1_shape"
    } else if mid.is_empty() {
        "end_concentrated"
    } else {
        "scattered"
    };
    // The paper: ALL spike channels share one trigger direction — max
    // pairwise s⋆ cosine across the spike blocks.
    let mut trigger_cos = 0f64;
    for i in 0..classified.len() {
        for j in (i + 1)..classified.len() {
            trigger_cos = trigger_cos.max(cos_dirs(&classified[i].s_star, &classified[j].s_star));
        }
    }

    // Sidecar JSON (canonical: BTreeMap key order, shortest-roundtrip
    // floats; the BLAKE3 is over exactly these bytes).
    fn jnum(x: f64) -> serde_json::Value {
        serde_json::Number::from_f64(x).map_or(serde_json::Value::Null, serde_json::Value::Number)
    }
    let file_name = model.file_name().map_or_else(
        || "<unnamed>".to_string(),
        |s| s.to_string_lossy().into_owned(),
    );
    let mut root = serde_json::Map::new();
    root.insert("schema".into(), "spike_census_v1".into());
    root.insert("model_file".into(), file_name.clone().into());
    root.insert("arch".into(), arch.into());
    root.insert("gamma_plus_one".into(), gamma_plus_one.into());
    root.insert("top_k".into(), args.top_k.into());
    root.insert("entry_z".into(), jnum(args.entry_z));
    root.insert("n_blocks_censused".into(), n_total.into());
    root.insert("median_max_s".into(), jnum(median));
    root.insert(
        "skipped".into(),
        serde_json::to_value(&skipped).unwrap_or_default(),
    );
    let mut jblocks = Vec::with_capacity(censuses.len());
    for c in &censuses {
        let mut jb = serde_json::Map::new();
        jb.insert("idx".into(), c.idx.into());
        jb.insert("gamma_found".into(), c.gamma_found.into());
        jb.insert("z_screen_fired".into(), c.z_screen_fired.into());
        jb.insert("n_candidates".into(), c.n_candidates.into());
        jb.insert("max_s".into(), jnum(c.max_s));
        let jch: Vec<serde_json::Value> = c
            .channels
            .iter()
            .map(|ch| {
                serde_json::json!({
                    "k": ch.k, "i": ch.i, "s": jnum(ch.s),
                    "cos_gate_up": jnum(ch.cos_gate_up), "z_entry": jnum(ch.z_entry)
                })
            })
            .collect();
        jb.insert("channels".into(), jch.into());
        let star: Vec<serde_json::Value> = c.s_star.iter().map(|v| jnum(f64::from(*v))).collect();
        jb.insert("s_star".into(), star.into());
        jblocks.push(serde_json::Value::Object(jb));
    }
    root.insert("blocks".into(), jblocks.into());
    let mut summary = serde_json::Map::new();
    summary.insert(
        "spike_blocks".into(),
        serde_json::Value::Array(spike_blocks.iter().map(|c| c.idx.into()).collect()),
    );
    summary.insert(
        "emerging_blocks_2x".into(),
        serde_json::Value::Array(emerging.iter().map(|c| c.idx.into()).collect()),
    );
    summary.insert("early_step_up".into(), early.clone().into());
    summary.insert("mid".into(), mid.clone().into());
    summary.insert("late_step_down".into(), late.clone().into());
    summary.insert("locality".into(), locality.into());
    summary.insert("cross_block_trigger_cos_max".into(), jnum(trigger_cos));
    root.insert("summary".into(), serde_json::Value::Object(summary));

    let body = serde_json::Value::Object(root).to_string();
    let digest = blake3::hash(body.as_bytes()).to_hex().to_string();
    match &args.out {
        Some(out) => {
            std::fs::write(out, &body).map_err(|e| format!("write {}: {e}", out.display()))?;
            let side = PathBuf::from(format!("{}.blake3", out.display()));
            std::fs::write(&side, format!("{digest}  {}\n", out.display()))
                .map_err(|e| format!("write {}: {e}", side.display()))?;
            println!("[census] sidecar {} blake3 {digest}", out.display());
        }
        None => {
            println!("[census] blake3 {digest} (pass --out to write the sidecar)");
        }
    }

    println!("[census] model {file_name} arch {arch} blocks {n_total}");
    println!(
        "[census] locality {locality}: step-up {} step-down {} mid {} | trigger cos max {trigger_cos:.4} | median max_s {median:.3e}",
        early.len(),
        late.len(),
        mid.len()
    );
    if spike_blocks.is_empty() && !emerging.is_empty() {
        let idxs: Vec<String> = emerging.iter().map(|c| c.idx.to_string()).collect();
        println!("[census] emerging (2-10x median): {}", idxs.join(", "));
    }
    for c in classified {
        let Some(top) = c.channels.first() else {
            continue;
        };
        println!(
            "[census] blk {:>3}: top k={:<5} i={:<6} s={:.3e} cos(g,u)={:.3} z={:.1} gamma_found={}",
            c.idx, top.k, top.i, top.s, top.cos_gate_up, top.z_entry, c.gamma_found
        );
    }
    for (b, why) in &skipped {
        println!("[census] blk {b}: skipped ({why})");
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Self-test — synthetic planted spikes, known answers
// ──────────────────────────────────────────────────────────────────────────

/// Deterministic xorshift64 fill in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        f64::from((x >> 33) as u32) / f64::from(u32::MAX) * 2.0 - 1.0
    }
}

const ST_D_MODEL: usize = 16;
const ST_N_FF: usize = 32;
const ST_N_EMBD: usize = 12;
const ST_K: usize = 7;
const ST_I: usize = 5;

struct Fixture {
    wdown: Vec<f32>,
    wgate: Vec<f32>,
    wup: Vec<f32>,
    gamma: Vec<f32>,
    trigger: Vec<f32>,
}

impl Fixture {
    /// The census folds γ into the trigger direction — the expected s⋆ is
    /// the normalized γ⊙gate row, not the raw planted direction.
    fn folded_trigger(&self) -> Vec<f32> {
        let mut t: Vec<f32> = self
            .trigger
            .iter()
            .zip(self.gamma.iter())
            .map(|(v, g)| v * g)
            .collect();
        let n: f32 = t.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in &mut t {
            *v /= n;
        }
        t
    }
}

fn st_trigger() -> Vec<f32> {
    let mut t: Vec<f32> = (0..ST_D_MODEL)
        .map(|j| (((j * 37) % ST_D_MODEL) as f32 - 7.5) / 7.5)
        .collect();
    let n: f32 = t.iter().map(|v| v * v).sum::<f32>().sqrt();
    for v in &mut t {
        *v /= n;
    }
    t
}

fn st_fixture(ternary: bool) -> Fixture {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut wdown = vec![0f32; ST_N_EMBD * ST_N_FF];
    let mut wgate = vec![0f32; ST_N_FF * ST_D_MODEL];
    let mut wup = vec![0f32; ST_N_FF * ST_D_MODEL];
    if ternary {
        for (j, slot) in wdown.iter_mut().enumerate() {
            *slot = ((j * 13) % 3) as f32 - 1.0;
        }
        for slot in wgate.iter_mut() {
            *slot = rng.next_f64().round() as f32;
        }
        for slot in wup.iter_mut() {
            *slot = rng.next_f64().round() as f32;
        }
    } else {
        for slot in wdown.iter_mut() {
            *slot = (rng.next_f64() * 0.1) as f32;
        }
        for slot in wgate.iter_mut() {
            *slot = (rng.next_f64() * 0.1) as f32;
        }
        for slot in wup.iter_mut() {
            *slot = (rng.next_f64() * 0.1) as f32;
        }
    }
    let gamma: Vec<f32> = (0..ST_D_MODEL).map(|j| 0.5 + 0.01 * j as f32).collect();
    let trigger = st_trigger();
    for (j, v) in trigger.iter().enumerate() {
        wgate[ST_I * ST_D_MODEL + j] = 30.0 * v;
        wup[ST_I * ST_D_MODEL + j] = 25.0 * v;
    }
    wdown[ST_K * ST_N_FF + ST_I] = if ternary { 100.0 } else { 50.0 };
    Fixture {
        wdown,
        wgate,
        wup,
        gamma,
        trigger,
    }
}

fn self_test_impl() -> Result<(), String> {
    // 1. Continuous: the z-screen fires and the planted channel wins with
    //    the planted trigger direction recovered.
    let f = st_fixture(false);
    let opts = CensusOpts {
        top_k: 3,
        entry_z: 4.0,
        gamma_plus_one: false,
        scale_aware: false,
    };
    let mk = |f: &Fixture, opts: CensusOpts| {
        census_block(
            &BlockTensors {
                idx: 0,
                wdown: &f.wdown,
                wgate: &f.wgate,
                wup: &f.wup,
                gamma: &f.gamma,
                n_embd: ST_N_EMBD,
                n_ff: ST_N_FF,
                d_model: ST_D_MODEL,
                gamma_found: true,
            },
            opts,
        )
    };
    let c = mk(&f, opts)?;
    let top = c
        .channels
        .first()
        .ok_or("self-test: no channels recovered")?;
    if top.k != ST_K || top.i != ST_I {
        return Err(format!(
            "self-test: planted (k={ST_K}, i={ST_I}) but census found (k={}, i={})",
            top.k, top.i
        ));
    }
    if !c.z_screen_fired {
        return Err("self-test: z-screen should fire on the planted spike".into());
    }
    let cos = cos_dirs(&c.s_star, &f.folded_trigger());
    if cos < 0.9999 {
        return Err(format!(
            "self-test: folded-trigger cos {cos} < 0.9999 (s⋆ must be the γ-folded direction)"
        ));
    }
    if top.cos_gate_up < 0.999 {
        return Err(format!(
            "self-test: gate/up cos {} < 0.999 (planted collinear)",
            top.cos_gate_up
        ));
    }

    // 2. Scale-aware (ternary): the screen is skipped, ranking still finds
    //    the planted channel via the anomalous group scale.
    let f = st_fixture(true);
    let opts = CensusOpts {
        scale_aware: true,
        ..opts
    };
    let c = mk(&f, opts)?;
    if c.z_screen_fired {
        return Err("self-test: scale-aware mode must skip the z-screen".into());
    }
    let top = c
        .channels
        .first()
        .ok_or("self-test: scale-aware found no channels")?;
    if top.k != ST_K || top.i != ST_I {
        return Err(format!(
            "self-test(ternary): planted (k={ST_K}, i={ST_I}) but found (k={}, i={})",
            top.k, top.i
        ));
    }

    // 3. (1+γ) convention: folding scales each row norm by
    //    ‖(1+γ)⊙u‖/‖γ⊙u‖ = √(num/den); the score multiplies TWO norms
    //    (gate and up share γ and the planted direction), so the expected
    //    score ratio is exactly num/den — computed from the fixture, not
    //    hand-banded.
    let f = st_fixture(false);
    let plain = mk(&f, opts)?;
    let plus = mk(
        &f,
        CensusOpts {
            gamma_plus_one: true,
            ..opts
        },
    )?;
    let (mut num, mut den) = (0f64, 0f64);
    for (u, g) in f.trigger.iter().zip(f.gamma.iter()) {
        num += (f64::from(*u) * (1.0 + f64::from(*g))).powi(2);
        den += (f64::from(*u) * f64::from(*g)).powi(2);
    }
    let expected = num / den;
    let (sp, sg) = (
        plus.channels.first().map(|c| c.s).unwrap_or(0.0),
        plain.channels.first().map(|c| c.s).unwrap_or(0.0),
    );
    if sg <= 0.0 || (sp / sg - expected).abs() / expected > 1e-3 {
        return Err(format!(
            "self-test: (1+γ) score ratio {sp}/{sg} = {} not within 0.1% of expected {expected}",
            sp / sg
        ));
    }
    if plus.channels.first().map(|c| c.k) != plain.channels.first().map(|c| c.k) {
        return Err("self-test: (1+γ) folding changed the argmax channel".into());
    }

    Ok(())
}

fn main() {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let r = if args.self_test {
        self_test_impl()
            .map(|_| println!("[self-test] PASS (3 arms: continuous / scale-aware / (1+γ))"))
    } else {
        run(&args)
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_test_arms_pass() {
        self_test_impl().expect("self-test");
    }

    #[test]
    fn block_filter_parser() {
        let b = parse_blocks("0, 3, 20-22").unwrap();
        assert_eq!(b, vec![0, 3, 20, 21, 22]);
        assert!(parse_blocks("5-3").is_err());
    }

    #[test]
    fn f16_round_trip() {
        // Exact for the f16-representable values the dequant paths see.
        for &(bits, want) in &[
            (0x0000u16, 0.0f32),
            (0x8000, -0.0),
            (0x3C00, 1.0),
            (0xBC00, -1.0),
            (0x4000, 2.0),
            (0x7C00, f32::INFINITY),
            (0x3555, 0.33325195),
        ] {
            let got = f16_to_f32(bits);
            assert_eq!(got, want, "bits {bits:#06x}");
        }
        // Subnormals (the pin that caught the precedent's off-by-one):
        // 0x0001 = 2^-24, 0x0002 = 2^-23, largest 0x03FF = 1023·2^-24.
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x0002), 2.0f32.powi(-23));
        assert_eq!(f16_to_f32(0x03FF), (1023.0f32 * 2.0f32.powi(-24)) as f32);
    }

    #[test]
    fn q4k_scale_packing() {
        // ggml reference vector: q[0..12] = 0..11 — the first four (sc, min)
        // pairs read the low 6 bits; j=4 folds the high bits of q[0..2].
        let q: Vec<u8> = (0..12).collect();
        let (d, m) = scale_min_k4(0, &q);
        assert_eq!((d, m), (0, 4));
        let (d, m) = scale_min_k4(4, &q);
        // d = (q[8] & 0xF) | ((q[0] & 0xC0) >> 2) = 8 | 0 = 8
        // m = (q[8] >> 4) | ((q[1] & 0xC0) >> 2) = 0 | 0 = 0
        assert_eq!((d, m), (8, 0));
        let q2 = [0xFFu8; 12];
        let (d, m) = scale_min_k4(4, &q2);
        assert_eq!((d, m), (0x0F | 0x30, 0x0F | 0x30));
    }
}
