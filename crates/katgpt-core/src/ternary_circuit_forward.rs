//! `ternary_circuit` Arm A — the full-forward port (Issue 1225; Research 677).
//!
//! Reproduces the pinned source's `_forward` + `encode_byte_grid`
//! (brianbell-x/codehygiene @ [`crate::ternary_circuit::SOURCE_PIN`]) on the
//! house `TernaryGroupWeights` matvec substrate
//! (`ternary_group_matvec_scalar`, katgpt-types — the sequential-ascending
//! Issue 650 reference kernel), with weights loaded from the T1a `.cact`
//! archive.
//!
//! # Numerics contract (pre-registered in Issue 1225, honored here)
//!
//! - **G1-A(a) bit-exact upstream of the first float operation**:
//!   [`encode_byte_grid`] reproduces the source's grid encoder law-for-law —
//!   `str.expandtabs(2)` (a tab advances to the next even column; `\n` and
//!   `\r` reset the column; every other codepoint, the splitlines boundary
//!   codepoints included, advances it by one), `str.splitlines`' boundary
//!   set, the 2048-line cap, the 128-column clip, the `min(cp, 127)`
//!   non-ASCII clamp and the NUL→space remap. The G1-A gate asserts the grid
//!   bytes EXACTLY against the committed fixture hex.
//! - **G1-A(b) post-`exp` band**: every probability surface is compared at
//!   |Δp| ≤ 1e-4 ABSOLUTE (relative error is ill-conditioned near zero).
//!   The port is band-exact, not bit-exact, past the grid stage. The two
//!   disclosed rounding divergences: (i) scale association — the source
//!   multiplies each matrix by its per-channel scale at load, this port
//!   folds the scale into the activation at use (`w·(x[c]·s[c])` vs the
//!   source's `(w·s[c])·x[c]` — the same two-factor product, one
//!   association apart, ≤1 ulp per term); (ii) `exp` — `f32::exp` vs numpy's
//!   SIMD `expf`, ≤1 ulp per call.
//! - **G1-A(c) fold order pinned sequential-ascending** (Issue 940): every
//!   reduction routes through `ternary_group_matvec_scalar` (groups
//!   ascending, columns ascending within the group) or an explicit ascending
//!   accumulation — no SIMD lane reassociation anywhere in this module.
//! - **softmax sites**: the two attention softmaxes are LOAD-BEARING (they
//!   feed `state`, hence the sigmoid heads) and are covered by the same
//!   absolute band as part of the primary path; the ordinal + style
//!   softmaxes are reproduction-only side outputs (banded, never exactness
//!   evidence) — exactly as pre-registered in the issue.
//!
//! # Orientation law
//!
//! The archive stores every projection in the source's own `(in, out)`
//! orientation (the source computes `x @ W`). The matvec contract is
//! `y = W' @ x` with `W' = Wᵀ`, so [`CodeHygieneModel::load`] transposes
//! every projection ONCE into `TernaryGroupWeights` (unit group scales —
//! the T1a trit layers carry row_scale ≡ 1.0, and f16(1.0) is exact). The
//! two `W_causal{1,2}` matrices are consumed in their ORIGINAL orientation
//! (their use is already `y = W @ x`). The conv kernels transpose to
//! `(Cout, k·Cin)` so each output channel folds over the flattened
//! `(offset, channel)` axis through the same matvec.
//!
//! G2-A (the ≤10 ms warm bar) and G4-A (alloc-free canary) are SEPARATE
//! pre-registered gates — this module makes them reachable, it does not
//! claim them.

// Every indexed loop in this module IS the pinned sequential-ascending fold
// order (G1-A(c), Issue 940): the index arithmetic names the storage law
// (bitplane block + column, the (line, col, channel) strides). Iterator
// adapters would obscure the order the band contract is audited against.
#![allow(clippy::needless_range_loop)]
use crate::kpt_archive::KptArchive;
use crate::ternary_circuit::Meta;
use katgpt_types::GROUP_SIZE;
use katgpt_types::TernaryGroupWeights;
use katgpt_types::TernaryWeights;
use katgpt_types::simd::ternary_group_matvec_scalar;

/// The source's grid width (`MAX_COLS`, `__init__.py:24` @ the pin).
pub const MAX_COLS: usize = 128;
/// The source's line cap (`encode_byte_grid`'s `max_lines` default).
pub const MAX_LINES: usize = 2048;
/// The source's relative-offset half-window (`MAX_REL_DIST`, `__init__.py:25`).
pub const MAX_REL_DIST: usize = 32;
/// The sketch-half split (`_SKETCH_DIM`, `__init__.py:26`).
pub const SKETCH_DIM: usize = 40;
/// Attention heads (`b_sink`'s pinned length; both layers ship 8).
pub const NUM_HEADS: usize = 8;
/// The dims the G1-A fixtures were pinned under (meta law at load).
pub const EMB_DIM: usize = 46;
pub const LEX_DIM: usize = 16;
pub const PREFIX_DIM: usize = 4;
pub const GATE_DIM: usize = 50;
pub const CELL_DIM: usize = 52;
pub const COL_SUM_DIM: usize = 16;
pub const CONV3_DIM: usize = 64;
pub const CONV7_DIM: usize = 128;
pub const CONV13_DIM: usize = 86;
pub const SKETCH_TOTAL: usize = 2 * SKETCH_DIM;
pub const LMLP_HIDDEN: usize = 124;
pub const LINE_STATS_DIM: usize = 146;
pub const STATE_DIM: usize = 226;
pub const QK_HEAD_DIM: usize = 16;
pub const QK_DIM: usize = NUM_HEADS * QK_HEAD_DIM; // 128
pub const NUM_SCORES: usize = 6;
pub const NUM_KINDS: usize = 8;
pub const NUM_ARCHETYPES: usize = 5;
pub const REL_TABLE: usize = 2 * MAX_REL_DIST + 1; // 65
/// The ordinal head's score levels (`_LEVELS`, `__init__.py:64`).
pub const LEVELS: [f32; 4] = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
/// The doc-pool epsilon (`_forward:236` — `1e-5`).
pub const POOL_EPS: f32 = 1e-5;
/// `_l2_norm`'s eps (`__init__.py:113` — `2.5`).
pub const L2_EPS: f32 = 2.5;
/// Logits layout: 2 noul + 6·4 ordinal + 5 style = 31.
pub const NOUL_HEADS: usize = 2;
pub const SCORE_LOGIT_END: usize = NOUL_HEADS + 4 * NUM_SCORES; // 26
pub const STYLE_LOGIT_END: usize = SCORE_LOGIT_END + NUM_ARCHETYPES; // 31

/// Everything that can go wrong resolving or running the pinned forward.
#[derive(Debug)]
pub enum ForwardError {
    /// A named param the forward requires is absent from the archive.
    MissingParam { name: String },
    /// A param's meta dims contradict the shape law this port is pinned under.
    ShapeLaw {
        name: String,
        got: Vec<usize>,
        want: Vec<usize>,
    },
    /// The archive layer at a meta position has the wrong flat shape.
    ArchiveShape {
        name: String,
        got: (usize, usize),
        want: (usize, usize),
    },
    /// The grid passed to [`CodeHygieneModel::forward`] exceeds the line cap.
    GridShape { lines: usize },
}

impl core::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingParam { name } => write!(f, "missing param {name:?}"),
            Self::ShapeLaw { name, got, want } => {
                write!(f, "param {name:?} meta dims {got:?} != pinned law {want:?}")
            }
            Self::ArchiveShape { name, got, want } => {
                write!(f, "param {name:?} archive shape {got:?} != meta {want:?}")
            }
            Self::GridShape { lines } => {
                write!(f, "grid lines {lines} exceed the {MAX_LINES} cap")
            }
        }
    }
}

impl std::error::Error for ForwardError {}

fn law(name: &str, got: &[usize], want: &[usize]) -> Result<(), ForwardError> {
    if got == want {
        Ok(())
    } else {
        Err(ForwardError::ShapeLaw {
            name: name.to_string(),
            got: got.to_vec(),
            want: want.to_vec(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The grid encoder — the G1-A(a) bit-exact surface.
// ─────────────────────────────────────────────────────────────────────────

/// `str.expandtabs(2)` — exact CPython semantics:
/// `\t` advances the column to the next multiple of 2 (at least one),
/// `\n` and `\r` reset the column to zero, EVERY other codepoint (the
/// splitlines boundary codepoints included) advances it by one.
fn expandtabs_2(code: &str) -> String {
    let mut out = String::with_capacity(code.len());
    let mut col = 0usize;
    for ch in code.chars() {
        match ch {
            '\t' => {
                let advance = 2 - (col % 2);
                for _ in 0..advance {
                    out.push(' ');
                }
                col += advance;
            }
            '\n' | '\r' => {
                out.push(ch);
                col = 0;
            }
            _ => {
                out.push(ch);
                col += 1;
            }
        }
    }
    out
}

/// Is `ch` a `str.splitlines` boundary? The full CPython set:
/// `\n \r \r\n \v(\x0b) \f(\x0c) \x1c \x1d \x1e \x85 \u2028 \u2029`.
/// (`\r\n` is ONE boundary — handled by the caller peeking `\r`.)
#[inline]
fn is_splitline_boundary(ch: char) -> bool {
    matches!(
        ch,
        '\n' | '\r'
            | '\u{000b}'
            | '\u{000c}'
            | '\u{001c}'
            | '\u{001d}'
            | '\u{001e}'
            | '\u{0085}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `str.splitlines()` over chars — boundaries consumed, `\r\n` one boundary,
/// no trailing empty segment after a final boundary.
fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0usize;
    let mut chars = text.char_indices().peekable();
    while let Some((i, ch)) = chars.next() {
        if is_splitline_boundary(ch) {
            lines.push(&text[start..i]);
            if ch == '\r' && chars.peek().map(|&(_, c)| c) == Some('\n') {
                chars.next();
            }
            start = i + ch.len_utf8();
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines.shrink_to_fit();
    lines
}

/// The source's `encode_byte_grid` (`__init__.py:169` @ the pin): a
/// `(lines ≤ 2048, 128)` u8 grid of clamped ASCII codes.
///
/// Laws (each asserted by the G1-A gate against the committed fixture hex):
/// tabs expand to two columns before splitting; the splitlines boundary set
/// includes `\x0b \x0c \x1c-\x1e \x85 \u2028 \u2029` (a naive `\n` splitter
/// diverges LOUD at the line count); an empty input yields one zero row;
/// each line clips at 128 codepoints; every codepoint maps to
/// `min(cp, 127)` with NUL→32; rows zero-pad to 128.
pub fn encode_byte_grid(code: &str) -> Vec<[u8; MAX_COLS]> {
    let expanded = expandtabs_2(code);
    let mut lines = splitlines(&expanded);
    if lines.is_empty() {
        lines.push("");
    }
    lines.truncate(MAX_LINES);
    let mut grid = Vec::with_capacity(lines.len());
    for line in &lines {
        let mut row = [0u8; MAX_COLS];
        for (col, ch) in line.chars().take(MAX_COLS).enumerate() {
            let cp = ch as u32;
            row[col] = if cp == 0 { 32 } else { cp.min(127) as u8 };
        }
        grid.push(row);
    }
    grid.shrink_to_fit();
    grid
}

// ─────────────────────────────────────────────────────────────────────────
// Param resolution — names → meta positions → transposed group weights.
// ─────────────────────────────────────────────────────────────────────────

/// The positional index of every named param, resolved from the meta the
/// archive was assembled under (trit-part order, then vec-part order — the
/// T1a layer-order law).
pub struct ParamIndex {
    trit: std::collections::BTreeMap<String, usize>,
    vec: std::collections::BTreeMap<String, usize>,
}

impl ParamIndex {
    pub fn resolve(meta: &Meta) -> Self {
        let mut trit = std::collections::BTreeMap::new();
        for (i, e) in meta.trits.iter().enumerate() {
            trit.insert(e.name.clone(), i);
        }
        let mut vec = std::collections::BTreeMap::new();
        for (i, e) in meta.vecs.iter().enumerate() {
            vec.insert(e.name.clone(), i);
        }
        Self { trit, vec }
    }

    fn trit(&self, name: &str) -> Result<usize, ForwardError> {
        self.trit
            .get(name)
            .copied()
            .ok_or_else(|| ForwardError::MissingParam {
                name: name.to_string(),
            })
    }

    fn vec(&self, name: &str) -> Result<usize, ForwardError> {
        self.vec
            .get(name)
            .copied()
            .ok_or_else(|| ForwardError::MissingParam {
                name: name.to_string(),
            })
    }

    fn vec_opt(&self, name: &str) -> Option<usize> {
        self.vec.get(name).copied()
    }
}

/// Fetch an archive layer as an owned [`TernaryWeights`], asserting its flat
/// shape equals the meta dims flattened (rank-2/3 → `(rows, in·out…)`).
fn trit_layer(
    archive: &KptArchive,
    idx: &ParamIndex,
    meta: &Meta,
    name: &str,
    want: &[usize],
) -> Result<TernaryWeights, ForwardError> {
    let i = idx.trit(name)?;
    let entry = &meta.trits[i];
    law(name, &entry.dims, want)?;
    let tw = archive.to_ternary_weights(i);
    let want_rows = want[0];
    let want_cols: usize = want[1..].iter().product();
    if (tw.rows, tw.cols) != (want_rows, want_cols) {
        return Err(ForwardError::ArchiveShape {
            name: name.to_string(),
            got: (tw.rows, tw.cols),
            want: (want_rows, want_cols),
        });
    }
    Ok(tw)
}

/// Fetch a vec layer's f32 payload (scale, bias, or table), asserting dims.
fn vec_layer(
    archive: &KptArchive,
    idx: &ParamIndex,
    meta: &Meta,
    name: &str,
    want: &[usize],
) -> Result<Vec<f32>, ForwardError> {
    let vi = idx.vec(name)?;
    let entry = &meta.vecs[vi];
    // The archive is positional (T1a layer-order law): every vec layer sits
    // AFTER every trit layer, at trit_count + vec_index.
    let i = meta.trits.len() + vi;
    let tw = archive.to_ternary_weights(i);
    // Dims compare FLATTENED: a vec entry may ship multi-dim (the attention
    // rel tables are (8, 65)); the caller states the flat length it needs.
    let want_len: usize = want.iter().product();
    let got_len: usize = entry.dims.iter().product();
    if got_len != want_len {
        return Err(ForwardError::ShapeLaw {
            name: name.to_string(),
            got: entry.dims.clone(),
            want: vec![want_len],
        });
    }
    if (tw.rows, tw.cols) != (got_len, 0) {
        return Err(ForwardError::ArchiveShape {
            name: name.to_string(),
            got: (tw.rows, tw.cols),
            want: (got_len, 0),
        });
    }
    Ok(tw.row_scale.clone())
}

/// Transpose `(in, out)` trits into the matvec contract `(out, in)` group
/// weights (unit group scales — the archive trit layers carry row_scale
/// ≡ 1.0, so the f16 group broadcast is exact).
fn transpose_to_group(tw: &TernaryWeights, in_dim: usize, out_dim: usize) -> TernaryGroupWeights {
    debug_assert_eq!((tw.rows, tw.cols), (in_dim, out_dim));
    let mut g = TernaryGroupWeights::new(out_dim, in_dim);
    for i in 0..in_dim {
        for o in 0..out_dim {
            let t = tw.get(i, o);
            if t != 0 {
                g.set(o, i, t);
            }
        }
    }
    g
}

/// Transpose a conv kernel `(k, Cin, Cout)` (stored `(k, Cin·Cout)`) into the
/// matvec contract `(Cout, k·Cin)`.
fn transpose_conv_to_group(
    tw: &TernaryWeights,
    k: usize,
    cin: usize,
    cout: usize,
) -> TernaryGroupWeights {
    debug_assert_eq!((tw.rows, tw.cols), (k, cin * cout));
    let mut g = TernaryGroupWeights::new(cout, k * cin);
    for o in 0..k {
        for i in 0..cin {
            for out_c in 0..cout {
                let t = tw.get(o, i * cout + out_c);
                if t != 0 {
                    g.set(out_c, o * cin + i, t);
                }
            }
        }
    }
    g
}

/// Fetch a projection `(in, out)` and return its transposed group weights.
fn proj_layer(
    archive: &KptArchive,
    idx: &ParamIndex,
    meta: &Meta,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<TernaryGroupWeights, ForwardError> {
    let tw = trit_layer(archive, idx, meta, name, &[in_dim, out_dim])?;
    Ok(transpose_to_group(&tw, in_dim, out_dim))
}

/// Fetch a conv kernel `(k, Cin, Cout)` transposed to `(Cout, k·Cin)`.
fn conv_layer(
    archive: &KptArchive,
    idx: &ParamIndex,
    meta: &Meta,
    name: &str,
    k: usize,
    cin: usize,
    cout: usize,
) -> Result<TernaryGroupWeights, ForwardError> {
    let tw = trit_layer(archive, idx, meta, name, &[k, cin, cout])?;
    Ok(transpose_conv_to_group(&tw, k, cin, cout))
}

// ─────────────────────────────────────────────────────────────────────────
// The resolved model.
// ─────────────────────────────────────────────────────────────────────────

/// One sink-attention layer's resolved weights. The two layers share the
/// q/k geometry (226→128, 8 heads × 16) but differ in v/wo (layer 1 folds
/// 96 = 8×12, layer 2 folds 128 = 8×16) — the dims come from the meta and
/// are asserted at load.
struct AttnLayer {
    wq: TernaryGroupWeights,
    wq_s: Vec<f32>,
    wk: TernaryGroupWeights,
    wk_s: Vec<f32>,
    wv: TernaryGroupWeights,
    wv_s: Vec<f32>,
    wo: TernaryGroupWeights,
    /// Flat `(NUM_HEADS, REL_TABLE)` relative-offset table.
    b_rel: Vec<f32>,
    b_sink: Vec<f32>,
    /// `wv.cols / NUM_HEADS` (12 on layer 1, 16 on layer 2).
    head_dim_v: usize,
}

/// One context-MLP layer's resolved weights (the `cmlp` pair beside each
/// attention layer: `w1 (STATE→HIDDEN)`, `w2 (HIDDEN→STATE)` + scales/biases).
struct CtxMlp {
    w1: TernaryGroupWeights,
    w1_s: Vec<f32>,
    b1: Vec<f32>,
    w2: TernaryGroupWeights,
    w2_s: Vec<f32>,
    b2: Vec<f32>,
    hidden: usize,
}

/// The resolved pinned forward: every trit matrix transposed into the matvec
/// contract at load, every f32 vector verbatim, every shape asserted against
/// the laws the G1-A fixtures were pinned under. Channel scales stay
/// SEPARATE from the bitplanes (folded into activations at use — the
/// disclosed association divergence, G1-A(b); an empty Vec means the source
/// ran that matrix unscaled).
pub struct CodeHygieneModel {
    tok_emb: TernaryWeights,
    col_emb: TernaryWeights,
    pre_conv3: TernaryGroupWeights,
    pre_conv3_s: Vec<f32>,
    pre_conv3_b: Vec<f32>,
    pref1_proj: TernaryGroupWeights,
    w_causal1_g: TernaryGroupWeights,
    gate1: TernaryGroupWeights,
    gate1_s: Vec<f32>,
    gate1_b: Vec<f32>,
    pref2_proj: TernaryGroupWeights,
    w_causal2_g: TernaryGroupWeights,
    col_fuse: TernaryGroupWeights,
    col_fuse_s: Vec<f32>,
    col_fuse_b: Vec<f32>,
    col_sum_proj: TernaryGroupWeights,
    col_sum_s: Vec<f32>,
    conv3: TernaryGroupWeights,
    conv3_s: Vec<f32>,
    conv3_b: Vec<f32>,
    conv7: TernaryGroupWeights,
    conv7_s: Vec<f32>,
    conv7_b: Vec<f32>,
    conv13: TernaryGroupWeights,
    conv13_s: Vec<f32>,
    conv13_b: Vec<f32>,
    sketch: TernaryGroupWeights,
    sketch_s: Vec<f32>,
    sketch_b: Vec<f32>,
    lmlp1: TernaryGroupWeights,
    lmlp1_s: Vec<f32>,
    lmlp1_b: Vec<f32>,
    lmlp2: TernaryGroupWeights,
    lmlp2_b: Vec<f32>,
    attn1: AttnLayer,
    ctx1: CtxMlp,
    attn2: AttnLayer,
    ctx2: CtxMlp,
    sq: TernaryGroupWeights,
    sq_s: Vec<f32>,
    sq_b: Vec<f32>,
    pool_gate: TernaryGroupWeights,
    pool_gate_s: Vec<f32>,
    pool_gate_b: Vec<f32>,
    doc1: TernaryGroupWeights,
    doc1_s: Vec<f32>,
    doc1_b: Vec<f32>,
    doc2: TernaryGroupWeights,
    doc2_b: Vec<f32>,
}

impl CodeHygieneModel {
    /// Resolve every param from the verified archive by its meta position,
    /// asserting the pinned shape laws (fail-closed on any drift).
    pub fn load(archive: &KptArchive, meta: &Meta) -> Result<Self, ForwardError> {
        let idx = ParamIndex::resolve(meta);
        // Per-channel scales: the absence of a `{stem}_s` vec entry means the
        // source ran that matrix UNSCALED — resolved as an empty Vec here
        // (the fold helper treats len 0 as no-fold), never as zeros.
        let scale = |name: &str| -> Result<Vec<f32>, ForwardError> {
            match idx.vec_opt(name) {
                Some(vi) => {
                    let entry = &meta.vecs[vi];
                    law(name, &entry.dims, &[entry.dims[entry.dims.len() - 1]])?;
                    let tw = archive.to_ternary_weights(meta.trits.len() + vi);
                    Ok(tw.row_scale)
                }
                None => Ok(Vec::new()),
            }
        };
        let bias = |name: &str, dim: usize| -> Result<Vec<f32>, ForwardError> {
            vec_layer(archive, &idx, meta, name, &[dim])
        };
        let proj = |name: &str,
                    in_dim: usize,
                    out_dim: usize|
         -> Result<TernaryGroupWeights, ForwardError> {
            proj_layer(archive, &idx, meta, name, in_dim, out_dim)
        };
        let conv = |name: &str,
                    k: usize,
                    cin: usize,
                    cout: usize|
         -> Result<TernaryGroupWeights, ForwardError> {
            conv_layer(archive, &idx, meta, name, k, cin, cout)
        };
        let attn = |root: &str, v_dim: usize| -> Result<AttnLayer, ForwardError> {
            Ok(AttnLayer {
                wq: proj(&format!("{root}_wq"), STATE_DIM, QK_DIM)?,
                wq_s: scale(&format!("{root}_wq_s"))?,
                wk: proj(&format!("{root}_wk"), STATE_DIM, QK_DIM)?,
                wk_s: scale(&format!("{root}_wk_s"))?,
                wv: proj(&format!("{root}_wv"), STATE_DIM, v_dim)?,
                wv_s: scale(&format!("{root}_wv_s"))?,
                wo: proj(&format!("{root}_wo"), v_dim, STATE_DIM)?,
                b_rel: bias(&format!("{root}_brel"), NUM_HEADS * REL_TABLE)?,
                b_sink: bias(&format!("{root}_bsink"), NUM_HEADS)?,
                head_dim_v: v_dim / NUM_HEADS,
            })
        };

        Ok(Self {
            tok_emb: trit_layer(archive, &idx, meta, "tok_emb", &[128, EMB_DIM])?,
            col_emb: trit_layer(archive, &idx, meta, "col_emb", &[MAX_COLS, EMB_DIM])?,
            pre_conv3: conv("pre_conv3_w", 3, EMB_DIM, LEX_DIM)?,
            pre_conv3_s: scale("pre_conv3_s")?,
            pre_conv3_b: bias("pre_conv3_b", LEX_DIM)?,
            pref1_proj: proj("pref1_proj", EMB_DIM + LEX_DIM, PREFIX_DIM)?,
            w_causal1_g: TernaryGroupWeights::from_ternary(&trit_layer(
                archive,
                &idx,
                meta,
                "W_causal1",
                &[MAX_COLS, MAX_COLS],
            )?),
            gate1: proj("gate1_w", EMB_DIM + LEX_DIM + PREFIX_DIM, GATE_DIM)?,
            gate1_s: scale("gate1_s")?,
            gate1_b: bias("gate1_b", GATE_DIM)?,
            pref2_proj: proj("pref2_proj", GATE_DIM, PREFIX_DIM)?,
            w_causal2_g: TernaryGroupWeights::from_ternary(&trit_layer(
                archive,
                &idx,
                meta,
                "W_causal2",
                &[MAX_COLS, MAX_COLS],
            )?),
            col_fuse: proj(
                "col_fuse_w",
                EMB_DIM + LEX_DIM + GATE_DIM + PREFIX_DIM,
                CELL_DIM,
            )?,
            col_fuse_s: scale("col_fuse_s")?,
            col_fuse_b: bias("col_fuse_b", CELL_DIM)?,
            col_sum_proj: proj("col_sum_proj", CELL_DIM, COL_SUM_DIM)?,
            col_sum_s: scale("col_sum_s")?,
            conv3: conv("conv3_w", 3, CELL_DIM, CONV3_DIM)?,
            conv3_s: scale("conv3_s")?,
            conv3_b: bias("conv3_b", CONV3_DIM)?,
            conv7: conv("conv7_w", 7, CELL_DIM, CONV7_DIM)?,
            conv7_s: scale("conv7_s")?,
            conv7_b: bias("conv7_b", CONV7_DIM)?,
            conv13: conv("conv13_w", 13, CELL_DIM, CONV13_DIM)?,
            conv13_s: scale("conv13_s")?,
            conv13_b: bias("conv13_b", CONV13_DIM)?,
            sketch: conv("sketch_w", 5, CELL_DIM, SKETCH_TOTAL)?,
            sketch_s: scale("sketch_s")?,
            sketch_b: bias("sketch_b", SKETCH_TOTAL)?,
            lmlp1: proj(
                "lmlp1_w",
                COL_SUM_DIM + CONV3_DIM + CONV7_DIM + CONV13_DIM,
                LMLP_HIDDEN,
            )?,
            lmlp1_s: scale("lmlp1_s")?,
            lmlp1_b: bias("lmlp1_b", LMLP_HIDDEN)?,
            lmlp2: proj("lmlp2_w", LMLP_HIDDEN, LINE_STATS_DIM)?,
            lmlp2_b: bias("lmlp2_b", LINE_STATS_DIM)?,
            attn1: attn("attn1", 96)?,
            ctx1: CtxMlp {
                w1: proj("cmlp1_w", STATE_DIM, 96)?,
                w1_s: scale("cmlp1_s")?,
                b1: bias("cmlp1_b", 96)?,
                w2: proj("cmlp1_2_w", 96, STATE_DIM)?,
                w2_s: Vec::new(), // cmlp1_2_s absent — unscaled in the source
                b2: bias("cmlp1_2_b", STATE_DIM)?,
                hidden: 96,
            },
            attn2: attn("attn2", QK_DIM)?,
            ctx2: CtxMlp {
                w1: proj("cmlp2_w", STATE_DIM, 64)?,
                w1_s: scale("cmlp2_s")?,
                b1: bias("cmlp2_b", 64)?,
                w2: proj("cmlp2_2_w", 64, STATE_DIM)?,
                w2_s: Vec::new(), // cmlp2_2_s absent — unscaled in the source
                b2: bias("cmlp2_2_b", STATE_DIM)?,
                hidden: 64,
            },
            sq: proj("sq_w", STATE_DIM, NUM_KINDS)?,
            sq_s: scale("sq_s")?,
            sq_b: bias("sq_b", NUM_KINDS)?,
            pool_gate: proj("pool_gate_w", STATE_DIM, 1)?,
            pool_gate_s: scale("pool_gate_s")?,
            pool_gate_b: bias("pool_gate_b", 1)?,
            doc1: proj("doc1_w", 2 * STATE_DIM, 64)?,
            doc1_s: scale("doc1_s")?,
            doc1_b: bias("doc1_b", 64)?,
            doc2: proj("doc2_w", 64, STYLE_LOGIT_END)?,
            doc2_b: bias("doc2_b", STYLE_LOGIT_END)?,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The forward — sequential-ascending everywhere (G1-A(c)).
// ─────────────────────────────────────────────────────────────────────────

/// Reusable scratch — one arena per forward call site, reused across calls.
/// Every buffer is `clear()` + `resize()`d, never reallocated after the
/// first call at a given line count (G4-A's surface; that gate owns the
/// alloc-free claim, this shape merely makes it reachable).
#[derive(Default)]
pub struct ForwardScratch {
    pub emb: Vec<f32>,
    pub lex: Vec<f32>,
    pub lex_cat: Vec<f32>,
    pub proj1: Vec<f32>,
    pub prefix1: Vec<f32>,
    pub gate_in: Vec<f32>,
    pub gates: Vec<f32>,
    pub proj2: Vec<f32>,
    pub prefix2: Vec<f32>,
    pub fuse_in: Vec<f32>,
    pub cells: Vec<f32>,
    pub conv_buf: Vec<f32>,
    pub conv3_sum: Vec<f32>,
    pub conv7_sum: Vec<f32>,
    pub conv13_sum: Vec<f32>,
    pub col_sum_acc: Vec<f32>,
    pub col_sums: Vec<f32>,
    pub raw_sketch: Vec<f32>,
    pub line_feats: Vec<f32>,
    pub line_hidden: Vec<f32>,
    pub line_stats: Vec<f32>,
    pub state: Vec<f32>,
    pub state_prev: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub logits: Vec<f32>,
    pub weights_row: Vec<f32>,
    pub ctx_hidden: Vec<f32>,
    pub mlp_out: Vec<f32>,
    pub doc_mean: Vec<f32>,
    pub doc_max: Vec<f32>,
    pub doc_vec: Vec<f32>,
    pub gate_col: Vec<f32>,
    pub relu_state: Vec<f32>,
    pub probs: Vec<f32>,
    // Per-call temporaries hoisted out of the block bodies (G4-A): each is
    // `clear()`/`resize()`d to its exact width before use, so after the
    // first call at a given line count every resize lands inside the
    // already-grown capacity and the steady state allocates nothing.
    pub xcol: Vec<f32>,
    pub ycol: Vec<f32>,
    pub gate_out: Vec<f32>,
    pub fuse_out: Vec<f32>,
    pub col_sum_out: Vec<f32>,
    pub lmlp_out: Vec<f32>,
    pub lmlp_out2: Vec<f32>,
    pub q_tmp: Vec<f32>,
    pub k_tmp: Vec<f32>,
    pub v_tmp: Vec<f32>,
    pub attn_acc: Vec<f32>,
    pub wo_out: Vec<f32>,
    pub ctx_h_tmp: Vec<f32>,
    pub ctx_o_tmp: Vec<f32>,
    pub sq_out: Vec<f32>,
}

/// The forward's four output surfaces — what the G1-A fixtures pin.
#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Per-line squiggle probabilities, `(lines, NUM_KINDS)` row-major.
    pub line_probs: Vec<f32>,
    /// The two sigmoid heads (the source's `noul_probs`).
    pub noul: [f32; NOUL_HEADS],
    /// The 6 ordinal sub-scores (softmax·LEVELS folds — banded side output).
    pub subscores: [f32; NUM_SCORES],
    /// The 5 style probabilities (softmax — banded side output).
    pub style_probs: [f32; NUM_ARCHETYPES],
}

/// The source's `_sigmoid`: `1 / (1 + exp(-clip(x, -30, 30)))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}

/// The source's `_softmax` over one row: max-shifted exp, ascending sum,
/// divide in place. `out.len() == x.len()`; `out` may alias `x`.
fn softmax_into(x: &[f32], out: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &t in x {
        if t > max {
            max = t;
        }
    }
    let mut sum = 0.0f32;
    for (o, &t) in out.iter_mut().zip(x.iter()) {
        let e = (t - max).exp();
        *o = e;
        sum += e;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

/// The source's `_l2_norm` (eps = [`L2_EPS`]): `x / sqrt(Σx² + eps)`,
/// ascending sum.
fn l2_inv_scale(x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &t in x {
        acc += t * t;
    }
    let denom = (acc + L2_EPS).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        // The caller needs the NORM, not the normalized vector — the source
        // normalizes a vector; here we return the reciprocal scale so the
        // caller can apply it once.
        denom.recip()
    }
}

/// `y[r] = s[r] · (W @ x)[r]` — the matvec contract through the house
/// scalar kernel (sequential-ascending), with the per-channel scale applied
/// to the OUTPUT row in f32.
///
/// Scale-axis law: the source folds each scale along the stored matrix's
/// LAST axis, which for every scaled layer here is the output channel
/// (`pre_conv3_w (3,46,16)` × `(16,)` scales Cout; `lmlp1_w (294,124)` ×
/// `(124,)` scales the hidden width) — so the fold lands AFTER the dot,
/// never inside it. An empty `s` (a layer the source shipped unscaled) is
/// the identity. The multiply stays f32: routing it through the kernel's
/// f16 group scales would round at 2⁻¹¹ relative and blow the 1e-4 band.
fn matvec_scaled(w: &TernaryGroupWeights, s: &[f32], x: &[f32], y: &mut [f32]) {
    ternary_group_matvec_scalar(w, x, y);
    if !s.is_empty() {
        for (yr, &sv) in y.iter_mut().zip(s.iter()) {
            *yr *= sv;
        }
    }
}

/// One 1-D conv pass followed by ReLU, written straight into `out` as
/// `(lines · width, out_dim)` row-major.
///
/// The argument count IS the kernel shape (weights, scale, bias, input, five
/// dims, output) — a config struct would hide the fold geometry behind a
/// second indirection; allowed here because every call site passes exactly
/// the archived dims.
#[allow(clippy::too_many_arguments)]
///
/// `w` is the transposed kernel `(Cout, k·Cin)`; the source's pad law is
/// `pad = (k-1)/2` on BOTH sides (k odd throughout — 3, 5, 7, 13), so
/// output column `c` reads padded columns `c + o` for `o in 0..k`, i.e.
/// input column `c + o - pad`, out-of-range = zero. Bias lands FIRST (the
/// source seeds the accumulator with the bias before folding), then each
/// offset folds ascending.
pub fn conv_relu(
    w: &TernaryGroupWeights,
    s: &[f32],
    b: &[f32],
    x: &[f32],
    lines: usize,
    width: usize,
    in_dim: usize,
    k: usize,
    out_dim: usize,
    out: &mut Vec<f32>,
) {
    let pad = (k - 1) / 2;
    out.clear();
    out.resize(lines * width * out_dim, 0.0);
    for l in 0..lines {
        for c in 0..width {
            // Seed with bias, then fold the k offsets ascending.
            let obase = (l * width + c) * out_dim;
            out[obase..obase + out_dim].copy_from_slice(b);
            for o in 0..k {
                let src_col = c + o;
                let in_col = src_col.wrapping_sub(pad);
                if src_col < pad || in_col >= width {
                    continue; // zero padding
                }
                let ibase = (l * width + in_col) * in_dim;
                // Fold this offset's contribution: out += W[o-slice] @ x[col].
                // The kernel row for offset o spans flat columns
                // [o·Cin, (o+1)·Cin) of the (Cout, k·Cin) transpose.
                for out_c in 0..out_dim {
                    let mut acc = 0.0f32;
                    let g_lo = (o * in_dim) / GROUP_SIZE;
                    let g_hi = ((o + 1) * in_dim).div_ceil(GROUP_SIZE);
                    for g in g_lo..g_hi {
                        let gs = (g * GROUP_SIZE).max(o * in_dim);
                        let ge = ((g + 1) * GROUP_SIZE).min((o + 1) * in_dim);
                        let mut gacc = 0.0f32;
                        for col in gs..ge {
                            let i = col - o * in_dim;
                            let t = w.get(out_c, col);
                            if t != 0 {
                                gacc += t as f32 * x[ibase + i];
                            }
                        }
                        acc += gacc;
                    }
                    // The kernel's channel scale rides Cout (the last stored
                    // axis) — applied to the OUTPUT channel after the fold,
                    // in f32 (G1-A(b)(i) disclosure).
                    if s.is_empty() {
                        out[obase + out_c] += acc;
                    } else {
                        out[obase + out_c] += s[out_c] * acc;
                    }
                }
            }
        }
    }
    for t in out.iter_mut() {
        if *t < 0.0 {
            *t = 0.0;
        }
    }
}

impl CodeHygieneModel {
    /// Block 0 — embeddings, the lex conv, the two causal prefixes, the
    /// gate, and the column fuse. Writes `(lines·128, CELL_DIM)` cells.
    pub fn block0(&self, grid: &[[u8; MAX_COLS]], s: &mut ForwardScratch) {
        let lines = grid.len();
        let width = MAX_COLS;
        // emb = tok_emb[byte] + col_emb[col] (both unscaled trit tables).
        s.emb.clear();
        s.emb.resize(lines * width * EMB_DIM, 0.0);
        for l in 0..lines {
            for c in 0..width {
                let byte = grid[l][c] as usize;
                let base = (l * width + c) * EMB_DIM;
                for e in 0..EMB_DIM {
                    s.emb[base + e] =
                        self.tok_emb.get(byte, e) as f32 + self.col_emb.get(c, e) as f32;
                }
            }
        }
        // lex = relu(pre_conv3(emb)) — k=3, pad 1.
        conv_relu(
            &self.pre_conv3,
            &self.pre_conv3_s,
            &self.pre_conv3_b,
            &s.emb,
            lines,
            width,
            EMB_DIM,
            3,
            LEX_DIM,
            &mut s.lex,
        );

        // lex_cat = [emb, lex] (62) → proj1 = lex_cat @ pref1_proj (4).
        s.proj1.clear();
        for l in 0..lines {
            for c in 0..width {
                let ebase = (l * width + c) * EMB_DIM;
                let lbase = (l * width + c) * LEX_DIM;
                s.lex_cat.clear();
                s.lex_cat.extend_from_slice(&s.emb[ebase..ebase + EMB_DIM]);
                s.lex_cat.extend_from_slice(&s.lex[lbase..lbase + LEX_DIM]);
                let mut out1 = [0.0f32; PREFIX_DIM];
                ternary_group_matvec_scalar(&self.pref1_proj, &s.lex_cat, &mut out1);
                s.proj1.extend_from_slice(&out1);
            }
        }

        // prefix1 = W_causal1 @ proj1 per line: 4 sequential matvecs of the
        // (128, 128) causal matrix — ORIGINAL orientation (y = W @ x). The
        // same-layout widening (`from_ternary`) carries row_scale 1.0 into
        // every group scale exactly.
        s.prefix1.clear();
        s.prefix1.resize(lines * MAX_COLS * PREFIX_DIM, 0.0);
        {
            s.xcol.clear();
            s.xcol.resize(MAX_COLS, 0.0);
            s.ycol.clear();
            s.ycol.resize(MAX_COLS, 0.0);
            for l in 0..lines {
                for j in 0..PREFIX_DIM {
                    for c in 0..MAX_COLS {
                        s.xcol[c] = s.proj1[(l * MAX_COLS + c) * PREFIX_DIM + j];
                    }
                    ternary_group_matvec_scalar(&self.w_causal1_g, &s.xcol, &mut s.ycol);
                    for r in 0..MAX_COLS {
                        s.prefix1[(l * MAX_COLS + r) * PREFIX_DIM + j] = s.ycol[r];
                    }
                }
            }
        }

        // gates = relu([emb, lex, prefix1] (66) @ gate1 + bias).
        s.gates.clear();
        s.gates.resize(lines * MAX_COLS * GATE_DIM, 0.0);
        {
            s.gate_in.clear();
            s.gate_in.reserve(EMB_DIM + LEX_DIM + PREFIX_DIM);
            s.gate_out.clear();
            s.gate_out.resize(GATE_DIM, 0.0);
            for l in 0..lines {
                for c in 0..MAX_COLS {
                    let ebase = (l * MAX_COLS + c) * EMB_DIM;
                    let lbase = (l * MAX_COLS + c) * LEX_DIM;
                    let pbase = (l * MAX_COLS + c) * PREFIX_DIM;
                    s.gate_in.clear();
                    s.gate_in.extend_from_slice(&s.emb[ebase..ebase + EMB_DIM]);
                    s.gate_in.extend_from_slice(&s.lex[lbase..lbase + LEX_DIM]);
                    s.gate_in
                        .extend_from_slice(&s.prefix1[pbase..pbase + PREFIX_DIM]);
                    matvec_scaled(&self.gate1, &self.gate1_s, &s.gate_in, &mut s.gate_out);
                    let gbase = (l * MAX_COLS + c) * GATE_DIM;
                    for (gi, yv) in s.gate_out.iter().enumerate() {
                        s.gates[gbase + gi] = (*yv + self.gate1_b[gi]).max(0.0);
                    }
                }
            }
        }

        // prefix2 = W_causal2 @ (gates @ pref2_proj).
        s.proj2.clear();
        s.prefix2.clear();
        s.proj2.resize(lines * MAX_COLS * PREFIX_DIM, 0.0);
        s.prefix2.resize(lines * MAX_COLS * PREFIX_DIM, 0.0);
        {
            let mut yout = [0.0f32; PREFIX_DIM];
            for l in 0..lines {
                for c in 0..MAX_COLS {
                    let gbase = (l * MAX_COLS + c) * GATE_DIM;
                    matvec_scaled(
                        &self.pref2_proj,
                        &[],
                        &s.gates[gbase..gbase + GATE_DIM],
                        &mut yout,
                    );
                    s.proj2[(l * MAX_COLS + c) * PREFIX_DIM
                        ..(l * MAX_COLS + c) * PREFIX_DIM + PREFIX_DIM]
                        .copy_from_slice(&yout);
                }
            }
            s.xcol.clear();
            s.xcol.resize(MAX_COLS, 0.0);
            s.ycol.clear();
            s.ycol.resize(MAX_COLS, 0.0);
            for l in 0..lines {
                for j in 0..PREFIX_DIM {
                    for c in 0..MAX_COLS {
                        s.xcol[c] = s.proj2[(l * MAX_COLS + c) * PREFIX_DIM + j];
                    }
                    ternary_group_matvec_scalar(&self.w_causal2_g, &s.xcol, &mut s.ycol);
                    for r in 0..MAX_COLS {
                        s.prefix2[(l * MAX_COLS + r) * PREFIX_DIM + j] = s.ycol[r];
                    }
                }
            }
        }

        // cells = relu([emb, lex, gates, prefix2] (116) @ col_fuse + bias).
        s.cells.clear();
        s.cells.resize(lines * MAX_COLS * CELL_DIM, 0.0);
        {
            s.fuse_in.clear();
            s.fuse_in.reserve(EMB_DIM + LEX_DIM + GATE_DIM + PREFIX_DIM);
            s.fuse_out.clear();
            s.fuse_out.resize(CELL_DIM, 0.0);
            for l in 0..lines {
                for c in 0..MAX_COLS {
                    let ebase = (l * MAX_COLS + c) * EMB_DIM;
                    let lbase = (l * MAX_COLS + c) * LEX_DIM;
                    let gbase = (l * MAX_COLS + c) * GATE_DIM;
                    let pbase = (l * MAX_COLS + c) * PREFIX_DIM;
                    s.fuse_in.clear();
                    s.fuse_in.extend_from_slice(&s.emb[ebase..ebase + EMB_DIM]);
                    s.fuse_in.extend_from_slice(&s.lex[lbase..lbase + LEX_DIM]);
                    s.fuse_in
                        .extend_from_slice(&s.gates[gbase..gbase + GATE_DIM]);
                    s.fuse_in
                        .extend_from_slice(&s.prefix2[pbase..pbase + PREFIX_DIM]);
                    matvec_scaled(
                        &self.col_fuse,
                        &self.col_fuse_s,
                        &s.fuse_in,
                        &mut s.fuse_out,
                    );
                    let cbase = (l * MAX_COLS + c) * CELL_DIM;
                    for (ci, yv) in s.fuse_out.iter().enumerate() {
                        s.cells[cbase + ci] = (*yv + self.col_fuse_b[ci]).max(0.0);
                    }
                }
            }
        }
    }
}

impl CodeHygieneModel {
    /// Block 1 — the k=3/7/13 conv passes (column-summed), the column-sum
    /// projection, the sketch stem, and the line MLP → `state`
    /// `(lines, STATE_DIM)`.
    pub fn block1(&self, lines: usize, s: &mut ForwardScratch) {
        let width = MAX_COLS;
        // conv3/7/13: each `conv_relu` writes (lines·width, out); then the
        // column sum folds the width axis ascending.
        let fold = |w: &TernaryGroupWeights,
                    sc: &[f32],
                    b: &[f32],
                    k: usize,
                    out_dim: usize,
                    dst: &mut Vec<f32>,
                    buf: &mut Vec<f32>| {
            conv_relu(w, sc, b, &s.cells, lines, width, CELL_DIM, k, out_dim, buf);
            dst.clear();
            dst.resize(lines * out_dim, 0.0);
            for l in 0..lines {
                for c in 0..width {
                    let bse = (l * width + c) * out_dim;
                    for j in 0..out_dim {
                        dst[l * out_dim + j] += buf[bse + j];
                    }
                }
            }
        };
        fold(
            &self.conv3,
            &self.conv3_s,
            &self.conv3_b,
            3,
            CONV3_DIM,
            &mut s.conv3_sum,
            &mut s.conv_buf,
        );
        fold(
            &self.conv7,
            &self.conv7_s,
            &self.conv7_b,
            7,
            CONV7_DIM,
            &mut s.conv7_sum,
            &mut s.conv_buf,
        );
        fold(
            &self.conv13,
            &self.conv13_s,
            &self.conv13_b,
            13,
            CONV13_DIM,
            &mut s.conv13_sum,
            &mut s.conv_buf,
        );

        // col_sums = (cells summed over width) @ col_sum_proj (52 → 16).
        s.col_sums.clear();
        s.col_sums.resize(lines * COL_SUM_DIM, 0.0);
        {
            s.col_sum_acc.clear();
            s.col_sum_acc.resize(CELL_DIM, 0.0);
            s.col_sum_out.clear();
            s.col_sum_out.resize(COL_SUM_DIM, 0.0);
            for l in 0..lines {
                s.col_sum_acc.fill(0.0);
                for c in 0..width {
                    let base = (l * width + c) * CELL_DIM;
                    for j in 0..CELL_DIM {
                        s.col_sum_acc[j] += s.cells[base + j];
                    }
                }
                matvec_scaled(
                    &self.col_sum_proj,
                    &self.col_sum_s,
                    &s.col_sum_acc,
                    &mut s.col_sum_out,
                );
                s.col_sums[l * COL_SUM_DIM..(l + 1) * COL_SUM_DIM].copy_from_slice(&s.col_sum_out);
            }
        }

        // raw_sketch: k=5 conv → column sum → clamp to [0, 1] via
        // relu(x) - relu(x - 1).
        fold(
            &self.sketch,
            &self.sketch_s,
            &self.sketch_b,
            5,
            SKETCH_TOTAL,
            &mut s.raw_sketch,
            &mut s.conv_buf,
        );
        for t in s.raw_sketch.iter_mut() {
            let v = *t;
            *t = v.max(0.0) - (v - 1.0).max(0.0);
        }

        // line_feats = [col_sums (16), conv3 (64), conv7 (128), conv13 (86)].
        s.line_feats.clear();
        for l in 0..lines {
            s.line_feats
                .extend_from_slice(&s.col_sums[l * COL_SUM_DIM..(l + 1) * COL_SUM_DIM]);
            s.line_feats
                .extend_from_slice(&s.conv3_sum[l * CONV3_DIM..(l + 1) * CONV3_DIM]);
            s.line_feats
                .extend_from_slice(&s.conv7_sum[l * CONV7_DIM..(l + 1) * CONV7_DIM]);
            s.line_feats
                .extend_from_slice(&s.conv13_sum[l * CONV13_DIM..(l + 1) * CONV13_DIM]);
        }

        // line_hidden = relu(feats @ lmlp1 + b); line_stats = hidden @ lmlp2 + b.
        let feat_dim = COL_SUM_DIM + CONV3_DIM + CONV7_DIM + CONV13_DIM;
        s.line_hidden.clear();
        s.line_hidden.resize(lines * LMLP_HIDDEN, 0.0);
        s.line_stats.clear();
        s.line_stats.resize(lines * LINE_STATS_DIM, 0.0);
        {
            s.lmlp_out.clear();
            s.lmlp_out.resize(LMLP_HIDDEN, 0.0);
            s.lmlp_out2.clear();
            s.lmlp_out2.resize(LINE_STATS_DIM, 0.0);
            for l in 0..lines {
                let fbase = l * feat_dim;
                matvec_scaled(
                    &self.lmlp1,
                    &self.lmlp1_s,
                    &s.line_feats[fbase..fbase + feat_dim],
                    &mut s.lmlp_out,
                );
                for (hi, yv) in s.lmlp_out.iter().enumerate() {
                    s.line_hidden[l * LMLP_HIDDEN + hi] = (*yv + self.lmlp1_b[hi]).max(0.0);
                }
                let hbase = l * LMLP_HIDDEN;
                // lmlp2 ships NO scale in the source (no lmlp2_s vec) — the
                // empty scale is the disclosed no-fold.
                matvec_scaled(
                    &self.lmlp2,
                    &[],
                    &s.line_hidden[hbase..hbase + LMLP_HIDDEN],
                    &mut s.lmlp_out2,
                );
                for (si, yv) in s.lmlp_out2.iter().enumerate() {
                    s.line_stats[l * LINE_STATS_DIM + si] = *yv + self.lmlp2_b[si];
                }
            }
        }

        // state = [line_stats (146), l2(sketch[:40]), l2(sketch[40:])] (226).
        s.state.clear();
        s.state.resize(lines * STATE_DIM, 0.0);
        for l in 0..lines {
            let sbase = l * STATE_DIM;
            s.state[sbase..sbase + LINE_STATS_DIM]
                .copy_from_slice(&s.line_stats[l * LINE_STATS_DIM..(l + 1) * LINE_STATS_DIM]);
            let half = &s.raw_sketch[l * SKETCH_TOTAL..(l + 1) * SKETCH_TOTAL];
            let inv1 = l2_inv_scale(&half[..SKETCH_DIM]);
            let inv2 = l2_inv_scale(&half[SKETCH_DIM..]);
            for j in 0..SKETCH_DIM {
                s.state[sbase + LINE_STATS_DIM + j] = half[j] * inv1;
                s.state[sbase + LINE_STATS_DIM + SKETCH_DIM + j] = half[SKETCH_DIM + j] * inv2;
            }
        }
    }

    /// One sink-attention pass + context MLP on `state` (Block 2's layer
    /// body, the source's `_sink_attn` + `cmlp` residual, in place).
    fn attn_ctx_layer(&self, attn: &AttnLayer, ctx: &CtxMlp, lines: usize, s: &mut ForwardScratch) {
        let hd_qk = QK_HEAD_DIM; // 16
        let hd_v = attn.head_dim_v;
        let scale = (hd_qk as f32).sqrt();

        // q/k/v projections per line: (lines, QK_DIM/QK_DIM/v_dim).
        s.q.clear();
        s.k.clear();
        s.v.clear();
        s.q.resize(lines * QK_DIM, 0.0);
        s.k.resize(lines * QK_DIM, 0.0);
        s.v.resize(lines * hd_v * NUM_HEADS, 0.0);
        {
            s.q_tmp.clear();
            s.q_tmp.resize(QK_DIM, 0.0);
            s.k_tmp.clear();
            s.k_tmp.resize(QK_DIM, 0.0);
            s.v_tmp.clear();
            s.v_tmp.resize(hd_v * NUM_HEADS, 0.0);
            for l in 0..lines {
                let sbase = l * STATE_DIM;
                matvec_scaled(
                    &attn.wq,
                    &attn.wq_s,
                    &s.state[sbase..sbase + STATE_DIM],
                    &mut s.q_tmp,
                );
                matvec_scaled(
                    &attn.wk,
                    &attn.wk_s,
                    &s.state[sbase..sbase + STATE_DIM],
                    &mut s.k_tmp,
                );
                matvec_scaled(
                    &attn.wv,
                    &attn.wv_s,
                    &s.state[sbase..sbase + STATE_DIM],
                    &mut s.v_tmp,
                );
                s.q[l * QK_DIM..(l + 1) * QK_DIM].copy_from_slice(&s.q_tmp);
                s.k[l * QK_DIM..(l + 1) * QK_DIM].copy_from_slice(&s.k_tmp);
                s.v[l * hd_v * NUM_HEADS..(l + 1) * hd_v * NUM_HEADS].copy_from_slice(&s.v_tmp);
            }
        }

        // Attention per head: logits[q][key] = q·k/sqrt(hd) + b_rel[h][clip(k-q)],
        // then one sink column appended; softmax over (lines + 1); sink dropped.
        // merged[q][h·hd_v..] = Σ_key weight·v[head][key].
        s.attn_out.clear();
        s.attn_out.resize(lines * hd_v * NUM_HEADS, 0.0);
        {
            s.logits.clear();
            s.logits.resize(MAX_LINES + 1, 0.0);
            for h in 0..NUM_HEADS {
                let q_off = h * hd_qk;
                let k_off = h * hd_qk;
                let v_off = h * hd_v;
                for q in 0..lines {
                    // logits over keys 0..lines, then the sink logit.
                    let mut max = f32::NEG_INFINITY;
                    for key in 0..lines {
                        let mut dot = 0.0f32;
                        for d in 0..hd_qk {
                            dot += s.q[q * QK_DIM + q_off + d] * s.k[key * QK_DIM + k_off + d];
                        }
                        let rel = key as i64 - q as i64;
                        let clipped = (rel.clamp(-(MAX_REL_DIST as i64), MAX_REL_DIST as i64)
                            + MAX_REL_DIST as i64) as usize;
                        let logit = dot / scale + attn.b_rel[h * REL_TABLE + clipped];
                        s.logits[key] = logit;
                        if logit > max {
                            max = logit;
                        }
                    }
                    let sink = attn.b_sink[h];
                    s.logits[lines] = sink;
                    if sink > max {
                        max = sink;
                    }
                    // softmax with the sink included, then fold keys ascending,
                    // sink excluded from the output.
                    let mut sum = 0.0f32;
                    for key in 0..=lines {
                        let e = (s.logits[key] - max).exp();
                        s.logits[key] = e;
                        sum += e;
                    }
                    s.attn_acc.clear();
                    s.attn_acc.resize(hd_v, 0.0);
                    for key in 0..lines {
                        let wgt = s.logits[key] / sum;
                        if wgt != 0.0 {
                            let vbase = key * hd_v * NUM_HEADS + v_off;
                            for d in 0..hd_v {
                                s.attn_acc[d] += wgt * s.v[vbase + d];
                            }
                        }
                    }
                    let obase = q * hd_v * NUM_HEADS + v_off;
                    s.attn_out[obase..obase + hd_v].copy_from_slice(&s.attn_acc);
                }
            }
        }

        // wo: merged (v_dim) @ wo → (STATE_DIM), residual add.
        s.ctx_hidden.clear();
        s.ctx_hidden.resize(lines * STATE_DIM, 0.0);
        {
            s.wo_out.clear();
            s.wo_out.resize(STATE_DIM, 0.0);
            for l in 0..lines {
                let vbase = l * hd_v * NUM_HEADS;
                let merged = &s.attn_out[vbase..vbase + hd_v * NUM_HEADS];
                matvec_scaled(&attn.wo, &[], merged, &mut s.wo_out);
                let sbase = l * STATE_DIM;
                for si in 0..STATE_DIM {
                    s.ctx_hidden[sbase + si] = s.state[sbase + si] + s.wo_out[si];
                }
            }
        }
        // state ← ctx_hidden (the attention residual becomes the new state),
        // then the context MLP residual: state += relu(x@w1+b1)@w2 + b2.
        core::mem::swap(&mut s.state, &mut s.ctx_hidden);
        s.mlp_out.clear();
        s.mlp_out.resize(lines * STATE_DIM, 0.0);
        let w1_dim = ctx.hidden;
        {
            s.ctx_h_tmp.clear();
            s.ctx_h_tmp.resize(w1_dim, 0.0);
            s.ctx_o_tmp.clear();
            s.ctx_o_tmp.resize(STATE_DIM, 0.0);
            for l in 0..lines {
                let sbase = l * STATE_DIM;
                matvec_scaled(
                    &ctx.w1,
                    &ctx.w1_s,
                    &s.state[sbase..sbase + STATE_DIM],
                    &mut s.ctx_h_tmp,
                );
                for (hi, yv) in s.ctx_h_tmp.iter_mut().enumerate() {
                    *yv = (*yv + ctx.b1[hi]).max(0.0);
                }
                matvec_scaled(&ctx.w2, &ctx.w2_s, &s.ctx_h_tmp, &mut s.ctx_o_tmp);
                for si in 0..STATE_DIM {
                    s.mlp_out[sbase + si] = s.state[sbase + si] + s.ctx_o_tmp[si] + ctx.b2[si];
                }
            }
        }
        core::mem::swap(&mut s.state, &mut s.mlp_out);
    }

    /// Block 2 — the two attention + context-MLP passes.
    pub fn block2(&self, lines: usize, s: &mut ForwardScratch) {
        self.attn_ctx_layer(&self.attn1, &self.ctx1, lines, s);
        self.attn_ctx_layer(&self.attn2, &self.ctx2, lines, s);
    }
}

impl CodeHygieneModel {
    /// Block 3 — the per-line squiggle sigmoid, the pooled doc MLP, and the
    /// three heads. Writes into `out` (the caller owns its storage).
    pub fn block3(&self, lines: usize, s: &mut ForwardScratch, out: &mut ForwardOutput) {
        // line_probs = sigmoid(state @ sq + sq_b) — (lines, 8).
        out.line_probs.clear();
        out.line_probs.resize(lines * NUM_KINDS, 0.0);
        {
            s.sq_out.clear();
            s.sq_out.resize(NUM_KINDS, 0.0);
            for l in 0..lines {
                let sbase = l * STATE_DIM;
                matvec_scaled(
                    &self.sq,
                    &self.sq_s,
                    &s.state[sbase..sbase + STATE_DIM],
                    &mut s.sq_out,
                );
                for (ki, yv) in s.sq_out.iter().enumerate() {
                    out.line_probs[l * NUM_KINDS + ki] = sigmoid(yv + self.sq_b[ki]);
                }
            }
        }

        // active_gate = sigmoid(state @ pool_gate + b) — (lines, 1).
        s.gate_col.clear();
        s.gate_col.resize(lines, 0.0);
        {
            let mut y = [0.0f32; 1];
            for l in 0..lines {
                let sbase = l * STATE_DIM;
                matvec_scaled(
                    &self.pool_gate,
                    &self.pool_gate_s,
                    &s.state[sbase..sbase + STATE_DIM],
                    &mut y,
                );
                s.gate_col[l] = sigmoid(y[0] + self.pool_gate_b[0]);
            }
        }

        // doc_mean = Σ_l (state · gate) / (Σ_l gate + 1e-5), ascending;
        // doc_max = max_l (relu(state) · gate) — max is order-free.
        s.doc_mean.clear();
        s.doc_max.clear();
        s.doc_mean.resize(STATE_DIM, 0.0);
        s.doc_max.resize(STATE_DIM, f32::NEG_INFINITY);
        let mut gate_sum = 0.0f32;
        for l in 0..lines {
            let g = s.gate_col[l];
            gate_sum += g;
            let sbase = l * STATE_DIM;
            for si in 0..STATE_DIM {
                let v = s.state[sbase + si];
                s.doc_mean[si] += v * g;
                let rg = v.max(0.0) * g;
                if rg > s.doc_max[si] {
                    s.doc_max[si] = rg;
                }
            }
        }
        for si in 0..STATE_DIM {
            s.doc_mean[si] /= gate_sum + POOL_EPS;
        }
        if lines == 0 {
            for si in 0..STATE_DIM {
                s.doc_max[si] = 0.0;
            }
        }

        // doc_vec = [doc_mean, doc_max] (452) → relu(doc1) → doc2 logits.
        s.doc_vec.clear();
        s.doc_vec.extend_from_slice(&s.doc_mean);
        s.doc_vec.extend_from_slice(&s.doc_max);
        let mut hidden = [0.0f32; 64];
        let mut logits = [0.0f32; STYLE_LOGIT_END];
        {
            matvec_scaled(&self.doc1, &self.doc1_s, &s.doc_vec, &mut hidden);
            for hi in 0..64 {
                hidden[hi] = (hidden[hi] + self.doc1_b[hi]).max(0.0);
            }
            matvec_scaled(&self.doc2, &[], &hidden, &mut logits);
            // The head bias rides OUTSIDE the matvec (the source folds it at
            // the logit site; doc2 ships no scale).
            for (lo, b) in logits.iter_mut().zip(self.doc2_b.iter()) {
                *lo += b;
            }
        }

        // heads: noul = sigmoid(logits[0..2]); ordinal = softmax(6×4)·LEVELS;
        // style = softmax(logits[26..31]).
        for i in 0..NOUL_HEADS {
            out.noul[i] = sigmoid(logits[i]);
        }
        {
            let mut row = [0.0f32; 4];
            let mut sm = [0.0f32; 4];
            for sc in 0..NUM_SCORES {
                for j in 0..4 {
                    row[j] = logits[NOUL_HEADS + sc * 4 + j];
                }
                softmax_into(&row, &mut sm);
                let mut acc = 0.0f32;
                for j in 0..4 {
                    acc += sm[j] * LEVELS[j];
                }
                out.subscores[sc] = acc;
            }
        }
        {
            let mut sm = [0.0f32; NUM_ARCHETYPES];
            softmax_into(&logits[SCORE_LOGIT_END..STYLE_LOGIT_END], &mut sm);
            out.style_probs.copy_from_slice(&sm);
        }
    }

    /// The zero-alloc apply path: every per-call temporary lives in
    /// `scratch` (added for G4-A) and the caller reuses `out`, whose
    /// `line_probs` capacity persists across calls at a fixed line count.
    /// Steady state — after one warm call — performs ZERO heap
    /// allocations; the counting-allocator canary
    /// (`ternary_circuit_forward_alloc_gates`) pins it.
    pub fn forward_into(
        &self,
        grid: &[[u8; MAX_COLS]],
        scratch: &mut ForwardScratch,
        out: &mut ForwardOutput,
    ) -> Result<(), ForwardError> {
        let lines = grid.len();
        if lines > MAX_LINES {
            return Err(ForwardError::GridShape { lines });
        }
        self.block0(grid, scratch);
        self.block1(lines, scratch);
        self.block2(lines, scratch);
        self.block3(lines, scratch, out);
        Ok(())
    }

    /// Convenience form — allocates a fresh [`ForwardOutput`] every call
    /// (the pinned G1-A exactness gate replays through this path).
    /// Steady-state serving uses [`forward_into`].
    pub fn forward(
        &self,
        grid: &[[u8; MAX_COLS]],
        scratch: &mut ForwardScratch,
    ) -> Result<ForwardOutput, ForwardError> {
        let mut out = ForwardOutput {
            line_probs: Vec::new(),
            noul: [0.0; NOUL_HEADS],
            subscores: [0.0; NUM_SCORES],
            style_probs: [0.0; NUM_ARCHETYPES],
        };
        self.forward_into(grid, scratch, &mut out)?;
        Ok(out)
    }
}
