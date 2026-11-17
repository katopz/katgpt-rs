//! `ternary_circuit` — the circuit→ternary compiled-lane family
//! (Issue 1225; Research 677 GOAT fusion: brianbell-x/codehygiene @
//! [`SOURCE_PIN`]).
//!
//! T1 (this module): the npz→`.cact` transcode lane — the lossless container
//! move from the source's `weights.npz` (`trits` / `vecs` / `meta` members,
//! DEFLATE-decoded once by the committed stdlib emitter) onto the house
//! `kpt_archive` container, so every later arm (Arm A forward port, Arm B
//! exact-circuit builder) loads ONE verified artifact instead of the raw
//! npz triple.
//!
//! # Convention-free by construction (the T1a law, carried into the container)
//!
//! The transcode interprets NO axis semantics. A 2-D matrix `(a, b)` stores
//! as a layer with `rows = a, cols = b`; a 3-D matrix `(a, b, c)` stores as
//! `rows = a, cols = b·c` — exactly the source's own flat C-order reshape in
//! `unpack_ternary_archive` — and every layer's `row_scale` stays the neutral
//! `1.0`. The source folds its per-channel scales at load time by
//! broadcasting along the LAST axis (`mat * vecs[scale_key]`, which is a
//! cols-axis semantics our per-row `row_scale` cannot and must not impersonate);
//! the scales therefore ride the archive VERBATIM as their own layers, and
//! the folding semantics belong to Arm A's forward port. The f32 vec part
//! (scales, biases, the `(8, 65)` relative-offset tables) rides as degenerate
//! layers (`cols = 0`, payload in `row_scale`) — the only f32 lane the
//! nameless-positional KPT1 format offers — with each entry's shape recorded
//! in the transcode manifest derived from `meta` at assembly time (never
//! hand-typed).
//!
//! # Grammar laws asserted here (the emitter's laws, re-asserted at assembly)
//!
//! - the meta grammar: `trit_part|vec_part`, entries `name:dims:count`
//!   joined by `" ; "`, dims comma-separated positive integers;
//! - per-matrix `packed_len == ceil(elements / 5)` and the SUM of packed
//!   lens equals the trit buffer length (the flat-stream law);
//! - the pad trits between a matrix's last element and its packed-region
//!   end decode to `0` (the packer zero-pads the tail byte — the neutral
//!   padding only works because digit 0 decodes to trit 0);
//! - per-vec-entry `count == prod(dims)` and the sum of counts equals the
//!   f32 buffer length;
//! - a matrix's scale entry (named by [`scale_key_for`], the source's own
//!   pairing rule) has `count == matrix last dim`.
//!
//! Layer order: all trit layers in trit-part meta order, then all vec layers
//! in vec-part meta order; [`Transcoded::manifest`] is the positional
//! name→index map the forward port resolves against.

use core::fmt;

use katgpt_types::TernaryWeights;
use serde::Serialize;

/// The inspected source commit the transcode (and the committed fixture
/// intermediate) were derived from. Everything byte-level here is pinned to
/// THIS tree; the tip has already moved past it.
pub const SOURCE_PIN: &str = "423c7e1f02ef409bbaea209268bfcde36e0f1cb0";

/// `_POW3` — the source's per-byte digit divisors, ascending (`__init__.py:27`).
/// Shared by the archive unpacker and the Arm-B builder packer (the one
/// digit law; `ternary_circuit_build::pack_trits` is its exact inverse).
pub const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Everything that can go wrong transcribing the npz triple into layers.
#[derive(Debug)]
pub enum TranscodeError {
    /// The meta string does not match the emitter's verified grammar.
    MetaGrammar { detail: String },
    /// A trit entry's `packed_len` is not `ceil(elements / 5)`.
    PackedLenLaw {
        name: String,
        packed_len: usize,
        expected: usize,
    },
    /// A vec entry's `count` is not `prod(dims)`.
    VecCountLaw {
        name: String,
        count: usize,
        expected: usize,
    },
    /// A matrix rank other than 2 or 3 (the source ships exactly these).
    RankTooHigh { name: String, rank: usize },
    /// The trit-part packed lens do not sum to the trit buffer length.
    TritLenMismatch { meta_sum: usize, have: usize },
    /// The vec-part counts do not sum to the f32 buffer length.
    VecLenMismatch { meta_sum: usize, have: usize },
    /// A pad trit (between a matrix's last element and its packed-region
    /// end) decoded non-zero — the tail byte was not zero-padded.
    PadTritNonZero {
        name: String,
        flat_index: usize,
        trit: i8,
    },
    /// A matrix's scale entry length ≠ the matrix's last dim.
    ScaleLenLaw {
        scale_name: String,
        count: usize,
        last_dim: usize,
    },
}

impl fmt::Display for TranscodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetaGrammar { detail } => write!(f, "meta grammar: {detail}"),
            Self::PackedLenLaw {
                name,
                packed_len,
                expected,
            } => write!(
                f,
                "meta {name}: packed_len {packed_len} != ceil(elems/5) {expected}"
            ),
            Self::VecCountLaw {
                name,
                count,
                expected,
            } => write!(f, "meta {name}: count {count} != prod(dims) {expected}"),
            Self::RankTooHigh { name, rank } => {
                write!(f, "meta {name}: rank {rank} unsupported (2-D or 3-D only)")
            }
            Self::TritLenMismatch { meta_sum, have } => write!(
                f,
                "trit-part packed lens sum to {meta_sum}, trit buffer has {have} bytes"
            ),
            Self::VecLenMismatch { meta_sum, have } => write!(
                f,
                "vec-part counts sum to {meta_sum}, f32 buffer has {have} elements"
            ),
            Self::PadTritNonZero {
                name,
                flat_index,
                trit,
            } => write!(
                f,
                "meta {name}: pad trit at flat {flat_index} decoded {trit} (tail byte not zero-padded)"
            ),
            Self::ScaleLenLaw {
                scale_name,
                count,
                last_dim,
            } => write!(
                f,
                "meta {scale_name}: count {count} != matrix last dim {last_dim}"
            ),
        }
    }
}

impl std::error::Error for TranscodeError {}

fn grammar(detail: impl Into<String>) -> TranscodeError {
    TranscodeError::MetaGrammar {
        detail: detail.into(),
    }
}

/// One trit-part meta entry (a ternary matrix).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TritEntry {
    pub name: String,
    pub dims: Vec<usize>,
    /// Packed byte length (`ceil(elements / 5)` — asserted at assembly).
    pub packed_len: usize,
}

/// One vec-part meta entry (an f32 vector: per-channel scale, bias, or
/// relative-offset table).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VecEntry {
    pub name: String,
    pub dims: Vec<usize>,
    /// Element count (`prod(dims)` — asserted at assembly).
    pub count: usize,
}

/// The parsed meta string (the emitter's `meta.txt`, grammar-verified).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Meta {
    pub trits: Vec<TritEntry>,
    pub vecs: Vec<VecEntry>,
}

/// Parse the meta string. The grammar is the emitter's verified law:
/// `trit_part|vec_part`, entries `name:dims:count` joined by `" ; "`.
pub fn parse_meta(meta: &str) -> Result<Meta, TranscodeError> {
    let meta = meta.trim();
    let Some((trit_part, vec_part)) = meta.split_once('|') else {
        return Err(grammar("no '|' separator"));
    };
    fn parse_entry(raw: &str) -> Result<(String, Vec<usize>, usize), TranscodeError> {
        let segs: Vec<&str> = raw.split(':').collect();
        if segs.len() != 3 {
            return Err(grammar(format!(
                "entry {raw:?}: expected name:dims:count, got {} segments",
                segs.len()
            )));
        }
        let name = segs[0].trim();
        if name.is_empty() {
            return Err(grammar("empty entry name"));
        }
        let mut dims = Vec::new();
        for d in segs[1].split(',') {
            let d = d.trim().parse::<usize>().map_err(|_| {
                grammar(format!("entry {name}: dim {d:?} is not a positive integer"))
            })?;
            if d == 0 {
                return Err(grammar(format!("entry {name}: zero dim")));
            }
            dims.push(d);
        }
        if dims.is_empty() {
            return Err(grammar(format!("entry {name}: no dims")));
        }
        let count = segs[2]
            .trim()
            .parse::<usize>()
            .map_err(|_| grammar(format!("entry {name}: count is not an integer")))?;
        Ok((name.to_string(), dims, count))
    }
    let mut trits = Vec::new();
    for raw in trit_part.split(" ; ") {
        if raw.is_empty() {
            return Err(grammar("empty trit entry"));
        }
        let (name, dims, packed_len) = parse_entry(raw)?;
        trits.push(TritEntry {
            name,
            dims,
            packed_len,
        });
    }
    let mut vecs = Vec::new();
    for raw in vec_part.split(" ; ") {
        if raw.is_empty() {
            return Err(grammar("empty vec entry"));
        }
        let (name, dims, count) = parse_entry(raw)?;
        vecs.push(VecEntry { name, dims, count });
    }
    Ok(Meta { trits, vecs })
}

/// The source's load-time scale pairing rule (`__init__.py:276-278` @ the
/// pin): `{stem}_s` when the matrix name ends in `_w`, else `{name}_s`, with
/// the one named special case `col_sum_proj` → `col_sum_s`.
pub fn scale_key_for(matrix_name: &str) -> String {
    if matrix_name == "col_sum_proj" {
        "col_sum_s".to_string()
    } else if let Some(stem) = matrix_name.strip_suffix("_w") {
        format!("{stem}_s")
    } else {
        format!("{matrix_name}_s")
    }
}

/// The source's 5-trits-per-byte unpacker digit law
/// (`__init__.py:265-266` @ the pin): digit `k` of byte `b` is
/// `(b / 3^k) % 3` (k ascending over [`POW3`]), and the trit is
/// `((digit + 1) % 3) - 1` — digit 0 → trit 0, digit 1 → +1, digit 2 → −1.
/// The packer zero-pads each matrix's tail byte, and digit 0 decoding to
/// trit 0 is what makes that padding neutral (asserted at assembly).
#[inline]
pub fn trit_at(buf: &[u8], flat_index: usize) -> i8 {
    let byte = buf[flat_index / 5];
    let digit = (byte / POW3[flat_index % 5]) % 3;
    ((digit + 1) % 3) as i8 - 1
}

/// One positional layer in the transcode manifest (the name→index map the
/// forward port resolves against — the archive itself is nameless).
#[derive(Debug, Clone, Serialize)]
pub struct ManifestRow {
    pub idx: usize,
    pub name: String,
    pub kind: LayerKind,
    /// `Some` on vec layers only: scale (paired to a matrix by
    /// [`scale_key_for`]) vs data (bias / offset table riding `params`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<VecRole>,
    /// The meta dims verbatim (a vec layer's flat count is `prod(dims)`).
    pub dims: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// A ternary matrix layer (bitplanes + neutral `row_scale`).
    Trit,
    /// An f32 degenerate layer (`cols = 0`, payload in `row_scale`).
    Vec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VecRole {
    /// The paired matrix's per-channel scale (last-axis semantics at the
    /// forward, per the source's load-time fold).
    Scale,
    /// A bias or offset table riding the source's `params` verbatim.
    Data,
}

/// The assembled archive input: layers in positional order + the manifest.
#[derive(Debug, Clone)]
pub struct Transcoded {
    pub layers: Vec<TernaryWeights>,
    pub manifest: Vec<ManifestRow>,
}

/// Assemble the archive layers from the lossless intermediate.
///
/// Trit layers first (trit-part meta order), then vec layers (vec-part meta
/// order). Every layer's `row_scale` is neutral `1.0` on trit layers and the
/// verbatim f32 payload on vec layers — the per-channel scale folding is a
/// LAST-axis semantics and belongs to the forward port (Arm A), never to the
/// container.
pub fn assemble_layers(
    trits: &[u8],
    vecs: &[f32],
    meta: &Meta,
) -> Result<Transcoded, TranscodeError> {
    // Scale names = the source's pairing rule applied over the matrix names;
    // a vec entry is a Scale iff its name lands in this set.
    let scale_names: std::collections::BTreeSet<String> =
        meta.trits.iter().map(|m| scale_key_for(&m.name)).collect();
    let scale_len_of: std::collections::BTreeMap<String, usize> = meta
        .trits
        .iter()
        .map(|m| {
            (
                scale_key_for(&m.name),
                *m.dims.last().expect("non-empty dims"),
            )
        })
        .collect();

    let total = meta.trits.len() + meta.vecs.len();
    let mut layers: Vec<TernaryWeights> = Vec::with_capacity(total);
    let mut manifest: Vec<ManifestRow> = Vec::with_capacity(total);

    // Upfront sum laws: every later buffer read is bounds-safe by
    // construction, so the loop can never panic past a typed error.
    let meta_trit_sum: usize = meta.trits.iter().map(|e| e.packed_len).sum();
    if meta_trit_sum != trits.len() {
        return Err(TranscodeError::TritLenMismatch {
            meta_sum: meta_trit_sum,
            have: trits.len(),
        });
    }
    let meta_vec_sum: usize = meta.vecs.iter().map(|e| e.count).sum();
    if meta_vec_sum != vecs.len() {
        return Err(TranscodeError::VecLenMismatch {
            meta_sum: meta_vec_sum,
            have: vecs.len(),
        });
    }

    let mut byte_offset = 0usize;
    for e in &meta.trits {
        let elems: usize = e.dims.iter().product();
        let need = elems.div_ceil(5);
        if e.packed_len != need {
            return Err(TranscodeError::PackedLenLaw {
                name: e.name.clone(),
                packed_len: e.packed_len,
                expected: need,
            });
        }
        let (rows, cols) = match e.dims.as_slice() {
            [a, b] => (*a, *b),
            [a, b, c] => (
                *a,
                b.checked_mul(*c)
                    .expect("meta dims product overflows usize"),
            ),
            other => {
                return Err(TranscodeError::RankTooHigh {
                    name: e.name.clone(),
                    rank: other.len(),
                });
            }
        };
        let mut tw = TernaryWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let t = trit_at(trits, byte_offset * 5 + r * cols + c);
                if t != 0 {
                    let idx = r * tw.blocks64 + (c >> 6);
                    let mask = 1u64 << (c & 63);
                    if t > 0 {
                        tw.pos_bits[idx] |= mask;
                    } else {
                        tw.neg_bits[idx] |= mask;
                    }
                }
            }
        }
        // The pad trits of this matrix's packed region must decode zero.
        for pad in elems..need * 5 {
            let t = trit_at(trits, byte_offset * 5 + pad);
            if t != 0 {
                return Err(TranscodeError::PadTritNonZero {
                    name: e.name.clone(),
                    flat_index: byte_offset * 5 + pad,
                    trit: t,
                });
            }
        }
        byte_offset += e.packed_len;
        manifest.push(ManifestRow {
            idx: layers.len(),
            name: e.name.clone(),
            kind: LayerKind::Trit,
            role: None,
            dims: e.dims.clone(),
            rows,
            cols,
        });
        layers.push(tw);
    }

    let mut vec_offset = 0usize;
    for e in &meta.vecs {
        let count: usize = e.dims.iter().product();
        if count != e.count {
            return Err(TranscodeError::VecCountLaw {
                name: e.name.clone(),
                count: e.count,
                expected: count,
            });
        }
        let role = if scale_names.contains(e.name.as_str()) {
            let last_dim = scale_len_of
                .get(e.name.as_str())
                .copied()
                .expect("scale_names membership implies the len row");
            if count != last_dim {
                return Err(TranscodeError::ScaleLenLaw {
                    scale_name: e.name.clone(),
                    count,
                    last_dim,
                });
            }
            VecRole::Scale
        } else {
            VecRole::Data
        };
        let mut tw = TernaryWeights::new(count, 0);
        tw.row_scale
            .copy_from_slice(&vecs[vec_offset..vec_offset + count]);
        vec_offset += count;
        manifest.push(ManifestRow {
            idx: layers.len(),
            name: e.name.clone(),
            kind: LayerKind::Vec,
            role: Some(role),
            dims: e.dims.clone(),
            rows: count,
            cols: 0,
        });
        layers.push(tw);
    }

    Ok(Transcoded { layers, manifest })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack5(trits: &[i8]) -> u8 {
        assert_eq!(trits.len(), 5);
        let mut b = 0u32;
        for (k, &t) in trits.iter().enumerate() {
            let digit = match t {
                0 => 0u32,
                1 => 1,
                -1 => 2,
                _ => panic!("bad trit"),
            };
            b += digit * 3u32.pow(k as u32);
        }
        b as u8
    }

    #[test]
    fn digit_law_known_bytes() {
        // Hand-derived from ((d+1)%3)-1 over _POW3 = [1,3,9,27,81].
        let buf = [0u8, 1, 2, 81, 242];
        assert_eq!(trit_at(&buf, 0), 0);
        assert_eq!(trit_at(&buf, 1), 0);
        assert_eq!(trit_at(&buf, 2), 0);
        assert_eq!(trit_at(&buf, 3), 0);
        assert_eq!(trit_at(&buf, 4), 0);
        // byte 1: d0 = 1 -> +1 at flat 5
        assert_eq!(trit_at(&buf, 5), 1);
        // byte 2: d0 = 2 -> -1
        assert_eq!(trit_at(&buf, 10), -1);
        // byte 3 (81 = 3^4): digits d0..d3 = 0, d4 = 1 -> +1 at flat 19
        for k in 0..4 {
            assert_eq!(trit_at(&buf, 15 + k), 0, "81 digit {k}");
        }
        assert_eq!(trit_at(&buf, 19), 1);
        // byte 4 (242 = 2·(1+3+9+27+81)): every digit 2 -> -1 at flats 20..24
        for k in 0..5 {
            assert_eq!(trit_at(&buf, 20 + k), -1);
        }
    }

    #[test]
    fn pack_unpack_identity_over_pattern_sweep() {
        // Pack (the emitter's inverse law) then unpack: identity over an
        // exhaustive single-digit sweep + a deterministic mixed pattern.
        let all = [
            [0i8; 5],
            [1, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1],
            [1, -1, 0, 1, -1],
        ];
        for group in all {
            let byte = pack5(&group);
            for (k, &t) in group.iter().enumerate() {
                assert_eq!(trit_at(&[byte], k), t, "byte {byte} digit {k}");
            }
        }
        // 242 is the canonical 5-digit base-3 max; 243..255 are non-canonical
        // (d4 would exceed 2 only past 242 — 243 = 3^5 decodes digit-wise to
        // d4 = 3 % 3 = 0 — still in-domain for the reader, never produced by
        // the packer).
        assert_eq!(pack5(&[-1, -1, -1, -1, -1]), 242);
    }

    #[test]
    fn scale_key_rule_matches_source() {
        assert_eq!(scale_key_for("gate1_w"), "gate1_s");
        assert_eq!(scale_key_for("col_sum_proj"), "col_sum_s");
        assert_eq!(scale_key_for("attn1_wq"), "attn1_wq_s");
        assert_eq!(scale_key_for("attn1_wo"), "attn1_wo_s");
        assert_eq!(scale_key_for("tok_emb"), "tok_emb_s");
    }

    #[test]
    fn meta_grammar_refusals() {
        assert!(parse_meta("no separator here").is_err());
        assert!(parse_meta("a:2,3:2|x:4:4").is_ok());
        // wrong segment count
        assert!(parse_meta("a:2,3|b:4:4").is_err());
        // non-integer dim
        assert!(parse_meta("a:x,3:6|b:4:4").is_err());
        // zero dim
        assert!(parse_meta("a:0,3:0|b:4:4").is_err());
        // empty entry
        assert!(parse_meta("a:2,3:6| ; b:4:4").is_err());
    }

    #[test]
    fn assembly_laws() {
        // m_a-like 2-D matrix (2x2 = 4 elems -> 1 packed byte, 1 pad trit)
        // + a 3-D matrix (2x2x2 = 8 elems -> 2 bytes) + one scale + one bias.
        let meta = "m_a:2,2:1 ; m_b:2,2,2:2|m_a_s:2:2 ; m_a_b:2:2";
        let m = parse_meta(meta).unwrap();
        // m_a occupies bytes 0..1 (flat trits 0..5); m_b bytes 1..3 (flat 5..15, 8 elems + 2 pads).
        let mut trits = vec![0u8; 3];
        // byte 0: digits d0.. = [1, 0, -1->2, 0, 0(pad)] -> trits [1, 0, -1, 0, 0(pad)]
        trits[0] = pack5(&[1, 0, -1, 0, 0]);
        // m_b elems 8 -> bytes 1..3 (10 trit slots, 2 pads): alternate +1/-1/0
        let mb = [1i8, -1, 0, 1, -1, 0, 1, -1, 0, 0];
        let (mb_lo, mb_hi) = mb.split_at(5);
        trits[1] = pack5(mb_lo);
        trits[2] = pack5(mb_hi);
        let vecs: Vec<f32> = [1.5, 2.5, 10.0, 20.0].to_vec();
        let out = assemble_layers(&trits, &vecs, &m).unwrap();
        assert_eq!(out.layers.len(), 4);
        assert_eq!(out.manifest.len(), 4);
        // m_a: 2-D -> rows 2 cols 2; flat trits 0..4 = [1, 0, -1, 0]
        let a = &out.layers[0];
        assert_eq!((a.rows, a.cols, a.blocks64), (2, 2, 1));
        assert_eq!(a.get(0, 0), 1);
        assert_eq!(a.get(0, 1), 0);
        assert_eq!(a.get(1, 0), -1);
        assert_eq!(a.get(1, 1), 0);
        assert!(a.row_scale.iter().all(|&s| s == 1.0));
        // m_b: 3-D -> rows 2 cols 4; flat trits 5..13
        let b = &out.layers[1];
        assert_eq!((b.rows, b.cols, b.blocks64), (2, 4, 1));
        for (i, &t) in mb.iter().take(8).enumerate() {
            assert_eq!(b.get(i / 4, i % 4), t, "m_b flat {i}");
        }
        // scale: rows 2 cols 0, verbatim payload, role Scale
        let s = &out.layers[2];
        assert_eq!((s.rows, s.cols), (2, 0));
        assert_eq!(s.row_scale, &[1.5, 2.5]);
        assert_eq!(out.manifest[2].role, Some(VecRole::Scale));
        // bias: role Data, verbatim
        let bias = &out.layers[3];
        assert_eq!(bias.row_scale, &[10.0, 20.0]);
        assert_eq!(out.manifest[3].role, Some(VecRole::Data));
    }

    #[test]
    fn assembly_refusals() {
        // packed_len law: 6 elems needs 2 bytes, meta says 1 (buffer length
        // matches the meta sum so the packed-len law is the one that fires)
        let m = parse_meta("m:2,3:1|b:1:1").unwrap();
        let err = assemble_layers(&[0u8], &[0.0], &m).unwrap_err();
        assert!(matches!(err, TranscodeError::PackedLenLaw { .. }));
        // pad-trit law: 4 elems -> 1 byte; pad digit non-zero -> refuse.
        // byte = pack5([1,0,0,0,0]) has d4 = 0; use digit pattern [1,0,0,0,1] -> pad trit +1
        let bad = pack5(&[1, 0, 0, 0, 1]);
        let m2 = parse_meta("m:2,2:1|b:1:1").unwrap();
        let err = assemble_layers(&[bad], &[0.0], &m2).unwrap_err();
        assert!(matches!(err, TranscodeError::PadTritNonZero { .. }));
        // trit-sum law: meta sums 2 bytes, buffer has 1
        let m3 = parse_meta("a:2,2:1 ; b:2,2:1|b_x:1:1").unwrap();
        let err = assemble_layers(&[0u8], &[0.0], &m3).unwrap_err();
        assert!(matches!(err, TranscodeError::TritLenMismatch { .. }));
        // scale-length law: scale count != matrix last dim
        // (the pairing follows the source's rule: m_w -> m_s)
        let m5 = parse_meta("m_w:4,3:3|m_s:2:2").unwrap();
        let trits5 = [
            pack5(&[0, 0, 0, 0, 0]),
            pack5(&[0, 0, 0, 0, 0]),
            pack5(&[0, 0, 0, 0, 0]),
        ];
        let err = assemble_layers(&trits5, &[0.0, 0.0], &m5).unwrap_err();
        assert!(matches!(err, TranscodeError::ScaleLenLaw { .. }));
        // vec-sum law (upfront: a short f32 buffer refuses typed, never panics)
        let m4 = parse_meta("a:2,2:1|b_x:2:2").unwrap();
        let err = assemble_layers(&[pack5(&[0, 0, 0, 0, 0])], &[0.0], &m4).unwrap_err();
        assert!(matches!(err, TranscodeError::VecLenMismatch { .. }));
    }
}
