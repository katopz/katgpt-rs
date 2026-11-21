//! Surgical read-patch artifact class — Issue 996, distilled from
//! Research 492 (the owl Sparse-Rewiring lane; owl weights/datasets are
//! CC BY-NC 4.0 — local comparison only, NEVER a shipped artifact).
//!
//! A row-sparse, WRITE-FROZEN weight delta: per layer, selected
//! MLP-neuron OUTPUT-ROW indices, ONE shared read-direction block
//! `a` `[rank × in_dim]`, and per-neuron coefficient rows `b`
//! `[n × rank]` for the gate and/or up projections. Merge =
//! `W[idx_i] += scale * b_i @ a` — an outer product scattered into the
//! selected rows and NOTHING else.
//!
//! # Laws (each pinned by the named test)
//!
//! - **Write-frozen.** A merge touches the selected rows only; every
//!   other row stays BIT-IDENTICAL
//!   (`merge_touches_only_selected_rows`,
//!   `write_frozen_across_projections`).
//! - **Deterministic merge.** The fold order is pinned — neurons
//!   ascending × rank ascending × in_dim ascending, plain `+=`, the
//!   Issue-940 sequential-fold law — never a platform-SIMD kernel
//!   (`merge_is_bit_identical_across_runs`).
//! - **Byte-committed.** The canonical encoding stores every `f32` as
//!   its raw LE bit pattern and pins a BLAKE3 digest; tampering refuses
//!   (`canonical_round_trip_bit_identical`, `golden_digest_pin`,
//!   `tampered_canonical_refuses`).
//! - **Monotonic hot-swap.** [`SurgicalPatchSlot`] mirrors the
//!   `GraftSlot` A2 law (`crate::graft_artifact`): a corrupt artifact
//!   refuses BEFORE any lineage comparison, fork + downgrade refuse
//!   loud, and every refusal leaves the mount untouched
//!   (`slot_monotonic_laws`, `slot_refuses_corrupt_and_stays_mounted`).
//! - **KB-scale.** 89 neurons / 20 layers / rank 4 / in_dim 4096
//!   encodes to ≈1.25 MiB — ≥7× smaller than the dense rank-4 gate+up
//!   LoRA twin at intermediate 12288 (`kb_scale_artifact_bound`).
//! - **Substrate-first apply.** [`SurgicalReadPatch::projection_to_lora`]
//!   materializes a dense-b [`LoraAdapter`] (exact `+0.0` off the
//!   selected rows) so the runtime side-patch rides the EXISTING
//!   `lora_apply` path — never a new kernel
//!   (`expand_to_lora_matches_row_merge`,
//!   `expand_b_rows_off_neurons_are_positive_zero`).
//! - **Ternary purity.** fp deltas are NEVER merged into ternary rows:
//!   this API accepts plain `f32` weights only. On a ternary-served
//!   model the compatible shape is the runtime side-patch (the LoRA
//!   expansion above), never a weight rewrite.
//!
//! # The three modelless lanes (Issue 996)
//!
//! (a) **Census scorer.** [`census_scores_into`] is the closed-form
//! `|mean(persona) − mean(base)|` per neuron (f64 sequential folds,
//! f32 out); [`select_top_into`] takes the top-n under the Issue-940
//! tie rule (strict `>` keeps the FIRST index; NaN never selected).
//! Consumers MUST run the random-ranking control: if random ≈ census,
//! the scorer is noise and the lane closes negative
//! (`census_scores_exact_zero_on_null_and_planted_top`,
//! `select_top_first_index_tie_rule_and_nan_skip`).
//!
//! (b) **Closed-form construction** (the fusion experiment's
//! construction half). [`construct_from_census_into`] derives the
//! shared read direction `a = normalized(persona-input-mean −
//! base-input-mean)` and per-neuron coefficients `b_i = dot(base row i,
//! a)`; identical corpora are the typed honest abstain
//! ([`SurgicalPatchError::NoSeparation`]). The QUALITY claim against
//! GD-trained values is PRE-REGISTERED AND NOT RUN (below)
//! (`construct_from_census_matches_reference`,
//! `construct_refuses_null_corpus`,
//! `construct_assemble_merge_bit_matches_reference`).
//!
//! (c) **Fact-swap plan.** [`plan_fact_swaps`] emits a seeded,
//! deterministic, BLAKE3-manifested swap-pair plan over one record
//! kind. Recipe: grade the artifact on the mutated corpus AND the
//! CONTROL (same artifact, UNmutated corpus — must stay identical);
//! the deltas are the measurement
//! (`fact_swap_plan_deterministic_and_pinned`,
//! `fact_swap_plan_refusals`).
//!
//! # Pre-registered, NOT RUN (consumer-gated)
//!
//! riir-train Issue 686 carries the trigger; until it fires this module
//! ships primitives only and makes NO quality claim:
//!
//! - fusion quality: closed-form construction vs GD-trained — graded
//!   in-character within 5 points of trained on a replicated eval AND
//!   persona-corpus val loss within +0.02 of trained;
//! - census vs trained-L0 ranking agreement;
//! - the attribution-vs-causal gap (detection ≠ prediction,
//!   Research 267).
//!
//! owl weights/datasets are CC BY-NC 4.0 — local comparison only,
//! NEVER a shipped artifact.
//!
//! # Posture
//!
//! Opt-in, KEYLESS under the existing `formats_weights` umbrella (the
//! Issue-1222 cap-era ride — katgpt-core sits at 296/300 of the
//! crates.io feature cap; the `lepski_cascade`/`epiplexity_gain`
//! precedent). DORMANT: no wired consumer; pick up only when the
//! Issue-996 trigger fires. Storage adoption (riir-train `lora.bin` /
//! `resolve_lora_pair`) is the consumer's follow-up.
//!
//! ```text
//! cargo test -p katgpt-core --features formats_weights --lib surgical_read_patch
//! cargo test -p katgpt-core --features formats_weights --test issue996_surgical_read_patch_gates
//! ```

use std::fmt;

use crate::simd::dot_f32_ordered;
use crate::types::{LoraAdapter, WeightEpoch};

/// The canonical + artifact encoding format version. Bumped ONLY by a
/// breaking encoding change; an unknown version refuses loud
/// ([`SurgicalPatchError::Encoding`]) — never a best-effort parse.
pub const SURGICAL_PATCH_FORMAT_VERSION: u32 = 1;

/// Canonical-form magic (`KSRPT` — katgpt surgical read patch).
const MAGIC: [u8; 5] = *b"KSRPT";

/// Artifact-envelope magic (`SRPA` — surgical read-patch artifact).
const ARTIFACT_MAGIC: [u8; 4] = *b"SRPA";

// ── The canonical byte codec ─────────────────────────────────────────────
//
// Self-contained (the `graft_artifact` shape, never graft types). The
// `Dec` bounds check is the overflow-safe form of `pos + n > len`: a
// hostile u64-sourced length must refuse, never wrap.

struct Enc(Vec<u8>);

impl Enc {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
    fn u8v(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    /// f32 as its raw bit pattern (fixed-width LE) — the canonical
    /// weight pin (Issue-940; `-0.0` and `0.0` are different content).
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
}

struct Dec<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Dec<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], SurgicalPatchError> {
        let end = match self.pos.checked_add(n) {
            Some(end) if end <= self.b.len() => end,
            _ => return Err(SurgicalPatchError::Encoding("truncated canonical patch")),
        };
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8v(&mut self) -> Result<u8, SurgicalPatchError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, SurgicalPatchError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, SurgicalPatchError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f32(&mut self) -> Result<f32, SurgicalPatchError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn done(&self) -> Result<(), SurgicalPatchError> {
        if self.pos != self.b.len() {
            return Err(SurgicalPatchError::Encoding(
                "trailing bytes after canonical patch",
            ));
        }
        Ok(())
    }
}

/// Lowercase-hex rendering of a 32-byte digest (logs, pins, wire).
fn hex32(digest: &[u8; 32]) -> String {
    blake3::Hash::from(*digest).to_hex().to_string()
}

// ── The error vocabulary ─────────────────────────────────────────────────

/// Every refusal the surgical read-patch lane can raise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurgicalPatchError {
    /// Canonical/artifact byte-level malformation (magic, version,
    /// truncation, trailing bytes, ordering).
    Encoding(&'static str),
    /// Structural validation on a well-formed encoding (duplicate ids,
    /// descending neurons, neither projection, degenerate dims).
    Validation(&'static str),
    /// The canonical bytes do not hash to the stored digest — the
    /// artifact was tampered with or corrupted after mint.
    DigestMismatch {
        /// The digest the artifact carries.
        stored: [u8; 32],
        /// BLAKE3 over the bytes as they stand now.
        computed: [u8; 32],
    },
    /// No layer carries this `layer_id`.
    UnknownLayer {
        /// The id nothing resolved to.
        layer_id: u32,
    },
    /// The layer exists but does not patch the requested projection.
    ProjectionMissing {
        /// The layer id.
        layer_id: u32,
        /// The projection this layer leaves untouched.
        which: Projection,
    },
    /// A buffer's length disagrees with the (rows × in_dim) shape.
    DimMismatch {
        /// The length the operation required.
        expected: usize,
        /// The length the caller handed over.
        got: usize,
    },
    /// A selected output-row index does not exist in the target
    /// weight matrix. The merge is atomic — refused before ANY write.
    RowOutOfRange {
        /// The layer that carries the row.
        layer_id: u32,
        /// The out-of-range row index.
        row: u32,
        /// The number of rows the target actually has.
        rows: usize,
    },
    /// A non-finite value where only finite weights may live.
    NonFinite(&'static str),
    /// The two corpora are identical — the closed-form construction
    /// has no direction to extract (the typed honest abstain).
    NoSeparation,
    /// Same epoch, different content — a lineage fork, refused.
    Fork {
        /// The epoch the slot currently mounts.
        mounted_epoch: u64,
        /// The incoming artifact's content digest.
        incoming_digest: [u8; 32],
    },
    /// Incoming epoch is LOWER than the mounted epoch — a downgrade,
    /// refused.
    Downgrade {
        /// The epoch the slot currently mounts.
        mounted_epoch: u64,
        /// The refused incoming epoch.
        incoming_epoch: u64,
    },
}

impl fmt::Display for SurgicalPatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoding(m) | Self::Validation(m) | Self::NonFinite(m) => {
                write!(f, "surgical-patch: {m}")
            }
            Self::DigestMismatch { stored, computed } => write!(
                f,
                "surgical-patch: digest mismatch (stored {}, computed {})",
                hex32(stored),
                hex32(computed)
            ),
            Self::UnknownLayer { layer_id } => {
                write!(f, "surgical-patch: unknown layer {layer_id}")
            }
            Self::ProjectionMissing { layer_id, which } => write!(
                f,
                "surgical-patch: layer {layer_id} has no {} patch",
                which.as_str()
            ),
            Self::DimMismatch { expected, got } => write!(
                f,
                "surgical-patch: dimension mismatch (expected {expected}, got {got})"
            ),
            Self::RowOutOfRange {
                layer_id,
                row,
                rows,
            } => write!(
                f,
                "surgical-patch: layer {layer_id} row {row} out of range (rows {rows})"
            ),
            Self::NoSeparation => write!(
                f,
                "surgical-patch: no separation between corpora (identical inputs)"
            ),
            Self::Fork {
                mounted_epoch,
                incoming_digest,
            } => write!(
                f,
                "surgical-patch: lineage fork at mounted epoch {mounted_epoch} (incoming digest {})",
                hex32(incoming_digest)
            ),
            Self::Downgrade {
                mounted_epoch,
                incoming_epoch,
            } => write!(
                f,
                "surgical-patch: epoch downgrade refused (mounted {mounted_epoch}, incoming {incoming_epoch})"
            ),
        }
    }
}

impl std::error::Error for SurgicalPatchError {}

/// Which projection a layer patch drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Projection {
    /// The gate projection (gate-matrix output rows).
    Gate,
    /// The up projection (up-matrix output rows).
    Up,
}

impl Projection {
    /// The canonical spelling (logs, error messages).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
        }
    }
}

/// One layer's surgical patch: selected output rows, the shared read
/// directions, and the per-neuron coefficient rows.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerPatch {
    /// The transformer layer this patch drives.
    pub layer_id: u32,
    /// Selected MLP-neuron (output-row) indices — STRICTLY ascending,
    /// unique; this order IS the merge fold order.
    pub neurons: Vec<u32>,
    /// Shared read directions, row-major `[rank × in_dim]` — ONE set
    /// per layer, shared by every projection this layer patches (the
    /// owl factorization).
    pub a: Vec<f32>,
    /// Gate-projection coefficient rows `[n × rank]`; `None` = gate
    /// untouched.
    pub gate: Option<Vec<f32>>,
    /// Up-projection coefficient rows `[n × rank]`; `None` = up
    /// untouched.
    pub up: Option<Vec<f32>>,
}

/// The row-sparse surgical read-patch: the validated whole (every
/// layer consistent with one shared rank, the write-frozen merge
/// contract). Construct through [`SurgicalReadPatch::from_parts`] —
/// the only validated entry.
#[derive(Debug, Clone, PartialEq)]
pub struct SurgicalReadPatch {
    /// The input dimension every direction row is sized against.
    pub in_dim: usize,
    /// Multiplicative patch scale applied at merge/expand (1.0 =
    /// as-frozen).
    pub scale: f32,
    /// Ascending, unique layer_id — canonical order.
    pub layers: Vec<LayerPatch>,
}

impl SurgicalReadPatch {
    /// Build from parts, validating the whole. Layers are SORTED into
    /// ascending `layer_id` (canonical order — a reversed input is a
    /// no-op sort, a duplicate id refuses). `rank` is derived from the
    /// first layer's direction block and must hold for every layer
    /// (a cross-layer rank mismatch surfaces as a
    /// [`SurgicalPatchError::DimMismatch`]).
    ///
    /// # Errors
    /// [`SurgicalPatchError::Validation`] on degenerate dims, duplicate
    /// layer ids, an off-lattice direction block, descending neurons
    /// or a neither-projection layer;
    /// [`SurgicalPatchError::DimMismatch`] on shape mismatches;
    /// [`SurgicalPatchError::NonFinite`] on non-finite weights or
    /// scale.
    pub fn from_parts(
        in_dim: usize,
        scale: f32,
        mut layers: Vec<LayerPatch>,
    ) -> Result<Self, SurgicalPatchError> {
        if in_dim == 0 || in_dim > u32::MAX as usize {
            return Err(SurgicalPatchError::Validation(
                "surgical patch: in_dim must be in 1..=u32::MAX",
            ));
        }
        if !scale.is_finite() {
            return Err(SurgicalPatchError::NonFinite("scale"));
        }
        if layers.is_empty() || layers.len() > u32::MAX as usize {
            return Err(SurgicalPatchError::Validation(
                "surgical patch: need at least one layer, u32-countable",
            ));
        }
        // Canonical order: ascending layer_id.
        layers.sort_by_key(|layer| layer.layer_id);
        if layers
            .windows(2)
            .any(|pair| pair[0].layer_id >= pair[1].layer_id)
        {
            return Err(SurgicalPatchError::Validation(
                "surgical patch: duplicate layer_id",
            ));
        }
        let a0 = layers[0].a.len();
        if a0 < in_dim || !a0.is_multiple_of(in_dim) {
            return Err(SurgicalPatchError::Validation(
                "surgical patch: direction block is not rank × in_dim",
            ));
        }
        let rank = a0 / in_dim;
        if rank > u32::MAX as usize {
            return Err(SurgicalPatchError::Validation(
                "surgical patch: rank exceeds the canonical u32 range",
            ));
        }
        Self::validate_layers(&layers, in_dim, rank, scale)?;
        Ok(Self {
            in_dim,
            scale,
            layers,
        })
    }

    /// Per-layer structural validation (shapes, finiteness, at least
    /// one projection). Also re-checks `scale` so the from_canonical
    /// route cannot skip it.
    fn validate_layers(
        layers: &[LayerPatch],
        in_dim: usize,
        rank: usize,
        scale: f32,
    ) -> Result<(), SurgicalPatchError> {
        if !scale.is_finite() {
            return Err(SurgicalPatchError::NonFinite("scale"));
        }
        for layer in layers {
            if layer.neurons.is_empty() || layer.neurons.len() > u32::MAX as usize {
                return Err(SurgicalPatchError::Validation(
                    "surgical patch: layer needs a non-empty, u32-countable neuron list",
                ));
            }
            if layer.neurons.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(SurgicalPatchError::Validation(
                    "surgical patch: neurons must be strictly ascending",
                ));
            }
            if layer.a.len() != rank * in_dim {
                return Err(SurgicalPatchError::DimMismatch {
                    expected: rank * in_dim,
                    got: layer.a.len(),
                });
            }
            for &v in &layer.a {
                if !v.is_finite() {
                    return Err(SurgicalPatchError::NonFinite("direction"));
                }
            }
            for (name, coeffs) in [("gate", &layer.gate), ("up", &layer.up)] {
                let Some(rows_b) = coeffs else { continue };
                if rows_b.len() != layer.neurons.len() * rank {
                    return Err(SurgicalPatchError::DimMismatch {
                        expected: layer.neurons.len() * rank,
                        got: rows_b.len(),
                    });
                }
                for &v in rows_b {
                    if !v.is_finite() {
                        return Err(SurgicalPatchError::NonFinite(name));
                    }
                }
            }
            if layer.gate.is_none() && layer.up.is_none() {
                return Err(SurgicalPatchError::Validation(
                    "surgical patch: layer patches neither gate nor up",
                ));
            }
        }
        Ok(())
    }

    /// The shared rank (direction rows ÷ in_dim — every layer carries
    /// the same rank or [`Self::from_parts`] refused).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.layers[0].a.len() / self.in_dim
    }

    /// The patch for `layer_id`, if any.
    #[must_use]
    pub fn layer(&self, layer_id: u32) -> Option<&LayerPatch> {
        self.layers.iter().find(|layer| layer.layer_id == layer_id)
    }

    /// The canonical byte encoding — layout:
    /// `MAGIC(5) | u32 version | u32 in_dim | u32 rank | f32 scale |
    /// u32 layers.len() | per layer { u32 layer_id | u32 n | u32
    /// neurons | u8 gate? | u8 up? | a f32s | [gate f32s] | [up f32s] }`.
    /// Every `f32` is its raw LE bit pattern (the canonical weight
    /// pin); the encoding is a pure function of the validated content.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut enc = Enc::new();
        enc.raw(&MAGIC);
        enc.u32(SURGICAL_PATCH_FORMAT_VERSION);
        enc.u32(self.in_dim as u32);
        enc.u32(self.rank() as u32);
        enc.f32(self.scale);
        enc.u32(self.layers.len() as u32);
        for layer in &self.layers {
            enc.u32(layer.layer_id);
            enc.u32(layer.neurons.len() as u32);
            for &neuron in &layer.neurons {
                enc.u32(neuron);
            }
            enc.u8v(u8::from(layer.gate.is_some()));
            enc.u8v(u8::from(layer.up.is_some()));
            for &v in &layer.a {
                enc.f32(v);
            }
            if let Some(gate) = &layer.gate {
                for &v in gate {
                    enc.f32(v);
                }
            }
            if let Some(up) = &layer.up {
                for &v in up {
                    enc.f32(v);
                }
            }
        }
        enc.0
    }

    /// Strict canonical decode: magic + version refuse loud; headers
    /// must be non-degenerate; layer and neuron ordering are checked
    /// ascending IN THE STREAM (non-canonical ordering refuses
    /// [`SurgicalPatchError::Encoding`]); every weight is finite-checked;
    /// a trailing byte refuses; and the result re-validates through
    /// [`Self::from_parts`] (its sort is a no-op on already-canonical
    /// input).
    ///
    /// ⚠ This alone does NOT detect body corruption (a flipped weight
    /// bit re-encodes to the same corrupted bytes) — the digest on
    /// [`SurgicalPatchArtifact`] is the corruption boundary.
    ///
    /// # Errors
    /// [`SurgicalPatchError::Encoding`] on every structural/canonicality
    /// fault, [`SurgicalPatchError::NonFinite`] on non-finite weights,
    /// plus everything [`Self::from_parts`] raises.
    pub fn from_canonical(bytes: &[u8]) -> Result<Self, SurgicalPatchError> {
        let mut dec = Dec::new(bytes);
        if dec.take(MAGIC.len())? != &MAGIC[..] {
            return Err(SurgicalPatchError::Encoding("bad canonical magic"));
        }
        let version = dec.u32()?;
        if version != SURGICAL_PATCH_FORMAT_VERSION {
            return Err(SurgicalPatchError::Encoding(
                "unsupported surgical patch canonical version (this build reads 1)",
            ));
        }
        let in_dim = dec.u32()? as usize;
        let rank = dec.u32()? as usize;
        let scale = dec.f32()?;
        if !scale.is_finite() {
            return Err(SurgicalPatchError::NonFinite("canonical scale"));
        }
        let n_layers = dec.u32()? as usize;
        if in_dim == 0 || rank == 0 || n_layers == 0 {
            return Err(SurgicalPatchError::Encoding("degenerate canonical header"));
        }
        let a_total = rank
            .checked_mul(in_dim)
            .ok_or(SurgicalPatchError::Encoding(
                "canonical direction block overflows",
            ))?;
        let mut prev_layer: Option<u32> = None;
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let layer_id = dec.u32()?;
            if prev_layer.is_some_and(|prev| layer_id <= prev) {
                return Err(SurgicalPatchError::Encoding(
                    "canonical layer order violated",
                ));
            }
            prev_layer = Some(layer_id);
            let n = dec.u32()? as usize;
            if n == 0 {
                return Err(SurgicalPatchError::Encoding("empty neuron list"));
            }
            let mut neurons = Vec::new();
            for _ in 0..n {
                let row = dec.u32()?;
                if neurons.last().is_some_and(|&prev| row <= prev) {
                    return Err(SurgicalPatchError::Encoding(
                        "canonical neuron order violated",
                    ));
                }
                neurons.push(row);
            }
            let has_gate = dec.u8v()? != 0;
            let has_up = dec.u8v()? != 0;
            if !has_gate && !has_up {
                return Err(SurgicalPatchError::Encoding(
                    "layer patches neither gate nor up",
                ));
            }
            let mut a = Vec::new();
            for _ in 0..a_total {
                let v = dec.f32()?;
                if !v.is_finite() {
                    return Err(SurgicalPatchError::NonFinite("canonical direction"));
                }
                a.push(v);
            }
            let total = n.checked_mul(rank).ok_or(SurgicalPatchError::Encoding(
                "canonical coefficient block overflows",
            ))?;
            let gate = read_coeffs(&mut dec, has_gate, total, "canonical coefficient")?;
            let up = read_coeffs(&mut dec, has_up, total, "canonical coefficient")?;
            layers.push(LayerPatch {
                layer_id,
                neurons,
                a,
                gate,
                up,
            });
        }
        dec.done()?;
        Self::from_parts(in_dim, scale, layers)
    }

    /// The stable identity of this patch's effective content —
    /// BLAKE3 over the canonical bytes under a domain-separated tag
    /// (the weight-epoch contract for cache exactness).
    #[must_use]
    pub fn weight_epoch(&self) -> WeightEpoch {
        let canonical = self.canonical_bytes();
        WeightEpoch::from_parts(b"katgpt-surgical-read-patch-v1", [canonical.as_slice()])
    }

    /// Merge this patch's `{which}` projection into `w` (row-major
    /// `[rows × in_dim]`). ATOMIC: any out-of-range neuron row refuses
    /// BEFORE ANY write. The fold order is pinned — neurons ascending ×
    /// rank ascending × in_dim ascending, plain `+=`, zero allocation,
    /// never a platform-SIMD kernel (the Issue-940 law). TERNARY
    /// PURITY: `w` is plain `f32` — never merge fp deltas into ternary
    /// rows (on a ternary-served model use [`Self::projection_to_lora`]
    /// as the runtime side-patch instead).
    ///
    /// # Errors
    /// [`SurgicalPatchError::UnknownLayer`],
    /// [`SurgicalPatchError::ProjectionMissing`],
    /// [`SurgicalPatchError::DimMismatch`] (w is not `rows × in_dim`),
    /// [`SurgicalPatchError::RowOutOfRange`]. The write-frozen law:
    /// rows the patch does not select stay BIT-IDENTICAL.
    pub fn merge_projection_into(
        &self,
        layer_id: u32,
        which: Projection,
        w: &mut [f32],
        rows: usize,
    ) -> Result<(), SurgicalPatchError> {
        let layer = self
            .layer(layer_id)
            .ok_or(SurgicalPatchError::UnknownLayer { layer_id })?;
        let coeffs = match which {
            Projection::Gate => layer.gate.as_ref(),
            Projection::Up => layer.up.as_ref(),
        }
        .ok_or(SurgicalPatchError::ProjectionMissing { layer_id, which })?;
        let expected = rows * self.in_dim;
        if w.len() != expected {
            return Err(SurgicalPatchError::DimMismatch {
                expected,
                got: w.len(),
            });
        }
        Self::merge_rows(
            &layer.neurons,
            &layer.a,
            coeffs,
            self.rank(),
            self.in_dim,
            self.scale,
            w,
            rows,
            layer_id,
        )
    }

    /// The pinned merge kernel. The validation pre-pass over ALL
    /// neurons runs before ANY write (atomic refusal); the fold is the
    /// outer product `scale * b_i @ a` scattered into the selected
    /// rows.
    #[allow(clippy::too_many_arguments)] // the pinned merge signature — every argument is one of the fold law's own axes
    fn merge_rows(
        neurons: &[u32],
        a: &[f32],
        b: &[f32],
        rank: usize,
        in_dim: usize,
        scale: f32,
        w: &mut [f32],
        rows: usize,
        layer_id: u32,
    ) -> Result<(), SurgicalPatchError> {
        for &row in neurons {
            if row as usize >= rows {
                return Err(SurgicalPatchError::RowOutOfRange {
                    layer_id,
                    row,
                    rows,
                });
            }
        }
        // Issue-940 fold: neurons ascending × rank ascending × in_dim
        // ascending, plain `+=`, zero allocation, no SIMD, no scratch.
        for (i, &row) in neurons.iter().enumerate() {
            let base = row as usize * in_dim;
            for (k, &coef_raw) in b[i * rank..(i + 1) * rank].iter().enumerate() {
                let coef = scale * coef_raw;
                for (j, &av) in a[k * in_dim..(k + 1) * in_dim].iter().enumerate() {
                    w[base + j] += coef * av;
                }
            }
        }
        Ok(())
    }

    /// Materialize the patch as a dense-b [`LoraAdapter`] so the
    /// runtime side-patch rides the EXISTING `lora_apply` path —
    /// never a new kernel (substrate-first). `b` is `out_dim × rank`
    /// with exact `+0.0` off the selected rows; `alpha` is
    /// `scale × rank`, which makes `lora_apply`'s per-row scale
    /// (`alpha / rank`) exactly this artifact's `scale`.
    ///
    /// # Errors
    /// [`SurgicalPatchError::UnknownLayer`],
    /// [`SurgicalPatchError::ProjectionMissing`],
    /// [`SurgicalPatchError::RowOutOfRange`] (any selected row ≥
    /// `out_dim`).
    pub fn projection_to_lora(
        &self,
        layer_id: u32,
        which: Projection,
        out_dim: usize,
    ) -> Result<LoraAdapter, SurgicalPatchError> {
        let layer = self
            .layer(layer_id)
            .ok_or(SurgicalPatchError::UnknownLayer { layer_id })?;
        let coeffs = match which {
            Projection::Gate => layer.gate.as_ref(),
            Projection::Up => layer.up.as_ref(),
        }
        .ok_or(SurgicalPatchError::ProjectionMissing { layer_id, which })?;
        let rank = self.rank();
        for &row in &layer.neurons {
            if row as usize >= out_dim {
                return Err(SurgicalPatchError::RowOutOfRange {
                    layer_id,
                    row,
                    rows: out_dim,
                });
            }
        }
        let mut b = vec![0.0f32; out_dim * rank];
        for (i, &row) in layer.neurons.iter().enumerate() {
            let dst = row as usize * rank;
            b[dst..dst + rank].copy_from_slice(&coeffs[i * rank..(i + 1) * rank]);
        }
        Ok(LoraAdapter {
            rank,
            in_dim: self.in_dim,
            out_dim,
            a: layer.a.clone(),
            b,
            alpha: self.scale * rank as f32,
        })
    }
}

/// Canonical coefficient-block read: absent flag → `None`; present →
/// `total` f32s, each finite-checked (the artifact lane refuses to
/// carry a NaN coefficient — malformed, not a pinned quirk).
fn read_coeffs(
    dec: &mut Dec<'_>,
    present: bool,
    total: usize,
    what: &'static str,
) -> Result<Option<Vec<f32>>, SurgicalPatchError> {
    if !present {
        return Ok(None);
    }
    let mut out = Vec::new();
    for _ in 0..total {
        let v = dec.f32()?;
        if !v.is_finite() {
            return Err(SurgicalPatchError::NonFinite(what));
        }
        out.push(v);
    }
    Ok(Some(out))
}

// ── The digest-pinned artifact envelope ──────────────────────────────

/// The freeze/thaw envelope for a surgical read patch: a versioned
/// canonical encoding pinned by a BLAKE3 digest, stamped with a
/// caller-supplied monotonic epoch. Mirrors the `GraftArtifact` laws
/// exactly — content only (the epoch is lineage, never inside the
/// digest), weights are bits, decode validates and never trusts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurgicalPatchArtifact {
    /// The encoding format version (always
    /// [`SURGICAL_PATCH_FORMAT_VERSION`] at mint; decode refuses any
    /// other value loud).
    pub format_version: u32,
    /// The lineage epoch — caller-supplied at mint (a registry counter
    /// or clock; this crate reads no wall clock). The mount law's
    /// ordering key.
    pub epoch: u64,
    /// BLAKE3 over the canonical content bytes. The corruption
    /// boundary: any post-mint tamper trips
    /// [`SurgicalPatchError::DigestMismatch`].
    pub patch_digest: [u8; 32],
    /// The canonical content encoding
    /// ([`SurgicalReadPatch::canonical_bytes`]).
    pub canonical: Vec<u8>,
}

impl SurgicalPatchArtifact {
    /// Stamp a patch into a versioned, digest-pinned artifact.
    #[must_use]
    pub fn mint(patch: &SurgicalReadPatch, epoch: u64) -> Self {
        let canonical = patch.canonical_bytes();
        let patch_digest = blake3::hash(&canonical).into();
        Self {
            format_version: SURGICAL_PATCH_FORMAT_VERSION,
            epoch,
            patch_digest,
            canonical,
        }
    }

    /// The content digest, lowercase hex (logs, issue records, wire).
    #[must_use]
    pub fn digest_hex(&self) -> String {
        hex32(&self.patch_digest)
    }

    /// Verify the stored digest against BLAKE3 over the canonical
    /// bytes — the corruption check.
    ///
    /// # Errors
    /// [`SurgicalPatchError::DigestMismatch`] when the canonical bytes
    /// do not hash to [`Self::patch_digest`].
    pub fn verify_digest(&self) -> Result<(), SurgicalPatchError> {
        let computed: [u8; 32] = blake3::hash(&self.canonical).into();
        if computed == self.patch_digest {
            Ok(())
        } else {
            Err(SurgicalPatchError::DigestMismatch {
                stored: self.patch_digest,
                computed,
            })
        }
    }

    /// Full verification + reconstruction: format version, digest,
    /// then [`SurgicalReadPatch::from_canonical`] (constructor
    /// validation + canonicality).
    ///
    /// # Errors
    /// [`SurgicalPatchError::Encoding`] on a wrong format version,
    /// [`SurgicalPatchError::DigestMismatch`] on tampered content, and
    /// every parse/canonicality error from
    /// [`SurgicalReadPatch::from_canonical`].
    pub fn decode(&self) -> Result<SurgicalReadPatch, SurgicalPatchError> {
        if self.format_version != SURGICAL_PATCH_FORMAT_VERSION {
            return Err(SurgicalPatchError::Encoding(
                "unsupported surgical patch artifact version (this build reads 1)",
            ));
        }
        self.verify_digest()?;
        SurgicalReadPatch::from_canonical(&self.canonical)
    }

    /// The wire encoding: `ARTIFACT_MAGIC(4) | u32 format_version |
    /// u64 epoch | digest(32) | u64 canonical.len() | canonical`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut enc = Enc::new();
        enc.raw(&ARTIFACT_MAGIC);
        enc.u32(self.format_version);
        enc.u64(self.epoch);
        enc.raw(&self.patch_digest);
        enc.u64(self.canonical.len() as u64);
        enc.raw(&self.canonical);
        enc.0
    }

    /// Wire decode: magic refuses loud, every field is read, a
    /// trailing byte refuses, and the digest verifies BEFORE the
    /// artifact is returned.
    ///
    /// # Errors
    /// [`SurgicalPatchError::Encoding`] on magic/truncation/trailing
    /// faults, [`SurgicalPatchError::DigestMismatch`] on tampered
    /// bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SurgicalPatchError> {
        let mut dec = Dec::new(bytes);
        if dec.take(ARTIFACT_MAGIC.len())? != &ARTIFACT_MAGIC[..] {
            return Err(SurgicalPatchError::Encoding("bad artifact magic"));
        }
        let format_version = dec.u32()?;
        let epoch = dec.u64()?;
        let mut patch_digest = [0u8; 32];
        patch_digest.copy_from_slice(dec.take(32)?);
        let len = dec.u64()? as usize;
        let canonical = dec.take(len)?.to_vec();
        dec.done()?;
        let artifact = Self {
            format_version,
            epoch,
            patch_digest,
            canonical,
        };
        artifact.verify_digest()?;
        Ok(artifact)
    }
}

// ── The monotonic mount ──────────────────────────────────────────────

/// What one [`SurgicalPatchSlot::install`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurgicalPatchInstall {
    /// A new artifact (empty slot) or a strict epoch advance took the
    /// slot; the mounted patch is the incoming one.
    Advanced,
    /// Same epoch AND same content digest as the mounted artifact —
    /// byte-equivalent content, already fully verified at its own
    /// install. The mount is kept; no re-decode.
    Idempotent,
}

/// One mounted patch with its lineage (epoch + content digest), kept
/// together so the digest can never drift from the patch it names.
#[derive(Debug, Clone)]
struct MountedPatch {
    epoch: u64,
    digest: [u8; 32],
    patch: SurgicalReadPatch,
}

/// The monotonic patch mount: the A2 hot-swap law at the seam —
///
/// - **empty slot** → install ([`SurgicalPatchInstall::Advanced`])
/// - **epoch advance** (incoming > mounted) → install
/// - **same epoch + same digest** → [`SurgicalPatchInstall::Idempotent`]
/// - **same epoch + different digest** → [`SurgicalPatchError::Fork`]
/// - **lower epoch** → [`SurgicalPatchError::Downgrade`]
///
/// Every refusal leaves the mounted patch untouched. The digest is
/// verified BEFORE any lineage comparison (a corrupt artifact reports
/// corruption, not a lineage verdict), and a full decode validates the
/// content before the slot advances — a slot never mounts unverified
/// bytes.
#[derive(Debug, Clone, Default)]
pub struct SurgicalPatchSlot {
    mounted: Option<MountedPatch>,
}

impl SurgicalPatchSlot {
    /// An empty slot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mount `artifact` under the monotonic law above.
    ///
    /// # Errors
    /// [`SurgicalPatchError::DigestMismatch`] (corrupt artifact),
    /// [`SurgicalPatchError::Fork`] / [`SurgicalPatchError::Downgrade`]
    /// (lineage refusals), and every decode error. The mounted patch
    /// is untouched by any error.
    pub fn install(
        &mut self,
        artifact: &SurgicalPatchArtifact,
    ) -> Result<SurgicalPatchInstall, SurgicalPatchError> {
        // Integrity BEFORE lineage: corruption is the stronger,
        // unambiguous verdict.
        artifact.verify_digest()?;
        if let Some(m) = &self.mounted {
            if artifact.epoch == m.epoch {
                if artifact.patch_digest == m.digest {
                    return Ok(SurgicalPatchInstall::Idempotent);
                }
                return Err(SurgicalPatchError::Fork {
                    mounted_epoch: m.epoch,
                    incoming_digest: artifact.patch_digest,
                });
            }
            if artifact.epoch < m.epoch {
                return Err(SurgicalPatchError::Downgrade {
                    mounted_epoch: m.epoch,
                    incoming_epoch: artifact.epoch,
                });
            }
        }
        let patch = artifact.decode()?;
        self.mounted = Some(MountedPatch {
            epoch: artifact.epoch,
            digest: artifact.patch_digest,
            patch,
        });
        Ok(SurgicalPatchInstall::Advanced)
    }

    /// The mounted patch, if any.
    #[must_use]
    pub fn patch(&self) -> Option<&SurgicalReadPatch> {
        self.mounted.as_ref().map(|m| &m.patch)
    }

    /// The mounted epoch, if any.
    #[must_use]
    pub fn epoch(&self) -> Option<u64> {
        self.mounted.as_ref().map(|m| m.epoch)
    }

    /// The mounted content digest, if any.
    #[must_use]
    pub fn digest(&self) -> Option<[u8; 32]> {
        self.mounted.as_ref().map(|m| m.digest)
    }
}

// ── Lane (a): the closed-form census scorer ──────────────────────────

/// Closed-form separation census: per neuron (column),
/// `|mean(persona) − mean(base)|` over row-major `[samples × n]`
/// corpora, written into `scores_out`. f64 sequential folds ascending
/// (the Issue-940 law), f32 out.
///
/// OFFLINE primitive — the internal f64 scratch allocates; the G4
/// alloc-free claim is the MERGE path only.
///
/// CONSUMER LAW: the scorer is a detector, not a predictor — always
/// run the random-ranking control beside it (same budget, shuffled
/// selection). If random ≈ census, the scorer is noise and the lane
/// closes negative.
///
/// # Errors
/// [`SurgicalPatchError::Validation`] on zero neurons or an empty
/// corpus, [`SurgicalPatchError::DimMismatch`] on a corpus length that
/// is not `samples × n`.
pub fn census_scores_into(
    persona: &[f32],
    base: &[f32],
    n_samples_p: usize,
    n_samples_b: usize,
    scores_out: &mut [f32],
) -> Result<(), SurgicalPatchError> {
    let n = scores_out.len();
    if n == 0 {
        return Err(SurgicalPatchError::Validation("census: zero neurons"));
    }
    if n_samples_p == 0 || n_samples_b == 0 {
        return Err(SurgicalPatchError::Validation("census: empty corpus"));
    }
    let expected_p = n_samples_p
        .checked_mul(n)
        .ok_or(SurgicalPatchError::Validation(
            "census: sample count overflows",
        ))?;
    if persona.len() != expected_p {
        return Err(SurgicalPatchError::DimMismatch {
            expected: expected_p,
            got: persona.len(),
        });
    }
    let expected_b = n_samples_b
        .checked_mul(n)
        .ok_or(SurgicalPatchError::Validation(
            "census: sample count overflows",
        ))?;
    if base.len() != expected_b {
        return Err(SurgicalPatchError::DimMismatch {
            expected: expected_b,
            got: base.len(),
        });
    }
    let mut sums_p = vec![0.0f64; n];
    let mut sums_b = vec![0.0f64; n];
    for chunk in persona.chunks_exact(n) {
        for (sum, &v) in sums_p.iter_mut().zip(chunk) {
            *sum += f64::from(v);
        }
    }
    for chunk in base.chunks_exact(n) {
        for (sum, &v) in sums_b.iter_mut().zip(chunk) {
            *sum += f64::from(v);
        }
    }
    for (i, out) in scores_out.iter_mut().enumerate() {
        let mean_p = sums_p[i] / n_samples_p as f64;
        let mean_b = sums_b[i] / n_samples_b as f64;
        *out = ((mean_p - mean_b) as f32).abs();
    }
    Ok(())
}

/// Top-n selection under the Issue-940 tie rule: a single ascending
/// scan per pick, strict `>` keeps the FIRST index on ties, NaN is
/// never selected. `work` (`work.len() == scores.len()`) is scratch —
/// it receives a copy of `scores` and selected entries are marked with
/// `f32::NEG_INFINITY`, the removal marker (census scores are ≥ 0, so
/// the marker cannot collide with a live score). Returns the number of
/// indices written (stops early when the scores run out; never writes
/// past `out.len()`).
///
/// # Errors
/// [`SurgicalPatchError::DimMismatch`] when `work.len() !=
/// scores.len()`.
pub fn select_top_into(
    scores: &[f32],
    work: &mut [f32],
    top_n: usize,
    out: &mut [u32],
) -> Result<usize, SurgicalPatchError> {
    if work.len() != scores.len() {
        return Err(SurgicalPatchError::DimMismatch {
            expected: scores.len(),
            got: work.len(),
        });
    }
    work.copy_from_slice(scores);
    let limit = top_n.min(out.len());
    let mut written = 0usize;
    while written < limit {
        let mut best: Option<(usize, f32)> = None;
        for (i, &v) in work.iter().enumerate() {
            if v.is_nan() || v == f32::NEG_INFINITY {
                continue;
            }
            if best.is_none_or(|(_, best_v)| v > best_v) {
                best = Some((i, v));
            }
        }
        let Some((idx, _)) = best else { break };
        out[written] = idx as u32;
        work[idx] = f32::NEG_INFINITY;
        written += 1;
    }
    Ok(written)
}

// ── Lane (b): closed-form construction (the fusion experiment's
//    construction half) ────────────────────────────────────────────────

/// Closed-form construction: derive the shared read direction
/// `a = normalized(persona-input-mean − base-input-mean)` and
/// per-neuron coefficients `b_i = dot(base row i, a)` — zero-alloc,
/// the pinned `dot_f32_ordered` fold (Issue-940). Identical corpora
/// (exact-zero norm) are the typed honest abstain
/// [`SurgicalPatchError::NoSeparation`].
///
/// PRE-REGISTERED, NOT RUN (riir-train Issue 686 carries the trigger):
/// the QUALITY claim vs GD-trained values — graded in-character
/// within 5 points of trained on a replicated eval AND persona-corpus
/// val loss within +0.02 of trained. Until that cell runs, this is a
/// construction primitive only. The attribution-vs-causal gap applies
/// (detection ≠ prediction, Research 267).
///
/// Note: `a_out` may be partially written before a late row-validation
/// refusal — treat outputs as undefined on `Err`.
///
/// # Errors
/// [`SurgicalPatchError::Validation`] on degenerate dims, descending
/// neurons or an out-of-range row;
/// [`SurgicalPatchError::DimMismatch`] on any shape mismatch;
/// [`SurgicalPatchError::NonFinite`] on a non-finite direction norm;
/// [`SurgicalPatchError::NoSeparation`] on identical corpora.
#[allow(clippy::too_many_arguments)] // the spec-pinned construction signature — each argument is one validated axis
pub fn construct_from_census_into(
    persona_input_mean: &[f32],
    base_input_mean: &[f32],
    base_rows: &[f32],
    rows: usize,
    neurons: &[u32],
    in_dim: usize,
    a_out: &mut [f32],
    b_out: &mut [f32],
) -> Result<(), SurgicalPatchError> {
    if in_dim == 0 {
        return Err(SurgicalPatchError::Validation(
            "construct: in_dim must be at least 1",
        ));
    }
    if persona_input_mean.len() != in_dim {
        return Err(SurgicalPatchError::DimMismatch {
            expected: in_dim,
            got: persona_input_mean.len(),
        });
    }
    if base_input_mean.len() != in_dim {
        return Err(SurgicalPatchError::DimMismatch {
            expected: in_dim,
            got: base_input_mean.len(),
        });
    }
    if a_out.len() != in_dim {
        return Err(SurgicalPatchError::DimMismatch {
            expected: in_dim,
            got: a_out.len(),
        });
    }
    let expected = rows
        .checked_mul(in_dim)
        .ok_or(SurgicalPatchError::Validation(
            "construct: rows × in_dim overflows",
        ))?;
    if base_rows.len() != expected {
        return Err(SurgicalPatchError::DimMismatch {
            expected,
            got: base_rows.len(),
        });
    }
    if neurons.is_empty() {
        return Err(SurgicalPatchError::Validation(
            "construct: neurons must be non-empty and strictly ascending",
        ));
    }
    if neurons.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SurgicalPatchError::Validation(
            "construct: neurons must be strictly ascending",
        ));
    }
    if b_out.len() != neurons.len() {
        return Err(SurgicalPatchError::DimMismatch {
            expected: neurons.len(),
            got: b_out.len(),
        });
    }
    for (j, out) in a_out.iter_mut().enumerate() {
        *out = persona_input_mean[j] - base_input_mean[j];
    }
    let norm_sq = dot_f32_ordered(a_out, a_out);
    if !norm_sq.is_finite() {
        return Err(SurgicalPatchError::NonFinite("construct: direction norm"));
    }
    let norm = norm_sq.sqrt();
    if norm == 0.0 {
        return Err(SurgicalPatchError::NoSeparation);
    }
    for v in a_out.iter_mut() {
        *v /= norm;
    }
    for (i, &row) in neurons.iter().enumerate() {
        let row_us = row as usize;
        if row_us >= rows {
            return Err(SurgicalPatchError::Validation(
                "construct: neuron row exceeds rows",
            ));
        }
        b_out[i] = dot_f32_ordered(&base_rows[row_us * in_dim..(row_us + 1) * in_dim], a_out);
    }
    Ok(())
}

// ── Lane (c): the fact-swap plan ─────────────────────────────────────

/// One deterministic swap pair (`left < right` always).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwapPair {
    /// The lower record index (strictly below `right`).
    pub left: u32,
    /// The higher record index.
    pub right: u32,
}

/// A seeded, deterministic, BLAKE3-manifested swap-pair plan over one
/// record kind — the measurement instrument for lane (c).
///
/// RECIPE: grade the artifact on the MUTATED corpus (records swapped
/// per this plan) AND the CONTROL (same artifact, UNmutated corpus —
/// must stay identical). The deltas are the measurement; a control
/// that moves invalidates the whole cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactSwapPlan {
    /// The record kind the plan swaps within.
    pub kind: u64,
    /// The number of records in the kind.
    pub count: u32,
    /// The plan seed (full determinism: same seed → same plan).
    pub seed: u64,
    /// The distinct swap pairs, in deterministic plan order.
    pub swaps: Vec<SwapPair>,
    /// BLAKE3 over the canonical plan encoding
    /// (`u64 kind | u32 count | u64 seed | u32 n | pairs`).
    pub plan_digest: [u8; 32],
}

/// xoshiro-free xorshift64* — a tiny, dependency-free, deterministic
/// PRNG, seeded `state = seed | 1` so a zero seed cannot collapse.
struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Lemire's debiased bounded draw with rejection — no modulo bias.
    fn below(&mut self, bound: u32) -> u32 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let x = self.next_u64();
            let m = (x as u128) * (bound as u128);
            if (m as u64) >= u64::from(threshold) {
                return (m >> 64) as u32;
            }
        }
    }
}

/// Plan `swaps` distinct swap pairs over `count` records of one
/// `kind`, deterministically from `seed` (same inputs → the same plan,
/// digest-pinned). Pairs are ordered `left < right`, drawn without
/// replacement (linear contains — plans are tiny, determinism beats a
/// hash set).
///
/// # Errors
/// [`SurgicalPatchError::Validation`] on `count < 2`, more swaps than
/// distinct pairs (`count × (count−1) / 2`), or a saturated plan (4096
/// draw attempts produced no fresh pair — reduce swaps or grow the
/// corpus).
pub fn plan_fact_swaps(
    kind: u64,
    count: u32,
    swaps: u32,
    seed: u64,
) -> Result<FactSwapPlan, SurgicalPatchError> {
    if count < 2 {
        return Err(SurgicalPatchError::Validation(
            "fact-swap: need at least 2 records in the kind",
        ));
    }
    let max_pairs = (count as u64) * ((count - 1) as u64) / 2;
    if swaps as u64 > max_pairs {
        return Err(SurgicalPatchError::Validation(
            "fact-swap: more swaps than distinct pairs in the kind",
        ));
    }
    let mut rng = Xorshift64Star::new(seed);
    let mut pairs: Vec<SwapPair> = Vec::new();
    for _ in 0..swaps {
        let mut placed = false;
        for _ in 0..4096 {
            let x = rng.below(count);
            let mut y = rng.below(count);
            while y == x {
                y = rng.below(count);
            }
            let pair = SwapPair {
                left: x.min(y),
                right: x.max(y),
            };
            if !pairs.contains(&pair) {
                pairs.push(pair);
                placed = true;
                break;
            }
        }
        if !placed {
            return Err(SurgicalPatchError::Validation(
                "fact-swap: plan saturated — reduce swaps or grow the corpus",
            ));
        }
    }
    let mut enc = Enc::new();
    enc.u64(kind);
    enc.u32(count);
    enc.u64(seed);
    enc.u32(pairs.len() as u32);
    for pair in &pairs {
        enc.u32(pair.left);
        enc.u32(pair.right);
    }
    let plan_digest = blake3::hash(&enc.0).into();
    Ok(FactSwapPlan {
        kind,
        count,
        seed,
        swaps: pairs,
        plan_digest,
    })
}

// ── The in-module gate battery lives in `surgical_read_patch/tests.rs`
// ── (split per the ~2048-line soft limit — verdict round 1, condition 1).

#[cfg(test)]
mod tests;
