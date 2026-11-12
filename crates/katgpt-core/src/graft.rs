//! `grafts` — fitted rule-circuit overlays on a frozen model (issue 988).
//!
//! The four-part sidecar primitive from Research 658 (Circuit grafts,
//! Levi Foster, 2026-11-07): **reads** (closed-form linear probes over
//! activation slices), **rules** (per-output-cell decision tables —
//! authored, or mined from solved examples by minimal-deciding-set
//! search), **writes** (computed logit bias at the sampling seam — never
//! residual-stream surgery), and **switches** (fitted sigmoid gates that
//! arm the graft per input class). Everything is fitted closed-form or
//! authored; nothing is trained — the Plan 025 deterministically
//! constructed-overlay doctrine's fitted-from-examples instance. The
//! model stays untouched behind the two callback traits
//! ([`ActivationReader`], [`LogitWriter`]); no model coupling lives
//! here.
//!
//! Honest scope: the VALIDATION domain is tasks whose answers are
//! decided by small tables over linearly-readable state (the paper's
//! arithmetic). The primitive claims the mechanism, not any task; any
//! accuracy claim runs in the T2b real-activation cell (riir-infer,
//! files its own issue there).
//!
//! Switch-label selectivity law (issue 998, measured 2026-11-12): a
//! fitted switch's label must be a STATE-READABLE property, not merely
//! in-sample correlated. The T2b `{base-wrong}` switch fit its 400
//! solved rows to AUC 1.0000 IN-SAMPLE (memorization: d = 9216 ≫ n =
//! 400 at λ = 0.1) and separated at CHANCE held-out (AUC 0.5966) —
//! while every fitted score stayed positive, so the pinned 0.5-sigmoid
//! threshold (score 0) armed ALL 600 rows and the gate contributed no
//! selectivity (the GOAT's G3(b) FAIL; `grafts` stays opt-in). Two
//! laws bind every future switch fit: (a) the label must be readable
//! from the captured state (task identity qualifies; a decode-time
//! outcome like the base argmax does not); (b) the threshold is a
//! PLACEMENT decision validated against the fitted score
//! distribution, never a default pin. See `.issues/998` for the full
//! resolution.
//!
//! Numerics laws: fit paths solve ridge normal equations through the
//! SHARED leaf solver ([`crate::linalg::ridge_solve_direct_f32`] — never
//! a second factorisation; attn-match's duplicate copy is tracked for
//! dedup in issue 995); every runtime fold is a sequential scalar
//! accumulation pinned by the Issue-940 determinism law; switch gates
//! evaluate [`crate::exact_sigmoid`]. The apply hot path is zero-alloc
//! (fixture-pinned by `tests/graft_g4_alloc.rs`).
//!
//! OPT-IN behind the `grafts` feature; byte-identical off (the module
//! compiles to nothing without the feature).

// Index loops are the numerically-fixed accumulation order (Gram
// assembly, sequential folds) — the beta_fitter precedent.
#![allow(clippy::needless_range_loop)]

use crate::linalg::ridge_solve_direct_f32;

// ── Errors ───────────────────────────────────────────────────────────────

/// Graft errors — typed, loud, never permissive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraftError {
    /// API/precondition misuse (arities, caps, empty surfaces).
    Usage(&'static str),
    /// The miner's pinned cap (k ≤ 3) found no deciding input subset for
    /// the cell — loud, never an unbounded search, never a guess.
    Undecidable {
        /// The output cell no subset decides.
        cell: u16,
    },
    /// Two entries in one rule share a key (a table cannot decide it).
    DuplicateEntry {
        /// The offending rule's output cell.
        cell: u16,
    },
    /// The canonical artifact bytes are malformed (bad magic, unsupported
    /// format version, truncation, trailing bytes, non-finite payload).
    /// [`crate::graft_artifact::GRAFT_ARTIFACT_FORMAT_VERSION`] names the
    /// version this build reads.
    Encoding(&'static str),
    /// The artifact's stored digest does not match BLAKE3 over its
    /// canonical bytes — the content was corrupted after mint.
    DigestMismatch {
        /// The digest the artifact carries.
        stored: [u8; 32],
        /// The digest computed over the canonical bytes.
        computed: [u8; 32],
    },
    /// The canonical bytes parse but are NOT in canonical form — the
    /// decoded graft re-encodes to different bytes (e.g. unsorted table
    /// or write entries). Content is recoverable only through a fresh
    /// mint; the artifact is refused.
    NonCanonical,
    /// The mount law (issue 988 T5, the freeze/thaw envelope shape):
    /// same epoch as the mounted artifact, different content digest —
    /// a fork of the lineage, refused loud. Advance past the mounted
    /// epoch instead.
    Fork {
        /// The epoch already mounted.
        mounted_epoch: u64,
        /// The incoming artifact's content digest.
        incoming_digest: [u8; 32],
    },
    /// The mount law: an epoch strictly below the mounted one — a
    /// downgrade, refused loud. The mounted graft stays untouched.
    Downgrade {
        /// The epoch already mounted.
        mounted_epoch: u64,
        /// The refused artifact's epoch.
        incoming_epoch: u64,
    },
}

impl core::fmt::Display for GraftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Usage(m) => write!(f, "graft usage: {m}"),
            Self::Undecidable { cell } => {
                write!(
                    f,
                    "graft undecidable: no subset of ≤3 inputs decides cell {cell}"
                )
            }
            Self::DuplicateEntry { cell } => {
                write!(f, "graft rule for cell {cell} has duplicate entry keys")
            }
            Self::Encoding(m) => write!(f, "graft artifact encoding: {m}"),
            Self::DigestMismatch { stored, computed } => write!(
                f,
                "graft artifact digest mismatch: stored {} computed {}",
                hex32(stored),
                hex32(computed)
            ),
            Self::NonCanonical => write!(
                f,
                "graft canonical bytes are not canonical (re-encode differs)"
            ),
            Self::Fork {
                mounted_epoch,
                incoming_digest,
            } => write!(
                f,
                "graft mount fork refused: epoch {mounted_epoch} already holds a different digest; incoming {}",
                hex32(incoming_digest)
            ),
            Self::Downgrade {
                mounted_epoch,
                incoming_epoch,
            } => write!(
                f,
                "graft mount downgrade refused: mounted epoch {mounted_epoch} > incoming epoch {incoming_epoch}"
            ),
        }
    }
}

impl std::error::Error for GraftError {}

/// Lowercase hex for a 32-byte digest (error Display + artifact logging).
#[must_use]
pub fn hex32(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ── The shared solver seam ───────────────────────────────────────────────

/// Solve the ridge least-squares problem `min ‖x·w − y‖² + λ‖w‖²` over
/// `n` samples × `d` columns (row-major `x`), returning `w` (length `d`).
///
/// Assembles the normal equations `(xᵀx + λI)·w = xᵀy` and hands them to
/// the leaf's SHARED Cholesky ridge solver ([`ridge_solve_direct_f32`])
/// — the graft lane never carries a second factorisation. The Gram
/// accumulates in a fixed sequential order (samples outer, lower
/// triangle, mirrored once) so identical inputs give bit-identical
/// weights (the Issue-940 law).
///
/// # Panics
/// Shape/precondition violations (`ridge > 0` is the linalg contract) —
/// fit paths are loud on caller bugs.
#[must_use]
fn solve_ridge(x: &[f32], n: usize, d: usize, y: &[f32], ridge: f32) -> Vec<f32> {
    assert_eq!(x.len(), n * d, "graft fit: x is n×d row-major");
    assert_eq!(y.len(), n, "graft fit: one target per sample");
    assert!(n > 0 && d > 0, "graft fit: empty fit surface");
    // NEGATED comparison by intent: a NaN λ must refuse (NaN <= 0 is
    // false, so the positive form cannot express the NaN refusal).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    {
        assert!(
            ridge > 0.0,
            "graft fit: ridge λ must be > 0 (linalg contract)"
        );
    }

    let mut gram = vec![0.0f32; d * d];
    let mut cov = vec![0.0f32; d];
    for s in 0..n {
        let row = &x[s * d..(s + 1) * d];
        let ys = y[s];
        for (acc, &r_i) in cov.iter_mut().zip(row.iter()) {
            *acc += r_i * ys;
        }
        for i in 0..d {
            let r_i = row[i];
            let gram_row = &mut gram[i * d..=i * d + i];
            for (acc, &r_j) in gram_row.iter_mut().zip(row.iter()) {
                *acc += r_i * r_j;
            }
        }
    }
    // Mirror the lower triangle → upper.
    for i in 1..d {
        for j in 0..i {
            gram[j * d + i] = gram[i * d + j];
        }
    }
    // λI on the diagonal.
    for i in 0..d {
        gram[i * d + i] += ridge;
    }

    let mut l_scratch = vec![0.0f32; d * d];
    let mut z_scratch = vec![0.0f32; d];
    let mut w_t = vec![0.0f32; d];
    ridge_solve_direct_f32(&mut w_t, &mut l_scratch, &mut z_scratch, &gram, &cov, d, 1);
    w_t
}

/// Quantize a probe score into the read's value domain: round to nearest
/// (f32 `round` — halves away from zero, pinned by test) then clamp into
/// `[0, domain−1]`.
#[inline]
#[must_use]
fn quantize(score: f32, domain: u16) -> u16 {
    let clamped = score.round().clamp(0.0, f32::from(domain - 1));
    clamped as u16
}

/// splitmix64 — a tiny deterministic mix for the scramble permutation
/// and the test fixtures (zero deps, pinned arithmetic).
#[must_use]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ── The model-agnostic seams ─────────────────────────────────────────────

/// The activation seam: a host model exposes per-layer hidden-state
/// slices. Layer indices are host-defined; a graft's segments name the
/// layers they read.
pub trait ActivationReader {
    /// The activation slice for `layer` (length must equal the segment's
    /// weight length — asserted loud on mismatch).
    fn read(&self, layer: u32) -> &[f32];
}

/// The write seam: the sampling-distribution surface. A fired graft adds
/// per-token logit biases; the model still writes its own reply.
pub trait LogitWriter {
    /// Add `bias` to `token`'s logit.
    fn add_bias(&mut self, token: u32, bias: f32);
}

// ── Reads (linear probes) ────────────────────────────────────────────────

/// One probe segment: a linear read over ONE layer's activation slice.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadSegment {
    /// The host layer this segment reads.
    pub layer: u32,
    /// Per-dimension weights (length = the layer's slice length).
    pub weights: Vec<f32>,
}

impl ReadSegment {
    /// Sequential scalar fold (the Issue-940 determinism law — never a
    /// SIMD lane kernel on a value feeding a decision).
    #[inline]
    #[must_use]
    pub fn score(&self, act: &[f32]) -> f32 {
        assert_eq!(
            act.len(),
            self.weights.len(),
            "graft read: activation slice length mismatch"
        );
        let mut s = 0.0f32;
        for (w, a) in self.weights.iter().zip(act.iter()) {
            s += w * a;
        }
        s
    }
}

/// One read: multi-layer probe segments + bias, quantized into a value
/// domain (`[0, domain−1]` u16) — the input cell a rule table keys on.
#[derive(Debug, Clone, PartialEq)]
pub struct GraftRead {
    /// The probe segments (one per layer slice).
    pub segments: Vec<ReadSegment>,
    /// The probe's bias term.
    pub bias: f32,
    /// The value domain; quantized scores live in `[0, domain−1]`.
    pub domain: u16,
}

impl GraftRead {
    /// Validate + construct.
    ///
    /// # Errors
    /// [`GraftError::Usage`] on empty segments/weights or `domain < 1`.
    pub fn new(segments: Vec<ReadSegment>, bias: f32, domain: u16) -> Result<Self, GraftError> {
        if segments.is_empty() {
            return Err(GraftError::Usage("graft read needs at least one segment"));
        }
        if segments.iter().any(|s| s.weights.is_empty()) {
            return Err(GraftError::Usage("graft read segment with empty weights"));
        }
        if domain < 1 {
            return Err(GraftError::Usage("graft read domain must be >= 1"));
        }
        Ok(Self {
            segments,
            bias,
            domain,
        })
    }

    /// The read's quantized value at `reader`.
    #[inline]
    #[must_use]
    pub fn value_at(&self, reader: &dyn ActivationReader) -> u16 {
        let mut s = self.bias;
        for seg in &self.segments {
            s += seg.score(reader.read(seg.layer));
        }
        quantize(s, self.domain)
    }
}

/// The shared fit body: validate the fit surface, build the augmented
/// samples (the ones column carries the bias when `with_bias`), solve,
/// split the weights into per-layer segments + bias. Both [`fit_read`]
/// and [`fit_switch`] delegate here — one validation home, one solver
/// call site.
///
/// # Errors
/// [`GraftError::Usage`] on shape/precondition violations or a
/// non-finite solution.
#[allow(clippy::too_many_arguments)]
fn fit_linear(
    x: &[f32],
    n: usize,
    d: usize,
    layers: &[u32],
    seg_dims: &[usize],
    targets: &[f32],
    ridge: f32,
    with_bias: bool,
    what: &'static str,
) -> Result<(Vec<ReadSegment>, f32), GraftError> {
    if x.len() != n * d {
        return Err(GraftError::Usage(if what == "fit_read" {
            "fit_read: x is n×d row-major"
        } else {
            "fit_switch: x is n×d row-major"
        }));
    }
    if n == 0 || d == 0 {
        return Err(GraftError::Usage(if what == "fit_read" {
            "fit_read: empty fit surface"
        } else {
            "fit_switch: empty fit surface"
        }));
    }
    if layers.is_empty() || layers.len() != seg_dims.len() {
        return Err(GraftError::Usage(if what == "fit_read" {
            "fit_read: one layer id per segment"
        } else {
            "fit_switch: one layer id per segment"
        }));
    }
    if seg_dims.contains(&0) {
        return Err(GraftError::Usage(if what == "fit_read" {
            "fit_read: empty segment"
        } else {
            "fit_switch: empty segment"
        }));
    }
    if seg_dims.iter().sum::<usize>() != d {
        return Err(GraftError::Usage(if what == "fit_read" {
            "fit_read: segment dims must sum to d"
        } else {
            "fit_switch: segment dims must sum to d"
        }));
    }
    // The λ precondition is a NEGATED comparison on f32 by intent: a NaN
    // λ must refuse (the linalg contract), which the negated form does
    // and `ridge <= 0.0` does NOT (NaN <= 0 is false — a NaN λ would
    // slip through to the solver).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(ridge > 0.0) {
        return Err(GraftError::Usage(if what == "fit_read" {
            "fit_read: ridge λ must be > 0"
        } else {
            "fit_switch: ridge λ must be > 0"
        }));
    }

    let d_f = d + usize::from(with_bias);
    let mut x_aug = vec![0.0f32; n * d_f];
    for s in 0..n {
        x_aug[s * d_f..s * d_f + d].copy_from_slice(&x[s * d..(s + 1) * d]);
        if with_bias {
            x_aug[s * d_f + d] = 1.0;
        }
    }
    let w = solve_ridge(&x_aug, n, d_f, targets, ridge);
    if w.iter().any(|v| !v.is_finite()) {
        return Err(GraftError::Usage(if what == "fit_read" {
            "fit_read: non-finite solution"
        } else {
            "fit_switch: non-finite solution"
        }));
    }

    let mut segments = Vec::with_capacity(layers.len());
    let mut offset = 0usize;
    for (k, &layer) in layers.iter().enumerate() {
        let dims = seg_dims[k];
        segments.push(ReadSegment {
            layer,
            weights: w[offset..offset + dims].to_vec(),
        });
        offset += dims;
    }
    let bias = if with_bias { w[d] } else { 0.0 };
    Ok((segments, bias))
}

/// Fit one read by closed-form ridge least squares: `targets[s]` are the
/// f32 probe scores to fit (quantization happens at apply time, never in
/// the fit — separating the fit from the value domain keeps recovery
/// exact on well-conditioned fixtures). `layers`/`seg_dims` describe how
/// the flat `d` columns map to runtime layer slices (one segment per
/// column group, in order); `with_bias` fits the bias term on an
/// internal ones column.
///
/// # Errors
/// [`GraftError::Usage`] on shape/precondition violations (including
/// `ridge <= 0` — the linalg contract) or a non-finite solution.
#[allow(clippy::too_many_arguments)]
pub fn fit_read(
    x: &[f32],
    n: usize,
    d: usize,
    layers: &[u32],
    seg_dims: &[usize],
    targets: &[f32],
    ridge: f32,
    domain: u16,
    with_bias: bool,
) -> Result<GraftRead, GraftError> {
    if targets.len() != n {
        return Err(GraftError::Usage("fit_read: one target per sample"));
    }
    if domain < 1 {
        return Err(GraftError::Usage("fit_read: domain must be >= 1"));
    }
    let (segments, bias) = fit_linear(
        x, n, d, layers, seg_dims, targets, ridge, with_bias, "fit_read",
    )?;
    GraftRead::new(segments, bias, domain)
}

// ── Switches (fitted gates) ──────────────────────────────────────────────

/// One fitted gate: `exact_sigmoid(linear read) ≥ threshold` arms the
/// graft. The same closed-form family as `SigmoidGateCalibrator`; the
/// gate value is [`crate::exact_sigmoid`] (the committed-value-path
/// form).
#[derive(Debug, Clone, PartialEq)]
pub struct GraftSwitch {
    /// The gate's probe segments.
    pub segments: Vec<ReadSegment>,
    /// The gate's bias term.
    pub bias: f32,
    /// Armed iff `exact_sigmoid(score) >= threshold`.
    pub threshold: f32,
}

impl GraftSwitch {
    /// Validate + construct.
    ///
    /// # Errors
    /// [`GraftError::Usage`] on empty segments/weights or a threshold
    /// outside `[0, 1]` (NaN refuses).
    pub fn new(segments: Vec<ReadSegment>, bias: f32, threshold: f32) -> Result<Self, GraftError> {
        if segments.is_empty() {
            return Err(GraftError::Usage("graft switch needs at least one segment"));
        }
        if segments.iter().any(|s| s.weights.is_empty()) {
            return Err(GraftError::Usage("graft switch segment with empty weights"));
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(GraftError::Usage(
                "graft switch threshold must be in [0, 1]",
            ));
        }
        Ok(Self {
            segments,
            bias,
            threshold,
        })
    }

    /// Is the switch armed at `reader`?
    #[inline]
    #[must_use]
    pub fn armed_at(&self, reader: &dyn ActivationReader) -> bool {
        let mut s = self.bias;
        for seg in &self.segments {
            s += seg.score(reader.read(seg.layer));
        }
        crate::exact_sigmoid(s) >= self.threshold
    }
}

/// Fit one switch by closed-form ridge least squares against `{0, 1}`
/// labels (`true` = should arm). The gate reads the fitted linear form;
/// arming compares [`crate::exact_sigmoid`] of it against `threshold`.
///
/// # Errors
/// [`GraftError::Usage`] — the same shape laws as [`fit_read`], plus a
/// label/target count mismatch, a threshold outside `[0, 1]`, or a
/// non-finite solution.
#[allow(clippy::too_many_arguments)]
pub fn fit_switch(
    x: &[f32],
    n: usize,
    d: usize,
    layers: &[u32],
    seg_dims: &[usize],
    labels: &[bool],
    ridge: f32,
    threshold: f32,
    with_bias: bool,
) -> Result<GraftSwitch, GraftError> {
    if labels.len() != n {
        return Err(GraftError::Usage("fit_switch: one label per sample"));
    }
    let targets: Vec<f32> = labels.iter().map(|&l| if l { 1.0 } else { 0.0 }).collect();
    let (segments, bias) = fit_linear(
        x,
        n,
        d,
        layers,
        seg_dims,
        &targets,
        ridge,
        with_bias,
        "fit_switch",
    )?;
    GraftSwitch::new(segments, bias, threshold)
}

// ── Rule tables ──────────────────────────────────────────────────────────

/// Pack a subset's cell values into the rule-local lookup key: `v0 |
/// v1<<16 | v2<<32`. Keys compare only WITHIN one rule (the live-slot
/// count is the rule's own `subset_len`, never the key's), so the dead
/// zero slots cannot collide.
#[inline]
#[must_use]
pub fn pack_key(vals: &[u16]) -> u64 {
    let mut k = 0u64;
    for (i, v) in vals.iter().take(3).enumerate() {
        k |= u64::from(*v) << (16 * i);
    }
    k
}

/// One output cell's decision rule: the smallest input subset that
/// decides the cell (mined or authored) + its sorted key→value entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellRule {
    /// The output cell id this rule decides.
    pub cell: u16,
    /// The deciding input cell ids (live in the first `subset_len` slots).
    pub subset: [u16; 3],
    /// Live slots in `subset` (1..=3, pinned cap).
    pub subset_len: u8,
    /// Sorted key→value entries (the table's decided facts).
    pub entries: Vec<(u64, u16)>,
}

impl CellRule {
    /// Validate + construct: sorts entries by key, refuses duplicates.
    ///
    /// # Errors
    /// [`GraftError::Usage`] on a subset outside `1..=3`;
    /// [`GraftError::DuplicateEntry`] on duplicate keys.
    pub fn new(
        cell: u16,
        subset: &[u16],
        mut entries: Vec<(u64, u16)>,
    ) -> Result<Self, GraftError> {
        if subset.is_empty() || subset.len() > 3 {
            return Err(GraftError::Usage(
                "cell rule subset must hold 1..=3 input ids",
            ));
        }
        entries.sort_by_key(|&(k, _)| k);
        for pair in entries.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(GraftError::DuplicateEntry { cell });
            }
        }
        let mut packed = [0u16; 3];
        packed[..subset.len()].copy_from_slice(subset);
        Ok(Self {
            cell,
            subset: packed,
            subset_len: subset.len() as u8,
            entries,
        })
    }

    /// The rule's decision at `input_vals`, if the key is present.
    #[inline]
    #[must_use]
    pub fn decide(&self, input_vals: &[u16]) -> Option<u16> {
        let live = self.subset_len as usize;
        let mut vals = [0u16; 3];
        for (dst, &src) in vals.iter_mut().zip(self.subset.iter().take(live)) {
            *dst = input_vals[usize::from(src)];
        }
        let key = pack_key(&vals[..live]);
        self.entries
            .binary_search_by_key(&key, |&(k, _)| k)
            .ok()
            .map(|idx| self.entries[idx].1)
    }
}

/// The decision table: one [`CellRule`] per output cell, sorted by cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraftTable {
    /// The rules, sorted by cell id.
    pub rules: Vec<CellRule>,
}

impl GraftTable {
    /// Validate + construct: sorts rules by cell, refuses duplicate cells.
    ///
    /// # Errors
    /// [`GraftError::Usage`] on an empty table or duplicate cell rules.
    pub fn from_rules(mut rules: Vec<CellRule>) -> Result<Self, GraftError> {
        if rules.is_empty() {
            return Err(GraftError::Usage("graft table with no rules"));
        }
        rules.sort_by_key(|r| r.cell);
        for pair in rules.windows(2) {
            if pair[0].cell == pair[1].cell {
                return Err(GraftError::Usage(
                    "graft table: duplicate rule for one cell",
                ));
            }
        }
        Ok(Self { rules })
    }

    /// The rule for `cell`, if present.
    #[inline]
    #[must_use]
    pub fn rule_for_cell(&self, cell: u16) -> Option<&CellRule> {
        let idx = self.rules.binary_search_by_key(&cell, |r| r.cell).ok()?;
        Some(&self.rules[idx])
    }
}

/// One solved example for the miner: the input cell values + the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraftExample {
    /// The input state (one value per input cell).
    pub input: Vec<u16>,
    /// The answer (one value per output cell).
    pub output: Vec<u16>,
}

/// Build the key→value entries a subset induces for `cell`, or `None`
/// when the subset does NOT decide the cell (two examples sharing the
/// subset's values but differing in the output).
#[must_use]
fn entries_for_subset(
    subset: &[u16],
    cell: usize,
    examples: &[GraftExample],
) -> Option<Vec<(u64, u16)>> {
    debug_assert!(!subset.is_empty() && subset.len() <= 3);
    let mut map: std::collections::HashMap<u64, u16> = std::collections::HashMap::new();
    for ex in examples {
        let mut vals = [0u16; 3];
        for (dst, &src) in vals.iter_mut().zip(subset.iter()) {
            *dst = ex.input[usize::from(src)];
        }
        let key = pack_key(&vals[..subset.len()]);
        let v = ex.output[cell];
        match map.get(&key) {
            Some(&prev) if prev != v => return None,
            Some(_) => {}
            None => {
                map.insert(key, v);
            }
        }
    }
    let mut entries: Vec<(u64, u16)> = map.into_iter().collect();
    entries.sort_unstable_by_key(|&(k, _)| k);
    Some(entries)
}

/// Mine the decision table from solved examples: per output cell, the
/// SMALLEST input subset (ascending size, then lexicographic input ids;
/// **the cap k ≤ 3 is PINNED** — an exhausted cap refuses
/// [`GraftError::Undecidable`], never an unbounded search) whose values
/// decide the cell on EVERY example. Already-found subsets are tried
/// first in insertion order — one rule serves many cells where the data
/// allows it (the paper's rule-reuse economy).
///
/// Bounds: `n_in ≤ 64` (C(64,3) subsets — the fixture-scale law), `n_out
/// ≤ 4096`, `max_subset` pinned to `1..=3`.
///
/// # Errors
/// [`GraftError::Usage`] on shape violations; [`GraftError::Undecidable`]
/// naming the first cell no subset decides.
pub fn mined_table(
    examples: &[GraftExample],
    n_in: usize,
    n_out: usize,
    max_subset: usize,
) -> Result<GraftTable, GraftError> {
    if examples.is_empty() {
        return Err(GraftError::Usage("mined_table: no examples"));
    }
    if max_subset == 0 || max_subset > 3 {
        return Err(GraftError::Usage(
            "mined_table: max_subset is pinned at 1..=3",
        ));
    }
    if n_in == 0 || n_in > 64 {
        return Err(GraftError::Usage("mined_table: n_in must be in 1..=64"));
    }
    if n_out == 0 || n_out > 4096 {
        return Err(GraftError::Usage("mined_table: n_out must be in 1..=4096"));
    }
    for ex in examples {
        if ex.input.len() != n_in || ex.output.len() != n_out {
            return Err(GraftError::Usage("mined_table: example arity mismatch"));
        }
    }
    let cap = max_subset.min(n_in);
    let mut rules: Vec<CellRule> = Vec::with_capacity(n_out);
    for cell in 0..n_out {
        // The reuse pass: an already-found subset decided this cell too?
        let mut found: Option<CellRule> = None;
        for rule in &rules {
            let live = rule.subset_len as usize;
            let subset = &rule.subset[..live];
            if let Some(entries) = entries_for_subset(subset, cell, examples) {
                found = Some(CellRule::new(cell as u16, subset, entries)?);
                break;
            }
        }
        if found.is_none() {
            'search: for k in 1..=cap {
                // Lexicographic combinations of input ids, ascending size.
                match k {
                    1 => {
                        for i in 0..n_in {
                            let subset = [i as u16];
                            if let Some(entries) = entries_for_subset(&subset, cell, examples) {
                                found = Some(CellRule::new(cell as u16, &subset, entries)?);
                                break 'search;
                            }
                        }
                    }
                    2 => {
                        for i in 0..n_in {
                            for j in (i + 1)..n_in {
                                let subset = [i as u16, j as u16];
                                if let Some(entries) = entries_for_subset(&subset, cell, examples) {
                                    found = Some(CellRule::new(cell as u16, &subset, entries)?);
                                    break 'search;
                                }
                            }
                        }
                    }
                    _ => {
                        for i in 0..n_in {
                            for j in (i + 1)..n_in {
                                for m in (j + 1)..n_in {
                                    let subset = [i as u16, j as u16, m as u16];
                                    if let Some(entries) =
                                        entries_for_subset(&subset, cell, examples)
                                    {
                                        found = Some(CellRule::new(cell as u16, &subset, entries)?);
                                        break 'search;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        match found {
            Some(rule) => rules.push(rule),
            None => return Err(GraftError::Undecidable { cell: cell as u16 }),
        }
    }
    GraftTable::from_rules(rules)
}

// ── Writes (the sampling seam) ───────────────────────────────────────────

/// One decided value's logit biases: `(token, bias)` pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteEntry {
    /// The decided cell value this entry arms.
    pub value: u16,
    /// `(token, bias)` pairs, applied in order when the value matches.
    pub biases: Vec<(u32, f32)>,
}

/// One write spec: for output cell `cell`, the per-value logit biases.
#[derive(Debug, Clone, PartialEq)]
pub struct GraftWrite {
    /// The output cell id (must exist in the graft's table).
    pub cell: u16,
    /// Entries sorted by `value` (binary-searched at apply).
    pub entries: Vec<WriteEntry>,
}

impl GraftWrite {
    /// Validate + construct: sorts entries by value, refuses duplicates
    /// and bias-less entries.
    ///
    /// # Errors
    /// [`GraftError::Usage`] on duplicates or empty bias sets.
    pub fn new(cell: u16, mut entries: Vec<WriteEntry>) -> Result<Self, GraftError> {
        if entries.iter().any(|e| e.biases.is_empty()) {
            return Err(GraftError::Usage("graft write entry with no biases"));
        }
        entries.sort_by_key(|e| e.value);
        for pair in entries.windows(2) {
            if pair[0].value == pair[1].value {
                return Err(GraftError::Usage("graft write: duplicate value entry"));
            }
        }
        Ok(Self { cell, entries })
    }
}

// ── The graft ────────────────────────────────────────────────────────────

/// What one [`Graft::apply`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraftApplied {
    /// Did the graft fire (all switches armed AND every cell decided)?
    /// `false` means the writer was never touched — the unarmed no-op
    /// law.
    pub fired: bool,
    /// How many table cells were decided (== the rule count when fired;
    /// the decided prefix when abstaining on a miss).
    pub decided: usize,
}

/// The assembled overlay: reads → table → writes, gated by switches.
#[derive(Debug, Clone, PartialEq)]
pub struct Graft {
    /// One read per INPUT cell of the rule table.
    pub reads: Vec<GraftRead>,
    /// The decision table.
    pub table: GraftTable,
    /// The write specs (the sampling seam).
    pub writes: Vec<GraftWrite>,
    /// The arming gates (ALL must arm for the graft to fire; empty =
    /// always armed — the authored, ungated posture).
    pub switches: Vec<GraftSwitch>,
    disarmed: bool,
}

impl Graft {
    /// Validate + construct.
    ///
    /// # Errors
    /// [`GraftError::Usage`] when the table references unknown input
    /// cells or a write names an unknown output cell.
    pub fn new(
        reads: Vec<GraftRead>,
        table: GraftTable,
        writes: Vec<GraftWrite>,
        switches: Vec<GraftSwitch>,
    ) -> Result<Self, GraftError> {
        if reads.is_empty() {
            return Err(GraftError::Usage("graft with no reads"));
        }
        let n_in = reads.len();
        for rule in &table.rules {
            let live = rule.subset_len as usize;
            if rule.subset[..live]
                .iter()
                .any(|&id| usize::from(id) >= n_in)
            {
                return Err(GraftError::Usage(
                    "graft table references an unknown input cell",
                ));
            }
        }
        for write in &writes {
            if table.rule_for_cell(write.cell).is_none() {
                return Err(GraftError::Usage(
                    "graft write names an unknown output cell",
                ));
            }
        }
        Ok(Self {
            reads,
            table,
            writes,
            switches,
            disarmed: false,
        })
    }

    /// The input-cell count (size `input_vals` to this).
    #[must_use]
    pub fn n_inputs(&self) -> usize {
        self.reads.len()
    }

    /// The output-cell count (size `out` to this).
    #[must_use]
    pub fn n_outputs(&self) -> usize {
        self.table.rules.len()
    }

    /// The unarmed no-op law's explicit seam: a disarmed graft's apply
    /// never touches the writer (the T3 switch-off arm's mechanism,
    /// fixture-pinned).
    pub fn disarm(&mut self) {
        self.disarmed = true;
    }

    /// The all-switches arm evaluation — one pass over the reader. The
    /// held-arm caller evaluates ONCE (at the fit-geometry prompt
    /// captures) and holds the decision for the whole decode chain:
    /// task identity belongs to the prompt, never to a decode step
    /// (issue 998 T2 pre-registration). A graft with no switches is
    /// always armed.
    #[must_use]
    pub fn switches_armed_at(&self, reader: &dyn ActivationReader) -> bool {
        self.switches.iter().all(|sw| sw.armed_at(reader))
    }

    /// Apply the graft at `reader`: all switches arm → read the input
    /// cells → decide every output cell (ALL-OR-NOTHING: one undecided
    /// cell abstains the whole graft — the model still writes its own
    /// reply) → add the fired writes' biases. Zero allocation. The
    /// switches are evaluated PER CALL here; the held-arm form that
    /// evaluates once per prompt is [`Self::apply_with_arm`].
    ///
    /// # Panics
    /// Buffer-shape violations (`input_vals`/`out` sizing) — loud caller
    /// bugs, never silent abstention.
    pub fn apply(
        &self,
        reader: &dyn ActivationReader,
        writer: &mut dyn LogitWriter,
        input_vals: &mut [u16],
        out: &mut [u16],
    ) -> GraftApplied {
        if self.disarmed {
            return GraftApplied {
                fired: false,
                decided: 0,
            };
        }
        for sw in &self.switches {
            if !sw.armed_at(reader) {
                return GraftApplied {
                    fired: false,
                    decided: 0,
                };
            }
        }
        self.apply_armed(reader, writer, input_vals, out)
    }

    /// Apply the graft under a CALLER-HELD arm decision (the per-prompt
    /// held-arm seam): `armed == true` takes the read→decide→write path
    /// switch-independently, `armed == false` is the no-op. The caller's
    /// bit NEVER bypasses [`Graft::disarm`] — a disarmed graft stays a
    /// no-op whatever the bit says (the unarmed no-op law; fixture
    /// pinned). Step-0 equivalence — `apply_with_arm(`[`Self::switches_armed_at`]`(
    /// reader), reader, …)` is byte-identical to [`Self::apply`] — is
    /// fixture-pinned too (issue 998 T2 pre-registration).
    ///
    /// # Panics
    /// The same buffer-shape violations as [`Self::apply`] (only reached
    /// when armed and not disarmed).
    pub fn apply_with_arm(
        &self,
        armed: bool,
        reader: &dyn ActivationReader,
        writer: &mut dyn LogitWriter,
        input_vals: &mut [u16],
        out: &mut [u16],
    ) -> GraftApplied {
        if self.disarmed || !armed {
            return GraftApplied {
                fired: false,
                decided: 0,
            };
        }
        self.apply_armed(reader, writer, input_vals, out)
    }

    /// The read→decide→write body, shared by [`Self::apply`] (after its
    /// own disarmed + switches checks) and [`Self::apply_with_arm`]
    /// (after the held-bit + disarmed checks). Zero allocation.
    fn apply_armed(
        &self,
        reader: &dyn ActivationReader,
        writer: &mut dyn LogitWriter,
        input_vals: &mut [u16],
        out: &mut [u16],
    ) -> GraftApplied {
        assert_eq!(
            input_vals.len(),
            self.reads.len(),
            "graft apply: input_vals must hold one slot per read"
        );
        assert_eq!(
            out.len(),
            self.table.rules.len(),
            "graft apply: out must hold one slot per table rule"
        );
        for (slot, read) in input_vals.iter_mut().zip(&self.reads) {
            *slot = read.value_at(reader);
        }
        let mut decided = 0usize;
        for (slot, rule) in out.iter_mut().zip(&self.table.rules) {
            match rule.decide(input_vals) {
                Some(v) => {
                    *slot = v;
                    decided += 1;
                }
                None => {
                    return GraftApplied {
                        fired: false,
                        decided,
                    };
                }
            }
        }
        for write in &self.writes {
            let idx = self
                .table
                .rules
                .binary_search_by_key(&write.cell, |r| r.cell)
                .expect("graft write cell validated at construction");
            let v = out[idx];
            if let Ok(i) = write.entries.binary_search_by_key(&v, |e| e.value) {
                for (token, bias) in &write.entries[i].biases {
                    writer.add_bias(*token, *bias);
                }
            }
        }
        GraftApplied {
            fired: true,
            decided,
        }
    }

    /// The T3 scramble arm's mechanism: the same weights and entries,
    /// the wiring shuffled — every rule's subset ids run through a
    /// seeded permutation of the input ids (splitmix64 Fisher–Yates;
    /// deterministic: the same seed gives the same scrambled graft).
    /// Scrambled decisions are wrong BY CONSTRUCTION, which is what makes
    /// the at/below-baseline control decisive (the T2b real cell runs
    /// the arm; this constructor is its in-repo instrument).
    #[must_use]
    pub fn scrambled(&self, seed: u64) -> Graft {
        let n = self.reads.len();
        let mut state = seed ^ 0xd1b5_49a7_34c9_5dbd;
        let mut perm: Vec<u16> = (0..n as u16).collect();
        if n >= 2 {
            for i in (1..n).rev() {
                let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
                perm.swap(i, j);
            }
        }
        let mut g = self.clone();
        for rule in &mut g.table.rules {
            for slot in rule.subset.iter_mut().take(usize::from(rule.subset_len)) {
                *slot = perm[usize::from(*slot)];
            }
        }
        g.disarmed = false;
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(input: &[u16], output: &[u16]) -> GraftExample {
        GraftExample {
            input: input.to_vec(),
            output: output.to_vec(),
        }
    }

    #[test]
    fn fit_read_recovers_planted_weights_and_bias() {
        let d = 8usize;
        let n = 256usize;
        let mut state = 0x000A_11CE_5EEDu64;
        let planted = [0.5f32, -1.0, 2.0, 0.25, 1.5, -0.5, 1.0, 0.75];
        let planted_bias = 3.0f32;
        let mut x = vec![0.0f32; n * d];
        let mut targets = vec![0.0f32; n];
        for s in 0..n {
            for j in 0..d {
                x[s * d + j] = 0.5 + (splitmix64(&mut state) % 1000) as f32 / 1000.0;
            }
            // The planted score in the same sequential fold order the read uses.
            let mut score = planted_bias;
            for j in 0..d {
                score += planted[j] * x[s * d + j];
            }
            targets[s] = score;
        }
        let read = fit_read(&x, n, d, &[0, 1], &[4, 4], &targets, 1e-6, 10, true).expect("fit");
        let mut got: Vec<f32> = read
            .segments
            .iter()
            .flat_map(|s| s.weights.iter().copied())
            .collect();
        got.push(read.bias);
        let mut planted_all = planted.to_vec();
        planted_all.push(planted_bias);
        for (g, want) in got.iter().zip(&planted_all) {
            assert!((g - want).abs() < 5e-3, "recovered {g} vs planted {want}");
        }
    }

    #[test]
    fn fit_read_without_bias_keeps_the_bias_zero() {
        let d = 4usize;
        let n = 64usize;
        let mut state = 0xB0B_u64;
        let planted = [1.0f32, -2.0, 0.5, 1.0];
        let mut x = vec![0.0f32; n * d];
        let mut targets = vec![0.0f32; n];
        for s in 0..n {
            for j in 0..d {
                x[s * d + j] = 0.5 + (splitmix64(&mut state) % 1000) as f32 / 1000.0;
            }
            let mut score = 0.0f32;
            for j in 0..d {
                score += planted[j] * x[s * d + j];
            }
            targets[s] = score;
        }
        let read = fit_read(&x, n, d, &[7], &[4], &targets, 1e-6, 8, false).expect("fit");
        assert_eq!(read.bias, 0.0);
        for (g, want) in read.segments[0].weights.iter().zip(&planted) {
            assert!((g - want).abs() < 5e-3, "recovered {g} vs planted {want}");
        }
    }

    #[test]
    fn fit_switch_separates_labels_and_pins_fixture_digest() {
        // Deterministic fixture (splitmix64) — BLAKE3-pinned so any
        // generator drift shifts the digest and reds this test.
        const SWITCH_FIXTURE_DIGEST: &str =
            "540313aae7b6490420085bf1a78be1e13c24c887786a5d00cf039f7b093c803d";
        let d = 4usize;
        let n = 64usize;
        let mut state = 0x5EED_5EED_CAFE_F00Du64;
        let mut x = vec![0.0f32; n * d];
        let mut labels = vec![false; n];
        for (s, slot) in labels.iter_mut().enumerate() {
            x[s * d] = if s % 2 == 0 { 8.0 } else { -8.0 };
            for j in 1..d {
                x[s * d + j] = ((splitmix64(&mut state) % 2000) as f32 / 1000.0) - 1.0;
            }
            *slot = s % 2 == 0;
        }
        let mut bytes = Vec::with_capacity(x.len() * 4);
        for v in &x {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let digest = blake3::hash(&bytes).to_hex();
        assert_eq!(
            digest.as_str(),
            SWITCH_FIXTURE_DIGEST,
            "switch fixture digest drifted (got {digest}) — regenerate the pin ONLY with the documented generator change"
        );
        let sw = fit_switch(&x, n, d, &[3], &[4], &labels, 1e-4, 0.5, true).expect("fit");
        for (s, &label) in labels.iter().enumerate() {
            let row = &x[s * d..(s + 1) * d];
            let score = sw.bias
                + sw.segments[0]
                    .weights
                    .iter()
                    .zip(row.iter())
                    .map(|(w, a)| w * a)
                    .sum::<f32>();
            assert_eq!(crate::exact_sigmoid(score) >= 0.5, label, "sample {s}");
        }
    }

    #[test]
    fn solve_ridge_survives_a_rank_deficient_gram() {
        // Column 1 duplicates column 0 — the unregularized Gram is
        // singular; the ridge diagonal keeps the solve finite.
        let n = 16usize;
        let d = 3usize;
        let mut x = vec![0.0f32; n * d];
        for s in 0..n {
            x[s * d] = 0.5 + (s as f32) * 0.01;
            x[s * d + 1] = x[s * d]; // duplicate
            x[s * d + 2] = 1.0 - (s as f32) * 0.01;
        }
        let y: Vec<f32> = (0..n).map(|s| 2.0 * x[s * d] + x[s * d + 2]).collect();
        let w = solve_ridge(&x, n, d, &y, 1e-4);
        assert!(w.iter().all(|v| v.is_finite()));
        for s in 0..n {
            let pred: f32 = (0..d).map(|j| w[j] * x[s * d + j]).sum();
            assert!((pred - y[s]).abs() < 1e-2, "row {s}: {pred} vs {}", y[s]);
        }
    }

    #[test]
    fn quantize_rounds_half_away_from_zero_and_clamps() {
        assert_eq!(quantize(2.5, 5), 3, "half away from zero");
        assert_eq!(quantize(2.4, 5), 2);
        assert_eq!(quantize(-0.5, 5), 0, "rounds to -1, clamps to 0");
        assert_eq!(quantize(9.7, 5), 4, "clamps to domain−1");
        assert_eq!(quantize(0.4, 1), 0, "single-value domain");
    }

    #[test]
    fn mined_table_finds_the_minimal_deciding_subset() {
        // out = f(in1, in3) only — every smaller or wrong subset fails.
        let mut examples = Vec::new();
        for a in 0..2u16 {
            for b in 0..3u16 {
                for c in 0..2u16 {
                    for dd in 0..2u16 {
                        examples.push(ex(&[a, b, c, dd], &[u16::from(b == 1) * 10 + dd]));
                    }
                }
            }
        }
        let table = mined_table(&examples, 4, 1, 3).expect("mine");
        let r = &table.rules[0];
        assert_eq!(r.subset_len, 2);
        assert_eq!(&r.subset[..2], &[1, 3]);
    }

    #[test]
    fn mined_table_reuses_a_subset_across_cells() {
        let mut examples = Vec::new();
        for a in 0..6u16 {
            examples.push(ex(&[a, u16::from(a > 2)], &[a, a]));
        }
        let table = mined_table(&examples, 2, 2, 3).expect("mine");
        assert_eq!(table.rules.len(), 2);
        for r in &table.rules {
            assert_eq!(r.subset_len, 1);
            assert_eq!(r.subset[0], 0, "both cells decided by input cell 0");
        }
    }

    #[test]
    fn mined_table_refuses_undecidable_loud() {
        // out = in0 + in1 — no size-1 subset decides (the other input
        // varies the sum).
        let mut examples = Vec::new();
        for a in 0..3u16 {
            for b in 0..3u16 {
                examples.push(ex(&[a, b], &[a + b]));
            }
        }
        assert_eq!(
            mined_table(&examples, 2, 1, 1),
            Err(GraftError::Undecidable { cell: 0 })
        );
    }

    #[test]
    fn mined_table_subset_cap_is_pinned_at_three() {
        let examples = vec![ex(&[0], &[0])];
        assert_eq!(
            mined_table(&examples, 1, 1, 4),
            Err(GraftError::Usage(
                "mined_table: max_subset is pinned at 1..=3"
            ))
        );
        assert_eq!(
            mined_table(&examples, 1, 1, 0),
            Err(GraftError::Usage(
                "mined_table: max_subset is pinned at 1..=3"
            ))
        );
    }

    #[test]
    fn cell_rule_refuses_duplicate_entry_keys() {
        assert_eq!(
            CellRule::new(0, &[0], vec![(7, 1), (7, 2)]),
            Err(GraftError::DuplicateEntry { cell: 0 })
        );
    }

    #[test]
    fn graft_table_refuses_duplicate_cell_rules() {
        let r0 = CellRule::new(0, &[0], vec![(1, 5)]).unwrap();
        let r1 = CellRule::new(0, &[0], vec![(2, 6)]).unwrap();
        assert!(GraftTable::from_rules(vec![r0, r1]).is_err());
    }

    #[test]
    fn mined_table_validates_shapes() {
        let examples = vec![ex(&[0], &[0])];
        assert!(mined_table(&examples, 2, 1, 3).is_err(), "arity mismatch");
        assert!(mined_table(&[], 1, 1, 3).is_err(), "no examples");
        assert!(mined_table(&examples, 1, 0, 3).is_err(), "no outputs");
    }

    // ── The end-to-end fixture (T2a) ─────────────────────────────────

    /// The synthetic activations CARRY the digit values at known dims
    /// (readability is true by construction — this fixture validates
    /// TABLE/SWITCH/WIRING mechanics, never accuracy; the T2b
    /// real-activation cell hosts any accuracy claim).
    struct FixtureReader {
        layer0: Vec<f32>,
        layer2: Vec<f32>,
    }

    impl ActivationReader for FixtureReader {
        fn read(&self, layer: u32) -> &[f32] {
            match layer {
                0 => &self.layer0,
                2 => &self.layer2,
                _ => panic!("fixture reader: unknown layer {layer}"),
            }
        }
    }

    struct VecWriter(Vec<(u32, f32)>);

    impl LogitWriter for VecWriter {
        fn add_bias(&mut self, token: u32, bias: f32) {
            self.0.push((token, bias));
        }
    }

    fn e2e_graft() -> Result<(Graft, Vec<GraftExample>), GraftError> {
        // reads: in0 = act0[0], in1 = act0[1]; domains 5.
        let read0 = GraftRead::new(
            vec![ReadSegment {
                layer: 0,
                weights: vec![1.0, 0.0, 0.0, 0.0],
            }],
            0.0,
            5,
        )?;
        let read1 = GraftRead::new(
            vec![ReadSegment {
                layer: 0,
                weights: vec![0.0, 1.0, 0.0, 0.0],
            }],
            0.0,
            5,
        )?;
        let reads = vec![read0, read1];
        // examples: all (in0, in1) ∈ 5×5; out = [(in0+in1)%5, in0, in1].
        let mut examples = Vec::with_capacity(25);
        for a in 0..5u16 {
            for b in 0..5u16 {
                examples.push(ex(&[a, b], &[(a + b) % 5, a, b]));
            }
        }
        let table = mined_table(&examples, 2, 3, 3)?;
        // switch: fitted over layer-2 slices (the answer-position flag).
        let n = 8usize;
        let mut x = vec![0.0f32; n];
        let mut labels = vec![false; n];
        for (s, slot) in labels.iter_mut().enumerate() {
            x[s] = if s % 2 == 0 { 8.0 } else { -8.0 };
            *slot = s % 2 == 0;
        }
        let sw = fit_switch(&x, n, 1, &[2], &[1], &labels, 1e-4, 0.5, true)?;
        // writes: cell0 → tokens 100+v, cell1 → 200+v, cell2 → 300+v.
        let (mut w0, mut w1, mut w2) = (Vec::new(), Vec::new(), Vec::new());
        for v in 0..5u16 {
            w0.push(WriteEntry {
                value: v,
                biases: vec![(100 + u32::from(v), 1.0)],
            });
            w1.push(WriteEntry {
                value: v,
                biases: vec![(200 + u32::from(v), 1.0)],
            });
            w2.push(WriteEntry {
                value: v,
                biases: vec![(300 + u32::from(v), 1.0)],
            });
        }
        let writes = vec![
            GraftWrite::new(0, w0)?,
            GraftWrite::new(1, w1)?,
            GraftWrite::new(2, w2)?,
        ];
        let graft = Graft::new(reads, table, writes, vec![sw])?;
        Ok((graft, examples))
    }

    #[test]
    fn apply_fires_end_to_end_and_writes_the_computed_answer() {
        let (graft, _examples) = e2e_graft().expect("graft");
        // table wiring: rule0 subset {0,1}, rule1 {0}, rule2 {1}.
        assert_eq!(graft.table.rules[0].subset_len, 2);
        assert_eq!(&graft.table.rules[0].subset[..2], &[0, 1]);
        let reader = FixtureReader {
            layer0: vec![2.0, 4.0, 0.0, 0.0],
            layer2: vec![8.0],
        };
        let mut input_vals = [0u16; 2];
        let mut out = [0u16; 3];
        let mut writer = VecWriter(Vec::new());
        let applied = graft.apply(&reader, &mut writer, &mut input_vals, &mut out);
        assert!(applied.fired);
        assert_eq!(applied.decided, 3);
        assert_eq!(input_vals, [2, 4]);
        assert_eq!(out, [1, 2, 4], "(2+4)%5=1");
        // Per-cell writes: only the DECIDED value's entry fires per spec —
        // the graft pushes toward the computed answer, never every
        // possible one.
        assert_eq!(writer.0, vec![(101, 1.0), (202, 1.0), (304, 1.0)]);
    }

    #[test]
    fn apply_abstains_on_an_undecided_cell_and_never_touches_the_writer() {
        let (mut graft, _examples) = e2e_graft().expect("graft");
        // Drop one decided fact from rule 0 (the (2,4) row).
        let key = pack_key(&[2, 4]);
        let r0 = &mut graft.table.rules[0];
        let pos = r0
            .entries
            .binary_search_by_key(&key, |&(k, _)| k)
            .expect("row present");
        r0.entries.remove(pos);
        let reader = FixtureReader {
            layer0: vec![2.0, 4.0, 0.0, 0.0],
            layer2: vec![8.0],
        };
        let mut input_vals = [0u16; 2];
        let mut out = [0u16; 3];
        let mut writer = VecWriter(Vec::new());
        let applied = graft.apply(&reader, &mut writer, &mut input_vals, &mut out);
        assert!(!applied.fired);
        assert_eq!(applied.decided, 0);
        assert!(writer.0.is_empty(), "the abstaining graft wrote nothing");
    }

    #[test]
    fn disarmed_apply_is_the_unarmed_no_op() {
        let (mut graft, _) = e2e_graft().expect("graft");
        graft.disarm();
        let reader = FixtureReader {
            layer0: vec![2.0, 4.0, 0.0, 0.0],
            layer2: vec![8.0],
        };
        let mut input_vals = [0u16; 2];
        let mut out = [0u16; 3];
        let mut writer = VecWriter(Vec::new());
        let applied = graft.apply(&reader, &mut writer, &mut input_vals, &mut out);
        assert!(!applied.fired);
        assert!(writer.0.is_empty());
    }

    /// Issue 998 T2 pre-registration: the held-arm seam's three laws —
    /// step-0 equivalence with `apply`, the false bit is the no-op, and
    /// a caller's `armed = true` NEVER bypasses `disarm()`.
    #[test]
    fn held_arm_step0_equivalence_and_disarm_precedence() {
        let (graft, _) = e2e_graft().expect("graft");
        let reader = FixtureReader {
            layer0: vec![2.0, 4.0, 0.0, 0.0],
            layer2: vec![8.0],
        };
        let run = |g: &Graft, armed: bool| {
            let mut input_vals = [0u16; 2];
            let mut out = [0u16; 3];
            let mut writer = VecWriter(Vec::new());
            let applied = g.apply_with_arm(armed, &reader, &mut writer, &mut input_vals, &mut out);
            (applied, writer.0)
        };

        // Step-0 equivalence: the held bit computed from the switches
        // reproduces `apply` byte-identically (same reader, same answer).
        let armed = graft.switches_armed_at(&reader);
        let (a_applied, a_writes) = run(&graft, armed);
        let mut input_vals = [0u16; 2];
        let mut out = [0u16; 3];
        let mut writer = VecWriter(Vec::new());
        let d_applied = graft.apply(&reader, &mut writer, &mut input_vals, &mut out);
        assert_eq!(a_applied.fired, d_applied.fired);
        assert_eq!(a_applied.decided, d_applied.decided);
        assert_eq!(a_writes, writer.0, "step-0 equivalence: same writes");

        // The held false bit is the no-op even though the switches arm.
        assert!(armed, "fixture switch must arm at the hot reader");
        let (f_applied, f_writes) = run(&graft, false);
        assert!(!f_applied.fired);
        assert!(f_writes.is_empty());

        // The caller's true bit NEVER bypasses disarm() — the unarmed
        // no-op law wins over the held bit.
        let mut cold = graft.clone();
        cold.disarm(); // switches still evaluate; disarm is operational state
        let (c_applied, c_writes) = run(&cold, true);
        assert!(!c_applied.fired);
        assert!(c_writes.is_empty());
    }

    #[test]
    fn a_switch_that_never_arms_abstains_the_whole_graft() {
        let (graft, _) = e2e_graft().expect("graft");
        let mut switches = graft.switches.clone();
        switches[0].bias = -50.0; // exact_sigmoid(≈−50) ≈ 0 < 0.5
        let cold = Graft::new(
            graft.reads.clone(),
            graft.table.clone(),
            graft.writes.clone(),
            switches,
        )
        .expect("graft");
        let reader = FixtureReader {
            layer0: vec![2.0, 4.0, 0.0, 0.0],
            layer2: vec![8.0],
        };
        let mut input_vals = [0u16; 2];
        let mut out = [0u16; 3];
        let mut writer = VecWriter(Vec::new());
        assert!(
            !cold
                .apply(&reader, &mut writer, &mut input_vals, &mut out)
                .fired
        );
        assert!(writer.0.is_empty());
    }

    #[test]
    fn scrambled_wiring_is_deterministic_and_differs() {
        // A 4-input fixture (identity-permutation probability 1/24 — the
        // assertion is stable for the pinned seed).
        let mut reads = Vec::new();
        for k in 0..4usize {
            let mut w = vec![0.0f32; 4];
            w[k] = 1.0;
            reads.push(
                GraftRead::new(
                    vec![ReadSegment {
                        layer: 0,
                        weights: w,
                    }],
                    0.0,
                    5,
                )
                .unwrap(),
            );
        }
        let mut examples = Vec::new();
        for a in 0..2u16 {
            for b in 0..2u16 {
                for c in 0..2u16 {
                    for dd in 0..2u16 {
                        examples.push(ex(&[a, b, c, dd], &[u16::from(b == 1) * 10 + dd]));
                    }
                }
            }
        }
        let table = mined_table(&examples, 4, 1, 3).expect("mine");
        let graft = Graft::new(reads, table, Vec::new(), Vec::new()).expect("graft");
        let original_subset = graft.table.rules[0].subset;
        let scrambled = graft.scrambled(7);
        let again = graft.scrambled(7);
        assert_eq!(scrambled.table, again.table, "same seed → same scramble");
        assert_ne!(
            scrambled.table.rules[0].subset, original_subset,
            "the wiring moved"
        );
        // The scrambled wiring decides differently on a fixed probe input.
        let reader = FixtureReader {
            layer0: vec![0.0, 1.0, 1.0, 0.0],
            layer2: Vec::new(),
        };
        let mut input_vals = [0u16; 4];
        let mut out_o = [0u16; 1];
        let mut out_s = [0u16; 1];
        let mut w1 = VecWriter(Vec::new());
        let mut w2 = VecWriter(Vec::new());
        let a1 = graft.apply(&reader, &mut w1, &mut input_vals, &mut out_o);
        let a2 = scrambled.apply(&reader, &mut w2, &mut input_vals, &mut out_s);
        assert!(a1.fired && a2.fired);
        assert_ne!(out_o, out_s, "the scrambled wiring decides differently");
    }

    #[test]
    fn apply_replays_bit_identically() {
        let (graft, _) = e2e_graft().expect("graft");
        let reader = FixtureReader {
            layer0: vec![1.0, 3.0, 0.0, 0.0],
            layer2: vec![8.0],
        };
        let mut iv1 = [0u16; 2];
        let mut o1 = [0u16; 3];
        let mut w1 = VecWriter(Vec::new());
        let r1 = graft.apply(&reader, &mut w1, &mut iv1, &mut o1);
        let mut iv2 = [0u16; 2];
        let mut o2 = [0u16; 3];
        let mut w2 = VecWriter(Vec::new());
        let r2 = graft.apply(&reader, &mut w2, &mut iv2, &mut o2);
        assert_eq!(r1, r2);
        assert_eq!(o1, o2);
        assert_eq!(w1.0, w2.0);
    }

    #[test]
    fn switch_threshold_refuses_nan_and_out_of_band() {
        let seg = ReadSegment {
            layer: 0,
            weights: vec![1.0],
        };
        assert!(GraftSwitch::new(vec![seg.clone()], 0.0, f32::NAN).is_err());
        assert!(GraftSwitch::new(vec![seg.clone()], 0.0, 1.5).is_err());
        assert!(GraftSwitch::new(vec![seg], 0.0, 0.0).is_ok());
    }

    #[test]
    fn read_domain_refuses_zero() {
        let seg = ReadSegment {
            layer: 0,
            weights: vec![1.0],
        };
        assert!(GraftRead::new(vec![seg], 0.0, 0).is_err());
    }

    #[test]
    fn graft_construction_refuses_dangling_references() {
        let read = GraftRead::new(
            vec![ReadSegment {
                layer: 0,
                weights: vec![1.0],
            }],
            0.0,
            5,
        )
        .unwrap();
        // A table rule referencing input cell 3 with only one read.
        let rule = CellRule::new(0, &[3], vec![(0, 1)]).unwrap();
        let table = GraftTable::from_rules(vec![rule]).unwrap();
        let err = Graft::new(vec![read.clone()], table, Vec::new(), Vec::new()).unwrap_err();
        assert!(matches!(err, GraftError::Usage(_)));
        // A write naming an unknown output cell.
        let rule_ok = CellRule::new(0, &[0], vec![(0, 1)]).unwrap();
        let table_ok = GraftTable::from_rules(vec![rule_ok]).unwrap();
        let write = GraftWrite::new(
            5,
            vec![WriteEntry {
                value: 0,
                biases: vec![(1, 1.0)],
            }],
        )
        .unwrap();
        let err = Graft::new(vec![read], table_ok, vec![write], Vec::new()).unwrap_err();
        assert!(matches!(err, GraftError::Usage(_)));
    }
}
