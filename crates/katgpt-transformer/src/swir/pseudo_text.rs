//! Latent pseudo-text readout + health sensors (Issue 942) — report-only.
//!
//! The inspection surface for the latent lanes (SwiR soft embeddings, Plan 275;
//! RiM buffer hidden states, Plan 172). When a latent span degrades or loops
//! there is no way to see what the model was "thinking"; this module renders
//! every INSPECTED latent step as pseudo-text: the cosine-nearest vocabulary
//! tokens against a pre-normalised embedding bank (the logit-lens / coda-lens
//! readout family, arXiv:2507.02199; HybridAgent's interpretable latent mode,
//! Research 622 — the delta here is only that ours doesn't ship).
//!
//! # Report-only (G2, structural)
//!
//! Nothing in the decode hot path calls this module. The host arms inspection
//! explicitly — one `readout_into` / `readout_topk_into` call per INSPECTED
//! step — and the disarmed path is bit-identical by construction:
//! [`SwiRController`](super::SwiRController) takes no bank parameter and no
//! sensor hooks. The sensors never write back into the controller.
//!
//! # Cost model (G2)
//!
//! Armed cost = ONE V·D matvec per inspected step (`scores[j] = ê_j · v`, fused
//! with the 1/‖v‖ scale as each row lands), plus O(V) argmax / top-k selection.
//! Bank build is once per table: O(V·D) normalise + unit-norm verification
//! (debug-asserted at build; pinned by the G1 tests below).
//!
//! # Allocation (G4)
//!
//! Every per-call API takes caller-provided scratch (`scores: &mut [f32]`,
//! top-k out-buffers, the render takes borrowed rows). The only allocations
//! are `LatentReadoutBank::build` (the unit rows) and sensor construction
//! (fixed-capacity prev/ring buffers). Zero per-call allocation.
//!
//! # Consumer seam (collapse_aware_thinking)
//!
//! The sensors are designed for `collapse_aware_thinking` consumption:
//! katgpt-pruners' `S2FCollapseDetector` observes ENTROPY streams; these
//! observe LATENT-VECTOR streams — `LatentLoopSensor::observe` and
//! `PseudoTextCollapseSensor::observe` are zero-alloc streaming booleans a
//! thinking controller folds into its budget-cut decision (the root crate
//! deps both, and is the wiring point when that lands). This issue ships the
//! sensors + the report lane only — no controller wiring.
//!
//! # Determinism
//!
//! Tie rule is fixed first-index everywhere (argmax keeps the earliest max;
//! top-k swaps only on strictly-greater) — the readout is a local instrument
//! (never crosses the sync/quorum boundary), but the fixed tie rule keeps
//! traces reproducible across identical runs regardless.
//!
//! # ASCII discipline
//!
//! The markdown trace is pure ASCII. Windows consoles outside UTF-8 garble
//! non-ASCII output, and a debug trace that mojibakes is worse than no trace.

use std::fmt;

/// One readout result: the cosine-nearest bank token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestToken {
    /// Bank row index (the vocabulary id the table was built with).
    pub token: usize,
    /// cos(v/‖v‖, ê_token) — in [-1, 1].
    pub cos: f32,
}

/// Build failure for [`LatentReadoutBank`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadoutBankError {
    /// `table.len() < vocab * dim` — the caller passed a short buffer.
    TableTooSmall { vocab: usize, dim: usize, len: usize },
    /// Row `row` has (near-)zero norm: it can never be a nearest neighbour
    /// and signals a corrupt table the instrument must not paper over.
    ZeroRow { row: usize },
}

impl fmt::Display for ReadoutBankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ReadoutBankError::TableTooSmall { vocab, dim, len } => write!(
                f,
                "embedding table too small: need {vocab} x {dim} = {} floats, got {len}",
                vocab * dim
            ),
            ReadoutBankError::ZeroRow { row } => {
                write!(f, "embedding row {row} has (near-)zero norm — corrupt table")
            }
        }
    }
}

impl std::error::Error for ReadoutBankError {}

/// Pre-normalised embedding bank for cosine nearest-token readout (Issue 942).
///
/// Built ONCE per embedding table (G4): every row is divided by its L2 norm so
/// a cosine against the bank is a plain dot product. Zero-norm rows are
/// refused loudly ([`ReadoutBankError::ZeroRow`]) — a row that can never win a
/// nearest-neighbour query is a loader bug, not a number to average over.
#[derive(Debug, Clone)]
pub struct LatentReadoutBank {
    /// Flattened `[vocab, dim]` row-major; every row unit-norm (G1).
    rows: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl LatentReadoutBank {
    /// Build the bank from a host embedding table (row-major `[vocab, dim]`,
    /// at least `vocab * dim` floats — the same matrix `soft_embedding`
    /// consumes). Normalises each row into owned storage.
    ///
    /// G1: unit-norm is verified at build in debug builds (every row's norm
    /// within 1e-5 of 1.0); the `bank_rows_are_unit_norm` test pins it on the
    /// default test lane.
    pub fn build(table: &[f32], vocab: usize, dim: usize) -> Result<Self, ReadoutBankError> {
        if dim == 0 || table.len() < vocab * dim {
            return Err(ReadoutBankError::TableTooSmall { vocab, dim, len: table.len() });
        }
        let mut rows = table[..vocab * dim].to_vec();
        for v in 0..vocab {
            let row = &mut rows[v * dim..(v + 1) * dim];
            let norm_sq: f32 = row.iter().map(|&x| x * x).sum();
            // NaN or zero norm both refuse (NaN sums propagate from a
            // corrupt row; a NaN row is not a bankable direction).
            if norm_sq.is_nan() || norm_sq <= 1e-24 {
                return Err(ReadoutBankError::ZeroRow { row: v });
            }
            let inv = 1.0 / norm_sq.sqrt();
            for x in row.iter_mut() {
                *x *= inv;
            }
            debug_assert!(
                {
                    let n: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    (n - 1.0).abs() < 1e-5
                },
                "row {v} not unit after normalise"
            );
        }
        Ok(Self { rows, vocab, dim })
    }

    /// Vocab size the bank was built with.
    #[inline]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Embedding width (latent vector length) the bank expects.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Cosine scores of `v` against every bank row, into caller scratch.
    ///
    /// ONE fused V·D matvec per call (G2): each row's dot is multiplied by
    /// 1/‖v‖ as it lands, so `scores` holds cosines directly. Returns `None`
    /// for a zero- or NaN-norm query (a collapsed latent — the caller's
    /// signal, not an error to unwrap).
    ///
    /// `scores.len()` must equal `vocab()`, `v.len()` must equal `dim()`
    /// (debug-asserted; release skips the checks per hot-path convention).
    pub fn cos_scores_into(&self, v: &[f32], scores: &mut [f32]) -> Option<()> {
        debug_assert_eq!(v.len(), self.dim, "query width != bank dim");
        debug_assert_eq!(scores.len(), self.vocab, "scores scratch != vocab");
        let mut norm_sq = 0.0f32;
        for &x in v.iter() {
            norm_sq += x * x;
        }
        // NaN or zero norm: a collapsed latent — the caller's signal, not an
        // error to unwrap (see the doc above).
        if norm_sq.is_nan() || norm_sq <= 0.0 {
            return None;
        }
        let inv = 1.0 / norm_sq.sqrt();
        // Chunked dot (8 accumulators break the FP dependency chain and give
        // LLVM the auto-vectorise shape — same pattern as soft_embedding).
        const CHUNK: usize = 8;
        let mut j = 0usize;
        while j < self.vocab {
            let off = j * self.dim;
            let row = &self.rows[off..off + self.dim];
            let mut acc = [0.0f32; CHUNK];
            let mut d = 0usize;
            while d + CHUNK <= self.dim {
                for k in 0..CHUNK {
                    acc[k] += row[d + k] * v[d + k];
                }
                d += CHUNK;
            }
            let mut dot: f32 = acc.iter().sum();
            while d < self.dim {
                dot += row[d] * v[d];
                d += 1;
            }
            scores[j] = dot * inv;
            j += 1;
        }
        Some(())
    }

    /// Nearest-token readout: `argmax_j cos(v/‖v‖, ê_j)` (Issue 942 D1).
    ///
    /// One V·D matvec + O(V) argmax scan. First-index tie rule. Zero alloc.
    /// `None` for a zero/NaN-norm query (see [`cos_scores_into`](Self::cos_scores_into)).
    pub fn readout_into(&self, v: &[f32], scores: &mut [f32]) -> Option<NearestToken> {
        self.cos_scores_into(v, scores)?;
        let mut best = usize::MAX;
        let mut best_cos = f32::NEG_INFINITY;
        for (j, &s) in scores.iter().enumerate() {
            if s > best_cos {
                best_cos = s;
                best = j;
            }
        }
        debug_assert_ne!(best, usize::MAX, "vocab > 0 guaranteed at build");
        Some(NearestToken { token: best, cos: best_cos })
    }

    /// Top-k cosine-nearest tokens, sorted by cos descending (Issue 942 D1).
    ///
    /// Fills `out` (k = `out.len()`; caller-sized) with `(token, cos)` pairs;
    /// returns the filled count (0 on zero/NaN-norm query or empty `out`).
    /// First-index tie rule: on equal cosines the earlier bank row wins.
    /// Zero alloc — the selection is an in-buffer insertion ladder, O(V·k)
    /// with k tiny.
    pub fn readout_topk_into(
        &self,
        v: &[f32],
        out: &mut [(usize, f32)],
        scores: &mut [f32],
    ) -> usize {
        if out.is_empty() || self.cos_scores_into(v, scores).is_none() {
            return 0;
        }
        let k = out.len();
        let mut filled = 0usize;
        for (j, &s) in scores.iter().enumerate() {
            if filled < k {
                out[filled] = (j, s);
                filled += 1;
                sift_up(out, filled);
            } else if s > out[k - 1].1 {
                out[k - 1] = (j, s);
                sift_up(out, k);
            }
        }
        filled
    }
}

/// Bubble the just-written `out[len-1]` up into sorted (desc) position.
/// Strict `<` keeps the first-index tie rule (equal scores never swap).
fn sift_up(out: &mut [(usize, f32)], len: usize) {
    let mut i = len - 1;
    while i > 0 && out[i - 1].1 < out[i].1 {
        out.swap(i - 1, i);
        i -= 1;
    }
}

/// Mean max-cosine over inspected steps — the span readability score
/// (Issue 942 D1: "span readability = mean max-cosine").
///
/// `None` when `max_cos` is empty: a blind instrument must not report a
/// number (the blindness-floor half of the report contract).
pub fn span_readability(max_cos: &[f32]) -> Option<f32> {
    if max_cos.is_empty() {
        return None;
    }
    Some(max_cos.iter().sum::<f32>() / max_cos.len() as f32)
}

/// cos(a, b) on raw slices; 0.0 when either side has zero/NaN norm (a zero
/// vector is not a repeat of anything — a smaller cos, not a NaN).
fn cosine_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

/// Loop sensor (Issue 942 D2): `cos(v_t, v_{t−1}) > θ_c` for `k` consecutive
/// steps. Streaming, zero-alloc per call (`prev` is allocated once at
/// construction and reused).
///
/// `observe` returns the LEVEL (streak >= k), not the edge — callers wanting
/// an edge diff against the previous return.
#[derive(Debug, Clone)]
pub struct LatentLoopSensor {
    theta_c: f32,
    k_required: usize,
    streak: usize,
    prev: Vec<f32>,
    has_prev: bool,
}

impl LatentLoopSensor {
    /// `theta_c` must be finite; `k_required >= 1` (0 would fire on the first
    /// observation, which is not a loop); `dim` is the latent width.
    pub fn new(theta_c: f32, k_required: usize, dim: usize) -> Self {
        debug_assert!(theta_c.is_finite(), "theta_c must be finite");
        debug_assert!(k_required >= 1, "k_required >= 1 (0 fires on first obs)");
        Self {
            theta_c,
            k_required: k_required.max(1),
            streak: 0,
            prev: vec![0.0; dim],
            has_prev: false,
        }
    }

    /// Observe one latent step. First observation resets the streak (nothing
    /// to repeat yet). Zero-norm steps break the streak (cos reads 0.0).
    pub fn observe(&mut self, v: &[f32]) -> bool {
        debug_assert_eq!(v.len(), self.prev.len(), "latent width mismatch");
        if self.has_prev {
            let c = cosine_slice(v, &self.prev);
            if c > self.theta_c {
                self.streak += 1;
            } else {
                self.streak = 0;
            }
        } else {
            self.streak = 0;
        }
        self.prev.copy_from_slice(v);
        self.has_prev = true;
        self.streak >= self.k_required
    }

    /// Current streak length (consecutive steps over θ_c).
    pub fn streak(&self) -> usize {
        self.streak
    }

    /// Reset for a new span (clears streak + prev).
    pub fn reset(&mut self) {
        self.streak = 0;
        self.has_prev = false;
    }
}

/// Collapse sensor (Issue 942 D2): the nearest-token is CONSTANT over a window
/// of `w` steps with score > θ throughout. Streaming, zero-alloc per call
/// (fixed-capacity ring allocated at construction).
///
/// Cannot fire before the window fills (`w >= 2` enforced — a window of 1 is
/// a score threshold, a different sensor).
#[derive(Debug, Clone)]
pub struct PseudoTextCollapseSensor {
    window: usize,
    theta: f32,
    ring: Vec<(usize, f32)>,
    len: usize,
    head: usize,
}

impl PseudoTextCollapseSensor {
    pub fn new(window: usize, theta: f32) -> Self {
        debug_assert!(window >= 2, "window >= 2 (1 degenerates to a score gate)");
        debug_assert!(theta.is_finite(), "theta must be finite");
        Self {
            window: window.max(2),
            theta,
            ring: vec![(usize::MAX, f32::NEG_INFINITY); window.max(2)],
            len: 0,
            head: 0,
        }
    }

    /// Observe one readout step; returns the LEVEL (collapsed now), not the
    /// edge. Fires only when the ring is full AND every token equal AND every
    /// cos > θ.
    pub fn observe(&mut self, n: NearestToken) -> bool {
        self.ring[self.head] = (n.token, n.cos);
        self.head = (self.head + 1) % self.window;
        if self.len < self.window {
            self.len += 1;
        }
        if self.len < self.window {
            return false;
        }
        let t0 = self.ring[0].0;
        self.ring.iter().all(|&(t, _)| t == t0) && self.ring.iter().all(|&(_, c)| c > self.theta)
    }

    /// Reset for a new span.
    pub fn reset(&mut self) {
        self.len = 0;
        self.head = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Trace surface (Issue 942 D3) — glyph-led ASCII markdown, Copy-as-Markdown
// ────────────────────────────────────────────────────────────────────────────

/// One rendered trace row: the borrowed top-k plus that step's sensor levels.
#[derive(Debug, Clone, Copy)]
pub struct TraceRow<'a> {
    pub step: usize,
    /// `(token, cos)` pairs, cos-descending — the pseudo-text for this step.
    pub top: &'a [(usize, f32)],
    pub loop_fired: bool,
    pub collapse_fired: bool,
}

/// Render the glyph-led markdown trace (Issue 942 D3).
///
/// - The output IS the Copy-as-Markdown surface: print it, pipe it, paste it.
/// - Blindness floor: when fewer than `min_steps` rows were inspected, the
///   verdict line reads `BLIND` and withholds every green claim — an
///   instrument that inspects nothing cannot print green.
/// - Pure ASCII (see module header).
///
/// `readability` comes from [`span_readability`]; pass its `Option` straight
/// through (`n/a` renders when no steps were inspected).
pub fn render_trace_markdown(
    rows: &[TraceRow<'_>],
    readability: Option<f32>,
    min_steps: usize,
    title: &str,
) -> String {
    let n = rows.len();
    let mut s = String::with_capacity(64 + n * 48);
    s.push_str("## Latent pseudo-text trace - ");
    s.push_str(title);
    s.push_str("\n\n- steps inspected: ");
    s.push_str(&n.to_string());
    s.push_str(" (floor ");
    s.push_str(&min_steps.to_string());
    s.push_str(")\n- verdict: ");
    if n >= min_steps {
        s.push_str("READ (");
        s.push_str(&n.to_string());
        s.push_str(" >= floor ");
        s.push_str(&min_steps.to_string());
        s.push(')');
    } else {
        s.push_str("BLIND - ");
        s.push_str(&n.to_string());
        s.push_str(" < floor ");
        s.push_str(&min_steps.to_string());
        s.push_str("; readout withheld from any green claim");
    }
    s.push_str("\n- span readability (mean max-cos): ");
    match readability {
        Some(r) => s.push_str(&format!("{r:.4}")),
        None => s.push_str("n/a (no steps inspected)"),
    }
    s.push_str("\n\n| step | nearest tokens (cos) | sensors |\n|-----:|----------------------|---------|\n");
    for row in rows {
        s.push_str("| ");
        s.push_str(&row.step.to_string());
        s.push_str(" | ");
        for (i, &(tok, cos)) in row.top.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("#{tok}({cos:.3})"));
        }
        if row.top.is_empty() {
            s.push('-');
        }
        s.push_str(" | ");
        let mut any = false;
        if row.loop_fired {
            s.push_str("LOOP");
            any = true;
        }
        if row.collapse_fired {
            if any {
                s.push(' ');
            }
            s.push_str("COLLAPSE");
            any = true;
        }
        if !any {
            s.push('-');
        }
        s.push_str(" |\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-table: row j = sin/cos mixture in j (no zero rows).
    fn synth_table(vocab: usize, dim: usize) -> Vec<f32> {
        let mut t = Vec::with_capacity(vocab * dim);
        for j in 0..vocab {
            for d in 0..dim {
                let x = (j as f32 * 0.37 + d as f32 * 0.11).sin()
                    + 0.5 * ((j as f32 * 0.13 - d as f32 * 0.07).cos());
                t.push(x + 0.25);
            }
        }
        t
    }

    #[test]
    fn bank_rows_are_unit_norm() {
        // G1: unit-norm verified at build.
        let vocab = 40;
        let dim = 16;
        let bank = LatentReadoutBank::build(&synth_table(vocab, dim), vocab, dim).unwrap();
        for v in 0..vocab {
            let row = &bank.rows[v * dim..(v + 1) * dim];
            let n: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5, "row {v}: norm {n}");
        }
        assert_eq!(bank.vocab(), vocab);
        assert_eq!(bank.dim(), dim);
    }

    #[test]
    fn bank_refuses_short_table_and_zero_row() {
        let dim = 4;
        // Short buffer.
        let e = LatentReadoutBank::build(&[0.0; 3], 2, dim).unwrap_err();
        assert_eq!(e, ReadoutBankError::TableTooSmall { vocab: 2, dim, len: 3 });
        // Zero row.
        let mut t = synth_table(3, dim);
        for x in t[2 * dim..3 * dim].iter_mut() {
            *x = 0.0;
        }
        let e = LatentReadoutBank::build(&t, 3, dim).unwrap_err();
        assert_eq!(e, ReadoutBankError::ZeroRow { row: 2 });
    }

    #[test]
    fn peaked_distribution_readout_is_the_peak_token() {
        // G1: for a peaked (one-hot) distribution the readout equals argmax
        // with ~1.0 agreement. soft_embedding(one-hot at j) == e(j), and the
        // bank row j is e(j) normalised -> cos == 1.0 (to f32).
        let vocab = 32;
        let dim = 24;
        let table = synth_table(vocab, dim);
        let bank = LatentReadoutBank::build(&table, vocab, dim).unwrap();
        let mut emb = vec![0.0f32; dim];
        let mut probs = vec![0.0f32; vocab];
        let mut scores = vec![0.0f32; vocab];
        let mut top = vec![(0usize, 0.0f32); 3];
        for j in 0..vocab {
            for p in probs.iter_mut() {
                *p = 0.0;
            }
            probs[j] = 1.0;
            for e in emb.iter_mut() {
                *e = 0.0;
            }
            crate::swir::soft_embedding(&probs, &table, dim, &mut emb);
            let near = bank.readout_into(&emb, &mut scores).unwrap();
            assert_eq!(near.token, j, "argmax mismatch at one-hot {j}");
            assert!((near.cos - 1.0).abs() < 1e-5, "cos at one-hot {j}: {}", near.cos);
            // Top-k agrees too, head first.
            let filled = bank.readout_topk_into(&emb, &mut top, &mut scores);
            assert_eq!(filled, 3);
            assert_eq!(top[0].0, j);
        }
    }

    #[test]
    fn zero_norm_query_reads_none() {
        let vocab = 8;
        let dim = 4;
        let bank = LatentReadoutBank::build(&synth_table(vocab, dim), vocab, dim).unwrap();
        let mut scores = vec![0.0f32; vocab];
        assert!(bank.readout_into(&vec![0.0; dim], &mut scores).is_none());
        // NaN query norm is also None (NaN > 0.0 is false).
        let nan_v = vec![f32::NAN; dim];
        assert!(bank.readout_into(&nan_v, &mut scores).is_none());
    }

    #[test]
    fn topk_sorted_desc_first_index_ties() {
        let vocab = 16;
        let dim = 8;
        let bank = LatentReadoutBank::build(&synth_table(vocab, dim), vocab, dim).unwrap();
        let v: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.3).cos()).collect();
        let mut scores = vec![0.0f32; vocab];
        let mut top = vec![(0usize, 0.0f32); 5];
        let filled = bank.readout_topk_into(&v, &mut top, &mut scores);
        assert_eq!(filled, 5);
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1, "not descending: {:?}", top);
        }
        // Duplicate rows build a tie: row 2 == row 3 -> first index wins.
        let mut t = synth_table(vocab, dim);
        let (a, b) = (2 * dim, 3 * dim);
        t.copy_within(a..a + dim, b);
        let bank2 = LatentReadoutBank::build(&t, vocab, dim).unwrap();
        let q = &t[a..a + dim];
        let mut top2 = vec![(0usize, 0.0f32); 4];
        bank2.readout_topk_into(q, &mut top2, &mut scores);
        assert_eq!(top2[0].0, 2, "first-index tie rule: got {:?}", top2[0]);
        assert_eq!(top2[1].0, 3);
        assert!((top2[0].1 - top2[1].1).abs() < 1e-6);
    }

    #[test]
    fn loop_sensor_fires_on_consecutive_repeats() {
        let dim = 6;
        let mut s = LatentLoopSensor::new(0.99, 2, dim);
        let v0: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).sin() + 1.0).collect();
        let mut v1 = v0.clone();
        v1[0] += 1e-4; // cos(v0, v1) ~ 1.0
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 * 2.1).cos() + 0.5).collect();
        assert!(!s.observe(&v0)); // first obs: streak 0
        assert!(!s.observe(&v1)); // streak 1 < 2
        assert!(s.observe(&v1)); // streak 2 -> fires (level)
        assert!(s.observe(&v1)); // stays fired
        assert!(!s.observe(&v2)); // broken
        assert_eq!(s.streak(), 0);
        // Reset clears state.
        s.reset();
        assert!(!s.observe(&v1)); // treated as first obs again
    }

    #[test]
    fn collapse_sensor_constant_high_score_window() {
        let mut s = PseudoTextCollapseSensor::new(3, 0.8);
        let hit = NearestToken { token: 41, cos: 0.95 };
        let other = NearestToken { token: 7, cos: 0.95 };
        let low = NearestToken { token: 41, cos: 0.5 };
        assert!(!s.observe(hit)); // window not full
        assert!(!s.observe(hit));
        assert!(s.observe(hit)); // full + constant + all > theta -> fires
        assert!(!s.observe(other)); // token changed -> no fire
        assert!(!s.observe(hit));
        assert!(!s.observe(hit)); // full again but window has `other` -> no
        let mut s2 = PseudoTextCollapseSensor::new(2, 0.8);
        assert!(!s2.observe(low));
        assert!(!s2.observe(low)); // constant but cos <= theta -> no fire
    }

    #[test]
    fn readability_mean_and_empty_none() {
        assert_eq!(span_readability(&[]), None);
        let r = span_readability(&[0.5, 1.0, 0.25]).unwrap();
        assert!((r - 0.5833333).abs() < 1e-5);
    }

    #[test]
    fn render_contains_glyphs_and_blindness_floor() {
        let top = [(41usize, 0.912f32), (7, 0.804), (13, 0.771)];
        let rows = [
            TraceRow { step: 0, top: &top, loop_fired: false, collapse_fired: false },
            TraceRow { step: 5, top: &top, loop_fired: true, collapse_fired: true },
        ];
        let md = render_trace_markdown(&rows, Some(0.83), 8, "healthy span");
        assert!(md.contains("## Latent pseudo-text trace - healthy span"));
        assert!(md.contains("verdict: BLIND - 2 < floor 8"));
        assert!(md.contains("readability (mean max-cos): 0.8300"));
        assert!(md.contains("#41(0.912) #7(0.804) #13(0.771)"));
        assert!(md.contains("LOOP COLLAPSE"));
        // Read verdict over the floor.
        let md2 = render_trace_markdown(&rows, None, 2, "x");
        assert!(md2.contains("verdict: READ (2 >= floor 2)"));
        assert!(md2.contains("readability (mean max-cos): n/a (no steps inspected)"));
        // Determinism: same input -> byte-identical output.
        let md3 = render_trace_markdown(&rows, Some(0.83), 8, "healthy span");
        assert_eq!(md, md3);
    }
}
