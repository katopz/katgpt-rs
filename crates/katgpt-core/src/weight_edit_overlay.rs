//! WeightEditOverlay — reasoning-authored, gate-gated weight-edit lane
//! (Issue 980 / Research 648 — "Sparks of Weight-Space Programming").
//!
//! A **deterministic weight-edit overlay**: an edit spec authored once (LLM
//! session or hand-written from diagnostics), applied by a fixed compiler,
//! gated by a pre-registered control battery, committed as a BLAKE3-pinned
//! frozen artifact. Two edit forms:
//!
//! 1. **Direction subtraction** — `ΔW[:, S] = −k·d` (rank-1, column-masked;
//!    the blog's Exp-8 loop fix — the one edit class unpublished before the
//!    blog).
//! 2. **Bypass rows** — a small sigmoid-gated pathway through existing units
//!    (the blog's Exp-4 ramp fix). The gate is a SIGMOID DRIVE on a measured
//!    quantity — never a hard switch-off (the measured oscillation shape;
//!    plan-317-class precedent).
//!
//! # The law the lane obeys (obstacle-vs-ability)
//!
//! Weight edits remove obstacles that hide abilities; they do not add
//! abilities. The **arming condition is obstacle telemetry**:
//! [`crate::gain_cost_halt::GainCostLoopHalter`] (`HaltReason::Oscillation`)
//! counters above threshold arm the lane; below threshold the lane REFUSES
//! "no obstacle found" ([`arm_from_telemetry`]). This is the falsifier built
//! into the arming logic — a model with no single failing component is out
//! of scope (the blog's Gemma control, carried as a refusal arm).
//!
//! # Modelless compliance
//!
//! No training, no gradient descent, no runtime weight mutation beyond the
//! overlay itself: authoring may be stochastic (hosted session); application
//! and verdict are deterministic closed-form arithmetic. The committed
//! artifact is mint-once-then-frozen — recipes are not durable, artifacts
//! are ([`WeightEditArtifact`]).
//!
//! # PQ/ternary posture (QuantErrorLora posture)
//!
//! Apply deltas **in the dequant domain at forward** ([`WeightEditOverlay::
//! forward_delta_into`]) — never requantize a packed base. The rank-1
//! column-masked delta is `Δy[i] = −k·d[i]·(Σ_{j∈S} h[j])`: O(|S| + out)
//! per forward, zero allocation, and the packed base bytes are never
//! touched. On a dense f32 base the compose form ([`WeightEditOverlay::
//! compose_layer_into`]) is the pinned-exact path.
//!
//! # Determinism (Issue 940 law)
//!
//! Every fold in this module is sequential in spec order (edits, then
//! columns, then rows). Nothing here crosses a sync/quorum boundary — it is
//! a local instrument — but the same discipline keeps runs bit-reproducible.
//!
//! # Demotion law (pre-registered, Issue-565 precedent)
//!
//! If the gate battery fails its control arms, the lane records the negative
//! and the carrier demotes to the `SparseErrorBypass` representation it
//! already is — no zombie lane. Research 648 §5.

use crate::gain_cost_halt::{GainCostLoopHalter, HaltDecision, angular_change, step_size};

// ─────────────────────────────────────────────────────────────────────
// Edit spec
// ─────────────────────────────────────────────────────────────────────

/// One authored edit form.
///
/// Encoding tags: `DirectionSubtraction = 1`, `BypassRows = 2`.
#[derive(Clone, Debug, PartialEq)]
pub enum EditForm {
    /// `ΔW[:, S] = −k·d` — rank-1, column-masked. `direction` has
    /// `out_dim` entries (the layer's output width); `strength` is `k`.
    DirectionSubtraction {
        /// Column indices `S ⊂ [0, in_dim)` whose weight columns are edited.
        columns: Vec<u32>,
        /// The (unit-norm by authoring convention) direction `d`, length
        /// `out_dim`. Applied elementwise as `W[i][col] -= k·d[i]`.
        direction: Vec<f32>,
        /// Strength `k`. Dose-response over a pinned grid is the G1 gate.
        strength: f32,
    },
    /// Sigmoid-gated bypass pathway: `y[i] += g·Σ_j V[i][j]·h[p_j]` with
    /// `g = exact_sigmoid(w·h_gate + b)` — a SIGMOID DRIVE on the measured
    /// gate input, never a hard switch. Composes into the layer's
    /// PRE-ACTIVATION output; weights are untouched (the pathway is a
    /// forward-side overlay, so "revert" for this form is structural
    /// absence — uninstall the overlay).
    BypassRows {
        /// Input dims the gate reads.
        gate_columns: Vec<u32>,
        /// Gate weights (same length as `gate_columns`).
        gate_weights: Vec<f32>,
        /// Gate bias.
        gate_bias: f32,
        /// Input dims the pathway reads (the "existing units").
        pathway_columns: Vec<u32>,
        /// Row-major readout `[out_dim][n_path]`.
        pathway_readout: Vec<f32>,
        /// Layer output width.
        out_dim: u32,
    },
}

/// One edit: which layer, which form.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightEditSpec {
    /// Layer index (consumer's numbering; the applier is told which weight
    /// matrix `layer` names).
    pub layer: u8,
    pub form: EditForm,
}

impl WeightEditSpec {
    /// Structural validation. `Err` carries the reason; decode and
    /// construction both refuse loud on malformed specs (tamper class).
    pub fn validate(&self) -> Result<(), &'static str> {
        match &self.form {
            EditForm::DirectionSubtraction {
                columns,
                direction,
                strength,
            } => {
                if columns.is_empty() {
                    return Err("direction edit: empty column set");
                }
                if direction.is_empty() {
                    return Err("direction edit: empty direction");
                }
                if !strength.is_finite() {
                    return Err("direction edit: non-finite strength");
                }
                if direction.iter().any(|v| !v.is_finite()) {
                    return Err("direction edit: non-finite direction entry");
                }
                Ok(())
            }
            EditForm::BypassRows {
                gate_columns,
                gate_weights,
                gate_bias,
                pathway_columns,
                pathway_readout,
                out_dim,
            } => {
                if gate_columns.is_empty() {
                    return Err("bypass edit: empty gate columns");
                }
                if gate_weights.len() != gate_columns.len() {
                    return Err("bypass edit: gate weights/columns length mismatch");
                }
                if pathway_columns.is_empty() {
                    return Err("bypass edit: empty pathway columns");
                }
                if !gate_bias.is_finite() {
                    return Err("bypass edit: non-finite gate bias");
                }
                if *out_dim == 0 {
                    return Err("bypass edit: zero out_dim");
                }
                if pathway_readout.len() != *out_dim as usize * pathway_columns.len() {
                    return Err("bypass edit: readout length != out_dim × n_path");
                }
                if gate_weights
                    .iter()
                    .chain(pathway_readout.iter())
                    .any(|v| !v.is_finite())
                {
                    return Err("bypass edit: non-finite weight entry");
                }
                Ok(())
            }
        }
    }

    fn encode_into(&self, buf: &mut Vec<u8>) {
        buf.push(self.layer);
        match &self.form {
            EditForm::DirectionSubtraction {
                columns,
                direction,
                strength,
            } => {
                buf.push(1);
                put_u32(buf, columns.len() as u64);
                for c in columns {
                    put_u32(buf, *c as u64);
                }
                put_u32(buf, direction.len() as u64);
                for v in direction {
                    put_f32(buf, *v);
                }
                put_f32(buf, *strength);
            }
            EditForm::BypassRows {
                gate_columns,
                gate_weights,
                gate_bias,
                pathway_columns,
                pathway_readout,
                out_dim,
            } => {
                buf.push(2);
                put_u32(buf, gate_columns.len() as u64);
                for c in gate_columns {
                    put_u32(buf, *c as u64);
                }
                put_u32(buf, gate_weights.len() as u64);
                for v in gate_weights {
                    put_f32(buf, *v);
                }
                put_f32(buf, *gate_bias);
                put_u32(buf, pathway_columns.len() as u64);
                for c in pathway_columns {
                    put_u32(buf, *c as u64);
                }
                put_u32(buf, pathway_readout.len() as u64);
                for v in pathway_readout {
                    put_f32(buf, *v);
                }
                put_u32(buf, *out_dim as u64);
            }
        }
    }

    fn decode(cursor: &mut Cursor<'_>) -> Option<Self> {
        let layer = cursor.read_u8()?;
        let form = match cursor.read_u8()? {
            1 => {
                let n_cols = cursor.read_u32()? as usize;
                let mut columns = Vec::with_capacity(n_cols);
                for _ in 0..n_cols {
                    columns.push(cursor.read_u32()?);
                }
                let n_dir = cursor.read_u32()? as usize;
                let mut direction = Vec::with_capacity(n_dir);
                for _ in 0..n_dir {
                    direction.push(cursor.read_f32()?);
                }
                let strength = cursor.read_f32()?;
                EditForm::DirectionSubtraction {
                    columns,
                    direction,
                    strength,
                }
            }
            2 => {
                let n_gate = cursor.read_u32()? as usize;
                let mut gate_columns = Vec::with_capacity(n_gate);
                for _ in 0..n_gate {
                    gate_columns.push(cursor.read_u32()?);
                }
                let n_gw = cursor.read_u32()? as usize;
                let mut gate_weights = Vec::with_capacity(n_gw);
                for _ in 0..n_gw {
                    gate_weights.push(cursor.read_f32()?);
                }
                let gate_bias = cursor.read_f32()?;
                let n_path = cursor.read_u32()? as usize;
                let mut pathway_columns = Vec::with_capacity(n_path);
                for _ in 0..n_path {
                    pathway_columns.push(cursor.read_u32()?);
                }
                let n_read = cursor.read_u32()? as usize;
                let mut pathway_readout = Vec::with_capacity(n_read);
                for _ in 0..n_read {
                    pathway_readout.push(cursor.read_f32()?);
                }
                let out_dim = cursor.read_u32()?;
                EditForm::BypassRows {
                    gate_columns,
                    gate_weights,
                    gate_bias,
                    pathway_columns,
                    pathway_readout,
                    out_dim,
                }
            }
            _ => return None,
        };
        Some(Self { layer, form })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Overlay
// ─────────────────────────────────────────────────────────────────────

const MAGIC_OVERLAY: [u8; 4] = *b"WEOL";
const MAGIC_ARTIFACT: [u8; 4] = *b"WEAR";

/// An authored overlay: an ordered set of edits. The ORDER is load-bearing —
/// application folds edits in spec order (determinism law).
#[derive(Clone, Debug, PartialEq)]
pub struct WeightEditOverlay {
    edits: Vec<WeightEditSpec>,
}

impl WeightEditOverlay {
    /// Construct + validate every edit. `None` on any malformed edit.
    pub fn new(edits: Vec<WeightEditSpec>) -> Option<Self> {
        for e in &edits {
            e.validate().ok()?;
        }
        Some(Self { edits })
    }

    pub fn edits(&self) -> &[WeightEditSpec] {
        &self.edits
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// BLAKE3 over the canonical encoding. This is the artifact payload's
    /// identity — mint-once-then-freeze pins THIS digest.
    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&self.to_bytes()).as_bytes()
    }

    /// Canonical LE bytes (length-prefixed everywhere — no concatenation
    /// aliasing; the `LoraAdapter::weight_epoch` convention).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC_OVERLAY);
        put_u32(&mut buf, self.edits.len() as u64);
        for e in &self.edits {
            e.encode_into(&mut buf);
        }
        buf
    }

    /// Decode + validate. `None` on truncation, bad magic, unknown form
    /// tag, or any edit failing validation.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 || bytes[0..4] != MAGIC_OVERLAY {
            return None;
        }
        let mut cursor = Cursor::new(&bytes[4..]);
        let n = cursor.read_u32()? as usize;
        let mut edits = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            edits.push(WeightEditSpec::decode(&mut cursor)?);
        }
        Self::new(edits)
    }

    /// Number of unique weight-matrix elements the compose of `layer`
    /// touches (deduped across edits — first touch wins the backup slot).
    /// Allocation-free (nested scan — spec-scale column counts).
    pub fn touched_count(&self, layer: u8, out_dim: usize, in_dim: usize) -> usize {
        let mut unique = 0usize;
        for e in &self.edits {
            if e.layer != layer {
                continue;
            }
            let EditForm::DirectionSubtraction { columns, .. } = &e.form else {
                continue;
            };
            for c in columns {
                if *c as usize >= in_dim {
                    continue;
                }
                if !self.column_already_touched(layer, e, c) {
                    unique += 1;
                }
            }
        }
        unique * out_dim
    }

    /// Was `col` already touched by an EARLIER edit (or an earlier column of
    /// the same edit)? The first-touch dedupe rule shared by compose, revert
    /// and count. Allocation-free.
    fn column_already_touched(&self, layer: u8, edit: &WeightEditSpec, col: &u32) -> bool {
        let mut earlier = false;
        for e in &self.edits {
            if std::ptr::eq(e, edit) {
                break;
            }
            if e.layer == layer
                && let EditForm::DirectionSubtraction { columns, .. } = &e.form
                && columns.contains(col)
            {
                earlier = true;
                break;
            }
        }
        if earlier {
            return true;
        }
        // Earlier columns of the SAME edit.
        if let EditForm::DirectionSubtraction { columns, .. } = &edit.form
            && let Some(pos) = columns.iter().position(|c| std::ptr::eq(c, col))
        {
            return columns[..pos].contains(col);
        }
        false
    }

    /// Compose (apply at load) every direction-subtraction edit targeting
    /// `layer` into the row-major weight matrix `weights` (`out_dim ×
    /// in_dim`). Touched elements are backed up into `backup` (original
    /// values, first-touch-per-column) so the revert is BIT-IDENTICAL —
    /// `W −= k·d` then `W += k·d` is NOT bit-exact in f32; the backup is
    /// the fix_verify-shaped revert (baseline → apply → re-check → revert).
    ///
    /// Returns the number of backup elements written, or `None` when
    /// `backup` is too small ([`Self::touched_count`] is the contract).
    /// Bypass edits compose nothing (they are forward-side).
    ///
    /// Zero allocation. Untouched columns are bit-identical by construction
    /// (only `S`-column elements are written).
    pub fn compose_layer_into(
        &self,
        layer: u8,
        weights: &mut [f32],
        out_dim: usize,
        in_dim: usize,
        backup: &mut [f32],
    ) -> Option<usize> {
        if weights.len() != out_dim * in_dim {
            return None;
        }
        let mut used = 0usize;
        for e in &self.edits {
            if e.layer != layer {
                continue;
            }
            let EditForm::DirectionSubtraction {
                columns,
                direction,
                strength,
            } = &e.form
            else {
                continue;
            };
            if direction.len() != out_dim {
                return None;
            }
            for c in columns {
                let col = *c as usize;
                if col >= in_dim {
                    // Out-of-range column: refuse loud (tamper class) —
                    // never silently skip a written check.
                    return None;
                }
                // First-touch dedupe: only an edit that is the FIRST to
                // touch `col` backs it up (later edits mutate the already
                // back-uped value; the original must survive).
                if !self.column_already_touched(layer, e, c) {
                    if used + out_dim > backup.len() {
                        return None;
                    }
                    for i in 0..out_dim {
                        backup[used] = weights[i * in_dim + col];
                        used += 1;
                    }
                }
                for i in 0..out_dim {
                    weights[i * in_dim + col] -= strength * direction[i];
                }
            }
        }
        Some(used)
    }

    /// Revert a compose: restore `backup` into the same first-touch pattern
    /// (spec order). Bit-identical to the pre-compose matrix. Returns the
    /// number of elements restored (== the compose's return on the same
    /// overlay/layer).
    pub fn revert_layer(
        &self,
        layer: u8,
        weights: &mut [f32],
        out_dim: usize,
        in_dim: usize,
        backup: &[f32],
    ) -> usize {
        let mut used = 0usize;
        for e in &self.edits {
            if e.layer != layer {
                continue;
            }
            let EditForm::DirectionSubtraction { columns, .. } = &e.form else {
                continue;
            };
            for c in columns {
                let col = *c as usize;
                if col >= in_dim {
                    break;
                }
                if self.column_already_touched(layer, e, c) {
                    continue;
                }
                for i in 0..out_dim {
                    weights[i * in_dim + col] = backup[used];
                    used += 1;
                }
            }
        }
        used
    }

    /// Forward-domain delta (the dequant-domain posture): add the rank-1
    /// column-masked contribution of every direction edit targeting
    /// `layer` into the layer's PRE-ACTIVATION output `y` (`y = W·h` was
    /// computed by the caller — packed base untouched). Edits fold in spec
    /// order; each edit folds its columns sequentially, then writes rows
    /// sequentially. Zero allocation; O(|S|·edits + out·edits).
    pub fn forward_delta_into(&self, layer: u8, h: &[f32], y: &mut [f32]) {
        for e in &self.edits {
            if e.layer != layer {
                continue;
            }
            let EditForm::DirectionSubtraction {
                columns,
                direction,
                strength,
            } = &e.form
            else {
                continue;
            };
            if direction.len() > y.len() {
                continue;
            }
            // Sequential column fold (Issue-940 order pin).
            let mut acc = 0.0f32;
            for c in columns {
                let j = *c as usize;
                if j < h.len() {
                    acc += h[j];
                }
            }
            if acc == 0.0 {
                // Bit-exact no-op — zero carrier mass (the retention law:
                // out-of-region inputs see the edit contribute exactly 0).
                continue;
            }
            for (i, d_i) in direction.iter().enumerate().take(y.len()) {
                y[i] -= strength * d_i * acc;
            }
        }
    }

    /// Forward-side bypass pathways for `layer`: `y[i] += g·(V·h_path)[i]`
    /// with `g = exact_sigmoid(w·h_gate + bias)`. `g == 0.0` skips the
    /// write (bit-exact no-op). Zero allocation.
    pub fn bypass_forward_into(&self, layer: u8, h: &[f32], y: &mut [f32]) {
        for e in &self.edits {
            if e.layer != layer {
                continue;
            }
            let EditForm::BypassRows {
                gate_columns,
                gate_weights,
                gate_bias,
                pathway_columns,
                pathway_readout,
                out_dim,
            } = &e.form
            else {
                continue;
            };
            let out = *out_dim as usize;
            if out > y.len() || pathway_readout.len() != out * pathway_columns.len() {
                continue;
            }
            let mut gate = *gate_bias;
            for (c, w) in gate_columns.iter().zip(gate_weights.iter()) {
                let j = *c as usize;
                if j < h.len() {
                    gate += w * h[j];
                }
            }
            let g = crate::exact_sigmoid(gate);
            if g == 0.0 {
                continue;
            }
            for i in 0..out {
                let mut dot = 0.0f32;
                for (pj, pcol) in pathway_columns.iter().enumerate() {
                    let j = *pcol as usize;
                    if j < h.len() {
                        dot += pathway_readout[i * pathway_columns.len() + pj] * h[j];
                    }
                }
                y[i] += g * dot;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Canonical byte helpers
// ─────────────────────────────────────────────────────────────────────

fn put_u32(buf: &mut Vec<u8>, v: u64) {
    let v = u32::try_from(v).unwrap_or(u32::MAX);
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn read_u8(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }
    fn read_u32(&mut self) -> Option<u32> {
        if self.pos + 4 > self.bytes.len() {
            return None;
        }
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.bytes[self.pos..self.pos + 4]);
        self.pos += 4;
        Some(u32::from_le_bytes(b))
    }
    fn read_f32(&mut self) -> Option<f32> {
        Some(f32::from_bits(self.read_u32()?))
    }
}

// ─────────────────────────────────────────────────────────────────────
// Obstacle arming gate — the falsifier
// ─────────────────────────────────────────────────────────────────────

/// Arming configuration. Defaults: threshold 0.5, halter patience 2, flat
/// cost tax 0.01 (the forward-wiring scale). A real consumer re-pins the
/// threshold on its OWN telemetry distribution — the defaults are the
/// fixture's posture, not a universal constant.
#[derive(Clone, Debug)]
pub struct ObstacleArmConfig {
    /// Oscillation-halt episode fraction at/above which the lane arms.
    pub oscillation_rate_threshold: f32,
    /// `oscillation_patience` handed to the halter (consecutive cos θ < 0
    /// steps before a Halt{Oscillation}).
    pub halter_patience: u8,
    /// Drift-cost flat tax as a fraction of the episode's first step size
    /// (the gain/cost scissors' cost side; the halter needs the full signal
    /// even though only the oscillation channel arms this lane).
    pub cost_flat_tax: f32,
}

impl Default for ObstacleArmConfig {
    fn default() -> Self {
        Self {
            oscillation_rate_threshold: 0.5,
            halter_patience: 2,
            cost_flat_tax: 0.01,
        }
    }
}

/// Why the lane refused to arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArmRefusal {
    /// Obstacle telemetry below threshold — the obstacle-vs-ability law's
    /// falsifier: a model with no single failing component is out of scope.
    NoObstacleFound {
        /// The measured rate (for the record).
        oscillation_rate: f32,
    },
}

/// The arming verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArmVerdict {
    Armed {
        oscillation_rate: f32,
        episodes: usize,
    },
    Refused(ArmRefusal),
}

/// Arm (or refuse) the lane from the eval protocol's obstacle telemetry.
///
/// `episodes` is the rollout protocol's per-episode hidden-state sequences
/// (`episodes[e][r]` = step `r`'s hidden vector). Each episode runs a fresh
/// [`GainCostLoopHalter`] fed: `gain = step_size(h_r, h_{r−1})`,
/// `cos θ = angular_change(Δh_r, Δh_{r−1})` (neutral 1.0 on the first
/// step), `cost = cost_flat_tax × first-step size`. An episode counts as
/// oscillating iff the halter fires `Halt { reason: Oscillation }`.
///
/// This is the shipped obstacle telemetry CONSUMED, not a parallel
/// detector — the substrate-first decision behind the whole lane.
pub fn arm_from_telemetry(episodes: &[Vec<Vec<f32>>], cfg: &ObstacleArmConfig) -> ArmVerdict {
    if episodes.is_empty() {
        return ArmVerdict::Refused(ArmRefusal::NoObstacleFound {
            oscillation_rate: 0.0,
        });
    }
    let mut oscillating = 0usize;
    for steps in episodes {
        if steps.len() < 3 {
            // Fewer than 3 hidden states cannot show a reversal.
            continue;
        }
        let dim = steps[0].len();
        let mut halter = GainCostLoopHalter::new(1.0, cfg.halter_patience, 1);
        let mut prev_step = vec![0.0f32; dim];
        let mut curr_step = vec![0.0f32; dim];
        let mut first_step_size = 0.0f32;
        let mut fired = false;
        for r in 1..steps.len() {
            let (h_prev, h_curr) = (&steps[r - 1], &steps[r]);
            let gain = step_size(h_curr, h_prev);
            if r == 1 {
                first_step_size = gain;
                for (cs, (a, b)) in curr_step.iter_mut().zip(h_curr.iter().zip(h_prev.iter())) {
                    *cs = a - b;
                }
                // Neutral alignment on the first step — no prior direction.
                let decision =
                    halter.halt_decision(r, gain, cfg.cost_flat_tax * first_step_size, 1.0);
                if matches!(decision, HaltDecision::Halt { .. }) {
                    fired = true;
                    break;
                }
                continue;
            }
            std::mem::swap(&mut prev_step, &mut curr_step);
            for (cs, (a, b)) in curr_step.iter_mut().zip(h_curr.iter().zip(h_prev.iter())) {
                *cs = a - b;
            }
            let cos = angular_change(&curr_step, &prev_step);
            let decision = halter.halt_decision(r, gain, cfg.cost_flat_tax * first_step_size, cos);
            if matches!(
                decision,
                HaltDecision::Halt {
                    reason: crate::gain_cost_halt::HaltReason::Oscillation
                }
            ) {
                fired = true;
                break;
            }
        }
        if fired {
            oscillating += 1;
        }
    }
    let rate = oscillating as f32 / episodes.len() as f32;
    if rate >= cfg.oscillation_rate_threshold {
        ArmVerdict::Armed {
            oscillation_rate: rate,
            episodes: episodes.len(),
        }
    } else {
        ArmVerdict::Refused(ArmRefusal::NoObstacleFound {
            oscillation_rate: rate,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Mint-once-then-freeze artifact envelope
// ─────────────────────────────────────────────────────────────────────

/// Authoring provenance. Every field is OPTIONAL — the artifact BLAKE3 is
/// the only mandatory commitment (Research 648 §4 item 7: prompt/seed/
/// proposer digests ride as provenance metadata; the pinned bytes are the
/// only source of truth).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProvenanceTrailer {
    /// Digest of the authoring prompt (opaque — the author's convention).
    pub prompt_digest: Option<[u8; 32]>,
    /// Authoring seed (which sample of a stochastic proposal this is).
    pub seed: Option<u64>,
    /// Author identity digest.
    pub author_id: Option<[u8; 32]>,
}

/// A minted, frozen artifact: BLAKE3-pinned overlay + provenance trailer.
/// Mint once, then freeze — no field mutates after [`Self::mint`]; the
/// digest chain (overlay → header → artifact) is verified at every load.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightEditArtifact {
    overlay_digest: [u8; 32],
    overlay_len: u64,
    artifact_digest: [u8; 32],
    provenance: ProvenanceTrailer,
}

impl WeightEditArtifact {
    /// Mint: digest the overlay's canonical bytes, fold into the header
    /// (with provenance), commit the header.
    pub fn mint(overlay: &WeightEditOverlay, provenance: ProvenanceTrailer) -> Self {
        let bytes = overlay.to_bytes();
        let overlay_digest = *blake3::hash(&bytes).as_bytes();
        let overlay_len = bytes.len() as u64;
        let mut artifact = Self {
            overlay_digest,
            overlay_len,
            artifact_digest: [0; 32],
            provenance,
        };
        artifact.artifact_digest = *blake3::hash(&artifact.header_preimage()).as_bytes();
        artifact
    }

    pub fn overlay_digest(&self) -> [u8; 32] {
        self.overlay_digest
    }

    pub fn provenance(&self) -> &ProvenanceTrailer {
        &self.provenance
    }

    pub fn artifact_digest(&self) -> [u8; 32] {
        self.artifact_digest
    }

    fn header_preimage(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC_ARTIFACT);
        buf.extend_from_slice(&self.overlay_digest);
        buf.extend_from_slice(&self.overlay_len.to_le_bytes());
        // Provenance: tag byte 0 = absent, 1 = present, then the payload.
        match self.provenance.prompt_digest {
            None => buf.push(0),
            Some(d) => {
                buf.push(1);
                buf.extend_from_slice(&d);
            }
        }
        match self.provenance.seed {
            None => buf.push(0),
            Some(s) => {
                buf.push(1);
                buf.extend_from_slice(&s.to_le_bytes());
            }
        }
        match self.provenance.author_id {
            None => buf.push(0),
            Some(a) => {
                buf.push(1);
                buf.extend_from_slice(&a);
            }
        }
        buf
    }

    /// Verify the artifact's own header commitment (the mint-once pin).
    pub fn verify_header(&self) -> bool {
        blake3::hash(&self.header_preimage()).as_bytes() == &self.artifact_digest
    }

    /// Verify a candidate overlay against the artifact (payload half:
    /// digest + exact length).
    pub fn verify_overlay(&self, overlay: &WeightEditOverlay) -> bool {
        let bytes = overlay.to_bytes();
        bytes.len() as u64 == self.overlay_len
            && blake3::hash(&bytes).as_bytes() == &self.overlay_digest
    }

    /// Serialize the artifact (envelope only; the overlay bytes travel
    /// alongside — the payload half is verified against this envelope at
    /// load).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC_ARTIFACT);
        buf.extend_from_slice(&self.overlay_digest);
        buf.extend_from_slice(&self.overlay_len.to_le_bytes());
        buf.extend_from_slice(&self.artifact_digest);
        match self.provenance.prompt_digest {
            None => buf.push(0),
            Some(d) => {
                buf.push(1);
                buf.extend_from_slice(&d);
            }
        }
        match self.provenance.seed {
            None => buf.push(0),
            Some(s) => {
                buf.push(1);
                buf.extend_from_slice(&s.to_le_bytes());
            }
        }
        match self.provenance.author_id {
            None => buf.push(0),
            Some(a) => {
                buf.push(1);
                buf.extend_from_slice(&a);
            }
        }
        buf
    }

    /// Thaw: parse + verify the header commitment. A flipped bit anywhere
    /// in the header/provenance region refuses loud (`None`) — the tamper
    /// refusal at load.
    pub fn thaw(bytes: &[u8]) -> Option<Self> {
        // MAGIC(4) + overlay_digest(32) + overlay_len(8) + artifact_digest(32)
        // + 3 provenance fields.
        if bytes.len() < 4 + 32 + 8 + 32 + 3 {
            return None;
        }
        if bytes[0..4] != MAGIC_ARTIFACT {
            return None;
        }
        let mut pos = 4;
        let mut overlay_digest = [0u8; 32];
        overlay_digest.copy_from_slice(&bytes[pos..pos + 32]);
        pos += 32;
        let mut len_b = [0u8; 8];
        len_b.copy_from_slice(&bytes[pos..pos + 8]);
        let overlay_len = u64::from_le_bytes(len_b);
        pos += 8;
        let mut artifact_digest = [0u8; 32];
        artifact_digest.copy_from_slice(&bytes[pos..pos + 32]);
        pos += 32;
        // Optional field reader: tag 0 = absent, tag 1 = 32-byte digest.
        let opt_digest = |bytes: &[u8], pos: &mut usize| -> Option<Option<[u8; 32]>> {
            if bytes.len() < *pos + 1 {
                return None;
            }
            match bytes[*pos] {
                0 => {
                    *pos += 1;
                    Some(None)
                }
                1 => {
                    *pos += 1;
                    if bytes.len() < *pos + 32 {
                        return None;
                    }
                    let mut d = [0u8; 32];
                    d.copy_from_slice(&bytes[*pos..*pos + 32]);
                    *pos += 32;
                    Some(Some(d))
                }
                _ => None,
            }
        };
        let prompt_digest = opt_digest(bytes, &mut pos)?;
        // Seed slot: tag 0 = absent, tag 1 = u64 LE.
        if bytes.len() < pos + 1 {
            return None;
        }
        let seed = match bytes[pos] {
            0 => {
                pos += 1;
                None
            }
            1 => {
                pos += 1;
                if bytes.len() < pos + 8 {
                    return None;
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[pos..pos + 8]);
                pos += 8;
                Some(u64::from_le_bytes(b))
            }
            _ => return None,
        };
        let author_id = opt_digest(bytes, &mut pos)?;
        let artifact = Self {
            overlay_digest,
            overlay_len,
            artifact_digest,
            provenance: ProvenanceTrailer {
                prompt_digest,
                seed,
                author_id,
            },
        };
        if !artifact.verify_header() {
            return None;
        }
        Some(artifact)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Battery stats + seeded direction generation (RandOpt posture)
// ─────────────────────────────────────────────────────────────────────

/// Paired lower confidence bound at 95% (normal approximation):
/// `mean − 1.6449·sd/√n`. `None` on empty or single-sample input. This is
/// the pre-registered statistic for the targeted-vs-best-random arm: the
/// best random arm is selected ONCE on its mean, then paired per-seed
/// deltas are computed against that single arm.
pub fn paired_lcb95(deltas: &[f32]) -> Option<f32> {
    let n = deltas.len();
    if n < 2 {
        return None;
    }
    let mut sum = 0.0f64;
    for d in deltas {
        sum += *d as f64;
    }
    let mean = sum / n as f64;
    let mut var = 0.0f64;
    for d in deltas {
        let x = *d as f64 - mean;
        var += x * x;
    }
    let sd = (var / (n - 1) as f64).sqrt();
    let sd32 = if sd.is_finite() {
        sd as f32
    } else {
        f32::INFINITY
    };
    Some(mean as f32 - 1.6449f32 * sd32 / (n as f32).sqrt())
}

/// Deterministic BLAKE3-derived seed: first 8 bytes of
/// `blake3(domain ‖ index LE)` as u64. The control battery's arm seeds
/// (`≥16` BLAKE3-seeded arms, RandOpt posture) come from here.
pub fn blake3_seed(domain: &[u8], index: u64) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&index.to_le_bytes());
    let mut b = [0u8; 8];
    b.copy_from_slice(&h.finalize().as_bytes()[..8]);
    u64::from_le_bytes(b)
}

/// Module-local SplitMix64 (the established substrate-side module-local
/// pattern; deterministic, allocation-free).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_unit_f32(&mut self) -> f32 {
        // 24-bit uniform in [0, 1) → [−1, 1).
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Draw a deterministic unit-norm direction from `seed` into `out`
/// (uniform in [−1,1)^dim, then L2-normalized; the all-zero draw falls
/// back to `e₀`). Norm-matching to a targeted direction is the caller's
/// assertion (`‖u‖ == ‖d‖` to f32 rounding) — both are unit vectors by
/// construction. Zero allocation.
pub fn norm_matched_random_direction(seed: u64, dim: usize, out: &mut [f32]) {
    assert_eq!(out.len(), dim, "out must hold dim entries");
    let mut rng = SplitMix64::new(seed);
    let mut norm_sq = 0.0f32;
    for slot in out.iter_mut() {
        let v = rng.next_unit_f32();
        norm_sq += v * v;
        *slot = v;
    }
    if norm_sq == 0.0 {
        // Deterministic fallback: astronomically unlikely, but the lane
        // never emits a zero direction.
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = if i == 0 { 1.0 } else { 0.0 };
        }
        return;
    }
    let norm = norm_sq.sqrt();
    for slot in out.iter_mut() {
        *slot /= norm;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// LE byte-view hash of an f32 slice (BLAKE3 pinning helper).
    fn f32_digest(v: &[f32]) -> blake3::Hash {
        let mut h = blake3::Hasher::new();
        for x in v {
            h.update(&x.to_le_bytes());
        }
        h.finalize()
    }

    fn direction_edit(layer: u8, cols: &[u32], out_dim: usize, strength: f32) -> WeightEditSpec {
        WeightEditSpec {
            layer,
            form: EditForm::DirectionSubtraction {
                columns: cols.to_vec(),
                direction: vec![0.5; out_dim],
                strength,
            },
        }
    }

    #[test]
    fn overlay_roundtrip_and_digest_stability() {
        let overlay = WeightEditOverlay::new(vec![
            direction_edit(1, &[4], 8, 2.0),
            WeightEditSpec {
                layer: 2,
                form: EditForm::BypassRows {
                    gate_columns: vec![0, 1],
                    gate_weights: vec![0.3, -0.7],
                    gate_bias: 0.1,
                    pathway_columns: vec![2, 3],
                    pathway_readout: (0..16).map(|i| i as f32 * 0.01).collect(),
                    out_dim: 8,
                },
            },
        ])
        .expect("valid overlay");
        let bytes = overlay.to_bytes();
        let decoded = WeightEditOverlay::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(decoded, overlay);
        assert_eq!(decoded.digest(), overlay.digest());
        // Digest is over canonical bytes: identical content → identical digest.
        assert_eq!(blake3::hash(&bytes).as_bytes()[..], overlay.digest()[..]);
    }

    #[test]
    fn decode_refuses_corrupt_inputs() {
        let overlay = WeightEditOverlay::new(vec![direction_edit(1, &[4], 8, 2.0)]).unwrap();
        let bytes = overlay.to_bytes();
        // Bad magic.
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(WeightEditOverlay::from_bytes(&bad).is_none());
        // Truncated.
        assert!(WeightEditOverlay::from_bytes(&bytes[..bytes.len() - 2]).is_none());
        // Trailing garbage appended: parse stops at edit count — still
        // decodes, and the digest is IDENTICAL: digest() hashes the
        // CANONICAL re-encoding, so transport-level trailing noise is
        // discarded (the envelope pins the canonical form).
        let mut extra = bytes.clone();
        extra.push(0xAB);
        assert!(WeightEditOverlay::from_bytes(&extra).is_some());
        assert_eq!(
            WeightEditOverlay::from_bytes(&extra).unwrap().digest(),
            overlay.digest()
        );
        // Non-finite strength refuses at construction.
        let bad_spec = WeightEditSpec {
            layer: 1,
            form: EditForm::DirectionSubtraction {
                columns: vec![4],
                direction: vec![0.5; 8],
                strength: f32::NAN,
            },
        };
        assert!(WeightEditOverlay::new(vec![bad_spec]).is_none());
    }

    #[test]
    fn compose_revert_bit_identical() {
        let out_dim = 16;
        let in_dim = 8;
        let mut w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let overlay = WeightEditOverlay::new(vec![
            direction_edit(1, &[4], out_dim, 1.5),
            direction_edit(1, &[4, 2], out_dim, 0.25), // overlaps col 4
        ])
        .unwrap();
        let mut backup = vec![0.0f32; overlay.touched_count(1, out_dim, in_dim)];
        let mut baseline = w.clone();
        let used = overlay
            .compose_layer_into(1, &mut w, out_dim, in_dim, &mut backup)
            .expect("backup sized");
        // Only S-column elements changed; untouched columns bit-identical.
        for i in 0..out_dim {
            for j in 0..in_dim {
                if j == 4 || j == 2 {
                    continue;
                }
                assert_eq!(
                    w[i * in_dim + j],
                    baseline[i * in_dim + j],
                    "col {j} untouched"
                );
            }
        }
        // Revert → bit-identical (BLAKE3 before/after).
        let before = f32_digest(&baseline);
        overlay.revert_layer(1, &mut w, out_dim, in_dim, &backup);
        let after = f32_digest(&w);
        assert_eq!(
            before.to_hex(),
            after.to_hex(),
            "revert must be bit-identical"
        );
        assert_eq!(f32_digest(&baseline).to_hex(), f32_digest(&w).to_hex());
        let _ = used;
        let _ = &mut baseline;
    }

    #[test]
    fn compose_too_small_backup_refuses() {
        let out_dim = 4;
        let in_dim = 4;
        let mut w = vec![1.0f32; out_dim * in_dim];
        let overlay = WeightEditOverlay::new(vec![direction_edit(0, &[1], out_dim, 1.0)]).unwrap();
        let mut tiny = vec![0.0f32; out_dim - 1];
        assert!(
            overlay
                .compose_layer_into(0, &mut w, out_dim, in_dim, &mut tiny)
                .is_none()
        );
    }

    #[test]
    fn forward_delta_matches_composed_forward() {
        // The rank-1 forward delta equals the composed-matrix forward up to
        // f32 rounding of the changed term's placement.
        let out_dim = 12;
        let in_dim = 6;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let k = 3.0f32;
        let overlay = WeightEditOverlay::new(vec![direction_edit(0, &[2, 5], out_dim, k)]).unwrap();
        let h: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.21).cos()).collect();
        // Composed forward.
        let mut w_edited = w.clone();
        let mut backup = vec![0.0f32; overlay.touched_count(0, out_dim, in_dim)];
        overlay
            .compose_layer_into(0, &mut w_edited, out_dim, in_dim, &mut backup)
            .unwrap();
        let mut y_composed = vec![0.0f32; out_dim];
        for i in 0..out_dim {
            let mut acc = 0.0f32;
            for j in 0..in_dim {
                acc += w_edited[i * in_dim + j] * h[j];
            }
            y_composed[i] = acc;
        }
        // Delta forward.
        let mut y_delta = vec![0.0f32; out_dim];
        for i in 0..out_dim {
            let mut acc = 0.0f32;
            for j in 0..in_dim {
                acc += w[i * in_dim + j] * h[j];
            }
            y_delta[i] = acc;
        }
        overlay.forward_delta_into(0, &h, &mut y_delta);
        for i in 0..out_dim {
            let scale = y_composed[i].abs().max(1.0);
            assert!(
                (y_composed[i] - y_delta[i]).abs() / scale < 1e-4,
                "row {i}: composed {} vs delta {}",
                y_composed[i],
                y_delta[i]
            );
        }
    }

    #[test]
    fn forward_delta_zero_carrier_is_bit_exact_noop() {
        let out_dim = 4;
        let h = vec![0.0f32; 6]; // carrier column mass exactly zero
        let mut y = vec![1.5f32; out_dim];
        let y_before = y.clone();
        let overlay = WeightEditOverlay::new(vec![direction_edit(0, &[2], out_dim, 9.0)]).unwrap();
        overlay.forward_delta_into(0, &h, &mut y);
        assert_eq!(y, y_before, "zero carrier mass must not move y at all");
    }

    #[test]
    fn bypass_gate_saturates_and_zero_gate_skips() {
        let out_dim = 4;
        let edit = WeightEditSpec {
            layer: 0,
            form: EditForm::BypassRows {
                gate_columns: vec![0],
                gate_weights: vec![10.0],
                gate_bias: 0.0,
                pathway_columns: vec![1],
                pathway_readout: vec![1.0, 1.0, 1.0, 1.0],
                out_dim: out_dim as u32,
            },
        };
        let overlay = WeightEditOverlay::new(vec![edit]).unwrap();
        // Positive gate input → g ≈ 1 → full pathway.
        let h_pos = vec![3.0f32, 2.0];
        let mut y = vec![0.0f32; out_dim];
        overlay.bypass_forward_into(0, &h_pos, &mut y);
        for v in &y {
            assert!((v - 2.0).abs() < 1e-5, "gate ≈1 → y ≈ pathway value");
        }
        // Negative gate input → g ≈ 0 → bit-exact skip.
        let h_neg = vec![-3.0f32, 2.0];
        let mut y2 = vec![7.0f32; out_dim];
        let before = y2.clone();
        overlay.bypass_forward_into(0, &h_neg, &mut y2);
        assert_eq!(y2, before, "g == 0 must skip writes bit-exactly");
    }

    #[test]
    fn arm_gate_fires_on_oscillation_refuses_on_monotone() {
        let cfg = ObstacleArmConfig::default();
        // Oscillating: steps alternate ±e.
        let osc: Vec<Vec<f32>> = (0..10)
            .map(|r| {
                let mut v = vec![0.0f32; 4];
                v[0] = if r % 2 == 0 { 1.0 } else { -1.0 };
                v
            })
            .collect();
        // Monotone: steps march in +e.
        let mono: Vec<Vec<f32>> = (0..10)
            .map(|r| {
                let mut v = vec![0.0f32; 4];
                v[0] = r as f32;
                v
            })
            .collect();
        let armed = arm_from_telemetry(&[osc], &cfg);
        assert!(
            matches!(armed, ArmVerdict::Armed { .. }),
            "alternating steps must arm: {:?}",
            armed
        );
        let refused = arm_from_telemetry(&[mono], &cfg);
        assert_eq!(
            refused,
            ArmVerdict::Refused(ArmRefusal::NoObstacleFound {
                oscillation_rate: 0.0
            })
        );
        // Empty telemetry refuses too.
        assert!(matches!(
            arm_from_telemetry(&[], &cfg),
            ArmVerdict::Refused(_)
        ));
    }

    #[test]
    fn envelope_mint_verify_tamper_thaw() {
        let overlay = WeightEditOverlay::new(vec![direction_edit(1, &[4], 8, 2.0)]).unwrap();
        let artifact = WeightEditArtifact::mint(
            &overlay,
            ProvenanceTrailer {
                prompt_digest: Some([7u8; 32]),
                seed: Some(42),
                author_id: Some([9u8; 32]),
            },
        );
        assert!(artifact.verify_header());
        assert!(artifact.verify_overlay(&overlay));
        // Payload tamper: a different overlay must fail verification.
        let other = WeightEditOverlay::new(vec![direction_edit(1, &[4], 8, 2.5)]).unwrap();
        assert!(!artifact.verify_overlay(&other));
        // Provenance optional: full absence verifies (artifact BLAKE3 is
        // the only mandatory commitment).
        let bare = WeightEditArtifact::mint(&overlay, ProvenanceTrailer::default());
        assert!(bare.verify_header());
        // Thaw roundtrip.
        let thawed = WeightEditArtifact::thaw(&artifact.to_bytes()).expect("thaw");
        assert_eq!(thawed, artifact);
        assert!(thawed.verify_header());
        // Tamper refusal: flip a byte in the artifact envelope.
        let mut tampered = artifact.to_bytes();
        tampered[10] ^= 0x01;
        assert!(
            WeightEditArtifact::thaw(&tampered).is_none(),
            "tamper must refuse"
        );
        // Truncated envelope refuses.
        let full = artifact.to_bytes();
        assert!(WeightEditArtifact::thaw(&full[..full.len() / 2]).is_none());
    }

    #[test]
    fn paired_lcb95_known_values() {
        // Constant deltas: sd = 0 → LCB = mean.
        let d = vec![1.0f32; 16];
        assert!((paired_lcb95(&d).unwrap() - 1.0).abs() < 1e-5);
        // n = 1 refuses.
        assert!(paired_lcb95(&[1.0]).is_none());
        assert!(paired_lcb95(&[]).is_none());
        // Spread: LCB strictly below mean.
        let d = vec![2.0f32, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
        let lcb = paired_lcb95(&d).unwrap();
        assert!(lcb < 1.0, "lcb {lcb} must sit below the mean 1.0");
    }

    #[test]
    fn seeded_directions_deterministic_and_norm_matched() {
        let dim = 64;
        let mut a = vec![0.0f32; dim];
        let mut b = vec![0.0f32; dim];
        let seed = blake3_seed(b"weight_edit_overlay/control", 0);
        norm_matched_random_direction(seed, dim, &mut a);
        norm_matched_random_direction(seed, dim, &mut b);
        assert_eq!(a, b, "same seed → same direction");
        norm_matched_random_direction(blake3_seed(b"weight_edit_overlay/control", 1), dim, &mut b);
        assert_ne!(a, b, "different seed → different direction");
        for v in &a {
            assert!(v.is_finite());
        }
        let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "unit norm, got {norm}");
    }
}
