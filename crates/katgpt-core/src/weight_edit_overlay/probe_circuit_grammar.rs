//! probe_circuit_grammar — probe-authored weight-edit derivation.
//!
//! **FOLDED (2026-11-05): the first consumer PoC failed its value gate.** The
//! twinarm linear-head consumer (riir-instinct Issue 028 / Research 653) ran
//! the pre-registered battery on the corrupted-teacher fixture: the repair beat
//! its unrepaired base (Δ +0.2400, LCB95 +0.1902) but LOST to fair b1
//! continue-train-on-gold (eval 0.9950 vs 1.0000, Δ −0.0050, LCB95 −0.0132) —
//! the fold law the feature row pre-registered fires. Record:
//! `../riir-instinct/.benchmarks/0064_g1_probe_repair_fixture/G1.md` (plus the
//! class-2 census negative, record 0063). The lane has FOLDED into
//! [`crate::weight_edit_overlay`] per its own demotion law: this is now a
//! submodule of the carrier, compiled whenever `weight_edit_overlay` is on —
//! there is no separate feature (the 2026-11-06 housekeeping merge). The
//! machinery + tests stay as the attribution record. A revival is a NEW
//! pre-registration with a fixture design that separates the repair from
//! continue-train (scarcer gold, broader poison).
//!
//! Research 649 (fusion on Research 648 / Issue 980): the 648 lane's author
//! was a reasoning model (or a hand that already knew the fixture); this
//! module replaces that author with **activation probes** — the lane peeks
//! at which neurons activate, parses the failing circuit into a
//! **fn-grammar** record (drive → wire → gap, with the band gate), and emits
//! the repair as a **params-vocab** sentence over the shipped
//! [`crate::weight_edit_overlay`] edit forms, strength-selected by
//! behavior verification (the dose sweep), never by a gradient.
//!
//! # The fn-grammar (the parse the probes author)
//!
//! ```text
//! fail ↚ drive[src[j] | gate |·| ≤ b] —wire(W[:, j], slope ŝ)→ gap[row o]
//! ```
//!
//! * **drive** — the source-layer activation column whose magnitude is
//!   band-gated in the gate variable (hot inside a band, cold outside).
//!   Found by a one-sided edge-contrast scan over (column, band) pairs.
//! * **gap** — the suspect layer's output row that carries the failing
//!   decision margin. Read from the readout rows the caller supplies (the
//!   row of `logit(+axis) − logit(−axis)`): `argmax |u|`, first-index tie.
//! * **wire** — the anomalous coupling arc: the least-squares slope of
//!   `dst_pre[gap]` on `src[drive]` pooled over captured steps. The wire is
//!   ranked across drive candidates by `|slope|`.
//!
//! # The params-vocab (the sentence the grammar emits)
//!
//! The vocabulary is the shipped `EditForm` set; the sentence for a parsed
//! arc is `DirectionSubtraction { columns: [j], direction: sign(ŝ)·e_o,
//! strength: k }` — and because the probe slope is an ESTIMATE (single-
//! regressor LS inherits confounds from legitimately y-driven channels),
//! the strength is never taken from the slope directly: the emission sweeps
//! the pinned grid and accepts the smallest `k` whose overlay clears a
//! caller-supplied behavior bar on an authoring eval set. **The grammar
//! proposes; behavior disposes** — the 648 accept/reject harness at the
//! word level.
//!
//! # Lineage + prior art (searched 2026-11-04, two sweeps)
//!
//! Components are published; the composition is not. Closest, with the
//! deltas that keep this lane its own thing:
//!
//! * **AI-Lancet** (ACM CCS 2021) — activation-diff localization + a
//!   discrete sign-flip edit, gradient-free. No grammar record, no
//!   vocabulary, no dose/behavior gate, CNN classifiers only.
//! * **ROME** (NeurIPS 2022) — causal-trace localization + a closed-form
//!   rank-one write. The edit is SOLVED per edit from data; here it is
//!   EMITTED from a discrete vocabulary at a pinned grid, and the goal is
//!   behavior repair, not fact insertion.
//! * **Compiling Activation Steering into Weights** (arXiv:2604.12359,
//!   ACL 2026) — a class-level activation diff compiled into a null-space
//!   weight edit, modelless. No neuron-level localization, no circuit
//!   record, no repair gate; the demonstrated objective is backdoor
//!   injection, not repair.
//! * **INNER** (ISSTA 2024) — probes + clean-vs-failing diffs + neuron
//!   localization, but repairs by masked gradient descent.
//! * **ASPD** (arXiv:2609.37731) — weight components with read/write roles
//!   composed into mechanism circuits; the decomposition is learned, the
//!   edits are causal experiments, there is no probe-diff parse and no
//!   emission pipeline.
//! * **RASP/Tracr/ALTA** — a grammar whose sentences become weights, but
//!   the direction is inverted (program → weights on hand-built models);
//!   no probe-derived roles, no checkpoint-edit semantics.
//!
//! The unpublished composition this module files: probe-localized circuit →
//! role-grammar record → discrete-vocab emission → deterministic overlay
//! apply → dose/behavior battery, end to end with no LLM and no gradient
//! anywhere.
//!
//! # Honest scope
//!
//! * Single-arc circuits (`drive → wire → gap`) are what the parse covers;
//!   multi-arc / multi-layer circuits are future work (the binding record
//!   is the extension point).
//! * The gate scan assumes ONE scalar gate variable (`gate_var`); a
//!   multidimensional gate region is future work.
//! * The twin policy is NOT an input — pools split by the policy's own
//!   outcome (failed vs succeeded episodes of the same model). The 980
//!   arming precondition ([`crate::weight_edit_overlay::arm_from_telemetry`])
//!   remains the lane's gate: a model with no obstacle telemetry refuses
//!   before the parse runs (the obstacle-vs-ability law).
//! * Authoring-time lane: allocation here is not a hot-path constraint
//!   (the apply path's G4 already ships in 980). Determinism still binds:
//!   sequential folds, first-index tie rules, no hash iteration.
//!
//! No feature of its own: compiled whenever `weight_edit_overlay` is on
//! (the 2026-11-06 fold — there is no separate `probe_circuit_grammar`
//! feature anymore).

use super::{EditForm, WeightEditOverlay, WeightEditSpec};

// ─────────────────────────────────────────────────────────────────────
// Capture types
// ─────────────────────────────────────────────────────────────────────

/// One captured step of a rollout with activation taps.
///
/// `src` is the suspect layer's INPUT activation vector (e.g. the previous
/// hidden state), `dst_pre` its OUTPUT PRE-activation vector (before the
/// nonlinearity — the linear coupling the wire slope reads lives there),
/// and `gate_var` the scalar environment variable the band gate is scanned
/// over (a world coordinate, a progress counter — caller's choice).
#[derive(Clone, Debug)]
pub struct StepTap {
    pub gate_var: f32,
    pub src: Vec<f32>,
    pub dst_pre: Vec<f32>,
}

/// Outcome-split capture pools: steps from episodes of the SAME policy that
/// ended failed vs succeeded. Twin-free by construction — the split is the
/// policy's own behavior, never a reference model.
#[derive(Clone, Debug, Default)]
pub struct ProbePools {
    pub failed: Vec<StepTap>,
    pub succeeded: Vec<StepTap>,
}

impl ProbePools {
    fn for_each_step<F: FnMut(&StepTap)>(&self, mut f: F) {
        for t in &self.failed {
            f(t);
        }
        for t in &self.succeeded {
            f(t);
        }
    }

    pub fn total_steps(&self) -> usize {
        self.failed.len() + self.succeeded.len()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate scan (the drive role)
// ─────────────────────────────────────────────────────────────────────

/// Gate-scan configuration.
#[derive(Clone, Debug)]
pub struct GateConfig {
    /// Candidate band half-widths for the gate variable.
    pub band_grid: Vec<f32>,
    /// Out-band half-window beyond the band edge (the edge contrast reads
    /// `(b, b + out_window]` against `[0, b]`).
    pub out_window: f32,
    /// Acceptance bar on the normalized one-sided contrast.
    pub z_bar: f32,
    /// Variance floor (never divide by a degenerate spread).
    pub eps: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            band_grid: vec![2.0, 3.0, 4.0, 5.0, 6.0],
            out_window: 2.0,
            z_bar: 1.0,
            eps: 1e-6,
        }
    }
}

/// One accepted drive candidate: column, its best band, the contrast z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveCandidate {
    pub col: u32,
    pub band: f32,
    pub z: f32,
}

/// Full scan: for every source column, the best band's normalized contrast.
/// Returns candidates above `z_bar`, sorted by z descending, ties by lower
/// column index first (determinism law).
///
/// The normalization divides the one-sided contrast by the pooled magnitude
/// spread `(rms + eps)` — a HOT feature with a large absolute scale still
/// has to show a real IN-vs-OUT jump to pass, which is what separates a
/// band-gated drive from a merely large feature.
pub fn discover_gated_drive(
    pools: &ProbePools,
    src_dim: usize,
    cfg: &GateConfig,
) -> Vec<DriveCandidate> {
    let mut out = Vec::new();
    for col in 0..src_dim {
        let mut best = DriveCandidate {
            col: col as u32,
            band: 0.0,
            z: f32::NEG_INFINITY,
        };
        for &band in &cfg.band_grid {
            let z = normalized_contrast(pools, col, band, cfg);
            if z > best.z {
                best = DriveCandidate {
                    col: col as u32,
                    band,
                    z,
                };
            }
        }
        if best.z >= cfg.z_bar {
            out.push(best);
        }
    }
    out.sort_by(|a, b| {
        b.z.partial_cmp(&a.z)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.col.cmp(&b.col))
    });
    out
}

/// Normalized one-sided contrast at (col, band): `(mean_in − mean_out) /
/// (pooled rms + eps)`. `NEG_INFINITY` when either pool is empty.
fn normalized_contrast(pools: &ProbePools, col: usize, band: f32, cfg: &GateConfig) -> f32 {
    let mut in_sum = 0.0f32;
    let mut in_sq = 0.0f32;
    let mut in_n = 0usize;
    let mut out_sum = 0.0f32;
    let mut out_sq = 0.0f32;
    let mut out_n = 0usize;
    pools.for_each_step(|t| {
        let a = t.src.get(col).copied().unwrap_or(0.0).abs();
        let g = t.gate_var.abs();
        if g <= band {
            in_sum += a;
            in_sq += a * a;
            in_n += 1;
        } else if g <= band + cfg.out_window {
            out_sum += a;
            out_sq += a * a;
            out_n += 1;
        }
    });
    if in_n == 0 || out_n == 0 {
        return f32::NEG_INFINITY;
    }
    let mean_in = in_sum / in_n as f32;
    let mean_out = out_sum / out_n as f32;
    let pooled_n = (in_n + out_n) as f32;
    let rms = ((in_sq + out_sq) / pooled_n).max(0.0).sqrt();
    (mean_in - mean_out) / (rms + cfg.eps)
}

// ─────────────────────────────────────────────────────────────────────
// Gap role + wire slope
// ─────────────────────────────────────────────────────────────────────

/// The gap row: `argmax |u|` over the caller-supplied readout difference
/// row (`logit(+axis) − logit(−axis)` dotted with the layer output).
/// First-index tie rule (determinism law). `None` when the row is all
/// zeros — no gap axis on this readout.
pub fn discover_gap_row(readout_diff: &[f32]) -> Option<usize> {
    let mut best = 0usize;
    let mut best_abs = 0.0f32;
    let mut any = false;
    for (i, &u) in readout_diff.iter().enumerate() {
        let a = u.abs();
        if !any || a > best_abs {
            best_abs = a;
            best = i;
            any = true;
        }
    }
    if any && best_abs > 0.0 {
        Some(best)
    } else {
        None
    }
}

/// The wire slope: least-squares slope of `dst_pre[gap_row]` on
/// `src[col]`, pooled over ALL captured steps (both pools). Sequential
/// folds; `0.0` when the drive column never varies (degenerate capture).
///
/// HONEST CAVEAT: a single-regressor slope inherits confounds from any
/// other source column correlated with the drive (legitimately y-driven
/// channels, in the fixture). The slope LOCALIZES the arc and carries its
/// approximate strength; the emitted word's strength is chosen by the
/// behavior-verified dose sweep, never by this estimate alone.
pub fn measure_wire_slope(pools: &ProbePools, col: usize, gap_row: usize) -> f32 {
    let mut n = 0usize;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut sxy = 0.0f32;
    pools.for_each_step(|t| {
        let x = t.src.get(col).copied().unwrap_or(0.0);
        let y = t.dst_pre.get(gap_row).copied().unwrap_or(0.0);
        n += 1;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    });
    if n == 0 {
        return 0.0;
    }
    let nf = n as f32;
    let var = sxx - sx * sx / nf;
    if var <= 1e-9 {
        return 0.0;
    }
    (sxy - sx * sy / nf) / var
}

// ─────────────────────────────────────────────────────────────────────
// The fn-grammar record
// ─────────────────────────────────────────────────────────────────────

/// Parse refusals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeRefusal {
    /// No source column cleared the gate bar (or the pools were empty).
    NoObstacleFound,
    /// The readout difference row named no gap row.
    NoGapRow,
    /// No wire slope among the drive candidates was usable.
    NoWire,
    /// No strength on the grid cleared the behavior bar.
    NoViableStrength,
}

/// The parsed single-arc circuit — the fn-grammar record:
/// `fail ↚ drive[src[j] | gate |·| ≤ b] —wire(W[:, j], ŝ)→ gap[row o]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CircuitBinding {
    /// Source-layer drive column (`j`).
    pub drive_col: u32,
    /// Suspect-layer output gap row (`o`).
    pub gap_row: usize,
    /// Accepted band half-width (`b`).
    pub band: f32,
    /// The accepted drive's normalized contrast.
    pub gate_z: f32,
    /// The wire slope estimate (`ŝ`).
    pub slope: f32,
}

impl CircuitBinding {
    /// The one-line fn-grammar rendering (stable field order — the digest
    /// pins it).
    pub fn grammar_line(&self) -> String {
        format!(
            "fail <- drive[src[{}]|gate |x|<={}] --wire(W[:,{}], slope {:.4})-> gap[row {}]",
            self.drive_col, self.band, self.drive_col, self.slope, self.gap_row
        )
    }

    /// BLAKE3 over the stable rendering — the binding's identity for
    /// freeze/provenance use.
    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(self.grammar_line().as_bytes()).as_bytes()
    }
}

/// Parse the circuit from outcome-split pools.
///
/// Pipeline: gate scan → gap row → wire ranking. Refuses loud at each
/// stage (the 980 arming precondition runs BEFORE this — a model with no
/// obstacle telemetry never reaches the parse).
pub fn parse_circuit(
    pools: &ProbePools,
    src_dim: usize,
    cfg: &GateConfig,
    readout_diff: &[f32],
) -> Result<CircuitBinding, ProbeRefusal> {
    if pools.total_steps() == 0 {
        return Err(ProbeRefusal::NoObstacleFound);
    }
    let candidates = discover_gated_drive(pools, src_dim, cfg);
    let drive = candidates
        .first()
        .copied()
        .ok_or(ProbeRefusal::NoObstacleFound)?;
    let gap_row = discover_gap_row(readout_diff).ok_or(ProbeRefusal::NoGapRow)?;
    // Wire ranking across ALL candidates (not just the top-z one): the
    // gated feature set may carry benign passengers; the anomalous arc is
    // the strongest coupling to the gap row. First-index tie on |slope|.
    let mut best_col = drive.col as usize;
    let mut best_slope = measure_wire_slope(pools, best_col, gap_row);
    for cand in candidates.iter().skip(1) {
        let s = measure_wire_slope(pools, cand.col as usize, gap_row);
        if s.abs() > best_slope.abs() {
            best_slope = s;
            best_col = cand.col as usize;
        }
    }
    if best_slope.abs() <= cfg.eps {
        return Err(ProbeRefusal::NoWire);
    }
    // The binding carries the DRIVE STAGE's band evidence; when the wire
    // winner is a different candidate, re-read that candidate's own scan
    // row so the record stays honest about which (col, band) it names.
    let band = if best_col == drive.col as usize {
        drive.band
    } else {
        candidates
            .iter()
            .find(|c| c.col as usize == best_col)
            .map(|c| c.band)
            .unwrap_or(drive.band)
    };
    let gate_z = if best_col == drive.col as usize {
        drive.z
    } else {
        candidates
            .iter()
            .find(|c| c.col as usize == best_col)
            .map(|c| c.z)
            .unwrap_or(drive.z)
    };
    Ok(CircuitBinding {
        drive_col: best_col as u32,
        gap_row,
        band,
        gate_z,
        slope: best_slope,
    })
}

// ─────────────────────────────────────────────────────────────────────
// The params-vocab emission
// ─────────────────────────────────────────────────────────────────────

/// The vocabulary word for a parsed arc at an explicit strength — the
/// caller-facing primitive (compose your own dose sweep if you want a
/// different acceptance rule).
///
/// `direction[gap_row] = sign(slope)`, everything else 0 (unit norm by
/// construction). The overlay applies `W[i][col] -= k·d[i]`, so a NEGATIVE
/// slope (the drive pushes the gap down) yields `d[gap] = −1` and the edit
/// ADDS `+k` to the anomalous row — canceling toward the healthy coupling.
pub fn binding_to_spec(
    binding: &CircuitBinding,
    layer: u8,
    out_dim: usize,
    strength: f32,
) -> Option<WeightEditSpec> {
    if binding.gap_row >= out_dim || !strength.is_finite() || strength <= 0.0 {
        return None;
    }
    let sign = if binding.slope < 0.0 {
        -1.0f32
    } else if binding.slope > 0.0 {
        1.0
    } else {
        return None;
    };
    let mut direction = vec![0.0f32; out_dim];
    direction[binding.gap_row] = sign;
    Some(WeightEditSpec {
        layer,
        form: EditForm::DirectionSubtraction {
            columns: vec![binding.drive_col],
            direction,
            strength,
        },
    })
}

/// Behavior-verified strength selection: sweep the grid ascending, build
/// the word at each `k`, and accept the FIRST overlay whose authoring-eval
/// rate clears `accept_bar`. Returns the accepted `(spec, k)`.
///
/// `authoring_eval` receives the built overlay and returns a scalar cost
/// (an obstacle rate, an error rate — caller's statistic) on the caller's
/// OWN authoring set. This is the grammar-proposes/behavior-disposes half:
/// the probe slope is an estimate; the behavior is the referee.
pub fn emit_word<F>(
    binding: &CircuitBinding,
    layer: u8,
    out_dim: usize,
    strength_grid: &[f32],
    accept_bar: f32,
    mut authoring_eval: F,
) -> Result<(WeightEditSpec, f32), ProbeRefusal>
where
    F: FnMut(&WeightEditOverlay) -> f32,
{
    let mut grid: Vec<f32> = strength_grid.to_vec();
    grid.retain(|k| k.is_finite() && *k > 0.0);
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    grid.dedup();
    for &k in &grid {
        let Some(spec) = binding_to_spec(binding, layer, out_dim, k) else {
            continue;
        };
        let Some(overlay) = WeightEditOverlay::new(vec![spec.clone()]) else {
            continue;
        };
        if authoring_eval(&overlay) <= accept_bar {
            return Ok((spec, k));
        }
    }
    Err(ProbeRefusal::NoViableStrength)
}

/// Derive the strength grid FROM the probe: a window of integer strengths
/// centred on the slope estimate — `[max(1, ⌊|ŝ|⌋ − margin) ..= ⌈|ŝ|⌉ +
/// margin]`, `margin ≥ 1`, intersected with the hard ceiling `cap`. The
/// slope LOCALIZES the window; the dose sweep inside it is the referee.
/// Never trusts `|ŝ|` alone (Research 649: the estimate is confounded —
/// the fixture's floor negative k=7-at-ŝ−7.52 oscillates 8/8).
///
/// Cap semantics: an EMPTY return is the loud refusal (window floor above
/// `cap` — no allowed strength can reach the anomaly, the caller's sweep
/// refuses `NoViableStrength`). A window whose TOP reaches `cap` may have
/// been TRUNCATED there — the caller sees the grid but not the truncation
/// (no marker on the bare `Vec<f32>`); if that ambiguity matters, pass a
/// larger `cap`.
pub fn derive_strength_grid(slope: f32, margin: usize, cap: f32) -> Vec<f32> {
    if !slope.is_finite() || slope == 0.0 || margin == 0 {
        return Vec::new();
    }
    let center = slope.abs();
    let low = ((center.floor() as i64) - margin as i64).max(1);
    let high = ((center.ceil() as i64) + margin as i64).max(low);
    let high = if cap.is_finite() && cap > 0.0 {
        high.min(cap as i64)
    } else {
        high
    };
    if low > high {
        return Vec::new(); // cap below the window floor — loud empty refusal
    }
    (low..=high).map(|k| k as f32).collect()
}

// ─────────────────────────────────────────────────────────────────────
// Unit tests (synthetic pools — no fixture dependency)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic pool where column 2 is hard-gated at |g| ≤ 3 (hot in,
    /// zero out) and every other column is constant or smooth.
    fn gated_pool() -> ProbePools {
        let mut failed = Vec::new();
        for g in [-3.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let mut src = vec![0.0f32; 8];
            src[2] = 1.7 * g.signum() * (g.abs() / 3.0);
            src[1] = 0.1 * g; // smooth in the gate variable
            src[5] = 0.9; // constant
            failed.push(StepTap {
                gate_var: g,
                src,
                dst_pre: vec![0.0; 4],
            });
        }
        let mut succeeded = Vec::new();
        for g in [4.0f32, 5.0, -4.0, -5.0] {
            let mut src = vec![0.0f32; 8];
            src[1] = 0.1 * g;
            src[5] = 0.9;
            succeeded.push(StepTap {
                gate_var: g,
                src,
                dst_pre: vec![0.0; 4],
            });
        }
        ProbePools { failed, succeeded }
    }

    #[test]
    fn gate_scan_finds_only_the_gated_column() {
        let pools = gated_pool();
        let cands = discover_gated_drive(&pools, 8, &GateConfig::default());
        assert_eq!(
            cands.len(),
            1,
            "only the hard-gated column passes: {cands:?}"
        );
        assert_eq!(cands[0].col, 2);
        assert_eq!(cands[0].band, 3.0);
        // The smooth and constant columns must sit far below the bar.
        let cfg = GateConfig::default();
        for col in [1usize, 5] {
            let z = normalized_contrast(&pools, col, 3.0, &cfg);
            assert!(
                z < cfg.z_bar,
                "column {col} must not read as gated (z = {z})"
            );
        }
    }

    #[test]
    fn empty_pools_refuse() {
        let pools = ProbePools::default();
        let cands = discover_gated_drive(&pools, 8, &GateConfig::default());
        assert!(cands.is_empty());
        assert_eq!(
            parse_circuit(&pools, 8, &GateConfig::default(), &[1.0, 0.0]),
            Err(ProbeRefusal::NoObstacleFound)
        );
    }

    #[test]
    fn gap_row_is_first_index_argmax() {
        assert_eq!(discover_gap_row(&[0.1, -2.0, 2.0]), Some(1));
        assert_eq!(discover_gap_row(&[0.0, 0.0]), None);
        assert_eq!(discover_gap_row(&[]), None);
    }

    #[test]
    fn wire_slope_is_exact_on_a_linear_synthetic() {
        let mut pools = ProbePools::default();
        // dst_pre[3] = 2.5 · src[4] + 1.0 — the slope must read exactly.
        for g in [-4.0f32, -2.0, 0.0, 1.0, 3.0, 5.0] {
            let mut src = vec![0.0f32; 8];
            src[4] = g;
            pools.failed.push(StepTap {
                gate_var: g,
                src,
                dst_pre: vec![0.0; 4],
            });
            let mut src2 = vec![0.0f32; 8];
            src2[4] = g * 0.5;
            let mut dst = vec![0.0f32; 4];
            dst[3] = 2.5 * (g * 0.5) + 1.0;
            pools.succeeded.push(StepTap {
                gate_var: g + 10.0,
                src: src2,
                dst_pre: dst,
            });
        }
        // Force the linear law on the failed pool too (the pooled slope
        // must survive pool mixing).
        for t in &mut pools.failed {
            let x = t.src[4];
            t.dst_pre[3] = 2.5 * x + 1.0;
        }
        let slope = measure_wire_slope(&pools, 4, 3);
        assert!(
            (slope - 2.5).abs() < 1e-4,
            "slope must read the true coupling: {slope}"
        );
    }

    #[test]
    fn parse_refuses_without_a_wire() {
        // Gated column exists, but dst_pre is constant → NoWire.
        let mut pools = gated_pool();
        for t in &mut pools.failed {
            t.dst_pre = vec![0.5; 4];
        }
        let binding = parse_circuit(&pools, 8, &GateConfig::default(), &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(binding, Err(ProbeRefusal::NoWire));
    }

    #[test]
    fn binding_to_spec_shape_and_refusals() {
        let b = CircuitBinding {
            drive_col: 4,
            gap_row: 1,
            band: 4.0,
            gate_z: 2.0,
            slope: -7.0,
        };
        let spec = binding_to_spec(&b, 1, 64, 7.0).expect("spec");
        let EditForm::DirectionSubtraction {
            columns,
            direction,
            strength,
        } = &spec.form
        else {
            panic!("wrong form");
        };
        assert_eq!(spec.layer, 1);
        assert_eq!(columns, &[4]);
        assert_eq!(strength, &7.0);
        let mut want = vec![0.0f32; 64];
        want[1] = -1.0; // sign(slope) — the edit ADDS to the anomalous row
        assert_eq!(direction, &want);
        // Positive slope flips the sign (the edit then SUBTRACTS).
        let b_pos = CircuitBinding { slope: 3.0, ..b };
        let spec_pos = binding_to_spec(&b_pos, 1, 64, 2.0).expect("spec");
        let EditForm::DirectionSubtraction { direction, .. } = &spec_pos.form else {
            panic!("wrong form");
        };
        assert_eq!(direction[1], 1.0);
        // Zero slope refuses; out-of-range row refuses.
        assert!(binding_to_spec(&CircuitBinding { slope: 0.0, ..b }, 1, 64, 1.0).is_none());
        assert!(binding_to_spec(&b, 1, 1, 1.0).is_none());
    }

    #[test]
    fn emit_word_picks_first_passing_strength_and_refuses_when_none() {
        let b = CircuitBinding {
            drive_col: 4,
            gap_row: 1,
            band: 4.0,
            gate_z: 2.0,
            slope: -7.0,
        };
        // Fake behavior: k >= 5 clears the bar.
        let picked = emit_word(&b, 1, 64, &[8.0, 1.0, 4.0, 6.0], 0.0, |ov| {
            if ov.digest()[0] == 0 { 1.0 } else { 0.0 }
        });
        // The fake eval returns 0.0 for every real overlay → the FIRST
        // ascending k (1.0) must be accepted.
        let (spec, k) = picked.expect("word emitted");
        assert_eq!(k, 1.0);
        let EditForm::DirectionSubtraction { strength, .. } = &spec.form else {
            panic!("wrong form");
        };
        assert_eq!(strength, &1.0);
        // A rejecting eval refuses the whole grid.
        assert_eq!(
            emit_word(&b, 1, 64, &[1.0, 2.0], 0.0, |_| 0.5),
            Err(ProbeRefusal::NoViableStrength)
        );
    }

    #[test]
    fn derive_strength_grid_windows_the_slope() {
        // ŝ = −7.5238, margin 2 → {5..=10}: ⌊7.5238⌋−2 = 5, ⌈7.5238⌉+2 = 10
        // — the window CENTRES on the estimate and extends BEYOND it (the
        // true k₀ may exceed |ŝ|).
        let g = derive_strength_grid(-7.5238, 2, 64.0);
        assert_eq!(g, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        // Floor clamp at 1.
        let g_small = derive_strength_grid(-1.4, 1, 64.0);
        assert_eq!(g_small, vec![1.0, 2.0, 3.0]);
        // Degenerate inputs refuse (empty grid — never an empty sweep).
        assert!(derive_strength_grid(0.0, 2, 64.0).is_empty());
        assert!(derive_strength_grid(f32::NAN, 2, 64.0).is_empty());
        assert!(derive_strength_grid(-7.0, 0, 64.0).is_empty());
        // The cap is a HARD ceiling: a window whose floor lies above it is
        // refused EMPTY (the caller's sweep refuses NoViableStrength); a
        // window whose TOP reaches the cap is truncated there — silently
        // (the grid is a bare Vec; see the fn docs for the semantics).
        assert!(derive_strength_grid(-50.0, 2, 12.0).is_empty());
        let g_trunc = derive_strength_grid(-11.0, 2, 12.0);
        assert_eq!(g_trunc, vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn grammar_line_and_digest_are_stable() {
        let b = CircuitBinding {
            drive_col: 4,
            gap_row: 1,
            band: 4.0,
            gate_z: 1.8,
            slope: -6.9,
        };
        let line = b.grammar_line();
        assert!(line.contains("src[4]"));
        assert!(line.contains("|x|<=4"));
        assert!(line.contains("gap[row 1]"));
        assert_eq!(b.digest(), b.digest());
        let b2 = CircuitBinding { slope: -6.8, ..b };
        assert_ne!(
            b.digest(),
            b2.digest(),
            "the digest pins the slope estimate"
        );
    }
}
