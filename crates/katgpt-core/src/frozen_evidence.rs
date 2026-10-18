//! Frozen-evidence deliberation kernel + step-drift metric.
//!
//! Issue riir-ai 953 / katgpt-rs Research 555 (arXiv:2609.06746 CVRR,
//! "Reason Through the Latent!"). Distilled modelless recurrence:
//! iterate a cognition state `h` against **frozen** evidence rows `v`
//! within ONE think cycle — the state re-reads the SAME evidence with
//! shifting attention, instead of integrating new evidence per tick
//! (`evolve_belief` does the latter; `reestimation` refits direction
//! vectors from new observations — neither re-reads frozen evidence).
//!
//! CVRR's measured signature (their §5.4): consecutive per-step causal
//! maps over the same frozen evidence have cosine ≈ 0.37 — *different
//! evidence regions become causally relevant as the state evolves*.
//! Drift ≈ 1 instead means the loop re-reads nothing new (decorative
//! recurrence — a no-op burning ticks); drift ≈ 0 means no state carry.
//!
//! # Modelless discipline
//!
//! Zero training, zero allocation (caller scratch), deterministic
//! (pure f32 arithmetic — same inputs, same bits). Gates are **sigmoid,
//! never softmax** (house rule). Per-step cost is `O(rows · d)`
//! (one dot-product + one sigmoid per row).
//!
//! # Stability (free theorem)
//!
//! The update `h_t = (1−β)·h_{t−1} + β·f(h_{t−1})` is a convex mix, so
//! `‖h_t‖ ≤ max(‖h_{t−1}‖, ‖f‖)` — per step the state cannot exceed the
//! larger of its previous norm and the transition output norm. With
//! row-normalized evidence (`‖v_j‖ = 1`), `‖f‖ ≤ rows`, giving the global
//! bound `‖h_t‖ ≤ max(‖h_0‖, rows)` for all `t` (no blow-up; property
//! test `convexity_norm_bound` pins this).
//!
//! # Raw/latent boundary
//!
//! Think-brain consumers ONLY (strategic decisions over belief-derived
//! scalars). Never the physics/anti-cheat raw path — raw sync correctness
//! is always-on by contract.
//!
//! Frozen-evidence deliberation kernel + step-drift metric (Issue riir-ai 953 /
//! katgpt-rs Research 555, arXiv:2609.06746 CVRR). Convex-mix sigmoid-gated
//! re-read of FROZEN evidence rows within one think cycle + the
//! decorative-recurrence drift detector. Opt-in; think-brain consumers only.

/// Evidence block: `k × d` row-major frozen rows (`row_len = d`).
///
/// The rows are READ-ONLY for the duration of a deliberation cycle —
/// mid-cycle evidence writes are the dynamic equivalent of CVRR's
/// KV-cache bypass (raw input steering the consumer directly). Snapshot
/// or freeze the evidence before entering [`deliberate`].
#[derive(Clone, Copy, Debug)]
pub struct FrozenEvidence<'a> {
    rows: &'a [f32],
    row_len: usize,
}

impl<'a> FrozenEvidence<'a> {
    /// Construct from a flat `k × d` buffer. `row_len` MUST divide the
    /// buffer length and be nonzero.
    #[inline]
    pub fn new(rows: &'a [f32], row_len: usize) -> Self {
        debug_assert!(row_len > 0 && rows.len().is_multiple_of(row_len));
        Self { rows, row_len }
    }

    /// Number of evidence rows (`k`).
    #[inline]
    pub fn row_count(&self) -> usize {
        self.rows.len().checked_div(self.row_len).unwrap_or(0)
    }

    /// Row `j` (the frozen evidence the state re-reads).
    #[inline]
    pub fn row(&self, j: usize) -> &[f32] {
        &self.rows[j * self.row_len..(j + 1) * self.row_len]
    }
}

/// Sigmoid-gated re-read transition: `f(h) = Σ_j σ(⟨h, v_j⟩/√d) · v_j`.
///
/// Written into `f_out` (caller scratch, `d` long, fully overwritten) —
/// zero allocation. Per the house rule the gate is a **sigmoid**, never a
/// softmax: rows gate independently rather than competing.
#[inline]
pub fn transition_into(h: &[f32], evidence: &FrozenEvidence, f_out: &mut [f32]) {
    let d = evidence.row_len;
    debug_assert_eq!(h.len(), d);
    debug_assert_eq!(f_out.len(), d);
    f_out.fill(0.0);
    if d == 0 {
        return;
    }
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    for j in 0..evidence.row_count() {
        let v = evidence.row(j);
        let mut dot = 0.0f32;
        let mut i = 0;
        // Chunk-4 inner loop (auto-vectorization friendly).
        while i + 4 <= d {
            dot += h[i] * v[i] + h[i + 1] * v[i + 1] + h[i + 2] * v[i + 2] + h[i + 3] * v[i + 3];
            i += 4;
        }
        while i < d {
            dot += h[i] * v[i];
            i += 1;
        }
        let gate = crate::sigmoid(dot * inv_sqrt_d);
        let mut k = 0;
        while k + 4 <= d {
            f_out[k] += gate * v[k];
            f_out[k + 1] += gate * v[k + 1];
            f_out[k + 2] += gate * v[k + 2];
            f_out[k + 3] += gate * v[k + 3];
            k += 4;
        }
        while k < d {
            f_out[k] += gate * v[k];
            k += 1;
        }
    }
}

/// Convex-mix deliberation over frozen evidence (CVRR §3.2, modelless).
///
/// Repeats `t_steps` times: compute [`transition_into`] into `scratch`,
/// then `h ← (1−β)·h + β·scratch`. `scratch` MUST be `h.len()` long and is
/// fully overwritten each step. `β` is clamped to `[0, 1]`; `t_steps == 0`
/// is a no-op. Zero allocation, deterministic.
///
/// Default budget (CVRR): `t_steps = 4`, `β = 0.5` — the +9.9% latency
/// class of one shared layer reused four times.
pub fn deliberate(
    h: &mut [f32],
    evidence: &FrozenEvidence,
    t_steps: usize,
    beta: f32,
    scratch: &mut [f32],
) {
    if t_steps == 0 || h.is_empty() {
        return;
    }
    let beta = beta.clamp(0.0, 1.0);
    if beta == 0.0 {
        return; // Pure no-op mix: h never moves.
    }
    let one_minus = 1.0 - beta;
    for _ in 0..t_steps {
        transition_into(h, evidence, scratch);
        for (h_el, f_el) in h.iter_mut().zip(scratch.iter()) {
            *h_el = one_minus * *h_el + beta * *f_el;
        }
    }
}

/// Per-step gate map: `I[j] = σ(⟨h, v_j⟩/√d)` — which evidence rows the
/// state is currently reading. `out` MUST be `row_count()` long.
#[inline]
pub fn gate_map_into(h: &[f32], evidence: &FrozenEvidence, out: &mut [f32]) {
    let d = evidence.row_len;
    debug_assert_eq!(h.len(), d);
    debug_assert_eq!(out.len(), evidence.row_count());
    if d == 0 {
        return;
    }
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    for (j, slot) in out.iter_mut().enumerate() {
        let v = evidence.row(j);
        let mut dot = 0.0f32;
        for (h_el, v_el) in h.iter().zip(v.iter()) {
            dot += h_el * v_el;
        }
        *slot = crate::sigmoid(dot * inv_sqrt_d);
    }
}

/// Cosine similarity between two gate maps (both `n > 0` long).
/// Identical maps → 1.0; orthogonal → 0.0.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        return 1.0; // Both constant/zero maps — "identical" reads.
    }
    dot / denom
}

/// Default drift band edges (fixture-calibrated; domain recalibration via
/// the Issue-953 T5 ε protocol before gating anything).
pub const DEFAULT_DRIFT_STUCK: f32 = 0.999;
pub const DEFAULT_DRIFT_CHAOS: f32 = 0.05;

/// Decorative-recurrence verdict over consecutive gate-map drifts.
///
/// - [`RecurrenceHealth::DriftStuck`] — every consecutive cosine ≥
///   [`DEFAULT_DRIFT_STUCK`]: the loop re-reads the SAME rows with the SAME
///   weights every step — decorative recurrence (a no-op burning ticks).
/// - [`RecurrenceHealth::NoCarry`] — every consecutive cosine ≤
///   [`DEFAULT_DRIFT_CHAOS`]: the state does not accumulate (maps
///   uncorrelated step to step).
/// - [`RecurrenceHealth::Healthy`] — attention shifts across steps
///   (CVRR's cosine ≈ 0.37 class).
///
/// `drifts` holds the cosine between gate map `t` and `t−1` for
/// `t = 1..T`; empty input is [`RecurrenceHealth::Healthy`] (a T=1 cycle
/// has no steps to drift between — not evidence of pathology).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecurrenceHealth {
    DriftStuck,
    NoCarry,
    Healthy,
}

pub fn classify_drift(drifts: &[f32]) -> RecurrenceHealth {
    if drifts.is_empty() {
        return RecurrenceHealth::Healthy;
    }
    if drifts.iter().all(|&d| d >= DEFAULT_DRIFT_STUCK) {
        return RecurrenceHealth::DriftStuck;
    }
    if drifts.iter().all(|&d| d <= DEFAULT_DRIFT_CHAOS) {
        return RecurrenceHealth::NoCarry;
    }
    RecurrenceHealth::Healthy
}

// ---------------------------------------------------------------------------
// Unit tests (Issue riir-ai 953 T1/T2 canaries)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 8;

    fn normalized_rows(seed_vals: [f32; 16]) -> ([f32; 16], usize) {
        // Two 8-d rows, each normalized to unit L2.
        let mut buf = seed_vals;
        for r in 0..2 {
            let norm: f32 = buf[r * D..(r + 1) * D]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            if norm > 1e-9 {
                for x in &mut buf[r * D..(r + 1) * D] {
                    *x /= norm;
                }
            }
        }
        (buf, D)
    }

    #[test]
    fn deterministic_same_inputs_same_bits() {
        let (rows, d) = normalized_rows([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0,
        ]);
        let ev = FrozenEvidence::new(&rows, d);
        let mut h1 = [0.5_f32; D];
        let mut h2 = [0.5_f32; D];
        let mut s1 = [0.0_f32; D];
        let mut s2 = [0.0_f32; D];
        deliberate(&mut h1, &ev, 4, 0.5, &mut s1);
        deliberate(&mut h2, &ev, 4, 0.5, &mut s2);
        assert_eq!(h1, h2, "pure deterministic math — bit-identical");
    }

    #[test]
    fn beta_zero_and_t_zero_are_noops() {
        let (rows, d) = normalized_rows([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ]);
        let ev = FrozenEvidence::new(&rows, d);
        let mut h = [0.3_f32; D];
        let mut s = [0.0_f32; D];
        deliberate(&mut h, &ev, 4, 0.0, &mut s);
        assert_eq!(h, [0.3_f32; D], "beta=0 → no-op mix");
        deliberate(&mut h, &ev, 0, 0.5, &mut s);
        assert_eq!(h, [0.3_f32; D], "t_steps=0 → no-op");
    }

    #[test]
    fn beta_one_sets_h_to_transition() {
        // beta=1: h_t = f(h_{t-1}) exactly after one step.
        let (rows, d) = normalized_rows([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let ev = FrozenEvidence::new(&rows, d);
        let mut h = [1.0_f32; D];
        let mut s = [0.0_f32; D];
        deliberate(&mut h, &ev, 1, 1.0, &mut s);
        let mut expected = [0.0_f32; D];
        let h0 = [1.0_f32; D];
        transition_into(&h0, &ev, &mut expected);
        assert_eq!(h, expected, "beta=1, T=1 → h = f(h_0)");
    }

    #[test]
    fn convexity_norm_bound() {
        // Row-normalized evidence: ||f|| <= k (k=2 rows) → global bound
        // ||h_t|| <= max(||h_0||, k) for every t.
        let (rows, d) = normalized_rows([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0,
        ]);
        let ev = FrozenEvidence::new(&rows, d);
        let k = ev.row_count() as f32;
        let mut h = [10.0_f32; D]; // ||h_0|| = 10*sqrt(8) ≈ 28.28
        let h0_norm: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
        let bound = h0_norm.max(k);
        let mut s = [0.0_f32; D];
        for step in 0..64 {
            deliberate(&mut h, &ev, 1, 0.9, &mut s);
            let norm: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                norm <= bound + 1e-3,
                "step {step}: ||h||={norm} exceeded bound {bound}"
            );
        }
    }

    #[test]
    fn drift_stuck_canary_orthogonal_state() {
        // h orthogonal to every row → σ(0) = 0.5 on every row, every step →
        // gate maps constant → drift 1.0 → DriftStuck (the decorative-
        // recurrence canary; two-sided per the audit-instrument law).
        let rows: [f32; 16] = [
            // Row 0: e0; row 1: e1 (both unit).
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ev = FrozenEvidence::new(&rows, D);
        let mut h = [0.0_f32; D];
        h[2] = 1.0; // Orthogonal to both rows.
        let mut s = [0.0_f32; D];
        let mut maps: Vec<[f32; 2]> = Vec::new();
        let mut m = [0.0_f32; 2];
        gate_map_into(&h, &ev, &mut m);
        maps.push(m);
        for _ in 0..3 {
            deliberate(&mut h, &ev, 1, 0.5, &mut s);
            gate_map_into(&h, &ev, &mut m);
            maps.push(m);
        }
        let drifts: Vec<f32> = maps.windows(2).map(|w| cosine(&w[0], &w[1])).collect();
        assert_eq!(classify_drift(&drifts), RecurrenceHealth::DriftStuck);
    }

    #[test]
    fn classify_drift_bands_direct() {
        assert_eq!(
            classify_drift(&[1.0, 0.9995, 1.0]),
            RecurrenceHealth::DriftStuck
        );
        assert_eq!(classify_drift(&[0.0, 0.01]), RecurrenceHealth::NoCarry);
        assert_eq!(
            classify_drift(&[0.37, 0.35, 0.4]),
            RecurrenceHealth::Healthy
        );
        assert_eq!(classify_drift(&[]), RecurrenceHealth::Healthy);
        // Mixed (some stuck, some not) → Healthy band by construction.
        assert_eq!(classify_drift(&[1.0, 0.3]), RecurrenceHealth::Healthy);
    }

    #[test]
    fn gate_map_values_in_unit_interval() {
        let (rows, d) = normalized_rows([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0,
        ]);
        let ev = FrozenEvidence::new(&rows, d);
        let h = [0.5_f32; D];
        let mut m = [0.0_f32; 2];
        gate_map_into(&h, &ev, &mut m);
        for g in m {
            assert!((0.0..=1.0).contains(&g), "sigmoid gate in (0,1): {g}");
        }
    }
}
