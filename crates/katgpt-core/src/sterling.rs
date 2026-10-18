//! Issue 672 — Sterling-derived modelless primitives (Research 491,
//! arXiv:2608.07594 "Scaling Inherently Interpretable Language Models",
//! Guide Labs / Steerling-8B).
//!
//! Three closed-form mechanisms distilled from the paper's inference-time
//! half, plus two riders. All are generic math — no game semantics, no
//! training, no gradient descent:
//!
//! 1. **ReLU-gated suppression** ([`relu_gated_suppression_into`]) — the
//!    paper §6.2's one-sided output-space suppression mask
//!    `ℓ_v ← ℓ_v − s·ReLU(a_c[v])`. The naive form `−s·a_c[v]` *promotes*
//!    every anti-aligned token (`a_c[v] < 0` gets a logit *boost* — paper
//!    Fig. 19); the ReLU gate penalizes only positive alignments and leaves
//!    anti-aligned tokens bit-unchanged. The output-space complement of
//!    MANCE activation erasure (R409).
//!
//! 2. **Exact-decomposition readout** ([`decomposed_readout_into`]) — when a
//!    consumer is a linear head over *additive* components
//!    `h = k̂₁ + … + k̂ₙ + ε` (the paper's additive bottleneck
//!    `h̄ = k̂ + û + ε`), every output decomposes exactly into per-component
//!    contributions + residual **in real arithmetic**. In f32 the invariant
//!    we ship and gate is `Σ parts + residual == fused` **bit-identical**,
//!    enforced by computing `fused` as the fixed-order sum of the very
//!    parts returned to the caller (never by re-dotting a pre-summed
//!    vector, which FP re-associates). Attribution becomes a byproduct of
//!    the computation it explains. First consumer: riir-ai Issue 732
//!    (per-emotion decision ledger).
//!
//! 3. **Lift-set steering targets** ([`LiftTableBuilder`]) —
//!    `lift(w, c) = P(w | chunks-tagged-c) / P(w)`, a pure two-pass corpus
//!    statistic with additive smoothing (zero training). Top-K lifted sets
//!    become expression targets for `latent_field_steering` (Plan 309) /
//!    γ=τ/peak calibration consumers, and bias tables for drafters via
//!    [`lift_set_to_bias_table`].
//!
//! Riders (T4):
//!
//! 4. **HSIC-style normalized cross-covariance gauge**
//!    ([`hsic_cross_covariance_gauge`]) — the paper's L_indep independence
//!    measure `‖ΨᵀΦ‖²_F / (d²(M−1))` as a *measure-only* disentanglement
//!    gauge (no training loss): 0 for exactly-orthogonal column spaces,
//!    maximal for identical blocks. For shard/affect channel audits.
//!
//! 5. **Noisy-OR span aggregation** — generalized from the civ salience
//!    gate's literal `1 − (1−c)(1−boost)` into the *ungated* core util
//!    [`crate::noisy_or`] (+ log1p-stable [`crate::noisy_or_stable`] for
//!    many-small-term spans). Lives at the crate root next to `sigmoid`
//!    because the riir-games-civ site delegates to it under the DEFAULT
//!    feature set (a feature-gated util would change civ's dep surface).
//!
//! # Domain classification
//!
//! All operators here are latent-space, local, never synced: logit masks and
//! attribution ledgers are per-call views; lift tables are offline corpus
//! statistics committed with BLAKE3. No sync dependency, no replay coupling.
//!
//! Feature: `sterling_primitives` (opt-in). Promotion to default requires a
//! consumer GOAT to pass (riir-ai Issue 732 — the exact-emotion-ledger NPC
//! decision surface — is the first candidate consumer).
//!
//! Sterling-derived modelless primitives (Issue 672 / Research 491,
//! arXiv:2608.07594 Steerling-8B): ReLU-gated logit suppression (the naive
//! subtraction promotes anti-aligned tokens), exact-decomposition readout
//! (Σ parts + residual == fused, bit-identical by fixed summation order),
//! lift-set steering targets (two-pass corpus statistic), γ=τ/peak logit-space
//! calibration, + the HSIC-style cross-covariance gauge (measure-only). The
//! noisy-OR rider lives UNGATED at the crate root (`noisy_or` / `noisy_or_stable`)
//! because the riir-games-civ salience gate delegates to it under the DEFAULT
//! feature set. Opt-in — promotion requires a consumer GOAT (riir-ai Issue 732,
//! the exact-emotion-ledger NPC decision surface, is the first candidate).

// ─────────────────────────────────────────────────────────────────────────────
// T1 — ReLU-gated suppression (paper §6.2)
// ─────────────────────────────────────────────────────────────────────────────

/// One-sided logit suppression mask: `out = logits − s · ReLU(alignment)`.
///
/// Given a concept-alignment vector `a = W·e_c ∈ ℝ^|V|` (how much each vocab
/// token expresses concept c) and strength `s ≥ 0`, penalizes ONLY positively
/// aligned tokens. Anti-aligned tokens (`a[v] < 0`) are left **bit-unchanged**
/// — the naive subtraction `ℓ − s·a` would *promote* them (paper Fig. 19's
/// failure mode; the falsifier test below pins the contrast).
///
/// Branch-free inner loop (`f32::max(0.0, ·)` is a max instruction, not a
/// branch) — SIMD/auto-vectorization friendly. Zero-alloc: caller-owned `out`.
///
/// # Panics
/// Panics (debug) if slices have mismatched lengths.
#[inline]
pub fn relu_gated_suppression_into(
    logits: &[f32],
    alignment: &[f32],
    strength: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(
        logits.len(),
        alignment.len(),
        "alignment must be |V|-matched"
    );
    debug_assert_eq!(logits.len(), out.len(), "out must be |V|-matched");
    for i in 0..logits.len() {
        // ReLU gate: only positive alignments are suppressed.
        out[i] = logits[i] - strength * alignment[i].max(0.0);
    }
}

/// The paper's per-direction steering calibration `γ = τ / peak(e_c)`
/// (logit-space cousin of MAG's activation-norm `calibrate_alpha`, R397 —
/// commensurates *output effect* rather than input magnitude).
///
/// `peak(e_c) = max_v e_cᵀ·W_y[v]` — the direction's largest logit effect
/// over the head rows. Returns the injection strength that moves the peak
/// logit by exactly `tau`. Returns `None` when `peak ≤ 0` or non-finite
/// (degenerate direction — caller should skip injection rather than divide).
#[inline]
#[must_use]
pub fn tau_over_peak_calibration(
    head_rows: &[f32],
    direction: &[f32],
    n_out: usize,
    tau: f32,
) -> Option<f32> {
    let d = direction.len();
    debug_assert_eq!(
        head_rows.len(),
        n_out * d,
        "head_rows must be n_out × d row-major"
    );
    let mut peak = f32::NEG_INFINITY;
    for v in 0..n_out {
        let row = &head_rows[v * d..(v + 1) * d];
        let mut dot = 0.0f32;
        for i in 0..d {
            dot += row[i] * direction[i];
        }
        peak = peak.max(dot);
    }
    if peak.is_finite() && peak > 0.0 {
        Some(tau / peak)
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// T2 — Exact-decomposition readout (paper §5.3 additive bottleneck)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a decomposed readout over one head row.
///
/// **The bit-identity invariant:** `contributions.iter().sum::<f32>() +
/// residual_contribution == fused` holds **bit-identically**, because
/// `fused` is *defined* as exactly that fixed-order sum (the same
/// accumulation the caller performs when displaying the ledger). In real
/// arithmetic (the paper's setting) this equals `wᵀ·(Σ k̂ᵢ + ε)` exactly;
/// in f32 the pre-summed-vector dot differs only in low bits (quantified in
/// tests at < 1e-5 relative for realistic dims — never claimed bit-identical).
#[derive(Debug, Clone, PartialEq)]
pub struct DecomposedReadout {
    /// `wᵀ·k̂ᵢ` per additive component, in caller order.
    pub contributions: Vec<f32>,
    /// `wᵀ·ε` — the residual channel's contribution.
    pub residual_contribution: f32,
    /// Fixed-order sum of [`Self::contributions`] + residual_contribution.
    pub fused: f32,
}

/// Decompose a linear readout over additive components.
///
/// Given head row `w`, additive components `k̂₁…k̂ₙ` (paper: known concepts +
/// unknown concepts), and residual `ε`, computes per-component
/// contributions, the residual contribution, and the fused total with the
/// bit-identity invariant (see [`DecomposedReadout`]).
///
/// Fixed dot-product order (ascending index) and fixed summation order
/// (component order, then residual) — deterministic, replay-stable.
///
/// Degenerate cases: empty `components` ⇒ `fused == residual_contribution`
/// exactly; `residual_vec` may be empty (treated as zero contribution 0.0 —
/// `wᵀ·0 = 0` exactly).
pub fn decomposed_readout(
    head_row: &[f32],
    components: &[&[f32]],
    residual_vec: &[f32],
) -> DecomposedReadout {
    let mut contributions = Vec::with_capacity(components.len());
    for &comp in components {
        contributions.push(fixed_order_dot(head_row, comp));
    }
    let residual_contribution = if residual_vec.is_empty() {
        0.0
    } else {
        fixed_order_dot(head_row, residual_vec)
    };
    // Canonical fixed order (shared verbatim with the GEMV variant): start
    // at 0.0, add contributions in component order, then the residual —
    // `(((0+c₀)+c₁)+r)`. Every consumer reproducing this order gets the
    // same bits.
    let mut fused = 0.0f32;
    for &c in &contributions {
        fused += c;
    }
    fused += residual_contribution;
    DecomposedReadout {
        contributions,
        residual_contribution,
        fused,
    }
}

/// Multi-output (full-head) decomposed GEMV into caller scratch.
///
/// Writes `out` as `(n_components + 2) × n_out` row-major:
/// rows `0..n` are per-component contribution vectors, row `n` is the
/// residual contribution vector, row `n+1` is the fused vector — where the
/// fused row is the fixed-order per-column sum of the rows above it
/// (bit-identity invariant holds column-wise, same as the scalar variant).
///
/// Returns the row stride (`n_components + 2`).
///
/// # Panics
/// Panics (debug) on length mismatches.
pub fn decomposed_readout_gemv_into(
    head_rows: &[f32],
    d: usize,
    components: &[&[f32]],
    residual_vec: &[f32],
    out: &mut [f32],
) -> usize {
    let n_out = head_rows.len() / d;
    let n_comp = components.len();
    let stride = n_comp + 2;
    debug_assert_eq!(head_rows.len(), n_out * d, "head_rows must be n_out × d");
    debug_assert!(out.len() >= stride * n_out, "out too small");
    for (ci, &comp) in components.iter().enumerate() {
        debug_assert_eq!(comp.len(), d);
        for v in 0..n_out {
            let row = &head_rows[v * d..(v + 1) * d];
            out[ci * n_out + v] = fixed_order_dot(row, comp);
        }
    }
    for v in 0..n_out {
        let row = &head_rows[v * d..(v + 1) * d];
        out[n_comp * n_out + v] = if residual_vec.is_empty() {
            0.0
        } else {
            fixed_order_dot(row, residual_vec)
        };
    }
    // Fused row: fixed-order column sum (components in order, then residual).
    for v in 0..n_out {
        let mut acc = 0.0f32;
        for r in 0..=n_comp {
            acc += out[r * n_out + v];
        }
        out[(n_comp + 1) * n_out + v] = acc;
    }
    stride
}

/// Ascending-index dot with a plain `f32` accumulator — the canonical
/// fixed order every decomposition consumer reproduces.
#[inline]
fn fixed_order_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────────────
// T3 — Lift-set steering targets (paper §10.2.4)
// ─────────────────────────────────────────────────────────────────────────────

/// Additive-smoothing default for lift ratios (Laplace-style).
pub const LIFT_DEFAULT_ALPHA: f32 = 1.0;

/// Streaming two-pass lift-set builder.
///
/// Pass 1 (global): [`Self::observe_global`] over every corpus chunk.
/// Pass 2 (tagged): [`Self::observe_tagged`] over chunks tagged with the
/// concept of interest. Then [`Self::finish`] returns the top-K lifted
/// tokens — the paper's expression targets (steering) / drafter bias tables.
///
/// `lift(w, c) = P(w | tagged-c) / P(w)` with additive smoothing α:
/// `((c_wc + α) / (N_c + αV)) / ((c_w + α) / (N + αV))`.
///
/// Determinism: `finish` iterates vocab in ascending index order with a
/// stable (lift DESC, word ASC) top-K selection — identical inputs produce
/// identical outputs. Counts are `u64` (no overflow at corpus scale).
///
/// This is an offline corpus statistic (cold path) — allocation here is not
/// on any tick; the finished [`LiftTable`] reads are plain slice indexing.
pub struct LiftTableBuilder {
    vocab: usize,
    alpha: f32,
    global_counts: Vec<u64>,
    tagged_counts: Vec<u64>,
    n_global: u64,
    n_tagged: u64,
}

impl LiftTableBuilder {
    /// New builder over a `vocab`-sized dictionary with smoothing `alpha > 0`.
    #[must_use]
    pub fn new(vocab: usize, alpha: f32) -> Self {
        assert!(
            alpha > 0.0,
            "alpha=0 makes lift(never-seen) 0/0; use the raw limit via tiny alpha"
        );
        Self {
            vocab,
            alpha,
            global_counts: vec![0; vocab],
            tagged_counts: vec![0; vocab],
            n_global: 0,
            n_tagged: 0,
        }
    }

    /// Pass-1 observation: one corpus chunk's tokens (the global denominator).
    pub fn observe_global(&mut self, tokens: &[u32]) {
        for &w in tokens {
            debug_assert!((w as usize) < self.vocab);
            self.global_counts[w as usize] = self.global_counts[w as usize].saturating_add(1);
            self.n_global = self.n_global.saturating_add(1);
        }
    }

    /// Pass-2 observation: one concept-tagged chunk's tokens.
    pub fn observe_tagged(&mut self, tokens: &[u32]) {
        for &w in tokens {
            debug_assert!((w as usize) < self.vocab);
            self.tagged_counts[w as usize] = self.tagged_counts[w as usize].saturating_add(1);
            self.n_tagged = self.n_tagged.saturating_add(1);
        }
    }

    /// Smoothed lift for one word (usable before/without `finish`).
    #[must_use]
    pub fn lift(&self, w: u32) -> f32 {
        let v = self.vocab as f32;
        let p_tagged = (self.tagged_counts[w as usize] as f32 + self.alpha)
            / (self.n_tagged as f32 + self.alpha * v);
        let p_global = (self.global_counts[w as usize] as f32 + self.alpha)
            / (self.n_global as f32 + self.alpha * v);
        p_tagged / p_global
    }

    /// Finish: top-K lifted tokens as `(word, lift)` pairs, sorted lift DESC
    /// then word ASC (deterministic). Requires at least one tagged
    /// observation (`n_tagged > 0`) — otherwise returns empty (lift
    /// undefined without a conditional distribution).
    #[must_use]
    pub fn finish(self, top_k: usize) -> LiftTable {
        if self.n_tagged == 0 || self.n_global == 0 {
            return LiftTable {
                alpha: self.alpha,
                entries: Vec::new(),
            };
        }
        let mut idx: Vec<u32> = (0..self.vocab as u32)
            .filter(|&w| self.tagged_counts[w as usize] > 0)
            .collect();
        idx.sort_by(|&a, &b| self.lift(b).total_cmp(&self.lift(a)).then(a.cmp(&b)));
        idx.truncate(top_k);
        let entries = idx
            .into_iter()
            .map(|w| (w, self.lift(w)))
            .collect::<Vec<_>>();
        LiftTable {
            alpha: self.alpha,
            entries,
        }
    }
}

/// Finished top-K lift set for one concept.
#[derive(Debug, Clone)]
pub struct LiftTable {
    /// The smoothing α used (recorded for reproducibility).
    pub alpha: f32,
    /// `(word, lift)` pairs, lift DESC / word ASC.
    pub entries: Vec<(u32, f32)>,
}

/// Convert a lift set into a drafter/sampling **bias table** (the consumer
/// demo for `TernaryDraftModel`-class drafters and logit-space steering
/// expression targets).
///
/// Writes `out[word] += gain · log2(lift)` for every entry (log-lift so the
/// bias is symmetric in evidence: lift 2 ⇒ +gain, lift ½ ⇒ −gain). `out` is
/// typically the vocab-sized logit/bias buffer the consumer already owns;
/// entries with `lift <= 0` are skipped. Zero-alloc (caller-owned `out`).
pub fn lift_set_to_bias_table(table: &LiftTable, gain: f32, out: &mut [f32]) {
    for &(w, lift) in &table.entries {
        if lift > 0.0 {
            let wi = w as usize;
            debug_assert!(wi < out.len());
            if wi < out.len() {
                out[wi] += gain * lift.log2();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// T4 rider — HSIC-style normalized cross-covariance gauge (paper L_indep)
// ─────────────────────────────────────────────────────────────────────────────

/// Measure-only disentanglement gauge: `‖ΨᵀΦ‖²_F / (d²·(M−1))`.
///
/// `psi` / `phi` are `M × d` row-major feature blocks (e.g. two channel
/// groups' activations over the same M samples). Columns are centered
/// in-place into `scratch_psi` / `scratch_phi` (caller-owned — zero-alloc),
/// then the d×d cross-products are accumulated in fixed order and the
/// Frobenius norm² normalized by `d²(M−1)` (the paper's linear-HSIC shape,
/// used as their L_indep independence loss; here it is ONLY a gauge).
///
/// **Controls (pinned by tests):** exactly-orthogonal column spaces ⇒
/// exactly `0.0`; identical blocks ⇒ the maximal self-covariance value;
/// the gauge is symmetric in its arguments.
///
/// O(M·d²) — an offline audit statistic, not a tick-path op.
pub fn hsic_cross_covariance_gauge(
    psi: &[f32],
    phi: &[f32],
    m: usize,
    d: usize,
    scratch_psi: &mut [f32],
    scratch_phi: &mut [f32],
) -> f32 {
    debug_assert_eq!(psi.len(), m * d);
    debug_assert_eq!(phi.len(), m * d);
    debug_assert!(scratch_psi.len() >= m * d);
    debug_assert!(scratch_phi.len() >= m * d);
    if m <= 1 || d == 0 {
        return 0.0;
    }
    // Column-center both blocks (mean over rows), into scratch.
    center_columns_into(psi, m, d, scratch_psi);
    center_columns_into(phi, m, d, scratch_phi);
    // ‖ΨᵀΦ‖²_F accumulated directly (no d×d materialization): for each row
    // pair product… — the Frobenius norm of the d×d cross-product matrix
    // Σ_{j,k} (Σ_i Ψ_ij Φ_ik)². Fixed order: j outer, k inner.
    let mut frob_sq = 0.0f32;
    for j in 0..d {
        for k in 0..d {
            let mut c_jk = 0.0f32;
            for i in 0..m {
                c_jk += scratch_psi[i * d + j] * scratch_phi[i * d + k];
            }
            frob_sq += c_jk * c_jk;
        }
    }
    frob_sq / (d as f32 * d as f32 * (m as f32 - 1.0))
}

/// Copy `src` (m×d row-major) into `dst` with each column mean-subtracted.
fn center_columns_into(src: &[f32], m: usize, d: usize, dst: &mut [f32]) {
    for j in 0..d {
        let mut mean = 0.0f32;
        for i in 0..m {
            mean += src[i * d + j];
        }
        mean /= m as f32;
        for i in 0..m {
            dst[i * d + j] = src[i * d + j] - mean;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── T1: ReLU-gated suppression ──────────────────────────────────────────

    /// The falsifier pair (G1): the NAIVE subtraction promotes anti-aligned
    /// tokens (the bug); the gated form leaves them bit-unchanged.
    #[test]
    fn t1_falsifier_naive_promotes_gated_does_not() {
        let logits = [0.5f32, -0.25, 1.0, 0.0];
        // Alignment: v0 strongly aligned, v1 ANTI-aligned, v2 aligned, v3 zero.
        let alignment = [2.0f32, -3.0, 0.5, 0.0];
        let s = 0.7f32;

        // Naive arm (the paper's Fig. 19 failure): v1 gets BOOSTED.
        let mut naive = [0f32; 4];
        for i in 0..4 {
            naive[i] = logits[i] - s * alignment[i];
        }
        assert!(
            naive[1] > logits[1],
            "naive subtraction promotes anti-aligned tokens"
        );

        // Gated arm: v1 and v3 bit-unchanged (delta exactly 0.0).
        let mut gated = [0f32; 4];
        relu_gated_suppression_into(&logits, &alignment, s, &mut gated);
        assert_eq!(
            gated[1].to_bits(),
            logits[1].to_bits(),
            "anti-aligned bit-unchanged"
        );
        assert_eq!(
            gated[3].to_bits(),
            logits[3].to_bits(),
            "zero-alignment bit-unchanged"
        );
        // Positively aligned ARE suppressed, by exactly s·a.
        assert_eq!(gated[0], logits[0] - s * 2.0);
        assert_eq!(gated[2], logits[2] - s * 0.5);
        // Ordering claim: gated suppression ≤ naive on aligned, and gated
        // never exceeds the original logits for s ≥ 0.
        for i in 0..4 {
            assert!(gated[i] <= logits[i] || alignment[i] <= 0.0);
        }
    }

    /// Branch-free gate is equivalent to the masked form on a grid.
    #[test]
    fn t1_matches_masked_form_on_grid() {
        let mut x = 7u32; // xorshift state
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for _ in 0..200 {
            let lg = (next() % 2000) as f32 / 100.0 - 10.0;
            let a = (next() % 2000) as f32 / 100.0 - 10.0;
            let s = (next() % 100) as f32 / 100.0;
            let gated = lg - s * a.max(0.0);
            let masked = if a > 0.0 { lg - s * a } else { lg };
            assert_eq!(gated.to_bits(), masked.to_bits());
        }
    }

    // ── T1 rider: γ = τ/peak calibration ────────────────────────────────────

    #[test]
    fn t1_tau_over_peak_scales_peak_logit_by_tau() {
        // 3×2 head; direction [1, 0] hits row dots [1, 2, -1] → peak 2.
        let head = [1.0f32, 0.0, 2.0, 0.0, -1.0, 0.0];
        let dir = [1.0f32, 0.0];
        let gamma = tau_over_peak_calibration(&head, &dir, 3, 0.5).expect("peak > 0");
        assert!((gamma - 0.25).abs() < 1e-7); // τ/peak = 0.5/2
        // Degenerate: all-negative direction rows → None (skip injection).
        let head_neg = [-1.0f32, 0.0, -2.0, 0.0];
        assert!(tau_over_peak_calibration(&head_neg, &dir, 2, 0.5).is_none());
    }

    // ── T2: exact-decomposition readout ─────────────────────────────────────

    /// The bit-identity invariant (G1): Σ parts + residual == fused,
    /// bit-identical, including degenerate empty-component cases.
    #[test]
    fn t2_bit_identity_sum_parts_plus_residual_eq_fused() {
        let w = [0.3f32, -1.2, 0.7, 2.0, -0.4];
        let k1 = [0.5f32, 0.1, -0.3, 1.0, 0.2];
        let k2 = [-0.7f32, 0.4, 0.9, -0.2, 0.6];
        let eps = [0.01f32, -0.02, 0.03, 0.04, -0.05];

        let r = decomposed_readout(&w, &[&k1, &k2], &eps);
        let sum = r.contributions.iter().sum::<f32>() + r.residual_contribution;
        assert_eq!(sum.to_bits(), r.fused.to_bits());

        // Degenerate: no components → fused == residual contribution exactly.
        let r0 = decomposed_readout(&w, &[], &eps);
        assert_eq!(r0.fused.to_bits(), r0.residual_contribution.to_bits());
        assert!(r0.contributions.is_empty());

        // Degenerate: empty residual vec → residual contribution exactly 0.
        let re = decomposed_readout(&w, &[&k1], &[]);
        assert_eq!(re.residual_contribution.to_bits(), 0f32.to_bits());
        let sum_e = re.contributions.iter().sum::<f32>() + 0.0;
        assert_eq!(sum_e.to_bits(), re.fused.to_bits());

        // Fully degenerate: nothing at all → fused exactly 0.
        let rn = decomposed_readout(&w, &[], &[]);
        assert_eq!(rn.fused, 0.0);
    }

    /// The paper's real-arithmetic exactness, quantified in f32: the
    /// decomposition matches the pre-summed fused dot to < 1e-5 relative at
    /// realistic dims (NOT claimed bit-identical — FP re-association).
    #[test]
    fn t2_matches_presummed_fused_within_fp_tolerance() {
        let d = 2304usize; // Gemma-2 2B n_embd
        let mut seed = 42u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let w: Vec<f32> = (0..d)
            .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let k: Vec<Vec<f32>> = (0..2)
            .map(|_| {
                (0..d)
                    .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
                    .collect()
            })
            .collect();
        let eps: Vec<f32> = (0..d)
            .map(|_| (next() % 200) as f32 / 10000.0 - 0.01)
            .collect();

        // Fused = w · (k1 + k2 + eps), summed in fixed vector order.
        let mut presum = vec![0f32; d];
        for i in 0..d {
            presum[i] = k[0][i] + k[1][i] + eps[i];
        }
        let fused_direct = fixed_order_dot(&w, &presum);

        let r = decomposed_readout(&w, &[&k[0], &k[1]], &eps);
        let rel = ((r.fused - fused_direct) / fused_direct).abs();
        assert!(rel < 1e-5, "rel err {rel} exceeds fp tolerance");
    }

    /// GEMV variant: column-wise bit-identity + agreement with the scalar
    /// path on every output row.
    #[test]
    fn t2_gemv_column_bit_identity_and_scalar_agreement() {
        let d = 8usize;
        let n_out = 5usize;
        let head: Vec<f32> = (0..n_out * d)
            .map(|i| ((i * 37) % 23) as f32 / 23.0 - 0.5)
            .collect();
        let k1: Vec<f32> = (0..d)
            .map(|i| ((i * 11) % 17) as f32 / 17.0 - 0.5)
            .collect();
        let k2: Vec<f32> = (0..d)
            .map(|i| ((i * 13) % 19) as f32 / 19.0 - 0.5)
            .collect();
        let eps: Vec<f32> = vec![0.01; d];
        let mut out = vec![0f32; (2 + 2) * n_out];
        let stride = decomposed_readout_gemv_into(&head, d, &[&k1, &k2], &eps, &mut out);
        assert_eq!(stride, 4);
        for v in 0..n_out {
            // Column-wise bit-identity: parts + residual == fused.
            let mut acc = 0.0f32;
            for r in 0..3 {
                acc += out[r * n_out + v];
            }
            assert_eq!(acc.to_bits(), out[3 * n_out + v].to_bits());
            // Agreement with the scalar path on the same row.
            let row = &head[v * d..(v + 1) * d];
            let s = decomposed_readout(row, &[&k1, &k2], &eps);
            assert_eq!(s.contributions[0].to_bits(), out[v].to_bits());
            assert_eq!(s.contributions[1].to_bits(), out[n_out + v].to_bits());
            assert_eq!(
                s.residual_contribution.to_bits(),
                out[2 * n_out + v].to_bits()
            );
            assert_eq!(s.fused.to_bits(), out[3 * n_out + v].to_bits());
        }
    }

    // ── T3: lift sets ────────────────────────────────────────────────────────

    /// Boundary identities: uniform-independence → lift ≈ 1; exclusive-to-
    /// tagged → lift > 1; never-in-tagged → lift → 0 as α → 0 (the issue's
    /// "all-0 → 0" reading: zero tagged mass vanishes in the raw limit).
    #[test]
    fn t3_lift_boundary_identities() {
        // V=4. Global corpus: 100 tokens with counts [40, 30, 20, 10].
        // Tagged corpus drawn at EXACTLY the same proportions: [20, 15, 10, 5].
        let mut b = LiftTableBuilder::new(4, 1e-6);
        let global: Vec<u32> = [40, 30, 20, 10]
            .iter()
            .enumerate()
            .flat_map(|(w, &n)| std::iter::repeat_n(w as u32, n))
            .collect();
        let tagged: Vec<u32> = [20, 15, 10, 5]
            .iter()
            .enumerate()
            .flat_map(|(w, &n)| std::iter::repeat_n(w as u32, n))
            .collect();
        b.observe_global(&global);
        b.observe_tagged(&tagged);

        // Independence: conditional == marginal → lift == 1 (to fp of tiny α).
        for w in 0..4u32 {
            assert!((b.lift(w) - 1.0).abs() < 1e-3, "w{w} lift {}", b.lift(w));
        }

        // A word exclusive to tagged chunks lifts above 1, monotonically in
        // its tagged mass.
        let mut b2 = LiftTableBuilder::new(3, 0.5);
        b2.observe_global(&[0, 0, 1, 1, 2, 2]); // N=6, counts [2,2,2]
        b2.observe_tagged(&[2, 2, 2, 2]); // N_c=4, counts [0,0,4]
        let l0 = b2.lift(0); // never in tagged
        let lift_before = b2.lift(2);
        assert!(lift_before > 1.0);
        b2.observe_tagged(&[2]); // more tagged mass on w2 → lift(w2) grows
        assert!(b2.lift(2) > lift_before, "monotone in tagged mass");
        // And w0 (all-zero tagged count) is small; α→0 drives it to ~0.
        assert!(l0 < 1.0);
        let mut b3 = LiftTableBuilder::new(3, 1e-7);
        b3.observe_global(&[0, 0, 1, 1, 2, 2]);
        b3.observe_tagged(&[2, 2, 2, 2]);
        assert!(
            b3.lift(0) < 1e-3,
            "α→0 raw-limit: zero tagged mass → ~0 lift"
        );
    }

    /// finish(): deterministic top-K ordering (lift DESC, word ASC ties) and
    /// the bias-table consumer demo wiring.
    #[test]
    fn t3_finish_topk_deterministic_and_bias_demo() {
        let mut b = LiftTableBuilder::new(6, 0.5);
        b.observe_global(&[0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5]); // N=12
        b.observe_tagged(&[4, 4, 4, 5, 5, 0]); // N_c=6: w4×3, w5×2, w0×1
        let t = b.finish(3);
        assert_eq!(t.entries.len(), 3);
        // w4 is the most over-represented (3/6 vs 1/12), then w5, then w0.
        assert_eq!(t.entries[0].0, 4);
        assert_eq!(t.entries[1].0, 5);
        assert_eq!(t.entries[2].0, 0);
        // Determinism: same input → same output bits.
        let mut b2 = LiftTableBuilder::new(6, 0.5);
        b2.observe_global(&[0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5]);
        b2.observe_tagged(&[4, 4, 4, 5, 5, 0]);
        let t2 = b2.finish(3);
        for (e1, e2) in t.entries.iter().zip(t2.entries.iter()) {
            assert_eq!(e1.0, e2.0);
            assert_eq!(e1.1.to_bits(), e2.1.to_bits());
        }
        // Consumer demo: lift set → drafter bias table (log-lift, symmetric).
        let mut bias = [0f32; 6];
        lift_set_to_bias_table(&t, 2.0, &mut bias);
        assert!(bias[4] > bias[5] && bias[5] > bias[0]);
        // w4 lifted ~ (3.5/8)/(1.5/14) = 4.083 → log2 ≈ 2.03 → bias ≈ 4.06.
        assert!((bias[4] - 2.0 * (t.entries[0].1).log2()).abs() < 1e-6);
        // Untouched slots stay exactly zero.
        assert_eq!(bias[1], 0.0);
        assert_eq!(bias[2], 0.0);
    }

    // ── T4 rider: HSIC gauge ─────────────────────────────────────────────────

    #[test]
    fn t4_hsic_controls_orthogonal_zero_identical_max() {
        let m = 12usize;
        let d = 4usize;
        // Exactly-orthogonal column spaces AFTER CENTERING: psi nonzero only
        // on rows 0..m/2 with zero column means (±1 alternating), phi only on
        // rows m/2..m likewise. Centering is then the identity (means are
        // exactly 0) and every cross dot has a zero factor in each product —
        // the gauge is EXACTLY 0. (Disjoint column *blocks* alone are NOT
        // orthogonal: the columns are vectors over the same rows.)
        let mut psi = vec![0f32; m * d];
        let mut phi = vec![0f32; m * d];
        for i in 0..m / 2 {
            for j in 0..d {
                psi[i * d + j] = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        for i in m / 2..m {
            for k in 0..d {
                phi[i * d + k] = if (i + k) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let mut sp = vec![0f32; m * d];
        let mut sp2 = vec![0f32; m * d];
        let g_orth = hsic_cross_covariance_gauge(&psi, &phi, m, d, &mut sp, &mut sp2);
        assert_eq!(g_orth, 0.0, "exactly-orthogonal column spaces ⇒ exactly 0");

        // Identical blocks: maximal (self-covariance), deterministic, and
        // symmetric in the arguments.
        let g_self = hsic_cross_covariance_gauge(&psi, &psi, m, d, &mut sp, &mut sp2);
        assert!(g_self > 0.0);
        assert_eq!(
            g_self.to_bits(),
            hsic_cross_covariance_gauge(&psi, &psi, m, d, &mut sp, &mut sp2).to_bits()
        );
        // Symmetry in arguments.
        let g_ab = hsic_cross_covariance_gauge(&psi, &phi, m, d, &mut sp, &mut sp2);
        let g_ba = hsic_cross_covariance_gauge(&phi, &psi, m, d, &mut sp, &mut sp2);
        assert!((g_ab - g_ba).abs() < 1e-6 * g_self.max(1.0));
        // A partially-overlapping block sits strictly between orthogonal and
        // identical (on these fixtures).
        let mut mixed = psi.clone();
        for i in 0..m {
            mixed[i * d + 2] = phi[i * d + 2];
        }
        let g_mix = hsic_cross_covariance_gauge(&psi, &mixed, m, d, &mut sp, &mut sp2);
        assert!(g_orth < g_mix && g_mix < g_self);
        // Degenerate: m <= 1 → 0.
        assert_eq!(
            hsic_cross_covariance_gauge(&[1.0], &[1.0], 1, 1, &mut sp, &mut sp2),
            0.0
        );
    }
}
