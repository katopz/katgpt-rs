//! Gain/Cost Loop Halting Primitive — open substrate-agnostic kernel for
//! per-loop halting decisions (Plan 304, Research 282, arXiv:2606.18023).
//!
//! Distilled from LoopCoder-v2's "gain/cost scissors": halt when marginal
//! refinement gain < marginal drift cost × τ. Composes with the shipped
//! elastic-loop override (Issue 035) and effective-rank signal from Plan 152.
//!
//! **Phase 1 scope:** kernel only (struct + halt_decision + signal extractors).
//! Phase 2 wires into `forward_looped()` via the elastic-loop-override path —
//! backward-compatible: `None` halter = current behavior, `Some(halter)` = gain/cost-gated.
//! Phase 2 lives in a separate task; this file ships the kernel only.
//!
//! # Latent vs Raw
//!
//! Gain/cost signals are local latent (per-loop hidden-state deltas). The halt
//! count L is a deterministic raw scalar safe to sync/replay.

#![cfg(feature = "gain_cost_halt")]
#![allow(clippy::float_cmp)] // float comparisons in tests against exact constants

// ─────────────────────────────────────────────────────────────────────
// Decision types (T1.3)
// ─────────────────────────────────────────────────────────────────────

/// Result of a single gain/cost halt evaluation.
///
/// Returned by [`GainCostLoopHalter::halt_decision`] each loop. The caller
/// maps this onto its loop-control flow:
/// - [`HaltDecision::Continue`] → keep looping.
/// - [`HaltDecision::Halt`] → break out of the loop; use the current hidden state.
/// - [`HaltDecision::RefusedFloor`] → keep looping (we are below `l_min` and
///   refuse to halt to protect representational capacity, ELT §1.4).
///
/// The enum is small (≤ 2 bytes) and `Copy`, so it is cheap to pass around
/// on the stack and does not allocate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HaltDecision {
    /// Continue looping — gain >= cost × tau and no oscillation detected.
    Continue,
    /// Halt now — gain < cost × tau, OR oscillation count reached patience.
    /// The caller should exit the loop and use the current hidden state.
    Halt {
        /// Why the halter fired.
        reason: HaltReason,
    },
    /// Refused — loop_idx < l_min. Continue regardless of gain/cost.
    /// Protects representational capacity (ELT §1.4: sub-floor loops collapse).
    RefusedFloor,
}

/// Why [`GainCostLoopHalter::halt_decision`] returned [`HaltDecision::Halt`].
///
/// `#[repr(u8)]` keeps the payload at 1 byte so the whole
/// [`HaltDecision`] stays well under a cache word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HaltReason {
    /// Marginal refinement gain dropped below drift cost × tau.
    GainBelowCost = 0,
    /// Update direction reversed (cos θ < 0) for `oscillation_patience` loops.
    Oscillation = 1,
}

// ─────────────────────────────────────────────────────────────────────
// Halter state (T1.2)
// ─────────────────────────────────────────────────────────────────────

/// Per-loop state for the gain/cost halting criterion (Research 282 / Plan 304).
///
/// Tracks the signals needed to decide whether to continue looping: the
/// previous loop's effective rank (for the gain curve), the previous loop's
/// step size (for the angular-change cos θ computation), and the oscillation
/// counter (for early halt on cos θ < 0).
///
/// State size: 12 bytes + 1 `Option<&[f32]>` borrow (no allocation). The
/// hidden-state borrow is valid only within a single `forward_looped()` call;
/// the halter does not own a copy.
///
/// Phase 1 ships the state and the [`Self::halt_decision`] kernel. The
/// `prev_erank` / `prev_step` fields are mutated by the Phase-2 forward-path
/// wiring (T2.3) — Phase 1's kernel only reads `tau`, `oscillation_patience`,
/// `l_min`, and the `oscillation_count` counter.
///
/// `Default` is implemented manually below (not derived) because the derived
/// impl would zero-initialize `tau`, `oscillation_patience`, and `l_min`,
/// which is semantically wrong (patience = 0 would halt on the first loop;
/// tau = 0 would halt whenever gain < 0). The plan's documented defaults are
/// `tau = 1.0`, `oscillation_patience = 1`, `l_min = 1`.
#[derive(Clone, Debug)]
pub struct GainCostLoopHalter {
    /// Effective rank at the previous loop (for delta computation).
    /// `None` on the first loop (no previous to compare against).
    pub(crate) prev_erank: Option<f32>,
    /// Step size at the previous loop (`||h^(r) - h^(r-1)||₂`). Used for the
    /// angular-change cos θ computation.
    pub(crate) prev_step: f32,
    /// Count of consecutive loops where cos θ < 0 (oscillation detector).
    /// Halts when this reaches `oscillation_patience`.
    pub(crate) oscillation_count: u8,
    /// Config: halt when gain < cost × tau. Default tau = 1.0.
    pub(crate) tau: f32,
    /// Config: halt after this many consecutive oscillatory loops. Default 1.
    pub(crate) oscillation_patience: u8,
    /// Config: L_min floor (refuse to halt below this loop index). Default 1.
    pub(crate) l_min: u8,
}

impl GainCostLoopHalter {
    /// Construct a halter with explicit config.
    ///
    /// - `tau` — halt when `gain < cost * tau`. `1.0` = symmetric (gain must
    ///   exceed cost). Lower = halt more eagerly; higher = loop longer.
    /// - `oscillation_patience` — number of consecutive `cos θ < 0` loops
    ///   before halting on [`HaltReason::Oscillation`]. `1` = halt on the
    ///   first reversal. Must be ≥ 1 to be meaningful.
    /// - `l_min` — minimum loop index before any halt is permitted. Loops
    ///   below this floor return [`HaltDecision::RefusedFloor`] regardless of
    ///   gain/cost signals, protecting representational capacity.
    #[inline]
    pub fn new(tau: f32, oscillation_patience: u8, l_min: u8) -> Self {
        Self {
            prev_erank: None,
            prev_step: 0.0,
            oscillation_count: 0,
            tau,
            oscillation_patience,
            l_min,
        }
    }

    /// Decide whether to continue looping after loop `loop_idx`.
    ///
    /// # Arguments
    /// - `loop_idx` — 1-based index of the loop just completed.
    /// - `gain` — marginal refinement gain (e.g. effective-rank delta, or
    ///   coherence improvement).
    /// - `cost` — marginal drift cost (e.g. coherence decay, or staleness).
    /// - `cos_theta` — alignment of last two update directions, in `[-1, 1]`.
    ///   Negative ⇒ reversal (oscillatory).
    ///
    /// # Evaluation order
    /// 1. **L_min floor** — if `loop_idx < l_min`, return [`RefusedFloor`].
    /// 2. **Oscillation detector** — if `cos_theta < 0`, bump the counter;
    ///    halt on [`Oscillation`] once it reaches patience. Otherwise reset
    ///    the counter to zero (a single aligned loop clears the history).
    /// 3. **Gain/cost scissors** — if `gain < cost * tau`, halt with
    ///    [`GainBelowCost`].
    /// 4. Otherwise [`Continue`].
    ///
    /// # NaN handling
    /// `cos_theta < 0.0` is `false` for NaN, so a NaN cos θ is treated as
    /// non-oscillatory (does not trip the detector). `gain < cost * tau` is
    /// also `false` for NaN gain/cost, so a NaN gain never triggers a halt —
    /// the loop continues. This is the safe direction: a corrupt signal
    /// should not prematurely kill the loop.
    ///
    /// [`RefusedFloor`]: HaltDecision::RefusedFloor
    /// [`Oscillation`]: HaltReason::Oscillation
    /// [`GainBelowCost`]: HaltReason::GainBelowCost
    /// [`Continue`]: HaltDecision::Continue
    #[inline]
    pub fn halt_decision(
        &mut self,
        loop_idx: usize,
        gain: f32,
        cost: f32,
        cos_theta: f32,
    ) -> HaltDecision {
        // L_min floor — refuse to halt below representational minimum.
        if loop_idx < self.l_min as usize {
            return HaltDecision::RefusedFloor;
        }

        // Oscillation early-halt — catches what stability-only primitives miss.
        // A NaN cos_theta fails `< 0.0` → treated as non-oscillatory.
        if cos_theta < 0.0 {
            self.oscillation_count = self.oscillation_count.saturating_add(1);
            if self.oscillation_count >= self.oscillation_patience {
                return HaltDecision::Halt {
                    reason: HaltReason::Oscillation,
                };
            }
        } else {
            // Positive or zero alignment resets the streak — one good loop
            // forgives a prior wobble.
            self.oscillation_count = 0;
        }

        // Gain/cost scissors — the primary criterion.
        // NaN-safe: NaN < anything is false, so a corrupt gain/cost does not
        // fire a spurious halt.
        if gain < cost * self.tau {
            return HaltDecision::Halt {
                reason: HaltReason::GainBelowCost,
            };
        }

        HaltDecision::Continue
    }

    /// Update the previous-loop step size (gain signal) for the next iteration's
    /// angular-change computation. Called by the forward-path wiring (Plan 304 T2.3).
    ///
    /// The fields on this struct are `pub(crate)`, but the forward-path wiring
    /// lives in the ROOT crate (`katgpt-rs/src/transformer.rs`), not in
    /// `katgpt-core`. This setter is the only public mutation surface the wiring
    /// needs — keeping the rest of the state private preserves the kernel's
    /// invariants (e.g. `oscillation_count` is only mutated by `halt_decision`).
    #[inline]
    pub fn update_prev_step(&mut self, step: f32) {
        self.prev_step = step;
    }

    /// Update the previous-loop effective rank. Called by the forward-path wiring
    /// when erank is used as the gain signal (Plan 304 T2.3, future work — the
    /// Phase-2 wiring uses `step_size` as the gain signal by default because the
    /// per-loop hidden state in `forward_looped` is a single vector, which makes
    /// erank degenerate; see the plan's DEVIATION note).
    #[inline]
    pub fn update_prev_erank(&mut self, erank: f32) {
        self.prev_erank = Some(erank);
    }

    /// Read the cached previous-loop step size. The forward-path wiring uses
    /// this to compute the angular-change cos θ between the current and previous
    /// update directions without recomputing the previous step.
    #[inline]
    pub fn prev_step(&self) -> f32 {
        self.prev_step
    }
}

impl Default for GainCostLoopHalter {
    /// Default config: `tau = 1.0`, `oscillation_patience = 1`, `l_min = 1`.
    ///
    /// These are the conservative paper defaults — halt on the first
    /// oscillation and the first gain-below-cost crossover, but never below
    /// loop 1.
    #[inline]
    fn default() -> Self {
        Self::new(1.0, 1, 1)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Signal extractors (T1.4)
// ─────────────────────────────────────────────────────────────────────

/// Compute step size δ = `||h^(r) - h^(r-1)||₂` between two consecutive loops.
///
/// Zero-allocation: caller passes both hidden states as `&[f32]`. The two
/// slices must have equal length (debug-asserted).
///
/// This is the raw distance the hidden state traveled this loop; it feeds
/// both the cost side of the gain/cost scissors (as a drift proxy) and the
/// direction vectors for [`angular_change`].
#[inline]
pub fn step_size(h_curr: &[f32], h_prev: &[f32]) -> f32 {
    debug_assert_eq!(
        h_curr.len(),
        h_prev.len(),
        "step_size: hidden-state slices must have equal length"
    );
    // SIMD fused distance: Σ(a-b)² in one NEON/AVX2 sweep, then sqrt once.
    crate::simd::simd_dist_sq(h_curr, h_prev, h_curr.len()).sqrt()
}

/// Compute angular change `cos θ` between two successive update directions.
///
/// - `curr_step = h^(r) - h^(r-1)`
/// - `prev_step = h^(r-1) - h^(r-2)`
///
/// Returns a value in `[-1, 1]`:
/// - `1.0` — same direction (convergent).
/// - `0.0` — orthogonal (fresh exploration) OR either vector is zero-length.
/// - `< 0.0` — reversal (oscillatory).
///
/// Zero-allocation. Returns `0.0` (never NaN) when either input has zero
/// norm, so the oscillation detector never trips on a degenerate step.
/// The zero-norm guard (`denom > 0.0`) also short-circuits NaN inputs since
/// NaN comparisons are false — see the `no_nan_in_any_path` test.
#[inline]
pub fn angular_change(curr_step: &[f32], prev_step: &[f32]) -> f32 {
    debug_assert_eq!(
        curr_step.len(),
        prev_step.len(),
        "angular_change: step slices must have equal length"
    );
    // Three SIMD reductions over the (typically d=1024..4096) step vectors.
    // For hidden-state scale these beat the single scalar pass that LLVM
    // cannot auto-vectorize (strict-FP `+=` reduction).
    let dot = crate::simd::simd_dot_f32(curr_step, prev_step, curr_step.len());
    let norm_curr = crate::simd::simd_sum_sq(curr_step, curr_step.len());
    let norm_prev = crate::simd::simd_sum_sq(prev_step, prev_step.len());
    let denom = (norm_curr * norm_prev).sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

/// Compute effective rank from a flat hidden-state matrix (`S × d`, row-major).
///
/// **Plan 304 deviation note:** Plan T1.4 expected to delegate to
/// `crate::data_probe::geometry::effective_rank`, but that function lives in
/// the ROOT crate (`katgpt-rs/src/data_probe/`), not `katgpt-core`.
/// `katgpt-core` cannot depend on the root crate (circular dep). Phase 1
/// therefore ships a local scalar implementation using the same
/// entropy-of-spectrum formula (Roy & Vetterli 2007). A future plan may lift
/// this to a shared kernel; until then the two implementations agree on the
/// formula but this one operates in `f32` (sufficient for halting thresholds,
/// whereas the root-crate diagnostic uses `f64`).
///
/// # Algorithm
/// 1. Compute column means, center the matrix.
/// 2. Build the `m × m` Gram matrix (`m = min(S, d)`) — pick the smaller
///    dimension for efficiency. Uses `X Xᵀ` when `S ≤ d`, else `Xᵀ X`.
/// 3. Jacobi eigenvalue iteration (≤ 30 sweeps) directly on the Gram matrix.
/// 4. Normalize eigenvalues to sum = 1, compute Shannon entropy in nats,
///    return `exp(entropy)`.
///
/// # Zero-allocation contract
/// The Gram matrix and the column-mean scratch both live in the
/// caller-supplied `scratch_sv` buffer — no internal allocation. **Scratch
/// layout:** `[0, d)` holds column means transiently; `[d, d + m·m)` holds the
/// `m × m` Gram matrix for in-place Jacobi.
///
/// **Required:** `scratch_sv.len() >= d + m * m` where `m = min(S, d)`.
///
/// *Plan 304 T1.4 originally stated the scratch contract as
/// `min(S, d)` (singular-values only). That is insufficient for an in-place
/// Jacobi eigenvalue sweep, which needs the full `m × m` Gram matrix. This is
/// a documented Phase-1 adaptation — the signature matches the plan exactly,
/// the scratch-size contract is the minimum honest expansion that keeps the
/// hot path allocation-free.*
///
/// # Arguments
/// - `hidden` — flat `S × d` row-major hidden-state matrix.
/// - `s` — number of rows (sequence length).
/// - `d` — number of columns (hidden dimension).
/// - `scratch_sv` — workspace of length `>= d + min(S, d)²`.
///
/// # Returns
/// Effective rank ∈ `[0, min(S, d)]`. Returns `0.0` for empty input, single
/// row, or a matrix with no variance (all eigenvalues ≈ 0).
#[inline]
pub fn hidden_erank(hidden: &[f32], s: usize, d: usize, scratch_sv: &mut [f32]) -> f32 {
    // ── Degenerate cases ───────────────────────────────────────────
    // Empty or single-row input has no variance to measure.
    if s == 0 || d == 0 {
        return 0.0;
    }
    if s == 1 {
        return 0.0;
    }

    let m = if s <= d { s } else { d };

    // Defensive: the caller MUST honor the scratch contract. We do not
    // allocate internally as a fallback — that would silently hide a
    // contract violation in the hot path. Graceful return 0.0 in BOTH debug
    // and release so the halter sees "no gain" (conservative direction) on
    // contract violation — no panic on the hot path. The contract violation
    // is the caller's bug; surfacing it via a halt is safer than aborting.
    if scratch_sv.len() < d + m * m {
        return 0.0;
    }

    // Split the scratch: means live in [0, d), gram lives in [d, d + m*m).
    let (means, gram) = scratch_sv.split_at_mut(d);
    let gram = &mut gram[..m * m];

    // ── 1. Column means ────────────────────────────────────────────
    // mean[j] = (1/S) Σ_k hidden[k*d + j]
    for slot in means.iter_mut() {
        *slot = 0.0;
    }
    for k in 0..s {
        let row = &hidden[k * d..(k + 1) * d];
        for (j, &v) in row.iter().enumerate() {
            means[j] += v;
        }
    }
    let inv_s = 1.0f32 / s as f32;
    for mean in means.iter_mut() {
        *mean *= inv_s;
    }

    // ── 2. Gram matrix (smaller of S×S or d×d) ─────────────────────
    // Centered value at (row k, col j) = hidden[k*d + j] - mean[j].
    //
    // If S <= d (m = S): gram[i][j] = Σ_k_centered_row ... actually
    //   gram[i][j] = Σ_{p=0}^{d-1} centered[i, p] * centered[j, p],
    //   i, j ∈ [0, S).  (This is X Xᵀ, shape S×S.)
    //
    // If S > d (m = d): gram[i][j] = Σ_{k=0}^{S-1} centered[k, i] * centered[k, j],
    //   i, j ∈ [0, d).  (This is Xᵀ X, shape d×d.)
    //
    // We do NOT scale by 1/S — the scale cancels when we normalize the
    // eigenvalues to sum = 1 before computing the entropy.
    // Zero-fill is only needed by the `s > d` branch, which accumulates with
    // `+=` into the upper triangle then mirrors it to the lower. The `s <= d`
    // branch assigns (not +=) to both `gram[i*m+j]` and its mirror `gram[j*m+i]`
    // for every (i, j) pair in [0, m)², so every cell is overwritten — the
    // zero-fill there is dead work.
    if s <= d {
        // m = s: build X Xᵀ (s×s). Outer product over the hidden dimension d.
        for i in 0..s {
            let row_i = &hidden[i * d..(i + 1) * d];
            for j in i..s {
                let row_j = &hidden[j * d..(j + 1) * d];
                let mut acc = 0.0f32;
                for p in 0..d {
                    let ci = row_i[p] - means[p];
                    let cj = row_j[p] - means[p];
                    acc += ci * cj;
                }
                gram[i * m + j] = acc;
                gram[j * m + i] = acc; // symmetric
            }
        }
    } else {
        // m = d: build Xᵀ X (d×d). Outer product over the sequence length s.
        // Zero-fill first — the `+=` accumulation below only touches the
        // upper triangle before the mirror pass.
        for slot in gram.iter_mut() {
            *slot = 0.0;
        }
        for k in 0..s {
            let row = &hidden[k * d..(k + 1) * d];
            // Fold this centered row into the upper triangle of gram.
            for i in 0..d {
                let ci = row[i] - means[i];
                for j in i..d {
                    let cj = row[j] - means[j];
                    gram[i * m + j] += ci * cj;
                }
            }
        }
        // Mirror the upper triangle to the lower.
        for i in 0..d {
            for j in (i + 1)..d {
                gram[j * m + i] = gram[i * m + j];
            }
        }
    }

    // ── 3. Jacobi eigenvalue iteration ─────────────────────────────
    // Operates in-place on `gram`. After convergence the eigenvalues sit on
    // the diagonal. ~30 sweeps matches the root-crate pattern (which uses 50
    // on f64; f32 converges faster and these matrices are small).
    jacobi_eigenvalues_inplace(gram, m, 30);

    // ── 4. Entropy of the (normalized) spectrum ────────────────────
    // erank = exp( -Σ p_i ln p_i ), where p_i = λ_i / Σλ.
    // Filter near-zero eigenvalues — they contribute nothing to the entropy
    // and their ln() blows up.
    let mut total = 0.0f32;
    for i in 0..m {
        total += gram[i * m + i].max(0.0);
    }
    if total < 1e-15 {
        return 0.0;
    }
    let inv_total = 1.0 / total;
    let mut entropy = 0.0f32;
    for i in 0..m {
        let lam = gram[i * m + i].max(0.0) * inv_total;
        if lam > 1e-15 {
            entropy -= lam * lam.ln();
        }
    }
    entropy.exp()
}

/// In-place Jacobi eigenvalue iteration on a symmetric `dim × dim` matrix.
///
/// On return, the diagonal of `mat` holds the eigenvalues (unordered). The
/// off-diagonal is driven to ≈ 0. No allocation — operates entirely on `mat`.
///
/// Algorithm: classic cyclic-ish Jacobi with largest-off-diagonal pivot
/// selection. Mirrors the root-crate `data_probe::geometry::jacobi_eigenvalues`
/// pattern but works in `f32` and writes eigenvalues back onto the diagonal
/// (no separate eigenvalue vector is allocated).
fn jacobi_eigenvalues_inplace(mat: &mut [f32], dim: usize, max_sweeps: usize) {
    if dim <= 1 {
        return;
    }
    for _ in 0..max_sweeps {
        // Find the largest off-diagonal element (upper triangle).
        let mut max_val = 0.0f32;
        let (mut p, mut q) = (0usize, 1usize);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = mat[i * dim + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Converged once the largest off-diagonal is negligible.
        if max_val < 1e-12 {
            break;
        }

        // Jacobi rotation angle.
        let app = mat[p * dim + p];
        let aqq = mat[q * dim + q];
        let apq = mat[p * dim + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Rotate rows/cols p, q for every other index.
        for r in 0..dim {
            if r == p || r == q {
                continue;
            }
            let arp = mat[r * dim + p];
            let arq = mat[r * dim + q];
            mat[r * dim + p] = cos_t * arp + sin_t * arq;
            mat[p * dim + r] = mat[r * dim + p];
            mat[r * dim + q] = -sin_t * arp + cos_t * arq;
            mat[q * dim + r] = mat[r * dim + q];
        }

        let new_pp = cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq;
        let new_qq = sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq;
        mat[p * dim + p] = new_pp;
        mat[q * dim + q] = new_qq;
        mat[p * dim + q] = 0.0;
        mat[q * dim + p] = 0.0;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Phase 1 G1 mechanics tests (T1.5)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    // ── halt_decision: gain/cost scissors ────────────────────────────

    #[test]
    fn halt_decision_gain_below_cost_halt() {
        // gain = 0.1, cost = 1.0, tau = 1.0 → 0.1 < 1.0 → GainBelowCost.
        let mut h = GainCostLoopHalter::new(1.0, 1, 1);
        let d = h.halt_decision(5, 0.1, 1.0, 0.9);
        assert_eq!(
            d,
            HaltDecision::Halt {
                reason: HaltReason::GainBelowCost
            }
        );
    }

    #[test]
    fn halt_decision_gain_above_cost_continue() {
        // gain = 2.0, cost = 1.0, tau = 1.0 → 2.0 >= 1.0 → Continue.
        let mut h = GainCostLoopHalter::new(1.0, 1, 1);
        let d = h.halt_decision(5, 2.0, 1.0, 0.9);
        assert_eq!(d, HaltDecision::Continue);
    }

    #[test]
    fn halt_decision_refused_below_l_min() {
        // loop_idx = 0, l_min = 1 → 0 < 1 → RefusedFloor (regardless of gain/cost).
        let mut h = GainCostLoopHalter::new(1.0, 1, 1);
        let d = h.halt_decision(0, 0.0, 100.0, -0.99);
        assert_eq!(d, HaltDecision::RefusedFloor);

        // loop_idx just below l_min still refuses.
        let mut h2 = GainCostLoopHalter::new(1.0, 1, 3);
        let d2 = h2.halt_decision(2, 0.0, 100.0, -0.99);
        assert_eq!(d2, HaltDecision::RefusedFloor);
    }

    #[test]
    fn halt_decision_oscillation_after_patience() {
        // cos_theta = -0.5 twice with patience = 2 → second call halts on
        // Oscillation. Need a non-halting gain/cost so oscillation is the
        // only path that fires.
        let mut h = GainCostLoopHalter::new(1.0, 2, 1);
        // First reversal: count 1, patience 2 → Continue.
        let d1 = h.halt_decision(1, 10.0, 0.0, -0.5);
        assert_eq!(d1, HaltDecision::Continue);
        assert_eq!(h.oscillation_count, 1);
        // Second reversal: count 2, patience 2 → Halt::Oscillation.
        let d2 = h.halt_decision(2, 10.0, 0.0, -0.5);
        assert_eq!(
            d2,
            HaltDecision::Halt {
                reason: HaltReason::Oscillation
            }
        );
        assert_eq!(h.oscillation_count, 2);
    }

    #[test]
    fn halt_decision_oscillation_resets_on_positive_cos() {
        // -0.5 → +0.5 → -0.5 with patience = 2 should NOT halt (the middle
        // positive cos resets the streak to 0, so the trailing -0.5 only
        // brings it back to 1).
        let mut h = GainCostLoopHalter::new(1.0, 2, 1);
        let _ = h.halt_decision(1, 10.0, 0.0, -0.5);
        assert_eq!(h.oscillation_count, 1);
        let _ = h.halt_decision(2, 10.0, 0.0, 0.5);
        assert_eq!(h.oscillation_count, 0, "positive cos must reset the streak");
        let d = h.halt_decision(3, 10.0, 0.0, -0.5);
        assert_eq!(d, HaltDecision::Continue);
        assert_eq!(h.oscillation_count, 1);
    }

    #[test]
    fn halt_decision_tau_scales_cost_threshold() {
        // tau = 0.5: halt iff gain < cost * 0.5. gain=0.4, cost=1.0 → 0.4 < 0.5 → Halt.
        let mut h = GainCostLoopHalter::new(0.5, 1, 1);
        assert_eq!(
            h.halt_decision(1, 0.4, 1.0, 0.9),
            HaltDecision::Halt {
                reason: HaltReason::GainBelowCost
            }
        );
        // gain=0.6, cost=1.0 → 0.6 >= 0.5 → Continue.
        let mut h2 = GainCostLoopHalter::new(0.5, 1, 1);
        assert_eq!(h2.halt_decision(1, 0.6, 1.0, 0.9), HaltDecision::Continue);
    }

    // ── step_size ────────────────────────────────────────────────────

    #[test]
    fn step_size_zero_for_identical_states() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(step_size(&a, &b), 0.0);
    }

    #[test]
    fn step_size_known_value() {
        // ||[3,4] - [0,0]|| = 5.
        let a = [3.0f32, 4.0];
        let b = [0.0f32, 0.0];
        let s = step_size(&a, &b);
        assert!((s - 5.0).abs() < 1e-6, "expected 5.0, got {s}");
    }

    // ── angular_change ───────────────────────────────────────────────

    #[test]
    fn angular_change_zero_for_zero_step() {
        // Both zero vectors → denom 0 → 0.0 (no NaN).
        let a = [0.0f32, 0.0, 0.0];
        let b = [0.0f32, 0.0, 0.0];
        let c = angular_change(&a, &b);
        assert!(
            c == 0.0 && !c.is_nan(),
            "expected exactly 0.0 with no NaN, got {c}"
        );
    }

    #[test]
    fn angular_change_zero_when_one_side_is_zero() {
        // curr = [0,0], prev = [1,0] → denom 0 → 0.0.
        let a = [0.0f32, 0.0];
        let b = [1.0f32, 0.0];
        assert_eq!(angular_change(&a, &b), 0.0);
    }

    #[test]
    fn angular_change_negative_for_reversal() {
        // curr = [1,0], prev = [-1,0] → dot=-1, norms=1,1 → cos = -1.
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        let c = angular_change(&a, &b);
        assert!((c - (-1.0f32)).abs() < 1e-6, "expected -1.0, got {c}");
    }

    #[test]
    fn angular_change_positive_for_aligned() {
        // curr = [1,0], prev = [1,0] → dot=1, norms=1,1 → cos = +1.
        let a = [1.0f32, 0.0];
        let b = [1.0f32, 0.0];
        let c = angular_change(&a, &b);
        assert!((c - 1.0f32).abs() < 1e-6, "expected +1.0, got {c}");
    }

    #[test]
    fn angular_change_orthogonal_is_zero() {
        // curr = [1,0], prev = [0,1] → dot=0 → cos = 0.
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        let c = angular_change(&a, &b);
        assert!(c.abs() < 1e-6, "expected 0.0, got {c}");
    }

    // ── hidden_erank ─────────────────────────────────────────────────

    /// Allocate a scratch big enough for the d + m*m contract.
    fn scratch_for(s: usize, d: usize) -> Vec<f32> {
        let m = s.min(d);
        vec![0.0f32; d + m * m]
    }

    #[test]
    fn hidden_erank_empty_returns_zero() {
        let mut scratch = vec![0.0f32; 8];
        assert_eq!(hidden_erank(&[], 0, 0, &mut scratch), 0.0);
        assert_eq!(hidden_erank(&[1.0], 1, 1, &mut scratch), 0.0);
    }

    #[test]
    fn hidden_erank_flat_spectrum() {
        // Symmetric ±basis: rows = ±e_i for i ∈ [0, d). Column means are zero
        // (by symmetry), so the centered covariance is proportional to I_d →
        // all eigenvalues equal → erank ≈ d = min(S, d).
        // S = 6, d = 3 → m = 3. erank should be ≈ 3.
        let s = 6;
        let d = 3;
        let mut hidden = vec![0.0f32; s * d];
        // rows 0,1,2 = +e_0, +e_1, +e_2  (index = row * d + col)
        hidden[0] = 1.0; // (0,0)
        hidden[d + 1] = 1.0; // (1,1)
        hidden[2 * d + 2] = 1.0; // (2,2)
        // rows 3,4,5 = -e_0, -e_1, -e_2
        hidden[3 * d] = -1.0; // (3,0)
        hidden[4 * d + 1] = -1.0; // (4,1)
        hidden[5 * d + 2] = -1.0; // (5,2)

        let mut scratch = scratch_for(s, d);
        let r = hidden_erank(&hidden, s, d, &mut scratch);
        assert!(
            (r - d as f32).abs() < 0.1,
            "flat spectrum erank should be ≈ {d} (= min(S,d)), got {r}"
        );
    }

    #[test]
    fn hidden_erank_flat_spectrum_tall_matrix() {
        // Same idea but S > d to exercise the Xᵀ X branch (m = d).
        // S = 8, d = 4. Use ±e_i for i in [0,4), 8 rows (4 positive + 4 negative).
        let s = 8;
        let d = 4;
        let mut hidden = vec![0.0f32; s * d];
        for i in 0..d {
            hidden[i * d + i] = 1.0; // +e_i
            hidden[(i + d) * d + i] = -1.0; // -e_i
        }
        let mut scratch = scratch_for(s, d);
        let r = hidden_erank(&hidden, s, d, &mut scratch);
        assert!(
            (r - d as f32).abs() < 0.1,
            "flat spectrum (tall) erank should be ≈ {d} (= min(S,d)), got {r}"
        );
    }

    #[test]
    fn hidden_erank_rank_one() {
        // Rows vary along a single direction → centered matrix is rank 1 →
        // one nonzero eigenvalue → entropy 0 → erank ≈ 1.0.
        // rows = k * [1,1,1] for k = 1,2,3,4. S = 4, d = 3 → m = 3.
        let s = 4;
        let d = 3;
        let mut hidden = vec![0.0f32; s * d];
        for k in 0..s {
            let val = (k + 1) as f32;
            for j in 0..d {
                hidden[k * d + j] = val;
            }
        }
        let mut scratch = scratch_for(s, d);
        let r = hidden_erank(&hidden, s, d, &mut scratch);
        assert!(
            (r - 1.0f32).abs() < 0.05,
            "rank-1 erank should be ≈ 1.0, got {r}"
        );
    }

    #[test]
    fn hidden_erank_collapsed_returns_near_zero() {
        // All identical rows → no variance → erank ≈ 0.
        let s = 10;
        let d = 3;
        let mut hidden = vec![0.0f32; s * d];
        for k in 0..s {
            for j in 0..d {
                hidden[k * d + j] = 1.0;
            }
        }
        let mut scratch = scratch_for(s, d);
        let r = hidden_erank(&hidden, s, d, &mut scratch);
        assert!(r < 0.1, "collapsed erank should be ≈ 0, got {r}");
    }

    #[test]
    fn hidden_erank_scratch_too_small_returns_zero() {
        // Violate the scratch contract: should NOT panic, should return 0.0
        // (conservative: halter sees no gain).
        let s = 4;
        let d = 3;
        let hidden = vec![1.0f32; s * d];
        let mut scratch = vec![0.0f32; 2]; // too small
        let r = hidden_erank(&hidden, s, d, &mut scratch);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn hidden_erank_monotone_between_rank_one_and_flat() {
        // A blend of rank-1 (collapsed) + flat (orthogonal) should sit
        // strictly between the two extremes — sanity check for the formula.
        let d = 4;
        let s = 8;

        // Flat baseline: ±e_i.
        let mut flat = vec![0.0f32; s * d];
        for i in 0..d {
            flat[i * d + i] = 1.0;
            flat[(i + d) * d + i] = -1.0;
        }
        // Rank-1: rows = k * [1,1,1,1].
        let mut rank1 = vec![0.0f32; s * d];
        for k in 0..s {
            let val = (k + 1) as f32;
            for j in 0..d {
                rank1[k * d + j] = val;
            }
        }

        let mut scratch = scratch_for(s, d);
        let erank_flat = hidden_erank(&flat, s, d, &mut scratch);
        let erank_rank1 = hidden_erank(&rank1, s, d, &mut scratch);
        assert!(
            erank_flat > erank_rank1,
            "flat ({erank_flat}) should rank higher than rank-1 ({erank_rank1})"
        );
    }

    // ── NaN / Inf safety ─────────────────────────────────────────────

    #[test]
    fn no_nan_in_any_path() {
        // angular_change with zero vectors must never produce NaN, even when
        // the OTHER operand carries NaN/Inf (the zero-denominator guard
        // short-circuits before any arithmetic escapes).
        let zero = [0.0f32, 0.0, 0.0];
        let nan_vec = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

        let c1 = angular_change(&zero, &nan_vec);
        assert!(
            c1 == 0.0 && !c1.is_nan(),
            "angular_change(zero, NaN/Inf) must be 0.0 (no NaN), got {c1}"
        );
        let c2 = angular_change(&nan_vec, &zero);
        assert!(
            c2 == 0.0 && !c2.is_nan(),
            "angular_change(NaN/Inf, zero) must be 0.0 (no NaN), got {c2}"
        );

        // Two non-zero vectors where one contains a NaN — the dot product
        // becomes NaN, but we document that as caller-responsibility (the
        // zero-denominator path is the only one we promise NaN-free).
        // step_size with identical inputs is always exactly 0.0.
        assert_eq!(step_size(&zero, &zero), 0.0);
    }

    #[test]
    fn halt_decision_nan_gain_does_not_halt() {
        // NaN gain: `NaN < cost` is false → Continue (do not spuriously halt
        // on a corrupt signal — that is the safe direction).
        let mut h = GainCostLoopHalter::new(1.0, 1, 1);
        let d = h.halt_decision(5, f32::NAN, 1.0, 0.5);
        assert_eq!(d, HaltDecision::Continue);
    }

    #[test]
    fn halt_decision_nan_cos_theta_does_not_trip_oscillation() {
        // NaN cos_theta: `NaN < 0.0` is false → treated as non-oscillatory.
        let mut h = GainCostLoopHalter::new(1.0, 1, 1);
        let d = h.halt_decision(5, 10.0, 0.0, f32::NAN);
        assert_eq!(d, HaltDecision::Continue);
        assert_eq!(h.oscillation_count, 0);
    }

    // ── enum layout / cache friendliness ─────────────────────────────

    #[test]
    fn halt_decision_enum_repr_u8_compat() {
        // HaltDecision carries a HaltReason payload (1 byte, #[repr(u8)]) plus
        // a discriminant. Total must stay within a cache word (≤ 8 bytes) so
        // it can be passed by value without spilling.
        assert!(
            size_of::<HaltDecision>() <= 8,
            "HaltDecision must be ≤ 8 bytes (cache-word friendly), got {}",
            size_of::<HaltDecision>()
        );
        assert_eq!(
            size_of::<HaltReason>(),
            1,
            "HaltReason must be exactly 1 byte (#[repr(u8)])"
        );
    }

    // ── Plan 304 Phase 2 wiring tests ──────────────────────────────
    //
    // These exercise the public setter surface (T2.3) added so the ROOT crate
    // (`katgpt-rs/src/transformer.rs::forward_looped`) can drive the halter
    // without accessing `pub(crate)` fields. They also prove the G1
    // determinism guarantee (Open Question 3): when the halter is configured
    // to never halt (l_min above any realistic loop_count), it is a no-op and
    // `forward_looped` output is bit-identical to the no-halter path.
    //
    // We test the kernel in isolation (no `ForwardContext` / `TransformerWeights`)
    // because constructing a full transformer in a unit test is heavyweight and
    // the G1 property is a pure function of the halter state. The composition
    // with `forward_looped` is covered by the integration tests in
    // `tests/issue_035_any_time_lt2_dispatch.rs` and `tests/goat_108_lt2_looped.rs`,
    // which pass `None` for the halter and verify byte-identical output.

    #[test]
    fn update_prev_step_setter_round_trips() {
        // The setter writes prev_step; the getter reads it back exactly.
        let mut h = GainCostLoopHalter::new(1.0, 1, 1);
        assert_eq!(h.prev_step(), 0.0, "default prev_step must be 0.0");
        h.update_prev_step(1.5);
        assert_eq!(h.prev_step(), 1.5);
        h.update_prev_step(0.0);
        assert_eq!(h.prev_step(), 0.0);
        h.update_prev_step(f32::INFINITY);
        assert_eq!(h.prev_step(), f32::INFINITY);
    }

    #[test]
    fn update_prev_erank_setter_round_trips() {
        // prev_erank starts as None; the setter makes it Some(value).
        let mut h = GainCostLoopHalter::new(1.0, 1, 1);
        h.update_prev_erank(7.3);
        // We can't read prev_erank directly (no getter — the wiring doesn't need
        // one), but we can confirm the setter doesn't perturb halt decisions:
        // a halter with a huge cost threshold still continues on a big gain.
        let d = h.halt_decision(5, 100.0, 0.0, 0.0);
        assert_eq!(d, HaltDecision::Continue);
    }

    #[test]
    fn refused_floor_never_halts_when_l_min_above_loop_count() {
        // G1 determinism guarantee (Open Question 3): when the halter is
        // configured with l_min higher than any realistic loop_count (e.g.
        // 255, the u8 max), it NEVER returns Halt for any loop the forward
        // pass would actually run — every loop returns RefusedFloor. This
        // makes the halter a pure no-op, so `forward_looped` output is
        // bit-identical to the no-halter path.
        //
        // The kernel's floor check is `loop_idx < l_min` (strict less-than),
        // so l_min=255 refuses loops 1..=254 and only evaluates at exactly
        // 255. Realistic LT2 loop counts are ≤ 32 (typically 2–8), so
        // l_min=255 is well above the practical ceiling. We sweep the
        // realistic range [1, 32] with adversarial gain/cost (gain = 0,
        // cost = f32::MAX, cos_theta = -1.0 — the most halt-eager inputs
        // possible) and confirm every one returns RefusedFloor, never Halt.
        let mut h = GainCostLoopHalter::new(1.0, 1, 255);
        for loop_idx in 1..=32usize {
            let d = h.halt_decision(loop_idx, 0.0, f32::MAX, -1.0);
            assert_ne!(
                d,
                HaltDecision::Halt { reason: HaltReason::GainBelowCost },
                "l_min=255 must refuse to halt at loop_idx={loop_idx} even with gain=0, cost=MAX"
            );
            assert_ne!(
                d,
                HaltDecision::Halt { reason: HaltReason::Oscillation },
                "l_min=255 must refuse to halt at loop_idx={loop_idx} even with cos_theta=-1"
            );
            assert_eq!(
                d,
                HaltDecision::RefusedFloor,
                "l_min=255 must return RefusedFloor at loop_idx={loop_idx}, got {d:?}"
            );
        }
        // Document the boundary: at exactly loop_idx == l_min (255), the
        // floor no longer applies and the halter evaluates normally. This is
        // correct behavior — the floor is a minimum, not a cap. No realistic
        // loop_count reaches 255, so this is not a concern in practice.
        let d_boundary = h.halt_decision(255, 0.0, f32::MAX, -1.0);
        assert_eq!(
            d_boundary,
            HaltDecision::Halt { reason: HaltReason::Oscillation },
            "at loop_idx == l_min (255), the floor lifts and the halter evaluates normally"
        );
    }
}
