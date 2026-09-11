//! Hebbian KARC readout — the `w_out` fit, Hebbian-flavored (riir-ai Plan 584,
//! ndb Plan 322 T3.2/T4.3).
//!
//! `KarcForecaster::fit_ridge` and `HebbianKernelMemory::construct` are the
//! SAME ridge closed form with different feature maps:
//!
//! | | `fit_ridge` | Hebbian `construct` |
//! |---|---|---|
//! | features | `φ(x) = basis(delay_state)` — fixed, `d_h` | `φ_A(z) = relu(A·z)` — seeded projection, `m` |
//! | solve | `(XᵀX + λI)·Woutᵀ = XᵀY` (f64 Cholesky) | whitened ridge → `(A, G, B)` (f32) |
//! | readout | `Wout` (`D × d_h`) | `B` (`64 × m`) |
//!
//! This module applies the paper's thesis ("MLPs are Hebbians", arXiv:2607.10034
//! §5.2) to the KARC readout: the forecaster's training pairs become FACTS —
//! key `k_i = pad64(delay_state_i)`, value `v_i = pad64(target_i)` — and the
//! constructed `(A, G, B)` triple plays the `w_out` role. What the unification
//! buys over `fit_ridge`:
//!
//! 1. **Edit semantics** — the fact journal loop (riir-ai Plan 583) applies to
//!    forecasting: "NPC learned transition x→y" = `Add`, a stale habit =
//!    `Remove`. `fit_ridge` has no edit story (whole-buffer refit only).
//! 2. **The margin audit** — `decoding_margin` grades forecast separability
//!    per construction; a ridge fit has no per-fact audit.
//! 3. **The freeze chain** — the constructed readout rides the existing
//!    `HebbianConstructedShard` envelope/sidecar commitment path (ndb).
//! 4. **The canonical value table** — observed `(delay_state → next-state)`
//!    pairs ARE the values (real per-NPC experience).
//!
//! # Zero-padding exactness
//!
//! `pad64` zero-pads `[f32; n] → [f32; 64]`. Inner products are preserved
//! exactly (the padding contributes nothing), so the Hebbian construction's
//! kernel evaluations — which depend only on inner products of keys/values —
//! are identical whether run on the raw or padded vectors, up to the seed
//! projection consuming 64-wide inputs. The forecast head is `out[..n]`.
//!
//! # Cadence
//!
//! `fit` is consolidation-tier (same cost class as the Hebbian construction,
//! `O(F·m·64)`); `forecast_into` is hot, zero-alloc (delegates to
//! [`HebbianKernelMemory::forward_into`] — Plan 559's audited kernel).
//!
//! # Precision (honest note)
//!
//! `fit_ridge` solves in f64; the Hebbian closed form is f32 throughout.
//! At belief-scale targets (sigmoid-projected scalars, `D = 8`) this is
//! the primitive's documented precision — consumers comparing the two fit
//! arms must state an f32 tolerance.

use crate::hebbian_kernel_memory::{
    ConstructionError, HebbianKernelMemory, HebbianMlpConfig, MarginError,
};

/// The key/value embedding width of the Hebbian memory this readout wraps.
pub const HEBBIAN_KARC_DIM: usize = 64;

/// Fit failure — the construction or the margin audit step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HebbianReadoutError {
    Construction(ConstructionError),
    Margin(MarginError),
}

/// Zero-pad a `n ≤ 64` vector into `out: &mut [f32; 64]` (`out[..n] = v`,
/// `out[n..] = 0`). The single padding definition — ndb/riir-ai consumers
/// reuse this, never hand-roll.
///
/// # Panics
/// Panics if `v.len() > 64`.
pub fn pad64_into(v: &[f32], out: &mut [f32; HEBBIAN_KARC_DIM]) {
    assert!(v.len() <= HEBBIAN_KARC_DIM, "pad64: input {} > 64", v.len());
    out[..v.len()].copy_from_slice(v);
    out[v.len()..].fill(0.0);
}

/// The fit report — the margin audit that `fit_ridge` has no equivalent of.
#[derive(Clone, Copy, Debug)]
pub struct HebbianFitReport {
    /// `γ_min` over the fact set (`+∞` at F = 1 — no competitors).
    pub gamma_min: f32,
    /// Number of facts (training pairs).
    pub n_facts: usize,
    /// The construction seed this fit used. Carried so the audit chain
    /// (ndb's `karc_hebbian_envelope`) reads it from ONE source of truth —
    /// the frozen metadata cannot record a seed the construction never saw.
    pub seed: u64,
    /// The config the memory was constructed with (thaw/verify needs it).
    pub config: HebbianMlpConfig,
}

/// The Hebbian-flavored KARC readout: a fact-storing MLP standing in for
/// `Wout`. Fit from `(delay_state, target)` pairs; forecast via
/// [`Self::forecast_into`].
///
/// Deterministic given `(pairs, config, seed)` — two nodes fitting the same
/// trajectory produce bit-identical memories (same `memory.blake3()`).
#[derive(Debug)]
pub struct HebbianKarcReadout {
    memory: HebbianKernelMemory<HEBBIAN_KARC_DIM>,
    /// Observed input width (the delay-state length).
    state_dim: usize,
    /// Observed target width (the forecast head size).
    target_dim: usize,
    report: HebbianFitReport,
}

impl From<ConstructionError> for HebbianReadoutError {
    fn from(e: ConstructionError) -> Self {
        Self::Construction(e)
    }
}

impl From<MarginError> for HebbianReadoutError {
    fn from(e: MarginError) -> Self {
        Self::Margin(e)
    }
}

impl HebbianKarcReadout {
    /// Fit the readout from KARC training pairs. Each `delay_states[i]` must
    /// have the same length `n ≤ 64` (the runtime KARC config: `n = K·D = 32`);
    /// each `targets[i]` has length `d ≤ 64` (runtime: `D = 8`).
    ///
    /// The fact pairing is positional (`fact_map = (i, i)`), the natural
    /// pairing for trajectory data. `seed` follows the bridge convention
    /// (low 64 bits of `BLAKE3(canonical fact bytes)`) but any deterministic
    /// derivation works — pass what your audit chain can re-derive.
    pub fn fit(
        delay_states: &[&[f32]],
        targets: &[&[f32]],
        config: HebbianMlpConfig,
        seed: u64,
    ) -> Result<(Self, HebbianFitReport), HebbianReadoutError> {
        let f = delay_states.len();
        if f == 0 {
            return Err(ConstructionError::EmptyFactSet.into());
        }
        if targets.len() != f {
            return Err(ConstructionError::FactMapLengthMismatch {
                keys: f,
                fact_map: targets.len(),
            }
            .into());
        }
        let state_dim = delay_states[0].len();
        let target_dim = targets[0].len();
        debug_assert!(state_dim <= HEBBIAN_KARC_DIM && target_dim <= HEBBIAN_KARC_DIM);

        // Project into 64-dim facts (one allocation set, reused by construct's
        // borrow-based API — consolidation-tier).
        let mut keys = vec![[0.0f32; HEBBIAN_KARC_DIM]; f];
        let mut values = vec![[0.0f32; HEBBIAN_KARC_DIM]; f];
        for (i, (k, v)) in delay_states.iter().zip(targets).enumerate() {
            debug_assert_eq!(k.len(), state_dim, "delay_state {i} length mismatch");
            debug_assert_eq!(v.len(), target_dim, "target {i} length mismatch");
            pad64_into(k, &mut keys[i]);
            pad64_into(v, &mut values[i]);
        }
        let key_refs: Vec<&[f32]> = keys.iter().map(|k| &k[..]).collect();
        let value_refs: Vec<&[f32]> = values.iter().map(|v| &v[..]).collect();
        let fact_map: Vec<(usize, usize)> = (0..f).map(|i| (i, i)).collect();

        let memory =
            HebbianKernelMemory::construct(&key_refs, &value_refs, &fact_map, config, seed)?;

        // Margin audit (F ≥ 2; F = 1 has no competitors — the Plan 583
        // convention: margin = +∞, trivially unambiguous).
        let gamma_min = if f >= 2 {
            memory.decoding_margin(&key_refs, &value_refs, &fact_map)?
        } else {
            f32::INFINITY
        };

        let report = HebbianFitReport { gamma_min, n_facts: f, seed, config };
        Ok((Self { memory, state_dim, target_dim, report }, report))
    }

    /// Forecast: `forward_into(pad64(delay_state))`, head = `out[..target_dim]`.
    /// Returns `false` (leaving `out` untouched) — there is no unfitted state
    /// on this type (construction is the only constructor), so this always
    /// returns `true` and exists for signature parity with
    /// `KarcForecaster::forecast_into`.
    ///
    /// Zero-alloc: `scratch_phi.len() ≥ config.m`, `out.len() ≥ target_dim`.
    /// Reuse both across calls (tick-path).
    pub fn forecast_into(
        &self,
        delay_state: &[f32],
        scratch_phi: &mut [f32],
        out: &mut [f32],
    ) -> bool {
        debug_assert_eq!(delay_state.len(), self.state_dim);
        let mut key = [0.0f32; HEBBIAN_KARC_DIM];
        pad64_into(delay_state, &mut key);
        let mut head = [0.0f32; HEBBIAN_KARC_DIM];
        self.memory.forward_into(&key[..], scratch_phi, &mut head[..]);
        let d = self.target_dim();
        out[..d].copy_from_slice(&head[..d]);
        true
    }

    /// Observed target width (the forecast head size).
    #[inline]
    pub fn target_dim(&self) -> usize {
        self.target_dim
    }

    /// The fit report (margin audit + config).
    #[inline]
    pub fn report(&self) -> &HebbianFitReport {
        &self.report
    }

    /// The underlying memory (freeze/thaw via the ndb bridge; direct
    /// `forward_into`/`retrieval_scores_into` for advanced consumers).
    #[inline]
    pub fn memory(&self) -> &HebbianKernelMemory<HEBBIAN_KARC_DIM> {
        &self.memory
    }

    /// Consume into the underlying memory (the ndb envelope path takes it
    /// by value for `HebbianConstructedShard::from_construction`).
    #[inline]
    pub fn into_memory(self) -> HebbianKernelMemory<HEBBIAN_KARC_DIM> {
        self.memory
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (Plan 584 T2 — G1 gates)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hebbian_kernel_memory::HebbianVariant;

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (*state >> 33) as f32 / (1u64 << 31) as f32;
        x - 0.5
    }

    /// Belief-scale pairs: delay 32-dim, target 8-dim (the runtime KARC config).
    fn pairs(f: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut s = seed;
        let delays: Vec<Vec<f32>> = (0..f)
            .map(|i| {
                let mut d = vec![0.0f32; 32];
                for x in &mut d {
                    *x = lcg(&mut s);
                }
                d[i % 32] += 3.0; // separable spikes
                d
            })
            .collect();
        let targets: Vec<Vec<f32>> = (0..f)
            .map(|i| {
                let mut t = vec![0.0f32; 8];
                for x in &mut t {
                    *x = lcg(&mut s);
                }
                t[i % 8] += 3.0;
                t
            })
            .collect();
        (delays, targets)
    }

    fn config() -> HebbianMlpConfig {
        HebbianMlpConfig {
            d: HEBBIAN_KARC_DIM,
            m: 128,
            ridge: 1e-6,
            variant: HebbianVariant::Whitened,
        }
    }

    fn seed_of(delays: &[&[f32]], targets: &[&[f32]]) -> u64 {
        let mut h = blake3::Hasher::new();
        h.update(&(delays.len() as u64).to_le_bytes());
        for (k, v) in delays.iter().zip(targets) {
            h.update(bytemuck::cast_slice::<f32, u8>(k));
            h.update(bytemuck::cast_slice::<f32, u8>(v));
        }
        let hash = *h.finalize().as_bytes();
        u64::from_le_bytes(hash[..8].try_into().unwrap())
    }

    #[test]
    fn fit_then_forecast_retrieves_observed_pairs() {
        // The associative check at belief scale: forecasting a TRAINED delay
        // state must retrieve its target (dot-dominance over the other
        // targets — the Plan 583 retrieval convention).
        let (delays, targets) = pairs(8, 42);
        let d_refs: Vec<&[f32]> = delays.iter().map(|d| &d[..]).collect();
        let t_refs: Vec<&[f32]> = targets.iter().map(|t| &t[..]).collect();
        let (readout, report) =
            HebbianKarcReadout::fit(&d_refs, &t_refs, config(), seed_of(&d_refs, &t_refs))
                .expect("fit");
        assert_eq!(report.n_facts, 8);
        assert!(report.gamma_min.is_finite() && report.gamma_min > 0.0, "margin {}", report.gamma_min);

        let mut phi = vec![0.0f32; 128];
        let mut out = [0.0f32; 8];
        for (d, t) in delays.iter().zip(&targets) {
            assert!(readout.forecast_into(d, &mut phi, &mut out));
            let dot_own: f32 = out.iter().zip(t).map(|(a, b)| a * b).sum();
            let best_other: f32 = targets
                .iter()
                .filter(|o| o.as_slice() != t.as_slice())
                .map(|o| out.iter().zip(o).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                dot_own > best_other,
                "forecast at a trained delay must dominate its own target (dot {dot_own:.3} vs {best_other:.3})"
            );
        }
    }

    #[test]
    fn fit_is_deterministic_and_bit_identical() {
        let (delays, targets) = pairs(6, 7);
        let d_refs: Vec<&[f32]> = delays.iter().map(|d| &d[..]).collect();
        let t_refs: Vec<&[f32]> = targets.iter().map(|t| &t[..]).collect();
        let (a, _) = HebbianKarcReadout::fit(&d_refs, &t_refs, config(), seed_of(&d_refs, &t_refs))
            .expect("fit a");
        let (b, _) = HebbianKarcReadout::fit(&d_refs, &t_refs, config(), seed_of(&d_refs, &t_refs))
            .expect("fit b");
        assert_eq!(
            a.memory().blake3(),
            b.memory().blake3(),
            "same pairs + seed → bit-identical memory"
        );
    }

    #[test]
    fn pad64_is_inner_product_exact() {
        let mut s = 99u64;
        let mut v = vec![0.0f32; 32];
        for x in &mut v {
            *x = lcg(&mut s);
        }
        let mut padded = [0.0f32; 64];
        pad64_into(&v, &mut padded);
        let dot_raw: f32 = v.iter().zip(&v).map(|(a, b)| a * b).sum();
        let dot_pad: f32 = padded.iter().zip(&padded).map(|(a, b)| a * b).sum();
        assert_eq!(dot_raw.to_bits(), dot_pad.to_bits(), "padding must preserve inner products exactly");
        assert!(padded[32..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn single_fact_has_infinite_margin() {
        let (delays, targets) = pairs(1, 5);
        let d_refs: Vec<&[f32]> = delays.iter().map(|d| &d[..]).collect();
        let t_refs: Vec<&[f32]> = targets.iter().map(|t| &t[..]).collect();
        let (_, report) =
            HebbianKarcReadout::fit(&d_refs, &t_refs, config(), seed_of(&d_refs, &t_refs))
                .expect("fit");
        assert!(report.gamma_min.is_infinite());
        assert_eq!(report.n_facts, 1);
        assert_eq!(report.seed, seed_of(&d_refs, &t_refs), "report carries the fit seed");
    }

    #[test]
    fn empty_pairs_error_shape() {
        let err = HebbianKarcReadout::fit(&[], &[], config(), 0).unwrap_err();
        assert_eq!(
            err,
            HebbianReadoutError::Construction(ConstructionError::EmptyFactSet)
        );
        let e2 = HebbianKarcReadout::fit(&[&[0.0; 32]], &[], config(), 0).unwrap_err();
        assert!(
            matches!(e2, HebbianReadoutError::Construction(ConstructionError::FactMapLengthMismatch { .. }))
        );
    }
}
