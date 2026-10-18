//! `mb_value` — bounded three-factor (dopamine) plasticity value circuit.
//!
//! The mushroom-body architecture class (Issue 767 / riir-ai Research 380,
//! distilled from adonis-singh/TMNF-C @ `eb6be045` MIT, which implements the
//! biological model of Bennett/Nowotny et al., Nat. Commun. 12:2569 (2021);
//! depression bounds after Handler et al. 2019): a fixed random sparse
//! projection → top-k sparse code → approach-minus-avoid linear readout,
//! whose **entire learning machinery is one bounded local rule** —
//!
//! ```text
//! w ← clamp(w − η · code_active · RPE · compartment_sign, 0, w0)
//! ```
//!
//! Three-factor in the neuroscience sense: pre-synaptic activity (the sparse
//! code) × a scalar modulator (the TD reward-prediction error, "dopamine") ×
//! a fixed per-compartment sign (PAM/reward compartments act on avoid-MBON
//! synapses, PPL1/punishment compartments on approach-MBON synapses). No
//! gradient descent, no loss surface, no optimizer — the update is legal
//! runtime latent-state mutation under the modelless mandate (the
//! `rating.rs` Elo precedent: online error-driven state mutation), and the
//! weight bound `[0, w0]` holds **by construction** under arbitrary (even
//! ±inf) RPE streams, a stability property unconstrained linear TD lacks.
//! Below-baseline dopamine (negative RPE on an avoid compartment) lets the
//! synapse *recover* toward `w0` — the sign structure raises `w` back.
//!
//! # What it is FOR (and the measured negatives it ships with)
//!
//! TMNF-C's measured results: value formation works (value-speed
//! correlation r = 0.8, first lap finishes at minute 3 of a 5-minute run);
//! **action selection from shared codes does not** (candidate actions
//! sharing 50% of the KC code → values barely differ → near-random choice).
//! This primitive therefore targets **value formation feeding external
//! selection** (utility scoring, pruners, gating) — not action selection
//! from the value alone. The G1 gate pins exactly that axis (ranking
//! quality vs the ridge-batch floor on the same code), plus a
//! distribution-shift arm where online adapts and a batch fit cannot.
//!
//! # Calibration is measurement, not learning
//!
//! [`MbCircuit::calibrate`] fixes feature z-scores, PN thresholds/gains
//! (quantile-calibrated), a 17-step bisection on the action-code gain to hit
//! a target code-overlap, per-MBON `w0` normalization (mean drive over
//! calibration codes = 1), readout gains, and the derived step
//! `η = α / (eff_approach + eff_avoid)` so ΔV-per-RPE ≈ `α` is scale-free
//! across circuit sizes — "nothing here is learned from reward"
//! (`@torch.no_grad` in the source).
//!
//! # UQ posture
//!
//! Not UQ-bearing: the circuit produces a **point value for ranking**. Any
//! future distribution/interval/coverage claim must first beat the
//! conformal-naive floor (the Report-the-Floor rule) — out of scope here.
//!
//! # Data-agnostic + freeze/thaw seam
//!
//! No connectome dataset ships (FlyWire/MaleCNS licensing; wiring is seeded
//! random — [`MbCircuitConfig::fly`] matches the fly's shape CLASSES, not
//! its data). The base wiring is fixed and committable; the learned `w` is
//! the only mutable state and is exposed as a plain `f32` slice
//! ([`MbCircuit::w`], [`MbCircuit::set_w`]) for BLAKE3/Merkle freeze —
//! frozen-weight replay (the TMNF-C `mb_replay` discipline) is the natural
//! thaw proof. No softmax anywhere (house rule); the readout is a linear
//! signed projection.
//!
//! mb_value — bounded three-factor (dopamine) plasticity value circuit
//! (Issue 767 / riir-ai Research 380, distilled from adonis-singh/TMNF-C @
//! eb6be045; mechanism after Bennett/Nowotny Nat. Commun. 12:2569 2021).
//! Fixed random sparse projection → top-k KC code → approach-minus-avoid
//! readout, with ONE bounded local rule as the entire learning machinery
//! (w ← clamp(w − η·code·RPE·sign, 0, w0) — the rating.rs Elo precedent for
//! legal online error-driven latent-state mutation). Value FORMATION feeding
//! external selection; no softmax; not UQ-bearing (point value for ranking);
//! no connectome data ships (seeded random wiring; fly-scale SHAPE classes
//! only). Calibration is measurement, not learning (z-scores, quantile
//! thresholds, 17-step action-gain bisection, derived η — ΔV-per-RPE ≈ α).
//! Opt-in — GOAT gate in .benchmarks/761_mb_value_goat.md.

// ─── Construction RNG (the house module-local SplitMix64 pattern) ───────────
//
// Wiring generation is a COLD path (construction + calibration); the tick
// path (code/value/update) is fully deterministic. The `effective_degree` /
// `tpr::validate` precedent: a private module-local copy, no cross-feature
// dep.

struct WireRng(u64);

impl WireRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `[0, 1)`.
    #[inline]
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// Standard-normal via Box-Muller (construction wiring only).
    #[inline]
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Uniform integer in `[0, n)` (n ≥ 1).
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    /// Sample `k` distinct indices from `[0, n)` (partial Fisher-Yates,
    /// without replacement — a duplicate KC→MBON synapse would double-apply
    /// the update rule). `k` is clamped to `n`.
    fn sample_distinct(&mut self, n: usize, k: usize, out: &mut Vec<u32>) {
        let k = k.min(n);
        out.clear();
        out.reserve(k);
        let mut pool: Vec<u32> = (0..n as u32).collect();
        for i in 0..k {
            let j = i + self.below(n - i);
            pool.swap(i, j);
            out.push(pool[i]);
        }
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────

/// Circuit shape + calibration constants. All wiring is generated from
/// `seed`; the same config always builds a bit-identical circuit.
#[derive(Debug, Clone)]
pub struct MbCircuitConfig {
    /// State-feature width (the caller's observation vector).
    pub n_features: usize,
    /// Action-code width (0 = state-only value; `code_into` then takes an
    /// empty action slice).
    pub n_action_dims: usize,
    /// Projection-neuron count.
    pub n_pn: usize,
    /// Features sampled per PN (without replacement; the fly reads ~8).
    pub pn_fanin: usize,
    /// Kenyon-cell count.
    pub n_kc: usize,
    /// PNs sampled per KC (without replacement; the fly reads ~6).
    pub kc_fanin: usize,
    /// Top-k KC code width (the fly fires 200 of 4,064 ≈ 4.9%).
    pub kc_active: usize,
    /// MBON count (readout neurons).
    pub n_mbon: usize,
    /// Distinct MBONs sampled per KC (the fly: ~15 of 97).
    pub kc_mbon_fanin: usize,
    /// First `n_approach` MBONs are approach (PPL1-compartmented); the rest
    /// are avoid (PAM-compartmented). At least 1 of each is enforced.
    pub n_approach: usize,
    /// Fraction of PNs that also read one action-code dimension.
    pub pn_action_fraction: f32,
    /// PN threshold quantile (the fly calibrates at the median, 0.5).
    pub pn_quantile: f32,
    /// Readout gain (the fly uses 40; V = 0 at calibration end).
    pub gain: f32,
    /// Target ΔV-per-RPE (the fly derives η from α = 0.02).
    pub alpha: f32,
    /// Wiring seed.
    pub seed: u64,
}

impl MbCircuitConfig {
    /// Fly-scale shape classes (686 PN / 4,064 KC / 97 MBON / 193 features /
    /// 5-dim factorized action code — TMNF-C's circuit numbers; the WIRING
    /// is seeded random, not the connectome's data).
    pub fn fly() -> Self {
        Self {
            n_features: 193,
            n_action_dims: 5,
            n_pn: 686,
            pn_fanin: 8,
            n_kc: 4_064,
            kc_fanin: 6,
            kc_active: 200,
            n_mbon: 97,
            kc_mbon_fanin: 15,
            n_approach: 48,
            pn_action_fraction: 0.5,
            pn_quantile: 0.5,
            gain: 40.0,
            alpha: 0.02,
            seed: 1,
        }
    }

    /// Small test/example circuit (4 features / 2-dim action / 64 PN /
    /// 256 KC top-32 / 16 MBON).
    pub fn toy() -> Self {
        Self {
            n_features: 4,
            n_action_dims: 2,
            n_pn: 64,
            pn_fanin: 4,
            n_kc: 256,
            kc_fanin: 6,
            kc_active: 32,
            n_mbon: 16,
            kc_mbon_fanin: 6,
            n_approach: 8,
            pn_action_fraction: 0.8,
            pn_quantile: 0.5,
            gain: 40.0,
            alpha: 0.01,
            seed: 1,
        }
    }

    fn validated(&self) -> Self {
        let mut c = self.clone();
        c.n_features = c.n_features.max(1);
        c.n_action_dims = c.n_action_dims.min(64);
        c.n_pn = c.n_pn.max(1);
        c.pn_fanin = c.pn_fanin.clamp(1, c.n_features);
        c.n_kc = c.n_kc.max(1);
        c.kc_fanin = c.kc_fanin.clamp(1, c.n_pn);
        c.kc_active = c.kc_active.clamp(1, c.n_kc);
        c.n_mbon = c.n_mbon.max(2);
        c.kc_mbon_fanin = c.kc_mbon_fanin.clamp(1, c.n_mbon);
        c.n_approach = c.n_approach.clamp(1, c.n_mbon - 1);
        c.pn_action_fraction = c.pn_action_fraction.clamp(0.0, 1.0);
        c.pn_quantile = c.pn_quantile.clamp(0.02, 0.98);
        c
    }
}

/// Calibration report (all measurement, no reward learning — cold path).
#[derive(Debug, Clone, Copy)]
pub struct MbCalibration {
    /// Action-code gain the bisection fixed (0 when `n_action_dims == 0`).
    pub action_gain: f32,
    /// Mean code overlap between different actions at that gain.
    pub action_overlap: f32,
    /// Fraction of PNs firing on the calibration batch.
    pub pn_active_fraction: f32,
    /// Derived step: `dw` scale per synapse; ΔV-per-RPE ≈ `alpha`.
    pub eta: f32,
    pub dv_per_rpe_approach: f32,
    pub dv_per_rpe_avoid: f32,
    pub mbons_connected: usize,
    pub approach_mbons_used: usize,
    pub avoid_mbons_used: usize,
}

// ─── Circuit ────────────────────────────────────────────────────────────────

/// The frozen base wiring + calibration state of one circuit, plus the
/// learned `w` (the only mutable state).
///
/// For the per-NPC consumer story the BASE (everything but `w`) is shared
/// per archetype; `w` is the per-NPC overlay. [`MbCircuit::w`] /
/// [`MbCircuit::set_w`] are the freeze/thaw seam.
#[derive(Debug, Clone)]
pub struct MbCircuit {
    n_features: usize,
    n_action_dims: usize,
    n_pn: usize,
    pn_fanin: usize,
    n_kc: usize,
    kc_fanin: usize,
    kc_active: usize,
    n_mbon: usize,
    n_approach: usize,
    kc_mbon_fanin: usize,
    pn_quantile: f32,
    gain: f32,
    alpha: f32,

    // PN wiring: each PN reads `pn_fanin` features (idx, weight).
    pn_feat_idx: Vec<u32>,
    pn_feat_w: Vec<f32>,
    // Optional action tap per PN: feature dim (u32::MAX = none) and sign.
    pn_action_col: Vec<u32>,
    pn_action_sign: Vec<f32>,

    // Calibration state (measurement, not learning).
    feat_mean: Vec<f32>,
    feat_scale: Vec<f32>,
    pn_theta: Vec<f32>,
    pn_scale: Vec<f32>,
    action_gain: f32,

    // KC wiring: each KC reads `kc_fanin` PNs, weights normalized to sum 1
    // (the fly's weighted-mean PN→KC convergence).
    kc_pn_idx: Vec<u32>,
    kc_pn_w: Vec<f32>,

    // KC→MBON wiring: per KC, `kc_mbon_fanin` distinct MBONs.
    kc_mbon_idx: Vec<u32>,
    /// Synapse initial weights = the upper bound `w0` (per-synapse layout,
    /// parallel to `kc_mbon_idx`).
    w0: Vec<f32>,
    /// Learned weights, bounded `[0, w0]` by construction.
    w: Vec<f32>,

    /// Per-MBON readout gain: `+gain/n_app` approach, `−gain/n_avd` avoid.
    readout: Vec<f32>,
    /// True for avoid (PAM) compartments.
    avoid: Vec<bool>,
    eta: f32,
}

impl MbCircuit {
    /// Build the circuit from `config` (wiring generated from the seed).
    /// Calibration state defaults to identity and `eta` is 0 until
    /// [`MbCircuit::calibrate`] — updates are no-ops before that.
    pub fn new(config: &MbCircuitConfig) -> Self {
        let c = config.validated();
        let mut rng = WireRng::new(c.seed);
        let cap = c.pn_fanin.max(c.kc_fanin).max(c.kc_mbon_fanin);
        let mut picked: Vec<u32> = Vec::with_capacity(cap);

        let mut pn_feat_idx = Vec::with_capacity(c.n_pn * c.pn_fanin);
        let mut pn_feat_w = Vec::with_capacity(c.n_pn * c.pn_fanin);
        for _ in 0..c.n_pn {
            rng.sample_distinct(c.n_features, c.pn_fanin, &mut picked);
            pn_feat_idx.extend_from_slice(&picked);
            for _ in 0..picked.len() {
                pn_feat_w.push(rng.normal());
            }
        }

        let mut pn_action_col = vec![u32::MAX; c.n_pn];
        let mut pn_action_sign = vec![0.0; c.n_pn];
        if c.n_action_dims > 0 {
            for j in 0..c.n_pn {
                if rng.uniform() < c.pn_action_fraction {
                    pn_action_col[j] = rng.below(c.n_action_dims) as u32;
                    pn_action_sign[j] = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                }
            }
        }

        let mut kc_pn_idx = Vec::with_capacity(c.n_kc * c.kc_fanin);
        let mut kc_pn_w = Vec::with_capacity(c.n_kc * c.kc_fanin);
        for _ in 0..c.n_kc {
            rng.sample_distinct(c.n_pn, c.kc_fanin, &mut picked);
            kc_pn_idx.extend_from_slice(&picked);
            let raw: Vec<f32> = (0..picked.len()).map(|_| 1.0 + rng.uniform()).collect();
            let sum: f32 = raw.iter().sum();
            for r in raw {
                kc_pn_w.push(r / sum.max(1e-9));
            }
        }

        let n_syn = c.n_kc * c.kc_mbon_fanin;
        let mut kc_mbon_idx = Vec::with_capacity(n_syn);
        for _ in 0..c.n_kc {
            rng.sample_distinct(c.n_mbon, c.kc_mbon_fanin, &mut picked);
            kc_mbon_idx.extend_from_slice(&picked);
        }

        let mut avoid = vec![false; c.n_mbon];
        for a in avoid.iter_mut().skip(c.n_approach) {
            *a = true;
        }

        Self {
            n_features: c.n_features,
            n_action_dims: c.n_action_dims,
            n_pn: c.n_pn,
            pn_fanin: c.pn_fanin,
            n_kc: c.n_kc,
            kc_fanin: c.kc_fanin,
            kc_active: c.kc_active,
            n_mbon: c.n_mbon,
            n_approach: c.n_approach,
            kc_mbon_fanin: c.kc_mbon_fanin,
            pn_quantile: c.pn_quantile,
            gain: c.gain,
            alpha: c.alpha,
            pn_feat_idx,
            pn_feat_w,
            pn_action_col,
            pn_action_sign,
            feat_mean: vec![0.0; c.n_features],
            feat_scale: vec![1.0; c.n_features],
            pn_theta: vec![0.0; c.n_pn],
            pn_scale: vec![1.0; c.n_pn],
            action_gain: if c.n_action_dims > 0 { 1.0 } else { 0.0 },
            kc_pn_idx,
            kc_pn_w,
            kc_mbon_idx,
            w0: vec![1.0; n_syn],
            w: vec![0.0; n_syn],
            readout: vec![0.0; c.n_mbon],
            avoid,
            eta: 0.0,
        }
    }

    // ── accessors ──────────────────────────────────────────────────────────

    /// Kenyon-cell count.
    pub fn n_kc(&self) -> usize {
        self.n_kc
    }

    /// MBON count.
    pub fn n_mbon(&self) -> usize {
        self.n_mbon
    }

    /// Active code width (top-k).
    pub fn kc_active(&self) -> usize {
        self.kc_active
    }

    /// Learned step (`alpha`-normalized at calibration; 0 until then).
    pub fn eta(&self) -> f32 {
        self.eta
    }

    /// The MBON targets of one KC's synapses (wiring accessor; distinct).
    pub fn kc_mbon_targets(&self, kc: usize) -> &[u32] {
        let a = kc * self.kc_mbon_fanin;
        &self.kc_mbon_idx[a..a + self.kc_mbon_fanin]
    }

    /// Learned weights (per-synapse layout, parallel to the wiring) — the
    /// freeze/thaw seam: BLAKE3 this slice to freeze, `set_w` to thaw.
    pub fn w(&self) -> &[f32] {
        &self.w
    }

    /// Synapse upper bounds (the `w0` the calibration normalized).
    pub fn w0(&self) -> &[f32] {
        &self.w0
    }

    /// Thaw: replace the learned weights. Panics on a length mismatch; the
    /// caller's frozen slice must come from the same circuit shape.
    pub fn set_w(&mut self, w: &[f32]) {
        assert_eq!(w.len(), self.w.len(), "set_w: shape mismatch");
        self.w.copy_from_slice(w);
    }

    // ── calibration (cold path; measurement, not learning) ─────────────────

    /// Calibrate from `features` (row-major `n_states × n_features`) and
    /// `action_codes` (row-major `K × n_action_dims`; empty when
    /// `n_action_dims == 0`): feature z-scores, PN thresholds/gains, the
    /// action-gain bisection to `target_action_overlap`, per-MBON `w0`
    /// normalization, readout gains, and the derived `eta`. Resets `w = w0`.
    pub fn calibrate(
        &mut self,
        features: &[f32],
        action_codes: &[f32],
        n_states: usize,
        target_action_overlap: f32,
    ) -> MbCalibration {
        assert!(n_states >= 1, "calibrate needs at least one state");
        assert_eq!(
            features.len(),
            n_states * self.n_features,
            "calibrate: features must be n_states × n_features"
        );
        let n_actions = if self.n_action_dims > 0 {
            assert!(
                !action_codes.is_empty() && action_codes.len().is_multiple_of(self.n_action_dims),
                "calibrate: action_codes must be K × n_action_dims"
            );
            action_codes.len() / self.n_action_dims
        } else {
            assert!(action_codes.is_empty());
            0
        };

        // Feature z-scores (population std, floor 0.05 like the source).
        for i in 0..self.n_features {
            let mut mean = 0.0f64;
            for s in 0..n_states {
                mean += features[s * self.n_features + i] as f64;
            }
            mean /= n_states as f64;
            let mut var = 0.0f64;
            for s in 0..n_states {
                let d = features[s * self.n_features + i] as f64 - mean;
                var += d * d;
            }
            self.feat_mean[i] = mean as f32;
            self.feat_scale[i] = ((var / n_states as f64).sqrt() as f32).max(0.05);
        }

        // Bisection on the action gain (skipped when there is no action code).
        if self.n_action_dims == 0 {
            self.action_gain = 0.0;
            self.set_pn_stats(features, action_codes, n_states, n_actions);
        } else {
            let (mut lo, mut hi) = (0.0f32, 64.0f32);
            for _ in 0..17 {
                let mid = 0.5 * (lo + hi);
                self.action_gain = mid;
                self.set_pn_stats(features, action_codes, n_states, n_actions);
                if self.mean_action_overlap(features, action_codes, n_states, n_actions)
                    > target_action_overlap
                {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            self.action_gain = 0.5 * (lo + hi);
            self.set_pn_stats(features, action_codes, n_states, n_actions);
        }
        let action_overlap = if n_actions > 1 {
            self.mean_action_overlap(features, action_codes, n_states, n_actions)
        } else {
            1.0
        };

        // Codes over the batch with random action assignments (the source's
        // calibration-drive distribution); collect per-MBON drive sums and
        // connected-pair counts.
        let mut rng = WireRng::new(0x5EED_00C7);
        let mut scratch = MbScratch::new(self);
        let mut code: Vec<u32> = vec![0u32; self.kc_active];
        let mut drive_sum = vec![0.0f64; self.n_mbon];
        let mut conn_count = vec![0u64; self.n_mbon];
        let mut pn_active = 0u64;
        for s in 0..n_states {
            let a = if n_actions > 0 {
                rng.below(n_actions) * self.n_action_dims
            } else {
                0
            };
            self.code_into(
                &features[s * self.n_features..(s + 1) * self.n_features],
                if n_actions > 0 {
                    &action_codes[a..a + self.n_action_dims]
                } else {
                    &[]
                },
                &mut scratch,
                &mut code,
            );
            pn_active += scratch.pn.iter().filter(|&&p| p > 0.0).count() as u64;
            for &kc in &code {
                let base = kc as usize * self.kc_mbon_fanin;
                for e in 0..self.kc_mbon_fanin {
                    let m = self.kc_mbon_idx[base + e] as usize;
                    drive_sum[m] += self.w0[base + e] as f64;
                    conn_count[m] += 1;
                }
            }
        }

        // Per-MBON w0 normalization: the MEAN drive over calibration codes
        // (drive_sum / (n_states · kc_active), the source's `drives.mean(0)`
        // with drives = (kc @ w)/kc_active) is 1 where connected.
        let code_norm = n_states as f64 * self.kc_active as f64;
        let mut scale = vec![0.0f32; self.n_mbon];
        let mut connected = vec![false; self.n_mbon];
        for m in 0..self.n_mbon {
            let mean_drive = drive_sum[m] / code_norm;
            connected[m] = mean_drive > 0.0;
            if connected[m] {
                scale[m] = (1.0 / mean_drive.max(1e-9)) as f32;
            }
        }
        for kc in 0..self.n_kc {
            let base = kc * self.kc_mbon_fanin;
            for e in 0..self.kc_mbon_fanin {
                let m = self.kc_mbon_idx[base + e] as usize;
                self.w0[base + e] *= scale[m];
            }
        }

        // Readout: equal group weight, V = 0 at start.
        let n_app = (0..self.n_approach).filter(|&m| connected[m]).count();
        let n_avd = (self.n_approach..self.n_mbon)
            .filter(|&m| connected[m])
            .count();
        for (m, r) in self.readout.iter_mut().enumerate() {
            *r = if !connected[m] {
                0.0
            } else if m < self.n_approach {
                if n_app > 0 {
                    self.gain / n_app as f32
                } else {
                    0.0
                }
            } else if n_avd > 0 {
                -self.gain / n_avd as f32
            } else {
                0.0
            };
        }

        // f_m = mean fraction of the active code connected to MBON m;
        // eta from the desired ΔV-per-RPE (scale-free across circuit sizes).
        let eff_app: f64 = (0..self.n_approach)
            .map(|m| (conn_count[m] as f64 / code_norm) * self.readout[m].max(0.0) as f64)
            .sum();
        let eff_avd: f64 = (self.n_approach..self.n_mbon)
            .map(|m| (conn_count[m] as f64 / code_norm) * (-self.readout[m]).max(0.0) as f64)
            .sum();
        let denom = eff_app + eff_avd;
        self.eta = if denom > 1e-12 {
            (self.alpha as f64 / denom) as f32
        } else {
            0.0
        };

        // Reset the learned state to the normalized start.
        self.w.copy_from_slice(&self.w0);

        MbCalibration {
            action_gain: self.action_gain,
            action_overlap,
            pn_active_fraction: pn_active as f32 / (n_states * self.n_pn) as f32,
            eta: self.eta,
            dv_per_rpe_approach: (self.gain as f64 * self.eta as f64 * eff_app) as f32,
            dv_per_rpe_avoid: (self.gain as f64 * self.eta as f64 * eff_avd) as f32,
            mbons_connected: connected.iter().filter(|&&c| c).count(),
            approach_mbons_used: n_app,
            avoid_mbons_used: n_avd,
        }
    }

    /// Recompute PN thresholds/gains at the current action gain.
    fn set_pn_stats(
        &mut self,
        features: &[f32],
        action_codes: &[f32],
        n_states: usize,
        n_actions: usize,
    ) {
        let mut rng = WireRng::new(0xC0FF_EE01);
        let mut pre = vec![0.0f32; self.n_pn];
        let mut pres = vec![0.0f32; n_states * self.n_pn];
        for s in 0..n_states {
            let a = if n_actions > 0 {
                rng.below(n_actions) * self.n_action_dims
            } else {
                0
            };
            self.pn_pre_into(
                &features[s * self.n_features..(s + 1) * self.n_features],
                if n_actions > 0 {
                    &action_codes[a..a + self.n_action_dims]
                } else {
                    &[]
                },
                &mut pre,
            );
            pres[s * self.n_pn..(s + 1) * self.n_pn].copy_from_slice(&pre);
        }
        let mut col = Vec::with_capacity(n_states);
        for j in 0..self.n_pn {
            col.clear();
            for s in 0..n_states {
                col.push(pres[s * self.n_pn + j]);
            }
            col.sort_by(|a, b| a.total_cmp(b));
            let qi = (self.pn_quantile * (n_states as f32 - 1.0)).round() as usize;
            self.pn_theta[j] = col[qi.min(n_states - 1)];
            let mut pos_sum = 0.0f64;
            let mut pos_n = 0u64;
            for s in 0..n_states {
                let v = pres[s * self.n_pn + j] - self.pn_theta[j];
                if v > 0.0 {
                    pos_sum += v as f64;
                    pos_n += 1;
                }
            }
            self.pn_scale[j] = if pos_n > 0 {
                (1.0 / (pos_sum / pos_n as f64).max(1e-9)) as f32
            } else {
                1.0
            };
        }
    }

    /// Mean overlap between codes of different actions at the same states
    /// (|code_i ∩ code_j| / kc_active, averaged over states and action
    /// pairs; subsampled to ≤1024 states like the source).
    fn mean_action_overlap(
        &self,
        features: &[f32],
        action_codes: &[f32],
        n_states: usize,
        n_actions: usize,
    ) -> f32 {
        if n_actions < 2 {
            return 1.0;
        }
        let step = (n_states / 1024).max(1);
        let mut scratch = MbScratch::new(self);
        let mut ca = vec![0u32; self.kc_active];
        let mut cb = vec![0u32; self.kc_active];
        let mut marks = vec![false; self.n_kc];
        let mut total = 0.0f64;
        let mut pairs = 0u64;
        for s in (0..n_states).step_by(step) {
            let feats = &features[s * self.n_features..(s + 1) * self.n_features];
            for ai in 0..n_actions {
                self.code_into(
                    feats,
                    &action_codes[ai * self.n_action_dims..(ai + 1) * self.n_action_dims],
                    &mut scratch,
                    &mut ca,
                );
                marks[..].fill(false);
                for &kc in &ca {
                    marks[kc as usize] = true;
                }
                for aj in (ai + 1)..n_actions {
                    self.code_into(
                        feats,
                        &action_codes[aj * self.n_action_dims..(aj + 1) * self.n_action_dims],
                        &mut scratch,
                        &mut cb,
                    );
                    let inter = cb.iter().filter(|&&kc| marks[kc as usize]).count();
                    total += inter as f64 / self.kc_active as f64;
                    pairs += 1;
                }
            }
        }
        if pairs == 0 {
            1.0
        } else {
            (total / pairs as f64) as f32
        }
    }

    // ── hot paths (zero-alloc; scratch + code buffer are caller-owned) ─────

    /// PN pre-activations (z-scored features → sparse weighted rows +
    /// action-code taps). `action` may be empty when `n_action_dims == 0`.
    #[inline]
    fn pn_pre_into(&self, features: &[f32], action: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.n_pn);
        for (j, o) in out.iter_mut().enumerate() {
            let base = j * self.pn_fanin;
            let mut acc = 0.0f32;
            for f in 0..self.pn_fanin {
                let idx = self.pn_feat_idx[base + f] as usize;
                let x = (features[idx] - self.feat_mean[idx]) / self.feat_scale[idx];
                acc += self.pn_feat_w[base + f] * x;
            }
            let col = self.pn_action_col[j];
            if col != u32::MAX {
                acc += self.action_gain * action[col as usize] * self.pn_action_sign[j];
            }
            *o = acc;
        }
    }

    /// Canonical top-k via `select_nth_unstable_by` under the TOTAL order
    /// (drive, idx) — no ties are possible in a total order, so the selected
    /// SET is exactly the full-sort reference set (pinned by the G1 tests
    /// against [`MbCircuit::code_reference_into`]). Zero-alloc.
    pub fn code_into(
        &self,
        features: &[f32],
        action: &[f32],
        scratch: &mut MbScratch,
        code_out: &mut [u32],
    ) {
        debug_assert_eq!(features.len(), self.n_features);
        debug_assert_eq!(action.len(), self.n_action_dims);
        debug_assert_eq!(code_out.len(), self.kc_active);
        self.pn_pre_into(features, action, &mut scratch.pn);
        for j in 0..self.n_pn {
            let v = scratch.pn[j] - self.pn_theta[j];
            scratch.pn[j] = if v > 0.0 { v * self.pn_scale[j] } else { 0.0 };
        }
        for c in 0..self.n_kc {
            let base = c * self.kc_fanin;
            let mut d = 0.0f32;
            for f in 0..self.kc_fanin {
                d += self.kc_pn_w[base + f] * scratch.pn[self.kc_pn_idx[base + f] as usize];
            }
            scratch.drives[c] = d;
        }
        // Split-borrow the scratch fields (drives shared, order mutable)
        // so the comparators can read drives while order is reordered.
        let MbScratch { drives, order, .. } = &mut *scratch;
        for (i, o) in order.iter_mut().enumerate() {
            *o = i as u32;
        }
        let k = self.kc_active;
        order.select_nth_unstable_by(k - 1, |&a, &b| {
            let (va, vb) = (drives[a as usize], drives[b as usize]);
            vb.total_cmp(&va).then(a.cmp(&b)) // better ranks "less"
        });
        let (winners, _) = order.split_at_mut(k);
        winners.sort_unstable_by(|&a, &b| {
            let (va, vb) = (drives[a as usize], drives[b as usize]);
            vb.total_cmp(&va).then(a.cmp(&b))
        });
        code_out.copy_from_slice(winners);
    }

    /// The FULL-SORT reference path (the `step_dense` precedent): identical
    /// forward computation, selection by sorting all of `order` under the
    /// same total order, then taking the first `kc_active`. O(n log n) where
    /// [`MbCircuit::code_into`] is O(n + k log k) — kept pub so consumers
    /// and the G1 gate can cross-check the fast path bit-for-bit.
    pub fn code_reference_into(
        &self,
        features: &[f32],
        action: &[f32],
        scratch: &mut MbScratch,
        code_out: &mut [u32],
    ) {
        debug_assert_eq!(features.len(), self.n_features);
        debug_assert_eq!(action.len(), self.n_action_dims);
        debug_assert_eq!(code_out.len(), self.kc_active);
        self.pn_pre_into(features, action, &mut scratch.pn);
        for j in 0..self.n_pn {
            let v = scratch.pn[j] - self.pn_theta[j];
            scratch.pn[j] = if v > 0.0 { v * self.pn_scale[j] } else { 0.0 };
        }
        for c in 0..self.n_kc {
            let base = c * self.kc_fanin;
            let mut d = 0.0f32;
            for f in 0..self.kc_fanin {
                d += self.kc_pn_w[base + f] * scratch.pn[self.kc_pn_idx[base + f] as usize];
            }
            scratch.drives[c] = d;
        }
        let MbScratch { drives, order, .. } = &mut *scratch;
        for (i, o) in order.iter_mut().enumerate() {
            *o = i as u32;
        }
        order.sort_unstable_by(|&a, &b| {
            let (va, vb) = (drives[a as usize], drives[b as usize]);
            vb.total_cmp(&va).then(a.cmp(&b))
        });
        code_out.copy_from_slice(&order[..self.kc_active]);
    }

    /// MBON drives of a code into `scratch.acc` (raw, pre-readout).
    pub fn drives_into(&self, code: &[u32], scratch: &mut MbScratch) {
        scratch.acc[..].fill(0.0);
        for &kc in code {
            let base = kc as usize * self.kc_mbon_fanin;
            for e in 0..self.kc_mbon_fanin {
                scratch.acc[self.kc_mbon_idx[base + e] as usize] += self.w[base + e];
            }
        }
    }

    /// Value of a code: approach-minus-avoid readout (V = 0 at calibration).
    pub fn value(&self, code: &[u32], scratch: &mut MbScratch) -> f32 {
        self.drives_into(code, scratch);
        let mut v = 0.0f32;
        for m in 0..self.n_mbon {
            v += self.readout[m] * scratch.acc[m];
        }
        v / self.kc_active as f32
    }

    /// The learning rule: `w ← clamp(w − η · code_active · RPE · sign, 0,
    /// w0)` over the active code's synapses. Returns mean |Δw| over ALL
    /// existing synapses (the source's observability axis). Non-finite RPE
    /// is treated as a no-op (0.0) — the bound `[0, w0]` holds regardless.
    pub fn dopamine_update(&mut self, code: &[u32], rpe: f32) -> f32 {
        if !rpe.is_finite() || self.eta == 0.0 {
            return 0.0;
        }
        let mut dw_sum = 0.0f64;
        for &kc in code {
            let base = kc as usize * self.kc_mbon_fanin;
            for e in 0..self.kc_mbon_fanin {
                let m = self.kc_mbon_idx[base + e] as usize;
                let sign = if self.avoid[m] { 1.0 } else { -1.0 };
                let old = self.w[base + e];
                let nw = (old - self.eta * rpe * sign).clamp(0.0, self.w0[base + e]);
                dw_sum += (nw - old).abs() as f64;
                self.w[base + e] = nw;
            }
        }
        (dw_sum / self.w.len() as f64) as f32
    }

    /// (floor, ceiling): fraction of existing synapses at 0 and at `w0` —
    /// the "how much has this circuit learned" observability pair.
    pub fn saturation(&self) -> (f32, f32) {
        let mut floor = 0u64;
        let mut ceiling = 0u64;
        let mut exist = 0u64;
        for k in 0..self.w.len() {
            if self.w0[k] > 0.0 {
                exist += 1;
                if self.w[k] <= 0.0 {
                    floor += 1;
                } else if self.w[k] >= self.w0[k] {
                    ceiling += 1;
                }
            }
        }
        let n = exist.max(1) as f32;
        (floor as f32 / n, ceiling as f32 / n)
    }
}

// ─── Scratch ────────────────────────────────────────────────────────────────

/// Pre-allocated scratch for the hot paths (the KARC `KarcScratch`
/// convention). One per execution context, `MbScratch::new` sizes it to the
/// circuit; the top-k code buffer is caller-owned and passed to
/// [`MbCircuit::code_into`] separately (avoids the returned-slice borrow
/// puzzle when the same scratch then feeds `value`).
#[derive(Debug, Clone)]
pub struct MbScratch {
    pn: Vec<f32>,
    drives: Vec<f32>,
    order: Vec<u32>,
    acc: Vec<f32>,
}

impl MbScratch {
    /// Sized for `circuit`.
    pub fn new(circuit: &MbCircuit) -> Self {
        Self {
            pn: vec![0.0; circuit.n_pn],
            drives: vec![0.0; circuit.n_kc],
            order: vec![0; circuit.n_kc],
            acc: vec![0.0; circuit.n_mbon],
        }
    }
}
