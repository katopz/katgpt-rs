//! The Issue-980 / Bench-976 seeded-degeneracy fixture, shared by the 980
//! gate battery and the Issue-993 LP-authoring gate — ONE definition, so
//! the 993 battery's "existing gate battery arms" claim reads the same
//! world.
//!
//! `forward_ext` / `rollout_trace` / `rollout_with_bias` are the 993-side
//! additions: a y2-bias hook for the steering probe (the frozen-model
//! quadrant) and an h1-recording rollout for trace authoring. Each
//! consumer uses a subset of this module.
#![allow(dead_code)]

use katgpt_core::weight_edit_overlay::{EditForm, WeightEditOverlay, WeightEditSpec};

// ─────────────────────────────────────────────────────────────────────
// Fixture constants (the analytic shape in the 980 gate file's header —
// do not tweak without re-verifying the attractor at every start bucket)
// ─────────────────────────────────────────────────────────────────────

pub const IN: usize = 128;
pub const H: usize = 64;
pub const LAYER_W2: u8 = 1;
pub const CARRIER_COL: u32 = 4;
pub const ISOLATION_COL: u32 = 5;
pub const DIR_CHANNEL: usize = 1;
pub const C_V: f32 = 1.6;
pub const C_H: f32 = 0.5;
pub const CARRIER_GAIN: f32 = 4.0;
pub const CARRIER_OFFSET: f32 = 0.05;
pub const INJECT_K0: f32 = 8.0;
pub const REGION_HALF: f32 = 4.0;
pub const GOAL_X: f32 = 6.0;
pub const MAX_STEPS: usize = 64;
pub const EVAL_SEEDS: usize = 16;
pub const RANDOM_ARMS: usize = 16;
pub const STARTS_PER_SEED: usize = 12;
pub const DOSE_GRID: [f32; 4] = [1.0, 2.0, 4.0, 8.0];

// ─────────────────────────────────────────────────────────────────────
// Fixture MLP + world
// ─────────────────────────────────────────────────────────────────────

/// Deterministic splitmix64-driven LCG for seeded start draws (test-only;
/// the shipped module has its own SplitMix64 — this is fixture code).
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    pub fn bucket(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 33) % n.max(1)
    }
}

pub struct Fixture {
    pub w1: Vec<f32>, // H × IN row-major
    pub w2: Vec<f32>, // H × H row-major
    pub w3: Vec<f32>, // 4 × H row-major
}

pub fn build_fixture(inject: bool) -> Fixture {
    let mut w1 = vec![0.0f32; H * IN];
    for i in 0..6 {
        w1[i * IN + i] = 1.0;
    }
    let mut w2 = vec![0.0f32; H * H];
    for i in 0..H {
        w2[i * H + i] = 1.0;
    }
    if inject {
        // ΔW2[:, carrier] = −k₀·d with d = −e₁ → W2[1, carrier] = −k₀.
        w2[DIR_CHANNEL * H + CARRIER_COL as usize] -= INJECT_K0;
    }
    let mut w3 = vec![0.0f32; 4 * H];
    // down − up gap = C_V · h2[1]
    w3[DIR_CHANNEL] = C_V / -2.0;
    w3[H + DIR_CHANNEL] = C_V / 2.0;
    // right − left gap = C_H · (h2[2] − h2[0])
    w3[2 * H] = C_H / 2.0;
    w3[2 * H + 2] = C_H / -2.0;
    w3[3 * H] = C_H / -2.0;
    w3[3 * H + 2] = C_H / 2.0;
    Fixture { w1, w2, w3 }
}

impl Fixture {
    /// The targeted repair direction: the injected loop direction
    /// `d = −e₁` (unit norm, vertical-decision channel).
    pub fn targeted_direction() -> Vec<f32> {
        let mut d = vec![0.0f32; H];
        d[DIR_CHANNEL] = -1.0;
        d
    }

    pub fn targeted_overlay(strength: f32) -> WeightEditOverlay {
        WeightEditOverlay::new(vec![WeightEditSpec {
            layer: LAYER_W2,
            form: EditForm::DirectionSubtraction {
                columns: vec![CARRIER_COL],
                direction: Self::targeted_direction(),
                strength,
            },
        }])
        .expect("targeted overlay valid")
    }

    /// Forward through the MLP. `overlay_delta` applies the lane's
    /// forward-domain (dequant-posture) delta on layer 1's pre-activation.
    /// Delegates to [`Fixture::forward_ext`] with zero bias — bit-identical
    /// to the pre-extraction path.
    pub fn forward(
        &self,
        x: &[f32],
        overlay_delta: Option<&WeightEditOverlay>,
        h1_out: Option<&mut Vec<f32>>,
        h2_out: Option<&mut Vec<f32>>,
    ) -> [f32; 4] {
        self.forward_ext(x, overlay_delta, 0.0, h1_out, h2_out)
    }

    /// The general forward: `y2[i] += y2_bias` before the second tanh (the
    /// steering-probe hook — a constant inference-time push on a decision
    /// channel, the frozen-model quadrant). `y2_bias == 0.0` skips the add
    /// entirely, keeping the plain path bit-identical.
    pub fn forward_ext(
        &self,
        x: &[f32],
        overlay_delta: Option<&WeightEditOverlay>,
        y2_bias: f32,
        h1_out: Option<&mut Vec<f32>>,
        h2_out: Option<&mut Vec<f32>>,
    ) -> [f32; 4] {
        let mut h1 = vec![0.0f32; H];
        for (i, slot) in h1.iter_mut().enumerate() {
            let row = &self.w1[i * IN..(i + 1) * IN];
            let mut acc = 0.0f32;
            for (w, xv) in row.iter().zip(x.iter()) {
                acc += w * xv;
            }
            *slot = acc.tanh();
        }
        let mut y2 = vec![0.0f32; H];
        for (i, slot) in y2.iter_mut().enumerate() {
            let row = &self.w2[i * H..(i + 1) * H];
            let mut acc = 0.0f32;
            for (w, hv) in row.iter().zip(h1.iter()) {
                acc += w * hv;
            }
            *slot = acc;
        }
        if let Some(ov) = overlay_delta {
            ov.forward_delta_into(LAYER_W2, &h1, &mut y2);
        }
        if y2_bias != 0.0 {
            for slot in y2.iter_mut() {
                *slot += y2_bias;
            }
        }
        let mut h2 = vec![0.0f32; H];
        for (slot, yv) in h2.iter_mut().zip(y2.iter()) {
            *slot = yv.tanh();
        }
        let mut logits = [0.0f32; 4];
        for (l, slot) in logits.iter_mut().enumerate() {
            let row = &self.w3[l * H..(l + 1) * H];
            let mut acc = 0.0f32;
            for (w, hv) in row.iter().zip(h2.iter()) {
                acc += w * hv;
            }
            *slot = acc;
        }
        if let Some(o) = h1_out {
            *o = h1;
        }
        if let Some(o) = h2_out {
            *o = h2;
        }
        logits
    }
}

pub fn make_input(x: f32, y: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; IN];
    v[0] = x / 8.0;
    v[1] = y / 8.0;
    v[2] = GOAL_X / 8.0;
    let bump = if y.abs() <= REGION_HALF { 1.0 } else { 0.0 };
    v[4] = bump * (y / 8.0 + CARRIER_OFFSET) * CARRIER_GAIN;
    v[5] = 0.02;
    v
}

pub fn argmax_first_index(logits: &[f32; 4]) -> usize {
    let mut best = 0usize;
    for a in 1..4 {
        if logits[a] > logits[best] {
            best = a;
        }
    }
    best
}

pub struct Rollout {
    pub actions: Vec<usize>,
    pub success: bool,
    pub hidden: Vec<Vec<f32>>,
}

/// Deterministic greedy rollout. Actions: 0=up (y+1), 1=down (y−1),
/// 2=left (x−1), 3=right (x+1). Success: x ≥ GOAL_X.
pub fn rollout(model: &Fixture, overlay: Option<&WeightEditOverlay>, start: (f32, f32)) -> Rollout {
    let mut x = start.0;
    let mut y = start.1;
    let mut actions = Vec::with_capacity(MAX_STEPS);
    let mut hidden = Vec::with_capacity(MAX_STEPS);
    let mut success = false;
    for _ in 0..MAX_STEPS {
        let inp = make_input(x, y);
        let mut h2 = Vec::new();
        let logits = model.forward(&inp, overlay, None, Some(&mut h2));
        let a = argmax_first_index(&logits);
        match a {
            0 => y += 1.0,
            1 => y -= 1.0,
            2 => x -= 1.0,
            _ => x += 1.0,
        }
        actions.push(a);
        hidden.push(h2);
        if x >= GOAL_X {
            success = true;
            break;
        }
    }
    Rollout {
        actions,
        success,
        hidden,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Issue-993 additions: trace authoring + the steering probe
// ─────────────────────────────────────────────────────────────────────

/// One decision step of a traced rollout: the pre-move state, the h1 the
/// policy saw, and the chosen action.
pub struct TraceStep {
    pub x: f32,
    pub y: f32,
    pub h1: Vec<f32>,
    pub action: usize,
}

pub struct TraceRollout {
    pub steps: Vec<TraceStep>,
    pub success: bool,
}

/// [`rollout`] with per-step layer-1 trace recording (the LP authoring
/// input: `h1` is the W2-input the per-column coefficients act through).
/// EXACT mirror of [`rollout`]'s loop — same inputs, same argmax, same
/// move semantics, same stop condition.
pub fn rollout_trace(
    model: &Fixture,
    overlay: Option<&WeightEditOverlay>,
    start: (f32, f32),
) -> TraceRollout {
    let mut x = start.0;
    let mut y = start.1;
    let mut steps = Vec::with_capacity(MAX_STEPS);
    let mut success = false;
    for _ in 0..MAX_STEPS {
        let inp = make_input(x, y);
        let mut h1 = Vec::new();
        let logits = model.forward(&inp, overlay, Some(&mut h1), None);
        let a = argmax_first_index(&logits);
        steps.push(TraceStep {
            x,
            y,
            h1,
            action: a,
        });
        match a {
            0 => y += 1.0,
            1 => y -= 1.0,
            2 => x -= 1.0,
            _ => x += 1.0,
        }
        if x >= GOAL_X {
            success = true;
            break;
        }
    }
    TraceRollout { steps, success }
}

/// [`rollout`] with a constant inference-time push `y2_bias` added to
/// every y2 entry before the second tanh — the steering-probe arm (the
/// frozen-model quadrant: inject at inference, weights untouched).
pub fn rollout_with_bias(model: &Fixture, start: (f32, f32), y2_bias: f32) -> Rollout {
    let mut x = start.0;
    let mut y = start.1;
    let mut actions = Vec::with_capacity(MAX_STEPS);
    let mut hidden = Vec::with_capacity(MAX_STEPS);
    let mut success = false;
    for _ in 0..MAX_STEPS {
        let inp = make_input(x, y);
        let mut h2 = Vec::new();
        let logits = model.forward_ext(&inp, None, y2_bias, None, Some(&mut h2));
        let a = argmax_first_index(&logits);
        match a {
            0 => y += 1.0,
            1 => y -= 1.0,
            2 => x -= 1.0,
            _ => x += 1.0,
        }
        actions.push(a);
        hidden.push(h2);
        if x >= GOAL_X {
            success = true;
            break;
        }
    }
    Rollout {
        actions,
        success,
        hidden,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Battery statistics (moved verbatim from the 980 gate file)
// ─────────────────────────────────────────────────────────────────────

/// The pinned obstacle statistic: a 3-length vertical alternation
/// (up/down/up or down/up/down) — two consecutive opposite pairs, matching
/// the halter's patience-2 oscillation channel.
pub fn episode_oscillates(actions: &[usize]) -> bool {
    if actions.len() < 3 {
        return false;
    }
    for w in actions.windows(3) {
        let vert = |a: &usize| *a == 0 || *a == 1;
        if vert(&w[0]) && vert(&w[1]) && vert(&w[2]) && w[0] != w[1] && w[1] != w[2] {
            return true;
        }
    }
    false
}

/// LE byte-view hash of an f32 slice (BLAKE3 pinning helper — safe, no
/// reinterpret: each f32 folded through its `to_le_bytes`).
pub fn f32_digest(v: &[f32]) -> blake3::Hash {
    let mut h = blake3::Hasher::new();
    for x in v {
        h.update(&x.to_le_bytes());
    }
    h.finalize()
}

/// BLAKE3 over an i8 slice (folded through LE bytes — safe, no reinterpret).
pub fn i8_digest(v: &[i8]) -> blake3::Hash {
    let mut h = blake3::Hasher::new();
    for x in v {
        h.update(&[*x as u8]);
    }
    h.finalize()
}

pub fn in_region_starts() -> Vec<(f32, f32)> {
    let xs = [-7.0f32, -5.0, -3.0, -1.0];
    let ys = [-3.0f32, -1.0, 1.0, 3.0];
    let mut v = Vec::new();
    for x in xs {
        for y in ys {
            v.push((x, y));
        }
    }
    v
}

pub fn clean_starts() -> Vec<(f32, f32)> {
    let xs = [-6.0f32, -3.0, 0.0];
    let ys = [-7.0f32, -6.0, -5.0, 5.0, 6.0, 7.0];
    let mut v = Vec::new();
    for x in xs {
        for y in ys {
            v.push((x, y));
        }
    }
    v
}

/// Seeded per-eval-seed start list (deterministic draws over the in-region
/// grid) — the pairing granularity for the control battery.
pub fn seeded_starts(seed_index: u64) -> Vec<(f32, f32)> {
    let mut rng = Lcg::new(katgpt_core::blake3_seed(b"issue980/eval", seed_index));
    let xs = [-7.0f32, -5.0, -3.0, -1.0];
    let ys = [-3.0f32, -1.0, 1.0, 3.0];
    (0..STARTS_PER_SEED)
        .map(|_| {
            let x = xs[(rng.bucket(4)) as usize];
            let y = ys[(rng.bucket(4)) as usize];
            (x, y)
        })
        .collect()
}

/// Obstacle rate + per-episode oscillation over a start list.
pub fn obstacle_rate(
    model: &Fixture,
    overlay: Option<&WeightEditOverlay>,
    starts: &[(f32, f32)],
) -> (f32, usize) {
    let mut osc = 0usize;
    for s in starts {
        let r = rollout(model, overlay, *s);
        if episode_oscillates(&r.actions) {
            osc += 1;
        }
    }
    (osc as f32 / starts.len() as f32, osc)
}

/// The dequant twin: W2 quantized symmetric per-row int8 (deterministic).
pub struct DequantTwin {
    pub q: Vec<i8>,
    pub scale: Vec<f32>,
    pub w1: Vec<f32>,
    pub w3: Vec<f32>,
}

pub fn build_dequant_twin(base: &Fixture) -> DequantTwin {
    let mut q = vec![0i8; H * H];
    let mut scale = vec![1.0f32; H];
    for i in 0..H {
        let row_max = (0..H)
            .map(|j| base.w2[i * H + j].abs())
            .fold(0.0f32, f32::max);
        let s = if row_max > 0.0 { 127.0 / row_max } else { 1.0 };
        scale[i] = s;
        for j in 0..H {
            let v = (base.w2[i * H + j] * s).round();
            let v = v.clamp(-127.0, 127.0) as i8;
            q[i * H + j] = v;
        }
    }
    DequantTwin {
        q,
        scale,
        w1: base.w1.clone(),
        w3: base.w3.clone(),
    }
}

impl DequantTwin {
    /// Forward: dequantize W2 at use (+ the lane's forward-domain delta on
    /// layer 1). The packed bytes are NEVER touched — the QuantErrorLora
    /// posture.
    pub fn forward(&self, x: &[f32], overlay: Option<&WeightEditOverlay>) -> [f32; 4] {
        let mut h1 = vec![0.0f32; H];
        for (i, slot) in h1.iter_mut().enumerate() {
            let row = &self.w1[i * IN..(i + 1) * IN];
            let mut acc = 0.0f32;
            for (w, xv) in row.iter().zip(x.iter()) {
                acc += w * xv;
            }
            *slot = acc.tanh();
        }
        let mut y2 = vec![0.0f32; H];
        for (i, slot) in y2.iter_mut().enumerate() {
            let row = &self.q[i * H..(i + 1) * H];
            let mut acc = 0.0f32;
            for (q, hv) in row.iter().zip(h1.iter()) {
                acc += (*q as f32 / self.scale[i]) * hv;
            }
            *slot = acc;
        }
        if let Some(ov) = overlay {
            ov.forward_delta_into(LAYER_W2, &h1, &mut y2);
        }
        let mut h2 = vec![0.0f32; H];
        for (slot, yv) in h2.iter_mut().zip(y2.iter()) {
            *slot = yv.tanh();
        }
        let mut logits = [0.0f32; 4];
        for (l, slot) in logits.iter_mut().enumerate() {
            let row = &self.w3[l * H..(l + 1) * H];
            let mut acc = 0.0f32;
            for (w, hv) in row.iter().zip(h2.iter()) {
                acc += w * hv;
            }
            *slot = acc;
        }
        logits
    }
}
