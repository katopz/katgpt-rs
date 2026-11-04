//! Research 649 — the probe-authored weight-edit derivation PoC
//! (fn-grammar + params-vocab over the Issue-980 overlay lane).
//!
//! The 980 battery hand-authored its edit against a KNOWN fixture (the
//! carrier column, direction, and strength were construction constants);
//! Research 648's blog used a frontier LLM as the author. THIS lane is
//! probe-blind: the authoring procedure sees only rollout taps (positions,
//! activations, outcome), the readout rows, and the behavior statistic —
//! never the injected constants, which exist per-config as ground truth.
//!
//! Verdict-round-2 posture (the four REVISE fixes, all in this file):
//! 1. **The strength grid is DERIVED from the probe** — `derive_strength_
//!    grid(slope, margin=3, cap=12)`, never a hard-coded list that ends at
//!    the answer. The emitted k is whatever the window's dose sweep accepts.
//! 2. **P3 carries the MATCHED blind baseline** — a column sweep over the
//!    SAME ±e_gap words, the SAME derived grid, the SAME acceptance rule,
//!    given the gap row free — the strongest blind searcher. The probe's
//!    measured contribution is evals saved + the repair LOCUS (the blind
//!    sweep's first-found word is a compensating edit at a different
//!    column; the probe names the anomalous circuit directly). The random-
//!    direction families stay as the reachability control (LCB95).
//! 3. **Held-out injection configs** — the parse/emission design was never
//!    re-tuned per fixture: FOUR configs are pinned ahead of the run
//!    (BASE = the 980 original + three held-out carrier/gap/band/k₀
//!    combinations) and P1/P2 run unchanged on all of them.
//! 4. **Acceptance is the FAILURE rate, not oscillation absence** — an
//!    edit that turns the oscillation into a drift-failure does not count
//!    as a repair (the obstacle-vs-ability law, measured directly).
//!
//! "Probe-blind" therefore means: runtime-blind to the injected constants.
//! The statistic DESIGN (edge-contrast gate scan, LS slope, band grid
//! {2..6}, z-bar, slope window) is fixed across all four configs — that is
//! what the held-out sweep buys.
//!
//! Fixture: `InjectedDegeneracyMlp` — a 128→64→64→4 tanh policy MLP over a
//! 2-D grid world (the 980 shape, parametrized). The injection edits
//! `W2[gap_row, carrier_col]` (off-diagonal in every pinned config), the
//! carrier feature rides input `carrier_col` gated by `|y| ≤ band`, and the
//! vertical readout routes through `h2[gap_row]`.
//!
//! Green-zero rule: whole-file `#![cfg]` + `required-features` row pair.

#![cfg(feature = "probe_circuit_grammar")]

use std::cell::Cell;

use katgpt_core::probe_circuit_grammar::{
    CircuitBinding, GateConfig, ProbePools, StepTap, binding_to_spec, derive_strength_grid,
    discover_gated_drive, emit_word, parse_circuit,
};
use katgpt_core::weight_edit_overlay::{
    ArmVerdict, EditForm, ObstacleArmConfig, ProvenanceTrailer, WeightEditArtifact,
    WeightEditOverlay, WeightEditSpec, arm_from_telemetry, blake3_seed,
    norm_matched_random_direction, paired_lcb95,
};

// ─────────────────────────────────────────────────────────────────────
// Fixture constants (the 980 analytic shape) + the PINNED config set
// ─────────────────────────────────────────────────────────────────────

const IN: usize = 128;
const H: usize = 64;
const LAYER_W2: u8 = 1;
const ISOLATION_COL: u32 = 5;
const C_V: f32 = 1.6;
const C_H: f32 = 0.5;
const CARRIER_GAIN: f32 = 4.0;
const CARRIER_OFFSET: f32 = 0.05;
const GOAL_X: f32 = 6.0;
const MAX_STEPS: usize = 64;
const EVAL_SEEDS: usize = 16;
const RANDOM_ARMS: usize = 16;
const STARTS_PER_SEED: usize = 12;
/// The 980 dose grid — the G1a monotonicity statistic only (the emission
/// grid is probe-derived below).
const DOSE_GRID: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
/// Derived-grid parameters (pinned before any run): the slope CENTERS the
/// window, `margin` extends it on both sides, `cap` truncates loud.
const GRID_MARGIN: usize = 3;
const GRID_CAP: f32 = 12.0;

/// One pinned injection config. Ground truth — the parse never sees these.
///
/// Constraints (from the fixture's analytic shape, NOT tuned per run):
/// `carrier_col ∈ {3,4,5}` (disjoint from the x/y/goal inputs 0/1/2 and the
/// const input 5's slot semantics), `gap_row ∈ {1,3,5}` (rows 0/2 carry the
/// horizontal readout), `carrier_col ≠ gap_row` (the diag must survive, so
/// the injected entry is off-diagonal and the twin value is exactly 0).
#[derive(Clone, Copy, Debug)]
struct FixtureParams {
    carrier_col: usize,
    gap_row: usize,
    k0: f32,
    band: f32,
}

/// The 980 original (the base the PoC battery runs against).
const BASE: FixtureParams = FixtureParams {
    carrier_col: 4,
    gap_row: 1,
    k0: 8.0,
    band: 4.0,
};
/// Held-out B — different carrier AND gap row AND band AND strength.
const HELD_B: FixtureParams = FixtureParams {
    carrier_col: 3,
    gap_row: 5,
    k0: 5.0,
    band: 3.0,
};
/// Held-out C — the largest strength, the widest band, carrier on the
/// const-input slot (v[5] becomes the carrier; no const channel remains).
const HELD_C: FixtureParams = FixtureParams {
    carrier_col: 5,
    gap_row: 3,
    k0: 10.0,
    band: 5.0,
};
/// Held-out D — smallest strength, band 3, gap on the base row (the
/// confounded-slope case: the smooth y/8 channel rides the same diag row).
const HELD_D: FixtureParams = FixtureParams {
    carrier_col: 3,
    gap_row: 1,
    k0: 6.0,
    band: 3.0,
};

const ALL_CONFIGS: [(&str, FixtureParams); 4] = [
    ("BASE", BASE),
    ("HELD_B", HELD_B),
    ("HELD_C", HELD_C),
    ("HELD_D", HELD_D),
];

// ─────────────────────────────────────────────────────────────────────
// Fixture MLP + world
// ─────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn bucket(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 33) % n.max(1)
    }
}

struct Fixture {
    w1: Vec<f32>, // H × IN row-major
    w2: Vec<f32>, // H × H row-major
    w3: Vec<f32>, // 4 × H row-major
    params: FixtureParams,
}

fn build_fixture(inject: bool, p: &FixtureParams) -> Fixture {
    let mut w1 = vec![0.0f32; H * IN];
    for i in 0..6 {
        w1[i * IN + i] = 1.0;
    }
    let mut w2 = vec![0.0f32; H * H];
    for i in 0..H {
        w2[i * H + i] = 1.0;
    }
    if inject {
        // ΔW2[gap, carrier] = −k₀·(d = −e_gap) → W2[gap, carrier] = −k₀
        // (the twin's value there is 0 — every pinned entry is off-diagonal).
        w2[p.gap_row * H + p.carrier_col] -= p.k0;
    }
    let mut w3 = vec![0.0f32; 4 * H];
    // down−up gap = C_V · h2[gap_row] (vertical routes ONLY through the
    // pinned gap row; rows 0/2 stay horizontal-only).
    w3[p.gap_row] = C_V / -2.0;
    w3[H + p.gap_row] = C_V / 2.0;
    // right − left gap = C_H · (h2[2] − h2[0])
    w3[2 * H] = C_H / 2.0;
    w3[2 * H + 2] = C_H / -2.0;
    w3[3 * H] = C_H / -2.0;
    w3[3 * H + 2] = C_H / 2.0;
    Fixture {
        w1,
        w2,
        w3,
        params: *p,
    }
}

fn make_input(x: f32, y: f32, p: &FixtureParams) -> Vec<f32> {
    let mut v = vec![0.0f32; IN];
    v[0] = x / 8.0;
    v[1] = y / 8.0;
    v[2] = GOAL_X / 8.0;
    let bump = if y.abs() <= p.band { 1.0 } else { 0.0 };
    v[p.carrier_col] = bump * (y / 8.0 + CARRIER_OFFSET) * CARRIER_GAIN;
    // Gap rows 3/5 have no base input of their own; their diag channel
    // needs the smooth y/8 input so the twin's centering behavior exists
    // and the degenerate oscillation SIGN-FLIPS in y2[gap] (active↔zero
    // swings never push the consecutive-state cosine negative — the halter
    // would not arm; measured on the first HELD_C run). Row 1 already
    // rides v[1].
    if p.gap_row == 3 || p.gap_row == 5 {
        v[p.gap_row] = y / 8.0;
    }
    v
}

impl Fixture {
    /// Forward with taps: logits + h1 (layer-1 activations) + y2 (layer-2
    /// PRE-activations — where the linear coupling lives).
    fn forward_tapped(
        &self,
        x: &[f32],
        overlay: Option<&WeightEditOverlay>,
    ) -> ([f32; 4], Vec<f32>, Vec<f32>) {
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
        (logits, h1, y2)
    }
}

fn argmax_first_index(logits: &[f32; 4]) -> usize {
    let mut best = 0usize;
    for a in 1..4 {
        if logits[a] > logits[best] {
            best = a;
        }
    }
    best
}

struct Rollout {
    actions: Vec<usize>,
    success: bool,
}

fn rollout(model: &Fixture, overlay: Option<&WeightEditOverlay>, start: (f32, f32)) -> Rollout {
    let mut x = start.0;
    let mut y = start.1;
    let mut actions = Vec::with_capacity(MAX_STEPS);
    let mut success = false;
    for _ in 0..MAX_STEPS {
        let inp = make_input(x, y, &model.params);
        let (logits, _, _) = model.forward_tapped(&inp, overlay);
        let a = argmax_first_index(&logits);
        match a {
            0 => y += 1.0,
            1 => y -= 1.0,
            2 => x -= 1.0,
            _ => x += 1.0,
        }
        actions.push(a);
        if x >= GOAL_X {
            success = true;
            break;
        }
    }
    Rollout { actions, success }
}

/// The pinned obstacle statistic (the 980 reversal-streak): a 3-length
/// vertical alternation.
fn episode_oscillates(actions: &[usize]) -> bool {
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

fn in_region_starts() -> Vec<(f32, f32)> {
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

fn clean_starts() -> Vec<(f32, f32)> {
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

fn seeded_starts(seed_index: u64) -> Vec<(f32, f32)> {
    let mut rng = Lcg::new(blake3_seed(b"r649/eval", seed_index));
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

fn obstacle_rate(
    model: &Fixture,
    overlay: Option<&WeightEditOverlay>,
    starts: &[(f32, f32)],
) -> f32 {
    let mut osc = 0usize;
    for s in starts {
        if episode_oscillates(&rollout(model, overlay, *s).actions) {
            osc += 1;
        }
    }
    osc as f32 / starts.len() as f32
}

/// The ACCEPTANCE statistic: the fraction of episodes that never reach the
/// goal. An oscillation→drift edit scores 1.0 here (REJECTED) — the
/// obstacle-vs-ability law measured directly (verdict round 2, fix 4).
fn failure_rate(
    model: &Fixture,
    overlay: Option<&WeightEditOverlay>,
    starts: &[(f32, f32)],
) -> f32 {
    let mut fails = 0usize;
    for s in starts {
        if !rollout(model, overlay, *s).success {
            fails += 1;
        }
    }
    fails as f32 / starts.len() as f32
}

/// Hidden-state series for the halter (the 980 arming posture).
fn hidden_series(model: &Fixture, start: (f32, f32)) -> Vec<Vec<f32>> {
    let mut x = start.0;
    let mut y = start.1;
    let mut hidden = Vec::new();
    for _ in 0..MAX_STEPS {
        let inp = make_input(x, y, &model.params);
        let (logits, _, y2) = model.forward_tapped(&inp, None);
        hidden.push(y2);
        match argmax_first_index(&logits) {
            0 => y += 1.0,
            1 => y -= 1.0,
            2 => x -= 1.0,
            _ => x += 1.0,
        }
        if x >= GOAL_X {
            break;
        }
    }
    hidden
}

/// The probe capture: rollout the DEGENERATE policy over mixed starts,
/// tapping (gate_var = y, src = h1, dst_pre = y2) per step; split the pools
/// by the episode's OWN outcome.
fn capture_pools(model: &Fixture, starts: &[(f32, f32)]) -> ProbePools {
    let mut pools = ProbePools::default();
    for s in starts {
        let mut x = s.0;
        let mut y = s.1;
        let mut actions = Vec::with_capacity(MAX_STEPS);
        let mut steps = Vec::with_capacity(MAX_STEPS);
        let mut success = false;
        for _ in 0..MAX_STEPS {
            let inp = make_input(x, y, &model.params);
            let (logits, h1, y2) = model.forward_tapped(&inp, None);
            steps.push(StepTap {
                gate_var: y,
                src: h1,
                dst_pre: y2,
            });
            let a = argmax_first_index(&logits);
            match a {
                0 => y += 1.0,
                1 => y -= 1.0,
                2 => x -= 1.0,
                _ => x += 1.0,
            }
            actions.push(a);
            if x >= GOAL_X {
                success = true;
                break;
            }
        }
        let failed = !success || episode_oscillates(&actions);
        if failed {
            pools.failed.extend(steps);
        } else {
            pools.succeeded.extend(steps);
        }
    }
    pools
}

/// The vertical readout difference row (weights arithmetic — the lane may
/// read the network it edits). Params-independent: it READS W3.
fn vertical_readout_diff(model: &Fixture) -> Vec<f32> {
    let (down, up) = (1usize, 0usize);
    (0..H)
        .map(|j| model.w3[down * H + j] - model.w3[up * H + j])
        .collect()
}

fn random_arm_overlay(seed: u64, col: u32, k: f32) -> WeightEditOverlay {
    let mut d = vec![0.0f32; H];
    norm_matched_random_direction(seed, H, &mut d);
    WeightEditOverlay::new(vec![WeightEditSpec {
        layer: LAYER_W2,
        form: EditForm::DirectionSubtraction {
            columns: vec![col],
            direction: d,
            strength: k,
        },
    }])
    .expect("control arm overlay valid")
}

fn sign_arm_overlay(binding: &CircuitBinding, col: u32, sign: f32, k: f32) -> WeightEditOverlay {
    let mut d = vec![0.0f32; H];
    d[binding.gap_row] = sign;
    WeightEditOverlay::new(vec![WeightEditSpec {
        layer: LAYER_W2,
        form: EditForm::DirectionSubtraction {
            columns: vec![col],
            direction: d,
            strength: k,
        },
    }])
    .expect("sign arm overlay valid")
}

// ─────────────────────────────────────────────────────────────────────
// P0 — capture sanity (BASE)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn poc_p0_capture_sanity() {
    let degenerate = build_fixture(true, &BASE);
    let twin = build_fixture(false, &BASE);
    let pools = capture_pools(&degenerate, &in_region_starts());
    assert!(!pools.failed.is_empty(), "in-region starts must fail");
    assert!(
        pools.failed.iter().any(|t| t.gate_var.abs() <= 3.0),
        "failed episodes must traverse the in-band march"
    );
    assert!(
        pools.failed.iter().any(|t| t.gate_var.abs() >= 4.0),
        "failed episodes must reach the band edge"
    );
    // Clean starts ALSO fail under the degenerate policy (the vertical pull
    // drags them into the region) — they feed the failed pool with the
    // out-of-band steps the gate scan's out-pool needs.
    let clean_pools = capture_pools(&degenerate, &clean_starts());
    assert!(
        clean_pools.failed.iter().any(|t| t.gate_var.abs() >= 5.0),
        "clean-start episodes must contribute out-of-band steps (|y| ≥ 5)"
    );
    assert!(
        clean_pools.succeeded.is_empty(),
        "the degenerate policy must not succeed from clean starts"
    );
    // The twin (no injection) DOES succeed from clean starts — the ability
    // the obstacle hides.
    for s in clean_starts() {
        assert!(
            rollout(&twin, None, s).success,
            "twin must succeed at {s:?}"
        );
    }
    for t in pools.failed.iter().take(5) {
        assert_eq!(t.src.len(), H);
        assert_eq!(t.dst_pre.len(), H);
    }
}

// ─────────────────────────────────────────────────────────────────────
// P1 — probe-blind identification, ALL FOUR pinned configs
// ─────────────────────────────────────────────────────────────────────

#[test]
fn poc_p1_parses_all_pinned_configs() {
    let cfg_arm = ObstacleArmConfig::default();
    for (name, p) in ALL_CONFIGS {
        let degenerate = build_fixture(true, &p);
        let twin = build_fixture(false, &p);

        // Lane order: the SHIPPED arming precondition first — twin refuses,
        // degenerate arms (the obstacle-vs-ability falsifier).
        let episodes_deg: Vec<Vec<Vec<f32>>> = in_region_starts()
            .iter()
            .map(|s| hidden_series(&degenerate, *s))
            .collect();
        let episodes_twin: Vec<Vec<Vec<f32>>> = in_region_starts()
            .iter()
            .map(|s| hidden_series(&twin, *s))
            .collect();
        assert!(
            matches!(
                arm_from_telemetry(&episodes_deg, &cfg_arm),
                ArmVerdict::Armed { .. }
            ),
            "{name}: degenerate telemetry must arm the lane"
        );
        assert!(
            matches!(
                arm_from_telemetry(&episodes_twin, &cfg_arm),
                ArmVerdict::Refused(_)
            ),
            "{name}: twin telemetry must refuse before any parse"
        );

        // The parse: capture + readout rows ONLY.
        let pools = capture_pools(&degenerate, &in_region_starts());
        let readout = vertical_readout_diff(&degenerate);
        let binding =
            parse_circuit(&pools, H, &GateConfig::default(), &readout).unwrap_or_else(|e| {
                panic!("{name}: probe-blind parse must recover the circuit: {e:?}")
            });

        // Ground-truth assertions (the constants the lane was never given).
        assert_eq!(
            binding.drive_col, p.carrier_col as u32,
            "{name}: the gate scan must find the injected drive column blind"
        );
        assert_eq!(
            binding.gap_row, p.gap_row,
            "{name}: the gap row must be the vertical-decision channel"
        );
        assert_eq!(
            binding.band, p.band,
            "{name}: the accepted band must be the right grid cell"
        );
        assert!(
            (binding.slope - -p.k0).abs() <= 3.0,
            "{name}: the wire slope must land in the pre-registered window around −{}: {}",
            p.k0,
            binding.slope
        );
        // Exactly one candidate clears the bar on every pinned config.
        let all = discover_gated_drive(&pools, H, &GateConfig::default());
        assert_eq!(
            all.len(),
            1,
            "{name}: exactly one column may clear the gate bar: {all:?}"
        );
        // Determinism: same starts → same binding → same identity.
        let pools2 = capture_pools(&degenerate, &in_region_starts());
        let b2 = parse_circuit(&pools2, H, &GateConfig::default(), &readout)
            .unwrap_or_else(|e| panic!("{name}: re-parse failed: {e:?}"));
        assert_eq!(
            binding.digest(),
            b2.digest(),
            "{name}: the parse is deterministic"
        );
        println!(
            "P1 {name}: {} (gate_z {:.3}) — true (col {}, row {}, k₀ {}, band {})",
            binding.grammar_line(),
            binding.gate_z,
            p.carrier_col,
            p.gap_row,
            p.k0,
            p.band
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// P2 — behavior-verified emission + repair battery (BASE full battery)
// ─────────────────────────────────────────────────────────────────────

fn per_seed_rates(model: &Fixture, overlay: Option<&WeightEditOverlay>) -> Vec<f32> {
    (0..EVAL_SEEDS as u64)
        .map(|seed| {
            let starts = seeded_starts(seed);
            obstacle_rate(model, overlay, &starts)
        })
        .collect()
}

#[test]
fn poc_p2_base_emitted_word_repairs() {
    let degenerate = build_fixture(true, &BASE);
    let twin = build_fixture(false, &BASE);
    let pools = capture_pools(&degenerate, &in_region_starts());
    let readout = vertical_readout_diff(&degenerate);
    let binding = parse_circuit(&pools, H, &GateConfig::default(), &readout).expect("parse");

    // Emission: the grid is DERIVED from the slope (margin 3, cap 12 —
    // pinned above), acceptance is the FAILURE rate on a fixed authoring
    // subset. The slope estimate is NOT trusted for k — the window sweep
    // is the referee.
    let grid = derive_strength_grid(binding.slope, GRID_MARGIN, GRID_CAP);
    assert!(
        grid.contains(&(BASE.k0)),
        "the derived grid must cover the true strength: grid {grid:?}, k₀ {}",
        BASE.k0
    );
    let authoring: Vec<(f32, f32)> = in_region_starts()[..8].to_vec();
    let evals = Cell::new(0usize);
    let (spec, k_emit) = emit_word(&binding, LAYER_W2, H, &grid, 0.0, |ov| {
        evals.set(evals.get() + 1);
        failure_rate(&degenerate, Some(ov), &authoring)
    })
    .expect("the behavior referee must accept a word on the derived grid");
    println!(
        "P2 BASE: emitted k = {k_emit} (slope {:.4}, grid {grid:?}, {} authoring evals)",
        binding.slope,
        evals.get()
    );
    // Measured (twice, deterministic): the exact restore is the first word
    // that turns failure into success on this fixture.
    assert_eq!(k_emit, BASE.k0, "the referee must accept the exact restore");
    let EditForm::DirectionSubtraction {
        columns,
        direction,
        strength,
    } = &spec.form
    else {
        panic!("wrong form");
    };
    assert_eq!(columns, &[BASE.carrier_col as u32]);
    assert_eq!(strength, &k_emit);
    let mut want_dir = vec![0.0f32; H];
    want_dir[BASE.gap_row] = binding.slope.signum();
    assert_eq!(
        direction, &want_dir,
        "the word is sign(slope)·e_gap at the parsed column"
    );

    let overlay = WeightEditOverlay::new(vec![spec.clone()]).expect("overlay valid");

    // Full-eval repair: obstacle 0.0 AND failure 0.0 on all 16 seeded lists.
    let repaired_rates = per_seed_rates(&degenerate, Some(&overlay));
    let repaired_mean = repaired_rates.iter().sum::<f32>() / repaired_rates.len() as f32;
    assert_eq!(
        repaired_mean, 0.0,
        "the emitted word must clear every seeded eval list: {repaired_rates:?}"
    );
    assert_eq!(
        failure_rate(&degenerate, Some(&overlay), &in_region_starts()),
        0.0,
        "every repaired in-region episode must REACH the goal"
    );

    // Success parity with the twin (the ability un-hidden, not a new one).
    for s in in_region_starts() {
        assert!(
            rollout(&degenerate, Some(&overlay), s).success,
            "repaired policy must succeed at {s:?}"
        );
        assert!(
            rollout(&twin, None, s).success,
            "twin must succeed at {s:?}"
        );
    }

    // Dose-response with the EMITTED direction over the 980 grid
    // (oscillation statistic — the G1a monotonicity arm): non-increasing.
    let mut dose_rates = Vec::new();
    for &k in &DOSE_GRID {
        let ov = WeightEditOverlay::new(vec![WeightEditSpec {
            layer: LAYER_W2,
            form: EditForm::DirectionSubtraction {
                columns: vec![BASE.carrier_col as u32],
                direction: want_dir.clone(),
                strength: k,
            },
        }])
        .expect("dose overlay");
        dose_rates.push(obstacle_rate(&degenerate, Some(&ov), &in_region_starts()));
    }
    for w in dose_rates.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-9,
            "dose-response must be monotone non-increasing: {dose_rates:?}"
        );
    }
    println!("P2 BASE dose-response (emitted direction): {dose_rates:?}");

    // Isolation: the same word at the neighbor column must NOT repair.
    let isolation_spec = binding_to_spec(
        &CircuitBinding {
            drive_col: ISOLATION_COL,
            ..binding.clone()
        },
        LAYER_W2,
        H,
        k_emit,
    )
    .expect("isolation spec");
    let isolation_ov = WeightEditOverlay::new(vec![isolation_spec]).expect("isolation overlay");
    let iso_rate = obstacle_rate(&degenerate, Some(&isolation_ov), &in_region_starts());
    assert!(
        iso_rate >= 0.95,
        "the isolation column must not repair: rate {iso_rate}"
    );

    // Retention: OUT-OF-REGION the delta cannot act (carrier mass exactly
    // 0 → delta exactly 0): single forwards bit-identical, rollout action
    // prefixes identical up to the first in-region step.
    for s in clean_starts() {
        let inp = make_input(s.0, s.1, &degenerate.params);
        let (_, _, y2_bare) = degenerate.forward_tapped(&inp, None);
        let (_, _, y2_dressed) = degenerate.forward_tapped(&inp, Some(&overlay));
        assert_eq!(
            format!("{y2_bare:?}"),
            format!("{y2_dressed:?}"),
            "out-of-region forward must be bit-identical at {s:?}"
        );
        let bare = rollout(&degenerate, None, s);
        let dressed = rollout(&degenerate, Some(&overlay), s);
        let mut y = s.1;
        for (a_bare, a_dressed) in bare.actions.iter().zip(dressed.actions.iter()) {
            if y.abs() <= BASE.band {
                break; // the repair's jurisdiction ends the prefix
            }
            assert_eq!(a_bare, a_dressed, "out-of-region action prefix at {s:?}");
            match a_bare {
                0 => y += 1.0,
                1 => y -= 1.0,
                _ => {}
            }
        }
    }

    // Artifact: mint-once-then-freeze; the fn-grammar record rides as the
    // prompt digest (the record IS this lane's authoring prompt).
    let artifact = WeightEditArtifact::mint(
        &overlay,
        ProvenanceTrailer {
            prompt_digest: Some(binding.digest()),
            seed: Some(pools.total_steps() as u64),
            author_id: Some(blake3::hash(binding.grammar_line().as_bytes()).into()),
        },
    );
    assert!(artifact.verify_overlay(&overlay), "minted overlay verifies");
    let thawed = WeightEditArtifact::thaw(&artifact.to_bytes()).expect("thaw");
    assert_eq!(
        thawed.provenance().prompt_digest,
        Some(binding.digest()),
        "the provenance carries the fn-grammar record's digest"
    );
}

// ─────────────────────────────────────────────────────────────────────
// P2b — held-out configs: identification already proven (P1); the SAME
// pipeline must emit a word the referee accepts and the full eval clears.
// No per-config tuning: same GateConfig, same GRID_MARGIN/CAP, same
// acceptance statistic.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn poc_p2b_held_out_configs_repair() {
    for (name, p) in ALL_CONFIGS {
        if name == "BASE" {
            continue; // the full battery lives in poc_p2_base_emitted_word_repairs
        }
        let degenerate = build_fixture(true, &p);
        let twin = build_fixture(false, &p);
        let pools = capture_pools(&degenerate, &in_region_starts());
        let readout = vertical_readout_diff(&degenerate);
        let binding = parse_circuit(&pools, H, &GateConfig::default(), &readout)
            .unwrap_or_else(|e| panic!("{name}: parse failed: {e:?}"));
        let grid = derive_strength_grid(binding.slope, GRID_MARGIN, GRID_CAP);
        assert!(
            grid.contains(&(p.k0)),
            "{name}: derived grid must cover the true strength: {grid:?} vs k₀ {}",
            p.k0
        );
        let authoring: Vec<(f32, f32)> = in_region_starts()[..8].to_vec();
        let (spec, k_emit) = emit_word(&binding, LAYER_W2, H, &grid, 0.0, |ov| {
            failure_rate(&degenerate, Some(ov), &authoring)
        })
        .unwrap_or_else(|e| panic!("{name}: emission refused: {e:?}"));
        let overlay = WeightEditOverlay::new(vec![spec]).expect("overlay valid");
        assert_eq!(
            failure_rate(&degenerate, Some(&overlay), &in_region_starts()),
            0.0,
            "{name}: every repaired in-region episode must reach the goal"
        );
        assert_eq!(
            obstacle_rate(&degenerate, Some(&overlay), &in_region_starts()),
            0.0,
            "{name}: no oscillation may remain"
        );
        assert_eq!(
            failure_rate(&degenerate, Some(&overlay), &clean_starts()),
            0.0,
            "{name}: clean starts must succeed too"
        );
        for s in in_region_starts() {
            assert!(
                rollout(&twin, None, s).success,
                "{name}: twin must succeed at {s:?}"
            );
            assert!(
                rollout(&degenerate, Some(&overlay), s).success,
                "{name}: repaired policy must succeed at {s:?}"
            );
        }
        println!(
            "P2b {name}: emitted k = {k_emit} (true k₀ = {}, slope {:.4}, grid {grid:?})",
            p.k0, binding.slope
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// P3 — peek vs blind search (verdict round 2, fix 2): the reachability
// control (random directions) PLUS the MATCHED blind baseline (same words,
// same grid, same acceptance, gap row given free) with honest eval counts.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn poc_p3_peek_beats_blind_search() {
    let degenerate = build_fixture(true, &BASE);
    let pools = capture_pools(&degenerate, &in_region_starts());
    let readout = vertical_readout_diff(&degenerate);
    let binding = parse_circuit(&pools, H, &GateConfig::default(), &readout).expect("parse");
    let grid = derive_strength_grid(binding.slope, GRID_MARGIN, GRID_CAP);
    let authoring: Vec<(f32, f32)> = in_region_starts()[..8].to_vec();

    // ── The probe-guided arm: parse → behavior-verified word. ──
    let probe_evals = Cell::new(0usize);
    let (spec_probe, k_probe) = emit_word(&binding, LAYER_W2, H, &grid, 0.0, |ov| {
        probe_evals.set(probe_evals.get() + 1);
        failure_rate(&degenerate, Some(ov), &authoring)
    })
    .expect("probe arm emits");
    let probe_ov = WeightEditOverlay::new(vec![spec_probe]).expect("overlay");
    println!(
        "P3 probe arm: k = {k_probe}, {} authoring evals (one capture on top)",
        probe_evals.get()
    );
    let probe_rates = per_seed_rates(&degenerate, Some(&probe_ov));

    // ── Reachability controls (random directions), budget-matched. ──
    // Each arm dose-sweeps the SAME grid with the SAME acceptance, keeps
    // its authoring-best dose, then takes the SAME eval.
    let mut family_a = Vec::new();
    let mut family_b = Vec::new();
    for i in 0..RANDOM_ARMS as u64 {
        let seed = blake3_seed(b"r649/ctrl", i);
        let mut rng = Lcg::new(blake3_seed(b"r649/col", i));
        let blind_col = rng.bucket(H as u64) as u32;

        let mut best_a = (f32::INFINITY, grid[0]);
        for &k in &grid {
            let ov = random_arm_overlay(seed, BASE.carrier_col as u32, k);
            let r = failure_rate(&degenerate, Some(&ov), &authoring);
            if r < best_a.0 {
                best_a = (r, k);
            }
        }
        let ov_a = random_arm_overlay(seed, BASE.carrier_col as u32, best_a.1);
        family_a.push(per_seed_rates(&degenerate, Some(&ov_a)));

        let mut best_b = (f32::INFINITY, grid[0]);
        for &k in &grid {
            let ov = random_arm_overlay(seed + 1, blind_col, k);
            let r = failure_rate(&degenerate, Some(&ov), &authoring);
            if r < best_b.0 {
                best_b = (r, k);
            }
        }
        let ov_b = random_arm_overlay(seed + 1, blind_col, best_b.1);
        family_b.push(per_seed_rates(&degenerate, Some(&ov_b)));
    }

    // ── The MATCHED blind baseline: ±e_gap words, same grid, same
    //    acceptance, gap row GIVEN FREE (it reads the same readout rows).
    //    Enumerated columns ascending, sign − then +, first acceptance
    //    stops. This is the strongest blind searcher on this word family.
    let blind_evals = Cell::new(0usize);
    let mut blind_found: Option<(u32, f32, f32)> = None;
    'outer: for col in 0..H as u32 {
        for sign in [-1.0f32, 1.0] {
            for &k in &grid {
                blind_evals.set(blind_evals.get() + 1);
                let ov = sign_arm_overlay(&binding, col, sign, k);
                if failure_rate(&degenerate, Some(&ov), &authoring) <= 0.0 {
                    blind_found = Some((col, sign, k));
                    break 'outer;
                }
            }
        }
    }
    let (b_col, b_sign, b_k) =
        blind_found.expect("the matched blind sweep must eventually find a repair");
    let blind_ov = sign_arm_overlay(&binding, b_col, b_sign, b_k);
    println!(
        "P3 matched blind: found repair at (col {b_col}, sign {b_sign}, k {b_k}) after {} evals — locus {}",
        blind_evals.get(),
        if b_col == BASE.carrier_col as u32 {
            "SAME as the anomalous column"
        } else {
            "DIFFERENT from the anomalous column (a compensating edit)"
        }
    );

    // ── The verdicts. ──
    let mean = |rates: &[f32]| rates.iter().sum::<f32>() / rates.len() as f32;
    let probe_mean = mean(&probe_rates);
    let a_means: Vec<f32> = family_a.iter().map(|r| mean(r)).collect();
    let b_means: Vec<f32> = family_b.iter().map(|r| mean(r)).collect();
    let best_a = a_means.iter().cloned().fold(f32::INFINITY, f32::min);
    let best_b = b_means.iter().cloned().fold(f32::INFINITY, f32::min);
    let best_random = best_a.min(best_b);
    println!("── P3 peek-vs-blind (budget-matched) ──");
    println!("probe-guided mean failure rate:  {probe_mean:.4}");
    println!("best random (right-column) arm:  {best_a:.4}");
    println!("best random (blind-column) arm:  {best_b:.4}");
    println!(
        "matched blind sweep: {} evals vs probe's {} ({}×)",
        blind_evals.get(),
        probe_evals.get(),
        blind_evals.get() as f32 / probe_evals.get() as f32
    );

    // (1) Reachability: the probe word repairs; no random-direction arm
    // does (paired LCB95 against the once-picked best random arm).
    assert!(
        probe_mean < best_random,
        "the probe-guided edit must beat the best budget-matched random arm"
    );
    let best_idx = a_means
        .iter()
        .enumerate()
        .chain(
            b_means
                .iter()
                .enumerate()
                .map(|(i, m)| (i + RANDOM_ARMS, m)),
        )
        .min_by(|(ia, ma), (ib, mb)| {
            ma.partial_cmp(mb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ia.cmp(ib))
        })
        .map(|(i, _)| i)
        .expect("non-empty arms");
    let best_arm_rates = if best_idx < RANDOM_ARMS {
        &family_a[best_idx]
    } else {
        &family_b[best_idx - RANDOM_ARMS]
    };
    let deltas: Vec<f32> = best_arm_rates
        .iter()
        .zip(probe_rates.iter())
        .map(|(r, p)| r - p)
        .collect();
    let lcb = paired_lcb95(&deltas).expect("paired deltas");
    println!("paired LCB95(best random − probe) = {lcb:.4}");
    assert!(lcb > 0.0, "paired LCB95 must be positive: {lcb}");

    // (2) The matched blind sweep DOES reach a repair (it is a strong
    // baseline — the reviewer's point), but it costs more authoring evals
    // than the probe's parse-and-verify (the probe's actual contribution:
    // evals saved + the repair locus).
    assert!(
        blind_evals.get() > probe_evals.get(),
        "the parse-and-verify lane must cost fewer authoring evals than the enumeration: probe {} vs blind {}",
        probe_evals.get(),
        blind_evals.get()
    );
    assert_eq!(
        failure_rate(&degenerate, Some(&blind_ov), &in_region_starts()),
        0.0,
        "the blind-found word must be a real repair (the baseline is strong)"
    );
}
