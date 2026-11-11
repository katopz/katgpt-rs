//! Issue 980 T1–T5 — the WeightEditOverlay gate battery (Research 648
//! modelless half).
//!
//! Fixture: `InjectedDegeneracyMlp` — a 128→64→64→4 tanh policy MLP over a
//! 2-D grid world with an INJECTED degenerate direction (seeded: the carrier
//! column of W2 gains `−k₀·d` along the vertical-decision channel, so a
//! two-action oscillation attractor forms at the pinned region boundary).
//! Plus a no-degeneracy twin (same seed, no injection). No LLM needed — the
//! POC proves the overlay + gate machinery.
//!
//! Pre-registered battery (the issue's T3, verbatim):
//! - (a) dose-response monotone non-increasing over pinned k ∈ {1,2,4,8}
//! - (b) neighbor-column isolation null (same edit at an uninvolved column ≈ no change)
//! - (c) norm-matched random control, ≥16 BLAKE3-seeded arms; paired
//!   LCB95 on (targeted − best random), best random picked ONCE on its mean
//! - (d) pre-screen refusal arm: no-degeneracy fixture ⇒ lane refuses
//! - (e) retention: clean-region behavior held (bit-identical outside the
//!   pinned region; k=k₀ repair restores the twin trajectory exactly)
//!
//! # Fixture analytic shape (pinned constants)
//!
//! W3 decision gaps: down−up = `1.6·h2[1]`, right−left = `0.5·(h2[2]−h2[0])`
//! (h2[0]=tanh(x/8), h2[2]=tanh(6/8)). The injection makes
//! `h2[1] = tanh(y/8 − 8·tanh(4·bump·(y/8+0.05)))`: inside the region the
//! sign flips (marches to the boundary), at the boundary `|y|=4↔5` the
//! region-gated carrier turns off/on each step and the policy oscillates
//! up/down forever with both gap magnitudes (≥0.887) exceeding the max
//! horizontal pull (0.67) — so it never advances on x and the episode fails.
//! The k₀=8 repair restores `W2[1, carrier] = 0` exactly → the twin's
//! center-then-march-right behavior, bit-identically. Base obstacle rates:
//! degenerate 1.0, twin 0.0.
//!
//! Green-zero rule: whole-file `#![cfg]` + `required-features` row pair.

#![cfg(feature = "gain_cost_halt")]

use katgpt_core::weight_edit_overlay::{
    ArmRefusal, ArmVerdict, EditForm, ObstacleArmConfig, ProvenanceTrailer, WeightEditArtifact,
    WeightEditOverlay, WeightEditSpec, arm_from_telemetry, norm_matched_random_direction,
    paired_lcb95,
};

// ─────────────────────────────────────────────────────────────────────
// The fixture (constants + world + rollouts) lives in `wsp980_common` —
// ONE definition shared with the Issue-993 LP-authoring gate (its
// "existing gate battery arms" claim reads the same world).
// ─────────────────────────────────────────────────────────────────────
#[path = "wsp980_common/mod.rs"]
mod wsp980_common;

use wsp980_common::*;

// ─────────────────────────────────────────────────────────────────────
// T1 — fixture + base obstacle rates + arming
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t1_base_obstacle_rates_and_arming() {
    let degenerate = build_fixture(true);
    let twin = build_fixture(false);
    let starts = in_region_starts();

    let (rate_deg, osc_deg) = obstacle_rate(&degenerate, None, &starts);
    let (rate_twin, osc_twin) = obstacle_rate(&twin, None, &starts);
    assert!(
        rate_deg >= 0.95,
        "degenerate fixture must oscillate: rate {rate_deg} ({osc_deg}/{})",
        starts.len()
    );
    assert!(
        rate_twin <= 0.02,
        "no-degeneracy twin must be obstacle-free: rate {rate_twin} ({osc_twin}/{})",
        starts.len()
    );

    // Ability axis: the twin REACHES the goal (center-then-march-right);
    // the degenerate model never advances (obstacle hides the ability).
    for s in &starts {
        assert!(
            rollout(&twin, None, *s).success,
            "twin must succeed at {s:?}"
        );
        assert!(
            !rollout(&degenerate, None, *s).success,
            "degenerate must fail at {s:?}"
        );
    }

    // Arming: the shipped obstacle telemetry CONSUMED — GainCostLoopHalter
    // per-episode oscillation rate arms the lane on the degenerate fixture
    // and refuses the twin (the falsifier).
    let episodes_deg: Vec<Vec<Vec<f32>>> = starts
        .iter()
        .map(|s| rollout(&degenerate, None, *s).hidden)
        .collect();
    let episodes_twin: Vec<Vec<Vec<f32>>> = starts
        .iter()
        .map(|s| rollout(&twin, None, *s).hidden)
        .collect();
    let cfg = ObstacleArmConfig::default();
    assert!(
        matches!(
            arm_from_telemetry(&episodes_deg, &cfg),
            ArmVerdict::Armed { .. }
        ),
        "degenerate telemetry must arm the lane"
    );
    assert!(
        matches!(
            arm_from_telemetry(&episodes_twin, &cfg),
            ArmVerdict::Refused(_)
        ),
        "twin telemetry must refuse (no obstacle found)"
    );
}

#[test]
fn t1_halter_telemetry_cross_agreement() {
    // The halter-based arming statistic must agree with the fixture's
    // reversal-streak statistic on every seeded episode — the two read the
    // same obstacle.
    let degenerate = build_fixture(true);
    let twin = build_fixture(false);
    let cfg = ObstacleArmConfig::default();
    for seed in 0..EVAL_SEEDS as u64 {
        let starts = seeded_starts(seed);
        let mut halter_osc = 0usize;
        let mut streak_osc = 0usize;
        let mut twin_halter_osc = 0usize;
        for s in &starts {
            let r = rollout(&degenerate, None, *s);
            if episode_oscillates(&r.actions) {
                streak_osc += 1;
            }
            if matches!(
                arm_from_telemetry(std::slice::from_ref(&r.hidden), &cfg),
                ArmVerdict::Armed { .. }
            ) {
                halter_osc += 1;
            }
            let rt = rollout(&twin, None, *s);
            if matches!(
                arm_from_telemetry(std::slice::from_ref(&rt.hidden), &cfg),
                ArmVerdict::Armed { .. }
            ) {
                twin_halter_osc += 1;
            }
        }
        assert_eq!(
            halter_osc, streak_osc,
            "seed {seed}: halter telemetry must match the reversal-streak statistic"
        );
        assert_eq!(twin_halter_osc, 0, "seed {seed}: twin must never oscillate");
    }
}

// ─────────────────────────────────────────────────────────────────────
// T2 — overlay carrier: compose / revert / dequant-domain arm
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t2_compose_untouched_bit_identical_and_revert_theta() {
    let base = build_fixture(true);
    let overlay = Fixture::targeted_overlay(4.0);
    let mut w2 = base.w2.clone();
    let before = f32_digest(&base.w2);

    let mut backup = vec![0.0f32; overlay.touched_count(LAYER_W2, H, H)];
    let used = overlay
        .compose_layer_into(LAYER_W2, &mut w2, H, H, &mut backup)
        .expect("backup sized");
    assert_eq!(used, H, "one column × 64 rows");

    // Untouched columns bit-identical; the touched column carries the delta.
    for i in 0..H {
        for j in 0..H {
            if j == CARRIER_COL as usize {
                continue;
            }
            assert_eq!(
                w2[i * H + j],
                base.w2[i * H + j],
                "column {j} must be bit-identical"
            );
        }
    }
    let expect = base.w2[DIR_CHANNEL * H + CARRIER_COL as usize] + 4.0; // −k·d[1] = −4·(−1) = +4
    assert_eq!(
        w2[DIR_CHANNEL * H + CARRIER_COL as usize],
        expect,
        "ΔW[:,S] = −k·d: (1, carrier) += −k·(−1) = +k"
    );

    // Determinism: a second compose on a fresh copy is bit-identical.
    let mut w2_again = base.w2.clone();
    let mut backup2 = vec![0.0f32; backup.len()];
    overlay
        .compose_layer_into(LAYER_W2, &mut w2_again, H, H, &mut backup2)
        .expect("second compose");
    assert_eq!(w2, w2_again, "compose must be deterministic");

    // Rejected-edit revert leaves θ bit-identical (BLAKE3 before/after).
    overlay.revert_layer(LAYER_W2, &mut w2, H, H, &backup);
    assert_eq!(
        f32_digest(&w2).to_hex(),
        before.to_hex(),
        "revert must restore θ bit-identically"
    );
}

#[test]
fn t2_dequant_domain_arm_packed_base_untouched() {
    let degenerate = build_fixture(true);
    // The QUANTIZED twin of the DEGENERATE base: the injection is baked in
    // before quantization, the overlay repairs it in the dequant domain.
    let twin_q = build_dequant_twin(&degenerate);
    let overlay = Fixture::targeted_overlay(INJECT_K0);

    // The PACKED base bytes are never touched by the overlay application:
    // capture at build, run in-region forwards WITH the overlay armed, then
    // re-hash — the bytes must be unchanged (QuantErrorLora posture).
    let bytes_before = i8_digest(&twin_q.q).to_hex();
    for s in &in_region_starts() {
        let inp = make_input(s.0, s.1);
        let _ = twin_q.forward(&inp, Some(&overlay));
    }
    assert_eq!(
        i8_digest(&twin_q.q).to_hex(),
        bytes_before,
        "packed base untouched (QuantErrorLora posture)"
    );

    // Out-of-region inputs: delta is exactly zero → bit-identical forward.
    for &(x, y) in &[(0.0f32, 6.0f32), (-5.0, -5.5), (3.0, 7.0)] {
        let inp = make_input(x, y);
        assert_eq!(
            twin_q.forward(&inp, None),
            twin_q.forward(&inp, Some(&overlay)),
            "out-of-region forward must be bit-identical at ({x},{y})"
        );
    }

    // In-region: the delta repairs the quantized twin (obstacle gone).
    let mut osc = 0usize;
    for s in &in_region_starts() {
        let mut actions = Vec::new();
        let (mut x, mut y) = *s;
        for _ in 0..MAX_STEPS {
            let inp = make_input(x, y);
            let logits = twin_q.forward(&inp, Some(&overlay));
            let a = argmax_first_index(&logits);
            match a {
                0 => y += 1.0,
                1 => y -= 1.0,
                2 => x -= 1.0,
                _ => x += 1.0,
            }
            actions.push(a);
            if x >= GOAL_X {
                break;
            }
        }
        if episode_oscillates(&actions) {
            osc += 1;
        }
    }
    let rate = osc as f32 / in_region_starts().len() as f32;
    assert!(
        rate <= 0.02,
        "dequant-domain delta must repair the quantized twin: rate {rate}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// T3 — the control battery
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t3a_dose_response_monotone() {
    let base = build_fixture(true);
    let starts = in_region_starts();

    // (a) pinned statistic: obstacle rate non-increasing over the grid.
    let mut prev_rate = f32::INFINITY;
    let mut rates = Vec::new();
    for k in DOSE_GRID {
        let overlay = Fixture::targeted_overlay(k);
        let (rate, _) = obstacle_rate(&base, Some(&overlay), &starts);
        rates.push(rate);
        assert!(
            rate <= prev_rate,
            "dose-response must be non-increasing: k={k} rate {rate} > prev {prev_rate}"
        );
        prev_rate = rate;
    }
    // The full dose restores the twin exactly → rate floor.
    assert!(
        *rates.last().unwrap() <= 0.02,
        "full-dose edit must clear the obstacle: {:?}",
        rates
    );

    // Continuous secondary probe: mean |h2[1]| at pinned probe points —
    // strictly decreasing (the vertical pull unwinds with the dose).
    let probes = [-1.0f32, 0.0, 1.0];
    let mut prev_pull = f32::INFINITY;
    for k in DOSE_GRID {
        let overlay = Fixture::targeted_overlay(k);
        let mut pull = 0.0f32;
        for py in probes {
            let inp = make_input(-3.0, py);
            let mut h2 = Vec::new();
            let _ = base.forward(&inp, Some(&overlay), None, Some(&mut h2));
            pull += h2[DIR_CHANNEL].abs();
        }
        pull /= probes.len() as f32;
        assert!(
            pull < prev_pull,
            "vertical pull must strictly decrease: k={k} pull {pull} ≥ prev {prev_pull}"
        );
        prev_pull = pull;
    }
}

#[test]
fn t3b_neighbor_column_isolation_null() {
    let base = build_fixture(true);
    let starts = in_region_starts();
    let (rate_base, _) = obstacle_rate(&base, None, &starts);

    // Same edit at an uninvolved column set (the always-on constant channel
    // with tiny mass): the oscillation must persist ≈ unchanged.
    let null_overlay = WeightEditOverlay::new(vec![WeightEditSpec {
        layer: LAYER_W2,
        form: EditForm::DirectionSubtraction {
            columns: vec![ISOLATION_COL],
            direction: Fixture::targeted_direction(),
            strength: INJECT_K0,
        },
    }])
    .expect("null overlay valid");
    let (rate_null, _) = obstacle_rate(&base, Some(&null_overlay), &starts);
    assert!(
        (rate_null - rate_base).abs() <= 0.05,
        "isolation null: base {rate_base} vs null {rate_null} must be ≈ unchanged"
    );
}

#[test]
fn t3c_norm_matched_random_control_battery() {
    let base = build_fixture(true);
    let eval_seed_starts: Vec<Vec<(f32, f32)>> =
        (0..EVAL_SEEDS as u64).map(seeded_starts).collect();

    // Targeted arm: per-seed rates.
    let targeted = Fixture::targeted_overlay(INJECT_K0);
    let targeted_rates: Vec<f32> = eval_seed_starts
        .iter()
        .map(|starts| obstacle_rate(&base, Some(&targeted), starts).0)
        .collect();
    let targeted_mean: f32 = targeted_rates.iter().sum::<f32>() / EVAL_SEEDS as f32;

    // ≥16 norm-matched BLAKE3-seeded random arms (RandOpt posture).
    let mut arm_rates = Vec::new();
    for m in 0..RANDOM_ARMS {
        let seed = katgpt_core::blake3_seed(b"issue980/control", m as u64);
        let mut dir = vec![0.0f32; H];
        norm_matched_random_direction(seed, H, &mut dir);
        let overlay = WeightEditOverlay::new(vec![WeightEditSpec {
            layer: LAYER_W2,
            form: EditForm::DirectionSubtraction {
                columns: vec![CARRIER_COL],
                direction: dir,
                strength: INJECT_K0,
            },
        }])
        .expect("random overlay valid");
        let rates: Vec<f32> = eval_seed_starts
            .iter()
            .map(|starts| obstacle_rate(&base, Some(&overlay), starts).0)
            .collect();
        arm_rates.push(rates);
    }

    // The best random arm is selected ONCE on its mean over the eval
    // protocol (reading (i) — not per-seed max-over-arms).
    let mut best_arm = 0usize;
    let mut best_mean = f32::INFINITY;
    for (m, rates) in arm_rates.iter().enumerate() {
        let mean = rates.iter().sum::<f32>() / EVAL_SEEDS as f32;
        if mean < best_mean {
            best_mean = mean;
            best_arm = m;
        }
    }

    // Beating only the ensemble MEAN fails; the paired LCB95 against the
    // once-picked best arm is the statistic.
    let deltas: Vec<f32> = arm_rates[best_arm]
        .iter()
        .zip(targeted_rates.iter())
        .map(|(r, t)| r - t)
        .collect();
    let lcb = paired_lcb95(&deltas).expect("16 paired deltas");
    assert!(
        targeted_mean < best_mean,
        "targeted mean {targeted_mean} must beat the best random mean {best_mean}"
    );
    assert!(
        lcb > 0.0,
        "paired LCB95 (targeted − best random) must be > 0: lcb {lcb}, deltas {deltas:?}"
    );
}

#[test]
fn t3d_obstacle_prescreen_refusal() {
    // The no-degeneracy twin's telemetry REFUSES the lane: zero fake gain.
    let twin = build_fixture(false);
    let episodes: Vec<Vec<Vec<f32>>> = in_region_starts()
        .iter()
        .map(|s| rollout(&twin, None, *s).hidden)
        .collect();
    let verdict = arm_from_telemetry(&episodes, &ObstacleArmConfig::default());
    let ArmVerdict::Refused(ArmRefusal::NoObstacleFound { oscillation_rate }) = &verdict else {
        panic!("twin must refuse, got {verdict:?}");
    };
    assert!(
        *oscillation_rate <= 0.02,
        "refusal rate must be the twin's obstacle-free rate: {oscillation_rate}"
    );
}

#[test]
fn t3e_retention_outside_region_bit_identical() {
    let base = build_fixture(true);
    let twin = build_fixture(false);
    let repair = Fixture::targeted_overlay(INJECT_K0);

    // (i) Forward-level: out-of-region inputs see the edit contribute
    // exactly zero → every stage bit-identical.
    for &(x, y) in &[(0.0f32, 6.0f32), (-7.0, -5.0), (2.0, 7.0), (-1.0, -6.0)] {
        let inp = make_input(x, y);
        let mut h2_base = Vec::new();
        let mut h2_edit = Vec::new();
        let l_base = base.forward(&inp, None, None, Some(&mut h2_base));
        let l_edit = base.forward(&inp, Some(&repair), None, Some(&mut h2_edit));
        assert_eq!(h2_base, h2_edit, "h2 bit-identical at ({x},{y})");
        assert_eq!(l_base, l_edit, "logits bit-identical at ({x},{y})");
    }

    // (ii) Rollout-prefix agreement on clean starts: actions identical
    // while outside the region (the centering drift walks into the region —
    // from that step on the repair legitimately diverges from the obstacle).
    for s in &clean_starts() {
        let rb = rollout(&base, None, *s);
        let re = rollout(&base, Some(&repair), *s);
        let mut y = s.1;
        for step in 0..rb.actions.len().min(re.actions.len()) {
            if y.abs() > REGION_HALF {
                assert_eq!(
                    rb.actions[step], re.actions[step],
                    "prefix action mismatch at {s:?} step {step}"
                );
            } else {
                break;
            }
            if rb.actions[step] == 0 {
                y += 1.0;
            } else if rb.actions[step] == 1 {
                y -= 1.0;
            }
        }
    }

    // (iii) The strongest retention form: the full-dose repair restores the
    // twin's trajectory EXACTLY (whole action sequence) on every in-region
    // start — the repair un-hides the twin's ability, nothing else changes.
    for s in &in_region_starts() {
        let r_repair = rollout(&base, Some(&repair), *s);
        let r_twin = rollout(&twin, None, *s);
        assert_eq!(
            r_repair.actions, r_twin.actions,
            "repaired trajectory must equal the twin exactly at {s:?}"
        );
        assert!(r_repair.success, "repaired model must succeed at {s:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// T4 — G2/G3/G4 posture arms
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t4_flag_off_byte_identical_boot() {
    // Flag-off (no overlay installed) is byte-identical to the empty
    // overlay at the API level: rollouts and forwards identical.
    let base = build_fixture(true);
    let empty = WeightEditOverlay::new(Vec::new()).expect("empty overlay");
    for s in in_region_starts().iter().take(6) {
        let r_none = rollout(&base, None, *s);
        let r_empty = rollout(&base, Some(&empty), *s);
        assert_eq!(r_none.actions, r_empty.actions);
        assert_eq!(r_none.success, r_empty.success);
        let inp = make_input(s.0, s.1);
        let mut h2_none = Vec::new();
        let mut h2_empty = Vec::new();
        let l0 = base.forward(&inp, None, None, Some(&mut h2_none));
        let l1 = base.forward(&inp, Some(&empty), None, Some(&mut h2_empty));
        assert_eq!(l0, l1);
        assert_eq!(h2_none, h2_empty);
    }
}

#[test]
fn t4_alloc_free_apply() {
    // Build everything BEFORE the measured window — the window covers ONLY
    // the apply path (compose + forward-delta + bypass-forward + revert).
    let base = build_fixture(true);
    let overlay = Fixture::targeted_overlay(4.0);
    let mut w2 = base.w2.clone();
    let mut backup = vec![0.0f32; overlay.touched_count(LAYER_W2, H, H)];
    let h1 = vec![0.5f32; H];
    let mut y = vec![0.0f32; H];
    // Warmup (dispatcher/lazy state), then the measured window.
    overlay
        .compose_layer_into(LAYER_W2, &mut w2, H, H, &mut backup)
        .unwrap();
    overlay.forward_delta_into(LAYER_W2, &h1, &mut y);
    overlay.bypass_forward_into(LAYER_W2, &h1, &mut y);
    overlay.revert_layer(LAYER_W2, &mut w2, H, H, &backup);
    assert_counter_is_live();
    let (_, allocs) = alloc_delta(|| {
        overlay
            .compose_layer_into(LAYER_W2, &mut w2, H, H, &mut backup)
            .unwrap();
        overlay.forward_delta_into(LAYER_W2, &h1, &mut y);
        overlay.bypass_forward_into(LAYER_W2, &h1, &mut y);
        overlay.revert_layer(LAYER_W2, &mut w2, H, H, &backup);
    });
    assert_eq!(
        allocs, 0,
        "apply path must be alloc-free, got {allocs} allocs"
    );
}

// ─────────────────────────────────────────────────────────────────────
// T5 — mint-once freeze envelope (integration: bytes → thaw → verify)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t5_envelope_mint_load_tamper_refusal() {
    let base = build_fixture(true);
    let overlay = Fixture::targeted_overlay(INJECT_K0);

    let artifact = WeightEditArtifact::mint(
        &overlay,
        ProvenanceTrailer {
            prompt_digest: Some(*blake3::hash(b"issue980 authoring prompt").as_bytes()),
            seed: Some(7),
            author_id: Some(*blake3::hash(b"issue980 author").as_bytes()),
        },
    );
    assert!(artifact.verify_header());
    assert!(artifact.verify_overlay(&overlay));

    // Load path: envelope bytes + payload bytes → thaw → verify → apply.
    let env_bytes = artifact.to_bytes();
    let payload_bytes = overlay.to_bytes();
    let thawed = WeightEditArtifact::thaw(&env_bytes).expect("thaw");
    assert_eq!(thawed, artifact);
    let payload = WeightEditOverlay::from_bytes(&payload_bytes).expect("payload decode");
    assert!(
        thawed.verify_overlay(&payload),
        "payload must verify at load"
    );

    // Tamper the ENVELOPE → thaw refuses loud.
    let mut tampered_env = env_bytes.clone();
    tampered_env[20] ^= 0x80;
    assert!(
        WeightEditArtifact::thaw(&tampered_env).is_none(),
        "tampered envelope must refuse"
    );
    // Tamper the PAYLOAD → overlay verification refuses.
    let mut tampered_payload = payload_bytes.clone();
    let last = tampered_payload.len() - 1;
    tampered_payload[last] ^= 0x01;
    let decoded = WeightEditOverlay::from_bytes(&tampered_payload).expect("still decodes");
    assert!(
        !thawed.verify_overlay(&decoded),
        "tampered payload must fail artifact verification"
    );

    // The thawed artifact's provenance round-trips.
    assert_eq!(thawed.provenance().seed, Some(7));
    assert!(thawed.provenance().prompt_digest.is_some());

    // Sanity: the repaired model the envelope pins is the GOAT arm.
    let mut w2 = base.w2.clone();
    let mut backup = vec![0.0f32; overlay.touched_count(LAYER_W2, H, H)];
    overlay
        .compose_layer_into(LAYER_W2, &mut w2, H, H, &mut backup)
        .unwrap();
    let repaired = Fixture {
        w1: base.w1.clone(),
        w2,
        w3: base.w3.clone(),
    };
    for s in &in_region_starts() {
        assert!(
            rollout(&repaired, None, *s).success,
            "repaired must succeed at {s:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// CountingAllocator glue (G4) — tests/common convention. The macro emits
// ALLOC_COUNT/DEALLOC_COUNT + alloc_delta + the Issue-714 liveness canary
// at this file's crate root.
// ─────────────────────────────────────────────────────────────────────

#[path = "common/mod.rs"]
mod common;
counting_allocator!();
