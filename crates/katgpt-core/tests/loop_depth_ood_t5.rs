//! Issue-1248 T5 — the FP-Solve edit arm on the `loop_depth_ood` fixture:
//! capture on the serving posture → protected-ridge solve → rank-r overlay
//! → mount → the repair measured at d+1 and the extrapolation bar at
//! d+2/d+3, the healthy-pass keep floors, and the Issue-940 determinism
//! golden.
//!
//! The loop-as-label law: the ONLY labels are the loop's own pass-d outputs
//! (dY = y_d − y_{d+1} per row); no facts, no teacher, no gradient.

#[path = "common/loop_depth_ood.rs"]
mod loop_depth_ood;

use loop_depth_ood::{
    capture_edit, cosine, fp_solve_edit, layer2_overlay, EditConfig, EditCapture, LoopDepthOod,
    BATCH, D_TRAIN, K_LADDER,
};

/// Run the ladder for one row with the overlay mounted (passes 1..d bare —
/// the overlay is the pass-agnostic weight edit, the keep term protects the
/// seen span; every pass reads the SAME mounted matrix — the merged
/// posture's semantics).
fn ladder_overlay(
    fx: &LoopDepthOod,
    art: &loop_depth_ood::EditArtifact,
    x0: &[f32],
) -> Vec<Vec<f32>> {
    let mut outs = Vec::with_capacity(K_LADDER);
    let mut x = x0.to_vec();
    for k in 1..=K_LADDER {
        let h1 = fx.layer1(&x);
        let phi = LoopDepthOod::pass_indicator_ramp(k);
        let (_g, h1g) = fx.gate_apply(&h1, phi);
        let y = layer2_overlay(fx, &h1g, &art.u32, &art.v32);
        x = y.clone();
        outs.push(y);
    }
    outs
}

/// The per-row healthy reference (pass-d output) and the unarmed ladder.
fn healthy_reference(fx: &LoopDepthOod, x0: &[f32]) -> Vec<f32> {
    let mut x = x0.to_vec();
    let mut y = x.clone();
    for k in 1..=D_TRAIN {
        let (_g, yk) = fx.pass(&x, k);
        x = yk.clone();
        y = yk;
    }
    y
}

#[test]
fn t5_edit_repairs_and_extrapolates() {
    let fx = LoopDepthOod::build();

    // The capture + solve.
    let capture = capture_edit(&fx);
    let cfg = EditConfig::default();
    let art = fp_solve_edit(&capture, &cfg);
    println!(
        "T5 solve: kept energy {:.4} dropped {:.4} (ratio {:.5}) envelope {}",
        art.kept_energy,
        art.dropped_energy,
        art.dropped_energy / art.kept_energy.max(1e-30),
        art.envelope_digest.to_hex()
    );

    // The repair + extrapolation, per row; the healthy keep-drift too.
    let mut fault_before = 0.0f32;
    let mut fault_after = 0.0f32;
    let mut ext = [0.0f32; K_LADDER - D_TRAIN - 1]; // d+2 .. d+3 recovery vs reference
    let mut keep_drift_min = 1.0f32;
    for i in 0..BATCH {
        let x0 = fx.warm_start(i);
        // Unarmed ladder.
        let (_g, outs) = fx.ladder(&x0);
        // The fault depth BEFORE the edit (pass d+1 vs the row's reference).
        let y_ref = healthy_reference(&fx, &x0);
        fault_before += cosine(&outs[D_TRAIN], &y_ref) / BATCH as f32;
        // Armed ladder.
        let outs_e = ladder_overlay(&fx, &art, &x0);
        fault_after += cosine(&outs_e[D_TRAIN], &y_ref) / BATCH as f32;
        for (t, slot) in ext.iter_mut().enumerate() {
            let k = D_TRAIN + 1 + t; // passes d+2 .. d+3
            *slot += cosine(&outs_e[k], &y_ref) / BATCH as f32;
        }
        // Keep floors: the edited healthy passes vs the unarmed ones.
        for k in 0..D_TRAIN {
            let c = cosine(&outs_e[k], &outs[k]);
            if c < keep_drift_min {
                keep_drift_min = c;
            }
        }
    }
    println!(
        "T5 repair: pass d+1 cos(vs reference) BEFORE = {:.4} AFTER = {:.4}",
        fault_before, fault_after
    );
    println!("T5 extrapolation: d+2 = {:.4} d+3 = {:.4}", ext[0], ext[1]);
    println!(
        "T5 keep drift: min cosine over healthy passes = {:.6}",
        keep_drift_min
    );

    // Bars. The repair: the fault pass recovers to the healthy reference.
    assert!(
        fault_after >= 0.95,
        "edit failed to repair d+1: {fault_after}"
    );
    // Bar (b) — the F10 extrapolation bar: the ONE overlay fit at d+1 must
    // hold at d+2/d+3. (The T6 ladder re-measures this on the frozen read.)
    assert!(ext[0] >= 0.90, "extrapolation d+2 failed: {}", ext[0]);
    assert!(ext[1] >= 0.90, "extrapolation d+3 failed: {}", ext[1]);
    // Bar (c) — soft keep: the healthy passes drift within floors (mixed
    // signs allowed, never identity — a 1.0 reading here would mean the
    // edit did nothing anywhere).
    assert!(
        keep_drift_min >= 0.995,
        "keep floor blown: {keep_drift_min}"
    );
    assert!(
        keep_drift_min < 1.0,
        "keep read as identity — the edit is inert"
    );

    // Issue-940 determinism: the same capture + solve is bit-identical.
    let capture2 = capture_edit(&fx);
    let art2 = fp_solve_edit(&capture2, &cfg);
    assert_eq!(capture.capture_digest, capture2.capture_digest);
    assert_eq!(art.envelope_digest, art2.envelope_digest);
    assert_eq!(art.u32, art2.u32);
    assert_eq!(art.v32, art2.v32);
}

/// The overlay law: the EMPTY overlay demotes byte-identically (the unarmed
/// layer-2 path untouched), and the envelope binds capture + config + the
/// artifact bytes (a config nudge moves the digest).
#[test]
fn t5_overlay_discipline() {
    let fx = LoopDepthOod::build();
    let capture: EditCapture = capture_edit(&fx);
    let cfg = EditConfig::default();
    let art = fp_solve_edit(&capture, &cfg);

    // The unarmed path and an explicitly EMPTY overlay (r = 0 rows) read
    // byte-identically.
    let x0 = fx.warm_start(3);
    let h1 = fx.layer1(&x0);
    let (_g, h1g) = fx.gate_apply(&h1, LoopDepthOod::pass_indicator_ramp(4));
    let bare = fx.layer2(&h1g);
    let empty = layer2_overlay(&fx, &h1g, &[], &[]);
    assert_eq!(bare, empty, "empty overlay must demote byte-identically");

    // The envelope binds the config: a different ridge moves the digest.
    let cfg_hi = EditConfig {
        ridge_frac: 0.08,
        ..EditConfig::default()
    };
    let art_hi = fp_solve_edit(&capture, &cfg_hi);
    assert_ne!(art.envelope_digest, art_hi.envelope_digest);
    // ...and the same (capture, config) pair reproduces the same digest.
    let art_again = fp_solve_edit(&capture, &cfg);
    assert_eq!(art.envelope_digest, art_again.envelope_digest);
}
