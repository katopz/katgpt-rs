//! Issue-1248 T2/T3 — the R2 fault-localizer + the R3 gate-saturation alarm
//! with graceful early exit, on the `loop_depth_ood` fixture.
//!
//! R2 (readout-screen class — offline, never a serving path): per-layer
//! pass-shift SVD census with the typed fault verdict; the bar is
//! LOCALIZATION — layer 0 reads healthy, layer 1 names the fault transition.
//! R3: the in-loop OOD guard fires at d+1 and never at 1..d, and the
//! graceful exit keeps the healthy passes bit-identically (never worse than
//! refusing the depth).

#[path = "common/loop_depth_ood.rs"]
mod loop_depth_ood;

use loop_depth_ood::{
    r2_fault_localizer, run_with_gate_alarm, LoopDepthOod, BATCH, D_TRAIN, K_LADDER,
};

/// R2 — the fault SITE is the constructed layer (its consumer faults first);
/// the fault-free layer's own verdict, when it fires, is the downstream
/// contamination wave (strictly LATER than the site's transition).
#[test]
fn r2_localizes_fault_to_constructed_layer() {
    let fx = LoopDepthOod::build();
    let report = r2_fault_localizer(&fx);
    assert_eq!(report.layers.len(), 2);
    let l0 = &report.layers[0];
    let l1 = &report.layers[1];
    println!(
        "R2 layer 0 projections = {:?} faulted_at = {:?}",
        l0.v1_projections, l0.faulted_at
    );
    println!(
        "R2 layer 1 projections = {:?} faulted_at = {:?}",
        l1.v1_projections, l1.faulted_at
    );
    // The SITE verdict: the constructed consumer, at the fault transition.
    assert_eq!(
        report.first_fault,
        Some((1, D_TRAIN - 1)),
        "site at {:?}, expected (layer 1, transition {})",
        report.first_fault,
        D_TRAIN - 1
    );
    assert_eq!(l1.faulted_at, Some(D_TRAIN - 1));
    // The named transition carries the sign flip: healthy projections
    // opposite to the fault projection (v₁ oriented at the fault).
    let fault_proj = l1.v1_projections[D_TRAIN - 1];
    assert!(
        fault_proj > 0.0,
        "v₁ not oriented at the fault: {fault_proj}"
    );
    for p in &l1.v1_projections[..D_TRAIN - 1] {
        assert!(*p < 0.0, "no sign flip on layer 1: healthy proj {p}");
    }
    // Layer 0 (the fault-free side) reads healthy through the FIRST d
    // transitions; any layer-0 verdict is the WAVE — strictly later than the
    // site (F3: the fault is contained, not fixed — the wave is the story).
    match l0.faulted_at {
        None => {}
        Some(t) => {
            assert!(t >= D_TRAIN, "layer-0 wave at {t} precedes the fault");
            println!("R2 layer 0 wave at transition {t} (post-fault propagation)");
        }
    }
}

/// R3 — the alarm fires at d+1, never at 1..d; the graceful exit keeps the
/// healthy passes bit-identically.
#[test]
fn r3_alarm_fires_at_d_plus_one_and_exits_gracefully() {
    let fx = LoopDepthOod::build();

    // Armed run: fires exactly at the fault pass, keeps d healthy passes.
    let armed = run_with_gate_alarm(&fx, true);
    println!(
        "R3 armed: fired_at = {:?} passes_completed = {} gate_means = {:?}",
        armed.fired_at, armed.passes_completed, armed.gate_means
    );
    assert_eq!(
        armed.fired_at,
        Some(D_TRAIN),
        "alarm at {:?}, expected pass d+1",
        armed.fired_at
    );
    assert!(
        armed.gate_means[..D_TRAIN]
            .iter()
            .all(|g| *g >= loop_depth_ood::GATE_FLOOR),
        "alarm telemetry dipped in the healthy band: {:?}",
        &armed.gate_means[..D_TRAIN]
    );
    assert_eq!(armed.passes_completed, D_TRAIN);

    // Graceful ≥ refusing: the kept passes are the healthy ladder's own
    // outputs, bit-identically (the disarm only removes the refusal branch).
    let unarmed = run_with_gate_alarm(&fx, false);
    assert!(unarmed.fired_at.is_none());
    assert_eq!(unarmed.passes_completed, K_LADDER);
    for k in 0..D_TRAIN {
        for r in 0..BATCH {
            assert_eq!(
                armed.outputs[k][r],
                unarmed.outputs[k][r],
                "graceful-exit output diverged from the unarmed ladder at pass {} row {}",
                k + 1,
                r
            );
        }
    }
}
