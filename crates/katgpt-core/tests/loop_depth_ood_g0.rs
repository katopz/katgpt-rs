//! Issue-1248 T1 — the `loop_depth_ood` fixture's G0 span census + the
//! fault-signature self-check (arm-1 baseline, standing).
//!
//! Pre-registered bars (`.issues/1248_closed_form_loop_depth_extension.md`
//! §"T1 fixture design"): healthy pass-cos ≥ 0.95; fault pass-cos ≤ 0.45;
//! gate means in [0.5, 0.75] in-band and ≤ 0.15 at d+1; shift-SVD v₁
//! dominance ≥ 10× with the SIGN FLIP (healthy projections negative, fault
//! positive — the Inkling trajectory, not a spike); layer-1 localization
//! (cosines ≥ 0.95 at every depth); G0 overlap strictly short of 1 (reported
//! with every result — verdict round-2 pin 1).

#[path = "common/loop_depth_ood.rs"]
mod loop_depth_ood;

use loop_depth_ood::{
    fault_signature, g0_span_census, FaultSignature, LoopDepthOod, BATCH, D, D_GATE, D_TRAIN,
    K_LADDER,
};

/// Build the capture matrices the G0 census reads (the T5 solve reuses the
/// same rows): keep = layer-2 inputs (the gated states) from passes 1..d,
/// edit = the pass-(d+1) gated states, per batch row.
fn capture_layer2_inputs(fx: &LoopDepthOod) -> (Vec<f32>, Vec<f32>) {
    let mut keep = Vec::with_capacity(BATCH * D_TRAIN * D_GATE);
    let mut edit = Vec::with_capacity(BATCH * D_GATE);
    for i in 0..BATCH {
        let x0 = fx.warm_start(i);
        let mut x = x0.clone();
        for k in 1..=K_LADDER {
            let h1 = fx.layer1(&x);
            let phi = LoopDepthOod::pass_indicator_ramp(k);
            let (_g, h1g) = fx.gate_apply(&h1, phi);
            if k <= D_TRAIN {
                keep.extend_from_slice(&h1g);
            }
            if k == D_TRAIN + 1 {
                edit.extend_from_slice(&h1g);
            }
            x = fx.layer2(&h1g);
        }
    }
    (keep, edit)
}

/// G0 — the span census. Asserts the edit rows are NOT inside span(keep)
/// with a real margin, and reports every number (the overlap mean/min ride
/// the test output beside every result).
#[test]
fn g0_edit_rows_not_inside_keep_span() {
    let fx = LoopDepthOod::build();
    let (keep, edit) = capture_layer2_inputs(&fx);
    let census = g0_span_census(&keep, &edit, D_GATE);

    // The pre-registered margin: overlap strictly short of 1 (the edit has
    // out-of-span room to write into) and NOT orthogonal (the Inkling regime:
    // a seen direction whose influence flips — not empty space). Both rigged
    // poles fail here.
    println!(
        "G0 overlap: mean = {:.4} min = {:.4} | keep rank {}/{} (min nz eig {:.3e}) | joint rank {}/{} (min nz eig {:.3e})",
        census.overlap_mean,
        census.overlap_min,
        census.keep_rank,
        BATCH * D_TRAIN,
        census.keep_min_nonzero_eig,
        census.joint_rank,
        BATCH * D_TRAIN + BATCH,
        census.joint_min_nonzero_eig,
    );
    for o in &census.overlaps {
        println!("G0 overlap row {}: {:.4}", o.row, o.overlap);
    }
    assert!(
        census.overlap_max < 0.98,
        "edit rows (nearly) inside keep span — solve-dead fixture: max = {}",
        census.overlap_max
    );
    assert!(
        census.overlap_min > 0.5,
        "edit rows nearly orthogonal to keep span — the one-hot failure: min = {}",
        census.overlap_min
    );
}

/// Arm-1 baseline — the fault signature must reproduce the Inkling shape.
#[test]
fn fault_signature_reproduces_inkling_shape() {
    let fx = LoopDepthOod::build();
    let sig = fault_signature(&fx);

    // Gate telemetry: healthy in-band, collapsed at d+1.
    println!(
        "fault signature: gate means = {:?} | shift norms = {:?} | fault ‖dY‖ = {:.3}",
        sig.gate_means, sig.shift_norms, sig.fault_shift_norm
    );
    for k in 0..D_TRAIN {
        assert!(
            (0.5..=0.75).contains(&sig.gate_means[k]),
            "gate mean pass {} = {} outside the healthy band",
            k + 1,
            sig.gate_means[k]
        );
    }
    assert!(
        sig.gate_means[D_TRAIN] <= 0.15,
        "gate did NOT collapse at d+1: {}",
        sig.gate_means[D_TRAIN]
    );

    // Pass-cosines: healthy transitions ≥ 0.95; the fault transition ≤ 0.45
    // (the 0.17-class with tuning room); the fault is LOCALIZED — layer 1
    // stays healthy at every depth (R2's premise).
    for t in 0..D_TRAIN - 1 {
        assert!(
            sig.cosines.layer2[t] >= 0.95,
            "healthy transition {}→{} cos = {} below the bar",
            t + 1,
            t + 2,
            sig.cosines.layer2[t]
        );
    }
    assert!(
        sig.cosines.layer2[D_TRAIN - 1] <= 0.45,
        "fault transition d→d+1 did NOT collapse: cos = {}",
        sig.cosines.layer2[D_TRAIN - 1]
    );
    for t in 0..K_LADDER - 1 {
        assert!(
            sig.cosines.layer1[t] >= 0.95,
            "layer-1 localization FAILED at transition {}→{}: cos = {}",
            t + 1,
            t + 2,
            sig.cosines.layer1[t]
        );
    }

    // ‖dY‖ decay census (F8): strictly decreasing across the HEALTHY passes
    // (indices 0..D_TRAIN-1 = transitions 1→2, 2→3; the fault transition is
    // index D_TRAIN-1 and is excluded — the repair must not flatten this
    // profile, the census instrument).
    for t in 0..D_TRAIN - 2 {
        assert!(
            sig.shift_norms[t + 1] < sig.shift_norms[t],
            "healthy ‖dY‖ not decaying at {}: {} → {}",
            t + 1,
            sig.shift_norms[t],
            sig.shift_norms[t + 1]
        );
    }

    // Shift-SVD: the fault transition dominates v₁ with a SIGN FLIP against
    // the healthy projections (the Inkling [−,−,+] trajectory, not a spike).
    check_shift_svd(&fx, &sig);
}

/// The shift-SVD census — leading left-singular direction of the pass-shift
/// matrix (batch-meaned shifts), per-pass projections, the dominance and
/// sign-flip bars.
fn check_shift_svd(fx: &LoopDepthOod, _sig: &FaultSignature) {
    // Batch-mean shift vectors per transition.
    let mut shifts = vec![vec![0.0f64; D]; K_LADDER - 1];
    for i in 0..BATCH {
        let x0 = fx.warm_start(i);
        let (_gates, outs) = fx.ladder(&x0);
        for t in 0..K_LADDER - 1 {
            for j in 0..D {
                shifts[t][j] += ((outs[t + 1][j] - outs[t][j]) as f64) / BATCH as f64;
            }
        }
    }
    // S = [s_1 .. s_K-1] as columns; S Sᵀ is D×D — compute v₁ as the leading
    // eigenvector of S Sᵀ (Jacobi, f64).
    let mut sst = vec![0.0f64; D * D];
    for i in 0..D {
        for j in 0..D {
            let acc: f64 = shifts.iter().map(|s| s[i] * s[j]).sum();
            sst[i * D + j] = acc;
        }
    }
    let mut v = vec![0.0f64; D * D];
    let mut m = sst.clone();
    let eigs = loop_depth_ood::jacobi_symmetric(&mut m, &mut v, D);
    // Leading eigenvector = the column with the largest eigenvalue (the
    // diagonal is column-aligned by contract).
    let lead_col = eigs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut v1 = vec![0.0f64; D];
    for r in 0..D {
        v1[r] = v[r * D + lead_col];
    }
    let v1n: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v1 {
        *x /= v1n;
    }
    // Sign convention: orient v₁ so the FAULT projection is positive — the
    // invariant under test is the OPPOSITION (healthy vs fault), independent
    // of the eigenvector's arbitrary global sign.
    let fault_raw: f64 = shifts[D_TRAIN - 1]
        .iter()
        .zip(&v1)
        .map(|(a, b)| a * b)
        .sum();
    if fault_raw < 0.0 {
        for x in &mut v1 {
            *x = -*x;
        }
    }

    // Per-pass projections onto v₁.
    let mut projs = vec![0.0f64; K_LADDER - 1];
    for t in 0..K_LADDER - 1 {
        projs[t] = shifts[t].iter().zip(&v1).map(|(a, b)| a * b).sum();
    }
    let fault = projs[D_TRAIN - 1];
    let healthy_med = {
        let mut hs: Vec<f64> = projs[..D_TRAIN - 1].iter().map(|p| p.abs()).collect();
        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        hs[hs.len() / 2]
    };
    println!(
        "shift-SVD: v₁ projections = {:?} | fault = {:.3} vs healthy |med| = {:.3}",
        projs, fault, healthy_med
    );
    assert!(
        fault.abs() >= 10.0 * healthy_med,
        "fault shift does not dominate v₁: |fault| = {fault} vs med {healthy_med}"
    );
    // The sign flip: healthy projections NEGATIVE on v₁, fault POSITIVE (v₁
    // oriented at the fault above, so the invariant is exact).
    for p in &projs[..D_TRAIN - 1] {
        assert!(
            *p < 0.0,
            "no sign flip at v₁: healthy proj {p} vs fault {fault}"
        );
    }
    assert!(fault > 0.0, "fault projection not positive: {fault}");
}
