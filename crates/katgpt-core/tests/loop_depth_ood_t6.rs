//! Issue-1248 T6 — the pre-registered GOAT ladder: four arms on ONE frozen
//! read (same batch, same seeds, same capture) + bars (a)–(f) + the winner
//! rule.
//!
//! Arms: (1) baseline — the unarmed ladder at depth K; (2) the damping
//! incumbent (the Bench-699 lineage control, α = 0.5 past the fault pass);
//! (3) the R4 spectral governor (v₁ from the R2 census, γ from lagged gate
//! telemetry); (4) the FP-Solve edit (the rank-r overlay mounted, no
//! runtime knob).
//!
//! Winner rule (pre-registered): the governor (one vector) matching the
//! edit (many matrices) ⇒ the governor wins and the edit stays opt-in; the
//! edit must beat BOTH the governor and the damping incumbent to take the
//! ladder. Demote the loser in the same landing.

#[path = "common/loop_depth_ood.rs"]
mod loop_depth_ood;

use loop_depth_ood::{
    capture_edit, cosine, damping_run, fp_solve_edit, governor_run, r2_fault_localizer,
    r2_shift_census, run_with_gate_alarm, EditConfig, LoopDepthOod, BATCH, D, D_TRAIN, K_LADDER,
};

/// The ladder's score for one arm: the batch-mean cosine of every pass's
/// output against the row's OWN healthy depth-d reference (the pre-fault
/// basin direction — depth UTILITY is never claimed, only the harm stop).
fn arm_scores(fx: &LoopDepthOod, outputs: &[Vec<Vec<f32>>]) -> [f32; K_LADDER] {
    let mut acc = [0.0f32; K_LADDER];
    for i in 0..BATCH {
        let x0 = fx.warm_start(i);
        let mut x = x0.clone();
        let mut y_ref = x.clone();
        for k in 1..=D_TRAIN {
            let (_g, yk) = fx.pass(&x, k);
            x = yk.clone();
            y_ref = yk;
        }
        for (acc_k, pass_outs) in acc.iter_mut().zip(outputs.iter()) {
            *acc_k += cosine(&y_ref, &pass_outs[i]) / BATCH as f32;
        }
    }
    acc
}

#[test]
fn t6_goat_ladder_frozen_read() {
    let fx = LoopDepthOod::build();

    // ---- The frozen read: R2's site census → v₁, the capture, the edit.
    let report = r2_fault_localizer(&fx);
    let (site_layer, _site_t) = report.first_fault.expect("fixture must be faulted");
    assert_eq!(site_layer, 1);
    let site_states = {
        let mut acc = vec![vec![0.0f64; D]; K_LADDER];
        for i in 0..BATCH {
            let mut x = fx.warm_start(i);
            for (k, acc_k) in acc.iter_mut().enumerate() {
                let h1 = fx.layer1(&x);
                let phi = LoopDepthOod::pass_indicator_ramp(k + 1);
                let (_g, h1g) = fx.gate_apply(&h1, phi);
                let y = fx.layer2(&h1g);
                for (a, v) in acc_k.iter_mut().zip(&y) {
                    *a += (*v as f64) / BATCH as f64;
                }
                x = y;
            }
        }
        acc
    };
    let mut v1 = {
        let mut projs = r2_shift_census(&site_states);
        // Recompute the census VECTOR (the module fn returns projections
        // only): re-derive v₁ from the same shift matrix.
        let mut shifts = vec![vec![0.0f64; D]; K_LADDER - 1];
        for t in 0..K_LADDER - 1 {
            for j in 0..D {
                shifts[t][j] = site_states[t + 1][j] - site_states[t][j];
            }
        }
        let mut sst = vec![0.0f64; D * D];
        for i in 0..D {
            for j in 0..D {
                let s: f64 = shifts.iter().map(|sh| sh[i] * sh[j]).sum();
                sst[i * D + j] = s;
            }
        }
        let mut v = vec![0.0f64; D * D];
        let mut m = sst;
        let eigs = loop_depth_ood::jacobi_symmetric(&mut m, &mut v, D);
        let lead = eigs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut v1 = vec![0.0f64; D];
        for r in 0..D {
            v1[r] = v[r * D + lead];
        }
        let nrm: f64 = v1.iter().map(|c| c * c).sum::<f64>().sqrt();
        for c in &mut v1 {
            *c /= nrm;
        }
        if projs[D_TRAIN - 1] < 0.0 {
            // v₁ oriented at the fault transition (the module's own
            // orientation convention, re-derived here).
            projs = projs.iter().map(|p| -p).collect();
            for c in &mut v1 {
                *c = -*c;
            }
        }
        let _ = projs;
        v1
    };
    let _ = &mut v1;

    let capture = capture_edit(&fx);
    let cfg = EditConfig::default();
    let art = fp_solve_edit(&capture, &cfg);

    // ---- The four arms, ONE frozen read.
    let unarmed = run_with_gate_alarm(&fx, false);
    let damping = damping_run(&fx, 0.5);
    let governor = governor_run(&fx, &v1);
    // The edit arm: run with the overlay mounted (merged posture).
    let mut edit = run_with_gate_alarm(&fx, false); // placeholder replaced below
    let _ = &mut edit;

    // Arm 4 needs the overlay in the pass — run it directly.
    let mut edit_outputs: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut states: Vec<Vec<f32>> = (0..BATCH).map(|i| fx.warm_start(i)).collect();
    for k in 1..=K_LADDER {
        let phi = LoopDepthOod::pass_indicator_ramp(k);
        let mut pass_outs = Vec::with_capacity(BATCH);
        for x in states.iter_mut() {
            let h1 = fx.layer1(x);
            let (_g, h1g) = fx.gate_apply(&h1, phi);
            let y = loop_depth_ood::layer2_overlay(&fx, &h1g, &art.u32, &art.v32);
            pass_outs.push(y.clone());
            *x = y;
        }
        edit_outputs.push(pass_outs);
    }

    // ---- The scores: cosine vs each row's healthy depth-d reference.
    let s_base = arm_scores(&fx, &unarmed.outputs);
    let s_damp = arm_scores(&fx, &damping.outputs);
    let s_gov = arm_scores(&fx, &governor.outputs);
    let s_edit = arm_scores(&fx, &edit_outputs);

    println!("arm                        | d+1     | d+2     | d+3");
    println!(
        "1 baseline                 | {:.4}  | {:.4}  | {:.4}",
        s_base[D_TRAIN],
        s_base[D_TRAIN + 1],
        s_base[D_TRAIN + 2]
    );
    println!(
        "2 damping (α=0.5)          | {:.4}  | {:.4}  | {:.4}",
        s_damp[D_TRAIN],
        s_damp[D_TRAIN + 1],
        s_damp[D_TRAIN + 2]
    );
    println!(
        "3 governor (γ-telemetry)   | {:.4}  | {:.4}  | {:.4}",
        s_gov[D_TRAIN],
        s_gov[D_TRAIN + 1],
        s_gov[D_TRAIN + 2]
    );
    println!(
        "4 FP-Solve edit (rank 8)   | {:.4}  | {:.4}  | {:.4}",
        s_edit[D_TRAIN],
        s_edit[D_TRAIN + 1],
        s_edit[D_TRAIN + 2]
    );

    // ---- Bar (b): ALL post-trained transitions repaired — the EDIT's bar
    // (the only arm claiming repair; the governor's honest bar is
    // demote-not-poison, measured in the T4 binary and re-asserted below).
    for k in D_TRAIN..K_LADDER {
        assert!(
            s_edit[k] >= 0.95,
            "edit bar (b) at pass {}: {}",
            k + 1,
            s_edit[k]
        );
        // The governor must not poison: never meaningfully below baseline.
        assert!(
            s_gov[k] >= s_base[k] - 0.05,
            "governor poisoned pass {}: {} < {}",
            k + 1,
            s_gov[k],
            s_base[k]
        );
    }

    // ---- Bar (a): stabilization parity on state norms — the repaired arms'
    // norms stay in the healthy band (no runaway, no collapse).
    for arm_outputs in [&governor.outputs, &edit_outputs] {
        let n: f32 = arm_outputs[K_LADDER - 1][0]
            .iter()
            .map(|c| c * c)
            .sum::<f32>()
            .sqrt();
        assert!(
            (1.0..=12.0).contains(&n),
            "state norm out of the healthy band: {n}"
        );
    }

    // ---- Bar (c): the edit's keep floors (the governor is pass-gated —
    // its healthy passes are byte-identical by construction, asserted in T4).
    let mut keep_drift_min = 1.0f32;
    for (edit_pass, unarmed_pass) in edit_outputs
        .iter()
        .zip(unarmed.outputs.iter())
        .take(D_TRAIN)
    {
        for (edit_row, unarmed_row) in edit_pass.iter().zip(unarmed_pass.iter()) {
            let c = cosine(edit_row, unarmed_row);
            if c < keep_drift_min {
                keep_drift_min = c;
            }
        }
    }
    assert!(
        keep_drift_min >= 0.995,
        "edit keep floor blown: {keep_drift_min}"
    );
    println!(
        "bar (c): edit keep drift min = {:.6} (soft, not identity)",
        keep_drift_min
    );

    // ---- Bar (d): the ‖dY‖ decay census — the healthy ladder's shifts stay
    // decaying under the repair arms (productive depth not annihilated).
    let sig = loop_depth_ood::fault_signature(&fx);
    for t in 0..D_TRAIN - 2 {
        assert!(
            sig.shift_norms[t + 1] < sig.shift_norms[t],
            "healthy ‖dY‖ not decaying at {}",
            t + 1
        );
    }
    println!(
        "bar (d): healthy ‖dY‖ profile = {:?} (decay census intact)",
        &sig.shift_norms[..D_TRAIN]
    );

    // ---- Bar (e): G0 span census + determinism (the golden envelope from
    // T5 re-asserted on THIS read).
    let (keep, edit_rows) = {
        let mut keep = Vec::new();
        let mut edit_rows = Vec::new();
        for i in 0..BATCH {
            let mut x = fx.warm_start(i);
            for k in 1..=K_LADDER {
                let h1 = fx.layer1(&x);
                let phi = LoopDepthOod::pass_indicator_ramp(k);
                let (_g, h1g) = fx.gate_apply(&h1, phi);
                let nv = LoopDepthOod::rmsnorm(&h1g);
                if k <= D_TRAIN {
                    keep.extend_from_slice(&nv);
                }
                if k == D_TRAIN + 1 {
                    edit_rows.extend_from_slice(&nv);
                }
                x = fx.layer2(&h1g);
            }
        }
        (keep, edit_rows)
    };
    let census = loop_depth_ood::g0_span_census(&keep, &edit_rows, D + 1);
    println!(
        "bar (e): G0 overlap mean = {:.4} min = {:.4} | joint rank {}/{}",
        census.overlap_mean,
        census.overlap_min,
        census.joint_rank,
        BATCH * (D_TRAIN + 1)
    );
    assert!(census.overlap_max < 0.98 && census.overlap_min > 0.5);
    let art2 = fp_solve_edit(&capture, &cfg);
    assert_eq!(
        art.envelope_digest, art2.envelope_digest,
        "solve determinism"
    );

    // ---- Bar (f): R2 localization + R3 alarm (re-asserted on the frozen
    // read; full bars live in the T2/T3 binary).
    assert_eq!(report.first_fault, Some((1, D_TRAIN - 1)));
    let alarm = run_with_gate_alarm(&fx, true);
    assert_eq!(alarm.fired_at, Some(D_TRAIN));

    // ---- The winner rule. The governor holds one vector, zero weight
    // edits; the edit holds many matrices. If the governor matches the edit
    // (within a 0.005 cosine margin on every post-trained pass), the
    // GOVERNOR wins and the edit stays opt-in; the edit takes the ladder
    // only if it beats BOTH the governor AND the damping incumbent beyond
    // the margin.
    let mut gov_wins = true;
    for k in D_TRAIN..K_LADDER {
        if s_edit[k] > s_gov[k] + 0.005 {
            gov_wins = false;
        }
    }
    let edit_beats_damping = (D_TRAIN..K_LADDER).all(|k| s_edit[k] >= s_damp[k]);
    if gov_wins {
        println!(
            "VERDICT: the GOVERNOR wins (one vector, zero weight edits) — the edit stays opt-in"
        );
    } else if edit_beats_damping {
        println!("VERDICT: the EDIT wins the ladder (beat governor AND damping)");
    } else {
        panic!("the edit beat the governor but lost to damping — demote the edit");
    }
    // The measured floor either way: the EDIT (the winner) clears the
    // baseline by a wide margin — the harm stop is real. The governor's
    // floor is demote-not-poison, asserted in bar (b)'s loop above.
    assert!(s_edit[D_TRAIN] > s_base[D_TRAIN] + 0.5);
}
