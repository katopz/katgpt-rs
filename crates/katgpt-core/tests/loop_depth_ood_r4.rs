//! Issue-1248 T4 — the R4 spectral governor arm on the `loop_depth_ood`
//! fixture (the ladder's middle rung).
//!
//! Bars (pre-registered, issue §"Proof path"): the healthy passes are
//! byte-identical to the unarmed ladder (γ = 0 there by construction); the
//! γ curve reads the lagged gate telemetry (0 in-band, ramping after the
//! collapse); the post-fault effect is MEASURED and REPORTED — the
//! governor-vs-edit adjudication is T6's business on one frozen read.

#[path = "common/loop_depth_ood.rs"]
mod loop_depth_ood;

use loop_depth_ood::{
    cosine, governor_gamma, governor_run, r2_fault_localizer, run_with_gate_alarm, LoopDepthOod,
    BATCH, D, D_TRAIN, K_LADDER,
};

#[test]
fn r4_governor_arm_bars() {
    let fx = LoopDepthOod::build();

    // The frozen v₁: R2's census vector at the fault site (layer 1's
    // consumer — the pass-output signal), from the SITE transition.
    let report = r2_fault_localizer(&fx);
    let site = report.first_fault.expect("the fixture must be faulted");
    let (site_layer, site_t) = site;
    assert_eq!(site_layer, 1);
    // The census vector: recompute the shift-SVD on the site layer's states.
    let states = {
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
    let mut shifts = vec![vec![0.0f64; D]; K_LADDER - 1];
    for t in 0..K_LADDER - 1 {
        for j in 0..D {
            shifts[t][j] = states[t + 1][j] - states[t][j];
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
    // Orientation: v₁ points AT the fault shift (the site transition's
    // projection positive).
    let proj_site: f64 = shifts[site_t].iter().zip(&v1).map(|(a, b)| a * b).sum();
    if proj_site < 0.0 {
        for c in &mut v1 {
            *c = -*c;
        }
    }

    // The unarmed reference for the byte-identity bar.
    let unarmed = run_with_gate_alarm(&fx, false);
    assert!(unarmed.fired_at.is_none());

    // The γ curve: pure telemetry, no fault knowledge.
    assert_eq!(governor_gamma(None), 0.0);
    assert_eq!(
        governor_gamma(Some(0.70)),
        0.0,
        "healthy gate must read γ = 0"
    );
    assert_eq!(
        governor_gamma(Some(0.0)),
        1.0,
        "collapsed gate must read γ = 1"
    );

    // The armed run.
    let run = governor_run(&fx, &v1);
    println!("R4 γ per pass = {:?}", run.gammas);
    println!("R4 gate means = {:?}", run.gate_means);
    // γ = 0 for passes 1..=d+1 (the lag: pass d+1 reads pass d's healthy
    // telemetry), full projection once the collapse enters the lag window.
    for k in 0..=D_TRAIN {
        assert_eq!(run.gammas[k], 0.0, "γ armed too early at pass {}", k + 1);
    }
    assert!(
        run.gammas[K_LADDER - 1] > 0.99,
        "γ never reached full projection"
    );

    // Healthy passes byte-identical to the unarmed ladder: exactly the γ = 0
    // passes (indices 0..=D_TRAIN — pass d+1 reads pass d's healthy lagged
    // telemetry); from pass d+2 on the governor legitimately intervenes.
    for k in 0..=D_TRAIN {
        for r in 0..BATCH {
            assert_eq!(
                run.outputs[k][r],
                unarmed.outputs[k][r],
                "divergence at pass {} row {} (γ = {})",
                k + 1,
                r,
                run.gammas[k]
            );
        }
    }

    // The measured effect, REPORTED (T6 adjudicates): per-pass output
    // cosine vs the HEALTHY depth-d reference (pass d's output), and the
    // state-norm bounds.
    let mut ref_acc = vec![0.0f32; D];
    let mut x = fx.warm_start(0);
    for k in 1..=D_TRAIN {
        let (_g, y) = fx.pass(&x, k);
        x = y.clone();
    }
    // x now holds pass d's output for row 0 — the healthy reference
    // direction.
    let rn: f32 = x.iter().map(|c| c * c).sum::<f32>().sqrt();
    for (a, c) in ref_acc.iter_mut().zip(&x) {
        *a = c / rn;
    }
    for k in D_TRAIN..K_LADDER {
        let y = &run.outputs[0][k];
        let c = cosine(&ref_acc, y);
        let n: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!(
            "R4 pass {}: cos(output, healthy-ref) = {:.4}  ‖y‖ = {:.3}",
            k + 1,
            c,
            n
        );
        assert!(
            n.is_finite() && n > 0.1,
            "state norm collapsed at pass {}",
            k + 1
        );
    }
}
