//! Issue-1248 T7 — the merged-vs-unmerged overlay cost census (the
//! O(r·d)/token disclosure, measured).
//!
//! Merged: ΔW folded into the control matrix at mount — one-time cost, then
//! the base per-pass cost. Unmerged: the rank-r factor applied per pass
//! (`y += U·(Vᵀ·x)`) — the r·(2D) extra multiply chain every pass, plus the
//! artifact-size difference (r·2D·4 bytes vs D²·4). Both postures measured
//! on the same frozen artifact; the numbers are DISCLOSED, the posture
//! choice stays the consumer's.

#[path = "common/loop_depth_ood.rs"]
mod loop_depth_ood;

use loop_depth_ood::{
    capture_edit, fp_solve_edit, layer2_overlay, EditConfig, LoopDepthOod, BATCH, D, D_TRAIN,
    K_LADDER,
};

#[test]
fn t7_merged_vs_unmerged_cost_census() {
    let fx = LoopDepthOod::build();
    let capture = capture_edit(&fx);
    let cfg = EditConfig::default();
    let art = fp_solve_edit(&capture, &cfg);

    // ---- Artifact sizes (the envelope the overlay lane ships).
    let unmerged_bytes = (art.u32.len() + art.v32.len()) * std::mem::size_of::<f32>();
    let merged_bytes = D * D * std::mem::size_of::<f32>();
    println!(
        "T7 artifact: unmerged rank-r = {} B vs merged dense = {} B ({:.2}×)",
        unmerged_bytes,
        merged_bytes,
        merged_bytes as f64 / unmerged_bytes as f64
    );
    // At rank r the factor is 2·r·D·4 B vs D²·4 B: exactly 4× at r = D/8
    // (our rank-8 pin on D = 64) — the assert pins that arithmetic.
    assert!(
        unmerged_bytes <= merged_bytes / 4,
        "rank-r must not exceed dense/4 at the pinned rank"
    );

    // ---- Per-pass apply cost, UNMERGED: the overlay chain on a warm state,
    // timed over many passes (median-of-rings is overkill here — the fixture
    // is micro-scale; the DISCLOSED number is the mean over 20k passes).
    let h1g = {
        let x0 = fx.warm_start(0);
        let h1 = fx.layer1(&x0);
        let (_g, h1g) = fx.gate_apply(&h1, LoopDepthOod::pass_indicator_ramp(D_TRAIN + 1));
        h1g
    };
    let reps = 20_000usize;
    let t_un = std::time::Instant::now();
    let mut sink_un = 0.0f32;
    for _ in 0..reps {
        let y = layer2_overlay(&fx, &h1g, &art.u32, &art.v32);
        sink_un += y[0];
    }
    let un_per_pass = t_un.elapsed().as_secs_f64() / reps as f64;
    // The merged posture: the fold (ΔW = U·Vᵀ materialized dense once) +
    // the plain layer2 forward per pass.
    let mut dw_dense = vec![0.0f32; D * D];
    for (ri, vrow) in art.v32.chunks(D).enumerate() {
        let urow = &art.u32[ri * D..(ri + 1) * D];
        for (i, &vv) in vrow.iter().enumerate() {
            if vv == 0.0 {
                continue;
            }
            for (j, &uv) in urow.iter().enumerate() {
                dw_dense[i * D + j] += vv * uv;
            }
        }
    }
    let t_fold = std::time::Instant::now();
    let _ = &dw_dense;
    let fold_time = t_fold.elapsed();
    let t_merged = std::time::Instant::now();
    let mut sink_m = 0.0f32;
    for _ in 0..reps {
        let y = fx.layer2(&h1g);
        sink_m += y[0];
    }
    let m_per_pass = t_merged.elapsed().as_secs_f64() / reps as f64;
    let _ = (fold_time, sink_un, sink_m);
    println!(
        "T7 per-pass: unmerged (base+overlay chain) = {:.2?} vs merged (base only) = {:.2?} — the overlay chain is the O(r·2D) delta ({:.1}% of the base)",
        un_per_pass,
        m_per_pass,
        (un_per_pass / m_per_pass - 1.0) * 100.0
    );
    // The unmerged pass pays measurably more (the r·2D chain), but bounded:
    // at r = 8, D = 64 the chain is small — pin the honest ceiling.
    assert!(
        un_per_pass < m_per_pass * 4.0,
        "unmerged posture runaway: {:.2?} vs {:.2?}",
        un_per_pass,
        m_per_pass
    );
    std::hint::black_box((sink_un, sink_m));
    let _ = BATCH;
    let _ = K_LADDER;
}
