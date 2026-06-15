//! Integration test for SwiR via the `ThinkingStrategy` adapter (Plan 275
//! Phase 2 T2.5).
//!
//! Drives [`SwiRStrategyAdapter`] through a mock decode loop with synthetic
//! logits whose entropy follows a controlled Gaussian-mixture-like schedule.
//! Verifies:
//!
//! - **G4 (convex hull):** every emitted soft embedding lies inside the
//!   per-dim `[min_v e(v)[d], max_v e(v)[d]]` box of the vocab.
//! - **Switch schedule:** the entropy schedule produces the expected
//!   Latent → Explicit → Latent → Explicit transitions, and `switch_count`
//!   increments only on Latent → Explicit.
//! - **Convergence (paper §3.4):** once `switch_count = ceil(½ · c_max)`,
//!   the controller injects `CloseThink` (`</think>`) — translated to its
//!   concrete vocab id by the adapter.
//! - **Termination (paper §3.4):** once `switch_count > c_max`, the
//!   controller injects `ForceAnswerPrefix` and starts the answer-budget
//!   countdown; after the budget exhausts, the adapter emits `Terminate`.
//!
//! Run with:
//!
//! ```bash
//! cargo test --features swir_switch_thinking --test swir_strategy_integration -- --nocapture
//! ```

#![cfg(feature = "swir_switch_thinking")]

use katgpt_rs::swir::{SwiRConfig, SwiRStrategyAdapter, ThinkMode};
use katgpt_rs::thinking_cot::{ControlTokenIds, StepContext, StepDirective, ThinkingStrategy};

/// Build a flat `[vocab, dim]` embedding matrix from row vectors.
fn mat(rows: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::new();
    for r in rows {
        out.extend_from_slice(r);
    }
    out
}

/// Inverse-softmax helper: build logits whose softmax approximates `probs`.
/// `logit_i = ln(p_i)` (max-shift invariant).
fn probs_to_logits(probs: &[f32]) -> Vec<f32> {
    probs.iter().map(|&p| (p.max(1e-12)).ln()).collect()
}

/// Per-dim convex-hull check (mirrors `swir::convex_hull_check::in_vocab_convex_hull`
/// but vendored here so this integration test doesn't pull in private modules).
fn in_vocab_convex_hull(soft: &[f32], emb: &[f32], dim: usize) -> bool {
    let vocab = emb.len() / dim;
    if vocab == 0 || soft.len() != dim {
        return false;
    }
    let tol = 1e-4f32;
    for d in 0..dim {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in 0..vocab {
            let e = emb[v * dim + d];
            if e < lo {
                lo = e;
            }
            if e > hi {
                hi = e;
            }
        }
        if soft[d] < lo - tol || soft[d] > hi + tol {
            return false;
        }
    }
    true
}

/// Host-side mock decode loop. Returns the full trace of directives so the
/// caller can assert on the schedule.
fn drive_loop(
    mut adapter: SwiRStrategyAdapter,
    logits_schedule: &[Vec<f32>],
    embedding_matrix: &[f32],
    embedding_dim: usize,
    ids: ControlTokenIds,
    max_steps: u32,
) -> Vec<StepDirective> {
    let mut out = Vec::with_capacity(logits_schedule.len());
    for (i, logits) in logits_schedule.iter().enumerate() {
        let mut ctx = StepContext {
            logits,
            step_index: i as u32,
            max_steps,
            embedding_matrix,
            embedding_dim,
            control_token_ids: ids,
        };
        let d = adapter.on_step(&mut ctx);
        // G4 invariant: every soft embedding must lie in the vocab hull.
        if let StepDirective::EmitSoftEmbedding(ref soft) = d {
            assert!(
                in_vocab_convex_hull(soft, embedding_matrix, embedding_dim),
                "G4 violation at step {i}: soft={soft:?}"
            );
        }
        let terminate = matches!(d, StepDirective::Terminate);
        out.push(d);
        if terminate {
            break;
        }
    }
    out
}

#[test]
fn latent_explicit_latent_explicit_schedule_drives_switches() {
    // 4-token vocab, 3-dim embeddings.
    let emb = mat(&[
        vec![1.0, -2.0, 3.0],
        vec![-4.0, 5.0, -6.0],
        vec![7.0, 8.0, 9.0],
        vec![0.5, -0.5, 0.25],
    ]);
    let dim = 3;

    // Schedule chosen to drive 4 mode transitions:
    //   step 0: high entropy (uniform) → reference set, Latent.
    //   step 1: low entropy (peaky)    → L→E, switch_count=1.
    //   step 2: low entropy (peaky)    → dwell++ in Explicit.
    //   step 3: high entropy           → E→L (dwell=2 ≥ w_e_to_l=1).
    //   step 4: low entropy            → L→E, switch_count=2.
    //   step 5: high entropy           → E→L.
    //   step 6: low entropy            → L→E, switch_count=3.
    let high = probs_to_logits(&[0.25, 0.25, 0.25, 0.25]);
    let low = probs_to_logits(&[0.97, 0.01, 0.01, 0.01]);
    let schedule = vec![high.clone(), low.clone(), low.clone(), high.clone(), low.clone(), high, low];

    let adapter = SwiRStrategyAdapter::with_config(
        4,
        dim,
        SwiRConfig {
            w_e_to_l: 1,
            w_l_to_e: 0,
            c_max: 20, // High — no convergence / termination in this test.
            c_convergence_fraction: 0.5,
            answer_budget_b: 256,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: schedule.len() as u32,
            kurtosis_escape_threshold: f32::INFINITY,
        },
    );
    let ids = ControlTokenIds::default();
    let trace = drive_loop(adapter, &schedule, &emb, dim, ids, schedule.len() as u32);

    // Expected directives (entropy schedule is deterministic):
    //   0: EmitSoftEmbedding (Latent init)
    //   1: EmitToken(0)       (L→E)
    //   2: EmitToken(0)       (stay Explicit)
    //   3: EmitSoftEmbedding  (E→L)
    //   4: EmitToken(0)       (L→E)
    //   5: EmitSoftEmbedding  (E→L)
    //   6: EmitToken(0)       (L→E)
    assert!(matches!(trace[0], StepDirective::EmitSoftEmbedding(_)));
    assert!(matches!(trace[1], StepDirective::EmitToken(0)));
    assert!(matches!(trace[2], StepDirective::EmitToken(0)));
    assert!(matches!(trace[3], StepDirective::EmitSoftEmbedding(_)));
    assert!(matches!(trace[4], StepDirective::EmitToken(0)));
    assert!(matches!(trace[5], StepDirective::EmitSoftEmbedding(_)));
    assert!(matches!(trace[6], StepDirective::EmitToken(0)));
}

#[test]
fn convergence_fires_close_think_at_half_cmax() {
    // 3-token vocab, 2-dim.
    let emb = mat(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
    let dim = 2;

    // c_max=4, conv=ceil(0.5*4)=2 → convergence window hits at switch_count ∈ [2,4].
    // Force 3 L→E switches by alternating entropy:
    //   step 0: high → Latent init.
    //   step 1: low  → L→E (switch_count=1, no conv yet).
    //   step 2: low  → dwell++.
    //   step 3: high → E→L.
    //   step 4: low  → L→E (switch_count=2 ≥ conv_at=2 → enqueue CloseThink).
    //   step 5: low  → drain CloseThink → InjectTokens([think_close_id]).
    let high = probs_to_logits(&[0.4, 0.3, 0.3]);
    let low = probs_to_logits(&[0.97, 0.02, 0.01]);
    let schedule = vec![
        high,
        low.clone(),
        low.clone(),
        probs_to_logits(&[0.4, 0.3, 0.3]),
        low.clone(),
        low, // step 5: drain CloseThink.
    ];

    let think_close_id = 42; // sentinel for easy detection.
    let ids = ControlTokenIds {
        think_open: 0,
        think_close: think_close_id,
        force_answer_prefix: 99,
    };

    let adapter = SwiRStrategyAdapter::with_config(
        3,
        dim,
        SwiRConfig {
            w_e_to_l: 1,
            w_l_to_e: 0,
            c_max: 4,
            c_convergence_fraction: 0.5, // conv_at = 2
            answer_budget_b: 256,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: schedule.len() as u32,
            kurtosis_escape_threshold: f32::INFINITY,
        },
    );
    let trace = drive_loop(adapter, &schedule, &emb, dim, ids, schedule.len() as u32);

    // Step 4 emits a token (the L→E step) — CloseThink is queued but not yet drained.
    assert!(
        matches!(trace[4], StepDirective::EmitToken(0)),
        "step 4 should EmitToken while queuing CloseThink, got {:?}",
        trace[4]
    );
    // Step 5 drains CloseThink.
    match &trace[5] {
        StepDirective::InjectTokens(ids) => {
            assert_eq!(ids, &vec![think_close_id], "CloseThink must resolve to think_close id");
        }
        other => panic!(
            "step 5 should inject CloseThink=[{think_close_id}], got {other:?}"
        ),
    }
}

#[test]
fn termination_fires_force_answer_then_terminate() {
    let emb = mat(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    let dim = 2;

    // Bypass convergence (conv threshold = 10) so we go straight to overthinking
    // guard at switch_count > c_max=1.
    let high = probs_to_logits(&[0.5, 0.5]);
    let low = probs_to_logits(&[0.99, 0.01]);
    // Drive two L→E switches (switch_count=2 > c_max=1).
    let schedule = vec![
        high.clone(), // 0: Latent init.
        low.clone(),  // 1: L→E, switch_count=1.
        low.clone(),  // 2: Explicit, dwell++.
        high.clone(), // 3: E→L.
        low.clone(),  // 4: L→E, switch_count=2 > c_max → enqueue ForceAnswerPrefix, budget=1.
        low.clone(),  // 5: drain ForceAnswerPrefix, budget 1→0.
        low.clone(),  // 6: budget=0 → Terminate.
        low,          // 7: shouldn't reach.
    ];

    let force_answer_id = 7;
    let ids = ControlTokenIds {
        think_open: 0,
        think_close: 1,
        force_answer_prefix: force_answer_id,
    };

    let adapter = SwiRStrategyAdapter::with_config(
        2,
        dim,
        SwiRConfig {
            w_e_to_l: 1,
            w_l_to_e: 0,
            c_max: 1,
            c_convergence_fraction: 10.0, // Skip convergence.
            answer_budget_b: 1,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: schedule.len() as u32,
            kurtosis_escape_threshold: f32::INFINITY,
        },
    );
    let trace = drive_loop(adapter, &schedule, &emb, dim, ids, schedule.len() as u32);

    // Step 4 should emit a token while queuing ForceAnswerPrefix.
    assert!(
        matches!(trace[4], StepDirective::EmitToken(0)),
        "step 4: {:?}",
        trace[4]
    );
    // Step 5 drains ForceAnswerPrefix.
    match &trace[5] {
        StepDirective::InjectTokens(v) => assert_eq!(*v, vec![force_answer_id]),
        other => panic!("step 5 should inject ForceAnswerPrefix, got {other:?}"),
    }
    // Step 6 terminates.
    assert!(
        matches!(trace[6], StepDirective::Terminate),
        "step 6 should Terminate, got {:?}",
        trace[6]
    );
    // Trace must stop after Terminate.
    assert_eq!(trace.len(), 7);
}

#[test]
fn soft_embedding_satisfies_g4_throughout_long_run() {
    // 8-token vocab, 4-dim embeddings with varied values (so the hull is non-trivial).
    let emb = mat(&[
        vec![1.0, -2.0, 3.0, 0.5],
        vec![-4.0, 5.0, -6.0, 2.5],
        vec![7.0, 8.0, 9.0, -1.0],
        vec![0.0, -1.0, 2.0, 1.0],
        vec![2.0, 2.0, -2.0, -2.0],
        vec![-1.0, 0.5, 1.5, 3.0],
        vec![3.5, -3.5, 0.0, 0.0],
        vec![-2.0, 1.0, -1.0, -3.0],
    ]);
    let dim = 4;

    // Long schedule of random-feeling but deterministic entropies. We force
    // the controller into Latent mode repeatedly to exercise the soft-embedding
    // path under many different prob distributions.
    let mut rng = 0xc0ffee_u32;
    let mut schedule: Vec<Vec<f32>> = Vec::with_capacity(64);
    for _ in 0..64 {
        let mut probs = [0.0f32; 8];
        let mut sum = 0.0;
        for p in probs.iter_mut() {
            rng = rng.wrapping_mul(2654435761).wrapping_add(98765);
            *p = (rng as f32) / (u32::MAX as f32);
            sum += *p;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        schedule.push(probs_to_logits(&probs));
    }

    let adapter = SwiRStrategyAdapter::with_config(
        8,
        dim,
        SwiRConfig {
            w_e_to_l: 1,
            w_l_to_e: 0,
            c_max: 64, // No termination this run.
            c_convergence_fraction: 10.0,
            answer_budget_b: 256,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: 64,
            kurtosis_escape_threshold: f32::INFINITY,
        },
    );
    let ids = ControlTokenIds::default();

    // Run; drive_loop asserts G4 on every EmitSoftEmbedding step.
    let trace = drive_loop(adapter, &schedule, &emb, dim, ids, 64);

    // Must have produced at least some soft-embedding steps (otherwise the test
    // is vacuous — assert to catch a regression where the controller never
    // enters Latent).
    let soft_count = trace
        .iter()
        .filter(|d| matches!(d, StepDirective::EmitSoftEmbedding(_)))
        .count();
    assert!(
        soft_count >= 5,
        "expected at least 5 soft-embedding steps to exercise G4, got {soft_count}"
    );
}

#[test]
fn default_adapter_starts_in_latent_mode() {
    // Sanity check: the adapter inherits the controller's initial Latent mode.
    let adapter = SwiRStrategyAdapter::new(4, 3);
    assert_eq!(adapter.controller().mode(), ThinkMode::Latent);
    assert_eq!(adapter.controller().switch_count(), 0);
}

#[test]
fn explicit_mode_emits_token_zero_placeholder() {
    // In Explicit mode the adapter emits 0 — the host must overwrite with the
    // sampled id. We verify the placeholder convention here so callers can
    // rely on it.
    let emb = mat(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    let dim = 2;
    let high = probs_to_logits(&[0.5, 0.5]);
    let low = probs_to_logits(&[0.99, 0.01]);

    let mut adapter = SwiRStrategyAdapter::with_config(
        2,
        dim,
        SwiRConfig {
            w_e_to_l: 64,   // Don't bounce back to Latent during this test.
            w_l_to_e: 0,
            c_max: 64,
            c_convergence_fraction: 10.0,
            answer_budget_b: 256,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: 8,
            kurtosis_escape_threshold: f32::INFINITY,
        },
    );
    let ids = ControlTokenIds::default();

    // Step 0: Latent.
    let mut ctx = StepContext {
        logits: &high,
        step_index: 0,
        max_steps: 8,
        embedding_matrix: &emb,
        embedding_dim: dim,
        control_token_ids: ids,
    };
    let _ = adapter.on_step(&mut ctx);
    // Step 1: Latent → Explicit.
    let mut ctx = StepContext {
        logits: &low,
        step_index: 1,
        max_steps: 8,
        embedding_matrix: &emb,
        embedding_dim: dim,
        control_token_ids: ids,
    };
    let d = adapter.on_step(&mut ctx);
    assert!(matches!(d, StepDirective::EmitToken(0)), "got {d:?}");
}
