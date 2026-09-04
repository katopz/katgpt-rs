//! Issue 699 T1–T3 — structural CoT halting PoC substrate: integration gates.
//!
//! Run with:
//! ```bash
//! cargo test --release --test issue_699_structural_cot_halt --features structural_cot_halt
//! ```
//!
//! Synthetic-trace gates over the [`StructuralTraceMonitor`] public API (the
//! T4 defend-wrong PoC in riir-poc drives this same API against real traces):
//!
//! - **G1 determinism** — the same trace through two fresh monitors produces
//!   a bit-identical decision sequence (no RNG, no HashMap iteration
//!   anywhere in the decision path — fixed arrays only).
//! - **G4 alloc-free** — ≥10k steps at steady state allocate exactly zero
// Issue 721 T3: install the tracking allocator in THIS test binary. The root
// lib no longer registers a `#[global_allocator]` as a library (that chose
// the process allocator for every downstream binary and conflicted with any
// consumer's own registration). Replaces the Issue-682 force-link, which
// existed only to keep the root's library-level shim linked.
#[path = "common/alloc_tracking.rs"]
mod alloc_tracking;

use fastrand::Rng;
use katgpt_core::structural_cot_halt::{
    BacktrackRevisitHalt, ClassifiedPattern, HaltPolicy, Pattern, SelfLoopHalt,
    StructuralHaltDecision, StructuralHaltReason, StructuralTraceMonitor, compose_votes,
    normalized_answer_hash,
};

/// Run a text trace, returning every decision (for G1 comparison).
fn run_text(monitor: &mut StructuralTraceMonitor, answers: &[&str]) -> Vec<u8> {
    answers
        .iter()
        .map(|a| match monitor.step(a) {
            StructuralHaltDecision::Continue => 0u8,
            StructuralHaltDecision::Halt { reason, step } => {
                1 + reason as u8 * 16 + (step % 16) as u8
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// Policy firing on the canonical synthetic traces
// ─────────────────────────────────────────────────────────────────────

#[test]
fn late_landing_trace_self_loop_fires_at_exactly_k() {
    // Paper self-loop heuristic, K=2: [A, B, C] wandering, then the model
    // lands on X and verifies. Halt must land EXACTLY on the K-th
    // consecutive verification — never at K-1, never later.
    let mut m = StructuralTraceMonitor::new(SelfLoopHalt::paper_default().into());
    let trace = ["A", "B", "C", "X", "X", "X", "X"];
    let decisions: Vec<StructuralHaltDecision> = trace.iter().map(|a| m.step(a)).collect();
    for (i, d) in decisions.iter().take(3).enumerate() {
        assert_eq!(
            *d,
            StructuralHaltDecision::Continue,
            "wander step {}",
            i + 1
        );
    }
    // Step 4 (X proposed): Correct, run 0. Step 5: first verify, run 1 < 2.
    assert_eq!(decisions[3], StructuralHaltDecision::Continue);
    assert_eq!(
        decisions[4],
        StructuralHaltDecision::Continue,
        "run=1 < K=2 must not fire"
    );
    // Step 6: the K-th verification fires.
    assert_eq!(
        decisions[5],
        StructuralHaltDecision::Halt {
            reason: StructuralHaltReason::SelfLoop,
            step: 6
        }
    );
    // Frozen episode: the post-halt step replays the recorded decision.
    assert_eq!(decisions[6], decisions[5]);
    assert_eq!(m.step_count(), 6, "frozen steps do not consume state");
}

#[test]
fn explorer_trace_revisit_fires_on_first_revisit() {
    // Paper backtrack heuristic: A,B,A — the first revisit of an abandoned
    // answer IS the cycle signal.
    let mut m = StructuralTraceMonitor::new(BacktrackRevisitHalt.into());
    let trace = ["A", "B", "A", "B"];
    let decisions: Vec<StructuralHaltDecision> = trace.iter().map(|a| m.step(a)).collect();
    assert_eq!(decisions[0], StructuralHaltDecision::Continue);
    assert_eq!(decisions[1], StructuralHaltDecision::Continue);
    assert_eq!(
        decisions[2],
        StructuralHaltDecision::Halt {
            reason: StructuralHaltReason::BacktrackRevisit,
            step: 3
        }
    );
    assert_eq!(decisions[3], decisions[2], "frozen replay");
}

#[test]
fn no_cycle_all_distinct_never_halts() {
    // 12 all-distinct answers exercise ring wraparound; no policy may fire.
    let answers: Vec<String> = (0..12).map(|i| format!("answer-{i}")).collect();
    let refs: Vec<&str> = answers.iter().map(String::as_str).collect();
    for policy in [
        HaltPolicy::Never,
        HaltPolicy::SelfLoop(2),
        HaltPolicy::BacktrackRevisit,
        HaltPolicy::Auto,
    ] {
        let mut m = StructuralTraceMonitor::new(policy);
        for (i, d) in refs.iter().map(|a| m.step(a)).enumerate() {
            assert_eq!(
                d,
                StructuralHaltDecision::Continue,
                "policy {policy:?} fired at step {} on an acyclic trace",
                i + 1
            );
        }
        assert!(!m.is_halted());
    }
}

// ─────────────────────────────────────────────────────────────────────
// T3 fusion end-to-end
// ─────────────────────────────────────────────────────────────────────

#[test]
fn auto_fusion_selects_backtrack_for_explorer_and_halts() {
    let mut m = StructuralTraceMonitor::auto();
    let c: ClassifiedPattern = m.classify_prefix();
    assert_eq!(
        c.pattern,
        Pattern::LateLanding,
        "empty prefix: converged default"
    );
    let _ = m.step("A");
    let _ = m.step("B");
    let c = m.classify_prefix();
    assert_eq!(c.pattern, Pattern::Explorer);
    assert_eq!(c.policy, HaltPolicy::BacktrackRevisit);
    assert_eq!(
        m.step("A"),
        StructuralHaltDecision::Halt {
            reason: StructuralHaltReason::BacktrackRevisit,
            step: 3
        }
    );
}

#[test]
fn auto_fusion_selects_self_loop_for_late_landing() {
    // Wander then a verify tail: once the tail reaches half the trace the
    // fusion flips to LateLanding/SelfLoop and cuts on the patience.
    let mut m = StructuralTraceMonitor::auto();
    let trace = ["A", "B", "C", "X", "X", "X", "X", "X", "X"];
    let decisions: Vec<StructuralHaltDecision> = trace.iter().map(|a| m.step(a)).collect();
    // Steps 1–7 Continue (tail < 1/2 even at run 3); the fusion flips at
    // step 8 (tail 4/8) where run 4 ≥ K=3 → halt lands on step 8.
    for (i, d) in decisions.iter().take(7).enumerate() {
        assert_eq!(*d, StructuralHaltDecision::Continue, "step {}", i + 1);
    }
    assert_eq!(
        decisions[7],
        StructuralHaltDecision::Halt {
            reason: StructuralHaltReason::SelfLoop,
            step: 8
        }
    );
    assert_eq!(decisions[8], decisions[7], "frozen replay");
    // K derivation: this histogram's purity is (25+1+1+1)/64 = 0.4375 < 0.75
    // → the contested-landing K=3, pinned via the prefix classifier.
    let mut obs = StructuralTraceMonitor::new(HaltPolicy::Never);
    for a in &trace[..8] {
        let _ = obs.step(a);
    }
    assert_eq!(
        obs.classify_prefix().policy,
        HaltPolicy::SelfLoop(3),
        "contested landing (purity 0.4375) must grade K=3"
    );
}

#[test]
fn purity_dominant_landing_grades_k2() {
    // A then X ×9: purity (100+1)/121 ≈ 0.835 ≥ 0.75 → K=2.
    let mut obs = StructuralTraceMonitor::new(HaltPolicy::Never);
    let mut trace = vec!["A"];
    trace.extend(std::iter::repeat_n("X", 9));
    for a in &trace {
        let _ = obs.step(a);
    }
    let c = obs.classify_prefix();
    assert_eq!(c.pattern, Pattern::LateLanding);
    assert_eq!(c.policy, HaltPolicy::SelfLoop(2));
}

// ─────────────────────────────────────────────────────────────────────
// G1 determinism
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g1_double_run_bit_identical_decision_sequence() {
    // A trace exercising every transition class + both policies + fusion,
    // run twice through fresh monitors: identical decision encodings.
    let mut trace: Vec<&str> = vec!["A", "B", "A", "C", "A", "A"];
    trace.extend(std::iter::repeat_n("X", 12));
    trace.extend(["Y", "Z", "Y", "W"]);

    let decisions_a = {
        let mut m = StructuralTraceMonitor::auto();
        run_text(&mut m, &trace)
    };
    let decisions_b = {
        let mut m = StructuralTraceMonitor::auto();
        run_text(&mut m, &trace)
    };
    assert_eq!(
        decisions_a, decisions_b,
        "G1: identical trace → identical decisions"
    );

    // And again with explicit policies over the same trace.
    let pol = [
        HaltPolicy::Never,
        HaltPolicy::SelfLoop(3),
        HaltPolicy::BacktrackRevisit,
    ];
    for p in pol {
        let a = {
            let mut m = StructuralTraceMonitor::new(p);
            run_text(&mut m, &trace)
        };
        let b = {
            let mut m = StructuralTraceMonitor::new(p);
            run_text(&mut m, &trace)
        };
        assert_eq!(a, b, "G1 under {p:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// Vote composition with the numeric family
// ─────────────────────────────────────────────────────────────────────

#[test]
fn compose_votes_merges_the_structural_family() {
    // The T4 composition surface: votes from multiple sources merge by the
    // documented precedence (any halt wins; earliest step; tie → slice
    // order). Cross-family composition (the numeric arbiter) is covered by
    // the in-module `vote_from_numeric_bridge` test, which runs when BOTH
    // features are enabled — this target stays single-feature.
    let a = StructuralHaltDecision::Continue;
    let b = StructuralHaltDecision::Halt {
        reason: StructuralHaltReason::SelfLoop,
        step: 6,
    };
    assert_eq!(compose_votes(&[]), StructuralHaltDecision::Continue);
    assert_eq!(compose_votes(&[a]), StructuralHaltDecision::Continue);
    assert_eq!(compose_votes(&[a, b]), b);
    // Earliest step wins within the family; tie → first in the slice.
    let c = StructuralHaltDecision::Halt {
        reason: StructuralHaltReason::BacktrackRevisit,
        step: 4,
    };
    assert_eq!(compose_votes(&[b, c]), c);
    assert_eq!(compose_votes(&[c, c]), c, "tie keeps the first slice vote");
}

// ─────────────────────────────────────────────────────────────────────
// Vacuity: the feature-off shape
// ─────────────────────────────────────────────────────────────────────

#[test]
fn never_halt_control_is_the_flag_off_shape() {
    // Three-layer vacuity story for the flag-off default build:
    // 1. the whole module is `#[cfg(feature = "structural_cot_halt")]` —
    //    with the flag off, the monitor does not exist to perturb anything
    //    (this test target itself is absent: `required-features`);
    // 2. the feature is NOT in any default list (opt-in per the issue);
    // 3. behaviorally, `HaltPolicy::Never` reproduces the no-monitor
    //    control: every decision Continue, state consumed, nothing halted —
    //    exactly what a consumer without the monitor experiences.
    let mut control = StructuralTraceMonitor::new(HaltPolicy::Never);
    let mut armed = StructuralTraceMonitor::new(SelfLoopHalt::paper_default().into());
    let mut monitorless: Vec<u8> = Vec::new();
    let mut nevers: Vec<u8> = Vec::new();
    let mut halt_event: Option<usize> = None;
    let trace = ["A", "B", "A", "A", "A", "C", "C"];
    for (i, a) in trace.iter().enumerate() {
        monitorless.push(0); // the no-monitor consumer always continues
        nevers.push(match control.step(a) {
            StructuralHaltDecision::Continue => 0,
            _ => 1,
        });
        if armed.step(a).is_halt() {
            halt_event.get_or_insert(i + 1);
        }
    }
    assert_eq!(nevers, monitorless, "Never ≡ no-monitor, step for step");
    // Exactly ONE halt event (the K-th verify at step 5); the two post-halt
    // steps replay the frozen decision rather than firing again.
    assert_eq!(
        halt_event,
        Some(5),
        "SelfLoop K=2 must fire exactly once, at step 5"
    );
}

// ─────────────────────────────────────────────────────────────────────
// G4 alloc-free steady state
// ─────────────────────────────────────────────────────────────────────

/// Liveness sentinel (Issue 682): FAIL the audit if the TrackingAllocator
/// is not actually installed. Debug builds only — the whole tracking
/// machinery is `#[cfg(debug_assertions)]` in katgpt-core (see its module
/// docs), so in a `--release` run the alloc assertions compile away and the
/// G4 test degrades to exercising the workload (the house bench_284
/// pattern; the release run of this gate is compile + behavior only).
#[cfg(debug_assertions)]
fn assert_alloc_tracking_live() {
    katgpt_core::alloc::reset_alloc_stats();
    let _probe: Vec<u8> = vec![0u8; 64];
    let (count, _) = katgpt_core::alloc::get_alloc_stats();
    assert!(
        count >= 1,
        "TrackingAllocator not installed in this binary — alloc gate is VACUOUS (Issue 682)"
    );
    katgpt_core::alloc::reset_alloc_stats();
}

#[cfg(not(debug_assertions))]
fn assert_alloc_tracking_live() {}

#[test]
fn g4_alloc_free_steady_state_10k_steps() {
    const STEPS: u64 = 10_000;
    const WARMUP: u64 = 256;

    assert_alloc_tracking_live();

    // Text path: normalization streams into a stack BLAKE3 hasher; the
    // answer String is never materialized. The answer CORPUS is pre-built
    // OUTSIDE the measured window — the test's own format! allocations must
    // not pollute the count (the first debug run caught exactly that: 1
    // alloc/step was the harness's format!, not the monitor).
    let answers: Vec<String> = (0..97usize).map(|i| format!("answer {i}")).collect();
    let refs: Vec<&str> = answers.iter().map(String::as_str).collect();
    let mut text = StructuralTraceMonitor::auto();
    for a in &refs {
        let _ = text.step(a);
    }
    #[cfg(debug_assertions)]
    {
        katgpt_core::alloc::reset_alloc_stats();
    }
    for k in 0..STEPS {
        let _ = text.step(refs[(k % 97) as usize]);
    }
    #[cfg(debug_assertions)]
    let (text_allocs, text_bytes) = katgpt_core::alloc::get_alloc_stats();

    // Raw-key path (the dd-tree / MCTS seam): no hashing at all.
    let mut raw = StructuralTraceMonitor::auto();
    for k in 0..WARMUP {
        let _ = raw.step_key(k % 8);
    }
    #[cfg(debug_assertions)]
    {
        katgpt_core::alloc::reset_alloc_stats();
    }
    for k in 0..STEPS {
        let _ = raw.step_key(k % 8);
    }
    #[cfg(debug_assertions)]
    let (raw_allocs, raw_bytes) = katgpt_core::alloc::get_alloc_stats();

    // Classifier is re-runnable allocation-free (the Auto policy calls it
    // per step already; pin it directly).
    #[cfg(debug_assertions)]
    {
        katgpt_core::alloc::reset_alloc_stats();
    }
    let mut pattern = Pattern::Explorer;
    for _ in 0..STEPS {
        let c = text.classify_prefix();
        pattern = c.pattern;
    }
    #[cfg(debug_assertions)]
    {
        let (cls_allocs, cls_bytes) = katgpt_core::alloc::get_alloc_stats();
        assert_eq!(
            text_allocs, 0,
            "text step() allocated {text_allocs} times / {text_bytes} bytes over {STEPS} steps"
        );
        assert_eq!(
            raw_allocs, 0,
            "step_key() allocated {raw_allocs} times / {raw_bytes} bytes over {STEPS} steps"
        );
        assert_eq!(
            cls_allocs, 0,
            "classify_prefix() allocated {cls_allocs} times / {cls_bytes} bytes over {STEPS} runs"
        );
    }
    assert!(matches!(pattern, Pattern::Explorer | Pattern::LateLanding));
}

// ─────────────────────────────────────────────────────────────────────
// Plug seam: MCTS budget loop
// ─────────────────────────────────────────────────────────────────────

/// A trivial 3-action game that never terminates within the test horizon.
#[derive(Clone)]
struct TrivialGame {
    tick: u32,
}

impl katgpt_core::traits::GameState for TrivialGame {
    type Action = u8;

    fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
        vec![0, 1, 2]
    }

    fn advance_inplace(&mut self, _action: &Self::Action, _player_id: u8) {
        self.tick += 1;
    }

    fn is_terminal(&self) -> bool {
        false
    }

    fn reward(&self, _player_id: u8) -> f32 {
        0.5
    }

    fn tick(&self) -> u32 {
        self.tick
    }
}

fn constant_heuristic(_s: &TrivialGame, _pid: u8) -> f32 {
    0.5
}

#[test]
fn mcts_seam_never_halt_monitor_matches_plain_search() {
    // A monitor whose closure never votes Halt must return the SAME action
    // as the unarmed search under the same seed (the seam only observes).
    let run = |armed: bool| {
        let state = TrivialGame { tick: 0 };
        let mut rng = Rng::with_seed(0xDA7A);
        if armed {
            let mut monitor = StructuralTraceMonitor::new(HaltPolicy::Never);
            katgpt_core::mcts::mcts_search_structural_halted(
                &state,
                0,
                256,
                8,
                &constant_heuristic,
                &mut katgpt_core::traits::RandomRolloutPolicy,
                &mut rng,
                &mut monitor,
            )
        } else {
            katgpt_core::mcts::mcts_search(&state, 0, 256, 8, &constant_heuristic, &mut rng)
        }
    };
    let plain = run(false);
    let seamed = run(true);
    assert_eq!(
        plain, seamed,
        "a never-halt monitor must not perturb the search outcome"
    );
}

#[test]
fn mcts_seam_halt_vote_cuts_the_budget_loop() {
    // SelfLoop K=1 fires on the FIRST repeated best-action identity. The
    // budget loop consults the monitor once per iteration, after
    // backpropagation: iteration 1 expands the root's first child, so the
    // first observation is that child's action identity (fresh answer);
    // iteration 2 observes the SAME identity (the tree still has that one
    // root child as most-visited) → verify → Halt → the search must cut
    // THERE. A broken seam would let the loop run toward the 10_000 budget
    // and the monitor's step count would blow far past 2.
    let state = TrivialGame { tick: 0 };
    let mut rng = Rng::with_seed(0xBEEF);
    let mut monitor = StructuralTraceMonitor::new(SelfLoopHalt::new(1).into());
    let _action = katgpt_core::mcts::mcts_search_structural_halted(
        &state,
        0,
        10_000,
        8,
        &constant_heuristic,
        &mut katgpt_core::traits::RandomRolloutPolicy,
        &mut rng,
        &mut monitor,
    );
    assert!(monitor.is_halted(), "the monitor must have fired");
    // Exact halt step for THIS seed is 4: expansion picks a random
    // unexpanded action and select_inline re-leafs at the root while it has
    // unexpanded actions, so the first three observations walk the
    // best-child tie-break (a → b → c, `max_by_key` last-max) before the
    // identity repeats and K=1 fires. The load-bearing bound is the CUT: an
    // unbroken loop against budget 10_000 runs ~1,100 iterations; the seam
    // must cut within a handful.
    assert!(
        monitor.step_count() <= 8,
        "the seam must cut the loop almost immediately (measured 4 for this \
         seed), got {} — a broken seam runs to budget",
        monitor.step_count()
    );
}

#[test]
fn mcts_budget_check_contract_cuts_at_first_halt() {
    // The BudgetHaltCheck CONTRACT: true = cut. Pin it directly (the
    // wrapper's closure is exactly this shape) so a regression in the seam
    // semantics is caught without depending on visit-pattern luck.
    let mut monitor = StructuralTraceMonitor::new(SelfLoopHalt::new(1).into());
    let mut check = |key: u64| matches!(monitor.step_key(key), StructuralHaltDecision::Halt { .. });
    // Same answer identity repeatedly: the first verify (k=1) fires.
    assert!(!check(42), "establishment never fires");
    assert!(check(42), "first verify fires at K=1 → the loop must cut");
    assert!(check(42), "frozen replay keeps voting Halt");
}

// ─────────────────────────────────────────────────────────────────────
// Plug seam: dd-tree TreeBuilder patience loop
// ─────────────────────────────────────────────────────────────────────

#[test]
fn dd_tree_structural_monitor_cuts_a_full_budget_build() {
    use katgpt_core::traits::NoPruner;
    use katgpt_speculative::dd_tree::TreeBuilder;

    // Token 3 is argmax at EVERY depth → the best-first pop order is
    // (d0,t3), (d1,t3), (d2,t3), … — the same token identity repeatedly, so
    // SelfLoop K=1 fires on the SECOND pop. Without the monitor the build
    // fills the entire tree_budget.
    let depth_probs = [0.05_f32, 0.05, 0.05, 0.85];
    let marginals: [&[f32]; 4] = [&depth_probs, &depth_probs, &depth_probs, &depth_probs];
    let mut config = katgpt_core::Config::micro();
    config.vocab_size = 4;
    config.draft_lookahead = 4;
    config.tree_budget = 16;
    // Disable the SCORE patience so only the structural monitor can cut.
    config.early_exit_patience = 0;
    config.early_exit_gap = 0.0;

    // Unarmed control: the build fills its budget.
    let mut builder = TreeBuilder::new(&config);
    let tree = builder.build(&marginals, &config, &NoPruner, false);
    let control_len = tree.len();
    assert!(
        control_len > 2,
        "control build should fill past the monitor's cut point, got {control_len}"
    );

    // Armed: SelfLoop K=1 cuts at the second consecutive dominant token.
    let mut builder = TreeBuilder::new(&config);
    builder.set_structural_halt_monitor(Some(StructuralTraceMonitor::new(
        SelfLoopHalt::new(1).into(),
    )));
    let tree = builder.build(&marginals, &config, &NoPruner, false);
    assert_eq!(
        tree.len(),
        2,
        "structural halt must cut the expansion at the second pop (K=1), got {}",
        tree.len()
    );
    // And the cut nodes are the dominant-token chain.
    assert_eq!(tree[0].token_idx, 3);
    assert_eq!(tree[1].token_idx, 3);

    // Removing the monitor restores the legacy behavior exactly.
    let mut builder = TreeBuilder::new(&config);
    builder.set_structural_halt_monitor(Some(StructuralTraceMonitor::new(
        SelfLoopHalt::new(1).into(),
    )));
    builder.set_structural_halt_monitor(None);
    let tree = builder.build(&marginals, &config, &NoPruner, false);
    assert_eq!(tree.len(), control_len, "None monitor ≡ unarmed build");
}

// ─────────────────────────────────────────────────────────────────────
// API surface sanity (the PoC drives exactly these)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn public_api_surface_smoke() {
    // The exact types the riir-poc T4 arm consumes.
    let _policies: [HaltPolicy; 4] = [
        HaltPolicy::Auto,
        HaltPolicy::SelfLoop(SelfLoopHalt::paper_default().k),
        BacktrackRevisitHalt.into(),
        HaltPolicy::Never,
    ];
    let _monitor = StructuralTraceMonitor::new(SelfLoopHalt { k: 2 }.into());
    let _auto = StructuralTraceMonitor::auto();
    // Normalization identity stability across the API.
    let h1 = normalized_answer_hash("Answer:  42");
    let h2 = normalized_answer_hash("answer: 42 ");
    assert_eq!(h1, h2);
}
