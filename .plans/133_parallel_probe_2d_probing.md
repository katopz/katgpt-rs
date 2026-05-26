# Plan 133: Parallel-Probe — 2D Probing for Efficient Parallel Thinking

> **Research:** [094 — Parallel-Probe 2D Probing](../.research/094_Parallel_Probe_2D_Probing_Parallel_Thinking.md)
> **Paper:** [arXiv:2602.03845](https://arxiv.org/pdf/2602.03845) — Training-free controller for efficient parallel reasoning via global consensus + deviation pruning
> **Feature Gate:** `parallel_probe` (**Opt-in**, requires GOAT proof before default-on promotion)
> **Depends on:** Plan 010 (multilayer transformer), Plan 005 (speculative module)
> **Status:** ✅ Complete (T1–T3 ✓, T4 GOAT benchmark recorded, pending real inference validation)

## Summary

Implement Parallel-Probe's **2D probing** controller — a training-free, model-agnostic method that monitors N parallel reasoning branches via periodic answer extraction, then uses **consensus-based early stopping** + **deviation-based branch pruning** to reduce sequential tokens by ~30% and total tokens by ~20% while maintaining accuracy.

The key insight: **answer-level consensus across parallel branches is a uniquely cheap global signal** (O(N) per probe step) that our existing features don't exploit. Our `EqrConvergence` uses distribution residuals (O(N×V)), our `TrajectoryPruner` uses bandit scores (requires reward signal). Answer consensus needs only string matching.

---

## Why This, Why Now

- Our `DDTreeBranchCache` already supports branch forking/forwarding/discarding — the infrastructure for parallel branch management exists
- Our `SpeculativeVerifier` trait provides the strategy-pattern extension point for a new `ParallelProbeVerifier`
- The paper proves the signal works across 4 model sizes (0.6B–8B) and 3 benchmarks — strong empirical foundation
- Fills a gap: we have per-trajectory convergence (`eqr_convergence`) and score-based pruning (`tes_loop`), but no **global vote-based** parallel branch control
- Training-free = zero model changes, pure inference-time controller = low risk

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │   SpeculativeVerifier trait    │
                    │  ┌──────────┐ ┌─────────────┐ │
                    │  │Simulated │ │ Leviathan    │ │
                    │  │Verifier  │ │ Verifier     │ │
                    │  └──────────┘ └─────────────┘ │
                    └──────────────────────────────┘
                              │ wraps
                    ┌─────────▼──────────────────────┐
                    │  ParallelProbeController        │
                    │  ┌───────────────────────────┐  │
                    │  │ BranchProbeState[]         │  │
                    │  │  - last_answer             │  │
                    │  │  - disagree_streak         │  │
                    │  │  - is_pruned / is_finished │  │
                    │  └───────────────────────────┘  │
                    │  + consensus_streak: usize      │
                    │  + last_consensus: Option<A>    │
                    │  + probe_step: usize            │
                    │                                 │
                    │  probe(answers) → ProbeDecision │
                    │   - Continue                    │
                    │   - Stop { answer }             │
                    │   - Prune { branch_ids }        │
                    │   - StopAndPrune { .. }         │
                    └─────────────────────────────────┘
                              │ reads
                    ┌─────────▼──────────────────────┐
                    │  ProbingMatrix (N×T)            │
                    │  answers: Vec<Vec<Option<A>>>   │
                    │  - branch_count: usize          │
                    │  - max_probes: usize            │
                    └─────────────────────────────────┘
```

---

## Tasks

### T1: Core Types (`speculative/parallel_probe.rs`)
- [x] `ParallelProbeConfig` — config struct with probe_interval, stability_patience (u), prune_patience (k), warmup_steps (W), min_active_branches, prune_vote_ratio
- [x] `BranchProbeState` — per-branch tracking: last_answer, disagree_streak, is_pruned, is_finished
- [x] `ProbeDecision` enum — Continue / Stop / Prune / StopAndPrune with answer + branch_ids
- [x] `ProbingMatrix<A>` — generic answer matrix N×T with push/row access
- [x] `ParallelProbeController<A>` — main controller: probe(), majority_vote(), should_stop(), should_prune()
- [x] Unit tests: consensus detection, deviation pruning, warmup suppression, edge cases (all agree, all disagree, single branch)

### T2: Answer Extraction Trait
- [x] `AnswerExtractor` trait — `fn extract_answer(&self, tokens: &[usize], config: &Config) -> Option<String>`
- [x] `RegexAnswerExtractor` — regex-based extraction for `\boxed{...}`, `The answer is ...`, numeric patterns
- [x] `ThinkTokenExtractor` — `</think⟩` boundary detection (paper's native approach)
- [x] `DiscreteActionExtractor` — for game domains (Bomber actions, Go moves)
- [x] Unit tests: various answer formats, edge cases (no answer found, multiple answers)

### T3: Integration with Speculative Pipeline
- [x] `ParallelProbeVerifier` wrapping any inner `SpeculativeVerifier`
- [x] Integration with `DDTreeBranchCache` — call `discard_branch()` on pruned branches
- [x] Hook into `speculative_step` — periodic probe at probe_interval tokens
- [x] Wire `ProbeDecision` responses: stop → return consensus answer, prune → discard branches
- [x] Feature gate `parallel_probe` in `Cargo.toml` + `speculative/mod.rs`

### T4: GOAT Proof + Benchmark
- [x] Benchmark: SCOUT-style offline simulation (pre-sample N=64 trajectories, simulate probe control)
- [x] GOAT proof targets (7/7):
  1. Accuracy preservation: probe ≥ SC baseline (within 2%)
  2. Sequential token reduction: ≥ 25%
  3. Total token reduction: ≥ 15%
  4. Warmup necessity: accuracy gap ≥ 2% with vs without
  5. Pruning effectiveness: token savings > 10%
  6. Consensus onset ratio: ≤ 0.5 average
  7. Hyperparameter robustness: < 3% accuracy variance across (k, W) sweep
- [ ] Ablation: each component removed, measure accuracy + token impact
- [ ] Benchmark file: `.benchmarks/023_parallel_probe_goat.md`

---

## Key Design Decisions

1. **Generic answer type `<A: Clone + Eq + Hash>`** — supports String answers (math), usize actions (games), or any discrete output
2. **Controller is standalone** — not embedded in verifier; can be used independently for SCOUT-style offline analysis
3. **Answer extraction is trait-based** — pluggable for different output formats (regex, think-token, game actions)
4. **Pruning uses `DDTreeBranchCache::discard_branch()`** — zero new allocation paths, reuses existing branch management
5. **Config has sensible defaults** — paper's hyperparams (k=3, W=10-15, vote_ratio=0.5) as defaults

---

## Hyperparameter Defaults (from paper)

| Parameter | Default | Range Tested | Sensitivity |
|-----------|---------|--------------|-------------|
| probe_interval (Δ) | 500 tokens | — | Low (paper fixes) |
| stability_patience (u) | varies | — | Moderate |
| prune_patience (k) | 8-12 | {8, 10, 12} | Low (moves along Pareto) |
| warmup_steps (W) | 12-15 | {12, 15} | Moderate (accuracy vs tokens) |
| min_active_branches | 3 | — | Low |
| prune_vote_ratio | 0.5 | — | Low |

---

## Relationship to Existing Plans

| Plan | Feature | Relation |
|------|---------|----------|
| 119 | `eqr_convergence` | Complementary: EqR = distribution residual, Probe = answer consensus |
| 086 | `tes_loop` | Different scope: TES prunes within tree, Probe prunes across parallel trees |
| 109 | `dmax_spd` | Different level: DMax = token-level diffusion, Probe = chain-level reasoning |
| 030 | `bandit` | Extension: Bandit selects pruner strategy, Probe adds budget control |
| 080 | `maxsim` | Unrelated: MaxSim is scoring, Probe is branch control |

---

## File Changes

```
src/speculative/
  ├── parallel_probe.rs     # NEW: ParallelProbeController, ProbingMatrix, ProbeDecision
  ├── answer_extract.rs     # NEW: AnswerExtractor trait + implementations
  ├── mod.rs                # MODIFY: add #[cfg(feature = "parallel_probe")] mod + re-exports
  └── types.rs              # MODIFY: add ParallelProbeConfig if shared

Cargo.toml                  # MODIFY: add parallel_probe = [] feature
.benchmarks/
  └── 023_parallel_probe_goat.md  # NEW: GOAT proof results
```
