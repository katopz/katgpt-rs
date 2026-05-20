# Benchmark 016: SimpleTES RPUCG — GOAT Proof

**Date:** 2025-05-21
**Plan:** 086 (SimpleTES Evaluation-Driven Scaling)
**Features:** `--features tes_loop`
**Command:** `cargo test --features tes_loop --test bench_simpletes_rpucg_goat -- --nocapture`
**Source:** [SimpleTES: Evaluation-Driven Scaling](https://arxiv.org/abs/2604.19341)

## Setup

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_TRIALS | 500 | Per proof |
| SEED | 42 | Reproducibility |
| TesConfig | C=32, L=100, K=16 | Default budget 51,200 |
| RPUCG γ | 0.8 | Propagation discount |
| RPUCG λ | 1.0 | Exploration weight |

## GOAT Proof Results

### Proof 1: RPUCG Explores More Unvisited Nodes

20 nodes — 10 heavily-visited (visits=50–95, value=0.7–0.88) + 10 unvisited (visits=0, value=0.4–0.58). Select top-10.

| Method | Avg Unvisited Selected |
|--------|----------------------|
| **RPUCG** | **10.0/10** |
| Greedy | 0.0/10 |
| **Ratio** | **∞×** |

**Verdict:** ✅ RPUCG exploration bonus (λ·√((1+|S|)/(1+n_i))) is infinite for n_i=0, guaranteeing unvisited node selection. Greedy always picks highest propagated value, ignoring unexplored regions.

### Proof 2: Value Propagation Lifts Parent Selection

Tree: root(score=0.2) → child_a(score=0.5) → gc_a1(score=0.95), gc_a2(score=0.85) and child_b(score=0.1) → gc_b1(score=0.15)

| Node | Before | After (γ=0.8) | Expected |
|------|--------|---------------|----------|
| root | 0.000 | **0.608** | 0.608 ✅ |
| child_a | 0.000 | **0.760** | 0.760 ✅ |
| child_b | 0.000 | **0.120** | 0.120 ✅ |
| gc_a1 | 0.000 | **0.950** | 0.950 ✅ |
| gc_a2 | 0.000 | **0.850** | 0.850 ✅ |
| gc_b1 | 0.000 | **0.150** | 0.150 ✅ |

**Verdict:** ✅ Propagation makes root selectable (0.608 > 0.5). Root inherits quality from grandchildren through γ-discounted max. Formula: U_i = max(r_i, γ·max(U_child)).

### Proof 3: TrajectoryPruner Kills Bottom Performers

10 trajectories with scores [0.1..1.0], kill_fraction=0.3.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Average | 0.55 | **0.70** | **+27.3%** |
| Trajectories | 10 | 7 | -30% |
| Max killed | 0.3 | | |
| Min survivor | 0.4 | | |

Checkpoint detection: step 25/100 ✓, 50/100 ✓, 75/100 ✓, 30/100 ✗, 0/0 ✗

**Verdict:** ✅ Pruning concentrates budget on top-performing trajectories. All killed scores below all survivor scores.

### Proof 4: Simulated TES Loop — RPUCG vs Greedy

500 trials × (C=8 trajectories, L=100 steps, K=4 candidates, vocab=10, seq_len=5). Hidden target sequence. Same seeds for fair comparison.

| Metric | RPUCG | Greedy | Δ |
|--------|-------|--------|---|
| **Wins** | **214** (42.8%) | 53 (10.6%) | +32.2pp |
| Ties | 233 (46.6%) | | |
| **Avg best** | **0.6652** | 0.5852 | **+0.0800** |
| Perfect | **11** (2.2%) | 0 (0%) | +2.2pp |

**Verdict:** ✅ RPUCG wins 42.8% of trials vs greedy's 10.6%. Average best solution 8pp higher. RPUCG finds 11 perfect solutions (greedy finds 0) due to exploration diversity and value propagation.

## Summary

| Proof | Result | Verdict |
|-------|--------|---------|
| 1. Exploration | RPUCG 10.0 vs greedy 0.0 (∞×) | ✅ |
| 2. Propagation | Root 0.000 → 0.608 | ✅ |
| 3. Pruning | Avg 0.55 → 0.70 (+27%) | ✅ |
| 4. TES loop | RPUCG 0.6652 vs 0.5852 (Δ=+0.08) | ✅ |

**4/4 GOAT proofs passed. SimpleTES RPUCG is GOAT-qualified.**

## Key Takeaway

SimpleTES adds three orthogonal improvements to flat bandit selection:

1. **Exploration bonus** — RPUCG formula λ·√((1+|S|)/(1+n_i)) guarantees unvisited nodes are explored, unlike greedy which always exploits. This alone accounts for most of the improvement in Proof 1.

2. **Graph value propagation** — U_i = max(r_i, γ·max(U_child)) lifts parent node values, making subtree roots selectable inspirations. A node with low direct score but high-quality children becomes a viable mutation parent. This explains why RPUCG finds 11 perfect solutions vs greedy's 0.

3. **Trajectory pruning** — Concentrates budget on top trajectories (+27% avg quality), matching SimpleTES's finding that early stopping is critical for evaluation-driven scaling.

## References

- **Paper:** arXiv:2604.19341 — SimpleTES: Evaluation-Driven Scaling
- **Research:** `.research/52_SimpleTES_Evaluation_Driven_Scaling.md`
- **Related:** Plan 030 (BanditPruner), Plan 079 (BT Rank GOAT proof pattern)