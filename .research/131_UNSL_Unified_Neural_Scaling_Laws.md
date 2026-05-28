# Research 131: Unified Neural Scaling Laws (UNSL)

**Paper:** [arXiv:2605.26248](https://arxiv.org/abs/2605.26248) — Caballero, Jaini, Krueger, Rish (Mila/Google DeepMind, 2026)
**Date:** 2026-05-28
**Verdict:** ⚠️ Marginal — Research-only, no implementation plan

---

## Summary

Presents a functional form (UNSL) that accurately models and extrapolates scaling behaviors of deep neural networks as multiple dimensions vary simultaneously:
- Number of model parameters
- Training dataset size
- Number of training steps
- Number of inference steps
- Hyperparameters (learning rate, weight init std, etc.)

Key innovations:
1. **Multivariate Broken Neural Scaling Law (MBNSL):** Sequence of smoothly connected hyperplanes in multi-log space, each "hyperbreak" representing a phase transition
2. **Bottleneck/non-bottleneck decomposition:** Separates performance limits caused by each input dimension being the bottleneck
3. **Oppositional forces:** Models overfitting and nonmonotonic hyperparameter effects as additive forces opposing good learning
4. **Compute-optimal allocation:** Given a fitted UNSL, solve a Lagrange multiplier system to find optimal param/data/steps split for a given compute budget

Results: UNSL beats CF (Chinchilla), DC (Muennighoff), and ablation baselines A1/A2/A3 on 60.87% of vision tasks and 88.89% of language tasks for extrapolation accuracy.

## Distillation for katgpt-rs / riir-ai

### Model-Based (riir-ai domain)

| Aspect | Relevance | Notes |
|--------|-----------|-------|
| LoRA training budget optimization | ⬜ Indirect | UNSL could theoretically tell us optimal rank/steps/data split for LoRA training — but requires running many LoRA experiments at varying scales first. We don't have this infrastructure. |
| wGPU LoRA hyperparameter prediction | ⬜ Indirect | The "oppositional force of hyperparameters" model could predict optimal LR/init-std for LoRA training — but again needs curve-fitting data we don't collect |
| Inference-time scaling prediction | ⬜ Indirect | Appendix 18.2 shows UNSL extrapolates inference (test-time) scaling. Could inform our test-time compute allocation |

### Modelless (katgpt-rs domain)

| Aspect | Relevance | Notes |
|--------|-----------|-------|
| Dynamic budget (Plan 026) | ⬜ Indirect | UNSL's compute-optimal allocation (Appendix 12) resembles our β-parameterized inference budgets, but UNSL requires a fitted model we don't have |
| SR²AM Bandit (Plan 112) | ⬜ Indirect | The concept of "phase transitions" (hyperbreaks) could inform when the bandit should switch strategies |
| Early exit patience | ⬜ Indirect | Overfitting detection insight from UNSL's "oppositional force" decomposition is conceptually related to our early exit gap |
| SpecHop (Plan 131) | ❌ | No direct connection |
| Heuristic learning (Plan 032) | ⬜ Indirect | The broken scaling law pattern (piecewise power laws) could inspire more expressive heuristic parametrizations |

### GOAT Pillar Assessment (per `27_mmo_goat_pillars_decision_matrix.md`)

| Criterion | Score | Reason |
|-----------|-------|--------|
| GOAT passed | ❌ | No measurable proof possible — UNSL is a meta-analysis tool, not a component |
| MMO-product | ❌ | Does not directly contribute to MMORPG server/player experience |
| LoRA-independent | ✅ | The theory is model-agnostic, but useless without training data at scale |
| Defensible | ❌ | Published paper — not secret knowledge |
| Secret coverage | ❌ | No coverage of A, A2, B, C, or D |

**NOT a GOAT pillar candidate.** Not a cross-cutting improvement. Pure academic reference.

## Key Insight Worth Noting

The "broken scaling law" pattern (piecewise power laws with smooth transitions) is a useful mental model:

1. **Phase transitions exist in scaling:** Small models may show one scaling exponent that shifts at some critical size. Our heuristic learning infrastructure (Plan 032) should account for this — the optimal heuristic for a 1B model may not be optimal for a 7B model.

2. **Nonmonotonic hyperparameters:** LR and init-std have an "oppositional force" — too high hurts, and the damage follows a predictable curve. This validates our entropy anomaly detection (Plan 061) approach of detecting when training goes wrong.

3. **Extrapolation requires proximity to hyperbreaks:** Section 5 shows you need observations close to each transition to predict beyond it. For our Bandit (Plan 112), this means we need to explore near decision boundaries to predict performance on the other side.

4. **Compute-optimal split is derivable:** Given enough data points, you can solve for optimal param/data/steps allocation. This could inform our domain inference budgets (Plan 026) if we ever collect enough training metrics.

## Why No Implementation Plan

1. **Requires JAX/KFAC infrastructure** — the paper's fitting procedure needs 20K optimizer steps × 20 seeds. We don't have this in Rust.
2. **Requires multi-scale training data** — we don't run experiments at 5+ different scales for any model.
3. **Marginal perf gain** — the insights are conceptual, not implementable as code that improves our inference or game AI pipeline.
4. **Not game-specific** — this is general ML scaling theory, not riir-ai domain knowledge.
5. **No feature gate warranted** — nothing to gate, nothing to prove via GOAT.

## Reference

```
@article{caballero2026unsl,
  title={Unified Neural Scaling Laws},
  author={Caballero, Ethan and Jaini, Priyank and Krueger, David and Rish, Irina},
  journal={arXiv:2605.26248},
  year={2026}
}
```

## Cross-References

- katgpt-rs Plan 026 (Domain Inference Budget) — β-parameterization is simpler version of compute-optimal allocation
- katgpt-rs Plan 112 (SR²AM Bandit) — "phase transition" concept informs strategy switching
- katgpt-rs Plan 061 (Entropy Anomaly Detection) — validates oppositional force concept
- katgpt-rs Research 052 (SimpleTES) — related evaluation-driven scaling work
- riir-ai Plan 026 (Domain Inference Budget) — TOML-configured β budgets
