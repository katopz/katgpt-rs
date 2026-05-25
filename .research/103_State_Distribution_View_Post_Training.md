# Research 103: State Distribution View of Post-Training (SFT, RL, OPD)

**Paper:** "Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation" (Nie, 2026, arXiv:2605.22731)
**Date:** 2026-05-25
**Verdict:** ✅ **ADOPT — Modelless OPD distillation, State-Source Taxonomy, Continuation-Based Scoring**

---

## Paper Summary

The paper reframes post-training through a **state-distribution lens** rather than the traditional objective-only view. A "state" = prompt + generated prefix. Three key findings:

1. **SFT can be gentle or destructive** — Mild SFT improves GSM8K (+6.4%) with ~0 forgetting. Stress SFT degrades both target (-2.8%) and retention (-17.4%).
2. **OPD can surpass a degraded teacher** — Student trained on student-sampled states with teacher supervision outperforms the teacher on ALL metrics (GSM8K +4.6pts, TruthfulQA +3pts, MMLU +6.6pts).
3. **Scalar drift is insufficient** — Stress SFT and OPD from stress teacher have identical MMD drift (0.01093 vs 0.01092) but wildly different retention (0.83 vs 0.95). *Where* updates are applied matters more than *how far* the distribution moves.

### Core Framework

Two axes decompose any post-training method:

| Method | State Source | Signal Source |
|--------|-------------|---------------|
| SFT | Dataset trajectories (off-policy) | Gold tokens |
| Offline KD | Teacher trajectories | Teacher logits/tokens |
| **OPD** | **Student trajectories (on-policy)** | **Teacher continuations** |
| RL | Current policy (on-policy) | Reward |
| DAgger | Learner trajectories | Expert actions |

### Key Insight: Continuation-Based OPD

One-step next-token KL OPD **collapsed** (GSM8K → 0.040). Continuation-based OPD — teacher generates short rollouts from student states — recovered target performance. The paper's practical recipe: **combine on-policy sampling with dense, trajectory-level local supervision**.

---

## Distillation to Our Architecture

### 1. State-Source Taxonomy → Modelless Distillation Stack

Our existing modelless distillation patterns map cleanly:

| Our Component | Paper's Axis | Analogy |
|---------------|-------------|---------|
| `AbsorbCompress` | Dataset-state learning (SFT-like) | Promotes heuristics from observed trajectories |
| `BanditPruner` δ-reward | On-policy reward (RL-like) | Updates Q-values on learner-visited states |
| `DeltaBanditPruner` (G-Zero Phase 1) | On-policy + hint signal | δ measured on learner states = on-policy dense signal |
| VPD E-step/M-step | Teacher/student co-evolution | BCO refines teacher, KL-gated distillation to student |
| SDAR sigmoid gate | Signal gating (not state-source) | Modulates reward intensity, not state distribution |

**Gap identified:** We have no modelless component that does **OPD-style state-source separation** — student controls states, teacher provides local guidance. This is architecturally different from VPD (which co-evolves teacher+student) and SDAR (which only gates signal intensity).

### 2. On-Policy State Visitation → BanditPruner Enhancement

The paper proves that **where updates are applied** matters more than signal magnitude. Our `BanditPruner` already uses on-policy states (Q-values updated from learner's own rollouts), but we don't track **state visitation coverage** — whether the bandit has explored diverse enough prefix states.

**Distillation:** Add state-visitation entropy tracking to `BanditPruner`. When coverage drops below threshold, boost exploration. This is the modelless analogue of the paper's "on-policy locality preserves capabilities" finding.

### 3. Continuation-Based Scoring → DDTree Enhancement

The paper found one-step KL collapsed but continuation-based OPD worked. Our `DDTree` already builds multi-token continuations (beams). We could score beams not just by log-prob but by **teacher-supplied continuation quality** — but this requires a model-based teacher.

**For modelless path:** Replace "teacher continuation" with "validator continuation" — extend `ConstraintPruner` to not just filter invalid tokens but provide short validated continuations from student states. The WASM validator already does this for games (valid next-move sequences). The insight is: **use the validator as a continuation source, queried on student-sampled states**.

### 4. OPD from Degraded Teacher → Self-Play Resilience

The paper's most striking result: student trained on its own states + degraded teacher guidance **surpasses** the teacher. This directly validates our G-Zero Phase 1 design (modelless first) and suggests:

- Even with a poor LoRA teacher (our Secret A risk from Decision Matrix), on-policy student sampling + WASM validator guidance can produce good behavior
- This strengthens the "game IP as secondary moat" argument: WASM validators (Secret A2) provide local guidance on student states, independent of LoRA quality

### 5. State Drift vs. Capability Retention → GOAT Metric Enhancement

The paper shows MMD drift alone doesn't predict forgetting. Our GOAT proofs measure throughput, accuracy, and overhead but not **state-distribution coverage**. Adding a "retention" dimension to GOAT proofs (e.g., "modelless method X preserves Y% of baseline action diversity") would catch the pattern the paper identifies.

---

## What NOT to Distill

| Paper Idea | Why Not |
|-----------|---------|
| Full OPD training loop (sampling + teacher forward pass) | Model-based — requires differentiable teacher. Lives in riir-ai domain if pursued |
| MMD drift metric | Our domains are game states, not text. Game-specific divergence metrics (action distribution KL, position entropy) are more appropriate |
| SFT stress/forgetting experiments | We don't do gradient-based SFT in katgpt-rs. That's riir-ai/riir-gpu domain |
| Qwen3-0.6B experimental setup | Specific model/hardware. Our benchmarks use our own micro transformer + game arenas |

---

## Applicable Distillations Summary

| # | Distillation | Target | Modelless? | Domain |
|---|-------------|--------|-----------|--------|
| D1 | State-source taxonomy alignment | Map our components to OPD/SFT/RL axes | ✅ | katgpt-rs research |
| D2 | State-visitation entropy for BanditPruner | Track prefix-state coverage, boost exploration when low | ✅ | katgpt-rs `.pruners` |
| D3 | Validator-as-continuation-source | WASM validator provides short valid continuations from student states (OPD analogue) | ✅ | katgpt-rs + riir-ai |
| D4 | GOAT retention dimension | Add state-diversity preservation to GOAT proof checklist | ✅ | katgpt-rs `.benchmarks` |
| D5 | OPD from degraded teacher → LoRA resilience argument | Strengthens Decision Matrix Pillar 2 (WASM validators provide local guidance regardless of LoRA quality) | ✅ | riir-ai strategic |

---

## Relationship to Existing Work

| Existing | Relationship |
|----------|-------------|
| Plan 049 G-Zero | Phase 1 (Hint-δ) is RL-like on-policy. This paper validates the modelless-first design |
| Plan 052 GFlowNet | Backward replay (D4) walks winning replays backward — that's dataset-state, off-policy. Paper suggests adding on-policy state exploration |
| Plan 071 ROPD Rubric | Multi-criterion scoring is signal-source innovation. This paper suggests pairing it with state-source awareness |
| Plan 072 SDAR | Sigmoid gating is signal gating only. Paper predicts this won't change action distributions (confirmed by arena negative result) |
| Plan 111 Data Gate | Task-level filtering before solver is dataset-state curation. Paper would classify this as "improving state quality" — aligned |
| Plan 120 VPD | Co-evolutionary teacher-student is closest to OPD but adds BCO teacher training. Paper suggests the KL-gated M-step is the key, not teacher quality |
| Research 037 REAP | Our model-based/modelless duality framework. This paper provides theoretical backing for why modelless (on-policy) can match model-based |

---

## References

- Nie, D. (2026). "Post-Training is About States, Not Tokens." arXiv:2605.22731
- Ross, S. et al. (2011). DAgger. AISTATS. (Paper's closest theoretical ancestor)
- Bengio, S. et al. (2015). Scheduled sampling. NeurIPS. (Exposure bias)
- Kim, Y. & Rush, A. (2016). Sequence-level KD. EMNLP. (Offline distillation baseline)
