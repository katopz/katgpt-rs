# Research 080: VPD — Variational Policy Distillation

**Paper:** [Learning from Language Feedback via Variational Policy Distillation](https://arxiv.org/abs/2605.15113)
**Authors:** Yang Li, Erik Nijkamp, Semih Yavuz, Shafiq Joty (Salesforce AI Research)
**Date:** 2026-05-20
**Reviewed:** 2026-05-24

---

## Summary

VPD reframes on-policy self-distillation from language feedback as a **Variational Expectation-Maximization (EM)** problem. Instead of treating the feedback-conditioned "self-teacher" as a fixed heuristic (as in SDPO), VPD **co-evolves** both teacher and student policies:

- **E-step:** Actively train the teacher to distinguish successful vs failed trajectories given diagnostic feedback (via unpaired BCO preference optimization)
- **M-step:** Distill the refined teacher back into the student via token-level KL on on-policy rollouts

Key insight: **Dynamic Reference Prior** — anchor the E-step to the *current student* (π_θ), not a frozen π_ref. This creates a sliding trust region that prevents distribution shift between teacher and student.

---

## Core Mechanics

### 1. Variational EM Formulation

The optimal policy under KL-regularized RLVR is the reward-tilted distribution:
```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
```

Since Z(x) is intractable, VPD introduces a teacher q_φ(y|x,C) conditioned on diagnostic feedback C as an approximate posterior. The ELBO naturally decomposes into EM:

- **E-step:** min_φ D_KL(q_φ || π*) — teacher refinement
- **M-step:** min_θ D_KL(π_θ || q_φ) — student distillation

Both share the **same network weights** (φ = θ), distinguished only by conditioning on feedback C.

### 2. E-step: Unpaired Preference Optimization (BCO)

Since each trajectory has unique feedback C, you can't construct paired preference comparisons. VPD uses **Binary Classifier Optimization** (BCO) which decouples DPO into independent positive/negative terms:

```
L_E-step = -E_{y+}[log σ(r̃(x,y+,C+) - δ)] - E_{y-}[log σ(-(r̃(x,y-,C-) - δ))]
```

Where `r̃ = β · log(q_φ / π_θ)` is the implicit reward with **dynamic prior** π_θ (not π_ref).

### 3. M-step: Token-level KL Distillation

```
L_M-step = E_{x,y~π_θ}[Σ_t D_KL(π_θ(·|x,y_{<t}) || sg[q_φ(·|x,C,y_{<t})])]
```

Standard on-policy KL distillation with stop-gradient on the teacher.

### 4. Dynamic Reference Prior (Critical Design)

Setting π_ref ← π_θ in the E-step creates a **sliding trust region**:
```
max_φ E_{y~q_φ}[r(x,y)] - β · D_KL(q_φ || π_θ)
```

This is equivalent to TRPO/PPO-style trust region but applied to the teacher. It ensures the teacher's targets remain reachable by the student, preventing the collapse that plagues fixed-prior methods.

---

## Key Results

| Setting | GRPO | SDPO | VPD (Ours) |
|---------|------|------|------------|
| LiveCodeBench v6 (Qwen3-8B) | 45.61 | 47.33 | **49.62** |
| SciKnowEval AVG (Qwen3-8B) | 73.11 | 74.44 | **77.15** |
| SciKnowEval AVG (Qwen3-1.7B) | 69.81 | 66.34 | **74.34** |
| SciKnowEval AVG (OLMo3-7B) | 65.71 | 66.07 | **70.80** |
| Math500 (Qwen3-8B) | **83.8** | collapse | delayed collapse |
| Cold-start (Qwen3-4B-Base) | **74.49** | collapse | 63.95 |

### Ablation Highlights

- **E-step frequency F=5** is optimal (F=1 too volatile, F=10 too stale)
- **Dynamic prior >> Fixed prior** (74.34 vs 67.84 on SciKnowEval 1.7B)
- **Self-critique** (no positive pairs) still works: 78.14 vs SDPO's 74.87 on Qwen3-8B
- **30-55% runtime overhead** from E-step, but shared-weight = zero extra memory

---

## Distillation to Our Architecture

### What We Already Have (Mapping)

| VPD Concept | Our Existing System |
|-------------|---------------------|
| Feedback-conditioned teacher | `sdar_gate` sigmoid gating (Plan 072) |
| Pairwise ranking | `bt_rank` Bradley-Terry (Plan 079) |
| On-policy rollout + reward | `g_zero` self-play loop (Plan 049) |
| KL distillation | `ropd_rubric` rubric distillation (Plan 071) |
| Trust region | `data_gate` stability gating (Plan 111) |
| Sparse RL baseline | `bandit` UCB1 + GRPO-style advantages |
| Environment feedback | `validator` constraint checking |
| Multi-source feedback | `sr2am_configurator` adaptive planning (Plan 112) |

### New Ideas from VPD Applicable to Us

#### 1. **EM-style Co-evolution** (Novel for modelless)
Our current modelless distillation (`sdar_gate`, `ropd_rubric`) treats the self-teacher as a **fixed passive function** — exactly the limitation VPD identifies. We could implement an alternating E/M cycle:

- **E-step:** Update the "teacher" bandit/pruner weights using BCO-style unpaired preference on feedback-conditioned trajectories
- **M-step:** Distill back to the "student" policy via KL-gated absorb-compress

This would make our `SdarBanditPruner` and `SdarGatedAbsorbCompress` into active co-evolvers rather than passive signal processors.

#### 2. **Dynamic Reference Prior** (Novel for modelless)
Our `sdar_gate` and `data_gate` use static thresholds. VPD's insight: **anchor to the current student, not a frozen baseline**. For our modelless bandit:
- Replace fixed `π_ref` references in BCO/DPO loss with the current bandit Q-values
- Create a sliding KL penalty: `D_KL(q_teacher || q_student_current)` instead of `D_KL(q_teacher || q_baseline)`

#### 3. **BCO Unpaired Preference** (Novel — extends `bt_rank`)
Our `bt_rank` requires paired comparisons (winner vs loser from same context). VPD's BCO approach works with **unpaired** positive/negative samples. This is critical when:
- All rollouts for a prompt fail (no positive sibling)
- Feedback is generated per-trajectory (no shared context for pairing)

Could extend `BtComparison` with `BcoSample { reward: f32, feedback: Vec<f32> }` for unpaired optimization.

#### 4. **Asymmetric Update Frequency** (Novel for modelless)
VPD shows F=5 (1 E-step per 5 M-steps) is optimal — like a target network in RL. Our modelless loop could:
- Run absorb-compress (M-step) 5× per self-play round
- Only update the "teacher" bandit weights (E-step) every 5th round
- This stabilizes the teacher's target distribution

#### 5. **Three Feedback Sources** (We partially support)
VPD validates three feedback mechanisms, all applicable to our game domains:
1. **Environment feedback** → compiler errors, assertion failures → our `validator` constraint system
2. **Contrastive sibling rollouts** → successful trajectory as hint → our `g_zero` hint-δ mechanism
3. **Self-critique** → model judges its own failures → our `memo_reflections` pipeline

### What Doesn't Apply / Limitations

1. **Math reasoning collapse** — VPD confirms pure RL (GRPO) wins for strict logical domains. Our game domains are less susceptible (non-binary, multi-objective rewards), but for Go endgame correctness, sparse RL may still dominate.

2. **Cold-start problem** — VPD confirms self-distillation needs instruction-following capability. Our modelless bandit doesn't have this issue (no LLM backbone), but any future model-based integration would.

3. **Token-level KL** — VPD operates on token distributions. Our modelless operates on **action distributions** (discrete action space). The KL divergence still applies but at the action level, not token level.

4. **Shared-weight architecture** — VPD's key efficiency win (φ = θ) is trivially satisfied in our modelless setting since teacher and student ARE the same bandit, just with different conditioning (feedback-augmented vs vanilla).

---

## Verdict

**Actionable — Feature Gate Recommended (`vpd_em_distill`)**

VPD's core contribution — **co-evolutionary EM with dynamic trust region** — directly addresses a real limitation in our modelless distillation pipeline. Currently, our `sdar_gate` and `ropd_rubric` treat the feedback-conditioned "teacher" as passive, which VPD proves leads to plateau and eventual collapse. The fix is well-scoped:

1. Add an E-step phase to the self-play loop (BCO-style unpaired preference on feedback-conditioned bandit)
2. Replace static reference priors with dynamic (current student) anchoring
3. Implement asymmetric F=5 update frequency

**Expected impact:** Better signal quality from feedback, delayed training collapse, more stable convergence — validated across 3 model families and 7 benchmarks in the paper.

**Risk:** Low. The EM framework is a drop-in extension of existing `sdar_gate` + `bt_rank` infrastructure. Can be feature-gated and A/B tested against current passive distillation.

**Priority:** Medium. Our current modelless already works well for game domains. VPD's gains are most impactful for domains where feedback quality matters (code generation, scientific reasoning) — less critical for our current game arenas but valuable for future LLM-based inference paths.

---

## Key Equations Reference

```
// E-step: Teacher refinement (BCO unpaired preference)
// L_E = -E_{y+}[log σ(β·log(q_φ/π_θ) - δ)] - E_{y-}[log σ(-(β·log(q_φ/π_θ) - δ))]

// M-step: Student distillation (token/action-level KL)
// L_M = E_{x,y~π_θ}[Σ_t D_KL(π_θ(·|x,y_{<t}) || sg[q_φ(·|x,C,y_{<t})])]

// Dynamic trust region (sliding KL penalty)
// max_φ E_{y~q_φ}[r(x,y)] - β·D_KL(q_φ || π_θ)

// Reward shift (BCO stability)
// δ = 0.5 · (E[r̃(y+)] + E[r̃(y-)])
```

---

## Related Research in Our Stack

| Research | Connection |
|----------|------------|
| 038 SDAR | VPD extends SDAR's self-distillation with active teacher refinement |
| 040 OpenDeepThink BT | VPD uses BT-style preference but via BCO (unpaired) |
| 037 REAP Model Duality | VPD is a concrete instantiation of model-based/modelless co-evolution |
| 075 Survive Or Collapse | VPD's dynamic prior addresses the data gate stability problem |
| 076 SR²AM | VPD's asymmetric frequency relates to SR²AM's adaptive planning |
| 055 Nemotron TriMode | VPD's EM is orthogonal to tri-mode inference but complementary for training |
| 060 MeMo | VPD's self-critique feedback source aligns with MeMo reflection pipeline |