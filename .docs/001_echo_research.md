# ECHO Research Deep Dive: Environment Prediction as Auxiliary Loss

> Source: arXiv:2605.24517 — "ECHO: Terminal Agents Learn World Models for Free"
> Authors: Vaishnavi Shrivastava, Piero Kauffmann, Ahmed Awadallah, Dimitris Papailiopoulos (Microsoft Research)
> Date: 2026-05-23

---

## 1. Core Algorithm

### 1.1 The Loss Function

ECHO is a **hybrid objective** that combines two terms sharing a single forward pass:

```
L_ECHO(θ) = L_GRPO(θ; A) + λ · L_Env(θ; O')
```

Where:
- **A** = indices of assistant-action token positions in the rollout
- **O'** = indices of terminal-output (environment) token positions (excluding harness warning prefix)
- **L_GRPO** = standard clipped policy-gradient loss with group-normalized advantages (Eq. 2 in paper)
- **L_Env** = mean cross-entropy on environment observation tokens:

```
L_Env(θ; O') = -1/|O'| · Σ_{t ∈ O'} log p_θ(x_t | x_{1:t-1})
```

### 1.2 Key Mechanism

1. **Single forward pass**: The model already computes logits at every position for GRPO's action-token loss. ECHO simply **gathers the already-computed logits at environment-token positions** and adds their cross-entropy to the same backward pass.
2. **No extra rollouts, no teacher model, no architecture changes**.
3. **On-policy by construction**: Targets come from the current policy's own rollouts. As the agent improves and visits new terminal states, the environment produces new responses → **self-evolving curriculum**.
4. **Auto-annealing**: As the model learns terminal-output statistics, L_Env falls rapidly, reducing auxiliary contribution without explicit schedule.

### 1.3 What Gets Trained On

Rollout structure:
```
[sys] [task] [action₁] [obs₁] [action₂] [obs₂] ... [action_K] [obs_K]
```

- **GRPO**: trains only on `action` positions, driven by sparse binary outcome reward
- **ECHO**: additionally trains on `obs` positions (terminal output only, NOT harness warnings)

**Why exclude warnings?** Warning tokens (format violation messages) have near-zero entropy and get memorized in ~60 steps. Terminal-output tokens (file names, test failures, stack traces, byte counts) have irreducible entropy of 0.05–0.10 nats and provide sustained gradient throughout training.

### 1.4 Hyperparameters

| Parameter | Value |
|-----------|-------|
| λ (loss weight) | 0.05 (productive range: 0.01–0.05) |
| GRPO rollouts per prompt | n = 16 |
| Learning rate | 1e-6 (constant, no warmup/decay) |
| Gradient clip | 0.2 |
| Sampling temperature | 0.8 (train), 0.6 (eval) |
| Training steps | 500 GRPO steps |
| Hardware | 8× A100/B200 |

---

## 2. Key Results

### 2.1 Main Performance (TerminalBench-2.0)

| Model | GRPO pass@1 | ECHO pass@1 | Multiplier |
|-------|------------|-------------|------------|
| Qwen3-8B | 2.70% | 5.17% | ×1.9 |
| OT-SFT (8B) | 7.64% | 7.87% | ×1.03 |
| Qwen3-14B | 5.17% | 10.79% | ×2.1 |

### 2.2 Training Efficiency

- **8B**: ECHO reaches GRPO's peak performance in **1.5–2.3× fewer steps**
- **14B**: Both peak at same step, but ECHO reaches a **higher plateau**
- **Inference**: ECHO cuts TB2 timeouts from 19.8% → 9.0% (8B), reduces completion tokens by 30%

### 2.3 World Model Transfer

On **held-out off-policy trajectories from Qwen3-32B** (a model that didn't generate these trajectories):

| Model | val100 CE drop | ITD CE drop | TBLite CE drop |
|-------|---------------|-------------|----------------|
| Qwen3-8B | 0.29→0.07 | 0.46→0.32 | 0.35→0.25 |
| Qwen3-14B | 0.24→0.07 | 0.39→0.31 | 0.30→0.23 |

GRPO alone **barely changes** env-token cross-entropy. ECHO sharply lowers it across all slices.

### 2.4 Expert SFT Gap Recovery

ECHO from base Qwen3-8B recovers:
- **101.6%** of expert-SFT gap on val100
- **103.9%** on ITD
- **88.9%** on TBLite
- **~50%** on TerminalBench-2.0

*Without using any of the ~15k expert demonstrations.*

---

## 3. Ablations & What Worked vs. Didn't

### What Worked

| Factor | Details |
|--------|---------|
| λ = 0.01–0.05 | Productive range. Below 0.01: gradient too small. Above 0.1: competes with policy. λ=0.2: collapse into degenerate easy-to-predict rollouts. |
| Targeting env tokens only | Warning tokens memorized in ~60 steps, then provide zero gradient. Env tokens sustain gradient throughout training. |
| Self-annealing via constant λ | As model learns env statistics, L_Env drops → natural decay without scheduling. |
| Clean rollout filtering for verifier-free | Filtering to parseable tool calls is critical for OOD verifier-free adaptation. |

### What Didn't Work / Edge Cases

| Issue | Details |
|-------|---------|
| λ too high (0.1–0.2) | Policy quality plateaus or degrades. At 0.2, runs collapse into degenerate rollouts where terminal outputs are easy to predict but no longer useful. |
| OT-SFT on TBLite | GRPO peaked earlier than ECHO (0.73× speedup). SFT initialization already provides interaction prior, so ECHO adds less marginal value. |
| TBLite verifier-free adaptation | -3.9pp degradation. Suspected cause: TBLite requires broader shell orchestration over less visible state; observed tokens are less directly action-linked. |
| 14B internal gains smaller | Larger model can internalize more, but gains appear more on TB2 (harder benchmark) than internal evals. Policy and env-prediction objectives compete more at smaller scales. |

### Verifier-Free Adaptation (§5.5)

Starting from best ECHO checkpoint, mask GRPO, train only L_Env for 100 steps:

| Target | Δ pass rate | Filter |
|--------|-----------|--------|
| val100 (in-dist) | +3.8pp | none |
| PyTerm (OOD) | +10.0pp | clean tool calls |
| ITD (OOD) | +5.2pp | clean tool calls |
| TBLite (OOD) | -3.9pp | clean tool calls |

**Key insight**: Verifier-free env-only adaptation works best when clean exploration exposes **predictive, action-linked feedback** (e.g., Python tracebacks). Fails when terminal output is weakly coupled to action quality (e.g., filesystem orchestration).

---

## 4. Related Work Taxonomy

### 4.1 Closest Cousins (Training-Time Auxiliary Prediction)

| Paper | Key Idea | Relation to ECHO |
|-------|----------|-----------------|
| **CWM** (FAIR CodeGen, 2025) arXiv:2510.02387 | 32B LLM trained on code world modeling data (Python/Docker traces). Predicts execution outcomes. | Trains a separate world model on a large offline corpus. ECHO is on-policy, in-line, no separate stage. |
| **PaW** arXiv:2606.02388 | "Policy and World Modeling Co-Training" — reuses RL rollouts by appending next-observation tokens with auxiliary next-token-prediction loss. | **Nearly identical idea** to ECHO. Key difference: appears to be concurrent/independent work. ECHO provides deeper ablations and the verifier-free result. |
| **RLTF** (Song et al., 2026) arXiv:2602.02482 | Trains model to predict judge-generated critiques as auxiliary loss. Dense text feedback from a feedback provider. | Predicts **judge critiques** (generated by external model), not raw environment output. Requires a judge/teacher. ECHO predicts raw env tokens directly. |
| **OpenClaw-RL** (Wang et al., 2026) arXiv:2603.10165 | Recovers next-state signals as scalar process rewards via judge, or token-level distillation via judge-extracted hints. On-policy distillation. | Uses a **judge** to extract signal from next state. ECHO uses raw environment tokens with no judge. Complementary approaches. |
| **Self-Distillation** (Hübotter et al., 2026) arXiv:2601.20802 | Reinforcement learning via self-distillation of agent experience. | Related in using agent's own experience, but different mechanism. |

### 4.2 Classical Auxiliary Prediction in RL

| Paper | Key Idea |
|-------|----------|
| **UNREAL** (Jaderberg et al., 2017) | Auxiliary tasks (pixel control, reward prediction) improve representations in model-free RL. |
| **Curiosity-driven** (Pathak et al., 2017) | Forward dynamics prediction as intrinsic reward. |
| **SPR** (Schwarzer et al., 2021) | Self-predictive representations for data-efficient RL. |
| **Future prediction** (Kwon et al., 2024) | Power of future prediction in partially observable environments. |

ECHO follows this lineage but applies it to **multi-turn LM-agent setting** where targets are textual observations already in the rollout.

### 4.3 World Models for Planning

| Paper | Key Idea |
|-------|----------|
| **Dreamer** series (Hafner et al., 2020–2025) | Learn latent world models for imagination-based planning. Nature 2025: Mastering diverse control tasks. |
| **MuZero** (Schrittwieser et al., 2020) | Planning with learned model without knowledge of environment dynamics. |
| **World Action Models** (Ye et al., 2026) arXiv:2602.15922 | "World action models are zero-shot policies" — embodied agents. |
| **Dual Preference Optimization** (Wang et al., 2025) | World modeling for embodied task planning via dual preference optimization. |

### 4.4 Inference-Time / Training-Free Approaches

**This is the key gap area.** Most world-model work operates at training time. Inference-time uses are limited:

| Approach | What It Does | Training Required? |
|----------|-------------|-------------------|
| **Speculative Decoding** (EAGLE, Medusa, etc.) | Draft model predicts future tokens → verify with target model. Uses feature-level autoregression. | Yes — draft model trained separately |
| **EAGLE** (2024–2025) | Uses **feature-level** (hidden state) autoregression, not token-level. Drafts tree-structured candidates. | Yes — lightweight head trained on features |
| **MCTS / Tree Search at inference** | Use model's own probability estimates to search at inference time (e.g., AlphaCode, QwQ, DeepThink). | No extra training, but requires compute budget |
| **Process Reward Models at inference** | Score intermediate steps during generation to guide search. | Yes — PRM must be trained |
| **Self-consistency** | Sample multiple completions, select by majority vote. | No training |
| **Verifier-free ECHO** (§5.5) | Continue training on env-prediction only, no reward signal. | Still updates weights (training, not inference) |

**Critical observation**: There is **no known inference-time-only technique** that directly applies ECHO-style "predict environment to improve policy" without weight updates. This is a research gap.

---

## 5. Research Gap: Inference-Time World Model for Policy Improvement

### The Opportunity

ECHO proves that environment prediction capability **correlates with policy quality**. The key question is:

> Can we use a model's internal environment-prediction capability at **inference time** (no weight updates) to improve action selection?

### Potential Approaches (untested/speculative)

1. **Implicit world-model scoring**: At each action step, sample multiple candidate actions. For each, use the model's own prediction of environment response as a **scoring signal** (higher likelihood of "good" env tokens = better action). No training, just beam-search-like scoring.

2. **Verification via environment simulation**: Before committing an action, the model "imagines" the terminal output. If the imagined output contains error patterns (exit code ≠ 0, stack traces), reject/modify the action. This is speculative decoding applied to environment tokens.

3. **Contrastive action selection**: Given multiple action candidates, compute p(action | context) × p(env_response | action, context). The joint probability picks actions that the model can both generate and predict consequences for. Related to "policy-shaped prediction" from Stanford HAI.

4. **Lookahead with world model head**: If the model has been trained with ECHO-style auxiliary loss, its env-prediction head can be repurposed at inference for **Monte Carlo Tree Search**-style rollouts: action → predicted env → next action → predicted env → ... → evaluate terminal state.

5. **Self-play with environment model**: At inference, generate K actions, for each predict the env response, then continue the trajectory in "imagined" mode. Score the imagined trajectories by whether they reach a completion signal.

### Why This Hasn't Been Done

- ECHO-style auxiliary loss is very new (May 2026)
- Most inference-time work focuses on **mathematical reasoning** (MCTS, PRM) not **interactive environments**
- Speculative decoding research focuses on **latency**, not **policy quality**
- The connection between "model can predict environment" and "model makes better decisions" was only recently formalized by ECHO

---

## 6. Summary & Implications for Our Projects

### ECHO's One-Liner
> "Environment observations are not merely context for future actions, but a dense, on-policy supervision signal already present in every rollout."

### Key Takeaways

1. **Dead simple to implement**: Change loss mask, add cross-entropy on env tokens. No architecture changes.
2. **2× pass@1 on hard benchmark** with no extra rollouts or compute.
3. **Learns transferable world model**: Off-policy evaluation shows the model genuinely understands terminal dynamics.
4. **Replaces ~half of expert SFT value**: The "interaction prior" (how terminals respond) can be learned from interaction.
5. **Verifier-free self-improvement is possible** but only when env feedback is dense and action-linked.

### Relevance to katgpt-rs / riir-ai

- The "predict environment to improve policy" paradigm maps directly to our RL training pipeline
- The verifier-free adaptation result suggests potential for **online self-improvement without reward models**
- The inference-time gap is the most interesting research direction: can we use a model's world-model capability at inference without retraining?

### TL;DR

ECHO adds cross-entropy loss on environment observation tokens to GRPO's action-token loss, sharing one forward pass. It doubles pass@1 on TerminalBench-2.0, learns transferable terminal dynamics (verified on off-policy trajectories), replaces ~50% of expert-SFT value, and enables verifier-free self-improvement. The inference-time-only application of world-model capabilities remains an open research gap — no existing work applies this without weight updates.
