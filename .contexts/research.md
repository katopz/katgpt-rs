# Research Workflow — Freeze/Thaw + Latent-to-Latent Focus

> **Pivot (issue 004, 2026-06-14):** Training-method research moved to `riir-train`. This repo (`katgpt-rs`) and `riir-ai` now ship **freeze/thaw runtime + self-learn/adaptive NPCs + latent-space operations**. No LoRA training, no adapter fine-tuning, no optimizer research here. If a paper's value is its training loop, it belongs in `riir-train/.research`. If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, or a modelless inference primitive — distill it here.

## Research Focus (what to look for)

**Primary** (distill here):
- **Latent-to-latent operations** — anything that stays in embedding/latent space: dot-product projections, cosine similarity retrieval, sigmoid-gated routing, manifold geometry, spectral methods on activations. Prefer operating on latents over decoding to tokens then re-encoding.
- **Freeze/thaw patterns** — versioned weight snapshots, atomic hot-swap, lock-free read paths, BLAKE3/commitment-checked adapter reload, per-entity personality divergence via snapshot versioning.
- **Runtime adapter routing** — selecting between frozen adapters by state/objective/context (Dynamic Pair, Polytope, dMoE — all inference-time, zero training).
- **Self-learn / adaptive CoT** — runtime curiosity, entropy-driven exploration, collapse detection/recovery, latent prediction SSL, trajectory folding. No LLM training, no backprop through weights — but runtime self-improvement via latent-space updates is welcome.
- **Modelless inference primitives** — ConstraintPruners, bandits, DDTree, speculative decode, sparse attention, quantization-aware inference.
- **MMORPG-scale game AI** — thousands of concurrent NPCs each with independent latent state, real-time latency budgets (20Hz tick, plasma/hot tier), spatial partitioning + fog-of-war, emergent social/economic behavior (factions, trade routes, reputation), zone-level attention routing, crowd-scale curiosity/exploration signals. Latent ops must batch across many entities; raw sync must stay bit-identical for deterministic replay/anti-cheat.

**Redirect to riir-train** (do NOT distill here):
- LoRA/OFT/SPEFT/IA3/QLoRA/ManifoldE/BAKE/GPart/MSA/Dendritic and all adapter-**training** methods.
- Training optimizers (Muon, Adam variants, symmetry-compatible optimizers).
- Training loss functions, curricula, distillation recipes.
- Anything that requires backpropagation through base weights.

## Distillation Targets (3-repo strategy)

Per [`003_Commercial_Open_Source_Strategy_Verdict.md`](.research/003_Commercial_Open_Source_Strategy_Verdict.md):

| Repo | Role | What lands here |
|------|------|-----------------|
| `katgpt-rs` (public, MIT) | Engine — modelless inference framework | Generic primitives: ConstraintPruner traits, bandits, DDTree, speculative decode, sparse attention kernels. **No game IP, no chain IP.** |
| `riir-ai` (private) | Game product — freeze/thaw runtime, self-learn, chain | Runtime IP: `LoRAWeightVersion`, `LoRAHotSwap`, `dispatch_lora_merge`, `TrainingProvider` trait, routing, game systems, neuro-symbolic chain. |
| `riir-train` (private) | Training research vault | **Only if the paper's value is its training method.** Adapter training, optimizers, loss functions. (Out of scope for this workflow — just note "→ riir-train" and move on.) |

Distill into:
- **Modelless** → `katgpt-rs/.research/` + `katgpt-rs/src/` (or `crates/katgpt-rs-core/`)
- **Runtime/game/chain** → `riir-ai/.research/` + `riir-ai/crates/`
- **Training-only** → note the redirect, do not create files in this session

## Workflow

0. **Read the paper** (or PDF via `https://r.jina.ai/https://arxiv.org/pdf/{ID}`). Ask: *is the value in the training loop, or in a latent-space / inference / routing insight?* If training-only → note "→ riir-train", stop.

1. **Distill fundamentally** — don't direct-map the paper. Find the transferable primitive: the geometric, spectral, or information-theoretic insight that works without the paper's training setup. Grep `.research/` and `.plans/` for related prior work. Verdict GOAT/gain by the commercial strategy doc. Create research `.md` at the right repo (see table above).

2. **If gain, plan it** — add plan `.md` to `katgpt-rs/.plans/` (modelless) and/or `riir-ai/.plans/` (runtime/game/chain). Use `## Task` sections with `- [ ]` per task. **Never** plan into riir-train from this workflow.

3. **Implement to unblock** — if a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

4. **Search if curious** — keyword search arxiv:
```
https://r.jina.ai/https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={KEYWORD}&terms-0-field=abstract&classification-computer_science=y&classification-mathematics=y&classification-physics_archives=all&classification-statistics=y&classification-include_cross_list=include&date-filter_by=all_dates&size=50&order=-announced_date_first
```
Good keywords: `latent space routing`, `adapter hot-swap`, `inference-time composition`, `spectral pruning`, `sigmoid gating`, `snapshot consistency`, `lock-free weight swap`.

## Constraints

1. **Modelless first** — inference-time only. No LLM training, no backprop through base weights. Closest to "training" allowed: freeze/thaw snapshot cycles and latent-space direction-vector updates at runtime.
2. **Latent-to-latent preferred** — operate in embedding/latent space as long as possible. Decode to tokens or project to raw scalars only at the boundary. Use dot-product + **sigmoid** (never softmax) for projections onto learned direction vectors. Semantic domain (emotion, mood, curiosity, style) → latent. Physical domain (position, HP, wallet balance) → raw, deterministic, synced.
3. **Freeze/thaw over fine-tuning** — the only weight mutation allowed at runtime is swapping a frozen snapshot (atomic, versioned, BLAKE3-checked). Never mutate weights in-place during inference. If a paper needs gradient updates, redirect to riir-train.
4. **Self-learn / adaptive CoT welcome** — runtime curiosity, latent prediction, trajectory folding, collapse detection. These update latent state / direction vectors / routing tables, NOT base weights.
5. **3-repo discipline** — katgpt-rs (public engine) → riir-ai (private runtime/game/chain) → riir-train (private training). Keep the commercial strategy intact. Training know-how never leaks to katgpt-rs.
6. **SOLID, DRY** — per [`optimization.md`](.contexts/optimization.md). Zero-allocation hot paths. Pre-computed lookup tables. Fixed-size arrays for bounded domains.
7. **Tests/examples** — before/after showing the gain (latency, quality, or security). For latent ops: show the projection preserves ranking. For freeze/thaw: show readers never see torn snapshots.
8. **CPU/GPU/ANE auto-route** — threshold-adaptive dispatch. Plasma (µs, CPU/SIMD) → Hot (sub-ms, GPU) → Warm/Cold (ms+, GPU/ANE). Latent ops that fit in L1 cache stay on SIMD; manifold ops that need batched matmul go to GPU.
9. **Plasma → Hot → Warm → Cold → Freeze tiering** — aim for perf on game side (plasma/hot latency budget) AND security on chain side (cold/freeze commitment, BLAKE3-hashed, tamper-evident). Latent state that crosses the sync boundary MUST be raw scalars (valence/arousal/desperation/calm/fear), never the full embedding vector.

## Anti-patterns (redirect to riir-train, do not implement here)

- "Train a LoRA adapter to do X" → riir-train
- "Fine-tune with method Y" → riir-train
- "Optimizer Z improves convergence" → riir-train
- "Distillation recipe from teacher to student" → riir-train
- "Quantization-aware training" → riir-train (quantization-aware **inference** stays here)
- "DPO/GRPO/SFT/RL training pipeline" → riir-train (runtime GRPO self-play stays in riir-ai — it updates latent state, not weights)
