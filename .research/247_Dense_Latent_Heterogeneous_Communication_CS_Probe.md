# Research 247: Dense Latent Communication Across Heterogeneous Agents — CS Head Importance Probe

> **Source:** [See What I See, Know What I Think: Dense Latent Communication Across Heterogeneous Agents](https://arxiv.org/pdf/2606.13594) — Chen, Zhang, Wu, Tremblay, Blukis, Birchfield, Vidal, Velasquez, Liu, Qu (UMich + NVIDIA + UPenn + CU Boulder + MSU), Jun 2026
> **Date:** 2026-06-16
> **Status:** Active
> **Classification:** Public (katgpt-rs/MIT) — open primitive only. Game-side selling point → riir-ai/.research/133.
> **Related Research:** 109 (Shard RoPE-strip), 213 (Still Perceiver un-rotate), 238 (MUX superposition), 242 (HLA belief state), 070 (SP-KV per-head gate), 139 (EGA spectral salience), 153 (Thinking Pixel pruner routing)
> **Related Plans:** 147 (ShardKV), 245 (StillKV), 238 (MUX-Latent), 070 (SP-KV)
> **Cross-ref (riir-ai):** Research 133 (NPC mind-reading guide), Plan 311 (runtime)
> **Verdict: Super-GOAT — fusion creates a new capability class (adaptive-bandwidth NPC latent communication). See §3 + riir-ai/.research/133 for the private selling-point guide.**

---

## TL;DR

The paper's headline mechanism (cross-model KV-cache adapter, trained via Phase-I reconstruction + Phase-II generation) is **training-only → riir-train**. But two transferable insights survive the modelless filter and neither exists in our codebase today:

1. **Compressed-sensing (CS) head-importance probe** — a training-free, post-hoc ablation+Lasso that ranks which KV groups actually carry task signal. Our `grep` for `head_importance|cs_lasso|ablation_mask` returns **zero** matches across all three repos. The closest cousins (RTPurbo retrieval heads, EGA spectral salience) identify important heads *at training time*; this paper's contribution is a **runtime, post-hoc, per-task** probe.
2. **The sparse-reasoning vs dense-knowledge duality** — when the receiver has its own context (context-aware), only ~10/288 KV groups (3.5%) carry the signal; when the receiver has no context (context-unaware), ~250/288 (87%) are required. This is a **~25× swing in required bandwidth depending on receiver-side context**, and no existing note or code in our repos treats it as a routing/budget axis.

**Distilled for katgpt-rs (modelless, inference-time):**
The CS-Lasso probe is pure inference: sample `M≈200` random binary ablation masks over `H` heads, measure task accuracy under each mask, solve a Lasso, aggregate per-KV-group. The output is a **fixed-size ranking vector** — a `ConstraintPruner`-style artifact that gates which KV groups transmit, with **sigmoid**-gated density (never softmax, per AGENTS.md). The sparse/dense duality becomes a single scalar `context_awareness ∈ [0,1]` that interpolates the K-budget between the sparse floor (~3.5%) and the dense ceiling (~87%).

**Already shipped (NOT reinvented — do not overclaim novelty on these):**
- Position-disentanglement (RoPE strip → transform → restore): `katgpt-rs/crates/katgpt-kv/src/shard_kv/rope.rs` (`undo_rope`/`reapply_rope`, Plan 147, GOAT-proved) AND `riir-ai/crates/riir-engine/src/lora_still/mod.rs` (Still Perceiver, Plan 213).
- Per-KV-group sigmoid gating: `katgpt-rs/crates/katgpt-kv/src/sp_kv/utility_predictor.rs` (`soft_gate_bias`, Plan 070) and `ega_attn.rs` (Plan 139).
- Layer-monotonic alignment: trivial; our GQA already maps Q-head → KV-group via `kv_group = q_head * n_kv_head / n_head`.

**What's genuinely new (and drives the Super-GOAT verdict):** the fusion of {CS-probe} × {sparse/dense duality as a routing axis} × {existing HLA per-NPC belief state} × {fog-of-war context-availability signal} → **adaptive-bandwidth NPC mind-reading**. See §2.Fusion and riir-ai/.research/133 for the selling point.

---

## 1. Paper Core Findings

### 1.1 The two communication regimes (the central conceptual contribution)

| Regime | Receiver sees input? | Channel role | Required KV density (Qwen3-4B, 288 KV groups) |
|--------|---------------------|--------------|----------------------------------------------|
| **Context-aware** | Yes (`X_R = X`) | Sparse reasoning steer on top of receiver's own context | ~10/288 (3.5%) reaches ceiling; CS-ranking beats random by +27pp at K=50 |
| **Context-unaware** | No (`X_R = ∅`) | Channel IS the only task signal — must carry both context AND reasoning | ~250/288 (87%); stays at chance until K>150, sharp phase transition K=150→200 |

**Headline number:** a ~25× swing in required channel density, gated entirely by whether the receiver has its own sensor access. This is the duality the title ("See what I see, know what I think") names: context-aware = "know what I think" (sparse reasoning); context-unaware = "see what I see AND know what I think" (dense knowledge + reasoning).

### 1.2 Compressed-sensing head importance probe (§3.1, the modelless gem)

Setup: homogeneous self-communication (Qwen3-4B → Qwen3-4B) to eliminate cross-model confounds.
1. Sample `M=200` binary ablation masks `Φ ∈ {0,1}^{M×H}` over `H=1152` query heads, each mask stratified to zero exactly 5% (~58 heads).
2. For each mask `m`, run full sender→receiver pipeline on `N=100` eval samples, record mean accuracy `y_m`.
3. Center: `ỹ = y - y_baseline`.
4. Solve Lasso: `x̂ = argmin (1/2M)‖ỹ - Φx‖² + α‖x‖₁`, `α=1e-4`. Rank heads by `-x̂_i` (most-negative = most important — masking it most degrades accuracy).
5. Aggregate to KV-group granularity (GQA): `score(l, h_KV) = Σ_{h_Q ∈ group(h_KV)} x̂(l, h_Q)`.
6. Keep top-K KV groups; sweep K ∈ {10,20,50,100,150,288}.

**Recovery-limit caveat (Appendix C.5):** with `M=200` measurements over `N=1152` heads, CS recovery bound gives `s_max ≈ M/log₂(N/s) ≈ 70-80` informative coefficients. Past Lasso rank ~80, coefficients are zero; after GQA aggregation to 288 KV groups, only ~70 KV groups inherit a meaningful ranking. **K ≤ 150 is the statistically reliable regime.**

### 1.3 Dense alignment architecture (§4 — partially training-only)

The adapter `T_θ` maps sender cache `C_S(X)` → receiver-compatible `C̃_R(X)`:
- **Position-disentanglement:** `C̃_m = RemoveRoPE_m(C_m)`; transform in position-free space; `C̃_R = AddRoPE_R(C'_R)`. (We already have this — Shard/Still.)
- **Layer alignment:** monotonic depth-preserving map `a(l) = round(l·(L_S-1)/(L_R-1))`. (Trivial.)
- **Per-KV-group MLP transform with learnable gate γ ∈ [0,1]:** `T_{★,θ}(z) = W_2 σ(W_1 z + b_1) + b_2`, `★ ∈ {K,V}`, with 16× head-dim expansion. (Per-KV-group sigmoid gating we have; the 16× MLP expansion + training is the new part.)
- **Two-phase training:** Phase I reconstruction loss `‖K̃_R - K_R‖² + ‖Ṽ_R - V_R‖²`; Phase II generation loss with `X_R` sampled 50/50 context-aware/context-unaware. **→ riir-train.**

### 1.4 Results (why it matters)

Context-aware: matches or exceeds text communication (T2T) across all 6 model-pair directions on in-domain tasks at **2-3× lower TFLOPs**. Cheaper than even bare Receiver-only in 5/6 directions (sender does no autoregressive decode; receiver prefill is 6× smaller than T2T's re-encoded text).

Context-unaware: prior C2C baselines collapse to 0-2% accuracy; T2T-context-unaware drops to 19-57% on GSM8K and MCQ-chance elsewhere. **The paper's method holds at 81-91% on GSM8K**, within 0-10pp of its own context-aware numbers. This is the headline — dense alignment makes "mind reading" actually work.

---

## 2. Distillation

### 2.1 What's training-only → riir-train (do NOT implement here)

- The cross-model adapter `T_θ` weights.
- Phase I reconstruction + Phase II generation training loops.
- The 16× head-dim MLP expansion parameterization.
- Receiver-self-guided trace construction (Appendix A).

These are optimizer/loss/training-recipe concerns → note "→ riir-train" and stop. They are NOT the value of this paper for katgpt-rs.

### 2.2 What's modelless and transferable (the open primitive)

**Primitive 1 — CS-KV-Importance Probe (`CsKvProbe`)**
A `ConstraintPruner`-style artifact that, given (a) a task-distribution eval function, (b) `M` ablation masks, (c) `N` samples, produces a fixed-size ranking of KV groups. Inputs: KV cache shape `(L, G_kv)`. Outputs: `[f32; G_kv]` importance scores (Lasso coefficients, GQA-aggregated). Pure inference, zero training. The probe is run **once per task family** and cached as a `KvGroupRanking` (BLAKE3-committed, feature-gated `cs_kv_probe`).

**Primitive 2 — Context-Awareness-Gated Density Budget**
A single scalar `ca ∈ [0,1]` (receiver-side context availability, e.g. sigmoid of receiver's sensor coverage) interpolates the K-budget:
```
K(ca) = round(K_sparse + ca·(K_dense − K_sparse))
K_sparse ≈ 0.035 · G_kv   // paper's context-aware floor
K_dense  ≈ 0.87  · G_kv   // paper's context-unaware ceiling
```
Apply via the existing SP-KV `soft_gate_bias` path: `bias[g] = log(rank_score[g] + ε)` for top-K, `-∞` otherwise. **Sigmoid, never softmax** (AGENTS.md rule). When `ca` is high (receiver has its own context), K shrinks to the sparse floor; when `ca` is low (receiver is blind), K expands to the dense ceiling. This is the paper's central insight distilled into one scalar + one interpolation.

**Primitive 3 — Position-Disentangled Cross-Shape KV Transport**
Reuse `undo_rope` / `reapply_rope` from `crates/katgpt-kv/src/shard_kv/rope.rs`. For cross-shape transport (different `d_head` or `G_kv`), apply a **frozen** linear projection `W: R^{d_S} → R^{d_R}` (trained in riir-train, frozen-loaded here via `TrainingProvider`). The projection operates in position-free space. **This is the boundary: katgpt-rs ships the RoPE strip/restore + the projection dispatch; riir-train ships the projection weights.**

### 2.3 Fusion (the Super-GOAT claim)

The CS-probe alone is a GOAT diagnostic. The sparse/dense duality alone is a Gain architectural insight. **Fusing them with our existing HLA + fog-of-war infrastructure produces a new capability class** — adaptive-bandwidth NPC mind-reading — that none of the components alone enables:

| Component | Source | Role in fusion |
|-----------|--------|---------------|
| CS-KV-Importance Probe | this paper (§3.1) | Ranks which HLA dimensions carry signal per task family |
| Context-awareness scalar `ca` | this paper (§3.2 duality) | Bandwidth allocation: sparse when receiver has sensors, dense when blind |
| `HlaCacheProxy` (per-NPC 8-dim belief state) | katgpt-rs crates/katgpt-sense/src/reconstruction.rs + riir-games riir-ai/crates/riir-games-shared/src/zone/mood.rs | The "cache" being transmitted between NPCs |
| Fog-of-war `visible_radius` | riir-armageddon + Plan 118 | Source of `ca`: `ca = sigmoid(coverage_overlap)` between emitter and receiver |
| `share_trajectory` stub | riir-ai Plan 298 T3.3 (DEFERRED) | The unblocked host system — currently a trait hook with no full impl |
| Per-KV-group sigmoid gate | katgpt-rs SP-KV (Plan 070) | The gating mechanism that applies the CS-ranked density budget |

**Closest cousins across both repos (notes + code, mandatory two-layer check):**
- `katgpt-rs/.research/109_Shard_Drop_In_10x_KV_Cache_Compression.md` + `crates/katgpt-kv/src/shard_kv/rope.rs` — RoPE-strip primitive, single-model compression. **Same RoPE mechanism, no CS-probe, no sparse/dense routing, no cross-agent.**
- `katgpt-rs/.research/213_Still_Perceiver_KV_Cache_Compaction.md` + `riir-ai/crates/riir-engine/src/lora_still/mod.rs` — 3-step un-rotate/compress/re-rotate. **Same position-disentanglement, single-model, trained Perceiver (riir-train material).**
- `katgpt-rs/.research/238_*` MUX-Latent + `.plans/238_mux_latent_context_compression.md` — superposition fusion for context compression. **Single-model, no receiver-context-awareness axis.**
- `katgpt-rs/.research/086_RTPurbo` + `.plans/126_rt_turbo_retrieval_head_sparse_decode.md` — retrieval head identification. **Training-time head identification, NOT post-hoc per-task CS probe.**
- `riir-ai/.plans/298_crowd_scale_progressive_mcgs_npc_emergent_behavior.md` T3.3 — `HlaCacheProxy::share_trajectory` **stub, deferred, no bandwidth allocation.**

**What the fusion produces that none alone can:** a guard NPC that saw a thief can broadcast, to other guards, *only the 3.5% of HLA dimensions that carry reasoning signal* if those guards also have line-of-sight (context-aware), OR *the full 87% dense HLA* if they're around a corner and blind (context-unaware). Bandwidth auto-adapts to receiver sensor coverage. No existing NPC comms system, federation coupling, or KV compression primitive in our repos does this.

### 2.4 Latent vs raw boundary (mandatory check)

| Data | Space | Synced? | Notes |
|------|-------|---------|-------|
| CS-probe ranking `[f32; G_kv]` | Latent (per-task) | No — local to zone, BLAKE3-committed in `ZoneExpertBundle` | Cached per task family; not per-NPC |
| Context-awareness scalar `ca` | Raw (derived from `visible_radius` overlap) | Yes — computed from synced `MapPos` | Deterministic, replayable |
| Transmitted HLA slice | Latent | **No** — local zone channel, never enters `SyncBlock` | Per AGENTS.md: local consumption stays latent |
| 5 emotion scalars (existing bridge) | Raw | Yes — unchanged | The sync boundary is NOT moved |
| CS-probe ablation masks `Φ` | Raw (binary) | No — diagnostic only | Run offline, not in hot path |

**Compliance:** dense HLA transport between NPCs in the same zone is local (per AGENTS.md: "If data is consumed locally ... it SHOULD be latent"). The 5-scalar sync rule (Plan 309) is unchanged — dense HLA never crosses `SyncBlock`. The `ca` scalar derives from already-synced `MapPos` + `visible_radius`. No new raw data crosses the quorum boundary.

---

## 3. Verdict

**Super-GOAT.** All four novelty-gate criteria pass:

| Gate | Criterion | Evidence |
|------|-----------|---------|
| Q1 No prior art? | CS-Lasso head importance + sparse/dense duality as routing axis | `grep` across katgpt-rs + riir-ai + riir-armageddon notes AND code: zero matches for `head_importance\|cs_lasso\|ablation_mask\|kv_group_score`. RoPE-strip shipped (Shard/Still) but that's one component, not the fusion. |
| Q2 New class of behavior? | Adaptive-bandwidth NPC latent communication | No existing NPC comms / federation / KV-compression primitive allocates bandwidth by receiver-side context availability. `share_trajectory` is a deferred stub (Plan 298 T3.3). |
| Q3 Product selling point? | "Our NPCs read each other's minds — guards share what they saw AND concluded, with bandwidth auto-allocated by whether the receiver has its own eyes on target." | One-sentence moat, citable, demoable in a 4-player arena. |
| Q4 Force multiplier? | ≥2 pillars | HLA belief state (Pillar 5), fog-of-war zone attention (Plan 118), Civilization Engine inheritance (Plan 168 G12), Crowd MCGS (Plan 298 T3.3), federation KL coupling (Plan 231). **5 systems.** |

**Selling point (the moat):** adaptive-bandwidth NPC mind-reading. The open primitive (CS-probe + density interpolator) is the adoption hook in katgpt-rs; the architectural guide (riir-ai/.research/133) is the private selling-point doc. The guide contains the validation protocol (G1-G8) — created now, BEFORE the gate runs, per the skill's anti-deferral rule.

**Routing:**
- Open primitive → `katgpt-rs/src/cs_kv_probe/` (new module) + Cargo feature `cs_kv_probe`. Plan 280.
- Architectural guide → `riir-ai/.research/133_NPC_Mind_Reading_Adaptive_Bandwidth_Guide.md`.
- Runtime plan → `riir-ai/.plans/311_npc_mind_reading_runtime.md`.
- Training (adapter weights for cross-shape projection) → note "→ riir-train", out of scope this session.

---

## 4. Risks and honest caveats

1. **CS-probe is per-task-family, not per-NPC.** Running 200 ablation masks × 100 samples per NPC per tick is infeasible. The probe must be cached per task family (e.g. "guard-spotting-thief", "merchant-pricing") and refreshed at zone-bundle freeze/thaw cycles, not per-tick. This bounds compute but means the ranking is coarse-grained.
2. **Recovery limit (Appendix C.5).** With `M=200` measurements, only ~70 KV groups get statistically reliable rankings. Below K=150 the CS-vs-random gap is real (+27pp); above K=150 the ranking degrades to noise. Our budget interpolator must clamp K to the reliable regime.
3. **The "context-awareness" scalar is an idealization.** Real fog-of-war isn't binary — it's partial coverage, stale observations, confidence decay (`sigmoid(-λ·Δt)` per the spatial-cognition rule). Mapping the continuous `ca` to the K-budget via the linear interpolation `K(ca) = K_sparse + ca·(K_dense - K_sparse)` is a first-order model; the true relationship may be non-linear (the paper's Fig 5 shows a phase transition, not a linear ramp).
4. **Cross-shape projection needs training.** The position-disentangled linear projection `W: R^{d_S} → R^{d_R}` for heterogeneous NPC classes (Knight HLA ≠ Mage HLA dimension layout) requires Phase-I reconstruction training → riir-train. For same-class NPCs (homogeneous), no projection needed — identity map. The katgpt-rs primitive ships the dispatch; riir-train ships the weights.
5. **Bandwidth reality check.** Paper reports ~20-30 MB KV payload per sample for Ours vs hundreds of bytes for text. For NPC-to-NPC intra-zone comms at 20Hz with thousands of NPCs, even the sparse 3.5% budget may be too chatty. The GOAT gate (riir-ai guide G7) must measure actual wire bytes per zone per tick.

---

## TL;DR

Super-GOAT. The paper's cross-model adapter training is riir-train material, but two modelless primitives survive: (1) a compressed-sensing KV-group importance probe (training-free, post-hoc, per-task — zero prior art in our repos), and (2) the sparse-reasoning/dense-knowledge duality as a bandwidth-allocation axis (~25× density swing gated by receiver context). Fusing these with our shipped HLA + fog-of-war + the deferred `share_trajectory` stub (Plan 298 T3.3) produces **adaptive-bandwidth NPC mind-reading** — a new capability class with a one-sentence moat, connecting 5 existing pillars. Open primitive → katgpt-rs Plan 280 (`cs_kv_probe` feature). Private selling-point guide → riir-ai/.research/133. Runtime plan → riir-ai/.plans/311. Position-disentanglement (RoPE strip/restore) is ALREADY shipped in `crates/katgpt-kv/src/shard_kv/rope.rs` — do not reinvent it; the novelty is the CS-probe + the duality-as-routing-axis + the NPC fusion, not the RoPE mechanism.

> **PASS-Redirects (synthesis):** Fu, Min, Zhang, Yan, Dai, Ouyang, Wang [arXiv:2510.03215 "Cache-to-Cache: Direct Semantic Communication Between Large Language Models"] — the founding member this paper's successor builds on (the "prior C2C baselines" of §1.4); trained fuser routes under riir-train Issue 523 T4 defer-with-trigger, deterministic cross-arch projection already refuted (riir-train Bench 422), modelless deltas (enrichment oracle, token-alignment spec, hub-and-spoke KV-Latent O(N) topology) consumer-free → classification record in `.research/635_C2C_Cache_to_Cache_Cross_Model_KV_Communication_PASS.md`.
