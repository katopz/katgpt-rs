# Research 269: Variable-Width `> <former` × On-the-Fly LoRA × Hydra Layer-Skip — Shape-Adaptive Adapter Fusion

> **Source:** Wu, Sieberling, Tan, Panda, Polyanskiy, Kim. *Variable-Width Transformers* (`> <former`). [arXiv:2606.18246](https://arxiv.org/abs/2606.18246). MIT / MIT-IBM Watson AI Lab. 16 Jun 2026.
> **Date:** 2026-06-19
> **Status:** Active — **fusion idea, novelty TBD (needs Q1–Q4 check before verdict)**. See [`.issues/034_shape_adapter_novelty_gate.md`](../.issues/034_shape_adapter_novelty_gate.md).
> **Related Research:** 148 (Hydra Effect → Hydra Budget), 231 (OPD per-module energy profile), 247 (Dense Latent cross-model adapters, training→riir-train pattern), 258 (Sink-Aware / compression valleys), 266 (DenseMesh adaptive width).
> **Related Plans:** 165 (Hydra Budget — layer skip via pre-computed profiles), 260 (Dynamic Pair LoRA routing), 279 (Manifold Power Iter MoE Router — snapshot-swap hook).
> **Cross-ref (riir-ai):** snapshot.rs `SnapshotMeta`, episode_buffer.rs `LoRAHotSwap`/`LoRAWeightVersion`, riir-gpu `AdapterShape`.
> **Classification:** Public (katgpt-rs engine note). The training recipe itself → `riir-train`.

---

## TL;DR

The paper proposes a `×`-shaped transformer (wide early/late, narrow middle) with a parameter-free **carry-forward residual** that lets inactive dimensions bypass narrowed layers. At parameter parity this gives ~3% perplexity win, ~22% FLOP reduction, ~15% KV reduction, and (in analysis) it **mitigates mid-layer representation collapse** — the same "compression valley" phenomenon our Sink-Aware work (R258) and Hydra Budget (R148) already target.

**Two honest calls in this note:**

1. **Architecture recipe → `riir-train`.** Pre-training a `> <former` from scratch is pure training research. The recipe (geometric width schedule, `ℓ*=0.75L`, `d_ℓ*=0.3d`, carry-forward expansion) belongs in the training vault, not here. One-line redirect; no files created in katgpt-rs for the architecture itself.

2. **Fusion idea, novelty TBD.** With **on-the-fly LoRA** (`LoRAHotSwap`, `dispatch_lora_merge`, `SenseHotSwap`) and a **live `riir-train`**, the paper's *insight* (variable width is a structural regularizer that prevents collapse and concentrates useful compute in early/late layers) becomes a **shape-adaptive adapter** primitive rather than a new base model. This is the distillation worth tracking. It is **not** a committed Super-GOAT — Q1 (no prior art) has not been verified against the adapter-composition literature, so per the research skill this is filed as "novelty TBD" with an issue for the gate.

**Distilled for katgpt-rs (modelless, inference-time):** the per-layer "effective width profile" of an adapter — which layers it concentrates capacity in vs suppresses — is a new routing/skip dimension that combines three shipped primitives (Hydra layer-skip, OPD per-module profile, on-the-fly hot-swap) into a runtime-selectable architecture shape.

---

## 1. Paper Core Findings (verified by reading)

| Finding | Mechanism | Relevance here |
|---|---|---|
| `×`-shape beats uniform at parameter parity (200M–2B dense, 3B/1B MoE) | Wide early/late, narrow middle, geometric schedule `d_ℓ = α·d_{ℓ−1}` with `ℓ*=0.75L`, `d_ℓ*=0.3d` | Training architecture → riir-train |
| **Carry-forward residual** (parameter-free) | Fixed global residual width = widest layer; each block reads/writes a slice; inactive dims bypass and are restored from the most recent layer that touched them; contraction = truncation, expansion = copy-or-zero-pad | Modelless primitive candidate — structured residual bypass |
| **Mid-layer collapse is real and severe** | Uniform models: normalized matrix entropy → ~0 by layer ~10 (compression valley, de Llano et al. 2026 — same paper our R258 cites). `> <former`: maintains higher entropy through the bottleneck | Already exploited by Hydra Budget (R148/P165) and Sink-Aware (R258/P287) |
| MLP activation Participation Ratio collapses | Uniform: width-normalized energy utilization <5% by layer ~10. `> <former`: maintains ~1000 effective dims through middle | Same metric we already compute (`participation_ratio` in SpectralQuant, `effective_rank` in data_probe) |
| Inference benefits follow automatically | Params ∝ d² (matched); attention FLOPs and KV ∝ d (linear), so nonuniform width strictly lowers avg d → 15% KV, 22% FLOP | This is the *training-time* source of the savings; the inference-time reflection is what Hydra/Sparse-MLP already harvest on uniform models |
| Carry-forward beats learned projection or zero-pad (Table 4, 500M) | Copy-from-prior-layer: 3.099; zero-pad: 3.124; trained projection: 3.150 | The parameter-free bypass is the load-bearing mechanism — and it's exactly what a frozen base + adapter pool can simulate |

## 2. Distillation

### 2.1 What's already shipped (the prior-art surface — three granularities)

| `> <former` insight | Shipped cousin | File / Plan | Granularity |
|---|---|---|---|
| Middle layers collapse → skip them | **Hydra Budget** `HydraSkipPlan { skip_layers: Vec<bool> }`, `HydraBudgetConfig { modelless: bool }` | `src/pruners/hydra_budget.rs`, P165, R148, default-on, GOAT 4/4 | **Layer** |
| MLP dead dimensions → skip them | **Sparse MLP** + **Prism** per-capability masks + **CNA** neuron discovery | P022, R191, R053 | **Dimension (within-layer)** |
| Per-layer capacity varies → adapt compute | **DenseMesh adaptive_width** `WidthDecision::{Contract,Neutral,Expand}` driven by Collapse-Aware + BreakevenRouter | `dense_mesh/adaptive_width.rs`, P266, R234 | **Topology (across-nodes)** |
| Normalized matrix entropy per layer | **`effective_rank`** (Roy-Vetterli) — `normalized_matrix_entropy = log(effective_rank)/log(r)` | `crates/katgpt-core/src/data_probe/geometry.rs` | Metric |
| MLP participation ratio per layer | **`participation_ratio`** `d_eff = (Σλ)²/Σ(λ²)` | SpectralQuant, P078, default-on, GOAT-proven | Metric |
| Attention sinks / compression valleys | **Sink-Aware Attention** (targets the de Llano 2026 finding the paper cites) | P287, R258 (deferred for latency; diagnostic ships) | Head |
| Per-module energy profile of adapter | **`ModuleEnergyProfile::PAPER_AVERAGE { ffn, attn, embed, other }`** | `src/inference_router/router_compute_target.rs`, R231 (OPD) | Module-type |
| Per-adapter shape descriptor | **`AdapterShape { rank, in_dim, out_dim }`** | `riir-ai/crates/riir-gpu/src/optimizer_amuse.rs` | Static per-adapter |
| Per-snapshot metadata + atomic swap | **`SnapshotMeta { blake3_hash, n_layers, ... }`**, `LoRAHotSwap`, `SenseHotSwap`, `KernelHotSwap` | `riir-ai/crates/riir-engine/src/snapshot.rs`, P276, P279 | Snapshot |

**The gap:** every shipped cousin is either (a) per-module-type (OPD: FFN vs Attn), (b) per-adapter-static (`AdapterShape`: fixed rank per adapter), or (c) per-layer-intrinsic (Hydra: skip based on the *base model's* profile). **Nothing characterizes an adapter by its per-LAYER shape profile** — which layers it concentrates capacity in vs suppresses. That is the dimension `> <former` operates on, and it's orthogonal to all three shipped axes.

### 2.2 The fusion (novelty TBD)

**Shape-Adaptive Adapter Routing**: `> <former × Hydra Budget × On-the-Fly LoRA × OPD`

```
riir-train:  Train adapter pool with explicit shape objectives
             (e.g. "combat" adapter: ×-shape, narrow middle, fast;
              "dialog" adapter: inverted-× or uniform, wide middle, deep)
             → each adapter ships a per-layer-width profile alongside its weights
             → profile committed in SnapshotMeta (BLAKE3-hashed extension)

katgpt-rs:   ShapeAdaptiveRouter (new modelless primitive)
             input complexity signal (Collapse-Aware entropy, EGA spectral salience)
               ↓
             pick adapter by shape×complexity match
               ↓
             Hydra Budget reads the adapter's per-layer profile
               ↓
             skip suppressed layers; carry-forward residual preserves info
               ↓
             Sparse MLP skips dead dims within active layers
             → one frozen base + N shape-profiled adapters = N runtime architectures

riir-ai:     LoRAHotSwap atomic swap between shape profiles per NPC / context
             NPC in 20Hz combat tick: narrow-middle adapter (Hydra skips layers 8–24)
             NPC in dialog: wide-middle adapter (full depth)
             Freeze/thaw persists shape profile in NeuronShard Cold tier
```

**Why this isn't just "Hydra Budget with a new signal":** Hydra's `modelless: bool` loads a profile *of the base model*. The fusion loads a profile *of the adapter* — meaning the skip plan **changes on hot-swap**, not just on model load. That's a runtime capability Hydra doesn't have today.

**Why this isn't just "OPD per-module profile":** OPD characterizes FFN-vs-Attn energy *within* a layer. The fusion characterizes layer-5-vs-layer-20 energy *across* layers. Orthogonal axes; both can compose.

### 2.3 Honest uncertainty on the mechanism

The paper achieves variable width **structurally** — narrowed layers literally lack weights for the bypassed dimensions. The fusion cannot do this on a frozen uniform base. Instead it would achieve **emergent narrowing** via three composed mechanisms:

1. Adapter learns to **suppress** its own contribution to certain middle-layer output dims (low-rank update that cancels the base).
2. Hydra Budget detects the suppressed layers (effective_rank / participation_ratio drop on a calibration set) and **skips them entirely**.
3. Residual stream **carries the information forward** (the paper's carry-forward insight — already structurally true in any residual transformer).

The open question is whether (1) is achievable with low-rank LoRA without hurting quality. This is a **riir-train** research question, not a modelless one. The modelless primitive (ShapeAdaptiveRouter + adapter-profile-driven Hydra) is well-defined regardless of how the profile is produced.

## 3. Verdict

**Fusion idea — novelty TBD, needs Q1–Q4 check before verdict.** Not a committed Super-GOAT.

| Gate | Status | Evidence |
|---|---|---|
| Q1 No prior art | ❓ **UNCERTAIN — must check literature** | Per-layer adapter shape profile is a new dimension in *our* codebase (confirmed via vocabulary-translated grep across both repos, both layers). But "shape-adapted adapters" / "layer-skipping adapters" exist in the broader adapter-composition literature — needs a proper arxiv survey before claiming novelty. See [Issue 034](../.issues/034_shape_adapter_novelty_gate.md). |
| Q2 New class of behavior | ✅ Likely yes | Runtime-selectable architecture *shape* (not just content) via adapter swap. No shipped primitive does this. |
| Q3 Product selling point | ✅ Likely yes | "One frozen base + N shape-profiled adapters = N runtime architectures. Combat NPCs run narrow-fast; dialog NPCs run wide-deep; all from one model." |
| Q4 Force multiplier | ✅ Yes | Connects OPD (R231), Hydra Budget (R148/P165), on-the-fly LoRA (riir-ai hot-swap), Sink-Aware (R258), Sparse MLP (P022), DenseMesh adaptive_width (R234/P266), Manifold Power Iter Router (R246/P279). ≥6 pillars. |

**Per the research skill:** because Q1 is not committed YES, this is filed as "novelty TBD" with an issue — NOT as "Super-GOAT candidate." If Issue 034 closes with Q1=YES (no prior art in literature), this note upgrades to Super-GOAT and the mandatory outputs (open primitive in katgpt-rs + private riir-ai guide + plans) become due in that follow-up session.

**The pure architecture recipe → `riir-train`.** One-line redirect; no katgpt-rs files created for the `×`-shape training method itself.

### Tier reasoning

- **Not Pass:** the fusion idea is legitimate and the prior-art surface leaves a real gap (per-layer adapter shape profile). Dismissing it as "already covered by Hydra Budget" was wrong — Hydra profiles the *base model*, not the *adapter*.
- **Not Super-GOAT (yet):** Q1 literature check is genuinely open. Adapter shape adaptation is an active research area; claiming novelty without the survey would repeat the `evolve_hla` overclaim failure mode.
- **Not GOAT/Gain:** those tiers require a committed primitive + benchmark. The primitive is well-defined but its value depends on the training side (riir-train) actually producing shape-profiled adapters that don't hurt quality — which is itself TBD.

## 4. What would change the verdict

| If Issue 034 finds... | Then... |
|---|---|
| Prior art on "per-layer adapter shape profile" in literature | Downgrade to **Gain** — the fusion is still useful as a composition of our shipped primitives, but no moat. Plan-only, feature-flagged. |
| No prior art; mechanism (adapter-driven layer suppression + Hydra skip + carry-forward) is novel | Upgrade to **Super-GOAT**. Mandatory outputs in follow-up session: (1) open `ShapeAdaptiveRouter` primitive in katgpt-rs; (2) private `riir-ai/.research/NNN_*.md` guide with validation protocol; (3) plans in katgpt-rs + riir-ai + riir-train. |
| Prior art exists but our specific composition (× OPD × Hydra × hot-swap) is novel-in-combination | **GOAT** — plan + implement behind feature flag, benchmark vs vanilla adapter routing, promote if it wins. |

## 5. Cross-references for the follow-up session

- **Closest cousins to fuse with:**
  - `katgpt-rs/.research/231_Sparse_Off_Principal_Task_Vector_OPD.md` — per-module energy profile; extend to per-layer
  - `katgpt-rs/.research/148_*.md` (Hydra Effect) + `katgpt-rs/.plans/165_*.md` (Hydra Budget) — layer skip machinery to make adapter-driven
  - `katgpt-rs/.research/247_Dense_Latent_Heterogeneous_Communication_CS_Probe.md` — same training→riir-train + modelless-survives pattern
  - `katgpt-rs/.research/266_DenseMesh_Latent_Node_Network.md` — topology-level width adaptation; the layer-level version is the gap
  - `katgpt-rs/.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md` — same compression-valley phenomenon
- **Runtime plumbing:** `riir-ai/crates/riir-engine/src/snapshot.rs::SnapshotMeta` (extend with per-layer profile), `riir-ai/crates/riir-engine/src/episode_buffer.rs::LoRAHotSwap` (atomic swap by shape profile).
- **Training side:** `riir-ai/crates/riir-gpu/src/optimizer_amuse.rs::AdapterShape` (currently static per-adapter; would need a per-layer variant).

## TL;DR

The architecture is training research (→ `riir-train`). The analysis methodology is `effective_rank`-equivalent math we already ship. BUT the user's pushback was correct: with on-the-fly LoRA and a live riir-train, the paper's *insight* (per-layer width profile is a structural regularizer; carry-forward preserves bypassed info) becomes a **shape-adaptive adapter** fusion that combines Hydra layer-skip + OPD per-module profile + hot-swap into runtime-selectable architecture shape. The gap in our shipped prior art is real: nothing characterizes an adapter by its **per-layer** shape profile (OPD is per-module-type, `AdapterShape` is static per-adapter, Hydra profiles the base model not the adapter). Verdict is **fusion — novelty TBD** because Q1 (no prior art) needs a literature check before committing Super-GOAT. Filed Issue 034 for the gate; no guide or plans created until Q1 resolves YES.
