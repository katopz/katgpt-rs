# Issue 034: Shape-Adaptive Adapter Novelty Gate — close Q1 before verdict on Research 269

**Opened:** 2026-06-19
**Blocks:** Final verdict on [Research 269](../.research/269_Variable_Width_Shape_Adapter_Fusion.md) (`> <former` × on-the-fly LoRA × Hydra layer-skip fusion).
**Owner:** unassigned
**Type:** novelty gate (literature survey + mechanism feasibility check)

---

## Context

Research 269 documents a fusion idea sparked by `> <former` (arXiv:2606.18246): **shape-adaptive adapter routing** — train LoRA adapters with explicit per-layer shape objectives (e.g. ×-shape narrow-middle for fast/combat, wide-middle for deep/dialog), hot-swap between shape profiles at runtime, and drive Hydra Budget's layer-skip plan off the *adapter's* per-layer profile rather than the *base model's* intrinsic profile.

The in-codebase novelty check passed (vocabulary-translated grep across both repos, both layers — no shipped primitive characterizes an adapter by its per-layer shape profile; OPD is per-module-type, `AdapterShape` is static per-adapter, Hydra profiles the base). But the **broader literature check (Q1)** was not done in-session and the research skill explicitly forbids committing Super-GOAT without it.

The honest call in R269 was "fusion — novelty TBD" rather than "Super-GOAT candidate" precisely because this gate is open. This issue tracks closing it.

## The four sub-questions to resolve

### Q1.a — Is "per-layer adapter shape profile" novel in the adapter-composition literature?

Survey arxiv for (use the keyword search URL from AGENTS.md):
- `layer-wise adapter capacity allocation`
- `adapter shape profile routing`
- `variable-width LoRA`
- `layer-skipping adapter composition`
- `per-layer adapter rank allocation` (this one almost certainly has hits — rank-per-layer is a known axis; check whether it's been combined with runtime routing)
- `adapter width profile hot-swap`

**Pass criterion:** no paper proposes (per-layer adapter capacity profile) × (runtime hot-swap between profiles) × (inference-time layer skip driven by the profile). If any two of the three exist together, the fusion is GOAT (novel-in-combination) not Super-GOAT (novel mechanism). If all three exist together, downgrade R269 to Gain.

### Q1.b — Is the "emergent narrowing" mechanism feasible?

The fusion cannot structurally narrow a frozen uniform base. It relies on:
1. Adapter learning to suppress its own contribution to middle-layer output dims (low-rank cancel).
2. Hydra Budget detecting the suppression via `effective_rank` / `participation_ratio` drop on a calibration set.
3. Residual stream carrying bypassed info forward (already structurally true).

**Open question for riir-train:** can a low-rank LoRA meaningfully "narrow" a layer's effective width without hurting quality? This is a training feasibility question, not modelless. File a separate riir-train issue if R269 promotes.

### Q1.c — Is "adapter-driven Hydra skip plan" novel?

Hydra Budget's `HydraBudgetConfig { modelless: bool }` today means "use a pre-computed profile of the *base model*." The fusion redefines this as "use a pre-computed profile of the *currently-loaded adapter*" — meaning the skip plan changes on hot-swap. Confirm this is not already implemented (grep `HydraBudgetConfig` call sites for any adapter-aware variant).

### Q1.d — Does `SnapshotMeta` extension break anything?

R269 proposes extending `riir-ai/crates/riir-engine/src/snapshot.rs::SnapshotMeta` with a per-layer width profile (BLAKE3-committed). Confirm the existing `SnapshotMeta` serialization is forward-compatible (serde-with-default fields) so old snapshots load without the profile.

## Resolution criteria

| Outcome | Action on R269 |
|---|---|
| Q1.a = no prior art AND Q1.b = mechanism feasible AND Q1.c = novel AND Q1.d = compatible | **Promote R269 to Super-GOAT.** Mandatory outputs due in the follow-up session: (1) open `ShapeAdaptiveRouter` primitive in `katgpt-rs/src/inference_router/`; (2) private `riir-ai/.research/NNN_shape_adaptive_adapter_guide.md` with validation protocol G1–Gn; (3) plans in katgpt-rs (modelless router) + riir-ai (hot-swap wiring) + riir-train (shape-objective training recipe). |
| Q1.a has partial prior art (2-of-3 exist) | **Downgrade R269 to GOAT.** Plan + implement behind `shape_adaptive_router` feature flag. Benchmark vs vanilla adapter routing (Dynamic Pair, Polytope). Promote if ≥5% latency win at iso-quality. |
| Q1.a has full prior art (all 3 exist together) OR Q1.b infeasible | **Downgrade R269 to Gain.** Plan-only, feature-flagged, low priority. Close this issue. |

## Tasks

- [ ] **T1** Run the six arxiv keyword searches above; tabulate hits with one-line relevance assessment each.
- [ ] **T2** Read top 3 closest papers from T1 in full (via `https://r.jina.ai/https://arxiv.org/pdf/{ID}`).
- [ ] **T3** Grep `HydraBudgetConfig` call sites; confirm no adapter-aware variant exists.
- [ ] **T4** Read `SnapshotMeta` serialization; confirm forward-compat.
- [ ] **T5** Write Q1.a–Q1.d verdict into R269 §3 and close this issue with the resolution action.

## Estimated effort

T1+T3+T4: ~30 min. T2: ~1 hr. T5: ~15 min. Total: ~2 hr.
