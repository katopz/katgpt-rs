# Research 311: Analytic Lattice Encoder/Decoder + Functional Attention Chain Composition

> **Source:** Synthesis — no single source paper. Fusion of:
> - `arxiv:2605.31559` Functional Attention (Xiao et al., ICML 2026) — operator composition
> - `arxiv:2606.05345` PJ-RoPE (Zhang 2026) — lattice-as-difference-module algebra
> - `arxiv:2110.13475` Gyrocalculus on SPD matrices (López et al., NeurIPS 2021) — closed-form curved-space distance per slot
> - `arxiv:2606.02427` Spectral Audit of Operator Networks (Gao/Yang/Karniadakis 2026) — Fourier-mode tangent audit (GOAT-gate verifier)
> **Date:** 2026-06-26 (revised 2026-06-26 — see revision note)
> **Status:** Active (Super-GOAT — see §3)
>
> **Revision note (2026-06-26):** The original draft claimed three novel
> primitives (`AnalyticLatticeEncoder` trait, `direction_vector_decode`,
> `compose_chain`). Mid-implementation audit found that the **encoder half is
> redundant** — `riir-ai/crates/riir-engine/src/fourier/encoder.rs` already
> ships `FourierEncoder::encode_position_into` / `encode_offset_into`, a
> closed-form analytic encoder. This revision narrows the novelty claim to the
> three primitives that are **genuinely missing**: (1) `compose_chain`
> (cross-entity operator product), (2) `batch_compose_chain` (zone-batched
> amortization), and (3) **ASOC (Adaptive Speculative Operator Cascade)** —
> the headline contribution, built on the already-shipped `GpuFuture` /
> `Join` / `block_on` substrate in `riir-gpu-async`. The Super-GOAT verdict
> (4/4) is preserved because the novelty class is unchanged; only the
> *encoder* attribution is corrected.
>
> **Related Research:** 290 (Latent Field Steering), 294 (Viable Manifold Graph), 298 (Inverting Bellman Closed-Form), 257 (FuncAttn spectral transport), 303 (Transolver/FuncAttn predecessor), 306 (Galerkin/FuncAttn grandparent), 296 (Stokes/DEC vocabulary)
> **Related Plans:** 335 (Zone Eggshell — substrate, SHIPPED), 312 (Viable Manifold Graph), 286 (Functional Attention), 309 (Latent Field Steering), 330 (this primitive's plan), riir-ai P339 (Bevy isometric demo)
> **Cross-ref (riir-ai):** Research 162 (game runtime Super-GOAT guide), Plan 339 (Bevy demo)
> **Classification:** Public (katgpt-rs note)

---

## TL;DR

**Distilled primitive (revised):** the modelless **ASOC cascade** — an autoplay
bot's per-tick action selector that, on every tick, (a) **always** emits a
cheap plasma-tier draft synchronously (so the bot never stalls), (b) joins 3
hot-tier rederive `GpuFuture`s (`C_boss`, `C_quest`, `C_player`) via
`GpuFuture::Join`, and (c) on `Poll::Pending` (GPU congestion) **returns the
stale plasma draft instead of propagating Pending** — the bot's action loop
never blocks on GPU.

ASOC is built on two genuinely-novel closed-form primitives:
1. **`compose_chain`** — cross-entity operator product
   `C_boss × C_quest × C_player` (vs `funcattn_compose` which is token-level).
2. **`batch_compose_chain`** — zone-batched amortization: factor the shared
   prefix `C_qb = C_boss × C_quest` once, then apply `C_player_i × C_qb` for
   N players in the same zone (O(N+k³) vs O(N·k³)).

**Redundant (intentionally dropped):** the `AnalyticLatticeEncoder` trait
originally proposed here is redundant with `FourierEncoder::encode_*_into`
which already ships closed-form `entity → [f32; N]` encoding. We do NOT
re-ship a parallel encoder API. The decoder primitive `direction_vector_decode`
remains novel as a *generalization* of `riir-games::scalar_projection` out of
HLA-specific 5-scalar semantics into a generic single-direction primitive.

The math substrate ships (Plan 335 `lattice_edge_utility_into`, Plan 286
`funcattn_forward`, Plan 273 `extract_functor`/`apply_functor`, plus
`riir-gpu-async` for the `GpuFuture` substrate). What's missing is the
**chain composer** + **ASOC orchestration** + **spectral audit verifier**.

---

## 1. What ships today (verified grep, 2026-06-26)

| Shipped primitive | File | Role in the synthesis |
|---|---|---|
| `lattice_edge_utility_into` | `katgpt-rs/crates/katgpt-core/src/...` (Plan 335 P5) | SIMD FMA `utility = sigmoid(x·w_x + y·w_y + z·w_z + f·w_f + ...)` over the LatCal eggshell lanes |
| `ZoneGeometryPod` w/ `[x,y,z,safety\|interest,occupancy,threat,destruction]` lanes | `riir-neuron-db/src/zone_geometry.rs` (Plan 335 P0) | The 8-lane SIMD register target — already typed per-slot |
| `LatCalEggshell::project_from`, `validate` | `riir-neuron-db/src/zone_geometry.rs` | Bridge between latent ops and raw committed lanes |
| `extract_functor`, `apply_functor`, `functor_gate` (rank-1 + rank-k + affine) | `riir-ai/crates/riir-engine/src/latent_functor/arithmetic.rs` | Displacement-based functor between (src,tgt) pairs — the "bending" primitive |
| `funcattn_forward`, `solve_convex_combo_dual`, `compute_basis_into`, `pre_rotate_basis_weights_into` | `katgpt-rs/crates/katgpt-core/src/funcattn.rs` | k×k spectral transport operator C (Tikhonov dual solver) |
| `funcattn_compose` (spectral_pre_rotate, chiar_blend, freeze_thaw) | `katgpt-rs/src/funcattn_compose/` | Token/weight-level composition (NOT cross-entity) |
| `FieldRegistry`, `FactionStanceRegistry`, `apply_all` | `riir-ai/crates/riir-engine/src/latent_field_wiring.rs` | Zone/faction latent field injection (Plan 309) |
| `find_path_into`, `find_distance_into` (A*) | `riir-ai/crates/riir-engine/src/pathfinder.rs` | Tactical single-path movement (different layer than eggshell — coexist by design per Plan 335 G2) |

**The gap (Q1 — no prior art?):**
- ✅ Closed-form `entity → [f32; N]` encoding — **already ships** as
  `FourierEncoder::encode_position_into` / `encode_offset_into` in
  `riir-ai/crates/riir-engine/src/fourier/encoder.rs`. NOT a novel contribution.
- ❌ No `compose_chain(&[TransportOp])` — `funcattn_compose/` chains at
  token/weight level, not across boss/quest/player entities. **NOVEL.**
- ❌ No `batch_compose_chain` — no zone-batched operator prefix factoring. **NOVEL.**
- ❌ No ASOC cascade — `GpuFuture`/`Join`/`block_on` substrate exists in
  `riir-gpu-async`, but no `ComposerTick: GpuFuture` that returns a stale
  plasma draft on `Poll::Pending` to keep the bot loop non-blocking. **NOVEL (headline).**
- ❌ No direction-vector action decoder — `lattice_edge_utility_into` is
  edge-utility, not entity-action projection. `scalar_projection.rs` is
  HLA-specific. **NOVEL (generalization, not parity).**
- ❌ No spectral audit verifier — no Fourier-mode tangent operator check. **NOVEL (G6 gate).**

Q1 = **YES (genuinely missing)** — but the novel set is compose_chain +
batch_compose_chain + ASOC + decoder-generalization + spectral-audit, NOT
the encoder (which ships in `fourier/`).

---

## 2. Distillation — fusion protocol

### 2.1 Vocabulary translation (paper ↔ code)

| Paper term | Codebase-equivalent | Verified shipped? |
|---|---|---|
| "analytic encoder" / "closed-form embedding" | `FourierEncoder::encode_position_into` / `encode_offset_into` (`riir-engine/src/fourier/encoder.rs`) | **YES — ships** (NOT novel; original R311 draft was wrong here) |
| "operator composition" / "functional correspondence" | `compose_chain` (NEW); `funcattn_compose` (token-level only) | Partial — `compose_chain` is the novel cross-entity gap |
| "zone-batched prefix factoring" | `batch_compose_chain` (NEW) | No — gap (amortizes N players in one zone) |
| "adaptive speculative cascade" / "stale-draft fallback" | ASOC `ComposerTick: GpuFuture` (NEW) | No — gap (uses shipped `GpuFuture`/`Join`/`block_on` as substrate) |
| "lattice coordinate" / "difference module" | LatCal eggshell lanes `[x,y,z,safety\|...]` | Yes (Plan 335) |
| "gyrovector addition" / "curved-space distance" | LatCal fixed-point bridge in `riir-chain/src/encoding/latcal_fixed.rs` | Yes (commitment side) |
| "direction vector projection" | `project_to_scalars` (decoder-only, HLA→5 scalars) | Partial — needs generalization to non-HLA directions |
| "spectral audit" / "tangent operator" | (NEW GOAT-gate verifier primitive) | No — gap |

### 2.2 Closest cousins (3)

1. **Plan 335 (Zone Eggshell)** — closest by far. Provides the substrate. The encoder/decoder is the missing API layer on top.
2. **Plan 312 (Viable Manifold Graph)** — provides safe-manifold navigation; consumes the encoder output to find geodesic paths.
3. **Research 290 / Plan 309 (Latent Field Steering)** — provides top-down field injection; the encoder can *emit* fields, multiplying this primitive.

### 2.3 Fusion — what novel combination does this enable?

**Synthesis (NEW capability class, revised):** "An autoplay bot that, every
tick, drafts a synchronous plasma-tier action in nanoseconds, then speculatively
joins three hot-tier transport-operator rederive futures (`C_boss`, `C_quest`,
`C_player`), and on GPU congestion returns the stale plasma draft instead of
blocking — composing `C_boss × C_quest × C_player` cross-entity (not token-
level) when the join completes. The bot loop never stalls; the GPU pipeline
amortizes across N players in one zone via prefix factoring."

| Fusion | Source A | Source B | Novel combination? |
|---|---|---|---|
| **ASOC stale-draft fallback × `GpuFuture::Join`** | plasma-tier `QuestDraftModel` (ships) | `riir-gpu-async` `Join` + `block_on` (ships) | **NEW (headline)** — `ComposerTick: GpuFuture` returns stale plasma draft on `Poll::Pending` |
| **Cross-entity chain compose × funcattn** | funcattn_compose (token-level only) | cross-entity `C_boss × C_quest × C_player` | **NEW** — `compose_chain(&[TransportOp]) -> TransportOp` |
| **Zone-batched prefix factoring** | `compose_chain` (above) | `papaya` lock-free `ShardIndex` (zone→shard) | **NEW** — `batch_compose_chain`: factor `C_qb` once, apply N× `C_player_i × C_qb` |
| **Decoder × direction vectors** | scalar_projection.rs (decoder-only, HLA-only) | gyrocalculus per-axis distance | **NEW (generalization)** — `direction_vector_decode(state, direction) -> action_score` for non-HLA directions |
| **Spectral audit verifier** | arxiv 2606.02427 tangent-Fourier | GOAT-gate discipline | **NEW** — Fourier-mode tangent operator check, the G6 verifier |
| ~~Analytic encoder × eggshell lanes~~ | ~~PJ-RoPE~~ | ~~Plan 335 SIMD lanes~~ | **NOT novel** — `fourier/encoder.rs` already ships closed-form encoding. Dropped. |

---

## 3. Verdict — **Super-GOAT**

| Q | Answer | Evidence |
|---|---|---|
| Q1 — No prior art? | **YES** | grep across all 5 repos, both layers (notes + code), both vocabularies (paper + codebase). Five primitives genuinely missing: `compose_chain`, `batch_compose_chain`, ASOC `ComposerTick`, generic `direction_vector_decode`, spectral audit. The original encoder-trait gap was a false positive (already ships in `fourier/encoder.rs`). |
| Q2 — New class of behavior? | **YES** | "Speculative GPU-cascade action selection with stale-draft fallback" — no incumbent ships non-blocking GPU-joined operator composition for autoplay bots. The closest shipped analog (`funcattn_compose`) is token-level, not cross-entity, and never returns a stale draft. |
| Q3 — Product selling point? | **YES** | "Our autoplay bot never blocks on GPU — every tick emits a plasma-tier action in nanoseconds, and speculatively joins three transport-operator rederive futures, falling back to the stale plasma draft on congestion." |
| Q4 — Force multiplier? | **YES** | Multiplies ≥8 shipped pillars: `riir-gpu-async` `GpuFuture` (the new substrate), `fourier/encoder.rs`, Plan 335 eggshell, latent_functor arithmetic, funcattn, latent_field_steering, viable_manifold_graph, riir-games plasma `QuestDraftModel`, riir-viz. |

**Selling point (one sentence, revised):** *The bot's action loop never blocks — every tick emits a plasma-tier draft in nanoseconds, then speculatively composes `C_boss × C_quest × C_player` cross-entity via `GpuFuture::Join`, falling back to the stale draft on GPU congestion.*

**Mandatory outputs (per workflow §1.5):**
1. ✅ Open primitive → `katgpt-rs/.plans/330_analytic_lattice_encoder_decoder_primitive.md` (this note's plan)
2. ✅ Private guide → `riir-ai/.research/162_analytic_lattice_encoder_decoder_game_runtime_guide.md`
3. ✅ Demo plan → `riir-ai/.plans/339_quest_manifold_bevy_isometric_demo.md`

---

## 4. Latent vs raw boundary (per AGENTS.md)

| Artifact | Domain | Sync? |
|---|---|---|
| Lattice vector `[x,y,z,f,g,h,...]` per entity | Latent (encoded geometry) | NO — local to entity |
| Direction vector for action decoding | Latent | NO |
| Composite transport operator `C_total` | Latent | NO |
| Action score scalar (the bot's chosen action rank) | Raw | YES — quorum-committed if part of replay |
| Encoded zone geometry (eggshell) | Latent | NO — derived artifact, BLAKE3-committed to source shard |
| Player position `MapPos {x,y}` | Raw | YES — bit-identical for anti-cheat |
| Player HLA 5-scalar affect | Raw | YES — synced scalar projections |

**Bridge rule:** the encoder takes raw inputs (player level, boss HP, quest reward table) and emits a latent lattice vector. The decoder takes latent state + direction and emits a raw scalar action score. The composite operator stays latent end-to-end; only the chosen-action scalar crosses sync.

---

## 5. Validation protocol (GOAT gate, in katgpt-rs)

- **G1 — Encoder determinism.** Same entity input → bit-identical lattice vector across ARM64/x86_64/wasm32. **Gate:** exact byte match.
- **G2 — Decoder ranking preservation.** For 100 random latent states, decode(state, direction) ranking matches a brute-force reference within cos ≥ 0.95. **Gate:** mean cos ≥ 0.95, worst ≥ 0.90.
- **G3 — Chain compose associativity.** `(A × B) × C ≈ A × (B × C)` within Frobenius ≤ 1e-5. **Gate:** max Frobenius ≤ 1e-5.
- **G4 — Sub-µs encode+decode+compose.** Single entity pipeline, wall-clock. **Gate:** < 1µs on release build.
- **G5 — Zero-alloc steady state.** `TrackingAllocator` audit. **Gate:** 0 allocations after warmup.
- **G6 — Spectral audit verifier.** Composite operator's tangent projected onto Fourier modes shows the intended phase-transport profile (no spurious mode coupling > 5%). **Gate:** max spurious coupling ≤ 5%.

**Promotion rule (per AGENTS.md):** all gates pass + gain is modelless → promote to `default`. If any gate fails, stay opt-in behind `analytic_lattice_encoder` feature flag.

---

## 6. What stays open vs private

| Component | Repo | Why |
|---|---|---|
| `compose_chain` operator product | katgpt-rs (open) | Generic k×k matmul, no game IP |
| `batch_compose_chain` zone-prefix factoring | katgpt-rs (open) | Generic prefix-amortization, no game IP |
| ASOC `ComposerTick: GpuFuture` cascade | katgpt-rs (open) | Generic orchestration over `GpuFuture`; plasma-draft trait is generic |
| `direction_vector_decode` SIMD (generic) | katgpt-rs (open) | Generic dot-product + sigmoid |
| Spectral audit verifier | katgpt-rs (open) | Generic Fourier-mode tangent check |
| ~~`AnalyticLatticeEncoder` trait~~ | ~~katgpt-rs~~ | **DROPPED — redundant with `fourier/encoder.rs`.** Reuse, do not re-ship. |
| Plasma-tier draft model (concrete game) | riir-ai (private) | Game design IP — ships as `riir-games::quest_draft::QuestDraftModel` |
| Quest/Zone/Boss/Player encoding schemas | riir-ai (private) | Game design IP |
| Game-specific direction vectors (per archetype) | riir-ai (private) | Game balance |
| Bevy isometric demo | riir-ai (private) | Showcase, not engine |

---

## 7. References

- Functional Attention (Xiao et al., ICML 2026) — `arxiv:2605.31559`
- PJ-RoPE (Zhang 2026) — `arxiv:2606.05345`
- Gyrocalculus on SPD (López et al., NeurIPS 2021) — `arxiv:2110.13475`
- Spectral Audit of Operator Networks (Gao/Yang/Karniadakis 2026) — `arxiv:2606.02427`
- Existing: Plan 335 (Zone Eggshell), Plan 312 (Viable Manifold Graph), Plan 286 (FuncAttn), Plan 309 (Latent Field Steering)

---

## TL;DR

Super-GOAT verdict: 4/4 novelty gate passes **(revised)**. The encoder half of
the original draft was a false-positive gap (already ships in `fourier/encoder.rs`);
the genuinely-novel contribution is the **ASOC cascade** (headline) plus
`compose_chain`, `batch_compose_chain`, generic `direction_vector_decode`, and
the spectral audit G6 verifier — all built on the already-shipped `GpuFuture` /
`Join` / `block_on` substrate in `riir-gpu-async`. Private guide in
riir-ai/.research/162; demo plan in riir-ai/.plans/339. GOAT gate G1–G6 must
pass before promotion to default.
