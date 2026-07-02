# Issue 007: Make katgpt-rs Cargo-consumable — Pillar Reorganization + HLA Substrate Extraction

> **Type:** Architecture / reorganization (unblocks cargo publish + kills cross-repo duplication)
> **Status:** **Substantively COMPLETE** (re-verified 2026-07-01). Phase 1 steps 1-7 done; Phase 2 dedup 2.2/2.3/2.5/2.6/2.7 done + 2.8 bit-identical PASS. **Re-audit (2026-07-01) found two formerly-deferred items are now DONE:**
> - **Step 3 `tokenizer`** — DONE as standalone `katgpt-tokenizer` crate (BPE/ToaST/ConvexTok; **no SentencePiece-sys dep at all** — the Q2 deferral concern was moot). Builds clean.
> - **Phase 4 `plotters` optional** — DONE behind the `plot` feature (Issue 355 Phase 2a). `plotters = { optional = true }` in root `Cargo.toml`.
> - **Phase E (undocumented in original issue)** — substrate extraction went further: 16 publishable leaf crates now exist (`katgpt-types`, `katgpt-hla`, `katgpt-transformer`, `katgpt-tokenizer`, `katgpt-speculative`, `katgpt-kv`, `katgpt-dec`, `katgpt-sense`, `katgpt-sleep`, `katgpt-spectral`, `katgpt-micro-belief`, `katgpt-personality`, `katgpt-attn-match`, `katgpt-pruners`, `katgpt-quant`, `katgpt-attn`). `katgpt-core` re-exports them (e.g. `pub use katgpt_hla as hla;`) for back-compat.
> - **2.1 hla role-aware** — DEFERRED BY DESIGN (not a defect): depends on private `role_transport` (Category C per 003). `katgpt-hla` lib docs document this split explicitly.
> - **Q5 RESOLVED (2026-07-02, policy A — lock down):** all 18 substrate leaf crates now carry `publish = false`, matching the established `release-plz.toml` policy ("only katgpt-core ships to crates.io"). The 11 crates previously defaulting to `publish = true` (katgpt-dec, katgpt-hla, katgpt-kv, katgpt-micro-belief, katgpt-personality, katgpt-sense, katgpt-sleep, katgpt-spectral, katgpt-speculative, katgpt-transformer, katgpt-types) are now consistent with the 7 that already had the line. See §Open questions Q5 for the resolution record.
> - katgpt-core lib tests: **661 passed / 0 failed** (re-verified 2026-07-01, isolated `CARGO_TARGET_DIR`).
> Phase 3 (cosmetic root subdir reorg) still deferred. Phase 5 (publish root) RESCINDED. See [Plan 008](../.plans/008_katgpt_core_substrate_extraction.md) for the full execution log + GOAT gates.
>
> **Phase F ACCEPTED + FULLY RESOLVED (2026-07-02):** `katgpt-forward` crate created + `ForwardContext` lifted (F.1–F.3); 4 of 26 audited composition files migrated to leaves (F.4a `gdn2`/`dash_attn` → katgpt-attn, F.4b `hla` → katgpt-forward; commit `c76722d2`, GOAT green). The remaining 22 audited files are pinned by a *deeper* join point (`crate::transformer::forward` the function, not just the type) — **resolved all-Option-C in [Issue 033](./033_forward_join_point_architectural_decision.md): each documented root-resident by design.** Phase F's structural goal (break the `ForwardContext` DAG cycle so substrate leaves are consumable without root) is fully achieved. (Original filing estimated 34/30; empirical audit corrected to 26/22 — see acceptance §Phase F below.)
> **Composition-layer pin — DIAGNOSED (2026-07-02), see §'The composition-layer pin' below.** The substrate dedup (Phase 1/E) is done — one definition per module lives in a leaf crate, root re-exports it. BUT **34 composition files** (`dash_attn/forward.rs`, `gdn2/forward.rs`, `hla/forward.rs`, `speculative/*`, `sleep/consolidation.rs`, benchmarks…) remain structurally pinned to root `src/` because they take `&mut ForwardContext`, and `ForwardContext` is the **DAG join point** that references both transformer substrate types AND 3 pruner types (`CnaModulator`, `SubstrateMask`, `HydraSkipPlan`). Those 3 pruner types already live in `katgpt-pruners`, which **already depends on `katgpt-transformer`** — so `ForwardContext` cannot move into either leaf without a cycle. **Resolution adopted (Option 2): new `katgpt-forward` crate on top of both, hosting `ForwardContext` + the composition layer. See §'The composition-layer pin' + Phase F below.** This is the structural fix that lets the 34 composition files finally `mv` out of root `src/`. *Note: the user's `dash_attn` example is NOT a copy-paste dupe — root's `src/dash_attn/mod.rs` is already a re-export shim (`pub use katgpt_attn::dash_attn::{chunk_summary, entmax, routing}`); what stays in root is `forward.rs` (composition), pinned by `ForwardContext`.*
> **Audit findings (2026-06-27):**
> - **Phase 5 RESCINDED** — `Cargo.toml:9` + `release-plz.toml:9-12` lock `katgpt-rs` root as `publish = false` permanently ("dev/examples aggregator — never published. Only katgpt-core ships to crates.io"). Decision was made AFTER this issue was filed and overrides its Phase 5.
> - **Phase 1 step 1 (`types` move) ALREADY DONE** — `katgpt-core/src/types/` has 14 files; root `src/types.rs` is a thin re-export shim.
> - **Phase 1 step 2 (`transformer` substrate) ✅ DONE 2026-06-27** — pure data types (LayerWeights, TransformerWeights, KV caches, PrefillContext, WallPrefixState, MtpProjection, ContiguousWeights) moved to new `katgpt-transformer` crate (separate from `katgpt-core` per the audit: forward functions stay in root because they compose cognitive modules — `crate::hla`, `crate::sleep`, `crate::tf_loop`, `crate::gdn2` — that don't exist in substrate; `ForwardContext` also stays because its fields reference root-only pruner types).
> - **Phase 2B premise INVERTED** — `cgsp` is already in core (correctly), `cce` is in root (correctly); no tier move needed.
> - **What's still real:** the cross-repo DRY violation for `hla`, `tokenizer`, `dd_tree`, `spec_types`, `mcts`, `sampling`, `delta_mem`, `simd` — `riir-engine/src/` still has divergent `crate::` copies.
> **Owner:** develop
> **Created:** 2026-06-27
> **Cross-repo:** katgpt-rs (primary), riir-ai, riir-neuron-db (consumers). riir-train/riir-chain unaffected.
> **Origin:** User directive — "I want others to use it as easily as possible aka cargo" + HLA scattering concern.
> **References:** [Plan 008](../.plans/008_katgpt_core_substrate_extraction.md) (executable scope) · [Issue 006](./006_x86_64_simd_target_feature.md) (x86_64 gate, now cleared) · `.research/28_Higher_order_Linear_Attention.md`

---

## TL;DR

Two problems, one fix:

1. **`katgpt-rs` (root) isn't cargo-consumable** because its public surface is ~100 flat feature-gated modules with heavy non-optional deps (`plotters`, platform `metal`/`coreml`). Anonymous consumers can't `cargo add` it.
2. **The inference substrate is duplicated across repos.** `hla` (Higher-order Linear Attention) — a pillar — lives in `katgpt-rs/src/hla/`, is **copy-pasted verbatim into `riir-ai/crates/riir-engine/src/hla/`** (same `forward_hla`/`MultiLayerHlaCache` signatures), and is stored as opaque `[f32; 8]` in `riir-neuron-db::NeuronShard`. Same goes for `transformer` and `types` — `riir-engine/src/hla/forward.rs` imports `crate::transformer::{ForwardContext, TransformerWeights}` and `crate::types::{Config}`, proving those are duplicated too.

**Fix:** move the inference substrate (the pillars every repo needs) down into `katgpt-core`, the publishable leaf. Organize the root crate into tiers. Then publish `katgpt-rs` with a small stable default surface + opt-in experimental features.

This is the single change that makes the engine cargo-consumable AND eliminates the cross-repo DRY violation at the substrate layer.

---

## Smell Audit (full inventory — HLA was the tip)

**Root cause found:** `riir-ai/crates/riir-engine/src/lib.rs:3` documents it itself — *"Extracted from katgpt-rs (MIT, frozen at v0.1.0)"*. riir-engine is a **fork** of katgpt-rs@v0.1.0. Both sides then evolved independently for hundreds of commits. Everything present at v0.1.0 got duplicated; nothing keeps the copies in sync. HLA is just the first one I confirmed.

### Category A — Substrate duplicated in riir-engine (the smells to fix)

These modules live in `riir-engine/src/` as **own copies** (use `crate::`, not `katgpt_core::`). Each is inference mechanics — the WHAT, public per the 003 strategy — stuck in the private fork:

| Module (riir-engine path) | Uses | Smell type | Strategy-doc status |
|---|---|---|---|
| `src/hla/` | `crate::hla::kernel`, `crate::transformer`, `crate::types` | Full pillar duplicated, diverged (`*_role_aware` variants) | inference mechanics — PUBLIC |
| `src/transformer/` (gemma2, llama) | `crate::types::{self, *}` | Forward pass duplicated | inference mechanics — PUBLIC |
| `src/types.rs` | — | Foundation (Config/Rng/InferenceResult) duplicated | inference mechanics — PUBLIC |
| `src/tokenizer.rs` | `crate::types` | SentencePiece loader duplicated | inference mechanics — PUBLIC |
| `src/dd_tree.rs` | `crate::spec_types::{ConstraintPruner, TreeNode, ...}` | DDTree duplicated | **named PUBLIC engine in 003** ("DDTree, ConstraintPruner trait") |
| `src/spec_types.rs` | — | `TreeNode`/`DDTreeBranchCache`/`SpeculativeContext` duplicated; **traits already moved to `katgpt-core/src/traits.rs`** but the dependent types were left behind — half-finished extraction | PUBLIC |
| `src/mcts.rs` | `crate::game_state` | MCTS mechanics duplicated | inference mechanics — PUBLIC |
| `src/sampling.rs` | `crate::types::Rng` | CDF sampling duplicated | inference mechanics — PUBLIC |
| `src/delta_mem/` | `crate::spec_types` | DeltaMemory duplicated | inference mechanics — PUBLIC |
| `src/turboquant.rs` | `crate::types` | **Stub by design** — its own header says *"real impl lives in katgpt-rs, port when needed"*. Deferred dedup, acknowledged. | inference mechanics — PUBLIC |
| `src/simd/wasm32.rs` | reimplements `simd_dot_f32`/`simd_sum_f32`/`simd_exp_inplace` etc. | WASM SIMD128 path **reimplemented** despite `katgpt-core/src/simd/` already shipping the same kernels (`wasm32_simd128_dot_f32` etc.) | inference mechanics — PUBLIC |

### Category B — Correct composition (the pattern to replicate everywhere)

These already do it right — `use katgpt_core::` / `use katgpt_rs::` to compose public primitives, keeping only the private wiring. This is the target end state for all of Category A:

| Module | Imports from | Why it's correct |
|---|---|---|
| `analytic_lattice/asoc.rs` | `katgpt_core::analytic_lattice::{ComposerCtx, ...}` | Composes the public primitive |
| `arg_runtime/*` (6 files) | `katgpt_core::arg::{LabelId, TaxonomyNode, ...}` | `_runtime` suffix = private composition layer |
| `bom_arena/*` | `katgpt_core::{ArenaAction, BeliefPlanner, ...}` | Public primitives, private game wiring |
| `cce_runtime/*` | `katgpt_rs::cce`, `katgpt_core::cgsp` | Public primitives, private per-NPC wiring |
| `cgsp_runtime/*` | `katgpt_core::cgsp::{Direction, sigmoid}` | Public primitive, private anti-cheat/bridge |

The `_runtime` suffix is already the **de-facto boundary marker** in this codebase: bare-name module = public primitive (should live in core); `*_runtime` module = private composition (correctly in riir-engine). The fix is to make every Category A module follow the Category B pattern.

### Category C — Correctly private (must NOT move — 003 strategy compliance)

These are game product IP per 003 §"What riir-ai Can Do". The audit confirms they correctly stay in riir-engine: `adapters/`, `arena/`, `bom_arena/` (game wiring), `cce_runtime/`, `cgsp_runtime/`, `cognitive_branches_runtime/`, `committed_blend/`, `cwm_runtime/`, `entity_cognition/`, `ict_runtime/`, `neuron_vessel_runtime/`, `policy_cache/`, `swir_validation/`, `zone/`, `kg*`, `game_state.rs`, `latent_field_wiring.rs`, `role_transport.rs`. Moving any of these to the public crate would violate the "How = private" rule.

### Tier inconsistency smell (separate from duplication)

`cgsp` lives in `katgpt-core` but `cce` lives in `katgpt-rs` root. Consumers reach them inconsistently: `use katgpt_core::cgsp` vs `use katgpt_rs::cce`. Same conceptual layer (game-theoretic runtime primitives), two different tiers. `cce` should move down to core to match `cgsp`, or both move up — pick one tier for the "public game-theory primitive" layer.

### Net assessment

- **10 substrate modules duplicated** in riir-engine (Category A), all PUBLIC inference mechanics per the 003 strategy. This is a strategy-doc violation in the OTHER direction: public WHAT is stranded in a private fork, forcing the fork to maintain its own divergent copies.
- The `*_runtime` convention already marks the boundary correctly. The codebase knows the pattern; it just hasn't been applied to the v0.1.0 fork residue.
- The single biggest DRY win is the substrate chain `types → transformer/weights → hla → dd_tree/spec_types`: move it to core, delete the riir-engine copies, and every downstream repo gets one canonical source.

### Cognitive/reasoning is a NEW MOAT (the basic/GOAT split)

A category of primitives is emerging as a competitive moat beyond pure inference mechanics: **cognitive and reasoning** primitives — `cce` (correlated equilibrium), `cgsp` (curiosity-guided self-play), `clr` (claim-level reliability), `compaction`, `claim_rubric`, etc. These are the decision-level / reasoning-layer mechanisms.

The commercial split for this layer (per refined strategy doc):

- **Basic version stays PUBLIC in `katgpt-rs` root** (engine tier, not core substrate). The adoption funnel — good enough to build on, attract dependency, demonstrate the capability. Ships WITH its examples/benches/related `.md` so the public surface is legible and evaluable.
- **GOAT/Super-GOAT tuned version stays PRIVATE in `riir-*`** (the `_runtime` modules + tuned parameters + game-coupled extensions). This is the version that actually wins — the validated thresholds, the collapse-recovery tuning, the game-specific wiring. "Good enough to adopt, not good enough to win."

This is why `cce`/`cgsp` belong in root (not core): they're cognitive-layer adoption primitives, not pure substrate. And it's why their `_runtime` siblings (`cce_runtime`, `cgsp_runtime`) correctly stay private in riir-engine — the GOAT tuning is the moat. The `*_runtime` convention already encodes this split; this issue just makes it explicit in the tier model.

**Implication for the reorg:** do NOT push cognitive/reasoning primitives down to core. Core = pure inference substrate only. Root = substrate re-exports + cognitive basics + engine primitives. riir-* = GOAT versions + composition + game/chain/shard IP.

---

## Evidence: HLA is duplicated, not just scattered

### Where HLA lives today

| Repo | File | What it has | How it's used |
|---|---|---|---|
| `katgpt-rs` (root) | `src/hla/{mod,types,kernel,forward}.rs` | Full pillar: `HlaLayerState`, `hla_state_update`, `hla_readout`, `forward_hla`, `forward_ahla`, `generate_hla_into`, AHLA + Parallax variants | The "canonical" copy |
| `riir-ai` | `crates/riir-engine/src/hla/{forward,...}.rs` | **Same signatures**: `forward_hla`, `forward_ahla`, `generate_hla_into`, `MultiLayerHlaCache` | Active runtime — `karc_runtime`, `committed_personality_runtime`, `latent_field_wiring` features all wire into "the HLA update loop" |
| `riir-neuron-db` | `src/index.rs`, `NeuronShard.hla_moments` | Opaque `[f32; 8]` field only — no kernel | Stores HLA moments as shard embedding for `ShardIndex` retrieval |
| `katgpt-core` | (comments only) | Doc references in `analytic_lattice`, `babel_codec`, `cgsp::HlaProjectionGuide`, `branching` | Does NOT contain the kernel. `HlaProjectionGuide` borrows the name but is a generic `QualityGuide` over abstract `Direction`s |

### The smoking gun

`riir-ai/crates/riir-engine/src/hla/forward.rs:15-19`:
```rust
use crate::hla::kernel::{ahla_layer_step_role_aware, hla_layer_update_role_aware};
use crate::hla::types::{MultiLayerAhlaCache, MultiLayerHlaCache};
use crate::transformer::{ForwardContext, TransformerWeights};
use crate::types::{self, Config};
```

The `crate::` prefix means riir-engine has its **own** `hla/`, `transformer`, and `types` modules — duplicated from katgpt-rs, not imported from it. The HLA substrate was copy-pasted, then both sides evolved independently (riir-engine added `*_role_aware` variants; katgpt-rs may have diverged elsewhere). This is a silent DRY violation: two sources of truth for the same pillar, no mechanism keeping them in sync.

### Why it's structured this way (the coupling trap)

`katgpt-core` is a clean leaf (minimal deps, the SIMD/types substrate). `hla` was placed in the **root** crate because it depends on `transformer::{ForwardContext, TransformerWeights}` and `types::Config` — which are ALSO in the root crate. So moving HLA down requires moving the transformer substrate down first (or together). The pillar stack — `types` → `transformer/weights` → `hla` — is a dependency chain that all lives in the root, forcing every compute consumer to pull the whole root.

---

## The composition-layer pin (newly diagnosed 2026-07-02)

Phase 1/E extracted the **substrate** (kernels, pure types) to 18 leaf crates, and root `lib.rs`/`mod.rs` files were converted to re-export shims (`pub use katgpt_attn::dash_attn::{chunk_summary, entmax, routing}`). That dedup IS done — verified: every same-name module dir in root references its leaf; zero blind copy-paste remains.

**But the composition layer is structurally pinned to root.** A second tier of files — `src/dash_attn/forward.rs`, `src/gdn2/forward.rs`, `src/hla/forward.rs`, `src/speculative/{step,prefill,dflash,verifier,...}.rs`, `src/sleep/consolidation.rs`, `src/benchmark/*`, `src/inference_backend.rs`, etc. (34 files) — cannot move into their leaf crates because they all take `&mut ForwardContext`, and `ForwardContext` is the **join point** at the top of the dependency DAG.

### Why `ForwardContext` can't move to either existing leaf

`ForwardContext` (`src/transformer.rs:59`) holds:
- ~40 transformer buffers (`x`, `q`, `k`, `v`, `scores`, `attn_out`, `kv_group_lut`, `wall_prefix: WallPrefixState`, …) — pure substrate, lives in `katgpt-transformer`.
- 3 pruner handles behind feature gates:
  - `cna_modulator: Option<crate::pruners::CnaModulator>` (`cna_steering`)
  - `substrate_mask: Option<crate::pruners::SubstrateMask>` (`substrate_gate`)
  - `hydra_skip_plan: Option<crate::pruners::HydraSkipPlan>` (`hydra_budget`)

Those 3 types already live in `katgpt-pruners` (verified: `cna.rs:90`, `substrate_types.rs:18`, `hydra_budget.rs:69`). And **`katgpt-pruners` already depends on `katgpt-transformer`** (`katgpt-pruners/Cargo.toml`). So:

- Move `ForwardContext` → `katgpt-transformer`: would require `katgpt-transformer` to depend on `katgpt-pruners` → **cycle** (`katgpt-transformer` → `katgpt-pruners` → `katgpt-transformer`).
- Move `ForwardContext` → `katgpt-pruners`: inverts the layering (a pruner crate would own the canonical transformer context), and pruners shouldn't own forward-pass buffers.

`ForwardContext` is therefore the **topmost type** — it sits above BOTH transformer substrate and pruners, and can only live in a crate that depends on both. Today that crate is the root.

### Why a simple `mv` doesn't work (the user's `dash_attn` example)

`src/dash_attn/forward.rs` does `use crate::transformer::ForwardContext` and defines `forward_dash_attn_prefill(ctx: &mut ForwardContext, …)`. If it moves into `katgpt-attn`, then `katgpt-attn` needs `ForwardContext`, which needs the pruner types, which need `katgpt-transformer`… → cycle. The same chain pins all 34 composition files. **The fix is to lift `ForwardContext` into a new top-tier crate so the composition layer can move with it (or beside it).**

### Resolution: Option 2 — new `katgpt-forward` crate

A thin top-tier crate, sitting above `katgpt-transformer` + `katgpt-pruners`, hosting:
1. `ForwardContext` (struct + `new()` + `reset_dequant` + simple accessors) — `pub(crate)` fields become `pub`.
2. The `depth_route_with_indices` helper (currently `src/transformer.rs:1694`, called by `ForwardContext::depth_route_blocks`) moves here too.
3. Composition files migrate per-leaf in dependency order (Phase F steps below).

DAG after the fix:

```
katgpt-forward ──► katgpt-pruners ──► katgpt-transformer ──► katgpt-core
      │                  ▲
      └─► composition files (dash_attn/gdn2/hla/speculative/…) move
          into their respective leaves, which then depend on katgpt-forward
```

The three considered alternatives (rejected):
- **(1) Trait-abstract the 3 pruner fields** (`Box<dyn Any>` / generic param): ugly type erasure, or threads a generic through every call site. Rejected.
- **(3) Keep composition in root (status quo):** leaves the 34-file mess in `src/` and doesn't fulfill the cargo-consumable goal. Rejected.

---

## Proposed reorganization

### Tier 0 — `katgpt-core` (the leaf, already on crates.io)

Move the **inference substrate** down here. These are the pillars every repo needs, with minimal deps, no game/application code:

```
crates/katgpt-core/src/
├── simd/           # ALREADY HERE
├── types.rs        # Config, Rng, etc. — MOVE FROM root src/types.rs
├── transformer/    # ForwardContext, TransformerWeights — MOVE FROM root
├── weights.rs      # MOVE FROM root
├── tokenizer/      # MOVE FROM root (if leaf-clean)
├── hla/            # MOVE FROM root src/hla/ — the case-study pillar
├── (existing core primitives: dec/, arg/, cgsp/, committed_field_blend/, ...)
```

**Migration rule for what moves to core:** a module moves down if (a) it's a pillar that riir-ai/riir-neuron-db need for *compute*, (b) it has no heavy/platform deps, (c) moving it doesn't create a cycle. `hla`, `transformer`, `types`, `weights` clearly qualify. `tokenizer` — verify deps first.

**What stays OUT of core:** anything that pulls `rayon`/`bevy_ecs`/`wasmi`/`plotters`/`metal`/`good_lp`. Those are engine/app concerns.

### Tier 1 — `katgpt-rs` (root, the engine — becomes publishable)

Organize the remaining ~100 flat modules into subdirs by role. **Mirror the existing `_runtime` convention** as the organizing principle — bare-name = public primitive (re-export from core), `*_runtime`-style suffix = composition layer:

```
src/
├── lib.rs
├── primitives/         # GOAT-gated research primitives (the WHAT), each feature-flagged.
│                       # Candidates to also push DOWN to core over time: cce, clr, compaction
│                       # (matches cgsp already being in core). Until then, live here as public.
│   ├── cce/            # tier-inconsistency candidate (see Smell Audit) — push to core to match cgsp
│   ├── clr/  compaction/  cgsp.rs  claim_rubric/  ...
│   └── (GOAT-passed default-on primitives form the stable publishable surface)
├── inference/          # higher-level inference wiring built ON core substrate
│   ├── attn_match/  speculative/  pruners/  still_kv/  turboquant/
│   └── (these are engine-tier, not substrate — depend on core's transformer/hla)
├── games/              # game engines + NPC brains — clearly app-level, opt-in features
│   ├── percepta/  bomber/  go/  sudoku/  monopoly/  npc_brain_router.rs
│   └── (NOT in katgpt-core — game IP stays public-engine per 003, but separate tier)
├── backends/           # platform backends (optional, target-gated) — gpu/ane/inference_router
└── bench/              # benchmark harnesses
```

**Critical:** after the substrate (Category A) moves to core, root `src/hla/`, `src/transformer.rs`, `src/types.rs`, `src/weights.rs`, `src/dd_tree.rs`, `src/spec_types.rs`, `src/mcts.rs`, `src/sampling.rs`, `src/tokenizer/`, `src/delta_mem/` all become thin `pub use katgpt_core::{...};` re-exports. No call site in katgpt-rs or its examples breaks. riir-engine deletes its copies and imports from core the same way `analytic_lattice` already does.

This is a **pure move + re-export** refactor — no logic changes. `lib.rs` keeps re-exporting at the top level so existing `use katgpt::clr_vote` call sites don't break.

### Feature-flag stability tiers (enables cargo publish)

Reuse the existing ~100 feature flags as the stability contract:

| Tier | Examples | Default | Semver promise |
|---|---|---|---|
| **Stable** | `simd`/`hla`/`transformer` (in core), core re-exports | ON | Breaking = major bump |
| **Engine** | `attn_match`, `compaction` (default-on, GOAT-passed) | ON | Best-effort, breaking = minor in 0.x |
| **Experimental** | most research primitives (opt-in) | OFF | No promise behind the flag |

This is exactly how `tokio`/`bevy` publish while churning. Default-off features can break freely; default-on is the curated surface.

---

## Making `katgpt-rs` publishable (the cargo goal)

After reorg, the remaining blockers to `cargo add katgpt-rs`:

1. **Audit non-optional deps.** Most heavy ones are already `optional = true` (`bevy_ecs`, `wasmi`, `good_lp`, `reqwest`, `rustfft`). **`plotters` is the blocker** — make it optional (only `plot.rs` + benches use it). `rayon`/`blake3`/`half`/`bytemuck`/`serde*`/`postcard`/`toml` are fine (small, leaf-ish, broadly acceptable).
2. **Platform deps stay target-gated** (`metal`/`coreml-native` under `[target.'cfg(target_os = "macos")']` — already correct, no change).
3. **Scrub hard `riir-*` code deps** from public files (name-drops for bragging are fine per user; only real `use riir_*` / path deps into private repos must go — there shouldn't be any in the public crate, but verify).
4. **release-plz config**: add `katgpt-rs` as a second published package with its own `git_tag_name = "katgpt-rs-v{{version}}"`. Versions stay **independent** (core evolves on its own semver; root starts at `0.1.0`). Do NOT couple versions — that was the earlier-discarded idea.
5. **x86_64 verifies clean** ✅ (Issue 006 cleared this for core; root crate will inherit once it publishes).

---

## Cross-repo consumer cleanup (the DRY payoff — consolidate, don't blindly delete)

Once the substrate is in `katgpt-core`, the riir-engine copies get retired — but **consolidation, not deletion**. riir-engine forked at v0.1.0 and some copies DIVERGED WITH IMPROVEMENTS. The rule:

- **For each Category A module:** diff the riir-engine copy against the new core canonical. If riir-engine added anything (a variant, an optimization, a bug fix, a `*_role_aware` extension) → PORT it into core first, behind a feature flag if needed. Only then delete the riir-engine copy.
- **Known divergence to consolidate:**
  - `hla/` — riir-engine added `forward_hla_role_aware` / `forward_ahla_role_aware` + `role_transport` wiring. Port the role-aware kernel variants into core's `hla` (behind `hla_role_aware` feature); keep riir-engine's `role_transport.rs` as the private composition (Category C).
  - `dd_tree`/`spec_types` — riir-engine's copy may have game-coupled additions; port the generic parts, leave game-specific in riir-engine.
  - `turboquant.rs` — riir-engine has a STUB; katgpt-rs has the real impl. Consolidation = riir-engine consumes the real one, stub deleted.
  - `simd/wasm32.rs` — riir-engine reimplemented WASM SIMD128; katgpt-core already ships it. Diff for any riir-engine-only kernel improvements, port if any, then delete the reimplementation.

**After consolidation:**
- **riir-ai/riir-engine**: deletes every Category A copy, imports from `katgpt_core` the same way `analytic_lattice`/`arg_runtime` already do. Zero divergent copies remain.
- **riir-neuron-db**: unchanged structurally (still stores `[f32; 8]`), but can now optionally call `katgpt_core::hla` kernels if it ever needs compute, without pulling the root engine.
- **katgpt-rs root**: `src/hla/`, `src/transformer.rs`, `src/types.rs`, `src/weights.rs`, `src/dd_tree.rs`, `src/spec_types.rs`, `src/mcts.rs`, `src/sampling.rs`, `src/tokenizer/`, `src/delta_mem/` become thin `pub use katgpt_core::{...};` re-exports (back-compat for existing call sites).

---

## Migration path (incremental, no big-bang)

Each phase is independently shippable and reversible:

- [ ] **Phase 1 — Substrate extraction to core.** Move the full Category A chain, in dependency order (each move is its own commit, full test suite green before next):
  1. `types` (foundation — move first, surfaces any hidden deps)
  2. `transformer` + `weights` (depends on types)
  3. `tokenizer` (depends on types — audit deps first, see Risk 4)
  4. `hla` (depends on transformer/types — port `*_role_aware` variants behind a core feature flag)
  5. `dd_tree` + `spec_types` types (traits already in `core/traits.rs`; move the dependent `TreeNode`/`DDTreeBranchCache`/`SpeculativeContext` to join them)
  6. `mcts`, `sampling`, `delta_mem` (leaf inference mechanics)
  7. Delete `riir-engine/src/simd/wasm32.rs`, consume `katgpt_core::simd` wasm32 path instead
  Each step: copy to core, `pub use katgpt_core::*` re-export at root, run tests. Core version bump: `0.3.0`.
- [ ] **Phase 2 — Cross-repo dedup.** In riir-ai/riir-engine, delete every Category A copy, import from `katgpt_core` the same way `analytic_lattice`/`arg_runtime` already do. Verify `forward_hla`/`dd_tree` bit-identical on existing tests in both repos. This is the single biggest DRY win.
- [ ] **Phase 2b — Cognitive/reasoning tier consistency (move UP, not down).** `cce` is in root, `cgsp` is in core — but both are cognitive/reasoning primitives, not substrate. **Move `cgsp` UP from core to root** to join `cce`, bringing its examples/benches/related `.md` along. Root becomes the home of the public cognitive/reasoning layer; core stays pure inference substrate (SIMD/types/transformer/hla/dd_tree). Do NOT push `cce` down to core — that was the wrong direction. Tier model: core = substrate, root = engine + cognitive basics, riir-* = GOAT/Super-GOAT tuning + composition.
- [ ] **Phase 3 — Root crate reorg.** Move root `src/*` into `primitives/`/`inference/`/`games/`/`backends/` subdirs per the `_runtime` convention. Top-level `pub use` re-exports preserve all call sites. Pure refactor.
- [ ] **Phase 4 — Dep audit for publish.** Make `plotters` optional. Verify `cargo check --no-default-features` clean on root.
- [ ] **Phase 5 — Publish katgpt-rs.** Add to `release-plz.toml` as second package (`git_tag_name = "katgpt-rs-v{{version}}"`), first publish `0.1.0`. Document feature-flag stability tiers in README.
- [x] **Phase F — Composition-layer unblock (new `katgpt-forward` crate).** ACCEPTED + FULLY RESOLVED (2026-07-02). Lifts the composition layer's shared join point (`ForwardContext`) into a top-tier crate. Original target was ~34 files; empirical audit found 26 audited (4 migrated to leaves, 22 declared root-resident by design per [Issue 033](./033_forward_join_point_architectural_decision.md) Option C). The cycle-breaking structural goal is fully achieved. See acceptance §Phase F below for the full audit + GOAT gate. Steps (each its own commit, build green before next):
  1. **Scaffold `katgpt-forward` crate.** New crate under `crates/katgpt-forward/`. Deps: `katgpt-transformer` (for buffer types + `WallPrefixState`), `katgpt-pruners` (for the 3 pruner types), `katgpt-core` (for `Config`/SIMD). Add the ~11 feature gates `ForwardContext` needs (`cna_steering`, `sparse_mlp`, `substrate_gate`, `delta_routing`, `coda_fusion`, `mls_aggregate`, `tiled_attention`, `tf_loop`, `hydra_budget`, `wall_attention`, `turboquant`) — forwarding to the upstream crates where applicable. Root `Cargo.toml` adds `katgpt-forward` dep + forwards the features.
  2. **Move `ForwardContext` + `depth_route_with_indices` into `katgpt-forward`.** `pub(crate)` fields → `pub`. `crate::pruners::{CnaModulator,SubstrateMask,HydraSkipPlan}` → `katgpt_pruners::{…}`. `types::kv_dim`/`DepthTier` → `katgpt_types`/`katgpt_core`. `WallPrefixState` → from `katgpt-transformer` (already there). Root `src/transformer.rs` keeps `forward*` functions + `attention_head` (the 33 real forward passes that compose cognitive modules) and re-exports `ForwardContext` from the new crate.
  3. **Verify build** — `cargo check --workspace` + `cargo test -p katgpt-core --lib` + root lib tests green. This is the GOAT gate for the crate move.
  4. **(Parallel subagents, disjoint write sets) Migrate composition files into their leaves.** Each leaf adds a `katgpt-forward` dep, the composition file(s) `mv` in, `crate::transformer::ForwardContext` → `katgpt_forward::ForwardContext`. Batches:
     - **F.4a — `katgpt-attn`:** `dash_attn/{forward,tests,sat_analysis,...}.rs`, `gdn2/{forward,...}.rs`.
     - **F.4b — `katgpt-hla`:** `hla/forward.rs`.
     - **F.4c — `katgpt-speculative`:** `speculative/{step,prefill,dflash,verifier,d2f_verifier,drafter_lora,flashar_*}.rs` (feature-gated `dd_tree` variants stay root — they reference root-only siblings).
     - **F.4d — `katgpt-sleep`:** `sleep/consolidation.rs`.
     - **F.4e — `katgpt-forward` itself:** `inference_backend.rs`, `inference_router.rs`, `sp_kv_forward_mod.rs`, `fold/`, `benchmark/*` (these are generic engine composition, belong with `ForwardContext`).
  5. Root `src/` shrinks to: `lib.rs` + the 33 `transformer.rs` forward passes (still compose root-only cognitive modules: `tf_loop`, `gdn2`, `hla` re-exports, cognitive primitives `cce`/`clr`/`compaction`/…) + game IP (`pruners/bomber/`). That residual is the *engine tier* by design — Phase 3 subdir reorg handles its cosmetics.

Phases 1–2 are the high-value, low-risk core (kills the duplication, unblocks clean consumption). Phases 3–5 are the cargo-publish polish. **Phase 1+2 alone deliver most of the value** — any repo can then `cargo add katgpt-core` and get the full inference substrate including HLA, DDTree, transformer.

---

## Risks

1. **Moving `transformer`/`types` to core may surface hidden deps** (e.g., `Config` referencing something in root). **Mitigation:** move `types` first (it's the dependency root), find out, deal with it incrementally. Don't move the whole chain in one commit.
2. **riir-engine's HLA diverged** (`*_role_aware` variants, `role_transport` wiring). **Mitigation:** role-aware is likely a superset — port the HLA kernel variants into core's `hla` behind a feature flag; keep riir-engine's `role_transport.rs` as the private composition layer (it's Category C). Phase 2 reconciliation.
3. **Version churn.** Core goes to `0.3.0` (new modules), root starts `0.1.0`. **Mitigation:** both are `0.x`, expected to churn. Document in READMEs.
4. **`tokenizer` may have deps that disqualify it from core** (SentencePiece C++ via `sentencepiece-sys`). **Mitigation:** audit first; if it pulls a C++ build dep, leave `tokenizer` in root and only move the trait/types. The riir-engine `tokenizer.rs` is already `#[cfg(not(target_arch = "wasm32"))]`-gated — core must preserve that.
5. **`dd_tree`/`spec_types` reconciliation** — riir-engine's copy may have diverged from whatever katgpt-rs root has (root `spec_types.rs` doesn't even exist per the audit). **Mitigation:** treat core as the new canonical source; port any riir-engine-only improvements during Phase 2; the traits are already in core so the hard part (the trait boundary) is done.
6. **Game-state coupling** — `mcts.rs` imports `crate::game_state::GameState`, which is Category C (game IP). **Mitigation:** `mcts` the algorithm (tree policy, UCB1, backprop) is public mechanics; `GameState` the trait stays wherever it is. Move the generic MCTS, parameterize over a `Game` trait from core if needed, leave game-specific impls in riir-engine.
7. **`ForwardContext` cycle (Phase F).** `ForwardContext` references 3 pruner types that live in `katgpt-pruners`, which already depends on `katgpt-transformer`. Moving it into either existing leaf creates a cycle. **Mitigation:** the new `katgpt-forward` crate sits ABOVE both (Option 2). Verified: `katgpt-pruners` has no `ForwardContext`/`TransformerWeights` field-level references that would re-introduce a back-edge (only two files reference the names, in comments/players — re-audit before F.2). **Visibility churn:** ~40 `pub(crate)` fields become `pub` — audit that no field was deliberately hidden as an invariant (the struct is a pre-allocated scratch buffer, so wide visibility is acceptable).

---

## Acceptance

- [x] **Phase 1:** all Category A substrate modules live in `katgpt-core` (or sibling crates), re-exported from root. `cargo test -p katgpt-core --lib` + `cargo test --lib` green. **Step 3 `tokenizer` DONE** (re-audit 2026-07-01): extracted as standalone `katgpt-tokenizer` crate — BPE/ToaST/ConvexTok, no SentencePiece-sys dep (the Q2 concern was moot).
- [~] **Phase 2:** riir-engine Category A dedup — **core path DONE**, bit-identical verification (2.8) PASS 2026-07-01. One item deferred by design: 2.1 hla role-aware (Category C — depends on private `role_transport`). 2.4 tokenizer unblocked now that Step 3 is done (`katgpt-tokenizer` exists) but the riir-engine `src/tokenizer.rs` dedup pass itself was not separately re-run in this re-audit; tracked as a non-blocking follow-up. `forward_hla`/`dd_tree`/`mcts` tests bit-identical.
- [-] **Phase 2b:** RESCINDED — premise inverted (see audit finding). `cgsp` correctly in core, `cce` correctly in root; no tier move needed.
- [-] **Phase 3:** DEFERRED (cosmetic root subdir reorg — not worth churn while 100+ features flatten at root).
- [-] **Phase 4:** DONE — re-audit 2026-07-01: `plotters = { version = "0.3", optional = true }` behind the `plot` feature (Issue 355 Phase 2a). `cargo check --no-default-features` clean. (Kept `[-]` checkbox because the acceptance line originally framed it as deferred; the work is complete — see Issue 355 Phase 2a for the GOAT.)
- [-] **Phase 5:** ~~RESCINDED~~ — conflicts with `Cargo.toml:9` + `release-plz.toml:9-12` decision to keep root private permanently. Only `katgpt-core` ships.
- [x] **Phase E (post-original-issue, 2026-06-28+):** substrate extraction went beyond Phases 1-2 into 16 publishable leaf crates (`katgpt-types`/`katgpt-hla`/`katgpt-transformer`/`katgpt-tokenizer`/`katgpt-speculative`/`katgpt-kv`/`katgpt-dec`/`katgpt-sense`/`katgpt-sleep`/`katgpt-spectral`/`katgpt-micro-belief`/`katgpt-personality`/`katgpt-attn-match`/`katgpt-pruners`/`katgpt-quant`/`katgpt-attn`). `katgpt-core` re-exports the consumed surface (e.g. `pub use katgpt_hla as hla;`). katgpt-core lib: **661/0 green** (re-verified 2026-07-01).
- [x] This issue updated with GOAT/bench evidence at each phase — every step's GOAT gate reported inline in [Plan 008](../.plans/008_katgpt_core_substrate_extraction.md); 2.8 bit-identical verification 2026-07-01.
- [x] **Phase F (composition-layer unblock):** `katgpt-forward` crate created; `ForwardContext` moved there; root `src/transformer.rs` re-exports it + keeps the 33 `forward*` composition functions. **ACCEPTED + FULLY RESOLVED (2026-07-02).** Phase F's structural goal (break the `ForwardContext` DAG cycle so substrate leaves are consumable without root) is **fully achieved** by F.1–F.3 + F.4a/F.4b (4 files migrated, GOAT green). The remaining 22 audited files are pinned by a *deeper* join point (`crate::transformer::forward` the function, not the type) — **resolved all-Option-C in [Issue 033](./033_forward_join_point_architectural_decision.md): each documented root-resident by design.** Option A (trait) rejected after audit (`inference_backend.rs` already has `InferenceBackend` trait; `step.rs` has 6 sibling deps). `cargo check --workspace` + `cargo test -p katgpt-core --lib` + root lib tests green after each step.
  - **F.1+F.2+F.3 DONE (2026-07-02):** `katgpt-forward` crate scaffolded (deps: katgpt-transformer + katgpt-pruners + katgpt-speculative + katgpt-types + katgpt-core; 11 feature gates forwarded). `ForwardContext` struct + impl + `depth_route_with_indices` + `DepthRouteIndicesArgs` moved in (fields `pub(crate)`→`pub`; SIMD paths preserved as `katgpt_core::simd::*`). Root `src/transformer.rs` re-exports via `pub use katgpt_forward::{…}`. **Orphan-rule discovery:** the `DflashCtx<TransformerWeights> for ForwardContext` impl (root `src/speculative/dflash.rs:55`) became illegal once ForwardContext went foreign — moved the impl into `katgpt-forward` (it travels with the type; added katgpt-speculative dep, no cycle). **GOAT gate PASS:** `cargo check --workspace` clean + 0 warnings; `cargo check --workspace --all-features` clean (combo-regression class satisfied — all 11 feature forwards correct, including the `wall_attention`-gated `WallPrefixState` import); `cargo test -p katgpt-core --lib` 666/0 green; root `cargo test --lib` **1681/0 green** (the bit-identical verification — every ForwardContext consumer behaves unchanged).
  - **F.4a–F.4e AUDITED + PARTIALLY EXECUTED (2026-07-02):** Pre-dispatch import audit revealed a **second, deeper join point** the F.4 plan missed: `crate::transformer::forward` (the function, not the type) composes root-only cognitive modules (`cce`, `clr`, `compaction`, `tf_loop`, `pruners::*`) and cannot move to any leaf. This pins every file that calls `forward()`.
    - **F.4a DONE (3 files):** `gdn2/forward.rs` + `dash_attn/forward.rs` moved to `katgpt-attn` (gdn2 + dash_attn modules). katgpt-attn gained optional `katgpt-forward` + `katgpt-transformer` deps activated by `gdn2_attention`/`dash_attn` features. `forward_dash_attn_decode_vortex` STRIPPED (vortex_flow cluster stays root-only — `meta_router` depends on `pruners::bandit` + `speculative::types`). Documented in leaf `dash_attn/forward.rs` module comment. Root re-exports via `pub use katgpt_attn::gdn2::forward::*` / `pub use katgpt_attn::dash_attn::forward::*`.
    - **F.4b DONE (1 file, redirected):** `hla/forward.rs` moved to **`katgpt-forward`** (NOT `katgpt-hla` as planned — `katgpt-core` depends on `katgpt-hla` for substrate re-export, and `katgpt-forward` depends on `katgpt-core`, so `katgpt-hla → katgpt-forward → katgpt-core → katgpt-hla` is a cycle). Placing the HLA forward composition in katgpt-forward (which gains a `katgpt-hla` dep) is acyclic. Module: `katgpt-forward/src/hla_forward.rs`. Root re-exports via `pub use katgpt_forward::{forward_hla, forward_ahla, ...}`.
    - **F.4c BLOCKED (8 files):** `step.rs`, `prefill.rs`, `dflash.rs`, `verifier.rs`, `d2f_verifier.rs`, `drafter_lora.rs`, `flashar_anchor.rs`, `flashar_consensus.rs` — ALL import `crate::transformer::forward` (the root forward function). Root's `forward()` composes root-only cognitive modules and cannot move to a leaf. Also depend on root-only `crate::dllm`, `crate::speculative::{d2f,kurtosis_gate,selectivity_router,...}` siblings. **Needs trait-based indirection** — see §"Revised architecture for blocked files" below.
    - **F.4d WRONG CRATE:** `sleep/consolidation.rs` (Plan 154, GDN2 fast-weight eviction) is unrelated to `crates/katgpt-sleep/` (arXiv:2504.13171, Sleep-Time Query Anticipator). Also depends on root-only `super::{eviction,types}` siblings + `crate::gdn2` (in katgpt-attn). Would need its own crate or a rename.
    - **F.4e BLOCKED (most files):** `inference_backend.rs` (CpuBackend delegates to `forward()`), `benchmark/{hla,simd,speculative}.rs` (all call `forward()`), `inference_router.rs` (depends on `crate::trigger_gate`, `crate::dllm_solver`, `crate::pruners::acceptance_variance`), `fold/` (ThoughtFold composes `ThinkingController` + `ScreeningPruner`), `sp_kv_forward_mod.rs` (depends on `crate::sp_kv::types`). All pinned by the `forward()` join point or root-only sibling deps.
    - **GOAT gate PASS (F.4a+F.4b):** `cargo check --workspace` clean; `cargo check --workspace --all-features` clean; `cargo test -p katgpt-core --lib` 666/0 green; `cargo test -p katgpt-forward --lib` 8/0 green (HLA forward tests); `cargo test -p katgpt-attn --lib --features gdn2_attention,dash_attn` 63/0 green (incl. 6 gdn2 forward + 4 dash_attn forward tests); root `cargo test --lib` 1663/0 green (1681−18=1663 — 18 tests moved with their files into the leaf crates).
  - **Net: 4 files migrated to leaves (F.4a-gdn2, F.4a-dash_attn, F.4b-hla); 22 audited files declared root-resident by design (Option C, Issue 033).** The `forward()` join point is resolved without a redundant trait — `inference_backend.rs` already defines `InferenceBackend` (the forward abstraction); `speculative/step.rs` has 6 root-only sibling deps beyond `forward()`. Phase F's cycle-breaking goal achieved; leaf-extraction 4/26 with remainder documented as intentional engine-tier. See [Issue 033](./033_forward_join_point_architectural_decision.md) for the full audit.

### Revised architecture for blocked files (the `forward()` join point)

The 22 audited blocked files share a common root cause: they call `crate::transformer::forward` (the function), which composes root-only cognitive modules and cannot move to a leaf. Two architectures can unblock them:

**(A) Trait-based forward dispatch (recommended).** Define a `ForwardPass` trait in `katgpt-forward` (or `katgpt-core`):
```rust
pub trait ForwardPass {
    fn forward(&mut self, ctx: &mut ForwardContext, weights: &TransformerWeights,
               cache: &mut MultiLayerKVCache, token: usize, pos: usize,
               config: &Config) -> &mut [f32];
}
```
Root's `forward()` impls this trait. The blocked files (`step.rs`, `inference_backend.rs`, etc.) move to their leaves and take `impl ForwardPass` as a parameter instead of calling `forward()` directly. This is the cleanest fix — it makes the forward function injectable, breaking the root dependency. Cost: threads a trait parameter through ~20 call sites.

**(B) Split `forward()` into generic + root-specific halves.** The generic half (QKV projection, attention, MLP — no cognitive modules) moves to `katgpt-forward`. The root-specific half (cce/clr/compaction composition) stays in root and calls the generic half. Files that only need the generic half can move; files that need the root-specific half stay. Cost: requires auditing which callers need which half — most speculative files call the full `forward()` so this may not unblock much.

**(C) Accept the engine tier stays in root.** Issue 007 §F step 5 already says root keeps "the 33 forward passes + the engine tier." The 22 audited blocked files ARE that engine tier. Option (C) = declare F.4 complete with the 4 migrated files and document the remaining 22 as intentionally-root-resident engine composition.

**Recommendation:** Option (A) for the high-value files (`speculative/step.rs`, `inference_backend.rs` — these are the most reusable). Option (C) for the rest (`fold/`, `inference_router.rs`, `benchmark/*` — these are root-engine-specific by nature). File as a separate issue; do NOT block Phase F acceptance on it.

---

## Open questions (need your call)

1. **Phase 1 scope:** the full 7-step Category A chain above, or a subset (e.g. just the `types`→`transformer`→`hla` core that unblocks the HLA case study first)? Recommend the full chain — anything left behind stays duplicated. *(Resolved: full chain done.)*
2. **`tokenizer`:** move to core (SentencePiece dep risk) or leave in root? Needs the Risk 4 audit. *(Resolved 2026-07-01: extracted as standalone `katgpt-tokenizer` crate; no SentencePiece-sys dep at all — the risk was moot. Crate builds clean.)*
3. **`mcts`/`dd_tree` generic-vs-game split:** how aggressively to parameterize over core `Game`/`Node` traits vs. leave game-coupled copies in riir-engine? *(Resolved per Q3 verdict: generic core trait + concrete game impls in riir-engine — done in Steps 5/6.)*
4. **Go order:** Phase 1+2 first (kills duplication, highest value), defer 3–5? Or push all the way to publish in one go? *(Resolved: Phase 1+2 done; Phase 5 rescinded.)*
5. **Publish policy — ✅ RESOLVED (2026-07-02, policy A):** 16→18 substrate leaf crates were marked `publish = true` (or defaulting to true) in their `Cargo.toml`, while `release-plz.toml` only releases `katgpt-core` and comments everywhere say "only katgpt-core ships to crates.io". This was inconsistent.
   - **(A) Lock down — APPLIED:** all 18 substrate leaf crates now carry `publish = false`. The 11 previously-missing crates (katgpt-dec, katgpt-hla, katgpt-kv, katgpt-micro-belief, katgpt-personality, katgpt-sense, katgpt-sleep, katgpt-spectral, katgpt-speculative, katgpt-transformer, katgpt-types) aligned with the 7 that already had it. This matches the policy already encoded in `release-plz.toml` ("only katgpt-core is released") and root `Cargo.toml` (`publish = false  # ... Only katgpt-core ships to crates.io`). Honest about current intent; no crates.io commitment.
   - **(B) Open up — DEFERRED:** extending `release-plz.toml` to publish the substrate leaves directly fulfills this issue's TL;DR goal ("I want others to use it as easily as possible aka cargo"), but requires a semver / public-API-freeze commitment per crate. This remains a future strategy call — flip individual crates to `publish = true` + add their `[[package]]` block to `release-plz.toml` when ready. katgpt-tokenizer's comment (`# flip to true when ready to ship`) already documents the unlock path.
   - **Build verified:** `cargo check --workspace` clean (isolated `CARGO_TARGET_DIR=/tmp/iss-fix`) after the change — `publish = false` does not affect `cargo build`/`check`.
