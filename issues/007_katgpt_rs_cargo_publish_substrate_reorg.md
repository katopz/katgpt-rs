# Issue 007: Make katgpt-rs Cargo-consumable — Pillar Reorganization + HLA Substrate Extraction

> **Type:** Architecture / reorganization (unblocks cargo publish + kills cross-repo duplication)
> **Status:** Open — proposal, awaiting go/no-go
> **Owner:** develop
> **Created:** 2026-06-27
> **Cross-repo:** katgpt-rs (primary), riir-ai, riir-neuron-db (consumers). riir-train/riir-chain unaffected.
> **Origin:** User directive — "I want others to use it as easily as possible aka cargo" + HLA scattering concern.
> **References:** [Issue 006](./006_x86_64_simd_target_feature.md) (x86_64 gate, now cleared) · `.research/28_Higher_order_Linear_Attention.md`

---

## TL;DR

Two problems, one fix:

1. **`katgpt-rs` (root) isn't cargo-consumable** because its public surface is ~100 flat feature-gated modules with heavy non-optional deps (`plotters`, platform `metal`/`coreml`). Anonymous consumers can't `cargo add` it.
2. **The inference substrate is duplicated across repos.** `hla` (Higher-order Linear Attention) — a pillar — lives in `katgpt-rs/src/hla/`, is **copy-pasted verbatim into `riir-ai/crates/riir-engine/src/hla/`** (same `forward_hla`/`MultiLayerHlaCache` signatures), and is stored as opaque `[f32; 8]` in `riir-neuron-db::NeuronShard`. Same goes for `transformer` and `types` — `riir-engine/src/hla/forward.rs` imports `crate::transformer::{ForwardContext, TransformerWeights}` and `crate::types::{Config}`, proving those are duplicated too.

**Fix:** move the inference substrate (the pillars every repo needs) down into `katgpt-core`, the publishable leaf. Organize the root crate into tiers. Then publish `katgpt-rs` with a small stable default surface + opt-in experimental features.

This is the single change that makes the engine cargo-consumable AND eliminates the cross-repo DRY violation at the substrate layer.

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

Organize the remaining ~100 flat modules into subdirs by role, so the public surface is legible and the stable-vs-experimental split is visible:

```
src/
├── lib.rs
├── primitives/         # GOAT-gated research primitives, each feature-flagged
│   ├── clr/            # (was src/clr/)
│   ├── compaction/     # (was src/compaction/)
│   ├── cgsp.rs         # (was src/cgsp.rs)
│   ├── claim_rubric/   # etc.
│   └── ...
├── inference/          # higher-level inference wiring built ON core
│   ├── attn_match/     # was src/attn_match/
│   ├── speculative/    # was src/speculative/
│   ├── pruners/        # was src/pruners/
│   └── ...
├── games/              # game engines + NPC brains (clearly app-level)
│   ├── percepta/       # was src/percepta/
│   ├── bomber/         # (wherever bomber lives)
│   ├── go/  sudoku/  monopoly/
│   └── npc_brain_router.rs
├── backends/           # platform backends (optional, platform-gated)
│   ├── gpu.rs  ane.rs  inference_router.rs
│   └── ...
└── bench/              # benchmark harnesses (was src/benchmark/)
```

This is a **pure move + `pub use` re-export** refactor — no logic changes. `lib.rs` keeps re-exporting at the top level so existing `use katgpt::clr_vote` call sites don't break. The subdir structure is for human/CI legibility and for scoping the publishable surface.

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

## Cross-repo consumer cleanup (the DRY payoff)

Once the substrate is in `katgpt-core`:

- **riir-ai/riir-engine**: delete `src/hla/`, `src/transformer`, `src/types` duplicates → `use katgpt_core::{hla, transformer, types}`. This is the single biggest DRY win — removes the silent divergence risk.
- **riir-neuron-db**: unchanged structurally (still stores `[f32; 8]`), but can now optionally call `katgpt_core::hla` kernels if it ever needs compute, without pulling the root engine.
- **katgpt-rs root**: `src/hla/`, `src/transformer.rs`, `src/types.rs`, `src/weights.rs` become thin `pub use katgpt_core::{hla, transformer, types, weights};` re-exports (back-compat for existing call sites).

---

## Migration path (incremental, no big-bang)

Each phase is independently shippable and reversible:

- [ ] **Phase 1 — Substrate extraction to core.** Move `types` → core (it's the root of the pillar chain, fewest deps). Then `transformer`/`weights`. Then `hla`. Each move: copy file, update `use` paths in katgpt-rs root (re-export from core), run full test suite. Core version bump: `0.3.0` (new public modules = breaking for core consumers until they `use katgpt_core::hla`).
- [ ] **Phase 2 — Cross-repo dedup.** In riir-ai, replace duplicated `hla`/`transformer`/`types` with `katgpt-core` imports. Delete the copies. Verify `forward_hla` bit-identical on the existing tests in both repos.
- [ ] **Phase 3 — Root crate reorg.** Move root `src/*` modules into `primitives/`/`inference/`/`games/`/`backends/` subdirs. Add top-level `pub use` re-exports so no call site breaks. Pure refactor, no logic.
- [ ] **Phase 4 — Dep audit for publish.** Make `plotters` optional. Verify `cargo check --no-default-features` is clean on the root.
- [ ] **Phase 5 — Publish katgpt-rs.** Add to `release-plz.toml`, first publish `0.1.0`. Document the feature-flag stability tiers in root README.

Phases 1–2 are the high-value, low-risk core (kills the duplication, unblocks clean consumption). Phases 3–5 are the cargo-publish polish. **Phase 1+2 alone deliver most of the value** — any repo can then `cargo add katgpt-core` and get the full inference substrate including HLA.

---

## Risks

1. **Moving `transformer`/`types` to core may surface hidden deps** (e.g., `Config` referencing something in root). **Mitigation:** move `types` first (it's the dependency root), find out, deal with it incrementally. Don't move the whole chain in one commit.
2. **riir-engine's HLA diverged** (`*_role_aware` variants). **Mitigation:** role-aware is likely a superset — port it into core's `hla` behind a feature flag, keep riir-engine's role transport wiring. Phase 2 reconciliation.
3. **Version churn.** Core goes to `0.3.0` (new modules), root starts `0.1.0`. **Mitigation:** both are `0.x`, expected to churn. Document in READMEs.
4. **`tokenizer` may have deps that disqualify it from core.** **Mitigation:** audit before moving; leave in root if it pulls anything heavy.

---

## Acceptance

- [ ] Phase 1: `hla`/`transformer`/`types`/`weights` live in `katgpt-core`, re-exported from root. `cargo test -p katgpt-core --lib` + `cargo test -p katgpt-rs --lib` green.
- [ ] Phase 2: riir-ai has no duplicated `hla`/`transformer`/`types`; consumes them from `katgpt-core`. riir-engine HLA tests bit-identical.
- [ ] Phase 3: root `src/` organized into subdirs; no call-site breakage (all `use katgpt::*` still resolve via re-exports).
- [ ] Phase 4: `cargo check --no-default-features` clean on root; `plotters` optional.
- [ ] Phase 5: `katgpt-rs@0.1.0` live on crates.io; `cargo add katgpt-rs` works.
- [ ] This issue updated with GOAT/bench evidence at each phase (per AGENTS.md "dont defer benchmark task").

---

## Open questions (need your call)

1. **Scope of Phase 1:** just `hla` (the called-out pillar), or the full substrate chain (`types`→`transformer`→`weights`→`hla`)? The chain is the right answer if we want to actually fix the duplication, since HLA depends on the others. Confirm: move the whole chain?
2. **`tokenizer`:** move to core or leave in root? Depends on its deps — needs a 5-min audit.
3. **Go order:** Phase 1+2 first (kills duplication, highest value), defer 3–5? Or push all the way to publish in one push?
