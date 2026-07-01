# Proposal 003 — Master `src/` consolidation: domain-stack crates + loser crate

Status: **proposed** (not yet implemented) — supersedes Proposal 001 and 002
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Audit basis: full `src/` classification (3 parallel subagent passes + direct
analysis of attention/quant/mux/sleep/pruners/kv/spectral/speculative)

## TL;DR

The end state: **`src/` contains only `main.rs` + `lib.rs`** (pure re-export
shims + retained transformer-forward glue). Everything else moves into
domain-stack crates following the established pattern (`katgpt-spectral`,
`katgpt-pruners`, `katgpt-kv`, `katgpt-sleep` — one crate per domain that
absorbs the whole domain).

Today `src/` holds ~95 items scattered across ~46 folders and ~49 files.
Attention is the worst case: base primitives in `katgpt-core`, one variant
spun out alone (`katgpt-attn-match`), and the rest (`dash_attn`, `gdn2`,
`hla`, `ega_attn`, `diagonal_gate`) marooned in root. This proposal fixes
the whole tree in one consistent pass.

Losers (dead stubs, superseded primitives, off-topic research toys) get
exiled to **`katgpt-deprecated`** so the winning crates stay clean.

This supersedes Proposals 001 (`katgpt-quant`) and 002 (`katgpt-dash-attn`):
both are absorbed as phases of the master plan with wider scope.

## Guiding principles (non-negotiable)

1. **One domain-stack crate per domain** — every primitive in a domain lives
   in that domain's crate. No scattering. `katgpt-spectral` is the template.
2. **`src/` ends at `main.rs` + `lib.rs`** — `lib.rs` is re-export shims +
   the thin transformer-forward glue that can't move (it needs `ForwardContext`).
3. **Forward-vs-primitive seam** — clean primitives promote to crates;
   transformer-bound forward integration (`*::forward.rs` needing
   `crate::transformer::ForwardContext`) stays in root until/unless
   `katgpt-transformer` grows a forward module. This is the `katgpt-attn-match`
   / Issue 359 precedent.
4. **Losers get exiled, not deleted** — `katgpt-deprecated` holds demoted
   primitives behind opt-in features, never default. Keeps regression
   comparability without cluttering winners.
5. **Back-compat re-export** — every move keeps `pub use katgpt_X as Y` in
   `lib.rs` so existing `katgpt_rs::*` paths resolve. Mirrors Issue 015 Phase 5.

## The crate taxonomy (target state)

### Existing domain crates (absorb into these)

| Crate | Absorbs from `src/` |
|---|---|
| `katgpt-core` | base math/SIMD + `alloc.rs`, `cce/`, `cumprodsum.rs`, `llmexec_guard.rs`, `memory_soup_lora.rs`, `mux_demux.rs`, `salience/`, `trigger_gate.rs`, `skill_opt/`, `ssd_block.rs`, hoist `sigmoid` out of `band_conditioner` |
| `katgpt-types` | (shared leaf — no `src/` absorptions, already complete) |
| `katgpt-transformer` | `transformer.rs` (stays root — owns `ForwardContext`), absorbs `mbu.rs`, `dense_mesh/`, `swir/`, `tf_loop.rs` |
| `katgpt-kv` | `cache_prune/`, `segment_checkpoint/`, `async_qdq.rs` (already has `sp_kv`) |
| `katgpt-pruners` | `closure_wire.rs`, `screening/` (already complete otherwise) |
| `katgpt-spectral` | `spectral_budget.rs`, `spectral_concentration.rs`, `spectral_retract.rs`, `stiff_anomaly/`, `gauge_invariant.rs`, `manifold_power_iter_router.rs`, `off_principal.rs`, `procrustes.rs`, `river_valley.rs`, `distill/peira` (split) |
| `katgpt-speculative` | `distill/{ilc,trd}` (split), `spechop/`, `rt_turbo/`, `precision_aware_draft.rs`, `sparse_compose.rs`, `spec_reconciliation/` |
| `katgpt-sleep` | `sleep/` (Plan 154 GDN2 consolidation), `closure_mining.rs` |

### Existing domain crates (leave as siblings)

| Crate | Note |
|---|---|
| `katgpt-attn-match` | already promoted (Plan 271/Issue 359). **Fold `rerank.rs` in** (MaxSim IS late-interaction attention). Otherwise unchanged. |

### New domain crates (create these)

| Crate | Absorbs | Why a new crate |
|---|---|---|
| **`katgpt-attn`** | FROM `katgpt-core`: `attention`, `parallax_attn`, `set_attention`, `funcattn`. FROM root: `ega_attn.rs`, `diagonal_gate.rs`, `gdn2/` (kernel+types), `hla/` (kernel+types), `dash_attn/` (13 primitive files), `rat_bridge/`, `static_cal.rs`, `funcattn_compose/`, `chiaroscuro/` | The attention stack. Verified nothing in `katgpt-core` consumes its own attention modules internally (zero `use crate::attention` hits) — they can move out. `chiaroscuro` is the entropy-routing layer `funcattn_compose` depends on; they move together. |
| **`katgpt-quant`** | `turboquant/`, `planar_quant/`, `iso_quant/`, `hybrid_oct_pq/`, `octopus/` | Proposal 001, unchanged. Rotation/codebook KV-compression family. |
| **`katgpt-band`** | `band_conditioner.rs`, `bckvss.rs`, `collider_pruner.rs`, `adaptive_cot_stopper.rs` | Plan 265 cluster (arXiv:2605.12733). Tightly inter-coupled; splitting would break internal cohesion. `sigmoid` hoists to core first. |
| **`katgpt-claim`** | `claim_rubric/`, `clr/` | Claim-Level Reliability pair (arXiv:2606.07612). Dev/CI tooling. |
| **`katgpt-sparse`** | `sparse_task_vector.rs`, `specialist_projection.rs` | SOPTV/SPLAT task-vector family (Plan 264/265). Distinct from spectral (different paper lineage). `sparse_compose.rs` → `katgpt-speculative` (it's draft composition). |
| **`katgpt-ruliology`** | `ruliology/` | Wolfram ruliology (Plan 188) — distinct domain. One cross-couple to `pruners::g_zero::delta_absorb` needs interface extraction. |
| **`katgpt-validator`** | `validator/` | Partial parser + syntax pruner. Clean leaf. |
| **`katgpt-bench`** | `benchmark/`, `plot.rs` | Tooling — depends on everything, must be top-level. |
| **`katgpt-deprecated`** | `feedback.rs` (dead stub), `unit_distance/` (number-theory toy) | The **loser crate**. See below. |

### Stays in `src/` (transformer-forward glue — can't move yet)

These need `crate::transformer::{ForwardContext, TransformerWeights}` which
lives in root (`katgpt-transformer` is types-only per its Cargo.toml). They
collapse into `src/lib.rs` as private `mod` declarations, not folders:

- `transformer.rs` — owns `ForwardContext`, the linchpin. Stays.
- `gdn2/forward.rs`, `hla/forward.rs`, `dash_attn/forward.rs` +
  `dash_attn/tests.rs` — forward integration. Stays as `mod dash_attn_forward`.
- `sp_kv_forward_mod.rs`, `attn_match_adaptive_cot.rs` — already root glue.
- `types.rs` — already a re-export shim (`katgpt_core::types::*`).
- `inference_backend.rs`, `inference_router.rs`, `gpu_backend.rs`,
  `ane_backend.rs` — backend dispatch, transformer-bound.

## The `katgpt-deprecated` loser crate

**Purpose:** exile demoted/dead/off-topic primitives so winning crates stay
clean. Never default-on. Kept (not deleted) for regression comparison and
to preserve the GOAT-gate audit trail.

**Membership criteria — 3 categories of opt-in (NOT a blanket sweep):**

Opt-in features are NOT automatically losers. The sweep classifies each into:

1. **Pending** — opt-in because the GOAT gate hasn't run yet (e.g. "Opt-in
   until G1–G4 pass"). → **stays in domain crate.** Exiling these punishes WIP.
2. **Benchmark-loser** — lost a head-to-head, kept so the winner-vs-loser A/B
   regression bench still works (the bench needs both in scope). → **stays in
   domain crate** behind its feature flag.
3. **Dead/failed** — gate ran and FAILED, OR explicitly demoted (e.g.
   `alien_sampler`: 1/4 PASS), OR dead stub (`feedback.rs`), OR off-topic
   (`unit_distance/`). → **exile to `katgpt-deprecated`.**

Only category 3 moves to `katgpt-deprecated`. The distinction matters: most
opt-in features in the Cargo.toml are category 1 (pending), not category 3.

**Cargo.toml shape:**
```toml
[package]
name = "katgpt-deprecated"
version = "0.1.0"
edition = "2024"
license = "MIT"
description = "Exiled primitives — dead stubs, off-topic research toys, and superseded mechanisms awaiting deletion. Never default-on. Kept for regression comparison and GOAT-gate audit trail."
publish = false

[features]
default = []   # ALWAYS empty — nothing here is ever default.
feedback = []        # dead TTT stub
unit_distance = []   # number-theory research toy
```

**Lifecycle:** items here are deletion candidates. Each carries a
`# TODO(deprecated): delete after <plan/issue> confirms no regression`
comment. The crate exists to make deletion safe and auditable, not to live
forever.

## Full destination map (every `src/` item)

> Legend — verdict: **W**=winner (default-worthy), **O**=opt-in (gated feature),
> **L**=loser (exile), **G**=glue (stays root, transformer-bound), **R**=re-export shim.

### → `katgpt-core`
| Item | Verdict |
|---|---|
| `alloc.rs` | W |
| `cce/` | W |
| `cumprodsum.rs` | W |
| `llmexec_guard.rs` | O |
| `memory_soup_lora.rs` | O |
| `mux_demux.rs` | O |
| `salience/` | O |
| `trigger_gate.rs` | W |
| `skill_opt/` | O |
| `ssd_block.rs` | O |
| `channel_simd.rs` | O |
| `sigmoid` (hoist from `band_conditioner`) | W |

> **Removed from this list (misclassification fix):** `alien_sampler/` was
> originally listed here as a winner based on its doc comment. The actual GOAT
> history (`issues/010` T6) records **"1/4 PASS — demoted to opt-in."** It is a
> demoted loser → exiled to `katgpt-deprecated`. This is the exact failure mode
> the Phase 0.5 loser-sweep exists to catch.

### → `katgpt-attn` (NEW — the attention stack)
| Item | Verdict |
|---|---|
| `attention`, `parallax_attn`, `set_attention`, `funcattn` (from core) | W/O |
| `ega_attn.rs` | O |
| `diagonal_gate.rs` | W |
| `gdn2/{kernel,types}.rs` | O |
| `hla/{kernel,types}` | O |
| `dash_attn/` (13 primitive files) | O |
| `rat_bridge/` | O |
| `static_cal.rs` | O |
| `chiaroscuro/` (entropy routing) | O |
| `funcattn_compose/` | O |

### → `katgpt-quant` (NEW — Proposal 001)
| Item | Verdict |
|---|---|
| `turboquant/`, `planar_quant/`, `iso_quant/`, `hybrid_oct_pq/`, `octopus/` | O (promote GOAT winner to W) |

### → `katgpt-spectral`
| Item | Verdict |
|---|---|
| `spectral_budget.rs`, `spectral_concentration.rs`, `spectral_retract.rs` | O/W |
| `stiff_anomaly/` | O |
| `gauge_invariant.rs`, `manifold_power_iter_router.rs` | O |
| `off_principal.rs`, `procrustes.rs` | O |
| `river_valley.rs` | O |
| `distill/peira` (split from `distill/`) | O |

### → `katgpt-kv`
| Item | Verdict |
|---|---|
| `cache_prune/`, `segment_checkpoint/`, `async_qdq.rs` | O |

### → `katgpt-pruners`
| Item | Verdict |
|---|---|
| `closure_wire.rs`, `screening/` | O |

### → `katgpt-speculative`
| Item | Verdict |
|---|---|
| `distill/{ilc,trd}` (split), `spechop/`, `rt_turbo/` | O |
| `precision_aware_draft.rs`, `sparse_compose.rs`, `spec_reconciliation/` | O |

### → `katgpt-sleep`
| Item | Verdict |
|---|---|
| `sleep/` (Plan 154 consolidation), `closure_mining.rs` | O/W |

### → `katgpt-transformer`
| Item | Verdict |
|---|---|
| `mbu.rs`, `dense_mesh/`, `swir/`, `tf_loop.rs` | O |

### → `katgpt-band` (NEW)
| Item | Verdict |
|---|---|
| `band_conditioner.rs`, `bckvss.rs`, `collider_pruner.rs`, `adaptive_cot_stopper.rs` | W/O |

### → `katgpt-claim` (NEW)
| Item | Verdict |
|---|---|
| `claim_rubric/`, `clr/` | W |

### → `katgpt-sparse` (NEW)
| Item | Verdict |
|---|---|
| `sparse_task_vector.rs`, `specialist_projection.rs` | O |

### → `katgpt-ruliology` (NEW)
| Item | Verdict |
|---|---|
| `ruliology/` | O |

### → `katgpt-validator` (NEW)
| Item | Verdict |
|---|---|
| `validator/` | O |

### → `katgpt-bench` (NEW — tooling)
| Item | Verdict |
|---|---|
| `benchmark/`, `plot.rs` | W (tooling) |

### → `katgpt-attn-match` (existing, absorb `rerank`)
| Item | Verdict |
|---|---|
| `rerank.rs` | O |

### → `katgpt-deprecated` (NEW — the loser crate)
| Item | Verdict | Reason |
|---|---|---|
| `feedback.rs` | L | dead stub — `log::debug!` only, no HTTP POST |
| `unit_distance/` | L | number-theory toy, no inference role |
| `alien_sampler/` | L | GOAT 1/4 PASS — demoted to opt-in (`issues/010` T6) |

> **This list is INCOMPLETE by design.** The full loser set is populated by the
> Phase 0.5 loser-sweep audit — every opt-in feature checked against its GOAT
> history in `.benchmarks/` / `.plans/` / `.issues/`. Known failures so far
> (e.g. SDPG bomber arena, `.benchmarks/011_sdpg_bandit_arena.md` GOAT ❌ FAIL)
> are tracked there; the sweep finds the rest before any absorption phase runs,
> so losers don't get absorbed into winning crates.

### Stays in `src/` (root glue — collapses into `lib.rs`)
| Item | Why |
|---|---|
| `main.rs` | binary entry |
| `lib.rs` | re-export shims + retained modules |
| `transformer.rs` | owns `ForwardContext` (linchpin) |
| `gdn2/forward.rs`, `hla/forward.rs`, `dash_attn/forward.rs` + `tests.rs` | transformer-bound forward |
| `sp_kv_forward_mod.rs`, `attn_match_adaptive_cot.rs` | already root glue |
| `inference_backend.rs`, `inference_router.rs`, `gpu_backend.rs`, `ane_backend.rs` | backend dispatch |
| `types.rs` | already a re-export shim |

### Inline-and-delete (collapse into `lib.rs`)
| Item | Action |
|---|---|
| `cgsp.rs` | delete file, `pub use katgpt_core::cgsp::*;` in `lib.rs` (already a 37-line shim) |

## Required splits (not moves)

1. **`distill/` splits two ways.** PEIRA (spectral alignment metric) →
   `katgpt-spectral`. ILC + TRD (speculative draft screening) →
   `katgpt-speculative`. The `distill/` umbrella conflates two paper
   lineages; it does not survive as a unit.
2. **`band_conditioner::sigmoid` hoists to `katgpt-core`** before the
   `katgpt-band` extraction. Three siblings (`adaptive_cot_stopper`,
   `bckvss`, `collider_pruner`) import it; it's a math utility, not band-domain.

## GOAT gate (per promotion phase)

Each new crate / each absorption passes the no-regression GOAT gate before
its phase commits:

- [ ] **G1 correctness** — existing tests pass unchanged via re-exports.
- [ ] **G2 perf** — hot-path latency unchanged (±2%); no indirection cost.
- [ ] **G3 no-regression** — `cargo check --workspace --all-features` clean.
- [ ] **G4 alloc-free hot path** — scratch buffers stay `&mut [T]`, no new heap.
- [ ] **G5 feature-matrix CI** — new crate added to `ci_feature_guard.sh`
      with `--all-features` (the `merkle_root`/`can_freeze` combo-regression guard).

Winners promoted to default-on additionally require a modelless gain
benchmark. Losers demoted to `katgpt-deprecated` are *already* losers by
definition — no gate, just exile.

## Phased rollout

Ordering: foundation first (hoists, splits), then domain crates biggest-first.

- [ ] **Phase 0 — foundation moves (no new crates).**
  - Hoist `sigmoid` from `band_conditioner.rs` → `katgpt-core`.
  - Split `distill/` → peira (stays, tagged for spectral) + ilc/trd (stays,
    tagged for speculative). Wire the split in-tree first.
  - Inline-and-delete `cgsp.rs` into `lib.rs`.
  - GOAT gate G3 on each.
- [x] **Phase 0.5 — loser-sweep audit (gates every absorption).** For EACH
  opt-in feature in root + every crate: grep its name against `.benchmarks/`,
  `.plans/`, `.issues/` for the GOAT verdict. Classify into pending /
  benchmark-loser / dead-failed per the 3-category rule. Exile every category-3
  to `katgpt-deprecated` with a one-line citation. **Must complete before
  Phases 4–11** so losers aren't absorbed into winning crates. Known starting
  set: `feedback.rs`, `unit_distance/`, `alien_sampler/` (demoted, `issues/010`
  T6), SDPG bomber arena (`benchmarks/011` FAIL). Output: a `katgpt-deprecated`
  membership table with a citation per row.
  **DONE 2026-07-01.** Audit: `.docs/001_loser_sweep_audit.md`. 17 exile
  candidates found (13 with code + 4 dead Cargo.toml entries). 3 src/ items
  exiled in Phase 3a; `dense_mesh` deferred (transformer-bound glue);
  cross-crate losers deferred to Phases 8/10.
- [ ] **Phase 1 — `katgpt-quant` crate** (Proposal 001). 5 modules / 25 files.
  Cleanest lift (leaf over core+types). Establishes the move pattern.
- [ ] **Phase 2 — `katgpt-attn` crate.** The attention stack. Move base
  primitives out of `katgpt-core` + absorb `ega_attn`/`diagonal_gate`/`gdn2`/
  `hla`/`dash_attn`/`rat_bridge`/`static_cal`/`chiaroscuro`/`funcattn_compose`.
  Forward glue stays root. Biggest payoff, biggest lift.
- [x] **Phase 3 — `katgpt-deprecated` crate.** Exile `feedback.rs` +
  `unit_distance` + `alien_sampler` (Phase 3a; `dense_mesh` deferred —
  transformer-bound glue). `katgpt-deprecated` crate created with 3 features
  (`feedback`, `unit_distance`, `alien_sampler`), all opt-in, `default = []`.
  Root re-exports preserved for back-compat. Cross-crate losers (dflare_*,
  sdpg_bandit, delta_mem, rmsd_distill, manifold_pruner, compression_drafter,
  stepcode) deferred to Phases 8/10 absorption. See `.docs/001_loser_sweep_audit.md`.
  **DONE 2026-07-01.**
- [ ] **Phase 4 — `katgpt-spectral` absorption.** Add `spectral_*`,
  `stiff_anomaly`, `gauge_invariant`, `manifold_power_iter_router`,
  `off_principal`, `procrustes`, `river_valley`, `distill/peira`.
- [ ] **Phase 5 — `katgpt-kv` absorption.** `cache_prune`,
  `segment_checkpoint`, `async_qdq`.
- [ ] **Phase 6 — `katgpt-speculative` absorption.** `distill/{ilc,trd}`,
  `spechop`, `rt_turbo`, `precision_aware_draft`, `sparse_compose`,
  `spec_reconciliation`.
- [ ] **Phase 7 — `katgpt-sleep` absorption.** `sleep/`, `closure_mining`.
- [ ] **Phase 8 — `katgpt-pruners` absorption.** `closure_wire`, `screening`.
  Fold `rerank.rs` → `katgpt-attn-match`.
- [ ] **Phase 9 — `katgpt-transformer` absorption.** `mbu`, `dense_mesh`,
  `swir`, `tf_loop`.
- [ ] **Phase 10 — `katgpt-core` absorption.** `alloc`, `cce`, `cumprodsum`,
  `llmexec_guard`, `memory_soup_lora`, `mux_demux`, `salience`, `trigger_gate`,
  `skill_opt`, `ssd_block`, `channel_simd`, `alien_sampler`.
- [ ] **Phase 11 — new domain crates.** `katgpt-band`, `katgpt-claim`,
  `katgpt-sparse`, `katgpt-ruliology`, `katgpt-validator`, `katgpt-bench`.
- [ ] **Phase 12 — final sweep.** `src/` should contain only:
    - **`lib.rs`** — minimal: `pub mod transformer` + retained forward-glue `mod`s + back-compat `pub use katgpt_*` re-exports. **No domain logic.** It is the feature-aggregation surface (cross-crate feature combos in `Cargo.toml` like `cgsp`/`sr2am_configurator` that forward to multiple sibling crates) + the transformer-runtime home (`ForwardContext`). Stays `publish = false` per repo policy — only `katgpt-core` ships to crates.io.
    - **`transformer.rs`** — owns `ForwardContext` (linchpin).
    - retained forward-glue: `gdn2/forward.rs`, `hla/forward.rs`, `dash_attn/forward.rs` + `tests.rs`, `sp_kv_forward_mod.rs`, `attn_match_adaptive_cot.rs`, backend dispatch (`inference_backend.rs`, `inference_router.rs`, `gpu_backend.rs`, `ane_backend.rs`), `types.rs` (re-export shim).
    - **DELETE `main.rs`** — it's a redundant binary bench runner; `examples/` (200+ entries) already covers every bench/demo need via `[[example]]`. The implicit `[[bin]]` forces the root crate to ship a binary it doesn't need. `rm src/main.rs`.
  Audit with `find src -type f`. Anything beyond the list above is a missed move — log + fix.
- [ ] **Phase 13 — commit + record.** Commit on `develop` with `refactor:`
  prefix per phase. Update this proposal status to **done** at Phase 12.

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| `katgpt-core` → `katgpt-attn` move breaks consumers expecting `katgpt_core::attention` | high | Re-export shim in `katgpt-core`: `pub use katgpt_attn::attention;` for one release cycle. |
| `benchmark/` depends on 6 root modules → `katgpt-bench` has a huge dep surface | medium | `katgpt-bench` is tooling, not a library — acceptable. Or keep as a binary in `examples/` instead of a crate. |
| `ruliology/mutation.rs` cross-couple to `pruners::g_zero::delta_absorb` | medium | Extract a trait boundary in Phase 11; ruliology depends on pruners (clean direction). |
| Forward-glue retention makes the seam look unfinished | low | Document it as intentional (Issue 359 precedent). The seam is honest: forward is transformer-bound, primitives aren't. |
| Phase 2 (katgpt-attn) is the biggest lift and highest blast radius | high | Do it immediately after Phase 1 (quant) which establishes the move pattern. Run G1–G5 per sub-module. |
| Hidden consumer of a moved module | medium | Each phase greps the full tree for `crate::<module>` before deletion; G3 `--all-features` catches stragglers. |
| `katgpt-deprecated` accumulates forever | low | Each item carries a deletion-TODO with a plan/issue reference. Quarterly prune pass. |

## Out of scope

- Growing `katgpt-transformer` to own forward-pass logic (so `ForwardContext`
  + forward glue can finally leave root). That's a separate architectural
  decision — it would unblock the last retained glue AND enable fully
  deleting `lib.rs` (turning the repo into a pure `[workspace]` with no root
  package). Deferred — the feature-aggregation role still needs a root package
  unless every consumer (riir-ai, examples, tests) takes over cross-crate
  feature orchestration itself.
- Publishing `katgpt-rs` to crates.io. Policy is explicit and unchanged:
  `publish = false` (Cargo.toml L9), `release = false` (release-plz.toml L11),
  "only katgpt-core ships to crates.io." `lib.rs` existing does NOT make it
  publishable — it's a local aggregator, not a shippable artifact.
- Cross-repo moves (anything into riir-ai / riir-chain / riir-neuron-db /
  riir-train). This proposal is katgpt-rs-internal only.
- Deleting `katgpt-deprecated` contents — exile only; deletion is a follow-up.

## References

- `katgpt-spectral` precedent: `crates/katgpt-spectral/Cargo.toml` (Issue 015).
- `katgpt-attn-match` precedent: Plan 271 / Issue 359 (primitive-vs-forward split).
- `merkle_root` / `can_freeve` combo-regression lessons: `riir-chain/AGENTS.md`,
  `riir-neuron-db/AGENTS.md`.
- Modelless mandate + GOAT gate: `katgpt-rs/AGENTS.md` §§ "Modelless-first
  mandate", "Feature Flag Discipline".
- Supersedes: `proposals/001_quant_crate_promotion.md`,
  `proposals/002_dash_attn_crate_promotion.md` (absorbed as Phases 1–2).

## TL;DR

`src/` → only `main.rs` + `lib.rs` (+ retained transformer-forward glue that
can't move yet). Every domain gets one stack crate. Three new domain stacks
beyond 001/002: `katgpt-band` (Plan 265 cluster), `katgpt-claim`,
`katgpt-sparse`. Losers exile to `katgpt-deprecated`. 13 phases, foundation
moves first, biggest-payoff attention stack second. Supersedes 001 and 002.
