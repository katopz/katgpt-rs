# Issue 033: The `forward()` join point — architectural decision for the 30 root-pinned composition files

> **Type:** Architecture / decision (spin-out from Issue 007 Phase F.4)
> **Status:** **OPEN — decision needed** (Option A vs C, or hybrid). Phase F.4a+F.4b shipped 4 of 34 files (commit `c76722d2`); the remaining 30 are blocked on this decision.
> **Owner:** develop
> **Created:** 2026-07-02
> **Origin:** Issue 007 Phase F.4 pre-dispatch import audit — discovered a second, deeper join point (`crate::transformer::forward` the *function*, not just `ForwardContext` the *type*).
> **References:**
> - [Issue 007](./007_katgpt_rs_cargo_publish_substrate_reorg.md) §"The composition-layer pin" + §"Revised architecture for blocked files" + Acceptance §Phase F
> - Commit `c76722d2` (F.4a+F.4b — the 4 migrated files + GOAT gate)
> - Commit `9a9df4be` (F.1+F.2+F.3 — `katgpt-forward` crate + `ForwardContext` move)

---

## TL;DR

`ForwardContext` (the type) was lifted into `katgpt-forward` in Phase F.1–F.3 — but the **function** `crate::transformer::forward` was not, and it is the deeper binding. Root's `forward()` composes root-only cognitive modules (`cce`, `clr`, `compaction`, `tf_loop`, `pruners::*`), so **any file that calls `forward()` is pinned to root** — a leaf can't depend on root (that's the cycle Phase F was supposed to kill). This pins 30 of the 34 Phase F.4 target files.

**Verdict (proposed, pending confirmation):** hybrid.
- **Option (A) trait-based `ForwardPass` dispatch** for the 2 high-value files (`speculative/step.rs`, `inference_backend.rs`) — these are the genuinely reusable leaves worth the trait-threading cost.
- **Option (C) accept root residency** for the other 28 — they ARE the engine tier that Issue 007 §F step 5 explicitly says stays in root. Declaring them intentionally-root-resident is honest, not a deferral.

Option (B) (split `forward()` into generic + root-specific halves) is **rejected** — most callers need the full forward, so the split unblocks little.

---

## The join point (the diagnosis)

`crate::transformer::forward` is not just a type — it's the **composition function** that wires together every cognitive module per token:

```rust
pub fn forward(ctx: &mut ForwardContext, weights: &TransformerWeights, ...) -> &mut [f32] {
    // ... QKV projection, attention, MLP ...
    // THEN composes root-only cognitive modules:
    cce::modulate(...);          // root-only
    clr::score(...);             // root-only
    compaction::maybe_compact(); // root-only
    tf_loop::step();             // root-only
    pruners::apply(...);         // root-only (bandit, screening, etc.)
}
```

Phase F.1–F.3 moved `ForwardContext` (the *type* that holds the mutable state) into `katgpt-forward`. But `forward()` (the *function* that mutates it) stayed in root because it composes modules that don't exist in any leaf. **Any file that imports `crate::transformer::forward` therefore cannot move to a leaf** — a leaf depending on root is the exact cycle Phase F exists to break.

This is a *deeper* join point than `ForwardContext`. Lifting the type was necessary but not sufficient.

---

## The 30 blocked files (inventory)

| Batch | Files | Count | Blocker |
|---|---|---|---|
| **F.4c** | `speculative/step.rs`, `speculative/prefill.rs`, `speculative/dflash.rs`, `speculative/verifier.rs`, `speculative/d2f_verifier.rs`, `speculative/drafter_lora.rs`, `speculative/flashar_anchor.rs`, `speculative/flashar_consensus.rs` | 8 | All import `crate::transformer::forward`; also depend on root-only `crate::dllm`, `crate::speculative::{d2f,kurtosis_gate,selectivity_router,...}` siblings |
| **F.4d** | `sleep/consolidation.rs` | 1 | **Wrong crate** — `crates/katgpt-sleep/` is the Sleep-Time Query Anticipator (arXiv:2504.13171, Plan 334); `src/sleep/consolidation.rs` is Sleep Consolidation (Plan 154, GDN2 fast-weight eviction). Unrelated features sharing the word "sleep." Also depends on root-only `super::{eviction,types}` + `crate::gdn2` |
| **F.4e** (forward-join) | `inference_backend.rs`, `benchmark/hla.rs`, `benchmark/simd.rs`, `benchmark/speculative.rs` | 4 | All call `forward()` directly |
| **F.4e** (sibling-deps) | `inference_router.rs`, `fold/*`, `sp_kv_forward_mod.rs` | ~17 | Depend on root-only siblings (`crate::trigger_gate`, `crate::dllm_solver`, `crate::pruners::acceptance_variance`, `crate::sp_kv::types`, `ThinkingController`, `ScreeningPruner`) |
| | **Total blocked** | **30** | |

**Migrated (commit `c76722d2`):** 4 files — `gdn2/forward.rs`, `dash_attn/forward.rs` (both → `katgpt-attn`), `hla/forward.rs` (→ `katgpt-forward`, redirected to avoid the `katgpt-core → katgpt-hla → katgpt-forward → katgpt-core` cycle).

---

## The three options

### (A) Trait-based `ForwardPass` dispatch — recommended for high-value files

Define a trait in `katgpt-forward`:

```rust
pub trait ForwardPass {
    fn forward(&mut self, ctx: &mut ForwardContext, weights: &TransformerWeights,
               cache: &mut MultiLayerKVCache, token: usize, pos: usize,
               config: &Config) -> &mut [f32];
}
```

Root's `forward()` impls this trait. Blocked files move to their leaves and take `impl ForwardPass` as a parameter instead of calling `forward()` directly.

- **Pros:** forward becomes injectable; the root dependency is broken cleanly; testable with mock forward; aligns with the existing `DflashCtx` / `SpeculativeGenerator` trait patterns already in the codebase.
- **Cons:** threads a trait parameter through ~20 call sites; signature churn touches every `forward()` caller.
- **Best fit:** `speculative/step.rs` (the core spec-decode step — genuinely reusable) and `inference_backend.rs` (`CpuBackend` delegates to `forward()` — the natural seam for a trait).

### (B) Split `forward()` into generic + root-specific halves — REJECTED

Move the generic half (QKV projection, attention, MLP — no cognitive modules) to `katgpt-forward`; keep the root-specific half (cce/clr/compaction composition) in root, calling the generic half. Files needing only the generic half can move; files needing the root-specific half stay.

- **Why rejected:** most speculative/benchmark callers invoke the *full* `forward()` (they need the cognitive composition to produce realistic logits). Auditing which callers need which half would be high-effort for low unblock yield. The trait approach (A) achieves the same injectability without the audit.

### (C) Accept the engine tier stays in root — recommended for the remaining 28

Issue 007 §F step 5 already says root keeps "the 33 forward passes + the engine tier." The 30 blocked files **are** that engine tier — they are root-engine composition by nature (`fold/`, `inference_router.rs`, `benchmark/*`, `flashar_consensus.rs`, etc.).

- **Pros:** zero churn; honest about the architecture (the engine tier composes cognitive modules that live nowhere else); matches the documented intent.
- **Cons:** leaves 30 files in root `src/`, so the "composition layer fully extracted" goal of Phase F is only partially met. But Phase F's actual goal was killing the `ForwardContext` cycle — that's done.
- **Best fit:** `fold/*`, `inference_router.rs`, `benchmark/*`, `flashar_*`, `drafter_lora.rs`, `sp_kv_forward_mod.rs`, `sleep/consolidation.rs`.

---

## Proposed verdict (hybrid)

- **Option (A)** for `speculative/step.rs` + `inference_backend.rs` (2 files). These are the highest-value reusable leaves; the trait seam is natural and the ~20 call-site churn is justified.
- **Option (C)** for the other 28 files. Document them as intentionally-root-resident engine composition; do not churn them through a trait they don't need.
- **Option (B)** rejected.
- **F.4d** (`sleep/consolidation.rs`): handle separately — it needs its own crate or a rename, NOT a trait. Tracked as a non-blocking follow-up; not part of the A/C decision.

**Net Phase F outcome under this verdict:** 4 (done) + 2 (Option A) = 6 of 34 files in leaves; 28 declared engine-tier-by-design in root. Phase F's cycle-breaking goal is fully achieved; the leaf-extraction goal is 6/34 with the remainder documented as intentional.

---

## Acceptance criteria

- [ ] **Decision recorded** (A vs C vs hybrid) in this issue's status line + Issue 007 §Phase F.
- [ ] **If Option (A) chosen:** `ForwardPass` trait defined in `katgpt-forward`; root `forward()` impls it; `speculative/step.rs` + `inference_backend.rs` migrated to leaves taking `impl ForwardPass`. GOAT gate: `cargo check --workspace --all-features` clean + `cargo test --lib` green (bit-identical — trait dispatch must not change behavior).
- [ ] **If Option (C) chosen for the 28:** each documented as intentionally-root-resident in its module header (one-line comment: `// Engine tier — root-resident by design (Issue 033 §C). Composes cognitive modules that live nowhere else.`).
- [ ] **Issue 007 Phase F checkbox** flipped to `[x]` once this issue is resolved (the 4 migrated files unblock acceptance; the 30 are tracked here, not blocking).
- [ ] **F.4d follow-up** filed separately if `sleep/consolidation.rs` is to be extracted (needs its own crate decision — out of scope for the A/C verdict).

---

## Notes

- **Why this is a separate issue, not a Phase F blocker:** Phase F's structural goal was breaking the `ForwardContext` DAG cycle so the substrate leaves can be consumed without root. That is **done** (F.1–F.3 + F.4a/F.4b, GOAT green). The 30 blocked files are an *additional* extraction goal that turned out to require an architectural choice; gating Phase F acceptance on it would conflate "cycle broken" with "every composition file moved."
- **The katgpt-hla cycle lesson (F.4b):** when threading the `ForwardPass` trait (if Option A), remember that `katgpt-core → katgpt-hla → katgpt-forward → katgpt-core` is a cycle. The trait goes in `katgpt-forward` (or `katgpt-core`), NOT in `katgpt-hla`. The HLA forward composition already lives in `katgpt-forward` for exactly this reason.
- **Vortex decode path:** `forward_dash_attn_decode_vortex` was stripped from the leaf migration (commit `c76722d2`). To re-add, either move the `vortex_flow` cluster into a crate that can depend on `bandit`/`speculative`, or inject the router via a trait. Documented in `katgpt-attn/src/dash_attn/forward.rs` module comment. Non-blocking; not part of this issue.
