# Issue 033: The `forward()` join point — architectural decision for the root-pinned composition files

> **Type:** Architecture / decision (spin-out from Issue 007 Phase F.4)
> **Status:** **RESOLVED (2026-07-02) — all Option C.** Empirical audit found the proposed hybrid infeasible: both Option-A candidates turned out to be engine-tier. `inference_backend.rs` already defines the `InferenceBackend` trait (same signature as a proposed `ForwardPass`); `speculative/step.rs` has 6 root-only sibling deps beyond `forward()`. All 22 audited blocked files are documented root-resident by design. Option A rejected as redundant/insufficient; Option B rejected (low yield); Option C adopted for ALL.
> **Owner:** develop
> **Created:** 2026-07-02
> **Origin:** Issue 007 Phase F.4 pre-dispatch import audit — discovered a second, deeper join point (`crate::transformer::forward` the *function*, not just `ForwardContext` the *type*).
> **References:**
> - [Issue 007](./007_katgpt_rs_cargo_publish_substrate_reorg.md) §"The composition-layer pin" + §"Revised architecture for blocked files" + Acceptance §Phase F
> - Commit `c76722d2` (F.4a+F.4b — the 4 migrated files + GOAT gate)
> - Commit `9a9df4be` (F.1+F.2+F.3 — `katgpt-forward` crate + `ForwardContext` move)

---

## TL;DR

`ForwardContext` (the type) was lifted into `katgpt-forward` in Phase F.1–F.3 — but the **function** `crate::transformer::forward` was not, and it is the deeper binding. Root's `forward()` composes root-only cognitive modules (`cce`, `clr`, `compaction`, `tf_loop`, `pruners::*`), so **any file that calls `forward()` is pinned to root** — a leaf can't depend on root (that's the cycle Phase F was supposed to kill). Original filing estimated ~30 of ~34; empirical audit (§inventory below) corrected this to **22 of 26 audited** files.

**Verdict (adopted 2026-07-02): all Option C.** The proposed hybrid was abandoned after empirical audit of the two Option-A candidates:
- **`inference_backend.rs`** already defines `InferenceBackend` — a trait with the *exact same signature* as a proposed `ForwardPass`. Creating a second trait would be redundant. Its providers (`CpuBackend`/`AneBackend`/`GpuBackend`) are all 1-line delegations to root's `forward()` and must stay in root.
- **`speculative/step.rs`** has 6 root-only sibling deps beyond `forward()` (`crate::speculative::{verifier, dd_tree, dflash, types, kurtosis_gate, selectivity_router}` — all root modules, not `katgpt-speculative` leaf). A `ForwardPass` trait solves 1 of 7 deps; it cannot unblock the file.

**All 22 blocked files are Option C (root-resident by design).** Option A rejected as redundant/insufficient; Option B rejected (low yield).

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

## The blocked files (inventory — 22 audited)

> **Count correction:** the original filing estimated "~30 blocked" with F.4e sibling-deps at "~17". The empirical audit (2026-07-02) found the actual count is **22**: F.4e sibling-deps is 9 (not ~17). All 22 are now documented root-resident by design.

| Batch | Files | Count | Blocker |
|---|---|---|---|
| **F.4c** | `speculative/step.rs`, `speculative/prefill.rs`, `speculative/dflash.rs`, `speculative/verifier.rs`, `speculative/d2f_verifier.rs`, `speculative/drafter_lora.rs`, `speculative/flashar_anchor.rs`, `speculative/flashar_consensus.rs` | 8 | All import `crate::transformer::forward`; also depend on root-only `crate::dllm`, `crate::speculative::{d2f,kurtosis_gate,selectivity_router,...}` siblings |
| **F.4d** | `sleep/consolidation.rs` | 1 | **Wrong crate** — `crates/katgpt-sleep/` is the Sleep-Time Query Anticipator (arXiv:2504.13171, Plan 334); `src/sleep/consolidation.rs` is Sleep Consolidation (Plan 154, GDN2 fast-weight eviction). Unrelated features sharing the word "sleep." Also depends on root-only `super::{eviction,types}` + `crate::gdn2` |
| **F.4e** (forward-join) | `inference_backend.rs`, `benchmark/hla.rs`, `benchmark/simd.rs`, `benchmark/speculative.rs` | 4 | All call `forward()` directly |
| **F.4e** (sibling-deps) | `inference_router.rs`, `fold/{attention_importance,chain_folder,fold_bandit,fold_cache,step_boundary,thinking_ext,types}.rs`, `sp_kv_forward_mod.rs` | 9 | Depend on root-only siblings (`crate::trigger_gate`, `crate::dllm_solver`, `crate::pruners::acceptance_variance`, `crate::sp_kv::types`, `ThinkingController`, `ScreeningPruner`) |
| | **Total** | **22** | |

**Migrated (commit `c76722d2`):** 4 files — `gdn2/forward.rs`, `dash_attn/forward.rs` (both → `katgpt-attn`), `hla/forward.rs` (→ `katgpt-forward`, redirected to avoid the `katgpt-core → katgpt-hla → katgpt-forward → katgpt-core` cycle).

---

## The three options

### (A) Trait-based `ForwardPass` dispatch — REJECTED after empirical audit

Define a trait in `katgpt-forward`:

```rust
pub trait ForwardPass {
    fn forward(&mut self, ctx: &mut ForwardContext, weights: &TransformerWeights,
               cache: &mut MultiLayerKVCache, token: usize, pos: usize,
               config: &Config) -> &mut [f32];
}
```

Root's `forward()` impls this trait. Blocked files move to their leaves and take `impl ForwardPass` as a parameter instead of calling `forward()` directly.

- **Pros:** forward becomes injectable; the root dependency is broken cleanly; testable with mock forward.
- **Cons:** threads a trait parameter through ~20 call sites; signature churn touches every `forward()` caller.
- **REJECTED because:**
  1. **`inference_backend.rs` already defines `InferenceBackend`** — a trait with the *exact same signature* (line 50-58: `fn forward<'a>(&'a mut self, ctx: &'a mut ForwardContext, ...) -> &'a mut [f32]`). Creating a second `ForwardPass` trait would duplicate it. The `CpuBackend`/`AneBackend`/`GpuBackend` providers are 1-line delegations to root's `forward()`; they must stay in root regardless.
  2. **`speculative/step.rs` has 6 root-only sibling deps** beyond `forward()` (`crate::speculative::{verifier, dd_tree, dflash, types, kurtosis_gate, selectivity_router}` — all root modules, the `katgpt-speculative` leaf only has `dd_tree` + `dflash`). A `ForwardPass` trait solves 1 of 7 dependencies. Moving the file requires either abstracting all 7 behind traits (massive churn) or cascading all 7 into the leaf. The trait alone is insufficient.
  3. Creating a redundant trait that doesn't enable migration would violate "production grade only."

### (B) Split `forward()` into generic + root-specific halves — REJECTED

Move the generic half (QKV projection, attention, MLP — no cognitive modules) to `katgpt-forward`; keep the root-specific half (cce/clr/compaction composition) in root, calling the generic half. Files needing only the generic half can move; files needing the root-specific half stay.

- **Why rejected:** most speculative/benchmark callers invoke the *full* `forward()` (they need the cognitive composition to produce realistic logits). Auditing which callers need which half would be high-effort for low unblock yield. The trait approach (A) achieves the same injectability without the audit.

### (C) Accept the engine tier stays in root — ADOPTED FOR ALL 22 FILES

Issue 007 §F step 5 already says root keeps "the 33 forward passes + the engine tier." The 22 blocked files **are** that engine tier — they are root-engine composition by nature (`fold/`, `inference_router.rs`, `benchmark/*`, `flashar_consensus.rs`, etc.).

- **Pros:** zero churn; honest about the architecture (the engine tier composes cognitive modules that live nowhere else); matches the documented intent.
- **Cons:** leaves 22 files in root `src/`, so the "composition layer fully extracted" goal of Phase F is only partially met. But Phase F's actual goal was killing the `ForwardContext` cycle — that's done.
- **ADOPTED:** each of the 22 files carries a `//! _Root-resident by design (Issue 033 §C, Option C)._` doc comment listing its specific root-only deps.

---

## Adopted verdict (all Option C)

- **Option (C) for ALL 22 files.** Each documented as root-resident by design with a `//!` module doc comment listing its specific root-only deps.
- **Option (A) rejected** — `inference_backend.rs` already has the equivalent `InferenceBackend` trait; `speculative/step.rs` is pinned by 6 additional siblings. See §(A) above for the full audit.
- **Option (B) rejected** — low yield (most callers need the full forward).
- **F.4d** (`sleep/consolidation.rs`): classified Option C (composes root-only GDN2 eviction). The "wrong crate" note (cannot go to `katgpt-sleep` — unrelated feature) remains as a non-blocking follow-up if extraction is ever desired.

**Net Phase F outcome:** 4 (done, F.4a/F.4b) of 26 audited composition files in leaves; 22 declared engine-tier-by-design in root. Phase F's cycle-breaking goal is fully achieved; the leaf-extraction goal is 4/26 with the remainder documented as intentional.

---

## Acceptance criteria

- [x] **Decision recorded** — all Option C, documented in this issue's status line + Issue 007 §Phase F.
- [x] **Option (A) rejected after audit** — `inference_backend.rs` already defines `InferenceBackend` (same signature as proposed `ForwardPass`); `speculative/step.rs` has 6 root-only sibling deps. No redundant trait created (production-grade constraint).
- [x] **All 22 blocked files documented** — each carries a `//! _Root-resident by design (Issue 033 §C, Option C)._` doc comment with file-specific root-only deps. Verified: F.4c (8) + F.4d (1) + F.4e forward-join (4) + F.4e sibling-deps (9) = 22.
- [x] **Issue 007 Phase F checkbox** is `[x]` (flipped in commit `4666722f`).
- [x] **GOAT gate:** `cargo check --workspace` clean (doc-only changes — verified by 3 parallel subagent batches with isolated `CARGO_TARGET_DIR`). No code changed.
- [-] **F.4d follow-up** — `sleep/consolidation.rs` classified Option C (root-resident). The "needs own crate if extracted" note is deferred; not blocking.

---

## Notes

- **Why this is a separate issue, not a Phase F blocker:** Phase F's structural goal was breaking the `ForwardContext` DAG cycle so the substrate leaves can be consumed without root. That is **done** (F.1–F.3 + F.4a/F.4b, GOAT green). The 22 audited blocked files are an *additional* extraction goal that turned out to require an architectural choice; gating Phase F acceptance on it would conflate "cycle broken" with "every composition file moved."
- **The katgpt-hla cycle lesson (F.4b):** when threading the `ForwardPass` trait (if Option A), remember that `katgpt-core → katgpt-hla → katgpt-forward → katgpt-core` is a cycle. The trait goes in `katgpt-forward` (or `katgpt-core`), NOT in `katgpt-hla`. The HLA forward composition already lives in `katgpt-forward` for exactly this reason.
- **Vortex decode path:** `forward_dash_attn_decode_vortex` was stripped from the leaf migration (commit `c76722d2`). To re-add, either move the `vortex_flow` cluster into a crate that can depend on `bandit`/`speculative`, or inject the router via a trait. Documented in `katgpt-attn/src/dash_attn/forward.rs` module comment. Non-blocking; not part of this issue.
