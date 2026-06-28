# Issue 005 (katgpt-rs): Promote `committed_field_blend` to Default-On

> **Parent plan:** [riir-ai Plan 336 §Phase 7 T7.2](../../riir-ai/.plans/336_committed_personality_runtime_integration.md) — "Propose (do NOT unilaterally execute) katgpt-rs Plan 321 promotion of `committed_field_blend` to default-on in `katgpt-rs/crates/katgpt-core/Cargo.toml`."
> **Open primitive:** [katgpt-rs Plan 321](../.plans/321_sampling_invariant_per_entity_moe_primitive.md) — `CommittedFieldBlend<N, D>` (commit `6029d835`, Phase 4 commit `76ac861c`)
> **GOAT bench:** [321_committed_field_blend_goat.md](../.benchmarks/321_committed_field_blend_goat.md) — G1–G5 ALL PASS
> **Runtime validation:** [riir-ai Plan 336](../../riir-ai/.plans/336_committed_personality_runtime_integration.md) — G6a–G6e + G7a ALL PASS, `committed_personality_runtime` promoted to default-on 2026-06-26
> **Source paper:** [arXiv:2510.00621](https://arxiv.org/abs/2510.00621) — FAME (Gao/Chen/Zhang, NeurIPS 2025)
> **Status:** OPEN — proposal only (per T7.2 "do NOT unilaterally execute"). Closes when the promotion lands and A1–A4 pass.
> **Created:** 2026-06-28.

---

## TL;DR

`committed_field_blend` is the open `katgpt-core` primitive (Plan 321) that
ships the FAME sampling-invariant per-entity MoE: a frozen sigmoid blend of
K=3 archetype operator fields whose weights `π` are computed ONCE from a
trajectory summary and committed via BLAKE3. Its defining property is
**sampling invariance** (FAME Prop. 3) — dense vs sparse observation of the
same trajectory yields bit-identical `π` and bit-identical dynamics.

The feature has been **opt-in since landing** (2026-06-25) pending two
deferral gates cited in the GOAT bench:

1. **Phase 4** of Plan 321 (examples + docs) — ✅ SHIPPED commit `76ac861c`.
2. **Runtime-integration validation** in riir-ai Plan 336 — ✅ SHIPPED 2026-06-26
   (all 7 phases done; G6a–G6e crowd-scale gates + G7a frozen-restoration
   bit-identical; `committed_personality_runtime` promoted to default-on in
   `riir-engine/Cargo.toml`).

**Both deferral conditions are satisfied.** This issue proposes the
promotion: add `committed_field_blend` to the `default` feature list in
`crates/katgpt-core/Cargo.toml` (and add the root passthrough +
default-list entry in `katgpt-rs/Cargo.toml` so workspace consumers see it
without `--features`).

The gain is **modelless** (closed-form sigmoid projection + BLAKE3 hash, no
training), so per `katgpt-rs/AGENTS.md` §Feature Flag Discipline the GOAT
gate passing unlocks promotion to default-on.

---

## 1. The proposal

### 1.1 Primary change — `crates/katgpt-core/Cargo.toml`

The feature is already defined at line 212:

```toml
committed_field_blend = ["personality_composition"]  # CommittedFieldBlend — ... Opt-in until G1–G5 GOAT gate passes; G2 (sampling invariance) is the make-or-break gate.
```

The change is one token in the `default` list (line 90-100):

```diff
 [features]
-default = ["sparse_mlp", "plasma_path", ..., "personality_composition", "depth_invariance", ..., "best_belief"]
+default = ["sparse_mlp", "plasma_path", ..., "personality_composition", "committed_field_blend", "depth_invariance", ..., "best_belief"]
```

Since `committed_field_blend = ["personality_composition"]`, enabling it
transitively enables `personality_composition` (already default — no change
observed by existing consumers). The comment on the feature line should be
updated from "Opt-in until G1–G5 GOAT gate passes" to the standard
DEFAULT-ON form citing the gate evidence (see §2).

### 1.2 Secondary change — root `katgpt-rs/Cargo.toml`

The root workspace Cargo.toml currently has **no `committed_field_blend`
passthrough feature at all** — only `personality_composition` is re-exposed:

```toml
personality_composition = ["katgpt-core/personality_composition"]  # ... DEFAULT-ON after GOAT G4 + G5 ...
```

Add the passthrough + the default-list entry:

```toml
committed_field_blend = ["katgpt-core/committed_field_blend"]  # CommittedFieldBlend — sampling-invariant per-entity MoE (Plan 321, Research 302, arxiv 2510.00621 FAME). DEFAULT-ON after Plan 321 G1–G5 + riir-ai Plan 336 G6a–G6e + G7a all PASS (2026-06-26). Implies personality_composition. Modelless gain: closed-form sigmoid projection + BLAKE3 commit, no training.
```

Then add `"committed_field_blend"` to the root `default = [...]` list
(line 67, the long feature list). The existing entry `personality_composition`
already appears in root default; adding `committed_field_blend` next to it
mirrors the structure.

### 1.3 Doc updates

- `katgpt-rs/.docs/01_overview.md` Feature Flags table — flip the
  `committed_field_blend` row from "Opt-in — Promotion deferred pending
  riir-ai Plan 336 runtime-integration validation" to "DEFAULT-ON — G1–G5
  (Plan 321) + G6a–G6e + G7a (riir-ai Plan 336) all PASS, 2026-06-26.
  Modelless."
- `katgpt-rs/.benchmarks/321_committed_field_blend_goat.md` Promotion status
  section — replace the "DEFERRED pending Phase 4 + Plan 336" note with a
  "PROMOTED to default-on (Issue 005)" note pointing at this issue and the
  commit that lands the promotion.
- `katgpt-rs/README.md` Feature Showcase — flip the Opt-In & Gated Features
  table row for `committed_field_blend` to a DEFAULT-ON row (mirrors how
  Plan 297 `personality_composition` and Plan 308 `karc_forecaster` were
  promoted).

---

## 2. Why now (the evidence)

### 2.1 Open-primitive GOAT gate (Plan 321, G1–G5)

From [.benchmarks/321_committed_field_blend_goat.md](../.benchmarks/321_committed_field_blend_goat.md):

| Gate | Property | Result |
|------|----------|--------|
| **G1** | mechanics (finite, bounded, sigmoid in [0,1]) | ✅ PASS |
| **G2** | sampling invariance (dense vs sparse → identical π + dynamics) — **the defining property** | ✅ PASS (worst-case Δπ=1.19e-6 over 100 entities) |
| **G3** | no regression on `PersonalityWeightedComposition` primitives | ✅ PASS |
| **G4a** | `apply_blended` zero-alloc (1000 iters) | ✅ **0 allocs** |
| **G4b** | `commit` zero-alloc (100 re-commits) | ✅ **0 allocs** |
| **G5** | BLAKE3 reproducible + tamper-detecting | ✅ PASS (4/4) |

13 unit tests + 1 bench, all green. Zero `#[ignore]`, zero threshold weakening.

### 2.2 Runtime-validation GOAT gate (riir-ai Plan 336, G6a–G6e + G7a)

From [riir-ai/.benchmarks/336_committed_blend_promotion_review.md](../../riir-ai/.benchmarks/336_committed_blend_promotion_review.md):

| Gate | Property | Result | Margin |
|------|----------|--------|--------|
| **G6a** | Crowd-scale blend diversity (10K NPCs) | ✅ PASS | p95 1.946 ≥ 0.3·range 0.770 (**2.5×**), 0 bit-identical pairs |
| **G6b** | Sampling invariance under fog-of-war | ✅ PASS | 100% under 1e-3 (div=0, by construction) |
| **G6c** | Replay determinism (1000 NPCs) | ✅ PASS | 0 π + 0 BLAKE3 mismatches |
| **G6d** | Latency at scale | ✅ PASS | median 0.177ms ≤ 10ms (**56×** under budget) |
| **G6e** | No regression (light, 1000 ticks) | ✅ PASS | dz bounded, lipschitz drift 0.0 < 1e-6 |
| **G7a** | Frozen-restoration (Phase 5 cross-repo) | ✅ PASS | bit-identical dz after freeze→drop→thaw→tick |

The weakest margin is G6a at 2.5×; the strongest is G6d at 56× headroom.

### 2.3 Modelless gain (the promotion criterion)

Per `katgpt-rs/AGENTS.md` §Feature Flag Discipline: *"Promotion requires
modelless gain. A perf gain on a biased/incorrect answer is NOT a modelless
gain — it's a speedup of a wrong result. The quality gate (G1 or equivalent)
must pass modellessly for the GOAT to hold."*

The committed personality blend is **modelless by construction**:
- `π` is a closed-form projection: `pi_k = clamp(dot(summary, direction_k), -pi_max, +pi_max)`. No gradient descent.
- The blend is frozen via BLAKE3 at commit time; the hot path reads only `blend.tau`/`blend.pi_max` (one cache line).
- Re-commit (Plan 336 Phase 5 T5.4) is a fresh projection on a major event — still no training.
- Freeze/thaw (Plan 336 Phase 5 T5.2) is a bit-exact Pod round-trip — no learned state, just committed scalars.

The gain (per-NPC emergent personality at crowd scale, 56× under the latency
budget, sampling-invariant under fog-of-war) is achieved without a single
gradient step. **The modelless-gain criterion is satisfied.**

### 2.4 Standalone composition (graceful degradation)

`committed_field_blend` works standalone and is **zero-cost when unused**:
- The primitive is `#[cfg(feature = "committed_field_blend")]`-gated; promoting to default does not pull in any new runtime dependency (it already implies `personality_composition`, which is itself default).
- No `karc_runtime`, no `committed_blend_freeze`, no chain/shard dependency.
- Promotion means the primitive is *available* without `--features`, NOT that any caller is forced to invoke it.

### 2.5 Latent-vs-raw boundary (held)

Per the global AGENTS.md latent-vs-raw rules — verified by the G7a
frozen-restoration test (bit-identical dz) and the chain-bridge unit tests:
- The full `π: [f32;3]` + `blake3` + `lipschitz_bound` are per-NPC latent-local.
- Only the 4 raw scalars (3 `pi` + 1 `lipschitz_bound`) cross the sync boundary as RAW committed scalars via the `BlendSnapshotCommit` POD.
- The archetype field DEFINITIONS never cross sync — they are library-side, referenced by `archetype_library_hash` only.

Promoting the open primitive to default-on does not change this boundary;
the boundary is enforced at the runtime layer (riir-ai Plan 336), not at the
primitive layer (this crate).

---

## 3. Scope (what this issue IS and IS NOT)

### IS

- One-token additions to two `default = [...]` feature lists (`crates/katgpt-core/Cargo.toml` + root `Cargo.toml`).
- One new passthrough feature line in root `Cargo.toml`.
- Three doc updates (overview table cell, GOAT bench promotion-status note, README feature-showcase row).
- Updating the feature-flag comment on the `committed_field_blend = [...]` line to cite the post-promotion evidence (G1–G5 + G6a–G6e + G7a).
- Verifying `cargo check --no-default-features` still compiles (the primitive's `#[cfg(feature=...)]` gate isolates it cleanly).

### IS NOT

- A behavior change. The primitive code itself is untouched — same module, same tests, same bench. Only the feature-default flag moves.
- A new dependency. `committed_field_blend = ["personality_composition"]` — the implied dep is already default-on.
- A runtime-mode default flip. `riir-ai` `HlaUpdateMode` stays `LeakyIntegrator` (per-NPC opt-in via `CommittedBlend`). Promoting the open primitive to default-on means it's *available* without `--features`; the riir-ai runtime decides whether any NPC actually uses it.
- A freeze/thaw substrate promotion. The persistence layer (`committed_blend_freeze`, ArchetypeBlendShard) stays opt-in — only the in-memory primitive is promoted here.
- An upstream training task. The K=3 HLA archetype library (riir-train Issue 307) is unrelated to this promotion — the open primitive's GOAT gate was satisfied with synthetic fields (sin/cos/linear) and remains correct under any archetype library.

---

## 4. Acceptance criteria

The issue closes when ALL of the following pass:

- **A1 (default compiles clean).** `cargo check` (no `--features`) on `katgpt-rs` and on `katgpt-core` succeeds with `committed_field_blend` in the default list. No new warnings vs the current opt-in build.
- **A2 (no-default still clean).** `cargo check --no-default-features` on both crates still compiles — the primitive is properly `#[cfg(feature = "committed_field_blend")]`-gated and does not leak into the no-default build.
- **A3 (tests still green).** `cargo test -p katgpt-core --lib committed_field_blend` passes the same 13 tests with no behavior change. `cargo bench -p katgpt-core --bench committed_field_blend_bench -- --nocapture` still reports 0 allocs on both G4a and G4b.
- **A4 (docs consistent).** The Feature Flags table in `.docs/01_overview.md`, the GOAT bench promotion-status section, and the README showcase row all read "DEFAULT-ON" with the gate evidence cited. No stale "Opt-in" or "deferred pending Plan 336" language remains anywhere in the repo for this feature.

A1–A4 are verifiable in under five minutes total (two `cargo check` invocations
+ one `cargo test` + one `cargo bench` + a `grep` for stale "Opt-in" text).
No long-running benchmark, no training run, no cross-repo coordination.

---

## 5. Anti-checklist (do NOT do these)

- ❌ Do NOT change `CommittedFieldBlend` code, tests, or bench. The primitive is correct as-shipped; this is purely a feature-default promotion.
- ❌ Do NOT promote `committed_blend_freeze` (the persistence layer) — that stays opt-in. It depends on `ArchetypeBlendShard` storage (riir-neuron-db Research 009) and is a separate concern from the in-memory primitive.
- ❌ Do NOT change the default `HlaUpdateMode` in riir-ai. That's a riir-ai decision, already settled in Plan 336 T7.1 (stays `LeakyIntegrator`, per-NPC opt-in via `CommittedBlend`).
- ❌ Do NOT add a root passthrough for `committed_blend_freeze`. Only the in-memory primitive is in scope here.
- ❌ Do NOT bundle this with the riir-train K=3 archetype library work (Issue 307). They are independent — this promotion's GOAT gate was satisfied with synthetic fields; the trained library is commercial-value only.
- ❌ Do NOT use softmax anywhere in the promotion review. The primitive uses sigmoid by design (per AGENTS.md); the review text must reflect that.
- ❌ Do NOT skip A2 (`--no-default-features`). The `merkle_root` lesson (katgpt-rs/AGENTS.md) applies: feature-default changes can break the no-default build in non-obvious ways, especially when the feature implies another feature that has its own gate.

---

## 6. Cross-references

- **Parent plan (the trigger):** [riir-ai Plan 336 §Phase 7 T7.2](../../riir-ai/.plans/336_committed_personality_runtime_integration.md)
- **Open primitive plan:** [katgpt-rs Plan 321](../.plans/321_sampling_invariant_per_entity_moe_primitive.md)
- **Open primitive GOAT bench:** [321_committed_field_blend_goat.md](../.benchmarks/321_committed_field_blend_goat.md)
- **Runtime GOAT bench:** [riir-ai 336_committed_blend_promotion_review.md](../../riir-ai/.benchmarks/336_committed_blend_promotion_review.md)
- **Runtime G6 results:** [riir-ai 336_committed_blend_g6_results.md](../../riir-ai/.benchmarks/336_committed_blend_g6_results.md)
- **Parent research (private guide):** [riir-ai Research 158](../../riir-ai/.research/158_per_npc_committed_personality_blend_guide.md)
- **Open primitive research:** [Research 302](../.research/302_FAME_Sampling_Invariant_Per_Entity_MoE.md)
- **Upstream training tracker (UNRELATED, do not bundle):** [riir-train Issue 307](../../riir-train/.issues/307_hla_archetype_field_library_training.md)

---

## TL;DR (closing note)

This is the smallest promotion in the committed-personality Super-GOAT
corpus: one default-list token + one passthrough feature + three doc cells.
The heavy lifting (G1–G5 on the primitive, G6a–G6e + G7a on the runtime) is
already done and committed. Promotion is the formal flip from "opt-in" to
"available by default" — the modelless gain is established, the deferral
conditions are met, the standalone composition is verified. A1–A4 should
clear in one CI cycle.
