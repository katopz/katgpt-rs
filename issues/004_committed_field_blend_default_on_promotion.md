# Issue 004: Promote `committed_field_blend` to Default-On (Runtime Validation Complete)

**Date:** 2026-06-25
**Parent:** Plan 321 (Sampling-Invariant Per-Entity MoE Primitive), katgpt-rs Research 302
**Cross-ref (runtime):** riir-ai Plan 336 (Committed Personality Blend Runtime Integration)
**Type:** Promotion (default-OFF → default-on)
**Priority:** Medium — unblocks commercial-grade runtime; not a correctness bug
**Status:** 🟡 PROPOSED (awaiting GOAT-gate cross-check + maintainer sign-off)

---

## Proposal

Promote the `committed_field_blend` feature in
`katgpt-rs/crates/katgpt-core/Cargo.toml` from opt-in to **default-on**.

The primitive already passed its own katgpt-rs GOAT gate (G1–G5, commit `6029d835`,
bench `.benchmarks/321_committed_field_blend_goat.md`). Plan 321's deferral note
explicitly said the open primitive is eligible for promotion *the moment the
riir-ai runtime validation plan lands*. Plan 336 Phases 1–4 + 6 landed on
2026-06-25 (commits `67ad96e8` → `71bd5ea8`), supplying the runtime-validation
gate the deferral was waiting on.

---

## Why now (the trigger)

Plan 336 Phase 6 — **G6 Crowd-Scale Validation** — passed all five gates in
`--release`:

| Gate | Metric | Result | Target |
|------|--------|--------|--------|
| G6a | p95 pairwise ‖π_i − π_j‖ over 10,000 NPCs | **1.946** | ≥ 0.770 |
| G6b | % NPCs with fog-of-war divergence < 1e-3 | **100%** | ≥ 99% |
| G6c | π / BLAKE3 mismatches across replay | **0** | 0 (bit-identical) |
| G6d | median total / per-NPC at 10k scale | **0.177 ms / 0.0177 µs** | ≤ 10 ms / ≤ 1 µs |
| G6e | dz max norm / lipschitz drift | **0.343 / 0.0** | ≤ 2.0 / < 1e-6 |

Full evidence: `riir-ai/.benchmarks/336_committed_blend_g6_results.md`.

The defining property — **sampling invariance under fog-of-war** — holds at crowd
scale (G6b) *for the right reason*: the committed `π` is frozen at commit time,
and `tick_committed_blend` ignores per-tick summary once committed. This is the
committed-runtime contract, not a vacuous pass (the primitive-level FAME
Proposition 3 invariance was already proven in Plan 321 G2).

---

## Why this is a modelless gain (AGENTS.md mandate)

The GOAT-gate promotion rule (katgpt-rs/AGENTS.md §Feature Flag Discipline)
requires:

> Promotion requires modelless gain. A perf gain on a biased/incorrect answer
> is NOT a modelless gain.

- ✅ **G1 correctness** — primitive passes modellessly (5/5 unit tests).
- ✅ **G2 sampling invariance** — the FAME Proposition 3 property, proven
  modellessly (no training, no gradient descent — only freeze/thaw + sigmoid
  projections on direction vectors).
- ✅ **G3 no-regression** — Plan 297 `PersonalityWeightedComposition` GOAT bench
  unaffected (verified during Plan 336 Phase 6 G6e smoke check).
- ✅ **G4 alloc-free** — `apply_blended` is zero-allocation (Plan 321 G4).
- ✅ **G5 BLAKE3 commitment** — bit-reproducible + tamper-detecting.

No weight mutation. No training. No backprop. The only runtime mutations are
the three modelless paths allowed by katgpt-rs/AGENTS.md (freeze/thaw,
raw/lora hot-swap, latent-space updates).

---

## Scope of the change

### `katgpt-rs/crates/katgpt-core/Cargo.toml`

```toml
[features]
default = [
    # ... existing defaults ...
    "committed_field_blend",  # was: opt-in only; now default-on
]
```

### Downstream (already prepared)

- `riir-ai/crates/riir-engine` — `committed_personality_runtime` feature still
  opt-in (it will be promoted to default-on in Plan 336 T7.1 *after* Phase 5
  cross-repo lands). Promoting the katgpt-rs primitive to default-on does NOT
  force `committed_personality_runtime` on in riir-ai — the runtime feature is
  separately gated.
- Other consumers of katgpt-core that don't reference `committed_field_blend`
  see no behavior change (the primitive is feature-scoped to
  `src/committed_blend/`).

---

## What this promotion does NOT do

- ❌ Does NOT promote `committed_personality_runtime` to default-on in riir-ai
  (that's Plan 336 T7.1, blocked on Phase 5 cross-repo).
- ❌ Does NOT enable KARC Bi-NCDE composition by default (KARC remains opt-in
  via `karc_runtime`).
- ❌ Does NOT ship a trained archetype library (still upstream in riir-train;
  the primitive ships with synthetic sin/cos/linear validation fields).

---

## Acceptance Criteria

- [ ] `cargo check -p katgpt-core` (default features) still compiles with
      `committed_field_blend` in the default set.
- [ ] `cargo test -p katgpt-core --lib` — all existing tests still pass.
- [ ] `cargo hack check --each-feature -p katgpt-core` — no single-feature
      regression.
- [ ] `cargo hack check --all-features -p katgpt-core` — no combo-only
      regression (the `merkle_root`-class lesson).
- [ ] Plan 321 G1–G5 bench re-run, numbers unchanged.
- [ ] Update Plan 321 status line to "promoted to default-on (YYYY-MM-DD)".
- [ ] Update katgpt-rs README / feature-flags doc if one exists for the
      primitive.

---

## References

- **Open primitive:** `katgpt-rs/crates/katgpt-core/src/committed_blend/`
- **Plan 321:** `katgpt-rs/.plans/321_sampling_invariant_per_entity_moe_primitive.md`
- **GOAT bench (katgpt-rs):** `katgpt-rs/.benchmarks/321_committed_field_blend_goat.md`
- **Research (katgpt-rs):** `katgpt-rs/.research/302_FAME_Sampling_Invariant_Per_Entity_MoE.md`
- **Runtime validation (riir-ai):** `riir-ai/.plans/336_committed_personality_runtime_integration.md`
- **G6 results (riir-ai):** `riir-ai/.benchmarks/336_committed_blend_g6_results.md`
- **Private guide (riir-ai):** `riir-ai/.research/158_per_npc_committed_personality_blend_guide.md`

## TL;DR

Plan 321's deferral is satisfied: Plan 336 Phases 1–4 + 6 landed with all five
G6 crowd-scale gates green (G6b sampling invariance = 100% under fog-of-war,
G6d latency 56× under budget). Promote `committed_field_blend` from opt-in to
default-on in `katgpt-core/Cargo.toml`. This is a modelless gain (no training,
no weight mutation — only freeze/thaw + sigmoid projections). The downstream
`committed_personality_runtime` runtime feature stays opt-in pending Plan 336
Phase 5 cross-repo.
