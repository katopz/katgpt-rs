# Plan 624 — Runetrace T2.1: the scene-as-text format crate (`katgpt-core::runetrace`)

**Status:** complete — T1–T7 done; measured 2161/0/8 at `--features runetrace --lib --test-threads=2` (the test-gate row floor); clippy clean at `--features runetrace --all-targets -D warnings` + default; count_features 8/8 sites green. NOTE: docs_gate has 2 PRE-EXISTING reds unrelated to this change (`skill_repo_set_gate` + `population_sync_gate`: `repo_set.txt` missing `seal-std` — live-workspace drift from a sibling's new repo; the fix belongs with seal-std's registration, not this plan).
**Proposal:** seal-remake `.proposals/005_runetrace_scene_condition_dag.md` (Phase 2 / T2.1)
**Branch:** `develop`
**Repos:** katgpt-rs (this plan) — T2.2/T2.3 land under their own riir-ai plan

---

## Why

The live game scene has no canonical TEXT form. Every consumer (hero balloon,
scenario panel, LLM lanes) hand-rolls its own translation, and an LLM agent
cannot consume the scene without bespoke glue. Proposal 005 (Runetrace) puts
the format where instinct/rethink can reach it without breaking the boundary
law: katgpt-core, the sibling of the frozen `decision_wire`.

## Scope of this plan (T2.1 only)

- [x] T1 `runetrace = []` feature (opt-in, the no-default-consumer rule) in
      `crates/katgpt-core/Cargo.toml` with the provenance comment.
- [x] T2 `crates/katgpt-core/src/runetrace.rs` — the DOC model
      (`RunetraceDoc` / `SceneHeader` / `EntityBlock` / `ConditionRow` /
      `DagRow` / `EntityKind` / `DagStage`), fail-closed `validate()`, the
      deterministic glyph-led text renderer (`render_into` + `render`), the
      BLAKE3 digest (`content_hash`), `RunetraceError`. Zero new deps (serde +
      blake3 already non-optional). Float spelling = Rust shortest-round-trip
      Display, pinned by golden fixtures.
- [x] T3 lib.rs gate `#[cfg(feature = "runetrace")] pub mod runetrace;` with
      the provenance doc comment (decision_wire sibling placement).
- [x] T4 G1 fixtures inline (`mod tests`): golden canonical text byte-pin,
      determinism (render twice + digest equal), serde JSON + postcard
      round-trips + JSON byte-stability, validate-accepts-canonical, one
      refusal arm per error variant, float-spelling pins, dag depth indent.
- [x] T5 Measure `cargo test -p katgpt-core --features runetrace --lib` and
      add the `katgpt-core:2161:runetrace` row to `scripts/test_gate.sh`
      (the green-zero defense: the module's tests are INVISIBLE at default
      features — the decision_wire precedent).
- [x] T6 README + examples/README flag-count bumps (678 → 679 total; opt-in,
      default-on 205 unchanged) at the spots `count_features.py` pins.
- [x] T7 `cargo clippy -p katgpt-core --features runetrace --all-targets` +
      default-features clippy (G3: default build bit-identical — feature off
      compiles the module to nothing) + `python3 scripts/count_features.py`.
- [x] T8 Commit + push; update `.highwater` (624) in the same commit.

## Out of scope (their own plans)

- T2.2 hero strategy metadata table (riir-ai) — the pod's enum→str becomes data.
- T2.3 `MonsterCognitionDebugPod`, ring-gated (riir-ai).
- Phase 3+ panels/lanes/sleep (seal-remake / instinct / rethink / ndb).

## Design decisions (the digest law drives all of them)

1. **Renderer allocates — by design.** Runetrace is a panel/lane/lane-debug
   artifact, never the tick path (proposal caveat 2). `render_into(&mut
   String)` allows buffer reuse across belt-cadence renders.
2. **Fail-closed render.** `render`/`render_into`/`content_hash` validate
   first and return `Result` — an unvalidated doc never produces text.
3. **Closed vocabularies** (`DagStage` 6 stages, `EntityKind` 4 kinds) —
   additive-only after adoption; free text lives in producer-authored fields.
4. **Position is `[f32; 3]` Option** — the MapPos3D f32 canonical; absent
   when a producer has no position. Vitals are generic `{name, value, max}`.
5. **No `fired` flag on think rows** — presence in `think` MEANS the
   condition fired this belt (proposal §4.1 wording); the dag section is the
   ordered decision path.
6. **DagRow.parent is `Option<u32>`**, must reference an EARLIER row
   (validate: parent < own index — forward refs and self-refs refused).

## Render grammar (v1, byte-pinned by fixtures)

```
runetrace/v1 tick 1204 scene Z1 rain
🧭 hero · player
  pos 12.5 3.2 0
  hp 12/40
  if hp_frac < 0.25 → drink_red (drive 0.81)
  perceive bee_warrior dist 4.1
    believe threat high
```

Entity line = `<glyph> <id> · <kind>`; fields two-space indented; dag depth =
two more spaces per depth; drives `(drive <f>)` shortest-round-trip.
