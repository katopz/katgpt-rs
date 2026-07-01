# Issue 018 — `katgpt-sense` extraction (in-flight) workspace build regression

**Status:** RESOLVED 2026-07-01 (extraction committed as `451ae9da`)
**Discovered:** 2026-07-01, while re-verifying riir-ai Issue 355 Phase 3c.
**Failure class:** same as Issue 017 (extraction → compile regression).
**Blocking:** `cargo check --workspace` in `riir-ai`, and any katgpt-rs
default-feature consumer's full build. The lib compiles in isolation
(`cargo check -p riir-engine --lib` ✅ in riir-ai) — only paths that turn
on `sense_composition` break.

## TL;DR

The `sense` module is mid-extraction into a new `katgpt-sense` crate
(Plan 338-style leaf extraction, uncommitted WIP on `develop`).
`katgpt-core/src/lib.rs:343` still declares:

```rust
#[cfg(feature = "sense_composition")]
pub mod sense;
```

…pointing at a `sense/` directory whose files have already been moved
(staged+working renames) into `crates/katgpt-sense/src/`. The module
file is gone; the `pub mod` declaration is not. Result: `E0583 file not
found for module sense` whenever `sense_composition` is on.

`sense_composition` is **default-on** in katgpt-rs (reached via
`schema_centroid = ["katgpt-core/schema_centroid", "sense_composition"]`,
which is default-on), so the breakage cascades to every downstream
workspace that consumes katgpt-rs with default features — e.g. riir-ai's
`cargo check --workspace`.

## Observed on-disk state (katgpt-rs working tree, 2026-07-01)

```
M  Cargo.toml
R  crates/katgpt-core/src/sense/spectral_threat.rs -> crates/katgpt-core/src/sense_threat.rs
RM crates/katgpt-core/src/sense/mod.rs            -> crates/katgpt-sense/src/lib.rs
RM crates/katgpt-core/src/sense/bake.rs           -> crates/katgpt-sense/src/bake.rs
RM crates/katgpt-core/src/sense/lod.rs            -> crates/katgpt-sense/src/lod.rs
RM crates/katgpt-core/src/sense/octree.rs         -> crates/katgpt-sense/src/octree.rs
R  crates/katgpt-core/src/sense/reconstruction.rs -> crates/katgpt-sense/src/reconstruction.rs
... (more renames under sense/ → katgpt-sense/src/)
?? crates/katgpt-sense/Cargo.toml   (untracked — not yet in workspace members)
```

`katgpt-core/src/lib.rs:343` (`pub mod sense;`) is **not** in the modified
set — it still resolves the old path. The `sense/` directory it points at
is being emptied by the renames above.

## Symptom

```
error[E0583]: file not found for module `sense`
  --> crates/katgpt-core/src/lib.rs:343:1
   |
343 | pub mod sense;
   | ^^^^^^^^^^^^^^
   = help: create file ".../src/sense.rs" or ".../src/sense/mod.rs"
```

Triggered by default features because:
`schema_centroid` (default-on) → `sense_composition` → `pub mod sense;`.

## Why this is filed (downstream dependency)

riir-ai Issue 355 Phase 3c ("wire closure_wire + closure_mining into
cognitive_branches_runtime", commit `6a9dd505`) enables
`cognitive_branches_runtime_closure` → `katgpt-rs/closure_instrument`,
whose feature chain reaches `sense_composition`. That Phase 3c commit's
claimed GOAT gate (14/14 `closure_bridge` tests, 129/129
`cognitive_branches_runtime` tests) **could not be independently
re-verified** at filing time because the feature chain won't compile
while katgpt-rs is in this state.

The Phase 3c code itself is fine — `cargo check -p riir-engine --lib`
(default) passes with it present. The re-verification is blocked purely
on katgpt-rs returning to a buildable state.

## Expected resolution

The extraction presumably still needs to, in some order:
1. Register `crates/katgpt-sense` in the workspace `Cargo.toml` members.
2. Remove or repoint `#[cfg(feature = "sense_composition")] pub mod sense;`
   in `katgpt-core/src/lib.rs:343` (re-export from `katgpt-sense`, or drop
   if the feature is being retired in favour of a direct `katgpt-sense`
   dependency).
3. Update `sense_composition` / `schema_centroid` feature definitions in
   `katgpt-rs/Cargo.toml` (lines ~192-209) so the default-on chain no
   longer reaches a missing module.
4. Commit the rename batch as one atomic extraction (matches the Plan 338
   Phase 1/2/2.5 cadence: `feat(katgpt-sense): extract …`).

Once katgpt-rs `develop` builds clean with default features again, riir-ai
Issue 355 Phase 3c's GOAT gate can be re-run.

## Non-blocking framing

This is a heads-up, not a blocker. The extraction is clearly intentional
and following the established Plan 338 pattern. Filed only so the
extraction agent knows:
- the workspace build is currently red downstream, and
- a riir-ai GOAT re-verification is queued behind katgpt-rs building again.

No action requested beyond completing the extraction as planned.

## Resolution (2026-07-01)

The extraction agent committed the completed work as
`451ae9da` (`feat(katgpt-sense)!: promote sense substrate to standalone
crate (Plan 338 Phase 3)`). Independently verified before/after the commit
landed:

- katgpt-rs `cargo check --workspace` ✅ green
- katgpt-rs `cargo check --workspace --all-features` ✅ green
- `cargo test -p katgpt-sense --all-features` → **85/85 pass**
- riir-ai `cargo check --workspace` ✅ green (the downstream consumer that
  was the original blocker)

The `pub mod sense;` shim at `katgpt-core/src/lib.rs` re-exports
`katgpt_sense::*` and forwards `spectral_threat` (which stayed local in
katgpt-core because it depends on `linoss`). External consumers'
`katgpt_core::sense::*` paths resolve bit-for-bit. Issue closed; the
queued riir-ai Phase 3c GOAT re-verification is now unblocked.
