# Issue 968 — kill the `block v0.1.6` future-incompat (vendored `[patch.crates-io]` fork; the "dependency-bump lane" has NO bump)

**Status:** LANDED 2026-10-27 — vendored fork + `[patch.crates-io]` at
`5e216a761` (all gates green, see below). Open residue: the other 7 locks
carrying registry `block 0.1.6` (riir-ai, riir-infer, riir-instinct,
riir-refine, riir-reflex, riir-rethink, riir-train) each need the same
per-workspace patch; measured reachability today = katgpt-rs + riir-train,
the other six pull it only under feature postures (their default
`cargo tree -i block@0.1.6` does not resolve it — trace with
`--all-features` when picking the patch up).
**Lease closed:** zed-amo-blockfix (claimed 1793121000) — the claimed work
landed same day at `5e216a761`.
**Claimed:** zed-amo-refine-blockpatch, 1793079375, expires 2026-10-29T05:45Z
(residue slice: the riir-refine lock only — same vendored fork +
`[patch.crates-io]` row; the other six locks stay unclaimed)

## The finding (why the bump lane cannot exist)

1. `block` 0.1.6 is the TERMINAL version — the registry has no 0.1.7+ (index
   checked 2026-10-27). Upstream (SSheldon/rust-block) is abandoned: last
   release 2021-01.
2. The only in-workspace consumer chain is `metal 0.33.0` →
   `crates/katgpt-backend` (optional `gpu_inference` feature). The registry
   index for `metal` shows **0.33.0 is also terminal and EVERY version
   0.11-0.33 depends on legacy `block` + `objc`** — there is no metal release
   that moved to `block2`/`objc2`. A `metal` bump changes nothing.
3. The actual future-incompat (reproduced in a scratch crate, rustc
   future-incompat report): **`static of uninhabited type`** (rust-lang/rust
   #74840) at `block-0.1.6/src/lib.rs:64` — `enum Class { }` (uninhabited)
   declared as the type of `extern static _NSConcreteStackBlock`. It becomes a
   **hard error in a future rustc release**. The static is only ever used for
   its ADDRESS (the ObjC block ABI's stack-block class pointer), never read as
   a value, so an inhabited opaque type is a behavior-identical fix.
4. Cross-check: zed, riir-shader, riir-auth, riir-refine graphs are ALREADY
   clean (block2 only) — the old HISTORY mentions are stale. Live scope is
   katgpt-rs (this repo) + riir-train (inherits via the katgpt-rs dev-dep
   chain; its lock picks the fix up only when ITS lane adds the same
   `[patch.crates-io]` — routing restriction keeps that lane out of this one).

## The fix

- `vendor/block/` — a vendored fork of block 0.1.6, MIT license preserved,
  with ONE delta: `enum Class { }` → an inhabited opaque `#[repr(C)]` struct.
  `Class` has exactly 3 use sites in the crate (the def, the extern static,
  an `isa: *const Class` ABI field) — all compile unchanged with the struct.
- Root `Cargo.toml` gains:

  ```toml
  [patch.crates-io]
  block = { path = "vendor/block" }
  ```

  Same name + version (0.1.6) so the patch slots into the existing graph with
  no requirement churn.
- `FORK.md` in the vendored dir records provenance + the removal condition:
  delete the whole patch when `katgpt-backend` migrates `metal` ->
  `objc2-metal` (the only real exit), or when upstream ever ships a fixed
  release.

## Gates (all measured green at `5e216a761`, isolated worktree,
CARGO_TARGET_DIR=/tmp/krs-blockfix-target)

- [x] `cargo build -p katgpt-backend --features gpu_inference` — completes
      with NO future-incompat report (previously named block v0.1.6); the
      fork compiles from `vendor/block` (path source in the build log).
- [x] **Liveness (replaced the gate below — see the note):** an ObjC-block
      harness against the fork (ConcreteBlock -> heap copy -> real runtime
      call, captured-variable readback) returns correct results — the ABI
      type change broke nothing. `cargo test -p block` is IMPOSSIBLE for a
      patched non-member ("requires dev-dependencies and is not a member of
      the workspace"), so the crate's own doc tests cannot run in-workspace;
      the harness is the stronger proof (public API + real ObjC runtime).
- [x] `cargo clippy -p katgpt-backend --features gpu_inference --all-targets`
      — ZERO warnings (en route: the vendored 2021-era `extern` sites tripped
      46 current-toolchain ABI deprecations under macro expansion; fixed
      mechanically in the fork with explicit "C" annotations, re-measured
      zero).
- [x] Cargo.lock `block` row: source registry -> path, version still 0.1.6.
- [x] Default (no-feature) posture + `cargo build -p katgpt-rs`: clean, no
      unused-patch noise.

## Residuals (NOT this issue)

- riir-train's own lock: same patch needed THERE (one `[patch.crates-io]` +
  the vendored dir or a relative path) — its lane's to land under the routing
  restriction.
- Any future repo that pulls `metal` (or any legacy-`objc`-family crate)
  inherits the warning until it adds the same patch — the patch is
  per-workspace by design.
