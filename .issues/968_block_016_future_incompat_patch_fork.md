# Issue 968 — kill the `block v0.1.6` future-incompat (vendored `[patch.crates-io]` fork; the "dependency-bump lane" has NO bump)

**Status:** OPEN — root cause measured 2026-10-27; the named "dependency-bump
lane" is IMPOSSIBLE (see the finding), so this lands as a vendored 1-line-fix
fork instead.
**Claimed:** zed-amo-blockfix, 1793121000, expires 2026-10-29T00:00:00Z

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

## Gates

- [ ] `cargo build -p katgpt-backend --features gpu_inference` (isolated
      worktree, fresh lock) completes with **no future-incompat report naming
      block**.
- [ ] `cargo test -p block` green in-workspace (the patched crate's doc tests
      exercise REAL ObjC blocks on macOS — the liveness proof that the ABI
      type change broke nothing).
- [ ] `cargo clippy -p katgpt-backend --features gpu_inference --all-targets`
      clean (repo rule: clippy over check).
- [ ] Cargo.lock source row for `block` moves registry -> path, version still
      0.1.6.

## Residuals (NOT this issue)

- riir-train's own lock: same patch needed THERE (one `[patch.crates-io]` +
  the vendored dir or a relative path) — its lane's to land under the routing
  restriction.
- Any future repo that pulls `metal` (or any legacy-`objc`-family crate)
  inherits the warning until it adds the same patch — the patch is
  per-workspace by design.
