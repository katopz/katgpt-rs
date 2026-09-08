# Issue 739 — katgpt-rs carries no rust-toolchain.toml: the box-default toolchain (1.93.0) fails to build HEAD (E0658 on katgpt-percepta `isolate_lowest_one`)

**Status:** OPEN — filed 2026-09-08 by the 4090 league session; evidence measured during riir-ai Issue 884 T5 (Bench 885 en-route fix 2); fix is a one-file, owner-reviewed commit (the toolchain pin is "a deliberate act" per the sibling pin's own header). ⚠ RENUMBERED from 738 the same day: the 4090 session and the wasm32-surface session dual-allocated 738 concurrently (`.highwater` read 737 in both); `738_wasm32_surface_the_package_set_itself_is_unpinned.md` keeps the number — it was landed first and is cited by AGENTS.md + the audit script. No other citation existed to rewrite.

## The defect

katgpt-rs is the only repo in the 17-repo workspace with **no `rust-toolchain.toml`**. Every
build therefore uses the box default — **1.93.0 on both the M3 Max and the 4090 box**
(`rustup default`, recorded 2026-08-12 in riir-ai's pin header) — while current HEAD uses an
API that is E0658-unstable there:

- `crates/katgpt-percepta/src/legacy/mod.rs` (`Sudoku9x9::solve_fast_rec`, ~L470):
  `bits.isolate_lowest_one()` — `u32::isolate_lowest_one` stabilized after 1.93 (compiles at
  the workspace's 1.98.1 pin).
- Measured (2026-09-07, riir-ai Bench 885): the league harness built katgpt-rs via path dep at
  the box default → `error[E0658]: isolating the lowest set bit is unstable` → the run failed
  to compile before measuring anything. The T5 fix was dropping the stale
  `RUSTUP_TOOLCHAIN=1.95.0` manifest override so the **riir-ai** workspace pin (1.98.1)
  governed — which papers over the katgpt-rs gap for sibling builds but leaves every DIRECT
  katgpt-rs build (clone → `cargo check`, the public funnel's first-contact path) broken on
  stock boxes.

## Why it bites this repo hardest

katgpt-rs is the **public** repo — external consumers and contributors clone it standalone,
where no sibling workspace pin rescues the build. The private repos all pin 1.98.1
(owner-directed 2026-09-04: riir-ai, riir-train, riir-chain, riir-clippy — "the stack is
single-toolchain again"); katgpt-rs was left unpinned, so it is the one repo where the
effective toolchain is whatever the box happens to have.

## Proposed fix (one file, mirroring the sibling pin)

`rust-toolchain.toml` at repo root:

```toml
[toolchain]
channel = "1.98.1"
components = ["clippy", "rustfmt"]
```

- Matches the owner-directed 2026-09-04 stack pin (no new toolchain decision to make).
- Cost: `rustup` downloads 1.98.1 on boxes that don't have it — the same cost every other
  workspace repo already carries; both our boxes have it installed.
- Optional hardening (separate commit if wanted): `rust-version = "1.98.1"` in
  `[workspace.package]` so cargo *resolution* also refuses older toolchains with a clear
  message (the riir-ai pin header documents the resolution-time enforcement precedent via
  cubecl-zspace's `rust-version`).
- Raising the pin later stays "a deliberate act … in its own reviewed commit" (sibling pin
  header convention).

## Tasks

- [ ] T1: add `rust-toolchain.toml` (channel 1.98.1 + clippy/rustfmt) — owner review, own commit.
- [ ] T2: verify a default-toolchain build: fresh shell (no `+toolchain`, no
      `RUSTUP_TOOLCHAIN`) `cargo check --workspace` + `cargo test -p katgpt-core --lib` green.
- [ ] T3 (optional): `rust-version` in `[workspace.package]`; document in README build section
      that the pin is authoritative.

## References

- Measured evidence: `../riir-ai/.benchmarks/885_issue884_t5_league_repin.md` §"En-route
  fixes" item 2 (E0658 at the box default, the stale-override mechanism).
- The sibling pin + its rationale header: `../riir-ai/rust-toolchain.toml` (1.98.1,
  owner-directed 2026-09-04).
- Use site: `crates/katgpt-percepta/src/legacy/mod.rs` (`isolate_lowest_one`).
