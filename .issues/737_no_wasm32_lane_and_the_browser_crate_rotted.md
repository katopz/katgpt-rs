# Issue 737 — nothing in this repo compiles for wasm32, and the browser crate had 15 live findings to prove it

**Status:** OPEN — T0–T3 landed (18 lint lines healed + `full_gate.sh` layer 2b, canaried); T4 (CI cadence) is an owner call.
**Owner:** this repo (`scripts/full_gate.sh`, `crates/katgpt-moka-wasm`, `crates/katgpt-types/src/simd`, `crates/katgpt-attn-match`).

## The finding

AGENTS.md already argues this exact point for a different axis:

> **The inverse holds too:** running **on** macOS silently drops every
> `not(target_os = "macos")` backend, `--all-features` included — **a platform
> is part of the claim, exactly as the profile is.**

`wasm32` is a second platform axis and it had no gate at all. Measured:

```
$ ls scripts/*.sh | xargs grep -ln wasm32
scripts/build-moka-wasm.sh
```

One hit, and it is not a gate — it is the deploy build script a human runs.
`full_gate.sh`, `test_gate.sh`, `docs_gate.sh` and every workflow ran zero
wasm32 builds, over a surface of **16 tracked `*.rs` files** carrying a
`target_arch = "wasm32"` cfg across `katgpt-core`, `katgpt-types` and
`katgpt-moka-wasm`.

⛔ **It is invisible on TWO axes, not one, and that is why it rotted.** The
hot kernels are gated `all(target_arch = "wasm32", target_feature =
"simd128")`, and `wasm32-unknown-unknown` defaults to **no** simd128 —
`build-moka-wasm.sh` says so itself, in a comment that exists because the
scalar fallback once shipped by accident and ran ~16× slower (Issue 205). So
even a hypothetical plain `--target wasm32-unknown-unknown` lane would have
compiled the SIMD half to nothing and reported green. Measured: the
simd128-**off** arm is **clean, 0 findings**; every one of the findings below
lives behind both cfgs at once.

## What was actually rotting

Cold target dir, `RUSTFLAGS='-C target-feature=+simd128' cargo clippy -p
katgpt-moka-wasm --lib --target wasm32-unknown-unknown` — **14 findings**:

| n | finding | where |
|---|---|---|
| 11 | `warning[E0133]` — `unsafe_op_in_unsafe_fn` (`v128_load`, `ptr::add`, `get_unchecked{,_mut}`) | `katgpt-moka-wasm/src/moka_int8.rs`, all inside `quantize_tensor_wasm_simd` |
| 2 | `missing_safety_doc` on `pub unsafe extern "C" fn` | `katgpt-moka-wasm/src/lib.rs` `bench_dot_f32`, `bench_dot_i8` |
| 1 | `unused_imports: f32x4` | `katgpt-types/src/simd/elementwise.rs:1444` |

plus, from the `-p katgpt-core` selection of the same crate (a different
feature set, hence a different build, hence different diagnostics):

| 1 | `missing_transmute_annotations` | `katgpt-types/src/simd/ternary.rs:460` |

The eleven are the serious ones. This crate is `edition = "2024"`, where
`unsafe_op_in_unsafe_fn` is a **warning on its way to a hard error** — an
un-gated crate whose entire purpose is to be shipped to a browser was
accumulating future-breakage with nothing to report it.

⛔ **This is not a dusty corner — it is a headline GOAT result.**
`.docs/06_game_arenas/go_arena.md` records `katgpt-moka-wasm` at **0.6 ms/move
in real Chrome, 10.7× faster than the real Moka package** (12.2× re-benched in
Node V8), and the whole 14.3× of that came from turning `+simd128` on. The
crate carrying a published cross-project benchmark win was the crate no gate
compiled. The same doc even states the mechanism —
*"`wasm32-unknown-unknown` defaults to no SIMD … the WASM build was silently
running the scalar fallback"* — which is the same double-invisibility this
layer exists to close, once as a perf bug and now as a lint one.

## The second content bug: an orphaned doc block on the public API

Widening the probe to `-p katgpt-rs --lib --target wasm32-unknown-unknown --
-D warnings` (which lints every workspace path dep, not just the three wasm32
crates) turned up three `doc_lazy_continuation` errors in
`crates/katgpt-attn-match/src/key_selection/highest_attn.rs` — and reading
them found a real defect, not a formatting nit:

```
23  /// Select top-t keys by aggregated attention score.
25  /// # Arguments
26  /// * `keys` - …                       ← nine bullets, one per parameter
35  /// * `scratch_attn` - …
36  /// Descending comparator that can never rank NaN into a top-t selection.
40  fn desc_nan_last(a: &f32, b: &f32) -> core::cmp::Ordering {   ← PRIVATE
…
53  pub fn select_highest_attn_keys(                              ← undocumented
```

The nine `# Arguments` bullets match `select_highest_attn_keys`'s nine
parameters exactly. The module's **only public export** (`pub use
highest_attn::select_highest_attn_keys`) had no documentation at all, and its
doc block — summary and full argument list — was rendering on a private
two-line comparator. Reattached; the doc block is now on the function it
describes and `desc_nan_last` keeps its own three lines.

⛔ Identical class to riir-mmorpg-examples Issue 101's third finding (a
nine-line `///` block orphaned by an insertion, rustdoc attaching it to the
wrong function, the public browser entry point left documented by one stray
sentence). Two repos, two orphaned doc blocks on public API, both found by a
wasm32 lane, neither found by any review.

**Why the wasm32 run saw it and a native run did not** is worth stating
plainly rather than guessing at: `-p katgpt-attn-match --lib -- -D warnings`
is clean on BOTH targets from a cold dir; the finding only appears when the
crate is compiled as a path dependency of the root crate. The mechanism is
unexplained. What is settled by measurement is that the defect is real and
target-independent — the doc block was on the wrong item in every
configuration; only the reporting differed.

## How it was found (a path-dependency warning, three repos down)

Not a review. riir-mmorpg-examples Issue 101 added a wasm32 lane; its
successor (Issue 102) needed one for the `wasm/` workspace; running THAT
surfaced a `dead_code` warning from riir-ai's `riir-games-mmorpg`, and fixing
that one surfaced this repo's `unused import: f32x4` — as a path-dependency
warning, which no `-D warnings` in a downstream repo can fail on.

Three repos, same class, same cause: **`build-*.sh` is not a gate.** The
chain only ran at all because somebody happened to build the leaf.

## Tasks

- [x] T0 — **all 15 findings healed** (plus, once the lane widened to the
      root package, the 3-line orphaned doc block below — 18 lint lines
      total), none by weakening a lint.
      `quantize_tensor_wasm_simd`'s body is wrapped in one explicit `unsafe`
      block carrying the caller's actual contract (`output` must be at least
      as long as `input` — every unchecked write indexes off `input.len()`
      and nothing re-checks it); the two `extern "C"` benchmarks got real
      `# Safety` sections; the unused `f32x4` import is gone; the transmute
      is annotated `::<[f32; 4], v128>`.
- [x] T1 — **`full_gate.sh` layer 2b**, modelled on layer 2 (the macOS axis)
      it sits next to: instrument check first (a zero wasm32/simd128 file
      count FAILS rather than passing vacuously), then a missing
      `wasm32-unknown-unknown` target is a **partial gate** that refuses
      unless `--allow-partial-platform`, then **both** simd128 arms.
- [x] T2 — **the package list is derived, the residue is pinned.** `-p` args
      come from `git grep -l 'target_arch = "wasm32"' -- '*.rs'` reduced to
      `crates/<name>/src/`, plus the **root package** when the root `src/`
      has a wasm32 site (it does: `src/kimi_k3/loader.rs`). A new
      wasm32-bearing crate joins the lane by existing.

      ⛔ Selecting the root package is what makes this lane WIDE, and that
      was not the reason it was added. clippy lints every **workspace path
      dependency** it pulls in — registry crates are `--cap-lints`'d,
      workspace ones are not — so `-p katgpt-rs --lib` puts the whole
      internal graph under `-D warnings` on the wasm32 target. That is how
      the orphaned doc block in `katgpt-attn-match` surfaced: a crate with no
      wasm32 code of its own, reached only as a dependency.

      Derivation cannot see a wasm32 site that no `-p … --lib` reaches, so
      the four-file residue is pinned by **membership**, not count. The gate
      reds when that set changes and says which of the two things to do
      (build it as a named target, or record the measured reason it cannot) —
      explicitly *not* "re-pin the list".
- [x] T3 — **the two wasm32 GOAT targets are in the lane; the other two
      residue files provably cannot be.** `--all-targets` was the wrong
      shape and that is measured, not assumed: `cargo clippy -p katgpt-core
      --all-targets --target wasm32-unknown-unknown` dies on **dev-deps**
      before reaching any of this repo's code —

      ```
      error[E0433]: cannot find module or crate `statrs`
      error[E0433]: cannot find module or crate `proptest`
      error: cannot find macro `proptest` in this scope
      error[E0405]: cannot find trait `Strategy` in this scope
      ```

      Named targets sidestep the dev-dep graph entirely. Both build clean at
      `-D warnings` on wasm32 with `+simd128`, so both are now lanes:

      | target | verdict |
      |---|---|
      | `-p katgpt-core --example simd_wasm32_goat` | exit 0, 0 findings |
      | `-p katgpt-core --bench bench_432_simd_lut_dequant_goat` | exit 0, 0 findings |

      That matters more than it sounds: those two targets ARE the wasm32
      GOAT evidence, so a lane that skipped them would have gated everything
      except the thing `.docs/06_game_arenas/go_arena.md` quotes.

      The remaining two — `examples/bomber_21_sonlt_arena.rs`,
      `examples/bomber_tjs_arena.rs` — cannot be built for wasm32 at all:
      they need `--features bomber`, and with it they fail at `cargo check`
      with `unresolved import sys::position` / `cannot find function
      enable_raw_mode in module sys`. crossterm has no wasm32 backend. A TUI
      arena cannot target a browser; that is a fact about the dependency, not
      a gap, and it is recorded at the pin so nobody re-derives it.
- [ ] T4 — **no CI trigger.** `full_gate.yml` runs weekly from the default
      branch on `macos-latest`; this layer inherits that cadence and nothing
      else. A wasm32 break would land on `develop` and sit for up to a week.
      Whether that is acceptable is an owner call — the cheap alternative is
      a per-push job, since the two arms are `--lib`-only and fast next to
      layer 3.

## Verification

- Cold target dir, `-D warnings`, both arms, all three derived packages: see
  the run recorded in T0/T1 (`/tmp/kg_wasm_on`, `/tmp/kg_wasm_off`).
- Native `cargo clippy -p katgpt-types -p katgpt-moka-wasm --lib -- -D warnings`
  exit 0 — every edit is inside a `target_arch = "wasm32"` cfg, so this
  proves the native surface is untouched rather than merely unbroken.
- `bash -n scripts/full_gate.sh` clean.

⛔ **Do not re-derive these counts from a warm target dir.** cargo does not
replay diagnostics for a crate it considers fresh, so a second run prints
almost nothing — full_gate.sh's own log-retention comment says this, and this
issue's first `-p katgpt-types` probe hit exactly that (silent, because the
`-p katgpt-moka-wasm` run had already built it) and would have under-counted
by one had the `-p katgpt-core` selection not rebuilt it under a different
feature set. The finding count is per (package selection, feature set, cold
dir), not a repo-wide scalar.
