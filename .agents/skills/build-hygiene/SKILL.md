---
name: build-hygiene
description: Generic Rust build-hygiene gate for compile time, binary size and the dev-edit loop. Use when adding deps or features or crates, touching cargo profiles, when a build/link/boot loop feels slow, when target dirs bloat, when choosing a dev-loop driver (plain cargo vs framework dynamic-linking vs blaze), before quoting any build-time or size or perf number, or quarterly as a hygiene pass. Enforces the profile discipline (dev line-tables-only + opt-3 deps, a shared opt-0 gate tree, dist fat-LTO strip), the determinism law against profile pins, isolated target dirs, the pre-registered A/B bar, and lint-healing before any perf claim.
---

# build-hygiene

Compile time, binary size, and the dev-edit loop are MEASURED postures, not vibes.
This skill is the workspace-agnostic discipline; where a repo documents its own
`[profile]` rows in `Cargo.toml` comments, those comments are the canonical text
and this skill defers to them.

## 1. The profile discipline

The canonical five-profile shape for a workspace that runs many build postures
(heavy test/guard matrices) and ships a real-time binary:

| Profile | Purpose | Key rows |
|---|---|---|
| `[profile.dev.package."*"]` | deps fast, own code debuggable | `opt-level = 3`, `debug = false` — `*` never touches workspace members |
| `[profile.dev]` | the daily build | `debug = "line-tables-only"` — the EXPLICIT string spelling, never `debug = 1`; backtraces keep file/line, debugger locals are the deliberate loss |
| `[profile.dev-full]` | the debugger escape hatch | `inherits = "dev"`, `debug = true` |
| `[profile.gate]` | multi-posture CI/guard runs | `inherits = "dev"`, `incremental = false`; deps at `opt-level = 0` in ONE shared cache instead of dozens of opt-3 dep copies |
| `[profile.dist]` | shipped binaries | `inherits = "release"`, `opt-level = 3` (never `"s"` on a real-time renderer), `lto = "fat"`, `codegen-units = 1`, `strip = true` |

Measured shape of the win (one Apple-M3-Max game workspace, 2026-10): deps-at-opt-3
dev made the mid-loop edit→linked-binary ~4.9 s against ~22.7 s in release; moving a
~40-posture guard matrix onto one shared opt-0 gate tree bounded a target dir that had
grown past 133 GB; the workspace `line-tables-only` change is what made every
per-posture build affordable.

Rules:

- **Plain `--release` stays untouched** — benches and dev runs compare against it;
  a release-profile tweak is a measurement-lane change with its own bench.
- **`incremental = false` for single-pass postures** — a posture that compiles
  exactly once per run gains nothing from incremental caches, and they cost tens
  of GB of `.rcgu.o`.
- **THE DETERMINISM LAW (hard rule).** A golden/fixture that forks across opt
  levels is a determinism defect in the code — e.g. libm `sin`+`cos` merging into
  a platform `sincos` at opt ≥ 1 can move results by 1 ULP — fix the math with a
  deterministic implementation (a pure-Rust trig, an integer path, a pinned
  reduction order). NEVER resurrect `[profile.dev.package.*]` opt-0 pins to mask
  the fork; a pin taxes every future dev loop and hides the defect.
- **The profile is part of every claim.** A timing or size number names its
  profile and feature set; `debug_assertions` is ON in dev — a dev-profile
  latency number is not a release claim.

## 2. The dev-loop ladder (pick by need, measure before adopting)

1. **Plain cargo dev** — the baseline. Playable/fast because deps ride opt-3
   while your crates stay debuggable.
2. **Dynamic-link the heavy framework deps** — e.g. `bevy/dynamic_linking` behind
   an opt-in dev-only feature. Expect roughly a 20%+ cut in edit-to-frame, biggest
   on cross-crate edits (the worst case is exactly where linking dominates).
   Conditions, all of them: DEV-LOOP ONLY · never in a default feature set ·
   never in a receipt or shipped/release lane · on macOS a dylib-linked binary
   needs `DYLD_FALLBACK_LIBRARY_PATH=<target>/debug/deps:$(rustc --print
   target-libdir)` passed as a VAR PREFIX (macOS strips `DYLD_*` through the
   SIP-protected `/usr/bin/env` — measured; the `env VAR=… cmd` route refuses).
3. **[blaze](https://github.com/indexable-inc/blaze)** — an external incremental
   daemon (rust-analyzer/MIR + Cranelift; patches changed functions into the
   already-built binary; claims ~10× warm-edit builds with honest per-project
   tables). Facts that gate adoption: it builds with its OWN toolchain (a dated
   nightly, whatever your repo pins), trades GB-scale memory, and is young —
   cargo + CI stay the final word. Adopt only through a measured pilot on your
   own loop, never on published numbers alone.
4. **Measure the LOOP, not the build half.** The metric is edit→first-frame
   (GUI) or edit→test-green (headless): stimulus apply → incremental build →
   observed result. Interleaved A→B pairs, per-arm isolated `CARGO_TARGET_DIR`,
   per-iteration logs kept. Every timed row verifies the BINARY SHA CHANGED —
   a comment/mtime touch does not relink under incremental; that row is
   INVALID, not slow.

## 3. Compile-time technique table

- **Target-dir bloat is a POSTURE-COUNT problem.** Dozens of postures × opt-3
  dep copies = 100+ GB. Fix: the `gate` profile (one shared opt-0 dep tree) +
  `incremental = false`. Reclamation is a separate approved step — never delete
  `target/release` anywhere, never delete a tool binary's release dir that
  symlinks or scripts point at.
- **Sibling build holding the lock:** `CARGO_TARGET_DIR=/tmp/<task>` — never wait
  on a lock, never share a target tree across concurrent sessions; move scratch
  targets to a roomy volume when the internal disk runs low.
- **The green-zero trap.** A whole-file `#![cfg(feature = "x")]`-gated test target
  run without the feature prints `ok. 0 passed`, exit 0 — byte-identical to a
  pass. `required-features` rows protect the reader. After any clean clippy sweep,
  check the blind spot: which gated targets did the sweep never compile? (the
  `cargo refine --oracle-census` report does this where the healer is installed;
  otherwise grep your own `[[test]]`/`[[bench]]` rows).
- **A wrong `required-features` row is worse than a missing one** — `cargo test
  --workspace` silently SKIPS the target; only `--all-features` builds it. Audit
  suspect rows by building them at their own feature set.
- **`cargo check` is not a compile-time probe** — time the thing you ship: the
  repo's quick gate layers, or a named `cargo build --profile <p> -p <pkg>` with
  the box quiet.
- **wasm/embedded lanes are optional lanes** — keep their deps out of the native
  default set; a union feature compiles the browser/cross lane; a manifest drift
  gate (`bundle vs manifest`) catches the stale-artifact class.
- **Workspace-excluded crates need mirrored profiles.** A crate built by
  `--manifest-path` outside the workspace does not inherit your `[profile]`
  rows — mirror the ones it needs, or its builds quietly regress to defaults.

## 4. Binary size

- Shipped = the `dist` profile; verify size claims against `target/dist/<bin>`,
  never `target/release/`.
- `strip = true` in the profile beats post-hoc `strip` invocations — declared,
  reproducible, cannot be forgotten.
- For leak-sensitive shipped binaries, build with
  `RUSTFLAGS="--remap-path-prefix $HOME=/build"` (or a fixed prefix) and run a
  binary string-leak scan — the home path, usernames and machine names must not
  ship.
- `opt-level = "s"`/`"z"` are REFUSED on real-time renderers/servers without a
  measured frame-time A/B proving no regression — the frame budget wins over bytes.
- Content/artifact size (compressed bundles, dataset archives) is a separate
  lane — compression at the artifact boundary, not profile flags.

## 5. The perf law (heal before you measure)

1. **Heal mechanical lints FIRST** — `cargo refine --fix` where the workspace
   healer is installed (compile-gated writes: `--fix --write --verify`, and
   `--verify-args "--features <set>"` for gated code), else `cargo clippy --fix`.
   A perf number taken over lint noise is not comparable to a clean baseline.
2. **Pre-register the bar BEFORE data** — write the want-faster threshold and
   the decision rule into the plan/bench doc first; the verdict is mechanical
   afterwards, not narrative.
3. **Interleaved pairs, not sequential arms** — two sequential runs measure the
   box's load drift (±5–22% on a shared box); interleave A→B pairs and report
   p50 + total + slowest.
4. **Box state is part of the number** — quote a provenance line (power source,
   power mode, load, swap; GPU exclusivity for GPU gates) beside every latency
   figure. Say "measured faster", never a guaranteed number; a marginal margin
   is disclosed as marginal and re-read before anyone relies on it.
5. **A latency number without its box state is not a measurement** — a probe run
   on a loaded box is evidence of nothing; re-run alone or discard.

## 6. The housekeeping pass

Quick pass (every housekeeping cycle, report-only):

```sh
du -sh target 2>/dev/null                     # per repo; report, do NOT delete
ls -la target/release/ target/dist/ 2>/dev/null | head -20   # binary sizes
grep -c "^\[profile" Cargo.toml               # vs the repo's own documented set — drift is a finding
time <the repo's quick gate>                  # e.g. ci_feature_guard.sh --quick, or a named build
cargo refine <paths>                          # DRY RUN on dirty files (or: cargo clippy for the baseline)
```

Findings are filed, never fixed inline: a bloated target → an approved cleanup
pass; a missing/drifted profile → an issue citing this skill; a binary growth
> 10% without a landed reason → an issue; a quick-gate time above its last
recorded baseline → bisect before touching profiles.

Full pass (quarterly, or after any profile/feature-set/crate-count change):

- re-run the dev-loop A/B harness (build one before adopting any driver) and
  compare against the last recorded baseline;
- re-time the quick gate layers and compare against the last recorded posture;
- re-check the profiles against §1 (the repo's `Cargo.toml` comments are the
  canonical text — prose drift is a finding);
- verify excluded/leaf workspaces carry mirrored profiles (§3);
- verify no `DYLD`/dynamic-linking posture leaked into a default set, receipt,
  or shipped lane (§2's conditions).
