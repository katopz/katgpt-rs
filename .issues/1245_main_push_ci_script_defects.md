# Issue 1245: main-push CI script defects — rft-gate tomllib import + wasm32 highs-sys cmake (both runner-env classes, found by the 2026-11-25 main ff)

**Status:** OPEN — unclaimed (no lease line yet; the claiming lane appends `**Claimed:** <session>, <epoch>, expires <ISO-UTC>` under this Status line and pushes before working).
**Date:** 2026-11-25
**Found by:** amo-idle-i635-gatemerge, the Plan 635 git-main gate merge (`bf479a709` — katgpt-rs develop→main ff, first main CI run in 1904 commits since 2026-10-07 `b444b3e57`)

## The two defects

| run (bf479a709) | verdict | evidence | class |
|---|---|---|---|
| wasm32 gate | FAILURE (54 s) | died compiling `highs-sys v1.14.2`'s cmake child for `wasm32-unknown-unknown` on `ubuntu-latest` ("System is unknown to cmake" ×3 → configure incomplete) | runner-env/toolchain gap: no wasm32-capable cmake toolchain on the runner; highs-sys rides `good_lp` (optional, `convex_tok` / `percepta_compile` gated). **LOCAL PASSES**: `scripts/full_gate.sh --wasm32-only` green both simd128 arms on amo (aarch64 macOS) — not a repo regression |
| required-features touched gate | FAILURE | LOCAL: `uv run --python 3.11 python3 scripts/required_features_touched_gate.py` → **0 rows selected, exit 0** (the push touched no row's src). On-box python3 is 3.9.6 and the gate CRASHES at `import tomllib` (line 63 of `required_features_build_audit.py`) — 3.11+ only. The workflow runs `macos-latest` with bare `python3`; the 2026-10-07 run passed, so the runner image's python3 moved (macos-26-era image) below 3.11 | runner-image python drift vs a 3.11-only import; needs a `tomli` fallback or `uv run --python 3.11` in the workflow step |
| Full gate | FAILURE | REAL repo content — **FIXED on develop `d4790394e`** (the 12 `-D` lint-floor residuals since 10-07: bench_986 ×4, bench_997 ×3, sparse_kl, reward_census ×2+cast, codebook ×2, pruners fpcg seed spelling). Verified: clippy `-D` 0 errors `--all-targets --all-features` both crates, lib 2580/0 + 126/0 (+fpcg 10/0), bench_997 release 1 passed, wasm32 both arms green | fixed; main picks this up at the next deliberate main ff |
| Docs gate / Lean proofs | success | — | — |

## Tasks

- [ ] T1 — rft gate python floor: either vendor a `tomli` fallback (`try: import tomllib except ModuleNotFoundError: import tomli as tomllib`) across the scripts/*py gate family that imports tomllib, or switch the cargo-invoking workflows to `uv run --python 3.11`. Grep first — this is a FAMILY (the 3.9 syntax-floor gate exists precisely because boxes carry 3.9.6; the CI macOS runner now agrees with them).
- [ ] T2 — wasm32 lane highs-sys: pin the exclusion or install a wasm32 cmake toolchain in the workflow (check what `scripts/full_gate.sh --wasm32-only` does about highs-sys locally — on amo the local run PASSES, so the difference is the runner's cmake, not the script; likely the runner needs `wasm32-unknown-unknown`-capable cmake OR the lane needs to skip the `good_lp`-dependent crates with a membership-pinned row like the rest of the gate's residue pins).
- [ ] T3 — after T1+T2 land on develop: re-ff main (or ride the next deliberate gate merge) so main's CI records a green; the red run's receipts stay as the false-GREEN-rate evidence.

## Notes

- The full-gate red at `bf479a709` is the one class this issue does NOT carry (fixed `d4790394e` on develop; see that commit body for the per-site list).
- Local evidence commands recorded in the Plan 635 line-87 row + the d4790394e commit body.
