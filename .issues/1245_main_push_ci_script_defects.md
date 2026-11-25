# Issue 1245: main-push CI script defects — rft-gate tomllib import + wasm32 highs-sys cmake (both runner-env classes, found by the 2026-11-25 main ff)

**Status:** OPEN — T1 claimed (tomllib family; the fix is the shared-shim + workflow-interpreter-pin pair, see the task rows below).
**Claimed:** amo-i1245-t1, 1795650240, expires 2026-11-27T09:30:00+00:00
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

- [x] T1 — rft gate python floor — **LANDED 2026-11-25 (this issue's T1 lease)**: the shared-shim + workflow-interpreter-pin pair, a family repair over **11 direct `import tomllib` sites** (the first grep census was truncated by `head -40`; the untruncated census found three more through import chains: `cfg_gated_target_audit` — itself imported by both rft instruments — `feature_cap_gate`, `all_ignored_target_audit`):
  - `scripts/toml_compat.py` (NEW): stdlib `tomllib` → `tomli` backport → loud `ModuleNotFoundError` naming both remedies (uv interpreter — docs_gate.sh's own `uv python find '>=3.11'` probe — or `pip install tomli`). `ModuleNotFoundError` deliberately (an `ImportError` subclass) so existing guards keep working: `docs_drift_sweep`'s lazy-auditor `except ModuleNotFoundError` and `suite_membership_audit`'s stub install (UNCHANGED — its stub satisfies the shim's stdlib import for its regex-based scan; posture preserved).
  - 11 one-line swaps: `import tomllib` → `from toml_compat import tomllib` (required_features_drift_sweep, count_features, wasm32_surface_audit, bench_doc_audit, feature_isolation_gate, cargo_comment_audit, cfg_row_implication_audit, required_features_build_audit, cfg_gated_target_audit, feature_cap_gate, all_ignored_target_audit).
  - 4 workflow pins: `actions/setup-python@v6` `python-version: '3.11'` ahead of the bare-`python3` invocations — required_features_touched.yml, sibling_required_features_touched.yml (its `runs_on` is caller-supplied, so the pin rides inside), feature_isolation.yml, feature_isolation_weekly.yml. Chose setup-python over the issue's `uv run --python 3.11` CI option because runners don't carry uv — setup-python is the platform-standard pin of the same floor (the uv option remains the DEV-box remedy, named in the shim).
  - Box posture: `tomli 2.5.0` installed `--user` on amo (the rft gate now runs end-to-end on the bare 3.9.6 interpreter — previously a ModuleNotFoundError traceback; after: `0 selected, exit 0`).
  - Verified: 3.9 syntax floor gate PASSED (115 files, 0/0); 12-module import battery green on BOTH the bare 3.9.6+tomli and uv 3.12 stdlib postures; rft gate `--base HEAD~1 --head HEAD` → 0 rows exit 0 on-box; docs_gate.sh full local run green (executes count_features + bench_doc_audit + cargo_comment_audit live through the shim).
  - NOT touched (documented postures): suite_membership_audit's stub, docs_drift_sweep's lazy import + guard comment, cfg_gated_target_audit's absence from suite_membership's comment (stale pointer — the stub guards the whole family transitively either way).
  - Sibling awareness (their repos' business, noted not fixed): `../riir-ai/scripts/` and `../riir-chain/scripts/` carry 1 top-level tomllib importer each.
- [ ] T2 — wasm32 lane highs-sys: pin the exclusion or install a wasm32 cmake toolchain in the workflow (check what `scripts/full_gate.sh --wasm32-only` does about highs-sys locally — on amo the local run PASSES, so the difference is the runner's cmake, not the script; likely the runner needs `wasm32-unknown-unknown`-capable cmake OR the lane needs to skip the `good_lp`-dependent crates with a membership-pinned row like the rest of the gate's residue pins).
- [ ] T3 — after T1+T2 land on develop: re-ff main (or ride the next deliberate gate merge) so main's CI records a green; the red run's receipts stay as the false-GREEN-rate evidence.

## Notes

- The full-gate red at `bf479a709` is the one class this issue does NOT carry (fixed `d4790394e` on develop; see that commit body for the per-site list).
- Local evidence commands recorded in the Plan 635 line-87 row + the d4790394e commit body.
