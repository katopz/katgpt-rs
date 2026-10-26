# Issue 965 — a Python 3.9 syntax-floor guard for the workspace's tracked `*.py` instruments

**Status:** CLOSED — 2026-10-26, landed the same session that filed it. The gate (`python_syntax_floor_gate.py`), the sweep (`python_syntax_floor_drift_sweep.py`), and the pins (`python_syntax_floor_drift_floors.txt`) landed with the workspace GREEN at the 3.9 grammar (391 tracked `*.py`, 0 findings). Landing commit: this repo (gate + sweep + wiring, `docs:`). ⚠ The premise below was CORRECTED at landing: the class was NOT at zero workspace-wide — see "The landing census correction". katopz/katgpt-rs issue; sibling fixes ref'd inline.

## The measured incident

The workstation boxes' `python3` is **3.9.6** (Xcode CLT: `/Applications/Xcode.app/.../Python3.framework/Versions/3.9`). Python `match` statements are **3.10+ syntax** — any instrument carrying one is a dead tool on these boxes: `import` fails with `SyntaxError: invalid syntax` at the `match` line, exit before any verdict. Measured 2026-10-26, `staged_set_audit.py` (the shared-worktree pre-commit audit, AGENTS.md §"Before committing in a shared worktree") died exactly this way when invoked per its own doc (`python3 ../katgpt-rs/scripts/staged_set_audit.py .` in a sibling repo).

Census (the subprocess-encoding lesson applied: the population is tracked `*.py`, not `scripts/*.py`):

| repo | file | sites | fixed at |
|---|---|---|---|
| katgpt-rs | `scripts/staged_set_audit.py` | 1 | `95e031d95` (2026-10-26) |
| riir-train | `scripts/plan335_phase5_gate.py` | 7 | `4948aecd` (2026-10-26) |
| riir-train | `scripts/plan335_t7_decide.py` | 1 | `4948aecd` (2026-10-26) |

9 sites, 3 files, 2 repos — all rewritten branch-for-branch identical (`if`/`elif`; guards folded into conditions), each verified on the box: `ast.parse(src, feature_version=(3, 9))` passes, plus real executions (`staged_set_audit` ran end-to-end exit 0; `decide()` 5/5 verdict combinations identical; `phase5_gate` no-args → doc + exit 2). All three files already carry `from __future__ import annotations`, so annotation-position `X | Y` never was a break.

## The landing census correction (2026-10-26, the implement session)

The feature-grep census above closed the class at zero — and was WRONG about the workspace. The landing session ran the actual detector (`ast.parse(feature_version=(3, 9))`) over all 391 tracked `*.py` in the 22 contract repos on this box and found **one live site the grep census had missed**: `reflex-site/scripts/render_flows.py` — a backslash inside an f-string EXPRESSION (`f"...{re.match(r'<(\w+)', t)...}"), PEP 701, legal only on 3.12+, `SyntaxError` on 3.9–3.11. Exactly the class this issue guards, alive in a repo whose scripts the closing session never greped for it. Fixed reflex-site `a24f4dd` the same day (hoist the regex out of the expression). Lesson recorded twice over: a census greps for KNOWN spellings, a grammar parse reads ALL of them — the detector is the floor, not the list.

Final measured state at landing (22 of 31 repos on amo, `DOCS_GATE_PARTIAL_CLONE=1` posture): 391 files, 391 parsed, 0 findings; floors pinned per repo in `python_syntax_floor_drift_floors.txt`; katgpt-rs floors (60/60) owned by the gate and asserted by the sweep.

## What lands (the guard)

Two halves, the family shape:

1. **Per-push gate half** — a `scripts/python_syntax_floor_gate.py` CHECK in `docs_gate.sh`: `ast.parse(source, feature_version=(3, 9))` over every tracked `*.py` in THIS repo (population via `tracked_walk`, the Issue-777 rule — a filesystem walk miscounted a nested repo once already). A `SyntaxError` at the 3.9 grammar is a finding; ceiling 0. Requirements per the standing gates: an unconditionally-invoked self-test arm (`check_validation_gate` — the flag-gated-arm class measured there: eight adversary arms landed flag-gated and ran on no push at all), an AGENTS.md CHECKS-table row (`docs_gate_checks_sync` — membership both ways + quantity words), and a floor on the population (a walk regression must red, not print a green zero — the `cfg_gated_floor_gate` lesson).
2. **Cross-repo drift-sweep half** — `python_syntax_floor_drift_sweep.py`: the same parse over every contract repo's tracked `*.py`, population from the contract walk (`derive_repos` — never a hand-typed set; the Issue-837 registration lesson), per-repo `min_py_files` floors (the subprocess-encoding split: `min_calls` was 0 in 10 of 16 repos, so the file-count floor was the only blindness detector — same shape here), pinned in `scripts/python_syntax_floor_drift_floors.txt`, and the family-wide mechanisms (`sweep_advisory()` — worktree advisory + known-extra + head-provenance + alias-open — per `sweep_advisory_membership_gate`, which reds on a sweep that skips them). Findings need no pin file: the ceiling is 0 everywhere and a `SyntaxError` is never legitimate (unlike a lint count, there is no DELIBERATE class — a file that cannot parse cannot run).

Non-goals: no runtime-version pin enforcement beyond the 3.9 grammar (`feature_version` rejects NEWER syntax; it does not pin the interpreter), no wasm/CI-runner axis (the instruments run on workstations), and no retro-format sweep of vendored trees (`.raw`/`vendor`/`node_modules`/`target` excluded, the standard walk rules).

## Evidence the detector works (measured)

`python3 -c "ast.parse(open('scripts/staged_set_audit.py').read(), feature_version=(3, 9))"` on the PRE-fix file raised `SyntaxError: invalid syntax` at line 472 (the `match`); on the post-fix file it passes. On a 3.10+ interpreter the same parse accepts both — the check is meaningful exactly where it matters (the 3.9 boxes) and inert where the syntax is legal, which is the correct failure direction for a workstation instrument (never blocks a newer box, always catches an older-box break).

## Cites

- The fix commits: katgpt-rs `95e031d95`, riir-train `4948aecd` (same session, 2026-10-26).
- Class precedent: Issue 778 (`text=True` locale decode — "nothing that could notice this ever runs it"), Issue 792 (Windows timing — an instrument blind on a box class), Issue 804 (console encoding — non-UTF-8 console deaths).
- The walk: `scripts/tracked_walk.py` (Issue 777, one copy, 8-arm self-test).
- AGENTS.md §"Before committing in a shared worktree" — `staged_set_audit.py`'s documented invocation, the one that died.
