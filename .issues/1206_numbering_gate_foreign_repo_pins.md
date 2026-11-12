# Issue 1206: numbering_gate.py on a foreign repo applies katgpt-rs-only pins — structurally exit 1

**Status:** In progress — repair design adjudicated (verdict round pending record); filed 2026-11-12 by the plan-373 overflow-continuation housekeeping unit (amo). Evidence reproduced same-day, both runs captured; no instrument change landed yet.
**Claimed:** zed-plan373-cont@amo, 1794574500, expires 2026-11-14T18:00:00+00:00

## Premise

`scripts/numbering_gate.py` accepts any repo as `argv[1]`, and the doc-sync skill instructs lanes generically: "run that repo's numbering gate before leaving — `py ../katgpt-rs/scripts/numbering_gate.py <repo>`". But two halves of the instrument are pinned for katgpt-rs and only katgpt-rs, so on a smaller repo the gate **cannot exit 0**, and its remediation text points at pins the foreign lane must not touch:

1. **Floors half is self-declared katgpt-rs-scope-only** — `numbering_floors.txt` header: "Pins for scripts/numbering_gate.py — katgpt-rs scope only (Issue 724 T4)" — yet `main()` applies its `min_files.*` population floors to ANY target. Measured on riir-train 2026-11-12: `.benchmarks 330 < 400`, `.plans 282 < 400`, `.proposals 1 < 10`, `.research 173 < 200` → "population FLOOR breached", exit 1. riir-train has held exactly ONE proposal file in its history — no commit can ever green that class there.
2. **Collision-half scalars are shared-file-only** — `collision_pins_for()` (the `d6f996199` per-repo side-file mechanism) reads only `max_hand_read` / `max_rename_residue` overrides from `number_collisions_<repo>.txt`; `min_numbers` (the 1200 history-walk floor) and `min_dirs` come from the shared scalars and are NOT overridable per-repo. riir-train's history walk = 800 numbers → "walk FLOOR: 800 < 1200", exit 1. A riir-train side file (which does not exist — only riir-ai / riir-game-sdk / riir-infer have one) would NOT fix this class.
3. **Unactionable remediation text** — the notes ("legacy collisions 9 < ratchet 64 — re-pin DOWN in this commit", "hand-read groups 10 < ratchet 59") advise re-pinning pins that describe katgpt-rs, from a repo that does not own them.

## Measured clean where it matters

- riir-train's OWN instrument (`riir-train/scripts/numbering_duplicate_gate.py`) PASSES same-day: 0 failures, 4 safe groups (family-explicit + suffix-family), 12 pinned finding groups, population 853.
- The gate's repo-meaningful classes on riir-train are clean: 0 tracked duplicates, 0 above-highwater, 0 malformed highwaters (no ✗ rows of those classes printed).
- riir-ai exits 0 on the identical invocation (side file present + floors genuinely met) — the mechanism works where populations meet katgpt-rs's floors; the gap is the smaller-repo class (riir-train today, and by construction any smaller BOUNDARY repo a lane points the gate at).

## Tasks

- [ ] Adjudicate the repair shape: (A) per-repo floors side files (`numbering_floors_<repo>.txt`) mirroring the d6f996199 collision pattern, with scalar overrides extended to `min_numbers`/`min_dirs` — honest but ~33 files of data authoring; (B) for `target != REPO_NAME` with no side file, SKIP the floors half with a LOUD named deferral line (never a silent green — the skip-loud convention) and run the collision half on per-repo scalars only; (C) doc-only: narrow the doc-sync skill wording to each repo's own sanctioned instrument (riir-train: `scripts/numbering_duplicate_gate.py`, guard last layer).
- [ ] Land the chosen repair with a fixture proving the foreign-target verdict shape (exit 0, or a LOUD deferral — never the current unactionable exit 1).
- [ ] Re-run the doc-sync-skill cross-repo invocation on a small repo (riir-train class) and confirm the green-or-loud-deferral verdict.
- [ ] On adoption of (B) or (A): reconcile `numbering_drift_sweep.py`'s cross-repo half, which shares the assembly seam (`collision_pins_for` is documented as the ONE home both call).

## Evidence

- 2026-11-12, amo box: `py ../katgpt-rs/scripts/numbering_gate.py .` from riir-train → exit 1 (population-floor block + walk-floor cfail + two ratchet ⚠ notes quoted above); the identical invocation from riir-ai → exit 0; riir-train `py scripts/numbering_duplicate_gate.py` → exit 0 PASS.
- Pre-flight: `numbering_floors.txt` at committed state (mtime Nov 8), `bash -n`-clean script (Python, no shell), no mid-edit; `dual_allocation_gate` 0/0/0 before this allocation.
