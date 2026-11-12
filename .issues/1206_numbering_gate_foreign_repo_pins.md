# Issue 1206: numbering_gate.py on a foreign repo applies katgpt-rs-only pins — structurally exit 1

**Status:** Closed — LANDED 2026-11-12 (verdict ping-pong 3 rounds: REVISE ×2 → AGREE; the row-armed foreign-target design). The foreign floors live in `scripts/numbering_gate.py` (`foreign_pins_for` + the main() foreign posture): a foreign target is ARMED from its own `numbering_drift_floors.txt` row (min_files over the serial scan-n sum, min_bench over tracked .benchmarks, min_numbers over the history walk — the sweep's exact populations), row floors HARD, katgpt-rs per-dir floors demoted to named advisory lines, empty walk still refuses exit 2; the row parser moved into the gate (`parse_drift_rows`/`DRIFT_PIN_FIELDS`) and the sweep re-binds it (`ng.parse_drift_rows`) — one parser, one-direction imports; the row lookup goes through `repo_alias` at module scope (contract names, alias-codec safe); the hand/rename caps are ONE record in two media — the row is authoritative, a `number_collisions_<repo>.txt` scalar must EQUAL the row (the gate raises naming both sides; measured agree: riir-ai 30=30, riir-infer 7=7, riir-game-sdk 0=0), row value is the fallback; allocator-integrity checks + the COLLISION wall + `collision_verdict` stay hard everywhere; self-target unchanged. Selftest: `foreign_arms()` (parser incl. loud malformed-row refusal, contract-name identity, row resolution, missing-floors-file raise) + the sweep's cap-sync arms. Measured: katgpt-rs self PASS 37/37 docs-gate checks (cross-module-attr + import-health green), riir-train/riir-ai/riir-reflex/riir-infer all exit 0 ARMED. AGENTS.md + docs_gate.sh CHECKS row extended in-commit.
**Lease closed:** zed-plan373-cont@amo, epoch 1794574500, claimed 2026-11-12 ~18:1xZ, was expires 2026-11-14T18:00:00+00:00 — landed same session, see the landing commit (claim record: zed-plan373-cont@amo, epoch 1794574500, was expires 2026-11-14T18:00:00+00:00).

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

- [x] Adjudicate the repair shape — verdict ping-pong 3 rounds (REVISE ×2 → AGREE, 2026-11-12): row-armed foreign targets from the EXISTING `numbering_drift_floors.txt` rows (no new file family — the reviewer's substrate-first correction of this issue's original A/B/C list), caps one-record-two-media, parser into the lower layer, contract-name lookup via `repo_alias`.
- [x] Land the repair with selftest arms proving the foreign-target verdict shapes (`foreign_arms()` + the sweep's cap-sync arms).
- [x] Re-run the doc-sync-skill cross-repo invocation on small repos — riir-train/riir-reflex/riir-infer all exit 0 (ARMED or row-armed-hard paths verified live).
- [x] Reconcile the sweep: `parse_rows` re-bound from `ng` (one parser); the cap equality is asserted sweep-side too. The sweep's own floors rows were ALREADY per-repo — no sweep data change needed.

## Evidence

- 2026-11-12, amo box: `py ../katgpt-rs/scripts/numbering_gate.py .` from riir-train → exit 1 (population-floor block + walk-floor cfail + two ratchet ⚠ notes quoted above); the identical invocation from riir-ai → exit 0; riir-train `py scripts/numbering_duplicate_gate.py` → exit 0 PASS.
- Pre-flight: `numbering_floors.txt` at committed state (mtime Nov 8), `bash -n`-clean script (Python, no shell), no mid-edit; `dual_allocation_gate` 0/0/0 before this allocation.
