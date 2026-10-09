# Issue 928 — cross-repo drift-sweep debt: sibling landings that did not re-pin (the 2026-10-09 seal-std registration census)

**Status:** OPEN — filed by the seal-std registration unit (katgpt-rs `c007cf543`); every row below MEASURED by running its owning sweep on this box (M3, 2026-10-09, post-`c007cf543`)

The seal-std registration ran all ~20 cross-repo drift sweeps to measure the
new repo's rows. The registration itself is green — but the runs surfaced
pre-existing committed drift in SIX sibling repos: landings that advanced
their content past this box's floor pins without a same-commit re-pin. These
sweeps are workstation-only (not in the docs-gate CHECKS), so the drift was
invisible until a full-family run.

Two classes — read before "fixing" any row:

- **PIN-STALE** (the sibling commit landed; the pin did not move): repair =
  re-pin to the measured value in the re-pin commit, citing the sibling
  commit that caused it.
- **CODE-DEBT** (the sweep names a repair tool and forbids raising the
  ceiling): repair = fix the SIBLING's code, never the pin.

## The census (sweep → repo → finding → class)

| sweep | repo | measured vs pinned | class |
|---|---|---|---|
| `cfg_gated` | riir-ai | load-bearing SILENT-NOW **3 committed > pinned 0** — targets whose names say their green is evidence report `ok. 0 passed` over empty binaries | **CODE-DEBT** (read the 3 targets: either they are genuinely load-bearing and their tests vanished, or the names lie — then re-pin with reasons) |
| `cfg_gated` | riir-dapps | (same run's ✗ row — take the numbers from a fresh run) | read-then-classify |
| `citation` | mmorpg-editor | CROSS **1 > pinned 0** + IN-LOCAL-RANGE **1 > pinned 0** (docs=2 cites=115) | **CODE-DEBT** (a cross-repo citation naming no repo — Issue 749's exact class; find and qualify the citation) |
| `console_encoding` | mmorpg-editor | undefended **1 > pinned 0** | CODE-DEBT (defend the stream or make it ASCII) |
| `console_encoding` | reflex-site | undefended **2 > pinned** (walk=10 pop=9 defended=7) | CODE-DEBT |
| `instrument_reachability` | mmorpg-remake | unreachable **8 > pinned 7** | **PIN-STALE-or-DEBT** (a new script no root names — wire it or pin with a reason) |
| `instrument_reachability` | reflex-site | unreachable **4 > pinned ?** | same |
| `locale_io` | riir-ai | LOCALE-IO **6 committed > pinned 0** | **CODE-DEBT** — the sweep names `scripts/locale_io_fix.py`, ceiling may NOT rise |
| `locale_io` | riir-deployer | LOCALE-IO **4 > pinned ?** | same |

len_derived also failed (`see ✗ rows above` — rows not captured in the
census pass; re-run `scripts/len_derived_drift_sweep.py` and take the ✗
rows before acting).

## Why this is a unit, not a hotfix

- Two repos (riir-ai, reflex-site) had a dirty worktree file at census time
  (sibling sessions live) — **Issue 797 applies**: counts from a dirty-tree
  run describe a state no commit contains. Re-measure on a clean/committed
  state before re-pinning anything; the CODE-DEBT rows are safe to READ now
  (the finding is in the sibling's committed lines, not its WIP).
- The locale_io rows are sibling CODE repairs (riir-ai `Path.read_text`
  sites etc.) — landing them means touching sibling repos; coordinate with
  their owning sessions or wait for quiet.

## The fix order

1. Re-run each sweep fresh; capture per-repo ✗ rows with numbers.
2. CODE-DEBT rows: repair the sibling code (locale_io via
   `scripts/locale_io_fix.py`; console_encoding via `scripts/console_safe.py`;
   citation via qualifying the citation in the sibling doc).
3. PIN-STALE rows: re-pin DELIBERATELY citing the sibling commit.
4. Full-family sweep run green; append the doc-sync run-log row.

## Related

- katgpt-rs `c007cf543` (the registration that surfaced this)
- Issue 797 (the worktree-advisory law), Issue 749 (citation qualification),
  Issues 829/830 (locale_io), Issue 804 (console_encoding)
