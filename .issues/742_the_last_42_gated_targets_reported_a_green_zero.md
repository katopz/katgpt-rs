# Issue 742 — the last 42 `#![cfg]`-gated targets in this repo reported a green zero; `SILENT-NOW` is now a WALL at 0

**Status:** RESOLVED 2026-09-09. `SILENT-NOW 42 → 0`, `max_silent_now` re-pinned
**42 → 0 as a WALL**. Rows DERIVED from the instrument, not hand-typed.

## The finding

42 targets opened a whole-file `#![cfg(feature = "…")]` with **no `[[test]]`
row** — auto-discovered. Such a file compiles to an **empty binary** when its
features are off, and cargo prints `test result: ok. 0 passed` and **exits 0**
— byte-for-byte a real pass. The `#![cfg]` protects the **count**;
`required-features` protects the **reader**, and only the second is visible in
the output. What changes: naming one *without* its features now **errors**.

This was the repo's own long-standing backlog, and `cfg_gated_floors.txt`
described it exactly right: *"the remaining 42 are a real backlog, not a claim
of zero."* Distribution: **32** root `tests/`, **6** katgpt-core, **2**
katgpt-types, **1** katgpt-tokenizer, **1** katgpt-speculative. All
`kind = test`, all with expressible features, **0 load-bearing**, and none
carrying a platform predicate — so every one is fully expressible.

## Rows derived, not typed

A hand-typed row is precisely the drift this class is about, so the rows were
generated from `scripts/cfg_gated_target_audit.py --json`. That required
adding **`silent_now_rows`** to the JSON: the report had always *printed* the
exact row to add per finding, but the machine-readable mode carried only totals
plus three unrelated path lists — so it could not drive the one action the
report exists to prompt, and every repo's fix so far had been driven by
re-parsing prose. Now `{path, kind, name, features, predicates, load_bearing}`
per finding, ordered by path so a diff of two runs is readable. Verified the
list length equals the `silent_now` count (42 = 42).

## Verification

**Statically, by two gates that already existed for exactly this** — and this
is a stronger check than the artifact line (see §The protocol correction):

| gate | before | after |
|---|---|---|
| `required_features_static_gate.py` — a row naming a feature its package cannot enable | 629 rows, 0 | **671 rows, 0** |
| `cfg_row_implication_gate.py` — a row that BUILDS and compiles its target to **NOTHING** | 629 rows, 0 empty-at-row | **671 rows, 447 with a leading `#![cfg]`, 0 empty-at-row** |

The second is the one that matters: `max_empty_at_row` is a WALL at 0 and it
**held** across all 42 new rows. Docs gate **14/14** clean.

**Population unchanged while the findings fell** — a falling count is otherwise
ambiguous between "repaired" and "the instrument went blind":

| | targets | `#![cfg]` | w/ req-f | SILENT-NOW | load-bear | latent | plat-exc | cfg(test) | any() | PROFILE* |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 931 | 550 | 405 | **42** | 0 | 101 | 0 | 1 | 1 | 2 |
| after | 931 | 550 | **447** | **0** | 0 | 101 | 0 | 1 | 1 | 2 |

`w/ req-f` 405 → 447 = 405 + 42 **exactly**; every other column unmoved, so
nothing was silently reclassified into a non-defect bucket.

## The protocol correction — an `Executable` line is NOT proof

Landing the sibling repos, this session used "cargo named an `Executable`
artifact" as the sufficiency proof. **That is necessary and not sufficient**,
and riir-chain's run is what proved it: a root build reported `Finished in
0.33s` immediately after another in the same feature set took `26.86s` — the
signature of an empty binary. It resolved by running each binary with
`--list`: all 28 reported 1–15 tests, so the fast builds were genuine cache
hits.

The hazard is precisely **row features ⊆ file `#![cfg]` features**: cargo
builds the target (so the row looks met), the body compiles to nothing, and an
`Executable` line is printed either way. **For this class the test count is the
proof; the artifact line is not.**

Closed workspace-wide rather than repo-locally, because the same weak evidence
was used in three repos: `scripts/cfg_row_implication_drift_sweep.py` measures
exactly this property statically, and on 2026-09-09 reported

```
17 repos · 2100 rows · 1437 with a leading #![cfg] · 0 EMPTY-AT-ROW · 9 UNRESOLVED
```

**0 EMPTY-AT-ROW** — so no row landed that day, in any of the six repos, is an
empty binary. That is what licenses the ~200 rows, and it is a stronger claim
than the per-repo `--list` spot-checks would have been.

## Pins moved

| file | key |
|---|---|
| `cfg_gated_floors.txt` | `max_silent_now` **42 → 0**, now a **WALL**. Same argument as `max_load_bearing`: a new gated target with no row reds the push that adds it, before its green zero is read as evidence. The fix the report prints is three lines of TOML. Do not raise it. |
| `cfg_row_implication_floors.txt` | `min_rows_scanned` 560 → **620**, `min_with_cfg` 359 → **400**. The old floors sat at 83%/80% of observed after +42 rows — drifting toward decorative; restored to the ~92%/89% slack they carried before. |
| `cfg_row_implication_drift_floors.txt` | `min_rows` / `min_with_cfg` raised for six repos (real new population), and `max_unresolved` raised in **three** — katgpt-rs 0→2, riir-neuron-db 1→2, riir-train 1→3 — all five instances being the `any(debug_assertions, feature = "alloc_tracking")` shape from Issue 741, which is genuinely undecidable statically because the verdict depends on the PROFILE. `max_empty` stays a WALL at 0 everywhere. |

⚠️ The sweep **caught its own desync**: raising `min_rows` in the cross-repo
floors while leaving `cfg_row_implication_floors.txt` alone reds with an
explicit `pin desync: katgpt-rs min_rows 620 here vs min_rows_scanned 560 in …`.
That cross-assertion is deliberate — where a sweep re-states a quantity its
per-push gate owns, it **asserts** the two agree rather than trusting two
hand-typed copies. It is also why this issue's pin edits are one commit and not
two. Note the sweep is **workstation-only** and NOT in `docs_gate.sh`'s
`CHECKS`, so a green docs gate does **not** cover it — run it by hand after
landing rows.

## Cross-refs

- `.issues/741` — the profile axis (`alloc_tracking`) and the instrument fix
  (read EVERY whole-file `#![cfg]`, not the first). **"Issue 741" is ambiguous
  in this workspace** (riir-ai has one); spell it `katgpt-rs Issue 741`.
- Same class, same day, six repos: riir-game-sdk 31 (`2380fc7`), riir-chain 28
  (`.issues/138`, `7a763045`), riir-clippy 16 (`.issues/083`),
  riir-neuron-db 8 (`.issues/616`, `5d5e537`), riir-mmorpg-examples 5 of 6
  (`.issues/105`, `48f2914`), riir-train (`.issues/530`, `33f5c98f`).
- `.issues/713` T4 / `.issues/728` T2 — the two earlier partial clears of this
  same backlog, and the "silently UNVERIFIED, not silently broken" finding.
