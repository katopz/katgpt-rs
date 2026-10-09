# Issue 928 — cross-repo drift-sweep debt: the 2026-10-09 seal-std registration census + the same-day 928 repair wave

**Status:** PARTIAL — the 10-09 wave-1 unit landed (the census rows + every quiet-repo row the fresh re-runs surfaced); the volume rows in five repos and the sibling-hot repos remain, ledgered below

The seal-std registration (katgpt-rs `c007cf543`) ran all ~20 cross-repo drift
sweeps and surfaced committed sibling drift. The original census captured 9
rows across 6 repos; the wave-1 fix-order step ("re-run each sweep fresh")
surfaced substantially more — the census pass had been lossy. Everything is
now measured, ledgered, and either REPAIRED (wave 1) or DEFERRED with a reason.

## Wave 1 — REPAIRED (2026-10-09, all commits `Session: dapps-121-followup/katgpt-928`)

| repo | sweep(s) | repair | commit |
|---|---|---|---|
| katgpt-rs | instrument floors | the seal-std row c007cf543 claimed but missed (19 of 20) | `ce0fb09e5` |
| riir-game-sdk | cfg_gated (5 SILENT-NOW, 1 LOAD-BEARING) | 5 `[[test]]` rows; cas_g2 RUN 3 passed | `1892808` |
| riir-ai | cfg_gated (3 LOAD-BEARING) | 3 rows; bench_952 RUN 5 passed; attn_mass_tap_g3 blocked by the wgpu-30 lock drift → **riir-ai Issue 1046 filed**; plan614 non-macOS (4090) | `32908a05f` |
| riir-dapps | cfg_gated 1 + console 1 | kat_decstat_reward row + RUN 18 passed; build_flow_walks defended | `d267c27` |
| riir-infer | cfg_gated 4 + console 5 + citation 1 | 4 rows (metal_fold_bits RUN 2 passed; two carry their own #[ignore] arms); 5 defences; Research 327–332 qualified to riir-ai | `54dd734` |
| riir-deployer | citation 1 + locale_io 4 | plan-253 qualified to mmorpg-editor; the locale rows turned out to be GITIGNORED `.deploy/` artifacts — **instrument defect** | `ba67482` |
| katgpt-rs | locale_io (instrument) | **tracked_walk fix**: an empty tracked population is valid, never an rglob-fallback trigger (arm I; the fallback had pulled gitignored artifacts in as "committed" findings) | `7843e9d9e` |
| mmorpg-editor | citation CROSS 1 + IN-LOCAL 1 + console 1 | "plans 584 entries" (verb!) reworded; the `- **012**` record row added (a00b60d4/c4e6b031); actor_mirror_census defended. IN-LOCAL re-pinned 0→1 (list-form blind spot, the riir-reflex precedent) | `d32d133e` (develop; NOT on the feature/run-pack-only ticket branch) |
| reflex-site | instrument 4 + console 2 | 4 real instruments wired into the AGENTS table (incl. the pairing GATE) + the stale render_tetris_flows name fixed; 2 defences | `27b3fdc`→`1342919` |
| riir-instinct | citation 1 + console 1 | reflex issue-074 line-reflowed into the qualifying window; stamp_doc_gold_axes defended | `396612f` |
| riir-kat | citation 1 | issue-106 qualified to riir-dapps | `8e314f4` |
| riir-mmorpg-examples | citation 1 | issue-920 qualified to riir-ai (the [repeat] repair) | `796d453` |
| riir-shader | citation 1 (+2 in-local pinned) | plan-237 qualified to mmorpg-editor; 050/051 list-form rows pinned 0→2 | `7eebf45` |
| riir-train | citation 1 | bench-023 qualified to riir-instinct (its in_local posture returns to the 2026-09-16 pin) | `e6a4a676` |

Post-wave verdicts: **cfg_gated PASSED family-wide** (0 load-bearing in the
workspace, 13 SILENT-NOW total). **citation** green except mmorpg-remake.
**console_encoding** green in 5 repaired repos. **locale_io** green in
riir-deployer (+ the instrument fixed). **instrument_reachability** green in
reflex-site. Floor moves: cfg_gated game-sdk silent 31→0; citation
mmorpg-editor in_local 0→1; citation shader in_local 0→2 (all with cited
sibling commits, in the katgpt-rs wave-1 commit).

## Remaining — the wave-2 ledger (measured 2026-10-09 post-wave-1)

### Volume rows in quiet repos (the next unit; read rows before any pin move)

| repo | instrument_unreach (pin→measured) | console | locale |
|---|---|---|---|
| riir-train | 65 → **79** | 53 → **61** | 0 → **4** |
| riir-refine | 11 → **24** | 0 → **7** | 0 → **2** |
| riir-reflex | 0 → **14** | 0 → **11** | 0 → **21** |
| riir-shader | 2 → **12** | 0 → **1** | 0 → **8** |
| riir-rethink | 1 → **5** | 0 → **3** | — |
| riir-ai | 7 → **15** | 0 → **4** | 0 → **6** |
| riir-dapps | 1 → **2** | ✓ | ✓ |
| riir-infer | 0 → **7** | ✓ | ✓ |

(locale/console repairs: `scripts/locale_io_fix.py` + the backslashreplace
idiom; instrument rows: wire real instruments into a root or re-pin the
plan-scoped one-off class DELIBERATELY, citing the sibling commit — the
floors header's read-before-pin law.)

### Sibling-hot at wave-1 time (coordinate before touching)

- **mmorpg-remake (seal-remake)** — Runetrace/player-style session live:
  citation CROSS **10** (9 distinct decisions: Issue 190, Plans 299/300/301/311
  ×2, 317, 323, Issue 186 [repeat], Issue 897) + instrument 7 → **8** +
  console 2. Its 3 dirty population files were the Issue-797 advisory.
- **riir-ai** beyond the cfg rows — locale 6 + console 4 + instrument +8:
  the games-mmorpg WIP session owned the tree; re-measure when quiet.

### Instrument findings (katgpt-rs lane)

- **len_derived stability**: dropping riir-ai flips 2 riir-infer verdicts
  (`elementwise_cubecl.rs:617/:652` input_handle EXACT-UPSTREAM → UNRESOLVED)
  — a partial-clone box would report a WRONG bucket. Needs a design decision
  (caller-side shape parameterization, or a partial-clone-safe pin class);
  not mechanical.

## The class law (unchanged)

- **PIN-STALE** (sibling landed; pin did not move): re-pin to measured in the
  re-pin commit, citing the sibling commit.
- **CODE-DEBT** (sweep names a repair tool; ceiling may NOT rise): fix the
  sibling's code, never the pin.
- **LOAD-BEARING walls never rise** — arm with `required-features` rows and
  RUN at the feature set (the Issue-728 law); check binary-counting floors
  first (riir-game-sdk's gate counts PASSED tests — safe; checked).

## Related

- katgpt-rs `c007cf543` (the registration), `ce0fb09e5`, `7843e9d9e` (+ the
  wave-1 pins commit)
- riir-ai Issue 1046 (the wgpu-30 lock drift the arming surfaced)
- Issues 728/749/781/797/804/823/828 (the law citations above)
