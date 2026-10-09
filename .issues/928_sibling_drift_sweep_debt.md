# Issue 928 — cross-repo drift-sweep debt: the 2026-10-09 seal-std registration census + the same-day 928 repair wave

**Status:** PARTIAL — wave 2 LANDED for every QUIET repo (2026-10-09: riir-train/refine/reflex/shader console+locale repaired to 0; infer+instinct+shader+reflex+refine+train instrument floors re-pinned at measured with per-row adjudication). Remaining: the four sibling-hot repos (mmorpg-remake, riir-ai, riir-dapps, riir-rethink) — re-measure + repair when their sessions end — and the len_derived design question.

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

## Wave 2 — LANDED 2026-10-09 (the quiet-repo volume rows; commits `Session: dapps-121-followup/katgpt-928`)

Repairs (console defence = the wave-1 backslashreplace idiom, line-ending-preserving, py_compile-gated; locale = `scripts/locale_io_fix.py`):

| repo | console | locale | repair commit |
|---|---|---|---|
| riir-shader | 1 defended | 8 sites / 7 files | `1f2a195` |
| riir-refine | 7 defended | 2 sites / 1 file | `c126f441` |
| riir-train | 61 defended | 4 sites / 1 file | `45fc7799` |
| riir-reflex | 11 defended | 21 sites / 6 files | `00b4ee1` |

Instrument floors re-pinned AT MEASURED (read-before-pin adjudication in the row comments, adding commits cited):

| repo | pin → measured | class |
|---|---|---|
| riir-train | 65 → 79 | +14 plan/issue one-offs (the repo's own adjudicated class) |
| riir-refine | 11 → 24 | +13 mining pack/freeze probes (the Issue-837 class) |
| riir-reflex | 0 → 14 | first re-measure since birth; issue/bench lane probes |
| riir-shader | 2 → 12 | bake/thumb one-offs documented in README (not a root — the mmorpg-editor precedent) |
| riir-infer | 0 → 7 | plan616/617/618 probes (consoles defended in wave 1) |
| riir-instinct | 0 → 3 | one-task artifacts recorded only in HISTORY.md |

Post-wave-2 verdicts: **console + locale sweeps green in every repaired repo** (undefended 0, locale 0); instrument green everywhere except the four sibling-hot repos. min floors raised to ~60% of measured everywhere they were birth-stale (reflex 0/0/0 → 15/12/0 etc.). ⚠ The CRLF lesson: the first console-defence pass converted four CRLF probe files to LF (splitlines+join) — the amended `c126f441` preserves line endings; the shared_temp_path_fix.py design law (ending-preserving repairs) applies to session tools too.

## Remaining — wave 3 (sibling-hot; re-measure when their sessions end)

- ~~mmorpg-remake (seal-remake)~~ — **DONE 2026-10-09** (`b69ee61`): the ten CROSS citations qualified
  (owners verified via git history: seal-game-editor issues 190/186 + plans 299/300/301/311/317/323;
  riir-ai issue 897 the opt-level substrate fork), the two poc consoles defended,
  instrument re-pinned 7→8 (+inspect_glb_plants, 035431a) and console ratchet lowered 4→0.
- **riir-ai** — instrument 7→15 + locale 6 (the vessel session owns the tree — untracked
  `dapp_vessel.rs` live in crates/riir-mcp-client at wave-3 time; console 4 too).
- **riir-dapps** — instrument 1→2 (the vessel-bench session).
- **riir-rethink** — instrument 1→5 + console 3 (the issue-025 T3 session).

### Instrument findings (katgpt-rs lane)

- ~~**len_derived stability**~~ — **RESOLVED 2026-10-09 (wave 3, the design decision made)**:
  the partial-clone-safe PIN CLASS. `len_derived_stability_expected.txt` holds
  adjudicated flips by MEMBERSHIP, line-free, with the flip pair recorded; the
  DIRECTION LAW is enforced in the parser (a partial box may only ever be pinned
  into UNRESOLVED — never a resolved bucket it cannot see); reds BOTH directions
  (new flip = new cross-repo dependence; retired flip = drop the row). Canary
  arms 11/11b/11c pin all three behaviours (15/15). The two measured rows
  (elementwise_cubecl sigmoid_f32/silu_f32 input_handle, resolved through a
  riir-ai caller that landed after the 09-14 zero-flip measurement) are pinned
  with the root fix NAMED as the retirement condition: caller-side shape
  parameterization — the kernels take the element count as a launch parameter
  instead of deriving it from input_handle's declared len (a riir-infer +
  riir-ai change, deferred while the riir-ai tree is hot).

## The superseded wave-2 ledger (measured 2026-10-09 post-wave-1; now landed)

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
