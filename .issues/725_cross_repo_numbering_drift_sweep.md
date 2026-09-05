# Issue 725 — the numbering gate covers ONE repo; 35 duplicates and 7 broken allocators sat in the other fifteen

**Status:** T1-T4c DONE (instrument landed, all 7 allocator defects repaired; duplicates: riir-ai 6/6 executed 2026-09-05, riir-clippy 4/4 via its Issue 069 `58e7c1d`, riir-train 13/13 via its Issue 514 `103ed351` — workspace dup 35 → **12, all remaining in read-only seal-game-editor**). T5 `[-]` deferred — the sibling-gate design question is recorded below.

## The finding, in one sentence

`scripts/numbering_gate.py` accepts a repo path and has been pointed at exactly
one repo since Issue 724 landed it, so a sibling allocating the same number
twice is indistinguishable from a sibling nobody looked at — and on the first
sweep, **katgpt-rs (the only gated repo) had zero duplicates while four
siblings had 35.**

This is the Issue 702 shape one instrument over, and the workspace already
carries the fix pattern for it (`scripts/docs_drift_sweep.py`). The gate was
written, canaried in five directions, wired into `docs_gate.sh`, and never
asked about anyone else.

## Measured 2026-09-05 — 16 contract repos, derived population

| class | count | repos |
|---|---|---|
| tracked duplicate numbers (serial dirs) | **35** | riir-train 13, seal-game-editor 12, riir-ai 6, riir-clippy 4 |
| `.highwater` present but NOT an integer | **5** | riir-train 2, riir-game-sdk 2, riir-ai 1 |
| `.highwater` below its directory max | **2 → 3** | riir-chain, riir-clippy, **+ riir-train (revealed by the repair)** |

### The malformed class is new, and it is why the ceiling could not fail

All five malformed files have one cause: **`echo -n <N> > .highwater` under a
shell whose builtin `echo` does not implement `-n`**, so the flag itself lands
in the file — `-n 872`, `-n 374`, `-n 005`. The number is still readable to a
human, which is why it survived; it is not readable to `int()`.

`scan()` swallowed the `ValueError` and returned `None` — **which is also what
an ABSENT allocator returns.** Not every numbered directory has one, so `None`
was treated as "nothing to check", and the `max > highwater` ceiling passed
over a corrupted allocator and a clean directory identically. katgpt-rs has no
malformed file, which is exactly why the blindness survived in the gated repo.

**The repair proved the point the same minute it landed.** Fixing
`riir-train/.plans/.highwater` from `-n 374` to `374` immediately exposed a
stale allocator underneath it (max 375 > 374) that had been invisible for as
long as the file was corrupt — a third defect the census could not see because
the instrument reading it was blind.

## Tasks

- [x] **T1 — `numbering_gate.py`: split ABSENT from MALFORMED.** `read_highwater()`
      returns `(value, malformed_raw)`; a present-but-unparseable file is its
      own finding, pinned `max_malformed_highwater = 0`. `selftest()` case 6
      canaries the collapse (regressing `read_highwater` to the old blind
      behaviour trips three pins → exit 2, verified).
- [x] **T2 — `scripts/numbering_drift_sweep.py`.** Workstation-only cross-repo
      sweep, derived population (BOUNDARY.md + a `.git` **dir**), committed
      expectations (`scripts/numbering_drift_floors.txt`). Deliberately NOT in
      `docs_gate.sh`'s CHECKS — CI has one checkout, so it would derive an empty
      population and print a confident green over zero repos (the
      `docs_drift_sweep.py` precedent, same reasoning verbatim).
- [x] **T3 — repair all 7 allocator defects** (5 malformed + 2 stale, then the
      third stale the repair revealed), value-preserving and padding-preserving:
      `riir-ai/.issues` 872 · `riir-game-sdk/.plans` 005 · `riir-game-sdk/.issues`
      026 · `riir-train/.plans` 374→375 · `riir-train/.research` 441 ·
      `riir-chain/.proposals` 007→008 · `riir-clippy/.plans` 78→80.
      Pinned 0 everywhere — these are one-line repairs with no arbitration
      attached, so there is no reason to tolerate one.
- [x] **T4a — the arbitration instrument + riir-ai's verdicts** (`scripts/citation_weight.py`,
      2026-09-05). The obvious count is the wrong one: **by-name citations are 0-2 per
      side and TIED in four of riir-ai's six pairs**, while the `Plan N` form carries
      33-92 sites each. The weight is entirely in the citations that do not say which
      document they mean, so they are ATTRIBUTED by token overlap in a context window,
      on a strict margin, with everything else in an UNRESOLVED bucket that is printed
      and never folded into a winner.

      | dup | keep (attributed + by-name) | move | unresolved |
      |---|---|---|---|
      | `.plans/175` | `civ_engine_emotion_proof` 41+1 | `lattice_calculus_latcal` 3+1 | 23 (34%) |
      | `.plans/182` | `luce_megakernel_deltanet_inference` 17+0 | `civ_map_2d` 4+0 | 12 (36%) |
      | `.plans/229` | `vortex_meta_routing_game_ai` 26+0 | `gm_tool_crate_daynight` 0+0 | 10 (28%) |
      | `.plans/313` | `step_attribution_branch_wiring` 31+2 | `swir_real_model_validation` 15+2 | 46 (50%) |
      | `.research/020` | `Zone_Expert_Bundles_Living_World` 14+2 | `Orbit_OFT_Adapter_First_RL` 2+3 | 1 (6%) |
      | `.research/148` | `think_brain_wasm_vessel` 23+10 | `Per_Tick_Emit_Salience_NPC_Guide` 7+5 | 30 (50%) |

      **No rename was executed, and the reason is the finding.** `.plans/229`'s day/night
      plan scores a clean ZERO, which reads as "nothing cites it, safe to move" — and a
      hand check found `.docs/01_orientation/overview.md:149` and `.proposals/007:236`
      both citing it, spelled `riir-gm-tool` and `day/night` against a stem of `gm_tool`
      and `daynight`. It is in the UNRESOLVED 10, exactly where the design puts it, but
      a reader skimming the winner column would have moved it. The report now refuses to
      let a zero pass silently (a warning line), and the attributor is
      separator-insensitive. **A zero in a bucketed audit is a claim about the
      vocabulary, not about the world.**

- [x] **T4b(riir-ai) — all 6 executed 2026-09-05.** `.plans` 175->568 (LatCal, 20
      citations), 182->567 (civ 2D map, 9), 229->566 (GM tool/day-night, 8),
      313->569 (step-attribution, 38 + the companion bench doc); `.research`
      020->362 (Orbit/OFT, 5), 148->363 (per-tick salience, 6). **riir-ai dup 6 -> 0**;
      its pin is now 0 so a regression reds. Workspace total 35 -> 29.

      Four things the execution taught that the T4a measurement could not:

      1. **The section number beats the prose.** On `.research/148` the token
         attributor put 11 sites on the salience guide and 10 of them were the WASM
         vessel — the two documents describe the same subsystem from opposite sides,
         so `gate`/`npc`/`tick`/`emit` do not discriminate at all. What did: every
         vessel citation carries §1.2/§1.4/§3 and every salience one §5. A citation's
         own sub-reference is a stronger signal than its context when two documents
         share a subject, and no token widening would have found it.
      2. **A third document is often in the population.** `Plan 313` in riir-ai also
         means **katgpt-rs**'s Plan 313 (AC-Prefix) at ~8 sites; `Research 148` also
         means katgpt-rs's Hydra Effect at one; `Research 020` also means katgpt-rs's
         TurboQuant. Bare cross-repo citations are ambiguous by construction and a
         rename cannot fix them — each renumber note says so.
      3. **Select inclusively, never exclusively.** `.plans/175`'s candidate set was
         built by dropping lines that mention the winner's vocabulary, and that
         silently dropped `.plans/183:6`, which cites five plans on ONE line including
         the loser. The inclusive residual grep — search for the LOSER's vocabulary
         among what remains — caught it. All other pairs used the inclusive form.
      4. **A tie needs a rule, not a nudge.** `.plans/313` was 45 vs 41 over 82 sites
         and had pointed the other way on the narrow-dialect pass. `TIE_FRACTION = 0.10`
         now names that, and the fallback is creation order (the allocator's own
         semantics) — confirmed independently by source-artifact cost: SwiR's number is
         in `[[test]]` filenames, the step-attribution plan's in nothing.

- [x] **T4b(rest) — MEASURED and FILED with the owners, 2026-09-05.** Not executed
      here, and the reasons differ per repo:

      - **riir-clippy 4 → `riir-clippy/.issues/069`. RESOLVED 2026-09-05** (riir-clippy
        `58e7c1d`, owner go given): all four renumbered per the table below — `.plans` 015→084,
        026→085; `.research` 083→135, 086→136 — **17 citation rewrites** (the 13 priced + 4
        stragglers caught by the both-names grep, incl. one live `**Plan:**` link); the
        `.research/083` owner concern priced at 2 sites was executed without breakage; T3's
        design call decided **(a) one counter, enforced** and recorded in that repo's AGENTS.md
        numbering section; the allocator repaired 78→80 in the same window.
      - **riir-train 13 → `riir-train/.issues/514`. RESOLVED 2026-09-05** (pushed
        `103ed351`, 14 commits): all 13 pairs arbitrated and executed — the four rows 725
        called mechanically UNDECIDABLE and the two ties all resolved by HAND reads (073's
        86%-unresolved mass was katgpt-rs's R073 LT2, an unrelated namesake; 083's "exact
        tie" was 3-1 on verbatim names; ties broken by creation order per the T4a fallback).
        Keepers + moves: `.plans` 186(ldt→377) 250(d2f→379) 252(wf→380) 253(memory_soup→385)
        255(quant_robust→378) 264×4(→381/382/383) 267(manifold_bake→386) 363(chat_dflash2→384);
        `.research` 073(→448) 083(→446) 084(→447) 086×2(→444/445) 095(→443). Known cost,
        documented per renumber note: number-baked test filenames (`goat_250_*`,
        `bench_253_*`, `bench_255_*`) and ~30 source comments stay stale for the next
        code-touching session. The lesson over 725's pessimism: **the instrument is
        advisory; the corpus does not distinguish near-synonyms, but reading the actual
        sites does.**
      - **seal-game-editor 12** — READ-ONLY to these sessions. Report only; the sweep
        keeps it ratcheted at 12.

**The ratchet's first catch was its own author, 20 minutes later.** The riir-train
issue was written as `513_` against a `.issues/.highwater` read at the START of this
session; `513_required_features_rows_are_unverified.md` took 513 at 07:56 while this
one landed at 08:16. `numbering_drift_sweep.py` went red on riir-train's pin (13 → 14)
within minutes and the issue was renumbered to 514 (riir-train `42c3bd1c`). The failure
was not the race — it was **reading the allocator early and writing late**. A
`.highwater` is only true at the instant it is read, and a session that caches it is
allocating from a memory rather than from the allocator. That is worth more than the
gate passing would have been: it is the mechanism this whole issue documents,
committed by the session documenting it, and caught by the instrument built for it.

- [x] **T4c — the ratchets come down as owners land pairs.** DONE 2026-09-05: `riir-ai 0`,
      `riir-clippy 0` (Issue 069), `riir-train 0` (Issue 514); `seal-game-editor 12` is
      READ-ONLY to these sessions — report only, ratchet stays at the measured count. Each pin
      lowered in katgpt-rs `scripts/numbering_drift_floors.txt` as its repo resolved (`riir-clippy`
      by the sibling's `58e7c1d` window, `riir-train` in this closeout). Workspace dup 35 → 12,
      all remaining read-only.
- [-] **T5 — should the siblings run the gate themselves?** DEFERRED (2026-09-05): the
      sweep is a workstation instrument; the per-push half only exists in katgpt-rs. The
      `sibling_docs_drift.yml` `workflow_call` pattern is the obvious answer and is NOT taken
      here, because `numbering_gate.py`'s pins file is katgpt-rs-scoped by its own first line
      (riir-clippy's Issue 069 T3 recorded the same conclusion from the other side — that repo
      uses the sweep, or gives itself its own pins). Reopen when a second repo wants its own
      per-push gate.

## Why the ceilings are a ratchet and not zero

A pin of 0 on `max_dup` would red on landing in four repos this session does not
own, and a red nobody can clear trains the next reader to assume the tool is
broken — the exact failure `docs_gate.sh`'s preamble records. Pinned at measured:
**a new collision reds immediately; the standing backlog is data in the pins file
rather than silence.**

## Record

- `scripts/numbering_drift_sweep.py` (docstring = the narrative),
  `scripts/numbering_drift_floors.txt` (the per-repo ratchet + the reasoning).
- `scripts/numbering_gate.py` `read_highwater()` + `selftest()` case 6.
- `scripts/citation_weight.py` — the T4 arbitration instrument (advisory report,
  always exit 0), with the zero-is-not-zero guard measured on `.plans/229`.
- Canaried before landing, four directions: a planted duplicate in a 0-pinned
  repo reds; a pinned-but-absent repo reds; an empty pins file exits 2; a
  regressed `read_highwater` exits 2.
