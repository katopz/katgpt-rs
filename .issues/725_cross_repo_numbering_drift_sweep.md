# Issue 725 — the numbering gate covers ONE repo; 35 duplicates and 7 broken allocators sat in the other fifteen

**Status:** T1-T3 + T4a DONE (instrument landed, all 7 allocator defects repaired, riir-ai's 6 duplicates arbitrated, 2026-09-05). T4b/T5 OPEN — 35 tracked duplicates across 4 repos are ratcheted so no new one can land; the renames need a per-site citation rewrite.

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

- [ ] **T4b — execute the renames.** Owner-by-owner, by CITATION WEIGHT
      per Issue 724 T2's precedent (the file with the most inbound mentions keeps
      the number; the other moves to a fresh one and its citations are updated).
      Ratcheted at the measured count per repo, so a new collision reds while the
      backlog stays visible. `seal-game-editor` (12) is READ-ONLY to these
      sessions — report only. Lower a repo's pin in the commit that resolves one.
      Each rename is a file move PLUS a hand rewrite of that document's inbound
      `Plan N` citations, which is why T4a stopped at the verdict: 28-50% of the
      sites in every riir-ai pair are UNRESOLVED, and a mis-attributed rewrite
      silently re-points a reader to the wrong document — strictly worse than the
      collision it fixes. Run `scripts/citation_weight.py <repo> <dir> <N> --show 20`
      and read the sites before moving anything.
- [ ] **T5 — should the siblings run the gate themselves?** The sweep is a
      workstation instrument; the per-push half only exists in katgpt-rs. The
      `sibling_docs_drift.yml` `workflow_call` pattern is the obvious answer and
      is NOT taken here, because `numbering_gate.py`'s pins file is
      katgpt-rs-scoped by its own first line. Deferred, not forgotten.

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
