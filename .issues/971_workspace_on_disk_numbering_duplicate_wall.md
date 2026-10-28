# Issue 971 — the on-disk numbering-duplicate wall is riir-train-scoped: the riir-ai Bench-1011 double-hold (2026-10-28) was visible to NO automatic gate in 15 of 16 repos — fan out the wall to the workspace sweep family

**Status:** OPEN — filed 2026-10-28 (m3 standing-duty lane). The incident is RESOLVED (riir-ai `cfd600ed0`); this issue owns the DETECTION gap so the next double-hold is caught by a gate, not by a sibling lane's closing-verdict session.
**Date:** 2026-10-28
**Type:** workspace gate fan-out (the numbering family; `numbering_drift_sweep.py` + riir-train's `numbering_duplicate_gate.py`)
**Discoverer:** Claude closing-verdict session 276450a7 (the 641-T4 owner-gate lane), handed to the m3 numbering lane by session d91ea5bb — both of whom should be cited on the fix.

## The incident (measured, 2026-10-28)

riir-ai `.benchmarks/1011` was doubly held on ONE line of history for ~15 h:

- `1011_641_t4_window3_m3_cell_promotion.md` — added by the 886-lane's renumber commit `9da4bffd7` (01:50 +07; the 641 record's own forced move off 1010, which the 886 lane had allocated first).
- `1011_890_t6a_gate_attestation_m5.md` — added 02:41 (`e092c8fab`) by a lane whose highwater view pre-dated the 01:50 renumber.

Resolved same-day at `cfd600ed0` (890 renames to 1015; first-allocation rule 1 decided — citation_weight's 890-side lead was WEAK and explicitly not awarded, the caveat rides the commit body).

**Every automatic gate read ZERO on the live defect:**

| gate | verdict on the live double-hold | why |
|---|---|---|
| `dual_allocation_gate.py` (katgpt-rs, per-allocation) | 0 | its documented reach limit: it sees CROSS-CHECKOUT divergence only; both holders sat on ONE checkout's history line |
| `numbering_drift_sweep.py` (katgpt-rs, workspace) | `dup=0` for riir-ai | `dup` covers the SERIAL dirs only — **`.benchmarks`/`.docs` are excluded by scope** ("a family per owner is the intended convention, so duplicate is not decidable without knowing which convention a file follows") |
| `numbering_duplicate_gate.py` (riir-train, issue 556 step 3 — the gate that CAN decide `.benchmarks` families via its FAMILY-EXPLICIT/SUFFIX-FAMILY/HAND-READ classifier) | not run — it is **riir-train-scoped** (hardcoded repo root + its own pins/floors files) | the wall exists in exactly one repo |

**Measured proof the classifier catches this exact class** (m3, 2026-10-28, scratch-copy reproduction of the pre-fix state — both 1011 files restored at `cfd600ed0~1` and the gate run against the copy): the gate reports `⛔ UNPINNED HAND-READ .benchmarks/1011: 641_t4_window3_m3_cell_promotion.md, 890_t6a_gate_attestation_m5.md` — a loud FAIL without any pin. The instrument is ready; it just does not run outside riir-train.

## The proposal

Fan the wall out to the workspace, lifting riir-train's classifier — DRY over re-derivation:

1. **Substrate + direction law.** katgpt-rs is UPSTREAM — a katgpt-rs script must NOT import riir-train. The DRY-preserving shape: the classifier stays ONE copy (riir-train, issue 556's), and the workspace layer either (a) subprocess-invokes the sibling gate per repo where present (loose coupling, the `ci_boundary_contract.sh` shape), or (b) the gate gains a `--repo-root` workspace-walk mode it already half-has, driven by a thin katgpt-rs sweep wrapper. Decide by reading `scripts/numbering_duplicate_gate.py`'s seams — the reproduction ran it perfectly via `--repo-root`, so (b) is mostly a population walk + a pins/floors story.
2. **`.benchmarks`/`.docs` come IN scope** for the duplicate class, with the classifier's own family conventions (FAMILY-EXPLICIT / SUFFIX-FAMILY / RENAME-RESIDUE / HAND-READ) deciding "duplicate" — this is exactly the decidability problem the drift sweep's scope exclusion dodged, and riir-train's gate already solved it (measured above: 809/872/1008 COLLISIONs surfaced in riir-ai's tree beside the 1011 double-hold — **three MORE live .benchmarks collisions the workspace has never gated**, see the observed-state note below).
3. **Advisory tiebreak columns, never auto-award.** First-allocation + citation_weight print as DATA beside each finding; the award stays human. The incident is the proof: weight pointed 890, first-allocation pointed 641 — an auto-award by weight would have moved the WRONG file.
4. **Per-repo, per-dir population floors** (the riir-trai​n gate's `floor|dir|min` rows), so a walk regression exits 2 rather than printing a confident zero. Floors must be measured per repo at landing, not copied — riir-ai's `.issues` floor read 57 from a fuller box vs 29 measured here (issues close-and-remove under the noise rule).
5. **Pins by membership with a reason per row** (`*_expected.txt` shape), red in both directions; the verdict rides the ROW KEY (repo, dir, number, verdict) per the Issue-797/815 laws. Family exemptions (plan-421-style sub-parts) get verified HAND-READ rows like riir-train's.
6. **Sweep-family wiring.** A new `*_drift_sweep.py` must call the family-wide mechanisms (worktree-advisory via `worktree_state`, known-extra exemption, head-provenance, alias-open) and be registered in `sweep_advisory_membership_gate`'s MECHANISMS registry AFTER the fan-out — the katgpt-rs landing protocol. If the shape chosen is (1b) — a wrapper over the sibling gate — the sweep_advisory registry question still applies to the wrapper.

## Observed state at filing (disclosure, not findings of this issue)

- riir-ai live `.benchmarks` collisions the reproduction surfaced: **809, 872, 1008** (1008 is a TRIPLE: `issue1127_t4_recell_bundle` + `issue1132_metal_launch_floor_census_w1` + `issue1132_w2_reduce.py`). Unadjudicated — owner plan/issue identity per group not yet determined; they become the new wall's first population when it lands.
- riir-ai pre-existing drift: `.issues` highwater stale vs holders 1135/1136; stale HAND-READ pins (421/423 plan-family groups "no longer on disk"); floors pinned from a fuller box. riir-ai's own lanes'.
- `numbering_drift_sweep.py` is red on riir-refine (hist=6 vs max_hist=5 — a new historical collision) and riir-ai (stale highwater + resets) as of this filing — pre-existing, disclosed so the new wall's first run is not blamed for red it inherited.
- The naming law quirk this incident also exercised: `dual_allocation_gate` is armed on every push but cannot see same-checkout divergences BY DESIGN; its complement (this wall) was armed in ONE repo. The pairing is the finding.

## Tasks

- [ ] Adjudicate riir-ai's surfaced live collisions (809, 872, 1008) — owner identity per group, renumber the losers per the rule stack (riir-ai's own lanes; the wall's first population).
- [ ] Land the workspace duplicate wall per the proposal above (substrate decision, `.benchmarks` in scope, advisory columns, floors, pins, sweep-family wiring + registry).
- [ ] Repin `numbering_drift_floors.txt` rows the landing moves (riir-ai stale rows at minimum), and re-pin or repair riir-refine's hist=6>5 breach or route it to refine's lanes.
- [ ] Optional hardening: `dual_allocation_gate`'s README gains a pointer at this issue (the two instruments are complements; neither's green implies the workspace clean).
