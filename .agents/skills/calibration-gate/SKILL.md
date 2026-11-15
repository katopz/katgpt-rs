---
name: calibration-gate
description: The evidence-gated calibration loop for any tunable surface - drive weights, need sigmoids, gate rho, thresholds, dispatch narrowing, eval bars, UQ calibrators, rig-posture cells. Verdict-first authority, per-group evidence census, pre-registered bars and falsifiers, hardened queue cycles that die loud in pre-flight, separate calibration and production cells, box-state provenance on every number, per-group classification with honest NO VERDICT. Use BEFORE planning or running any calibration or tuning sweep, when a calibration run keeps dying pre-run, when pinning perf bars from a calibration run, when adjudicating Calibrated vs Sparse vs Fallback, or when asked how a calibration went. Sibling to build-hygiene (pre-registered A/B bars) and grammar-not-answers (contamination gates).
---

# calibration-gate — the evidence-gated calibration loop

The loop the fleet already runs when it tunes anything measurable. It is written down here
because every lane re-derived it once — this file is the next lane's head start. The
canonical walk-through is `../seal-sprints/.plans/029_shopping_chain_useitem_and_shopper_chaos.md`
(the shopping-drive calibration, 2026-11-13→15): read its Status line as the worked example —
delegated verdict round 1 (AGREE + binding amendments A1–A4), four machinery-review rounds
(REVISE ×3 → AGREE round 4), then calibration queue cycles v1/v2/v3 where each death bought a
permanent pre-flight defense. The precedents at the bottom name one lane per step.

The loop is a process representation, not a recipe: every area plugs in its own evidence
classes, groups, and bar vocabulary. The skill removes the re-derivation, not the judgment.

## When this fires

- BEFORE planning or running any calibration, tuning sweep, A/B, or bar-pinning run.
- When a calibration run keeps dying pre-run — that is the loop working; each death buys a
  permanent defense, never a weaker guard.
- When someone asks "how did calibration go" — answer in this loop's vocabulary
  (authority → census → pre-registration → cycles → classification → pinned bars).
- When adjudicating a result — Calibrated / Fallback / Sparse / NO VERDICT are the only verdicts.

## The loop

1. **Authority before runs.** Plan-first for any calibration that touches content or spends
   rig time: file the plan, claim it (`**Claimed:** <session>, <epoch>, expires <ISO-UTC>` —
   ONE claim line per file; a malformed line — prose after the expiry token, a millisecond
   epoch, a +07:00 offset — blinds the lease checker for EVERY lane), and get the delegated
   verdict (Claude ping-pong, proposal 001; max rounds per `agent.verdict_max_rounds`) BEFORE
   code or runs. Record binding amendments BY NAME (A1–A4 style) and carry them forward —
   verdict rounds revise the plan, never silently widen it. Machinery reviews are
   round-numbered in the plan so a successor thread picks up at the right round.

2. **Census the evidence before tuning anything.** Count the labeled evidence PER GROUP
   before proposing values — pin RAW counts as data and derive the verdict, so a floor change
   never re-mines. Precedents — riir-refine i233 pinned per-(domain × shape) RAW census
   counts (180 labeled spans) before any narrowing; katgpt-rs bench 982 printed its label
   census (315 fail / 320 requests) and REFUSED to adjudicate when one class was degenerate.
   If a group has no evidence, it has no verdict — an empty window ABSTAINS (the adoption-gate
   posture), it never invents.

3. **Pre-register the decision rule before pair 1.** Bars, thresholds, floors, falsifiers,
   promotion bar, stop rule — written and committed BEFORE the run (sr1095's rule: the
   promotion bar + stop rule bound before pair 1; riir-infer plan 633 landed every gate
   explicit — G1 FAIL +1.46 CER, G2 FAIL 0.57× vs the CPU lane, G4 PASS). A post-hoc bar is
   a tune-to-pass, and a tune-to-pass measures the tuner. Include up front: the small-group
   rule (e.g. n < 20 → Sparse; the 30/30 two-class label-census floor), the permutation-null
   or canary falsifier that proves the gate CAN fail, and the demote literal (env-switch law:
   a promoted arm's env is DEMOTE-ONLY, an opt-in arm's PROMOTE-ONLY — spell it now).

4. **Separate the calibration cell from the production cell.** Bars are pinned FROM the
   calibration run; the production cell (N=100, real contention, don't go easy) then runs
   UNDER those bars — never the reverse (sprint#029 tasks 6/8; bars are never read off the
   calibration load). A green production run that never stressed the thing it claims to
   stress is a SKIP, not a pass. And a calibration law discovered in the calibration cell
   PROPAGATES to the production cell BY NAME (fresh press + money-only pre-mint + account
   key are load-bearing for purchases — a reused old vessel SKIPs on zero purchases by
   construction).

5. **Queue cycles die loud, die cheap — and each death lands a defense.** Expect the first
   queue cycles to die in pre-flight (v1 died-by-design on a missing pin; v2's vessel dump
   revealed three fatal gaps — empty shop/merchant books, penniless bots, a throwaway-key
   signature; v3 carried the full corrected sequence IN the job). The law: every pre-flight
   catch becomes a PERMANENT defense in code (boot strips, funded wallets, empty-book
   pre-flight refusals that name the fix), never a one-off patch on a job script. Never
   weaken a guard to get a green. Press fresh whatever the verdict depends on (content,
   signing key) — a stale artifact's green proves the stale artifact.

6. **Measure with provenance and stability.** Every quoted bar carries its
   `bench_preflight` PROVENANCE line (box state, power source, load class). A/B pairs are
   interleaved and two-run stable (`katgpt-rs/scripts/sequential_ab_timing_audit.py` — two
   sequential arms measure the scheduler, +5–22%); paired non-inferiority δ = max(1.0pp,
   2.5·SE). Queue long cells via `agent-mail rig run --exclusive --est <estimate>
   --timeout <bound>` on a verified-quiet rig — never bench a BUSY box, never bench beside
   a GPU consumer, and re-run any row a contender touched ALONE before quoting it.

7. **Classify per group, never pool; withhold on absent evidence.** The verdict vocabulary
   is the calibrated_dispatch shape — **Calibrated** (Wilson 95% LCB ≥ τ over ≥ N labeled),
   **Fallback** (measured under-floor → demote the lever), **Sparse** (n < N → withhold;
   narrowing stays OFF — the evidence DEMOTES the unproven lever, it never hopes).
   Conservative per-bit: one non-Calibrated bit falls the whole surface back to the full
   posture (the dispatch narrows, never widens). An undecidable cell prints NO VERDICT
   (exit 3, every envelope term named) — never fold it into a pass; count unknowns loud;
   pool nothing across groups, ever (the per-predicate law).

8. **Pin, inherit, demote, record.** Pin the bars beside their PROVENANCE; carry the
   inherited laws into the next cell's plan (named, not folklore); demote the losing arm
   (never delete it); record verdict + re-open conditions in the plan/issue so the next
   lane starts at step 4. A negative calibration is a RESULT — i1216's head-surface PoC
   died at its own T0 headroom gate; plan 498's QAD head collapsed; the i237 composition
   cell NO-GO — the record IS the yield.

## The orchestration half — watched, not babysat

Long calibrations are park-and-notify: `agent-mail rig run` with `--est` (the tiered queue
sorts by estimate) + a timeout; END the turn; mail arrives at start/poison/end. Continuity
across context overflow is the plan's Status line — a successor thread reads the plan and
the out-dirs, never re-derives; fold superseded claims into `Lease-closed` rows so exactly
one Claimed line remains. Watchdog pickups poll on their cadence; a backstop wake is
`agent-mail send --to <own-inbox>@<rig> --at <time> "check <job>"` then END the turn.
Verdict pickups answer the verdict round in the SAME session, not a new thread. Rig log
purges completed jobs — check the WORK DIR before declaring a job vanished (the duel
d241af5d + amo 20979a65 near-mistakes, 2026-11-15). Housekeeping lanes verify their own
windows only; they never re-run or re-adjudicate a live lease.

## Never rules

- Never tune to pass; a bar written after the run is not a bar.
- Never pool group verdicts; never widen on Sparse evidence.
- Never quote a number without its box state; never bench a BUSY rig.
- Never re-pin a floor from a dirty, poisoned, or stale-posture run — disclose, re-run clean.
- Never gate correctness behind the thing being calibrated (lossy surfaces gate on
  per-family conditional retention, never aggregates).
- Sigmoid, never softmax; decline and abstain are typed answers, not guesses.
- A calibration that cannot fail is not a calibration — arm the canary, prove it fires.

## Precedents (one lane per step)

- sprint#029 shopping-drive calibration (`../seal-sprints/.plans/029_...`) — the full
  walk-through: verdict rounds, v1/v2/v3 pre-flight deaths, the inherited-laws handoff.
- riir-refine i233 `calibrated_dispatch` — census → Wilson LCB → Calibrated/Sparse/Fallback,
  conservative per-bit narrowing, the re-run census instrument.
- katgpt-rs bench 982 (i1210 C2) — the census-floor NO VERDICT + permutation-null falsifier,
  re-open conditions recorded.
- riir-rethink ESC ρ gates (issues 052/054/064) — per-suite gate-rho tables, rate-targeted
  fit, collapse caveats carried on the promote-eligible verdict.
- seal-remake sprint!053 — drive weights past the sigmoid ceiling → the zeroing form;
  verdicts as lane-posture pins with an armed witness row.
- riir-train plan 497 — the +700% artifact re-attributed to +131.5% real (per-dispatch
  shader compiles + blocking readbacks), paired harness as the record.
- riir-infer plan 633 — every gate explicit and honest, negatives quoted beside positives.
- sprint#029 tasks 6/8 — calibration cell separate from the chaos cell, SKIP-not-pass,
  PROVENANCE beside every bar.

## Instruments

- `agent-mail rig status` / `agent-mail rig run --exclusive --est` — queue, quiet-gate, tiers.
- `riir-reflex/scripts/bench_preflight.sh` — the PROVENANCE line (power, load, box state).
- `katgpt-rs/scripts/sequential_ab_timing_audit.py` — the interleaved-A/B law.
- Wilson/Beta LCB precedents — katgpt-core `beta_lcb_order_into`, riir-instinct `src/stats.rs`.
- `katgpt-rs/scripts/claim_lease_check.py --lint <file>` — the claim-line grammar, lint first.
- seal-e2e lanes — `seal-e2e-multi` §2/§4 owns the rig queue posture and the stress axes.
