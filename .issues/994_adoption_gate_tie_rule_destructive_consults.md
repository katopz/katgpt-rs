# Issue 994 — adoption_gate tie rule on destructive consults (+ the α=1 endpoint degeneracy; + optional digest-keyed holdout)

**Status:** OPEN — filed from the riir-refine G3 real-store cell (riir-refine Bench 148, 2026-11-10; consumer issue 220 closed, plan 633 T3 executed). Not claimed; the design decision below is pre-registered for the NEXT cell, never patched into the executed one.
**Claimed:** zed-m3-i994@m3, 1794347251, expires 2026-11-12T21:47:00Z — the idle-loop issue-first unit: decide + land the tie rule (Option A direction, pending the delegated-owner verdict), fixtures + module-doc rule name + pre-registered bars for the next G3 re-run; the push is the arbiter.

## The finding (measured, riir-refine Bench 148)

`AdoptionGate::decide`'s tie rule (`TIE_GOES_TO_CHALLENGER`, landed `cf65c58a7`) is wrong ON DESTRUCTIVE CONSULTS — not only at the α=1 endpoint where the cell's G3-b bar failed, but at the DEFAULT α:

1. **The α=1 endpoint degeneracy (the measured G3-b FAIL).** On zero-rejection windows, `score ≡ 1.0` on both sides at α=1 (retention is weighted zero; neither side repeats anything) → exact tie → tie→challenger ADOPTS → demoting the ONLY productive rule in a clean window proceeds. Measured: 5 α-flips across 34 real consults, ALL this one class (riir-dao ×2, riir-auth ×2, riir-grams ×1), confirmed empirically at `--alpha 1`.
2. **The default-α tie PROCEED (the delegated owner's widening — the sharper half).** An exact tie at ANY α is a NO-EVIDENCE state: the window cannot tell the challenger from the incumbent. NoEvidence already means keep-the-incumbent (decline-first). The tie rule contradicts that: measured at α=0.5, riir-instinct demote `hashmap-with-capacity` PROCEEDED on an exact tie (0.97059 = 0.97059), demoting a rule that carries an accepted fix **on zero evidence of harm**. Same conflict, default posture.

## The decision to make (then pre-register for the next cell)

- **Option A (the house-shaped default): tie → incumbent on destructive consults.** An uninformative window declines. This aligns the tie rule with decline-first and NoEvidence semantics; `TIE_GOES_TO_CHALLENGER` may stay for ADOPTION-shaped candidates if any exist, or flip globally — the fixture suite pins whichever is chosen.
- **Option B: guard only the endpoint** (α=1 refuse on zero-rejection windows) — narrower, keeps tie→challenger at other α, leaves the default-α tie PROCEED unfixed. Weaker; recorded so the choice is deliberate.

Whichever is chosen lands with: updated fixture pins (the pure-scorer fixtures), a name for the chosen rule in the module docs, and a PRE-REGISTERED bar set for the next G3 re-run (the consumer lane re-runs the cell after this lands — never mid-cell edits to the executed posture).

## Optional hardening (same landing or a follow-up)

**Digest-keyed holdout:** the decontamination holdout keys on the accepted-count POSITION (every 5th NEW accepted outcome). Flipping any accepted verdict shifts the count sequence, so holdout membership moves for downstream rows — measured in the cell's leave-one-out arm (an audit-filed rule's readmit verdict flipped when ANY upstream accepted row flipped). Never-double-file still holds per replay; this is noise-sensitivity, not a decontamination defect. Keying by `digest mod 5` (BLAKE3 output) makes membership verdict-flip-invariant. Changes the fixture pins + the `Decontaminator` seam; zero deps.

## Refs

- riir-refine `.benchmarks/148_issue220_g3_real_store_cell.md` (the cell, the numbers, the delegated-owner REVISE verdict) · riir-refine issue 220 (closed; `git log --follow`) · katgpt-rs `.plans/633_adoption_retention_regression_gate.md` T3 · issue 987 priority-1 (AGCL, Research 657 §A5)
- Producer: `crates/katgpt-core/src/adoption_gate.rs` (feature `adoption_gate`, landed `cf65c58a7`)
