# Issue 986 — Deterministic interior slot placement + effective conditioning strength (the Issue 985 negative's hypothesis test)

**Status:** OPEN — preregistered follow-up, filed from Issue 985's MEASURED NEGATIVE (delegated verdict AGREE, session `a842c972-dfca-4189-9108-244817f2d5b3`, 2026-11-06). Claimed: nobody yet. NOT an amendment to 985 — 985 is closed; this tests its two recorded hypotheses on a NEW preregistration.

## The hypotheses (labeled as such — neither is tested yet)

985's cell measured two failure signatures on the modelless soft read:

1. **Placement (the 985 record's hypothesis):** farthest-point anchors maximize min-distance → they spread to the store's OUTSKIRTS. The interior rows a query needs (multi-hop chains' middle hops) get no anchors. HYPOTHESIS: density-aware deterministic placement (seeded k-means++ seeds / k-medoids / query-density-weighted anchors) puts slots where reads need them.
2. **Conditioning strength (the narrowed run's measured signature):** head-alignment cos(q_k, q) = **+0.997 on every store** — the β-residual update moves each head by ~0.3%, all 16 heads are effectively the SAME head, unique-candidate count = **1.0** (16 slots, ONE distinct pick). A 1-candidate read loses to 16-candidate hard top-16 by construction, whatever the anchors are. HYPOTHESIS: scaling the residual to an EFFECTIVE magnitude (β_k·⟨q,d_k⟩ normalized to a per-slot target displacement, e.g. cos(q_k, q) ≈ 0.85-0.95) is what makes 16 heads into 16 different readers.

985's own verdict note: the expected max of 16 random cosines at D=64 is ≈ 0.22, and hard16-under-noise measured 0.287-0.304 — materially above chance, i.e. even "random" stores give aligned selection a lift; the deep level (c=4) barely discriminates for ANY method (hard16 real 0.323 vs noise 0.304).

## The preregistered cell (one run, no tuning passes)

Fixture: the Bench 985 composition fixture UNCHANGED (chain decay 0.5, 18 distractors/chain, N=512, K=16, τ=0.1, b=−2, 4 reps, seeds as committed). The bar and the arms are the 985 bar verbatim:

- **Bar:** soft16 ≥ hard16 at every level AND > empty (0), with Issue 984's arms (real ≥ noise + 0.05, noise ≤ empty + 0.05, shuffled ≤ real − 0.05) at the deepest level.
- **Arm A (placement):** k-means++-seeded anchors (seeded farthest-first over a k-means++ init — density-aware, deterministic given the store digest) replacing FPS, conditioning strength UNCHANGED (β·⟨q,d⟩ as shipped).
- **Arm B (effective conditioning):** FPS anchors unchanged, per-slot residual normalized to a TARGET displacement: q_k = normalize(q + s·r_k·⟨q,d_k⟩·d_k / ‖⟨q,d_k⟩·d_k‖) with the scale s set so mean cos(q_k, q) lands ≈ 0.90 (preregistered: exactly 0.90; not swept).
- **Arm C (both).**
- **Mechanism-liveness gate (NEW, from the 985 finding):** mean cos(q_k, q) ≤ 0.95 across slots is a PRECONDITION for gating any arm — an arm whose heads don't differ cannot test placement, only re-measure 985's 1-candidate collapse.
- **Reported per arm:** soft16 per level, unique-candidate count (must be ≥ K/2 for the arm to be interpretable), head-alignment spread, hard16 reference (unchanged code path), Issue 984 arms at the deepest level.

## Bound laws (inherited from 985)

- Deterministic everything (BLAKE3(store)-keyed anchors; seeded streams; replay bit-identity).
- Zero-alloc steady state; scratch-buffer API; const-generic K.
- LOCAL-ONLY aggregates (Issue 940 comment at the code site).
- Modelless: no learned weights, no gradient descent — placement and scaling are CONSTRUCTED per store epoch, never trained.
- One run per arm; a losing arm is recorded and NOT re-tuned (the 985 discipline).

## Why this is worth one cell

The 985 verdict endorsed filing this as the clean hypothesis test ("it tests the hypothesis cleanly and doesn't reopen 985"). If Arm C clears the bar, the modelless floor lane re-opens with a real mechanism (and 985's negative gets a footnote: placement+strength, not the read itself). If all arms lose, the "query-conditioned aggregate for non-LLM consumers" axis closes at the FLOOR level for good — the trained lane (riir-train Plan 488 R1) proceeds regardless (its bar is hard top-16, the standing champion).

## Tasks

- [ ] T1: k-means++ seeded anchor constructor in `slot_bank.rs` (SlotMode::KMeansPlusPlus), digest-keyed, coverage-entropy gate
- [ ] T2: effective-conditioning residual (target-displacement scale) in `soft_read.rs` (posture knob, default unchanged)
- [ ] T3: the three-arm cell + mechanism-liveness gate as `bench_986_slot_placement_goat.rs` (report posture per the 985 precedent: gates print, exit 0, ONE live assert on fixture validity)
- [ ] T4: verdict recorded here + Research 656 (re-open footnote or the axis-closes-at-floor coda)
