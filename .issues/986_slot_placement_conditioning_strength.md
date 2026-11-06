# Issue 986 — Deterministic interior slot placement + effective conditioning strength (the Issue 985 negative's hypothesis test)

**Status:** CLOSED NEGATIVE (2026-11-06, one cell per the preregistration, no tuning passes) — both hypotheses FALSIFIED on the committed fixture; the query-conditioned aggregate axis closes at the floor level. Landed: `slot_bank` `SlotMode::KMeansPlusPlus` + `soft_read` `with_head_cos_target` (both opt-in, kept for replication) + `benches/bench_986_slot_placement_goat.rs` (report posture, ONE live assert). **Lease closed:** zed-488-memorilla@m3, epoch=1793940783, expires 2026-11-08T04:53:03+00:00. NOT an amendment to 985 — 985 is closed; this tested its two recorded hypotheses on a NEW preregistration.

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

## The measured cell (2026-11-06, bench_986, `--no-default-features --features soft_read,distance_abstain`)

Fixture validity: hard16 collapse OK (c=1 1.0000 → c=4 0.3230, ≥0.1 drop).

G1 — soft16 per level (best-gated-row cosine; the bar is soft16 ≥ hard16 AND > 0):

| level | hard16 | 985-FPS (ref) | A k-means++ | B conditioning | C both |
|---|---|---|---|---|---|
| c=1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| c=2 | 0.3983 | 0.3770 | 0.3770 | 0.3770 | 0.3770 |
| c=3 | 0.2271 | 0.1941 | 0.1941 | 0.1941 | 0.1941 |
| c=4 | 0.3230 | 0.2415 | 0.2415 | 0.2415 | 0.2415 |

G2 — mechanism liveness (the preregistered precondition): **all three arms NOT INTERPRETABLE** —

| arm | head-cos | unique candidates | entropy |
|---|---|---|---|
| A placement | +0.9975 | 1.0 | 0.973 |
| B conditioning | **+0.9305** | **1.0** | 0.984 |
| C both | +0.9288 | 1.0 | 0.973 |
| 985-FPS shipped (ref) | +0.9970 | 1.0 | 0.984 |

G4 — Issue 984 store arms at c=4: every arm reads noise (0.288-0.295) ABOVE real (0.2415) — I1 FAIL, I2 FAIL on all arms; I3 OK (shuffled 0.103-0.185 ≤ real − margin).

## The verdict (why both hypotheses are falsified, not just unconfirmed)

- **Placement (A) is falsified as a sufficient fix, and metrically INERT here:** the k-means++ bank is genuinely different (coverage entropy 0.973 vs FPS 0.984) yet every soft16 number is byte-identical to the shipped 985 posture. The mechanism: with the shipped conditioning, cos(q_k,q) ≈ 0.998 regardless of where the anchor sits — the heads are all still `q`, so ALL anchor placements produce the same one candidate. Placement cannot matter until the heads differ; the two hypotheses are ORDERED, not independent, and 986's cell shows the ordering arrow points from conditioning to placement.
- **Conditioning (B) is falsified at the preregistered strength, and the liveness gate caught the deeper truth:** the knob works mechanically (head-cos 0.997 → 0.930 — a ~7° tilt per head) but unique candidates stayed **1.0**. A 7° tilt does not move the top-gate argmax off the globally-best-aligned row: 16 heads still read ONE candidate. Spreading the heads enough to pick 8+ distinct rows would need tilts that rotate the head past the top row toward hops-2..4 — at which point the per-head cosine to those hops (≤0.5 by construction) still gates them BELOW the distractor cloud under b=−2, τ=0.1 — the 985 finding that no deterministic projection reaches, now measured at two conditioning strengths.
- **The noise-store inversion is the decisive read:** garbage (0.29) beats real (0.24) on every arm. The real store's best-aligned row IS the distractor cloud (cosine 0.24); a noise store has no distractor structure, so the shared head's pick is a pure chance row at E[cos] ≈ 0.29. The read is not merely evidence-insensitive — at this depth the store's own structure is what POISONS the pick. This is the strong form of the 985 negative.

**Axis disposition (the preregistered coda):** the "query-conditioned aggregate for non-LLM consumers" axis CLOSES at the modelless floor level — for good (two preregistered cells, 985 + 986; any re-open now needs NEW information, e.g. a consumer whose queries differ structurally from hop-1 mentions). The trained lane (riir-train Plan 488 R1) proceeds regardless — its bar is hard top-16, the standing champion on this fixture class. The two posture knobs (`SlotMode::KMeansPlusPlus`, `SoftRead::with_head_cos_target`) stay opt-in behind `soft_read`, MEASURED NEGATIVE-labeled, for replication.

## Tasks

- [x] T1: k-means++ seeded anchor constructor in `slot_bank.rs` (`SlotMode::KMeansPlusPlus`), digest-keyed (BLAKE3-derived first index + SplitMix64 stream), coverage-entropy gate read per arm in the bench
- [x] T2: effective-conditioning residual (target-displacement scale) in `soft_read.rs` (`with_head_cos_target(0.90)`, posture knob, default byte-identical; small-arc solve s = √(2(1−t)/mean β²))
- [x] T3: the three-arm cell + mechanism-liveness gate as `bench_986_slot_placement_goat.rs` (report posture per the 985 precedent: gates print, exit 0, ONE live assert on fixture validity — the assert FIRED PASS: collapse exists)
- [x] T4: verdict recorded here + Research 656 (the axis-closes-at-floor coda)
