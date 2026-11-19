# Issue 1235: ACE fusion rows — adaptive steering calibration + core-space merge diagnostic

**Status:** OPEN — fusion ideas from Research 683 (arXiv:2602.22600); novelty TBD; no burning consumer; claimable. **2026-11-19 (katgpt-rs-57@m3):** the producer landed (Plan 640 DONE, `relevance_core`, Bench 988), so ACE cores are now available to (a)/(b). Both rows remain CONSUMER-GATED by their own text: (a) needs a consumer to name its decision margin, and (b) needs a failing or contested merge lane plus two checkpoints with known outcomes. Routed: steering-consumer owners (riir-ai `latent_steering`/`subspace_steering` users) name a margin, which unblocks (a). Not claimed; (c) stays record-only.

Two self-adaptive fusion rows from the ACE distillation that have named consumers-to-be but no burning need. File-level gating per the ungoat law: each row lands only with its own proof path.

## (a) Adaptive steering-strength calibration (`s* = (m_target − m0)/g`)

The steering consumers ship (`latent_steering` Plan 309, `subspace_steering` Plan 412, `region_subspace` Plan 416) and consume discovered bases with hand-set α. ACE's generation intervention adds a closed-form controller (arXiv:2602.22600 App. D): a task-mass GATING check (intervene only when the relevant token mass crosses a threshold), a local-gain PROBE (`g = (m1 − m0)/s0` from one small perturbation), and the strength `s* = (m_target − m0)/g` targeting a minimal margin crossing, capped. Lanes: the margin task for OUR consumers is decision-side (a scored decision's margin), not LM generation — the transfer needs a margin definition per consumer, which is why this stays an idea until a consumer names its margin.

**Proof path when claimed:** extend the `subspace_steering` gates with a two-arm test (hand-α vs calibrated s*) on a planted-margin fixture; bar: calibrated arm hits m_target within tolerance at ≤ the hand-α arm's overshoot, determinism G0.

## (b) Core-space merge-compatibility diagnostic

Research 683 §5: the paper's merge discussion (cores across seeds live in near-orthogonal subspaces → weight interpolation leaves the solution manifold) + our `gauge_invariant.rs` adapter compose suggests a diagnostic: merge compatibility measured as core-subspace agreement (`basis_similarity` on ACE/probe cores of the two models) BEFORE composing adapters. Idea-class: no merge lane is actively failing today; gauge_invariant already removes the scale gauge, and this would add a rotation-gauge read.

**Proof path when claimed:** a report-only example over two adapter-trained checkpoints with known merge outcomes (one good, one bad); the diagnostic must separate them.

## (c) Record-only: operator identification in core coordinates

Fit `z_{t+1} ≈ A z_t` in core coords + spectrum report (the paper's "Identify" half). Adjacent to `KarcForecaster::fit_ridge`; no consumer needs it yet. Recorded so the idea does not re-file; do not implement without a consumer.
