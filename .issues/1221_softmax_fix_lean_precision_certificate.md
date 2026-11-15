# Issue 1221 — Lean precision certificate poc for the softmax_fix family (RP predicate + one sign/mixed-bound theorem)

**Status:** OPEN — poc/proof task (arXiv:2610.31642 companion arc, Research 673 §2/§5; the `request_verdict` review of Research 673 flagged the proof-path scope as the weakest point — this issue IS the runnable proof path)

- [ ] Scope the theorem before any Lean work: candidate A (sign theorem, cheapest) — `fixed_log_softmax_target`'s target log-prob is finite, negative, and nonzero whenever the target logit is maximal and at least one rival logit is finite (any IEEE format modeled in FloatLib-style rationals; the f32/f64 instantiations are `decide`-checkable concrete instances). Candidate B (mixed bound, harder) — `|computed − exact| ≤ r·|exact| + a` for the argmax-excluded-LSE + softplus form under the repo's pinned fold (first-index argmax + sequential ascending-j, Issue 940). Pick A for the poc; B is the follow-up only if A lands clean.
- [ ] Model the format the repo way: a simplified binary-float → ℝ model (`Float e m` + `toReal` + `round`) suffices — the companion arc proves the softmax certificate with exactly this shape (MiniFloat 4/3, `Real.log (8/7)` roundRP); no need to formalize full IEEE 754.
- [ ] Land in `.proofs/KatgptProof/` per the Plan-293/441 conventions: new spec module + `SpecTests.lean` concrete instances + `PrintAxioms` line + the axiom-budget bump in the gate script (no `sorry`; `[propext, Classical.choice, Quot.sound]` only).
- [ ] Paired Rust spec-match test proving the Rust matches the Lean spec on finite samples (the repo's convention — mirror `hla_bounds_spec_match`-class pairing): walk the Issue-959 G1 ladder (0..103 nats) asserting the theorem's predicate holds in Rust where the Lean proof says it must.
- [ ] GOAT gate: the certificate must add a claim the f64-oracle G1 test cannot make — i.e. it covers an unbounded input region (all finite logits in the pinned posture), not the sampled ladder. If it only re-asserts sampled points, it fails the poc and the lane closes NEGATIVE with the reason recorded.
- [ ] Close-out: note + ref the commit hash in Research 673 §5, remove this file.

Non-goals: no FloatLib dependency, no cross-port of Lean files from private `.proofs/` instances (coordinator rule C4), no change to the shipped `softmax_fix` Rust (the theorem rides above it).
