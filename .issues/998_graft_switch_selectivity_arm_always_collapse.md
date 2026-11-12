# Issue 998: graft switch selectivity — the arm-always collapse and the two fix lanes

**Status:** OPEN — filed from issue 988 T4's measured G3(b) FAIL (the GOAT bench `bench_988_grafts_goat`, frozen-read replay posture).

**Claimed:** amo-i998-t1@amo, 1794492300, expires 2026-11-14T10:00:00Z

- [ ] **T1** — diagnose + document the collapse precisely: the fitted switch's per-row scores over the frozen corpus (export already carries the crosstab — zero disarmed rows in 600); confirm the mechanism (LS against labels whose conditional expectation given the input state is ≈ the base rate 0.92 collapses to the constant predictor — the state at the operand positions does not encode whether the model's ARGMAX will be right; that is a decode-time property). Record as a short `.research/` note or fold into this issue's resolution.
- [ ] **T2** — lane A (model-side): a TASK-DETECTOR switch — refit the switch with labels = task-row vs non-task-row, which requires a non-task activation surface in the fit (a small riir-infer cell addition: capture rows from off-task prompts; the export bin gains the non-task cache input). GOAT: the same bench G3(b) bars (KL ≤ 0.02 nats, flips ≤ 0.5%) on the same base-right corpus + a NEW off-task corpus leg.
- [ ] **T3** — lane B (deployment law, no model work): the consumer mounts the graft ONLY inside a task-detected context (e.g. riir-infer's serving lane mounts `grafts` artifacts solely for arithmetic-lane prompts). Zero katgpt-core changes; the selectivity contract moves to the mount site. Document the law in the graft module doc + the artifact doc.
- [ ] **T4** — whichever lane passes its GOAT, re-run `bench_988_grafts_goat` and record the promotion verdict under the Feature Flag Discipline (the feature's default posture flips only on a full GOAT PASS).

## Context (the measured facts)

The T2b cell's switch (`fit_switch`, labels = {base_wrong} over the solved split, threshold 0.5) armed on EVERY captured row — solved right+armed 31 / wrong+armed 369; dev right+armed 17 / wrong+armed 183 (the export crosstab; zero disarmed anywhere). Consequence, measured by the GOAT bench (2026-11-12): on the 48 base-right rows (traffic the base already answers) the armed graft fires 31× at the pinned write delta 8.0 → mean KL(top-40) 2.71 nats vs the pre-registered 0.02 bar and top-1 flip rate 64.6% vs the 0.5% bar → G3(b) FAIL → `grafts` NOT promoted (stays opt-in). Conditional == unconditional fire sets — the switch contributes no selectivity.

**Why the fit collapsed:** `base_wrong` is an output-space property (what the greedy argmax will do), weakly encoded in the input state at the capture position; the LS fit learned the prior (92% of labels are 1) because no separating direction exists in the state. The primitive itself (reads → table → writes) is untouched by this — G1/G2/G3(a)/G4 all pass; only the arming gate's SELECTIVITY failed, and only on the no-regression axis.

## Non-goals

- No re-fitting against sealed (the cell law holds).
- No removal of the switch mechanism — the seam is correct; the LABEL was wrong for selectivity.
- No softmax anywhere (sigmoid gates stay sigmoid).
