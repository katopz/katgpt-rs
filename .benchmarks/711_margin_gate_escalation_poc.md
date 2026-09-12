# Plan 595 — Margin-Gated Verification Escalation PoC (Bench 711)

**Date:** 2026-09-11
**Bench:** `crates/katgpt-core/benches/bench_711_margin_gate_escalation_poc.rs` (`harness = false`, release)
**Issue:** `.issues/745_margin_gate_escalation_poc.md`
**Plan:** [`.plans/595_margin_gate_escalation_poc.md`](../.plans/595_margin_gate_escalation_poc.md)
**Research:** [`.research/548_TriSpec_Margin_Gated_Verification_Escalation.md`](../.research/548_TriSpec_Margin_Gated_Verification_Escalation.md)
**Source paper:** [TriSpec (arXiv:2601.23180)](https://arxiv.org/abs/2601.23180) — Qwen Team, Feb 2026
**Feature:** `katgpt-core/margin_gate` (implies `indicator_cascade`) — **stays OPT-IN**
**Verdict: ALL 9 GATES PASS — SPLIT verdict.** The **cascade mapping is REFUTED** at ε=0.5% (both toy geometries, both polarities); the **accept mapping is VIABLE** — 50.5% target-invocation cut at 0.25% regression (tail 0.5%) and genuinely ε-sensitive (the control fires at tail 2%). Mechanism gates green (5–7ns operator, ≤1.0× run ratio, 0 allocs). The bandit-λ arm produced two standalone findings: the meta-router reward shape is blind to the lossy tail, and a point-estimate feasibility mask starves good arms.

---

## Substrate correction (recorded against Research 548 §2)

Research 548 claimed "the margin operator is ABSENT workspace-wide" (grepped
`top1_share`/`top2`/`rank_gap`). That grep missed the shipped substrate by
vocabulary: `SamplerFeatures::from_logits`
(`katgpt-forward/src/diffusion_sampler.rs`) has carried `margin`
(top1_prob − top2_prob) since the Plan 399 relocation era (2026-07-05). What
was genuinely absent is the **gate predicate + cascade fusion**. The
vocabulary-translation rule (grep 3+ name variants) is the defense — a
`margin`-vocabulary grep would have found it.

## What shipped

- `pruners/margin_gate.rs` (feature `margin_gate`, default-off):
  - `margin_split(&[f32]) -> MarginSplit { top1, top2, gap, argmax }` —
    zero-alloc two-running-maxima pass, branchless body (maxss/minss + one
    select), stable lowest-index tie-break matching `or_fused_fire`.
  - `MarginPolarity { DecisiveTrusted, CoherentTrusted }` — the trust sign is
    an explicit workload-geometry choice, not a constant.
  - `MarginGate { lambda, polarity }` + `MarginDecision { Clean, Trusted, Escalated }`
    + `MarginGatedCascade` — graded stage-1 firing predicate over the Plan 320
    cascade; flagged-only semantics preserved; escalation re-uses
    `speculative::sample_residual_distribution_into` (Leviathan Eq. 3 — the
    exact substrate `katgpt-forward::d2f_verifier` + `step.rs` consume).
  - 12 unit tests incl. the T1 property tests (uniform → gap 0; one-hot →
    gap = 1 − p₂; sort-reference cross-check with ties).
- `[[bench]] bench_711_margin_gate_escalation_poc` (required-features:
  `margin_gate, indicator_cascade`).

## The defend-wrong matrix (three worlds)

| World | Mapping | Genuine event | Wrong shape | Correct polarity |
|---|---|---|---|---|
| A:cluster | cascade (margin over 8 sigmoid probe scores) | correlated indicator PAIR | lone decisive spike | CoherentTrusted |
| B:decisive | cascade | single decisive spike | ambiguous co-fire | DecisiveTrusted |
| C:accept | draft-accept (margin over V=32 token distribution) | peaked draft (gap ≥ ~0.52) | ambiguous draft (gap ≤ ~0.45); a `tail_rate` fraction confidently wrong | DecisiveTrusted |

## Gate results (ε = 0.005, min qualifying cut = 20%, 40k trials/world, release, M3 Max)

| Gate | Contract | Result | Verdict |
|---|---|---|---|
| G1a (world A) | NO (polarity, λ) cell reaches cut ≥ 20% at regression ≤ ε — refutation holds | best attempt: λ=0.1 coherent, cut +77.5% at regression 0.033 (6.6× over ε) | ✅ PASS (refuted) |
| G1a (world B) | same | best attempt: λ=0.3 decisive, cut +22.4% at regression 0.080 (16× over ε) | ✅ PASS (refuted) |
| G1b (world C, tail 0.5%) | ∃ λ qualifying | λ=0.5: regression 0.00248 ≤ ε at cut **+50.5%** | ✅ PASS |
| G3 (world C) | the qualifying cut is real | target re-decide invocations −50.5% at λ=0.5 | ✅ PASS |
| G1c (world C, tail 2%) | the control MUST fire: no λ qualifies at 2× TriSpec's ≤1% ceiling | no λ qualifies (best regression 1.05%) | ✅ PASS (control fires) |
| G2 | operator latency + run ratio | `margin_split` 5–7 ns @ N=8; gated run 84 ns vs baseline 83–167 ns (ratio 0.50–1.00×) | ✅ PASS |
| G4 | alloc-free | 0 allocs / 100 mixed gated runs | ✅ PASS |
| T4b | UCB1 vs fixed λ ∈ {0.3, 0.5, 0.7} (world B, paired) | paired per-seed diff −404.9 (−0.74% of best) | ✅ PASS |
| T4c | ε-feasibility-masked UCB1 (world C) | pulls [400, 39241, 358]; trusted-wrong rate 0.448% ≤ ε; paired diff −0.19% | ✅ PASS |

Run:

```bash
cargo bench -p katgpt-core --features margin_gate \
  --bench bench_711_margin_gate_escalation_poc -- --nocapture
```

## Findings (each load-bearing, each measured)

1. **The trust polarity is workload geometry, not a constant.** TriSpec's
   `gap ≥ λ` decisive-trust, deployed on the cascade's cluster world,
   auto-confirms exactly the lone-spike false positives (FPR 0.033 at λ=0.1,
   6.6× the correct polarity's) — the stage-2 verifier exists to reject those.
   The mirror fails identically on the decisive world. `MarginPolarity` ships
   both arms; choosing wrong is a measured FPR explosion, not a style slip.
2. **The cascade mapping is refuted at ε=0.5%.** In the saturated sigmoid
   score domain (8 probe scores, noise floor ~N(0,2) raw), the trusted vs
   escalated gap distributions overlap: false two-spike co-fires compress to
   gap ≈ 1e-4 — indistinguishable from true clusters — so every λ that cuts
   invocations also confirms a 3–19% false-flag tail. Best honest cells:
   world A 77.5% cut at 3.3% regression; world B 22.4% cut at 8.0% regression.
   The margin signal needs a PEAKED score domain (real token distributions),
   not saturated probe scores over a noise floor.
3. **The accept mapping is viable and the ε control genuinely fires.** World C
   (V=32, peaked corrects / ambiguous wrongs / explicit confidently-wrong
   tail): λ=0.5 halves target invocations at 0.25% regression; doubling the
   tail to 2% breaks every λ. **The gate inherits the world's tail rate —
   the deploy rule is: measure the peaked-wrong rate first; deploy only if
   ≤ ε.**
4. **The meta-router reward is blind to the lossy tail.** Mapped reward =
   correct × (1 + verifier_saved): a trusted wrong and an escalated wrong both
   score 0, so the ε-violating λ=0.3 TIES the best arm (39976.1 = 39976.1)
   on raw reward. Any prod bandit governing a lossy gate needs the constraint
   as a mask, not in the reward.
5. **A point-estimate feasibility mask starves good arms.** First T4c run: the
   0.25%-tail arm drew 2 wrongs in 196 pulls (1.02% > ε) → permanently masked
   → the bandit burned 42% of its rounds on the suboptimal λ=0.7 (−8.8%
   reward). Fix: one-sided z=3 confidence bound (condemn only when wrongs >
   ε·n + 3√(nε(1−ε))) — the violating arm (19% tail) still crosses within the
   grace window; the bandit then converges (pulls [400, 39241, 358], −0.19%).

## Honest caveats

- Toy distributions by construction; world C's gap separation is planted.
  The real deploy precondition is the measured gap-AUROC + tail rate of an
  actual draft distribution (world C is the measurement harness shape).
- Oracle verifier/stage-2: a real verifier's own noise strictly SHRINKS the
  viable region — the refutation half is conservative, the viability half is
  optimistic by exactly the verifier's error rate.
- ε=0.5% is half of TriSpec's published ≤1% lossy ceiling; a different owner ε
  moves the λ table, not the structure of the verdict.
- The gated run's 0.50× ratio vs baseline is real (trusted fires skip the
  trait-object verifier dispatch), not a timing artifact — both directions
  were observed across runs (0.50–1.00×); the gate asserts ≤1.5× either way.

## Promotion decision (T5): NOT PROMOTED — stays opt-in

`margin_gate` remains default-off. The `similarity_inference` demotion
precedent governs: 24 days default-on with zero consumers ended in demotion —
a default promotion on toy-only evidence would repeat it. The candidate live
consumer is the d2f draft-accept loop (`katgpt-forward::d2f_verifier` /
`step.rs`), which is an engine-lane owner call (it changes the perf-league
comparison axis; Research 548 §5). Until a consumer lands and re-gates on real
traces, the feature ships as measured: mechanism green, cascade refuted,
accept viable-at-low-tail.
