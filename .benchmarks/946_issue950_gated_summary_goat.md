# Bench 946 — Issue 950 T7 GOAT: InfiLoop gated N/D running summary (modelless extraction, arXiv:2610.11570)

**Status:** COMPLETE — ALL GATES PASS (2026-10-19, M3 Max). Kernel + tests landed opt-in (`gated_summary`); no promotion claim (consumers unwired — riir-train Plan 465 is the training-track consumer).

## What was measured

The falsifiable question: does the full InfiLoop posture (β-decay + content-gated
writes) hold a running summary through adversarial overwrites AND re-converge
after a truth step, where the shipped incumbent (`PrecisionVector` posture:
β=1, fixed w, no gate) and the naive externals (leaky EMA, carry-last) cannot
do both?

Harness (all seeded, deterministic, one frozen read; D=8, 2 620-write
overwrite phase + 300-write adaptation phase):

- **Stream**: one smooth per-dim sine trajectory drawn once from the seed;
  Fig. 5 injection noise ε ∈ [1e−4, 0.3] on clean writes.
- **Overwrite events**: 6 × (4 adversarial writes `z_adv = −2·z_true` + 40
  recovery) — sized to the paper's per-write perturbation shape (see
  Protocol history).
- **Modelless scorer**: `s = 8·(0.5 − min(1, mean|z−h|))` — coherence with
  the current summary (clean ≈ +4, adversarial saturates −4); same scorer
  for every content-armed cell. A constant score is not a valid primary arm
  (paper §D: q=0 costs 6–8 pts).
- **Metrics**: paired end displacement (clean-twin vs injected, same arm);
  MID-BURST PEAK paired displacement (the retention readout — endpoint
  metrics let recovering arms hide burst damage); post-jump ADAPTATION
  ERROR (mean |h − new_truth| over the last 40 of 300 stepped writes — the
  plasticity readout); direction flips (report-only).

## Verdict table (the frozen read)

| arm | end-displ | mid-burst (retention) | adapt-err (plasticity) | flips |
|---|---|---|---|---|
| 1 PV posture (β=1, w=1) | 0.422520 | 0.083791 | 0.739111 | 0 |
| 2 decay only (β=.55, w=1) | 1.408253 | 2.338197 | 0.044850 | 27 |
| 3 content only (β=1, gate) | **0.020162** | **0.018580** | 1.073457 (worst) | 0 |
| 4 FULL (β=.55 + gate) | 1.404523 | 0.060030 | **0.035451 (best)** | 13 |
| x leaky EMA (α=.1) | 1.425276 | 0.893767 | 0.066331 | 15 |
| x carry-last | 1.427365 | 2.568163 | 0.082031 | 57 |

**The full posture is the only retention×adaptation Pareto arm** — the
stability–plasticity trade measured, not asserted:

- **G-RET PASS**: full mid-burst 0.060 < PV 0.084; = 0.067× leaky (≤ 0.15
  bound); = 0.023× carry-last (≤ 0.05); the decay plasticity tax is 3.2×
  content-only (≤ 4.0 bound); premise PV > 3× content-only holds at 4.5×
  (the gate axis is load-bearing).
- **G-ADAPT PASS**: full adaptation error 0.0355 = 0.048× content-only
  (1.073) and 0.048× PV (0.739) — 21–30× inside the 0.2 bound. β=1 arms
  hold unbounded mass, their write gain collapses, and they lag the
  stepped truth by ~the whole jump.
- **G-DELTA PASS**: the content gate owns retention (0.019 vs PV 0.084);
  the decay axis owns adaptation (decay-only 0.045 ≤ 0.2× PV 0.739).
- **G2 PASS**: ns/write 31.37 / 71.88 / 457.00 at D=8/64/512 (--release,
  bench profile; black_box chain defense — the score of write t+1 derives
  from the state after write t, so the loop cannot fold or vanish; no
  loud zeros).
- **G4 PASS**: 0 allocations over 10 000 content-scored writes
  (CountingAllocator, bench_013/945 pattern).
- **G3 PASS**: additive-only landing (`PrecisionVector` untouched) + the
  P4 oracle — β=1 + fixed w=1 reproduces `PrecisionVector`'s mean AND
  precision trajectory bit-for-bit over the seeded 512-step unit test
  (and the 64-step bench re-assert). A tolerance match would have hidden
  an ordering divergence; the oracle is bit-for-bit by design.

## Protocol history (disclosed — two revisions, both recorded)

1. **First run (32-write starvation bursts)**: the full posture FAILED its
   own battery — β=0.55's forget horizon. Rejected writes still decay the
   mass to the rejection floor `e_adv/(1−β)` in ~10 writes, after which
   the gate softens (gain → 0.45) and the burst drags the summary (full
   flipped 24× where content-only held at 0). SCOPE FINDING recorded: the
   decay arm trades overwrite-resistance for adaptation under sustained
   starvation; consumers must size their rejection stretches to
   `(1−β)`-class horizons or slow β.
2. **Second run (4-write events, endpoint-only metrics)**: burst damage
   was invisible at the endpoint (leaky recovers 98.5% over 40 recovery
   writes; carry-last's endpoint is just the last noise) and the protocol
   was retention-only — the decay axis's gain was structurally unobservable
   on a stationary trajectory (full was a pure plasticity tax there).
   Mid-burst-peak and the adaptation phase were added.
3. **Final run (two-axis, the design above)**: gates pinned from the second
   run's table + gain arithmetic BEFORE this run; every gate passed on the
   first two-axis read with 20–50% margin. No post-hoc gate adjustment.

## Scope findings (mechanism-level, recorded for consumers)

- **Decay-only is no defense**: fixed-weight decayed mass equilibrates at
  `w/(1−β) ≈ 2.2`, so every write — adversarial included — lands at gain
  ~0.31 (mid-burst 2.34). The gate axis is what rejects; decay alone only
  forgets.
- **β=1 (content-only) is the retention champion and the adaptation
  worst** (1.073 — worse than PV's 0.739): unbounded mass eventually
  refuses even the new truth. The learned-`ln λ` training track (Plan 465)
  exists precisely to tune this trade per task.
- The flip counter measures trajectory-phase misalignment as much as
  damage (arms with different effective gains cross zero at different
  ticks) — reported, never gated.

## Box state (provenance)

- date: 2026-10-19T18:20Z; box: Mac15,8 (M3 Max), macOS 26.6.2, 16 cores
- power: AC Power (charged); load averages 5.82 4.89 4.44; memory 44% free
- commit: 3cf587e0f924 (the claim-lease commit; kernel + bench in the
  landing commit), profile: bench (optimized), toolchain pinned 1.98.1
- Reproduce: `cargo bench -p katgpt-rs --bench bench_946_infiloop_gated_summary_goat --features gated_summary`
