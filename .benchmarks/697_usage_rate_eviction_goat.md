# Bench 697 — Usage-Rate (Mass/Age) KV Eviction GOAT (Plan 585)

**Status: DONE (2026-09-02) — MIXED VERDICT, honest regime boundary; ADDENDUM (2026-09-06): null control landed — signal value CONFIRMED beyond protection (see §T3.6/T3.7 Addendum).** The primitive
wins its designed regime decisively (2× raw-H2O recall at cap=32, 100% at cap≥48,
R_median 1.0 vs 8.0) and LOSES at the extreme-pressure point (cap=16, 8% budget,
11× turnover). **Stays opt-in** (`usage_rate_eviction = []`): no runtime consumer
(riir-ai Issue 836 is pull-gated on this GOAT), and the G8 sweep records one
genuine regime miss. Per the plan's own rule the miss does not demote the
primitive — it bounds the claim: mass/age is the right statistic when the budget
covers ≥ ~2× the churn horizon; at extreme pressure lifetime-mass luck dominates.

- **Plan:** [`.plans/585_usage_rate_eviction_primitive.md`](../.plans/585_usage_rate_eviction_primitive.md)
- **Research:** [`.research/523_H2O_Norm_Age_Normalized_KV_Eviction.md`](../.research/523_H2O_Norm_Age_Normalized_KV_Eviction.md) (arXiv:2608.19920 "Learning how to Forget", Seeger et al., AWS 2026)
- **Substrate:** `katgpt_core::kv_eviction` (feature `usage_rate_eviction`), `crates/katgpt-core/src/kv_eviction/mod.rs`
- **Harness:** `crates/katgpt-core/benches/bench_697_usage_rate_eviction_goat.rs` (harness=false, release)
- **Run:** `cargo test -p katgpt-core --features usage_rate_eviction --bench bench_697_usage_rate_eviction_goat --release -- --nocapture`
- **Box:** 4090/Windows (CPU-only bench; no GPU dependency — GPU lanes were sibling-active all session). Addendum run: M3 Max/macOS (2026-09-06).

## T3.6/T3.7 Addendum (2026-09-06) — the null control + protection factorial (Research 531)

Research 531 (arXiv:2609.03430 "Random Attention") supplied the control the T3.2
matrix lacked: prompt-pinned per-head uniform-random eviction — no score. Four
arms appended (six original cells bit-identical to the 2026-09-02 run — the
draw stream is DEDICATED, so the fixture is untouched; G1 covers the full
10-arm matrix): `rand` (null, unpinned), `rand_keystone` (the paper's headline
arm), `mass_age_keystone` + `ega_energy_keystone` (the ±pin factorial).
Keystone = needle row (the prompt-protection rule translated to this fixture).
32 seeds × 12 needles, M3 Max, release.

### Factorial table (recall, 384 = ceiling)

| policy | pin | cap=16 | cap=32 | cap=48 | cap=64 | keys@16 | evict@16 |
|---|---|---|---|---|---|---|---|
| mass_age | − | 0.0% | 50.0% | 100.0% | 100.0% | 0/384 | 5632 |
| mass_age | + | 100.0% | 100.0% | 100.0% | 100.0% | 384/384 | 5632 |
| ega_energy | − | 17.4% | 29.9% | 41.7% | 56.0% | 67/384 | 5632 |
| ega_energy | + | 100.0% | 100.0% | 100.0% | 100.0% | 384/384 | 5632 |
| **rand (null)** | − | **1.8%** | **9.9%** | **20.8%** | **26.0%** | 7/384 | 5632 |
| rand | + | 100.0% | 100.0% | 100.0% | 100.0% | 384/384 | 5632 |

- **Pin-honored gate PASS** (all +pin arms 384/384 at every cap — the mask is
  load-bearing, verified on real traffic, not by construction only).
- **T3.6 non-vacuity — signal value CONFIRMED, the demote-the-loser branch did
  NOT fire:** the pre-registered passcode-regime prediction held (the unpinned
  null collapses at cap=16: 7/384 = floor-class), and mass_age strictly beats
  the null at every regime cap — **192 vs 38 (5.0×) at 32, 384 vs 80 (4.8×) at
  48, 384 vs 100 (3.8×) at 64**. The complexity of mass/age buys real
  needle-finding, not just protection.
- **The cap=16 inversion is now measured against the null too:** rand 7/384 vs
  mass_age 0/384 — at extreme pressure the null's geometric survival (the
  paper's implicit soft-recency window, `((K−ℓp)/(K+r−ℓp))^n`) retains a thin
  recent tail while mass/age's rate signal actively loses to it. The
  extreme-pressure miss list gains the null alongside raw-H2O.
- **Protection-alone ceiling is real on this fixture:** every arm pinned hits
  100% at every cap — the paper's Table-2 shape reproduced (protection is the
  dominant lever when the keystone oracle exists; the deltas −pin→+pin are the
  per-arm protection deficits: mass_age 50→100 @32, EGA 29.9→100, rand 9.9→100).
- **Keep-log note (recorded honestly):** in this fixture keystone-survival ≡
  recall (a surviving needle row IS a recalled needle — one row per needle
  token, payload/admission immutable), so the paper's keep-log statistic
  collapses onto the recall axis; the factorial table is the deliverable.
- **Instrument find (the canary + null together):** `rand` at cap=32 PASSES the
  runaway canary (R_median 1.000, p_cap 0.00, gen out-len 8.0) while recalling
  only 9.9% — the generation chain starts from the LAST needle (planted at 85%
  of the stream), which the null's soft-recency keeps. **The runaway canary
  measures budget health, not recall; the null control is the complementary
  instrument.** A policy can pass the canary and still be memory-holing the
  context.

### T3.8 — the round-cost axis + the registered alternative, now measured

- **CPU round cost:** the score update is 1.22 ns/row (4090 box) / 1.78 ns/row
  (M3, this run) — both ≥5× under the 10 ns budget. Scoring is free in CPU
  terms; the paper's +32–43% serving margin lives where scoring is a KERNEL
  pass over paged state (the T4.1 mass-byproduct kernel, riir-ai Issue 836 —
  our kernels return no column sums, note 523 §1.5).
- **The registered alternative is now a measured cell, not a strawman:** with
  the keystone oracle, `rand_keystone` = 100% recall at every cap with ZERO
  scoring work — on this workload, if deep-needle recall is not load-bearing
  AND the oracle exists, **skip the kernel and ship the null**. Without the
  oracle the null is not viable (1.8–26% unpinned): the decision reduces to
  whether the serving workload has a structural keystone set (prompt, system
  rows) and whether once-stated facts at depth matter.
- Standing state unchanged: `usage_rate_eviction` stays opt-in, no consumer;
  promotion re-gate = Issue 836 consumer + this matrix incl. the null control.

### T3.9 — the promotion-gate extension (landed)

`katgpt_core::kv_eviction::{PolicyControl, beats_random_prompt_pin}` — the
null-policy bar encoded beside `runaway_gate`: a scored policy keeps its slot
only if its recall STRICTLY exceeds the pinned-random null's at matched budget
(at equal recall the null wins on cost; ties hand the slot to the null);
non-finite recall fails closed. Protection matching is the caller's protocol
step (run the ±pin factorial first). Unit tests: strictness, NaN fail-closed,
negative control. **Standing promotion rule for ANY lossy KV policy is now:
`runaway_gate` ∧ `beats_random_prompt_pin` ∧ the protection factorial.**

### Addendum gates

| gate | verdict |
|---|---|
| G1 determinism (10-arm matrix double-run bit-identical) | PASS |
| Pin-honored (+pin arms 384/384 every cap) | PASS |
| T3.6 collapse prediction (null at floor @16) | CONFIRMED (7/384) |
| T3.6 signal value (mass_age > rand at 32/48/64) | **CONFIRMED** (5.0×/4.8×/3.8×) |
| G2 update latency (M3) | 1.78 ns/row PASS |
| Original six cells vs 2026-09-02 record | bit-identical (append-only) |

## Modelless construction (read before quoting numbers)

The "model" is a **constructed induction-pair KV cache** — the deterministic
abstraction of a trained induction head's cache (row j: key = token, value =
next-token payload). NO training, NO gradients, NO learned weights: the policy
axis (which rows a fixed policy keeps under budget) is what is under test, not
the model. The workload is a drifted Zipf token stream (the late hot set mirrors
the early one) with queries sampled over the tokens **currently live** in the
cache; each admission self-attends (every row carries positive mass from birth);
needles are **mid-frequency recurring tokens** (queried every 8 ticks) carrying a
unique payload. The eviction trim runs AFTER the tick's attention — the
post-attention state real serving decides on (trimming a zero-mass newborn makes
it the guaranteed victim of every mass-based policy; the harness refuses that
degeneracy — found and fixed during the bench's own bring-up, see §History).

## Results

### T3.1 planted age-bias fixture — PASS (must-fire)

| arm | raw-H2O | mass/age |
|---|---|---|
| tie (mass 1.0/1.0) | tie-indifferent (index tie-break) | strictly evicts old-cold ✓ |
| strict (old mass 1.1 > 1.0) | strictly evicts the young-hot row ✓ | strictly evicts old-cold ✓ |

### T3.2 recall at matched budget — 32 seeds × 12 needles, stream=192

| policy | cap=16 | cap=32 | cap=48 | cap=64 | gen out-len @16 |
|---|---|---|---|---|---|
| ring | 8.3% | 16.7% | 25.0% | 33.3% | 50.00 |
| raw_h2o | 6.2% | 24.2% | 41.4% | 51.3% | 46.50 |
| **mass_age** | 0.0% | **50.0%** | **100.0%** | **100.0%** | **8.00** |
| mass_age_sink | 0.0% | 49.2% | 100.0% | 100.0% | 8.00 |
| ega_energy | 17.4% | 29.9% | 41.7% | 56.0% | 64.00 |
| ega_x_usage | 16.7% | 24.0% | 41.7% | 57.6% | 37.75 |

- **The designed regime is a decisive win:** 2.06× raw-H2O at cap=32, 2.41× at
  cap=48, 1.95× at cap=64; generation length healthy (8.0 = target) vs raw's 46.5.
- **The miss is real, not noise:** at cap=16 (8% budget, ~11× lifetime turnover)
  mass_age = 0/384 vs raw 24/384. Mechanism: under extreme churn every old row
  dies regardless of rate — the needle's recurring-use rate cannot save a row
  that is old at every eviction; raw's frozen-mass luck retains ~0.5 needles/trial.
- **Fusion honest note:** ega_x_usage ≈ raw_h2o (no win at 32/48) — at these caps
  the static-energy half DILUTES the rate signal (24.0% vs pure mass_age 50.0% at
  cap=32). The Research-523 fusion shape needs its α tuned per workload before it
  earns its complexity; pure mass_age is the winner here. The one registered MISS
  (92 < 93 at cap=32) is this dilution, noise-level in magnitude, recorded not hidden.
- Sink pin (β=1) is a no-op here: no sink-row pathology exists in this workload —
  recorded as expected-inert (the pin exists for the Issue-716 sink class).

### Canary demo (RunawayStats + runaway_gate, r_max=1.5, p_cap_max=0.05)

| policy | cap | R_median | p_cap | gate |
|---|---|---|---|---|
| ring (over-eviction arm) | 8 | 8.000 | 0.91 | **FAIL** ✓ |
| raw_h2o | 32 | 8.000 | 0.56 | FAIL |
| **mass_age** | 32 | **1.000** | 0.00 | **PASS** ✓ |
| ega_energy | 32 | 8.000 | 1.00 | FAIL |

Non-vacuity PASSes: the canary separates healthy from runaway with identical
thresholds — and it catches raw_h2o too (its budget management loses the chain),
which perplexity-style metrics would read as fine. The Issue-750 generation-axis
extension is demonstrated, not just encoded.

### T3.3 Kendall-τ — per-head vs batch-summed keep-rankings (cap=32)

τ = 0.689–0.748 across 4 heads. Per-head and summed rankings DISAGREE
materially → per-(b,h) bookkeeping matters on drifted workloads; keep per-head
(the primitive's construction makes it free). Recorded for the kernel-side
decision (riir-ai Issue 836).

### T3.4 gates

| gate | verdict |
|---|---|
| G1 determinism (full matrix double-run bit-identical) | PASS |
| G2 update latency | **1.22 ns/row** (budget 10) PASS |
| G3 default-features no-regression (module fully gated) | PASS (`cargo check -p katgpt-core` clean; `kv_eviction` invisible) |
| G4 zero steady-state allocs (TrackingAllocator, test-pinned) | **PASS — after the gate caught 1 alloc/step** (see §History) |
| G8 mass/age family ≥ raw_h2o at every cap | **MIXED** — PASS at 32/48/64, FAIL at 16 |

## GOAT verdict

**Opt-in, regime-bounded, consumer-pull pending.** The primitive earns its slot
in `katgpt-core` (G1–G4 pass; the score is O(1)/row/step at 1.22 ns; the canary
is demonstrated load-bearing). It does NOT promote: no runtime consumer exists
and the cap=16 miss bounds the claim. Revisit the promotion question when
riir-ai Issue 836's consumer (summed-attention-weights kernel byproduct)
materializes — at real serving budgets (≥ ~17% of context) the measured recall
win is 2–4× raw-H2O on this workload.

## Harness bring-up history (instrument lessons, preserved)

Four sequential harness bugs were caught by INTERNAL CONSISTENCY failures before
any number was recorded — each is a class other harnesses hit:

1. **Zero-mass tie domination** (raw ≡ mass_age bit-identical across all seeds):
   rows never queried in their lifetime score exactly 0.0 under BOTH rankings →
   the ascending-index tie-break picks identical victims → the policies cannot
   diverge. Fix: admission self-attention (every row born with positive mass).
2. **Newborn-guaranteed-victim trim order**: evicting BEFORE the tick's
   attention makes the just-admitted row (mass 0, age 0) the argmin every tick.
   Fix: trim AFTER attention (post-attention state — what real serving decides on).
3. **EGA needle-detector artifact**: a fixed projection monotone in token id
   makes EGA (energy = dot(key, w_proj)) a perfect needle detector when needles
   occupy the high-id space. Fix: pseudo-random permutation of ids — energy
   uncorrelated with needle-ness.
4. **Priority-order contract slip** (`select_evict_into`): the first no-scratch
   rewrite skipped the sort when k ≥ candidates, silently returning index order.
   Caught by the module's own fixture. Fix: sort unconditionally, truncate after.
5. **G4 caught a live alloc**: the original `select_evict_into` allocated a
   candidate Vec per call (1 alloc/step) — the TrackingAllocator gate fired on
   first run of the alloc test. The no-scratch rewrite (sort `out` itself with
   score lookup) is both alloc-free and faster (G2 1.90 → 1.22 ns/row).

## Per-stack ledger

- **Slot:** KV/eviction (serving hot path).
- **Decision:** opt-in `usage_rate_eviction`; promotion re-gate = consumer
  presence (Issue 836) + a real-corpus re-run of this matrix.
- **Boundary recorded:** win regime = budget ≥ ~2× churn horizon (cap ≥ ~17%
  of stream here); extreme-pressure regime (8%) belongs to lifetime-mass or
  hybrid policies — the fusion's α-dilution finding suggests the hybrid design
  needs its own measurement before reuse.
