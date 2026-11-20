# Bench 990: JetSpec++ outcome-distribution speculation cache — Issue 976 gate battery

**Status:** DONE. All pre-registered gates PASS on the frozen teacher-forced proxy trace. The primitive ships opt-in (`outcome_cache`). Promotion is NOT claimed: it is posture-gated (see Verdict).

## Provenance

- m3, arm64, AC power, load 9–11 (other agents compiling). The gates measure hit counts, coverage and CRPS. They are deterministic and time nothing, so box load cannot change a number.
- Command: `cargo test -p katgpt-speculative --features outcome_cache,bigram_markov --release --test outcome_cache_trace_gates -- --nocapture`. G4: `--test outcome_cache_alloc_check`.
- Trace, frozen and modelless:
  - Corpus: `tests/data/austen_pride_prejudice_tail.txt`, BLAKE3 `4318b9af…f748`, pinned in the test. Text is lowercased and split into word tokens.
  - Drafter: `bigram_markov` greedy chain (top-m 16, γ=8), fitted on tokens [0, 40%).
  - Verifier: teacher forcing on the text. This is greedy-lossless: k is the leading run of drafted tokens that equal the text, and the bonus is the text token after that run.
  - Calibration rounds come from [40%, 70%) and number 5319. Test rounds come from [70%, 100%) and number 5311.
  - Mean accepted count on the test window is k = 0.088, a weak drafter.
- Frozen Pod: KL freeze CONVERGED (0.0051 ≤ 0.05), BLAKE3 `56610e9c…1ba7`. λ = 0.02626 at α = 0.1.

## Results — paired hit rate at equal B (Prop 2b)

| B | per-round ctx+backoff (shipped arm) | ctx, no backoff (ablation) | per-round, Pod only | static_top_b | geometric_ladder | stale (M6) | ceiling | W/L vs static |
|---|---|---|---|---|---|---|---|---|
| 1 | **5.87%** | 4.39% | 2.73% | 2.73% | 2.73% | 1.30% | 24.68% | 240/73 |
| 2 | **9.47%** | 7.55% | 4.97% | 4.97% | 4.97% | 2.79% | 24.68% | 361/122 |
| 4 | **14.42%** | 10.75% | 9.58% | 9.58% | 9.58% | 4.44% | 24.68% | 462/205 |
| 8 | **20.77%** | 15.35% ✗ | 16.63% | 16.63% | 16.63% | 8.23% | 24.68% | 564/344 |
| 16 | **24.27%** | 16.46% ✗ | 17.19% ✗ | 17.25% | 17.13% | 10.21% | 24.68% | 612/239 |

✗ marks a cell that fails strict ≥ against a fixed allocation.

| gate | bar | measured | verdict |
|---|---|---|---|
| Prop 2b dominance | per-round ≥ static_top_b AND ≥ geometric_ladder at every B ∈ {1,2,4,8,16} | +3.1 to +7.0 pts; 2.15× at B=1 | PASS |
| Conformal coverage | ≥ 1−α (0.90), finite-sample slack 0.02 disclosed | 0.9062 | PASS (at nominal) |
| E[k] forecast vs conformal-naive floor | Report the Floor | CRPS 0.690 vs 1.891 (ratio 0.365); Winkler ratio 0.422; coverage 0.986 vs 0.978; `BeatsFloor`; MAE E[k] 0.157 vs naive 0.164 | PASS |
| G4 alloc-free | per-round path allocates 0 B (10 000 rounds) | 0 B | PASS |
| Issue 940 determinism | second full run is byte-identical (Pod digest, λ bits, every hit row); first-index ties unit-pinned | identical | PASS |
| M6 staleness ladder | stale ≤ predicted ≤ true, per B, not pooled | holds at every B (table) | PASS |

## Findings, from the gate runs themselves

1. **The context-free Pod alone does NOT dominate.** With `p̂(a|k)` frozen and context-free, per-round selection only varies through P(k). It ties the static cache at B ≤ 8 and loses at B=16 by 3 rounds (17.19% vs 17.25%). So Prop 2b needs a round-conditioned bonus posterior; the histogram alone is not enough.
2. **The round-conditioned residual alone also fails at large B.** This arm draws the bonus from the drafter's residual (the rejected draft token removed, renormalized by 1−p_rej). It wins at B ≤ 4 but loses at B ≥ 8: the bigram row misses the common function words the frozen vocabulary captures, and a fully-supported row leaves no escape mass. The same gap is why λ collapsed to 0 in that run.
3. **The shipped arm is Katz-style backoff** (`BonusRows::with_backoff`). It keeps the residual row, scales it to leave ≥ `BACKOFF_ESCAPE` (0.05), and spreads the tail over the Pod's `p̂(a|k)`. It dominates at every B and raises the ceiling from 16.6% to 24.7%. This is the modelless fusion: no gradient, every number is counted.

## Disclosed

- **The trace is a PROXY**, not a serving trace. It uses a word-level bigram drafter, a teacher-forced verifier, and a single stream. It satisfies the "frozen trace" half of the pre-registration: deterministic, BLAKE3-pinned input, never tuned against the test window. It does not stand for the multi-request serving workload the issue's posture triggers name. Absolute hit rates will move with a real drafter. The claim here is the *shape*: per-round posterior with backoff ≥ fixed allocations.
- Coverage carries a 0.02 slack below nominal, because consecutive text rounds are not exactly exchangeable. Measured coverage is above nominal anyway.
- Recalibration (`ConfidenceRecalibrator`) is per-position decile histogram binning with a Beta(1,1) prior. It is counted, not fitted. The Pod-only arm gains about 0.3–1.9 pts from it.
- No latency (G2) claim. The per-round cost is an O(B·81) selection over stack arrays, and there is no serving caller to time it in.

## Verdict

**GOAT on the proxy trace; opt-in; NOT promoted.** The Feature Flag Discipline needs a modelless gain *where it ships*. There is no multi-request or pipelined serving caller yet:
- riir-infer `multi_host` is still opt-in and foreground-chunked;
- no external crate calls Plan 339's `HardwareAwarePrefixScheduler` (checked 2026-11-20).

Promotion re-arms on Issue 976's original triggers, now recorded in HISTORY.md. When either fires, re-run this battery on that caller's frozen trace.

Open question from the issue: "does repo prose count as the frozen trace?" Answered NO, by a Claude verdict recorded here. The stranded corpus was this repo's own markdown, which made it self-referential. The trace now uses the tracked public-domain Austen fixture with a pinned digest. The stranded ref `refs/wip/i976-stranded` holds an incomplete stub (`try_freeze` never compiled) and is superseded by this landing.
