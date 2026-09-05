# The percentile-index / tail-support audit — the repair that blinded the gate

**Status:** RECORD — instrument change landed + canaried; 2 sibling sites open
(riir-ai, riir-train), both sibling-owned. katgpt-rs scope clean, and §9–§10
record what resolving its UNRESOLVED rows by hand on 2026-09-04 turned up: a
**third** instrument axis (severity, not population) and a real defect in
shipped `katgpt-core` library code.

**Instruments:** `scripts/percentile_index_audit.py` (report, 19 repos, always
exit 0) + `scripts/percentile_floor_gate.py` (gate, katgpt-rs only, pins in
`scripts/percentile_floors.txt`, run per-push by `scripts/docs_gate.sh`).

## Why this record exists

The audit's whole record lived in AGENTS.md prose, and by 2026-09-03 evening
every number in it was stale — for a *good* reason, which is the finding.

## 1. The repair campaign worked, and it is verified

AGENTS.md recorded **12 DEGENERATE, all print-only** as of the morning
measurement. Re-run the same evening: **0 DEGENERATE across all 19 repos.**

That drop is repair, not blindness, and the evidence is the commits — not the
count:

| repo | commit | what it did |
|---|---|---|
| riir-ai | `03a91ed59` (Issue 853) | `SeriesStats.p99` nearest-rank + `p99_support`, and the **10-site DEGENERATE sweep** |
| riir-mmorpg-examples | `ee9da24` (Issue 093) | the G2 gate's "p99" was the max of 100 samples — the one DEGENERATE-**ASSERTED** site in the workspace |
| riir-game-sdk | `f896bca` (Issue 023 T4) | the p99 three budget rows rest on had 3 samples — nearest rank + tail support |
| riir-chain | `7f3a3910` / `ffb39061` (bench_012) | percentile-semantics canary; the engram-commit GOAT was suite-less, found *by* the per-site read |

10 + 1 + 1 accounts for the 12 exactly. The instrument found the defects, four
owners fixed them independently, and the re-run confirms it. That is the
audit working end to end.

## 2. …and the fix took the sites OUT of the population

Total went **130 → 114** in the same window. A fixed site should still be
*counted* (as SAFE), so a falling total wanted an explanation.

The repairs all consolidated the arithmetic behind one helper:

```rust
fn nearest_rank(sorted: &[f64], p: f64) -> (f64, usize) {
    let n = sorted.len();
    let idx = ((p * n as f64).ceil() as usize).clamp(1, n) - 1;
    (sorted[idx], n - idx)
}
```

`p` is now a **parameter**. Every pattern in the audit's vocabulary required a
*literal* p (`0.99`, `* 99 / 100`), so neither the ~101 call sites nor the
helper body matched anything, and 16 sites left the population. Measured:
**seven byte-identical copies of that helper across five repos** — katgpt-rs
`examples/monopoly_04_bench.rs:143`, riir-ai ×4 (`games-shared/src/stats.rs`,
`riir-rag`/`riir-gpu` examples, `bench_392`), riir-chain
`tests/bench_012_engram_commit_goat.rs:158`, riir-game-sdk
`riir-e2e/src/percentiles.rs:75`, riir-mmorpg-examples
`tests/e2e_topology.rs:82` (as `quantile`) — **all invisible.**

So `max_degenerate = 0` was green over a population that no longer contained
riir-ai's percentile surface *at all*. This is the **fourth** instance of the
classifier-narrowness failure this audit already documents three times — and
the first one reached by a **correct** fix rather than a bug. Nobody did
anything wrong; the vocabulary just stopped naming the code.

## 3. What landed (2026-09-03)

- **`TRUNC-VAR`, a fifth verdict.** Two new VOCAB entries match
  `<p> * <n> as fNN` / `<n> as fNN * <p>` with a **variable** p. Correctness
  is decidable from the shape even though p is unknown, so a rounded body
  clears as `SAFE` via the existing `ROUNDED_RE` path and a truncating one is
  a **finding** — strictly more than `UNRESOLVED` can say.
- **Audit the helper, not its call sites.** Only the body is classified. One
  row beats 101, and the body is the single point of truth.
- **A scope-name discriminator, because a variable-p multiply is otherwise
  ordinary arithmetic.** `HELPER_SUBSTR` / `HELPER_EXACT` are committed DATA
  (`in_percentile_scope`), matched against the enclosing fn **and** the
  nearest enclosing closure binding — the closure half is not optional, since
  riir-train's site is a `let quartile = |q: f64|` inside `fn main`.
  Measured against all **27** `let <idx-ish> = … as fNN * <var>` candidates in
  the workspace: **admits 8, rejects all 19** non-percentile ones, including
  the two a bare `rank` substring would have swallowed —
  `spectral_adaptive_ranks` (truncating, but a rank *budget*) and
  `principal_rank`. A report that cries wolf on rank budgets is a report
  nobody runs.
- **A hole in `ROUNDED_RE` closed.** It cleared `.trunc()` as "explicit
  rounding" while its own comment said "the bug is floor/truncation".
  `x.trunc()` **is** `x as usize` for non-negative x, so the defect spelled a
  second way classified as SAFE. Latent when found (zero percentile-context
  `.trunc()` sites, measured) — but a ceiling is only as wide as its
  classifier, and this one had a spelling-shaped gap. `.floor()` was never in
  the set and must never be: it is the defect's own name.
- **Five end-to-end `selftest()` canaries** through `audit_file` (not the
  regex), because the anti-flood discriminator lives in the caller. The one
  that earns its keep is case (d)/(e): the *same* truncating shape outside a
  percentile scope must produce **nothing**.
- **A fourth ceiling**, `max_trunc_var = 0`, canaried three ways: a planted
  truncating helper → exit **1**; `HELPER_SUBSTR` narrowed to nonsense → exit
  **2** (instrument untrustworthy, no verdict); both reverted → exit **0**.

Population after (2026-09-03 evening): **126 sites over 9 of 19 repos — 0
DEGENERATE, 2 TRUNC-VAR, 6 WEAK, 31 OK, 62 UNRESOLVED, 25 SAFE**, katgpt-rs
40 → **41**.

Re-measured **2026-09-04 over the live 16** (three repos retired to
`git/obsolete/`, and §9–§11's instrument changes landed): **125 sites**,
katgpt-rs still 41 — 0 DEGENERATE, 0 TRUNC-VAR, 0 WEAK, 20 OK, **2**
UNRESOLVED, **19** SAFE. ASSERTED 4 → 5 workspace-wide. Read the 126 → 125
edge as three repos leaving and one site changing class, not as a measurement
of anything.

UNRESOLVED 11 → 2 is §11: the rows were **read**, not reclassified by an
instrument change. The population is unchanged at 41 throughout, which is the
property §2 says to check.

Re-measured **2026-09-06**: **110 sites over 9 of 16 repos — 0 DEGENERATE, 0
TRUNC-VAR, 0 WEAK-ASSERTED, 30 OK, 38 UNRESOLVED, 42 SAFE**, katgpt-rs still
**41**. All four gated classes are zero in all sixteen repos, and the workspace
walk is 11,132 `.rs` files.

**The 125 → 110 edge was checked before it was pinned, not after.** A falling
audit population is a REPAIR or a BLINDNESS and the count cannot tell the two
apart — §2 above is the record of that happening, and the fourth instance of it
was caused by a *correct* fix. Three things decide it here, and none of them is
the total:

1. The **instrument is unchanged**: `scripts/percentile_index_audit.py`'s last
   commit is `0333c2c1`, 2026-09-04 01:04, before the 125 measurement. No
   vocabulary, scope or rounding rule moved.
2. **katgpt-rs holds at exactly 41**, as it has through every measurement in
   this document. A tokenizer regression is not repo-selective; a population
   that is invariant in the one repo whose sources did not move is the
   signature of sibling churn, not of a blind classifier.
3. The churn is **measured, not assumed**: 512 sibling commits and 47 deleted
   `.rs` files in the window, headed by riir-mmorpg-examples deleting **40**
   during its in-flight consumer-extraction campaign.

What is NOT claimed: the 15 sites are not attributed one by one to the commits
that removed them. The claim is the discriminator — instrument fixed, katgpt-rs
invariant, sibling churn large — not a per-site reconciliation.

**And the zero is now held by something.** `scripts/percentile_drift_sweep.py`
+ `scripts/percentile_drift_floors.txt` gate all four classes at 0 across every
contract repo, with two population floors (`min_sites` per repo, and
`min_rs_files` for the seven repos whose site floor is 0 and therefore detects
nothing). Until 2026-09-06 the 12-site repair campaign's result was held only
in katgpt-rs, whose own gate could never have seen a sibling.

## 4. The shape claim, stated exactly — and a retraction

The first cut of the new selftest asserted *"`ceil(p*n)-1` can never be the
max"*. **That is false**: p=0.75, n=3 gives `ceil(2.25) = 3` → idx 2 = n−1.
The true relation is a one-rank shift of the *same* boundary:

| form | is the max for |
|---|---|
| `floor(p*n)` (`as usize`) | every **n ≤ 1/(1−p)** |
| `ceil(p*n) − 1` (nearest rank) | every **n < 1/(1−p)** |

Nearest rank is never worse, is strictly better at **exactly one** n (the
integral boundary — pinned, per p, over n ∈ 2..5000 in exact `Fraction`
arithmetic, because at p=0.999 the float comparison pins the wrong claim), and
is *still* the max below it — where no such quantile exists in the sample at
all. That last part is why the helpers return `support`, and why `SAFE` in this
report means **"correct form"**, never "cannot be one observation".

## 5. Two corrections to the reasoning in AGENTS.md

**(a) "false RED, not a false green" is assert-direction dependent.** The naive
index is one rank too **high**, which makes a `p99 < budget` assert stricter —
true, and that is the common shape. But riir-ai's `bench_336` G6a row asserts
`p95 >= 0.3 * range` (a *diversity* floor), where a too-high tail is a false
**GREEN**. Read the direction off the assert, not off the defect.

**(b) `asserted` is structurally blind for helpers, by design.**
`is_load_bearing` is deliberately same-fn-scoped (a wider scope manufactured
false ASSERTED rows on the first cut). A percentile *helper* is one call frame
removed from the assert its return value decides: riir-ai's `fn percentile` is
"print-only" by that test while its p95 is the subject of a GOAT assertion two
frames up. So `TRUNC-VAR` is reported and gated **regardless** of `asserted` —
the only class here that is.

## 6. Why the floor is deliberately NOT ratcheted to the measurement

`min_sites_scanned` stays at **30** against a measured 41. A repair campaign
can legitimately *shrink* this number — consolidating ten inline index
computations behind one correct helper removes nine sites — and that is exactly
what took the workspace 130 → 114. **The population count cannot distinguish
repair from blindness**; only the committed vocabulary + `selftest()` can, and
the external evidence in §1 is what settled it this time. A floor that
ratcheted up to the last measurement would red the next such refactor and
teach whoever hit it that the gate is noise.

## 7. Open — sibling-owned, low severity

Filed where they live, not fixed here: riir-ai `.issues/861` (it moves a
GOAT gate's measured value) and riir-train `.issues/508`.

**One of the two is now CLOSED** (verified 2026-09-04 in the source, not
inferred from the issue's absence — the file is gone per the noise-reduction
rule, which on its own is indistinguishable from never having been filed):
riir-train `68a08fba` reindexed the probes nearest-rank *and* renamed them
`positional`, since `lengths` is in visit order — so the mislabel noted below
was fixed alongside the arithmetic. riir-ai `.issues/861` is still OPEN and is
the workspace's only remaining **TRUNC-VAR** (2 → 1).

Re-measured 2026-09-04, the 6 WEAK rows are all siblings' and all correctly
`asserted=False` — checked per site against the one-hop alias chase of §9, not
taken from the flag: `bench_336:710`'s `p95_total_ms` is `println!`-only while
that fn's two asserts are on `median_total_ms` / `per_npc_us`.

| site | shape | severity |
|---|---|---|
| riir-ai `crates/riir-engine/tests/bench_336_committed_blend_goat.rs:177` | `fn percentile` uses `floor(p*n)` + `.min(len-1)` | **low, but load-bearing.** n = 10,000 / 1,000 / 1,000 at its three call sites, so not degenerate — a one-rank-too-high bias. Direction is false-GREEN (§5a). A straggler of that repo's own Issue 853 sweep, missed because the literal-p vocabulary could not see it either. |
| riir-train `crates/riir-train-engine/examples/plan318_sft_train.rs:1121` | `let quartile = \|q: f64\|` uses `floor(q*n)` + `.min(len-1)` | **low.** Print-only `eprintln!` in an example, guarded by `order.len() >= 4` — and at exactly n=4 the "p75" is the max. Separately mislabeled: `lengths` is in *visit* order, so these are positional probes along the order, not quantiles of a distribution. |

## 8. One observation, not a finding

`walk_rs` skips `target` / `.git` / `node_modules` / `.venv` and nothing else,
so it walks **2,525 mined third-party `.rs` files** under `riir-train/data/`
(of that repo's 3,961 total). Zero findings come from there today — verified
per-site, not assumed — so the pollution is latent. Left alone deliberately: a
`data/` skip rule is a guess about a directory name, and this report's own
history is a run of classifiers that were narrow rather than wrong.

## 9. A third instrument axis — severity, not population (2026-09-04)

Every narrowness instance above made the report **smaller**. This one made it
look **milder**, which is worse: a miss leaves a hole someone may eventually
notice, a downgrade prints a plausible row.

`is_load_bearing` searched the assert's arguments for the name bound on the
site's **own line**. The normal shape puts one hop between them:

```rust
let p99_idx = (READS as f64 * 0.99) as usize;   // the site,  var = p99_idx
let p99_ns = latencies_ns[p99_idx];             // the hop
assert!(p99_ns < 200, "G5 FAIL: get_chunk p99 {p99_ns}ns >= 200ns");
```

That is `katgpt-core/src/content_store/goat.rs:323` — a percentile deciding a
G5 gate inside **shipped library code**, printed with `asserted=False`, i.e.
in the "print-only (misleading a reader)" bucket rather than
"+ ASSERTED (deciding a verdict)".

Closed by `_subscript_aliases` (`01c06ccd`), deliberately **one** hop and
deliberately **subscript-only**: `var` must appear inside a bracket pair, so
the alias IS the sample the rank selects. Chasing arbitrary arithmetic, or
chasing transitively, buys shapes nobody writes at the cost of a false
ASSERTED — the same defect pointed the other way, and §5b is why a false
ASSERTED is expensive here.

Measured over all 16 contract repos **before** committing: ASSERTED **4 → 5**,
exactly the one site found by hand, zero false positives, no verdict crossed
into a gated bucket, no pin moved. Three selftest cases, each canaried by
reintroducing what it catches, each exiting **2** (untrustworthy instrument)
rather than 1: same-name-only, any-mention alias, transitive chase. The third
pins the **bound**, so a future reader widens deliberately instead of reading
a pass as evidence that two hops were covered.

The site itself is **not** a finding: `READS = 1_000_000`, idx 990,000, tail
support 10,000.

## 10. The per-site read of UNRESOLVED paid — `StabilitySnapshot` (2026-09-04)

§"UNRESOLVED is not clean" is the report's own instruction, and katgpt-rs's 11
rows had never been read. Two of the eleven are tokenizer noise (a `sin()`
data generator, a weighted score — neither indexes anything), eight are bench
harnesses whose n is a runtime loop count, and **one was a real defect in a
default-on library API**:

`katgpt-core/src/speculative/types.rs` — `StabilitySnapshot::compute`, feature
`stability_metrics` (**default-on**), consumed by `katgpt-forward`:

```rust
let p99_idx = ((n as f64) * 0.99).floor() as usize;
let p99_idx = p99_idx.min(n - 1);
```

`floor(n * 0.99) == n - 1` for **every n ≤ 100**, so `p99_ns` was the single
worst sample — and `stability_score = 1 - p99/p50` was therefore *decided* by
it — in any run of at most 100 steps. The `.min(n - 1)` prevented a panic, not
a wrong statistic. It was UNRESOLVED rather than DEGENERATE because `n` is a
slice length inside library code; the callers supply it, and the caller that
pinned the behaviour asserted the defect outright:

```rust
let known: Vec<u64> = (100..200).collect();      // n = 100, max = 199
assert_eq!(kn.p99_ns, 199, "P99 at index 99");   // tests/bench_102:85
```

Fixed with `nearest_rank_p99`, plus a new `p99_tail_support` field — the
quantity AGENTS.md says nobody prints. Both constructors are private to the
impl (grepped: zero struct-literal sites outside them across katgpt-rs,
riir-ai, riir-train, riir-game-sdk), so the field addition is contained.

Two details worth keeping:

- **Integer arithmetic, not `.ceil()`.** `(100.0_f64 * 0.99).ceil()` is not
  reliably 99 — 0.99 has no exact binary form — so a float ceil can land back
  on the max at *exactly* the boundary being fixed. `(n * 99).div_ceil(100)` is
  exact for every n, widened to `u64` so the multiply cannot overflow a 32-bit
  `usize` (wasip2 is a live target).
- **p50 was left at `n / 2`** on purpose: it can only land on `n - 1` for
  n ≤ 2, so it is not the max-landing shape, and moving the median convention
  would churn every published p50 for no correctness gain.

**And the fix blinded the auditor, exactly as §2 predicts.** No float, no `/`,
and `ROUNDED_RE`'s empty-parens `\.ceil\(\)` cannot see `.div_ceil(100)` — so
the repaired site left the population (41 → 40) and every ceiling above went
green over the hole. Third occurrence of this shape, second reached by a
*correct* fix. Closed in the same commit by an `int_div_ceil` VOCAB entry,
safe-by-construction, with its own selftest case: the site now reads **SAFE**
and the population is 41 again (UNRESOLVED 11 → 10, SAFE 10 → 11). Measured
workspace-wide before landing: **1 match, the intended one, zero false
positives.**

The lesson is not "remember to update the vocabulary". It is that **fixing a
site and keeping the instrument able to see it are one change**, and belong in
one commit — otherwise the gate's next green is over a smaller world than the
one it names.

## 11. T4 disposition: the stability gate stays print-only (2026-09-04)

The T1/T2 fix landed (`0333c2c1`); the remaining task — Plan 102 T6's dropped
`stability_score > 0.7` assertion — resolved **print-only by owner call**.
The 0.7 candidate failed 5/12 of the issue's own post-fix observations
(0.625–0.785 across 4 runs × 3 configs at n=500, tail support 6), and a
wall-clock tail-ratio gate on a shared box is the documented load-flaky class
the 1-layer arm re-demonstrated on the confirmation run (CV 1.16 — one
outlier step in the tail; scores 0.667/0.750/0.785). The 0.303 candidate (the
plan's own parenthetical, P99 < 3.3×P50) passes with margin so wide it
asserts nothing the adjacent `cv < 1.0` row does not already. What this issue
buys is that the metric now means what it says and is non-vacuously tested
(ordering + constant-pin tests, `types.rs:1483`); a wall-clock gate over it
was never landed and is recorded as deliberately not wanted. Issue file
removed at close; the record is this section + the AGENTS.md index row.

## 11. Reading all 10 UNRESOLVED rows found 3 DEGENERATE and 4 WEAK (2026-09-04)

§"UNRESOLVED is not clean" is this report's own instruction and katgpt-rs's
rows had never been followed. Every one, resolved by reading the caller —
which is the only place the sample count exists:

| site | n | idx | support | verdict |
|---|---|---|---|---|
| `katgpt-speculative/benches/bench_136_weaver_f16_latency.rs:177` | **50** (`const N`) | 49 | **1** | **DEGENERATE** |
| `examples/kimi_k3_hello_world.rs:298` | **8** (`KIMI_N_TOKENS` default) | 7 | **1** | **DEGENERATE** |
| `examples/kimi_k3_4b_hello_world.rs:308` | **8** (same) | 7 | **1** | **DEGENERATE** |
| `examples/corr_budget_01_bench.rs:41` | 500 (`N_ITERS`) | 495 | 5 | **WEAK** |
| `examples/lodestar_01_bench.rs:101` | 500 | 495 | 5 | **WEAK** |
| `examples/rosetta_01_bench.rs:42` | 500 | 495 | 5 | **WEAK** |
| `examples/best_buddies_01_bench.rs:48` | 500 | 495 | 5 | **WEAK** |
| `examples/datrie_01_bench.rs:123` | 10,000 / 1,000 | 9,900 / 990 | 100 / 10 | OK, one rank high |
| `crates/katgpt-core/src/speculative/types.rs:237` | caller's slice len | — | — | **the library defect, §10** |
| `crates/katgpt-types/src/simd/tests.rs:850` | — | — | — | tokenizer noise |
| `tests/precision_aware_draft_goat.rs:201` | — | — | — | tokenizer noise |

**Seven real findings that `max_degenerate = 0` was green over**, because a
degenerate site whose `n` lives in the caller is UNRESOLVED, not DEGENERATE.
The gate is not wrong — it gates what it can resolve — but "0 DEGENERATE" and
"clean" are different statements, and the gap between them is exactly the size
of the UNRESOLVED bucket.

Two of the three DEGENERATE ones were **published**: `bench_136` printed the
maximum of 50 samples in a `P99` table column beside an f16-vs-f32 speedup
ratio, and both `kimi_k3*_hello_world` printed the slowest single token as
`p99` on every default run (`KIMI_N_TOKENS` defaults to 8).

All 8 fixed to integer nearest rank. **And at n=8 and n=50 nearest rank does
not rescue the label** — a 99th percentile does not exist in 8 or 50 samples,
so both of those sites now print their **tail support** next to the number
instead of pretending otherwise. That is the §4 point made concrete: `SAFE`
means "correct form", never "cannot be one observation".

The two remaining UNRESOLVED rows are tokenizer noise, read and recorded here
so nobody re-derives them: a `(i as f32 * 0.97 - 18.0).sin()` data generator
and a `clean as f32 * 0.90 + boundary as f32 * 0.70` weighted score. Neither
indexes anything. They are **left in the population deliberately** — narrowing
the vocabulary to exclude them is how this file's four documented narrowness
instances happened, and two benign rows cost one read while a narrower
tokenizer costs an unknown number of misses.

Not fixed, and named rather than absorbed: the five byte-identical
`compute_stats` bodies across `examples/*_01_bench.rs` are the same duplication
AGENTS.md records as "seven byte-identical copies of that helper across five
repos". A canonical `nearest_rank` in `katgpt-core` would be the DRY answer and
was **declined here**: its only consumers would be examples and benches, so it
would be new public API with no production consumer — the shape Issue 719 T2
sits `[-]` for. The duplication is recorded as debt, not silently paid with
API surface.

## 12. The same read across the siblings — and what the shortcut got wrong

katgpt-rs's own 10 rows paid at 7 findings (§11), so the workspace's other 51
UNRESOLVED rows are worth the same treatment. Distribution, measured
2026-09-04:

| repo | UNRESOLVED |
|---|---|
| riir-ai | 23 |
| riir-chain | 14 |
| riir-neuron-db | 7 |
| katgpt-rs | 2 (both tokenizer noise, §11) |
| riir-game-sdk / riir-train | 2 each |
| riir-clippy / riir-mmorpg-examples / seal-remake | 1 each |

Filed where they live, each with the sites **verified in source** and the rest
listed explicitly as unverified candidates:

- **riir-ai Issue 863** (`1a97d75a5`; file RESOLVED + removed 2026-09-04 per
  the noise-reduction rule — **outcome in §13 below**) — two confirmed:
  `bench_416_cognitive_branch_goat.rs:107`, where `per_tick_us` is pushed
  exactly `const TIMED_TICKS = 100` times so `len * 99 / 100 == 99 == n - 1`,
  and the same helper also returns `sorted.last()` — **that gate's `p99` and
  `max` columns are the same observation printed side by side**; and
  `gemma2_cpu_bench.rs:358`, where n = `num_rounds`(3) × `measure_tokens`(20)
  = 60. Both print-only.
- **riir-neuron-db `.issues/610`** (`56ce87f`) —
  `bench_471_local_kv_delete_compaction_goat.rs`: `measure_ns` is called with
  `iters` = **1, 5, 50, 50, 50**, and `fmt_us` prints p50 and p99 together, so
  at `iters = 1` the bench prints **one sample twice under two statistic
  names**. Print-only.

### The shortcut, and its measured error rate

To make those issues actionable rather than "go read 23 sites", the candidate
list was generated by a **file-scoped scan of `const`/`let` integer literals
and `for _ in 0..X` bounds** — deliberately the thing `resolve_n` refuses to
do, since a binding in a different function is not in scope (§"first cut",
item 2).

**It was wrong twice out of the handful checked, both times in the
false-positive direction:**

| flagged as MAX | truth on reading the caller |
|---|---|
| riir-chain `bench_011_anomaly_sink_goat.rs:98` | `n = MISMATCH_ITERS / BATCH = 100_000 / 100 = 1000` → idx 990, **tail support 10** |
| riir-neuron-db `bench_323_local_kv_commit_levels_goat.rs:49` | `measure_ns(.., 1000, 100)` → idx 990, **tail support 10** |

In both cases the scan saw a small literal in the file (`100`) and the real n
was a *quotient* or a *call argument*. So the heuristic is reported in both
issues as a candidate generator with that error rate stated, never as a
verdict — which is the same discipline the report itself follows by putting
those rows in UNRESOLVED instead of guessing. **A candidate list that is
presented as findings is worse than no list**: it spends the owner's trust,
and the next real finding arrives to a reader who has learned to discount it.

One more shape worth naming, found in the same pass and *not* filed as a
defect: `riir-neuron-db/benches/bench_593_two_stage_knn.rs:130` has the
truncating index, and **every** call site discards it
(`let (p50, _) = measure(...)`) with one passing `n_iter = 30`. It is latent,
not live — whoever starts reading that tuple's second element inherits the max.
And `seal-remake/crates/seal-view/tests/texture_vessel_bench.rs:589` is
`assert_eq!((n * 99) / 100, n - 1, "the shape that reads a max as a p99")` — a
**canary asserting the defect exists**, i.e. the one site in the workspace
where this shape is the intended behaviour. Both are the reason a per-site read
cannot be replaced by a pattern.


## 13. Issue 863 closed — the riir-ai read, and the two the fix batch still missed (2026-09-04)

§12 filed the riir-ai rows and recorded the two sites verified in source; this
is the **outcome**. All 21 remaining UNRESOLVED rows in that repo were read to
their actual sample count, and the read resolved to **2 DEGENERATE + 5 WEAK +
14 OK**. The fix batch landed in riir-ai `b0c21283f` (all 7 flagged rows,
nearest-rank + a printed `(sN)` tail-support label, each target live-run).

**Both DEGENERATE finds were NEW** — the §12 candidate-token table had not
priced either:

| site | p | n | why |
|---|---|---|---|
| `riir-games-civ/benches/bench_479_emotion_mux_goat.rs` | .99 | **50** = `ROUNDS` | idx 49 = n−1 = the MAX; the candidate table had only priced this file's p95 |
| `riir-games/benches/swarm_tick_perf.rs` (cluster row) | .99 | **≈25** | the cluster row samples every **40th** tick of 1000, so its n is not the loop bound the scan saw |

**Every flagged row was print-only, so nothing was mis-gated** — the finding is
about numbers people quote out of gate output, which is the same severity split
§9 draws.

### A post-fix re-scan found 2 MORE, and that is the durable lesson

Re-running the audit *after* the batch landed surfaced two further real
DEGENERATE sites that the classifier had been resolving as **UNRESOLVED**:
`riir-engine/tests/bench_332_karc_goat.rs` G5 p95 at `MEASURE_ITERS = 100`
(`floor(100·0.95) = 95 = n−1`, invisible because the file-level `percentile_95`
helper is not statically resolvable) and `riir-games-civ/tests/prof_civ_world_state`
p99 at n = 200. Both fixed in the same commit. **Post-fix audit over riir-ai
reads 0 DEGENERATE / 0 TRUNC-VAR / 0 WEAK, with UNRESOLVED 20 → 12** — the 12
survivors are exactly the verified-OK large-n rows no static pass can reach.

This is the §2 / §"a repair can blind the gate" pattern arriving a third way:
the UNRESOLVED bucket is not a queue that empties monotonically, so a
zero-ceiling green after a repair wave is worth exactly one re-scan.

### The candidate generator's measured error rate — confirmed at 4 of 4

§12 reports the file-scoped literal scan as "a candidate generator, never a
verdict", at 2 wrong out of a handful. The full read raises that to **4
mis-attributed feeding loops out of 4 resolvable cases**:

| flagged n | true n |
|---|---|
| `bench_411_stat_modifier_goat.rs` — 256 | **200,000** = `N` (256 was warmup) |
| `bench_424_live_composition_goat.rs` — 50 / 1000 | **200** = `TIMED_ITERS` |
| `bench_488_npc_migration_dispatch_latency.rs` — 200 | **10,000** = `ITERATIONS` |
| `g_zero_04_player_ab_benchmark.rs` — 1000 | **≈10⁵–10⁶** (one push per alive player per tick × 1000 rounds) |

Four for four in the same direction the shortcut was warned about. The
discipline held — the rows were shipped as candidates and every verdict came
from reading the caller — but the error rate is now measured rather than
estimated, and it is high enough that a candidate list presented as findings
would have been actively misleading.

### The two ASSERTED sites are record-only — read the DIRECTION off the assert

Kept as record with no code change, because at their real n both are sound and
the reusable part is which way each would fail:

| site | assert | direction | health at real n |
|---|---|---|---|
| `bench_332_karc_goat.rs` G1 | `p95 >= 0.3 * range` (a diversity floor) | **false-GREEN** — a too-high tail passes a floor it should not | sound: n ≈ 4005–4950 (pairwise i<j over 90–100 forecasts), support ≈ 245 |
| `bench_488_npc_migration_dispatch_latency.rs` | `p99 <= budget` | **false-RED** — the §4 default direction | healthy: n = 10,000, support 100 |

§5's first correction says the "false RED, not a false green" claim is
assert-direction dependent; these two are the paired instance of it in one
repo. **Read the direction off the assert, not off the defect** — the defect's
sign is constant (one rank too high) and the consequence inverts with the
comparison.
