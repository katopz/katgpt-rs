# The percentile-index / tail-support audit — the repair that blinded the gate

**Status:** RECORD — instrument change landed + canaried; 2 sibling sites open
(riir-ai, riir-train), both sibling-owned. katgpt-rs scope clean.

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

Population after: **126 sites over 9 of 19 repos — 0 DEGENERATE, 2 TRUNC-VAR,
6 WEAK, 31 OK, 62 UNRESOLVED, 25 SAFE**, katgpt-rs 40 → **41**.

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

## 7. Open — both sibling-owned, both low severity

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
