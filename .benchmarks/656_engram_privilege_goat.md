# Benchmark 656 — Counterfactual privilege gating for engram fusion (GOAT gate)

> **Issue:** `656` ·
> **Source:** riir-train [Research 419 §5.2](../../riir-train/.research/419_LOPD_Latent_Privileged_Context_OPSD.md) (LOPD, arXiv:2608.13040), modelless corollary
> **Feature:** `engram_privilege` (opt-in) · **Date:** 2026-08-16 · **Host:** M3 Max (aarch64, release)
> **Verdict:** **T1 DEFENDS** the mechanism. G1–G4 PASS. **Stays opt-in** — the win is real but
> scope-limited, and the scope limit is structural, not parametric.

## What shipped

`crates/katgpt-core/src/engram/privilege.rs` (+ one cfg-gated kernel sibling in `kernel.rs`):

| Symbol | Role |
|---|---|
| `PrivilegeLedger` | Per-slot advantage EMA `Δ` + cached factor `p = σ((Δ−m)/s)` + dual `β` |
| `PrivilegeConfig` | `alpha`, `margin`, `scale`, `gate_floor`, `dual_eta`, `veto_epsilon` |
| `PrivilegeTrace` | Stack-only `Copy` record of which slots fused, for the deferred outcome report |
| `CreditAssignment` | `Uniform` / `GateWeighted` split for the cheap aggregate-δ path |
| `fuse_into_hidden_state_privileged` | `out = (base_gate · p_slot) · v`, zero-alloc |
| `sigmoid_fuse_scaled_into` | Kernel sibling: applies a scalar to the gate, **returns the unscaled gate** |

### Two deliberate deviations from the issue's scoping

1. **The ledger is a side-car, not a table field.** Issue 656 §1 scoped `Δ_slot` as a
   field inside the engram table ("table layout change, versioned (freeze/thaw bump)").
   Shipped instead as a separate `PrivilegeLedger` indexed by slot, because
   `InMemoryEngramTable` is a **frozen, BLAKE3-committed** snapshot — adding mutable
   per-slot state either poisons the commitment (two tables with identical patterns but
   different usage histories would stop sharing a root) or gets excluded from it, at
   which point it was never part of the table. The side-car needs **no trait change and
   no freeze/thaw bump**, and composes with every `EngramTable` impl.
2. **The privilege factor rides on the scalar gate, not a second vector pass.**
   `sigmoid_fuse_scaled_into` folds `p` into the gate before the `D`-element store — one
   f32 multiply per head instead of `D`. This is what makes the G2 budget reachable at
   all. `gate_scale = 1.0` is **bit-identical** to the shipped `sigmoid_fuse_into`
   (pinned by `scaled_fuse_with_unit_scale_is_bit_identical`).

## T1 — planted-drift PoC

`crates/katgpt-core/tests/bench_656_engram_privilege_poc.rs`
(`D=32`, 64 slots, 16 active, 25% poison, 400 train rounds, 64 eval queries).

Error metric is **per-path and scale-invariant** — each arm is compared against its own
uncorrupted self, because the privileged path applies a uniform attenuation that has
nothing to do with poison:
`rel_err = |S(poisoned) − S(clean)| / |S(clean)|`, `recovery = (err_naive − err_priv)/err_naive`.

The oracle is "poison slots **evicted**", not "poison replaced by good patterns" — a veto
can zero a bad entry, it cannot conjure a good one. Scoring against replace-with-good
would cap recovery at exactly 0.5 by construction, making the issue's own ≥50% bar
unreachable-with-margin.

| Regime | err_naive | err_priv | recovery | purity | p_good | p_poison |
|---|---|---|---|---|---|---|
| **A · sign-opposed drift** | 0.3332 | 0.0001 | **100.0%** | 1.000 | 1.0000 | 0.0000 |
| B · similarity-separable (control) | 0.0036 | 0.0016 | 55.7% | 0.997 | 1.0000 | 0.4471 |
| C · clean table (G1) | 0.0000 | 0.0000 | n/a | 1.000 | 1.0000 | — |
| D · class-conditional (scope limit) | 0.3333 | 0.0566 | 83.0% | 0.946 | 1.0000 | 0.1697 |

**Regime A is the load-bearing result.** Poison is constructed with *identical cosine to
the query* as the good patterns but the opposite utility projection, so the similarity
gate is provably blind — it assigns both the same gate. Purity goes 0.500 → 1.000 and the
poison factor is driven to ~0. The similarity-only gate cannot do this at any temperature.

## G1–G4

| Gate | Bar | Measured | Verdict |
|---|---|---|---|
| **G1** relevance preserved on clean tables | rel_err ≤ 1e-3 | **0.0** (exact) | PASS |
| **G2** amortized overhead at retrieval events | ≤ +20% | **1.180×** at period 64, holding 100% recovery | PASS |
| **G3** existing engram tests, both feature states | no regression | 142 pass (off) / 167 pass (on) | PASS |
| **G4** zero-alloc hot path | 0 allocs | **0/0/0/0** (fuse / update / read / trace) | PASS |

G1 has two independent anchors beyond the PoC: `all_privilege_one_matches_unprivileged_fuse`
(saturated ledger reproduces the shipped fuse to 1e-4) and
`uniform_cold_ledger_is_a_uniform_scale_of_the_plain_fuse` (a uniform-Δ ledger is *exactly*
a scalar multiple of the plain output — ranking preserved by construction, not by
measurement).

G3's build matrix: `--features engram` / `--features engram_privilege` / default /
`--no-default-features` / `--all-features` all clippy-clean.

### Cost ladder (interleaved min-of-5 × 20k reps)

| Update cadence | Cost vs. plain fuse | Recovery (regime A) | Updates over 1600 events |
|---|---|---|---|
| fuse only, no updates | 1.106× | — (never learns) | 0 |
| every 1 | 4.128× | 100.0% | 1600 |
| every 4 | 1.877× | 100.0% | 400 |
| every 16 | 1.298× | 100.0% | 100 |
| every 64 | **1.180×** | 100.0% | 25 |
| every 256 | 1.149× | 100.0% | 7 |

**G2 is gated on the amortized cost at a cadence that still clears the recovery bar**, not
on the fuse-only floor. The floor (1.106×) is the cost of a ledger that never updates —
i.e. never learns anything — so quoting it as "the cost of privilege gating" would be
measuring the wrong quantity. Quality and cost are measured at the *same* cadence.

#### Measurement note — the estimator matters here

An earlier draft used median-of-3 at 4k reps with the baseline timed once up front. On
identical code it read **0.977× / 1.122× / 1.201×**, straddling the 1.20× bar — the
verdict was a coin flip on machine load. Two errors: (a) median deliberately retains a
noise-inflated sample, when contention only ever *adds* time to a timed loop, so **min**
is the correct estimator for latency; (b) dividing every variant by a single up-front
baseline pushed load drift straight into the ratio. Now interleaved min-of-5 at 20k reps:
three consecutive runs read fuse-only **1.069× / 1.110× / 1.120×** and all gates PASS every
time.

## Scope findings (measured, not asserted)

### 1. Control B — the similarity gate already suffices on similarity-separable drift

When poison is anti-aligned with the query, `σ(dot/τ)` suppresses it on its own:
`err_naive` is **0.0036 vs. regime A's 0.3332 — 93× smaller**. Privilege gating recovers
55.7% *of an already-negligible penalty*. Honest reading: **for corruption the similarity
gate can see, this primitive is not worth its cost.** It earns its keep only where
relevance and utility genuinely diverge.

### 2. Scope limit D — a per-slot scalar cannot be query-conditional

The issue frames δ as query-conditional ("utility must be measured at use time,
conditioned on the current query"). That is true of the **measurement** but not of the
**state**: `Δ_slot` is one scalar per slot, averaged over every query that touched it. Its
query-conditionality is exactly `EvidenceTier`'s — none — just continuous instead of
3-tier.

Regime D builds the real LOPD F2 shape (three orthogonal directions; one entry that helps
query class 0 and hurts class 1, similarity-identical to a good entry). Headline recovery
looks decent at 83.0%, **but `recency_latch = 0.6622`** — training one extra round, which
flips which query class the EMA saw last, swings `p_poison` by 0.66. The ledger is not
converging; it is **oscillating**, and a sharp gate (`scale = 0.03`) turns that oscillation
into a coin flip. The 83% is an artifact of where training happened to stop.

**Do not read regime D as a win.** Under class-conditional utility the gate is unstable,
and mitigation (larger `scale`, smaller `alpha`) buys stability by giving up
discrimination. Genuine query-conditional gating needs the query in the ledger key — a
different primitive, not a tuning of this one.

### 3. The cheap aggregate path fails exactly where the gate is needed

`CreditAssignment::{Uniform, GateWeighted}` split one aggregate δ using **unsigned**
weights, so every traced slot receives credit of the same sign. On regime A the aggregate
δ is ≈ 0 (good and poison cancel) and recovery is **−0.0% vs. 100% exact**. The 8.5×
scorer-call saving (800 vs. 6800) buys nothing in the only regime that motivates the
primitive. The cheap path is for same-sign fuses; it is shipped and documented as such,
not removed, because it is the first thing a cost-sensitive host reaches for.

### 4. "7 updates suffice" is a noise-free artifact

The cadence sweep is flat at 100% down to 7 updates — because the fixture's δ is exact.
Under outcome noise scaled as a multiple of |δ| (regime A, recovery %):

| period | noise 0× | noise 1× | noise 3× | noise 8× |
|---|---|---|---|---|
| 1 | 100.0% | 100.0% | 99.5% | −11.2% |
| 4 | 100.0% | 100.0% | 77.9% | 6.0% |
| 16 | 100.0% | 100.0% | 90.0% | −9.4% |
| 64 | 100.0% | 100.0% | 98.0% | 36.8% |
| 256 | 100.0% | 100.0% | 66.9% | 51.6% |

Robust through 1× noise; degrades badly and non-monotonically at 8×, where a sharp gate
latches onto noise. **Sparse updating and noisy outcomes are not independently safe
choices** — budget update cadence against the host's actual outcome-label quality, not
against this fixture.

## T4 — cross-references

- **`EvidenceTier` (riir-clippy `src/memory.rs:66`) is the discrete predecessor.** A 3-tier
  state machine (`Certified` / `Heuristic` / `Withdrawn`) advanced by
  `advance(success, verified, consecutive_fails)`, with `Withdrawn` absorbing except via
  explicit re-certification. Same job — "has this entry earned the right to be used?" —
  at 3 discrete levels from **history only**. `PrivilegeLedger` is its continuous sibling:
  `Δ ∈ ℝ` instead of 3 tiers, δ measured against the *current* query instead of a boolean
  success flag, and no absorbing state (a vetoed slot recovers if its δ recovers). Neither
  is query-conditional in its *state* — see scope finding 2.
- **`clr_reliability_scores` (`katgpt-core/src/set_attention.rs:701`)** is the nearest
  in-repo substrate: `r_j = (mean_m σ(h_j · dir_m))^M`. Query-**independent** (a function
  of fixed direction vectors), so it cannot express "useful for *this* query". Not a
  duplicate.
- **`BayesianFilterArm` (`katgpt-core/src/manifold_bandit/mod.rs:220`)** tracks per-arm
  outcomes with a Beta-Bernoulli posterior + drift. Closest structural analogue, but it
  drives *arm selection* via Thompson sampling, not a multiplicative gate on a retrieved
  value, and it takes a `[0,1]` reward rather than a signed counterfactual advantage.
- **NON-goal (per the issue):** query-conditional upgrade of `LatentFixMemory` retrieval
  weighting (riir-clippy). Separate repo, no demonstrated binding, not attempted here.

## Promotion decision — stays opt-in

The GOAT gate passes and the gain is **modelless** (two evaluations and a comparison; the
ledger is runtime latent state like a routing table — no gradients, no weight mutation).
It is **not** promoted to default because:

1. The win is regime-specific. Control B shows that where the similarity gate can see the
   corruption, this primitive costs ~18% for a negligible return. Default-on would tax
   every engram consumer for a benefit only some of them can collect.
2. It requires the host to supply a **scorer** and an outcome signal. There is no sensible
   default for `margin` / `scale` — they are in the host's score units — and a
   mis-scaled config silently degrades to either a no-op or a blanket veto.
3. Scope finding 2 (instability under class-conditional utility) is a real hazard that a
   default-on flag would inflict on hosts who never asked for it.

**Promote when** a host demonstrates the regime-A shape on real data (relevance and utility
genuinely diverging) *and* can supply outcome labels at ≥1× noise quality.

## Reproduce

```bash
cargo test -p katgpt-core --features engram_privilege \
    --test bench_656_engram_privilege_poc --release          # T1 + G1 + G2
cargo test -p katgpt-core --features engram_privilege \
    --test bench_656_privilege_alloc_check --release         # G4
cargo test -p katgpt-core --features engram_privilege --lib engram   # G3 (on)
cargo test -p katgpt-core --features engram --lib engram             # G3 (off)
```

## Known unrelated failure

`bench_360_engram_staging_goat` G2 fails on this host (staging COW 3.1× vs. the ≥10× bar)
**with `engram_privilege` off as well** — pre-existing, a 1M-slot memory-bandwidth gate
untouched by this work. Not a regression from Issue 656.
