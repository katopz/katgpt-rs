# Bench 883 — Issue 875 T3: time-annealed solver sampling ranges + zero-terminal-weight truncation predicate

**Status:** GOAT PASS (opt-in; no promotion — the no-default-consumer rule)
**Date:** 2026-09-23
**Issue:** `875` T3 · **Research:** [582](../.research/582_Probability_Flow_Distillation_Wasserstein_Gradient_Flow.md) (arXiv:2605.09071 PFD)
**Commits:** katgpt-rs — this commit (see git log for the exact SHA) · riir-train — the `feat(875-t3)` commit whose message cites this bench (the consumer side, lands after the upstream)

## What shipped

1. **`TimeAnnealRange`** (katgpt-core `horizon_weights`, the T1 substrate's
   schedule layer): the paper's iteration-indexed sampling-range anneal
   `[0.02T, 0.98T] → [0.02T, 0.70T]` over the last 30% of iterations —
   ceiling eases linearly, floor FIXED (low-noise = max law weight, never
   traded away). Orthogonal to `dllm_solver`'s STATE-indexed entropy
   switching (TIME-indexed; the two compose). Endpoint postures pinned
   exactly (`a + (b−a)` does not round to `b` in f32 — `range_at` special-
   cases `p ∈ {0, 1}`); invalid fields fall back per-field to DEFAULT
   (the `RenoiseCeHorizon` pattern); `total` too small to contain an
   anneal window never anneals (no iteration falls inside it).
2. **`terminal_truncation_ceiling` / `truncated_w_mass_fraction`**: the
   zero-terminal-weight truncation predicate in closed form —
   `ε(t_cut) = ((T−t_cut)/(T−t_min))²` and its inverse
   `t_cut = T − (T−t_min)·√ε`. The paper's own `[0.02T, 0.70T]` IS this
   predicate at `ε = (0.30/0.98)² ≈ 0.0937`: skipping the top 30% of the
   range discards 9.37% of the (T−t) mass, squared-root-small because it
   sits where the law is weakest (`w(T) = 0`).
3. **The `dllm_solver` consumer seam** (combined gate
   `critical_interval_gate` + `horizon_weights`, the
   `renoise_ce_score_horizon` precedent): `annealed_renoise_range` (the VE
   σ-range for q-sample renoise at decode iteration `i` of `n`; VP callers
   bridge `ᾱ(σ) = 1/(1+σ²)`) + `renoise_level_skippable` (the predicate's
   operational form — skip renoise at levels whose remaining w-mass is
   within ε).
4. **The cross-repo quality gate** (riir-train `pfd_anneal` forwarding
   feature): the C9 toy harness consumes `TimeAnnealRange` directly —
   `TRangeMode::{Flat, AnnealPlain, AnnealRenorm}` remaps the GL8 node
   ladder into the schedule's per-update window. The fixture is consumed
   IN PLACE (never retrained, never duplicated — no second copy of the
   frozen bytes exists in this repo; the BLAKE3 pin stays single-sourced
   in riir-train). Flat is bit-identical to the C9 incumbent (pinned by
   `t3_flat_mode_is_bit_identical_to_the_incumbent_path` + full-scale
   parity below).

## G1 — correctness (in-module + root gate)

- Truncation round-trip `ceiling(fraction(x)) ≈ x` (rel < 1e-5) across the
  eps × t_min × horizon sweep; `w(T) = 0` anchor bit-exact; monotone in ε;
  conservative-1.0 NaN/degenerate policy.
- Paper alignment: `fraction(0.70 | t_min=0.02) = (0.30/0.98)² ± 1e-6`;
  `ceiling(that ε) = 0.70 ± 1e-6`.
- Anneal law: pre-anneal exactly `(0.02, 0.98)`, final exactly
  `(0.02, 0.70)`, fixed floor + non-increasing ceiling across the full
  schedule, linear mid-window, determinism (bit-equal repeats), monotone
  overrun extension, per-field fallbacks verified, `total = 0` asserts.
- f64 oracle for the anneal progress and both predicate forms.
- Consumer seam: σ scaling, ᾱ bridge validity `(0,1)`, skippable
  boundary at exactly the ceiling.
- Root gate `tests/bench_875_time_anneal_goat.rs` (required-features
  `horizon_weights` + `critical_interval_gate`): 4/4 debug + release.

## G2 — latency class (release, shared `best_of_us` harness)

`range_at` + `terminal_truncation_ceiling`: **3.76 ns/op** vs the STRONG
hand-rolled baseline (clamp-lerp + sqrt inline) 1.42 ns/op — **2.66× the
trivial arithmetic**, under the 3× bar and the 50 ns absolute bar. NOT a
per-token path: the seam is consulted once per decode iteration (it takes
the iteration index, not a position — structural).

## G2 — quality-at-fixed-budget (the cross-repo gate; release, 3 arms, 845 s)

Identical arm/seed/aux protocol/node COUNT — only the range schedule
differs. riir-train gate_full posture (frozen teacher fixture
`pfd_toy_teacher_v1.bin`, BLAKE3 `acb1e5e4…`):

| arm | final W1 (radial) | ring1 frac | radial std | vs flat |
|---|---|---|---|---|
| Flat (the C9 incumbent) | **0.4084** | 0.714 | 0.966 | — |
| AnnealRenorm (fixed total weight) | **0.4263** | 0.727 | 0.966 | +4.4% |
| AnnealPlain (Δ-scaled truncation) | **0.4451** | 0.745 | 0.941 | +9.0% |

- Flat reproduces C9's recorded full-scale PFD number (0.408) exactly —
  the refactor is behavior-preserving on the default posture.
- **The corollary's cost law, measured**: plain truncation drops 9.37% of
  the (T−t) mass and pays **+9.0% W1** — first-order cost ∝ mass dropped,
  the paper's "first-order lossless because w(T)=0" made quantitative on
  the toy (lossless at the margin; linear at bulk).
- AnnealRenorm stays within the not-worse bar (≤ ×1.05: measured ×1.044)
  and keeps both rings (ring2 = 0.273 > 0.25). **Regime boundary
  (honest):** the renormalized anneal is NOT a win on THIS target — the
  two-ring toy carries no fine detail for low-noise concentration to
  exploit, so the anneal trades coarse convergence speed for placement;
  the paper's fine-detail claim needs a fine-detail-bearing target (the
  dLLM C5 arm, riir-train 569). Smoke-scale (110 updates) measured the
  reverse ordering (renorm 0.4343 < flat 0.4537) — scale-dependent, both
  recorded.

## G3 — no-regression

- katgpt-core default `--lib`: **2063/0** (the T2/T4 recorded baseline,
  unchanged). `--no-default-features` check clean. clippy
  `-D warnings` clean at default / `horizon_weights` /
  `horizon_weights,critical_interval_gate` / the new root target. wasm32
  check with `horizon_weights` clean.
- riir-train-engine default `--lib`: **1650/0** (the C9 baseline,
  unchanged — the anneal arms compile away feature-off); clippy clean at
  both postures; `gate_ordering_smoke` green under the refactor.
- ⚠ Pre-existing (not this landing's): `bench_875_horizon_weights_goat`'s
  G2 bar has no debug-posture guard — it measures 0.617 vs the ≤0.5 bar
  in a DEBUG run (T1's recorded numbers are release; release passes
  2/2). The gate's documented posture is `--release`.

## G4 — allocation

`TimeAnnealRange` is `Copy`; `range_at`/both predicate fns return tuples.
TrackingAllocator: **0 allocs** over 3×1024 mixed schedule calls
(in-module, `anneal_and_truncation_are_alloc_free`).

## Validation harness note (honest provenance)

The cross-repo gate was executed in an 8-worktree redirect harness
(katgpt-rs/riir-train/riir-ai/riir-chain/riir-neuron-db/riir-infer/
riir-dapps/riir-kat/riir-auth `.w875t3` worktrees, every sibling path ref
redirected to the katgpt-rs worktree carrying this change) because the
shared main checkouts could not be reconciled mid-flight (sibling WIP +
ahead/behind divergence). The harness resolves (cargo metadata exit 0,
681 packages) and runs the REAL katgpt-core symbol (no local twin) — the
numbers above are the integrated build's. The MAIN-checkout
`--features pfd_anneal` run becomes available to any box after both
repos land and checkouts sync.

## Verdict

Stays **OPT-IN** per the no-default-consumer rule (no default-path call
site of the seam exists; promotion re-opens with one). The truncation
predicate's closed form is the durable GOAT: a caller can now PRICE any
range truncation exactly (`ε` known before skipping). The anneal's
quality verdict on the toy is "safe, not a win" — the fine-detail claim
routes to riir-train 569 C5 (the dLLM lane), which this landing unblocks
mechanically (the struct + predicate are the C5 consumption).
