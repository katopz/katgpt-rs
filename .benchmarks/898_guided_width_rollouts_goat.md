# Bench 898 — Guided width rollouts GOAT (Issue 895 T7 / Research 590)

**Status:** COMPLETE. **G1 FAILED** as pre-stated: width wins on MULTI and loses on SINGLE. **The demote condition is TRIGGERED**: the T5 direction table ties on MULTI and loses on SINGLE, so guided stays off-by-default forever (closed-negative). E9, the mass arm, G2, G3 and G4 PASS. **T9: no promotion.** Both features stay **OPT-IN**.

**Completes [Plan 095](../.plans/095_gram_width_vs_depth.goat.md)'s pending width-vs-depth G1/G3** on a stochastic host. This is the host 095 was waiting for, and no third lane is opened. Result: 095 G1 **PASS** (+14.1 pp on MULTI). 095 G3 is **not proven**: SINGLE is a tie, and the arena domains are unmeasured (see § Plan 095).

- Source: GRAM, arXiv:2605.19376, re-distilled in [Research 590](../.research/590_GRAM_Generative_Recursive_Reasoning.md) (supersedes Research 058 on the belief host only).
- Primitive: `katgpt_core::guided_width` (`a4421939f` substrate, `ae7b13320` GOAT target + two fixes found by this bench).
- Features: `guided_width_rollouts`, and `guided_width_hodge` for arm (b). Both katgpt-core, both opt-in.
- Consumer: riir-ai Issue 1008 (T8, not claimed here).

Run (release):

```bash
cargo test -p katgpt-core --release --features guided_width_hodge,sense_composition \
  --test bench_898_guided_width_rollouts_goat -- --nocapture
```

## Box state (G2 is taken on a loaded box)

M3 Max, macOS, **AC power, 99% charging, `powermode 2` (High Power)**. A sibling session ran a long 64K needle job plus concurrent cargo builds throughout. The gate prints the box state itself.

| run | loadavg (1/5/15) | free + speculative | swap used |
|---|---|---|---|
| 1 | 38.3 / 26.1 / 21.1 | 4.88 GB | 1.07 / 2.0 GB |
| 2 | 40.8 / 32.8 / 24.8 | 3.65 GB | 1.07 / 2.0 GB |
| 3–5 | 29.4 / 30.9 / 24.5 | 3.45–3.73 GB | 1.07 / 2.0 GB |

G1, G3, G4 and the mass arm are deterministic (BLAKE3-seeded) and were identical across all 5 runs. G2 numbers are from runs 3–5, with runs 1–2 shown as the heavier-load range.

## Fixture

A graph 3-colouring CSP on n = 10 vertices over 4 planted-3-colourable graphs (12–17 edges, 72–864 proper colourings). The deterministic refinement step (GRAM's proposal `u_t`) is one projected-gradient step on the soft assignment `p ∈ [0,1]^{10×3}`. The energy is edge-conflict + one-hot + clue penalties, with η = 0.2. There are no weights and no training. Decoding is argmax per vertex.

Every instance starts from the uninformed uniform assignment p = 1/3. There are 128 test and 128 train instances per family, and train and test clue sets are disjoint.

| family | class | clues | completions |
|---|---|---|---|
| **MULTI** | N-Queens / graph-colouring (GRAM's guided-wins family) | 2 | ≥ 4 (mean 49.1) |
| **SINGLE** | Sudoku (GRAM's zero-mean-wins family) | 2–5 | exactly 1 (enumeration-verified) |

Equal compute: every arm spends **N·K = 128** step calls per instance. Selection is always the decode-free T3 `latent_value`, with no reward, no decode and no oracle. The oracle columns are reported beside it.

| arm | configuration |
|---|---|
| D | deterministic 1×128 (the incumbent) |
| Z | zero-mean width 8×16: transversal ε, stagnation-gated σ (σ_max 0.25, α 2, w₀ 2), Sobol init, table-absent |
| I | Z with isotropic ε (ablation) |
| G | Z + the T5 table (r = 6 per graph, success-SVD over the TRAIN split's valid Δh), Beta posterior updated on train outcomes, κ = 1 |
| R | Z + T6 trap-kill-reallocate (`TrapReallocConfig::DEFAULT`) |

## G1 — width vs depth on BOTH families

| arm | MULTI selected | any-valid | coverage | branch-valid | SINGLE selected | any-valid | coverage | branch-valid |
|---|---|---|---|---|---|---|---|---|
| D 1×128 | 0.586 | 0.586 | 0.59 | 0.586 | **0.938** | 0.938 | 0.94 | **0.938** |
| Z 8×16 | **0.703** | 0.992 | 3.32 | 0.711 | 0.750 | 0.977 | 0.98 | 0.705 |
| I 8×16 | 0.703 | 0.992 | 3.32 | 0.712 | 0.750 | 0.977 | 0.98 | 0.705 |
| G 8×16 | 0.734 | 1.000 | **4.82** | 0.712 | 0.766 | 1.000 | 1.00 | 0.680 |
| R 8×16 | 0.656 | 0.977 | 3.12 | 0.547 | 0.703 | 0.969 | 0.97 | 0.632 |

Paired over test instances, verdict at ±2 SE:

| comparison | MULTI | SINGLE |
|---|---|---|
| **Z − D selected** (G1) | **+0.117 ± 0.049 WIN** | **−0.188 ± 0.036 LOSS** |
| E9 coverage Z − D | +2.73 distinct valid solutions | +0.04 |
| **G − Z branch-valid** (demote metric) | **+0.001 ± 0.014 TIE** | **−0.025 ± 0.012 LOSS** |
| G − Z selected (report) | +0.031 ± 0.040 | +0.016 ± 0.029 |
| G − Z coverage (report) | +1.50 ± 0.11 | +0.02 ± 0.01 |
| R − Z selected (report) | −0.047 ± 0.027 | −0.047 ± 0.019 |
| I − Z branch-valid (report) | +0.001 ± 0.001 | +0.000 ± 0.000 |

- **G1 FAIL** (pre-stated: no LOSS on either family, and a WIN on at least one). Width beats depth on MULTI, where the deterministic arm is stuck at 0.586 and width lifts coverage from 0.59 to 3.32 distinct solutions. That is GRAM's mode-collapse signature, reproduced. Width LOSES on SINGLE. Noise costs per-branch quality there (branch-valid 0.938 → 0.705), and the decode-free selector does not recover the deterministic branch even though it is in the set (oracle any-valid 0.977 vs selected 0.750). The pre-stated bar is recorded as failed and is not re-specified.
- **E9 PASS:** there is a measured delta over the deterministic arm on both families, of opposite sign. The feature is not inert.
- **Demote condition TRIGGERED:** the table is a TIE on MULTI and a LOSS on SINGLE, on the pre-stated decision metric (paired branch-valid rate). Per Research 590 this is **closed-negative: guided stays off-by-default forever**.
  - ⚠ The table **does** lift MULTI coverage, +1.50 ± 0.11 distinct valid solutions (4.82 vs 3.32), well past 2 SE. That is GRAM's own multi-solution metric.
  - It was NOT the pre-stated decision metric, so it is recorded and does **not** overturn the demote. Anyone reopening this must pre-state coverage as the metric on a NEW fixture. Re-reading this run with the other metric is not allowed.
- **The selector is the bottleneck on both families.** The oracle any-valid is 0.99 / 0.98, against selected 0.70 / 0.75. The decode-free scorer (self-consistency + convergence residual) is T3's honest limit. A learned LPRM needs training (riir-train Plan 419).
- **T6 is not a gain here.** R loses 4.7 pp to Z on both families. The flip detector reads exploration noise as a trap, and kill-and-respawn throws progress away.
- **Transversal ≡ isotropic on these fixtures** (|Δ| ≤ 0.001). The stagnation gate applies most of its noise when the deterministic update is ≈ 0, which is exactly where there is no û to project off, so P_⊥ = I. The projection is inert under this gate, not wrong.

## Plan 095 — width sweep at NK = 128 (arm Z, selected / any-valid / coverage)

| N×K | MULTI | SINGLE |
|---|---|---|
| 1×128 | 0.586 / 0.586 / 0.59 | 0.938 / 0.938 / 0.94 |
| 2×64 | 0.578 / 0.938 / 1.42 | 0.938 / 0.945 / 0.95 |
| 4×32 | 0.594 / 0.961 / 2.22 | 0.906 / 0.977 / 0.98 |
| 8×16 | **0.727** / 0.969 / 3.36 | 0.766 / 0.984 / 0.98 |
| 16×8 | 0.570 / 0.953 / 3.80 | 0.773 / 0.992 / 0.99 |
| 32×4 | 0.375 / 0.906 / 3.36 | 0.898 / 0.953 / 0.95 |

- **095 G1 (width improves ≥ 10 pp on any domain): PASS.** MULTI goes from 0.586 to 0.727 at 8×16, +14.1 pp.
- **095 G3 (width ≫ depth on ≥ 2 of 3 domains): NOT PROVEN.** The bench's pre-stated `≥` rule reads 2/2, but SINGLE counts only through a **tie** (2×64 matches 1×128 at 0.938, and every wider split is lower). On 095's literal "≫" reading, it is 1 of 2 measured domains. The third domain (Go / Bomber / FFT arenas) is unmeasured here.
- So Plan 095 moves from GOAT PENDING 1/3 to **2/3**: G2 (prior), and G1 via this bench. G3 still waits for a stochastic arena domain (riir-ai Issue 1008 T5 is the next lane).
- The oracle any-valid column shows width's coverage law holding on both families. At K = 4, the selector and the too-shallow per-branch depth, not coverage, are what give way.

## Mass-conservation arm (T1 b, `guided_width_hodge`)

| check | result |
|---|---|
| `belief_mass_divergence(ε) == 0.0` exactly, grid 12×12 (264 edges), 500 seeds × σ ∈ {1e-4, 0.01, 0.25, 0.7, 2.0} | **PASS: 0 non-zero of 2500 draws** (bitwise, by construction) |
| cochain host `h ← h − 0.05·Δ₁h`, 6×12 rollouts: max \|δ₁h_b − δ₁h₀\| over branches | **PASS: mass arm 1.34e-7** (f32 rounding) vs transversal arm 8.63e-1 |

The first run read **1.28e-1** for the mass arm. The Sobol init offset and `apply_kick`'s isotropic kick were not divergence-free. Both are now drawn by the arm (`Perturbation::owns_init` / `admits_kick`, fix in `ae7b13320`). This is an architectural invariant only. There is no quality claim, because nothing measured here says belief failures involve mass drift (Research 590 § Caveats).

## G2 — latency (belief host, d = 8, real `evolve_belief`, release)

| measure | bar | runs 3–5 | runs 1–2 (heavier load) |
|---|---|---|---|
| best-of decision N = 8, K = 16 | ≤ 50 µs | **14.88–14.96 µs PASS** | 17.88–18.38 µs |
| t(K = 32) / t(K = 16) (O(K)) | [1.5, 2.5] | **2.00 PASS** | 1.86–1.89 |
| t(N = 16) / t(N = 8) at K = 16 (report) | — | 2.07–2.18 | 2.03–2.06 |
| paired width 8×16 / depth 1×128 (`ab_median_ratio`, report) | — | depth/width 0.097–0.098 (width ≈ 10.2× depth wall) | 0.087–0.098 |

- `black_box` wraps the state, the config and the result.
- The equal-compute wall ratio is ~10×. That is not width overhead in the sense of the paper. `evolve_belief` is a ~13 ns leaky step (1.65 µs / 128). Each of the 112 noisy branch-steps adds one BLAKE3 ε draw and the projection, ~150 ns amortised including the Sobol init and the scorer. On a host whose step costs ≥ 1 µs the ratio tends to 1. Report only.
- **O(K) parallel latency** is structural: branches share nothing before selection, so the dependency depth is K. The serial cost is N·K steps plus the O(N²d) scorer, as the t(N=16)/t(N=8) ≈ 2 reading shows.

## G3 — kill switch

- **PASS:** N = 1 and σ_max = 0 are `to_bits`-identical to the incumbent K×`evolve_belief()` loop over 256 random states × both configs, with K = 1–24. There were 0 mismatches.
- **PASS:** table-absent is identical to an empty table, with 0 differences over 64 decisions. It also matches a dimension-mismatched table (unit test).

## G4 — allocations

**PASS: 0 allocations over 1000 steady-state decisions.** Each decision is a belief host decision with a table, trap-realloc, the frozen scorer direction, the posterior update, the farthest-point returned set, plus a mass-arm cochain rollout with the circulation probe.

The first run read **378 allocs/decision**. `SobolQmc::new_multi` allocates in its primitive-polynomial search (`prime_factors_u64`). The fix is a new `SobolQmc::reseed`, pinned bit-identical to `new_multi`, with the scratch caching one source (`ae7b13320`).

## T9 — promotion verdict

**No promotion. `guided_width_rollouts` and `guided_width_hodge` stay OPT-IN.**

- The guided (T5 table) path is **closed-negative per the pre-stated demote condition**. It stays off-by-default forever, and Research 058 §8.3's "do NOT make guided noise the default" stands, now with a measurement behind it.
- The zero-mean width path also fails the promotion rule. It has a modelless gain on MULTI only, and loses on SINGLE.
- The arm (b) invariant, the kill switch, zero-alloc and latency all pass. These are substrate properties, recorded so a consumer can rely on them.
- The consumer lane (riir-ai Issue 1008) owns any host-specific re-test. It must run both families with its own pre-stated metric.
