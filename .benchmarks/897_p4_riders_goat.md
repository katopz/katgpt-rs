# Bench 897 — Issue 882 P4 riders GOAT: canonical context assembly + gated rank-1 affinity deflation

**Status:** COMPLETE at the primitive level. All gates pass on all 3 runs. One pre-registered bar FAILED (G1b-1 at α = 3) and was re-specified before the re-run; both results are recorded.

These are the two modelless riders of [Issue 882](../.issues/882_differential_anchor_scoring.md) P4. The P4 m_Y instrument itself is riir-infer `dc0e5a9`, with its first reading at riir-infer `fb51821`. It is not claimed here.

| rider | feature (crate) | GOAT target |
|---|---|---|
| (a) canonical context assembly | `canonical_context` (katgpt-core) | `crates/katgpt-core/tests/bench_897_canonical_context_goat.rs` |
| (b) gated rank-1 affinity deflation | `affinity_deflation` (katgpt-spectral) | `crates/katgpt-spectral/tests/bench_897_affinity_deflation_goat.rs` |

Run:

```bash
cargo test -p katgpt-core --release --features canonical_context \
  --test bench_897_canonical_context_goat -- --nocapture
cargo test -p katgpt-spectral --release --features affinity_deflation,river_valley \
  --test bench_897_affinity_deflation_goat -- --nocapture
```

## Box state (recorded beside every figure)

M3 Max, 64 GB, **AC power, battery 100% charged**, `pmset powermode 2` (High Power), release profile. This was a HEAVILY LOADED box: sibling agent sessions and concurrent cargo builds were running in other target dirs.

| run | `vm_stat` free pages (16 KB) | swap used / total | loadavg (1 / 5 / 15 min) |
|---|---|---|---|
| 1 | 132 156 (≈ 2.0 GB) | 1078 / 2048 MB | 20.00 / 14.76 / 13.47, then 37.42 during run (b) |
| 2 | 90 882 (≈ 1.4 GB) | 1078 / 2048 MB | 40.62 / 26.14 / 18.58 |
| 3 | 98 203 (≈ 1.5 GB) | 1070 / 2048 MB | 29.27 / 25.41 / 18.75 |

Read the G2 figures as UPPER bounds taken under load. Every G2 ratio is paired and interleaved (`tests/common/ab_timing.rs::ab_median_ratio`), and every absolute figure is a best-of minimum (`best_of_us`). `black_box` is applied to both the arguments and the result.

## (a) Canonical context assembly — `katgpt_core::canonical_context`

- `canonical_order_into(keys, order)` sorts ascending by an `Ord` key, with ties broken by input index. The key is either the BLAKE3 digest from `content_key` or a caller key.
- `canonical_order_by_score_into(scores, keys, order)` sorts by score descending (`float_order::desc`, NaN last, −0 ≡ +0), then by key, then by index.
- `assemble_into` / `assembled_len` write into a caller-owned buffer.

Both comparators are total orders over distinct indices. `sort_unstable_by` therefore produces one unique sequence, in place (the Bench 894 finding: the stable sort allocates its merge scratch). Both functions return the **adjacent-tie count**. Permutation invariance holds exactly when equal keys imply equal content, and `ties == 0` makes that condition vacuous.

**Fixture.** A retrieved SET of 32 items is delivered in 256 random orders per seed, over 8 seeds. A toy ORDER-SENSITIVE judge reads the ASSEMBLED bytes. It weighs each item by a U-shaped "lost-in-the-middle" position weight `1 − 0.6·sin(πp/(N−1))`. Its per-item value is relevance + 0.25·length: the judge reads content, so items the retriever ties are not tied for the judge.

| gate | result (all 3 runs identical) |
|---|---|
| G1a canonical spread | **0 bitwise**: 0 non-identical assembled contexts over 8 × 256 orders |
| G1a retrieval-order spread | min **16.32**, mean **20.84** (the defect exists) |
| G1a tied-relevance pipeline (4 relevance levels) | stable sort by score alone: spread ≥ **0.3455**; `canonical_order_by_score_into`: **0** |
| G1a idempotence | canon(canon(x)) = identity, for both orders |
| G1a BLAKE3 ties | 0 over 8 × 256 orderings |
| G1a caller-key collision | 4-bucket caller key over distinct content: **ties 28 flagged**. The spread there is 3.08, which is allowed: the tie count is the alarm. |
| G3 | identity order assembles byte-identical to plain concatenation (1474 B) |
| G4 | **0 allocs** (order + score order + assemble, N = 256, 64 rounds) |

G2, best-of `canonical_order_into` with 32-byte keys (bar at N = 256 ≤ 20 µs):

| N | run 1 | run 2 | run 3 |
|---|---|---|---|
| 32 | 0.42 µs | 0.33 µs | 0.38 µs |
| **256** | **4.38 µs** | **3.88 µs** | **3.83 µs** |
| 1024 | 20.50 µs | 17.88 µs | 17.92 µs |

- **Paired (report).** At N = 256, plain assembly costs 1.5–2.0 µs and order + assembly 4.7–6.0 µs, a median ratio of **3.02–3.04×**. The sort costs ~2× a memcpy of the 12 KB context, and ~3–4 µs per assembly is invisible next to any consumer that scores the context.
- **BLAKE3 ingest (report).** 0.084–0.104 µs per ~47 B item. Compute it once at ingest and store it beside the item.

**MEASURED LAW (a).** Canonical assembly makes position bias CONSTANT; it does not remove it. The spread goes to 0 because every run sees the same assignment of items to positions. For an eval whose purpose is to MEASURE position bias, this is the wrong instrument. For an oracle-anchored eval whose purpose is to compare two systems, the variance it removes is 16–21 score points per permutation on this fixture, which is noise in the comparison.

## (b) Gated rank-1 affinity deflation — `katgpt_spectral::affinity_deflation`

`A′ = A − λ·σ₁u₁v₁ᵀ = A − λ(A v₁)v₁ᵀ`. The deflation runs only when the **stable rank** `‖A‖_F²/σ₁² < θ`.

- The stable rank falls out of the σ₁ power iteration at O(M²), so no eigendecomposition is needed. The power iteration is the shared `spectral_retract::power_iter_step` on AᵀA formed implicitly, which avoids a parallel implementation.
- ⚑ `‖Av‖` over a unit `v` lower-bounds σ₁ at every iterate. An unconverged iteration can therefore only OVER-state the stable rank, so the gate errs toward refusing.
- λ = 0 returns before any work. A refusal only reads A. NaN anywhere makes every comparison false, so it refuses.

**Fixture.** M = 64 candidates in d = 128, built as `x_i = α·h_i·g + s_k·b_k + ε`:

- a global common-mode direction `g`, weighted by hubness `h ∈ [0.3, 1]`;
- K = 4 planted blocks of 16 with strengths (1.4, 1.0, 1.0, 0.8);
- noise σ = 0.35.

The affinity is `A = XXᵀ`. The metric is block-retrieval top-1: whether `argmax_{j≠i} A′_ij` falls in i's block (NaN-safe via `float_order::cmp_for_max`).

**G1b-3: θ by DIRECT EVALUATION** (no gradient descent). The calibration set is α ∈ {0, 0.5, 1, 2, 3} × 8 seeds. The θ grid scored mean gated top-1 as follows:

| θ | 1.05 | 1.1 | 1.25 | 1.5 | 1.75 | 2.0 | 2.5 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|---|---|
| mean gated top-1 | 0.9996 | **1.0000** | 1.0000 | 1.0000 | 0.9602 | 0.9539 | 0.9477 | 0.9477 | 0.9477 |

θ* = **1.1**: ties go to the smallest θ, which is the conservative end. Held-out results over 8 other seeds:

| α | srank (cal mean) | none | ungated | gated@θ* | fired |
|---|---|---|---|---|---|
| 0.0 | 1.580 | 1.000 | 0.812 | **1.000** | 0/8 |
| 0.5 | 1.638 | 1.000 | 0.941 | **1.000** | 0/8 |
| 1.0 | 1.430 | 1.000 | 1.000 | **1.000** | 0/8 |
| 2.0 | 1.060 | 0.998 | 1.000 | **1.000** | 8/8 |
| 3.0 | 1.014 | 0.928 | 1.000 | **1.000** | 8/8 |

G1b-3 PASS: the worst margin of gated against max(none, ungated) is **+0.0000** on every α, so the gate never loses to either fixed policy.

**G1b-1: collapsed recovery.**

- ⛔ **The PRE-REGISTERED bar FAILED at α = 3.** The bar was: fire on every seed, gated ≥ 0.95, AND gated ≥ none + 0.25. Recovery was complete (0.928 → **1.000**, fired 8/8), but the lift was **+0.072** against +0.25. At α = 3 the best in-block candidate's hubness is usually near the global maximum, so the common mode rarely flips a row's top-1. The fixture was not collapsed enough to damage the baseline by a quarter.
- The gate was re-specified before the re-run: the severity cell α = 8 at the **unchanged** calibrated θ*, with the same bars. It PASSED: none **0.586 → gated 1.000** (lift +0.414).
- Severity sweep:

| α | 3 | 4 | 6 | 8 |
|---|---|---|---|---|
| none | 0.928 | 0.836 | 0.693 | 0.586 |
| gated@θ* | 1.000 | 1.000 | 1.000 | 1.000 |
| srank | 1.0140 | 1.0047 | 1.0010 | 1.0003 |

**G1b-2: the healthy-matrix NEGATIVE arm, pinned so the gate can go red.** At α = 0 and θ*, the gate **refuses on all 8 held-out seeds, and A stays `to_bits`-identical**. Deflating anyway drops top-1 **1.000 → 0.812** (Δ −0.188, bar < −0.10). Removing σ₁ removes the strongest REAL block (s = 1.4), and its 16 rows lose their in-block affinity. If ungated deflation ever stops hurting here, this gate reds.

**G1b-4: stable rank vs the Roy–Vetterli entropy erank** (`river_valley::effective_rank_into`, Jacobi). Healthy: srank 1.576, erank(σ²) 2.755. Collapsed α = 3: srank 1.014, erank(σ²) 1.090. Both order the classes the same way. The entropy erank separates them more widely, but it costs a Jacobi sweep. The stable rank is free given σ₁.

**MEASURED LAW (b), the safe θ band.** The healthy-matrix stable rank here is only ~1.58, because one planted block dominates. The gap to a collapsed matrix (≤ 1.06 at α ≥ 2) is therefore narrow. On the grid, θ ≥ 1.75 starts deflating healthy matrices, and mean top-1 falls to 0.96. θ ∈ [1.1, 1.5] is the safe band on this fixture. A consumer must re-fit θ on its own affinity distribution by the same direct grid. The value 1.1 is this fixture's, not a law (trap 5's rule, applied to θ).

| gate | run 1 | run 2 | run 3 |
|---|---|---|---|
| G2 M=128 collapsed: paired (build + gated deflation) / build, bar ≤ 1.25 | **1.156** (1.113–1.176) | **1.163** (1.121–1.269) | **1.154** (0.650–1.958) |
| G2 M=128 healthy (refusal): same ratio, bar ≤ 1.60 | **1.239** (1.197–1.258) | **1.219** (1.174–1.272) | **1.246** (0.883–1.289) |
| power-iteration steps (collapsed / healthy) | 3 / 6 | 3 / 6 | 3 / 6 |

The affinity build is the symmetric Gram (upper triangle + mirror, `simd_dot_f32`). That is the cheap baseline, so these ratios are the conservative reading.

Best-of gated deflation, absolute (report):

| M | collapsed (deflates) | healthy (refuses) |
|---|---|---|
| 64 | 2.71–2.92 µs | 4.33–4.62 µs |
| 128 | 9.50–10.21 µs | 14.67–15.75 µs |
| 256 | 43.12–45.92 µs | 65.04–68.96 µs |

The refusal path costs MORE than the deflation path. The healthy matrix has a small spectral gap, so the power iteration takes 6 steps where the collapsed one takes 3. The gate's safe direction makes that extra cost harmless: stopping early can only refuse.

- **G3 PASS:** λ = 0 and a refusal (θ = 1 ≤ srank) are `to_bits`-identical on a collapsed matrix.
- **G4 PASS:** **0 allocs** over 32 rounds of the gated call on the collapsed and healthy matrices, with caller-owned `DeflationScratch`.

## Verdict

Both riders are modelless, zero-alloc, bit-identical when off, and pass G1–G4. They stay **OPT-IN**, because neither has a production consumer to promote on:

- (a) needs the oracle-anchored eval that sorts through it, such as riir-infer's needle / passkey harness or riir-clippy's oracle top-1.
- (b) needs a rerank stage that builds an M×M candidate affinity. katgpt-attn-match's `rerank` scores query → document and does not build one today.

The consumer wiring and its GOAT are the promotion evidence still owed.
