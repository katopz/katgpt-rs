# Bench 896 — activation-diagonal weight-quant fit GOAT (Issue 886 P0 + P1 modelless half)

**Status:** COMPLETE (modelless half) — G1 RECORDED (synthetic; sign **positive at both tiers**, see the caveats — the honest prior is NOT settled by this fixture) · G2 PASS · G3 PASS (uniform diagonal ⇒ payload bytes identical, pinned) · G4 PASS. Features `act_channel_moments` (katgpt-core) and `act_aware_fit` (katgpt-types, forwarded by katgpt-core) stay **OPT-IN** — no production weight-quant authoring consumer exists, and the deciding G1 (per-family conditional retention walk on a real checkpoint) is model-bound: **riir-infer Issue 014**.

**Box:** M3 Max (64 GB), macOS, **AC power, 100% charged, `powermode 2` (High Power)**, **LOADED**: load averages 14.4–18.7 across the three runs (concurrent sibling-agent cargo builds), swap 1.08 / 2.0 GB used, ~2.7 GB free pages + ~23 GB inactive (vm_stat, 16 KB pages). Release profile. G2 via the shared interleaved `ab_median_ratio` (15 rounds × 2 iters/arm) and `best_of_us`; `black_box` on results AND arguments. G1 is fully deterministic (fixed xorshift seeds) — identical across the three runs.

```
cargo test -p katgpt-core --features act_channel_moments,act_aware_fit \
  --release --test bench_896_act_diagonal_quant_fit_goat -- --nocapture
```

## What shipped

- **P0 `katgpt_core::act_channel_moments`** — `ActChannelMoments::new(&widths)` (per-layer input widths), `observe(layer, x)` / `observe_batch(layer, xs)`: per-input-channel `{Σ|x|, Σx²}` in f64 + count, alloc-free, non-finite ⇒ panic (the 883 poison law). `freeze()` → `ActChannelDiagonal { mean|x| (AWQ s_X), E[x²] (imatrix weight) }` with a **BLAKE3 commitment over a canonical little-endian image** (`KACD` magic, version 1, per-layer (width, count), then the f32 moments); `to_bytes` = image ‖ digest, `from_bytes` refuses truncation / trailing bytes / negative-or-non-finite moments / digest mismatch (every single-byte flip refused, unit-tested; digest pinned). Joins the `StaticCalTable` commit family but spelled platform-independently (StaticCalTable hashes native-endian bytes). Weight-binding = the caller's `calibration_staleness::SnapshotBound<ActChannelDiagonal>`.
- **Co-collection with the 883 pass:** same tapped forward, different tap points — 883 taps K/V projection OUTPUTS, this wants linear INPUTS; 883's grand Σx² is over K/V and is not a substitute (documented in the module). No code hook needed beyond calling `observe` beside `LayeredVkCalibration::observe_layer`.
- **P1 `TernaryGroupWeights::quantize_from_f32_act_aware(w, rows, cols, diag: &[f32], fit)`** (katgpt-types; the diagonal is a plain slice — no katgpt-core dep). The quantizer core was refactored to take a per-group scale closure (`quantize_with_group_scale`); the baseline / `pot_scales` / act-aware arms differ ONLY in that closure, so the payload format, f16 store, `threshold = 0.5·s` and carry loop are kernel-identical. Two fits:
  - `WeightedMeanAbs` — `s = Σu|w| / Σu`, `u = h / max_group(h)` (division, so a uniform `h` gives `u ≡ 1.0` exactly ⇒ **G3 bit-identity**).
  - `WeightedSearch` — imatrix class: 21-point grid `s_wma × {0.50 … 1.50}` + weighted least-squares refit, each candidate evaluated as its f16-rounded value **through the real carry loop** on `Σu(w − s·q)²`; the ×1.0 anchor is evaluated first and replaced only on strict improvement (never worse than `WeightedMeanAbs` on the weighted objective, unit-tested group by group). At a uniform diagonal this is the **activation-blind search** — the control that separates "search helps" from "the diagonal helps".

## G1 — held-out output reconstruction error `E‖(W−Ŵ)x‖² / E‖Wx‖²`

Fixture: W 256×1024 (8 groups/row), σ=0.02 Gaussian or Laplace; x_j = σ_j·z + μ_j with 512 calibration samples (→ `ActChannelMoments`) and 256 **independent held-out** samples for the metric. Δ% vs the activation-blind mean-abs baseline (− = better). The INT4 rows are a **bench-local reference** (symmetric `q∈[-8,7]`, f16 group-128 scale, RTN `absmax/7` baseline + the same weighted grid/LS search) — katgpt-types ships no 4-bit weight container.

| W | activation dist | ternary base | WMA[E x²] | WMA[mean\|x\|] | Search[E x²] | Search[mean\|x\|] | Search[blind] | INT4 RTN | INT4 Search[blind] | INT4 Search[E x²] | INT4 Search[mean\|x\|] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gauss | D0 uniform (control) | 0.44357 | +0.10% | +0.13% | −26.17% | −26.34% | −26.34% | 0.013941 | −26.57% | −26.55% | −26.59% |
| Gauss | D1 1% heavy ×20, zero-mean | 0.43207 | −36.67% | −12.27% | **−66.39%** | −49.60% | −26.46% | 0.013546 | −30.55% | **−72.46%** | −59.54% |
| Gauss | D2 1% heavy ×20, mean 0.5σ | 0.40400 | −39.15% | −12.88% | **−68.29%** | −50.86% | −27.14% | 0.013410 | −32.69% | **−71.08%** | −59.25% |
| Gauss | D3 log-normal σ=e^N(0,1) | 0.43510 | −3.26% | −2.11% | **−36.66%** | −35.38% | −26.34% | 0.013616 | −24.58% | **−44.53%** | −40.61% |
| Laplace | D0 uniform (control) | 0.58385 | −0.11% | −0.09% | −20.75% | −20.66% | −20.93% | 0.025997 | −25.95% | −25.89% | −25.97% |
| Laplace | D1 1% heavy ×20, zero-mean | 0.57710 | −45.22% | −14.54% | **−68.59%** | −47.00% | −20.86% | 0.025646 | −30.88% | **−66.65%** | −54.52% |
| Laplace | D2 1% heavy ×20, mean 0.5σ | 0.54439 | −46.70% | −12.33% | **−70.69%** | −47.24% | −20.77% | 0.024171 | −26.42% | **−62.64%** | −51.16% |
| Laplace | D3 log-normal σ=e^N(0,1) | 0.58120 | −5.25% | −2.09% | **−31.61%** | −27.60% | −19.77% | 0.026385 | −27.39% | **−44.46%** | −41.20% |

**Decomposition (the diagonal's OWN contribution = Search[E x²] ÷ Search[blind] − 1):** Gaussian D1 ternary −54.3% / INT4 −60.3%; D3 ternary −14.0% / INT4 −26.5%; D0 control ternary +0.23% / INT4 +0.03% (the empirical 512-sample diagonal is not exactly uniform — noise-level, both signs). `E[x²]` (imatrix) beats `mean|x|` (AWQ's s_X) as the fit weight in every non-control row, consistent with `E[x²]` being the diagonal of the output-error quadratic form.

### Findings

1. **The activation-BLIND scale rule leaves ~20–27% on the table by itself** (Search[blind] row, D0 control included): mean-abs is not the MSE-optimal ternary group scale for dense Gaussian/Laplace weights under the carry loop, and a one-shot 21-point grid recovers it without any activation data. This is a finding about `quantize_from_f32` independent of Issue 886.
2. **The diagonal adds a large further gain on planted heavy channels** (−54% over blind search at ternary, D1) and a moderate one on a smooth log-normal spread (−14%). The INT4 reference gains more in the spread case (−26.5%), directionally consistent with the prior "4-bit is where it pays" — but the ternary gain is **not** small on this fixture.
3. **Why this does NOT refute the honest prior (read before citing any number above):**
   - The fixture is **PTQ-to-ternary of dense-float weights** (baseline relative output error 0.44 — ternary is a catastrophic quantizer for Gaussian weights). The production ternary lane (Ternary-Bonsai) is **born-ternary**: its codes were trained, so the only free parameter is the group scale of already-ternary weights. That is a different, much smaller lever.
   - The planted distributions (1% × 20 channels, independent Gaussians) are a **best case** for a diagonal fit; real LLM activations have correlated channels (the off-diagonal the diagonal approximation drops) and the Bonsai lane already applies a Hadamard rotation (`bonsai2_hadamard`), which spreads outlier channels — precisely the rotation lever QuIP#/AQLM say dominates per-channel scaling at ≤3 bits. After rotation the diagonal is much flatter, i.e. closer to the D0 control row, where the diagonal contributes nothing.
   - Held-out reconstruction MSE is not the deployed-path metric. The lossy-surface law requires the per-family conditional retention walk (riir-ai Bench 948 pattern); aggregate metrics can be flat while families flip.
   So the modelless half is **mechanism-verified** (the fit does what it says and the diagonal carries signal when the signal exists), not **gain-verified** on a deployed surface. That is riir-infer Issue 014.
4. `WeightedMeanAbs` alone is weaker than the search (it only re-weights the mean) but is closed-form, +12.5% cost, and the G3 anchor.

## G2 — cost

| quantity | run 1 | run 2 | run 3 | bar |
|---|---|---|---|---|
| G2a `observe_batch` ns/element (width 4096, 64 rows, best-of) | 0.276 | 0.318 | 0.316 | ≤ 2.0 ✓ |
| fit ns/weight — baseline (256×4096, best-of) | 6.125 | 6.902 | 6.883 | — |
| fit ns/weight — WMA | 6.722 | 7.771 | 7.737 | — |
| fit ns/weight — Search | 138.0 | 152.8 | 147.2 | — |
| G2b refactored / legacy baseline (`ab_median_ratio` median) | 1.0042 | 1.0052 (0.866..1.319) | 0.9958 (0.910..1.046) | ≤ 1.05 ✓ |
| G2c WMA / baseline (median) | 1.1249 | 1.1268 (1.038..1.240) | 1.1248 (1.020..1.300) | ≤ 1.50 ✓ |
| G2d Search / baseline (best-of ratio) | 22.54 | 22.14 | 21.39 | ≤ 46 ✓ |

G2b is the shipped-path no-regression check: the closure refactor of the quantizer core is within ±0.5% of a verbatim transcription of the pre-refactor loop (payload also byte-identical, G3). Search ≈ 22× the baseline = its 23 carry passes per group; at ~0.15 µs/weight it re-fits 1 B weights in ~2.5 CPU-minutes, one-shot.

**Comparator note — riir-train `ZeroQatCalibrator`** (Plan 255 Ph4, same insertion point, GD-family): central finite differences on an injected loss, `scale -= lr·∇`, 100 steps × 2 loss evaluations per scale, lr/convergence knobs. Its per-scale cost is 200 loss evaluations of whatever the loss is (a layer or model forward) — not measured here (no riir-train dep, by boundary). The closed-form fit here is one moment pass + ≤ 23 carry passes per group, deterministic, no loss function, no forward, BLAKE3-committable input. The accuracy comparison against ZeroQAT is part of riir-infer Issue 014's retention walk.

## G3 — bit-identity at a uniform diagonal

`ActChannelDiagonal::uniform(&[1024], 0.8, 0.64, 512)` round-tripped through `to_bytes`/`from_bytes`; both its `mean_sq` and `mean_abs` slices fed to `WeightedMeanAbs` ⇒ payload bytes (pos planes ‖ neg planes ‖ f16 scale bits, 69 632 B) **identical** to `quantize_from_f32`; and the refactored baseline is identical to the pre-refactor transcription. Unit-pinned too (`ternary_group_act_aware::tests`: six uniform values incl. inexact-reciprocal ones, a trailing partial group, the all-zero-diagonal fallback, plus an FNV-64 payload digest `0x2d77e077501d6b6c` pinned by value).

## G4 — allocations

`observe` ×64 + `observe_batch` (64 rows) + a second-layer `observe`: **0 allocs**. `quantize_from_f32_act_aware` (both fits): **3 allocs = the baseline's 3** (the output container; the search's scratch is a `[f32; 128]` stack buffer).

## Not done here (and where it lives)

- Model-bound G1 (real checkpoint diagonal → re-authored ternary → per-family conditional retention walk vs mean-abs and the ZeroQAT class): **riir-infer Issue 014**.
- P2 (AWQ α-rescale spelling, GGUF-writer recipe, LoTA merge weighting, outlier-collapse tripwire): deferred until a consumer opens (Issue 886 P2).
