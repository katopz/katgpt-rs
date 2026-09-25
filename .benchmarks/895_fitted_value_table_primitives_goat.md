# Bench 895 — fitted token-value table primitives (Issue 883 P1–P4, primitive half)

**Status:** P1/P2/P3 primitives LANDED, **OPT-IN**. G1/G3/G4 pass on every arm. **Two G2 bars FAILED and are recorded as failed:** P1's fused dequant+restore costs +4.8–5.2% against a ≤ +1% bar, and P3's reconstruct-from-K costs 14–15× against a ≤ 1.00× bar. P2 G2 passes. This bench makes no model-level quality claim; that half is riir-infer Issue 013.

- **Issue:** [883](../.issues/883_fitted_value_anchor_tables.md) · **Research:** [587](../.research/587_Memory_Attention_Fitted_Token_Value_Tables.md) · **Substrate:** Bench 886 (`fitted_anchor_table.rs`, P0) · **Model-bound G1:** riir-infer Issue 013
- **Features:** `fitted_value_tables` (P1+P2, implies `fitted_anchor_tables`) · `fitted_v_reconstruct` (P3, implies `fitted_value_tables` + `position_group_action`). katgpt-kv forwards `fitted_value_tables` so P1 is gated against the real KVarN backend.
- **Targets:**
  - `crates/katgpt-kv/tests/bench_895_mean_removed_v_quant_goat.rs` (P1; required-features `kvarn`, `fitted_value_tables`)
  - `crates/katgpt-core/tests/bench_895_fitted_value_table_goat.rs` (P2/P3/P4; required-features `fitted_v_reconstruct`)
- **Module:** `crates/katgpt-core/src/fitted_value_table.rs`

## Box state (G2 law — every perf figure below was taken under this)

M3 Max, 64 GiB, **AC power, battery 100% charged, `powermode 2` (High Power)**. Load average was **12.2–16.5** across all three runs, with other agent sessions running concurrently. Free RAM was 2.0–2.3 GiB by `vm_stat` (128k–153k free 16 KiB pages), and swap was 1078 / 2048 MiB used. That is a loaded box, so every G2 figure is a paired interleaved median from `tests/common/ab_timing.rs` `ab_median_ratio`, `--release`, with `black_box` on both arguments and results. Each G2 ran 3× and all three medians are quoted.

## What shipped

| Task | Primitive | Kill switch / fallback |
|---|---|---|
| table | `FittedTokenTable::from_calibration(&LayeredVkCalibration, VkSignal::{ValueMean,Residual}, λ_js)`. It freezes the P0 streaming tables with James–Stein shrinkage. `row(layer, tok) -> Option<&[f32]>` | untracked token → `None` → the consumer's plain path |
| **P1** | `MeanRemovedValueCache<C: QuantizedKVCache>`: a decorator over ANY in-tree KV backend (KVarN, TurboQuant, …). It stores `q(V − E^V_l[s])` and reads `dequant + E^V_l[s]`. The fused epilogue is `accumulate_value` / `axpy_mean_restored` | miss / all-`+0.0` table ⇒ bit-identical to the backend |
| **P2** | `v_from_k_plus(k, row, λ, out)` = `K + λ·E_l[s]` | λ = 0 / miss ⇒ a COPY of K (the `V := K` path); the multiply never runs |
| **P3** | `reconstruct_v_from_rope_k(rope, p, K̂, row, λ, out)` = `G(−θp)·K̂ + λE`. It does ONE `PositionGroupAction::apply_inverse_at` per head and never round-trips (trap 2). `read_v(VReadPath, …)` | `VReadPath::FullCache` ⇒ a bit-identical copy of the stored V |
| **P4** | `v_projection_flops` (ΔF_V), `storage_bytes` (P(K)), `v_cache_fraction` (n_v/(n_kv+n_v)) | — |

Substrate-first: the KV quantizer is **consumed, not rebuilt**. The decorator sits on `katgpt_types::QuantizedKVCache`, which 9 in-tree backends implement. The rotation is **consumed** via `RopeAction::apply_inverse_at` (`position_group_action`), and the table source is the P0 `LayeredVkCalibration`. A new frozen-table type was needed because the P0 substrate is an accumulator: `EngramTable` is n-gram-addressed, not (layer, token)-addressed.

## P1 — token-mean-removed V quant over KVarN

Fixture: 64 Zipf(1) tokens, kv_dim 128, planted per-token means `μ_s ~ N(0, σ_μ²)` centred under the sampling law, plus `N(0,1)` noise. The table is fitted on a 100k-token calibration split by the P0 substrate. The held-out split is 4096 positions, which is 32 full KVarN tiles of 128.

**G1a — dashboard prediction vs measurement (the issue's arbiter).** The prediction is `removed/plain quant-MSE = 1 − ρ_V`, with ρ_V the calibration dashboard aggregate. The argument is that KVarN's RTN is scale-equivariant, so on Gaussian rows MSE ∝ row variance. The tolerance is ±10% relative. **9/9 PASS.**

| ρ target | ρ_V (dashboard) | predicted | bits=2 | bits=4 | bits=8 |
|---|---|---|---|---|---|
| 0.25 | 0.2414 | 0.7586 | 0.7535 (−0.7%) | 0.7628 (+0.5%) | 0.7615 (+0.4%) |
| 0.50 | 0.4611 | 0.5389 | 0.5412 (+0.4%) | 0.5512 (+2.3%) | 0.5522 (+2.5%) |
| 0.75 | 0.7293 | 0.2707 | 0.2708 (+0.0%) | 0.2839 (+4.9%) | 0.2834 (+4.7%) |

The 4/8-bit arms (Sinkhorn var-norm on) drift toward under-delivering as ρ grows: +4.7–4.9% at ρ = 0.75, still inside the tolerance. The 2-bit arm (grouped RTN, no var-norm) tracks the prediction to <1%.

**G1c — the absmax caveat (REPORT, recorded whichever way it went).** The fixture token is bimodal: 75% of its occurrences carry a +20σ spike on 4 channels and 25% do not, so the fitted mean sits between the two modes.

| bits | overall (pred 0.4139) | spike rows | **no-spike rows** | other tokens |
|---|---|---|---|---|
| 2 | 0.4156 (+0.4%) | 0.171 | **3.795** | 0.490 |
| 4 | 0.3951 (−4.5%) | 0.440 | 0.867 | 0.364 |
| 8 | 0.3963 (−4.3%) | 0.455 | 0.856 | 0.364 |

The caveat is **real at 2 bits**. The off-mean occurrences' quant MSE grows **3.8×**, because mean removal hands them a −15σ offset and grouped RTN scales by range. At 4/8 bits KVarN's per-channel Sinkhorn scaling absorbs it, and those rows still improve (0.86×). The aggregate stays on the prediction in every cell. The per-occurrence regression is exactly the kind of thing the aggregate hides, which is why riir-infer's per-family retention walk exists.

**G1d — trap 3 (sinks).** A consistent +20σ sink sits at every tile start. **3/3 PASS.** Sink rows drop to 0.05–0.09× MSE, because the table absorbs the sink mean rather than exempting it. Non-sink rows in sink-bearing tiles are 0.53×, so neither population regresses. `kv_sink_window` needs no special case for this: the sink is simply a tracked token.

**G3 — bit-identity:** 4/4 PASS. The miss table and the E=0 table each give 0 differing elements of 524,288, at bits 2 and 4. **G4:** PASS. The backend loop allocates 0, the decorated loop allocates 0, Δ = 0 over 4096 stores + 8192 reads.

**G2 — the decode V-aggregation kernel** (T=4096, kv_dim 128, 4-bit; dequant + weighted accumulate). The bar is the issue's ≤ +1% kernel time.

| run | fused `accumulate_value` / plain | per-round range | two-pass read (report) |
|---|---|---|---|
| 1 | **1.0515** | 0.93–1.64 | 1.1227 |
| 2 | **1.0475** | 0.67–1.13 | 1.1426 |
| 3 | **1.0477** | 0.90–1.09 | 1.1272 |

**FAIL, 3/3: +4.8–5.2%.** The plain arm costs ~118 ns per 128-wide row, i.e. 483–520 µs per 4096-position pass. The fused epilogue adds one load and one add per element, from a Zipf-hot row in L1/L2, on top of that. Fusing into the axpy halves the two-pass cost (+12–14%), but it does not reach +1%. This is not weakened: the bar stays. The remaining lever is folding the restore into the backend's own dequant FMA, which would need a KVarN-side `dequantize_value_add_into`. That is not attempted here.

## P2 — fitted K=V+ retrofit

**G1 — the refund law.** Synthetic `V = K + E*[s] + noise`, 64 Zipf tokens, d=64. The table is fitted on 200k calibration tokens and measured on 50k held-out tokens. The prediction is `MSE(λ)/MSE(0) = 1 − (2λ − λ²)·ρ(V−K)`. The tolerance is ±3%. **6/6 PASS**, all within 0.11%:

| ρ(V−K) | λ=0.5 measured / predicted | λ=1 measured / predicted |
|---|---|---|
| 0.1773 | 0.8671 / 0.8670 | 0.8230 / 0.8227 |
| 0.4813 | 0.6388 / 0.6390 | 0.5185 / 0.5187 |
| 0.7896 | 0.4075 / 0.4078 | 0.2102 / 0.2104 |

This is the dashboard's go/no-go made quantitative. On this law, gemma-2-2b's measured ρ(V−K) = 0.49 (riir-infer Bench 004) predicts the table refunds **~49%** of the per-token V:=K residual energy at λ=1. Whether that refund reaches perplexity is riir-infer Issue 013's question and is not claimed here. James–Stein shrinkage is reported only: at n_s ≫ λ_js it changes nothing (λ_js = 10: 1.0000), and at λ_js = 1000 it costs 1.8–29% MSE. Shrinkage is a Zipf-tail guard, not a free win.

**G2 — `v_from_k_plus` vs the W_V GEMV it deletes** (gemma-2-2b shape, f32 weights, d=2304 → d_v=1024). The bar is ≤ 0.05×. **PASS, 3/3:** medians **0.00095 / 0.00098 / 0.00097**. The GEMV costs 241–264 µs and K+λE costs 244–266 ns per token-layer. As a report line, K+λE costs 2.6× the bare `V := K` copy (69 vs 179 ns): that is the price of the refund over free sharing. **G3:** PASS. λ=0 and a missing row are bitwise copies of K, including a `−0.0` input and a `f32::MAX` table entry. **G4:** 0 allocs over 30k calls, covering P2 and P3.

## P3 — V-cache halving by reconstruction

**G1 — exactness.** The fixture is `G(−θp)·(G(θp)·K) + λE` vs `K + λE`, which is P2's served V. It uses hd=256 × 4 heads, positions 0..131071 (512 sampled, including 0, 1, and 131071), λ ∈ {0, 0.37, 1}, occasional 60σ channels, and θ ∈ {1e4, 1e6}. The stated bound is per element `|err| ≤ 8ε·(‖k_pair‖₂ + |result|)`, with ε = `f32::EPSILON`. **PASS:** worst **1.57ε** (θ=1e4) and **1.83ε** (θ=1e6); max absolute error 1.5e-5 on 60σ channels. The angle's own f32 rounding (≈131072·ε rad at the far end) cancels exactly: the inverse uses `(−p)·ω = −(p·ω)` bit-exactly, so only the cos/sin evaluation and FMA roundings remain.

**Trap 2 pinned (report).** Repeated forward/inverse round-trips at p = 100003 compound the error: 2.4e-7 after 1, 1.9e-6 after 8, 8.3e-6 after 64. The primitive does exactly one inverse rotation per read.

**G3:** PASS. `VReadPath::FullCache` is a bitwise copy of the stored V.

**G2 — one decode-attention head** (online softmax, fused score + V accumulate, T=32768, hd=256). The full cache streams 2048 B/position (K+V, 64 MiB > SLC). Reconstruct streams 1024 B/position of K; the table row is Zipf-hot (256 rows = 256 KiB). The bar is ≤ 1.00×.

| run | reconstruct (`RopeAction`) / full | TableRope adapter (report) / full |
|---|---|---|
| 1 | **15.35×** (2.43 → 36.85 ms) | 1.63× |
| 2 | **15.22×** | 1.60× |
| 3 | **14.15×** | 1.64× |

**FAIL, 3/3.** The 2× byte saving is real, but the shipped action is `RopeAction`, a vocabulary bridge whose own doc says it is not a hot-path kernel. It computes 128 `sin`/`cos` pairs per head per read, which makes the read transcendental-bound. A test-local table-driven `PositionGroupAction` removes the transcendentals and still loses (1.6×). Its per-position cos/sin table is itself 1024 B/position of streamed data, which cancels the saving, and the scalar rotate loop is ALU-bound. The primitive is correct and generic over the action. What P3 G2 needs is a rotation whose angles cost no streamed bytes and no transcendentals per read, e.g. a block angle-addition recurrence `R(p+1) = R(p)·R(1)` with exact re-anchoring. That kernel does not exist in-tree. Filing it is left to riir-infer Issue 013's tg128 cell, which is where the model-level G2 lives anyway.

## P4 — the laws (two-line bench check + storage dial)

- **Line 1, FLOP law.** The counted multiply/add ops of a reference GEMV equal `d_v·(2d − 1)` exactly (803 = 407 mul + 396 add at d=37, d_v=11). **PASS.** For gemma-2-2b the law gives `ΔF_V = L·S·d_v·(2d−1)` = 26·128·1024·4607 = **15.70 GFLOP @ S=128**, i.e. 4.718 MFLOP per token-layer.
- **Line 2, measured.** The deleted f32 W_V GEMV takes 241–264 µs per token-layer, i.e. **17.9–19.6 GFLOP/s at 35.8–39.2 GB/s** of streamed weights. Removing it saves ~6.3–6.9 ms per token over 26 layers on this box. At 19.6 GFLOP/s against an M3 Max's peak, the saving is **byte-shaped**: it tracks the 9.4 MB of W_V per layer, not the FLOPs. That matches the paper's own caveat that value-arithmetic savings do not uniformly become latency savings. With f16/quantized weights the saving shrinks in proportion to the weight bytes.
- **Storage dial** `P(K) = b_w·L·K·d_v` (f32, gemma-2-2b L=26, d_v=1024):

  | K | table size |
  |---|---|
  | 1024 | 104 MiB |
  | 8192 (the P0 calibration's K, 97.1% coverage on the Kimi fixture) | 0.81 GiB |
  | 256k (full vocab, the paper's K=N case) | 25.4 GiB |

  K is read off the P0 `coverage_curve()`, so the tail beyond K takes the plain path.
- **50% cache law.** The droppable fraction is `n_v/(n_kv + n_v)`, computed per token as `v_width/(k_width + v_width)`. It is exactly 1/2 whenever K and V heads are equal in count and width. GQA changes `n_kv` for both K and V, so the fraction stays 1/2 (gemma-2-2b 8q:4kv hd 256 → 0.500). It departs from 1/2 only when K and V widths differ (e.g. MLA with separate rope dims: 192/128 → 0.40 droppable). Sliding-window layers scale both K and V with the window, so the fraction is unchanged.

## Verdict

- **P1:** the variance law is confirmed at the quantizer level on KVarN within ±5% across 2/4/8 bits, and sinks are absorbed (trap 3). The absmax caveat is real per-occurrence at 2 bits (3.8× on off-mean rows). The ≤ +1% latency bar FAILED at +5% and is recorded as failed.
- **P2:** the refund law is exact to 0.11%, λ=0 is bitwise V:=K, and the primitive costs 0.1% of the GEMV it deletes. PASS.
- **P3:** the reconstruction is exact to 1.8ε, the kill switch is bitwise, and trap 2 is pinned. G2 FAILED at 14–15× with the shipped `RopeAction`; a cheap-angle rotation kernel is the named missing piece.
- **P4:** the FLOP law is verified; the saving is byte-shaped, not FLOP-shaped. The storage dial and the 50%/GQA law are recorded.
- **All four stay OPT-IN.** No model-level quality claim is made. PPL at matched bits, the per-family retention walk, the K=V+ λ ladder with NIAH, and tg128 KV bytes/token all belong to riir-infer Issue 013.
