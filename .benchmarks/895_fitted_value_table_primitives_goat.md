# Bench 895 — fitted token-value table primitives (Issue 883 P1–P4, primitive half)

**Status:** P1/P2/P3 primitives LANDED, **OPT-IN**. G1/G3/G4 pass on every arm. **Two G2 bars FAILED and are recorded as failed:** P1's fused dequant+restore costs +4.8–5.2% against a ≤ +1% bar, and P3's reconstruct-from-K costs 14–15× against a ≤ 1.00× bar. P2 G2 passes. P1 G2 was re-attempted with the dequant-fold lever and **still fails** (see the addendum at the end; the original figures below are unchanged). Issue 894 then landed the vectorized KVarN dequant (`bf7d37244`, bit-identical, value 4-bit **2.46×**); on that kernel P1 G2 re-measures at **+12.9–13.3%**, still FAIL (Addendum II). This bench makes no model-level quality claim; that half is riir-infer Issue 013.

- **Issue:** 883 (closed 2026-09-25, HISTORY.md § Issue 883) · **Research:** [587](../.research/587_Memory_Attention_Fitted_Token_Value_Tables.md) · **Substrate:** Bench 886 (`fitted_anchor_table.rs`, P0) · **Model-bound G1:** riir-infer Issue 013
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

## Addendum — 2026-09-25: the dequant-fold lever (P1 G2 re-attempt)

**Verdict: P1 G2 still FAILS.** The lever named above, folding the table-row add-back into KVarN's own dequant epilogue, is slower than the shipped axpy-epilogue form in every variant tried. It was committed so it can be reproduced (`60f1e7baa`) and then reverted (`a24112aa7`). The bar stays at ≤ +1%. The original P1 G2 table above is unchanged.

**What was tried (`60f1e7baa`).** `QuantizedKVCache::dequantize_value_add_into(layer, pos, out, bias)` was added to `katgpt-types` as a trait method whose default dequantizes and then does one f32 `+` per element, so every other backend compiles unchanged. KVarN overrode it with a const-generic `dequantize_value_impl::<BIAS>` that appends `+ bias[ch]` to each bit-width's epilogue, falling back to two passes when Hadamard is on. `MeanRemovedValueCache::{dequantize_value_into, accumulate_value}` routed through it. There was no existing `dequant…add` / `_with_bias` spelling to reuse: `git grep` found only the per-backend `dequantize_value_into` and an unrelated `route_with_bias`.

**New gates, kept after the revert.**
- **G3b** checks a non-zero table: the decorated read and `accumulate_value` must match plain dequant followed by the unfused `+ E[s]` / `w·(x + m)` **bitwise**, at bits 2, 4 and 8 and with Hadamard. Every fold of the add-back into a backend epilogue now has to pass it.
- **Three G2 REPORT arms:**
  - an **A/A** control (a second plain cache), which measures the protocol's floor on this box;
  - **lookup-only** (a miss table), which isolates the per-position token → row resolution;
  - **deferred restore**, the next lever, described below.

**Box state.** M3 Max, 64 GiB, AC power, battery 100% charged, `powermode 2`. Swap was 1078.44 / 2048 MiB used (1070.44 MiB in the last block). Other agent sessions were running throughout. Each figure is an `ab_median_ratio` median (21 rounds × 10 iters, 3 warm-up), built with `--release`, with `black_box` on the weight argument and on the accumulator sink. Free RAM comes from `vm_stat` and load is the 1-minute average; both were read immediately before each run.

**The fold, bit-identical (`60f1e7baa` tree). FAIL 3/3.**

| run | free / load | **fused / plain** | two-pass read | A/A | lookup-only | deferred |
|---|---|---|---|---|---|---|
| 1 | 1.38 GiB / 24.4 | **1.1392** | 1.1380 | 0.9977 | 1.0119 | 1.0373 |
| 2 | 2.41 GiB / 28.5 | **1.1388** | 1.1409 | 0.9986 | 1.0121 | 1.0154 |
| 3 | 2.26 GiB / 29.0 | **1.1425** | 1.0353 | 1.0006 | 1.0135 | 1.0184 |

G3, G3b and G4 all pass: 0 differing elements and 0 allocations. The fold is exact, but at **+13.9–14.2%** it is nearly 3× the shipped +4.8–5.3%.

**Working-tree probes (not committed).**
- **FMA-contracted fold.** The var-norm epilogues became `(fma(q,s,zp)·s_col).mul_add(var_row, e)`, which replaces a multiply instead of adding an op. This is `scale·q + (zero + E)` in the only place the add can be absorbed: every var-norm arm multiplies after the affine term, so `E` cannot go into `zp`.
  - Timings: **1.0864 / 1.0941 / 1.0057** (free 1.41 / 1.17 / 1.69 GiB, load 39.3 / 40.2 / 40.2). The third run sat inside a 0.49–1.97 per-round band at a plain pass of 686 µs, so the box was thrashing.
  - Bit-identity: it **breaks G3b** at bits 4 and 8, with 154,225 and 153,932 of 524,288 read elements differing, and 89 and 105 of 128 accumulate elements. The 2-bit and Hadamard arms stay bitwise, because their epilogues have no trailing multiply to contract.
  - A ULP bound was not pinned, because the variant loses on both axes. For the record, per element the difference is bounded by `½ε·(|rnd(t·v)| + |u| + |f|)` (ε = `f32::EPSILON`), where `u` is the unfused result and `f` the fused one.
- **Zip-vectorized 4-bit loop.** The 4-bit full-pair loop was rewritten as `chunks_exact` zips with no bounds checks, keeping the same per-element ops in the same order. G3b passes on the new kernel.
  - **The plain pass drops from 494–531 µs to 156–162 µs, i.e. 3.2× faster.** This is a finding in its own right: the shipped KVarN 4-bit value dequant is scalar. It was filed as Issue 894, now closed and landed: see [HISTORY.md § Issue 894](../HISTORY.md) and Addendum II below.
  - On that kernel, though, the fold costs **1.2777 / 1.2779 / 1.2889** (free 1.20 / 0.94 / 1.10 GiB, load 38.0 / 35.7 / 37.3). Deferred restore costs +3.1–3.3% and lookup-only +1.6–3.5%.
  - The effect runs against the bar: vectorizing the backend shrinks the denominator, so every per-position cost the restore adds becomes a larger share of it.

**The shipped form, re-measured after the revert (`a24112aa7` tree).**

| run | free / load | **fused / plain** (plain µs) | two-pass read | A/A | lookup-only | deferred |
|---|---|---|---|---|---|---|
| 1 | 5.68 GiB / 34.7 | **1.0480** (436.5) | 1.1219 | 0.9947 | 1.0248 | 1.0197 |
| 2 | 5.67 GiB / 32.8 | **1.0464** (459.0) | 1.1206 | 0.9966 | 1.0284 | 1.0211 |
| 3 | 6.00 GiB / 32.8 | **1.0491** (510.7) | 1.1176 | 0.9941 | 1.0249 | 1.0146 |

This reproduces the original +4.8–5.2% at a heavier load, with the A/A floor at −0.3 to −0.6%.

**Where the cost lives, and the next lever.**
- **Scalar loops pay more for the add.** An extra load and add per element costs more inside a scalar, bounds-checked dequant loop (+13.9%) than inside the vectorized axpy (+4.8%). The recorded lever put the add in the wrong loop.
- **The lookup alone breaks the bar.** The per-position token → row resolution, with no add at all, costs **+1.2–2.8%**, measured as a decorator whose table misses everything. So no restore that resolves a row per position inside the V loop can reach ≤ +1% on this kernel, and the vectorized kernel makes it worse.
- **Next lever: deferred restore by linearity.** The aggregation is linear, so `Σ_p w_p·(v̂_p + E[s_p]) = Σ_p w_p·v̂_p + Σ_s W_s·E[s]` with `W_s = Σ_{p: s_p=s} w_p`. The V loop becomes exactly the plain loop plus one scalar bucket add per position, and each table row is read once per distinct token at the end. The test-local probe uses a pre-resolved per-position index (one load, not the decorator's two-level lookup) and measures **+2.0 / +2.1 / +1.5%** on the shipped kernel. That is the closest any form has come, and it still FAILS.
  - The remaining scalar add belongs in the consumer's softmax-weight loop, which already walks every position. That is the model-level decode kernel, i.e. riir-infer Issue 013.
  - This lever is **not bit-identical**, because it reassociates the sum. A ship candidate must pin a stated bound, for example the recursive-summation bound `2γ_{T+2}·Σ_p |w_p|(|v̂_p| + |E[s_p]|)`, and must keep the miss / E=0 path bitwise.
- **The lever that was tried is demoted.** P1 stays OPT-IN, and the G2 bar is unchanged.

## Addendum II — 2026-09-25: Issue 894 landed (vectorized KVarN dequant), and P1 G2 re-measured on it

**Verdict: the kernel rewrite PASSES its own GOAT and ships under `kvarn` (no new flag). P1 G2 still FAILS, now at +12.9–13.3%.** The bar stays at ≤ +1%.

**What landed (`bf7d37244`).** The per-bit-width dequant loops of `KVarNKVCache::{dequantize_value_into, dequantize_key_into}` moved to `crates/katgpt-kv/src/kvarn/dequant.rs`. They walk the same elements with `zip` / `as_chunks` / an exact-length strided iterator over slices cut to the loop length, so no access keeps a bounds check. Every per-element op and its order is unchanged (`fma(q, scale, zp) · s_a · s_b`, left-associated, `mul_add` exactly where it was). That covers all rewritten arms: value 4-bit, value 2-bit grouped / ungrouped / var-norm, value 8-bit, and the same four shapes on the key side. The generic-bits fallbacks are unchanged. The methods now build a public read-only `KVarNValueRowView` / `KVarNKeyColView` and pass it to the kernel. That view is the seam through which the oracle, the pre-894 loops kept verbatim in `crates/katgpt-kv/tests/common/kvarn_dequant_oracle.rs`, sees exactly the inputs the shipped kernel sees.

**T1 — bit-identity, old vs new (`src/kvarn/dequant_oracle_tests.rs`). PASS.** The oracle covers bits {2, 3, 4, 8}, where 3 is the unchanged fallback. It crosses them with every `(skip_varn, group_size)` mode, including var-norm-on 2-bit and var-norm-off 4/8-bit, which `with_config` never selects and are reached through a `cfg(test)` setter. Those are crossed with 12 `kv_dim`s (1, 2, 3, 5, 7, 36–39, 64, 128, 130), with full, partial-last and counted-but-unquantized tiles, with Hadamard on/off, and with both value rows and key columns. **960 cases, 10,407,928 elements, 0 differing bits** (`to_bits`) in both debug and release. It was revert-probed: reassociating one multiply reds it. Positions whose OLD read panics (grouped 2-bit on an unquantized tile's empty metadata) are skipped by an explicit precondition, because the new code panics under the same condition.

**T3 — GOAT gate (`crates/katgpt-kv/tests/bench_894_kvarn_dequant_zip_goat.rs`, `required-features = ["kvarn"]`).**

- **G3** is a bitwise check over the modes `with_config` produces: bits {2, 4, 8} × Hadamard × kv_dim {128, 37} × a partial tile, K+V. **12/12 PASS**, 0 differing.
- **G4** checks allocations on the shipped K+V dequant. **0 allocs** at bits 2, 4 and 8.
- **G2** is a paired interleaved `ab_median_ratio` of new/old on the plain decode pass (T=4096, kv_dim 128, dequant + axpy), 21 rounds × 10 iters, 3 warm-up, `--release`, with `black_box` on the position, the output buffer, the view, the weight and the accumulator sink. Both arms read one shared cache.
- **Box state** (same for all three runs): M3 Max, AC power, battery 100% charged, `powermode 2`, swap 1070.44 / 2048 MiB. Per-run free RAM (`vm_stat`) and 1-minute load:

| run | free / load | **value 4-bit** (bar ≤ 0.90) | old → new µs | value 2-bit | value 8-bit | key 4-bit | key 2-bit | key 8-bit | A/A |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.11 GiB / 17.6 | **0.4077** (2.45×) | 387.1 → 151.0 | 0.6901 | 0.8108 | 0.7708 | 0.8964 | 0.7271 | 1.0023 |
| 2 | 3.70 GiB / 16.5 | **0.4064** (2.46×) | 372.8 → 151.5 | 0.6867 | 0.8031 | 0.7686 | 0.8971 | 0.7258 | 1.0043 |
| 3 | 3.81 GiB / 16.5 | **0.4065** (2.46×) | 380.4 → 151.9 | 0.6815 | 0.8054 | 0.7695 | 0.8989 | 0.7312 | 1.0014 |

Every other rewritten arm is gated as no-regression (≤ 1.05), and all pass, 1.11–1.47× faster. **ALL GATES PASS, 3/3.** The issue's probe measured 3.2× at load 35–38, against a 494–531 µs old arm. Here the old arm is 373–387 µs at load 16–17, and the new arm reads 151–152 µs in both. The speedup is smaller because the old arm is faster on a quieter box, not because the new kernel is slower.

**T4 — Issue 883 P1 G2 on the new kernel (`bench_895_mean_removed_v_quant_goat`, same `a24112aa7` decorator, 3 runs). FAIL 3/3; bar unchanged.** Box: M3 Max, AC, 100% charged, `powermode 2`, swap 1070.44 / 2048 MiB.

| run | free / load | **fused / plain** (plain µs) | two-pass read | A/A | lookup-only | deferred |
|---|---|---|---|---|---|---|
| 1 | 2.47 GiB / 14.0 | **1.1285** (137.9) | 1.3048 | 0.9925 | 1.0278 | 1.0262 |
| 2 | 3.45 GiB / 13.9 | **1.1333** (145.8) | 1.3141 | 0.9919 | 1.0218 | 1.0356 |
| 3 | 3.46 GiB / 13.9 | **1.1333** (137.9) | 1.3079 | 0.9859 | 1.0214 | 1.0351 |

Every other P1 gate still passes, and G3 / G3b are bitwise on the new kernel. The plain pass dropped from 436–510 µs to 138–146 µs. The fused epilogue's absolute added cost stayed about the same (~18 µs per pass now, ~23 µs before), so its share of the ratio grew 2.7×: +4.8% became +12.9–13.3%. That is the interaction Issue 894 T4 predicted. The deferred restore (+2.6–3.6%) and lookup-only (+2.1–2.8%) report arms both still sit above +1%. The next lever is unchanged: move the scalar per-position add into the consumer's softmax-weight loop (riir-infer Issue 013).

