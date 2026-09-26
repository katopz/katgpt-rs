## Issue 900 (2026-09-26) — issue-717 damping G4 failed at the wider feature combo: the gain_cost_halt probe buffers allocated unconditionally per call: CLOSED (lazy-sized; the measured region is combo-stable; commit `e272057be`)

- **Found-by:** Issue-898 twin validation (2026-09-26) — the G4 alloc gate for the damping hot loop was run at a WIDER combo than its own `required-features` and failed: `16 allocs / 1024 B over 8 deep runs` at `lt2_deep_stability,gain_cost_halt,cadence_gate,loop_stability_fix`, while passing at its declared `lt2_deep_stability` alone. Pre-existing, not 898 regression (reproduced without the 898 features).
- **Root cause (arithmetic-exact):** `forward_looped`'s two step-direction scratch `Vec`s (`prev_step_buf`/`curr_step_buf`) were `Vec::with_capacity(n)` at fn entry whenever `gain_cost_halt` was COMPILED, even when the caller passed `None` halter — the default probe posture of every gate and the bit-identity path. 2 dead allocs/call; at the micro fixture's `n_embd=16` that is 64 B each → 2 × 64 B × 8 runs = 1024 B, 16 allocs — matching the failure byte-for-byte. The buffers are only read inside the `halter.as_deref_mut()` block, which never runs at `None`.
- **Fix (option (a) — combo-stable region):** size the buffers iff the halter can actually run (`halter_active && halter.is_some()`), `Vec::new()` otherwise — zero alloc, observationally identical (an unused `Vec::new()` has no capacity and nothing reads one). The `Some(halter)` path is unchanged: still `with_capacity(n)` once per call, outside the per-iteration loop. Also repairs the false "Zero cost when the halter is inactive" comment claim the buffers contradicted. Same lazy shape `_gate_scratch_logits` already used. Direction (b) (a pinned suite row at the wider combo) NOT needed: the measured region is allocation-free by construction at any combo now.
- **Validated:** wide combo g4 1/1 (0 allocs) + full file 5/5; narrow declared combo 5/5; widest `+weight_shared_advantage_gate` 5/5; halter-active `issue_698_t4_halter_floors` 1/1 (its own required-features — the Some(halter) sizing path still works, floors intact); clippy lib+tests clean at the combo.
- **Static sibling hunt (same day):** a 6-line cfg-attr × allocating-`let` window over tracked `src/` found ZERO second instances of the class — the only other candidates adjudicated not-the-class (`hl_player.rs` bandit cfg fields are `None` struct literals; `ppot/resample.rs` hits are `#[cfg] #[test]` fns; `rim_extend_tokens` is a one-shot prompt helper, not an alloc-gate-measured loop). Bounds, stated: the window grep is not block-aware and blind wider-combo runs of the ~20 alloc gates were not taken — the residual risk is cfg-gated allocations in a measured region beyond a 6-line span, judged low-yield per compile cost.
- **Note on the numbering space:** `Bench 900` (`.benchmarks/`) is a DIFFERENT counter — this is `.issues/900`. File removed per the noise-reduction rule; git history holds it.

## Issue 899 (2026-09-26) — second-moment, null-normalized drift alignment (the Plan 610 redesign): CLOSED (negative on 2 of 6 bars; became the default summary inside opt-in `arm_drift_alignment`; no promotion)

- **Shipped:** katgpt-core `DriftSummary` strategy (`FirstMomentDrift`, `SecondMomentDrift`) behind one `TrajectoryAlignedCuriosity<S>` at `3485ec59e` ([Bench 901](.benchmarks/901_second_moment_drift_alignment.md)); pre-registrations at `099b9d4a6` / `b1615393d`, all before the run they govern.
- **Mechanism:** score arm `k` in ARM space through the pool's squared-cosine kernel `C_kj = ⟨ĝ_k, ĝ_j⟩²` on the incumbent's own per-arm derivative, z-scored against the i.i.d.-arm-noise null. Squared cosines add where the first moment's `±e_i` pairs cancel (Bench 900 defect 1); the null is basis-free where the axis-aligned preconditioner was not (defect 2).
- **Three variants on one fixture, all reported; stopped there to avoid fitting it:**
  - v1 as pre-registered: G1 0.963 / held-out 0.926, noise control **FAIL** 0.949. The pass was mostly the kernel's zero-init EMA transient (common-mode drift read in proportion to row sum ⇒ dense-cluster bias).
  - v1 + warm start: G1 **FAIL** 0.816 / 0.633, noise **FAIL** the other way (0.186). Simplex shares are negatively correlated, so the i.i.d. denominator over-states dense rows' null variance.
  - v2 (simplex-centered null `c̃_k = C_k − (s·C_k)·1`, numerator unchanged since `Σd = 0`): G1 0.883 PASS, **held-out 0.766 FAIL**, noise 0.559 PASS, G3 forward −79 and reversed −63 cycles vs its matched-uniform bonus (both PASS), **G4 2.07× FAIL** (bar 2.0×).
- **Verdict:** the only summary loop-sound in both directions, so it is the feature's default summary; the feature stays opt-in. Honest-null clause applied: riir-ai guide 389 P2 stays retired.
- **Promotion criterion (owner-delegated verdict, 2026-09-26):** planted-truth G1 held-out is NOT replaced by loop gates. The loop win is measured on the same fixture the three variants were chosen on, so it is the weaker generalization evidence; swapping the criterion after seeing which bar passed is the post-hoc move the pre-registration exists to prevent. Loop soundness in BOTH reward directions is ADDED as a mandatory bar for curiosity-class primitives (necessary, not sufficient) — Bench 900's first-moment form passed G3 forward and lost the reversed direction by +108 cycles.
- **Side fixes:** `DerivativeCuriosity::cycle_curiosity` allocated once per cycle despite its alloc-free claim (now `ensure_len`, gated at 0); Bench 900's "un-normalized score is a density prior under noise (1.000)" was the same zero-init transient — 0.525 warm-started, addendum in Bench 900.

## Issue 895 (2026-09-25) — guided width rollouts on the belief host (GRAM re-distill, Research 590): CLOSED (T1–T7 + T9 landed opt-in; G1 FAILED, demote TRIGGERED, no promotion; T8 handed to riir-ai Issue 1008)

- **Shipped:** katgpt-core opt-in `guided_width_rollouts` + `guided_width_hodge` at `a4421939f` / `ae7b13320` ([Bench 898](.benchmarks/898_guided_width_rollouts_goat.md)). T1 structured perturbation (transversal on [f32;8] belief; coexact∪harmonic ε on 2D cochains); T2 stagnation-sigmoid σ_t; T3 decode-free `latent_value` scorer; T4 Sobol/BLAKE3 diversity init + farthest-point; T5 success-SVD table on `thin_svd_into` (katgpt-canon = package cycle), Beta reweight, BLAKE3 freeze; T6 trap-kill-reallocate via `saddle_escape`. New shared helpers pinned bit-identical: `blake3_noise_fill`, `select_diverse_subset_in_place`, `SobolQmc::reseed`.
- **Fixture:** graph 3-colouring, 128 instances/family, 8×16 vs 1×128 steps.
- **G1 FAILED:** width beats depth on MULTI +0.117 ± 0.049, loses on SINGLE −0.188 ± 0.036.
- **Demote TRIGGERED:** table ties zero-mean on MULTI (+0.001 ± 0.014), loses on SINGLE (−0.025 ± 0.012). **Guided table off-by-default forever: closed-negative** (negative_results §43).
- **Pass arms:** E9 MULTI coverage 0.59 → 3.32; mass arm 0 non-zero divergences / 2500; G2 14.9 µs/decision, K linear 2.00×; G3 σ=0 / N=1 bit-identical to `evolve_belief`; G4 0 allocs.
- **Findings:** selector is the bottleneck (≥1 valid branch 98–99%, valid selected 70–75%); T6 trap-kill costs 4.7 pp (flip detector reads exploration noise as trap); transversal ≡ isotropic here; the +1.50 coverage gain was not pre-registered — reopening needs a new fixture with coverage pre-registered.
- **T9: no promotion** (058 §8.3 stands). **Plan 095** GOAT PENDING 2/3 (G1 passes; G3 needs a stochastic arena). **T8** → riir-ai Issue 1008; DDTree host out of scope.

## Issue 897 (2026-09-25) — katgpt-kv's KVarN `static_cal_tables` branches were gated on an undeclared feature and named a missing module: CLOSED (deleted, not wired; `unexpected_cfgs` allow removed; a second instance, `memory_soup_dtree`, restored)

- **Shipped (`f3e3a4f56`), T1 DELETE on evidence:** removed `KVarNConfig::static_cal`, the cache field, both `quantize_*_tile` static-cal branches (`not(static_cal_tables)` arms now unconditional), 3 init sites (`eval.rs`, 2 tests). No build changes — the code compiled nowhere.
- **Bisected:** live at root from `25aca61b1` (Plan 227 Phase 1); dead since the Issue 015 extraction `b61a34f7b` (no feature/module declared; crate-wide `#![allow(unexpected_cfgs)]` hid it). ~3 months of "static cal replaces Sinkhorn in KVarN" described uncompiled code.
- **Not wired — edge illegal:** `katgpt-attn = { optional = true }` in katgpt-kv makes `cargo metadata` fail (*cyclic package dependency* katgpt-attn → forward → pruners → speculative → kv → attn; attn also deps kv under default-on `dash_attn`). Cargo rejects optional cycles too.
- **Not re-homed — quality fails anyway:** per-row scale with `s_col = 1` is absorbed by KVarN's per-row affine RTN, so the branch is **plain RTN**. Scratch probe (128×128 tiles): static vs RTN max |Δ| **1.2e-4** over 4 families × 3 seeds × {key,val} × bits {2,3,4}; static MSE ≤ Sinkhorn in only **6/72** cells, loses every outlier family (seed-1: val channel-outlier 30.5/6.02/1.33 vs 6.78/1.26/0.267 at 2/3/4 bits, cosine 0.691 vs 0.899; key token-outlier 11.2 vs 1.71 at 2-bit; key both-outliers 4-bit 11.7 vs 2.13). Latency **0.034×** Sinkhorn (paired `ab_median_ratio`, `--release`, 31×40, range 0.028–0.037; M3, AC, powermode 2, load 13–16) — a no-op's speed. Also mis-indexed (V by token position, K by channel).
- **`static_cal_goat` actually measures** 128 table lookups vs one Sinkhorn `variance_normalize`; its "perplexity delta" is 1 − cosine of two positive vectors (~0.0000 by construction; re-run 6/6). Never runs KVarN. Root comment, `static_cal.rs` doc, test header, Plan 227 (wiring task retracted `[-]`) corrected; feature stays default-ON as the katgpt-attn table.
- **T2:** `#![allow(unexpected_cfgs)]` removed from `crates/katgpt-kv/src/lib.rs`; it surfaced a **second instance**: `segment_checkpoint::memory_soup` gated on undeclared `memory_soup_dtree` since the Proposal 003 Phase 5 move (`70d418b8e`). Restored as opt-in katgpt-kv feature, root forwards (3 tests pass).
- **Checks:** clippy `-p katgpt-kv --all-targets -D warnings` clean across default/`kvarn`/`kvarn,fitted_value_tables`/`kvarn,targeted_precision`/`memory_soup_dtree`/`ssc_spec_draft,memory_soup_dtree`/`mi_probe`/`--all-features`; root kvarn consumers clean; lib tests 33/37/33/53 pass; `count_features`, `cargo_comment_audit` (440, 0 mismatches), `markdown_fence_gate`, `bench_doc_audit` pass.

## Issue 896 (2026-09-25) — KVarN's raw tile buffer was shared across layers, and the in-progress tile dequantized to zeros (2-bit keys panicked): CLOSED (fixed under `kvarn`; layer-major bit-identical; no consumer's published quality figure carried it)

- **Shipped (`62fd22b5f`), T1 option (a):** `key_buffer`/`val_buffer` are one raw tile per layer (`n_layers × kv_dim × tile_size` f32, allocated once in `with_config`); hot path 0-alloc, +one offset. Memory +`(n_layers − 1) × kv_dim × tile_size × 8` B (26 layers × 1024 × 128: ~1 → ~27 MB). T2: `TileMeta.quantized` set by `quantize_*_tile`, cleared by `reset()`; views return `None` unless quantized. Non-quantized tiles served EXACTLY from raw (`read_raw_key`/`read_raw_value`); unstored slot reads 0.
- **Layer-major pin:** 894 oracle (`src/kvarn/dequant_oracle_tests.rs`) **960 cases, 10,427,200 elements, 0 differing bits** (up from 10,407,928: in-progress positions now asserted bitwise instead of skipped).
- **T3 (`src/kvarn/issue_896_tests.rs`):** decode-order stores, n_layers {2,3} × bits {2,3,4,8} × every mode × Hadamard × kv_dim {16,37} × 3 tile geometries, every position read after every store: **240 cases, 21,687,600 elements, 0 differing bits**; plus partial-tile and `reset()` cases. Revert-probed: all six red pre-fix (L0 served L1; in-progress read zeros).
- **T4 consumers:** siblings none (`KVarNKVCache`/`KVarNConfig` constructed nowhere; riir-ai `riir-gpu/src/kvarn/*` uses unchanged free functions; riir-infer decode uses `TurboQuantKVCache`). In-tree: all tests/benches single-layer, quantized-only (`bench_694`, `bench_894`/`bench_895`, `pseudo_decode_eval`, `static_cal_goat`, `octpq_kvarn_fusion`). Only in-progress reader `examples/kvarn_goat_proof.rs` Phase 5 (info-only timing): 2.66/2.56/3.08 → 2.58/2.68/2.67 µs, GOAT figures identical (cosine 0.9979, ratio 1.0129). `benches/kv_cache_flatten_bench.rs` unaffected, not re-run.
- **Perf (no regression):** Bench 894 GOAT + bench_895, `--release`, interleaved pre-fix/fixed, 3 rounds; all 894 gates PASS both. New-kernel µs per 4096-position pass; new/old ratio:

  | arm | pre-fix | fixed |
  |---|---|---|
  | value 4-bit | 147.8/146.7/146.5 µs; 0.4067/0.4059/0.4061 | 146.7/157.4/146.7 µs; 0.4100/0.4088/0.4081 |
  | value 2-bit | 194.9/192.2/197.4; 0.6919/0.6863/0.6912 | 189.8/191.1/192.2; 0.6830/0.6848/0.6855 |
  | value 8-bit | 121.0/125.4/126.5; 0.8090/0.8077/0.8052 | 118.8/117.8/117.8; 0.7762/0.7751/0.7739 |
  | key 4-bit | 493.6/481.8/486.4; 0.7684/0.7702/0.7719 | 492.4/487.9/486.6; 0.7756/0.7740/0.7715 |
  | key 2-bit | 547.0/527.9/525.5; 0.9027/0.8937/0.8972 | 513.6/530.0/512.5; 0.9019/0.9083/0.8928 |
  | key 8-bit | 459.7/461.8/458.1; 0.7315/0.7292/0.7268 | 456.7/459.1/460.3; 0.7284/0.7308/0.7314 |
  | 895 P1 G2 fused/plain | 1.1222/1.1175/1.1325, plain 134.7/135.6/137.6 µs | 1.1109/1.1161/1.1094, plain 135.6/137.2/139.2 µs |

  - Value 4-bit ratio drifted +0.6% in all 3 rounds with absolute time flat; far inside ≤ 0.90. 895 plain +1.1% within spread; the 895 G2 FAIL is Issue 883's, identical pre-fix. Box: M3 Max, AC, powermode 2, load 16.4–19.3, free 2.7–3.4 GiB, swap 1070/2048 MiB.
- **Checks:** clippy `-p katgpt-kv -D warnings` clean at default/`kvarn`/`+fitted_value_tables`/`+targeted_precision` (`--lib`, `--all-targets`); root consumers (`kvarn_goat_proof`, `kvarn_thinking_demo`, `octpq_kvarn_fusion`, `chiaroscuro_03_collapse_discovery`, `static_cal_goat`, `kv_cache_flatten_bench`) clean; lib tests 33/33/37.
- **Found while closing → Issue 897 (closed, above):** undeclared `static_cal_tables` + missing `crate::static_cal`, hidden by `#![allow(unexpected_cfgs)]`.
- **Residual, unchanged:** `TileMeta::count` is a store counter, not a slot map; re-storing without `reset()` double-counts. Out of scope (`QuantizedKVCache` has no overwrite semantics).

## Issue 883 (2026-09-25) — fitted token-value tables (K=V+ retrofit, mean-removed V quant, V-cache halving): CLOSED (P0–P4 landed opt-in; two primitive-level G2 bars FAILED and are recorded; the model-bound half is riir-infer Issue 013)

Source: Research 587 (Memory Attention, arXiv:2609.28399); V-side twin of Issue 882.

- **P0 dashboard:** `fitted_anchor_table.rs` + `LayeredVkCalibration` (Bench 886). Gemma: riir-infer `vk_calibration` Bench 004, mean ρ_l(V−K) = 0.49 over 120k tokens → GO. Kimi-K3: Bench 889 (`ba82bad60`), MLA ρ ≈ 0.52–0.60 all three — fixture-class null.
- **P1–P3** at `0b768e95d`, `crates/katgpt-core/src/fitted_value_table.rs`, opt-in `fitted_value_tables` + `fitted_v_reconstruct` ([Bench 895](.benchmarks/895_fitted_value_table_primitives_goat.md)): P1 `MeanRemovedValueCache<C: QuantizedKVCache>` — `1 − ρ_V` predicts KVarN MSE ratio ±5% on 9/9; absmax caveat real at 2 bits (off-mean 3.8× worse); sinks absorbed. P2 `v_from_k_plus` — refund law `1 − (2λ − λ²)·ρ(V−K)` within 0.11%, 0.001× the W_V GEMV. P3 `reconstruct_v_from_rope_k` — exact to 1.83ε to position 131071, drift 8.3e-6 after 64 round-trips. G1/G3/G4 PASS all arms.
- **P4 laws:** FLOP law `d_v·(2d−1)` exact; saving byte-shaped (18–20 GFLOP/s at 36–39 GB/s); storage dial + `n_v/(n_kv+n_v)` law in `.docs/02_inference/kv_compression.md` §7.
- **FAILED G2, NOT weakened:** P1 fused restore +4.8–5.2% vs ≤ +1%; folding into dequant (`60f1e7baa`, reverted `a24112aa7`) worse +13.9–14.2%; after Issue 894 (plain 2.46× faster) +12.9–13.3% (same absolute cost, bigger share). Floor = token→row lookup +2.1–2.8%; next lever deferred restore +1.5–3.6%, last add belongs in consumer's softmax loop. P3 14–15× slower with `RopeAction` (sin/cos per read), table-driven still 1.6×. Both levers → riir-infer Issue 013 T4.
- **Convention trap:** riir-infer gemma-2 RoPE is half-split (NeoX), `RopeAction` interleaved — consumer must pass a half-split `PositionGroupAction` or get silent corruption.
- **Side findings:** Issue 894 (dequant scalar → 1.11–2.46× faster, bit-identical); Issue 896 (shared raw tile buffer, in-progress zeros; riir-infer 013 T1 needs it).
- **Promotion** waits on riir-infer 013 T1–T3 + loser demoted.

## Issue 894 (2026-09-25) — KVarN dequant loops were scalar (bounds-checked indexing); rewritten as bit-identical zips: CLOSED (GOAT PASS, ships under `kvarn`)

- **Shipped (`bf7d37244`):** per-bit-width loops of `KVarNKVCache::{dequantize_value_into, dequantize_key_into}` moved to `crates/katgpt-kv/src/kvarn/dequant.rs` as `zip`/`as_chunks`/exact-length strided iterators; per-element ops and order unchanged (no reassociation). Read-only `KVarNValueRowView`/`KVarNKeyColView` are the oracle seam. No new flag.
- **T1 bit-identity** vs pre-894 loops (verbatim in `crates/katgpt-kv/tests/common/kvarn_dequant_oracle.rs`), bits {2,3,4,8} × every mode × 12 `kv_dim`s × full/partial/unquantized × Hadamard × K/V: **960 cases, 10,407,928 elements, 0 differing bits**, debug + release. Revert-probed (one reassociated multiply reds it).
- **T3 GOAT** ([Bench 895 Addendum II](.benchmarks/895_fitted_value_table_primitives_goat.md#addendum-ii--2026-09-25-issue-894-landed-vectorized-kvarn-dequant-and-p1-g2-re-measured-on-it); `crates/katgpt-kv/tests/bench_894_kvarn_dequant_zip_goat.rs`): G3 12/12, G4 0 allocs; G2 paired interleaved, T=4096, kv_dim 128, `--release`, 3 runs: **value 4-bit 0.4077/0.4064/0.4065 = 2.46× faster** (373–387 → 151–152 µs; bar ≤ 0.90). Others faster: value 2-bit 1.45–1.47×, 8-bit 1.23–1.25×, key 4-bit 1.30×, 2-bit 1.11–1.12×, 8-bit 1.37–1.38×; A/A +0.1–0.4%. Box: M3 Max, AC, powermode 2, load 16.5–17.6, free 3.7–4.1 GiB.
- **T4, Issue 883 P1 G2 on new kernel** (`bench_895_mean_removed_v_quant_goat`): fused/plain **1.1285/1.1333/1.1333, FAIL** vs ≤ +1%. Plain 138–146 µs (from 436–510); add-back ~18 vs ~23 µs, so share grew +4.8% → +13%. Deferred restore +2.6–3.6%, lookup-only +2.1–2.8%. Load 13.9–14.0. Figures for the 883 owner.
- **Found while closing → Issue 896 (closed at `62fd22b5f`, above), unchanged by 894:** shared raw tile buffer (L0 serves L1's values); in-progress tile reads zeros, 2-bit key reads panic.
## Issue 882 (2026-09-25) — differential anchor scoring (common-mode rejection on score surfaces we own): CLOSED (P0–P4 landed, all opt-in; promotion owed to consumers)

Source: Research 586 (Diff Transformer, arXiv:2410.05258). All primitives opt-in with GOAT benches.

- **P0 `differential_anchor` + `fitted_anchor_table`** ([Bench 886](.benchmarks/886_differential_anchor_goat.md)): G1 24/24 at λ\*=0.55; G2 0.031 µs; G3 bit-pinned; G4 0 allocs. **Law:** mean-sim skewness is invariant under a mean-query anchor, so hubness = top-1 WIN-COUNT distribution. riir-clippy rerank (`7f27de9d`, Bench 102 there) NOT promoted: +1/138, heal rate flat, skewness rose.
- **P1 `attention_snr`**, `c0628f4de` ([Bench 887](.benchmarks/887_attention_snr_goat.md)): exact streaming entropy + participation ratio in online softmax; G2 +0.4–1.1%, G3 bit-identical.
- **P2 `row_logit_floor`**, `dea5d9ec1` ([Bench 888](.benchmarks/888_row_logit_floor_goat.md)): clamp is a FLOOR `max(l, m_r − w)`, `w = ln(n/ε)`. Model-bound half → riir-infer Issue 011.
- **P3 `differential_kv_eviction`**, `f4926c44a` ([Bench 894](.benchmarks/894_differential_kv_eviction_goat.md)), synthetic hub-heavy needle:

    | Cache budget | Differential | λ=0 max-recent | Shipped usage-rate |
    |---|---|---|---|
    | 25% | 0.945 | 0.680 | **0.000** |
    | 50% | 0.977 | 0.758 | **0.000** |

  - Trap 4 pinned negative (1.35–1.47× baseline error under shift). Sink trap fires only at λ > 1; gate re-specified at λ=1.5, the λ ≤ 1 bar FAILED and is recorded. `select_evict_into` stable sort allocated at 4096 keys → unstable + index tie-break. Needle@64K → riir-infer Issue 012.
- **P4:** m_Y instrument riir-infer `dc0e5a9`, first reading `fb51821` (L10H5 0.928 on needle, m_Y ≤ 0.0009). Riders at `c8d22abb9` ([Bench 897](.benchmarks/897_p4_riders_goat.md)): `canonical_context` permutation spread 0 bitwise (bias constant, not absent); `affinity_deflation` gated on stable rank θ\*=1.1, healthy-matrix negative 1.000 → 0.812; α=3 bar FAILED (+0.072 vs +0.25), re-specified α=8: 0.586 → 1.000.
- **Promotion owed to consumers:** riir-infer Issues 011/012 (P2/P3); an oracle-anchored eval via `canonical_context`; an M×M rerank-affinity stage (not built by katgpt-attn-match `rerank`); riir-ai Issue 1006.

## Issue 886 (2026-09-25) — activation-diagonal weight-quant fitting (AWQ/imatrix-class substrate): CLOSED (P0 + P1 modelless half landed opt-in; model-bound half handed off; P2 deferred)

Source: Research 588 (AWQ, arXiv:2306.00978).

- **P0 `act_channel_moments`** (katgpt-core), `0fb2254d9` ([Bench 896](.benchmarks/896_act_diagonal_quant_fit_goat.md)): streaming `{mean|x|, E[x²]}` at 0.28–0.32 ns/element, 0 allocs; `freeze()` BLAKE3-committed LE table, byte flips refused.
- **P1 `act_aware_fit`** (katgpt-types): `quantize_from_f32_act_aware(w, rows, cols, diag, fit)`, diagonal a plain slice (no katgpt-core dep); quantizer refactored to per-group scale closure (±0.5%, payload identical); G3 uniform diag bit-identical. Synthetic G1: blind search −20…−27%; diagonal adds −54% (1% heavy channels at 20×) and −14% (log-normal, ternary) vs INT4 ref −60%/−26.5%; `E[x²]` beats `mean|x|`. **Prior NOT settled:** PTQ-of-dense fixture vs born-ternary Hadamard-rotated Bonsai.
- **Handed off:** real-checkpoint retention walk → riir-infer Issue 014 (`14dd5ae`), vs mean-abs, blind search, `ZeroQatCalibrator`.
- **Deferred `[-]` P2 (AWQ α-rescale), reopen on consumer:** GGUF-writer recipe; riir-train LoTA merge-weighting A/B; outlier-collapse tripwire (riir-train Issue 503 P3's complement).

## Issue 887 (2026-09-25) — `ladder_gate` streak-gated advancement + corrective backtracking FSM: CLOSED (GOAT PASS, stays opt-in)

- **Shipped:** `crates/katgpt-core/src/ladder_gate.rs`, opt-in `ladder_gate = []`, `ba7fb89ba` (Research 589 / arXiv:2609.19717, ATC). Advance after m consecutive evals ≥ τ, one below resets; retention retreats to the SHALLOWEST failing stage; malformed probe fails closed to stage 1; no auto-demote.
- **Gates ([Bench 891](.benchmarks/891_ladder_gate_goat.md)):** G1 17/17 (streak reset, argmin retreat, two paper negatives; λ=0 never passes stage 2); G2 0.8–1.2 ns/eval, k64/k8 0.977 (O(k)); G3 default lib 2063 unchanged; G4 0 allocs.
- **Toy caveat:** dwell pin asserts (0.9, 5) > (0.9, 1); (0.98, 1) excluded since toy's 0.95 cap makes τ = 0.98 unreachable.
- **Opt-in:** no production consumer (873/874/875 precedent). Named: riir-clippy corpus staging, riir-reflex `cal_min_obs`, riir-ai CGSP zone unlock, riir-train ATC rig (Plan 352); each files its own wiring.

## Issue 885 (2026-09-25) — `laya-tetris-v3` real-hard-drop lane (Issue 884 path B): CLOSED (every lane landed and deployed)

- **katgpt-rs:** `1a05a9764` `DropRule { DeepestFit (v2), FromTop (v3) }` + `landing_options_with` + enumerator `--grammar`/`--join`/`--carry-from` (v2 default unchanged). `6a35cde32` v3 fixture `tests/fixtures/tetris_oracle_laya_en_v3.jsonl` (blake3 `12035ebf…`, sha256 `eb67bc16…`); oracle re-run on 3 changed states only (102 forwards, M3 Metal), 117 carried verbatim (byte-identity asserted). `_meta` parity 0/99 bit-exact, max |Δp| 4e-6; 3 of 2660 options move (`holes`).
- **riir-reflex:** `ca1483c` serves v3 head; `a56d850` `fixture_pins()` four-hash (was length-only).
- **reflex-site:** `6116e70` v3 + golden sha256 pin (2660/2660); `2e99eca` v3-refit wasm head (anchors 44/120 → 42/120, LOO λ 1, Bench 892), demo walks re-recorded, parity 809/809; `793dcdd` regenerated `demo_oracle.json`. Deployed 2026-09-25: live `arena/demo_oracle.json` + `assets/arena_head.wasm` sha256-identical to `b935412`.
- Boundary rule now in [`.docs/06_game_arenas/tetris_sim_fidelity_boundary.md`](.docs/06_game_arenas/tetris_sim_fidelity_boundary.md) (ex-Issue 878). v4 preview: Plan 609 / Bench 890.

## Issue 878 (2026-09-24) — tetris_sim fidelity boundary: RECORD RELOCATED (2026-09-25)

Record (never swap in a guideline Tetris engine; new fidelity = NEW lane with own fixture and head) moved to [`.docs/06_game_arenas/tetris_sim_fidelity_boundary.md`](.docs/06_game_arenas/tetris_sim_fidelity_boundary.md); issue removed (standing rule = documentation). Sim path updated to `crates/katgpt-tetris/src/sim.rs` (Issue 893).

## Issue 873 (2026-09-22) — governed paged-pool primitives from mini-AGI: CLOSED (all three landed, opt-in; consumer work handed off)

- **Shipped** (Research 581, volotat/mini-AGI @ `96784b7`, MIT), opt-in katgpt-core:
  - A `pool_admission` `eabd0cb80` ([Bench 873](.benchmarks/873_pool_admission_goat.md)): G1 16/16; G2 75.6 ns/cycle K=32, 422.9 ns K=256, post-sweep fair_turn 0.3 ns; G4 0.
  - B `rate_control` `ee01f1598` ([Bench 875](.benchmarks/875_rate_control_goat.md)): G1 13/13; G2 42.0 ns/observe; G4 0; `b7f9fc370` SNAPSHOT byte seam.
  - C `dying` `650faeac7` ([Bench 874](.benchmarks/874_dying_goat.md)): G1 11/11; G2 1.30 ns/row; G4 0.
  - A and C were a same-day TWIN landing; origin's commits canonical (HISTORY.md § "2026-09-22 — Issue 873 primitive B landed twice").
- **Handed off:** B5 consumer A/B = riir-train Plan 416 Phase 2 (T2.2 wired into live Bonsai GDN SFT at riir-train `e2721e83`, T2.3 running on 4090, tracked by T2.5). C4 waits on first-consumer GOATs: ndb `shard_compactor` (assessed `07689b5c4`, needs a retrieval-trace A/B plan), riir-clippy corpus retirement, belief GC, riir-ai working sets; each files its own issue.

## Issue 884 (2026-09-25) — tetris_sim `hard_drop` tunnels through roofs: CLOSED (fix path A, site-side; corpus untouched)

- **Trigger:** owner screenshot (reflex.gist.rs/arena) of a floating remnant — that is **naive gravity**, standard Tetris, not a bug.
- **Real defect:** `examples/common/tetris_sim.rs::hard_drop` (and JS `hardDrop`) rests at the DEEPEST collision-free row scanning bottom-up, tunnelling under overhangs (roof row 17 cols 0..3: `O` lands 18–19, real 15–16). v2 exposure **3 of 2660 options**, 3 of 120 states.
- **Decision (Claude verdict, owner-gated per 878):** path A — live play filters top-unreachable spots (`tetris_view.reachableFromTop` / `liveOptions`); replays keep v2 order; sim and fixture UNCHANGED. Path B only as a new lane.
- **Landed:** reflex-site `c1fcf88` (filter + three tests incl. corpus count pinned at 3; golden 2660/2660, demo checks PASS), deployed (version `31744b04`). `arena_prod_smoke` not completed (no local engine).
- ⚠ The pinned count (3) is a TEST, so a v3 fixture fixing `hard_drop` reds it — the signal to drop the filter.

## Issue 880 (2026-09-24) — calibrated-mass router gate, `exact_mass_admit` consumer lane (a): CLOSED (shipped opt-in, G2 FAIL by construction)

Filed by the 4090 session from Bench 884's candidates; implemented on the M3 (CPU-only lane).

- **Shipped:** `katgpt_spectral::manifold_power_iter_router::gate_sigmoid_topk_mass_into`, opt-in `calibrated_mass_gate` (implies `manifold_power_iter_router` + `katgpt-core/exact_mass_admit`). Keeps incumbent logits + exact-k; weights σ((z − τ)/T) with τ solved so Σm = k; ranks by logit; zero-alloc.
- **DRY refactor:** incumbent `gate_sigmoid_topk_into` uses shared `router_logits_into` + `select_topk_desc_into`; t14 pins it bit-identical incl. tie order at β = 400 (`>` → `>=` mutation reds).
- **Measured** ([Bench 885](.benchmarks/885_calibrated_mass_gate_goat.md), M3, interleaved, 3 runs): |Σm − k| ≤ 9e-7 (incumbent ≈ N/2 regardless of k). Cost 11.0×/6.6×/1.75× at N=64 k=4 / N=256 k=8 / N=1000 k=100. G1 t15–t19, G3 PASS; G4 0.
- ⛔ **G2 bar (≤ 1.05×) inherited from a comparison of ALTERNATIVES** (Bench 884's 0.906× at k = N/10); the upgrade COMPOSES them (keeps the O(k·N) sort + adds sigmoid passes), so no calibrated gate can meet it. A ratio between two operators does not transfer to one built from both.
- Remaining lever: safeguarded-Newton τ solve in `katgpt-core::exact_mass_admit` (~5–8 vs ~32 passes, still ~3×), recorded in Bench 885, not filed. Promotion → future consumer GOAT; `.docs/09_feature_catalog/opt_in_features.md` has both entries.

## Issue 881 (2026-09-24) — the numbering sweep's standing backlog, three repos over their ratchets: CLOSED

Filed from the citation-sweep run: riir-ai `max_resets` 26 > 9, riir-shader
`max_hist` 5 > 0, mmorpg-editor `max_resets` 6 > 5 — an issue, not re-typed pins.
**Only the one real event loosened a pin, and it was read first.**

- **T1 riir-ai — 17 of 17 new resets were walker PHANTOMS** (`216defb9b`). riir-ai
  `892dec017` committed a two-line `.issues/.highwater`; `counter_history` framed
  `git cat-file --batch` by LINE, shifting every older value. Parser now reads by
  byte size (`parse_cat_file_batch`, 0 mismatches vs `git show` over 4594 events);
  riir-ai 9 = pin, workspace 46 → 28. ⚠ One commit can corrupt the READING of many.
- **T2 riir-shader — five collisions were renames `-M` cannot pair** (`3141a0467`,
  `_queue` → `_port`). Repair `numbering_gate.collapse_renames` (same commit or
  author ≤ 1h, Jaccard ≥ 0.25, predecessor first); line similarity rejected (real
  recycle katgpt-rs `.plans/236` scores 0.065). `removed_by_number` gained
  `--full-history`. Workspace 195 → 180 (15 collapsed, 2 surfaced).
- **T3 mmorpg-editor — one REAL reset** (`.plans 249→248 @ 86713c4e8c`, merge
  taking lower parent, Issue 770's shape); re-spent nothing (`.plans/249` one
  holder, counter at 250). Pin 5 → 6 with specimen in row.

Numbering sweep: **PASSED workspace-wide** at close.

## 2026-09-23 — the owner-gates menu v2 executed: schedules re-armed (row 5), pipefail residue closed (4e), toolchain batch-pin (4f), and the opt-in verdicts recorded (4a/4b/4c/4d)

Owner approved menu v2 wholesale; batch record:

- **Row 5 — CI posture:** three weekly schedules RE-ARMED (`test.yml` Tue 05:03,
  `full_gate.yml` Mon 04:17, `feature_isolation_weekly.yml` Mon 04:47) — repo is
  public, minutes free. `push` stays `branches: [main]`; private siblings stay
  suspended. **Owner infra left: self-hosted 4090 runner for ~8 private repos.**
- **Row 4e — pipefail "47 kill-shapes": closed as stale** (standing state 10 pinned).
  One unpinned residue `riir-neuron-db:release/dist-repo/install.sh:43` (empty grep
  skipped sha256 verification) fixed at riir-neuron-db `c5c11b5a273e` +
  `6f457a6260ca` (`|| true` + emptiness fail-loud). Sweep PASSED (10 findings, 0
  unpinned, 0 unparsed).
- **Row 4f — toolchain batch-pin landed; "13/20" stale** — real unpinned 5, pinned
  (1.98.1 + clippy/rustfmt): seal-game-editor `8c7755b3fdbb`, seal-online-remaster
  `c47227e06fcb`, katgpt-web `d0c917f1eb60` (default `main`), riir-llm
  `ee8b6c2bee6c`, riir-viewbridge `d54fbe8ebf32`. riir-esp32 has no root
  `Cargo.toml`. ⚠ `toolchain_override_audit.py` walks 0 files in the aliased
  `mmorpg-*` repos; seal-* pins verified on disk.
- **Row 4a — `certified_frontier` (Plan 580): opt-in; promote with first production
  consumer** (riir-ai Bench 822 closed the gates).
- **Row 4b — `gw_alignment` (Plan 594): do NOT promote** (riir-clippy `gw_corpus`
  Bench 083 precision@60 = base rate); reopen on a semantic fix-shape embedding.
- **Row 4c — `hint_regret` Phase 5 (Plan 576): opt-in; remaining arms unwired**
  until a consumer (landed consumer behind `demo_coverage_curiosity`).
- **Row 4d — Issue 815 options 1/3: answered-by-precedent** — Issue-842 alias codec
  retired `DOCS_GATE_KNOWN_EXTRA` (4090: 2/33 → 33/33); contract names already
  claim `mmorpg-editor`/`mmorpg-remake`/`mmorpg-remaster`. Local
  `scripts/repo_alias.local.txt` has the 3 rows (gitignored).
- **Rows 1 (riir-ai Plan 611), 2 (crates.io ×3), 3 (riir-dapps mainnet): recorded
  in home repos** — plateau accepted (b); publication closed/deferred; bundle under
  one trigger, `NO_REMINT` SET, second-live-mainnet closed.

Session: owner-gates-m2, 1790121600

## 2026-09-23 — Issue 877 CLOSED (merge `b701cf564`): the main↔develop sync — origin/main merged into develop with `-s ours`, zero content delta, ancestry restored

main was not an ancestor of develop: three main-only commits over merge-base
`37bb9cbf8`, all transplants (main's HISTORY: "Cherry-picked onto main from develop `8da93896`"): `569daf98e` lthash (twin of `8da938961` + `b85bda6e6`),
`3c844aebb` lthash wiring (inside `8da938961`), `5e2b730f2` exact_sigmoid (twin of
`5458dd69b`, evolved by `da89c386b` + `a36895e32` Issue 861 + `9b09783d9` Issue 870).
`git merge-tree`: 11 conflicted, 2 clean, 3 identical adds; every main hunk existed on
develop evolved (duplicate-`exact_sigmoid` trap refuted).

Merge `b701cf564` (parents `9ef11a3eb` + `5e2b730f2`) changes ZERO content. `-s ours`,
not `-X ours` (which still merges non-conflicting main content; tripwire
`HEAD^{tree} == HEAD^1^{tree}`). Pushed `--atomic origin develop HEAD:main`; the
279-commit range fired every main lane once (owner-accepted; a red is a true develop
finding).

10 files (`.issues/747_…`–`756_…`) exist on main, not develop — deleted under
noise-reduction; that diff is the record, not a loss.

Lessons: (1) **merge-tree hashes are spelling-bound** (conflict markers carry the ref
spellings) — quote the FILE SET, never the hash. (2) **a pin measured before the commit
carrying it is stale** (`4ad737a9…`); use self-referential `HEAD^{tree}` vs
`HEAD^1^{tree}`. Claude verdict, 3 rounds (REVISE → REVISE → AGREE).

## 2026-09-23 — Issue 867 CLOSED (+ the issue): the non-hidden-state canonical-AST construction — G5 returned NO attributable signal; instrument-broken, not claim-refuted

T1 `65b199b0` (38-bin `source_features` AST-histogram extractor, opt-in
`canon_source_features`) and T2 `67884461` (`SourceFeatureAdapter` ridge fit +
zero-alloc apply) shipped in `katgpt-canon`. T3 G5 cross-arch gate ran riir-train-side
at [Bench 605](../riir-train/.benchmarks/605_issue567_g5_source_features_gate.md)
(harness `48623505`, corpus `49a4a72f`): PRIMARY +0.50..0.55 at k∈{2,4,8,16}, **but the
fit-time shuffle null alone gives +0.41..0.44** (within ~1σ, clears the +0.3 bar) — no
attributable rung.

Closes because T4 ("control passes AND G5 fails") did not fire — control failed =
*instrument broken, never claim closed* (issue-825 clause); T5 needs an attributable
>+0.3 rung. Both permanently non-executable.

Findings: real **aggregate** contrast direction (AGGo +0.44 at k=16, noisy), weak
pair-specific correspondence (OBS +0.16..0.22 vs null +0.11..0.18). Reopen paths in
riir-train Bench 605 §Verdict(6): observed-level gate with mandatory fit-time null,
intervention study, and/or gemma-reliability-adequate corpus. Reopen authority:
Research 459 (CLOSED 2026-07-27, non-hidden-state constructions only). Proposal 010 =
G5-bar authority.

## 2026-09-23 — Issue 875 T3 CLOSED (+ the issue): time-annealed sampling ranges + the closed-form truncation predicate (Bench 883)

Last open task of the PFD modelless arm. `TimeAnnealRange` + truncation predicate
(`ε = ((T−t_cut)/(T−t_min))²`, inverse `t_cut = T − (T−t_min)·√ε`) in
`horizon_weights`; `dllm_solver` seam (`annealed_renoise_range` /
`renoise_level_skippable`) behind the combined gate. Quality gate ran CROSS-REPO on the
riir-train C9 toy (fixture consumed in place; deviation from C9's "vendors" note
recorded). Release, gate_full: flat 0.4084 = C9's PFD (behavior-preserving);
AnnealPlain +9.0% W1 for 9.37% mass dropped — **truncation cost law measured**;
AnnealRenorm +4.4% (within not-worse bar) — safe-not-a-win on the two-ring toy; the
fine-detail claim routes to riir-train 569 C5. Validated in an 8-worktree redirect
harness running the REAL katgpt-core symbol. Issue 875 CLOSED: five tasks, all arms
OPT-IN.

## 2026-09-22 — Plan 607 ACCEPTED + T0a/T4a/T0b landed: the modelless game-decision lane's first three tasks (substrate gate, the laya-Tetris enumerator, the G1-oracle fixture)

Owner accepted ("607 accepted"). Thesis: laya's protocol (code arithmetic → templated
English → BERT) is the closed-grammar regime where corpus-limited scoring is lossless;
prove the modelless lane wins latency/deployment by construction and accuracy by
protocol, with a GOAT gate.

- **T0a:** six substrate pieces adjudicated. T1 = NEW `state_option_scoring` module
  consuming `exact_sigmoid` + `float_order::cmp_for_max` + unit-normalize; reflex
  `engine.rs` route_terms is the upstreamed shape; drafter stays OUT of the hot loop (R3).
- **T4a:** `examples/tetris_01_state_enum.rs` + shared `examples/common/tetris_sim.rs`.
  120 deterministic states (12 archetypes × 7 pieces + 36-state Dellacherie ladder),
  2,660 options, byte-identical. **Protocol correction:** laya Tetris is one sentence PER
  SPOT → P(clean) → code argmaxes; number WORDS. **v1→v2 grammar:** v1 had 39 distinct
  sentences, tying 66/120 states; v2 adds landing-side + resulting-height (245 distinct),
  cross-sentence ties 0, 33 residual same-sentence ties. Pre-clear law: uniform-height
  floors are pre-full rows, so heights archetypes carry a 0-height shaft.
- **T0b:** generator committed to riir-reflex first (`laya_oracle_batch` @ `e4bf657`,
  generic), 2,660 noul forwards (396 s CPU) → `tests/fixtures/tetris_oracle_laya_en_v2.jsonl`
  + README; byte-identical re-run verified. p_clean 0.027–0.850; Spearman vs Dellacherie
  mean +0.365 (positive 94/120) — not constant-picking (reflex Issue-004-T7 trap checked
  before freezing).
- **Next:** T1 + T4 co-developed, first GOAT gates {T2, T3}; T5 precedes any default-on (R4).

## 2026-09-22 — Issue 874 closed: G8 re-founded on real claims — the dead speedup gate became a bit-contract pin + a relocated, executing throughput floor (two sessions, one issue)

Issue-871 T5 fallout. Census session (T3, `ec345a8c`+`e880d97c`+`6c2d8b04`) read all 16
`*_vs_scalar` sites: the trap is ONE gate (g8), plus two riir-ai by-catches (`f04df9241`).
Repair session took T1 (verdict, round 2 REVISE folded) and T2.

- **Dead both ways:** G8 asserted ≥1.5× SIMD vs scalar, but both routes run the same
  `dot_8wide` (Plan-271 consolidation), strict-ordered with no SIMD (Issue 871 T5); the
  "SIMD" arm also ran stabilize, so ratio <1 and it always SKIPped. Pre-consolidation
  3.01× NEON (2026-06-14) was real; later PASSes were noise.
- **Verdict** (3 rounds, AGREE): Option 3, retire the relative claim. Option 1 refuted
  (strict ordered reduction can't vectorize without changing bits); Option 2 swaps the
  claim; the real contrast is `algebraic_dot` (Bench 871).
- **G8a `g8_route_bit_agreement`:** exact `to_bits` over 16 384 outputs between routes;
  failure names both causes (summation order, `inv_sqrt_d` derivation).
- **G8b `g8_throughput_floor`:** relocated from in-crate `test_simd_throughput_smoke` (n=8,
  t=512, d=64, 5 ms) onto an executing lane (bench_271 in the x86_64 matrix); `best_of_us`
  min-of-200 + black_box. 50.3 µs/call, 100× headroom — regression floor. Duplicate deleted.
- **Validation:** bench_271 10/10 release; crate lib 121; clippy clean;
  `timed_region_guard_gate` PASSED.

G7's release fallback printed "100000 calls in 0ns" (constant-folded). Fixed in follow-up
(2026-09-22): result consumed inside the timed region, loud-zero assert.

Closed in place, then removed 2026-09-22 (noise rule; 16-site census recoverable via
`.issues/874*` history). Census sibling's `27b48552` T3 addendum landed mid-rebase
(bench_256's stale "SIMD (ns/call)" labels repaired). Healthy population defended three
ways (bench_148 anti-vectorization, fast_bpe fallback floor, bench_578 loud skip).

## 2026-09-22 — Issue 873 primitive B landed twice in one evening: the twin-duplicate resolution (third of the class)

Two sessions took "primitive B (`rate_control`) remains" from one handoff summary. The
sibling's `ee01f1598` (21:12 +0700) reached origin first with a complete GOAT-gated port
(Bench 875); the second push was rejected non-fast-forward and its duplicate
`git reset --hard` away (Batch-169 precedent). Its verifier ran the winner's gates on the
reset tree (13/13 G1, G2 PASS, docs 637) and read the module: no defect; differing
constants are documented defaults.

- **Class, third instance** (riir-clippy Batch 169 → Issue 825 → this): a handoff summary
  naming remaining work is an ASSIGNMENT to every reader. `git fetch` + check before
  pushing a primitive; on the twin tell, drop yours and VERIFY the winner.
- **Follow-up RESOLVED same day:** sign convention (rising `val` = improving,
  `e = +slope/σ`, opposite of mini-AGI's loss) now in the module doc ("Sign convention",
  `-loss` fix), not waiting on riir-train Plan 416 Phase 2.

## 2026-09-22 — Issue 871 closed: the `algebraic_*` A/B measured end-to-end — feature-gated `algebraic_dot` ADOPTED (owner verdict a′), the in-crate codegen truth repaired, T6/T7 deferred on-record

T1–T3 measured (`e2f77e70`), T4 verdict (3 rounds, (a′) mechanism-only GO), adoption
(`c108f2fe` + `fbe86280`), T5 codegen verification (`2331961c`).

- **Shipped (T4):** `katgpt-attn-match` opt-in `algebraic_dot`, twin of `dot_8wide`
  (1.77×–8.8× at both x86_64 arms, G1 accuracy IMPROVED on cancellation data), zero
  consumers; strict stays default. Six module tests (accuracy-not-worse, NaN no-poison).
  First argmax consumer requires a real-logits retention walk (Issue-750-T3 shape); the
  1024-trial walk was low-power (9 near-ties).
- **Ban (`c108f2fe`):** `algebraic_div`/`algebraic_rem` banned, enforced by
  `scripts/algebraic_op_ban_gate.py` (ceiling 0, floored walk, planted arms, masker from
  `platform_dead_code_audit`).
- **T5 codegen (`2331961c`):** 0 packed float-math at both x86_64 arms (122 scalar
  `mulss`/`addss`) — "auto-vectorizes" claims false. aarch64: packed `fmul.4s` + ordered
  scalar `fadd`, NO `fmla`; the 2026-07-29 "optimal fmla" story was never true, and the
  1.26× refutation's mechanism is a bounds-check confound. Nine claim sites repaired.
  Strict-side repair REFUTED: serial add order IS cross-arch bit-equality.
- **Ill-conditioned G1:** algebraic ≥ strict both classes (ortho 4.58e-9 vs 8.25e-9, 191
  vs 256 flips; rank-deficient 6.6e-18 vs 5.4e-9, 0 vs 245); f978a20b counterexample does
  not reproduce.
- **Issue 874 stays OPEN** (g8 both routes via `dot_8wide`).
- **Deferred:** T6 (Bench 871 on M3/aarch64), T7 (riir-clippy `rust_perf` rule with first
  consumer).
- **Three-way collision:** the sibling's duplicate yielded after a TOML duplicate-feature
  error (Issue-665 discipline; recoverable at `2331961c`).

Validation: clippy both states, 6/6 tests, ban gate over 2533 files, docs gates green;
`numbering_gate` + `issue_citation_gate` resolve "Issue 871" via `removed_by_number()`.
Issue removed; record [Bench 871](.benchmarks/871_algebraic_dot_ab.md).

## 2026-09-22 — Issue 872: SIMD bitstream whitespace splitter for `encode_into_pretok` (the bitcannon-class port) — scan 1.60×/1.69× measured, bit-identity by differential, four live bugs caught by the harness

Bench 191 §Phase 3's deferred SIMD pretokenization, unblocked by HF `tokenizers` v1's
"bitcannon" post (2026-09-21) showing the technique on stable Rust (Bench 191 blamed
nightly `portable_simd`).

In `katgpt-tokenizer` under opt-in `fast_bpe`:

- `fast_bpe/simd_split.rs` — zero-alloc `WhitespaceSplitter` (borrowed ranges;
  `pretoken_bytes` buffer deleted). 16B SSE2 / 32B AVX2 (runtime-probed, the
  `shipped_target_feature_gate` law) / 16B NEON (SWAR movemask) / scalar. Bytes ≥ 0x80
  via `char::is_whitespace` — Unicode bit-identity. ASCII ws runs coalesce.
- Measured (loaded i7-13700K, median of 9 interleaved): **avx2 1.60× vs scalar-mask,
  1.69× vs old per-char loop**, scan-only. NEON typechecks via cross `cargo check`.
- Differential harness caught **four bugs**: (1) `u8::is_ascii_whitespace` excludes VT
  (six ASCII `White_Space` bytes); (2) `cmpgt(v^0x80, 0)` misses `b == 0x80`; (3)
  `1u32 << 32` release-wraps to zero mask; (4) multibyte ws at word end swallowed. All pinned.
  Bench 872, Research 580.
- Gates: lib 25/25; `fast_bpe_goat_simd_split` 3/3; pretok GOAT + G4 zero-alloc green;
  clippy clean; wasm32 + aarch64 cross-checks. Pre-existing, not this:
  `fast_bpe_goat::g2_perf_smoke_per_call_short_input_documented_regression` fails at clean
  HEAD too (M3-calibrated).
- fast_bpe stays opt-in. Issue removed.

## 2026-09-22 — Issue 870 closed: distance_abstain's rationale-free sigmoid copy → exact_sigmoid delegation, measured 3-ULP envelope, bench_845 GOAT re-run identical

Substrate-first Mode 2 audit (`683d06d8`) found `distance_abstain.rs:47` with an unpinned
local `1.0/(1.0+(-x).exp())` beside sanctioned patterns (`closure/bridge.rs`, `d2f` →
`fast_sigmoid`; six modules → `exact_sigmoid`). Copy-class family (riir-neuron-db Issue
611 / riir-chain Issue 156).

- **Delegation** to `crate::exact_sigmoid`, envelope documented in-source.
- **Measured:** ≤1 ULP pin draft RED; sweep found **max 3 ULPs for x<0** in [−20, 20]
  (x=−4.851, x=−16.743), bit-identical x≥0. Pin
  `sigmoid_delegation_matches_frozen_legacy_body` freezes the legacy body, bound 3 ULPs.
- **GOAT:** bench_845 G1–G5 PASS, identical (W1 fused AURC 0.1894 vs 0.2289, Δ(ρ=30%)
  +0.0097, 8/8 positive).
- Bench's own W1 `sigmoid(1.2·logit p − 0.2)` (Research 576 §2.1) is the world
  definition — untouched.

Validation: `cargo test -p katgpt-core --features distance_abstain --lib distance_abstain`
8/8; clippy clean; bench_845 PASSED. Issue removed (recoverable from `683d06d8`).

## 2026-09-22 — Issue 869 T5: the mini dllm lane goes per-layer honest end-to-end — the gradient check caught a live backward bug the loss-decreases gates never could

Training, eval and decode honor `config.n_layer`; `D2fContext` decode depth defaults to it
(old default 1 is now explicit `set_decode_layers(1)`).

- **Training** (`src/dllm/mod.rs`): contexts carry `n_layer` × `block_size` planes;
  `forward_save`, `forward_save_set_causal`, `backward` (three-phase per layer, reverse),
  `sgd_update` per-layer; eval forwards generalized (incl. bench_602's
  `CpuSetCausalForward`). Bit-identical at `n_layer == 1`.
- **FINDING — backward bug:** the `!is_masked[p]` skip is valid only at the readout layer;
  inner layers get gradient at unmasked positions. It corrupted layer-0 attention grads
  under partial masking (~10-150% per-element, attn_wv sign flip), EXACT under full
  masking — why loss-decreases gates passed. Caught by
  `two_layer_backward_matches_finite_differences` (1-layer <0.7%). Fix
  `last && !is_masked[p]`.
- **`bench_602_ar_ness_cross_tab` calibration is model-class-specific:** AR-drag direction
  within seed noise (seeds 42-45: ±0.13 around ~+0.03, 2/4 invert; pre-T5 PASS a coin
  flip); `ORDER_STATS_TO_W_TABLE` worst retention 0.914 < 0.95. g3 now 2 seeds/regime,
  reports direction + retention, hard-gates liveness + derail floor (≤ worst-fixed ×1.05).
  Re-open at Bonsai scale.
- **Text dividends:** bench_601 margin 0.39 (NLL 2.499 vs unigram 2.888, bar 0.15); 809
  seam-parity byte-identical; 817 2/2; 602 3/3. New pins incl. gradient check at both depths.

Gates: root lib 211/217; katgpt-forward 131/167/180/181; dllm 27/27; pattern-lane
bit-identity; text benches green; `full_gate.sh --allow-partial-platform` clean (Windows
PARTIAL). Files: `src/dllm/mod.rs`, `src/dllm/tests.rs`,
`crates/katgpt-forward/src/{forward_positions, forward_set_causal,d2f_context}.rs`,
`d2f/tests.rs`, `tests/ bench_602_ar_ness_cross_tab.rs`, `.issues/869_multi_layer_d2f_taps.md`.

## 2026-09-22 — Issue 869: multi-layer D2F decode + taps at depth — the Issue-865 Bonsai-scale unblock lands; bitcos x86_64-lane clippy debt repaired in passing

Executed the riir-train mandate ("Bonsai-scale probe training is gated on the katgpt-rs
multi-layer D2F kernel extension"; Issue 865's re-open condition, Bench 847/850):

- **T1 kernel:** `forward_block_causal_with` over `D2fContext::decode_n_layer` —
  per-layer KV planes (`l * block_size * kvd`), chained residual (`h_0 = rmsnorm(emb)`
  keeps layer-0 double-norm quirk), logits at final layer. **Depth defaults to 1**
  (preserves every pinned gate); multi-layer is `set_decode_layers(n)` opt-in.
- **T2 taps:** `set_probe_tap_layers(&[usize])` (validated; too-deep tap panics), layered
  `probe_tap_flat` (`[slot][pos][dim]`), `ProbeCtx { tap_layers, tap_plane }`,
  `WeakLogitProbe::tap_layer()`, `set_guidance` install-time validation. The old
  `tap_layer != 0` rejection is REMOVED — the unblock. Default `[0]` byte-compatible.
- **Finding:** `micro_dllm_text()` declares `n_layer = 2` but the whole mini lane
  (training, eval, decode) indexes `layers[0]` only — layer 1 never trained or read. T5
  (own landing) migrates it. Not required for the Bonsai GPU lane.
- **Verification:** depth-1 bit-identity via every pinned gate (fixture G0 + goat 5/5,
  bench_601 3/3, bench_809/817/602, bench_600 ×2, dmax_spd, tri_mode, ugc_g1b, dllm lib
  23/23) + 10 new tests. `scripts/full_gate.sh --allow-partial-platform` clean.
- **Rider — bitcos x86_64 clippy debt** (`katgpt-types/bitcos.rs`, `simd/bitcos.rs`,
  `bench_864_bitcos_goat.rs`; Issue 864 landing — 9 arch-gated findings, Issue-819 class):
  `manual_isolate_lowest_one` ×4, `needless_range_loop` ×2, `manual_is_multiple_of` ×2
  rewritten; bitcos 13/13 on avx2.

Issue: `.issues/869_multi_layer_d2f_taps.md` (T1–T4 landed; T5 open). No bench — no perf
claim; perf belongs to the scale lane (Bench 847's inverted bars).

## 2026-09-21 — Issue 864 closed: BITCOS tier ships opt-in — footprint PASS, latency honestly LOSES on this host (Bench 846)

`Issue 864` from [Research 577](.research/577_BITCOS_Distribution_Adaptive_Ternary_Layout.md):
`bitcos` in katgpt-types — presence bitmap + compacted neg-sign stream (2−z bits/w + f16
scale), z-meter `zero_density_report`, three GEMVs (scalar bit-identical; 256-entry LUT,
GPU-portable; pdep+SWAR AVX2 runtime-probed AVX2∧BMI2), dispatch `should_use_bitcos(z, γ, β)`.

- **G1 PASS** (bit-exact roundtrip, scalar/LUT identity), **G2 PASS** (footprint beats both
  tiers above z=0.375 — 0.950/0.892/0.854× vs trit — larger below, asserted), **G4 PASS**.
- **G2b(a) FAIL → opt-in:** >L3 streaming at z=0.5 bitcos 0.767× vs bit-plane SWAR /
  0.846× vs trit. γ = 1.23 B/ns < β = 1.96 B/ns — instruction-bound on this 13700K (paper's
  Lunar Lake class). Dispatch refuses; verdict == gate-outcome assert pinned.
- **Two codec bugs pre-timing:** pack must pext not pdep (tripped `is_canonical`); scalar
  pos plane is `p & !neg`. Research 577's "compacted pos bits" is inverted vs its own
  convention — neg-compaction ships.
- Record [Bench 846](.benchmarks/846_bitcos_goat.md); feature count 626→627; riir-gpu CUDA
  LUT-arm pointer kept.

# HISTORY.md — katgpt-rs

Historical record moved out of `AGENTS.md` (2026-09-06 compaction, from commit `1801c0ab`);
every section preserved verbatim from the pre-compaction `AGENTS.md`.

## 2026-09-22 — Issue 868 closed NEGATIVE: engram-fused PUCT G5 FAIL — the evidence gate worked, and that is why nothing happened (Bench 848)

`Issue 868` (file removed) via [Plan 605](.plans/605_engram_fused_puct_poc.md): Proposal
013's engram×PUCT fusion — `engram_puct` (opt-in, native-gated), `engram_fuse.rs` (TT-key
`(board, ko, to_play)` → 4 words; BLAKE3 `MinedTable`; evidence gate σ((n−8)/4); damped
Q-init `visits=1, total=gate·v̄`; prior sharpening γ ∈ [0.5, 2]), miner + arena examples,
GOAT tests. Feature-off is source-identical; T2.1 **size-identical** wasm (381,353 B), no
katgpt-core in default wasm32.

- **G5 FAIL — 296/616 = 48.1%** h2h, Wilson lower **44.8%** < 50%. Budget arm (fused b25 vs
  plain b50) **35.7%**. Control vs GREEDY: fused 93.0% vs plain 97.0% (**−4.0 pp, n.s.**).
  G2 **293 ns/read** vs <100 ns bar.
- ⛔ **Control arm's first numbers corrupted by a reward-inversion bug** (51/51 "tie" was the
  artifact); caught against harness reference (85–94% vs GREEDY), fixed, re-measured.
- **Interpretable negative:** memory FIRED on 98.4% of 1.06 M lookups but 100% of 16,930
  mined positions have n < 4 (136 repeats in 17,066 plies), so the gate correctly damped
  99.98% to ≤0.18. Collisions 22.0% (2²⁰ slots), mooted.
- **PASSED:** G1 (empty table bit-identical), G4 (zero-alloc), G6 (deterministic,
  tamper refusal), Q-init direction arm.
- **Re-open:** transposition-dense domain, neighbourhood generalization, or larger corpus +
  re-mining (`EngramHotSwap`). [Bench 848](.benchmarks/848_engram_puct_arena_g5.md);
  Proposal 013 → MEASURED NEGATIVE.
Operational rules live in `AGENTS.md`; removed issue files: git history.

contents: modelless-first canonical-failure story · full-gate narratives ·
docs-gate descriptions · cfg-gated / required-features / percentile audit
histories · staged-set + shared-target-dir narratives · feature-flag rule
history (lossy surface, Report the Floor, Plan 467) · the Repo count
paragraph's drift history · the resolved issue log.

## 2026-09-21 — Issue 858 closed: g8's arch-conditional bar gets its aarch64 executing lane (PERF_ROWS)

**RESOLVED + REMOVED. Record: `.issues/858_g8_cached_faster_than_uncached_is_RED_on_aarch64.md` (git history).**

**Finding (2026-09-19):** `g8_cached_faster_than_uncached` in `tests/belief_drafter_goat.rs`
red on the M3 alone — median 0.6844–0.7111 over 5 runs vs `ab.median < 0.5` (Plan 217,
calibrated at `dd8dadbba` on x86_64: 0.39–0.41). Load, concurrency, unseeded draw and Issue
855's vanished-work class all excluded. Found by accident.

**T1 (4090):** 13/13 x86_64 PASS — +avx2 0.4523–0.4712, plain 0.3896–0.3973. Arch gap ~0.23
≈ 6× spread. **Arch CONFIRMED; cache regression REFUTED.** (`dd8dadbba` matches the plain
build; +avx2 headroom under load 2.9–4.8 points.)

**T2 dual pin:** `G8_BAR` aarch64 0.75 (worst 0.7111 + ~5%), else strict 0.5 (unmeasured arch
must meet the claim). Bench-806-T7 form.

**T3 lane gap:** new `scripts/test_gate.sh` row kind **`PERF_ROWS`** (`pkg:floor:target`,
`--release --test-threads=1`), first row `katgpt-rs:12:belief_drafter_goat` (0.03 s; floor 12
= full inventory, arch-invariant). `--canary` floor-bombs each list. **4090: PASS
207/2063/249/150 + 12/12; canary FAILS as designed.** aarch64 reading on next M3 run. Record:
`.docs/10_audits/ci_compile_vs_execute_axis.md` §2026-09-21.

## 2026-09-21 — the x86_64 execution matrix caught a day-old test that had never executed on x86_64: the ordered-dot anti-dedup pin was crafted at NEON's vector width

**RECORD 2026-09-21 · matrix at `b6dc1d16` (cells 1-6 green) · repair in this commit.**

Cell 7 (katgpt-types `--lib --all-features`, `+avx2`, debug) CONFIRMED red (3/3 alone) on
`simd::dot::ordered_dot_tests::ordered_dot_differs_from_simd_dot`, landed the day before in
`5458dd69` (`dot_f32_ordered`, riir-chain Issue 156 T1, Bench 844). Never executed on x86_64
before.

**Mechanism:** the pin asserted `simd_dot_f32 != dot_f32_ordered` on a len-4 input — NEON's
width, but below AVX2's 8-wide floor, where the scalar tail IS the ordered fold (1.0 == 1.0).
The scalar fallback converges at len 4 too. Kernels correct; the TEST's craft was the defect.

**Repair:** recrafted at len 16 (smallest length engaging a grouped path on every backend).
Input `[1e8, 1, 1, 1, −1e8, 1, 1, 1, 0×8] · [1;16]`: ordered = 3.0, every reassociating
backend 6.0 (hand-computed per backend). Ordered pinned `assert_eq! 3.0`; doc records the
width dependency. Rerun 7/7.

Cell 8's PASSED-ALONE rows (`bench_176_router_forward_cpu`, `t3_latency_p99`): sequential
latency BARs (Issue-723/833 family), no finding. `X86_MATRIX_DIR` needed on E: (C: 99%).

## 2026-09-21 — the first full-gate run since 09-16 caught the Issue-860 landing RED: 8 `-D`-list errors in the opt-in feature's test code, invisible to every default-feature lane

**RECORD 2026-09-21 · gate run at `c9939347` · repair in this commit.**

First full gate since ~09-16: Layer 3 red with **8 errors in `successor_density_critic.rs`
`#[cfg(test)]`** (5× `needless_range_loop`, 3× `identity_op`, e.g. `b.n_sa[1 * 2 + 0]`).
Issue-803 class on the non-default axis — opt-in module compiled to nothing everywhere but
all-features; 860 landing never linted `--all-targets`.

**Repair:** `sa` closure beside `cell`; `nxt.iter_mut().enumerate()`; `exact.iter()`
enumerate. 12/12 tests green. Healer took 6 `doc_markdown` edits; loop/identity manual.

**Riders:** `set_diffusion_schedule.rs` `(0.3..=0.5).contains(&w)`; `ugc_schedule.rs` dead
`TableJoint::index` deleted; `bench_602_ar_ness_cross_tab.rs` `CHANCE_NELBO` bounds moved into
`const { }`.

**After repair:** `⚠ full gate PARTIAL — every layer that RAN is clean (0 errors, 0
unbuildable)`, 1120 s warm.

## 2026-09-20 — the x86_64 execution matrix's first full run since 09-16 (354 commits of drift): 11,344 assertions PASSED, and one more load-flipped bar caught by execution

**RECORD 2026-09-20 · matrix at `7f10d4b7` · repair in this commit.**

First run since ~09-16 (354 commits, incl. Bench 816/817/818 and the 841/843 landings), alone, quiet box
(16.6/17.6 GiB avail):

- Cells 1–7 green: katgpt-attn 440 · core 5136 · dec 298 · pruners 3025 · rs 577 · tokenizer
  74 · types 266 (counts ~doubled since floors set).
- Cell 8: 223 targets, 1528 passed, 1 failed → **`bench_105_gdn2_goat::goat_6_context_scaling_flat_o1`
  PASSED-ALONE 3/3** (0.306 vs 0.30 bar). Membership set stays empty.

Fourth Issue-833 member found by execution (GOAT 2 third): four positions each in ONE
sequential window, `(max−min)/mean < 0.30`. Seeded, no temp paths — pure window sensitivity.

**Repair:** `tests/common/ab_timing.rs` gains `best_of_arms` — round-robin N-arm sampling,
per-arm MINIMUM, loud-zero (neither `best_of_us` nor two-arm `ab_median_ratio` covered it).
GOAT 6's GDN2 side migrates; flat-KV side stays sequential (~67% slack). After: spreads
0.038 / 0.051 / 0.102 vs 0.30. `cargo clippy --test bench_105_gdn2_goat -- -D warnings` clean.

## 2026-09-20 — Issue 861 CLOSED: the ugc_alloc_check Windows G4 alloc was a per-call env read — and the issue's own isolation table was wrong

**CLOSED 2026-09-20, fix `1ec9b812`; full text: git history.**

`ugc_alloc_check` (Issue-664 G4) failed 50/50 on Windows/MSVC: 1 alloc/iteration,
debug-only, green on macOS. The filing table ruled out `estimate_interval` and blamed the
sampler.

The prescribed `Backtrace::force_capture()` armed-window allocator (disarm before capture)
settled it the OTHER way: `std::env::var::<&str>` ←
`katgpt_core::ugc_schedule::estimate_interval` at `src/ugc_schedule.rs:353`, via
`to_u16s -> Vec::with_capacity -> getenv` (size 20, align 2).

Root cause: `#[cfg(debug_assertions)] if std::env::var("UGC_DEBUG").is_ok()` per call in the
hot path; Windows converts the name to UTF-16 `Vec<u16>` (Unix `getenv` doesn't allocate). A
lone sampler call allocates ZERO — the filing row was wrong.

Fix: `ugc_debug_enabled()` caches in `OnceLock<bool>` (the `tpr::kill_switch` shape).
Verified: G4 **0 allocations** (was 50) · `ugc_664_poc` 12/12 · lib ugc 6 + 12
(`--features decode_order_metrics`) · clippy clean.

Lesson: **a debug-only logging gate still runs on the measured profile**; every hot-path env
read takes the `OnceLock` form.

## 2026-09-20 — Issue 859 CLOSED: Jev structured reads — POC GOAT + 4090 reference + T5 policy arm measured; promotion declined on layer posture (evidence-banked)

**CLOSED 2026-09-20, T0–T6 done; code + records `72718f81` (pre-rebase `d1075311`); full text: git history.**

Research 574 (vLLM PR #57250 read-only structured decisions) → Issue 859 → substrate mapping
first → POC `structured_read[_into]` + `sample_label_index` behind opt-in `structured_reads`
(katgpt-forward), G1/G1b/G2/G3/G4 PASS (Bench 816: exact logprobs, canvas bit-identity,
0.4588× latency, 154/154, 0 allocs) → T1 on the 4090 (PR overlay on vllm-openai:nightly;
corpora 10/10 · 10/10 · 9/12-corrected; ~4.4 GiB mm-video-profiler finding; Research 574 §8,
Issue-665 yield) → **T5 (Bench 817):** agreement bars (N=4, t=1) do NOT beat single-read
analytic confidence (Δ CIs [−0.089, −0.071] / [−0.056, −0.030]); **maxprob beats entropy on
wide sets** (Δ CI [+0.015, +0.026]) → guidance `label_entropy` narrow / `argmax_label_prob`
wide → **T6: stays opt-in, evidence-banked** — conditions MET but `katgpt-forward`
`default = []` is minimal, zero consumers (flashar_anchor precedent). Root feature forward
landed; re-arm = first production consumer.

Reopen: a production consumer, or a deployment needing per-read stochastic forwards (data-spec
in Research 574 §8).

## 2026-09-19 — the open-issue backlog is cleared by owner call (12 files removed; triggers recorded here)

Every open issue adjudicated in one owner session: resolved work got a closure record,
open work a PARKED / PULL-GATED / DECLINED verdict with reopen trigger; files removed (git
history; `number_collisions_expected.txt` rows untouched — closing a holder makes the pin the
only record). No number reused.

## Issue 780 — CLOSED as PARKED: OnlineLinearReadout waits on a consumer that measured itself absent (2026-09-19)

**Status: CLOSED (PARKED) 2026-09-19, owner call.** The primitive (third half of
the linear-probe lineage — frozen offline / indicator bank / online fit) stands,
but its live consumer closed NEGATIVE first (riir-clippy Issue 107, `5dfa1daff`
2026-09-15: horizon 1 < 2 on every corpus — the strategy prior is a sufficient
statistic; T3–T6 closed UNBUILT), and T5's promotion bar needs a live consumer.
Building now = a synthetic-fixture GOAT pass speaking to nothing measured.
Blocker corrected 2026-09-17 (`15386365`). **Reopen triggers** (any one, from
riir-clippy 107): (1) an ORGANIC fixseq ring with ≥2-revert runs AND the
span-embed column populated (the lane's only revival path); (2) riir-train
densifying the store per R135 / Bench 047; (3) any post-keep-fix corpus where
resolved rate moves between orderings. Type sketch + mutation-class
justification in git history (`780_online_linear_readout_primitive.md`).

## Issue 827 — CLOSED: T4 decided by owner call — known-extra repos ARE legitimate oracles; the freshness guards T2/T5 landed are the correct limit (2026-09-19)

**Status: CLOSED 2026-09-19.** T1/T2/T3/T5 landed (`6d084c38`, `7464fc6e`,
`3948f0e2`): ORACLE-STALE bucket (never clean/counted/ratcheted — suppression
at QUALIFICATION, not the tag), per-row oracle disclosure, two-sided fixture
arms, and the fetch-age `⚠ UNVERIFIED UPSTREAM` advisory so `(0, 0)` no longer
means "not measured". **T4 (owner-gated): STATUS QUO — a known-extra repo
remains a legitimate oracle.** The four motivating citations (riir-shader →
seal-game-editor) are followable to a real repo; refusing adjudication would
lose information, and the measured hazard was FRESHNESS, not AUTHORITY — now
guarded twice (T2 behind-origin suppression + T5 fetch-age line).
`DOCS_GATE_KNOWN_EXTRA` keeps the box-level acknowledgement loud. Reopen if a
known-extra repo is REMOVED from a box while citations still name it.

## Issue 833 — CLOSED: the class is measured, the executed backlog is 3 of 38, and the residue is routed (2026-09-19)

**Status: CLOSED 2026-09-19.** T1 (`bench_105` GOAT 2 migrated) and the second
firing (`g8`) hold in-cell; T3's resolvers (provenance-bound, REL-DIFF,
comparison-shape, `n == 0` ordering fix — `381f01f7`, `e56e6773`, `1b7756c3`,
`71d97908`) closed every STATED blind spot they could; T2 EXECUTED all 38
root-`tests/` GATES rows in release at their required-features (`1f519578`,
`c55a3238`, `b0736d8c`) — **3 candidates** (bench_176 ×2, bench_164), plus two
other-class defects filed separately (Issues 855, 856 — both closed). **T4: no
verdict half, deliberately** — migration is a four-axis per-target read
(orientation, claim direction, chunk size, `black_box`), never a codemod.
Residue: the 3 candidates + bench_008's doc-vs-assert divergence convert when
the x86_64 matrix next fires them (PASSED-ALONE is the discovery instrument —
Issue 834); the non-root half is answered by Issue 834 T3 (no shared crate).
Record: git history (`833_sequential_ab_timing_ratio_is_a_box_measurement.md`).

## Issue 834 — CLOSED: census shipped, the count reframed as a sampling population, and the shared-crate question is decided NO (2026-09-19)

**Status: CLOSED 2026-09-19.** T1 shipped `scripts/sequential_ab_timing_audit.py`
(`8045fae9`, phantom-repo correction `e1a572f2`); T2: the DECIDED count is a
population every matrix run SAMPLES FROM (two firings, one pre-committed clean
run 3 — `60264768`), record-the-NAME discriminator adopted, matrix discloses
box state on its verdict line (`d8bfa9b8`). **T3 owner call: `ab_timing.rs`
STAYS a `#[path]`-included module — NO shared crate**: katgpt-rs is the public
upstream funnel and a test-harness crate adds a versioned public surface for a
test-only concern; DRY residue is 9 hand-rolled twins across three siblings vs
12 adopted sites here — below a published crate's maintenance cost, and per
Issue 833 T4 a twin is often the better instrument for its claim direction.
**Reopen threshold: a fourth sibling hand-rolls a twin.** T4 stands (no verdict
half; re-measure after the residue converts).

## Issue 835 — CLOSED: T4 decided by owner call — the path dep is the workspace convention; publishing is contrary to Research 003 (2026-09-19)

**Status: CLOSED 2026-09-19.** T1 (docs-gate CHECK 28, `cross_repo_path_dep_gate.py`,
`5b09c3f4` + unexercised-buckets disclosure `aba3827f`), T2, T3 done 2026-09-18
— both questions measured and DECLINED an instrument; riir-llm cloned and
registered, gate green over 311 deps / 183 manifests. **T4: riir-llm keeps its
path dep, not published** — path deps against `/git/` ARE the workspace
contract, Research 003 ("anything `riir-*` is internal") forbids a public
registry surface, a git dep adds version-sync overhead, and no consumer asked
to escape the layout. Reopen if a consumer outside `/git/` needs riir-llm.

## Issue 839 — CLOSED: kron_tile 4/4 GOAT; T7 DECLINED on measurement (2026-09-19)

**Status: CLOSED 2026-09-19.** T1–T6 + T8 done (`a623d7d4`, Bench 839 — all
four gates PASS, stays opt-in per the no-default-consumer rule; T2's bit-parity
ask replaced by a measured 4.3e-7 tolerance pin plus `W² = I` /
factor-orthogonality anchors). **T7 (ternary-factor fusion) DECLINED**: Bench
843's width sweep measured ternary/f32 **1.56× slower at w=32, 2.24× at w=64,
peak 3.70× at m=512** (`5a1da65d`) — a 32×32 f32 factor is already L1-resident
at 4 KiB and add/sub-accumulate loses to FMA at every tile width, so the latency
premise is refuted; quality was never modelless-decidable (riir-train's lane).
Reopen only if a ternary kernel beats f32 FMA at these widths on measurement.

## Issue 841 — CLOSED as lead-backlog dissolved: 7 rows landed, the remaining leads stay at their research-note homes (2026-09-19)

**Status: CLOSED 2026-09-19, owner call.** Landed: `.kpt` archive POC (opt-in
`kpt_archive`, Bench 841, 17 bad-injection arms, owner gate D3); fix_verify
refusal-vs-truncation audit (repaired in riir-clippy `651728c0`); riir-rag
embedder identity (verified; wiring filed as riir-ai `.issues/983`);
grammar-forced vocab-projection skip (`legal_token_set` — PROMOTED default-on,
20.6–22.5× tree build); KV permanent sinks + bounded window (`kv_sink_window`,
5/5 GOAT, opt-in, promotion blocked on the corpus → Issue 857); calibration
staleness + all four seams (`2057ac3b`, `9badceb2`, `4af13a60`, `4483b84e`).
**Remaining OPEN leads stay at their provenance**: CQ-W2A8 GEMV, CLAWS
activation-sparse decode, HiDRA-v2, TurboQuant-H riir-rag lane, functional_embed
(PETE), SAN league cell, engram-delta fine-tune, SAN npc_brain v2 → Research
568/570/571 rows (re-file one issue per lead when picked up); the
`VocabChannelPruner` hook follow-up rides the `legal_token_set` module docs; the
sealed long-context corpus is Issue 857's pull-gated chain. Deferred rows'
reasons in git history.

## Issue 852 — CLOSED as PARKED: the τ(t) P_e-LUT POC waits on a D2F model consumer (2026-09-19)

**Status: CLOSED (PARKED) 2026-09-19, owner call — filed and parked same day.**
Off Research 572 row 1. The modelless half (`ScheduleKind::DecodingErrorLut` LUT
+ monotone-regression builder + τ↔t inversion) is landable, but GOAT G2 needs a
**measured P_e curve at matched step budget** and no in-repo D2F model harness
exists — landing now is the synthetic-fixture GOAT pass Issue 780 refused.
**Reopen triggers**: (1) a D2F lane with a real model artifact wants a
data-adaptive schedule; (2) riir-train fires the RecFM |V|≥32k invert-CDF
t-sampling twin (its Issue 563). Lead: Research 572 rows + git history
(`852_fmlm_tau_lut_schedule.md`).

## Issue 853 — CLOSED as PARKED: the autoguidance POC waits on its measurement arms (2026-09-19)

**Status: CLOSED (PARKED) 2026-09-19, owner call — filed and parked same day.**
Off Research 572 row 2. Arm A needs a DDTree acceptance harness over a served
model; Arm B is riir-train Issue 563's free-rider. The bare kernel (η=1
identity, zero-alloc) is trivial but has no consumer and no measured claim.
**Reopen triggers**: (1) riir-train 563's Arm B fires (~0 training GPU-hours);
(2) a DDTree acceptance-rate bench lands in riir-ai's serving lane. Prior art
in Research 572 (logit-space arm = contrastive-decoding ADOPTION, Li et al.
2022; dual-dropout latent arm = the fusion candidate) + git history
(`853_autoguidance_residual_extrapolation.md`).

## Issue 854 — CLOSED: the wedge is diagnosed, the repair scoped, and T3 stays trigger-gated by design (2026-09-19)

**Status: CLOSED 2026-09-19.** Landed (`dded36d8`, `65bf2944`, `d8b55a34`,
`c83f0f84`): T1 (SLOW-vs-WEDGED recipe; verdict line says whether the tree held
still), T2 (census: exposure is in mutation runs, not the 131 git call sites —
no gate/convention/sweep; census preserved), T4 (NO in-process wall bound — it
would share the watchdog's defect; the external per-module timeout is
practice), T5, T6. **T3 (non-interactive git env) stays UNLANDED by its own
rule** ("relevance unmeasured until T1 names the call"); no stall seen since.
**Reopen trigger**: a stall on a NON-mutation run voids T2 and T3 lands; a stall
in a mutation run uses the T1 recipe (tail the log, kill the git child —
measured to resume).

## Issue 855 — CLOSED: every task resolved; the guard gate + ratchet sweep both shipped (2026-09-19)

**Status: CLOSED 2026-09-19 — every task closed.** T1 filed (`dd502fe6`); T2
repaired 5/5 arms (`black_box` both ends; bars unchanged); T3 ran all 34
asserting timed regions at n ≥ 1000 — **7 VANISHED, 27 SURVIVED**, all 7
repaired, bars/pass counts unchanged; T4: the static `let _ =` detector
REFUTED by execution (20.0% vs 21.1% base rate), `timed_region_guard_gate.py`
shipped as a docs-gate CHECK gating the DEFENCE (`196a4cda`); T5 ran all 33
sibling rows — 2 VANISHED = 6.3% vs this repo's 20.6%, repaired at riir-ai
`3712d51b6`, unbuildable one filed as riir-chain `.issues/157` (`c7e84919`,
`88e5d37f`); T6 `timed_region_drift_sweep.py` — 21 repos, ratchet EARNED by
execution (`72ee0f5b`). AGENTS.md's `timed_region_guard_gate.py` row stands.

## Issue 857 — CLOSED as PULL-GATED: the corpus builds when riir-ai 882's consumer fires, not before (2026-09-19)

**Status: CLOSED (PULL-GATED) 2026-09-19, owner call.** Boundary answered
(`51291548`): corpus + harness = riir-train (`kimi_k3_long_context` packer),
canary wiring = riir-ai `.issues/882` T2, scoring instrument = here (shipped).
**riir-ai 882's T-Pull-1 is the pull-gate, owned there**; a second copy here
would be the drifting duplicate Issue 857 warned against. The lossless-window
escape hatch (`window ≥ d_max` ⇒ bit-identity) is in git history and Bench
841. **Fire condition**: 882 T-Pull-1 lands a bounded decode KV budget → file
the riir-train corpus issue citing it, size to THAT window (never
`kv_sink_window`'s default 260), pin `r_max`/`p_cap_max` vs a no-policy
baseline, run `UNBOUNDED` as the lr=0 control, re-gate `kv_sink_window` and
promote or refuse.

## Issue 843 — CLOSED: the plasma_path ternary dense matvec loses below L3 on every served shape; T4 resolved as per-shape dispatch (2026-09-19)

**Status: CLOSED 2026-09-19, T1–T4 all resolved (T4 by owner call, gate D1).**

**Finding.** Default-on `plasma_path` ternary matvec is **1.9–3.0× slower** than
f32 `simd_matvec` whenever the f32 operand fits L3 — every served shape
(768×3072 2.10× NEON; 1024² 1.94–2.26× AVX2). x86_64 crosses at L3 (§T2: m=4096,
64 MiB, 13700K/30 MB L3); NEON narrows 3.04→1.83, never crossing to 1 GiB (§T1).
T3: sign extraction outnumbers arithmetic ~8:1 (NEON ~8.5 GMAC/s; AVX2
latency-bound, `vcvtdq2ps` suspect).

**T4 owner call (gate D1): per-shape dispatch** — f32 below L3, ternary above,
as `simd_matvec_plasma_dispatch` (+ cached `l3_cache_bytes()`:
`KATGPT_PLASMA_L3_BYTES` override → macOS sysctl → Linux sysfs → 32 MiB
fallback biased to f32) in `katgpt-types::simd::plasma_dispatch`, for callers
holding BOTH representations. Option (b) (pre-expanded i8 sign rows, ~1.7–2× at
5.3× footprint) NOT taken. GOAT G1+G2 GREEN
([Bench 843-dispatch](.benchmarks/843_plasma_dispatch_goat.md)): bit-identical
across a forced boundary; 2.11× vs ternary at 1024² (bar 1.5×); −1.6% vs dense
(≤10%). `plasma_path` STAYS DEFAULT-ON (21× footprint); five manifests state the
trade. Downstream wiring (riir-ai npc_brain) is consumer opt-in.

Size-sweep instrument (`tests/bench_843_ternary_size_sweep.rs`) preserved;
issue text in git history (`.issues/843_*` removed).

## Issue 831 — CLOSED: bench_171 P3 reclassified as instrument-health + mechanism by owner call (2026-09-19)

**Status: CLOSED 2026-09-19, T1–T5 all resolved.** The coin flip was repaired
2026-09-18 (T2/T3 `03d729fdf`: the screener's work was deleted by `let _ = acc`
— the Issue-723 elimination shape; with one `black_box` 64.2% release / 63.1%
debug x86_64; T1: aarch64 63.1–63.8% over 20+20 runs).

**T5 owner call (gate D2): P3 is instrument-health, not a perf bar —
`thinking_prune`'s status does not rest on it.**
`tests/bench_171_thinking_prune_goat.rs` P3 (renamed
`proof_p3_instrument_and_mechanism`) gates two load-immune claims and PRINTS
wall-clock as diagnostic (63.5% at reclassification):

- **Instrument health** — `a_ns_per_iter() > 0`, `b_ns_per_iter() > 0`,
  median finite.
- **Mechanism as exact CALL COUNT** — FrozenBaseGuard `should_screen_full` is
  `hop >= total_hops - 1`, so over 3 hops exactly 1/3 of Uniform's screener
  calls (13,824 vs 41,472). A regression breaks it to 1:1.

The ≥30% latency assert is REMOVED, not lowered. Issue file removed.

## Issue 847 — CLOSED: `simd_lut_dequant`'s AVX2 kernels compiled to NOTHING on every ordinary x86_64 build — and the bf16 sweep that followed was right for one kernel of three (2026-09-19)

**Status: CLOSED 2026-09-19, all seven tasks (T1, T2, T2a, T3, T4, T5, T6).**
Filed as 846, renumbered to 847 — `dual_allocation_gate` caught it at
allocation time (Issue 791 T2's protocol reached it only at merge).

- **T1/T2a — class.** `#[cfg(all(target_arch = "x86_64", target_feature =
  "avx2"))]` on a SHIPPED path compiles to **nothing** by default, so a
  **default-on** feature ran scalar. Repaired with runtime `simd_level()`:
  **1.7x** `dequant_via_lut`, **4.4–5.6x** `dequant_dot_via_lut`.
- **T2 — reflex repair right for ONE of three.** RNE narrowing **2.4–2.5x**,
  probed in; trunc narrowing a **REGRESSION** (intrinsics slower than scalar);
  widen reported undecidable.
- **T3 — WALLED.** `scripts/shipped_target_feature_gate.py` (docs-gate CHECK);
  166/166 `target_feature` attrs in `src/` are here — no sweep. Three exclusions
  counted; the probe body PINNED.
- **T4 — aarch64 (M3).** RNE NEON wins 1.23x; widen and trunc NEON lose to
  LLVM autovectorisation. Bit-identity asserted per family.
- **T5 — trunc AVX2 DELETED; transcription REFUTED.** `_autovec` (scalar body
  + `#[target_feature(enable = "avx2")]`) 1556–1591 both builds; post-deletion
  `+avx2` dispatcher 1594 vs 1591: **LLVM's default vectorisation of
  `bits >> 16` beats its AVX2 one ~1.3–1.4x.** (Issue 844 T4 per-ISA rule.)
- **T6 — widen is BIMODAL on BUFFER ALIGNMENT.** Quiet box: scalar stable
  **<1%** (1692–1704 ns, n=32768) while AVX2 arms split ~98 / ~170 at n=4096.
  Axis `src.as_ptr() % 32` (`t6_widen_bimodality_vs_buffer_alignment`): one fast
  residue, 1.6–2.2x slower elsewhere; `vec![]` alignment is a coin flip.
  Controlled: **2.2x at n=256**, **1.57x at n=4096**, ties elsewhere → probed
  in; n=32768 memory-bound, STATED. ⚠ Comparing VALUES failed on `NaN != NaN`.

⛔ **Through-line: three wrong answers from sound-looking measurements** (T2
regression read as undecidable; T5 blamed an innocent transcription; T6 blamed
a busy box for deterministic alignment) — the instrument had to change before
the kernel could be judged. Record: git history `.issues/847_*`.

## Issue 850 — CLOSED: the UNVERIFIED-upstream guard challenged only ONE of `behind_origin`'s two silent readings — and then PRINTED a false statement about the one it gained (2026-09-19)

**Status: CLOSED 2026-09-19, all four tasks.** Filed as 849, renumbered to 850
— a collision `dual_allocation_gate` could NOT see, hence T4.

- **T1 — gap.** Issue 827 T5's guard covered `(0, 0)` but not the STRONGER
  silent `(n, 0)`; repair `elif beh is not None`. Cost:
  `shared_temp_path_drift_sweep` gave 6 findings in 3 repos (22, 9, 3 behind),
  **4 already FIXED upstream**; six sibling "repairs" committed pre-fetch, all
  reset. ⚠ Causation unclaimed (fetch destroyed the state).
- **T4 — COUNTER axis.** A number allocated and CLOSED in one commit leaves no
  document, so `--diff-filter=A` sees nothing; both sides bumping `.highwater`
  past the merge base is the missing document. Armed both ways (injected git
  reader per push; real two-repo fixture under `--prove-fires`).
- **T2 — sweep FETCH? NO, measured:** **250.2s serial / 50.2s 8-way** cold,
  21.5s warm vs sweeps 0.04–40s and a 32-check docs gate ~164s — **5–30x**,
  ~19×/family run; it also writes others' refs (Issue 797's class). ⛔ The remedy
  named CONTRACT spellings, nonexistent on an aliased box (`mmorpg-editor` vs
  `seal-game-editor`). Landed `scripts/fetch_contract_repos.py` (origin NAMED,
  arms assert HEAD/worktree untouched).
- **T3 — guard PRINTED a false statement**, observed by CONSTRUCTION (stubbed
  clock, REAL sweep; repos were 0.2–9.7h fresh): the FINAL line said *"report
  'up to date'"* — FALSE for `(n, 0)`. `unverified` now carries `(age,
  commits_behind)` per repo. ⚠ Old arms asserted PRESENCE
  (`"UNVERIFIED UPSTREAM" in ln`), not CONTENT. `worktree_state` selftest
  **148 → 157**, 7 of 9 new arms red on the old renderer. Record: git history
  `.issues/850_*`.

## Issue 832 — CLOSED: the last fixed shared-temp site is repaired — the non-demo backlog is zero, the ratchet is the wall (2026-09-19)

**Status: CLOSED 2026-09-19 — T5 complete. Final site, seal-remake `crates/seal-view/tests/quest_sim_front.rs` pid-suffixed at seal-remake `db68f5e`, validated (`cargo test -p seal-view --features quest_sim,sync_client --test quest_sim_front` — 1/1). Re-pinned `mmorpg-remake 16 1 → 18 0`. ADJUDICATED: seal-game-editor `services.rs` scratch root `max_fixed 1` (`6964117e`); riir-clippy's 9 roots `[documented-fixed-root]` + mtime fix (`83f4cd89`). Test-class sites repaired with sibling SHAs (riir-ai `9d138531b`, riir-chain `8a3b0f5`, riir-game-sdk `ef66117`, riir-clippy `f5ada0ec`, riir-deployer `b551028`, mmorpg-remake `db68f5e`, mmorpg-editor + mmorpg-remaster same-day 09-19); demo class (`examples/`+`src/bin/`) adjudicated. x86_64 matrix fully green (11,177 assertions, `a77c46c0`). Record: git history `.issues/832_*`.**

## Issue 844 — CLOSED: the dot-delegation crossover is length 24, per-ISA, measured on BOTH arches — not a backlog (2026-09-19; full record at the 09-19 dated entry below, file removed this commit)

## Issue 840 — RESOLVED: two `.research/569` documents landed forty minutes apart and tripped two detectors — the rewrite set was EMPTY, and the attribution lessons became the staged-set rules (2026-09-19)

**Status: RESOLVED 2026-09-18 — T1/T2/T3 landed by `katgpt-rs-fa` after both live peers confirmed neither document was theirs; `numbering_gate.py` passed and both rows cleared together (T3's assertion), no pin touched. ⛔ T2's premise was wrong in the safe direction: hand-derived rewrite set 3 sites, measured set EMPTY. Durable output: the attribution discipline in AGENTS.md §staged-set (shared authorship, one shared reflog, elimination over an incomplete roster, quote-the-`from=`-pipe — `4e82cc489` + `c0995ea4b`). No code change beyond the peers' renumbering. Record: git history `.issues/840_*`.**

## Issue 836 — CLOSED: a `git worktree` made the entire head-provenance mechanism silently inert — `.git` answers TWO questions and each spelling was wrong for the other (2026-09-19)

**Status: CLOSED 2026-09-18 — T1 `f6749af5c` (every `worktree_state` guard used `.is_dir()`, so in a worktree `sweep_advisory` returned `[]` — Issue 797's defect inside its prevention), T2+T4 `c12e96415` (three private copies — `console_encoding_gate.tracked_scripts`, `sweep_advisory_membership_gate.tracked_sweeps`, `numbering_drift_sweep.head_listing` — measured then DELEGATED to `worktree_state.is_checkout`; the silent `.name`-mismatch branch made loud), T3 `52f3de398` (cross-repo: 3 sites in 2 siblings, 1 defect, repaired at riir-ai `9637d09ea`). The `.git` two-questions table (canonical REPO → `.is_dir()`; run-git-HERE → `.exists()`) loaded into context via `a9bf8c785`. Record: git history `.issues/836_*`.**

## Issue 845 — CLOSED: `channel_aware`'s duplicated dot kernel deleted; T5 measured NEON PARITY — the x86_64 1.8–7.8× penalty does not transfer to aarch64 (2026-09-19)

**Status: RESOLVED (repair `32056164e`, filed-and-fixed together; T5 measured 2026-09-19). The crate's `simd_dot_f32` was a ~200-line duplicate of `katgpt_types::simd::simd_dot_f32` (already a dep; seven sibling files used the shipped one), its AVX2 arm gated `#[cfg(target_feature = "avx2")]` — off by default, so a 4-accumulator scalar loop ran: 2.72–7.76× slower default, 1.83–4.78× under `-C target-feature=+avx2`. Repaired by delegation: wrapper kept, three private kernels deleted, `unsafe` surface ZERO, bench converted to a bit-identity gate whose canary (planted kernel, max |Δ| 4.77e-7 — inside every old tolerance) proves only bit equality fires.**

**T5 (aarch64/NEON, DRY half only): PARITY — three interleaved A/B runs (`ab_median_ratio`, 11 rounds, release, M3 Max; load 6.8–10.2, 952%-CPU sibling in run A, 85% memory free, battery) old/shipped 0.98–1.04× at 32–256, ~0.95× at 1024 (fits the Issue-700 reslice + explicit-len); agreement ≤ 1.5e-6. "Shipped is faster" is x86_64-only. Temp bench deleted (Issue-843 precedent).**

## Issue 851 — CLOSED: `muon_update` step scale repaired to the canonical √max family (update RMS ≈ 1.0) (2026-09-19)

**Status: RESOLVED (session `research-2502.16982-cont`; fix `1ebe39f61` (post-rebase; cae72cf0f pre-rebase)). The default-on Muon wrapper scaled by `1/max(rows, cols)` — D× below canonical at D×D, d-dependent (riir-train Bench 492's un-absorbable-by-LR class) — while its comment claimed "standard Muon scaling" and its doc carried a third formula (`1.0/rows`). Zero production consumers (latent).**

- **Wrong prediction:** assertions were NOT scale-invariant (Gram-vs-identity err ≈ 0.98 under a 1.0 bar — vacuous); now normalized by mean diagonal (0.2503 @ 8×8; threshold 0.35; NS5 band 0.21–0.44).
- **Regression pin:** update RMS ∈ [0.68, 1.12] (NS5 band, Bench 050) at 64×64 / 128×64 / 64×128 — measured 0.9667.
- Scale: `√max(rows, cols)` (riir-train `muon_step_dense`, RMS 1.0); Keller `√max(1,A/B)` and Moonlight `0.2·√max(A,B)` (arXiv:2502.16982 Eq 7) differ by LR-absorbable constants (Bench 492: 13% at D=32). Doc names all three.
- Validation: GOAT bench_152 25/25 PASS; muon tests 3/3; clippy `-D warnings` clean; rustfmt clean; Bench 050 perf unaffected.
- Provenance: arXiv:2502.16982 (Moonlight) distill (NS5 core = Plan 152; weight decay + √max shipped and superseded in riir-train; the Muon-SFT-no-advantage negative in the removed issue's git history).

## Issue 842 — CLOSED: the alias-mapped sweeps opened a directory that does not exist, and the first real read surfaced a workspace of hidden findings (2026-09-19)

**Status: RESOLVED (T1–T4, session `katgpt-rs-c5`, 837/846 lane). `derive_repos` returns CONTRACT names and 17 of 19 drift sweeps opened `WORKSPACE / <contract-name>` — nonexistent on a box whose `repo_alias.local.txt` maps the three seal repos to mmorpg-* spellings, so every walk returned 0. Seven sweeps red walk floors: TRUE pins against the WRONG DIRECTORY. `docs_drift_sweep` / `numbering_drift_sweep` imported the alias but built `ws / contract-name` too; `worktree_state.sweep_advisory` silently `continue`d past them.**

- **T1 — one seam.** `sweep_population.open_repo(name, workspace)` for sweeps + `repo_alias.real(repo)` = `repo.parent / disk(repo.name)` for audits — identity when unmapped (FIXTURE-SAFE). `sweep_advisory` resolves via `disk()`, prints via `display()` (alias content never on stdout). Box-independent arms (injected `_loaded`).
- **T2 — labeling law: resolve the path you READ FROM, keep the name you LABEL WITH** — docs (`repos_real`, `display()` in `run_auditor`), citation (`audit()`/`crate_map`/`unreliable_oracles`), lda (`Kernel(repo.name, …)`), `percentile_index_audit.main`, `restatement` (`apply()`).
- **T3 — first real read.** 19/19 non-citation sweeps GREEN. Hidden findings adjudicated: seal-game-editor 3 console-encoding + 2 locale-io `open()` + 1 shared-temp + 4 unreachable instruments; seal-remake 3 shared-temp + 1 ungated `FrameRateCap` (FILED seal-remake `.issues/032`, concurrent WIP) + 2 silent-now targets + 7 plan-scoped scripts; seal-online-remaster 1 shared-temp. Repairs: seal-game-editor `6964117e`, seal-remake `df81497`, seal-online-remaster `5fb3122`. ⚠ The seal-remake commit also swept in a concurrent session's staged refactor + two deletions via an index race — disclosed in its message.
- **T4 — pins.** Ten files re-pinned with reasons: seal-online-remaster shrinkage (seal-core/seal-edge-worker moved out: orphaned_attr 335→248, platform_dead_code 360/7400→248/2251, percentile 335→248, wasm32 2/2→1/1, cfg_gated 1→0); seal-remake growth (platform_dead_code 33/270→441/10779); control renamed mmorpg-poc-submodule → seal-poc-submodule; shared_temp_path (`TmpDirStore` DELIBERATE max_fixed 1); numbering/instrument_reachability/len_derived first measured.
- **T5 — citation sweep stayed red: true positives.** ~35 citation defects across 6 repos (seal-remake CROSS 15, riir-game-sdk 7, riir-dapps 4, riir-clippy 3, riir-dao 3) + riir-ai IN-LOCAL-RANGE 8 > 6 (a RECLASSIFICATION, re-pinned 6→8). Filed as Issue 846.

## Issue 846 — CLOSED: the alias seam has a PROSE half — 44 true CROSS rows, 37 cleared by teaching the qualifier the on-disk spellings, 7 by prose (2026-09-19)

**Status: RESOLVED same day (session `katgpt-rs-c5`). The filed table undercounted (5 repos, 10 red): the live run found 44 CROSS rows — mmorpg-remake 15, riir-dapps 4, riir-clippy 3, riir-dao 3, riir-game-sdk 7, riir-kat 4, riir-shader 5, riir-mmorpg-examples 1, riir-neuron-db 1 (+ IN-LOCAL-RANGE 3 > 2), riir-viewbridge 1. Read the live sweep, never a filed table.**

- **Nearly every row was the ON-DISK spelling**, not a pre-migration qualifier: `mmorpg-editor` / `mmorpg-remake` / `mmorpg-remaster` exist only in `repo_set.txt`; on both boxes the dirs are `seal-game-editor` / `seal-remake` / `seal-online-remaster`. `seal-remake Plan 011` is CORRECT. Rewriting to spellings no box has would worsen the docs — repair went the other way.
- **Classifier half (37 rows).** `issue_citation_gate.spelling_aliases()` — in code, NOT the gitignored `repo_alias.local.txt` (verdicts must not go machine-local) — used by `qualifiers()` in the 40-char lead + 3-line window (so "(Plan 192 T4.1, seal-game-editor, …)" qualifies). LENIENCY ONLY (`written_names` contract-only). ⚑ `\bseal-remake\b` matched inside `seal-remake-unity` — spellings use the `_NAME` boundary regex.
- **Prose half (7 rows):** riir-dao `160f9a1` (Proposal 007 → riir-clippy; Proposal 006 B6 + Plan 035 → riir-dapps), riir-kat `51994ee` (Bench 053 ×2 → riir-clippy; Proposal 006 Q1 → riir-dapps — ⚠ local `25006d1` was a TWIN of an Issue-837 commit on origin, auto-skipped in rebase), riir-neuron-db `3f6cf67` (Proposal 032 + Issue 036 → riir-game-sdk; Issue 168 → seal-game-editor; Issue 037 → riir-clippy).
- **Landing:** sweep rc=0 — 0 CROSS, 23 IN-LOCAL-RANGE (riir-ai 8→5, riir-neuron-db 3→0), 0 ORPHAN, 2 ORACLE-STALE (katgpt-rs 18 / riir-ai 9 behind). No re-pin. AGENTS.md §Docs gate note added.

## Issue 838 — CLOSED-as-decided: a shell script spawning a NATIVE child is a third encoding seam, and the population is measured at zero live instances, so it is deliberately NOT gated (2026-09-18)

**Status: MEASURED and CLOSED as a decision — no instrument (`c9afca9dd`; found by session `katgpt-rs-54` when a `·` in a PowerShell format string in `scripts/x86_64_execution_matrix.sh`'s box-state block came back mangled via cp874; repaired by keeping the child ASCII and letting bash own separators).** A `.sh` handing non-ASCII to `powershell.exe`/`wmic`/`cmd.exe` falls between `console_encoding_gate` and `subprocess_encoding_gate`.

Census, 197 tracked `*.sh` in 16 repos: **43 native-child lines, 2 with non-ASCII — both comments** (`riir-ai/scripts/perf_rematch.sh:509`, `riir-train/scripts/c13_auto_gate.sh:243`). Live: **0**. A gate would govern an empty population and need masking machinery just to suppress its two rows — cries-wolf by construction.

Like `check_validation_gate` T4 (789), unlike `console_encoding_gate` (wrong by seven repos): **measure the population, then decide**; keep native children ASCII.

Adjacent gap (not closed there): the matrix picked the memory source on whether `/proc/meminfo` EXISTED, not ANSWERED — MSYS has one without `MemAvailable`/`CommitLimit`; repaired by selecting on answers and printing which did (Issue-835 `.git`-FILE shape).

## Issue 837 — CLOSED: registering a contract repo reds the whole sweep family, and the ONE file a gate checks is not the twenty-one that make them red (2026-09-18)

**Status: RESOLVED same day (T1–T3). Found by session `katgpt-rs-c5` (`✗ riir-llm … UNPINNED` in an Issue-836 run). ⚠ The issue file claimed T2 landed when nothing existed; T2 built afterwards, and T1 completed — four more repos (`katgpt-web`, `riir-dao`, `riir-deployer`, `riir-esp32`) owed rows in 8 floors files (Issue-798 class).**

`riir-llm` registered (`b5dd81dc`) with only `repo_set.txt` + AGENTS.md §Repo count; 19 of 21 sweeps red `UNPINNED`, hiding (riir-kat 3 unqualified citations, fixed riir-kat `51994ee`; riir-shader `gamefx_feature_matrix.sh` EXPOSED trap window, fixed riir-shader `176da06`; riir-clippy 7 plan-scoped scripts, ratcheted at measured; riir-shader `.plans/.highwater` stale at 004, fixed riir-shader `6f045d4`). Registration is a 22-file operation with one file gated — eleventh instance of the rule-in-one-instrument shape (777, 778, 782, 783, 789, 793, 797, 820, 822, 836).

- **T1 — pin rows** from each sweep's own printed row (`2d348a470`; cloning riir-auth's was wrong eleven times — *a copied pin is a diary*). Arity check fixed two 4-vs-6/7-column headers against `FIELDS`.
- **T2 — `scripts/repo_registration_gate.py`, docs-gate CHECK 29:** every `*_drift_floors.txt` has a row per on-disk canonical repo (registry ∩ `derive_repos`). Subset files (`docs_drift_floors`, `restatement_drift_floors`) declared in `scripts/repo_registration_scope.txt`, reds both ways (STALE-SCOPE, UNKNOWN-SCOPE, INCOMPLETE). `TOTALS` excluded by shape regex. Arms + `--prove-fires`.
- **T3 — inverse MEASURED at zero**: only repo-shaped outsiders were six PACKAGE names in `x86_64_matrix_floors.txt` (heuristic false positive).

Standing census lives in the gate's PASS line, never prose. It does not claim a row's VALUE is correct.

## Issue 825 — CLOSED POSITIVE, after a same-day RETRACTION of its own negative close. Coulomb crowd redistribution ships as `coulomb_flow`; the "negative result" was a bench walker with two defects (2026-09-18)

**Status: RESOLVED. `coulomb_flow` ships opt-in ([Bench 825](.benchmarks/825_coulomb_crowd_redistribution_goat.md), G1–G4 PASS). [Bench 815](.benchmarks/815_coulomb_redistribution_poc.md) is the independent second implementation and now agrees.**

⛔ **Replaces a record reading `CLOSED NEGATIVE — the solve transfers, the
first-arrival readout does not`.** Two sessions implemented this concurrently
and landed opposite verdicts four hours apart: one shipped `coulomb_flow` all
green; the other measured endpoint MAE 0.1190 vs a ≤ 0.01 bar, filed the
negative, removed the issue file and wrote the negative here.

**Settled by running the LOSER's fixtures** (`grid_2d(4,3)/(8,6)/(12,9)`, sinks
0.6/0.4) through the WINNER's readout: MAE `0.0 / 1.2e-7 / 3.0e-8` vs `0.119 /
0.078 / 0.037`. Disagreement is a cheaper oracle than self-consistency.

### The two defects, both in `bench_815`'s `absorb()`

1. **A sink absorbed 100% of what reached it.** Correct stop probability is
   `μ₁(v) / (inflow(v) + μ₀(v))`, 1 only with no outflow; via
   `CrowdRouter::consistent_absorption`, sink `n−2` has `out_flow = 0.209` and
   absorption **0.657**. Adjacent sinks stranded mass: `[0.4810, 0.5190]` vs
   `[0.6, 0.4]`.
2. **Order-dependent proportional split.** `share = packet[v] * (p / total)`
   read `packet[v]` while the loop decremented it — conserved mass (invisible
   to a conservation gate), wrong proportions (`p₁ / (p₁ + p₂²)`), invisible at
   degree-1 vertices.

### The lesson is about the INFERENCE, not the arithmetic

`0.119 → 0.078 → 0.037` was read as *"discretization error in the readout"* —
the exact mechanism the corrected readout proves exact. **A converging error
fits many mechanisms; distrust closing on one** — derive what the absorption
must be and check the instrument against it.

⚠ The negative removed the issue file, wrote "the paper's 20× does not transfer
to zone graphs" here, and set three re-open conditions — all answers to a
defect. It began after `katgpt-dec::coulomb` was on `develop`: the
substrate-first gate's target.

### What shipped

`coulomb_flow` (opt-in, katgpt-core → katgpt-dec): `CoulombFlowField` solves
`δ(dφ) = μ₁ − μ₀` on a zone graph, returning `j = dφ` + per-vertex
`CrowdRouter`. Conservation is the equation; arrival is `μ₁` exactly, so per-NPC
error is sampling error and Bench 825 G2 gates the `1/√N` DECAY (21.7× over
1e3 → 1e6). G1 conservation 5.96e-8, G3 naive ratio 1312× vs 10× bar, G4 zero
allocations over 100 solves. Consumer wiring is riir-ai's; flag stays opt-in.

Bench 815 repaired, not deleted: its f64 pinned-Gaussian solver and mass-packet
walker are independent of Bench 825's CG solve and sampled walk. They SHARE
only the absorption rule — imported from `CrowdRouter::consistent_absorption`.

⚠ Open: scale beyond 108 zones, irregular zone graphs, the `to_flow_vectors`
bridge.

**En-route gain:** katgpt-dec root re-exports the zero-alloc `_into` family
(`exterior_derivative_into`, `codifferential_into`, `graph_laplacian_into`,
`hodge_laplacian_into`), previously unreachable from the root.

## Issue 830 — CLOSED: Issue 829's anchor class one seam deeper — the locale-I/O classifier was anchored to a CALL-NAME SET, and it had repaired one side of a round trip in this repo's own instrument (2026-09-18)

**Status: RESOLVED same day (T1–T5).**
**Severity: a tracked instrument WRITES a fixture with the system locale and
READS it back as UTF-8 — Issue 829's defect, surviving Issue 829's repair.**
**Origin: read Issue 829's residual instead of trusting its count.**

### The symptom

Issue 829's **273 → 2** counted **three call names**. `cfg_row_implication_audit.py`
writes a fixture via `tempfile.NamedTemporaryFile("w", …)` (line 407, locale)
and reads it with `read_text(encoding="utf-8", …)` (line 192, repaired by 829)
— passing only because bodies are ASCII.

### The finding

`locale_io_fix.sites()` matched `PATH_METHODS = {"write_text", "read_text"}`
plus builtin `open`. The class is **a text-mode file object whose encoding
defaults to the locale**; also in it:

| form | why it is in the class |
|---|---|
| `p.open("w")` | `Path.open` / `io.open`, text mode |
| `tempfile.NamedTemporaryFile("w", …)` | explicit text mode (default `"w+b"` is safe) |
| `os.fdopen(fd, "w")` | text mode; default `"r"` is text too |
| `io.TextIOWrapper(fh)` | always text |
| `codecs.open(p, "w")` | encoding is positional |

Fourth anchoring (Issue 823 POSITION, Issue 828 DELIMITER SET, Issue 787
DOCUMENT, now a NAME SET); probe by enumerating the class's *mechanism*.

### What was measured (T1, before anything was claimed)

16 repos: `.open(<text mode>)` 7 (riir-train 6, riir-clippy 1);
`NamedTemporaryFile` TEXT 2 and `os.fdopen` TEXT 2 (katgpt-rs); `io.TextIOWrapper`
/ `codecs.open` 0 — **11 total**. The `os.fdopen` rows in
`trap_exit_launder_audit.py` write **shell scripts `bash` executes**.

### T2 — the discriminator, and its blind spot is MEASURED rather than argued

`.open` can't be duck-typed; discriminator = **literal text-mode argument**
(non-empty `str` over `rwxat+U`), no denylist:

| excluded | by what |
|---|---|
| `os.open(devnull, os.O_WRONLY)` | mode not a string literal |
| `tarfile.open(arc)` | no mode argument |
| `tarfile.open(path, "r:gz")` | `:` not a mode char |
| `Image.open(hero_path)` | no mode argument |
| `f.open("rb")` | `b` |

⚠ **Cost: no-mode `p.open()`** (text `"r"`) is unseen — counted **5 sites, all
5 the table above** (not-in-class). A receiver denylist would fail OPEN against
a wall at 0; the blind spot is printed.

### T3 — the repair

`locale_io_fix.sites()` gains the forms behind `_text_mode()` — **one
classifier** for gate, sweep, repair (Issue 755); `repair()` unchanged; 12 new
arms in `locale_io_fix.selftest()`.

### T4 — the eleven sites

This repo's four in the landing commit; siblings cited per the Issue-798 rule:

| repo | sites | commit |
|---|---|---|
| riir-train | 6 | `15db2c67` |
| riir-clippy | 1 | `54b999de` |

### T5 — the floors move, in the direction that proves the walk grew

`FLOOR_IO_CALLS` 450 → re-pinned on the measured count; per-repo sweep rows
likewise.

### T6 — the sibling seam does NOT carry the same anchor (a measured negative)

PIPE seam (Issue 778) probed first: `universal_newlines=True` already in
`subprocess_encoding_gate.scan_text`; `os.popen()` in class but **0 sites over
16 repos**, no `encoding=` to insert — not added. Re-measure before
generalising.

### What this issue does NOT claim

- FILE seam closed: `codecs.open` / `io.TextIOWrapper` are walls at **0
  sites**; no-mode `p.open()` a **stated** blind spot (cost 0).
- Name-based classifiers wrong: `read_text` is Path-specific; `open` is not.


## Issue 828 — CLOSED: Issue 823's anchor class, third position — the heading oracle was anchored to a DELIMITER SET, and the biggest unread family is this repo's own house style (2026-09-18)

**Status: RESOLVED same day (T1–T4). T4 answers Issue 823 T5 by PRICING it —
blast radius 2 rows workspace-wide, so the unsound widening is also not worth
doing.**

### The symptom

Of Issue 823 T5's ~213 unread records, **60 are not the `resolved` family**.

### The finding

The rule *nothing between the number and its delimiter* was also anchored to a
**delimiter SET** (`(` leading, `[:,]` dated) lacking the **em dash**.

Measured (16 repos, pinned documents, foreign-filtered, unread only):

| first char after the number | n | verdict |
|---|---|---|
| `—` (leading form) | **56** | delimiter — **sound to read** |
| `(` (dated form) | **4** | leading form accepts it, dated did not — **sound to read** |
| `resolved` / `RESOLVED` | 64 | interstitial — Issue 781's family |
| `closed` / `close-out` / `CLOSED` | 28 | interstitial |
| `T1` … `T8`, `Arm`, `wave`, `phase`, `complete` | ~45 | interstitial |
| `follow-up`, `filed` | 3 | interstitial — the **pinned negative** |
| other | ~13 | interstitial |

The 56 are `## Issue 788 — …: CLOSED (2026-09-14)` — **katgpt-rs's own house
style**. ⛔ **Not the unsound widening**: that is dropping the discriminator;
this adds a delimiter and keeps it (`## Issue 043 follow-up — title` still
rejected, pinned in both delimiters).

### T1 — the delimiter set

- `_SELF_HEADING_DASH`: same discriminator, either position, delimiter
  `—`/`–` or ASCII `-` **space-separated both sides**. ⛔ Bare hyphen rejected:
  riir-ai's live `## Issue 366-class (pos-uniform chunk forward) FIXED in
  riir-gpu`; arm pins it.
- `_SELF_HEADING_DATED` gains `(` — four riir-chain records (`## 2026-09-16 —
  Plan 062 (Proposal 010 D6/T1.2):`) had been rejected by an asymmetry.
- New pattern's group(3) = whole remainder, so the foreign filter is more
  likely to REJECT (safe for a suppressing path). ⛔ `_SELF_HEADING`'s scope
  left alone — widening it would reject `## Issue 059 (2026-01-01) — <sibling>
  did X`, the suppression Issue 754 landed.

### T2 — the arms

`citation_drift_sweep.selftest()` arm 2's fixture gains the delimiter axis:
`## Issue 047 — title` reads; `## Issue 048 follow-up — title` does **not**
(the key arm); `## Issue 049-class — title` does **not**; `## 2026-01-01 —
Issue 050 (parenthetical): title` reads. The style-blind meter's expected pair
moves with the fixture.

### T3 — the numbers move in the SUPPRESSING direction only

`heading_allocated()` only suppresses; every ratchet in
`citation_drift_floors.txt` greens or holds; **not** re-pinned downward.

### T4 — ANSWERED by pricing it: the blast radius is TWO rows

The semantic answer stays NO (`resolved`, `closed`, `T3`, `Arm C`, `follow-up`
share a SHAPE; arm 2 pins `follow-up` negative in every position/delimiter).
But `heading_allocated` is one member of a union whose others answer from a
FILE, so **only the residue can change a verdict**. Over 16 repos: of **153
unread records, 2** carry a number no other oracle knows — riir-ai's `## Issue
969 resolved — …` and riir-clippy's `## Issue 097 resolved — …`, both the
Issue-754 never-committed shape. **98.7% redundant**; the count does not
justify the rule. ⛔ The two rows are real and land as IN-LOCAL-RANGE
(UNDECIDED, never a pass).

⚠ Both figures DERIVED per run on `citation_drift_sweep.py`'s `heading oracle
COST` line, per repo as `novel=` beside `heading_unread=a/b` (sessions quoted
209 vs 213, each correct for its run). **Take them from the line.**

`issue_citation_gate.file_and_history_allocated()` (= `allocated()` minus
headings) makes the contribution measurable; `_heading_records()` is ONE walker
under both meters; armed both ways.

### ⛔ The arm that could not pass, and why it is a separate issue

T2 gave `(1, 5)` vs `(3, 6)` (scratch copy: `(3, 6)`, `{42, 47, 50}`): bare
`Path.write_text(...)` on **cp874** writes U+2014 as `0x97`, read back as
U+FFFD. **27 sites in that file; 155 here; 273 across 8 repos** = **Issue
829**; the 27 land here.

## Issue 823 — CLOSED: the heading oracle was anchored to a POSITION, and its own blindness meter was anchored to the same one; T5 ANSWERED by Issue 828 T4 (2026-09-18)

**Status: RESOLVED (T1–T4, T6 same day). T5 ANSWERED 2026-09-18 by Issue 828 T4 — by PRICING: of 153 unread records only 2 carry a number no other oracle knows. Read `heading oracle COST` on the sweep's summary line.**

### The symptom

`riir-clippy IN-LOCAL-RANGE 13 > pinned 12`, a day after a `10 -> 12` bump that
blamed "that repo's convention" for riir-clippy Issue 113's unparseable `##
<date> — Issue NNN:` heading. Second bump refused (*a diary, not a wall*); it
is six repos, 74 records.

### The defect, and it is TWO defects stacked

**(a) The oracle.** `_SELF_HEADING`'s rule (rejects `## Issue 043 follow-up
(…)`, reads `## Issue 043 (…)`) was anchored to the **leading** kind, so
riir-clippy's `## 2026-09-16 — Issue 113: the auto-oracle` was unreadable;
Issue 113 has no `.issues/` file or `git log` deletion (Issue-754 shape), so
its citations fell to IN-LOCAL-RANGE (`.highwater` 121 proves ownership).

**(b) The meter, the worse half.** `_HEADING_SHAPED` (measures what the oracle
rejects) had the same anchor and failed toward clean:

| repo | meter printed | actually unread |
|---|---|---|
| riir-chain | `heading_unread=0/1` — **perfect** | **20 of 21** |
| riir-dapps | `0/0` | **22 of 23** |
| riir-clippy | 30 unread | 54 |

Workspace: admitted **245** records where **341** exist.

### The repair (T1–T3)

`_SELF_HEADING_DATED` / `_HEADING_SHAPED_DATED` — same discriminator at the new
position (delimiter `:` or `,`, the date-led form's `(`). ⛔ **Not the unsound
widening** (dropping the discriminator): it rejects `## 2026-09-16 — Issue 152
resolved: …` and `## 2026-09-16 — Plan 064 T3 landed (…)`; arm 2's negative
pinned in both positions by two new arms. Foreign-repo filter runs over the
whole remainder (safe direction).

18 records became readable, all hand-verified genuine. Effect:

- workspace IN-LOCAL-RANGE **54 -> 27**
- oracle **110/319** read (was 92 of a believed 245)
- `riir-clippy` 13 -> **11**, `riir-neuron-db` 3 -> **2** — ratchets
  **tightened** in the landing commit
- `citation_drift_sweep.py` **PASSES**

### T4 — the stale prose

Sweep summary + `_HEADING_SHAPED` comment now say WHICH widening is unsound.

### T5 — OPEN: the residual, and the number this issue should NOT be read as

⚠ **Count DERIVED — read the sweep's `heading oracle` line** (209, then 213).
Residual = Issue 781's `## Issue NNN resolved — title (date)` family; **781's
class NOT closed** — its denominator was understated by 96. Open: does
`resolved —` admit a sound discriminator (AGENTS.md: no).

⚠ **Do not re-pin `max_in_local_range` for a heading-blind row again** (two
sessions did). Read `heading_unread=a/b` first — a repo near `b/b` cannot have
its IN-LOCAL-RANGE count trusted as editorial.

### T6 — the width bound had no floor, and its blind output is a PERFECT score

`heading_style_blind` was printed and asserted by nothing: a regressed shaped
pattern reads `0/0 records read, 0 UNREAD` — perfect coverage when blind.
`min_heading_shaped = 260` in `citation_drift_floors.txt`:

- **GLOBAL** (per-repo is legitimately 0; the wasm32 sweep's `TOTALS` solution).
- **Pinned under a PARTIAL clone** (319 over 16 of 20) — asserts the PARSER;
  never moves for an absent repo.
- ⚑ **Canaried — two independent detectors**: impossible pin (99999) fires the
  floor, exit 1; a broken shaped regex is caught earlier by the
  `heading_style_blind` arm, exit 2.
- An arm asserts the pin is present and > 0 (`parse_pins` would default quietly).

### Postscript — the write-up reproduced the class, again

Quoting riir-clippy's heading in AGENTS.md landed an IN-LOCAL-RANGE row vs
`max_in_local_range 0`; fixed by naming the owner in the 3-line window
(Issue-780 precedent). ⚑ Issue 797's split: DISPLAY `0` (worktree), PIN `1`
(HEAD) until commit — by design.

### Cross-repo

Nothing to file; the six repos need no heading change.

## Issue 829 — CLOSED: the FILE seam under Issue 778's PIPE seam: 273 text-I/O sites decode with the system locale, and the first one found was making a whole file of arms pass for the wrong reason (2026-09-18)

**Status: RESOLVED same day (T1-T6). 273 sites -> 2, and the 2 are outside the contract.**

### How it was found, which is the entire argument for a gate

Issue 828's dash arm failed (`(1, 5)` vs `(3, 6)`) where a scratch copy
passed: bare `Path.write_text(...)` uses `locale.getencoding()`, and on
**cp874** U+2014 reads back as U+FFFD via `read_text(encoding="utf-8",
errors="replace")`. ⛔ Existing arms passed anyway — **green for the wrong
reason**. Issue 778's class on the **FILE** seam, unseen on UTF-8 boxes (macOS,
`ubuntu-latest`, M3).

### T1 — the census

AST over tracked `*.py`, 16 repos:

| repo | sites | repo | sites |
|---|---|---|---|
| katgpt-rs | **155** | riir-chain | 7 |
| riir-train | 68 | riir-mmorpg-examples | 2 |
| riir-clippy | 27 | riir-shader | 2 |
| riir-ai | 10 | seal-game-editor | 2 |

**273 sites over 195 tracked `*.py` in 8 of 16 repos.** Two live:
`riir-clippy/scripts/gen_dashboard.py` reads `git log --pretty=%s` (every
subject has an em dash); `riir-train/scripts/build_clippy_v3_corpus.py`
round-trips corpus files via bare `read_text`/`write_text`.

### T2 — the repair half, `scripts/locale_io_fix.py`

AST-driven, idempotent, 18 arms; each lesson armed:

- ⛔ **`col_offset` is a UTF-8 BYTE offset** — char math desyncs two per em dash
  (first run asserted on `' '` 59664 chars in).
- ⛔ **Newline style is part of the file** — writing `\n` flipped
  `suite_membership_audit.py` CRLF→LF, an 11-site repair became a **515-line
  diff**; now reads bytes and preserves style.
- ⛔ **The kwarg joins the last ARGUMENT, not the `)`** — else leading-comma
  continuations (`        , encoding="utf-8")`); 128 sites went in before
  anyone read one.
- `**kwargs` splat = **UNKNOWN**, left alone (conservative only).

### T3 — the verdict half, `scripts/locale_io_gate.py` (docs_gate CHECK)

Walled at **0**, sharing `locale_io_fix.sites()` (Issue 755).

- Population = tracked `*.py`, not `scripts/*.py` (sites in
  `.agents/skills/doc-sync/tools/`, `.benchmarks/`, `.agents/`).
- **Two floors** — `min_py_files` (walk), `min_io_calls` (AST; counts
  compliant calls too, else it restates the ceiling).
- **UNPARSED reds.**
- Exemptions by membership with reasons, `locale_io_expected.txt`,
  **deliberately empty** (Issue 785); stale rows red.
- `--canary` arms the pin arithmetic (Issue 775), incl. the exemption reader's
  permissive direction and each floor independently.
- `--prove-fires 072a083b`: 128 offenders at Issue 828's landing commit.

### T4 — the sweep half, `scripts/locale_io_drift_sweep.py`

Landed with the gate (one-half shipping recorded nine times: Issues 777, 778,
793, 782, 783, 789, 797, 820). ⚠ **RATCHET on the derivative** (118 sites in
seven unowned repos, Issue 785); rows now **0**, still a ratchet for joining
repos. Wired to the Issue-797 advisory + Issue-822 `head_delta` provenance
(`sweep_advisory_membership_gate.py`). ⚑ First run reported **128 MASKED
rows** with the repair uncommitted.

### T5 — katgpt-rs repaired

155 sites: 27 in `citation_drift_sweep.py` (with Issue 828), 128 in 84 other
files. `docs_gate.sh` green, 26 CHECKS.

### T6 — the 118 sibling rows, LANDED, with the SHAs

Per the cite-the-sibling-commit rule; one commit per repo, tracked `*.py`
only, staged by NAME (concurrent `.rs`/`Cargo.lock` WIP present):

| repo | sites | commit |
|---|---|---|
| riir-train | 68 | `39d0b8d4` |
| riir-clippy | 27 | `638fa224` |
| riir-ai | 10 | `4df109c65` |
| riir-chain | 7 | `bebf78a` |
| riir-mmorpg-examples | 2 | `36005d2` |
| riir-shader | 2 | `7a65dd4` |

Every `locale_io_drift_floors.txt` row → 0 with its SHA. **273 → 2**; the 2
are `seal-game-editor`, a known-extra repo 260 commits behind upstream (the
Issue-798 advisory names it) — repairing it here would repair a checkout.

⛔ riir-train's raw diff (**+101/-72 over 33 files**) included concurrent
`Cargo.lock` / `plan402_p1_gemma_boundary_sweep.rs` edits; `*.py` only: **31
files, +68/-68**. *Read a shared diff through the population you changed.*

#### The gate's own foreign-repo mode, found by using it

`scripts/locale_io_gate.py ../riir-shader` reported `parse FLOOR breached: 2
text-I/O call site(s) < 450` — katgpt-rs floors applied to a foreign repo. A
named sibling is now a **REPORT** (no floors/exemptions, final line says so);
per-repo pins stay in `locale_io_drift_floors.txt`.

## Issue 824 — CLOSED: the family gate watched the failure it was built for happen beside it (2026-09-17)

**Status: RESOLVED same day (T1–T3).**

### The symptom

`cfg_row_implication_drift_sweep.py` hard-red with **zero content findings**:
`seal-game-editor` / `seal-online-remaster` / `seal-remake`: "no pin row — a
new repo must be pinned deliberately" — all three acknowledged by
`DOCS_GATE_KNOWN_EXTRA`. Issue 821's symptom, in a sweep 821 closed.

### What actually happened

Issue 821's `sweep_population.pin_row_exempt()` close-out said *"wired into
**16** sweeps"*; the family is **19**:

| sweep | state | why it wasn't noticed |
|---|---|---|
| `cfg_row_implication` | **hard red, live** | not run since 821 |
| `restatement` | hard-fails, **latent** | `.proofs` subset; no acknowledged extra has one *today* |
| `docs` | advisory only | would advise pinning a repo the contract doesn't claim |

⛔ The same three AGENTS.md names from Issue 782 as "quieter" — missed twice by
two repairs: **a hand-grep of a family finds the loud members.**

### The real finding — and it is about the gate, not the sweeps

`scripts/sweep_advisory_membership_gate.py` stayed green beside 821 because it
governed `sweep_advisory()` *by name*; the class is **any mechanism every member
must call**. Eighth instance (777, 778, 793, 782, 783, 789, 797, 821) — the
*prevention* didn't generalise.

### The repair (T1–T3)

**T1** — all 19 call `pin_row_exempt()`; the three green; exemption nested
INSIDE `if pin is None:` (per 821's note, else the else-branch dereferences the
missing row).

**T2 — `MECHANISMS` registry** `slug -> (why, call-names)`; verdict per
mechanism, **never pooled** (pooling reported the 16-of-19 state as wired);
arms walk the registry, so a new mechanism is armed by EXISTING.

**T3 — pin key `(mechanism, sweep)`** (NAMES-never-`=1` asymmetry); unqualified
rows and unknown slugs REFUSED; file **deliberately empty**.

Canaried by arms (advisory-only fixture → UNWIRED/WIRED independently) and the
production path (`✗ UNWIRED [known-extra-exemption] citation_drift_sweep.py`,
exit 1).

### Postscript — how it was found

Reading EXIT CODES of every sweep (a glyph grep printed `<no verdict line>` 19
times). ⚠ `numbering` exit **127, 0-byte log** under memory squeeze (1.29 GB
free, 29.6 GB training job), passes alone: 127 + empty log = infrastructure;
re-run alone (`x86_64_execution_matrix` rule).

## Issue 821 — CLOSED: Issue 815's marker never reached the sweep family: 8 sweeps red on 0 findings, and two live ratchet breaches sat behind them (2026-09-17)

**Status:** CLOSED 2026-09-17
**Severity:** a sweep that always reds is a sweep nobody runs — measured, twice now
**Owner:** this session

### The finding

Issue 815's `DOCS_GATE_KNOWN_EXTRA` landed in `skill_repo_set_gate.known_extra_state` and `sweep_population.population_verdict` (the final-line disclosure) — **not** in the per-repo pin loop the sweeps run earlier. With both markers set (`DOCS_GATE_PARTIAL_CLONE=1`, `DOCS_GATE_KNOWN_EXTRA=seal-game-editor,seal-online-remaster,seal-remake`) a sweep printed `✗ seal-game-editor … UNPINNED — add a row` above a final line saying those repos are "not measured and not expected to be".

### Measured, 2026-09-17 — with both markers set

| sweep | rc | why it reds |
|---|---|---|
| `orphaned_attr` | 1 | seal-* UNPINNED · **0 findings** |
| `percentile` | 1 | seal-* UNPINNED · **0 findings** |
| `markdown_fence` | 1 | seal-* UNPINNED · **0 findings** |
| `subprocess_encoding` | 1 | seal-* UNPINNED · **0 findings** |
| `trap_sentinel` | 1 | seal-* UNPINNED · **0 findings** |
| `platform_dead_code` | 1 | seal-* "a repo joined" · **0 findings** |
| `console_encoding` | 1 | seal-* UNPINNED **+ real breach** |
| `instrument_reachability` | 1 | seal-* UNPINNED **+ real breach** |
| `wasm32_surface` | 1 | seal-* UNPINNED + NEW UNCOVERED row in an extra repo |

**Eight of nine red on repos where they found nothing**, hiding two genuine riir-train ratchet breaches: `console_encoding` undefended 56 > pinned 53; `instrument_reachability` unreachable 63 > pinned 61.

⛔ **Second recorded instance:** Issue 793 found seven sweeps hard-redding with `DOCS_GATE_PARTIAL_CLONE=1` set, hiding the Issue-777 findings and four citation drift rows; repaired by `sweep_population.py`. Issue 815 added a second marker without putting it through that mechanism's other half.

### Why one shared helper and not 14 edits of the same idea

AGENTS.md: *"grep the whole family and land the repair as one shared mechanism."* Issue 820 wrote that down and then fixed the marker in `numbering_drift_sweep.py` alone.

### Tasks

- [x] **T1** — `sweep_population.pin_row_exempt(name)`; the acknowledged/stale split stays in `population_verdict` (the loop only visits PRESENT repos, so a re-derived per-row split would be a second copy).
- [x] **T2** — 14 uniform call sites + `toolchain_override_drift_sweep`'s `repo_flags()` + fold `numbering_drift_sweep`'s Issue-820 copy onto the helper.
- [x] **T3** — arms in `sweep_population.selftest`: exempt when declared, NOT when undeclared (why 815 takes NAMES not `=1`), no marker exempts nobody.
- [x] **T4** — re-run the nine; the two riir-train breaches are **riir-train's to adjudicate** — this issue owes making them VISIBLE.

### Not in scope

- **The two riir-train breaches** — filed onward; that repo's `scripts/` is mostly plan-scoped one-offs (predicate over-captures), so 63-vs-61 is a derivative question.
- **Whether seal-* belong in the workspace** — owner-owned per Issue 815.

### Landed

`pin_row_exempt(name)` wired into **16** sweeps (13 uniform, `toolchain_override`'s `repo_flags()`, `platform_dead_code`'s set-difference, `numbering`'s Issue-820 copy). Before → after: sweeps red 9/14 → 4/14; red on 0 findings 8 → **0**. The four still red were hidden behind the noise: the two riir-train breaches, `cfg_gated` riir-ai (silent_now 2, load_bearing 1), `citation` riir-ai (1 CROSS row).

⚠ **`wasm32_surface` needed more:** its UNCOVERED check sits outside the `row is not None` branch. **An acknowledged extra contributes no VERDICT, not merely no pin row.** The row is still PRINTED (Issue 797's display-reads-box / pins-read-contract split); a pinned row naming an extra repo must still be removed deliberately.

### Two process notes, both of which cost time

⛔ **The mechanical 13-file edit crashed:** flat `if row is None and not pin_row_exempt(name):` sent the exempt case into an `else:` dereferencing `row` (`TypeError`). Nest the exemption INSIDE `if row is None:`. A scripted multi-file edit needs a `py_compile` sweep and one real run.

⛔ **Python `write_text` flipped files to CRLF** (345/345 whole-file rewrites). `git diff --numstat` after scripted edits; write BYTES or pass LF newline.

## Issue 820 — CLOSED: the numbering sweep is blind to Issue 795's class, so 15 repos read `dup=0` over 113 collisions (2026-09-17)

**Status:** CLOSED 2026-09-17
**Severity:** the sweep prints a confident green over the MAJORITY case
**Owner:** this session

### The finding

`scripts/numbering_gate.py` gained `historical_collisions()` at Issue 795 (a doc closed and deleted leaves nothing on disk; 70 collisions in scope, 9 from one 57-commit divergence). `scripts/numbering_drift_sweep.py` — the cross-repo verdict half — never got it: it read only the worktree, so the class was measured in **one repo of sixteen**. Eighth instance of the never-generalised shape (Issues 777, 778, 793, 782, 783, 789, 797).

### Measured, 2026-09-17

| repo | collisions | ≥ 700 | numbers walked | time |
|---|---:|---:|---:|---:|
| katgpt-rs | 70 | 9 | 1407 | 0.7s |
| riir-ai | **71** | **3** | 1435 | 0.4s |
| riir-train | 11 | 0 | 553 | 0.5s |
| seal-game-editor | 10 | 0 | 337 | 0.4s |
| riir-chain | 6 | 0 | 225 | 0.3s |
| seal-online-remaster | 5 | 0 | 72 | 0.1s |
| riir-mmorpg-examples | 4 | 0 | 135 | 0.2s |
| riir-clippy | 3 | 0 | 443 | 0.2s |
| riir-neuron-db | 3 | 0 | 111 | 0.2s |
| (7 others) | 0 | 0 | 196 | — |
| **TOTAL** | **183** | **12** | **4914** | **3.7s** |

**113 in repos the sweep called clean**, 3 above the era boundary, all riir-ai: `.issues/702` (l2_normalize_suite_order_numeric_divergence / persistent_grid_stride_gemv), `.issues/753` (qwen38_cudarc_kv_dtype_f32_lever / webrtc_tier_gate_too_narrow_for_whip_client_server), `.issues/959` (feature_gap_dead_code_consumer_sweep / heart_icon_text2d_raster_hierarchy_traps). ⚠ A prior hand-off said riir-ai had "3 unpinned" — the tracked-only view by eye; the instrument says **71**. 3.7s workspace-wide: no cost argument for the omission.

### Tasks

- [x] **T1** — `audit()` gains `hist` + `n_numbers`, from `ng.historical_collisions` **imported, not re-derived** (Issue 755).
- [x] **T2** — two columns in `numbering_drift_floors.txt`:
      - `max_hist` — **RATCHET at measured** (not a wall: each is an Issue 724 T2 arbitration, 113 in unowned repos; not membership: invented reasons are a backlog wearing a pin, Issue 785).
      - `min_numbers` — **FLOOR on the HISTORY walk**, distinct from `min_files`: a `git log` regression collapses `n_numbers` to the on-disk count while `min_files` stays.
- [x] **T3** — katgpt-rs's row asserts the **gate's verdict** (`collision_verdict(rows, *parse_collision_pins(...))` returns no failures) rather than restating a count true by construction, so a stale pin in `number_collisions_expected.txt` reds the sweep.
- [x] **T4** — arms over the new arithmetic (Issue 775): ratchet both directions, walk floor firing, reasonless/short row REFUSED.

### Not in scope

- **Resolving any of the 113** — Issue 724 T2 arbitrations in other sessions' repos.
- **riir-ai's 3 above-boundary rows** — riir-ai's to adjudicate; the ratchet keeps them from growing.
- **A per-push gate in riir-ai** — it has no `docs_gate.sh` CHECKS array (Issue 789 T4).

### Landed

Sweep line: `files= nums= dup= hist= stale= malformed= resets= unbumped=`; first green run: **0 tracked duplicates vs 184 historical collisions over 16 repos** (184: a concurrent session allocated `.issues/819` that day).

**T5, unplanned — live above-boundary collision.** `numbering_gate.py` was RED on `develop`: two sessions allocated 819 off one counter. Hand census 10 sites / 0% UNRESOLVED: **7 to the sigmoid prior-logit forecast issue** (`.research/566` x5, `.research/258`, `.research/392`) vs **3 to the x86_64 lint lane** (`AGENTS.md` x2, `HISTORY.md` x1); citation_weight agreed 6-3. Sigmoid issue KEEPS 819 (Issue 724 T2). ⚠ Hand-checked because `lane` is a winning-file token and Layer 2c prose is about a lint LANE — it did not mis-hand sites. Loser CLOSED and REMOVED, so the pair is permanent in the recovery walk; repair = pin + three-site disambiguation (Issue 794), not a renumber.

**T6, unplanned — Issue 815's marker did not reach the sweeps' pin loop.** Excused by NAME now, both directions asserted.

**T7, unplanned — riir-clippy `.research` allocator stale at 176 with 177 on disk** (next allocation guaranteed to collide). Confirmed against origin (Issue 798); repaired at riir-clippy `672ba8a8`.

### Verification addendum (2026-09-17) — the floor was argued, then EXECUTED

Prompted by Issue 823 (`heading_style_blind` printed a perfect `0/0` when blind): *what does this print when it breaks?* T2's `min_numbers` was argued, not run; synthetic arms can't show the real pins are tight. Executed by blinding `citation_weight.removed_by_number` (returns `{}` as a failing `git log -M --diff-filter=D` does):

| repo | nums | blinded | `min_numbers` | hist | blinded | fires? |
|---|---:|---:|---:|---:|---:|---|
| katgpt-rs | 1412 | 1047 | 1200 | 71 | 0 | ✓ |
| riir-ai | 1436 | 678 | 1200 | 71 | 0 | ✓ |
| riir-chain | 225 | 80 | 180 | 6 | 0 | ✓ |
| riir-clippy | 444 | 332 | 350 | 3 | 0 | ✓ |
| riir-dapps | 92 | 44 | 70 | 0 | 0 | ✓ |
| riir-game-sdk | 42 | 9 | 33 | 0 | 0 | ✓ |
| riir-mmorpg-examples | 135 | 32 | 108 | 4 | 0 | ✓ |
| riir-neuron-db | 111 | 52 | 88 | 3 | 0 | ✓ |
| riir-shader | 23 | 5 | 18 | 0 | 0 | ✓ |
| riir-train | 553 | 372 | 440 | 11 | 0 | ✓ |
| riir-viewbridge | 8 | 0 | 6 | 0 | 0 | ✓ |

All 184 collisions vanish and every non-zero floor reds first — no green zero. `riir-auth`/`riir-kat` carry `min_numbers 0` (3 and 0 numbers): a disclosed gap. ⚠ Not landed as a tracked script (one-shot `--prove-fires`-shaped validation; an unrooted `scripts/*.py` is the `instrument_reachability_gate` finding). Durable artifacts: the floor + this table.

### What this does NOT claim

- The 113 sibling collisions are **not** adjudicated; the ratchet reds the 114th.
- Seven repos absent from this box: new columns **UNMEASURED**, pinned `0 0` saying so; `max_hist = 0` will red deliberately on the first full run. Re-pin all seven from one full-checkout run.

## Issue 822 (2026-09-18) — CLOSED. An UNCOMMITTED row was counted into a RATCHET; all 19 sweeps adjudicate against HEAD now, and the mechanism is GATED

**The defect.** Every `*_drift_sweep.py` walks the WORKING TREE, shared by 5+ concurrent sessions, so a ceiling comparison depended on one session's uncommitted index. Measured: `console_encoding` undefended 56 > pinned 53, one new row on a file staged by another session mid-commit, invisible to `git log`.

**The repair, three instruments** (`worktree_state.py`), chosen by classifier shape:

| instrument | for | cost |
|---|---|---|
| `head_delta` | per-file classifiers | \|dirty ∩ population\| `git show` calls, **zero** clean |
| `head_overlay` + `delta_of` | cross-file, re-classified from a dict | same reads + one classification |
| `head_tree` | MULTI-SEAM (`git grep` + `git ls-files` + reads) | materialised HEAD: **22-32s per dirty repo**, 0.04-0.26s clean |

`head_tree` (Issue 822 T5g): `git archive` HEAD + `git init` + `git add -A -f` so the classifier runs UNMODIFIED (`-f` load-bearing vs extracted `.gitignore`). `paths=` narrows the archive (riir-train 604 MB → 30.6 MB, **2.2x** end to end, not 20x); `extra_dirty=` widens the trigger for `os.walk` sweeps; checkout named after the SOURCE repo since `len_derived` keys rows on directory name.

**Rules, each measured:**

* ⛔ **The VERDICT belongs in the row KEY** — a key-matched row is filed COMMITTED with the WORKTREE's object, so any field the key omits is silently overridden (T5f, T5g, T5i, T5j).
* ⛔ **A FLOOR is a pin too** (Issue 797: 607 → 601 citations). HEAD's walk size is DERIVED; only a worktree deletion (walk one LARGER at HEAD) is reachable by uncommitted work.
* ⛔ **Classes can split by ORACLE:** `numbering`'s `dup`/`above`/`malformed` read the worktree, `hist`/`resets` read `git log`, `unbumped` is a WORKTREE quantity by construction.
* ⛔ **A census over ONE REPRESENTATION is blind to what it omits:** grepping `> row["max_*"]` missed `docs_drift`'s wall written `if b_mis:` (Issue 787's finding).
* ⚠ **An arm over inputs that cannot express its rule passes for the wrong reason** — three fixtures rewritten (EQUIVALENT line-in-key mutant; `cfg_gated` `features` can't vary; `required-features = []` outside `cfg_row_implication`'s population). katgpt-rs-54 hit it the same night: an 827 fixture used an unowned number where `is_qualified` returns True by construction.

⛔ **The shared generalisation: reasoning from ONE VIEW of a CONTENDED thing returns the reassuring answer** — census over one representation, verification over a snapshot, fixture that cannot express its rule; all live 2026-09-18, all read clean.

**Diagnostic rule** (katgpt-rs-54): a numbering sweep run during ~90s of another session's `git rebase` gave two breaches true of that instant only; checking worktree, HEAD blob, 30 commits and `.git/rebase-merge` inspects STATE, which had moved. **The one decisive test is to RE-RUN THE INSTRUMENT.**

**Closed with a REGISTRY row** (T6): `head-provenance` in `sweep_advisory_membership_gate.MECHANISMS`, entry points `head_delta`/`head_overlay`/`head_tree`/`head_text` (the last credits `citation_drift_sweep`'s inline original). 19/19 on all three mechanisms, exemption file empty; registered only AFTER the fan-out (Issue 785). **Take family size and wiring from that gate's PASS line, never prose.**

⚑ **First live MASKED row, four hours later (2026-09-18):**

    ⛔ riir-ai/scripts/repair_t1_outcomes_json.py  [MASKED — committed, and this worktree hides it]
    ✗ riir-ai   scripts=9  roots=4  unreachable=7 (8 committed, 1 MASKED)

Script committed (riir-ai `7e8a6b35a`, its Issue 976 repair); the AGENTS.md line making it reachable was uncommitted, so a worktree-only sweep would have PASSED at 7. Cleared at riir-ai `f9f183a0f` by committing the doc, not raising the ceiling — "landed means committed" covers docs too. ⚠ Masked by its own uncommitted REPAIR (minutes-long window), the mildest form. ⛔ The worktree run passes **only on the box holding the fix**; the class is not rare, only previously INVISIBLE.

⛔ **Attribution hazard:** every session commits as `katopz <katopz@gmail.com>`, so author lines identify nobody; misreading one produced a wrong WORK SPLIT (two sweeps untouched by both). `ListAgents` showed TWO live `katgpt-rs-54`s and a fourth session two eliminations missed. Only an explicit `Session: <name>` body line is reliable. ⚠ A claim "ListAgents shows only the two of us" was made without running `ListAgents` — an unperformed verification is as unreliable as the inference it replaces.

## Issue 819 (2026-09-17) — CLOSED. The x86_64 arm was LINTED by nothing; the finding is not the 30, it is the sibling

⚠ **819 is held by two documents**; the same-day sigmoid prior-logit forecast issue KEEPS it (7 of 10 inbound, 0% unresolved — Issue 724 T2). This is the *x86_64 lint lane*; adjudication pinned in `scripts/number_collisions_expected.txt`.

Found on the Issue 808 workstation: `cargo clippy -p katgpt-attn --all-features --lib` with `+avx2` → **30** `unsafe_op_in_unsafe_fn` (edition 2024), all in `crates/katgpt-attn/src/dash_attn/channel_aware.rs` AVX2 arm; **0** with avx2 off.

**The finding is the SIBLING:**

| arm | cfg | body |
|---|---|---|
| `simd_dot_neon` | `target_arch = "aarch64"` | `// SAFETY:` + whole-body `unsafe { … }` |
| `simd_dot_avx2` | `all(target_arch = "x86_64", target_feature = "avx2")` | **neither** |

`full_gate.sh` is macOS/aarch64, so the AVX2 twin compiled to nothing even under `--all-features`. *A repair applied to the arm somebody can see is a measurement of which arms are visible.*

**Why every lane was blind:** Layers 2/3/6 are aarch64; Layer 2b/`wasm32_gate.yml` build wasm32; `test_gate.sh` doesn't lint; `x86_64_execution_matrix.sh` (2026-09-16) EXECUTES with `+avx2` but doesn't lint — the inverse hole. No lane declared the surface, so `--allow-partial-platform` couldn't name it. `target_feature = "avx2"` is a SECOND gate (Issue 737's wasm32/simd128 shape: 0 vs 14 there, 0 vs 30 here).

**T2 — repair:** whole-body `unsafe { … }` + `// SAFETY:` matching NEON (healer silent on this class). After: 0 warnings, 427/0 unchanged; diff 46/43 verified by numstat not to be a line-ending rewrite.

**T3 — `full_gate.sh` Layer 2c**, mirroring 2b: derived `-p` list off positive `target_arch = "x86_64"`, BOTH avx2 arms, `-D warnings`, `--keep-going`, instrument floor, missing target = PARTIAL refusal without `--allow-partial-platform`.

- ⛔ **The TRIPLE is the design decision, printed on the verdict line:** host triple on x86_64 hosts, else `x86_64-apple-darwin` (Darwin) / `x86_64-unknown-linux-gnu`. `grep -qx` not `grep -q` against `rustup target list --installed`.
- ⚠ **`--all-features` required:** `katgpt-attn` is `default = []`, so `dash_attn` compiles to nothing at defaults — a green zero. Also restores Layer 3 coverage on x86_64 and avoids a hand-typed feature list beside four `required-features` targets.
- **Residue pin = non-`src/` surface MINUS named rows, expected EMPTY** (pinning the paths would restate its input). Four arms incl. the `set -euo pipefail` all-filtered case (`grep -v` exits 1); a row whose file is gone is caught by cargo.

**T4 — population:** `katgpt-attn`, `katgpt-core`, `katgpt-pruners`, `katgpt-tokenizer`, `katgpt-types` + four named targets, 28 x86_64 / 6 avx2 files: **0 findings** outside `channel_aware.rs` (measured, cf. Issue 737's unexpected 14).

**Two-sided validation:** green on the repair, exactly `30 previous errors` on `HEAD~1`'s file.

⚠ **Operational hazard, not papered over:** `--all-features` pulls `good_lp` → `highs-sys` → cmake/C++; under load the 24-way `cmake --parallel` died with `cl … D8040`; `CMAKE_BUILD_PARALLEL_LEVEL=4` built clean. Not capped (Layer 3 has the same exposure); remedy recorded at the layer.

## Issue 819, second holder (2026-09-17) — the sigmoid prior-logit lane KEPT the number; RESOLVED (file removed)

Won the same-day dual allocation (7 of 10 inbound, 0% unresolved, Issue 724 T2; pinned in `scripts/number_collisions_expected.txt`). T1–T3 + T5 LANDED with Bench 813 (`.benchmarks/813_prior_lane_margin_goat.md`) G1–G4 ALL PASS: sigmoid prior-logit lane + sink-stability forecast (Research 566, arXiv:2601.15380) as two OPT-IN features (no-default-consumer rule). T4 (owner-gated) closed unclaimed. Task record in git history (`819_sigmoid_prior_logit_lane_sink_margin_forecast`). "Issue 819" in .research/566, 258, 392 means THIS lane, never the x86_64 lint lane (disambiguated per Issue 794).

## Issue 815 (2026-09-17) — CLOSED by its own criterion: the box is green. Option 2 landed as `DOCS_GATE_KNOWN_EXTRA`, and options 1 and 3 remain the owner's

⚠ **The REVERSIBLE option taken absent a decision.** Options: join the three `seal-*` repos (1), known-extra marker (2), box hygiene (3). Option 2 touches neither the 20-repo contract nor read-only `seal-online-remaster`; 1 and 3 stay cheap, and dropping the marker restores prior behaviour.

MIRROR of `DOCS_GATE_PARTIAL_CLONE` (absent vs present-unregistered). ⛔ Takes **NAMES, never `=1`** (`=1` would excuse the next unregistered repo). Reds both directions (name gone or since registered = STALE). Never auto-detected. One classifier (`skill_repo_set_gate.known_extra_state`) for all three call sites (Issue-793 rule).

**Three instrument defects found while landing:**

1. ⛔ `skill_repo_set_gate` printed "16 of 20 canonical repos present" over 13 canonical + 3 extra — the partial-set-as-whole defect in the gate's own display.
2. ⛔ Both selftests read the AMBIENT env while building a SYNTHETIC workspace, failing two Issue-765 arms on a correct invocation; markers now saved/cleared/restored around arms.
3. ⛔ `population_sync_gate --canary` died with UnicodeEncodeError on cp874 (defence only in `main()`) — the Issue-804 class; `console_safe.apply()` moved to `__main__`.

⛔ **A fourth, via `arm_reach_gate`, closed by EXTRACTION:** the sole UNPINNED survivor was an `and -> or` flip in `known_extra_state`'s snapshot parse, which existed **three times**; in the new copy the flip was genuinely EQUIVALENT — the trap (a third of "EQUIVALENT" rows were once real gaps). Repair per Issue 755: one `read_snapshot()`, armed by the existing Issue-790 fixture; mutants 100 → 97, survivors back to the two pinned rows.

Arms: 16/16 `population_sync_gate --canary`; 13 COUNTED in `sweep_population.selftest` (hand-typed "7 assertion(s)" replaced by a derived count, Issue 798 T3); and the key `skill_repo_set_gate` arm: an UNNAMED extra still reds beside a named one.

## Issue 808 (2026-09-17) — CLOSED: T1 landed option 2, the AVX2 `argtopk` dispatch narrowed to k ≤ 4 on x86_64; the measured loss is GONE and the wins are kept

Owner's call on Bench 810: **option 2, x86_64 only** — `AVX2_ARGTOPK_K_MAX = 4` inside x86_64 `argtopk_simd` (the `argtopk_with_scratch` predicate stays `k ≤ 16`); NEON keeps k ≤ 16 (algorithm witness, not a measured loss). Option 1 refused: `N_MIN` is distribution-sensitive (k=8: 256 `locality`, 768 `iid_uniform`, >1024 `late_peak`) and the dispatch can't see the distribution.

**Measured** (4090, i7-13700K, `+avx2`, release, interleaved median-of-ratios): `late_peak` k=8 **0.41–0.72× → 0.98–0.99×**; worst cell above the bound **0.95×** (6 distributions × k ∈ {8,16} × n ∈ {64..512}). Wins kept: k=2 **5.11×**, k=4 **3.51×**, k=1 **6.73×** (n=1024).

**T3's lesson applied:** narrowing would make the two Issue 806 regression tests assert the SCALAR path at k ∈ {8,12,16}; the kernel was hoisted to `argtopk_avx2_kernel` and `test_argtopk_avx2_kernel_matches_reference_above_dispatch_bound` calls it directly over both 806 fixtures at k ∈ {4,5,8,12,16}. Both new tests verified NON-VACUOUS by perturbation.

**T2 closed in two halves:** load-immune `test_avx2_argtopk_dispatch_bound_is_pinned` (perf bars gave four failing sets across four runs of one commit, Bench 806 T7) + timing `bench_simd_topk_issue808_t2_above_bound_is_not_a_loss` floored at **0.85×** (expected ~1.00, defect 0.41×). The ≥2-microarchitecture bar is **not** discharged — irrelevant to narrowing, still gates any WIDENING; reopen instrument `bench_simd_topk_issue808_crossover_nmin` KEPT.

## Issue 806 (2026-09-17) — CLOSED: T8 was the last open task and it was entirely Issue 808's T1

T6/T7 closed 2026-09-16 (Bench 806 Addenda I+II); T8 went to Issue 808, now closed. Record: `.benchmarks/806_x86_64_execution_matrix.md`; matrix `scripts/x86_64_execution_matrix.sh` stays a workstation verdict, no CI lane.

## Issue 818 CLOSED (2026-09-17) — the four no-default `--all-targets` breaks gated both halves; bench_412's green-zero row found in the same sweep

Fix `78a1ac5d7`. `cargo check -p katgpt-core --no-default-features --all-targets` named four targets, each fixed with whole-file `#![cfg]` + matching `required-features` row (Plan 599 house spelling): `bench_371` re-gained the row deleted at its Plan-371-Phase-6 promotion (harness=false bench emptied by `#![cfg]` → E0601, so the ROW is load-bearing); `bench_416` → `region_subspace_steering`; `bench_778` → `subspace_intervention` (also broken at default `--all-targets`); `velocity_field_ensemble_alloc_check` → `velocity_field_ensemble`. **Fifth:** `bench_412` had the cfg but no row (green-zero class); row added. Fixed targets NON-VACUOUS (3+1i / 4+1i / 2 / 1 passed); no-default/default/all-features `--all-targets` clean; default lib 2063/0. Corollary: promoting/demoting a default-on feature must MOVE its target gates, never delete them.

## Issue 808 (2026-09-17) — T4 LANDED: `argtopk` AVX2 dispatch re-measured on six realistic block-score distributions + per-k `N_MIN` crossover (Bench 810)

**Bench 810** (4090; option-4 half + T2's Raptor-Lake row): the "k≤16 is a loss" table came from ONE i.i.d. fixture. Across six distributions × both profile arms the loss SURVIVES except `early_peak` and DEEPENS on `late_peak` (0.41–0.72× below n=512, 0.90–0.95× at n=1024 — **no n-floor rescues k=8**); k≤4 is distribution-robust (win/tie past n=64–128, up to 3.9×); `N_MIN` is distribution-sensitive, so T2 needs the distribution axis. Instrument: Issue-723 interleaved median-of-ratios (`tests/common/ab_timing.rs`), tests `bench_simd_topk_issue808_{distribution_matrix,crossover_nmin}` in `tests/bench_256_simd_topk.rs`. T2's ≥2-microarchitecture bar 1 of 2.

## Issue 817 (2026-09-17) — single-pass AVX2 `argmax` port measured NEGATIVE + reverted; the NEON premise does not transfer (Bench 812)

Filed and closed same day; this heading is the allocation record. **Bench 812**: x86_64 `simd_argmax_f32` two-pass cost depends on max position; a single-pass AVX2 port of the NEON kernel (8-lane (max,index), blend-on-strict-gt, correctness green) was a NET LOSS: `iid` 0.24–0.28×, `early` 0.20–0.23× at n ≥ 256 both profiles; won only `late` (1.5–1.9×) and n=64. Demote-on-loss: dispatch reverted, kernel deleted, numbers + reopen trigger in the doc comment, `bench_817_argmax_dispatch_ab.rs` KEPT as reopen instrument, equivalence tests kept.

## Issue 811 CLOSED (2026-09-17) — DBTM confidence-commit anchor rule: PoC PASS → Plan 600 landed end-to-end, promotion EXECUTED via Plan 601

[`.research/563_DBTM_Discrete_Beckmann_One_Step_Language.md`](.research/563_DBTM_Discrete_Beckmann_One_Step_Language.md) (`e0e63f47b`) distilled arXiv:2609.15903 §5.1 κ ∪ floor + §4.3 t\* law; PoC (`ee105eb78`) converged 3.1–4.7× faster than strided at quality parity, termination-within-k proven at every (κ, k). [Plan 600](.plans/600_flashar_confidence_commit_upgrade.md): `ConfidenceAnchorConfig` + `select_confidence_anchors` + `anchor_then_fill_with` (`8bd6d7132`), docs (`261d80fd2`), T8+T9 GOAT green (`df49ed4c5`, `5c6cb58e5`, Bench 600). [Plan 601](.plans/601_flashar_realtext_eval.md): real-text D2F eval, char-level Austen (`a769a42ec`, Bench 601: paired Δ +0.03…+0.11, 1.2–2.8× steps / 0.72–0.91× wall, realized-KL 0.48–0.87×) → **promotion executed**: `ConfidenceAnchorConfig::default()` = κ 0.9 + floor; strided stays the no-floor comparator. Deferrals discharged: T3 sweep arm via the Issue-813 seam + Bench 809 (`d392600d6`), NO-SEPARATION resolved by the Issue-816 trainer (`b3016516e`) — prob-t\* wins (2.59/2.61 vs uniform 2.89/2.88 nats); T5 UGC cross-check = Plan 600 T9 green (MC 8192×2: KL ratio 0.996 pattern / 0.48–0.87× real text). t\* half in katgpt-core: `commit_time_star` (`ignition_schedule`) + `probability_order` (`set_diffusion_schedule`). Follow-on under Plans 600/601 and Bench 809 addendum.

## riir-ai Issue 964 C2 LANDED (2026-09-16) — `CalibratedActionBridge`: decision-level confidence calibration for the ABSTAIN threshold (Bench 808)

Second `sigmoid_calibration` consumer: `bridge::calibrated::CalibratedActionBridge<A, D>` (no new flag) puts the Platt calibrator in front of the ABSTAIN threshold — observe (raw, succeeded), refit off-hot-path. **Argmax invariant:** selection stays on raw scores; calibration changes only what confidence MEANS. GOAT (Bench 808, planted `sigmoid(1.4·logit − 0.3)`, 4096/4096): ECE 0.0220 → 0.0088 (2.5×); log-loss + Brier beat raw and base-rate floor; cold start bit-identical; 0 winner mismatches; ABSTAIN point at τ=0.75 moved 4.4× toward oracle (0.3315 → 0.2764 vs 0.2866); observe+select 0 allocs. Caveats: thin G2 margins, fitted-b recovery 0.043 wide; riir-ai `arg_runtime` Step-8/9 wiring is the follow-on. Tests 21/21, lib 2060/0, clippy clean.

## riir-ai Issue 964 C1 LANDED (2026-09-16) — `clr_calibration`: the CLR verifier becomes the first `sigmoid_calibration` consumer (Bench 807)

CLR verdicts were only bounded (Bench 284 G2's fixture used the verifier's OWN sigmoid as truth). `clr_calibration` (katgpt-claim, opt-in) ships `CalibratedVerifier<V>` wrapping any `ClaimVerifier` with the Issue-810 calibrator in front of `r_k = (mean_m v)^M`. GOAT (Bench 807): planted drift ECE 0.0924 → **0.0164** (5.6×), recovery (0.642, 0.252) vs planted (0.625, 0.250); log-loss 0.4819 / Brier 0.1591 beat raw and base-rate floor; calibrated G2 fixture undisturbed (ΔECE +0.0002, ±0.005 no-harm band); vote winner stable 25/25; zero-alloc. Substrate fix: `apply` identity fast path at `(w, c) = (1, 0)` (logit→sigmoid not bit-exact in f32), pinned by two tests + G3a. Dev note: first draft's splits used different direction vectors (ECE 0.35); now shared. Docs `clr/traits.rs`, `clr/verifier.rs`; catalog §114; README 612→613; clippy clean; docs gate 25/25.

## Issue 812 CLOSED (2026-09-16) — the bench_doc_audit BlindRead context split was a TMPDIR-FORM split; the arm now matches paths form-independently

A red/green split on an identical blob traced to `TMPDIR=/tmp/...` (unresolved symlink): the arm matched by `startswith(str(root))` while `audit_repo` resolves to `/private/tmp/...`, so both unreadable arms reported `got False, want True`. Not interpreter (3.11–3.14 green) nor patch target. Fix: `root_forms = (str(root), str(root.resolve()))` in `scripts/bench_doc_audit.py` `blindness_arms()`; green both forms, pre-fix red is the canary; docs gate 25/25. Only `setattr(Path, ...)` arm in the workspace. Issue file removed.

## Issue 810 CLOSED (2026-09-16) — `sigmoid_calibration`: the Platt-style calibrated sigmoid gate lands as a PoC with all four gates green

From Research 562: sigmoid scalars are Lean-proven bounded but their MEANING is proven nowhere. `crates/katgpt-core/src/sigmoid_calibration.rs` (opt-in): `SigmoidGateCalibrator` (FIFO observe zero-alloc; deterministic 2-param Newton refit off-hot-path; apply = logit + fma + sigmoid; BLAKE3 commitment incl. `n_obs_total`), `CalibratedGateSet<const N>`, `brier_score` / `log_loss` / `expected_calibration_error`. Guard `w = 1/T > 0` → never reorders (G3 by construction). Planted `sigmoid(1.6z−0.4)`, n=2048: **G1** ECE 0.0696 → 0.0322, recovery T=0.627/b=0.243 (0.625/0.25); **G2** log-loss 0.5923 < 0.6066 < floor 0.6878, Brier 0.2026 < 0.2090 < 0.2473; **G3** lib 2060/0; **G4** zero-alloc (Issue-741 predicate). README 611→612, catalog §113, 11/11 tests. Consumers filed as riir-ai Issue 964. Issue file removed.

## Issue 809 T1+T2 CLOSED (2026-09-16) — the global-RNG census is read; the class is gated; the T3 `Rng::new()` census is deferred with a reason

WIDE predicate (every free-function `fastrand::<primitive>()` over tracked `*.rs`): **20 sites / 9 files**. One defect: `bench_pflash_maxsim_block_scoring` fixture unseeded → seeded `Rng::with_seed(809)`, no verdict moved. Rest deliberate (SDAR stochastic by design; cgsp identity entropy per `Uuid::now_v7()`; ≥6.3σ statistical tests; demos). T2: `scripts/global_rng_gate.py`, docs gate's 25th check — membership, LINE-FREE `path::call::count` keys, both directions, walk + predicate floors, masking, unconditional arms. Canary found an UNTRACKED planted file invisible (`git ls-files` IS the population) — redone staged. arm_reach 22 killed / 2 EQUIVALENT. T3 LANDED same day at `6d120abeb` — `Rng::new()`/`Rng::default()` joined (147 sites / 55 files, all pinned); issue file removed.

## Issue 806 T6 CLOSED (2026-09-16, the M3 side) — t698_t5_kv_mean adjudicated arch-dependent with dual pins; kda grad-check floor was below its own noise

[Bench 806 addendum](.benchmarks/806_x86_64_execution_matrix.md). Passes on aarch64 (pin `23d0daab3f087159`); x86_64 measures `4d0b592740db9358` (band bits one ulp apart, `0x3e5f_d968` vs `0x3e5f_d970`; behaviour gates green both) → arch-conditional dual pins; stale `t698_t5_kv_mean_gates` row removed from `x86_64_matrix_expected.txt`. Also: `kda_backward_grad_check` floor × tolerance (1.25e-5) was below measured FD noise (2.8e-5) — floor 5e-4 → 2e-3, 7/7 both platforms. The local block_topk hunk was DROPPED for upstream `086dd9127` (also removed the remaining4 OOB read).

## Issue 807 (2026-09-16) CLOSED — `lthash`: incremental homomorphic multiset hash, the shared substrate for the Agave-mined commitment/state-hash proposals

Mined from Agave (`riir-clippy/.raw/agave @ c95d8706`, eprint 2019/227 LtHash) for riir-chain Proposal 010 D1 (commitment_root 31.6 ms @ N=1e5 → O(1), per `dispatch.rs` + Bench 028 #16) + riir-dapps Proposal 005 D1 (`kat:statehash`). Zero existing multiset-hash; katgpt-core is the only shared home. `crates/katgpt-core/src/lthash.rs`, OPT-IN `lthash` (const-generic lanes default 1024, wrapping add mod 2¹⁶, BLAKE3 checksum, domain-separated BLAKE3-XOF over a LENGTH-PREFIXED part list). [Bench 771](.benchmarks/771_lthash_goat.md): G1 10/10; G2 `replace` 31.6 ns vs 1000-member rebuild 33.8 µs ≈ 1069×; G4 zero-alloc. Lessons: `2 * N` sizing needs `generic_const_exprs` (fixed 1 KB XOF chunk instead); `LtHash::identity()` doesn't take the const default. Awaits first consumer; counts 610→611.

## First x86_64 execution of the katgpt-core/katgpt-types SIMD suites (2026-09-16) — 15 latent AVX2 bugs caught and fixed; Bench 800's execution-parity caveat resolved NEGATIVE then closed

[Bench 800 addendum](.benchmarks/800_bf16_simd_goat.md). 4090 (i7-13700K) @ `d1f9be27a`, `git archive`d to scratch. katgpt-types `--lib`: **139/139**, first x86_64 run incl. `simd_exp_sum_extreme_inputs_underflow_not_wrap` (riir-train Issue 549). katgpt-core `--features bf16_simd` with `+avx2` (without it the scalar fallback runs): **15 FAILED**, all never-executed AVX2 defects: (1) `f32_to_bf16_rne_avx2` dropped the final `t >> 16` and `_mm_packus_epi32` saturated; (2) `dequant_via_lut_avx2` + `dequant_dot_via_lut_avx2` used SSE4.1 `_mm_cvtepu8_epi32` instead of `_mm256_cvtepu8_epi32` (lanes 4–7 gathered `lut[0]`; 12 cascading failures). Fixed, cross-checked via `cargo check --target x86_64-unknown-linux-musl` +avx2: **2069/0** (was 2054/15). Lessons: a parity caveat resting on shared-algorithm argument is a conjecture until run; record `target_feature` flags with every run.

## Issue 805 (2026-09-16) CLOSED — `numbering_gate.py --help` printed ten "remove the row" lines about a repo it could not read

`--help` was read as a repo PATH; `tracked_paths` turns git failure into an EMPTY set, so every pin read STALE with destructive `remove the row` remedies first (Issue 795: a pin row may be the only record). Repair: `unmeasurable(repo)` → exit **2**, via toplevel EQUALITY (`git -C` walks UP). ⛔ Census of 14 repo-path instruments: `orphaned_attr_gate.py` (a CHECK) printed `✓ PASSED … in nonexistent-repo` at exit 0 — fixed differently since `tracked_files` legitimately falls back to a walk: not-a-directory → UNSEEN, 0 `.rs` → UNSEEN. `unmeasured()` EXTRACTED so it can be armed. `arm_reach_gate` found 4 survivors hand perturbation missed; final 25 modules / 691 mutants / 465 killed / 38 pinned / 0 UNREACHED / 0 NO-ARM / 0 BASELINE. Landed `d1f9be27`. Issue file removed.

## Issue 804 (2026-09-16) CLOSED — 28 instruments crashed when run the way AGENTS.md says to run them; the cross-repo axis is seven repos, not one

On a cp874 console `print()` of `✓`/`✗` raises `UnicodeEncodeError` with NO verdict; `docs_gate.sh`'s `PYTHONIOENCODING` only covers gate runs, not the DIRECT runs AGENTS.md documents. ⛔ Cost is UNREAD FINDINGS: `restatement_drift_sweep.py` (4 repos / 255 theorems) was *unlooked at* while the family read green. Repair: `scripts/console_safe.py` (`errors="backslashreplace"`), 28 callers, 42 inline defences left (gate credits both), `console_encoding_gate.py` joined CHECKS.

Cross-repo half: Issue 789 T4 says **re-measure before answering**. Taken (`bc98dc6b`): 16 repos, 159 scripts, 144 in population, **71 undefended across SEVEN repos** (riir-train 53/53, riir-clippy 5/5, riir-ai 4/5, mmorpg-remake 4/4, mmorpg-editor 3/3, riir-dapps 1/1, riir-mmorpg-examples 1/1) vs katgpt-rs 0/72. ⚠ cp874 is this box's property and riir-train's rows are plan-scoped over-capture, so `console_encoding_drift_sweep.py` ratchets the DERIVATIVE (Issue 787 T6), katgpt-rs at `max_undefended = 0` and `min_population == console_encoding_gate.MIN_POPULATION`. Issue file removed.

## Issue 803 (2026-09-16) CLOSED — the off-macOS partial gate printed the SAME final line as a full pass

`develop` was RED under AGENTS.md's whole-repo command — 24 × `error[E0560]` (a `katgpt-core` field deleted with 24 live sites in root `tests/`/`benches/`) + 4 `-D` lint errors — for ten hours while `test_gate.sh` and `wasm32_gate` were green. `full_gate.sh --allow-partial-platform` would have read it but printed a final line byte-identical to a full pass (the deferral rode a Layer-2 line far above). Repaired: final line carries the partial verdict and names unmeasured axes; documented in AGENTS.md. ⛑ Renumbered 799 → 803 pre-push by the Issue-796 rule (`dual_allocation_gate.py` INDEPENDENT, 5 vs 1 per Issue 724 T2). Landed `3ceb541b` (T1+T2) and follow-up (T3+T4). Issue file removed.

## Issue 802 (2026-09-16) CLOSED — commitment-gap calibration rig: residue DEAD-BY-DOMINATION at micro scale; stability features (item 3) shipped earlier in the day

[Bench 802 calibration-rig record](.benchmarks/802_commitment_gap_calibration_rig.md), rig `../riir-ai/crates/riir-poc/benches/commitment_gap_calibration.rs`. Items 2/3/5 shipped earlier (`f23b2d81b`, `226be38d9`). Item 1: G1 REGIME-DEPENDENT (31–82× cascade-dominated, 2.4× when commits collapse into 1–2 waves); G2 PASS (0.2587 @ 1.76 fwd vs 0.2538 @ 2.47); G3 FAIL — no-gate one-forward baseline dominates (1.00 fwd @ 0.2937; +3.5 acc at half NFE). Mechanism: self-consistency labels blind to context contamination; off-policy calibration misses (p* 0.90 → 0.61). Item 4 horizon axis UNDEFINED at micro (cascade ≤ 2 waves). Revival needs all of: large would-miss mass, improving revisions, iterating baseline, DAgger-style calibration. Also: root `katgpt_rs::speculative` re-exports `StabilityTracker`/`TOPK_DRIFT_K`/`N_STABILITY_FEATURES`. Issue file removed.

## Issue 800 Arm C phase 1 (2026-09-16) — GraphStablePool<T> extracted: the common contract verified across 4 sites (a 4th found in-repo)

`877e06eb2` + [Bench 800-C](.benchmarks/800_graphstablepool_phase1.md). Contract ships FOUR times (radix_prefix / PagedKVCache / riir-gpu arenas + `BranchBank` in `branching/bank.rs`). Narrowest common form: index-stable slots, LIFO free list, append-only growth. INDEX stability is the pool's; payload-ADDRESS stability belongs to the stored type or pre-allocation discipline (Qwen38LaneSet allocate-once — killed the chunked-layout option). 6 contract tests, 2092/0, wasm32 clean. Re-points open, one repo per commit (katgpt-kv → katgpt-transformer → riir-gpu + optional BranchBank).

## Issue 801 (2026-09-16) CLOSED — T4 PoC: composition-quality REFUTED, disagreement-trace CONFIRMED; T5 routing: meld stays opt-in as a contradiction detector, Super-GOAT Q3 blocked-as-refuted, T2 audit stands

T1 transcribed; T2 audit `ee6993a77` (Research 560: 4 INADMISSIBLE / 1 PARTIAL / 1 N.A.); T3 `43f15f7c8`+`f314d5006` (commutativity bitwise, non-associativity 512/512, λ⋆ ≤3.24e-7). **T4 `286687cd0` (riir-ai Bench 932): REFUTED** — mean matches/beats every meld arm on same-multiset D_eff (margins ≤ +0.008 vs +0.05 bar; σ ∈ [0.25,2] × β ∈ [0.1,10] sweep never restores tanh ≥ law-8; separation ≈ 0.5); mean's theorem-signature CONFIRMED (0.499–0.502 within, 0.590→0.625 across). **Positive: the λ̃ disagreement trace** — r = −0.791, contradiction AUC 0.976, precision@8 = 0.997, mean trace-free — shipped in `katgpt_core::meld` (opt-in). T5: keep primitive, Research 560 DOWNGRADED; F1/F2 lose composition premise, keep λ̃ half; F6 (riir-neuron-db Issue 618) unaffected. Revival path (not owed): amplitude-vector 2AFC + stochastic-noise readout. Issue file removed.

## Issue 800 Arms A+B closed (2026-09-16) — bf16 SIMD G2 FAIL-honest (autovec parity); slot-flip DECLINE (ties mpsc); JSD kernel lands (Issue 802 item 2)

- **Arm A** (`f314d5006`, [Bench 800](.benchmarks/800_bf16_simd_goat.md)): LLVM auto-vectorizes the scalar reference (1.00× widen/trunc, 1.19–1.23× RNE vs ≥4× gate). G1/G4 pass; opt-in, A4 wiring deferred, AVX2 compile-verified only.
- **Arm B** ([Bench 800-B](.benchmarks/800_slotflip_arm_b_decline.md)): slot-ownership beats serial +21–46% but TIES mpsc (+9.1/+0.5/+1.7/+0.0%, sign flips) → DECLINE, `async_qdq.rs` stays single-threaded. Protocol + staleness proof kept in `crates/katgpt-kv/tests/slot_flip_staleness.rs` (4/4 ×5).
- **Issue 802 item 2** ([Bench 802](.benchmarks/802_jsd_topk_kernel.md)): NaN-safe bounded top-K JSD in `katgpt_core::jsd_topk` (opt-in) — disjoint → bitwise ln 2, identical → 0.0, zero-alloc, 2069/0; task-sketch sign error corrected. Prerequisite for 802 items 1 and 4 (open).

## Issue 801 T1–T3 (2026-09-16) — NAP audit + `meld` primitive: the census survives code-level scrutiny; the algebra holds, the quality ladder did NOT transfer (T4 is the sole adjudicator)

Research 560 addendum `ee6993a77` (T2: 6 composers vs 5 NAP invariants + 3 theorem screens; 4 INADMISSIBLE / 1 PARTIAL (tpr) / 1 N.A. (wedge `retrieve_diverse`); `frozen_attractor` and `retrieve_diverse` live in riir-neuron-db; the gauge composition LAW is upstream at `katgpt-sparse/src/sparse_task_vector.rs:318-396`, riir-engine's `GaugeInvariantComposer` a thin bridge). `meld` `43f15f7c8` + wiring `f314d5006` (T3, opt-in): closed-form soft-min λ⋆ (disc = 4(4−t²)), BIT-EXACT commutativity via canonical (hi,lo) ordering, normalized-Hadamard W, tanh/DN/law-8 arms; commutativity 4800 cases, non-associativity 512/512, λ⋆ ≤3.24e-7, boundedness depth-64. **Bench 801 red flag:** paper's ladder did not reproduce at D=32 — tanh-meld 0.380 vs law-8 0.870 at d=5; mean-pooling doesn't collapse on nearest-centroid (blindness theorem is about the S₂/D_eff readout); no-W collapses harder. T4 spec sharpened; Super-GOAT Q3 blocked pending T4. Deviation: DN is `y/√(ε+Σy²)`, not `/D` (bounded by √D, fails depth-64).

## Issue 800 Arm A (2026-09-16) — bf16⇄f32 SIMD kernels: G2 FAIL-honest — the pufferlib kernel-shape premise does not exist on M3+rustc

`f314d5006` + [Bench 800](.benchmarks/800_bf16_simd_goat.md). NEON/AVX2/scalar bf16⇄f32 kernels, RNE bit-exact vs `half` incl. NaN class (sNaN-payload-1 special-cased; `|0x0040` matched), truncation opt-in, `into_buf` zero-alloc; G1 exhaustive + 2²⁰ oracle + NEON-vs-scalar parity. **G2 FAIL: 1.00× widen/trunc, 1.19–1.23× RNE vs ≥4×** — LLVM auto-vectorizes the scalar loops into identical NEON. `bf16_simd` stays opt-in; **A4 deferred**. Kept for G1 correctness + ISA guarantee (RNE+NaN is where autovec trails). AVX2 compile-verified only. Arms B and C open.

## Issue 799 (2026-09-16) — bevy_ecs 0.15→0.19 bump landed: the arenas are load-bearing evidence infrastructure, bevy_ecs is a schedule-free utility layer

The ARENA is load-bearing (kernel_blend / binned_blend GOAT evidence, Bench 432 mean delta +78.5, CI [+26.3, +130.8], from bomber tournaments); bevy_ecs is not deeply so (surface = `World` + `query{,_filtered}` + `Messages` drain + derives; ≈23.5K LOC to rewrite for no gain). DECISION: bump 0.15→0.19.1, landed `e0f02ab93`.

Migration: `Events<E>`→`Messages<E>`, `#[derive(Event)]`→`#[derive(Message)]`, `World::send_event`→`write_message`, `Entity::from_raw(u32)`→`from_raw_u32(u32).unwrap()`. uuid 1.12→1.26 pulls **getrandom 0.4** (third wasm pin `getrandom_04`; `rng-getrandom` on katgpt-spectral's uuid).

Verified: wasm32 `--features bomber` 1m17s / `bomber-wasm` 24.6s (stale `secure_vessel` mention fixed); clippy clean; tests 353 / 203 / 289 / 10+1+1+5 pass. BOUNDARY.md §May depend on names bevy_ecs (optional-only).

GOAT **Bench 799** (`.benchmarks/799_bevy_ecs_019_bump_arena_goat.md`): G1 byte-identity PASS (kill attribution 96.4%, ScoreBoards, bomb counts); G2 PASS with documented cost — full-game harness ~2× slower (median 366→728 µs; re-run 410/414 vs 749/777), pure compute unchanged or faster; floors clear ≥10×.

## Issue 798 (2026-09-15) — a tracked landing record claimed a sibling-repo repair that was never committed

Two `scripts/` files (`cb03c8dc`) recorded cross-repo repairs as landed; neither existed, both sweeps RED throughout. `toolchain_override_drift_floors.txt` claimed "all five markers are in": **2 of 5** existed (both katgpt-rs), sweep `drift 1 · unresolved 1 · ✗ FAILED`. `pipefail_discard_expected.txt` **dropped a row** for riir-chain `teardown.sh:24`'s nonexistent tail (`✗ UNPINNED`) — the unrecoverable direction. Sibling edits were run green in the worktree and never committed: **Issue 797's class via a PROSE record**; 797's advisory (`41ecdcbd`) named exactly those repos on the repair run.

Repaired: riir-ai `194cdc9b5` + riir-game-sdk `61f11e7` (deliberate markers — scripts bake `FROM rust:1.95.0-bookworm`), riir-chain `5f814a2` (marker + `|| true` tail). Both sweeps PASS.

**No new instrument** — the sweep IS the verification. Rule: *a cross-repo repair is landed only when COMMITTED in the sibling, and the record must cite the sibling commit* (AGENTS.md § Before committing in a shared worktree).

⚠ **Landed TWICE concurrently;** discovered at push (non-fast-forward in three repos); duplicates dropped for the remote's — `reset --hard origin/develop` in the clean repos, `reset --mixed HEAD~1` + single-file `checkout` in riir-ai, leaving the other session's six dirty files and six commits untouched.

⛔ **This session then committed the defect it filed:** `b592a213` cited three local SHAs dropped minutes later. *Cite the sibling commit AND check it resolves* (`git -C ../<repo> cat-file -e`).

**T2 — worktree behind ORIGIN.** riir-ai 109 commits behind (14 in population) made the sweep RED on a fixed defect — the **mirror of MASKED**, invisible to 797's HEAD comparison. `behind_origin()` in `worktree_state.py`, wired into `sweep_advisory()` (all 18 sweeps, zero call-site changes); four answers (None / (0,0) / (n,0) / (n,k>0)); three-dot `HEAD...ref` so AHEAD isn't stale; sweep STAYS RED. Shares `_match_count` with `dirty_in_scope` (a pathspec differs on bare `Dockerfile`).

**T3 — hand-typed `36 assertion(s)` was 40 at the parent.** `n_assertions()` derives it (50), over `*_arms` only, returns 0 rather than raising.

⚠ Non-finding: pipefail PASS "every pinned row firing" beside `50 FINDING · 51 pinned` is correct (`pipefail_discard_drift_sweep.py:425-431` skips absent `riir-deployer`'s one row).

## riir-train Issue 549 fixed in `490b662e` (2026-09-15, M3 + 4090 session) — avx2_exp_sum_inplace: the one exp kernel missing the n-clamp

The fused exp+sum kernel behind every `softmax` lacked the [−126, 127] n-clamp before `(n+127) << 23` (all other exp kernels have it). Below −87.3 nats the exponent WRAPS: **exp(−300) = 6.9e23** (4090 negative control); spread > ~87 nats garbles softmax and NaNs loss — AVX2 only.

Surfaced as riir-train Issue 511's "Windows seed-1000 collapse": loss → NaN, Δfull 0.0000 via `.max(1e-9)`. Post-fix game 0 Δfull 6.7941 (4090) vs 6.7940 (M3).

Missed because truth sweeps stayed clear of |x| > ~88. New `simd_exp_sum_extreme_inputs_underflow_not_wrap` pins all three paths (32-wide, 8-wide, scalar tail): underflow ~0, positive saturation, fused/unfused parity, denominator invariant. 105/105 aarch64 and AVX2, pre-fix RED as control. Clamp is two ALU ops, hidden. Blast radius: `katgpt-types::math::softmax`/`softmax_scaled` + direct `simd_exp_sum_inplace` users on AVX2; fixed upstream.

## Issue 779 T1+T2 (2026-09-15, M3 session) resolved — subspace_intervention promoted + the FUNCATTN spectral arm POSITIVE (Bench 766)

`katgpt_core::subspace_intervention` (`subspace_intervention = ["subspace_phase_gate"]`, OPT-IN): Issue-778 POC promoted — ridge probe, frozen-head eval, basis projection, random control, `basis_similarity`, `three_arm_eval` / `three_arm_eval_on_basis`, `affinity_sweep`. Zero new deps, zero-alloc eval.

**Real POC bug caught:** 778's "ridge" collapsed to class-sum `W = XᵀY` (`(G⁺)⁻¹·G⁺·M = M`); shipped version solves `Σⱼ vⱼ·(vⱼᵀM)/σⱼ` (w = ±0.5 exact). Trap: `labels = i%C` at even C aliased the split to chance, and the projection-identity gate still held.

**T2 — `spectral_pre_rotate` deferred eval POSITIVE:** real `calibrate_eigenbasis` — eigen-aligned 0.802 vs random 0.354 at k=2, and > full 0.656 (projection DENOISES). Bench 766; composition as a katgpt-attn test. T3 deferred (cross-repo real-model run).

Gates: 7 module tests, release+alloc_tracking clean, default 2060 unchanged, clippy clean, --all-features clean.

## Issue 779 T3 (2026-09-16, M3 session) resolved — real-bank affinity: saturated plateau, NO re-pin; three-arm POSITIVE on real tensors (Bench 767)

riir-ai `future_probe_bank_capture` (`aa11cb162`) over gemma-2-2b-it f16: 384 prompts (6 classes × 8 shells × 8 topics, shells 6+7 held out), all-26-layer last-token residuals, label audit passes. **G0 measured:** bank regenerated with BLAKE3 IDENTICAL `99edecca…8b81`. Capture 1581 s (4.12 s/prompt; the earlier 2053 s was under two sibling loads).

**Affinity: NEGATIVE for re-pin.** CEILING PLATEAU — L02–L25 all 1.000 (floor 0.167) at every λ_scale ∈ {0.003, 0.01, 0.03, 0.1}; L00 0.990–1.000, L01 0.927–0.969; "best layer 25" / "peak L00" are tie-break artifacts. **FutureBehaviorProbe needs no re-pin; the terminal layer default is fine.**

**Three-arm: POSITIVE** (R557 M2 real-tensor half). L25: top-4 probe-SVD dims = full (1.000) vs random-4 0.146 (6.8×), 0.062 at k=6 (16×); projecting top-4 OUT collapses 0.344 → 0.000 (k=6 residual 0.000 is the degenerate 0-dim complement; identity holds to 1e-6).

Protocol discriminates both directions (778 planted peaks, 767 saturation). Revival needs a corpus where layers differ. Issue file removed.

## Issue 782 (2026-09-15, M3 session) resolved — slt_sweep: the noise-sweep λ̂ estimator (781 T4), GOAT G1–G4 ALL PASS, promoted default-on

`katgpt_core::slt::sweep` (`slt_sweep = ["slt"]`, **default-on**; Bench 765): frozen-weight λ̂ instrument — deterministic Gaussian stream (xorshift64* + Box–Muller), per-scale ladder, median; caller-owned `NoiseSweepScratch` zero-alloc; `1 + K·m` evals (6145 default) — freeze-seam only.

**v1 → v2:** windowed mirrored-Hill is unbiased only for exact power laws, which no polynomial loss has under the Gaussian measure; v1 was −28…−37%. v2 fits the log-log CDF slope over an order-statistic ladder — the shell-cancelling volume-codimension form (Murfet et al. 2020 eq. 4.3) (χ²₄: ~2% vs v1 −17%).

**Geometry:** TUBE/CONE sublevel sets near-unbiased (RRR −1.1%); ISOLATED regular minima −18…−19% at d ≥ 4 (+1…+3% d ≤ 2), ranking preserved. Feasible domain λ ≪ d/2; large-λ stays riir-train Plan 404 (SGLD).

**ReLU toy boundary:** paper cell (H=5, m=3, 32² quadrature) measures LOCAL λ̂ ≈ 2.77 (2.7710 at 9× quadrature, 2.91 at 16× draws) ≪ d/2 = 10.5, but not the global SGLD 0.526 — recorded as the instrument boundary in Bench 765. G3 bit-identical under seed; G4 0 allocs release; default suite 2053 → 2060; clippy + --all-features clean; docs_gate 17/17 after the 604/203 count bumps.

## Issue 781 (2026-09-15, M3 session) resolved in `580bda30` — slt: the RLCT λ + WBIC selection primitive, GOAT G1–G4 + floor ALL PASS, promoted default-on

`katgpt_core::slt` (feature `slt = []`, **default-on since landing**; Bench 764): six closed-form singular-learning-theory selection functions — `rlct_reduced_rank` λ(r) = r(a+b−r)/2 (Aoyagi–Watanabe 2005; exactly LoRA structure), `wbic` nL + λ·log n, `free_energy` with the (m−1)·loglog n multiplicity term, `bayes_gap` λ/n (the BAYES-predictive gap; a point fit realizes C/n, C = 2λ — the module doc makes the distinction load-bearing), `sigmoid_wbic_weight` σ(−ΔWBIC/τ) (sigmoid-native; K-way mixes compose Bradley-Terry products, never softmax), and `bic_overpenalty_nats` (r²/2·ln n — the gauge orbit naive parameter counting over-charges, present at EVERY rank incl. full).

**GOAT (Bench 764):** G1 planted-rank recovery (a=b=8, r*=6, n=2000, seeded) — WBIC picks 6, raw loss picks 8 (r_max, monotone failure), naive BIC picks 5 (realized gain ≈24 nats inside the (Δλ≈19, Δd/2≈30)·ln n window — over-penalization measured). **Loss convention:** per-SAMPLE nats (Σ_dims), not per-dim averages — the first draft averaged, shrinking gains by 1/a=8 and collapsing all margins. UQ floor gate: bayes CRPS/s 0.598 vs incumbent d/2n floor 0.604 vs constants 0.632+ at equal 0.972 coverage, measured on the WBIC-mixture predictor's realized gap; **thin ~1% margin recorded honestly** (gauge over-count small vs λ at a=b=8; widens with r/k*). G2 sub-µs O(k); G3 default count 2041→2053 (test_gate floor raised in-commit); G4 0 allocs under `--release --features slt,alloc_tracking`; `--all-features` clean; docs_gate 17/17 after 603-total/202-default count bumps (README ×2 + examples/README + 'and 117 more' → 118).

**T0 novelty (noise-sweep λ̂ estimator): KEEP with caveat** — searches (2026-09-15) + LLC survey (Emergent Mind 2025-10-15) find only SGLD/tempered-posterior (arXiv:2308.12108, 2402.03698, 2507.21449), exact-algebraic 2-D (2608.20183), linear-response (2605.07970); no Gaussian-perturbation V(t) power-law route. But λ̂ = m/Σ ln(u_max/uⱼ) is the classical Hill estimator (1975) — novelty is the APPLICATION. T4 (estimator + calibration ladder: quadratic bowl ⇒ d/2, planted RRR, ReLU toy ≈0.53) deferred as its own unit. T6 consumers filed: riir-ai (freeze/thaw WBIC tie-break + sigmoid mixture weights), riir-neuron-db (free-energy cross-n ledger in Raven/δ-Mem merge/keep ranking).

Anti-Laplace rule in the module doc (R558 §5): no Hessian/curvature generalization prediction — measured ~10³× the true λ. λ is a freeze/consolidation-seam scalar, never per-tick.

## Issue 775 (2026-09-14, M3 session) resolved in `3a59abe1` — dual_wave: the PC-ALM dual accumulator + closed-form rate laws (core) + the ballistic DEC wave kernel (dec), GOAT ALL PASS, opt-in

Research 554 (PC-ALM, arXiv:2605.31022) → opt-in `dual_wave` in BOTH katgpt-core and katgpt-dec. Core `dual`: accumulator/shift/credit/energy (T1), Jury setters + regime classifier with α-independent annulus (T2), arrival laws t_infl = L/√(αη) / alpha_reach / budget_ticks = 2L (T3), exact-adjoint readout λ → −δ (T8); rates from power iteration on the STACKED constraint operator's AᵀA. dec `wave_kernel`: interleaved (h, λ) recurrence on CochainField pairs (T5) + hodge_triage (T9). α=0 bit-identical to incumbent diffusion (T4, unit-pinned).

Lessons (in the bench doc): (1) per-LAYER Jury rates blow up the coupled chain — bound applies to the stacked operator; (2) Gershgorin (1+σ̂)² overshrinks η ~2× — power-iterate AᵀA itself; (3) power-iteration estimate is ‖AᵀAv‖, NOT its square (dense truth 3.28 vs buggy 10.7). Physics: settling is low-mode-limited (~L² worst case; Jury caps ηρσ²_max < 2, settling needs ηρσ₁²T ≳ 6), so T=2L holds at L ≤ 8 — the paper's own "finite-T misaligns" limit; the convergence-detected protocol (cosine 0.96–1.0 every layer, 95–264 ticks) is the honest gate. GOAT (Bench 763): G2 reach wave 18/97/212 ticks at L=16/64/128 (linear) vs heat 44/954/4687 (quadratic, ×9 growth — Eq-23 law); G4 2.38 µs @ K=100, 29.9 µs @ K=1024, 0 allocs; G1-adjoint cosine ≥ 0.9 everywhere. NOT promoted (R554 Q3: game zone hierarchies are L≈4 — modest gain; closed-form laws are the durable value). Counts 601→602. Record: Issue 775 file + `.benchmarks/763_dual_wave_goat.md`; hash referenced in R554.

## Issue 777 (2026-09-14, M3 session) resolved in `7e2a2638` — modality_additive belief kernel: FLYNN's linear-sensory-integration property distilled, measured, GOAT-passed, promoted same-day

Research 556 (FLYNN, arXiv:2607.00025) predicted `leaky_step` (divisive `1/total` gain + `−0.5·total` centering) breaks modality superposition. PoC `modality_superposition_bench` CONFIRMED it, refuting two predictions (note §7): the failure is RATIO-flavored (|Σ singles|/|full| = 5.17; cosine alone false-passes at 0.97), and `AttractorKernel` PASSES in the near-linear regime — so P1 is necessary-not-sufficient (additivity ≠ R304's stability axis; pairs WITH G2.1). T2 `evolve_belief_additive` (per-kind drive `2σ(η·k)−1`, per-kind retention α, no cross-modality terms) measures EXACT superposition (cos/ratio/worst-pair 1.0, max|Δdim| 0.0, predictability err 0) at 26.8 ns/tick — G1–G4 PASS, promoted to katgpt-sense `default`. Hygiene: unsink'd timing loop DCEs to 0.0 ns (black_box); single-run Instant on ~20 ns kernels flips (best-of-5 min); divisive kernels saturate to ±1 at large T (T=16 pre-saturation). G2: a 2× comparative gate is ill-posed for a COEXISTING method — gate absolute budget (≤50 ns D=8), report the comparison. Counts 599→601 / 200→201 (docs gate 17/17). T5 (connectome-vs-rewire on lif_graph) deferred data-blocked: wiring is synthetic (FlyWire licensing, riir-ai R379), so it can't attribute. Record: `.research/556` §7; issue file removed.
## Issue 789 (2026-09-14) — a gate whose own failure path is asserted by nothing: CLOSED

⚠ Heading uses `## Issue NNN (date) — …` rather than `## Issue NNN — …: CLOSED (date)`, which `heading_allocated()` cannot read (Issue 781's style gap: katgpt-rs scores 7 of 29).

### How it was found

Issue 775 landed `platform_dead_code_floor_gate.py` with canary arms over the gate's own pin arithmetic "which the classifier's self-test cannot reach" — a rule landed in one gate, never generalised: the **sixth** instance (Issues 777, 778, 793, 782, 783). Found by asking one level up from Issue 787: "is every gate's own verdict validated?"

⛔ **The first census over-reported:** grepping `--canary` / `--prove-fires` / `--self-test` flagged **nine of twenty** CHECKS bare; three **delegate** to their classifier (`percentile_floor_gate` → `percentile_index_audit.selftest`, `cfg_row_implication_gate`, `trap_sentinel_gate`) — the correct DRY answer. A census over one representation is blind to what it omits (787's lesson, reproduced in ten minutes).

### The measurement

By AST, crediting `selftest` / `self_test` / `canary` / `gate_selftest` / `prove_fires`, local or imported:

| | count |
|---|---|
| invokes its own arm | 10 |
| invokes a **delegated** arm | 3 |
| invokes both (775's shape) | 1 |
| **invokes NOTHING** | **6** |

`issue_citation_gate` 750 · `cargo_comment_audit` 480 · `skill_repo_set_gate` 306 · `count_features` 261 · `docs_gate_checks_sync` 141 · `markdown_fence_gate` 112 = **2,050 lines** of per-push logic whose failure path never executed. Two of the three checks `docs_gate.sh` was written for were RED on `develop` when it landed.

### T1 — the shared fence parser was blind to half of CommonMark

`skill_repo_set_gate.fenced_blocks` is read by **three** per-push gates (Issue 755 DRY call) and no test touched it; its first canary had been swallowed by the mis-phasing bug and never replaced. `fence_run()` counted **backticks only**; CommonMark allows backtick **or tilde**, non-interoperating. Ten arms: **two failed**.

| arm | expected | got |
|---|---|---|
| `~~~` / `x` / `~~~` | `(1,3)` | `[]` — invisible |
| `~~~` / ` ``` ` / `~~~` | `(1,3)` | `[(2,-3)]` — phantom unterminated |

Damage both ways: **silent** (tilde blocks never scanned) and **loud at the wrong address**. **Exposure LATENT: 0 tilde-fence lines over 5116 tracked `.md` / 16 repos** — forbidden anyway, as the gate claimed "CommonMark-ish" on exactly that axis.

`selftest()`: 19 parser arms (both families), 11 over `scan()`'s detector arithmetic, and the Issue-765 partial-clone deferral both ways. Exits **2**: a mis-phasing scanner's failure is the absence of a verdict, not a finding.

⛔ **Two of nine perturbations red nothing, recorded where read:**

- `ch == open_ch` is **REDUNDANT today** (`.strip(open_ch)` already discriminates). **A line a canary cannot red is not doing the work you think.** Kept: independent CommonMark requirements, each load-bearing once the other loosens.
- a lookbehind arm aimed at the name's **right** side (trailing slash already does the work) — kept, labelled inert, joined by a discriminating nested-path arm.

T4's free half: `markdown_fence_drift_sweep.py` shares the parser; 0 unterminated over 5121 `.md` with tildes visible.

### T3 — the four remaining bare checks, and two more real defects

**`cargo_comment_audit`** — `WEAK_DEFAULT_RE`'s lookahead excluded ``default (`0.82L`)`` (in no manifest) while the live instance is ``default `(0.82L→0.45L)` `` (`cross_stage_relocation`, root `Cargo.toml`). Latent (rung 1 decides first). Fix covers both orders, **zero** reclassifications over **7,465** inline comments. ⚠ Not widened to any backtick: that moves **21** "Not in `default` directly; transitively enabled via `X`" comments to `unknown` — measured; their unread claim recorded as a separate gap. Pinned: the rung-2 paren guard is load-bearing only with rung 3.

**`count_features`** — `CLAIMS` hoisted, `outside_parens()` extracted; `SWEEP` arms use phrasings that once escaped ("999 tunable flags", "999 default features"). "Canaried at each widening" was TRUE but the canary was a person — a narrowing would be silent.

**`docs_gate_checks_sync`** — arms over both parsers + quantity extractor (addresses-are-not-quantities, four shapes); `expect_exit()` swallows `✗ INSTRUMENT` output so a clean run prints no fake failures.

**`issue_citation_gate`** — documented wrong **twice** toward absolution (752's `named != {}` read 45 rows clean; 754 refuted that census's blind `allocated()`). 39 arms aim at **suppressing** paths first: `is_qualified` owner-consistency + ORPHAN, `heading_allocated`'s three filters incl. Issue 781's two negatives, window/adjacent split, alias direction, list expansion at four separators, `fenced_lines` failing **safe**. Runs before the deferral branches — a deferral atop a broken classifier must be impossible.

### T2 — the mechanism: `scripts/check_validation_gate.py`

Predicate: **invokes an arm UNCONDITIONALLY**. `docs_gate.sh` runs checks with **no arguments**, so `population_sync_gate.py`'s eight arms (Issue 788, the day before) sat behind `--canary` and ran on **no push**; they cost **0.17s**. Now unconditional (`--canary` = verbose); perturbing its docstring count reds `main()` rc=2.

`ARM_NAMES` is the **permissive** direction — a widened set (add `main`) greens silently, so an arm asserts `main` is excluded. Floors: `MIN_CHECKS` (array parse), `MIN_ARMED` (AST resolution). `check_validation_expected.txt` **deliberately empty** (785's rule: "not written yet" is a backlog wearing a pin); reasonless rows refused, stale rows red.

⚠ **Not asserted:** arm quality. **Seven** arms here certified nothing until fixed (wrong anchor, no terminated fence, wrong lookbehind side, pre-sorted input). Not statically decidable.

`--prove-fires 6804d983` (the commit that FILED 789): **7** unarmed there (6 bare, 1 flag-gated), named. ~0.3s, opt-in, `scripts/` only.

### T4 — no sweep, and that is a measurement

**katgpt-rs is the only repo (of 16) with a `scripts/docs_gate.sh` CHECKS array** (riir-train 58 `scripts/*.py`, riir-ai 7, no array). A sweep would green over a population of ONE — why `ci_gate_coverage.py` is outside CHECKS. The generalising question ("findable?") is ratcheted by `instrument_reachability_drift_sweep.py`. **Don't add a sweep by symmetry; re-measure first.**

### Verification

docs gate **21/21** (CHECKS 20 → 21). New check ~**0.11s**, cheapest; fourth same-day CHECKS move. `markdown_fence_drift_sweep` 0 unterminated over 5121 `.md`, both postures. **53 perturbations** (30 + 14 + 9); all red after four inert arms re-aimed.

### The standing lesson, now recorded seven times

A rule landed in one instrument and never generalised (777, 778, 779, 782, 783, 789); a census over one representation is blind to what it omits (787, then 789's first pass). Grep the whole family and land one shared mechanism; ask which representation a census read.

## Issue 788 — the population-predicate registry was hand-maintained, and a careful reading missed two of ten: CLOSED (2026-09-14)

`population_sync_gate.py`'s `PREDICATES` tuple is DATA ("a one-line change here rather than an eighth silent divergence") — and nobody made the change: **seven registered while ten existed**, printing "7 predicates agree". The filing census counted NINE; the completeness check found the ninth and tenth (`docs_drift_sweep.derive_population`, `wasm32_surface_audit.derive_population`) the census missed via a name-shape test. A careful reading missed two of ten.

**Three real defects fell out:**

1. **The eighth was WRONG:** `len_derived_binding_audit.derive_repos` tested `(d / ".git").exists()` not `.is_dir()`, admitting worktree-shaped dirs — caught on first run by the synthetic `worktree-shaped` arm. Latent; Issue 786's measurements unchanged.
2. **Two were UNPARAMETERISED** (root from `__file__`), untestable by the synthetic CI half; both take optional `root` now.
3. **Real-workspace verdict coupled to unrelated failures** (`if not bad:` guarded "all N agree"); now a local flag.

**`SUBSET_PREDICATES`** is its own tuple: `restatement_theorem_audit.repos` adds `.proofs` (4 of 16) and would red as an equal; the exclusion was recorded nowhere. Subsets get a strict-subset assertion.

**The detector's boundary was wrong first** while its docstring claimed "exactly the nine": `def` mentioning `BOUNDARY.md` + `.git` reported **23** (`main()`/`selftest()` bodies, this gate's `build_synthetic()`). The discriminator is **directory iteration**. `ast` deliberately unused — a sibling's syntax error must not blind this gate.

Escape hatch `population-predicate: not a contract-repo walk` on the `def` line, used twice correctly (`restatement_drift_sweep.main`; this gate's `canary()`, whose fixtures embed predicate source — the Issue 787 limitation). 8 canary arms incl. one pinning the docstring's headline count. CHECKS stays 20; quantity words *seven* → *ten*.

## Issue 787 — a census reads the DOCUMENT, so an undocumented instrument is invisible to it: CLOSED (2026-09-14)

**Correction first:** `1a5b6571` bounded Issue 785's close-out to "every cross-repo class **whose verdict is walled at a small number** has both halves" — still false by one: `len_derived_binding_audit.py` (Issue 786). Both censuses enumerated audits **AGENTS.md documents**, and it didn't name that one. A census reading the document can't see what it omits — a blindness floor one level up, the DOCUMENTATION unfloored.

**Predicate is REACHABLE:** roots `AGENTS.md`, `scripts/docs_gate.sh`, `.github/workflows/*.yml`; closure follows script → script. Required: `all_ignored_target_audit.py`, `cfg_row_implication_audit.py`, `ci_test_execution_report.py` appear in no document but run per-push via documented instruments.

⛔ **`HISTORY.md` is not a root** — the archive isn't loaded into sessions; counting it would make the gate vacuous on 786.

**Measured 2026-09-14:** 63 tracked `scripts/*.py`, 13 roots, 56 reachable, **9 unreachable**. Two real instruments **wired into AGENTS.md** (the default): `list_unresolved_percentile_sites.py` (percentile section) and `citation_weight.py` (Numbering Discipline — on riir-ai's six duplicates by-NAME citations are 0-2 per side, TIED in four, while the `Plan 175` form carries 35-98 each). Seven pinned by MEMBERSHIP with reasons: five `kimi_ref/` files, a manual CoreML generator, `gguf_header_audit.py` (outside the family per `1a5b6571`).

**Known-answer:** `--prove-fires 18dbe980` — at the parent, the 786 audit was the TENTH unreachable; at the fix, reachable. **`min_roots`** guards the silent direction: a widened root set makes everything reachable and greens.

**Sweep changed the pin design:** workspace **95 unreachable of 152** over 16 repos; **riir-train 61 of 61** (plan-scoped one-offs like `plan341_band_pool.py`, `plan346_diversity_gate.py`, `t504_harvest.py`) — the predicate over-captures there. So the sweep is a **RATCHET** (`max_unreachable` at measured), constraining the derivative; membership stays for this repo's 7. ⚠ Not Issue 785's forbidden target (that is an ownerless *unanswered* backlog; this is *unfindable*, owned, locally actionable) and not `suite_membership_audit` (1,203 rows, report-only). Six repos have 0 scripts (floors vacuous, Issue 783's shape); `DOC_ROOTS` is a REFUSAL, not a floor.

**⛔ The closure is TEXTUAL, proven on itself three times:** any basename mention credits reachability (can't narrow — real invocations are literals, the root is prose). First staged run red on `scripts/kimi_ref/fla_stub.py` because a canary named it as fixture data; then the UNREACHABLE arm's literal did the same; then the comment explaining runtime assembly spelled it out. False *reachable* is the dangerous direction — if a row leaves the unreachable set without a wiring commit, check what started naming it.

Canaries: **9 on the gate, 8 on the sweep** (`--canary`), incl. basename-collision refusal and `bump()` perturbing by FIELD INDEX (the Issue 786 canary failure designed out). Gate cost **~0.24s**, cheapest. CHECKS 19 → **20**.

## Issue 786 — the `.len()`-derived binding audit had no verdict half, and the reason it went unnoticed is the finding: CLOSED (2026-09-14)

`scripts/len_derived_binding_audit.py` (983 lines, 16 repos, 8,694 tracked `.rs`, 52 `.len()`-deriving cube kernels, 164 bind sites, nine buckets) had no gate, no sweep, **no `AGENTS.md` entry** — the last such cross-repo instrument after `1a5b6571` bounded the two documented exceptions (`suite_membership_audit.py`, 1,203 unpinned rows; `gguf_header_audit.py`).

**Ninth instance** (Issues 777, 778, 793, 782, 783, 784, 785), the QUIETEST: 784/785 surfaced via stale public figures (784 46% stale); an undocumented instrument has no symptom — it just stops running. Hence a census. It had already gone blind once (Issue 777: filesystem walk crediting mmorpg-remaster's nested `mmorpg/` and riir-train `OUT_DIR`; migrated to `tracked_walk`; 11,132 → 8,694 `.rs`) — a floored sweep would have asserted that.

**Three measurements decided the pins:**

1. **Bimodal population:** riir-ai 43 kernels / 143 binds, riir-train 9 / 21, other 14 repos 0 / 0 — floors vacuous in 14 of 16 (Issue 783's shape, not Issue 784's). **No `TOTALS` row** (unlike `wasm32_surface_drift_floors.txt`): it would red on partial clones that `population_verdict()` should DEFER.
2. **Verdicts are cross-repo by construction** (HALF C resolves via WORKSPACE callers), so a partial clone can mis-measure a PRESENT repo — `DEFERRED` doesn't cover it. **7 of 251** caller refs cross-repo; **leave-one-out over 16 repos: 0 flips**. The sweep re-runs a TARGETED leave-one-out each time over the derived supplier set (riir-ai + riir-train, ~8s each).
3. **UNRESOLVED 118 of 164 stays unpinned** (785's rule; wasm32 walled its UNRESOLVED at 0 only because Issue 738 T1 ANSWERED the rows); HALF C can't reach it. Reported with reason, never folded — `suite_membership_audit` precedent.

**No `min_rs_files` column:** three sweeps already floor that identical walk (and disagree: katgpt-rs 1500/1400/1500, riir-ai 1500/1500/1800 vs 2415/2626 measured — harmless slack floors). Delegation ASSERTED: each pinned repo needs a non-zero row in `orphaned_attr_drift_floors.txt`, and an unreadable file is REFUSED, not `{}`.

**T1 enabling work:** classification extracted from `main()` as `classify_workspace(repos)`, floors lifted to `FLOOR_RS_FILES` / `FLOOR_KERNELS`; report byte-identical.

**12 canary arms** (`--canary`, opt-in since they re-enter `main()`): baseline, both parse floors, UNPINNED, `max_findings` wall (a bind relabelled `CAPACITY`), EYES membership both ways, EYES count within an address, delegation break, unreadable-delegation refusal (exit 2), empty-pins refusal (exit 2), cross-repo flip. The UNPINNED arm failed first — **its anchor was wrong** (`11\n` vs `11             0`), perturbing nothing (cf. Issue 775's `vendor/` arm). Marker-on DEFERS the four absent repos by name; marker-off reds UNSEEN.

Standing (2026-09-14, 16 of 20 repos): **0 joined findings · 4 EYES · 118 UNRESOLVED** over 8,694 `.rs` / 52 kernels / 164 binds — riir-ai's and riir-train's to adjudicate.

## The platform-dead_code class got an instrument — `scripts/platform_dead_code_audit.py`, and it was wrong on its first sweep (2026-09-14, 4090 session)

The NEON_U8 class (`ea4c2873`, below) was found by hand on an unowned lane. The classifier flags module-scope decls whose EVERY occurrence sits under a narrower platform cfg than the declaration's, composing cfg from item/block/blockless-statement attrs, file `#![cfg]`, and `mod foo;` gates resolved ACROSS FILES. 2415 files in 6 s here; 8694 `.rs` / 16 repos in 26 s. Rules: AGENTS.md §"An item can be dead on a platform NO lane compiles".

Validation: `--prove-fires ea4c2873` (`git archive` of `ea4c2873~1` + fix; 1 finding → 0 over 268 files / 3635 candidates); 24 self-test arms every invocation, classifier MISS exits **2**.

**Three things wrong first:**

1. **It INVENTED a finding** its header swore impossible: masking strings dropped Rust 2021 inline format args, so riir-ai's `SWEEP_COUNTS` (`println!("Sweep sizes: {SWEEP_COUNTS:?}")` ungated in `main`, macOS elsewhere) read dead. Two arms pin it.
2. **A `mod` row is not a rustc finding:** `katgpt-types/src/simd/mod.rs:49`'s `mod horizontal;` has all 15 refs x86_64-gated, yet wasm32 `cargo check` is silent — the module is EMPTY, not dead; an appended ungated `fn` warns on the **fn**. MOD-REF is its own bucket.
3. **An arm passing under its own perturbation certifies nothing:** neutering `vendored_p` red zero arms (no `.git` → filesystem branch, redundant `"vendor"` in `SKIP_DIRS`). Duplicate removed.

**Two real rows, compile-verified, fixed on riir-ai `develop`:** `riir-gpu` `note_ane_dispatch` (ungated at `ane_prefill/mod.rs:535`, call sites all `all(target_os = "macos", target_arch = "aarch64")`; warns on x86_64 Windows with `--features ane_prefill` — default is a green ZERO) and `riir-games-shared` `gen_u64_bytes` (only caller in a `target_pointer_width = "64"` arm; warns on `wasm32-unknown-unknown --features chacha20_rng`). Both gated to mirror callers; an adjacent `unused_mut` got conditional `cfg_attr(not(...), allow(unused_mut))`. Standing: **0 findings · 1 MOD-REF** over 8694 files / 3433 units / 119452 decls / 16 repos. The macOS+aarch64 arm isn't compiled here — argument, not measurement.

⚠ `docs_gate.sh` ran **17/17** on Windows for the first time (Python **3.14** `python3` shim fixing the four `tomllib` gates + `DOCS_GATE_PARTIAL_CLONE=1`): **1.81s CPU / 14.1s wall** vs documented 13.37s CPU. Don't revise: mechanism unmeasured (interpreter, or the shim's `sh -c` defeating `times`). Wall in range.

## The Windows all-features lane — NEON_U8 platform gate, first specimen of the platform-dead_code class (2026-09-14, 4090 session)

Idle Protocol B sweep (`cargo clippy --workspace --all-targets --all-features`, unowned: `full_gate` is macOS/aarch64, `wasm32_gate` wasm32). `NEON_U8` (`katgpt-pruners` `interval_pruner/simd.rs:27`) ungated, used only in aarch64-gated `neon_is_interval_closed` — dead on non-aarch64. Vintage `432cacf7` (2026-06-12); the `8914b79d` "--all-features backlog 141→0" ran where it's alive — host-scoped truth. Fixed `ea4c2873` (mirrors `AVX2_U8`; 166/166 lib tests on x86_64). Instrument lesson (riir-clippy snapshot): a `clippy::`-prefixed JSON grep reads FALSE all-clean — rustc codes (`dead_code`, `unused_*`) have no prefix; filter Windows hard-link cache noise by message, not unit tally. Recurred same-day in the zed fork (4 specimens, incl. a macOS-only Metal example and `KEYCHAIN_SERVICE`) — riir-clippy `.distill/001` P22 (`61bb0d54`).

## Post-riir-train-513 develop drift — the 09-12→09-14 touched-rows window audited green on the workstation (2026-09-14, M3 session)

riir-train Issue 513's T2 sweep measured this repo's 623 rows through 09-05..09-11; rows since drifted to 710 via develop landings the main-only CI never audits. T4's `required_features_touched_gate.py` over base `9b8cf60e7` (2026-09-12 00:00 +07) → HEAD, 191 commits → **25 selected rows** (katgpt-core 13 · root 7 · katgpt-attn 3 · katgpt-kv 1 · katgpt-backend 1) — **25/25 BUILDS at EXACT feature set · 0 FAIL · 0 NO-FEAT · 0 UNSEEN** (`/tmp/katgpt-rs-rf`, ~11 min at load ≈5-7; left warm). Narrow per the gate's NOTE (library-change-only rows stay the full sweep's). 09-11→09-12 sliver unmeasured; other repos' tails (30 rows / 7 repos) defer to new hardware with riir-ai's record.

## Issue 774 (2026-09-14, M3 session) resolved in `22e65be4` — the wasm32 surface audit's BY-DEP verdict: the row predicate was dep-blind, closed with a five-canary self-test

Filed/landed same day. The resolver credited only ROW evidence, so transitive builds read UNCOVERED: `-p riir-shader-showcase --target wasm32` compiles its path deps, yet riir-shader core (1 site) + effects (2) read UNCOVERED — compile-verified (`cargo check -p riir-shader-effects --target wasm32-unknown-unknown` exit 0, 30.3s).

Fix: fourth verdict `✓ by-dep`, never folded into NAMED (it dies by someone else's manifest edit). Credit: non-optional path deps in `[dependencies]` + target tables POSITIVELY naming wasm32; `workspace = true` via root `[workspace.dependencies]` (root-RELATIVE — first draft got it wrong, canary caught it); dev/build, optional, native-target, cross-repo credit nothing. Seeds = named ∪ derived.

`--self-test`: five canaries in a throwaway repo sharing the one `verdict_for` — named / by-dep / uncovered (mmorpg-remake `.issues/010` lineage) / optional-not-credited / workspace-table-resolved.

Landing: **26 NAMED · 2 BY-DEP · 0 UNRESOLVED · 1 UNCOVERED** over 216 files / 29 packages / 20 repos (was 26/0/3; 2026-09-08's 23 NAMED / 191 files / 23 packages was stale). Remaining UNCOVERED: `mmorpg-poc-submodule` (mmorpg-remaster; CI-excluded, needs protoc, depended on by nothing) — the negative control; owner's call. Notes sorted at print (PYTHONHASHSEED churn masked verdict flips).

## Issue 748 (2026-09-14, M3 session) resolved — option (a): all three unwired Lean negative tests now run in their lean_proofs.yml CI jobs (~162s/main push)

Gap 2: three of four `proof_negative_test.sh` were invoked by NOTHING; only riir-chain ran both. Owner picked (a):

| repo | commit | marginal cost / main push |
|---|---|---|
| katgpt-rs | `3c97358c` | ~119s (Mathlib, reuses `.lake`) |
| riir-ai | `ecb21f3f7` | ~37s (Mathlib) |
| riir-neuron-db | `4a68575` | ~6s (Mathlib-free) |
| riir-chain | (already wired, Plan 016) | ~15s |

Each adds the script to BOTH `paths:` lists and a step after `proof_gate.sh` in the SAME job (reusing `.lake`). Stale step names fixed (riir-ai "16 theorems" → 22; riir-neuron-db "34" → 58) plus three stale sentinel comments. Workstation-validated: gate PASS (39 / 22 / 58), negative 8/8 / 15/15 / 17/17, clean rebuild, `.proofs` byte-clean.

Gap 1 (no `develop` lane) UNCHANGED BY DESIGN — owner's 2026-09-09 main-only Actions call; workstation runs cover develop; `workflow_dispatch` exists.

## Issue 773 (2026-09-14, M3 session) resolved — the 772-B2 removal's stale re-export: root lib E0432 under `flashar_consensus,plasma_path`, found by riir-ai's guard through the path dep

`3c3c52ce` (Issue 772 B2) deleted `ternary_fusion_gate` but missed the `plasma_path`-gated re-export (`src/speculative/flashar_consensus.rs:19`), compiled only under non-default `flashar_consensus` (`src/speculative/mod.rs:324`) — the double cfg hid it from the 772 wave's default lanes while riir-ai's guard Layer 1 (forwarding both features into the path dep) died E0432. No consumers anywhere. Filed `1d873f8c`, fixed `b7fcabd8` (3-line deletion). Sixth scoped-closeout instance this week — first found cross-repo by a DOWNSTREAM gate.

## Issue 771 (2026-09-14, M3 session) resolved — the radix-tree prefix KV cache primitive (RadixAttention index) shipped opt-in; G1–G4 ALL PASS, promotion deferred to the serving lane

The stack had PagedAttention (`PagedKVCache` refcounted pages, `fork`/`rollback` CoW), the single-stream prefix cache (`riir-gpu` `Qwen38PrefixCache`; riir-ai Bench 750 calls radix "equivalent for single-stream reuse"), and unwired `KvSegmentPool` — not the composition.

**Shipped** (Bench 762, `--release --features radix_prefix_cache`):

- `katgpt_kv::radix_prefix::RadixPrefixTree`: 16-token chunk spans, chunk-floor longest-prefix match (trailing partial chunk re-prefilled — no CoW needed), leaf-preferential lock-aware LRU, in-place splits keeping node ids valid. Tree owns page INDICES, never buffers — CUDA-graph address stability by construction.
- Pool half: 4 ungated `PagedKVCache` methods (`chunk_page_tables` / `retain_chunk_pages` / `release_chunk_pages` / `adopt_chunk_pages`). Refcount = live-seq holds + 1-while-indexed; locks are hit-rate only.
- G1 bit-identity via `to_bits` (filler yields NaN payloads where `!=` lies; small_target caught it), branch isolation (refcount exactly 3), pool stability. G2 hit-rate **2.45×** flat control at 50% budget (16 convs × 8 turns round-robin; sequential workload measured EQUAL — duplication only hurts under interleaving + pressure); match latency **9.8×** (0.21 vs 2.06 ms / 2,560 lookups). G4 0 allocs on match (warm scratch first).
- SGLang divergences documented: chunk-floor, no per-chunk hash filter (memcmp at 64 B/chunk), node-per-request, locks-on-head splits.

**STAYS OPT-IN** — no production consumer; every lane is single-stream (`riir-ai .research/034`). Consumer note (T5) in `riir-gpu`'s `qwen38_prefix_cache.rs`: consume this, don't re-derive.

**Follow-up (2026-09-14, 4090):** a racing duplicate implementation dropped per numbering discipline; its new tests adapted: `randomized_oracle_equivalence` (24 seeds × 40 inserts × 80 queries vs floor(lcp/page_tokens)) and `G2[width]` (78 → 186 ns/req at 8× nodes, ≤3× per-request; first draft compared raw totals and false-REDA; ~2.6× headroom over measured 2.4×).

## Issue 770 (2026-09-13, M3 session) resolved — the counter walker rebuilt per-commit; the 769 adjudication was partly an instrument artifact (verdict-review round 2)

Re-deriving each reset row against its commit's diff showed the 769 walker lineage-blind:

- **15 of 31 rows named FORWARD-stepping commits** — hunks ordered by DATE around one `current`, so after a real backward event the other lineage's `+1`s re-flagged. riir-chain's ×4 and riir-train Issue's ×2 phantom; riir-train Bench's 8 really 2.
- **"All 31 non-merge" true BY CONSTRUCTION** — `git log -p` doesn't diff merges, hiding merges taking the LOWER side.

Repair (`highwater_contiguity_audit.py::counter_history`): each HEAD-reachable commit's counter read at the commit and each parent (one `git cat-file --batch`; the blank separator must be consumed — fixtures caught the desync), judged against `max(parent values)`. Merge resets now visible (riir-ai Issue `627→614`, mmorpg-editor Plan `152→150` ×3). **27 resets** (katgpt-rs 5, riir-ai 9, riir-clippy 5, riir-train 2, mmorpg-editor 4, riir-mmorpg-examples 1, riir-shader 1; riir-chain 0). Floors 5→6 fields; `max_unbumped` (CHECKOUT state, ref-dependent) now REPORT-ONLY. HEAD-reachable, never `--all`. Inert selftest arm fixed (stub spans both arms). The sweep caught this session's own missed 769→770 highwater bump. Issue-768 verdict STANDS: 438 gaps + 27 resets over 73 counters (creation gaps `0→N` now counted).

## Issue 769 (2026-09-13, M3 session) resolved — the counter-reset class lands in the numbering sweep; all 31 measured resets adjudicated

*(Correction, same session — read with Issue 770: counts below are the DATE-ORDERED walker's; 15 of 31 phantoms, "all non-merge" true by construction; corrected 27 incl. merges. Class, pin plumbing and DRY walker import stand.)*

`numbering_drift_sweep.py` gains two classes over `.highwater` HISTORY (walker imported from `highwater_contiguity_audit.py`):

- **resets** — committed BACKWARD moves; re-climb re-spends numbers (`.issues/121` class). T2: all 31 NON-MERGE stale-lineage writebacks (katgpt-rs Bench `564→204`/`564→205`; riir-train Bench ×8; katgpt-rs Issue `577→25` from a base-24 worktree). max_dup proves no live duplicates. Ratchet-pinned.
- **unbumped** — worktree counter below committed max: mmorpg-editor ×3 (150<152, 191<194, 1<2) — REPORT-to-owner (read-only repo).

Pins 5→7 fields (`max_resets`, `max_unbumped`); the diverged-lineage fixture caught a REAL walker defect (`4→3` at walk 2 read as a climb); `base = max(current, old)` moved 17 rows (332→315 gaps).

## Issue 768 (2026-09-13, M3 session) resolved — the .highwater ownership witness REFUTED by measurement; highwater_contiguity_audit.py landed

Closed MEASURED-NEGATIVE. T3: can `.highwater` be a fourth `allocated()` witness ("n ≤ hw ⇒ allocated here")? `scripts/highwater_contiguity_audit.py` (report-only, selftested; static unwitnessed count + committed transition walk) says NO:

- **438 gaps + 27 resets over 73 counters** (Issue-770 walk; first landing said 332 + 31) — no major repo contiguous. katgpt-rs Issue: 19 gaps + `577→25`; riir-ai Issue: 38 gaps + 2 resets; riir-neuron-db Issue `33→589` (deliberate rebaseline); mmorpg-editor/Plan: 46 gaps; riir-shader Issue `11→9`. riir-auth's `.benchmarks` counter is COUNT-BASED by design — excluded.
- An over-claiming witness VALIDATES wrong addresses — the 766/4573af13 class inverted.
- A contiguous-suffix witness is unpredictable, keeps dual-allocation risk, and T1 found ZERO live rows it would change → **declined. Decline is a correct answer.**

Protection stays on the canonical `## Issue NNN (date)` heading `heading_allocated()` reads (Issue 754). Reset class as a sweep check filed as Issue 769.

## The all-features E0252 root-name collision — ooo_audit::AuditScratch aliased (2026-09-13, M3 idle sweep)

Idle `cargo clippy --workspace --all-targets --all-features`: E0252 — root re-exported `AuditScratch` twice (`latent_confounder_audit`, Bench 194; `direction_bank_audit`, `83518b30` 09-12). Collides only with both opt-ins on; single-feature validation and main-only CI missed it. Fixed `a0ca7d36`: `AuditScratch as OooAuditScratch` at root only; zero root-path consumers. Validation: default unaffected; `-p katgpt-core --features direction_bank_audit` clippy clean + 2051/2051 lib tests; workspace all-features clippy clean (only upstream `block v0.1.6` future-compat note).

## The Windows default-features workspace lane — bench_mtp_metal_batch_floor platform gate (2026-09-13, 4090 session)

Idle `cargo clippy --workspace --all-targets` at DEFAULT: 13×E0433 — the only Metal example without `target_os = "macos"` guards. Root `default` → `async_qdq_overlap` → `inference_router` → `gpu_inference` satisfies its `required-features` everywhere, but `metal` is macOS-gated. Heal commit `2cb97410` had logged it "pre-existing (proven at HEAD 00cfe345)". Fixed per `bench_439_*`: item cfg on all 22 items + loud `not(macos)` stub, rustfmt'd. Windows: `-p katgpt-backend --all-targets --features gpu_inference` clean; default workspace lane 0 warnings for the first time here.

## Issue 766 (2026-09-13, 4090 session) resolved — len_derived audit: caller tracer (HALF C) + two instrument defects found and fixed

Follow-up to riir-train Issue 515 (220 UNRESOLVED wrapper-param binds). Three additions to `scripts/len_derived_binding_audit.py`:

- **HALF A window fix:** the fixed 6000-char window bled past kernels (deltanet tree-verify kernels flagged off host-side `TreeVerifyPlan::from_parents_topo`). 22 of 74 kernels false positives; 89 of 253 binds phantom (incl. all 8 PERSISTENT rows 515-T2 triaged). Now brace-matched body; `min_kernels` 40 → 45 (52 measured).
- **HALF C caller tracer:** for path-form wrappers, every `Struct::fn::<T>(args)` caller collected (turbofish-tolerant, comment-stripped — inline `// [0..n]` broke the Split4 call) and (handle, length) classified: EXACT-UPSTREAM, TRIMMED-UPSTREAM, PERSISTENT-UPSTREAM (eyes list), CAPACITY-UPSTREAM. Clean needs ALL callers clean; bare/method-form stay UNRESOLVED. False-clean canaries pinned.
- **GUARD-ONLY:** a len-use only bounding thread indices is capacity-tolerant; 14 rows re-verdict.

Standing: **52 kernels, 164 binds** — 25 GUARDED, 14 GUARD-ONLY, 3 EXACT-UPSTREAM, 4 PERSISTENT-UPSTREAM BENIGN (exactly-sized gemma2 F16 + ternary allocations), **118 UNRESOLVED — the honest floor** (a work list). T4 N/A.

*Heading correction (2026-09-13, M3): first landed bare, then 4573af13 mis-"qualified" it to riir-ai. Witness: `e4792a4b` bumps `.issues/.highwater` 765→766 (issue file never committed — Issue 754 invisible class), so katgpt-rs owns 766; riir-ai's 766 is a different issue (cf. `Issue 665` in riir-clippy). riir-train's "katgpt-rs Issue 766 resolved the standing list" stands. Heading now in the `## Issue NNN (date)` form (Issue 754). Residual gap — `allocated()` ignores `.highwater` — filed as Issue 768.*

## WeightEpoch — the KV-cache weight-identity epoch (riir-ai Issue 938; Plan 025 contract) (`49f5d245`, 2026-09-13)

`LoraAdapter::weight_epoch() -> WeightEpoch` (katgpt-types `lora`): BLAKE3 over domain-separated serialization (tag `katgpt-lora-weight-epoch-v1`, rank/in_dim/out_dim/alpha + length-prefixed a/b). Identity, not an install counter — A→B→A may reuse cache; any byte difference is a new epoch. `WeightEpoch::none()` for no adapter. O(adapter) at swap, O(1) memcmp.

Motivation: RLT §5.4/App-C staleness law (riir-train `.research/453`; riir-ai Issue 938) — a KV entry is exact only under the same epoch. `LoraPair` documents its BY-DESIGN mixed-epoch acceptance; `examples/core_04_prefill.rs` Proof 3 prints both epochs at the switch. riir-ai consumes it at `CpuInferenceBackend` (refusing mixed-epoch reads) — riir-ai HISTORY §Issue-938.

## Modelless-first mandate — original section (incl. the canonical-failure story)

## Modelless-first mandate (the core principle)

**This repo ships modelless inference primitives.** No training, backprop, or gradient descent. Allowed runtime weight mutations:

1. **Freeze/thaw** — atomic, versioned, BLAKE3-checked snapshot swap.
2. **Raw/lora hot-swap** — a **deterministically constructed** LoRA overlay via `LoraPair { reader, writer }` (Plan 025).
3. **Latent-space updates** — direction projections, sigmoid gates, routing tables; latent state, NOT base weights.

### MANDATORY: exhaust modelless paths before deferring to riir-train

Before deferring anything as "needs training", check the three paths (research skill §3.5, `.agents/skills/research/SKILL.md`). **Systematic, characterizable biases are modelless-correctable candidates** — try a deterministic reader-LoRA or freeze-state correction first.

**Canonical failure — AC-Prefix G1 (Plan 313, 2026-06-24):** G1 was prematurely deferred to riir-train though the doubled-signal bias was systematic. Reverted; the modelless investigation (Issue 003, resolved-and-removed in `552b4632`) is in `.benchmarks/313_ac_prefix_modelless.md` (Path 2: `attends_dedup` removes the bias bit-identically to iterative-MLM on single-layer micro-GPT, 0.0 diff). `ac_prefix` re-promoted DEFAULT-ON; multi-layer equivalence a non-blocking riir-train follow-up.

## Boundary contract — original section

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative contract: owns / does not own (with homes), crate allowlist, cross-repo rule links, drift ledger. It wins over this file.

- **Domain test:** is this a **modelless inference primitive** with no riir dep (this repo is upstream of everything)? NO → file in another repo.
- **Read it before** adding a dep, crate, module, System impl, or vocabulary type.
- **Enforcement:** `../riir-ai/scripts/ci_boundary_contract.sh` fails on undeclared cross-repo deps, drift rows without open issues, and contract-vs-graph mismatch. Use the `boundary-guard` skill.
- **Violation?** File `.issues/NNN_boundary_*.md` FIRST, add the drift row, then fix; closing removes the row in the same commit.

## The full gate — original section (narratives)

### The full gate — none of the above is a whole-repo claim

Every command above is narrow in an **independent** axis. (The count is deliberately unwritten — this sentence said "three" for months while the table carried five.)

| Axis | Blind spot |
|---|---|
| `check` vs `clippy` | two `cargo heal` escape classes rejected by clippy typeck, accepted by `check` (E0689, E0631 in `redundant_closure`) |
| default vs `--all-features` | non-default gated code compiles to **nothing** |
| `-p <crate>` vs `--workspace` | *same default features*: the ROOT crate's defaults can switch on a crate's non-default feature once selected |
| no `--all-targets` | skips tests / benches / examples — where gated code lives |
| dev vs `--release` | `debug_assertions` always **ON**, so `#[cfg(debug_assertions)]` code only compiles where it works (`.docs/10_audits/debug_release_profile_axis.md`) |
| `--all-targets` vs **doc-tests** | doc-tests excluded. First full-workspace *execution* (2026-09-04): **8 crates' doctests never built**, 31 lines / 14 files still `use katgpt_rs::...` after the split, plus 3 defects (one example asserting values its formula can't produce). Only `cargo test --doc` (`.issues/723` Class F) |
| **compile vs EXECUTE** | above axes are compilation. **Scoped core EXECUTED weekly** (`test.yml` + `scripts/test_gate.sh`, Issue 718 T3(b), 2026-09-04: katgpt-rs + katgpt-core `--lib`, count floors, riir-train 507 shape) — 2 of 32 packages. 477 integration + 176 bench targets unexecuted (`.docs/10_audits/ci_compile_vs_execute_axis.md`; full `--all-features --release` PRICED: 11,542 CPU-s cold / 45.5 min / 497 ok + 45 FAILED over 39 targets; `--all-features` is not a supported TEST config — per-feature RNG/GOAT calibrations; `.benchmarks/701_full_workspace_execution_pricing.md`, six classes in `.issues/723`) |

**A green full gate claims only that the workspace *compiles*** under one feature set, platform, profile. An uninvoked assertion is *unknown*: including the 39 GOAT gates armed in Issue 713 T3 ("All 39 pass" was a one-off **workstation** `--release` run). Read green as "it built".

`-p` vs `--workspace`: `cargo test -p katgpt-backend --lib` clean while `--workspace --lib` failed (`gpu.rs` behind `katgpt-backend/gpu_inference`, via `katgpt-rs/default -> async_qdq_overlap -> inference_router -> gpu_inference`). Per-crate also *shrinks* coverage: four "0 tests" crates contributed 704 under `--workspace`.

The **fifth** row isn't closed by the command below (dev profile). 2026-09-03: `--release` gave **2 errors**; `cargo test --release -p katgpt-core --lib` didn't compile — the command `.docs/10_audits/cfg_gated_silent_zero_pass.md` T2b prescribes — because two `#[cfg(test)]` blocks imported `debug_assertions`-only `crate::alloc` counters. Fixed in `.docs/10_audits/debug_release_profile_axis.md` T1; axis is T2. Debug **manufactured** four false perf reds (713), **hid** a two-day release break (715), and the gate compiles debug code only where it works (716). **Neither profile is the safe default — the profile is part of the claim.**

So before claiming a repo-wide green, run:

```bash
cargo clippy --workspace --all-targets --all-features --keep-going -- -D clippy::needless_range_loop -D clippy::map_clone -D clippy::iter_cloned_collect -D clippy::identity_op -D clippy::bool_comparison -D clippy::manual_is_multiple_of -D clippy::collapsible_if -D clippy::map_all_any_identity -D clippy::unnecessary_cast -D clippy::manual_repeat_n -D clippy::question_mark -D clippy::empty_line_after_outer_attr -D clippy::unusual_byte_groupings -D unused_mut -D unused_parens
```

The `-D` list (Issue 701 R3b, 2026-09-03): mechanical lints healed to ZERO residual (67 → 13 findings; 13 judgement-class survivors stay warnings). Residual > 0 must NOT be added.

`--keep-going` is mandatory. The gate was **red on `develop` from at least 2cb97410 until `c284dbb2`** (5 broken targets) while narrower gates were green. AGAIN 2026-09-07: red from `c69e651d` (Issue 731 T1, `forward_looped` `residual_exit`) until `c571d5b9` — **32 errors** (28 × E0061 missing trailing param; 2 × E0308 + 2 × E0614 Issue-729 stragglers, fixed `26ba3519`). T1's "all 27 sites aligned" was TRUE, and 28 more lived in `#![cfg]`-gated all-features targets, textually identical — found by rustc. Corollary: an alignment completeness claim must state its POPULATION FRAME. The T3 bench also lacked a `[[test]]` row (SILENT-NOW +1, caught by `cfg_gated_floor_gate.py`); both gates green by `26ba3519`.

Don't run it by hand — `scripts/full_gate.sh` is the assertion (refuses to pass off macOS, checks this document quotes its command).

**The inverse, unenforced:** on macOS every `not(target_os = "macos")` backend drops out even with `--all-features`. 2026-09-03, from `riir-gpu/src/lib.rs` module **declarations**: **9 modules, 25,212 lines** (`qwen38_dense_cudarc` 8,599). `riir-ai` `6bf51b592` landed a non-compiling CUDA lib (`E0599`) that stood **7h45m**. Record: `riir-ai` `.issues/857`. Same shape as `.docs/10_audits/cfg_gated_silent_zero_pass.md` one axis over. **A platform is part of the claim.**

**Since 2026-09-04 the compile half is reachable from the M3:**

```bash
scripts/check_platform_gated_modules.sh ../riir-train riir-train-gpu numeric_drift_cuda
scripts/check_platform_gated_modules.sh --canary ../riir-train riir-train-gpu \
    crates/riir-train-gpu/src/numeric_drift_tap.rs numeric_drift_cuda
```

`cargo check` never links. Build-script blockers: `blake3` NEON C (`CARGO_FEATURE_NO_NEON=1`), `libsqlite3-sys` + `sentencepiece-sys` (**Android NDK clang**'s linux sysroot; macOS SDK's `sys/cdefs.h` errors). Header details: the shim's `--target` must come **after** `"$@"` (cc-rs injects its own); `-llog` for `__android_log_write`. Green = compile only. First use (riir-train `53538538`) typechecked two modules whose edits had been routed to "whoever next builds on the 4090". **`--canary` is not optional** — requires `E0425` from a planted call, else "Finished" can mean compiled-to-nothing.

`.github/workflows/full_gate.yml` declares weekly cron, dispatch, and a NARROW push/PR lane firing only on `scripts/full_gate.sh` / the workflow (2m17s; `**/*.rs` rejected — see preamble). Preamble also records `.issues/705`: first two CI runs passed over ZERO units (ANSI codes defeated `^`-anchored counters); closed 2026-09-02.

Until 2026-09-01 `schedule`/`workflow_dispatch` had **never fired**: they run only from the DEFAULT branch, which was `main` (frozen at v0.1.1, no workflows). Fixed by making `develop` default (`.issues/704`). A workflow file looks identical whether or not it can execute: `scripts/ci_gate_coverage.py` reports per-workflow fireable triggers (**dead** / **unmeasured** / **untracked** / **PR-only**), taking the workspace from 7 dead to 1.

"Can fire" ≠ "does fire": `riir-chain`, `riir-dao`, `riir-neuron-db` carried their Rust surface in dispatch-only files (main-only owner call; `push` inert as `main` has no copy). The report now multiplies coverage × reachability. RESOLVED 2026-09-02 (`.issues/706`): weekly `schedule` added (`riir-chain` `b4a9b6e7` Tue 04:13 UTC, `riir-neuron-db` `9d041d1` 04:29, `riir-dao` `9848811` 04:43, which now runs `scripts/ci_feature_guard.sh`). Same day `riir-neuron-db`'s standalone-dep gate was RED nine days stale (`29af2b0` changed the patch set to `katgpt-device-verify` without re-pinning `EXPECTED`; fixed `97e5161`).

**Issue 758 (2026-09-12, closed same day): the (release × default-features) cell was asserted by nothing** — Layer 6's `--all-features` SUPPLIES `alloc_tracking`. slice_tca's module-level `use crate::alloc::{…}` broke `cargo test --release -p katgpt-core --lib` (E0432) while Layer 6 stayed GREEN; found by Issue 757's release harness. Fixed with the FULL Issue-741 predicate `#[cfg(any(debug_assertions, feature = "alloc_tracking"))]` + in-fn `use`; option (b) (`slice_tca = […, "alloc_tracking"]`) REJECTED — would default `alloc_tracking` for every consumer, violating "MUST stay opt-in". Verified: release-default 2027 passed / dev 2035 (757 baseline) / `--release --features alloc_tracking` the test RUNS (1 passed). Lane: **full_gate Layer 6b** — test_gate population (katgpt-rs, katgpt-core, katgpt-dec@pca_global) at `cargo check --tests --release`, not `--workspace` (metal examples E0433 ×10+ off macOS; Layer 2's axis). Two-sided canary. Also fixed: all three `mktemp -t <prefix>` calls (Layers 3, 6, 6b) were BSD-only — GNU rejects X-less templates; now full-path-with-X's form.

## Docs gate — original section (descriptions + narratives)

### The docs gate — same discipline, opposite cadence

`scripts/docs_gate.sh` runs the manifest/doc/skill drift assertions;
`.github/workflows/docs_gate.yml` runs it **per-push** on ubuntu-latest — both
deliberately inverse to the full gate: no `cfg(target_os)` surface, ~11s not >13 min.

Was ~3s when written; `percentile_floor_gate.py` alone is **7.0s** (walks 2,330
`.rs` files) since 2026-09-03 — stale three days. Per check 2026-09-06: percentile
7.01s · bench_doc_audit 0.88 · cargo_comment_audit 0.86 · count_features 0.81 ·
orphaned_attr 0.62 · cfg_gated_floor 0.52 · rest ≤0.10s. Re-time before quoting.

Per-push scoped to **`main` only** (owner call 2026-09-03, was
`[main, develop]`): run `./scripts/docs_gate.sh` locally for develop work. The
same change fast-forwarded `main` to develop, because a push trigger reads the
workflow from the PUSHED ref — a `main` without the file is a dead trigger (the
pre-704 rot shape).

The `CHECKS` array is the list; **count not written here** (it said "the three"
one commit after the fourth landed). The original three existed before wiring and
**nothing invoked them**; two were red on false positives. Uninvoked = unknown,
not passing. Checks worth knowing:

- `skill_repo_set_gate.py` (Issue 703) fails on a `SKILL.md` command block that
  hand-types the repo set. Separates **vocabulary** (committed
  `scripts/repo_set.txt`, re-derived by every workstation run) from
  **population** (12 `SKILL.md` locally, 8 in CI), printing both — skipping in
  CI would be the vacuous green it catches. Opt-out: `<!-- repo-set-ok: <reason> -->`.
- `agents_repo_set_gate.py` pins §"Repo count" against `scripts/repo_set.txt` —
  membership FIRST, cardinality second. On 2026-09-03 the paragraph named retired
  `riir-armageddon` and omitted new `mmorpg-remake-unity` **while its count (19)
  stayed correct** — a count is not a checksum over a set. Gates ONE paragraph on
  purpose: a whole-repo version false-positives on history (boundary-guard's
  ledger *should* name the retired repo; the 227→225 edge delta is its two edges
  leaving). Both inputs committed → runs in CI. Parser regression exits **2**,
  not 1.

- `cfg_gated_floor_gate.py` (Issue 713 T4): the GATE over the report below,
  katgpt-rs-scoped, pins in `scripts/cfg_gated_floors.txt`. `max_load_bearing = 0`
  reds a new default-off-gated `*_goat.rs` with no `required-features` row.
  **Some pins are FLOORS** — a ceiling cannot fail once the instrument goes
  blind. First hazard: `docs_gate.yml`'s `paths` filter had no `.rs` glob, so the
  gate could not fire on the push it exists for.

  Second, the one to remember: `max_load_bearing = 0` is only as wide as
  `is_load_bearing`'s **vocabulary**. T4c (2026-09-03, `2272b262`) found it
  missing the `*_correctness`/`*_alloc_check`/`*_determinism`/`*_equivalence`/
  `*_floor`/`*_grad_check` dialect. Seven tokens added, each measured against all
  2,157 workspace target names (`budget`, `check`, `calibration` **rejected**);
  17 more load-bearing katgpt-rs targets appeared (8 `*_alloc_check` G4 budgets, a
  Report-the-Floor UQ gate). All 17 armed and RUN in release: 45/45 pass —
  *unverified*, not broken, as in
  `.docs/10_audits/alloc_gate_per_thread_counter.md` T3. Found sideways. Re-run
  the corpus token table when a new dialect appears.

  Third: the widening moved load-bearing ALL-IGNORED **3 → 5** while the
  pins-file header and `.docs/10_audits/cfg_gated_silent_zero_pass.md` item 7
  kept old numbers — nothing pinned that count. The count is not gateable
  (`#[ignore]` is legit for slow/hardware tests), but **a set is gateable where
  its cardinality is not**: `scripts/all_ignored_load_bearing.txt` pins the five
  paths by MEMBERSHIP with reasons from source; `check_membership` reds both ways
  (new arrival, same-size **swap** — own selftest case, emptied set); an empty
  allowlist is refused. All four directions canaried. Not in `REQUIRED_PINS` (not
  an integer).
- `population_sync_gate.py` asserts the **six** "which repos are contract repos"
  predicates agree — `cfg_gated_target_audit.derive_repos`,
  `numbering_drift_sweep.contract_repos`, `percentile_index_audit.repos`,
  `ci_gate_coverage.derive_repos`, `skill_repo_set_gate.derive_repos`,
  `suite_membership_audit.derive_repos` (2026-09-06: 16 repos, identical, =
  `repo_set.txt`; nothing had asserted it). One drifting predicate silently audits
  a different set with a confident green — already paid once (three instruments
  covering 7, 12, 15 of 18 repos). `docs_gate_paths_sync.py` one axis over.

  **Runs in CI**: tests the PREDICATE on a synthetic workspace, incl. a
  `worktree-shaped` entry whose `.git` is a FILE (must be a directory test).
  Real-workspace cross-check only with >1 repo, REPORTED either way; a broken
  predicate still reds in simulated single-checkout CI.

- `required_features_static_gate.py` (riir-train Issue 513): verdict half of the
  free static pass — a row naming a feature its package cannot enable reds the
  push. Gateable because it needs no compiler and is **never legitimate** (cargo
  skips such a target everywhere, `--all-features` and `cargo test --workspace`
  included → green zero forever). Pins `scripts/required_features_floors.txt`;
  `min_rows_scanned` is a **FLOOR**. Canaried three ways; the first found
  `parse_rows` shadowing the package's declared-feature set with the row's own
  list — gate structurally incapable of firing.
- `bench_doc_audit.py` runs a `selftest()` every invocation pinning tokenizer line
  shapes; without it a regex regression prints "0 mismatches" — how 26 riir-chain
  benchmark docs audited clean while unreadable
  (`.docs/10_audits/sibling_doc_drift_auditors.md`).
- `percentile_floor_gate.py`: GATE over `percentile_index_audit.py` (pins
  `scripts/percentile_floors.txt`), katgpt-rs-scoped. A new site landing on
  `n - 1` reds the push **before** it is quoted in `.benchmarks/` as a tail (input
  to promote/demote decisions). Imports the report's tokenizer, runs its
  `selftest()` first, exits **2** if untrustworthy. `min_sites_scanned` is a
  **FLOOR** (tokenizer regression → population ~0 → every ceiling passes).
  Canaried both directions (planted degenerate site → 1; floor above population → 1).

### The docs gate covers ONE repo — two more tiers cover the rest

Auditors accept a repo path but pointed nowhere else for months. Tiers, none
subsuming another:

| instrument | where | cadence | scope |
|---|---|---|---|
| `docs_gate.yml` / `docs_gate.sh` | CI + workstation | per-push | katgpt-rs only |
| `sibling_docs_drift.yml` | sibling CI (reusable) | caller's choice | one caller |
| `scripts/docs_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/numbering_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/required_features_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/percentile_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/cfg_gated_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/cfg_row_implication_drift_sweep.py` | workstation | on demand | every contract repo |

**Sixth (2026-09-06):** `scripts/cfg_row_implication_drift_sweep.py` +
`cfg_row_implication_drift_floors.txt` ask whether a `required-features` row,
resolved WITH package defaults, satisfies the target's leading `#![cfg]`. If not,
the row builds the target to **nothing** — yet the row is PRESENT, so
`cfg_gated_target_audit.py` counts it protected and
`required_features_build_audit.py` rightly reports BUILDS. Measured: **16 repos,
1,868 rows, 1,172 with leading `#![cfg]`, 1 EMPTY-AT-ROW, 2 UNRESOLVED**. Open:
riir-ai `self_advantage_hla_bench` (ratcheted, Issue 513 instance 7). Found
riir-train `bench_dflare_heterogeneous_kv` naming the weaker feature; fix took
`0 passed` → `1 passed` (`054a39a2`). `min_with_cfg` floor is load-bearing (`leading_inner_cfgs` returns `[]` for "no
cfg" and a missing file; an early cut skipped `tests/<name>/main.rs`). Its
katgpt-rs row vs the per-push pins **caught a desync on first run**.

**Fifth — the one with a backlog.** `scripts/cfg_gated_drift_sweep.py` +
`cfg_gated_drift_floors.txt`: ceilings are a **RATCHET at measured count**. Found
Issue 728's **12 load-bearing SILENT-NOW targets** across six repos (ten
sibling-owned), all auto-discovered, no `[[test]]` row, default-off gate →
`ok. 0 passed`. Measured: **16 repos, 2,942 targets, 1,765 `#![cfg]`-gated, 261
SILENT-NOW, 12 load-bearing**. Asserts **all four** katgpt-rs numbers vs
`cfg_gated_floors.txt`; selftest pins the classifier both ways (six widened tokens
fire, seven homonyms don't).

**Fourth, same day, over the percentile audit.** Holds the earned zero
(2026-09-03: **12 DEGENERATE sites** fixed same day by four sibling owners). `scripts/percentile_drift_sweep.py` +
`percentile_drift_floors.txt` gate all four classes at 0: **16 repos, 11,132
`.rs` files, 110 sites, 0/0/0/0**. `min_*` slack as in `percentile_floors.txt`
(repairs SHRINK sites). Shared `pia.tally()` (WEAK only when `asserted`, TRUNC-VAR
regardless). The 125 → 110 drop was **checked before pinning** (repair vs
blindness): instrument's last commit predates 125, **katgpt-rs invariant at 41**,
sibling churn measured (512 commits, 47 deleted `.rs`, 40 from one extraction).
Record: `.docs/10_audits/percentile_index_tail_support.md`.

**Third (2026-09-06) found nothing — a result.**
`scripts/required_features_drift_sweep.py`, pins
`scripts/required_features_drift_floors.txt`: **16 repos, 138 manifests, 1,829
rows, 0 invalid**, <1s. **Pins** a clean workspace. `max_invalid = 0` is a **WALL, not a
ratchet** (no backlog, never legitimate). **Two floors**: seven repos have ZERO rows
and `parse_rows` swallows `TOMLDecodeError`/`OSError`, so `min_manifests` bites
where `min_rows` can't; unparseable manifests counted. Canaried eleven ways; **the first
end-to-end canary red for the wrong reason** (missing `katgpt-rs` temp-pin row
tripped another assert first); re-run with a PASS control arm. **Check WHICH
assertion red a canary.**

**Same blindness on `numbering_gate.py`, 2026-09-05 (`.issues/725`).** katgpt-rs
was the only clean repo: **35 tracked duplicate numbers — riir-train (13),
mmorpg-editor (12), riir-ai (6), riir-clippy (4)**. Plus **five `.highwater`
files not integers** (`echo -n <N> > .highwater` under a builtin `echo` ignoring
`-n` → `-n 872`); `scan()` swallowed `ValueError` → `None`, same as ABSENT, so the
ceiling passed a corrupted allocator. Repairing one exposed a stale allocator
(riir-train `.plans` max 375 > 374). All 7 allocator defects repaired, pinned at
0. Duplicates 35 → 12 same day (Issue 725 T4b/T4c): **riir-ai 6 → 0** via T4a's
`scripts/citation_weight.py` (by-name citations 0-2/side, TIED in four of six, so
`Plan N` mentions ATTRIBUTED by token overlap with UNRESOLVED bucket and a
zero-is-not-zero guard measured on `.plans/229`; 175→568, 182→567, 229→566,
313→569, R020→362, R148→363), **riir-clippy 4 → 0** (its Issue 069, `58e7c1d`),
**riir-train 13 → 0** (its Issue 514, `103ed351` — hand reads overturned four
UNDECIDABLE rows). Remaining **12 all in read-only mmorpg-editor**, ratcheted.
Reading the sites decides.

Sweeps are **not** in `docs_gate.sh`'s `CHECKS`: CI's single checkout derives an
empty population → confident green over zero repos. Population derived
(BOUNDARY.md + `.git` dir); expectations committed (`scripts/docs_drift_floors.txt`)
— deriving both from one walk makes a cross-repo gate permanently green.

`sibling_docs_drift.yml` (`workflow_call`) asserts the audited tree is the
caller's — auditors default to **katgpt-rs**, so an omitted path passes forever.

`scripts/ci_gate_coverage.py` is a **report** (exit 0): which contract repos gate
their full compile+lint surface in CI, following workflows into scripts, **and
whether anything automatically starts it**. Hand answers were wrong
twice (`.issues/701` R2). Every-run `selftest()` (five shapes), canaried with the
original bug: a data-borne mention in `riir-chain`'s scheduled `toolchain_drift.yml`
vouched for a dispatch-only `rust.yml`. A weak automatic gate must not speak for a
strong manual one.

## cfg-gated targets — original section

### A green test count can be a count of nothing — `scripts/cfg_gated_target_audit.py`

A test file opening `#![cfg(feature = "x")]` compiles to an **empty binary** when
`x` is off; cargo prints `ok. 0 passed`, **exits 0** — byte-identical to a pass.
`#![cfg]` protects the **count**, `required-features` the **reader**; both needed.

```bash
scripts/cfg_gated_target_audit.py            # all contract repos (derived)
scripts/cfg_gated_target_audit.py ../riir-ai # or one, by path
```

A **report** (exit 0): `target_os`/`miri` cfgs and `any(...)` of features cannot be
`required-features` (AND-only); reported as own classes so the report doesn't cry
wolf.

**Next axis down:** `scripts/suite_membership_audit.py` (first census 2026-09-04,
`.docs/10_audits/suite_membership_census.md`) reports `[[test]]` targets no
script/workflow names — the "gate nobody runs" class (865, 868). "Unpinned" =
unnamed, not broken; actionable cut = load-bearing + unpinned + default-visible +
no broad `cargo test` in repo, which on first census lands exactly on this repo's
723 and riir-train's 507. Run when landing a gate: add a suite row or record why not.

**Read the severity split, never the pooled total.** Default-on gated targets run
on plain `cargo test`; default-off report a green zero whenever named. Pooled 702
meant nothing; split: **382 SILENT-NOW / 320 latent** across 19 repos, many
load-bearing by name. Record: `.docs/10_audits/cfg_gated_silent_zero_pass.md` (two same-day
corrections: over-count of 48 from explicit `path` rows; token matcher 87 vs table
93, **five of six its misses** — `g16f`/`g2p`/`g9gov`, `drills`, `regate`; now
matches per repo). Agreement licenses `max_load_bearing = 0`. Run the script.

**Fourth measurement 2026-09-06, widening the POPULATION (`.issues/728`).** Over
all 16 repos: `silent_now_load_bearing = 0` in **every one** over 261 SILENT-NOW —
meaning "speaks one repo's dialect". Six tokens + adjacent-pair compound
`spec_match` → **0 → 12**, **two katgpt-rs's own**. Neither half works alone
(`spec` = speculative decoding here; `match` admits `attn_match_*`). katgpt-rs has
seven `*_spec_match` targets; **five were classified only because they also carry
`g1`**. Visible-by-accident is not covered.

**Agreement is not width** — T4c widening surfaced 17 more. Agreement guards a
**bug**, not **narrowness**; defence is the corpus-wide token table
(`.docs/10_audits/cfg_gated_silent_zero_pass.md` T4c).

**`debug_assertions` dimension (2026-09-03)** — what a green `w/ req-f` column lies
about. Pooled with `target_os`/`miri`, it hid the worst case: `not(debug_assertions)`
is silent under plain **`cargo test`**, and **survives the fix** (a row moves it to
"w/ req-f" without compiling it). Reported as an overlapping **dimension**, not a
bucket. 19 repos: **133 targets, 29 load-bearing, 11 "covered"**; **130
`not(debug_assertions)`** (26 load-bearing, mostly riir-ai GPU benches) vs **3 bare
`debug_assertions`** (green zero under `--release`) — all three load-bearing alloc
gates, vanishing exactly when gates are run in release per the rule. Found by
riir-ai `.issues/855` Class 2; fix: split alloc assertion from wall-clock (no single
profile observes both).

Verdict half: `scripts/cfg_gated_floor_gate.py` (docs-gate check), separate from the
report so the report stays runnable over siblings without Issue 713 T3.

Fixed singly twice in a week (riir-train `5821cba9`, 11 assertions never run;
riir-clippy `19beece`) before anyone counted.

Adding rows changes cargo one way: `cargo test --workspace` *skips* targets with
unmet features; naming one without features errors 101. Verified before the
katgpt-rs batch (`180be9c5`, 39 GOAT gates, SILENT-NOW 102 → 63, re-measured with
the corrected auditor).

**"Adding rows does not red existing CI" was false (2026-09-06)** against gates
counting **green binaries**: an empty gated binary prints `test result: ok. 0 passed`
and counts. Three repos count that way (`grep -c '^test result: ok'` in `riir-auth`,
`riir-dapps`, `riir-viewbridge` `scripts/ci_feature_guard.sh`); two reddened:

| repo | floor | before arming | after | note |
|---|---|---|---|---|
| riir-viewbridge L4 | `>= 13` | 13 | **12** | one row (`net_ffi_roundtrip`) |
| riir-dapps L2 | `>= 32` | 32 | **31** → **7** | one row, then all 25 |

Both floors ask *"a target silently compiled to nothing?"* and were **satisfied
by exactly that** (riir-dapps: 25 empty + 7 real binaries, 111 tests). Repair: a
**passed-test floor** beside the binary floor (riir-viewbridge `d6f18f5`,
riir-dapps `dd6261e`). `--lib`-scoped floors (riir-ai) and rows naming a target
*with* features (riir-chain `test_gate.sh` `mnemonic_spec_match`) don't move.

**Run armed gates with `--release`.** All 39 pass; a debug sweep showed four reds,
two nearly filed as perf regressions (`fast_bpe_goat` 388 s debug vs 15.6 s
release). Arming surfaced
`.docs/10_audits/alloc_gate_per_thread_counter.md` (alloc gate counting a sibling
test's allocations; reproduces in release).

## required-features rows — original section

### A `required-features` row can EXIST and be WRONG — `scripts/required_features_build_audit.py`

Audits treat a target with a row as protected; a wrong row is worse than none:

- `cargo test --workspace` **skips** it — nothing reds.
- `--all-features` **builds** it (union supplies the forgotten feature).
- Every audit counts it **protected**.

Not static (cfg-gated `lib.rs` glob re-exports); ask the compiler per target:

```bash
scripts/required_features_build_audit.py --list            # rows only, no builds
scripts/required_features_build_audit.py ../riir-train     # one repo
scripts/required_features_build_audit.py . --grep pruners  # one slice
scripts/required_features_build_audit.py ../riir-train --batch  # 1 run per set
```

**Report** (exit 0), also for cost: **1,829 rows / 16 repos** (2026-09-05) at
~28 s/row warm — hours; filterable (`--package`, `--kind`, `--grep`, `--limit`),
`--target-dir` when a sibling builds.

`--batch`: **one cargo run per (package, EXACT feature set)** (a superset may
supply the forgotten import). **1,829 rows → 1,070 groups (1.71x)**, not the
speedup: riir-clippy (44 rows / 25 groups) **-11% / -8% CPU-s**, wall flipped sign
(-12%, +13%) under load. Verdicts per-target from `--message-format=json`
`compiler-message`/`compiler-artifact`, `--keep-going`. **Neither error nor
artifact = UNSEEN, never BUILDS** (`attribute()` pinned by `selftest()` both ways,
canaried). Warns when another cargo holds `target/`, by working directory, per
§"Several sessions, one target dir".

Instances: first by hand (riir-train `9da3420f`, `test_cubecl_backward_grads`
omitting `gpu_training_resident`; fix revealed 9 passed / 1 failed). Second by this
script **in its first six rows, here** — `bench_001_pruners_goat`, `["bomber",
"go"]`, `E0432`; twin `bench_001_pruners_goat_proof` fixed by Issue 723 T7, which
missed the neighbour. A hand fix does not close a class.

**Free verdict:** a row naming an un-enableable feature is manifest-decidable —
static pre-pass: **0 invalid over 1,829, <1s** (2026-09-05). `required-features`
accepts **`dep/feat`** and `dep?/feat`; a first cut flagged 10 riir-ai benches. A
`/tmp` `compile_error!` probe (cargo 1.98.1): satisfied via an enabling feature,
`dep/feat`, or `--all-features`; skipped only in a plain build. Renamed (`package = `) and `[target."cfg(…)".dependencies]` count.

**Correct AND insufficient (2026-09-06).** Sixth instance: riir-train
`lora_muon_optimizer` forwarded `riir-gpu/amuse_optimizer` (the DEPENDENCY's)
while `pub mod optimizer_amuse` in `riir-train-gpu` gates on its OWN
`amuse_optimizer` — valid `dep/feat`; only the compiler tells "exists" from "gates
the module".

**It widened the class:** instances 1-5 are wrong ROWS; 6 is a feature set where the
**LIBRARY** doesn't compile (`cargo check -p riir-train-gpu --lib --features
lora_muon_optimizer` → `E0432` ×2), so ten rows report **UNSEEN**. **UNSEEN never
folds into pass**: "will clear after the two FAILs" was TESTED and FALSE after
instances 4 and 5 landed — how the library break was found.

**Repaired in OPPOSITE directions — read the USE SITES, not the error.**
`dasd_lora_goat` uses `AsftConfig`/`asft_loss` unconditionally → widen the ROW.
`goat_235b_filter_training` had an ungated import whose only use carried
`#[cfg(feature = "lora_outlier_guard")]` with `..Default::default()` → narrow the
cfg (widening would defeat per-leg design). Widen only when the body needs the
feature unconditionally.

### The per-push half — `scripts/required_features_touched_gate.py`

The sweep (1,866 rows / 16 repos, ~28 s/row) is scheduled; every instance
arrived in a **commit**, so gate rows this push could break (cost ∝ diff). `.github/workflows/required_features_touched.yml` runs per push
and PR on macos-latest (platform is part of the claim).

Selection: a changed file that IS a row's target source; a changed `Cargo.toml`
selects rows whose `(kind, name, required-features)` tuple **differs base vs head**
— a ROW diff, not the package (`riir-train-gpu` has 440 rows, home of three of six
instances). Both sides parse via the sweep's `rows_from_manifest`.

**Green is narrow, printed every run:** a changed `src/**.rs` can break rows in
dependents (instances 1 and 6), unbounded, so left to the sweep (`--src-fanout`
opts in; `--max-rows` REFUSES rather than truncates). Cost: 0 rows ~0.4s · 1 warm
row 3.1s · 1 graph-swinging row 266.7s.

Record: riir-train `.issues/513`.

## Percentile index — original section (classifier history)

### A reported "p99" is often the MAX — `scripts/percentile_index_audit.py`

`sorted[(n as f64 * 0.99) as usize]` and `sorted[n * 99 / 100]` land on `n - 1`
(the **maximum**) for every `n <= 1/(1-p)`: n ≤ 100 at p99, ≤ 20 at p95, ≤ 1000 at
p999. A `.min(len - 1)` clamp prevents a panic, not a wrong statistic.

Naive index is one rank **too high**: `p99 < budget` gets *stricter* (false
**RED**); cost is missing real regressions (tail = one sample).
**Tail support** = `n - idx`: **1 at n=100**, 2 at n=200, 10 at n=1000; <10 is weak.

```bash
scripts/percentile_index_audit.py             # all contract repos (derived)
scripts/percentile_index_audit.py ../riir-ai  # or one, by path
```

**Report** (exit 0) — like `cfg_gated_target_audit.py`, half the sites size from
runtime lengths no static pass reaches. **UNRESOLVED is not "clean"**.

Record: `.docs/10_audits/percentile_index_tail_support.md`. **2026-09-03
evening: 126 sites / 9 of 19 repos — 0 DEGENERATE, 2 TRUNC-VAR, 6 WEAK, 31 OK, 62
UNRESOLVED, 25 SAFE.** DEGENERATE zero earned: four owners fixed all 12 same day
(riir-ai `03a91ed59` swept 10, riir-mmorpg-examples `ee9da24` the one
DEGENERATE-**ASSERTED**, riir-game-sdk `f896bca`, riir-chain `7f3a3910`).

**The fix blinded the gate.** Repairs moved behind `nearest_rank(sorted, p)`
(variable p; vocabulary required *literal* p): 130 → 114, `max_degenerate = 0`
green without riir-ai's surface, **seven byte-identical helper copies across five
repos** invisible — caused by a *correct* fix. Closed by `TRUNC-VAR` (truncating
variable-p rank in a percentile-named scope; `floor(p*n)` is the max for every
n ≤ 1/(1−p)), ceiling `max_trunc_var = 0`, and fixing the `.trunc()` hole.

Corrections: **"false RED" is assert-direction dependent** (`p95 >= floor` →
false GREEN); **`asserted` is blind for helpers** (same-fn-scoped) — so `TRUNC-VAR`
gates regardless.

**Wrong once per vocabulary gap, never by a bug; each narrow version looked like good
news** (count not written):

1. First cut grepped only **float** forms, published a 14-row hand table as "all 19
   repos"; integer `n * 99 / 100` (more common) invisible; riir-e2e's found by accident.
2. `resolve_n` and `is_load_bearing` **file-scoped**: false ASSERTED (riir-neuron-db
   `bench_003`, assert on neighbour `mean_us`) and a slice *parameter* sized from an
   unrelated caller (riir-chain `bench_012`).
3. Literal-only reported **riir-game-sdk zero sites** while its `percentiles` helper
   (`|p: usize| durs[(n * p / 100)...]`, `at(50)`, `at(99)`) feeds budget gates.
   Variable p must land in UNRESOLVED, not vanish.
4. Blind to variable-p **helper bodies**: the 2026-09-03 campaign moved twelve sites
   behind seven `nearest_rank(sorted, p)` copies out of the population instead of to
   SAFE. Closed by `TRUNC-VAR` + scope-name discriminator measured on all 27
   candidate sites (admits 8, rejects 19, incl. two a bare `rank` substring would
   swallow). Defence: the corpus-wide table.

Vocabulary is **data** (`VOCAB`), population **derived** (BOUNDARY.md + `.git`).
`selftest()` every run, **exits 2**, pinning tokenizer, scoping, rounding exclusion,
arithmetic; canaried by reintroducing bug 1 (greedy class swallowing `sorted[(n`)
→ all UNRESOLVED; without it: 130 sites, zero findings, "clean".

`.trunc()` in the rounding exclusion was a bug (`x.trunc()` **is** `x as usize`
for x ≥ 0); latent, fixed. Legitimate exclusions: `((n - 1) as f64 * 0.99)` bounded by `n - 2`
(verified n ∈ 2..=20000; used by
`katgpt-speculative/tests/weaver_real_checkpoint.rs`, the one site right), and
`.ceil()` / `.round()` (correct nearest-rank; also removes top-p **nucleus size**
`0.95 * n as f32` false positives).

## Staged-set audit + shared target dir — original sections

### Before committing in a shared worktree — `scripts/staged_set_audit.py`

`git add -A` in a shared worktree is indistinguishable from intent: `b2527521`
committed three agents' WIP in six files, one a build regression unseen for a day
(`.issues/709`). Stage **named files** (`git -C <repo> add <paths>`), never `-A`,
and before a multi-file commit run:

```bash
scripts/staged_set_audit.py            # any repo: pass its path as $1
```

**Report** (exit 0); a refusing pre-commit hook was **decided against** (`.issues/709`
T3b, 2026-09-03) — every cheap signal has legit false positives. Signals:

1. **mtime clusters** — worktree mtimes cluster by editing episode (a 204-file
   rustfmt sweep lands in 3s); two clusters = two episodes, the older probably not
   yours.
2. **also-dirty** — staged path with unstaged changes: a concurrent editor inside
   your window.
3. **stale-vs-HEAD** — dirty/staged file LACKING substantive lines the newest commit
   on its path added; committing reverts them. Found live: `tpr/als.rs`, written
   20:04:39 by a rustfmt sweep while `0ef7f078` landed a 22-line Issue 712 fix at
   21:08. Audits the **dirty** set too.

Signal 3 is two-stage: `mtime < commit time` alone flags your own edit-then-commit;
line-set **containment** on specific lines (not `}`) confirms. 19-repo sweep: one hazard.

4. **rustfmt round-trip** (`--fmt`) — `git show HEAD:$f | rustfmt --emit stdout |
   diff - $f`. Identical ⇒ provably **zero content**; the only *proof*. First run refuted a belief: 15
   files called "sibling rustfmt churn" were **0 churn, 15 content**, real lint fixes
   (`989f1bdf`). `skip` verdict pinned by `selftest()` — `churn` authorises a revert.

Single-linkage clustering (no gap ≥ `GAP_SECONDS` = one episode). `selftest()`
pins ten shapes — both failure modes are silent.

Committing into a file a sibling edits: commit **your blob** — HEAD + your edit,
`git hash-object -w`, `git update-index --cacheinfo` (used for `bench_707` in
`8c7ca74b`).

### Several sessions, one target dir — a gate can produce a FALSE RED

Costlier than wrongly-green: **a count-pinned or feature-switching
gate run concurrently with another cargo in the same `target/` reports a failure
that passes alone** — cargo replaces test binaries mid-run.

Read the **shape**: `error: test failed, to rerun pass …` with **no `failures:`
block and no `test … FAILED`** = harness *died*; the count is truncated too (cargo
stops at the first failing binary) → fake failure *and* apparent pin drift.

Measured 2026-09-03 (riir-game-sdk `.issues/023` T4): `105 passed (pinned 182)`
naming deterministic `prod_l3_partition_heal`; the binary run **directly** passed 6/6. Three
`cargo test -p riir-e2e` runs on three feature sets were live, two mine.

**Running the compiled binary directly needs no build lock** — under
`target/<profile>/deps/`, filtering `#![cfg]`-gated copies `--list` shows as 0
tests. A gate the box can invalidate should **refuse**: riir-game-sdk
`scripts/test_gate.sh` detects concurrent cargo by **working directory** (`lsof`
over `pgrep -x cargo`), not command-line pattern. Lock checks can't work: cargo releases
`target/<profile>/.cargo-lock` *before* running test binaries.

## Feature Flag Discipline — rule histories (lossy surface, Report the Floor, Plan 467)

**Lossy-surface promotion rule (adopted 2026-08-28, riir-ai Issue 750 T3):** lossy
surfaces (quantization, compression, bit-changing) gate on **deployed-path behavior
— per-family, conditional retention**, not bit-identity (lossless-only) or aggregate
perplexity. Three arrivals: Research 502 ("Behavior Before Perplexity"), Bench 696
(KVarN sink-guard GOAT), riir-ai Issue 750 bisection (gemma-2-2b Q4_K: first flip at
prefix k=1 — layer 0 alone flips the sealed family; restoring costs 106.7 MiB, T2
override probe). External (riir-clippy Research 125, walk #7): arXiv 2609.01962
(Qwen3-4B ternarization: aggregate 64.5→54.7% but retention 84.6% BoolQ vs 43.8%
ARC-Challenge; lossless packing holds PPL, lossy excluded from the claim) and arXiv
2608.12700 (contract-grade verifier rejects 1,487/2,638 kernels a tolerance harness
accepted — walk #6 fault-class confirmation).

**UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule, adopted 2026-06-28 per Research 322 / Plan 340).** Any **UQ-bearing** primitive (distribution, interval, quantile, coverage, confidence, calibrated uncertainty) MUST benchmark against the **conformal-naive floor** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340 `m=1`, split conformal) — on CRPS / coverage / Winkler; can't beat it ⇒ GOAT FAILS. Grandfathered (BoMSampler Plan 281, Sleep-Time Anticipator Plan 334, Best-Belief Beta Selector Plan 336, KARC+overlay) add it at next re-gate. Tracked `.issues/010`; floor shipped Plan 340 Phase 1 (2026-06-30). **Issue 010 FULLY CLOSED (T1-T7)** — `.benchmarks/010_report_the_floor_consolidated.md`. **T7 (2026-07-20)** `conformal_floor_karc_overlay.rs`: KARC+overlay SCOPE-LIMITED to chaotic regimes (BEATS Lorenz-x crps_ratio 0.0047, K=4; LOSES stationary seasonal 5.74, K=4), coverage calibrated both. **K-sweep** refuted "K=4 too shallow": K=12 seasonal CRPS 5.74 → 20.26, Lorenz 0.0047 → 0.0018 — **structural** (Chebyshev basis + ridge fit vs periodic data). Guidance: pick K by chaotic memory; periodic data → use the floor.

**Plan 467 / Proposal 007 (2026-07-18):** `DualLeoOracle` shipped as QGF's 3rd `QGradientOracle` (LEO teacher + UVFA student via `DualLeoMixer::combine_into` at gradient level; sibling to `LeoHeadOracle` Plan 268 + `FlowFieldOracle`). G1–G4 PASS. **G5 FAIL synthetic (riir-ai Bench 553, 2026-07-18): dual 0.00% vs single 0.50% on T7 Go; QGF+LeoHeadOracle ≡ baseline bit-identically — near-flat Q-fields.** **G5 FAIL civ real (riir-ai Bench 558, 2026-07-19): dual +2.69% (35.68% → 36.64%) vs ≥3% gate — fourth-axis stop.** Closed per riir-ai Research 322 (UQ primitives forecast states, not per-action Q-gradients). Plan 460 invariant: no operator between mix and consumer. Opt-in (`qgf_oracle + dual_leo`); reopens only on mmorpg-remake integration gain, new-domain positive G5, or Q-vs-forecast breakthrough.

## Substrate-First Gate — original section

## Substrate-First Gate (MANDATORY before implementing)

Before any new System impl, trait, perception/cognition/emotion pipeline, state
management, spatial query, or vocabulary type, run
`.agents/skills/substrate-first/SKILL.md`:

1. **Vocabulary translation** — grep 3+ name variants (operator names like
   `GenericSpatialBelief`, not "threat field"); single-vocabulary grep returns ZERO.
2. **Codebase grep** — `*.rs` across all 8 repos, not just `.plans`/`.docs`/`.issues`.
3. **Architectural rule check** — domain classification, two-brain, sync boundary, bridge.
4. **Consume vs. build** — consume existing substrate; else file an issue in the right repo FIRST.

Prevents parallel systems duplicating shipped substrate (ThreatField Issue 047,
orchard/motivation riir-ai Issues 490/493).

## Research Workflow — original section

`.agents/skills/research/SKILL.md`: paper classification, 7-repo routing,
fusion-first distillation, novelty gate, GOAT gate, modelless-unblock protocol (§3.5).

## Repo count — the full original paragraph (drift history)

> **Repo count:** the **product/distillation set is 7** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-dapps` (private). Workspace total **16 repos**, all with root
> `BOUNDARY.md` (add `riir-mmorpg-examples`, `riir-clippy`, `riir-viewbridge`,
> `riir-auth`, `katgpt-web`, `riir-dao`, `riir-deployer`,
> `mmorpg-editor`, `mmorpg-remake`).
>
> **19 → 16 on 2026-09-04 00:01** (owner act): `riir-burner`, `riir-unity`,
> `mmorpg-remake-unity` moved to `/Users/katopz/git/obsolete/`, intact
> (`riir-burner` last sweep pushed `ce54122`). **Lineage only; do not route work.**
> Caught by `skill_repo_set_gate.py` (stale `scripts/repo_set.txt`) then
> `agents_repo_set_gate.py`. Edges 2026-09-04: **exit 0 — 16 repos / 224 edges**,
> re-confirmed via `--list-deps` (full run exit 1 only on a worktree-only C6
> rustfmt artifact); 226 after the Issue-092 ndb-sdk edge. Don't re-type; run
> `../riir-ai/scripts/ci_boundary_contract.sh --list-deps`.
>
> **Said 8 and 18 until 2026-09-03 — worse than stale.** `riir-armageddon`
> de-enrolled (directory GONE), `mmorpg-remake-unity` enrolled same window
> (boundary-guard 18th run: 19 repos / 225 edges; 227→225 = armageddon's two
> allowlist edges, new repo zero). **Total stayed 19 while MEMBERSHIP changed** — a
> count is not a checksum over a set. Derived instruments correct throughout
> (`scripts/repo_set.txt` regenerated at `d2cb9979`); only prose stale.
> **Re-measured 2026-09-01** by `../riir-ai/scripts/ci_boundary_contract.sh` —
> *"boundary contract clean — 18 repos, 211 cross-repo dep edges measured"* (was 15
> at 2026-08-21; contracts added, paragraph didn't move). **19 later that day**:
> `mmorpg-remake` scaffolded 23:41 with BOUNDARY.md (211-edge figure NOT
> re-measured); `scripts/repo_set.txt` regenerated (`01e19858`) after the docs gate
> reddened. Don't re-type:
>
> ```bash
> cd /Users/katopz/git && for d in */; do
>   [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
> done
> ```
>
> `riir-armageddon` consumed `riir-games` + `katgpt-core` unaudited (the 227→225
> edges). A MATCHING count is a claim too. "5-repo quintet" = katgpt-rs + 4 riir-*
> distillation targets; later `riir-game-sdk` (game vocabulary facade), `riir-armageddon`
> (arena types; **retired** 2026-09-02, lineage only), `riir-dapps` (game outcome →
> chain settlement, 2026-08-20). Canonical boundary: Research 003.
>
> **Two axes.** Research 003's repo table = *public/private*; §"The Second Axis:
> Layering (game / dApp / chain)" = which private repo. **Three tests, all must
> pass** (revised 2026-08-20; one-question form admitted FAME as value, ignored write
> rate): **(1) Product** — would a chain commerce customer want it? NFT yes, quest no.
> **(2) Value** — BigInt fungible currency, token, or authority binding? FAME / XP /
> items / reputation are game scalars. **(3) Rate** — fits Glacial tier (≤0.1 Hz)?
> Binds hardest; `riir-neuron-db` is 1,627× cheaper per write and one chain tx at
> 10⁵ accounts eats 63% of a 20 Hz hot tick.
> Canonical failure: quest / bounty / crafting / reputation (two moving no money) in
> `riir-chain`'s consensus-critical program set — `riir-chain` Issues 096 + 097,
> closed on the layering side by `riir-dapps`.

## Resolved issue log (verbatim from pre-compaction AGENTS.md)

## Issue log (resolved)

- **Issue 792 (allocated as 776; renumbered per Issue 791) — the docs gate's CPU self-timing printed a well-formed number that measured nothing on Windows** FILED + RESOLVED + removed
  (2026-09-14, same session; row + git history are the durable record; found landing Issue 775's 18th check).
  Besides the already-guarded forked-`times` `0m0.000s` case, a second failure prints a plausible number: on
  Windows/MSYS `times` counts MSYS children and ~nothing for NATIVE ones, so an all-Python run reported **1.26s
  CPU against a 19.7s wall** (`sed`/`tail` overhead) — the figure AGENTS.md says to cite against an M3 series of 13.37s.
- **Issue 776 (the OTHER 776 — dual allocation, Issue 791) — contrastive matched-swap + norm-matched noise interventions, CVRR §2.1/§5.3** DONE T1–T7, removed
  2026-09-15 (research 555, arXiv:2609.06746; commits `be4ff672` + `d936b5fd`; Issue 791 adjudicated the number
  to this document, weight 16 vs 5). Landed `perturb_matched_swap` / `perturb_norm_matched_noise{_rows}` +
  `probe_matched_swap{_into}` / `probe_norm_noise{_into}` (zero-alloc, caller scratch), the battery's sixth
  `norm_matched` arm, `LatentSpace::norm_matched_noise` (katgpt-core `interpolation_geometry`, opt-in), bench cost
  rows (matched_swap 0.37 µs / norm_noise 46.2 µs at n=4096 — ≪1 ms, 21–2700× headroom); 73 tests green, clippy
  `-D warnings` clean, default-off unaffected. The arm separates norm-readers from structure-readers. Downstream:
  riir-neuron-db's `KarcWoutSpace` (norm_matched_noise_slice, katgpt-rs `33f77706`).

  T1, one child at a time (~2s CPU each): MSYS `bash -c` → **1.796s user + 0.468s sys**; `py -c` → **0.000s +
  0.015s**; python.exe by ABSOLUTE PATH → **0.000s + 0.045s** — the MSYS/native boundary, not the shim (T2
  refuted). The fix deviates from T3's CPU-vs-wall ratio (the gate's own 12.65s CPU on a 299.1s wall = 4%, so no
  ratio separates platform from busy box): it CALIBRATES — burn 0.25s CPU in a child of the resolved interpreter,
  require `times` to see ≥half, else print `CPU SUPPRESSED` + wall-only + remedy (T4). Measured: native 0.000s →
  suppressed; MSYS 0.358s → printed. Cost ~0.35s/run. Verdicts 18/18 green throughout; only the timing line was wrong.

- **Issue 775 — `platform_dead_code_audit.py` had no VERDICT half, and nothing automatic ran it** RESOLVED + removed
  (2026-09-14; row + git history durable; filed the day the report half landed, `a0cbc398`). The class is invisible
  to every automatic lane (`full_gate` macOS/aarch64, `wasm32_gate` wasm32; the x86_64 lane emitting `dead_code` is
  a workstation lane), so an unrun instrument just moved the hole up. Landed: (1)
  `scripts/platform_dead_code_floor_gate.py` in `docs_gate.sh` CHECKS (18), globs in BOTH trigger `paths:` lists;
  pins in `scripts/platform_dead_code_floors.txt` — `max_findings = 0`, floors `min_rs_files` (walk) +
  `min_candidate_decls` (parse — broke 3× during construction, each a confident `0 findings`), the one MOD-REF row
  by MEMBERSHIP (path + name, no line number). (2) `scripts/platform_dead_code_drift_sweep.py`, workstation, pinned
  in `scripts/platform_dead_code_drift_floors.txt` (16 measured rows; katgpt-web / riir-dao / riir-deployer /
  riir-esp32 deliberately unpinned — an unmeasured floor certifies nothing). Carry forward: population from
  `repo_set.txt` + walk via `partial_clone_state()` (partial box DEFERS loudly under `DOCS_GATE_PARTIAL_CLONE=1`,
  both postures verified); sweep cross-asserts its katgpt-rs row against the gate's pins incl. `max_modref ==
  len(pinned MOD-REF names)`; the gate canaries its OWN pin arithmetic (6 arms incl. a MOD-REF SWAP with count 1),
  which the classifier's 24-arm self-test never touches. `--prove-fires ea4c2873` default in sweep, opt-in on gate
  (~5.6s `git archive`). Measured: gate ~6.2s (2415 .rs / 29819 decls, 0 findings, 1 MOD-REF), sweep ~30s / 16
  repos, docs gate 18/18. ⚠ CPU self-timing read **1.26s vs 19.7s wall** on Windows — filed as Issue 792, not re-pinned.

- **Issue 765 — the docs gate's 3 workstation-only instruments red with a MISLEADING remedy on a partial clone (the 4090 box class)** RESOLVED + removed
  (2026-09-13; the `DOCS_GATE_PARTIAL_CLONE=1` marker axis; row + git history durable; filed by the 4090 session
  resolving Issue 764). On 14 of 20 repos, `skill_repo_set_gate` / `population_sync_gate` / `issue_citation_gate`
  red, and the first two said "regenerate repo_set.txt and commit" — which deletes the 6 absent-but-live repos.
  Deliberately NOT auto-detected: a forgotten removal is set-identical to a partial clone (same reasoning as
  DOCS_GATE_CI). Landed: (1) the marker gives a loud instrument-alive DEFERRAL on the population axis on each
  check's FINAL line (predicate agreement + local axes still run); the citation gate reuses `ci_deferred` with a
  `posture` param, deferring the cross-repo half even above the floor (else MISATTRIBUTED rows for absent repos);
  (2) gone-only disagreement WITHOUT the marker reds naming BOTH hypotheses + corruption warning; (3) a repo on
  disk the snapshot lacks reds in EVERY posture. Shared `partial_clone_state()` in `skill_repo_set_gate.py`;
  `WORKSPACE_ROOT` override in the other two. Validated: M3 17/17 both postures; 14-of-20 symlink farm — unmarked
  3 reds with new remedy, marked 3 deferred greens, unregistered repo reds even WITH marker; CI paths unchanged.
  4090 usage: `DOCS_GATE_PARTIAL_CLONE=1 ./scripts/docs_gate.sh`.

- **Issue 763 — signed-graph LIF reservoir primitive, event-driven sparse propagation (fly-connectome survey fusion)** RESOLVED + removed
  (2026-09-13; landed `74fe08f1`; row + [Bench 760](.benchmarks/760_lif_graph_goat.md) + git history durable).
  Opt-in `lif_graph` in katgpt-core: `SignedAdjacency` (signed CSR, sign pre-folded — RuVector layout) + `LifParams`
  (Shiu et al. Nature 2024 LIF constants, exact exponential integration, 3 muls/active neuron/tick) +
  `LifReservoir` (timing-wheel delay ring, PSP-mV weights — 0.275 mV/synapse vs 7 mV gap, ~26 synchronous synapses
  to fire) + `fit_readout` via `linalg::ridge_solve_direct_f64` (closed-form; joined the linalg cfg any-list, the
  Issue-701 class). Core: the exact-parity active set — quiescence is the bitwise fixed point, so event-driven
  `step` and dense `step_dense` are bit-identical. G1: 7 tests (sparse-burst / cascade / chain-ring; bitwise
  state; determinism; exact delay 0/18/36; Maslov–Sneppen degree preservation; active-set churn). Hazards caught:
  (1) spike-order ULP accumulation (2-ULP divergence at g[295]) → canonical ascending spike order both paths;
  (2) ring-slot alignment — 6 recurring allocs because period 1400 wasn't a multiple of ring length 18 → period
  1404 = 18×78. G2 (Bench 760, 4090 host CPU): **97,285×** vs dense-W matvec at N=10k/3.1% active (gate ≥3×),
  **34.8×** vs CSR full-scan; saturated 0.91×/1.08× parity. G4: 0 steady-state allocs both paths. Controls:
  `er_matched`, `maslov_sneppen`. **Stays OPT-IN** (consumer-first); consumer path riir-ai per-archetype circuit
  shard + per-NPC readout (riir-ai Research 379 §7). Catalog §108; claim sites 596→597, default count 200.

- **Issue 764 — `LoraPair` KV-cache weight-epoch contract unspecified (public twin of riir-ai Issue 938)** RESOLVED + removed
  (2026-09-13; landed `49f5d245` + `aa163896` + `a7d6d8f0`; verified on the 4090 issue sweep; row + git history
  durable). `49f5d245` — `WeightEpoch` (BLAKE3) + `LoraAdapter::weight_epoch()` + mixed-epoch acceptance docs +
  `core_04_prefill.rs` swap-site docs (items 1+2); `aa163896` — `WeightEpoch::from_parts` (frozen-model axis for
  riir-ai Gemma2/KVCA v2); `a7d6d8f0` — `WeightEpoch::as_bytes` (KVCA v2 wire header). All five surfaces present at
  HEAD (`katgpt-types/src/lora.rs` :518/:540/:554/:563 + `core_04_prefill.rs` :294-300). Item 3 (optional refusal-arm
  gate) NOT taken — consumer lives in riir-ai `CpuInferenceBackend` (riir-ai Issue 938).

- **Issue 743 — `gw_alignment`, the Gromov–Wasserstein quotient-alignment primitive** RESOLVED + removed
  (2026-09-11, `ae04a98b`; Plan 594 + Bench 709 carry the narrative). Opt-in `katgpt_core::gw_alignment` (pure std):
  structure-only GW of two distance matrices (n,m ≤ 64) via deterministic multi-start (greedy pairing-mass,
  entropic softmin, uniform+tilt, brute-force at n ≤ 8) → product-graph power iteration → f64 loss +
  `sigmoid(−β·loss)`. G1–G4 ALL PASS (11 gates; planted-vs-shuffled ≥13/16, AUC ≥0.93; default build compiles to
  nothing; zero steady-state allocs). Opt-in pending the riir-poc consumer (riir-ai Issue 912's tail). Greedy
  second-order init is load-bearing (uniform start saddle-blind); 2-opt polish on ΣP removed (0.0016 → 0.145 — ΣP
  is not the GW objective). riir-ai Issue 912 T4 chose BUILD; this is it.

- **Issue 742 — the last 42 `#![cfg]`-gated targets in this repo reported a green zero; `SILENT-NOW` is now a WALL at 0** RESOLVED + removed
  (2026-09-09, `2ae0d20a`; katgpt-rs slice of the arming sweep — ndb 616, chain 138, riir-ai 906 same day).
  `SILENT-NOW 42 → 0`, `max_silent_now` **42 → 0 as a WALL**, rows derived from the instrument. Filing:
  `git show 2ae0d20a^:.issues/742_the_last_42_gated_targets_reported_a_green_zero.md`.
- **Issue 741 — alloc gates were unrunnable in the shipped profile + the auditor read one of N cfgs** RESOLVED
  (2026-09-09; fix `da498fa6`, free_gib follow-up `3115f7f4`; filing:
  `git show d43a0dea:.issues/741_alloc_gates_only_measurable_in_a_profile_nobody_ships.md`). The two
  `#![cfg(debug_assertions)]` alloc targets now RUN and PASS under `--release`; the auditor reads every whole-file
  `#![cfg]` (rustc ANDs them). Narrative: `AGENTS.md` §"cfg-gated targets — the green-zero rule". Siblings:
  riir-game-sdk `.issues/028` (`2380fc7`), riir-clippy `.issues/083`, riir-dao `.issues/003` (open, owner-gated).
- **Issue 741-as-filed — `is_load_bearing` cannot name a security gate that is named after its THREAT** RESOLVED
  (2026-09-09; fix `ba26462b`; filing: `git show d43a0dea:.issues/741_load_bearing_vocabulary_misses_the_threat_dialect.md`).
  **Numbering:** DUAL-ALLOCATED 741 with the alloc-gates issue (`741_alloc_gates_only_measurable_in_a_profile_nobody_ships.md`,
  fix `da498fa6`), caught by `numbering_gate.py`; the bare number stays with alloc-gates (blast radius), this one is
  cited by path + `d43a0dea`. Finding: `LOAD_BEARING_TOKENS` named ASSERTED properties, while a convention names the
  THREAT — riir-game-sdk's `prod_l<tier>_<threat>` suite classified **0/31** (2,328 names / 27 repos). Fix: 11 ADMIT
  tokens (`forgery` ×2, `mitm` ×1, `anticheat` ×1, `chaos` ×4, `crash` ×4, `agreement` ×4, `finiteness` ×2,
  `partition` ×1, `sigkill` ×1, `overflow` ×1, `fuzz` ×1) + 3 bigrams (`crash_replay`, `divergence_injection`,
  `front_run`); REJECT homonyms (`replay` 10 · `divergence` 9 · `injection` 3 · `rejection` 1 · `watermark` 2;
  zero-hit `tamper`/`spoof`/`dos`/`adversar`/`byzantine`/`exploit` RESERVED) in
  `.docs/10_audits/cfg_gated_silent_zero_pass.md` §T4f. Impact **0 → 0** in all 17 repos — the counterfactual is the
  finding: the 31 pre-`2380fc7` names replay **11** load-bearing SILENT-NOW under a silent `max_load_bearing = 0`
  wall. Third vocabulary-gap instance (713 T4c, 728, this).

- **Issue 740 — Regime-Probe Primitives: Entropy Gap, Basin Probe, Gardner LUT (arXiv:2604.26841)** RESOLVED
  (2026-09-09; T1–T9 landed; row + git history durable. Opt-in `regime_probe`, `crates/katgpt-core/src/regime_probe/`
  (impl `781264aa`, docs `53fc8b90`; [`.benchmarks/702_regime_probe_goat.md`](.benchmarks/702_regime_probe_goat.md)):
  conditional entropy via the SHARED `katgpt-types::simd::logsumexp_parts` (factored from
  `breakeven/fidelity.rs::cross_entropy`, bit-identical — no parallel entropy code); entropy-gap detector with BLAKE3
  artifacts (`KRPG`); corrupt→renovate basin probe over `FrozenRenovator` (`fastrand::with_seed`); Gardner LUT
  (`OnceLock` 4096-pt grid, worst err 5.2e-10 vs 1e-6 gate). G1 PASS (memorizer gap 1.376→−0.006 nats across
  capacity, generalizer 0.004 flat, crossover 0.375 vs 0.125), G2 PASS WITH CAVEATS (Hebbian ρ_c ≥ bound at γ=1/4;
  Whitened fails the CLT premise, basins 64→865 — a correlator result), G3 PASS (bit-identical), G4 PASS (0 bytes).
  UQ floor (T9): conformal-naive exemption recorded. FIRST CONSUMER 2026-09-09: riir-clippy Issue 077 T1+T4
  (`1994a8a`+`ffdd7a9`, `src/score_bench/ood.rs`); run #84 gap **−0.729 nats** / d −1.11 → **anomalous_negative** —
  corpus proximity, not the OOD label, decided the sign (riir-clippy `.docs/08_benchmarks/entropy_gap_ood_axis.md` +
  Bench 702 addendum `0a9994fd`). PROMOTION: stays opt-in — convergent validity unshown; unblock = provenance-disjoint
  pairing + re-read of Bench 702 + 077's convergent-validity GOAT (open in riir-clippy Issue 077, with T2/T3). Gates:
  katgpt-core lib 2001 @ `regime_probe` / 1979 @ default, clippy clean, docs_gate 14/14; consumer 1396/0.)

- **Issue 738 — the wasm32 lanes compile what they NAME; nothing checks that what they name is the whole surface** RESOLVED
  (2026-09-08; T0–T3 — `scripts/wasm32_surface_audit.py` (POSITIVE-cfg predicate, comments excluded, population from
  BOUNDARY.md + `.git` dir; NAMED/UNRESOLVED/UNCOVERED with walk size); T1 resolved 14/15 UNRESOLVED on row-bearing
  static evidence; the 15th, riir-ai `riir-examples`, UNCOMPILABLE for wasm32 (uuid lacks `js`) → riir-ai
  `.issues/894`, resolved same day (uuid `js` + a clippy fix + a LITERAL `-p riir-examples` row in guard layer 1.22);
  mmorpg `warm-tier-do` lane gap landed `b23dc52`; T3 excluded riir-ai's vendored `wgpu-hal`. Standing 23 NAMED ·
  0 UNRESOLVED · 0 UNCOVERED / 23 packages / 191 files / 17 repos (2026-09-08). Three instrument bugs (0-file walk
  from Python `\s` in POSIX ERE; 17 false UNCOVERED from a derived `-p` list; 2 from a `--manifest-path "$unit/…"`
  lane) recorded as the classifier-lessons canon; narrative in AGENTS.md §"A lane compiles what it NAMES".)

- **Issue 737 — nothing in this repo compiled for wasm32; the browser crate had 15 live findings to prove it** RESOLVED
  (2026-09-07; T0–T3 — 18 lint lines healed, `full_gate.sh` layer 2b with both simd128 arms, derived package list
  incl. root, membership-pinned residue, two wasm32 GOAT targets named; first CI red (missing `targets:`) fixed
  `e0b7c9e0`; T4 `091d29cd` — `--wasm32-only` + `.github/workflows/wasm32_gate.yml`, 4m38s cold, full-mode 14m04s
  green. Nine-repo audit COMPLETE (riir-chain `.issues/130` T2, `d44b240a`).)
  **FOLLOW-UP (2026-09-11, `25c89432`): the T4 lane never passed on CI — all five runs (34296642292…
  34314937986) died on the same refusal**: dtolnay installed wasm32 into STABLE but doesn't export
  `RUSTUP_TOOLCHAIN`, so cargo resolved `rust-toolchain.toml` (1.98.1, no wasm32 std) — the `e0b7c9e0` `targets:`
  install was necessary but NOT sufficient; mirrored full_gate.yml's `RUSTUP_TOOLCHAIN: stable` (`25c89432`);
  dispatch run 34578372428 PASSED both arms — first CI green. The mismatch class exists only in this repo's two
  lanes; other repos' `targets:` workflows have no `rust-toolchain.toml`.

- **Issue 734 — a shell gate that ABORTS mid-run reports exit 0** RESOLVED
  (2026-09-07; T0–T11 — `rc=$?; cleanup; exit $rc` can't repair it (saved rc is 0); completion sentinel in 37
  scripts / 10 repos (`70eff640` + waves), 40/41 SENTINELLED, `riir-ai/scripts/e2e_internet.sh` PROVEN inert (trap
  at line 41 of 43, zero triggers); verdicts `scripts/trap_sentinel_gate.py` (membership-pinned, canaried) +
  `scripts/trap_sentinel_drift_sweep.py` (17 repos, max_exposed + min_scripts, exit 2 if untrustworthy) + premise
  instrument `scripts/trap_launder_premise_matrix.py` (11 interpreters). Three canary-caught self-corrections:
  REPLACED false positive (`trap - EXIT` deregisters), awk DATA brace → UNPARSED verdict, and errexit (not nounset)
  as precondition — over-claimed 15/41, spun off Issue 735. Narrative: AGENTS.md §"A gate that ABORTS reports exit 0".)

- **Issue 736 — leakage_probe + cross-space diagnostics: the modelless defender-side attribute-leak audit** RESOLVED
  (2026-09-07; T1–T6 — `katgpt-core/src/leakage_probe/` (opt-in): `probe()` → `LeakReport` tiers
  InsufficientAlignment/Low/Elevated/High; `neighborhood_hit_rate`, `alignment_mean_cos` on the same kNN kernel;
  GD-free unpaired transport (subspace iteration → PCA whitening via `linalg::symmetric_eig` → CSLS Sinkhorn →
  Procrustes, multi-start vs ICP lock-in). G1/G1b/G1c/G2 10/10 PASS; T4 → **Super-GOAT** (unshipped in literature and
  workspace); T5 consumer in riir-neuron-db (`51e2ca1`+`6e14f4d`, Bench 495: top1 0.828 vs chance 0.086, lift 9.64,
  monotone 0.828→0.082 over α 1.0→0.05, 380 ms @ n=256); T6 guide riir-neuron-db `.research/308`. Record: README +
  `.docs/09_feature_catalog` §95 + `.research/540`.)

- **Issue 735 — Issue 734's laundering premise is bash-3.2-ONLY, and "and 5.x" was never measured** RESOLVED
  (2026-09-07; T0–T5 + T2b `a95d2bd6` + peers across 11 repos; instrument `scripts/trap_launder_premise_matrix.py`).
  Errexit abort entering the EXIT trap with `$?` 0 is **3.2-only** (4.4/5.0/5.2/5.3/dash/busybox preserve); 41 files
  corrected across 11 repos; errexit not nounset; mode matters (127 `bash -c` vs 1 script file). T3 answered by 737
  run `34137014037`: `macos-26-arm64` ships bash **3.2.57 ONLY** and reproduces all five LAUNDERS cells — sentinel
  load-bearing IN CI. T4 **do not pin — measure** (a `shell:` pin can't govern `#!/usr/bin/env bash`; probe re-measures every run).

- **Issue 732 — Fresh-z₀ breadth-restart arm + D-first law for `best_of_k_rollouts` (EqR RI axis): FreshZ0 is a decisive quality NEGATIVE; perturbation breadth pays from K=4 at every measured depth** RESOLVED
  (2026-09-07, `8777f6fc` T1 + `d8eae02b` T1–T4; Research 079 §10). T1 `restart_mode` (Perturb default bit-identical;
  FreshZ0 = seeded σ=4.0 z₀) behind `eqr_convergence`; rider: `MostFrequent` HashMap tie broke determinism → first-seen
  tie-break. T2 (20 trials × K ∈ {1,4,8,16,32}): FreshZ0 collapses 0.59 → 0.12–0.37, never beats K=1 — DDTree has no
  pull-back, EqR's restart premise doesn't transfer. T3 negative control VIOLATED (positive readings void). T4 (D ∈
  {2,4,8}): Perturb+MostFreq pays at K=4 every D, agreement 1.00 by K=8 at D=2. T5 deferred `[-]` (riir-ai Issue 881
  composition). Bench: `tests/bench_732_fresh_z0_restart.rs`.

- **Issue 731 — Residual-gated early exit for the weight-tied looped forward (EqR action item 7.2): `LoopResidualExit`** RESOLVED
  (2026-09-09; T1/T2/T3/T5/T6 landed, T4 deferred `[-]`. Lineage Plan 119 `eqr_convergence`, Research 079 §10. Opt-in
  `cadence_gate`, katgpt-core `convergence_cadence.rs` (T1 `c69e651d`; `forward_looped` alignment `c571d5b9`): exit when
  L=3 step-norm mean < τ OR cadence `Settled`, never before d_min, `None` bit-identical (Issue-035 precedent). T2
  `a5edd8e6` (pre-reg `c5f45402`): knee k=10, settle leads (~5-6), no τ qualifies at d_min=4, Research-440 control
  false-fired at τ=10 (boundary τ ≤ 3). T3 REJECTED v1–v3 (pre-regs `4332b056`/`284942d0`/`9c3b6d60`, closed
  `0fca1390`) — parity needs d_min ≥ ~10, 2× margin needs d_min ≤ K*/2; v4 (α=3.0, pre-reg `f7a12f5d`, measured
  `e05dc0c1`) G2 PASS at exactly 2.0× — existence-proof grade. T6 (pre-reg `ab59b9a0`, fix `14177e00`, pointer
  `30124225`): P1 REFUTED the 2.0× ceiling (seed 1002 2.40×); P2 1/12 in band → grade STANDS; P3 caught a PROBE DEFECT
  — the shape arm's rule-3 fall-through false-converged (control fired 40× on seed 1003) → `with_shape_persistence`
  (default 2 Settled windows; `persistence=1` kept as control), strict improvement (mean exit dist 8.66e-4 → 1.02e-4,
  control 40 → 0), confirmed on riir-ai Bench 887, `dd80296d`. T5 seam `with_cadence_config` `e562195d` (riir-ai Issue
  881 GOAT FAIL; Bench 875 `3e328368`). T4 DEFERRED — flows through riir-ai
  [Proposal 045](../riir-ai/.proposals/045_cce_margin_gated_commit_rule.md) (owner verdict pending). Closeout
  `6010a558`: a pin at `decay_ratio_max = 0.0` was ILLEGAL under the Issue-720 debug_assert (release-only pass) →
  swept to (0.1, false). Gates: cadence lib 2002/0, `issue_731_t1_residual_exit` 3/3, `bench_731_t3_heterogeneous_corpus`
  7/7 both profiles; `tests/bench_731_t2_residual_calibration.rs` + `tests/bench_731_t3_heterogeneous_corpus.rs`.)

- **Issue 733 — `EngramHotSwap::with_table` did not hold the writer lock: a nested same-thread `swap` dropped the old table under a live borrow** RESOLVED
  (2026-09-07, `31bf0012`; row + module doc durable; found by riir-chain Plan 046 §2b). Unsound under same-thread nested
  `swap` AND cross-thread `swap`. Fix: closure CASes and HOLDS the writer lock (panic-safe Drop guard); `swap` fails
  closed → `Err(new_table)`; nested `with_table` panics loud. Cost: one CAS + one Release store per closure. G5: 100
  swaps / 1.77M lookups / 0 torn reads. Contract tests pin all three. riir-chain Plan 047's `EngramIngress` is the
  first consumer: dispatch inside the closure, `swap` fails closed on contention.

- **Issue 730 — 256K prefill KV-offload double-buffer: T0 measured the wall 4× smaller than claimed; the offload premise is refuted for every lane the stack serves** CLOSED-N/A-PREMISE-REFUTED
  (2026-09-06; instrument kept: `scripts/gguf_header_audit.py`; riir-ai Issue 879 T2 next consumer).
  `Ternary-Bonsai-27B-Q2_0.gguf` (v3, `qwen35`, 851 tensors): **16 full-attention blocks of 64** (verified via tensor
  table: 16 `attn_output`, 48 `ssm_out`), 48 GDN; KV 4 KiB/token/layer @f16 → **64 KiB/token = 16 GiB @262,144 (f32: 32
  GiB)**; all-64-layer accounting reproduces R436's ≈67 GB — the wall never modeled the hybrid. `dspark-Q4_1` (6 blocks,
  **context_length 4096**, 48 MiB) has no wall. Verdict: refuted for every served lane (≤32K: 2 GiB f16 / 4 GiB f32);
  256K f16 fits (22.7 vs 23.98 GiB), f32 answered by `QWEN38_KV_DTYPE=f16` (riir-ai Issue 753, Bench 756-validated).
  Reopen: a real 256K lane AND f16 fit fails → T1. For riir-ai 879 T2: `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`,
  `ssm_norm`, RMSNorms **F32**; `ssm_alpha`/`ssm_beta` TYPE_142 ternary; TYPE_142 × 498, F32 × 353.
- **Issue 729 — the NaN-comparator class never got its katgpt-rs wave: ~160 legacy `partial_cmp` sites + 13 NaN-promoting `total_cmp` positions, in the repo that OWNS `float_order`** RESOLVED
  (T1–T5 2026-09-06, sweep `f2c305dd` + stragglers `649ce5fe` + cross-repo close `2bd3e704`; closeout residue sweep;
  row + `katgpt_core::float_order` module doc durable, riir-ai Issue-878 row the other half). **Group A** (~150 legacy
  sites / ~120 files) → float_order terminals; **Group B** (13 `total_cmp` in NaN-promoting positions — totalOrder
  ranks NaN above `+inf`; sharpest: ruliology `bandit::best_arm`) → `cmp_for_max`/`cmp_for_min` (+`_f64`), bit-identical
  on NaN-free input. No-dep crates keep local fallbacks; `total_cmp` stays correct for sorts and binary-search/`Ord`.

  **Closeout residue sweep — three classes.** (1) A wider multi-line selection scan found **~35 more Group B sites**,
  all converted: core `external_regret` ×2,
  `cgsp/dual_pool`, `mcts`, `slod`; forward `d2f` ×2 + `cluster_head` + `d2f_verifier`; kv `cs_kv_probe::argmax`;
  pruners `bandit` ×3 + `expression_pruner` + `sketch_population::best_elo` + `sketch_sampler` ×2; speculative
  `belief_drafter`/`blueprint`/`and_or_builder`/`adaptive`/`ilc`/`trd` + `dd_tree` ×10; and ruliology
  `bandit::best_unpromoted_arm` — **the twin of the issue's sharpest site, missed by `f2c305dd` in the same file (comment
  updated, loop not)**. All seven crates' lib suites count-identical (core 1974/0 · speculative 305/0 · pruners 126/0 ·
  dec 225/0 · ruliology 93/0 · forward 125/0 · kv 24/0). (2) `spechop/hop_tree` ×2 passed f64 to `cmp_for_max`
  (E0308) → `cmp_for_max_f64`, the `649ce5fe` class. (3) `.enumerate().max_by(|(_, a), (_, b)|` binds `a: &&f32` — the
  compile gate is the only detector. **Documented leaves (14, in-class):** test fixtures, `dec/heat_kernel` (guarded by
  `< NULL_SPACE_THRESHOLD`), `d2f_verifier::argmax_total_cmp` (by contract), micro-belief `coherence_bench`. riir-ai lane
  RESOLVED by riir-ai `.issues/878` (`1ee35da79`, ~140 sites incl. civ `map_tick`), which also fixed `649ce5fe` here.
- **Issue 728 — `silent_now_load_bearing` was 0 in all 16 repos because the classifier speaks ONE repo's dialect** RESOLVED
  (T1–T5, 2026-09-06; record `.docs/10_audits/cfg_gated_silent_zero_pass.md` §T4d/§T4e). T1 widened `is_load_bearing`
  (6 tokens + `LOAD_BEARING_BIGRAMS`, measured on 3,081 names) → **0 → 12**, two katgpt-rs's own (so the local wall had
  been green over a population missing them). T4 landed `cfg_gated_drift_sweep.py` + floors. **T2+T3 armed all 12 and
  ran each — 49 assertions, 49 pass** (riir-chain `81a00607` 25 · katgpt-rs 13 · riir-viewbridge `d6f18f5` 10 · riir-ai
  `5a6ac2d2e` 6 · riir-train `55754e7d` 5 · riir-dapps `dd6261e` 3, which armed all 25 of its silent targets).
  `silent_now` **261 → 225**, `load_bearing` **12 → 0** everywhere, `max_load_bearing` a **WALL**. **T5:** adding rows
  CAN red a gate counting green binaries — see §"A green test count can be a count of nothing". riir-ai's two
  `#[ignore]`d GPU parity tests were EXECUTED.

- **Issue 727 — SP-KV misses BOTH T16 bars once the gate is measured at a realistic sequence length** RESOLVED
  (2026-09-05 `adbc003d`; filed by 723 T7). Repaired instrument (the "50% pruned" arm had pruned 0/16 under
  `block_size = 16`): gate-bias +8.0/+8.1/+8.4% vs <3%, prune-skip 1.046/1.042/1.015x vs >1.05x. T2 hoist
  (`attention_head_core` → NoBias + hoisted GateBias, zero alloc): prune-skip **1.12–1.58x PASSES** at t_n 128/512/2048;
  bit-identity across 6×2 cases. **T1: "zero-overhead gate bias" is false** — restated as +7–12% (hd=4); `#[ignore]` kept,
  no bar re-pinned. `#[inline(always)]` cost a **1.66x layout penalty** on NoBias; `#[inline(never)]` split restored it.
  Record: `tests/bench_sp_kv.rs` + `katgpt-kv` `sp_kv` comment.
- **Issue 726 — `gauge_rebalance` is 3.7x its Plan 279 target; the rank-wise accumulate is scalar** RESOLVED
  (2026-09-05 `d225fffa`; filed by 723 T7). Scalar accumulate ~77–78% of the call. Swap to
  `katgpt_core::simd::simd_fused_scale_acc`: t08 **19.21 → 9.00 µs (−53%)** (G1 ≥20% ×3 PASS). Scalar loop was NOT
  FMA-contracted, so results move ≤1 ULP (1.19e-7); all exactness assertions pass, none loosened. `t08` re-pinned **30 →
  15 µs**; 5 µs target kept as aspiration (floor ~5.4–6.5 µs). Record: `tests/bench_270_gauge_invariant_goat.rs` +
  `gauge_invariant.rs`.
- **Issue 723 — the first full-workspace EXECUTION is red: 47 targets, six distinct classes** RESOLVED
  (T1–T8 2026-09-04/05; **G1–G6 ALL MET**). Doc-tests GREEN (34 suites / 98 passed); Class C resolved by measurement
  (`--all-features` unification, the Issue-830 twin, never re-pinned); Class E closed per-target. **"Class A's reds
  are partly the box" was wrong** — load topped none of 8 wall-clock reds; 5 were REPAIRED INSTRUMENTS (vanished
  denominator as 30x, non-matching arms, result-only black_box, in-clock regen, bar measuring the fixture). Rule:
  **repair the instrument first, decide disposition second** — three would have been re-pinned off by 5x/7x/140x.
  Shortfalls filed as `.issues/726` / `.issues/727`. Artifacts: `tests/common/ab_timing.rs` +
  `.docs/10_audits/ci_compile_vs_execute_axis.md`.
- **Issue 725 — the numbering gate covers ONE repo; 35 duplicates and 7 broken allocators sat in the other fifteen** RESOLVED
  (T1-T4c 2026-09-05). `scripts/numbering_drift_sweep.py` (+ `numbering_drift_floors.txt`) found 35 duplicates + 5
  malformed `.highwater` (`echo -n` writing its flag) + 2 stale. T1 split ABSENT/MALFORMED (`read_highwater()`). T3
  repaired all 7. T4a `scripts/citation_weight.py` (attributes ambiguous mentions by token overlap on a strict margin;
  `.plans/229` scored 0 under different spellings). T4b **riir-ai 6 → 0** (`.plans` 175→568, 182→567, 229→566,
  313→569; `.research` 020→362, 148→363; 86 rewrites), **riir-clippy 4 → 0** (Issue 069 `58e7c1d`), **riir-train 13 → 0**
  (Issue 514 `103ed351`; stale test filenames/comments remain). The author's own `513_` from a stale highwater was the
  first catch (→ 514). mmorpg-editor 12, READ-ONLY — ratchet. T5 deferred `[-]`.
  **Closeout catches:** riir-clippy 2 stale allocators (Plan 086, Research 137 — `4db7a18`) and riir-train `.issues/511`
  dual-allocated (census keeps 511 per AGENTS.md row + Issue 513 vs zero refs; genrm moved 511→518 — `e938cdc0`). Sweep
  PASSES: 12 duplicates (mmorpg-editor) · 0 stale · 0 malformed.
  **Next run (2026-09-06, 4090).** (1) Sweep exited 2 on Windows — pins read with cp1252 died on 0x86; now
  `encoding="utf-8"`. (2) riir-train showed 13 duplicates + `'-n 374'`/`'-n 441'` highwaters (`66193bac`) because the
  box had `main` checked out while the Issue-514 repair lived on `origin/develop` (90 ahead / 6 behind, tip `88a7f563`).
  **RESOLVED same day:** `1a6d128c` + `823d669c` merged into develop at `936977fa`; re-run **files=332 dup=0 stale=0
  malformed=0**; riir-train's AGENTS.md never claimed 'no develop' (that note was stale); `origin/main` frozen at
  `1a6d128c`. (3) katgpt-web / riir-dao / riir-deployer / mmorpg-editor uncloned — box-scoped absences.
  **Second 4090 pass.** (4) Sweep died mid-report — ✓ ✗ · → can't encode via cp874 stdout. (6) FAMILY-WIDE: replaced
  `encoding="utf-8"` with `staged_set_audit`'s `reconfigure(errors="backslashreplace")` in 21 scripts; validated
  natively. `tomllib` needs ≥3.11 (`py -3.14` works). A heredoc patcher rewrote `docs_gate_paths_sync.py`'s mixed EOLs
  (+164/−155) — redone byte-preserving (+9/0). Commit:
  (5) riir-mmorpg-examples `.issues/.highwater` `-n 97` → `097` (dir max `.issues/097_*`; now tracked); a same-unit collision with the M3's
  identical `adc8877` (push-wins, local reset, zero cost). Lessons: fetch BEFORE work on a repo not touched this
  session; Windows `Path.write_text` writes CRLF — use bytes or `newline=''` for allocators.
- **Issue 724 — `.plans/` numbering collisions regrew after a hand-sweep; nothing gated the allocator** RESOLVED
  (T2/T3/T4 2026-09-04 `24e349e9`/`28c353a1`/`322769b2`; T4b + T1/T5 closeout `866df2a7`). `449` resolved by CITATION
  WEIGHT (Poincaré kept 449 with 27 mentions; ActionBridge → `587`); allocators re-pinned (`.plans` 587, `.benchmarks`
  701). Gates: `scripts/numbering_gate.py` + `numbering_floors.txt` (duplicate, stale, population floor,
  tracked/untracked; five canaries) and `scripts/docs_gate_paths_sync.py` (two trigger `paths:` lists set-identical).
  T1/T5 moot (`075_riir_ai_m3_campaign_*.md` vanished). CHECKS now 10 (44 = 44). Record: docstrings +
  `scripts/numbering_floors.txt`.

- **Issue 721 — the root crate registers a `#[global_allocator]` as a library** RESOLVED
  (T1/T2/T4 2026-09-03; T3 2026-09-04, owner sequencing). Lib allocator now `cfg(all(test, debug_assertions))`; no
  downstream binary receives one. Consumers self-register — katgpt-rs via `tests/common/alloc_tracking.rs` (14 targets +
  kimi; 12 Issue-682 force-link blocks deleted), riir-train, riir-ai (~50 files; civ's `alloc_delta.rs` gained the
  liveness sentinel). Validated across all 4 repos (katgpt-rs 14/14, lib 203/0; riir-ai cgsp/evpi 392/0, poc 22/0, etc.).
  **A downstream `#[global_allocator]` is always legal now; the Issue-682 force-link pattern (`extern crate
  katgpt_rs;`) is dead — never reintroduce.** katgpt-rs pushed first (inverse order = duplicate registration).
  `riir-agents`' katgpt-rs dev-dep now unreferenced (owner call).

- **Issue 719 — conditioning-consistency audit PoC (`cond_audit`)** RESOLVED
  (T1, 2026-09-03, `995dea6d`) — opt-in `cond_audit`: paired forward (compressed vs full teacher) → per-junction KL →
  Pinsker `TV ≤ sqrt(eps_KL/2)` + flip counter + zero arm; KL via `stale_residual::kl_logits`. G8 PASS (12-nat
  corruption → tv 4.97, flips 8/8; control 0.0); G2 1.487× vs 4.0 budget. T2–T4 deferred `[-]` — reopen on semantic
  eviction PRs, riir-train Plan 343 T1.6, or Research 523 H2O. Record: `.benchmarks/700_cond_audit_poc.md`.

- **Issue 739 — katgpt-rs carried no rust-toolchain.toml: the box default
  failed to build HEAD (E0658 on katgpt-percepta `isolate_lowest_one`)**
  RESOLVED (T1+T2, 2026-09-08, `87dfa778`) — pins 1.98.1 (minimal + clippy/rustfmt); check green, core lib 1979/0,
  clippy clean. **Load-bearing half:** `dtolnay/rust-toolchain` doesn't export `RUSTUP_TOOLCHAIN`, so the pin would have
  frozen full_gate.yml's deliberate @stable rot lane — now an explicit job-level `RUSTUP_TOOLCHAIN: stable`; `test.yml`
  installs the pin via `rustup show`. M3 default drifted to 1.98.1 on 2026-09-04, so the break reproduces only on older
  boxes. T3: docs DONE; `rust-version` deferred `[-]` (no `[workspace.package]`, ~30 manifests; owner call).

- **Issue 767 — mb_value: bounded three-factor (dopamine) plasticity value circuit** RESOLVED + removed
  (2026-09-13; row + [Bench 761](.benchmarks/761_mb_value_goat.md) + git history durable). Mushroom-body class (riir-ai
  Research 380, from adonis-singh/TMNF-C @ `eb6be045`; Bennett/Nowotny 2021) as opt-in `mb_value`: sparse PN → quantile
  ReLU → top-k KC (`select_nth_unstable_by`, total order, G1-pinned) → approach-minus-avoid readout; learning is only
  `w ← clamp(w − η·code·RPE·compartment_sign, 0, w0)` — online × RPE × context × bounded, a quadrant nothing shipped
  covers. Bounds hold under ±inf/NaN RPE; calibration by measurement (action-gain bisection lands 0.4999 on 0.5);
  `w()`/`set_w()` freeze/thaw; no softmax; no connectome data ships. G1–G4 ALL PASS (Bench 761): r=0.9705 vs ridge floor
  0.9998 (margin 0.029 < 0.05); shift arm online 0.9444 vs frozen −0.9945; toy 2.2 µs (1,000 NPCs ≈ 4.4% of 20 Hz);
  fly 28.9 µs (2.28× full-sort), saturated 0.90×. Joined `linalg` any-list. OPT-IN pending consumer (Research
  380 §8).

## Issue 744 — HRM-Text second-pass modelless extraction queue: CLOSED as resolved-negative (2026-09-11)

8 candidates (Research 547 §Path 0) closed after a consumer hunt — **no graduating consumer**. Record:
`git log -- .issues/744_hrm_text_modelless_extractions.md`. #1 negative (no cu_seqlens analogue), #2 negative (no
normalize-then-gate incumbent), #3 negative katgpt-rs-scoped (shuffles are ablations; siblings unhunted), #4 negative
(no trunc-normal consumer), #5 single-consumer (`drafter_lora::make_lora_random`; `LoraPair` never constructs), #6/#8
want-gated, #7 `evolve_belief` riir-ai-side. **Re-file when a consumer materializes.**

## Issue 745 — Margin-gated verification escalation PoC (TriSpec distill): CLOSED as resolved-split (2026-09-11)

Plan 595 + Bench 711 (`a140c223`; `git log -- .issues/745_margin_gate_escalation_poc.md`). **Split, all 9 gates PASS:**
CASCADE mapping (margin over probe scores) REFUTED at ε=0.5% (sigmoid saturation + noise overlap → 3–19% false-flag
tail); ACCEPT mapping (margin over draft distribution) VIABLE — 50.5% invocation cut at 0.25% regression, tail=2%
control fires. Findings: (1) Research 548's "margin ABSENT" was a vocabulary miss (`SamplerFeatures.margin`, Plan 399
era); (2) trust polarity is workload geometry (`MarginPolarity` ships both); (3) meta-router `compute_reward` is blind
to the lossy tail; (4) a point-estimate mask starves a good arm (1.02% > ε → −8.8%) — z=3 bound fixes it; (5)
`margin_gate` OPT-IN pending the d2f draft-accept consumer (Research 548 §5). Counts 588 → 589.

## Issue 746 — Looped-Flows modelless extraction candidates: CLOSED, split verdict (2026-09-11)

Bench 712 (`git log -- .issues/746_looped_flows_modelless_extractions.md`). **Split, all gates PASS:**

- **Row 2 (marginal-calibrated backtrack): LIVE** — opt-in `marginal_rewind`
  (`crates/katgpt-core/src/marginal_rewind.rs`): rewind the current flow state at EXACT marginal variance (`a = s/t`,
  `sqrt((1−s)²−(a−s)²)`), γ-dial, no clean point. PoC (K=8192): calibrated beats additive-at-equal-budget **5.71×**
  (41.7% vs 7.3%), beats restart at lower budget when collapse-prone (26.1% @ σ=0.74 / 41.7% @ σ=0.90 vs 20.9% @
  σ=0.95); restart WINS clean-prior (50.3% vs 42.4%) — the consumer decision rule. Full-variance resample TIES at deep
  rewind — the mechanism is the de-commit shrink. Two harness bugs caught by the measurement.
- **Row 1 (anytime commitment schedule): CLOSED** — no time-grid consumer (tf_loop `DampedEuler`,
  `CommittedFieldBlend`, `set_diffusion_schedule` differ); the schedule (`r_i = Δt/(1−t_i)`) stays in the PoC harness.

Substrate-first: `renoise_ce`, `q_sample_step`, `saddle_escape`, `cgsp/dual_pool` checked across 18 repos — BUILD NEW.
Counts 589 → 590. Promotion owner-gated on a consumer (cgsp collapse recovery, fog-of-war re-exploration).

## Plan 596 — sliceTCA modelless slice-rank decomposition: COMPLETE + PROMOTED (2026-09-12)

Parallel subagent (coordinator pre-wired Cargo.toml/lib.rs/bench rows); Bench 714 is the GOAT record. **G1–G4 ALL PASS → DEFAULT-ON.**

- `slice_tca = ["subspace_phase_gate", "tucker_factorization"]`
  (`crates/katgpt-core/src/slice_tca/{mod,types,svd,als,rank,tests}.rs`): three truncated-SVD factorizations
  (`thin_svd_into`), closed-form covariability classifier with SIGMOID routing, deterministic-ALS demixer (frozen slice
  matrices), HOSVD init (**Tucker's first in-tree consumer**, Gram-SVD fallback beyond SVD_MAX_RANK=16),
  canonicalization with deterministic tie-breaks.
- vs the paper's SGD: bit-identical factors (BLAKE3-pinned), zero hyperparameters; fitting novelty NOT claimed.
- G1: routing 12/12 over noise 0→0.2; mixed [64,128,32] loss 0.0263 vs 0.4835 for both naive floors (18.4×). G2: ALS
  improves HOSVD 0.0305→0.0263; fit 29.99ms ≤ 50ms; 2-comp slice 0.375µs (4-comp 1.04–1.34µs, not gated). G4: 0 allocs.
- Held-out CV is structurally impossible for slice models; "blocked CV" is a structural-fit plateau grid. Counts 198 → 200
default-on (total 593; opt-in since ba754109).

## Plan 597 — BMR + EFE-over-models: COMPLETE + PROMOTED (2026-09-12)

Parallel subagent; Bench 715. **G1–G4 ALL PASS → DEFAULT-ON.** Unblocks riir-ai `.issues/925`.

- `bmr` (`crates/katgpt-core/src/bmr.rs`): `ln_beta`/`ColumnSums`, `Counts`, `bmr_log_evidence` (Eq 7/9),
`posterior_over_models`, `ModelSpace` sparse-Δ + `predictive_posterior_into` (Eq 11, one column recomputed),
`efe_model_gain` (Eq 10), `occam_log_bayes_factor` (Eq 12), `enumerate_isomorphic_rules` (Eq 14). Scope: discovery only.
- Dedup: Lanczos ln_gamma moved to shared `pub(crate) mod special_fn`, bit-identical (1992 tests).
- G1: vs brute-force oracle |Δ| = 7.1e-14; ablation (64 seeds × 40): full-info-gain 64/64 (Occam +36.8) vs
states+params 2/64 vs random 0/64 — the
paper's separation reproduced. G2: 1.92µs vs 983.8µs → **513×** (≥100×). G4: 0 allocs / 3000 calls.
- Deviations: (1) Research 551's Eq-7/9 sign flip corrected (oracle-arbitrated); (2) 81/81 rules vs paper's 79; (3)
priors tuned (λ=4, ã=4.0), never thresholds; (4) premature commits 0 in a deterministic world.

## 2026-09-12 — the citation gate's first CI run was a blind red (promote `35ac604f`)

The Issue 749 gate landed in a main-only CI window, so its first CI run was promote `c478ab9f..35ac604f` (267 commits)
— exit 2: `derived 1 contract repos < floor 15`. The refusal was correct, but a check added while its lane never fires
is **deployed but never exercised** (the `.issues/704` class, one level up).

1. **`issue_citation_gate.py` — CI-deferred verdict.** Under explicit `DOCS_GATE_CI=1` (never auto-detected) it checks
   local axes (pinned docs, `max_single_digit`, `min_citations_scanned`) and DEFERS cross-repo adjudication on its LAST
   line; the workstation run stays the verdict (this promote's was 17/17).
2. **docs_gate.yml — fourth trigger-omission instance.** Four checks (trap_sentinel 734 T9, citation 749, checks_sync
   750, markdown_fence 756) lacked `paths:` globs; added to BOTH lists + the env marker.

**required_features_touched REFUSED by design** — `84 selected > --max-rows 24` (run 33990209894 the 32-row
precedent); the 84 rows audited on the workstation with `required_features_build_audit.py --batch` — **0 FAILS / 0
NO-FEAT / 0 UNSEEN over 8 packages**. Other lanes green.

## Issue 755 — a quoted heading in a fence is not an allocation: CLOSED as resolved (2026-09-12)

Allocation path only, delegating to the canonical scanner (`620840ce` via `fenced_lines()`; `3a3d4bc1`
`skill_repo_set_gate.fenced_blocks()`; `37bb9cbf` CI-deferred wiring). Measured: `heading_allocated()` 0 of 57 fenced
(EXCLUDE); `citations()` 57 of 2972 fenced (do NOT exclude — they carry sibling attributions). Unterminated fences fail
SAFE (exit 2); 0 at landing. 3 selftest arms canaried (naive toggle wrong both ways). No verdict moved. Record:
`git log -- .issues/755_a_quoted_heading_in_a_fence_is_not_an_allocation.md`.

## Issue 756 — an unterminated fence swallows the rest of its file: CLOSED as resolved (2026-09-12)

19 files / 611 swallowed lines (5048 `.md`, 19 repos), all repaired same day: katgpt-rs 12 (`1bf768cd` +
`markdown_fence_gate.py`; `ed455885` walk widened to untracked `.md`), riir-ai 5 (`a1a205681`, `6e73d89c5`),
riir-train 2 (`d761a375`, `ca21b763` — missing-OPENER). The reported line is the DANGLING fence; three shapes (missing
closer / stray / missing opener). Verified 0 unterminated at siblings' origin/develop. Record:
`git log -- .issues/756_an_unterminated_fence_swallows_the_rest_of_the_file.md`.

## Issue 749 — a cross-repo `Issue N` citation rebinds to the WRONG document: CLOSED (2026-09-12)

AGENTS.md's bare `Issue 750` meant riir-ai's while the local highwater read 748 — rebinding two allocations later.
Resolved by `scripts/issue_citation_gate.py` + `scripts/issue_citation_floors.txt` (`e258fdaa`; `7fdfc554` aliases after
~50% FP; `60bc76aa` alias must OWN the number; `37bb9cbf` CI-deferred). Population reuses
`numbering_drift_sweep.contract_repos`; walk reads git history; floors on repos AND citations. 8 rows (riir-train Issue
513 ×3, riir-ai Issue 750 ×3, riir-ai Issues 490/493 ×2) qualified; revert-probed (list-tail probe exposed a head-only
scan hiding 493). 63 shared numbers reported, not gated. Spawned Issue 751; local 750 allocated one commit after —
hazard demonstrated, now held shut. Record: `git log -- .issues/749_cross_repo_citations_rebind_to_the_wrong_document.md`.

## Issue 751 — the cross-repo citation sweep (18 repos katgpt-rs cannot see): CLOSED as resolved (2026-09-12)

`scripts/citation_drift_sweep.py` + `scripts/citation_drift_floors.txt` (`d8041de5`): 19 repos / 35 docs / 2,939
citations; per-repo min_citations + max_cross + max_in_local_range + max_orphan; katgpt-rs row asserted against the
gate. FP rate **7/43 = 16%** (pre-752). Backlog worked to 0 CROSS same day. Record:
`git log -- .issues/751_cross_repo_citation_sweep.md`.

## Issue 752 — a repo name QUALIFIES a citation even when that repo does not own the number: CLOSED as resolved (2026-09-12)

Qualification never asked "does that repo own it?" — of 368 qualified, **45 unfollowable** (37 WINDOW_ONLY + 8
ADJACENT). `60bc76aa`: a directory qualifier counts only if that repo ALLOCATED the number. Revert-probed; floors
re-pinned (ten rose for the sharper instrument). The census's 0/45 was re-rated by Issue 754 to 1/45. Record:
`git log -- .issues/752_qualification_is_owner_blind.md`.

## Issue 753 — the citation rules' own COSTS were recorded once and never re-measured: CLOSED as resolved (2026-09-12)

Measured (`5ea1f40a`): (a) the `\d{2,4}` width bound is LOAD-BEARING — `\d{1,4}` makes 51 false heads (single-digit
`.benchmarks` section headings) at 0 true, over 19 repos; pinned `max_single_digit = 0` (exit 2), two-sided probe. (b)
40-char alias LEAD: widening buys 0 repairs, hides true findings (`chain` in `chain_viz` prose). One katgpt-web row
repaired, ceiling lowered. Record: `git log -- .issues/753_the_citation_rules_own_costs_were_never_measured.md`.

## Issue 754 — an allocation that exists only as a HEADING is invisible to allocated(): CLOSED as resolved (2026-09-12)

A file created and removed without a commit leaves only its `## Issue NNN (date, RESOLVED)` heading; under 752's rule
that blindness INVERTS into ⛔MISATTRIBUTED (riir-game-sdk's `riir-mmorpg-examples Issue 059`). `heading_allocated()`
(`8e0ffaa0`, four-arm selftest). 7 numbers recovered, 123 CROSS retired, Issue 752's census 0/45 → 1/45 — a census
inherits its oracle's blind spots. Record: `git log -- .issues/754_heading_only_allocations_are_invisible_to_the_file_walk.md`.

## Issue 761 — a Lean theorem can RESTATE its own definition: CLOSED as resolved (2026-09-12)

`e5d9836c` report, `3d5c87ab` verdict. Class: `theorem X_eq_sum : X = <X's own body>`, closed by `decide`/`rfl` for any
values. Four shipped in riir-neuron-db as "the `merkle_root` guard" — `lake build` green, axiom-free, counted by
`proof_gate.sh`, `proof_negative_test.sh` **17/17**. Removed by riir-neuron-db Issue 617 (`24957a2`, `387b4fc`).

**Criterion: symbolic equality over LEAF constants** — unfold composite nullary `def`s, keep numeral leaves symbolic;
unfolding to numerals would condemn every literal pin.

**CROSS-DEF is not a finding** (`commitmentOffset = RAW_PREFIX_LEN` pins two independent defs; perturbation reds it).
**HYPOTHETICAL split from STRUCTURAL** (199 of 255).

⛔ **The oracle found the instrument's defect:** at `24957a2^` it reported the right four but the caveat named
`Shard.zoneHashOffset` via ExperienceGraph's same-named def. Now scoped per module + imports; a 2-module fixture pins it.

**T1 changed shape when its blocker (Issue 750) lifted:** a CHECKS row would see one repo (katgpt-rs **1** composite
def vs riir-neuron-db 44) — reads as coverage. Landed as `restatement_drift_sweep.py` + floors. **A blocker lifting
is a prompt to re-derive the design.**

Proven: planted restatement moves 0 → 1 in all 4 repos; floor above measured REDS; absent repo REDS as UNSEEN; empty
floors REFUSES. `min_theorems` bites (a `:=` tokenized as `:` + `=` took the def table to 0).

Standing (2026-09-12): **0 RESTATEMENT-INLINE · 0 IDENTITY** / 4 repos / 68 `.lean` / 255 theorems / 284 examples; 1
CROSS-DEF. 19 UNRESOLVED + 6 conjuncts read by hand — all value pins. `[-]` T4 (`∧` conjuncts) deferred: 2 theorems,
both UNRESOLVED. Filing: `git show 3d5c87ab:.issues/761_restatement_theorem_class_has_no_verdict_half.md`.

## Issue 750 — docs_gate's CHECKS array vs the AGENTS.md table documenting it: CLOSED (2026-09-12)

Found drifted ("six" vs seven predicates, Issue 734 added one). `scripts/docs_gate_checks_sync.py` (`d10202b1`):
MEMBERSHIP both directions, never cardinality, plus QUANTITY WORDS; `MIN_ROWS = 10` floors both parses. Revert-probed
five ways; checks its own registration. 750 was allocated one commit after `e258fdaa` qualified the riir-ai citation
(Issue 749) — intended demonstration. Record: `git log -- .issues/750_checks_array_vs_its_own_documentation.md`.

## Issue 760 — mmorpg-remaster joined the workspace but not `scripts/repo_set.txt`: CLOSED as resolved (2026-09-12)

Filed from the 4090 (14 repos): `mmorpg-remaster/` (own BOUNDARY.md + .git) missing from the set. Fixed in `bfccffea`
(repo_set.txt + AGENTS.md §Repo count 19→20 + SKILL.md counts). M3 half: cloned at `/Users/katopz/git/mmorpg-remaster`
(`99064c5`); docs gate at `5773a614` **17/17** — the 4090 reds were partial-clone topology; the M3 derives exactly 20,
no regeneration. `markdown_fence_drift_sweep.py` caught its unterminated fence (`.plans/005_layer3_reducer.md:600`,
repaired `99064c5`), floored at 50 (`5773a614`). Record: `git log -- .issues/760_mmorpg_remaster_missing_from_repo_set.md`.

## Issue 757 — linking_fold detector Option B (the 50 ms @ n=2×1000 remainder): CLOSED as resolved (2026-09-12)

Evidence: Bench 717; Research 391 §6; arXiv:2606.31856. Single-pass k-NN (`select_nth_unstable_by`, squared
distances), longest-first witnesses, certified Gauss pruning (pruned ≡ full, 200 pins), Y-cycle grid, corrected
`max_cycles_per_cloud`. ≈31× (115.74 → 3.69 ms @ n=2×200, d=8); ORIGINAL budget RESTORED: G2b **28.28 ms ≤ 50 ms @
n=2×1000**, enforced. GOAT ALL PASS; lib 2035 unchanged. **KEEP OPT-IN** — `LinkingFoldCorrector` consumes the FOLD,
not the detector. Deferred: healer-surface consumer PoC; side-finding slice_tca release compile (Issue 741 class,
closed by Layer 6b). Record: `git log -- .issues/757_linking_detector_option_b.md`.

## Issue 747 — ASEntmax modelless mining (damping schedule, derived-k, eviction window, incremental entmax): CLOSED (2026-09-12)

arXiv:2506.16640 / Research 549: α-entmax needs **damping ∝ (log n)^{−0.5}**; our `entmax_1p5` had no length term.
Four primitives behind OPT-IN `asentmax_schedule` (katgpt-attn); Bench
713 + P1/P2/P3/P0.7 addenda:

- **P0** `AsentmaxSchedule` + `apply_asentmax_inplace` + `RollingSigmaEstimator`, zero-alloc;
  `EntmaxRouter::with_asentmax_schedule()`, `None` bit-identical to the Plan 106 path.
- **P1** k̂ = 4/Δ̂² — length-independent up to 1M; vs sigmoid SPLIT (sigmoid coverage, derived 4× cost efficiency).
- **P2** eviction window bit-identical (432 configs); 98.9–99.99% KV evictable at n=1M.
- **P3** incremental decode — 0.046 µs/step vs 68,448 µs/step at 512k; 0 allocs.
- **P0.7** (`ab0d79dc` + `603075f7`) real Ternary-Bonsai-8B re-gate (llama-perplexity 16.9614 vs 16.9778), 1856 rows:
  **NO modelless gain** — σ̂ = 0.1409, far below σ ≥ 1; support 4–7; needle 96.3%/96.3%; oracle-mass −0.017; latency
  +4–8%. **STAYS OPT-IN**. Harness `tests/asentmax_p07_realmodel_regate.rs` 6/6 (27B infeasible in-repo).

P4 deferred to Issue 762. Record: `git log -- .issues/747_asentmax_modelless_mining.md`.

## Issue 762 — ASEntmax P4 stretch (HoldConcentration, Kamath regime detector, per-head grid, RoPE cutoff): CLOSED (2026-09-14)

Origin: Issue 747's P4 deferral. All four tasks decided:

- **T0 long-context re-measure DONE** (2026-09-13, Bench 713 addendum; `asentmax_long_context_regate`, 5 tests over the 11,093-token / 173-block fixture): σ̂ climbs 0.14 → 0.35 but SATURATES below the σ ≥ 1 regime at every bucket (8 → 173) — the T4.3/T4.4 premise surface does not exist on real paths. Scheduled arm wins mean oracle-mass at every n ≥ 24 (+0.034 overall, +0.05–0.07 at n ≥ 64) and doubles deep-needle top-8 retention (59.3% vs 29.6%) — budget-confounded (support 2.1×).
- **T4.1 HoldConcentration DONE** (Bench 759 G1a): `SsmaxMode::HoldConcentration{c,k}`, exact finite-n multiplier `ln((n−k)c/(k(1−c)))/Δ̂`; Fixed/Adaptive bit-identical (re-pinned); 7–30 ns/call; ships in default-on ssmax (zero cost unless constructed).
- **T4.2 logit_regime DONE** (Bench 759): Kamath detector `ρ = Δ̂/(2σ̂√(2 ln n))` + normalized-entropy dispersion (katgpt-core, opt-in `logit_regime`); independent two-pass σ̂; Gaussian band [0.35, 1.15]; 8.2–8.6 ns/elem; 0 allocs.
- **T0.1 DECIDED (owner, 2026-09-14): option (a) — `asentmax_schedule` STAYS OPT-IN**; gain remains budget-confounded. Reopen: an equal-budget recall axis or the P1 derived-k controller must show the SELECTION (not the 2.1× support) wins.
- **T4.3/T4.4 CLOSED-deferred** — σ ≥ 1 absent at n ≤ 173; re-measure before building.

Issue file removed (`git log -- .issues/762_asentmax_p4_stretch.md`).

## Issue 772 — config-audit first pass over katgpt-rs (7 inert/assert-only knobs + orphan-report layer): CLOSED (2026-09-14)

First in-repo `--config-audit`/`--orphan-report` runs (riir-clippy Plans 124/126 instruments) filed 7 class-A/B knobs + an orphan-report layer (0 orphaned / 33 stillborn → S1–S6). All fixed same day, wire-vs-delete per finding:

- **A1/A2 WIRED** (`3c3c52ce`): monopoly `execute_turn` reads `GameConfig.max_jail_turns`/`max_doubles` (consts had shadowed the knobs); two wire-proof tests red under the old consts.
- **S1 `BranchRouter.tau_spawn` DELETED, not wired** — tests PIN spawn-on-no-snap (`route_returns_spawn_when_below_snap_threshold` expects Spawn at cosine 0.707); wiring would invert the [τ_spawn, τ_snap) band and break 4+ tests. Field + `DEFAULT_TAU_SPAWN` + `new()` 3rd param + re-exports deleted here and in riir-engine's `cognitive_branches_runtime` (riir-ai `8e748e936`).
- **B2 `use_ternary_gate` + `ternary_fusion_gate` DELETED** despite the issue's wire-preference: no weight source, a linear gate cannot encode `route_one`'s cascade, lane opt-in after its G1 FAIL (Issue 136), only test asserted the default `false`.
- **B3 `TransformerConfig.stop_token` DELETED** — vocab owns stopping (`test_predict_token_argmax`: config "halt" vs vocab "c").
- **B1 `BcConfig.anneal`, B4 `lod_adaptive`, B5 `quant_levels` (→ `new(quant_scale)`), S2 `fft_size`, S3 `injection_layer`, S4 `avg_bits_v`, S5 `solve_rate_floor/ceiling` DELETED**. S4 honesty fix: test_147 "Proof 7 asymmetric vs symmetric" was fiction (V path identical in both arms); relabeled as the K-bit sweep.
- **S6: 12 stillborn knobs DELETED** (`a3ef1830`): `DepthInvarianceConfig.magnitude_slope_collapse`, `HydraBudgetConfig.` `.cumulative_threshold`/`.modelless`, `CollapseDetectorFrozen.` `.budget_ema_mean` (the COLP wire struct in collapse_detector.rs is a different type, untouched), `InfluenceConfig.min_repetition_length`, `InfoNceConfig.default_critic`, `QbConfig.causality_strict`, `QueryFeatures.expected_output_len`, `SpKvConfig.predictor_lr_mult`, `TrdConfig.max_refinement_steps`/ `.refine_correct_branches`/`.elf_noise_scale`. Sibling grep verified zero consumers per field.
- **FP record (C2 trait-operational class)**: `ColinearityBatchGate`, `RuleBasedVerifier`, `EntropyConflictDetector` — reads inside `impl Trait for X` are operational; riir-train 546 added `Display`-impl reads. Harvested into riir-clippy `.distill/001` as FP class #3 (`eb35dfcd`).

Verification: clippy `--all-targets` per crate INCLUDING each module's feature arms (monopoly compiled to nothing under bare `--all-targets` and hid E0596); suites green (92 monopoly, 15 shard_kv, 11 flashar, 9 test_147, 2048 core, …); workspace `cargo check` 0 errors; riir-ai `cargo check -p riir-engine` green. Twin riir-train 546 landed same day (`b9f42028`).

Issue file removed
(`git log -- .issues/772_config_audit_inert_knobs.md`).


## Issue 777 — an instrument whose population is a FILESYSTEM walk audits code no repo owns: CLOSED (2026-09-14)

Found by running `percentile_drift_sweep.py` on the Windows box: both reds were walk artifacts.

**The class.** `percentile_index_audit.py` and `len_derived_binding_audit.py` walked the filesystem behind a hand-typed skip set, wrong three ways:

1. **Gitignored NESTED REPO.** `mmorpg-remaster/mmorpg/` (own `.git`): 1404 `.rs` credited to mmorpg-remaster (2015 walked vs 611 tracked), supplying 19 of 23 sites incl. the sweep's only finding, a TRUNC-VAR at `mmorpg/crates/mmorpg-bot/src/metrics.rs:129` — a defect at an address where repair cannot be made.
2. **Build artifacts under unnamed dirs.** riir-train's 48 untracked `.rs` are OUT_DIR sources under `.runs/target-release/`, `-cuda`, `-v2cpu`, `-bench`.
3. **Two fabricated floors**: riir-train `min_rs_files = 2500` vs 1129 tracked; riir-chain `500` vs **460** at creation (482 now).

**Second defect.** `percentile_index_audit.py:820` hard-coded `root = "/Users/katopz/git"`, so the documented no-arg invocation was a `FileNotFoundError` on every other box; the sweep imports the module and never calls `main()`.

**Repair.** Tracked-only had landed twice unshared (Issue 734, Issue 738 T3), so it lives in ONE file, `scripts/tracked_walk.py`, with an 8-arm / 11-assertion self-test (incl. a companion assertion that `SKIP_DIRS` still does not name the alternate target dir). Consumers: `platform_dead_code_audit.py`, `percentile_index_audit.py`, `len_derived_binding_audit.py`, `orphaned_attr_gate.py`.

**After.** Three instruments now report one population: **8694 tracked `.rs` over 16 repos**. Percentile figure 11,132 → 8,694 files, 110 → 124 sites (riir-ai's 252 vendored `wgpu-hal`, `mmorpg/`'s 1404, riir-train's 48 left). Floors re-pinned at ~65% (riir-train 2500 → 730, riir-chain 500 → 310). Sweep green on 16 present repos; the 4 absent are Issue 778's scope. `orphaned_attr_gate.py` verdict unchanged (2415 files, 7124 sites, 0).

Not repaired: `restatement_theorem_audit.py` and `suite_membership_audit.py` walk named subtrees, not repo roots. The `mmorpg` truncation is deliberately not filed from here (wrong-address defect again).

Issue file removed (`git log -- .issues/777_filesystem_walk_population.md`).

## Issue 778 — `subprocess.run(..., text=True)` decodes with the SYSTEM locale: CLOSED (2026-09-14)

Found when `citation_drift_sweep.py` crashed on the Windows box (`TypeError … got 'NoneType'`).

**Defect.** `text=True` decodes with `locale.getencoding()`; this box is **cp874**. Em dash `E2 80 94` → `0xe42 0x20ac 0x201d` under `text=True`, `0x2014` under `encoding="utf-8"`. Two modes: (1) silent mojibake — rc 0, a regex with an em-dash matches nothing, reads **zero findings**; (2) decode raises in the reader THREAD, `run()` returns `stdout = None` with rc preserved (`gate_says()` got `(rc=0, stdout=None)`). `PYTHONIOENCODING=utf-8` pins only the CHILD — both halves needed.

**Blast radius.** 28 sites / 16 scripts, zero explicit encodings, while `staged_set_audit.py` carried the correct form and a comment naming the defect since 2026-09-04.

**Repair.** `encoding="utf-8", errors="replace"` everywhere (`replace`: U+FFFD is visible, a raise is a new failure mode); `text=True` removed; three `sys.executable` spawns pin `env={**os.environ, "PYTHONIOENCODING": "utf-8"}`.

**Gate** — `scripts/subprocess_encoding_gate.py`, docs-gate check 19: ceilings 0 on DECODE and CHILD-ENCODER, pinned separately; floors 60 tracked `*.py` / 44 call sites; self-test every run. Text scanning reported four offenders in the gate's own selftest fixtures, so it scans the **AST** (UNPARSED reds). First run found a **28th** site: `.agents/skills/doc-sync/tools/linkcheck_sweep.py`.

**Verified.** `gate_says()` returns `(1, 266, 1)`; docs gate 19/19. The remaining partial-clone red is Issue 793.

CHECKS went 17 → 18 → 19 the same day, so the 18-check CPU figure will never be measured — write the CHECKS count beside timings.

Issue file removed (`git log -- .issues/778_subprocess_text_true_locale.md`).

## Issue 793 (allocated as 779; renumbered per Issue 791) — the workstation sweeps' partial-clone verdict, one copy: CLOSED (2026-09-14)

After the Issue 777 repair, **seven of eight sweeps FAILED with every content assertion green**, each on four absent-repo rows, on a known 16-of-20 box with `DOCS_GATE_PARTIAL_CLONE=1` set — a byte-identical copy-pasted `pinned but ABSENT` loop. `platform_dead_code_drift_sweep.py` did it right (post-Issue 765, `partial_clone_state()`). Third never-generalised instance in two days, so it became `scripts/sweep_population.py`, which the platform sweep also consumes.

**Three verdicts, never pooled:** UNREGISTERED (on disk, not in `repo_set.txt`; reds in EVERY posture) · UNSEEN (absent, no marker; never a pass) · DEFERRED (with marker; rides the FINAL line in both directions). Marker stays explicit opt-in. Seven self-test assertions incl. UNREGISTERED reddening under the marker.

**T3.** `gate_says()` could not parse the gate's partial-clone line (`partial: N citations scanned`) and exited 2; it now returns **-2**, a third state ("declined to adjudicate" ≠ "could not be read").

**Behind the reds** — four live Issue-749-class rows:

| repo | row | owner |
|---|---|---|
| katgpt-rs | `HISTORY.md:112` "Issue 513's T2 sweep" | riir-train — fixed `25b7bf6b` |
| riir-clippy | `HISTORY.md:6161` "Issue 150 removed" | mmorpg-editor — fixed `c5c30fee` |
| riir-train | `HISTORY.md:28` "Issue 671" | riir-ai (`a8c3aec4a`) — fixed `96041bf6` |
| riir-ai | `HISTORY.md:250` "Research 453 session" | riir-train — **deliberately untouched** |

The riir-ai row: HEAD already reads `riir-train Research 453`; a sibling's UNCOMMITTED edit removed the qualifier — recorded, not fixed. Both postures verified end to end.

Issue file removed (`git log -- .issues/779_sweep_family_partial_clone.md`).

## Issue 792 (allocated as 776; renumbered per Issue 791) — the docs gate's CPU self-timing printed a well-formed number that measured nothing on Windows: CLOSED (2026-09-14)

Full record in the session entry below (filed as 776). This heading exists so the number is READABLE by `issue_citation_gate.heading_allocated`, which reads `## Issue NNN (…) — title`, not bullets. Verified both ways.

## Issue 790 (2026-09-14) — an arm that exists and RUNS may still reach nothing: CLOSED (2026-09-15)

Quality is not statically decidable, but **reach** is measurable by execution; Issue 789 had measured it 53 times by hand (seven arms certified nothing).

`scripts/arm_reach_audit.py` (report) + `scripts/arm_reach_gate.py` (workstation verdict, ~200s): mutate outside arm bodies, re-exec, run the arm. Survivors pinned by MEMBERSHIP with reasons in `arm_reach_survivors_expected.txt`; wall 0 UNPINNED; `UNREACHED` / `NO-ARM` / `BASELINE` walled separately.

**Nine findings, all in already-green instruments:**

1. ⛔ Harness exec'd into a bare dict, so `dataclasses` failed — **seven classifiers CRASHED unmutated, 796 of 2382 mutants**. ⚠ Premise (CPython ≤3.12 guards, 3.14 not); self-test asserts registered-namespace exec and prints the side.
2. ⛔ `BASELINE` missing: an arm failing unmutated reports 100% reach and **inflates** `MIN_KILLED`.
3. `required_features_touched_gate.selftest` Windows-broken since written.
4. ⛔ Gate caught its own change: bucket decision extracted to pure `classify(row)`; first repair wedged on `git ls-files` (17 min at 0.02s CPU), replaced by injection.
5. ⛔ A mutant can hang (13-min run burned TWO HOURS); interrupting credited KILLED via `except BaseException`. `TIMEOUT` verdict, deadline derived from baseline (10×, floor 30s).
6. ⛔ `required_features_build_audit` used POSIX-only `os.statvfs` in production; no other site in 16 repos.
7. NO-ARM vs UNREACHED unified into one `has_runnable_arm`.
8. ⛔ Blocking-C-call wedge = the BOX (167 orphaned `git.exe`); thread + `interrupt_main` watchdog cannot reach it.
9. ⚠ Starved box → confident WRONG verdict (`bench_doc_audit` `97 labels, 56 mismatches`, 0 on re-run); resolved 2026-09-15, inference wrong twice — see `BlindRead`'s docstring.

**T5 DECLINED on measurement**: only **2 of 17** sibling arm-bearing scripts admissible.

`--include-all` **55 of 55**: 2376 mutants · 1182 KILLED · 652 live · 535 exempt · 1 CRASHED · 6 TIMEOUT · 1 NO-ARM. ⚠ The 652 is an unread backlog (Issue 785 forbids ratcheting). Run module by module under an external timeout.

⛔ **Pattern:** CLASSIFIERS well armed, **VERDICTS** not; a third of the 47 "EQUIVALENT" survivors were plain functions — real gaps. **Writing the reason is the adjudication.** 47 → 27 → 26.

Follow-through (2026-09-15): `feature_isolation_gate` 4 → 24 killed, `citation_weight` 3 → 12 → 19, `ci_gate_coverage` 4 of 74 → **54 of 73** — each needed an EXTRACTION or INJECTION first.

## Issue 791 (2026-09-15) — three numbers allocated twice across a 57-commit divergence: CLOSED

Two sessions both allocated **776, 779 and 780** from `.issues/.highwater`; rebase `max(ours, theirs)` makes double-allocation invisible.

**T1.** `citation_weight.candidates()` saw only on-disk files, blind to the majority case (one side closed). `removed_candidates()` recovers from `git log -M --diff-filter=D` (`-M` load-bearing vs renumbers). ⛔ Still blind to create-and-remove-uncommitted (Issue 754's spot; `numbering_drift_sweep.heading_allocated`). ⚠ New `of which HISTORY.md:` column, reported apart, never subtracted (Issue 724 T2's rule prices moving a number).

**T2 — this session moved all three:**

| number | weight | outcome |
|---|---|---|
| 776 | theirs 16 vs mine 5 (+13 / 21) | mine moves by rule |
| 779 | **mine 13** vs theirs 8 (+5 / 21) | mine moves anyway |
| 780 | mine +4, UNRESOLVED **10** > decided 4 | tool DECLINES; mine moves |

⛔ 779: weight can't see that moving theirs needs another session's agreement; mine was closed, theirs live — coordination-free beats a 5-site lead.

50 lines rewritten (`scripts/**` wholesale, 31 citations verified; `AGENTS.md`/`HISTORY.md` by line; `crates/**`, `benches/**`, `.research/**` untouched). Headings carry `(allocated as NNN; renumbered per Issue 791)`. ⚠ Instance lists (`Issues 777, 778, 793, 782, 783`) kept CHRONOLOGICAL, not sorted.

**T3 deferred** pending an FP-rate measurement (long-lived branches allocate ahead; divergence was 57 commits). Procedural half: fetch before allocating.

## Issue 797 (allocated as 796; renumbered — the other session allocated 796 the same hour and pushed first) (2026-09-15) — a sweep reads the WORKTREE, so a finding may exist in NO commit: CLOSED

The last standing CROSS row, carried as *"blocked"*, was actually **nothing to fix**. Every sweep walks the **working tree**, shared by five-plus sessions.

**Measurement** (`citation_drift_sweep.audit()` worktree vs HEAD blobs, 16 of 20 repos):

| repo | dirty tracked | in scope | worktree | HEAD |
|---|---|---|---|---|
| riir-ai | 6 | `HISTORY.md` | **CROSS = 1** | **CROSS = 0** |
| mmorpg-remake | 1 | — | — | — |

The entire CROSS finding was an artifact (HEAD reads `riir-train Research 453`). Population moved too: `n_cites` 601 vs 607, `ambiguous` 162 vs 163.

**Two directions**: UNCOMMITTED (false accusation) and **MASKED** (false green, committed defect hidden) — MASKED measured **0**, not absent as a class.

- **T1** `scripts/worktree_state.py`: `dirty_files()`, `head_text()`, `split_rows()`, `dirty_in_scope()`, `sweep_advisory()`, 36 arms.
- **T2** wired into all **16** sweeps at `population_verdict()`, own-population globs; ADVISORY only. Landing run: `*.rs` riir-ai (3), `*.md` riir-ai (1), `numbering` katgpt-rs (1) + riir-ai (1), `subprocess_encoding` katgpt-rs (16), `trap_sentinel`/`restatement` silent.
- **T3** row-level split via injected `read(path) -> str | None`; DISPLAY reads worktree, PINS read HEAD.
- **T4** AGENTS.md section; docs gate 21/21; arm-reach green.

Own-code errors fixed:
- ⛔ `\` → `/` normalisation arm certified nothing — `git status --porcelain` is POSIX everywhere (measured on Windows); arm asserts git's output shape.
- ⛔ `("*.rs")` is not a tuple — matches everything; **8 of 15** call sites had it. Helper coerces.
- ⛔ Row key must be LINE-FREE.

Arm reach: **19 of 23** (survivors are fixture-builder kwargs).

**T5 — MEMBERSHIP gate**, because "all sixteen" was stale within two hours (`pipefail_discard_drift_sweep.py`, `toolchain_override_drift_sweep.py` landed unwired). Seventh never-generalised instance (777, 778, 793, 782, 783, 789), first repaired mechanically: `scripts/sweep_advisory_membership_gate.py` reds on a `*_drift_sweep.py` calling neither `sweep_advisory()` nor `worktree_advisory()`. Exemptions reasoned, file EMPTY, stale pins red. ⛔ First run flagged `citation_drift_sweep` UNWIRED (predicate named one of two entry points). Standing: 18 sweeps wired, floors 15/15. ⚠ Asserts the CALL, not that patterns match the population.

**T6** — sibling's new `dual_allocation_gate` was UNREACHED (**0 killed of 28**; `selftest()` delegated to the probe module; delegation cannot reach the consumer — Issue 775). The reaching arms existed as `prove_fires()`, which `arm_reach` never invokes; `selftest()` calls them now (1.45s vs the 80.2s/4.4s `git archive` cost basis). Reach **0 → 11 killed**; 12 survivors adjudicated (11 fixture kwargs, one `or`→`and` EQUIVALENT by measurement).


⚠ Its own number collided: the other session's 796 pushed first, and `dual_allocation_gate.py` replayed on the real divergence printed `⛔ INDEPENDENT 796`, exit 1. ⚠ Green after rebase; `numbering_gate`'s Issue-795 wall caught the residue, removed by rewriting two unpushed commits.

Issue file removed (`git log -- .issues/797_worktree_state_sweep_findings.md`).

## Issue 795 (2026-09-15) — 70 numbering collisions the gate could not see, 9 of them live: CLOSED

Issue 791 recorded three; a full scan says **70** over 1374 numbers (`.issues` 51 · `.research` 10 · `.plans` 9 · `.proposals` 0).

⛔ **791's "three" counted what the instrument could see.** Both-closed collisions leave nothing on disk — the MAJORITY case; six of nine recent ones were invisible, and `numbering_gate`'s tracked-duplicate wall had never fired. All nine ≥ 700 (741, 775–782) are one divergence event.

⛔ **First scan over-reported by 52**: `.benchmarks/` families per owner are intended, as `numbering_floors.txt` records. **A tree-derived population is not the governed population** (74% inflation).

**Two regimes** (`scripts/number_collisions_expected.txt`):
- **≥ `era_boundary = 700`: WALL by MEMBERSHIP**, reason per row, reds both directions.
- **Below: RATCHET**, counted, never pinned (61 pre-gate rows, `.issues/121` era; Issue 785). A drop is a note.
- Boundary measured: highest legacy 575, lowest divergence 741.
- Two blindness floors.

⛔ **Six of nine NOT renumbered**: leads +1 to +5 over 11–36 decided sites, 21–53% UNRESOLVED, one `TIE_FRACTION` tie (778), one DECLINED (775). Margins on each pin row.

Allocation-time gate DEFERRED on its FP rate; the wall catches the collision on the MERGE commit.

⛔ Both this gate's and `citation_weight`'s arms reached NOTHING until moved from `main()` into `selftest()`. Reach: `numbering_gate` 7 of 48 → **22 of 47**; `citation_weight` 12 of 49 → **19**.

791 and 795 were found by pointing an instrument at one more case — *a census is exhaustive over ROWS, not over its ORACLE* (Issue 754).

## Issue 796 — the allocation-time dual-allocation gate: the FP rate measured, the gate built CLASSIFIED: RESOLVED (2026-09-15)

`scripts/dual_allocation_fp_probe.py` measured the FP rate from reflog-reconstructed (local_tip, upstream_tip) pairs, union-sampled over both timelines (local-only sampling missed the target RED). 18 repos, ~90 d: **634 divergent pairs, 7202 one-sided (green BY CONSTRUCTION), 164 RED pair-instants = 39 incidents = 31 TWIN + 8 INDEPENDENT.**

Split by filename STEM (subject-equality mislabels). The 8 INDEPENDENT: riir-ai 722/780/935, riir-chain 30+72/34, riir-clippy 79/83, mmorpg-editor 192–195.

Verdict: fear MOOT, caution RIGHT (naive gate cries wolf 31/39). `scripts/dual_allocation_gate.py` CLASSIFIED: TWIN exit-neutral, INDEPENDENT exit 1 naming both adding commits. `--prove-fires` fixture caught the probe's VOID first sweep (a `--.issues` pathspec typo; 105 pairs measured nothing). Reach limit: the 791 divergence lived on the wire, not in this box's reflogs. Joined docs_gate CHECKS (796 T5) with table row + count note in one commit; green-exits in CI by construction, live reach is the workstation loop.

## Issue 794 (allocated as 780; renumbered per Issue 791) — a wrong address reads as UNDECIDED when its number is in local range: CLOSED (2026-09-14)

`⛔MISATTRIBUTED` (Issue 752's `is_qualified()`) was gated on `cls is CROSS`, but bucketing runs first (`cls = (IN_RANGE if n <= top[kind] else CROSS if owners else ORPHAN)`), so an explicitly misattributed citation under the citing repo's ceiling landed in **IN-LOCAL-RANGE** — never counted or gated.

Such a row refutes IN-LOCAL-RANGE's premise (Issue-754 oracle found no local allocation *and* another repo is named on it). **Followable, to the wrong place** → own class `MISATTRIBUTED-IN-RANGE`, kept in IN_RANGE for `max_in_local_range`, printed first, never truncated.

**The hidden row was AGENTS.md's own example**: riir-train Issue 513, written up as `katgpt-rs Issue 513`, at riir-neuron-db `AGENTS.md:82`. katgpt-rs never allocated 513 (`git log --all -- '.issues/513*'` empty); riir-train owns it (`513_required_features_rows_are_unverified.md`, filed `389a0a6b`, removed `5a4265df`); riir-neuron-db tops at 617. Repaired to `riir-train Issue 513 T6`.

**Boundary measured** (16 repos, 3,392 citations):

| bucket | rows | hand-read |
|---|---|---|
| `n <= top`, not allocated (IN-RANGE) | **1** | 1 true, 0 false |
| `n in mine` (locally allocated) | **19** | **0 true, 19 false** |

The 19 are contrast prose (riir-chain's ``riir-ai Issue 853 / this repo's Issue 093``, riir-mmorpg-examples' ``riir-ai Issues 574/589/537/672 + local Issue 059``, ``in `riir-neuron-db/src/local_kv.rs` (Issue 043``, ``at `riir-game-sdk/crates/riir-games-cluster/`. Plan 010``) — a local number has a referent to contrast. Rule stops at IN-RANGE. ⚠ n = 1 is not a rate.

**T2 NO-OP**: `issue_citation_gate.py` has no IN-RANGE bucket; `gate_says()` already pins the partition. The leniency was labelling + per-repo ceiling in the 15 ungated repos.

Ceiling: global wall `max_misattributed_in_range = 0`; missing pin refused (exit 2). Four self-test arms incl. the 19/19 exemption; live corpus proven both directions.

⛔ DRY defect caught by its canary: two separate `cls is IN_RANGE and bad` tests → one predicate.

Issue file removed (`git log -- .issues/780_misattributed_is_computed_only_in_the_cross_bucket.md`).

## Issue 781 — the heading oracle matches one house STYLE: CLOSED as a measured, printed blind spot (2026-09-14)

`heading_allocated()` (Issue 754) anchors the parenthetical immediately after the number: `## Issue NNN (date) — title` reads, `## Issue NNN resolved — title (date)` does not. **64 of 152 read, 88 unread**, split by house style:

| repo | shaped | read | unread |
|---|---|---|---|
| riir-mmorpg-examples | 43 | **43** | 0 |
| mmorpg-remake | 15 | 14 | 1 |
| katgpt-rs | 29 | 7 | **22** |
| riir-ai | 25 | **0** | 25 |
| riir-clippy | 25 | **0** | 25 |
| riir-train | 13 | **0** | 13 |
| riir-game-sdk | 2 | 0 | 2 |

katgpt-rs's newest closes (777–781) are unreadable by its own instrument.

**Closed as a REPORT**: arm 2 pins `## Issue NNN follow-up (date)` as a NEGATIVE; `follow-up` and `resolved —` are the same shape, so a widened pattern failed that arm on the live workspace — and this is the only path that can **suppress** a finding.

Cost printed every run (`heading_unread=N/M` + workspace line), foreign-repo filter applied to both sides; self-test arm (042 reads · 043 style loss · 044 foreign rejection), canaried. Snapshot lives in the function docstring.

Consequence: incomplete local set → UNDECIDED noise (riir-clippy's 10 rows are its own four numbers); incomplete owners set → false `⛔MISATTRIBUTED` (Issue 754's failure, inherited by Issue 794). 0 live.

⛔ **The Issue 794 write-up introduced four rows of its own class** (three quoting riir-train Issue 513 under katgpt-rs's name; one riir-chain's ``riir-ai Issue 853 / this repo's Issue 093``). Repair: name the true owner in the 3-line window, or write `NNN` — not a pin. This section's first draft tripped it too (one MISATTRIBUTED-IN-RANGE + one UNDECIDED), fixed by naming owners.

Issue file removed (`git log -- .issues/781_the_heading_oracle_matches_one_house_style.md`).

## The x86_64-pc-windows-msvc axis, measured: clean at all-features/all-targets (2026-09-14)

A third platform axis nothing had compiled (`full_gate.yml` is macOS/aarch64, `wasm32_gate.yml` wasm32). rustc 1.98.1, host `x86_64-pc-windows-msvc`: `cargo clippy --workspace --all-targets --all-features --keep-going -- <the AGENTS.md -D list>` → exit 0, 32 packages, 8m17s, **zero code findings**. All 1,034 warnings are the NTFS `hard linking files in the incremental compilation cache failed` message.

Scope: compiles the `not(target_os = "macos")` half (`scripts/check_platform_gated_modules.sh` only typechecks it from the M3), but `target_os = "macos"`, `target_os = "linux"` and wasm32 arms compiled to nothing; dev profile, `debug_assertions` ON. One cell added, matrix not closed.

## Issue 782 — a pinned repo absent from the walk is never visited: CLOSED (2026-09-14)

Issue 793 fixed seven sweeps; the three left alone were **all non-exempt** — the census selected on symptom.

| sweep | what it had | why it looked clean |
|---|---|---|
| `cfg_row_implication_drift_sweep` | **no absence check** | confident green over 16 of 20 |
| `docs_drift_sweep` | copy-pasted loop | its 8 repos all present |
| `restatement_drift_sweep` | copy-pasted loop | its 4 `.proofs` repos all present |

The first is the live defect: it iterates derived repos, so pins→walk is unchecked — `katgpt-web`, `riir-dao`, `riir-deployer`, `riir-esp32` rows in `cfg_row_implication_drift_floors.txt` were evaluated by nothing under `✓ … PASSED`. It survived the 779 census because it was quieter.

All eleven share the verdict; both postures verified per repair.

⚠ **A subset sweep has TWO populations**: `population_verdict` needs the **contract walk**; handed the `.proofs` subset it reported **16 phantom rows**. The hole the shared verdict cannot see — pinned, present, dropped out of the subset — is a per-sweep `⛔ DROPPED` check, now in the restatement sweep.

Issue file removed (`git log -- .issues/782_a_pinned_repo_absent_from_the_walk_is_never_visited.md`).

## The executed-test gate on x86_64-pc-windows-msvc: exact floors, all three rows (2026-09-14)

`scripts/test_gate.sh` claims platform-invariant floors; the weekly schedule was suspended since 2026-09-09, so it had one platform. Measured here (1.98.1): katgpt-rs `--lib` 203/203, katgpt-core `--lib` 2041/2041, katgpt-dec `--lib (pca_global)` 249/249 → `test_gate: PASS`. **Exact on every row** — execution, not compilation. Scope unchanged: default features, dev, scoped core; 477 integration-test and 176 bench targets still executed by nothing automatic.

## Issue 783 — the subprocess-encoding gate is katgpt-rs-only: CLOSED (2026-09-14)

`scripts/subprocess_encoding_gate.py` (Issue 778) shipped with no sweep half — a skipped step; `scan()` already took a repo path.

**2026-09-14, 16 of 20 repos:**

| repo | tracked .py | subprocess calls | DECODE | CHILD-ENCODER |
|---|---|---|---|---|
| riir-train | 68 | 15 | **12** | **1** |
| riir-clippy | 5 | 12 | **9** | **1** |
| riir-ai | 8 | 8 | **6** | 0 |
| riir-dapps | 1 | 1 | **1** | 0 |
| mmorpg-editor | 2 | 2 | **1** | 0 |
| katgpt-rs + 10 others | 79 | 44 | 0 | 0 |
| **total** | **163** | **82** | **29** | **2** |

Seventh instance, seventh finding (list in `markdown_fence_drift_sweep.py`'s docstring). Two not latent: `riir-clippy/scripts/gen_dashboard.py:552` reads em-dash commit subjects across siblings; `riir-train/scripts/plan344_phase0_full_bandwidth.py:318` reads a `git ls-files` list and opens the paths.

All 31 repaired; ceilings **0/0**, a wall (a two-token repair earns no backlog).

**Both floors**: `min_calls` is **0 in 10 of 16** repos, so `min_py_files` is the only blindness detector there. katgpt-rs's pair asserted equal to `subprocess_encoding_gate.FLOOR_PY_FILES` / `FLOOR_CALLS`, canaried.

Canaries: planted DECODE in riir-dapps reds; marker-off reds UNSEEN on the four absent repos marker-on DEFERS; seven self-test arms (walk-boundary arm on a `git add`-ed temp repo — unstaged exercises the rglob fallback, the Issue-775 vendor-arm failure).

katgpt-web, riir-dao, riir-deployer, riir-esp32 deliberately unpinned (not on box; UNPINNED-red on first full checkout), same as `platform_dead_code_drift_floors.txt` — pin both in one commit.

Drive-by: `SyntaxWarning` on `riir-train/scripts/bonsai_vs_gemma_codegen.py:16` (backslash-escaped backtick in a non-raw docstring, future `SyntaxError`); one instance in 163 files — repair, not class.

**Standing failure mode, five times** (Issues 777 tracked-walk, 778 itself, 779 sweep-population, 782 the three quiet sweeps, 783 this one): a rule landed in one instrument and never generalised — grep the family and land one shared mechanism.

Issue file removed (`git log -- .issues/783_the_subprocess_encoding_gate_is_katgpt_rs_only.md`).

## Issue 784 — the orphaned-attr gate's cross-repo claim was hand-run: CLOSED (2026-09-14)

`scripts/orphaned_attr_gate.py` was the last cross-repo CHECK without a sweep half; eighth instance.

**First of the eight with no new offenders**: `max_offenders` 0 across three measurements and two population definitions. It found a stale **warrant**: the docstring's hand-typed total did not follow Issue 777's tracked-walk migration (`820bf8b6`):

| quantity | docstring (2026-09-06, filesystem) | measured (tracked) | drop |
|---|---|---|---|
| `.rs` files | 11,132 | **8,694** | 22% |
| outer-`#[cfg]` sites | 49,624 | **26,598** | **46%** |
| orphaned | 0 | **0** | — |

**23,026 sites** were in unowned trees (`mmorpg/`, vendored `wgpu-hal`, `.runs/target-*`). Two live copies found — `orphaned_attr_gate.py:38` and `.docs/10_audits/percentile_index_tail_support.md:121` — both repaired with the pre-777 number dated, the new number beside it, and the mechanism named.

**The structural repair** is `orphaned_attr_drift_sweep.py`: the total is now MEASURED, not hand-typed.

**Both floors live in all 16 repos** (smallest, riir-viewbridge: 24 `.rs` / 20 sites) — opposite warrant to Issue 783, stated in both pin files. `OUTER_CFG` vs `ANY_ATTR` imported, takes **2,044** broad-shape sites to 0; selftest arm 3 pins it.

Canaries: planted orphan in riir-shader's `camera.rs` reds naming attribute + item; marker-off/on postures; `min_cfg_sites` +1 trips the shared-floor assertion vs `orphaned_attr_gate.FLOOR_CFG_SITES`. Seven arms, `git add`-ed walk-boundary fixture.

Four repos unpinned; THREE floors files now owed one repair: `platform_dead_code_drift_floors.txt`, `subprocess_encoding_drift_floors.txt`, `orphaned_attr_drift_floors.txt`.

Issue file removed (`git log -- .issues/784_the_orphaned_attr_cross_repo_claim_is_hand_run_and_stale.md`).

## Issue 785 — the wasm32 surface audit had no verdict half: CLOSED (2026-09-14)

`scripts/wasm32_surface_audit.py` was cross-repo with **no verdict at all**; standing lived as a hand-typed AGENTS.md sentence.

**Why a wall**: an UNCOVERED package never compiles and nothing says so — mmorpg-remake's wasm32 block (`.issues/010` T2, katgpt-rs Issue 738) was uncompilable from day one.

**T1: one classifier.** Buckets were computed inline in `main()`; the 738 resolver + 774 path-dep closure separate **25 NAMED from 17 false UNCOVERED**. Extracted `RepoSurface` + `classify_repo()`; verdicts byte-identical (25 NAMED · 2 BY-DEP · 0 UNRESOLVED · 1 UNCOVERED over 213 files / 28 packages / 16 repos); sweep selftest invokes the canary.

| bucket | pinned as | why not the others |
|---|---|---|
| UNRESOLVED | `max_unresolved = 0`, WALL | a ratchet on *unanswered* is a backlog; reached 0 by answering (738 T1: 15 → 0) |
| UNCOVERED | **membership**, `scripts/wasm32_uncovered_expected.txt` | a count goes green on a swap |
| population | per-repo floors **+ global `TOTALS`** | `min_files`/`min_packages` 0 in 7 of 16 repos; a fully blind instrument passes every per-repo floor |

Not hypothetical: a Python `\s` in a `git grep -E` pattern once gave a **0-file** walk with full buckets. Missing `TOTALS` row → exit 2. Membership pin reds both directions.

Canaries: remove pinned row → `⛔ NEW`; add covered package (`riir-shader-core`) → red; `TOTALS` min_files 500 → red; delete `TOTALS` → exit 2; restored rc=0.

`mmorpg-remaster: mmorpg-poc-submodule` is the single pinned row — a NEGATIVE CONTROL proving by-dep credit is no amnesty.

Four repos unpinned; canonical 20-repo figure 216 files / 29 packages vs 213 / 28 here. **Four** floors files owed one visit — `platform_dead_code`, `subprocess_encoding`, `orphaned_attr`, `wasm32_surface`.

Every cross-repo class **walled at a small number** now has both halves; 783/784/785 were the same defect.

⚠ **Not "every cross-repo class"**: `suite_membership_audit.py` is report-only by design — **1,203 load-bearing unpinned rows across 15 repos** (katgpt-rs 442, riir-ai 410, riir-train 223, … of 3,036 targets); same argument as `staged_set_audit.py` and `highwater_contiguity_audit.py` (reset verdict in `numbering_drift_sweep.py`). 783/784/785 were gateable because their ceilings were already an earned 0. `gguf_header_audit.py` is not a class audit.

Issue file removed (`git log -- .issues/785_the_wasm32_surface_standing_figure_is_asserted_by_nothing.md`).

## Issue 800 Arm C complete — GraphStablePool site re-points: 1 landed, 1 N.A., 2 declined on evidence (2026-09-16)

Phase 1 (`877e06eb2`) extracted the type; re-points (`bdb1091a4` katgpt-rs, `35108b6a7` riir-ai) settled four sites, verdicts in the `graph_stable_pool` module doc tables:

- **Site 1 `radix_prefix` — RE-POINTED.** `RadixPrefixTree.nodes` → `GraphStablePool<RadixNode>`; inline free stack gone; eviction drops node buffers eagerly. Pool gained `iter()` for the three arena scans; zombie filter structural. `radix_prefix_cache` implies `katgpt-core/graph_stable_pool`. bench_762 GOAT release PASS (G1 bit-identity + stability; G2 hit-rate 0.184 vs flat 0.075, match 0.22 ms vs 2.17 ms; G4 0 allocs), 14/14 + pool 7/7, clippy `-D warnings`, default + wasm32 clean. LIFO order reproduced exactly.
- **Site 2 `PagedKVCache` — DECLINED.** Refill-in-place at stable index is load-bearing (DDTree rollback; riir-engine forward_paged G4); a re-point needs a buffer stash outside the pool (two structures for one). `pages`/`free_pages`/`page_ref_counts` are `pub` instruments for bench_414's oracle; rollback already prevents use-after-free.
- **Site 3 `Qwen38LaneSet` — N.A.** No free list, fixed `n`, allocate-once arenas, graph cache inside the set — already the discipline (riir-ai `35108b6a7`).
- **Site 4 `BranchBank` — DECLINED.** Wire format pins in-band `Removed` slots + explicit `free_slots` order; byte-identity would expose pool internals, and the bytes feed neuron-db freeze (a versioned migration).

Lesson: a contract extraction earns its keep where the contract is the site's whole job; where it is a subset of a bigger load-bearing shape (refcount, wire pin, graph pointers), leave the lineage. Decline-with-evidence is the same honest class as Bench 800-B's tie.

Issue file removed (HISTORY + module doc tables + Bench 800-C are the record).

## The full gate's wasm32 layer counted NEGATIVE cfgs as surface — derivation fixed positive-only (2026-09-16)

The 802-followup sweep found layer 2b red on three "new" wasm32 sites — `benches/plan598_refinement_marginal_bench.rs`, `crates/katgpt-core/tests/bench_779_real_bank_affinity.rs`, `tests/refinement_marginal_tokenizer_bridge.rs` — all `#![cfg(not(target_arch = "wasm32"))]`. The bare `git grep -l` couldn't see negation (Issue 738 T3's class; fourth instrument to meet it).

Fix (`448c77f91`): `WASM_FILES` counts only compile-time, non-negated wasm32 cfgs (`not(...)` and runtime `cfg!` excluded). Line-based; multi-line `not(` over-includes (safe direction). `WASM_RESIDUE_EXPECTED` empty by construction; GOAT targets stay via `WASM_EXTRA_TARGETS`; `-p` list unchanged.

Layer 3 then ran for the first time since the red: 3 `needless_range_loop` errors in Plan 598 test code (`0fbac8248`, landed under default features only) + an unused bench import — fixed (`enumerate().take(n)`, `zip().enumerate()`, 5/5). Seven feature-gated warnings (not in the -D list) left to owners.

First full-gate PASS on this box (`✓ full gate PASSED — 0 errors, 0 unbuildable targets`); test_gate 203 · 2060 · 249 · 139 at floor. slt.rs `field_reassign_with_default` blocker (`dfa6d3ff0`, since `7352a75ab`) cleared.

## Issue 844 (2026-09-19) — the dot-delegation crossover measured on BOTH arches; the NEON answer refuted the filing session's own expectation: CLOSED

T4 on the M3 (NEON, three runs, ±0.13): **no crossover — delegation wins at EVERY length ≥ 4** (1.94–2.02× @4, 1.70× @16, 3.19–3.32× @32, 9.06–9.08× @256), refuting "crossover LOWER". aarch64 dispatch is compile-time (~0.8 ns floor) vs x86_64 runtime CPUID (~3.3 ns); plain loop slower on NEON (1.5 vs 1.1 ns @4; 113 vs 78 ns @256). Rule 1 is ISA-conditional: arch-weighted crates should delegate even at D=8–16. The bench's "no crossover" branch now names the winning arm instead of flagging a model violation.

T2/T3 per-site read (398 fn-dot defs → 247 src candidates, excluding `katgpt-types/src/simd/**`):

- **8 CHUNKED**: `katgpt-core/cgsp/types.rs:41 dot_f32_fma4`, `katgpt-kv/still_kv/perceiver.rs:486 dot_chunk4`; `katgpt-dec/ simd.rs:50` (⚠ zero-dep crates.io crate — owner call); riir-ai `cross_game_prefix.rs:528`, `motivation/math.rs:38`, `lora_still_forward.rs:575`; riir-train `embedding_translator/model.rs:568 dot8`, `edge_lora/sigmoid_gate.rs:233 dot_product_chunked`.
- **10 large-D naive**: riir-ai 982 six, riir-neuron-db 621 two, katgpt-rs `score_matrix_simd.rs:121 dot_8wide` and `specialist_projection.rs:206 dot_truncated`.
- `newton_schulz::blocked_dot8{,_neon,_scalar}` **KERNEL-HOME** (8-output GEMM micro-kernel).
- 56-UNRESOLVED resolved as predicted (dot_acc_into, tropical, const-generic, test fns, f64/i8, name collisions). **No hidden findings.**
- `katgpt-moka-wasm` exemption SOFT — recorded.

Disposition: **riir-ai Issue 982**, **riir-train Issue 562**, **riir-neuron-db Issue 621**, with the repair contract (summation order change, max |Δ| ~3e-6 @64; adjacent gates re-run; determinism sites out of scope). kron_tile note ISA-conditional (NEON n∈{8,16} WIN 1.8×/1.7×).

## Issue 849 (2026-09-19) — the 844 per-site dot read, katgpt-rs-own sites: four delegations landed + the full repair record-back: CLOSED

The T5/T3 record-back for riir-ai 982, riir-train 562, riir-neuron-db 621.

**Internal repairs:**
- `katgpt-core/cgsp/types.rs` — `dot_f32_fma4` → `dot_f32`, delegates to `crate::simd::simd_dot_f32` (3.13× x86_64 / 5.2× NEON @64); BLAKE3 commitment hashes bytes, so determinism holds per binary/arch. `--features cgsp` 44/44.
- `katgpt-kv/still_kv/perceiver.rs dot_chunk4` → delegates (name kept). `--features still_kv` 18/18.
- `katgpt-attn-match/score_matrix_simd.rs dot_8wide` → `katgpt_core::simd::simd_dot_f32`; katgpt-core made NON-optional (avoids Issue-845 cfg-dual kernel; `publish = false`). Default check + `--features maxsim` 12/12 + `--all-features`.
- `katgpt-sparse/specialist_projection.rs dot_truncated` → delegates at `a.len().min(b.len())`. 39/39. Clippy clean.

**Deferred (owner):** `katgpt-dec/simd.rs:50` zero-dep posture.

**Sibling record-back:**

| repo | issue | commit | scope | gates |
|---|---|---|---|---|
| riir-train | 562 | `3cf49ebb` | `dot8` (mixed 12–32) + `dot_product_chunked` | engine 11/11 incl. gradient; gpu edge_lora 197/197 |
| riir-neuron-db | 621 | `5d86dd3` | `transition_error_taxonomy::dot` (64) + `hebbian_bridge::phi_dot` (64, truncating) | lib 54/54 |
| riir-ai | 982 | `dbc5639d4` | F1 cross_game_prefix / F2 motivation / F3 lora_still_forward / C1 cce (64) / C2 log_salience_dot (32) / C3 kg_hyperedge | engine 159+1ign / civ 329 / gpu 15 |
| riir-ai (deferred) | 982 T4 | — | riir-poc C4–C6 (32/64/64) — `[-]` | — |

No site had a bit-determinism contract; ~3e-6 @64 passed all gates. Issue file removed.

## Issue 848 (2026-09-19) — a rename privatised a delegation target and a rework deleted a shared fixture; both of that module's EXTERNAL callers are docs-gate CHECKS: CLOSED

`6c6ca2ee` renamed `is_checkout` → `_is_checkout` in `scripts/worktree_state.py` (seven in-module callers updated, two external not) and deleted `worktree_fixture`. `console_encoding_gate.py` and `sweep_advisory_membership_gate.py` died with `AttributeError`: **`develop` red 6h40m**, NO verdict (Issue 804's class).

⛔ Both names had written contracts naming their consumers (AGENTS.md line + fixture docstring); nothing read them. Python has no link step; `docs_gate.yml` is main-only, Rust lanes don't run it, and `arm_reach_gate` (would report `BASELINE-CRASH`) is a 157.6s workstation verdict.

### What landed

- **T1** — names restored; `worktree_fixture` verbatim from `6c6ca2ee~1` with a why-public comment at the definition. Selftest 144 assertions; docs gate green.
- **T2** — `scripts/cross_module_attr_gate.py` (STATIC): sibling-module attributes / `from X import n` resolved against X's top-level names; LINE-FREE keys. Known answer: **0 at `6c6ca2ee~1`, exactly 4 at `6c6ca2ee`**, 0 repaired, 90 files. Stated blind spots: `getattr`, star-imports, function-local names, `__all__` (deliberately unread).
- **T3** — `scripts/import_health_gate.py` (EXECUTION): **6.238s of 6.34s** over 86 modules was one all-top-level module (`list_unresolved_percentile_sites`, now guarded). Per-module subprocess 10.28s vs 7.52s, same verdicts — not worth it. **MISSING-DEP bucket never flagged** (would be box-dependent).
- **T4** — cross-repo: **10 of 17 repos carry `scripts/*.py` (196 files)** but **713 of 749 resolved references are here** → no sweep; `cross_module_attr_gate.py --workspace` re-derives the table.

⚑ T3 found two live defects: `subprocess_encoding_gate` red on a `PYTHONIOENCODING` dict bound to a local; `console_encoding_gate` red on the newly-guarded module entering its population. **A repair that grows a population owes the other gates a run.**

⚠ Main-only CI is an owner call and stands; the resulting `develop` lane is **zero** (`ci_gate_coverage.py`: 12 of 16 repos) — first time it cost a red `develop` in Python.

## Issue 856 (2026-09-19) — the green zero has TWO spellings and the audit built for it saw one: CLOSED

`cfg_gated_target_audit` only matched the inner `#![cfg]`; a whole-body `#[cfg(feature = "x")] mod tests { … }` zeroes identically — an unstated blind spot. It printed `SILENT-NOW 0` over 26 such targets, **175 assertions**, 7 `*_goat`.

T1–T5 same day: `whole_body_cfg_mod` (sharing `platform_dead_code_audit.mask_file`); predicate = gated items are the WHOLE body (`tests/test_freeze_thaw.rs` the measured true negative); runs gated by `any(...)`; top-level `use` ignored; `cfg_row_implication_audit` shares it. Twelve `required-features` rows added, compiler-verified both ways. Two sibling breaches repaired there (`seal-remake 964e780`, `seal-game-editor fbb5a931`); no ceiling raised.

⛔ First cut anchored with `\A` and a non-zero `pos` → `None` for every file; **the repair read exactly like the defect**, caught only by the summary line not moving.

Full record: `.docs/10_audits/cfg_gated_silent_zero_pass.md` §"The SECOND spelling". Commits `4e2f28f2d` · `6399faf69`.

## Issue 860 (2026-09-20) — `successor_density_critic`: tabular discounted count-ratio goal-critic: CLOSED

Modelless CRL extraction (riir-ai Research 386; arXiv:2206.07568) landed at `2c7a1f157`: dense `[S][A][S]` f64 tables, one O(L·G) reverse hindsight sweep, `Discounted` (default, exact) + `CLearning` (parity, non-default — breaks G1 exactness), BLAKE3 freeze/thaw at full precision, Lemma-4.1 ranking invariance as a property. **Bench 818 GOAT PASS**: G1a 0.00841 ≤ 0.01 vs the behavior-continued Bellman fixed point (first oracle was wrong; gate caught it), G1b 128/128, G1c bit-identity ×0.001…×1e6, G1d byte-identical freeze, G2 score 0.9 ns / argmax_a 4.3 ns / argmax_g 44.0 ns, G3a–c 0 discordances vs 95 raw-count, G4 zero allocs — release and dev, M3.

Consumer pull-gate **satisfied**: riir-ai Issue 991 lane (b) `goal_salience` (opt-in) forwards `katgpt-core/successor_density_critic` in `riir-engine/Cargo.toml` (grep-verified), code at `riir-engine/src/cgsp_runtime/goal_salience.rs` + `riir-games` swarm, riir-ai Bench 949 (6.3× first-reach) + riir-ai Bench 950 A/B. riir-train Plan 413 remains its own lane. **Stays opt-in**; riir-ai promotion is production-host-gated (owner). Catalog §119; `bench_818_successor_density_critic_goat`. 12/12 module tests green at removal.

## The exact_sigmoid / dot_f32_ordered substrate promotion (2026-09-20) — the riir-chain Issue 156 T1 landing executed in this repo

Landed `5458dd69b` (develop) + `5e2b730f2` (main, cherry-picked via detached
worktree per lthash precedent — git-dep consumers pin main; branch topology
owner-gated). Ungated additive always-compiled primitives (`float_order`
precedent): `exact_sigmoid` / `exact_sigmoid_f64` in
`katgpt-types/src/simd/activations.rs` (two-branch libm, no Cephes, no ±40
clamp) and `dot_f32_ordered` in `katgpt-types/src/simd/dot.rs` (sequential
index-order fold, deterministic). `katgpt_core::sigmoid`'s doc now names the
exact variant, closing the silent-approximation trap.

**Bench 844 GOAT PASS** (`844_exact_sigmoid_ordered_dot_substrate.md`): G1a
f32 max **2 ULP** vs f64-narrowed reference against `fast_sigmoid`'s
**580,601,137 ULP** (ULP, not abs-error, separates the clamped tails); G1b
f64 reflection/monotone/bounds (no ULP oracle — it IS libm); G1c pin
`[1e8, 1.0, −1e8, 1.0]` → 1.0 where every SIMD backend reads 0 (anti-dedup,
reds if a backend converges); G2 reported; G3/G4 by construction. ⚠ Cephes
speed claim inverts on aarch64: `exact_sigmoid` 1.7 ns vs `fast_sigmoid`
3.1 ns (best-of-50, loaded box) — don't quote "~1.7× faster than libm" here.

Consumer: riir-chain 156 T2 delegated `curator_bridge::{sigmoid,
dot_product}` + `forensic/recover::sigmoid` behind `to_bits` bit-identity
pins (`f7eb85e4` + `b0b710e0`, then `7facc3cd`); `consensus/congestion`
refused (`x < 0` reachable via pub inputs; consensus numerics is its own
call). In-repo copies delegated by Issue 861 (`salience/gate.rs`,
`breakeven/mod.rs`, `escalation_sigmoid`, `inv_log_reveal_odds`,
`p_successor`; kept: test oracle, gate.rs `dot_fma`). README showcase
landed; ungated primitives take no catalog row.

## Issue 865 T3 (2026-09-21) — the probe_guidance λ-sweep GOAT gate ran NEGATIVE; the negative verdict is the pinned gate: OPEN (lane re-opens at Bonsai scale)

Bench 847 (`9c2acbfaa` + `3ebc51944` for T4 scoping): trained-probe guided
vs unguided temperature front on mini-dLLM, with a ZERO-logit null
(`λ·logits = logits/(T/λ)` — pure temperature, not an RNG re-roll) and a
mean-zero directionality control (monotone −1.17 pts). Guided best λ=1.25
(+0.21 pts) is DOMINATED by T=1.0 at matched diversity (100.00% @ 3.2171
nats); trained probe loses to the null at every λ ≥ 1.5 (−2.11 at λ=2).
Cause: saturated T2b trunk (loss 0.0000, one-hot); weak side (held-out CE
0.2166) has nothing to exploit. G1 (λ=1 bit-identity) + G4 (1,000 calls, 0
allocs) PASS — machinery qualified, mechanism needs a non-saturated trunk.

Pinned in `tests/probe_guidance_goat.rs` (G0 < 0.5 CE canary): (1) pooled
unigram entropy is polarity-inverted on deterministic lanes — use
per-position resample entropy; (2) read any λ sweep against the zero-logit
null; (3) G2a/G2b INVERTED into pins that red the day guidance wins — the
Bonsai-scale re-open (multi-layer kernel + trunk with headroom; T4 AR arm
shares scope) just runs them. Opt-in; dropout arm deferred (owner scope).
`.benchmarks/847_probe_guidance_lambda_sweep_goat.md`.

## Issue 866 (2026-09-22) — KARC D3 promotion coverage audit: VERDICT QUALIFY — the contract's passing legs live on configs nobody constructs: CLOSED

Bench 849 (`crates/katgpt-core/examples/karc_deployed_shape_quality.rs`):
Bench 308's D3 contract (NRMSE ≤ 1e-3, threshold ≥ 8 LT) was never measured
on a constructed config nor ratified downstream. (1) Both passing legs are
`ChebyshevBasis`; every deployed shape is `FourierBasis` R=1 (Lod0 F<4>/K=2,
Lod1 F<8>/K=4, Lod2 F<8>/K=8) and all fail both bars on the double-scroll
fixture (NRMSE 2.4–53; 0.03–0.16 LT) — first-order Fourier can't
reconstruct it. (2) None of six `karc_runtime` GOATs measures forecast
accuracy. (3) Deployed use is one-step (`tick_karc`, re-fit each
`tau_reest`): Lorenz leaky-belief fixture NRMSE 1.2–4.1e-3 at λ=1e-4 — the
QUALIFY record; autonomous rollout unstable (~2.2×/step, λ-independent) — a
precondition for any rollout consumer. Not DEMOTE. riir-ai corrected Plan
332's "shape fixed at Plan 308 GOAT" and `karc_bridge/lod.rs` (2-variant,
Lod1 never dispatches). `faer` spike stays conditional. Scope noted in Bench
308 addendum. `.benchmarks/849_karc_deployed_shape_quality.md`.

## Issue 865 file hygiene (2026-09-22) — issue file removed per noise-reduction, the lane's record was already durable: CLOSED (hygiene)

T1/T2/T5 landed, T3 negative (Bench 847), T4 deferred until the multi-layer
kernel. All durable elsewhere (catalog
`.docs/09_feature_catalog/opt_in_features.md` §120, Research 578, riir-train
recipe rows A/B, G2a/G2b pins in `tests/probe_guidance_goat.rs`). Removed
with no content change; the Bonsai-scale re-open files its own issue.


## Issue 865 follow-up (2026-09-22) — arm (b) unblocked (DropoutHeadProbe) + the headroom study: the negative EXTENDED to every mini-lane regime (Bench 850)

Closes both Bench 847 gaps (no dropout substrate; saturated trunk):

- **`DropoutHeadProbe`** (`katgpt-forward/src/weak_probe_mlp.rs`, +3 tests):
  frozen head over a deterministic 50%-masked tap (LCG keyed by `(position,
  denoise step)`, no RNG, zero-alloc) — no kernel dropout needed; weak side
  stays a noisy copy of the SAME function.
- **Headroom rerun** (847 method: resample entropy, zero-logit null, 256 ×
  8) over high-data (2048 seqs), low-data (96 seqs) and strict-decode
  (τ_conf 0.7 / 8 steps; front 82–99%): **never beats the null** (−0.13 …
  −1.42 pts).
- A "+2.7 pts" pooled-unigram-axis positive was retracted — redistribution
  along the refuted axis; second instance of the axis lesson.
- Mini lane is structurally incapable; only the Bonsai-scale lane
  (multi-layer kernel + natural text + recipe row B) remains, Bench 847's
  G2a/G2b as decider. `tests/probe_guidance_headroom_study.rs`;
  `.benchmarks/850_probe_guidance_headroom_study.md`.

Session: katgpt-rs-865-followup, 2026-09-22


## Issue 876 (2026-09-23) — the flappy render widened to v3; the decoded arm reads Δ0 vs the structured arm: CLOSED

Bench 881: v2 decoded arm was constant-pick (77/100, one distinct). Grammar
v3 (`laya-flappy-v3`): quantized OFFSET clause (±2) + neutral
"drifting"/"holding" motion (v1 "rising" was the Bench 880 confound); oracle
split 48 flap / 52 coast, no ties.

- Same 100 states, asserted vs v2; oracle riir-reflex `laya_oracle_batch` @
  `63b1552`, G5 parity re-verified.
- **Δ0**: structured 96/100 and decoded (structured-units reconstruction)
  96/100, in + LOO, 4/100 flips; digests pinned. Raw ordinals read 51/100.
- v2 frozen (`render_option_sentence_v2` pinned); Bench 880/881 anchors hold.
- `.benchmarks/882_flappy_v3_render_widening.md` · Catalog §125; code
  `515230244`.

Session: katgpt-rs-876-flappy-v3, 2026-09-23

## Issue 879 (2026-09-24) — MAttr budget primitives landed opt-in: exact mass at router-regime cost; calibrated-mass column says what the hard cut cannot: CLOSED

Research 584 (arXiv:2609.25518 "Matryoshka attribution") POC, opt-in
`exact_mass_admit`: `exact_mass_admit_into` (bisect τ so Σσ((s−τ)/T) = k, on
`simd::exact_sigmoid_f64`) + `log_frontier` (modelless `AdaptiveLogK`,
caller-owned randomness). Record: Bench 884.

- G1 14/14 (sum-to-k, dyadic shift BIT-invariance, nestedness), G4 0
  allocs, wasm32 clean, default unchanged.
- Bench 884 (release, 4090RTX): `|Σm−k|` ~1e-8 relative at N 1e3–1e7 vs
  −6.4% hard-cut drift; CHEAPER than `gate_sigmoid_topk_into` at N=1e3
  (69.8 vs 77.1 µs, 0.906×); ~45× `select_nth` at 1e7 (1.10 s vs 25 ms).
- No consumer → no promotion; candidate consumers listed in Bench 884.
- ⚠ `exact_mass_admit` ≠ `gate_sigmoid_topk` — opposite mass semantics.

## Issue 892 (2026-09-25) — tetris strategy RULEBOOK (ruliology surface) + chance-node PUCT + laya head-to-head: the hybrid FSM champion out-scores laya 92–299× and Bench 891 4×: CLOSED

Owner directive: encode the owner's Tetris techniques as ruliology,
transplant moka+PUCT, run laya h2h. Records
`.benchmarks/892_tetris_rulebook_arena.md`, `.benchmarks/892_laya_h2h.md`,
`.benchmarks/892_chance_puct_goat.md`.

- T0 `1bbde0ac9` — `examples/common/tetris_lookahead.rs` extracted; Bench
  891's "85%" fill was a prose error (75%).
- T1/T2 `1d03ec07c` — `examples/common/tetris_rulebook.rs`: 16 rules as
  data, Build/Downstack/Survive FSM, BLAKE3 `Genome`; T-spin inapplicable.
- T3 `13dcc660e` (sibling) — `chance_puct` opt-in: G1 6/6, 0 allocs; wins
  only at 19@75 (21 → 30/60) at 5–150× latency.
- T4 `45b9ff27b`, `c84d9e8e2` — champion `ed5aa14b7d68472e` (9-1 stack
  emerged); hybrid `68cae9d382014662`: equal survival (22/60 · 35/40 ·
  20/20) at 4.0–4.7× points.
- T5 `9c472531f` — laya 0/60; hybrid 60/60, 92×/119×/299× points at 1–3%
  latency.
- T6 — hybrid PROMOTED, Bench 891 → reference. Negatives: always-on
  downstack (−37 pieces/g), no-hold climb; hold (+390) sim-only.


## Issue 893 (2026-09-25) — the tetris substrate promoted from `examples/common/` into the leaf crate `katgpt-tetris`: CLOSED

P0 of riir-train Research 457: leaf crate `crates/katgpt-tetris` (not a
core feature; precise dep for riir-reflexer). Deps `rayon`, `blake3`,
`fastrand`.

- T1/T4 — `tetris_sim.rs` → `src/sim.rs` (byte-identical);
  `tetris_lookahead.rs` → `src/lookahead.rs`; `tetris_rulebook.rs` →
  `src/rulebook.rs`. Nine importers use-aliased; `grammar_tables.rs`
  untouched; `tetris_fixture.rs` re-pointed; originals deleted.
- T2 G1 — base-vs-new byte-identical: h2h fingerprints, fixture replay
  (1,113,251 / 1,113,238 B), tetris_04/06/09, decode_01 300/300; 14/14 tests.
- T3 G2 — 0.321–0.332 vs base 0.322–0.344 ms/decision (AC, load 4.33).
- T5 — `Genome::champion_hybrid()` pins `68cae9d382014662`; evolved values
  live in the private loop.
