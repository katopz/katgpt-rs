# Bench 889 — Issue 883 P0 second fixture: the Kimi-K3 MLA/KDA R² dashboard (fixture-class-null territory, trap 4)

**Status:** COMPLETE — the dashboard-only fixture ran on REAL weights (public
`inference-optimization/Kimi-K3-0.40B` `model.safetensors`, sha256
`a39b7ed2…57a678`). **MEASUREMENT-ONLY (the 883 P0 law): no quality claim, no
GOAT gates** — this fixture exists to say, before anything is built, that the
architecture lacks the mechanism 883's products retrofit (trap 4). The
mechanism-bearing fixture for P1/P2/P3 remains gemma-2-2b-it (riir-infer
Bench 004: mean ρ(V−K)=0.49, go/no-go GO).

**Box:** 4090 workstation (i7-13700K, Windows 11, AC power, CPU lane — GPU
idle, no compute consumers; no concurrent cargo during the recorded runs).
Release profile. Throughput 80–81 tok/s @seq_len=1024 (110 tok/s @512) —
the O(seq) per-cached-token W_UK/W_UV up-projections of the
explicit-up-projection `mla_forward_token` dominate; the 120k pass ≈ 25 min.

## What was measured (and how the tap is defined on MLA)

Kimi-K3-0.40B is a hybrid: 6 KDA layers (linear/delta attention, NO KV cache)
+ 2 MLA layers (0-indexed **[3, 7]**, from `full_attn_layers: [4, 8]`
1-indexed). On MLA the production cache stores the **128-d normed latent
`c_kv`** + the 32-d shared rope key (160 floats/token); K/V are NEVER
materialized — they are up-projected per cached token at attention time.
The dashboard's tap ("where the cache path would consume K/V") is therefore
**k = W_UK·c vs v = W_UV·c**, replayed **bit-identically** from
`MlaKVCache::latent_kv_at(j)` after each chunk using the same
`simd_matmul_rows` call the forward itself uses — zero production-forward
changes (`benches/bench_889_kimi_k3_vk_dashboard.rs`; the replay IS the
forward's step-3 arithmetic on the forward's step-5-cached input).

Corpus: the same sibling riir-train `chat_probe` HF-pages family as the
gemma dashboard (cross-fixture comparability), tiktoken-encoded. Slice
120,000 tokens, top_k=8192 @ **97.1% tracked mass**, seq_len=1024
(« 4096 max positions), per-head width 64 (d_h = v_h = 64 — the elementwise
V−K residual is well-defined).

## The dashboard (the P0 deliverable)

| model layer | ρ_l(V) | ρ_l(K) | ρ_l(V−K) | tracked mass |
|---|---|---|---|---|
| 3 | 0.4746 | 0.4518 | 0.4620 | 0.971 |
| 7 | 0.5594 | 0.6118 | 0.5976 | 0.971 |
| **mean** | **0.5170** | **0.5318** | **0.5298** | |

Per-head spread 0.43–0.63 (8 heads × 2 layers, both signals). Zipf coverage
(layer-3 V table): K=16→0.27, K=256→0.53, K=1024→0.69, K=8192→0.97.
Top-10 n_s: [5013, 4459, 3755, 3448, 2794, 2292, 2261, 1426, 1011, 1008].

## Findings

1. **The "coupled through one latent" signature is empirical, not just
   architectural: within each layer ρ(V) ≈ ρ(K) ≈ ρ(V−K)** (layer 3:
   0.475/0.452/0.462; layer 7: 0.559/0.612/0.598). In MLA, V−K =
   (W_UV−W_UK)·c is a linear function of the SAME latent that produces K
   and V — so token identity explains the residual at essentially the same
   rate as the signals themselves. On gemma-2 (Bench 004) K and V come from
   INDEPENDENT matrices applied to the same hidden state; the aggregates
   land in the same 0.45–0.61 range for the opposite reason. **Same ρ
   magnitude, opposite product applicability — the exact distinction trap 4
   exists to draw, now measured.**
2. **The fixture-class null is about PRODUCT substrate, not ρ magnitude —
   and the numbers say so loudly.** ρ here is healthy (~0.5), yet: KDA
   layers [0,1,2,4,5,6] have no KV cache at all (P1/P2/P3 have nothing to
   act on); MLA layers already carry 6.6× structural KV compression
   (160 vs 1056 full-attention-equivalent floats/token — P1's "separate V
   quant" and P3's "halving" are null-class: the latent IS the cache). A
   naive "ρ ≥ 0.4 ⇒ GO" reading of a dashboard like this would build a
   product with no substrate — the trap-4 analysis block ships inside the
   dashboard output itself for exactly that reader.
3. **The ONE live question this fixture can inform: P2 on the
   explicit-up-projection path.** `mla_forward_token` (this repo's current
   MLA decode) up-projects W_UK and W_UV per cached token per step — the
   V-side is 50% of that work. ρ_l(V−K) ≈ 0.46–0.60 is the fit-ability
   read for serving V := K + E_l[s] with the W_UV up-projection deleted
   (65,536 params/layer + one 512-wide matvec per cached token per head).
   Under weight absorption (production MLA serving), even that question
   dissolves — the absorbed path never materializes K/V either. P0 records
   the number; P2-on-MLA is a possible future lane, gated on a serving-path
   decision, never a default.
4. **Cross-fixture table substrate held (DRY): the layered builder is now
   substrate-owned.** `LayeredVkCalibration` (the table triplet + row map +
   V−K scratch) was promoted from riir-infer's gemma harness into
   `katgpt-core::fitted_anchor_table` — one builder, two fixtures, the
   gemma harness re-exports it under its historical name
   (`CalibrationTables`), zero behavior change (riir-infer clippy green).

## Reproduce

```sh
# weights (public, no key):
curl -sL -o data/kimi-k3-0.40b/model.safetensors \
  https://huggingface.co/inference-optimization/Kimi-K3-0.40B/resolve/main/model.safetensors
cargo bench --features "kimi_k3_loader fitted_anchor_tables" \
  --bench bench_889_kimi_k3_vk_dashboard -- --max-tokens 120000 --seq-len 1024 \
  --report /tmp/kimi_k3_vk_dashboard_120k.md
```

Substrate unit gates: `cargo test -p katgpt-core --features
fitted_anchor_tables --lib fitted_anchor_table` (10 tests, incl. the two
new `LayeredVkCalibration` row-map/residual gates).

## Posture

Opt-in (`kimi_k3_loader` + `fitted_anchor_tables` root forward — new).
Dashboard-only, measurement-only, no default feature changes, no
production-path changes. Issue 883 P0 is now FULLY landed (gemma half =
riir-infer Bench 004; Kimi half = this bench); P1–P4 proceed on the
gemma fixture per the go/no-go GO.
