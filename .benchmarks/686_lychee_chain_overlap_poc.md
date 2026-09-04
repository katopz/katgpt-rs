# Bench 686 — LycheeDecode chain-overlap PoC (Issue 694): adjacent-layer same-head top-k overlap on Kimi-K3-0.40B real weights
**Status:** DONE 2026-08-28 — verdict **KILL** for `fm_chains` on the measured model (span-4 chain overlap ≈ chance at every k); the paper's adjacent-layer question is structurally unmeasurable on this architecture; generality unproven (T3 [-], 4090 busy + no attention-prob tap).

**Sources:** `Issue 694` · [Research 514](../.research/514_LycheeDecode_Hybrid_Head_Chain_Propagation.md) (§1.2 Fig. 2 is the measurement reproduced here on OUR weights) · [arXiv:2602.04541](https://arxiv.org/abs/2602.04541) (LycheeDecode, ICLR 2026) · probe: [`686_lychee_overlap_probe.py`](686_lychee_overlap_probe.py)

---

## 0. Headline architecture discovery (changes what the PoC can measure)

**Kimi-K3-0.40B is a HYBRID linear-attention model — the issue's premise (8 full-attention MLA layers) is wrong.** `config.json` → `linear_attn_config`:

- `kda_layers: [1,2,3,5,6,7]` (1-indexed; `configuration_kimi_k3.py:155` matches `(layer_idx + 1) in kda_layers`) → layer_idx 0,1,2,4,5,6 = `KimiDeltaAttention` (linear/delta-rule attention, no attention matrix exists)
- `full_attn_layers: [4,8]` → layer_idx **3 and 7** = `KimiMLAAttention` (full attention)

Confirmed on the checkpoint itself (`model.safetensors`: layers 0/1/2/4/5/6 carry `A_log`/`dt_bias`/`*_conv1d` KDA weights; layers 3/7 carry MLA `kv_a_proj_with_mqa`/`q_a/b_proj`). Runtime layer types: `['KDA','KDA','KDA','MLA','KDA','KDA','KDA','MLA']`.

Consequences, stated honestly:

1. **There are ZERO adjacent full-attention layer pairs.** The paper's adjacent-layer chain mechanism (`S^(l+1)_h = S^(l)_h` reuse between neighboring full-attn layers) is **structurally inapplicable** on this model. The only measurable full-attention "chain" is **layer 3 → layer 7, span 4** — which is exactly the chain a hybrid architecture would need (the sparse-decode path only touches full-attn layers, so any chain must span the gap). That is what this PoC measures.
2. The paper's Fig. 2 statistic (per-head adjacent-layer overlap 0–100% on Llama3-8B, 32×32 dense) **cannot be reproduced on this model** — the required layer pairs do not exist.

## 1. Harness

- **Model:** `data/kimi-k3-0.40b/` real weights, CPU, float32, `attn_implementation="eager"`, batch 1, greedy decode (`do_sample=False`), `torch.manual_seed(0)`.
- **CPU execution path (no triton on macOS):** the fla (flash-linear-attention) kernels are triton-only. The probe (a) pre-registers constructor-parity stub modules for the exact `fla.*` import names in `modeling_kimi_linear.py` (forwards never called), and (b) **replaces `KimiDeltaAttention.forward` with a pure-torch reimplementation** of the fla KDA semantics — causal depthwise conv1d + silu with last-W-inputs cache; l2-normalized q/k; sigmoid beta; per-step per-dim log-decay `g = -exp(A_log)·softplus(g_proj + dt_bias)`; delta-rule recurrence `S ← S·exp(g); S ← S + β·k⊗(v − k·S); o = (q·scale)·S`; RMSNorm·sigmoid output gate (full-rank `g_proj`). Attention capture: `modeling_kimi_linear.eager_attention_forward` is wrapped to stash per-layer last-query-row probs `[B,H,1,ctx]` (the remote code computes but discards them; `output_attentions=True` is not honored by this remote code).
- **Validation of the reimplementation:** the model dir ships `ref_logits_bos.npy` (raw float32 dump, 163,840 = one vocab-sized BOS logits row). Probe reproduces it at **cosine 0.999999, max|Δ| 0.0023, identical argmax (420)** — the pure-torch KDA path is faithful up to fp32 accumulation order.
- **Prompts (fixed, deterministic):** 9 prompts = 3 needle depths {25%, 50%, 75%} × 3 needle values {48371, 90215, 65932}; haystack = 12 distinct filler sentences cycled to ~3.5K tokens; needle "The magic number found in the book is XXXXX."; suffix question "Question: What is the magic number found in the book? Answer:". All contexts 3,637 tokens (max_position 4096 leaves decode room). Needle token spans verified by boundary tokenization (11 tokens each; spans 912/1812/2712 by depth).
- **Decode:** 48 greedy steps per prompt, `use_cache=True` (KimiDynamicCache). At each decode step the query row's attention over the full context is captured per (layer, head).
- **k grid:** raw k ∈ {512, 1024, 2048} and block-granularity 64 tokens (top {8, 16, 32} blocks ≡ 512/1024/2048 tokens; block sets = sets of block indices, blocks scored by attention mass — the flashmemory selection shape). The issue's raw k=4096 is **degenerate** on this context length (3,637 < 4,096 → top-k = everything → trivially 1.0) and is excluded; reported here as degenerate per the task note.
- **Statistics:** overlap(l→7, h) = |topK_h^l ∩ topK_h^7| / k per decode step, averaged over steps × prompts, per head (population = 8 heads × 1 layer pair).

## 2. Primary tables — layer 3 → layer 7 same-head overlap

**Chance baseline matters:** for two independent k-subsets of n positions, expected overlap = k/n. Mean query-time context n ≈ 3,662 (57.2 blocks).

| variant | per-head mean overlap (h0..h7) | median | Q1–Q3 | chance (k/n) | median/chance | #heads ≥0.5 / ≥0.7 / ≥0.9 |
|---|---|---|---|---|---|---|
| raw512 | .064 .081 .135 .135 .159 .122 .180 .318 | **0.135** | .122–.180 | 0.140 | **0.96** | 0 / 0 / 0 |
| raw1024 | .161 .252 .235 .204 .278 .211 .237 .414 | **0.236** | .211–.278 | 0.280 | **0.84** | 0 / 0 / 0 |
| raw2048 | .522 .526 .544 .523 .546 .528 .511 .655 | **0.527** | .523–.546 | 0.559 | **0.94** | 8 / 0 / 0 |
| blk8 (512 tok) | .163 .289 .100 .108 .073 .174 .131 .215 | **0.147** | .108–.215 | 0.140 | **1.05** | 0 / 0 / 0 |
| blk16 (1024 tok) | .238 .432 .158 .294 .280 .296 .308 .270 | **0.287** | .270–.308 | 0.281 | **1.02** | 0 / 0 / 0 |
| blk32 (2048 tok) | .450 .718 .404 .574 .589 .534 .511 .471 | **0.522** | .471–.589 | 0.561 | **0.93** | 5 / 1 / 0 |

**Reading:** every measured overlap sits at or slightly BELOW the random-chance baseline (ratios 0.84–1.05). At raw2048/blk32 the ≥0.5 head-counts are pure budget artifacts (k ≈ 55% of context — the sets can't help overlapping). **The layer-3 and layer-7 selections are statistically independent.**

### Cross-head structure (raw1024, mean over steps × prompts)

- Same-head (l3 h → l7 h): median 0.236.
- **Same-head margin (same-head − best-other-head) is NEGATIVE for 6 of 8 heads**: −.134, −.058, −.047, −.058, +.013, −.070, −.012, +.017. Head *identity* carries no cross-layer alignment — for most heads a *different* layer-7 head matches the layer-3 selection better than the same-index one.
- **Within-layer off-diagonal overlap: L7 0.680, L3 0.574** — an order-of-structure gap over the cross-layer same-head 0.236. Attention top-k sets cluster by **layer**, not by head identity across layers.

### Depth trend (spec T2)

N/A as specified: with one full-attention pair there is no layer-pair index to trend. The two measurable trends are both flat:

- Across decode steps (raw1024): first-8-steps 0.2596 vs last-8-steps 0.2542 — flat.
- Across needle depths: 0.2491 / 0.2489 / 0.2489 (25%/50%/75%) — flat.

## 3. Harness-validity contrast: the time axis is alive, the depth axis is dead

Same-head **step-to-step** (decode step s → s+1, same layer) overlap at raw1024 — this is the axis `flashmemory_sparse` already amortizes (τ-step refresh):

| layer | per-head step-to-step overlap | mean | chance |
|---|---|---|---|
| L7 | .788 .700 .736 .728 .751 .783 .762 .778 | **0.753** | 0.280 |
| L3 | .374 .383 .415 .359 .404 .385 .374 .353 | **0.381** | 0.280 |

L7's selections are strongly time-stable (2.7× chance) — the probe detects real selection structure exactly where the production path exploits it. **Against that contrast, the depth axis (0.84× chance) is specifically dead** — this is not a broken measurement, it is the result.

## 4. Secondary observation (retrieval-y heads × overlap) — not measurable on this model

Per-head needle-span attention mass vs per-head overlap (raw1024): L7 Pearson **−0.005** (Spearman 0.0), L3 Pearson −0.709 (Spearman −0.43). The L3 number is driven by a single outlier (head 7: lowest needle mass .0016, highest overlap .414) at n=8 — not meaningful. The deeper problem: **the model fails the passkey task outright.** All 9 prompts produce byte-identical non-retrieval continuations (" it continues. The 20 meter pacer test…"), regardless of needle value; a control at ~1K context with BOS prepended behaves identically. Retrieval-y heads therefore cannot be identified by behavior on this 0.4B toy — the secondary is recorded as **not measurable here**, not as evidence either way.

## 5. T3 disposition — Kimi-K3-only, generality unproven

- 4090 check (robust pattern): LAN `192.168.1.33` connect timeout; Tailscale `100.85.179.44` reachable. `nvidia-smi`: 535 MiB / 20% util, **and a sibling compute process resident** (`riir_poc-4157b47e1e558497`, 10.3 GB RSS in `tasklist`) — per the owner's standing GPU-exclusivity rules the box is busy; not used.
- Independent second reason (recorded per the issue's own escape hatch): even on a free 4090, capturing per-head attention *probabilities* from the Bonsai-27B Rust GGUF/cudarc path is substantial harness work with no existing attention-prob tap — Issue 717's layer capture is hidden-state only. Cheap path did not exist.
- **Disposition: Kimi-K3-only + generality unproven.**

## 6. Verdict table

| criterion (issue T4) | measured | verdict contribution |
|---|---|---|
| KILL — median overlap < ~0.5 on measured models | raw512 **0.135**, raw1024 **0.236**, raw2048 0.527 (=chance 0.559); blk8 .147 / blk16 .287 / blk32 .522 (=chance) | **MET** — and stronger than the criterion: overlap is at/below chance at every k, no head above chance at any budget ≤1024 tokens |
| GO — majority of head-population ≥ ~0.7 | 0 / 8 heads ≥ 0.7 at every variant except blk32 (1, = chance) | not met |
| SPLIT — bimodal per-head distribution | distributions are unimodal and tight (Q1–Q3 spans ~0.06–0.12); the paper's 0–100% head heterogeneity is entirely absent | not met |

**VERDICT: KILL for `fm_chains` on Kimi-K3-0.40B.** The one chain this architecture could host (L3→L7, span 4) has zero signal — selections are independent across depth (≈ chance), head identity is anti-informative (6/8 negative margins), and structure clusters by layer (within-layer 0.57–0.68) rather than by head across layers. The paper's adjacent-layer mechanism remains untested for dense-transformer models — this result does not refute LycheeDecode; it (a) kills the chain mode for this model family and (b) documents that Kimi-K3-0.40B cannot host adjacent-layer chains at all.

## 7. Limitations

1. **8 heads × 1 layer pair = 8 head-slots** on a 0.4B toy vs the paper's 32×32 Llama3-8B — the population is tiny and the verdict is provisional on model scale whatever it is.
2. **Span-4 ≠ adjacent.** Even a positive result here would have been a weaker claim than the paper's; the negative result kills the span-4 chain specifically. Adjacent-layer chaining (the paper's actual mechanism) is unmeasured on any of our weights.
3. **The model fails passkey retrieval** at 3.6K (and ~1K with BOS) — the calibration prompts did not induce retrieval behavior, so the measured attention is "generic decode attention", not retrieval-engaged attention. (The paper's Fig. 2 was measured on passkey-retrieval data.) If retrieval-engagement materially changes cross-layer overlap, this PoC would not see it — but note the time-axis structure IS strong in the same regime, so the depth axis being at chance is not an artifact of a dead model.
4. Pure-torch KDA reimplementation differs from the fla kernels in fp accumulation order (validated: BOS logits cos 0.999999, argmax match); attention top-k overlap is robust to perturbations of this size, and the determinism check is exact.
5. Prompts tokenize without BOS (`add_special_tokens=False`); a BOS control showed no behavioral change.
6. Block-64 granularity measured on 57-block contexts; the 256K production regime (4,096 blocks) is extrapolation.

## 8. Determinism

Two complete probe passes (9 prompts × 48 steps each, same fixed prompts, greedy, CPU fp32) produced **byte-identical summary tables** (`determinism_identical: true`, asserted in the probe via `--determinism-check`).

## 9. Reproduce

```bash
# full run + determinism check (~8 min on M3 Max, CPU)
uv run --python /opt/homebrew/bin/python3 --with einops \
  python .benchmarks/686_lychee_overlap_probe.py --determinism-check --json-out /tmp/lychee.json
# quick smoke (2 prompts × 8 steps)
uv run --python /opt/homebrew/bin/python3 --with einops \
  python .benchmarks/686_lychee_overlap_probe.py --smoke
```

## 10. Next step

Per the issue's KILL branch: **no `fm_chains` plan.** Research 514 gets this outcome as the negative record (§Actionable addendum). If chain propagation is ever revisited, it needs a dense full-attention model (Bonsai-27B on an exclusive 4090 window, or a small dense HF model with a native attention-prob tap) and a retrieval-capable calibration set — reopen against Research 514 with those preconditions stated.
