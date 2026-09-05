# Research 523: Learning How to Forget — Age-Normalized Usage-Rate KV Eviction + Policy-in-Place Co-Training

> **Source:** "Learning how to Forget: Fine-tuning for Long-Context Sparse Attention" — Seeger, Zhang, Patil, Benidis, Schelter (AWS/UvA/TU Berlin), arXiv:2608.19920, Aug 2026. Library: awslabs/keys_values.
> **Date:** 2026-08-31
> **Status:** Active — both tracks filed
> **Related Research:** 100 (EGA energy-gated attention salience), 213 (Still Perceiver KV compaction), 487 (massive activations / sink-aware KV quant), 159 (KVarN), 201 (RAT-Plus train-dense-infer-sparse), 202 (QAT-Infusion), 502 (behavior before perplexity), 516 (TTT-KVB), 469 (collective intelligence — no), 435 (TTPO — the per-track verdict precedent)
> **Related Plans:** katgpt-rs 585 (usage-rate eviction primitive, PRIMARY), riir-train 367 (policy-in-place co-training, SECONDARY)
> **Cross-ref (riir-ai):** Issue 836 (delta-encoded KV chain retention; runtime wiring pull-gate)
> **Classification:** Public

---

## TL;DR

The paper fine-tunes LLMs **with a KV-cache eviction policy active during training** (policy-agnostic, via a replay log + delta-encoded KV autograd), showing dense-trained models collapse under sparse-attention inference (output length 35–128× target, saturating the token cap with nonsense) while co-trained models output target-length answers and often win on accuracy. Two modelless extractions ride along: (1) the **normalized H2O score** — cumulative attention mass divided by residency age — a usage-rate eviction estimator that fixes raw-H2O's age bias and helps even the *untrained* checkpoint (their Table 4: h2o_norm 42.2 vs raw h2o 20.5 on 128k nq); (2) the **R/p128 generation-runaway canary** — a cheap diagnostic that catches train/inference attention mismatch that perplexity-style metrics are blind to.

**Distilled for katgpt-rs (modelless, inference-time):**
`score(j) = cum_mass(j) / max(1, t − admission(j))` — an O(1)/row/step incremental usage-rate estimator over caller-supplied per-row attention-mass increments, with pinned-sink exclusion and per-(b,h) selection. Pure scoring primitive (the `causal_head_importance::suspect_indices` house pattern: caller supplies the observational signal). Fused with our stack: **EGA static key-energy (admission prior) × usage-rate (online correction) × β-sink pin (eviction axis of the Issue 716 sink discipline) × R/p128 canary (generation-side gate the Issue 750 lossy-surface rule lacks)**.

---

## 1. Paper Core Findings

1. **Co-adaptation beats train-dense→infer-sparse.** SP-trained (exact attention) checkpoints under sparse-attention inference produce runaway generations: output/target length ratio R ≈ 35–128×, p128 ≈ 100% (saturating a 128-token cap with nonsense), while the SAME checkpoints under exact inference have R ≈ 1. Co-trained checkpoints: R ≈ 1 everywhere. "Models should be trained under the same conditions and restrictions which govern inference later on."
2. **Metric blindness.** SubEM (target-is-substring) reads fine on runaway outputs; Accuracy collapses. The failure mode is invisible to tolerant metrics — the KV-eviction twin of our Issue 750 T3 lossy-surface rule (aggregate perplexity flat while conditional behavior flips).
3. **Policy-agnostic training method.** Replay log (record policy decisions {π(b,h,t)} in forward; backward replays them — no differentiation through the policy) + nested activation checkpointing (cells ≈ KV-cache-sized) + **delta-encoded KV buffers in autograd**: neighboring chunk caches differ in only S rows via `keys' = scatter(keys, index, key_new)`; store `(index, delta_key, delta_value)` via saved-tensor hooks (fingerprint-matched by `gather(x,index) == delta_key`) → autograd memory O(N_C·D), comparable to inference. Runs on 4× A100-40GB for a 4B model, LoRA r=16 α=16 all-linear, AdamW 5e-4, ≤5 epochs, N_C=32768, S∈{1024,2048}, 8-bit KV quantization in the loop.
4. **Normalized H2O score.** Raw cumulative attention mass monotonically favors residency age (an old-but-cold row at 0.001 mass/step × 1000 steps ties a young-but-hot row at 0.5/step × 2). `φ_norm = mass / (t − token_pos)` converts cumulative evidence into a per-step usage rate. Also: per-(b,h) selection (original H2O sums scores across batch before selecting — pure coarsening since each head owns its rows), and β pinned sink tokens (lastrec) as structural protection — sinks sit at pos≈0 where age is maximal, so pure φ_norm would evict exactly the load-bearing rows.
5. **Summed attention weights as SDPA byproduct.** H2O-class policies need column sums `w_j = Σ_i m[i,j]` that Flash-family kernels discard; they ship Triton code returning them alongside a FlashInfer kernel, plus a FlexAttention two-call trick (flip Q/K, pass exp(−λ) as values; `w = exp(λ̃) ∘ ỹ`) recovering exact sums despite online-softmax rescaling.
6. **Systems finding.** Sparse attention loses to CP/SP on latency (sequential chunk decisions); vLLM's PagedAttention cannot express per-head eviction (pages span all heads). Their `smart_lastrec` general variant (content-derived [M0,M1) protected range) measured NO better than the fixed prefix variant (their footnote 11) — fixed β stands.

## 2. Distillation

### 2.1 Path 0 inventory (per-component)

| Component | Training? | Modelless form | Coverage in-tree |
|---|---|---|---|
| Normalized H2O score (mass/age) | No | Closed-form incremental statistic, O(1)/row/step | **NONE.** Signal-diff vs `ega_eviction.rs`: EGA consumes `dot(key[first_head], w_proj)` — static, intrinsic, projection-based key energy; no attention feedback, no age axis. vs `decay_confidence`: `sigmoid(−λ·age)` exponential hazard on *beliefs* — different formula (rate-MLE vs exponential decay) and different domain (belief confidence vs KV row usage). Clean. |
| β-sink pinning (eviction axis) | No | Exclusion predicate on candidate set | Partial cousin: Issue 716 q8kv sink guard pins sinks on the **quantization** axis (f32 sidecar); Issue 487 sink-aware quant. The **eviction** axis (pinned rows excluded from eviction candidates) is unshipped. Composable, orthogonal. |
| Per-(b,h) selection | No | One removed reduction | Our KV layouts are already per-(b,h); nothing sums across heads at eviction. Covered-by-omission — trivial to keep. |
| Summed-attention-weights byproduct | No | Kernel engineering (accumulate w in-register during the softmax tile loop; the two-call trick as fallback) | **NONE.** `attention_cubecl`/cudarc kernels return no column sums. HLA note: linear-attention recurrent state *is* cumulative — usage mass may be readable directly there (unprobed; filed as a probe arm). |
| R/p128 runaway canary | No | Eval-harness statistic + promotion-rule encoding | **NONE as a generation gate.** Issue 750 T3 / Research 502 give per-family conditional *logit* gates (Bench 802 gate 8); no output-length diagnostic exists. Complementary, cheap. |
| Delta-encoded KV chain | No (at inference) | `K_c = scatter(K_{c+1}, I_c, gather(K_c, I_c))` — base + Σ deltas, reconstruct any checkpoint from nearest base | Cousin: `checkpoint_speculative_gpu` full dtod snapshots (Issue 717 seam); Issue 746 established rollback-via-re-execution as correct on append-only KV (deltas would be a *retention* win, not correctness). Consumer pull weak today (tree verify closed-negative; spec K≤16) → **filed, not planned** (riir-ai Issue 836). |
| Replay log | Training device | At inference the decision trace is already computed at eviction time — free telemetry | Telemetry→Beta-LCB policy selection: substrate exists (katgpt-core `rating`, hint_regret); no serving consumer for policy-variant selection → deferred `[-]` in Plan 585. |
| Co-adaptation fine-tuning (replay-log backward, nested checkpointing, LoRA recipe) | **Yes** | None — a frozen base cannot co-adapt; paths 1–3 fail (the correction is statistical output-length discipline, not a closed-form weight transform; deployment vehicle IS the freeze/thaw LoRA swap) | RAT-Plus (201 / riir-train 086) is the **contradicted doctrine**: "train dense, infer sparse" holds for short-context but the paper measures it failing catastrophically for long-context generation. → riir-train Plan 367 (Path 0.5). |

### 2.2 Fusion (the novel combination)

**Paper's φ_norm × our EGA × our sink discipline × our lossy-surface promotion rule:**

"**Energy-seeded, usage-rate-corrected, sink-pinned eviction, canary-gated promotion**" — none of the four alone does this:
- EGA alone is query-agnostic and age-blind (static prior).
- φ_norm alone evicts sinks (age ≈ maximal ⇒ score ≈ 0) — the paper needs lastrec's β pin to make it work; we already own the sink-pin vocabulary (Issue 716) on the quantization axis.
- The fusion: admission prior = EGA energy z-score; online correction = mass/age; sink rows pinned; eviction decision per-(b,h) at matched budget; **no policy promotes without passing the R/p128 canary** — the generation-side extension of the Issue 750 rule.

Second fusion (training side, our own measured instance): Issue 721's drafter acceptance collapse (live-from-BOS 0.05–0.21 vs replay-based 0.8785) is the same train/replay-vs-live mismatch class — co-training the drafter with verify-time context policy active is Plan 367 P5.

### 2.3 Vocabulary translation notes

Paper "KV cache policy / eviction / heavy hitters / attention sink / chunk / prefill" ↔ our `ega_evict` / `compact_kv_cache` / sliding-ring (`transformer_still`, Issue 752 plain-modulo convention) / q8kv `sink_rows` sidecar / `chunk_max` (Bench 710 chunked prefill) / prefill tails. Grep sets run both ways; "cumulative attention" initially hits only `causal_head_importance` (per-head aggregates for bystander detection — different consumer, same signal family: the caller-supplies-mass pattern is the reuse point).

## 3. Verdict

**One verdict per track (the 435/TTPO rule):**

| Track | Tier | Reasoning | Output |
|---|---|---|---|
| **Modelless (a+b) — usage-rate eviction primitive + canary** | **GOAT** (PRIMARY by serving-envelope fit: the score runs at eviction time, inside the hot path) | Provable gain over raw-H2O-class and static-energy eviction at matched budget (paper's own Table 4 shows h2o_norm ≫ h2o on the *untrained* checkpoint); not a new capability class for the *literature* (the paper published the score) — the fusion (EGA×φ_norm×sink-pin×canary) is the novel combination and carries the selling point: *"Our long-context serving evicts by realized attention usage rate with pinned sinks and refuses to promote any policy that fails a generation-behavior canary."* Force multiplier: KV stack (KVarN/Still/sink-guard) + game long-context serving + eval discipline. | katgpt-rs Plan 585 (primitive + canary + GOAT bench). Super-GOAT framing honestly noted on the fusion; the score itself is a distillation, not our invention. |
| **Model-based (c) — policy-in-place co-training** | **GOAT recipe** (SECONDARY: offline training, outside the serving envelope; deployment via freeze/thaw LoRA swap — the runtime stays modelless) | Corrects our own canonicalized RAT-Plus doctrine for long-context generation; enables co-training against our exact Rust serving policies (replay-log = our eviction decision log; discrete policies only); G1 premise probe is a ~1-day falsifier. No prior art on the method (KVPop/Apple-RL learn the *policy*; NSA/DSA bake sparsity at pretraining; DMC bakes a mechanism; LongLoRA reshapes but doesn't compress). | riir-train Plan 367 (Path 0.5 documented: paths 1–3 exhausted, GD genuinely required for the adaptation; dual-track contribution recorded). |
| Fusion idea — delta-encoded KV chain | Issue | Retention optimization; rollback correctness already established via re-execution (Issue 746); weak consumer pull today | riir-ai Issue 836 |

**Discards with auditable reasons:**
- **Bonsai-27B GDN co-training**: GDN's fixed-size recurrent state has no KV eviction; transfer is analogy-only (attention layers + ternary quant noise) — separate future plan only if the transformer ladder lands.
- **`smart_lastrec` general [M0,M1) variant**: the paper's own footnote 11 measured it NO better than the fixed prefix variant — fixed β stands; demote-loser applies to policies too.
- **Cell-grouping memory-floor law (inference side)**: covered — Bench 710 chunked-prefill harness already owns the transient-memory-vs-context curve; the training-side half lives in Plan 367 P2.
- **Kendall-τ per-head disagreement diagnostic**: not discarded — folded into Plan 585 P3 as a bench statistic (decides whether per-(b,h) bookkeeping pays on our workloads).

**MOAT gate:** katgpt-rs in-scope (attention/KV serving primitive; per-stack ledger slot = KV/eviction). riir-ai consumer wiring is pull-gated on Plan 585's GOAT (Issue 836 records the wiring surface). riir-train moat: active (training-method implementations are its charter). No routing conflicts.

## 4. Prior-Art Search Record (§4)

Searched (2026-08-31): headline verbatim; "fine-tuning with KV eviction policy in training co-adaptation"; "normalized H2O heavy hitter recency age cumulative normalize"; in-tree greps (`h2o|cumulative.*attention|summed.*weight`, `lru|recency|residency|age_norm`).

Found and cleared:
- **Apple "Learning to Evict from KV Cache" (arXiv:2602.10238)** — RL-learns the *policy* (opposite direction: model fixed, policy trained). Not coverage.
- **KVPop (arXiv:2607.05061)** — learned cache policy vs future-attention target (xLSTM policies). Learns the policy.
- **Token Retention gates (OpenReview)** — trainable retention scores. Learns scores through gates (model-side training, architecture-coupled).
- **Norm-Guided KV Eviction (ICLR 2026)** — ℓ2-key-norm eviction: the same *static intrinsic* family as our EGA; not usage-rate.
- **NSA/DSA, LongLoRA, DMC, IndexCache, qTTT, OOMB** — pretraining-baked sparsity / reshaping / mechanism-baking / test-time gradients; none do fixed-policy post-training co-adaptation.
- Conclusion: **no published fixed-policy-in-place co-training method prior to this paper; no published age-normalized cumulative-attention eviction score prior to this paper.** Both novelty claims are the paper's; our claims are the fusion + integration + canary discipline.

## 5. Panel Record

Two advocates spawned in one parallel batch with the §4 searches (brief hygiene held: repos described by shipped file/type names, no shape conclusions leaked).
- **No-GD advocate** returned 9 extractions + boundary declaration (merged above; items filed as: P1 score+sink-pin, P2 canary, P3 bench+τ, Issue 836 delta-chain, `[-]` telemetry-selection, `[-]` smart-β, discard cell-law-inference).
- **Model-based advocate** verified baselines on disk (`086_RAT_Plus…`, `087_QAT_LoRA_Fusion…`, `qwen38_kv_f16_gates`, `katgpt-kv/kvarn`, `edge_lora/`, `quest_grammar/{grammar_training,quest_training}.rs`) and returned 7 recipe items with GPU-hour estimates (0.4B bring-up 1–3h → 27B flagship 60–120h on the 4090) and a 5-gate GOAT skeleton whose G1 (dense-trained baseline must show R ≥ 2× under our policy) is the cheapest falsifier. Merged into Plan 367.

## 6. PoC Addendum

None yet. The quality-parity claim ("mass/age ≥ raw-H2O at matched budget") is exactly what Plan 585's GOAT bench tests on a planted age-bias fixture (falsifiable by construction: a config where raw-H2O provably evicts the hot row must exist for the bench to be non-vacuous). The training track's G1 premise probe (Plan 367 P0) is its own falsification-first PoC.

## 7. Cross-Ref (2026-09-03)

> **Cross-ref (Research 528 / Issue 719):** any un-defer of this eviction policy must gate on the **conditioning-consistency audit** (per-junction forward-KL → unconditional Pinsker `TV <= sqrt(eps/2)` between evicted-conditioned and full-context forwards) — hit-rate/quality metrics alone do not measure the train/serve conditioning-gap class (arXiv:2609.00865). T4 of Issue 719.

> **PASS-Redirects (synthesis):** Wang et al. [arXiv:2609.03430 "Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning"] — the null hypothesis this plan's bench lacked: prompt-pinned per-head random matches the strongest scored evictors on reasoning tasks (+32–43% serving throughput, no scoring pass); most inter-selector gap is prompt protection, not the score. Plan 585 addendum (T3.6–T3.9) adds the control arm + protection factorial; distillation in katgpt-rs `.research/531_Random_Attention_Null_Eviction.md`.
> **PASS-Redirects (synthesis):** Muyu He ["Disabling Attention Layers", riddlehe.github.io/blog/disabling-attention-layers.html, 2026-08/09 blog] — mechanism-level support for the mass/importance axis: attention to a token is partly SELF-REINFORCING across positions ("contagion" — a query's attention to target t declines when other queries lose access to t, causally proven by score patching), so accumulated-attention statistics capture load-bearing roles that other tokens' computations depend on — a second-order argument for protecting heavily-attended rows at eviction beyond their own direct contribution. No scoring change required; the finding supports (does not contradict) any mass-based evictor and the sink-protection defaults.
