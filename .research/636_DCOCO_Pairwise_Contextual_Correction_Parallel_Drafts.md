# Research 636: DCOCO — Pairwise Contextual Correction of Parallel Drafts (the candidate-grid correction class, with code)

**Status:** Done — Gain (lane intel → riir-ai Issue 1011 watch note; modelless fusion idea → katgpt-rs Issue 966; discharges the walk-14 C+ row on this arXiv id, riir-refine Research 258)

> **Source:** [arXiv:2610.12570 "Contextual Correction of Parallel Drafts for Speculative Decoding"](https://arxiv.org/abs/2610.12570) — Roy Eisenstadt, Harel Lidar, Tal Gonen, Omer Luxembourg, Itamar Zimerman, Lior Wolf, Eli David, Ilan Naiman (Stealth Startup Tel Aviv / TAU / BGU), 8 Oct 2026. (DCOCO = **D**Flash **C**ontextual **CO**rrect**O**r.)
> **Code:** [github.com/two-delta/DCOCO](https://github.com/two-delta/DCOCO) @ `7a39cf1ec2d273894082463b41ef95d823b4fc7a` (cloned to `.raw/dcoco` this session, **removed at close**). License: per-file Apache-2.0 SPDX headers ("Copyright contributors to the vLLM project") — the repo ships **no root LICENSE file**, provenance is per-file-header only. Python/vLLM overlay on `vllm-openai:v0.31.0` (docker), NOT Rust — **distill axis: corpus NO** (no domain accepts a Python vLLM plugin; the FlashMap precedent). Models: `TD-Labs/Qwen3-8B-DCOCO`, `TD-Labs/Qwen3-4B-DCOCO` (vLLM-native checkpoints, block size 15+anchor).
> **Date:** 2026-10-26 (zed-m3max/M3, research + distill skills, code read at the pin)
> **Related Research:** 634 (FlashMap — the joint-head acceptance arm, the sibling verdict this note extends), 316 (DSpark + the bigram path-2 redirect → Issue 659), 407 (DFlash-TfM acceptance ceiling), 002 (speculative decoding); riir-refine: 077 (DFlash2 GGUF facts — no `tok_embd`/`lm_head`, encoder consumes 5×5120 target features/position), Bench 043 (the DFlash2 selection-headroom POC — clippy domain, T1 negative *there*), 258 walk-14 row 10 (the C+ ecosystem grade this note discharges)
> **Related Plans/Issues:** [riir-ai Plan 614](../riir-ai/.plans/614_dflash2_spec_lane_m5_ultra.md) + [riir-ai Issue 1011](../riir-ai/.issues/1011_league_spec_decode_stance_tensorfold_watch.md) (the lane; watch note appended this session), katgpt-rs Issue 659 (bigram_markov — the modelless prior substrate), Plan 200 / `correlation_budget.rs` (per-depth acceptance EMA — the coverage-instrument vehicle), katgpt-rs **Issue 966** (filed this session: the modelless fusion idea)
> **Classification:** Public

---

## TL;DR

DCOCO is the **second paper in the candidate-grid correction class** (LiLiCorr, arXiv:2608.20530, is the first — same DFlash backbone + top-K retention + a learned correlator, already in vLLM stable): keep the parallel drafter's **top-K candidates per slot**, score all **(B−1)K² adjacent-pair transitions in one batched pass** with a tiny MLP (score = successor's own draft logit + learned pair delta), then decode the grid with a **prefix-first greedy walk**. No sequential module on the critical path — unlike Domino/DSpark, whose per-position recurrence loses to plain DFlash at ≥16 concurrent streams. Measured on four verifiers × ten benchmarks: best average speedup **in all twelve single-stream cells, by 8–19% over the strongest baseline in each** (paper's Tab-2 discussion); **+22.6% over DFlash** specifically on Qwen3-8B T=0 (959.7 vs 783.1 tok/s, 3.62× vs 2.95×), up to 6.6× over vanilla; under concurrent serving **+6–23% over the strongest competitor across 4–32 streams**; vs the uncorrected drafter it pays **3–4% cycle cost for 18–27% longer accepted prefixes**. The load-time implementation factoring (per-candidate weight split, `materialize_inference_buffers`) makes the K² grid cost a few percent of the drafting cycle.

**For our lane (riir-ai Plan 614 / Issue 1011), three payloads:**

1. **The coverage evidence** — top-16 coverage ≫ top-1 everywhere, and the gap GROWS deeper in the block (0.12–0.14 at slot 1 → 0.38–0.45 at slot 15 on every verifier; first slot ≈95% vs ≈81%). The class premise is now the best-evidenced number in the DFlash ecosystem — but it is measured on DENSE drafter features. Our posture's taps are quant-drifted (746/747); **the per-slot coverage measurement on OUR GGUF posture is the open gate** for this whole class (instrument: extend `correlation_budget.rs`).
2. **The cheapest model-based acceptance arm on record** — the **frozen-backbone ablation**: corrector trained as a pure drop-in on a FIXED drafter = **+4.4% throughput** (783.1 → 817.7 tok/s) for ~2.2M params; joint training adds ≈15% more. The corrector is ADDITIVE to any drafter (backbone unchanged) — it composes with A0's drift fix and with any future A1 head class.
3. **The selection law** — prefix-first greedy **beats Global-Sum (Viterbi-style max-sum) on all nine suites** and ≈ geometric-discounted (λ=0.75) on acceptance with **+2.8% throughput everywhere**: under prefix acceptance, the aggregate-path objective is the WRONG objective. League intel for the prism fork's #284 selector-lattice beam (width 2–4): the beam may be optimizing the wrong thing; a prefix-first-greedy ablation is a cheap kernel-side candidate.

**Modelless fusion idea filed** (Issue 966, novelty TBD): prefix-first re-selection over `dflash.rs`'s factorized grid, transition-scored by `bigram_markov.rs`'s static corpus table (top-m=16 — the exact K scale, already shipped, Issue 659) — zero trained parameters beyond the drafter. The paper's own ablation supports weak-signal robustness (−pair-state still retains most of the gain; a candidate-blind variant still beats DFlash).

**Verdict: Gain, not Super-GOAT.** Same shape as R634: the mechanism is model-based (a trained pair head), the class has published prior art (LiLiCorr), the in-stack lane consumes it as intel + one training option; no new katgpt-rs primitive ships from the paper itself — the modelless variant is OUR fusion (Issue 966, gated on the coverage premise).

---

## 1. Paper core (compact)

**The failure being corrected.** A parallel (diffusion) drafter proposes the whole block q(y₁:K|c) = ∏qₖ(yₖ|c) — each slot is predicted without the *realized identities* of preceding slots, so individually plausible tokens form incoherent combinations ("New York" + "San Diego" → "New Diego"). Autoregressive drafters (EAGLE family) keep the dependencies but pay sequential latency; DSpark/Domino restore dependencies with a *sequential* correction module (rank-256 Markov head / causal-encoder width 1024 + rank-256 correction) whose latency sits on every drafting cycle's critical path.

**Coverage (§3, the premise).** Top-k coverage = fraction of drafted slots whose top-k candidates contain the token the verifier eventually commits. Qwen3-8B at T=0: top-16 coverage stays **>0.60 at the LAST slot** while top-1 falls **<0.16**; the top-16−top-1 gap grows **0.12–0.14 (slot 1) → 0.38–0.45 (slot 15) on every verifier** (Tab 18: first slot top-16 ≈95% vs top-1 ≈81%; last slot pool 57–61% vs ~one-in-five). *"The drafter does not lack the right tokens; it fails to select them, increasingly so deeper in the block."* Same regularity as DFlash2's own Table 1 (Recall@1 85.4% / Recall@16 99.5% — riir-refine Bench 043's premise), now measured per-slot across four verifiers.

**Mechanism (§4, verified against the code).** Per slot i, DFlash yields hidden state hᵢ ∈ ℝ^d and top-K candidates {cᵢₖ} with draft logits ℓᵢ(k). Transition score:

```
S_p(a,b) = ℓ_{p+1}(b)  +  Δθ(s_p, z_pa, z_{p+1,b}, ℓ_p(a), ℓ_{p+1}(b))
  z_ik = LN_γβ(W_fuse [ LN(W_cand E[c_ik]) ; LN(W_ctx h_i) ])        (candidate repr; E = FROZEN target embedding table)
  s_p  = W_pair [ h_p ; h_{p+1} ]                                    (candidate-independent pair state, computed once per adjacent pair)
  Δθ   = two-layer MLP (GELU), input [s_p; z_pa; z_{p+1,b}; ℓ_p(a); ℓ_{p+1}(b)]
```

All (B−1)K² = 14×16² = **3584** transitions scored in ONE batched op; the **zero-initialized output layer** means Δθ≡0 at init → every score reduces to the successor's draft logit → prefix-first selection reproduces the drafter's argmax exactly (step-0 equivalence, asserted in both paper and code docstring).

**Prefix-first greedy walk (§4.3, Eq. 7).** k₁ = argmaxₖ ℓ₁(k); k_{p+1} = argmax_b S_p(k_p, b). O(BK) sequential walk on top of the O(BK²) parallel scoring — *"only this lightweight selection is sequential, while neural scoring remains parallel."* Motivation: verification accepts a contiguous prefix and stops at the first rejection, so a correction at slot 5 behind a rejection at slot 1 is never examined — selection must prioritize the early prefix, not an aggregate path score.

**Training (App A).** Init from a 6-epoch DFlash checkpoint; +6 joint epochs (drafter + corrector, drafter loss + correction loss unit weight). Correction loss: exponentially down-weighted (ρ=4) cross-entropy over the K² jointly-softmax'd transition scores; label = the pair of verifier-argmax tokens at two adjacent slots **when both lie inside their slots' top-K** (masked otherwise — unreachable transitions are dropped from supervision, never taught); embedding table frozen; optimizer AdamW 6e-4, cosine, 4% warmup; ~800K prompts (Nemotron PT v2 stem/math/code + CodeAlpaca-20k), continuations regenerated per verifier (greedy). Training is offline teacher-forced — no online speculative loop; the paper argues verifier-correct prefixes are precisely the states where the corrector's output is ever consumed (prefix acceptance), so this is not a train/serve mismatch.

**Results.** Qwen3-8B T=0 no-thinking ten-suite averages (TPS, B200, concurrency 1): DFlash 783.1 (2.95×) · Domino 839.3 (3.16×) · DSpark 765.6 (2.88×) · **DCOCO 959.7 (3.62×)** — +22.6% over DFlash, +14.3% over Domino; up to 6.6× over vanilla (AIME-class). DCOCO best in **all twelve** verifier×thinking cells (8–19% over the strongest baseline); best accepted length in 11/12. Vs uncorrected DFlash: **3–4% cycle cost for 18–27% longer accepted prefixes**.

**Ablations (App B).** −pair state: τ −7.3%/−5.6% (T=0/T=1), TPS −6.3%/−4.6%. −candidate reprs: τ −4.8%/−3.7%, TPS −4.6%/−3.3%. **Frozen backbone** (corrector as pure drop-in on the fixed 6-epoch DFlash): 783.1 → 817.7 tok/s (**+4.4%**); freezing costs 15.9%/12.9% vs joint; the joint phase is where the arm's gain accrues (baselines extended to 12 epochs move ≤1.9% — converged, not budget).

**Selection-rule ablation (§5.3, Tab 3).** Over the SAME trained scores: **Global-Sum** (max-sum path) yields LOWER accepted length AND throughput than **prefix-first Greedy** on all nine evaluated suites; **Geometric(λ=0.75)** ≈ Greedy on acceptance (5.258 vs 5.254 avg) but Greedy wins throughput on every suite (+2.8% avg). The prefix structure of verification, not score quality, decides the walk.

**Concurrency (§5.2, Tab 19).** Qwen3-8B T=0 aggregate throughput: DCOCO/DFlash = 1.23× (c=1) · 1.25× (4) · 1.24× (8) · 1.23× (16) · **1.19× (32)**; margin over the strongest alternative +14.3% → +23.0% → +19.2%. Domino and DSpark progressively lose and **both fall below the uncorrected drafter** at high load (their per-cycle sequential overhead grows while the verifier's cycle amortizes across streams). DCOCO's per-cycle overhead stays thin because it is one batched pass + an O(BK) walk.

**Lossless stochastic decoding (App G).** DCOCO selects deterministically even under T=1 — the proposal is a point mass; standard speculative sampling then accepts ŷⱼ with probability qⱼ(ŷⱼ) and resamples the renormalized-excluding residual on rejection. Target distribution preserved exactly; *"stochastic target decoding does not require stochastic draft selection."* The transition scores are never softmax-normalized at inference.

**Corrector size (App A).** d_e=128, d_h=128, K=16; MLP input 386 → hidden 256 (GELU) → scalar. Qwen3-8B (d=4096): **2,229,505 params**; Qwen3-4B: 1,443,073. BF16 inference. At our target width (d=5120, the Qwen3.8-27B tap width per Research 077) the same architecture is ≈2.8M params.

## 2. Code read (what the repo adds beyond the paper)

Pin `7a39cf1`; all quotes verbatim from the clone.

- **`docker/overlay/vllm/model_executor/models/dcoco.py`** (267 lines) — `DCOCOHead` + `DCOCO(DFlashQwen3Model)`. Docstring carries the scoring equations verbatim (identical to paper Eq. 2–5). The headline implementation fact: **`materialize_inference_buffers` splits `trans_mlp[0]`'s weight into state/predecessor/successor blocks at load time**, so the first layer evaluates as `W_state@state + W_prev@[pe_prev,c_prev] + W_cur@[pe_cur,c_cur]` — *"computes each candidate's contribution once instead of materializing the K*K concatenation."* The K² grid then costs one broadcast add (`pre.unsqueeze(-2) + cur.unsqueeze(-3)`) + the MLP tail. This is the "few percent of the drafting cycle" claim's mechanism.
- **Consumed inputs are exactly four**: candidate token ids → `target_embeddings(candidate_ids)` (the **target's** frozen embedding table — the DFlash2 GGUF has no `tok_embd`, but the target GGUF does, and LiLiCorr's binding supplies it); raw candidate logits ℓ (a dedicated `LogitsProcessor` — *"DCOCO is trained on raw lm_head logits: no scale, softcap or softmax"*; `has_own_lm_head` optional — sharing the target head is the default, which is what prism fork #30111 "DFlash output head sharing" already does); drafter hidden states h; the top-K sets. **All four exist in our fork posture.**
- **`DCOCOForCausalLM` refuses truncated vocab** (`draft_id_to_target_id` → ValueError: "DCOCO candidates require the full target vocabulary") — the corrector reads raw candidate ids, never remapped ones.
- **`speculate/dcoco/speculator.py`** (35 lines) — `DCOCOSpeculator(LiLiCorrSpeculator)`: inherits LiLiCorr's **target embedding/lm_head binding + the shared candidate walk**; overrides only `_score_candidates`. `prepare_context_anchor` is a no-op ("DCOCO reads no context anchor"); `_supports_sample_from_anchor = True` (the anchor can be a prediction slot — hence block_size 16 → 15 drafted + anchor).
- **Walk semantics** live upstream in vLLM's LiLiCorr speculator: greedy prefix-first (Eq. 7) or temperature-sampled for T=1 serving; row 0 of the score table = slot-0 raw logits broadcast over predecessors.
- **`dflash/speculator.py`** (809 lines) — the uncorrected baseline: independent per-slot top-1 commit (the thing DCOCO's walk replaces).
- **`eagle.py` config** — `method == "dflash"` routes architectures named `*DFlash*`, `LiLiCorrDraftModel`, `DCOCODraftModel`; per-draft config keys `dcoco_candidate_topk` / `dcoco_emb_dim` / `dcoco_state_dim` / `dcoco_mlp_dim`.
- **No root LICENSE file** — per-file Apache-2.0 SPDX headers only (recorded for provenance; quote-reverify at the pin survives because the pin is the clone sha).

## 3. Class check — LiLiCorr is the prior; DCOCO's deltas

| Axis | LiLiCorr (2608.20530, NVIDIA, Aug 2026; vLLM stable) | DCOCO (this paper) |
|---|---|---|
| Backbone | DFlash, top-K per slot | same, unchanged |
| Scorer | learned correlator over candidate combinations | pairwise two-layer MLP over (pair state, candidate reprs, raw logits), zero-init |
| Selection | (vLLM-shared candidate walk) | **prefix-first greedy**, with the Tab-3 evidence it beats sum-max |
| Evidence | 4 citations | 4 verifiers × 10 benchmarks + coverage analysis + concurrency + ablations |

For OUR purposes the CLASS is the unit, not the paper: **the grid-correction class is established, mainline-shipped (vLLM stable since ~Sep 2026), and its premise (top-K ≫ top-1 coverage) is now the best-evidenced number in the DFlash ecosystem.** DCOCO's contribution over LiLiCorr is the selection law + the evidence corpus; the class premise transfers regardless of which scorer wins.

Closest in-corpus cousins (both layers):

| DCOCO delta | Closest cousin | Signal-diff (what the cousin does NOT cover) |
|---|---|---|
| Top-K grid retention + re-selection under prefix acceptance | riir-refine Bench 043 (the DFlash2 selection-headroom POC) | Bench 043 measured the premise in the **CLIPPY domain** (healer rule selection) and found it already harvested there by the T3.5.5 instantiation re-ranker — T1 negative *for that domain*. The DECODING domain on our quant posture is **unmeasured**; the class survives the clippy-domain kill. |
| Learned pairwise transition scorer | `domino.rs` / `domino_lora.rs` (in-stack causal correction, R316 family) | Domino's correction is a sequential per-position recurrence (the paper's own losing class under load); DCOCO's scorer is one batched pass. Different cost point; both are model-based. |
| Prefix-first walk beats aggregate-path selection | prism fork #284 `selector-lattice-beam-over-greedy-walk` (riir-refine B198 corpus rule; beam width 2–4 over the DFlash2 selector lattice) | The beam optimizes a per-step path score on the SELECTOR lattice during drafting; DCOCO's law says aggregate-path objectives are the wrong objective under prefix acceptance **at equal signal**. An ablation (beam vs prefix-first-greedy at the fork's walk) is the transferable test — different surface, same objective question. |
| Per-slot coverage instrument | `correlation_budget.rs` (Plan 200, per-depth acceptance EMA) | The shipped allocator tracks ACCEPTED-depth agreement; it does not track top-k pool coverage (whether the committed token was even IN the drafter's top-k at that slot). The coverage extension is small and is the A0-premise gate for the whole class (Issue 1011 watch note). |
| "Parallel correction survives load; sequential does not" | R634 FlashMap verdict (2-NFE joint head, flat conditional curve) | Third independent angle on the same lane stance: FlashMap (head class), DCOCO (grid corrector), lithos-DSpark note (GPU-resident verify) — all parallel-structure-preserving mechanisms; the lane's no-sequential-module posture (S2/S4) is now corroborated by the competitor class's own failure mode under load. |

**Published prior art (§4 searches, 2026-10-26):** headline ("DCOCO contextual correction parallel drafts") + component (candidate-grid correction, LiLiCorr 2608.20530 found — the class prior, cited) + selection-rule (prefix-first vs Viterbi spec-decode: no published comparison beyond this paper's Tab 3) + training-free re-selection class (FLy 2609.02897-class = entropy-gated drafting; Training-Free Loosely Speculative = semantic acceptance loosening; Per-Step Lossy = lossy tree verification — **none is training-free grid RE-SELECTION**; the Issue-966 claim is pinned narrowly against exactly these). No prior art kills the lane-intel verdict; the modelless fusion claim is filed novelty-TBD with a re-search obligation at execution.

## 4. Why not Super-GOAT (novelty gate Q1–Q4)

- **Q1 (no prior art): NO.** The class has published prior art — LiLiCorr (2608.20530), mainline-shipped. DCOCO's increments (pairwise MLP shape, prefix-first law, coverage evidence) are real but sit inside an established class.
- **Q2 (new behavior class): NO.** Block drafting + correction both exist in-stack conceptually (`dflash.rs` + `domino.rs`); the grid corrector is a new member of a known class, not a new capability.
- **Q3 (product selling point): NO.** Decode-throughput serving concern; no per-NPC behavior signal. Healer/consumer reframe (priority #2): the healer's own selection surface harvested its selector headroom already (Bench 043 T1) — no healer surface here.
- **Q4 (force multiplier): the lane, yes; the corpus, no.** Multiplies the ACTIVE Plan-614 lane's option space (recorded as lane intel + one owner-gated training option), and seeds Issue 966's modelless fusion — but no pillar connection in katgpt-rs itself.

⇒ **Gain.** The note exists to (a) discharge the walk-14 C+ row with the code now read, (b) route the lane intel + the coverage-gate obligation where the lane lives, (c) file the modelless fusion idea properly (Issue 966), (d) prevent re-evaluation.

## 5. Track split (§3.5 discipline — one verdict per track)

**Track a — modelless extractions (recorded; the fusion idea filed):**

| Extractable | Shipment path | Status |
|---|---|---|
| Prefix-first > aggregate-path selection law under prefix acceptance | a decision law for ANY prefix-verified candidate grid (dd_tree deep-argmax Plan 424; the fork's #284 beam ablation; Issue 966's walk) | **Filed** as Issue 966's walk + recorded as league intel for the beam ablation |
| Per-slot top-k coverage instrument | extend `correlation_budget.rs` (per-depth EMA → + per-depth top-k-pool coverage of the committed token) | **The A0-premise gate** — recorded in the Issue 1011 watch note as the measurement that must run BEFORE any corrector arm |
| Lossless point-mass stochastic recipe (App G) | a verification-side recipe for any deterministic-proposal drafter (already the shape `sample_from_distribution` consumes) | Recorded — validates the in-stack contract; no code change |
| Modelless grid re-selection (bigram prior over dflash grid) | `dflash.rs` grid + `bigram_markov.rs` table + prefix-first walk; feature flag `grid_reselect`; GOAT G1–G4 in the issue | **Issue 966** (fusion idea, novelty TBD, gated on the coverage premise) |

**Track c — model-based (the training option, owner-gated):** a DCOCO-class pairwise corrector trained **ON our quant-drifted taps** (in-distribution on the drift the lane diagnosed), frozen-backbone first — the paper's frozen ablation (+4.4% for 2.2M params on a fixed drafter) is the evidence the drop-in class works without co-adaptation. Order: **coverage-premise measurement → frozen corrector → (optionally) joint**. Cost basis: 6 joint epochs on 800K prompts × 8 GPUs in the paper; the corrector-only phase is far smaller (2–3M params, labels = verifier-argmax pairs already computable from our verify path) — a 4090-class project, not a multi-day 32×H200 one. **Owner-gated riir-train**; rides the Issue 1011 watch note as the third named option after A0 (drift arm) and A1 (head class), and composes with both (the corrector is drafter-agnostic). GGUF/fork port of the head (the fork's dflash path would need top-K retention + the tiny MLP + the walk in CUDA) is a separate artifact decision — only after the premise gate.

## 6. Lane intel → Plan 614 / Issue 1011 (the actionable core)

The lane's state (unchanged by this note): acceptance is the binding constraint (live L_eff 2.83 doc-prose vs the 2.63–3.33 flip band at C_K 1.0–1.3, Plan 614 Phase-3/Bench 957 §6), caused underneath by target-quant feature drift at the taps (Benches 964/973, 746/747). DCOCO does not address the drift cause — **same honest reframe as R634: it is a proposal-quality (τ) lever, a SECONDARY acceptance lever.** But it adds three things the lane record did not have:

1. **The premise gate, made concrete.** The class works only where the drafter's top-K pool contains the committed token. On dense features: ≈95% pool coverage at slot 1, 57–61% at slot 15. On OUR drifted taps: unmeasured. If drift collapses pool coverage (not just top-1), NO corrector — modelless (Issue 966) or trained — can harvest, and the class closes for this lane with one cheap measurement. The measurement: per-slot top-k coverage of the verifier-committed token over our GGUF + live L_eff harness (Bench-043-shaped, decoding domain; extends `correlation_budget.rs`).
2. **The cheapest arm on record.** Frozen drop-in corrector: +4.4% (dense, matched posture) for ~2.2M params — an order of magnitude cheaper than any head/drafter retrain the lane has priced, and ADDITIVE to whatever drafter posture wins (DFlash2 as-is, A0's drift-fixed head, a future A1 joint head — the corrector reads only hidden states + logits + top-K + the target embedding table).
3. **The selection law as league intel.** Prefix-first greedy > sum-max on every suite at equal scores. The prism fork's DFlash2 selector carries a beam (#284, width 2–4, log-softmax per-step, lexicographic ties) — an ablation of that beam against plain prefix-first greedy is a cheap kernel-side candidate for the next fork-window intake (kernel_opt/rust_perf rider class, quote-verified at the fork pin at composition).

**NOT transferable:** absolute numbers (B200, dense BF16, Qwen3-8B/4B vs our PQ2_0 27B on 4090-now/M5-Ultra-production — the standing league caveat); the vLLM-native checkpoints (wrong format for the fork lane; a GGUF/fork head port is a new artifact behind the premise gate); Domino/DSpark comparative numbers (sequential class, out of lane posture by the S2/S4 stance); tree-verify variants (closed-negative, Issue 1068 — DCOCO is chain-shaped anyway, no tree).

**Stance: UNCHANGED** — the drafted axis stays COMPETED (S2/S4); this note widens the option space (premise gate → frozen corrector → joint; Issue 966 modelless arm) and adds one fork-intake lead (the beam ablation). Zero GPU this session (M3 research pass; the premise measurement runs on the league box, the training arms are 4090-class).

## 7. Fusion (the §1 step-6 subsection)

paper × A × B: **DCOCO's selection law × `bigram_markov.rs`'s corpus-built transition table × `dflash.rs`'s factorized grid** = a zero-training grid corrector: S_p(a,b) = ℓ_{p+1}(b) + λ·log P_bigram(b|a), prefix-first walk, O(BK) — none of the three alone does this (the law needs a second signal; the table has no consumer on a grid; the grid has no cross-slot selection). That is Issue 966. Secondary fusion recorded, not filed: the coverage instrument × correlation_budget (lane measurement); the selection law × the fork beam (league ablation lead).
