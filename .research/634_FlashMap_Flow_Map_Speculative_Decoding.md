# Research 634: FlashMap — Flow Map Speculative Decoding (the joint-head evidence for the DFlash2 acceptance axis)

**Status:** Done — Gain (lane intel routed to riir-ai Issue 1011 / Plan 614; training redirect owner-gated; discharges the walk-16 watch row)

> **Source:** [arXiv:2610.24365 "FlashMap: Flow Map Speculative Decoding"](https://arxiv.org/abs/2610.24365) — Jeff Shen, Zhejian Zhou, Ryan Lee, Jonathan May, Shirley Ho (Princeton / USC ISI / Polymathic AI / NYU / Flatiron), 17 Oct 2026, CC BY 4.0.
> **Code:** [github.com/al-jshen/flashmap](https://github.com/al-jshen/flashmap) — MIT (code) / Apache-2.0 (weights, following Qwen3). Python (torch + transformers) + SGLang/vLLM plugins; training recipes in speculators + SpecForge forks; drafters `j-shen/FlashMap-Qwen3-8B` (500M) / `-4B` (475M). Paper implemented in DeepSpec (deepseek-ai, MIT) @ `005e03b`. **NOT cloned** — pure Python, no Rust surface to mine (distill axis: corpus NO).
> **Date:** 2026-10-25 (zed-m3max/M3, research + distill skills)
> **Related Research:** 366 (Self-Conditioned FMLM★ — the family check this note performs per R366 §5's standing instruction), 265 (CoFRe, direct FMLM sibling), 407 (DFlash-TfM — Weaver + the factorized acceptance ceiling + GDN tree-verify), 316 (DSpark — confidence-scheduled semi-AR post-processing), 002 (speculative decoding, original), 344 (Implicit FP RNN — closed the FP-for-LM design space)
> **Related Plans:** [riir-ai Plan 614](../riir-ai/.plans/614_dflash2_spec_lane_m5_ultra.md) (the DFlash2 spec lane this note feeds), 424 (dd_tree deep argmax), 200 (correlation budget), 108/136 (LoopMode FP blocks), [riir-ai Issue 1011](../riir-ai/.issues/1011_league_spec_decode_stance_tensorfold_watch.md) (the lane-intel issue — watch note appended this session)
> **Classification:** Public

---

## TL;DR

FlashMap is the **first speculative-decoding drafter built on a continuous flow**: a DiT flow-map language model (FMLM-class) conditioned on the target's hidden states (DFlash's conditioning, DFlash's frozen embedding + LM head), sampling the whole K=16 block **jointly in one forward pass** — replacing DFlash's factorized discrete-diffusion head, whose per-slot independence leaves the total-correlation KL that "no amount of model capacity removes." Against a DFlash retrained under identical data/settings: **+7.4% avg speedup at 1 NFE (up to 6.14×) with HALF the drafter parameters** (0.5B vs 1B), **+10% acceptance at 2 NFE (18/18 settings)**, and a **flat conditional-acceptance curve within the block** where DFlash decays. The analysis instrument — the **shuffle test** (Δ = γ_shared − γ_shuffled) — attributes acceptance to marginals vs joint structure and yields the paper's key mechanism finding: **a raw flow at 1 NFE captures ZERO joint structure; integration over steps is what resolves noise into a coherent block** (16-step teacher Δ ≈ ~1 token; the 2-NFE flow map recovers most of it; the 1-NFE map only a small fraction).

**Verdict: Gain, not Super-GOAT.** The flow-map family is saturated in this corpus (R366/265/344 — R366's own §5 instruction mandated this exact check); the in-stack speculative lane (dflash + dd_tree + keyed walk) already exists; the modelless-extractable deltas are instrument/config-level; the paper's value is model-based (a trained drafter, ~4,600 GPU-h total). But it is **the highest-relevance lane intel of the walk series**: it lands on riir-ai's ACTIVE owner-ordered DFlash2 spec-decode lane (Plan 614, Phase-3 front), whose own record names acceptance (L_eff — the LIVE metric) as the binding constraint, and FlashMap's controlled A/B is evidence that head class moves acceptance on dense-BF16 targets — **a secondary lever; the lane's named binding cause is quantized-tap feature drift (746/747), addressed first by training arm A0** (the joint flow-map head, A1, only after A0, so the head-type increment reads against a drift-free baseline). Routed: Issue 1011 watch note + this note; both training arms stay **owner-gated riir-train** (the semigroup-distillation recipe family is already routed there by R366).

---

## 1. Paper core (compact)

- **Factorized approximation:** a one-pass discrete-diffusion drafter proposes q(y₁:K|c) = ∏qₖ(yₖ|c); per-slot CE is minimized by the marginals; the residual gap is the total correlation KL(p(y₁:K) ∥ ∏p(yₖ)) — structural, not capacity. ("new york"/"san diego" → "new diego"/"san york" half the time.)
- **Method:** rectified flow over one-hot draft blocks (I_t = (1−t)x₀ + t·x₁, x₀ ~ N(0,I)); denoiser D_t predicts clean data (low-rank), CE-trained on the simplex; **flow-map distillation** (Boffi et al. semigroup framework — the same framework R366 recorded for FMLM★): two-time denoiser δ_{s,t}, student anchored to the frozen teacher at zero-length jumps, bootstrapped to long jumps via the stop-gradient midpoint composition (Eq. 7); one evaluation δ₀,₁ drafts the whole block.
- **Conditioning (DFlash-style):** target hidden states from layers {1,9,17,25,33}, concatenated + projected, plus the anchor-token embedding; adaLN-Zero time modulation; self-attn over K slots + cross-attn to context; frozen target embedding + LM head.
- **Time reparameterization:** decoding-error τ(t) (equal steps = equal token-resolution progress), closed form P_e(t) via the normal CDF (Eq. 6) — adopted from FMLM (R366 family).
- **Appendix A featurization:** embed the posterior π_t(xᵏ) = softmax(t·xᵏ/(1−t)²)·E — "the natural choice, the linear read-out xᵏE … the signal is drowned out"; the 1-NFE student adds a raw-state path (t−s)·xᵏE because π₀ is noise-blind (a flow map evaluated at s=0 NEEDS the noise as input or it reverts to factorized marginals — the same mechanism the shuffle test exposes).
- **Appendix B fast multi-step:** precompute noise-only vocabulary projections W(z), S(z) for a bank of 8,192 noise draws; the second jump reads |Tᵏ|=16 rows of E per slot instead of |V|=151,936 → ~70% cheaper second jump, acceptance essentially unchanged.
- **Results (Qwen3-8B, 9 benchmarks, H200, SGLang c=1):** 1-NFE FlashMap fastest on every task (6.14× GSM8K T=0); acceptance > DFlash in 15/18 with half the params; largest wins on chat (+7.4% acc / +11.1% speedup at T=0). 2-NFE: best parallel-drafter acceptance, 18/18, flat conditional curve. EAGLE-3 (TTT=7, tree 16): higher acceptance at T=1 (6/9) but 35% slower — AR latency dominates.
- **Shuffle test (Table 2):** shared vs per-slot-independent noise. Flow @1 NFE: Δ=0. Flow @16 steps: Δ ≈ ~1 token. Flow map @2 NFE: recovers most of the teacher's Δ. ⇒ joint structure lives in INTEGRATION; the map approximates it.
- **Multi-sample drafting (Appendix D):** several blocks from independent noise draws, merged into a trie, verified as one tree — +11% avg speedup (4–24%) at the same verification budget. ("Similar to DDTree, but every branch comes from a separate noise draw.")
- **Compute:** 8B teacher ~25 h on 32×H200; distill ~9 h; ~4,600 GPU-h across all drafters/ablations. Limitations: one model family, 500K-sample subset, c=1 only, K=16, possible undertraining (FMLM trained 1M steps vs their 39K+23K).

## 2. Family check (the R366 §5 mandate) — cousin table

R366's standing instruction: *"A future paper in this family (e.g., a new flow-map distillation variant) should be checked against R265 + R366 before any new note is created."* This is that check.

| FlashMap delta | Closest in-corpus cousin | Signal-diff (what the cousin does NOT cover) |
|---|---|---|
| Flow-map distillation of an LM (teacher → δ_{s,t} student) | R366 (FMLM★): CDEQ + semigroup distillation, **routed riir-train** | Same recipe family, already routed. FlashMap adds no new distillation math (adopts Boffi verbatim). The INFERENCE path R366 names as "already shipped" (LoopMode::WeightShared + is_converged halt) is the standalone-LM shape, not a drafter. |
| Target-conditioned drafter (hidden-state cross-attn, frozen embed/LM head) | `katgpt-speculative/src/dflash.rs` (factorized block drafter, in-stack) + z-lab DFlash2 checkpoint (Plan 614 substrate) | The cousin's head is discrete-diffusion FACTORIZED; FlashMap replaces the head class. This is the lane-intel payload (§5), not a modelless primitive. |
| Joint-vs-factorized acceptance decomposition (shuffle test Δ) | `correlation_budget.rs` (Plan 200): per-depth acceptance EMA — the ONLINE twin of FlashMap's Fig. 3 conditional-acceptance curves | The shipped allocator tracks per-depth agreement but CANNOT separate marginals from joint structure — it has no noise-manipulation arm. Δ does not ship anywhere. Instrument-level only (needs a noise-dependent drafter to exist; factorized Δ=0 by construction). |
| Late-slot acceptance decay of joint drafts | R407 (DFlash-TfM): factorized acceptance ceiling — argmax-of-marginal beats full-marginal sampling at depth (Plan 424) | The two results are two views of ONE axis: R407 shows per-slot argmax maximizes conditional acceptance at depth; FlashMap shows a 1-NFE joint sample is NOT the per-slot argmax, hence decays vs DFlash — while a 2-NFE joint recovers dependence and flattens. No contradiction; FlashMap completes R407's ceiling from the joint side. |
| Post-hoc joint recovery from a factorized draft | R316 (DSpark): confidence-scheduled semi-AR post-processing; `domino.rs`/`domino_lora.rs` (causal correction); Weaver (`weaver.rs`) | Those add sequential computation AFTER a parallel draft; FlashMap moves the joint structure INTO the drafting pass. Different cost point on the same spectrum. |
| Multi-sample noise-draw trie drafting | `dd_tree/` (build_dd_tree over marginals; Plan 012/424) | dd_tree branches on marginals; FlashMap branches on noise draws batched in one pass. Also: tree VERIFY is closed-negative in the Plan-614 lane (Issue 1068 records draft-chain-only) — not transferable there. |
| "flow" naming | `nf_flow.rs` (NFCoT FlowScore, Plan 229/R204) | A normalizing-flow density SCORER over DDTree marginals — different mathematics entirely (scoring, not generative transport). Named here only to kill a vocabulary false-positive. |
| Prior distilled FMLM-family verdicts | R366 (Gain, leaning Pass), R265 (CoFRe, Gain), R369 (renoise-CE, GOAT — a different delta), R344 (design-space closure) | FlashMap is an APPLICATION paper (drafter), not a new flow-map math paper — the family check resolves to: no new flow-map math; new application surface = lane intel. |

**Published prior art (§4 searches, 2026-10-25):** the paper's own claim — "to our knowledge, the first speculative decoding drafter built on a continuous flow" — held across headline + component searches (flow-map/consistency + spec-decode drafting; joint block sampling; target-conditioned flow drafter). The competitive set it names (DFlash 2602.06036, DSpark 2607.05147, Domino 2605.29707, draft trees 2604.12989, DBLAST, DeLS-spec, DiffuSpec, SpecDiff-2) all recover joint structure post-hoc or factor-restricted — none drafts jointly in-pass. Our workspace watched the cluster at C+ grade (riir-refine Research 311 walk-16); this note is the discharge.

## 3. Why not Super-GOAT (novelty gate Q1–Q4)

- **Q1 (no prior art): NO for the family, YES for the application.** The flow-map LM family is saturated (R366/265/344/369); the drafter application is new in the literature but new-in-corpus ≠ a new primitive for us: there is no flow-drafter substrate in-stack to consume the deltas.
- **Q2 (new behavior class): NO.** Spec-decode block drafting ships (dflash lane); tree drafting ships (dd_tree); the joint class would be a head swap inside the existing lane, not a new capability.
- **Q3 (product selling point): NO.** Serving-layer decode acceleration; no per-NPC behavior signal (game-context reframe honestly thin: faster decode = cheaper NPC cognition serving, nothing more). Healer/consumer reframe (priority #2): no healer surface — the drafter is token-level; the corpus/selection machinery has no analogous joint-sampling seam (nf_flow's scorer analogy is a name collision, not a mechanism transfer).
- **Q4 (force multiplier): the lane, yes; the corpus, no.** It multiplies the ACTIVE Plan-614 lane's decision space — that is lane intel (recorded), not a pillar connection.

⇒ **Gain.** No new katgpt-rs primitive, no feature flag, no benchmark owed here. The note exists to (a) discharge R366's family-check mandate, (b) route the lane intel where the lane lives, (c) prevent re-evaluation.

## 4. Track split (§3.5 discipline — one verdict per track)

**Track a — modelless extraction (all recorded, all consumer-gated):**

| Extractable | Shipment path | Consumer status |
|---|---|---|
| Shuffle test Δ (marginals-vs-joint decomposition) | a bench instrument over any noise-dependent drafter | GATED: requires a joint head; factorized drafters (z-lab DFlash2, bigram, dflash.rs) have Δ=0 by construction |
| τ(t) decoding-error reparameterization + closed-form P_e(t) | already in the R366 family's FMLM routing; would ride any future flow head | GATED: no flow head in-stack |
| Appendix-B noise-bank precompute (bank of 8,192 noise-only vocab projections; top-T correction) | a kernel-class trick for vocabulary-sized state ops (riir-infer shape) | GATED: pays only on a multi-jump flow head |
| "Integration creates joint structure" (flow @1 NFE ≡ factorized) | an interpretation of the shipped LoopMode::WeightShared FP block (Plan 108) | Recorded — validates the shipped FP-iteration design from the drafting side; no code change |

**Track c — model-based (the redirect):** a FlashMap-class joint head for OUR target (Qwen3.8-27B taps [5,19,33,47,61], keyed-sampled posture) → **riir-train**, recipe = flow teacher over one-hot blocks → Boffi semigroup flow-map distillation (already routed by R366's redirect row). Cost reality check: the paper's 8B teacher was ~25 h × 32 H200 + 9 h distill; a 27B-tap drafter on the 1×4090 is a multi-day project. **Owner-gated** — the lane currently consumes z-lab's free DFlash2 checkpoint and its open front is Phase-3 kernels, not head training. The "m5 job" marking per the owner's session instruction: this research unit is M3/M5-side (done); any joint-head training is a 4090 issue, and the lane's production target is M5 Ultra (Plan 614 Phase 3).

## 5. Lane intel → Plan 614 / Issue 1011 (the actionable core)

The lane's own measurements, with the metric each figure is: [riir-ai Plan 614](../riir-ai/.plans/614_dflash2_spec_lane_m5_ultra.md) Phase 3 (summarising Bench 965) — *"the binding constraint on this silicon is ACCEPTANCE (L_eff), not kernel work"* (L_eff is the LIVE loop metric; live L_eff measured 2.83 on doc-prose); Benches 964/973 — OFFLINE mean-accepted 0.18–0.42 (per-anchor A/B metric; Plan 614 records that the offline-to-live transfer FAILED — live mean_j on code measured 2.423 vs the 3.7 offline projection), far below the ~2.6–3.3 live-L_eff flip band. The lane's record names TWO causes: the acceptance shortage itself (965), and underneath it *"the DRAFTER'S distribution quality at these taps (the 746/747 target-quant feature drift)"* (964/973). FlashMap's controlled A/B ran on **dense BF16** Qwen3-8B with matched conditioning: it isolates the HEAD TYPE and is direct evidence that head class moves acceptance **in that setting** (+10% at 2 NFE, 18/18; +7.4% speedup at 1 NFE with half the params; flat conditional curve at 2 NFE). It says nothing about quantized-tap drift — and a +10% head-type effect cannot close a 0.18→2.6 gap. Honest reframe: **head class is a SECONDARY acceptance lever; the lane's named binding cause is tap feature drift.**

- **Transferable:** (1) the head-class option + recipe pointer for the owner-gated riir-train arm — **split A0/A1** (below) so causes stay attributable; (2) the shuffle test as the honest A/B metric for ANY future head-class claim (decompose Δ before attributing a win); (3) slot-resolved acceptance reporting — `correlation_budget.rs` already tracks per-depth EMA; extending the reporting is zero new machinery; (4) the appendix-B noise-bank pattern IF a 2-NFE flow head ever lands.
- **NOT transferable (with the measured negatives):** tree/draft-verify variants — this lane's tree-verify is closed-negative (Issue 1068: "the verify lane … tree-verify closed-negative — this is the draft-chain form only"), so Appendix D's +11% has no lane surface today; absolute speedups (H200 / dense BF16 Qwen3-8B vs 4090-now + M5-Ultra-production / PQ2-mixed 27B — the league absolute-numbers caveat); the EAGLE-3 comparison (our lane has no AR-drafter arm).
- **Honest tension to carry into any future A/B:** at 1 NFE a joint sample is NOT the per-slot argmax, so late-slot acceptance can LOSE to a factorized argmax drafter (FlashMap's own Fig. 3 decay) — the head-class comparison that favors joint is 2-NFE-vs-factorized, or explicit argmax-vs-joint at matched NFE. R407's ceiling and FlashMap's decay are the same axis read from both sides.
- **Recorded:** Issue 1011 watch note appended this session (the DSpark-watch-note pattern); no stance change — the drafted axis stays COMPETED; this widens the head-class option space, it moves no matrix cell.

## 6. Distill axis (the second skill on the code repo)

`al-jshen/flashmap` (+ DeepSpec): **corpus NO — Python-only** (torch/transformers + SGLang/vLLM plugins; zero Rust), so no clippy_lints / rust_perf / kernel_opt surface exists; not even MARGINAL (nothing to mine). **LEAGUE INTEL HIGH** — recorded per the distill skill's lane-intel rule: the dated line landed in `riir-refine/.distill/001_mining_queue_snapshot.md` this session, with the Issue-1011 filing recommendation carried out (watch note appended). Provenance: MIT code / Apache-2.0 weights, paper CC BY 4.0; repo NOT cloned (language mismatch — no `.raw/` entry owed).

## 7. What this note prevents

1. **Re-running the family gate** — R366 §5's instruction is discharged: any future flow-map-family paper checks here + R366 + R265 first.
2. **Mis-reading FlashMap as a katgpt-rs primitive** — no flow-drafter substrate ships; the deltas are instrument/config-level until a joint head exists.
3. **A future head-class A/B without the decomposition** — any acceptance win claimed for a new drafter head must report the shuffle-test Δ (marginals vs joint), or the attribution is unsupported.
4. **Double-recording** — the walk-16 C+ watch row (riir-refine Research 311) is superseded by this note + the Issue-1011 watch note.

## Action items

- [x] Research note (this file) — katgpt-rs `.research/634`, highwater 633→634, dual-allocation gate clean (0/0/0).
- [x] riir-ai Issue 1011 watch note appended (lane intel, DSpark-pattern).
- [x] riir-refine `.distill/001_mining_queue_snapshot.md` lane-intel line (distill skill recording rule).
- [-] **A0 — the drift arm (order first):** retrain the EXISTING factorized DFlash-class head ON OUR quantized taps — this addresses the lane's NAMED binding cause (the 746/747 target-quant feature drift), is the cheaper run, and isolates the drift fix. OWNER-GATED riir-train. NOT started this session (heavy = 4090-class, per the session's m5-job marking).
- [-] **A1 — the head-class arm (order only after A0):** the joint flow-map head (recipe + cost basis in §4). Worth ordering only once A0 exists, so the head-type increment is measured against a drift-free baseline — without the split, a future win cannot be attributed to either cause (the same attribution discipline as the shuffle-test Δ). OWNER-GATED riir-train. NOT started this session.
- [-] **Shuffle-test instrument** — becomes buildable only with a noise-dependent drafter; recorded as the mandatory A/B metric for that day (§7.3).
