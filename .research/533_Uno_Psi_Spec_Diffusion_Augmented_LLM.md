# Uno — Ψ-Spec Diffusion-Augmented LLM (TV-Loss Self-Drafter) Distill

**Status:** RECORD — dual-track verdicts: training track GOAT→ `riir-train/.plans/376_uno_tv_self_drafter_distillation.md`; modelless protocol track Gain→ folded into that plan's serving/consumption phases (+ katgpt-rs-side primitives flagged for the drafter trait).

- **Paper:** arXiv:2609.04010 "Unlocking Lossless Speedups in LLMs via Discrete Diffusion" (Sahoo et al., IFM-AI / MBZUAI / Cornell / Cerebras, 2026-09-03).
- **Code:** `github.com/ifm-ai/uno` @ `46fbdb66f026bae9c68a1e5a3f97a17c7805c778` (Apache-2.0), cloned to `.raw/uno` (ephemeral, removed at landing; sha is the provenance — every code anchor below was read at this pin).
- **Checkpoints:** HF `s-sahoo/uno-qwen3-8B` @ `b8a7577b3223bdcf2b3af0f2fc6e95258b3bbc29` (code-pinned default in `training/constants.py:3-4`), `IFM/K2-Horizon-7B-Uno`, `IFM/K2-Horizon-0.9B-Uno`.

## TL;DR

An AR LLM gains a second, parallel generation pathway: **per-matrix gated-LoRA adapters** trained by **one-step block-denoising distillation against the frozen model's own AR logits**, sampled **losslessly** through a draft-verify protocol (Ψ-Spec). Beats EAGLE-3 and DFlash across all batch sizes (2.5× at batch-1, 1.6× at max batch on Qwen3-8B); single KV cache; 0.35B extra params (vs DFlash 1.05B); adapters frozen through RL retain 94% of TPF. **For us:** this is a measured candidate for the league chat wall's own reopen trigger ("a new head family", riir-train Issue 493), the TV-loss recipe is a new riir-train capability, and the RL-freeze finding plugs directly into our GRPO plans.

## The mechanism (distilled)

1. **Two pathways, one architecture.** Frozen AR weights `θ_AR` (quality, verifier) + LoRA adapters `θ_Δ` on **all 7 projections** (q,k,v,o,gate,up,down) of every layer (speed, drafter). Paper Table 14: all-projections beats attention-only and O-only at parameter parity.
2. **Gated LoRA (Samragh et al. 2025).** One forward over `[last_clean_token, noise×(B−1)]`: adapter disabled at clean positions, enabled at noisy positions (`lora_mask[:,0]=0`). Single pass yields (a) clean-position logits from base weights only, (b) draft logits at noisy positions from base+adapter.
3. **Training = Diffusion Distillation, but the ablation collapses it.** Objective `α·L_DCD + β·L_TV`. Headline ablation (Table 12): **TV-only ≥ KL+TV at every weighting; 0.01×KL+TV marginally best** (KL is naturally ~10× larger). The shipped code default IS TV-only: `training/constants.py:23-25` → `DEFAULT_CE_ALPHA=0.0, DEFAULT_KL_BETA=0.0, DEFAULT_TV_GAMMA=1.0`. `L_TV` = blockwise total-variation |student−teacher| — the acceptance-length maximizer (Leviathan et al. Cor. 3.6: α_accept = 1 − TV). `_ChunkedTotalVariation` (training/losses.py) computes exact full-vocab L1 chunked, teacher detached, hand-written backward `g = p_s ⊙ (sign(p_s−p_t) − Σsign·p_s)` — never materializes dense fp32 probs.
4. **Ψ-Spec sampler (Algorithm 1).** Draft B−1 tokens i.i.d. from adapter logits; the **first token samples the base-weights seed row → distribution-identical to the verifier → always accepted** (the "free" token). Verify with one frozen-base forward; accept longest prefix by rejection test `r_i > min(1, p_i/q_i)`; on rejection sample replacement from `[p−q]+` (Gumbel-max in their fused kernel). **TPF bound: 1 ≤ TPF ≤ (B+1)/2** — the floor is 1.0, not classic spec-decode's 0.5: a *garbage drafter still costs nothing*.
5. **Paper Thm B.2 — the license.** At single-step inference the entire discrete-diffusion machinery (corruption schedule, Ψ predictor-corrector, consistency distillation) collapses exactly to "sample from the adapter logits". The adapter is a **substitutable proposal source**: rejection sampling against the verifier's own distribution is lossless for ANY drafter. The diffusion view matters only as the training objective/curriculum.
6. **Robustness findings.** (a) Adapters trained at the SFT checkpoint, frozen through full-param DAPO RL on 4 experts: TPF 2.25 → 2.10 (−6%), 40% end-to-end RL-training speedup (Table 8). (b) Adapters trained on OpenThoughts accelerate out-of-distribution despite mismatch, while AR fine-tuning on the same data *degrades* accuracy up to 15pts (Table C.4) — lossless verification doesn't care about adapter quality, only acceptance.

## Code-verified anchors (pin `46fbdb66`)

| Anchor | File:line | Value |
|---|---|---|
| TV-only is the default | `training/constants.py:23-25` | `ce=0.0, kl=0.0, tv=1.0` |
| Released checkpoint pin | `training/constants.py:3-4` | `s-sahoo/uno-qwen3-8B@b8a7577b…` |
| Global batch (code wins over prose) | `training/constants.py:17` | `128` (paper prose says 64) |
| Exact chunked TV + hand backward | `training/losses.py` `_ChunkedTotalVariation` | fp32 normals from bf16 logits, chunk 2048 |
| Sparse lm_head (teacher-logit memory fix) | `training/losses.py` `project_hidden_states` | lm_head runs only at supervised (noised) positions |
| LoRA targets | `training/lora.py` | preset `"all"` = 7 projections, fp32 A/B |
| Tree builder | `nano_vllm_uno/engine/draft_tree.py` | prefix-closed best-first max-heap by cumulative log-mass; deterministic 5-tuple `(-mass, depth, rank, token_id, parent)`; exact node budget V |
| Cycle accounting | `two_pass_decoding.py` | `forwards += 2` per cycle; committed ∈ [2, L+1]; bonus token on full acceptance |
| Fused verify | `fused_verify_kernel.py` | dual online softmax + Gumbel-max residual, one launch, zero host sync |

## Path 0 inventory (training-target decomposition)

| Component | Track | Coverage in our stack | Extraction verdict |
|---|---|---|---|
| Ψ-Spec draft-verify protocol | modelless | Partial: `forward_speculative_verify` + tree verify + spec_pool rollback ship (riir-ai 717/721/742) | **Extract**: free-seed always-accept token, TPF≥1 floor accounting, (B,K,V) config, verify-mode dispatch → plan consumption phases |
| Gated single-forward draft layout | modelless | Partial: `launch_qv_apply` applies adapters QV-only at decode | **Extract as plan task**: adapter-on-suffix-only draft pass; G0 measures QV-only vs all-proj acceptance |
| Log-mass best-first prefix tree (deterministic 5-tuple) | modelless | Partial: TreePath/tree-verify substrate exists; deterministic tie-break + node-budget shape is new | **Extract**: katgpt-rs drafter-trait/tree-construction candidate |
| Deterministic hash noise (splitmix64, no RNG state) | modelless | Partial: BLAKE3-deterministic embedders house-style; per-slot mixing form is new | **Extract**: trivial, fold into drafter protocol |
| Fused rejection+Gumbel verify kernel | modelless | Partial: verify kernels exist; Gumbel-residual-in-one-kernel shape is new | **Record** (kernel-family follow-up, rides Phase-3 serving) |
| **TV-loss acceptance-maximizing distillation** | model-based | **No analog** — riir-train has LoRA/GRPO/DPO losses, no TV-against-frozen-self drafter loss | **Plan** → `riir-train/.plans/376` (the core) |
| Block-denoising curriculum (B∈{2..16} progressive) | model-based | No analog | **Plan** (Phase-2/3 task) |
| One-step DCD distillation | model-based | No analog; but ablation says nearly dispensable (TV-only ≥) | **Discard as required** — auditable reason: Table 12 shows TV-only ≥ KL+TV; DCD is a convergence accelerant only, not needed for the recipe |
| 2L paired clean/noisy fused forward (flex-attention block-causal mask) | model-based | No analog | **Defer** — auditable reason: v1 two-forward pays the same 2× compute with standard causal masks; fused layout is a v2 optimization with zero semantic content (advocate finding, adopted) |
| RL-freeze drafter adapter | model-based (recipe) | No analog | **Plan** (Phase 4; applies to loss_grpo.rs plans + Plan 501-505 game-trajectory RL) |
| Inference-time scaling axis (T > B denoising steps) | either | No consumer | **Discard** — auditable reason: paper leaves it to future work with no quality numbers; no measured gain to chase |

## Dual-track verdicts (§1.5 one-verdict-per-track)

**Training track (model-based): GOAT.** Filed `riir-train/.plans/376_uno_tv_self_drafter_distillation.md`. Rationale: the recipe fills a measured capability gap (no acceptance-maximizing drafter loss anywhere in the stack), the league chat wall's own reopen trigger names "a new head family" (riir-ai league doc §qwen3.8 chat row; riir-train Issue 493 training lanes v1 2.385 / v2 1.949 / v3 2.233/2.076 all landed BELOW z-lab's 2.648 — under whatever loss those lanes used), and Uno publishes τ 3.89 (linear B4, temp 1) / 5.97 (tree 16,32,32) / 8.37 (tree-greedy V60, temp 0) on Qwen3-8B — straddling the wall's ≥4.0 mean-len requirement. GOAT gates inside the plan (G0 <5 GPU-h consuming their released checkpoint; Phase-2 2B validation ~20 GPU-h; Phase-3 league 27B ~160-330 GPU-h) compare trained-adapter vs our modelless NgramDrafter/lookup baseline on acceptance + tok/s, per Path 0.5's trained-vs-modelless rule.

**Modelless track (inference protocol): Gain.** The decision-layer items are real but none is a standalone Super-GOAT (all are consumption details of a drafter-verify seam we already ship). They ride the plan's serving phases: free-seed token + TPF floor accounting (G-gate items), (B,K,V) config space (league matrix rows), deterministic 5-tuple tree construction (katgpt-rs drafter-trait follow-up), hash-noise determinism (G1 gate support). No separate research-moat claim — the protocol is the paper's.

## Novelty gate + prior-art landscape (§4 searches done, 3-agent parallel round)

Pinned claim before searching: *"TV-loss-trained gated-LoRA self-drafter (draft = adapter-ON logits, verify = frozen adapter-OFF logits, always-accept first token) unblocking our Issue-742/league chat gap through the existing riir-gpu verify seam, trained via the riir-train LoRA pipeline."*

| Search axis | Finding |
|---|---|
| Headline class | Uno occupies the cell alone; no replications/extensions yet (days old). I-DLM R-ISD (Yu et al. 2604.11035) is the refuted competitor (their sampler is lossy per Uno's Suppl. C.5) |
| TV-loss axis — **decisive prior art** | **LK losses (Nebius, arXiv:2602.23881)**: pure TV **from scratch is unstable** (sign-only gradients, O(√k/V) norm at init); their usable forms are −log-acceptance and hybrid λ·KL+(1−λ)·TV — for **separate** draft models. EAGLE/Medusa/DFlash are all KL-family (DFlash: KL + fixed position decay; D-PACE 2605.18810 refines it). **Uno's TV-only works because the adapter warm-starts on a frozen base** — a regime LK explicitly does not study. This caveat is load-bearing for our plan: any TV training must be warm-start (zero-init adapter on frozen base), never from-scratch |
| Same-model acceptance-trained drafter | **DVI (arXiv:2510.05421)**: layer-split self-drafter, verifier decisions as supervision, KL→RL schedule, 2.16× ≈ EAGLE-2 — not adapter, not TV, doesn't beat EAGLE-3. Samragh 2507.11851: gated-LoRA MTP but **trained via SFT+consistency** (not acceptance loss, no lossless verify) |
| RL interaction | **ReSpec/DAS (MLSys 2026)**: "drafter staleness under continual actor updates" recognized for separate drafters (mitigation = KD-evolve during RL). Uno's adapter-freeze finding is the adapter-regime answer — no prior hit for adapters surviving their own base's RL |
| League landscape | DFlash2 head exists for OUR league model: `z-lab/Qwen3.8-27B-DFlash2` (5-layer block-diffusion drafter, up to 3.4× claims; our league row measures their loop at 3.53 tok/step vs our 2.648) |
| Surveys | No 2025-26 general spec-decode survey found (canonical is Xia 2024); self-drafting-via-adapter sits in no survey — taxonomy gap, consistent with the composition being recent |

**Empty cells confirmed (both flagged claims):** (a) TV-only adapter self-drafter against frozen self-logits; (b) adapter drafter surviving its own base's RL post-training. Our *consumption* of these is not a novelty claim — the paper owns them — but the **fusion** (league-model adapter + our verify seam + wall re-arithmetic) is ours, and it is a Gain-tier engineering moat, not a research-moat claim.

## Game-context reframe (mandatory step 4)

The serving path IS the game cognition path: riir-router/riir-gpu serves NPC cognition at 20Hz; a 1.6–2.5× lossless decode speedup is a direct cognition-rollout gain (more NPC tokens per tick budget, deeper deliberation L2 searches at equal latency). Second-order: riir-train's game-behavior RL (Plan 501-505 civ trajectory collection, GRPO plans) is rollout-generation-bound — the RL-freeze finding (drafter trained once at SFT, frozen through RL, 40% e2e RL speedup) applies verbatim to game-behavior training runs. Per-NPC behavioral signal: none new (lossless = the cognition distribution is unchanged — that's the *point*).

## Consumer-context reframe (mandatory step 4b — priority ladder)

- **#2 healer**: no hot local decode loop — CF llama is remote, L4 local-Bonsai is opt-in + unarmed, the healer's decode-adjacent surfaces (l4 fallback, rustc_errors playbooks) don't run a local draft-verify loop. Honest answer: **no healer surface**. (The one weak hook: `bonsai_clippy_l4_eval` decode runs would speed up if the league lane ever promotes — transitive only.)
- **#3 inference-perf league**: **the primary consumer.** The chat wall row (league doc) closed-negative with reopen = "a new head family or a target swap"; Uno is a new head family whose published τ straddles the wall's ≥4.0 mean-len bar. The plan's Phase-3 GOAT *is* a league cell.
- **#1 game runtime**: transitive via serving + RL rollouts (above).
- **#5 training**: hosts the recipe (the plan).

## Fusion (what paper × our stack produces that neither alone has)

1. **The wall re-arithmetic (new head family, honestly priced).** Wall 1 closed the DFlash2-shaped loop at a ~40 ms cycle floor (Q4_K weight pass). A self-drafter pays **two** full weight passes per cycle (~80 ms) — so ≥100 tok/s needs ≈8 tok/cycle, i.e. tree-greedy-class acceptance (published 8.37 at temp 0 V60 on 8B; unknown on our 27B). Linear mode (~2 full forwards, 3.9–4.9 tok/cycle) lands ≈ the banked standing 1.24× class — an improvement over 49.9 tok/s but not ≥100. **Thm-B.2's substitutability gives the escape hatch**: the draft pass may run on any cheaper proposal source (quantized sibling, distilled head) with TV training against the full model's logits — collapsing cycle cost back to ~40 ms while keeping the acceptance recipe. That hybrid is the genuinely new fusion candidate the plan's Phase-3 measures.
2. **TV warm-start vs Issue-493 lanes.** The 493 training lanes landed monotone-DOWN; whether they used CE/KL vs TV-against-frozen-self is the Phase-0 read — the LK instability result says the warm-start TV form is mechanistically different from whatever failed, and the plan treats "493 v1-v3 loss" as the first signal-diff to check before spending GPU.
3. **RL-freeze × GRPO.** Verify against base+policy-adapter (the true product), drafter frozen, refresh trigger at −10% τ via short TV resume — structurally easier than the paper's full-param RL case.

## Advocate round (merged; 3 parallel spawns)

- **No-GD advocate** returned 12 ranked modelless items (free-seed/TPF-floor; TPF accounting + ceiling normalization τ/((B+1)/2); deterministic 5-tuple log-mass tree; (B,K,V) config + K=1≡linear invariant; gated single-forward draft layout with KV-frontier invariant asserts; splitmix64 deterministic noise; fused rejection+Gumbel kernel; uncached-last-token KV rotation; depth-as-position tree attention + reverse-order FA3 cascade page table; sync-free in-place KV compaction; greedy/fused/sparse verify dispatch; linear-vs-tree regime routing) + 9 file-pinned engineering steals. Merged into the plan's serving/consumption tasks; the KV-rotation, page-table, and compaction items are recorded-not-filed (kernel-family follow-ups gated on Phase-3 promotion).
- **Model-based advocate** returned the recipe table + G0→B→C→D plan skeleton, the 24GB fit analysis (r=64 all-proj fp32-A/B + Muon momentum-only + Q2_0 base ≈ 21GB peak; the paper's r=128+AdamW does NOT fit), the sparse-lm_head + chunked-TV teacher-memory pattern, and the G0 partial-application question (our serve applies QV-only; all-proj-trained adapter partially applied = untrained approximation — G0 measures survival). Adopted into `.plans/376` with honest 4090 re-scaling.
- **Web advocate** returned the landscape above; the LK-losses warm-start caveat and the empty claim-cells are its load-bearing yields.

Discard ledger: DCD-as-required (Table 12), fused 2L layout for v1 (same compute, no semantics), inference-time-T scaling (no numbers), katgpt-rs standalone primitives for protocol items (consumption details; the tree 5-tuple is the one flagged for the drafter trait if Phase-3 promotes).

## GOAT / validation protocol

Owned by `.plans/376`: G0 acceptance harness on the released checkpoint (lossless bit-identity + τ vs NgramDrafter/lookup + QV-only A/B), Phase-2 2B gates (τ_chat ≥ 2× modelless, bit-identical greedy stream, chunked-TV exactness, grad-check), Phase-3 league gates (chat τ ≥ 2× modelless at matched verify economics, e2e ≥ 1.3× decode, wall re-arithmetic table, GPU-exclusive per Bench 649 rule), promote/demote per axis (lookup keeps doc-repro; adapter takes chat/OOD if gates pass).

## Priority routing + files

- `riir-train/.plans/376_uno_tv_self_drafter_distillation.md` — the plan (training recipe + consumption phases + league gates).
- Serving integration executes in riir-ai (spec seam); file the riir-ai `.issues/` entry at Phase-3 start per boundary.
- katgpt-rs: no new crate surface now; the deterministic 5-tuple tree construction is a drafter-trait follow-up gated on Phase-3.
