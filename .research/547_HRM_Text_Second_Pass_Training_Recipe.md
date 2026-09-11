# Research 547: HRM-Text Second Pass — Training-Recipe Distill Against the Current Stack

> Source: [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) — local clone `.raw/HRM-Text/`, **pinned `aaa948ea674fd84b7bc455c9cfb455ecfefdf914`** ("Update HRM-Text banner"), Apache-2.0. Clone removed after distill; every quote re-verified at that sha.
> First pass: `048_HRM_Text_Hierarchical_Recurrent_Pretraining.md` (2025-07) → Plan 082b (4 techniques). This is the **second pass**: 12 additional techniques mined against the CURRENT stack state (2026-09-11), at owner direction ("082b got really good results for 1B — distill more against current repos, maybe more, e.g. fusion with riir-clippy Proposal 009").
> **Verdict: GAIN (two tracks).** Training track → `riir-train/.plans/394_hrm_text_trainer_recipe.md` (the recipe that makes the Plan-387-successor trainer honest BEFORE the arch re-run). Modelless track → `katgpt-rs/.issues/744_hrm_text_modelless_extractions.md` (8 extraction candidates, 3 ranked GOAT-worthy). NOT Super-GOAT: every mechanism is published prior art (Adam-atan2 = Everett et al. µP-scaling paper); the value is the recipe fusion, not mechanism novelty.
> **Audit row (drift found during this pass): Plan 082b T2 is FALSE in the current tree.** The `MultipackSampler` exists in NO repo — not katgpt-rs, not riir-train, not riir-ai, not even git history (`git log -S "MultipackSampler"` across all three: only the plan-doc commit `746e5346`). T1/T3/T4 DO exist, dead-coded, in `riir-train/crates/riir-train-gpu/src/optimizer.rs` (L998 `CpuAdamAtan2Step`, L1070 `BackpropWarmupConfig`, L1124 `trunc_normal_init`). The katgpt-deprecated `alien_sampler/sampler.rs` file named by Plan 082b now holds the unrelated Alien-Sampler distill (arXiv:2603.01092).

## TL;DR

Re-mining the HRM-Text tree after Plan 082b yields **12 techniques the first pass did not take**, and the stack has changed enough that the fusion targets are different ones:

1. The 1B-scale techniques 082b proved out (Adam-atan2, bp-warmup, learned init) are **still dead code** in `riir-train-gpu` (Issue 525 OPEN, never wired).
2. The would-be consumers moved: **Plan 387 CLOSED NEGATIVE** (looped pre-LN/conv-NCA class, 2.0%/0.0% exact-match, representational plateau — loss–exact-match decoupled ~60×/~660×); its named-next axes are batched-episode/on-device-loss training, arch review vs paper 449's per-cell compute, and the diffusion-curriculum axis.
3. **riir-clippy Plan 049 (trained drafter) is CLOSED-STOP** — its own stop rule fired (in-vocab band 0.94pp < 2.0pp). The Proposal-009 fusion asked about is therefore NOT a drafter revival; the honest fusion lands in the **trainer** lane, which is the blocked prerequisite for any future settle-refiner POC (Proposal 009 addendum next-action 2).

The second-pass haul, ranked for those targets:

| # | Technique | Where in HRM-Text | Track | Lands in |
|---|---|---|---|---|
| 1 | Sum-reduction + global valid-token divisor + in-loop per-sequence exact-match | `pretrain.py` L126-128, `lm_head.py` L52-70, `common.py` L16-18 | training | Plan 394 A1 (Issue 529 escalation) |
| 2 | Two-level H/L bp allocation + grad-free cycles | `hrm_nocarry_bp_warmup.py` L78-89 | training | Plan 394 A2 (upgrades dead `BackpropWarmupConfig`) |
| 3 | Adam-atan2 + EMA + `weights_only_resume_from_ema` | `adam_atan2.py` L95-108, `pretrain.py` L74/L262-266 | training | Plan 394 A3+B2 (extends Issue 525) |
| 4 | LPT episode packing (target_only masking, empty-sample prep filter, Philox epoch index) | `multipack_sampler.py`, `dataset_new.py` L107-127, `prepare_sft_data.py` | training | Plan 394 A4+B4 (fresh implementation — 082b T2 never landed) |
| 5 | Learned `zL_init` + init regime menu (lecun/megatron/scaled-embedding) | `hrm_nocarry_bp_warmup.py` L69, `transformer.py` L32-62 | training | Plan 394 B1 |
| 6 | SwiGLU + sigmoid-gated fused-gqkv block (Research 048's D6, never implemented) | `layers.py` L124-155, L158-168 | training (arch menu) | Plan 394 B3 |
| 7 | Condition tokens (direct/cot/noisy/synth — one model, many modes) | `prepare_sft_data.py` L50-54/L106-107 | both | Plan 394 C2; modelless routing → Issue 744 |
| 8 | `packing_sequence_sum` — one-cumsum segmented reduction | `common.py` L16-18 | modelless | Issue 744 #1 |
| 9 | atan2 positive homogeneity → normalize-free SNR gate + rank-invariance law | property of `atan2(m, v√)` | modelless | Issue 744 #2 |
| 10 | Seekable O(1) deterministic permutation (upgrade over HRM's materialized epoch files) | `prepare_sft_data.py` L165-173 | modelless | Issue 744 #3 |
| 11 | fmod truncated normal (±3σ branch-free, closed-form 1.014762601732121 = 1/√E[(Z mod 3)²]) | `common.py` L10-13 | modelless | Issue 744 #4 |
| 12 | Megatron depth law `out_std = in_std/√(2L)` → variance budget for deterministically-constructed LoRA overlays | `transformer.py` L57-58 | modelless | Issue 744 #5 |

Convention-grade (recorded, no issue): saturating cache-length counters (`simple_inference_engine.py` L180); empty-span boundary law ("hot kernels may assume len ≥ 1; degenerate spans rejected once at prep" — `prepare_sft_data.py` L99-103, crash provenance: packed batch of only-empty responses crashes FA3 backward).

## The 082b audit (what the first pass actually left behind)

| 082b task | Plan claim | Current tree |
|---|---|---|
| T1 Adam-atan2 WGSL + CPU ref | "all tests pass" | WGSL kernel in riir-ai `riir-gpu/src/kernels/adam_atan2.wgsl`; CPU ref `adam_atan2_step_cpu` in riir-train-gpu `optimizer.rs` L1016 — `#[allow(dead_code)]`, **never wired** (Issue 525 OPEN 2026-09-06) |
| T2 MultipackSampler | "10 tests pass, registered in lib.rs" | **GONE — zero code in any repo, zero git history.** Only the plan text remembers it |
| T3 BackpropWarmupConfig | "3 tests pass" | Exists L1070, linear ramp only — **the two-level H/L allocation was never ported** |
| T4 trunc_normal_init | "4 tests pass" | Exists L1124, Box-Muller ±2σ |

Lesson (extends the HISTORY.md drift ledger): a plan checkbox asserts the state at plan-write time; the `docs_gate`-class instrument for "does the code the checkbox names still exist" is `git log -S` + tree grep at consume time. The loser-sweep exile/deletion lifecycle is the likely deletion path for T2; the exact commit is unattributable because the code never entered history.

## Path 0 merge table (adversarial panel, both advocates)

Panel: No-GD advocate (tracks a+b) + Model-based advocate (track c), run in the same parallel batch as the §4 prior-art searches. Merge; discards carry mechanism-level reasons.

| HRM-Text item | Modelless verdict (No-GD advocate) | Training verdict (Model-based advocate) | Coordinator merge |
|---|---|---|---|
| Param-EMA + swap + resume-from-EMA | Core NOT modelless (weight-space). Extract: swap-as-evaluate-under-shadow (thaw-preview on `MerkleFrozenEnvelope`); EMA-over-runtime-state as closed-form IIR consolidation | R5b: EMA shadow 0.999 + eval-from-EMA at every checkpoint + `weights_only_resume_from_ema`; gate: eval exact-match(EMA) ≥ eval(raw) both halves; ~0 GPU-h | Split: R5b → Plan 394 B2; thaw-preview recorded in Issue 744 as freeze/thaw fusion note (no standalone primitive — `can_freeze` already convergence-gates) |
| H/L bp allocation + grad-free cycles | Core NOT modelless (BPTT bookkeeping). Extract: the two-timescale scheduler law (slow-first, reserve-one: `slow ≤ B−1 ∧ fast ≥ 1`) for deliberation-vs-perception tick budgets | R2: faithful port to the J=8 loop (backprop through bp_steps of J, rest forward-only); gate: ≥1.5× step throughput at equal loss-EMA vs v2 recipe; ~6 GPU-h | R2 → Plan 394 A2. The scheduler-law reframe recorded in Issue 744 #6 (riir-engine consumer, scenario-lab measurable) — one consumer minimum before it earns code |
| Sum-reduction + global divisor + scale invariance | Core NOT modelless (FSDP plumbing). Extracts: (a) atan2 homogeneity → normalize-free `sigmoid(α·atan2(dot, √energy))` SNR gate with rank-invariance (evidence-tier selection, `traj_store`/`elo`); (b) single-global-divisor aggregation law | R1: THE correctness precondition for Issue 529's batched-episode escalation — sum-formulation makes N-episode accumulation produce the identical update to averaging, decoupling LR from batch size (v1's divergence incident was exactly batch-1 constant-LR instability); gate: bit-identical BLAKE3 weights vs mean-formulation at batch-1; ~2 h | R1 → Plan 394 A1 (highest priority — unblocks the named-next axis). SNR gate → Issue 744 #2 (UQ caveat: ordering-statistic only, else conformal floor binds) |
| LPT packing + target_only + prep filters | packing_sequence_sum is pure modelless (segmented reduction, 3 consumers: AstChunker stream, zone-delta sums, consolidation stats). target_only core is training; masked-scoring semantics extract as oracle-contract | R7 + R10: LPT serves bonsai l4 LoRA corpora (heterogeneous clippy spans — where LPT pays, ~20-30% throughput); empty-sample/max-length/Philox-prep gates are ~0 cost hygiene. Boundary note: training substrate must land in riir-train, never katgpt-rs (082b T2's katgpt-deprecated placement was itself the smell) | R7+R10 → Plan 394 A4+B4 (fresh implementation). packing_sequence_sum → Issue 744 #1 |
| Condition tokens | ✅ Modelless — mode-as-latent-routing, zero param mutation; G1 = mode separation + no-condition-path bit-identity | R9: c-adjacent only; "No direct recipe item for axis (c)" — needs a capable base first | Issue 744 #7 (modelless routing vocab); Plan 394 C2 conditional |
| fmod truncated normal | ✅ Modelless — closed-form constant VERIFIED: E[(Z mod 3)²] ≈ 0.971117 → 1/√0.971117 = 1.014763 (matches to 5dp); caveat: folded-tail ≠ rejection-truncated (kurtosis shift ~0.27% mass) — init-grade, not statistics-grade | (no training content — riir-train-gpu already ships Box-Muller ±2σ as dead code; the closed-form constant is the only delta) | Issue 744 #4 (record the derivation; adopt only with a bulk-init consumer) |
| Init regime menu | ✅ Modelless — variance budgets for constructed operators; megatron `1/√(2L)` law unblocks the modelless mandate's deterministically-constructed-LoRA path (stability of constructed overlays through depth) | R4: learned zL_init as PRE-REGISTERED ablation for the axis-(b) arch review (HRM-Text and TRM both treat init state as learned; our fixed Box-Muller draw does not); ~4 GPU-h | R4 → Plan 394 B1. Menu → Issue 744 #5 |
| Sigmoid-gated fused-gqkv | ✅ Modelless-shape (gate source must be deterministic directions/reader-LoRA, not trained) — but the GEMV-fusion win is inference-perf, M3-Metal measurable | R8: block-menu item for the arch re-run (SwiGLU intermediate = expansion·hidden·2/3 rounded to 256 — parameter-neutral vs MLP-4×; gated attention output; RoPE kept fp32) | R8 → Plan 394 B3 (arch menu, rides the arch-review decision). Modelless HLA-gate variant recorded in Issue 744 #8, consumer minimum applies |
| Saturating cache counters + empty-span prep law | Convention-grade, property-test-shipped, sub-GOAT | R10's hygiene gates carry them | Recorded here only (no files) |
| Philox epoch index files | Upgraded by advocate: seekable O(1) permutation (counter-RNG/Feistel+cycle-walking) beats materializing files; determinism-doctrine fit | R10: data-order determinism extends the resume contract | Seekable permutation → Issue 744 #3; R10 → Plan 394 A4 |

## Fusion (what the user asked: HRM-Text × the current repos × Proposal 009)

**Primary fusion — the trainer recipe (riir-train Plan 394).** Plan 387's negative is representational, but its own verdict names the confound: the trainer was measured while (i) diverging at batch-1 constant-LR (v1), (ii) back-propping through full J every step (~4-orders compute-honesty gap vs EqR), (iii) computing exact-match in a separate 2.3-min eval pass, (iv) with a fixed not-learned carry init. HRM-Text is a recipe document for exactly this class — a weight-shared recurrent reasoner trained to 1B competence for ~$1000 — and its answers map 1:1: sum-accounting (R1), grad-free cycle budgeting (R2), bounded atan2 updates with a stress-arm gate (R3), learned init + init menu (R4), EMA eval (R5b), LPT packing (R7). The plan's honest ledger, inherited from the advocate: **none of this moves the 2% plateau** — it makes the axis-(b) arch re-run measurable on an honest trainer so a negative re-verdict isn't confounded the way v1-vs-v2 was.

**Arch data point for the axis-(b) review.** Research 529 (dissecting-HRM) measured vanilla-RNN@16 ≈ HRM ≫ one-pass — iteration with a *transformer-block* state update is the class that forms the constraint-satisfaction attractor; Plan 387's failed class was looped pre-LN/**conv-NCA**. HRM-Text supplies the concrete two-level form (H_cycles×L_cycles weight-shared, 16 physical → 96 effective layers) and the block recipe (pre-RMSNorm, fp32 RoPE, SwiGLU 2/3-expansion, gated gqkv, truncated-LeCun menu, learned zL_init). The arch review should run BOTH the one-level (TRM/paper-449) and two-level (HRM-Text) forms — our J=8 loop is an L_cycles analog with no H level.

**Secondary fusion — freeze/thaw posture (the 009 addendum's line, strengthened).** `weights_only_resume_from_ema` is the training-side twin of the house freeze/thaw rule: the EMA shadow IS the smoother freeze candidate; `swap_ema()` IS an atomic snapshot swap; "fine-tune from EMA with reset optimizer" IS starting a new lineage from the committed snapshot instead of mutating in place. This is an application-fusion note, not a new primitive (`MerkleFrozenEnvelope` already has the envelope; `can_freeze` already convergence-gates) — recorded in Issue 744, no standalone build.

**Healer fusion — honest negative.** Proposal 009's drafter lane is CLOSED-STOP by its own fired stop rule (Plan 049 + Bench 061: in-vocab band 0.94pp < 2.0pp, corpus-absent 99.06%); its reopen trigger is corpus-side (the fired-histogram roadmap — modelless proposer work), and the settle-refiner addendum is gated on a trainer capability gate that Plan 387 just failed. So this distill does NOT reopen either: it feeds the trainer those lanes would consume IF they reopen. The one live healer tail: modelless extraction #1 (`packing_sequence_sum` segmented reduction) has `AstChunker`-stream and oracle-metric consumers regardless of any training lane.

## Prior art (§4 — pinned before searching)

Claims pinned in-conversation before the panel/searches: (1) "HRM-Text second-pass recipe for the Plan-387-successor trainer, consuming riir-train-gpu's dead optimizer substrate, distinguished from Issue 525 (optimizer-only) by full-recipe + arch scope"; (2) "EMA-as-freeze-candidate + swap-as-atomic-swap application fusion".

- **Adam-atan2 is published prior art** — Everett et al., *Scaling Exponents Across Parameterizations and Optimizers* (µP-scaling line; "Adam-atan2, a new numerically stable, scale-invariant version of Adam that eliminates the epsilon hyperparameter entirely"); implementations: `lucidrains/adam-atan2-pytorch`, `pytorch-optimizers` (Adam-ATAN2 entry). Scale-invariance is the documented property (cf. MultiAdam, ICML 2023, parameter-wise scale-invariant optimizers). **No mechanism novelty is claimed anywhere in this note** — the distill's value is the recipe fusion and the stack wiring.
- Param-EMA + swap + eval-from-EMA is standard practice (timm/diffusion/SWA lineage). The freeze/thaw mapping is application-level.
- Condition tokens are prompt-conditioning, standard; the modelless routing reframe is the only local angle.
- In-stack coverage (grepped): no param-EMA anywhere (only `curvature_curriculum.rs` scalar metric EMA — a different thing); no condition tokens; no H/L allocation in `BackpropWarmupConfig`; no segmented-reduction primitive; no seekable permutation; Issue 525 already owns the Adam-atan2 wiring gap.

## Verdict scoring (§1.5)

- **Q1 prior art:** fails at mechanism level (all published) — only application fusion survives. Not Super-GOAT.
- **Q2 new behavior class:** no — recipe + utilities.
- **Q3 selling point:** n/a at this tier.
- **Q4 force multiplier:** the recipe connects to ≥2 lines (Plan-387 successor trainer, bonsai l4 LoRA corpora, future settle-refiner POC) — GOAT-tier connectivity, Gain-tier novelty.

**GAIN.** Files: `riir-train/.plans/394_hrm_text_trainer_recipe.md` (training track, OPEN — NOT SCHEDULED; scheduling owner-gated per the check-4090-before-mining rule), `katgpt-rs/.issues/744_hrm_text_modelless_extractions.md` (modelless queue). One verdict per track, no cascade between them (TTPO rule).

## References

- First pass: `048_HRM_Text_Hierarchical_Recurrent_Pretraining.md` → Plan 082b (T1/T3/T4 landed dead-coded in riir-train-gpu; T2 never landed — audit row above)
- HRM corpus line: R9 (HRM), R10 (TRM), R529 (dissecting-HRM — cadence + decodable≠causal), R48 (this source, first pass)
- Consumed-by: `riir-train/.plans/387_sudoku_maze_algo_reasoning_bench.md` (CLOSED NEGATIVE — named-next axes), `riir-train/.issues/525_adam_atan2_wire_optimizer.md` (OPEN — this pass prioritizes it and adds the stress-arm gate), riir-train Issue 529 (GPU backend + batched-episode escalation), `riir-clippy/.proposals/009_trained_clippy_drafter_moka_puct.md` (+ its 2026-09-05 settle-refiner addendum) and `riir-clippy/.plans/049` (CLOSED-STOP — unchanged by this note)
- External: Everett et al. *Scaling Exponents Across Parameterizations and Optimizers* (Adam-atan2); `lucidrains/adam-atan2-pytorch`; `pytorch-optimizers` Adam-ATAN2; MultiAdam (ICML 2023)
