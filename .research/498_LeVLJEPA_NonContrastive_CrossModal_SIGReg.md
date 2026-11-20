# Research 498: LeVLJEPA — Non-Contrastive Cross-Modal Alignment + Sketched Gaussianity (SIGReg)

> **Source:** [LeVLJEPA: End-to-End Vision-Language Pretraining Without Negatives](https://arxiv.org/abs/2607.00784) — Kuhn, Serra, Balestriero, Buettner, 2026-07
> **Date:** 2026-08-21
> **Related Research:** 138 (LeJEPA — direct ancestor, same Balestriero/LeCun lineage), 115 (PEIRA closed-form inter-view predictor), 394 (GNN within-class erank), 475 (ICA non-Gaussian directions), 200 (quantization outlier collapse → Plan 224 KS substrate)
> **Related Plans:** 224 (OAQG ks_d_statistic), 252 (riir-train LoRA Outlier Guard — training-time KS twin), 568 (RRQ consumes KS as scalar)
> **PASS-Redirects (synthesis):** Kuhn, Maes, Serra, Le Lidec, LeCun, Balestriero, Buettner [arXiv:2608.27395 "LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics"] — third LeJEPA-family member (after 138 LeJEPA theory + this note's SIGReg→probe distillation): first video encoder trained with invariance+SIGReg only (no target encoder/predictor/stop-grad/masked reconstruction), matching V-JEPA 2 at 5.6–20.8× less compute. Its deltas vs this note are all video-PRETRAINING recipe facts with no in-stack consumer (video lane explicitly out of scope, riir-train Issue 403 non-goals): 95% uniform random token dropping improves IN1k monotonically (33.9→47.6%) while structured tube dropping HURTS when the objective imputes nothing (the sparsity pattern must match the objective: structured serves imputation, uniform random serves pure observation — validates our fog-of-war write-gate design + attention-mass-scored KV eviction, no contradicting default); block-causal across-frame attention matches bidirectional at frozen probing (51.2 vs 50.7 IN1k) — validates our causal-by-construction belief kernels (sense/evolve_belief, GenericSpatialBelief decay); per-frame τ=1 tokenization beats τ=2 temporal aggregation at matched token budget; dense patch-token structure emerges from CLS-only supervision (strengthens this note's pooled-vs-dense fusion row). Motion-vs-appearance sparsity asymmetry (SSv2 degrades beyond ρ=0.3 while IN1k improves) is the temporal-cadence axis for the frame-sampling bandit (riir-ai 124) — validation, no gap. PASS: no file, no plan, no issue.
> **Status:** DISTILLED — pending owner decision on riir-ai #743 (Phase 1 erank guard); katgpt-rs #681 (sketched gaussianity probe) COMPLETE 2026-08-22 — GOAT ALL PASS, opt-in `gaussianity_probe` (Bench 673); the SIGReg training-loss A/B (743 Phase 2) remains gated on Phase 1 evidence
> **Domain:** katgpt-rs (open primitive: sketched gaussianity probe) + riir-ai (edge_lora hidden-space guard)

---

## TL;DR

LeVLJEPA is the first fully non-contrastive end-to-end vision-language pretraining method: cross-modal prediction with stop-gradient targets + per-modality **SIGReg** (Sketched Isotropic Gaussian Regularization — project embeddings onto random 1D directions, run a characteristic-function normality test per projection, O(B·d·|A|), batch-size invariant). No negatives, temperature, momentum encoder, or teacher-student. Headline empirical finding: non-contrastive pretraining yields markedly stronger **dense per-token features** (segmentation, frozen VLM backbone) while contrastive wins only pooled zero-shot alignment — on these encoders, zero-shot accuracy was **inversely related** to backbone quality.

**Verdict: GAIN (not Super-GOAT).** The training recipe itself is out of scope (no VL model, no image corpus — 256 H100-hours on 92M pairs). The Path-0 decomposition surfaced two actionable, modelless-validable deltas, both grounded in shipped substrate: (1) a **sketched multi-direction gaussianity probe** for embedding populations — the one distribution-health axis no shipped diagnostic covers (all shipped metrics are second-moment; bimodal/heavy-tailed/discrete full-rank populations pass `effective_rank` as healthy); (2) an **edge_lora hidden-space distribution guard** — the cross-game translation + sleep-consolidation surfaces train/cluster unregularized hidden-state populations with no erank/gaussianity check wired anywhere. The paper's symmetric-collapse ablation does NOT map to edge_lora (coordinator correction, see Path 0 table) — targets there are recorded episodes, stop-grad by construction.

---

## Key Mechanisms

| # | Mechanism | What it is | Training-only? |
|---|-----------|------------|----------------|
| 1 | Cross-modal prediction + stop-gradient | Predictor MLPs h_v/h_t predict the OTHER modality's detached embedding; MSE to sg targets. Asymmetry is load-bearing (Table 7) | Yes (gradients) — closed-form analog ships as PEIRA |
| 2 | SIGReg | Random 1D projections + Epps-Pulley CF normality test per projection → isotropic-Gaussian marginal pressure. Linear in B and d; no batch coupling | The **test half is modelless**; the loss half is training |
| 3 | Batch-size invariance | No batch-level term → performance flat across B ∈ {1024, 2048, 4096}; InfoNCE improves with B | Property; our sigmoid-not-softmax house rule guarantees it structurally |
| 4 | Rank ≠ usefulness | Symmetric MSE + SIGReg: eff. rank 170, ImageNet ZS 1.82%. Full objective: rank 477, ZS 25.24%. High rank certifies nothing about transfer | Diagnostic caution for every erank consumer |
| 5 | Pooled vs dense readouts | Zero-shot/linear-probe (pooled CLS) don't measure per-token feature quality; SigLIP = best zero-shot, worst backbone; LeVLJEPA inverse | Evaluation-protocol insight |

Ablation insight worth keeping verbatim: **symmetric cross-modal regression collapses even WITH per-modality distributional regularization** — marginal regularization alone cannot overcome symmetric-coupling degeneracy; the predictor + stop-gradient asymmetry is what lets each side's distribution be shaped independently.

---

## Path 0 Decomposition (component → coverage → extraction)

| Component | Coverage (analog ships?) | Extraction (modelless-computable?) | Verdict |
|---|---|---|---|
| Cross-modal predictor | **YES** — `katgpt-core/src/peira.rs` closed-form inter-view ridge predictor (`(N+λI)⁻¹` + `linalg::ridge_solve`); cross-view → cross-modal is data plumbing. Also `distill_attention.rs` frozen teacher (sg by construction) | Residual `‖Y − XW*‖²` as alignment-drift diagnostic between frozen embedding families | Covered; fusion opportunity only |
| SIGReg distributional reg | **PARTIAL** — univariate KS-vs-Gaussian ships 3× (`katgpt-spectral/spectral.rs:837` + `outlier_guard.rs` load-time + `rrq_quant.rs` scalar consumer + riir-train Plan 252 training-time twin). Second-moment metrics ship (`effective_rank`, `within_class_effective_rank`, cosine, `spectral_flatness`). **Multi-direction sketched probe on embedding populations: NOT shipped** | **YES** — the test half is pure closed-form statistics: fixed random directions (BLAKE3-seeded Rademacher, the `spectral.rs` L820 generation pattern) + per-direction 1D normality + aggregate. Cramér-Wold licenses the sketch | **Issue katgpt-rs #681** |
| Stop-gradient asymmetry | **YES (training side)** — CISPO `detach(ratio)` (GOAT-proved, "1473× more stable than PPO-clip"), `loss_asft` sg weights, `collimation_lora` detached logprobs, `sdpg` sg-advantage, `gemma2_d2f_sc` x̂₀ self-conditioning — 5× proven in-stack | Modelless analog = one-way-gate audit rule (reader never writes target store — the freeze/thaw reader-invariant shape) | Covered; validated by the paper, no action |
| Batch invariance | **Structural** (house sigmoid rule) but no mechanical audit gate | Audit kit: partition B ∈ {1, 7, 64, full}, assert per-item scores bit-identical; negative control = softmax scorer must fail | Fusion opportunity (note only) |
| Rank ≠ usefulness | **PARTIAL** — `gold_share.rs` documents "erank is content-agnostic"; `hope_bridge.rs` rejects intrinsic_dim→γ; but `can_freeze` (rank-sufficient + flat ⇒ commit permanently) has no utility probe; `geometry.rs` test `g3_random_init_high_effective_rank` itself shows random noise = max erank | Planted-neighborhood retrieval recall floor on frozen artifacts (`ShardIndex::retrieve_diverse` exists) | Fusion opportunity (note only) |
| Pooled vs dense | **PARTIAL** — `geometry.rs` docs "aggregate symptom vs mechanism locator"; KG `compute_quality_metrics` gates on pooled `avg_confidence` while consumers read per-triple | Per-triple quantile floors (p10 confidence alongside mean) | Fusion opportunity (note only) |

### Model-based advocate findings (merged + corrected)

The panel's model-based arm found the training surfaces; coordinator verdicts per §3.5 (discards need auditable reasons):

1. **edge_lora cross-game anti-collapse** — advocate claimed "EXACTLY the paper's ablation-#4 collapse shape". **CORRECTED (discard of the collapse claim):** `CrossGameEpisode.target_output` is recorded episode data — targets are stop-gradient **by construction** within `TopologyTrainingLoop` (`training.rs` L326-335, `g = 2·reward·diff` against fixed dataset targets). The paper's symmetric co-training collapse does not apply. **What survives:** the hidden-state populations on BOTH sides are unregularized AND unmonitored (no erank/gaussianity check anywhere in the loop; `sleep_consolidation.rs` clusters these same states — collapse silently degrades consolidation). → **Issue riir-ai #743** (guard-first; optional SIGReg-as-aux-loss A/B <2 GPU-h M3).
2. **DualEncoderIndexer BCE→continuous-mass regression** — real but bench-only surface (opt-in `trained_indexer`; the bench's own header concedes the thresholded labels are low-signal). Keep as fusion opportunity in this note; not filed (no production consumer).
3. **SIGReg aux loss on edge_lora hidden states** — folded into #743 phase 2.
4. **gemma2_directions re-open with trained head** (<1 GPU-h, decisive either way) — re-opens the Bench 571 measured negative; owner call, not filed.
5. Stop-grad asymmetry 5× in-stack — coverage, no action.

### No-GD advocate findings (merged)

- The gaussianity probe's signal-diff vs `effective_rank` (verified against `geometry.rs` source): erank = entropy of covariance eigenvalues (second moment); a **bimodal mixture** `½N(−μe,σI)+½N(+μe,σI)` has covariance σ²I+μ²eeᵀ → near-full rank, "healthy"; projection onto e is two-point-separated → any normality test rejects. Same for heavy-tail (5% @ 10σ — outliers *inflate* eigenvalues) and discrete/quantized marginals. This is the exact "shard population that is two disjoint styles glued together" failure a consolidation pipeline wants to catch before freezing.
- **Shipped-but-unguarded assumption:** `katgpt-band/src/band_conditioner.rs` L31-33 — Fisher-z "requires approximate Gaussianity of residuals" — no runtime check. The probe turns the assumption into a checkable precondition.
- `band_conditioner`'s "sigmoid-bounded p-value for downstream routing" is the output-shape precedent for the probe score (house sigmoid rule).

---

## Novelty gate (honest — NOT Super-GOAT)

1. **Prior art?** In-stack: pieces heavily covered (PEIRA, ks_d_statistic ×3 surfaces, CISPO, erank family). Published: the SIGReg machinery is the Balestriero/LeCun lineage (LeJEPA, LeWorldModel arXiv:2603.19312 uses the same projections+normality trick in training; VL-JEPA is contrastive-framed). No found prior art on **inference-time multi-direction gaussianity probes for embedding-population health** in production systems — but this is new-to-stack application, not world-novel math. → NO (strict).
2. **New behavior class?** No — better diagnostics/stability, not a new capability. → NO.
3. **Product selling point?** No. → NO.
4. **Force multiplier?** Moderate — connects `data_probe` family, freeze gate, edge_lora, band_conditioner. → Partial.

Gain. Two issues, no Super-GOAT guide.

---

## Distillation

### Ships now (coverage — no action)

- Closed-form inter-view predictor: `peira.rs` (+ `linalg::ridge_solve`)
- Univariate KS-vs-Gaussian: `ks_d_statistic` × (load-time | quant-router | LoRA-training-time)
- Second-moment population health: `effective_rank` / `within_class_effective_rank` / `avg_cosine_similarity`
- Stop-gradient asymmetry: CISPO et al. (5 surfaces)
- Batch invariance: house sigmoid rule (structural)

### New (filed)

| Issue | Repo | What | GOAT sketch |
|---|---|---|---|
| #681 | katgpt-rs | `data_probe/gaussianity.rs` — sketched multi-direction projection-normality probe, feature `gaussianity_probe` | G1: isotropic fixture passes (bit-identical ×3); bimodal/heavy-tail/discrete fixtures reject while `effective_rank` on the SAME fixtures passes (non-redundancy pin, the `p415_g2` pattern); G2: latency vs erank (no O(d³) eigensolve — should win at audit cadence); G3 default-untouched; G4 zero-alloc; cross-crate agreement vs `katgpt_spectral::ks_d_statistic` on 1D projections (leaf constraint: core cannot dep on spectral — the rrq_quant scalar-inversion note; the agreement test lives in katgpt-spectral, which CAN see core) |
| #743 | riir-ai | edge_lora hidden-space distribution guard — wire `within_class_erank` (+ #681 probe when landed) as training-time advisory + `sleep_consolidation` precondition; optional phase-2 SIGReg aux-loss A/B (<2 GPU-h M3, λ ∈ [0.005, 0.04] — the paper's stable band) | G1: erank floor held across long runs (planted-collapse fixture must trip the guard); G2 ≤5% step-time overhead; G3 arena mixed-episode win-rate no-regression (Plan 298 G3 ≥5pp must hold); G4 guard scratch-owned |

### Fusion opportunities (recorded, not filed)

- **DualEncoderIndexer**: replace BCE-on-thresholded-labels with per-sample MSE regression onto continuous attention mass (sg target = data). The bench's own header concedes the labels are low-signal; the paper's core trade (regression to targets instead of discriminative negatives) maps 1:1. Bench-only; revisit if `trained_indexer` promotes.
- **gemma2_directions re-open** (Bench 571): depth-2 trained head over frozen Gemma-2 embeddings, targets = SharedContextLog success/failure directions (supervision exists, passed G1-G5). The paper's finding #5 claims regression objectives improve frozen-backbone downstream features — the exact failure mode of 571. <1 GPU-h, decisive either way. Owner call (re-opens a measured negative).
- **Batch-invariance audit kit**: `data_probe/batch_invariance.rs` — partition-invariance bit-identity harness with a softmax negative control. Converts the house rule from prose to a pinned gate for riir-ai scoring paths. Cheap; file when a consumer asks.
- **KG per-triple quantile floor**: `kg/mod.rs compute_quality_metrics` gates on pooled `avg_confidence`; add `p10_confidence ≥ floor`. The pooled-vs-dense insight at minimal cost.
- **Freeze-gate utility floor**: planted-neighborhood retrieval recall on frozen shards — the "rank-healthy-but-useless" probe (paper Table 7 caution mapped to `can_freeze`).
- **ICA lens fusion** (Research 475): the probe samples random directions; FastICA finds the MOST non-Gaussian directions deterministically. A cheap hybrid: probe random directions for the aggregate score, escalate to ICA directions on borderline cases.

### Validations of house choices (no action)

- SigLIP's sigmoid decoupling + LeVLJEPA's batch-invariance result = published empirical support for the sigmoid-not-softmax rule (InfoNCE's cross-item normalization is precisely the batch-level coupling term).
- Note 138's theory (linear identifiability ⟺ Gaussian latents) gets its experimental shadow here: the objective that pressures Gaussian marginals produced the best dense features for LINEAR consumption (frozen bridges, linear heads). Our latent-to-latent ops are linear (dot + sigmoid) — Gaussian marginals are the regime where they're optimal.

---

## Advocate-correction record (§3.5 discipline)

- Model-based finding #1's "EXACTLY ablation-#4 collapse shape" — **discarded** after coordinator read of `cross_game_edge.rs` L59-69 + `training.rs` L326-335: targets are recorded episodes (fixed data), not co-trained encoder outputs. Symmetric-collapse requires both sides gradient-coupled. Surviving half (distribution guard) filed as #743 with honest framing.

> **PASS-Redirects (synthesis):** Ren, Wang, Hoang, Kuang, LeCun [AGL-TR-2026-04 "The Content–Geometry Tradeoff in World Models: When JEPA Learns Latent Maps"] — the floor-only sibling of SIGReg (one-sided variance hinge along random directions, trained WITHOUT stop-gradient). Confirms this note's distilled-diagnostic reading: the floor's diagnostic content is covered by shape-per-random-direction (this probe) + per-coordinate activity floors (#687) — a std≥1 floor without the training target is scale-arbitrary. Recipe → riir-train Plan 507. Verdict: **PASS** (modelless track). See [Research 686](686_Content_Geometry_Tradeoff_JEPA_Latent_Maps.md).

## References

- Kuhn, Serra, Balestriero, Buettner. "LeVLJEPA: End-to-End Vision-Language Pretraining Without Negatives." arXiv:2607.00784, 2026.
- Balestriero, LeCun. "LeJEPA: Provable and scalable self-supervised learning without the heuristics." arXiv:2511.08544, 2025. (Research 138)
- Maes et al. "LeWorldModel" arXiv:2603.19312 — same SIGReg machinery in world-model training.
- Chen et al. "VL-JEPA" arXiv:2512.10942 — JEPA-framed but contrastive signal (batch negatives).
