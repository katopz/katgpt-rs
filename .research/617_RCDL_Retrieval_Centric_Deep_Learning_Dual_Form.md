# Research 617: RCDL — Retrieval-Centric Deep Learning in Growing Nonparametric Neural Networks

> **Source:** [Retrieval-Centric Deep Learning in Growing Nonparametric Neural Networks](https://arxiv.org/abs/2610.03858) — Schlegel, Nasser, Kobayashi, Schimpf, Sieberling, Obryk, Irie, Sacramento, von Oswald (Google Paradigms of Intelligence + MIT CSAIL + Yale), 2 Oct 2026. Public code: `github.com/paradigms-of-intelligence/rcdl`.
> **Date:** 2026-10-11
> **Status:** Done — verdict **Gain** (this record + riir-train Issue 628 + cross-ref in 024); verdict ping-pong PASSED (round 1 AGREE, claude_code reviewer).
> **Related Research:** 024 (δ-Mem — the primal cousin; carries the NL-DGD redirect this paper's §4 independently confirms), 070 (GDN2 — the linear-attention family), 028 (HLA), 615 (CMM Sync EWMA)
> **Related Plans:** —
> **Cross-ref (riir-train):** Issue 628 (delta-rule-decay + mesa-FOOF optimizer probe; `acba5ba8`)
> **Classification:** Public

---

## TL;DR

The paper formalizes "retrieval-centric deep learning" (RCDL): every linear layer trained by gradient descent is *exactly* a growing key-value memory retrieved by unnormalized linear attention (the Irie-2022 dual form), and upgrading the retrieval kernel — RBF or a new "weighted softmax" — yields growing nonparametric networks whose entire learning rule is "append `(key=input, value=prediction-error)` per sample". For this stack the paper is a **unification + prior-art fence, not a primitive source**: the core growing-error-memory mechanism is **KLMS (2008)** — predated, consumable, not claimable; every consumable element maps onto shipped substrate (`delta_mem` IS the fixed-rank primal of the dual form; the surprise gate ≈ novelty-gated growth; engram privilege gating ≈ query-conditional utility; laprop's bias correction IS the timestamp-momentum coefficient). The one actionable output: §4 derives the delta-rule operator as a *training-time* optimizer/regularizer that beats weight decay (74% vs 67%), the **second independent convergence** with NL-DGD after Research 024's unfiled redirect → **riir-train Issue 628**.

**Distilled for katgpt-rs (modelless, inference-time):**
The dual-form law — *a plain linear-kernel KV memory appended with (input, error) pairs is exact SGD on the implicit weight matrix* (Irie 2022); the delta-rule/gated variant we ship as `delta_mem` is the DeltaNet member of the same family, whose dual-form reading is §4's optimizer connection (rank-one forgetting, below) — plus three memory-management laws: (1) weight decay ≡ uniform `(1−γ)` decay of all stored values ⇒ principled ε-pruning bound; (2) momentum ≡ a per-tuple creation-timestamp coefficient `c(t,tᵢ)=(1−μ^(t−tᵢ+1))/(1−μ)` computed at read time, zero extra storage; (3) sublinear growth via a novelty threshold `min_j‖x−k_j‖>τ` on appends, else parametric refinement of stored tuples.

---

## 1. Paper Core Findings

1. **Dual form (background, Irie et al. 2022):** GD on a linear layer unrolls to `W_T = W_0 + Σ v_t x_tᵀ` with `v_t = −η∇_{y_t}L`, so inference is `y = W_0 q + Σ v_t (x_t·q)` — unnormalized linear attention over every training point. Store-the-data ≡ store-the-weights, exactly.
2. **Growing kernel layers:** replacing the linear kernel φ by a positive-definite kernel (RBF) makes SGD on the implicit RKHS weight matrix *exactly* equal to appending `(k=x_t, v=−η∇L)` per sample; retrieval `f(q)=Σ φ(k,q)·v`. Non-linearizable kernels cannot be compressed to fixed size — the growing database IS the model.
3. **Weighted softmax (the paper's own novelty):** standard softmax breaks smooth appends (any new key discontinuously alters the denominator). Fix: per-entry weights `(w,k,v)` triples; the continuous softmax over a Gaussian-mixture weight function is exactly the discrete weighted softmax (variance terms cancel). Functional gradient via Fréchet derivative in `L²(μ)` yields the append rule `k_new=Σ_k q_t`, `w_new ∝ η⟨s_t,g_t⟩`, `ṽ_new ∝ −ηΣ_v g_t`. Negative weights: clip, or **gauge invariance** (translation symmetry values↔bias; gauge fixing guarantees `w_new>0`; the update stays LOCAL — old tuples unaltered).
4. **Memory laws:** weight decay = decay all stored `(w, ṽ)` by `(1−γ)` ⇒ geometric value decay ⇒ prune below ε with bounded perturbation; momentum = store creation timestamp, scale by `c(t,tᵢ)` at retrieval (no velocity storage); hybrid growth = append only when `min_j‖x−k_j‖₂ > τ`, otherwise Adam-refine stored tuples (sublinear capacity at matched accuracy).
5. **Advanced LA → optimizers (§4):** plugging the mesa-layer (cumulative regularized least squares) into the dual form = the **FOOF** optimizer (Benzing 2022; KFAC simplification; input-whitening ≈ approximate second-order). DeltaNet in the dual form = GD "preceded by targeted forgetting along the current activation direction": `W_t = W_{t−1}(I−λx_t x_tᵀ) + (−λη∇L)x_tᵀ`. As a regularizer on a 10-layer MLP (2000 epochs, CIFAR-10): **74% vs 67%** for multiplicative weight decay; the normalized per-sample write-strength variant *degrades* (activation-norm-coupled LR instability).
6. **Experiments:** growing GWS/RBF (plain-SGD learning rules) beat SGD/Adam/RMSProp MLPs and are competitive with Muon on CIFAR-10/100; single-pass (online) regime: growing methods beat ALL parametric arms incl. Muon/Mesa; ViT FFN→RCDL drop-in: 48.4%→54.4% (1 epoch). RBF ≈ weighted softmax throughout (58.8 vs 59.4 single-epoch, within ~1.5σ). Cost: the growing DB is huge (61.5 GB for 50-epoch CIFAR; 4 TPU shards).

## 2. Distillation — substrate map (signal-diff per row)

| Paper element | Closest shipped substrate | Signal-diff (what each consumes) |
|---|---|---|
| Growing (k=x, v=−η∇L) memory + kernel retrieval | `katgpt-core::delta_mem` (fixed-rank delta-rule state) + `PkmEpisodicStore` (growing-slot gated writes) + `engram` | delta_mem consumes prediction error (delta rule) but **compresses into a rank-r state** — for the PLAIN linear kernel the dual form proves growing-KV ≡ GD weights; delta_mem ships the gated delta-rule (DeltaNet) variant, one step beyond plain GD (its dual reading is §4's optimizer, not the plain equivalence). PKM/engram store observations with admission gates, not error values. The exact growing error-valued form = **KLMS (2008), predated**. |
| Novelty-gated growth (`min‖x−k‖>τ`) | delta_mem surprise gate (`θ_surprise`, suppresses writes when temporal-derivative surprise < θ) + PKM `write_idf` | Ours gate on surprise/IDF, paper's on key distance (Platt RAN 1991 gates on error) — same admission-law family, different novelty signal; no gap. |
| Query-conditional utility at read | engram **counterfactual privilege gating** (Issue 656, shipped `9c88e1cc`, Bench 656) | Shipped — the LOPD δ. |
| Timestamp momentum `c(t,tᵢ)` | `katgpt-core/laprop.rs` `NormalizedMomentumAccumulator` (μ^t maintained incrementally, O(1) bias correction) | Same truncated-geometric coefficient, optimizer-side vs memory-read-side; Adam bias correction / exponential histograms / event-driven SNN decay — textbook. |
| Weight-decay ⇒ value decay ⇒ ε-prune | none as a value-decay law (PKM is fixed-slot overwrite; ndb consolidation is policy-tier) | Component predated (RAN resource pruning, NEC recency eviction, FadeMem arXiv:2601.18642); adoptable law; **no documented consumer gap** (no growing unbounded store on a hot path). |
| DeltaNet decay as optimizer `W(I−λxxᵀ)` | delta_mem ships the operator as RUNTIME memory update | As a TRAINING optimizer: NL-DGD (arXiv:2512.24695 §4.5) prior art + this paper's dual-form derivation + 74v67 benchmark → **riir-train Issue 628**. |
| mesa-layer ≡ FOOF | — (Muon-orbit preconditioner family) | Bridge between two pre-existing artifacts (FOOF arXiv:2201.12250; MesaNet arXiv:2506.05233); community already reads mesa as exact RLS (arXiv:2607.19390). Consumable recipe, not claimable. |
| Weighted-softmax Fréchet rule (paper's own novelty) | — | **Not consumed**: softmax-normalized retrieval readout is off house style (sigmoid-never-softmax); the paper's own RBF arm matches it within noise, validating unnormalized per-key gates. Reopen trigger: a serving surface needing normalized retrieval competition with principled appends. |

**The unification lens (the real take-home):** the dual form says our memory family is one thing — a plain growing (input, error) KV store and a GD-trained linear layer are two views of the same learner; `delta_mem` (fixed-rank delta rule) is the DeltaNet step beyond plain GD (add rank-one forgetting); engram/PKM/lattice are retrieval structures sitting between. Any linear-kernel KV memory appended with error values can be read as SGD on the implicit weight matrix. Useful as a bench-interpretation lens (e.g. Bench 408's PKM-vs-δMem comparison is retrieval-structure-vs-compressed-state at matched task), not as new code.

## 3. Verdict

**Per-track (never pooled):**

- **Track (a) modelless: COVERED / PREDATED — no new primitive, no feature flag.** Q1 novelty: **NO** — terminal prior art KLMS (Liu, Príncipe & Haykin, IEEE TSP 2008: append `(x, prediction-error)`, readout `Σ η e_i k(x,x_i)`, unnormalized per-key kernel gates, exact-SGD-in-RKHS theorem — every element of the would-be claim), plus component-level anticipation (RAN 1991, Kadirkamanathan 1993, Pokharel 2009, Dragutinović arXiv:2510.10425, Collins NeurIPS 2024). Q2–Q4 moot. The paper *validates* shipped design (unnormalized RBF-class gates ≈ weighted softmax; error-driven delta-rule memories are principled) — validation is not actionable.
- **Track (c) model-based: GAIN → riir-train Issue 628.** §4/F.8.1 is a cheap, concrete, twice-converged optimizer probe (NL-DGD + RCDL) priced and queued, not scheduled.
- **Paper-level: GAIN** — this record + Issue 628 + the cross-ref line in 024. The note exists because the prior-art fence is load-bearing: without it a future session re-distills this paper as a "novel growing memory primitive".

**MOAT gate:** katgpt-rs = fundamental-principle tier — this is a design-law + prior-art record, no code, moat-neutral (defensive). riir-train = active moat — the recipe probe fits its charter (optimizer research; LaProp ships here).

## 4. Prior-art record (the §4 searches, so nobody re-runs them)

- **KLMS** — Liu, Príncipe, Haykin, *IEEE Trans. Signal Processing* 56(2), 2008. TERMINAL prior art for the growing error-valued kernelized memory. Growth sparsification: Pokharel et al. 2009 ("KLMS with constrained growth"); Engel et al. KRLS/ALD 2004.
- Platt, *A Resource-Allocating Network for Function Interpolation*, Neural Computation 1991 — error-gated growth of RBF units + resource pruning. Kadirkamanathan & Niranjan 1993 — function-space view.
- Irie, Csordás, Schmidhuber, arXiv:2202.05798 (ICML 2022) — the linear-kernel dual form. von Oswald et al. ICML 2023 — transformers learn in-context by GD.
- Dragutinović, arXiv:2510.10425 (Oct 2025) — softmax attention as RBF-kernel gradient descent. Collins et al., NeurIPS 2024 — softmax attention ≡ functional GD in RKHS.
- Grubb & Bagnell, ICML 2010 — functional-gradient construction of deep networks (admitted by the paper as ref [29]; covers the PDK rule, not growing retrieval memories).
- Nested Learning, arXiv:2512.24695 §4.5 "DGD" — the `(I−η'xxᵀ)` decay as training optimizer (already redirected in Research 024). FADE, arXiv:2604.27063 — adaptive weight-decay-as-forgetting framing (nearest to §4's claim). FadeMem, arXiv:2601.18642 — decay-gated pruning of agent memory. KLA, arXiv:2605.08587 — delta rule as Kaczmarz projection.
- FOOF — Benzing, arXiv:2201.12250 (ICML 2022); treated as standard by Eva (arXiv:2308.02123). MesaNet — arXiv:2506.05233 (ICLR 2026); independently read as exact RLS by arXiv:2607.19390.
- Adam bias correction (arXiv:1412.6980), exponential histograms (Datar–Gionis–Indyk–Motwani 2002), event-driven SNN state decay — the timestamp-coefficient trick in three literatures.

## 5. Fusion (recorded, novelty TBD — no issue without a consumer)

- **Dual-form audit lens on Bench 408/919:** PKM-vs-δMem-vs-lattice comparisons are primal/dual/retrieval-structure views of one learner; a future bench could pin the equivalence empirically at the linear kernel (append-error KV vs rank-r delta state, matched stream). Novelty TBD; no consumer today.
- **Decay-prune law for a growing store:** if a hot path ever grows an unbounded error-valued store (e.g. a self-correcting corpus), the `(1−γ)`-decay + ε-prune bound is the law to reach for — one line of prior art (this note §1.4) when that day comes.
- **"Silent memory" resonance:** the paper's discussion (forgetting = retrieval failure, content preserved — Tonegawa engram line) is exactly the think-brain belief-decay design (`sigmoid(−λ·Δt)`, beliefs fade not deleted). Validation, not action.
- **Weighted-softmax rule:** recorded above; reopen trigger stated in the substrate map.

## References

- Paper: arXiv:2610.03858 (+ repo `paradigms-of-intelligence/rcdl`).
- Internal: Research 024 (δ-Mem, `katgpt-core::delta_mem`), 070 (GDN2), 526 (FWA continual learning — NLMS-normalized online ridge read of the same family), Issue 656 bench (engram privilege gating), `katgpt-core/laprop.rs`, Bench 408 (PKM episodic fusion), Bench 919 (lattice GOAT).
- External: see §4.
