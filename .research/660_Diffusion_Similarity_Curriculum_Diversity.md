# Research 660: Diffusion-Similarity Curriculum Diversity (Self-Play Theorem Proving Theory)

> **Source:** "A Theoretical Framework for Self-Play Theorem Proving Algorithms" — Thomas Chen, Zhiyuan Li, arXiv:2606.01861, June 2026
> **Date:** 2026-11-09
> **Status:** Active
> **Related Research:** 365 (PhysiFormer heat kernel — DEC eigenbasis precedent), 323 (TEMP diversity fingerprint), 564 (R4T set admission)
> **Related Plans:** 634 (diffusion_diversity primitive), 274 (CGSP triad), 599 (set_admission), 359 (DEC heat kernel), 341 (TEMP)
> **Cross-ref (riir-ai / riir-chain / riir-neuron-db):** riir-ai Research 407 (guide), riir-ai Plan 646 (consumer), reflex Issue 115, refine Issue 222, riir-train Issue 680
> **Classification:** Public

---

## TL;DR

Chen & Li give the first theory of prover/conjecturer self-play: theorems form a similarity graph; a conjecturer is a random walk on it; knowledge grows exponentially under a connectivity (isoperimetric) condition; and the empirically observed failure — conjecturers drift toward artificially complex, cluster-local items (STP's elegance filters, Minimo's longer-not-deeper) — is diagnosed as a DIVERSITY failure with a closed-form measure `Div_P^τ(μ) = 1/‖μᵀP^τ‖²_{ℓ2(1/π)}` (effective spread of a distribution under τ-step graph diffusion, stationary-measure weighted) plus a walk-time conjecturing algorithm (Algorithm 4) that locally maximizes it via a tiny neighbor QP over diffusion similarities.

**Distilled for katgpt-rs (modelless, inference-time):** four closed-form primitives, no training anywhere: (1) **diffusion similarity** `φ_τ(x,y) = ⟨p_τ(x), p_τ(y)⟩_{ℓ2(1/π)}` with `p_τ(x) = (P^τ)ᵀ e_x`, computable two ways — exact local BFS dynamic program over the (τ+1)-neighborhood (paper Alg. 5; Lemma B.13: local subgraph suffices exactly) or spectral truncation `Σ_k λ_k^{2τ} ψ_k(x)ψ_k(y)/√(π(x)π(y))` over eigenvectors ψ of the normalized adjacency, selected top-k by |λ| (reversibility identity, Lemma B.4; computed with katgpt-core's own `karc::jacobi_eigen`, NOT a DEC import — the Plan-634 boundary keeps complex-geometry operators in katgpt-dec); (2) **the diversity scalar** `Div_P^τ(μ)` — one diffuse-matvec per cycle, a frontier-growth predictor (Lemma 5.3 lower-bounds the τ-expanded provable set); (3) **the walk-time diversity-maximizing selector** — at each hop, solve a small PSD QP `argmin μᵀM′μ` over the neighbor simplex (floor η/deg for irreducibility), which upweights inter-cluster transitions and provably enlarges the spectral gap by Θ(M/r²) on clique-cluster graphs (Prop 5.6) — a PRIMITIVE whose escape guarantee binds only to walk-shaped consumers (none shipped today); (4) the **φ_τ batch selector** — the consumer-facing form, built on set_admission's log-det machinery (made const-generic `const D: usize`, default 8 so existing consumers compile unchanged; the Vendi certificate keeps the pinned spectral_pencil Jacobi) fed spectral diffusion coordinates `e_τ(x)_k = λ_k^τψ_k(x)/√π(x)` over top-D ≤ 64 eigenpairs by |λ| — `⟨e_τ(x),e_τ(y)⟩` = truncated φ_τ, the linear-kernel boundary intact.

---

## 1. Paper Core Findings

1. **Theorem graph**: nodes = theorems; edge (x,y) iff every prover in the class has |succ(x,f) − succ(y,f)| ≤ ε. Continuity of success rate across edges (Assumption 3) + neighbor oracle (Assumption 4) are the only graph accesses needed.
2. **Exponential growth** (Thm 4.2): conjecturer = reversible nearest-neighbor random walk; if the walk satisfies the strong isoperimetric inequality with constant κ, the proved-set size grows exponentially in iterations. Proof machinery: `∇*∇f = −(P−I)f` (their Lemma A.10 — a DEC statement), Dirichlet forms, ‖P‖ ≤ 1 − 1/(2κ²) (Cheeger-type). Training-error budget ϵ < 1/(2κ²).
3. **Diversity measure** (Def 5.2 + Lemma 5.3): `Div_P^τ(μ)` lower-bounds `|K^τ_p(f)|`, the τ-step expansion of the provable set — diversity of the curriculum is a *predictor of frontier growth*, not just spread.
4. **Improved conjecturer** (Alg. 4): per-hop neighbor distribution `C(x) = argmin_{μ∈Δ(N(x)), μ≥η/deg} μᵀ P^τ diag(1/π)(P^τ)ᵀ μ` — an MMR/DPP-shaped QP whose kernel is the DIFFUSION similarity, not raw adjacency. Prop 5.6: on clique-cluster graphs the induced meta-walk has Θ(M/r²)-larger spectral gap — cluster escape the simple random walk cannot do (it lingers ~M steps per cluster).
5. **Contrastive identity** (Thm 5.7, extending HaoChen et al. 2022): global minimizers of the spectral contrastive loss with augmentation P^τ ARE the truncated spectral diffusion embeddings — so the diffusion similarity is computable WITHOUT any training: `Σ_k λ_k^{2τ} φ_k(x)φ_k(y)`.
6. **τ-selection**: no closed form in the paper; the right τ is the "cluster-forgetting" scale (same-cluster p_τ ≈ equal, cross-cluster distinct). Offline sweep per graph suffices.

## 2. Distillation

### 2.1 What transfers (modelless)

| Paper component | Closed form? | Shipped analog (signal-diff) | Delta that survives |
|---|---|---|---|
| Diffusion similarity φ_τ | YES (local DP or spectral sum) | DEC `heat_kernel.rs` (exp(tA) on CellComplex — field trajectories, not pair similarity; no 1/π weighting, no general similarity graph) | pair-wise similarity operator on general weighted graphs; stationary-measure weighting; two exact postures with a built-in cross-check |
| Div_P^τ(μ) | YES (one diffuse-matvec) | `set_admission` Vendi certificate (set-level kernel entropy, fixed cosine kernel, d=8 ceiling, no graph/diffusion/distribution-push); `ColinearityBatchGate` (raw 1-hop cosine, binary verdict) | distribution-level effective spread under multi-hop graph structure; sees cluster-locality pairwise checks cannot; continuous, not binary |
| Alg. 4 walk-time QP | YES (tiny deterministic QP, deg(x) vars) | `PoolConjecturer` (priority roulette — no diversity term); MMR/DPP class (subset/one-shot, raw kernel); WJRW walk sampling (correction, not per-hop QP); one-shot heat-kernel coreset (arXiv:2606.22002 — selection, not walk-time) | per-hop neighbor distribution choice; diffusion kernel; proven gap-enlargement |
| Isoperimetric κ → gap bound | YES (ordering law) | DEC operators own the algebra (∇, ∇*, Δ); no shipped bound | fixture-testable ordering law; cheapest GOAT |
| Contrastive identity (Thm 5.7) | YES (it IS the no-training proof) | spectral_pencil jacobi / DEC eigenbasis | justification that the trained-embedding lane is redundant for us |
| Thm 4.2 growth theory | inspiration only | — | no shipped consumer trains a prover; transfers as design validation |

### 2.2 Path 0 inventory (per-track endpoints)

| # | Component | Endpoint |
|---|---|---|
| 1 | Diffusion similarity (two postures) | (a) **filed** — Plan 634 (open primitive, `diffusion_diversity` feature, katgpt-core `diversity/` home) |
| 2 | Div_P^τ scalar + frontier-predictor framing | (a) **filed** — Plan 634 |
| 3 | Alg. 4 walk-time selector QP | (a) **filed** — Plan 634 |
| 4 | κ → spectral-gap ordering law + Prop 5.6 gap-ratio | (a) **filed** — Plan 634 fixture rows (cheapest GOAT proof) |
| 5 | Contrastive-embedding TRAINING lane | (c) **audited discard** — Thm 5.7 proves the trained embedding equals the spectral embedding; our eigen machinery computes it offline; training buys nothing at any margin we need (mechanism-level reason, not class-level) |
| 6 | CGSP consumer wiring (Div emission + diffusion-kernel BATCH selection; no walk claim) | (a) **filed** — riir-ai Plan 646 (modelless runtime lane; ladder priority: self-evolve/game runtime) |
| 7 | Synth-corpus Div arm (reflex corpus-synthesis echo/collapse surface) | (b) **issue** — reflex Issue 115 (report-only arm; signal-diff vs echo gates recorded there) |
| 8 | Rulewright batch-intake diversity arm | (b) **issue** — refine Issue 222 (signal-diff vs dedup gates recorded there) |
| 9 | Training recipes: conjecturer-gated GRPO curriculum; quest-LoRA curriculum intake | (b) **issue** — riir-train Issue 680 (fusion ideas pending primitive GOAT; GRPO plans verified to exist: `.plans/328`, `448`, `090`) |
| 10 | Thm 4.2 exponential-growth story as a runtime claim | (c) **audited discard** — requires infinite theorem space + prover retraining; on a finite frozen pool the win is bounded by pool coverage; transfers as motivation for the ordering-law fixtures only |

### 2.3 Fusion (§1 protocol)

The fusion is paper × CGSP × set_admission. The CGSP triad (Plan 274) has a conjecturer — `PoolConjecturer`, a priority-weighted GLOBAL roulette over the whole pool (the `CuriosityConjecturer` contract takes `target/priorities/out`; there is no current node and no neighbor set, so it is NOT a walk) — a difficulty filter (`BreakevenDifficultyFilter`, per-candidate solve-rate estimates = the paper's `succ(·,f)` already estimated at runtime), and post-hoc batch/collapse detectors (`ColinearityBatchGate`, `EntropyCollapse`, MCTS δmg). The paper's contribution slots in as: (i) the Div scalar — a continuous curriculum-diversity readout none of the detectors provide; (ii) diffusion-kernel BATCH selection for the conjecture batch (set_admission-shaped, kernel swapped to φ_τ); (iii) the walk-time selector as a PRIMITIVE whose proven cluster-escape guarantee awaits a genuinely walk-shaped consumer (KG neighbor walks, quest-graph expansion — each needs its own consumer issue; none is wired today). Nobody in the stack currently computes multi-hop diffusion similarity, gates a curriculum batch by a diffusion kernel, or links a diversity scalar to frontier-growth prediction.

### 2.4 Novelty gate (§1.5) — GOAT, per track A (modelless primitive + CGSP/corpus fusion)

**Pinned claim:** *diffusion-kernel batch selection and a distribution-level diversity scalar — ⟨p_τ,p_τ⟩_{ℓ2(1/π)} over a solve-rate-similarity graph, Div_P^τ(μ) as a frontier-growth predictor and batch gate — applied modellessly to the CGSP conjecture loop and synthetic-corpus curation; plus the walk-time neighbor-QP selector as a generic graph primitive whose Prop-5.6 escape guarantee binds only when a walk-shaped consumer exists.*

- **Q1 no prior art:** mechanism-level YES for the composition AS PINNED — diffusion-kernel batch selection + the Div frontier-growth scalar, applied modellessly to conjecture/corpus loops (the walk-time selector is a primitive-only claim). Published class coverage exists and is cited (diffusion maps/distance; MMR; DPP; diverse beam search; one-shot heat-kernel coreset arXiv:2606.22002; WJRW walk sampling; STP dedup; cluster-diversity metrics arXiv:2410.15226) — none ships the composition. Internal grep: zero hits for the arXiv ID/title across all `.research/`.
- **Q2 new behavior class:** **NO.** Selection-time diversity prevention already ships — `set_admission` (Plan 599) does greedy log-det marginal-gain ADMISSION plus a Vendi certificate; batch gates ship in CGSP. The deltas are kernel/weighting/scalar-level (multi-hop φ_τ vs fixed cosine; 1/π weighting; Div's growth link), not a new capability class. This is the honest downgrade from the round-1 draft.
- **Q3 selling point:** "Our curriculum loops gain a distribution-level diversity scalar linked to frontier growth — predicting curriculum collapse before it costs a cycle — and graph-walk surfaces gain a proven cluster-escape selector."
- **Q4 force multiplier:** CGSP (katgpt-core) + DEC/spectral eigen machinery + riir-ai cgsp_runtime + corpus-synthesis lanes ≥ 2 pillars.

Q2 fails ⇒ **Super-GOAT fails; verdict GOAT** — provable gain (kernel + scalar + ordering-law fixtures) over the shipped set_admission/batch-gate approach, promoted per the stack feature-flag discipline.

Track B (training recipes): **Gain** — filed as riir-train Issue 680, promotion trigger stated there. Track C (embedding training): discard per Path-0 row 5.

### 2.5 Weakest points (named)

(a) **The CGSP conjecturer is a global roulette, not a walk** (verdict round 1, verified at `cgsp/traits.rs` + `cgsp/conjecturer.rs`): a uniform roulette jumps anywhere in one draw, so it has no cluster-lingering problem for Alg. 4 to fix — the Prop-5.6 escape guarantee has NO shipped consumer today. The selector therefore ships as a primitive; its CGSP wiring is the batch-level diffusion-kernel selector (no Prop-5.6 claim), and walk-shaped consumers (KG neighbor walks, quest-graph expansion) are named open follow-ups, never silently assumed.
(b) **set_admission is selection-time prevention** (greedy log-det admission), not post-hoc detection — the round-1 draft's "detect-after vs prevent-during" contrast was wrong for the nearest cousin; the surviving deltas are the multi-hop kernel, the 1/π weighting, and the Div scalar.
(c) **Regime-dependence** — on small well-mixed pools the selector ≈ any baseline; Plan 634 F4 pins the tie-or-demote tripwire against BOTH baselines (SRW and global uniform jump), and all growth-story claims are scoped away (finite pool ⇒ ordering laws only).

### 2.6 PoC / validation protocol (defend-wrong §3.6)

Quality-parity is NOT claimed — no head-to-head vs the paper's training setup is needed (architectural + closed-form claims only). The falsifiable gates live in Plan 634: F1 gap-ratio ordering (Prop 5.6 shape, planted clique-graph), F2 local-DP vs spectral cross-check (two independent derivations, pinned ε), F3 Div-vs-admission Spearman on CGSP fixtures (report-only first), F4 well-mixed tie arm (demote tripwire), G3 feature-off byte-identity.

---

## 3. Verdict

| Track | Verdict | Routing |
|---|---|---|
| A — modelless primitive + CGSP/corpus fusion | **GOAT** (Q2 fails — set_admission already does selection-time prevention; deltas are kernel/scalar-level. Proof path = Plan 634 fixtures + riir-ai Plan 646 A/B) | Open primitive → katgpt-core `diversity/diffusion.rs` behind `diffusion_diversity`; guide → riir-ai Research 407; consumer → riir-ai Plan 646 (Div emission + diffusion-kernel batch selection; walk selector awaits a walk-shaped consumer); corpus consumers → reflex Issue 115 + refine Issue 222 |
| B — training recipes | **Gain** | riir-train Issue 680 (fusion ideas; Path 0.5 deferral with promotion trigger — no runnable proof path until the primitive GOAT passes and a GRPO/quest-LoRA consumer is chosen) |
| C — contrastive embedding training | **Discard** (audited, Path-0 row 5) | none — closed-form substitution is lossless |

**MOAT gate (katgpt-rs):** base graph-diffusion primitive via fusion, DEC-family adjacent (heat/wave kernel precedent, Plan 359) — in scope; no game/chain/shard IP in the primitive. GOAT feature flag + benchmark + promote/demote per stack discipline. Demoted-loser rule: if F4 (tie-vs-both-baselines) fails, the selector stays opt-in and the Div scalar may still promote report-only.

## 4. Routing summary

- `katgpt-rs/.research/660` (this note) · `katgpt-rs/.plans/634_diffusion_diversity_primitive.md`
- `riir-ai/.research/407_diffusion_diversity_curriculum_guide.md` · `riir-ai/.plans/646_cgsp_diffusion_diversity_consumer.md`
- `riir-reflex/.issues/115_synth_corpus_diversity_arm.md` · `riir-refine/.issues/222_batch_intake_diversity_arm.md` · `riir-train/.issues/680_curriculum_diversity_training_recipes.md`

## 5. PASS-Redirects

None (verdict is not PASS). Cousins updated instead: none needed — every cited cousin (Plan 274 cgsp, Plan 599 set_admission, Plan 341 TEMP, Plan 359 heat_kernel) is referenced above by name+number so future greps land here.
