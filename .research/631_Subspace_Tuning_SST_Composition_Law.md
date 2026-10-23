# Research 631: Subspace Tuning (SST) — the Final-Delta Composition Law

**Status:** Active — verdict filed; Issue 958 carries the modelless substrate, riir-train Plan 470 the training recipe, sibling issues carry the consumer-side lanes

> **Source:** "Towards Continual Learning: Mitigating Forgetting With Subspace Tuning" — Chavan, Sareen, Tiwari, Chandra, Obando-Ceron, Keutzer, Niekum, Dhillon, Khatri (UC Berkeley / Mila / McGill / UdeM / UMass / Google / UT Austin), arXiv:2610.22293, submitted 2026-10-15
> **Date:** 2026-10-23
> **Related Research:** 406 (SAR — spectral weight-delta purification; closest internal cousin), 231 (SOPTV — off-principal task vectors), 501 (SVCCA), 557 (subspace intervention), 238 (LoRA-Muon spectral manifold), 163 (EoS curvature allocation — training-track cousin)
> **Related Plans:** 470 (riir-train — SST continual refit recipe), 025 (LoraPair hot-swap — factored-tail overlay consumer), 094 (TIES merge — anatomy-report consumer)
> **Cross-ref:** riir-neuron-db Issue 629 (freeze anchor sidecar), riir-ai Issue 1116 (merge-admission gate + hot-swap wiring)
> **Classification:** Public (mechanism + math); per-model protection bases are private fuel

---

## TL;DR

SST (Subspace Tuning) fights catastrophic forgetting by projecting every **applied** optimizer update through the two-sided complement of the base weight's top-k singular subspace: `Π(ΔW) = (I − U_k U_kᵀ)·ΔW·(I − V_k V_kᵀ)`, protecting r = 0.99 of singular directions and adapting densely in the ~1% tail, with a Gram-based basis refresh every K steps as an explicit adaptation–retention control. Its load-bearing *finding* (not just recipe): the projection must hit the **final applied update** — Adam's diagonal preconditioner does not commute with Π, so gradient-space projection leaks 69% of the final change back into protected directions (paper Fig. 4 controlled study). Retention: −1 pp prior capability vs −25..−57 pp for plain SFT at matched in-domain accuracy; 2–4× less KL drift; sequential fine-tune chains become cumulative; independently trained SST experts merge with near-zero interference (−0.2..−7.6 pp vs SFT's −9.8..−23.2 pp).

**Distilled for katgpt-rs (modelless, inference-time):** the mechanism is a **projection operator plus a composition law**, not a training technique. Two modelless tools fall out, with DIFFERENT consumers:

1. **Delta anatomy** (works on ALL existing deltas): decompose any weight delta into protected-block vs free-tail energy using the base matrix's singular basis. The paper's own controlled study says an unconstrained delta carries ~98% of its energy in the top-99% protected span — so the anatomy report is an interference predictor for merges and a drift quantifier for freeze/thaw deltas. It is NOT a compliance check for existing deltas — they would fail one, by construction.
2. **Tail certification + factored-tail shipping** (applies ONLY to deltas produced under tail confinement — Plan 470 outputs, or overlays constructed directly in the tail): a delta confined to the tail factors as `ΔW = U_⊥ C V_⊥ᵀ`, and the tail span is CLOSED under that product — so quantize `C` (or the factors) and the decoded delta is still exactly tail-confined, certified by construction. **Dense-project-then-quantize is unsalvageable**: quantization rounding is roughly isotropic, so a fraction `1−(1−r)² ≈ 99.99%` of it lands in the protected block at magnitude comparable to the delta itself. The certifier that remains is a RELATIVE-tolerance check against the expected quantization-noise floor, never a near-zero absolute one.

The composition-order law survives both: certify/factor the FINAL applied delta, never intermediate scaled steps — any elementwise transform between accumulation and projection leaks (training: Adam's D, 69%; shipping: quantizer scales ≈ the same diagonal shape; rounding: the isotropic case above).

---

## 1. Paper Core Findings

1. **The operator.** `Π(ΔW) = Q_U ΔW Q_V` with `Q_U = I − U_k U_kᵀ`, `Q_V = I − V_k V_kᵀ`, `(U_k, V_k)` = top-k singular vectors of the reference weights `W_ref`, `k = ⌊r·min(d_out, d_in)⌋`, r = 0.99. Two-sided: blocks (1)–(3) of the protected/free decomposition must ALL be discarded; keeping only block (4) is exactly Π. Protected singular triplets `(σ_i, u_i, v_i), i ≤ k` are preserved exactly.
2. **Enforcement point is load-bearing (§2.2 controlled study, Qwen2.5-0.5B, 60 Adam steps, r=0.99).** Unconstrained: **98% of the weight change enters the protected (top-99%) directions**, held-out NLL +13.0. Project the *gradient* (pre-Adam): 69% still leaks — Adam's diagonal `D` rescales coordinates and `DΠ ≠ ΠD`. Re-project the *cumulative* change: 0% leak but NLL +3.28 (mis-scaled step from projected-gradient moments). **Project the *applied update* (SST): 0% leak, NLL +0.13.** The optimizer runs untouched; only its produced step is replaced by `W_t + Π(W̃_{t+1} − W_t)`.
3. **Retention at matched learning.** Qwen3-8B across Science/FinQA/Pathology/Medicine: plain SFT loses 25–57 pp of the 5-benchmark prior-capability average; SST variants stay within ~1–4 pp (no-refresh strictest) at equal or better in-domain accuracy. Scales with model size (8B+: ~1 pp forgetting; 14B/32B better). At OUR trainable scales the effect is present but weaker (Qwen3-1.7B: SST −6.9..−8.6 pp vs SFT −18..−31 pp; 4B: −1..−3.6 pp vs −24..−36 pp). Works under SFT, SDFT (self-distillation), and Muon — optimizer-agnostic.
4. **KL drift 2–4× lower** than unconstrained recipes (WikiText-2 KL(base‖θ)); preserves qualitative behavior accuracy misses (a thinking model's response length; SDFT's fabricated-reference rate drops).
5. **Sequential chains become cumulative** (A→B, A→B→C): each domain holds its trained level under later stages; plain SFT loses 9–28 pp on dormant domains.
6. **Experts merge cleanly** (Appx B.1): three independently trained SST experts, naive task-arithmetic merge `W_base + ΣΔ_d`, degrade −0.2..−7.6 pp vs their own experts; SFT experts −9.8..−23.2 pp. Exact algebra: each purified delta has identically zero protected-block content, so merges cannot collide there *by construction*; residual interference is free-block overlap. ⚠ Scope note: these are TAIL-constrained deltas (keep the bottom block) — a different regime from Research 406's Fusion A, which keeps the TOP block; no confirmation relation (see §7).
7. **Rank ablation (FinQA):** retention improves monotonically with protected fraction r (23.6 → 82.3 prior-capability across r ∈ {0(SFT), 0.1, 0.5, 0.9, 0.99}); in-domain accuracy rises too until r = 0.99 where one instruct-model case needed r = 0.90 (the ~1% tail too small to fit the task).
8. **Refresh (R100/R10):** re-anchor `W_ref ← W_{t+1}`, recompute `(U_k, V_k)` every K steps via Gram eigendecomposition (smaller Gram side). Opens rotated directions for continued adaptation at a mild retention cost — K is an explicit adaptation–retention control.
9. **Negative row:** reverse-KL penalty toward the base model does NOT reproduce the trade-off (weak λ forgets anyway; strong λ suppresses learning without restoring capability). The constraint must be a hard, closed-form projection or it is nothing.
10. **Cost:** projection +1.5%/step (252 matrices, 8×L40S, bf16 Qwen3-8B); refresh R100 +5.1%, R10 +37.6% amortized (0.6%/4.7% when the SVD is split across 8 GPUs). Zero inference-time cost.

## 2. Distillation

### 2.1 §3.5 Modelless Unblock Protocol — Path 0 decomposition (mandatory)

| # | Component | Class | Modelless extraction | Disposition |
|---|-----------|-------|----------------------|-------------|
| 1 | Tail projector Π. At r=0.99 neither basis is uniformly "skinny": on the NARROW side the complement `U_⊥` has ~1%·min columns (truly skinny); on the WIDE side the complement of V_k has `d_in − k` columns (e.g. 6953 of 11008) — store whichever of `{complement, protected}` is narrower PER SIDE and apply `Q = U_⊥U_⊥ᵀ` or `Q = I − V_kV_kᵀ` accordingly. Never materialize m×m. | Closed-form | Ships as substrate (Issue 958). **Operator itself has prior art** (§7) — the anatomy report + factored-tail form are the surviving deltas | **Issue 958** (lane 1/3) |
| 2 | Basis computation: Gram eig on the smaller side (`V_k ← eig(WᵀW)`, `U_k ← W V_k Σ_k⁻¹`, reciprocal precomputed + zero-guard per `algebraic_op_ban_gate`) | Offline compute | One-shot per freeze version; sidecar keyed to the base | **Issue 958** (lane 1); `thin_svd_into`/`jacobi_eigen`/`newton_schulz` consumed, never forked |
| 3 | **Anchor commitment must be rotation-invariant.** At r=0.99 the cut sits inside the Marchenko–Pastur bulk where near-degenerate gaps make the tail basis defined only up to rotation — raw U/V bytes are solver-dependent. Commit the PROJECTOR digest (`BLAKE3(P)` over `U_⊥U_⊥ᵀ` / `V_kV_kᵀ`, ordered folds, pinned fp tolerance), never raw basis bytes | Format law | Deterministic across devices | **Issue 958** (lane 6); freeze-sidecar consumer is riir-neuron-db **Issue 629** |
| 4 | Ordering law, NARROWED to what is provable and needed: for rank-1 elementwise `A⊙X = diag(a)·X·diag(b)`, zero protected-block leak holds iff each protected span is invariant under its diagonal — i.e. span(U_k) [resp. span(V_k)] splits along the LEVEL SETS of a [resp. b]. Scalar diagonals are the trivial case. Ship: (i) level-set-aligned / scalar ⇒ no leak (axiom-free, concrete instances); (ii) an explicit 2×2 diagonal counterexample that leaks (the perturbation arm's negative). The general iff over arbitrary A is OUT OF SCOPE | Provable theorem (scoped) | Lean-4, axiom-free | **Issue 958** (lane 2) |
| 5 | **Factored-tail shipping (replaces the naive "post-transcode certification"):** a tail-confined delta factors `ΔW = U_⊥ C V_⊥ᵀ`; the tail span is closed under that product, so quantizing C (or the factors) preserves confinement EXACTLY — certified by construction. Quantizer SCALES are the paper's diagonal-D leak face; the ROUNDING face is isotropic noise (`1−(1−r)² ≈ 99.99%` into the protected block at r=0.99) and is why dense-project-then-quantize cannot be certified at near-zero tolerance. The residual certifier is a RELATIVE check vs the expected quantization-noise floor | Transfer invariant (NEW — no internal note carries it) | LoraPair-shaped: reader = U_⊥, writer = V_⊥ᵀ, payload = quantized C — deterministically constructed overlay | **Issue 958** (lanes 3/4 — highest value) |
| 6 | **Delta anatomy:** protected-block vs free-tail energy split per matrix, `E_prot = ‖P_UΔ‖² + ‖ΔP_V‖² − ‖P_UΔP_V‖²` via the stored per-side bases. Applies to ALL existing deltas (freeze/thaw, LoRA overlays, merges, consolidation) as an interference predictor and merge-planning report. NOT a compliance check — existing unconstrained deltas carry ~98% protected-block energy and would fail one by construction | Closed-form report | O(k·m·n) skinny GEMMs | **Issue 958** (lane 1) |
| 7 | Merge purification (paper finding 6) | Operator + evidence | 406 Fusion A keeps the TOP block; SST merges keep the TAIL — opposite blocks, no confirmation relation (406 §8.4 already marks Fusion A WEAKENED). The clean-merge result transfers only to tail-produced deltas | Cross-ref only; consumer gate is riir-ai **Issue 1116** |
| 8 | Fixed-anchor chain composition for tail deltas: `U_⊥ᵀ(ΣΔᵢ)V_⊥` — chained factored-tail overlays stay tail-confined; the composite is certified if every link is; free-block drift budget additive (triangle inequality) | Composition law | Certification ledger rides the freeze lineage (riir-neuron-db Issue 629) | **Issue 958** (lane 5) |
| 9 | Re-anchor churn policy: a re-anchor invalidates every outstanding certification; stale certs refuse loud | Policy + cost model | Ships as freeze-envelope version bump; churn cost inherits the paper's K-schedule shape | **Issue 958** (lane 6) + Issue 629 |
| 10 | Retention-vs-r monotone + the r≈0.99 operating point | Schedule/predictor (values are the paper's priors) | Offline sweep on deltas we hold; sigmoid drive `σ(a(r−r₀))`, never a hard edge; Report-the-Floor: beat a monotone-linear-in-r null | Advisory config — recorded as prior, no separate file |
| 11 | KL-drift merge-admission gate on a pinned probe set | Offline gate | Deterministic, cheap; floor = raw-sum merge drift; **consumes tail-produced deltas** | riir-ai **Issue 1116** |
| 12 | Scale prior — DEMOTED. Bonsai-2-27B is ternary and inference-only; the paper's ~1 pp cell does not transfer to anything Plan 470 trains. The honest prior at OUR trainable scales: 4B ≈ −1..−3.6 pp retention, 1.7B ≈ −6.9..−8.6 pp (vs SFT −18..−36 pp) — strong but not ~1 pp | Prior (demoted) | Recorded in Plan 470 with the corrected numbers | Recorded in Plan 470 |
| 13 | Soft penalties ≠ hard subspace constraints | Negative law | Do not emulate protected-subspace behavior with latent soft penalties anywhere in the stack | Negative row, pinned here + in 958 |
| 14 | Transform-agnostic corollary: the ordering law binds ANY post-hoc elementwise map (quantizer scales, per-channel gains, per-head scalers), not just optimizers — for the diagonal face; the rounding face is the isotropic case (row 5) | Theorem rider | Inside lane 2's scope | **Issue 958** (inside lane 2) |
| 15 | Delta signature (free-block energy vector as retrieval/dedup key) | Optional garnish | Only worth shipping if a retrieval consumer appears | Recorded, not filed |

**Path 0 verdict: MODELLESS-VALIDABLE.** No riir-train deferral is needed for the substrate (operator, anchors, anatomy, theorem, factored-tail form). The training loop itself (producing NEW tail-confined deltas under GD) is Path 0.5-applicable → Plan 470. Paths 1–3 (freeze/thaw correction, deterministic LoRA overlay, latent correction): path 2 IS the factored-tail overlay — `U_⊥ C V_⊥ᵀ` with deterministic C is a deterministically-constructed LoraPair overlay; path 1 consumes the anatomy report for freeze deltas.

**⚠ Consumer dependency (stated, per verdict round 1):** the certification lanes (5/8/9) have NO existing-delta consumers — every delta the stack ships today (freeze/thaw, LoraPair, TIES/archetype merges, consolidation) was produced WITHOUT tail confinement and carries ~98% protected-block energy (paper Fig. 4). Their value is gated on Plan 470 producing tail-confined deltas (or on hand-constructed factored-tail overlays). The anatomy report (6) is the modelless value for existing deltas and is consumer-independent. There IS a soft cascade from the training track to the certification lanes; it does not cascade onto the substrate/operator/theorem.

**Three-track adversarial panel (ran 2026-10-23, one spawn round):** No-GD advocate returned a 15-item inventory with in-tree host seams verified; Model-based advocate returned a 7-pipeline integration inventory with GPU-hours and gates, CONDITIONAL GO, cheapest falsifier a CPU pilot. Round-1 verdict corrections applied above (rows 3/4/5/6/7/12 rewritten; the advocate's original "post-transcode certification" and "confirms Fusion A" claims were WRONG and are corrected here — the reviewer's math: 406's on-manifold cut is top-~1%, SST's protected span is top-99%, and 98% of unconstrained-delta energy sits in the latter).

### 2.2 Where this lands in the stack

| Surface | Consumer | Mechanism | Boundary |
|---|---|---|---|
| katgpt-core (new `sst_certified`, feature `sst_project`, opt-in) | per-side bases + projector + anatomy + factored-tail form + certifier API | items 1/2/5/6 | katgpt-rs Issue 958 (substrate + theorem ONLY) |
| katgpt-rs `.proofs/KatgptProof` | the scoped non-commutativity theorem + 2×2 leak counterexample | items 4/14 | Issue 958 |
| riir-neuron-db `freeze.rs` | anchor sidecar row (rotation-invariant projector digest), stale-cert refusal | items 3/9 | riir-neuron-db **Issue 629** |
| riir-ai `LoRAHotSwap` / `dispatch_lora_merge` / archetype merges | merge-admission gate (KL probe, floor = raw-sum drift) + overlay wiring | items 7/11 | riir-ai **Issue 1116** |
| riir-train (Plan 470) | the training recipe that PRODUCES tail-confined deltas — the consumer that unblocks the certification lanes | the full recipe | Plan 470 |
| Priority-ladder note: game runtime (riir-ai) is the #1 fusion consumer via Issues 1116 + 629; the healer (riir-refine) has no weight deltas in its default surface — **no healer fusion** (audited: refine ships binaries, not model weights) | | | |

---

## 3. Fusion

- **F1 — SST × SAR (406): one spectrum, two opposite cuts.** 406 cuts off the TOP ~1% and studies what survives; SST protects the TOP 99% and adapts in the ~1% tail. Their "off-principal" findings and SST's confinement live at opposite ends of the spectrum — a 406-off-principal delta is still ~98% inside SST's protected span. The two cuts are complementary DIAGNOSTICS (406: where capability concentrates; SST: where updates may write), not confirmations. 406 §8.4 already marks its Fusion A WEAKENED; nothing in this paper revives it.
- **F2 — SST × SOPTV (231):** 231's off-principal retrieval cut is the complement of the top-10% (k_frac 0.1) — a third, intermediate cut of the same spectrum. Adapter retrieval (231) and update confinement (SST) can share basis computation but are different consumers; no confirmation relation claimed.
- **F3 — SST × quantization pipeline (NEW):** factored-tail shipping. `U_⊥ C V_⊥ᵀ` with quantized C is the ONLY certifiable way a quantized delta crosses the tail boundary — and it is exactly a deterministically-constructed LoraPair overlay (Plan 025's shape), so the raw/lora hot-swap law consumes it without modification.
- **F4 — SST × Plan 470 (training):** the recipe produces deltas that are certified-by-construction; the merge result then gives serving consolidation (1 base + Σ tail deltas instead of N full checkpoints) — the rethink RAM FAQ is the consumer.

## 4. Verdict

### Tier — per track (never pooled)

**Track (a) modelless inference/composition: GAIN.**
One-line: the bare operator has internal (406/231) and external (TSV-Merge, TATR, Subspace-Boosted Merging) prior art; the surviving modelless deltas are the anatomy report (consumer-independent), the factored-tail shipping form (consumer-gated on Plan 470), the scoped ordering theorem, and the rotation-invariant anchor commitment → Issue 958 (substrate) + Issues 629/1116 (consumer sides), feature `sst_project` opt-in, GOAT-gated; not Super-GOAT (Q1 fails below).

**Track (c) model-based training: PLAN FILED (Path 0.5 applicable — riir-train active moat).**
One-line: a real recipe for the exact recurring pain (specialist refits degrading prior suites/adapters/base encoder), optimizer-agnostic over our existing Adam+Muon lanes, CPU-pilot-cheap to falsify → riir-train Plan 470, flag `sst_tuning` default-off, per-pipeline GOAT gates. PRIMARY per the serving-envelope rule (reaches serving through retrained artifacts).

### Novelty Gate (§1.5)

| Q | Track (a) | Track (c) |
|---|-----------|-----------|
| Q1 no prior art? | **FAIL** for the bare operator — internal: 406 `spectral_rewire` (weight-delta SVD projection), 231 `off_principal_project` (vector variant). External: Task Singular Vectors/TSV-Merge (CVPR 2025, arXiv:2412.00081), TATR (KAUST 2025), Subspace-Boosted Model Merging (arXiv:2506.16506). **Open for the survivors:** factored-tail quantized shipping + rotation-invariant projector commitment + the scoped theorem — no internal note or external hit found carries them | **Class-dense, instance-clear** — Sculpting (ICLR 2026), OPLoRA (AAAI 2026), OGD/GPM/DOC lineage; we claim no recipe novelty, we adopt SST's instance wholesale via Plan 470 |
| Q2 new behavior class? | Partial — "tail-confined quantized deltas certified by construction" is a capability no incumbent has (surviving delta, consumer-gated) | Yes for our stack: cumulative adapter chains + safe trunk unfreeze (neither exists today) |
| Q3 product selling point? | Narrow — certification is infra hygiene, not a demo | "Our specialist refits never degrade the boards they don't touch; N experts serve as 1 base + tail deltas" |
| Q4 force multiplier? | Yes — freeze/thaw + LoRA hot-swap + merge + quantizer lanes | Yes — riir-train lanes × rethink serving × game adapters |

⇒ Track (a) capped at GAIN by Q1; Track (c) is a Plan by the §1.55/Path-0.5 rule.

### MOAT Gate (§1.6)

| Domain | Assessment |
|---|---|
| katgpt-rs | In scope: fundamental weight-space spectral primitive + scoped provable theorem; public; feature `sst_project` opt-in; GOAT decides promotion. Issue 958 + this note. |
| riir-train | Active moat — Plan 470 is the filing. Recipe + GPU-hours + gates per Path 0.5 documentation rule. |
| riir-ai / riir-neuron-db | Consumer sides filed as Issues 1116 / 629 in their own repos (boundary: a katgpt-rs issue cannot bind sibling-repo work). |
| riir-refine | **No fusion** — healer surface carries no model weights (audited); the "did I ask what this does for the healer?" check returns a reasoned no. |

## 5. Vocabulary Crosswalk (paper → codebase)

| Paper term | Our term / substrate |
|---|---|
| Subspace Tuning (SST) | `sst_certified` module; protected-complement projection |
| protected directions / principal subspace (top 99%) | top-k singular span `(U_k, V_k)`, `k = ⌊r·min(d_out,d_in)⌋` |
| free subspace / tail (~1%) | bottom singular complement `U_⊥, V_⊥` (narrow side) / `(I − V_kV_kᵀ)` (wide side) |
| rank fraction r = 0.99 | protected fraction; per-lane override (the r = 0.90 exception) |
| project the applied update | final-delta certification / factoring (order law) |
| Adam preconditioner leak | scoped non-commutativity theorem (Lean, Issue 958 lane 2) |
| refresh R100/R10 | re-anchor; basis churn; stale-cert refusal |
| catastrophic forgetting / prior capability | retention Δ on pinned probe board |
| KL drift (WikiText-2) | probe-set drift gate (merge admission, Issue 1116) |
| expert merging (task arithmetic) | `W_base + ΣΔ` merge gate; tail-produced deltas only |
| Gram eigendecomposition | `jacobi_eigen` (spectral_pencil) / `thin_svd_into` (subspace_phase_gate) |
| SDFT / on-policy | self-distillation fine-tune; Plan 470 B-lane composition |
| (406/231) "off-principal" = complement of top-1%/10% | DIFFERENT cut than SST's complement-of-top-99% — opposite/intermediate ends of one spectrum |

## 6. Open Primitive Sketch (katgpt-core, `sst_certified.rs`, feature `sst_project`)

```rust
/// Per-side basis, whichever representation is narrower:
/// narrow side -> complement U_perp (0.01*min cols); wide side -> protected V_k (k cols).
/// Commitment is the PROJECTOR digest (rotation-invariant), never raw basis bytes.
pub struct SstSide { pub basis: Vec<f32>, pub cols: usize, pub kind: SideKind, pub projector_blake3: [u8; 32] }
pub struct SstAnchor { pub out: SstSide, pub inn: SstSide, pub base_blake3: [u8; 32], pub r: f32 }

pub fn fit_anchor(w_ref: &[f32], (m, n): (usize, usize), r: f32) -> SstAnchor;   // Gram eig on smaller side

/// Anatomy (ALL deltas): protected-block energy, never materializing m×m.
pub fn protected_energy(a: &SstAnchor, delta: &[f32], (m, n): (usize, usize), scratch: &mut [f32]) -> f32;

/// Factored-tail shipping (tail-produced deltas): ΔW = U_⊥ C V_⊥ᵀ; quantize C, confinement is
/// preserved by span closure. Certify the DECODED product with a RELATIVE tolerance against the
/// expected quantization-noise floor — never a near-zero absolute bar.
pub fn factor_tail(a: &SstAnchor, delta: &[f32], (m, n): (usize, usize)) -> TailFactor { u_perp, c, v_perp_t }
pub fn certify_decoded_tail(a: &SstAnchor, decoded: &[f32], rel_tol: f32, ...) -> Result<SstCert, SstLeak>;
```

GOAT sketch: G1 `r=0 ⇒ Π = I` byte-identical (disarmed posture) + the 2×2 leak counterexample arm REDs + anatomy vs brute-force oracle agreement; G2 **ms-class per matrix at 4k-class shapes** (measured, `--release`, bench_preflight PROVENANCE — re-pinned from µs: at r=0.99 one application moves ~0.7–5 GFLOP, ms on CPU, sub-ms on GPU); G3 all-features/no-default clean; G4 zero-alloc; ordered folds at every reduction crossing a commitment boundary (`dot_f32_ordered` precedent). Modelless by construction → promotion unobstructed by the modelless-gain rule once gates pass.

## 7. Related Work — cousins and the §3.6 signal-diffs

| Cousin | What it ships | Signal-diff vs SST |
|---|---|---|
| **Research 406 (SAR)** | `spectral_rewire` — weight-delta SVD projection keeping the TOP ~1% block; purify/merge/extract; §8.5 ledger: concentration REFUTED at NPC and 1.5B scales; §8.4 marks Fusion A WEAKENED | **Opposite cut, opposite block:** 406 keeps what SST discards. 406's off-principal (≤0.191 on-manifold at 1.5B) is evidence deltas avoid the TOP 1% — it says NOTHING about the top-99% span, where the paper measures 98% of unconstrained-delta energy. No confirmation relation; 406's diagnostic (where capability concentrates) and SST's constraint (where updates may write) are complementary instruments. No ordering law, no certification, no re-anchor policy in 406. |
| **Research 231 (SOPTV)** | `off_principal_project(q)` — projects a length-d query VECTOR off the top-10% for adapter retrieval; SparseTaskVector storage | Third, intermediate cut (complement of top-10%) of the same spectrum; object is a vector for retrieval, not a matrix delta for composition. Shares basis computation; different consumer; no confirmation relation. |
| **TSV-Merge (arXiv:2412.00081, CVPR 2025)** | task singular vectors to reduce merge interference | External operator prior art — kills Q1 for the bare merge operator. Does not carry the training-loop enforcement, the factored-tail quantized shipping, or the re-anchor policy. |
| **TATR (KAUST 2025) / Subspace-Boosted Merging (arXiv:2506.16506)** | trust-region / subspace-boosted merging | Same kill class as TSV; same survivors. |
| **Sculpting (ICLR 2026) / OPLoRA (AAAI 2026)** | training-side subspace protection (gradient-projection / factorized) | The paper's own §4.1 baselines: both leak or mis-scale vs applied-update projection. Plan 470 adopts SST's instance; no novelty claim filed on the recipe. |
| **Research 238 / 163 / 547** | LoRA-Muon spectral manifold; EoS curvature allocation; HRM second-pass | Training-track neighbors; none constrains the update subspace for retention. Plan 470 cites them as composition context. |

## 8. Honest Caveats

1. **The factored-tail shipping claim is inferred, not measured.** Row 5's value rests on span closure (exact algebra) plus the paper's §2.2 controlled study (theirs, fp training, no quantizer). The PLANTED-LEAK and NOISE-FLOOR GOAT runs on a real transcoded tail delta are what upgrade it from architectural to quality-grade; until then its grade is architectural.
2. **All retention numbers are the paper's** (Qwen3/Olmo, their domains/benches). At our trainable scales the retention edge is real but smaller (1.7B: −6.9..−8.6 pp, not ~1 pp). Plan 470's gates re-measure; a G1 failure there kills the training track honestly and this note records the negative.
3. **The r = 0.99 tail is tiny at small widths** — at min-dim 64 it is rank ≤ 1 (capacity confound, not an SST measurement). Plan 470 carries a free-rank floor; per-lane r override (the paper's own r = 0.90 exception).
4. **406's NPC-scale concentration failure stands and is a DIFFERENT claim** — it measures top-1% concentration of unconstrained deltas; it neither supports nor refutes tail confinement. The certification lanes target matrix deltas produced UNDER tail confinement (Plan 470 outputs / factored-tail overlays); they are NOT proposed for 64-dim vector shards or for existing unconstrained deltas.
5. **Anchors are per-matrix and per-checkpoint; commitments are rotation-invariant but tolerance-pinned.** Near-degenerate Marchenko–Pastur bulk cuts make raw basis bytes solver-dependent — only the projector digest is portable. **Wide-side sidecars are roughly base-matrix-sized at MLP shapes** (`V_k` at 4055×11008 ≈ 44.6M floats vs the ~45M base) — anchors are NOT cheap for wide matrices; the anatomy report carries the per-matrix cost. Anchor churn on re-fit is real cost (the paper's own K-schedule); the stale-cert refusal law is what keeps it honest.
6. **Boundary note:** lanes that live in sibling repos (freeze sidecar, merge gate) are filed THERE (riir-neuron-db Issue 629, riir-ai Issue 1116), not in katgpt-rs Issue 958 — katgpt-rs is upstream and its issues cannot bind sibling-repo work.

## PASS-Redirects (synthesis)

> **Research 631 redirects (both cousins updated in this commit, corrected per verdict round 1):** arXiv:2610.22293 "Towards Continual Learning: Mitigating Forgetting With Subspace Tuning" — applied-update complement projection + refresh; see Research 631 (anatomy + factored-tail composition law), riir-train Plan 470 (training recipe), riir-neuron-db Issue 629 (freeze sidecar), riir-ai Issue 1116 (merge gate). Relation to cousins: OPPOSITE/INTERMEDIATE cuts of one spectrum — 406 keeps the top ~1% block, 231 drops the top 10%, SST confines into the complement of the top 99%; no confirmation relation holds between their merge claims.
