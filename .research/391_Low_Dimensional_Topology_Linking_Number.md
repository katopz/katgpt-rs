# Research 391: Low-Dimensional Topology of Deep Neural Networks — Linking-Number Detector + Fold Correction

> **Source:** Junyu Ren & Lek-Heng Lim, *Low-dimensional topology of deep neural networks*, ICML 2026 (PMLR 306).
> **arXiv:** [2606.31856](https://arxiv.org/abs/2606.31856)
> **Date:** 2026-07-07 (addendum 2026-09-12)
> **Status:** EXECUTED + Option B CLOSED — Plan 410 shipped 2026-07-07 (`linking_fold_fold` DEFAULT-ON, all GOAT gates PASS); Issue 757 (detector Option B) DONE 2026-09-12: the original 50 ms @ n=2×1000 budget RESTORED (28.3 ms linked, G2b gate; 3.69 ms @ n=2×200, was 115.7) — [Bench 717](../.benchmarks/717_linking_detector_option_b.md). Detector stays opt-in pending a consumer.
> **Related Research:** 219 (DEC substrate), 242 (topological state tracking), 294 (viable manifold graph), 296 (Stokes vocabulary crosswalk), 317 (Gibbs attractor — same Plan-276 caveat class), 371 (Hopf bifurcation — different "Hopf").
> **Related Plans:** 251 (DEC operators), 314 (Stokes wrappers), 410 (this paper — queued).
> **Classification:** Public (katgpt-rs)
>
> **PASS-Redirects (synthesis):** Faramarzi, Lamb, Rish [arXiv:2607.03580 "When Geometry Aligns: Dihedral Hidden-State Transformations in UNet, ViT, and DiT Architectures"] — applies D4 dihedral reflections (horizontal/vertical/diagonal flips) to hidden states of diffusion/vision models. **Strongest conceptual match in the corpus:** the paper's D4 reflections ARE fold operations in this note's sense — `|x|` is a reflection, and this note (Theorem 3.7) proves reflections are the unique way to break topological linking. The synthesis: reflections are a **dual-use geometric tool** — the paper uses *consistent* D4 reflections during fine-tuning as a **regularizer** (capacity-reducing, stability-preserving per Rademacher complexity); this note uses *targeted* fold reflections at inference as a **corrector** (link-breaking, separability-restoring). Opposite uses of the same operation. The paper doesn't add actionable insight to this note's shipped `fold_projection` (Plan 410, DEFAULT-ON) because: (1) the paper's domain is 2D spatial feature maps in vision/diffusion models (out of scope), (2) the paper's multi-branch consistency principle (transform ALL attention heads or none) doesn't apply to our fold (operates on full latent vectors, not per-head), (3) our latent states carry trivial symmetry groups (R314). The connection is conceptual — both are reflection-based geometry interventions — not actionable.

## TL;DR

The paper proves that **linking number** — an *extrinsic* topological invariant of how two class manifolds are embedded in ambient space — is preserved by every width-d feedforward network with coordinate-wise **monotonic** activations (ReLU, sigmoid, tanh). Therefore no such network, however deep, can linearly separate two topologically linked class manifolds (link ≠ 0). Only **folding** operations break the constraint: ResNet skip (`|x| = x + 2·ReLU(−x)`), attention (two-token V-shape ≈ smoothed `|x|`), and non-monotonic activations (GELU/Swish/Mish). Width ≥ d+1 also suffices (universal approximation threshold).

**Distilled for katgpt-rs (modelless, inference-time):**
Two transferable modelless primitives, both entirely new to the codebase (zero prior-art across all five repos — see §3.1):

1. **`linking_detector`** — Algorithm 1: take two point clouds X, Y in R^d (e.g., two clusters of HLA states, two NPC behavior classes, two NeuronShard style clusters), PCA-project to R^3, build ε-filtered k-NN graphs, extract a fundamental cycle basis per graph via spanning forest, compute the Gauss linking integral over O(β_X · β_Y) basis-cycle pairs. **Verdict: linked (link ≠ 0) or unlinked.** This is the diagnostic half: "are these two latent clusters topologically entangled?"
2. **`fold_projection`** — a deterministically constructed coordinate fold `x ↦ c + |x − c|` (or any GELU/Swish-style surrogate with a strict local extremum on the data domain), applied coordinate-wise to a latent subspace. This is the **correction** half: when the linking detector fires, monotonic projection is provably doomed; a single fold layer unlinks the manifolds and restores linear separability. No training, no backprop — a closed-form correction.

The pair is a GOAT (not Super-GOAT) — see §3.2 for the honest accounting.

---

## 1. Paper Core Findings

### 1.1 The linking number (Definition 3.2 / 4.1)

For two disjoint, oriented, closed manifolds M^m, N^n ⊂ R^d with d = m + n + 1, the **linking number** `link(M, N) ∈ Z` is the topological degree of the Gauss map `G(x,y) = (x − y)/|x − y| : M × N → S^(d−1)`. Combinatorially (d = 3): `link(X, Y) = ½ Σ_p ε_p` over crossings of any regular projection. For the Hopf link (two interlocked circles), `link = ±1`.

**Key property (extrinsic, not intrinsic):** linking depends on how manifolds are *embedded in ambient space*, not on the manifolds' intrinsic topology. This distinguishes it sharply from the Betti-number TDA the codebase already touches (Research 219 / Plan 251 DEC operators compute intrinsic homology via `d∘d=0`).

### 1.2 The impossibility theorems (Thm 3.7 / 4.7)

**Width-d feedforward network with coordinate-wise monotonic activations cannot linearly separate linked manifolds.** Proof spine:

- **Invertible affine layers** = ambient homeomorphisms → preserve link up to sign (Lemma B.2).
- **Monotonic activations** = link homotopy `H_t(x) = (1−t)x + tσ(x)` applied simultaneously to both manifolds; monotonicity rules out coordinate-wise collisions during interpolation, so `link(σ(M), σ(N)) = (−1)^r · link(M, N)` (Lemma C.7).
- **Rank-deficient affine layers** force intersection when link ≠ 0 (Lemma C.8) — sliding components along the kernel vector would otherwise give a link homotopy to a linearly-separable (hence link = 0) configuration, contradiction.
- **Linear separability ⟹ link = 0** (Lemma C.6) — separating hyperplane puts each manifold in a convex half-space; straight-line contraction to a point gives a link homotopy to link = 0.

Applies to ReLU, sigmoid, tanh, Leaky-ReLU, ELU — *any* coordinate-wise monotonic σ. Sigmoid (the codebase's mandated non-softmax projection) is explicitly covered (Theorem 4.7, §4.2). Depth provides no escape (Table 2: ReLU mean accuracy *degrades* with depth on Hopf link).

### 1.3 The folding escape mechanisms (§5)

| Mechanism | Why it breaks the homotopy argument | Construction |
|---|---|---|
| **Non-monotonic activation** (GELU, Swish, Mish, `abs`) | Local extremum creates a fold line; the straight-line homotopy crosses it, allowing mirror-image points to collide | Coordinate-wise `abs(x)` applied iteratively (Fig 4 / 9) |
| **ResNet skip connection** | A single residual block realizes the fold map with only monotonic ReLU + skip: `abs(x) = x + 2·ReLU(−x)` | Theorem 5.2 |
| **Attention (pure transformer)** | Two-token attention with distinct positional encodings yields a sigmoid attention weight; explicit scalar weights produce a smoothed V-shape near the origin | Theorem 5.3, Fig 5b |
| **Width ≥ d+1** | Hanin–Sellke universal approximation kicks in | Theorem D.1 |

### 1.4 Algorithm 1 — link detection for point clouds (§H)

This is the modelless inference-time primitive of the paper:

```
1. Project X ∪ Y to R^3 via PCA.
2. Build ε-filtered k-NN graphs G_X, G_Y (optionally mutual k-NN).
3. Spanning-forest BFS → fundamental cycle basis C_X, C_Y
   (generates H_1(G; Z); every cycle is an integer combination of basis cycles).
4. For each (C ∈ C_X, D ∈ C_Y):
     ℓ ← Gauss integral (midpoint quadrature, N_sub subdivisions) over C × D.
     If ℓ ≠ 0 → LINKED, return witness (C, D, ℓ).
5. Else → NOT LINKED.
```

Complexity: `O(n log n + β_X β_Y L² N_sub²)` — runs in tens of seconds on CIFAR-10 class pairs (~10⁴ points, β ~ 10², L ~ 20). The paper validates it on real CIFAR-10 PCA-3D: detects bird–deer link = −1 at ε = 0.034 (Fig 7), and linking consistency correlates with within-CIFAR confusion (Spearman r ≈ 0.48, p < 0.001).

### 1.5 Corollary E.1 — minimum-width lower bound

For any continuous coordinate-wise monotonic activation, the minimum width for universal approximation on R^d is `w_min ≥ d + 1`. The proof is a direct corollary of the impossibility theorem (use a non-trivially linked pair in R^d). This is the same d+1 threshold the codebase's `subspace_phase_gate` (Plan 301) and `tucker` HOSVD (Plan 326) already respect implicitly for rank-bounded SVD/Jacobi — but the paper gives the *topological* reason.

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper → codebase — both layers checked, both empty)

| Paper term | Codebase equivalent | Status |
|---|---|---|
| linking number, Hopf link, unlink | (none) | **zero hits** — grep `linking\|hopf\|fundamental cycle\|winding number\|cycle basis` returns OAuth account-unlinking (seal-online-remaster), KG triple lineage linking (riir-games), and Plan 371's Hopf *bifurcation* (different "Hopf" — ODE eigenvalue, not link) |
| extrinsic / ambient topology | (none) | **zero hits** — grep `extrinsic topolog\|ambient topolog\|ambient homeomorphism\|extrinsic invariant` returns nothing |
| folding map, coordinate fold, non-monotonic activation | (none in the topological sense) | "folding" hits = ThoughtFold chain compaction (Plan 195), `xor-folding` hashing, RMSNorm gamma *folding* (Plan 160) — all unrelated |
| monotonic activation preservation | (none as a linking theorem) | "monotonic" hits = version counters, `monotonically-decreasing gain`, `partition-of-unity` B-spline — none about linking |
| topological obstruction, class manifold separation | `subspace_phase_gate` (Plan 301 — phase transition), `DEC operators` (Plan 251 — exterior calculus) | **distinct structures** — phase gate is about participation-ratio phase transitions; DEC is intrinsic homology via `d∘d=0`. Neither detects *extrinsic* linking |

**Confirmed:** this is the canonical "vocabulary gap" pattern (Research 296 lesson). The codebase's topology substrate (DEC/Stokes) and the paper's topology (extrinsic linking) are different branches of topology — `curl(grad)=0` (DEC) and `link(M,N) ≠ 0` (this paper) don't overlap. Genuinely no prior art; not a vocabulary mismatch on a shipped mechanism.

### 2.2 What the paper adds beyond our shipped stack

| Paper claim | Shipped? | Gap |
|---|---|---|
| Width-d monotonic activation cannot unlink linked manifolds | ❌ No theorem ships | **The novel theoretical contribution** — closes a gap the codebase has implicitly (sigmoid is mandated but never proven topologically doomed in specific cases) |
| Algorithm 1: modelless linking detector | ❌ Nothing like it ships | **The novel inference primitive** — katgpt-rs has no detector for "are these two latent clusters topologically entangled" |
| Coordinate fold as unlinking correction | ❌ Nothing ships | **The novel correction primitive** — a closed-form `|x|`-style fold is a new class of modelless projection |
| Skip/attention/GELU break the constraint | ⚠ Partially — skip/GELU are widely used, but the *topological reason* was never the design rationale | Architectural rationale only — not a new primitive |
| Width ≥ d+1 threshold | ⚡ Implicit — `subspace_phase_gate` and `tucker` SVD bounds respect this | Paper gives the topological proof; not a new primitive |
| Higher-dim linking (S^n ⊔ S^n in R^(2n+1)) | ❌ Not shipped | The R^3 algorithm extends; n=2 case used in experiments |

### 2.3 Latent-space reframing (mandatory step 3) — where it lands in the 7 Super-GOAT factory modules

| Substrate | Reframing | Tier |
|---|---|---|
| **HLA state** (`riir-engine/src/hla/`) | Per-NPC 8-dim HLA states of two NPC classes (e.g., "fleeing" vs "fighting") can be linked in R^8. The 5 synced affect scalars (valence/arousal/desperation/calm/fear) are **dot-product + sigmoid** projections → monotonic → **provably cannot linearly separate linked HLA class manifolds**. The detector says *when* the projection is doomed; the fold is the fix. | **Private selling point** → riir-ai guide |
| **`latent_functor/`** | Functor applications as vector ops; sigmoid gates are coordinate-wise monotonic → preserve link. A functor whose gate is sigmoid is structurally unable to unlink linked NPC behavior manifolds. | Private selling point → riir-ai guide |
| **`cgsp_runtime/`** | Curiosity-driven exploration may wander into linked regions; the detector diagnoses "stuck exploring two entangled zones." | Auxiliary, riir-ai only |
| **LatCal** (`riir-chain/src/encoding/`) | Out of scope — LatCal operates on fixed-point raw numerics, not manifold geometry. The scalar outputs of a fold projection cross the sync boundary raw; the *fold itself* is local-latent. | Out of scope for chain |
| **`NeuronShard`** (`riir-neuron-db/src/shard/mod.rs`) | `style_weights[64]` is a fixed-size Pod. Two shards' style clusters can be linked in R^64. Retrieval by scalar projection (the `ItemEmbedIndex` cosine query, Plan 362) is monotonic-ish → **doomed if clusters are linked**. Detector + fold is a new retrieval-quality gate. | **Private selling point** → riir-neuron-db guide |
| **DEC** (`katgpt-rs/crates/katgpt-dec/src/`) | **Complementary, not overlapping.** DEC computes *intrinsic* homology (curl, divergence, harmonic). Linking is *extrinsic* ambient topology. The two together cover both branches: DEC detects belief-mass divergence; linking detects class-manifold entanglement. Fusion possible (see §2.4) but distinct substrates. | Open primitive in katgpt-rs |

### 2.4 Fusion opportunities (the Super-GOAT hook — flagged but not committed)

1. **Linking detector × DEC harmonic projector** (katgpt-rs internal) — the harmonic component of a belief cochain (Plan 251 `harmonic_projector`) detects the *flat-minimum basin*; the linking detector says whether two basins are *interlocked*. Together: "two NPCs' belief states are in linked basins → no monotonic projection can tell them apart → apply a fold." This is a fusion-GOAT, not a single-paper Super-GOAT. Flagged for a future cross-paper pass.
2. **Linking detector × `subspace_phase_gate`** (Plan 301) — the phase gate detects the N ≥ d participation-ratio transition; linking detects a *different* kind of obstruction (extrinsic entanglement vs intrinsic rank deficit). Two failure modes, two detectors, one routing decision. Fusion candidate.
3. **Fold projection × `PersonalityWeightedComposition`** (Plan 297) — personality blend uses sigmoid-gated latent layer composition. If two personality direction vectors become linked, the blend is topologically doomed. The fold projection is a modelless pre-blend correction. Fusion candidate → riir-ai.

These fusions are flagged in the note but NOT pursued in Plan 410 (which ships the open primitives only). Per §1.5, a fusion claim requires all-4-YES on the novelty gate; the fusions above are Q4-multiplier candidates that need Q1–Q3 verification in a dedicated pass.

---

## 3. Verdict

### 3.1 Novelty gate (Q1–Q4)

| Q | Answer | Evidence |
|---|---|---|
| **Q1. No prior art?** | ✅ YES | Triple-layer grep across all 5 repos (notes + code + vocabulary translation): zero hits for `linking number`, `Hopf link` (as a link, not Plan 371's bifurcation), `extrinsic/ambient topology`, `coordinate fold`, `non-monotonic activation as linking-preserving mechanism`. The DEC/Stokes substrate (Plans 251/314, Research 219/296) is *intrinsic* homology — a different branch of topology. |
| **Q2. New class of behavior?** | ✅ YES (for the detector) | The codebase has *no* modelless way to ask "are these two latent clusters topologically entangled such that monotonic projection is doomed." It's a new diagnostic class. |
| **Q3. Product selling point?** | ⚠️ MODERATE | "Our NPCs detect when their affect scalars are topologically insufficient and apply a deterministic fold to unlink" — real capability, but niche. Not a headline ("our NPCs feel emotions" tier). Closer to a quality/retrieval gate. |
| **Q4. Force multiplier?** | ✅ YES | Connects to HLA (P-equivalent), `latent_functor`, `NeuronShard` retrieval, DEC (complementary). Multi-pillar. |

### 3.2 Tier: **GOAT** (not Super-GOAT)

**One-line reasoning:** Novel modelless diagnostic + correction primitives (Q1+Q2+Q4 all YES), but Q3 is moderate — this is a quality/retrieval gate, not a headline product capability. The Super-GOAT bar is "product selling point" and this doesn't clear it; the GOAT bar is "provable gain over existing approach" and this clears it cleanly (a monotonic projection that is provably doomed is strictly worse than one gated by the fold correction).

**MOAT gate per domain:**
- **katgpt-rs (public engine):** The detector + fold projection are generic math with no game/chain/shard semantics → **in scope** as a paper-derived fundamental primitive. Ships behind a feature flag. ✅
- **riir-ai (private runtime):** The HLA/functor application is a private selling point → riir-ai guide (cross-ref from this note). Defer to a follow-up plan in riir-ai if the katgpt-rs primitive ships and proves useful.
- **riir-neuron-db (private shards):** The shard-retrieval application is a private selling point → riir-neuron-db guide (cross-ref). Defer similarly.
- **riir-chain / riir-train:** Out of scope.

### 3.3 Why not PASS ("already ships")?

The architecture-theory half (use GELU/skip/attention) is well-known practice and would justify a PASS on those grounds alone. But Algorithm 1 (the detector) and the coordinate-fold correction are genuinely not shipped anywhere in the codebase, and the §3.6 defend-wrong rule says a PASS verdict backed only by architectural reasoning is the #1 false-PASS failure mode. The detector is a *new diagnostic class*, not a re-skin of an existing primitive. GOAT is the honest tier.

### 3.4 Modelless-unblock check (§3.5 — N/A here, but noted)

This paper doesn't gate any existing feature, so the §3.5 protocol doesn't fire. But the fold projection *is* a §3.5 path-3-style latent-space correction (deterministic, closed-form, no GD) — it's exactly the kind of modelless correction the protocol prefers over riir-train deferral. If a future gate fails because "two HLA classes are topologically linked," the fold is the §3.5 fix, not "train a deeper network."

---

## 4. Plan — queued (Plan 410)

**Target:** `katgpt-rs/crates/katgpt-core/src/topo/` (new module) + Cargo feature `linking_fold`.

**Open primitive scope:**
1. `linking_detector` — Algorithm 1 (PCA-3D + ε-kNN + cycle basis + Gauss integral).
2. `fold_projection` — coordinate-wise `|x − c|` fold (and a GELU-surrogate variant) as a deterministic unlinking map.
3. GOAT gate: G1 (detects synthetic Hopf link, link = ±1, on noisy point cloud), G2 (sub-ms on ≤10⁴-point clouds), G3 (no regression), G4 (alloc-free hot path for the fold; detector may allocate), G5 (determinism — link is integer-valued, fold is closed-form).

**Out of scope for Plan 410:**
- HLA integration (riir-ai follow-up).
- Shard retrieval integration (riir-neuron-db follow-up).
- DEC × linking fusion (§2.4 #1 — separate fusion pass).
- Lean 4 proof that the fold unlinks (the paper proves it over ℝ; an f32 spec-match test suffices for v1, mirroring the riir-ai HLA-boundedness pattern).

The plan file `katgpt-rs/.plans/410_*.md` is **not** created in this session — the user invoked the research skill, not "implement." The plan is queued; the user can open it as a follow-up.

---

## 5. Cross-references (read on demand)

- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` — the *intrinsic* homology substrate (DEC operators). This note adds the *extrinsic* linking complement.
- `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` — the canonical vocabulary-translation lesson this note followed (paper vocabulary alone returns zero hits; codebase vocabulary alone returns zero hits; the lesson is *both were genuinely empty* here, not a translation miss).
- `katgpt-rs/.plans/251_dec_operators_cell_complex.md` — DEC substrate (d, δ, hodge_decompose). Complementary, not overlapping.
- `katgpt-rs/.plans/301_runtime_subspace_phase_gate_primitive.md` — phase-transition detector. Different obstruction class (rank deficit vs extrinsic linking); fusion candidate (§2.4 #2).
- `riir-ai/.research/123_Latent_Functor_Runtime_Guide.md` — functor applications as monotonic vector ops; linking preservation applies directly.
- `riir-neuron-db/.research/012_egg_shell_pruner_funcattn_item_retrieval_fusion.md` — `ItemEmbedIndex` cosine retrieval; linking detector is a new retrieval-quality gate.

---

## TL;DR

**Pre-flight done:** 4 READMEs + `riir-ai/.docs/README.md` + 4 `.research/` corpora + latent_functor/hla/dec/sense module trees + 5-repo × 2-layer vocabulary-translated grep (paper vocab + codebase vocab, both returned zero prior-art hits — genuinely novel, not a vocabulary miss).

**Verdict: GOAT** (modelless). The paper ships two genuinely-new modelless primitives — a **linking detector** (Algorithm 1: PCA-3D + ε-kNN + cycle basis + Gauss integral) and a **fold projection** (`|x−c|` coordinate fold) — that close a gap the codebase has implicitly (every sigmoid projection is monotonic, hence topologically doomed on linked manifolds, but the codebase has no way to detect when). The detector is a new diagnostic class; the fold is a §3.5 path-3 latent correction. Not Super-GOAT because Q3 (product selling point) is moderate — it's a quality/retrieval gate, not a headline capability.

**Routing:** Open primitive → `katgpt-rs` (`linking_fold` feature). Private guides (HLA + functor + shard-retrieval applications) → deferred to riir-ai/riir-neuron-db follow-up plans if the katgpt-rs primitive ships and proves useful. Plan 410 queued but not created this session. *(Superseded — see §6.)*

---

## 6. Post-implementation addendum (2026-09-12, closure session)

**Shipped state (verified against the tree, not the note's 2026-07-07 text):**

- **Plan 410 executed without a plan file** — the implementation (code + bench + Cargo comments) references "Plan 410" phases (T4.1 bench, T4.2 alloc gate, T4.4 Option C split) but no `.plans/410_*.md` was ever committed; `.plans/.highwater` never reached 410 via this work. Recorded here so the number is not re-derived as "missing".
- **`linking_fold_fold` is DEFAULT-ON** (`default` feature list, Cargo Phase 12, 2026-07-07, "T4.4 Option C: hot-path fold ships default-on; cold-path detector stays opt-in").
- **`linking_fold_detector` stays opt-in** — Issue 050 **resolved 2026-07-07 (Option A: accept the recalibrated 500 ms @ n=2×200 audit-cadence budget)**; the original 50 ms @ n=2×1000 plan budget was brute-force-unreachable. Option B (algorithmic work toward the original budget) remained open with no issue file — **now tracked at `.issues/757`**.
- **Perf work DID land after the 407 ms measurement**: `CycleBounds::may_link` bounding-sphere early-skip (provable bound `gap² > P_C·P_D/(2π)` ⇒ integral rounds to 0 — correctness-safe), full-SoA auto-vectorized Gauss quadrature, and the opt-in `max_cycles_per_cloud` cap. Fresh re-bench (2026-09-12, 4090 box, `CARGO_TARGET_DIR=/tmp/lf410 cargo bench -p katgpt-core --features linking_fold --bench bench_410_linking_fold_goat`):

| Gate | 2026-09-12 measured | Budget | Verdict |
|---|---|---|---|
| G1 correctness smoke | Hopf `link=1`; unlinked `0`; fold-unlinked `0` | exact | ✅ |
| G2 detector cold-path @ n=2×200, d=8 | **115.74 ms median** (min 115.18, max 118.91) | ≤ 500 ms | ✅ (3.5× better than the historical 407 ms the Cargo comment still quotes) |
| G2 fold Abs D=8 / Gelu D=8 | 16.25 / 15.76 ns | ≤ 50 ns | ✅ |
| G2 fold Abs D=64 / Gelu D=64 | 18.18 / 26.14 ns | ≤ 500 ns | ✅ |
| G5 determinism | detector `link=1` ×3; fold bit-identical ×100 | exact | ✅ |

No `.benchmarks/410_*.md` doc exists — the GOAT numbers live in the Cargo feature comment and this addendum; the properly-numbered bench doc lands with the Issue 757 re-gate.

**Consumer-context reframe (the step-4 question this note pre-dated, cf. the 2026-09-02 canonical-failure-#5 mandate):** the healer surface (fusion priority #2) — two lint-rule embedding clusters that are linked in PCA-3D of the riir-rag span-embedding space would predict retrieval top-1 confusion between those rules (the paper's CIFAR-10 analogue: linking consistency ↔ confusion, Spearman r ≈ 0.48). **Fusion idea, novelty TBD** — tracked as the follow-up section of `.issues/757`, not committed as a plan; the healer corpus is discrete AstChunker spans, not smooth manifolds, so the k-NN-cycle machinery's manifold assumption is unverified there.

**v2 re-check (arXiv v2, 2026-08-18):** the note's distillation covers the v2 content (Thms 3.7/4.7/5.2/5.3, Algorithm 1 §H, Corollary E.1, CIFAR-10 §I). The v2-only appendix material (autoencoder bottleneck Corollary A.5 — "bottleneck dimension d alone determines the topological constraint; place nonmonotonic/skip layers before compression"; layer-by-layer link+d_min tracking §G.8) is architectural guidance for bottleneck surfaces (KV-compression ladders, drafter bottlenecks), noted here for grep; not actionable as a primitive.
