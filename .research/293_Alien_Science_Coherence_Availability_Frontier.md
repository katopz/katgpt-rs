# Research 293: Alien Space of Science — Coherence × Availability Frontier Sampler

> **Source:** [The Alien Space of Science: Sampling Coherent but Cognitively Unavailable Research Directions](https://arxiv.org/abs/2603.01092) — Artiles, Weiss, Brinkmann, Rahwan, Schölkopf, Pal, Larochelle, Goyal, Rahaman (Tiptree / MPI / Mila / Polytechnique, May 2026, arXiv:2603.01092v2)
> **Date:** 2026-06-23
> **Status:** Done — GOAT verdict
> **Related Research:** 089 (OPUS redundancy selection), 240 (SGS CGSP), 276 (PersonalityWeightedComposition), 281 (Salience Tri-Gate)
> **Related Plans:** 129 (OPUS), 274 (CGSP), 282 (Dual-Pool CGSP), 297 (Personality composition), 303 (Salience Tri-Gate), **311 (Alien Sampler primitive, this note)**
> **Cross-ref (riir-ai):** R126 (NPC CGSP Super-GOAT guide — closest cousin), R145 (CWM NPC emergent diversity), R148 (Per-Tick Salience NPC guide), R153 (latent field steering game guide)
> **Classification:** Public

---

## TL;DR

The paper introduces a clean **two-axis decomposition of search**: separate *coherence* (do these components form a viable whole?) from *availability* (is this combination likely to be produced by an existing community?), and rank candidates by a z-scored linear fusion `Fβ = (1−β)·zC(S) + β·zU(S)` at `β≈0.7`. On 16,068 LLM papers, this ** Alien sampler explores a 3.5–7× broader effective atom vocabulary than frontier LLM ideation** (Claude/Gemini), matches them on quality, and avoids the "motif collapse" failure mode where LLM baselines all converge to SAE/interpretability themes. Trained through-2024 models recover 2025 NeurIPS papers (e.g. Cosmos at rank 19 of 3.35M triples).

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is **not** the paper's training setup (autoregressive coherence transformer + contrastive availability dual-encoder). It is the *search objective*: a z-scored linear fusion of two scores that are **precomputed** — at inference time the sampler is pure ranking. The three reusable pieces are: (1) a **vocabulary of recombinable atoms** (anything discrete-embeddable); (2) a **coherence scorer** (any fn `&[atom] -> f32`); (3) an **availability scorer** (any fn `&[atom] -> f32` against a community/repertoire bank). The `Fβ` fusion + within-pool z-scoring is ~30 lines of generic code and is the open primitive worth shipping.

---

## 1. Paper Core Findings

### 1.1 Method (3 stages)

1. **Idea atoms.** Distill 16,068 NeurIPS/ICLR/ICML/NLP papers into ~82k "conceptual units", cluster via BGE+UMAP+HDBSCAN into a shared vocabulary of **273 reusable atoms** (methodological building blocks). Paper ↔ sparse atom-set mapping (mean 3.38 atoms/paper at 80% coverage).
2. **Coherence model `C(S)`.** Autoregressive transformer over atom sequences (273-token vocab, 42.8M params), trained on multi-serialized paper atom sequences. Score = length-normalized log-likelihood. Robust to atom ordering (within-set permutation std << coherent-vs-random gap).
3. **Availability model `A(S)`.** **Dual-encoder contrastive** — set encoder `ϕ` + author encoder `ψ`, shared atom embeddings. Trained with symmetric InfoNCE on `(author, held-out query subset, complement)` triples — disjoint split prevents solving via raw overlap. At inference, `A(S) = median over top-m authors of cos(ψ(R_i), ϕ(S))`. **Top-m median (not top-1)** is the key trick: prevents a single broad-author "generalist" from marking a direction as available.

### 1.2 Alien sampler

```
Fβ(S) = (1−β)·zC(S) + β·zU(S),    U(S) = −A(S),    β = 0.7
```

Within-pool z-scoring makes `β` comparable across candidate pools. `β=0` = coherence-only (most paper-like); `β=1` = least available (sacrifices plausibility). β=0.7 chosen by originality×coherence sweep.

### 1.3 Key empirical results

| Result | Number |
|---|---|
| Alien top-10 atom concentration | **34.3%** vs Claude 95.7%, Gemini 76.7% |
| Alien effective atom coverage | **72.5%** vs Claude 10.5%, Gemini 20.6% |
| Temporal recovery (β=0, top-1000) | 23 / 2477 held-out 2025 atom-sets, **31.1× random** |
| Cosmos (NeurIPS'25) rank at β=0.7 | **19 / 3,353,896** triples |
| Human overall quality | Alien 4.03 ≈ Claude 4.03 > Gemini 3.68 |
| Autoresearch rank percentile | Alien **0.633** > Claude 0.574 > Gemini 0.495 > Random 0.295 |
| **Motif collapse** (autoresearch SAE-interpretability reports) | Claude 2/10→**8/10**, Gemini 4/10→5/10, Alien **0/10→0/10** |

### 1.4 Critical ablation: dual-encoder vs density estimator

The paper explicitly compares the author-conditioned dual-encoder against an author-agnostic density estimator (same set-encoder, random negatives). The density estimator scores 0.499 AUC at separating paper-supported from author-only triplets — i.e. it **cannot tell community support from broad-individual bridges**. The dual-encoder scores 0.797. This is the load-bearing finding: **author/community conditioning is the whole point**, not just density estimation.

---

## 2. Distillation

### 2.1 What is genuinely transferable (modelless)

Strip the training setup. What remains:

- **`AlienSampler<V, C, A>`** — generic over a vocabulary `V`, coherence scorer `C: Fn(&[V::Item]) -> f32`, availability scorer `A: Fn(&[V::Item]) -> f32`. Method `rank(&candidates, beta) -> Vec<(score, idx)>` z-scores both within the pool and returns the linear fusion.
- **`AvailabilityBank`** trait — precomputed embeddings of community/repertoire items; method `availability(candidate_embedding) -> f32` returns `median_top_m` cosine. This is the dual-encoder distillation: the contrastive training happened offline; the **median-top-m** retrieval is the runtime primitive.
- **`Fβ` fusion** — ~10 lines, but the within-pool z-scoring + β parameterization is the reusable insight.

The atom-vocabulary construction (BGE+UMAP+HDBSCAN+LLM-cluster-naming) is **out of scope** for katgpt-rs — it's a one-time corpus-distillation step, not an inference primitive. Consumers bring their own vocabulary.

### 2.2 What is NOT transferable (training-side, redirect notes)

- Autoregressive coherence transformer training → riir-train if anyone wants the real paper reproduction.
- Contrastive dual-encoder training → riir-train.
- Paper-distillation / conceptual-unit extraction pipeline → not our domain.

### 2.3 Fusion — what novel combination does this paper × our stack produce?

This is the mandatory section per the research skill. Three closest cousins across the 5-repo corpus:

| Cousin | Repo | Mechanism shared with paper | Mechanism NOT shared |
|---|---|---|---|
| **CGSP** (R240 / P274 / riir-ai R126) | riir-ai | `CuriosityConjecturerPool` = atom vocab; Guide "relevance × elegance × non-redundancy" = coherence × availability; intermediate-difficulty filter = Fβ frontier; collapse detector = anti-motif-collapse | **No community availability model** — non-redundancy is local-only (vs NPC's own previous selections), not population-conditioned |
| **OPUS** (R089 / P129) | katgpt-rs | Boltzmann sampling + CountSketch redundancy penalty = scalar availability penalty; P2 GOAT proved ≥ Thompson on arm diversity | **Scalar penalty, not contrastive retrieval**; no top-m-median community aggregation; redundancy is intra-bandit, not cross-instance |
| **PersonalityWeightedComposition** (R276 / P297) | katgpt-rs | Sigmoid-gated composition of N layer-direction vectors into a behavior vector = "atom composition" | **Static composition** — no autoregressive coherence, no search, no availability |

**Fusion idea A (primary, GOAT-tier): Population-availability-aware CGSP.** Replace CGSP's local-only non-redundancy penalty with a `CommunityAvailabilityBank` populated from the local zone's NPC behavior emissions. Each NPC's Conjecturer samples candidate directions, the Guide scores coherence (existing), the bank scores availability against the *zone population* (new), Fβ at β≈0.7 selects alien directions. **Predicted gain:** reduced crowd motif-collapse — analogous to the paper's Claude→Alien shift from 95.7%→34.3% top-10 concentration. This is a refinement of an existing Super-GOAT (CGSP), not a new class.

**Fusion idea B (secondary, speculative — noted not planned): AlienSampler over our own `.research/` corpus.** Run the paper's pipeline on the 293+ research notes to surface "alien" research directions (coherent given our corpus but cognitively unavailable given what we've already shipped). This is a *process tool* for the research workflow itself, not a primitive in any of the 5 repos. Out of scope for Plan 311 — would be a separate meta-tool. Worth revisiting if motif-collapse becomes visible in our own research output (e.g. every recent plan touches LoRA routing or KV compression).

**Fusion idea C (tertiary, noted not planned): CWM × AlienSampler for emergent game-mode invention.** Each NPC induces a CWM (R145) of game rules; an alien sampler over the community's CWM pool could surface "alien rule combinations" — coherent rule sets no faction has tried. This would be a *new capability class* (NPCs inventing minigames), but it is too speculative to plan without first validating that population-availability (Fusion A) actually improves crowd diversity in our domain. Deferred.

---

## 3. Verdict

### GOAT — plan only, behind feature flag, benchmark before promotion.

**One-line reasoning per tier:**

| Tier | Reasoning |
|---|---|
| ~~Super-GOAT~~ | **Fails Q1** (no prior art): the principle is shipped in CGSP (riir-ai R126) + OPUS (P129) + PersonalityWeightedComposition (P297) + Dual-Pool CGSP (P282) + CLR personality divergence + Mind-Reading adaptive bandwidth + Policy Cascade. **Fails Q3** (selling point): "crowd doesn't collapse to monoculture" is already shipped as CLR/Mind-Reading/Policy-Cascade's selling point. The dual-encoder availability *mechanism* is novel; the *capability class* is not. |
| **GOAT** ✅ | Provable gain over the existing scalar-redundancy-penalty approach (OPUS) on population-diversity quality. The paper's evidence (95.7%→34.3% top-10 concentration, motif collapse 8/10→0/10) is exactly the gain we'd predict. Plan + feature flag + benchmark; promote if win, demote OPUS's local-only redundancy path if lose. |
| ~~Gain~~ | The dual-encoder is more than incremental — it's a structurally different availability signal (community vs local), and the Fβ fusion is composable across at least 4 existing systems (CGSP, CLR, Salience Tri-Gate, Mind-Reading). |
| ~~Pass~~ | The principle is highly relevant to our crowd-scale diversity stack; passing would lose a useful refinement. |

### Why this is NOT Super-GOAT (anti-overclaim audit)

The skill's R269 / R242 failure modes warn against defaulting to adapter routing when a latent reframing is available, and against overclaiming Super-GOAT when prior art ships under different vocabulary. Audit:

- **Latent reframing attempted:** Yes — re-cast as (a) HLA per-NPC state (atoms = 8-dim affect channels), (b) latent_functor (Fβ as a functor producing alien-ness scalar), (c) cgsp_runtime (population-availability as curiosity signal), (d) NeuronShard style_weights[64] (atoms = style channels), (e) LatCal (commitment of fusion scores for provenance). The strongest reframing is (c) — and it lands on CGSP, which is already shipped.
- **Vocabulary translation done:** Yes — paper "atoms" → codebase "Conjecturer pool direction vectors" / "personality layer directions" / "style_weights channels"; "availability" → "non-redundancy" / "CountSketch penalty" / "adaptive bandwidth"; "alien frontier" → "intermediate-difficulty filter" / "dual-pool E/X routing"; "motif collapse" → "policy monoculture" / "priority entropy collapse". Grepping these landed on the cousins above.
- **Honest call:** The latent-space reframing converges on CGSP, which already ships the principle. Adding a community-availability signal is a GOAT-tier refinement (better diversity quality, mech-analogous to paper's Claude→Alien improvement), not a Super-GOAT-tier new class.

---

## 4. Open primitive (target: `katgpt-rs/src/alien_sampler/`)

```rust
// Generic, no game semantics. MIT.

pub trait CoherenceScorer<V> {
    /// Score how coherent a candidate atom-set is. Higher = more coherent.
    fn coherence(&self, atoms: &[V]) -> f32;
}

pub trait AvailabilityScorer<V> {
    /// Score how available a candidate is to the reference community.
    /// Higher = MORE available (more community support). The sampler negates.
    fn availability(&self, atoms: &[V]) -> f32;
}

pub struct AlienSampler<'a, V, C: CoherenceScorer<V>, A: AvailabilityScorer<V>> {
    vocab: &'a [V],
    coherence: &'a C,
    availability: &'a A,
    beta: f32,         // 0.0 = coherence-only, 1.0 = availability-only. Paper default 0.7.
    top_m: usize,      // For MedianTopM availability implementations.
}

impl<'a, V, C, A> AlienSampler<'a, V, C, A>
where
    C: CoherenceScorer<V>,
    A: AvailabilityScorer<V>,
{
    /// Rank `candidates` by Fβ = (1−β)·zC + β·zU, U = −A.
    /// Within-pool z-score. Returns (score, candidate_idx) sorted desc.
    pub fn rank<'b>(
        &self,
        candidates: &'b [Vec<V>],
        scratch_c: &mut [f32],
        scratch_a: &mut [f32],
    ) -> Vec<(f32, usize)>
    where
        V: Clone + 'b,
    {
        // ... fill scratch_c, scratch_a, z-score both, fuse, sort desc
    }
}
```

Plus a `MedianTopMAvailability` struct that takes a `&[Vec<f32>]` community bank + `embedding_fn` and implements `AvailabilityScorer` via the paper's `median over top-m cosines` rule (the load-bearing community-aggregation trick from §1.4).

Feature flag: `alien_sampler`. Default-OFF until GOAT gate passes.

### 4.1 What stays private (riir-ai consumer)

The **`CommunityAvailabilityBank` populated from zone NPC behavior emissions** is private game-runtime IP — the embedding source, the emission filtering, the zone-scope parameterization, the bridge to CGSP's Conjecturer. That lands in `riir-ai/crates/riir-engine/src/cgsp_runtime/` as a new `alien_bridge.rs` module, gated by `cgsp_alien_availability`. The open `AlienSampler` is the math; the bank + bridge is the selling-point wiring.

### 4.2 What does NOT cross the sync boundary

Per AGENTS.md latent/raw rules:

| Quantity | Space | Synced? |
|---|---|---|
| `AlienSampler` Fβ score (per NPC, per cycle) | Latent | NO — local curiosity signal |
| `CommunityAvailabilityBank` embedding vectors | Latent | NO — per-zone aggregate, never per-NPC synced |
| Candidate atom-set / direction vector | Latent | NO — per-NPC Conjecturer output |
| Selected behavior emission (the *action* taken) | Raw | YES — existing TxDelta / KG-triple path, unchanged |
| `AlienPrioritySnapshot` (freeze/thaw) | Raw bytes (committed) | YES (via Cold tier) — BLAKE3-hashed personality checkpoint, content latent |

Bridge function: `latent → raw` is the existing action-emit path (unchanged). No new sync dependency introduced.

---

## 5. Validation protocol (GOAT gate, Plan 311)

The gate must prove the dual-encoder availability is *better* than the scalar-redundancy baseline (OPUS) on a population-diversity metric. Synthetic "motif collapse" scenario:

- **Setup:** 100 NPCs in a zone, each with a 16-dim Conjecturer pool. Without availability pressure, all NPCs' priority tables converge to the same top-3 directions (the "motif").
- **Baseline A (OPUS-style):** per-NPC local CountSketch redundancy penalty against own previous selections.
- **Baseline B (no availability):** coherence-only (β=0).
- **Treatment (AlienSampler):** `CommunityAvailabilityBank` populated from zone emissions, β=0.7.
- **Metric G1 (motif collapse):** top-10 direction concentration across the zone after 10k cycles. Treatment ≤ 50% of baseline A's concentration (paper shift: 95.7%→34.3% is ~36% of baseline).
- **Metric G2 (quality preservation):** mean Guide coherence score of selected directions. Treatment ≥ 90% of baseline B (coherence-only) — diversity must not destroy quality.
- **Metric G3 (perf):** `rank()` for 1000 candidates ≤ 500µs on SIMD, ≤ 50ms for 1M candidates on GPU. Plasma-tier for per-NPC per-tick use; warm-tier for batch zone-wide re-rank.
- **Metric G4 (latent/raw boundary):** zero `Vec<f32>` from the bank or per-NPC Fβ score crosses `SyncBlock` in a 1000-tick audit scan.

Promote `alien_sampler` to default if G1+G2+G3+G4 all pass. If G1 fails (no diversity gain over OPUS), the dual-encoder is not worth the complexity over scalar redundancy — demote, note honestly, move on.

---

## 6. References

- Paper: [arXiv:2603.01092v2](https://arxiv.org/abs/2603.01092)
- Code: https://github.com/alejandrohdez00/alien-space-of-science
- Closest cousins:
  - riir-ai `.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md` — CGSP Super-GOAT guide (Conjecturer pool + non-redundancy Guide)
  - katgpt-rs `.research/089_OPUS_Optimizer_Induced_Projected_Utility_Selection.md` — scalar redundancy penalty baseline
  - katgpt-rs `.research/276_Personality_Weighted_Latent_Layer_Composition.md` — sigmoid-gated atom composition
  - katgpt-rs `.research/281_Per_Tick_Salience_Tri_Gate_Speak_Silent_Delegate.md` — per-NPC per-tick decision primitive
- Plan: [katgpt-rs/.plans/311_alien_sampler_primitive.md](../.plans/311_alien_sampler_primitive.md)

---

## TL;DR

**Verdict: GOAT.** The paper's *principle* (coherence × (1−availability) frontier search for diversity) is shipped across 4+ existing systems (CGSP, OPUS, PersonalityWeightedComposition, CLR personality divergence, Mind-Reading). The paper's *mechanism* (dual-encoder contrastive availability + autoregressive coherence over discrete atoms + Fβ z-scored fusion) is novel, with the **author-conditioned dual-encoder + top-m-median aggregation** being the load-bearing structural innovation (paper proves density-estimator ablation fails this test). The transferable primitive for katgpt-rs is the generic `AlienSampler<V, C, A>` + `MedianTopMAvailability` (~150 LOC of pure ranking math). Private consumer in riir-ai: `cgsp_runtime/alien_bridge.rs` wiring a `CommunityAvailabilityBank` into CGSP's Conjecturer. **Not Super-GOAT** because the capability class (crowd-scale emergent diversity) is already shipped — this refines diversity quality via a better availability signal, predicting paper-style motif-collapse reduction (95.7%→34.3% top-10 concentration analog). Plan 311 behind feature flag `alien_sampler`; GOAT gate compares against OPUS local-redundancy baseline on a synthetic motif-collapse scenario.
