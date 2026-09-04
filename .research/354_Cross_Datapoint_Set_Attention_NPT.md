# Research 354: Cross-Datapoint Set Attention (NPT) — Sigmoid-Gated Permutation-Equivariant Cross-Entity Refinement

> **Source:** Kossen, Band, Lyle, Gomez, Rainforth, Gal — "Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning" (NeurIPS 2021), [arXiv:2106.02584](https://arxiv.org/pdf/2106.02584)
> **Date:** 2026-07-01
> **Status:** Active — Super-GOAT candidate (see verdict §3). Open primitive scoped.
> **Related Research:** 234 (DenseMesh — multi-node LLM comms, demoted), 126 (MoA — within-token activation mix), 278 (Engram — hash lookup, not set attention), 290 (Latent Field Steering — top-down broadcast, not joint inference), 232 (TJS identifiability — nonparametric theory), 303 (Transolver FUNCATTN — physics-set attention predecessor)
> **Cross-ref (riir-ai private guide):** [Research 167 — Crowd Joint Inference via Cross-NPC Set Attention](../../riir-ai/.research/167_crowd_joint_inference_cross_npc_set_attention_guide.md)
> **Cross-ref (riir-chain):** none — this primitive is latent-local, no sync boundary crosses; the Q/K/V projections can be BLAKE3-committed in `NeuronShard` but that is a neuron-db concern, not a chain concern.
> **Classification:** Public (katgpt-rs/MIT). The NPC selling point stays private in riir-ai/167.
> **Related-Paper (actionable fusion):** Wang, Su, Wang, Plotkin [PNAS 122(51) e2516535122 "Individual incentives that promote collective intelligence"] (SIAM News summary 2026-07-24) — Gain. The paper's "feedback payoff" $\pi_A = Y_A(Y - \hat{Y})$ (Lyapunov-convergent credit assignment that rewards individuals pulling the collective toward truth, even when their individual prediction is wrong) is **precisely the credit-assignment shape that converts this primitive's plain averaging into amplification** — closing the documented G8 collective-inference failure ("averaging cannot amplify detection"). CLR (P284, `(mean_m v_k,m)^M`, +78pp over majority) is the operational analog. Full distillation: [Research 469](469_collective_intelligence_payoff_schemes.md). PoC to close G8: `Issue 575`.
> **PASS-Redirects (synthesis):** DyTopo [arXiv:2602.06039 "DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning via Semantic Matching"] (Lu et al., Feb 2026) — PASS. The paper's per-round sparse directed communication graph via Q("need")·K("offer") semantic matching is **structurally this primitive** (cross-entity sigmoid-gated attention, hard-threshold sparsified). Compute-unit translation (R368 lesson): "LLM agent emits query+key" → "per-NPC cognition emits Q+K direction vectors" — substrate-independent, the mechanism is not new. The specific DyTopo deltas (hard binary threshold τ_edge, manager-set round goal, synchronization barrier, manager halting, topological-sort message ordering) each have a shipped analog: `SetAttentionConfig.top_k` (sparse threshold), Motivation Brain P392 + Salience Tri-Gate P303 (round-goal conditioning of Q/K emission), Sheaf-ADMM P407 z-update (synchronization barrier), Gain/Cost Loop Halting P304 (manager halting), proof_cert/chain.rs topological_sort + GDN tree P424 (message ordering). The "dynamic topology as interpretable coordination trace" selling point is already claimed by the α_ij matrix this primitive produces + the Crowd Joint Inference guide (riir-ai/167). CLR-weighted Set Attention (Plan 570, 2026-08-06) already closed the G8 collective-inference gap DyTopo's crowd-routing targets. No new primitive needed.
> **PASS-Redirects (synthesis):** Bronstein, Bruna, Cohen, Veličković [arXiv:2104.13478 "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"] — the GDL blueprint survey subsuming this primitive's symmetry axis (its Graphs chapter: GNN/Transformer/DeepSets all as permutation-equivariant architectures). Ships: Plan 354 set attention + Plan 570 CLR-weighted sibling + the riir-ai crowd set-attention guide (167) ARE the permutation-equivariant aggregation GDL unifies; the survey's Scale-Separation principle likewise ships as multi-scale V-cycle (Plan 413) + shard compaction/consolidation coarse-graining. No new mechanism → PASS.

---

## TL;DR

NPT's headline finding: deep-learning models conventionally predict each input
independently from their parameters; NPTs instead feed **the entire dataset as
input** and use **self-attention between datapoints** (Attention Between
Datapoints, ABD) so each prediction is a learned function of *other
datapoints*. NPTs beat CatBoost/XGBoost on 4/10 UCI tabular benchmarks, solve
cross-datapoint lookup and relational-reasoning tasks that no parametric model
can, and — critically — *learn end-to-end when to use the cross-datapoint path
and when to ignore it* (some datasets collapse back into purely parametric
prediction). The architecture alternates ABD (across rows) with Attention
Between Attributes (ABA, across columns), is **permutation-equivariant over the
datapoint axis by construction**, and trains via BERT-style stochastic masking
on both features and targets.

**Distilled for katgpt-rs (modelless, inference-time):**

The paper's value is end-to-end *training* of the Q/K/V projections — that
stays in riir-train. What survives into the public engine is the **inference-time
set-attention operator**: a sigmoid-gated (never softmax, per AGENTS.md §2),
permutation-equivariant cross-entity refinement kernel where the Q/K/V matrices
are **deterministically constructed** (from offline CS-rankings, identity,
or freeze/thaw-supplied direction vectors) rather than learned. The mechanism
reduces to:

```text
// Given N entity belief states h_i ∈ R^d (e.g. HLA latents of NPCs in a zone):
q_i = W_Q h_i              // query (DENSE → low-dim if CS-ranked)
k_j = W_K h_j              // key (DENSE → low-dim if CS-ranked)
α_ij = σ(q_i · k_j / √d)   // sigmoid gate (NEVER softmax) ∈ (0,1)
v_j = W_V h_j              // value (often identity in the modelless floor)
h_i' = h_i + Σ_j α_ij · v_j // permutation-equivariant cross-entity refinement
```

No training, no gradient descent. Q/K/V are **frozen artifacts** (BLAKE3-committed
in a `NeuronShard`, swapped atomically via freeze/thaw). The set-attention output
is still a per-entity latent — never decoded to tokens, never exposed at the sync
boundary as anything other than the existing 5 emotion scalars.

This is the missing **third quadrant** of crowd-scale NPC cognition. Existing
systems are *one-way broadcast* (Mind-Reading R133: A sends compressed belief to
B; Latent Field Steering R290: designer injects direction into each NPC) or
*per-NPC independent* (HLA evolves each NPC's belief from its own observations;
Latent Functor R123 lets NPC X predict Y's stance from X's own state). NPT-style
**cross-entity set attention** is the first mechanism in the codebase where each
NPC's belief is refined by *attention over the set of nearby NPCs' beliefs* —
genuine joint inference, not transmission.

---

## 1. Paper Core Findings

### 1.1 The challenge to parametric modeling

Standard deep learning predicts `p(y* | x*; θ)` — a function of params and a
single input. NPT predicts `p(y* | x*, D_train; θ)` — a function that also
depends on *other datapoints in the training set*. The paper formalises this as
a **general-purpose deep-learning architecture** that takes the entire dataset
`X ∈ R^{n×d}` as input and produces `n×d` predictions via learned interactions
between rows.

### 1.2 The three ingredients

1. **Dataset as input.** The whole dataset (features + targets + a binary mask
   matrix indicating which entries are observed vs. masked) is the model input.
   For large datasets, minibatching approximates "the whole dataset" by sampling
   a batch that contains both query points and context points.

2. **Self-attention between datapoints (ABD).** Each row of `X` is flattened to
   a single `R^h` vector (where `h = d·e`, the per-attribute embedding dim ×
   number of attributes), and multi-head self-attention is computed **across
   rows** — i.e., datapoint `i` attends to datapoint `j`. This is distinct from
   standard NLP attention, which attends across tokens *within* a sequence.

3. **Stochastic masking objective (BERT-style).** Random entries (both features
   and targets) are masked; the model predicts the masked values. Crucially,
   **stochastic target masking** during training means many training targets are
   *unmasked* to the model — so NPT learns to predict masked training targets by
   *attending to the unmasked targets of similar training rows*. This is what
   gives NPT its non-parametric lookup behavior.

### 1.3 ABD + ABA alternation

NPT alternates ABD (across rows) with Attention Between Attributes (ABA, the
familiar within-row token attention). ABA helps the model form good per-row
representations for the cross-row attention to operate on. Ablation (Table 5):
removing ABA costs 30% accuracy on Poker Hand (which requires complex
intra-row feature reasoning); on simple regression datasets ABA removal is
neutral. **Recommendation: default-on ABA, drop for pure cross-row tasks.**

### 1.4 Permutation-equivariance (proven, Appendix A)

NPT is provably equivariant over the datapoint axis: shuffling the input rows
shuffles the predictions correspondingly. This is non-trivial — multi-head
self-attention is *not* obviously equivariant because the `QKᵀ` product permutes
both rows and columns. The proof (Lemma 4) shows the column permutation cancels
out via the final `·V` multiplication. **This is the architectural property we
exploit**: cross-entity attention over a *set* of NPC belief states is well-defined
regardless of which NPCs happen to be in the zone.

### 1.5 Empirical evidence the cross-datapoint path is actually used

The paper's strongest evidence is the **data corruption experiment** (§4.3,
Table 2): at test time, for each target prediction, the model randomises every
other datapoint's attributes by independently permuting each column. If the model
genuinely relies on cross-datapoint attention, performance should collapse. It
does, dramatically, on most datasets (Protein RMSE drops by −52%; CIFAR-10
accuracy by −1.2%). But on some datasets (Forest Cover, Breast Cancer), corruption
has no effect — the model has *learned to ignore* the cross-datapoint path and
collapse back to parametric prediction. **This learned on/off switch is itself a
headline result**: the architecture doesn't force non-parametricity, it allows it.

### 1.6 What kind of interactions are learned? (§4.4)

The **data deletion experiment** iteratively removes datapoints that don't
significantly change a target prediction. Surviving (kept) datapoints have
significantly lower feature-space distance to the target than deleted ones
(Wilcoxon p ≈ 8.77·10⁻¹³⁰). Conclusion: **NPT learns k-NN-like behavior** —
attend to similar datapoints, average out their observation noise.

### 1.7 Limitations (§5, Appendix B.6)

- O(n²) attention cost — paper uses random minibatching at ~8000 datapoints.
- Peak GPU memory 19.18 GB on Higgs (11M datapoints) — much heavier than baselines.
- Paper explicitly notes **sparse attention approximations** (Performer,
  Longformer, linear attention) as future work to address scaling.

---

## 2. Distillation

### 2.1 What's training-only (→ riir-train)

| Paper mechanism | Why it needs training | Routing |
|---|---|---|
| End-to-end backprop on Q/K/V projections | The whole point — learn the attention pattern | riir-train |
| BERT-style stochastic masking objective | Self-supervised signal for the projections | riir-train |
| Stochastic target masking during training | The non-parametric lookup behavior emerges from this | riir-train |
| The λ cosine anneal between feature loss and target loss | Optimisation dynamics | riir-train |

If the user wants to train an NPT for a real tabular benchmark, that's a
riir-train LoRA-training task. katgpt-rs hosts the **inference-time operator
only**.

### 2.2 What's modelless (stays in katgpt-rs)

The pure inference kernel: given (a) a set of N entity state vectors `H ∈ R^{N×d}`,
(b) a frozen/committed `W_Q, W_K ∈ R^{d×d_q}`, (c) a frozen/committed `W_V ∈ R^{d×d}`
(often identity in the modelless floor), compute the permutation-equivariant
sigmoid-gated set-attention update. The Q/K/V matrices are supplied externally —
they are NOT learned inside katgpt-rs. They can be:

1. **Identity / uniform** — `W_Q = W_K = W_V = I`, the modelless floor (just
   sigmoid-weighted averaging of nearby entity states; a learned-free
   consensus primitive).
2. **CS-ranking-derived** — Project the Q/K onto the top-k dimensions identified
   by an offline Lasso (the existing CS-KV-Importance Probe, R247/P280). This
   is *exactly* what R133 Mind-Reading already produces offline; reuse it.
3. **Direction-vector-derived** — `W_Q` rows are existing emotion direction
   vectors (R144 Functional Emotions); `W_K` rows are functor direction vectors
   (R123 Latent Functor).
4. **Freeze/thaw-supplied** — `W_Q, W_K, W_V` are frozen into a `NeuronShard`
   (via `style_weights[64]` or a new `karc_shard`-style Pod), committed via
   BLAKE3, swapped atomically at runtime.

The **permutation-equivariance property** (paper's Lemma 4, Appendix A) survives
all four constructions — it's a property of the operator, not the matrices. This
is the architectural guarantee we get for free.

### 2.3 The sigmoid gate (AGENTS.md §2 compliance)

Paper uses softmax. **We use sigmoid**, per the global rule. Concrete substitution:

```text
// Paper (softmax, normalised, forces a probability simplex):
α_ij = softmax_j(q_i · k_j / √d)            // Σ_j α_ij = 1

// Ours (sigmoid, unnormalised, per-pair independent):
α_ij = σ(q_i · k_j / √d)                     // each α_ij ∈ (0,1) independently
```

The semantic shift is real and important: softmax forces *exactly one* datapoint
(or a strict distribution) to win; sigmoid allows *zero, one, or many* datapoints
to contribute. For NPC crowd inference this is the right shape — a guard may
attend to zero other guards (lonely patrol), one similar guard (paired patrol),
or many guards (formation). Softmax would force artificial competition.

### 2.4 Fusion opportunities (the Super-GOAT angle)

The Super-GOAT case rests on fusing this primitive with **existing systems** in
the codebase. None of these fusions are solo-novel; the combination is.

| Fuse with | What it gains | Mechanism |
|---|---|---|
| **HLA belief state** (R242, Pillar 5 substrate) | The set of latents to attend over | HLA's 8-dim per-NPC state is the natural `h_i`. Cross-NPC attention refines each NPC's belief using similar NPCs' beliefs. |
| **Mind-Reading CS-rankings** (R133, P311) | Deterministic offline construction of `W_Q` / `W_K` | R133 already produces a per-task-family ranking of which HLA dims carry signal; that ranking IS a low-rank Q/K projection. **Reuses offline work that already shipped.** |
| **Latent Functor** (R123, P303/317/318) | Direction vectors as Q/K projection axes | Functor `f` between two NPCs is a natural key for "how relevant is NPC j to NPC i". |
| **Crowd MCGS** (P298) | Joint-inference layer above the game tree | Currently each NPC plays independently in the crowd tree; cross-NPC set attention adds a per-tick belief-refinement step before the game-tree expansion. |
| **Zone Expert Bundles** (R020, P163) | The set definition — NPCs in the same zone | Bundle = the attention set. A zone with K NPCs runs K queries over K keys. |
| **CCE Crowd Batch** (P328, R143) | Joint payoff computation | Currently each NPC evaluates deviations against its own payoff table; cross-NPC attention lets NPCs *borrow* each other's payoff estimates for unseen states. |
| **NeuronShard freeze/thaw** (riir-neuron-db) | Persistence + commitment of the projections | `W_Q, W_K, W_V` frozen into a `NeuronShard` Pod, BLAKE3-committed, atomic Arc swap at zone-load time. |
| **Cognitive Branches** (P338, R161) | Branch-local set attention | Each non-interference branch holds its own Q/K/V; set attention is computed per-branch, branches stay orthogonal. |

**Closest cousins in the corpus (read before claiming novelty):**
- **R234 DenseMesh** (Plan 266) — multi-LLM-node dense comms, **DEMOTED** (Gate 2 failed: 0/1000 wins on Bomber). The mechanism is "LLMs as nodes, LoRA edges as comms" — fundamentally different from "NPC beliefs as datapoints, set attention as the operator". DenseMesh relies on *trained edges*; NPT-style set attention uses *frozen* Q/K/V.
- **R126 MoA** (Plan 158) — token-adaptive activation mixing within a single FFN. Not set attention.
- **R278 Engram** (Plan 299) — hash-addressed static memory lookup. O(1) per-token lookup, not O(N) cross-entity attention. Complementary, not overlapping.
- **R290 Latent Field Steering** (Plan 309) — top-down *broadcast*: designer injects a direction vector into every NPC's latent. NPT is *peer-to-peer*: each NPC attends over its peers' latents. Different direction of information flow.
- **R303 Transolver / FUNCATTN** — physics-set attention, the predecessor lineage. Noted as crowd-scale reframing candidate in R303 §2.3. Same family of mechanism, different application (physics nodes vs. NPC beliefs).

> **Cross-ref to Plan 440 (Lifelong LaCAM LLLG) — the physical-space twin.** Set Attention (this note) is the **latent-domain** crowd primitive: each NPC attends over peers' HLA beliefs in latent space, sigmoid-gated, permutation-equivariant, zero raw sync crosses. Plan 440's Lifelong LaCAM LLLG is the **physical-domain** twin: each agent avoids peers' cells at `t+1` via PIBT over per-agent space-time A* guidance, operating entirely on raw `MapPos` that MUST stay raw/synced/replayed (anti-cheat, deterministic replay, quorum). The two compose — the private fusion (riir-ai/489) runs LLLG for physical collision-free motion and Set Attention for latent belief refinement in the same tick, bridged by the existing 5-scalar emotion bridge (`compute_animal_emotions()`). Set Attention's `HlaProjectedGuidance` seam (Plan 440 T3.1) is the literal fuse point: `α_i = α_base · (1 + β · σ(dot(HLA_i, D_frustration)))` modulates an agent's LLLG collision penalty using the Set-Attention-refined HLA state.

**The fusion idea (write into §Distillation as a Fusion subsection):**

> NPT × Mind-Reading × Latent Functor = **Crowd Joint Inference**: each NPC's
> HLA belief is refined by sigmoid-gated attention over the set of NPCs in its
> zone, where the Q projection is the Mind-Reading CS-ranking, the K projection
> is the Latent Functor direction set, and the V projection is identity. None
> of these alone produces joint inference; the fusion does. This is the private
> Super-GOAT selling point — see riir-ai/167.

### 2.5 Latent vs raw boundary (AGENTS.md sync rule)

| Data | Space | Synced? | Reason |
|---|---|---|---|
| Per-NPC HLA latent `h_i` | Latent (8-dim affective) | NO (already the case) | Per-NPC subjective belief, fog-of-war gated |
| Cross-NPC attention output `h_i'` | Latent (8-dim affective, refined) | NO | Same category as input — still per-NPC subjective belief |
| Q/K/V projection matrices | Latent (frozen artifacts) | NO (BLAKE3-committed in NeuronShard at freeze/thaw) | Loaded at zone-init, never sent over wire |
| 5 emotion scalars (existing bridge) | Raw | YES (existing) | Unchanged — `compute_animal_emotions()` bridge still produces the synced scalars |
| `α_ij` attention weight | Latent | NO | Diagnostic only; never synced |

**Critical: no new raw data crosses the sync boundary.** The cross-NPC attention
is purely a local latent-state refinement. The existing 5-scalar bridge
(valence/arousal/desperation/calm/fear) is the only thing that crosses, exactly
as before. Anti-cheat, deterministic replay, quorum sync — all unaffected.

### 2.6 The modelless floor (sanity check)

Even with all of `W_Q = W_K = W_V = I` (the dumbest possible construction), the
primitive is meaningful: it computes a sigmoid-weighted consensus of nearby NPCs'
HLA states. Two guards with similar belief states reinforce each other; two
guards with dissimilar states don't interfere. This is a defensible baseline.

**GOAT gate against the floor:** any non-identity Q/K/V construction must beat
the identity-floor on (a) prediction quality (cosine similarity of refined
belief to ground-truth NPC action ranking) and (b) latency (the floor is O(N·d²)
matmuls; CS-ranking-derived low-rank Q/K should be O(N·k²) where k ≪ d).

---

## 3. Verdict

**Super-GOAT.** One-line reasoning: **NPT-style cross-entity set attention is
the missing third quadrant of crowd-scale NPC cognition — every existing crowd
system in the codebase is one-way broadcast (Mind-Reading, Latent Field Steering)
or per-NPC independent (HLA, Latent Functor, CCE Crowd). Cross-entity set
attention is the first mechanism where each NPC's belief is refined by attention
over its peers' beliefs, and the fusion with HLA + Mind-Reading CS-rankings +
Latent Functor direction vectors produces a new capability class ("collective
inference") that none of them has alone.**

### Novelty gate (all 4 YES)

1. **No prior art?** YES. Vocabulary-translated grep across all 5 repos
   (`.research/` + `.plans/` + `.docs/` for intent; `src/` + `crates/` for shipped
   code) with BOTH paper vocabulary (`NPT`, `ABD`, `non-parametric transformer`,
   `attention between datapoints`, `set transformer`, `cross-datapoint attention`,
   `Kossen`) AND codebase vocabulary (`cross-entity attention`, `inter-NPC
   attention`, `set attention`, `permutation equivariant`, `cross-attend`,
   `crowd attention`, `crowd coherence`, `crowd inference`) returns ZERO direct
   hits. Closest cousins (R133 Mind-Reading, R123 Latent Functor, R290 Latent
   Field Steering, R234 DenseMesh, R278 Engram, R303 Transolver) are all
   *transmission* or *per-NPC* — none implements peer-to-peer set attention for
   joint inference. **Read the TL;DRs of each cousin before claiming novelty**;
   the gap is real.

2. **New class of behavior?** YES. The codebase has *transmission* (A→B), *broadcast*
   (designer→all), and *independent per-NPC prediction*. It does NOT have *joint
   inference via attention* (each NPC's belief is a function of attention over
   the set of nearby NPCs' beliefs). This is a fourth quadrant — a capability
   no incumbent has.

3. **Product selling point?** YES. "A guard patrol collectively infers a threat
   pattern even when no single guard saw the whole picture; a market crowd
   settles on a fair price via mutual attention over each other's estimates;
   a herd of prey animals collectively detects a predator from each member's
   partial observations. None of this requires per-NPC tuning — the attention
   weights are derived offline from existing CS-rankings and direction vectors."

4. **Force multiplier?** YES. Connects 6+ existing systems (HLA, Mind-Reading,
   Latent Functor, Crowd MCGS, Zone Expert Bundles, CCE Crowd Batch) + the
   freeze/thaw persistence substrate (NeuronShard). The selling point is the
   *fusion*, not any single piece.

### Mandatory outputs (this session)

Per the Super-GOAT rule ("no candidate escape hatch"), committing all 4 YES
triggers:

1. **Open primitive** → this research note (`katgpt-rs/.research/354_*.md`) +
   `katgpt-rs/.plans/354_cross_datapoint_set_attention_primitive.md` (open math,
   no game semantics).
2. **Architectural GUIDE** → `riir-ai/.research/167_crowd_joint_inference_cross_npc_set_attention_guide.md`
   (the private selling-point doc — TL;DR with commercial value, distilled
   primitive, connection map, latent vs raw boundary, what stays private vs open,
   validation protocol G1–Gn, implementation priority P0–P3).
3. **Private runtime plan** → `riir-ai/.plans/355_crowd_joint_inference_runtime.md`
   (HLA wiring, zone-bundle set construction, Mind-Reading CS-ranking reuse,
   crowd-scale G5/G6 gates).

### Tiers (high → low) — applied

| Tier | Criteria | Routing |
|---|---|---|
| **Super-GOAT** ✅ | Novel mechanism (no prior art) + new capability class (collective inference) + product selling point (NPCs that collectively reason) + force multiplier (6+ systems) | Open primitive → katgpt-rs Plan 354. **Architectural guide → riir-ai/.research/167**. Plans → both repos. |
| GOAT | (would be: provable gain over existing approach, no new capability class) | n/a — cleared the Super-GOAT bar |
| Gain | (would be: incremental, useful but not headline) | n/a |
| Pass | (would be: training-only or not relevant) | n/a |

---

## 4. What ships where (5-repo discipline)

| Repo | What lands | Why |
|---|---|---|
| `katgpt-rs` (public MIT) | Open primitive: `set_sigmoid_attention_into(query, key, value, output, scratch)` + `SetAttentionConfig` + permutation-equivariance tests + identity-floor benchmark | Generic math, no game semantics. Adoptable by anyone. |
| `riir-ai` (private) | Runtime guide (R167) + plan (P355): HLA wiring, zone-bundle set construction, Mind-Reading CS-ranking reuse, crowd G5/G6 gates, Bevy demo | The selling point lives here — NPC crowd joint inference. |
| `riir-chain` (private) | None directly | The primitive is latent-local; no sync-boundary crosses. (If a future variant wants to commit Q/K/V via LatCal fixed-point, that's a separate riir-chain plan.) |
| `riir-neuron-db` (private) | Optional: `W_Q/W_K/W_V` frozen into a `NeuronShard` Pod via existing `style_weights[64]` or a new dedicated slot; BLAKE3 commitment reuses existing `freeze.rs`. | Persistence of the projections. Small change, reuses existing substrate. |
| `riir-train` (private) | Out of scope for this workflow. If a future variant wants to *train* Q/K/V via the paper's masking objective, that's a riir-train LoRA-training plan. | The modelless primitive is the priority; training is a follow-up. |

---

## 5. Validation protocol (GOAT gate for the open primitive)

The open primitive must clear these gates before promoting to default-on. The
guide (riir-ai/167) extends with crowd-scale runtime gates G5/G6.

| Gate | Criterion | Target | Failure → demote to |
|---|---|---|---|
| **G1** (permutation equivariance) | Output is identical (bit-exact for f32) under any permutation of input rows | bit-exact | Gain (broken operator) |
| **G2** (identity-floor meaningfulness) | Identity `W_Q=W_K=W_V=I` on a synthetic 2-cluster HLA set produces visibly different per-cluster means | cluster-mean separation > 3σ | Gain (operator is noise) |
| **G3** (latency) | `set_sigmoid_attention_into` on N=64 entities × d=8 HLA dims | < 5 µs (well within 20Hz tick budget = 50ms) | Gain (too slow for plasma tier) |
| **G4** (zero-alloc) | No allocation in steady state (pre-allocated scratch) | 0 allocs/call | Gain (allocates) |
| **G5** (sigmoid-not-softmax correctness) | With σ gate, an NPC may attend to 0 peers (lonely patrol) and the output equals the input | bit-exact when all `α_ij < 0.5` and the residual is gated to 0 | Gain (forces spurious contribution) |
| **G6** (CS-ranking reuse — fusion gate) | Plugging R133's CS-ranking as `W_Q` produces different output than identity, on a recorded NPC crowd trace | cosine sim < 0.95 vs identity floor | GOAT-not-Super-GOAT (fusion doesn't add value) |

The Super-GOAT bar is G1–G5 (the primitive is sound) + G6 (the fusion with
existing systems adds value). Crowd-scale emergent-behavior gates (long-horizon
stability, crowd coherence under noise) live in the riir-ai guide.

---

## 6. Open questions / risks

1. **The "weak" Super-GOAT concern.** The mechanism (attention) is well-known;
   the novelty is the *application* (NPC crowd joint inference) + the
   *modelless construction* (deterministic Q/K/V from existing offline work) +
   the *crowd-scale emergent behavior* angle. This is a legitimate Super-GOAT
   per the criterion "new capability class" — but the underlying math is not
   novel. Be honest in the guide: the moat is the *application + fusion*, not
   the operator.

2. **NPT's headline results rely on training.** The paper's UCI/CIFAR-10 wins
   come from end-to-end backprop. The modelless version with deterministic Q/K/V
   is a *strict subset* of NPT's capability — it cannot learn novel attention
   patterns, only apply pre-computed ones. The selling point must therefore
   ground in "we already have the offline CS-rankings and direction vectors"
   (R133, R123), NOT in "NPT beats XGBoost". **If the user wants the full NPT,
   that's riir-train.**

3. **O(N²) scaling.** The paper notes this explicitly. For NPC crowds at 20Hz,
   N=100 NPCs/zone is plausible; N=1000 needs sparse attention approximations
   (top-k sigmoid, locality-sensitive hashing). The open primitive should ship
   with a `top_k` config from day 1.

4. **Cousin-density risk in `riir-ai/.research/`.** The per-NPC runtime corpus
   is saturated (R123, R126, R128, R133, R146, R147, R148, R152, R155, R156,
   R158, R159, R160, R161, R163, R165, R166). Before adding R167, confirm the
   "joint inference via set attention" framing is NOT already covered by any of
   them under different vocabulary. (Checked during this session: none of them
   frames it this way — they're all one-way or per-NPC.)

---

## TL;DR

NPT (Kossen et al. NeurIPS 2021) introduces self-attention *between datapoints*
(ABD) — each prediction is a learned function of attention over *other*
datapoints, not just the input's own features. The paper is fundamentally a
training paper (end-to-end backprop on Q/K/V via BERT-style masking), so the
training know-how routes to riir-train. **What survives into katgpt-rs is the
inference-time operator: a sigmoid-gated (never softmax, per AGENTS.md),
permutation-equivariant cross-entity set-attention kernel where Q/K/V are
deterministically constructed (identity floor, CS-ranking-derived from R133
Mind-Reading, direction-vector-derived from R123 Latent Functor, or
freeze/thaw-supplied from a NeuronShard).** Novelty gate clears all 4 (no prior
art across 5 repos with vocabulary translation; new capability class — joint
inference, distinct from one-way broadcast or per-NPC independent prediction;
selling point — NPCs that collectively reason; force multiplier — connects 6+
existing systems). **Verdict: Super-GOAT.** Mandatory outputs created in this
session: open primitive (this note + katgpt-rs Plan 354), private guide
(riir-ai/167), private runtime plan (riir-ai/355). Latent-vs-raw boundary
respected (no new sync-boundary crosses; only existing 5 emotion scalars sync).
The moat is the application + fusion, not the operator itself — be honest about
this in the guide.
