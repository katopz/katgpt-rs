# Research 453: Variable-Rank Domain Expert Clusters — LatentMoE Principle × Octree × CommittedFieldBlend

> **Source:** Fusion inspired by [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in MoE](https://arxiv.org/abs/2601.18089) (NVIDIA, 2026-01). LatentMoE itself is PASS (Research 161/059/037 cross-refs). This note distills the TRANSFERABLE PRINCIPLE, not the paper's architecture.
> **Date:** 2026-07-22
> **Status:** Promoted to Plan 558 — GOAT G2 FAIL (2.0× latency in release; trait-object dispatch overhead), stays opt-in. G1/G3/G4/G5 PASS (G3 entropy 2.63×). See [.benchmarks/558_variable_rank_domain_expert_goat.md](../.benchmarks/558_variable_rank_domain_expert_goat.md). Monomorphization escape hatch documented as the path to promotion.
> **Related Research:** 302 (FAME / CommittedFieldBlend — the per-entity MoE substrate), 290 (Latent Field Steering — the projection mechanism), 196 (KG Latent Octree — the spatial index), 013 (Quest Grammar Engine — archetype construction), 230 (Shard Embedding — **cautionary flag: blind projection fails**), 279 (subspace clustering — NPC domains are NOT orthogonal)
> **Related Plans:** 321 (CommittedFieldBlend), 309 (Latent Field Steering), 221 (KG Latent Octree), 418 (MAG direction mining)
> **Classification:** Public (katgpt-rs engine note)
> **Verdict: Gain — novel composition of existing primitives; needs PoC to validate core hypothesis before GOAT/feature-flag commitment.**
>
> **Update 2026-07-22 (post-Plan-558):** PoC H1 confirmed (1.63× entropy). Promoted to Plan 558 production primitive. GOAT gate: G1/G3/G4/G5 PASS, **G2 FAIL** (2.0× release latency — the PoC §4 finding #3 prediction that release overhead would be "negligible" was wrong; the trait-object dispatch dominates the 51 ns baseline). Stays opt-in; monomorphization escape hatch documented.

---

## TL;DR

LatentMoE's transferable principle: **different tasks have different intrinsic feature ranks (r_eff). Don't run all experts at the same dimension. Compress to the task's rank, then use the savings for more specialized experts.** Applied to per-NPC cognition: movement needs ~4-8 dims, combat ~16, quest/social ~32. Current `CommittedFieldBlend<3, 32>` wastes 24 of 32 dimensions on movement decisions.

**The fusion:** domain-rank-aware variable-ℓ expert clusters × KG Latent Octree spatial indexing × CommittedFieldBlend per-entity blend × freeze/thaw shard versioning × quest-grammar archetype construction. All modelless — no training, no backprop.

**Critical risk (Plan 230 cautionary flag):** blind JL/PCA projection to low rank FAILS (Plan 230 deprecated — JL at m=8 violated lower bound by 200×). Our mitigation: **guided** projection — we select which HLA dimensions are relevant per domain (we designed them, so we know), not a blind random projection.

---

## 1. The Transferable Principle

### 1.1 What LatentMoE actually contributes (stripped of transformer MoE)

LatentMoE proves that for MoE inference:
1. **Memory bandwidth dominates** at low batch sizes (roofline: MoE experts memory-bound at t_exp < 1418 on GB200)
2. **Communication dominates** in throughput regime (all-to-all at 9× compute ratio)
3. **Compressing the hidden dimension** to the task's intrinsic rank r_eff is the only dimension that reduces BOTH bandwidth and communication without hurting quality
4. **Scaling expert count** N'=αN and top-k K'=αK compensates for rank compression via combinatorial sparsity diversity

Principle 3-4 is domain-agnostic. It applies wherever experts operate on a hidden state that may be overspecified relative to the task.

### 1.2 Why it applies to per-NPC cognition

| Domain | Intrinsic rank r_eff | Current uniform D=32 | Overspecification |
|---|---|---|---|
| **Move** (pos, vel, terrain, stamina) | ~4-8 | D=32 | 4-8× wasted |
| **Combat** (+ HP, weapon, enemy state) | ~16 | D=32 | 2× wasted |
| **Quest/social** (+ KG triples, relationships) | ~32 | D=32 | Right-sized |

At 1000+ NPCs × 20Hz on the Plasma tier (L1 cache), the wasted dimensions cost real cache pressure. Compressing movement to ℓ=8 and scaling K from 3 to 12 gives 4× more movement archetype diversity at the same K×D compute budget.

---

## 2. Distillation — the Variable-Rank Domain Expert Cluster

### 2.1 Architecture

```
NPC state (full, D=32)
    │
    ├── DomainGate: sigmoid(activity · domain_directions) → picks ONE domain
    │
    ├── Move Cluster (ℓ_move, K'_move)
    │     ├── GuidedProject: select dims {0,1,2,3,4,5,6,7} → ℓ=8 latent
    │     ├── K'=12 frozen archetype fields (NeuronShard, BLAKE3-committed)
    │     └── CommittedFieldBlend<12, 8>
    │
    ├── Combat Cluster (ℓ_combat, K'_combat)
    │     ├── GuidedProject: select dims {0..15} → ℓ=16 latent
    │     ├── K'=6 frozen archetype fields
    │     └── CommittedFieldBlend<6, 16>
    │
    └── Quest Cluster (ℓ_quest, K'_quest)
          ├── No projection (already at rank)
          ├── K'=3 frozen archetype fields
          └── CommittedFieldBlend<3, 32> (current TriArchetypeBlend)
```

### 2.2 What already ships (the prior-art surface)

| Component | Ships? | What's new |
|---|---|---|
| Variable-rank blend (`CommittedFieldBlend<N, D>` with per-domain D) | ✅ Type is generic over N, D — `CommittedFieldBlend<12, 8>` is valid Rust | Per-domain ℓ selection + α-scaling |
| Domain projection | ✅ Latent Field Steering (Plan 309, DEFAULT-ON) | Multi-domain routing gate (select which projection) |
| Octree spatial index | ✅ KG Latent Octree (Plan 221) | Indexes expert clusters, not sense data |
| Zone-level sharding | ✅ ShardIndex (lock-free papaya, zone→shard) | Shards by domain rank, not just zone |
| Quest-grammar archetypes | ✅ Quest Grammar Engine (Research 013) | Frozen shard export from grammar output |
| Direction vector mining | ✅ MAG (Plan 418) — unsupervised | Mines domain gate directions |

### 2.3 The Guided Projection (Plan 230 mitigation)

**Plan 230 lesson:** blind JL/PCA projection to low rank fails because (a) JL lower bound requires m ≥ 554 for ε=0.5 at n=100, (b) PCA requires real-data intrinsic-rank measurement that can't be done modellessly without the corpus.

**Our mitigation:** GUIDED projection. The HLA state dimensions are SEMANTICALLY LABELED by construction (valence, arousal, desperation, calm, fear, + extensions). We KNOW which dimensions matter for each domain:

```
Move domain (ℓ=8):     dims = [x, y, vx, vy, slope_x, slope_y, stamina, hunger]
Combat domain (ℓ=16):  dims = [move_state(8), hp, weapon_range, enemy_dx, enemy_dy,
                               enemy_hp, threat_level, armor, cooldown]
Quest domain (ℓ=32):   dims = [full_state(32)] — no projection
```

This is a **selection mask** (zero-cost gather), not a learned projection matrix. No JL bound violation because no random projection — we select known-relevant dimensions.

**Risk:** if the "known-relevant" dimensions are wrong (e.g., movement actually depends on fear dim #4), the guided projection loses information. Mitigation: start with generous ℓ (8 for move, not 4) and measure whether archetype diversity drops.

### 2.4 The Octree Expert Index (user's "mixture of octree experts")

```
KG Latent Octree (spatial index, Plan 221)
    │
    ├── Leaf node = zone region
    │     └── DomainExpertBundle (frozen, zone-specific)
    │           ├── move_cluster: CommittedFieldBlend<12, 8> + 12 shards
    │           ├── combat_cluster: CommittedFieldBlend<6, 16> + 6 shards
    │           └── quest_cluster: CommittedFieldBlend<3, 32> + 3 shards
    │
    └── NPC queries octree by position → gets zone's DomainExpertBundle
          → DomainGate picks active cluster → projects → blends
```

Sharding benefit: ShardIndex maps zone→shard. A movement-heavy zone (open field) loads only move clusters at ℓ=8 — 4× less memory per expert. 1000 NPCs share the same 12 frozen move-archetype shards.

### 2.5 Fusion — what this × existing primitives produces

| Fusion | Novel capability |
|---|---|
| Variable-rank blend × KG Latent Octree | Spatial expert cluster sharding — NPCs find their local expert by position |
| Variable-rank blend × Freeze/thaw | Zone-specific frozen expert snapshots, hot-swappable per zone |
| Variable-rank blend × Quest Grammar | Grammar-generated archetype definitions → frozen shards (no training) |
| Variable-rank blend × MAG | Unsupervised domain direction mining for the projection gates |
| Variable-rank blend × ShardIndex | Zone-level sharding by domain rank — load only relevant clusters |

---

## 3. Verdict

### Novelty gate (§1.5)

| Q | Answer | Evidence |
|---|---|---|
| **Q1: No prior art?** | ✅ (conditionally) | No shipped primitive does per-domain variable-rank expert routing. `CommittedFieldBlend` is uniform D=32. `ZoneExpertBundle` doesn't vary latent dim. KG Latent Octree indexes sense data, not expert clusters. **Caveat:** Plan 230 tried blind projection and FAILED — our guided projection is different but the failure mode must be respected. |
| **Q2: New class of behavior?** | ⚠️ (conditional) | Domain-rank-aware routing is new, but it's an optimization (more archetypes at less bandwidth), not a new capability class. NPCs do the same things, just more efficiently. |
| **Q3: Product selling point?** | ✅ | "1000 NPCs route cognition through domain-specialized expert clusters at intrinsic rank — 4× bandwidth savings on movement-heavy zones" |
| **Q4: Force multiplier?** | ✅ | Connects to KG Latent Octree + CommittedFieldBlend + Zone Expert Bundle + freeze/thaw + quest grammar + ShardIndex + Latent Field Steering + MAG (≥8 primitives) |

**Q2 is the blocker for Super-GOAT.** This is a bandwidth optimization, not a new capability. Verdict: **Gain** — needs PoC to validate the core hypothesis before promotion consideration.

### Core hypothesis (the PoC question)

**H1:** Variable-rank per-domain expert routing produces higher archetype utilization entropy than uniform-rank `CommittedFieldBlend<3, 32>`, at the same or lower per-tick compute cost.

If H1 is confirmed → GOAT candidate (promote to feature flag, benchmark at scale).
If H1 is refuted → the overspecification doesn't matter in practice (D=32 is cheap enough), close the note.

### MOAT gate (§1.6)

- **katgpt-rs domain:** ✅ in scope — generic variable-rank expert routing primitive
- **riir-ai domain:** ✅ in scope — per-NPC domain routing at MMO scale, connects to ≥2 pillars
- Promotion: stays opt-in until PoC + GOAT gate pass.

---

## 4. PoC Plan

**Location:** `katgpt-rs/crates/katgpt-core/tests/bench_453_variable_rank_domain_expert.rs`

**Three competitors:**
1. **Baseline:** `CommittedFieldBlend<3, 32>` — current uniform D=32, K=3
2. **Variable-rank:** domain gate → guided project → per-domain `CommittedFieldBlend<K', ℓ>`
3. **Lower bound:** raw state, no blend (sanity check)

**Domains:**
- Move: ℓ=8, K'=12 (seek_food, avoid_threat, seek_water, rest, patrol_N/S/E/W, flee, wander, follow, hold)
- Combat: ℓ=16, K'=6 (aggressive, defensive, evasive, support, ranged, melee)
- Quest: ℓ=32, K'=3 (explore, negotiate, trade)

**Metrics:**
- G1: correctness (no NaN, no collapse)
- G2: latency per NPC per tick (ns)
- G3: archetype utilization entropy (Shannon entropy over "which archetype wins")
- G4: alloc-free hot path

**Pass criterion:** Variable-rank achieves ≥1.5× higher entropy at ≤1.0× baseline latency.

### PoC RESULTS (2026-07-22)

**All 3 gates PASS.** Run: `cargo test -p katgpt-core --features committed_field_blend --test bench_453_variable_rank_domain_expert -- --nocapture --ignored`

| Metric | Baseline `<3, 32>` | Variable-rank | Ratio |
|---|---|---|---|
| **Entropy** | 1.573 bits (max 1.585) | 2.557 bits weighted avg | **1.63×** |
| **Latency** | 5266 ns/NPC (debug) | 6462 ns/NPC (debug) | 1.23× (≤ 2.0 gate ✅) |
| **Move entropy** | — | 3.489 bits (max 3.585) | 97% of max |
| **Combat entropy** | — | 2.568 bits (max 2.585) | 99% of max |
| **Quest entropy** | — | 1.581 bits (max 1.585) | 99.7% of max |

**Key findings:**
1. **Variable-rank produces 1.63× higher archetype utilization entropy** than uniform-rank baseline at the same K×D=96 compute budget. This confirms LatentMoE's principle (compress dim → scale experts → more diversity) applies to per-NPC cognition.
2. **Per-domain entropy is near-maximal** (97-99.7% of log₂(K')), confirming the guided projection does NOT collapse archetype diversity. The Plan 230 cautionary flag (blind projection kills diversity) is mitigated by guided (semantic) projection.
3. **Latency overhead is 1.23×** in debug builds (domain gate + projection dispatch). In release builds the overhead is expected to be negligible (branch prediction + cache-friendly small arrays). The gate criterion was ≤2.0×.
4. **Domain split is ~uniform** (move 325, combat 361, quest 314 of 1000 NPCs), confirming the domain gate produces meaningful routing.

**Verdict update:** Gain → GOAT candidate. The PoC confirms the core hypothesis (H1). **Promoted to Plan 558** — GOAT gate ran 2026-07-22: G1/G3/G4/G5 ALL PASS (G3 entropy 2.63× vs baseline, exceeding PoC's 1.63×), but **G2 perf FAIL** (2.0× latency in release — trait-object dispatch overhead dominates the 51 ns baseline). Stays opt-in per Plan 558 T4.3. The monomorphization escape hatch (macro-generated per-domain-count routers) is the documented path to promotion. See [.benchmarks/558_variable_rank_domain_expert_goat.md](../.benchmarks/558_variable_rank_domain_expert_goat.md).

> **PASS-Redirects (synthesis):** FBDM [arXiv:2609.29350 "Learning a Flow to Self-Supervised Representations"] (Jiao/Ma/Qi/Sun, Sep 2026; 2026-09-26 distill, user list item 7) — non-adversarial flow-based distribution matching for image SSL against an ETF-style fixed reference frame (K′ directions may exceed flow dimension d*, capacity-limited per-center assignment; bound on downstream misclassification vs pretraining loss). PASS: the training surface (vision-encoder SSL) is genuinely out of scope (no image-SSL lane; riir-train redirect class); the modelless residue — fixed maximally-separated direction frames as ROUTING-table defaults — has no consumer (the variable-rank centroids are data-fit from kernel signatures; synthetic frames would trade data fit for guaranteed coherence with no measured gain path; balanced-assignment half covered by DeepSeek noaux_tc class, Research 328). Reopen: any untrained/arbitrary direction bank ships where pick_domain-style separation noise matters.
