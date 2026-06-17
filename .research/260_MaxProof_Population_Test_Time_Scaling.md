# Research 260: MaxProof — Population-Level Test-Time Scaling for Proof/Reasoning

> **Source:** [MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling](https://arxiv.org/abs/2606.13473) — Jiacheng Chen, Xinyu Zhang, et al. (MiniMax / CUHK / Fudan / PKU / Tsinghua), 2026-06-11
> **Date:** 2026-06-17
> **Status:** Active — GOAT verdict
> **Classification:** Public (katgpt-rs modelless test-time scaling)
> **Related Research:** 093 (Committee Search — Copeland tournament), 239 (MLEvolve — Base/Stepwise/Diff modes ≈ PATCH/REWRITE), 240 (CGSP — curiosity-guided self-play), 040 (BT pairwise ranking), 088 (AlphaProof Nexus), 170 (LEAP blueprint DAG), 182 (STV self-trained verification), 255 (VibeThinker CLR)
> **Related Plans:** TBD (pending GOAT gate)
> **Training redirect:** Proof Expert RL training (CISPO + defense-in-depth verifier), Verifier Expert SFT, Fixer Expert rejection-sampling FT → riir-train. This note distills only the inference-time MaxProof loop.

---

## TL;DR

MaxProof is a **population-level test-time scaling framework** that converts best@K into stable pass@1 via an evolution-inspired search loop: N=32 initial candidates → conservative verifier scoring (min across K_verify=4 samples) → R=10 rounds of dual PATCH (exploitation) / REWRITE (exploration) refinement → pairwise tournament final selection (not argmax). With population-level early stop (≥2 perfect candidates). Achieves 35/42 IMO 2025, 36/42 USAMO 2026 (gold-medal threshold).

**Distilled for katgpt-rs (modelless):** The MaxProof loop is a modelless test-time scaling framework that composes our existing primitives (DDTree expansion, BtRank tournament, ScreeningPruner verification, BanditPruner selection) into a population-search architecture with three novel elements: (1) PATCH/REWRITE dual refinement, (2) tournament self-pick under noisy verifier, (3) redundancy-checked early stop.

---

## 1. Paper Core Findings

### 1.1 The MaxProof Loop (Algorithm 1)

```
Initialize: Sample N=32 candidate proofs from generator G
For each candidate: verify K_verify=4 times, take MIN score (conservative fitness)
Archive = {(candidate, min_score, critique, summary)}

For round r = 1..R=10:
    If ≥2 candidates have score 7/7: BREAK (population-level early stop)
    Select top-M=4 diverse parents (by fitness, lexical-distance diversity filter)
    For each parent:
        PATCH offspring: fix specific errors from critique (exploitation)
        REWRITE offspring: try different route using sibling summaries (exploration)
        Verify each offspring K_verify times, add to archive

Final: Pairwise tournament over top-K=4 by fitness, K_ranker=3 votes per match
```

### 1.2 Three Novel Selection Mechanisms

1. **Conservative fitness (min aggregation)** — verify each candidate K_verify times, take the MIN score. Deliberately pessimistic: false negative discards one candidate; false positive promotes a flawed proof. `fitness = min(verify_1, verify_2, ..., verify_{K_verify})`

2. **Tournament self-pick (not argmax)** — when verifier scores are clustered, direct pairwise comparison breaks ties more reliably than absolute scoring. Final selection is a tournament over top-K=4, with K_ranker=3 votes per match.

3. **Population-level early stop (redundancy check)** — stop only when ≥2 candidates reach max fitness. Single perfect candidate might be a false positive; two independent perfect candidates are unlikely to both be false positives.

### 1.3 PATCH vs REWRITE — Exploitation vs Exploration

| Mode | Input | Action | Analog |
|------|-------|--------|--------|
| PATCH | (problem, flawed_proof, critique) | Fix specific errors, preserve correct parts | Exploitation (local search) |
| REWRITE | (problem, flawed_proof, sibling_summaries) | Try different route, avoid siblings' failures | Exploration (divergent search) |

Both receive compact summaries of other archive candidates — population context enables cross-pollination without full-proof inclusion.

### 1.4 Diverse Parent Selection

Top-M=4 parents by fitness, with **lexical-distance diversity filter**: two candidates sharing a long common prefix are not both selected. Prevents spending all refinement calls on near-duplicates.

### 1.5 Defense-in-Depth Verifier (Training Side → riir-train)

4 layers suppressing distinct reward-hacking failure modes:
1. Bad-case filtering (empty/boilerplate → score 0)
2. Solution normalization (strip format)
3. Multi-judge scoring (3 judges: 2 rubric + 1 no-rubric)
4. Pessimistic min aggregation

**Reward hacking taxonomy (diagnostic, modellessly useful):**
- Length bias (longer = easier to match rubric keywords)
- Format hacking (converge on templates)
- Semantic shortcut ("it can be shown" at hard steps)
- Judge-specific preference (learn judge idiosyncrasies)

### 1.6 Results

| System | IMO 2025 | USAMO 2026 |
|--------|----------|------------|
| M3 (one-shot) | 27/42 | 26/42 |
| M3 + MaxProof | **35/42** | **36/42** |
| Gain | +8 | +10 |

---

## 2. Distillation — Modelless Path

### 2.1 What Maps to Our Existing Stack

| MaxProof Concept | katgpt-rs Equivalent | Status |
|------------------|---------------------|--------|
| Generator (N samples) | DDTree branch expansion | ✅ Shipped |
| Verifier (K_verify samples) | ScreeningPruner + ConstraintPruner | ✅ Shipped |
| Conservative min fitness | NOT shipped (we use single-check) | ⚠️ Gap |
| PATCH refinement | NOT shipped (no critique-conditioned repair) | ⚠️ Gap |
| REWRITE refinement | DDTree re-expansion (partial) | ⚠️ Gap |
| Diverse parent selection | BanditPruner diversity (partial) | ⚠️ Gap |
| Tournament self-pick | BtRank pairwise tournament (R040, R093) | ✅ Shipped |
| Population early stop | NOT shipped | ⚠️ Gap |
| Sibling summaries | NOT shipped (no cross-candidate context) | ⚠️ Gap |

### 2.2 The Modelless Primitive — Population Search with Dual Refinement

The distilled framework:

```rust
struct MaxProofLoop<G, V, R, Q> {
    generator: G,      // DDTree or any SpeculativeGenerator
    verifier: V,       // ConstraintPruner or ScreeningPruner
    refiner: R,        // PATCH/REWRITE operator
    ranker: Q,         // BtRank pairwise
    n_initial: usize,  // 32
    k_verify: usize,   // 4
    n_rounds: usize,   // 10
    m_parents: usize,  // 4
    top_k_tournament: usize, // 4
}

impl MaxProofLoop {
    fn run(&mut self, problem) -> Solution {
        let mut archive = self.sample_initial(problem);  // N candidates
        self.score_conservative(&mut archive);             // min across K_verify

        for round in 0..self.n_rounds {
            if self.count_perfect(&archive) >= 2 { break; } // redundancy early stop
            let parents = self.select_diverse_parents(&archive);
            for parent in parents {
                let patch = self.refiner.patch(problem, parent, &archive);
                let rewrite = self.refiner.rewrite(problem, parent, &archive);
                self.score_conservative_add(&mut archive, patch);
                self.score_conservative_add(&mut archive, rewrite);
            }
        }
        self.tournament_select(&archive)  // BtRank pairwise, not argmax
    }
}
```

### 2.3 Conservative Fitness — Why Min Aggregation Matters

Our current verifiers do single-check (`is_valid` → bool). MaxProof proves that under noisy verification, **min across K samples** is strictly better than mean or single-check:
- Single-check: false positive rate = p_fp
- Min across K: false positive rate = p_fp^K (exponentially lower)
- Cost: K× verification compute

For our modelless setting: if our `ConstraintPruner` has non-zero false-positive rate (it does — soft constraints), then verifying K times and taking min is a free quality gain at K× cost.

### 2.4 PATCH vs REWRITE as Bandit Arms

PATCH = exploit (local repair), REWRITE = explore (divergent retry). This maps naturally to our BanditPruner:

```rust
enum RefinementArm {
    Patch,   // exploitation: fix specific errors
    Rewrite, // exploration: try different route
}

// BanditPruner learns which arm works better for which problem type
let arm = bandit.select_arm(context_features);
let offspring = match arm {
    Patch => refiner.patch(problem, parent, critique),
    Rewrite => refiner.rewrite(problem, parent, sibling_summaries),
};
```

### 2.5 Population-Level Early Stop — Redundancy Check

Current: we run to fixed budget or first success. MaxProof: run until **≥2 independent successes**. The probability that two independently-generated false positives both pass is p_fp² ≪ p_fp. This is a cheap insurance policy against verifier false positives.

---

## 3. Fusion Ideas

### F1: DDTree × MaxProof — Population Tree Search

DDTree currently expands a single tree. Fuse with MaxProof: expand N=32 independent DDTree branches (population), verify each conservatively, refine top-M=4 with PATCH (extend branch) / REWRITE (re-expand from parent), tournament-select final. **Gain:** converts DDTree from single-tree to population-tree search, robust to branch-level noise.

### F2: BtRank × MaxProof — Conservative Tournament

BtRank already does pairwise Bradley-Terry ranking. Fuse: replace single-check scoring with K_verify=4 min-aggregated scoring, then BtRank tournament over top-K=4. **Gain:** tournament under noisy verifier is more robust than argmax under noisy verifier.

### F3: Committee Search (R093) × MaxProof — Unified Population-Committee Framework

Research 093 (Committee Search) has Copeland tournament + k proposers + m critics. MaxProof adds PATCH/REWRITE refinement + conservative min + early stop. Fusion: Committee Search with MaxProof's refinement loop = population committee with dual repair modes. **Gain:** the two frameworks are complementary — Committee Search handles the propose/critic/tournament axis, MaxProof adds the refine/early-stop axis.

### F4: CGSP (R240) × MaxProof — Curiosity-Guided Population Search

CGSP generates subgoals via curiosity. Fuse: use CGSP's curiosity signal to drive the REWRITE arm (explore curiosity-driving directions), while PATCH handles exploitation. **Gain:** curiosity-guided exploration × critique-guided exploitation = balanced population search.

### F5: MLEvolve (R239) × MaxProof — Progressive MCGS Population Search

MLEvolve has Base/Stepwise/Diff modes (≈ PATCH/REWRITE/Base). MaxProof's PATCH ≈ Diff, REWRITE ≈ Base. Fusion: Progressive MCGS graph with MaxProof's dual-refinement + tournament selection. **Gain:** graph-based population search without credit pollution.

---

## 4. Verdict: GOAT

**One-line reasoning:** MaxProof's modelless distillation (population search with PATCH/REWRITE dual refinement, conservative min-fitness verification, tournament self-pick, redundancy early stop) composes our existing primitives (DDTree, BtRank, BanditPruner, ScreeningPruner) into a novel test-time scaling framework with provable gain (+8-10 points on competition math) — but it's a composition of existing capabilities, not a new capability class.

**Not Super-GOAT because:**
- Q1 (no prior art): PARTIAL — BtRank (R040), Committee Search (R093), MLEvolve (R239) cover tournament, population, and dual-mode refinement respectively. The *combination* is novel, but the *components* exist.
- Q2 (new class): NO — it's a better test-time scaling, not a new capability class.
- Q3 (selling point): Weak — "population search with dual refinement" is an inference-time optimization, not a unique product feature.
- Q4 (force multiplier): YES — connects to DDTree, BtRank, BanditPruner, CGSP, Committee Search.

**GOAT gate criteria:**
- G1: PATCH/REWRITE dual refinement must beat single-mode (patch-only or rewrite-only) by ≥5% on bomber_arena
- G2: Conservative min-fitness (K=4) must reduce false-positive selections by ≥50% vs single-check
- G3: Tournament self-pick must match or beat argmax when verifier noise > 10%
- G4: Redundancy early stop must reduce avg compute by ≥20% without quality loss
- G5: Full loop overhead must be <3× single-pass DDTree cost

---

## 5. What Stays Where (3-Repo Discipline)

| Component | Repo | Why |
|-----------|------|-----|
| Population search loop framework | katgpt-rs (MIT) | Generic test-time scaling |
| PATCH/REWRITE refinement traits | katgpt-rs (MIT) | Generic refinement operators |
| Conservative min-fitness verifier wrapper | katgpt-rs (MIT) | Generic verifier composition |
| Tournament self-pick (already shipped) | katgpt-rs (MIT) | BtRank |
| Game-side refinement operators (NPC dialogue repair, quest refinement) | riir-ai (private) | Game IP |
| Proof Expert / Verifier Expert / Fixer Expert training | riir-train (private) | RL + SFT training |

---

## 6. Diagnostic Value — Reward Hacking Taxonomy

Even without implementing the full loop, MaxProof's **reward hacking taxonomy** is immediately valuable as a diagnostic lens for our existing systems:

| Pattern | Our Analog | Diagnostic |
|---------|-----------|------------|
| Length bias | DDTree deeper branches always score higher | Check if score correlates with depth |
| Format hacking | BanditPruner converges on same arm | Check arm diversity over time |
| Semantic shortcut | ScreeningPruner passes "plausible but wrong" | Check false-positive rate on adversarial inputs |
| Judge-specific preference | Verifier overfits to training distribution | Check verifier agreement across distributions |

---

## TL;DR

**Verdict: GOAT.** MaxProof's modelless distillation is a population-level test-time scaling loop that composes DDTree (N candidates) + ScreeningPruner (K_verify conservative min) + PATCH/REWRITE dual refinement + BtRank (tournament self-pick) + redundancy early stop. The components exist in our stack; the *combination* with conservative min-fitness and dual refinement is novel and provably gains +8-10 points on competition math. Five fusion targets: DDTree population tree, BtRank conservative tournament, Committee Search unified framework, CGSP curiosity-guided REWRITE, MLEvolve Progressive MCGS. The reward hacking taxonomy is immediately useful as a diagnostic. Training pipeline (Proof/Verifier/Fixer Expert RL+SFT) → riir-train. No files beyond this note per GOAT protocol; plan creation deferred pending GOAT gate.
