# Research 156: Speculative Reconciliation Engine — Modelless Offline State Verification

> **Type:** Architecture Research (Novel fusion of speculative decoding + game state reconciliation)
> **Index:** 156
> **Status:** 📋 Plan Created
> **Depends On:** Plan 032 ✅ (HL Infrastructure), Plan 053 ✅ (DeltaMemory), Plan 155 ✅ (LEO trait framework), Plan 194 ✅ (Adaptive CoT)
> **Classification:** MIT Engine — core inference primitive
> **Related:** Research 037 (REAP Model-Based/Modelless Duality), Research 012 (LEO All-Goals), Research 024 (Neuro-Symbolic Chain Transport, in riir-ai)
> **Plan:** Plan 177

---

## TL;DR

When a player disconnects at T=10:00 and reconnects at T=10:01, they present a local action sequence that may be legitimate lag or a hack attempt. Instead of linear replay (O(n) per action) or blind trust (insecure), we apply the same **speculative verification** architecture used in token decoding to **game state reconciliation**.

The engine generates a **Plausibility Manifold** — a convex hull of K possible trajectories — using existing LEO Q-values and player HLA hidden states. The client's submitted trajectory is verified against this manifold via cosine similarity + velocity bounds + entropy bounds. All inside a single-threaded WASM sandbox in <1ms.

**Verdict: GO.** This is a direct application of existing `SpeculativeVerifier` + `ScreeningPruner` + `BanditPruner` infrastructure to a new domain. Zero new ML training required. The "speculative rollout" is modelless — it uses LEO Q-values (bandit arms), not LLM forward passes.

---

## 0. GOAT Verdict — Go/No-Go Decision

### Commercial Alignment (per Research 003)

| Criterion | Assessment |
|-----------|-----------|
| **Strengthens the moat?** | ✅ Yes — "Speculative Reconciliation" is unique. No game engine verifies offline state via latent manifold geometry. |
| **Uses `ConstraintPruner` trait?** | ✅ Yes — `ReconciliationPruner` implements `ConstraintPruner` for hard bounds (velocity, position). |
| **Engine/Fuel split intact?** | ✅ Yes — all code in `katgpt-rs` (MIT engine). Reconciliation is an inference-time primitive. |
| **Composes existing infrastructure?** | ✅ Yes — reuses `SpeculativeVerifier`, `ScreeningPruner`, `BanditPruner`, `DeltaMemory`, `LEO`, `HLA`. |
| **Feature-gated?** | ✅ Yes — `spec_reconciliation` feature gate, zero impact on base speculative decode. |

### Performance Alignment (per optimization.md)

| Principle | Compliance |
|-----------|-----------|
| Profile first | Manifold generation is LEO Q-value lookup (O(1) via HashMap) × K trajectories. Sub-µs. |
| Fixed-size arrays | `TrajectoryPoint` is fixed-layout `[f32; 8]` — player state vector. |
| Pre-compute lookup tables | LEO Q-values are pre-computed during gameplay, not during reconciliation. |
| Zero-copy serialization | Same Freeze/Thaw binary format as existing `WalletWeight`. |
| Don't GPU for µs work | All reconciliation checks are CPU-side scalar + SIMD cosine similarity. |
| Don't allocate in hot loops | Manifold buffer pre-allocated, reuse across reconciliations. |

### SOLID Compliance

| Principle | Compliance |
|-----------|-----------|
| **S** (Single Responsibility) | `SpecReconciler` handles trajectory verification only; state merge handled by existing `ConsensusPipeline`. |
| **O** (Open/Closed) | New manifold generators added via `ManifoldGenerator` trait, no modification of `SpecReconciler`. |
| **L** (Liskov) | `ReconciliationPruner` is a `ConstraintPruner`; `ManifoldScorer` is a `ScreeningPruner`. |
| **I** (Interface Segregation) | Hard bounds (`ConstraintPruner`) and soft scoring (`ScreeningPruner`) are separate traits. |
| **D** (Dependency Inversion) | Depends on `LeoHead` trait, not concrete LEO implementation. |

### Decision: **GO**

1. Reuses 80% existing infrastructure (`SpeculativeVerifier`, `ScreeningPruner`, `LEO`, `HLA`, `DeltaMemory`).
2. No LLM training — the manifold is generated from bandit Q-values, not neural forward passes.
3. The "high-dimensional geometry check" is just cosine similarity (existing SIMD) against LEO-scored trajectories.
4. Feature-gated → zero perf impact on existing speculative decoding.
5. GOAT-provable: inject fake trajectory → verify rejection in <1ms.

---

## 1. Problem: The Offline/Desync Gap

### 1.1 Current State of the Art

| Approach | Latency | Security | User Experience |
|----------|---------|----------|-----------------|
| **Server-authoritative replay** | O(n) × actions | Perfect | Long loading screens for >30s offline |
| **Client trust + post-hoc audit** | Instant | Weak (hackable) | Smooth but exploitable |
| **Snapshot diff + hash comparison** | Fast | Medium | Only detects tampering, not plausibility |
| **ECS rollback + re-sim** | O(n) × tick | Good | CPU-intensive, blocks main loop |

### 1.2 The Key Insight

The system already has everything needed for a better approach:

1. **`LEO` Q-values** — tell us what a player *would likely do* given their goal state (survive, hunt, explore)
2. **`HLA` hidden states** — encode the player's behavioral fingerprint (aggressive, cautious, curious)
3. **`SpeculativeVerifier`** — already verifies candidate tokens against a target model
4. **`ScreeningPruner`** — already scores candidates on a [0,1] relevance scale
5. **`BanditPruner`** — already learns which "arms" (trajectories) are worth exploring

The fusion: **treat each possible offline trajectory as a "speculative token" and verify it against a manifold of LEO-predicted plausible futures.**

---

## 2. Architecture: The Speculative Reconciliation Pipeline

### 2.1 Core Abstractions

```text
┌──────────────────────────────────────────────────────────────┐
│                   SPECULATIVE RECONCILIATION                  │
│                                                              │
│  Client submits:  T_client = [action_1, action_2, ...]      │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐     │
│  │ ManifoldGenerator│    │ ReconciliationPruner (hard)  │     │
│  │ (LEO + HLA)     │    │ - velocity_bound             │     │
│  │                  │    │ - position_bound             │     │
│  │ Generate K       │    │ - entropy_bound (kill rate)  │     │
│  │ plausible        │    │                              │     │
│  │ trajectories     │    │ Implements: ConstraintPruner │     │
│  └────────┬─────────┘    └──────────────┬───────────────┘     │
│           │                             │                     │
│           ▼                             ▼                     │
│  ┌─────────────────────────────────────────────────────┐      │
│  │           ManifoldScorer (soft)                      │      │
│  │  similarity = max_j(cosine(T_client, T_spec_j))     │      │
│  │                                                      │      │
│  │  Implements: ScreeningPruner                         │      │
│  │  Output: f32 ∈ [0, 1] plausibility score             │      │
│  └──────────────────────┬──────────────────────────────┘      │
│                         │                                     │
│                         ▼                                     │
│  ┌─────────────────────────────────────────────────────┐      │
│  │           ReconciliationVerdict                      │      │
│  │  Accept  (score ≥ θ_accept)  → merge state           │      │
│  │  Quarantine (score < θ_quarantine) → rollback        │      │
│  │  Uncertain → escalate to MAPE-K / ConflictFSM        │      │
│  └─────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Trait Mapping to Existing Infrastructure

| New Component | Existing Trait | Implementation Strategy |
|---------------|---------------|------------------------|
| `ReconciliationPruner` | `ConstraintPruner` | Hard bounds: velocity, position, kill-rate. Returns `bool`. |
| `ManifoldScorer` | `ScreeningPruner` | Soft scoring: cosine similarity against LEO manifold. Returns `f32 ∈ [0,1]`. |
| `SpecReconciler` | `SpeculativeVerifier` | Orchestrates manifold generation + verification. Returns `ReconciliationVerdict`. |
| `ManifoldGenerator` | New trait | Generates K plausible trajectories from LEO + HLA. Pure function, no state. |
| `AdaptiveThreshold` | `BanditPruner<ManifoldScorer>` | Learns optimal accept/quarantine thresholds per-player via bandit. |

### 2.3 The ManifoldGenerator — Modelless Trajectory Prediction

The key constraint: **no LLM forward pass during reconciliation**. Instead:

```text
ManifoldGenerator:
  Input:
    h_last: HLA hidden state at disconnect time (8-dim vector, already computed)
    q_goals: LEO Q-values for all active goals (Vec<f32>, already computed)
    K: number of speculative trajectories (default: 16)
    dt: offline duration in seconds
    
  Output:
    T_spec: [TrajectoryPoint; K] — K plausible final states
    
  Algorithm (modelless):
    1. For each trajectory j ∈ [1, K]:
       a. Sample goal g_j from q_goals via softmax → argmax (LEO's best guess)
       b. Compute velocity vector v_j = goal_direction(g_j) × max_speed × personality_scale(h_last)
       c. Add exploration noise: ε ~ N(0, σ) where σ = f(dt) — uncertainty grows with offline time
       d. T_spec[j] = h_last.position + v_j × dt + ε
       
    2. Return T_spec as fixed-size array
```

**No neural network involved.** The "prediction" is:
- LEO Q-values → which goal the player pursues (already computed)
- HLA hidden state → player personality/aggression (already computed)
- Physics → maximum speed × time → hard position bound (deterministic)
- Gaussian noise → uncertainty grows with offline time (simple RNG)

This is a **bandit-weighted physics simulation**, not an LLM forward pass.

### 2.4 The Verification Pipeline

```text
verify(T_client, h_last, q_goals, dt) → ReconciliationVerdict:

  // Step 1: Hard bounds (ConstraintPruner — O(1) per check)
  if velocity_exceeds(T_client, dt, max_speed) → Quarantine
  if position_exceeds(T_client, map_bounds) → Quarantine  
  if kill_rate_exceeds(T_client, dt, chebyshev_5σ(local_area)) → Quarantine
  
  // Step 2: Generate plausibility manifold (modelless — O(K) where K=16)
  T_spec = ManifoldGenerator::generate(h_last, q_goals, K=16, dt)
  
  // Step 3: Soft scoring (ScreeningPruner — O(K) cosine similarity, SIMD)
  score = max_j(cosine_sim(T_client, T_spec[j]))
  
  // Step 4: Adaptive verdict (BanditPruner learns threshold)
  if score ≥ θ_accept → Accept(merge state deltas)
  elif score < θ_quarantine → Quarantine(rollback + flag)
  else → Uncertain(escalate to MAPE-K)
```

**Total cost:** O(K) where K=16 fixed. Sub-millisecond on any CPU.

---

## 3. The Dual State Architecture

### 3.1 What Gets Reconciled

| State Layer | Reconciliation Policy | Existing Infrastructure |
|-------------|----------------------|------------------------|
| **Chain State** (wallet, tokens, trades) | **Never speculative.** Queued as signed `TxDelta` envelope. Requires server β-key handshake. | `ChainConsensus`, LatCal matrix ops, split-key protocol |
| **Game State** (position, health, kills) | **Speculative.** Verified against manifold on reconnect. | `SpecReconciler`, LEO, HLA, WASM sandbox |
| **AI State** (NPC behavior, world mood) | **Deterministic.** Re-derived from VibeSchedule + seed. No reconciliation needed. | `VibeSchedule`, `GameRng` ChaCha20 |

### 3.2 The TxDelta Envelope (Offline Queue)

```text
// When offline, chain-relevant actions are queued locally:
struct TxDeltaEnvelope {
    intent:    [u8; 32],     // BLAKE3 hash of action intent
    nonce:     u64,          // Monotonic, prevents replay
    alpha_seed: [u8; 32],   // Client-side key fragment
    timestamp: u64,          // When action was attempted
    // Does NOT contain: value, wallet state, or any decryptable data
}
```

On reconnect:
1. Server processes `TxDeltaEnvelope` queue sequentially
2. Each envelope gets the β-key handshake + LatCal matrix validation
3. Failed envelopes are discarded (not rolled back — they were never applied)

---

## 4. The Catch-Up User Experience

| Offline Duration | Manifold Width (σ) | Verification Time | User Experience |
|------------------|---------------------|-------------------|-----------------|
| <5s (jitter) | Narrow (σ=0.1) | <100µs | Invisible — state merges instantly |
| 5-30s (lag spike) | Medium (σ=0.5) | <500µs | Brief "syncing..." overlay |
| 30s-5min (DC) | Wide (σ=2.0) | <2ms | Loading wheel with progress |
| >5min (extended offline) | Very wide (σ=5.0) | <10ms | Full catch-up screen with replay |

The manifold naturally widens with offline duration. Players accept longer sync times for longer offline periods because it matches MMO expectations.

---

## 5. GOAT Proof Specification

### 5.1 Formal Verification Gates

| Gate | What It Proves | Test Method |
|------|---------------|-------------|
| **G1: Velocity invariant** | No trajectory exceeds `max_speed × dt` | Property test: generate random trajectories, verify rejection |
| **G2: Position invariant** | No trajectory places player outside map | Property test: fuzz positions, verify rejection |
| **G3: Kill-rate bound** | Kill count within Chebyshev 5σ of zone average | Property test: inject inflated kill counts, verify rejection |
| **G4: Manifold coverage** | K=16 trajectories cover >95% of legitimate play patterns | Monte Carlo: simulate 10K legitimate trajectories, verify >95% fall within manifold |
| **G5: Latency bound** | Full reconciliation completes in <1ms for <5min offline | Micro-benchmark: 10K iterations, report P50/P99 |
| **G6: False positive rate** | <1% of legitimate trajectories are quarantined | Monte Carlo: 10K legitimate trajectories, verify accept rate ≥99% |
| **G7: False negative rate** | >99% of injected hacks are quarantined | Adversarial: inject 1K tampered trajectories, verify quarantine rate ≥99% |
| **G8: Matrix soundness** | `WalletWeight` invariants preserved post-reconciliation | Assert determinant audit passes on all accepted state merges |

### 5.2 Example: Before/After Test

```text
// tests/spec_reconciliation_proof.rs

#[test]
fn goat_g7_hack_rejection() {
    let reconciler = SpecReconciler::new(config);
    let h_last = player_hla_at_disconnect();  // Real player state at T=10:00
    let q_goals = player_leo_q_values();       // Real LEO Q-values
    
    // Legitimate trajectory: player walked 500m in 60s
    let legit = trajectory(position(0,0), position(500,0), kills=3, dt=60.0);
    assert!(matches!(reconciler.verify(&legit, &h_last, &q_goals, 60.0), Accept(_)));
    
    // Hack: player "teleported" 5000m in 60s (impossible speed)
    let hack_teleport = trajectory(position(0,0), position(5000,0), kills=3, dt=60.0);
    assert!(matches!(reconciler.verify(&hack_teleport, &h_last, &q_goals, 60.0), Quarantine(_)));
    
    // Hack: player killed 200 monsters in 60s (impossible rate)
    let hack_kills = trajectory(position(0,0), position(500,0), kills=200, dt=60.0);
    assert!(matches!(reconciler.verify(&hack_kills, &h_last, &q_goals, 60.0), Quarantine(_)));
    
    // Hack: position is legitimate but trajectory violates manifold (wrong goal direction)
    let hack_direction = trajectory(position(0,0), position(-500,0), kills=3, dt=60.0);
    // Player's LEO says "head north" but they went south → low plausibility
    assert!(reconciler.verify(&hack_direction, &h_last, &q_goals, 60.0).score() < 0.5);
}
```

---

## 6. Relationship to Existing Research

| Research | Connection |
|----------|-----------|
| **R012 (LEO All-Goals)** | Q-values drive manifold generation — which goals the player pursues |
| **R037 (REAP Modelless)** | This IS the modelless path — no neural forward pass, bandit-weighted physics |
| **R024 (Neuro-Symbolic Chain, riir-ai)** | Chain state uses split-key protocol; game state uses speculative reconciliation |
| **R029 (LatCal, riir-ai)** | Matrix rollback for quarantined states — `A⁻¹` undoes the speculative delta |
| **R007 (Five-Tier Memory)** | Reconciliation lives in Hot tier; state merges flow through Hot→Cold pipeline |
| **R194 (Adaptive CoT)** | Bandit-learned thresholds → self-improving reconciliation accuracy |
| **R098 (Stiff Anomaly)** | Eigenbasis decomposition for detecting structural state manipulation |
| **R091 (SpecHop)** | Same multi-hop verification architecture, applied to game trajectories instead of tokens |
| **R035 (RSM Config Matrix)** | Reconciliation mode varies by cluster topology (C1-C5) |

---

## 7. What Makes This Novel

1. **Not LLM-based verification.** The manifold is modelless: LEO Q-values + physics + noise.
2. **Not linear replay.** O(K) where K=16 fixed, regardless of offline action count.
3. **Not static bounds.** The bandit learns per-player thresholds over time.
4. **Not separate from inference.** Same `SpeculativeVerifier` + `ScreeningPruner` pipeline used for token decoding.
5. **The "convex hull of plausible futures"** is a new primitive — it converts cheat detection from a rule-checking problem into a geometry problem.

---

## 8. Constraints Compliance

| Constraint | Compliance |
|-----------|-----------|
| **Modelless first** | ✅ No LLM inference. LEO Q-values are pre-computed bandit arms, not neural forward passes. |
| **Land in riir-ai domain** | ✅ Core traits in `katgpt-rs` (MIT engine); game integration in `riir-ai` (chain/crates). |
| **LoRA only for training** | ✅ No training during reconciliation. NeuronShard weights are read-only during verify. |
| **Self-learning adaptive CoT** | ✅ `BanditPruner<ManifoldScorer>` learns optimal thresholds per-player. |
| **SOLID/DRY** | ✅ Reuses 5 existing traits, adds 2 new ones. No duplication. |
| **Tests/examples** | ✅ 8 GOAT gates with before/after demonstrations. |
| **CPU/GPU auto-route** | ✅ All reconciliation is CPU-side (sub-ms). GPU reserved for gameplay rendering. |
