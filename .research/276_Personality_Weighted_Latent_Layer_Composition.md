# Research 276 (katgpt-rs): Personality-Weighted Latent Layer Composition — Open Primitive

> **Source:** Internal design — open half of the riir-ai Super-GOAT (Research 146)
> **Date:** 2026-06-21 (refined 2026-06-21)
> **Status:** Active — implemented in Plan 297 (katgpt-rs), promoted to DEFAULT-ON after GOAT G4 (79.585ns) + G5 (zero alloc) passed. See `.plans/297_personality_weighted_composition.md` and `.benchmarks/297_personality_composition_goat.md`.
> **Classification:** Public
> **Related Research:** 242 (`MicroRecurrentBeliefState`), 111 (Analogical Reasoning)
> **Cross-ref (riir-ai):** Research 146 (private Super-GOAT guide — game-specific 9-layer mapping, species archetype table, taming transition, validation gates)
> **Companion Doc:** `riir-ai/.docs/60_entity_cognition_stack.md`

---

## TL;DR

A generic modelless primitive: compose `N` latent direction vectors `dᵢ ∈ ℝ^D` into a single behavior vector via a personality weight vector `w ∈ ℝ^N` with sigmoid gating, and update `w` via an EMA on reward prediction error. **No game semantics, no game IP, no entity-kind assumptions.** Pure math — the kernel, the drift rule, and a trait surface that any host (game, robot, recommender) can implement for any kind of agent (NPC, player avatar, predator, prey, robot arm, etc.).

**Distilled for katgpt-rs (modelless, inference-time):**
- Composition: `behavior = Σᵢ sigmoid(wᵢ / τ) · dᵢ`
- Drift: `Δwᵢ = α · (R_observed − R_expected) · dᵢ_recent`
- Clamp: `wᵢ ∈ [−w_max, +w_max]`
- Optional belief-gating: `dᵢ ← belief_confidence_i · dᵢ` (for remote layers acting on stale belief)
- Snapshot: `(base_kernel, archetype, w, version, blake3)` — versioned, hashable, atomic-swappable

---

## 1. The Math

### 1.1 Composition kernel

Given `N` direction vectors `d₁..d_N ∈ ℝ^D` and weights `w₁..w_N ∈ ℝ`:

```text
behavior = Σᵢ sigmoid(wᵢ / τ) · belief_confidence_i · dᵢ
```

Where `belief_confidence_i ∈ (0, 1]` is an optional host-supplied scalar (default 1.0). For remote layers acting on stale belief (e.g. a wolf's KIN(pack) layer when the pack is not visible), the host decays confidence via `sigmoid(−λ · Δtick_since_last_observation)` and the layer's contribution shrinks accordingly. For immediate layers (plasma-tier SENSE/SITUATION), confidence is always 1.0.

Properties:
- Output is in `ℝ^D` (same space as inputs — latent-to-latent)
- Each layer's contribution is in `[0, 1]` (sigmoid × confidence); negative `wᵢ` → contribution approaches 0 (resistance), positive → approaches confidence (embodiment bounded by belief freshness)
- `τ` is the personality-sharpness temperature: `τ → ∞` flattens all contributions to 0.5 (no personality); `τ → 0` makes weights binary (extreme personality)
- Cost: `O(N · D)` — for `N=9, D=32`, that's 288 multiplies; trivially SIMD-able

### 1.2 Drift rule

```text
Δwᵢ = α · (R_observed − R_expected) · dᵢ_recent
wᵢ ← clamp(wᵢ + Δwᵢ, −w_max, +w_max)
```

Where:
- `α ∈ (0, 1)` is the plasticity (host-configured)
- `R_observed ∈ ℝ` is the host-supplied reward this step
- `R_expected ∈ ℝ` is the EMA of recent `R_observed` (host maintains)
- `dᵢ_recent ∈ ℝ^D` is the EMA of the layer's recent direction vector (host maintains)
- `w_max` is the clamp bound (host-configured, prevents runaway)

The drift is **sign-driven by reward surprise × recent direction**. If layer i contributed heavily to an action that beat expectations, `wᵢ` increases. If it underperformed, `wᵢ` decreases.

### 1.3 Why sigmoid, not softmax

Per AGENTS.md, sigmoid is mandated for projections onto learned direction vectors. Softmax over layers would destroy the "negative weight = resistance" semantics — softmax always assigns non-trivial probability to every layer. Sigmoid allows a layer to contribute ~0 (the NPC ignores it) or ~1 (the NPC embodies it), with signed resistance.

### 1.4 Why this is modelless

- No weight updates to any underlying kernel
- No backprop
- Drift is O(N) EMA — fits in L1 cache
- Snapshot is the freeze/thaw artifact — atomic swap, BLAKE3-committed

---

## 2. Proposed Trait Surface

```rust
/// A host-supplied source of a latent direction vector for one layer.
///
/// The host (game, robot, recommender) implements this per layer per entity.
/// The composition kernel calls `direction` once per tick per layer.
///
/// The trait is entity-agnostic: a wolf's KIN(pack) layer and a shopkeeper's
/// COMPANIONS(party) layer both implement this trait with different internals.
pub trait LayerDirectionSource {
    /// Returns the direction vector `d ∈ ℝ^D` for this layer at this tick.
    ///
    /// Implementations should be zero-allocation; reuse a scratch buffer.
    fn direction(&self, scratch: &mut [f32]) -> &[f32];

    /// Returns the EMA-smoothed recent direction (for drift computation).
    /// Default: returns the current direction.
    fn recent_direction(&self) -> &[f32];

    /// Optional belief confidence in (0, 1]. For remote layers acting on
    /// stale think-brain belief (e.g. KIN/COMPANIONS/COMMUNITY when target
    /// not visible), the host decays this via `sigmoid(-λ · Δtick)`.
    /// For plasma-tier immediate layers, returns 1.0.
    ///
    /// The kernel multiplies `direction() * belief_confidence()` before
    /// applying the personality weight sigmoid.
    fn belief_confidence(&self) -> f32 {
        1.0
    }
}

/// The personality-weighted composition kernel.
///
/// Generic over the number of layers `N` and direction dimension `D`.
/// The host owns the layer sources and the reward signal.
pub struct PersonalityWeightedComposition<const N: usize, const D: usize> {
    /// Personality weights, one per layer. Signed; clamped to `[−w_max, +w_max]`.
    pub w: [f32; N],
    /// Personality-sharpness temperature.
    pub tau: f32,
    /// Plasticity (drift learning rate).
    pub alpha: f32,
    /// Clamp bound on `w`.
    pub w_max: f32,
    /// EMA of observed reward (for drift).
    r_expected: [f32; N],
}

impl<const N: usize, const D: usize> PersonalityWeightedComposition<N, D> {
    /// Compose `N` layer direction vectors into a single behavior vector.
    ///
    /// Each layer's contribution is `sigmoid(wᵢ/τ) · belief_confidence_i · dᵢ`.
    /// Writes into `out` (length `D`). Returns `&mut out` for chaining.
    /// Zero-allocation: caller provides scratch and out buffers.
    pub fn compose_into(
        &self,
        layers: &[&dyn LayerDirectionSource; N],
        scratch: &mut [f32],
        out: &mut [f32],
    ) -> &mut [f32] {
        debug_assert_eq!(out.len(), D);
        debug_assert!(scratch.len() >= D);
        out.fill(0.0);
        for (i, layer) in layers.iter().enumerate() {
            let d = layer.direction(scratch);
            debug_assert_eq!(d.len(), D);
            let weight = sigmoid(self.w[i] / self.tau) * layer.belief_confidence();
            for j in 0..D {
                out[j] += weight * d[j];
            }
        }
        out
    }

    /// Update personality weights from observed reward.
    ///
    /// `r_observed` is the host-supplied reward this step. Updates the
    /// per-layer EMA `r_expected` and applies the drift rule. Clamps `w`.
    pub fn drift(
        &mut self,
        layers: &[&dyn LayerDirectionSource; N],
        r_observed: f32,
        ema_decay: f32,
    ) {
        for (i, layer) in layers.iter().enumerate() {
            let d_recent = layer.recent_direction();
            let surprise = r_observed - self.r_expected[i];
            for j in 0..D.min(d_recent.len()) {
                self.w[i] += self.alpha * surprise * d_recent[j];
            }
            self.w[i] = self.w[i].clamp(-self.w_max, self.w_max);
            // EMA update of expected reward
            self.r_expected[i] = ema_decay * self.r_expected[i] + (1.0 - ema_decay) * r_observed;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Numerically stable sigmoid per AGENTS.md
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
```

---

## 3. What the Host Implements

| Trait/method | Host responsibility |
|---|---|
| `LayerDirectionSource::direction` | Compute the layer's direction vector at this tick (game-specific: KG-driven, faction-driven, WASM-validated, etc.) |
| `LayerDirectionSource::recent_direction` | Maintain EMA of recent direction vectors |
| `LayerDirectionSource::belief_confidence` | For remote layers, decay via `sigmoid(-λ · Δtick)` based on think-brain staleness; for immediate layers, return 1.0 |
| Reward signal `r_observed` | Game-mode-specific payoff (combat damage, food eaten, gold earned, social approval, etc.) |
| Archetype label | Optional species/role label that seeds initial `w` (predator/prey/monster/player/NPC/wildlife/pet, or recommender user-segment, etc.) |
| `N`, `D`, `alpha`, `tau`, `w_max`, `ema_decay` | Host-configured constants |

katgpt-rs provides:
- The composition kernel (with belief-gating)
- The drift rule
- The sigmoid helper
- Snapshot integration (extend `MicroRecurrentKernelSnapshot` with `archetype: ArchetypeLabel` and `w: [f32; N]`)

---

## 4. Why This Belongs in katgpt-rs (Public)

- **Generic math.** No game terms (no "faction", no "law", no "family"). The kernel is `N × D` linear algebra + sigmoid + EMA.
- **Reusable beyond games.** A recommender could compose `(user_trait × item_price × friend_recommendation × ...)` with personality weights. A robot could compose `(safety × task × curiosity × ...)`.
- **Composes with existing public primitives.** `MicroRecurrentKernelSnapshot` (R242) is already public; this extends it with a `w` field.
- **No IP leak.** The 7-layer mapping, archetype table, WASM LAW functors, faction vectors — all stay private in riir-ai (Research 146).

---

## 5. Tests Required (Pre-Merge)

- `compose_zero_weights_uniform` — when all `wᵢ = 0` and `τ` finite, output is `0.5 × Σ dᵢ` (uniform personality)
- `compose_extreme_positive_weight_selects_layer` — large `wᵢ` → output ≈ `belief_confidence · dᵢ`
- `compose_extreme_negative_weight_zeros_layer` — large negative `wᵢ` → layer contributes ~0
- `compose_belief_confidence_decay_shrinks_contribution` — layer with `belief_confidence = 0.1` contributes 10% of full
- `drift_positive_surprise_reinforces` — `r_observed > r_expected` with positive `d_recent` → `wᵢ` increases
- `drift_negative_surprise_penalizes` — `r_observed < r_expected` with positive `d_recent` → `wᵢ` decreases
- `drift_clamps_to_w_max` — repeated positive drift saturates at `w_max`
- `sigmoid_stable_for_extreme_inputs` — no overflow/NaN for `|x| > 50`
- `compose_zero_allocation` — heap profiler confirms 0 allocations in `compose_into`

---

## 6. Verdict

**Super-GOAT (open half).** Novel composition primitive (no prior art in katgpt-rs); new capability class when wired to game-specific layers (per riir-ai Research 146); product selling point ("emergent NPC moral character without per-NPC training"); force multiplier across 9 existing systems (R242, R123, R141, R142, R143, R145, `evolve_hla`, Fourier, `DualSignalGate`). Open primitive in katgpt-rs (generic math); Super-GOAT guide in riir-ai Research 146 (game-specific wiring + validation gates).

---

## TL;DR

A generic personality-weighted latent layer composition primitive for katgpt-rs: `behavior = Σᵢ sigmoid(wᵢ/τ) · belief_confidence_i · dᵢ`, with drift `Δwᵢ = α(R_obs − R_exp) · dᵢ_recent` and clamp `wᵢ ∈ [−w_max, +w_max]`. Zero-allocation, sigmoid-gated, belief-gated, snapshot-integrated. Entity-agnostic — applies to any agent (NPC, player, predator, prey, robot, recommender user). The open half of the riir-ai Super-GOAT (Research 146) — game-specific 9-layer mapping, species archetype table, taming transition, and validation gates stay private.
