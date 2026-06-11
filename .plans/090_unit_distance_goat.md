# Plan 090: Unit Distance GOAT Proof — Number-Theoretic Lattice Constructions

> **Research:** [56_OpenAI_Unit_Distance_Disproof.md](../.research/056_OpenAI_Unit_Distance_Disproof.md)
> **Source:** OpenAI autonomous proof (2026), Remarks by Alon, Bloom, Gowers et al. (2026)
> **Feature Gate:** `unit_distance`
> **Type:** Modelless (T1–T3) / Light model-based (T4) / Model-based (T5)
> **Priority:** P2 (behind riir-ai Plan 089 ASFT/Plan 088 LDT)

## Tasks

- [x] **T1: MinkowskiLattice<f64>** — High-dimensional lattice embedding with sup-norm packing bounds
- [x] **T2: ClassGroupPigeonhole** — CM field element counting via ideal class pigeonhole
- [x] **T3: UnitDistanceGOAT** — Verify ν(n) ≥ n^(1+δ) for explicit small examples
- [x] **T4: CmFieldConstruction** — Pro-2 tower with prescribed splitting (light model-based)
- [x] **T5: InfiniteTowerSearch** — G-Zero self-play for optimal tower parameters (model-based) ✅ UCB1 bandit over (family, primes, denom) configurations
- [x] **T6: Feature Gate Audit** — `unit_distance` gate with zero default impact

---

## Context

OpenAI's model autonomously disproved Erdős's 1946 unit distance conjecture using a construction from algebraic number theory: an infinite tower of CM fields K_j = F_j(i) with bounded root discriminant, where fixed rational primes split completely. Projecting Minkowski-embedded lattice cosets to C gives planar point sets with ν(P) ≥ |P|^(1+δ) unit distances.

The construction has three reusable components for our GOAT proof infrastructure:
1. **Minkowski lattice averaging** — generic tool for combinatorial bounds
2. **Class group pigeonhole** — counting useful algebraic objects via ideal class frequency
3. **Prescribed splitting towers** — Chebotarev + Golod–Shafarevich for controlled field extensions

These map directly to our model-based/modelless duality:
- Modelless: static lattice + pigeonhole counting (no learning)
- Light model-based: field construction parameter search (one-shot optimization)
- Model-based: G-Zero self-play for optimal tower parameters

---

## T1: MinkowskiLattice<f64>

### What

A generic high-dimensional lattice module for combinatorial geometry proofs.

```rust
/// Lattice in C^f with sup-norm packing utilities.
pub struct MinkowskiLattice {
    /// Complex dimension (field degree f)
    dim: usize,
    /// Basis vectors in C^f (row-major)
    basis: Vec<Complex<f64>>,
    /// Minimum separation δ in sup-norm
    min_sep: f64,
    /// Covolume (determinant of basis matrix)
    covol: f64,
}
```

### Key Operations

1. **`injective_projection(&self, coord: usize) -> bool`** — Check if projection to one coordinate is injective
2. **`polydisc_cap(&self, center: &[Complex<f64>], radius: f64) -> Vec<...>`** — Points in B_R ∩ (a + Λ)
3. **`packing_bound(&self, radius: f64) -> usize`** — Upper bound on |X| via D-separated packing
4. **`coset_average_unit_pairs<U>(&self, unit_set: &[Complex<f64>], radius: f64) -> f64`** — Expected unit-distance pair count averaged over cosets

### Where

`src/unit_distance/minkowski.rs` behind `unit_distance` feature gate.

---

## T2: ClassGroupPigeonhole

### What

Count elements of absolute value 1 in CM fields via ideal class pigeonhole.

```rust
/// Result of class group pigeonhole on a CM field.
pub struct PigeonholeResult {
    /// Number of conjugate prime pairs {P_s, cP_s}
    num_prime_pairs: usize,
    /// Lower bound on |U| (norm-one elements)
    unit_set_lower_bound: usize,
    /// Denominator D for Q^(-2) O_K embedding
    denominator: u64,
    /// Class number bound used
    class_number_bound: usize,
}
```

### Algorithm (Lemma 2.2 from Remarks paper)

```
Input: CM field K, prime pairs {(P_s, cP_s)}, exponents {k_s}
Output: Lower bound on |{u ∈ Q^(-2) : |u| = 1}|

1. For each binary vector ε ∈ {0,1}^(m), form ideal A_ε = Π P_s^(ε_s) · cP_s^(k_s - ε_s)
2. Map ε → [A_ε] ∈ Cl(K) (ideal class group)
3. Find largest fiber: size ≥ Π(k_s + 1) / h(K)
4. For ε, η in same fiber: u = α_ε / c(α_ε) where (α_ε) = A_ε · A_η^(-1)
5. Then |σ(u)| = 1 for all embeddings σ
6. Distinct ε → distinct u (valuation vectors differ)
```

### Where

`src/unit_distance/pigeonhole.rs` behind `unit_distance` feature gate.

---

## T3: UnitDistanceGOAT

### What

GOAT proof that verifies the construction for explicit small examples.

### Proof Design

**Proof 1: Gaussian Integer Baseline (Q(i))**
- Erdős's original: √n × √n grid
- Verify ν(P) ≥ n^(1+c/log log n) statistically
- This is the sanity check — must match known results

**Proof 2: Q(√5, i) Small Example**
- CM field K = Q(√5, i), degree 4
- One split prime q = 5 (≡ 1 mod 4)
- Verify pigeonhole produces ≥ (k+1)^2 / h(K) unit elements
- Check projection injectivity

**Proof 3: Explicit ν(n) ≥ n^(1+δ) for small field**
- Use simplified construction from Remarks paper (pro-2 tower base)
- T = {3,5,7,11,13,17}, S = {101,∞}
- Verify finite-layer approximation gives δ > 0
- Statistical test: generate point sets, count unit distances

### GOAT Assertion Pattern

```rust
#[cfg(feature = "unit_distance")]
#[test]
fn goat_unit_distance_q_i_baseline() {
    // Erdős grid: √n × √n Gaussian integers
    // Assert ν(P) ≥ n^(1 + c/log_log_n) for n = 100, 1000, 10000
    println!("🐐 GOAT PROOF: Unit Distance Q(i) Baseline");
    // ...
    println!("✅ GOAT Proof passed. Q(i) baseline matches Erdős bound.");
}
```

### Where

`tests/bench_unit_distance_goat.rs` behind `#[cfg(feature = "unit_distance")]`.

---

## T4: CmFieldConstruction (Light Model-Based)

### What

Construct small CM fields with prescribed splitting for explicit examples.

```rust
/// A CM field K = L(i) with L totally real.
pub struct CmField {
    /// Totally real subfield L
    base_field: TotallyRealField,
    /// Degree [L:Q]
    degree: usize,
    /// Split rational primes (≡ 1 mod 4)
    split_primes: Vec<u64>,
    /// Class number h(K)
    class_number: u64,
    /// Root discriminant rd(K)
    root_discriminant: f64,
}
```

### What This Enables

- Explicit computation for T3 GOAT proofs
- Comparison of different tower parameters
- Verification that Chebotarev splitting works as claimed

### Complexity

This is **light model-based** because:
- Field construction is a one-shot computation (no gradient updates)
- Parameters are chosen by Chebotarev density (deterministic)
- Only needed for verification, not production inference

### Where

`src/unit_distance/cm_field.rs` behind `unit_distance` feature gate.

---

## T5: InfiniteTowerSearch (Model-Based, Deferred)

### What

Use G-Zero self-play to search for optimal tower parameters (T set, S set, prime p choice).

### Implementation

Implemented as `tower_search.rs` — a self-contained UCB1 bandit over tower parameter
configurations. Each arm represents a `(family, num_split_primes, denominator)` triple.
The reward signal is the computed δ exponent from `DeltaEstimate::from_field_params()`.

**Key types:**
- `TowerFamily` — enum: Qi, QSqrt5I, Pro2Tower, Cyclotomic
- `TowerArm` — search arm with field parameters
- `TowerBandit` — UCB1 bandit with warm-up, select/observe cycle
- `TowerSearchConfig` — search configuration (rounds, prime counts, denominators)
- `TowerSearchResult` — best arm, best δ, rankings, success flag
- `TowerSearch` — static runner: generate_arms → UCB1 loop → result

**GOAT proofs (10 tests):** G1 (positive δ), G2 (monotonicity), G3 (concentrates on best),
G4 (cross-family), G5 (determinism), G6 (scaling), G7 (theoretical formula), G8 (full pipeline),
G9 (exploration/exploitation), Summary.

### Connection to G-Zero

The tower parameter search is a **pure exploration** problem:
- State: (T, S, p) parameters
- Action: modify one parameter
- Reward: δ value achieved (larger is better)
- This maps directly to our Phase 1 modelless search (T1–T5 in G-Zero)

The implementation is self-contained (no import from `g_zero` module) but follows the
same UCB1 pattern as `DeltaBanditPruner`. Future work could integrate with the full
G-Zero infrastructure for more sophisticated search strategies.

---

## T6: Feature Gate Audit

### What

Ensure `unit_distance` feature gate has zero impact on default build.

### Checklist

- [x] `Cargo.toml`: `unit_distance` in `[features]` with optional deps only
- [x] All `src/unit_distance/` modules behind `#[cfg(feature = "unit_distance")]`
- [x] No `use` of unit_distance types in default modules
- [x] `cargo build` succeeds without `--features unit_distance`
- [x] `cargo test` passes without feature
- [x] `cargo clippy` clean without feature

---

## Module Structure

```
src/unit_distance/
├── mod.rs              # Feature-gated module index
├── minkowski.rs        # T1: MinkowskiLattice<f64>
├── pigeonhole.rs       # T2: ClassGroupPigeonhole
├── cm_field.rs         # T4: CmFieldConstruction (light model-based)
├── tower_search.rs     # T5: InfiniteTowerSearch (UCB1 bandit)
└── types.rs            # Shared types: Complex, primes, field params

tests/
├── bench_unit_distance_goat.rs  # T3: GOAT proofs (8 tests)
└── goat_090_tower_search.rs     # T5: GOAT proofs (10 tests)
```

---

## Dependencies

- `num-complex` — Complex<f64> arithmetic (already in workspace)
- `num-prime` or custom — Prime enumeration for split prime selection
- No new heavy deps — keep it minimal for research feature

---

## Success Criteria

1. **T1:** MinkowskiLattice packing bound matches theoretical prediction (±5%)
2. **T2:** Pigeonhole count ≥ Π(k_j + 1) / h(K) for explicit small fields
3. **T3:** GOAT proof passes: ν(n) ≥ n^(1+δ) for at least 3 different point sets
4. **T6:** Zero impact on default build (cargo build/test/clippy clean)
5. **Documentation:** README section added under 🔬 research features

---

## Priority Rationale

**P2** — This is valuable research infrastructure but:
- The technique is proven (by OpenAI), so there's no urgency to validate
- GOAT proofs for combinatorial geometry are useful but not on the critical path
- ASFT (Plan 089 in riir-ai) and LDT (Plan 088) have higher production impact
- Can be parallelized with other work since it's fully behind a feature gate
