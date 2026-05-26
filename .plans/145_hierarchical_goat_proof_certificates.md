# Plan 145: Hierarchical GOAT Proof Certificates — Formal Verification Methodology from Shock with Confidence

> **Research:** [106 — Shock with Confidence Formal PDE Verification](../.research/106_Shock_Confidence_Formal_PDE_Verification.md)
> **Paper:** [arXiv:2503.13877](https://arxiv.org/abs/2503.13877) — Formal Proofs of Correctness for Hyperbolic PDE Solvers (Gorard & Hakim, 2025)
> **Feature Gate:** `proof_cert` (opt-in, NOT default-on)
> **Status:** ✅ COMPLETE
> **GOAT Pillar:** ❌ Not a pillar — proof methodology enhancement. See [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md).
> **Domain:** `katgpt-rs` — generic proof certificate infrastructure. No game-specific code.
> **Blocks:** None. Enhances existing GOAT proof methodology.
> **Cross-ref:** Plan 128 (proof sketch evolution), Plan 143 (Nexus Elo Plackett-Luce P-UCB)

---

## Summary

Extract three proof methodology patterns from "Shock with Confidence": (1) **Hierarchical proof chains** where deeper properties imply shallower ones, (2) **Graduated proof results** (full/conditional/partial) instead of binary pass/fail, and (3) **Proof certificates** as standalone verifiable artifacts. All modelless, no training, no model changes. Enhances our GOAT proof methodology to be more informative and diagnostic.

---

## Why

1. **Flat GOAT proofs are fragile:** Our current GOAT proofs are independent threshold checks. If one fails, we don't know if it's a shallow issue (threshold too strict) or a deep issue (fundamental property violated). Hierarchical proofs give diagnostic structure.

2. **Binary pass/fail loses information:** A GOAT proof that says "passes given constraint X" is more useful than "passes" — if X changes, the proof may no longer hold. Conditional results make our proofs self-documenting.

3. **Proof certificates enable verification:** Currently our GOAT proofs live in markdown benchmark files. Structured proof certificates (JSON/toml) can be programmatically verified, enabling CI integration.

4. **Methodology paper validates our approach:** The paper's factorization strategy (decompose 4D → 2×2 pairs) matches our DDTree branch decomposition. This confirms our architectural approach.

5. **Cross-pollenates with Plan 143:** Plan 143 (Nexus Elo) provides the search strategy. This plan provides the verification methodology. Together: search + verify pipeline.

---

## Architecture

### Phase 1: Proof Certificate Format (T1–T3)

- [x] **T1: `ProofCertificate` struct**

```rust
/// A standalone proof certificate for a verified property.
/// Inspired by Shock with Confidence's "symbolic Racket code that can be independently verified."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCertificate {
    /// Unique identifier for this proof
    pub id: String,
    /// What property is being proved
    pub property: ProofProperty,
    /// The result (full, conditional, partial)
    pub result: ProofResult,
    /// Prerequisites that must hold for this proof to be valid
    pub prerequisites: Vec<String>,
    /// Implied properties (things this proof implies)
    pub implies: Vec<String>,
    /// Human-readable explanation
    pub explanation: String,
    /// Machine-readable evidence (benchmark data, hash values, etc.)
    pub evidence: ProofEvidence,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofProperty {
    /// Position invariance (Fourier MCTS)
    SpatialConsistency { game: String, board_size: usize },
    /// Deterministic correctness (WASM validator)
    DeterministicCorrectness { game: String, n_comparisons: usize },
    /// Real-time feasibility (NPC dialog, frame sampling)
    RealtimeFeasibility { domain: String, target_latency_us: u64 },
    /// Convergence (bandit, MCTS)
    Convergence { algorithm: String, metric: String },
    /// Custom property
    Custom { name: String, description: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofResult {
    /// Property fully proved
    Full { value: f64, threshold: f64 },
    /// Property proved given stated conditions
    Conditional {
        value: f64,
        threshold: f64,
        conditions: Vec<String>,
    },
    /// Property partially proved (some sub-properties hold)
    Partial {
        proved: Vec<String>,
        unproved: Vec<String>,
        reason: String,
    },
    /// Property could not be proved
    Failed { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofEvidence {
    /// Benchmark result with raw data
    Benchmark {
        n_samples: usize,
        mean: f64,
        std_dev: f64,
        min: f64,
        max: f64,
    },
    /// Hash-based deterministic verification
    Deterministic { seed: u64, expected_hash: String, actual_hash: String },
    /// Comparison-based verification
    Comparison { baseline: String, challenger: String, delta: f64 },
    /// Custom evidence
    Custom { data: serde_json::Value },
}
```

- [x] **T2: Proof chain verification**

```rust
/// Verify that a chain of proof certificates is consistent.
/// If certificate A implies B, and B implies C, then A's proof implies C.
pub fn verify_proof_chain(certificates: &[ProofCertificate]) -> ProofChainResult {
    let mut proven: HashSet<String> = HashSet::new();
    let mut failed: Vec<String> = Vec::new();

    // Topological sort by dependencies
    let sorted = topological_sort(certificates);

    for cert in sorted {
        // Check prerequisites
        let prereqs_met = cert.prerequisites.iter().all(|p| proven.contains(p));

        match &cert.result {
            ProofResult::Full { .. } if prereqs_met => {
                proven.insert(cert.id.clone());
                for implied in &cert.implies {
                    proven.insert(implied.clone());
                }
            }
            ProofResult::Conditional { conditions, .. } if prereqs_met => {
                proven.insert(cert.id.clone());
                // Mark implied as conditional
            }
            ProofResult::Partial { proved: sub_proved, .. } => {
                for p in sub_proved {
                    proven.insert(format!("{}.{}", cert.id, p));
                }
            }
            _ => {
                failed.push(cert.id.clone());
            }
        }
    }

    ProofChainResult { proven, failed }
}
```

- [x] **T3: Serialization and CI integration**

```rust
/// Save proof certificates as a verifiable artifact.
/// Format: JSON with blake3 checksum.
pub fn save_certificates(
    certificates: &[ProofCertificate],
    path: &Path,
) -> Result<blake3::Hash> {
    let json = serde_json::to_string_pretty(certificates)?;
    let hash = blake3::hash(json.as_bytes());
    std::fs::write(path, json)?;
    Ok(hash)
}

/// Load and verify proof certificates.
pub fn load_certificates(path: &Path) -> Result<Vec<ProofCertificate>> {
    let json = std::fs::read_to_string(path)?;
    let certs: Vec<ProofCertificate> = serde_json::from_str(&json)?;
    // Verify chain consistency
    let result = verify_proof_chain(&certs);
    if !result.failed.is_empty() {
        return Err(anyhow!("Proof chain broken: {:?}", result.failed));
    }
    Ok(certs)
}
```

---

### Phase 2: Hierarchical GOAT Proof Refactoring (T4–T6)

- [x] **T4: Define proof chains for existing GOAT pillars**

Map the 4 GOAT pillars to hierarchical proof chains:

```text
Pillar 1: Fourier Spatial AI
  ├── P1.1: Hash collision <5% → implies spatial discrimination
  ├── P1.2: Position invariance 100% → implies map-independence
  │     └── implies P1.3: Spatial consistency (no position bias)
  ├── P1.4: MMO scale (100 floors, etc.) → implies scalability
  └── P1.1 + P1.2 + P1.4 → implies P1.FULL: Fourier MCTS is spatially sound

Pillar 2: WASM Validators
  ├── P2.1: 0 critical mismatches (30K A/B) → implies deterministic correctness
  ├── P2.2: Latency 0.37-0.55µs/call → implies real-time feasibility
  │     └── implies P2.3: Production-viable (<1µs target)
  ├── P2.4: LoRA+WASM > LoRA alone (+31) → implies validator adds value
  └── P2.1 + P2.2 + P2.4 → implies P2.FULL: WASM validators are production-ready

Pillar 3: NPC Dialog
  ├── P3.1: Retrieval <5ms → implies knowledge access speed
  ├── P3.2: Full dialog turn <10ms → implies real-time conversation
  │     └── implies P3.3: Production-viable (<100ms target)
  ├── P3.4: 13 E2E tests pass → implies correctness
  └── P3.1 + P3.2 + P3.4 → implies P3.FULL: NPC dialog works modelless

Pillar 4: Frame-Sampling Bridge
  ├── P4.1: 3 POC examples pass → implies basic functionality
  ├── P4.2: Configurable ratio → implies tunability
  └── P4.1 + P4.2 → implies P4.FULL: Frame-sampling bridges AI to game loop
```

- [x] **T5: Implement proof certificate generation for one pillar (Pilot)**

Start with Pillar 2 (WASM Validators) as pilot — cleanest data, clearest properties:

```rust
#[cfg(feature = "proof_cert")]
pub fn generate_wasm_validator_certificates(
    n_comparisons: usize,
    mismatches: usize,
    latency_us: f64,
    lora_wasm_delta: i32,
) -> Vec<ProofCertificate> {
    vec![
        ProofCertificate {
            id: "P2.1".into(),
            property: ProofProperty::DeterministicCorrectness {
                game: "bomber".into(),
                n_comparisons,
            },
            result: ProofResult::Full {
                value: mismatches as f64,
                threshold: 0.0,
            },
            prerequisites: vec![],
            implies: vec!["P2.3".into()],
            explanation: format!(
                "Zero critical mismatches in {} A/B comparisons",
                n_comparisons
            ),
            evidence: ProofEvidence::Comparison {
                baseline: "native".into(),
                challenger: "wasm".into(),
                delta: mismatches as f64,
            },
            timestamp: Utc::now(),
        },
        // ... more certificates
    ]
}
```

- [x] **T6: Migrate remaining pillars to certificate format**

Refactor GOAT proofs in Plans 061, 034, 099, 070 to emit `ProofCertificate` structs instead of markdown threshold tables.

---

### Phase 3: Conditional Proof Support (T7–T8)

- [x] **T7: Conditional proof macro**

```rust
/// Macro for declaring conditional GOAT proofs.
/// Usage:
///   conditional_proof!(
///       "P5.1",
///       property = Convergence { algorithm: "bandit", metric: "win_rate" },
///       value = 0.65,
///       threshold = 0.60,
///       conditions = ["bandit_arms >= 8", "n_games >= 100"],
///       implies = ["P5.2"]
///   );
#[macro_export]
macro_rules! conditional_proof {
    (
        $id:expr,
        property = $prop:expr,
        value = $val:expr,
        threshold = $thresh:expr,
        conditions = [$( $cond:expr ),* $(,)?],
        implies = [$( $imp:expr ),* $(,)?]
    ) => {{
        ProofCertificate {
            id: $id.into(),
            property: $prop,
            result: ProofResult::Conditional {
                value: $val,
                threshold: $thresh,
                conditions: vec![$( $cond.into() ),*],
            },
            prerequisites: vec![],
            implies: vec![$( $imp.into() ),*],
            explanation: String::new(),
            evidence: ProofEvidence::Custom { data: serde_json::json!({}) },
            timestamp: chrono::Utc::now(),
        }
    }};
}
```

- [x] **T8: Proof chain CLI tool**

```bash
# Verify all GOAT proof certificates
cargo run --features proof_cert --bin goat-verify

# Output:
# ✅ P1.1: Hash collision <5% → 2.3% [PASS]
# ✅ P1.2: Position invariance 100% → 100.0% [PASS]
#   └── implies P1.3: Spatial consistency [DERIVED]
# ✅ P1.FULL: Fourier Spatial AI [CHAIN COMPLETE]
# ✅ P2.1: 0 critical mismatches → 0 [PASS]
# ⚠️ P2.5: Conditional — requires seed=42, board=9x9
# ...
```

---

### Phase 4: Percepta Mutation Strategy Benchmarking (T9–T10) — Super-GOAT

> ⚠️ This phase requires Plan 128 (proof sketch evolution) and Plan 143 (Nexus Elo).
> Feature-gate: both `proof_cert` and `percepta`.

- [x] **T9: Mutation strategy comparison framework**

Inspired by the paper's finding that different flux limiters have different proof success rates:

```rust
#[cfg(all(feature = "proof_cert", feature = "percepta"))]
pub struct MutationStrategyBenchmarker {
    strategies: Vec<MutationStrategy>,
    rater: PlackettLuceRating,
}

pub enum MutationStrategy {
    /// Conservative: only modify definitions (analogous to minmod limiter)
    DefinitionsOnly,
    /// Balanced: modify definitions + lemmas (analogous to MC limiter)
    DefinitionsAndLemmas,
    /// Aggressive: modify definitions + lemmas + values (analogous to superbee limiter)
    AllBlocks,
    /// Smooth: parameterized continuous mutation (analogous to van Leer limiter)
    Smooth { smoothness: f64 },
}

impl MutationStrategyBenchmarker {
    /// Benchmark all strategies and produce proof certificates
    /// showing which strategies prove which properties.
    pub fn benchmark(&mut self, n_rounds: usize) -> Vec<ProofCertificate> {
        // For each strategy, run N mutation rounds
        // Track: proof success rate, convergence speed, diversity
        // Produce conditional proof certificates:
        //   "Strategy X proves property Y with 90% success rate"
        //   "Strategy Z fails property W — simplification insufficient"
    }
}
```

- [x] **T10: Proof-success rate heatmap**

```rust
/// Generate a proof certificate showing which mutation strategies
/// successfully prove which properties — analogous to Tables 1-5 in the paper.
#[cfg(all(feature = "proof_cert", feature = "percepta"))]
pub fn mutation_proof_heatmap(
    results: &[(MutationStrategy, ProofProperty, ProofResult)],
) -> ProofCertificate {
    // Produce a certificate summarizing:
    // "minmod-like: 100% symmetry + TVD proof success"
    // "superbee-like: 100% TVD, 0% symmetry proof success"
    // "vanLeer-like: 100% symmetry, 0% TVD proof success"
}
```

---

## Feature Gates

```toml
[features]
default = []
proof_cert = ["serde", "serde_json", "chrono"]  # Phases 1-3
# Super-GOAT: requires both proof_cert and percepta
# Mutation strategy benchmarking (Phase 4) activates only when both are enabled
```

**Why feature-gated:**
- Proof certificates add serialization dependency (serde, chrono)
- Hierarchical proof chains are a methodology change, not a code change
- Phase 4 (Percepta mutation benchmarking) is the selling point — keep it opt-in until proven

---

## GOAT Proof Targets

| Target | Metric | Threshold |
|--------|--------|-----------|
| T3: Certificate serialization | Round-trip correctness | 100% (blake3 checksum match) |
| T5: WASM pillar certificates | All P2.* proofs certifiable | 5/5 certificates generated |
| T6: All 4 pillars certified | P1-P4 full chain certificates | 4/4 chains complete |
| T8: CLI verification tool | Chain verification accuracy | 100% (no false positives) |
| T10: Mutation benchmark | ≥2 strategies with >80% proof success | At least minmod-like + MC-like strategies pass |

---

## What This Is NOT

- ❌ Not a new game feature
- ❌ Not a GOAT pillar (per [decision matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md))
- ❌ Not model-based — entirely modelless proof methodology
- ❌ Not a symbolic algebra system — we use concrete benchmark data, not symbolic proofs
- ❌ Not a replacement for existing GOAT proofs — a structured enhancement

---

## What This Enables

- ✅ More diagnostic GOAT proofs (conditional results explain *why* something passes)
- ✅ CI-verifiable proof artifacts (JSON certificates instead of markdown tables)
- ✅ Proof chain validation (detect when upstream proof breakage cascades)
- 🔒 Super-GOAT: Percepta mutation strategy benchmarking (selling point, feature-gated)

---

## Module Structure

```
katgpt-rs-core/src/
├── proof_cert/                    # NEW
│   ├── mod.rs                     # Module root
│   ├── certificate.rs             # ProofCertificate, ProofResult, ProofEvidence
│   ├── chain.rs                   # verify_proof_chain, topological sort
│   ├── serde_impls.rs             # Serialization helpers
│   └── macros.rs                  # conditional_proof! macro
└── traits.rs                      # existing, unchanged

katgpt-rs/src/
├── proof_cert/                    # NEW
│   ├── mod.rs                     # Module root
│   ├── wasm_certificates.rs       # P2.* proof generation #[cfg(feature = "proof_cert")]
│   ├── fourier_certificates.rs    # P1.* proof generation #[cfg(feature = "proof_cert")]
│   ├── npc_certificates.rs        # P3.* proof generation #[cfg(feature = "proof_cert")]
│   └── frame_certificates.rs      # P4.* proof generation #[cfg(feature = "proof_cert")]
├── percepta/                      # existing, +mutation benchmarking
│   └── mutation_bench.rs          # NEW: #[cfg(all(feature = "proof_cert", feature = "percepta"))]
└── bin/
    └── goat_verify.rs             # NEW: CLI tool for proof chain verification
```

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| Plan 143 (Nexus Elo) | Provides search strategy (Plackett-Luce + P-UCB). This plan provides verification methodology. Together: search + verify. |
| Plan 128 (Proof sketch evolution) | Phase 4 of this plan benchmarks mutation strategies for Percepta sketches. Depends on Plan 128. |
| Plan 061 (Fourier MCTS) | P1.* certificates formalize Fourier MCTS GOAT proofs. |
| Plan 034 (WASM Validator) | P2.* certificates formalize WASM validator GOAT proofs. Pilot target. |
| Plan 099 (NPC Dialog) | P3.* certificates formalize NPC dialog GOAT proofs. |
| Plan 070 (Frame-Sampling) | P4.* certificates formalize frame-sampling GOAT proofs. |
| Plan 064 (Percepta) | Phase 4 requires Percepta feature gate. |

---

## References

- Research: [106 — Shock with Confidence](../.research/106_Shock_Confidence_Formal_PDE_Verification.md)
- Cross-ref: Research 088/104 (AlphaProof Nexus), Plan 143 (Nexus Elo Plackett-Luce P-UCB)
- [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md)
