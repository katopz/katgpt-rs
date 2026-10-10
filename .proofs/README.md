# KatgptProof — Lean 4 formal verification for the sigmoid ranking-preservation property

Second Lean 4 formal-verification instance in the 7-repo stack (katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train / riir-game-sdk / riir-armageddon), and the **first in the public MIT repo** (`katgpt-rs`). The first instance is `riir-chain/.proofs/RiirChainProof` (Plan 004 — LatCal fixed-point round-trip).

## What this proves

| File | Theorem | Statement |
|---|---|---|
| `Bridge/Basic.lean` | (spec only) | `dot` product over `ℝ` mirroring `ActionBridge::select_action`'s `mul_add` loop; sigmoid = Mathlib's `Real.sigmoid` |
| `Bridge/RankingPreserved.lean` | `action_bridge_ranking_preserved` | `∀ (q d₁ d₂ : ι → ℝ), dot q d₁ > dot q d₂ → sigmoid (dot q d₁) > sigmoid (dot q d₂)` |
| `Bridge/RankingPreserved.lean` | `action_bridge_argmax_preserved` | If `d₁` has the strictly largest dot product, it also has the strictly largest sigmoid score |
| `Ssmax/Basic.lean` | (spec only) | `alphaGold N c = 1 / (1 + (N−1)·N^(−c))` — the paper's dilution bound; `alphaGold_bounded` proves `(0,1)` |
| `Ssmax/DilutionBound.lean` | `alphaGold_strictMono_in_c` | For `N > 1`, `alphaGold` is strictly increasing in `c` — the monotonicity that makes SSMax work |
| `Ssmax/DilutionBound.lean` | `ssmax_dominates_base` | For `s_L · log(N) ≥ 1` and `c_base > 0`: SSMax does not decrease gold mass (threshold is `s_L·log(N)≥1`, NOT `N≥2`) |
| `Ssmax/Asymptotic.lean` | `tendsto_leakage_zero` | For `s_L · Δ > 0`: the leakage term `(N−1)·N^(−s_L·log N·Δ) → 0` as `N → ∞` (squeeze against `0 ≤ leakage ≤ 1/N`) |
| `Ssmax/Asymptotic.lean` | `tendsto_alphaGold_one` | **SSMax asymptotically defeats dilution**: for `s_L · Δ > 0`, `α_gold(N, s_L·log N·Δ) → 1` as `N → ∞` |
| `Hope/Basic.lean` | (spec only) | `reluSelfKernel γ β = (γ²+β²)·Φ(β/|γ|) + β·|γ|·φ(β/|γ|)` — HOPE Eq 3; `normalCdf` modeled as constant `1/2` (spec simplification, see file doc) |
| `Hope/SpecTests.lean` | (spec self-tests) | `reluSelfKernel(1,0) = 1/2`; `reluSelfKernel(γ,0) = γ²/2` for γ>0; `reluSelfKernel(γ,β) = reluSelfKernel(-γ,β)` (γ-sign symmetry) |
| `Pencil/Sym.lean` | `sym_isometry_norm_sq` | `‖sym(v)‖_F² = ‖v‖₂²` for the mirrored-√2 packing — the storage layout of `SymPacked` (Issue 678 T1; Research 495 / arXiv:2608.08003) |
| `Pencil/RayleighCF.lean` | `cf_ge` / `cf_dual` | Courant–Fischer min–max sandwiches, built from Mathlib's spectral theorem (Mathlib ships neither CF nor Weyl — the load-bearing machinery for T2–T4) |
| `Pencil/Weyl.lean` | `weyl_lipschitz` | `|λᵢ(A) − λᵢ(B)| ≤ ‖A−B‖₂` — eigenvalues are 1-Lipschitz in the spectral norm (T2; the paper's Cor. 1 global feature-influence bound rides on it) |
| `Pencil/Loewner.lean` | `loewner_mono` / `mirror_dual` | `B−A ⪰ 0 ⇒ λᵢ(A) ≤ λᵢ(B)` (T3; the shape DSL's soundness core) + `λⱼ(−A) = −λᵢ(A)` at `j = D−1−i` |
| `Pencil/Eigengap.lean` | `eigengap_ge_half` | a unit eigengap survives `+ E + s·I` with `‖E‖ ≤ ¼` keeping `≥ ½` (T4 analytic core; paper Lemma 2's shift + Weyl argument) |
| `Pencil/Eigengap.lean` | `eigval_diagonal_antitone` | the antitone-sorted eigenvalue array of a decreasing diagonal is the diagonal itself — the concrete-eigenvalue-pinning substrate (singles are exact eigenvectors; CF on coordinate spans) |
| `Pencil/Eigengap.lean` | `ladder_unit_gap` | the ladder `diag(1,…,0@k,…,−1)` has `λk − λk₊₁ = 1` exactly — Lemma 2's input gap pinned |
| `Pencil/Eigengap.lean` | `eigengap_ladder_ge_half` | **T4 final assembly**: the seeded pencil `ladder + E + s·1` keeps `≥ ½` of the unit gap under any Hermitian `‖E‖ ≤ ¼` — the paper's Lemma 2 complete |
| `ScaleInvariant/Basic.lean` | `scale_invariant_decade_mass_bounded` | the harmonic-sum interval bound (Plan 622 Phase 6): `ατ·log((TΔ+1+τ)/(T+1+τ)) ≤ Σ_{t=T+1}^{TΔ} mass ≤ ατ·log((TΔ+τ)/(T+τ))` for the boundary-pinned schedule (α = e^{1/2}); plus the Def-3.1 Θ(1) band legs — `decadeMass_le_logDelta` (≤ ατ·log Δ for every T), `decadeMass_ge_floor` + `decadeMassFloor_pos` (≥ the T-independent positive floor) |
| `ScaleInvariant/Sigmoid.lean` | `sigmoid_transfer_halves_constant` | the tilt transfer (Plan 622, the novel fusion half): under the schedule `E[σ(L_t)] = α·c(a_t)/(t/τ+1)` and the constant HALVES — `lim (t/τ+1)·E[σ(L_t)] = α/2`; the Gaussian-integral facts (the tilt rule, dominated convergence) are named hypotheses per the `Hope/Basic.lean` precedent, validated numerically by the Rust spec-match; `sigmoid_transfer_gap` (hypothesis-free) is the exact ε(t): `|…−α/2| = α·|c(a_t) − 1/2|` |
| `ScaleInvariant/Dominates.lean` | `scale_invariant_dominates_ssmax_on_decade_mass` | the comparator ordering (Plan 622): the schedule holds the T-independent floor while both position-decaying comparator families collapse — the ALiBi exponential shape (`C·q^t`) and the SSMax-dilution polynomial shape (`C·(1+t)^{−κ}`) both → 0; SSMax/ALiBi fail the Def-3.1 Θ(1) band |

The headline theorem is `action_bridge_ranking_preserved`: it proves that `ActionBridge::select_action`'s sigmoid projection preserves dot-product ordering. This is the ∀-form of the empirical `g1_3_bridge_ranking_preservation` test in `crates/katgpt-core/src/micro_belief/tests.rs` (Plan 281 G1.3), which samples only 1000 random triples. The Lean theorem holds for **every** triple.

## Why this exists

The bridge projects latent Q-values to raw action scores via `sigmoid(dot(q, direction))`. The whole point — per `AGENTS.md`'s latent-vs-raw rules — is that downstream consumers can rank entities by the *scalar* projection without needing the latent vector. This is only sound if sigmoid preserves the dot-product ordering, i.e. if sigmoid is strictly monotone.

Before this proof, that property was enforced by:
1. A doc comment ("never softmax").
2. The empirical G1.3 test (1000 random triples).

After this proof, it is enforced by a Lean theorem (over `ℝ`) plus a Rust spec-match test that fails CI if the Rust `fast_sigmoid` drifts from the Mathlib `Real.sigmoid` spec.

## Why Mathlib (and why the toolchain differs from riir-chain)

`RiirChainProof` (riir-chain Plan 004) deliberately avoids Mathlib: its theorem reduces to integer linear arithmetic, decided by Lean core's `omega` tactic, keeping `lake build` under 5 seconds with a pinned `leanprover/lean4:v4.31.0`.

`KatgptProof` cannot avoid Mathlib: sigmoid's strict monotonicity depends on the transcendental analysis of `exp` (`Real.exp`), which is not in Lean core. Mathlib ships `Real.sigmoid_strictMono` (in `Mathlib.Analysis.SpecialFunctions.Sigmoid`) — the exact lemma this proof needs. Adding Mathlib forces the toolchain to `leanprover/lean4:v4.32.0-rc1` (Mathlib's current requirement), which is *higher* than riir-chain's pinned version. This is an unavoidable consequence of depending on Mathlib for transcendental analysis. The first `lake build` downloads Mathlib's precompiled cache (8592 files from the lake cache server), so build time stays reasonable.

## How to run

```bash
# 1. Install Lean 4 toolchain (one-time, no root needed)
curl https://elan-init.lean-lang.org/elan-init.sh -sSf | sh

# 2. Build the proofs
cd katgpt-rs/.proofs
lake build

# 3. Verify the spec-match test on the Rust side
cd katgpt-rs
cargo test --features action_bridge --test bridge_spec_match
```

Both must pass for the proof to be valid:
- `lake build` proves the math (`Real.sigmoid_strictMono` ⟹ ranking preserved).
- `cargo test --features action_bridge --test bridge_spec_match` proves the Rust `fast_sigmoid` / `select_action` match the Lean spec.

If either fails, the proof is invalid.

## Axioms

All three theorems depend only on Lean's three standard foundational axioms:
- `propext` (propositional extensionality)
- `Classical.choice` (axiom of choice)
- `Quot.sound` (quotient soundness)

No `sorry`. No `sorryAx`. Verified by `#print axioms`. These are the same axioms Mathlib itself is built on.

## Layout

```
.proofs/
├── lakefile.toml              # Lean 4 build manifest (requires Mathlib)
├── lean-toolchain             # Pins Lean version (v4.32.0-rc1, Mathlib's requirement)
├── .gitignore                 # .lake/, lake-manifest.json
├── README.md                  # this file
└── KatgptProof/
    ├── Bridge/
    │   ├── Basic.lean                  # Spec: dot product + sigmoid (Mathlib's Real.sigmoid)
    │   ├── RankingPreserved.lean        # Theorems: ranking + argmax preservation (Plan 293)
    │   └── SpecTests.lean               # Spec self-tests on concrete dot/sigmoid instances (Plan 441)
    ├── Hope/
    │   ├── Basic.lean                   # Spec: ReLU self-kernel (HOPE Eq 3, Plan 469)
    │   └── SpecTests.lean               # Spec self-tests on concrete kernel instances (Plan 441/469)
    └── Ssmax/
        ├── Basic.lean                   # Spec: alphaGold dilution bound + monotonicity of N^c (Plan 411 S3)
        ├── DilutionBound.lean            # Theorems: alphaGold strictMono in c + ssmax_dominates_base (Plan 411 S3)
        ├── Asymptotic.lean               # Theorem: alphaGold → 1 as N → ∞ (Plan 411 S3 asymptotic follow-up)
        └── SpecTests.lean                # Spec self-tests on concrete dilution-curve instances (Plan 441)
    └── Pencil/                            # Spectral pencil (Issue 678 / Research 495, arXiv:2608.08003)
        ├── Sym.lean                       # T1: the sym-√2 packing isometry
        ├── RayleighCF.lean                # Courant–Fischer core (built from Mathlib's spectral theorem)
        ├── Weyl.lean                      # T2: Weyl 1-Lipschitz
        ├── Loewner.lean                   # T3: Loewner monotonicity + mirror duality
        ├── Eigengap.lean                  # T4: shift lemma + antitone-diagonal pinning + ladder unit gap + the final assembly
        └── SpecTests.lean                 # Spec self-tests: T1 packing, T2 Weyl tightness, T3 Loewner, T4 exact perturbed gap
    └── ScaleInvariant/                       # Scale-invariant attention (Plan 622 / Research 610, arXiv:2505.17083)
        ├── Tendsto.lean                      # Shared explicit-bound Tendsto helpers (atTop algebra, no lemma-name roulette)
        ├── Basic.lean                        # Schedule closed forms + the harmonic-sum interval bound + the Def-3.1 Θ(1) band
        ├── Sigmoid.lean                      # The tilt transfer: the halving to α/2 + the exact ε(t) gap
        ├── Dominates.lean                    # The comparator ordering: ALiBi/dilution families → 0 vs the schedule's floor
        └── SpecTests.lean                    # Spec self-tests: boundary pins, the t = τ paper cell, σ(0) = 1/2, the empty decade, the antitone profile
```

## The f32 caveat (and why it doesn't break the theorem)

The Lean theorem is stated over `ℝ` (infinite precision). The Rust `fast_sigmoid` is an `f32` approximation:
- For `|x| > 40`: saturates to exactly `0.0` or `1.0`.
- Near `±18`: f32's ~6e-8 spacing near 1.0 causes distinct dot products to map to the *same* f32 sigmoid value (a tie).

Neither affects the theorem's validity:
- **Saturation ties** are consistent — the bridge breaks them by first-wins insertion order. No action can *outrank* another via sigmoid that didn't already win via dot product.
- A genuine **flip** (larger dot → strictly smaller sigmoid) would violate the theorem and is caught by the `empirical_ranking_preserved_within_f32_precision` spec-match test.

## Regenerating after bridge changes

If `katgpt-rs/crates/katgpt-core/src/bridge/mod.rs::select_action` or `simd/activations.rs::fast_sigmoid` changes:
1. If the projection is no longer `sigmoid` (e.g. swapped to softmax), the Lean theorem is invalid — the bridge must keep using a strictly-monotone function.
2. If `fast_sigmoid`'s mathematical definition changes, update `Bridge/Basic.lean`'s doc to match.
3. Run `lake build` — the theorem should still hold (it depends only on `Real.sigmoid`'s monotonicity, not the Rust implementation).
4. Run `cargo test --features action_bridge --test bridge_spec_match` — the spec-match tests must still pass.

## Cross-references

- Plan: `.plans/293_action_bridge_lean4_monotonicity_proof.md`
- Research: `.research/292_Bridge_Neuro_Symbolic_Formal_Verification_Gap.md`
- Plan: `.plans/411_ssmax_goldshare.md` (Stretch S3 — SSMax dilution-bound)
- Research: `.research/392_*` (SSMax / GoldShare distillation)
- Sibling instance (Tier 1): `riir-chain/.proofs/RiirChainProof` (Plan 004 — LatCal round-trip, Mathlib-free)
- Empirical test (complementary ∃-check): `crates/katgpt-core/src/micro_belief/tests.rs::g1_3_bridge_ranking_preservation`
- Rust implementation: `crates/katgpt-core/src/bridge/mod.rs::ActionBridge::select_action`, `crates/katgpt-core/src/simd/activations.rs::fast_sigmoid`
- Rust implementation (SSMax): `crates/katgpt-core/src/ssmax.rs::apply_ssmax_inplace`, `SsmaxMode`

## Status

**Phase 1–3 of Plan 293: COMPLETE.** All gates pass:

- **G1** (toolchain bootstraps): ✅ `lake build` succeeds, Lean `v4.32.0-rc1` + Mathlib.
- **G2** (theorem type-checks): ✅ `action_bridge_ranking_preserved` + `action_bridge_argmax_preserved` + `action_bridge_ranking_preserved'` all compile, no `sorry`, axioms = `{propext, Classical.choice, Quot.sound}`.
- **G3** (Rust spec matches Lean): ✅ `cargo test --features action_bridge --test bridge_spec_match` — 6/6 tests pass.

**Plan 411 S3 (SSMax dilution-bound theorems): COMPLETE.** Added 2026-07-07.

- **G1** (Lean builds): ✅ `lake build` succeeds — `Ssmax/Basic.lean` + `Ssmax/DilutionBound.lean` + `Ssmax/Asymptotic.lean`.
- **G2** (theorem type-checks): ✅ `alphaGold_strictMono_in_c` + `alphaGold_lt_of_c_lt` + `ssmax_dominates_base` + `alphaGold_bounded` + `tendsto_leakage_zero` + `tendsto_alphaGold_one` all compile, no `sorry`, axioms = `{propext, Classical.choice, Quot.sound}`.
- **G3** (Rust spec matches Lean): ✅ `cargo test --features ssmax_temperature --test ssmax_spec_match` — 8/8 tests pass (now runs by default since `ssmax_temperature` is Phase 13 DEFAULT-ON).

**Issue 678 (Pencil spectral package): COMPLETE 2026-08-22.** All four
theorems + the concrete-instance spec tests landed:

- **G1** (Lean builds): ✅ 6-module Pencil instance; 38 audited theorems
  (17 pre-Pencil → 35 core → 38 with the ladder closeout), all within
  `{propext, Classical.choice, Quot.sound}`, zero `sorry`.
- **G2** (negative tests): ✅ 6/6 simulated spec typos caught
  (`scripts/proof_negative_test.sh`) — including the ladder closeout's
  P5 (wrong expected gap) + P6 (negated diagonal pin).
- **G3** (Rust spec matches Lean): ✅ `cargo test -p katgpt-core --features
  spectral_pencil --test pencil_spec_match` — 7/7 (T1 ×3, T2 Weyl,
  T4 shift, T4 ladder-exact, T4 seeded-jitter).

**Plan 622 Phase 6 (ScaleInvariant — the Lean trio): COMPLETE 2026-10-11.**

- **G1** (Lean builds): ✅ 5-module ScaleInvariant instance (Tendsto helpers
  + Basic + Sigmoid + Dominates + SpecTests); `EXPECTED_THEOREMS` 39 → 53
  (14 audit heads), `proof_gate.sh` **PASSED** — 53 audited, all within
  `{propext, Classical.choice, Quot.sound}`, zero `sorry`, no
  `native_decide`.
- **G2** (honesty of the spec): the Gaussian-integral facts (the exponential
  tilt rule; the dominated-convergence limit `c → 1/2`) are NAMED
  HYPOTHESES of `sigmoid_transfer_halves_constant` per the `Hope/Basic.lean`
  precedent — the proven/assumed split is documented in the file headers,
  never silently assumed. Everything schedule-side is proven.
- **G3** (Rust spec matches Lean): ✅ `cargo test -p katgpt-core --features
  scale_invariant_attn --test scale_invariant_spec_match` — 6/6 (closed
  forms + boundary pins, the harmonic profile, the interval bounds + band +
  floor, the tilt identity + exact gap + the monotone drift, the comparator
  ordering, the `.proofs/` sentinel). The Rust side validates the Lean
  hypotheses' Gaussian-integral facts numerically (adaptive quadrature,
  deterministic grids, no RNG). 2151/0 default-posture G3 unchanged;
  2171/0 wired-posture; clippy 0 at the feature posture.

**The Floor algebra sharpened nothing but confirmed the shape**: the
T-independent floor's positivity reduces to `(1+τ)(T−1)(Δ−1) ≥ 0` — nonneg
exactly on the `T ≥ 1, Δ ≥ 1` quadrant, the same tight-threshold shape the
SSMax proof found (`s_L·log N ≥ 1`, not `N ≥ 2`).

**The headline of the closeout:** Mathlib ships the spectral theorem but
neither Courant–Fischer nor Weyl — both were built from scratch (the CF
sandwiches on the subspace dimension argument), and the ladder-value
pinning rides a general substrate: `eigval_diagonal_antitone` (a
decreasing diagonal IS its own antitone eigenvalue array, via CF on
coordinate spans — the standard singles are the exact eigenvectors of a
diagonal). The Rust sort-order correspondence (ascending Jacobi vs
antitone Lean) is pinned in the spec-match header.

**The formal-verification value-add:** the Lean proof *sharpened the plan's
threshold*. Plan 411 S3 sketched "`s_L = 1, N ≥ 2 ⇒ SSMax ≥ base`", but the
correct condition is `s_L · log(N) ≥ 1` (i.e. `N ≥ 3` for `s_L = 1`, since
`log(2) ≈ 0.693 < 1`). The `ssmax_dominates_base` theorem establishes the
tight threshold; the `spec_threshold_is_s_l_times_log_n_geq_one_not_n_geq_two`
Rust test guards both the dominance (for `N ≥ 3`) and the reversal at `N = 2`.

**Asymptotic complement** (`Ssmax/Asymptotic.lean`, added 2026-07-07): the
finite-N theorems above show SSMax helps at every fixed `N ≥ 3` (for `s_L = 1`);
the asymptotic theorem `tendsto_alphaGold_one` shows SSMax *completely defeats*
dilution in the large-`N` limit — `α_gold(N, s_L·log N·Δ) → 1` as `N → ∞` for
any `s_L · Δ > 0`. Proof via squeeze: `0 ≤ leakage N ≤ 1/N` eventually (where
`leakage = (N−1)·N^(−s_L·log N·Δ)`), and `1/N → 0`. The key rate comparison is
that `(log N)²` dominates `log N`, so `N^(−s_L·log N·Δ) = exp(−s_L·Δ·(log N)²)`
collapses super-polynomially.

Verified by:
```bash
cd katgpt-rs/.proofs && lake build    # → Build completed successfully (2281 jobs)
cd katgpt-rs && cargo test -p katgpt-core --test ssmax_spec_match  # → 8 passed
cd katgpt-rs && cargo test -p katgpt-core --test bridge_spec_match --features action_bridge  # → 6 passed
```

## Spec self-tests (Plan 441 — the "spec tested on vectors" convention)

Each `Basic.lean` spec ships a paired `SpecTests.lean` with concrete-instance
`example` proofs testing the spec against independently-known values. This is
distilled from SymCrypt `feature/verifiedcrypto` §4 ("Running the Lean spec on
test vectors") and closes the spec-authority gap in C3:

- The proof theorems prove the spec against itself — they don't catch spec
  authoring errors.
- The Rust spec-match test tests Rust against the spec — if both have the
  same typo (written by the same author), the test passes.
- The spec self-test tests the spec against known-good values from the source
  paper or mathematical definition — the independent authority.

**The sign-typo case study (Plan 441 G2):** if `alphaGold`'s `N^(-c)` is
mistranscribed as `N^c`, all 5 Ssmax spec test examples fail at `lake build`
time. The monotonicity theorems still type-check (they prove monotonicity of
whatever formula is transcribed); the Rust spec-match test still passes (Rust
likely has the same typo). Only the spec self-test catches it.

## License

Same as the rest of `katgpt-rs` — MIT (public).
