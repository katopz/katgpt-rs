# Plan 298: SmearClassifier — GOAT Gate Results (Phase 1 + 3)

**Date:** 2026-06-21
**Plan:** [katgpt-rs/.plans/298_smear_aware_faithfulness_probe.md](../.plans/298_smear_aware_faithfulness_probe.md)
**Research:** [katgpt-rs/.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md](../.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md)
**Source paper:** [arXiv:2606.20560](https://arxiv.org/abs/2606.20560) — Engels et al., "How Transparent is DiffusionGemma?", DeepMind, Jun 2026
**Private guide (riir-ai):** [129_Cognitive_Integrity_Layer_Guide.md](../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md) (T4.3 cross-repo update still TODO)
**Hardware:** Apple Silicon arm64 (M-series), release build.

---

## Phase 1 — Correctness (G1)

### Unit Tests: 6/6 PASS

`cargo test -p katgpt-core --features smear_classifier smear`

| Test | Description | Result |
|---|---|---|
| `coherent_single_one_dominant_direction` | single non-zero hypothesis → `CoherentSingle` | ✅ PASS |
| `token_smear_parallel_directions_across_sites` | 3 parallel rows (cosine=1.0) → `TokenSmear { span: 3 }` | ✅ PASS |
| `sequence_smear_orthogonal_directions_one_site` | 2 orthogonal rows (cosine=0) → `SequenceSmear { n_hypotheses: 2 }` | ✅ PASS |
| `epsilon_filters_low_norm_hypotheses` | sub-ε norms dropped before classification | ✅ PASS |
| `tau_same_boundary` | distance exactly at τ_same → `TokenSmear` (inclusive `<=`) | ✅ PASS |
| `deterministic_for_fixed_input` | same input → bit-identical `SmearReport` | ✅ PASS |

### G1 (determinism + correctness): ✅ PASS

Covered by Phase 1 unit tests. No additional Phase 3 work needed.

---

## Phase 3 — Useful Discrimination (G2) + Latency (G3)

### G2 — Ternary classification predicts audit outcome ✅ PASS

`cargo test --features smear_classifier --test bench_298_smear_classifier_goat -- --nocapture`

**Workload:** 1000 random memory trials per smear class. Each trial:
1. Build a `[k*d]` row-major weight slice matching one of the three smear classes (CoherentSingle / TokenSmear / SequenceSmear).
2. Construct a synthetic consumer whose behavior = `(1/k) · Σ_i dot(memory, h_i)`.
3. Run the Plan 278 `faithfulness_profile` audit suite (5 interventions) with threshold 0.5.
4. Record whether `is_faithfully_used(0.5)` returns `false` (unfaithful).

**Parameters:** k=8, d=16, threshold=0.5 (Plan 278 default — NOT tuned per smear class).

| Smear class | Unfaithful count | Rate |
|---|---|---|
| `CoherentSingle` | 1000/1000 | 1.0000 |
| `TokenSmear` | 443/1000 | 0.4430 |
| `SequenceSmear` | 936/1000 | 0.9360 |

**SequenceSmear / TokenSmear ratio = 2.11× ≥ 2.0× target. ✅ PASS**

**Mechanism:** the synthetic consumer's effective readout direction is the average `(1/k) Σ h_i` over its claimed hypotheses.
- **TokenSmear**: all `h_i` are the same direction → effective norm = `‖h_0‖ = 1` → strong behavioral signal → audit perturbations reliably move behavior → lower unfaithfulness rate.
- **SequenceSmear**: `h_i` are pairwise orthogonal → effective norm = `√k / k = 1/√k ≈ 0.354` for k=8 → weaker behavioral signal → audit perturbations are diluted across k orthogonal directions → higher unfaithfulness rate.
- The norm ratio `√k = 2.83×` for k=8 produces an unfaithfulness rate ratio of 2.11×, exceeding the 2.0× threshold.

`CoherentSingle` at 100% unfaithful is expected: effective norm is `1/k = 0.125` (only one of k hypotheses nonzero), well below the audit threshold. This class is outside the G2 criterion (which only compares SequenceSmear vs TokenSmear) but confirms the mechanism scales monotonically with effective norm.

### G3 — Latency ✅ PASS

`cargo bench --features smear_classifier --bench smear_classifier_bench`

Measures `CosineSmearClassifier::classify` on full-rank `[k*d]` weights (all k rows significant, all k*(k-1)/2 pairs evaluated — the worst case). 1000 warmup iters + 100000 timed iters per combo. Apple Silicon arm64 SIMD.

| k | d | ns/op | Verdict |
|---|---|---|---|
| 2 | 8 | 12.8 | — |
| 2 | 16 | 13.0 | — |
| 2 | 32 | 13.5 | — |
| 4 | 8 | 30.9 | — |
| 4 | 16 | 31.5 | — |
| 4 | 32 | 35.0 | — |
| 8 | 8 | 95.5 | — |
| 8 | 16 | 99.8 | — |
| **8** | **32** | **107.6** | **✅ PASS** |

**k=8, d=32 at 107.6 ns/op ≤ 200 ns target. ✅ PASS**

The `simd_dot_f32` inner products dominate the cost. For k=8, d=32: 8 self-dots (norms) + 28 pairwise dots = 36 dot products of dimension 32 = 1152 muladds, which at ~10 GFLOP/s (NEON) gives ~115 ns theoretical floor. The measured 107.6 ns is at the floor — the implementation is SIMD-bound, not allocation/call-overhead-bound. This is plasma-tier: the classifier is cheaper than one matvec row.

**Debug-mode caveat:** `cargo bench` runs the release profile by default, so the G3 number above is authoritative. If the bench is invoked through `cargo test --features smear_classifier --bench smear_classifier_bench` (debug profile), the bench detects `cfg!(debug_assertions)` and scales the threshold 5× (200 → 1000 ns) with a clear banner — debug builds don't engage SIMD and the 200 ns plasma target is unreachable without it. The debug-scaled verdict is necessary-but-not-sufficient; the authoritative gate is the release number above.

---

## Phase 3 Exit: ✅ ALL GATES PASS

### GOAT Gate Decision (T3.4)

| Gate | Result | Action |
|---|---|---|
| G1 | ✅ 6/6 correctness + determinism tests pass | — |
| G2 | ✅ SequenceSmear/TokenSmear ratio 2.11× ≥ 2.0× | — |
| G3 | ✅ k=8 d=32 at 107.6 ns ≤ 200 ns | — |

**Decision:** **`smear_classifier` → OPT-IN (unchanged).** Both G2 and G3 pass. The classifier is correct (G1), useful (G2 — produces measurably different downstream decisions than the binary probe), and fast (G3 — plasma-tier latency). However, it stays opt-in because:

1. **It's a diagnostic, not a hot-path component.** The smear report enriches the audit stream; it does NOT change the inject/skip decision of `TriggeredInjectionGate` (which remains the source of truth, Plan 278 default-on).
2. **The G2 evidence is synthetic.** The 2.11× ratio is proven on a constructed workload where the mechanism (effective-norm dilution under orthogonal superposition) is mathematically clean. Real-workload evidence — does the ternary classification actually improve Cognitive Integrity Layer decisions on live NPC cognition? — requires riir-ai Plan 308 integration, which is out of scope for this plan (T4.3).
3. **No default-on promotion without real-workload proof.** Per the user's "find GOAT" rule: synthetic GOAT is necessary but not sufficient for default-on. The trigger for promotion is riir-ai reporting back that the `SequenceSmear` flag correlated with actual unfaithfulness events in production.

The classifier is ready for integration. Downstream consumers (riir-ai Cognitive Integrity Layer, anti-cheat, sync integrity) can opt in via `smear_classifier = ["katgpt-core/smear_classifier"]` and wire `CosineSmearClassifier` into `DefaultFaithfulnessProbe::with_smear_classifier`.

### Feature Structure (post-GOAT)

- `smear_classifier` (opt-in, depends on `faithfulness_probe`): gates `src/faithfulness/smear.rs` (the standalone classifier) AND the `smear: Option<Box<dyn SmearClassifier>>` field + `probe_intervention_full` / `faithfulness_profile_full` methods on `DefaultFaithfulnessProbe` (the Phase 2 integration).
- When the feature is off, ALL smear-aware symbols vanish from `DefaultFaithfulnessProbe` (zero-overhead-off, preserves Plan 278 G8).

---

## Cross-References

- **Plan:** [298_smear_aware_faithfulness_probe.md](../.plans/298_smear_aware_faithfulness_probe.md)
- **Research:** [277_DiffusionGemma_Transparency_Smearing_Faithfulness.md](../.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md)
- **Host plan (FaithfulnessProbe):** [278_faithfulness_probe_modelless.md](../.plans/278_faithfulness_probe_modelless.md)
- **Prior GOAT benchmark:** [278_faithfulness_probe_goat.md](../.benchmarks/278_faithfulness_probe_goat.md)
- **Private guide (riir-ai):** [129_Cognitive_Integrity_Layer_Guide.md](../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md) — T4.3 cross-repo vocabulary update still TODO
- **Superposition sources this classifier consumes:**
  - Plan 178 (MUX — `crates/katgpt-core/src/mux/`)
  - Plan 281 (BoMSampler — `crates/katgpt-core/src/micro_belief/bom.rs`)
- **Source paper:** [arXiv:2606.20560](https://arxiv.org/abs/2606.20560)
- **Implementation:**
  - `crates/katgpt-core/src/faithfulness/smear.rs` (Phase 1 — standalone classifier)
  - `crates/katgpt-core/src/faithfulness/probe.rs` (Phase 2 — `SmearSource` trait + `InterventionOutcome` + `FaithfulnessProfileFull` + `with_smear_classifier` + `probe_intervention_full` + `faithfulness_profile_full`)
- **Tests:** `tests/bench_298_smear_classifier_goat.rs` (G2)
- **Bench:** `benches/smear_classifier_bench.rs` (G3)
- **API docs:** [`.docs/faithfulness_probe.md`](../.docs/faithfulness_probe.md)

## TL;DR

**All GOAT gates pass.** G1 (6/6 correctness + determinism) ✅. G2 (SequenceSmear/TokenSmear unfaithfulness ratio 2.11× ≥ 2.0× on 3000 synthetic trials) ✅. G3 (k=8 d=32 at 107.6 ns ≤ 200 ns on Apple Silicon arm64) ✅. **`smear_classifier` stays opt-in** — it's a correct, useful, fast diagnostic, but default-on promotion requires real-workload evidence from riir-ai Plan 308 integration (T4.3, out of scope). The classifier is ready for downstream consumers to wire into `DefaultFaithfulnessProbe::with_smear_classifier`.
