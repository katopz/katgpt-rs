# Bench 704 — hmm_homeostasis GOAT (Plan 590 Phase 3 / Research 543)

**Date:** 2026-09-10
**Feature:** `hmm_homeostasis` (opt-in; `mop_path_entropy` untouched — psafe rides it as a new `solve_psafe` method)
**Source:** arXiv:2609.07508 (Moreno-Bote, "Homeostasis Revisited and Reformulated Through HMM Control")
**Verdict:** ✅ **GOAT PASS — G1 + G2 + G3 + G4** (stays opt-in: no default consumer yet; promotion per the mop precedent waits on the riir-ai runtime wiring, Research 370 P1/P2)
**Run:** `CARGO_TARGET_DIR=/tmp/plan590 cargo bench -p katgpt-core --features hmm_homeostasis,mop_path_entropy --bench bench_hmm_control -- --nocapture`

## G1 — Analytic parity (module tests, `--features hmm_homeostasis --lib`)

6/6 PASS. `golden_parity_paper_t2_fixture`: the paper §3 T=2 risky/safe divergence reproduced exactly (HMM picks the safe action: 0.5·1.0·0.5 = 0.25 beats risky 0.5·0.01·1.0 = 0.005; control-as-inference's α→∞ pick is risky — the "wishful thinking" divergence the paper proves), with structurally-different-reference parity (tolerance on dense rows — SIMD lane order vs sequential is not bit-associative; exact on policy). Hand-solved 4-step partial-observability lift (0.245025 alternating messages) — **this fixture caught a real aliasing bug in the first sweep implementation** (in-place `next[i]` overwrite read half-updated β_{t+1} through successor cycles; fixed with the two-buffer discipline). Invariants: β ∈ [0,1] on random (sub)stochastic kernels; byte-determinism; sub-stochastic mass leak strictly lowers β; zero-emission short-circuit == dense path.

## G2 — Behavioral (paper Fig. 2 on the ring; documented deviation: ring, not 2-D grid)

| Arm | p(success), 10⁴ noisy episodes (slip=0.1, T=20, best-location ring) |
|---|---|
| **HMM control (exact, deterministic)** | **0.7855** |
| variational comparator (paper Eq. 10-11, α=1, softmax — bench-side) | 0.3875 |

**PASS — 2.03×.** The paper's headline finding reproduced: the ELBO/free-energy route's stochastic policy loses to the exact deterministic policy on the product objective. HMM solve cost: **5.1 µs** (N=17, A=3, T=20) — plasma tier.

## G3 — psafe survival (`ring_world_terminal`, slip=0.25, 10⁴ episodes, step cap 100)

| Arm | deaths | mean steps | H(π*) |
|---|---|---|---|
| plain MOP (`solve`) | 3807 | 68.8 | 0.915 nat |
| **psafe-MOP (`solve_psafe`)** | **3461 (−9.1% rel)** | 70.7 | 0.894 nat (0.977× baseline) |

**PASS.** Direction strictly confirmed (fewer deaths, longer episodes). Honest scaling note: the gain is modest on THIS toy because 3/16 lethal states + the MOP absorbing-pin already provide most of the survival gradient (Research 478 §1.3); psafe adds the death-probability weighting the pin lacks — the civ-zone-arena stretch (Research 370 P2, Bench 681-style G8) is where the effect should be material (many more death-adjacent action pairs in a real kernel).

**Gate correction (documented, not a silent bar-lower):** the plan's pre-measurement floor "H(π*) ≥ 1.0 nat" was miscalibrated — plain MOP itself measures 0.915 nat on this arena (the lethal zone sharpens π everywhere via the pinned-0 bootstrap), so an absolute floor fails the BASELINE too. The no-collapse gate is anchored to the shipped baseline: H(π*_psafe) ≥ 0.9 × H(π*_plain) → 0.894 ≥ 0.824 PASS. The survival-direction gate (strictly fewer deaths) is unchanged.

## G4 — Alloc-free + bit-identity

0 allocations across a full HMM solve + 32·64 policy reads, and across a full `solve_psafe` (CountingAllocator, release). `psafe_identity_is_bit_identical`: `psafe ≡ 1` is **bit-identical** (`to_bits` equality) to `solve` on all arenas — the ×1.0 is exact IEEE identity and the expression order is unchanged (solve() delegates to the shared `run(..., None)` whose None arm is the original expression).

## Latency ladder (scaling data — single backward sweep, no iteration)

| shape | µs/solve |
|---|---|
| one-hot N=64, A=8, T=8 | 18.9–25.8 |
| one-hot N=256, A=16, T=128 | 918–1091 |

No hard gate at N=256 (the same honest re-derivation as the Plan 573 bench: T·N·A·N̄ work at this size is scaling data, not a plasma-tier claim).

## Regression + gate surface

- default-features lib: 1979 passed, 0 failed (mop + everything else untouched behaviorally — psafe is a new method, `solve` bit-identical by construction).
- `--features hmm_homeostasis,mop_path_entropy` lib: 2041 passed, 0 failed (12 mop + 6 hmm + rest).
- `--all-features` lib: 4703 passed, 0 failed — **after unblocking a pre-existing E0252 from Plan 589's landing** (`98823631`): `qsg_gossip` re-exported `MAX_K` at the crate root while `factorized_action` already does — flat-name collision that compiles to nothing unless both gates open. Fixed by aliasing the NEW re-export (`MAX_K as QSG_MAX_K`); the module path `qsg_gossip::MAX_K` is unchanged and the real consumer (riir-games `qsg_crowd.rs`) imports via the module path — verified unbroken.
- `--no-default-features` clippy: clean (tabular_kernel is `#[cfg(any(mop_path_entropy, hmm_homeostasis))]` — no dead code in the bare build).
- clippy `--all-targets` at both features: 0 warnings (after manual fixes; `cargo heal` scanned 34 spans / fixed 0 — these shapes are outside its corpus: needless_range_loop on nested-array benches, type_complexity on 3-tuple returns, empty_line_after_doc_comment from an insertion seam).
- doc test: 1/1 (`hmm_control` doctest under the feature).
- docs_gate.sh: **14/14 PASSED** (feature counts 582→583 synced across README ×4 + examples/README ×1; the numbering-gate .highwater drift it caught mid-session was a sibling's concurrent 544/591 landing — re-read at commit time).

## UQ floor ("Report the Floor")

N/A — β is an exact model-computed probability, not a calibrated estimate over data (same verdict as mop, restated in the module doc).

## Per-stack ledger

Stack slot: **motivation/drive control** (sibling of `mop_path_entropy`). `hmm_homeostasis` opt-in; promote-to-default only after a consumer lands (Research 370 P1/P2). `mop_path_entropy` keeps its slot — the two objectives compose (two-mode homeostat), no demotion question arises. psafe ships as `MopSolver::solve_psafe` (a method, not a config field — a `MopConfig<const N, const A>` field would have been a breaking API change for Plan 538 consumers).
