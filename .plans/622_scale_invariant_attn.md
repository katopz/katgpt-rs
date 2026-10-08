# Plan 622: Scale-Invariant Attention — `scale_invariant_attn`

**Status:** Planned — GOAT verdict (Research 610; verdict ping-pong AGREE 2026-10-08, 4 amendments adopted). Opt-in `scale_invariant_attn` until G1–G4 pass. Track (a) modelless, PRIMARY. Training track: riir-train Plan 449 (SECONDARY, owns the pp-RoPE quality claim).

Source: arXiv:2505.17083 v2 (Anson/Wang/Aitchison, NeurIPS 2025). Distillation: `.research/610_Scale_Invariant_Attention.md`. Signal-diff vs shipped `ssmax.rs` (Plan 411): position-DEPENDENT per-distance (a_t, m_t) vs position-INDEPENDENT scalar s_L·log(N) — real delta, not covered.

## Non-negotiable laws (from the verdict review)

- **Ordering law:** when `scale_invariant_attn` is active, the SSMax length multiplier is BYPASSED. Both are length temperatures; stacking double-sharpens. A G1 test pins the mutual exclusion.
- **Tilt-transfer G1 law:** the sigmoid arm's expected mass constant is `E[σ(−L′_t)]`, `L′ ~ N(1, a_t²)` = `Φ(−1/a_t)` + O(a_t⁻³). G1 asserts `|E_GH − Φ(−1/a_t)| ≤ C·a_t⁻³` with C fixed ONCE in the test. NEVER a one-sided `1/(a√2π)` bound (loose at small t); NEVER pin the illustrative ≈0.29 / ≈0.41 values — derive from the formula at test time. The drift toward α/2 is slow (O(1/√log t)) and EXPECTED.
- **Calibration precondition:** the schedule assumes ~unit-variance base logits (paper's models carry QK-norm). Raw-logit checkpoints get a training-free per-head calibration (measure logit spread once at load → frozen BLAKE3-checked per-head scale table → normalize before the schedule). G3 is SPLIT: "calibrated zero-shot" and "uncalibrated zero-shot" are separate rows, never pooled.
- **pp-RoPE is KERNEL-ONLY here.** Dropping low-freq RoPE components on a full-RoPE checkpoint changes the trained position representation — zero-shot quality with pp-RoPE is EXPECTED TO FAIL and the G3 row says so in advance (recorded expectation, not a surprise to be explained later). The pp-RoPE quality claim lives in riir-train Plan 449.
- **Sink carve-out:** the BOS/sink token is EXCLUDED from the transform (the paper's Gaussianity premise excludes it; App. J).
- **Zero-alloc:** preallocated interleaved (a_t, m_t) LUT at session start; read-only hot path; unit-stride descending reads (contiguous distance slice per query); counting-allocator canary.

## Phase 1 — LUT + primitive (`katgpt-core/src/scale_invariant.rs`)

- [ ] `ScaleInvariantLut { am: Vec<f32>, tau: f32 }` interleaved a_t|m_t, `build(tau, max_ctx)` with the β ≥ α·log α validity assert; constants α=β=e^0.5 derived from the (a₀², m₀)=(1,0) boundary pin, not hard-coded.
- [ ] `apply_inplace(&self, scores: &mut [f32], query_i, key_start)` — distance-indexed affine `s → a_t·s + m_t` over the contiguous reversed slice; chunked 8-wide for LLVM auto-vectorization (ssmax.rs house pattern).
- [ ] Feature `scale_invariant_attn` (opt-in); lib.rs module doc states the ordering law + sink carve-out.
- [ ] Boundary/safety tests: a₀²=1, m₀=0 exact; near-identity ramp for t ≤ τ; no NaN for t up to 10⁶.

## Phase 2 — Attention wiring (both arms)

- [ ] `attention.rs` SDPA: per-key affine hook in the pre-softmax score path, before online-max tracking (FlexAttention score_mod shape); flag-off = bit-identical.
- [ ] `parallax_attn`: a_t rides a new per-key MULTIPLICATIVE lane (sibling of the existing gated `prior_logit_lane` additive lane); m_t reuses the additive lane shape; both arms (softmax + normalized sigmoid) consume the same LUT.
- [ ] Sigmoid-arm mass assertion (unit test): measured E[σ(L_t)]·(t/τ+1)/α equals the Gauss–Hermite E[σ(−L′_t)] within 1e-6 (algebra identity).
- [ ] SSMax mutual-exclusion: enabling both together is a loud config error (or the schedule bypasses SSMax silently with a one-line log) — pick one, pin it in a test.

## Phase 3 — Per-head logit calibration (training-free)

- [ ] Calibration probe: one forward pass at load (or a cached measurement), per-head logit std estimate.
- [ ] Frozen per-head scale table: BLAKE3-checked sidecar; loud-fail on mismatch; opt-in `scale_invariant_calibrated` or an env knob — decide at implementation, document here.
- [ ] Normalize logits by the per-head scale before the schedule when calibration is armed; pass-through when not.

## Phase 4 — pp-RoPE table variant (`riir-infer/src/rope.rs`, KERNEL-ONLY)

- [ ] `RopeFreqTable::new_pp(theta, head_dim, max_train_len)` — mask frequencies whose wavelength exceeds ~max_train_len (paper: effective base 1024 vs θ=10k); zero GD; unit test on component count.
- [ ] Doc-comment states: quality claim DEFERRED to riir-train Plan 449; zero-shot pairing expected to regress (G3 row pre-records the expected FAIL).
- [ ] G5-style parity: CPU table vs GPU kernel reads (existing rope parity discipline) before any published number.

## Phase 5 — Diagnostics (`katgpt-attn/src/chiaroscuro/`)

- [ ] Per-decade mass probe: `E[Z_t^{tΔ}]` over t ∈ {10²,10³,10⁴,10⁵}, Δ ∈ {2,10}; assert Θ(1) band on Gaussian logits; report real-logit curves (Bonsai feed) as bench output.
- [ ] Entropy-in-range probe: H ~ √log t (si) vs H ~ log t (unscaled, negative control) — replay paper Fig. 2 / App. I.1 curves.
- [ ] τ sweep driver {1, 10, 100} × in-dist ppl delta (App. H protocol) as a bench target.

## Phase 6 — Lean theorems + spec-match (`.proofs/KatgptProof` + `crates/katgpt-core/tests/`)

- [ ] `scale_invariant_decade_mass_bounded` (harmonic-sum interval bounds; axiom budget unchanged).
- [ ] `sigmoid_transfer_halves_constant` (tilt identity + dominated convergence; explicit ε(t)).
- [ ] `scale_invariant_dominates_ssmax_on_decade_mass` (comparator ordering: SSMax/ALiBi fail Def 3.1).
- [ ] Paired Rust mirrors in the `ssmax_spec_match.rs` shape; `proof_gate.sh` + `EXPECTED_THEOREMS` bump in the same commit.

## Phase 7 — GOAT gates + benchmarks

- [ ] G1 correctness: LUT vs closed form ≤ 1e-6; boundary; validity assert; decade-mass flatness to t=10⁶; sigmoid transfer vs Φ(−1/a_t) within C·a_t⁻³ (C fixed once; values derived at test time); no-double-temperature; Bonsai real-logit decade diagnostics.
- [ ] G2 perf (`--release`, `bench_preflight.sh` PROVENANCE line, GPU-exclusive): LUT build <1 ms @64k; pp2048/tg128 within ±3%; Metal/CUDA parity before any published number.
- [ ] G3 SPLIT: (i) calibrated zero-shot on Bonsai-27B GGUF — 16× length ppl + retrieval probe, zero weight edits; (ii) uncalibrated zero-shot, same protocol, separate row; (iii) pp-RoPE zero-shot row — EXPECTED FAIL recorded in advance; in-dist ppl tolerance pinned per τ sweep.
- [ ] G4 alloc: counting-allocator canary on the attention path (zero allocations, LUT preallocated).
- [ ] League: record pp16384/pp65536 cells + train-length provenance column (standing law from riir-train 449).
- [ ] Promotion decision: default-on ONLY if G1–G4 all pass AND gain is modelless (it is); demote/keep ssmax per slot outcome; re-gate on any feature touch.

## Explicitly out of scope (here)

- Training/CPT to realize the gains on real checkpoints → riir-train Plan 449.
- HLA transfer → audited discard (no per-key logits in the recurrent-state form; Research 610 Path-0 row 7).
- Healer surfaces → none (attention is not a healer surface; no riir-refine filing).
