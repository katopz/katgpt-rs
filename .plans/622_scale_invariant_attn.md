# Plan 622: Scale-Invariant Attention — `scale_invariant_attn`

**Status:** Active — Phase 1 + Phase 2 LANDED 2026-10-10 (opt-in `scale_invariant_attn`; LUT primitive + SDPA hook + parallax both-arm wiring + the tilt/Lemma G1 instruments, 404 tests green at the wired posture / 2151 default-posture no-regression / clippy 0 both postures). **Phase 3 LANDED 2026-10-11** (kernel-side: `SpreadAccumulator` + `HeadCalibration` sidecar + `SiArm` + the fused calibrated apply + the calibrated SDPA wrapper; 21 si-module tests + 5 SDPA si tests green, 2182 wired-posture / 2151 default-posture G3 unchanged, clippy 0 ×3 postures; feature decision RECORDED — no second flag, calibration is runtime data under `scale_invariant_attn`; parallax-arm calibrated consumption DEFERRED — its si apply stays at 1.0 until a consumer needs it; the real-checkpoint probe run + the G3 calibrated rows ride Phase 7's box/GPU gates via the Phase-4 riir-infer loader). Phase 4 targets riir-infer (contended this session); Phases 5–7 pending (5 diagnostics, 6 Lean, 7 GOAT — G2/G3 box/GPU-gated). Track (a) modelless, PRIMARY. Training track: riir-train Plan 449 (SECONDARY, owns the pp-RoPE quality claim).

Source: arXiv:2505.17083 v2 (Anson/Wang/Aitchison, NeurIPS 2025). Distillation: `.research/610_Scale_Invariant_Attention.md`. Signal-diff vs shipped `ssmax.rs` (Plan 411): position-DEPENDENT per-distance (a_t, m_t) vs position-INDEPENDENT scalar s_L·log(N) — real delta, not covered.

## Non-negotiable laws (from the verdict review)

- **Ordering law:** when `scale_invariant_attn` is active, the SSMax length multiplier is BYPASSED. Both are length temperatures; stacking double-sharpens. A G1 test pins the mutual exclusion.
- **Tilt-transfer G1 law:** the sigmoid arm's expected mass constant is `E[σ(−L′_t)]`, `L′ ~ N(1, a_t²)` = `Φ(−1/a_t)` + O(a_t⁻³). G1 asserts `|E_GH − Φ(−1/a_t)| ≤ C·a_t⁻³` with C fixed ONCE in the test. NEVER a one-sided `1/(a√2π)` bound (loose at small t); NEVER pin the illustrative ≈0.29 / ≈0.41 values — derive from the formula at test time. The drift toward α/2 is slow (O(1/√log t)) and EXPECTED.
- **Calibration precondition:** the schedule assumes ~unit-variance base logits (paper's models carry QK-norm). Raw-logit checkpoints get a training-free per-head calibration (measure logit spread once at load → frozen BLAKE3-checked per-head scale table → normalize before the schedule). G3 is SPLIT: "calibrated zero-shot" and "uncalibrated zero-shot" are separate rows, never pooled.
- **pp-RoPE is KERNEL-ONLY here.** Dropping low-freq RoPE components on a full-RoPE checkpoint changes the trained position representation — zero-shot quality with pp-RoPE is EXPECTED TO FAIL and the G3 row says so in advance (recorded expectation, not a surprise to be explained later). The pp-RoPE quality claim lives in riir-train Plan 449.
- **Sink carve-out:** the BOS/sink token is EXCLUDED from the transform (the paper's Gaussianity premise excludes it; App. J).
- **Zero-alloc:** preallocated interleaved (a_t, m_t) LUT at session start; read-only hot path; unit-stride descending reads (contiguous distance slice per query); counting-allocator canary.

## Phase 1 — LUT + primitive (`katgpt-core/src/scale_invariant.rs`)

- [x] `ScaleInvariantLut { am: Vec<f32>, tau: f32 }` interleaved a_t|m_t, `build(tau, max_ctx)` with the β ≥ α·log α validity assert; constants α=β=e^0.5 derived from the (a₀², m₀)=(1,0) boundary pin, not hard-coded.
- [x] `apply_inplace(&self, scores: &mut [f32], query_i, key_start)` — distance-indexed affine `s → a_t·s + m_t` over the contiguous reversed slice; chunked 8-wide for LLVM auto-vectorization (ssmax.rs house pattern). **Landed as `apply_inplace(scores, query_i, key_start, logit_scale)`** — the SDPA kernels fold `scale` at their exp step, so the affine must enter in pre-scale space (`m_t/scale`; a_t commutes) — folding scale AFTER the affine would wrongly rescale m_t (the same unscaled-lane law as the prior lane). Callers already in logit space (parallax) pass 1.0 — `m·1.0 == m` IEEE-exact, bit-identical. The plan's 8-wide chunk form is ALSO a recorded deviation: the transform is a gather-indexed affine (per-element distance), not a uniform op — LLVM cannot vectorize it either way; the shipped loop is the branch-free causal-segment enumerate form (segment split = future-key bit-exact skip + sink carve-out), and the plan language moves to G2 where the real vectorization question lives.
- [x] Feature `scale_invariant_attn` (opt-in); lib.rs module doc states the ordering law + sink carve-out.
- [x] Boundary/safety tests: a₀²=1, m₀=0 exact; near-identity ramp for t ≤ τ; no NaN for t up to 10⁶.

## Phase 2 — Attention wiring (both arms)

- [x] `attention.rs` SDPA: per-key affine hook in the pre-softmax score path, before online-max tracking (FlexAttention score_mod shape); flag-off = bit-identical. (`tiled_attention_forward_si`, cfg `all(tiled_attention, scale_invariant_attn)`; hook rides both the tiled kernel and the materialized fallback.)
- [x] `parallax_attn`: a_t rides a new per-key MULTIPLICATIVE lane (sibling of the existing gated `prior_logit_lane` additive lane); m_t reuses the additive lane shape; both arms (softmax + normalized sigmoid) consume the same LUT. (`ParallaxConfig::scale_invariant: Option<Arc<ScaleInvariantLut>>`; applied FIRST among the lanes — si REPLACES ssmax, prior lane stays after.)
- [x] Sigmoid-arm mass assertion (unit test): measured E[σ(L_t)]·(t/τ+1)/α equals the Gauss–Hermite E[σ(−L′_t)] within 1e-6 (algebra identity). **Instrument note:** the σ integrand's transition (width ~1/a in the standardized variable) is MISSED by Gauss–Hermite nodes once a ≳ 5 — 40-node GH read 15× noise at a=10; the shipped instrument is adaptive Simpson (±40σ, tol 1e-12) for every σ-integrand test, GH retained only where it is the right tool (the e^x Lemma — entire integrand, spectrally exact; + the polynomial-exactness moment guard). Both sides of the identity now carry independent accurate quadrature; the 1e-6 bar holds on all six t-decades to 10⁴.
- [x] SSMax mutual-exclusion: enabling both together is a loud config error (or the schedule bypasses SSMax silently with a one-line log) — pick one, pin it in a test. **Picked the loud config error** (assert in `tiled_attention_parallax_forward_retaining`, fires in both profiles, message names the ordering law; `#[should_panic]` test pinned). At the SDPA API level the two wrappers are separate calls — no composition exists to refuse.

## Phase 3 — Per-head logit calibration (training-free)

- [x] Calibration probe: one forward pass at load (or a cached measurement), per-head logit std estimate.
  → **LANDED 2026-10-11** — `SpreadAccumulator` (`scale_invariant.rs`): per-head Welford (count/mean/M2), order-exact under chunked feeds, `std_estimate` None below 2 observations (never NaN); the SINK-DROP contract documented (the schedule never transforms key 0, so the sink's logit must not enter σ_h — the pipeline test feeds it dropped). The REAL-checkpoint probe run (one Bonsai forward at load) rides the Phase-4 riir-infer loader — the instrument is what this repo owes.
- [x] Frozen per-head scale table: BLAKE3-checked sidecar; loud-fail on mismatch; opt-in `scale_invariant_calibrated` or an env knob — decide at implementation, document here.
  → **LANDED 2026-10-11** — `HeadCalibration`: build-side + decode-side validity (finite positive, every head), BLAKE3 digest over magic+version+count+LE scales, `to_sidecar`/`from_sidecar` loud-fail naming the cause (digest / magic / version / length / scale), `validate_heads` (the checkpoint-binding check). **DECISION: no second feature flag** — the calibration is RUNTIME DATA under `scale_invariant_attn` (the absent-table path is the pinned byte-identical posture; a second flag would triple the posture matrix for zero compile savings; the sidecar's digest is the safety, an env knob belongs to the model-loader consumer). Pinned by the sidecar round-trip + corruption-matrix test.
- [x] Normalize logits by the per-head scale before the schedule when calibration is armed; pass-through when not.
  → **LANDED 2026-10-11** — `SiArm { lut, head_scale }` threaded through the SDPA impl chain (one param instead of two — the divide is meaningless without the schedule); `apply_inplace_calibrated` = ONE fused loop (`s → a_t·(s/σ_h) + m_t/logit_scale`, no extra pass, no alloc, same sink/future carve-outs); `head_scale = 1.0` BYTE-IDENTICAL both at the apply level and end-to-end through the wrapper (G3 pins); `tiled_attention_forward_si_calibrated` matches the divide-then-affine reference on BOTH internal paths (fallback 16 / tiled 130). Parallax-arm calibrated consumption DEFERRED (`- [-]` below).
- [-] Parallax-arm calibrated consumption (`apply_scale_invariant_to_row` gaining the head divide).
  → DEFERRED — the parallax arm's si apply stays at 1.0 (exactly today's posture; nothing changes for existing consumers). The G3 calibrated rows target the Bonsai GGUF = the SDPA arm. One mechanical pass when a parallax consumer needs the calibrated posture.

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
