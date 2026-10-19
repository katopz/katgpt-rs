# Bench 947 — Issue 949 `path_kernel_sim` POC GOAT: T6 latency + the T9 keep-off verdict

**Status:** COMPLETE 2026-10-20 — G2 PASS, T9 verdict **KEEP-OFF** (the feature ships opt-in, promoted nowhere). Commit: this landing (`feat: path_kernel_sim...`). Issue: katgpt-rs `.issues/949_response_space_similarity_kernel_poc.md` (T1–T9 ticked; T10 stays deferred). Parent research: `.research/625_Path_Kernel_Exemplar_Superposition_Lens.md` (verdict addendum appended).

## What ran

`cargo bench -p katgpt-core --features path_kernel_sim --bench path_kernel_sim_goat -- --nocapture`
(`--release` by construction; M3 Max, dev build box state — latency numbers are
HEADROOM-class claims vs µs bars, not league measurements; two runs quoted).

## G2 latency (the T6 bars)

| surface | bar | run 1 | run 2 |
|---|---|---|---|
| `fingerprint_into` p=64 | ≤ 5 µs/call | **79 ns** | **132 ns** |
| `similarity` p=64 | ≤ 1 µs/pair | **49 ns** | **52 ns** |

`black_box` on args AND receiver (the timed_region law — the measured work
cannot be deleted). 63×/20× (worst-run) headroom over the issue's µs-class
bars. PASS.

## T9 — the planted-class battery (release posture)

Fixture (T5c): exponent PURE cross-term `f = σ(w1·w2·u·v + w3·w4·s·t)`; labels
`z ∈ {0,1}` from a seeded LCG, drawn BEFORE the class-conditional `x` sampling
(independence by construction — the scorer's response never feeds back);
signal = the magnitude ratio between the two cross-terms (`c = 1` vs `1.8`);
n = 80 anchors, jitter keeps within-class mass.

| space | LOOCV accuracy |
|---|---|
| fingerprint kernel | **1.000** |
| input cosine (baseline i) | **1.000** |
| output-vector baseline, m=4 probes (baseline ii) | **1.000** |

**THREE-WAY TIE → T5d's own law applies: "if the output-vector baseline ties
or beats the Jacobian fingerprint, the tangent kernel adds nothing → do not
promote." VERDICT: KEEP-OFF.**

## The structural finding (why the tie is expected, not a fixture failure)

For ANY scorer, `Ŝ(x)` is a deterministic transform of `x`. A probe battery
with m ≥ p informative (unsaturated, linearly independent) settings observes
the same class statistic the gradient direction encodes — gradient information
⊆ local-response information. The Jacobian kernel can only add information in
regimes the POC deliberately prices out:

1. **probe coverage** — a consumer whose probe budget m is genuinely ≪ p
   (the fingerprint reads all p dims for the cost of p+1 evals; the baseline
   needs m ≥ p evals to match),
2. **probe degradation** — saturated/clipped probes lose the local signal
   the normalized gradient keeps (the σ' scalar cancels on normalization).

Both are consumption-side preconditions, not properties of the kernel. The
re-open bar: a consumer whose probe budget is genuinely ≪ p.

## G1/G3/G4 (the supporting gates)

- **G1** (lib tests, 7/7 at the feature posture): T5a the linear-in-knobs
  oracle — normalized fingerprint cosine ≡ input-feature cosine to 1e-5
  (the collapse caveat PROVEN, not assumed); T5b amplitude-sine (maxima a
  period apart are kernel-identical); T1 zero-norm abstain (stationary
  point → sentinel, never NaN; NaN-input defense); T2 arms-agree;
  T3 Spearman monotone/degenerate; T4 known-spectra PR/effective-rank +
  the collinear-Gram end-to-end (rank-1 → PR 1.000).
- **G3**: purely additive feature — the OFF posture compiles the module
  away; the full default lib suite (2157 passed / 8 ignored) is untouched.
- **G4**: `tests/path_kernel_sim_g4_alloc.rs` — counting allocator, warm
  scratch, 100 steady-state fingerprint+similarity pairs at p=64: **0
  allocations**. The set-level `gram_spectrum_into` diagnostic allocates a
  small descending-order Vec by design (documented, out of the hot path).

## Verdict

**KEEP-OFF.** `path_kernel_sim` ships opt-in (default-off, no root forward,
no consumer) as a *diagnostic instrument* — the correctness gates make it a
sound one. Promotion stays closed until a consumer with a genuinely ≪ p
probe budget exists. Determinism note (Issue 940): all outputs are LOCAL to
the computing process in this scope; if a consumer ever routes them into
synced belief attribution, the sequential-fold + first-index-tie +
`freeze_version_hash` coverage binds before any wire crossing.
