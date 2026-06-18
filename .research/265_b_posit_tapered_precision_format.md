# Research 265: B-Posit — Tapered Precision Format with Bounded Regime

> **Source:** Closing the Gap Between Float and Posit Hardware Efficiency — A. Jonnalagadda, R. Thotli, J. L. Gustafson (BITS Pilani + ASU). arXiv:2603.01615v1 [cs.AR], 2 Mar 2026.
> **Date:** 2026-06-18
> **Status:** Active — Gain (deferred, awaiting hardware target)
> **Related Research:** 039 (SpectralQuant eigenbasis KV compression), 065 (RotorQuant block-diagonal rotation), 159 (KVarN variance-normalized KV quant), 200 (Quantization outlier collapse security), 223/224 (ANE distillation verdicts)
> **Related Plans:** 179 (KVarN), 271 (Attention Matching compaction)
> **Cross-ref (riir-ai):** none yet — see Issue 031 follow-up
> **Classification:** Public

---

## TL;DR

The **b-posit** ("bounded posit") format caps the posit regime field at `rS = 6` bits, collapsing the variable-length regime/exponent/fraction partition into a **fixed menu of 5 layouts** that decode via a 5-input multiplexer (no leading-bit-counter, no barrel shifter, no subnormal-handling hardware). The paper's headline: at 32-bit, b-posit decode is **39% faster than IEEE float decode** while preserving posit's tapered precision; at 64-bit it beats IEEE float on power, area, **and** delay simultaneously. B-posit also gives a wide **golden zone** (`2^-64 … 2^64` for `⟨32,6,5⟩`) where 75% of bit patterns deliver ≥ float32 accuracy, with **no subnormals and no FTZ/DAZ ambiguity** — exactly the determinism property our raw-domain sync layer needs.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is *not* the hardware circuit (we don't fab chips). It is the **bounded-regime numerical encoding** as a storage / commitment format with three software-visible properties we currently lack: (1) bit-identical cross-platform decode (no FTZ/DAZ drift), (2) tapered precision that allocates extra significand bits in the `±[1, 16)` "fovea" where most sigmoid-projected latent scalars live, (3) a **fixed-layout decode tree** (5-case multiplexer) that admits a branch-free, SIMD-friendly software implementation competitive with hand-rolled quantization schemes.

---

## 1. Paper Core Findings

### 1.1 The format

A b-posit is parameterized as `⟨N, rS, eS⟩` (precision N, max regime size rS, exponent size eS). With `rS = 6`, the regime field can be only 2, 3, 4, 5, or 6 bits wide — five possible layouts total. Recommended configurations:

| Precision | Format | Dynamic range | Fovea | Min sig digits |
|---|---|---|---|---|
| 16 | `⟨16, 6, 3⟩` | `2^-64 … 2^64` | `2^-8 … 2^8` | 2 |
| 32 | `⟨32, 6, 5⟩` | `2^-192 … 2^192` (~`10^-58 … 10^58`) | `2^-32 … 2^32` | 6 |
| 64 | `⟨64, 6, 5⟩` | same as 32-bit (range is fixed by `rS,eS`, not N) | `2^-32 … 2^32` | 14 |

Key contrast with standard posits: the dynamic range is **bounded independent of N**, so adding bits past N=12 always goes to the significand. This is the opposite of standard `⟨N, eS⟩` posits, whose dynamic range grows with N (and whose regime decode logic grows with N).

### 1.2 Why the hardware wins

Standard posit decode has a serial critical path: `leading-bit-counter → barrel shifter → exponent/fraction extract → priority encode`. Both the LBC and the shifter scale linearly with N. IEEE float decode avoids this for normals but pays the same cost for subnormals (and most GPUs ship non-compliant FTZ hardware that silently flushes subnormals to zero, breaking cross-vendor bit-identity).

B-posit decode replaces this with: `5-bit XOR → one-hot decode → 5-input mux`. The mux width grows with N but its **fan-in stays at 5**, so logic depth is constant across N. Reported post-layout numbers at 45 nm:

| Config | Decode power | Decode area | Decode delay | vs posit | vs IEEE float |
|---|---|---|---|---|---|
| `⟨16,6,5⟩` | 0.11 mW | 335 µm² | 0.39 ns | –66% / –52% / –45% | slightly worse |
| `⟨32,6,5⟩` | 0.20 mW | 553 µm² | 0.52 ns | –79% / –71% / –60% | **–39% delay** (b-posit faster) |
| `⟨64,6,5⟩` | 0.37 mW | 994 µm² | 0.65 ns | –83% / –75% / –57% | beats float on all three |

Worst-case energy per op (decode×2 + encode, pJ): b-posit **ties IEEE float at 32-bit, uses 40% less energy than IEEE float at 64-bit**.

### 1.3 Mathematical properties preserved

- **No subnormals, no NaR-as-exception ambiguity** — posit's "NaR" sentinel handles all exceptions; comparison hardware = signed-int compare.
- **Minimum-significant-bits guarantee** (b-posit never drops below `N − rS − eS − 1` sig bits even at extreme magnitudes) — required to prove numerical-analysis error bounds, lost by IEEE subnormals.
- **Tapered accuracy** — the fovea (`2^-32 … 2^32` for `⟨32,6,5⟩`) covers 75% of bit patterns and delivers float32-or-better accuracy across the entire region; standard posit32's golden zone is only `2^-20 … 2^20`.
- **Round-to-nearest ties-to-even**, single rounding mode — no rounding-mode flags to drift across implementations.

---

## 2. Distillation

### 2.1 What does NOT transfer

- **Hardware decode/encode speedups** — we are a software Rust codebase. On commodity CPUs without posit SIMD instructions, software-emulated b-posit decode is ~50–200 ns/op (XOR + 5-case LUT + shift), competitive with our existing TurboQuant store path but slower than native f32 arithmetic. The paper's power/area/delay wins only materialize in silicon we don't have.
- **Quire / exact-accumulate** — out of scope; we don't have an 800-bit accumulator primitive and our hot path is dot products, not exact dot products.

### 2.2 What DOES transfer

Three properties, in order of near-term value:

1. **Subnormal-immune cross-platform encoding.** Our raw-domain sync layer (`SyncBlock → ChainConsensus → Cold tier`) currently uses raw f32/f64 and relies on platform FTZ/DAZ behavior being consistent. `riir-ai/crates/riir-gpu/src/gemv_q4k_cubecl.rs` and `attention_q8kv_cubecl.rs` already have explicit "subnormals treated as zero" comments — we are already papering over the exact ambiguity b-posit eliminates. A b-posit storage layer at the cold tier would make the commitment canonical.
2. **Tapered precision for sigmoid-projected scalars.** The 5 scalars we sync across the raw↔latent bridge (`valence, arousal, desperation, calm, fear`) are all sigmoid outputs in `[0,1]` or `[-1,1]` — squarely inside b-posit's fovea. They get **2× the accuracy of IEEE float** at the same bit width in that region.
3. **Fixed-layout decode tree** admits a SIMD-friendly branch-free Rust implementation (5-element lookup table indexed by one-hot regime size) — no leading-zero count, no variable shift.

### 2.3 Where it does NOT beat our existing stack

- **KV-cache quantization.** SpectralQuant (research 039) already uses an eigenbasis rotation that concentrates variance into a few directions and applies per-direction bit budgets. B-posit's *global* tapered precision is strictly weaker than SpectralQuant's *data-adaptive* allocation. B-posit × SpectralQuant fusion would only help if the post-rotation residuals still cluster around the fovea (TBD — speculative).
- **Hot-path inference math.** Native f32 SIMD on Apple Silicon / x86 AVX2 will beat any software posit emulation. B-posit belongs at the **storage / sync / commitment boundary**, not in `simd_dot_f32`.
- **Weight storage.** GGUF / safetensors are f16/bf16. Adding b-posit weight loading means every model loader needs rewriting, and we lose BF16 hardware paths on ANE. Not worth it.

---

## 3. Fusion

### Closest existing notes / plans / code across both repos

| Cousin | Repo / location | Relation |
|---|---|---|
| **Research 039 — SpectralQuant** | `katgpt-rs/.research/039_*` | Eigenbasis KV compression — strongest fusion candidate (data-adaptive tapered precision) |
| **Plan 179 — KVarN** | `katgpt-rs/.plans/179_*` | Variance-normalized KV quant — could borrow b-posit's fovea concept for variance-clustered bins |
| **Research 065 — RotorQuant** | `katgpt-rs/.research/065_*` | Block-diagonal rotation quantization — orthogonal axis, could compose |
| **Research 200 — Quantization Outlier Collapse Security** | `katgpt-rs/.research/200_*` | Security angle — b-posit's bounded range closes some outlier-collapse attack surface |
| **gemv_q4k_cubecl.rs / attention_q8kv_cubecl.rs** | `riir-ai/crates/riir-gpu/src/` | Concrete FTZ/DAZ pain point b-posit eliminates |
| **WeightDtype { F32, F16, BF16 }** | `katgpt-rs/crates/katgpt-core/src/types.rs` | The enum we would extend if we ever adopted b-posit as a storage dtype |

### Fusion idea — novelty TBD, needs Q1–Q4 check before verdict

**B-posit × SpectralQuant "fovea-allocation" fusion.** SpectralQuant currently allocates bits per eigen-direction based on variance. If post-rotation residuals in low-variance directions still cluster around the fovea (which they should, by construction — low-variance = concentrated near mean = inside `[1, 16)`), then storing those directions as b-posit instead of uniform-precision int8 buys extra significand bits for free. The high-variance "outlier" directions stay on the existing TurboQuant path. **Hypothesis:** at matched bit budget, this beats pure SpectralQuant on residual cosine for distributions where the post-rotation spectrum is steep (long tail of small eigenvalues). Untestable today without a benchmark; tracked in Issue 031.

This is a **fusion idea, not a Super-GOAT claim** — it needs Q1–Q4 novelty-gate work (does a "tapered-precision-per-eigenvalue" scheme already exist in the literature? does our codebase already do something equivalent under different vocabulary?) before any verdict promotion.

---

## 4. Verdict

**Gain (deferred).**

**Reasoning:**

- **Novelty (Q1): PASS.** Zero prior art for posit / b-posit / tapered regime / Gustafson / takum / fovea / golden zone across both `.research/` and `.plans/` and across `katgpt-rs/src/`, `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-armageddon/crates/` (vocabulary-translated grep on both paper and codebase terms). The format is genuinely new to the codebase.
- **New class of behavior (Q2): FAIL.** A new number format is not a new *capability* for our system. We can already achieve bit-identical cross-platform sync via fixed-point or raw-byte commitment. B-posit would be a *better encoding* for that capability, not a new one.
- **Product selling point (Q3): PARTIAL.** "Our synced state is subnormal-immune across CPU/GPU/ANE" is real, but not headline-worthy — customers don't buy inference engines on numerical encoding choices. The selling point only materializes if/when an ANE or GPU vendor ships posit hardware (none has, as of 2026-06).
- **Force multiplier (Q4): PARTIAL.** Touches quantization (039/065/179), sync boundary (riir-armageddon raw domain), ANE backend (155/223/224), and chain commitment (BLAKE3). But the connection to each is *storage-format-level*, not *algorithm-level* — b-posit doesn't change what any of those systems *do*, only how they *encode* their I/O.

**Q2 + Q3 fail → not Super-GOAT, not GOAT.** No new plan: there is no software-level speedup to gate behind a feature flag today. The fusion idea in §3 is speculative and untestable without benchmark infrastructure we don't have for tapered-precision quantization.

**Action:** Track in `katgpt-rs/.issues/031_b_posit_cross_platform_deterministic_encoding.md`. The trigger for re-evaluation is concrete: (a) a hardware target we ship to adds posit/b-posit support, OR (b) SpectralQuant fovea-allocation fusion produces a measurable cosine gain in `tests/bench_spectralquant.rs`. Until then, this note is reference material.

**What we do NOT do:**
- ❌ Implement software b-posit decode/encode now — no consumer.
- ❌ Extend `WeightDtype` with a `BPosit32` variant — no model loader, no hardware path.
- ❌ Plan the SpectralQuant × b-posit fusion — needs a literature novelty sweep first (is tapered-precision-per-eigenvalue already published?).

---

## 5. Limitations & honest caveats

1. **The hardware wins are not ours to claim.** Every power/area/delay number in §1.2 is post-layout silicon at 45 nm — irrelevant to a Rust inference engine until a vendor ships the format.
2. **Software emulation cost is real.** A branch-free 5-case LUT decode in Rust is ~50–200 ns/op depending on width; native f32 ops are sub-nanosecond. B-posit is *storage*, not *arithmetic*, for us.
3. **The fovea-allocation fusion hypothesis is unproven.** "Post-SpectralQuant residuals cluster in the b-posit fovea" sounds right but is empirical — could be dominated by the same outlier-collapse dynamics research 200 already characterizes. Don't claim a Super-GOAT without benchmarking.
4. **Posit Standard (2022) momentum.** The b-posit authors position their work as a *proposal for the next standard revision*. If the Posit Working Group ratifies b-posit, hardware adoption accelerates and this note's verdict should be re-evaluated upward. If they ratify takum instead (Hunhold's competing bounded-range format), b-posit's window closes.

---

## TL;DR

B-posit caps the posit regime at 6 bits → 5 fixed layouts → multiplexer-only decode, beats IEEE float on power+area+delay at 32- and 64-bit in silicon, eliminates subnormals entirely. **Verdict: Gain (deferred)** — novel to our codebase (zero posit/tapered-regime prior art in notes or code), but the paper's wins are hardware-side and we don't fab chips; software value is bounded to (1) subnormal-immune cold-tier storage, (2) tapered precision for sigmoid scalars at the raw↔latent bridge, (3) a speculative SpectralQuant fovea-allocation fusion (Issue 031). No plan created — no software-level GOAT gate to run today. Re-evaluate when (a) a target hardware platform adds posit support, or (b) the SpectralQuant × b-posit fusion shows measurable cosine gain in `bench_spectralquant.rs`.
