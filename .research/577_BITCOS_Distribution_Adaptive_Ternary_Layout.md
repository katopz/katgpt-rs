# Research 577: BITCOS — Distribution-Adaptive Ternary Layout (Presence Bitmap + Compacted Signs)

> **Source:** "Breaking the 1.58-bit Barrier for Ternary LLMs" — [arXiv:2609.16338](https://arxiv.org/abs/2609.16338), Evangelos Georganas, Alexander Heinecke, Pradeep Dubey (Intel), 2026-09-14
> **Date:** 2026-09-21
> **Status:** DISTILLED — Gain verdict; Issue 864 filed (codec + z-meter + GOAT gate)
> **Related Research:** 110 (Ciot ternary CPU distillation — the shipped 2-bit-plane substrate), 418 (StreamDQ SIMD LUT dequant), 508 (Pipeline-Native CPU decode), 538 (GDN W4A4)
> **Related Plans:** none yet (Issue 864 is the entry point)
> **Cross-ref (riir-ai):** `TernaryGroupWeights`/`TernaryHandle` (riir-gpu CUDA/CubeCL kernels) — the direct consumer surface; riir-clippy kernel_opt corpus rules `base-3-trit-packing-5-per-byte`, `telescoped-digit-coefficients-byte-fma-dot`, `swar-lane-widened-base3-dp4a-dot`
> **Classification:** Public

---

## TL;DR

Real ternary LLM checkpoints are NOT equiprobable over {−1, 0, +1}: measured zero density z spans **0.297–0.515 across 29 SOTA checkpoints** (Bonsai 1.7B–27B: 0.297–0.399; CAT-Q Qwen3: up to 0.515). BITCOS stores a ternary tensor as a **dense presence bitmap (1 bit/weight) + compacted sign vector (1 bit/nonzero)** = `2−z` bits/weight — beating the deployed five-trit packing (1.625 bits) for z > 0.375 (26/29 models) and beating **2-bit packing at every z** (1.16–1.32× weight-traffic reduction incl. scales). The conversion is **bit-exact** — layout only, no retraining, applies to existing checkpoints.

**Why it matters here:** our ternary substrate lives in **katgpt-types** as three containers — `TernaryWeights` (plasma_path, bit-planes, 2.0 bits/w), `TernaryGroupWeights` (Issue 578, `ternary_group_scale`, bit-planes + f16 group scales = **2.125 bits/w** — the Bonsai `Q2_0_g128` serving container), and `TernaryTritWeights` (Issue 582, `ternary_trit_pack`, 5 trits/byte + scale = **1.75 bits/w**, 5.90 GB for Bonsai-27B, GOAT-benched) — with riir-ai/riir-gpu as consumers (`TernaryHandle` bit-plane + `TernaryTritHandle` trit GPU handles). Against the **bit-plane tier** BITCOS wins at every z (1.16–1.32× incl. scales); against the **trit tier** (1.75) it wins only z > 0.375 — thin for the Bonsai family (1.7B: 1.01×, 4B/8B: 1.00×, **27B: 0.96× — a regression**) and meaningful only for high-z checkpoints (BitNet z=0.42: 1.03×; CAT-Q-class z=0.515: 1.09×). The paper also ships the two-term roofline (`ebw = min(β, B(z)/γ)`) that predicts **when the layout LOSES** (bandwidth-rich/few-core → instruction-bound; measured 0.47–0.70× on Lunar Lake — the in-repo twin is Issue 582's G2b note: cache-resident ⇒ storage PASS + latency FAIL, the `binary_plasma` precedent) — mandatory reading before wiring it anywhere.

**Distilled for katgpt-rs (modelless, inference-time):** a distribution-adaptive ternary codec in **katgpt-types** (beside the three existing containers, following Issue 582's `from_group`/`to_group` repack precedent) — pack (from `TernaryGroupWeights` or f32; the conversion from planes is nearly free: bitmap = `pos|neg`, signs = compacted `pos` bits) + unpack kernels (portable, AVX2-class byte-materialization, LUT-based pdep-replacement for GPU) + a zero-density meter + a roofline-gated format dispatch across the now-FOUR-tier ladder (bit-plane / trit / bitcos). No training content anywhere in the paper (§3.5 panel legitimately skipped — clearly inference-side).

---

## 1. Paper Core Findings

### 1.1 The zero-density measurement (the founding observation)

29 ternary checkpoints across 7 families (BitNet b1.58 2B4T, Bonsai 1.7–27B, CAT-Q Qwen3 1.7B–235B, ParetoQ 125M–1.5B, TriLM/Spectra 99M–3.9B, Maple 20B-A1B, BitCPM-CANN 0.5–8B). Zeros account for **29.7%–51.5%** of weights. The conventional rate accounting (log₂3 ≈ 1.585; 5-trits/byte = 1.6 bits, rounding to 1.625 at power-of-two block 128) assumes equiprobable symbols — false for every measured model.

Rows most relevant to us (symbols-only rate; "+scale" adds measured 16-bit group scales, group 128):

| Model | z | BITCOS bits/w | 2-bit +scale (2.125) red. | 5-trit +scale (1.750) red. |
|---|---|---|---|---|
| Bonsai 1.7B | 0.3989 | 1.601 | **1.23×** | 1.01× |
| Bonsai 4B | 0.3771 | 1.623 | **1.22×** | 1.00× |
| Bonsai 8B | 0.3825 | 1.618 | **1.22×** | 1.00× |
| Bonsai 27B | 0.2966 | 1.703 | **1.16×** | 0.96× (worse) |
| BitNet b1.58 2B4T | 0.4219 | 1.578 | **1.27×** | 1.03× |
| CAT-Q Qwen3-1.7B | 0.5148 | 1.485 | **1.32×** | 1.09× |

Note the honest split: for Bonsai-27B BITCOS is *worse than 5-trit* (z below the 0.375 crossover) but still *better than 2-bit* — and our shipped layout is the 2-bit class.

### 1.2 The layout

- **Presence bitmap**: dense, positional, 1 bit per weight (bit set = nonzero). ⌈K/32⌉ × N u32 words — adjacent columns adjacent in memory (block-loadable).
- **Compacted sign vector**: 1 bit per nonzero, in tensor order. The j-th sign bit belongs to the j-th set bit of the bitmap. Per-column read position ("cursor") = popcount of preceding presence bits — the only per-column state the bitmap alone doesn't give away.
- Cost: `B(z) = 1 + (1−z) = 2−z` bits/weight. **Bit-exact decode** — same ternary values as the source model.

### 1.3 The unpack sequences (the engineering core)

- **AVX-512** (17 instrs/32 weights; unpack = 3 of them): `pdep` scatters the compact signs back to the positions the presence mask marks; two masked moves (`vmovdqu16` zero-masked +s, merge-masked −s) select; one `vfmadd231ph`. Scales held as ±s pairs via one `vporq 0x8000`. Sign-stream addressing = `shrx` unaligned window + `popcnt` advance.
- **AVX2 + AVX-VNNI-INT8** (no mask registers, no FP16): presence/sign materialized as byte masks (`Pi=−pi`, `Ni=−ni` via broadcast+shuffle+`vpcmpeqb`), ternary byte = `wi = pi − 2ni` (mask 0xFE + `vpsubb`), then `vpdpbssd`. pdep still available on x86.
- **Xe2 GPU** (no pdep): a **256-entry SLM LUT** indexed by (4-bit presence nibble m, 4-bit sign window s) returning 4 fp16 ternary codes — "a precomputed, four-bit-wide pdep composed with the sign→ternary map". Two popcounts with distinct roles (per-group drives the window shift; per-block drives the gather address) keep the loop-carried chain short. d32x3 gather covers two 32-row blocks (span ≤ 94 bits < 96).

### 1.4 The roofline (the regime boundary — the paper's most reusable tool)

Per 32 weights: `B(z) = 8.5 − 4z` bytes read. `ebw = min(β, B(z)/γ)` where γ = L1-resident microkernel cost (cycles), β = per-core bandwidth share.

- **Memory-bound** (B(z)/γ > β): smaller payload converts to time savings. EMR 64c (245 GB/s): BITCOS wins at every z, 1.14–1.28× GEMV over SOTA 2-bit in the deployed band; e2e decode 1.10–1.18×.
- **Instruction-bound** (B(z)/γ < β): decode cost dominates and BITCOS **LOSES**. Lunar Lake 8c (108 GB/s, 3.2–3.7 B/cyc available vs 0.87–1.38 consumable): BITCOS GEMV sustains ≤33 GB/s vs 2-bit's 74.7; e2e **0.47–0.70×**. "A bandwidth-rich client platform with a limited number of cores is exactly the case where a cheaper decode beats kernels with smaller payload and more expensive decode."
- ARL 24c sits between (P-cores memory-bound 1.02–1.15× e2e; E-cores at the knee).
- GPUs: Arc 140V 1.04–1.14× GEMV / 1.09–1.22× e2e; Arc Pro B70 1.01–1.12× / 1.02–1.27×. Realized gain < byte ratio on GPU (decode work per weight doesn't shrink with z).

### 1.5 Related-work deltas the paper draws (and we should inherit)

- vs **Ternary CSC** (index-based sparse, their ref [32]): helps compute-bound prefill only; decode GEMV *worse* than dense; needs 75–88% z. BITCOS is dense-bitmap (sequential access, vectorizable) and wins in the *bandwidth-bound* regime at real z.
- vs **Sparse-BitNet / Sherry** (semi-structured N:M / 3:4): both **require training** and cost accuracy (Sparse-BitNet: +0.17–0.32 ppl). BITCOS uses zeros existing checkpoints already contain; layout-only, bit-exact.

---

## 2. Distillation

### 2.1 What ships here today (vocabulary-translated inventory)

**Ownership corrected per the verdict review (round 3): the ternary container tier is OWNED by katgpt-rs/katgpt-types; riir-ai/riir-clippy are consumers.**

| Surface | Location | Format | Nominal rate |
|---|---|---|---|
| Plasma bit-plane container | `katgpt-types/src/ternary.rs` (`plasma_path`; Issue 145 reclassified binary as Plasma, this as Hot) | pos/neg u64 planes + row f32 scale | 2.0 + row scale |
| Group-scale serving container | `katgpt-types/src/ternary_group.rs` (`ternary_group_scale`, Issue 578) — **the Bonsai `Q2_0_g128` container**, 7.17 GB @27B | pos/neg u64 planes + f16 group-128 scales | **2.125** |
| Trit container | `katgpt-types/src/ternary_trit.rs` (`ternary_trit_pack`, Issue 582) — 5.90 GB @27B; GOAT gate `tests/bench_582_trit_pack_goat.rs` (G2 footprint ≤0.83× arithmetic, G4 alloc-free; G2b: cache-resident latency loss EXPECTED, `binary_plasma` precedent — the in-repo twin of the paper's LNL row) | group-aligned base-3 trits (28 B/128 w) + f16 scale | **1.75** |
| GPU serving handles | riir-ai `riir-gpu`: `TernaryHandle::from_weights` (bit-plane) + `TernaryTritHandle::from_weights` via `repack_to_group_aligned_trits` (trit) — BOTH tiers live on CubeCL; `gemv_ternary_cuda_raw.rs` bit-plane on CUDA; embedding dequant; `ane_prefill/requant.rs` | consumes both katgpt-types tiers | 2.125 / 1.75 |
| LUT dequant substrate | katgpt-rs `dequant_dot_via_lut` (4.4–5.6× over scalar, shipped); StreamDQ Plan 431 | LUT consume | — |
| Healer drafter | riir-clippy `TernaryDraftModel` (L1; small cache-resident matrices — the LOSING regime per Issue 582 G2b / the paper's LNL row) | bit-planes | 2.0 + scale |
| SenseModule directions | `katgpt-core/src/types.rs` `TernaryDir` pos/neg bits | 2 bit-planes | 2.0 (fixed-width, tiny — not a target) |

**Nothing anywhere measures z. Nothing compacts signs. Nothing adapts rate to the distribution.** Every shipped format is fixed-rate — across a now-THREE-container ladder whose best tier (trit, 1.75) already sits below the 5-trit-in-128-block rate the paper benchmarks (1.750 = exactly it: 26 B payload + 2 B scale per 128).

**Win band against BOTH shipped tiers (the honest arithmetic, symbols+scale):** BITCOS = 2.125−z. vs bit-plane 2.125: wins at every z (1.16–1.32×, the paper's table). vs trit 1.75: wins iff z > 0.375 — Bonsai 1.7B (z=0.399) 1.01×, 4B/8B ≈ 1.00×, **27B (z=0.297) 0.96× = regression**; BitNet 2B4T (z=0.422) 1.03×; CAT-Q 1.7B (z=0.515) 1.09×. The meaningful-opportunity band vs our BEST tier is **z ≳ 0.43**, which the Bonsai family does not reach.

**Doc-code drift found in passing (attribution corrected per review):** `katgpt-types/src/ternary.rs` L39 documents the bit-plane `TernaryWeights` as "~1.58 bits/weight (log₂3)" — the layout stores 2.0 bits/weight (two planes); log₂3 is the information-theoretic floor, not this layout's rate. riir-ai Research 007's tier table inherited the figure. Fixed at the origin in this commit (comment-only); the riir-ai propagation is the follow-through pointer in §4.

### 2.2 The distilled primitive (open, katgpt-types — retargeted per review; katgpt-core was the wrong home)

A `bitcos` module in **katgpt-types** (beside `ternary.rs`/`ternary_group.rs`/`ternary_trit.rs`, behind an opt-in feature implying `ternary_group_scale`, following Issue 582's `from_group`/`to_group` repack precedent — `pack_from_group` is the third instance of that pattern):

1. **Codec**: `pack_from_group(TernaryGroupWeights)` + `pack_from_planes(pos, neg)` (bitmap = `pos|neg`; signs = compacted `pos` bits at bitmap-set positions — the j-th set position takes the j-th sign bit; sign bit 0 = +1, 1 = −1), `pack_from_f32` (row-wise error-compensated quant per Research 110 §quantization, then pack), `unpack_row`, `dequant_dot`, `to_group`/`from_group` round-trip. Bit-exact roundtrip pinned by test (every (bitmap, signs) pair ↔ pos/neg planes).
2. **z-meter**: popcount-based zero-density report per tensor/model — the cheap measurement that gates everything downstream (and corrects the 1.58-doc drift with real numbers for OUR checkpoints).
3. **GEMV consumer** (portable first): scalar + AVX2-class (byte-materialization `wi = pi − 2ni`; pdep via `u64::deposit` where available — no stable Rust pdep intrinsic; use `core::arch::x86_64::_pdep_u32` under `cfg(target_arch)` per the shipped_target_feature law: runtime `simd_level()` probe, never a compile-time `target_feature` cfg on a shipped path) + the **LUT variant** (256-entry, presence-nibble+sign-window key — the Xe2 trick, which is ALSO the CUDA/WGSL-portable one: no pdep needed, `__constant__`/shared-memory table).
4. **Roofline harness**: measure γ (L1-resident loop) and effective bandwidth at a z-sweep; gate dispatch on measured `B(z)/γ vs β` — the paper's own LNL row is the canonical negative control and MUST be reproduced as a test that the dispatcher refuses instruction-bound configurations.

### 2.3 Fusion (the cross-pollination section)

- **BITCOS × our 2-bit planes (the free converter)**: our pos/neg planes ARE an uncompressed BITCOS (bitmap = `pos|neg`, sign = `pos` masked). Conversion is O(n) popcount/shift, offline, bit-exact — a serving fleet can A/B the layout on the same checkpoints with a ~100-line packer. This is the cheapest credible GOAT gate imaginable.
- **BITCOS × LUT substrate** (`dequant_dot_via_lut`, StreamDQ Plan 431, `threadgroup-subset-sum-lut-keyed-by-activation` corpus rule): the Xe2 256-entry LUT is the same consume mechanism — the fusion is a CUDA/CubeCL BITCOS GEMV using a shared-memory LUT (pdep-free). Riir-gpu's `gemv_ternary_cuda_raw.rs` gains a BITCOS arm.
- **BITCOS × telescoped-digit FMA** (corpus rule `telescoped-digit-coefficients-byte-fma-dot`): the compacted-sign window consumption could fold into activation-side coefficients — an unexplored combination (paper uses masked moves/LUT, not telescoping).
- **Zero-density-gated format dispatch** (the modelless meta-primitive): `z > 0.375 ∧ bandwidth-bound → BITCOS; else 2-bit planes` — a *measured, per-model* dispatch, sigmoid-free, one comparison. This is the generalization the corpus rule family (`bool-fields-over-bitset-hot-reads` — "pack when bytes streamed dominate; don't when read frequency dominates") already states qualitatively; BITCOS gives it a quantitative z-threshold + roofline form.
- **riir-clippy kernel_opt corpus rule candidate** (future mining batch, not this session): "distribution-adaptive-bitmap-compacted-sign-layout" with the regime boundary rider (wins bandwidth-bound; loses instruction-bound — LNL numbers as the negative anchor).

### 2.4 Honest scoping (what BITCOS is NOT for us)

- **NOT a lever for the league's tg128**: our 4090 decode tg128 is GPU-busy-bound at the stage-sum intrinsic (94.6–94.9); batch-128 GEMM is compute-bound — weight-format density does not move it. Prefill is compute-bound — likewise.
- **NOT a Bonsai-27B footprint lever**: BITCOS+scale = 1.828 bits/w (6.16 GB) vs our OWN shipped trit tier at 1.75 (5.90 GB) — **0.96×, a regression against the best in-tree tier**; 4B/8B ≈ parity. The footprint/traffic case vs our best tier lives only at z ≳ 0.43 (BitNet 1.03×, CAT-Q-class up to 1.09×).
- **NOT for cache-resident consumers**: the healer L1 drafter and PlasmaPath game paths run small matrices L1/L2-resident — instruction-bound territory; the paper's LNL row AND Issue 582's G2b note ("cache-resident ⇒ the traffic saving has nothing to pay the decode with; a loss here is expected and shippable — `binary_plasma` precedent: storage PASS + latency FAIL") both say BITCOS loses there. Do not wire the drafter.
- **IS for**: (a) high-z checkpoints (z ≥ 0.43: BitNet, CAT-Q, Maple-class) on the bandwidth-bound serving lane — where Issue 582's own header observation holds ("Ternary GEMV is memory-bound on every backend we ship"); (b) the **z-meter + roofline dispatch as a general primitive** — the four-tier ladder (bit-plane / trit / bitcos / binary) selected per-model by MEASURED z and measured γ/β, which is the actual durable value independent of any single checkpoint's density; (c) any future CPU serving lane (the 4090 host is an i7-13700K — Raptor Lake 8P+8E hybrid; expect the paper's ARL picture — 1.02–1.15× on P-cores, E-cores at the knee — not the EMR 64c figure; the measured dispatch decides, not the paper's table).
- **M3 caution**: M3-class (few cores, generous per-core bandwidth) resembles LNL more than EMR — the roofline harness must decide, not the paper's numbers.

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|---------|
| **Super-GOAT** | — | not reached: no new behavior class (Q2 weak — it is a storage/throughput primitive, not a capability), selling point partial (Q3). |
| **GOAT** | Provable gain over existing approach | **Candidate after our own bench** — the paper's gains are on Intel platforms vs LIBXSMM; our GOAT claim requires the Issue-864 gate (pack roundtrip + GEMV z-sweep + roofline dispatch + LNL-style negative control). Until then: Gain. |
| **Gain** ✅ | Incremental, useful, actionable | **This verdict.** Issue 864 (katgpt-rs): `bitcos` codec + z-meter + LUT GEMV + roofline dispatch behind feature flag. Fusion pointers recorded for riir-ai (serving decode-batch-1, `gemv_ternary_cuda_raw` BITCOS arm) and riir-clippy (corpus rule). |

**One-line reasoning:** no shipped surface measures z or adapts rate — BITCOS adds the missing adaptive rung to a three-container fixed-rate ladder with a bit-exact, training-free conversion from `TernaryGroupWeights` — but the gain is regime-gated (bandwidth-bound only) AND tier-relative (clear vs the 2.125 bit-plane tier at every z; vs the 1.75 trit tier only z > 0.375, a 0.96× regression at Bonsai-27B), so it must be proven on our platforms against BOTH shipped tiers before any promotion (Issue 582's G2b/LNL are the standing counterexamples).

**MOAT gate (§1.6) — katgpt-rs:** in scope ("quant-aware inference", transformer stack weight path; the container tier is already THIS repo's — katgpt-types Issues 578/582). The codec + roofline dispatch is a base primitive; public. The *measured z-atlas of our checkpoints* and the CUDA/CubeCL consumer arms are private value (riir-ai) — the note records the split, Issue 864 builds only the open half in katgpt-types.

**Modelless-first (§3.5):** trivially satisfied — zero training content (panel legitimately skipped: clearly inference-side). Path 0 inventory: every component (layout, pack, unpack×3 ISA, roofline, z-measurement) extracts modelless; no analog ships for any of them except the LUT-consume *mechanism* (StreamDQ/dequant_dot_via_lut) which is a consumer, not the layout.

**Defend-wrong PoC (§3.6):** no quality-parity claim made (storage layout is bit-exact by construction — the paper's own argument; correctness pin is a roundtrip test, not a PoC). Performance claims are attributed to the paper's platforms and explicitly NOT claimed for ours until the Issue-864 gate runs. No coverage dismissal required signal-diff beyond §2.1's table (fixed-rate vs adaptive is a formula-level difference, not a name match).

**Prior art (§4 searches, run 2026-09-21):**
- Headline ("BITCOS ternary layout", "presence bitmap compacted sign vector zero density") → only the paper itself + coverage (alphaxiv/emergentmind/YouTube summaries).
- Closest published prior art for the layout CLASS: **ExTernD** (arXiv:2607.13511, Jul 2026) — "sparse mask+sign packing" for ternary PTQ. Delta: ExTernD is a PTQ method that *trains in* its sparsity (accuracy-affecting, needs calibration/training); BITCOS is layout-only, bit-exact, on existing checkpoints. Same mask+sign storage shape though — the note records this honestly: the *storage shape* is not novel per se; the *serving-kernel + checkpoint-measurement + roofline* package is.
- **TENET** (Microsoft, 2025-09): sparsity-aware LUT-centric *architecture* (hardware co-design) — different layer.
- **Sakana "Sparser, Faster, Lighter"**: unstructured-sparsity GPU kernels for general LLMs (not ternary-specific layout).
- **Ternary CSC / Sparse-BitNet / Sherry**: covered in §1.5 — index-based or training-dependent; neither applies bit-exactly to existing checkpoints.
- Workspace sweep: no z-adaptive anything (§2.1); corpus rules fixed-rate only.

**Weakest point (named by us, sharpened by the round-3 review):** the realized gains on OUR platforms are unproven, regime-gated, AND tier-relative — (1) against the best shipped tier (`ternary_trit_pack`, 1.75) BITCOS is a 0.96× regression at Bonsai-27B and ≈1.00× at 4B/8B, so the flagship-model footprint case evaporates and the win band narrows to z ≳ 0.43 checkpoints; (2) two of our three obvious consumer surfaces (healer drafter: cache-resident; league tg128: GPU-busy) sit in the losing regime; (3) the remaining value concentrates in the z-meter + roofline dispatch primitive itself, whose payoff depends on Issue 864's T4/T5 negative controls being built as designed.

---

## 4. Actionable Outputs

- **Issue 864** (katgpt-rs/.issues/864_bitcos_distribution_adaptive_ternary_codec.md): the open half — codec + z-meter + LUT GEMV + roofline dispatch + GOAT gate, in **katgpt-types**, benchmarked against BOTH shipped tiers (bit-plane 2.125 AND trit 1.75 — passing vs bit-plane while losing to trit is a FAIL vs the best shipped tier).
- **Doc-fix at origin (this commit)**: `katgpt-types/src/ternary.rs` L39 documented the 2.0-bit-plane `TernaryWeights` as "~1.58 bits/weight (log₂3)" — corrected to state the actual 2.0 rate with log₂3 named as the floor. **riir-ai follow-through**: Research 007's tier table inherited the 1.58 figure for the Plasma row — correct it there once Issue 864 T2 produces OUR measured z (file in riir-ai at that time).
- **riir-ai pointer** (recorded here, no issue filed): once Issue 864's gate is green AND a served checkpoint measures z ≥ ~0.43, a BITCOS arm beside `TernaryHandle`/`TernaryTritHandle` for the bandwidth-bound decode slice + a z-atlas of our checkpoints (the paper's numbers are for *their* downloads; ours should be measured from our own files).
- **riir-clippy pointer**: `distribution-adaptive-bitmap-compacted-sign-layout` as a kernel_opt corpus rule candidate in a future mining batch, with the LNL/G2b regime-boundary rider.

## Review record (§5)

- Round 1–2: `request_verdict` (claude_code backend) timed out twice (180 s each).
- Fallback per skill §5: spawn_agent reviewer returned **AGREE** with two non-blocking nits (RPL host attribution; riir-ai doc-fix follow-through) — both applied.
- Closing `final_round` call reached the claude_code backend (round 3/3) → **REVISE** with three substantive corrections, all verified against the tree and applied: (1) the ternary container tier is OWNED by katgpt-types (`ternary_group.rs` Issue 578, `ternary_trit.rs` Issue 582 — the latter absent from the original inventory), not riir-ai; (2) the win band must be stated against BOTH shipped tiers — BITCOS is a 0.96× regression vs the trit tier at Bonsai-27B, narrowing the opportunity to z ≳ 0.43 + the dispatch primitive; (3) target crate retargeted katgpt-core → katgpt-types with Issue 582's repack precedent and G2b citation. The reviewer explicitly held the Gain-tier conclusion unchanged ("None of this touches the AGREE-level conclusion that this is a Gain"). Round cap hit with no remaining disagreement — corrections adopted in full.

## References

- [arXiv:2609.16338](https://arxiv.org/abs/2609.16338) — Georganas, Heinecke, Dubey, "Breaking the 1.58-bit Barrier for Ternary LLMs", Intel, 2026-09-14.
- Their SOTA baseline: Georganas et al., "Pushing the envelope of LLM inference with ultra-low-bit quantized models", arXiv:2508.06753 (LIBXSMM/XeTLA 2-bit kernels).
- ExTernD: arXiv:2607.13511 (closest layout-class prior art, PTQ-side).
- Workspace: Research 110 (Ciot substrate), Research 418 (StreamDQ LUT), riir-clippy kernel_opt corpus (`base-3-trit-packing-5-per-byte` + the B20 regime caveat, `telescoped-digit-coefficients-byte-fma-dot`, `swar-lane-widened-base3-dp4a-dot`, `threadgroup-subset-sum-lut-keyed-by-activation`, `bool-fields-over-bitset-hot-reads`).

> **PASS-Redirects (synthesis, added 2026-10-20):** Yang et al. [arXiv:2509.16989 "PTQTP: Post-Training Quantization to Trit-Planes for Large Language Models"] — the dual-trit-plane PTQ decomposition (2×1.58-bit, multiplication-free); training/calibration-side, riir-train routing note. Grella [arXiv:2608.08910 "Tied Trit-Planes: Constraining PTQTP to a Uniform Nine-Level Quantizer, with a Persistent Folded Format for Disk-Streamed Mixture-of-Experts Serving"] — the layout-class sibling this note's §4 prior-art sweep missed (published 2026-08-08, six weeks before this note): tying the two planes' per-group scales at ratio 3 collapses the pair to a uniform nine-level quantizer (the balanced-ternary identity), folding LOSSLESSLY into one 4-bit code plane served in a single integer dot pass — measured vs Q4_K: 6.7% faster decode, 9% smaller files, on SSD-streamed DeepSeek-V4-Flash MoE. Deltas vs this note's verdict: the fold MECHANISM itself already ships in-corpus under an operator name (`telescoped-digit-coefficients-byte-fma-dot`, the a+3b digit telescoping); the paper's genuinely new atoms are (a) the SCALE-TIE precondition that makes the fold lossless (a format law, not a kernel rule) and (b) the disk-streamed-MoE serving posture (4-bit persistent bytes) — a different regime from our 2-bit bandwidth-bound GPU decode lane, hence detection-only, no batch. Cited here so future arXiv-ID greps land on this record.
