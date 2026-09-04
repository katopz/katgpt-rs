# Bench 691: Numeric-Stability Deviation-Contextualization Probe — Phase 1 GOAT (Issue 697)

**Status:** Phase 1 GOAT **PASS** (G1 ✓ golden/idempotence/planted-gate, G2 ✓ 151–163 µs/report @ N=4096 vs 250 µs bound, G3 ✓ 1978 default / 1996 feature lib tests, G4 ✓ steady-state 0 allocs). **Phase 2 LANDED 2026-08-29** (attention lab + T3.2 schedule — see the Phase 2 section below; feature lib 2005/0/9, default 1978/0/7 unchanged). Opt-in `numeric_stability`. T3.3 (consumer follow-ups) files on first consumption; riir-train side: Issue 492.

> **Issue:** katgpt-rs Issue 697 · **Research:** 515 · **Paper:** "Is Flash Attention Stable?", [arXiv:2405.02803](https://arxiv.org/abs/2405.02803) (Golden et al., Meta FAIR + Harvard 2024) · **Feature:** `numeric_stability` (opt-in) · **Date:** 2026-08-28 · **Box:** M3 Max (G2 numbers release; suite counts debug unless noted)

---

## What shipped (T1.1–T1.5 + T3.1)

`crates/katgpt-core/src/numeric_stability/` (`mod.rs` module docs + `probe.rs` implementation), feature `numeric_stability = []`, no new deps.

| Item | API | Notes |
|---|---|---|
| T1.1 format emulator | `truncate_mantissa(value: f64, bits: u32) -> f64` | Clears the low mantissa bits (RTZ in mantissa space — the paper's precision knob). Sign-preserving, idempotent (pinned), ±0 bit-exact. Documented honest edges: mantissa-width-ONLY (no exponent modeling — f16 overflow etc. out of scope); subnormals lose low bits (may truncate to signed zero); NaN payloads may collapse to ±Inf when all payload bits sit below the cut (measurement paths validate finite first, so unreachable there); `bits >= 52` is the identity; f32-representable normals are the identity at `bits >= 23` while f32 subnormals can lose bits at any width. |
| T1.2 deviation report | `DeviationReport { max_diff, wasserstein_1d }` + `compute` / `compute_into` | Two-surface protocol (elementwise bound + distribution-aware). `compute_into(x, y, &mut Vec<f32>, &mut Vec<f32>)` is the zero-alloc hot path (grow-only scratch, Plan 418 `*_into` pattern). 1-D Wasserstein DELEGATES to `mag::transfer` — the shared quantile-grid core (`wasserstein1d_sorted_core` + `quantile_interp`, extracted verbatim from the shipped `wasserstein1d`, which now calls the same core; behavior-identical refactor). Errors: `LengthMismatch` / `Empty` / `NonFinite` (NaN rejected at the boundary — the documented NaN policy — so the substrate sort only ever sees a total order). |
| T1.3 acceptance rule | `accept(reports: &[DeviationReport], refs: &ReferenceBands, margin: f32) -> Verdict` | Dominance vs the BINDING (smaller) band per surface: `report.m < band_m·margin` passes, `== ` is the margin line, `>` rejects. Quantifier: every report × every surface. Any reject → `Reject`; else any margin-line → `Inconclusive`; else `Accept`. Guard rails: empty reports / non-finite-or-negative margin / invalid bands → `Inconclusive` (no verdict from nonsense); non-finite or negative report values → `Reject` (fail-closed — NaN compares false and would otherwise silently pass). **`margin` is an explicit parameter — NO default exists and none is derived from the paper's context-specific 2–5× headline** (doc-truth: `margin_has_no_default_doc_truth`). |
| T1.4 reference builders | `reference_r1_two_draws(a, b)`; `roundtrip_truncate_mantissa{,_into}`; `reference_r2_roundtrip(baseline, bits)`; `reference_r2_custom(baseline, quantize)` | R1 = two draws from the caller's init distribution. R2 = ONE quantize→dequant round-trip, labeled **"single-step lower bound"** everywhere it appears (`R2_LABEL` const + module docs + both builder docs + the `ReferenceBands::r2` field doc), pinned by `r2_lower_bound_label_tripwire`: (a) label present ≥4× in the API docs + once in module docs, (b) behavior pin — the band is bit-identical to EXACTLY ONE truncation pass (an in-test manual round-trip), (c) demonstration — a composed two-event pipeline (amplify between quantizations) exceeds the single-step band across a deterministic ladder. The round-trip cast back to f32 is exact by construction (truncation preserves f32-representability; no double rounding). |
| T1.5 scope-limit tripwire | module docs + `scope_limit_tripwire` | Pinned sentence: "Scope limit: this protocol bounds DIVERGENCE SIMILARITY only — it is NOT a training-stability proof" + the arXiv:2510.04212 citation (the stability mechanism's owner) + negative greps (the docs must never CLAIM stability). Whitespace-normalized matching (doc wrapping survives reflow). |
| T3.1 falsifiability | `t31_planted_deviation_gate_accept_margin_line_reject` | Deviations planted at 0.1× / 1.0× / 10× of the binding max-diff band land **Accept / Inconclusive (exact margin line) / Reject** at `margin = 1.0`. The plant is bit-exact (zeroed plant site: `|y[0] − 0| == y[0]`), the verdict uses exact f32 comparisons, and a fixture sanity assert pins the W1 surface strictly inside the band (single-element plants barely move the distribution) so a future fixture change fails LOUD instead of silently flipping verdicts. A gate that cannot fail proves nothing — this one demonstrably can. |
| Substrate cross-pin | `mag::transfer::tests::scalar_into_matches_multidim_substrate_bit_exactly` | `wasserstein1d_scalar_into` (new `pub(crate)`, feature-gated) ≡ `wasserstein1d` on `[[f32; 1]]` views, bit-exactly. Plus `wasserstein_matches_independent_reference`: the report's W1 ≡ an in-test transcription of the quantile grid (stable `total_cmp` sort twin) — pins the metric against substrate drift. |

## GOAT gates

| Gate | Result |
|---|---|
| **G1 correctness** | **PASS** — golden known-values for the emulator (3.0@width1 stays 3.0, 3.0@width0 collapses to 2.0, f64::MAX@width0 → 2^1023, ±0 bit-preservation, documented NaN→Inf + subnormal-cut pins with exact bit patterns); idempotence ladder (512 LCG f64s × 10 widths, `to_bits` equality); golden quantile-grid report (`[0,1]` vs `[0.5,1]` → max_diff 0.5, W1 0.125 exact); error paths (length/empty/NaN/Inf); determinism suite (bit-identical repeats, symmetry `compute(x,y) ≡ compute(y,x)`, permutation invariance via a deterministic Fisher-Yates); acceptance polarity + margin-moves-the-line + all guard rails; **planted-deviation gate 0.1×/1×/10× → Accept/Inconclusive/Reject**; R2 label tripwire (docs + single-round-trip behavior pin + composed-exceeds-single demonstration). |
| **G2 perf** | **PASS — 151–163 µs/report at N=4096** (release, 5 runs: 151.4 / 152.6 / 154.6 / 163.1 / 162.4 µs; bound 250 µs = 1.5× the observed ceiling; the `#[cfg_attr(debug_assertions, ignore)]` house pattern). The two 4096-element comparator sorts dominate (~37 ns/element including the quantile grid) — W1 is the expensive surface by design; `max_diff` alone is a linear scan. |
| **G3 no-regression** | **PASS** — `cargo check -p katgpt-core` (default) clean; default lib suite **1978 passed / 0 failed / 7 ignored** (identical to pristine HEAD — the feature is opt-in, and the only default-visible change is the behavior-identical `wasserstein1d` refactor, which the same 1978 validate); feature-state lib suite **1996 passed / 0 failed / 8 ignored**, green 3× consecutively; `cargo clippy -p katgpt-core --lib` default AND `--features numeric_stability`: **0 warnings**. |
| **G4 alloc** | **PASS** — `g4_compute_into_steady_state_zero_alloc` (debug, repo `TrackingAllocator`): after one warm-up call, 20 × `compute_into` = **0 allocations** (the first implementation failed this gate with 2 allocs/call — std's comparator `sort_by` allocates its merge buffer; fixed by moving the shared substrate sort to the in-place unstable sort, see the deviation note below). `accept` allocates nothing by construction; band builders are cold-path by design. |

## Honest notes / deviations

1. **Unstable sort in the shared substrate path (deliberate, documented).** The zero-alloc gate forced `sort_unstable_by` in `mag::transfer::sort_f32_ascending` (std's comparator `sort_by` allocates a merge buffer → 2 allocs/call measured). Determinism is unaffected: the unstable sort is a deterministic algorithm and the metric reads only the sorted VALUE sequence, in which equal f32s are bit-identical and interchangeable — bit-identity to a stable sort is pinned by the in-test stable-sort twin (`wasserstein_matches_independent_reference`). `wasserstein1d`'s values are unchanged (its own tests green; the refactor moved the per-dimension math verbatim into `wasserstein1d_sorted_core`).
2. **Delegation shape.** The issue offered "widen `wasserstein1d` to `pub(crate)`" or "thin wrapper inside transfer.rs reusing sort + quantile_interp". Shipped the SECOND form (`wasserstein1d_scalar_into`, feature-gated `#[cfg(feature = "numeric_stability")]` so the default build stays dead-code-free) BECAUSE the zero-alloc `_into` requirement needs scratch-carrying sort buffers the allocating `wasserstein1d` cannot accept. Both paths share one grid core; nothing is duplicated in the consumer module.
3. **The margin has no default — enforced, not just documented.** The paper's "2–5×" is a footnote in the docs (context-specific to Meta's model/seq-len/hardware), grep-pinned absent as a constant.
4. **G2's bound is headroom-based, not aspirational:** 250 µs = 1.5× the observed 5-run ceiling on this box; the honest measured range is recorded here.
5. **Debug-suite citizenship:** the first G4 draft (100 × n=2048 in debug) perturbed the parallel debug timing gate `subspace_phase_gate::jacobian_svd_r8x8_latency_gate` (<100 µs guard, 65.8 µs isolated) into failing under full-suite load (227 µs, 2/2 feature-suite runs, while pristine HEAD's full suite passed). Lightened G4 to 20 × n=1024 (equally probative — an alloc-free path shows 0 after one steady-state call); the feature suite then passed 3× consecutively. Recorded per the house flake rule: debug timing gates under parallel load are load-sensitive; nothing in the shipped code regressed.
6. **Doc-truth greps are whitespace-normalized and scoped to the API/docs region** (everything before the test module) — `include_str!` reads the whole file, and the test module's own needle literals would otherwise self-trip the negative assertions.

## Scope (what is deliberately NOT here)

- **Phase 2 (T2.1/T2.2) — LANDED 2026-08-29**, see the Phase 2 section below.
- **T3.2 `tol(S)` schedule — LANDED 2026-08-29** (pinned table + two-length probe; the Issue 753 f16-KV consumer wiring remains T3.3's).
- **T3.3 consumers** — the riir-ai gate layer consumes first (do not fork the probe); the riir-train drift probe + divergence ledger is riir-train Issue 492.
- **Anti-claim held:** nothing here establishes training stability (arXiv:2510.04212 owns the mechanism); the probe contextualizes divergence similarity only.

Run it:

```bash
cargo test -p katgpt-core --features numeric_stability --lib numeric_stability
cargo test -p katgpt-core --features numeric_stability --lib --release numeric_stability -- --nocapture   # G2 numbers
cargo test -p katgpt-core --lib                                                                            # G3 default
cargo clippy -p katgpt-core --features numeric_stability --lib
```

---

# Phase 2 — the perturbable reference attention lab + the tol(S) schedule (Issue 697 T2.1/T2.2/T3.2, 2026-08-29)

**Status: LANDED — G1 ✓ (two-tier golden identity + 9 gate tests), G2 n/a-by-design (offline instrument, no hot path — honest scope recorded), G3 ✓ (feature 2005/0/9, default 1978/0/7 unchanged, clippy 0 both states), determinism ✓ (bit-identical repeats pinned).** New file `crates/katgpt-core/src/numeric_stability/lab.rs` (~700 lines incl. tests), same opt-in feature, no new deps.

## What shipped

- **T2.1 — the lab**: `lab_attention` (scalar f64, canonical flash-attention update order: rescale running `l`/`acc` BEFORE adding the new tile's PV; direct division, not reciprocal-multiply — that choice is load-bearing for the golden identity) over knobs `LabConfig { seq_len, head_dim, bc, br, axis_swap, mantissa_bits, quantize_ops, scale }`; `naive_attention_f64` (two-pass FP64 golden, ascending-`j` accumulation); `truncate_mantissa` reused from Phase 1 (storage quantization on inputs always; `quantize_ops` extends truncation to every intermediate op — the "arithmetic in format F" emulation).
- **T2.2 — the four ordering laws pinned as tests** (the paper's constants deliberately NOT imported):
  1. **Mantissa law** (`mantissa_ordering_law_two_lengths`): deviation non-increasing across a 10-point ladder (6→52 bits) at S=128 AND S=256. PASS.
  2. **Rescale-count law** (`rescale_count_spearman_grid`): Spearman ρ(R, dev) over a 3×3 (S,T) grid = **0.8892** (floor pinned 0.7); endpoint law R=15 strictly exceeds R=0. PASS.
  3. **Tile-area law** (`tile_area_ordering_two_formats`): larger Bc → less deviation, reproduced at bits=10 AND bits=16. PASS.
  4. **Dim-order swap + square control** (`dim_order_swap_changes_deviation_square_invariant`): swap measurably changes deviation at (Bc=64, Br=16) — std 9.143e-3 vs swapped differs; **square row (Bc=Br=32) is BIT-IDENTICAL under the swap** (the paper's free negative control). PASS.
- **T3.2 — the `tol(S)` schedule**: pinned table `TOL_TABLE_PINNED` (4 rows, S ∈ {64,128,256,512}, measured deviation × `TOL_HEADROOM`=2.0 policy), `band_at` linear interpolation, blake3 fit-inputs hash `TOL_FIT_HASH_PINNED` (config pinned exactly; row VALUES pinned within the ±20% cross-libm band `TOL_FIT_BAND` — `f64::exp` is platform-libm, the honest portability concession). **Two-length probe through the Phase-1 acceptance rule** (`tol_schedule_table_two_length_no_class_flip`): at S₀=64 Accept under its own band; at 8·S₀=512 the STALE S₀ band REJECTS (the flip hazard, asserted non-vacuous); under the SCHEDULE band at 512 still ACCEPTS — the verdict class is preserved, which is the issue's requirement.

## Measured (M3 Max, debug-profile suite runs, deterministic inputs LAB_SEED)

| Quantity | Value |
|---|---|
| Multi-tile f64 golden (S=128, D=16, Bc=Br=32, R=3) | max rel diff **1.523e-15** (≈ 7 ulps; pinned bound 3.0e-15) |
| Table rows (bits=10, D=16, T=32, ×2 headroom) | md: 1.0273e-2 / 2.3168e-2 / 4.8828e-2 / 9.3769e-2 · w1: 6.6093e-3 / 1.7704e-2 / 3.8393e-2 / 7.7170e-2 |
| Deviation growth 64→512 (md) | **9.1×** (R ratio 15/1 = 15 — sub-linear in R at this range, inside the 32× envelope assert) |
| Spearman ρ(R, dev), 9 grid points | **0.8892** (floor 0.7) |
| Swap (Bc=64, Br=16, S=128, bits=10) | dev_std = 9.143e-3, ≠ dev_swapped |
| Feature lib suite | **2005 passed / 0 failed / 9 ignored** |
| Default lib suite | **1978 passed / 0 failed / 7 ignored** (unchanged — feature-gated) |
| clippy (default + feature states) | **0 warnings** |

## Honest notes

1. **The golden identity is two-tier because it must be.** Single-tile configs are BIT-IDENTICAL to the naive reference (the correction is exactly `exp(−inf)=0`, the running max IS the global max, and every accumulation matches the naive order — including `0 + lsum = lsum` exactness and direct division, NOT reciprocal-multiply). Multi-tile exact equality is STRUCTURALLY impossible (fp addition is not associative across the rescale-grouped numerator — that non-associativity IS the measured mechanism), so the gate is the pinned 1.5e-15 relative bound. Any future "simplify the lab to reciprocal-multiply" would silently break the bit-identity tier.
2. **The dimension-order knob is the axis ASSIGNMENT swap, and the square-tile control is why.** On a scalar emulator a pure loop re-nest cannot change bits (per-element accumulation order is unchanged) — the implementable form of the paper's dim-order knob is swapping which tile size rides Q vs K. At Bc ≠ Br that changes the partition (and R — the paper's fixed-R isolation is documented, not reproduced); at Bc == Br it is provably the identity, which is EXACTLY the paper's square-tile negative control, pinned bit-identical. The paper's law survives as: swap measurably changes deviation (non-square) + square row invariant.
3. **G2 is honestly n/a by design**: the lab allocates per call and runs scalar f64 — it is a measurement apparatus for gates and pre-swap triage, never a hot path; the Plan 418 zero-alloc convention governs production paths and Phase 1's probe (which ships `_into`), not this instrument. Recorded so nobody "optimizes" the lab into a production kernel by accident.
4. **The debug-timing flake class fired once during validation** (`subspace_phase_gate::jacobian_svd_r8x8_latency_gate`, a <100 µs debug guard, failed once under a sibling's concurrent cargo build at 70% CPU, passed on immediate re-run and in the final green suite) — the same load-sensitivity already recorded in Phase 1's note 5. Nothing regressed; recorded per the house flake rule.
5. **The TOL_HEADROOM=2.0 is a documented POLICY constant** (the lab's own deviation must sit strictly inside the band so the acceptance verdicts are strict comparisons, not margin-line ties → Inconclusive); it is NOT the paper's 2–5× (that stays grep-absent per Phase 1's rule).
6. **Cross-libm portability**: same-platform re-measurement is bit-exact; the pinned rows are asserted within ±20% so the gate survives the M3↔4090 libm difference. The fit-input HASH pins the config exactly — config determinism and value portability are separate pins by design.

Run it:

```bash
cargo test -p katgpt-core --features numeric_stability --lib numeric_stability::lab
cargo test -p katgpt-core --features numeric_stability --lib numeric_stability::lab -- --ignored --nocapture print_measurements   # re-pin constants
cargo test -p katgpt-core --lib                                                                            # G3 default
cargo clippy -p katgpt-core --features numeric_stability --lib
```

## Resolution — Issue 697 CLOSED (2026-08-29, all tasks done; file removed per the noise-reduction rule)

- **T3.2** — `TOL_TABLE_PINNED` + `band_at` + blake3-pinned fit landed (commit `c5139028`, with Phase 2).
- **T3.3 (first consumption)** — riir-ai `23d0b775f`, [Bench 798](../../riir-ai/.benchmarks/798_numeric_stability_first_consumption.md): the `cpu_reference` f32-accumulation claim (riir-gpu, Issue 709 H2) contextualized via R1/R2 — Accept at margin 1.0 on all three surfaces (gap 175–744× below the single-step f16 precision-change band); "f64 optional" is now measured evidence. Planted controls live (10× band → Reject). Consumed via feature-forward, not forked.
- **GPU-lane follow-ups filed** — riir-ai `Issue 775` (Bench-773-successor gate → `accept()` contextualization, M3 Metal window) + `Issue 776` (qwen38 f16-KV gates → `tol(S)` schedule, 4090 lane). riir-train side remains Issue 492.
