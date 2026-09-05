# Issue 723 — the first full-workspace EXECUTION is red: 47 targets, six distinct classes

**Status:** OPEN (T1–T8 ALL DONE; open only on the two follow-ups it spawned)
— filed 2026-09-04 by Issue 718 T3(a) (the full-workspace pricing run, the
first time anything ever *executed* this workspace's test surface). **Classes
A, A2, B, D, E and F are all FIXED.** Class C is RESOLVED BY MEASUREMENT (the
gates are correct at their own feature sets and meaningless under
`--all-features` unification — T3), which is a finding, not a fix, and it is
why the full unified run still cannot pass at any budget. What remains is not
this issue's content: `.issues/726` (`gauge_rebalance` is 3.7x its paper
target) and `.issues/727` (SP-KV misses both T16 bars) are the two real
primitive shortfalls that T7's repaired instruments exposed. 718 is CLOSED and
REMOVED (durable record: `.docs/10_audits/ci_compile_vs_execute_axis.md`); this
issue inherits the reds it found. The cadence question 718 left open is
answered by its own reds: the full `--all-features` run **cannot pass at any
budget** while Class C stands, so it stays dispatch-only and the scoped weekly
job is the standing executed tier.

## Provenance — why this was invisible until now

718's measurement: no automatic trigger in this repo ever ran `cargo test`.
`cargo clippy --workspace --all-targets --all-features` **compiles** every
target and executes none, so 508 targets' assertions were *unknown*, not
passing. T3(a) priced the missing run and, in doing so, ran it. The reds
below are not a regression — they are the accumulated, never-observed state
of a surface that had never been executed as a whole.

Cost and census of the run itself: `.benchmarks/701_full_workspace_execution_pricing.md`.

## The classes

Read the class before the count — these do not share a fix, and pooling them
into "47 red targets" is what would make this issue unactionable.

### Class A — wall-clock / throughput bars, measured on a box at load 8-16 (8) — FIXED

`bench_001_pruners_goat_proof` (G5 A/B, 31.9 ms vs 54.8 ms),
`bench_231_union_bound_goat` (G4.4 scaling 643.6x vs a 250x bar),
`bench_257_gpart_adapter_goat` (G2 206.1% vs ≤200%),
`bench_270_gauge_invariant_goat` (t08 30.9 µs > 5 µs),
`goat_234_manifold_pruner` (G7 throughput), `substrate_gate_goat`
(G2 ratio 30.267x), `bench_sp_kv` (gate-bias overhead),
`spec_reconciliation_bench` (p50/p99 bars).

These are the surface the 718 T3(b) scoping decision **deliberately left
workstation-owned** — a Linux runner cannot hold them. The action was framed
as a per-target choice between the `make-wallclock-gates-load-invariant`
treatment (interleave a control, assert the per-pair ratio) and an explicit
`#[ignore]` with a reason string.

**The framing was wrong, and that is the class's real finding.** Applying the
load-invariant treatment first — before deciding anything — showed that
**the box was the sole cause of exactly zero of the eight**. Seven were broken
instruments (an arm the optimiser had deleted; setup inside the timed region;
a clamped denominator; a loop-invariant baseline hoisted out; adapter
construction charged to one arm only; a "50% pruned" arm that pruned nothing;
a p50 over five samples) and one was an honestly unreachable bar. Load noise
was never the top term in any of them, and three of the eight moved by
**more than 5x** once the instrument was fixed — one by 140x, in the direction
that turned a "30x regression" into a 4.6x *win*. Full record and the
per-target dispositions: T7 below.

### Class A2 — timer-resolution degenerate: `0 ns` → `NaN%` (3)

Distinct from A and **not** a load artifact. In a release build the measured
work drops below timer resolution, the gate divides by a zero baseline, and
`NaN` fails every comparison:

- `bench_gdsd_modelless::goat_169_g3_overhead` — `NoScreeningPruner: 0ns`,
  `GdsdPruner: 0ns`, `Overhead: NaN%`, bar `≤ 20%`.
- `rv_gated_routing::goat_t11_latency_improvement` — `P50 baseline=0ns,
  P50 RV-gated=0ns, improvement=NaN%`.
- `bench_104_mls_k_sweep::bench_mls_stability_across_positions` — `NaN`.

A gate that reports `NaN%` is an instrument that measured nothing, and it is
the mirror of this repo's own "a green test count can be a count of nothing":
here the count is real and the *quantity* is absent. Fix modellessly — raise
the iteration count until the baseline clears resolution, and assert a
denominator floor so a future zero is a loud FAIL rather than a `NaN`.

### Class B — missing linker capability, not a code defect (2 targets, 12 tests)

`__heap_base not exported — add -Wl,--export=__heap_base to linker flags`:
`test_percepta_rust_wasm` (10 tests) and `bench_064_futamura_evaluator`
(2 tests). The wasm compile path needs the export flag. Environmental and
reproducible; it should be a skip-with-reason when the flag is absent, never
a red that trains readers to ignore the suite.

### Class C — fixture pins vs `--all-features` (9)

`issue_698_t1`..`t8` (9 targets). Every one reports fixture hash
`54473b7c30dfb793`, which is *neither* the recorded aarch64 nor the
x86_64-windows pin, and T1 adds the quantitative form: `pinned spectrum
drifted at r=1: measured 8.526659e0 vs pinned 1.241034e1`.

**Mechanism, identified:** fixture weights are seeded from an RNG stream
whose **draw count depends on feature-gated struct fields**, so under
`--all-features` every committed platform pin reads a different hash. Same
mechanism as the `gated_mlp` d2f fix (`d3454eff`) that this run also produced:
an extra weight draw shifts the whole stream downstream.

**So these tests are correct under their own feature set and MEANINGLESS under
unification** — this is the katgpt-rs twin of riir-ai Issue 830's
"`--all-features` was never a checked configuration". It is exactly the caveat
718 wrote down *before* the run — *"`--all-features` on a test RUN is not the
same claim as on a compile"* — now measured rather than predicted.

**Do NOT re-pin to the unified hash.** That destroys the per-feature claim the
pin exists to make. The fix is a feature-set-qualified pin, or running these
gates under their own committed feature sets (owner option (iii) in
`.benchmarks/701`). Confirm cheaply first by re-running ONE target at default
features.

### Class D — hard-coded `target/release` (1) — FIXED

`bench_294_ict_g6` read a literal `target/release` to find the rlib it `nm`s,
so it panicked (`libkatgpt_core-*.rlib not found in target/release`) under a
`CARGO_TARGET_DIR=/tmp/...` run — **the very workflow AGENTS.md mandates**
when another cargo holds `target/`. The gate was unpassable under the
prescribed workflow and nothing noticed, because nothing ran it. Fixed in the
filing commit: the dir is read from `CARGO_TARGET_DIR` and the panic names
the resolved path. Swept — it was the only such site in the repo.

### Class E — quality / correctness reds (13 targets)

The ones that need a per-target read, headed by the only genuine **panic in
library code**:

| target | signature |
|---|---|
| `bench_238_mux_latent_integration` | **panics in `katgpt-core/src/mux_latent/buffer.rs:91` — `chunk size must be non-zero`** |
| `bench_238_mux_latent_model_goat` | G2 logit cosine sim 0.0197 (X4) / −0.0471 (X8) |
| `bench_064_futamura_evaluator` | specialized dims 300 > universal 216 (assert says ≤) |
| `bench_102_tilert_pipeline_goat` | draft logits −0.6473794 vs −0.64738727 — a ~1e-5 tolerance against release-build float reassociation |
| `heterogeneous_g1` | `rps: entry 1 mismatch homogeneous 1 vs heterogeneous 0` |
| `bench_turboquant` | 2-bit score correlation −0.1600, cos_sim −0.0952 |
| `test_drafter_lora_goat` | baseline 0.010 == trained 0.010; assert wants `>` |
| `test_mtp_gating_topk`, `test_mtp_lora_gated_integration` | multi-token never fires with MTP on |
| `test_129_opus_boltzmann_goat` | regret does not converge (759.90 → 768.20) |
| `bench_gdsd_modelless::goat_169_g1` | acceptance gain +0.00% (both paths 389.81) |
| `go_komi_test` | `adaptive_komi_reduces_black_dominance` |
| `bench_171_thinking_prune_goat`, `bench_378_cross_dim_procrustes`, `bench_ldt_lattice_deduction`, `bench_fixed_vs_procedural`, `issue_717_t1_t2` / `t3_t4` | quality / variance bars |

`bench_238_mux_latent_integration` is the one to take first: a library panic
is not a gate tolerance, and `chunk size must be non-zero` reads like an
unguarded divisor on a config the test legitimately constructs.

### Class F — doc-tests (8 targets, 26 doctests) — FIXED

**A seventh blind-spot axis, and the reason it deserves its own row in
AGENTS.md: `--all-targets` does NOT include doc-tests.** So the full gate —
whose whole purpose is to compile everything — never compiled a single doc
example, and 8 crates' doctests had never been built at any revision.

31 doctest lines across 14 files still wrote `use katgpt_rs::...` after the
root crate was split into `katgpt-*` sub-crates that do not (and must not)
depend on the root. Fixed to each crate's own path in the filing commit.
Three more were genuine, independent defects rather than stale paths:

- `multi_agent_path`: `WarmStartCache::new(scheme, 0.0)` — `w_phi` is `usize`.
- `variable_rank_domain_expert`: `variable_rank_router_static!` is
  `#[macro_export]`, so it lives at the crate root, not in its own module.
- `linking_fold::fold_gelu_into`: **the example asserted values its own
  formula never produces.** `gelu_smoothed_abs = sqrt(x² + α⁻²) − α⁻¹` lands
  *under* `|x|` by up to `α⁻¹ = 0.1`, yet the doc asserted `|state[1] − 0.5|
  < 0.01` on a true value of `0.409902`. The tolerances were also inverted
  against the smoothing — loosest (`0.1`) on the coordinate furthest from the
  center, where the shortfall is smallest. Re-pinned to the computed f32
  values at `1e-6` with the sign of the effect written down.

The 24 `ignore`-fenced and 20 prose `katgpt_rs::` mentions were **left
alone**: the prose ones correctly name the root re-export path, and an
`ignore` fence is not compiled. Only the 31 lines inside compiling fences
were defects.

## Tasks

- [x] **T1 — Class F: doc-tests.** 34 defective lines over 15 files (31 stale
      root-crate paths + 3 real defects), residual in compiling fences swept to
      0, and a second layer that only appeared once the first unblocked
      compilation (a missing `WarmStartCache<P>` annotation, a `&` vs `&mut dyn`
      argument, and a `progressive_mcgs` example that never added a root so
      `assert!(res.is_some())` could not hold). **Verified: `cargo test
      --workspace --all-features --release --doc` = 34 suites, 98 passed, 0
      failed, 111 ignored.** Adds the `--all-targets` excludes doc-tests axis to
      AGENTS.md.
- [x] **T2 — Class D: hard-coded target dir.** `CARGO_TARGET_DIR`-derived,
      panic names the resolved path, repo swept for other sites (none).
- [x] **T3 — Class C first, before any other class.** CONFIRMED 2026-09-04:
      `issue_698_t1_gain_spectrum` PASSES under its own committed feature set
      (`--features lt2_looped,loop_stability_fix`, 0.24 s, pinned spectrum
      reproduces within the hybrid band) and fails only under `--all-features`
      unification. The gates are CORRECT under their own features and
      MEANINGLESS under unification — exactly the Issue-830 twin. Per G3 the
      evidence is preserved (no re-pin); the reds are documented-expected under
      unification until owner option (iii) (dedicated lanes) is built.
- [x] **T4 — Class E: `bench_238_mux_latent_integration`.** FIXED 2026-09-04.
      Root cause: `LatentContextBuffer::new_adaptive` fed `config.window_size`
      straight into `chunks()` — and `MuxLatentConfig::default()` carries 0,
      while the encoder resolves 0 to "no windowing" (encoder.rs). The buffer
      path now resolves 0 the same way (one window of all tokens); regression
      test `test_buffer_adaptive_default_window_size_zero_no_panic` pins the
      default-config roundtrip. Validation: mux_latent 38/38 under
      `lclm_adaptive_lod`; the formerly-panicking target 5/5.
- [x] **T5 — Class A2: the three `NaN%` gates.** FIXED 2026-09-04 — and the
      mechanism was NOT timer resolution (the clock is fine; `sleep` measures
      true): **rustc 1.98.1 + fat LTO eliminates inlined-callee work whose
      outer result is dead, even through `black_box` inside the callee** (a
      direct-call-with-used-result measured 16.6 µs; `let _ = f()` over the
      same fn in the same binary read ~0). Fixes: (1) `rv_gated_routing` —
      simulated forwards are xorshift chains whose results accumulate into
      caller-consumed sinks, iters raised to clear the ~42 ns mach tick
      (measured: baseline p50 16,500 ns, RV-gated 1,625 ns, improvement
      **90.2%** ≥ 10% — the gate's first real measurement in its release life);
      (2) `bench_gdsd_modelless` G3 — the fold-elimination was REAL under
      `--features gdsd_distill` alone: the `let _ =` loops measured 0 ns at
      2M iters and the new loud assert fired (+inf printed) — the earlier
      −2.4% reading was a different feature unification's inlining, not a
      stable instrument. Final fix, three defenses: bit-exact data-dependent
      sinks (a per-iteration xorshift mix XORed with the relevance bits —
      the baseline pruner's relevance is a CONSTANT 1.0, so a bare constant
      sink would itself fold), 9 interleaved (baseline, gdsd) chunks with a
      median-of-ratios (two sequential 2M arms measured +5.2% and +21.7%
      thirty seconds apart — a single ratio is box-load-fragile at the 20%
      bar; the Bench 828/831 discipline), and the loud all-chunks-zero
      assert. Verdict-stable ×2: median −0.0% / +0.6%, PASS; per-round
      ranges 0.91–1.07. Debug marked `#[ignore]` with reason (wrapper cost
      ~106% is the profile artifact); (3) `bench_104_mls_k_sweep` — no timing at
      all (its NaN was Inf/Inf cos_sim from exploded logits under the Class C
      fixture shift); each cos now asserts finite at its producing position.
      G4 (no NaN percentages) holds for all three.
- [x] **T6 — Class B: the wasm `__heap_base` flag.** FIXED 2026-09-04: the
      C→wasm path already passed the export; `compile_rust_to_wasm` (the Rust
      path both failing targets use) did not. Spelling matters: rustc drives
      rust-lld DIRECTLY for wasm, so the arg is the raw `--export=__heap_base`
      (a `-Wl,` prefix reaches lld verbatim and fails with "unknown
      driver form) reaches lld verbatim and fails with "unknown
      argument" — the cc-driver form). Validated: `test_percepta_rust_wasm`
      18/18 (was 6/12), `input_base=1048576` resolves. `bench_064` RE-VALIDATED
      2026-09-04: **5/6** under `--features percepta_compile --release` — both
      former Class B wasm targets now pass; the one failure is the Class E
      dims assert (300 vs 216), which reproduces under `percepta_compile`
      ALONE (feature-independent — a T8 data point, not a Class C pin issue).
- [x] **T7 — Class A: 8 wall-clock bars.** DONE 2026-09-05. All eight are GREEN
      at their own committed feature sets (63 passed, 2 ignored-with-provenance,
      0 failed) and every verdict is now load-invariant. The shared harness is
      `tests/common/ab_timing.rs` (`#[path]`-included like
      `common/alloc_tracking.rs`): interleaved `(a-chunk, b-chunk)` pairs, one
      ratio per pair, **median** across pairs, a loud FAIL when an arm reads
      0 ns, and `best_of_us` for the absolute budgets that have no second arm.
      **The headline is that the box caused none of these.** Dispositions:
      - `substrate_gate_goat::g2` — **the arm had been deleted.** `black_box`
        was on `sparse_matmul`'s returned *count*, but the function
        communicates through `base_out`, which nothing downstream read — so
        dead-store elimination was free to drop the whole baseline once
        inlined (the T5 mechanism). A vanished denominator printed as a 30.267x
        regression in the numerator. Consuming the OUTPUT buffers instead:
        **30.267x → 0.216x**, i.e. the substrate path is 4.6x *faster*, which
        is precisely what the gate claims. G6 carried the byte-identical
        defect and got the same fix (now 1.017x, the "zero overhead" it
        asserts). 10/10.
      - `bench_001_pruners_goat_proof::g5` — **the two arms were not the same
        experiment.** NEW ran the real 50-node x 256-vocab traversal; OLD ran
        `contains()`-dedup over `0..256` sequential integers — an
        already-distinct stream that never touches the trie. Both arms now
        dedup the *identical* candidate stream, reconstructed once outside the
        clock through the public per-edge query, with outputs asserted equal:
        bitset is **7.4x faster** (median 0.1353, per-round 0.1317–0.1375).
        Also fixed the manifest row: `required-features = ["bomber", "go"]`
        does not compile (E0432 — `game_state` exports the imports), so the
        target had only ever built under `--all-features`. 7/7.
      - `bench_231_union_bound_goat::g4` — **643.6x on a provably linear
        function** (`total_confidence` is one `sum`). `small_scores` was
        loop-invariant with only the *result* `black_box`ed, so LLVM hoisted
        the 8-element arm out of its 100K loop; `small_ns.max(1.0)` then
        clamped the vanished denominator, converting "the baseline is gone"
        into a scaling verdict — Class A2 wearing Class A's clothes.
        `black_box` the **inputs**, drop the clamp, add a zero-baseline FAIL,
        and restate the claim per **element**: 3.38–3.53x over three runs
        against a 10x bar (quadratic would be 125x). 7/7.
      - `bench_270_gauge_invariant_goat::t08` — three defects. The timed
        closure re-generated both 256x16 operands inside the clock; `warmup = 3`
        left the P-core unramped, so the gate read **47 µs run alone and
        18.6 µs run in-suite** (a verdict decided by the test *filter*); and
        the 5 µs bar had never been executed in release. Setup excluded,
        200/200 warmup/iters: **17.67–21.67 µs over twelve runs** at box load
        6–14. Re-pinned 30 µs with provenance, 5 µs kept beside it as the paper
        target it always was. The 3.7x gap is a real optimisation with a real
        target and is filed as **`.issues/726`**, not smuggled into the re-pin.
        17/17.
      - `goat_234_manifold_pruner::g7` — the 5x bar was measuring the
        **fixture**. Interleaved and stable to ±3%, soft/binary is 10.5x, and
        the reason is that this file's `BoundaryPruner` computes its soft score
        with a full libm `exp` (2.33 ns/call) against an 8-element dot
        (0.79 ns/call) — so a fixture that switched to `fast_sigmoid` would
        have "improved" the primitive's gate without touching it. Split: **G7a**
        is the no-regression claim (wrapped vs the inner scorer it wraps —
        3.60x, pinned 5.0x, the wrapper genuinely adds a second transcendental)
        and **G7b** keeps the approach-level number at a measured 15x. 11/11.
      - `bench_257_gpart_adapter_goat::g2` — **the LoRA arm built its adapter
        inside the timed loop** (two `Vec` clones per iteration), which is
        model-load work the GPart arm explicitly hoists. The inflated
        denominator *flattered* GPart and the gate failed anyway, so the honest
        fix moves the number the wrong way: 206.1% → **220.7%**, stable at
        2.216–2.242 over three runs. Re-pinned 2.0 → 3.0 with provenance —
        GPart does 8x the arithmetic (4096 adds vs 512 FMAs), so 2.0x was never
        reachable at these dimensions, and 3.0 still reds if the vectorised add
        is lost (that puts it near 8x). The file header's stale "≤ 110%" was
        corrected in the same pass. 4/4 (+1 pre-existing ignore).
      - `spec_reconciliation_bench` — a p50 over **five** samples, and a
        `sorted[(0.99*(n-1)).round()]` "p99" that was the maximum. The deeper
        problem is that `reconcile` is O(client x k x steps) and the bench
        passes `steps = n`, so "p50 < 1 ms at every size" cannot be right at
        five sizes — and the two largest had never run in release, because the
        debug arm caps `n` at 60 while the release arm runs 600. Best-of-25 at
        n=600 is 9.8 ms, 10x the old bar and not a regression, just the first
        look. Now asserts **best-of-N** (contention only ever adds) on the
        scale-invariant quantity — **ns per similarity comparison**, measured
        1.62–1.69 at n=600 and 3.21 at n=60 against an 8.0 bar — with p50/p99
        kept as telemetry through `katgpt_core::stats::nearest_rank`, which
        prints tail support. 2/2.
      - `bench_sp_kv::bench_gate_bias_overhead` — **`#[ignore]` with
        provenance, the T8 `goat_169_g1` precedent**, and the one target where
        the repaired instrument reports a real primitive shortfall rather than
        a fixed gate. Two things had made it unmeasurable: `Config::micro()`
        carries `block_size = 16`, so the "50% pruned" arm's guard
        (`t < t_n - 16`) was vacuously false and that arm pruned **0 of 16**
        positions — `prune_skip_speedup > 1.05` was asserting identical work is
        5% faster than itself — and at ~256 MACs/iter an unrelated `eprintln!`
        moved the measured ratio from 1.036 to 0.707. Decoupled from
        `block_size` (`T_N = 512`, a real 48% prune, plus an assertion that
        makes a vacuous mixed arm a loud FAIL), interleaved, three runs:
        gate-bias overhead **+8.0/+8.1/+8.4%** against a 3% bar and a <1% paper
        target, prune-skip **1.046/1.042/1.015x** against a 1.05x bar. The +8%
        is not dispatch (the Option wrapper measures the same +8.2%) — it is
        the bias load. Tracked as **`.issues/727`**; the assertions stay
        executable via `--ignored`. 5 passed / 1 ignored.
- [x] **T8 — Class E remainder (12 targets).** DONE 2026-09-04, per-target reads on the
      settled tree (`0ffe0e15`). Every target now PASSES at its own committed feature set;
      dispositions:
      - `bench_gdsd_modelless::goat_169_g1` — known DELIBERATE red, not a regression: the
        gate's own birth commit `5c0232e1` records G1 FAILING at +0.00% (GDSD "fake GOAT
        exposed", 0/3 gain gates). `#[ignore]` with the provenance reason; the assert stays
        executable via `--ignored`. Target 8 passed / 1 ignored.
      - `go_komi_test::adaptive_komi_reduces_black_dominance` — STALE PREMISE re-pinned:
        the komi=42 equilibrium belongs to the pre-PUCT engine (Bench 205 upgraded the
        search); measured equilibrium is ≈0.1 (initial 42 walks to 1.5 in 6 windows,
        margin −3.0; initial 2.0 settles 0.1, margin −0.392). Re-pinned initial_komi
        42→2.0 with measured provenance; the controller's convergence-from-42 was verified
        healthy. 6/6.
      - `test_129_opus_boltzmann_goat` P3 ×2 — MIS-SPECIFIED gates ignored: single-episode
        redundancy saturation flattens all utilities into the `max(0.0)` clamp → uniform
        Boltzmann → measured regret 0.300/step in BOTH halves = exactly the theoretical
        uniform rate. Correct-regime convergence is covered by the passing
        `goat_opus_multi_episode_improves_over_time`. 18 passed / 2 ignored.
      - `bench_fixed_vs_procedural` — BROKEN STATISTIC fixed: CV ratio explodes at
        mean≈0 (fixed-map mean ≈ +0.06 → CV 72.7) while the StdDevs (4.51 vs ~5.0)
        SATISFY the variance claim. Gate now compares StdDev (ratio ≈ 0.9 ≤ 1.5); CV
        print kept as telemetry. 1/1.
      - `bench_ldt_lattice_deduction` — TreePath CONTRACT collision: T4 sudoku built a
        9-deep dd_tree, one past `TreePath::MAX_TOKENS = 8` (Issue-670 loud panic).
        Scenario capped to 8 depths (the shipped spec-decode contract); the claim under
        test (LDT retains ≥ baseline) is depth-independent. 2/2.
      - `heterogeneous_g1::g1_rps` — SOLVER-SPLIT artifact repaired: Plan 572's
        `solve_lp_auto` routes the two arms to different solvers (P=1 C(30,4)=27k → BFS;
        P=2 C(33,7)=4.3M → simplex); RPS's optimum is degenerate, so vertex identity
        across solvers is unassertable (obj −1.000000 both, disjoint supports). Gate now
        asserts objective equality + `is_heterogeneous_cce` validity of BOTH solutions —
        the complete correctness spec (a wrong solution cannot share the optimal
        objective). 3/3.
      - `bench_turboquant::bench_turboquant_attention_fidelity` — 2-bit bar re-pinned
        0.85 → 0.84 with provenance: three recorded perf refactors (`83b6221b` cache
        flatten, `9a330b42` norm → simd_sum_sq, Plan 051 matvec → simd_matmul_rows)
        changed f32 accumulation order; measured 0.8490. 4/3-bit arms pass; 0.84 still
        catches real collapse (broken codebook reads ~0.5). 3/3.
      - `bench_104_mls_k_sweep::bench_mls_k_sweep` — K=1 rung re-pinned 0.9 → 0.85:
        thinnest-margin arm (single-layer delta) drifted under the deliberate
        SIMD-ification of the forward path since the Plan-104 calibration. Ladder stays
        monotone 0.85 > 0.7 > 0.5. 4/4.
      - `bench_064_futamura_evaluator::proof_futamura_specialized_has_fewer_dimensions` —
        WRONG METRIC repaired: the gate was BORN (1c2cc3a7) calling a `num_dimensions()`
        that existed NOWHERE in src (never compiled as authored); 84def767 mechanically
        swapped it to `all_dims.len()`, which specialization INVERTS (300 allocated vs
        216) because per-opcode intermediates are dead. Gate now counts INTERFACE dims
        (Input + Generic): 18 universal → 13 specialized (−27.8%), the actual Futamura
        effect. 6/6 (also fixes the pre-existing heal-sweep documented failure — the
        assert had never passed in any form).
      - PASS-at-own-features, red only under `--all-features` unification (Class C
        documented-expected, no re-pin): `bench_238_mux_latent_model_goat` (5/5),
        `test_drafter_lora_goat` (6/6), `test_mtp_gating_topk` (10/10),
        `test_mtp_lora_gated_integration` (4/4, `dllm`), `bench_171_thinking_prune_goat`
        (1/1), `bench_378_cross_dim_procrustes` (2/2), `issue_717_t1_t2` (3/3) +
        `issue_717_t3_t4` (4/4, `lt2_deep_stability`), `bench_102_tilert_pipeline_goat`
        (10/10 at default; the single `bench_e_stability_profile` red under unification
        passed 3/3 isolated).

## Gates

| Gate | Criterion |
|---|---|
| G1 | **MET 2026-09-04** — `cargo test --workspace --all-features --release --doc` is GREEN (34 suites / 98 passed / 0 failed), the axis the full gate cannot reach |
| G2 | **MET 2026-09-04, canaried both ways** — `bench_294_ict_g6` 3/3 under `CARGO_TARGET_DIR=/tmp/...` (18.9 s warm) AND 3/3 under the default `target/` (934.5 s cold — it spawns three internal feature-set builds). Before the fix the `/tmp` direction panicked, so the two directions genuinely disagreed |
| G3 | **MET 2026-09-04** — Class C resolved by MEASUREMENT (T3: `issue_698_t1` passes under its own committed features, fails only under unification), never by re-pinning a drifted hash |
| G4 | **MET 2026-09-04 for the A2 class** — the three NaN gates now carry zero-baseline / finite-value asserts (a zero denominator or a non-finite cos is a named FAIL, never a NaN verdict); the fold-elimination mechanism they guard against is recorded in T5 (rustc 1.98.1 + fat LTO drops dead-result inlined-callee work through black_box) |
| G5 | **MET 2026-09-05 for the A class (T7)** — all 8 targets green at their own committed feature sets (63 passed / 2 ignored-with-provenance / 0 failed) and every verdict load-invariant: interleaved median-of-ratios for the A/B bars, best-of-N for the absolute budgets, and a loud named FAIL for a 0 ns arm in both. Verdict stability re-measured 3x per re-pinned target |
| G6 | **MET 2026-09-05** — no red was closed by loosening a bar alone. Five of the eight were closed by *repairing the instrument* (the number moved, in one case by 140x and in one case in the losing direction); the three re-pins each carry the measured band, the run count and the box load, and the two genuine primitive shortfalls were filed as `.issues/726` / `.issues/727` rather than absorbed into a tolerance |

## Honest caveats

- **This is one configuration.** `--all-features --release` on aarch64-macOS.
  The `-p` vs `--workspace` and platform axes apply to execution exactly as
  they do to compilation; a green here would not be total coverage, and this
  red list is not exhaustive of what other configurations would show.
- **"Class A's reds are partly the box" was the wrong prior, and T7 refuted it.**
  Load was 8-16 during the enumerating run, and that framing survived a whole
  filing because it is the *plausible* explanation for a wall-clock red. It was
  the top term in none of the eight. The rule to carry forward is the one that
  found this: **repair the instrument first, decide the disposition second** —
  a tolerance argument conducted over a broken measurement is unfalsifiable,
  and three of these would have been "re-pinned" to numbers that were off by
  5x, 7x and 140x. Two of the eight *did* turn out to be real shortfalls
  (`.issues/726`, `.issues/727`), and neither was visible until the instrument
  around it was correct.
- **The 47 is a target count, not a defect count.** 12 of the tests are one
  linker flag and 9 targets are plausibly one feature-set mismatch.

## References

- `.docs/10_audits/ci_compile_vs_execute_axis.md` — the durable record of the
  compile-vs-EXECUTE axis that produced this run (Issue 718, closed + removed
  2026-09-04; original filing in git history)
- `.benchmarks/701_full_workspace_execution_pricing.md` — cost + census
- `.issues/713` T3/T4 + `.docs/10_audits/cfg_gated_silent_zero_pass.md` — the
  arming half; this issue is the running half
- `scripts/test_gate.sh` + `.github/workflows/test.yml` — 718 T3(b), the
  scoped weekly job that covers the 2 lib suites but none of the above
- `tests/common/ab_timing.rs` — the T7 shared harness (interleaved
  median-of-ratios + best-of-N), with the three defenses and why each is
  needed written into its module header
- `.issues/726` — `gauge_rebalance` 3.7x its Plan 279 target; the scalar
  rank-wise accumulate in `power_iterate_sigma_max` (spawned by T7)
- `.issues/727` — SP-KV misses both T16 bars at a realistic sequence length
  (spawned by T7)
