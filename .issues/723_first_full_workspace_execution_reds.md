# Issue 723 — the first full-workspace EXECUTION is red: 47 targets, six distinct classes

**Status:** OPEN — filed 2026-09-04 by Issue 718 T3(a) (the full-workspace
pricing run, the first time anything ever *executed* this workspace's test
surface). **Class F (doc-tests), Class D (hard-coded `target/release`), and
Class B (the wasm `__heap_base` flag, both targets) are FIXED.** Classes
A/A2/C/E remain and are the tracker's content. 718 is CLOSED and
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

### Class A — wall-clock / throughput bars, measured on a box at load 8-16 (8)

`bench_001_pruners_goat_proof` (G5 A/B, 31.9 ms vs 54.8 ms),
`bench_231_union_bound_goat` (G4.4 scaling 643.6x vs a 250x bar),
`bench_257_gpart_adapter_goat` (G2 206.1% vs ≤200%),
`bench_270_gauge_invariant_goat` (t08 30.9 µs > 5 µs),
`goat_234_manifold_pruner` (G7 throughput), `substrate_gate_goat`
(G2 ratio 30.267x), `bench_sp_kv` (gate-bias overhead),
`spec_reconciliation_bench` (p50/p99 bars).

These are the surface the 718 T3(b) scoping decision **deliberately left
workstation-owned** — a Linux runner cannot hold them. The action is not
"fix the code": it is to decide per target between the
`make-wallclock-gates-load-invariant` treatment (interleave a control, assert
the per-pair ratio) and an explicit `#[ignore]` with a reason string. Three
of this repo's last four commits (`a9576e20`, `ff6a4d46`, `172f5520`) are
already tolerance re-pins from this same run, so the class is live.

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
- [ ] **T7 — Class A: 8 wall-clock bars.** Per target, choose load-invariant
      treatment or `#[ignore]` with a reason. Owner call on which.
- [ ] **T8 — Class E remainder (12 targets).** Per-target read; several are
      likely stale quality bars rather than live defects. Data points gathered
      2026-09-04: `bench_064` dims assert (300 vs 216) reproduces under
      `percepta_compile` ALONE — feature-independent, not a Class C pin issue;
      `bench_104::bench_mls_k_sweep` fails at DEFAULT features too (K=1 cos
      0.8839 vs the 0.9 bar, deterministic, bit-identical to the failure the
      heal-sweep records documented pre-718) — a stale/machine-marginal
      calibration bar, not a regression; `bench_gdsd_modelless::goat_169_g1`
      (acceptance +0.00%) reproduces under `gdsd_distill` alone with the rest
      of the target green (8 passed).
      **Precondition:** run the per-target reads on a SETTLED tree — this
      worktree carried 13 sibling WIP files across katgpt-core / katgpt-dec /
      katgpt-kv / katgpt-transformer when these datapoints were gathered, and
      a quality-bar verdict measured against a moving lib is unattributable
      (the Issue-066/074 class: sibling drift flips scenario verdicts).

## Gates

| Gate | Criterion |
|---|---|
| G1 | **MET 2026-09-04** — `cargo test --workspace --all-features --release --doc` is GREEN (34 suites / 98 passed / 0 failed), the axis the full gate cannot reach |
| G2 | **MET 2026-09-04, canaried both ways** — `bench_294_ict_g6` 3/3 under `CARGO_TARGET_DIR=/tmp/...` (18.9 s warm) AND 3/3 under the default `target/` (934.5 s cold — it spawns three internal feature-set builds). Before the fix the `/tmp` direction panicked, so the two directions genuinely disagreed |
| G3 | **MET 2026-09-04** — Class C resolved by MEASUREMENT (T3: `issue_698_t1` passes under its own committed features, fails only under unification), never by re-pinning a drifted hash |
| G4 | **MET 2026-09-04 for the A2 class** — the three NaN gates now carry zero-baseline / finite-value asserts (a zero denominator or a non-finite cos is a named FAIL, never a NaN verdict); the fold-elimination mechanism they guard against is recorded in T5 (rustc 1.98.1 + fat LTO drops dead-result inlined-callee work through black_box) |

## Honest caveats

- **This is one configuration.** `--all-features --release` on aarch64-macOS.
  The `-p` vs `--workspace` and platform axes apply to execution exactly as
  they do to compilation; a green here would not be total coverage, and this
  red list is not exhaustive of what other configurations would show.
- **Class A's reds are partly the box.** Load was 8-16 during the enumerating
  run. That does not dismiss them — a gate whose verdict the box decides is a
  gate that cannot run in CI, which is the finding either way.
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
