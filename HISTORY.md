Compacted 2026-10-30: every `##` heading kept verbatim, one compact entry per
record (≤2 short lines); the full pre-compaction text is in git history.
Operational rules live in `AGENTS.md`; removed issue files: git history.

## AGENTS.md compaction (2026-10-30) — dated incident arcs moved out of the operational doc

Repo births/registrations: riir-rethink 10-03, seal-std 10-07, riir-refab 10-11, jup-tool 10-14, seal-render 10-21, seal-game-editor + zed 10-27 (codename row retired by Issue 969).
Per-instrument "measured on day X" narratives dropped from AGENTS.md (105KB→58KB); the rules stayed — full text at the parent of this commit.

## Issue 974 (filed + closed 2026-10-30) — walk-proximity context ranker lifted to katgpt-core (riir-neuron-db issue 640 T7 ledger): CLOSED — T1–T4 LANDED `d5c7c3da8` (filed `74090cbe8`); opt-in `walk_rank` (`crates/katgpt-core/src/walk_rank.rs`): `WalkGraphView` trait + stage-1 visit-count ranker + stage-2 budgeted context, BLAKE3 per-walk seeds, as-of guard before sampling. GOAT PASS [Bench 974](.benchmarks/974_issue974_walk_rank_goat.md): G2 ≈ 0.29 ms/call worst (36× under bar), G1 identity asserted every timed iteration, G4 alloc gate green. ndb `walk_rank` re-exports at the same paths. Not promoted — ndb is the only consumer. File removed per noise-reduction; full text: `git log --follow -- .issues/974_walk_proximity_primitive.md`.

## Issue 972 (2026-10-29 → closed 2026-10-30) — cross-tail interference instrument (Nestor forgetting law × sst_certified, Research 639): CLOSED — T1–T5 LANDED, `katgpt_core::sst_cross_tail` `e5ee29c8c` (behind the SAME `sst_project` gate, no new flag; re-export beside sst_certified): `cross_tail_interference[_into]` over two same-anchor `TailFactor`s, refusal classes mirroring `compose_tail_into` VERBATIM (anchor digest / shapes / tail bases → `ComposeMismatch`, never a silent re-factor) + non-finite/zero-payload guards; outputs per-side, never pooled — uncentered-linear-CKA scalars over the input/output Grams (Gram-normalized: the raw-norm form reaches 1.0 only at rank 1) + cos² principal-angle spectra on σ-truncated eigenbases (cutoff 1e-6 σ-relative, recorded in the report beside below-floor energy fractions); zero-alloc over caller-owned `CrossTailScratch`, f64 sequential folds + sorted canonical eigenbases + `karc` jacobi (the Issue-940 law). Gates in C-coordinates (exact f32-representable Givens family, never trig): spectra 1e-12, monotone in θ, EXACT 0.0 at 90°, equal-payload identity, the full-rank separation arm (the trivial-angles killer), five refusal arms; predictor-vs-measured with pinned units and claims SPLIT (spectrum = hard bound with the truncation term ε₁; scalar = RANKING only — the rank-1-vs-identity counterexample is why it carries no upper-bound guarantee); determinism double-runs bit-identical; rotation invariance 1e-12 exact / 1e-5 general (f32 wire floor, honest deviation recorded — the issue's 1e-10 assumed f64 payloads). T5 report-only API law: the instrument prices, never refuses beyond input guards — no gate behavior ships. **Consumer half LANDED same day riir-ai-side** (riir-ai Issue 1141 `698b61d07`+`968cb3a89`: `SstAdmissionReport::record_cross_tail` + sidecar digest-pair comparability + four count-pinned gates; pointer discharge `762776af6`). Gates: clippy −D clean (default / sst_project / --all-targets), 11/11 instrument + 15/15 sst_certified at posture, default lib 2167 passed. Claimed zed-m3-972-instrument, landed same session. File removed per noise-reduction; full text: `git log --follow -- .issues/972_cross_tail_principal_angle_interference.md`.

Outcome as headed; full text in git history.

## Issue 971 (2026-10-28 → closed 2026-10-29) — the on-disk numbering-duplicate wall (the riir-ai Bench-1011 double-hold was visible to NO automatic gate): CLOSED — detection gap landed same-cycle. Wall: katgpt-rs `a1f0751aa` (riir-train's duplicate classifier ported into the shared scanner `numbering_gate.classify_stems` + `numbering_drift_sweep` extended — COLLISION wall at 0 in every contract repo, rename/hand ratchets, min_bench floors; SUBSTRATE DEVIATION from the issue's one-copy proposal AGREED by the filer lane: two copies (riir-train private, katgpt-rs public) pinned to agree by 12 golden vectors encoding the gate's arms incl. the four founding collisions — a subprocess-over-the-sibling lane would silently degrade on public CI, and the vectors make cross-copy drift loud; cross-copy notes both docstrings, riir-train `bd029dc4`). Adjudications (first-allocation, citations moved in-commit): riir-ai 809→1016 / 872→1017 / 1008→1018 (`b5f9a8c2f`+`4427a6547`), riir-infer 884→926 (`2d591ea`); dual_allocation complement pointer `e4cebb910`. Routing pass (the filer lane, m3 housekeeping): refine max_hist 5→6 + riir-ai max_resets 9→10 ratchets `0f3abc835`; riir-ai `.issues/.highwater` 1134→1136 live-hazard repair `636a449ea` (the issue-1134 arc's stale-counter filing had 1135/1136 live-collideable). Close-out verification: the issue's last open item (the refine `.research/344` renumber) was ALREADY LANDED at `e7e8df51` (citation-weight 59:1, both instruments agree, citations moved, highwater 345) — the resolution block's "stays open for the renumber" was stale; pin stays 6 (measured post-renumber; the walk did not collapse this pair). File removed per noise-reduction; full text: `git log --follow -- .issues/971_workspace_on_disk_numbering_duplicate_wall.md`.

Outcome as headed; full text in git history.

## Issue 963 (2026-10-25 → arc landed 2026-10-28) — toolchain bump 1.98.1 → 1.99.0 (+ rust-src): the anchor pin moved (`8d54ddb96` + `rust-version` floor raised in the same commit per the pin's own discipline). Full gate GREEN at 1.99.0 on macOS with real evidence — the gate's own loud-zero defense first fired INCONCLUSIVE on the warm cache left by the `RUSTUP_TOOLCHAIN=stable` rot-gate posture (stable = 1.99.0 on this box), discharged by wiping `target/debug/.fingerprint` and re-running: Layers 2/2b/2c (wasm32 both simd128 arms, x86_64-apple-darwin both avx2 arms) + Layer 3 (347 units compiled, **0 warnings — no new 1.99 lint fires on this tree**, the `-D` list unchanged with zero unknown-lint errors) + Layers 6/6b release profiles + Layer 7 fmt (2832 files, 0 drift) all ✓; scoped core 2167/0. T4 revert path: a bad 1.99.0 surface is `git revert` of the bump commit (pin + floor move together, one commit — never investigate, roll back). **T3 (seal-remake + seal-game-editor downstream pins) DEFERRED, not forgotten:** both repos had a live sibling session mid-flight at claim time (seal-remake writes at 02:41, sge at 01:51–02:28, 2026-10-28) — flipping a toolchain under a running guard would run half its layers per rustc and corrupt the verdict; the pins follow on their next quiet touch, per the anchor semantics (downstream follows, never the reverse). riir-refine's `AHEAD_OF_PIN_1_99` registry is NOT this arc — it empties at RIIR-REFINE's own pin bump (batch 230's discipline), and its probes were already verified at an isolated 1.99 toolchain.

Outcome as headed; full text in git history.

## Issue 950 (2026-10-18 → closed 2026-10-24) — InfiLoop gated N/D summary modelless extraction POC (Research 626, arXiv:2610.11570): CLOSED — P1–P7 LANDED, GOAT ALL PASS (bench_946), P8 deferred-until-pull by its own gate. Opt-in `gated_summary` in `crates/katgpt-pruners/src/posterior/gated_summary.rs`: `GatedSummary<const D>` — two-counter normalized running summary (`N_t = β·N_{t−1} + e_t·z_t`, `D_t = β·D_{t−1} + e_t`, β ∈ (0,1) refused loud outside; sigmoid step view γ = σ(s − ln(β·D)), never softmax) with the `PrecisionVector` substrate consumed as the β=1 + fixed-w BIT-IDENTICAL oracle (never forked), the t=1 γ≡1 edge pinned as bits, ±8 score clip overflow-proof, convex-hull + never-decays-to-zero invariants tested, four-control GOAT battery (PV posture / decay-only / content-only / full) — the full posture is the only retention×adaptation Pareto arm (mid-burst retention 0.060 inside the pinned 4× tax, post-jump adaptation 0.0355 = 21–30× the controls; per-delta clean: gate owns retention, decay owns adaptation); G2 31/72/457 ns/write at D=8/64/512, G4 0 allocs. Scope findings on record in [Bench 946](.benchmarks/946_issue950_gated_summary_goat.md): the β-decay forget horizon `e_adv/(1−β)` under sustained rejection (~10 writes at β=0.55) and decay-only = no defense (`w/(1−β)` equilibrium admits everything). No promotion claim — consumers unwired; riir-train Plan 465 consumes the kernel. P8 (phase_belief two-signal commit fusion) stays DEFERRED-UNTIL-PULL: re-opens when a real phase_belief consumer posture exists (Issue 939's shape). File removed per noise-reduction; full text: `git log --follow -- .issues/950_infiloop_gated_summary_poc.md`.

Outcome as headed; full text in git history.

## Issue 956 (2026-10-22 → closed 2026-10-24) — binary-diffusion modelless extraction — conflict-triggered lattice reopening + Ψ-sampler knobs (arXiv:2610.28543, Research 480; training track = riir-train Issue 641): CLOSED — T0–T4 all LANDED, one defer with a structural re-open trigger. T1 (`bb95e404d`): `bench_ldt_hardness_stratified_reporting` — 480 deterministic instances, validation-fixed easy/mid/hard bins, `MassScreener` making θ LIVE — the pooled read misrepresents the hard stratum by ~20 pts (LDT hard 15.5 < mid 34.9 < easy 54.2 at budget 500); en-route: the Plan-088 T4 proof never ran `screening_threshold` binding (the bench is the first live θ read). T2 (`729bb73ff`): conflict-triggered targeted candidate reopening — `lattice_reopen.rs` + `ConflictClauseDB::implicated_positions`, elimination-time screener replay at chain context, singleton clause learning, `score^β`-ranked revival at relevance 1.0 (the screening penalty REMOVED so revived candidates compete on their own LLM mass), re-propagation through the same `TreeBuilder`; α-propagator monotonicity untouched; termination = max_rounds + empty-restoration + monotone ledger + no-cycle filter; G3 default-off node-exact; hard +5.6 pts at budget 500, chain read 0 → 2.8/9.6%; the tight-budget regression (−5.6 at 150) named the budget law. T2b (`79dbcd44f`): expected-reopen-count preservation — `eta`/`gamma`/headroom drive with pool-spacing γ normalization (the shipped absolute γ=20 measured ZERO reopens — the loud-zero gate caught it; a distribution-blind fixed slope is the recurring calibration lesson), the deterministic count-above-cut; tight budget 150 hard +25.4 pts (the regression inverted into the bench's largest gain), anti-thrash asserted (134 vs fixed-b 1920 reopens). T3(a)+(c) (`e3eae22ca`): the κ confidence-budgeted commit arm on the d2f decode core (`kappa_eta`/`kappa_gamma`, disarmed byte-identical, EB precedence); the capped-mean bisection LIFTED to the shared `katgpt_core::sigmoid_budget` leaf (305 speculative tests green); T3(b) λ DEFERRED — the decode loop has NO per-position noise level (the whole block shares the step's t); re-open = per-position noise levels land in the decode loop (Plan 108's shape). T4 (`5a97948c3`, probe-first, test-only): flip-move pricing — closed-form reverse rates over the fixture μ — flip-parsimony preserves hard-instance evidence at finite budgets (K=1 pooled 13.3% vs resample 0.0% everywhere) with TWO named production preconditions: set-conditioned scoring (fixed μ starves the flip chain's ADD side) + a set-size floor (T2b's η·headroom shape); the GOAT gate, if pursued, reads a set-conditioned flip lane at the lattice's real verification budget. G2 honest-cost: reopen wall fixed-b 6.45× / η 5.59× (five-run ambient re-pin 2026-10-23 `3752dad8a`, load 2.4–3.5/16 disclosed, absolutes ms-class) — the +10% gate reads DISCLOSURE, the replay + re-propagation genuinely cost it; machinery all opt-in behind `lattice_deduction`, G3 byte-identity at every knob's zero. File removed per noise-reduction; the landing state lives in the code citations (`sigmoid_budget`, `lattice_reopen`, `alpha`, `d2f`) + the four bench targets in `tests/bench_ldt_lattice_deduction.rs`.

Outcome as headed; full text in git history.

## Issue 958 (2026-10-23 → closed 2026-10-24) — `sst_certified`, the tail-subspace weight-delta substrate (arXiv:2610.22293 adoption, Research 631): CLOSED — LANDED OPT-IN, G1–G4 + Lean ALL PASS (Bench 953). Six lanes in one session (claim `zed-m3-sst958`, epoch 1792784038, code `93aa3dbad`, Lean `2ffc89e22`, close-out `0f1554968`): `crates/katgpt-core/src/sst_certified.rs` — `fit_anchor` (Gram on the smaller side via karc `jacobi_eigen`, f64 sequential folds, sorted eigenpairs paired with σ — the σ-order bug class caught by the roundtrip test before landing), `anatomy`/`anatomy_factored_into` (two-positive-terms `‖P_lΔ‖² + ‖P_⊥lΔP_r‖²`, never the cancelling three-term form; rank-r path via r×r Grams, never m×n), `factor_tail` (`ΔW = U_⊥CV_⊥ᵀ`; the wide-side tail is the σ-propagated σ-visible truncation ≤ min−k with dropped energy REPORTED — full wide-tail needs the wide Gram, infeasible, documented), `certify_decoded_tail` (relative-tolerance band [1e-5, 1] = the f32 resolution floor), `compose_tail` (same-anchor + bit-identical bases; leak bounds additive), and the rotation-invariant anchor digest (BLAKE3 over the projector's 64-probe action — exactly invariant under basis sign flips, refit-deterministic; the Lane-6 stale-cert gate `verify_against`). G1: the isotropic-rounding negative — dense-project-then-quantize FAILS the certifier at a noise budget the factored arm PASSES (non-vacuity proven); thin_svd oracle subspace agreement < 1e-4; planted-leak refusal MEASURED (α=1/3 → fraction 0.1±0.05). G2 honest-cost (the issue's µs-class round-1 target was wrong at rank, as it predicted): anatomy Θ(mnk) — 9 ms dense at 128×512, 8.6 s at (1024,4096); factored rank-8 575 µs → 81 ms at (1024,4096) — the issue's 0.7–5 GFLOP figure confirmed on the factored path; fit is Jacobi-sweep-dominated calibration-time (137 s at (1024,4096); the 4k² Gram is hours-class — a randomized-range fit lane is future work, priced not built). G4 zero-alloc 1000 cycles over all armed paths. Lean pair (`.proofs/KatgptProof/Sst/`): the diagonal no-leak scalar arm + level-set instance + the 2×2 `diag(1,0)` counterexample + the leak read (4 theorems, proof_gate 53→57, axiom budget clean; en-route: `Mathlib.LinearAlgebra.Span.Basic` alone does NOT pull the ℝ instances on this pin — the Sigmoid import carries them). Opt-in `sst_project` (implies `subspace_phase_gate` + `karc_forecaster`, the basis_split precedent); feature count 710→711 re-pinned at every claim site; AGENTS.md substrate section; consumer sides stay in riir-neuron-db Issue 629 + riir-ai Issue 1116 (unclaimed); producer riir-train Plan 470. Records: [Bench 953](.benchmarks/953_sst_certified_goat.md). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 954 (2026-10-20 → closed 2026-10-20) — `kron_memory`, the pair-addressed scalar-target delta-rule cell (Kronecker Delta Rule, Research 630, arXiv:2610.26834): CLOSED — LANDED OPT-IN, **G0 FALSIFIER FIRED, promotion refused** (Bench 951). The full cell shipped in one session: fixed-size `[[f32; DV]; DK]` Pod + pair registry, Newton-step writes under a contraction governor, exact-zero write elision (the frozen-state identity fixed point hash-pinned over 64 replayed writes — the Issue 940 `dot_f32_ordered` law end to end, `KRON_READ_ORDER v1`), the pre-write interference audit into caller scratch (zero alloc, loud truncation), audited surgical unlearning with closed-form repair, single-level tolerance-pinned undo (`EPS_UNDO = 1e-5`) + bit-exact touched-rows snap undo, analytic health gauges (λ̂_min over EMA marginal Grams, half-life, the norm-budget envelope, measured independence-violation — all lower-bound heuristics with the caveats on their face), LS-snap consolidation over the tracked moments (shipped `symmetric_eig`, ridge-regularized), and the `kron_chunk_wy` chunked form (Gram Hadamard product + unit-LT solve). 19/19 G1 property tests; G4 zero-alloc; G2 write 1017 ns / read 1010 ns / audit 2867 ns (all PASS). **The negatives, honestly:** G0 (the pre-registered falsifier) — at matched 16 KB state and 48 pairs (1.2% slot load) the plain one-hot slot table wins BOTH statistics (collateral 0.00016 vs 0.00482, paired LB95 −0.00494 < 0; recall 0.994 vs 0.780) → the flag stays opt-in forever; the double gate IS real against the vector contract (17× more surgical, LB95 +0.081 > 0) but loses to the table at low load; re-open = a G0′ at matched load ≥ 50%, own pre-registration, owner-gated. G2 chunk arm 0.56× (SLOWER than sequential — unpromoted; re-open = blocked-GEMM Grams via kron_tile/simd_matmul, priced not attempted). G5b: the analytic CI dropped (lost CRPS to the conformal-naive floor 2.156 vs 2.036 despite winning coverage 0.896 vs 0.203 and Winkler 2.62 vs 11.64 — all three required) → ships signal-only (`residual_signal()`). **What validated (G5a PROMOTED):** the self-consistent interference-envelope sizer `MSE(P) ≈ σ²_c/(1 − P·β²/(d_k·d_v))` held the [0.5, 2] band at x = 0.0625/0.25 — promoted for NEW-config sizing in the sub-saturation regime; the measured mechanism (no-decay erosion SATURATES at ≈ σ²_c — an interference floor, not a retention cliff) is the durable finding. En-route fairness fixes to the bench itself before recording: the floor observes the cal slice (a starved floor is not the canonical floor) and the first envelope draft's missing base-variance term (a derivation error, fixed pre-verdict). Records: [Bench 951](.benchmarks/951_issue954_kron_memory_goat.md) + [`.docs/03_memory/kron_memory.md`](.docs/03_memory/kron_memory.md) + feature catalog §154 + negative_results §47. Feature count 703 → 705 (README + examples/README re-pinned, docs gate 35/35). Claimed `katgpt-rs-kron-m3`, landed same session. File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 959 (2026-10-23 → closed 2026-10-24) — fixed-form softmax numerics primitives (arXiv:2610.31642 adoption, Research 632): CLOSED — T1–T6 all landed. T1–T4 (`14c356c03`, session epoch 1792774280): `crates/katgpt-core/src/softmax_fix.rs` — FloatFormat + PRECISION_CLIFF_NATS + precision_cliff(_bits) (ungated, the c_d = p·ln2 law is ours — the paper states numbers), margin_equilibrium (the Slingshot golden 18.42 > 16.64 HAZARD; CLI scripts/margin_equilibrium_audit.py — its console_safe defense landed with T5 after the console-encoding gate caught it), the −σ(−m) doctrine citation + the cliff-band fast_sigmoid property test (caught the issue's garbled expm1 reference — it is −tanh(m/2), not −σ(−m)), and opt-in logspace_diff(_f64) (anchored expm1, exact +0.0 at equality, overflow-safe). T5 (`9ccd7fa1a`, epoch 1792776198): opt-in fixed_log_softmax — fixed_log_softmax(_f64) + fixed_log_softmax_target(_f64) (the NLL-oracle shape, bit-identical to the full form's slot); the argmax-excluded-LSE softplus form log p_i = (z_i − z_a) − ln_1p(R), R = Σ_{j≠a} e^{z_j − z_a} — the target slot resolves from the cliff c_d to exponent underflow (~103 nats f32) instead of saturating at exactly 0.0; overflow-free by construction; branch-free hot loop + the argmax-slot overwrite (also fixes a lone +inf max); feature count 709→710 at all claim sites. G1: 207-rung ladder 0..103, both slots ≤ 2 ulp of the f64 oracle (the f32 libm composition budget — expf ≤ 1 ulp + log1p ≤ 1 ulp — disclosed; the issue's aspirational 1-ulp holds at most rungs); the incumbent's exact-0.0 saturation witnessed at the cliff in both f32 and f64. TEST-SUITE FINDING worth keeping: the G1 oracle's naive ln(1.0 + e^{−m}) close reproduced Softmax Collapse INSIDE the test — on this box at m = 28.5, ln(1.0 + 4.1938e-13) returns 4.19442e-13, a 1.5e-4 relative corruption of ε (the add round's half-ulp of 1.0 is 2.65e-4 of ε) — the oracle reads ln_1p and the incident is documented at the test. G2: benches/softmax_fix_log_softmax_goat (interleaved A/B ×9, median, black_box'd; M3 Max AC 94% mem free load 4.4–6.3, epoch 1792777601): ratios 0.562 (n=2, fixed FASTER) / 1.212 (n=64) / 1.315 (n=1024), gate ≤ 1.5 PASS ×3 — the branch-free loop took n=1024 from 1.472 to 1.315. G4: tests/softmax_fix_g4_alloc, 0 allocs/1000 mixed calls. G3: default lib 2167/0 unchanged. Determinism-law seam: first-index argmax + sequential ascending-j += fold pinned at the site (Issue 940 class); LOCAL instrument — sync crossings go through closure/bridge_certified. T6 (riir-infer `019bd45`, same epoch): opt-in golden_nll (forwards katgpt-core/fixed_log_softmax) — riir_infer_core::golden_nll (GoldenNllScratch::read_target: widen-once scratch → per-token NLL via fixed_log_softmax_target_f64 + the top-2 margin coordinate) + si_zero_shot_ppl --golden-nll (base-only reference mode, pooled Δ, the fp32/f64 confidence bands, the disagreement census; loud refusal without the feature). Module tests: golden ≈ incumbent f64-lse 1e-12 at ordinary margins; at a 100-nat margin the incumbent f64 form reads EXACTLY 0.0 while golden resolves ~e^-100 (the gate-relevant delta). Real-model smoke (league PQ2_0, 65 scored): pooled Δ +1.3e-15 nats/token, max |Δ| 6.7e-14 — f64-composition noise on a healthy model; the PQ2_0/GDN_STATE_HALF lossy gates diff THEIR variant checkpoints through this readout in their own lanes. Validation T5: 12 gated + 5 ungated module tests, lib 2167/0, clippy -D ×3 postures, all-features check, fmt clean, docs_gate 35/35. Validation T6: clippy -D ×3 postures, bin at both golden postures, release smoke green, refusal verified. Files removed per noise-reduction; the landing state lives in the feature catalog §156 + AGENTS.md numerics family section.

Outcome as headed; full text in git history.

## Issue 955 (2026-10-20 → closed 2026-10-20) — `CellComplex::grid_2d(0, h)` / `(w, 0)` wraps its loop bounds and pushes forever (release-mode OOM): CLOSED — one-branch guard landed (`0cacf71d6`). `grid_2d` computed `w-1`/`h-1` on `usize` for the counts and loop bounds; with a zero dimension the counts stayed zero (`usize::MAX * 0`) so the `reserve_exact`s pre-allocated nothing and nothing looked wrong, while the B₂ face loop's outer bound wrapped to `usize::MAX` and pushed 4 triplets per iteration forever → realloc → unbounded growth → OOM-kill in release (measured ~8 s on M3 Max, ~87% of profiler samples in `RawVec::finish_grow` under `grid_2d`); debug builds panicked on the overflow — the release shape (silent hang) was the dangerous one. Found from the consumer side wiring riir-ai Issue 1100 T0 (a caller reading a not-yet-allocated field naturally has dims `(0,0)`); this issue owned the substrate half. Fix: `assert!(w > 0 && h > 0)` — the contract, not a limitation (a 0-dim grid has no cells of any rank, no operators apply; a silent empty complex would resurface the same silence one layer later in the laplacian edge-list path), matching the `grid_3d` precedent in the same file. All shipped callers pass w,h ≥ 1 (benches 3×3..256×256; `grid_coarsen_2x2` asserts `> 0` upstream) — no caller changes. Gates: katgpt-dec lib 253/253 including the three `should_panic` zero-dim arms + the `grid_2d(1,1)` keeps-working pin; clippy `-D warnings` clean; fmt clean. The first full run's `bench_hodge_decompose_256x256` latency-bar red passed alone at 110.8 s vs its ~120 s bar (load-class, PASSED-ALONE — the change is one branch at construction). No GOAT implications (no perf surface; one branch on a constructor). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 957 (2026-10-23 → closed 2026-10-23) — the int8-activation ternary matvecs panicked `RefCell already borrowed` when two forwards shared the rayon pool: CLOSED — shared `simd::with_thread_scratch` landed (`4ede8e7b4`). `simd_ternary_q2_matvec_q8_parallel[_n]` and `simd_ternary_group_matvec_q8_parallel` ran their row-parallel rayon region INSIDE a `thread_local! RefCell::with_borrow_mut` over the activation scratch; a worker waiting in that region work-steals, and a stolen job that is another forward calling the same matvec re-entered the borrowed cell on the same thread. Found from riir-infer Issue 1111 T3: one more concurrent-forward test in `bonsai2_rotation_load` turned that binary red ~1 run in 100, a different victim each time, never alone — the shared-state concurrency class that passes alone BY CONSTRUCTION (backtrace `with_q2_activations` → `on_rayon_worker`). Fix: move the scratch out for the call and back after (no borrow spans `f`; a re-entrant call allocates its own — only on actual re-entry; steady state still zero-alloc, a `Vec` move allocates nothing). Family sweep over every tracked `*.rs` in katgpt-rs/riir-infer/riir-train/riir-ai with `thread_local!` + a borrow + a parallel construct: these two sites; riir-infer-gpu `weight_buffer_cache` STAGING and `ATTN_TAP` hold no borrow across one. Regression arms `nested_parallel_calls_are_reentrant_and_bit_identical` in both modules (64 outer rayon tasks × the parallel matvec, 8 rounds, bit-identical to serial) — RED before (both panic), GREEN after; katgpt-types lib 204 / 212 / 196 pass (`ternary_q2_q8` / `+decode_spin_pool` / `ternary_group_q8_act`), clippy clean both sets; downstream riir-infer `bonsai2_rotation_load` ≈4/435 red before → 0/600 after. No GOAT surface moves (two `mem::replace`s of 2–4 empty-or-moved `Vec`s per call vs a µs-scale matvec — not re-benched, the box was under sweep load). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 949 (2026-10-19 → closed 2026-10-20) — `path_kernel_sim` POC (sensitivity-fingerprint response-space kernel): CLOSED — T1–T9 LANDED, T9 verdict **KEEP-OFF**. Feature `path_kernel_sim` opt-in in katgpt-core: `SensitivityFingerprint` (p ≤ 64 knob-gradient at w0, scratch-buffer API) + closed-form σ'-arm / pinned-h central-difference arm + normalized kernel + Spearman ρ + Gram PR/effective-rank (substrate reuse: `linalg::symmetric_eig` + `subspace_phase_gate::participation_ratio` — no new solver). Near-zero-norm fingerprints ABSTAIN (sentinel, never NaN). G1 7/7 lib tests (T5a the linear-in-knobs collapse identity PROVEN — kernel ≡ input-feature cosine to 1e-5; T1 stationary-point + NaN abstain; T4 rank-1 Gram end-to-end) · G2 79–132 ns/call + 49–52 ns/pair at p=64 (bars 5 µs / 1 µs, black_box args+receiver) · G3 default suite untouched (2157 passed, purely additive feature) · G4 zero-alloc hot path (counting allocator, 0 allocs/100 pairs at p=64). **T9: the planted-class battery (independent labels, pure-cross-term exponent, release) read a THREE-WAY TIE — fingerprint 1.000 · input-cosine 1.000 · output-vector 1.000 — and T5d's own law fires: the tangent kernel adds nothing over gradient-free response probing at matched information → promoted nowhere, ships opt-in as a diagnostic instrument.** Structural finding: Ŝ(x) is a deterministic transform of x, so a full-rank probe battery (m ≥ p) is at-least-as-informative; the residual edge regimes (probe coverage m ≪ p, probe degradation) are consumption-side — re-open bar = a consumer whose probe budget is genuinely ≪ p. Records: [Bench 947](.benchmarks/947_issue949_path_kernel_sim_goat.md) + Research 625 §7 + feature catalog §153. File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 948 (2026-10-18 → closed 2026-10-19) — Terminal-RSI modelless extractions (FDR grader + SPR ordering + T3 reshape): CLOSED — ALL THREE LANDED/EXECUTED. **T1+T2** (`1f0c9300b`, 2026-10-18): `fdr_credit` + `spr_order` opt-in features in `katgpt-pruners` (family-calibrated partial-credit `delta_absorb` hinge + success-offset lexicographic ordering; 13 tests, flag-OFF compiles to nothing; feature catalog §151). **T1 GOAT + T2 paired read ALL GATES PASS** ([Bench 945](.benchmarks/945_issue948_fdr_spr_goat.md), 2026-10-19: G1a FDR ordering dominance Spearman 0.632 vs raw-P 0.364 (15/16 seeds), G2 bootstrap stability (divisor CV 0.043 mean, ordering τ 0.878), G3 top-3 never-worse 16/16, G4 zero-alloc; T2 = honest no-op, 0 flips — the tie-group guard ships as the drift guard; honest scope: synthetic adjudicated harness, estimator saturates at small K). **T3 EXECUTED same day — honest-negative for promotion** ([riir-ai Bench 990](../riir-ai/.benchmarks/990_issue948_t3_bandit_reshape_goat.md), `bench_990_bandit_reshape_vs_decay_goat`, riir-engine, feature `cgsp_runtime`): the SPR `P/(P+D')` reshape read as a bandit priority collapses to the sample-average estimator; under the SHIPPED CDF-sampling consumer law it carries +31.1% post-shift regret (cap-regime shift) / +8.0% (demotable shift) vs the shipped `DEFAULT_BANDIT_DECAY = 0.7` — the reshape ships nowhere (G3 structural: zero production delta). Recorded nuances: the reshape is variance-free and wins the cap-regime p90 tail (623.6 vs 703.9) + adoption 16/16 vs 6/16 — the shipped posture's cap-regime lock-out tail (10/16 seeds) is the drift-guard datum; its loss is partly an exploration tax under CDF sampling (entropy 0.97 vs 0.24). Deterministic (frozen seeds, re-run byte-identical). Features stay opt-in (no wired consumer). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 917 (2026-10-03 → closed 2026-10-19) — EB-Sampler-class entropy-bounded commitment (D2F/DDTree/Drafter commit paths): CLOSED — the full arc landed opt-in and the lane A/B EXECUTED. T1 primitive `katgpt-core/src/entropy_bounded_commit.rs` (2026-10-03, catalog §141, [Bench 917](.benchmarks/917_entropy_bounded_commit_goat.md): oracle PASS — no-stall everywhere, EB needs 5–19% fewer NFE than the floored τ=0.9 incumbent at equal validity); T3 DDTree + DFlash wiring (10-07); T2 D2F decode wiring + cousins flashar/set-diffusion (10-08, `D2fDecodeConfig.eb_gamma ≥ 0` arms / −1.0 disarmed default, two-level flag gate, flag-off byte-identical); the riir-side `gemma2_d2f` GPU model-forward lane wired 10-16 (riir-infer `44ca3cb` + riir-gpu `34e462835`, `Gemma2D2fConfig.eb_gamma` disarm-default). **Lane A/B (riir-train Bench 645 = Plan 437 T0.2, 2026-10-18, M3 Metal — the 4090 was busy, a same-device relative read is valid): EB converges 16/16 at 8.00 NFE vs every τ ∈ {0.5, 0.7, 0.9} stuck at 16.00 — the near-uniform ~0.02 committed confidence of the barely-trained head never crosses any fixed τ — at matched confidence (0.0202 vs 0.0199) and the same 0/128 quality floor; a POLICY-BEHAVIOR read on the weak Phase-0 0.4B checkpoint, priced as such by the 2026-10-08 owner-gate verdict (3-round Claude → AGREE: Phase 0 first, relative quality only, never an absolute claim).** The §7 parity gate's first real-checkpoint run caught the Metal `tanh` NaN negative-tail defect (cubecl's safe_tanh guards only x > 43; the trained adapter pushed masked-token scores to raw/softcap ≈ −177) — fixed by the ±42 clamp in all four softcap kernels, `tanh(±42) = ±1.0` exact in f32 (riir-infer `2f65031` + regression test; clean parity 3.3e-3 rel, cos 1.000000). En-route the Phase-0 trainer revival itself: riir-train `c1f48627` (621 filed, `WIRING-ARCH: gemma2-04b` named) → 621 CLOSED 10-18 (smoke `18dd21f6`, the 4090 10k-step window, T0.3 AR control arm `65a5443c`, T0.2 parity + A/B). Stays OPT-IN: promotion needs a checkpoint whose committed confidence actually separates (plan-410 T3 / Plan 437 Phase 4, owner-gated) re-running the A/B with the quality axis live; the γ dial stays uncalibrated for production until that lane. File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 944 (2026-10-16 → closed 2026-10-17) — one ULP metric + ApproxContract (the ink-lang Research 475 consolidation): CLOSED — LANDED (`7da012f55`). `katgpt_types::approx`, deliberately UNGATED (the green-zero law: a feature here would compile every migrated gate to a silent `ok. 0 passed`; test/bench-only, zero hot-path) — the issue pre-authorized the ungated alternative as "the simpler law". `ulp_distance(f32, f32) -> Option<u32>` settles the semantics the three copies punted on: signed zeros share ONE position (the raw map's ~2³¹ artifact retired), non-finite FAILS CLOSED (`None`, never a bit-distance over NaN payloads), finite same-sign domains bit-identical to the raw two's-complement map (deterministic-stride test pin, no RNG). `ApproxContract` is the declarative half (measure-across-sweep + assert-within), fail-closed on all three silent-pass shapes: non-finite poisons, an EMPTY contract refuses to assert (an unrun sweep is not a pass — the repo's loud-zero law at contract granularity), a violation names the worst pair. Migrated all three: bench_844 G1a (two named contracts; verdict identical — exact max 2 ULP), `simd/activations.rs` exact_sigmoid tests, bench_871 (`ulp_delta` now a one-line `expect` wrapper; its "same-sign finite" precondition subsumed by the shared law; gates ≤1/4/5/18/55 unchanged; root `Cargo.toml` gains the `katgpt-types` dev-dep). Gates: 169 katgpt-types lib + 2 doctests, 2157 katgpt-core lib, docs gate 35/35, clippy clean on every touched target, both benches re-run green. Declined: the optional Lean face (the analytic bound has no decision riding on it; the Rust pins are the contract). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 940 (2026-10-16 → closed 2026-10-16) — float reduction-order + tie-band determinism audit (handwired distill, Research 621): CLOSED — AUDIT + ALL 11 FIX UNITS LANDED, one working day, 13 commits. The law (AGENTS.md §"Determinism law"): an f32 reduction or pick whose result crosses the sync/quorum boundary (committed, exported, hashed, replay-compared) pins its reduction order and tie rule — the sequential fold (`dot_f32_ordered`, plain `+=`/`.sum()`, never a platform-lane SIMD kernel) and one fixed strict tie rule (first-index or documented last-index — never a tolerance band, never HashMap iteration order); anything else reclassified local-only with the decision at the site. T1 census: 24 committed-state reduction sites across 9 crates (headline: the certified `dot_f32_ordered` pattern existed in exactly ONE lane; every other committed lane used platform-lane kernels). Part C fix queue 11/11: ① MB-1/2 micro-belief bridge+attractor → ordered (`b66056fcc`); ② PE-1 personality drift → ordered, the synced BLAKE3 commitment's cross-node stability pinned (`bb0ffb339`); ③ MB-4 bom_arena RNG-seed dot → ordered (`bef1f0f2f`); ④ HL-1/2/3 + the missed fourth site — HLA committed-moments dots → ordered, G2 discharged FIRST (Bench 939, +36.7 ns = +0.8%/step) (`052f7e732`); ⑤ SE-2/3/4/7 sense SIMD names DELEGATE to the pinned scalar cores + two census-missed matrix paths pinned, the `simd_fused_sub_scale_inplace` per-cell hole recorded (`0419d5e52`); ⑥ SP-2/3 + CL-1/2/3 ordered folds + the census-missed procrustes residual arm + the KERNEL repair: `simd_fused_scale_acc{,_f16}` tails were two-op vs FMA bodies (unit-⑤ class, inverted; procrustes' KG-committed M flows through it) — four tails fixed to single-rounding `mul_add` at the root (`5b4f3fdf7`); ⑦ DE-2 canonical mass order: `boundary_flux_mass_indexed` CANONICAL, scan paths DIAGNOSTIC (1e-4 envelope, zero production consumers — pure documentation) (`fa45702f5`); ⑧ NaN-hardened calibration sorts → `total_cmp` (`4fde9c260`); ⑨ UNST ×4: all collection-order ties → lowest-key determinism (ruliology promoted_winner lower arm index — was a run-to-run random winner; best_elo/sorted_by_elo/sampler/trial_log lower id/arm) (`4e3ad9a84`); ⑩ TOL ×4: brevity band KEPT+documented (the module doc's false "never changes which answer wins" corrected; pins both sides), percepta hard-attention 1e-9 band → BIT-EQUALITY (matches hull + the law; ~4M-ulp leniency retired), tournament EPSILON band → exact equality (rates differ ≥ 1/total — the band only ever fired on ties), α-grid fallthrough documented+executable-pinned, interpolation declined (`4ac43fba7`); ⑪ 41 LAST-index pin comments across the 22 audit-named files + the `usize::MAX` sentinel guard in `complexity_prior::top_k` (all-`-inf` round → lowest unchosen index, never the sentinel) (`23a04077c`). ~25 load-bearing pins + 3 module-doc false-claim corrections + the per-platform craft-search technique (squares can't cancel — divergence found at the 2³⁰ ulp boundary, s = 32768). ⚠ One-time committed-bits changes recorded per unit (hla_moments h/n, attractor beliefs, bridge scalars, personality w + snapshot BLAKE3s, sense activations, procrustes/rewire/CLR/fused-scale-acc-tail lanes). Gated-posture discipline re-proven twice (spectral procrustes/rewire + induced_cwm_tournament — the default lib runs compile both to nothing). Known-remaining (recorded, not fixed): `simd_fused_sub_scale_inplace` scalar-vs-SIMD arm split (tolerance-domain callers only); the x86_64 execution matrix re-run over the new pins is the next quarterly hygiene pass. Durable record: [`.docs/10_audits/float_reduction_order_tie_band_determinism_audit.md`](.docs/10_audits/float_reduction_order_tie_band_determinism_audit.md) (§"The fix commits" carries every hash). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 943 (2026-10-15 → closed 2026-10-16) — the issue-level claim lease: CLOSED — ALL FOUR DELIVERABLES LANDED. The convention: a session starting work on an `.issues/NNN_*.md`/`.plans/NNN_*.md` item appends `**Claimed:** <session>, <epoch>, expires <ISO-UTC>` under the Status line and pushes BEFORE implementing (48 h expiry; the mining-queue claim-marker shape extended to the surfaces concurrent sessions actually allocate against; re-claim after expiry is a normal re-read). The instrument `scripts/claim_lease_check.py` (`35ef26420`): cross-repo callable (`git fetch origin && py ../katgpt-rs/scripts/claim_lease_check.py --item <NNN> [--epoch N]`), never fetches (observers, not ref writers) — FETCH_HEAD age DISCLOSED every run so a stale fetch cannot read as a clean scan; scans `.issues/`+`.plans/`, prints every live/expired lease; target live-leased by a foreign epoch = a loud WARN, NEVER a block (the push stays the arbiter — the private-lane yield proved post-hoc arbitration correct); exit 0 clean/own-epoch/warned, exit 1 only when the instrument cannot read (malformed claim line, ambiguous multiple leases, unreadable dirs, self-test failure). Grammar: exact-shape, case-sensitive, colon required — the first cross-repo run caught the wrapped-prose false-positive class (`claimed. Reopen condition: …`, riir-refine .plans/019) and the defense is pinned as a self-test arm; template placeholders (`<...>`) ignored so a spec quoting the convention cannot red its own repo. `--prove-fires` replays the Issue-825 second-session posture over temp trees (no epoch → WARN; own epoch → clean; expired → clean re-claim posture; malformed → loud; FETCH_HEAD age resolves on a real fetched clone and only there). Wire: AGENTS.md beside the substrate-first gate (`35ef26420`) + the riir-refine loop skill (`101a6157`). Claim-first dogfooded on this very issue (`dd6f4e4d5`). Deferred v2 (semantic term-overlap search) stays deferred — unblocks ONLY on a genuinely semantic fixture (different numbers AND different vocabulary); building it speculatively is the §1.55 "validates-our-design" trap. File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 938 (2026-10-13 → closed 2026-10-13) — lattice_memory 3D axis remedy + measured-erosion capacity gate (the μLattice distill): CLOSED — T1 LANDED (gates PASS), T2 LANDED + RECORDED NEGATIVE at G2. T2.1/T2.2 (`f5e088925`): the lattice is const-generic over the axis count (`LshAxes<A>`/`LatticeConfigG<A>`/`LatticeMemoryG<A>`; `LatticeMemory`/`LatticeConfig` stay the A=2 aliases — bench 919 re-run DIGIT-IDENTICAL G1 0.9830@10⁶ + identical verdicts + N_max 63,336, the G3 no-regression evidence); `lattice_memory_3d` (opt-in, implies `lattice_memory`); the G6 pricing pinned BEFORE the build (76³ = 438,976 cells × 4096 B = 1.674 GiB at d_v=32, under the 2 GiB bar; d_v=64 the recorded boundary row); `capacity_per_cell_eroded` (defaults bit-for-bit `0.35·d_k`, asserted d_k 1..128) + `sized_eroded` (same-cell E[cos²] from the fit corpus, fit-time only) + `read_direct` (the exact positional-copy lane, T4). T2.3–T2.5 (Bench 938, release, PROVENANCE AC/load 2.49/high): at the cell 2D cannot size (10⁶/0.05σ, λ=2.28) — G1 recall 0.9345 PASS, G5 graded recovery 0.184 vs Engram 0.004 PASS (curve 0.934→0.917→0.766→0.186 through the support), G6 857 MB PASS, **G2 FAIL: 7379 ns vs PKM 5171 ns (0.70×)** — cause measured by the cell-matched tap arm (27 vs 9 taps = 2.90× kernel MACs, the predicted ≈3×); G4 0 allocs both arities. The negative is Bench 919 rider #4 — the envelope's SECOND measured remedy (widen d_k breaks key=embedding; widen AXES clears recall+memory, not latency); the 2-tap kernel (8 vs 27 taps) is the named re-open lever (T3.2, deferred). T1.3 (G1, `lattice_memory_gates::erosion_survival_brackets_the_predicted_curve`): random keys bracket the aggregate law (promise 0.723 ≥ 0.5 at λ=11); clustered keys (ê=0.184) SATURATE above it (0.30 vs predicted 0.131 at λ=11) and track the exact key-recursion algebra k_eff ← k_eff − (k_effᵀkw)kw ± 0.08 — the eroded surface is CONSERVATIVE for parallel keys (the safe direction); mLattice's 0.083 never adopted (hypothesis input in one unit-test arithmetic pin). Records: [`.benchmarks/938_lattice_3d_axis_goat.md`](.benchmarks/938_lattice_3d_axis_goat.md) + Bench 919 rider. File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 936 (2026-10-13 → closed 2026-10-13) — decode-scoped spin-barrier pool for the row-parallel matvecs: CLOSED — ALL TASKS DONE, PROMOTED BOTH LANES. T1–T3 (bb3b24f2e, 65574c5d9, 9c4f7f9dd): thread-local SpinPool (idle-spin ≤ 100 µs then condvar park), opt-in `decode_spin_pool`, bit-identical; promoted downstream into riir-infer `deltanet_ternary_inference` (27a4da7): Bonsai-2-27B CPU decode ~100 → ~72 ms/token on the M3. T3b: the 11-worker macOS default measured in-model (11 < 10 < 9 < 7 ms/tok, 4 interleaved rounds, 0 mismatches; katgpt-rs-21, issue 937 T4 closed c47a1dffd). T5 Metal hooked lane (riir-infer `decode_spin_pool_gpu_goat` bdc0dd0): G1 0/24 bit-identity, G2 no-harm median 0.994 (pool 218.3 vs rayon 220.0 ms/tok) — the idle window parks workers through the ~2 ms Metal dispatches exactly as designed; the hooked lane stays a fallback posture (CPU 72 ms/tok). T4 off-macOS PROMOTION (cbaeac926): the two prior sessions' 0.97–3.93× losses predated the idle window; the worktree-isolated re-measure on the 4090 i7-13700K (the box's main katgpt-core manifest was sibling-WIP-dirty — dual worktrees + repointed path deps, never touched the WIP) reads a WIN at every grid point WITH the idle window — 0.862 @ 23 / 0.851 @ 15 / 0.866 @ 11 / 0.902 @ 7 workers, G1 bit-identical ×4, flat plateau 23/15/11 → default-on off-macOS at ≥ 8 logical CPUs (n−1 workers; the boundary IS the measurement floor — smaller hosts unmeasured, keep rayon via the 0-worker refusal), confirmed env-free on the box (default 23 engaged, 0.870, G1 clean); `machine_default` test re-pins the contract; KATGPT_SPIN_POOL_WORKERS overrides everywhere. Files removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 933 (2026-10-11 → closed 2026-10-13) — attn_res decode online-merge + the fixed-query modelless probe: CLOSED — T1 DECLINED on analysis (single-token decode has no reuseable online state — the (sublayer, summary) pair is used exactly once), T2/T3/T5 DONE (`12e031cff`, Bench 937: push-time Σv² cache + prepared score weight, bit-identical, live path 9–28% / prepared 21–31% faster at N=2/8/32) + **T4 GREEN-candidate** (the E5 probe, Research 619 MSAR): substrate `apply_attn_res_depth_profile` (`ca35f9ca9` — score[i] = −λ·age[i], a deterministic exponential-depth profile replacing the learned `score_weight`; gated by the EXISTING `transformer_attn_res`, no feature-count change; λ=∞ the disarmed byte-identity posture, pinned bitwise by test) + the riir-infer probe fork (`issue933_attn_res_probe`, `a9f5b0c`: the gemma-2 f16 forward re-forked at the three Kimi-semantics wiring points) + the frozen-read ppl A/B (riir-infer Bench 882, chat_probe 4096 tok paired): **λ=∞ bitwise-identical to stock (4096/4096 positions ΔNLL≡0); λ=4 Δppl −0.65% (BETTER than base); λ=3 −0.01%; λ=2 +21.6% (the cliff)** — the pre-registered |Δppl|≤1% bar met twice ⇒ the modelless fixed-query posture is VIABLE (opt-in, default-off); the improvement direction UNCONFIRMED (single corpus/offset; confirm reads recorded in the bench — second slice, λ∈[3.5,5] sweep, benchmark read; promotion owner-gated). Consumer honesty stands: the only `apply_attn_res` consumer is `src/kimi_k3/` (test-arch-only) — substrate hygiene throughout. File removed per noise-reduction.

T3 fallout closed 2026-10-15: residuals() reads fixed in 16 sites; here `2de11d40a`, riir-ai `7e7fb2e37`, riir-train `5db202b2`.
Found latent stale_residual sum_sq cache bug that silently no-oped attn-res mix.

## Issue 935 (2026-10-13 → closed 2026-10-13) — the Q2_0_g128 ternary matvec is the whole Bonsai-2-27B CPU decode (compute-bound ~25×): CLOSED — ALL TASKS DONE. T1 chunk-cap 32 (0.67–0.72× wall, work-stealing over 12P+4E) + T1b real-model (`0c5ed0fe7`: tg 1.96/2.03→2.46/2.47 tok/s, pp +32%, 65/65 greedy identical) + T2 `ternary_group_q8_act` int8-activation SDOT kernel (~7 ops/16 weights, NEON bit-identical to the scalar integer reference) + T3 real-model GOAT (top-1 256/256 + 510/512, confident 165/165 + 348/348, NLL Δ −0.003/−0.019, wall 0.319/0.326×; PROMOTED in riir-infer — `deltanet_ternary_inference` implies `ternary_q8_act`; e2e tg 7.86 tok/s, Bench 877) + T4 AVX2 port `2522dc2a9` (`avx2_q8_row_range`, runtime-probed `is_avx2_fma_available`, psignb+pmaddubsw+pmaddwd, integer inner sum ⇒ bit-identical to the same scalar reference ⇒ the aarch64 real-model GOAT transfers to x86_64 exactly; Rosetta-verified 21/21 + `saturated_extremes` + mutation canary, clippy clean both triples) + T4b native 4090 wall (this close; synced via the `gitsync` schtask per the task's own instruction, after the Plan-620 session released the box): q8 = **0.266/0.324/0.255/0.336×** the shipped f32 kernel on the four Bonsai-2 decode shapes (462.6/580.3/440.4/232.8 µs vs 1745.0/1767.2/1708.3/691.3 µs; i7-13700K, 24 rayon threads, idle box — 0 compute procs, GPU 1%) — beats the Rosetta-translated 0.467–0.482× indication AND every f32 chunk arm (best 0.69–0.75×; q8 ≈ 0.38× of chunk-16 on the big shapes); the `twt_perf_tg_pp` e2e leg artifact-blocked on this box (`--collapsed` GGUFs + `audition_calib.txt` are M3-side, parser requires both — indicative only, no promotion rides on it). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 934 (2026-10-13 → closed 2026-10-13) — divisive `evolve_belief` saturates ≥ 6/8 dims for per-tick never-reset hosts: CLOSED — verdict (c) RECORDED-INTENDED (bounded-episode kernel). Consumer census: every production host is bounded (riir-engine `sense/brain` `reconstruct` builds a fresh state per call; riir-games `multi_hypothesis` reloads per K-step branch) or already bypasses (riir-ai mmorpg `sync_relation_lane` feeds its channels directly — it had measured the same saturation independently); the three per-tick entry points (`sense_tick`, `observe_reconstruction`, `maybe_evolve_belief`) have no game-loop caller. Landed: the full caveat on `ReconstructionState::evolve_belief` (this repo, closing commit) + one-line pointers at the three entry points (riir-ai `d16c911a4`). T2 (a substrate evidence-retention knob) NOT built: zero consumers chose windowing. The probe (`56ad37c4c`) pins the premise. File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 932 (2026-10-12 → closed 2026-10-13) — belief-views multi-scale temporal retrieval (Research 619 MSAR time-depth transpose): CLOSED NEGATIVE at T3 — "single-stream belief stays". Probe `56ad37c4c` (real `route → accumulate → evolve_belief`, 16 seeds × 2000 ticks): the divisive kernel has NO self-dynamics (δ reads evidence only; interior map a pure translation, drift 1.6e-7) and cumulative evidence pins ≥ 6/8 dims at −1 for ANY stream (96.9% channel-ticks clamped, 96/128 channels constant) → views over a constant are dead (failure mode 2) and the clamp destroys only a zero-information signal; the additive kernel never clamps and is exactly invertible (b_{t-1} = (b_t − (1−α)d_t)/α) → nothing destroyed for retrieval to recover, and its multi-view analogue is a multi-α EMA bank (`temporal_deriv` already ships fast/slow). T2/T4–T7 not run (gated on T1+T3). The saturation itself was a new substrate finding → Issue 934 (consumer-side, riir-ai). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 930 (2026-10-10 → closed 2026-10-12) — CMM sync-relation-surprise primitive (SyncBank, arXiv:2610.07907): CLOSED — PROMOTED TO DEFAULT. All measurable tasks landed: T1b accessor, T3 property tests, T4 G1a PASS, T5 G1b real-fixture PASS (δ_surprise 23/23 warm events at EVERY matched FP budget vs value-surprise 0/23; Wilson 95% intervals disjoint; McNemar 23-0-0; AUC 0.72 vs 0.55 — the pinning test re-measures per run), T6 G2 quiet-box (D=8 observe 54.8 ns, readouts 59.8 ns/pair, D=64 observe 2354.1 ns; load 2.15/16, AC), G4 zero-alloc, unarmed-per-instance path byte-identical (γ=0). Promotion = owner GO (do-as-suggest): katgpt-rs `605bc3d2b` (root+core default arrays, comments, README counts, catalog §147) + the riir-ai consumer half `9a1137859` (engine `sync_relation_gate` DEFAULT-ON per the 1049 Defense-3 shape; mmorpg forward + LTM stay opt-in; w_sync stays 0.0 — follow-ups filed at riir-ai Issue 1060). File removed per noise-reduction.

Outcome as headed; full text in git history.

## Issue 931 (2026-10-12 → closed 2026-10-12) — the "symmetric_eig QL divergence" was two basis_split defects manufacturing ±inf into the eigensolve: CLOSED — root cause round 3: the ZCA whitener built the wrong matrix (scaled eigvecs ROWS; eigenvectors are COLUMNS — ≈Σ⁻¹ only on the diagonal sigmas every test used) and the whitened-moment product aliased (M' written into the buffer still read as T, compounding W row-on-row to f64 overflow ~row 41 at d=64). Fixed at `be9b2cbd2`: true W = V Λ⁻¹ᐟ²Vᵀ (column convention) + three-buffer M' product + 3 two-sided regression tests (WΣWᵀ=I on rotated graded σ; moment_scratch == naive W M Wᵀ; graded style-shard e2e d=64) + the g4 alloc canary serialized (latent parallel-test race on the shared allocator counter) + 3 mechanical lint fixes in the T1/T2 tests. GOAT re-run: G1 PASS, G2 2.081 ms (<10 ms; the corrected whitener costs ~8% over the defective 1.923), G4 PASS ×5, adversarial PASS. T3 ai-half: the basis_purity_audit re-run via the riir-ai Bench-980 lattice (ai `37b029ae7` / riir-ai Bench 980 Addendum 2) — affect A1-fear SURVIVES (0.295→0.264 identity mass; robust, not an artifact); style's FIRST complete run reads S1 0/77 (no block structure — ai-1056 finding). symmetric_eig itself was never defective — its round-2 guards caught the poison exactly as designed (file removed per noise-reduction).

Outcome as headed; full text in git history.

## Issue 928 (2026-10-09 → closed 2026-10-09) — cross-repo drift-sweep debt: seal-std registration census + 4-wave repair: CLOSED — console/locale/instrument sweeps ALL GREEN family-wide (file removed per noise-reduction)

Census `c007cf543`; four waves closed family-wide: riir-shader `1f2a195`, riir-refine `c126f441`, riir-train `45fc7799`, riir-reflex `00b4ee1`, seal-remake `b69ee61`.
80 consoles + 35 locale sites repaired.

## Issue 922 (2026-10-07 → closed 2026-10-07) — loop-straightness monitor family, the DEC-native stalemate signal (LiFT modelless residue): CLOSED — primitive half landed, GOAT ALL PASS (file removed per noise-reduction)

Landed katgpt-dec `loop_straightness` (path_action, action_efficiency eta); Bench 922 GOAT all pass.
G2 38.7 ns @ K=8 dim=4; G1 sweep caught the landing's own formula bug.

## Issue 924 (2026-10-07 → closed 2026-10-07) — citation_weight's dialects are case-sensitive; lowercase `plan N` prose is invisible to duplicate adjudication: CLOSED — both instruments repaired, oracle confirmed (file removed per noise-reduction)

Fix `b98c340dc`: citation_re() scoped case-insensitive over long-form prefixes; issue_citation_gate got the same fix.
Scanned 475 to 476 citations.

## Issue 923 (2026-10-07 → closed 2026-10-07) — escalation_guard, the shared serving-escalation guard primitive (riir-refine Plan 202 R1): CLOSED — all tasks landed, cross-recorded (file removed per noise-reduction)

Landed `6118539c7`: katgpt-core `escalation_guard` (RollingRateLatch, demote_only_decode); Bench 923: 1.74 ns/observe, 0 allocs.
Consumer halves in riir-refine Plan 202 R1.

## Issue 921 (2026-10-07 → closed 2026-10-07) — citation oracle two-lane self-reference blind spot: CLOSED — Arm B landed, corpus repaired, sweep green (file removed per noise-reduction)

Arm B landed `3249672d1` (two-lane gap via .highwater_local); riir-infer corpus repair `f0191e3` + `71c8eed`.
IN-LOCAL-RANGE 9 to 0.

## Issue 919 (2026-10-06 → closed 2026-10-07) — spike census + delimiter-sink PoC: CLOSED — census channel key REFUTED; measured-diagonal KV exemption + delimiter folding both NO-GO (file removed per noise-reduction)

Three verdicts, nothing promoted: T1 census `ff2f40e64`, T2 channel screen REFUTED `eccb13b9c` (P@4 = 0), T3 KV exemption NO-GO `0952590e1`.
Delimiter folding also NO-GO.

## Issue 906 (2026-10-05) — owner-gate pickup CLOSED: E12 Gemma base-model check RETIRED (option b); E13 rust-version pin scoped to publish=false only (file removed per noise-reduction)

E13: rust-version removed from 8 crates.io manifests (landing `880619731` over-scoped), kept on 25 publish=false.
E12 retired on standing negative (12/12 harmful recirc cells).

## Issue 918 (2026-10-05) — posterior (truncated-Gumbel) inverse sampler, max-first: CLOSED (file removed per noise-reduction)

`keyed_posterior_gumbel_noise` in default-on ac_prefix, `9ad5e21ce`-era lane; Bench 918 all gates pass (G2 2.08x, 0 allocs).
Consumer: riir-train Plan 437.

## Issue 916 (2026-10-03 → closed 2026-10-03) — two_fidelity_bai G2 cost debt: CLOSED — δ-correct baseline (a) + lazy-discharge resolver (b) (file removed per noise-reduction)

(a) LUCB-MCTS/UGapE baseline `3aa27c149`, Bench 615 GOAT PASS; (b) opt-in `lazy_discharge` `ca9fb1617`, 236-1117x below v1.

## Issue 915 (2026-10-02 → closed 2026-10-03) — two_fidelity_bai certified-interval invalidity: CLOSED — root cause was the BENCH FIXTURE, not the module (file removed per noise-reduction)

Root cause was the bench fixture's slow oracle violating MinimaxSpace; fixed, re-arm 8/8 certified.
Next finding filed as Issue 916.

## Issue 914 (2026-10-01 → closed 2026-10-02) — Pseudo-head mixing runtime (IHA distill, Research 600): CLOSED NEGATIVE before any implementation (file removed per noise-reduction)

Killed by riir-train Issue 606 evidence: Bench 621 (`e6b94720`) arm 5 catastrophic on both tasks.
No trained-with-mixing consumer exists.

## Issue 913 (2026-10-01) — Grouped-evidence noise-weighting primitives (riir-train Research 463, EasyPPO modelless half): CLOSED (T1–T4 landed opt-in; file removed per noise-reduction)

Four primitives in `grouped_evidence.rs` behind `grouped_evidence_weighting`, `9ad5e21ce`, Bench 905; T3 coverage 0.883 at nominal 0.95.
Consumer riir-clippy Issue 139 (`e5b4d56d`).

## Issue 912 (2026-10-01) — Dirichlet-distribution primitives: sampler repair + exact explore-dial sampling + Dirichlet-EMA (Research 596): CLOSED (T1–T5 landed opt-in; file removed per noise-reduction)

T1 `7745cfb9c` sampler repair; T2-T5 `9b98ee3dc` (Bench 904) dirichlet_dist.rs, thinning 183 ns @ N=8, DirichletEma.

## Plan 612 (2026-09-30) — PISA pyramid Top-K + LSE block selection: CLOSED with a G2 iso-quality NEGATIVE (latency slope CONFIRMED; two Phase-1 defects caught by the gate's pins, fixed with regressions)

Bench 612: pyramid_lse -0.133 Recall@8 at L>=32K, pyramid_topk stays opt-in; latency slope 1.07 vs 1.99.
Gate caught two Phase-1 defects before any number was quoted.

## Issue 898 (2026-09-30) — KL effective depth exit calibration for the looped runtime: CLOSED (instrument landed, G1 PASS on all three fixtures; NO Config default justified — calibrated exits are checkpoint-dependent; the contingent BO arm deferred with its unblock recorded here)

`loop_depth_probe` `aa3a7d382` (Bench 899), G1 holdout PASS; hand-tuned defaults stand.
BO arm blocked on riir-train `.plans/421` Phase 4 (`9560ce0a3`).

## Issue 911 (2026-09-30) — ties stopped at their source: the resolution-aware `w` floor + the zero-tolerance AUC guard: CLOSED (verdict round 2; the SE tolerance retired, the "preserved by construction" docs made f32-honest; companion reflex promotion landed in the same arc)

`resolution_w_floor` (2-ulp) + zero-tolerance AUC guard; clamp cost <= ~1e-5.
reflex `--gate-fit-calibrated` promoted default.

## Issue 910 (2026-09-30) — the Platt solver needs Lin–Lin–Weng, not a candidate fallback as the primary repair: CLOSED (LLW solve + the AUC ranking guard; the round-1 review's reference optima reproduced EXACTLY; held-out floors + a 48-cell property sweep pin the behavior)

`solve_smoothed_mle` is LLW form (Armijo backtracking); f64 reference reproduces production optima exactly.
Benches 807 4/4 + 808 GOAT PASS; COMMITMENT_VERSION 1 to 2.

## Issue 909 (2026-09-30) — `sigmoid_calibration` narrow-window degenerate stall + f32 tie-collapse: CLOSED (three-candidate loss-argmin fallback + output-side saturation guard; the audit refuted the early-break theory; reflex Issue 056's AUC regression confirmed fixed end-to-end)

Three-candidate loss-argmin + `window_keeps_resolution` guard; Bench 808 re-run PASS.
Early-break theory corrected by Issue 910.

## Issue 908 (2026-09-29) — Plan 612 T2.1 FA-layer Q/K capture lane: CLOSED (qwen3.8-27B captured at all four lengths; Bonsai PRIMARY a documented loader blocker; riir-infer `6cb354d`+`836d28e`)

Qwen3.8-27B Q/K captured at four lengths; 50 fixture bins (~65 MB) in pyramid_612; riir-infer `6cb354d` + `836d28e`.
Bonsai blocked: load_weights_gpu refuses Q2_0.

## Issue 907 (2026-09-29) — KVarN V-row bit arms are not a pure bit ladder on real rows: CLOSED (cache-level reproduction + crossed-config attribution on REAL captured rows — the 3-bit var-norm arm is the defect site; consumer non-interpolation rule in kvarn docs; commit `2c93727be`)

Commit `2c93727be`; Bench 903: b2<b4<b3 inversion reproduces on real rows, 3-bit var-norm arm is the defect site.
Rule: never interpolate V-row quality across KVarN bit arms.

## Issue 904 (2026-09-28) — `fit_woodbury` f32 sample-Gram Cholesky panic bypassed `tick_karc`'s warn-and-keep: CLOSED (fit paths degrade to `FitError::Singular`; commit `b8913aaa5`)

Commit `b8913aaa5`: try_cholesky_f32 / try_ridge_solve_woodbury_f32 return Result; fit_woodbury maps to FitError::Singular = 2.
Found by riir-ai Proposal 048 T8 GOAT.

## Issue 903 (2026-09-27) — `row_logit_floor` promotion lane — all four boxes PASS (Bench 902); the model-bound G1 was closed by riir-infer Bench 003; the `ForwardContext.logit_floor` default flip stays the owner act: CLOSED (staged, not deferred; commits `fdca94d7a` + `93740fe21`)

Commits `fdca94d7a` + `93740fe21`; Bench 902 all four boxes PASS, model-bound G1 closed by riir-infer Bench 003.
Default flip stays a two-part owner act.

## Issue 902 (2026-09-26) — md-only contract repos can never pass `len_derived`/`shared_temp_path` (the delegated `.rs` floor demanded non-zero): CLOSED (measured zero-row acceptance; commit `26ba6afdd`)

Commit `26ba6afdd`: `sweep_population.zero_walk_floor_accepted` accepts a zero row only while tracked_walk measures zero .rs.
Caught riir-reflexer's stale md-only premise.

## Issue 900 (2026-09-26) — issue-717 damping G4 failed at the wider feature combo: the gain_cost_halt probe buffers allocated unconditionally per call: CLOSED (lazy-sized; the measured region is combo-stable; commit `e272057be`)

Commit `e272057be`: probe scratch Vecs now lazily sized (was 16 allocs / 1024 B over 8 runs).
Static sibling hunt found zero second instances.

## Issue 899 (2026-09-26) — second-moment, null-normalized drift alignment (the Plan 610 redesign): CLOSED (negative on 2 of 6 bars; became the default summary inside opt-in `arm_drift_alignment`; no promotion)

`SecondMomentDrift` `3485ec59e`, Bench 901 is the default summary in opt-in `arm_drift_alignment`; held-out 0.766 FAIL, G4 2.07x FAIL.
New bar: loop soundness in both reward directions.

## Issue 895 (2026-09-25) — guided width rollouts on the belief host (GRAM re-distill, Research 590): CLOSED (T1–T7 + T9 landed opt-in; G1 FAILED, demote TRIGGERED, no promotion; T8 handed to riir-ai Issue 1008)

`a4421939f`/`ae7b13320`, Bench 898: MULTI +0.117, SINGLE -0.188, G1 FAILED, guided table off forever.
G2 14.9 us/decision; T8 to riir-ai Issue 1008.

## Issue 897 (2026-09-25) — katgpt-kv's KVarN `static_cal_tables` branches were gated on an undeclared feature and named a missing module: CLOSED (deleted, not wired; `unexpected_cfgs` allow removed; a second instance, `memory_soup_dtree`, restored)

Deleted dead static_cal_tables branches (dead since Issue-015 extraction `b61a34f7b`); `f3e3a4f56` removed unexpected_cfgs allow.
Surfaced `memory_soup_dtree`, restored opt-in.

## Issue 896 (2026-09-25) — KVarN's raw tile buffer was shared across layers, and the in-progress tile dequantized to zeros (2-bit keys panicked): CLOSED (fixed under `kvarn`; layer-major bit-identical; no consumer's published quality figure carried it)

Fix `62fd22b5f`: one raw tile per layer (~1 to ~27 MB); layer-major oracle 960 cases / 10,427,200 elements, 0 differing bits.
Bench 894 GOAT unchanged.

## Issue 883 (2026-09-25) — fitted token-value tables (K=V+ retrofit, mean-removed V quant, V-cache halving): CLOSED (P0–P4 landed opt-in; two primitive-level G2 bars FAILED and are recorded; the model-bound half is riir-infer Issue 013)

P0-P3 `0b768e95d`, Bench 895; promoted default 2026-10-05 via riir-infer `fe75497` (Bench 017: 1.050x/1.123x @4097).
P1 and P2 demoted on Bench 011/012 negatives.

## Issue 894 (2026-09-25) — KVarN dequant loops were scalar (bounds-checked indexing); rewritten as bit-identical zips: CLOSED (GOAT PASS, ships under `kvarn`)

`bf7d37244`: bit-identical zip/as_chunks rewrite, 960 cases / 10,407,928 elements, 0 differing bits.
G2 value 4-bit 2.46x faster; 883 P1 G2 +13% FAIL recorded.

## Issue 882 (2026-09-25) — differential anchor scoring (common-mode rejection on score surfaces we own): CLOSED (P0–P4 landed, all opt-in; promotion owed to consumers)

P0 `differential_anchor` (Bench 886, lambda*=0.55), P1 attention_snr, P2 row_logit_floor, P3 differential_kv_eviction (Bench 894), P4 riders (Bench 897).
Model-bound halves in riir-infer Issues 011/012.

## Issue 886 (2026-09-25) — activation-diagonal weight-quant fitting (AWQ/imatrix-class substrate): CLOSED (P0 + P1 landed opt-in incl. the model-bound half; P2 deferred)

P0 `0fb2254d9` (Bench 896) + P1 act_aware_fit; riir-infer Issue 014 / Bench 007: born-ternary negative, dense-parent -34.6%.
P2 deferred to a PTQ-of-dense consumer.

## Issue 887 (2026-09-25) — `ladder_gate` streak-gated advancement + corrective backtracking FSM: CLOSED (GOAT PASS, stays opt-in)

`ba7fb89ba` (Research 589, arXiv:2609.19717): streak-gated advance, shallowest-failure retreat; Bench 891 G1 17/17, 0.8-1.2 ns/eval, G4 0.
No production consumer.

## Issue 885 (2026-09-25) — `laya-tetris-v3` real-hard-drop lane (Issue 884 path B): CLOSED (every lane landed and deployed)

`1a05a9764` DropRule v2/v3; reflex `ca1483c`+`a56d850`; reflex-site `6116e70`+`2e99eca` deployed, parity 809/809.

## Issue 878 (2026-09-24) — tetris_sim fidelity boundary: RECORD RELOCATED (2026-09-25)

Rule (never swap in a guideline Tetris engine; new fidelity = new lane) now at `.docs/06_game_arenas/tetris_sim_fidelity_boundary.md`.

## Issue 873 (2026-09-22) — governed paged-pool primitives from mini-AGI: CLOSED (all three landed, opt-in; consumer work handed off)

Research 581: `pool_admission` `eabd0cb80` (Bench 873), `rate_control` `ee01f1598` (Bench 875), `dying` `650faeac7` (Bench 874); all GOAT opt-in.
Consumer A/B to riir-train Plan 416.

## Issue 884 (2026-09-25) — tetris_sim `hard_drop` tunnels through roofs: CLOSED (fix path A, site-side; corpus untouched)

Path A: live-play filter in reflex-site `c1fcf88` (deployed `31744b04`); hard_drop tunnels 3 of 2660 options; sim and fixture unchanged.

## Issue 880 (2026-09-24) — calibrated-mass router gate, `exact_mass_admit` consumer lane (a): CLOSED (shipped opt-in, G2 FAIL by construction)

`gate_sigmoid_topk_mass_into` (opt-in `calibrated_mass_gate`, Bench 885): |sum m - k| <= 9e-7; G2 fails by construction.
Lesson: a ratio between alternatives does not transfer to their composition.

## Issue 881 (2026-09-24) — the numbering sweep's standing backlog, three repos over their ratchets: CLOSED

T1 riir-ai: 17 walker phantoms (`216defb9b`); T2 riir-shader renames (`3141a0467`); T3 mmorpg-editor one real reset, pin 5 to 6.

## 2026-09-23 — the owner-gates menu v2 executed: schedules re-armed (row 5), pipefail residue closed (4e), toolchain batch-pin (4f), and the opt-in verdicts recorded (4a/4b/4c/4d)

Schedules re-armed; pipefail residue closed (riir-neuron-db `c5c11b5a273e`); toolchain batch-pin in 5 repos (`8c7755b3fdbb`, `c47227e06fcb`, `d0c917f1eb60`, `ee8b6c2bee6c`, `d54fbe8ebf32`).
Verdicts: certified_frontier promote-with-consumer, gw_alignment NO, hint_regret opt-in.

## 2026-09-23 — Issue 877 CLOSED (merge `b701cf564`): the main↔develop sync — origin/main merged into develop with `-s ours`, zero content delta, ancestry restored

Merge `b701cf564` (-s ours), zero content delta, pushed atomically to main.
Lesson: merge-tree hashes are spelling-bound; quote the file set.

## 2026-09-23 — Issue 867 CLOSED (+ the issue): the non-hidden-state canonical-AST construction — G5 returned NO attributable signal; instrument-broken, not claim-refuted

T1 `65b199b0` + T2 `67884461` in katgpt-canon; G5 (riir-train Bench 605) PRIMARY +0.50..0.55 but shuffle null alone +0.41..0.44.
Instrument broken, not claim refuted.

## 2026-09-23 — Issue 875 T3 CLOSED (+ the issue): time-annealed sampling ranges + the closed-form truncation predicate (Bench 883)

`TimeAnnealRange` + truncation predicate eps=((T-t_cut)/(T-t_min))^2; Bench 883, cross-repo on riir-train C9 toy: +9.0% W1 for 9.37% mass dropped.

## 2026-09-22 — Plan 607 ACCEPTED + T0a/T4a/T0b landed: the modelless game-decision lane's first three tasks (substrate gate, the laya-Tetris enumerator, the G1-oracle fixture)

T4a `tetris_01_state_enum.rs` (120 states / 2,660 options, 245 sentences, 0 ties); T0b oracle in riir-reflex `e4bf657`.

## 2026-09-22 — Issue 874 closed: G8 re-founded on real claims — the dead speedup gate became a bit-contract pin + a relocated, executing throughput floor (two sessions, one issue)

G8 retired as relative claim: G8a bit-agreement + G8b throughput floor on bench_271 in x86_64 matrix; G7's constant-folded 0ns fixed same day.

## 2026-09-22 — Issue 873 primitive B landed twice in one evening: the twin-duplicate resolution (third of the class)

Origin-first `ee01f1598` won, duplicate reset away. Rule: a handoff naming remaining work is an assignment to every reader; git fetch and check before pushing.

## 2026-09-22 — Issue 871 closed: the `algebraic_*` A/B measured end-to-end — feature-gated `algebraic_dot` ADOPTED (owner verdict a′), the in-crate codegen truth repaired, T6/T7 deferred on-record

`algebraic_dot` adopted opt-in (1.77x-8.8x); `algebraic_div`/`algebraic_rem` banned via `scripts/algebraic_op_ban_gate.py`.

## 2026-09-22 — Issue 872: SIMD bitstream whitespace splitter for `encode_into_pretok` (the bitcannon-class port) — scan 1.60×/1.69× measured, bit-identity by differential, four live bugs caught by the harness

`fast_bpe/simd_split.rs` scan 1.60x/1.69x (avx2); differential harness caught four live bugs; stays opt-in.

## 2026-09-22 — Issue 870 closed: distance_abstain's rationale-free sigmoid copy → exact_sigmoid delegation, measured 3-ULP envelope, bench_845 GOAT re-run identical

Delegated to `exact_sigmoid` with 3-ULP envelope pin; audit `683d06d8`; bench_845 G1-G5 identical.

## 2026-09-22 — Issue 869 T5: the mini dllm lane goes per-layer honest end-to-end — the gradient check caught a live backward bug the loss-decreases gates never could

Gradient check caught backward bug (`!is_masked[p]` skip valid only at readout layer); fix `last && !is_masked[p]`.

## 2026-09-22 — Issue 869: multi-layer D2F decode + taps at depth — the Issue-865 Bonsai-scale unblock lands; bitcos x86_64-lane clippy debt repaired in passing

`forward_block_causal_with` + `set_probe_tap_layers` land the multi-layer D2F unblock; `micro_dllm_text()` indexes layers[0] only (T5 migrates).

## 2026-09-21 — Issue 864 closed: BITCOS tier ships opt-in — footprint PASS, latency honestly LOSES on this host (Bench 846)

`bitcos` footprint PASS above z=0.375, latency G2b(a) FAIL on the 13700K, so opt-in (Bench 846); two codec bugs caught pre-timing.

## 2026-09-22 — Issue 868 closed NEGATIVE: engram-fused PUCT G5 FAIL — the evidence gate worked, and that is why nothing happened (Bench 848)

`engram_puct` G5 FAIL 48.1% h2h (Plan 605, Wilson lower 44.8%); G1/G4/G6 PASS; re-open on transposition-dense domain.

## 2026-09-21 — Issue 858 closed: g8's arch-conditional bar gets its aarch64 executing lane (PERF_ROWS)

Arch confirmed, cache regression refuted (13/13 x86_64 PASS); dual pin `G8_BAR` + `PERF_ROWS` row kind in `scripts/test_gate.sh`.

## 2026-09-21 — the x86_64 execution matrix caught a day-old test that had never executed on x86_64: the ordered-dot anti-dedup pin was crafted at NEON's vector width

`ordered_dot_differs_from_simd_dot` (`5458dd69`) red on x86_64: len-4 pin below AVX2 floor; recrafted at len 16, rerun 7/7.

## 2026-09-21 — the first full-gate run since 09-16 caught the Issue-860 landing RED: 8 `-D`-list errors in the opt-in feature's test code, invisible to every default-feature lane

Layer 3: 8 `-D` errors in `successor_density_critic.rs` test code (opt-in feature, all-features only); repaired, 1120s warm.

## 2026-09-20 — the x86_64 execution matrix's first full run since 09-16 (354 commits of drift): 11,344 assertions PASSED, and one more load-flipped bar caught by execution

11,344 assertions PASSED; `bench_105_gdn2_goat` goat_6 PASSED-ALONE (0.306 vs 0.30); fixed via `best_of_arms` in `tests/common/ab_timing.rs`.

## 2026-09-20 — Issue 861 CLOSED: the ugc_alloc_check Windows G4 alloc was a per-call env read — and the issue's own isolation table was wrong

Per-call `std::env::var("UGC_DEBUG")` allocated on Windows; fixed with `OnceLock<bool>` caching, G4 0 allocations.

## 2026-09-20 — Issue 859 CLOSED: Jev structured reads — POC GOAT + 4090 reference + T5 policy arm measured; promotion declined on layer posture (evidence-banked)

`structured_read[_into]` opt-in (Research 574, Bench 816 GOAT, Bench 817 T5); agreement bars do not beat single-read confidence; stays opt-in.

## 2026-09-19 — the open-issue backlog is cleared by owner call (12 files removed; triggers recorded here)

Every open issue adjudicated in one owner session; 12 files removed, reopen triggers recorded; no number reused.

## Issue 780 — CLOSED as PARKED: OnlineLinearReadout waits on a consumer that measured itself absent (2026-09-19)

Primitive stands but consumer closed NEGATIVE (riir-clippy Issue 107, `5dfa1daff`); reopen triggers recorded.

## Issue 827 — CLOSED: T4 decided by owner call — known-extra repos ARE legitimate oracles; the freshness guards T2/T5 landed are the correct limit (2026-09-19)

T1/T2/T3/T5 landed (`6d084c38`, `7464fc6e`, `3948f0e2`): ORACLE-STALE bucket, fetch-age advisory; T4 status quo (hazard was freshness).

## Issue 833 — CLOSED: the class is measured, the executed backlog is 3 of 38, and the residue is routed (2026-09-19)

T3 resolvers (`381f01f7`, `e56e6773`, `1b7756c3`, `71d97908`); T2 executed 38 GATES rows, 3 candidates; spawned Issues 855/856; no verdict half.

## Issue 834 — CLOSED: census shipped, the count reframed as a sampling population, and the shared-crate question is decided NO (2026-09-19)

T1 shipped `scripts/sequential_ab_timing_audit.py` (`8045fae9`); T3: `ab_timing.rs` stays `#[path]`-included, no shared crate.

## Issue 835 — CLOSED: T4 decided by owner call — the path dep is the workspace convention; publishing is contrary to Research 003 (2026-09-19)

T1 `cross_repo_path_dep_gate.py` (`5b09c3f4`), green over 311 deps / 183 manifests; riir-llm keeps its path dep.

## Issue 839 — CLOSED: kron_tile 4/4 GOAT; T7 DECLINED on measurement (2026-09-19)

kron_tile 4/4 GOAT (`a623d7d4`, Bench 839, opt-in); T7 declined: Bench 843 ternary/f32 1.56x slower at w=32.

## Issue 841 — CLOSED as lead-backlog dissolved: 7 rows landed, the remaining leads stay at their research-note homes (2026-09-19)

Landed `.kpt` archive POC (Bench 841), `legal_token_set` default-on (20.6-22.5x), `kv_sink_window` 5/5 GOAT, calibration staleness (`2057ac3b`); leads at Research 568/570/571.

## Issue 852 — CLOSED as PARKED: the τ(t) P_e-LUT POC waits on a D2F model consumer (2026-09-19)

Modelless half landable but G2 needs a D2F model harness that does not exist; reopen on real D2F lane or riir-train Issue 563.

## Issue 853 — CLOSED as PARKED: the autoguidance POC waits on its measurement arms (2026-09-19)

Arm A needs a DDTree acceptance harness; Arm B is riir-train Issue 563's free-rider; prior art Research 572.

## Issue 854 — CLOSED: the wedge is diagnosed, the repair scoped, and T3 stays trigger-gated by design (2026-09-19)

Landed `dded36d8`, `65bf2944`, `d8b55a34`, `c83f0f84`: SLOW-vs-WEDGED recipe, census, external per-module timeout; T3 trigger-gated.

## Issue 855 — CLOSED: every task resolved; the guard gate + ratchet sweep both shipped (2026-09-19)

7 of 34 timed regions vanished, all repaired; `timed_region_guard_gate.py` (`196a4cda`); riir-ai `3712d51b6`; sweep `72ee0f5b`.

## Issue 857 — CLOSED as PULL-GATED: the corpus builds when riir-ai 882's consumer fires, not before (2026-09-19)

Boundary answered (`51291548`): corpus/harness = riir-train, canary = riir-ai Issue 882 T2, scorer here; pull-gated on riir-ai 882 T-Pull-1.

## Issue 843 — CLOSED: the plasma_path ternary dense matvec loses below L3 on every served shape; T4 resolved as per-shape dispatch (2026-09-19)

Ternary matvec 1.9-3.0x slower when f32 fits L3; per-shape `simd_matvec_plasma_dispatch` GOAT GREEN; `plasma_path` stays default-on (Bench 843).

## Issue 831 — CLOSED: bench_171 P3 reclassified as instrument-health + mechanism by owner call (2026-09-19)

P3 now gates instrument health + exact call count (13,824 vs 41,472); the >=30% latency assert removed, not lowered.

## Issue 847 — CLOSED: `simd_lut_dequant`'s AVX2 kernels compiled to NOTHING on every ordinary x86_64 build — and the bf16 sweep that followed was right for one kernel of three (2026-09-19)

Runtime `simd_level()` replaced compile-time avx2 gate (1.7x, 4.4-5.6x, RNE 2.4-2.5x); `shipped_target_feature_gate.py` walls it.

## Issue 850 — CLOSED: the UNVERIFIED-upstream guard challenged only ONE of `behind_origin`'s two silent readings — and then PRINTED a false statement about the one it gained (2026-09-19)

Guard gap repaired (6 findings, 3 repos); sweep FETCH? NO (250.2s vs 50.2s 8-way); `fetch_contract_repos.py`; COUNTER axis armed.

## Issue 832 — CLOSED: the last fixed shared-temp site is repaired — the non-demo backlog is zero, the ratchet is the wall (2026-09-19)

Final site seal-remake `db68f5e`; siblings riir-ai `9d138531b`, riir-chain `8a3b0f5`, riir-game-sdk `ef66117`, riir-clippy `f5ada0ec`; matrix 11,177 assertions green.

## Issue 844 — CLOSED: the dot-delegation crossover is length 24, per-ISA, measured on BOTH arches — not a backlog (2026-09-19; full record at the 09-19 dated entry below, file removed this commit)

Crossover is length 24 on x86_64, delegation wins at every length >=4 on M3 NEON; full record in the 09-19 dated entry.

## Issue 840 — RESOLVED: two `.research/569` documents landed forty minutes apart and tripped two detectors — the rewrite set was EMPTY, and the attribution lessons became the staged-set rules (2026-09-19)

Neither document was a peer's; rewrite set empty; durable output is the AGENTS.md staged-set attribution rules (`4e82cc489`, `c0995ea4b`).

## Issue 836 — CLOSED: a `git worktree` made the entire head-provenance mechanism silently inert — `.git` answers TWO questions and each spelling was wrong for the other (2026-09-19)

T1 `f6749af5c` (`.is_dir()` -> worktree guard); T2+T4 `c12e96415` delegate to `worktree_state.is_checkout`; riir-ai `9637d09ea`.

## Issue 845 — CLOSED: `channel_aware`'s duplicated dot kernel deleted; T5 measured NEON PARITY — the x86_64 1.8–7.8× penalty does not transfer to aarch64 (2026-09-19)

~200-line duplicate AVX2 arm cfg-off (2.72-7.76x slower) deleted by delegation (`32056164e`); NEON parity 0.98-1.04x.

## Issue 851 — CLOSED: `muon_update` step scale repaired to the canonical √max family (update RMS ≈ 1.0) (2026-09-19)

Fix `1ebe39f61`: scale `1/max` -> `sqrt(max(rows, cols))`; regression pin update RMS in [0.68, 1.12]; Bench 050 unaffected.

## Issue 842 — CLOSED: the alias-mapped sweeps opened a directory that does not exist, and the first real read surfaced a workspace of hidden findings (2026-09-19)

17 of 19 sweeps opened nonexistent dirs; one seam `sweep_population.open_repo` + `repo_alias.real`; hidden findings surfaced.

## Issue 846 — CLOSED: the alias seam has a PROSE half — 44 true CROSS rows, 37 cleared by teaching the qualifier the on-disk spellings, 7 by prose (2026-09-19)

Qualifier taught on-disk spellings (`issue_citation_gate.spelling_aliases()`, leniency only); fixes riir-dao `160f9a1`, riir-kat `51994ee`, riir-neuron-db `3f6cf67`.

## Issue 838 — CLOSED-as-decided: a shell script spawning a NATIVE child is a third encoding seam, and the population is measured at zero live instances, so it is deliberately NOT gated (2026-09-18)

Census (`c9afca9dd`): 43 native-child lines, 2 non-ASCII comments; population zero so not gated; keep native children ASCII.

## Issue 837 — CLOSED: registering a contract repo reds the whole sweep family, and the ONE file a gate checks is not the twenty-one that make them red (2026-09-18)

`riir-llm` registration reddened 19 of 21 sweeps; T2 `scripts/repo_registration_gate.py` + `repo_registration_scope.txt`.

## Issue 825 — CLOSED POSITIVE, after a same-day RETRACTION of its own negative close. Coulomb crowd redistribution ships as `coulomb_flow`; the "negative result" was a bench walker with two defects (2026-09-18)

`coulomb_flow` ships opt-in (Bench 825, G1-G4 PASS); earlier negative was two walker defects; MAE 0.0 vs 0.119; Bench 815 repaired.

## Issue 830 — CLOSED: Issue 829's anchor class one seam deeper — the locale-I/O classifier was anchored to a CALL-NAME SET, and it had repaired one side of a round trip in this repo's own instrument (2026-09-18)

`locale_io_fix.sites()` gains text-mode file-object forms (11 sites); siblings riir-train `15db2c67`, riir-clippy `54b999de`.

## Issue 828 — CLOSED: Issue 823's anchor class, third position — the heading oracle was anchored to a DELIMITER SET, and the biggest unread family is this repo's own house style (2026-09-18)

Em dash added to the delimiter set (56 unread were house style); pricing: 2 of 153 unread novel (98.7% redundant); cp874 mangling = Issue 829.

## Issue 823 — CLOSED: the heading oracle was anchored to a POSITION, and its own blindness meter was anchored to the same one; T5 ANSWERED by Issue 828 T4 (2026-09-18)

`_SELF_HEADING_DATED` repair; IN-LOCAL-RANGE 54 -> 27; the blindness meter failed toward clean; answered by Issue 828 T4.

## Issue 829 — CLOSED: the FILE seam under Issue 778's PIPE seam: 273 text-I/O sites decode with the system locale, and the first one found was making a whole file of arms pass for the wrong reason (2026-09-18)

273 text-I/O sites in 8 repos; `scripts/locale_io_fix.py` + `locale_io_gate.py`; 118 sibling rows (riir-train `39d0b8d4`, riir-clippy `638fa224`, riir-ai `4df109c65`, riir-chain `bebf78a`).

## Issue 824 — CLOSED: the family gate watched the failure it was built for happen beside it (2026-09-17)

Family is 19 not 16 sweeps; repair is the `sweep_advisory_membership_gate.MECHANISMS` registry, pinned per `(mechanism, sweep)`.

## Issue 821 — CLOSED: Issue 815's marker never reached the sweep family: 8 sweeps red on 0 findings, and two live ratchet breaches sat behind them (2026-09-17)

`sweep_population.pin_row_exempt(name)` wired into 16 sweeps; two hidden riir-train breaches surfaced.

## Issue 820 — CLOSED: the numbering sweep is blind to Issue 795's class, so 15 repos read `dup=0` over 113 collisions (2026-09-17)

183 historical collisions over 16 repos (113 hidden); `max_hist` ratchet + `min_numbers` floor; 819 dual-allocation kept by sigmoid lane.

## Issue 822 (2026-09-18) — CLOSED. An UNCOMMITTED row was counted into a RATCHET; all 19 sweeps adjudicate against HEAD now, and the mechanism is GATED

`worktree_state.py` head_delta/head_overlay/head_tree; verdict in row KEY, floors are pins; `head-provenance` MECHANISMS row.

## Issue 819 (2026-09-17) — CLOSED. The x86_64 arm was LINTED by nothing; the finding is not the 30, it is the sibling

30 `unsafe_op_in_unsafe_fn` in AVX2 arm of `channel_aware.rs`; T3 `full_gate.sh` Layer 2c; 819 dual-held, pinned in `number_collisions_expected.txt`.

## Issue 819, second holder (2026-09-17) — the sigmoid prior-logit lane KEPT the number; RESOLVED (file removed)

Sigmoid prior-logit lane kept 819 (7 of 10 inbound); Bench 813 G1-G4 PASS; two opt-in features (Research 566).

## Issue 815 (2026-09-17) — CLOSED by its own criterion: the box is green. Option 2 landed as `DOCS_GATE_KNOWN_EXTRA`, and options 1 and 3 remain the owner's

`DOCS_GATE_KNOWN_EXTRA` mirrors PARTIAL_CLONE (names only, reds both ways); `skill_repo_set_gate.known_extra_state`; options 1 and 3 remain owner's.

## Issue 808 (2026-09-17) — CLOSED: T1 landed option 2, the AVX2 `argtopk` dispatch narrowed to k ≤ 4 on x86_64; the measured loss is GONE and the wins are kept

`AVX2_ARGTOPK_K_MAX = 4` on x86_64: late_peak k=8 0.41-0.72x -> 0.98-0.99x; k=2 5.11x, k=1 6.73x kept.

## Issue 806 (2026-09-17) — CLOSED: T8 was the last open task and it was entirely Issue 808's T1

T6/T7 closed 2026-09-16 (Bench 806 Addenda); T8 was Issue 808's T1; matrix stays a workstation verdict.

## Issue 818 CLOSED (2026-09-17) — the four no-default `--all-targets` breaks gated both halves; bench_412's green-zero row found in the same sweep

Fix `78a1ac5d7`: whole-file `#![cfg]` + `required-features` rows for four targets; bench_412 green-zero found.

## Issue 808 (2026-09-17) — T4 LANDED: `argtopk` AVX2 dispatch re-measured on six realistic block-score distributions + per-k `N_MIN` crossover (Bench 810)

Six distributions: k<=16 loss survives, deepens on late_peak; k<=4 robust; `N_MIN` distribution-sensitive (Bench 810).

## Issue 817 (2026-09-17) — single-pass AVX2 `argmax` port measured NEGATIVE + reverted; the NEON premise does not transfer (Bench 812)

Single-pass AVX2 argmax a net loss (0.20-0.28x at n>=256); dispatch reverted, kernel deleted; `bench_817_argmax_dispatch_ab.rs` kept (Bench 812).

## Issue 811 CLOSED (2026-09-17) — DBTM confidence-commit anchor rule: PoC PASS → Plan 600 landed end-to-end, promotion EXECUTED via Plan 601

Research 563 -> Plan 600 (`8bd6d7132`, Bench 600) -> Plan 601 promotion: `ConfidenceAnchorConfig::default()` kappa 0.9 + floor.

## riir-ai Issue 964 C2 LANDED (2026-09-16) — `CalibratedActionBridge`: decision-level confidence calibration for the ABSTAIN threshold (Bench 808)

`CalibratedActionBridge` (Bench 808): ECE 0.0220 -> 0.0088; ABSTAIN point 4.4x toward oracle; 0 allocs.

## riir-ai Issue 964 C1 LANDED (2026-09-16) — `clr_calibration`: the CLR verifier becomes the first `sigmoid_calibration` consumer (Bench 807)

`CalibratedVerifier<V>` in katgpt-claim (opt-in, Bench 807): ECE 0.0924 -> 0.0164; identity fast path at (w,c)=(1,0).

## Issue 812 CLOSED (2026-09-16) — the bench_doc_audit BlindRead context split was a TMPDIR-FORM split; the arm now matches paths form-independently

Fix: `root_forms = (str(root), str(root.resolve()))` in `scripts/bench_doc_audit.py` (TMPDIR-form split).

## Issue 810 CLOSED (2026-09-16) — `sigmoid_calibration`: the Platt-style calibrated sigmoid gate lands as a PoC with all four gates green

`SigmoidGateCalibrator` (FIFO, 2-param Newton, BLAKE3 commit); ECE 0.0696 -> 0.0322; consumers filed as riir-ai Issue 964.

## Issue 809 T1+T2 CLOSED (2026-09-16) — the global-RNG census is read; the class is gated; the T3 `Rng::new()` census is deferred with a reason

20 sites / 9 files, one defect fixed; T2 `scripts/global_rng_gate.py`; T3 `6d120abeb`: 147 `Rng::new()` sites pinned.

## Issue 806 T6 CLOSED (2026-09-16, the M3 side) — t698_t5_kv_mean adjudicated arch-dependent with dual pins; kda grad-check floor was below its own noise

aarch64 pin `23d0daab3f087159`, x86_64 `4d0b592740db9358` -> dual pins; kda grad-check floor 5e-4 -> 2e-3.

## Issue 807 (2026-09-16) CLOSED — `lthash`: incremental homomorphic multiset hash, the shared substrate for the Agave-mined commitment/state-hash proposals

`crates/katgpt-core/src/lthash.rs` opt-in (eprint 2019/227); Bench 771: `replace` 31.6 ns vs 33.8 us rebuild ~1069x; awaits consumer.

## First x86_64 execution of the katgpt-core/katgpt-types SIMD suites (2026-09-16) — 15 latent AVX2 bugs caught and fixed; Bench 800's execution-parity caveat resolved NEGATIVE then closed

15 FAILED on first x86_64 `+avx2` run (bf16 RNE dropped `t >> 16`, SSE4.1 gather misuse); all fixed; Bench 800 caveat closed negative.

## Issue 805 (2026-09-16) CLOSED — `numbering_gate.py --help` printed ten "remove the row" lines about a repo it could not read

`--help` read as repo path -> ten false STALE lines; fix `unmeasurable(repo)` exit 2; arm_reach 25 modules / 691 mutants / 465 killed.

## Issue 804 (2026-09-16) CLOSED — 28 instruments crashed when run the way AGENTS.md says to run them; the cross-repo axis is seven repos, not one

console_safe.py (backslashreplace) across 28 callers; console_encoding_gate.py joined CHECKS. Cross-repo 71 undefended in 7 repos vs katgpt-rs 0/72.

## Issue 803 (2026-09-16) CLOSED — the off-macOS partial gate printed the SAME final line as a full pass

Final line of the partial gate now names the partial verdict and unmeasured axes (`3ceb541b`); develop had carried 24 E0560 + 4 lint errors for ten hours.

## Issue 802 (2026-09-16) CLOSED — commitment-gap calibration rig: residue DEAD-BY-DOMINATION at micro scale; stability features (item 3) shipped earlier in the day

Bench 802 G3 FAIL: no-gate one-forward baseline dominates (+3.5 acc at half NFE). Root speculative re-exports StabilityTracker/TOPK_DRIFT_K/N_STABILITY_FEATURES.

## Issue 800 Arm C phase 1 (2026-09-16) — GraphStablePool<T> extracted: the common contract verified across 4 sites (a 4th found in-repo)

`877e06eb2` + Bench 800-C: index-stable slots, LIFO free list; contract shared by 4 sites (radix_prefix, PagedKVCache, riir-gpu arenas, BranchBank).

## Issue 801 (2026-09-16) CLOSED — T4 PoC: composition-quality REFUTED, disagreement-trace CONFIRMED; T5 routing: meld stays opt-in as a contradiction detector, Super-GOAT Q3 blocked-as-refuted, T2 audit stands

riir-ai Bench 932: composition quality REFUTED; lambda disagreement trace (AUC 0.976, precision@8 0.997) shipped opt-in in `katgpt_core::meld`. Research 560 downgraded.

## Issue 800 Arms A+B closed (2026-09-16) — bf16 SIMD G2 FAIL-honest (autovec parity); slot-flip DECLINE (ties mpsc); JSD kernel lands (Issue 802 item 2)

Arm A `f314d5006`: bf16 SIMD 1.00x widen/trunc vs >=4x gate, FAIL-honest opt-in. Arm B slot-flip DECLINE (ties mpsc). `jsd_topk` kernel landed.

## Issue 801 T1–T3 (2026-09-16) — NAP audit + `meld` primitive: the census survives code-level scrutiny; the algebra holds, the quality ladder did NOT transfer (T4 is the sole adjudicator)

`meld` (`43f15f7c8`) bit-exact commutative soft-min, lambda* <= 3.24e-7; Research 560 audit 4 INADMISSIBLE/1 PARTIAL/1 N.A.; paper quality ladder did not reproduce at D=32 (Bench 801).

## Issue 800 Arm A (2026-09-16) — bf16⇄f32 SIMD kernels: G2 FAIL-honest — the pufferlib kernel-shape premise does not exist on M3+rustc

bf16<->f32 NEON/AVX2/scalar kernels RNE bit-exact vs half; G2 FAIL 1.00x (LLVM autovec parity). `bf16_simd` opt-in, AVX2 compile-verified only.

## Issue 799 (2026-09-16) — bevy_ecs 0.15→0.19 bump landed: the arenas are load-bearing evidence infrastructure, bevy_ecs is a schedule-free utility layer

bevy_ecs 0.15->0.19 bump `e0f02ab93` (Events->Messages); Bench 799 G1 byte-identity PASS, full-game harness ~2x slower, documented.

## Issue 798 (2026-09-15) — a tracked landing record claimed a sibling-repo repair that was never committed

Record claimed uncommitted sibling repairs; rule: cite the sibling commit and check it resolves (`b592a213` repeated the defect). T2 behind_origin(), T3 n_assertions().

## riir-train Issue 549 fixed in `490b662e` (2026-09-15, M3 + 4090 session) — avx2_exp_sum_inplace: the one exp kernel missing the n-clamp

riir-train Issue 549 fixed `490b662e`: avx2_exp_sum_inplace lacked n-clamp (exp(-300)=6.9e23 wrapped); new test pins all three paths.

## Issue 779 T1+T2 (2026-09-15, M3 session) resolved — subspace_intervention promoted + the FUNCATTN spectral arm POSITIVE (Bench 766)

`subspace_intervention` opt-in; Bench 766 spectral_pre_rotate POSITIVE: eigen-aligned 0.802 vs random 0.354, above full 0.656.

## Issue 779 T3 (2026-09-16, M3 session) resolved — real-bank affinity: saturated plateau, NO re-pin; three-arm POSITIVE on real tensors (Bench 767)

Bench 767 real gemma-2-2b bank, BLAKE3-identical regeneration; affinity ceiling plateau, no re-pin; top-4 probe-SVD dims = full (1.000) vs random-4 0.146.

## Issue 782 (2026-09-15, M3 session) resolved — slt_sweep: the noise-sweep λ̂ estimator (781 T4), GOAT G1–G4 ALL PASS, promoted default-on

`slt::sweep` v2 (log-log CDF slope) GOAT G1-G4 PASS, promoted default-on (Bench 765); v1 was -28..-37% biased; isolated minima -18..-19% at d>=4.

## Issue 781 (2026-09-15, M3 session) resolved in `580bda30` — slt: the RLCT λ + WBIC selection primitive, GOAT G1–G4 + floor ALL PASS, promoted default-on

`580bda30`: `katgpt_core::slt` six closed-form RLCT/WBIC functions, G1-G4 + floor PASS, default-on (Bench 764); WBIC recovers planted rank 6; UQ margin ~1%.

## Issue 775 (2026-09-14, M3 session) resolved in `3a59abe1` — dual_wave: the PC-ALM dual accumulator + closed-form rate laws (core) + the ballistic DEC wave kernel (dec), GOAT ALL PASS, opt-in

`3a59abe1`: opt-in `dual_wave`, GOAT PASS; reach wave linear (18/97/212 ticks at L=16/64/128) vs heat quadratic; not promoted (zone hierarchies L~4).

## Issue 777 (2026-09-14, M3 session) resolved in `7e2a2638` — modality_additive belief kernel: FLYNN's linear-sensory-integration property distilled, measured, GOAT-passed, promoted same-day

`7e2a2638`: `evolve_belief_additive` exact superposition at 26.8 ns/tick, promoted to katgpt-sense default; PoC refuted two FLYNN specifics.

## Issue 789 (2026-09-14) — a gate whose own failure path is asserted by nothing: CLOSED

Six of twenty CHECKS invoked no arm; added `scripts/check_validation_gate.py`; no sweep half by design (katgpt-rs sole repo with a CHECKS array).

## Issue 788 — the population-predicate registry was hand-maintained, and a careful reading missed two of ten: CLOSED (2026-09-14)

PREDICATES registered 7 of 10; fixed (.exists vs .is_dir, params, coupling); SUBSET_PREDICATES split; discriminator is directory iteration.

## Issue 787 — a census reads the DOCUMENT, so an undocumented instrument is invisible to it: CLOSED (2026-09-14)

Reachability predicate (roots AGENTS.md + docs_gate.sh + workflows); of 9 unreachable, 2 wired, 7 pinned with reasons; sweep is a ratchet; HISTORY.md not a root.

## Issue 786 — the `.len()`-derived binding audit had no verdict half, and the reason it went unnoticed is the finding: CLOSED (2026-09-14)

Ninth instance of the undocumented-instrument class; verdict half added; floors vacuous in 14 of 16 repos; UNRESOLVED 118 of 164 stays unpinned.

## The platform-dead_code class got an instrument — `scripts/platform_dead_code_audit.py`, and it was wrong on its first sweep (2026-09-14, 4090 session)

Audit wrong on first sweep (invented a finding, MOD-REF bucket, vendor arm); two real riir-ai rows fixed. Standing 0 findings, 1 MOD-REF.

## The Windows all-features lane — NEON_U8 platform gate, first specimen of the platform-dead_code class (2026-09-14, 4090 session)

`NEON_U8` ungated but aarch64-only used; fixed `ea4c2873`. Lesson: a `clippy::`-prefixed JSON grep reads false all-clean.

## Post-riir-train-513 develop drift — the 09-12→09-14 touched-rows window audited green on the workstation (2026-09-14, M3 session)

Touched-rows gate over the drift window: 25/25 BUILDS, 0 FAIL, 0 UNSEEN; rows had drifted to 710.

## Issue 774 (2026-09-14, M3 session) resolved in `22e65be4` — the wasm32 surface audit's BY-DEP verdict: the row predicate was dep-blind, closed with a five-canary self-test

`22e65be4`: fourth verdict `by-dep` (non-optional in-repo path deps naming wasm32), never folded into NAMED; standing 26 NAMED/2 BY-DEP/0 UNRESOLVED/1 UNCOVERED.

## Issue 748 (2026-09-14, M3 session) resolved — option (a): all three unwired Lean negative tests now run in their lean_proofs.yml CI jobs (~162s/main push)

Three Lean negative tests wired into lean_proofs.yml (~162s/main push): katgpt-rs `3c97358c`, riir-ai `ecb21f3f7`, riir-neuron-db `4a68575`.

## Issue 773 (2026-09-14, M3 session) resolved — the 772-B2 removal's stale re-export: root lib E0432 under `flashar_consensus,plasma_path`, found by riir-ai's guard through the path dep

Stale root re-export under `flashar_consensus,plasma_path` (E0432) found by riir-ai's guard; fixed `b7fcabd8`, 3-line deletion.

## Issue 771 (2026-09-14, M3 session) resolved — the radix-tree prefix KV cache primitive (RadixAttention index) shipped opt-in; G1–G4 ALL PASS, promotion deferred to the serving lane

`RadixPrefixTree` opt-in, G1-G4 PASS: hit-rate 2.45x flat control, match latency 9.8x; stays opt-in (all lanes single-stream).

## Issue 770 (2026-09-13, M3 session) resolved — the counter walker rebuilt per-commit; the 769 adjudication was partly an instrument artifact (verdict-review round 2)

Counter walker rebuilt per-commit: 15 of 31 resets were phantoms; 27 real resets; Issue-768 verdict stands.

## Issue 769 (2026-09-13, M3 session) resolved — the counter-reset class lands in the numbering sweep; all 31 measured resets adjudicated

numbering_drift_sweep gains `resets` (ratchet) + `unbumped` (report); walker from highwater_contiguity_audit.py, read with Issue 770's correction.

## Issue 768 (2026-09-13, M3 session) resolved — the .highwater ownership witness REFUTED by measurement; highwater_contiguity_audit.py landed

438 gaps + 27 resets over 73 counters: ownership witness REFUTED; contiguous-suffix witness DECLINED; protection stays on the canonical heading.

## The all-features E0252 root-name collision — ooo_audit::AuditScratch aliased (2026-09-13, M3 idle sweep)

Root re-exported `AuditScratch` twice under both opt-ins (E0252); fixed `a0ca7d36` (alias OooAuditScratch at root only).

## The Windows default-features workspace lane — bench_mtp_metal_batch_floor platform gate (2026-09-13, 4090 session)

bench_mtp_metal_batch_floor gets macOS item cfgs on 22 items + not(macos) stub; Windows default workspace lane hit 0 warnings.

## Issue 766 (2026-09-13, 4090 session) resolved — len_derived audit: caller tracer (HALF C) + two instrument defects found and fixed

len_derived caller tracer (HALF C) + brace-matched body fix (22 of 74 false positives); 118 UNRESOLVED floor; witness `e4792a4b` bumps highwater 765->766.

## WeightEpoch — the KV-cache weight-identity epoch (riir-ai Issue 938; Plan 025 contract) (`49f5d245`, 2026-09-13)

`49f5d245`: `LoraAdapter::weight_epoch()` BLAKE3 identity epoch (not a counter); riir-ai consumes at CpuInferenceBackend, refusing mixed-epoch reads.

## Modelless-first mandate — original section (incl. the canonical-failure story)

Parked narrative; live rule in AGENTS.md. AC-Prefix G1 (Plan 313) modelless fix landed bit-identically, `ac_prefix` default-on.

## Modelless-first mandate (the core principle)

Parked copy; live rule in AGENTS.md §Modelless-first mandate.

## Boundary contract — original section

Parked narrative; live pointer in AGENTS.md §Boundary contract.

## Boundary contract — read `BOUNDARY.md` first

Parked copy; live section in AGENTS.md.

## The full gate — original section (narratives)

Parked full-gate narratives; live rule in AGENTS.md §The full gate.

## Docs gate — original section (descriptions + narratives)

Parked docs-gate narratives; live rows in AGENTS.md §Docs gate + drift sweeps.

## cfg-gated targets — original section

Parked cfg-gated narratives; live rule in AGENTS.md §cfg-gated targets.

## required-features rows — original section

Parked required-features narratives; live rule in AGENTS.md §required-features.

## Percentile index — original section (classifier history)

Parked percentile classifier history; live rule in AGENTS.md §percentile.

## Staged-set audit + shared target dir — original sections

Parked staged-set narratives (`b2527521` incident); live rules in AGENTS.md.

## Feature Flag Discipline — rule histories (lossy surface, Report the Floor, Plan 467)

Parked rule histories (lossy-surface, Report the Floor); Plan 467 DualLeoOracle G5 FAIL, closed per riir-ai Research 322.

## Substrate-First Gate — original section

Parked pre-compaction copy; live rule in AGENTS.md §Substrate-First Gate.

## Substrate-First Gate (MANDATORY before implementing)

Parked copy; live section in AGENTS.md.

## Research Workflow — original section

Pointer to `.agents/skills/research/SKILL.md`.

## Repo count — the full original paragraph (drift history)

Count drifted 8, 18, 19 -> 16 (2026-09-04); a matching count is not a checksum. Canonical failure: riir-chain Issues 096 + 097.

## Resolved issue log (verbatim from pre-compaction AGENTS.md)

Pre-compaction resolved-issue log; compacted to one line each under the next heading.

## Issue log (resolved)

Resolved issues 719-792 condensed (e.g. Issue 792 CPU SUPPRESSED, 742 `2ae0d20a`, 739 `87dfa778`, 733 `31bf0012`); full one-liners in git history.

## Issue 744 — HRM-Text second-pass modelless extraction queue: CLOSED as resolved-negative (2026-09-11)

8 candidates closed as resolved-negative: no graduating consumer; re-file when one materializes.

## Issue 745 — Margin-gated verification escalation PoC (TriSpec distill): CLOSED as resolved-split (2026-09-11)

Resolved-split, 9 gates PASS: CASCADE refuted (3-19% false-flag), ACCEPT viable (50.5% invocation cut at 0.25% regression); `margin_gate` opt-in.

## Issue 746 — Looped-Flows modelless extraction candidates: CLOSED, split verdict (2026-09-11)

Row 2 live as opt-in `marginal_rewind` (5.71x over additive at equal budget); row 1 closed, no time-grid consumer.

## Plan 596 — sliceTCA modelless slice-rank decomposition: COMPLETE + PROMOTED (2026-09-12)

`slice_tca` Bench 714 G1-G4 pass, promoted default-on; loss 0.0263 vs 0.4835 for naive floors (18.4x).

## Plan 597 — BMR + EFE-over-models: COMPLETE + PROMOTED (2026-09-12)

`bmr` Bench 715 G1-G4 pass, promoted default-on; |delta| 7.1e-14 vs brute-force oracle, G2 513x.

## 2026-09-12 — the citation gate's first CI run was a blind red (promote `35ac604f`)

First CI run was a blind red (267 commits, exit 2 under floor 15); `issue_citation_gate.py` CI-deferred under `DOCS_GATE_CI=1`; four checks got `paths:` globs.

## Issue 755 — a quoted heading in a fence is not an allocation: CLOSED as resolved (2026-09-12)

Allocation path only delegates to the canonical scanner; `heading_allocated()` excludes fenced headings (0 of 57); unterminated fences fail safe (exit 2).

## Issue 756 — an unterminated fence swallows the rest of its file: CLOSED as resolved (2026-09-12)

19 files / 611 swallowed lines repaired same day; three shapes (missing closer, stray, missing opener).

## Issue 749 — a cross-repo `Issue N` citation rebinds to the WRONG document: CLOSED (2026-09-12)

Resolved by `scripts/issue_citation_gate.py` (`e258fdaa`); 8 rows qualified and revert-probed; 63 shared numbers reported, not gated.

## Issue 751 — the cross-repo citation sweep (18 repos katgpt-rs cannot see): CLOSED as resolved (2026-09-12)

`scripts/citation_drift_sweep.py` (`d8041de5`): 19 repos / 2,939 citations, FP 7/43 = 16%; CROSS backlog worked to 0 same day.

## Issue 752 — a repo name QUALIFIES a citation even when that repo does not own the number: CLOSED as resolved (2026-09-12)

Fix `60bc76aa`: a directory qualifier counts only if that repo allocated the number; 45 of 368 unfollowable; census 0/45 re-rated to 1/45.

## Issue 753 — the citation rules' own COSTS were recorded once and never re-measured: CLOSED as resolved (2026-09-12)

`\d{2,4}` width bound is load-bearing (`\d{1,4}` gives 51 false heads), pinned `max_single_digit = 0`; widening the 40-char alias lead buys 0 repairs.

## Issue 754 — an allocation that exists only as a HEADING is invisible to allocated(): CLOSED as resolved (2026-09-12)

`heading_allocated()` (`8e0ffaa0`) recovers heading-only allocations: 7 numbers recovered, 123 CROSS retired.

## Issue 761 — a Lean theorem can RESTATE its own definition: CLOSED as resolved (2026-09-12)

Four theorems restated their definition in riir-neuron-db (removed `24957a2`); criterion is symbolic equality over leaf constants; CROSS-DEF is not a finding.

## Issue 750 — docs_gate's CHECKS array vs the AGENTS.md table documenting it: CLOSED (2026-09-12)

Drifted (six vs seven predicates); `scripts/docs_gate_checks_sync.py` (`d10202b1`) checks membership both directions plus quantity words; `MIN_ROWS = 10`.

## Issue 760 — mmorpg-remaster joined the workspace but not `scripts/repo_set.txt`: CLOSED as resolved (2026-09-12)

Fixed `bfccffea`; 4090 reds were partial-clone topology; fence sweep caught `.plans/005_layer3_reducer.md:600` (repaired `99064c5`).

## Issue 757 — linking_fold detector Option B (the 50 ms @ n=2×1000 remainder): CLOSED as resolved (2026-09-12)

Single-pass k-NN + certified Gauss pruning: about 31x, G2b 28.28 ms <= 50 ms at n=2x1000; kept opt-in.

## Issue 747 — ASEntmax modelless mining (damping schedule, derived-k, eviction window, incremental entmax): CLOSED (2026-09-12)

Four primitives behind opt-in `asentmax_schedule` (Bench 713); real-model re-gate showed no modelless gain (sigma 0.1409 << 1); stays opt-in.

## Issue 762 — ASEntmax P4 stretch (HoldConcentration, Kamath regime detector, per-head grid, RoPE cutoff): CLOSED (2026-09-14)

T4.1 HoldConcentration + T4.2 `logit_regime` done (Bench 759); `asentmax_schedule` stays opt-in (budget-confounded); T4.3/T4.4 deferred.

## Issue 772 — config-audit first pass over katgpt-rs (7 inert/assert-only knobs + orphan-report layer): CLOSED (2026-09-14)

7 class-A/B and 12 stillborn knobs fixed same day, wire-vs-delete per finding (A1/A2 wired `3c3c52ce`; S1/B2/B3 deleted).

## Issue 777 — an instrument whose population is a FILESYSTEM walk audits code no repo owns: CLOSED (2026-09-14)

Repair: one `scripts/tracked_walk.py` (8-arm self-test); a gitignored nested repo had 1404 `.rs` credited to the outer repo; after: 8,694 tracked `.rs` over 16 repos.

## Issue 778 — `subprocess.run(..., text=True)` decodes with the SYSTEM locale: CLOSED (2026-09-14)

28 sites repaired (`encoding="utf-8", errors="replace"`); `scripts/subprocess_encoding_gate.py` scans the AST; its first run found a 28th site.

## Issue 793 (allocated as 779; renumbered per Issue 791) — the workstation sweeps' partial-clone verdict, one copy: CLOSED (2026-09-14)

Remedy `scripts/sweep_population.py` (UNREGISTERED / UNSEEN / DEFERRED); behind the reds, three live rows fixed (`25b7bf6b`, `c5c30fee`, `96041bf6`).

## Issue 792 (allocated as 776; renumbered per Issue 791) — the docs gate's CPU self-timing printed a well-formed number that measured nothing on Windows: CLOSED (2026-09-14)

Placeholder heading so the number is readable by `heading_allocated`; Windows CPU self-timing measured nothing; full record in the Issue-log entry.

## Issue 790 (2026-09-14) — an arm that exists and RUNS may still reach nothing: CLOSED (2026-09-15)

`scripts/arm_reach_audit.py` + `arm_reach_gate.py`: nine findings in green instruments (796 mutants via bare-dict exec); `--include-all` 55 of 55.

## Issue 791 (2026-09-15) — three numbers allocated twice across a 57-commit divergence: CLOSED

Two sessions allocated 776/779/780; recovery via `removed_candidates()` from `git log -M --diff-filter=D`; all three moved by citation weight.

## Issue 797 (allocated as 796; renumbered — the other session allocated 796 the same hour and pushed first) (2026-09-15) — a sweep reads the WORKTREE, so a finding may exist in NO commit: CLOSED

Last CROSS row was an uncommitted sibling edit's artifact; `scripts/worktree_state.py` wired into all 16 sweeps (advisory); membership gate `sweep_advisory_membership_gate.py`.

## Issue 795 (2026-09-15) — 70 numbering collisions the gate could not see, 9 of them live: CLOSED

70 numbering collisions (9 live) invisible to the gate; two regimes: wall by membership at >= 700, ratchet below; six of nine not renumbered.

## Issue 796 — the allocation-time dual-allocation gate: the FP rate measured, the gate built CLASSIFIED: RESOLVED (2026-09-15)

`dual_allocation_fp_probe.py`: 7202 one-sided pairs green, 39 incidents = 31 TWIN + 8 INDEPENDENT; classified `dual_allocation_gate.py` joined CHECKS.

## Issue 794 (allocated as 780; renumbered per Issue 791) — a wrong address reads as UNDECIDED when its number is in local range: CLOSED (2026-09-14)

New class `MISATTRIBUTED-IN-RANGE` walled at 0 (the check only ran in the CROSS bucket); 19 of 19 locally-allocated rows were false.

## Issue 781 — the heading oracle matches one house STYLE: CLOSED as a measured, printed blind spot (2026-09-14)

Closed as a measured, printed blind spot (64 of 152 read by house style); widening is unsound, cost prints every run.

## The x86_64-pc-windows-msvc axis, measured: clean at all-features/all-targets (2026-09-14)

x86_64-pc-windows-msvc all-features/all-targets: exit 0, 32 packages, zero code findings; one cell added, matrix not closed.

## Issue 782 — a pinned repo absent from the walk is never visited: CLOSED (2026-09-14)

`cfg_row_implication_drift_sweep` had no absence check (green over 16 of 20); all eleven share the verdict; per-sweep `DROPPED` check added.

## The executed-test gate on x86_64-pc-windows-msvc: exact floors, all three rows (2026-09-14)

`scripts/test_gate.sh` on Windows: 203/203, 2041/2041, 249/249 exact; 477 integration and 176 bench targets still unexecuted automatically.

## Issue 783 — the subprocess-encoding gate is katgpt-rs-only: CLOSED (2026-09-14)

Gate lacked a sweep half: 29 DECODE + 2 CHILD-ENCODER over 5 repos, all 31 repaired; ceilings walled 0/0.

## Issue 784 — the orphaned-attr gate's cross-repo claim was hand-run: CLOSED (2026-09-14)

No new offenders but a stale warrant (hand-typed totals); `orphaned_attr_drift_sweep.py` now measures the total.

## Issue 785 — the wasm32 surface audit had no verdict half: CLOSED (2026-09-14)

UNRESOLVED walled at 0; UNCOVERED pinned by membership (`scripts/wasm32_uncovered_expected.txt`); per-repo floors plus a TOTALS row.

## Issue 800 Arm C complete — GraphStablePool site re-points: 1 landed, 1 N.A., 2 declined on evidence (2026-09-16)

Site 1 radix_prefix re-pointed (`bdb1091a4`); PagedKVCache + BranchBank declined on evidence; Qwen38LaneSet N.A.

## The full gate's wasm32 layer counted NEGATIVE cfgs as surface — derivation fixed positive-only (2026-09-16)

Fix `448c77f91`: count only compile-time non-negated cfgs; first full-gate PASS on this box.

## Issue 844 (2026-09-19) — the dot-delegation crossover measured on BOTH arches; the NEON answer refuted the filing session's own expectation: CLOSED

M3 NEON: no crossover, delegation wins at every length >= 4 (9.06x @256); dispositions filed as riir-ai Issue 982, riir-train Issue 562, riir-neuron-db Issue 621.

## Issue 849 (2026-09-19) — the 844 per-site dot read, katgpt-rs-own sites: four delegations landed + the full repair record-back: CLOSED

Four delegations to `simd_dot_f32` landed; sibling record-back riir-train `3cf49ebb`, riir-neuron-db `5d86dd3`, riir-ai `dbc5639d4`; ~3e-6 @64 passed all gates.

## Issue 848 (2026-09-19) — a rename privatised a delegation target and a rework deleted a shared fixture; both of that module's EXTERNAL callers are docs-gate CHECKS: CLOSED

`6c6ca2ee` privatised `is_checkout`, develop red 6h40m; added `cross_module_attr_gate.py` (T2) and `import_health_gate.py` (T3).

## Issue 856 (2026-09-19) — the green zero has TWO spellings and the audit built for it saw one: CLOSED

Whole-body `#[cfg(feature)] mod tests {}` zeroes like inner `#![cfg]`: 26 targets, 175 assertions, 7 `*_goat`; predicate widened to whole-body.

## Issue 860 (2026-09-20) — `successor_density_critic`: tabular discounted count-ratio goal-critic: CLOSED

Modelless CRL extraction `2c7a1f157`, Bench 818 GOAT pass; consumer pull satisfied by riir-ai Issue 991 `goal_salience`; stays opt-in.

## The exact_sigmoid / dot_f32_ordered substrate promotion (2026-09-20) — the riir-chain Issue 156 T1 landing executed in this repo

Ungated `exact_sigmoid` (2 ULP) and `dot_f32_ordered` landed (`5458dd69b`, main `5e2b730f2`); the Cephes speed claim inverts on aarch64 (1.7 vs 3.1 ns).

## Issue 865 T3 (2026-09-21) — the probe_guidance λ-sweep GOAT gate ran NEGATIVE; the negative verdict is the pinned gate: OPEN (lane re-opens at Bonsai scale)

Guided best lambda=1.25 dominated by T=1.0; trained probe loses to the zero-logit null at every lambda >= 1.5; negative pinned as the gate; lane re-opens at Bonsai scale.

## Issue 866 (2026-09-22) — KARC D3 promotion coverage audit: VERDICT QUALIFY — the contract's passing legs live on configs nobody constructs: CLOSED

Bench 849: passing legs are ChebyshevBasis only, deployed FourierBasis R=1 fails both bars; verdict QUALIFY (not DEMOTE).

## Issue 865 file hygiene (2026-09-22) — issue file removed per noise-reduction, the lane's record was already durable: CLOSED (hygiene)

Issue file removed with no content change; record durable elsewhere (catalog section 120, Research 578, G2a/G2b pins).

## Issue 865 follow-up (2026-09-22) — arm (b) unblocked (DropoutHeadProbe) + the headroom study: the negative EXTENDED to every mini-lane regime (Bench 850)

`DropoutHeadProbe` (Bench 850) never beats the null across three regimes (-0.13 to -1.42 pts); a +2.7 pts positive retracted; only Bonsai-scale lane remains.

## Issue 876 (2026-09-23) — the flappy render widened to v3; the decoded arm reads Δ0 vs the structured arm: CLOSED

Grammar v3 `laya-flappy-v3`: structured 96/100 = decoded 96/100 (delta 0), raw ordinals 51/100; `515230244`, Bench 882.

## Issue 879 (2026-09-24) — MAttr budget primitives landed opt-in: exact mass at router-regime cost; calibrated-mass column says what the hard cut cannot: CLOSED

`exact_mass_admit_into` + `log_frontier` (Bench 884): |sum m - k| ~1e-8 vs the hard cut's -6.4% drift, cheaper than the sigmoid gate at N=1e3.

## Issue 892 (2026-09-25) — tetris strategy RULEBOOK (ruliology surface) + chance-node PUCT + laya head-to-head: the hybrid FSM champion out-scores laya 92–299× and Bench 891 4×: CLOSED

16-rule rulebook + Build/Downstack/Survive FSM; hybrid `68cae9d382014662` scores 4.0-4.7x points, laya 0/60 vs hybrid 60/60; hybrid promoted.

## Issue 893 (2026-09-25) — the tetris substrate promoted from `examples/common/` into the leaf crate `katgpt-tetris`: CLOSED

Leaf crate `katgpt-tetris` (rayon/blake3/fastrand); byte-identical migration (fixture 1,113,251 / 1,113,238 B, 14/14 tests); champion pinned `68cae9d382014662`.

## Issue 919 T1 (2026-10-06) — the weight-derived spike census landed (Research 605)

`spike_census.rs` offline GGUF scan landed (Research 605); self-test caught an f16 subnormal off-by-one; Bonsai-8B clean, 27B soft, gemma-2-2b weak; nothing promoted.

## Issue 927 (2026-10-09) — the SplitMix64 twin cross-pin: RESOLVED (file removed per noise-reduction)

Cross-pin landed `36823de3f`: `splitmix64_twin_pin.rs` pins both streams bit-identical (6 seeds x 10k); delegate option declined (zero-dep posture).

## Issue 941 (2026-10-16) — the token/latent position-provenance convention lifted into the swir host contract: LANDED (file removed per noise-reduction)

`swir/position_provenance.rs` (`PositionProvenance`, `ProvenanceLedger`) landed under opt-in `swir_switch_thinking`; 6 contract tests, 51/51 swir tests green.

## Issue 942 (2026-10-16) — the latent pseudo-text readout + health sensors (report-only instrument): CLOSED (file removed per noise-reduction)

`swir/pseudo_text.rs` report-only readout + health sensors landed `76d6d4fef` by shikuwa-4090-agent; G1/G2/G4 gates; abstention leg out of scope.

## Issue 929 (2026-10-18) — DiscoLoop loop-alignment probe + decode-reinject PoC: CLOSED ADJUDICATED (file removed per noise-reduction)

Probe confirmed (Bench 942: auroc_margin peak 0.9327 @loop2 ood); decode-reinject closed negative (Bench 943: acc 0.0778 vs stock 0.3667); riir-infer `c43d36b` flipped default phrase; T2.x deferred.

## Issue 939 (2026-10-18) — phase-carrying belief kernel + constructed phase-code counters: LANDED (P1/P2/P4 `97a4c0ace`), P3 deferred-until-pull (file removed per noise-reduction)

`phase_belief.rs` opt-in landed (P1/P2/P4 `97a4c0ace`; 11 gates): ModularPhaseCounter + DampedPhaseBlock; P3 FSM real-boot gate deferred until a consumer pull.

## Issue 945 (2026-10-18) — QRR quotient-convergence extraction (rate-aware halting, settling budgets): LANDED P1/P2/P3/P5 `f112eecaa` + P6 `8e164b339`, P4 deferred-until-pull (file removed per noise-reduction)

Landed `qrr_convergence` (ModeFitter, mode_verdict, advisory settling budgets; 14 gates) f112eecaa + P6 Sylvester report 8e164b339; P4 deferred-until-pull.
fp32 rotation law: f32 drifts ~1.0e-3 over 1000 steps vs 4.0 for the bf16 counterfactual.

## Issue 946 (2026-10-18) — formatting policy: ADOPTED + GATED (file removed per noise-reduction)

Adopted + gated: format commit `6760d76f5` (131 cold files; 158 of 2,770 .rs divergent, not 832), `scripts/fmt_gate.py` as full_gate Layer 7.
Tuned rustfmt.toml hope refuted by the residual curve.

## Issue 947 (2026-10-18) — katgpt-core lib.rs over the 3,200-line ceiling: LANDED `474a9bfc4` (file removed per noise-reduction)

lib.rs 3,894 to 1,853 lines via `474a9bfc4`: 252 provenance blocks moved into module `//!` headers.
Zero public-surface change (non-comment lines byte-identical to HEAD).

## Issue 960 (2026-10-24) — KV-quant behavioral-subspace diagnostic + sensitivity-guided bit allocation (modelless PoC): LANDED T0a/T0b/T1/T2, T3 design closed (file removed per noise-reduction)

Landed T0a `ec7d3e2af` (kv_sensitivity diagnostic) + T0b/T1 `3b5bc8faf` (Bench 954, BitTable allocation, G1 nested 0.840); T3 design closed `08942dfe3`/`4a5347905`.
T3 eval deferred to the quiet-box lane; full record via `git show 4a5347905`.

## Issue 961 (2026-10-25) — FlashMemory row-relative margin-to-max selection (`flashmemory_margin_select`): LANDED opt-in `c6fb48962`, forward-level G1 FAIL, NOT promoted (file removed per noise-reduction)

Landed opt-in `c6fb48962` (`with_margin(delta)`); forward-level G1 FAIL at matched retention (riir-train Bench 669), NOT promoted.
Stays an evaluation instrument; recall at cum-90% mass is not a sufficient selector G1.

## Issue 962 (2026-10-25) — FlashMemory budget (top-ρ) block selection (`flashmemory_budget_select`): LANDED opt-in + the serving consumer via riir-infer 1117 (file removed per noise-reduction)

Landed opt-in `with_budget(rho)` T1 `aa439587c`; riir-train Bench 671 128K: top-rho beats Quest and window rules and the absolute rule at lower retention.
Consumer via riir-infer Issue 1117 (`51f52f4`, `3d3152fbd`, riir-infer Bench 893 v2 `47268ea`); default stays Absolute.

## Issue 964 (2026-10-25) — katgpt-assign solver static-invariant precompute (bbtree-class audit): LANDED ungated, GOAT G1–G4 PASS (file removed per noise-reduction)

Landed ungated: precomputed `group_members` and `heat_order` at compile/new, GOAT G1-G4 PASS.
Two bonus G4 findings (e.g. `leaf_deltas` under-sizing) caught by new gate tests and fixed.

## Issue 966 (2026-10-27 → closed negative 2026-10-28) — modelless prefix-first grid re-selection (`grid_reselect` λ sweep): G1 FAIL honest-negative, the keyed incumbent owns the available win (file removed per noise-reduction)

Closed negative: G1 FAIL, `grid_reselect` lambda sweep (plan 632, 4090 `2bc7c691b`); code +5.26% exactly tied with the keyed incumbent, prose +0.00%.
Substrate stays opt-in; `TopKPoolCoverage` instrument kept; tables in riir-ai Bench 1013.

## Issue 970 (2026-10-28/29) — the 963 floor bump hard-broke every downstream pin still reading 1.98.1: measured matrix + the T3-generalized rollout ledger (file removed per noise-reduction; the rollout is COMPLETE, 2026-10-29)

Anchor bump `8d54ddb96` (rust-version 1.99.0) broke downstream 1.98.1 pins; rollout ledger COMPLETE 2026-10-29, last row riir-rethink `24bf40c`.
Other rows landed per repo (e.g. riir-chain `2261f07a`, riir-train `1fea63cf`); riir-esp32 untouched (pins branch main).

## Issue 969 (2026-10-27) — mmorpg-editor was seal-game-editor: one repo double-registered under its codename and its on-disk name; codename row retired (filed + closed in one commit, file never kept per noise-reduction)

mmorpg-editor was seal-game-editor: double-registered repo; codename row retired from repo_set and 21 pin files (`39762d55f` was the original registration).
Repo count 33 to 32; codename kept as citation leniency; filed and closed in one commit.

## Issue 968 (2026-10-27) — `block` 0.1.6 future-incompat killed by a vendored `[patch.crates-io]` fork in every workspace that reaches it (file removed per noise-reduction)

`block` 0.1.6 uninhabited-static future-incompat fixed by vendored fork `vendor/block` (`5e216a761`) via `[patch.crates-io]`; no upstream bump exists.
Gates: backend gpu_inference build + clippy, ObjC-block liveness arm.

## Issue 975 (2026-10-31) — `step_noise` modelless thrash monitor + DDR gate PoC: G1–G4 PASS, T5 consumers filed (file removed per noise-reduction)

The full PoC arc (T1–T5): `349e07ee3` (T1 module skeleton — EwmaCom lag law, half_split_dispersion labeled proxy, StepNoiseMonitor 4 streaming EMAs zero-alloc, Regime/Remedy, ddr/min_ddr_at INF law, guided_width interop), `48fd0be4d` (T2–T4 — six-archetype fixture classification ALL PASS on DEFAULT calibration, synthetic + DualPoolBandit causal toggles Learning→Thrash→Learning, 28.3 ns/observe at N=8 vs the ≤100 ns bar, G4 zero alloc; PROVENANCE powermode=2(high)). Substrate-first table + pre-registered mappings: this entry's git history. Consumer follow-ups filed: riir-ai `1143` (engine bridge third verdict row), riir-refine `190` (elo_k_eff + score-bench readout), riir-dao `009` (advisory evidence_scaled_budget); katgpt-sense deferred (not checked out). Model-based twin: riir-train Plan 479 (Phases 1+2 landed `7ca94b18` + `42707a54`). Feature `step_noise` stays opt-in (GOAT-tier integration, not Super-GOAT — Research 485's verdict); consumer promotion rides each filed issue's own gates.

Follow-up consumer-routing pass (2026-10-31, owner-directed, verdict-gated): the instinct/rethink family. Filed: riir-rethink `047` (ESC escalation VoI readout — standalone counters/ratios, motivated-by, NOT consuming step_noise: the ESC gate is exact/stationary, so churn×slope was retracted under verdict as a category error; hard cross-constraint on rethink Issue 046 keeping the cheap answer on escalated questions), riir-instinct `025` (OWNER-GATED `EscalateSpec` escalation-precision-floor/damp-field grammar question, sequenced after rethink 047), riir-instinct `026` (LRU evict/reload hysteresis — `INSTINCT_LRU_MIN_IDLE_SECS`, default 0 byte-identical, overshoot-capped, motivated-by not consuming). Both instinct/rethink filings re-scoped/corrected under their verdict: the ESC family has no adaptive loop to protect, so 047/025/026 consume nothing from this PoC — recorded here so the consumer ledger stays complete.

## Lessons

Distilled lessons (instrument generalisation, census blindness, green-zero, box state, concurrency, attribution); details in git history.
