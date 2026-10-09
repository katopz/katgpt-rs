Compacted 2026-10-06: every `##` heading kept verbatim, one compact entry per
record (≤2 short lines); the full pre-compaction text is in git history.
Operational rules live in `AGENTS.md`; removed issue files: git history.

## Issue 928 (2026-10-09 → closed 2026-10-09) — cross-repo drift-sweep debt: seal-std registration census + 4-wave repair: CLOSED — console/locale/instrument sweeps ALL GREEN family-wide (file removed per noise-reduction)

The seal-std registration census (`c007cf543`) surfaced committed sibling drift
across the contract set; four same-day waves closed it. W1 (katgpt-rs):
instrument floors re-measured, the load-bearing green-zero test targets armed.
W2 (4 quiet repos): riir-shader `1f2a195`, riir-refine `c126f441` (the CRLF lesson
— EOL-preserving repairs), riir-train `45fc7799`, riir-reflex `00b4ee1` —
80 consoles + 35 locale sites. W3 (seal-remake `b69ee61`): 10 CROSS citations
qualified, hot-repo instrument rows re-pinned at stable HEADs (riir-ai's
artifact_boundary.py = a real gate findable from NO root, wiring stays that
repo's owners' call), len_derived stability pin class landed (`8bc96908c`,
canaries 15/15). W4 (sessions ended): riir-ai `dc37d0693` + riir-rethink
`25074d7` — the last 7 consoles + 6 locale sites. End state: console 252/252
defended over 28 repos, locale 0/1345, instrument exit 0 — the drift family
fully defended workspace-wide. Same session (wave 4 tail): the elementwise_cubecl
shape-parameterization ROOT FIX landed (riir-infer `02e7a65` — sigmoid/silu bound
by the EXPLICIT params[0]=n, the copy_f32_grid2d idiom, oversized-binding sentinel
test; both stability rows retired at katgpt-rs `d956bf716`, canary 15/15), riding
the repair of the vendored wgpu-hal metal adapter's u64/usize E0308 (which had
broken every macOS cubecl_runtime build since 3564a72 — the metal arm never
compiled on the author's CUDA lane). Successors: Issues 920/926/929 (quiet-box
lanes).

## Issue 922 (2026-10-07 → closed 2026-10-07) — loop-straightness monitor family, the DEC-native stalemate signal (LiFT modelless residue): CLOSED — primitive half landed, GOAT ALL PASS (file removed per noise-reduction)

Landed same-day: katgpt-dec `loop_straightness` (opt-in; katgpt-core + root
forwards) — `path_action` (Σ‖Δ_k‖²/δ_k, chord², Σδ in one pass) +
`action_efficiency` (η = chord²/(Σ·Σδ) ∈ (0,1]; 1 iff straight at constant
grid-speed, 1−η the exact wasted-motion fraction). Signal-diff executed
pre-implementation: all four named cousins are counters/entropy/classifiers,
none geometric — recorded in the module doc + Bench 922. GOAT (Bench 922):
G1 8 tests — the 500-walk nonuniform-grid floor sweep caught the landing's
OWN formula bug (first cut multiplied by Σδ, η=1.38 > 1; uniform fixtures
mask that class — recorded as the lesson); G2 38.7 ns @ K=8 dim=4, 26× under
bar, K=64 dim=32 disclosed 1,137 ns / 29.4× scaling; G3 clippy both postures
(denied `needless_range_loop` + `redundant_slicing` caught pre-landing);
G4 0 allocs/1000. Row `katgpt-dec:257:loop_straightness`; counts 674→675;
count_features green (675=675). Consumer half = riir-ai Issue 1037 (each
row owes its own signal-diff); promotion waits on a production consumer.

## Issue 924 (2026-10-07 → closed 2026-10-07) — citation_weight's dialects are case-sensitive; lowercase `plan N` prose is invisible to duplicate adjudication: CLOSED — both instruments repaired, oracle confirmed (file removed per noise-reduction)

Landed `b98c340dc`: `citation_re()` scopes `(?i:…)` over long-form prefixes
(bare letters stay uppercase — lowercase `p<NNN>`/`b<NNN>` measured as
identifier noise, 68/80 sites in riir-ai); measured counts recorded in the
DIALECTS comment (seal-game-editor lowercase-DOMINANT: `plan` 1,119 vs 591,
issue 719 vs 228; riir-ai 406/340; katgpt-rs 224/146). Family sweep:
`issue_citation_gate.py` carried the same class — `_HEAD`/`_HEAD_1D`/`_TAIL_1D`
+ the trail re-search got the same scoped `(?i:)`, `citations()` normalizes to
the canonical KIND (scanned 475→476: the previously-invisible `issue 739` in
AGENTS.md); `numbering_drift_sweep` (no dialect regex) and `citation_drift_sweep`
(by-name stems) adjudicated NOT the class; 163 lowercase heads across sibling
AGENTS/HISTORY docs reported, none gated (this gate scans its own repo's pinned
docs). riir-kat HISTORY: three unpadded single-digit citations re-spelled
`issue 003` (width-bound readability, that
file's own zero-pad convention). Oracle re-run: sge 241 sites 4→15 (verdict
WEAK → `241_glb_delivery_lane` +5 over 9 decided), 272 3→12; all 4 hand-read
lowercase sites visible + attributable, PLUS ~10 `.rs`-comment sites the
hand-read's `.md`-only pass missed (`tools/neuron-publish/src/*`, `regate.py`).
Validation: citation_weight + gate selftests PASS, numbering_gate PASS,
numbering_drift_sweep PASS, docs_gate 35/35 (86.3s wall).

## Issue 923 (2026-10-07 → closed 2026-10-07) — escalation_guard, the shared serving-escalation guard primitive (riir-refine Plan 202 R1): CLOSED — all tasks landed, cross-recorded (file removed per noise-reduction)

Landed `6118539c7`: katgpt-core `escalation_guard` (opt-in) — `RollingRateLatch`
(both-bounds + cost-ceiling-only shapes, rate() None until full, sticky demotion,
O(1) alloc-free observe), `demote_only_decode` + env helper, ns-parameterized
receipt tiers (`ns="ESC"` reproduces rethink's bytes exactly),
`rate_bounds_valid`. GOAT (Bench 923): G1 17 tests, G2 1.74 ns/observe
(ceiling 20), G3 default surface unchanged (2141 default lib count), G4
0 allocs/1000 observes. Test-gate row `katgpt-core:2158:escalation_guard`.
Consumer halves cross-recorded in riir-refine Plan 202 R1 (refine manifest half
+ instinct consumption `9e1fc37` — `EscalateSpec::validate` consumes the ONE
`rate_bounds_valid` definition). The tpr/ugc private kill-switch copies STAY
(default-on surfaces an opt-in feature cannot serve; truth tables cross-pinned
both sides). Remaining follow-up lives in riir-refine Plan 202 R1's
"Remaining" block: rethink's migration onto the shared latch via instinct's
pub surface — deliberately its own unit.

## Issue 921 (2026-10-07 → closed 2026-10-07) — citation oracle two-lane self-reference blind spot: CLOSED — Arm B landed, corpus repaired, sweep green (file removed per noise-reduction)

riir-infer's nine IN-LOCAL-RANGE rows were narrated-only local-lane numbers
(004–009 never existed as files on any ref, either box). Owner verdict: Arm B
(two-lane gap via `.highwater_local`) LANDED; Arm A (own-name heading variant)
and Arm C (accept the red) REJECTED by verdict. Hash-pinned: corpus repair —
`f0191e3` (qualify `riir-ai Issue 980`) + `71c8eed` (six self-allocation
headings rewritten into the readable date-led grammar, AGENTS.md grammar rule
pinned) in riir-infer; Arm B `3249672d1` here (`citation_drift_sweep.py`
two-lane gap boundary + two-sided `--prove-fires`; ILR 9→0, novel 3→0,
`max_in_local_range: 0` pin holds; gap rows carry a `[two-lane gap …]` note;
`.highwater_local` rides both advisory patterns). Normative design lives in
AGENTS.md (§the heading oracle / the two-lane gap). The two ghost proposal
citations (numbers 17 and 13 — rows in riir-dapps / riir-instinct docs naming
proposals owned elsewhere; full rows in this file's git-recovered 921 record)
remain owner-gated.

## Issue 919 (2026-10-06 → closed 2026-10-07) — spike census + delimiter-sink PoC: CLOSED — census channel key REFUTED; measured-diagonal KV exemption + delimiter folding both NO-GO (file removed per noise-reduction)

Filed from Research 605 (arXiv:2603.05498). Nothing promoted; three
hash-pinned verdicts:

- **T1 census** (`ff2f40e64`): weight-derived FFN spike census over four
  packs — Bonsai-8B matches_table1_shape, gemma-2-2b weak (sandwich+QKNorm
  suppression confirmed on an untested model); self-test caught the asentmax
  f16 subnormal off-by-one. Sidecars + blake3 pins:
  `tests/fixtures/spike_census/`.
- **T2 channel screen REFUTED** (`eccb13b9c`): census top-K vs measured
  channel maxima P@4 = 0 on both cells — the stat screens big weight-rows,
  not activation-carrying channels; block-level Spearman +0.48 survives.
- **T3 measured-diagonal KV exemption NO-GO** (cell 1 `0952590e1` /
  riir-infer `47efae7`; decisive cell 2 `3c7fe7709` / riir-infer `9ebf3e2` +
  `d91971b`): gemma G-MAIN PASS but thin (+0.0008 ppl vs seeded-random
  equal-budget); Bonsai-27B G-MAIN NEGATIVE, G-487 Δ −0.0011 — **QK-norm
  squashes per-channel scale upstream of the cache, so the 487 Q8KV-gap
  premise does not apply on QK-normed archs.** Same lane fixed the
  riir-infer `QuantizedKvMirror` multi-layer refresh bug the G0 control
  exposed (`47efae7`; Issue-013 absolute-PPL caveat recorded).
- **T4 delimiter-sink constancy NEGATIVE** (`3b800c932` / riir-infer
  `b473196`): pre-RoPE K is context-dominated, not constant — pooled cos
  0.80–0.89, worst per-cell median 0.37–0.50 vs the pre-registered 0.99
  bar; V worse everywhere. Conditional T5 never opened.
- **T6** satisfied as a pointer: the census artifact stays supplied to
  riir-train Issue 614 T4 (the merge-drift consumer). The FFN-side surface
  (down_proj inputs, the 140–581× class) is deliberately UNFILED — backlog
  only, if ever re-opened.

## Issue 906 (2026-10-05) — owner-gate pickup CLOSED: E12 Gemma base-model check RETIRED (option b); E13 rust-version pin scoped to publish=false only (file removed per noise-reduction)

Claude verdict ruled every row. E13: `rust-version.workspace` removed from the 8
crates.io-shipping manifests (landing `880619731` had over-scoped), kept on the 25
`publish = false` packages; E12 retired on its standing negative (12/12 harmful
recirc cells; record `.plans/613_e12_gemma_base_model_options.md`).

## Issue 918 (2026-10-05) — posterior (truncated-Gumbel) inverse sampler, max-first: CLOSED (file removed per noise-reduction)

`keyed_posterior_gumbel_noise` (default-on `ac_prefix`, Plan-614 substrate, commit
`9ad5e21ce`-era lane) — exact posterior noise given a realized pick; Bench 918 ALL
GATES PASS (G1 argmax identity + KS joint law 0.0078–0.0135 while the biased
construction fails; G2 2.08×; G4 0 allocs). Consumer: riir-train Plan 437 GSF
training coupling.

## Issue 916 (2026-10-03 → closed 2026-10-03) — two_fidelity_bai G2 cost debt: CLOSED — δ-correct baseline (a) + lazy-discharge resolver (b) (file removed per noise-reduction)

(a) baseline rebuilt as the LUCB-MCTS/UGapE class (`3aa27c149`, cap `321a4a7c6`) —
GOAT PASS (Bench 615); (b) the ~10–50× cost attributed structurally (trace
`a3b3d435c`), resolved by opt-in `SearchConfig::lazy_discharge` (`ca9fb1617`,
ported from the authors' `twoffs.py`) — 236–1117× below v1, knob off
bit-identically.

## Issue 915 (2026-10-02 → closed 2026-10-03) — two_fidelity_bai certified-interval invalidity: CLOSED — root cause was the BENCH FIXTURE, not the module (file removed per noise-reduction)

The fixture's slow oracle violated its own `MinimaxSpace` contract (`mu` filled only
at leaves); the module's containment induction was sound. Fix: `slow_sample` draws
around `vstar[node]` + a FIXTURE-honesty gate arm; re-arm 8/8 certified, 0
guard-fires; the next finding filed as Issue 916.

## Issue 914 (2026-10-01 → closed 2026-10-02) — Pseudo-head mixing runtime (IHA distill, Research 600): CLOSED NEGATIVE before any implementation (file removed per noise-reduction)

Blocked-on riir-train Issue 606 evidence, which killed it: Bench 621 (`e6b94720`)
arm 5 catastrophic on both tasks (ternary residual == the token-only-majority bound
exactly); no trained-with-mixing consumer exists ⇒ the pre-registered close clause
fired. Arm 4 stays a training-side question.

## Issue 913 (2026-10-01) — Grouped-evidence noise-weighting primitives (riir-train Research 463, EasyPPO modelless half): CLOSED (T1–T4 landed opt-in; file removed per noise-reduction)

Four primitives in `crates/katgpt-core/src/grouped_evidence.rs` behind
`grouped_evidence_weighting` (`9ad5e21ce`, Bench 905); T3 shipped as a ranking score
(coverage 0.883 at nominal 0.95 — no UQ claim, the conformal floor binds any
consumer that claims coverage). Consumer: riir-clippy Issue 139 (`e5b4d56d`); kill
clock 2026-10-31.

## Issue 912 (2026-10-01) — Dirichlet-distribution primitives: sampler repair + exact explore-dial sampling + Dirichlet-EMA (Research 596): CLOSED (T1–T5 landed opt-in; file removed per noise-reduction)

T1 `7745cfb9c` repaired `sample_dirichlet_into` to honor α (α=1 path kept verbatim);
T2–T5 `9b98ee3dc` (Bench 904) `dirichlet_dist.rs`: `sample_conc_into`, log-space
`thinning_into` (G2 183 ns @ N=8), `DirichletEma` with the drawn-variance correction
`C = ε(1−β^M)`. Feature `dirichlet_dist`, no coverage claim.

## Plan 612 (2026-09-30) — PISA pyramid Top-K + LSE block selection: CLOSED with a G2 iso-quality NEGATIVE (latency slope CONFIRMED; two Phase-1 defects caught by the gate's pins, fixed with regressions)

The gate caught two real Phase-1 defects BEFORE any number was quoted. Bench 612 on
real qwen38 FA captures: pyramid_lse −0.133 Recall@8 at L ≥ 32K vs exact-LSE —
`pyramid_topk` STAYS OPT-IN; aggregate latency slope 1.07 vs 1.99 confirmed;
riir-train Issue 586 + riir-ai Issue 1017 consume the harness/ladder.

## Issue 898 (2026-09-30) — KL effective depth exit calibration for the looped runtime: CLOSED (instrument landed, G1 PASS on all three fixtures; NO Config default justified — calibrated exits are checkpoint-dependent; the contingent BO arm deferred with its unblock recorded here)

`katgpt_core::loop_depth_probe` (`aa3a7d382`, Bench 899) — G1 holdout PASS; the
hand-tuned defaults STAND (calibrated exits are checkpoint-dependent); en-route
fixed a latent Issue-717 `LoopDeepStats::clear()` defect (twin filed Issue 900). BO
arm blocked on a BO-trained checkpoint (riir-train `.plans/421` Phase 4, `9560ce0a3`).

## Issue 911 (2026-09-30) — ties stopped at their source: the resolution-aware `w` floor + the zero-tolerance AUC guard: CLOSED (verdict round 2; the SE tolerance retired, the "preserved by construction" docs made f32-honest; companion reflex promotion landed in the same arc)

`resolution_w_floor` (2-ulp floor; clamped candidates re-solve intercepts before the
loss-argmin) + a zero-tolerance AUC guard; near-constant fits keep every distinct
score at measured ≤ ~1e-5 clamp cost. Companion: reflex `--gate-fit-calibrated`
PROMOTED default (closes reflex Issue 056).

## Issue 910 (2026-09-30) — the Platt solver needs Lin–Lin–Weng, not a candidate fallback as the primary repair: CLOSED (LLW solve + the AUC ranking guard; the round-1 review's reference optima reproduced EXACTLY; held-out floors + a 48-cell property sweep pin the behavior)

`solve_smoothed_mle` is the LLW form (base-rate init + Armijo backtracking); the f64
reference reproduces the production optima EXACTLY (banking77 8.6736/71.5657 etc.).
Benches 807 4/4 + 808 GOAT PASS; `COMMITMENT_VERSION` 1→2.

## Issue 909 (2026-09-30) — `sigmoid_calibration` narrow-window degenerate stall + f32 tie-collapse: CLOSED (three-candidate loss-argmin fallback + output-side saturation guard; the audit refuted the early-break theory; reflex Issue 056's AUC regression confirmed fixed end-to-end)

Fixtures are the real production windows (`RIIR_DEBUG_CAL_WINDOW`); the recorded
early-break theory overstated (corrected by Issue 910: root cause is Platt's
undamped full Newton step from identity). Repair: three-candidate loss-argmin +
`window_keeps_resolution` guard; Bench 808 re-run PASS.

## Issue 908 (2026-09-29) — Plan 612 T2.1 FA-layer Q/K capture lane: CLOSED (qwen3.8-27B captured at all four lengths; Bonsai PRIMARY a documented loader blocker; riir-infer `6cb354d`+`836d28e`)

The `qwen38_pyramid_capture` bin + Q-tap/K-download seams (riir-infer-gpu); 50
committed fixture bins + `manifest.json` in
`crates/katgpt-attn/tests/fixtures/pyramid_612/` (~65 MB). Bonsai blocked:
`load_weights_gpu` refuses Q2_0 (`expected Q4_K/Q6_K`) — the error is the record.

## Issue 907 (2026-09-29) — KVarN V-row bit arms are not a pure bit ladder on real rows: CLOSED (cache-level reproduction + crossed-config attribution on REAL captured rows — the 3-bit var-norm arm is the defect site; consumer non-interpolation rule in kvarn docs; commit `2c93727be`)

Bench 903: the b2 < b4 < b3 inversion reproduces at cache level on real rows
(riir-infer `vrow_capture`); crossed-config sweep attributes by ELIMINATION — the
3-bit var-norm arm is the defect site. Rule landed in kvarn docs: never interpolate
V-row quality across KVarN bit arms.

## Issue 904 (2026-09-28) — `fit_woodbury` f32 sample-Gram Cholesky panic bypassed `tick_karc`'s warn-and-keep: CLOSED (fit paths degrade to `FitError::Singular`; commit `b8913aaa5`)

`try_cholesky_f32`/`try_ridge_solve_woodbury_f32` return `Result` (the error carries
pivot + tol); `fit_woodbury` maps to the append-only `FitError::Singular = 2` and
commits `Wout` only on success. Found by the riir-ai Proposal 048 T8 GOAT's first
run; panicking forms stay for callers that want the assert.

## Issue 903 (2026-09-27) — `row_logit_floor` promotion lane — all four boxes PASS (Bench 902); the model-bound G1 was closed by riir-infer Bench 003; the `ForwardContext.logit_floor` default flip stays the owner act: CLOSED (staged, not deferred; commits `fdca94d7a` + `93740fe21`)

riir-infer Bench 003 closed the model-bound G1 (6-bit the admissibility floor; the
sink exemption is LOAD-BEARING — a ppl-only gate would have promoted the defect);
katgpt-rs Bench 902 per-family walk + full-forward G2 A/B pass, 8-bit the width of
record. The default flip is a TWO-PART owner act (first riir-infer default feature +
the cfg field), staged with its verdict.

## Issue 902 (2026-09-26) — md-only contract repos can never pass `len_derived`/`shared_temp_path` (the delegated `.rs` floor demanded non-zero): CLOSED (measured zero-row acceptance; commit `26ba6afdd`)

Shared predicate `sweep_population.zero_walk_floor_accepted`: the zero delegated row
is accepted only while `tracked_walk` measures ZERO tracked `.rs` — re-measured
every run, the acceptance PRINTED. The measurement caught riir-reflexer's stale
md-only premise (16 tracked `.rs`) and re-addressed 4 gemv EYES rows to riir-infer.

## Issue 900 (2026-09-26) — issue-717 damping G4 failed at the wider feature combo: the gain_cost_halt probe buffers allocated unconditionally per call: CLOSED (lazy-sized; the measured region is combo-stable; commit `e272057be`)

Two step-direction scratch `Vec`s were sized whenever `gain_cost_halt` was COMPILED
even with a `None` halter (16 allocs / 1024 B over 8 runs, byte-for-byte); now sized
iff the halter can run. Static sibling hunt over a 6-line cfg-attr × allocating-`let`
window found ZERO second instances.

## Issue 899 (2026-09-26) — second-moment, null-normalized drift alignment (the Plan 610 redesign): CLOSED (negative on 2 of 6 bars; became the default summary inside opt-in `arm_drift_alignment`; no promotion)

`SecondMomentDrift` (`3485ec59e`, Bench 901) is the only summary loop-sound in both
directions, so it is the feature's default; held-out 0.766 FAIL + G4 2.07× FAIL
recorded. New mandatory bar for curiosity-class primitives: loop soundness in BOTH
reward directions (necessary, not sufficient).

## Issue 895 (2026-09-25) — guided width rollouts on the belief host (GRAM re-distill, Research 590): CLOSED (T1–T7 + T9 landed opt-in; G1 FAILED, demote TRIGGERED, no promotion; T8 handed to riir-ai Issue 1008)

`guided_width_rollouts` + `guided_width_hodge` (`a4421939f`/`ae7b13320`, Bench 898):
width beats depth on MULTI (+0.117 ± 0.049) but loses on SINGLE (−0.188 ± 0.036) —
G1 FAILED, guided table off-by-default forever. Pass arms: E9 MULTI coverage 0.59 →
3.32, G2 14.9 µs/decision, G4 0.

## Issue 897 (2026-09-25) — katgpt-kv's KVarN `static_cal_tables` branches were gated on an undeclared feature and named a missing module: CLOSED (deleted, not wired; `unexpected_cfgs` allow removed; a second instance, `memory_soup_dtree`, restored)

Dead since the Issue-015 extraction (`b61a34f7b`) — ~3 months of "static cal replaces
Sinkhorn" described uncompiled code; quality fails anyway (static ≡ plain RTN, max
|Δ| 1.2e-4; latency 0.034× Sinkhorn). Removing `#![allow(unexpected_cfgs)]`
(`f3e3a4f56`) surfaced `memory_soup_dtree` — restored as opt-in.

## Issue 896 (2026-09-25) — KVarN's raw tile buffer was shared across layers, and the in-progress tile dequantized to zeros (2-bit keys panicked): CLOSED (fixed under `kvarn`; layer-major bit-identical; no consumer's published quality figure carried it)

One raw tile per layer (`62fd22b5f`, ~1 → ~27 MB); layer-major oracle 960 cases /
10,427,200 elements / 0 differing bits; 240 decode-order cases bit-exact;
revert-probed. Perf: Bench 894 GOAT + bench_895 unchanged.

## Issue 883 (2026-09-25) — fitted token-value tables (K=V+ retrofit, mean-removed V quant, V-cache halving): CLOSED (P0–P4 landed opt-in; two primitive-level G2 bars FAILED and are recorded; the model-bound half is riir-infer Issue 013)

P0–P3 (`0b768e95d`, Bench 895). Promotion 2026-10-05: HOLD then re-fire the SAME
DAY — PROMOTED TO DEFAULT via the fused deferred-restore kernel + bitwise (sin,cos)
table (riir-infer `fe75497`, Bench 017: 1.050×/1.123× @4097 under the ≤ 1.20 bar);
the losers (P1 mean-removal, P2 K=V+) demoted on the Bench 011/012 negatives.

## Issue 894 (2026-09-25) — KVarN dequant loops were scalar (bounds-checked indexing); rewritten as bit-identical zips: CLOSED (GOAT PASS, ships under `kvarn`)

`bf7d37244` — zip/as_chunks strided iterators, per-element ops and order unchanged;
bit-identity 960 cases / 10,407,928 elements / 0 differing bits (revert-probed).
G2: value 4-bit 2.46× faster (bar ≤ 0.90); 883 P1 G2 FAIL recorded (+13%).

## Issue 882 (2026-09-25) — differential anchor scoring (common-mode rejection on score surfaces we own): CLOSED (P0–P4 landed, all opt-in; promotion owed to consumers)

P0 `differential_anchor` (Bench 886, λ*=0.55; law: hubness = top-1 WIN-COUNT
distribution), P1 `attention_snr`, P2 `row_logit_floor` (→ riir-infer Issue 011),
P3 `differential_kv_eviction` (Bench 894; the model-bound half closed SPLIT in
riir-infer Issues 012 — the λ axis is synthetic-regime-bound; λ=0 max-recent is the
real-text policy), P4 riders (Bench 897).

## Issue 886 (2026-09-25) — activation-diagonal weight-quant fitting (AWQ/imatrix-class substrate): CLOSED (P0 + P1 landed opt-in incl. the model-bound half; P2 deferred)

P0 `act_channel_moments` (`0fb2254d9`, Bench 896) + P1 `act_aware_fit`; the
model-bound half (riir-infer Issue 014, Bench 007): born-ternary clean negative (the
shipped amax payload IS the fit family's fixed point), dense-parent lane TRANSFERS
(−34.6%). P2 (AWQ α-rescale) deferred, reopen on a PTQ-of-dense consumer.

## Issue 887 (2026-09-25) — `ladder_gate` streak-gated advancement + corrective backtracking FSM: CLOSED (GOAT PASS, stays opt-in)

`ba7fb89ba` (Research 589 / arXiv:2609.19717 ATC): advance after m consecutive evals
≥ τ, retreat to the SHALLOWEST failing stage, fail closed; Bench 891 G1 17/17, G2
0.8–1.2 ns/eval, G4 0. No production consumer (opt-in).

## Issue 885 (2026-09-25) — `laya-tetris-v3` real-hard-drop lane (Issue 884 path B): CLOSED (every lane landed and deployed)

`1a05a9764` DropRule v2/v3 + enumerator; v3 fixture blake3 `12035ebf…` (oracle re-run
on 3 changed states only); reflex `ca1483c`+`a56d850` serve v3; reflex-site
`6116e70`+`2e99eca` deployed (parity 809/809; wasm sha256-identical to `b935412`).

## Issue 878 (2026-09-24) — tetris_sim fidelity boundary: RECORD RELOCATED (2026-09-25)

The standing rule (never swap in a guideline Tetris engine; new fidelity = NEW lane
with its own fixture and head) lives at
`.docs/06_game_arenas/tetris_sim_fidelity_boundary.md`; sim path updated to
`crates/katgpt-tetris/src/sim.rs`.

## Issue 873 (2026-09-22) — governed paged-pool primitives from mini-AGI: CLOSED (all three landed, opt-in; consumer work handed off)

Research 581 (volotat/mini-AGI @ `96784b7`): `pool_admission` `eabd0cb80` (Bench
873), `rate_control` `ee01f1598` (Bench 875), `dying` `650faeac7` (Bench 874) — all
GOAT, opt-in. Consumer A/B handed to riir-train Plan 416 (`e2721e83`).

## Issue 884 (2026-09-25) — tetris_sim `hard_drop` tunnels through roofs: CLOSED (fix path A, site-side; corpus untouched)

Real defect: hard_drop rests at the DEEPEST collision-free row, tunnelling under
overhangs (3 of 2660 options). Claude verdict: path A — live-play filter
top-unreachable spots (reflex-site `c1fcf88`, deployed `31744b04`); sim + fixture
unchanged; the pinned count-3 test is the drop-the-filter signal.

## Issue 880 (2026-09-24) — calibrated-mass router gate, `exact_mass_admit` consumer lane (a): CLOSED (shipped opt-in, G2 FAIL by construction)

`gate_sigmoid_topk_mass_into` (opt-in `calibrated_mass_gate`, Bench 885): |Σm−k| ≤
9e-7; G2 FAILS BY CONSTRUCTION — the ≤1.05× bar was inherited from a comparison of
ALTERNATIVES, and the upgrade composes them. Lesson: a ratio between two operators
does not transfer to one built from both.

## Issue 881 (2026-09-24) — the numbering sweep's standing backlog, three repos over their ratchets: CLOSED

T1 riir-ai: 17 of 17 new resets were walker PHANTOMS (`216defb9b` — `git cat-file
--batch` framed by LINE; parser now reads by byte size). T2 riir-shader: five
collisions were renames `-M` cannot pair (`3141a0467`; `collapse_renames` +
`--full-history`). T3 mmorpg-editor: one real reset; pin 5 → 6.

## 2026-09-23 — the owner-gates menu v2 executed: schedules re-armed (row 5), pipefail residue closed (4e), toolchain batch-pin (4f), and the opt-in verdicts recorded (4a/4b/4c/4d)

Schedules RE-ARMED (`test.yml`, `full_gate.yml`, `feature_isolation_weekly.yml`);
pipefail residue closed (last site riir-neuron-db `c5c11b5a273e`); toolchain
batch-pin in 5 repos (`8c7755b3fdbb`, `c47227e06fcb`, `d0c917f1eb60`,
`ee8b6c2bee6c`, `d54fbe8ebf32`); verdicts: `certified_frontier` promote-with-consumer,
`gw_alignment` NO, `hint_regret` opt-in; the alias codec retired
`DOCS_GATE_KNOWN_EXTRA` (local `scripts/repo_alias.local.txt` carries the 3 rows).

## 2026-09-23 — Issue 877 CLOSED (merge `b701cf564`): the main↔develop sync — origin/main merged into develop with `-s ours`, zero content delta, ancestry restored

Main's three commits were transplants of develop work (`git merge-tree`: every main
hunk existed on develop, evolved); `-s ours` restored ancestry, pushed `--atomic
origin develop HEAD:main`. Lessons: merge-tree hashes are spelling-bound (quote the
FILE SET); a pin measured before the commit carrying it is stale (`HEAD^{tree} ==
HEAD^1^{tree}`).

## 2026-09-23 — Issue 867 CLOSED (+ the issue): the non-hidden-state canonical-AST construction — G5 returned NO attributable signal; instrument-broken, not claim-refuted

T1 `65b199b0` + T2 `67884461` in `katgpt-canon`; G5 (riir-train Bench 605) PRIMARY
+0.50..0.55 but the fit-time shuffle null alone gives +0.41..0.44 — no attributable
rung. Control failing = instrument broken, never claim closed (the Issue-825
clause); reopen paths in riir-train Bench 605 §Verdict(6).

## 2026-09-23 — Issue 875 T3 CLOSED (+ the issue): time-annealed sampling ranges + the closed-form truncation predicate (Bench 883)

`TimeAnnealRange` + the truncation predicate (`ε = ((T−t_cut)/(T−t_min))²`) in
`horizon_weights` + the `dllm_solver` seam; the gate ran CROSS-REPO on the
riir-train C9 toy: flat 0.4084 = PFD, the truncation cost law measured (+9.0% W1 for
9.37% mass dropped). All arms OPT-IN.

## 2026-09-22 — Plan 607 ACCEPTED + T0a/T4a/T0b landed: the modelless game-decision lane's first three tasks (substrate gate, the laya-Tetris enumerator, the G1-oracle fixture)

Thesis: laya's protocol (code arithmetic → templated English → BERT) is the
closed-grammar regime where corpus-limited scoring is lossless. T4a
`examples/tetris_01_state_enum.rs` (120 states / 2,660 options; v2 grammar 245
distinct sentences, cross-sentence ties 0); T0b oracle generator lives in
riir-reflex first (`e4bf657`), fixture byte-identical.

## 2026-09-22 — Issue 874 closed: G8 re-founded on real claims — the dead speedup gate became a bit-contract pin + a relocated, executing throughput floor (two sessions, one issue)

G8 asserted ≥1.5× SIMD but both routes ran the same `dot_8wide` — it always SKIPped.
Verdict: retire the relative claim; G8a `g8_route_bit_agreement` (exact to_bits) +
G8b `g8_throughput_floor` relocated onto bench_271 in the x86_64 matrix (`best_of_us`
+ black_box). G7's constant-folded "0ns" fixed the same day.

## 2026-09-22 — Issue 873 primitive B landed twice in one evening: the twin-duplicate resolution (third of the class)

Two sessions took "primitive B (`rate_control`) remains" from one handoff summary;
origin-first `ee01f1598` won, the duplicate reset away, its verifier ran the
winner's gates. Rule: a handoff naming remaining work is an ASSIGNMENT to every
reader — `git fetch` + check before pushing; on the twin tell, drop yours and
VERIFY the winner.

## 2026-09-22 — Issue 871 closed: the `algebraic_*` A/B measured end-to-end — feature-gated `algebraic_dot` ADOPTED (owner verdict a′), the in-crate codegen truth repaired, T6/T7 deferred on-record

`katgpt-attn-match` opt-in `algebraic_dot` (twin of `dot_8wide`, 1.77×–8.8×, accuracy
IMPROVED on cancellation data); `algebraic_div`/`algebraic_rem` BANNED
(`scripts/algebraic_op_ban_gate.py`). T5 codegen: 0 packed float-math at both
x86_64 arms — the 2026-07-29 "optimal fmla" story was never true.

## 2026-09-22 — Issue 872: SIMD bitstream whitespace splitter for `encode_into_pretok` (the bitcannon-class port) — scan 1.60×/1.69× measured, bit-identity by differential, four live bugs caught by the harness

`fast_bpe/simd_split.rs` (16B SSE2 / 32B AVX2 runtime-probed / NEON SWAR / scalar),
Unicode bit-identity via `char::is_whitespace`; avx2 1.60×/1.69×. The differential
harness caught four live bugs (VT exclusion; `b == 0x80`; `1u32 << 32` wrap;
multibyte ws at word end). fast_bpe stays opt-in.

## 2026-09-22 — Issue 870 closed: distance_abstain's rationale-free sigmoid copy → exact_sigmoid delegation, measured 3-ULP envelope, bench_845 GOAT re-run identical

The substrate-first Mode 2 audit (`683d06d8`) found an unpinned local sigmoid beside
sanctioned patterns; delegated to `exact_sigmoid` with a measured 3-ULP envelope pin
(`sigmoid_delegation_matches_frozen_legacy_body`); bench_845 G1–G5 PASS identical.

## 2026-09-22 — Issue 869 T5: the mini dllm lane goes per-layer honest end-to-end — the gradient check caught a live backward bug the loss-decreases gates never could

Training/eval/decode honor `config.n_layer`; the gradient check caught the
`!is_masked[p]` skip valid only at the readout layer (~10-150% per-element corruption
at inner layers, EXACT under full masking — why the loss gates passed). Fix `last &&
!is_masked[p]`; bench_602's calibration is model-class-specific (re-open at Bonsai
scale).

## 2026-09-22 — Issue 869: multi-layer D2F decode + taps at depth — the Issue-865 Bonsai-scale unblock lands; bitcos x86_64-lane clippy debt repaired in passing

`forward_block_causal_with` over `D2fContext::decode_n_layer` (per-layer KV planes;
depth defaults to 1) + `set_probe_tap_layers` (the old `tap_layer != 0` rejection
REMOVED — the unblock). Found: `micro_dllm_text()` declares n_layer=2 but indexes
layers[0] only (T5 migrates it).

## 2026-09-21 — Issue 864 closed: BITCOS tier ships opt-in — footprint PASS, latency honestly LOSES on this host (Bench 846)

`bitcos` (presence bitmap + compacted neg-sign stream, z-meter, three GEMVs,
dispatch): footprint beats both tiers above z=0.375; G2b(a) latency FAIL → opt-in
(instruction-bound on the 13700K). Two codec bugs caught pre-timing (pack must pext
not pdep; the scalar pos plane is `p & !neg`).

## 2026-09-22 — Issue 868 closed NEGATIVE: engram-fused PUCT G5 FAIL — the evidence gate worked, and that is why nothing happened (Bench 848)

`engram_puct` (Plan 605): G5 FAIL 48.1% h2h (Wilson lower 44.8% < 50%), budget arm
35.7%; the evidence gate correctly damped 99.98% of lookups (mined positions all
n < 4). G1/G4/G6 PASS; re-open on a transposition-dense domain or larger corpus +
re-mining (`EngramHotSwap`).

## 2026-09-21 — Issue 858 closed: g8's arch-conditional bar gets its aarch64 executing lane (PERF_ROWS)

`g8_cached_faster_than_uncached` red on the M3 alone (0.68–0.71 vs bar 0.5) — arch
CONFIRMED, cache regression REFUTED (13/13 x86_64 PASS at 0.39–0.47). Dual pin
`G8_BAR` (aarch64 0.75, else strict 0.5) + a new `scripts/test_gate.sh` `PERF_ROWS`
row kind (`katgpt-rs:12:belief_drafter_goat`, `--release --test-threads=1`).

## 2026-09-21 — the x86_64 execution matrix caught a day-old test that had never executed on x86_64: the ordered-dot anti-dedup pin was crafted at NEON's vector width

Cell 7 confirmed `ordered_dot_differs_from_simd_dot` (landed `5458dd69`) red 3/3
ALONE: the len-4 pin sits below AVX2's 8-wide floor where the scalar tail IS the
ordered fold. Recrafted at len 16 (hand-computed per backend: ordered 3.0, every
reassociating backend 6.0); rerun 7/7.

## 2026-09-21 — the first full-gate run since 09-16 caught the Issue-860 landing RED: 8 `-D`-list errors in the opt-in feature's test code, invisible to every default-feature lane

Layer 3: 8 errors in `successor_density_critic.rs` `#[cfg(test)]` — the Issue-803
class on the non-default axis (an opt-in module compiles to nothing everywhere but
all-features). Repaired; riders in `set_diffusion_schedule.rs`, `ugc_schedule.rs`,
`bench_602`; after: `⚠ full gate PARTIAL — every layer that RAN is clean`, 1120s warm.

## 2026-09-20 — the x86_64 execution matrix's first full run since 09-16 (354 commits of drift): 11,344 assertions PASSED, and one more load-flipped bar caught by execution

Cells 1–7 green; cell 8: `bench_105_gdn2_goat::goat_6_context_scaling_flat_o1`
PASSED-ALONE 3/3 (0.306 vs 0.30 bar) — the fourth Issue-833 member found by
execution. Repair: `tests/common/ab_timing.rs` gains `best_of_arms` (round-robin
N-arm, per-arm MINIMUM, loud-zero); after: spreads 0.038 / 0.051 / 0.102 vs 0.30.

## 2026-09-20 — Issue 861 CLOSED: the ugc_alloc_check Windows G4 alloc was a per-call env read — and the issue's own isolation table was wrong

50/50 G4 failures on Windows/MSVC traced (via the armed-window allocator) to
`std::env::var("UGC_DEBUG")` per call — Windows converts the name to a UTF-16
`Vec<u16>` (Unix `getenv` doesn't allocate); the filing table had blamed the sampler.
Fix: `ugc_debug_enabled()` caches in `OnceLock<bool>`; G4 0 allocations. Lesson:
every hot-path env read takes the OnceLock form.

## 2026-09-20 — Issue 859 CLOSED: Jev structured reads — POC GOAT + 4090 reference + T5 policy arm measured; promotion declined on layer posture (evidence-banked)

Research 574 (vLLM PR #57250) → `structured_read[_into]` + `sample_label_index`
(opt-in `structured_reads`, katgpt-forward); Bench 816 GOAT; 4090 T1 corpora 10/10;
T5 (Bench 817): agreement bars do NOT beat single-read analytic confidence; maxprob
beats entropy on wide sets. Stays opt-in, evidence-banked; re-arm = first production
consumer.

## 2026-09-19 — the open-issue backlog is cleared by owner call (12 files removed; triggers recorded here)

Every open issue adjudicated in one owner session: resolved work got a closure
record, open work a PARKED / PULL-GATED / DECLINED verdict with its reopen trigger;
files removed (git history); no number reused.

## Issue 780 — CLOSED as PARKED: OnlineLinearReadout waits on a consumer that measured itself absent (2026-09-19)

The primitive stands but its live consumer closed NEGATIVE first (riir-clippy Issue
107, `5dfa1daff`); building now would be a synthetic-fixture GOAT pass speaking to
nothing measured. Reopen triggers recorded (organic fixseq ring + populated
span-embed column; riir-train store densification; post-keep-fix corpus ordering
shift).

## Issue 827 — CLOSED: T4 decided by owner call — known-extra repos ARE legitimate oracles; the freshness guards T2/T5 landed are the correct limit (2026-09-19)

T1/T2/T3/T5 landed (`6d084c38`, `7464fc6e`, `3948f0e2`): the ORACLE-STALE bucket
(never clean/counted/ratcheted), per-row oracle disclosure, two-sided fixture arms,
and the fetch-age `⚠ UNVERIFIED UPSTREAM` advisory. T4 STATUS QUO — the measured
hazard was FRESHNESS, not AUTHORITY.

## Issue 833 — CLOSED: the class is measured, the executed backlog is 3 of 38, and the residue is routed (2026-09-19)

T3's resolvers closed every STATED blind spot they could (`381f01f7`,
`e56e6773`, `1b7756c3`, `71d97908`); T2 EXECUTED all 38 root-`tests/` GATES rows in
release — 3 candidates (bench_176 ×2, bench_164) + two other-class defects (Issues
855, 856). T4: no verdict half, deliberately — migration is a four-axis per-target
read.

## Issue 834 — CLOSED: census shipped, the count reframed as a sampling population, and the shared-crate question is decided NO (2026-09-19)

T1 shipped `scripts/sequential_ab_timing_audit.py` (`8045fae9`); the decided count
is a population every matrix run SAMPLES FROM. T3 owner call: `ab_timing.rs` STAYS a
`#[path]`-included module — NO shared crate (katgpt-rs is the public upstream
funnel); reopen threshold: a fourth sibling hand-rolls a twin.

## Issue 835 — CLOSED: T4 decided by owner call — the path dep is the workspace convention; publishing is contrary to Research 003 (2026-09-19)

T1 = docs-gate CHECK `cross_repo_path_dep_gate.py` (`5b09c3f4`); riir-llm cloned and
registered, gate green over 311 deps / 183 manifests. T4: riir-llm keeps its path
dep, not published — path deps against `/git/` ARE the workspace contract.

## Issue 839 — CLOSED: kron_tile 4/4 GOAT; T7 DECLINED on measurement (2026-09-19)

T1–T6 + T8 done (`a623d7d4`, Bench 839; stays opt-in per the no-default-consumer
rule). T7 DECLINED: Bench 843's width sweep measured ternary/f32 1.56× slower at
w=32, peak 3.70× at m=512 — a 32×32 f32 factor is already L1-resident and
add/sub-accumulate loses to FMA at every tile width.

## Issue 841 — CLOSED as lead-backlog dissolved: 7 rows landed, the remaining leads stay at their research-note homes (2026-09-19)

Landed: `.kpt` archive POC (opt-in `kpt_archive`, Bench 841), the fix_verify
refusal-vs-truncation audit (riir-clippy `651728c0`), `legal_token_set` PROMOTED
default-on (20.6–22.5× tree build), `kv_sink_window` (5/5 GOAT, pull-gated on Issue
857), calibration staleness + all four seams (`2057ac3b` etc.). Remaining leads stay
at their Research 568/570/571 homes.

## Issue 852 — CLOSED as PARKED: the τ(t) P_e-LUT POC waits on a D2F model consumer (2026-09-19)

The modelless half is landable, but GOAT G2 needs a measured P_e curve at matched
step budget and no in-repo D2F model harness exists — landing now is the
synthetic-fixture pass Issue 780 refused. Reopen: a real D2F lane, or riir-train's
RecFM twin (its Issue 563).

## Issue 853 — CLOSED as PARKED: the autoguidance POC waits on its measurement arms (2026-09-19)

Arm A needs a DDTree acceptance harness over a served model; Arm B is riir-train
Issue 563's free-rider. Prior art in Research 572 (the logit-space arm =
contrastive-decoding ADOPTION; the dual-dropout latent arm = the fusion candidate).

## Issue 854 — CLOSED: the wedge is diagnosed, the repair scoped, and T3 stays trigger-gated by design (2026-09-19)

Landed (`dded36d8`, `65bf2944`, `d8b55a34`, `c83f0f84`): the SLOW-vs-WEDGED recipe,
the census (exposure is in mutation runs, not the 131 git call sites), NO in-process
wall bound (it would share the watchdog's defect), and the external per-module
timeout as practice. Reopen: a stall on a NON-mutation run voids T2.

## Issue 855 — CLOSED: every task resolved; the guard gate + ratchet sweep both shipped (2026-09-19)

T3 ran all 34 asserting timed regions at n ≥ 1000 — 7 VANISHED, all repaired; T4:
the static `let _ =` detector REFUTED by execution, `timed_region_guard_gate.py`
shipped as a docs-gate CHECK (`196a4cda`); T5 ran all 33 sibling rows (repaired at
riir-ai `3712d51b6`); T6 `timed_region_drift_sweep.py` (`72ee0f5b`).

## Issue 857 — CLOSED as PULL-GATED: the corpus builds when riir-ai 882's consumer fires, not before (2026-09-19)

Boundary answered (`51291548`): corpus + harness = riir-train, canary wiring =
riir-ai Issue 882 T2, the scoring instrument = here (shipped). riir-ai 882's
T-Pull-1 is the pull-gate, owned there; the fire condition and sizing rules are
recorded.

## Issue 843 — CLOSED: the plasma_path ternary dense matvec loses below L3 on every served shape; T4 resolved as per-shape dispatch (2026-09-19)

Ternary matvec is 1.9–3.0× slower whenever the f32 operand fits L3 — every served
shape; x86_64 crosses at L3, NEON never to 1 GiB. T4 owner call: per-shape dispatch
`simd_matvec_plasma_dispatch` (+ cached `l3_cache_bytes()`); Bench 843-dispatch GOAT
GREEN; `plasma_path` STAYS DEFAULT-ON.

## Issue 831 — CLOSED: bench_171 P3 reclassified as instrument-health + mechanism by owner call (2026-09-19)

The coin flip was repaired 2026-09-18 (the screener's work was deleted by `let _ =
acc`; 63–64% release with one `black_box`). T5 owner call: P3 gates instrument
health + mechanism as exact CALL COUNT (13,824 vs 41,472); the ≥30% latency assert
REMOVED, not lowered.

## Issue 847 — CLOSED: `simd_lut_dequant`'s AVX2 kernels compiled to NOTHING on every ordinary x86_64 build — and the bf16 sweep that followed was right for one kernel of three (2026-09-19)

`#[cfg(all(x86_64, target_feature = "avx2"))]` on SHIPPED paths compiles to nothing
by default — repaired with runtime `simd_level()` (1.7×, 4.4–5.6×; RNE 2.4–2.5×
probed in; trunc a REGRESSION). T3 walled `scripts/shipped_target_feature_gate.py`.
Through-line: three wrong answers from sound-looking measurements.

## Issue 850 — CLOSED: the UNVERIFIED-upstream guard challenged only ONE of `behind_origin`'s two silent readings — and then PRINTED a false statement about the one it gained (2026-09-19)

T1 gap: `(n, 0)` was silent; the repair surfaced 6 findings in 3 repos, 4 already
FIXED upstream. T2: sweep FETCH? NO — 250.2s serial / 50.2s 8-way vs sweeps
0.04–40s; `scripts/fetch_contract_repos.py` landed. T4 armed the COUNTER axis both
ways. T3: the guard's false statement now carries `(age, commits_behind)` per repo.

## Issue 832 — CLOSED: the last fixed shared-temp site is repaired — the non-demo backlog is zero, the ratchet is the wall (2026-09-19)

Final site seal-remake `db68f5e`; test-class sites repaired with sibling SHAs
(riir-ai `9d138531b`, riir-chain `8a3b0f5`, riir-game-sdk `ef66117`, riir-clippy
`f5ada0ec`, riir-deployer `b551028`); the demo class adjudicated; the x86_64 matrix
fully green (11,177 assertions).

## Issue 844 — CLOSED: the dot-delegation crossover is length 24, per-ISA, measured on BOTH arches — not a backlog (2026-09-19; full record at the 09-19 dated entry below, file removed this commit)

Full record: the 2026-09-19 dated entry below (M3 NEON: delegation wins at EVERY
length ≥ 4; x86_64 crossover 24).

## Issue 840 — RESOLVED: two `.research/569` documents landed forty minutes apart and tripped two detectors — the rewrite set was EMPTY, and the attribution lessons became the staged-set rules (2026-09-19)

Both live peers confirmed neither document was theirs; the hand-derived rewrite set
(3 sites) was EMPTY when measured. Durable output: the attribution discipline in
AGENTS.md §staged-set (`4e82cc489` + `c0995ea4b`).

## Issue 836 — CLOSED: a `git worktree` made the entire head-provenance mechanism silently inert — `.git` answers TWO questions and each spelling was wrong for the other (2026-09-19)

T1 `f6749af5c` (every `worktree_state` guard used `.is_dir()` — in a worktree
`sweep_advisory` returned `[]`); T2+T4 `c12e96415` (three private copies measured
then DELEGATED to `worktree_state.is_checkout`); T3 cross-repo (3 sites in 2
siblings, 1 defect) repaired at riir-ai `9637d09ea`.

## Issue 845 — CLOSED: `channel_aware`'s duplicated dot kernel deleted; T5 measured NEON PARITY — the x86_64 1.8–7.8× penalty does not transfer to aarch64 (2026-09-19)

The crate's `simd_dot_f32` was a ~200-line duplicate whose AVX2 arm was cfg-off by
default (2.72–7.76× slower); repaired by delegation (`32056164e`), the bench
converted to a bit-identity gate. T5 NEON: parity 0.98–1.04× — "shipped is faster"
is x86_64-only.

## Issue 851 — CLOSED: `muon_update` step scale repaired to the canonical √max family (update RMS ≈ 1.0) (2026-09-19)

The default-on Muon wrapper scaled `1/max(rows, cols)` (D× below canonical) while
claiming standard scaling; fix `1ebe39f61` to `√max(rows, cols)` (Keller and
Moonlight forms differ by LR-absorbable constants); regression pin update RMS ∈
[0.68, 1.12]; Bench 050 unaffected.

## Issue 842 — CLOSED: the alias-mapped sweeps opened a directory that does not exist, and the first real read surfaced a workspace of hidden findings (2026-09-19)

17 of 19 drift sweeps opened `WORKSPACE / <contract-name>` — nonexistent on an alias
box, every walk returned 0. T1: one seam (`sweep_population.open_repo` +
`repo_alias.real`, identity when unmapped); T2 labeling law: resolve the path you
READ FROM, keep the name you LABEL WITH; T3 the first real read surfaced the hidden
findings; T5 the citation sweep stayed red = true positives (Issue 846).

## Issue 846 — CLOSED: the alias seam has a PROSE half — 44 true CROSS rows, 37 cleared by teaching the qualifier the on-disk spellings, 7 by prose (2026-09-19)

Nearly every row was the ON-DISK spelling (`seal-*` vs contract `mmorpg-*`).
Classifier: `issue_citation_gate.spelling_aliases()` in code, LENIENCY ONLY, with
the `_NAME` boundary regex (`seal-remake` must not match inside
`seal-remake-unity`). Prose half fixed at riir-dao `160f9a1`, riir-kat `51994ee`,
riir-neuron-db `3f6cf67`; sweep rc=0.

## Issue 838 — CLOSED-as-decided: a shell script spawning a NATIVE child is a third encoding seam, and the population is measured at zero live instances, so it is deliberately NOT gated (2026-09-18)

Census (`c9afca9dd`): 43 native-child lines across 197 tracked `*.sh`, only 2 with
non-ASCII (both comments) — a gate would govern an empty population. Keep native
children ASCII. Adjacent gap: the matrix picked the memory source on EXISTENCE not
ANSWERS (MSYS `/proc/meminfo` shape).

## Issue 837 — CLOSED: registering a contract repo reds the whole sweep family, and the ONE file a gate checks is not the twenty-one that make them red (2026-09-18)

`riir-llm` registered with only `repo_set.txt` + AGENTS.md — 19 of 21 sweeps red
UNPINNED, hiding four live findings. T2: `scripts/repo_registration_gate.py` (every
`*_drift_floors.txt` needs a row per on-disk canonical repo; subset scope declared
in `scripts/repo_registration_scope.txt`, reds both ways). Registration is a
22-file operation with one file gated.

## Issue 825 — CLOSED POSITIVE, after a same-day RETRACTION of its own negative close. Coulomb crowd redistribution ships as `coulomb_flow`; the "negative result" was a bench walker with two defects (2026-09-18)

Two sessions landed opposite verdicts four hours apart; settled by running the
LOSER's fixtures through the WINNER's readout (MAE 0.0 vs 0.119) — disagreement is a
cheaper oracle than self-consistency. `coulomb_flow` ships opt-in (Bench 825,
G1–G4 PASS); Bench 815 repaired (the absorption rule IMPORTED, not copied).

## Issue 830 — CLOSED: Issue 829's anchor class one seam deeper — the locale-I/O classifier was anchored to a CALL-NAME SET, and it had repaired one side of a round trip in this repo's own instrument (2026-09-18)

The class is a text-mode FILE OBJECT (`p.open("w")`, `tempfile` text factories,
`os.fdopen`, `io.TextIOWrapper`) — `locale_io_fix.sites()` gains the forms behind
`_text_mode()`; `.open` admitted only on a literal TEXT mode, the no-mode blind spot
counted (5 sites) and PRINTED. 11 sites total; siblings cited (riir-train
`15db2c67`, riir-clippy `54b999de`).

## Issue 828 — CLOSED: Issue 823's anchor class, third position — the heading oracle was anchored to a DELIMITER SET, and the biggest unread family is this repo's own house style (2026-09-18)

The delimiter set lacked the em dash — 56 of the unread are katgpt-rs's own
`## Issue NNN — …: CLOSED (date)` house style; adding a delimiter KEEPS the
discriminator (`follow-up` still rejected, pinned in both delimiters). T4 ANSWERED
Issue 823 T5 by PRICING: of 153 unread records only 2 carry a number no other oracle
knows — 98.7% redundant. ⛔ The arm that could not pass was cp874 mangling the
fixture's em dash = Issue 829.

## Issue 823 — CLOSED: the heading oracle was anchored to a POSITION, and its own blindness meter was anchored to the same one; T5 ANSWERED by Issue 828 T4 (2026-09-18)

`_SELF_HEADING` anchored to the LEADING kind — six repos write the date first, 74
records unreadable; the meter `_HEADING_SHAPED` had the same anchor and failed
toward clean (riir-chain printed a PERFECT 0/1 over 20 unread). Repair
`_SELF_HEADING_DATED`; workspace IN-LOCAL-RANGE 54 → 27, ratchets tightened. ⛔ Do
not re-pin `max_in_local_range` for a heading-blind row — read `heading_unread=a/b`
first.

## Issue 829 — CLOSED: the FILE seam under Issue 778's PIPE seam: 273 text-I/O sites decode with the system locale, and the first one found was making a whole file of arms pass for the wrong reason (2026-09-18)

Found by Issue 828's dash arm failing where a scratch copy passed (cp874: bare
`Path.write_text` writes U+2014 as `0x97`). Census 273 sites / 8 repos; repair half
`scripts/locale_io_fix.py` (`col_offset` is a UTF-8 BYTE offset; newline style
preserved; the kwarg joins the last ARGUMENT); verdict half
`scripts/locale_io_gate.py` (UNPARSED reds, exemptions deliberately empty); the
sweep half landed in the SAME CHANGE (118 sibling rows with SHAs: riir-train
`39d0b8d4`, riir-clippy `638fa224`, riir-ai `4df109c65`, riir-chain `bebf78a`).

## Issue 824 — CLOSED: the family gate watched the failure it was built for happen beside it (2026-09-17)

Issue 821's close-out said "wired into 16 sweeps"; the family is 19 —
`cfg_row_implication_drift_sweep` hard-red live. Repair: the
`sweep_advisory_membership_gate.MECHANISMS` registry (verdict per mechanism, never
pooled); pin key `(mechanism, sweep)`. Found by reading EXIT CODES (a glyph grep
printed `<no verdict line>` 19 times).

## Issue 821 — CLOSED: Issue 815's marker never reached the sweep family: 8 sweeps red on 0 findings, and two live ratchet breaches sat behind them (2026-09-17)

`DOCS_GATE_KNOWN_EXTRA` landed in the final-line disclosure but not the per-repo pin
loop — 8 of 9 sweeps red on repos where they found nothing, hiding two genuine
riir-train breaches. Repair: `sweep_population.pin_row_exempt(name)` wired into 16
sweeps. ⛔ A scripted multi-file edit needs a `py_compile` sweep; Python
`write_text` flips files to CRLF.

## Issue 820 — CLOSED: the numbering sweep is blind to Issue 795's class, so 15 repos read `dup=0` over 113 collisions (2026-09-17)

`numbering_gate.historical_collisions()` never reached the sweep — 183 historical
collisions over 16 repos in 3.7s, 113 in repos called clean. `max_hist` ratchet at
measured + `min_numbers` floor on the HISTORY walk; katgpt-rs's row asserts the
gate's VERDICT. T5 unplanned: 819 dual-allocated — the sigmoid lane KEEPS it (7 of
10 inbound).

## Issue 822 (2026-09-18) — CLOSED. An UNCOMMITTED row was counted into a RATCHET; all 19 sweeps adjudicate against HEAD now, and the mechanism is GATED

`worktree_state.py` three instruments by classifier shape (`head_delta` per-file /
`head_overlay` cross-file / `head_tree` multi-seam, 22–32s per dirty repo); rules:
the VERDICT belongs in the row KEY, a FLOOR is a pin too, classes split by ORACLE.
Closed with the `head-provenance` MECHANISMS registry row; the first live MASKED row
landed four hours later. ⛔ Attribution hazard: shared author + one reflog — only a
`Session: <name>` body line is reliable.

## Issue 819 (2026-09-17) — CLOSED. The x86_64 arm was LINTED by nothing; the finding is not the 30, it is the sibling

30 `unsafe_op_in_unsafe_fn` in `channel_aware.rs`'s AVX2 arm (0 with avx2 off); the
NEON twin had the `unsafe { }` + `// SAFETY:` — a repair applied to the arm somebody
can see is a measurement of which arms are visible. T3: `full_gate.sh` Layer 2c
(both avx2 arms; the TRIPLE printed on the verdict line). ⚠ 819 is held by two
documents; adjudication pinned in `scripts/number_collisions_expected.txt`.

## Issue 819, second holder (2026-09-17) — the sigmoid prior-logit lane KEPT the number; RESOLVED (file removed)

Won the same-day dual allocation (7 of 10 inbound, Issue 724 T2). T1–T3 + T5 landed
with Bench 813 (G1–G4 ALL PASS): the sigmoid prior-logit lane + sink-stability
forecast (Research 566) as two OPT-IN features. "Issue 819" in .research/566, 258,
392 means THIS lane, never the x86_64 lint lane (disambiguated per Issue 794).

## Issue 815 (2026-09-17) — CLOSED by its own criterion: the box is green. Option 2 landed as `DOCS_GATE_KNOWN_EXTRA`, and options 1 and 3 remain the owner's

MIRROR of `DOCS_GATE_PARTIAL_CLONE`; takes NAMES never `=1`, reds both directions,
one classifier (`skill_repo_set_gate.known_extra_state`). Three instrument defects
found landing (the partial-set-as-whole display; ambient-env selftests; a cp874
UnicodeEncodeError) + a fourth via arm_reach closed by EXTRACTION.

## Issue 808 (2026-09-17) — CLOSED: T1 landed option 2, the AVX2 `argtopk` dispatch narrowed to k ≤ 4 on x86_64; the measured loss is GONE and the wins are kept

`AVX2_ARGTOPK_K_MAX = 4` (x86_64 only; NEON keeps k ≤ 16): late_peak k=8
0.41–0.72× → 0.98–0.99×; wins kept (k=2 5.11×, k=1 6.73×). Option 1 refused:
`N_MIN` is distribution-sensitive. T2 closed in two halves (a load-immune pin + a
timing floored at 0.85×).

## Issue 806 (2026-09-17) — CLOSED: T8 was the last open task and it was entirely Issue 808's T1

T6/T7 closed 2026-09-16 (Bench 806 Addenda I+II); record
`.benchmarks/806_x86_64_execution_matrix.md`; the matrix stays a workstation
verdict, no CI lane.

## Issue 818 CLOSED (2026-09-17) — the four no-default `--all-targets` breaks gated both halves; bench_412's green-zero row found in the same sweep

Fix `78a1ac5d7`: four targets fixed with whole-file `#![cfg]` + matching
`required-features` rows; the fifth (`bench_412`) had the cfg but no row (green-zero
class). Corollary: promoting/demoting a default-on feature must MOVE its target
gates, never delete them.

## Issue 808 (2026-09-17) — T4 LANDED: `argtopk` AVX2 dispatch re-measured on six realistic block-score distributions + per-k `N_MIN` crossover (Bench 810)

Across six distributions × both profile arms the k≤16 loss SURVIVES except
early_peak and DEEPENS on late_peak (no n-floor rescues k=8); k≤4 is
distribution-robust; `N_MIN` is distribution-sensitive. Instrument: interleaved
median-of-ratios (`tests/common/ab_timing.rs`).

## Issue 817 (2026-09-17) — single-pass AVX2 `argmax` port measured NEGATIVE + reverted; the NEON premise does not transfer (Bench 812)

A single-pass AVX2 port of the NEON kernel was a NET LOSS (iid 0.24–0.28×, early
0.20–0.23× at n ≥ 256); won only `late` and n=64. Demote-on-loss: dispatch reverted,
kernel deleted, numbers + reopen trigger in the doc comment;
`bench_817_argmax_dispatch_ab.rs` KEPT as the reopen instrument.

## Issue 811 CLOSED (2026-09-17) — DBTM confidence-commit anchor rule: PoC PASS → Plan 600 landed end-to-end, promotion EXECUTED via Plan 601

Research 563 (arXiv:2609.15903) → Plan 600 `ConfidenceAnchorConfig` +
`anchor_then_fill_with` (`8bd6d7132`, Bench 600 GOAT) → Plan 601 real-text eval
(Bench 601) → promotion EXECUTED: `ConfidenceAnchorConfig::default()` = κ 0.9 +
floor. Deferrals discharged via the Issue-813 seam + Bench 809 + the Issue-816
trainer.

## riir-ai Issue 964 C2 LANDED (2026-09-16) — `CalibratedActionBridge`: decision-level confidence calibration for the ABSTAIN threshold (Bench 808)

The second `sigmoid_calibration` consumer: calibration in front of the ABSTAIN
threshold, argmax invariant (selection stays on raw scores). Bench 808 GOAT: ECE
0.0220 → 0.0088; the ABSTAIN point moved 4.4× toward oracle; 0 allocs.

## riir-ai Issue 964 C1 LANDED (2026-09-16) — `clr_calibration`: the CLR verifier becomes the first `sigmoid_calibration` consumer (Bench 807)

`CalibratedVerifier<V>` (katgpt-claim, opt-in): planted-drift ECE 0.0924 → 0.0164;
the calibrated G2 fixture undisturbed; substrate fix: the `apply` identity fast path
at (w,c)=(1,0) (logit→sigmoid is not bit-exact in f32).

## Issue 812 CLOSED (2026-09-16) — the bench_doc_audit BlindRead context split was a TMPDIR-FORM split; the arm now matches paths form-independently

`TMPDIR=/tmp/...` vs resolved `/private/tmp/...` — the fix:
`root_forms = (str(root), str(root.resolve()))` in `scripts/bench_doc_audit.py`;
the pre-fix red is the canary.

## Issue 810 CLOSED (2026-09-16) — `sigmoid_calibration`: the Platt-style calibrated sigmoid gate lands as a PoC with all four gates green

`SigmoidGateCalibrator` (FIFO observe, deterministic 2-param Newton refit
off-hot-path, BLAKE3 commitment); the guard `w = 1/T > 0` never reorders (G3 by
construction). Planted: ECE 0.0696 → 0.0322. Consumers filed as riir-ai Issue 964.

## Issue 809 T1+T2 CLOSED (2026-09-16) — the global-RNG census is read; the class is gated; the T3 `Rng::new()` census is deferred with a reason

Wide predicate: 20 sites / 9 files, one defect (an unseeded bench fixture → seeded
`Rng::with_seed(809)`). T2 `scripts/global_rng_gate.py` (membership, LINE-FREE
keys, both directions; the canary found an UNTRACKED planted file invisible — `git
ls-files` IS the population). T3 landed same day `6d120abeb`: `Rng::new()` /
`Rng::default()` joined (147 sites / 55 files, all pinned).

## Issue 806 T6 CLOSED (2026-09-16, the M3 side) — t698_t5_kv_mean adjudicated arch-dependent with dual pins; kda grad-check floor was below its own noise

Passes on aarch64 (pin `23d0daab3f087159`); x86_64 measures `4d0b592740db9358` (band
bits one ulp apart) → arch-conditional dual pins; the stale row removed from
`scripts/x86_64_matrix_expected.txt`. kda grad-check floor raised 5e-4 → 2e-3 (was
below its own measured FD noise).

## Issue 807 (2026-09-16) CLOSED — `lthash`: incremental homomorphic multiset hash, the shared substrate for the Agave-mined commitment/state-hash proposals

Mined from Agave (eprint 2019/227 LtHash) for riir-chain Proposal 010 D1 +
riir-dapps Proposal 005 D1; `crates/katgpt-core/src/lthash.rs` opt-in; Bench 771:
`replace` 31.6 ns vs a 1000-member 33.8 µs rebuild ≈ 1069×. Awaits first consumer.

## First x86_64 execution of the katgpt-core/katgpt-types SIMD suites (2026-09-16) — 15 latent AVX2 bugs caught and fixed; Bench 800's execution-parity caveat resolved NEGATIVE then closed

15 FAILED on the first x86_64 `+avx2` run, all never-executed AVX2 defects (bf16 RNE
dropped the final `t >> 16`; the LUT gathers used SSE4.1 `_mm_cvtepu8_epi32`).
Lessons: a parity caveat resting on a shared-algorithm argument is a conjecture
until run; record `target_feature` flags with every run.

## Issue 805 (2026-09-16) CLOSED — `numbering_gate.py --help` printed ten "remove the row" lines about a repo it could not read

`--help` was read as a repo PATH; git failure → EMPTY set → every pin STALE with
destructive remedies first. Repair: `unmeasurable(repo)` → exit 2 via toplevel
EQUALITY. The census found `orphaned_attr_gate.py` passing over a nonexistent repo
at exit 0 — fixed (UNSEEN); final arm_reach: 25 modules / 691 mutants / 465 killed /
0 UNREACHED / 0 NO-ARM / 0 BASELINE.

## Issue 804 (2026-09-16) CLOSED — 28 instruments crashed when run the way AGENTS.md says to run them; the cross-repo axis is seven repos, not one

On cp874, `print()` of the verdict glyphs dies with NO verdict — UNREAD findings
(the restatement sweep was unlooked at while the family read green). Repair:
`scripts/console_safe.py` (`errors="backslashreplace"`), 28 callers;
`console_encoding_gate.py` joined CHECKS. Cross-repo: 71 undefended across SEVEN
repos vs katgpt-rs 0/72 — the sweep ratchets the derivative.

## Issue 803 (2026-09-16) CLOSED — the off-macOS partial gate printed the SAME final line as a full pass

`develop` carried 24 `error[E0560]` + 4 `-D` lint errors for ten hours while the
narrow gates were green; `--allow-partial-platform` would have read it but printed a
final line byte-identical to a full pass. Repaired: the final line carries the
partial verdict and names the unmeasured axes (`3ceb541b`).

## Issue 802 (2026-09-16) CLOSED — commitment-gap calibration rig: residue DEAD-BY-DOMINATION at micro scale; stability features (item 3) shipped earlier in the day

Bench 802: G3 FAIL — the no-gate one-forward baseline dominates (+3.5 acc at half
NFE); mechanism: self-consistency labels blind to context contamination. Revival
needs all of: large would-miss mass, improving revisions, an iterating baseline,
DAgger-style calibration. Also: root `katgpt_rs::speculative` re-exports
`StabilityTracker`/`TOPK_DRIFT_K`/`N_STABILITY_FEATURES`.

## Issue 800 Arm C phase 1 (2026-09-16) — GraphStablePool<T> extracted: the common contract verified across 4 sites (a 4th found in-repo)

`877e06eb2` + Bench 800-C: index-stable slots, LIFO free list, append-only growth —
ships 4× (radix_prefix / PagedKVCache / riir-gpu arenas + `BranchBank`); INDEX
stability is the pool's; payload-ADDRESS stability belongs to the stored type.

## Issue 801 (2026-09-16) CLOSED — T4 PoC: composition-quality REFUTED, disagreement-trace CONFIRMED; T5 routing: meld stays opt-in as a contradiction detector, Super-GOAT Q3 blocked-as-refuted, T2 audit stands

T4 (riir-ai Bench 932): mean matches/beats every meld arm — composition quality
REFUTED. Positive: the λ̃ disagreement trace (contradiction AUC 0.976, precision@8
0.997) shipped in `katgpt_core::meld` (opt-in). Research 560 DOWNGRADED.

## Issue 800 Arms A+B closed (2026-09-16) — bf16 SIMD G2 FAIL-honest (autovec parity); slot-flip DECLINE (ties mpsc); JSD kernel lands (Issue 802 item 2)

Arm A (`f314d5006`): LLVM auto-vectorizes the scalar reference — 1.00× widen/trunc,
1.19–1.23× RNE vs the ≥4× gate; FAIL-honest, opt-in. Arm B: slot-ownership beats
serial +21–46% but TIES mpsc → DECLINE. The NaN-safe bounded top-K JSD kernel
(`jsd_topk`) landed.

## Issue 801 T1–T3 (2026-09-16) — NAP audit + `meld` primitive: the census survives code-level scrutiny; the algebra holds, the quality ladder did NOT transfer (T4 is the sole adjudicator)

Research 560 audit: 4 INADMISSIBLE / 1 PARTIAL / 1 N.A.; `meld` (`43f15f7c8`)
bit-exact commutative soft-min with λ⋆ ≤ 3.24e-7. Bench 801 red flag: the paper's
quality ladder did not reproduce at D=32 — T4 is the sole adjudicator.

## Issue 800 Arm A (2026-09-16) — bf16⇄f32 SIMD kernels: G2 FAIL-honest — the pufferlib kernel-shape premise does not exist on M3+rustc

NEON/AVX2/scalar bf16⇄f32, RNE bit-exact vs `half` incl. the NaN class; G2 FAIL
1.00× widen/trunc (LLVM auto-vectorizes the scalar loops into identical NEON). Kept
for G1 correctness + the ISA guarantee; `bf16_simd` opt-in, AVX2 compile-verified
only.

## Issue 799 (2026-09-16) — bevy_ecs 0.15→0.19 bump landed: the arenas are load-bearing evidence infrastructure, bevy_ecs is a schedule-free utility layer

The ARENA is load-bearing (Bench 432 evidence), bevy_ecs ≈23.5K LOC of `World` +
queries — bumped `e0f02ab93` (`Events<E>`→`Messages<E>`, `#[derive(Event)]`→
`#[derive(Message)]`); uuid 1.26 pulls getrandom 0.4 (wasm pin). Bench 799: G1
byte-identity PASS; G2 full-game harness ~2× slower, documented.

## Issue 798 (2026-09-15) — a tracked landing record claimed a sibling-repo repair that was never committed

Two `scripts/` files recorded cross-repo repairs as landed; neither existed, both
sweeps RED throughout (`toolchain_override_drift_floors.txt` "all five markers are
in" — 2 of 5; `pipefail_discard_expected.txt` DROPPED a row — the unrecoverable
direction). Rule: a cross-repo repair is landed only when COMMITTED in the sibling,
and the record must cite the sibling commit. ⛔ This session then committed the same
defect (`b592a213` cited three dropped SHAs) — cite AND check it resolves. T2:
`behind_origin()` (the mirror of MASKED); T3: `n_assertions()` derives the count.

## riir-train Issue 549 fixed in `490b662e` (2026-09-15, M3 + 4090 session) — avx2_exp_sum_inplace: the one exp kernel missing the n-clamp

Below −87.3 nats the AVX2 exponent WRAPS: exp(−300) = 6.9e23 — surfaced as
riir-train Issue 511's Windows collapse. New
`simd_exp_sum_extreme_inputs_underflow_not_wrap` pins all three paths; missed
because the truth sweeps stayed clear of |x| > ~88.

## Issue 779 T1+T2 (2026-09-15, M3 session) resolved — subspace_intervention promoted + the FUNCATTN spectral arm POSITIVE (Bench 766)

`katgpt_core::subspace_intervention` opt-in (the Issue-778 POC promoted; a real POC
bug caught — 778's "ridge" collapsed to class-sum `W = XᵀY`). T2
`spectral_pre_rotate` POSITIVE: eigen-aligned 0.802 vs random 0.354, and > full
0.656 — projection DENOISES.

## Issue 779 T3 (2026-09-16, M3 session) resolved — real-bank affinity: saturated plateau, NO re-pin; three-arm POSITIVE on real tensors (Bench 767)

riir-ai `future_probe_bank_capture` over gemma-2-2b: the bank regenerates
BLAKE3-IDENTICAL (`99edecca…8b81`). Affinity: CEILING PLATEAU (L02–L25 all 1.000) —
no re-pin needed. Three-arm POSITIVE: top-4 probe-SVD dims = full (1.000) vs
random-4 0.146; projecting them OUT collapses 0.344 → 0.000.

## Issue 782 (2026-09-15, M3 session) resolved — slt_sweep: the noise-sweep λ̂ estimator (781 T4), GOAT G1–G4 ALL PASS, promoted default-on

`katgpt_core::slt::sweep` default-on (Bench 765): v1's windowed mirrored-Hill was
−28…−37% biased; v2 fits the log-log CDF slope over an order-statistic ladder
(χ²₄ ~2%). Geometry: TUBE/CONE near-unbiased; ISOLATED minima −18…−19% at d ≥ 4
(ranking preserved); the ReLU toy boundary recorded as the instrument boundary.

## Issue 781 (2026-09-15, M3 session) resolved in `580bda30` — slt: the RLCT λ + WBIC selection primitive, GOAT G1–G4 + floor ALL PASS, promoted default-on

`katgpt_core::slt` (Bench 764): six closed-form SLT selection functions;
planted-rank recovery — WBIC picks 6, raw loss picks r_max, naive BIC over-penalizes
(~24 nats inside the window). Loss convention: per-SAMPLE nats (the first draft
averaged, shrinking gains 8×). UQ floor: a thin ~1% margin recorded honestly. T0
novelty KEEP with caveat (λ̂ is the classical Hill estimator; the novelty is the
application).

## Issue 775 (2026-09-14, M3 session) resolved in `3a59abe1` — dual_wave: the PC-ALM dual accumulator + closed-form rate laws (core) + the ballistic DEC wave kernel (dec), GOAT ALL PASS, opt-in

Research 554 (PC-ALM) → opt-in `dual_wave` in katgpt-core + katgpt-dec; reach wave
LINEAR (18/97/212 ticks at L=16/64/128) vs heat quadratic (×9 growth — the Eq-23
law). Lessons in the bench doc: per-layer Jury rates blow up the coupled chain; the
power-iteration estimate is ‖AᵀAv‖, NOT its square. NOT promoted (zone hierarchies
are L≈4).

## Issue 777 (2026-09-14, M3 session) resolved in `7e2a2638` — modality_additive belief kernel: FLYNN's linear-sensory-integration property distilled, measured, GOAT-passed, promoted same-day

The PoC confirmed FLYNN's prediction and refuted two of its specifics (the failure
is RATIO-flavored; `AttractorKernel` passes near-linear — P1 necessary-not-sufficient).
`evolve_belief_additive` measures EXACT superposition at 26.8 ns/tick — promoted to
katgpt-sense default. Hygiene: an unsink'd timing loop DCEs to 0.0 ns; a 2×
comparative gate is ill-posed for a COEXISTING method (gate an absolute budget).

## Issue 789 (2026-09-14) — a gate whose own failure path is asserted by nothing: CLOSED

Six of twenty CHECKS invoked NO arm at all (2,050 lines of per-push logic whose
failure path never executed). The first census over-reported (grepped CLI flags;
three checks DELEGATE to their classifier). T1: the shared fence parser was blind to
tilde fences (exposure LATENT: 0 tilde lines workspace-wide); T2
`scripts/check_validation_gate.py`; T4: NO sweep — katgpt-rs is the only repo with a
CHECKS array; don't add one by symmetry. The standing lesson, recorded seven times:
a rule landed in one instrument and never generalised.

## Issue 788 — the population-predicate registry was hand-maintained, and a careful reading missed two of ten: CLOSED (2026-09-14)

`PREDICATES` registered seven while ten existed. Three defects fell out: the eighth
was WRONG (`.exists()` not `.is_dir()`); two were UNPARAMETERISED; the
real-workspace verdict coupled to unrelated failures. `SUBSET_PREDICATES` is its own
tuple; the discriminator for predicate-shaped defs is DIRECTORY ITERATION (`ast`
deliberately unused — a sibling's syntax error must not blind this gate).

## Issue 787 — a census reads the DOCUMENT, so an undocumented instrument is invisible to it: CLOSED (2026-09-14)

Both censuses enumerated audits AGENTS.md documents — `len_derived_binding_audit.py`
was in none. The predicate is REACHABLE (roots AGENTS.md + `scripts/docs_gate.sh` +
workflows; closure follows script → script); HISTORY.md deliberately NOT a root. Of
9 unreachable: two wired into AGENTS.md (the default), seven pinned by MEMBERSHIP
with reasons. The sweep is a RATCHET (riir-train 61 of 61 over-captured). The
closure is TEXTUAL — false reachable is the dangerous direction.

## Issue 786 — the `.len()`-derived binding audit had no verdict half, and the reason it went unnoticed is the finding: CLOSED (2026-09-14)

Ninth instance, the QUIETEST: an undocumented instrument has no symptom — it just
stops running. Three measurements decided the pins: a bimodal population (floors
vacuous in 14 of 16), cross-repo-by-construction verdicts (leave-one-out 0 flips),
UNRESOLVED 118 of 164 stays unpinned. No `min_rs_files` column — three sweeps floor
that identical walk; the delegation is ASSERTED.

## The platform-dead_code class got an instrument — `scripts/platform_dead_code_audit.py`, and it was wrong on its first sweep (2026-09-14, 4090 session)

Three things wrong first: it INVENTED a finding (masking strings dropped Rust 2021
inline format args); a `mod` row is not a rustc finding (MOD-REF its own bucket); an
arm passing under its own perturbation certifies nothing (the vendor arm). Two real
rows compile-verified and fixed on riir-ai (`note_ane_dispatch`, `gen_u64_bytes`).
Standing: 0 findings · 1 MOD-REF.

## The Windows all-features lane — NEON_U8 platform gate, first specimen of the platform-dead_code class (2026-09-14, 4090 session)

`NEON_U8` (katgpt-pruners `interval_pruner/simd.rs:27`) ungated, used only in
aarch64-gated code — dead on non-aarch64, fixed `ea4c2873` (mirrors `AVX2_U8`).
Instrument lesson: a `clippy::`-prefixed JSON grep reads FALSE all-clean (rustc
codes have no prefix). Recurred same-day in the zed fork (riir-clippy
`.distill/001` P22).

## Post-riir-train-513 develop drift — the 09-12→09-14 touched-rows window audited green on the workstation (2026-09-14, M3 session)

The touched-rows gate over the drift window: 25 selected rows, 25/25 BUILDS at EXACT
feature set · 0 FAIL · 0 UNSEEN. Rows since 09-11 had drifted to 710 via develop
landings the main-only CI never audits.

## Issue 774 (2026-09-14, M3 session) resolved in `22e65be4` — the wasm32 surface audit's BY-DEP verdict: the row predicate was dep-blind, closed with a five-canary self-test

The resolver credited only ROW evidence, so transitively-built crates read UNCOVERED
(compile-verified riir-shader effects). Fix: the fourth verdict `by-dep`, never
folded into NAMED; credit = non-optional in-repo path deps positively naming wasm32.
Standing 26 NAMED · 2 BY-DEP · 0 UNRESOLVED · 1 UNCOVERED (the negative control).

## Issue 748 (2026-09-14, M3 session) resolved — option (a): all three unwired Lean negative tests now run in their lean_proofs.yml CI jobs (~162s/main push)

Three of four `proof_negative_test.sh` invocations were wired to NOTHING; each repo
adds the script to BOTH `paths:` lists + a step after `proof_gate.sh` in the SAME
job (katgpt-rs `3c97358c`, riir-ai `ecb21f3f7`, riir-neuron-db `4a68575`). Gap 1 (no
develop lane) UNCHANGED BY DESIGN.

## Issue 773 (2026-09-14, M3 session) resolved — the 772-B2 removal's stale re-export: root lib E0432 under `flashar_consensus,plasma_path`, found by riir-ai's guard through the path dep

The double cfg hid the stale re-export from the 772 wave's default lanes;
riir-ai's guard Layer 1 died E0432. Fixed `b7fcabd8` (3-line deletion) — the first
scoped-closeout found cross-repo by a DOWNSTREAM gate.

## Issue 771 (2026-09-14, M3 session) resolved — the radix-tree prefix KV cache primitive (RadixAttention index) shipped opt-in; G1–G4 ALL PASS, promotion deferred to the serving lane

`katgpt_kv::radix_prefix::RadixPrefixTree` (16-token chunks, leaf-preferential LRU,
the tree owns page INDICES — CUDA-graph address stability by construction); G2
hit-rate 2.45× flat control, match latency 9.8×. STAYS OPT-IN (every lane
single-stream); a racing duplicate dropped per numbering discipline, its tests
adapted.

## Issue 770 (2026-09-13, M3 session) resolved — the counter walker rebuilt per-commit; the 769 adjudication was partly an instrument artifact (verdict-review round 2)

15 of 31 reset rows named FORWARD-stepping commits (date-ordered hunks around one
`current`); "all 31 non-merge" was true BY CONSTRUCTION (`git log -p` doesn't diff
merges). Repair: each commit's counter read at the commit AND each parent; merge
resets now visible. 27 real resets; the Issue-768 verdict stands.

## Issue 769 (2026-09-13, M3 session) resolved — the counter-reset class lands in the numbering sweep; all 31 measured resets adjudicated

`numbering_drift_sweep.py` gains `resets` (committed BACKWARD moves re-spend
numbers; ratchet-pinned) + `unbumped` (worktree counter below committed max —
REPORT-to-owner). The walker imported from `highwater_contiguity_audit.py`; read
with Issue 770's correction (15 of 31 phantoms).

## Issue 768 (2026-09-13, M3 session) resolved — the .highwater ownership witness REFUTED by measurement; highwater_contiguity_audit.py landed

438 gaps + 27 resets over 73 counters — no major repo contiguous; an over-claiming
witness VALIDATES wrong addresses. The contiguous-suffix witness DECLINED (T1 found
zero live rows it would change — decline is a correct answer). Protection stays on
the canonical heading.

## The all-features E0252 root-name collision — ooo_audit::AuditScratch aliased (2026-09-13, M3 idle sweep)

The root re-exported `AuditScratch` twice — collides only with both opt-ins on;
fixed `a0ca7d36` (`AuditScratch as OooAuditScratch` at root only, zero root-path
consumers).

## The Windows default-features workspace lane — bench_mtp_metal_batch_floor platform gate (2026-09-13, 4090 session)

13×E0433 — the only Metal example without `target_os = "macos"` guards; fixed with
item cfgs on all 22 items + a loud `not(macos)` stub. The Windows default workspace
lane hit 0 warnings for the first time here.

## Issue 766 (2026-09-13, 4090 session) resolved — len_derived audit: caller tracer (HALF C) + two instrument defects found and fixed

HALF A's fixed 6000-char window bled past kernels (22 of 74 false positives) →
brace-matched body; HALF C classifies (handle, length) per caller —
PERSISTENT-UPSTREAM is the eyes list; GUARD-ONLY re-verdicts 14 rows. Standing: 118
UNRESOLVED — the honest floor. Heading correction: katgpt-rs owns 766 (witness
`e4792a4b` bumps `.issues/.highwater` 765→766; the issue file was never committed).

## WeightEpoch — the KV-cache weight-identity epoch (riir-ai Issue 938; Plan 025 contract) (`49f5d245`, 2026-09-13)

`LoraAdapter::weight_epoch()` — BLAKE3 over domain-separated serialization; IDENTITY,
not an install counter (A→B→A may reuse cache; any byte difference is a new epoch).
Motivation: the RLT staleness law (riir-train `.research/453`); riir-ai consumes it
at `CpuInferenceBackend`, refusing mixed-epoch reads.

## Modelless-first mandate — original section (incl. the canonical-failure story)

Pre-compaction narrative parked here (2026-09-06, from `1801c0ab`); the live rule is
AGENTS.md §Modelless-first. Canonical failure: AC-Prefix G1 (Plan 313) was
prematurely deferred to riir-train though the doubled-signal bias was systematic —
the modelless fix landed bit-identically (`.benchmarks/313_ac_prefix_modelless.md`);
`ac_prefix` re-promoted DEFAULT-ON.

## Modelless-first mandate (the core principle)

Parked copy of the mandate as it stood at the 2026-09-06 compaction; the live rule
lives in AGENTS.md §Modelless-first mandate.

## Boundary contract — original section

Parked pre-compaction narrative; the live contract pointer is AGENTS.md §Boundary
contract.

## Boundary contract — read `BOUNDARY.md` first

Parked copy of the boundary section from the pre-compaction AGENTS.md; the live
section lives in AGENTS.md.

## The full gate — original section (narratives)

Parked full-gate narratives (the axis table, the CI fire history, the Layer 6b
landing, the mktemp BSD fix, the inverse-macOS module table); the live compressed
rule lives in AGENTS.md §The full gate.

## Docs gate — original section (descriptions + narratives)

Parked docs-gate descriptions and sweep narratives (check histories, the
two-more-tiers table, the first-census results); the live rows live in AGENTS.md
§Docs gate + drift sweeps.

## cfg-gated targets — original section

Parked cfg-gated narratives (the severity splits, population widenings, the
`debug_assertions` dimension, arming side effects on binary-counting floors); the
live rule is AGENTS.md §cfg-gated targets.

## required-features rows — original section

Parked required-features narratives (UNSEEN-never-folds, the opposite-direction
repairs, the touched-gate cost table); the live rule is AGENTS.md
§required-features.

## Percentile index — original section (classifier history)

Parked percentile classifier history (four vocabulary gaps, the TRUNC-VAR closure,
the legitimate exclusions); the live rule is AGENTS.md §percentile.

## Staged-set audit + shared target dir — original sections

Parked staged-set narratives (the `b2527521` incident, the signal details, the
riir-game-sdk false-red measurement); the live rules are AGENTS.md §staged-set +
shared target dir.

## Feature Flag Discipline — rule histories (lossy surface, Report the Floor, Plan 467)

Rule histories: the lossy-surface rule arrivals (Research 502, Bench 696, riir-ai
Issue 750 bisection; external confirmations arXiv 2609.01962 + 2608.12700); the
Report-the-Floor rule (Issue 010 FULLY CLOSED; K-sweep structural — pick K by
chaotic memory, periodic data uses the floor); Plan 467 `DualLeoOracle` G5 FAIL at
both postures — closed per riir-ai Research 322.

## Substrate-First Gate — original section

Parked pre-compaction copy; the live rule is AGENTS.md §Substrate-First Gate.

## Substrate-First Gate (MANDATORY before implementing)

Parked copy of the gate from the pre-compaction AGENTS.md; the live section lives
in AGENTS.md.

## Research Workflow — original section

Pointer to `.agents/skills/research/SKILL.md` (paper classification, 7-repo
routing, fusion-first distillation, novelty + GOAT gates, modelless-unblock §3.5).

## Repo count — the full original paragraph (drift history)

The drift history: said 8 and 18 while membership changed; 19 → 16 on 2026-09-04
(retirements to obsolete/, `riir-burner` last sweep `ce54122`); a count that MATCHES
is still not a checksum over a set. The three-test rule (product / value / rate) and
the canonical failure (game programs in riir-chain's consensus set — riir-chain
Issues 096 + 097) live in AGENTS.md + BOUNDARY.md.

## Resolved issue log (verbatim from pre-compaction AGENTS.md)

The pre-compaction resolved-issue log; the records are compacted to one line each
under the heading below.

## Issue log (resolved)

- **Issue 792** (as 776) — docs-gate CPU self-timing printed a well-formed number measuring nothing on Windows (`times` counts MSYS children, not native — 1.26s vs 19.7s wall); the gate CALIBRATES (0.25s child burn) else prints `CPU SUPPRESSED`.
- **Issue 776** (the OTHER 776) — contrastive matched-swap + norm-matched noise interventions (Research 555, `be4ff672`+`d936b5fd`); Issue 791 adjudicated the number to this document (weight 16 vs 5).
- **Issue 775** — `platform_dead_code_floor_gate.py` (CHECK 18) + `platform_dead_code_drift_sweep.py` landed; `--prove-fires ea4c2873`; the Windows CPU figure filed as Issue 792.
- **Issue 765** — the `DOCS_GATE_PARTIAL_CLONE=1` marker axis: a loud DEFERRAL on the population axis, never auto-detected; gone-only disagreement reds naming BOTH hypotheses.
- **Issue 763** — signed-graph LIF reservoir (opt-in `lif_graph`, `74fe08f1`, Bench 760: 97,285× vs dense at 3.1% active); hazards: canonical ascending spike order; ring period 1404 = 18×78.
- **Issue 764** — `WeightEpoch` + `from_parts` + `as_bytes` (`49f5d245`+`aa163896`+`a7d6d8f0`); the optional refusal-arm gate NOT taken (the consumer lives in riir-ai).
- **Issue 743** — `gw_alignment` (Plan 594, Bench 709): structure-only GW, G1–G4 PASS; greedy second-order init load-bearing; 2-opt polish removed.
- **Issue 742** — the last 42 `#![cfg]`-gated targets reported a green zero; SILENT-NOW now a WALL at 0 (`2ae0d20a`).
- **Issue 741** — alloc gates unrunnable in the shipped profile + the auditor read one of N cfgs (`da498fa6`); Issue-741-as-filed: `is_load_bearing` couldn't name threat-dialect gates (`ba26462b`, 11 ADMIT tokens; counterfactual 11 load-bearing silent under a silent wall).
- **Issue 740** — regime-probe primitives (`regime_probe`, Bench 702); first consumer riir-clippy Issue 077 (gap −0.729 nats — corpus proximity, not the OOD label, decided the sign); stays opt-in.
- **Issue 738** — the wasm32 lanes compile what they NAME (`wasm32_surface_audit.py`); 15 live findings in the browser crate; classifier-lessons canon (0-file walk; 17 false UNCOVERED).
- **Issue 737** — nothing compiled for wasm32: layer 2b both simd128 arms + `wasm32_gate.yml`; follow-up: the `rust-toolchain.toml` vs CI `targets:` mismatch (`25c89432` — explicit `RUSTUP_TOOLCHAIN: stable`).
- **Issue 734** — a shell gate that ABORTS mid-run reports exit 0: completion sentinel in 37 scripts / 10 repos; verdict + premise instruments; `riir-ai/scripts/e2e_internet.sh` PROVEN inert.
- **Issue 736** — leakage_probe + cross-space diagnostics (opt-in); T4 Super-GOAT; consumer riir-neuron-db Bench 495 (top1 0.828 vs chance 0.086).
- **Issue 735** — the laundering premise is bash-3.2-ONLY (11 interpreters measured); the sentinel is load-bearing IN CI (macos-26-arm64 ships 3.2.57 only); do not pin — measure.
- **Issue 732** — FreshZ0 breadth-restart: decisive quality NEGATIVE (collapses 0.59 → 0.12–0.37; EqR's restart premise doesn't transfer); perturbation breadth pays from K=4 at every depth.
- **Issue 731** — `LoopResidualExit` (opt-in `cadence_gate`): v4 G2 PASS at exactly 2.0× (existence-proof); P3 caught a probe defect → `with_shape_persistence`.
- **Issue 733** — `EngramHotSwap::with_table` didn't hold the writer lock (`31bf0012`): the closure CASes and HOLDS; nested same-thread swap panics loud.
- **Issue 730** — 256K prefill KV-offload: the wall was 4× smaller than claimed (16 of 64 blocks full-attention); premise refuted for every served lane; `scripts/gguf_header_audit.py` kept.
- **Issue 729** — the NaN-comparator wave: ~160 legacy `partial_cmp` + 13 NaN-promoting `total_cmp` → float_order terminals / `cmp_for_max`; the closeout found ~35 more Group B sites.
- **Issue 728** — `silent_now_load_bearing` was 0 everywhere because the classifier speaks one repo's dialect: widened 0 → 12, all armed (49/49); adding rows CAN red a gate counting green binaries.
- **Issue 727** — SP-KV missed both T16 bars once measured at realistic length: the repaired instrument first (the "50% pruned" arm pruned 0/16); "zero-overhead gate bias" is false (+7–12%).
- **Issue 726** — `gauge_rebalance` 3.7× its target: scalar accumulate → `simd_fused_scale_acc` (−53%); t08 re-pinned 30 → 15 µs.
- **Issue 723** — the first full-workspace EXECUTION red: six classes; "Class A's reds are partly the box" was wrong — five were REPAIRED INSTRUMENTS; repair the instrument first, decide disposition second.
- **Issue 725** — the numbering gate covered one repo: 35 duplicates + 7 broken allocators in the other fifteen; T4a `scripts/citation_weight.py`; riir-ai 6→0, riir-clippy 4→0, riir-train 13→0; cp874/cp1252/EOL lessons.
- **Issue 724** — `.plans/` collisions regrew: `numbering_gate.py` + `numbering_floors.txt` + `docs_gate_paths_sync.py`; 449 resolved by citation weight (Poincaré kept; ActionBridge → 587).
- **Issue 721** — the root crate registered a `#[global_allocator]` as a library: now `cfg(all(test, debug_assertions))`; the Issue-682 force-link pattern is dead — never reintroduce.
- **Issue 719** — `cond_audit` PoC (G8 PASS); T2–T4 deferred (reopen on semantic-eviction PRs / riir-train Plan 343 T1.6 / Research 523 H2O).
- **Issue 739** — no rust-toolchain.toml: pinned 1.98.1 (`87dfa778`); load-bearing half: full_gate.yml's deliberate @stable lane is now an explicit job-level `RUSTUP_TOOLCHAIN: stable`.
- **Issue 767** — `mb_value` bounded three-factor (dopamine) plasticity value circuit (Bench 761, opt-in; r=0.9705 vs ridge floor, margin 0.029; toy 2.2 µs).

## Issue 744 — HRM-Text second-pass modelless extraction queue: CLOSED as resolved-negative (2026-09-11)

8 candidates closed after a consumer hunt — no graduating consumer (#1–#4 negative,
#5 single-consumer, #6/#8 want-gated, #7 riir-ai-side). Re-file when a consumer
materializes.

## Issue 745 — Margin-gated verification escalation PoC (TriSpec distill): CLOSED as resolved-split (2026-09-11)

Split, all 9 gates PASS: the CASCADE mapping REFUTED (3–19% false-flag tail); the
ACCEPT mapping VIABLE (50.5% invocation cut at 0.25% regression). Findings: the
"margin ABSENT" reading was a vocabulary miss; trust polarity is workload geometry
(`MarginPolarity` ships both); `margin_gate` OPT-IN.

## Issue 746 — Looped-Flows modelless extraction candidates: CLOSED, split verdict (2026-09-11)

Row 2 LIVE — opt-in `marginal_rewind` (calibrated beats additive-at-equal-budget
5.71×; restart WINS clean-prior — the consumer decision rule; the mechanism is the
de-commit shrink). Row 1 CLOSED (no time-grid consumer).

## Plan 596 — sliceTCA modelless slice-rank decomposition: COMPLETE + PROMOTED (2026-09-12)

`slice_tca` (Bench 714) — G1–G4 ALL PASS → DEFAULT-ON; bit-identical factors vs the
paper's SGD, zero hyperparameters; mixed [64,128,32] loss 0.0263 vs 0.4835 for the
naive floors (18.4×). Held-out CV is structurally impossible for slice models.

## Plan 597 — BMR + EFE-over-models: COMPLETE + PROMOTED (2026-09-12)

`bmr` (Bench 715) — G1–G4 ALL PASS → DEFAULT-ON; vs the brute-force oracle
|Δ| = 7.1e-14; ablation 64/64 vs 2/64 vs 0/64; G2 513×. The Eq-7/9 sign flip
corrected (oracle-arbitrated).

## 2026-09-12 — the citation gate's first CI run was a blind red (promote `35ac604f`)

The gate landed in a main-only window so its first CI run covered 267 commits — exit
2 (1 contract repo < floor 15): the refusal was correct, but a check added while its
lane never fires is deployed-but-never-exercised. `issue_citation_gate.py`
CI-deferred under `DOCS_GATE_CI=1`; docs_gate.yml's fourth trigger-omission instance
fixed (four checks lacked `paths:` globs).

## Issue 755 — a quoted heading in a fence is not an allocation: CLOSED as resolved (2026-09-12)

The allocation path only delegates to the canonical scanner: `heading_allocated()`
0 of 57 fenced (EXCLUDE); `citations()` 57 of 2972 fenced (do NOT exclude — they
carry sibling attributions). Unterminated fences fail SAFE (exit 2).

## Issue 756 — an unterminated fence swallows the rest of its file: CLOSED as resolved (2026-09-12)

19 files / 611 swallowed lines, all repaired same day (katgpt-rs 12, riir-ai 5,
riir-train 2); three shapes (missing closer / stray / missing opener); the reported
line is the DANGLING fence.

## Issue 749 — a cross-repo `Issue N` citation rebinds to the WRONG document: CLOSED (2026-09-12)

AGENTS.md's bare `Issue 750` meant riir-ai's — resolved by
`scripts/issue_citation_gate.py` (+ floors, `e258fdaa`); 8 rows qualified,
revert-probed; 63 shared numbers reported, not gated.

## Issue 751 — the cross-repo citation sweep (18 repos katgpt-rs cannot see): CLOSED as resolved (2026-09-12)

`scripts/citation_drift_sweep.py` + floors (`d8041de5`): 19 repos / 2,939 citations;
FP rate 7/43 = 16% (pre-752); the backlog worked to 0 CROSS same day.

## Issue 752 — a repo name QUALIFIES a citation even when that repo does not own the number: CLOSED as resolved (2026-09-12)

Of 368 qualified, 45 unfollowable (37 WINDOW_ONLY + 8 ADJACENT); `60bc76aa`: a
directory qualifier counts only if that repo ALLOCATED the number. The census's
0/45 was re-rated to 1/45 by Issue 754 — a census inherits its oracle's blind spots.

## Issue 753 — the citation rules' own COSTS were recorded once and never re-measured: CLOSED as resolved (2026-09-12)

(a) the `\d{2,4}` width bound is LOAD-BEARING (`\d{1,4}` makes 51 false heads at 0
true) — pinned `max_single_digit = 0`; (b) the 40-char alias LEAD: widening buys 0
repairs and hides true findings.

## Issue 754 — an allocation that exists only as a HEADING is invisible to allocated(): CLOSED as resolved (2026-09-12)

A file created and removed without a commit leaves only its heading;
`heading_allocated()` (`8e0ffaa0`) recovers it — 7 numbers recovered, 123 CROSS
retired, 752's census 0/45 → 1/45.

## Issue 761 — a Lean theorem can RESTATE its own definition: CLOSED as resolved (2026-09-12)

Four shipped in riir-neuron-db as "the merkle_root guard" (removed by its Issue 617,
`24957a2`); criterion: symbolic equality over LEAF constants; CROSS-DEF is not a
finding; the oracle then found the instrument's own scoping defect (per module +
imports now).

## Issue 750 — docs_gate's CHECKS array vs the AGENTS.md table documenting it: CLOSED (2026-09-12)

Found drifted ("six" vs seven predicates); `scripts/docs_gate_checks_sync.py`
(`d10202b1`): MEMBERSHIP both directions, never cardinality, plus QUANTITY WORDS;
`MIN_ROWS = 10` floors both parses.

## Issue 760 — mmorpg-remaster joined the workspace but not `scripts/repo_set.txt`: CLOSED as resolved (2026-09-12)

Fixed `bfccffea`; the 4090 reds were partial-clone topology (the M3 derives exactly
20); `markdown_fence_drift_sweep.py` caught its unterminated fence at
`.plans/005_layer3_reducer.md:600` (repaired `99064c5`).

## Issue 757 — linking_fold detector Option B (the 50 ms @ n=2×1000 remainder): CLOSED as resolved (2026-09-12)

Single-pass k-NN + certified Gauss pruning: ≈31× and the ORIGINAL 50 ms budget
restored and enforced (G2b 28.28 ms ≤ 50 ms @ n=2×1000). KEEP OPT-IN —
`LinkingFoldCorrector` consumes the FOLD, not the detector.

## Issue 747 — ASEntmax modelless mining (damping schedule, derived-k, eviction window, incremental entmax): CLOSED (2026-09-12)

Four primitives behind opt-in `asentmax_schedule` (Bench 713 + the P0.7 real-model
re-gate: NO modelless gain — σ̂ = 0.1409, far below σ ≥ 1; STAYS OPT-IN).

## Issue 762 — ASEntmax P4 stretch (HoldConcentration, Kamath regime detector, per-head grid, RoPE cutoff): CLOSED (2026-09-14)

T4.1 HoldConcentration + T4.2 `logit_regime` DONE (Bench 759); the T0 owner call:
`asentmax_schedule` STAYS OPT-IN (the gain is budget-confounded); T4.3/T4.4
CLOSED-deferred (σ ≥ 1 absent at n ≤ 173).

## Issue 772 — config-audit first pass over katgpt-rs (7 inert/assert-only knobs + orphan-report layer): CLOSED (2026-09-14)

7 class-A/B knobs + 12 stillborn knobs fixed same day, wire-vs-delete per finding
(A1/A2 wired `3c3c52ce`; S1/B2/B3 deleted — wiring would invert tested bands). FP
record: trait-impl-internal reads are operational (riir-clippy `.distill/001` FP
class #3).

## Issue 777 — an instrument whose population is a FILESYSTEM walk audits code no repo owns: CLOSED (2026-09-14)

A gitignored nested repo (`mmorpg/`: 1404 `.rs` credited to the outer repo — a
correctly-shaped defect at an address where repair cannot be made) + OUT_DIR sources
+ two fabricated floors. Repair: ONE `scripts/tracked_walk.py` (8-arm self-test);
after: 8,694 tracked `.rs` over 16 repos; floors re-pinned at ~65%.

## Issue 778 — `subprocess.run(..., text=True)` decodes with the SYSTEM locale: CLOSED (2026-09-14)

Found on cp874: the em dash reads as three wrong chars under `text=True`, or the
decode raises in the reader THREAD (`stdout = None`, rc preserved). 28 sites repaired
(`encoding="utf-8", errors="replace"`); the gate `scripts/subprocess_encoding_gate.py`
scans the AST (the text scanner reported offenders in its own fixtures); the first
run found a 28th site (`.agents/skills/doc-sync/tools/linkcheck_sweep.py`).

## Issue 793 (allocated as 779; renumbered per Issue 791) — the workstation sweeps' partial-clone verdict, one copy: CLOSED (2026-09-14)

Seven of eight sweeps carried a copy-pasted "pinned but ABSENT" loop and hard-red
with every content assertion green — the remedy became `scripts/sweep_population.py`
(UNREGISTERED / UNSEEN / DEFERRED, never pooled). Behind the reds: four live
Issue-749-class rows, three fixed (`25b7bf6b`, `c5c30fee`, `96041bf6`), one
deliberately untouched.

## Issue 792 (allocated as 776; renumbered per Issue 791) — the docs gate's CPU self-timing printed a well-formed number that measured nothing on Windows: CLOSED (2026-09-14)

This heading exists so the number is READABLE by `heading_allocated` (which reads
`## Issue NNN (…) — title`, not bullets); the full record is the Issue-log entry
above. Verified both ways.

## Issue 790 (2026-09-14) — an arm that exists and RUNS may still reach nothing: CLOSED (2026-09-15)

`scripts/arm_reach_audit.py` + `scripts/arm_reach_gate.py`: mutate outside arm
bodies, re-exec, run the arm. Nine findings in already-green instruments (a bare-dict
exec crashed seven classifiers = 796 mutants; BASELINE missing; TIMEOUT credited
KILLED via `except BaseException`); `--include-all` 55 of 55 (652 live = an unread
backlog, never ratcheted). Pattern: CLASSIFIERS well armed, VERDICTS not — writing
the reason is the adjudication.

## Issue 791 (2026-09-15) — three numbers allocated twice across a 57-commit divergence: CLOSED

Two sessions both allocated 776/779/780; rebase `max(ours, theirs)` makes it
invisible. T1: `removed_candidates()` recovers from `git log -M --diff-filter=D`; T2
moved all three by citation weight (779 moved anyway — coordination-free beats a
5-site lead); T3 deferred. Headings carry "(allocated as NNN; renumbered per Issue
791)".

## Issue 797 (allocated as 796; renumbered — the other session allocated 796 the same hour and pushed first) (2026-09-15) — a sweep reads the WORKTREE, so a finding may exist in NO commit: CLOSED

The last standing CROSS row was an artifact of an uncommitted sibling edit; the
population moved too (601 vs 607). `scripts/worktree_state.py` wired into all 16
sweeps (ADVISORY only); the row-level split DISPLAY-worktree / PINS-HEAD; T5 the
MEMBERSHIP gate (`scripts/sweep_advisory_membership_gate.py`) — "all sixteen" was
stale within two hours. ⛔ `("*.rs")` is not a tuple — 8 of 15 call sites had it.

## Issue 795 (2026-09-15) — 70 numbering collisions the gate could not see, 9 of them live: CLOSED

791's "three" counted what the instrument could see — both-closed collisions leave
nothing on disk (the MAJORITY case). Two regimes: ≥ 700 a WALL by MEMBERSHIP,
below a RATCHET; six of nine NOT renumbered (margins on the pins). A tree-derived
population is not the governed population (74% inflation from `.benchmarks/`
families).

## Issue 796 — the allocation-time dual-allocation gate: the FP rate measured, the gate built CLASSIFIED: RESOLVED (2026-09-15)

`scripts/dual_allocation_fp_probe.py`: 7202 one-sided pairs green BY CONSTRUCTION,
39 real incidents = 31 TWIN + 8 INDEPENDENT — fear moot, caution right.
`scripts/dual_allocation_gate.py` CLASSIFIED (TWIN exit-neutral, INDEPENDENT exit 1);
joined the docs-gate CHECKS.

## Issue 794 (allocated as 780; renumbered per Issue 791) — a wrong address reads as UNDECIDED when its number is in local range: CLOSED (2026-09-14)

`⛔MISATTRIBUTED` was computed only in the CROSS bucket, so an explicitly
misattributed citation under the citing repo's ceiling landed IN-LOCAL-RANGE, never
gated — own class `MISATTRIBUTED-IN-RANGE`, wall 0. The hidden row was AGENTS.md's
own example (riir-train Issue 513 written under katgpt-rs's name). Boundary
measured: 19 of 19 locally-allocated rows are FALSE (contrast prose).

## Issue 781 — the heading oracle matches one house STYLE: CLOSED as a measured, printed blind spot (2026-09-14)

64 of 152 read — split by house style (katgpt-rs's newest closes unreadable by its
own instrument). Closed as a REPORT: the widening is unsound (`follow-up` is the
same shape as `resolved —`, and this is the only path that can SUPPRESS); the cost
prints every run. ⛔ The Issue-794 write-up introduced four rows of its own class —
repair by naming the true owner in the window.

## The x86_64-pc-windows-msvc axis, measured: clean at all-features/all-targets (2026-09-14)

A third platform axis nothing had compiled: exit 0, 32 packages, zero code findings
(all 1,034 warnings are the NTFS hard-link message). Scope: compiles the
`not(macos)` half; one cell added, matrix not closed.

## Issue 782 — a pinned repo absent from the walk is never visited: CLOSED (2026-09-14)

The three sweeps Issue 793 left alone were all non-exempt — the census selected on
symptom; `cfg_row_implication_drift_sweep` had NO absence check (confident green
over 16 of 20). All eleven share the verdict. ⚠ A subset sweep has TWO populations;
the hole is a per-sweep `⛔ DROPPED` check.

## The executed-test gate on x86_64-pc-windows-msvc: exact floors, all three rows (2026-09-14)

`scripts/test_gate.sh` measured on Windows: 203/203 · 2041/2041 · 249/249 — exact on
every row, execution not compilation. 477 integration-test and 176 bench targets
still executed by nothing automatic.

## Issue 783 — the subprocess-encoding gate is katgpt-rs-only: CLOSED (2026-09-14)

The gate shipped with no sweep half — measured 29 DECODE + 2 CHILD-ENCODER over 5
repos; two not latent (`riir-clippy/scripts/gen_dashboard.py:552`). All 31 repaired;
ceilings a wall at 0/0; `min_calls` is 0 in 10 of 16 repos, so `min_py_files` is the
only blindness detector there. Standing failure mode, five times: a rule landed in
one instrument and never generalised.

## Issue 784 — the orphaned-attr gate's cross-repo claim was hand-run: CLOSED (2026-09-14)

The first of the eight with NO new offenders — but it found a stale WARRANT: the
docstring's hand-typed totals didn't follow the tracked-walk migration (22% / 46%
drops; 23,026 sites in unowned trees). The structural repair:
`orphaned_attr_drift_sweep.py` — the total is now MEASURED, not hand-typed.

## Issue 785 — the wasm32 surface audit had no verdict half: CLOSED (2026-09-14)

Cross-repo with NO verdict at all; standing lived as a hand-typed AGENTS.md
sentence. UNRESOLVED walled at 0 (reached by answering, not ratcheting); UNCOVERED
pinned by MEMBERSHIP (`scripts/wasm32_uncovered_expected.txt`); per-repo floors + a
global TOTALS row. ⚠ Not "every cross-repo class": `suite_membership_audit.py` is
report-only by design (1,203 load-bearing unpinned rows).

## Issue 800 Arm C complete — GraphStablePool site re-points: 1 landed, 1 N.A., 2 declined on evidence (2026-09-16)

Site 1 radix_prefix RE-POINTED (`bdb1091a4`); PagedKVCache + BranchBank DECLINED
(refill-in-place / wire pins are load-bearing); Qwen38LaneSet N.A. Lesson: a
contract extraction earns its keep where the contract is the site's whole job.

## The full gate's wasm32 layer counted NEGATIVE cfgs as surface — derivation fixed positive-only (2026-09-16)

Layer 2b red on three `#![cfg(not(wasm32))]` files — the bare `git grep -l` couldn't
see negation (the fourth instrument to meet the class). Fix `448c77f91`: count only
compile-time non-negated cfgs. First full-gate PASS on this box.

## Issue 844 (2026-09-19) — the dot-delegation crossover measured on BOTH arches; the NEON answer refuted the filing session's own expectation: CLOSED

M3 NEON: NO crossover — delegation wins at EVERY length ≥ 4 (9.06× @256), refuting
"crossover LOWER"; aarch64 dispatch is compile-time vs x86_64's runtime CPUID.
Per-site read: 8 CHUNKED + 10 large-D naive; dispositions filed as riir-ai Issue
982, riir-train Issue 562, riir-neuron-db Issue 621.

## Issue 849 (2026-09-19) — the 844 per-site dot read, katgpt-rs-own sites: four delegations landed + the full repair record-back: CLOSED

Internal: `dot_f32_fma4`, `dot_chunk4`, `dot_8wide` (katgpt-core made NON-optional),
`dot_truncated` — all delegate to `simd_dot_f32`. Sibling record-back with gates:
riir-train `3cf49ebb`, riir-neuron-db `5d86dd3`, riir-ai `dbc5639d4`. No site had a
bit-determinism contract; ~3e-6 @64 passed all gates.

## Issue 848 (2026-09-19) — a rename privatised a delegation target and a rework deleted a shared fixture; both of that module's EXTERNAL callers are docs-gate CHECKS: CLOSED

`6c6ca2ee` renamed `is_checkout` → `_is_checkout` updating seven in-module callers
and neither external one — develop red 6h40m, NO verdict. T2
`scripts/cross_module_attr_gate.py` (static, LINE-FREE keys); T3
`scripts/import_health_gate.py` (execution; 6.238s of 6.34s was one all-top-level
module, guarded; MISSING-DEP never flagged). ⚠ A repair that grows a population owes
the other gates a run.

## Issue 856 (2026-09-19) — the green zero has TWO spellings and the audit built for it saw one: CLOSED

A whole-body `#[cfg(feature)] mod tests {}` zeroes identically and the
inner-`#![cfg]` regex never saw it — 26 targets, 175 assertions, 7 `*_goat`. The
predicate now: gated items are the WHOLE body; runs of gated modules gated by
`any(...)`; ⛔ the first repair read exactly like the defect (caught only by the
summary line not moving).

## Issue 860 (2026-09-20) — `successor_density_critic`: tabular discounted count-ratio goal-critic: CLOSED

Modelless CRL extraction (`2c7a1f157`, Bench 818 GOAT PASS: G1a 0.00841 vs the
Bellman fixed point — the first oracle was wrong and the gate caught it). The
consumer pull-gate is satisfied: riir-ai Issue 991 `goal_salience`; stays opt-in.

## The exact_sigmoid / dot_f32_ordered substrate promotion (2026-09-20) — the riir-chain Issue 156 T1 landing executed in this repo

Ungated additive primitives (`5458dd69b` + main `5e2b730f2`): `exact_sigmoid` (2 ULP
vs `fast_sigmoid`'s 580,601,137) and `dot_f32_ordered` (sequential fold, the
deterministic anti-dedup pin). ⚠ The Cephes speed claim inverts on aarch64 (1.7 vs
3.1 ns) — don't quote "~1.7× faster than libm" here.

## Issue 865 T3 (2026-09-21) — the probe_guidance λ-sweep GOAT gate ran NEGATIVE; the negative verdict is the pinned gate: OPEN (lane re-opens at Bonsai scale)

Guided best λ=1.25 DOMINATED by T=1.0 at matched diversity; the trained probe loses
to the zero-logit null at every λ ≥ 1.5. Pinned: pooled unigram entropy is
polarity-inverted on deterministic lanes; read any λ sweep against the null; G2a/G2b
inverted into pins that red the day guidance wins.

## Issue 866 (2026-09-22) — KARC D3 promotion coverage audit: VERDICT QUALIFY — the contract's passing legs live on configs nobody constructs: CLOSED

Bench 849: both passing legs are `ChebyshevBasis`; every deployed shape is
FourierBasis R=1 and fails both bars; no `karc_runtime` GOAT measures forecast
accuracy. Not DEMOTE; riir-ai corrected Plan 332's claim.

## Issue 865 file hygiene (2026-09-22) — issue file removed per noise-reduction, the lane's record was already durable: CLOSED (hygiene)

Removed with no content change; all durable elsewhere (catalog §120, Research 578,
riir-train recipe rows, the G2a/G2b pins).

## Issue 865 follow-up (2026-09-22) — arm (b) unblocked (DropoutHeadProbe) + the headroom study: the negative EXTENDED to every mini-lane regime (Bench 850)

`DropoutHeadProbe` (a deterministic masked tap, no RNG) + the headroom rerun over
three regimes: never beats the null (−0.13…−1.42 pts). A "+2.7 pts" pooled-unigram
positive was RETRACTED — redistribution along the refuted axis. Only the
Bonsai-scale lane remains.

## Issue 876 (2026-09-23) — the flappy render widened to v3; the decoded arm reads Δ0 vs the structured arm: CLOSED

Grammar v3 (`laya-flappy-v3`: quantized OFFSET + neutral motion; v1 "rising" was the
Bench-880 confound): structured 96/100 = decoded 96/100 (Δ0); raw ordinals read
51/100. `515230244`, Bench 882.

## Issue 879 (2026-09-24) — MAttr budget primitives landed opt-in: exact mass at router-regime cost; calibrated-mass column says what the hard cut cannot: CLOSED

`exact_mass_admit_into` (bisect τ so Σσ = k) + `log_frontier`; Bench 884: |Σm−k| ~1e-8
vs the hard cut's −6.4% drift, CHEAPER than the sigmoid gate at N=1e3. ⚠
`exact_mass_admit` ≠ `gate_sigmoid_topk` — opposite mass semantics.

## Issue 892 (2026-09-25) — tetris strategy RULEBOOK (ruliology surface) + chance-node PUCT + laya head-to-head: the hybrid FSM champion out-scores laya 92–299× and Bench 891 4×: CLOSED

16 rules as data + a Build/Downstack/Survive FSM; champion `ed5aa14b7d68472e`, hybrid
`68cae9d382014662` (equal survival at 4.0–4.7× points); laya 0/60 vs hybrid 60/60.
`chance_puct` opt-in (wins only at 19@75). Hybrid PROMOTED; Bench 891 → reference.

## Issue 893 (2026-09-25) — the tetris substrate promoted from `examples/common/` into the leaf crate `katgpt-tetris`: CLOSED

Leaf crate (deps rayon/blake3/fastrand) for riir-reflexer's precise-dep need;
byte-identical migration (fixture replay 1,113,251 / 1,113,238 B; 14/14 tests);
`Genome::champion_hybrid()` pins `68cae9d382014662`.

## Issue 919 T1 (2026-10-06) — the weight-derived spike census landed (Research 605)

`crates/katgpt-attn/examples/spike_census.rs`: the offline GGUF scan of the
dominant-rank-1 stat; the self-test caught a real asentmax precedent bug (f16
subnormal off by one — every subnormal read 2× large). Verdicts: Bonsai-8B clean
Table-1 shape; Bonsai-27B inject-early/cancel-late softly; gemma-2-2b weak (the
suppression prediction). T2 next; nothing promoted.

## Lessons

- A rule landed in one instrument and never generalised — grep the whole family and land the repair as one shared mechanism (recorded ten+ times).
- A census over one representation is blind to what it omits; a census is exhaustive over ROWS, not over its ORACLE.
- A count that MATCHES is not a checksum over a set; "every" typed as a NUMBER went stale within hours, twice.
- An uninvoked assertion is unknown, not passing; a green zero is byte-identical to a pass.
- A platform and a profile are part of the claim — exactly like the feature set.
- A latency number without its BOX STATE is not a measurement; a latency gate in a debug build measures an unoptimised binary.
- Repair the instrument first, decide disposition second — three would-be re-pins were off by 5×/7×/140×.
- A pin re-typed after every run is a diary, not a wall; writing the reason is the adjudication.
- A handoff naming remaining work is an assignment to every reader — fetch and check before building (the twin-duplicate class).
- A converging error fits many mechanisms — distrust closing on one; run the LOSER's fixtures through the WINNER's readout.
- A cross-repo repair is landed only when COMMITTED in the sibling, and the record must cite the sibling commit (checked with `git cat-file -e`).
- A CONCURRENCY defect passes alone BY CONSTRUCTION — "TRANSIENT, passed alone" is the wrong conclusion for that class.
- Say what you CHECKED, not who you concluded; attribute by the `from=` pipe, never the signature; put `Session: <name>, <epoch>` in the commit body.
- Consistency across runs is not reproducibility when every run shares one box and one hour.
- Decline is a correct answer; a measured negative is a result, not a failure of nerve.
