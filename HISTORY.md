## 2026-09-23 — the owner-gates menu v2 executed: schedules re-armed (row 5), pipefail residue closed (4e), toolchain batch-pin (4f), and the opt-in verdicts recorded (4a/4b/4c/4d)

The owner approved the corrected owner-gates decision table (menu v2) wholesale;
this entry is the batch record. Per-row verdicts live in their home artifacts:

- **Row 5 — CI posture (THIS repo):** the three suspended weekly schedules are
  RE-ARMED (`test.yml` Tuesdays 05:03, `full_gate.yml` Mondays 04:17,
  `feature_isolation_weekly.yml` Mondays 04:47). Grounds: this repo is public —
  Actions minutes are free, so the 2026-09-09 spending premise is gone here.
  The other half of that owner call stands unchanged: every `push` trigger
  stays `branches: [main]` (no CI on `develop` pushes). Every private sibling's
  schedules stay suspended under the same spend call. **Remaining owner
  infrastructure (not executable by an agent): the self-hosted 4090 runner for
  the ~8 private repos — needs runner registration + a box-uptime commitment.**
- **Row 4e — pipefail "47 kill-shapes awaiting triage": closed as stale.** The
  figure was the superseded first-run census; the standing state was 10 pinned
  rows. The sweep found exactly ONE unpinned residue —
  `riir-neuron-db:release/dist-repo/install.sh:43` (TAIL-KILL: an empty grep
  capture silently skipped sha256 verification) — fixed at the source in
  riir-neuron-db `c5c11b5a273e` + `6f457a6260ca` (`|| true` neutralizer + the
  emptiness fail-loud the record's own rule demands; the nested-quote spelling
  was then hoisted so the classifier reads the neutralizer). Post-fix sweep:
  **PASSED — every repo within its pins, every pinned row firing** (10
  findings, 0 unpinned, 0 unparsed).
- **Row 4f — toolchain batch-pin: landed, and the "13/20 unpinned" figure was
  stale too** — most repos were pinned since the menu was drafted; the audit's
  real unpinned population on this box was 5, all pinned this batch (channel
  1.98.1 + clippy/rustfmt, the reflex shape): seal-game-editor `8c7755b3fdbb`,
  seal-online-remaster `c47227e06fcb`, katgpt-web `d0c917f1eb60` (its default
  branch is `main`), riir-llm `ee8b6c2bee6c`, riir-viewbridge `d54fbe8ebf32`.
  riir-esp32 has no root `Cargo.toml` (not a cargo workspace root) — nothing to
  pin. ⚠ `toolchain_override_audit.py` blind-walks the three aliased `mmorpg-*`
  repos on this box (0 files walked, loud ⛔ disclosure) — the seal-* pins were
  verified on disk directly; the audit's alias seam is its own follow-up.
- **Row 4a — `certified_frontier` (Plan 580): keep opt-in; promote the day a
  production consumer lands, promote+consumer in the same window.** No gate is
  open (riir-ai Bench 822 closed them); the blocker is the no-default-consumer rule —
  an owner call, now made.
- **Row 4b — `gw_alignment` (Plan 594): do NOT promote; stays opt-in.** The
  consumer PoC measured NEGATIVE (riir-clippy `gw_corpus` Bench 083, GW
  precision@60 = base rate); reopen trigger on record: a semantic fix-shape
  embedding.
- **Row 4c — `hint_regret` Phase 5 (Plan 576): stays opt-in; the two remaining
  Phase-5 arms stay unwired until a consumer needs them** (the landed consumer
  runs behind opt-in `demo_coverage_curiosity`; a default flip would ship an
  unexercised surface in every build).
- **Row 4d — Issue 815 options 1/3: closed as answered-by-precedent.** The
  Issue-842 alias codec retired the `DOCS_GATE_KNOWN_EXTRA` marker (measured on
  the 4090: 2/33 → 33/33 with the alias file), and the AGENTS repo-count note
  already claims `mmorpg-editor`/`mmorpg-remake`/`mmorpg-remaster` under
  contract names. The membership question is settled; no 22-file registration
  ripple. This box's `scripts/repo_alias.local.txt` carries the same 3 rows
  (machine-local, gitignored, never committed).
- **Row 1 (riir-ai Plan 611), Row 2 (crates.io ×3), Row 3 (riir-dapps mainnet
  bundle): recorded in their home repos** — riir-ai (plateau formally accepted,
  decision (b)), riir-infer + riir-reflex (publication kept closed/deferred),
  riir-dapps + riir-chain (bundle sequenced under one trigger; `NO_REMINT`
  ratified SET; the second-live-mainnet question closed).

Session: owner-gates-m2, 1790121600

## 2026-09-23 — Issue 877 CLOSED (merge `b701cf564`): the main↔develop sync — origin/main merged into develop with `-s ours`, zero content delta, ancestry restored

main was NOT an ancestor of develop: three main-only commits over merge-base
`37bb9cbf8`, all transplants of develop's own work (main's own HISTORY said so —
"Cherry-picked onto main from develop `8da93896`"): `569daf98e` lthash primitive
(twin of develop `8da938961` + lint fix `b85bda6e6`), `3c844aebb` lthash wiring
(twin of the wiring inside `8da938961`), `5e2b730f2` exact_sigmoid/dot_f32_ordered
(twin of develop `5458dd69b`, evolved by `da89c386b` + `a36895e32` Issue 861 +
`9b09783d9` Issue 870). `git merge-tree` measured 11 conflicted files, 2 clean
auto-merges, 3 byte-identical adds; develop-vs-merge-tree diff was exactly those
11, and every main-side hunk existed on develop in evolved form (highwaters 876/883
vs 807/844; `lthash.rs` let-chains fix; `dot.rs` len-16 x86_64-matrix pin;
README 641/204 counts vs main's 594→595; the avx2 n-clamp and plasma_dispatch/bitcos
on the auto-merge side; the duplicate-`exact_sigmoid` trap checked and refuted).

The merge (`b701cf564`, parents `9ef11a3eb` + `5e2b730f2`) therefore changes ZERO
content vs develop — it exists to make main an ancestor. Strategy `-s ours` (not
`-X ours`, which only resolves conflicting hunks ours-ward and still merges
non-conflicting main content — the tripwire `HEAD^{tree} == HEAD^1^{tree}` reds on
exactly that mis-invocation). Pushed `--atomic origin develop HEAD:main` (one
transport, both-or-neither; main CI fires once on the final state — owner-accepted:
the 279-commit range matches every path filter, so docs_gate / full_gate (macOS 10×
minute multiplier) / required_features_touched / lean_proofs / wasm32_gate run — the
only automatic whole-repo lane this content has ever had; a red there is a true
finding about develop, not a merge defect).

Recovered-by-history note: 10 files (`.issues/747_…`–`756_…`) exist on main and not
on develop — all present at merge-base, deleted on develop under the noise-reduction
rule, their records in this file + git history. The 3 main-only commits touch nothing
under `.issues/` but `.highwater`. A future main↔develop diff will show them as
"missing on develop"; that is the record, not a loss.

Durable instrument lessons (would each have cost an hour): (1) **merge-tree hashes
are spelling-bound** — `git merge-tree --write-tree A B` labels its conflict markers
with the ref spellings passed (`<<<<<<< HEAD` vs `<<<<<<< develop`), so the blob
bytes and tree hash are functions of spelling, not just content; three spellings of
one commit pair produced three different trees, each deterministic. Never quote a
merge-tree hash as a pin — quote the FILE SET. (2) **a pin measured before the commit
carrying it is already stale** — the develop tree hash `4ad737a9…` was measured,
then written into the issue-amendment commit, which itself moved the tree; the
surviving form is self-referential (`HEAD^{tree}` vs `HEAD^1^{tree}`), which cannot
go stale. Adjudication: Claude verdict, 3 rounds (REVISE → REVISE → AGREE), round 2
catching the stale pin and the merge-tree variance mechanism.

## 2026-09-23 — Issue 867 CLOSED (+ the issue): the non-hidden-state canonical-AST construction — G5 returned NO attributable signal; instrument-broken, not claim-refuted

T1–T3 all landed; T4/T5 are dead branches by their own conditions. T1
`65b199b0` (the 38-bin `source_features` AST-histogram extractor, opt-in
`canon_source_features`) and T2 `67884461` (`SourceFeatureAdapter` ridge
fit + zero-alloc apply) shipped in `katgpt-canon`. T3 — the decisive G5
cross-arch gate — ran riir-train-side at [Bench
605](../../riir-train/.benchmarks/605_issue567_g5_source_features_gate.md)
(harness riir-train `48623505`, corpus `49a4a72f`): PRIMARY
+0.50..0.55 at k∈{2,4,8,16}, **but the fit-time shuffle null manufactures
+0.41..0.44 on its own** (within ~1σ of PRIMARY at every k, clearing the
+0.3 signal bar by itself) — no rung is attributable to the construction.

Why the file closes: T4's own condition ("fires only if the control passes
AND G5 fails") did not fire — the control failed, which is *instrument
broken, never claim closed* (the issue-825 clause); T5 needs T3's >+0.3
attributable rung, which does not exist. Both remaining tasks are
permanently non-executable; the tracker is complete.

Durable findings kept: the models share a real **aggregate** contrast
direction (AGGo to +0.44 at k=16, noisy) with weak pair-specific observed
correspondence (OBS +0.16..0.22 vs its own null +0.11..0.18). Reopen paths
live in riir-train Bench 605 §Verdict(6): an observed-level gate with the fit-time
null mandatory, an intervention study, and/or a corpus at
gemma-reliability-adequate scale. Reopen authority: Research 459 (CLOSED
2026-07-27 — reopens only on a non-hidden-state construction). Proposal 010
remains the G5-bar authority document.

## 2026-09-23 — Issue 875 T3 CLOSED (+ the issue): time-annealed sampling ranges + the closed-form truncation predicate (Bench 883)

The last open task of the PFD modelless arm. `TimeAnnealRange` + the
zero-terminal-weight truncation predicate (`ε = ((T−t_cut)/(T−t_min))²`,
inverse `t_cut = T − (T−t_min)·√ε`) shipped in `horizon_weights`; the
`dllm_solver` seam (`annealed_renoise_range` / `renoise_level_skippable`)
behind the combined gate; the quality gate ran CROSS-REPO on the riir-train
C9 toy (TRangeMode remap, fixture consumed in place — no second copy of
the frozen bytes; the deviation from C9's "vendors" note is recorded in
the bench). Headline numbers (release, gate_full): flat 0.4084 = C9's
recorded PFD (behavior-preserving refactor); AnnealPlain +9.0% W1 for
9.37% mass dropped — **the truncation corollary's cost law measured**;
AnnealRenorm +4.4% (within the not-worse bar, ring2 kept) with the honest
regime boundary: the two-ring toy has no fine detail to exploit, so the
anneal is safe-not-a-win HERE and the paper's fine-detail claim routes to
riir-train 569 C5 (mechanically unblocked). Validated in an 8-worktree
redirect harness (the shared main checkouts were sibling-blocked mid-
flight; the harness runs the REAL katgpt-core symbol — provenance in the
bench). Issue 875 now CLOSED: all five tasks, all arms OPT-IN per T5.

## 2026-09-22 — Plan 607 ACCEPTED + T0a/T4a/T0b landed: the modelless game-decision lane's first three tasks (substrate gate, the laya-Tetris enumerator, the G1-oracle fixture)

The owner accepted the plan ("607 accepted") and the first three tasks of
the co-developed ordering landed the same day. The lane's thesis: laya's
game protocol (code does the arithmetic → templated English sentence →
BERT reads it) is exactly the closed-grammar regime where corpus-limited
scoring is lossless — the plan's job is to PROVE the modelless lane wins
on latency/deployment by construction and on accuracy by protocol
choice, with a measured GOAT gate.

- **T0a (substrate-first)**: all six named substrate pieces read and
  adjudicated (§Substrate check in the plan). Headline: T1 will be a NEW
  `state_option_scoring` module (capability-named) consuming
  `exact_sigmoid` + `float_order::cmp_for_max` + the unit-normalize
  idiom; reflex `engine.rs` route_terms is the shape being upstreamed;
  the fitters are the T3 precedent; the drafter stays OUT of the hot
  loop (R3).
- **T4a (enumerator + grammar)**: `examples/tetris_01_state_enum.rs` +
  the shared `examples/common/tetris_sim.rs` (#[path] module, the
  tests/common precedent). 120 deterministic states (12 archetypes × 7
  pieces + a 36-state seeded Dellacherie-greedy ladder), 2,660 options,
  byte-identical dumps. **Protocol correction from the live laya page**:
  their Tetris is one sentence PER SPOT → P(clean) per spot → code
  argmaxes (noul-shaped, not choice-over-34) — number WORDS in-protocol.
  **The v1→v2 grammar lesson**: the two-clause grammar produced 39
  distinct sentences and the first oracle run tied 66/120 states at
  identical p_clean (argmax degenerating to the index tie-break); v2
  adds landing-side + resulting-height clauses (245 distinct sentences),
  cross-sentence ties → 0, residual 33 exact ties all SAME-sentence
  (honest equivalences on symmetric boards). Pre-clear law discovered:
  uniform-height floors are pre-full rows — every heights archetype
  carries a 0-height shaft column.
- **T0b (oracle fixture)**: the generator committed to riir-reflex FIRST
  (`laya_oracle_batch` @ `e4bf657`, pushed — a GENERIC game-free batch
  oracle; T5's arenas reuse it), then 2,660 english-checkpoint noul
  forwards (396 s CPU) committed as the self-joined fixture
  `tests/fixtures/tetris_oracle_laya_en_v2.jsonl` + provenance README;
  byte-identical re-run from the committed generator verified (the
  katgpt-device-verify rule held end to end). Oracle sanity: p_clean
  spans 0.027–0.850; Spearman(p_clean, Dellacherie) mean +0.365
  (positive 94/120) — the checkpoint reads the sentences and prefers
  clean placements; the oracle is meaningful, not constant-picking (the
  reflex Issue-004-T7 trap, checked BEFORE the fixture was frozen).
- **Next**: T1 + T4 co-developed (the primitive + the arena over the
  fixture), first GOAT reading gates {T2, T3}; T5 (Flappy + lanes)
  precedes any default-on consideration (R4).

## 2026-09-22 — Issue 874 closed: G8 re-founded on real claims — the dead speedup gate became a bit-contract pin + a relocated, executing throughput floor (two sessions, one issue)

The Issue-871 T5 fallout, closed in one evening by two complementary
sessions: the census session (T3, `ec345a8c`+`e880d97c`+`6c2d8b04`) read
all 16 workspace `*_vs_scalar` gate/bench sites one by one and proved the
consolidation trap is ONE gate — g8 itself — repairing two unrelated
riir-ai by-catches on the way (`f04df9241`); the repair session (this
entry) took T1 through the verdict ping-pong (round 2 REVISE folded in)
and landed T2.

- **Why the old gate was dead in both directions**: G8 asserted ≥1.5×
  "SIMD vs scalar" speedup, but both routes run the identical double loop
  through the same `dot_8wide` (the Plan-271-era DRY consolidation), a
  strict ordered kernel with no SIMD left on any target (Issue 871 T5:
  0 packed float-math crate-wide on x86_64 at both compile arms; aarch64 =
  packed muls + ordered scalar adds — "there is no strict vectorized
  add"). The timed "SIMD" arm additionally ran the stabilize pass, so the
  ratio was structurally <1 and the gate deterministically SKIPped
  (returned Ok). Pre-consolidation PASS entries (3.01× NEON, 2026-06-14)
  were real; post-consolidation ones were timer noise between identical
  binaries.
- **The verdict** (3 rounds, final AGREE): Option 3 — retire the relative
  claim. Option 1 (naive scalar arm) is refuted on crate-independent
  ground: a strict ordered f32 reduction cannot vectorize without
  changing the bits, so a naive-vs-dot_8wide gate compares identical
  codegen — a third fake contrast. Option 2 (re-point at serial-vs-rayon)
  substitutes a different claim; the only real speed contrast in this
  crate is strict-vs-`algebraic_dot`, which has its own lane (feature
  `algebraic_dot`, Bench 871).
- **G8a — `g8_route_bit_agreement`** (new): exact `to_bits` equality over
  16 384 outputs between the two public routes — pins the documented
  "bit-identical by construction" contract; the failure message names BOTH
  red causes (one route's summation order changed, or the internal
  `inv_sqrt_d` derivation moved — the routes derive it differently).
- **G8b — `g8_throughput_floor`** (new): the absolute floor RELOCATED from
  the in-crate `test_simd_throughput_smoke` — same size (n=8, t=512, d=64),
  same 5 ms ceiling, but on a lane that actually executes (bench_271 runs
  in the x86_64 execution matrix integration cell; the in-crate copy had
  no test_gate row, no matrix floors row, and a debug-skip that bypassed
  dev lib runs). Defence upgraded to `best_of_us` min-of-200 +
  argument+result black_box. Measured 50.3 µs/call on this box — the bar
  is a regression floor (100× headroom), not a speedup claim. The in-crate
  duplicate is deleted in the same commit (one claim, one home).
- **Validation**: bench_271 10/10 in release (G1–G7 untouched, PASS);
  crate lib 121 (smoke removed); clippy clean on both surfaces;
  `timed_region_guard_gate` PASSED (the new region carries the best_of_us
  defence — no new unguarded row); no test_gate/matrix pins referenced
  bench_271 or the crate's counts.

Adjacent observation recorded, out of scope: G7's release fallback prints
"100000 calls in 0ns" — constant-folded work in a print-only sanity path
(the load-bearing G7 alloc gate is the debug TrackingAllocator one).

Fixed in the follow-up commit (2026-09-22): the fallback consumes the result
inside the timed region (black_box both ends, the G8b defence stack) and
asserts a loud zero -- the print reports real work.

Issue file CLOSED IN PLACE (status CLOSED, all tasks terminal) — removal
deferred to the next backlog-clear pass rather than racing the census
sibling, which was still appending at close-out time (its `27b48552`
T3 addendum landed mid-rebase: a second census pass converging on the
one-gate verdict + bench_256's stale pre-808 "SIMD (ns/call)" report
labels repaired). The file keeps the full 16-site census table; durable
summary: trap = ONE gate; healthy population defended three independent
ways (bench_148's anti-vectorization assert, fast_bpe's fallback-catch
floor, bench_578's same-code-aware loud skip).
The issue file was removed 2026-09-22 (noise rule, this pass; the 16-site census table stays recoverable via the git history of `.issues/874*`).

## 2026-09-22 — Issue 873 primitive B landed twice in one evening: the twin-duplicate resolution (third of the class)

The 873-handoff context-overflow relay produced the exact Issue-825 shape:
two sessions picked up "primitive B (`rate_control`) remains" from the same
handoff summary and both implemented the dual-EWLS effect-size controller
independently. The sibling's landing (`ee01f1598`, 21:12 +0700) reached
origin first with a complete GOAT-gated port (f32 internals + an exact
algebraic rolling-origin recenter every 256 ticks, σ-floors against
cancellation, predict-next jump detection, Bench 875, catalog + test-gate
rows); the second session's push arrived minutes later, was rejected
non-fast-forward, and — per the Batch-169 twin precedent — the unpushed
duplicate was `git reset --hard` away (reflog only), origin's commit
 canonical. The duplicate's verifier then ran the winner's full gate set
on the reset tree (13/13 G1, G2 bench PASS, docs counts 637 consistent)
and read the module against the source before standing down: no material
defect; the differing constants are documented house defaults
(λ/gain/width sextet source-pinned; T_MID/EFFECT/clamps/jump thresholds
pinned at first landing with provenance stated).

- **The class, third instance** (riir-clippy Batch 169 twins → Issue 825
  negative-result → this): a handoff summary that names remaining work is
  an ASSIGNMENT to every session that reads it. The 825 lesson held:
  before pushing any primitive, `git fetch` + check whether a sibling
  already shipped it. Push rejection with your own commit subject on
  origin is the twin tell — resolve by dropping yours, then VERIFY the
  winner (its tests, its bench, its docs surfaces) rather than walking
  away; the disagreement between two independent ports of one source is
  the cheapest oracle either session gets.
- **Standing follow-up (minor, on-record) — RESOLVED same day:** the
  landed module's caller-facing sign convention (rising `val` = improving,
  `e = +slope/σ` — the OPPOSITE of mini-AGI's falling-loss convention) is
  pinned by test fixtures and now stated in the module doc's caller terms
  (a "Sign convention" section naming the raw-loss trap and the `-loss`
  fix). Originally to ride the first consumer's commit; landed early
  because the convention is already fixed by the shipped API and the
  first consumer (riir-train Plan 416 Phase 2) is QUEUED behind a quiet
  4090 window — a documented caller-facing trap should not wait on an
  unscheduled plan. The consumer still decides its own feed direction;
  it just reads the doc instead of the fixtures.

## 2026-09-22 — Issue 871 closed: the `algebraic_*` A/B measured end-to-end — feature-gated `algebraic_dot` ADOPTED (owner verdict a′), the in-crate codegen truth repaired, T6/T7 deferred on-record

The full arc, three sessions, one issue: T1–T3 measured (`e2f77e70`), T4
decided by verdict ping-pong (3 rounds, final AGREE — the (a′)
mechanism-only GO), the adoption lane landed (`c108f2fe` + `fbe86280`),
and T5's in-crate codegen verification closed the loop (`2331961c`).

- **What shipped (T4, `c108f2fe`/`fbe86280`):** `katgpt-attn-match`'s opt-in
  `algebraic_dot` — the Bench-871-measured twin of `dot_8wide` (1.77×–8.8×
  at both x86_64 compile arms, G1 accuracy IMPROVED vs strict on
  cancellation-realistic data) behind a feature flag with zero consumers
  wired; strict stays the sole default and every correctness gate still
  exercises it. Six module tests pin the lane, incl.
  accuracy-not-worse-than-strict and NaN no-poison (the `algebraic_*`
  contract deliberately omits `nnan`/`ninf`). The wiring precondition rides
  the feature row itself: the first argmax-bearing consumer requires the
  real-logits retention walk (the Issue-750-T3 shape) — the issue's own
  1024-trial fixture walk was low-power (9 near-tie events).
- **The mechanical ban (`c108f2fe`):** `algebraic_div`/`algebraic_rem` are
  banned outright — `arcp`/remainder semantics are a far larger numerics
  change than add/mul reassociation. Enforcement is the tracked
  `scripts/algebraic_op_ban_gate.py` (docs_gate CHECKS row, ceiling 0 over
  a floored walk, planted-source predicate arms, masker imported from
  `platform_dead_code_audit`) — a prose ban is not enforcement.
- **T5's codegen truth (`2331961c`):** 0 packed float-math instructions
  crate-wide at BOTH x86_64 arms (SSE2 baseline and `+avx2,+fma`; 122
  scalar `mulss`/`addss`) — every "auto-vectorizes" doc claim was false on
  x86_64-windows. The aarch64 cross-emit twin: packed `fmul.4s` products +
  an ordered scalar `fadd` chain, bit-identical, NO `fmla` — the 2026-07-29
  M3 "optimal fmla sequence" story was never codegen-true, and the 1.26×
  8-accumulator refutation stands with its mechanism attribution corrected
  (bounds-check confound, not accumulator count). Nine claim sites repaired
  across the crate. The strict-side repair is REFUTED: the serial add order
  IS the cross-arch bit-equality — there is no strict vectorized add to
  build.
- **The ill-conditioned G1 arm (T4's landing precondition, `2331961c`):**
  algebraic ≥ strict on BOTH data classes (ortho: 4.58e-9 vs 8.25e-9 mean
  err, 191 vs 256 sign-flips; rank-deficient dup: 6.6e-18 vs 5.4e-9, 0 vs
  245) — the f978a20b Cholesky reassociation counterexample does not
  reproduce for the plain dot kernel.
- **Issue 874 stays OPEN** (filed from the T5 verification):
  `g8_simd_vs_scalar` compares two routes that BOTH go through `dot_8wide`
  — a standing GOAT-gate inertness on a DEFAULT-ON feature, live
  independent of this lane.
- **Deferred on-record:** T6 (re-run Bench 871 on the M3/aarch64 arm when a
  lane exists — this box cannot reach the M3) and T7 (the riir-clippy
  `rust_perf` suggest-only rule lands with the FIRST real consumer, not
  before — the healer must not emit numerics-changing rewrites with no
  consumer to retain against).
- **The three-way session collision** (one lane of this issue's history):
  the T4/T5 sibling's duplicate `algebraic_dot` implementation was yielded
  to the landing lane after a TOML duplicate-feature-key error caught it at
  manifest load (Issue-665 discipline); its issue-file note records the
  yield explicitly (recoverable at `2331961c`).

Validation at the adoption lane: clippy `-D warnings` at both feature
states, 6/6 module tests, the ban gate green over 2533 tracked files,
`count_features`/`docs_gate_checks_sync`/`check_validation_gate`/
`cargo_comment_audit`/`bench_doc_audit` green. Close-out validated:
`numbering_gate` + `issue_citation_gate` (citations to "Issue 871"
resolve through git history — `removed_by_number()`'s majority case).

Issue file removed per the noise rule (content recoverable from
`2331961c`). The measured record is [Bench 871](.benchmarks/871_algebraic_dot_ab.md).

## 2026-09-22 — Issue 872: SIMD bitstream whitespace splitter for `encode_into_pretok` (the bitcannon-class port) — scan 1.60×/1.69× measured, bit-identity by differential, four live bugs caught by the harness

The deferred item from Bench 191 §Phase 3 ("SIMD pretokenization — a
meaningful project"), unblocked by prior art: HF `tokenizers` v1's
"bitcannon" blog post (2026-09-21) demonstrated the bit-parallel splitter
technique class on **stable** Rust — the exact blocker Bench 191 cited was
upstream's nightly `portable_simd`, and v1 showed it isn't needed.

Landed in `katgpt-tokenizer` behind the existing opt-in `fast_bpe` feature:

- `fast_bpe/simd_split.rs` — `WhitespaceSplitter` iterator (zero-alloc,
  word events borrow byte ranges from the input; the per-char
  `pretoken_bytes` accumulation buffer is deleted). 16B SSE2 (x86_64
  baseline) / 32B AVX2 (**runtime-probed** `#[target_feature(enable)]`
  kernel — the `shipped_target_feature_gate` law) / 16B NEON (aarch64,
  u64-lane SWAR movemask) / scalar elsewhere through the same iterator.
  Bytes ≥ 0x80 classify via the exact scalar `char::is_whitespace`
  predicate — full Unicode `White_Space` bit-identity by construction.
  Identical-byte ASCII ws runs coalesce to one vocab lookup per run.
- Measured (loaded i7-13700K, median of 9 interleaved rounds): **avx2 1.60×
  vs the scalar-mask level, 1.69× vs the old per-char loop shape**,
  scan-only. NEON execution belongs to the M3 lane (same differential unit
  tests); the aarch64 NEON arm typechecks via cross `cargo check`.
- The differential harness caught **four live bugs** before any green run:
  (1) `u8::is_ascii_whitespace` excludes vertical tab — Unicode `White_Space`
  on ASCII is SIX bytes; (2) `cmpgt(v^0x80, 0)` misses exactly `b == 0x80`
  (bx == 0) — non-ASCII is "the raw byte is negative as i8"; (3) `1u32 <<
  32` release-wraps to a zero all-ones mask (AVX2 width) making every
  non-matching ws chunk read as all-matching; (4) a multibyte ws char at
  word end was swallowed by the advance-past-decode. All four are pinned by
  tests now. Full record: Bench 872, Research 580.
- Gates: lib 25/25; `fast_bpe_goat_simd_split` 3/3 (new); pretok GOAT +
  hypothesis + **G4 zero-alloc audit** all green (the rewrite removed an
  allocation site); clippy clean at default/`fast_bpe`/all-features;
  wasm32 + aarch64 cross-checks pass. Pre-existing, NOT this change:
  `fast_bpe_goat::g2_perf_smoke_per_call_short_input_documented_regression`
  fails on this box at clean HEAD too (isolated-worktree verified —
  box-sensitive 16MB-per-call allocation gate, M3-calibrated).
- fast_bpe stays opt-in (Bench 191's Phase 3 deferral stands). Issue file
  removed per the noise rule.

## 2026-09-22 — Issue 870 closed: distance_abstain's rationale-free sigmoid copy → exact_sigmoid delegation, measured 3-ULP envelope, bench_845 GOAT re-run identical

The 09-22 substrate-first Mode 2 audit (detection commit `683d06d8`) found
`distance_abstain.rs:47` carrying a local single-branch
`1.0/(1.0+(-x).exp())` with no rationale and no pin — in the same crate as
three sanctioned patterns (`closure/bridge.rs` and `d2f` delegate to
`fast_sigmoid` with rationale; six modules consume `exact_sigmoid`). The
copy-class family (riir-neuron-db Issue 611 / riir-chain Issue 156 precedent).

The fix (this commit) is the Issue-156 permanent-pin pattern:

- **Delegation**: the local `fn sigmoid` now calls `crate::exact_sigmoid`
  (the libm two-branch reference form), with the divergence envelope
  documented in-source.
- **Measured, not assumed**: the first pin draft asserted ≤1 ULP and RED —
  the honest re-measure (0.001-step sweep, standalone probe) found **max
  3 ULPs for x<0** across [−20, 20] (x=−4.851 inside the gate's validated
domain, x=−16.743 in margin); bit-identical for x≥0. The pin
  (`sigmoid_delegation_matches_frozen_legacy_body`) freezes the legacy body
  inline and bounds at the measured 3 ULPs — future form drift reds it.
- **GOAT re-run**: bench_845 at the delegated form — G1–G5 ALL PASS with
  readings numerically identical to the recorded baseline (W1 fused AURC
  0.1894 vs score 0.2289, Δ(ρ=30%) +0.0097, 8/8 replicates positive;
  G3 negative control vanishes as designed). 3 ULPs on values ≲5e-7
  relative move nothing at the bench's printed precision.
- The bench's own `sigmoid` (the W1 world-model formula,
  `sigmoid(1.2·logit p − 0.2)` from Research 576 §2.1) is the synthetic
  world definition, not the gate — untouched.

Validation: `cargo test -p katgpt-core --features distance_abstain --lib
distance_abstain` 8/8; `cargo clippy -p katgpt-core --features
distance_abstain --lib` clean; bench_845 GATES PASSED. Issue file removed
per the noise rule (content recoverable from `683d06d8`).

## 2026-09-22 — Issue 869 T5: the mini dllm lane goes per-layer honest end-to-end — the gradient check caught a live backward bug the loss-decreases gates never could

The same-day follow-up landing closing Issue 869: training, eval, and decode
now all honor `config.n_layer`, and `D2fContext`'s decode depth defaults to
it (the pre-T5 default of 1 — the lane's accidental effective semantics — is
now the explicit `set_decode_layers(1)` truncated-trunk posture).

- **Training lane** (`src/dllm/mod.rs`): `ForwardSaveContext`/
  `ForwardActivations`/`TrainingGradients`/`BackwardContext` carry
  `n_layer` × `block_size`-capacity planes (mirroring the decode kernel's KV
  layout); `forward_save`, `forward_save_set_causal`, `backward` (three-phase
  per layer, reverse order, `d_h_next` stream handoff), `sgd_update` all
  per-layer. Eval/inference forwards generalized in lockstep
  (`forward_bidirectional_positions_into` + `BidirectionalContext`,
  `forward_block_causal_positions`, `forward_set_causal_positions` — the
  last one feeds bench_602's `CpuSetCausalForward`). All bit-identical at
  `n_layer == 1` (every plane index 0; the layer loop degenerates) — proven
  by the full pattern-lane battery re-running green.

- **THE FINDING — a real backward bug the historical gates could not see**:
  the pre-869 `!is_masked[p]` skip in backward Phases 1/2 is valid ONLY at
  the readout layer. Inner-layer streams receive gradient at UNMASKED
  positions (downstream attention reads the k/v derived from them at masked
  queries); skipping them corrupted every layer-0 attention-path gradient
  under partial masking — measured ~10-150% per-element error (attn_wv sign
  flip), EXACT under full masking, which is exactly why every historical
  loss-decreases/non-zero-grads gate passed. Caught by the new
  `two_layer_backward_matches_finite_differences` (analytic vs central
  finite differences, every grad family, at BOTH 1 and 2 layers — 1-layer
  arm passes at <0.7%, proving the rewrite preserves the verified
  single-layer math). Fix: `last && !is_masked[p]` — at `n_layer == 1` the
  only layer is the readout, semantics bit-identical.

- **`bench_602_ar_ness_cross_tab`'s calibration is model-class-specific** (the honest negative):
  post-honest-depth, the gap-predictor's AR-drag DIRECTION is within
  training-seed noise at the micro scale (seeds 42-45: AR−UNI ΔALR swings
  ±0.13 around ~+0.03, 2/4 invert; the pre-T5 PASS was one seed's coin
  flip), and the 09-20 `ORDER_STATS_TO_W_TABLE` (measured on the effective-
  1-layer class) no longer tracks NLL-optimal w within the 0.95 retention
  floor (worst 0.914). bench_602 g3 now trains 2 seeds/regime, REPORTS
  direction + retention with the caveat inline, and hard-gates liveness +
  a catastrophic-derail floor (chosen ≤ worst-fixed ×1.05). Re-open at the
  Bonsai-scale trunk.

- **Text-lane dividends**: bench_601's corpus-honesty margin widened to
  0.39 (NLL 2.499 vs unigram 2.888, bar 0.15 — the 2-layer-honest model
  genuinely learned more bigram structure); 809's seam-parity pin
  byte-identical through BOTH generalized set-causal forwards; 817 2/2;
  602 3/3. New pins: fwd-vs-inference bit-identity at depth, training-moves-
  layer-1, kernel-vs-positions-forward consistency at depth 2, the gradient
  check (both depths), decode-default = n_layer.

Gates: root lib 211/217 (default/set_diffusion); katgpt-forward 131/167/
180/181 (default/set_diffusion/probe_guidance/+new pin); dllm 27/27;
pattern lane bit-identity (bench_600 ×2, dmax_spd, d2f_verifier,
diffusion_sampler, probe_guidance_goat + headroom + alloc); four text
benches green; `full_gate.sh --allow-partial-platform` — every layer that
ran clean (standard Windows PARTIAL). Files: `src/dllm/mod.rs`,
`src/dllm/tests.rs`, `crates/katgpt-forward/src/{forward_positions,
forward_set_causal,d2f_context}.rs`, `d2f/tests.rs`, `tests/
bench_602_ar_ness_cross_tab.rs`, `.issues/869_multi_layer_d2f_taps.md`.

## 2026-09-22 — Issue 869: multi-layer D2F decode + taps at depth — the Issue-865 Bonsai-scale unblock lands; bitcos x86_64-lane clippy debt repaired in passing

Executed the riir-train recipe row's mandate ("Bonsai-scale probe training is gated on the
katgpt-rs multi-layer D2F kernel extension — file there first"; Issue 865's re-open condition,
Bench 847/850):

- **Kernel (T1)**: `forward_block_causal_with` generalized over
  `D2fContext::decode_n_layer` — per-layer KV planes (`l * block_size * kvd`),
  chained residual stream (`h_0 = rmsnorm(emb)` preserving the lane's
  double-norm quirk at layer 0; single-norm inputs above), logits only at the
  final layer. **Depth defaults to 1, not `n_layer`** — the finding (below)
  is that the whole mini lane trains/evals/decodes single-layer regardless of
  config, so depth 1 preserves every pinned gate's world exactly; multi-layer
  decode is an explicit `set_decode_layers(n)` opt-in.
- **Taps at depth (T2)**: `set_probe_tap_layers(&[usize])` (sorted, deduped,
  validated against the decode depth — a tap deeper than the decode can
  never be written, so asking for one is a loud panic), layered
  `probe_tap_flat` (`[slot][pos][dim]`), `ProbeCtx { tap_layers, tap_plane }`,
  `WeakLogitProbe::tap_layer()` default method, and `set_guidance`
  install-time validation (a probe declaring a layer the context does not
  capture panics with the remedy — the old constructor-time `tap_layer != 0`
  rejection is REMOVED; that is the unblock: artifacts pinned to any layer
  now load, and `MlpWeakProbe` reads its own plane). Default tap set `[0]`
  keeps every existing artifact/gate byte-compatible (slot 0 at offset 0).
- **The finding, recorded in the issue**: `micro_dllm_text()` declares
  `n_layer = 2` ("the smallest capacity that learns English bigram structure")
  but the ENTIRE mini dllm lane is single-layer end-to-end — training
  (`forward_save` ×2, `backward`, `sgd_update`, `TrainingGradients`), eval
  (`evaluate_accuracy`'s bidirectional forward), and decode all index
  `layers[0]` only. Layer 1 is allocated, never trained, never read. The
  T5 follow-up (open, its own landing) migrates training/eval per-layer,
  then flips the decode default to `n_layer`, then re-runs the four
  micro_dllm_text benches. NOT required for the Bonsai-scale GPU lane
  (frozen trunk, GPU extraction — the artifact consumer + kernel contract
  were what that lane needed).
- **Verification**: bit-identity at depth 1 proven by re-run of every pinned
  gate (fixture G0 + goat 5/5 — the committed fixture still pairs with the
  trunk through the NEW kernel; headroom study verdicts unchanged; bench_601
  3/3, bench_809/817/602, bench_600 ×2, dmax_spd, tri_mode sampler, ugc_g1b,
  dllm lib 23/23) + 10 new tests (kernel-level ungated + tap-level gated:
  planes-differ, capture bit-identity at depth, λ=1 identity at depth with a
  deeper-tap artifact, install/depth/tap-set validation panics, exact plane
  reads, pre-869 structural bit-identity pin, committed-prefix at depth).
  `scripts/full_gate.sh --allow-partial-platform`: every layer that ran
  clean.
- **Rider — pre-existing bitcos x86_64-lane clippy debt repaired**: the full
  gate's first run red on `katgpt-types/bitcos.rs` + `simd/bitcos.rs` +
  `bench_864_bitcos_goat.rs` (Issue 864's landing, 9 findings, ALL
  arch-gated — invisible to every lane this box's default clippy runs, the
  Issue-819 class): `manual_isolate_lowest_one` ×4 → `.isolate_lowest_one()`,
  `needless_range_loop` ×2 → slice iteration/`fill`, `manual_is_multiple_of`
  ×2 → `.is_multiple_of()`. Behavior-identical mechanical rewrites;
  bitcos tests 13/13 green on the avx2 lane post-repair.

Issue: [`.issues/869_multi_layer_d2f_taps.md`](.issues/869_multi_layer_d2f_taps.md) (T1–T4
landed; T5 open). No bench — no perf claim (depth 1 is the exact old op sequence; the GOAT/perf
question belongs to the scale lane's own gate, pre-wired as Bench 847's inverted bars).

## 2026-09-21 — Issue 864 closed: BITCOS tier ships opt-in — footprint PASS, latency honestly LOSES on this host (Bench 846)

Executed `Issue 864` (filed from
[Research 577](.research/577_BITCOS_Distribution_Adaptive_Ternary_Layout.md) the same day):
`bitcos` feature in katgpt-types — the distribution-adaptive ternary container (presence bitmap +
compacted neg-sign stream, rate 2−z bits/w + f16 scale), the z-meter (`zero_density_report`), and
three GEMV consumers (scalar bit-identical to the bit-plane reference; 256-entry LUT
presence-nibble×sign-window — the GPU-portable pdep-free mechanism; pdep+SWAR AVX2 arm
runtime-probed AVX2∧BMI2) + the roofline dispatch `should_use_bitcos(z, γ, β)`.

- **G1 PASS** (roundtrip bit-exact incl. scales; scalar/LUT bit-identity chains), **G2 PASS**
  (footprint beats BOTH tiers at every z above the 0.375 crossover — 0.950/0.892/0.854× vs trit —
  and is honestly LARGER than trit below it, asserted), **G4 PASS** (alloc-free kernels).
- **G2b(a) FAIL → opt-in by the issue's own promotion clause**: >L3 streaming at z=0.5 measures
  bitcos 0.767× vs the shipped bit-plane SWAR / 0.846× vs trit. Measured roofline: γ = 1.23 B/ns
  (bitcos decode) < β = 1.96 B/ns (achieved streaming) — instruction-bound on this 13700K host,
  the paper's Lunar Lake class reproduced on ours. The dispatch refuses, and the
  dispatch-verdict == gate-outcome assert pins that the predicate tracks the knee.
- **Two codec bugs caught by the gates pre-timing**: pack must pext (compact), not pdep (scatter)
  — strayed high bits tripped `is_canonical`; and the scalar kernel's pos plane is `p & !neg`,
  not presence (a negative weight has both bits set). Research 577's "compacted pos bits" wording
  is inverted against its own 0=+1/1=−1 convention — neg-compaction ships, pinned in-code.
- Record: [Bench 846](.benchmarks/846_bitcos_goat.md); feature-count claim sites updated
  626→627; the riir-gpu CUDA LUT-arm pointer stays recorded (now with an in-tree reference).

# HISTORY.md — katgpt-rs

Historical record moved out of `AGENTS.md` (2026-09-06 compaction, from
commit `1801c0ab`) so agent context stays small. Nothing was deleted: every
section below is preserved verbatim from the pre-compaction `AGENTS.md`.

## 2026-09-22 — Issue 868 closed NEGATIVE: engram-fused PUCT G5 FAIL — the evidence gate worked, and that is why nothing happened (Bench 848)

Executed `Issue 868` (record lives in git
history; file removed per the noise-reduction rule) via
[Plan 605](.plans/605_engram_fused_puct_poc.md): Proposal 013's engram×PUCT fusion
implemented end-to-end — `engram_puct` feature (opt-in, native-gated, optional katgpt-core
dep), `engram_fuse.rs` (TT-key packing `(board, ko, to_play)` → 4 words; `MinedTable`
BLAKE3-committed stats artifact with strict key-sorted determinism; count-sigmoid evidence
gate σ((n−8)/4); one-shot damped Q-init `visits=1, total=gate·v̄`; bounded prior sharpening
γ = exp(2·ln2·(b−0.5)·gate) ∈ [0.5, 2]), miner + arena examples (required-features), and the
GOAT gate tests. All fusion sites cfg-gated at item granularity — the feature-off compile
is source-identical, and T2.1 measured **size-identical** (381,353 B raw wasm pre/post;
sha differs only via cargo's `-C metadata` permutation) with zero katgpt-core in the
default wasm32 tree.

- **G5 FAIL — 296/616 = 48.1%** head-to-head (paired, both colours per seed), Wilson
  one-sided 95% lower **44.8%** < 50%. Budget arm (fused b25 vs plain b50): **35.7%** —
  memory does not compensate halved budget. Independent-opponent control: fused 93.0% vs
  plain 97.0% vs GREEDY — **−4.0 pp, not significant**. G2: **293 ns/read** vs the <100 ns
  bar (16 runtime-modulus divisions; ~1% of a forward pass, gate-absolute FAIL).
- ⛔ **The control arm's first numbers were corrupted by a reward-inversion bug** (White-game
  rewards counted for the PUCT arm — both arms forced to ~50%, the 51/51 "tie" was the
  artifact). Caught by checking the output against the harness reference (85–94% vs
  GREEDY); fixed + re-measured. The h2h and budget arms inverted explicitly and were
  never affected.
- **The interpretable negative (T1.3 sanity gates did their job)**: the memory FIRED at
  98.4% of 1.06 M lookups, but 100% of the 16,930 mined positions have n < 4 (only 136
  repeat visits in 17,066 plies — 9×9 self-play barely transposes), so the count-based
  gate — the issue's own anti-rumor rule — correctly damped 99.98% of rows to ≤0.18
  strength. "Memory doesn't help" is now separable from "memory never fired": it fired
  everywhere and could not legally trust what it read. Collisions: 22.0% of entries share
  ≥1 of their 16 slots (2²⁰ slots) — real but mooted by the gate.
- **Gates that PASSED**: G1 (empty table ⇒ bit-identical to feature-off, 40 positions +
  2 full games), G4 (zero-alloc read; counting allocator, mutex-serialized because the
  counter is process-wide), G6 (deterministic build + round-trip + tamper refusal), and
  the Q-init direction arm (a v̄=+1 row strictly suppresses the move leading into it).
- **Re-open conditions**: a transposition-dense domain, neighbourhood generalization (the
  M-MCTS mechanism hard-hash routing lacks by construction), or a much larger mining
  corpus + re-mining cadence (`EngramHotSwap` exists). Record:
  [Bench 848](.benchmarks/848_engram_puct_arena_g5.md); Proposal 013 status line
  updated to MEASURED NEGATIVE.
Operational rules live in `AGENTS.md`; removed issue files: git history.

contents: modelless-first canonical-failure story · full-gate narratives ·
docs-gate descriptions · cfg-gated / required-features / percentile audit
histories · staged-set + shared-target-dir narratives · feature-flag rule
history (lossy surface, Report the Floor, Plan 467) · the Repo count
paragraph's drift history · the resolved issue log.

## 2026-09-21 — Issue 858 closed: g8's arch-conditional bar gets its aarch64 executing lane (PERF_ROWS)

**Status: RESOLVED + REMOVED (this commit). Full issue record: git history
`.issues/858_g8_cached_faster_than_uncached_is_RED_on_aarch64.md`.**

**The finding (2026-09-19):** `g8_cached_faster_than_uncached` in
`tests/belief_drafter_goat.rs` was red on the M3, reproducibly, ALONE —
median ratio 0.6844–0.7111 over 5 runs against the then-universal
`ab.median < 0.5` bar ("at least 2× faster", Plan 217's claim, calibrated
at `dd8dadbba` on x86_64: 8 runs 0.39–0.41, a=360.3 b=141.7 ns). Every
PASSED-ALONE class excluded: not load (3.9% spread, quiet box), not
concurrency, not an unseeded draw (25 interleaved rounds, whole distribution
above the bar), not Issue 855's vanished-work class (both arms `black_box`'d
at input and output, a≈243 b≈165 ns/iter — real work). Found by ACCIDENT:
adding a `println!` to two neighbouring arms meant running the whole target,
and the target was already red.

**T1 — the arch reading (2026-09-19, 4090 box):** 13/13 x86_64 runs PASS —
default features, `--release`, `--exact`: +avx2 (matrix-lane cfg) median band
0.4523–0.4712 (4.2% spread), plain 0.3896–0.3973; b≈137–155 ns across all
runs, consistent with the `dd8dadbba` b=141.7. The arch gap (~0.23) is ~6×
the run-to-run spread on both boxes. **Arch hypothesis CONFIRMED; cache
regression REFUTED.** (Honest riders: the `dd8dadbba` band matches today's
PLAIN build almost exactly — its attribution to "release + avx2" may
describe a plain-build reading; and +avx2 headroom under sibling-build load
is 2.9–4.8 points — a future red there is a BOX-CONDITIONS question first.)

**T2 — the dual pin (2026-09-19):** `G8_BAR` in the test is
arch-conditional — aarch64 0.75 (M3 worst median 0.7111 + ~5% headroom,
mirroring x86_64's own headroom over 0.4712), everything else the strict 0.5
claim (an unmeasured arch must meet the stated claim or red loudly — never
silently inherit the relaxed bar). The Bench-806-T7 sanctioned form, not a
silent global 0.75; the x86_64 claim did not move. Both x86_64 configs
green post-edit.

**T3 — the lane gap, closed by this commit:** the standing finding was that
a target whose repair was calibrated on an arch with no executing lane is
*unknown*, not green — full_gate is macOS/aarch64 but compile+lint; the
x86_64 execution matrix executes the integration cell but refuses off
x86_64; and test_gate (the only executing lane either arch has, schedule
suspended) never reached integration targets. Fixed by a new row kind in
`scripts/test_gate.sh`: **`PERF_ROWS`** (`pkg:floor:target`, executed at
`--release --test-threads=1` — timing rows measure; the lib rows'
threads=2 is a count-floor noise compromise), first row
`katgpt-rs:12:belief_drafter_goat`. Whole target 0.03 s in release — the
row's cost is the build, not the run. Floor 12 = the target's full test
inventory (platform- and profile-invariant: no `#[ignore]`, no cfg-gated
test fns — the only arch-conditional item is the `G8_BAR` const).
`--canary` now floor-bombs the first row of EACH list. **Measured on the
4090 (x86_64, no concurrent cargo): full gate PASS 207/2063/249/150 +
12/12 on the new row; canary FAILS on both lists as designed.** The
aarch64 reading lands the next time the gate runs on the M3 — the state
moves from unknown to executed on both arches. `suite_membership_audit`
already counted the target "pinned" (plan files name it on paper) — the
gap was EXECUTION naming, which this row now provides in
`scripts/test_gate.sh`. Record doc:
`.docs/10_audits/ci_compile_vs_execute_axis.md` §2026-09-21.

## 2026-09-21 — the x86_64 execution matrix caught a day-old test that had never executed on x86_64: the ordered-dot anti-dedup pin was crafted at NEON's vector width

**Status: RECORD 2026-09-21 · matrix run at `b6dc1d16` (cells 1-6 green at
floors) · repair in this commit.**

First matrix run since the 09-20 drift-sweep run — which predates the
same-day `5458dd69` landing, so the new test had still never executed
on x86_64. Cell 7 (katgpt-types `--lib --all-features`,
`+avx2`, debug) CONFIRMED red — failed in-cell AND 3/3 alone — on
`simd::dot::ordered_dot_tests::ordered_dot_differs_from_simd_dot`, landed
the DAY BEFORE in `5458dd69` (the `dot_f32_ordered` committed-value
substrate, riir-chain Issue 156 T1, Bench 844). The test executed on
x86_64 by nothing until this run: test_gate is macOS, full_gate is
compile+lint, wasm32 builds a third triple. The compile-vs-EXECUTE row's
own doctrine, live: an uninvoked assertion is *unknown*, not passing —
and this one was red.

**Mechanism (the craft was arch-relative):** the anti-dedup pin asserted
`simd_dot_f32 != dot_f32_ordered` on a len-4 crafted cancellation input.
len 4 is NEON's full vector width — the M3 lane where the landing session
validated it diverges (pairwise `vaddvq` → 0.0 vs 1.0). It is strictly
below AVX2's 8-wide floor: `chunks4 = 4/32 = 0`, `remaining = 4/8 = 0`,
and the `while i < len` scalar tail IS the ordered fold bit-for-bit →
1.0 == 1.0 → `assert_ne!` red. The scalar fallback converges at len 4
too (one element per accumulator + lane-ordered `acc.iter().sum()`), so
the doc claim "Every simd_dot_f32 backend reassociates" was false below
one full vector on every non-NEON backend. The test's own doc named the
duty — "the two kernels have converged and this module's reason to exist
must be re-adjudicated" — and the re-adjudication is: the KERNELS are
correct (the sub-vector tail being the ordered fold is cheap, harmless,
and irrelevant to the committed-value contract, which needs the ordered
fold to EXIST, not the simd path to always differ); the TEST's craft was
the defect.

**Repair (this commit):** recrafted at len 16 — the smallest length that
engages a grouped path on EVERY backend (NEON 16-wide unroll, AVX2 8-wide
remainder loop, wasm-simd128 16-wide unroll, scalar 4-accumulator
chunks). Input `[1e8, 1, 1, 1, −1e8, 1, 1, 1, 0×8] · [1;16]`:
ordered = 3.0 (the three +1s before −1e8 are lost, ulp 8 at 1e8; the
three after survive); every reassociating backend pairs the cancellation
inside one lane/accumulator group and keeps all three +1s → 6.0
(hand-computed per backend: AVX2 `hi+lo` elementwise then the pairwise
128 reduce; NEON `acc0+acc1` then `vaddvq`; scalar `acc=[0,2,2,2]`). The
ordered reference is pinned in-test (`assert_eq! 3.0` — every
intermediate exactly representable), and the doc comment records the
vector-width dependency so the next len-4-style craft reds review, not
debug. Matrix-posture rerun green (7/7 dot-module tests, `+avx2` debug
`--all-features`).

Cell 8's two PASSED-ALONE rows (`bench_176_router_forward_cpu`,
`t3_latency_p99`) adjudicated per the three-class checklist: both are
sequential-arms latency BARs (the Issue-723/833 family; 176 already a
documented candidate at 13.3 pts slack), seeded RNG, no fixed temp
paths — load-class evidence only, no finding. Also operational: this run
needed `X86_MATRIX_DIR` on E: — C: sat at 99% / 9.8 GiB free, too tight
for a cold matrix scratch (disk-clean candidate).

## 2026-09-21 — the first full-gate run since 09-16 caught the Issue-860 landing RED: 8 `-D`-list errors in the opt-in feature's test code, invisible to every default-feature lane

**Status: RECORD 2026-09-21 · gate run at `c9939347` · repair in this commit.**

The workstation full gate (`--allow-partial-platform`) had not run since
~09-16 while develop moved through the 818/860 wave. First run back:
Layer 3 (workspace `--all-targets --all-features` clippy with the
mechanical `-D` list) red with **8 errors, all in
`successor_density_critic.rs`'s `#[cfg(test)]` code** — 5×
`needless_range_loop` + 3× `identity_op`, e.g. `b.n_sa[1 * 2 + 0]` and
three range loops indexing `nxt`/`exact` directly. The Issue-803 class
again, on the non-default blind-spot axis specifically: the module is
`#[cfg(feature = "successor_density_critic")]` (opt-in POC), so every
default-feature gate — test_gate, the per-crate runs, the landing
session's own checks — compiled the file to nothing; only the
all-features layer reads it. The 860 landing session validated GOAT
execution (`--features` runs) but never linted at `--all-targets`.

**Repair** (this commit): the two `n_sa` stride asserts gain a `sa`
closure beside the existing `cell` closure (keeps the `(s·A + a)`
documentation intent clippy-clean); the fixed-point sweep iterates
`nxt.iter_mut().enumerate()`; the G1 error loop iterates
`exact.iter()` with enumerate (the flat-index arithmetic on `n_sag`/
`n_sa` stays — those uses are not index-only). Semantics identical:
12/12 module tests green, `g1_exactness_and_ranking_vs_analytic_ring`
included. The healer took 6 `doc_markdown` edits in the same file first
(compile-gated, kept); the loop/identity shapes were manual — the index
math is documentation, and `1 * 2 + 0` reduced to `sa(1, 0)` preserves
it.

**Riders** from the same log: `set_diffusion_schedule.rs` 
`manual_range_contains` → `(0.3..=0.5).contains(&w)`; `ugc_schedule.rs`
deleted the never-called `TableJoint::index` test helper (dead code,
the digits-inverse fold); `bench_602_ar_ness_cross_tab.rs`'s const-value
pin `assert!(CHANCE_NELBO > 3.29 && CHANCE_NELBO < 3.30)` moved into a
`const { }` block — compile-time tripwire now, strictly stronger.

**Verdict after repair**: `⚠ full gate PARTIAL — every layer that RAN is
clean (0 errors, 0 unbuildable)` — 1120 s warm, the macOS device-backend
axis disclosed as always on this box. The 1 ungated warning finding the
pre-repair run counted was the bench_602 const assert (fixed above).

## 2026-09-20 — the x86_64 execution matrix's first full run since 09-16 (354 commits of drift): 11,344 assertions PASSED, and one more load-flipped bar caught by execution

**Status: RECORD 2026-09-20 · matrix run at `7f10d4b7` · repair in this commit.**

The matrix had not run since ~2026-09-16 while develop moved 354 commits —
including the Bench 816/817/818 (structured reads + successor-density-critic
GOAT, the goal_salience substrate) and 841/843 landings. Full run, alone on a
quiet box (23:30, endpoint samples 16.6/17.6 GiB avail):

- Cells 1–7 (lib, `--all-features`, debug, `+avx2`): all green — katgpt-attn
  440 · katgpt-core 5136 · katgpt-dec 298 · katgpt-pruners 3025 · katgpt-rs
  577 · katgpt-tokenizer 74 · katgpt-types 266. Every floor cleared; the
  counts have roughly doubled since the floors were set — floors fire
  downward only, by design.
- Cell 8 (root `--tests --release`, default features, `+avx2`): 223 targets,
  1528 passed, 1 failed → **`bench_105_gdn2_goat::goat_6_context_scaling_flat_o1`
  PASSED-ALONE 3/3** (spread 0.306 against the 0.30 bar in-cell). 0 confirmed
  failures, 0 pinned rows — the membership set stays empty.

GOAT 6 is the fourth member of the Issue-833 class found by execution rather
than census — the same file's GOAT 2 was the third. The shape: four positions
`[1, 8, 64, 128]`, each measured in ONE sequential 2000-iter window, then
`(max−min)/mean < 0.30` asserted; load drift between windows lands on single
positions (here 0.306 — 2% over). Seeded RNG (42), no temp paths: the
adjudication's two non-latency classes excluded, pure sequential-window
sensitivity.

**Repair** (this commit): `tests/common/ab_timing.rs` gains `best_of_arms` —
the N-arm composition of the module's two defenses, which neither existing
primitive covers (`best_of_us` samples one arm back-to-back, so a drift
covering that arm's whole window lands entirely on it; `ab_median_ratio`
interleaves but takes exactly two arms and reduces to a ratio). Round-robin
sampling so adjacent samples from different arms share a load window,
per-arm MINIMUM (contention only ever adds time), loud-zero per arm. GOAT 6's
asserted GDN2 side migrates to it (4 states pre-built after prefill+warmup,
5 rounds × 2000 iters, one warmup window); the flat-KV side stays sequential —
its claim is structural growth > 1.5× with ~67% measured slack, load-immune
per the 833 four-axis read. Measured alone after the repair: spreads
0.038 / 0.051 / 0.102 across 3 runs against the 0.30 bar — the sequential
form's razor is gone. `cargo clippy --test bench_105_gdn2_goat -- -D warnings`
clean (the healer took the doc_markdown half; the needless_range_loop shape
was manual — the index feeds a callback, not a slice read).

## 2026-09-20 — Issue 861 CLOSED: the ugc_alloc_check Windows G4 alloc was a per-call env read — and the issue's own isolation table was wrong

**Status: CLOSED (RESOLVED) 2026-09-20, fix in `1ec9b812`; full issue text: git history.**

`ugc_alloc_check` (the Issue-664 G4 gate) failed 50/50 on this Windows/MSVC
box: exactly 1 heap allocation per loop iteration, debug-only, release-clean,
green on macOS. The filing session's characterization table had ruled out
`estimate_interval` ("isolated out") and named the sampler as the failing
path, with `Backtrace::force_capture()` in the allocator as the prescribed
next step.

That instrument settled it in one run — and settled it the OTHER way. The
armed-window allocator (disarm before capture — `force_capture` allocates
internally, recursion hazard) caught the site directly:

```
19: std::env::var::<&str>
20: katgpt_core::ugc_schedule::estimate_interval
        at src/ugc_schedule.rs:353
15-17: to_u16s -> Vec::with_capacity -> getenv   (Layout { size: 20, align: 2 })
```

Root cause: `#[cfg(debug_assertions)] if std::env::var("UGC_DEBUG").is_ok()`
— a debug print gated on an env var read **per call inside the hot path**. On
Windows the env-var NAME is converted to a UTF-16 `Vec<u16>` for the
W-series Win32 API: one heap allocation per call, every call. On Unix
`getenv` does not allocate — which is exactly why the Issue-664 G4 PASS held
on macOS/aarch64 while the same binary failed 50/50 here. **A single armed
sampler call allocates ZERO** — the filing's "sampler-only loop failed 50"
row was wrong (its experiment cannot be reconstructed; the backtrace is the
evidence), a live instance of the repo's own rule that a characterization
row is a claim, not a fact, until the instrument pins it.

Fix: `ugc_debug_enabled()` — the env read cached in a `OnceLock<bool>`, the
exact shape `tpr::kill_switch` already ships (the only other hot-path env
read in the crate; grep found one further `env::var` site and it is
`#[cfg(test)]`-scoped). Steady state after the one-time read is a plain
atomic load — zero-allocation everywhere, both profiles, both platforms.
`UGC_DEBUG=1` still prints the interval diagnostics in debug builds.

Verified on this box: `ugc_alloc_check` G4 = **0 allocations** (was 50) ·
`ugc_664_poc` 12/12 · lib ugc tests 6 (default) + 12
(`--features decode_order_metrics`) · clippy clean.

Lesson (the third restatement of the repo's own posture rule, now on the
alloc axis): **a debug-only logging gate is still code that runs on the
measured profile** — and `std::env::var` is allocation-shaped on exactly one
mainstream platform, which is the platform a third of this workspace's
lanes never execute. Every env read that can sit inside a hot path takes the
`OnceLock` form, not the per-call form.

## 2026-09-20 — Issue 859 CLOSED: Jev structured reads — POC GOAT + 4090 reference + T5 policy arm measured; promotion declined on layer posture (evidence-banked)

**Status: CLOSED (RESOLVED) 2026-09-20, all tasks T0–T6 done; code + records in `72718f81` (cited post-rebase — the pre-rebase landing commit was `d1075311`, superseded when a sibling doc-sync push forced a rebase); full issue text: git history.**

The arc, end to end: Research 574 distilled vLLM PR #57250's read-only structured-decision
contract → Issue 859 (owner-routed to the 4090) → substrate mapping written BEFORE implementing
(the substrate-first gate) → POC landed as `structured_read[_into]` + `sample_label_index` behind
opt-in `structured_reads` (katgpt-forward) with GOAT G1/G1b/G2/G3/G4 PASS (Bench 816: exact
full-marginal logprobs, canvas bit-identity, 0.4588× full-loop latency, 154/154 suite, 0 allocs)
→ T1 executed on this box (python-only overlay of the PR head onto vllm-openai:nightly; corpora
10/10 · 10/10 · 9/12-corrected; the mm-video-profiler ~4.4 GiB memory finding; Research 574 §8,
including the Issue-665 dual-allocation yield) → **T5 measured (Bench 817, this commit's
companion)**: on the trained `micro_dllm_text` fixture, agreement bars (N=4 re-reads, t=1 — the
reference's protocol) do NOT beat single-read analytic confidence — CI-decisive in both label
arms and every M3 gate subset (Δ(agree−(−H1)) [−0.089, −0.071] / [−0.056, −0.030]); the genuinely
open sub-question resolved the OTHER way: **maxprob beats entropy on wide option sets** (Δ CI
[+0.015, +0.026]) — deployable guidance `label_entropy` narrow / `argmax_label_prob` wide →
**T6 promotion decision: stays opt-in, evidence-banked** — every discipline condition is MET
(GOAT + modelless gain + accuracy axis) but `katgpt-forward`'s `default = []` is deliberately
minimal, the feature would drag the dllm stack into every default build, and there are zero
production consumers today (the flashar_anchor precedent). Root feature forward landed for
findability; re-arm trigger = first production consumer (one-line promotion).

Reopen triggers: a production consumer for `structured_read` (promote then), or evidence that a
deployment path needs per-read stochastic forwards (the one regime where agreement bars could
carry information beyond the readout — the reference-side proxy re-run data-spec is in Research
574 §8's note, optional).

## 2026-09-19 — the open-issue backlog is cleared by owner call (12 files removed; triggers recorded here)

Every open issue in `.issues/` was adjudicated in one owner session: resolved
work got its closure record below, genuinely-open work got an explicit PARKED /
PULL-GATED / DECLINED verdict with a reopen trigger, and the files were removed
per the noise-reduction rule (full text: git history; `number_collisions_expected.txt`
rows deliberately untouched — closing a holder is when a pin becomes the only
record). No number is reused.

## Issue 780 — CLOSED as PARKED: OnlineLinearReadout waits on a consumer that measured itself absent (2026-09-19)

**Status: CLOSED (PARKED) 2026-09-19, owner call.** The primitive argument
(the third half of the linear-probe lineage — frozen offline / indicator bank /
online fit) stands, but the live consumer closed NEGATIVE first (riir-clippy
Issue 107, `5dfa1daff` 2026-09-15: horizon 1 < 2 on every corpus — the strategy
prior is a sufficient statistic; T3–T6 closed UNBUILT), and T5's promotion bar
requires a live consumer by construction. Building now would be a
synthetic-fixture GOAT pass that speaks to nothing measured. Blocker corrected
2026-09-17 (`15386365`). **Reopen triggers** (any one — inherited from
riir-clippy 107's own): (1) an ORGANIC fixseq ring with ≥2-revert runs AND the
span-embed column populated (it accumulates from that capture and is the lane's
only revival path); (2) riir-train densifying the store per R135 / Bench 047;
(3) any post-keep-fix corpus where resolved rate moves between orderings at all.
Type sketch + mutation-class justification preserved in git history
(`780_online_linear_readout_primitive.md`).

## Issue 827 — CLOSED: T4 decided by owner call — known-extra repos ARE legitimate oracles; the freshness guards T2/T5 landed are the correct limit (2026-09-19)

**Status: CLOSED 2026-09-19.** T1/T2/T3/T5 landed (`6d084c38`, `7464fc6e`,
`3948f0e2`): the ORACLE-STALE bucket (never clean, never counted, never
ratcheted — suppression at the QUALIFICATION step, not the tag), per-row oracle
disclosure, two-sided fixture arms, and the fetch-age `⚠ UNVERIFIED UPSTREAM`
advisory so `(0, 0)` no longer means "not measured". **T4 (the owner-gated
second axis): STATUS QUO — a known-extra repo remains a legitimate oracle.**
Rationale: the four motivating citations (riir-shader → seal-game-editor) are
genuinely followable to a real repo with real numbering; refusing adjudication
would move correct rows to UNDECIDED and lose information, while the measured
hazard was never the oracle's AUTHORITY, only its FRESHNESS — and that is now
guarded twice (T2's behind-origin suppression + T5's fetch-age line).
`DOCS_GATE_KNOWN_EXTRA` keeps the box-level acknowledgement loud for everything
else. Reopen if a known-extra repo is ever REMOVED from a box while citations
still name it — the one shape the freshness guard cannot see.

## Issue 833 — CLOSED: the class is measured, the executed backlog is 3 of 38, and the residue is routed (2026-09-19)

**Status: CLOSED 2026-09-19.** T1 (`bench_105` GOAT 2 migrated) and the second
firing (`g8`) hold in-cell; T3's resolver set (provenance-bound, REL-DIFF,
comparison-shape, `n == 0` ordering fix — `381f01f7`, `e56e6773`, `1b7756c3`,
`71d97908`) closed every STATED blind spot it could; T2 EXECUTED all 38
root-`tests/` GATES rows in release at their own required-features
(`1f519578`, `c55a3238`, `b0736d8c`) — **3 candidates** (bench_176 ×2,
bench_164), plus two defects of OTHER classes found by the read and filed
separately (Issues 855, 856 — both closed). **T4 stands as written: no verdict
half, deliberately** — migration is a four-axis per-target read (orientation,
claim direction, chunk size, `black_box`), never a codemod; a ceiling over it
would be a backlog wearing a pin. Residue routed: the 3 candidates + bench_008's
doc-vs-assert divergence convert when the x86_64 matrix next fires them
(PASSED-ALONE is the discovery instrument — Issue 834's sampling finding); the
non-root half (`crates/*/tests|benches`) is answered by Issue 834 T3's owner
call below (no shared crate). Full record: git history
(`833_sequential_ab_timing_ratio_is_a_box_measurement.md`).

## Issue 834 — CLOSED: census shipped, the count reframed as a sampling population, and the shared-crate question is decided NO (2026-09-19)

**Status: CLOSED 2026-09-19.** T1 shipped `scripts/sequential_ab_timing_audit.py`
(`8045fae9`, phantom-repo correction `e1a572f2`); T2's measured reframe stands:
the DECIDED count is a population every matrix run SAMPLES FROM (two firings,
one pre-committed clean run 3 — `60264768`), the record-the-NAME discriminator
is adopted, and the matrix now discloses box state on its verdict line
(`d8bfa9b8`). **T3 owner call: `ab_timing.rs` STAYS a `#[path]`-included
module — NO shared crate.** Rationale: katgpt-rs is the upstream public funnel
and a test-harness crate would add a public, versioned surface for a test-only
concern; the measured DRY residue is 9 hand-rolled twins across three siblings
against 12 adopted sites here — real, but below the maintenance cost of a
published crate, and per Issue 833 T4 each hand-rolled twin is often the better
instrument for its own claim direction anyway. Siblings keep copying the module
per repo (the existing `#[path]` convention, already documented). **Reopen
threshold: a fourth sibling hand-rolls a twin** — that is the DRY line. **T4
stands as written** (no verdict half; re-measure after the residue converts).

## Issue 835 — CLOSED: T4 decided by owner call — the path dep is the workspace convention; publishing is contrary to Research 003 (2026-09-19)

**Status: CLOSED 2026-09-19.** T1 (docs-gate CHECK 28, `cross_repo_path_dep_gate.py`,
`5b09c3f4` + the unexercised-buckets disclosure `aba3827f`), T2 and T3 done
2026-09-18 — both open questions measured and DECLINED an instrument on the
measurement; riir-llm cloned and registered, gate green over 311 deps / 183
manifests. **T4: riir-llm keeps its path dep and is not published.** Rationale:
every sibling consumes via path deps against the `/git/` layout — that IS the
workspace contract — and Research 003 ("anything `riir-*` is internal, no
exceptions") forbids a public registry surface; a git dep would work but adds
version-sync overhead the path layout already solves, and no consumer has asked
to escape the layout. The gate keeps the box readable, which is the issue's own
stated limit. Reopen if a consumer outside the `/git/` layout ever needs
riir-llm.

## Issue 839 — CLOSED: kron_tile 4/4 GOAT; T7 DECLINED on measurement (2026-09-19)

**Status: CLOSED 2026-09-19.** T1–T6 + T8 done (`a623d7d4`, Bench 839 — all four
gates PASS, stays opt-in per the no-default-consumer rule; T2's bit-parity ask
was correctly replaced by a measured 4.3e-7 tolerance pin plus the `W² = I` /
factor-orthogonality anchors bit-parity structurally cannot provide). **T7
(ternary-factor fusion) DECLINED**: Bench 843's width sweep measured ternary/f32
at **1.56× slower at w=32, 2.24× at w=64, peaking 3.70× at m=512** (`5a1da65d`)
— the register-file intuition pointed the wrong way (a 32×32 f32 factor is
already L1-resident at 4 KiB), add/sub-accumulate loses to FMA on this box at
every width the tile would use, so the latency premise is refuted; the quality
axis was never modelless-decidable here (riir-train's lane). Reopen only if a
future ternary kernel generation beats f32 FMA at these widths on measured
evidence.

## Issue 841 — CLOSED as lead-backlog dissolved: 7 rows landed, the remaining leads stay at their research-note homes (2026-09-19)

**Status: CLOSED 2026-09-19, owner call.** Landed from this tracker:
`.kpt` archive POC (opt-in `kpt_archive`, Bench 841, 17 bad-injection arms,
owner gate D3); fix_verify refusal-vs-truncation audit (repaired in riir-clippy
`651728c0`); riir-rag embedder identity (verified; wiring filed as riir-ai
`.issues/983`); grammar-forced vocab-projection skip (`legal_token_set` — the
one PROMOTED default-on, 20.6–22.5× tree build); KV permanent sinks + bounded
window (`kv_sink_window`, 5/5 GOAT, opt-in, promotion blocked on the corpus —
routed to Issue 857); calibration staleness + all four seams (`2057ac3b`,
`9badceb2`, `4af13a60`, `4483b84e`). **The remaining OPEN leads are not lost —
they stay recorded at their provenance**: CQ-W2A8 GEMV, CLAWS
activation-sparse decode, HiDRA-v2, TurboQuant-H riir-rag lane,
functional_embed (PETE), SAN league cell, engram-delta fine-tune, SAN
npc_brain v2 → Research 568/570/571 rows (re-file one issue per lead when a
session picks it up — the pre-existing convention this tracker itself was
built from); the `VocabChannelPruner` hook follow-up rides the
`legal_token_set` module docs; the sealed long-context corpus is Issue 857's
pull-gated chain. Deferred rows keep their reasons in git history.

## Issue 852 — CLOSED as PARKED: the τ(t) P_e-LUT POC waits on a D2F model consumer (2026-09-19)

**Status: CLOSED (PARKED) 2026-09-19, owner call — filed and parked the same
day.** Off Research 572 row 1. The modelless half (a
`ScheduleKind::DecodingErrorLut` LUT + monotone-regression builder + τ↔t
inversion) is landable, but its GOAT G2 requires a **measured P_e curve at
matched step budget** and no D2F model harness exists in-repo to measure P_e
against — landing the primitive now would be exactly the synthetic-fixture
GOAT pass Issue 780 was closed for refusing, and the issue's own honest-scope
clause already anticipates a negative at toy vocab. **Reopen triggers**: (1) a
D2F lane with a real model artifact wants a data-adaptive schedule; (2)
riir-train fires the RecFM |V|≥32k invert-CDF t-sampling twin (its Issue 563) —
the effect is in-regime there and the measurement transfers. Lead preserved:
Research 572 rows + git history (`852_fmlm_tau_lut_schedule.md`).

## Issue 853 — CLOSED as PARKED: the autoguidance POC waits on its measurement arms (2026-09-19)

**Status: CLOSED (PARKED) 2026-09-19, owner call — filed and parked the same
day.** Off Research 572 row 2. Arm A needs a DDTree acceptance harness over a
served model; Arm B is explicitly riir-train Issue 563's free-rider. The bare
kernel (η=1 identity, zero-alloc) is trivial, but without either arm it is a
primitive with no consumer and no measured claim — the no-default-consumer rule
and the GOAT discipline both refuse. **Reopen triggers**: (1) riir-train 563's
Arm B fires (a dual-dropout RecFM LoRA exists to A/B against — ~0 training
GPU-hours); (2) a DDTree acceptance-rate bench lands in riir-ai's serving lane.
Prior-art honesty preserved in Research 572 (logit-space arm = contrastive-
decoding ADOPTION, Li et al. 2022; dual-dropout latent arm = the fusion
candidate) + git history (`853_autoguidance_residual_extrapolation.md`).

## Issue 854 — CLOSED: the wedge is diagnosed, the repair scoped, and T3 stays trigger-gated by design (2026-09-19)

**Status: CLOSED 2026-09-19.** T1 (SLOW-vs-WEDGED recipe + the verdict line now
says whether the tree held still), T2 (census: the exposure is in mutation
runs, not the 131 git call sites — no gate, no convention, no sweep, and the
census is preserved as the starting point), T4 (NO in-process wall bound — it
would have the watchdog's defect BY CONSTRUCTION; the external per-module
timeout is the documented practice), T5, T6 landed (`dded36d8`, `65bf2944`,
`d8b55a34`, `c83f0f84`). **T3 (non-interactive git env) stays UNLANDED by its
own rule** — "do not land this as the fix; its relevance is unmeasured until T1
names the call" — and no stall has been observed since T1 made runs observable.
**Reopen trigger**: a stall observed on a NON-mutation run voids T2's scoping
answer and T3 lands with it; a stall inside a mutation run is handled by the T1
recipe (tail the log, kill the git child — measured to resume the run).

## Issue 855 — CLOSED: every task resolved; the guard gate + ratchet sweep both shipped (2026-09-19)

**Status: CLOSED 2026-09-19 — every task closed.** T1 filed with the measurement
(`dd502fe6`); T2 repaired 5-of-5 arms (`black_box` both ends; every bar
unchanged); T3 ran all 34 asserting timed regions at n ≥ 1000 — **7 VANISHED,
27 SURVIVED**, all 7 repaired, bars and pass counts unchanged; T4: the proposed
static `let _ =` detector REFUTED by the execution run (20.0% vs a 21.1% base
rate), `timed_region_guard_gate.py` shipped instead as a docs-gate CHECK gating
the DEFENCE, not the symptom (`196a4cda`); T5 ran all 33 sibling rows — 2
VANISHED = 6.3% against this repo's 20.6%, repaired at riir-ai `3712d51b6`,
unbuildable one filed as riir-chain `.issues/157` (`c7e84919`, `88e5d37f`);
T6 `timed_region_drift_sweep.py` — 21 repos, a ratchet EARNED by execution
(`72ee0f5b`). Full record: git history + the commit trail; AGENTS.md's
`timed_region_guard_gate.py` row stands.

## Issue 857 — CLOSED as PULL-GATED: the corpus builds when riir-ai 882's consumer fires, not before (2026-09-19)

**Status: CLOSED (PULL-GATED) 2026-09-19, owner call.** The boundary question is
answered (`51291548`): corpus + harness = riir-train (`kimi_k3_long_context`
packer), canary wiring = riir-ai `.issues/882` T2, scoring instrument = here
(already shipped). The remaining tracker value was a trigger that is someone
else's — **riir-ai 882's T-Pull-1 is the pull-gate and is already owned there**;
keeping a second copy of the condition here would be the drifting duplicate
Issue 857 itself warned against (its "no sibling issue on purpose" rule,
applied to itself). The honest ordering and the lossless-window escape hatch
(`window ≥ d_max` ⇒ bit-identity, no corpus needed) are recorded in git history
and Bench 841. **Fire condition**: riir-ai 882 T-Pull-1 lands a bounded decode
KV budget → file the riir-train corpus issue citing it, size to THAT consumer's
window (never `kv_sink_window`'s default 260), pin `r_max`/`p_cap_max` against a
no-policy baseline, run `UNBOUNDED` as the lr=0 control, then re-gate
`kv_sink_window` and promote or refuse.

## Issue 843 — CLOSED: the plasma_path ternary dense matvec loses below L3 on every served shape; T4 resolved as per-shape dispatch (2026-09-19)

**Status: CLOSED 2026-09-19, T1–T4 all resolved (T4 by owner call, gate D1).**

**The finding.** The default-on `plasma_path` ternary dense matvec is
**1.9–3.0× slower** than the f32 `simd_matvec` it replaces at every shape
whose f32 operand fits L3 — which includes every shape this workspace
serves (768×3072 → 9 MiB reads 2.10× on NEON; 1024² reads 1.94–2.26× on
AVX2). The kernel is not wrong, it is size-conditional: the x86_64
crossover lands exactly on the L3 boundary (§T2: m=4096, 64 MiB operand,
13700K/30 MB L3); on NEON the ratio narrows monotonically 3.04→1.83 and
never crosses, all the way to a 1 GiB operand (§T1) — the crossover is
box-local bandwidth, not a kernel property. T3's read: the stall is the
FORMULATION — sign extraction outnumbers the multiplication-free
arithmetic ~8:1 on both arms (NEON throughput-bound at ~8.5 GMAC/s, AVX2
chain-latency-bound with `vcvtdq2ps` the suspect).

**T4 — the owner call (gate D1, 2026-09-19): per-shape dispatch.** f32
below the L3 boundary, ternary above, landed as
`simd_matvec_plasma_dispatch` (+ `l3_cache_bytes()` cached probe:
`KATGPT_PLASMA_L3_BYTES` override → macOS sysctl → Linux sysfs → 32 MiB
fallback biased toward f32) in `katgpt-types::simd::plasma_dispatch`, for
callers holding BOTH representations (the quantize-from-dense seam).
Option (b) (pre-expanded i8 sign rows, ~1.7–2× back at 5.3× footprint) NOT
taken — the bitplane footprint is not the product at served shapes.
GOAT G1+G2 GREEN ([Bench 843-dispatch](.benchmarks/843_plasma_dispatch_goat.md)):
bit-identical to the selected kernel both sides of a forced boundary; 2.11×
vs pure ternary at 1024² (bar 1.5×, the house slack convention); −1.6% vs
pure dense (≤10% bar). `plasma_path` STAYS DEFAULT-ON — the 21× footprint
is its real product — and the manifest comments in all five manifests now
state the measured trade instead of implying a latency win. The kernels
are untouched; every existing consumer keeps byte-identical behavior.
Downstream wiring (riir-ai's npc_brain quantize seam is the natural first
caller) is a consumer-side opt-in, deliberately not done here.

The size-sweep instrument (`tests/bench_843_ternary_size_sweep.rs`) and its
measurements are preserved unchanged; the historical issue text is in git
history (`.issues/843_*` removed per the noise-reduction rule).

## Issue 831 — CLOSED: bench_171 P3 reclassified as instrument-health + mechanism by owner call (2026-09-19)

**Status: CLOSED 2026-09-19, T1–T5 all resolved.** The coin flip that filed
this issue was repaired 2026-09-18 (T2/T3 `03d729fdf`: the screener's work
was deleted by `let _ = acc` — the Issue-723 elimination shape; with one
`black_box` the effect measured 64.2% release / 63.1% debug x86_64, and T1
closed 2026-09-19 with aarch64 at 63.1–63.8% across 20+20 runs). What
remained was T5, the owner-gated promotion question.

**T5 resolved by owner call (gate D2, 2026-09-19): P3 is instrument-health,
not a perf bar — `thinking_prune`'s status does not rest on this row.**
`tests/bench_171_thinking_prune_goat.rs` P3 (renamed
`proof_p3_instrument_and_mechanism`) now gates exactly two load-immune
claims and PRINTS the wall-clock number as the diagnostic record
(measured 63.5% at the reclassification commit, matching the recorded
63–64% band on both arches):

- **Instrument health** — both arms measurably real (`a_ns_per_iter() > 0`,
  `b_ns_per_iter() > 0`, median finite): the eliminated-arm class this
  harness exists to catch, with no comparison of the two arms' times.
- **The mechanism as an exact CALL COUNT** — `should_screen_full` is
  `hop >= total_hops - 1` for FrozenBaseGuard, so over 3 hops the schedule
  must make exactly 1/3 of Uniform's screener calls (measured 13,824 vs
  41,472 — exact). Arch-independent, load-immune, and the thing the
  wall-clock number was always a proxy for; a regression that stops
  skipping intermediate hops breaks it to 1:1, not to ~0% speedup.

The ≥30% latency assert is GONE by this call — removed, not lowered; the
30% bar was itself the T2/T3 repair and its measured basis stands in this
record. Issue file removed per the noise-reduction rule.

## Issue 847 — CLOSED: `simd_lut_dequant`'s AVX2 kernels compiled to NOTHING on every ordinary x86_64 build — and the bf16 sweep that followed was right for one kernel of three (2026-09-19)

**Status: CLOSED 2026-09-19, all seven tasks (T1, T2, T2a, T3, T4, T5, T6).**
Filed as 846, renumbered to 847 — `dual_allocation_gate` caught it at
allocation time, which is the case Issue 791 T2's protocol used to reach only
at merge time.

- **T1/T2a — the class.** `#[cfg(all(target_arch = "x86_64", target_feature =
  "avx2"))]` on a SHIPPED path compiles the fast arm to **nothing** on every
  ordinary build, because `target_feature = "avx2"` is OFF by default — so a
  **default-on** feature silently ran its scalar fallback. Repaired with the
  runtime `simd_level()` probe: **1.7x** on `dequant_via_lut`, **4.4–5.6x** on
  `dequant_dot_via_lut`.
- **T2 — the reflex repair was right for ONE kernel of three**, and that is
  the finding worth keeping. RNE narrowing: **2.4–2.5x**, probed in. Trunc
  narrowing: a **REGRESSION** — the intrinsics are measured SLOWER than the
  scalar body. Widen: reported undecidable.
- **T3 — the class is WALLED.** `scripts/shipped_target_feature_gate.py`, a
  docs-gate CHECK, population counted rather than inherited (166 of 166
  `target_feature` cfg attributes in `src/` were in this repo, zero in every
  sibling — so a gate and no sweep). Three exclusions counted on the verdict
  line; the runtime probe's own body could not be a predicate and is PINNED.
- **T4 — aarch64, measured on the M3.** RNE NEON wins 1.23x everywhere; widen
  and trunc NEON both LOSE to LLVM's own autovectorised loop. Bit-identity
  asserted per family.
- **T5 — the trunc AVX2 kernel DELETED, and the transcription hypothesis
  REFUTED.** Re-measured three times over two builds: the `_autovec` arm (the
  SAME scalar body carrying `#[target_feature(enable = "avx2")]`) measures
  1556–1591 in BOTH builds — it lands on the intrinsics, never on the scalar
  — and after the deletion a `+avx2` build's dispatcher measures 1594 against
  autovec's 1591, ratio exactly 1.00, because they are the same code. So
  nothing was wrong with the transcription: **LLVM's default-target
  vectorisation of `bits >> 16` beats its own AVX2 vectorisation by ~1.3–1.4x
  however the loop is spelled.** The kernel was competing with a better
  compiler output, not a worse one. No aarch64 measurement was needed — the
  warning was about the FAMILY, and NEON is implied by the arch (Issue 844
  T4's per-ISA rule).
- **T6 — there is no crossover; the widen arm is BIMODAL on BUFFER
  ALIGNMENT.** T6 was filed as *"measure it on a QUIET box"*. The quiet box
  refuted the framing: the scalar arm is stable to **under 1%** (1692–1704 ns
  at n=32768 over five runs) while every AVX2-bearing arm beside it sits in
  **two clusters** — ~98 or ~170 at n=4096, a 1.75x gap with nothing between,
  picked per process AND per arm. A loaded box produces a spread; two clusters
  next to a stable arm is a property of the RUN. The axis is
  `src.as_ptr() % 32`, tested directly by
  `t6_widen_bimodality_vs_buffer_alignment` (one oversized allocation sliced at
  controlled offsets, residue printed beside the time): each arm has a single
  fast residue and is 1.6–2.2x slower at every other, and `vec![]` makes
  32-byte alignment a coin flip. With the residue controlled the intrinsics win
  **2.2x at n=256** and **1.57x at n=4096** and TIE elsewhere, so they are
  probed in; n=32768 is memory-bound and a wash, STATED in the dispatcher
  rather than hidden. ⚠ The arm's first draft compared VALUES and failed on two
  byte-identical vectors (`Vec<f32>` is `PartialEq`, `NaN != NaN`) — the exact
  trap T2's entry already recorded one arm over.

⛔ **The through-line across T2, T5 and T6: three different wrong answers,
each produced by a sound-looking measurement.** T2 read a regression as
undecidable; T5's write-up blamed a transcription that was innocent; T6's task
blamed a busy box for a deterministic alignment effect. In every case the
instrument had to be changed before the kernel could be judged. File removed
this commit; full record: git history `.issues/847_*`.

## Issue 850 — CLOSED: the UNVERIFIED-upstream guard challenged only ONE of `behind_origin`'s two silent readings — and then PRINTED a false statement about the one it gained (2026-09-19)

**Status: CLOSED 2026-09-19, all four tasks.** Filed as 849, renumbered to
850 — the second collision of that session, and the one `dual_allocation_gate`
could NOT see, which is why T4 exists.

- **T1 — the gap.** `behind_origin()` has four readings AGENTS.md documents as
  *"never pooled"*, two of them SILENT: `(0, 0)` "up to date" and `(n, 0)`
  "behind, but on nothing in this sweep's population". Issue 827 T5's fetch-age
  guard was wired to the first ONLY, while the second is the STRONGER claim —
  it asserts something about the CONTENT of the commits it is behind by, read
  from the same unrefreshed ref. Repair: `elif beh is not None`. Cost, measured:
  `shared_temp_path_drift_sweep` reported 6 findings across 3 repos with no
  advisory; those repos were 22, 9 and 3 commits behind, **4 of the 6 were
  already FIXED upstream** and 2 were correct at their pins — and six
  "repairs" were committed across three siblings before anyone fetched, one of
  which would have undone a recorded adjudication and changed a user-visible
  path. All reset. ⚠ That the gap CAUSED that silence is **not** claimed: the
  pre-fetch state was destroyed by the fetch that diagnosed it.
- **T4 — `dual_allocation_gate` gained the COUNTER axis.** A number allocated
  and CLOSED in one commit leaves no document in any tree, so `--diff-filter=A`
  reports nothing; both sides bumping `.highwater` past the merge base is the
  missing document. Armed both ways — arithmetic on every push with the git
  reader injected, a real two-repo fixture under `--prove-fires`.
- **T2 — should a sweep FETCH? NO, on a MEASUREMENT.** The contract-repo fetch
  costs **250.2s serial / 50.2s at 8-way** cold, 21.5s warm, against a sweep
  that costs 0.04–40s and a 32-check docs gate at ~164s wall — **5–30x the
  cost of the thing it precedes**, paid ~19 times over a family run. The design
  half agrees: a fetch WRITES refs in repos other sessions own (Issue 797's
  class with the sweep as perpetrator), makes a verdict depend on the network,
  and a `--fetch` flag nobody passes is not a repair. Freshness is a property
  of the BOX at a moment, not of a sweep. ⛔ Measuring it surfaced a defect
  worth more than the cost answer: the remedy text said *"`git fetch` in the
  named repo"* and named CONTRACT spellings — correctly, since alias content
  must never reach stdout — which on an aliased box are directories that DO
  NOT EXIST (`mmorpg-editor` vs the checkout `seal-game-editor`). Both rules
  right, remedy unusable. Landed `scripts/fetch_contract_repos.py` (origin
  NAMED, population delegated, arms ASSERT that a fetch leaves HEAD and the
  worktree untouched).
- **T3 — the guard PRINTED a false statement, found by observing it.** The
  task asked for a re-run after a day idle; all 17 repos were 0.2–9.7h fresh,
  so the trigger was unreachable by waiting and the observation was made by
  CONSTRUCTION — stub only the clock seam, run a REAL sweep, read its stdout.
  Reachable, on the FINAL line, rc = 0. And the line said, of every repo,
  *"report 'up to date'"* — FALSE for `(n, 0)`, about the one fact the sweep
  had already read. T1 honoured "never pooled" in the TRIGGER and the DISPLAY
  pooled them one layer down, the `heading_style_blind` shape. `unverified`
  carries `(age, commits_behind)` now and each repo renders its own reading on
  ONE line. ⚠ The existing arms asserted PRESENCE, not CONTENT — every one
  tested `"UNVERIFIED UPSTREAM" in ln`, true of a line saying anything at all.
  `worktree_state` selftest **148 → 157**, 7 of the 9 new arms proven to red
  against the pre-repair renderer, 0 at HEAD, both directions of the split
  armed. File removed this commit; full record: git history `.issues/850_*`.

## Issue 832 — CLOSED: the last fixed shared-temp site is repaired — the non-demo backlog is zero, the ratchet is the wall (2026-09-19)

**Status: CLOSED 2026-09-19 — T5 complete. The final unrepaired site, seal-remake `crates/seal-view/tests/quest_sim_front.rs` (the relay-hub temp dir), landed its pid-suffix repair at seal-remake `db68f5e` (the concurrent-WIP blocker named in the 09-19 census cleared with that lane's landing), validated in-repo with the guard layer-16 invocation (`cargo test -p seal-view --features quest_sim,sync_client --test quest_sim_front` — 1/1, the front floor). Floors re-pinned `mmorpg-remake 16 1 → 18 0` in the same commit: `fixed=0` verified by the sweep, and the site count grew 16→18 through the sibling lane landing (all pid-suffixed). The two other standing rows are ADJUDICATED, not outstanding: seal-game-editor `services.rs` is the deliberate production scratch root pinned `max_fixed 1` at the Issue-842 close-out (`6964117e`), and riir-clippy's 9 production scratch roots carry `[documented-fixed-root]` markers + the read-before-write mtime-churn fix (`83f4cd89`). Every test-class site is repaired per-repo with sibling SHAs cited (riir-ai `9d138531b`, riir-chain `8a3b0f5`, riir-game-sdk `ef66117`, riir-clippy `f5ada0ec`, riir-deployer `b551028`, mmorpg-remake `db68f5e`, mmorpg-editor + mmorpg-remaster same-day 09-19); the demo class (`examples/`+`src/bin/`) stands adjudicated. The x86_64 matrix is fully green (11,177 assertions, `a77c46c0`). File removed this commit; full record: git history `.issues/832_*`.**

## Issue 844 — CLOSED: the dot-delegation crossover is length 24, per-ISA, measured on BOTH arches — not a backlog (2026-09-19; full record at the 09-19 dated entry below, file removed this commit)

## Issue 840 — RESOLVED: two `.research/569` documents landed forty minutes apart and tripped two detectors — the rewrite set was EMPTY, and the attribution lessons became the staged-set rules (2026-09-19)

**Status: RESOLVED 2026-09-18 — T1/T2/T3 all landed by `katgpt-rs-fa` after both live peers confirmed neither document was theirs; `numbering_gate.py` passed and both rows cleared together, which was T3's own assertion holding — no pin was touched. ⛔ T2's premise was WRONG in the safe direction and the correction WAS the finding: the hand-derivated rewrite set was 3 sites, the measured set was EMPTY. The durable output is the attribution discipline now in AGENTS.md §staged-set (shared authorship, one shared reflog, the elimination-over-an-incomplete-roster failure, and quote-the-`from=`-pipe — `4e82cc489` + `c0995ea4b`), which AGENTS.md loads into every session; the incident itself needed no code change beyond the peers' own renumbering. File removed this commit; full record: git history `.issues/840_*`.**

## Issue 836 — CLOSED: a `git worktree` made the entire head-provenance mechanism silently inert — `.git` answers TWO questions and each spelling was wrong for the other (2026-09-19)

**Status: CLOSED 2026-09-18 — T1 `f6749af5c` (the severe five: every `worktree_state` guard took `.is_dir()` and every false branch returns the nothing-to-report value, so in a worktree `sweep_advisory` returned `[]` against a modified tracked file — Issue 797's founding defect reintroduced inside the mechanism built to prevent it), T2+T4 `c12e96415` (the three private copies — `console_encoding_gate.tracked_scripts`, `sweep_advisory_membership_gate.tracked_sweeps`, `numbering_drift_sweep.head_listing` — measured then DELEGATED to `worktree_state.is_checkout`; plus the gate cross-check's silent `.name`-mismatch branch made loud), T3 `52f3de398` (cross-repo axis counted first: 3 sites in 2 siblings, 1 defect, repaired at riir-ai `9637d09ea`), T4 folded into T2's fixture. The `.git` two-questions table (canonical REPO → `.is_dir()`; can-I-run-git-rooted-HERE → `.exists()`) is now loaded into context via `a9bf8c785`. File removed this commit; full record: git history `.issues/836_*`.**

## Issue 845 — CLOSED: `channel_aware`'s duplicated dot kernel deleted; T5 measured NEON PARITY — the x86_64 1.8–7.8× penalty does not transfer to aarch64 (2026-09-19)

**Status: RESOLVED (repair `32056164e`, filed-and-fixed in the same change; T5 — the aarch64 re-measure the x86_64-only record owed — measured 2026-09-19, this commit). The crate's `simd_dot_f32` was a ~200-line same-named duplicate of `katgpt_types::simd::simd_dot_f32` (already in the crate's dep graph; seven sibling files called the shipped one), and its AVX2 arm was gated `#[cfg(target_feature = "avx2")]` — OFF by default on x86_64, so the shipped path ran a 4-accumulator scalar loop on every ordinary build: 2.72–7.76× slower than the runtime-probed kernel it duplicated in the default build, 1.83–4.78× under `-C target-feature=+avx2`, measured in BOTH configurations before the deletion. Repaired by delegation: the 2-argument wrapper kept, the three private kernels deleted, the file's `unsafe` surface to ZERO, the bench converted to a bit-identity regression gate whose canary (a planted private kernel, max |Δ| 4.77e-7 — well inside every tolerance the old tests used) proves a tolerance gate would have passed the exact regression this issue exists to prevent; bit equality is the only form that fires.**

**T5 (the last open task): every number above is x86_64. On aarch64/NEON — where the duplicate's arm was never compile-gated (`target_arch = "aarch64"` implies NEON), so only the DRY half of the finding could apply — the deleted kernel measured PARITY with the shipped one, not slower: three interleaved A/B runs (`ab_median_ratio`, 11 surviving rounds per length, release, M3 Max) put old/shipped at 0.98–1.04× at lengths 32–256 and a stable ~0.95× at 1024 across all three runs and both load states (box: load 6.8–10.2, one 952%-CPU sibling test live during run A, 85% memory free, on battery — the RATIO is the quantity, per the sequential-A/B discipline; agreement ≤ 1.5e-6 relative, summation order differs by construction). The one stable signal — ≤5% at the longest length — is consistent with the shipped kernel's Issue-700 soundness reslice + explicit-len signature, a clean-panic-over-OOB trade already made deliberately. Verdict: the deletion costs nothing measurable on NEON and buys one kernel owner for seven-plus callers; "the shipped kernel is faster" stays an x86_64 claim and is no longer quotable as an aarch64 one. Temporary fourth-arm bench (the Issue-843 precedent) deleted after recording.**

## Issue 851 — CLOSED: `muon_update` step scale repaired to the canonical √max family (update RMS ≈ 1.0) (2026-09-19)

**Status: RESOLVED (session `research-2502.16982-cont`, the arXiv:2502.16982 distill follow-up; fix `1ebe39f61` (hash is post-rebase — the branch moved under the fix; cae72cf0f was the pre-rebase SHA)). The public default-on Muon wrapper scaled its update by `1/max(rows, cols)` — D× smaller than canonical at square D×D and d-dependent, the exact class riir-train Bench 492 measured as un-absorbable by an LR grid — while its own inline comment claimed "standard Muon scaling" and its doc comment carried a THIRD formula (`1.0/rows`). Zero production consumers at repair time (latent).**

- **The prediction that was wrong, and why the re-run mattered:** the issue expected the GOAT assertions to be scale-invariant. They were not — Gram-vs-identity at the old scale passed near-vacuously (err ≈ 0.98 under a 1.0 bar: the output was tiny, not orthogonal). T3.1 + the module orthogonality test rewritten normalized-by-mean-diagonal so the assertion tests geometry, not scale (measured 0.2503 @ 8×8; threshold 0.35, consistent with the raw-NS5 T1.x error band 0.21–0.44).
- **New regression pin:** update RMS ∈ [0.68, 1.12] (the NS5 singular-value band, Bench 050) at 64×64 / 128×64 / 64×128 — measured 0.9667. Any future scale regression moves RMS off 1.0 with D; the d-dependence is now a red test, not a latent API trap.
- Scale chosen: `√max(rows, cols)` — the riir-train `muon_step_dense` family (RMS 1.0); Keller `√max(1,A/B)` and Moonlight `0.2·√max(A,B)` (arXiv:2502.16982 Eq 7) differ only by LR-absorbable constants (Bench 492 measured 13% at D=32 between the named forms). Doc now names all three so the next reader does not re-derive a "fair" match.
- Validation: GOAT bench_152 25/25 PASS at the new scale; module muon tests 3/3; clippy `-D warnings` clean (katgpt-core lib + root test target); rustfmt clean. Bench 050's perf records unaffected.
- Provenance: filed from the arXiv:2502.16982 (Moonlight) distill — the paper was otherwise fully distilled in-workspace (NS5 core = Plan 152; weight decay + √max scaling shipped and twice superseded in riir-train; the SFT negative result — Muon-SFT gives no advantage on AdamW-pretrained bases — is preserved in the removed issue file's git history).

## Issue 842 — CLOSED: the alias-mapped sweeps opened a directory that does not exist, and the first real read surfaced a workspace of hidden findings (2026-09-19)

**Status: RESOLVED (T1–T4, this session — session `katgpt-rs-c5`, continuing the 837/846 lane). The class: `derive_repos` returns CONTRACT names, and 17 of 19 drift sweeps opened `WORKSPACE / <contract-name>` directly — on a box whose `repo_alias.local.txt` maps the three seal repos into the mmorpg-* contract spellings, that directory does not exist, and every tracked-file walk on it returns 0. Seven sweeps red walk floors against pins typed from the real repos: every red was a TRUE pin measured against the WRONG DIRECTORY. The two sweeps that DID import the alias (`docs_drift_sweep`, `numbering_drift_sweep`) built `ws / contract-name` paths — the same hole wearing the codec. `worktree_state.sweep_advisory`'s bare-name branch silently `continue`d on the nonexistent dirs, so the advisory never saw the aliased repos either.**

- **T1 — the seam, one mechanism not 19 patches.** `sweep_population.open_repo(name, workspace)` for the sweeps (the issue's own prescription) + `repo_alias.real(repo)` = `repo.parent / disk(repo.name)` for the audit modules — identity for any unmapped name, which is what makes it FIXTURE-SAFE: every selftest's `tmp_ws / "fakerepo"` resolves to itself, so resolution could live at the file-access seam without breaking a single arm. `worktree_state.sweep_advisory` resolves bare names through `disk()` and prints scope keys through `display()` — the alias CONTENT itself must never reach stdout (run logs get pasted into tracked docs). Arms for all three, box-independent (injected `_loaded`, not the machine's alias file; the two-sided advisory arm plants a MAPPED repo dirtier than an identity one so a run that opened the wrong directory reads "(1)" and fails instead of passing against a lookalike).
- **T2 — the labeling law.** Audits attribute findings by `repo.name` (lda's `Kernel`/`CallSite`/`BindSite`, citation's `alloc` keys, cga's `RepoReport`), so passing resolved paths would have leaked the on-disk names into pins and stdout. The shape everywhere: **resolve the path you READ FROM, keep the name you LABEL WITH** — `docs` keeps the contract handle list for its loop and hands `repos_real` to the auditors, remapping child-auditor header keys through `display()` in `run_auditor`; citation resolves inside `audit()`/`crate_map`/`unreliable_oracles`/the alloc + blind builders; lda resolves at each `rs_files` walk while `Kernel(repo.name, …)` stays the contract spelling; `percentile_index_audit.main` keeps the contract name beside the resolved target; `restatement` translates BOTH populations through `apply()` before opening.
- **T3 — the first real read.** 19/19 non-citation sweeps GREEN post-fix (the family's first honest look at three repos). What the zero-read had hidden, all adjudicated in the same commit: seal-game-editor 3 undefended console-encoding scripts + 2 locale-io `open()` sites + 1 shared-temp test path + 4 unreachable instruments; seal-remake 3 shared-temp test paths + 1 platform-dead-code ungated `FrameRateCap` decl (FILED as seal-remake `.issues/032` — the file carries a concurrent session's WIP; the one-attribute repair is theirs) + 2 silent-now targets + 7 plan-scoped unreachable scripts; seal-online-remaster 1 shared-temp test path. Sibling repairs landed THERE, cited: seal-game-editor `6964117e`, seal-remake `df81497`, seal-online-remaster `5fb3122`. ⚠ The seal-remake commit also carries a CONCURRENT SESSION'S staged refactor + two script deletions swept in by an index race — disclosed in that commit's own message; the refactor and deletions are theirs, the temp-path line is ours.
- **T4 — the pins.** Ten floors/expected files re-pinned from first-real measurements, each comment carrying its reason: the three repos' post-2026-09-16-migration shrinkage (seal-core/seal-edge-worker moved OUT of seal-online-remaster — orphaned_attr 335→248, platform_dead_code 360/7400→248/2251, percentile 335→248, wasm32 2/2→1/1, cfg_gated min_gated 1→0; and seal-remake's GROWTH — platform_dead_code 33/270→441/10779) + the wasm32 UNCOVERED control's rename (mmorpg-poc-submodule → seal-poc-submodule, expected file) + shared_temp_path first-real rows (two sites repaired, `TmpDirStore` adjudicated DELIBERATE at max_fixed 1, quest_sim_front's site deferred to its WIP-file owner) + numbering/instrument_reachability/len_derived families measured for the first time.
- **T5 — the citation sweep did NOT go green, and that red is true positives.** First real reads reclassified the workspace with REAL allocation reads: ~35 file-addressed citation defects across 6 repos (seal-remake CROSS 15, riir-game-sdk 7, riir-dapps 4, riir-clippy 3, riir-dao 3) — the pre-migration qualifiers written for the old world, every row an editorial read of a doc window — plus riir-ai IN-LOCAL-RANGE 8 > 6, a RECLASSIFICATION (the aliased allocs now resolve; re-pinned 6→8, not new defects). Filed as Issue 846 with the run's table; the repair pass is mechanical once somebody reads each row.

## Issue 846 — CLOSED: the alias seam has a PROSE half — 44 true CROSS rows, 37 cleared by teaching the qualifier the on-disk spellings, 7 by prose (2026-09-19)

**Status: RESOLVED same day (session `katgpt-rs-c5`, continuing 842). The issue's own table undercounted the live run (5 repos listed, 10 red): re-running the sweep at the repair session found 44 CROSS rows across 10 repos — the five filed (mmorpg-remake 15, riir-dapps 4, riir-clippy 3, riir-dao 3, riir-game-sdk 7) plus riir-kat 4, riir-shader 5, riir-mmorpg-examples 1, riir-neuron-db 1 (+ its IN-LOCAL-RANGE 3 > 2 breach), riir-viewbridge 1. Read the live sweep, never a filed table.**

- **The shape of nearly every row was NOT the pre-migration qualifier the issue hypothesized** — it was the ON-DISK spelling. The contract names `mmorpg-editor` / `mmorpg-remake` / `mmorpg-remaster` exist in `repo_set.txt` only; on BOTH measured boxes the directories are `seal-game-editor` / `seal-remake` / `seal-online-remaster`, and every doc citing their plans names them that way — prose the qualifier matcher could not see (it knows contract full names + stem aliases). `seal-remake Plan 011` is a CORRECT address the instrument read as unqualified. Bulk-rewriting 43 rows to contract spellings no box has on disk would have made the docs worse to serve the instrument — the repair went the other way.
- **The classifier half (37 rows, katgpt-rs this commit).** `issue_citation_gate.spelling_aliases()` — a code-level table, NOT the gitignored `repo_alias.local.txt` (a box-local qualifier table would make verdicts machine-local) — consumed by `qualifiers()`: a spelling qualifies in exactly the two places the contract full name already does, the 40-char lead and the 3-line window (which reads the citation's own line FORWARD, so the trailing-parenthetical shape "(Plan 192 T4.1, seal-game-editor, …)" qualifies without a prose edit). LENIENCY ONLY: `written_names` stays contract-only, so a spelling clears a row but never accuses one — no new ⛔MISATTRIBUTED class. ⚑ The boundary arm caught a real defect before landing: the short-alias path matches with plain `\b`, and `\bseal-remake\b` MATCHES inside `seal-remake-unity` (the retired repo) — spellings match with the `_NAME` boundary regex instead. `alias_trail_owners` deliberately excludes spellings (a spelling trailing on the citation's own line is already inside the window — counting it would report a widening appetite for rows the window settles). Arms in both selftests (gate units + sweep `audit()` fixtures resolving through `repo_alias.disk()`, box-independent).
- **The prose half (7 rows, committed in the siblings, cited):** riir-dao `160f9a1` (Proposal 007 → riir-clippy; Proposal 006 B6 + Plan 035 → riir-dapps), riir-kat `51994ee` (Bench 053 ×2 → riir-clippy; Proposal 006 Q1 → riir-dapps — ⚠ the local repair `25006d1` turned out a byte-identical TWIN of a commit the Issue-837 close-out had already pushed to origin; the box's checkout was simply behind, so the sweep re-found origin-fixed rows — the rebase auto-skipped the twin via cherry-pick detection), riir-neuron-db `3f6cf67` (Proposal 032 + Issue 036 → riir-game-sdk; Issue 168 → seal-game-editor; Issue 037 → riir-clippy — the last three sat in the IN-LOCAL-RANGE undecided bucket, and qualification skips a row before cls is computed, so naming the true owner clears them; ownership verified against each sibling's `.issues` walk + git history before any edit).
- **Landing measurements:** sweep rc=0 for the first time since 842 opened the aliased repos — 0 CROSS over 0 adjudications, 23 IN-LOCAL-RANGE (all ratchets hold; riir-ai 8→5 and riir-neuron-db 3→0 because the spelling acceptance qualified rows whose prose already named the owner), 0 ORPHAN, 2 ORACLE-STALE (uncounted; katgpt-rs 18 behind / riir-ai 9 behind — the shared-worktree pull is the sibling session's). Floors held WITHOUT re-pinning: every cross wall is 0 and every in-range ratchet measures at or under its pin. Doc note added to AGENTS.md §Docs gate beside the sweep's error-rate paragraph.

## Issue 838 — CLOSED-as-decided: a shell script spawning a NATIVE child is a third encoding seam, and the population is measured at zero live instances, so it is deliberately NOT gated (2026-09-18)

**Status: MEASURED and CLOSED as a decision — no instrument landed (`c9afca9dd`; found by session `katgpt-rs-54` adding the box-state block to `scripts/x86_64_execution_matrix.sh`, where a `·` inside a PowerShell format string came back mangled through the box's cp874 console; repaired at source by keeping the child ASCII and letting bash own the separators).** The workspace gates two encoding seams — our own prints (`console_encoding_gate`) and a Python child's PIPE (`subprocess_encoding_gate`); a `.sh` handing non-ASCII to `powershell.exe`/`wmic`/`cmd.exe` falls between them, and neither instrument can see it.

The census, over 197 tracked `*.sh` in 16 contract repos: **43 native-child invocation lines, 2 carrying non-ASCII on the same line — both comments** (`riir-ai/scripts/perf_rematch.sh:509`, `riir-train/scripts/c13_auto_gate.sh:243`). Live instances: **0**, the one real case having been repaired at source. A gate would govern an empty population AND need the comment/string-masking machinery three instruments already grew, just to suppress its only two rows — a cries-wolf instrument by construction.

This is the `check_validation_gate` T4 shape, deliberately the OPPOSITE of the `console_encoding_gate` mistake: 789 measured a population of one and declined a sweep, correctly; `console_encoding_gate` then *assumed* the answer carried across and was wrong by seven repos. The rule both episodes teach: **measure the population, then decide** — re-run the census before reopening, and keep native children ASCII in the meantime (a non-ASCII glyph in a `.sh` comment is harmless and not this class).

Adjacent gap recorded in the issue, NOT closed there: the matrix's first box-state read selected the memory source on whether `/proc/meminfo` EXISTED rather than whether it ANSWERED — MSYS ships a readable `/proc/meminfo` with no `MemAvailable`/`CommitLimit`; the Issue-835 `.git`-FILE shape one level over, repaired by selecting on ANSWERS and printing which one did.

## Issue 837 — CLOSED: registering a contract repo reds the whole sweep family, and the ONE file a gate checks is not the twenty-one that make them red (2026-09-18)

**Status: RESOLVED same day (T1–T3). Found by session `katgpt-rs-c5` when an Issue-836 sweep run reported `✗ riir-llm … UNPINNED`, a failure unrelated to the change under test. ⚠ Honest landing note: the session's issue file claimed T2 landed when it had not — no gate file, no commit, no CHECKS row existed on develop; T2 was built and landed afterwards (this record's session), and T1 was completed then too: the sweeps' own UNPINNED verdicts on the current tree showed four more registered repos (`katgpt-web`, `riir-dao`, `riir-deployer`, `riir-esp32`) owed rows in 8 more floors files — the Issue-798 class, claim preceding disk, caught by reading the tree instead of the prose.**

The class: `riir-llm` was registered (`b5dd81dc`) with `repo_set.txt` + AGENTS.md §Repo count updated — everything `agents_repo_set_gate` asserts — and every per-repo pin file keyed on that registry was left behind. 19 of 21 drift sweeps then red `UNPINNED` for a bookkeeping reason on a repo nobody had looked at; four live findings sat behind those reds (riir-kat 3 unqualified citations, fixed riir-kat `51994ee`; riir-shader `gamefx_feature_matrix.sh` EXPOSED trap window, fixed riir-shader `176da06`; riir-clippy 7 undocumented plan-scoped scripts, ratcheted at measured — the documented OVER-CAPTURE class; riir-shader `.plans/.highwater` stale at 004, fixed riir-shader `6f045d4`). Registration is a 22-file operation with exactly one file gated — the eleventh recorded instance of the rule-landed-in-one-instrument shape (777, 778, 782, 783, 789, 793, 797, 820, 822, 836).

- **T1 — the pin rows.** Every value typed from the owning sweep's own printed row, never a neighbour's (`2d348a470` + the completion rows; the first attempt cloned riir-auth's row and was wrong eleven times — reverted; *a pin copied from a neighbour is a diary, not a wall*). The arity assertion caught two floors files whose format headers declared 4 columns while rows carried 6/7 — repaired against the sweeps' own `FIELDS` tuples. Final state on the landing run: every `*_drift_floors.txt` file complete for the demanded set.
- **T2 — `scripts/repo_registration_gate.py`, docs-gate CHECK 29:** every `*_drift_floors.txt` pin file carries a row for every on-disk canonical repo (registry ∩ `derive_repos` — the sweeps' own demanded set, delegated not re-derived). The two legitimately subset-scoped files (`docs_drift_floors` — its sweep demands rows only for repos with labels; `restatement_drift_floors` — population is repos with `.proofs`) are declared with reasons in `scripts/repo_registration_scope.txt`, membership, reds in BOTH directions: a scope row over a file that now covers everything is STALE-SCOPE, a scope row for a non-candidate is UNKNOWN-SCOPE, a reasonless row is refused, a non-scoped file missing a repo is INCOMPLETE by name. The inverse direction (a repo-shaped row for a repo outside the registry) is asserted too — measured at zero, asserted anyway, one set difference the gate already computes. Non-repo keys (`TOTALS`, citation's global floors) are out of scope by the shape regex, not an allowlist. Unconditional arms + `--prove-fires` (rewinds the founding state in memory: the console file without its four repos must red INCOMPLETE naming them).
- **T3 — the inverse direction, MEASURED at zero** across every floors/expected pin file: the only repo-shaped keys outside the registry were the six PACKAGE names in `x86_64_matrix_floors.txt` — a package-keyed file, a heuristic false positive, not a stale row. Zero is a measurement, not an absence of the class.

The standing census lives in the gate's own PASS line (candidate count, demanded-set size, scoped files, absent-repo posture) — never in prose. What it does NOT claim: that a present row is CORRECT. The sweeps own their row values; this gate owns the set.

## Issue 825 — CLOSED POSITIVE, after a same-day RETRACTION of its own negative close. Coulomb crowd redistribution ships as `coulomb_flow`; the "negative result" was a bench walker with two defects (2026-09-18)

**Status: RESOLVED. `coulomb_flow` ships opt-in ([Bench 825](.benchmarks/825_coulomb_crowd_redistribution_goat.md), G1–G4 all PASS). [Bench 815](.benchmarks/815_coulomb_redistribution_poc.md) is the independent second implementation and now agrees.**

⛔ **This record replaces one that read `CLOSED NEGATIVE — the solve transfers,
the first-arrival readout does not`, and the replacement is the point of
keeping it.** Two sessions implemented this issue concurrently from the same
research row and landed opposite verdicts within four hours: one shipped the
primitive behind `coulomb_flow` with every GOAT gate green, the other measured
an endpoint MAE of 0.1190 against a ≤ 0.01 bar, filed the negative-result
clause, removed the issue file and wrote the negative into this document.

**What settled it was not re-reading either bench.** It was running the
LOSING bench's own three fixtures — `grid_2d(4,3)/(8,6)/(12,9)`, sources
uniform, two sinks at 0.6/0.4 — through the WINNING bench's shipped readout:
MAE `0.0 / 1.2e-7 / 3.0e-8` where the same fixtures had read `0.119 / 0.078 /
0.037`. Same graph, same solve convention, same targets. A disagreement
between two implementations is a cheaper oracle than either one's internal
consistency, and neither run alone could have produced it — Bench 815's own
gates were all satisfied at the moment it declared the construction falsified.

### The two defects, both in `bench_815`'s `absorb()`

1. **A sink absorbed 100% of what reached it.** The flow decomposition stops a
   passing particle at `v` with probability `μ₁(v) / (inflow(v) + μ₀(v))`,
   which is 1 only when `v` has no outflow. Measured through the shipped
   `CrowdRouter::consistent_absorption` on 815's own fixture: sink `n−2` has
   `out_flow = 0.209` and a consistent absorption of **0.657**. The two sinks
   are adjacent on a grid, so mass bound for the heavy one passed through the
   light one and was stranded — one sink's excess exactly the other's deficit,
   `[0.4810, 0.5190]` against `[0.6, 0.4]`.
2. **The proportional split was order-dependent.** `share = packet[v] * (p /
   total)` was evaluated while that same loop decremented `packet[v]`, so the
   second out-edge was sized from the reduced remainder. Mass stays conserved
   (the residue re-forwards and the geometric series sums to 1), so a
   conservation gate cannot see it; the proportions do not — with two
   out-edges at `p₁ + p₂ = 1` the first gets `p₁ / (p₁ + p₂²)`. It is also
   invisible at degree-1 vertices, which is most of a small fixture.

### The lesson is about the INFERENCE, not the arithmetic

The refinement axis read `0.119 → 0.078 → 0.037` and was written up as *"the
miss shrinks monotonically with refinement ⇒ discretization error in the
readout, not a broken solve"*. That sentence is a mechanism inferred from a
monotone sequence of three points, and the mechanism it named —
proportional splitting failing to carry the continuum hitting measure — is
precisely the one the corrected readout proves is exact. What the trend
actually showed was two adjacent sinks becoming a smaller share of a growing
graph's transport. **A converging error is consistent with many mechanisms;
naming one and closing an issue on it is the move to distrust.** The correct
next step from that trend was the one the winning session took independently:
derive what the absorption probability HAS to be and check the instrument
against the derivation.

⚠ And the negative was not cheap to place: it removed the issue file, wrote a
"the paper's 20× does not transfer to zone graphs" paragraph into this
document, and recorded three re-open conditions — (a) a characteristic-
preserving per-particle readout, (b) a resolution where the gate passes, (c) a
consumer contract surviving (a). All three were answers to a defect. The
substrate-first gate exists to stop exactly this shape one step earlier: the
second implementation began after `katgpt-dec::coulomb` was already on
`develop`.

### What shipped

`coulomb_flow` (opt-in, katgpt-core → katgpt-dec): `CoulombFlowField` solves
`δ(dφ) = μ₁ − μ₀` on a zone graph and hands back `j = dφ` plus the per-vertex
`CrowdRouter` a crowd consumes. Mass conservation is the equation rather than
an approximation, and the arrival distribution is `μ₁` exactly as a flow
decomposition — the only per-NPC error is sampling error, so Bench 825's G2
gates the `1/√N` DECAY (21.7× over 1e3 → 1e6) rather than one MAE at one N.
G1 conservation 5.96e-8, G3 naive ratio 1312× against a 10× bar, G4 zero
allocations over 100 solves. Consumer wiring is riir-ai's per the boundary
contract, so there is no default-path consumer here and the flag stays opt-in.

Bench 815 was repaired rather than deleted: its f64 pinned-Gaussian solver and
its deterministic mass-packet walker are independent of Bench 825's CG solve
and sampled walk, and two independent implementations agreeing is worth more
than one. The single thing they now SHARE is the absorption rule — imported
from `CrowdRouter::consistent_absorption`, not copied — because that is the
one thing they disagreed about.

⚠ Still open and recorded rather than claimed: scale beyond 108 zones, an
irregular (non-grid) zone graph, and the `to_flow_vectors` bridge.

**En-route substrate gain (kept from the retracted work):** the katgpt-dec
crate root now re-exports the zero-alloc `_into` operator family
(`exterior_derivative_into`, `codifferential_into`, `graph_laplacian_into`,
`hodge_laplacian_into`) — the alloc-free twins were previously unreachable
from the root, which is the kind of gap that quietly teaches every consumer to
use the allocating wrapper.

## Issue 830 — CLOSED: Issue 829's anchor class one seam deeper — the locale-I/O classifier was anchored to a CALL-NAME SET, and it had repaired one side of a round trip in this repo's own instrument (2026-09-18)

**Status: RESOLVED same day (T1–T5).**
**Severity: a tracked instrument in this repo WRITES a fixture with the system
locale and READS it back as UTF-8 — Issue 829's exact defect, surviving Issue
829's own repair.**
**Origin: read the residual of Issue 829 instead of trusting its count.**

### The symptom

Issue 829 closed with **273 → 2**, the 2 outside the contract, the gate walled
at 0 and the sweep ratcheted. Every number in that record is correct. It is
also a count of **three call names**.

`cfg_row_implication_audit.py` is a tracked instrument in the docs gate's
dependency closure. Its selftest, at line 407:

```python
with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as fh:
    fh.write(body)
    name = fh.name
try:
    return leading_inner_cfgs(name)
```

and `leading_inner_cfgs`, in the same file, line 192:

```python
text = Path(path).read_text(encoding="utf-8", errors="replace")
```

The read side carries an explicit `encoding=`. **Issue 829 put it there.** The
write side is `tempfile.NamedTemporaryFile("w", …)`, which encodes with
`locale.getencoding()` — and is not `write_text`, not `read_text`, and not the
builtin `open`, so the classifier never saw it.

That is a **round-trip mismatch inside one file**: a cp874 write followed by a
UTF-8 read, twelve lines apart, half of it repaired by the issue whose entire
subject was that mismatch. The fixture bodies are ASCII today, so it passes —
which is Issue 829's own canonical failure verbatim, *"a test suite that had
been green for the wrong reason … it stayed invisible until an arm was written
whose subject was the corrupted character."*

### The finding

`locale_io_fix.sites()` reads:

```python
PATH_METHODS = {"write_text", "read_text"}
...
elif isinstance(node.func, ast.Name) and node.func.id == "open":
```

The class is **a text-mode file object whose encoding defaults to the locale**.
The predicate is **three names**. Everything else that constructs one was
invisible:

| form | why it is in the class |
|---|---|
| `p.open("w")` | `Path.open` / `io.open`, text mode |
| `tempfile.NamedTemporaryFile("w", …)` | text mode; the default `"w+b"` is safe, an explicit text mode is not |
| `os.fdopen(fd, "w")` | text mode; default `"r"` is text too |
| `io.TextIOWrapper(fh)` | always text |
| `codecs.open(p, "w")` | encoding is positional there |

This is the **fourth** time a rule in this family has been found anchored to
one representation of its own subject — Issue 823 (anchored to a POSITION),
Issue 828 (anchored to a DELIMITER SET), Issue 787 (a census anchored to the
DOCUMENT), and now a classifier anchored to a NAME SET. The shape is not that
somebody was careless; it is that **a classifier's coverage is only ever
testable against inputs somebody thought to write down**, and the cheapest
probe for it is to enumerate the class's *mechanism* and ask what else
produces it.

### What was measured (T1, before anything was claimed)

An AST probe over tracked `*.py` in all 16 checked-out contract repos, for
every attribute call that can produce a locale-encoded text handle:

```
    7  .open(<text mode literal>)         katgpt-rs 0 · riir-train 6 · riir-clippy 1
    2  tempfile.NamedTemporaryFile TEXT   katgpt-rs 2
    2  os.fdopen TEXT                     katgpt-rs 2
    0  io.TextIOWrapper
    0  codecs.open
   11  TOTAL real sites
```

**Four of the eleven are in this repo's own instruments, and all four write a
fixture some other code then reads**: the two `NamedTemporaryFile` rows write
a `.rs` and a `.yml` that the same selftest parses back, and the two
`os.fdopen` rows in `trap_exit_launder_audit.py` write **shell scripts that
`bash` then executes** — where a cp874 body is not mojibake in a comparison,
it is a script the interpreter reads as garbage.

### T2 — the discriminator, and its blind spot is MEASURED rather than argued

`.open` cannot be duck-typed by name the way `read_text` can. The workspace
carries `os.open`, `tarfile.open`, `Image.open` — none of them text files, all
of them `ast.Attribute` with `attr == "open"`. The existing classifier's stance
(*"`sock.read_text(x)` — duck-typed: the name IS the population"*) does not
transfer, because that name is Path-specific and `open` is not.

The discriminator is a **literal text-mode argument**: a `Constant` `str` whose
characters are a subset of `rwxat+U` and which is non-empty. It excludes, in
one rule and with no denylist of module names:

| excluded | by what |
|---|---|
| `os.open(devnull, os.O_WRONLY)` | mode is not a string literal |
| `tarfile.open(arc)` | no mode argument |
| `tarfile.open(path, "r:gz")` | `:` is not a file-mode character |
| `Image.open(hero_path)` | no mode argument |
| `f.open("rb")` | `b` |

⚠ **Its cost is the no-mode case**: `p.open()` defaults to text `"r"` and IS in
the class, and this rule cannot see it. That cost is not estimated — the probe
counted it. **5 sites workspace-wide carry a no-mode `.open`, and all 5 are the
table above.** The blind spot and the false-positive exclusion are the same
set, and today that set is 5/5 not-in-class. A denylist of receiver names would
have caught the no-mode case and would fail OPEN on the next library somebody
imports, against a gate walled at 0 with a deliberately empty exemptions file —
a false positive there is a red build, not a report.

The rule is the same shape as the one `locale_io_fix` already documents for
`**kwargs`: *"counts as UNKNOWN and is left alone … this repair may only ever
be conservative."* The blind spot is printed, not remembered.

### T3 — the repair

`locale_io_fix.sites()` gains the three forms behind `_text_mode()`. **One
classifier, still** — the gate, the sweep and the repair tool share it, so
there is no second copy to drift (Issue 755's rule, and the reason 829 was
built this way).

`repair()` needed **no change**: every new form accepts an `encoding=` keyword
and the insertion point is the same last-argument rule.

Arms: 12 new cases in `locale_io_fix.selftest()` — each new form positive, each
measured negative above, `mode=` as a keyword, a `b` mode by keyword, and the
idempotence re-run.

### T4 — the eleven sites

This repo's four, in the landing commit. The seven siblings, each cited by SHA
per the Issue-798 rule (*a cross-repo repair is not landed until it is
COMMITTED in the sibling repo*):

| repo | sites | commit |
|---|---|---|
| riir-train | 6 | `15db2c67` |
| riir-clippy | 1 | `54b999de` |

### T5 — the floors move, in the direction that proves the walk grew

`FLOOR_IO_CALLS` 450 → re-pinned on the measured count; the per-repo sweep
rows likewise. A floor that did not move would mean the new forms found
nothing.

### T6 — the sibling seam does NOT carry the same anchor (a measured negative)

The obvious next move after T3 is to assume the PIPE seam (Issue 778) is
anchored the same way and go widen it. **It is not**, and the probe was run
before the assumption was acted on:

* `universal_newlines=True` — `text=True`'s legacy alias with identical
  semantics — is **already** in `subprocess_encoding_gate.scan_text`, with its
  own positive arm. Whoever wrote that gate did not anchor to one spelling.
* `os.popen()` is always text at the locale codec and has no `encoding=`
  parameter at all, so it IS in the class and is outside the gate's
  `subprocess` predicate. Measured: **0 sites over 16 repos**. It is not added,
  because the remedy is not mechanical — there is no keyword to insert, the
  repair is "use `subprocess`" — and a wall with no load, no repair path and no
  measured risk is documentation pretending to be a gate.

Recorded because *"the rule landed in one instrument and never generalised"* is
this repo's most-repeated finding, and the reflex it breeds is to generalise
without re-measuring. Here the answer was no, and finding that out cost one
probe.

### What this issue does NOT claim

- That the FILE seam is now closed. `codecs.open` and `io.TextIOWrapper` are
  implemented and measured at **0 sites**, so they are walls with no load on
  them yet; `p.open()` with no mode is a **stated** blind spot with a measured
  cost of 0. Both are re-measured on every sweep run rather than remembered.
- That a name-based classifier is wrong in general. `read_text` is
  Path-specific and duck-typing it is correct. `open` is not, and the
  difference is the whole of T2.


## Issue 828 — CLOSED: Issue 823's anchor class, third position — the heading oracle was anchored to a DELIMITER SET, and the biggest unread family is this repo's own house style (2026-09-18)

**Status: RESOLVED same day (T1–T4). T4 answers Issue 823 T5 by PRICING it —
the blast radius of the widening is 2 rows workspace-wide, so the rule that is
unsound to widen is also not worth widening.**

### The symptom

Issue 823 T5 left one open question and one number:

> Open question, deliberately not answered here: whether the `resolved —`
> family admits a sound discriminator at all. AGENTS.md's answer is no, and
> nothing measured here contradicts it.

The number — read off the sweep, never typed — was ~213 unread heading
records. **T5 was answered by measuring the residual instead of arguing about
it, and 60 of the 213 are not the `resolved` family at all.**

### The finding

`_SELF_HEADING` is sound because of ONE rule, written in its own comment:

> nothing may sit between the number and its delimiter

Issue 823 found that rule anchored, silently, to the **kind LEADING the
line**, and moved it one POSITION over. It was anchored a second time, to the
**delimiter SET** — `(` for the leading form, `[:,]` for the dated one. The
workspace's most common title delimiter is the **em dash**, and it was in
neither set.

Measured (16 repos × the pinned documents, foreign-filtered, unread only):

| first char after the number | n | verdict |
|---|---|---|
| `—` (leading form) | **56** | delimiter, nothing interstitial — **sound to read** |
| `(` (dated form) | **4** | the leading form accepts it; the dated one did not — **sound to read** |
| `resolved` / `RESOLVED` | 64 | genuinely interstitial — Issue 781's family, untouched |
| `closed` / `close-out` / `CLOSED` | 28 | interstitial |
| `T1` … `T8`, `Arm`, `wave`, `phase`, `complete` | ~45 | interstitial |
| `follow-up`, `filed` | 3 | interstitial — the **pinned negative** |
| other | ~13 | interstitial |

The 56 are `## Issue 788 — the population-predicate registry …: CLOSED
(2026-09-14)`. That is **katgpt-rs's own house style, its own newest closes**,
in the repo that owns the instrument. The discriminator was never violated by
them; the pattern simply could not spell their delimiter.

⛔ **This is NOT the widening AGENTS.md calls unsound**, and the distinction is
Issue 823's, restated at a third position: the unsound widening is *dropping
the discriminator* (accepting text between the number and its delimiter). This
adds a delimiter and keeps the rule. `## Issue 043 follow-up — title` is
rejected by the new pattern exactly as `## Issue 043 follow-up (…)` is rejected
by the old one, and T2 pins that in both delimiters.

### T1 — the delimiter set

- `_SELF_HEADING_DASH`: the same discriminator, either position (the date
  prefix is optional), delimiter = `—`/`–`, or an ASCII `-` **that is
  space-separated on both sides**.
  - ⛔ The ASCII hyphen must NOT be accepted bare: riir-ai ships the live
    heading `## Issue 366-class (pos-uniform chunk forward) FIXED in riir-gpu`,
    where the hyphen is part of a WORD. The `\s+…-(?=\s)` shape rejects it and
    an arm pins the case.
- `_SELF_HEADING_DATED` gains `(` — the leading form always accepted it, and
  four riir-chain records (`## 2026-09-16 — Plan 062 (Proposal 010 D6/T1.2):`)
  were rejected by an asymmetry between two patterns documented as the same
  rule at two positions.
- The new pattern's group(3) is the **whole remainder**, the dated form's
  precedent, not the leading form's parenthetical: the foreign filter then runs
  over more text, which is strictly more likely to REJECT — the safe direction
  for the only path here that can SUPPRESS a finding.
  - ⛔ `_SELF_HEADING`'s own scope is deliberately **left alone**. Widening it
    to the whole remainder would reject `## Issue 059 (2026-01-01) — <sibling>
    did X`, which is the suppression Issue 754 landed; the "safe direction" for
    a NEW pattern is a regression for an existing one.

### T2 — the arms

`citation_drift_sweep.selftest()` arm 2, the four-negative fixture, gains the
delimiter axis on the SAME fixture rather than a second one:

- `## Issue 047 — title` reads (the new positive).
- `## Issue 048 follow-up — title` does **not** (the discriminator survives the
  new delimiter — the arm the whole issue rests on).
- `## Issue 049-class — title` does **not** (the bare ASCII hyphen).
- `## 2026-01-01 — Issue 050 (parenthetical): title` reads (the dated `(`).

The style-blind meter's expected pair moves with the fixture, so the width
bound still measures exactly the style gap.

### T3 — the numbers move in the SUPPRESSING direction only

`heading_allocated()` can only ever suppress. More numbers read ⇒ fewer
IN-LOCAL-RANGE rows and fewer false `⛔MISATTRIBUTED` — every ratchet in
`citation_drift_floors.txt` goes green-er or holds. Re-pinning a ratchet
DOWNWARD is deliberately **not** done here: they are ceilings, and tightening
them on one box's run is the diary AGENTS.md refuses.

### T4 — ANSWERED by pricing it: the blast radius is TWO rows

Issue 823 T5's open question — *"whether the `resolved —` family admits a
sound discriminator at all"* — was argued twice, semantically, by two
sessions, and never priced. The semantic answer is NO and stands: `resolved`,
`closed`, `T3`, `Arm C` and `follow-up` are the same SHAPE, no punctuation
rule separates commentary from allocation, and
`citation_drift_sweep.selftest()` arm 2 pins `follow-up` as a negative in
every position and delimiter.

**The question nobody asked is what the rule would BUY.** `heading_allocated`
is one member of a union; every other member answers from a FILE — in the
worktree, or recovered from `git log`. A record whose number is already known
that way contributes nothing whichever way the rule goes. Only the residue can
change a verdict, so **the residue is the blast radius**.

Measured over 16 repos: of **153 unread records, 2** contribute a number no
other oracle knows. One is riir-ai's `## Issue 969 resolved — …` heading. The
other is riir-clippy's heading, quoted —

`## Issue 097 resolved — …`

— and both are exactly the Issue-754 never-committed shape the heading path
exists for. Everything else is redundant.

So the 153 was never a backlog. It is a cost figure that is **98.7%
redundant**, and the case for adopting a rule this document calls unsound in
order to recover it does not survive its own arithmetic. ⛔ **Not "the class
is closed"** — the two rows are real and land as IN-LOCAL-RANGE, which is
UNDECIDED and never a pass. What is closed is the question of whether the
*count* justifies the *rule*.

⚠ Both figures are DERIVED per run and printed on
`citation_drift_sweep.py`'s own `heading oracle COST` line, per repo as
`novel=` beside `heading_unread=a/b`. They move whenever a sibling edits a
heading; two sessions already quoted the unread count at each other hours
apart as 209 and 213, each correct for its own run. **Take both from the
line, never from this paragraph.**

`issue_citation_gate.file_and_history_allocated()` is `allocated()` with the
heading path removed, split out so the heading oracle's own CONTRIBUTION is
measurable rather than argued about; `_heading_records()` is ONE walker under
both meters, because the width bound and the cost bound disagreeing about
what a record IS would make the pair useless. Armed both directions on arm
2's own fixture — a meter that always says 0 retires the question by looking
like an answer.

### ⛔ The arm that could not pass, and why it is a separate issue

T2's first run reported `(1, 5)` where `(3, 6)` was expected — 047 unread, 050
not even *shaped* — while the same fixture, written by hand in a scratch
script, returned `(3, 6)` and `{42, 47, 50}`. The fixture in
`citation_drift_sweep.selftest()` is written with a bare `Path.write_text(...)`,
which encodes with the **system locale**. This workstation is **cp874**, where
U+2014 encodes to the single byte `0x97`; the oracle then reads the file back
with `encoding="utf-8", errors="replace"` and sees U+FFFD. Every em dash in
every selftest fixture in that file was mangled.

The existing arms passed anyway — none of them depended on the dash — so the
corruption was invisible until an arm was written that did. **27 sites in that
one file; 155 in this repo; 273 across 8 repos**, measured. That is **Issue
829**, filed with its own gate; the 27 sites in `citation_drift_sweep.py` land
here because T2's arms are not real without them.

## Issue 823 — CLOSED: the heading oracle was anchored to a POSITION, and its own blindness meter was anchored to the same one; T5 ANSWERED by Issue 828 T4 (2026-09-18)

**Status: RESOLVED (T1–T4, T6 same day). T5 ANSWERED 2026-09-18 by Issue 828 T4 — not by settling the semantics, which stand, but by PRICING them: of the 153 unread records only 2 contribute a number no other oracle knows, so the widening this issue declined is also not worth doing. Read `heading oracle COST` on the sweep's own summary line.**

### The symptom

`citation_drift_sweep.py` red: `riir-clippy IN-LOCAL-RANGE 13 > pinned 12`.

The pin had been bumped `10 -> 12` the **previous day** (2026-09-16), with the
mechanism diagnosed correctly in the pin comment:

> Both new rows are riir-clippy's OWN Issue 113 … They read UNDECIDED because
> `heading_allocated()` cannot parse that repo's `## <date> — Issue NNN:`
> heading form … the repair is that repo's heading convention, which is its
> owner's call.

Twenty-four hours later the bucket was 13. That is AGENTS.md's own warning
made literal — *a pin file re-typed after every run is a diary, not a wall,
and re-typing it is how a real regression gets absorbed as "probably the box
again"*. The second bump was refused and the instrument was read instead.

The last clause of that diagnosis is also the part that was wrong: it is not
that repo's convention. Six repos write it, 74 records deep.

### The defect, and it is TWO defects stacked

**(a) The oracle.** `_SELF_HEADING` is sound because of exactly one rule:
nothing may sit between the number and its delimiter, so `## Issue 043
follow-up (…)` is rejected while `## Issue 043 (…)` is read. That rule is
about the text **after** the number. It had been anchored, silently and
incidentally, to the kind **leading the heading**. Six repos write the date
first — riir-clippy's `## 2026-09-16 — Issue 113: the auto-oracle` — and the
whole family was unreadable.

Consequence chain, measured end to end on a live specimen: riir-clippy's
Issue 113 exists in **no** `.issues/` file and **no** `git log` deletion — the
Issue-754 shape, where the HISTORY.md heading is the *whole* allocation record.
The oracle could not read it, so `113 not in mine`, so both citations of it
fell to IN-LOCAL-RANGE, so a per-repo **ratchet breached** and the sweep went
red. `.highwater` says 121: the repo provably owns the number.

**(b) The meter, which is the worse half.** `_HEADING_SHAPED` exists *only* to
measure what the oracle rejects — the width bound whose whole job is printing
the cost of (a). It was anchored to the **same leading position**, so it could
not see the family either, and it failed in the direction that reads as clean:

| repo | meter printed | actually unread |
|---|---|---|
| riir-chain | `heading_unread=0/1` — a **perfect** score | **20 of 21** |
| riir-dapps | `0/0` — nothing to measure | **22 of 23** |
| riir-clippy | 30 unread | 54 |

A blindness detector that cannot see a whole house style is not a width bound.
Workspace-wide it admitted **245** records where **341** exist.

### The repair (T1–T3)

`_SELF_HEADING_DATED` / `_HEADING_SHAPED_DATED` — the **same discriminator**,
at the position it was never applied. The number must be followed immediately
by its title delimiter (`:` or `,`, the date-led form's `(`).

⛔ **This is NOT the widening AGENTS.md calls unsound, and the distinction is
the entire justification.** That argument is against *dropping the
discriminator* — accepting `resolved` / `follow-up`, which no punctuation rule
separates from an allocation. Measured on the live corpus, the new pattern
rejects `## 2026-09-16 — Issue 152 resolved: …` and `## 2026-09-16 — Plan 064
T3 landed (…)` exactly as the leading form rejects their siblings. Arm 2's
pinned negative survives — **in both positions**, asserted by two new arms.

The foreign-repo filter runs over the whole date-led remainder rather than a
parenthetical: strictly more likely to reject, the safe direction for the only
path in this gate that can SUPPRESS a finding.

18 records became readable. Every one was read by hand at landing; all 18 are
genuine self-allocation records. Measured effect:

- workspace IN-LOCAL-RANGE **54 -> 27**
- oracle **110/319** read (was 92 read of a 245 the meter believed was the
  whole population)
- `riir-clippy` 13 -> **11**, `riir-neuron-db` 3 -> **2** — both ratchets
  **tightened** in the landing commit, never loosened
- `citation_drift_sweep.py` **PASSES**

### T4 — the stale prose

The sweep's own summary line asserted "Widening is UNSOUND" without the
qualifier that makes it true, and so did the comment on `_HEADING_SHAPED`.
Both now say which widening: dropping the discriminator, not adding a
position. A correct claim stated too broadly is how the next session concludes
the class is closed.

### T5 — OPEN: the residual, and the number this issue should NOT be read as

⚠ **The count is DERIVED per run — read it off the sweep's `heading oracle`
summary line, not from this section.** It was **209** at landing and **213**
hours later, both correct for the run that produced them; it moves whenever a
sibling repo edits a HISTORY heading. Two sessions quoted the two figures at
each other before noticing they were the same measurement at different times.

Those records remain unread, and they are the `## Issue NNN resolved — title
(date)` family the original Issue 781 measured. **This issue does not touch
them and must not be read as having closed Issue 781's class** — it removed a
*position* blind spot that was hiding underneath it, and in doing so made 781's
own figure honest for the first time (the denominator was understated by 96).

Open question, deliberately not answered here: whether the `resolved —` family
admits a sound discriminator at all. AGENTS.md's answer is no, and nothing
measured here contradicts it.

⚠ **Do not re-pin `max_in_local_range` for a heading-blind row again.** Two
sessions have now done it. The row is not a backlog entry; it is the instrument
reporting that it cannot read a record the repo owns. Read
`heading_unread=a/b` on the sweep's own per-repo line first — a repo at or near
`b/b` cannot have its IN-LOCAL-RANGE count trusted as an editorial quantity.

### T6 — the width bound had no floor, and its blind output is a PERFECT score

Found by writing (b) up: the paragraph claimed "a width bound gets its own
floor" and this one did not have one. `heading_style_blind` was printed every
run and asserted by nothing, so a regressed shaped pattern takes `shaped` to 0
and the summary line reads `0/0 records read, 0 UNREAD`. The instrument whose
whole job is reporting blindness reports **perfect coverage** when it goes
blind — which is (b) again, one level up, and it is the reason this issue
exists rather than a tidy symmetry.

`min_heading_shaped = 260` in `citation_drift_floors.txt`:

- **GLOBAL, not per-repo.** Per-repo is legitimately 0 in every repo with no
  self-allocation headings — the vacuous-floor shape, solved the same way the
  wasm32 sweep solves it (a reserved `TOTALS` row floors the population
  globally).
- **Pinned under a PARTIAL-clone measurement** (319 over 16 of 20), so a full
  checkout clears it by construction. The floor asserts the PARSER; it must
  never move because a repo was absent.
- ⚑ **Canaried, and it turned out to be two independent detectors.** An
  impossible pin (99999) fires the floor through its own comparison path,
  exit 1. A deliberately broken shaped regex is caught EARLIER, by the
  existing `heading_style_blind` arm, exit 2. Different failure modes, both
  covered — the arm sees a pattern that stopped matching, the floor sees a
  population that collapsed for any reason at all.
- An arm asserts the pin is present and > 0: a floor that is silently absent
  is the same as no floor, and `parse_pins` would have defaulted quietly.

### Postscript — the write-up reproduced the class, again

The T1–T3 AGENTS.md paragraph quotes riir-clippy's heading as a SPECIMEN, and
the instrument cannot tell a quoted citation from a live one: it landed a new
`IN-LOCAL-RANGE` row against katgpt-rs's own `max_in_local_range 0`. AGENTS.md
already records this trap for the Issue-780 write-up and already prescribes
the repair — name the true owner inside the citation's own 3-line window, not
a pin. Applied.

⚑ It also demonstrated Issue 797's split working as designed: the sweep's
per-repo DISPLAY read the fixed worktree (`0`) while the PIN adjudicated HEAD
(`1`), so the red persisted until the fix was committed. Exactly the intended
behaviour, and worth knowing before chasing it as a bug.

### Cross-repo

Nothing to file elsewhere. The defect and its repair are both in this repo's
`scripts/`; the six affected repos need no prose change, which is the point —
the previous diagnosis would have asked all six to rewrite their HISTORY.md
heading convention.

## Issue 829 — CLOSED: the FILE seam under Issue 778's PIPE seam: 273 text-I/O sites decode with the system locale, and the first one found was making a whole file of arms pass for the wrong reason (2026-09-18)

**Status: RESOLVED same day (T1-T6). 273 sites -> 2, and the 2 are outside the contract.**

### How it was found, which is the entire argument for a gate

Not by a census. Issue 828 added an arm to
`citation_drift_sweep.selftest()` asserting that the oracle reads a
dash-delimited heading. It failed — `(1, 5)` where `(3, 6)` was expected —
while **the same fixture, typed by hand into a scratch script, returned
`(3, 6)`**. Same bytes, same classifier, two answers.

The fixture is written with a bare `Path.write_text(...)`, which encodes with
`locale.getencoding()`. This workstation is **cp874**, where U+2014 encodes to
a single byte that a subsequent `read_text(encoding="utf-8",
errors="replace")` returns as U+FFFD. Every em dash in every selftest fixture
in that file had been silently corrupted since the day it was written.

⛔ **The existing arms passed anyway** — none of them depended on a dash — so
this is not "a latent defect". It is a test suite that had been **green for
the wrong reason**, and it stayed invisible until an arm was written whose
subject was the corrupted character. A locale write does not raise, does not
look wrong, and does not change a verdict until it does.

This is `subprocess_encoding_gate.py`'s class (Issue 778) one seam over — the
**FILE** seam rather than the **PIPE** seam — and it went unlooked-at for that
issue's own stated reason: macOS, `ubuntu-latest` and the M3 all speak UTF-8,
so **nothing that could notice ever runs it**.

### T1 — the census

AST over tracked `*.py`, 16 repos on this box:

| repo | sites | repo | sites |
|---|---|---|---|
| katgpt-rs | **155** | riir-chain | 7 |
| riir-train | 68 | riir-mmorpg-examples | 2 |
| riir-clippy | 27 | riir-shader | 2 |
| riir-ai | 10 | seal-game-editor | 2 |

**273 sites over 195 tracked `*.py` in 8 of 16 repos.** Two are not latent in
the "someday" sense: `riir-clippy/scripts/gen_dashboard.py` reads `git log
--pretty=%s` across the workspace and **every commit subject here uses an em
dash**; `riir-train/scripts/build_clippy_v3_corpus.py` reads and rewrites
corpus files with a bare `read_text`/`write_text` pair.

### T2 — the repair half, `scripts/locale_io_fix.py`

AST-driven, idempotent, 18 arms. Three things it learned the hard way, each
with an arm:

- ⛔ **`col_offset` is a UTF-8 BYTE offset.** Computing the insertion point in
  characters desyncs by two per em dash earlier on the line, and this repo's
  sources are full of them — the first run asserted on `' '` 59664 chars in.
- ⛔ **The newline style is part of the file.** Reading with universal
  newlines and writing `\n` flipped `suite_membership_audit.py` from CRLF to
  LF and turned an 11-site repair into a **515-line diff**, which is a repair
  nobody can review. It reads bytes, detects the style, and writes it back.
- ⛔ **The kwarg joins the last ARGUMENT, not the closing paren.** A call whose
  `)` is on its own line otherwise grows a leading-comma continuation
  (`        , encoding="utf-8")`) — valid Python, unreviewable, and 128 sites
  went in before anyone read one.
- A `**kwargs` splat is **UNKNOWN** and left alone: the caller may be supplying
  the encoding, and a repair may only ever be conservative.

### T3 — the verdict half, `scripts/locale_io_gate.py` (docs_gate CHECK)

Walled at **0** over a floored population, sharing `locale_io_fix.sites()` so
the repair and the verdict can never disagree about the rule (Issue 755).

- Population is tracked `*.py`, **not** `scripts/*.py`: 778's first real run
  found a site in `.agents/skills/doc-sync/tools/`, and this census found two
  more in `.benchmarks/` and `.agents/`. The seam is an idiom, not a directory.
- **Two floors that break separately** — `min_py_files` (the walk) and
  `min_io_calls` (the AST pass). The predicate floor counts *compliant* calls
  too, or it restates the ceiling and *a pin that restates its own input
  cannot fail*.
- **UNPARSED reds**, never folded into the pass column.
- Exemptions by membership with a reason per row, `locale_io_expected.txt`,
  **deliberately empty** — the repair is one mechanical pass, so a row would be
  a backlog wearing a pin (Issue 785). A stale row reds too.
- `--canary` arms the gate's own pin arithmetic (Issue 775's rule: the
  classifier's self-test cannot reach the verdict), including the exemption
  READER's permissive direction and the two floors *independently* — pooling
  them means repairing one hides the other.
- `--prove-fires 072a083b` is a known answer: 128 offenders at Issue 828's
  landing commit.

### T4 — the sweep half, `scripts/locale_io_drift_sweep.py`

Landed in the **same change** as the gate, because shipping one half is the
failure this workspace has now recorded nine times (Issues 777, 778, 793, 782,
783, 789, 797, 820).

⚠ Its ceiling was landed as a **RATCHET on the derivative** rather than a
wall, on a measurement: 118 sites sat in seven repos this session did not own,
and ratcheting a bucket nobody has read is Issue 785's forbidden shape. T6 then
read them, so every row is **0** today — but the file stays a ratchet by
construction, because the next repo to join the workspace joins with whatever
it has, and a wall would make that repo's arrival somebody's emergency.

Wired to both family-wide mechanisms at landing (the gate that requires it is
`sweep_advisory_membership_gate.py`): the Issue-797 worktree advisory and
Issue-822 head provenance via `head_delta` — per-file row independence holds
by construction, since the classifier is an AST pass over one module's source.

⚑ The provenance wiring earned its keep on its first run: with the repair in
the worktree and not yet committed, it reported **128 MASKED rows** — committed
defects this worktree hides — instead of a clean sweep.

### T5 — katgpt-rs repaired

155 sites: 27 in `citation_drift_sweep.py` landed with Issue 828 (its arms are
not real without them), 128 in the other 84 files here. `docs_gate.sh` green,
26 CHECKS.

### T6 — the 118 sibling rows, LANDED, with the SHAs

This repo's own rule: *a cross-repo repair is not landed until it is COMMITTED
in the sibling repo, and a record HERE claiming one must CITE THE SIBLING
COMMIT*. Six repos, one commit each, tracked `*.py` only (three of them carry
concurrent sessions' uncommitted `.rs` and `Cargo.lock` edits, which is why
the staging is by NAME and never `-A`):

| repo | sites | commit |
|---|---|---|
| riir-train | 68 | `39d0b8d4` |
| riir-clippy | 27 | `638fa224` |
| riir-ai | 10 | `4df109c65` |
| riir-chain | 7 | `bebf78a` |
| riir-mmorpg-examples | 2 | `36005d2` |
| riir-shader | 2 | `7a65dd4` |

Every row in `locale_io_drift_floors.txt` dropped to 0 in the same change, each
carrying its SHA. **Workspace measurement after: 273 → 2**, and the 2 are
`seal-game-editor`, a known-extra repo outside the contract that is 260
commits behind its upstream — the Issue-798 advisory names it on every run, so
repairing it from this box would be repairing a checkout rather than a repo.

⛔ The riir-train run is the one to read, because its raw diff was **+101/-72
over 33 files** and looked like the newline defect T2 had just fixed. It was
not: `--numstat` over the whole worktree includes a concurrent session's
`Cargo.lock` and `plan402_p1_gemma_boundary_sweep.rs`. Scoped to `*.py` it is
**31 files, +68/-68**, exactly 1:1. *Read a shared worktree's diff through the
population you changed, or another session's work reads as your bug.*

#### The gate's own foreign-repo mode, found by using it

`scripts/locale_io_gate.py ../riir-shader` — the form its own docstring
advertises — reported `parse FLOOR breached: 2 text-I/O call site(s) < 450`.
The floors are katgpt-rs's measured population and applying them to a repo
with two Python files states something true of the pin and false of the tree.
A named sibling is a **REPORT** now: floors and exemptions are not applied,
the final line says so, and the per-repo pins stay the sweep's
(`locale_io_drift_floors.txt`), which is the only place they mean anything.

## Issue 824 — CLOSED: the family gate watched the failure it was built for happen beside it (2026-09-17)

**Status: RESOLVED same day (T1–T3).**

### The symptom

`cfg_row_implication_drift_sweep.py` hard-red on three repos with **zero
content findings**:

    seal-game-editor: no pin row — a new repo must be pinned deliberately
    seal-online-remaster: no pin row — …
    seal-remake: no pin row — …

All three are acknowledged by `DOCS_GATE_KNOWN_EXTRA`. This is Issue 821's
symptom exactly, in a sweep Issue 821 closed.

### What actually happened

Issue 821 landed `sweep_population.pin_row_exempt()` and its close-out says
*"wired into **16** sweeps"*. The family is **19**. The three missed are:

| sweep | state | why it wasn't noticed |
|---|---|---|
| `cfg_row_implication` | **hard red, live** | nobody had run it since 821 |
| `restatement` | hard-fails, **latent** | its population is the `.proofs` subset and no acknowledged extra carries one *today* |
| `docs` | advisory only | it would merely *advise* pinning a repo the contract does not claim — an action the reader cannot correctly take |

⛔ **Those three are not a random subset.** They are precisely the three
AGENTS.md already names, from Issue 782, as the ones that "were not exempt —
they were quieter". The same three were missed twice, by two different repairs,
for the same reason both times: **a hand-grep of a family finds the loud
members.**

### The real finding — and it is about the gate, not the sweeps

`scripts/sweep_advisory_membership_gate.py` exists *because of this class*. Its
own docstring:

> That is the standing failure mode of this repo, recorded seven times now …
> **a rule landed in one instrument and never generalised.** Every previous
> instance was repaired by grepping the family by hand and fixing the copies.
> This one is repaired by making the family gate itself.

And then Issue 821 did it again, to a different mechanism, **while that gate
was standing and green** — because the gate governed `sweep_advisory()` *by
name*. The class is not "the advisory"; the class is **any mechanism every
member of the family must call**. A gate scoped to one instance of a class it
was written to prevent can watch the next instance happen next to it and report
a pass.

That is the eighth recorded instance (777, 778, 793, 782, 783, 789, 797, 821),
and the first where the *prevention itself* was the thing that didn't
generalise.

### The repair (T1–T3)

**T1 — wire the three.** All 19 now call `pin_row_exempt()`. `cfg_row_implication`,
`docs`, `restatement` green; the nesting follows Issue 821's own process note
(the exemption goes INSIDE `if pin is None:`, never flattened into it, or the
else-branch dereferences the missing row).

**T2 — `MECHANISMS` is a registry, not a name.** The gate takes
`slug -> (why, call-names)`; the verdict is computed **per mechanism and never
pooled**, because a pooled verdict is exactly what reports a sweep wired for
one and missing the other as wired — the 16-of-19 state. Adding the next
family-wide mechanism is one row, and the arms **walk the registry** rather
than hard-coding it, so a mechanism added later is armed by EXISTING.

**T3 — the pin key is `(mechanism, sweep)`.** A bare per-sweep row would excuse
that sweep from the mechanism nobody has looked at yet. This is the
`DOCS_GATE_KNOWN_EXTRA` asymmetry (NAMES, never `=1`) applied one file over. An
unqualified row and an unknown slug are both REFUSED. The file stays
**deliberately empty**, and both mechanisms now carry their own argument for
why a row there would be wrong.

Canaried two ways:
- **arms**, on a fixture wired for the advisory only — it must read UNWIRED for
  the exemption and WIRED for the advisory, independently;
- **production path**, by unwiring `citation_drift_sweep` in the real tree:
  `✗ UNWIRED [known-extra-exemption] citation_drift_sweep.py`, exit 1, naming
  the mechanism.

### Postscript — how it was found

Not by a sweep run, but by running **every** sweep and reading EXIT CODES
rather than grepping verdict lines for a glyph. The glyph grep had silently
matched nothing on the first two attempts and printed `<no verdict line>` 19
times, which reads like a harness problem rather than a finding.

⚠ The same run reported `numbering` at **exit 127 with a 0-byte log** — the
process never started, during a box-wide memory squeeze (free RAM 1.29 GB, a
concurrent 29.6 GB training job). It passes standalone. **A batch runner that
reports exit codes reports infrastructure failures in the same column as
instrument failures**; 127 with an empty log is the signature, and it is worth
re-running alone before believing it — the `x86_64_execution_matrix` rule
(every failure is RE-RUN ALONE, then adjudicated) reached by a different road.

## Issue 821 — CLOSED: Issue 815's marker never reached the sweep family: 8 sweeps red on 0 findings, and two live ratchet breaches sat behind them (2026-09-17)

**Status:** CLOSED 2026-09-17
**Severity:** a sweep that always reds is a sweep nobody runs — measured, twice now
**Owner:** this session

### The finding

Issue 815 added `DOCS_GATE_KNOWN_EXTRA` so a box carrying siblings the contract
does not claim could be READ. It landed in `skill_repo_set_gate.known_extra_state`
and in `sweep_population.population_verdict`, which is where every sweep gets
its final-line disclosure — and **not** in the per-repo pin loop those same
sweeps run six hundred lines earlier.

So on this box, with **both** documented markers set
(`DOCS_GATE_PARTIAL_CLONE=1` and `DOCS_GATE_KNOWN_EXTRA=seal-game-editor,
seal-online-remaster,seal-remake`), a sweep prints

    ✗ seal-game-editor  rs=305  cfg_sites=278  orphaned=0
          ✗ UNPINNED — add a row (or it can never red)
    ...
    ⚠ 3 known-extra repo(s) outside the contract — acknowledged by
      DOCS_GATE_KNOWN_EXTRA, not measured and not expected to be

The final line says the repos are not expected to be measured. The rows above
it demand a pin so that they can be.

### Measured, 2026-09-17 — with both markers set

| sweep | rc | why it reds |
|---|---|---|
| `orphaned_attr` | 1 | seal-* UNPINNED · **0 findings** |
| `percentile` | 1 | seal-* UNPINNED · **0 findings** |
| `markdown_fence` | 1 | seal-* UNPINNED · **0 findings** |
| `subprocess_encoding` | 1 | seal-* UNPINNED · **0 findings** |
| `trap_sentinel` | 1 | seal-* UNPINNED · **0 findings** |
| `platform_dead_code` | 1 | seal-* "a repo joined the population" · **0 findings** |
| `console_encoding` | 1 | seal-* UNPINNED **+ a real breach, below** |
| `instrument_reachability` | 1 | seal-* UNPINNED **+ a real breach, below** |
| `wasm32_surface` | 1 | seal-* UNPINNED + one NEW UNCOVERED row in an extra repo |

**Eight of nine red on repos where they found nothing.** And behind those reds,
two genuine ratchet breaches in a repo the contract *does* claim:

    console_encoding         riir-train  undefended 56 > pinned 53
    instrument_reachability  riir-train  unreachable 63 > pinned 61

Three new instruments that print a non-ASCII glyph and defend neither stream,
and two new scripts no root and no document names. Both are exactly what their
ratchets exist to catch. Both were invisible, because the reader who runs a
sweep and sees `✗` on three repos they were told to ignore stops reading.

⛔ **This is the second recorded instance of this precise failure.** Issue 793
found seven sweeps hard-redding on a known-partial box with
`DOCS_GATE_PARTIAL_CLONE=1` already set and every content assertion green, and
measured what was sitting behind them: *"the percentile sweep's Issue-777
findings, and four live citation drift rows."* The repair was
`sweep_population.py`, one shared mechanism. Issue 815 then added a second
marker and did not put it through that mechanism's other half.

### Why one shared helper and not 14 edits of the same idea

AGENTS.md, in the sentence this issue is an instance of: *"before fixing such a
class, grep the whole family and land the repair as one shared mechanism."*
Issue 820 wrote that down this morning and then fixed the marker in
`numbering_drift_sweep.py` alone — which is how the count got to 14.

### Tasks

- [x] **T1** — `sweep_population.pin_row_exempt(name)`: is this repo a declared
      known-extra? The acknowledged/stale split stays `population_verdict`'s,
      on the final line, because the loop only ever visits repos that are
      PRESENT — passing a visited repo is the acknowledged case by
      construction, and a helper that re-derives the split per row would be a
      second copy of it.
- [x] **T2** — the 14 uniform call sites + `toolchain_override_drift_sweep`'s
      `repo_flags()` (different shape, same rule) + fold
      `numbering_drift_sweep`'s Issue-820 copy onto the shared helper so the
      family is ONE shape.
- [x] **T3** — arms, in `sweep_population.selftest` where the rest of the
      marker's arms already live: exempt when declared, NOT exempt when
      undeclared (the wall must stay absolute for the next unregistered repo —
      the whole reason 815's marker takes NAMES rather than `=1`), and no
      marker at all exempts nobody.
- [x] **T4** — re-run the nine and record what the reds were hiding. The two
      riir-train breaches are **riir-train's to adjudicate**, not this issue's
      to repair; this issue owes making them VISIBLE.

### Not in scope

- **The two riir-train ratchet breaches.** Filed onward once visible. Read
  `instrument_reachability`'s standing note first: that repo's `scripts/` is
  almost entirely plan-scoped one-offs where the predicate OVER-CAPTURES, so
  63-vs-61 is a derivative question, not 63 repairs.
- **Whether the seal-* repos belong in the workspace.** Issue 815 recorded that
  as owner-owned and it still is. The marker makes the box readable; it does
  not make the box right.

### Landed

`sweep_population.pin_row_exempt(name)`, one definition, wired into **16**
sweeps: 13 uniform call sites, `toolchain_override`'s `repo_flags()` call site,
`platform_dead_code`'s set-difference, and `numbering`'s Issue-820 local copy
folded onto it.

Measured before → after, both markers set:

| | before | after |
|---|---|---|
| sweeps red | 9 of 14 | 4 of 14 |
| red on 0 findings | 8 | **0** |

The four still red are red for reasons that were **hidden behind the noise**:

    console_encoding         riir-train  undefended 56 > pinned 53
    instrument_reachability  riir-train  unreachable 63 > pinned 61
    cfg_gated                riir-ai     silent_now 2, load_bearing 1
    citation                 riir-ai     1 CROSS row

⚠ **`wasm32_surface` needed more than the pin row, and finding out why is the
generalisable part.** Its UNCOVERED membership check sits OUTSIDE the
`row is not None` branch, so excusing the pin row alone still produced a hard
finding about a repo the marker had just declared outside the contract. **An
acknowledged extra contributes no VERDICT, not merely no pin row.** The row is
still PRINTED — display reads the box, pins read the contract, which is Issue
797's split — and the membership file's other direction is untouched, so a
pinned row naming an extra repo still has to be removed deliberately.

### Two process notes, both of which cost time

⛔ **The mechanical 13-file edit was WRONG and crashed.** The flat guard
`if row is None and not pin_row_exempt(name):` sent the exempt case into an
`else:` branch that dereferences `row` — `TypeError: 'NoneType' object is not
subscriptable`. The correct shape nests the exemption INSIDE `if row is None:`
so the else-branch precondition is preserved. It failed loudly on the first
run, which is the good direction; a scripted edit across 13 files is worth a
`py_compile` sweep and one real execution before it is worth trusting.

⛔ **Python `write_text` flipped every file to CRLF**, twice, turning 2-line
changes into whole-file rewrites (`345 insertions, 345 deletions` on a 345-line
file). `git diff --numstat` after any scripted edit; write BYTES, or pass an
explicit LF newline argument.

## Issue 820 — CLOSED: the numbering sweep is blind to Issue 795's class, so 15 repos read `dup=0` over 113 collisions (2026-09-17)

**Status:** CLOSED 2026-09-17
**Severity:** the sweep prints a confident green over the MAJORITY case
**Owner:** this session

### The finding

`scripts/numbering_gate.py` grew `historical_collisions()` at Issue 795,
against a measured reason written into its own docstring:

> A document closed under the noise-reduction rule is DELETED, so a double
> allocation where both sides have closed leaves NOTHING on disk and reads as
> clean — and every number this repo allocates is expected to end up removed.
> Measured 2026-09-15: 70 collisions in scope, 9 of them from one 57-commit
> divergence, and this gate had never reported one.

`scripts/numbering_drift_sweep.py` — **the cross-repo verdict half of that
exact gate** — never got it. It calls `ng.tracked_paths()` and `ng.scan()`,
both of which read the WORKTREE, and reports `dup=` over what is on disk. So
the class Issue 795 exists for is measured in **one repo of sixteen**, and the
sweep's summary line says `0 tracked duplicate(s)` for the other fifteen.

This is the eighth recorded instance of the never-generalised shape
(Issues 777, 778, 793, 782, 783, 789, 797) and the second one where the
un-generalised half is the *verdict for everybody else*.

### Measured, 2026-09-17

`numbering_gate.historical_collisions(repo, SERIAL_DIRS)` run over the derived
contract population, unchanged, on this box:

| repo | collisions | ≥ 700 | numbers walked | time |
|---|---:|---:|---:|---:|
| katgpt-rs | 70 | 9 | 1407 | 0.7s |
| riir-ai | **71** | **3** | 1435 | 0.4s |
| riir-train | 11 | 0 | 553 | 0.5s |
| seal-game-editor | 10 | 0 | 337 | 0.4s |
| riir-chain | 6 | 0 | 225 | 0.3s |
| seal-online-remaster | 5 | 0 | 72 | 0.1s |
| riir-mmorpg-examples | 4 | 0 | 135 | 0.2s |
| riir-clippy | 3 | 0 | 443 | 0.2s |
| riir-neuron-db | 3 | 0 | 111 | 0.2s |
| (7 others) | 0 | 0 | 196 | — |
| **TOTAL** | **183** | **12** | **4914** | **3.7s** |

katgpt-rs's 70 are the ones its own gate already pins. **113 are in repos the
sweep covers and calls clean**, and **3 of those are at or above the era
boundary** — live ambiguities in prose people are writing today, in riir-ai:

    .issues/702  702_l2_normalize_suite_order_numeric_divergence
                 702_persistent_grid_stride_gemv
    .issues/753  753_qwen38_cudarc_kv_dtype_f32_lever
                 753_webrtc_tier_gate_too_narrow_for_whip_client_server
    .issues/959  959_feature_gap_dead_code_consumer_sweep
                 959_heart_icon_text2d_raster_hierarchy_traps

⚠ A prior session's hand-off recorded riir-ai as carrying "3 unpinned
historical numbering collisions". That figure was the **tracked-only** view
arrived at by eye; the instrument says **71**. The hand count was not wrong
about the three that matter — it was wrong about the population, which is the
reason to have the instrument point at every repo instead.

**3.7s for the whole workspace.** There was never a cost argument for the
omission either; it is a step that was skipped, not a judgement that was made.

### Tasks

- [x] **T1** — `audit()` gains a `hist` class and the `n_numbers` population
      that produced it, from `ng.historical_collisions` **imported, not
      re-derived** (the `-M` rename exclusion is subtle enough that a second
      copy is a second thing to get wrong — Issue 755).
- [x] **T2** — two new columns in `numbering_drift_floors.txt`, and they must
      fail differently:
      - `max_hist` — a **RATCHET at the measured count**, the existing
        `max_dup` / `max_resets` doctrine. Not a wall: resolving one is a
        citation-weight arbitration (Issue 724 T2), 113 of them are in repos
        this session does not own, and a pin of 0 would red on landing and be
        ignored — the cries-wolf failure `docs_gate.sh`'s preamble records.
        Not a membership set either: the gate's ≥700 wall needs a reason per
        row, and 15 repos' worth of invented reasons is a backlog wearing a
        pin (Issue 785).
      - `min_numbers` — a **FLOOR on the HISTORY walk**, which is a genuinely
        different instrument from `min_files`. `min_files` counts tracked
        files on disk; `min_numbers` counts distinct numbers recovered from
        `git log -M --diff-filter=D` **plus** disk. A `git log` regression
        leaves `min_files` untouched and takes `n_numbers` down to the on-disk
        count, and every ceiling then passes green over a blind instrument.
- [x] **T3** — katgpt-rs's row does not RESTATE its gate. The sweep imports
      the same function, so a count comparison would be true by construction —
      *a pin that restates its own input cannot fail* (the Layer 2c rule). It
      asserts instead that the **gate's own verdict** on those rows is clean:
      `collision_verdict(rows, *parse_collision_pins(...))` must return no
      failures, so a stale membership pin or a legacy ratchet drifting in
      `number_collisions_expected.txt` reds the sweep too.
- [x] **T4** — arms. The existing `selftest()` pins the scope split and the row
      parser; extend it over the new column's own arithmetic, which no
      classifier reaches (Issue 775's rule): the ratchet in BOTH directions
      (over → fail, under → "re-pin DOWN" note), the walk floor firing, and a
      reasonless/short row being REFUSED by the widened parser.

### Not in scope

- **Resolving any of the 113.** Each is an Issue 724 T2 citation-weight
  arbitration over a corpus in a repo another session owns. The ratchet makes
  the standing backlog visible and reds the next one; that is the repair this
  issue owes.
- **riir-ai's 3 above-boundary rows.** They are real and they are riir-ai's to
  adjudicate. Recorded here so the number is written down somewhere; the
  ratchet is what keeps them from growing to 4.
- **A per-push gate in riir-ai.** riir-ai has no `docs_gate.sh` CHECKS array
  (`check_validation_gate.py`'s own measurement, Issue 789 T4). The workstation
  sweep is the right cadence.

### Landed

Sweep line is now `files= nums= dup= hist= stale= malformed= resets= unbumped=`;
first green run with the class measured workspace-wide reports **0 tracked
duplicates against 184 historical collisions over 16 repos**. (184, not the
183 this issue was filed on: a concurrent session allocated `.issues/819` the
same day — see below.)

**T5, unplanned — a live above-boundary collision, found by running the gate
this issue's work made me run.** `numbering_gate.py` was RED on `develop` and
nothing had noticed: two sessions allocated 819 off one counter value. Hand
census 10 sites / 0% UNRESOLVED, splitting **7 to the sigmoid prior-logit
forecast issue** (`.research/566` x5, `.research/258`, `.research/392`) against
**3 to the x86_64 lint lane** (`AGENTS.md` x2, `HISTORY.md` x1); citation_weight
agreed 6-3. The sigmoid issue KEEPS 819 under the Issue 724 T2 rule.
- ⚠ The margin was hand-checked because `lane` is a distinctive token of the
  winning file and AGENTS.md's Layer 2c prose is about a lint LANE — the exact
  shape that would hand the loser's own sites to the winner. It did not.
- The loser is CLOSED and REMOVED, so there is no file to `git mv` and the `-M`
  rename exclusion has nothing to exclude: **this pair is permanent in the
  recovery walk.** Repair is the pin plus a three-site disambiguation in the
  losing prose (Issue 794's rule), not a renumber.

**T6, unplanned — Issue 815's marker did not reach the sweeps.**
`DOCS_GATE_KNOWN_EXTRA` was honoured by `population_verdict` and not by the
per-repo pin loop, so a box carrying acknowledged extras printed "not measured
and not expected to be" on its final line while reddening those same three
repos for having no pin row. Excused by NAME now, both directions asserted.

**T7, unplanned — riir-clippy's `.research` allocator was stale at 176 with
177 on disk**, so the next allocation was *guaranteed* to collide. Confirmed
against origin first (Issue 798 — this box was 7 behind and the defect was
upstream too). Repaired at riir-clippy `672ba8a8`.

### Verification addendum (2026-09-17) — the floor was argued, then EXECUTED

Prompted by katgpt-rs-9a's Issue 823 finding: `heading_style_blind` printed
`0/0 records read` — a **perfect** score — when its own regex went blind. The
generalisable question they posed is worth applying to every triage quantity
here: *what does this print when it breaks?*

T2's `min_numbers` was **argued** ("a git-log regression leaves `min_files`
untouched and collapses `n_numbers` to the on-disk count") and never run. The
two `selftest` arms pin the mechanism on a synthetic fixture — that recovery
WIDENS the population, and that a collapsed population trips the ceiling — but
neither can tell you whether the **real** pins are tight enough that a real
blinding actually crosses them. Arms use invented numbers.

Executed by blinding `citation_weight.removed_by_number` exactly as a failing
`git log -M --diff-filter=D` does today (it returns `{}` on error), across the
derived population:

| repo | nums | blinded | `min_numbers` | hist | blinded | fires? |
|---|---:|---:|---:|---:|---:|---|
| katgpt-rs | 1412 | 1047 | 1200 | 71 | 0 | ✓ |
| riir-ai | 1436 | 678 | 1200 | 71 | 0 | ✓ |
| riir-chain | 225 | 80 | 180 | 6 | 0 | ✓ |
| riir-clippy | 444 | 332 | 350 | 3 | 0 | ✓ |
| riir-dapps | 92 | 44 | 70 | 0 | 0 | ✓ |
| riir-game-sdk | 42 | 9 | 33 | 0 | 0 | ✓ |
| riir-mmorpg-examples | 135 | 32 | 108 | 4 | 0 | ✓ |
| riir-neuron-db | 111 | 52 | 88 | 3 | 0 | ✓ |
| riir-shader | 23 | 5 | 18 | 0 | 0 | ✓ |
| riir-train | 553 | 372 | 440 | 11 | 0 | ✓ |
| riir-viewbridge | 8 | 0 | 6 | 0 | 0 | ✓ |

**All 184 collisions vanish under the blinding and not one repo reports a green
zero** — every non-zero floor reds first. `riir-auth` and `riir-kat` carry
`min_numbers 0` because they have 3 and 0 numbers respectively; there is
genuinely nothing to floor there, which is a disclosed gap rather than a
silent one.

⚠ Deliberately NOT landed as a tracked script. It is a one-shot validation of a
frozen design argument — the `--prove-fires` shape — and a tracked
`scripts/*.py` that no root names is the `instrument_reachability_gate` finding
one level down. The durable artifacts are the floor and this table.

### What this does NOT claim

- The 113 sibling collisions are **not** adjudicated. The ratchet makes them
  visible and reds the 114th; each one is an Issue 724 T2 arbitration in a
  repo another session owns.
- Seven repos are absent from this box and their two new columns are
  **UNMEASURED**, pinned `0 0` and saying so on the row. `max_hist = 0` will
  red for any of them that has collisions on the first canonical-workstation
  run — deliberately, because a ceiling invented to be comfortable is a ceiling
  that never fires. Re-pin all seven from one full-checkout run.

## Issue 822 (2026-09-18) — CLOSED. An UNCOMMITTED row was counted into a RATCHET; all 19 sweeps adjudicate against HEAD now, and the mechanism is GATED

**The defect.** Every `*_drift_sweep.py` walks the WORKING TREE, and this
workspace runs five-plus concurrent sessions against shared worktrees. So a
count compared to a tracked ceiling was a function of one box's checkout AND
one session's uncommitted index. Measured on `console_encoding`: `undefended
56 > pinned 53`, where one of the three new rows sat on a file `git log` could
not see at all — staged by another session mid-commit. The pressure to type 56
into the pin is the whole hazard, and nothing in the run distinguished the case.

**The repair, in three instruments** (`worktree_state.py`), chosen by the
CLASSIFIER's shape and never by preference:

| instrument | for | cost |
|---|---|---|
| `head_delta` | per-file classifiers | \|dirty ∩ population\| `git show` calls, **zero** on a clean run |
| `head_overlay` + `delta_of` | cross-file, re-classified whole from a dict | same reads, one extra classification |
| `head_tree` | MULTI-SEAM classifiers — `git grep` + `git ls-files` + direct reads | a materialised HEAD checkout: **22-32s per dirty repo**, 0.04-0.26s clean |

`head_tree` (Issue 822 T5g) `git archive`s HEAD, extracts it, and runs `git
init` + `git add -A -f` so the tree answers `git grep` and `git ls-files` as a
real checkout does — the classifier runs UNMODIFIED. `-f` is load-bearing: an
extracted `.gitignore` would otherwise exclude content HEAD tracks. It gained
`paths=` (narrow the archive; riir-train 604 MB → 30.6 MB, worth **2.2x** end
to end and not the 20x the size suggests) and `extra_dirty=` (widen the
TRIGGER for a sweep whose walk is `os.walk` rather than `git ls-files`, so an
untracked file is in its population and produces zero `git status` dirt). Its
checkout is named after the SOURCE repo, because `len_derived` keys every row
on the directory name and two repos materialised at once would both be `head`.

**Rules that came out of it, each measured rather than reasoned:**

* ⛔ **The VERDICT belongs in the row KEY.** `delta_of`/`head_delta` file a
  key-matched row as COMMITTED carrying the WORKTREE's object, so any field the
  key omits is a field where the worktree silently overrides HEAD — and every
  ceiling in the family partitions by verdict. Reached independently in four
  sweeps (T5f, T5g, T5i, T5j).
* ⛔ **A FLOOR is a pin too.** Issue 797 measured this class on a POPULATION
  (607 → 601 citations), not on a finding. HEAD's walk size is DERIVED rather
  than re-listed: a staged-new file makes HEAD's walk one SMALLER and a
  worktree deletion one LARGER, and only the second is reachable by something
  nobody committed.
* ⛔ **The classes can split by ORACLE, not by instrument.** `numbering`'s
  `dup`/`above`/`malformed` read the worktree, `hist`/`resets` read `git log`
  and a dirty tree cannot move them, and `unbumped` is a WORKTREE quantity by
  construction — adjudicating it to HEAD would destroy it.
* ⛔ **A census over ONE REPRESENTATION is blind to what that representation
  omits.** The task re-derived the exposed set by grepping every
  `> row["max_*"]` and concluded `docs_drift` had no ceiling; it has a wall at
  0 written `if b_mis:`. Issue 787's finding, committed by the census looking
  for instances of it.
* ⚠ **An arm over inputs that cannot express its rule passes for the wrong
  reason.** A line-in-key perturbation over a fixture whose content never moves
  is an EQUIVALENT mutant; `cfg_gated`'s `features` field can never vary
  between two findings because the finding IS "no `required-features` row";
  `required-features = []` is not in `cfg_row_implication`'s population at all.
  Three fixtures had to be rewritten before their arms asserted anything.
  katgpt-rs-54 hit the same class independently the same night: an 827 fixture
  used a number NOBODY owned, where `is_qualified` returns True by construction
  ("nothing to be consistent with"), so the arm asserted the suppression of a
  finding that never existed.

⛔ **The generalisation those two findings share, which is the one to carry:
reasoning from ONE VIEW of a CONTENDED thing returns the reassuring answer.**
A census over one representation cannot see what that representation omits; a
verification over a snapshot cannot see what the snapshot omits; a fixture that
cannot express its rule cannot fail. All three were live in this workspace on
2026-09-18, in two sessions, and every one of them read as clean.

**A diagnostic rule, bought by a peer session's hour** (katgpt-rs-54): it ran
the numbering sweep during ~90 seconds of another session's `git rebase` and
got two ceiling breaches that were TRUE of that instant and false of every
commit. It then verified the worktree, HEAD's blob, 30 commits and
`.git/rebase-merge`, found nothing, and concluded the instrument was broken.
**Every one of those checks inspects STATE, and state had moved.** A
transient-worktree finding is in no commit by definition, so scanning commits
is structurally incapable of separating "phantom" from "true but gone" — it can
only return the reassuring answer. **The one cheap decisive test is to RE-RUN
THE INSTRUMENT.**

**Closed with a REGISTRY row, not a tick** (T6): `head-provenance` in
`sweep_advisory_membership_gate.MECHANISMS`, four entry points
(`head_delta`/`head_overlay`/`head_tree`/`head_text` — the last so
`citation_drift_sweep`, which carries the inline original, is credited as WIRED
rather than exempted). 19/19 on all three mechanisms, exemption file still
empty. Registered only AFTER the fan-out completed: a registry row over unwired
sweeps reds the docs gate on `develop`, and pinning them meanwhile is a backlog
wearing a pin (Issue 785). **Take the family size and the wiring verdict from
that gate's PASS line, never from a count in prose** — this issue is the class
where that goes wrong, and its own tables went stale twice while being written.

⚑ **The first live MASKED row, four hours after the fan-out landed
(2026-09-18).** AGENTS.md records MASKED as "**0 today, which is a measurement
and not an absence of the class** — nothing had ever looked". It is 1:

    ⛔ riir-ai/scripts/repair_t1_outcomes_json.py  [MASKED — committed, and this worktree hides it]
    ✗ riir-ai   scripts=9  roots=4  unreachable=7 (8 committed, 1 MASKED)
        ✗ unreachable 8 committed > pinned 7

The script is committed (riir-ai `7e8a6b35a`, its Issue 976 repair); the
AGENTS.md line that makes it REACHABLE is uncommitted on this box. So at HEAD
it is an instrument no root names, and **without this adjudication the sweep
would have read the worktree, counted 7, and PASSED** — a committed defect
reported clean, invisible until somebody ran the sweep on a box without that
edit. The repair is Issue 798's class from the other side and needs no pin
change: the fix was already written, and what was missing was the COMMIT. The
documentation is the artifact here, which is why "a repair is not landed until
it is committed" has to cover docs and not only code. Cleared at riir-ai
`f9f183a0f` by committing the doc, not by raising the ceiling.

⚠ **Read the specimen correctly, because MASKED's definition invites reading
every instance as concealment.** This row was concealed by its own UNCOMMITTED
REPAIR, not by an unrelated edit: script lands undocumented → sweep reds →
author writes the doc → the doc masks the finding → commit clears it, a window
of minutes. The mildest possible version of the class, and the label is what
sent its author to commit rather than stop at a green worktree run.

⛔ **And the counterfactual is the part that generalises, stated hard:** the
worktree run passes **because the fix was on that one box**. Every other
checkout would have red. That is the same "reds on every box but the one and
the hour that produced it" shape this document already names for floors
measured against unowned content — which makes the first live MASKED specimen
an argument that the class is not RARE, only previously INVISIBLE. It needed a
sweep that reads HEAD to be seen at all.

⛔ **The workspace hazard it kept tripping over, worth more than the issue.**
Every session commits as `katopz <katopz@gmail.com>`, so a SHA's author line
identifies NOBODY, and "Nth of mine" in a commit body identifies somebody you
cannot resolve. Reading one such line as a messaging peer's claim produced a
wrong WORK SPLIT — two sweeps left untouched by both sessions that were
talking, each believing the other had them. Even the session NAME is not a key:
`ListAgents` showed TWO live sessions called `katgpt-rs-54`, distinguished only
by ref, and a fourth session existed that two independent eliminations had both
missed. The only reliable signal is an explicit `Session: <name>` line in the
body. ⚠ Second-order specimen from the same hour: a claim of *"ListAgents shows
only the two of us live"* was asserted, in a message arguing against inferring
things from git, **without having run `ListAgents`**. A claimed verification is
exactly as unreliable as the inference it replaces when the verification was
not performed.

## Issue 819 (2026-09-17) — CLOSED. The x86_64 arm was LINTED by nothing; the finding is not the 30, it is the sibling

⚠ **819 is held by two documents.** A concurrent session allocated it the same
day off the same counter value for the sigmoid prior-logit forecast issue,
which KEEPS the number on weight (7 of 10 inbound sites, 0% unresolved —
Issue 724 T2). This record is the *x86_64 lint lane*; the adjudication is
pinned in `scripts/number_collisions_expected.txt`.

Found while measuring clippy on the Issue 808 workstation: `cargo clippy -p katgpt-attn --all-features --lib` with
`RUSTFLAGS="-C target-feature=+avx2"` returned **30** warnings, every one `E0133` / `unsafe_op_in_unsafe_fn` on
edition 2024, every one in `crates/katgpt-attn/src/dash_attn/channel_aware.rs`, every one inside the x86_64 AVX2
arm — against **0** on the avx2-off arm.

**The finding is the SIBLING.** That file carries two transcriptions of one kernel:

| arm | cfg | body |
|---|---|---|
| `simd_dot_neon` | `target_arch = "aarch64"` | `// SAFETY: …` + a whole-body `unsafe { … }` |
| `simd_dot_avx2` | `all(target_arch = "x86_64", target_feature = "avx2")` | **neither** |

The edition-2024 repair was made on one arm and not on its twin, and the reason is mechanical rather than an
oversight a reader could catch: `full_gate.sh` is macOS/**aarch64**, so it compiles and lints `simd_dot_neon` while
`simd_dot_avx2` compiles to **nothing** there — under `--all-features` too, because the gate is arch-gated rather
than feature-gated. *A repair applied to the arm somebody can see is not a repair; it is a measurement of which
arms are visible.*

**Why every lane was blind, one reason each.** Layers 2/3/6 of `full_gate.sh` are macOS/aarch64. Layer 2b and
`wasm32_gate.yml` build a third triple. `test_gate.sh` does not lint, and runs at default target-features.
`x86_64_execution_matrix.sh` — landed the day before, 2026-09-16 — is the right arch with `+avx2` ON and
**EXECUTES** rather than lints. That last one is the sharp case: it closed AGENTS.md's `compile vs EXECUTE` row for
x86_64 and left the **inverse** hole standing for a day, an arch whose execution was covered and whose lint surface
was not. Nothing in the tree had ever declared the x86_64 lint surface a surface, so `--allow-partial-platform`
could not name it as unmeasured either — a gate cannot report a lane it does not have.

**`target_feature = "avx2"` is a SECOND gate on top of `target_arch`**, which is Issue 737's wasm32/simd128 shape
one platform over — there, the simd128-off arm was clean and the on arm had 14 findings; here, 0 and 30. Both arms
run in the lane for that reason.

**T2 — the repair.** Whole-body `unsafe { … }` plus a `// SAFETY:` comment, the shape the NEON sibling has carried
since it was written. Not a `cargo heal` job: the healer is silent on this class and the repair is one block, not
30 edits. Measured after: 0 warnings, `katgpt-attn --all-features --lib` 427/0 unchanged. Diff 46/43 because the
body is indented into the block to match the sibling — verified by `git numstat` not to be a whole-file
line-ending rewrite.

**T3 — `full_gate.sh` Layer 2c**, mirroring Layer 2b: derived `-p` list off the positive `target_arch = "x86_64"`
surface (so a new x86_64-bearing crate joins by EXISTING), BOTH avx2 arms, `-D warnings`, `--keep-going`, an
instrument floor that refuses a confident zero, and a missing target reported as a PARTIAL gate that refuses
without `--allow-partial-platform`.

- ⛔ **The TRIPLE is the one real design decision, and the lane discloses it on its own verdict line.** 2b names
  `wasm32-unknown-unknown` literally because there is exactly one; x86_64 has three in play here and they differ
  in `target_os`, which is Layer 2's whole subject. Host triple when the host is x86_64 — the configuration T1 was
  measured in, and the only one needing no extra `rustup target add` — and a named cross triple otherwise
  (`x86_64-apple-darwin` on Darwin, `x86_64-unknown-linux-gnu` elsewhere). A green whose triple is not printed
  means something different on every box. `grep -qx`, not `grep -q`, against `rustup target list --installed`: a
  substring match on triples is a wrong answer waiting for the next triple.
- ⚠ **`--all-features`, unlike 2b, and it is not optional here.** `katgpt-attn` has `default = []`, so at default
  features `dash_attn` — and with it BOTH files the 30 findings live in — compiles to nothing. A default-features
  version of this lane is the green ZERO it exists to catch. It also makes the layer exactly Layer 3's feature
  coverage re-run on the x86_64 arch, which is the claim being RESTORED rather than a new one, and it removes a
  hand-typed feature list beside the four named targets, each of which carries a `required-features` row.
- **The residue pin is the non-`src/` surface MINUS what a named row covers, expected EMPTY.** Pinning the four
  paths themselves would restate the table one line down, and *a pin that restates its own input cannot fail.*
  Proven in four arms including the `set -euo pipefail` all-filtered case (`grep -v` exits 1 when it outputs
  nothing). The other direction — a row whose file is gone — is caught by cargo itself, so the table cannot only
  ever loosen either.

**T4 — the other packages.** The lane's population is five (`katgpt-attn`, `katgpt-core`, `katgpt-pruners`,
`katgpt-tokenizer`, `katgpt-types`) plus four named non-`src/` targets, over 28 x86_64 files and 6 avx2 files.
Measured: **0 findings** outside `channel_aware.rs`, both arms. Recorded as a measurement and not an expectation —
Issue 737 measured 14 on wasm32 where 0 was expected.

**Two-sided, known-answer validation.** The lane is green on the repaired tree and reds with exactly `30 previous
errors` on `HEAD~1`'s `channel_aware.rs`. A lane that would be green either way certifies nothing.

⚠ **An operational hazard the lane deliberately does NOT paper over.** `--all-features` turns on
`katgpt-tokenizer`'s optional `good_lp`, hence `highs-sys`, hence cmake and a C++ toolchain. On this box, under a
g200 training run and three resident agent sessions, the 24-way `cmake --parallel` died with
`cl : command line error D8040` and took the lane red on a defect that was not in any Rust file;
`CMAKE_BUILD_PARALLEL_LEVEL=4` built the same package clean. It is NOT capped in the layer: Layer 3 has the
identical exposure, and capping one and not the other is this repo's most-repeated drift shape. The failure is
loud and self-identifying and the remedy is one line, both recorded at the layer.

## Issue 819, second holder (2026-09-17) — the sigmoid prior-logit lane KEPT the number; RESOLVED (file removed)

The OTHER 819 — the one that won the same-day dual allocation (7 of 10 inbound
sites, 0% unresolved, Issue 724 T2; adjudication pinned in
`scripts/number_collisions_expected.txt`). T1–T3 + T5 LANDED with Bench 813
(`.benchmarks/813_prior_lane_margin_goat.md`) G1–G4 ALL PASS: the sigmoid
prior-logit lane + sink-stability forecast (Research 566, arXiv:2601.15380)
shipped as two OPT-IN features under the no-default-consumer rule. T4
(owner-gated) closed unclaimed. Full task record: git history
(`819_sigmoid_prior_logit_lane_sink_margin_forecast`, removed per
noise-reduction); the surviving records are Bench 813 + the collision pin row
+ this section. Citations saying "Issue 819" in the research thread
(.research/566, 258, 392) mean THIS lane, never the x86_64 lint lane (whose
record is the 819 section above, disambiguated in place per Issue 794).

## Issue 815 (2026-09-17) — CLOSED by its own criterion: the box is green. Option 2 landed as `DOCS_GATE_KNOWN_EXTRA`, and options 1 and 3 remain the owner's

⚠ **Read this as the REVERSIBLE option taken in the absence of a decision, not as the decision.** The issue asked the
owner to choose between joining the three `seal-*` repos to the contract (1), a known-extra marker (2), and box
hygiene (3). Option 2 is the only one that touches neither the 20-repo contract nor `seal-online-remaster`, whose
main/develop are READ-ONLY by standing instruction — so it is what landed. Options 1 and 3 are unaffected and
cheap to switch to: option 3 in particular makes the marker unnecessary, and dropping the marker restores the
previous behaviour exactly.

The marker is the MIRROR of `DOCS_GATE_PARTIAL_CLONE`, one bucket over: that one covers repos that are ABSENT, this
covers repos that are PRESENT and unregistered. ⛔ It takes **NAMES, never `=1`** — `=1` would excuse the next
unregistered repo too, and UNREGISTERED means "a repo JOINING", the posture this workspace keeps loud on purpose.
It reds in BOTH directions: a name gone from the box, or since registered in `repo_set.txt`, is a STALE
acknowledgement and fails, so the marker cannot only ever loosen. Never auto-detected, for the partial marker's own
reason. One classifier (`skill_repo_set_gate.known_extra_state`), consumed by all three call sites — the Issue-793
"land it in the family, not in one instrument" rule, which this repo records seven prior failures of.

**Three defects found while landing it, all in the instruments rather than the contract:**

1. ⛔ **`skill_repo_set_gate` printed "16 of 20 canonical repos present" over 13 canonical + 3 extra** — a scope line
   crediting the extras as canonical and understating the absence by exactly their number. That is the
   partial-set-as-whole-one defect the gate exists to catch, committed by the gate's own display. The walk is not
   the canonical set once a box carries known-extra repos, and the two must not be conflated.
2. ⛔ **Both selftests read the AMBIENT environment while building a SYNTHETIC workspace**, so setting the new marker
   to this box's three real repos failed two Issue-765 arms whose synthetic walk has never heard of them — the gate
   reported INSTRUMENT-unreadable on a CORRECT invocation. Both markers are now saved, cleared and restored around
   the arms in both modules; the arms must measure the arms.
3. ⛔ **`population_sync_gate --canary` died with UnicodeEncodeError and no verdict** on this cp874 console: the
   stream defence lived inside `main()` only, so the canary path printed `✓` straight at the console encoding. That
   is the Issue-804 class exactly — not findings *unknown* but findings *unlooked at*. `console_safe.apply()` moved
   to the `__main__` entry both paths go through.

⛔ **A fourth, caught by `arm_reach_gate` and closed by EXTRACTION rather than by a pin.** The full 26-module run
reported exactly one UNPINNED survivor in the whole population, and it was the new code: an `and -> or` flip on
`l.strip() and not l.startswith("#")` inside `known_extra_state`, distinguished by no arm. The snapshot parse existed
**three times** in one file. In the two older copies the parsed list is RETURNED, so blank and comment lines leaking
in change the result and the Issue-790 fixture (which carries both a comment AND a blank line for exactly this
reason) kills the mutant; in the new copy the set was only tested with `n not in snap` against declared repo NAMES,
so the junk changed the set and no verdict — genuinely EQUIVALENT, and therefore pinnable with a straight face. That
is the trap. AGENTS.md records that about a third of rows once labelled EQUIVALENT were real gaps wearing the label,
and the repair for a duplicated rule is to stop duplicating it (Issue 755), not to adjudicate each copy on its own
merits. `read_snapshot()` is one site now, reached by all three callers and armed by the fixture that was already
there; the mutant population dropped 100 → 97 and the survivor set is back to the two pre-existing pinned rows.

Arms: 16/16 in `population_sync_gate --canary` (5 new + the documentation arm), 13 COUNTED in
`sweep_population.selftest` (its pass line hand-typed "7 assertion(s)" and would have gone stale on arrival —
Issue 798 T3's shape, so the count is derived now), and the `skill_repo_set_gate` arms including the one that
matters most: an UNNAMED extra repo still reds beside a named one, which is the arm a blanket `=1` marker would pass.

## Issue 808 (2026-09-17) — CLOSED: T1 landed option 2, the AVX2 `argtopk` dispatch narrowed to k ≤ 4 on x86_64; the measured loss is GONE and the wins are kept

Owner's call on the four options, taken on Bench 810's evidence: **option 2**, and scoped to **x86_64 only**. The
dispatch predicate in `argtopk_with_scratch` is arch-independent `k ≤ 16`, so the narrowing lives inside the x86_64
`argtopk_simd` against a new `AVX2_ARGTOPK_K_MAX = 4`; NEON keeps the full k ≤ 16 range, because it is the algorithm
witness the AVX2 port was transcribed from, it is not measured as a loss, and narrowing it would be a behaviour
change on aarch64 that no measurement asked for. Option 1 was refused on its own data: Bench 810 measured `N_MIN` as
DISTRIBUTION-sensitive (k=8: 256 on `locality`, 768 on `iid_uniform`, >1024 on `late_peak`) and the dispatch cannot
observe the distribution, so a per-k n-floor would be a gate on a quantity the gate cannot see.

**Measured after the change** (4090, i7-13700K, `+avx2`, release, interleaved median-of-ratios): the recorded loss is
gone — `late_peak` k=8, the deepest cell at **0.41–0.72×**, now reads **0.98–0.99×**, and the worst cell anywhere
above the bound is **0.95×** across six distributions × k ∈ {8,16} × n ∈ {64..512}. The wins below the bound are
untouched: k=2 reaches **5.11×** (`iid_uniform`, n=1024), k=4 **3.51×**, k=1 **6.73×** (`locality`, n=1024).

**T3's lesson applied to T1's own fix.** Narrowing a dispatch RECREATES the condition both Issue 806 defects grew in:
the two 806 regression tests reach the kernel through `argtopk_with_scratch`, so at k ∈ {8,12,16} they would have
silently started asserting the SCALAR path on x86_64 and the AVX2 arm would be executed by nothing again. The kernel
was therefore hoisted out of its enclosing fn into a named `argtopk_avx2_kernel` so a test can still reach it ABOVE
the dispatch bound, and `test_argtopk_avx2_kernel_matches_reference_above_dispatch_bound` calls it directly over both
806 fixtures (the ascending ramp, the 1..7-element partial tail) at k ∈ {4,5,8,12,16}. Both new tests were verified
NON-VACUOUS by perturbation (reversing the kernel's output order reds them, plus 7 pre-existing).

**T2 closed under option 2, in two halves that fail differently.** The load-immune half is
`test_avx2_argtopk_dispatch_bound_is_pinned` — a constant does not oscillate, and this repo has measured perf bars
producing four different failing sets across four runs of ONE commit (Bench 806 T7). The timing half is
`bench_simd_topk_issue808_t2_above_bound_is_not_a_loss`, floored at **0.85×** against an expected ~1.00 and a defect
that was 0.41×; it carries the confirm-alone discipline in its own failure message. T2's ≥2-microarchitecture bar is
**not** discharged and was never load-bearing for NARROWING — narrowing is the conservative direction, falling back
to code that runs everywhere. It still gates any WIDENING back toward option 1, and the reopen instrument is
`bench_simd_topk_issue808_crossover_nmin`, deliberately KEPT even though it now reads ~1.00 above k=4 on x86_64:
deleting it would delete the reopen path.

## Issue 806 (2026-09-17) — CLOSED: T8 was the last open task and it was entirely Issue 808's T1

T6 and T7 closed 2026-09-16 (Bench 806 Addenda I+II, quiet-box calibration); T8 was filed out to Issue 808 as the
owner's choice between four options. With 808's T1 landed and 808 closed, 806 owns nothing further. Standing record:
the nine-cell matrix, the five defects it found, and the method pins live in `.benchmarks/806_x86_64_execution_matrix.md`;
the matrix itself is `scripts/x86_64_execution_matrix.sh`, documented in AGENTS.md, and remains a workstation verdict
with no CI lane requested.

## Issue 818 CLOSED (2026-09-17) — the four no-default `--all-targets` breaks gated both halves; bench_412's green-zero row found in the same sweep

Fix `78a1ac5d7`. The repro (`cargo check -p katgpt-core --no-default-features --all-targets`) named four targets, all
fixed per the issue's own fix shape — whole-file `#![cfg]` + the matching `required-features` row, both halves (the
cfg protects the count, the row protects the reader; the Plan 599 fanout-row comment is the house spelling):
`bench_371` re-gained the row deleted at its Plan-371-Phase-6 promotion (the promotion removed the gate instead of
moving it; a harness=false bench emptied by its own `#![cfg]` has no `main` → E0601, so there the ROW is the
load-bearing half and the in-file cfg stays); `bench_416` gated `region_subspace_steering` (implies
`subspace_steering`, both imports); `bench_778` gated `subspace_intervention` (implies `subspace_phase_gate`; the
bench_779 row is the naming precedent — note it was ALSO broken at plain default `--all-targets`, the state nobody
runs, since `subspace_phase_gate` is opt-in); `velocity_field_ensemble_alloc_check` gated
`velocity_field_ensemble` (row placed beside its heterogeneous sibling in the alloc-check cluster). **The sweep's
fifth target**: `bench_412` had the cfg half but no row — the green-zero class (compiles empty, `ok. 0 passed`
under no-default), not a break, caught because bench_416's doc names it as the convention mirror; row added. All
four fixed targets verified NON-VACUOUS at their feature sets (3+1i / 4+1i / 2 / 1 passed — the counts fire, not
green zeros); no-default, default, and all-features `--all-targets` all check clean; default lib 2063/0
bit-unchanged. Standing corollary for promotions: DEMOTING/promoting a default-on feature must MOVE its target
gates in the same commit, never delete them (bench_371 is the measured instance).

## Issue 808 (2026-09-17) — T4 LANDED: `argtopk` AVX2 dispatch re-measured on six realistic block-score distributions + per-k `N_MIN` crossover (Bench 810)

Two units on the 4090 box, one evidence-gathering, one a same-day negative. **Bench 810** (Issue 808's option-4 half + T2's
Raptor-Lake row): the recorded "AVX2 `argtopk` k≤16 is a measured loss" table was measured on ONE i.i.d. fixture — the re-measure across six
realistic block-score distributions (iid / gauss-sigmoid / locality / early-peak / late-peak / bimodal-sparse) × both profile arms (default
release; `+avx2`) found the loss SURVIVES every distribution except `early_peak` and DEEPENS on `late_peak` (0.41–0.72× below n=512,
still 0.90–0.95× at n=1024 — **no n-floor rescues k=8 there**); k≤4 is the distribution-robust carve-out (win or tie past n=64–128, up to
3.9×); and `N_MIN` is distribution-sensitive (k=8: 256 on `locality`, 768 on `iid_uniform`, >1024 on `late_peak`) — so the eventual T2 gate
needs the distribution axis, not one n per k. Instrument: the shared Issue-723 interleaved median-of-ratios (`tests/common/ab_timing.rs`),
new tests `bench_simd_topk_issue808_{distribution_matrix,crossover_nmin}` in `tests/bench_256_simd_topk.rs`. T1 stays the owner's call;
T2's ≥2-microarchitecture bar is 1 of 2 met.

## Issue 817 (2026-09-17) — single-pass AVX2 `argmax` port measured NEGATIVE + reverted; the NEON premise does not transfer (Bench 812)

Filed and closed the same day, alongside the Issue 808 T4 unit above; the file was removed per the
noise-reduction rule, so this heading is the whole allocation record. **Bench 812**: the same matrix's k=1 rows exposed the
x86_64 `simd_argmax_f32` two-pass cost depending on WHERE the maximum sits (`position(== max)` rescans), and `argmax.rs` carried a standing
offer to port the NEON single-pass kernel to AVX2. The port was built (8-lane (max,index) tracking, blend-on-strict-gt, correctness green on
known-answer + tie + AVX2-tail + NaN-pin tests) and measured a NET LOSS on this box: `iid` 0.24–0.28× and `early` 0.20–0.23× at n ≥ 256 on
both profiles (the two-pass's ILP'd `max_ps` reduce + vectorized early-exiting `position` beat the latency-bound cmp/blend chain); it only
won `late` (1.5–1.9×) and n=64 (fixed overhead). Demote-on-loss applied: dispatch reverted same day, kernel deleted, two-pass doc comment
carries the numbers + reopen trigger, the `bench_817_argmax_dispatch_ab.rs` harness KEPT as the reopen instrument (now reads ~1.00 — both
arms the same code), and the new equivalence tests kept (cross-platform guards). The NEON single-pass premise does not transfer to x86_64.

## Issue 811 CLOSED (2026-09-17) — DBTM confidence-commit anchor rule: PoC PASS → Plan 600 landed end-to-end, promotion EXECUTED via Plan 601

The full arc, PoC to production: [`.research/563_DBTM_Discrete_Beckmann_One_Step_Language.md`](.research/563_DBTM_Discrete_Beckmann_One_Step_Language.md)
(`e0e63f47b`) distilled arXiv:2609.15903 §5.1's κ ∪ floor commit rule + §4.3's t\* law; the three-arm PoC
(`ee105eb78`) measured the DBTM rule converging 3.1–4.7× faster than the matched strided arm at quality
parity with termination-within-k property-proven at every (κ, k) cell; [Plan 600](.plans/600_flashar_confidence_commit_upgrade.md)
landed the production seam — `ConfidenceAnchorConfig` + `select_confidence_anchors` + `anchor_then_fill_with`
(`8bd6d7132`), docs (`261d80fd2`), T8+T9 GOAT gates green on the lane's non-saturated pattern corpus
(`df49ed4c5`, `5c6cb58e5`, Bench 600) — and [Plan 601](.plans/601_flashar_realtext_eval.md) grew the real-text
D2F eval the acceptance bar demanded and re-ran every gate green on char-level Austen (`a769a42ec`, Bench 601:
paired Δ +0.03…+0.11, 1.2–2.8× steps / 0.72–0.91× wall, realized-KL 0.48–0.87× incumbent) →
**promotion executed**: `ConfidenceAnchorConfig::default()` = κ 0.9 + floor is the anchor-then-fill decode
default; the strided entry stays as the no-floor comparator (demote-on-loss rule stands).
Both of the issue's deferrals were discharged on the way: the deferred T3 sweep-bench arm landed through the
Issue-813 custom-order seam + Bench 809 (`d392600d6`) and its NO-SEPARATION verdict was resolved same-day by
the Issue-816 masked-target trainer (`b3016516e`) — prob-t\* WINS the table on both eval seeds (2.59/2.61 vs
uniform 2.89/2.88 nats), the DBTM ordering hypothesis measured green on real text; the deferred T5 UGC
cross-check is Plan 600 T9, measured green (MC 8192×2 seeds: greedy reveal distorts LESS than the incumbent,
KL ratio 0.996 pattern / 0.48–0.87× real text; Caveat #1 empirically benign). The t\* half landed pure in
katgpt-core: `commit_time_star` (`ignition_schedule`) + `probability_order` (`set_diffusion_schedule`).
Follow-on work continues under Plans 600/601 (complete) and the sweep lane's Bench 809 addendum, not here.

## riir-ai Issue 964 C2 LANDED (2026-09-16) — `CalibratedActionBridge`: decision-level confidence calibration for the ABSTAIN threshold (Bench 808)

The second `sigmoid_calibration` consumer. `ActionBridge`'s `sigmoid_confidence` gates the ABSTAIN threshold — a threshold on an
uncalibrated score is a threshold on a number whose meaning is proven nowhere. `bridge::calibrated::CalibratedActionBridge<A, D>` (rides the
`sigmoid_calibration` feature, no new flag) wraps the bridge with the Platt calibrator: observe (raw confidence, action_succeeded) at outcome
time, refit off-hot-path, calibrated confidence in front of the threshold. **Argmax invariant**: selection stays on raw scores; one strictly
monotone transform on a shared score scale cannot reorder it — calibration never changes WHICH action wins, only what the confidence MEANS.
GOAT (Bench 808, planted overconfidence `sigmoid(1.4·logit − 0.3)`, 4096/4096 split): decision-level ECE 0.0220 → 0.0088 (2.5×); log-loss + Brier
beat raw AND the base-rate floor; cold start bit-identical (confidences + abstain decisions); 0 winner mismatches after real refit; the ABSTAIN
operating point at τ=0.75 moved 4.4× toward oracle (0.3315 → 0.2764 vs oracle 0.2866, the katgpt-rs G3 fire-rate shape); observe+select 0 allocs
(per-thread counting_allocator). Honest caveats in the bench doc: mild corpus-level drift (thin G2 margins — the ABSTAIN movement is the
load-bearing result), fitted-b recovery 0.043 wide (little low-confidence mass at A=4), and the riir-ai `arg_runtime` Step-8/9 wiring is the
unblocked follow-on. Bridge tests 21/21, lib 2060/0, isolation clean, clippy clean.

## riir-ai Issue 964 C1 LANDED (2026-09-16) — `clr_calibration`: the CLR verifier becomes the first `sigmoid_calibration` consumer (Bench 807)

The CLR verifier's verdicts were bounded (Bench 284 G2, ECE 0.0087 on a fixture whose ground
truth was the verifier's OWN sigmoid) — boundedness again, not calibration. `clr_calibration`
(katgpt-claim, opt-in, root forwarder) ships `CalibratedVerifier<V>`: wraps any `ClaimVerifier`
with the Issue-810 Platt calibrator — `observe(raw_verdict, outcome)` at the existing ECE harness
points, off-hot-path `refit`, `apply` in front of the reliability gate `r_k = (mean_m v)^M`.
GOAT (Bench 807, deterministic seeds): planted-drift fixture (world on `sigmoid(1.6·dot − 0.4)`,
verifier on `sigmoid(dot)`) ECE 0.0924 → **0.0164** (5.6×) with planted recovery (T, b) =
(0.642, 0.252) vs (0.625, 0.250); log-loss 0.4819 and Brier 0.1591 both beat the uncalibrated
verdict AND the base-rate floor (Report-the-Floor); the already-calibrated G2 fixture undisturbed
(ΔECE +0.0002, pinned as the ±0.005 bin-noise no-harm band — the honest reading of C1's "≤"
on a saturated fixture); vote winner stable 25/25 seeds under a near-identity refit; observe+apply
zero-alloc (dev + release `alloc_tracking`). Substrate companion fix: `apply` gained the identity
fast path — at `(w, c) = (1, 0)` the input returns bit-identically (a logit→sigmoid roundtrip is
NOT bit-exact in f32; the documented "identity until evidence" contract is exact-value) — cold
start changes nothing, pinned by two new substrate tests + the consumer G3a bit-identity gate.
Dev note recorded in the bench doc: the first draft's train/test splits used DIFFERENT random
direction vectors (calibrated predictions scored against another direction's labels — ECE 0.35
with correct recovered params); splits share one direction pool now. Module docs tightened
bounded→calibrated (`clr/traits.rs`, `clr/verifier.rs`); feature catalog §114; README counts
612→613. Clippy `-D warnings` clean (katgpt-claim `--all-targets --all-features`, katgpt-core
`sigmoid_calibration`); bench_284 G1/G2/G5 + G4 re-run green; docs gate 25/25.

## Issue 812 CLOSED (2026-09-16) — the bench_doc_audit BlindRead context split was a TMPDIR-FORM split; the arm now matches paths form-independently

The verdict-review second opinion measured a deterministic red/green split on a byte-identical
`bench_doc_audit.py` blob; the reproduction protocol (this box) found every interpreter green and
isolated the trigger to `TMPDIR=/tmp/...` — an UNRESOLVED-SYMLINK temp-root spelling. The arm's
predicate matched patched paths by `str(self).startswith(str(root))`, while `audit_repo`
canonicalizes (`repo_root.resolve()`) — under the unresolved TMPDIR the fixture root spells
`/tmp/...` and the audit's walked paths spell `/private/tmp/...`, so the OSError never fired and
BOTH unreadable arms reported `got False, want True` on a healthy instrument. Not an interpreter
difference (3.11–3.14 all green pre-fix under the default resolved TMPDIR) and not a patch-target
miss (`Path.open`/`Path.read_text` ARE the audit's read layer). Fix: `root_forms = (str(root),
str(root.resolve()))` with the pred accepting either spelling, premise documented at the arm —
`scripts/bench_doc_audit.py` `blindness_arms()`. Verified both directions: resolved-form green,
unresolved-form green post-fix; the pre-fix red under the same TMPDIR is the recorded canary;
docs gate 25/25. Population checked: this is the workspace's only `setattr(Path, ...)` monkeypatch
arm — nothing to generalize. Issue file removed per the noise-reduction rule.

## Issue 810 CLOSED (2026-09-16) — `sigmoid_calibration`: the Platt-style calibrated sigmoid gate lands as a PoC with all four gates green

Filed from Research 562 (Jev distill): every decision/confidence scalar in the stack is a
sigmoid output whose boundedness is Lean-proven but whose MEANING is proven nowhere. The PoC
lands `crates/katgpt-core/src/sigmoid_calibration.rs` behind opt-in `sigmoid_calibration`:
`SigmoidGateCalibrator` (observe → FIFO ring, zero-alloc · refit → deterministic 2-param
Newton on Platt-smoothed targets, off-hot-path · apply → one logit + fma + sigmoid, zero-alloc
· commitment → BLAKE3 over versioned canonical bytes incl. the monotone `n_obs_total`, the
`closure::commitment` convention), `CalibratedGateSet<const N>` for the 5-affect-scalar shape,
and the public metric substrate (`brier_score` / `log_loss` / `expected_calibration_error`).
Monotonicity guard `w = 1/T > 0` (anti-correlated windows project to `W_MIN`) — calibration
can never reorder decisions, so G3 is by construction. Measured (planted fixture
`p_true = sigmoid(1.6z−0.4)`, train/test split, n=2048 eval): **G1** ECE 0.0696 → 0.0322
(2.16×, ≤ 0.05) with planted-transform recovery T=0.627/b=0.243 (planted 0.625/0.25);
**G2 (Report the Floor)** log-loss 0.5923 < uncalibrated 0.6066 < base-rate floor 0.6878 and
Brier 0.2026 < 0.2090 < 0.2473 — both dumb baselines beaten on both metrics; **G3** default
lib suite 2060/0 unchanged + ranking/optimal-threshold/fire-rate-movement assertions;
**G4** observe+apply 1000-call loop zero-alloc (Issue-741 predicate — runs in dev AND
`--release --features alloc_tracking`). Docs: README/examples counts 611→612, feature catalog
§113. 11/11 module tests, clippy `-D warnings` both feature states, docs gate 25/25. Consumers
(CLR verifier, ActionBridge `sigmoid_confidence`, the 5 affect scalars) filed as riir-ai Issue
964 — opt-in per the no-default-consumer rule until a consumer GOAT lands. Issue file removed
per the noise-reduction rule.

## Issue 809 T1+T2 CLOSED (2026-09-16) — the global-RNG census is read; the class is gated; the T3 `Rng::new()` census is deferred with a reason

The 806 matrix found the defect; this landed the wall. The T1 census used the WIDE predicate
(every free-function `fastrand::<primitive>()` over tracked `*.rs`), not the issue's original
narrow list — **20 sites across 9 files**, several never enumerated before (cgsp's Uuid-v7
snapshot ids, the katgpt-types simd tests, the examples). Exactly ONE defect-shaped pair: the
issue's own prediction held — `bench_pflash_maxsim_block_scoring`'s fixture generation was the
latent class (every GOAT re-run measured a different synthetic corpus); seeded
`Rng::with_seed(809)`, needle margins realization-independent so no verdict moved. Everything
else is deliberate with the reason on its row: the SDAR soft gate is stochastic BY DESIGN (the
pure fn takes the draw as an argument); the cgsp sites are identity entropy (the sanctioned
`Uuid::now_v7()` semantics); the rest are statistical tests (≥6.3σ / ~18σ margins) and
demo/example players. T2: `scripts/global_rng_gate.py` — the docs gate's 25th check. Membership
pin, LINE-FREE `path::call::count` keys, both directions red, floors on walk + predicate,
comment/string masking, unconditional arms (check_validation 25/25 at landing). The canary run
found its own population hole before landing: a planted UNTRACKED file is invisible (`git
ls-files` IS the population) — correct for a per-push CHECK, and the canary was redone staged.
arm_reach: 22 killed / 2 EQUIVALENT (in-string EOL-backslash flips — the multi-line-string
class the masker documents out of scope; pinned at the arms header). T3 LANDED same day at
`6d120abeb` — the `Rng::new()`/`Rng::default()` constructor class joined the gate (147 global
sites across 55 files, every row pinned); issue file removed per the noise-reduction rule.

## Issue 806 T6 CLOSED (2026-09-16, the M3 side) — t698_t5_kv_mean adjudicated arch-dependent with dual pins; kda grad-check floor was below its own noise

[Bench 806 addendum](.benchmarks/806_x86_64_execution_matrix.md), landed from the M3 the same
day as the 4090 matrix. The one M3 run T6 asked for: the test PASSES on aarch64 — the pin
`23d0daab3f087159` reproduces — so T6's first branch holds and the fixture is genuinely
arch-dependent (x86_64 measures `4d0b592740db9358`; band bits one ulp apart, `0x3e5f_d968`
vs `0x3e5f_d970`; argmax 0/12 and every behavior gate green on both). Arch-conditional dual
pins with the recorded delta; the stale `t698_t5_kv_mean_gates` row removed from
`x86_64_matrix_expected.txt` in the same commit per the file's ratchet. Plus, not in any
upstream cell: `kda_backward_grad_check`'s magnitude floor × tolerance (1.25e-5) was below
its own measured FD noise (2.8e-5, matching the `EPSILON·|loss|/ε` bound) — M3-tuned test
constants, not a derivation defect (the L=1 check and the exact token-vs-sequence identity
pass on x86_64); floor 5e-4 → 2e-3, 7/7 both platforms. The local session that landed this
had independently found the argtopk sentinel bug and was fixed for it the honest way: its
block_topk hunk was DROPPED in favor of upstream `086dd9127` (which additionally removed the
remaining4 OOB read) when the collision surfaced at cherry-pick.

## Issue 807 (2026-09-16) CLOSED — `lthash`: incremental homomorphic multiset hash, the shared substrate for the Agave-mined commitment/state-hash proposals

Mined from the Agave validator snapshot (`riir-clippy/.raw/agave @ c95d8706`, the eprint
2019/227 / Facebook LtHash instantiation Agave ships for account-state commitment) via four
read-only subagents for riir-chain Proposal 010 D1 (commitment_root 31.6 ms @ N=1e5 → O(1),
the fix `dispatch.rs` + Bench 028 #16 already specify) + riir-dapps Proposal 005 D1
(`kat:statehash` per-batch tamper-evidence). Substrate-first grep: zero existing
multiset-hash; katgpt-core is the only home both private consumers reach without a cross-repo
dep. Ships `crates/katgpt-core/src/lthash.rs` behind OPT-IN `lthash` (const-generic lanes,
default 1024, wrapping add mod 2¹⁶; insert=add/remove=subtract/merge=sum/checksum=BLAKE3;
domain-separated BLAKE3-XOF element derivation over a LENGTH-PREFIXED part list — no
concatenation ambiguity). GOAT at [Bench 771](.benchmarks/771_lthash_goat.md): G1 10/10
(order-invariance, drift incremental==rebuild, merge≡incremental, multiset semantics,
encoding unambiguity, domain separation, N=128 arm, hex-pinned KAT); G2 incremental
`replace` 31.6 ns vs 1000-member rebuild 33.8 µs ≈ 1069× (derive-inclusive ~1.4 µs vs
~1.39 ms); G4 zero-alloc by construction (fixed arrays, ns-scale ops). Stable-const-generic
lesson: `2 * N` array sizing needs `generic_const_exprs` — the derive path fills through a
fixed 1 KB XOF chunk instead (any N, still zero-alloc); associated-fn calls like
`LtHash::identity()` do NOT take the const default — every construction site needs a pinned
type. Promotion to default waits on the first consumer (no-default-consumer rule). README /
examples/README / feature-catalog counts bumped 610→611.

## First x86_64 execution of the katgpt-core/katgpt-types SIMD suites (2026-09-16) — 15 latent AVX2 bugs caught and fixed; Bench 800's execution-parity caveat resolved NEGATIVE then closed

[Bench 800 addendum](.benchmarks/800_bf16_simd_goat.md). The run: 4090 box (i7-13700K), repo
@ `d1f9be27a`, `git archive`-extracted to a scratch dir (the checkout there carries sibling WIP —
never touched). katgpt-types `--lib` default build: **139/139** — the first x86_64 execution of
the crate's suite, including the `simd_exp_sum_extreme_inputs_underflow_not_wrap` regression
(riir-train Issue 549) that the AGENTS.md execute-axis table recorded as "executed by NOTHING".
katgpt-core `--lib --features bf16_simd` at `RUSTFLAGS="-C target-feature=+avx2"` (bf16_convert's
AVX2 arm is compile-time gated on `target_feature` — without the flag the run silently exercises
the scalar fallback and proves nothing): **15 FAILED**, all AVX2-arm defects never executed by
any lane (NEON runs on the M3; no automatic x86_64-native lane exists — the AGENTS.md platform-
axis warning, measured): (1) `f32_to_bf16_rne_avx2` dropped the NEON identity's final `t >> 16`
and `_mm_packus_epi32` saturated instead of truncating — every narrow packed mask garbage;
(2) `dequant_via_lut_avx2` + `dequant_dot_via_lut_avx2` used the SSE4.1 4-element
`_mm_cvtepu8_epi32` where the 256-bit gather needs `_mm256_cvtepu8_epi32` — lanes 4–7 of every
eight gathered `lut[0]`; all 12 simd_lut_dequant failures cascade from those two sites. The
multi-stage kernel knew the trap (its comment names it) — the single-stage kernels missed it.
Fixed same-session, cross-checked from the M3
via `cargo check --target x86_64-unknown-linux-musl` +avx2 (no linking needed), rerun green on
the 4090: **2069 passed / 0 failed** (was 2054/15). Two standing lessons: an execution-parity
caveat that "rests on the shared-algorithm argument" is a conjecture until the platform runs; a
compile-time `target_feature` gate means even a plain x86_64 `cargo test` exercises nothing —
record the flags with every such run.

## Issue 805 (2026-09-16) CLOSED — `numbering_gate.py --help` printed ten "remove the row" lines about a repo it could not read

`--help` was read as a repo PATH, and `tracked_paths` converts a git failure into an EMPTY
set — so every pin read STALE and the gate printed ten `remove the row` remedies plus
`re-pin DOWN`, with the floor breach LAST. Exit was always 1, so nothing was silently
wrong; the ORDER and the REMEDIES were, and the top-of-output remedy is destructive
(Issue 795: a pin row is the only record left once both holders are removed). Repair:
`unmeasurable(repo)` → exit **2**, and the check is toplevel EQUALITY, not "did
`rev-parse` succeed" — `git -C` walks UP, so a non-repo directory inside a repo answers
with its PARENT's paths. ⛔ The first Scope claim was WRONG and the census caught it:
handed a nonexistent path, 14 repo-path instruments misbehave, and one of them —
`orphaned_attr_gate.py`, a docs_gate CHECK — printed `✓ PASSED … in nonexistent-repo` at
**exit 0**. Fixed with a DIFFERENT predicate, because it goes through `tracked_files`,
which deliberately falls back to a filesystem walk with no `.git` (a `git archive` tree is
a legitimate population): not-a-directory → UNSEEN, a sibling walking to 0 `.rs` → UNSEEN
rather than FAILED, since a TS-only repo walks to 0 legitimately. `unmeasured()` was
EXTRACTED because `main` calls `selftest` and an inline decision cannot be armed without
recursing — the same extraction-is-the-repair pattern AGENTS.md records for the three
weakest CHECKS modules. `arm_reach_gate` then found **4 survivors hand perturbation
missed** (`and2or`, `dropnot`, `eq2ne`); final 25 modules / 691 mutants / 465 killed / 38
pinned survivors / 0 UNREACHED / 0 NO-ARM / 0 BASELINE. Ten report-only instruments left
alone deliberately. Landed `d1f9be27`. Issue file removed per the noise-reduction rule.

## Issue 804 (2026-09-16) CLOSED — 28 instruments crashed when run the way AGENTS.md says to run them; the cross-repo axis is seven repos, not one

Every instrument in `scripts/` prints `✓`/`✗`/`⛔`/`⚠`. On a non-UTF-8 console — this
workstation is **cp874** — `print()` raises `UnicodeEncodeError` and the process dies with
NO verdict. `docs_gate.sh` exports `PYTHONIOENCODING=utf-8`, so a per-push CHECK survives
*when the gate runs it*; nothing protects an instrument run DIRECTLY, which is how AGENTS.md
documents most of them. So the class concentrated in exactly the instruments with no
automatic lane. ⛔ The cost is not a crash, it is UNREAD FINDINGS:
`restatement_drift_sweep.py` was the one member of the sweep family without the defence, so
its 4 repos / 255 theorems were not *unknown* but *unlooked at* while the family was
reported green. Repair: `scripts/console_safe.py` (`errors="backslashreplace"` — the
console encoding is not ours to choose, and forcing UTF-8 gives mojibake instead of an
exception), the 28 call it, the 42 that already inlined it are left alone (the gate credits
both forms, because the property is *the streams are defended*), and
`console_encoding_gate.py` joined the docs-gate CHECKS.

The cross-repo half was written down as "⚠ unmeasured, deliberately" on the
`check_validation_gate` Issue 789 T4 precedent — which does not say *decline a sweep when
the exposure is local*, it says **re-measure the population before answering**, and 789
earned its "no sweep" with a measurement that returned ONE. Taken (`bc98dc6b`): 16 repos,
159 tracked `scripts/*.py`, 144 in population, 73 defended, **71 undefended across SEVEN
repos** — riir-train 53/53, riir-clippy 5/5, riir-ai 4/5, mmorpg-remake 4/4, mmorpg-editor
3/3, riir-dapps 1/1, riir-mmorpg-examples 1/1, against katgpt-rs's 0 of 72. ⚠ The exposure
caveat SURVIVES the measurement and is what sets the pin design rather than what cancels
the sweep: a cp874 console is this box's property, and riir-train's 53 rows are the same
plan-scoped over-capture `instrument_reachability` measures on this identical walk. So
`console_encoding_drift_sweep.py` ratchets the DERIVATIVE (Issue 787 T6's answer to the
identical shape), with katgpt-rs's row asserted at `max_undefended = 0` and
`min_population == console_encoding_gate.MIN_POPULATION`. Issue file removed per the
noise-reduction rule.

## Issue 803 (2026-09-16) CLOSED — the off-macOS partial gate printed the SAME final line as a full pass

Two halves. `develop` was RED under the exact command AGENTS.md quotes as the whole-repo
claim — 24 × `error[E0560]` (a field deleted in `katgpt-core` with 24 live construction
sites in the ROOT package's `tests/` and `benches/`) plus 4 `-D`-listed lint errors — for
ten hours, while `test_gate.sh` (every row at its floor) and `wasm32_gate` were both green.
The per-crate gate that landed it was right for what it changed; nothing read what it
changed for everyone else. And the instrument that would have —
`full_gate.sh --allow-partial-platform`, the ONLY thing on a non-macOS workstation that
reads the consequences of a per-crate change — existed the whole time and printed a final
line byte-identical to a full pass, so the deferral rode a Layer-2 line six hundred lines of
build output earlier. That is this repo's own most-repeated rule broken by the one
instrument that is not a sweep. Repaired: the final line now carries the partial verdict and
names each unmeasured axis, and the lane is documented in AGENTS.md next to `test_gate.sh`.
⛑ Renumbered 799 → 803 before the push by the Issue-796 rule — `dual_allocation_gate.py`
reported INDEPENDENT, adjudicated 5 inbound vs 1 by Issue 724 T2; the gate is the reason it
was caught before the push rather than at merge time. Landed `3ceb541b` (T1+T2) and the
Landed `3ceb541b` (T1+T2) and the
follow-up (T3+T4). Issue file removed per the noise-reduction rule.

## Issue 802 (2026-09-16) CLOSED — commitment-gap calibration rig: residue DEAD-BY-DOMINATION at micro scale; stability features (item 3) shipped earlier in the day

[Bench 802 calibration-rig record](.benchmarks/802_commitment_gap_calibration_rig.md), rig at
`../riir-ai/crates/riir-poc/benches/commitment_gap_calibration.rs`. Items 2/3/5 shipped earlier
2026-09-16 (`f23b2d81b`, `226be38d9`). Item 1's defend-wrong verdict, executed per the issue's own
rule: G1 (TSD-gap separates would-miss states) is REGIME-DEPENDENT — 31–82× when the decode is
cascade-dominated, 2.4× when commits collapse into 1–2 waves and 38% of early states revise; G2
PASS — per-(class,t) calibrated thresholds beat a tuned global rule exactly where the precision
constraint binds (0.2587 @ 1.76 fwd vs 0.2538 @ 2.47); G3 FAIL — the no-gate one-forward baseline
dominates the entire joint-gate family (1.00 fwd @ 0.2937; +3.5 acc at half the NFE). Mechanism,
measured: self-consistency labels are blind to context contamination; single-pass calibration
misses its own precision target off-policy (p* 0.90 → 0.61 realized). Item 4's horizon axis
measured UNDEFINED at micro (reference cascade ≤ 2 waves — no per-position structure to reallocate
verify budget against; the rig prints the per-position table, so the measurement rides any future
scale-up run for free). Revival condition (all required): large would-miss mass + revisions that
improve rather than corrupt context + a genuinely-iterating baseline + a DAgger-style on-policy
calibration round. Also landed: root `katgpt_rs::speculative` re-export surface completed with
`StabilityTracker`/`TOPK_DRIFT_K`/`N_STABILITY_FEATURES` (item 3 shipped them in `katgpt-forward`;
the root shim missed them). Issue file removed per the noise-reduction rule; git history + this
row + the bench record are the trail.

## Issue 800 Arm C phase 1 (2026-09-16) — GraphStablePool<T> extracted: the common contract verified across 4 sites (a 4th found in-repo)

`877e06eb2` + [Bench 800-C](.benchmarks/800_graphstablepool_phase1.md). The DRY-extraction
arm verified its premise before extracting — and improved it: the never-move pool
contract ships not three but FOUR times (the issue's radix_prefix / PagedKVCache /
riir-gpu arenas + `BranchBank` in katgpt-core's own `branching/bank.rs`, found by the
substrate-first read). The common contract EXISTS in its narrowest form — index-stable
slots, LIFO free list, append-only growth — and the type documents the load-bearing
distinction loudly: INDEX stability is the pool's contract; payload-ADDRESS stability
belongs to the stored type (heap indirection, PagedKVCache's recipe) or to
pre-allocation discipline (Qwen38LaneSet bakes device pointers via allocate-once +
scoping, NOT chunking — which killed the chunked-layout design option). 6 contract
tests (bench_762's g1_address_stability generalized + LIFO/stability/noop-disclosure/
address-stability/G4-zero-alloc-churn), 2092/0 four-feature combo, wasm32 clean.
Re-points remain open as incremental follow-ups (one repo per commit: katgpt-kv →
katgpt-transformer → riir-gpu + optional BranchBank), each with its per-site adapter
documented in the module doc.

## Issue 801 (2026-09-16) CLOSED — T4 PoC: composition-quality REFUTED, disagreement-trace CONFIRMED; T5 routing: meld stays opt-in as a contradiction detector, Super-GOAT Q3 blocked-as-refuted, T2 audit stands

The arc, one session: T1 PDF-transcribed (2026-09-15); T2 audit `ee6993a77` (Research 560
addendum — the census survives code-level scrutiny: 4 INADMISSIBLE / 1 PARTIAL / 1 N.A.);
T3 primitive `43f15f7c8`+`f314d5006` (algebra fully validated — commutativity EXACT
bitwise, non-associativity 512/512, λ⋆ ≤3.24e-7 from brute force; Bench 801 red flag:
paper's depth-ladder shape did not transfer); **T4 PoC `286687cd0` (riir-ai Bench 932):
REFUTED** — mean matches/beats every meld arm on the adjudicating same-multiset D_eff
readout (all margins ≤ +0.008 vs the +0.05 bar; tanh ≥ law-8 never restored across the
σ ∈ [0.25,2] × β ∈ [0.1,10] sweep; best meld-family D_eff separation ≈ chance 0.5
everywhere), with the mean's own theorem-signature CONFIRMED empirically (blind within
multiset 0.499–0.502, sees across multisets 0.590→0.625) and Bench 801's tanh-operating-
point mechanism confirmed for absolute quality (σ=0.25 nearly doubles tanh's d=5
recovery — but the ordering never crosses). **The one positive extraction: the λ̃
disagreement trace — r(mean λ̃, ‖u−v‖) = −0.791, contradiction AUC 0.976, per-coordinate
localization precision@8 = 0.997, mean provably trace-free** — the one axis the T2 audit
found absent from every shipped composer (axis iv), now shipped in `katgpt_core::meld`
(opt-in feature `meld`). T5 routing per the issue's refuted branch: keep the primitive,
record the negative (Research 560 addendum §PoC Verdict + status DOWNGRADED), keep the
audit as the standing finding; fusion rows F1/F2 lose the composition-for-structure
premise (their disagreement-trace half survives via λ̃); F6's provenance hook (riir-neuron-db Issue 618
drift ledger) unaffected. Revival path on record (not owed): amplitude-vector 2AFC to
localize where bracketing info lives + the stochastic-noise readout for the clean
exp(S₂) identification. Issue file removed per the noise-reduction rule; this row +
Research 560 are the record.

## Issue 800 Arms A+B closed (2026-09-16) — bf16 SIMD G2 FAIL-honest (autovec parity); slot-flip DECLINE (ties mpsc); JSD kernel lands (Issue 802 item 2)

Three more measured verdicts in one batch, all honest closes:
- **Arm A** (`f314d5006`, [Bench 800](.benchmarks/800_bf16_simd_goat.md)): the pufferlib
  kernel-shape premise does not exist on M3+rustc — LLVM auto-vectorizes the scalar
  reference into the same NEON code (1.00× widen/trunc, 1.19–1.23× RNE vs the ≥4× gate).
  G1/G4 pass (RNE bit-exact vs `half` incl. NaN class; zero-alloc); stays opt-in, A4
  consumer wiring deferred, AVX2 execution parity compile-verified only (4090 run
  optional).
- **Arm B** ([Bench 800-B](.benchmarks/800_slotflip_arm_b_decline.md)): the Cleanba
  slot-ownership mechanic beats serial +21–46% but TIES mpsc (+9.1/+0.5/+1.7/+0.0%,
  sign flips across runs) → B2 DECLINE, `async_qdq.rs` stays single-threaded; no
  consumer lives in the short-compute regime where the mechanism could matter. The
  normative lock-free protocol + its staleness proof land as
  `crates/katgpt-kv/tests/slot_flip_staleness.rs` (4/4 ×5 runs) — the reusable artifact
  if a future consumer ever needs real cross-thread overlap.
- **Issue 802 item 2** ([Bench 802](.benchmarks/802_jsd_topk_kernel.md)): the NaN-safe
  bounded top-K JSD kernel lands in `katgpt_core::jsd_topk` (opt-in) — disjoint
  supports → BITWISE ln 2, identical → bitwise 0.0, NaN-safe by construction (the
  KL-trap that poisons gates is structurally unreachable), zero-alloc INTO form,
  scale-invariant, 2069/0. The task-sketch sign error in the JSD expansion was caught
  and corrected (shipped term ≥ 0 by convexity). Prerequisite for 802's calibration
  tables (1) + graded tri_mode verdicts (4) — both still open.

## Issue 801 T1–T3 (2026-09-16) — NAP audit + `meld` primitive: the census survives code-level scrutiny; the algebra holds, the quality ladder did NOT transfer (T4 is the sole adjudicator)

Three landings, one session: Research 560 addendum `ee6993a77` (T2 audit — all 6
shipped composers scored against the 5 NAP invariants + the paper's three theorem
screens; 4 INADMISSIBLE / 1 PARTIAL (tpr — structure via role bookkeeping, not the
law) / 1 N.A. (wedge `retrieve_diverse` — exposes, never fuses); two census location
corrections (`frozen_attractor` and `retrieve_diverse` ship in riir-neuron-db) and
one upgrade (the gauge composition LAW lives upstream here,
`katgpt-sparse/src/sparse_task_vector.rs:318-396` — riir-engine's
`GaugeInvariantComposer` is a thin bridge). `meld` primitive `43f15f7c8` + wiring
`f314d5006` (T3, opt-in feature `meld`): closed-form soft-min λ⋆ (quadratic, disc =
4(4−t²), stable root, candidate enumeration) with BIT-EXACT commutativity via
canonical (hi,lo) daughter ordering; normalized-Hadamard W; tanh/DN/law-8 arms;
self-tests: commutativity 4800 cases bitwise, non-associativity 512/512, λ⋆ ≤3.24e-7
from brute force, boundedness depth-64, disagreement-coding limits. **Bench 801's
honest red flag: the paper's bracketing-recovery ladder did not reproduce on our D=32
fixture** — tanh-meld 0.380 at d=5 vs law-8 0.870 (paper: meld ≥ law-8 everywhere);
mean-pooling does NOT collapse on full-vector nearest-centroid for ordered leaves
(per-leaf depth weights are tree-dependent — the blindness theorem is about the
S₂/D_eff readout, which is exactly T4(b)'s mandated PR discipline); no-W collapses
HARDER than the paper. T4's spec in the issue is sharpened accordingly (tanh
operating-point sweep first; statistics readout mandatory; law-8 kept in the
comparison). Super-GOAT Q3 stays blocked pending T4 — the §3.6 discipline (never
claim quality from architecture) is what caught this at bench time instead of at
consumer time. Deviation on record: DN form is `y/√(ε+Σy²)` (hard |out|≤1), not the
`/D` variant the task sketched — that form is bounded by √D and cannot meet the
depth-64 gate.

## Issue 800 Arm A (2026-09-16) — bf16⇄f32 SIMD kernels: G2 FAIL-honest — the pufferlib kernel-shape premise does not exist on M3+rustc

`f314d5006` + [Bench 800](.benchmarks/800_bf16_simd_goat.md). The lead arm of the
pufferlib distill shipped and was measured: NEON/AVX2/scalar bf16⇄f32 batch kernels,
RNE bit-exact vs `half` INCLUDING the NaN class (the branch-free add-form maps
sNaN-payload-1 to +Inf — special-cased; half's qNaN-forcing `|0x0040` convention
matched), truncation as the documented-bias opt-in arm, `into_buf` zero-alloc APIs,
G1 exhaustive + 2²⁰-pattern oracle + NEON-vs-scalar parity executing on the M3.
**G2 verdict: FAIL at 1.00× widen/trunc, 1.19–1.23× RNE vs the ≥4× gate.** Root
cause with evidence (identical medians): LLVM auto-vectorizes the scalar reference
loops into the same NEON code — rustc on aarch64 already delivers the pufferlib
kernel shape to any plain scalar loop of this class, so hand intrinsics buy ~nothing
on this platform. Consequences per the feature-flag discipline: `bf16_simd` stays
opt-in, NOT promoted; **A4 consumer wiring deferred** (the dequantize_row consumer
path already receives auto-vectorized codegen; no measured win to buy). Kept
anyway: G1-passing correct kernels + the ISA-guarantee argument (explicit intrinsics
are exempt from optimizer heuristics — the RNE+NaN arm is the one place the
auto-vectorizer measurably trails, 1.19×). AVX2 arm compile-verified only
(no x86_64 hardware this session); closing that caveat needs one run on the 4090
box. Arms B (slot-flip vs mpsc) and C (GraphStablePool DRY) remain open.

## Issue 799 (2026-09-16) — bevy_ecs 0.15→0.19 bump landed: the arenas are load-bearing evidence infrastructure, bevy_ecs is a schedule-free utility layer

The domain-test question the issue posed: is the bomber/monopoly bevy_ecs
usage load-bearing for a modelless-inference repo, or would a lighter
substrate serve? Measured answer: the ARENA is load-bearing (kernel_blend /
binned_blend GOAT evidence — Bench 432's mean delta +78.5, CI [+26.3,
+130.8] — was measured on bomber tournaments; retiring it orphans the
feature story and breaks benchmark comparability), while bevy_ecs itself is
NOT load-bearing in any deep sense — the entire exercised surface is
`World` + `query{,_filtered}` + `Messages` drain + derives, zero
scheduler/Commands/change-detection (≈23.5K LOC of arena code would need
rewriting and re-validation for zero mandate gain). DECISION: bump 0.15→0.19.1,
aligning with the workspace 0.19 wave. Landed at `e0f02ab93`.

Migration surface (the whole 0.15→0.19 delta on the exercised API, three
mechanical renames + one fallibility fix): `Events<E>`→`Messages<E>`,
`#[derive(Event)]`→`#[derive(Message)]` (buffered),
`World::send_event`→`write_message`, `Entity::from_raw(u32)`→
`from_raw_u32(u32).unwrap()`. Transitive tree: uuid 1.12→1.26 pulls
**getrandom 0.4** — the third wasm backend pin (root Cargo.toml
`getrandom_04`), plus `rng-getrandom` on katgpt-spectral's uuid so `v7`
no longer pulls bare getrandom.

Verified: wasm32 lanes green (`--features bomber` 1m17s, `--features
bomber-wasm` 24.6s; `secure_vessel` no longer exists here — the stale
mention in the getrandom comment fixed in the same commit); clippy clean at
both arena features (one pre-existing `unused_mut` under narrow features
silenced); tests 353 (bomber lib) / 203 (monopoly lib) / 289 (pruners
monopoly) / 10+1+1+5 (bench files) all pass. BOUNDARY.md §May depend on now
names bevy_ecs explicitly (optional-only condition + the domain-test
verdict) — the dep contract gap the issue was filed under.

GOAT gate: **Bench 799** (`.benchmarks/799_bevy_ecs_019_bump_arena_goat.md`)
— G1 game-semantics byte-identity PASS (every outcome-asserting diagnostic
identical before/after: kill attribution 96.4%, ScoreBoards, bomb counts,
late-game rates); G2 PASS with a DOCUMENTED cost: full-game harness ~2×
slower on bevy_ecs 0.19 World paths (interleaved A/B median 366→728 µs,
corroborated by an independent re-run 410/414 vs 749/777), pure-compute
cells unchanged or faster. Every assertion floor still clears with ≥10×
headroom; no recorded verdict flips; the cost is re-measurable by re-running
the A/B.

## Issue 798 (2026-09-15) — a tracked landing record claimed a sibling-repo repair that was never committed

Two tracked files in `scripts/` recorded cross-repo repairs as landed and
green on 2026-09-15 (`cb03c8dc`). Neither existed in the sibling repo, and
both sweeps were RED for the whole interval.

`toolchain_override_drift_floors.txt` read *"RESOLVED the same day: all five
markers are in"* and quoted `0 DRIFT / 3 DELIBERATE / 0 UNRESOLVED / 1
UNRESOLVED-MARKED`. Measured by `git grep toolchain-override-deliberate` over
the four named repos: **2 of the 5 existed**, both the katgpt-rs pair; the
sweep read `drift 1 · unresolved 1 · ✗ FAILED`.
`pipefail_discard_expected.txt` had **dropped a row** because riir-chain
`teardown.sh:24` *"got the same tail in the riir-chain commit"*; the tail did
not exist and the row read `✗ UNPINNED`. Dropping a pin for an absent fix is
the unrecoverable direction — nothing then points at the site.

The shape is unambiguous: the sibling files were edited in the worktree, the
sweeps run green against those uncommitted edits, the records written from
that run, katgpt-rs committed, the sibling edits never committed anywhere —
not in any sibling's worktree, index or stash today. That is **Issue 797's
class reached by a PROSE record rather than by a floor**, and 797's worktree
advisory (`41ecdcbd`) postdates the write-up by hours; it names exactly those
three repos on the repair run, making this the first real-world validation of
797 against an incident 797 did not know about.

Repaired at the four sibling commits, each now cited by SHA in the record it
belongs to: riir-ai `194cdc9b5` + riir-game-sdk `61f11e7` (deliberate markers
— both scripts BAKE their own `FROM rust:1.95.0-bookworm` image, so the
override tracks the container, not the workspace pin), riir-chain `5f814a2`
(the workflow marker + the `|| true` tail, which restored an unreachable
`already gone` branch). Both sweeps PASS.

**No new instrument.** Do not gate on prose landing claims: the sweep IS the
verification and it was red from the moment the record was written. The rule
adopted instead — *a cross-repo repair is not landed until it is COMMITTED in
the sibling repo, and a record here claiming one must cite the sibling
commit* — is in AGENTS.md § Before committing in a shared worktree.

⚠ **The repairs were landed TWICE, concurrently.** A second session landed
equivalent repairs upstream at the same time; both were discovered only at
`git push`, rejected non-fast-forward in all three repos. Theirs were already
on the remote, so this session's duplicates were dropped in favour of them —
`reset --hard origin/develop` in the two clean repos, and in riir-ai a
`reset --mixed HEAD~1` + single-file `checkout`, leaving six dirty files and
six unpushed commits belonging to the other session untouched.

⛔ **This session then committed the very defect it had just filed.**
`b592a213` cited three SHAs it had created locally, and those commits were
dropped minutes later — a record citing three hashes resolvable in no remote.
A SHA is verifiable only if it is PUSHED, so the rule is *cite the sibling
commit AND check that it resolves*; all three are now verified with
`git -C ../<repo> cat-file -e`.

**T2 — a sweep reads the WORKTREE, and a worktree can be behind ORIGIN.**
Dropping the duplicates surfaced a class Issue 797 cannot see: the toolchain
sweep went RED on riir-ai again, not because the fix was missing but because
this box's checkout is **109 commits behind origin**, 14 of them touching the
sweep's population. 797's advisory compares the worktree to LOCAL HEAD, so a
checkout matching its own HEAD while 109 commits stale is invisible to it —
the **mirror of MASKED** (a committed defect read clean) and therefore a false
RED, the cries-wolf outcome this family refuses to pay for. `behind_origin()`
in `worktree_state.py`, wired into `sweep_advisory()` so all 18 sweeps get it
with zero call-site changes; four answers never pooled (None / (0,0) / (n,0) /
(n,k>0)); the three-dot `HEAD...ref` diff keeps a merely-AHEAD repo from
reading as stale; the sweep deliberately STAYS RED, because a deferral on
staleness would let a genuinely-unfixed drift hide behind it. `dirty_in_scope`
and `behind_origin` share one `_match_count`: a git pathspec was the obvious
implementation for the second and answers differently for a bare `Dockerfile`,
so the two axes would have disagreed about what a sweep's population is.

**T3 — the selftest's hand-typed assertion count was already wrong.** The line
read `36 assertion(s)`; counted by AST over its own `*_arms` functions at the
PARENT commit, before any change, it was **40**. Stale on arrival, in the
module whose whole subject is records drifting away from what they describe.
`n_assertions()` derives it now (50 today), counted over `*_arms` only so a
`check` in production code cannot inflate it, returning 0 rather than raising
because a selftest that PASSED must not be crashed by its own summary line.

⚠ Non-finding, recorded so it is not re-investigated: the pipefail sweep's
PASS line says *"every pinned row firing"* beside `50 FINDING · 51 pinned
row(s)`. The both-directions check is real
(`pipefail_discard_drift_sweep.py:425-431`) and correctly skips repos absent
from the box. `riir-deployer` holds exactly one row and is one of the four
absent repos: 51 − 1 deferred = 50 checked = 50 findings.

## riir-train Issue 549 fixed in `490b662e` (2026-09-15, M3 + 4090 session) — avx2_exp_sum_inplace: the one exp kernel missing the n-clamp

The fused exp+sum SIMD kernel behind every `softmax` call was the ONLY exp
kernel without the n-clamp to [−126, 127] before the `(n+127) << 23` 2^n
bit-trick — `avx2_exp_inplace`, every NEON/wasm variant, and
`cephes_exp_scalar` all carry it. Below −87.3 nats the unclamped shift
WRAPS the f32 exponent field: **exp(−300) = 6.9e23** (4090 negative
control); above +88.7 it wraps negative-tiny. Any softmax whose input
spread exceeds ~87 nats then garbles weights and NaNs the loss — AVX2
machines only (aarch64 NEON clamps; wasm clamps; scalar guards).

Surfaced as riir-train Issue 511's census finding "Windows seed-1000
full-training collapse": seed 1000 game 0's attention score spread crossed
the boundary mid-training, loss → NaN, Δfull read 0.0000 through the old
`.max(1e-9)` clamp, and the retention gate compared two garbage ratios.
Post-fix on the 4090: game 0 Δfull 6.7941 vs M3's 6.7940 — the platforms
agree to the third decimal once the wrap is gone (the residual NEON/AVX2
rounding difference is benign).

Why nothing caught it: the truth-referenced sweeps deliberately "stay
clear of the n-clamp boundary (|x| > ~88)" — the boundary was known, and
the fused kernel's out-of-range behavior was never asserted. The new
regression test `simd_exp_sum_extreme_inputs_underflow_not_wrap` pins the
contract across all three code paths (32-wide main loop, 8-wide remainder,
scalar tail), asserting underflow-to-~0 on the low side, positive
saturation on the high side (inf from the scalar early-exit OR finite
~2^127 from clamped vector lanes), fused/unfused parity, and the softmax
denominator invariant. Verified on both platforms: 105/105 aarch64,
105/105 AVX2 (4090), plus the pre-fix RED on real silicon as the negative
control. Perf: the clamp is two integer ALU ops per 8-lane step, hidden
under the polynomial's FP dependency chain.

Blast radius: `katgpt-types::math::softmax`/`softmax_scaled` and every
direct `simd_exp_sum_inplace` consumer on AVX2 — rare at inference
logit spreads, routine during early training and any diverged-activation
state. Fixed in the shared upstream; all consumers inherit.

## Issue 779 T1+T2 (2026-09-15, M3 session) resolved — subspace_intervention promoted + the FUNCATTN spectral arm POSITIVE (Bench 766)

`katgpt_core::subspace_intervention` (feature `subspace_intervention =
["subspace_phase_gate"]`, OPT-IN per the no-default-consumer rule): the
Issue-778 POC triad promoted to a reusable module — the modelless ridge
probe, frozen-head eval, basis projection, seeded random control,
principal-angle `basis_similarity`, `three_arm_eval` / `three_arm_eval_on_basis`
(caller-supplied aligned basis — the FUNCATTN arm), `affinity_sweep` over a
packed (layer × sample × dim) bank. Zero new deps; zero-alloc eval paths.

**The promotion caught a real POC bug:** the 778 harness's "ridge"
collapses to the class-sum readout `W = XᵀY` (it formed `G⁺·M` then divided
by `σⱼ²` = `(G⁺)⁻¹·G⁺·M = M`) — a serviceable nearest-mean probe, which is
why every gate passed, but not ridge. The shipped primitive computes the
true `Σⱼ vⱼ·(vⱼᵀM)/σⱼ` solve (two-class sanity: w = ±0.5 exact). A second
trap: a parity split with `labels = i%C` at even C aliased the split — the
probe sat at chance with BOTH triad arms equal, and the projection-identity
gate held anyway (it cannot catch a broken head — both sides were chance);
kept as a test note.

**T2 — `spectral_pre_rotate`'s deferred eval (its "Not a GOAT gate" block)
closed, POSITIVE:** the REAL `calibrate_eigenbasis` path on the
SpectralQuant-hypothesis geometry — eigen-aligned 0.802 vs random 0.354 at
k=2 matched budget, and 0.802 > full 0.656 (the projection DENOISES);
residual ≈ chance. Bench 766; the composition runs as a katgpt-attn test
(dev-dep feature-forward — the lib surface unchanged). T3 (real-bank
affinity capture) deferred — cross-repo real-model run, sibling lanes
active; `affinity_sweep` is the ready instrument.

Gates: 7 module tests (identity/contrast/affinity/similarity/
orthonormality/sanity/G4-alloc), release+alloc_tracking clean, default
suite 2060 unchanged (opt-in), clippy -D warnings both crates,
--all-features clean.

## Issue 779 T3 (2026-09-16, M3 session) resolved — real-bank affinity: saturated plateau, NO re-pin; three-arm POSITIVE on real tensors (Bench 767)

The deferred real-model run executed end-to-end once the capture lane
cleared: riir-ai `future_probe_bank_capture` (commit `aa11cb162`) over
gemma-2-2b-it f16 — 384 prompts (6 behavior-intent classes × 8 shells × 8
topics, shells 6+7 held out), all-26-layer last-token residual capture,
greedy-8 prefix label audit (all 6 classes elicit their labeled behavior).
**G0 upgraded to a measurement:** the original /tmp bank was wiped between
sessions; regenerated from the pinned GGUF + tokenizer + committed example
with BLAKE3 IDENTICAL `99edecca…8b81` — the bit-reproducibility claim is
now demonstrated, not asserted. Capture 1581 s (4.12 s/prompt; the
recorded 2053 s was a two-sibling-load run — cite both).

**Affinity axis: NEGATIVE close for the re-pin question (legitimate per the
issue's outcome criteria).** The curve is a CEILING PLATEAU, not a peak:
L02–L25 all decode the 6 classes at 1.000 (floors 0.167) at every
λ_scale ∈ {0.003, 0.01, 0.03, 0.1}; L00 0.990–1.000, L01 0.927–0.969. The
printed "best layer 25" / "peak L00" are argmax tie-break artifacts (Rust
`max_by` last-max; first-max per-class tracker), not structure. Coarse
instruction intents are linearly decodable at ceiling from every layer ≥ L02
— **FutureBehaviorProbe consumers need no measured layer re-pin; the
terminal layer (the collectors' existing default) is as good as any.**

**Three-arm axis: POSITIVE (R557 M2's real-tensor half closes).** At L25
through the frozen ridge head: top-4 probe-SVD dims alone = FULL accuracy
(aligned 1.000) vs random-4 0.146 (6.8×) and 0.062 at k=6 (16×); projecting
the top-4 OUT collapses the readout (0.344 → 0.000). The task signal on a
real residual stream is low-rank and subspace-carried — the freeze-policy
datum transfers from synthetic banks to real tensors. (k=6 residual 0.000
is the degenerate 0-dim complement: bias-only classifies every row wrong;
the projection identity aligned@rank == full holds to 1e-6.)

Instrument verdict: the protocol discriminates in BOTH directions — planted
peaks recovered (778), real saturation measured AS saturation (767). No
revival at this scale; a future affinity revival needs a corpus where
layers plausibly differ (fine-grained / confidence-graded labels, not
course instruction intents). Issue file removed per the noise-reduction
rule — Bench 767 + this row are the trail.

## Issue 782 (2026-09-15, M3 session) resolved — slt_sweep: the noise-sweep λ̂ estimator (781 T4), GOAT G1–G4 ALL PASS, promoted default-on

`katgpt_core::slt::sweep` (feature `slt_sweep = ["slt"]` → **default-on
since landing**; Bench 765): the measurement half of the slt module — a
frozen-weight λ̂ instrument. Deterministic Gaussian direction stream
(xorshift64* + Box–Muller, the module-canonical stream promoted to shipped
surface) + a per-scale-independently-streamed ladder + median aggregation;
caller-owned `NoiseSweepScratch` keeps the estimate path zero-alloc;
`1 + K·m` loss evals per estimate (6145 at default m=1024/K=6) — a
freeze/consolidation-seam instrument, never per-tick.

**The v1 → v2 estimator-form lesson (kept in the module doc as the design
record):** the windowed mirrored-Hill on the smallest-k deficits is exactly
unbiased for exact power laws — but NO polynomial-loss family has one under
the GAUSSIAN probe measure (only the Lebesgue sublevel volume is exact; the
Gaussian radial tail is the cutoff). v1 measured −28…−37% systematic on
the bowl/quartic/RRR anchors. The shipped v2 fits the log-log CDF slope
over an order-statistic ladder (counts ~ m/512 … m/16) — the
**shell-cancelling volume-codimension form** (Murfet et al. 2020 eq. 4.3):
the Gaussian shell factor multiplies both thresholds and divides out of the
slope (closed-form check on χ²₄: ~2% where v1 sat at −17%).

**The geometry-class finding:** TUBE/CONE sublevel sets — singular points
with flat directions (RRR's fiber −1.1%; the ReLU toy's dead-unit cones) —
are near-unbiased because the along-tube Gaussian factor is s-independent;
ISOLATED regular minima (star bodies: bowl, quartic) carry −18…−19% at
d ≥ 4 (+1…+3% at d ≤ 2), monotone in d — within-family ranking preserved.
Sample wall: the near-zero window is Γ(d/2)-starved for λ ≳ 3 — the
feasible domain is the SINGULAR regime λ ≪ d/2, exactly what the module
prices; large-λ recovery stays riir-train Plan 404 (SGLD).

**The ReLU toy boundary (the honest headline):** the paper's cell (H=5,
m=3, K a fixed 32² midpoint sum — the paper's own empirical-L_n shape)
measures LOCAL tangent-cone λ̂ ≈ 2.77, stable at 2.7710 under 9× denser
quadrature and 2.91 at 16× draws — decisively ≪ d/2 = 10.5 (the
singularity IS seen) but NOT the tempered-global SGLD value 0.526: a
local isotropic probe resolves the cone mixture's dominant slope, not the
posterior-weighted global RLCT. Recorded as the instrument boundary in
Bench 765; 0.526 is not a local-probe target. G3: bit-identical under
seed (to_bits); G4: 0 allocs release-verified (Issue-741 predicate);
default suite 2053 → 2060 (floor bumped, measured); clippy -D warnings
clean; --all-features clean; docs_gate 17/17 after the 604/203 count bumps.

## Issue 781 (2026-09-15, M3 session) resolved in `580bda30` — slt: the RLCT λ + WBIC selection primitive, GOAT G1–G4 + floor ALL PASS, promoted default-on

`katgpt_core::slt` (feature `slt = []` → **default-on since landing**; Bench 764):
six closed-form functions over the singular-learning-theory selection
currency — `rlct_reduced_rank` λ(r) = r(a+b−r)/2 (Aoyagi–Watanabe 2005;
LoRA structure exactly), `wbic` nL + λ·log n, `free_energy` with the
(m−1)·loglog n multiplicity term, `bayes_gap` λ/n (the BAYES-predictive gap
law — a point fit realizes C/n, C = 2λ for this family; the module doc
makes the distinction load-bearing), `sigmoid_wbic_weight` σ(−ΔWBIC/τ)
pairwise mixture weights (sigmoid-native; K-way mixes compose
Bradley-Terry products, never a softmax), and `bic_overpenalty_nats`
(r²/2·ln n — the gauge orbit the naive parameter count over-charges vs
the manifold; present at EVERY rank incl. full).

**GOAT evidence (Bench 764):** G1 planted-rank recovery — a=b=8, r*=6,
n=2000, seeded; WBIC picks 6, raw loss picks 8 (r_max, monotone — the
failure mode), naive-parameter BIC picks 5 at the tuned marginal
direction (realized gain ≈24 nats inside the (Δλ≈19, Δd/2≈30)·ln n
window — the over-penalization measured, not asserted). **Load-bearing
loss convention found the hard way:** per-SAMPLE nats (Σ_dims), not
per-dim averages — the first draft averaged over dims, shrinking every
gain by 1/a=8 and collapsing all selection margins (kept as a note so
the next harness doesn't re-find it). UQ floor gate — bayes CRPS/s
0.598 vs the incumbent d/2n floor 0.604 vs constants 0.632+ at equal
0.972 coverage; measured on the WBIC-mixture predictor's realized gap
(the λ/n law's actual referent; same-s scoring isolates the center);
**thin ~1% margin recorded honestly** (a=b=8 — the gauge over-count is
small against λ; widens with r/k*). G2 sub-µs O(k); G3 default count
2041→2053 (test_gate floor raised in-commit, measured); G4 0 allocs
under `--release --features slt,alloc_tracking`; `--all-features`
check clean; docs_gate 17/17 after the 603-total/202-default count
bumps (README ×2 sites + examples/README + the 'and 117 more' → 118).

**T0 novelty gate (the noise-sweep λ̂ estimator): KEEP, with the honest
caveat** — two targeted searches (2026-09-15) + the LLC-estimation
survey (Emergent Mind 2025-10-15) surface only SGLD/tempered-posterior
(arXiv:2308.12108, 2402.03698, 2507.21449), exact-algebraic 2-D
(2608.20183), and linear-response (2605.07970) routes; no
Gaussian-perturbation V(t) power-law route published. BUT the estimator
FORM λ̂ = m/Σ ln(u_max/uⱼ) is the classical Hill estimator (1975) — the
novelty is the APPLICATION (loss-deficit ratios under frozen-weight
noise → λ), not the statistics. T4 (estimator + calibration ladder:
quadratic bowl ⇒ d/2, planted RRR, ReLU toy ≈0.53) deferred as its own
unit, unblocked by the verdict. T6 consumer wiring filed: riir-ai
(freeze/thaw WBIC tie-break + sigmoid mixture weights) +
riir-neuron-db (free-energy cross-n ledger in Raven/δ-Mem merge/keep
ranking).

Anti-Laplace rule shipped in the module doc (R558 §5): no
Hessian/curvature generalization prediction — that instrument class
measured ~10³× the true λ. λ is a freeze/consolidation-seam scalar,
never a per-tick signal.

## Issue 775 (2026-09-14, M3 session) resolved in `3a59abe1` — dual_wave: the PC-ALM dual accumulator + closed-form rate laws (core) + the ballistic DEC wave kernel (dec), GOAT ALL PASS, opt-in

Research 554 (PC-ALM, arXiv:2605.31022) distilled into a two-crate opt-in
feature (`dual_wave` in BOTH katgpt-core and katgpt-dec, same name). Core
`dual` module: the accumulator/shift/credit/energy arithmetic (T1), the
Jury setters + regime classifier with the α-independent annulus (T2), the
arrival laws t_infl = L/√(αη) / alpha_reach / budget_ticks = 2L (T3), and
the exact-adjoint readout λ → −δ (T8) — rates from power iteration on the
STACKED constraint operator's AᵀA. dec `wave_kernel`: the interleaved
(h, λ) recurrence on CochainField pairs (T5) + hodge_triage (T9). α=0
bit-identical to the incumbent diffusion step (T4, unit-pinned).

Three measured engineering lessons are recorded in the bench doc because
they cost real iterations: (1) per-LAYER Jury rates blow up the COUPLED
chain — the bound applies to the stacked operator; (2) the Gershgorin
bound (1+σ̂)² overshrinks η ~2× and kills convergence — power-iterate AᵀA
itself; (3) a power-iteration estimate is ‖AᵀAv‖, NOT ‖AᵀAv‖² — the
first implementation shipped the square (dense-truth probe 3.28 vs the
buggy 10.7). And one honest physics finding: settled-readout settling is
low-mode-limited (~L² worst case — Jury caps ηρσ²_max < 2 while settling
needs ηρσ₁²T ≳ 6), so the T=2L shortcut holds at L ≤ 8 and is the paper's
own "finite-T misaligns" limitation beyond; the convergence-detected
protocol (cosine 0.96–1.0 at every layer, 95–264 ticks) is the honest
gate. GOAT (Bench 763): G2 reach — wave 18/97/212 ticks at L=16/64/128
(linear) vs heat 44/954/4687 (quadratic, ratio ×9 growth — the Eq-23
law); G4 — 2.38 µs @ K=100, 29.9 µs @ K=1024, 0 allocs; G1-adjoint —
cosine ≥ 0.9 every layer every chain. NOT promoted (R554 Q3: game zone
hierarchies are L≈4 — the improvement is real but modest at game depth;
the closed-form laws are the durable value). README/examples counts
synced 601→602 (docs-gate count_features green). Full record: Issue 775
file + `.benchmarks/763_dual_wave_goat.md`; issue closed with the commit
hash referenced in R554.

## Issue 777 (2026-09-14, M3 session) resolved in `7e2a2638` — modality_additive belief kernel: FLYNN's linear-sensory-integration property distilled, measured, GOAT-passed, promoted same-day

Research 556 (FLYNN, arXiv:2607.00025) predicted the shipped belief fusion
(`leaky_step`: divisive `1/total` gain + `−0.5·total` centering) breaks
modality superposition. The same-session PoC (bench `modality_superposition_bench`)
CONFIRMED the failure — but refuted two predictions on the way (both recorded in
the note's §7): the failure is RATIO-flavored (|Σ singles|/|full| = 5.17 — cosine
alone false-passes at 0.97), and `AttractorKernel` PASSES the superposition
protocol in the near-linear regime, so P1 is a necessary-but-not-sufficient
kernel gate (additivity ≠ R304's stability axis — it pairs WITH G2.1, never
replaces it). The T2 primitive `evolve_belief_additive` (per-kind drive
`2σ(η·k)−1`, per-kind retention α, NO cross-modality terms) measures EXACT
superposition (cos/ratio/worst-pair 1.0, max|Δdim| 0.0, predictability err 0)
at 26.8 ns/tick — GOAT G1–G4 ALL PASS, promoted to katgpt-sense `default`.
Measurement-hygiene lessons that cost two bench iterations: an unsink'd timing
loop DCEs to 0.0 ns (black_box required); single-run Instant on ~20 ns kernels
flips run-to-run (best-of-5 min); the divisive kernels saturate to ±1 at large
T making the protocol degenerate (T=16 pre-saturation). G2 shape lesson: a
2× comparative latency gate is ill-posed for a COEXISTING method — gate the
absolute budget (≤50 ns D=8 class), report the comparative number. README /
examples feature counts synced 599→601 / 200→201 (docs gate 17/17). T5
(connectome-vs-rewire dissociation on lif_graph) deferred data-blocked: our
wiring is synthetic shape-class (FlyWire licensing, riir-ai R379), so the
experiment would compare random-vs-rewired-random — structurally unable to
attribute. Full record: `.research/556` §7; issue file removed per the
noise-reduction rule.
## Issue 789 (2026-09-14) — a gate whose own failure path is asserted by nothing: CLOSED

⚠ This heading is deliberately in the form `## Issue NNN (date) — …` rather
than this file's recent `## Issue NNN — …: CLOSED (date)`, because the second
form is one `heading_allocated()` cannot read (Issue 781's measured style gap,
where katgpt-rs scores 7 of 29 and "its own newest closes in the form its own
instrument cannot read"). Writing the record in the shape the instrument reads
costs nothing.

### How it was found

Issue 775 landed `platform_dead_code_floor_gate.py` with *"six canary arms over
the gate's own pin arithmetic, **which the classifier's self-test cannot
reach**"*. That sentence was in AGENTS.md, it was correct, and it named a rule.
The rule landed in **one** gate and was never generalised — the **sixth**
recorded instance of that shape (Issues 777, 778, 793, 782, 783). It was found
by asking the question one level up from Issue 787: 787 mechanised *"is every
instrument findable?"*, and nothing asked *"is every gate's own verdict
validated?"*

⛔ **The first census was wrong, in the over-reporting direction.** It grepped
the CLI flag strings `--canary` / `--prove-fires` / `--self-test` and reported
**nine of twenty** CHECKS bare. Three of those nine invoke an arm they
**delegate** to the classifier they read (`percentile_floor_gate` →
`percentile_index_audit.selftest`, plus `cfg_row_implication_gate` and
`trap_sentinel_gate`), which is the DRY answer and the correct one. A census
over one representation — here a CLI flag string — is blind to whatever that
representation omits. Issue 787's lesson, reproduced within ten minutes of
going looking for a fresh instance of it.

### The measurement

Resolved by AST, crediting a call to `selftest` / `self_test` / `canary` /
`gate_selftest` / `prove_fires` whether defined locally or reached through an
imported module:

| | count |
|---|---|
| invokes its own arm | 10 |
| invokes a **delegated** arm | 3 |
| invokes both (775's shape) | 1 |
| **invokes NOTHING** | **6** |

`issue_citation_gate` 750 lines · `cargo_comment_audit` 480 ·
`skill_repo_set_gate` 306 · `count_features` 261 · `docs_gate_checks_sync` 141 ·
`markdown_fence_gate` 112 = **2,050 lines** of per-push gate logic whose failure
path no test had ever executed. `docs_gate.sh`'s own header is the argument:
*"An assertion nobody invokes is decoration, and a red one nobody invokes is
worse: it trains the next reader to assume the tool is broken."* Two of the
three checks that file was written for were RED on `develop` when it landed.

### T1 — the shared fence parser was blind to half of CommonMark

Two of the six shared one function, and a third (`issue_citation_gate`) imports
it too. `skill_repo_set_gate.fenced_blocks` is read by **three** per-push gates
under Issue 755's explicit DRY call — *"a second copy of a rule this subtle is a
second thing to get wrong"* — which was right, and which concentrated the whole
risk into one function no test had ever touched. AGENTS.md documented the
mis-phasing hazard it exists to prevent and recorded that **its own first canary
was swallowed by exactly that bug**. The canary was never replaced.

`fence_run()` counted leading **backticks only** from the day it was written.
CommonMark fences are backtick **or tilde** and the families do not interoperate.
Ten known-answer arms: eight passed, **two failed**.

| arm | expected | got |
|---|---|---|
| `~~~` / `x` / `~~~` | one block `(1,3)` | `[]` — invisible |
| `~~~` / ` ``` ` / `~~~` | one block `(1,3)` | `[(2,-3)]` — a phantom unterminated fence on a body line |

Both directions of damage: **silent** (a tilde-fenced command block is never
scanned, so `skill_repo_set_gate` certifies a repo set it never read, and a
tilde-fenced *unterminated* block is invisible to the gate written for exactly
that class) and **loud at the wrong address** (an odd number of backtick lines
inside a tilde block reds at a line that is not the defect — the failure mode
`markdown_fence_gate`'s own docstring warns the reader about).

**Exposure was LATENT: 0 tilde-fence lines over 5116 tracked `.md` across 16
repos.** The `orphaned_attr_gate` standing — pinned at zero, measured zero
everywhere, worth forbidding because the guarded gate's docstring claimed to be
"CommonMark-ish" on precisely the axis it could not read.

`selftest()` is the replacement canary: 19 parser arms over both families, 11
over `scan()`'s own detector arithmetic, and the Issue-765 partial-clone
deferral in both directions. Covering only the imported parser would have marked
the gate "validated" while 306 lines of verdict logic stayed unasserted — the
same over-crediting the first census committed. It exits **2**, not 1: a
mis-phasing scanner reports clean in both directions, so its failure is not a
finding, it is the absence of a verdict.

⛔ **Two of nine perturbations red nothing, and both are recorded where they are
read rather than quietly patched:**

- `ch == open_ch` is **REDUNDANT today** — `.strip(open_ch)` already
  discriminates the family, so perturbing the family test away alone reds zero
  arms. The first version of that comment called it load-bearing; the
  perturbation refuted it. **A line a canary cannot red is not doing the work
  you think it is.** It stays, because the two are independent CommonMark
  requirements and each becomes load-bearing the moment the other is loosened.
- one lookbehind arm was aimed at the name's **right** side, where the pattern's
  trailing slash already does the work. Kept, labelled inert, and joined by the
  nested-path arm that discriminates it.

T4's free half: `markdown_fence_drift_sweep.py` imports the same parser, and 0
unterminated holds over 5121 tracked+untracked `.md` with tildes now visible.

### T3 — the four remaining bare checks, and two more real defects

**`cargo_comment_audit`** — `WEAK_DEFAULT_RE`'s negative lookahead excluded
``default (`0.82L`)``, a shape appearing in **no manifest in any repo**, while
the one live instance is the other order — ``default `(0.82L→0.45L)` ``, the
exact string the lookahead's own comment has always quoted
(`cross_stage_relocation`, root `Cargo.toml`). Latent, because that line also
says "promotion blocked" and rung 1 reaches a verdict before rung 6 misfires.
The fix covers both orders and changes **zero** classifications over **7,465**
inline comments workspace-wide.

⚠ Deliberately **not** widened to any backtick: `(?!\s*\(?`)` reads a code
reference as no claim and takes **21** comments of the form *"Not in `default`
directly; transitively enabled via `X`"* from `default` to `unknown` — silently
out of the cross-check. Measured before choosing. Those 21 do carry a claim no
rung reads, which is a separate gap, recorded where `WEAK_DEFAULT_RE` is read.

Also pinned: the **rung-2 paren guard is load-bearing only in combination with
rung 3.** Its comment names `default-on (behavior opt-in …)` as the false match
it prevents; the lowercase form falls through rung 3 and rung 4 classifies it
`default` anyway. Both arms pinned so the asymmetry is a measurement.

**`count_features`** — `CLAIMS` hoisted to module level and the paren-depth
tokenizer extracted as `outside_parens()` so an arm can reach it. `SWEEP`'s arms
are the exact phrasings its comments record having escaped ("999 tunable flags",
"999 default features"). Those comments said it was *"canaried at each
widening"* and that was TRUE — but the canary was a person at a terminal, so
nothing re-ran it and a later **narrowing** would have been silent.

**`docs_gate_checks_sync`** — arms over both parsers and the quantity extractor,
including addresses-are-not-quantities at four shapes. Its refusal paths print
an `✗ INSTRUMENT` line, so `expect_exit()` swallows the arm's output: a clean
run that prints two fake failures is a gate whose next reader assumes it is
broken.

**`issue_citation_gate`** — the largest, and the one with the worst record:
AGENTS.md documents it wrong **twice** in the direction that absolves (752's
`named != {}` read 45 rows as clean; 754 then refuted the census that found
them, because all 45 reads asked the same blind `allocated()` the same
question). So its 39 arms aim at the **suppressing** paths first —
`is_qualified`'s owner-consistency and ORPHAN branch, `heading_allocated`'s
three filters (the only path here that can make a finding disappear) including
Issue 781's two measured negatives, the window/adjacent split and the alias's
one direction, list expansion at all four separators, and `fenced_lines`
failing **safe**. It runs before the deferral branches, not after: this gate's
loudest posture is an instrument-ALIVE deferral, and a deferral printed on top
of a broken classifier is the one output here that must not be possible.

### T2 — the mechanism: `scripts/check_validation_gate.py`

The predicate is **invokes an arm UNCONDITIONALLY**, not "has an arm", and that
distinction found a case one day old. `docs_gate.sh` runs each check as
`"$PY" "$script"` — **no arguments** — so `population_sync_gate.py`'s eight
adversary arms, landed by Issue 788 the day before, sat behind `--canary` and
ran on **no push at all**. They cost **0.17s**, so the flag was never buying
anything. The canary is unconditional now (output swallowed on success,
`--canary` kept as the verbose mode), and perturbing its docstring headline
count reds `main()` with rc=2, which it could not do before.

`ARM_NAMES` is the **permissive** direction and the floors alone do not guard
it: an empty set reds every check and is impossible to miss, while a set that
quietly widens (add `main`) greens every check silently — so a canary arm
asserts `main` is not in the vocabulary. Two blindness floors: `MIN_CHECKS` the
array parse, `MIN_ARMED` the AST resolution, because a walk that finds every
check and credits none looks exactly like nobody having written any arms.

`check_validation_expected.txt` is **deliberately empty**: T3 wrote the four
remaining arms rather than pinning them, because a row reading "not written yet"
is a backlog wearing a pin (785's rule). A reasonless row is refused; a row
whose check has since grown an arm reds.

⚠ **What it does not assert:** that an arm which exists and runs is any *good*.
**Seven** arms written for this issue certified nothing until they were fixed —
one whose anchor string was wrong, one whose fixture had no terminated fence for
the fail-safe to discard, one aimed at the wrong side of a lookbehind, one whose
input order already matched sorted order. Arm quality is not statically
decidable and is not claimed; the verdict is the weaker thing, said out loud
where it is read.

`--prove-fires 6804d983` (the commit that FILED 789) is two-sided against an
independently known answer: **7** checks unarmed there, 6 bare and 1 flag-gated,
named individually. ~0.3s, opt-in, `scripts/` only.

### T4 — no sweep, and that is a measurement

Every other verdict class here has a workspace half because the question
generalised. This one does not. Measured over the 16 repos on this box:
**katgpt-rs is the only repo with a `scripts/docs_gate.sh` CHECKS array at
all** — riir-train has 58 `scripts/*.py` and riir-ai 7, and neither has such an
array. A sweep would derive a population of ONE and print a confident green over
it, which is the stated reason `ci_gate_coverage.py` is kept out of the CHECKS
set. The cross-repo question that *does* generalise is "is this instrument
findable?", already ratcheted by
`instrument_reachability_drift_sweep.py`. **Do not add a sweep here by symmetry
with the family; re-run the measurement first.**

### Verification

docs gate **21/21** (CHECKS 20 → 21) after every landing. New check ~**0.11s**
standalone — the cheapest in the set, and the fourth same-day CHECKS move, which
is the argument for writing the count next to the timing figure rather than the
figure alone. `markdown_fence_drift_sweep` 0 unterminated over 5121 `.md`, both
population postures. 30 + 14 + 9 = **53 perturbations** run across the new arms;
all red after the four inert ones were re-aimed.

### The standing lesson, now recorded seven times

A rule landed in one instrument and never generalised (777, 778, 779, 782, 783,
789) — and a census over one representation is blind to whatever that
representation omits (787, then 789's own first pass, ten minutes in). Before
fixing a class, grep the whole family and land the repair as one shared
mechanism; and before trusting a census, ask which representation it read.

## Issue 788 — the population-predicate registry was hand-maintained, and a careful reading missed two of ten: CLOSED (2026-09-14)

`population_sync_gate.py` exists because a hand-duplicated **predicate** drifts
exactly as a hand-typed count does: if one instrument's "which repos are
contract repos" answer diverges, that instrument quietly audits a different set
and still prints green. Its `PREDICATES` tuple is DATA, and its comment always
said adding an instrument was *"a one-line change here rather than an eighth
silent divergence."*

The one-line change is the part nobody makes. **The registry sat at seven while
ten existed**, and the gate printed "7 predicates agree" the whole time.

**Read the numbers in order — they are the argument for mechanising this rather
than reading carefully.** The census that filed the issue counted NINE and
registered the eighth. Then the completeness check found the **ninth and
tenth** — `docs_drift_sweep.derive_population` and
`wasm32_surface_audit.derive_population` — which that census had missed,
because its shape test was "defines `derive_repos`/`repos`/`contract_repos`"
and these two are named `derive_population`. A careful reading missed two of
ten, on the same day, in the issue about a registry being hand-maintained.

**Three real defects fell out, none of which was the one being looked for.**

1. **The eighth was WRONG.** `len_derived_binding_audit.derive_repos` tested
   `(d / ".git").exists()`, not `.is_dir()`, so it admitted a worktree-shaped
   directory and would double-count a repo already in the walk. It had done so
   since the audit was written; it was caught on the **first run after
   registering it**, by the synthetic-workspace arm that has had a
   `worktree-shaped` case all along. Latent, not active — no such directory in
   this workspace — so Issue 786's measurements are unchanged.
2. **Two of the ten were UNPARAMETERISED**, hard-coding their root from
   `__file__`. The synthetic-workspace half of this gate — the half that works
   in CI, where there is no workspace to walk — could not have tested them even
   if somebody had registered them. Both take an optional `root` now.
3. **The real-workspace verdict was coupled to unrelated failures.** `if not
   bad:` guarded the "all N predicates agree" line, so any other red suppressed
   it and a reader could not distinguish *they disagree* from *we did not say*.
   A local flag now.

**`SUBSET_PREDICATES` is its own tuple, because "not registered" and
"deliberately not registered" were the same state.**
`restatement_theorem_audit.repos` adds a `.proofs` test (4 of 16) and would red
every run if registered as an equal. That exclusion was recorded **nowhere**:
the next reader either re-derives it or registers it and breaks the gate.
Subset predicates get a weaker but real assertion — a strict subset of the
agreed answer — which catches a `.proofs` walk that has silently started
matching something else, and nothing else in this workspace would.

**The detector's own boundary was wrong first, and its docstring asserted the
result before anything had been run.** The first version asked only for a `def`
whose body mentions `BOUNDARY.md` and `.git`, with a docstring claiming it
"measures exactly the nine real predicates". It reported **23** — mostly
`main()` and `selftest()` bodies that merely name the two strings, plus this
gate's own `build_synthetic()`, which *writes* those files rather than walking
for them. The discriminating term is the **directory iteration**. Same lesson
as `platform_dead_code_audit.py`'s first sweep: a conservative-by-construction
argument is a claim about code somebody else wrote, and it does not survive a
real corpus.

`ast` is deliberately not used: the gate must classify a file it cannot import,
because a syntax error in a sibling instrument is somebody else's finding and
not a reason for this gate to go blind.

**The escape hatch is explicit and it got used twice, correctly.** A `def` line
may carry `population-predicate: not a contract-repo walk` — noisy to type,
greppable to review. `restatement_drift_sweep.main` (it *calls* the registered
subset predicate) and this gate's own `canary()` (its fixtures embed predicate
source as data, which no textual detector can tell from the real thing — the
same limitation Issue 787 records for its closure) carry it.

8 canary arms, both directions, including one that pins the docstring's
headline count against the tuple — the number a reader trusts without running
anything, and exactly what went stale for four instruments. CHECKS stays at 20;
the row's quantity words moved from *seven* to *ten*.

## Issue 787 — a census reads the DOCUMENT, so an undocumented instrument is invisible to it: CLOSED (2026-09-14)

**The correction first, because this issue exists to make it.** `1a5b6571`
bounded the Issue 785 close-out from "every cross-repo class in `scripts/` now
has both halves" to "every cross-repo class **whose verdict is walled at a small
number** has both halves". That bounded claim was **still false**, by exactly
one instrument: `len_derived_binding_audit.py` was cross-repo, its joined
buckets were walled at 0, and it had no verdict half — Issue 786, filed and
closed the same day. The bounding correction was itself incomplete.

The miss is not the finding. The **mechanism** is. Both censuses — the one that
produced 783/784/785 and the one that produced the bounding correction —
enumerated the audits **AGENTS.md documents** against their sweep halves, and
AGENTS.md did not name that audit at all. A census that reads the document
cannot see an instrument the document omits, and it reports a confident,
complete-sounding answer over the subset it can see. That is every blindness
floor in this repo one level up, with the DOCUMENTATION as the population
nothing floored.

**The predicate is REACHABLE, not "documented in prose".** Roots are
`AGENTS.md`, `scripts/docs_gate.sh` and `.github/workflows/*.yml`; the closure
follows script → script references from there. That is not a convenience — the
cases demand it. `all_ignored_target_audit.py`, `cfg_row_implication_audit.py`
and `ci_test_execution_report.py` appear in no document either, yet each is
invoked by an instrument that IS documented and each runs per-push as a result.
A bare "must be named in AGENTS.md" rule reds all three and teaches whoever hits
it to stop reading the gate.

⛔ **`HISTORY.md` is deliberately not a root** — this file. It is the archive,
its own header says operational rules live in AGENTS.md so agent context stays
small, and it is not loaded into a session. An instrument findable only from
here is precisely the instrument that stops being run, which is what 786
measured. Counting it would have made the gate vacuous on the one case that
motivated it.

**Measured 2026-09-14, this repo:** 63 tracked `scripts/*.py`, 13 roots, 56
reachable, **9 unreachable**. Two of the nine were real instruments and both
were **wired into AGENTS.md rather than exempted** — that is the default:

- `list_unresolved_percentile_sites.py` → the percentile section. It dumps the
  UNRESOLVED rows the main audit prints only as a tally, which is the one
  bucket that needs a per-site read.
- `citation_weight.py` → Numbering Discipline. It decides which of two
  documents sharing a number keeps it, and the obvious count is measured to be
  the wrong one: on riir-ai's six duplicates the by-NAME citations are 0-2 per
  side and TIED in four of six, while the `Plan 175` form carries 35-98 each.

The remaining seven are pinned by MEMBERSHIP with a **reason per row** (a
reasonless row is refused): five `kimi_ref/` reference-implementation files, a
manual CoreML generator, and `gguf_header_audit.py`, which `1a5b6571` had
already placed outside the class-audit family.

**The known-answer validation is free and two-sided.** `--prove-fires 18dbe980`
extracts the commit and its parent: at the parent, `len_derived_binding_audit.py`
was named only in HISTORY.md and was the TENTH unreachable script; at the fix it
is reachable. The gate would have caught 786 before either census missed it.

**`min_roots` is the floor that is easy to leave out**, and it guards the
direction nobody notices. An empty or unreadable root set makes EVERY script
unreachable and reds loudly all by itself. But a root set that quietly *widens*
— a generated file, a directory of YAML that happens to mention every script
name — makes every script REACHABLE and prints a confident green.

**The first sweep run changed the pin design, and that is the second finding.**
Workspace-wide the same predicate finds **95 unreachable of 152** tracked
`scripts/*.py` over 16 repos, and **riir-train is 61 of 61**: its `scripts/` is
almost entirely plan-scoped one-offs (`plan341_band_pool.py`,
`plan346_diversity_gate.py`, `t504_harvest.py`) and its AGENTS.md names none of
them. Read honestly, the predicate **over-captures** there — a plan artifact is
not an instrument, and "unfindable from AGENTS.md" is the correct state for a
script whose whole life was one plan task.

So the sweep is a **RATCHET**, uniquely in this family: `max_unreachable`
pinned at each repo's measured count. Membership with a reason is right for the
7 rows this repo owns; it is not right for 95 rows whose judgement calls belong
to 15 other repos. The ratchet constrains the DERIVATIVE — the commit that adds
ANOTHER unfindable script reds — which is the strongest claim this repo can
honestly make about somebody else's tree.

⚠ That is deliberately not the Issue 785 rule's target. That rule forbids
ratcheting a bucket meaning *unanswered*, a backlog with no owner. This bucket
means *unfindable*, every row has an owner, and the action on a red is immediate
and local. It is also not the `suite_membership_audit` outcome (1,203 rows,
report-only, no verdict at all): there, no per-commit action follows from the
number.

Six repos have 0 tracked `scripts/*.py`, so both floors are 0 and neither
detects anything there — Issue 783's population shape again. What rescues those
rows is that the gate's `DOC_ROOTS` handling is a REFUSAL and not a floor: a
repo whose AGENTS.md the walk cannot see is an instrument failure, not a clean
zero.

**⛔ The closure is TEXTUAL, and the gate proved it on itself — three times, in
one sitting.** A basename mentioned anywhere in a tracked script credits
reachability, string literal and comment alike, and it cannot be narrowed by
parsing (a genuine invocation IS a string literal, and the primary root is
prose). On the gate's **first staged run** it red on
`scripts/kimi_ref/fla_stub.py`: a canary arm named that real exempt script as
fixture data, the arm's own file is in the population and reachable from
`docs_gate.sh`, so the subject became reachable and the membership pin's other
direction fired — correctly. Fixing that surfaced the same mechanism in the
UNREACHABLE arm, whose injected path was a literal in the same file; assembling
the basename at runtime fixed it, and then the arm failed AGAIN because the
comment written to explain the assembly spelled the name out contiguously.

Read the leniency direction honestly: for this class a false *reachable* hides
exactly the instrument the gate exists to surface, so it is the dangerous one.
A row that leaves the unreachable set without a wiring commit is suspect — check
WHAT started naming it.

Canaries: **9 arms on the gate, 8 on the sweep**, both as `--canary` flags
rather than transcripts. Two more are worth naming — the basename-collision
refusal (with two `x.py` in a tree, a basename hit cannot say which is meant,
and guessing credits coverage to the wrong file) and the sweep's `bump()`
helper, which perturbs a row by FIELD INDEX rather than by a literal string, so
an arm cannot silently stop perturbing when a pin is re-measured. That is the
Issue 786 canary failure, designed out.

The gate is the cheapest check in `docs_gate.sh` at **~0.24s** — the closure
re-reads only files a root or a script actually names. CHECKS: 19 → **20**.

## Issue 786 — the `.len()`-derived binding audit had no verdict half, and the reason it went unnoticed is the finding: CLOSED (2026-09-14)

`scripts/len_derived_binding_audit.py` — 983 lines, 16 repos, 8,694 tracked
`.rs`, 52 `.len()`-deriving cube kernels, 164 bind sites, nine verdict buckets
— had no gate, no sweep, and **no `AGENTS.md` entry at all**. Its only mentions
in this repo were three incidental `HISTORY.md` lines. It was the last
cross-repo instrument in `scripts/` in that position, after `1a5b6571` bounded
the two documented exceptions (`suite_membership_audit.py`, 1,203 load-bearing
unpinned rows; `gguf_header_audit.py`, model-file introspection).

**Ninth instance of one shape** (Issues 777, 778, 793, 782, 783, 784, 785), and
the QUIETEST. 784 and 785 were found because a hand-typed cross-repo figure had
gone stale in public — 46% stale in 784's case. This one had no such figure to
go wrong. An instrument nobody is told about does not drift into error; it
simply stops being run, and there is no symptom to select on. That is the
argument for the census: enumerate the instruments against their verdict
halves, rather than waiting for one of them to say something false.

It had also **already gone blind once** without its output betraying it. Issue
777 found it walking the filesystem behind a hand-typed skip set, crediting
mmorpg-remaster's gitignored nested `mmorpg/` repository and riir-train's
cargo `OUT_DIR` sources to their enclosing repos, and migrated it to
`tracked_walk` in that commit. A floored sweep would have made the population
shift (11,132 → 8,694 `.rs`) an assertion instead of a paragraph.

**Three measurements decided the pin design, and each one contradicted a shape
that could have been copied from the previous sweep.**

1. **The population is bimodal.** riir-ai 43 kernels / 143 bind sites,
   riir-train 9 / 21, **the other 14 repos 0 / 0**. So `min_kernels` and
   `min_binds` are vacuous in 14 of 16 — Issue 783's shape, not Issue 784's
   (where both floors bite everywhere). There is deliberately **no reserved
   `TOTALS` row**, unlike `wasm32_surface_drift_floors.txt`: a global kernel
   floor reds on a partial-clone box the moment riir-ai or riir-train is
   absent, which is exactly the case `population_verdict()` exists to DEFER,
   and the two non-zero per-repo rows already catch a workspace-wide
   classifier break.
2. **The verdicts are cross-repo by construction, and `DEFERRED` does not cover
   that.** HALF C resolves a wrapper parameter's provenance through WORKSPACE
   callers, so a partial clone can corrupt the verdict of a row in a repo that
   IS present — a row measured WRONG, not a row not measured. No other sweep in
   the family can do this; they all classify per repo. Measured both
   directions: **7 of 251** cited caller references are cross-repo (riir-ai
   rows resolved through riir-train callers), and **leave-one-out over all 16
   repos produces 0 verdict flips** — every one of those 7 edges is an
   additional caller on a row a same-repo caller already decided. So per-repo
   pins are sound TODAY, and rather than carry that measurement forward as a
   claim the sweep re-runs a TARGETED leave-one-out every time, over a supplier
   set derived from the run (today riir-ai + riir-train, ~8s each). A new
   cross-repo edge joins the check by existing.
3. **UNRESOLVED is 118 of 164 and stays unpinned.** Issue 785's rule: a ratchet
   on a bucket meaning *unanswered* is a backlog. wasm32 could wall its
   UNRESOLVED at 0 only because Issue 738 T1 drove it there by ANSWERING the
   rows; here the bucket is "provenance one level up, caller not a path-form
   associated fn", which HALF C cannot reach at all. Reported, reason printed
   where it is READ, never folded into a pass or a fail — the
   `suite_membership_audit` precedent.

**The one sweep in the family with no `min_rs_files` column.** Three others
floor that quantity per repo over the identical `tracked_files(repo, "*.rs")`
call and the identical population, so a fourth copy adds zero detection power.
They also already DISAGREE — katgpt-rs pinned 1500 / 1400 / 1500 and riir-ai
1500 / 1500 / 1800 across `orphaned_attr`, `platform_dead_code` and
`percentile`, against 2415 and 2626 measured. Harmless (each is an
independently chosen slack floor, not an equality) and not a shape worth
extending. ⚠ "Somebody else covers it" is an assumption unless checked, so the
sweep ASSERTS the delegation: every repo it pins must still carry a non-zero
`min_rs_files` row in `orphaned_attr_drift_floors.txt`, and the reader that
parses that file REFUSES a shape it cannot read rather than returning `{}` —
an empty dict would turn the whole assertion into a no-op, which is the failure
it exists to prevent.

**T1 was the enabling work, as it was in 785.** The classification lived inline
in `main()`, so a sweep could only reuse it by copying HALF A + HALF B + HALF C
+ the guard-only re-verdict pass — a copy of the whole instrument. Extracted as
`classify_workspace(repos)`, with the two global floors lifted to
`FLOOR_RS_FILES` / `FLOOR_KERNELS`; the report's output is byte-identical
before and after.

**Canaries: 12 arms, all measured, embedded as `--canary`, and one of them
failed first.** They are a flag on the sweep rather than a one-off transcript,
because a pin nobody has watched fail certifies nothing — opt-in rather than
default (contrast `platform_dead_code_drift_sweep`'s `--prove-fires`) since the
arms re-enter `main()` and a verdict that runs its own adversary on every
invocation is one more thing between a reader and the answer. Baseline
green · both parse floors · UNPINNED · the `max_findings` wall (a real bind
relabelled `CAPACITY` through the shared classifier, the only honest way to
plant the joined defect without writing Rust into a sibling) · the EYES
membership pin in both directions · the EYES count WITHIN one address · the
delegation break · the unreadable-delegation refusal (exit 2) · the empty-pins
refusal (exit 2) · the cross-repo flip. The arm that failed was the UNPINNED
one, and it failed because **its own anchor string was wrong** — it matched on
`11\n` where the pinned row ends `11             0`, so it perturbed nothing
and the green it got was real. That is the canary failure mode this repo keeps
recording (Issue 775's `vendor/` arm certified the code path it was not aimed
at): an arm that does not perturb certifies nothing, and only the fact that it
was EXPECTED to red made it visible.

Population direction verified both ways as well: marker-on DEFERS the four
absent repos by name on the PASS line, marker-off reds with UNSEEN over the
same four.

Standing at close (2026-09-14, 16 of 20 repos): **0 joined findings · 4 EYES
addresses · 118 UNRESOLVED** over 8,694 tracked `.rs` / 52 kernels / 164 bind
sites. The EYES rows and the UNRESOLVED rows are riir-ai's and riir-train's to
adjudicate — this repo is upstream of both, and what 786 delivers is that the
set stops being unasserted.

## The platform-dead_code class got an instrument — `scripts/platform_dead_code_audit.py`, and it was wrong on its first sweep (2026-09-14, 4090 session)

The class below (NEON_U8, `ea4c2873`) was found by a human running clippy on a
lane no CI owns. This session finished the classifier that answers "how much
of this is there" without needing that lane: module-scope decls whose EVERY
identifier occurrence resolves to a narrower platform cfg than the
declaration's own, with the cfg context composed from item attrs, block
attrs, blockless-statement attrs, file-leading `#![cfg]`, and `mod foo;` gates
resolved ACROSS FILES up the directory chain. 2415 files in 6 s for this repo,
8694 tracked `.rs` over 16 repos in 26 s. Rule, buckets, invocation:
AGENTS.md §"An item can be dead on a platform NO lane compiles".

Validation is two-sided by construction — `--prove-fires ea4c2873` extracts
`ea4c2873~1` and `ea4c2873` via `git archive` and requires `NEON_U8` PRESENT
at the parent and absent at the fix (measured: 1 finding → 0, over 268 files /
3635 candidates), and 24 self-test arms run on EVERY invocation with a
classifier MISS exiting **2**, not 1.

**Three things it got wrong before it got anything right**, all recorded in
the script's own header because the pattern is the lesson:

1. **It INVENTED a finding, in the one direction its header swore it could
   not.** Masking string literals dropped Rust 2021 inline format args, so
   riir-ai's `SWEEP_COUNTS` — `println!("Sweep sizes: {SWEEP_COUNTS:?}")`
   ungated in `main`, `target_os = "macos"` everywhere else — read as dead.
   A conservative-by-construction argument is a claim about code somebody
   else wrote, and this one survived review and died on first contact with a
   real corpus. Two arms pin it now (format arg / plain mention).
2. **A `mod` row is not a rustc finding, and only a compile says so.**
   `katgpt-types/src/simd/mod.rs:49`'s `mod horizontal;` is ungated with all
   15 references `target_arch = "x86_64"` — and `cargo check -p katgpt-types
   --target wasm32-unknown-unknown` emits NOTHING, because every item inside
   is itself x86_64-gated and the module is EMPTY there rather than dead.
   Appending one ungated `fn` to that same file reproduces the warning
   immediately, on the **fn**. So MOD-REF is its own bucket, never folded
   into the count, and the katgpt-rs row stands as an observation.
3. **An arm that passes under its own perturbation certifies nothing.**
   Neutering `vendored_p` red zero arms: the synthetic trees have no `.git`,
   took the filesystem-walk branch, and a redundant `"vendor"` in `SKIP_DIRS`
   was doing the filtering. One exclusion, two code paths, and the canary
   watched the path it was not aimed at. Removing the duplicate made it
   two-sided; the other two perturbations (format restore, MOD pooling) each
   red their arm.

**The two real rows, both compile-verified before repair** (riir-ai
`develop`): `riir-gpu` `note_ane_dispatch` — ungated at
`ane_prefill/mod.rs:535` with all three call sites in `exec`/`exec_zc`, both
`all(target_os = "macos", target_arch = "aarch64")` — reproduced as
`warning: function note_ane_dispatch is never used` on x86_64 Windows with
`--features ane_prefill` (the default-feature check is a green ZERO: the
module is `#[cfg(feature = "ane_prefill")]`). And `riir-games-shared`
`gen_u64_bytes`, whose only call site is `gen_usize`'s
`target_pointer_width = "64"` arm — `warning: method gen_u64_bytes is never
used` on `wasm32-unknown-unknown --features chacha20_rng`, i.e. a 32-bit
target this workspace actually builds, not a hypothetical one. Both gated to
mirror their call sites; a third adjacent-class row (`unused_mut` on a `ctx`
mutated only inside the same macOS block) took a **conditional**
`cfg_attr(not(...), allow(unused_mut))` so the ANE platform still warns.
Standing after repair: **0 findings · 1 MOD-REF** over 8694 files / 3433
units / 119452 candidate decls / 16 repos. The macOS+aarch64 arm of the
riir-gpu gate is NOT compiled here — the cfg is copied verbatim from the call
sites' own, so it cannot narrow anything they do not already carry, but that
is an argument, not a measurement.

⚠ Unrelated measurement worth not over-reading: `docs_gate.sh` ran **17/17**
on this Windows box for the first time (Python **3.14** on PATH as the
`python3` shim — the four `tomllib` gates the 3.10 shim permanently broke all
pass — plus `DOCS_GATE_PARTIAL_CLONE=1`), and printed **1.81s CPU / 14.1s
wall** against the documented 13.37s CPU at the same 17 checks. Do NOT revise
the documented figure on it: the mechanism is unmeasured and there are two
live hypotheses — the interpreter, and the shim wrapping a native Windows
`.exe` in `sh -c`, which may defeat `times` child-CPU accounting entirely.
The wall figure is in range; only the CPU one moved.

## The Windows all-features lane — NEON_U8 platform gate, first specimen of the platform-dead_code class (2026-09-14, 4090 session)

Found by the idle Protocol B sweep (`cargo clippy --workspace --all-targets
--all-features`, the lane NO CI owns: `full_gate` runs macOS/aarch64 where the
const is USED, `wasm32_gate` compiles wasm32 — neither ever compiles
x86_64-native with all features). `NEON_U8` (`katgpt-pruners`
`interval_pruner/simd.rs:27`) was declared without an arch gate while its only
use sits inside the aarch64-gated `neon_is_interval_closed` — dead_code on
every non-aarch64 host whenever `interval_pruner` compiles. Warning vintage
`432cacf7` (2026-06-12); the `8914b79d` "--all-features backlog 141→0" sweep
ran on a host where the const is alive, so its zero was host-scoped truth —
the standing "a platform is part of the claim" lesson with a fresh instance.
Fixed `ea4c2873` (gate mirrors the adjacent `AVX2_U8` line; interval_pruner
lib tests 166/166 on x86_64). The same sweep found the instrument lesson
recorded in riir-clippy's snapshot: a `clippy::`-prefixed JSON grep reads a
FALSE all-clean — rustc-family codes (`dead_code`, `unused_*`) carry no
prefix; grep all warning codes, and filter the Windows hard-link
incremental-cache noise by message content, never by unit tally. The class
recurred same-day in the zed fork (4 more specimens, incl. an ungated
macOS-only Metal example and the issue-021 file's own `KEYCHAIN_SERVICE`) —
class record + detection shape: riir-clippy `.distill/001` P22 (`61bb0d54`).

## Post-riir-train-513 develop drift — the 09-12→09-14 touched-rows window audited green on the workstation (2026-09-14, M3 session)

riir-train Issue 513's T2 sweep measured this repo's 623 rows through its 09-05..09-11 window; the
row count has since drifted to 710 with develop landings the main-only CI lane never
audits. T4's own gate (`required_features_touched_gate.py`) run over the defined window
base `9b8cf60e7` (2026-09-12 00:00 +07) → HEAD, 191 commits → **25 selected rows**
riir-train Issue 513's T2 sweep measured this repo's 623 rows through its
09-05..09-11 window; the row count has since drifted to 710 with develop
landings the main-only CI lane never audits. T4's own gate
(`required_features_touched_gate.py`) run over the defined window base `9b8cf60e7` (2026-09-12 00:00 +07) → HEAD, 191 commits → **25 selected rows**
(katgpt-core 13 · katgpt-rs root 7 · katgpt-attn 3 · katgpt-kv 1 · katgpt-backend 1) —
**25/25 BUILDS at their own EXACT feature set · 0 FAIL · 0 NO-FEAT · 0 UNSEEN** (isolated
`/tmp/katgpt-rs-rf`, ~11 min wall at load ≈5-7, the transient-build class the 09-13
owner call leaves to sessions; dir left warm for the next run). Read narrowly per the
gate's own NOTE — rows reachable only through a library change remain the full sweep's.
The 09-11→09-12 inter-window sliver (between T2's per-repo HEADs and this base) is
unmeasured; the other row-carrying repos' same-window tails (30 rows / 7 repos) defer
to the new hardware with riir-ai's record — this box no longer runs multi-repo batches.

## Issue 774 (2026-09-14, M3 session) resolved in `22e65be4` — the wasm32 surface audit's BY-DEP verdict: the row predicate was dep-blind, closed with a five-canary self-test

Filed and landed the same day, from the quiet-repo audit sweep. The
resolver upgraded a package only on ROW evidence (`-p` / literal
`--manifest-path` / the two 738-T1 shapes), so a lane building a package
TRANSITIVELY read as UNCOVERED: `-p riir-shader-showcase --target wasm32`
compiles the showcase's in-repo path deps too, and riir-shader's core (1
site) + effects (2 sites) read UNCOVERED while every `build-wasm.sh`
bundle build compiles them — compile-verified before filing
(`cargo check -p riir-shader-effects --target wasm32-unknown-unknown`
exit 0, 30.3s).

The fix is a FOURTH verdict, `✓ by-dep`, never folded into NAMED — the
distinction is load-bearing: by-dep coverage dies by an innocent
dep-graph edit in someone else's manifest, and the reader must see which
kind of coverage they hold. Credit rules (all conservative): non-optional
path deps in plain `[dependencies]` + target tables whose cfg POSITIVELY
names wasm32; `workspace = true` entries resolve through the root
`[workspace.dependencies]` table (root-RELATIVE paths — the first draft
joined them against the member manifest and the canary caught it);
dev/build deps, optional deps, native-target tables, and cross-repo path
targets credit nothing. Seeds are named ∪ derived — a derived row's `-p`
list compiles its path-dep closure exactly like a literal one.

`--self-test` (five canaries in a throwaway git repo, sharing the ONE
`verdict_for` classifier with main — two copies would be the
two-parsers-disagree trap): named / by-dep / uncovered (the mmorpg-remake
`.issues/010` lineage the upgrade must NOT collapse) / optional-not-credited
/ workspace-table-resolved. All five hold; the workspace-table arm is the
one that caught the root-relative path bug.

Landing measurement (2026-09-14, post-774): **26 NAMED · 2 BY-DEP · 0
UNRESOLVED · 1 UNCOVERED** over 216 files / 29 packages / 20 repos (was
26/0/3 pre-upgrade; the 2026-09-08 standing of 23 NAMED / 0 UNCOVERED over
191 files / 23 packages was stale — siblings had added wasm32 surface).
The 1 remaining UNCOVERED is `mmorpg-poc-submodule` (mmorpg-remaster):
deliberately excluded from that repo's CI (`--exclude mmorpg-poc-submodule`,
needs protoc), depended on by nothing (it deps on mmorpg-core, not the
reverse) — the standing negative control for the new verdict, and an
arm-vs-row decision that stays with that repo's owner (read-only here).
Notes are sorted at print now too: the resolver's note sets iterate in
PYTHONHASHSEED order and an unsorted report is not diffable run-to-run —
cosmetic churn was masking real verdict flips in the landing diff.

## Issue 748 (2026-09-14, M3 session) resolved — option (a): all three unwired Lean negative tests now run in their lean_proofs.yml CI jobs (~162s/main push)

The issue's gap 2: three of the four Lean negative tests
(`proof_negative_test.sh`) were invoked by NOTHING — not even on `main`.
Only riir-chain ran both scripts in one job. katgpt-rs, riir-ai and
riir-neuron-db ran only `proof_gate.sh`, so in each the one artifact that
proves the gate is non-inert never executed in CI. The owner picked option
(a) — wire all three:

| repo | commit | measured marginal cost / main push |
|---|---|---|
| katgpt-rs | `3c97358c` | ~119s (Mathlib-backed, same job reuses the gate's `.lake`) |
| riir-ai | `ecb21f3f7` | ~37s (Mathlib-backed, same) |
| riir-neuron-db | `4a68575` | ~6s (Mathlib-free) |
| riir-chain | (already wired, Plan 016) | ~15s already paid |

Each wiring adds the script to BOTH `paths:` lists (so editing the harness
itself fires the lane) and a step AFTER `proof_gate.sh` in the SAME job —
the negative test reuses the `.lake` state the gate just built, which is why
the marginal cost is seconds, not a second Mathlib download. Two stale step
names fixed in passing (riir-ai "16 theorems" at an audited 22; riir-neuron-db
"34 theorems" at 58 — the exact step-name drift class this repo's own
workflow comment warns about), plus the three sentinel comments that claimed
a hand-run-only / no-lane life. Validated on the workstation before landing,
per repo: gate PASS (39 / 22 / 58 theorems) + negative test 8/8 / 15/15 /
17/17 + clean-tree rebuild + `.proofs` byte-clean after each.

Gap 1 (no lane covers `develop`) is UNCHANGED BY DESIGN — main-only triggers
are the owner's 2026-09-09 Actions-spending call, recorded in the issue and
still true: a green `lean_proofs.yml` badge says nothing about a develop
commit; the workstation run remains that coverage. The `workflow_dispatch`
button exists for naming a run on demand.

## Issue 773 (2026-09-14, M3 session) resolved — the 772-B2 removal's stale re-export: root lib E0432 under `flashar_consensus,plasma_path`, found by riir-ai's guard through the path dep

`3c3c52ce` (Issue 772 B2) deleted `ternary_fusion_gate` from
katgpt-forward but missed the root shim's `plasma_path`-gated re-export
(`src/speculative/flashar_consensus.rs:19`). The shim compiles only under
`flashar_consensus` (`src/speculative/mod.rs:324`, non-default) — the
double cfg is why the 772 wave's own default-lane validations never saw
it (the cfg-gated green-zero class, biting its own landing lane) while
riir-ai's guard Layer 1 DID: its default graph forwards both features
into the path-depped katgpt-rs root crate and died E0432 before checking
any riir-ai crate. No consumers anywhere (in-repo grep + six sibling
repos). Filed `1d873f8c`, fixed `b7fcabd8` (3-line deletion; red→green
under `--features flashar_consensus,plasma_path`, default lane
unchanged). The scoped-closeout meta-pattern's sixth instance this
bevy-lane week — and the first found cross-repo, by a DOWNSTREAM full
gate.

## Issue 771 (2026-09-14, M3 session) resolved — the radix-tree prefix KV cache primitive (RadixAttention index) shipped opt-in; G1–G4 ALL PASS, promotion deferred to the serving lane

**The question that opened it:** "do we have RadixAttention yet?" — No. The
stack had the PagedAttention half (`PagedKVCache` ref-counted pages,
`fork`/`rollback` CoW for spec-decode), the single-stream whole-prefix
cache (`riir-gpu` `Qwen38PrefixCache`, whose riir-ai Bench 750 note documents the
radix-tree divergence as "equivalent for single-stream reuse"), and the
unwired segment matcher (`KvSegmentPool`) — but not the composition.

**What shipped** (Bench 762, all gates green at
`--release --features radix_prefix_cache`):

- `katgpt_kv::radix_prefix::RadixPrefixTree` — the index half: chunk-
  granular spans (16 tokens), chunk-floor longest-prefix match (the
  trailing partial chunk is re-prefilled — this is what makes CoW
  unnecessary: a request never writes into a page another branch reads),
  leaf-preferential LRU with lock-aware eviction, in-place splits that keep
  node ids valid for lockers. The tree owns page INDICES, never buffers —
  CUDA-graph address stability by construction (the vLLM capture×prefix-
  cache corruption class is structurally impossible).
- The pool half: 4 ungated seam methods on `PagedKVCache`
  (`chunk_page_tables` / `retain_chunk_pages` / `release_chunk_pages` /
  `adopt_chunk_pages`) — pure ref-count mechanics. Discipline: pool
  refcount = live-seq holds + 1-while-tree-indexed; locks are hit-rate
  optimization, never safety.
- G1 bit-identity (via `to_bits` — the filler produces NaN payloads where
  float `!=` lies; micro passed by luck, small_target caught it), branch
  isolation (shared trunk pages identical, refcount exactly 3, no
  cross-branch leak), pool stability. G2: hit-rate **2.45×** the flat
  whole-prefix control at equal 50% budget (16 convs × 8 turns round-robin
  — the FIRST workload, sequential conversations, measured EQUAL because
  nothing revisits; the flat cache's duplication only hurts under
  interleaving + pressure, which is the honest RadixAttention workload
  class); match-only latency **9.8×** (0.21 vs 2.06 ms / 2,560 lookups,
  release). G4: 0 allocs on the match path (counting global allocator; the
  first draft counted its own query-construction allocs — warm the scratch,
  measure after).
- Divergences from SGLang documented in-module: chunk-floor matching, no
  per-chunk hash filter (memcmp is the authority and the filter costs more
  at 64 B/chunk), node-per-request over edge-extension, locks-on-head
  splits.

**Promotion verdict: STAYS OPT-IN** — G1–G4 pass and the gain is modelless,
but there is no production consumer: every inference lane is single-stream
(`riir-ai .research/034` recorded the radix tree N/A for single-stream; the 4090
lane's flat cache is the correct shape there). The `drift_segment`
precedent: GOAT PASS + consumers landed → promotion candidate. The
consumer note (T5) lives in `riir-gpu`'s `qwen38_prefix_cache.rs` module
doc — when a multi-request serving lane lands, consume this primitive, do
NOT re-derive a private tree.

**Follow-up hardening (2026-09-14, 4090 session — the racing duplicate's
additive tail):** a parallel session independently built the same primitive
(the dual-allocation landed first here — the pushed implementation won,
the duplicate dropped per the numbering-collision discipline). Its two
genuinely-new test classes were adapted onto the landed API and added:
`randomized_oracle_equivalence` (24 seeds × 40 random inserts × 80 mutated
queries vs the brute-force floor(lcp/page_tokens) oracle — the class no
hand-shaped test enumerates) and the `G2[width]` gate (78 → 186 ns/req at
exactly 8× nodes, ≤3× per-request — the depth-not-width law, gated
per-request after the first draft compared raw round totals across 8×
request counts and false-REDA; violation mode is O(width) ≈ 8×, so 3.0
leaves ~2.6× headroom over the measured 2.4× arena-locality effect).

## Issue 770 (2026-09-13, M3 session) resolved — the counter walker rebuilt per-commit; the 769 adjudication was partly an instrument artifact (verdict-review round 2)

The verdict reviewer re-derived every landed reset row against its commit's
OWN diff and found the 769 walker lineage-blind — two structural defects,
both now regression-fixture-pinned:

- **15 of the 31 rows named a FORWARD-stepping commit.** The walker ordered
  ALL refs' hunks by commit DATE around one `current`; after any real
  backward event, every ordinary `+1` on the OTHER lineage landed below the
  walk's base and was re-flagged. riir-chain's entire ×4 and riir-train
  Issue's ×2 were phantoms; riir-train Bench's 8 was really 2.
- **"All 31 non-merge" was true BY CONSTRUCTION.** `git log -p` emits no
  diff for merge commits, so a merge resolving a counter conflict by taking
  the LOWER side — the dangerous case — was invisible while an innocent
  concurrent bump took the blame. The T2 adjudication checked a hypothesis
  the instrument was structurally incapable of falsifying.

The repair (`highwater_contiguity_audit.py::counter_history`): every
HEAD-reachable commit's counter blob is read at the commit AND at each
parent (one `git cat-file --batch`; the response stream's blank separator
after each blob had to be consumed explicitly — skipping it desynced half
the reads, caught by the fixtures), and the event is judged against
`max(parent values)` only. Merge commits are ordinary rows: riir-ai Issue
`627→614` at a Merge commit and mmorpg-editor Plan `152→150` ×3 at
merges are now visible and correctly blamed. Corrected count: **27 resets
workspace-wide** (katgpt-rs 5, riir-ai 9, riir-clippy 5, riir-train 2,
mmorpg-editor 4, riir-mmorpg-examples 1, riir-shader 1; riir-chain 0).
The floors re-pinned from the corrected walk (format 5→6 fields; the
7-field max_unbumped column lasted one landing — the worktree-below-history
quantity is CHECKOUT state (mmorpg-editor's bevy worktree vs other refs'
194), ref-set dependent, and is now REPORT-ONLY, printed like the citation
sweep's undecided rows, never gated). The walk is HEAD-reachable only,
never `--all`, for the same reproducibility reason. The selftest's inert
in-flight arm (the stub was restored before `got4` ran — mutation-proven
by the reviewer) is fixed; the stub now spans both arms. And the sweep
caught THIS session's own highwater write-back miss (filed 770, forgot the
769→770 bump — the stale-allocator class red on cue). The Issue-768
measured-negative verdict STANDS — the reviewer's findings only strengthen
it (the old walker was even less able to certify contiguity than its report
claimed); its figures re-measured under the corrected walk: 438 gaps + 27
resets over 73 counters (creation gaps `0→N` now counted honestly where the
old first-hunk special case suppressed them).

## Issue 769 (2026-09-13, M3 session) resolved — the counter-reset class lands in the numbering sweep; all 31 measured resets adjudicated

*(Correction 2026-09-13, same session, verdict-review round 2 — read with
Issue 770: the counts and the adjudication below are the DATE-ORDERED
walker's. 15 of the 31 rows were phantoms naming forward-stepping commits;
the "all non-merge" partition was true by construction (`git log -p` never
diffs merges — the merge-resolution resets were invisible, and at least one
row blamed an innocent concurrent bump). The corrected per-commit count is
27, merge resets included; the ratchet doctrine survived with honest
numbers. The class itself, the pin plumbing, and the DRY single-walker
import all stand.)*

Filed and resolved the same session, out of Issue 768's T3 measurement.
`numbering_drift_sweep.py` gains two verdict classes over `.highwater`
HISTORY (the walker imported from `highwater_contiguity_audit.py`, not
re-implemented):

- **resets** — committed BACKWARD moves. After `517→511` the re-climb
  re-spends numbers: the never-reuse rule broken at counter granularity, the
  `.issues/121` collision class reborn. **T2 adjudication: all 31 measured
  resets workspace-wide are NON-MERGE stale-lineage writebacks** — a
  diverged-checkout session committing its local counter over a newer
  mainline value (katgpt-rs Bench `564→204`/`564→205` — a long-diverged
  lane allocating 204/205 while main sat at 564; riir-train Bench ×8; katgpt-rs
  Issue `577→25` — a worktree branch from base 24). The historical hazard
  was absorbed by gap fast-forwards and push-wins renumbers; the max_dup
  column proves no live file duplicates. Pinned at measured (ratchet): a NEW
  reset reds at its pin.
- **unbumped** — the WORKTREE counter below its committed history max
  (a checkout state, not a commit): mmorpg-editor ×3 (150<152, 191<194,
  1<2) — REPORT-to-owner class, the repo is read-only to these sessions.

Pin format 5→7 fields (`max_resets`, `max_unbumped`); the selftest grew a
synthetic diverged-lineage fixture that caught a REAL walker defect on
landing — a `4→3` landing when the walk sat at 2 read as a climb to 3,
because the walker ignored the diverged lineage's own `old` (numbers up to
it were spent THERE); `base = max(current, old)` now, and 17 rows moved
from the gap bucket to their true hazard classes (332→315 gaps). The
instrument proving itself before it shipped is the whole point of the
fixture.

## Issue 768 (2026-09-13, M3 session) resolved — the .highwater ownership witness REFUTED by measurement; highwater_contiguity_audit.py landed

Filed the same day from the Issue-766 mis-repair arc; closed MEASURED-NEGATIVE
in one session. T3 asked whether `.highwater` is contiguous enough to serve
as a fourth ownership witness in `allocated()` ("n ≤ hw ⇒ n was allocated
here" — the counter as allocation ledger). The landed
`scripts/highwater_contiguity_audit.py` (report-only, selftested,
population-derived; two views — the static unwitnessed count, and the
counter's own committed transition walk) answers NO, workspace-wide:

- **438 gaps + 27 resets over 73 counters** under the Issue-770 per-commit
  walk (the figures at first landing were 332 gaps + 31 resets from the
  date-ordered walker — 15 of its reset rows were phantoms and its merge
  resets were invisible; re-measured after the repair, with creation gaps
  `0→N` now counted where the old first-hunk special case suppressed them) —
  no major repo's counter is contiguous. katgpt-rs Issue: 19 gaps + a
  `577→25` lineage reset; riir-ai
  Issue: 38 gaps + 2 resets; riir-neuron-db Issue jumped `33→589` (a
  deliberate rebaseline — 34..588 were never allocated there);
  mmorpg-editor/Plan carries 46 gaps; riir-shader's Issue counter moved
  `11→9`. riir-auth's `.benchmarks` counter is COUNT-BASED by its own
  AGENTS.md (records, not numbers) — excluded from witness semantics by
  design.
- The blanket witness would therefore CLAIM never-allocated numbers — and an
  ownership witness that over-claims does not merely miss findings, it
  VALIDATES wrong addresses: the exact failure class (766/4573af13) inverted.
- A maximal-contiguous-suffix witness is derivable but unpredictable
  (coverage starts after each counter's LAST gap/reset, wherever history put
  it) and still carries the T2 dual-allocation false-validation risk; T1
  measured ZERO live rows the witness would change (the corpus sits at 0
  CROSS after the same-day repairs). Benefit ≈ 0 today, cost real, soundness
  refuted → **declined. Decline is a correct answer.**

The 766-class protection stands on the already-landed repair instead: the
canonical `## Issue NNN (date)` heading form IS the allocation record
`heading_allocated()` reads (Issue 754). Standing counter-hygiene findings
from the audit (the resets are number-reuse-shaped; mmorpg-editor's
worktree counter sits BELOW its committed history max) are re-runnable per
repo via the script; the reset class as a NUMBERING sweep check is filed as
Issue 769.

## The all-features E0252 root-name collision — ooo_audit::AuditScratch aliased (2026-09-13, M3 idle sweep)

Found by the idle clippy sweep (`cargo clippy --workspace --all-targets
--all-features`): E0252 — the crate root re-exported the name `AuditScratch`
twice, once per opt-in feature (`latent_confounder_audit`, Bench 194, the
older resident; `direction_bank_audit`, landed `83518b30` 09-12, the
newcomer). Both-features-on is the only state that collides: the landing
lane's single-feature validation had `latent_confounder_audit` off, and CI
is main-only since 09-09, so develop carried the break unverified — the
dual of the documented "non-default gated code compiles to nothing" blind
spot (here even the all-features lane broke, because TWO opt-in features
collide only in combination). Fixed at the newcomer's re-export
(`a0ca7d36`): `AuditScratch as OooAuditScratch` at the crate root only;
module paths unchanged — zero root-path consumers existed (both benches,
riir-poc, and riir-clippy score_bench all import via module paths).
Validation: default unaffected (`direction_bank_audit = []`, the edited
block cfg-off); `-p katgpt-core --features direction_bank_audit` clippy
clean + 2051/2051 lib tests; workspace `--all-features --all-targets`
clippy clean (only the upstream `block v0.1.6` future-compat note,
shared with riir-shader's tree).

## The Windows default-features workspace lane — bench_mtp_metal_batch_floor platform gate (2026-09-13, 4090 session)

Found by the idle clippy sweep (`cargo clippy --workspace --all-targets` at
DEFAULT features): 13×E0433 — the only Metal example without
`target_os = "macos"` guards. The lane chain: root `default` →
`async_qdq_overlap` → `inference_router` → `gpu_inference` forwards into
katgpt-backend, satisfying the example's `required-features` on EVERY
platform — so the example compiled on Windows where `use metal::*` cannot
resolve (the dep is macOS-target-gated). The M3 lane never sees it; the
heal commit `2cb97410` had already recorded it as "workspace E0433 metal
example pre-existing (proven at HEAD 00cfe345)" and left it. Fixed with
the sibling pattern (`bench_439_*`): item-level cfg on all 22 top-level
items + a loud `not(macos)` main stub ("requires macOS with Metal"), file
rustfmt'd (it had never been formatted). Verified on Windows:
`-p katgpt-backend --all-targets --features gpu_inference` clippy-clean,
the stub prints, and the workspace default-features lane is 0-warnings
for the first time on this box.

## Issue 766 (2026-09-13, 4090 session) resolved — len_derived audit: caller tracer (HALF C) + two instrument defects found and fixed

Follow-up to riir-train Issue 515's standing work list (220 UNRESOLVED
wrapper-param binds). Three additions to
`scripts/len_derived_binding_audit.py`, each shrinking the honest residue —
and the first two are INSTRUMENT DEFECTS the backlog surfaced:

- **HALF A window fix (the big one)**: the fixed 6000-char body window bled
  into whatever followed a kernel. Measured: the deltanet tree-verify
  kernels were flagged off `parent.len()` in the HOST-side
  `TreeVerifyPlan::from_parents_topo` below them — while deriving every dim
  from `params` (the correct pattern). 22 of 74 kernels were such bleed
  false positives; 89 of 253 bind rows were phantom (including ALL 8
  PERSISTENT rows T2-of-515 had triaged "benign" — moot, never
  `.len()` kernels). Body window is now the brace-matched fn body;
  `min_kernels` floor re-pinned 40 → 45 (52 measured).
- **HALF C caller tracer (the backlog item)**: for path-form wrappers
  (`impl Struct { fn launch(params) }`, no self), every workspace caller
  `Struct::fn::<T>(args)` is collected (turbofish-tolerant, comment-stripped
  — inline `// [0..n]` comments after an arg's comma were riding into the
  NEXT arg and breaking field detection, measured on the Split4 call) and
  the (handle, length) param pair classified per site: EXACT-UPSTREAM
  (`create_from_slice(&v)`+`v.len()`, `empty(k)`+`k`), TRIMMED-UPSTREAM
  (slice views), PERSISTENT-UPSTREAM (struct-field handles — the eyes
  list), CAPACITY-UPSTREAM (size-method/capacity-constant lengths — the
  compact_temp join one level up). Clean requires ALL workspace callers
  clean; bare-name/method-form callers stay UNRESOLVED (unsound by name).
  Selftest pins include the false-clean canaries (a field handle NEVER
  resolves exact; a call result is never a field handle).
- **GUARD-ONLY class**: a len-use that only bounds thread indices (`let n =
  output.len(); if tid < n`) is capacity-tolerant — the compact hazard
  needs STRUCTURAL derivation feeding index math. 14 rows (copy/fill/
  lm_head_lora training kernels) re-verdict GUARD-ONLY automatically.

Standing after: **52 kernels, 164 binds** — 25 GUARDED, 14 GUARD-ONLY,
3 EXACT-UPSTREAM (verified true), 4 PERSISTENT-UPSTREAM adjudicated BENIGN
  by allocation-site reads (gemma2 F16 weights + ternary down_proj/ffn/x —
  all exactly-sized at construction; structural derivation recovers true
dims), **118 UNRESOLVED — the honest floor** (mixed-caller sets,
pass-through params, method-form wrappers; a work list, never a defect
set). T4 N/A: no capacity-vs-live mismatch found, no new guards needed.

*Heading correction (2026-09-13, M3 session): this row's heading first
landed bare (`## Issue 766 resolved — …`) and 4573af13 then "qualified" it
to riir-ai — a mis-repair, caught while clearing riir-train's Issue-515
citation rows. The allocation witness was in the fix commit all along:
`e4792a4b` bumps `.issues/.highwater` 765→766 (file-and-resolve, the issue
file never committed — the Issue 754 invisible class), so katgpt-rs owns
766. riir-ai's own 766 is the August 4090 holding queue — a different
issue entirely (the dual-allocation class; cf. `Issue 665` in riir-clippy).
riir-train's addendum "katgpt-rs Issue 766 resolved the standing list" was
correct as written and stands. The heading is now in the `## Issue NNN
(date)` form `heading_allocated()` reads (Issue 754), so the citation
instrument sees the allocation the highwater already witnessed. Residual
gap — `allocated()` never consults `.highwater`, so a highwater-only
allocation with no readable heading stays invisible — filed as Issue 768
with the measured-widening plan (the suppression direction must be measured
before it is widened).*

## WeightEpoch — the KV-cache weight-identity epoch (riir-ai Issue 938; Plan 025 contract) (`49f5d245`, 2026-09-13)

`LoraAdapter::weight_epoch() -> WeightEpoch` (katgpt-types `lora` module):
BLAKE3 over a domain-separated canonical serialization (tag
`katgpt-lora-weight-epoch-v1`, rank/in_dim/out_dim/alpha + length-prefixed
a/b bytes). Identical adapters share an epoch — an A→B→A round trip may
keep using its cache (identity, not installation counter); any byte
difference defines a new epoch. `WeightEpoch::none()` for the no-adapter
state. O(adapter) at swap time, O(1) memcmp, zero hot-path cost.

Motivation: the RLT §5.4/App-C cache-staleness law (riir-train
`.research/453`; riir-ai Issue 938) — a KV-cache entry is exact only when
computed under the SAME weight epoch it is read with. `LoraPair` now
documents its BY-DESIGN mixed-epoch acceptance (reader-prefill /
writer-decode sharing one cache) at the type that creates it, and
`examples/core_04_prefill.rs` Proof 3 prints the reader/writer epochs + the
documented-acceptance line at the switch site. riir-ai consumes the type at
its `CpuInferenceBackend` seam (refusing mixed-epoch reads loudly) — record:
riir-ai HISTORY §Issue-938.

## Modelless-first mandate — original section (incl. the canonical-failure story)

## Modelless-first mandate (the core principle)

**This repo ships modelless inference primitives.** No training, no backprop,
no gradient descent. The only weight mutations allowed at runtime are:

1. **Freeze/thaw** — swapping a frozen snapshot (atomic, versioned, BLAKE3-checked).
2. **Raw/lora hot-swap** — applying a **deterministically constructed** (not
   trained) LoRA overlay via `LoraPair { reader, writer }` (Plan 025).
3. **Latent-space updates** — direction-vector projections, sigmoid gates,
   routing tables. These update latent state, NOT base weights.

### MANDATORY: exhaust modelless paths before deferring to riir-train

Before deferring ANY gate, mechanism, or plan task to riir-train ("this needs
training"), you MUST check whether the three modelless paths above can fix it.
See the research skill §3.5 (`.agents/skills/research/SKILL.md`) for the full
decision protocol.

**Systematic, characterizable biases are modelless-correctable candidates,
NOT automatic riir-train dependencies.** If a gate fails because of a known,
named bias (e.g., "signal doubled", "position offset", "attention asymmetry"),
check whether a deterministically constructed reader-LoRA or freeze-state
correction can fix it before concluding "needs gradient descent."

**Canonical failure — AC-Prefix G1 (Plan 313, 2026-06-24):** G1 was prematurely
deferred to riir-train without checking whether the doubled-signal bias could
be corrected modellessly via a deterministic reader-LoRA. The bias was
systematic and characterizable — exactly the case where raw/lora hot-swap
might work. The deferral was premature and has been reverted; the modelless
investigation (Issue 003, resolved-and-removed in commit `552b4632`) is
captured in `.benchmarks/313_ac_prefix_modelless.md` (Path 2: `attends_dedup`
eliminates the bias bit-identically to iterative-MLM on single-layer
micro-GPT, 0.0 diff). `ac_prefix` re-promoted to DEFAULT-ON on that
modelless pass; multi-layer equivalence remains a non-blocking riir-train
follow-up.

## Boundary contract — original section

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative per-repo contract: what this
repo **owns**, what it **does not own** (with the correct home for each), the
crate-granular **allowlist** of what it may depend on, links to the cross-repo
rules' one canonical home, and the **drift ledger** of known gaps. On any
conflict with prose in this file, BOUNDARY.md wins.

- **Domain test:** is this a **modelless inference primitive** with no riir dep (this repo is upstream of everything)? NO → it belongs in another repo; file there.
- **Read it before** adding any dep, crate, module, System impl, or vocabulary
  type — and before assuming a concern is yours to implement.
- **Enforcement** is not prose: `../riir-ai/scripts/ci_boundary_contract.sh`
  fails on an undeclared cross-repo dep, on a drift row without its open issue,
  and on a contract row that no longer matches the measured graph. Run boundary
  checks VIA the `boundary-guard` skill, not as ad-hoc greps.
- **Found a violation?** File the issue FIRST (`.issues/NNN_boundary_*.md`), add
  the drift row, then fix. Closing the issue removes the row in the same commit.

## The full gate — original section (narratives)

### The full gate — none of the above is a whole-repo claim

Every command listed above is narrow in at least one **independent** axis, and
a green result says nothing about what it compiled to nothing. (The count is
deliberately not written here — this sentence said "three" for months while
the table below carried five, which is the drift the table exists to catch,
committed by the sentence introducing it.)

| Axis | Blind spot |
|---|---|
| `check` vs `clippy` | two `cargo heal` escape classes are rejected by clippy's typeck and accepted by `check` (E0689 ambiguous-integer, E0631 deref-coercion in `redundant_closure`) |
| default vs `--all-features` | non-default gated code compiles to **nothing** |
| `-p <crate>` vs `--workspace` | *at the same default features*: a crate's own non-default feature can be switched on by the ROOT crate's defaults once the root is in the selected set |
| no `--all-targets` | skips every test / bench / example — which is where gated code lives |
| dev vs `--release` | `debug_assertions` is always **ON**, so every item behind `#[cfg(debug_assertions)]` — and everything that depends on one — is only ever compiled in the configuration where it works (`.docs/10_audits/debug_release_profile_axis.md`) |
| `--all-targets` vs **doc-tests** | `--all-targets` does **not** include doc-tests — so the gate whose whole point is to compile everything never compiled a single doc example. Measured 2026-09-04 by the first full-workspace *execution*: **8 crates' doctests had never been built at any revision**, 31 lines across 14 files still writing `use katgpt_rs::...` after the root crate was split into sub-crates that must not depend on it, plus 3 independent defects — including one example asserting values its own formula cannot produce. Only `cargo test --doc` reaches this (`.issues/723` Class F) |
| **compile vs EXECUTE** | every axis above is about *compilation*. **The scoped core is now EXECUTED weekly** (`test.yml` + `scripts/test_gate.sh`, Issue 718 T3(b) landed 2026-09-04: katgpt-rs + katgpt-core `--lib` at default features with count floors, the riir-train 507 shape) — but that is 2 of 32 packages' lib suites. The other 477 integration-test targets and 176 bench targets over 32 packages remain executed by nothing automatic (`.docs/10_audits/ci_compile_vs_execute_axis.md` — the full-workspace `--all-features --release` run is now PRICED: 11,542 CPU-s cold / 45.5 min wall / 497 ok + 45 FAILED across 39 targets, and the finding is that `--all-features` is not a supported TEST configuration — fixture RNG streams and GOAT calibrations are per-feature; the full run needs per-target triage pins before it can gate anything — cost table in `.benchmarks/701_full_workspace_execution_pricing.md`, the 45 reds in six classes in `.issues/723`) |

**The last axis is the one that changes how to read every gate below.** A
green full gate is a claim that the workspace *compiles* under one feature
set on one platform in one profile — never that an assertion in it holds.
This repo's own rule is that an uninvoked assertion is *unknown*, not
passing, so by that standard every Rust assertion here is unknown: the 39
GOAT gates armed with `required-features` in Issue 713 T3 included, because
arming made a **named** run honest and nothing names them. AGENTS.md records
"All 39 pass there" under `--release` — that was a **workstation** run, and
nothing repeats it. The scoped core (2 of 32 packages' lib suites) now has a
scheduled answer via `test.yml`; everything outside it is still
unknown-by-default, and what is *not* optional is reading a green gate as
nothing more than "it built" — because that is all it ever said.

The `-p` vs `--workspace` axis is the least obvious. `cargo test -p katgpt-backend --lib`
compiled clean while `cargo test --workspace --lib` failed, because `gpu.rs` is
behind `katgpt-backend/gpu_inference` and the chain
`katgpt-rs/default -> async_qdq_overlap -> inference_router -> gpu_inference`
only fires when the root crate is selected. It also silently *shrinks* coverage:
four crates reporting "0 tests" per-crate contributed 704 under `--workspace`.

The **fifth** row is the newest and the command below does **not** close it —
it runs in the dev profile. Measured 2026-09-03: adding `--release` produced
**2 errors** and `cargo test --release -p katgpt-core --lib` did not compile at
all, which is the very command `.docs/10_audits/cfg_gated_silent_zero_pass.md` T2b tells everyone to use. Two
`#[cfg(test)]` blocks imported `crate::alloc`'s counters, which are
`debug_assertions`-only *by design*. Fixed in `.docs/10_audits/debug_release_profile_axis.md` T1; the axis itself
is T2.

Read that together with `.docs/10_audits/cfg_gated_silent_zero_pass.md` T2b and `.docs/10_audits/debug_release_profile_axis.md`, because the three
point in different directions and that is the lesson: debug **manufactured**
four false perf reds (713), debug **hid** a two-day release build break (715),
and the full gate compiles `debug_assertions` code only in the profile where it
works (716). **Neither profile is the safe default — the profile is part of the
claim.**

So before claiming a repo-wide green, run:

```bash
cargo clippy --workspace --all-targets --all-features --keep-going -- -D clippy::needless_range_loop -D clippy::map_clone -D clippy::iter_cloned_collect -D clippy::identity_op -D clippy::bool_comparison -D clippy::manual_is_multiple_of -D clippy::collapsible_if -D clippy::map_all_any_identity -D clippy::unnecessary_cast -D clippy::manual_repeat_n -D clippy::question_mark -D clippy::empty_line_after_outer_attr -D clippy::unusual_byte_groupings -D unused_mut -D unused_parens
```

The `-D` list (Issue 701 R3b, 2026-09-03) is the mechanical lints whose
all-features warning surface was healed to ZERO residual (67 → 13 distinct
findings; the 13 survivors are judgement-class and stay warnings), so a
regression now reds the gate instead of silently re-growing the ungated
warning surface. A lint with residual > 0 must NOT be added to it.

`--keep-going` is not optional — without it the run stops at the first failing
target and under-reports. This gate was **red on `develop` from at least
2cb97410 until `c284dbb2`/this commit** (5 broken targets) while every gate in
the block above was green. Treat a green gate as a claim about its literal
command, not about the code.

It happened AGAIN, 2026-09-07: red from at least `c69e651d` (Issue 731 T1, the
`forward_looped` `residual_exit` param) until `c571d5b9` — **32 compile
errors** this time (28 × E0061, a call site missing the new trailing param;
plus 2 × E0308 and 2 × E0614 from Issue-729 comparator stragglers riding the
same run, fixed in `26ba3519`). The mechanism is a sharpening of the same
lesson: T1's landing note recorded "all 27 construction sites aligned", and
it was TRUE — 27 sites were aligned — while 28 MORE call sites existed in
`#![cfg]`-gated test targets that compile only under `--all-features`, a set
no narrower gate names and no grep-for-callers distinguishes (the aligned 27
and the missed 28 are textually identical call shapes). The sites were found
by rustc, not by review, exactly as designed. Corollary recorded: an
alignment pass's completeness claim must state its POPULATION FRAME — "all
sites in targets compilable at the feature states I ran" — or it is a claim
about a subset with the grammar of a whole. The T3 bench target itself was
also auto-discovered with no `[[test]]` row (SILENT-NOW +1, caught by
`cfg_gated_floor_gate.py` the same day — its row landed with the fix), and
`docs_gate.sh` had not run on the branch either; both gates were green again
by `26ba3519`.

Don't run it by hand — `scripts/full_gate.sh` is the assertion (it also refuses
to report a pass off macOS, where the `target_os = "macos"` device backends
compile to nothing even with `--all-features`, and checks that this document
still quotes the command it runs).

**And the inverse holds, with nothing to enforce it.** That caveat protects the
`target_os = "macos"` backends by refusing to *report* off macOS. Running **on**
macOS silently drops every `not(target_os = "macos")` backend, `--all-features`
included — so the command above, which is run on the M3 by policy, is
structurally incapable of compiling them. Measured 2026-09-03 by parsing
`riir-gpu/src/lib.rs`'s module **declarations** (a file can mention the cfg
without being gated on it): **9 modules, 25,212 lines**, headed by
`qwen38_dense_cudarc` at 8,599. It is not theoretical — `riir-ai` `6bf51b592`
landed a CUDA-only lib that did not compile (`E0599`) and it stood for **7h45m**
until a 4090 re-pin happened to need a CUDA build; nothing else in the workspace
builds that code. Record and the open axis: `riir-ai` `.issues/857`.

This is the same shape as `.docs/10_audits/cfg_gated_silent_zero_pass.md` one axis over — there a
`#![cfg]`-gated test compiles to an empty binary and reports a green zero; here
a `cfg(not(target_os))` module compiles to nothing and reports a green build.
**A platform is part of the claim, exactly as the profile is.**

**As of 2026-09-04 the compile half of that axis is REACHABLE from the M3**, so
"we cannot build that code here" is no longer the answer:

```bash
scripts/check_platform_gated_modules.sh ../riir-train riir-train-gpu numeric_drift_cuda
scripts/check_platform_gated_modules.sh --canary ../riir-train riir-train-gpu \
    crates/riir-train-gpu/src/numeric_drift_tap.rs numeric_drift_cuda
```

`cargo check` never links — it needs only rust-std for the target and build
scripts that exit 0. Three build scripts stand in the way and none needs a real
cross toolchain: `blake3`'s NEON C (answered by `CARGO_FEATURE_NO_NEON=1`, the
pure-Rust path) and `libsqlite3-sys` + `sentencepiece-sys` (answered by the
**Android NDK's clang**, already installed here, which ships a complete linux
sysroot — the macOS SDK cannot, because `sys/cdefs.h` answers a linux-gnu
target with `#error Unsupported architecture`). Two details cost an hour each
and are in the script's header: cc-rs injects its own `--target`, so the shim's
must come **after** `"$@"` or clang honours cc-rs's and loses the sysroot; and
`-llog` is required because sentencepiece's cmake build links helper binaries
that need `__android_log_write`.

Read a green run narrowly: it is the **compile** half only, nothing ran, and
much of that CUDA code has never executed anywhere. But a green run is exactly
what `riir-ai` `6bf51b592` did not have when it shipped a CUDA-only lib that
did not compile and stood 7h45m. First use (riir-train `53538538`) typechecked
two `not(target_os = "macos")` modules whose five edited call sites the issue
had routed to "whoever next builds on the 4090". **`--canary` is not optional**
— it plants an undefined call inside the gated module and requires `E0425`,
because otherwise "Finished" is indistinguishable from the modules compiling
to nothing again, which is the entire failure this section describes. `.github/workflows/full_gate.yml` **declares** a
weekly cron, a manual dispatch, and a NARROW per-push/PR lane — the trigger is
real but fires only when the gate's own definition changes
(`scripts/full_gate.sh` / the workflow file itself; measured 2m17s made that
affordable, broadening to `**/*.rs` remains rejected — see the file's preamble
for the cost story and the promotion criterion). An ordinary code push does not
gate here; the rot check is the Monday cron. That
preamble also carries the liveness-sentinel record from `.issues/705` — the
gate's first two CI runs passed over ZERO compiled units (ANSI color codes
defeated every `^`-anchored counter, including the error count); closed +
removed 2026-09-02, full narrative in git history.

That was a declaration and not a schedule until 2026-09-01. `schedule` and
`workflow_dispatch` run **only from a repository's DEFAULT branch**, and this
repo's default was `main` — frozen at the v0.1.1 promote with no
`.github/workflows/` at all — so **neither trigger had ever fired**. The file's
own comment calls the schedule "the rot check"; the rot check had rotted, and
this paragraph advertised it as running. Fixed by moving the default branch to
`develop` (`.issues/704`), which is where AGENTS.md already says work lands;
both triggers are live as of that change.

Don't take that as permanently settled — a workflow file is identical on disk
whether or not it can execute, which is why this went unnoticed. The axis is
now measured: `scripts/ci_gate_coverage.py` reports, per workflow, which
declared triggers can actually fire, keeping **dead**, **unmeasured** (no remote
refs), **untracked** (committed by nobody yet — a colleague's in-flight file is
not a defect) and **PR-only** apart. It took the workspace from 7 dead workflows
to 1. Run it rather than re-reading trigger blocks by hand.

"Can fire" is still not "does fire". A workflow reachable only by
`workflow_dispatch` is a button, not a schedule, and three sibling repos
(`riir-chain`, `riir-dao`, `riir-neuron-db`) carried their whole Rust
compile/lint surface in exactly such a file — each by a *documented* main-only
owner call whose `push` is inert anyway, because `main` carries no copy of the
workflow. The report crosses the two axes rather than printing them side by
side, which is how that state stayed invisible: the coverage table credited the
command and the reachability table listed the trigger, and nothing multiplied
them together. RESOLVED 2026-09-02 (`.issues/706` closed + removed) — all
three now carry the `riir-clippy`-shape weekly `schedule`, the one trigger
that fires from the default branch while `main` stays frozen, with the
no-develop-push owner call untouched (`riir-chain` `b4a9b6e7` Tue 04:13 UTC,
`riir-neuron-db` `9d041d1` 04:29, `riir-dao` `9848811` 04:43 — whose workflow
also stopped hand-mirroring its guard layers and runs
`scripts/ci_feature_guard.sh`). The dormancy was not hypothetical: the same
the same day, `riir-neuron-db`'s standalone-dep gate was found RED nine days stale
(`29af2b0` changed the katgpt-rs patch set to `katgpt-device-verify` without
re-pinning `EXPECTED`; fixed `97e5161`) — invisible for exactly this reason,
because nothing ran the gate.

**Issue 758 (2026-09-12, closed same day): the profile axis had a feature-shaped
hole.** Layer 6 runs `--all-features`, which SUPPLIES `alloc_tracking` — so the
(release × **default features**) cell of the matrix was asserted by nothing:
(dev, default) = test_gate, (dev, all) = Layer 3, (release, all) = Layer 6,
(release, default) = nobody. Phase 31's slice_tca landed a **module-level**
unconditional `use crate::alloc::{…}` in `tests.rs` (the intent was the
Issue-741-blessed "never debug_assertions-only"; the mechanism was wrong) and
`cargo test --release -p katgpt-core --lib` went E0432 — while Layer 6 stayed
GREEN, because `--all-features` had turned the very feature on whose absence
broke the build. Found by Issue 757's release profile harness dying before any
linking code compiled. Fixed T1(a)-shaped with the FULL Issue-741 predicate —
`#[cfg(any(debug_assertions, feature = "alloc_tracking"))]` + in-fn `use` (the
option-(b) shape, `slice_tca = […, "alloc_tracking"]`, was REJECTED: it would
transitively default `alloc_tracking` on for every consumer, violating the
twice-documented "MUST stay opt-in" contract in both Cargo.tomls). Verified in
all three arms: release-default 2027 passed / dev 2035 passed (the 757 baseline
unchanged) / `--release --features alloc_tracking` the gated test RUNS (1
passed, not a green zero). The lane is **full_gate Layer 6b** — the test_gate
population (katgpt-rs, katgpt-core, katgpt-dec@pca_global) at
`cargo check --tests --release`, NOT `--workspace`: a workspace run inherits
the platform axis (katgpt-backend's metal examples are unresolvable off macOS
at ANY feature set — measured, E0433 ×10+ on Windows), which Layer 2 owns.
Canaried two-sided: the 758 import reintroduced → the katgpt-core row reds with
`could not compile (lib test)`; restored → clean. Landing 6b on a Windows box
also caught a latent GNU-portability break in the gate itself: all three
`mktemp -t <prefix>` calls (Layers 3, 6, 6b) are BSD-only — GNU mktemp rejects
a template with no X's — and the gate had simply never run anywhere but macOS;
now the full-path-with-X's form, portable both ways.

## Docs gate — original section (descriptions + narratives)

### The docs gate — same discipline, opposite cadence

`scripts/docs_gate.sh` runs the manifest/doc/skill drift assertions and
`.github/workflows/docs_gate.yml` runs it **per-push** on ubuntu-latest. Both
choices are deliberately the inverse of the full gate's, and both files say why:
this gate has no `cfg(target_os)` surface so platform cannot change its verdict,
and it costs ~11s rather than >13 min.

That number was **~3s when this sentence was written and is now ~11s**, and the
correction is worth more than the figure: `percentile_floor_gate.py` alone is
**7.0s of it** — it walks all 2,330 `.rs` files and tokenizes them — and it has
been since it landed on 2026-09-03, so this line was stale for three days while
reading as current. Measured 2026-09-06, per check: percentile 7.01s ·
bench_doc_audit 0.88 · cargo_comment_audit 0.86 · count_features 0.81 ·
orphaned_attr 0.62 · cfg_gated_floor 0.52 · everything else ≤0.10s. Re-time it
before quoting it; a cost in prose is a claim, exactly like a count.

Per-push is scoped to **`main` only** (owner call 2026-09-03, was
`[main, develop]`): develop pushes no longer fire the gate, so the
introduce-commit catch now applies only to the promote lane — run
`./scripts/docs_gate.sh` locally for develop work, or the drift surfaces at the
next main push. The same change fast-forwarded `main` to the develop tip,
because a push trigger reads the workflow file from the PUSHED ref and a `main`
without this file would be a dead trigger — the exact shape of the pre-704 rot
check.

The `CHECKS` array in that script is the list. **The count is deliberately not
written here** — this paragraph said "the three" for one commit after the fourth
was added, which is the drift the gate itself exists to catch, committed by the
paragraph describing it.

The original three existed before that wiring and **nothing invoked any of
them**; two were red on `develop`, both on false positives against docs that
were correct. Treat an uninvoked assertion as unknown, not as passing.

Some of the checks are worth knowing about specifically (no count here — this
paragraph said "the three" for one commit after the fourth was added):

- `skill_repo_set_gate.py` (Issue 703) fails on a `SKILL.md` command block that
  types the repo set by hand instead of deriving it. It reads sibling repos,
  which CI does not have, so it separates its **vocabulary** (committed
  `scripts/repo_set.txt`, re-derived and failed-on-drift by every workstation
  run) from its **population** (12 `SKILL.md` locally, 8 in CI) and prints both.
  A gate that skipped in CI instead would be the vacuous green it exists to
  catch. Mark a deliberately narrow block `<!-- repo-set-ok: <reason> -->`.
- `agents_repo_set_gate.py` pins §"Repo count" above against
  `scripts/repo_set.txt` — membership FIRST, cardinality second. It exists
  because on 2026-09-03 that paragraph named a retired repo
  (`riir-armageddon`) and omitted a new one (`mmorpg-remake-unity`) **while its
  count stayed correct**: one left, one arrived, total unchanged at 19. Every
  count in sight agreed and the set was wrong anyway, which is why the
  paragraph's own warning ("read a count in prose as a claim, not a fact") was
  not enough — a count is not a checksum over a set. It gates ONE paragraph on
  purpose: the obvious whole-repo version false-positives on history, and
  boundary-guard's ledger *should* still name the retired repo, because the
  227→225 edge delta IS that repo's two edges leaving. Both its inputs are
  committed, so it runs in CI; `repo_set.txt`'s freshness against the real
  workspace is a separate, workstation-only concern owned by the check above.
  A parser regression exits **2**, not 1 — an untrustworthy instrument is not
  the same finding as drift.

- `cfg_gated_floor_gate.py` (Issue 713 T4) is the GATE over the report below,
  katgpt-rs-scoped, with its pins in `scripts/cfg_gated_floors.txt` (the count
  is deliberately not written here — see the docs-gate preamble above). The one
  that earns its keep is `max_load_bearing = 0`: a new `*_goat.rs` gated on a
  default-off feature with no `required-features` row reds the push that adds
  it, before its green zero is cited as evidence. **Some of the pins are
  FLOORS**, on the population the auditor claims to have scanned, because a
  ceiling cannot fail once the instrument goes blind and reports zero. The
  first hazard found here was not the pins but the **trigger list** —
  `docs_gate.yml`'s `paths` filter carried no `.rs` glob at all, so the gate
  could not have fired on the only push it exists for.

  The **second** was worse and is the one to remember: `max_load_bearing = 0`
  is only as wide as `is_load_bearing`'s **vocabulary**, and a token-set gap is
  indistinguishable from a clean repo. T4c (2026-09-03, `2272b262`) found the
  set knew `goat`/`gate`/`g<N>`/`drill`/`proof`/… and did **not** know the
  `*_correctness` / `*_alloc_check` / `*_determinism` / `*_equivalence` /
  `*_floor` / `*_grad_check` dialect. Seven tokens added — each measured
  against all 2,157 workspace test+bench target names first, which is why
  `budget`, `check` and `calibration` were **rejected** — and 17 more
  load-bearing katgpt-rs targets appeared, including 8 `*_alloc_check` G4
  budgets and a Report-the-Floor UQ gate. All 17 armed and RUN in release:
  45 assertions, 45 pass, 0 fail — silently *unverified*, not broken, same as
  `.docs/10_audits/alloc_gate_per_thread_counter.md` T3. Found sideways, by adding tests to one such file, not by
  auditing the gate. Re-run the corpus token table when a new dialect appears.

  The **third** is what that widening did to the prose describing it. T4c made
  the classifier wider, so katgpt-rs's load-bearing ALL-IGNORED count went
  **3 → 5** — and both the pins-file header and
  `.docs/10_audits/cfg_gated_silent_zero_pass.md` item 7 kept their old
  numbers, because **nothing was pinned on that count**. The pins file argues
  the ALL-IGNORED *count* is not gateable (`#[ignore]` is the right marker for
  a slow or hardware-gated test) and that is correct but incomplete: **a set is
  gateable where its cardinality is not.** `scripts/all_ignored_load_bearing.txt`
  pins the five paths by MEMBERSHIP, each with the reason string read out of
  its own source, and `check_membership` reds on drift either way — so a sixth
  arrival reds the push, a same-size **swap** fails (the repo-set incident's
  lesson, one axis over, pinned as its own selftest case), and an emptied
  measured set reds on five removals instead of passing like every ceiling in
  this family does. An empty *allowlist* is refused for the mirror-image
  reason. All four directions canaried. It is deliberately NOT in
  `REQUIRED_PINS`: it is not an integer.
- `population_sync_gate.py` asserts that the **six** independent
  "which repos are contract repos" predicates agree —
  `cfg_gated_target_audit.derive_repos`, `numbering_drift_sweep.contract_repos`,
  `percentile_index_audit.repos`, `ci_gate_coverage.derive_repos`,
  `skill_repo_set_gate.derive_repos`, `suite_membership_audit.derive_repos`.
  They do (2026-09-06: 16 repos, identical, equal to `repo_set.txt`), and
  nothing asserted it. The failure is silent in the worst way: one predicate
  drifts, that one instrument quietly audits a different set of repos, and
  still prints a confident green over its own slice — which this workspace has
  already paid for once, with three instruments covering 7, 12 and 15 of 18
  repos. It is `docs_gate_paths_sync.py` one axis over: a hand-duplicated
  *predicate* drifts exactly like a hand-duplicated *value*.

  **It runs in CI, where none of the sweeps can**, because it tests the
  PREDICATE rather than the population — against a synthetic workspace carrying
  every case the real walk distinguishes, including a `worktree-shaped` entry
  whose `.git` is a FILE (admitting it double-counts a repo already in the
  walk, which is why the `.git` test must be a directory test). The
  real-workspace cross-check runs only when the walk finds more than one repo
  and is REPORTED either way, never silently skipped. The canary that matters
  is the last one: in a simulated single-checkout CI, a broken predicate still
  reds — so the gate is not vacuous in the environment it actually runs in.

- `required_features_static_gate.py` (riir-train Issue 513) is the verdict half of the
  free static pass above: a row naming a feature its package cannot enable
  reds the push that adds it. Gateable where the report's other two verdicts
  are not, because it needs no compiler and is **never legitimate** — cargo
  silently skips such a target in every invocation that does not name it,
  `--all-features` and `cargo test --workspace` included, so it reports a
  green zero forever while every audit counts it as protected. Pins in
  `scripts/required_features_floors.txt`; `min_rows_scanned` is a **FLOOR**
  for the usual reason (a manifest-parse regression takes the population to
  0 and the ceiling passes). Canaried three ways — plant an invalid row,
  blind the population, break the validity model — and the first canary
  earned its keep immediately: it found that `parse_rows` shadowed the
  package's declared-feature set with the row's own feature list, which made
  every feature trivially "declared" and the gate structurally incapable of
  firing.
- `bench_doc_audit.py` runs a `selftest()` on every invocation pinning the line
  shapes its tokenizer must recognise. Without it a regex regression is silent:
  the audit recognises fewer labels and still prints "0 mismatches". That is how
  26 riir-chain benchmark docs audited as clean while being unreadable
  (`.docs/10_audits/sibling_doc_drift_auditors.md`).
- `percentile_floor_gate.py` is the GATE over
  `percentile_index_audit.py` (pins in `scripts/percentile_floors.txt`),
  katgpt-rs-scoped like the cfg-gated one and separate from its report for the
  same reason. What it buys: a new site whose percentile index lands on `n - 1`
  reds the push that adds it, **before** that number is quoted in a
  `.benchmarks/` table as though it were a tail — print-only or asserted, a
  misleading number in a benchmark doc is the input to somebody's
  promote/demote decision. It imports the report rather than re-implementing
  the tokenizer, and runs the report's `selftest()` first, exiting **2** if the
  instrument itself is untrustworthy — a distinct outcome from a moved pin.
  `min_sites_scanned` is a **FLOOR** for the reason spelled out in that file:
  the three ceilings are green over whatever the vocabulary can NAME, so a
  tokenizer regression takes the population to ~0 and every ceiling passes,
  indistinguishable from a clean repo. Canaried in both directions before
  landing (a planted degenerate site → exit 1; the floor raised above the
  measured population → exit 1).

### The docs gate covers ONE repo — two more tiers cover the rest

Both auditors accept a repo path and audit any repo, and for months nothing
pointed them anywhere but here. A sibling with stale labels then looks exactly
like a sibling nobody has checked. Three tiers now, deliberately different
cadences, and none of them subsumes another:

| instrument | where | cadence | scope |
|---|---|---|---|
| `docs_gate.yml` / `docs_gate.sh` | CI + workstation | per-push | katgpt-rs only |
| `sibling_docs_drift.yml` | sibling CI (reusable) | caller's choice | one caller |
| `scripts/docs_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/numbering_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/required_features_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/percentile_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/cfg_gated_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/cfg_row_implication_drift_sweep.py` | workstation | on demand | every contract repo |

**A sixth joined 2026-09-06, over a verdict none of the other five can
reach.** `scripts/cfg_row_implication_drift_sweep.py` +
`cfg_row_implication_drift_floors.txt` ask whether a target's
`required-features` row, resolved WITH the package's defaults, satisfies that
target's own leading `#![cfg]`. If it does not, the row builds and compiles the
target to **nothing** — and unlike every other silent-zero in this section, the
row is PRESENT, so `cfg_gated_target_audit.py` counts the reader as protected
and `required_features_build_audit.py` reports BUILDS and is *right*. Measured
2026-09-06: **16 repos, 1,868 rows, 1,172 with a leading `#![cfg]`, 1
EMPTY-AT-ROW, 2 UNRESOLVED.** The one open instance is riir-ai's
`self_advantage_hla_bench` (ratcheted, Issue 513 instance 7). It found
riir-train's `bench_dflare_heterogeneous_kv`, whose row named the weaker of two
real features; fixing it took that target from `0 passed` to `1 passed`
(`054a39a2`).

Two details are worth copying. Its floors sit under BOTH ceilings, and
`min_with_cfg` is the load-bearing one: `leading_inner_cfgs` returns `[]` for
"no cfg" *and* for a file it cannot find, and an early cut silently skipped
cargo's DIRECTORY target form (`tests/<name>/main.rs`). And it asserts its
katgpt-rs row against the per-push gate's own pins — which **caught a desync on
its first run**, the hand-duplicated-value hazard `docs_gate_paths_sync.py`
exists for one axis over.

**The fifth completes the family, and it is the one with a backlog.**
`scripts/cfg_gated_drift_sweep.py` + `cfg_gated_drift_floors.txt` gate the
cfg-gated silent-zero ceilings across every contract repo. Its ceilings are a
**RATCHET at each repo's measured count**, not a wall like the two above:
running it is what found Issue 728's **12 load-bearing SILENT-NOW targets**
across six repos, ten of them sibling-owned and not one session's to arm. Every
one is the same shape — auto-discovered, no `[[test]]` row at all, gated on a
default-off feature — so each prints `ok. 0 passed` over an empty binary on a
plain `cargo test`. Measured 2026-09-06: **16 repos, 2,942 targets, 1,765
`#![cfg]`-gated, 261 SILENT-NOW, 12 load-bearing**.

Two details worth copying. It asserts **all four** of katgpt-rs's numbers
against `cfg_gated_floors.txt` rather than one spot check, since every field
names the same quantity as the identically-named key there. And its selftest
pins the **classifier in both directions** by name — the six widened tokens
must still fire, the seven homonyms must still not — because every ceiling
below it goes green over a smaller population the moment `is_load_bearing`
narrows, which is precisely Issue 728.

**A fourth followed the same day, over the percentile audit.**
`percentile_floor_gate.py` was katgpt-rs-scoped for the same structural reason,
and the zero it protects was *earned*: on 2026-09-03 the report found **12
DEGENERATE sites** and four sibling owners fixed every one the same day. Nothing
outside this repo was holding that result.
`scripts/percentile_drift_sweep.py` + `percentile_drift_floors.txt` now gate all
four classes at 0 across every contract repo — measured 2026-09-06: **16 repos,
11,132 `.rs` files, 110 sites, 0/0/0/0**. Same two-floor design and the same
`min_*` slack reasoning as `percentile_floors.txt` (a repair campaign
legitimately SHRINKS the site count; a ratcheting floor would red the next
correct refactor). The classification was factored into the report as
`pia.tally()` so the gate and the sweep cannot disagree about what a DEGENERATE
is — including the deliberate asymmetry that WEAK counts only when `asserted`
while TRUNC-VAR counts regardless.

Its 125 → 110 population edge was **checked before it was pinned**, because a
falling audit population is a repair or a blindness and the count cannot tell:
the instrument's last commit predates the 125 measurement, **katgpt-rs is
invariant at 41 across every measurement in the record**, and the sibling churn
is measured (512 commits, 47 deleted `.rs` files, 40 of them from one in-flight
extraction). Full record: `.docs/10_audits/percentile_index_tail_support.md`.

**A third instrument was pointed at the siblings 2026-09-06, and this one
found nothing — which is a result, not a non-event.**
`scripts/required_features_drift_sweep.py` runs the free static verdict (a row
naming a feature its package cannot enable) over every contract repo, against
committed pins in `scripts/required_features_drift_floors.txt`: **16 repos, 138
manifests, 1,829 rows, 0 invalid**, in under a second. The two sweeps before it
were pointed at siblings for the first time and immediately returned 7 dead
workflows and 35 duplicate numbers; this one confirms a clean workspace and
**pins it there**, which is the whole point — the 0 was already measured by hand
the day before and nothing was keeping it at 0.

Two things about its pins are worth copying. `max_invalid = 0` is a **WALL, not
a ratchet** (the numbering sweep pins each repo at its measured backlog; there
is no backlog here and the defect is never legitimate). And it carries **two
floors**, because the usual one is blind where it matters most: **seven of the
sixteen repos legitimately have ZERO rows**, so `min_rows = 0` there detects
nothing, while `parse_rows` swallows `TOMLDecodeError`/`OSError` with a bare
`continue` — a manifest that stops parsing is indistinguishable from one
carrying no rows, and in a 0-row repo, from the repo itself. `min_manifests`
still bites there, and unparseable manifests are counted and reported rather
than skipped.

Canaried eleven directions before landing, and **the first pass of the
end-to-end canary passed for the wrong reason** — with no `katgpt-rs` row in
the temp pins, the hand-duplicated-pin assert fired and exited 1 before the
planted invalid row was ever evaluated, so a green "the verdict fires" was
really "some other check fires". Re-run isolated, with a control arm that must
PASS, it fires on the row itself. That is the same shape as the shadowing bug
this gate family was already bitten by (`parse_rows` shadowing the package's
declared-feature set, which made the per-push gate structurally incapable of
firing): **a canary that reds proves nothing until you check WHICH assertion
red it.**

**The same one-repo blindness was measured a second time, on a different
instrument, 2026-09-05 (`.issues/725`).** `numbering_gate.py` also accepts a
repo path and had also never been pointed anywhere but here — and katgpt-rs,
the one gated repo, was the only clean one: **35 tracked duplicate numbers
across riir-train (13), mmorpg-editor (12), riir-ai (6) and riir-clippy
(4)**, in allocator-serial directories where `Plan N` now resolves to two
documents. Plus a defect class the local gate could not have: **five
`.highwater` files that are not integers at all**, every one of them `echo -n
<N> > .highwater` under a shell whose builtin `echo` ignores `-n`, so the flag
lands in the file (`-n 872`). `scan()` swallowed the `ValueError` and returned
`None` — *which is also what an ABSENT allocator returns* — so the
above-highwater ceiling passed over a corrupted allocator and a clean directory
identically. Repairing one of them **immediately exposed a stale allocator
underneath it** (riir-train `.plans` max 375 > 374) that had been invisible for
as long as the file was corrupt. All 7 allocator defects are repaired and
pinned at 0. The 35 duplicates were then RESOLVED DOWN TO 12 the same day
(Issue 725 T4b/T4c): **riir-ai 6 → 0** (T4a's `scripts/citation_weight.py`
attribution instrument — by-name citations are 0-2 per side and TIED in four
of six pairs, so `Plan N` mentions are ATTRIBUTED by token overlap with an
UNRESOLVED bucket and a zero-is-not-zero guard measured on `.plans/229`;
175→568, 182→567, 229→566, 313→569, R020→362, R148→363), **riir-clippy 4 → 0**
(its Issue 069, `58e7c1d`), **riir-train 13 → 0** (its Issue 514, `103ed351`
— hand reads overturned the four UNDECIDABLE rows). The remaining **12 are
all in read-only mmorpg-editor**, ratcheted at the measured count; the
lesson over the pessimism: the instrument is advisory — the corpus does not
distinguish near-synonyms, but reading the actual sites does.

Both sweeps are **deliberately not** in `docs_gate.sh`'s `CHECKS`: CI has a single
checkout, so it would derive an empty population and print a confident green
over zero repos. Its population is derived (BOUNDARY.md + a `.git` dir); its
expectations are committed (`scripts/docs_drift_floors.txt`), because deriving
both from the same walk is what makes a cross-repo gate permanently green.

`sibling_docs_drift.yml` is `workflow_call` so the three easy-to-get-wrong facts
live once rather than once per sibling — the worst being that the auditors
default to auditing **katgpt-rs**, so a sibling that omits the path argument
passes forever against the wrong repository. The workflow asserts the audited
tree is the caller's.

`scripts/ci_gate_coverage.py` is a **report, not a gate** (always exit 0): which
of the derived contract repos actually gate their full compile+lint surface in
CI, following each workflow into the scripts it calls, **and whether anything
automatically starts it**. Run it instead of re-typing the answer — that
question has been answered by hand twice and been wrong both times
(`.issues/701` R2).

Its own join is pinned by a `selftest()` that runs on every invocation, five
shapes, and the pin was canaried by reintroducing the bug it exists to catch —
a first cut asked only whether *any* scheduled workflow carried a cargo signal,
and a data-borne mention in `riir-chain`'s scheduled `toolchain_drift.yml` was
enough to vouch for a dispatch-only `rust.yml`. A weak automatic gate must not
speak for a strong manual one.

## cfg-gated targets — original section

### A green test count can be a count of nothing — `scripts/cfg_gated_target_audit.py`

A test file opening with `#![cfg(feature = "x")]` compiles to an **empty
binary** when `x` is off. Cargo prints `running 0 tests` / `ok. 0 passed` and
**exits 0**, which is byte-for-byte a real pass. The `#![cfg]` protects the
**count**; `required-features` protects the **reader**. Both are needed, and
only the second one is visible to whoever reads the output.

Do not answer "how much of this is affected" by reading manifests. Run:

```bash
scripts/cfg_gated_target_audit.py            # all contract repos (derived)
scripts/cfg_gated_target_audit.py ../riir-ai # or one, by path
```

A **report, not a gate** (always exit 0), for the same two reasons
`ci_gate_coverage.py` is: a `cfg` on `target_os` / `miri` genuinely cannot be
expressed as `required-features`, and neither can an `any(...)` of features
(cargo's is AND-only). Those are reported as their own classes — a report that
cries wolf on the shape cargo cannot fix gets ignored on the ones it can.

**The sibling instrument answers the next axis down: is a specific target
named by ANY committed suite?** `cfg_gated_target_audit.py` classifies how a
target compiles; `scripts/suite_membership_audit.py` (first census
2026-09-04, `.docs/10_audits/suite_membership_census.md`) reports which
`[[test]]` targets no script/workflow names — the "gate nobody runs" class
(865, 868) as a standing census instead of an ad-hoc hand grep. "Unpinned"
means unnamed, not broken; the actionable cut is load-bearing + unpinned +
default-visible + no broad `cargo test` run in the repo, and on the first
census that cut lands exactly on the two documented populations (this repo's
723 and riir-train's 507). Run it when landing a new gate: if nothing names
it, either add a suite row or record why not.

**Read the severity split, never the pooled total.** A target gated on a
*default-on* feature still runs on a plain `cargo test` and only vanishes under
`--no-default-features`. A *default-off* one reports a green zero every time
anyone names it. Pooled, the first measurement read 702 and meant nothing; split,
it is **382 SILENT-NOW / 320 latent** across 19 repos, a large minority of them
load-bearing by name (`goat`, `gate`, `g<N>`, `drill`, `proof`, …). Full record
and the per-repo table: `.docs/10_audits/cfg_gated_silent_zero_pass.md`.

Do not re-type those numbers from here — `.docs/10_audits/cfg_gated_silent_zero_pass.md` carries **two** same-day
corrections. The first cut over-counted by 48, keying declared targets by name
against the filename stem, so a row with an explicit `path` read as undeclared.
The second was in the *load-bearing* classifier built for T4: a token matcher
written independently returned 87 where the published table's ad-hoc substring
grep said 93, and **five of the six disagreements were the token matcher's
misses** (`g16f`/`g2p`/`g9gov` — G\<N\> with a variant suffix; `drills` — a
plural; `regate` — a compound). It now reproduces the table exactly, per repo.
Two independently-built classifiers agreeing is what licenses the
`max_load_bearing = 0` pin; a false negative would have made that pin a
permanent green. Run the script.

**Measured a fourth time on 2026-09-06, and this time by widening the
POPULATION rather than the vocabulary (`.issues/728`).** The classifier had
only ever run over katgpt-rs, because the gate is katgpt-rs-scoped. Run over
all 16 repos it reported `silent_now_load_bearing = 0` in **every one**, over
261 SILENT-NOW targets — a zero that reads as "nobody ships a silent
load-bearing gate" and means "the classifier speaks one repo's dialect". Six
tokens plus the adjacent-pair compound `spec_match` took the workspace count
**0 → 12**, **two of them katgpt-rs's own**. The compound is the mechanism
worth knowing: neither half survives alone, because `spec` is a homonym for
*speculative* decoding **in this repo** and `match` admits `attn_match_*`. And
the reason the two local instances hid is the sharpest part — katgpt-rs ships
seven `*_spec_match` targets and **five were only ever classified because they
also carry `g1`**. A convention that is visible only by accident is not
covered.

**And do not read that agreement as more than it is.** The two classifiers
agreed, and they agreed on the wrong *population* — T4c widened the token set
and 17 more katgpt-rs targets appeared. Agreement licenses the pin against a
classifier **bug**; nothing licenses it against a classifier being
congenitally **narrow**. The defence is the corpus-wide candidate-token table
in `.docs/10_audits/cfg_gated_silent_zero_pass.md` T4c, not a second opinion.

**A third kind was added 2026-09-03, and it is the one a green `w/ req-f`
column lies about.** The report modelled feature-expressible gates and
`target_os`/`miri` ones. `debug_assertions` was pooled with the latter, and
that hid the worst case in the set: every *other* predicate is silent only in
a configuration somebody **chose** (the wrong platform, miri,
`--no-default-features` typed), while `not(debug_assertions)` is silent under
plain **`cargo test`** — the default invocation, on the right machine, with no
flags. It also **survives the fix**: adding a `required-features` row moves the
target into "w/ req-f", which reads as protected, and does not make it compile.
So it is reported as an overlapping **dimension**, not a fourth bucket, leaving
the partition assertion untouched. Measured over 19 repos: **133 targets, 29
load-bearing, 11 already "covered"** — and split by direction, since pooling
would repeat the error that report already documents for platform gates:
**130 `not(debug_assertions)`** (green zero on `cargo test`; 26 load-bearing,
almost all riir-ai GPU benches) against **3 bare `debug_assertions`** (green
zero under `--release`) — and *all three of those are load-bearing alloc
gates, which vanish exactly when someone follows the rule above to run gates
in release*. The instruction and the gate were in direct conflict and nothing
said so. Found by riir-ai `.issues/855` Class 2, whose fix pattern for the
conflict is to split the alloc assertion from the wall-clock one, since no
single profile can observe both.

The verdict half is `scripts/cfg_gated_floor_gate.py`, a docs-gate check. The
report and the gate are deliberately separate files: the report must stay
runnable over the siblings whose owners have not taken Issue 713 T3, and a
report that exits 1 on them is a report nobody runs.

It was fixed one target at a time twice in one week — riir-train `5821cba9`
(11 real assertions reporting as a green suite having run none) and riir-clippy
`19beece` — before anyone asked how many there were. Fixing them one at a time
is how it stayed invisible.

Adding the rows changes cargo's behaviour in exactly one way: `cargo test
--workspace` silently *skips* a target whose required-features are off, and
naming the target without its features errors with exit 101 instead of
reporting a green zero. That was verified, not assumed, before the katgpt-rs
batch landed (`180be9c5`, 39 GOAT gates, SILENT-NOW 102 → 63, baseline
re-measured with the corrected auditor rather than inferred from the delta).

**This paragraph used to open "Adding the rows is safe and does **not** red an
existing CI", and that was false — measured 2026-09-06.** It is a claim about
*cargo*, and it does not survive contact with a gate that counts **green test
binaries**, because an empty `#![cfg]`-gated binary prints `test result: ok.
0 passed` and is counted as one. Arming such a target removes the LINE, so the
count falls. Three repos count that way (`grep -c '^test result: ok'` in
`riir-auth`, `riir-dapps`, `riir-viewbridge` `scripts/ci_feature_guard.sh`) and
two of them were reddened by a one-row arming:

| repo | floor | before arming | after | note |
|---|---|---|---|---|
| riir-viewbridge L4 | `>= 13` | 13 | **12** | one row (`net_ffi_roundtrip`) |
| riir-dapps L2 | `>= 32` | 32 | **31** → **7** | one row, then all 25 |

Both floors' own failure message asks *"a target silently compiled to
nothing?"* — and both were **satisfied by exactly that**. riir-dapps' `>= 32`
was 25 empty binaries plus 7 real ones; the 7 real binaries carry 111 passing
tests. The mechanism was pinned in a throwaway crate rather than inferred: one
gated test, feature off, `grep -c '^test result: ok'` reads **3** with no row
and **2** with one.

The repair is not a re-pin. A **passed-test floor** beside the binary floor
cannot be satisfied by a target compiling to nothing (+1 binary, +0 passed),
and both siblings now carry one (riir-viewbridge `d6f18f5`, riir-dapps
`dd6261e`). Check for a binary-counting floor before arming a target in any
repo — and note the inverse holds for `--lib`-scoped floors (riir-ai) and for
count-pinned rows that name a target *with* its features (riir-chain's
`test_gate.sh` row for `mnemonic_spec_match`): those do not move.

**Run the armed gates with `--release`.** All 39 pass there. A first sweep
without it reported four reds and nearly filed two of them as perf
regressions: a latency gate in a debug build measures an unoptimised binary,
and `fast_bpe_goat` is 388 s in debug against 15.6 s in release. Re-measuring
three times on a quiet box gave a sub-1% spread and made the wrong number look
*more* trustworthy — ruling out the confounder you thought of says nothing
about the one you didn't. Arming the gates still paid: it surfaced
`.docs/10_audits/alloc_gate_per_thread_counter.md`, an alloc gate counting a sibling test's allocations, which
reproduces in release.

## required-features rows — original section

### A `required-features` row can EXIST and be WRONG — `scripts/required_features_build_audit.py`

Every audit above treats a target as protected once it **has** a
`required-features` row. That is the right check for the failure they were
built for. It says nothing about whether the row is *correct*, and a row that
exists and is wrong is strictly worse than a missing one:

- `cargo test --workspace` silently **skips** the target (features unmet), so
  nothing reds.
- `--all-features` **builds** it, because the union supplies whatever the row
  forgot — so the one configuration anybody runs it in passes.
- Every audit counts it in the "w/ req-f" column, i.e. **protected**.

It cannot be answered statically: the row is wrong relative to what the file
*imports*, and the import resolves through `lib.rs` re-exports that are
themselves cfg-gated — the glob-re-export problem this repo already documents
as defeating grep. So the check is to ask the compiler, once per target:

```bash
scripts/required_features_build_audit.py --list            # rows only, no builds
scripts/required_features_build_audit.py ../riir-train     # one repo
scripts/required_features_build_audit.py . --grep pruners  # one slice
scripts/required_features_build_audit.py ../riir-train --batch  # 1 run per set
```

A **report, not a gate** (always exit 0) and, unlike its siblings, also for a
cost reason: **1,829 rows over 16 repos** (2026-09-05), at ~28 s/row warm on
the first katgpt-rs slice — a full sweep is hours, so it is filterable
(`--package`, `--kind`, `--grep`, `--limit`) and takes `--target-dir` for when
a sibling session is building.

`--batch` is the cost lever, and its constraint is what makes it sound:
**one cargo run per (package, EXACT feature set)**, never per subset. Building
a target at a SUPERSET of its own row and seeing it succeed proves nothing —
the extra features may supply the very import the row forgot, which is the
failure the whole report exists to catch (`--all-features` builds every wrong
row). Equality is therefore the only batchable relation, and it is worth
batching: **1,829 rows collapse to 1,070 groups (1.71x)** cargo invocations.
Do not read that ratio as the speedup. Measured on riir-clippy (44 rows / 25
groups, cold dirs, two pairs run in both orders): **-11% and -8% CPU-seconds**,
while **wall-clock flipped sign** (-12%, then +13%) on a box with sibling
builds live — 19 saved invocations over one package whose feature sets share a
dependency graph is not separable from load. The mechanism that pays is the
per-invocation resolve, so expect more where a feature set swings the graph
(riir-train's CUDA sets) and near-nothing where it does not. Verdicts stay
per-target — attributed from `--message-format=json`
`compiler-message`/`compiler-artifact` target names, with `--keep-going` so
one red target does not truncate the run. A row with **neither** an error nor
an artifact reports **UNSEEN, never BUILDS**: silence is not evidence, and
reading it as success would be the same green-zero this family exists to
refuse (`attribute()` is pinned by `selftest()` in both directions, canaried
by reintroducing exactly that collapse). It warns when another cargo holds the repo's
`target/`, by working directory, for the reason
§"Several sessions, one target dir" gives.

Two instances so far, and the second is the argument for the instrument: the
first was found by hand (riir-train `9da3420f`, `test_cubecl_backward_grads`
omitting `gpu_training_resident`; fixing it immediately reported 9 passed /
1 failed, so the wrong row was hiding a real defect). The second was found by
this script **in the first six rows it checked, in this repo** —
`bench_001_pruners_goat`, `["bomber", "go"]`, `E0432`. Its twin
`bench_001_pruners_goat_proof` had the identical defect *and it was already
fixed*, by Issue 723 T7, which did not look at the file next to it. Five
assertions that had never been runnable at their own row now run and pass. A
hand fix of one instance does not close a class.

**One of its three verdicts is free.** A row naming a feature the package
cannot enable is decided by the manifest alone, so it runs as a static
pre-pass before any build: **0 invalid rows over all 1,829, in under a
second** (2026-09-05). Do not re-derive that check by hand — the obvious
model is wrong. `required-features` accepts **`dep/feat`** and `dep?/feat`,
naming a DEPENDENCY's feature rather than one of ours, and a first cut that
treated those as undefined reported 10 riir-ai benches as dead targets. A
`/tmp` probe with a `compile_error!` canary inside the target (cargo 1.98.1)
settles it: cargo satisfies such a row via a package feature that enables it,
by naming `dep/feat` directly, and under `--all-features`, and skips the
target only in a plain no-features build. Renamed (`package = `) and
`[target."cfg(…)".dependencies]` entries count as dependencies too.

**And that free verdict is correct AND insufficient — measured 2026-09-06.**
The sixth instance of the class is `riir-train`'s `lora_muon_optimizer`, which
forwarded `riir-gpu/amuse_optimizer` — the DEPENDENCY's feature — while
`pub mod optimizer_amuse` in `riir-train-gpu` is gated on its OWN
`amuse_optimizer`. That is a perfectly valid `dep/feat` entry, so the static
pass is right to pass it, and only the compiler distinguishes "names a feature
that exists" from "names the feature that gates the module".

**The sixth instance also widened the class, and the widening is the part to
carry.** Instances 1-5 are wrong ROWS: a target that cannot build at exactly
the feature set its own `[[test]]` declares. Instance 6 is a feature set under
which the **LIBRARY** does not compile — `cargo check -p riir-train-gpu --lib
--features lora_muon_optimizer` was `E0432` ×2 — so cargo errors before
reaching any target and ten rows can only report **UNSEEN**. Both are invisible
to `cargo test --workspace` (which skips the target) and to `--all-features`
(which supplies what was forgotten), for the same reason.

Which is why **UNSEEN must never be folded into the pass column**, and this is
now measured rather than argued. The obvious reading of those ten — "downstream
of the two FAILS beside them, they will clear when those are fixed" — was
TESTED and is FALSE: re-running after instances 4 and 5 landed left all ten
still UNSEEN, which is the only reason the non-compiling library was found. A
report that had counted them green would have hidden it behind two unrelated
fixes.

**Two instances repaired in OPPOSITE directions, which turns an anecdote into a
rule: read the USE SITES, not the error.** `dasd_lora_goat` omitted `asft_loss`
and its body uses `AsftConfig`/`asft_loss` unconditionally, so the ROW was
wrong — widen it. `goat_235b_filter_training` had an UNGATED import whose only
use site already carried `#[cfg(feature = "lora_outlier_guard")]`, with
`..Default::default()` covering its absence — widening the row would have forced
an optional leg on and defeated the file's per-leg design, so the IMPORT was
wrong; narrow the cfg. Widening is the tempting repair and is right only when
the body needs the feature unconditionally.

### The per-push half — `scripts/required_features_touched_gate.py`

The sweep above is the right instrument at the wrong cadence: 1,866 rows over
16 repos at a ~28 s/row mean is a scheduled job. But every instance in the
record arrived in a **commit**, so the affordable gate is not "check every row",
it is "check the rows this push could have broken" — cost bounded by the diff.
`.github/workflows/required_features_touched.yml` runs it per push and per PR
on macos-latest (the platform is part of the claim here, unlike the pure-text
docs gate).

Two selection rules. A changed file that IS a row's target source selects that
row; a changed `Cargo.toml` selects the rows whose own `(kind, name,
required-features)` tuple **differs between base and head**. The second is a
ROW diff, not "every row in the package", and that is the whole design —
selecting the package would refuse every manifest edit in `riir-train-gpu` (440
rows), the repo three of the six instances live in. Both sides parse through
the sweep's own `rows_from_manifest` so the two instruments cannot disagree
about what a row IS.

**Read a green narrowly, and the gate prints why on every run.** A changed
`src/**.rs` can break a row in any dependent package — instances 1 and 6 are
both that axis — and selecting on it is unbounded, so it stays the sweep's
(`--src-fanout` opts in, `--max-rows` REFUSES rather than truncates; a silent
truncation would be the blind spot the gate exists to name). Measured cost: 0
rows selected ~0.4s · 1 warm row 3.1s · 1 row at a feature set that swings the
graph 266.7s.

Record and the open sweep: riir-train `.issues/513`.

## Percentile index — original section (classifier history)

### A reported "p99" is often the MAX — `scripts/percentile_index_audit.py`

`sorted[(n as f64 * 0.99) as usize]` and `sorted[n * 99 / 100]` both land on
`n - 1` — the **maximum** — for every `n <= 1/(1-p)`: n ≤ 100 at p99, n ≤ 20 at
p95, n ≤ 1000 at p999. Below that boundary the site reports one observation
under a percentile's name. A `.min(len - 1)` clamp prevents a panic, not a
wrong statistic.

Direction matters and is the opposite of the usual worry: the naive index is
one rank **too high**, so a `p99 < budget` assert becomes *stricter*. The
failure mode is a false **RED**, not a false green — nothing currently green
turns red by fixing a site. What it costs is the ability to notice a real
regression, because the tail is decided by a single sample.

The quantity nobody prints is **tail support** = `n - idx`, the number of
samples at or above the reported rank: **1 at n=100**, 2 at n=200, 10 at
n=1000. The report calls anything under 10 weak whether or not it is
degenerate.

```bash
scripts/percentile_index_audit.py             # all contract repos (derived)
scripts/percentile_index_audit.py ../riir-ai  # or one, by path
```

A **report, not a gate** (always exit 0), for the reason
`cfg_gated_target_audit.py` is: half the sites take their sample count from a
runtime length or a fn parameter that no static pass can reach, and a report
that exits 1 on those is a report nobody runs. Read the split — **UNRESOLVED
is not "clean"**, it is "needs a per-site read".

Do not re-type its numbers from here — the durable record with the per-repo
table, the commit evidence and the open sibling rows is
`.docs/10_audits/percentile_index_tail_support.md`. Measured
**2026-09-03 evening: 126 sites over 9 of 19 repos — 0 DEGENERATE, 2 TRUNC-VAR,
6 WEAK, 31 OK, 62 UNRESOLVED, 25 SAFE.** The DEGENERATE zero is real and
earned: four owners fixed all 12 the same day (riir-ai `03a91ed59` swept 10,
riir-mmorpg-examples `ee9da24` the one DEGENERATE-**ASSERTED** site,
riir-game-sdk `f896bca`, riir-chain `7f3a3910`).

**And the fix blinded the gate, which is the part to remember.** All four
repairs consolidated the arithmetic behind a `nearest_rank(sorted, p)` helper,
so `p` became a **parameter** — and every pattern in the vocabulary required a
*literal* p. Sixteen sites left the population (130 → 114); `max_degenerate = 0`
then read green over a population that no longer contained riir-ai's percentile
surface at all, and **seven byte-identical copies of that helper across five
repos** were invisible. Fourth instance of the classifier-narrowness failure
below, first one reached by a *correct* fix. Closed by a fifth verdict,
`TRUNC-VAR` (a truncating variable-p rank inside a percentile-named scope — a
finding without a resolvable n, since `floor(p*n)` is the max for every
n ≤ 1/(1−p) whatever p is), a fourth ceiling `max_trunc_var = 0`, and a
`.trunc()` hole in the rounding exclusion that had been clearing the defect's
own second spelling as SAFE.

Two corrections to the reasoning above, both in that record: **"false RED, not
a false green" is assert-direction dependent** — it holds for `p99 < budget`
and inverts for a `p95 >= floor` diversity row, where a too-high tail is a
false GREEN — and **`asserted` is structurally blind for helpers**, since
`is_load_bearing` is deliberately same-fn-scoped while a helper sits one call
frame from the assert it decides. That is why `TRUNC-VAR` is the one class
gated regardless of `asserted`.

**This file's own history is the lesson about classifiers.** The audit has been
wrong once per vocabulary gap — never by a bug — and every time the narrow
version looked like good news. (The count is deliberately not written here; the
list grew a fourth entry the same day this sentence would have said "three".)

1. The first cut grepped only the **float** forms and published a 14-row hand
   table as an audit of "all 19 contract repos". The integer form
   `n * 99 / 100` is the more common one here and was invisible; riir-e2e's
   copy was found by accident.
2. Both `resolve_n` and `is_load_bearing` were **file-scoped**, so a binding
   or an `assert!` in a *different function* counted. That manufactured a
   false ASSERTED (riir-neuron-db `bench_003`, where the assert is on
   `mean_us` — the p99's neighbour in a returned tuple) and sized a slice
   *parameter* from an unrelated caller (riir-chain `bench_012`).
3. The literal-only pattern reported **riir-game-sdk as having zero sites**
   while its `percentiles` helper — `|p: usize| durs[(n * p / 100)...]`,
   called as `at(50)`, `at(99)` — feeds that repo's wall-clock budget gates.
   A variable percentile is not statically known, so it must land in
   UNRESOLVED, not vanish.
4. The literal-only patterns were blind to a variable-p **helper body**, so
   the 2026-09-03 repair campaign — which consolidated twelve defective sites
   behind seven byte-identical copies of `nearest_rank(sorted, p)` — removed
   them from the population instead of moving them to SAFE, and the ceiling
   read green over the gap. Closed by `TRUNC-VAR` + a scope-name
   discriminator measured against all 27 candidate sites in the workspace
   (admits 8, rejects 19, including the two a bare `rank` substring would
   have swallowed). **A correct fix caused this one** — which is why the
   defence is the corpus-wide candidate table, not a second opinion.

So the vocabulary is **data** (listed exhaustively in `VOCAB`) and the
population is **derived** (BOUNDARY.md + a `.git` dir), per the workspace rule
that deriving both from one walk is what makes a cross-repo report
permanently empty. `selftest()` runs on every invocation and **exits 2 rather
than printing**, pinning the tokenizer, the scoping, the rounding exclusion
and the arithmetic. It was canaried by reintroducing bug 1 above — a greedy
character class that swallowed `sorted[(n` — which took every site to
UNRESOLVED; without the selftest the run would have printed 130 sites, zero
findings, and read as a clean repo.

A third "exclusion" was **not** deliberate and has been fixed: `.trunc()` sat
in the rounding exclusion beside `.ceil()` / `.round()` while the same
comment called truncation the bug. `x.trunc()` **is** `x as usize` for
non-negative x, so the defect spelled a second way cleared as SAFE. Latent
when found (zero percentile-context `.trunc()` sites, measured), and
`.floor()` was never in the set — it is the defect's own name. The two below
are the legitimate ones:
`((n - 1) as f64 * 0.99)` is bounded by `n - 2` and can never return the max
(verified over n ∈ 2..=20000; it is the shape
`katgpt-speculative/tests/weaver_real_checkpoint.rs` uses, and the one site in
the workspace that got this right), and `.ceil()` / `.round()` on the product
is the *correct* nearest-rank form — requiring it also removes the largest
false-positive class, `0.95 * n as f32` computing a top-p **nucleus size**,
which indexes nothing.

## Staged-set audit + shared target dir — original sections

### Before committing in a shared worktree — `scripts/staged_set_audit.py`

Several agent sessions write into one worktree routinely, and `git add -A` from
a repo root is indistinguishable, to git, from intent. `b2527521` committed
three agents' WIP in six files; one was a build regression nobody saw for a day
(`.issues/709`). Stage **named files** (`git -C <repo> add <paths>`), never
`-A` — and before a multi-file commit, run:

```bash
scripts/staged_set_audit.py            # any repo: pass its path as $1
```

Also a **report, not a gate** (always exit 0). The refusing pre-commit hook was
**decided against** (`.issues/709` T3b, 2026-09-03) on the evidence the report
itself produced: every signal cheap enough for a blocking hook has a
legitimate-use false positive, and the one signal with none (`--fmt`) costs a
rustfmt subprocess per file. A report that is read beats a gate that is
bypassed.
Measuring the signal is not. Three signals, none subsuming another:

1. **mtime clusters** — the technique that caught this by hand twice:
   **worktree mtimes cluster by editing episode.** A 204-file rustfmt sweep
   lands in a 3-second window; your own edits land in the window you were
   running. Two clusters in one staged set means two episodes, and the older
   one is probably not yours.
2. **also-dirty** — a staged path that *still* has unstaged changes: a
   concurrent editor writing into the same file right now, which mtime
   clustering cannot see because their write may land inside your window.
3. **stale-vs-HEAD** — a dirty-or-staged file that LACKS substantive lines the
   newest commit on its own path added. Committing it reverts them. Both other
   signals are blind to this: a whole-repo sweep is *one* episode and its files
   are not also-dirty. Found live on the first run — `tpr/als.rs`, written
   20:04:39 in a rustfmt sweep, while `0ef7f078` landed a 22-line Issue 712
   correctness fix in the same file at 21:08. It audits the **dirty** set too,
   because the hazard exists before anything is staged.

Signal 3 is two-stage on purpose. `mtime < commit time` alone false-positives
on the commonest shape there is — you edit at 21:03 and commit at 21:04, so the
newest commit on that path is your own edit; it flagged two such files on the
first run. Line-set **containment** is the confirmation and is what the warning
claims, restricted to lines specific enough that their absence is evidence (a
`}` proves nothing). A workspace sweep over all 19 contract repos returned
exactly one hazard, with two other repos dirty-but-clean — so it is not
always-on.

4. **rustfmt round-trip** (`--fmt`, Rust only) — `git show HEAD:$f | rustfmt
   --emit stdout | diff - $f`. Identical ⇒ the worktree copy is exactly "HEAD,
   formatted", provably carries **zero content**, and reverting cannot lose
   work. The only signal here that yields a *proof*; the other three are
   heuristics. Diffing against `rustfmt(HEAD)` also **isolates the content**,
   which is the review you want on a file whose real edit is buried under a
   whole-file reformat — `-w` cannot produce it, because rustfmt re-wraps
   tokens *across* lines. On its first run it refuted a standing belief: 15
   files described for hours as "a sibling's rustfmt churn" came back **0
   churn, 15 content**, every one a real lint fix (`989f1bdf`). Its `skip`
   verdict is pinned by `selftest()` — a `churn` verdict authorises a revert,
   so unlike the other three this signal can destroy work if it errs that way.

Single-linkage, not fixed-width bins: a session editing continuously for an
hour is one episode because no two consecutive edits are `GAP_SECONDS` apart.
Fixed bins would split it and false-positive on exactly the sessions doing the
most work. A `selftest()` pins ten shapes on every invocation — the chaining
case and signal 3's line-specificity predicate — because both failure modes are
silent: the clustering degrades to "1 episode, always" and the containment
check to "nothing is ever missing", each printing a confident pass
indistinguishable from a real one.

When you must commit into a file a sibling is editing, commit **your blob**
rather than the worktree's: build HEAD's version + your edit, `git hash-object
-w`, then `git update-index --cacheinfo`. Their hunks stay uncommitted and the
worktree stays coherent for them (used for `bench_707` in `8c7ca74b`).

### Several sessions, one target dir — a gate can produce a FALSE RED

Every axis above is about a gate being *wrongly green*. This one is the
inverse and it costs more, because a red gets investigated: **a count-pinned
or feature-switching gate run concurrently with another cargo process in the
same `target/` reports a failing test that passes when run alone.** Cargo
rebuilds and replaces test binaries while another run is executing them.

Read the failure's **shape** before its content. `error: test failed, to
rerun pass …` with **no `failures:` block and no `test … FAILED` line** means
the harness process *died*; nothing asserted anything. The count is
truncated too, because cargo stops at the first failing binary — so one
environmental cause emits two independent wrong signals, a fake failing test
*and* apparent pin drift.

Measured 2026-09-03 (riir-game-sdk `.issues/023` T4): a row read `105 passed
(pinned 182)` and named `prod_l3_partition_heal`. That suite has no
wall-clock component — seeded deterministic drills — so load flake was ruled
out and a riir-chain comparator change from the day before looked like a
cross-repo regression. It was neither: the freshly built binary run
**directly** passed 6/6. Three `cargo test -p riir-e2e` runs were live on
three different feature sets at once, two of them mine.

Two things follow. **Running the compiled binary directly needs no build
lock**, so it is the way to diagnose while siblings are compiling — find it
under `target/<profile>/deps/`, filtering out the `#![cfg]`-gated copies that
`--list` reports as 0 tests. And a gate whose verdict the box can invalidate
should **refuse**, not warn: riir-game-sdk `scripts/test_gate.sh` now detects
concurrent cargo by **working directory** (`lsof` over `pgrep -x cargo`)
rather than by command-line pattern — a pattern is blind to a plain `cargo
build` in the same dir and conversely matches harmless sibling-repo runs
whose command lines mention our crates. A lock-based check cannot work at
all: cargo releases `target/<profile>/.cargo-lock` *before* running the test
binaries, which is exactly the damage window.

## Feature Flag Discipline — rule histories (lossy surface, Report the Floor, Plan 467)

**Lossy-surface promotion rule (adopted 2026-08-28, riir-ai Issue 750 T3):** a
promotion of a **lossy** surface (quantization, compression, any bit-changing
transform) gates on **deployed-path behavior — per-family, conditional
retention**, not on bit-identity or aggregate perplexity alone: bit-identity
is only available to lossless surfaces. Three independent arrivals at this
rule: Research 502 ("Behavior Before Perplexity"), Bench 696 (the KVarN
sink-guard GOAT), and riir-ai Issue 750's measured bisection (gemma-2-2b Q4_K:
first behavior flip at prefix k=1 — layer 0 alone flips the sealed family;
restoring it costs 106.7 MiB, priced by the T2 override probe). Aggregate
perplexity can be flat while family-conditional behavior flips.
External measured confirmations (riir-clippy Research 125, walk #7):
arXiv 2609.01962 (post-training ternarization of Qwen3-4B — aggregate
accuracy 64.5→54.7% yet per-task chance-corrected retention spreads
84.6% BoolQ vs 43.8% ARC-Challenge; and a lossless packing run holds PPL
while a lossy one was excluded from the artifact claim) and arXiv 2608.12700
(a contract-grade verifier rejects 1,487/2,638 kernels a standard tolerance
harness accepted — the tolerance-budget fault-class confirmation queued at
walk #6).

**UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule, adopted 2026-06-28 per Research 322 / Plan 340).** Any primitive that claims a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty (collectively: **UQ-bearing**) MUST benchmark against the **conformal-naive floor** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340 with `m=1`, plain split conformal) — on CRPS / coverage / Winkler score. If the primitive cannot beat the floor, the GOAT gate FAILS. Existing UQ-bearing primitives (BoMSampler Plan 281, Sleep-Time Anticipator Plan 334, Best-Belief Beta Selector Plan 336, KARC+overlay) are grandfathered but must include the floor at their next re-gate; future UQ primitives must include it from the initial gate. Tracked in `.issues/010`. The floor shipped in Plan 340 Phase 1 (2026-06-30); the rule is now enforceable. **Issue 010 is FULLY CLOSED (T1-T7 all complete)** — see `.benchmarks/010_report_the_floor_consolidated.md` for the cross-primitive summary. **T7 (2026-07-20)** added the KARC+overlay dedicated floor test (`conformal_floor_karc_overlay.rs`) — the composite is SCOPE-LIMITED to chaotic regimes (BEATS on Lorenz-x at crps_ratio 0.0047 with K=4; LOSES on stationary seasonal at crps_ratio 5.74 with K=4), but coverage stays calibrated on both — no false-confidence signature. **T7 K-sweep (2026-07-20)** refuted the prior "K=4 too shallow" hypothesis: K=12 (matching the period) LOSES WORSE on seasonal (CRPS 5.74 → 20.26) and WINS HARDER on Lorenz (CRPS 0.0047 → 0.0018) — the scope-limit is **structural** (KARC's Chebyshev basis + ridge-fit doesn't fit periodic data regardless of K), not parametric. Production guidance: pick K by chaotic-regime memory needs; for periodic data use the floor directly.

**Plan 467 / Proposal 007 (2026-07-18):** Shipped `DualLeoOracle` as QGF's 3rd `QGradientOracle` impl — fuses a LEO teacher head + UVFA student head via `DualLeoMixer::combine_into` at the gradient level. Sibling to `LeoHeadOracle` (Plan 268) + `FlowFieldOracle`. G1–G4 PASS mechanistically; **G5 measured FAIL on synthetic data (riir-ai Bench 553, 2026-07-18): dual 0.00% vs single 0.50% on T7 Go puzzles, but the correctness invariant (QGF+LeoHeadOracle ≡ baseline) held bit-identically — mechanism correct, quality gate FAILs because synthetic data produces near-flat Q-fields.** **G5 also measured FAIL on civ real networks (riir-ai Bench 558, 2026-07-19): dual +2.69% vs single 35.68% → 36.64% on civ action-prediction, ≥3% gate — fourth-axis stop rule.** The civ dual-LEO investigation is fully closed per riir-ai Research 322 (the "alternative critic" escape hatch was category-confused — UQ primitives produce state forecasts, not per-action Q-gradients). The Plan 460 max-pool washout lesson is encoded as a design invariant (no operator between mix and consumer). Stays opt-in (`qgf_oracle + dual_leo`) with documented unproven G5 across both synthetic and civ real-network regimes; reopens only on mmorpg-remake integration gain, new game domain positive G5, or Q-vs-forecast research breakthrough.

## Substrate-First Gate — original section

## Substrate-First Gate (MANDATORY before implementing)

Before implementing ANY new System impl, trait, perception/cognition/emotion
pipeline, state management, spatial query, or vocabulary type, run the
`.agents/skills/substrate-first/SKILL.md` skill. It enforces:

1. **Vocabulary translation** — grep 3+ name variants (concepts ship under
   operator names like `GenericSpatialBelief`, not English names like "threat
   field"). A single-vocabulary grep returns ZERO hits even when substrate
   fully exists.
2. **Codebase grep** — search `*.rs` source across all 8 repos, not just
   `.plans`/`.docs`/`.issues`.
3. **Architectural rule check** — domain classification, two-brain model, sync
   boundary, bridge pattern.
4. **Consume vs. build decision** — if substrate exists, consume it; if not,
   file an issue in the right repo FIRST.

This prevents the recurring drift pattern where an agent builds a parallel
system that duplicates already-shipped substrate under a different name
(canonical failures: ThreatField Issue 047, orchard/motivation riir-ai Issues 490/493).

## Research Workflow — original section

See `.agents/skills/research/SKILL.md` for the full research workflow:
paper classification, 7-repo routing, fusion-first distillation, novelty gate,
GOAT gate, and the mandatory modelless-unblock protocol (§3.5).

## Repo count — the full original paragraph (drift history)

> **Repo count:** the **product/distillation set is 7** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-dapps` (private). That is NOT the repo total: the
> workspace is **16 repos**, all of which carry a root `BOUNDARY.md`
> (add `riir-mmorpg-examples`, `riir-clippy`, `riir-viewbridge`,
> `riir-auth`, `katgpt-web`, `riir-dao`, `riir-deployer`,
> `mmorpg-editor`, `mmorpg-remake`).
>
> **19 → 16 on 2026-09-04 00:01**, by the same owner act that retired
> `riir-armageddon`: `riir-burner`, `riir-unity` and `mmorpg-remake-unity` were
> moved to `/Users/katopz/git/obsolete/`. Nothing was lost — the directories
> are intact there and `riir-burner`'s last sweep was pushed (`ce54122`) 19
> minutes before the move. They are named here as **lineage only; do not route
> work to them**, exactly as with `riir-armageddon`. This time the drift was
> caught by an instrument rather than by reading: `skill_repo_set_gate.py` went
> red on the stale `scripts/repo_set.txt` within the hour, and
> `agents_repo_set_gate.py` then went red on this paragraph — which is the
> membership-first gate doing precisely the job it was written for, on its
> second day. The cross-repo **edge** count WAS re-measured post-retirement
> (2026-09-04, boundary-guard log): the C0e-landing contract run printed
> **exit 0 — 16 repos / 224 edges**, and the 23rd run re-confirmed 224 via
> `--list-deps` (its full run exited 1 only on the worktree-only C6 artifact
> from the sibling's 461-file rustfmt sweep — all five keeper pins exact at
> HEAD, proven formatting-only on a clean /tmp checkout). Live count drifts
> with landings — this session's `--list-deps` already reads 226 rows after
> the Issue-092 ndb-sdk edge landed — so don't re-type a number, run:
> `../riir-ai/scripts/ci_boundary_contract.sh --list-deps`.
>
> **This paragraph said 8 and 18 until 2026-09-03, and the way it was wrong
> is worse than a stale number.** `riir-armageddon` was de-enrolled by an
> owner act — the directory is GONE — and `mmorpg-remake-unity` was enrolled
> in the same window (boundary-guard's 18th run: 19 repos / 225 edges, the
> 227→225 delta being exactly armageddon's two allowlist edges leaving with
> the directory, the new repo adding zero). **One repo left and one arrived,
> so the total stayed 19 while the MEMBERSHIP changed** — every count in
> sight agreed, and the set was wrong anyway. A count is not a checksum over
> a set. The derived instruments were all correct throughout
> (`scripts/repo_set.txt` was regenerated at `d2cb9979`, and the repo-set
> gate is what caught the snapshot mid-session); only the prose was stale,
> which is precisely the split the command below exists to enforce.
> **Re-measured 2026-09-01** by
> `../riir-ai/scripts/ci_boundary_contract.sh` — *"boundary contract clean —
> 18 repos, 211 cross-repo dep edges measured"*. It was 15 at the 2026-08-21
> run recorded here before; the count moved because contracts were added, not
> because repos were, and **this paragraph did not** — which is the failure it
> warns about, committed by the paragraph itself. **19 later the same day**,
> and this time because a repo genuinely was added: `mmorpg-remake` was
> scaffolded at 23:41 with a root BOUNDARY.md (derived, not re-run through
> `ci_boundary_contract.sh` — so the 211-edge figure above is NOT re-measured
> and should be read as of the earlier run). `scripts/repo_set.txt` was
> regenerated the same hour (`01e19858`) because the docs gate went red on the
> drift, which is the instrument working. Don't re-type the number:
>
> ```bash
> cd /Users/katopz/git && for d in */; do
>   [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
> done
> ```
>
> Four of the 2026-08-21 repos had no contract at all until that run, and
> `riir-armageddon` had been consuming `riir-games` + `katgpt-core` unaudited
> (those are the two edges whose departure the 227→225 delta measures — the
> repo is retired, so this sentence is history, not a live gap).
> Read a count in prose as a claim, not a fact — and read a count that
> MATCHES as a claim too, per the membership swap recorded above.
> The historical "5-repo quintet" terminology referred to the 5 distillation
> targets (katgpt-rs + 4 riir-* siblings); `riir-game-sdk` (game vocabulary
> facade + dev-tool workspace) and `riir-armageddon` (arena/game-product domain
> types) were added later, and `riir-dapps` (the dApp layer — game outcome →
> generic chain settlement) on 2026-08-20. **`riir-armageddon` has since been
> retired** (2026-09-02, owner act) — it is named here as lineage only; do
> not route work to it. See Research 003 for the canonical boundary.
>
> **Two axes, not one.** Research 003's repo table is the *public/private*
> axis; its §"The Second Axis: Layering (game / dApp / chain)" is the
> *layering* axis — which private repo a game concern goes in. **Three tests,
> all must pass** (revised 2026-08-20; the earlier one-question form admitted
> FAME as "value" and ignored write rate):
> **(1) Product** — would a commerce customer of the chain want this in their
> dependency? An NFT is a token, so yes; a quest, no. **(2) Value** — BigInt
> fungible currency, a token, or an authority binding? FAME / XP / items /
> reputation are game scalars, not money. **(3) Rate** — does it fit a Glacial
> tier (≤0.1 Hz)? Binds hardest; `riir-neuron-db` is 1,627× cheaper per write
> and one chain tx at 10⁵ accounts eats 63% of a 20 Hz hot tick.
> Canonical failure: game rules (quest / bounty / crafting / reputation, two of
> them moving no money at all) shipped inside `riir-chain`'s
> consensus-critical program set — `riir-chain` Issues 096 + 097, closed on the
> layering side by `riir-dapps`.

## Resolved issue log (verbatim from pre-compaction AGENTS.md)

## Issue log (resolved)

- **Issue 792 (allocated as 776; renumbered per Issue 791) — the docs gate's CPU self-timing printed a well-formed number that measured nothing on Windows** FILED + RESOLVED + removed
  (2026-09-14, same session; issue file removed at close, this row + git history are
  the durable record; found while landing Issue 775's 18th check). The gate's own
  header already documents one way its CPU total stops being a measurement (a forked
  `times` reports `0m0.000s`) and guards it at ~0. This is the OTHER way, and it
  prints a plausible number instead of a zero: on Windows/MSYS, `times` accounts for
  MSYS children and reports essentially nothing for NATIVE ones, so a run whose work
  is all Python reported **1.26s CPU against a 19.7s wall** — built from `sed`/`tail`
  overhead — while AGENTS.md instructs the reader to cite exactly that figure against
  an M3 series of 13.37s.
- **Issue 776 (the OTHER 776 — dual allocation, Issue 791) — contrastive matched-swap + norm-matched noise interventions, CVRR §2.1/§5.3** DONE T1–T7, removed
  2026-09-15 (research 555, arXiv:2609.06746; commits `be4ff672` + `d936b5fd`; the
  issue file was the other session's live allocation of 776, and Issue 791
  adjudicated the number to this document (weight 16 vs 5). Landed: `perturb_matched_swap` / `perturb_norm_matched_noise{_rows}`
  + `probe_matched_swap{_into}` / `probe_norm_noise{_into}` (zero-alloc, caller
  scratch) + the battery's sixth `norm_matched` arm + `LatentSpace::norm_matched_noise`
  ( katgpt-core `interpolation_geometry`, opt-in) + bench cost rows (matched_swap
  0.37 µs / norm_noise 46.2 µs at n=4096 — both ≪1 ms, 21–2700× headroom); 73 tests
  green, clippy `-D warnings` clean, default-off unaffected. The norm-matched arm
  separates norm-readers from structure-readers (magnitude-preserved,
  structure-destroyed noise). Downstream consumer of the slice form:
  riir-neuron-db's `KarcWoutSpace` (norm_matched_noise_slice, katgpt-rs `33f77706`).

  T1, one child at a time, each burning ~2s of CPU: MSYS
  `bash -c` loop -> children **1.796s user + 0.468s sys**; `py -c` -> **0.000s +
  0.015s**; python.exe by ABSOLUTE PATH -> **0.000s + 0.045s**. So the interpreter
  shim was NOT the cause and T2 is refuted by measurement — resolving a real
  executable recovers nothing; it is the MSYS/native boundary. The fix DEVIATES from
  T3's proposal (a CPU-vs-wall ratio): this gate's own history is 12.65s CPU on a
  299.1s wall — 4% — so no ratio separates an unaccounted platform from a busy box.
  It CALIBRATES instead, the house instrument-alive idiom: burn a known 0.25s of CPU
  in a child of the RESOLVED interpreter, require `times` to have seen at least half,
  and print `CPU SUPPRESSED` + wall-only + the remedy when it did not (T4). Both arms
  measured on the same box: native child -> 0.000s seen, suppressed; MSYS child ->
  0.358s seen, figure printed. Cost ~0.35s per run. Every check's VERDICT was
  unaffected throughout — 18/18 green before and after; only the timing line was
  wrong.

- **Issue 775 — `platform_dead_code_audit.py` had no VERDICT half, and nothing automatic ran it** RESOLVED + removed
  (2026-09-14; issue file removed at close, this row + git history are the durable
  record; filed the same day the report half landed, `a0cbc398`). The report was a
  snapshot nothing defended — and this class in particular is invisible to every
  automatic lane the workspace runs (`full_gate` is macOS/aarch64, `wasm32_gate` is
  wasm32, and the x86_64-native lane that emits the `dead_code` warning is a
  WORKSTATION lane), so an instrument that closes that hole and is itself run by
  nothing had moved the hole one level up. Landed, both halves of the family shape:
  (1) `scripts/platform_dead_code_floor_gate.py`, katgpt-rs-scoped, in `docs_gate.sh`
  CHECKS (18 now) with its three globs in BOTH hand-duplicated trigger `paths:` lists
  — pins in `scripts/platform_dead_code_floors.txt`: `max_findings = 0`, two blindness
  floors (`min_rs_files` = the WALK, `min_candidate_decls` = the PARSE, which is the
  one that moves when the token pass breaks on an unchanged tree — it did three times
  during the classifier's construction, each as a confident `0 findings`), and the one
  MOD-REF row by MEMBERSHIP, path + name, deliberately WITHOUT the line number so an
  unrelated insertion above it cannot red the gate. (2)
  `scripts/platform_dead_code_drift_sweep.py`, workstation, every contract repo,
  pinned in `scripts/platform_dead_code_drift_floors.txt` (16 measured rows; katgpt-web
  / riir-dao / riir-deployer / riir-esp32 deliberately have NO row — a floor nobody
  measured certifies nothing — and arrive as UNPINNED reds on the first full-checkout
  run). Three things worth carrying forward: the sweep takes its population from
  `repo_set.txt` as well as from the walk (via the shared `partial_clone_state()`), so
  a partial box DEFERS loudly under `DOCS_GATE_PARTIAL_CLONE=1` instead of greening
  over 16 of 20 — verified in both postures; the sweep cross-asserts its katgpt-rs row
  against the gate's own pins rather than trusting two files to agree, INCLUDING
  `max_modref == len(pinned MOD-REF names)`; and the gate canaries its OWN pin
  arithmetic (6 arms, both directions, including a MOD-REF SWAP whose count stays 1 —
  the arm a cardinality pin cannot fail), because the classifier's 24-arm self-test
  pins the classifier and never touches the comparison. `--prove-fires ea4c2873` runs
  by DEFAULT in the sweep and is opt-in on the gate: ~5.6s of `git archive` to re-prove
  a fact about a frozen commit is worth a workstation run, not a per-push one.
  Measured at landing: gate ~6.2s (2415 .rs / 29819 candidate decls, 0 findings, 1
  MOD-REF), sweep ~30s over 16 repos, docs gate 18/18 green. ⚠ The docs gate's own CPU
  self-timing read **1.26s against a 19.7s wall** on this Windows box — not comparable
  to the M3 series (13.37s at 17 checks) and not caught by the gate's ~0 guard; filed
  as Issue 792 rather than silently re-pinned.

- **Issue 765 — the docs gate's 3 workstation-only instruments red with a MISLEADING remedy on a partial clone (the 4090 box class)** RESOLVED + removed
  (2026-09-13; the `DOCS_GATE_PARTIAL_CLONE=1` marker axis; issue file removed at close,
  this row + git history are the durable record; filed by the 4090 session while
  resolving Issue 764). The trap: on a box carrying 14 of the 20 contract repos,
  `skill_repo_set_gate` / `population_sync_gate` / `issue_citation_gate` red, and the
  first two printed "regenerate repo_set.txt and commit" — executing that on the
  partial box deletes the 6 absent-but-live repos from the canonical set. The fix
  DEVIATES from the issue's auto-detect proposal (disk ⊂ file → defer) deliberately:
  a genuine removal whose `repo_set.txt` update was forgotten is set-identical to a
  partial clone from the walk alone, and an inferred green would ship the stale file —
  the same reasoning `issue_citation_gate`'s docstring already records for
  DOCS_GATE_CI ("never auto-detected, so a blind WORKSTATION run still refuses").
  Landed, per instrument: (1) a known-partial box exports `DOCS_GATE_PARTIAL_CLONE=1`
  and gets a loud instrument-alive DEFERRAL on the population axis — predicate
  agreement + the local axes still run, the deferral rides each check's FINAL line
  (docs_gate.sh forwards `tail -1`); the citation gate reuses `ci_deferred` with a
  `posture` param (the marker defers the cross-repo half even ABOVE the floor — a
  15-of-20 box would otherwise run the full path and manufacture MISATTRIBUTED rows
  for citations naming the 5 absent repos, who cannot own anything); (2) gone-only
  disagreement WITHOUT the marker reds naming BOTH hypotheses (partial clone →
  marker; stale snapshot → regenerate) with the corruption warning inline; (3) a
  repo on disk the snapshot does not know reds in EVERY posture, marker or not.
  Shared helper `partial_clone_state()` in `skill_repo_set_gate.py` (the fence-scanner
  import precedent), `WORKSPACE_ROOT` test override added to the other two (the
  pre-existing skill_repo_set precedent) enabling the symlink-farm sim. Validation:
  M3 full box 17/17 green in both postures (marker inert when walk == snapshot); a
  14-of-20 symlink farm — unmarked: 3 reds with the new two-hypothesis remedy;
  marked: 3 deferred greens; negative guard: an unregistered repo on the farm reds
  skill_repo_set + population_sync even WITH the marker; CI single-checkout paths
  (DOCS_GATE_CI / unmarked refusal) byte-identical in behavior. 4090 usage:
  `DOCS_GATE_PARTIAL_CLONE=1 ./scripts/docs_gate.sh` (shell profile or per-run).

- **Issue 763 — signed-graph LIF reservoir primitive, event-driven sparse propagation (fly-connectome survey fusion)** RESOLVED + removed
  (2026-09-13; landed `74fe08f1`; issue file removed at close, this row + [Bench 760](.benchmarks/760_lif_graph_goat.md) + git history are the durable record).
  The fly-connectome architecture class as a katgpt-core primitive behind opt-in
  `lif_graph`: `SignedAdjacency` (weighted signed CSR, sign pre-folded — the RuVector
  layout) + `LifParams` (Shiu et al. Nature 2024 canonical current-based LIF constants,
  exact exponential integration: `a_m`/`a_s`/`c_gs` precomputed, 3 muls per active
  neuron per tick) + `LifReservoir` (timing-wheel delay ring, PSP-mV weight units
  `w·t_mbr/tau_syn` — single-synapse 0.275 mV vs the 7 mV gap = coincidence
  detection, ~26 synchronous synapses to fire) + `fit_readout` consuming
  `linalg::ridge_solve_direct_f64` (the KARC precedent — closed-form, gradient-free;
  `lif_graph` joined the linalg cfg any-list at birth, the Issue-701 class).
  **The design's core: the exact-parity active set** — quiescence = the bitwise fixed
  point of the shared per-node update, so skipping a quiescent node IS the identity:
  event-driven `step` and dense `step_dense` produce bit-identical trajectories (not
  tolerance-based — the RuVector dropout is approximate by design). G1: 7 tests across
  sparse-burst / cascade-saturation / chain-ring regimes (per-tick spike sets + final
  v/g/refrac bitwise; run-twice determinism; exact-delay chain 0/18/36; Maslov–Sneppen
  degree preservation; never-hit nodes bitwise at rest; active-set churn + drain).
  Two parity/alloc hazards caught by the gates during development, both recorded for
  the class: (1) **spike-order ULP accumulation** — the event path originally scheduled
  ring writes in active-set visit order while dense collected ascending; two spikers
  hitting one target accumulated `g += w` in different orders → 2-ULP divergence (the
  G1 gate caught it at g[295]); fix = canonical ascending spike order both paths.
  (2) **the ring-slot alignment rule** — a G4 fixture failed with 6 recurring allocs at
  every fire tick: one bucket's amortized doubling 4→…→128, recurring because the
  period (1400) was not a multiple of the ring length (18), so the schedule slot
  shifted 14/period cycling through 9 lazy-priming buckets; fix = slot-aligned period
  (1404 = 18×78). G2 (Bench 760, 4090 host CPU): vs the classic dense-W·spike matvec
  baseline **97,285× at N=10k/3.1% active** (gate ≥3×); vs the CSR full-scan arm
  **34.8×** at 3.1% active (the honest active-set axis); saturated cells 0.91×/1.08×
  — parity, the RuVector saturation-collapse class measured not asserted. G4: 0
  steady-state allocs on both step paths (2,800-tick warmup, slot-aligned stationary
  orbit, non-vacuous 60+ spikes). Controls ship as builders (`er_matched`,
  `maslov_sneppen` — degree preservation pinned). **Stays OPT-IN** per the
  consumer-first promotion rule (karc_lod_tier/hebbian precedents); the named consumer
  path is the riir-ai per-archetype circuit shard + per-NPC readout (riir-ai Research
  379 §7). Feature catalog §108; README/examples claim sites 596→597 (default count
  unchanged at 200 — lif_graph is opt-in); count_features.py all-green at 597.

- **Issue 764 — `LoraPair` KV-cache weight-epoch contract unspecified (public twin of riir-ai Issue 938)** RESOLVED + removed
  (2026-09-13; landed `49f5d245` + `aa163896` + `a7d6d8f0`; resolution verified on the 4090 session's issue sweep —
  issue file removed at close, this row + git history are the durable record).
  All substance landed the same day it was filed, minutes apart: `49f5d245` —
  `WeightEpoch` (BLAKE3 weight-identity epoch) + `LoraAdapter::weight_epoch()` +
  the `LoraPair` by-design mixed-epoch acceptance docs + the `core_04_prefill.rs`
  swap-site documentation (issue items 1+2); `aa163896` — `WeightEpoch::from_parts`,
  the general tagged weight-walk constructor (item 1's remaining half, the
  frozen-model axis riir-ai's Gemma2/KVCA v2 lane consumes); `a7d6d8f0` —
  `WeightEpoch::as_bytes`, the raw-identity accessor the KVCA v2 wire header
  carries. Verification on close: all five surfaces present at HEAD
  (`katgpt-types/src/lora.rs` :518/:540/:554/:563 + `core_04_prefill.rs`
  :294-300). Item 3 (the optional epoch-tagged-cache refusal-arm property gate)
  NOT taken — optional per the filing, and the exactness-refusal consumer lives
  in riir-ai (its `CpuInferenceBackend` seam, riir-ai HISTORY §Issue-938);
  nothing actionable remained here. riir-ai-side consumption: riir-ai Issue 938.

- **Issue 743 — `gw_alignment`, the Gromov–Wasserstein quotient-alignment primitive** RESOLVED + removed
  (2026-09-11, `ae04a98b`; Plan 594 + Bench 709 — the plan/issue/bench carry the full narrative).
  `katgpt_core::gw_alignment` behind opt-in `gw_alignment` (zero deps, pure std):
  structure-only GW alignment of two distance matrices (n,m ≤ 64, uniform weights) via a
  deterministic multi-start (greedy-on-pairing-mass, entropic softmin, uniform+tilt, brute-force
  best permutation at square n ≤ 8) → product-graph power iteration with fixed-schedule
  checkpointing → sum-exact f64 loss + `sigmoid(−β·loss)` score. G1–G4 ALL PASS (11 gates:
  analytic 2×2 + permutation-dominance n=4 + planted recovery n=8 + direct-form identity;
  per-level planted-vs-shuffled dominance ≥13/16 + AUC ≥0.93; default build compiles the
  module to nothing with the default test count unchanged; zero steady-state allocs).
  Stays opt-in per the consumer-first rule — the riir-poc consumer PoC (zone-belief quotient
  alignment → social KG triples) is the promotion path, filed as riir-ai Issue 912's tail.
  Measured evolution vs the issue sketch: the greedy second-order init is load-bearing
  (uniform-start power iteration is saddle-blind); the sketched 2-opt polish on ΣP was
  removed (walked 0.0016 → 0.145 — ΣP is not the GW objective). Downstream: riir-ai
  Issue 912 T4's consume-vs-build decision landed on BUILD; this primitive is it.

- **Issue 742 — the last 42 `#![cfg]`-gated targets in this repo reported a green zero; `SILENT-NOW` is now a WALL at 0** RESOLVED + removed
  (2026-09-09, `2ae0d20a`; the workspace arming sweep's katgpt-rs slice — ndb 616, chain 138, riir-ai 906 landed the same day).
  `SILENT-NOW 42 → 0`, `max_silent_now` re-pinned **42 → 0 as a WALL**; rows DERIVED from the instrument, not hand-typed.
  Issue file removed at close, this row + git history are the durable record (read the filing with
  `git show 2ae0d20a^:.issues/742_the_last_42_gated_targets_reported_a_green_zero.md`).
- **Issue 741 — alloc gates were unrunnable in the shipped profile + the auditor read one of N cfgs** RESOLVED
  (2026-09-09; fix `da498fa6`, free_gib crash follow-up `3115f7f4`, floors/auditor records in the same arcs;
  issue file removed at close, this row + git history are the durable record — read the filing with
  `git show d43a0dea:.issues/741_alloc_gates_only_measurable_in_a_profile_nobody_ships.md`). This repo's two
  `#![cfg(debug_assertions)]`-gated alloc targets now RUN and PASS under `--release`; the auditor reads every
  whole-file `#![cfg]` (rustc ANDs them). Narrative: `AGENTS.md` §"cfg-gated targets — the green-zero rule"
  (both traps, the capability-vs-profile-property test). Three sibling instances filed in their own repos:
  riir-game-sdk `.issues/028` (`2380fc7`), riir-clippy `.issues/083`, riir-dao `.issues/003` (open, owner-gated).
- **Issue 741-as-filed — `is_load_bearing` cannot name a security gate that is named after its THREAT** RESOLVED
  (2026-09-09; fix `ba26462b`; issue file removed at close, this row + git history are the durable record —
  read the filing with `git show d43a0dea:.issues/741_load_bearing_vocabulary_misses_the_threat_dialect.md`).
  **Numbering note:** the filing and the alloc-gates issue (`741_alloc_gates_only_measurable_in_a_profile_nobody_ships.md`,
  fix `da498fa6`) DUAL-ALLOCATED 741 within one evening; `numbering_gate.py` caught it on its next docs-gate run.
  The bare number stays with the alloc-gates issue (blast-radius tiebreaker: landed fix + floors/AGENTS/source
  citations vs one file); this finding is cited by its file path + the `d43a0dea` hash everywhere durable.
  Finding: every token in `LOAD_BEARING_TOKENS` named a property the file ASSERTS, while an entire naming
  convention names the THREAT the file DEFENDS AGAINST — riir-game-sdk's `prod_l<tier>_<threat>` drill suite
  classified **0/31 load-bearing** while `prod_l4_security_rejections` (same repo/directory/tier/purpose)
  classified, decided by whether the author wrote "security". Measured over 2,328 target names in 27 repos.
  Fix: 11 ADMIT tokens (`forgery` ×2, `mitm` ×1, `anticheat` ×1, `chaos` ×4, `crash` ×4, `agreement` ×4,
  `finiteness` ×2, `partition` ×1, `sigkill` ×1, `overflow` ×1, `fuzz` ×1) + 3 bigrams (`crash_replay`,
  `divergence_injection`, `front_run`), each with its measured hit count in the source comment; the homonym
  REJECT table (`replay` 10 perf-probe homonym · `divergence` 9 measured quantity · `injection` 3 bench
  technique · `rejection` 1 rejection-sampling · `watermark` 2 senses, `agreement` covers the gated one; zero-hit
  `tamper`/`spoof`/`dos`/`adversar`/`byzantine`/`exploit` RESERVED) recorded in
  `.docs/10_audits/cfg_gated_silent_zero_pass.md` §T4f. Impact **0 → 0**: `silent_now_load_bearing` stayed 0 in
  all 17 repos, `scanned`/`gated` unchanged — the counterfactual is the finding: the 31 pre-`2380fc7` names
  replay **11** load-bearing SILENT-NOW against a `max_load_bearing = 0` wall that never said a word. Floors file
  carries the caveat: "arming those is churn" and "the classifier cannot read the name" are indistinguishable
  from inside the report. The third instance of the vocabulary-gap class in this file's history (713 T4c, 728,
  this).

- **Issue 740 — Regime-Probe Primitives: Entropy Gap, Basin Probe, Gardner LUT (arXiv:2604.26841)** RESOLVED
  (2026-09-09; T1–T9 landed, all `[x]` — issue file removed at close, this row + git history are the durable
  record. Landed behind the opt-in `regime_probe` feature as `crates/katgpt-core/src/regime_probe/` (impl
  `781264aa`, docs `53fc8b90`; bench record [`.benchmarks/702_regime_probe_goat.md`](.benchmarks/702_regime_probe_goat.md)):
  per-position conditional entropy of a categorical via the SHARED kernel `katgpt-types::simd::logsumexp_parts`
  (factored from `breakeven/fidelity.rs::cross_entropy`, delegation bit-identical — the reuse mandate held, zero
  parallel entropy code anywhere); two-sample entropy-gap detector with BLAKE3 artifacts (`KRPG`); corrupt→renovate
  basin probe over the `FrozenRenovator` trait seam (eq-12, seeded Fisher–Yates, `fastrand::with_seed` — the
  `data_probe::markov` RNG convention); Gardner capacity LUT (`OnceLock` 4096-pt uniform-log-γ grid + quadratic
  interpolation, Φ via A&S 7.1.5, golden-vs-bisection worst err 5.2e-10 vs the 1e-6 gate). GOAT: G1 PASS (Fig 1B
  cross-over — memorizer gap 1.376→−0.006 nats across the capacity crossing, generalizer 0.004 flat, crossover
  0.375 vs 0.125), G2 PASS WITH CAVEATS (Unwhitened Hebbian-correlator ρ_c ≥ bound at γ=1/4; scope boundary
  recorded — the Whitened interpolant fails the CLT premise outright, needle basins 64→865, so the bound is a
  correlator result not a readout-family result), G3 PASS (bit-identical artifacts), G4 PASS (0 bytes steady
  state). UQ floor (T9): bare detectors, conformal-naive exemption recorded, re-affirm at promotion if calibrated
  intervals ever appear. FIRST CONSUMER LANDED 2026-09-09: riir-clippy Issue 077 T1+T4 (`1994a8a`+`ffdd7a9` —
  `score_bench` forwards `katgpt-core/regime_probe`, `src/score_bench/ood.rs`, additive JSONL field, no heal.rs
  edit, shared-worktree blob-commit for their dirty Cargo.toml); measured run #84: gap **−0.729 nats** / d −1.11
  → **anomalous_negative** — the OOD-labeled gen6 set shares provenance family with the seed corpus, so corpus
  proximity, not the held-out label, decided the gap side (both readings in riir-clippy
  `.docs/08_benchmarks/entropy_gap_ood_axis.md` + the Bench 702 addendum `0a9994fd`). PROMOTION: stays opt-in —
  the consumer requirement is met but convergent validity is not yet demonstrated; unblock = a provenance-disjoint
  reference/OOD pairing in score_bench fixtures + re-read of Bench 702 + 077's convergent-validity GOAT (the open
  thread lives in riir-clippy Issue 077, whose T2/T3 margin-weighted mining also remains). Engine serving-health
  audit recorded as an OPTIONAL second consumer — T8's "and/or" is satisfied by the score_bench axis alone.
  Closeout: gates at close — katgpt-core lib 2001 passed @ `regime_probe` / 1979 @ default, clippy `-D warnings`
  clean both feature sets, docs_gate 14/14 (impl-time); consumer suite 1396/0 at `score_bench` (riir-clippy).

- **Issue 738 — the wasm32 lanes compile what they NAME; nothing checks that what they name is the whole surface** RESOLVED
  (2026-09-08; T0–T3 landed — `scripts/wasm32_surface_audit.py` (POSITIVE-cfg predicate, comments excluded, derived
  population from BOUNDARY.md + a `.git` dir so throwaway worktrees are not double-counted; NAMED/UNRESOLVED/UNCOVERED
  buckets with the walk size printed under the verdict); T1 resolved 14/15 UNRESOLVED packages on ROW-BEARING static
  evidence (two resolver shapes, four-way canaried); the 15th — riir-ai `riir-examples` — measured UNCOMPILABLE for
  wasm32 (uuid missing the `js` randomness feature) and filed as riir-ai `.issues/894`, a finding not a folding —
  894 resolved the SAME DAY in riir-ai (uuid `js` feature + a real clippy fix the never-linted wasm32 arm was
  carrying + a LITERAL `-p riir-examples` example row in that repo's guard layer 1.22, since a variable row reads
  as derived); the per-package reads also surfaced mmorpg's standalone `warm-tier-do` lane gap, landed same day
  `b23dc52`; T3 excluded riir-ai's vendored `wgpu-hal` fork. Standing headline 23 NAMED · 0 UNRESOLVED ·
  0 UNCOVERED over 23 packages / 191 files / 17 repos (measured 2026-09-08). Three instrument bugs — a confident 0-file walk (Python `\s`
  into POSIX ERE), 17 false UNCOVERED (a derived `-p` list read as the worst bucket), 2 more (a `--manifest-path
  "$unit/…"` lane) — recorded in the issue as the classifier-lessons canon. AGENTS.md §"A lane compiles what it
  NAMES" carries the narrative; issue file removed at close, this row + git history are the durable record.)

- **Issue 737 — nothing in this repo compiled for wasm32; the browser crate had 15 live findings to prove it** RESOLVED
  (2026-09-07; T0–T3 landed earlier the same day — 18 lint lines healed, `full_gate.sh` layer 2b with both simd128
  arms, derived package list incl. the root package, membership-pinned residue, the two wasm32 GOAT targets as
  named lanes, first CI red caught the workflow's missing `targets:` install and was fixed `e0b7c9e0`; T4 landed
  `091d29cd` — `--wasm32-only` (Layer 2b alone, sentinel armed above Layer 1, skipped-lane refusal) +
  `.github/workflows/wasm32_gate.yml`, the ubuntu per-push lane, measured 4m38s cold, three canaries + a full-mode
  14m04s green. AGENTS.md §trigger health names the lane. Issue file removed at close — the nine-repo audit it
  anchored is COMPLETE (its last open item, riir-chain `.issues/130` T2, landed `d44b240a` same day); this row +
  git history are the durable record.)
  **FOLLOW-UP (2026-09-11, `25c89432`): the T4 lane never passed on CI — all five of its runs (34296642292…
  34314937986, 2026-09-09) died on the same refusal** — dtolnay/rust-toolchain installed the wasm32 target into
  STABLE, but the action does not export `RUSTUP_TOOLCHAIN`, so the gate's cargo resolved `rust-toolchain.toml`
  (1.98.1, no wasm32 std) and Layer 2b correctly refused a partial pass: the `targets:` install from the `e0b7c9e0`
  class was necessary but NOT sufficient — the second half is full_gate.yml's job-level `RUSTUP_TOOLCHAIN: stable`
  override, which the new lane copied neither. Fixed by mirroring it (`25c89432`); dispatch run 34578372428 on
  develop PASSED both simd128 arms (`katgpt-core katgpt-moka-wasm katgpt-rs katgpt-types + 2 named targets`) —
  the lane's first CI green. Workspace sweep the same day: the pin+targets mismatch class exists ONLY in this
  repo's two lanes (both overridden now); every other repo's `targets:` workflow has no `rust-toolchain.toml`, so
  install and resolution agree by construction.

- **Issue 734 — a shell gate that ABORTS mid-run reports exit 0** RESOLVED
  (2026-09-07; T0–T11 landed — `rc=$?; cleanup; exit $rc` cannot repair it (the saved rc is itself 0); the
  completion-sentinel pattern landed in 37 scripts across 10 repos (`70eff640` + the T2–T5 waves), 40/41 tracked
  scripts SENTINELLED, the single EXPOSED remainder `riir-ai/scripts/e2e_internet.sh` PROVEN inert (trap registered
  at line 41 of 43, zero abort triggers in the window); verdict halves `scripts/trap_sentinel_gate.py` (this repo's
  docs gate, membership-pinned, canaried both directions) + `scripts/trap_sentinel_drift_sweep.py` (17 repos, two
  floors: max_exposed ceiling + min_scripts walk floor, exit 2 when the instrument is untrustworthy) + the premise
  instrument `scripts/trap_launder_premise_matrix.py` (11 interpreters via docker). Three self-corrections, all
  canary-caught: the REPLACED verdict's 1-of-1 false positive (`trap - EXIT` is a deregistration, not a handler),
  the brace counter mis-reading an awk DATA brace → the UNPARSED verdict (never pooled), and the premise naming
  errexit — not nounset — as the precondition, which over-claimed on 15/41 rows and spun off Issue 735. AGENTS.md
  §"A gate that ABORTS reports exit 0" carries the narrative; issue file removed at close, this row + git history
  are the durable record.)

- **Issue 736 — leakage_probe + cross-space diagnostics: the modelless defender-side attribute-leak audit** RESOLVED
  (2026-09-07; T1–T6 landed — `katgpt-core/src/leakage_probe/` (opt-in, zero deps): `probe()` → `LeakReport` with
  verdict tiers InsufficientAlignment/Low/Elevated/High; cross-space diagnostics (`neighborhood_hit_rate`,
  `alignment_mean_cos`) sharing the probe's kNN kernel; GD-free unpaired transport (deterministic subspace iteration
  → PCA whitening via the shared `linalg::symmetric_eig` → CSLS-corrected entropic Sinkhorn → orthogonal-Procrustes
  polar factor, deterministic multi-start against wrong-basin ICP lock-in). Gates G1 planted-leak + G1b
  monotone-in-noise + G1c honest-negative + G2 smoke 10/10 PASS; T4 novelty deep-search → **Super-GOAT** (a
  modelless cross-space leak SCORE is unshipped in literature and workspace); T5 consumer GOAT landed in
  riir-neuron-db (`51e2ca1`+`6e14f4d`, Bench 495: planted top1 0.828 vs chance 0.086, lift 9.64, monotone
  0.828→0.082 across α 1.0→0.05, probe 380 ms @ n=256); T6 stolen-DB risk-quantifier architecture guide filed as
  riir-neuron-db `.research/308`. Durable record: README feature table + `.docs/09_feature_catalog` §95 +
  `.research/540`; issue file removed at close.)

- **Issue 735 — Issue 734's laundering premise is bash-3.2-ONLY, and "and 5.x" was never measured** RESOLVED
  (2026-09-07; T0–T5 + T2b landed `a95d2bd6` + peers across 11 repos; T3 answered + T4 resolved same day — issue
  file removed at close, this row is the durable record; full narrative + per-commit tally in git history,
  `scripts/trap_launder_premise_matrix.py` is the instrument). The premise: bash aborting under errexit enters the
  EXIT trap with `$?` already 0 — measured **3.2-only** (4.4/5.0/5.2/5.3/dash/busybox ash all preserve); 41 files
  of inherited "3.2 and 5.x" wording corrected across 11 repos; errexit (not nounset) is the precondition; the
  measurement MODE is part of the claim (127 from `bash -c` vs 1 from a script file). T3's runner probe answered
  EARLY via the 737 layer-2b push run `34137014037`: GitHub's `macos-26-arm64` ships bash **3.2.57 ONLY** (PATH =
  `/bin` = `env`; no Homebrew bash in PATH) and reproduces all five errexit LAUNDERS cells — the sentinel is
  load-bearing IN CI, not just on workstations. T4 resolved **do not pin — measure**: both candidate pins resolve
  to the same interpreter on the current image; a `shell:` pin cannot govern the scripts' `#!/usr/bin/env bash`
  shebangs anyway; the sentinel is correct under BOTH bashes and the probe re-measures every run, so drift is
  observed, never silent.

- **Issue 732 — Fresh-z₀ breadth-restart arm + D-first law for `best_of_k_rollouts` (EqR RI axis): FreshZ0 is a decisive quality NEGATIVE; perturbation breadth pays from K=4 at every measured depth** RESOLVED
  (2026-09-07, `8777f6fc` T1 + `d8eae02b` T1–T4; issue file removed at close — this row is the durable record.
  EqR re-audit action item, Research 079 §10). T1 `restart_mode` knob (Perturb default = bit-identical pre-732;
  FreshZ0 = seeded σ=4.0 Gaussian z₀ per rollout) behind `eqr_convergence`; rider: the `MostFrequent` selector's
  HashMap count-tie broke replay determinism — deterministic first-seen tie-break (the bench's own invariant
  caught it). T2 matched-NFE bench (20 trials × K ∈ {1,4,8,16,32}): FreshZ0 collapses quality 0.59 → 0.12–0.37
  at every K and never beats its own K=1 — the DDTree has no pull-back dynamics, so EqR's restart premise does
  not transfer (categorical, not scale-marginal). T3 pre-registered negative control VIOLATED (the fixture
  indicts itself; positive readings void). T4 D-first sweep (D ∈ {2,4,8}; 16/64 unreachable on Config::draft()):
  Perturb+MostFreq breadth-pays at K=4 for EVERY D, agreement saturates 1.00 by K=8 at D=2; FreshZ0 pays nowhere.
  T5 (Δ_PI metric + four-mode proxy diagnostics) deferred `[-]` — revisit only with a shaped-landscape corpus
  (riir-ai Issue 881 composition). Bench: `tests/bench_732_fresh_z0_restart.rs`.

- **Issue 731 — Residual-gated early exit for the weight-tied looped forward (EqR action item 7.2): `LoopResidualExit`** RESOLVED
  (2026-09-09; T1/T2/T3/T5/T6 landed, T4 deferred `[-]` — issue file removed at close, this row + git history are
  the durable record. The loop-EXIT half of the EqR pair (732 was the selection half; lineage Plan 119
  `eqr_convergence`, Research 079 §10). Landed behind the opt-in `cadence_gate` feature as `LoopResidualExit`
  (katgpt-core `convergence_cadence.rs`, T1 `c69e651d`; the all-features-only `forward_looped` call-site alignment
  caught by the first full-gate run on the branch is `c571d5b9`): exit when the L=3 step-norm window mean < τ OR
  the cadence verdict is `Settled`, never before d_min, `None` probe = bit-identical (caller-owned slot, the
  Issue-035 precedent). T2 calibration `a5edd8e6` (pre-reg `c5f45402`): knee k=10, the settle signal LEADS the
  knee (~5-6 vs 10), no τ qualifies at d_min=4, and the Research-440 magnitude-only control false-fired at τ=10
  (boundary amended τ ≤ 3, recorded, not silently). T3 micro-fixture campaign closed REJECTED: v1–v3 (pre-regs
  `4332b056`/`284942d0`/`9c3b6d60`, closed `0fca1390`) with the floor-cap mechanism (quality parity needs
  d_min ≥ ~10, the ≥2× margin needs d_min ≤ K*/2 — unsatisfiable on micro); the named v4 lever (loop-weight scale
  α=3.0, pre-reg `f7a12f5d`, measured `e05dc0c1`) G2 PASS at exactly the 2.0× bar — existence-proof grade by
  scan-selection disclosure. T6 held-out replication (pre-reg `ab59b9a0`, fix `14177e00`, ceiling-refutation
  pointer `30124225`): P1 REFUTED the "exact 2.0× ceiling" (held-out seed 1002 margin 2.40× — the ceiling was the
  scan's coverage, not the axis); P2 = 1/12 inside its pre-declared band ⇒ the existence-proof grade STANDS; P3
  CAUGHT A REAL PROBE DEFECT — the OR'd shape arm's rule-3 decay fall-through false-converged a churning loop
  (the InterLoopNorm control fired 40× on held-out seed 1003), falsifying T1's "guarded by construction" — fixed
  by `with_shape_persistence` (default 2 consecutive Settled windows; `persistence=1` retained as the recorded
  control arm, no loser to demote), a STRICT improvement on every corpus (v4 mean exit dist 8.66e-4 → 1.02e-4,
  max 12× better; control 40 fires → 0) and confirmed on a real workload (riir-ai Bench 887, `dd80296d`: +1
  iteration cost, better max|Δer|, G4 allocs unchanged). T5 consumer seam `with_cadence_config` `e562195d`
  (motivated by riir-ai Issue 881's measured GOAT FAIL on CCE-scale residuals; first-consumer unblock evidence
  Bench 875 `3e328368`). T4 promotion DEFERRED — the synthetic evidence is existence-proof grade and the decision
  flows through riir-ai [Proposal 045](../riir-ai/.proposals/045_cce_margin_gated_commit_rule.md) (CCE
  margin-gated commit rule, owner verdict pending) — the pointer this record must keep alive. Closeout
  2026-09-09 `6010a558`: the T6 ratio-sweep had pinned `decay_ratio_max = 0.0`, an ILLEGAL config under the
  Issue-720 constructor debug_assert, so the pin test panicked under debug_assertions and could only ever have
  passed release-measured; swept to a legal (0.1, false) instead (same mechanism pinned). Gates re-certified at
  the closeout HEAD: cadence lib 2002/0 at `cadence_gate`, `issue_731_t1_residual_exit` 3/3,
  `bench_731_t3_heterogeneous_corpus` 7/7 in BOTH profiles; campaign instruments live at
  `tests/bench_731_t2_residual_calibration.rs` + `tests/bench_731_t3_heterogeneous_corpus.rs`.)

- **Issue 733 — `EngramHotSwap::with_table` did not hold the writer lock: a nested same-thread `swap` dropped the old table under a live borrow** RESOLVED
  (2026-09-07, `31bf0012`; issue file removed at close — this row + the module doc are the durable record.
  Found by riir-chain Plan 046 §2b while forcing an orphan-envelope test through a "locked" hotswap; fix direction 1
  of the three filed). The unlocked pointer load was unsound under TWO interleavings — same-thread nested `swap`
  (the repro the issue named) AND cross-thread `swap` during the closure (the in-code essay's own "NOT formally
  safe under all interleavings" admission). Fix: the closure CASes the writer lock and HOLDS it (panic-safe Drop
  guard releases lock + thread-local depth flag); `swap`'s CAS then fails closed → `Err(new_table)` for both cases,
  swap's drop-safety argument is literally true, and a nested `with_table` panics loud instead of self-deadlocking.
  Cost: one uncontended CAS + one Release store per closure (control-plane readers only — the hot lookup path is the
  cache hierarchy). G5 re-run: 100 swaps / 1.77M lock-holding lookups / 0 torn reads over ~2s. Contract-pinning
  tests: nested-swap→Err + borrow stays valid; nested-with_table→panic; cross-thread swap→Err + borrow unchanged.
  riir-chain side (Plan 047's `EngramIngress`) is the first consumer whose soundness story this enforces by
  construction: dispatch runs INSIDE the `with_table` closure; publish (`swap`) after it now fails closed on any
  contention instead of racing the borrow.

- **Issue 730 — 256K prefill KV-offload double-buffer: T0 measured the wall 4× smaller than claimed; the offload premise is refuted for every lane the stack serves** CLOSED-N/A-PREMISE-REFUTED
  (2026-09-06, T0 verify-first gate; issue file removed at close — instrument kept:
  `scripts/gguf_header_audit.py`, stdlib GGUF header/tensor introspector; riir-ai Issue 879 T2 is its next consumer).
  Measured on `Ternary-Bonsai-27B-Q2_0.gguf` (gguf v3, arch `qwen35`, 851 tensors): **16 full-attention blocks of 64**
  (blk.3,7,11,…,63 — `full_attention_interval=4`, verified against the TENSOR TABLE not just metadata: 16 `attn_output`,
  48 `ssm_out`), 48 GDN. Full-attn KV geometry: head_count_kv=4 × (key_length 256 + value_length 256) =
  4 KiB/token/layer @f16 → **64 KiB/token hybrid = 16 GiB @262,144 (f32: 32 GiB)**; the all-64-layer pre-hybrid
  accounting reproduces R436's ≈67 GB figure (64 GiB) — the wall number never modeled the hybrid. The literal T0
  subject (`dspark-Q4_1`, arch `dspark`, 6 blocks, **context_length 4096**, KV 12 KiB/token → 48 MiB @4096) has no
  long-context wall at all; its `target_layers [1,16,31,46,61]` confirms it reads the 27B's full-attn blocks.
  Verdict: "offload unavoidable at prefill regardless of compression" is refuted for every lane the stack actually
  serves (≤32K league lanes: 2 GiB f16 / 4 GiB f32); at the aspirational 256K lane f16 fits with ~1.3 GiB headroom
  (16 + 6.67 weights = 22.7 GiB vs 23.98) and f32's overflow is answered by the already-shipped
  `QWEN38_KV_DTYPE=f16` lever (riir-ai Issue 753, Bench 756-validated), not a new double-buffer system.
  Reopen trigger: a production 256K prefill lane materializes AND f16-KV's marginal fit fails in practice →
  T1 (serialized-offload pricing) is the first step. Bonus for riir-ai 879 T2 (read en route): `ssm_a [48]`,
  `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm`, and all RMSNorms are **F32** (post-activation gate params unquantized ✓)
  while `ssm_alpha`/`ssm_beta` gate projections are TYPE_142 ternary weight matrices (the paper's pre-activation
  shield applies ✓); census: TYPE_142 × 498, F32 × 353.
- **Issue 729 — the NaN-comparator class never got its katgpt-rs wave: ~160 legacy `partial_cmp` sites + 13 NaN-promoting `total_cmp` positions, in the repo that OWNS `float_order`** RESOLVED
  (T1–T5 2026-09-06, sweep `f2c305dd` + deref stragglers `649ce5fe` + cross-repo close `2bd3e704`; **closeout residue sweep this commit**;
  file removed at close — this row + the `katgpt_core::float_order` module doc are the durable record, riir-ai Issue-878 AGENTS.md row
  carries the class's other half). The 832 T4 sibling sweep never included katgpt-rs, and the earlier total_cmp wave had fixed only part
  of src, leaving benches/examples/tests plus a src tail. **Group A** (~150 legacy-idiom sites / ~120 files) swept to the float_order
  terminals; **Group B** (13 `total_cmp`-in-NaN-promoting-position sites — IEEE-754 totalOrder ranks NaN above `+inf`, so a descending
  sort/selection promotes the corrupt value to rank 0; sharpest named: ruliology `bandit::best_arm`) swept to `cmp_for_max`/`cmp_for_min`
  (+`_f64` twins), bit-identical on NaN-free input so no gate re-baselines. No-dep crates keep documented local fallbacks (attn-match
  `desc_nan_last`, micro-belief's in-source note); `total_cmp` remains correct for sorts (total order — no abort) and binary-search
  probes/`Ord` chains (direction-neutral determinism is the contract).

  **Closeout residue sweep (this commit) — the tail had its own tail, in three classes.** (1) A wider selection-shaped scan (max_by/min_by
  closures + `is_gt` accumulators over `total_cmp`, multi-line-window aware — the census's own Lesson-1 class one window deeper) found **~35
  more production Group B-shaped sites the 13-site census had undercounted**, all now converted: core `external_regret` ×2 (`is_gt` → `>`),
  `cgsp/dual_pool` (NaN-priority eviction), `mcts` (NaN UCB winning the descent), `slod`; forward `d2f` greedy argmax ×2 + `cluster_head`
  + `d2f_verifier` min; kv `cs_kv_probe::argmax`; pruners `bandit` best-arm ×3 + `expression_pruner` + proof `sketch_population::best_elo`
  + `sketch_sampler` ×2 (f64 UCB); speculative `belief_drafter`/`blueprint`/`and_or_builder`/`adaptive`/`ilc`/`trd` + `dd_tree` ×10; and
  ruliology `bandit::best_unpromoted_arm` — **the sibling method of the issue's own sharpest named site, missed by `f2c305dd` in the same
  file: the comment was updated but the twin loop's `total_cmp` was not, so a NaN payoff could be selected as best unpromoted arm AND poison
  `best_payoff`**. All bit-identical on NaN-free input; **all seven touched crates' lib suites count-identical to the T4 baselines** (core
  1974/0 · speculative 305/0 · pruners 126/0 · dec 225/0 · ruliology 93/0 · forward 125/0 · kv 24/0) + workspace `check --all-targets
  --keep-going` clean. (2) An 8th deref/f64 twin from `f2c305dd` itself: `spechop/hop_tree` ×2 passed `f64 confidence` to `cmp_for_max`
  (E0308) — found by the all-features compile, fixed to `cmp_for_max_f64`, the same class `649ce5fe` recorded. (3) The deref-depth lesson
  applied ONE level deeper than 878 recorded it: the sweep's own first pass wrote `*a` where `.enumerate().max_by(|(_, a), (_, b)| ...)`
  binds `a: &&f32` (the closure receives a REFERENCE TO THE ITEM, so destructuring adds a second ref) — rustc's E0308 + suggestion caught
  every one; the compile gate remains the only detector. **Documented leaves (verified remaining, 14 scan hits, all in-class):** test-mod/
  test-fixture sites (dllm_solver ×2, subspace_phase_gate, bigram_markov, moka_int8 ×2, sense ×2, attn-match score_matrix, percepta
  legacy/tests), `dec/heat_kernel` (NaN can win the min but the `< NULL_SPACE_THRESHOLD` guard rejects it — gated neutral arm, now commented
  at the site), `d2f_verifier::argmax_total_cmp` (deliberately named — "branch-free, NaN-deterministic" IS its contract), micro-belief
  `coherence_bench` (in-source no-float_order note). Cross-repo: the riir-ai lane was RESOLVED same day by riir-ai `.issues/878`
  (`1ee35da79`, ~140 sites incl. PRODUCTION civ `map_tick`), which also fixed HERE the deref-depth stragglers `f2c305dd` shipped in
  feature-gated surfaces (`649ce5fe`). Record: this row + the float_order module doc + git history (issue file removed at close).
- **Issue 728 — `silent_now_load_bearing` was 0 in all 16 repos because the classifier speaks ONE repo's dialect** RESOLVED
  (T1–T5, 2026-09-06; file removed per noise-reduction — durable record in `.docs/10_audits/cfg_gated_silent_zero_pass.md` §T4d/§T4e,
  full narrative in git history). T1 widened `is_load_bearing` (6 tokens + `LOAD_BEARING_BIGRAMS` adjacent-pair matching, every
  candidate measured against the full 3,081-name corpus, homonym rejections pinned) taking the workspace count **0 → 12** —
  **two of them katgpt-rs's own**, so `max_load_bearing = 0`, the pin this file calls the one that earns its keep, had been green
  over a population excluding two local targets. T4 landed `cfg_gated_drift_sweep.py` + `cfg_gated_drift_floors.txt`, the fifth
  sweep-family member. **T2 + T3 armed all 12 and RAN each at its own feature set — 49 assertions, 49 pass, 0 fail** (riir-chain
  `81a00607` 25 · katgpt-rs 13 · riir-viewbridge `d6f18f5` 10 · riir-ai `5a6ac2d2e` 6 · riir-train `55754e7d` 5 · riir-dapps
  `dd6261e` 3, which also armed **all 25** of its silent targets — 25 of 30, the workspace's highest proportion — verified 25/25
  BUILD at their own exact feature sets). Silently *unverified*, not broken. Workspace `silent_now` **261 → 225**, `load_bearing`
  **12 → 0 in every repo**, and `max_load_bearing` is now a **WALL**; canaried by un-arming `bridge_spec_match`, which reds both
  ceilings, each naming its own assertion. **T5 is the finding that was not in the report:** "adding the rows cannot red an
  existing CI" is FALSE for a gate that counts green test *binaries* — see §"A green test count can be a count of nothing" above
  for the measured table and the passed-test-floor repair. En-route: riir-ai's two `#[ignore]`d GPU parity tests were EXECUTED
  (`-- --ignored`) rather than recorded as unrun, since the M3 has the GPU and the 7 GB GGUF the reason names.

- **Issue 727 — SP-KV misses BOTH T16 bars once the gate is measured at a realistic sequence length** RESOLVED
  (2026-09-05 `adbc003d`; filed by 723 T7; file removed per noise-reduction — full narrative in git history).
  The repaired instrument (T_N decoupled from `Config::micro()`'s `block_size = 16` — the "50% pruned" arm
  had been pruning 0/16) measured gate-bias overhead +8.0/+8.1/+8.4% vs a <3% bar and prune-skip
  1.046/1.042/1.015x vs a >1.05x bar. T2 hoist landed: `attention_head_core` split into a verbatim NoBias
  impl + a hoisted GateBias impl (64-position chunk scan → active (position, bias) pairs on the stack, zero
  alloc) — prune-skip **1.12–1.58x PASSES its bar** at t_n 128/512/2048 ×3; bit-identity at `to_bits` across
  6 bias cases × 2 head offsets (two documented divergences are unreachable via `build_gate_biases`).
  **T1 verdict: "zero-overhead gate bias" is a false claim** — any gated attention reads the gate once per
  position per head; restated as a measured +7–12% budget (hd=4) in the bench provenance, the primitive doc
  and both Cargo.toml comments; `#[ignore]` KEPT with the updated reason (the issue's second T4 branch — no
  bar re-pinned, G3 held). En-route catch: a single `#[inline(always)]` body measured a **1.66x layout
  penalty on the NoBias baseline**; the `#[inline(never)]` dispatcher split restored it. Record:
  `tests/bench_sp_kv.rs` provenance + the `katgpt-kv` `sp_kv` feature comment.
- **Issue 726 — `gauge_rebalance` is 3.7x its Plan 279 target; the rank-wise accumulate is scalar** RESOLVED
  (2026-09-05 `d225fffa`; filed by 723 T7; file removed per noise-reduction — full narrative in git history).
  T1 priced the scalar accumulate at **~77–78% of the whole call** (stub A/B, 3 interleaved runs). T2 swap to
  `katgpt_core::simd::simd_fused_scale_acc`: t08 best-of-200 **19.21 → 9.00 µs (−53%)**, interleave medians
  54.1/52.7/50.9% (G1 ≥20% ×3 PASS). T3 bit-compared via `to_bits`: **the scalar loop was NOT
  FMA-contracted by LLVM**, so the swap moves results ≤1 ULP (max 1.19e-7) — every exactness assertion passes
  at its committed tolerance (t01 84x headroom; bench_279 G6 + katgpt-sparse 39/39 re-run); no assertion
  loosened. T4: `t08` re-pinned **30 → 15 µs** with the full provenance block; the 5 µs paper target kept as
  aspiration (the remaining floor is the σ-dots + two full-matrix scales + the final ‖M·v‖ pass, ~5.4–6.5 µs
  — not reachable by this swap alone). G3 zero new alloc. Record: `tests/bench_270_gauge_invariant_goat.rs`
  t08 provenance + the `gauge_invariant.rs` comment.
- **Issue 723 — the first full-workspace EXECUTION is red: 47 targets, six distinct classes** RESOLVED
  (T1–T8 2026-09-04/05; **G1–G6 ALL MET**; file removed per noise-reduction — full narrative in git history).
  Doc-tests GREEN (34 suites / 98 passed / 0 failed — the `--all-targets`-excludes-doc-tests axis added to
  this file); Class C resolved by MEASUREMENT (gates pass at their own committed feature sets,
  documented-expected under `--all-features` unification — the Issue-830 twin, never re-pinned); Class E
  closed per-target at committed features. **The load-bearing refutation: "Class A's reds are partly the
  box" was wrong** — load was the top term in none of the eight wall-clock reds; 5 of 8 were closed by
  REPAIRING THE INSTRUMENT (a vanished denominator printing as 30x, two arms that were not the same
  experiment, loop-invariant inputs black-boxed only in the result, in-clock operand regen, a bar measuring
  the fixture), and the rule to carry forward is **repair the instrument first, decide the disposition
  second** — three of the eight would have been re-pinned to numbers off by 5x/7x/140x. The two genuine
  primitive shortfalls were filed as `.issues/726` / `.issues/727` (both resolved same day) rather than
  absorbed into tolerances. Durable artifacts: `tests/common/ab_timing.rs` (interleaved median-of-ratios +
  best-of-N + loud 0-ns FAIL) and `.docs/10_audits/ci_compile_vs_execute_axis.md`.
- **Issue 725 — the numbering gate covers ONE repo; 35 duplicates and 7 broken allocators sat in the other fifteen** RESOLVED
  (T1-T4c 2026-09-05; file removed per noise-reduction — full narrative in git history). The sweep
  (`scripts/numbering_drift_sweep.py`, workstation-only, derived population, committed expectations in
  `numbering_drift_floors.txt`) found katgpt-rs clean while four siblings carried 35 duplicates + 5 malformed
  `.highwater` files (`echo -n` writing its own flag into the file, disarming the above-highwater check) + 2 stale
  allocators. T1 split ABSENT from MALFORMED in `numbering_gate.py` (`read_highwater()` → `(value, malformed_raw)`,
  selftest case 6 canaries the collapse). T3 repaired all 7 allocator defects, pinned 0 everywhere. T4a landed
  `scripts/citation_weight.py` (advisory arbitration instrument; by-name citations tie while `Plan N` carries the
  weight, so ambiguous mentions are ATTRIBUTED by token overlap on a strict margin with an UNRESOLVED bucket that
  is printed and never folded into a winner; a clean-zero gets a loud warning — `.plans/229` scored 0 while two
  citations existed under different spellings). T4b resolved **riir-ai 6 → 0** (`.plans` 175→568, 182→567,
  229→566, 313→569; `.research` 020→362, 148→363 — 86 citation rewrites; four execution lessons recorded:
  section number beats prose on same-subsystem pairs, third cross-repo documents share numbers, select
  inclusively never exclusively, ties break by creation order per `TIE_FRACTION`), **riir-clippy 4 → 0** (its
  Issue 069 `58e7c1d`, 17 citation rewrites), **riir-train 13 → 0** (its Issue 514 `103ed351`, 14 commits —
  hand reads overturned the UNDECIDABLE verdicts; known cost: number-baked test filenames + ~30 source comments
  stay stale for the next code-touching session). The issue's own author was the ratchet's first catch (a
  `513_` allocation from a stale highwater read went red in minutes; renumbered 514). Remaining: **mmorpg-editor
  12, READ-ONLY** to these sessions — ratchet at the measured count, report only. T5 (siblings run the per-push
  gate themselves) deferred `[-]` — `numbering_gate.py`'s pins file is katgpt-rs-scoped; reopen when a second repo
  wants its own per-push gate. Record: the three scripts + `numbering_drift_floors.txt` + this paragraph.
  **The ratchet's next catches landed at this closeout, minutes after the file's removal** — fresh drift the
  sweep found on the verification run, both repaired same-day: riir-clippy 2 stale allocators (`.plans` 85→86
  for Plan 086, `.research` 136→137 for Research 137, both landed unbumped by the prior session — `4db7a18`) and
  riir-train `.issues/511` dual-allocated (the genrm-corpus issue took 511 from a stale highwater read while the
  Sep-04 all-features census held it; citation weight keeps the census — AGENTS.md test-gate row + Issue 513 vs
  zero refs — and the genrm file moved 511→518, highwater bumped — `e938cdc0`). Sweep PASSES at closeout:
  12 tracked duplicates (all mmorpg-editor) · 0 stale · 0 malformed.
  **Next run (2026-09-06, the 4090 box) — one instrument fix + one live divergence, neither a pins change.**
  (1) INSTRUMENT: the sweep exited 2 on Windows — `parse_rows`/`parse_pins` read their pins files with the locale
  codec (cp1252) and died on UTF-8 punctuation (byte 0x86); both now read `encoding="utf-8"` explicitly. The M3's
  UTF-8 default had hidden the class — a gate that only runs green on the author's platform is the platform-axle
  lesson again, now fixed at the read site. (2) DIVERGENCE (owner-gated, DO NOT re-pin): riir-train measured
  13 doc duplicates + 2 malformed allocators (`.plans/.highwater = '-n 374'`, `.research/.highwater = '-n 441'` —
  the PowerShell `echo -n` corruption class, written by `66193bac` 09-03 on this box) — because THIS box has
  `main` checked out while the Issue-514 repair series (13 → 0, `103ed351`) and all newer doc work live on
  `origin/develop`: **90 commits ahead / 6 behind, develop's tip `88a7f563` landed 09-06 05:49 +0700, main's side
  last touched 09-04** — the two boxes are on different branches of riir-train (its AGENTS.md still claims 'no
  develop branch exists'; stale), and a session is active on EACH side. Reconciliation + the main-side corrupt
  highwaters are the owner's; the floors' `dup=0` describes develop's correct state — raising the pin would mask
  the split. **RESOLVED later the same day (4090 box, owner-directed consolidation):** the split is CLOSED —
  main's last commits (`1a6d128c` docs(509) + `823d669c` feat(509)) merged into develop at `936977fa`, the box
  now checks out `develop`, and the sweep re-run from the consolidated tree reads **riir-train files=332 dup=0
  stale=0 malformed=0** (the `-n 374`/`-n 441` corrupt allocators and the 13 duplicates are gone on develop;
  all six `.highwater` files are tracked integers). riir-train's AGENTS.md does NOT claim 'no develop branch
  exists' — that parenthetical was itself stale (the Branch section has documented develop as the default
  working branch since 09-04); `origin/main` stays frozen at `1a6d128c` as lineage, strictly behind develop. (3) ENVIRONMENTAL: katgpt-web / riir-dao / riir-deployer / mmorpg-editor are pinned but not
  cloned on this box — absence rows are box-scoped, not drift; the 12 mmorpg-editor duplicates are
  unmeasurable here. riir-mmorpg-examples `-n 97` remains the known 069 left-for-owner row.
  **Later the same day (second 4090 pass) — the mmorpg row REPAIRED and a second instrument axis caught.**
  (4) INSTRUMENT AGAIN, print side: with the reads fixed the sweep died MID-REPORT on Windows — the status
  glyphs (✓ ✗ · →) cannot encode through a piped stdout using this box's locale codec (cp874; measured
  empirically: ✓ ✗ · → all FAIL to encode, em-dash OK — the staged_set_audit docstring's claim confirmed).
  (6) FAMILY-WIDE, and the pattern CORRECTED: the first fix pinned `encoding="utf-8"`, which `staged_set_audit`'s
  own docstring argues against (mojibake on legacy Thai-console runs); the whole family now carries that
  script's house pattern instead — `reconfigure(errors="backslashreplace")`, locale kept, previously-fatal
  chars degrade to visible `\u2713` escapes, byte-identical output for everything that rendered before.
  21 scripts hardened (the 2 numbering + 19 more printing the failing glyphs: the repo-set pair, the
  cfg/percentile/required-features/docs drift+floor family, citation_weight, count_features, docs_gate_paths_sync,
  feature_isolation_gate, generate_npc_brain_model, orphaned_attr_gate, population_sync_gate, staged-adjacent);
  validated natively on this box — percentile_floor_gate PASSED (5 pins held, 41 sites), docs_gate_paths_sync
  both checks PASS, docs_drift/agents_repo_set/skill_repo_set run to completion. One pre-existing environment
  note: `count_features`/`population_sync_gate`/`cfg_gated_target_audit` import `tomllib` (Python ≥3.11) and the
  box's `python` resolves to 3.10 — but `py -3.14` IS installed and runs all three natively (population_sync
  then fails only on the box-scoped repo_set absence rows, the 4 uncloned repos, not on tomllib). Patch-mechanics lessons: a Python heredoc patcher that inserts `9` lines can silently rewrite
  a file's EOLs — `docs_gate_paths_sync.py` was committed MIXED (155 CRLF + 195 LF) and the first patch pass
  normalized it (+164/−155); redone byte-preserving per-line EOL (+9/0 final). Commit:
  (5) riir-mmorpg-examples `.issues/.highwater` `-n 97` → `097`: directory max is 97 (`.issues/097_*`), so the
  intended value was correct and only the encoding was broken; repaired AND now git-tracked (was untracked local
  state, which is why it persisted per-box). The repair itself was a same-unit collision (Issue-665 class, third+
  occurrence): this box's commit was written while the M3's active mmorpg session landed the identical repair
  (`adc8877`); the push was rejected, fetch showed theirs first (push-wins), and the local duplicate was reset —
  yield cost zero because both repairs are value-identical. Two lessons en route: (a) the stale-fetch trap fired
  — `git status -sb` shows no behind-count without a fetch, so mmorpg (not in this session's primary four) was
  repaired against a stale origin view; the standing rule is fetch BEFORE any work on a repo not touched this
  session; (b) the Windows repair `Path.write_text('097\n')` silently wrote CRLF — the echo-n class in miniature,
  benign (the parser strips) but byte-noisy; on Windows write bytes or `newline=''` when repairing allocators.
- **Issue 724 — `.plans/` numbering collisions regrew after a hand-sweep; nothing gated the allocator** RESOLVED
  (T2/T3/T4 2026-09-04 `24e349e9`/`28c353a1`/`322769b2`; **T4b + T1/T5 closeout 2026-09-04 `866df2a7`**;
  file removed per noise-reduction — full narrative in git history). The tracked `449` collision
  resolved by CITATION WEIGHT, not creation order (Poincaré kept 449 with 27 mentions; ActionBridge
  moved to `587`); both loaded allocators re-pinned (`.plans` 587, `.benchmarks` 701). Two standing
  gates landed: `scripts/numbering_gate.py` + `numbering_floors.txt` (duplicate-number, stale-allocator,
  per-dir population FLOOR, tracked-vs-untracked split; canaried five directions incl. exit-2 pins)
  and `scripts/docs_gate_paths_sync.py` (docs_gate.yml's two hand-duplicated trigger `paths:` lists
  must stay set-identical — drift exits 1 naming each side's globs; the workflow's own LF line
  endings preserved through a binary-safe edit). T1/T5 moot: the untracked `075_riir_ai_m3_campaign_*.md`
  vanished from the tree before renumbering; nothing to arbitrate. Both gates wired into
  `docs_gate.sh`'s CHECKS (now 10) with the paths-sync script globbed as its own workflow input
  (44 = 44, the 713/704 trigger-omission class closed for this file). Record: the two scripts'
  docstrings + `scripts/numbering_floors.txt`.

- **Issue 721 — the root crate registers a `#[global_allocator]` as a library** RESOLVED
  (T1/T2/T4 2026-09-03; **T3 2026-09-04**, the owner sequencing call executed; file removed per
  noise-reduction — full inventory in git history). The lib-level
  `#[cfg(debug_assertions)] #[global_allocator]` is now `cfg(all(test,
  debug_assertions))` (this crate's unit tests only, the katgpt-core house
  pattern): no downstream binary receives an allocator from this crate in any
  feature set. Every alloc-gate consumer target across the 4 repos
  self-registers instead — katgpt-rs via the new `tests/common/alloc_tracking.rs`
  module (14 test targets + the kimi example; 12 Issue-682 force-link blocks
  deleted), riir-train (xhc's T4 five-term guard deleted as predicted;
  bench_558/490 own statics), riir-ai (~50 files: engine lib + 20 tests +
  example off force-links; games-civ's `alloc_delta.rs` now ships the allocator
  AND the liveness sentinel the crate never had; quest's
  `not(quest_compression_draft)` guard dropped; poc's dual-profile macro debug
  arm self-registers; agents GOAT inline). Validated: 14/14 katgpt-rs targets
  compile under exact features + bench_271 9/9 / cross_res 1/1 / issue_717 G4
  1/1 / lib 203/0; riir-train G4 1/1 + the formerly-conflicting kimi arm
  compiles; riir-ai forward_base 3/3, gemma4 ring 4/4, cgsp/evpi 392/0, civ
  canary 2/2, quest tpr 10/1i, poc 22/0, agents 3/3. **The conflict class this
  issue documents is closed at the source — a downstream `#[global_allocator]`
  is now always legal; the Issue-682 force-link pattern (`extern crate
  katgpt_rs;` to keep a library shim linked) is dead and must not be
  reintroduced.** Push order note: katgpt-rs landed before the consumer repos
  (the inverse window is inert counters; this order would have been duplicate-
  registration compile errors). `riir-agents`' katgpt-rs dev-dep is now
  unreferenced (removal = owner call, BOUNDARY.md row).

- **Issue 719 — conditioning-consistency audit PoC (`cond_audit`)** RESOLVED
  (T1, 2026-09-03, `995dea6d`) — opt-in `cond_audit` in katgpt-core: paired
  forward (compressed-conditioned vs full-context teacher) → per-junction
  forward-KL → Pinsker `TV ≤ sqrt(eps_KL/2)` + greedy-flip counter +
  calibrated-zero arm; KL delegates to `stale_residual::kl_logits`. G8
  non-vacuity PASS (planted 12-nat corruption → tv 4.97 ≫ 0.05, flips 8/8;
  control arm exactly 0.0); G2 measured 1.487× the paired-forward cost vs the
  4.0 budget. T2–T4 deferred `[-]` — reopen on any semantic
  eviction/windowing PR, riir-train Plan 343 T1.6 (Gemma-4 ring), or
  Research 523 H2O un-defer. Record: `.benchmarks/700_cond_audit_poc.md`.

- **Issue 739 — katgpt-rs carried no rust-toolchain.toml: the box default
  failed to build HEAD (E0658 on katgpt-percepta `isolate_lowest_one`)**
  RESOLVED (T1+T2, 2026-09-08, `87dfa778`) — `rust-toolchain.toml` pins
  channel 1.98.1 (minimal profile + clippy/rustfmt), mirroring the
  owner-directed 2026-09-04 stack pin; measured: rustup resolves the pin via
  the file, `cargo check --workspace` green, `cargo test -p katgpt-core
  --lib` 1979/0, clippy 0.1.98 clean. **The CI interplay was the load-bearing
  half:** `full_gate.yml` documents a DELIBERATE no-pin rot-gate design (@stable
  so deny-level clippy lints red the weekly run) and `dtolnay/rust-toolchain`
  does NOT export `RUSTUP_TOOLCHAIN` — so the bare pin would have silently
  frozen the rot lane at 1.98.1 forever. full_gate.yml now carries an explicit
  job-level `RUSTUP_TOOLCHAIN: stable` override (rot design preserved,
  documented); `test.yml` installs the pinned channel via `rustup show`
  (measures what consumers build); release-plz + feature-isolation workflows
  converge to the pin unchanged. Premise update recorded: the M3 box default
  drifted to stable=1.98.1 on 2026-09-04 (owner `rustup update stable`), so
  the break reproduces today only on boxes whose default is older — exactly
  the nondeterminism the pin exists to remove. T3 split: README/AGENTS pin
  documentation DONE (same commit); the `workspace.package` `rust-version`
  half deferred `[-]` — this workspace has no `[workspace.package]`
  inheritance table, so resolution-time enforcement would mean touching ~30
  member manifests (owner call if ever wanted; the toolchain file already
  gates every command).

- **Issue 767 — mb_value: bounded three-factor (dopamine) plasticity value circuit** RESOLVED + removed
  (2026-09-13; landed with this commit; issue file removed at close, this row +
  [Bench 761](.benchmarks/761_mb_value_goat.md) + git history are the durable record).
  The mushroom-body architecture class (riir-ai Research 380, distilled from
  adonis-singh/TMNF-C @ `eb6be045`; mechanism after Bennett/Nowotny Nat. Commun.
  12:2569 2021) as a katgpt-core primitive behind opt-in `mb_value`: fixed random
  sparse PN rows → quantile-calibrated ReLU → top-k KC code (`select_nth_unstable_by`
  under the total order (drive, idx) — set ≡ full-sort reference, G1-pinned incl.
  forced-ties) → approach-minus-avoid readout over per-KC distinct MBON synapses;
  the ENTIRE learning machinery is `w ← clamp(w − η·code·RPE·compartment_sign, 0,
  w0)` — online × reward-RPE × context-generalizing × bounded, the quadrant no
  shipped mechanism covers (ridge/Hebbian batch, Elo/Beta context-free, cgsp
  prediction-error). Bounds hold under adversarial RPE (±inf no-op, NaN no-op);
  calibration is measurement (z-scores, quantile θ, 17-step action-gain bisection —
  measured landing 0.4999 on target 0.5 — per-MBON w0 normalization, derived η =
  α/(eff_app+eff_avd)); `w()`/`set_w()` the freeze/thaw seam; no softmax; NO
  connectome data ships (fly()/toy() are shape classes). GOAT G1–G4 ALL PASS
  (Bench 761): corridor value formation r=0.9705 vs ridge-batch floor 0.9998 on
  the SAME codes (margin 0.029 < 0.05), shift arm — online re-adapts 0.9444 while
  the frozen batch fit inverts −0.9945; toy cycle 2.2 µs (1,000 NPCs ≈ 4.4% of the
  20 Hz tick budget); fly code 28.9 µs (2.28× full-sort, sparse), saturated
  honesty line 0.90×. Joined the `linalg` cfg any-list at birth (the Bench-696
  rule; the G1 floor consumes `ridge_solve_direct_f64`). Stays OPT-IN pending
  consumer (riir-ai per-archetype circuit + per-NPC dopamine readout, Research
  380 §8).

## Issue 744 — HRM-Text second-pass modelless extraction queue: CLOSED as resolved-negative (2026-09-11)

The 8-candidate queue (3 ranked GOAT-worthy, 5 consumer-gated; source
Research 547 §Path 0 merge table) closed after a full consumer-hunt pass —
**no graduating consumer found in the workspace**. Every item's disposition
is recorded in the issue file's hunt record at closure (git history,
`git log -- .issues/744_hrm_text_modelless_extractions.md`): #1 negative
(packed cu_seqlens layout has no workspace analogue), #2 negative (no
normalize-then-gate incumbent in `rating`/`beta_lcb`), #3 negative
katgpt-rs-scoped (all `shuffle` hits are faithfulness-probe ablations, not
materialized permutation orders; sibling-repo side unhunted that day),
#4 negative (no bulk trunc-normal consumer; spectral_pencil's Kaiming-uniform
fan-in is by design), #5 gap-real-but-single-consumer
(`drafter_lora::make_lora_random` is the only construction site — hardcoded
Kaiming + zero-B for the TRAINED drafter; the modelless `LoraPair` path
loads, never constructs — menu does not graduate on one consumer), #6/#8
stay want-gated, #7's `evolve_belief` substrate lives riir-ai-side (deferred
to a quiet day there). **Re-file an item when its consumer materializes** —
the ranking + GOAT gates live in the removed file's git history.

## Issue 745 — Margin-gated verification escalation PoC (TriSpec distill): CLOSED as resolved-split (2026-09-11)

Executed as Plan 595 + Bench 711 (commit `a140c223`; the removed issue file's
task ledger lives in git history, `git log -- .issues/745_margin_gate_escalation_poc.md`).
**Split verdict, all 9 gates PASS:** the CASCADE mapping (margin over probe
scores) is REFUTED at ε=0.5% on both toy geometries — sigmoid score-domain
saturation + noise-floor gap overlap make every λ that cuts verifier
invocations also confirm a 3–19% false-flag tail; the TriSpec-faithful ACCEPT
mapping (margin over the draft token distribution) is VIABLE — 50.5%
target-invocation cut at 0.25% regression (tail 0.5%), and the tail=2%
sensitivity control fires. Five recorded findings: (1) Research 548's "margin
operator ABSENT" claim was a vocabulary miss (`SamplerFeatures.margin`
shipped 2026-07-05, Plan 399 era); (2) the trust polarity is workload
geometry — TriSpec's decisive-trust on a cluster world auto-confirms exactly
the lone-spike FPs the verifier exists to reject (`MarginPolarity` ships both
arms); (3) the meta-router `compute_reward` shape is BLIND to the lossy tail
(trusted-wrong and escalated-wrong both score 0 → the ε-violating λ ties the
best arm); (4) a point-estimate feasibility mask starves a good arm on
sampling noise (2 wrongs in 196 pulls = 1.02% > ε → permanent mask → −8.8%
reward) — the z=3 confidence-bound mask is the fix; (5) `margin_gate` stays
OPT-IN pending a live consumer (similarity_inference demotion precedent; the
candidate consumer is the d2f draft-accept loop, an engine-lane owner call
per Research 548 §5). Feature-count claims bumped 588 → 589 across
README/examples (docs_gate count_features enforcement).

## Issue 746 — Looped-Flows modelless extraction candidates: CLOSED, split verdict (2026-09-11)

Executed as Bench 712 (the removed issue file's task ledger lives in git
history, `git log -- .issues/746_looped_flows_modelless_extractions.md`).
**Split verdict, all gates PASS:**

- **Row 2 (marginal-calibrated backtrack): LIVE** — ships as opt-in
  `marginal_rewind` (`crates/katgpt-core/src/marginal_rewind.rs`): rewind
  the CURRENT flow state to an earlier confidence level at the EXACT
  marginal variance (`a = s/t`, `sqrt((1−s)²−(a−s)²)`; identity
  `a²(1−t)²+(1−s)²−(a−s)² = (1−s)²`), γ-dial for continuous depth, no
  clean-point knowledge. Defend-wrong PoC (K=8192 paired stuck states,
  nearest-mode collapse synthetic): calibrated beats additive-at-equal-
  budget **5.71×** (41.7% vs 7.3%), dominates hard restart at LOWER
  budget in the collapse-prone regime (26.1% @ σ=0.74 / 41.7% @ σ=0.90 vs
  restart 20.9% @ σ=0.95); restart WINS the clean-prior regime (50.3% vs
  42.4%) — recorded as the consumer decision rule, not hidden. Honest
  exactness finding: naive full-variance resample TIES calibrated at deep
  rewind (±0.1pp) — the mechanism is the de-commit shrink; the identity
  buys on-manifold level statistics (the `q_sample_step` cousin cannot do
  this without the clean point). Two harness bugs were caught BY the
  measurement (pre-detection segment integrated to t=1; restart arm drew
  an unbiased prior) — the defend-wrong discipline working as designed.
- **Row 1 (anytime commitment schedule): CLOSED** — no consumer with a
  time-grid need exists (tf_loop `DampedEuler` = fixed-β damped iteration;
  `CommittedFieldBlend` = sigmoid frozen weights; `set_diffusion_schedule`
  = reveal-time CDFs, a different family). The 20-line schedule
  (`r_i = Δt/(1−t_i)`, telescoping annihilation) is exercised by the PoC
  harness (G0 asserts the annihilation) and stays there until a
  flow-integrating consumer materializes (AC-Prefix Issue-002 precedent:
  no consumer ⇒ dead code).

Substrate-first record (pre-implementation gate): searched backtrack/
rewind/renoise/DampedEuler/q_sample/collapse-recovery/marginal variants
across the 18-repo workspace — `renoise_ce` (perturb-and-SCORE),
`q_sample_step` (re-noises a CLEAN x0_hat), `saddle_escape` (uncalibrated
eps·u kicks), `cgsp/dual_pool` (proactive routing only) — decision: BUILD
NEW (no x0-free marginal-preserving rewind existed). Feature-count claims
bumped 589 → 590 across README/examples (docs_gate count_features
enforcement). Promotion owner-gated on a live consumer (candidates: cgsp
collapse recovery, stale-belief fog-of-war re-exploration).

## Plan 596 — sliceTCA modelless slice-rank decomposition: COMPLETE + PROMOTED (2026-09-12)

Executed by parallel subagent (coordinator pre-wired Cargo.toml/lib.rs/
bench rows so write sets stayed disjoint; Bench 714 is the GOAT record).
**GOAT G1–G4 ALL PASS → DEFAULT-ON.**

- Ships as `slice_tca = ["subspace_phase_gate", "tucker_factorization"]`
  (`crates/katgpt-core/src/slice_tca/{mod,types,svd,als,rank,tests}.rs`):
  three single-class truncated-SVD factorizations (Eckart–Young via
  `thin_svd_into` through the small-side Gram), closed-form covariability
  classifier (3 unfolding spectra over the shared ‖X‖²_F denominator +
  SIGMOID routing — never softmax), joint deterministic-ALS demixer with
  loading-only block updates (one contraction + ε-floored divide per
  component per sweep; slice matrices frozen at birth so class assignment
  cannot drift), HOSVD joint init (**Tucker's first in-tree consumer**, with
  documented Gram-SVD fallback beyond TuckerConfig's SVD_MAX_RANK=16), and
  canonicalization (unit-slice norm, largest-|·|-positive sign rule with
  first-index tie-break, variance-desc sort with lexicographic bit-pattern
  tie-break).
- Delta vs the paper's SGD fitter: deterministic pure-function contract
  (same input bytes → bit-identical factors; BLAKE3-pinned across 16 calls
  + rebuild) and zero-hyperparameter updates (no LR/schedule/masking
  curriculum). Fitting novelty NOT claimed (ALS lineage cited in docs).
- G1: pure-class routing 12/12 across noise 0→0.2; mixed-class
  [64,128,32] joint loss 0.0263 vs 0.4835 for BOTH naive floors
  (majority-class AND best-per-unfolding single-SVD) — 18.4×, class split
  asserted unflipped (lossy-surface rule). G2: ALS monotone, improves
  HOSVD init 0.0305→0.0263; full fit 29.99ms ≤ 50ms on [64,128,32];
  canonical 2-comp entity slice 0.375µs (4-comp 1.04–1.34µs documented as
  L1-store-floor physics — honest caveat, not gated). G4: 0 allocs on
  shares/route/reconstruct/entity_slice.
- Honest findings: true held-out CV is structurally impossible for slice
  models (every axis indexes free slice parameters; measured flat CV
  surface) — the opt-in "blocked CV" selector is a per-block structural-fit
  plateau grid, documented in rank.rs. Feature-count claims 198 → 200
default-on across README (total unchanged 593; both features existed
opt-in since the scaffold commit ba754109).

## Plan 597 — BMR + EFE-over-models: COMPLETE + PROMOTED (2026-09-12)

Executed by parallel subagent; Bench 715 is the GOAT record.
**GOAT G1–G4 ALL PASS → DEFAULT-ON.** Unblocks riir-ai `.issues/925`
(scientist-NPC fusion).

- Ships as `bmr` (`crates/katgpt-core/src/bmr.rs`): `ln_beta`/`ColumnSums`
incremental cache, bounded `Counts` column store, `bmr_log_evidence`
(Eq 7/9), `posterior_over_models` (Eq 9), `ModelSpace` sparse-Δ engine +
`predictive_posterior_into` (Eq 11 — Δa = ŝ⊗ô touches ONE column, only
that column's Beta terms recomputed), `efe_model_gain` (Eq 10, KL in log
space, caller-supplied scratch), `occam_log_bayes_factor` (Eq 12),
`enumerate_isomorphic_rules` (Eq 14). Anti-FEP scope guard in the module
header: discovery axis ONLY — MOP/HMM control own policy extraction.
- **Substrate dedup**: the Lanczos ln_gamma extracted from best_belief.rs
into the shared ungated `pub(crate) mod special_fn` — ONE ln_gamma in the
crate, bit-identity verified (1992 default tests green including
best_belief's statrs numerics pins).
- G1: BMR vs chained-exact brute-force oracle (no lgamma — structurally
independent) worst |Δ| = 7.1e-14 < 1e-9; three-ball ablation (64 seeds ×
40 trials): full-info-gain 64/64 discovery (KL→0, Occam +36.8) vs
states+params-only 2/64 (≥2 plausible in 62/64) vs random 0/64 — the
paper's qualitative separation reproduced. G2: sparse-Δ 1.92µs/action,
naive full recompute 983.8µs → **513×** (target ≥100×). G4: 0 allocs /
3000 steady-state hot-path calls.
- Honest deviations (bench doc + module header): (1) Research 551's Eq-7/9
transcription carried the `ln B(a_c)` sign flipped — implemented the
corrected exact-Bayes-factor form, arbitrated by the plan's own T2.2
oracle; (2) our tensor encoding yields 81/81 unique rules vs the paper's
79 (their table encoding is unreproducible from the paper text) — pinned
honestly; (3) T4.3 debugging tuned priors (λ=4, ã=4.0), never thresholds;
(4) premature commits = 0 in a deterministic world (the paper's nonzero
mode needs observation noise) — recorded as a tunable, not a bug.

## 2026-09-12 — the citation gate's first CI run was a blind red (promote `35ac604f`)

The Issue 749 citation gate landed 2026-09-12 in a **main-only CI window**
(every push trigger `branches: [main]` since 2026-09-09), so it had never run
in CI when the promote push `c478ab9f..35ac604f` (267 commits) fired
docs_gate — and red it, exit 2: `derived 1 contract repos < floor 15`. The
refusal itself was CORRECT (the ownership lookup needs the 19-repo workspace;
a single checkout would read every cross-repo citation as a false finding),
but it meant the per-push lane could never carry the check as wired: a check
added to CHECKS during a window in which its own lane never fires is
**deployed but never exercised** — the same shape as the `.issues/704`
trigger-rot class, one level up.

Two defects, one fix commit:

1. **`issue_citation_gate.py` — the CI-deferred verdict.** Under an explicit
   `DOCS_GATE_CI=1` marker (set by docs_gate.yml, never auto-detected — a
   blind WORKSTATION run still refuses), the gate verifies the
   locally-decidable axes (pinned documents present, `max_single_digit` width
   bound, `min_citations_scanned` walk floor) and DEFERS cross-repo
   adjudication to the workstation run, saying so in the LAST line — the one
   docs_gate.sh tail-prints on a pass (the skill_repo_set_gate scope-line
   pattern). The workstation docs_gate.sh run remains the adjudicating
   verdict; main only advances by promote and the promote protocol includes
   it (this promote's workstation run was 17/17).
2. **docs_gate.yml — the fourth instance of the trigger-omission class.**
   The four checks added 2026-09-10..12 (trap_sentinel 734 T9, citation 749,
   checks_sync 750, markdown_fence 756) landed WITHOUT their own `paths:`
   globs — a push editing the CHECK itself would not re-fire the gate that
   runs it. All four globs added to BOTH lists (push + pull_request;
   docs_gate_paths_sync.py asserts the two stay identical), plus the job env
   marker.

Same push, second lane: **required_features_touched REFUSED by design** —
`84 selected > --max-rows 24` on a 267-commit promote base (the gate refuses
rather than truncate; run 33990209894 is the 32-row precedent). The named
remedy was executed on the workstation the same day: the 84 selected rows
audited with `required_features_build_audit.py --batch` per package
(isolated `/tmp` target dir) — **0 FAILS / 0 NO-FEAT / 0 UNSEEN across all
8 packages**. The Full gate + Lean proofs + wasm32 lanes on the same push
were green; content was never in question, only the lanes' wiring.

## Issue 755 — a quoted heading in a fence is not an allocation: CLOSED as resolved (2026-09-12)

Landed on the allocation path only, DELEGATING to the canonical scanner
(`620840ce` fenced-heading exclusion via `fenced_lines()`; `3a3d4bc1` the
delegation refactor — `skill_repo_set_gate.fenced_blocks()`, no second
parser; `37bb9cbf` the CI-deferred docs_gate wiring). The asymmetry was
measured on BOTH sides before landing: `heading_allocated()` 0 of 57
headings fenced (EXCLUDE), `citations()` 57 of 2972 fenced (do NOT exclude
— the fenced rows carry the sibling-layout attributions, exactly where
cross-repo ownership is written down most explicitly). Unterminated fences
fail SAFE (empty exclusion set + surfaced; `issue_citation_gate.main()`
exits 2 INSTRUMENT on one) — measured 0 across the 35 scoped documents at
landing. 3 selftest arms canaried both ways: filter removed → 3 red; naive
toggle → 4 red, wrong in BOTH directions (drops a real allocation, admits
two quoted ones). No verdict moved (0 fenced allocations today) — landed
for the direction it closes. The removed issue file's full record lives in
git history, `git log -- .issues/755_a_quoted_heading_in_a_fence_is_not_an_allocation.md`.

## Issue 756 — an unterminated fence swallows the rest of its file: CLOSED as resolved (2026-09-12)

Measured 19 files / 611 swallowed lines across 19 contract repos (5048
tracked `.md`); all 19 repaired the same day, in three repos: katgpt-rs 12
(`1bf768cd` repairs + the `markdown_fence_gate.py` docs-gate row, both arms
canaried; `ed455885` the walk widened to untracked-not-ignored `.md` after
the gate missed its own landing issue file), riir-ai 5 (`a1a205681`,
follow-up `6e73d89c5`), riir-train 2 (`d761a375`, follow-up `ca21b763` —
both the missing-OPENER shape). The reported line is the DANGLING fence,
not the defect: three shapes measured (missing closer / stray fence /
missing opener), discriminated by the first non-blank body line after it.
Independent verification 2026-09-12: `fenced_blocks()` reports 0
unterminated across all 7 sibling files at the sibling repos' origin/develop.
The removed issue file's full record (three-shape table, the two wrong
classifiers, the false-positive analysis that left the shared scanner
alone) lives in git history, `git log -- .issues/756_an_unterminated_fence_swallows_the_rest_of_the_file.md`.

## Issue 749 — a cross-repo `Issue N` citation rebinds to the WRONG document: CLOSED (2026-09-12)

`numbering_gate.py` guards double-allocation inside one repo and cannot see
a citation whose referent lives in another: AGENTS.md's bare `Issue 750`
meant riir-ai Issue 750 while the local highwater read 748 — dangling that
day, silently rebound to an unrelated local issue two allocations later.
Resolved by `scripts/issue_citation_gate.py` +
`scripts/issue_citation_floors.txt` in docs_gate's CHECKS (landing
`e258fdaa`; hardened by `7fdfc554` short-form repo aliases after a measured
~50% classifier FP rate, `60bc76aa` an alias only qualifies if the repo
OWNS the number, `37bb9cbf` the CI-deferred verdict). Population reuses
`numbering_drift_sweep.contract_repos` (the seventh predicate — no eighth
silent derivation); the walk reads git history, not just the worktree, so
the noise-reduction rule cannot strand a live citation; floors on repos
AND citations scanned — a ceiling is green over whatever the instrument
sees. The 8 unqualified rows (riir-train Issue 513 ×3, riir-ai Issue 750
×3, riir-ai Issues 490/493 ×2, each resolved uniquely by title) were
qualified in AGENTS.md + HISTORY.md; revert-probed — un-qualify exits 1
naming the repo, the list-tail probe (riir-ai Issues 490/493) yields TWO
rows because the first scan read only the head and hid 493, both floors
exit 2. The 63 numbers existing both locally and in a sibling are reported
every run, deliberately NOT gated. Spawned Issue 751 (the cross-repo
sweep); this repo's own 750 was allocated one commit AFTER the riir-ai
citation was qualified — the hazard demonstrated live, now held shut by
the gate. Issue file removed per the noise-reduction rule; the full record
lives in git history
(`git log -- .issues/749_cross_repo_citations_rebind_to_the_wrong_document.md`).

## Issue 751 — the cross-repo citation sweep (18 repos katgpt-rs cannot see): CLOSED as resolved (2026-09-12)

The workstation-wide half the citation gate lacked (the per-push
`issue_citation_gate.py` sees this repo only; a citation's referent lives
somewhere else by definition). `scripts/citation_drift_sweep.py` +
`scripts/citation_drift_floors.txt` landed (`d8041de5`): 19 contract
repos / 35 documents / 2,939 citations walked at landing; every repo pinned
min_citations + max_cross + max_in_local_range + max_orphan; the sweep
ASSERTS its katgpt-rs row against the gate's own parsed run. T1's stratified
spot-check measured the FP rate the count must be quoted with: **7/43 = 16%**
(pre-752 corpus). 4 per-repo issues filed from T1's verified rows; the
backlog they tracked was worked off same-day — 0 CROSS across all 19 repos
— the campaign record lives in the floors file's header. Issue file
removed per the noise-reduction rule; the full record lives in git history
(`git log -- .issues/751_cross_repo_citation_sweep.md`).

## Issue 752 — a repo name QUALIFIES a citation even when that repo does not own the number: CLOSED as resolved (2026-09-12)

The qualification predicate asked "is a repo named?" and never "does that
repo own the number?" — a citation naming the WRONG repo read as clean.
Measured before repair: of 368 qualified citations, **45 the reader cannot
follow** (37 WINDOW_ONLY + 8 ADJACENT). The repair (`60bc76aa`) makes
ownership load-bearing: a directory-name qualifier counts only if that
repo ALLOCATED the number (crate hints were already sub-labelled
⛔MISLEADING and counted as findings). Revert-probed; floors re-pinned in
the same commit — a ceiling that fell was ambiguous between repair and a
sharper instrument, and ten rose for the sharper-instrument reason. The
census's 0/45 did not survive Issue 754, which re-rated it to 1/45. Issue
file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/752_qualification_is_owner_blind.md`).

## Issue 753 — the citation rules' own COSTS were recorded once and never re-measured: CLOSED as resolved (2026-09-12)

Two deliberate narrowings had been measured once, written into docstrings,
and read forever as facts — by a module whose own governing lesson is that
a count that matches is still a claim. Measured (`5ea1f40a`): (a) the
`\d{2,4}` width bound is LOAD-BEARING, not free money — widening to
`\d{1,4}` manufactures 51 false heads (single-digit section numbering
inside `.benchmarks` headings — the kind-prefixed `N:` heading shape; the
live example lives in the gate's docstring and the removed issue file) at
0 true ones, measured over every tracked .md in 19 repos; the class is now
pinned `max_single_digit = 0` (exit-2 breach) with a two-sided revert probe.
(b) alias reach = 40-char LEAD re-measured per run: widening buys 0 repairs
and hides true findings (the one row is `chain` inside prose about the
`chain_viz` crate). One prose row repaired in katgpt-web, its ceiling
lowered in the same commit. Issue file removed per the noise-reduction
rule; the full record lives in git history
(`git log -- .issues/753_the_citation_rules_own_costs_were_never_measured.md`).

## Issue 754 — an allocation that exists only as a HEADING is invisible to allocated(): CLOSED as resolved (2026-09-12)

Both of `allocated()`'s walks (worktree + `git log`) see nothing when a
file is created and removed without an intervening commit — the NORMAL
shape for a same-day issue, whose whole allocation record is then the
repo's own `## Issue NNN (date, RESOLVED)` heading. Under 752's
owner-consistency rule that blindness INVERTS: correct prose gets
⛔MISATTRIBUTED (the live case: riir-game-sdk's
`riir-mmorpg-examples Issue 059` — which that repo's own HISTORY heading
proves it owns). `heading_allocated()` landed (`8e0ffaa0`) with a four-arm
two-sided selftest; fenced headings excluded by 755's scanner when that
class appeared. 7 numbers recovered workspace-wide, 123 CROSS rows
retired, Issue 752's census re-rated 0/45 → 1/45 — the lesson that a
census inherits its oracle's blind spots at 100%. Issue file removed per
the noise-reduction rule; the full record lives in git history
(`git log -- .issues/754_heading_only_allocations_are_invisible_to_the_file_walk.md`).

## Issue 761 — a Lean theorem can RESTATE its own definition: CLOSED as resolved (2026-09-12)

Filed and closed the same day, in two commits (`e5d9836c` the report,
`3d5c87ab` the verdict). The class: `theorem X_eq_sum : X = <the body X is
DEFINED as>`, which `decide`/`rfl` discharges for ANY constant values. Four
shipped in riir-neuron-db for months with doc comments claiming each was "the
`merkle_root` guard" — the exact bug it provably could not see — and nothing in
the stack could tell them apart from a theorem that proves something:
`lake build` green, `#print axioms` axiom-free, counted in `proof_gate.sh`'s
audited surface, and `proof_negative_test.sh` **17/17**, because no arm ever
red on them and a pass count cannot report which theorem did not fire. They
were removed by riir-neuron-db Issue 617 (`24957a2`, `387b4fc`); this issue
owns the class-level instrument and the sweep.

**The criterion is where the work is: symbolic equality over LEAF constants.**
Unfold every composite nullary `def`, keep numeral-bodied leaves SYMBOLIC, and
compare as polynomials. Unfolding all the way to numerals instead would compare
`464` with `464` and condemn every sound literal pin — the leaf boundary IS the
classifier.

Two bucket boundaries earned their separation and both would have been wrong to
pool. **CROSS-DEF is not a finding:** `commitmentOffset = RAW_PREFIX_LEN` is
symbolically equal too, but it pins two *independently maintained* definitions
and an independent perturbation arm proves it reds; the removed four had an RHS
that existed only inside the theorem, so deleting the theorem deletes the
duplicate and nothing is left to drift. **HYPOTHETICAL is split from
STRUCTURAL** because it is 199 of 255 theorems, and pooling hides how few
statements the arithmetic pass ever sees.

⛔ **The oracle found the instrument's own defect, which is the whole argument
for having one.** Run against a tree whose answer was known by other means —
riir-neuron-db at `24957a2^` — it reported the right four, but the caveat named
the WRONG repo's symbol: a repo-wide `def` table was unfolding
`Shard.zoneHashOffset` through ExperienceGraph's same-named composite. The
verdict had survived by luck. Defs are scoped per module + transitive imports
now, shadowing applied both ways, and a 2-module fixture pins it (pooled it
reads VALUE-DEPENDENT, scoped it reads RESTATEMENT).

**T1 changed shape when its blocker lifted.** The filing specified a
`docs_gate.sh` CHECKS row, blocked on Issue 750; 750 closed mid-session in a
concurrent one, and re-deriving the design rather than inheriting it showed the
row was never right — CI has a single checkout, so the derived population is
ONE repo, and it is the repo where the class barely exists (katgpt-rs has **1**
composite def; riir-neuron-db has 44). A gate that can only see that is worse
than none because it reads as coverage. Landed as the documented
workstation-sweep shape instead: `restatement_drift_sweep.py` + committed
floors. **A blocker lifting is a prompt to re-derive the design, not a green
light for the design that was blocked.**

Verdict paths proven on the real code, not argued: a planted restatement moves
the count in all 4 repos (0 → 1); a floor above measured REDS; a
pinned-but-absent repo REDS as UNSEEN rather than shrinking the population; an
empty floors file REFUSES instead of passing over zero rows. Two floors, and
the second is the one that bites — `min_theorems` is what moves when a
tokenizer regresses on an UNCHANGED tree, measured during development when a
`:=` tokenized as `:` + `=` took the def table to 0 with the file count
identical.

Standing (2026-09-12): **0 RESTATEMENT-INLINE · 0 IDENTITY** over 4 repos /
68 `.lean` / 255 theorems / 284 examples; 1 CROSS-DEF (keep). The 19 UNRESOLVED
rows + the 6 conjuncts behind the 2 `∧`-of-`=` rows were read one by one and
are all value pins the arithmetic model cannot reduce — a census adjudicated by
READING, so it bounds reader error only, in the direction somebody thought to
look. `[-]` T4 (a `∧` conjunct is only checked when every conjunct is an `=`)
deferred on measurement: 2 such theorems workspace-wide, both decomposing to
UNRESOLVED. Issue file removed at close; this row + git history are the durable
record (read the filing with
`git show 3d5c87ab:.issues/761_restatement_theorem_class_has_no_verdict_half.md`).

## Issue 750 — docs_gate's CHECKS array vs the AGENTS.md table documenting it: CLOSED (2026-09-12)

Two hand-duplicated lists of the same thing with nothing comparing them —
found already drifted (docs_gate.sh said "six" contract-repo predicates;
the script and AGENTS.md's table both say seven, Issue 734 added it).
Resolved by `scripts/docs_gate_checks_sync.py` in CHECKS (landing
`d10202b1`): MEMBERSHIP of script names both directions, never cardinality
— a count that MATCHES is not a checksum over a set; plus QUANTITY WORDS
only, after issue refs are stripped — the lists legitimately differ in
emphasis, but a number may not differ because a quantity is a claim.
`MIN_ROWS = 10` floors both parses — a parser reading zero rows reports
perfect agreement between two empty sets. Revert-probed five ways, every
verdict distinguishable; registering it added a row to both lists, so it
checks its own registration. The number 750 is the one Issue 749 was
about: allocated here one commit after `e258fdaa` qualified the riir-ai
citation — intended demonstration, not coincidence; the citation gate
keeps it that way. Issue file removed per the noise-reduction rule; the
full record lives in git history
(`git log -- .issues/750_checks_array_vs_its_own_documentation.md`).

## Issue 760 — mmorpg-remaster joined the workspace but not `scripts/repo_set.txt`: CLOSED as resolved (2026-09-12)

Filed from the 4090 box (14 live repos) after a docs-gate run surfaced real
drift amid topology artifacts: `mmorpg-remaster/` was a NEW contract
repo (own root BOUNDARY.md + .git) missing from the canonical repo set. The
katgpt-rs-side drift was fixed in the filing commit `bfccffea` (repo_set.txt
evidence-backed typed line + AGENTS.md §Repo count 19→20 + research/
substrate-first SKILL.md prose counts). The M3-side half completed
2026-09-12: the repo is cloned at `/Users/katopz/git/mmorpg-remaster`
(synced, `99064c5`); the full-workstation `./scripts/docs_gate.sh` run at
`5773a614` is **17/17 green** — the three 4090 reds were all partial-clone
topology (6 canonical repos absent on that box), and the M3 walk — the only
box that can validate the axis — derives exactly the 20 pinned repos, so no
`repo_set.txt` regeneration was needed (population_sync + agents_repo_set +
skill_repo_set all agree at 20). Bonus, same day: the repo's late arrival is
the exact shape `markdown_fence_drift_sweep.py` exists for — its first
workspace run caught the new repo's one unterminated fence
(`.plans/005_layer3_reducer.md:600`, repaired `99064c5`) and floored it at
50 (`5773a614`). Issue file removed per the noise-reduction rule; the full
record lives in git history
(`git log -- .issues/760_mmorpg_remaster_missing_from_repo_set.md`).

## Issue 757 — linking_fold detector Option B (the 50 ms @ n=2×1000 remainder): CLOSED as resolved (2026-09-12)

The algorithmic remainder toward the linking_fold detector's original
50 ms @ n=2×1000 budget (the earlier audit-cadence recalibration had been
accepted; Option B was the perf work). Landed in full (evidence: Bench 717;
Research 391 §6; source arXiv:2606.31856): single-pass k-NN
(`select_nth_unstable_by`, no full sorts, squared distances only),
longest-first witness ordering, certified chunk-level Gauss pruning with the
rigorous [total−bound_sum, total+bound_sum] rounding rule (pruned ≡ full,
200 randomized-pair pins), the Y-cycle uniform grid with conservative per-X
reach, and `max_cycles_per_cloud` corrected to the documented median-closest
semantics. ≈31× at the audit point (115.74 → 3.69 ms @ n=2×200, d=8) and the
ORIGINAL budget RESTORED: G2b **28.28 ms ≤ 50 ms @ n=2×1000 linked, |link|=1**,
now an enforced bench row. GOAT re-run ALL PASS; G3 default lib suite 2035
passed unchanged. Verdict: **KEEP OPT-IN** — perf no longer blocks promotion,
but the only runtime consumer (`LinkingFoldCorrector`) consumes the FOLD
(default-on), not the detector; promotion re-opens when an audit-cadence
consumer exists. One deliberate defer recorded in the issue text (the
healer-surface consumer PoC — fusion idea, novelty TBD, no plan until that
PoC exists) and one side-finding (slice_tca release-profile compile — the
Issue 741 class, since closed by full_gate Layer 6b). Issue file removed
per the noise-reduction rule; the full record lives in git history
(`git log -- .issues/757_linking_detector_option_b.md`).

## Issue 747 — ASEntmax modelless mining (damping schedule, derived-k, eviction window, incremental entmax): CLOSED (2026-09-12)

arXiv:2506.16640 / Research 549: softmax needs sharpening ∝ log n (SSMax,
shipped) — α-entmax needs **damping ∝ (log n)^{−0.5}**; our `entmax_1p5`
routing had no length term, so growing candidate sets over-sparsify (the
paper's Copy-table failure mode). Four primitives shipped behind the OPT-IN
`asentmax_schedule` feature (katgpt-attn), all GOAT-gated; evidence in Bench
713 + its P1/P2/P3/P0.7 addenda:

- **P0** `AsentmaxSchedule` (None/Derived/Generalized) +
  `apply_asentmax_inplace` + `RollingSigmaEstimator` (Kamath range law),
  zero-alloc (G4); wired into `EntmaxRouter::with_asentmax_schedule()` —
  `None` default bit-identical to the shipped Plan 106 path (unit-pinned,
  re-pinned on real rows at P0.7).
- **P1** derived support controller k̂ = 4/Δ̂² — length-independence exact at
  the Lemma-2 boundary for all n up to 1M; head-to-head vs the sigmoid arm
  SPLIT (sigmoid wins coverage, derived wins 4× cost efficiency) — no
  replacement, recorded.
- **P2** Prop 6 eviction window — windowed-vs-full entmax bit-identical
  (`f32::to_bits`, 432 adversarial configs); 98.9–99.99% of KV evictable at
  n=1M under Kamath range bounds + ALiBi 8-head slopes.
- **P3** Lemma 1 incremental decode entmax — below-τ pushes O(1) and
  bit-exact; 0.046 µs/step vs 68,448 µs/step full-resort at 512k; 0 allocs
  steady-state.
- **P0.7** (2026-09-12, `ab0d79dc` + `603075f7`) the real-model re-gate: an
  in-repo harness prefills the REAL Ternary-Bonsai-8B (GGUF reader +
  Q2_0_g128 dequant + faithful qwen3 prefill; forward validated against
  llama-perplexity 16.9614 vs 16.9778 reference, 0.1%) and replays 1856
  committed routing rows through the router. **Measured verdict: NO
  modelless quality gain on the real path** — real σ̂ = 0.1409, an order
  below the σ ≥ 1 over-sparsification regime; raw support healthy 4–7 (no
  collapse at n ≤ 32); needle retention parity 96.3%/96.3%; mean oracle-mass
  −0.017 scheduled; G3 latency +4–8%. **STAYS OPT-IN** (feature-gate-audit
  discipline — promotion requires a gain); the wiring ships so large-σ
  regimes can arm it with one builder call. Harness + fixture reusable
  (`tests/asentmax_p07_realmodel_regate.rs`, 6/6; the 27B is a qwen35 GDN
  hybrid — infeasible in-repo, recorded as the honest 8B partial).

P4 stretch (T4.1–T4.4) deferred to Issue 762. Issue file removed per the
noise-reduction rule; the full record lives in git history
(`git log -- .issues/747_asentmax_modelless_mining.md`).

## Issue 762 — ASEntmax P4 stretch (HoldConcentration, Kamath regime detector, per-head grid, RoPE cutoff): CLOSED (2026-09-14)

Origin: Issue 747's P4 deferral. Terminal states, all four tasks decided:

- **T0 long-context re-measure DONE** (2026-09-13, Bench 713 long-context
  addendum; `asentmax_long_context_regate` gate, 5 tests over the committed
  11,093-token / 173-block fixture): σ̂ climbs with n (0.14 → 0.35) but
  SATURATES below the σ ≥ 1 over-sparsification regime at every measured
  bucket (8 → 173) — the T4.3/T4.4 premise surface does not exist on real
  paths. Second finding: the scheduled arm wins mean oracle-mass at every
  n ≥ 24 (+0.034 overall, +0.05–0.07 at n ≥ 64) and doubles deep-needle
  top-8 retention (59.3% vs 29.6%) — budget-confounded (support 2.1×),
  recorded honestly in the bench.
- **T4.1 HoldConcentration DONE** (2026-09-13, Bench 759 G1a):
  `SsmaxMode::HoldConcentration{c,k}` — Lemma 2's softmax side, the exact
  finite-n multiplier `ln((n−k)c/(k(1−c)))/Δ̂`; Fixed/Adaptive
  bit-identical (re-pinned); 7–30 ns/call; ships in the default-on ssmax
  module (Adaptive-variant precedent, zero cost unless constructed).
- **T4.2 logit_regime DONE** (2026-09-13, Bench 759): the Kamath range-law
  detector `ρ = Δ̂/(2σ̂√(2 ln n))` + normalized-entropy dispersion
  (katgpt-core, opt-in feature `logit_regime`) — independent two-pass moment
  σ̂ (only the ratio detects); Gaussian band [0.35, 1.15]; 8.2–8.6 ns/elem;
  0 allocs. Consumers: the ASEntmax arm decision + the equal-budget reopen
  axis.
- **T0.1 promotion-review DECIDED (owner, 2026-09-14): option (a) —
  `asentmax_schedule` STAYS OPT-IN.** The long-context addendum qualified
  the P0.7 "no gain" verdict but the gain remains budget-confounded —
  promotion is not justified by the evidence (feature-gate-audit
  discipline). Reopen condition (measurable, stands): an equal-budget
  recall axis or the P1 derived-k controller comparison must show the
  SELECTION (not just the 2.1× support size) wins before any default-on
  proposal.
- **T4.3/T4.4 CLOSED-deferred** — the σ ≥ 1 regime does not appear at
  n ≤ 173 (σ̂ saturates 0.35); reopen only if a real large-σ surface or a
  longer-context regime appears — re-measure before building.

Issue file removed per the noise-reduction rule; the full record lives in
git history (`git log -- .issues/762_asentmax_p4_stretch.md`).

## Issue 772 — config-audit first pass over katgpt-rs (7 inert/assert-only knobs + orphan-report layer): CLOSED (2026-09-14)

The 09-13 14-repo `--config-audit`/`--orphan-report` consumer sweep
excluded this repo as sibling-hot; the first in-repo runs (2026-09-14,
riir-clippy Plans 124/126 instruments) filed 7 verified class-A/B knobs +
an orphan-report layer (0 orphaned / 33 stillborn → S1–S6). Every fix
landed same day, wire-vs-delete adjudicated per finding:

- **A1/A2 WIRED** (`3c3c52ce`): monopoly `execute_turn` reads
  `GameConfig.max_jail_turns`/`max_doubles` (the consts had shadowed the
  knobs at both logic sites); two wire-proof tests discriminate — red
  under the old consts (scripted `StubPlayer` AI + self-locating seed
  probe for the dice).
- **S1 `BranchRouter.tau_spawn` DELETED, not wired** — the sharpest
  call of the batch: the router tests PIN spawn-on-no-snap as the
  intended semantics (`route_returns_spawn_when_below_snap_threshold`
  expects Spawn at cosine 0.707; the empty-tokens variant at cosine
  0.0), so wiring `best_score < tau_spawn` would invert the
  [τ_spawn, τ_snap) band to Frozen and break 4+ pinned tests. Field +
  `DEFAULT_TAU_SPAWN` + `new()` 3rd param + re-exports deleted here and
  in riir-engine's `cognitive_branches_runtime` re-export (riir-ai
  `8e748e936`).
- **B2 `use_ternary_gate` + `ternary_fusion_gate` DELETED, overriding
  the issue's wire-preference with evidence**: no weight source exists
  (a 6→1 `TernaryWeights` row nobody constructs), a linear gate cannot
  encode `route_one`'s threshold cascade, the lane is opt-in after its
  G1 quality FAIL (Issue 136), and the only test asserted the default
  `false`. `simd_ternary_matvec` remains substrate with its own
  consumers.
- **B3 `TransformerConfig.stop_token` DELETED** — the vocab owns
  stopping; the config mirror was write-only (`test_predict_token_argmax`
  was the proof: config "halt" vs vocab "c").
- **B1 `BcConfig.anneal`, B4 `lod_adaptive`, B5 `quant_levels` (→ 1-param
  `new(quant_scale)`), S2 `fft_size`, S3 `injection_layer`, S4
  `avg_bits_v`, S5 `solve_rate_floor/ceiling` DELETED** — each with a
  no-wiring-point rationale on record in the issue. S4's bonus honesty
  fix: test_147 "Proof 7 asymmetric vs symmetric" was fiction (the V
  path was identical VQ in both arms — `avg_bits_v` never changed
  anything); relabeled as the K-bit sweep it actually measured.
- **S6: 12 stillborn knobs DELETED** (`a3ef1830`)
  `DepthInvarianceConfig.magnitude_slope_collapse`, `HydraBudgetConfig.`
  `.cumulative_threshold`/`.modelless`, `CollapseDetectorFrozen.`
  `.budget_ema_mean` (option_stripper — the serialized COLP 24-byte wire
  struct in collapse_detector.rs is a DIFFERENT type, untouched),
  `InfluenceConfig.min_repetition_length`, `InfoNceConfig.default_critic`,
  `QbConfig.causality_strict` (grep refuted the doc's "kept for
  riir-train consumers"), `QueryFeatures.expected_output_len`,
  `SpKvConfig.predictor_lr_mult`, `TrdConfig.max_refinement_steps`/
  `.refine_correct_branches`/`.elf_noise_scale` (refinement depth is
  UCB1-bandit-chosen; correct branches are never refined). Sibling grep
  (riir-ai/riir-train/riir-neuron-db/riir-game-sdk) verified zero
  consumers per field before each delete.
- **FP record (the C2 trait-operational class)** kept in the issue for
  the riir-clippy post-mining harvest: `ColinearityBatchGate`,
  `RuleBasedVerifier`, `EntropyConflictDetector` — reads inside
  `impl Trait for X` bodies are operational (externally dispatchable),
  unlike inherent validate-shaped methods; the cheap detector
  refinement is exempting trait-impl bodies from the
  "own-validate-impl" bucket. riir-train 546 extended the class:
  `Display`-impl reads are operational too. Harvest landed in riir-clippy
  `.distill/001` Post-mining intake as FP class #3 (`eb35dfcd`, 2026-09-14 —
  the C2 trait-impl-body exemption + the wire-schema `[wire-const]`
  downgrade + the wire-vs-delete protocol note).

Verification: clippy `--all-targets` per touched crate INCLUDING the
feature arms that gate each module (`monopoly`, `shard_kv`,
`precision_aware_draft`, `flashar_consensus,dllm`, `depth_invariance`,
`hydra_budget`, `sp_kv`, `trd_refined_draft`, …) — the cfg-gated-target
trap bit once during landing (monopoly compiled to nothing under the
bare `--all-targets` run and hid E0596); affected suites green (92
monopoly, 15 shard_kv, 11 flashar, 9 test_147, 2048 core, …);
workspace `cargo check` 0 errors; riir-ai `cargo check -p riir-engine`
green. Twin issue riir-train 546 landed same day (2 wired, 12 deleted,
3 structs deleted — `b9f42028`).

Issue file removed per the noise-reduction rule; the full record lives
in git history
(`git log -- .issues/772_config_audit_inert_knobs.md`).


## Issue 777 — an instrument whose population is a FILESYSTEM walk audits code no repo owns: CLOSED (2026-09-14)

Found by running `percentile_drift_sweep.py` from the Windows workstation for
the first time. It red on two rows and **both reds were artifacts of the
walk**, not findings — which is the only reason anybody looked.

**The class.** `percentile_index_audit.py` and `len_derived_binding_audit.py`
walked the filesystem behind a hand-typed directory-name skip set
(`{"target", ".git", "node_modules", ".venv"}`). A name list cannot express
"not ours", and it was wrong three independent ways, all measured:

1. **A gitignored NESTED REPOSITORY.** `mmorpg-remaster/mmorpg/` is
   `.gitignore:82 /mmorpg/` and carries its own `.git`. Its 1404 `.rs` files
   were credited to mmorpg-remaster (2015 walked vs 611 tracked), and
   supplied 19 of the 23 percentile sites attributed to that repo —
   **including the sweep's only finding**, a TRUNC-VAR at
   `mmorpg/crates/mmorpg-bot/src/metrics.rs:129`. A correctly-shaped defect at
   an address where the repair cannot be made is worse than a false positive;
   it is the failure mode `issue_citation_gate.py` exists for, one axis over.
2. **Build artifacts under a directory the skip set does not NAME.**
   riir-train's 48 untracked `.rs` are cargo OUT_DIR sources (`glutin_wgl_sys`,
   `serde_core`, `thiserror`) under `.runs/target-release/`, `-cuda`,
   `-v2cpu`, `-bench`. The set names `target`; none of those IS `target`.
3. **Two floors fabricated by (1) and (2), neither ever edited.**
   `percentile_drift_floors.txt` pinned riir-train `min_rs_files = 2500`
   against 1129 tracked (1177 filesystem), and riir-chain `500` against the
   **460** that repo had on the day the file was created — it has GROWN to
   482 since, and the floor was above it the whole time. A floor exists to
   catch an instrument going blind; measured over content the repo does not
   own, it reds on every box but the one and the hour that produced it.

**The second defect, same file.** `percentile_index_audit.py:820` hard-coded
`root = "/Users/katopz/git"`, so the no-argument invocation AGENTS.md
documents as "all contract repos (derived)" was a `FileNotFoundError`
traceback on every box but one. The only such path in `scripts/`, and
`skill_repo_set_gate.py:64` already carried the rule in a comment. The sweep
half never noticed because it imports the module and never calls `main()`.

**The repair.** Tracked-only had been landed TWICE and never generalised —
Issue 734 for the trap audit ("25 findings in a gitignored vendored drop no
repo owns") and Issue 738 T3 for the platform/wasm32 pair — so the rule now
lives in ONE file, `scripts/tracked_walk.py`, with an 8-arm / 11-assertion
self-test: tracked-vs-untracked in both directions, the gitignored nested
repo, the alternate target dir *plus a companion assertion that `SKIP_DIRS`
still does not name it* (an arm whose hazard has been fixed elsewhere
certifies nothing), `vendor/` exclusion and its COUNT, the no-`.git`
fallback, the `git -C` walk-up probe, deleted-but-indexed, and the pattern
parameter on both branches. Consumers: `platform_dead_code_audit.py` (its
`list_rs_files` is now a seam over the shared call — 24/24 arms still pinned),
`percentile_index_audit.py`, `len_derived_binding_audit.py`,
`orphaned_attr_gate.py`.

**Measured after.** Three instruments that walked three different populations
now report one: **8694 tracked `.rs` over 16 repos** (percentile audit,
len-derived audit, platform-dead_code sweep — identical). The percentile
workspace figure moved 11,132 → 8,694 and 110 → 124 sites; the drop is not
deletion but riir-ai's 252 vendored `wgpu-hal` files, `mmorpg/`'s 1404 and
riir-train's 48 leaving the population. Floors re-pinned at this file's stated
~65% rule with the measurement in the row comment (riir-train 2500 → 730,
riir-chain 500 → 310). The sweep is green on all 16 present repos; the 4
absent ones red on the partial-clone axis, which is Issue 778's scope.
`orphaned_attr_gate.py` is unchanged in verdict (2415 files, 7124 sites, 0) —
katgpt-rs' tracked and filesystem counts are equal, which is exactly why it
was worth converting: nothing pinned that they would stay equal.

Not repaired, checked and recorded: `restatement_theorem_audit.py` and
`suite_membership_audit.py` also walk the filesystem, but both are scoped to a
named subtree (`.proofs/`, `PIN_DIRS`) rather than to a repo root, so neither
can reach a drop at the root. The `mmorpg` truncation itself is deliberately
NOT filed anywhere from here — once the walk is tracked-only that site leaves
this workspace's population, and filing it from katgpt-rs would be the
wrong-address defect a second time.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/777_filesystem_walk_population.md`).

## Issue 778 — `subprocess.run(..., text=True)` decodes with the SYSTEM locale: CLOSED (2026-09-14)

Found because `citation_drift_sweep.py` crashed on the Windows workstation with
`TypeError: expected string or bytes-like object, got 'NoneType'`. The
traceback was the lucky outcome.

**The defect.** `text=True` decodes the child's pipe with
`locale.getencoding()`. macOS, `ubuntu-latest` and the M3 are all UTF-8, so
nothing that could notice it ever ran it; this box is **cp874**, and every
instrument in `scripts/` prints `✓`, `✗`, `⛔` and em-dashes. Measured against
this repo's own `git log -3 --format=%s`: the em dash `E2 80 94` comes back as
`0xe42 0x20ac 0x201d` — three cp874 characters — under `text=True`, and as
`0x2014` under an explicit `encoding="utf-8"`.

Two failure modes, and the crash is the better one:

1. **Silent mojibake.** rc 0, a plausible string, and a caller matching
   `re.search(r"FAILED — (\d+)", out)` — em-dash in the pattern, mojibake in
   the text — matches nothing and reads a confident **zero findings**.
2. **`stdout = None` with the returncode PRESERVED.** Where a byte is undefined
   in the locale codec the decode raises inside `subprocess`'s reader THREAD,
   where the exception dies. `run()` returns normally.
   `citation_drift_sweep.gate_says()` got `(rc=0, stdout=None)`.

`PYTHONIOENCODING=utf-8` does not fix mode 1 and makes mode 2 more likely: it
pins the CHILD's encoder, so the child emits correct UTF-8 that the parent then
decodes as cp874. Both halves are needed.

**Blast radius, and why it survived.** 28 call sites across 16 scripts carried
bare `text=True` and **zero** passed an explicit encoding — while
`staged_set_audit.py` had carried the correct form *and a comment naming this
exact defect, dated 2026-09-04*, since the day it was written. Landed once,
never generalised: the same shape as Issue 777 two hours earlier, and the
reason both repairs ended in a single enforced mechanism rather than a sweep.

**The repair.** Every site becomes `encoding="utf-8", errors="replace"` —
`replace` and not `strict`, because a parser that raises on one odd byte in a
sibling's commit message is a new failure mode, and U+FFFD in a path is visible
where mojibake is not. The redundant `text=True` came off (an explicit
`encoding=` already implies text mode). The three `sys.executable` spawns also
pin the child: `env={**os.environ, "PYTHONIOENCODING": "utf-8"}`.

**The gate** — `scripts/subprocess_encoding_gate.py`, docs-gate check 19.
Ceilings of 0 on **two separately pinned classes**: DECODE (`text=True` /
`universal_newlines=True` with no `encoding=`) and CHILD-ENCODER (a
`sys.executable` spawn with no `PYTHONIOENCODING` in `env=`) — they are found
by different halves of the classifier and a shared pin would hide which
regressed. Floors under both (60 tracked `*.py`, 44 `subprocess` call sites), a
self-test that runs on every invocation with both directions per class.

The scanner went through the same lesson the classifier in Issue 775 did, in
one commit. Text scanning has to be paren-matched rather than line-scoped
(`encoding=` sits on a later line than `text=True` in every wrapped call
here) — and the paren-matched version then reported **four offenders in the
gate's own file**, every one a fixture string inside its `selftest()`. The
repairs on offer were to exempt the gate from itself or to obfuscate its test
data, and an exempt gate certifies nothing. It scans the **AST** instead: a
string literal is a literal, `text=flag` and `text=False` are not this defect,
`sys.executable` and `PYTHONIOENCODING` are matched structurally rather than
as source text, and a file the parser cannot read is **UNPARSED** and reds
rather than passing.

It earned its keep on its first run: a **28th** site nobody had grepped for,
`.agents/skills/doc-sync/tools/linkcheck_sweep.py`, outside `scripts/`
entirely.

**Verified after.** `citation_drift_sweep.gate_says()` returns `(1, 266, 1)` in
the normal posture where it previously crashed; docs gate 19/19. The remaining
partial-clone red in that sweep — the gate prints its DEFERRED line instead of
`scanned N citations`, so the cross-assert refuses — is a different class and
is filed as Issue 793.

**Cost of the CHECKS move.** The docs-gate set went 18 → 19 on the same day it
went 17 → 18, so the 18-check CPU figure will never be measured; AGENTS.md now
says the M3 run owes a 19-check number. That is the argument for writing the
CHECKS count beside the timing rather than the timing alone.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/778_subprocess_text_true_locale.md`).

## Issue 793 (allocated as 779; renumbered per Issue 791) — the workstation sweeps' partial-clone verdict, one copy: CLOSED (2026-09-14)

Running the whole sweep family from the Windows box after the Issue 777 walk
repair: **seven of eight sweeps reported FAILED with every content assertion
green**, each for exactly four absent-repo rows — on a box that is a known
16-of-20 partial clone with `DOCS_GATE_PARTIAL_CLONE=1` already set.

Seven carried this loop, byte-identical, copy-pasted:

```python
    for name in sorted(set(pins) - present):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")
```

`platform_dead_code_drift_sweep.py` was the one that did it correctly, because
it was written after Issue 765 and consumes `partial_clone_state()`. Landed
once, never generalised — the **third** instance of that exact pattern in two
days (Issue 777's tracked walk, Issue 778's subprocess encoding), which is why
this one is a shared mechanism (`scripts/sweep_population.py`) that the
platform sweep also repoints at, rather than a sweep of call sites leaving two
copies behind.

**Three verdicts, never pooled.** UNREGISTERED (on disk, absent from
`repo_set.txt` — a repo JOINING the workspace; reds in EVERY posture, because
no amount of partial checkout explains a directory that is right there) ·
UNSEEN (pinned or in the snapshot, absent, no marker — never a pass) · DEFERRED
(the same set with the marker, riding the sweep's FINAL line in BOTH
directions, because a deferral printed only on failure is a deferral nobody
reads on the run that passes). The marker stays an explicit opt-in: a genuine
removal whose row update was forgotten is set-identical to a partial clone from
the walk alone. Seven self-test assertions, including **UNREGISTERED reddening
under the marker** — the arm that proves the marker is not a blanket amnesty.

**T3, the second half.** `citation_drift_sweep.gate_says()` could not read the
gate's partial-clone output — `issue_citation_gate.py` prints `partial: N
citations scanned in …` under the marker instead of `scanned N citations`, the
regex returned -1, and the sweep exited **2** declaring its own instrument
untrustworthy. A correct refusal reached for the wrong reason. It returns **-2**
now, a THIRD state never folded into either neighbour, because "the gate could
not be read" and "the gate declined to adjudicate" call for opposite responses.

**What was behind the reds** — the issue's own argument, measured. With the
family runnable, the citation sweep surfaced four live Issue-749-class rows
that no run on this box had ever reached:

| repo | row | owner |
|---|---|---|
| katgpt-rs | `HISTORY.md:112` "Issue 513's T2 sweep" | riir-train — fixed `25b7bf6b` |
| riir-clippy | `HISTORY.md:6161` "Issue 150 removed per the noise rule" | mmorpg-editor — fixed `c5c30fee` |
| riir-train | `HISTORY.md:28` "chunk-size invariance, Issue 671" | riir-ai (`a8c3aec4a`) — fixed `96041bf6` |
| riir-ai | `HISTORY.md:250` "the Research 453 session" | riir-train — **deliberately untouched** |

The riir-ai row is the interesting one and is left alone on purpose: **HEAD
already reads `riir-train Research 453`**, and a sibling's UNCOMMITTED worktree
removed the qualifier. It is a live `staged_set_audit.py` STALE-vs-HEAD case in
somebody else's editing session, not a defect in the committed tree — and
committing a repair into a file another agent is editing would overwrite their
reconciliation with mine. Recorded rather than fixed.

Both postures verified end to end: without the marker the same runs print
UNSEEN and red; with it, six pass carrying the deferral on the pass line.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/779_sweep_family_partial_clone.md`).

## Issue 792 (allocated as 776; renumbered per Issue 791) — the docs gate's CPU self-timing printed a well-formed number that measured nothing on Windows: CLOSED (2026-09-14)

Full record in the session entry below, where it was filed as 776. This heading
exists so the number is READABLE: `issue_citation_gate.heading_allocated` reads
`## Issue NNN (…) — title` and does not read a bullet, so 793 and 794 were
discoverable after the renumbering and 792 was not. Verified both ways.

## Issue 790 (2026-09-14) — an arm that exists and RUNS may still reach nothing: CLOSED (2026-09-15)

`check_validation_gate` asserts that every CHECK invokes an arm and had to state
its own limit: *"arm QUALITY is not statically decidable and is not claimed."*
True, and too strong — quality is not **statically** decidable, but **reach** is
measurable by EXECUTION, and Issue 789 had measured it 53 times by hand, finding
seven arms that certified nothing. A census done by hand is a census that stops
being done.

`scripts/arm_reach_audit.py` (report) + `scripts/arm_reach_gate.py` (verdict,
workstation, ~200s): mutate a module's source OUTSIDE its arm bodies, re-exec,
run its arm, ask whether the arm noticed. Survivors pinned by MEMBERSHIP with a
reason per row in `arm_reach_survivors_expected.txt`; the wall is 0 UNPINNED,
and `UNREACHED` / `NO-ARM` / `BASELINE` are walled SEPARATELY because pooling
any of them into the survivor count destroys the finding.

**The nine findings, all of them in already-green instruments.**

1. ⛔ **The harness was judging itself.** `run_arm` exec'd modules into a bare
   dict, so `dataclasses` could not resolve `cls.__module__` — **seven
   classifiers read CRASHED on their own unmutated source, 796 of 2382
   mutants.** ⚠ The bare-dict direction is a PREMISE, not an assertion: CPython
   ≤3.12 guards that lookup and 3.14 does not, so the self-test asserts only
   that a `@dataclass` module EXECs in the registered namespace and prints which
   side the interpreter is on.
2. ⛔ **`BASELINE` was missing, and one of its two arms looks like a PERFECT
   score.** An arm already failing unmutated kills every mutant, reports 100%
   reach, and does not merely escape `MIN_KILLED` — it **inflates** it.
3. `required_features_touched_gate.selftest` was Windows-broken since written
   (POSIX path literals vs `Path`).
4. ⛔ **The gate caught the commit that changed it.** The bucket decision sat
   inline in `measure()` between two calls into the audit. Extracted to a pure
   `classify(row)`. The first repair re-entered the harness and wedged on `git
   ls-files` — 17 minutes at 0.02s CPU — and was replaced with injection.
5. ⛔ **A mutant can never RETURN, and the hang is the MILD half.** A run
   predicted at 13 minutes burned a core for TWO HOURS. Interrupting it credits
   the mutant KILLED, because `except BaseException` reads `KeyboardInterrupt`
   as the arm noticing. `TIMEOUT` is its own verdict with CRASHED's standing,
   and the deadline is DERIVED from the module's own baseline (10×, floor 30s)
   rather than typed — one constant cannot mean the same thing to a 0.03s gate
   and an 8.3s workspace sweep.
6. ⛔ `required_features_build_audit` used POSIX-only `os.statvfs` in
   PRODUCTION; the whole module was unimportable on Windows. Swept all 16
   repos: no other live site.
7. NO-ARM vs UNREACHED was decided twice, differently. One `has_runnable_arm`.
8. ⛔ **The blocking-C-call wedge has a mechanism and it is the BOX** — 167
   orphaned `git.exe` holding pipes. The watchdog is a thread +
   `interrupt_main` (SIGALRM is POSIX-only), so it reaches a pure-Python loop
   and NOT a blocking C call; naming the 10% it misses is the point of writing
   it down.
9. ⚠ **A starved box makes a gate print a confident WRONG verdict.**
   `bench_doc_audit` reported `97 labels, 56 mismatches` on a tree that had just
   passed (0 on re-run). Mechanism INFERRED and deliberately left unrepaired —
   resolved 2026-09-15 by measurement, and the inference was wrong twice; see
   `BlindRead`'s docstring.

**T5 DECLINED on a measurement, not blocked.** A cross-repo sweep would EXECUTE
~700 mutated copies of another repo's gate scripts, and re-measuring the
population found only **2 of 17** sibling arm-bearing scripts admissible. Do not
land it by symmetry with the eleven drift sweeps.

`--include-all` completed at **55 of 55** modules: 2376 mutants · 1182 KILLED ·
652 live · 535 exempt · 1 CRASHED · 6 TIMEOUT · 1 NO-ARM. ⚠ Read that against
the CHECKS population and not as a comparable number — the 652 is an unread
backlog, exactly the shape Issue 785 forbids ratcheting. Operating instruction:
run it **module by module under an external timeout**; one invocation is
unbounded in the worst case and not resumable, and the worst case happened twice
on the day it was written.

⛔ **The pattern worth carrying forward, found by fixing rather than by
reading:** in every module the CLASSIFIER was well armed and the **VERDICT** was
not, and about a third of the 47 survivors that were about to be pinned as
"EQUIVALENT" turned out to be plain functions over plain data with no fixture
and no subprocess between an arm and the decision — real gaps wearing an
EQUIVALENT label. **Writing the reason is the adjudication**; a classification
made while reading a list is not the same act. Closing them took the set 47 →
27, then 26.

Weakest-instrument follow-through (2026-09-15): `feature_isolation_gate` 4 → 24
killed, `citation_weight` 3 → 12 → 19, `ci_gate_coverage` 4 of 74 → **54 of
73**. Each needed an EXTRACTION or an INJECTION first — the reach was not
missing because nobody wrote arms, it was missing because the decisions were
unreachable by construction.

## Issue 791 (2026-09-15) — three numbers allocated twice across a 57-commit divergence: CLOSED

Two sessions both read `.issues/.highwater`, both incremented it correctly from
their own view, and both allocated **776, 779 and 780**. Neither was wrong about
the counter; a counter records the NEXT free number and is not a ledger of who
took what, and the rebase's `max(ours, theirs)` — the only sound merge for a
monotonic counter — is exactly what makes a double-allocation invisible.

**T1 — the instrument could not see the commonest collision shape.**
`citation_weight.candidates()` enumerated `.md` files ON DISK, so a
double-allocation where one side has CLOSED leaves one file and reads as "not a
duplicate". Every number this repo allocates is expected to end up removed, so
that is the majority case, not an edge. `removed_candidates()` recovers it from
`git log -M --diff-filter=D`; `-M` is load-bearing, because without rename
detection a RENUMBERED document reports as a deletion at its old number and the
tool would resurrect a collision somebody already resolved. ⛔ It still cannot
see a file created and removed with no commit in between — Issue 754's blind
spot, covered only by `numbering_drift_sweep.heading_allocated`.

⚠ The report grew a `of which HISTORY.md:` column, reported apart and never
folded or subtracted. A removed document has no file left, so its whole score
can come from its own obituary while a live rival's comes from third-party use;
those are not the same evidence. Not deducted, because Issue 724 T2's rule is
about the COST of moving a number and a HISTORY line is an edit that must move.

**T2 — adjudicated, and this session moved in all three.**

| number | weight | outcome |
|---|---|---|
| 776 | theirs 16 vs mine 5 (+13 / 21 decided) | mine moves, by the rule outright |
| 779 | **mine 13** vs theirs 8 (+5 / 21 decided) | mine moves anyway |
| 780 | mine +4, UNRESOLVED **10** > decided 4 | instrument DECLINES; mine moves |

⛔ **The 779 row is the one worth reading.** Weight measures the cost of moving
a number and it cannot see that one side's move requires another session's
agreement. Mine is CLOSED and removed — its whole footprint is mechanical
citations in files this session owns — and theirs is live open work.
Coordination-free beats a 5-site lead, and a rule that would have renumbered a
colleague's open file on a 5-site margin is a rule to bound, not to follow off
a cliff.

50 lines rewritten: `scripts/**` wholesale (all 31 citations verified as this
session's by reading them), `AGENTS.md` and `HISTORY.md` by LINE because both
are mixed corpora, and `crates/**`, `benches/**`, `.research/**` and the
other session's own HISTORY entry untouched. The three headings carry
`(allocated as NNN; renumbered per Issue 791)` so a commit message that already
says the old number stays followable.

⚠ The instance lists (`Issues 777, 778, 793, 782, 783`) are left in
CHRONOLOGICAL order and now read out of numeric sequence. That is deliberate:
the sentence counts recurrences of a shape, 793 was the THIRD, and sorting it
last would assert a false chronology to make a list look tidy.

**T3 — deferred, and the reason is a measurement not yet taken.** A gate that
reds when a commit allocates a number its remote parent already allocated needs
its false-positive rate measured first: a long-lived branch legitimately
allocates ahead of its remote, and this repo's own divergence was 57 commits.
The cheap half is procedural and already written down (fetch before allocating).

## Issue 797 (allocated as 796; renumbered — the other session allocated 796 the same hour and pushed first) (2026-09-15) — a sweep reads the WORKTREE, so a finding may exist in NO commit: CLOSED

Found by trying to close the last standing CROSS row in
`citation_drift_sweep.py`. It had been carried across a context boundary as
backlog reading *"blocked — that session has HISTORY.md uncommitted"*. The
correct verdict was not *blocked*; it was **there is nothing to fix**, and no
amount of reading the sweep's own output could say which.

Every instrument in the sweep family walks the **working tree**, and this
workspace runs five-plus concurrent agent sessions against **shared worktrees**
— `staged_set_audit.py` exists for exactly that hazard one axis over.

**The measurement** (`citation_drift_sweep.audit()` run twice per dirty repo,
once over the worktree and once with every dirty in-scope document replaced by
its HEAD blob, 16 of 20 repos):

| repo | dirty tracked | in scope | worktree | HEAD |
|---|---|---|---|---|
| riir-ai | 6 | `HISTORY.md` | **CROSS = 1** | **CROSS = 0** |
| mmorpg-remake | 1 | — | — | — |

The workspace's **entire** standing CROSS finding was an artifact. HEAD carries
`Filed … from the riir-train Research 453 session`; an uncommitted edit by
another session strips the qualifier. The POPULATION moved too — `n_cites` 601
(worktree) vs 607 (HEAD), `ambiguous` 162 vs 163 — because that session's
uncommitted deletion of a 30-line block took six citations out of the
denominator, so a floor re-pinned from such a run is wrong on every other box.

**Two directions, and the second had never been looked at.** UNCOMMITTED (the
worktree carries a row HEAD does not) is a false accusation: loud, and the
repair means editing a file another session holds open. **MASKED** (HEAD
carries a row the worktree does not) is a false green — the defect is
committed, in the repo, and the sweep says clean. Measured **0** today, which
is a measurement and not an absence of the class.

**T1** — `scripts/worktree_state.py`, one copy: `dirty_files()`, `head_text()`,
`split_rows()`, `dirty_in_scope()`, `sweep_advisory()`, 36 arms.
**T2** — wired into all **16** sweeps at the existing `population_verdict()`
call site, each with the globs naming its OWN population. ADVISORY, never a
failure: a sweep that hard-reds on an ordinary dirty worktree is a sweep nobody
runs. Verified per-population on the landing run — the `*.rs` sweeps reported
riir-ai (3), `*.md` riir-ai (1), `numbering` katgpt-rs (1) + riir-ai (1),
`subprocess_encoding` katgpt-rs (16), and `trap_sentinel` / `restatement`
printed nothing at all.
**T3** — the row-level split in `citation_drift_sweep.py`, via an injected
`read(path) -> str | None` so the same classifier can be pointed at HEAD. The
DISPLAY reads the worktree; the PINS read HEAD.
**T4** — AGENTS.md section; docs gate 21/21; arm-reach gate green.

**Three things this got wrong before it got them right**, all in my own code:

- ⛔ The `\` → `/` normalisation in `dirty_files` is DEFENSIVE and its arm
  certified nothing — deleting the line reds NOTHING, because `git status
  --porcelain` emits POSIX separators on every platform, measured on the
  Windows box where a naive reading expects the opposite. The arm asserts
  git's OUTPUT SHAPE now.
- ⛔ `("*.rs")` is not a tuple. Iterating it yields characters, `fnmatch(rel,
  "*")` matches everything, and the advisory silently reports every dirty file
  in the repo — **8 of 15** call sites were written that way in the wiring
  commit. The helper coerces and an arm pins both sides.
- ⛔ The row key must be LINE-FREE. Any edit above a citation shifts its line,
  so a line-bearing key reports every row in an edited document as UNCOMMITTED
  *and* MASKED at once.

Arm reach for the new module (`--include-all`): **19 of 23**, the three live
survivors being `check=True` / `capture_output=True` on the fixture BUILDERS,
with the reason written at the line.

**T5 — "every sweep" is a MEMBERSHIP assertion now, because the count was
wrong within two hours.** The landing commit said *"wired into all sixteen
sweeps"* and AGENTS.md said so too. A concurrent session then pushed
`pipefail_discard_drift_sweep.py` and `toolchain_override_drift_sweep.py`,
neither wired, and nothing noticed — the prose was stale before the commit
carrying it had finished being pushed. So this is the **seventh** instance of
the never-generalised shape (777, 778, 793, 782, 783, 789) and the first one
repaired mechanically: `scripts/sweep_advisory_membership_gate.py`, a docs-gate
CHECK, reds on a tracked `*_drift_sweep.py` calling neither `sweep_advisory()`
nor `worktree_advisory()`. **Membership, not a count** — a count is green on a
swap, and *a set is gateable where its cardinality is not*
(`cfg_gated_floor_gate`'s rule). Exemptions carry a reason each, the file is
deliberately EMPTY, and a stale pin reds so it cannot only ever loosen.

⛔ **Its FIRST run reported `citation_drift_sweep` — the most thoroughly wired
member, the only one carrying the row-level split — as UNWIRED**, because the
predicate named one of the mechanism's TWO entry points. A criterion that
condemns the most careful caller is the criterion that is wrong. Standing:
18 sweeps, all wired, floors 15/15.
⚠ It asserts the CALL and never that the patterns name the sweep's own
population: a sweep passing `("*.lean",)` over a Rust walk is silent forever
and reads as wired. Per-sweep read, not statically decidable — the same limit
`check_validation_gate` records about arm quality.

**T6 — the sibling's brand-new `dual_allocation_gate` was UNREACHED, and its
own collision is what exposed it.** Joining the CHECKS set put it in
`arm_reach_gate`'s population, which measured **0 killed of 28**: `selftest()`
delegated ENTIRELY to the probe module's arms, and *delegation cannot reach the
consumer* (Issue 775's sentence, Issue 789's rule). The fixture arms that DO
reach `classify` / `added_stems` / `adding_commits` already existed — as
`prove_fires()`, a name `arm_reach` deliberately never invokes.

`selftest()` calls them now. ⚠ The `RUN_ARMS` exclusion is a COST decision
measured against arms that `git archive` a frozen tree (80.2s vs 4.4s, 436 git
invocations); this one builds two small temp repos in **1.45s**, against the
docs gate's ~26s wall — cheaper than the `platform_dead_code_floor_gate` check
already in the set. Reach went **0 → 11 killed**, UNREACHED back to its walled
0, and the 12 survivors are adjudicated: 11 fixture-builder kwargs, and one
`or`→`and` in `classify`'s resolve guard that is **EQUIVALENT by measurement**
— deleting `refs/remotes/origin/X` makes `rev-parse --abbrev-ref
HEAD@{upstream}` fail too, so `upstream_of_head` returns empty and the guard one
line up fires first. The first attempt at an arm there asserted the wrong skip
string and proved the branch unreachable instead, which is how the pin reason
got written.


⚠ **Its own number collided, which is the joke and also the evidence.** The
other session allocated 796 the same hour for the allocation-time
dual-allocation gate and pushed first, so this renumbered to 797 — and their
brand-new `dual_allocation_gate.py`, replayed against the real pre-rebase
divergence, classified it correctly on its first live incident:
`⛔ INDEPENDENT 796 — two documents claim one number`, exit 1, both adding
commits named. ⚠ It reads green AFTER a rebase, because the divergence it
measures is gone; `numbering_gate`'s Issue-795 era-boundary wall is what caught
the residue, and rewriting the two unpushed commits is what removed it.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/797_worktree_state_sweep_findings.md`).

## Issue 795 (2026-09-15) — 70 numbering collisions the gate could not see, 9 of them live: CLOSED

Issue 791 recorded **three** double-allocated numbers and closed. Pointing the
instrument it had just built at one more number found a fourth, and a full scan
says the real figure is **70**, over 1374 numbers, in the four directories
`numbering_gate` already governs (`.issues` 51 · `.research` 10 · `.plans` 9 ·
`.proposals` 0).

⛔ **791's "three" was not a count of the problem; it was a count of what the
instrument could see.** A document closed under the noise-reduction rule is
DELETED, so a collision where BOTH sides have closed leaves nothing on disk and
reads as "not a duplicate" — and every number this repo allocates is expected to
end up removed, so that is the MAJORITY case rather than an edge. Six of the
nine recent collisions were invisible for exactly that reason, and
`numbering_gate`'s tracked-duplicate wall — correct about what it measures — had
never reported one in its life.

All nine at or above 700 (741, 775–782) are one event: two sessions both read
`.issues/.highwater`, both incremented correctly from their own view, and the
rebase merged the counter with `max(ours, theirs)` — the only sound rule for a
monotonic counter, and exactly what makes a double-allocation invisible.

⛔ **The first scan over-reported by 52, and the reason was already written
down.** Walking every numbered directory found 122; 52 were `.benchmarks/`,
where the leading number is the OWNING plan or issue and a family per owner is
the intended convention. `numbering_floors.txt` records that exclusion, measured
2026-09-04, with the note that checking there *"would have been the cries-wolf
instrument AGENTS.md warns gets ignored."* **A population derived from the tree
is not the population the gate governs** — the difference was 74% inflation
straight into a measured false-positive class.

**The verdict runs in TWO regimes, because one would be wrong in both
directions** (`scripts/number_collisions_expected.txt`):

- **At or above `era_boundary = 700`: a WALL, pinned by MEMBERSHIP with a reason
  per row.** A count is green on a swap; the arms assert that exact case. Reds
  in both directions — a pinned row that is no longer a collision is a finding
  too, because the pin and its removal belong in the same commit.
- **Below it: a RATCHET, counted and never pinned.** Those 61 are the pre-gate
  archive (`.issues/121`'s number-recycling era) and adjudicating them is a
  backlog; Issue 785's rule forbids ratcheting a bucket that means *unread*. A
  count that DROPS is a note, not a failure — refusing the commit that resolved
  a collision would be the gate punishing the repair.
- The boundary is **measured, not round**: highest legacy 575, lowest divergence
  741, nothing between them, so no row sits on the wrong side by judgement.
- Two blindness floors — every verdict above is a ceiling, and a git-history
  regression empties the population and passes all of them.

⛔ **Six of the nine were adjudicated and deliberately NOT renumbered**: leads of
+1 to +5 over 11–36 decided sites with 21–53% UNRESOLVED, one an outright
`TIE_FRACTION` tie (778) and one where the tool DECLINED because unresolved
outnumbered decided (775). Renumbering on a 2-site lead with 47% unresolved is
the mistake `TIE_FRACTION`'s own docstring names — *"pretending it can arbitrate
is how a coin flip gets recorded as a measurement."* The margin is on each pin
row, so the decision is re-readable rather than remembered.

An earlier, allocation-time gate is DEFERRED on its unmeasured false-positive
rate (a long-lived branch legitimately allocates ahead of its remote; this
repo's own divergence was 57 commits). ⚠ Note what the wall already buys: the
collision is caught on the MERGE commit — late, but not silent, and the first
time anything catches it at all.

⛔ **Both this gate's arms and `citation_weight`'s were written, passing, and
reaching NOTHING** until they were moved out of `main()` into `selftest()`:
`arm_reach_audit` invokes an arm only by the names in its vocabulary. Measured
twice in one session, which is why it is written here and not remembered. Arm
reach: `numbering_gate` 7 killed of 48 → **22 of 47** (survivors 17 → 2);
`citation_weight` 12 of 49 → **19** (18 → 11).

**How both 791 and 795 were found, recorded because it repeats:** by pointing an
instrument at one more case, not by a symptom. 791's table was written from
three known pairs and nothing asked whether there were others, because the tool
that would have answered was blind in the direction that mattered. *A census is
exhaustive over ROWS, not over the ORACLE it checks them against* — Issue 754's
sentence, holding for a third time.

## Issue 796 — the allocation-time dual-allocation gate: the FP rate measured, the gate built CLASSIFIED: RESOLVED (2026-09-15)

791 T3 and 795 both deferred the allocation-time gate on its unmeasured false-
positive rate. `scripts/dual_allocation_fp_probe.py` measured it from git's own
records — reflog-reconstructed (local_tip, upstream_tip) pairs, union-sampled
over BOTH timelines' entry timestamps (a fetch moves the remote-tracking ref
WITHOUT moving HEAD, so local-only sampling misses the divergence-discovery
moment; measured: the first cut saw 48 green katgpt-rs pairs and missed the
very RED it was built for). 18 repos, ~90 d: **634 divergent pairs, 7202
one-sided (the feared long-lived-branch shape — green BY CONSTRUCTION), 164
RED pair-instants = 39 distinct incidents = 31 TWIN + 8 INDEPENDENT.**

The twin/independent split is structural, by filename STEM per colliding
number — the subject-equality test MISLABELS (measured: mmorpg-remake `2`'s twin
commits carry different subjects; the stems are identical). The 8 INDEPENDENT,
all stem-verified: riir-ai 722/780/935, riir-chain 30+72/34, riir-clippy
79/83, mmorpg-editor 192–195.

Verdict: the deferral's fear is MOOT (one-sided allocation never fires) and
its caution RIGHT (a naive hard gate cries wolf 31/39 on twins — one incident
contributed 43 pair-REDs). `scripts/dual_allocation_gate.py` is therefore
CLASSIFIED: TWIN annotates exit-neutral (your own rebased line; fetch
resolves it), INDEPENDENT exits 1 with both sides' adding commits named.
`--prove-fires` builds the two-session fixture in a temp dir and asserts both
verdicts against real git — and earned its keep before the gate ever ran: the
probe's FIRST sweep was VOID (a `--.issues` pathspec typo made every
measurement the empty set — 105 green katgpt-rs pairs that measured nothing)
and the fixture caught it before any rate was recorded. Reach limit, binding
on every single-box instrument: the 791 divergence itself is not
reconstructible from this box's reflogs — it lived on the wire between boxes;
the gate sees what the running box participates in, at fetch/push time.
Joined to docs_gate CHECKS (796 T5, same day): a workstation run first, then
the CHECKS row + the AGENTS.md table row + the prose count-move note in one
commit. In CI the check green-exits by construction (a main-push checkout has
HEAD == origin/main, merge base == HEAD) — its live reach is the workstation
dev loop, where the divergence exists at run time.

## Issue 794 (allocated as 780; renumbered per Issue 791) — a wrong address reads as UNDECIDED when its number is in local range: CLOSED (2026-09-14)

`is_qualified()` (Issue 752) fixed the *question* — "is a repo named?" →
"does that repo **own** the number?" — and `citation_drift_sweep.audit()`
annotates the answer with `⛔MISATTRIBUTED`. The annotation was gated on
`cls is CROSS`, and the three-way bucketing runs first:

```python
cls = (IN_RANGE if n <= top[kind] else CROSS if owners else ORPHAN)
```

So a citation carrying an explicit attribution **on** it to a repo that does
not own the number was reclassified **IN-LOCAL-RANGE** — "UNDECIDED, never
clean" — whenever the number also fell under the *citing* document's own
ceiling. Never counted, never tagged, never gated, and printed only inside a
4-row truncation of undecided noise.

**IN-LOCAL-RANGE's premise is refuted by such a row's own text.** It reaches
that bucket only when `n not in mine[kind]` — the Issue-754 oracle (worktree
AND `git log` AND headings) found no local allocation — *and* the author wrote
a different repo's name directly on the citation. Both halves of "a local
referent is plausible" are gone. **CROSS is unfollowable; this is followable,
to the wrong place**, so it is its own class (`MISATTRIBUTED-IN-RANGE`), kept
inside the IN_RANGE bucket for the `max_in_local_range` ceiling (the undecided
*population* did not change) and printed ahead of it, never truncated.

**The one row it was hiding was the example this repo's own AGENTS.md names.**
riir-train Issue 513, written up as `katgpt-rs Issue 513`, sits in
`is_qualified`'s docstring and in the §citation paragraph as a worked instance
of a wrong address — and it was still standing in the workspace: riir-neuron-db
`AGENTS.md:82`, a `# Workstation-only pre-push layer` comment citing
riir-train Issue 513 as katgpt-rs's. katgpt-rs has never allocated 513 (`git log --all -- '.issues/513*'` empty, `.highwater` 779);
**riir-train** owns it (`513_required_features_rows_are_unverified.md`, filed
`389a0a6b`, closed + removed `5a4265df`), and that issue's own T6 is the
paragraph's subject by name — *"BLOCKED ON OWNER CALL (Actions spend) …
2026-09-11 owner verdict: DECLINED for riir-neuron-db … the workstation
`required_features_touched.sh` layer already covers it."* riir-neuron-db's top
allocation is 617, 513 ≤ 617, and that arithmetic is the whole reason nobody
saw it. Repaired to `riir-train Issue 513 T6` (riir-neuron-db, its own commit).

**The boundary is measured, and it is not the obvious one.** The same predicate
has a second home — the `n in mine[kind]` short-circuit one branch up, where
the number *is* locally allocated. Measured over 16 repos × AGENTS.md+HISTORY.md
(3,392 citations), after `is_qualified` clears every row some named owner covers:

| bucket | rows | hand-read |
|---|---|---|
| `n <= top`, not allocated (IN-RANGE) | **1** | 1 true, 0 false |
| `n in mine` (locally allocated) | **19** | **0 true, 19 false** |

The 19 are one shape: the prose is *contrasting* a local number with a remote
one and the 40-char lead catches the neighbour's address — riir-chain's
``riir-ai Issue 853 / this repo's Issue 093``, riir-mmorpg-examples' ``riir-ai
Issues 574/589/537/672 + local Issue 059``, ``in `riir-neuron-db/src/local_kv.rs` (Issue 043``, ``at
`riir-game-sdk/crates/riir-games-cluster/`. Plan 010``. That asymmetry is
mechanism rather than luck: a locally-allocated number **has** a local referent
for the prose to contrast against, and a never-allocated one does not. So the
rule stops at IN-RANGE, the short-circuit stays exactly as it was, and the
exemption is a measurement instead of an oversight. ⚠ Read the other column
honestly too — it is **n = 1**, so "0 false positives" is one row's worth of
evidence, not a rate.

**T2 resolved as a NO-OP, and that is the second finding — but state it
precisely.** The per-push `issue_citation_gate.py` has **no IN-RANGE bucket at
all**: it skips only the *allocated* set and reds on everything else
unqualified. For katgpt-rs the two instruments cannot disagree on the COUNT and
already assert it — `gate_says()` pins `gate findings == CROSS + IN_RANGE +
ORPHAN`, a partition. The leniency is in the **labelling plus the per-repo
ceiling**, and it bites in the 15 repos the gate never runs in: a wrong address
bucketed as "undecided" sits under a non-zero `max_in_local_range` ratchet and
is tolerated, which is exactly what happened to riir-neuron-db. So the defect
was the divergence in *what the row is called*, not a number the cross-check
could ever have caught.

Ceiling design: a **global wall** (`max_misattributed_in_range = 0`), not a 5th
per-repo ratchet field — the class has no backlog anywhere, so per-repo pins
would be 16 zeros. A missing pin is **refused** (exit 2), never defaulted, or
the wall reads as "absent, so anything passes" — the green-zero shape this
family exists to refuse.

Four self-test arms, and the class is defined as much by what it must **not**
promote: the finding fires; a correctly-addressed in-range citation produces
nothing; a bare one stays UNDECIDED; a *window-only* repo name (not on the
citation — `adj` is lead-only by Issue 752) does not promote; and a
locally-allocated number with an adjacent non-owner name stays AMBIGUOUS, which
is the 19/19-false exemption asserted rather than assumed. Both directions
proven on the **live** corpus too: with the riir-neuron-db line stashed the
sweep reds naming it, and restored it goes clean.

⛔ One DRY defect found by its own canary: the row-list append and the tag were
two separate `cls is IN_RANGE and bad` tests, so disabling one left the other
certifying. One predicate, one place — the canary caught it because the arm
that survived reported a *different* failure than the one that was disabled.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/780_misattributed_is_computed_only_in_the_cross_bucket.md`).

## Issue 781 — the heading oracle matches one house STYLE: CLOSED as a measured, printed blind spot (2026-09-14)

`heading_allocated()` (Issue 754) is the path that recovers a number whose file
was created and removed without an intervening commit — the repo's own heading
is then the whole allocation record. Its pattern anchors the parenthetical
**immediately** after the number, so `## Issue NNN (date) — title` reads and
`## Issue NNN resolved — title (date)` does not. Measured over 16 repos ×
AGENTS.md+HISTORY.md: **64 of 152 read, 88 unread**, and the split is by
**house style**, not by correctness —

| repo | shaped | read | unread |
|---|---|---|---|
| riir-mmorpg-examples | 43 | **43** | 0 |
| mmorpg-remake | 15 | 14 | 1 |
| katgpt-rs | 29 | 7 | **22** |
| riir-ai | 25 | **0** | 25 |
| riir-clippy | 25 | **0** | 25 |
| riir-train | 13 | **0** | 13 |
| riir-game-sdk | 2 | 0 | 2 |

katgpt-rs's own newest closes — Issues 777–781 included — are in the form its
own instrument cannot read.

**Closed as a REPORT, not a repair, and the reason is the instrument's own
self-test.** Arm 2 pins `## Issue NNN follow-up (date)` as a measured
NEGATIVE: a heading that *comments* on a number is not an allocation of it, and
crediting it would absolve a wrong address. `NNN follow-up (…)` and
`NNN resolved — … (…)` are the **same shape**; no punctuation rule separates
commentary from allocation, the distinction is semantic. A widened pattern was
run end to end against the live workspace and fails that arm — which is the arm
doing its job. This is also the only path in the instrument that can
**suppress** a finding, so a speculative widening trades a reported blind spot
for an unreported one.

So the cost is **measured every run and printed**, with the standing of
AMBIGUOUS and the width-bound complement (`heading_unread=N/M` per repo, plus a
workspace line). The foreign-repo filter is applied to both sides over the whole
heading, so a row rejected for *naming a sibling* is not miscounted as a style
loss — the gap printed is exactly the style gap. Both directions pinned by a
self-test arm (042 reads · 043 is the style loss · 044 is a foreign-name
rejection excluded from both), canaried by making the probe blind. AGENTS.md
carries the magnitude only; the dated snapshot lives in the function's own
docstring, where it is a measurement record rather than a claim.

Consequence, both directions: an incomplete **local** set lands as UNDECIDED
noise — riir-clippy's 10 undecided rows are its own four numbers, every one
recorded in its own HISTORY.md in the unread style. An incomplete **owners**
set manufactures a **false** `⛔MISATTRIBUTED` — Issue 754's exact failure,
inherited by Issue 794's `MISATTRIBUTED-IN-RANGE`. 0 live instances today,
which is precisely why it is printed rather than remembered.

⛔ **The write-up of Issue 794 introduced four rows of the class it
documents**, and that is the lasting lesson here: a document that discusses a
misattribution has to reproduce it, and the instrument cannot tell a quoted
specimen from a live one. Three were riir-train Issue 513 quoted under
katgpt-rs's name; one was a quoted false positive of riir-chain's
(``riir-ai Issue 853 / this repo's Issue 093``).
The repair is **not** a pin — it is to name the true owner inside the
citation's own 3-line window ("riir-train Issue 513, written up as `katgpt-rs
Issue 513`", "riir-chain's ``…``"), which clears the row *and* makes the
sentence followable; where the number is incidental to the example, write
`NNN`. Reach for those before ratcheting a ceiling for prose about prose.

This section was written under the same constraint, and the new class caught it
**in the act**: the first draft of the paragraph above produced one
`⛔MISATTRIBUTED-IN-RANGE` and one UNDECIDED row of its own. Repaired by naming
the owners ("riir-train Issue 513 quoted under katgpt-rs's name", "a quoted
false positive of riir-chain's") — a live two-directional proof worth more than
the synthetic arms.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/781_the_heading_oracle_matches_one_house_style.md`).

## The x86_64-pc-windows-msvc axis, measured: clean at all-features/all-targets (2026-09-14)

AGENTS.md §full gate names two platform axes — macOS (where
`not(target_os = "macos")` backends compile to nothing) and wasm32 (gated
twice, by triple and by `simd128`). There is a third, and until this run
nothing had ever compiled it: **`x86_64-pc-windows-msvc`**. It is the lane
`platform_dead_code_audit.py` describes as "a workstation lane no automatic
gate in this workspace ever sees" — `full_gate.yml` is macOS/aarch64,
`wasm32_gate.yml` is the wasm32 triple, and neither emits an
x86_64-native dead-code warning.

Measured on the pinned toolchain (rustc 1.98.1, host
`x86_64-pc-windows-msvc`, `rust-toolchain.toml` channel, no override):

```
cargo clippy --workspace --all-targets --all-features --keep-going -- <the AGENTS.md -D list>
→ exit 0, 32 packages, 8m17s, ZERO code findings
```

All 1,034 warning lines are one message — `hard linking files in the
incremental compilation cache failed, copying files instead` — an NTFS
artifact of the `target/` directory, not a lint.

**Read the scope, not the headline.** This compiles the
`not(target_os = "macos")` half that a macOS run drops — which is the inverse
blind spot `scripts/check_platform_gated_modules.sh` exists to *typecheck*
from the M3, here compiled and linted for real — but it is **not** a
substitute for either named axis: `target_os = "macos"` and
`target_os = "linux"` code both compiled to nothing here, and so did every
wasm32 arm. A platform is part of the claim, exactly as the profile is; this
run adds one cell, it does not close the matrix. The profile axis is also
untouched: this is dev, so `debug_assertions` was ON throughout.

## Issue 782 — a pinned repo absent from the walk is never visited: CLOSED (2026-09-14)

Issue 793 gave the sweep family one shared partial-clone verdict
(`scripts/sweep_population.py`) because **seven** sweeps carried a copy-pasted
"pinned but ABSENT from the derived walk" loop and hard-red on a known
16-of-20 box. Three were left alone for not carrying that loop. **All three
were non-exempt** — the census had selected on *symptom*, not on predicate.

| sweep | what it actually had | why it looked clean |
|---|---|---|
| `cfg_row_implication_drift_sweep` | **no absence check at all** | printed a confident green over 16 of 20 |
| `docs_drift_sweep` | the copy-pasted loop | its 8 label-bearing repos are all checked out here |
| `restatement_drift_sweep` | the copy-pasted loop | its 4 `.proofs` repos are all checked out here |

The first is the live defect and the worse direction of the two. It iterates
the **derived** repos and asks `pins.get(repo.name)`, so walk→pins reds
(UNREGISTERED) and **pins→walk is unchecked**: a pinned repo the walk never
found is never iterated and nothing says so. `katgpt-web`, `riir-dao`,
`riir-deployer` and `riir-esp32` each carry a row in
`cfg_row_implication_drift_floors.txt` and were evaluated by nothing, under the
line `✓ … PASSED — every repo within its pins`. A sweep that hard-reds is
annoying and impossible to misread; this one reports a green over a subset,
which is the green-zero shape the whole family exists to refuse — **and it
survived the 779 census precisely because it was quieter.**

All eleven share the verdict now, and both postures are verified on this box
for each repair: with the marker, a pass carrying a **named** DEFERRED line for
the four; without it, a red naming them UNSEEN and no "every repo" claim
surviving on the final line.

⚠ **A subset-population sweep has TWO populations and they are not
interchangeable.** `population_verdict(pins, present)` wants the **contract
walk** for `present`; handed `restatement_drift_sweep`'s own derived set
(repos carrying `.proofs`), every contract repo *without* proofs read as
absent — measured at **16 phantom rows**, and the run failed. Fixed by passing
the contract walk and keeping the subset for the measurement loop.

That split leaves a hole the shared verdict structurally cannot see, because it
asks only about the contract walk: a repo **pinned and checked out** that has
dropped out of the sweep's own subset — its `.proofs` directory removed — is
skipped in silence by `for repo in present: if repo not in floors: continue`,
and a floors row nothing measures is a ceiling that cannot fail. That is a
per-subset-sweep check (`⛔ DROPPED`), not a shared one, and it is now in the
restatement sweep alongside the shared verdict.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/782_a_pinned_repo_absent_from_the_walk_is_never_visited.md`).

## The executed-test gate on x86_64-pc-windows-msvc: exact floors, all three rows (2026-09-14)

`scripts/test_gate.sh`'s own header says its floored counts are *expected* to
be platform-invariant — katgpt-core has zero `#[cfg(target_os)]` attributes and
the root lib's are all behind opt-in features — and that "the first scheduled
run is the measurement: if deltas surface, that is the rot check finding real
debt, not a reason to widen silently." The weekly schedule has been suspended
since 2026-09-09 (Actions spending limit), so the claim had been asserted on
one platform only.

Measured here, second platform, pinned toolchain 1.98.1:

```
katgpt-rs   --lib (default)      passed=203  floor=203
katgpt-core --lib (default)      passed=2041 floor=2041
katgpt-dec  --lib (pca_global)   passed=249  floor=249
test_gate: PASS
```

**Exact on every row — no deltas, no slack.** Platform invariance is now a
measurement on `x86_64-pc-windows-msvc` as well as the lane it was written
for, and it is EXECUTION rather than compilation: the axis AGENTS.md calls out
as the one where an uninvoked assertion is *unknown*, not passing. Scope
unchanged otherwise — default features, dev profile, the scoped core only; the
477 integration-test and 176 bench targets remain executed by nothing
automatic.

## Issue 783 — the subprocess-encoding gate is katgpt-rs-only: CLOSED (2026-09-14)

`scripts/subprocess_encoding_gate.py` (Issue 778) shipped as a per-push gate
with no sweep half. Eleven other verdict classes here carry both, and the
asymmetry was not a judgement call that was made — it was a step that was
skipped. `scan()` already took a repo path, so the question was answerable the
whole time.

**The answer, 2026-09-14, 16 of 20 repos:**

| repo | tracked .py | subprocess calls | DECODE | CHILD-ENCODER |
|---|---|---|---|---|
| riir-train | 68 | 15 | **12** | **1** |
| riir-clippy | 5 | 12 | **9** | **1** |
| riir-ai | 8 | 8 | **6** | 0 |
| riir-dapps | 1 | 1 | **1** | 0 |
| mmorpg-editor | 2 | 2 | **1** | 0 |
| katgpt-rs + 10 others | 79 | 44 | 0 | 0 |
| **total** | **163** | **82** | **29** | **2** |

Seventh instance of one shape, and the seventh time pointing it anywhere but
here found something — the precedent list is in
`markdown_fence_drift_sweep.py`'s docstring.

**Two of the 31 were not latent.** `riir-clippy/scripts/gen_dashboard.py:552`
reads `git -C <d> log --since=… --pretty=%s` across the sibling repos, and
every commit subject in this workspace uses an em-dash — on a non-UTF-8 box
that is silent mojibake at best and `stdout = None` with the returncode intact
at worst, either way a dashboard section that renders a confident nothing.
`riir-train/scripts/plan344_phase0_full_bandwidth.py:318` reads a
`git ls-files '*.md'` list and then opens the paths.

All 31 repaired in the same change — `encoding="utf-8", errors="replace"` on
29 parent reads, `env={**os.environ, "PYTHONIOENCODING": "utf-8"}` on the two
`sys.executable` children — and the ceilings pinned at **0/0**, a wall rather
than a ratchet. A backlog on a class whose repair is two tokens teaches
whoever reads the pin that the class is tolerated.

**Both floors, and the fence sweep's argument does NOT transfer.**
`markdown_fence_drift_sweep` gets away with one floor because `min_md_files`
is non-zero in all 20 repos. Here `min_calls` is **0 in 10 of the 16
measured** — those repos have `.py` files and no `subprocess` at all — so the
parse floor cannot detect anything in the majority of the population, and
`min_py_files` is the only blindness detector there. Both are pinned and
neither is derived from the other. katgpt-rs's two are asserted equal to
`subprocess_encoding_gate.FLOOR_PY_FILES` / `FLOOR_CALLS` rather than trusted
(the `docs_gate_paths_sync.py` pattern), and that assertion was canaried by
perturbing the pin file.

Canaries, all measured: a planted DECODE in riir-dapps reds the run and names
the row; the marker-off posture reds with UNSEEN over the same four absent
repos the marker-on posture DEFERS by name; seven self-test arms cover both
verdicts firing through `scan()`, both controls, UNPARSED surfacing, the
tracked-only walk boundary (the temp repo is `git add`-ed on purpose — an
unstaged one exercises `tracked_walk`'s rglob FALLBACK and certifies the branch
the arm is not aimed at, the Issue-775 vendor-arm failure), population
derivation and pin arity.

**Four repos have NO row, deliberately** — katgpt-web, riir-dao,
riir-deployer, riir-esp32 are not on this box, and a pin nobody measured is a
number rather than an expectation. They ride the shared population verdict as
DEFERRED and will report UNPINNED on the first full checkout, which is the
intended red. Identical posture to `platform_dead_code_drift_floors.txt`'s
same four rows; pin both files in one commit, from one run, on the box that
can see them.

Drive-by, found by the sweep's own output rather than by looking: the walk
emitted a `SyntaxWarning` from `ast.parse` on
`riir-train/scripts/bonsai_vs_gemma_codegen.py:16` — a backslash-escaped
backtick in a non-raw docstring, a forward-incompatible escape that becomes a `SyntaxError` in a
later Python. One instance workspace-wide over 163 files, so it is a repair
and not a class: no backlog, nothing to gate.

**The standing failure mode, now recorded five times** (Issues 777 tracked-walk,
778 itself, 779 sweep-population, 782 the three quiet sweeps, 783 this one): a
rule landed in one instrument and never generalised. Before fixing such a
class, grep the whole family and land the repair as one shared mechanism.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/783_the_subprocess_encoding_gate_is_katgpt_rs_only.md`).

## Issue 784 — the orphaned-attr gate's cross-repo claim was hand-run: CLOSED (2026-09-14)

`scripts/orphaned_attr_gate.py` was the LAST row in `docs_gate.sh`'s CHECKS
whose class is cross-repo (Rust source, in every repo) and which had no
workstation sweep half. Issue 783 closed the same gap for `subprocess_encoding`
earlier the same day; this is the one it left, and it is the **eighth** instance
of the shape.

**It is the first of the eight that found no new offenders — and that is the
honest headline.** `max_offenders` has now held at 0 across three independent
measurements and TWO population definitions, which is a stronger statement than
any single count. What it found instead was a stale **warrant**.

The gate's docstring carries the workspace figure as the argument for why the
class can be gated at all — *"a zero over 49,624 sites is evidence; a zero over
a walk that has gone blind is not"* — and somebody had been typing that total in
by hand since 2026-09-03. Issue 777 migrated the gate to the TRACKED walk the
same day (`820bf8b6`), and the hand-typed warrant did not follow:

| quantity | docstring (2026-09-06, filesystem walk) | measured (tracked walk) | drop |
|---|---|---|---|
| `.rs` files | 11,132 | **8,694** | 22% |
| outer-`#[cfg]` sites | 49,624 | **26,598** | **46%** |
| orphaned | 0 | **0** | — |

The verdict never moved. **23,026 of the sites offered as its warrant were in
trees no repo owns** — mmorpg-remaster's gitignored `mmorpg/` nested
repository, riir-ai's vendored `wgpu-hal` fork, riir-train's cargo `OUT_DIR`
sources under `.runs/target-*`.

**This is Issue 777's second-order damage one class over.** 777 repaired the two
`percentile_drift_floors.txt` rows it measured and that file's own prose, and
left every OTHER prose restatement of the same walk standing. A `grep` for the
figure found exactly two live copies — `orphaned_attr_gate.py:38` and
`.docs/10_audits/percentile_index_tail_support.md:121` — and a widened hunt over
`.docs/`, `.agents/`, `scripts/` and AGENTS.md for any other four-or-five-digit
`.rs`/`.py`/`.md` population figure found no third. Both repaired in the
`percentile_drift_floors.txt` style that 777 got right: **the pre-777 number
kept as a dated record, the tracked-walk number next to it, and the mechanism
named** — a reader who cannot see that the population DEFINITION changed reads a
46% drop as deleted code.

**Correcting two numbers is not the repair.** The reason the figure went stale
is structural: the gate audits one repo per invocation and nothing re-asserted
the total. The gate's own docstring already warned about this exact shape (until
2026-09-04 its PASS line printed "measured 0 across 19 repos" on every run, a
cross-repo claim no run had made). The fix applied then was to stop the PASS
line making the claim — which stopped it being *printed* stale, and the number
went on being hand-typed for eleven more days. `orphaned_attr_drift_sweep.py`
makes the claim MEASURED.

**Both floors, and this population's warrant is the opposite of Issue 783's.**
There `min_calls` is 0 in 10 of 16 repos and detects nothing across most of the
population, so `min_py_files` carries it alone. Here both quantities are
non-zero in all 16 — the smallest, riir-viewbridge, has 24 tracked `.rs` and 20
outer-`#[cfg]` sites — so both floors are live everywhere. Same two-floor shape,
different warrant, stated as a measurement in both pin files so neither carries
the other's argument.

The narrowing is imported, never restated: `OUTER_CFG` vs `ANY_ATTR` is the
whole classifier, and it is what takes the broad shape's **2,044** sites to 0 —
inner `#![cfg(...)]` binds to the enclosing module and is conventionally
followed by a blank line. Selftest arm 3 pins that directly; if it ever passes a
finding the sweep reports thousands.

Canaries, all measured: a planted orphan in riir-shader's `camera.rs` reds the
run and names both the attribute and the item it re-bound to; marker-off reds
with UNSEEN over the same four repos marker-on DEFERS by name; perturbing
katgpt-rs's `min_cfg_sites` by 1 trips the shared-floor assertion against
`orphaned_attr_gate.FLOOR_CFG_SITES`. Seven self-test arms, the walk-boundary
one on a `git add`-ed temp repo on purpose — an unstaged one exercises
`tracked_walk`'s rglob FALLBACK and certifies the branch the arm is not aimed
at (the Issue-775 vendor-arm failure).

Four repos deliberately unpinned (katgpt-web, riir-dao, riir-deployer,
riir-esp32) — not on this box, DEFERRED, UNPINNED-red on the first full
checkout. That is now THREE floors files owed the same one-commit repair:
`platform_dead_code_drift_floors.txt`, `subprocess_encoding_drift_floors.txt`,
`orphaned_attr_drift_floors.txt`.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/784_the_orphaned_attr_cross_repo_claim_is_hand_run_and_stale.md`).

## Issue 785 — the wasm32 surface audit had no verdict half: CLOSED (2026-09-14)

Issues 783 and 784 closed the last two per-push gates with no cross-repo sweep.
This is the mirror gap: `scripts/wasm32_surface_audit.py` is already cross-repo
(it derives its own population) and had **no verdict at all**. Nothing pinned
its buckets, so a package falling out of coverage was a line in a report
somebody runs on demand — and the workspace standing lived in AGENTS.md as a
hand-typed sentence, the exact shape 784 had closed hours earlier after the same
kind of total went 46% stale.

**Why this class earns a wall.** An UNCOVERED package is code that has never
compiled and that nothing will ever tell you about, because the arch it is gated
on is one no lane passes. Measured: mmorpg-remake's positive
`#[cfg(target_arch = "wasm32")]` block (`.issues/010` T2, katgpt-rs Issue 738)
was **uncompilable from the day it was written** — it called a
`cfg(not(wasm32))` function — and nothing said so for months.

**T1 was the real work: one classifier, not two.** The buckets were computed
inline inside `main()`'s print loop, so the sweep could only reuse them via an
extraction or a copy — and a copy of the 738 derived-row resolver plus the 774
path-dep closure is a copy of the entire instrument. Those two rules are what
separate **25 NAMED from 17 false UNCOVERED**; the audit's own history is three
confident wrong answers in a row, all of them in exactly this classifier.
Extracted as `RepoSurface` + `classify_repo()`, with `main()` rewired to consume
it: verdicts byte-identical before and after (25 NAMED · 2 BY-DEP · 0
UNRESOLVED · 1 UNCOVERED over 213 files / 28 packages / 16 repos), five-verdict
`--self-test` still green, and the sweep's own selftest **invokes that canary**
rather than restating it.

**Three pin shapes, one per bucket, and none of them interchangeable:**

| bucket | pinned as | why not the others |
|---|---|---|
| UNRESOLVED | `max_unresolved = 0`, a WALL | the audit refuses to fold it into either neighbour — "a human has not answered it" — and a ratchet on *unanswered* is a backlog. It reached 0 by being answered (738 T1: 15 → 0; 774 kept it there) |
| UNCOVERED | **membership**, `scripts/wasm32_uncovered_expected.txt` | a `max_uncovered = 1` count goes green the day the pinned row is repaired and a different package regresses. A count is not a checksum over a set |
| population | per-repo floors **plus a global `TOTALS` row** | `min_files` and `min_packages` are both **0 in 7 of 16** repos (no wasm32 surface at all), and unlike `orphaned_attr_drift_floors.txt` there is no third quantity. So an instrument blind EVERYWHERE passes every per-repo floor |

That last one is not hypothetical. The audit walks via `git grep -E` — POSIX
ERE — and a Python `\s` in the pattern made its first version report a walk of
**0 files** with a full bucket breakdown printed over it. It was caught only
because the walk size prints next to the verdict. `TOTALS` is that observation
turned into an assertion, and a pins file without that row is **refused**
(exit 2), not silently skipped.

The membership pin reds in **both** directions. A package that stops being
UNCOVERED is a pin that has stopped asserting anything, and leaving it makes the
next real regression at that address read as already-known — so it must be
dropped in the commit that covered it.

Canaries, all measured: removing the pinned row makes it `⛔ NEW` and reds;
adding a covered package (`riir-shader-core`) reds in the other direction;
raising `TOTALS` min_files to 500 reds on the only floor that catches a grep
regression; deleting the `TOTALS` row exits 2. Restored state re-verified at
rc=0.

`mmorpg-remaster: mmorpg-poc-submodule` is the single pinned row and is a
deliberate NEGATIVE CONTROL — excluded from its repo's CI, depended on by
nothing, that repo read-only from here. It is also what proves the Issue-774
by-dep credit did not become a blanket amnesty.

Four repos deliberately unpinned, as in the two sweeps before it. **TOTALS is
the row that most needs the full checkout**: riir-dapps and riir-deployer both
drive wasm32 lanes over derived unit lists, and the canonical 20-repo figure is
216 files / 29 packages against the 213 / 28 this box can see. That makes
**four** floors files owed one visit — `platform_dead_code`,
`subprocess_encoding`, `orphaned_attr`, `wasm32_surface`.

With this, every cross-repo class in `scripts/` whose verdict is **walled at a
small number** has both halves, and the three landed today (783, 784, 785) were
all the same defect — a rule that existed in one place and was re-asserted by
nothing.

⚠ **"Every cross-repo class" would be an over-claim, and the exception is
measured.** `suite_membership_audit.py` is cross-repo and has no verdict half
**by design, stated in its own header**: its actionable column is LOAD-BEARING
unpinned test targets, and that column is **1,203 rows across 15 repos**
(katgpt-rs 442, riir-ai 410, riir-train 223, …, against 3,036 targets). A
ratchet over a backlog that size is a number nobody acts on — the same argument
`staged_set_audit.py` records for refusing a pre-commit hook, and
`highwater_contiguity_audit.py` for staying report-only while its reset verdict
lives in `numbering_drift_sweep.py`. The distinction that makes 783/784/785
gateable is that each one's ceiling was already **0** and had been earned.
`gguf_header_audit.py` is not in this family at all — it is model-file
introspection, not a class audit.

Issue file removed per the noise-reduction rule; the full record lives in git
history (`git log -- .issues/785_the_wasm32_surface_standing_figure_is_asserted_by_nothing.md`).

## Issue 800 Arm C complete — GraphStablePool site re-points: 1 landed, 1 N.A., 2 declined on evidence (2026-09-16)

Arm C closed the same day it opened. Phase 1 (`877e06eb2`) extracted the type;
the re-point pass (`bdb1091a4` katgpt-rs, `35108b6a7` riir-ai) settled all four
sites, and the honest split is the finding: **one re-pointed, one N.A., two
declined** — each verdict with stated evidence, in the `graph_stable_pool`
module doc site tables.

- **Site 1 `radix_prefix` — RE-POINTED.** `RadixPrefixTree.nodes` is now
  `GraphStablePool<RadixNode>`; the inline `Vec<Node>` + `free_nodes` stack is
  gone (the pool's alloc/free ARE it). Liveness became slot occupancy: eviction
  `free`s the victim, dropping the node value — token/page buffers release
  eagerly instead of parking cleared-but-allocated until the next overwrite
  always dropped them un-read. The pool gained `iter()` (live-slot `(index, &T)`
  scan) for the three arena scans the tree already ran (LRU eviction victim
  pick, `total_locks`, `for_each_held_page`); the `!tokens.is_empty()` zombie
  filter is structural now. Wiring: `radix_prefix_cache` implies
  `katgpt-core/graph_stable_pool` (the pool module is feature-gated). Gates:
  bench_762 GOAT release PASS (G1 bit-identity + isolation + address/pool
  stability; G2 hit-rate 0.184 vs flat 0.075, match latency 0.22 ms vs 2.17 ms
  flat — the one-`Option`-layer walk overhead is invisible at the 10× margin;
  G4 0 allocs), radix_prefix_cache_tests 14/14, pool lib 7/7 (incl. the new
  `iter` contract test), clippy `-D warnings` clean, default + wasm32 checks
  clean. The LIFO order contract held exactly: pool free-stack order reproduces
  both the old evict-push and the `clear()`'s `(1..len)` build (highest index
  pops first).
- **Site 2 `PagedKVCache` — DECLINED.** Its recycle contract is refill-in-place
  at a stable index (`alloc_page` pops the free list and `fill(0.0)`s the SAME
  `Vec<f32>` — the DDTree fork/rollback churn path is why the zero-realloc
  property is load-bearing, gated by riir-engine's forward_paged G4 tests). A
  pool re-point must park the recycled buffers OUTSIDE the pool between
  `free` and `alloc` (the returned-value escape hatch) to keep that property —
  one bookkeeping structure (`free_pages`) becomes two (pool free-stack +
  buffer stash). Add: `pages`/`free_pages`/`page_ref_counts` are `pub` and are
  the deliberate measurement instrument of bench_414's legacy-replica oracle +
  root transformer tests; and the pool's None-on-freed safety targets a
  use-after-free class that rollback's table truncation already prevents
  (tables truncate BEFORE pages free). Contract-matched lineage row stands —
  the pool was distilled FROM this shape; re-pointing is not what makes the
  claim true.
- **Site 3 `Qwen38LaneSet` — N.A.** Reading the code settled what the module
  doc had predicted: no free list, no slot churn, fixed `n`, flat
  allocate-once `alloc_zeros` arenas, and the captured-graph cache living
  INSIDE the set so whole-set replacement drops the baked device pointers
  with the arenas. The set already IS the lifetime-scoped flat-arena
  discipline; a pool wrapper adds alloc/free machinery nothing calls. Verdict
  recorded in the struct doc (riir-ai `35108b6a7`).
- **Site 4 `BranchBank` — DECLINED.** The wire format pins the exact shape a
  pool adoption would destroy: slots serialized by value with in-band
  `Removed` zombie lifecycle slots, plus the EXPLICIT `free_slots` stack in
  stack order. A pool-backed rebuild cannot reproduce from_bytes byte-identity
  without exposing the pool's internal free-stack order (an implementation
  detail), and the bytes feed neuron-db freeze — a wire change is a versioned
  migration event, not a refactor. The type works and is in-repo; churn
  unjustified.

The lesson generalizes the Arm B one: **a four-site contract extraction earns
its keep at the sites whose whole job is the contract; sites where the
contract is a subset of a bigger load-bearing shape (refcount policy, wire
pin, graph-pointer scoping) are better left as the lineage the type was
distilled from.** Decline-with-evidence is the same honest verdict class as
Bench 800-B's channel tie.

Issue file removed per the noise-reduction rule (HISTORY + the module doc
site tables + Bench 800-C are the record).

## The full gate's wasm32 layer counted NEGATIVE cfgs as surface — derivation fixed positive-only (2026-09-16)

The 802-followup clippy sweep (queue item 3, last run 09-13) opened with the
full gate red at layer 2b: three "new" wasm32 sites not covered by a `--lib`
lane — `benches/plan598_refinement_marginal_bench.rs`,
`crates/katgpt-core/tests/bench_779_real_bank_affinity.rs`,
`tests/refinement_marginal_tokenizer_bridge.rs`. All three carry
`#![cfg(not(target_arch = "wasm32"))]` — native-only guards, each compiling
to NOTHING on wasm32 by its own declaration. The layer's bare
`git grep -l 'target_arch = "wasm32"'` cannot see the negation, so every
native-only test/bench added to the tree demanded a residue pin — the exact
class `wasm32_surface_audit.py` was built to fix (Issue 738 T3: "the
predicate is the positive cfg"). This is the fourth instrument to meet the
class, and the stale bomber rows proved the cost in advance: their pin reason
describes a pre-guard crossterm failure the file no longer has.

The fix (`448c77f91`) is the derivation, not a bigger pin list: `WASM_FILES`
now counts a file only when it carries a COMPILE-TIME, non-negated wasm32 cfg
— `not(...)` guards and runtime `cfg!` branches are both excluded (a runtime
branch compiles both arms on every target, so the native --all-targets lane
already compiles the wasm32 arm). Line-based, with the measured limit that a
multi-line `not( ... )` wrapper over-includes — the safe direction (the file
lands in the residue pin and demands a human read instead of silently
vanishing). `WASM_RESIDUE_EXPECTED` is empty by construction; the two GOAT
targets stay built FOR wasm32 via `WASM_EXTRA_TARGETS` (the lane is their
pin, unchanged). Derived `-p` list unchanged: katgpt-core, katgpt-moka-wasm,
katgpt-types + the root package.

The same sweep run then paid for itself on the layer it had been shielding:
layer 3 (workspace × all-features × all-targets × the -D list) had not
executed since the 2b red, and it found 3 `needless_range_loop` errors in the
Plan 598 test code (landed `0fbac8248` under default features — the
all-features axis gap in the landing gate, the repo's most-repeated rule
again) plus an unused import in the 598 bench that only compiles under its
feature. Fixed in the same commit; `naive_marginal` became
`enumerate().take(n)` + the brute-force compare became `zip().enumerate()`,
semantics identical (5/5 module tests). Seven pre-existing feature-gated
WARNINGS remain, none in the -D list, left to their owning lanes — one is a
bench-loop semantics question (bench_775 single-element loop), not a
mechanical fix.

First full-gate PASS on this box after the fix (`✓ full gate PASSED — 0
errors, 0 unbuildable targets`), and the executed axis held too: test_gate
203 · 2060 · 249 · 139, every row at its floor. The slt.rs
`field_reassign_with_default` blocker (`dfa6d3ff0`, fired under
--all-targets only since `7352a75ab`) was cleared on the way in.

## Issue 844 (2026-09-19) — the dot-delegation crossover measured on BOTH arches; the NEON answer refuted the filing session's own expectation: CLOSED

T4 ran `bench_844_dot_delegation_crossover` on the M3 (aarch64/NEON, three
runs, ratios stable to ±0.13) and the result is the strongest kind of
refutation: **there is no crossover at any measured length — delegation wins
at EVERY length ≥ 4** (1.94–2.02× @4, 1.70× @16, 3.19–3.32× @32, 9.06–9.08×
@256). The filing text predicted "crossover expected LOWER" (NEON 4-wide vs
AVX2 8-wide); the truth is the crossover sits below length 4, where no real
site lives. Both mechanisms are legible: the aarch64 dispatch is compile-time
(`cfg` + direct call, ~0.8 ns kernel floor) while x86_64 pays a runtime CPUID
probe (~3.3 ns floor), and the plain runtime-length loop is slower on NEON
than x86_64 at every length (1.5 vs 1.1 ns @4; 113 vs 78 ns @256). The rule
is now per-ISA — and rule 1 ("plain small-D loop is CORRECT") is
ISA-conditional: the same loop concedes 1.7–1.9× on NEON, so arch-weighted
crates (game runtime, wasm32) should lean to delegation even at D=8–16.

The bench's own "no crossover" print branch was repaired in the same change:
it read as a fixed-overhead-model violation ("should be investigated before
being quoted") when the truth on NEON is the model WORKING — the branch now
names the arm that won throughout.

T2/T3 closed with a full workspace per-site read (re-derived census: 398
fn-dot defs → 247 src candidates after excluding test trees and
`katgpt-types/src/simd/**`; three parallel reads, the sharp findings
spot-verified by hand):

- **8 CHUNKED findings** (rule c): `katgpt-core/cgsp/types.rs:41
  dot_f32_fma4` and `katgpt-kv/still_kv/perceiver.rs:486 dot_chunk4`
  (katgpt-types already a dep of both — one-line delegations); `katgpt-dec/
  simd.rs:50` (⚠ zero-dep-by-design crate published to crates.io — adding the
  dep is an owner call, not a heal); riir-ai `cross_game_prefix.rs:528`,
  `motivation/math.rs:38` (STAT_DIM=16, arch-split decision recorded),
  `lora_still_forward.rs:575`; riir-train `embedding_translator/model.rs:568
  dot8`, `edge_lora/sigmoid_gate.rs:233 dot_product_chunked`.
- **10 large-D naive candidates** (delegation wins on both arches): riir-ai
  982 carries six (64/32-dim engine + poc sites), riir-neuron-db 621 carries
  two (both fixed-64), plus katgpt-rs' own `score_matrix_simd.rs:121
  dot_8wide` (runtime head_dim, perf-tested at d=64) and
  `specialist_projection.rs:206 dot_truncated` (d_hidden ≥32).
- `newton_schulz::blocked_dot8{,_neon,_scalar}` adjudicated **KERNEL-HOME**
  (batched 8-output GEMM micro-kernel — one accumulator per OUTPUT, not the
  multi-acc-within-one-dot shape; remainder columns already delegate).
- The 56-UNRESOLVED bucket resolved as predicted: accumulate-into-slice
  (`rrq_quant::dot_acc_into` — dequant-fused GEMV), tropical semirings,
  const-generic wrappers, `#[cfg(test)]` fns, f64/i8 families, name
  collisions (DoT damage-over-time, UI dots). **No hidden findings.**
- The `katgpt-moka-wasm` exemption is SOFT (its manifest declares
  katgpt-types) — recorded, not acted on.

Disposition: repairs transfer to the owning repos — **riir-ai Issue 982**,
**riir-train Issue 562**, **riir-neuron-db Issue 621** — each carrying the
repair contract (delegation changes summation order, max |Δ| ~3e-6 @64;
adjacent gates re-run on repair; determinism-contract sites out of scope).
The kron_tile consumer note is now ISA-conditional too (its n∈{8,16} cost is
x86-only; on NEON those widths WIN 1.8×/1.7×).

## Issue 849 (2026-09-19) — the 844 per-site dot read, katgpt-rs-own sites: four delegations landed + the full repair record-back: CLOSED

The 844 disposition's katgpt-rs-own sites are repaired, and every
sibling-repo repair verdict is recorded here per the disposition's contract
(this section IS the T5/T3 record-back for riir-ai 982, riir-train 562, and
riir-neuron-db 621).

**Internal repairs (this commit):**

- `katgpt-core/cgsp/types.rs` — `dot_f32_fma4` (4-acc FMA chunk) renamed
  `dot_f32` and delegated to `crate::simd::simd_dot_f32` (HLA width,
  default 64: 3.13× x86_64 / 5.2× NEON). The BLAKE3 snapshot commitment
  hashes serialized bytes — verify re-hashes data, never re-derives scores —
  so the "Deterministic" contract holds per binary/arch. Gate:
  `--features cgsp` 44/44.
- `katgpt-kv/still_kv/perceiver.rs dot_chunk4` → delegates (runtime
  `head_dim`, wired >=32; fn name kept — the private test references it).
  Gate: `--features still_kv` 18/18.
- `katgpt-attn-match/score_matrix_simd.rs dot_8wide` → delegates to
  `katgpt_core::simd::simd_dot_f32`. katgpt-core made NON-optional in that
  manifest: score_matrix_simd is default-compiled while the dep was
  optional-only, and the alternative was a cfg-dual kernel — the exact
  Issue-845 anti-pattern; `publish = false`, so the posture change carries
  no crates.io weight. Gates: default `cargo check` + `--features maxsim`
  12/12 + `--all-features` check.
- `katgpt-sparse/specialist_projection.rs dot_truncated` → delegates at
  `a.len().min(b.len())` (truncation semantics preserved; d_hidden >=32).
  Gate: `--features specialist_projection` 39/39. Clippy clean on all four
  crates.

**Deferred (owner call, the katgpt-dec zero-dep posture):**
`katgpt-dec/simd.rs:50` carries its own local `simd_dot_f32` in a
zero-dep-by-design crate published to crates.io — adding the katgpt-types
dep (or accepting the duplication) is an owner decision, not a heal.

**Sibling repairs recorded back (the disposition's transfer contract):**

| repo | issue | commit | scope | gates |
|---|---|---|---|---|
| riir-train | 562 | `3cf49ebb` | `dot8` (dims recorded: mixed 12–32, tiny cfg only) + `dot_product_chunked` (single-kernel property preserved) | engine 11/11 incl. gradient checks; gpu edge_lora 197/197 |
| riir-neuron-db | 621 | `5d86dd3` | `transition_error_taxonomy::dot` (64) + `hebbian_bridge::phi_dot` (64, truncating) | lib 54/54 |
| riir-ai | 982 | `dbc5639d4` | F1 cross_game_prefix / F2 motivation math (arch-split → delegate) / F3 lora_still_forward (mixed-arch → delegate) / C1 cce signal (64) / C2 log_salience_dot (32) / C3 kg_hyperedge (sibling pattern) | engine 159+1ign / civ 329 / gpu 15 |
| riir-ai (deferred) | 982 T4 | — | riir-poc C4–C6 (32/64/64) — `[-]`, next poc-touching pass | — |

All sites verified no bit-determinism contract before delegation; the
summation-order change (max |Δ| ~3e-6 @64, measured) passed every adjacent
gate. The kron_tile consumer note stays ISA-conditional (844 disposition).
Issue file removed per the noise-reduction rule.

## Issue 848 (2026-09-19) — a rename privatised a delegation target and a rework deleted a shared fixture; both of that module's EXTERNAL callers are docs-gate CHECKS: CLOSED

`6c6ca2ee` reworked `scripts/worktree_state.py`'s own arms and, beside the
repair, renamed `is_checkout` → `_is_checkout` (updating the seven in-module
callers and neither of the two external ones) and deleted `worktree_fixture`
outright. `console_encoding_gate.py` and `sweep_advisory_membership_gate.py`
consume both names and are docs-gate CHECKS; both died with an
`AttributeError`. **`develop` was red for 6h40m** — not a wrong verdict, NO
verdict, which is Issue 804's class one seam over.

⛔ **Both names carried a written contract NAMING their consumers**, so this
was not a judgement call that went the other way: AGENTS.md's own line
(*"`worktree_state.is_checkout` — delegate to it too"*) and the deleted
fixture's docstring (*"Public, because the three delegating consumers each owe
an arm … and that arm needs exactly this fixture"*). The document said the
right thing; nothing read it — `instrument_reachability_gate`'s finding one
level down.

Nothing caught it because Python has no link step and every lane that could
EXECUTE those modules was out of reach: `docs_gate.yml` is main-only since
2026-09-09, `full_gate`/`test_gate`/`x86_64_execution_matrix` are Rust lanes,
and `arm_reach_gate` — which does execute them and would have reported
`BASELINE-CRASH` — is a 157.6s workstation verdict, not a CHECK.

### What landed

- **T1** — both names restored; `worktree_fixture` came back verbatim from
  `6c6ca2ee~1` with a comment at the definition recording why it is public,
  so the next rework reads the contract at the line rather than in a docstring
  it is deleting. `worktree_state` selftest 144 assertions, docs gate green.
- **T2** — `scripts/cross_module_attr_gate.py`, the STATIC wall: every
  sibling-module attribute and `from X import n` resolved against X's
  top-level names, re-exports and conditional definitions credited, keys
  LINE-FREE. Two-sided known answer — **0 findings at `6c6ca2ee~1`, exactly
  those 4 at `6c6ca2ee`**, 0 in the repaired tree, over 90 files with 0
  unparsed. STATED and printed: `getattr`, star-import OPAQUE modules,
  function-local names, and `__all__` (deliberately unread — it constrains
  `import *`, not attribute access, so reading it would INVENT findings).
- **T3** — `scripts/import_health_gate.py`, the EXECUTION wall, and its
  affordability measurement found its own blocker: **6.238s of the 6.34s** an
  import pass over 86 modules cost was ONE module whose entire body was
  top-level (`list_unresolved_percentile_sites`, guarded in the same change;
  `arm_reach_gate` had been paying that per mutant). A per-module subprocess
  was measured too and is not worth 3x (10.28s vs 7.52s, identical verdicts).
  **MISSING-DEP is its own bucket and never flagged** — a pin there would make
  the gate box-dependent in the worst direction, correct on a box without the
  package and STALE on one with it.
- **T4** — the cross-repo question, counted and NOT inherited, and the two
  ways of counting disagree by an order of magnitude: **10 of 17 repos carry
  tracked `scripts/*.py` (196 files)**, but **713 of 749 resolved references
  are in this repo**. Only the second is the class's population, so: no sweep,
  and `cross_module_attr_gate.py --workspace` re-derives the table every run
  rather than leaving the figure in prose.

⚑ Landing T3 found two more live defects in gates that already existed, which
is the argument for a small check over a big one: `subprocess_encoding_gate`
red on a `PYTHONIOENCODING` env dict bound to a LOCAL rather than written at
the call site, and `console_encoding_gate` red on
`list_unresolved_percentile_sites` — which entered that gate's population for
the first time, because adding a `__main__` guard is what makes a module an
INSTRUMENT by its predicate. **A repair that grows a population owes the other
gates a run.**

⚠ This is not an argument against the main-only CI call — that is an owner
decision about Actions spend and it stands. What it records is that the
resulting `develop` lane is **zero, not reduced**, which `ci_gate_coverage.py`
already prints for 12 of 16 repos; this is the first time it cost this repo a
red `develop` in Python rather than in Rust.

## Issue 856 (2026-09-19) — the green zero has TWO spellings and the audit built for it saw one: CLOSED

`cfg_gated_target_audit`'s predicate was a single regex for the **inner**
attribute. A file whose whole body is `#[cfg(feature = "x")] mod tests { … }`
zeroes its binary identically — `ok. 0 passed`, exit 0 — and the comment
beside that regex distinguishes `#![cfg]` from `#![allow]`, never deciding
anything about the outer-on-a-module form: an **unstated** blind spot, which
is worse than a scoped one. It printed `SILENT-NOW 0` for this repo over 26
such targets, hiding **175 assertions**, 7 of the targets named `*_goat`.

T1–T5 closed the same day. Classifier widened (`whole_body_cfg_mod`, sharing
`platform_dead_code_audit.mask_file`), so every consumer inherited it; the
predicate is *the gated items are the WHOLE body*, with `tests/test_freeze_thaw.rs`
as the measured true negative that corrected the filing's own count. A run of
gated modules is gated by `any(...)`, not `all(...)`; a top-level `use` never
makes a target non-empty. `cfg_row_implication_audit` shares the predicate or
the repair would have reproduced the defect one gate over. Twelve
`required-features` rows added and verified by the compiler in both
directions. Two sibling repos breached their pins and were repaired there
(`seal-remake 964e780`, `seal-game-editor fbb5a931`); no ceiling was raised
anywhere.

⛔ The first cut anchored its module matcher with `\A` and passed a non-zero
`pos`, so it returned `None` for every file — **the repair read exactly like
the defect**, and it was caught by the summary line not moving rather than by
anything failing.

Full record, with the before/after table, the per-repo cross-repo read and the
four perturbation canaries: `.docs/10_audits/cfg_gated_silent_zero_pass.md`
§"The SECOND spelling". Fix commits `4e2f28f2d` · `6399faf69`.

## Issue 860 (2026-09-20) — `successor_density_critic`: tabular discounted count-ratio goal-critic: CLOSED

The modelless CRL extraction (riir-ai Research 386; arXiv:2206.07568) landed
complete at `2c7a1f157`: dense `[S][A][S]` f64 weighted-count tables,
hindsight-geometric observation as ONE O(L·G) reverse sweep, `Discounted`
(default, exactly consistent) + `CLearning` (parity variant, divergence
recorded — non-default by design because it breaks the G1 oracle's exactness
certification), BLAKE3 freeze/thaw at full bit precision, Lemma-4.1 ranking
invariance as an executable property. **Bench 818 GOAT PASS** — G1a exactness
0.00841 ≤ 0.01 against the behavior-continued Bellman fixed point (the first
draft's greedy-action-repeat closed form was the WRONG oracle and the gate
caught it), G1b ranking 128/128 decisive, G1c prior-perturbation
bit-identity ×0.001…×1e6, G1d byte-identical freeze, G2 absolute budgets
(score 0.9 ns / argmax_a 4.3 ns / argmax_g 44.0 ns), G3a–c structure/prior
bite/0 discordances vs 95 raw-count, G4 zero allocs — release AND dev,
box state M3 (this arch; no 858-class cross-arch exposure).

The consumer pull-gate half — the issue's one open item — is **satisfied**:
riir-ai Issue 991 lane (b) landed `goal_salience` (opt-in), which forwards
`katgpt-core/successor_density_critic` in `riir-engine/Cargo.toml` (verified
by grep at close, not inherited from prose), consumer code at
`riir-engine/src/cgsp_runtime/goal_salience.rs` + `riir-games` swarm, GOAT
mechanism-gate riir-ai Bench 949 (6.3× first-reach) + QUEST-WORLD promotion-gate A/B
riir-ai Bench 950. riir-train Plan 413's tabular arm remains that repo's own lane at
adoption. The feature **stays opt-in** as written: promotion rides the
consumer lanes, and riir-ai's promotion-to-default is production-host-gated
(owner). Catalog §119; bench `bench_818_successor_density_critic_goat`.
Hygiene close: 12/12 module tests green on `develop` at removal
(`successor_density` filter, feature on).

## The exact_sigmoid / dot_f32_ordered substrate promotion (2026-09-20) — the riir-chain Issue 156 T1 landing executed in this repo

Landed `5458dd69b` (develop) + `5e2b730f2` (main, cherry-picked via a
detached worktree following the lthash precedent — the git-dep consumers pin
main, and the branch topology flag remains owner-gated). Three ungated pure-
math primitives, additive, always compiled (the `float_order` precedent):
`exact_sigmoid(f32)` / `exact_sigmoid_f64(f64)` in
`katgpt-types/src/simd/activations.rs` (re-exported `katgpt_core::`), the
two-branch libm form with NO Cephes polynomial and NO ±40 clamp; and
`dot_f32_ordered(&[f32], &[f32])` in `katgpt-types/src/simd/dot.rs`, the
sequential index-order fold that is deterministic by construction.
`katgpt_core::sigmoid`'s doc now names the exact variant — the trap where a
new caller silently gets the approximation is closed at the doc seam.

**Bench 844 GOAT PASS** (`844_exact_sigmoid_ordered_dot_substrate.md`):
G1a f32 exactness max **2 ULP** vs the f64-computed-and-narrowed reference,
against `fast_sigmoid`'s **580,601,137 ULP** on the same grid — clamp-
dominated far tails where the abs-error column does not separate the
variants, which is why the ULP metric is the load-bearing one; G1b f64
reflection/monotone/bounds (no ULP oracle — the f64 impl IS libm, an
"≤1 ULP vs libm" gate would be circular) + f32 bounds incl. the
representable-nonzero far tail; G1c ordered-dot frozen-value pin (the
`[1e8, 1.0, −1e8, 1.0]` cancellation input reads 1.0 where every SIMD
backend reads 0 — the anti-dedup pin, target-independent by design: it reds
the day a backend converges with the sequential fold); G2 reported not
barred; G3 no-regression by construction (additive, no call site rerouted);
G4 alloc-free by inspection. **Honest finding:** the Cephes speed claim
inverts on aarch64 — `exact_sigmoid` 1.7 ns vs `fast_sigmoid` 3.1 ns
(best-of-50 minima, loaded box, `/tmp` target dir; upper bounds) — the
`fast_sigmoid` doc's "~1.7× faster than libm" must not be quoted on this
arch un-re-measured.

Consumer (same unit): riir-chain 156 T2 delegated
`curator_bridge::{sigmoid, dot_product}` + `forensic/recover::sigmoid` to
these fns behind permanent `to_bits`-level bit-identity pins carrying the
frozen legacy bodies (`f7eb85e4` + `b0b710e0`, then `7facc3cd`), green
before AND after the flip; `consensus/congestion::inclusion_probability`
refused-and-recorded — its `x < 0` domain is reachable through pub inputs
and a consensus-path numerics change is its own decision. The ≥5 in-repo
copies of the two-branch shape were delegated by the recorded follow-up
(Issue 861, executed 2026-09-21 — `salience/gate.rs`, `breakeven/mod.rs`,
`refinement_marginal::escalation_sigmoid`, `ugc_schedule::
inv_log_reveal_odds`, `successor_density_critic::p_successor`; the
deliberately-not-delegated set — the link-identity TEST oracle, gate.rs's
FMA-convention `dot_fma`, and riir-chain's congestion refusal — recorded in
the issue, file removed at close, this row + the Bench-844 follow-up section
are the record). README showcase section landed with this row (the
float_order surface set: README + HISTORY; ungated primitives take no
catalog row).

## Issue 865 T3 (2026-09-21) — the probe_guidance λ-sweep GOAT gate ran NEGATIVE; the negative verdict is the pinned gate: OPEN (lane re-opens at Bonsai scale)

The gate (Bench 847, `9c2acbfaa` + `3ebc51944` for the T4 scoping): the
trained-probe guided front vs the unguided temperature front on the
mini-dLLM lane, with two controls that carried the verdict — a ZERO-logit
probe (with which the λ combine is EXACTLY temperature scaling
`λ·logits = logits/(T/λ)`: the no-information null, not an RNG re-roll — the
first calibration draft mislabeled it and the arithmetic corrected it) and a
mean-zero token-0 directionality control (monotone −1.17 pts: the harness
can see a followed direction). Measured: the guided best (λ=1.25, +0.21 pts
at lower resample diversity) is DOMINATED by unguided T=1.0 at matched
diversity (100.00% at the same 3.2171 nats), and the trained probe loses to
the null at every λ ≥ 1.5 (−2.11 pts at λ=2). Root cause measured, not
argued: the T2b trunk is saturated (loss 0.0000, one-hot) — the weak side
(side: the trained probe, held-out CE 0.2166, G-health PASS) carries no
disagreement the extrapolation can exploit. G1 PASS (λ=1 bit-identity,
pipeline level, real artifact) and G4 PASS (1,000 probe calls, 0
allocations) — the MACHINERY is qualified; the MECHANISM verdict needs a
non-saturated trunk.

Three lessons encoded in the pinned gate (`tests/probe_guidance_goat.rs`,
deterministic, fixture-pairing canary G0 < 0.5 CE against trunk drift):
(1) pooled unigram entropy is polarity-inverted on deterministic-structure
lanes — a CORRECT decoder maximizes it (the ground truth is itself
high-entropy) — per-position resample entropy is the working form;
(2) the zero-logit null gives every λ sweep a free no-signal reference —
read the trained arm AGAINST it before believing any small bump (the
apparent λ=1.25 win is +0.21 over the null's +0.00, and gone by λ=1.5);
(3) G2a/G2b are INVERTED into regression pins that red the day guidance
genuinely wins — the promotion decider is pre-wired, so the Bonsai-scale
re-open (multi-layer kernel extension + a trunk with headroom; T4's AR arm
folds into the same scope, both prerequisites shared) needs only to run it.
The feature stays opt-in; the dropout-autoguidance arm deferred (no kernel
dropout exists; moot post-verdict; owner scope at re-open). Full data +
box state: `.benchmarks/847_probe_guidance_lambda_sweep_goat.md`.

## Issue 866 (2026-09-22) — KARC D3 promotion coverage audit: VERDICT QUALIFY — the contract's passing legs live on configs nobody constructs: CLOSED

The audit (Bench 849, `crates/katgpt-core/examples/karc_deployed_shape_quality.rs`):
Bench 308's D3 split-config G1 contract (NRMSE ≤ 1e-3, threshold ≥ 8 LT) was
never measured on any config a consumer constructs, and no downstream gate
supplies a substitute. Three findings carry the verdict. (1) Both passing
legs sit on `ChebyshevBasis` (NRMSE at K=8/M=8/R=2 λ=5e-2; threshold at
K=8/M=24/R=1 λ=5e-3); every deployed monomorphization is `FourierBasis` R=1
(Lod0 F<4>/K=2, Lod1 F<8>/K=4, Lod2 F<8>/K=8, period 4.0), and all three
fail BOTH D3 bars on the D3 record's own double-scroll fixture (1-LT NRMSE
2.4–53; threshold 0.03–0.16 LT) — first-order Fourier cannot reconstruct
that attractor (the paper's headline needed second-order; Phase 5.3's R=1
floor was the Chebyshev one). (2) The six riir-engine `karc_runtime` GOATs
measure divergence / curiosity-ratio / detection / wire-exactness / latency
/ commitment — none measures absolute forecast accuracy, so the D3 contract
was never ratified downstream either. (3) The deployed consumer exercises
ONE-STEP forecasting from observed delay rings (`tick_karc`, re-fit each
`tau_reest`) and never rolls out autonomously; on a Lorenz-driven
leaky-belief fixture built through the runtime's own `leaky_step` math, the
deployed shapes' one-step NRMSE is 1.2–4.1e-3 at the deployed λ=1e-4 — the
deployed configs' quality record per the issue's QUALIFY arm. The same
fixture measured the autonomous rollout as violently unstable (~2.2×/step
amplification, λ-independent) — irrelevant to `tick_karc`, a live
precondition check for any future multi-step/rollout consumer. Not DEMOTE:
the consumer-relevant properties are separately gated and one-step quality
is strong. Two stale statements corrected in riir-ai in the same window
(Plan 332's "shape fixed at Plan 308 GOAT" line; `karc_bridge/lod.rs`'s
"3-variant enum" doc block — it is 2-variant, Lod1 never dispatches). T2's
`faer` spike stays conditional — the second-heavy-BLAS-consumer trigger is
still unmet. Scope limitation recorded in Bench 308 (addendum) + the Phase
22 feature-def comment; proposal status line updated. Full data + box state:
`.benchmarks/849_karc_deployed_shape_quality.md`.

## Issue 865 file hygiene (2026-09-22) — issue file removed per noise-reduction, the lane's record was already durable: CLOSED (hygiene)

The file's own status line said it: T1/T2/T5 LANDED, T3 measured NEGATIVE
(Bench 847), T4 DEFERRED — "nothing remains actionable in this lane until
the multi-layer kernel extension lands; re-open there." Every landing and
defer already has its durable home (the T3 HISTORY entry below, the catalog
row `.docs/09_feature_catalog/opt_in_features.md` §120, Research 578,
riir-train recipe rows A/B, and the G2a/G2b inverted regression pins in
`tests/probe_guidance_goat.rs` that red the day guidance wins). The file was
therefore pure noise; removed with no content change. Re-open path
unchanged: the Bonsai-scale re-open (multi-layer kernel extension + a trunk
with headroom) files its own issue where G2a/G2b are the pre-wired promotion
decider.


## Issue 865 follow-up (2026-09-22) — arm (b) unblocked (DropoutHeadProbe) + the headroom study: the negative EXTENDED to every mini-lane regime (Bench 850)

The sibling-session T3 record (Bench 847) deferred the dropout-autoguidance arm for lack of a
kernel-dropout substrate and named the saturated trunk as the negative's root cause ("the
mechanism verdict needs a trunk with headroom"). Both halves were closed by this follow-up:

- **`DropoutHeadProbe` ships** (`katgpt-forward/src/weak_probe_mlp.rs`, +3 lib tests): the
  frozen trunk head over a deterministically 50%-dropout-masked TAP — fixed LCG stream keyed
  by `(position, denoise step)`, zero runtime RNG, zero-alloc. Masking the tap needs no
  inference-time kernel dropout, so the deferral reason is dissolved. The weak side stays a
  noisy version of the SAME function (the correlated-dynamics requirement), unlike the
  structurally-damaged truncation class.
- **The headroom rerun ran under Bench 847's own methodology** (per-position resample
  entropy, the zero-logit null `λ·logits ≡ logits/(T/λ)`, the temperature front, 256 prompts
  × 8 resamples) across THREE regimes: high-data 12-epoch trunk (2048 seqs), low-data
  12-epoch trunk (96 seqs), and the strict-decode cell (τ_conf 0.7 / 8 steps — the
  decode-uncertainty regime where the unguided front spans 82–99%). **The dropout arm never
  beats the null — at no λ, in no regime** (beyond-sharpening deltas −0.13 … −1.42 pts; at
  matched diversity vs the temperature front, every point negative except two inside the
  pinned noise envelope).
- **A preliminary positive was retracted with cause**: an in-session run of the strict-decode
  cell on the POOLED unigram-entropy axis (Bench 847's refuted metric — on
  deterministic-structure lanes the axis measures correctness-collapse with inverted
  polarity) had read "+2.7 pts at matched entropy" for the same dropout arm. Under the
  resample axis + the null it is entirely explained as redistribution along the axis the
  pooled metric secretly measures. Second measured instance of the axis lesson.
- **The re-open condition is sharpened**: the mini lane is structurally incapable of a
  modelless guidance win (three trunk/decode regimes; training-time headroom does not
  survive the decode loop; the strict-config decode uncertainty is unstructured — which
  pattern tokens commit late, carrying nothing a weak side knows better than the trunk).
  The Bonsai-scale lane (multi-layer kernel + natural-text uncertainty + riir-train recipe
  row B) remains the only path; Bench 847's inverted G2a/G2b bars remain the promotion
  decider. Study instrument committed as `tests/probe_guidance_headroom_study.rs` (asserts
  λ=1 identity per regime; prints the fronts). Full data + box state:
  `.benchmarks/850_probe_guidance_headroom_study.md`.

Session: katgpt-rs-865-followup, 2026-09-22


## Issue 876 (2026-09-23) — the flappy render widened to v3; the decoded arm reads Δ0 vs the structured arm: CLOSED

Bench 881's losslessness arm found the flappy v2 render too coarse for
decode-based consumption (decoded arm 77/100 = constant-pick, ONE distinct
pick). The fix was render-side, per the issue's two candidates, landed as
grammar v3 (`laya-flappy-v3`): a quantized OFFSET clause (fine post_rel,
clamped ±2) + a NEUTRAL post-motion clause (kinematic "drifting"/"holding"
wording — the v1 "rising"-style value-loaded motion was the measured Bench
880 confound). The structural caveat (post_v is action-determined, so the
clause names the action) is stated in the grammar docs and the fixture
meta; the measured defense held — the v3 oracle split 48 flap / 52 coast
with zero p ties, no constant-flap degeneration.

- **The oracle fixture was regenerated over the IDENTICAL 100-state set**
  (same seed, same exclusions — asserted state-by-state against the v2
  record), so the delta isolates the render. Oracle: riir-reflex
  `laya_oracle_batch` @ `63b1552`, run in an isolated worktree with the G5
  parity gate re-verified green at that exact commit before the run.
- **The gate is MET with the stronger outcome: Δ0.** Structured arm
  96/100 (in + LOO); decoded arm — fills reconstructed into the STRUCTURED
  UNITS (exact post_rel for |rel| ≤ h, crash tails at ±(h+1), post_v exact,
  pre_rel band-clamped) — 96/100 (in + LOO), 4/100 flips, distinct 2.
  G1 HOLDS, discrimination PASS; both v3 head digests pinned in full.
- **The decoded feature design is part of the measurement**: the first v3
  decoding (raw fill ordinals) read 51/100, BELOW constant-pick — a linear
  head over ordinals cannot represent the band×offset joint. Both designs
  are recorded; the structured-units reconstruction (the lanes anchor's
  pattern) is the landed one.
- **The v2 record is untouched**: `render_option_sentence_v2` frozen +
  pinned by a literal-string test; `flappy_02_arena` replays the v2
  fixture byte-identically; the Bench 880/881 anchors still assert.
- Full record: `.benchmarks/882_flappy_v3_render_widening.md` · fixtures
  README (the v1 → v2 → v3 grammar history) · Catalog §125. Code +
  fixture landed at `515230244`; this entry + the issue-file removal are
  the docs commit of the same landing.

Session: katgpt-rs-876-flappy-v3, 2026-09-23
