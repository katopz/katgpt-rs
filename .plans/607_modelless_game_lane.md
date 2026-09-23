# Plan 607 — Modelless game-decision lane (the laya game arenas)

**Status:** COMPLETE — owner-accepted 2026-09-22 ("607 accepted").
T0a + T4a + T0b + T1 + T4 LANDED 2026-09-22 (substrate gate clean, the
state enumerator + `laya-tetris-v2` grammar pinned, the G1-oracle fixture
committed with airtight provenance, the `state_option_scoring` primitive
shipped — Bench 876, and the Tetris arena replayed the fixture; first
GOAT reading: G1 (agreement) DID NOT HOLD — raw 10.8% tied the
constant-pick baseline; {T2, T3} gated OPEN). **T3 LANDED 2026-09-23 and
G1 NOW HOLDS** — the corpus-fitted head (closed-form ridge LS over the
frozen features, consuming `linalg::ridge_solve`'s f64 path) reads
**in-corpus 30.0% / LOO 29.2% vs constant-pick 10.8%** (Bench 878; the
in-corpus ≈ LOO gap is 0.8 pp — the fit generalizes at n=120). **T5
LANDED 2026-09-23 (Bench 880) and G1 HOLDS ON BOTH MICRO-ARENAS** —
Flappy v2 96/100 (LOO 96, constant-pick 77, chance 50) + three-lanes
84/100 (LOO 84, constant-pick 41, chance 33.3), zero primitive widening
(the T1/T3 surface transferred as-is); the flappy v1 grammar's motion
clause was a measured wording confound (85/100 oracle flap-bias pinning
every scorer at constant-pick) and is recorded with its structural v2 fix.
**T2 LANDED 2026-09-23 (Bench 881) — the losslessness arm splits the
arenas three ways**: lanes EXACTLY lossless (Δ0, 0/100 flips, identical
head digest) · tetris decoded BEATS structured (+8/+9 → 44/120 — laya's
read is a function of the sentence, and the side/position band the
render carries beats the numerics that lack it) · flappy decoded
DEGENERATES to constant-pick (Δ−19, distinct picks 1 — the v2 band-only
render dropped exact post_rel/post_v; the render, not the decoder, is
the bottleneck — render-side work item, not tuned here). New opt-in
`template_decode` primitive (closed-grammar template tables, loud
Unknown/Ambiguous, `verify_closed` full-space proof, zero-alloc decode);
the width-genericized fit recipe reproduces all three published anchors
bit-identically. **T6 + T7 CLOSED 2026-09-23 — ALL TASKS DONE; the plan
is COMPLETE** (PUCT/MCTS note stays DEFERRED per its row). Follow-up
filed: Issue 876 (flappy render widening — the only open lane item,
render-side).
- Lane priority: **co-developed ordering** (T0a → T4a → T0b → T1+T4 →
  first GOAT reading → T5 → {T2,T3} evidence-gated → T6 → T7).
- T2 Gate A: **approved in principle, build deferred, lapses on the first
  GOAT reading** (the lapse condition is IN T2's row — R5).

katgpt-rs Plan 607 (qualified everywhere — riir-ai `.plans/.highwater` is
ALSO 607; a bare "Plan 607" is the misattribution class). Spawned by the
owner prompt 2026-09-22 after https://brainfunctioncollapse.com/laya demoed
THREE games (Flappy Bird, three-lanes, Tetris) played via templated English
state sentences + one typed question. Verdict ping-pong round 1 = REVISE
(lane split moved to this repo, owner gate B dropped, T2 provenance fixed,
flag renamed off "game", citations repointed); **round 2 = REVISE, R1–R6
applied**: T0 split around the arena (R1 — the fixture needs the state
enumerator + renderer FIRST, and the local laya forward makes per-decision
oracles generatable, so the aggregate-degradation clause is deleted);
discrimination-floor + constant-pick/chance baselines in T1's G1 (R2); G2
restated per decision SET with the option count printed + the hot path
decided centroid-cosine-only (R3 — the old `10 µs/decision` was 6× TIGHTER
than the 0.06 ms floor it cited as headroom); T1 signature generic + T5 a
precondition for any default-on consideration (R4); T2 lapse condition (R5);
the feature-boundary fairness argument scoping T2 (R6). Round-2 AGREE
residuals also applied: the headline latency claim restated per decision
SET (with the T6 matched-units precondition) and the T1 determinism row
names the corpus centroid table.

## The thesis — why we should win the game domain

LayA's own page concedes the three facts that define its game protocol:

1. **It cannot read numbers** — "Do the arithmetic in code and hand Laya
   the conclusion in words."
2. **Action questions answer backwards** — it must be asked WHERE things
   are, never WHAT to do (their Flappy protocol: "The bird is a little
   below the gap" → P(below) > 0.5 → flap).
3. **Wording sensitivity is a measured trap** — "blocked by a barrier"
   separated lanes by 0.75, "blocked by a train" 0.45.

So laya's game protocol is: **code does the arithmetic → code renders a
templated English sentence → BERT reads it**. The sentences are
code-generated from a **closed grammar** — which is exactly the regime
where corpus-limited decode is lossless (the quest-grammar lesson, pattern
lineage only). Our modelless lane's headline weakness (0.04–0.48 on OOD
text suites, Plan 603 harness) is **irrelevant under this protocol**: the
input vocabulary is closed, and the model contributes only a
state→threshold read that a corpus-scored feature vector can match
without any text at all.

The numbers we already hold (Plan 603/606, reflex README): modelless
p99 59–79 µs per decision SET — ~20 questions answered in one call; the
0.06–0.5 ms figure is the per-SET spread across the family suites — vs
laya 21–4500 ms per forward, ONE decision per forward. The per-question
ratio spans roughly 10³–10⁵×, and **T6 must restate it at matched units
(per-question vs per-decision) before any number is quoted** — never a
per-set figure wearing a per-decision label. Zero model weights vs
650 MB + 2.3 GB download; deterministic; no GPU. Winning the game domain
is expected on latency/deployment by construction, and on accuracy by
protocol choice — the plan's job is to PROVE it with a measured GOAT
gate, not assert it.

## Lane split (verdict round 1 — the correction that matters)

- **Arena + bench live HERE (katgpt-rs).** Precedent: the bomber/monopoly
  arenas are already chartered in `BOUNDARY.md` (the `bevy_ecs` row) as
  "GOAT evidence infrastructure … an evaluation harness with no riir dep,
  upstream of everything" (Bench 432, Bench 799). Tetris/Flappy/lanes are
  the same class. A riir-ai home would put the G1 oracle DOWNSTREAM of the
  primitive it gates — katgpt-rs cannot depend on riir-ai (dependency
  cycle), so the flag's own repo could never execute its gate
  (test_gate/full_gate/x86 matrix all blind). That is the green-zero shape
  one repo over.
- **riir-ai gets a consumer LINE, not the lane** — a separate later plan
  (its own riir-ai number) for when a shipped NPC actually consumes the
  primitive. `goal_salience`/swarm are NPC latent cognition in a game
  runtime, a different domain from a decision arena; T4 does not consume
  them.
- **riir-reflex stays game-free.** Its BOUNDARY.md domain test is explicit
  ("NOT game runtime"); Plan 603/606's scope stands. The arena book cites
  NUMBERS only.

## Execution order (verdict round 2, R1)

T0a → T4a → T0b → T1 + T4 (co-developed) → first GOAT reading →
T5 → {T2, T3} evidence-gated → T6 → T7. **The fixture cannot precede the
arena**: trace generation needs the state enumerator + renderer first, and
per-decision laya outputs are generatable LOCALLY (riir-reflex's G5-parity
`RiirAgent` lane), so the oracle is a real trace, never published aggregates.

## Tasks

- [x] **T0a — substrate-first gate** (before any code): grep + read
  `decision_wire`, `compression_drafter` (Lz4FlexDrafter),
  `variable_rank_domain_expert` (`pick_domain`), `rating` (Elo/Beta-LCB),
  `CorpusDistanceGate`, and `katgpt-attn-match`'s `beta_fitter.rs` /
  `value_fitter.rs` (closed-form-LS-warm-started NNLS fitting — shipped
  precedent). Consume; never re-implement. Any T1/T2/T3 piece these cover
  is a forward, not new code.
  **Record (2026-09-22):** all six read; findings + consume/build
  decisions in §Substrate check below. Headline: T1 = NEW katgpt-core
  module `state_option_scoring` consuming `exact_sigmoid` +
  `float_order::cmp_for_max` + the unit-normalize idiom + `pick_domain`'s
  const-generic shape; reflex's `engine.rs` route_terms
  (`sigmoid(dot(state, centroid)·ROUTE_SCALE)`) is the shape being
  upstreamed; the fitters are the T3 precedent (not consumed at T1); the
  drafter is OUT of the hot loop per R3.
- [x] **T4a — the state enumerator + laya-format renderer** (katgpt-rs
  `examples/`, the arena's first half): enumerates Tetris states (board +
  piece + the ~34 landing options) and renders the laya-protocol English
  sentence for each — closed grammar, WHERE-form questions, no numbers in
  prose (arithmetic stays in code; their own wording-sensitivity trap is
  why the grammar is pinned). Dumps (sentence, structured_state) pairs to
  disk. Both the fixture (T0b) and the arena (T4) consume this; nothing
  downstream of it can exist first.
  **Record (2026-09-22):** `examples/tetris_01_state_enum.rs` +
  `examples/common/tetris_sim.rs` (the shared `#[path]` module — the
  `tests/common/ab_timing.rs` precedent — so T4's arena imports the same
  sim). 120 states (12 authored archetypes × 7 pieces + a 36-state
  seeded Dellacherie-greedy play ladder, seed 607), 2,660 options,
  byte-identical across runs (blake3
  `aa07b3b4ea75afca60fe5c6c7b75c0c4eca27383f44b32e5985dfe6cd64f4e72`).
  7 unit tests pin rotation dedupe, option counts (O=9/I=17 exact), the
  hard-drop landing law, hole/clear mechanics, and grammar closure (no
  digits in prose). **Protocol correction from the live laya page:**
  their Tetris is ONE sentence PER SPOT (`The piece leaves one hole under
  it and makes a small bump on top`) → P(clean) per spot → code argmaxes
  — a noul-shaped question per option, not a choice question over ~34
  options; number WORDS are in-protocol ("one hole" is the counted
  conclusion handed over in words). Grammar widened v1→**v2** after the
  first oracle run: the two-clause grammar (39 distinct sentences) tied
  66/120 oracle states at identical p_clean — the argmax degenerated to
  the index tie-break; v2 adds landing-side + resulting-height clauses
  (245 distinct sentences), taking cross-sentence ties to zero (the 33
  residual exact ties are all SAME-sentence — honest equivalences on
  symmetric boards; see the fixture README). Pre-clear law: every
  heights-based archetype carries a 0-height shaft column (uniform floors
  are pre-full rows — real boards never carry complete rows).
- [x] **T0b — the G1-oracle fixture** (generated in riir-reflex, committed
  INTO katgpt-rs with its provenance sha — the `katgpt-device-verify` rule:
  copying code across the seam creates drift, copying the fixture detects
  it): run riir-reflex's LOCAL laya forward (`laya::riir::RiirAgent`, the
  G5-parity lane: top-1 1.000, drift ≤ 3.1e-6, no skip path) over the T4a
  dump → commit (sentence, structured_state, decision) **TRIPLES**.
  R1: the old "if only aggregates are available, G1 DEGRADES to outcome
  parity" clause is DELETED — per-decision outputs are generatable locally
  (the laya forward is in-tree, not a published API), so that constraint
  does not exist. The real degradation trigger is OPERATIONAL — laya
  weights absent on the generating box — and then fixture generation
  BLOCKS; it never quietly relabels the gate.
  **Record (2026-09-22):** generator committed to riir-reflex FIRST
  (`examples/laya_oracle_batch` @ `e4bf657`, pushed — a GENERIC batch
  oracle over a JSONL manifest, game-free: the game vocabulary lives in
  the input data; T5's Flappy/lanes fixtures reuse it). Run: english
  checkpoint, CPU posture, 2,660 noul forwards (396 s, ~124 ms each),
  decisions + p_clean per option committed as the SELF-JOINED fixture
  `tests/fixtures/tetris_oracle_laya_en_v2.jsonl` (+ provenance README;
  dump blake3 `aa07b3b4…`, oracle blake3 `2261dbf3…`). Byte-identical
  re-run from the committed generator verified. Oracle sanity (recorded,
  not gated): p_clean spans 0.027–0.850; Spearman(p_clean, Dellacherie)
  mean +0.365, positive 94/120 — laya reads the sentences and prefers
  clean placements.
- [x] **T1 — the option-scoring primitive, minimal + generic** (katgpt-core,
  opt-in feature `state_option_scoring` — named for the CAPABILITY, never
  "game"): v1 signature takes `(state vector, option feature matrix)` with
  the option count bounded/const-generic and **nothing Tetris-specific in
  the signature** (R4 — a public upstream surface may not be shaped by one
  arena); surface scoped to what T4 exercises, widened only when T5 lands.
  **Design decision (R3): the hot path is centroid-cosine scoring ONLY —
  the compression drafter is OUT of the per-decision loop** (drafter deltas,
  if used at all, are a precomputed axis); decided HERE, never a number
  relaxed at T6.
  **Record (2026-09-22):** LANDED — `crates/katgpt-core/src/
  state_option_scoring.rs`, feature `state_option_scoring =
  ["distance_abstain"]` (the implication is the DRY form: T1 consumes
  distance_abstain's `unit` normalize — now `pub(crate)` — instead of
  forking a bit-parity-critical helper). Surface: `CentroidTable<D, K>`
  (`new`/`len`/`is_empty`/`row`/`rows`/`score_into`/`pick`) — const-generic
  K (`pick_domain`'s shape), `exact_sigmoid` (Issue 870), `cmp_for_max`
  argmax with the pinned lowest-index tie-break, zero-alloc, deterministic
  folds. GOAT (Bench 876): G1a planted 200/200 · G1b distinct 34 · G2 p99
  1.1–4.1 µs per decision SET with option counts printed (≤1 ms bar) · G4
  0 allocs (separate alloc-check binary) · determinism bit-identical (table
  blake3 `f2e231b2…`, decisions `8f5d9f91…`). Test-gate rows:
  `katgpt-core:2079:state_option_scoring` + the alloc-check PERF_ROW.
  **GOAT gate:** G1 = decision agreement vs the T0b triples AND vs a
  **constant-pick baseline** AND vs chance — never vs laya alone (R2:
  reflex T7 measured 4-of-5 families constant-picking — a short option
  encoding never moves a long shared context's compressed length, and
  class-balanced fixtures made "exactly chance" indistinguishable from
  "constant pick"; Tetris is the same shape at ~34 options); plus the
  **discrimination floor** = distinct picks ≥ 2 over distinct state vectors
  (reflex's shipped floor); G2 = **p99 ≤ 1 ms per decision SET with the
  OPTION COUNT printed beside it** (the sibling's shipped bar form —
  measured 59–79 µs at ~20 options; a latency bar without its option
  count is the box-state defect class); G3 = count-pinned no-regression;
  G4 = alloc-free with canary; **determinism row** = same corpus →
  bit-identical scoring state — at T1 that is the corpus CENTROID TABLE
  (the fitted head is T3 and deferred; this row must not read as asserting
  something T1 does not build) — two runs, two boxes.
- [x] **T4 — the Tetris arena** (`examples/` + `.benchmarks/`, bomber
  precedent): code computes the features (holes, bumpiness, stack height,
  line clears — Dellacherie-class, solved engineering); T1 scores every
  landing spot; the piece goes to argmax. **First GOAT reading** = replay
  the T0b fixture + bench latency + lines-cleared against their published
  numbers. The reading gates {T2, T3}.
  **Record (2026-09-22):** LANDED — `examples/tetris_02_option_arena.rs` +
  `examples/common/hash_embed.rs`. Drift check: all 120 states / 2,660
  options recompute byte-identically (placements, features, both sentence
  layers) BEFORE any scoring. Embedder substrate note (recorded): no
  in-tree generic TEXT embedder exists (engram hashes CanonicalId token
  ids, not text; reflex/clippy span embedders stay in their repos) —
  minimal arena-side trigram hashbag glue, std-only integer ops, and
  UNTUNED by plan (the first reading must measure the honest scorer; the
  argmax is scale-invariant for scale > 0 anyway — the reflex ROUTE_SCALE=8
  shape is carried but never moves a decision here). **First GOAT reading
  (Bench 876): G1 DOES NOT HOLD** — raw agreement 13/120 (10.8%) TIES the
  constant-pick baseline (13/120, oracle-majority index 16) and beats only
  chance (5.5%); the scorer discriminates (21 distinct picks — NOT the
  reflex constant-pick failure) but is not yet accurate; Dellacherie
  agreement 7/120 (context); lines-cleared 195.6 vs 3.4 mean (context);
  the 33/120 oracle same-sentence ties re-derived from option-level
  p_clean match the fixture README's pin exactly. **The reading gates {T2,
  T3} OPEN** — T2's losslessness arm (structured vs sentence agreement
  delta) and T3's corpus-fitted head are the evidence-directed levers.
  Live-game option widths are unbounded in practice (blocked columns), so
  the arena pads every decision to the widest table — zero rows can never
  win the argmax (non-negative embeddings → cosine ≥ 0; ties at the floor
  break to the lowest index, which is always a real row), proven by the
  two-pass digest `1921b19a…` being byte-identical before/after the change.
- [x] **T5 — Flappy + three-lanes micro-arenas**: same shape, STATE
  question form (where is it, never what to do — laya's own lesson).
  **T5 is a PRECONDITION for even considering default-on promotion of the
  T1 flag** (R4 — one arena cannot promote a flag).
  **Record (2026-09-23):** LANDED — Bench 880, **G1 HOLDS on both arenas
  with the T3 fitted head, ZERO primitive widening** (`CentroidTable<D, K>`
  at K=4 padded; `HeadFitter<D>` at D=9 — the T4-era surface transferred
  as-is, so the "widened only when T5 lands" clause closes with no
  widening needed). Flappy: 96/100 in-corpus = 96/100 LOO vs
  constant-pick 77 and chance 50 (K=2); three-lanes: 84/100 = 84/100 LOO
  vs constant-pick 41 and chance 33.3 (K=3); discrimination floor PASS
  both (2 and 3 distinct picks); G2 p99 42 ns/decision-set at both option
  counts; determinism anchors recorded (heads `4ac0a13c…`/`7d3f1d8e…`,
  decisions `cc89ff49…`/`8d2f9c75…`). Fixtures:
  `tests/fixtures/flappy_oracle_laya_en_v2.jsonl` (grammar
  `laya-flappy-v2`) + `lanes_oracle_laya_en_v1.jsonl` (grammar
  `laya-lanes-v1`), provenance in `tests/fixtures/micro_oracle_README.md`.
  **Two measured findings recorded en route:** (a) **the flappy v1
  grammar's motion clause was OUR OWN wording confound** — laya's read
  keyed on ", rising."/", falling fast." and ignored the position clause,
  so the v1 oracle went 85/100 to flap and pinned every scorer at the
  constant-pick ceiling; v2 renders the position band alone + the
  enumerator excludes both degenerate classes (v = +2 same-cell;
  same-band identical sentences), and the same scorer that tied
  constant-pick under v1 reads 96% under v2 (the traps section of the
  plan's thesis, reproduced in our own grammar and fixed structurally,
  not by tuning); (b) **on lanes, the head beats the code-arithmetic
  policy at imitating laya** (84% vs clearest-lane 74%) — laya's
  lane-safety read carries noun-dependent weights the head absorbs from
  the corpus, exactly the lane's thesis (match the model's read at
  ~10⁶× lower cost, never out-fly arithmetic). T1 cosine failed on both
  (39% / 9% — 0-for-3 across game shapes with Tetris): the fitted head is
  the lane's scorer, recorded not relaxed. Arena machinery shared in
  `examples/common/micro_dump.rs` (serde-only, keeps the 01 enumerators
  ungated) + `micro_fit.rs` (the tetris_03 recipe made reusable);
  enumerators `flappy_01_state_enum` / `lanes_01_state_enum` (dump +
  manifest + `--join` self-join mode); arenas `flappy_02_arena` /
  `lanes_02_arena`. 32 sim/arena unit tests green; bench-number collision
  with the sibling's `879_renoise_surprise_goat` (landed mid-flight on
  origin) resolved by renumbering MINE to 880 — theirs is committed and
  keeps the number, dual_allocation_gate green post-move.
- [x] **T2 — bounded template decode** (katgpt-core, feature-gated).
  **OWNER GATE A — APPROVED IN PRINCIPLE (verdict round 2), build DEFERRED
  until after T4's first GOAT reading; the approval LAPSES if that reading
  shows decode buys neither agreement nor a second consumer — re-approval
  is then required** (R5 — an approval without a reopen condition is a
  permanent half-state that gets cashed against evidence that has moved).
  Grammar→slot decoder for CLOSED sentence grammars (the laya-protocol
  state sentences). **The fairness argument that scopes it (R6):** in
  laya's own protocol the game code computes the features and renders
  English ONLY because BERT cannot read numbers — the sentence is laya's
  input requirement, NOT the task's; rendering is pure loss over state the
  code already holds. Our lane reading the structured state is the honest
  architecture, and the head-to-head belongs at the FEATURE boundary. T2
  therefore has exactly two jobs: (a) the **losslessness measurement arm**
  — decode the T0b sentences and score sentence-arm vs structured-arm,
  reported as an AGREEMENT DELTA where a non-zero delta is a finding about
  the RENDER, not automatically a decode bug; (b) **third-party
  laya-format traffic intake** — the only durable consumer justification.
  Provenance per Proposal 014 §Fusion: **Lz4FlexDrafter lineage** (pattern;
  `quest_grammar` is riir-ai's wrapper and is never a dep). Decode-only,
  corpus-limited. Run substrate-first on T0a findings before writing.
  **Record (2026-09-23, Bench 881):** LANDED — new katgpt-core module
  `template_decode` (opt-in `template_decode = []`, independent of
  `state_option_scoring`): closed template tables (`Seg::Lit`/`Seg::Slot`,
  closed fill vocabularies), `decode` → (template, fill indices) with LOUD
  `Unknown`/`Ambiguous` refusals (a >1-derivation sentence is refused,
  never guessed), `verify_closed` walking every template's full fill
  product (render → decode identity over the WHOLE closed space — the
  bounded claim is checked, not assumed), zero-alloc backtracking decode,
  `u8` fills / ≤ `MAX_SLOTS`=8 slots / ≤ 256 vocabs asserted at build.
  Substrate-first re-run: no grammar→slot decoder exists in-tree (the
  `decode` hits are latent decoders, byte codecs, tokenizers);
  `Lz4FlexDrafter` is the recorded pattern lineage. Tables + fill→feature
  mappers in `examples/common/grammar_tables.rs` (vocabulary order =
  feature ordinal, contract; forward mappers mirror the renderers' matches,
  corpus round-trip is the drift detector); measurement in
  `examples/decode_01_losslessness.rs` (required-features both flags).
  **The decode layer is asserted before any scoring**: 5/5 closed-space
  proofs PASS; tetris 2660/2660 + 120/120 state sentences, flappy
  200/200 + 100/100 (v/h recover EXACTLY), lanes 300/300 — all re-renders
  byte-identical, all fills == the semantic forward. **The agreement
  delta splits three ways:** lanes Δ0 (lossless anchor: 300/300 decoded
  rows bit-identical to structured rows → identical head digest, 0/100
  flips) · tetris Δ+8/+9 (decoded 44/120 = 44/120 LOO vs structured
  36/35 — laya's read is a function of the SENTENCE, so the decoded head
  tracks it better than numerics lacking the side/position band; G1
  HOLDS on the decoded arm, distinct picks 23) · flappy Δ−19 (decoded
  77/100 ties constant-pick 77 with ONE distinct pick — discrimination
  FAIL; the v2 band-only render dropped exact post_rel/post_v the
  structured head reads: the RENDER is the bottleneck, recorded as a
  render-side work item, never tuned here). The width-genericized fit
  recipe (`micro_fit`: `Standardizer<F>`, `HeadCorpus<D>`, method-level
  `design<const D>`) reproduces all three published structured anchors
  bit-identically (tetris head `65409c14…` FULL match; flappy
  `4ac0a13c…`/lanes `7d3f1d8e…` prefixes; `flappy_02_arena`,
  `lanes_02_arena`, `tetris_03_head_fit` re-run byte-identical — G3 held
  by digest). Test-gate row `katgpt-core:2074:template_decode`; catalog
  §125 (renumbered at the rebase — origin took §123/§124 for the Issue-875
  pair); README/examples flag counts re-pinned 640→641.
- [x] **T3 — the corpus-fitted head, determinism-constrained** (NOT an
  owner gate — closed-form/NNLS corpus fitting is admitted precedent in
  THIS repo: `katgpt-attn-match/src/beta_fitter.rs` warm-starts projected
  gradient from a clamped closed-form LS solution, in the public
  modelless repo): closed-form/NNLS-class fit over frozen features as the
  "80–90% corpus-viable" lever the owner hypothesized — **built only after
  T4's first GOAT reading shows plain corpus scoring falling short of
  agreement** (evidence-gated, same reading that gates T2). The line to
  hold is **determinism, not closed-form-ness**: fixed iteration count, no
  RNG, no gradient descent on base weights, head reconstructible from the
  corpus alone (bit-identical two boxes — asserted in the GOAT, a G1 row).
  Calibration point: jimothy's TF-IDF 82% on banking77 — a TRAINED
  baseline, cited as the bar to beat modellessly, never as a method to
  adopt.
  **Record (2026-09-23):** LANDED — `state_option_scoring::head` (same
  feature, no new flag): `FittedHead<D>` (zero-alloc score/pick; weights
  are the determinism-committed artifact) + `HeadFitter<D>` (the D×D
  scratch owned once and reused across refits — stable Rust rejects
  `[0.0; D * D]` under a plain const-generic; the cold fit path owns the
  heap scratch, OUT of the G4 window, documented). The fit CONSUMES
  `linalg::ridge_solve`'s f64 path (KARC Plan 308's fit math — T0a's
  consume-not-fork rule; `state_option_scoring` joined linalg's gate list
  at birth). No RNG / no iterations / no gradient descent — determinism
  held by construction (scalar f64 `mul_add` + IEEE `sqrt` are exactly
  rounded → two-box portable). Arena: `examples/tetris_03_head_fit.rs` +
  the shared `examples/common/tetris_fixture.rs` extraction (fixture
  schema + drift detector + play loop + Dellacherie policy — one copy
  behind the tetris_02/tetris_03 pair; tetris_02's anchors byte-identical
  post-extraction). Method: corpus-side standardization (fixed column
  order), intercept in-matrix (D=12), y = oracle p_clean; **λ selected by
  state-level LOO MSE over the pinned grid — the agreement number is
  never selected on**; the reading reports in-corpus AND
  leave-one-STATE-out. **GOAT (Bench 878): G1 HOLDS — in-corpus 36/120
  (30.0%) · LOO 35/120 (29.2%) vs constant-pick 13/120 (10.8%) and chance
  5.5%;** the 0.8 pp in-corpus/LOO gap says the fit generalizes at n=120
  (ridge + 11 features keep it honest) — the corpus-viable claim and the
  generalization claim are nearly the same number, which is what makes
  the HOLD honest rather than circular. G1a planted-fit 200/200 (near-tie
  pairs redrawn deterministically); G2 p99 42/84/167 ns at K=9/17/34
  (D=12, ≤1 ms bar); G4 0 allocs (separate
  `state_option_head_alloc_check` binary; HeadFitter's one-time scratch
  construction deliberately outside the window). Lines-cleared 60.75 mean
  vs T1-sentence's 3.4 (Dellacherie 195.6 context — the head imitates
  LAYA's p_clean, and laya agrees with Dellacherie on only 7/120).
  Anchors: head `65409c14…`, decisions `04644b0c…`; test-gate rows
  `katgpt-core:2085:state_option_scoring` (re-measured, was 2079) +
  `katgpt-core:1:state_option_head_alloc_check:state_option_scoring`.
- [x] **T6 — bench doc + GOAT verdict**; the numbers become citable by
  the reflex arena book (numbers only; the reflex repo itself untouched).
  **Record (2026-09-23):** the four bench records carry the consolidated
  GOAT verdict — [876](../.benchmarks/876_state_option_scoring_goat.md)
  (T1 primitive: G1a 200/200, G1b distinct 34, G2 p99 per decision SET
  1.1–4.1 µs at K=9/17/34 D=64, G4 0 allocs, determinism bit-identical;
  arena G1 then open) · [878](../.benchmarks/878_state_option_head_goat.md)
  (T3 head: G1 HOLDS 36/35 vs constant 13, G2 42–167 ns per decision SET
  D=12, G4 0 allocs) · [880](../.benchmarks/880_micro_arena_goat.md)
  (T5: G1 HOLDS both arenas 96/84, zero primitive widening, G2 42 ns per
  decision SET D=9) · [881](../.benchmarks/881_template_decode_losslessness.md)
  (T2: decode layer exact, losslessness delta −19/0/+8; G1 HOLDS on the
  decoded arm for tetris, FAILS for flappy — the render's fault, Issue
  876). **Matched-units restatement (the T6 precondition):** every G2 row
  across all four benches is stated on ONE basis — p99 per decision SET
  with the option count (K) and design width (D) printed beside it
  (1.1–4.1 µs D=64 cosine · 42–167 ns D=12 head · 42 ns D=9 micro heads) —
  all µs/ns-scale against the 1 ms bar; the decode path adds ZERO
  decision-latency cost (decode is intake/measurement, never in the
  decision loop). The lane's GOAT verdict: **HOLDS — the corpus-fitted
  head over frozen features is the scorer (T3/T5), cosine is the T1
  baseline, decode is the traffic/intake arm; every flag stays OPT-IN
  (default-off) — no promotion claimed.** The numbers are citable as-is
  (self-contained G1 baselines + per-SET units + anchors in each doc).
- [x] **T7 — doc-sync**: AGENTS.md feature-table rows for the new flags;
  arena book citation.
  **Record (2026-09-23):** katgpt-rs AGENTS.md carries no per-feature
  table (boundary context only, by design) — the feature-table rows live
  where the repo's gates pin them, and all landed at T1/T2/T5 commit time:
  README + examples/README flag-count claims re-pinned per landing
  (count_features green; 640→641 at T2) · catalog §122
  (`state_option_scoring`, incl. the T3 addendum) + §125
  (`template_decode`; renumbered from §123 at the rebase — origin took
  §123/§124 for the Issue-875 pair) · test-gate rows
  `katgpt-core:2085:state_option_scoring`,
  `katgpt-core:2074:template_decode`, the two alloc-check PERF_ROWS ·
  docs gate 35/35 green at close. Arena book citation: the numbers are
  published in the four bench docs in citable form (per-SET units,
  baselines, anchors); riir-reflex untouched per the T6 law — if the
  reflex arena book cites them, it points at Bench 876/878/880/881.
  History: the plan-close record lives in this file + the bench docs;
  HISTORY.md entry not required (no boundary/gate narrative beyond what
  the benches record).
- [-] **PUCT/MCTS research note — DEFERRED** (verdict round 1): Dellacherie-class
  heuristics settle Tetris; PUCT over a modelless policy prior is
  modelless-legal (search is not learning) but low value-per-effort. File
  a `.research/` note only if T4's oracle replay shows the argmax scorer
  losing decisions to laya on multi-step lookahead.
- [x] **T8 — the LIVE arena on reflex.gist.rs (owner addendum, 2026-09-23,
  "still see no tetris running side by side at reflex.gist.rs like
  original, what block?")**: the plan's arenas were OFFLINE benches
  (numbers only — the T6 law) and the site had no games; the owner asked
  for the original laya page's live-Tetris experience. Two-sided landing:
  **riir-reflex** (`ea197d8` + `536d65f`) — the laya comparison lane over
  HTTP behind `RIIR_REFLEX_LAYA=1`: `X-Reflex-Lane: laya` on `/decide`,
  served by the SAME G5-parity RiirAgent through a one-thread actor (the
  backend is !Send; forwards serialize), fail-closed in every non-ready
  state (never a silent modelless fallback — the per-lane-claims law),
  `/healthz` lane map JSON, PNA preflight consented for allow-listed
  origins, mapping unit-tested weights-free (12 lane tests; clippy clean
  at default/all-features/no-default; the pre-existing
  `harness_families_gates` G2 red at HEAD is box-load, reproduced on a
  clean tree at load 18-34).
  **reflex-site** (`60bd192` + `4eba4aa` + `c7aa731` + `20388c9`, deployed
  live) — `/arena/`: Tetris + Flappy + three-lanes, TWO boards side by
  side (laya | modelless) from the same seeded stream; the games are the
  exact JS ports of the T4a/T5 sims + pinned grammars, golden-checked
  node-side against the committed fixtures (tetris 2660/2660 options +
  120/120 state sentences; flappy/lanes 100/100 each — byte-identical),
  plus a bit-exact fastrand port (`rng.js`). Verification: a fixture
  replay through the LIVE HTTP lane agrees with the committed oracle
  p_cleans spot-level and argmax-level (`arena_protocol_check.mjs`); the
  production page smoke passes in headless Chromium (chips ready,
  modelless decision p50 3-6 ms, laya board reading spots). Boundary
  note: the engine stays game-free — the game lives in the site JS, the
  engine answers English noul sentences (the T0b `laya_oracle_batch`
  precedent: game vocabulary in the INPUT data).

## T9 addendum — the owner ask round 2: "still not see tetris on reflex.gist.rs" (2026-09-23)

- **The block, measured:** the arena WAS live at `/arena/` (200, both
  boards, engine detection + PNA hint) — but (a) the homepage carried it
  only as a small nav item on a long page, and (b) without a locally
  running engine the boards idled on "checking for a local engine…".
  The user opened the root URL twice and saw no games either time. Not a
  build gap this round — a discoverability + cold-start gap.
- **Fix 1 — recorded demo (reflex-site `b098ade`, deployed live, version
  `433a1293`):** with no engine, `/arena/` now auto-plays a labelled
  recorded demo. Data: `arena/demo_oracle.json` (102 KB) generated from
  the SAME committed fixtures the golden tests bind. Tetris replays the
  fixture's recorded play walk — chain-verified at generation time
  (every recorded placement reproduces the next recorded board under the
  site's own enumeration; 36 turns; mixed capture policy disclosed in
  the banner — the walk was state collection, not argmax play, 16/35
  picks are argmax). Flappy/lanes ship as recorded decision reels (the
  fixture states come from `enumerateStates`, a DIFFERENT rng stream
  than the live boards' `play_game` shape — a self-driving demo would
  miss the oracle, so the reels replay the recorded states directly).
  The modelless board plays its real out-of-the-box behavior: abstain →
  labelled random fallback. Board heads relabelled "laya lane ·
  model-based" / "modelless lane · reflex core" (the owner's own
  vocabulary); homepage gained an arena teaser section (`d210399`) with
  the live/recorded split spelled out.
- **Verification:** `scripts/arena_demo_check.mjs` (node, PASS — walk
  chain + 100+100 reel sentence/option/argmax parity vs fixtures),
  `scripts/arena_demo_smoke.mjs` (headless Chromium with
  `127.0.0.1:7331` ROUTE-BLOCKED — deterministic even on a box where a
  sibling's engine is up; PASS local + re-proven against the live prod
  URL post-deploy), golden tests 8/8 still green, prod curl markers on
  all four surfaces.
- **Honesty law kept:** the demo is labelled everywhere (banner, status
  text, SOURCE readout "· demo", "recorded play" answers, footnote
  rewritten to scope "nothing is scripted" to the live path). Recorded
  probabilities are 6-dp — 4 orders finer than the ±0.02 protocol-check
  tolerance; argmax parity verified 100/100 on both reels.
- **Live-path note:** the v0.1.1 release archives predate the laya lane,
  so a visitor following the run command with a stock install sees the
  laya board fail closed with the enable hint (modelless goes live
  fine); the v0.1.2 cut in flight carries the lane.

## Substrate check (substrate-first skill, T0a — 2026-09-22)

- Searched for: option scoring, centroid cosine, corpus routing, typed
  decision wire, NNLS/closed-form fitting — `decision_wire`,
  `Lz4FlexDrafter`/`compression_drafter`, `pick_domain`/
  `variable_rank_domain_expert`, `rating` (Elo/Beta-LCB),
  `CorpusDistanceGate`/`distance_abstain`, `beta_fitter`/`value_fitter`
  (katgpt-attn-match), and riir-reflex `engine.rs` route_terms.
- Found (all in-tree): the typed wire (`Question::choice/noul/score` —
  the arena + oracle wire); `pick_domain<N,A>` (argmax over unit
  centroids, deterministic tie-break — the routing math);
  `CorpusDistanceGate` (cosine + sigmoid over unit-normalized corpus rows,
  zero-alloc — the hot-path idiom); the fitters (closed-form-LS warm
  start + fixed-iter projected gradient — the T3 determinism precedent);
  reflex `engine.rs:410-436` route_terms (`sigmoid(dot(state, centroid_i)
  · ROUTE_SCALE)` + drafter-delta blend — the exact per-option scoring T1
  upstreams, born reflex-side in Issue 004 T7).
- Decision: **T1 BUILDS NEW** — a katgpt-core module behind the opt-in
  `state_option_scoring` feature (capability-named, no "game"),
  CONSUMING `exact_sigmoid` (the Issue-870 exact form),
  `float_order::cmp_for_max` (the tie-break), the unit-normalize-at-build
  idiom, and `pick_domain`'s const-generic shape; the compression
  drafter stays OUT of the per-decision loop (R3 — reflex measured
  drafter deltas cannot rank short options). No existing primitive
  covers per-option sigmoid scoring over a corpus centroid table.
- **T4 embedder note (recorded at landing):** no in-tree generic TEXT
  embedding primitive exists to consume — katgpt-core's engram trigram
  hashing operates on `CanonicalId` token ids, not text, and the
  span-embedding hashbags in riir-reflex / riir-clippy are those repos'
  own code (the lane split keeps them there). The arena ships minimal
  arena-side glue (`examples/common/hash_embed.rs`: byte-trigram
  hashbag, splitmix64-finaled FNV mix, std-only integer ops) — NOT
  promoted to katgpt-core, NOT shared, and NOT tuned against the oracle
  (anti-goal recorded at Bench 876: the first reading must measure the
  untuned scorer).
- Architectural rules checked: semantic domain (latent dot + sigmoid) ✓;
  no sync boundary crossed (offline arena) ✓; sigmoid-never-softmax on
  OUR primitives ✓ (the laya port keeps the reference's softmax by its
  own parity law — that is laya's semantics, not ours); naming law ✓
  (no "game" in katgpt-core flags/types; game vocabulary lives in
  `examples/` where the charter puts it).

## Boundary + provenance notes

- **Naming:** no "game" in any katgpt-core flag/type name — name the
  capability; "game" lives in the arenas (`examples/`), where this repo's
  charter already puts it (BOUNDARY.md `bevy_ecs` row).
- **Citation repair riding this plan's commit:** the "Proposal 017"
  counter-case cited by `.proposals/014` line 25 and
  `../riir-reflex/BOUNDARY.md` line 44 **never existed** (`.proposals`
  highwater = 014; no 017 on disk; none in history). Canonical record:
  **riir-ai Plan 484** (`../riir-ai/.plans/484_riir_games_domain_split_corrected.md`).
  Two commits by necessity (cross-repo repair rule — a record here claims
  nothing until the sibling commit exists): riir-reflex repointed at
  `3cbdca63df4141eaa214311afd37f6ce3c90e7ee` (committed FIRST); the
  katgpt-rs half (`.proposals/014` line 25) rides this plan's commit.
- **Numbering:** every mention qualified "katgpt-rs Plan 607" (riir-ai
  `.plans/.highwater` = 607 too). Allocation gate clean at write time
  (0 twin / 0 independent / 0 counter).
- **jimothy** (`github.com/AndrewPrifer/jimothy`) = adjacent, NOT our
  plan: per-task TRAINED classifier bundles (MiniLM/TF-IDF + teacher
  distill from Jev) with the SAME typed wire as our `decision_wire`
  (choice/boolean/noul/score). Validates the market direction; its
  per-task-bundle model is the opposite of our shared-engine stance.
  **The one steal is filed**: its threshold-recommendation pattern
  (train-time cutoff as model metadata, null on thin data, runtime
  returns everything) → riir-reflex Issue 009 (`b41bc7b`).
- **laya rust-vs-python compare: already done** (Plan 603/606): G5 parity
  green both lanes (top-1 1.000, drift ≤ 3.1e-6); Bench 001 interleaved
  latency — torch MPS 30.3 ms ≈ candle Metal 31.1 ms (torch ~1.3–1.5×
  faster, kernel maturity), riir CPU 188.7 ms at candle-CPU parity; our
  wins are deployment (one SHA-pinned binary, no Python, no torch,
  offline).

## T10 addendum — the serving consumer (riir-reflex v0.2.2, 2026-09-23)

The decoded heads now SERVE: riir-reflex v0.2.2 (`367766c` + `932a2a3`,
Plan 001 in that repo's `.plans/001_game_head_serving.md`) boots the
decoded Tetris head (this plan's Bench 881 anchors — λ=1, 44/120 in+LOO —
reproduced bit-identically by that repo's tests, serving digest
`00aa6221…c6e`) from a verbatim BLAKE3-pinned copy of this repo's
`tests/fixtures/tetris_oracle_laya_en_v2.jsonl`, and answers the pinned
spot question on `/decide` before its cosine engine's abstain. The arena's
modelless Tetris board plays out of the box (reflex.gist.rs/arena, site
`b09bffb`). Lanes + flappy remain honest abstains in that repo — the
cross-lane feature dependency (this plan's lanes grammar reads the other
lanes) and the site's frozen v2 flappy render are the two recorded
unblock paths (riir-reflex `.issues/011`). The fixture copies are
digest-pinned both sides; this repo stays the source of truth for the
oracle data and the fit recipe.

## T11 addendum — owner round-3 Q&A on the arena (2026-09-23, site `91af68d` + sibling `b09bffb`)

The owner watched the recorded demo (laya 580/13 lines vs modelless 40/1)
and asked five questions. Verdicts, all measured:

1. **Why does modelless score lower?** Not the text input — both lanes read
   the same sentences. In the recorded demo the modelless board's
   probabilities are null by construction (`scoreOptions` returns recorded
   ps to the laya lane only), so every turn is the honest abstain →
   labelled random fallback; a full-random tetris game scores ~40. The gap
   is exactly the missing game corpus, and the fitted head closes it —
   **v0.2.2 now serves it** (T10 above): the modelless board plays at the
   44/120-class instead of random (10.8%-class).
2. **Is laya python or our rust?** Our Rust port, always, on this site —
   G5-parity-verified against the reference checkpoints (top-1 1.000,
   drift ≤ 3.1e-6); the python/candle lane was deleted 2026-09-22 by owner
   directive ("no candle at all cost", "No Python anywhere"). The site
   footnote now states the provenance (site `91af68d`).
3. **Three recorded lanes incl. "laya python"?** Python board REFUSED: a
   G5-parity-verified port replays identical decisions (drift ≤ 3e-6) at
   1.3–1.5× slower — an information-free duplicate that would also violate
   the no-python directive. The meaningful three-tier arena is
   **laya-rust (teacher) / latent-first fitted head (µs, plays) / raw
   modelless (honest abstain baseline)** — the baseline needs an engine
   lane-override knob (`X-Reflex-Lane: raw`) before it can be shown live
   beside the head, since the serve path tries the head first.
4. **Why does a recording have a seed input?** The seed's only demo effect
   is shuffling the modelless board's random abstain fallback (the recorded
   walk itself is fixed bytes); live, it seeds the shared piece stream (same
   seed = same game on both lanes). Fixed site-side: the control relabels
   to "fallback seed" in demo mode with a title explaining exactly that
   (site `91af68d`), so a recording never looks seed-driven.
5. **Can it run in the browser?** Matrix: the recorded demo already runs
   fully in-browser (that is the no-engine mode); the live laya lane stays
   native-only (~650 MB weights + Metal/gemm — never a browser target);
   and the fitted head — grammar tables + decoded features + a linear
   score, zero allocations — is exactly the artifact that CAN run
   client-side, so the latent-first lane can ultimately play live in the
   tab with no engine at all. That is the owner's "latent first" endgame,
   and it is the same distillation shape as bonsai/gemma freeze/thaw:
   deterministic construction over recorded teacher outputs, never
   gradient descent on the teacher.

Roadmap tasks (unchecked):

- [ ] 3-board arena layout (latent-first head vs laya vs explicit raw
  baseline) — the engine lane-override knob (`X-Reflex-Lane: raw`) is
  LANDED 2026-09-24 (riir-reflex `4c657f8`, T17 below; issue 014 closed:
  fail-closed, `/healthz` advertises `raw`, per-lane-claims law,
  paired-smoke both directions). UNGATED site-side now: the site's lane
  discovery reads `/healthz`, so the raw board lights up for engines ≥
  `4c657f8` and shows an honest unavailable state on older engines.
- [x] Fresh full-argmax laya recording + a recorded fitted-head game — DONE
  2026-09-23 (site `6269565`, T12 record below): `record_demo_walks.mjs`
  plays tetris against a live v0.2.2 engine and records both walks with
  per-decision ms; the mixed-play capture is retired.
- [x] Browser-live head: compile the fitted head to wasm so the
  latent-first lane plays in-tab with zero engine — DONE 2026-09-23 (T13
  addendum below).
- [x] Flappy v3 render + lanes cross-lane context — the two recorded
  unblock paths (riir-reflex `.issues/011`) so all three games play on the
  latent-first lane — flappy half DONE 2026-09-24 (T14 addendum below: the
  site renders v3 + the flappy head plays live in-tab); LANES half DONE
  2026-09-24 (T15 addendum below: the lanes head plays live in-tab through
  the joined-state protocol, the published 84/100 anchor reproduced, zero
  engine). The ENGINE-side serving of both remains that repo's recorded
  TODO.

## T12 addendum — the demo boards now play recorded GAMES (2026-09-23, site `6269565`)

The owner's round-4: the recorded demo still showed the modelless board at
its abstain walk (score 40, 36 abstains, random-class mess, no timing) and
"both lanes look the same speed". Fixed at the root — the demo now replays
genuine engine-played games recorded through the LIVE `/decide` wire:

- `scripts/record_demo_walks.mjs` (site): plays tetris with the site's own
  enumeration against a running v0.2.2+ engine (same seed 607 for both
  games), sequential per-option decisions, argmax picks, chain-verified
  before writing. `tetris_head_walk` = the modelless fitted-head game (ps +
  picks + per-decision ms); `tetris_walk` = the laya game at TRUE argmax —
  the old mixed-play state-capture is retired.
- Recorded on this box (engine `00aa6221…c6e` serving digest, CPU laya):
  head **140 pts / 3 lines / 46 pieces @ p50 0.4 ms**; laya **660 pts / 11
  lines / 70 pieces @ p50 394.6 ms** — the ~1000× latency gap and the
  play-quality gap are both now ON the page instead of described in prose.
- arena.js: per-board walk selection (head walk present → modelless shows
  real ps + recorded ms; absent → old honest abstain), recorded-p50 timing
  readout in demo, auto-start demo LOOPS (a frozen dead board is not a
  demo), flappy/lanes modelless keep the abstain via an explicit
  `allowDemoPs` opt-in — the first cut dropped the lane gate and the smoke
  caught flappy/lanes modelless replaying laya's ps; the opt-in shape is
  the fix.
- `arena_demo_check.mjs`: verifies BOTH walks (chain/arity/ms arity) + a
  head-quality floor (head walk must clear ≥ 2 lines — random class ~1 must
  never come back) + head ms presence. `arena_demo_smoke.mjs`: asserts the
  head answer + `recorded · p50 N ms` timing line + `tetris_head_walk`
  presence in the served oracle.
- Validated: demo check PASS (70 + 46 turns chain-verified), demo smoke
  PASS, deployed (CF version `a576f7c7`), prod oracle verified (46/70
  turns, 0.4/394.6 ms).

## T13 addendum — the browser-live head: the latent-first lane plays in-tab, zero engine (2026-09-23, site `87ad03a`)

T11 Q5's endgame landed: the fitted Tetris head now ships compiled to
WebAssembly on the arena page, and the modelless board PLAYS LIVE in the
tab with no engine at all. Not a snapshot of weights — the page boots the
PUBLISHED RECIPE (standardize → ridge at the LOO-selected λ → linear
score) over the same BLAKE3-pinned oracle fixture the engine fits from,
as a 35 KB corpus blob inside a 72 KB artifact (`wasm-head/` in the site
repo, zero deps, no wasm-bindgen).

- **Bit-parity is proven, never assumed.** Every op is a correctly-rounded
  IEEE-754 f64 primitive with a pinned accumulation order (`fmadd` on
  arm64; libm `fma`/`sqrt` on wasm32 — no wasm fma opcode exists). Gates:
  (1) build time — `wasm-head/tests/recipe.rs`: blob regenerates
  byte-identically, the FULL recipe incl. LOO hits λ=1 + 44/120 + 44/120,
  all 2660 corpus sentences decode+re-render byte-identically, the u8
  standardizer is bit-equal to the f64 one; (2) boot — the wasm re-validates
  the blob, RE-RUNS the fit in-tab, and refuses (`head_init != 0`) unless
  the 44/120 anchor reproduces; (3) behavior — before any play, the page
  probes ALL 836 recorded (sentence → f32 P(clean)) pairs of the demo's
  head game and requires `Math.fround` bit-exactness on every one; any
  failure keeps the recorded demo (the head never half-plays).
- **Measured**: boot ~2 ms, ~1.2 µs/decision through the full JS ABI (node
  probe: `scripts/arena_head_parity.mjs` PASS 836/836); headless-Chromium
  demo smoke shows the live board at ~22 µs/spot including enumeration
  overhead — the µs story is now live on the page next to laya's recorded
  394.6 ms.
- The wasm lane answers ONLY the pinned spot question (grammar-gated —
  flappy/lanes sentences refuse → the honest abstain, mirroring the
  engine's fall-through; the T12 `allowDemoPs` leak class cannot recur by
  construction). Live board plays its own seeded game (the seed control is
  real again for the modelless lane); the demo loops; labels say
  `modelless · wasm head (in-tab)`.
- Engine-connected behavior is UNCHANGED (the engine is the canonical
  server of the same fit); the wasm is the zero-engine posture. flappy +
  lanes modelless keep the recorded honest-abstain reels (`.issues/011`
  unblock paths unchanged).
- Site: `assets/arena_head.js` (loader + probe + score), `arena.js` wiring,
  `arena/index.html` copy (demo banner + explain + footnote), smoke updated
  to assert the live posture (LIVE wasm or recorded fallback, SOURCE label
  check). `scripts/arena_demo_check.mjs` PASS; `arena_demo_smoke.mjs` PASS
  (LIVE wasm in headless Chromium). The `arena_smoke.mjs` live-engine smoke
  was NOT re-run (no engine up on the box this session); its engine-path
  assertions are untouched by the change (the wasm branch is demo-only).
- Remaining roadmap: the 3-board layout (gated on the engine `X-Reflex-Lane:
  raw` knob) + lanes cross-lane context + the engine-side flappy serving.

## T14 addendum — the flappy head joins the browser-live lane (2026-09-24, site `9230b6e`)

The flappy half of issue 011's site-side unblock path is landed: the site's
`flappy.js` renders the **v3 grammar** (band + quantized offset + neutral
post-motion — v2's band-alone arm is the measured-degenerate 77/100
constant-pick, Bench 881), the demo reel regenerates from
`flappy_oracle_laya_en_v3.jsonl` (merge semantics: the recorded tetris walks
are owned by the recorder and preserved), and a SECOND wasm head — Bench
882's decoded arm — plays flappy live in-tab with zero engine.

- The flappy head answers the **(state, option) sentence pair** — the
  joined-state protocol issue 011 records (the head row needs the state's
  pre-rel/v/h), trivially natural in-tab where the call takes both
  sentences. The reconstruction law (structured units from the decoded
  fills, crash tails at ±(h+1), pre_rel clamped ±2) is ported verbatim and
  tested exact against the fixture's own feature arrays wherever the render
  is exact.
- Published anchors reproduced BIT-FOR-BIT in the browser artifact: λ=1,
  in-corpus 96/100, LOO 96/100, and the **full head digest**
  `c93d36dc…e3c5` (`decode_01_losslessness`'s
  `FLAPPY_V3_DECODED_HEAD_ANCHOR`) from the committed blob through the
  boot path.
- Measured caught-by-the-gate defect: the first flappy blob stored the
  raws as `as u8` — Rust's saturating cast turned every NEGATIVE structured
  unit (post_v, pre_rel, edge_margin) into 0, corrupting 846/1600 cells.
  The anchors still passed (96/100 — the head is that robust) but the
  digest gate red'd; raws travel as i8 now, and the i8-vs-f64
  standardizer-equality test pins the cast in both directions.
- Gates all green: wasm-head tests 17/17; parity — tetris 836/836
  bit-exact + flappy agreement 96/100 = the published anchor over the
  corpus reel; demo check 100/100 sentence+option parity on the v3 reel;
  golden tests 7/7 rebinding v3; headless demo smoke + a PROD no-engine
  smoke (live page, engine route-blocked): both boards live in-tab
  (tetris ~33 µs/spot, flappy live decisions beside the laya reel).
- The wasm ABI grew a ready MASK (bit0 tetris / bit1 flappy) and
  `head_score_state(state, option)` — a per-head failure degrades that
  head alone; the other keeps playing.
- Still open: the LANES half (its head reads the other lanes' sentences —
  needs the joined-state protocol end to end), the 3-board layout (gated
  on the engine `X-Reflex-Lane: raw` knob), and the ENGINE-side serving of
  the flappy head (riir-reflex's recorded TODO — engine-connected flappy
  modelless still abstains there; the wasm head is the zero-engine
  posture).

## T15 addendum — the LANES head joins the browser-live lane: the joined-state protocol end to end (2026-09-24, site `aff4bcc`, prod CF `a677747e`)

Issue 011's lanes half is CLOSED on the site side: the published lanes
head (Bench 880: λ=0.01, 84/100 in-corpus + LOO, head digest prefix
`7d3f1d8e`) now plays three-lanes live in-tab with zero engine — the
third and last wasm head. The blocker (this plan's lanes grammar reads
the OTHER lanes: feature columns 6–7 count the other lanes' obstacles, so
a single-sentence path cannot reproduce the head) dissolves in-tab: the
page holds the whole turn, so `head_score_lanes(p0,l0,p1,l1,p2,l2,lane)`
takes ALL THREE option sentences in pinned lane order plus the lane to
score — the joined-state protocol issue 011 records, implemented exactly
(end to end) in the wasm, no engine involved.

- **The decode arm is EXACTLY lossless here** (Bench 882: Δ0, 0/100
  flips, decoded rows bit-identical to structured → the same head digest),
  and the wasm port proves it per-cell: `gen::parse_lanes` asserts decoded
  == the fixture's own feature arrays on ALL 300 rows — a single drifted
  cell is a loud panic, never "close enough". The fit then reproduces λ
  0.01 (LOO-selected, travels as generated data), 84/84, and the digest
  prefix `7d3f1d8e` — the same head the published record names.
- Grammar port: `laya-lanes-v1`'s TWO templates (clear / blocked) decode
  through the counting walker with the derivation count summed ACROSS
  templates — zero or ≥2 derivations refuse (the engine's multi-template
  decode twin). A sentence naming the wrong lane for its position, a lane
  index ≥ 3, or any off-grammar sentence refuses (NaN → the honest
  abstain).
- Gates all green: wasm-head 23/23 (blob regen; recipe anchors incl. the
  digest prefix; the 300/300 lossless cells; grammar round-trips over all
  300 sentences; boot determinism; i8 standardizer; score paths incl.
  cross-grammar + wrong-lane + lane≥3 refusals); parity — tetris 836/836
  bit-exact + flappy 96/100 + **lanes 84/100 = the published anchor** over
  the corpus reel (~2.2 µs/decision incl. the 3-sentence decode); demo
  check (lanes reel 100/100 sentence+option+argmax parity); headless demo
  smoke + a PROD no-engine smoke (live page, engine route-blocked): all
  three modelless boards LIVE in-tab (tetris ~33 µs/spot, flappy live
  P(clean), lanes live lane ps beside the laya recorded reel). Golden
  tests 8/8 (renderers untouched).
- lanes ready-mask bit2 (`head_lanes_lambda`/`head_lanes_anchor` exports);
  a per-head failure degrades that head alone. Artifact 92 KB (wasm-opt
  -Oz; three incompressible corpus blobs). Boot ~2.8 ms (three heads).
- Engine-connected behavior is UNCHANGED: the arena's engine lane still
  forwards each lane sentence ALONE as the state (the measured laya
  protocol), and the engine's serving-side lanes/flappy abstains stay
  riir-reflex's recorded TODO (issue 011) — sibling-active repo, not
  touched this session. The wasm head is the zero-engine posture and the
  engine path never reads it.
- Remaining roadmap: the 3-board layout (gated on the engine `X-Reflex-Lane:
  raw` knob) + the ENGINE-side serving of the flappy/lanes heads
  (riir-reflex issue 011). Site-side, every unblock path this plan names
  is now LANDED.

## T16 addendum — the engine knob issue filed (2026-09-24, riir-reflex `35b5400`)

The one remaining gating artifact is now TRACKED: riir-reflex
`.issues/014_engine_lane_override_raw_knob.md` records the
`X-Reflex-Lane: raw` lane-override knob end to end — the why (T11's
ratified three-tier arena cannot show its raw baseline beside the
head-first serve path), the design constraints (T8's lane pattern
verbatim: explicit override, default posture byte-identical, fail-closed
with the per-lane-claims law, `/healthz` advertises `raw`, unknown lanes
still refuse), what `raw` shows today (modelless abstain on
flappy/lanes — the honest baseline — while issue 011's engine-side
serving TODO is independent) and the paired-smoke acceptance that pins
BOTH directions.

The 3-board layout itself stays unchecked: it is implemented site-side
the day `/healthz` advertises the lane. Engine edits were deliberately
NOT attempted this session — riir-reflex carries two active sibling
agents (the game-heads G1 bench refresh + the Metal attention lane)
whose working surface (`src/game_heads.rs`, `src/laya/riir/*`) is the
exact seam the knob touches; the `.issues/`-only write avoided the
conflict entirely.

## T17 addendum — the knob LANDED (2026-09-24, riir-reflex `4c657f8`)

The `X-Reflex-Lane: raw` lane-override knob is IMPLEMENTED and pushed
(riir-reflex develop `4c657f8`, issue 014 closed into that repo's
HISTORY.md): the serve edge accepts `raw` alongside `laya`/`modelless`,
skips the game-head try, and answers from the raw modelless engine —
the abstain IS the answer, never a head fallback; `/healthz` advertises
`"raw":"ready"`; unknown lanes still 400; default posture byte-identical
(no header = head-first). Per-lane claims hold by construction — the
engine's response discloses itself (lane `modelless`, the engine's own
routing reason), never the head's. Paired smoke pins BOTH directions on
the fixture spot question (head-first by default, raw abstain under the
override); flappy/lanes raw pins the abstain baseline with the
WITHOUT-header side deliberately unpinned (it moves when `.issues/011`'s
engine-side serving lands). Gating suite green: clippy -D at default
features, serve_lanes 9, game_heads_serve 6, serve_cors 7,
engine_gates 8 (its `/healthz` exact-body pin re-pinned for the
additive key). The Metal sibling's WIP (`src/laya/riir/*`,
`tests/metal_ops_smoke.rs`, `.issues/008`) was left unstaged throughout.

The 3-board layout is now UNGATED site-side work (the site's lane
discovery reads `/healthz`, so it lights up for engines ≥ `4c657f8`).
