# Bench 975 — consumer dependency noise from katgpt-rs / riir-ai (build-hygiene pass)
**Status:** LANDED 2026-10-31 — riir-ai `a9b116faf`, seal-remake `c387482f`, riir-refine `29323e2b`, seal-game-editor (inside `ac520c54b`), katgpt-core linalg gate fix (this commit); compile-time A/B MEASURED 2026-11-02 (below) — katgpt-core refine-set build 88% less wall time, bar PASS.

Skill: `.agents/skills/build-hygiene/SKILL.md`. Instruments: `cargo tree -e normal --workspace`
(unique package count), `cargo shear` 1.14.0 (unused / misplaced deps, every finding
re-read by hand before acting — `getrandom` wasm `js` unification rows are false positives
and stay).

## Verdict

The noise was not "too many crates" — it was **three wrong edges**:

1. **riir-engine → katgpt-rs ROOT** via the DEFAULT feature `latent_cce_moderator`
   (`dep:katgpt-rs` + `katgpt-rs/cce_moderator`) — the root only forwards
   `katgpt-core/cce_moderator`. Same shape: `sigmoid_graded_reject` (root forwards
   `katgpt-pruners/…`).
2. **riir-games-civ → katgpt-rs ROOT** via default `npc_clr` / `proactive_salience`
   (+ opt-in `npc_cognitive_branches`) — **zero** `katgpt_rs::` uses in civ code.
3. **riir-refine → katgpt-core with 83 default features** — refine calls ~21
   modules, needing 4 features.

Plus `riir-engine`'s `katgpt-forward` / `katgpt-speculative` normal deps that `src/`
never imports (doc comments only) → dev-dep / optional (`dep:` on the opt-in
forwards, weak `katgpt-speculative?/sr2am_configurator` on the default one).

## Measured — dependency graph (normal edges, default features)

| consumer | before | after | Δ | what left |
|---|---:|---:|---:|---|
| seal-remake (workspace) | 653 pkgs | 643 pkgs | −10 | katgpt-rs root, katgpt-attn, -backend, -band, -deprecated, -quant, -ruliology, -sparse, -tokenizer, -validator |
| riir-refine | 58 pkgs | 42 pkgs | −16 | katgpt-dec, -micro-belief, -personality, -sense + their externals; katgpt-core features 85 → 6 |
| riir-rethink | 52 pkgs | 52 pkgs | 0 | already lean (`default-features = false` via riir-instinct / riir-reflex); shear clean |
| seal-std / seal-render | — | — | 0 | no katgpt / riir-ai edge in the default graph |

Unused direct deps removed (shear + hand re-read, compile-verified):

- seal-remake `seal-core`: `bitflags`, dev `proptest`, dev `tracing-subscriber`
  (`getrandom` kept — wasm `js` unification).
- seal-remake `seal-edge-worker`: `bytemuck`, `console_log` (the code calls
  `worker::console_log!`, not the crate), dev `anyhow`.
- seal-game-editor: `editor-data` `seal-asset`, `feat-audit` `serde_json`,
  `feat-pet` `egui-snarl`, `feat-revision` `blake3` (landed inside a sibling's
  `wip` commit `ac520c54b` — it swept the uncommitted hunks).

## Not cut (and why)

- `riir-engine → riir-infer-core → katgpt-forward/speculative/transformer/kv/spectral`
  is a REAL use (the inference core). Getting it out of game builds is a crate
  split, filed as riir-ai Issue 1146.
- riir-ai shear leftovers (`riir-engine` dev `riir-data`, optional `rane`,
  `riir-cognition-sdk` optional `wasmi`) and katgpt-rs leftovers (`katgpt-kv`
  `bytemuck`/`rayon`/`serde`, `katgpt-pruners` `half`, `katgpt-moka-wasm`
  `katgpt-types`) — provider-internal, not in a consumer's default graph; see
  the follow-up list below.

## Verification

- riir-ai: `cargo clippy -p riir-engine -p riir-games-civ --lib --tests
  --features riir-engine/sigmoid_graded_reject,riir-games-civ/npc_cognitive_branches`
  clean (19m02s cold); `-p riir-engine --lib --features
  weaver_runtime,bigram_markov_gate,peira_distill,sigmoid_graded_reject` clean.
- riir-refine: `cargo clippy --all-targets` (default) clean in 57.9 s warm and
  `--all-targets --all-features` clean in 21m27s cold. `--all-targets` needed one
  test-only module (`katgpt_core::poincare`) → dev-dep row
  `features = ["poincare_navigator", "subspace_phase_gate"]` (unifies into test
  builds only, never the shipped binary).
- seal-remake `-p seal-core -p seal-edge-worker --all-targets` clean (10m03s);
  seal-game-editor `-p editor-data -p feat-audit -p feat-pet -p feat-revision
  --all-targets` clean (11m10s).

## Found on the way — a latent katgpt-core feature-closure bug

`poincare.rs` (`poincare_navigator`) calls `crate::linalg::ridge_solve`, but
`poincare_navigator` was missing from `linalg`'s `any(...)` gate. Every consumer
built it with katgpt-core's defaults, and some default feature opened `linalg`, so
the bug never showed. The first consumer to turn the defaults off and name only
its own features hit E0433. Fixed in `katgpt-core/src/lib.rs` by joining the gate
(the module's own "joined at birth" rule), and verified with
`-p katgpt-core --lib --no-default-features --features
poincare_navigator,subspace_phase_gate`. Lesson: `default-features = false` in a
consumer is also an audit of the provider's feature closure.

## Compile time — MEASURED 2026-11-02 (bar PASS)

Pre-registered protocol executed verbatim: katgpt-core `cargo build -p katgpt-core
--lib` dev, refine feature set (`entropic_tilt,rating,best_belief,plasma_path`,
riir-refine's `[dependencies]` row) vs default, interleaved A→B ×3, isolated
target dirs (`/tmp/ab_a`, `/tmp/ab_b`), deps warm, `cargo clean -p katgpt-core`
between rows, artifact-existence check per row. Bar: refine set ≥ 30 % less wall
time.

| row | A default | B refine set |
|---|---:|---:|
| 1 | 4.06 s | 0.49 s |
| 2 | 4.05 s | 0.49 s |
| 3 | 4.04 s | 0.49 s |
| **median** | **4.05 s** | **0.49 s** |

**Verdict: PASS — 0.49 / 4.05 = 12 % of default wall time (88 % less; 8.3×), far
past the ≥ 30 % bar.** Corroborated by clean sizes: A's katgpt-core artifacts
201.7 MiB / 526 files vs B's 31.1 MiB / 396 files — the 4-feature gate compiles a
fraction of the crate. The "graph-only gain" caveat is retired: the consumer
feature-set discipline buys a measured, large, stable compile-time win on the
provider's biggest crate.

**Provenance (box state is part of the claim):** amo (Mac17,15, M5 Ultra),
`rig run --exclusive` (quiet-box lane), `load averages: 1.78 6.99 6.61` at start
(the 5-min tail is the prior shared job decaying; the 1-min is the quiet box), AC
Power, swap 0.00M, `cargo 1.99.0` (the rust-toolchain.toml pin), measured SHA
`e667b253b` (= this record's pushed develop HEAD). Rows stable to ±0.01 s across
pairs — no drift, no re-read needed.

**Run note (loud-zero incident):** the first rig attempt returned "DONE exit 0"
with a ZERO-byte log — the heredoc→`bash -s` script never reached the remote
(rig-run does not forward stdin; `bash -s` on empty stdin exits 0 doing nothing;
`/tmp/ab_*` artifacts absent on the box proved nothing executed). Re-run passes
the script as an argv/base64 payload and carries an in-band `SCRIPT-START`
marker. Lesson for every future `rig run`: argv-pass the script, never heredoc
stdin; a run whose log has no in-band start marker is VOID, never a pass.

## Follow-ups

- [x] riir-ai Issue 1146 — engine inference split for game consumers, **CLOSED** (riir-ai `517745791` + `0261a0802`, HISTORY row `96d4d14a3`; katgpt-rs `222edb0ae` fixed a pre-existing `InferenceResult` E0063 it surfaced). `riir-infer-core` is optional behind `inference` (default-on). seal-remake workspace **643 → 636 pkgs** (`riir-infer-core`, `katgpt-forward`, `sentencepiece` + `sentencepiece-sys` C++/cmake, `prost`, `prost-derive`, `num-derive`). Wall-time A/B on bmo (M5 Ultra, clean dev riir-engine with the cognition set, interleaved × 3, preflight `load=0.63`): **125 → 115 units, median 12 s → 10 s**, 3/3 pairs, 1 s resolution. Next possible cut: `katgpt-pruners` → `katgpt-speculative`. **Measured 2026-10-31, NOT taken.** `cargo tree` on riir-engine with the game cognition set, riir-ai `origin/develop`: 114 crates. `katgpt-speculative`'s only parent is `katgpt-pruners`, and its dependencies (`katgpt-core`, `katgpt-types`, `rayon`, `katgpt-transformer`) are all needed elsewhere. `katgpt-transformer` stays regardless: riir-engine binds it directly, and `katgpt-spectral` reaches it via `katgpt-kv`. So the cut removes exactly ONE crate (27k lines). The price: moving `ScreeningPruner`/`NoScreeningPruner`/`ConstraintPruner` (implemented across 55 of 147 pruner files) plus the `dd_tree` builders down to katgpt-types/core, with re-exports. A cross-crate trait move for one compile unit is not worth it. Re-open only if a game build profile shows `katgpt-speculative` on the critical path.
- [x] provider-internal shear rows above (katgpt-kv, katgpt-pruners, katgpt-moka-wasm,
      riir-engine dev riir-data) — LANDED 2026-11-02: katgpt-rs `406e06362`
      (kv `bytemuck`/`rayon`/`serde` · pruners `half` · moka-wasm `katgpt-types`;
      `katgpt-nn` KEPT — used by `src/moka.rs`; kv `half` KEPT — `still_kv` uses it),
      riir-ai `303e0b256` (engine dev `riir-data`). Verified per crate: clippy
      `--all-targets` default + `--all-features` (kv, pruners),
      `cargo check --target wasm32-unknown-unknown` (moka-wasm), consumer pass
      `-p katgpt-speculative -p katgpt-attn-match -p katgpt-forward`, and on the
      riir-ai side `-p riir-engine` + `-p riir-games --all-targets`
      (`npc_mind_reading` is a riir-games default, so the katgpt-kv
      `default-features = false` consumer edge compiles through it).
      `cargo shear 1.14.0` + hand grep agree on every row.
