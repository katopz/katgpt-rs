# Issue 729 — the NaN-comparator class never got its katgpt-rs wave: ~160 live `partial_cmp` sites + 13 NaN-promoting `total_cmp` positions, in the repo that OWNS `float_order`

RESOLVED — 2026-09-06, T1–T4 complete (commit ref in the closing commit message; file kept as
the durable record — the riir-ai AGENTS.md Issue-832 row is sibling-contended this window, so
this file carries the katgpt-rs half of the class record).

riir-ai Issue 832's T4 sibling sweep (2026-09-02) enumerated riir-neuron-db / riir-chain /
riir-mmorpg-examples / riir-game-sdk (plus zero-hit repos) — **katgpt-rs was never in the
population**. The earlier katgpt-rs wave that introduced `total_cmp` fixed only part of the
src surface (its sites carry the "`total_cmp` is branch-free and NaN-deterministic" comment),
leaving benches/examples/tests plus a src tail on the legacy idiom.

## Census (final, multi-line-aware — the first single-line grep census undercounted)

- **Group A — legacy idiom** `partial_cmp(..).unwrap_or(Equal)` / `.unwrap()` in a
  comparator: **~150 sites / ~120 files**, spanning every crate (katgpt-core src+benches+
  tests+examples, katgpt-pruners, katgpt-forward, katgpt-speculative, katgpt-band,
  katgpt-attn-match, katgpt-sparse, katgpt-percepta, katgpt-moka-wasm, katgpt-sense,
  katgpt-ruliology, katgpt-kv, katgpt-micro-belief, katgpt-dec, katgpt-tokenizer,
  katgpt-deprecated, root src/benches/tests/examples). Defects (per the float_order module
  doc): the intransitive-with-NaN comparator makes std's sort **abort in release** at
  production sizes; `max_by` hands the selection to whichever element the tie landed on.
- **Group B — `total_cmp` in NaN-promoting positions** (13 sites): IEEE-754 totalOrder ranks
  NaN ABOVE `+inf`, so a descending sort/selection written as `b.total_cmp(&a)` promotes the
  corrupt value to rank 0 — the exact hazard `float_order::desc`/`cmp_for_max` were built
  for (`float_order.rs:20-26`; the pre-832 katgpt-rs fixes predate that doc). Sites:
  `dllm_solver::mbr_select`, `mcts_state_action_cache`, `pruners/remax::expected_max_over_m`,
  `katgpt-band/bckvss`, `katgpt-forward/step::extract_ddtree_paths`,
  `katgpt-forward/forward::select_topk_indices_into_buf`,
  `katgpt-attn-match/highest_attn` (×2 — descending top-t KEY selection),
  `katgpt-speculative/prefill::compress_prompt`, `katgpt-speculative/vocab_coreset`,
  `cce/primal_dual::project_onto_simplex` (descending simplex sort),
  `factorized_action/codebook` (farthest-code max_by), `katgpt-ruliology/bandit::best_arm`
  (manual loop — NaN beat −inf under total_cmp), root `bomber/validator_agent` (desc).
  The ruliology one is the sharpest: a NaN payoff would be SELECTED as the best arm.

## Terminals (the substrate's own table, `katgpt_core::float_order` — ungated, lib.rs:100)

| consumer | pass |
|---|---|
| sort, largest first | `desc` / `desc_f64` |
| sort, smallest first | `asc` / `asc_f64` |
| `max_by` family | `cmp_for_max` / `cmp_for_max_f64` |
| `min_by` family | `cmp_for_min` / `cmp_for_min_f64` |
| `binary_search_by` probe / three-way `match` / `Ord` impl chain | `total_cmp` (deterministic, direction-neutral) |

Guarantee: identical ordering to the replaced idiom on all NaN-free input (pinned by
float_order's own corpus tests), so no gate re-baselines.

**No-dep crates** (katgpt-core is optional or absent): `katgpt-dec` ("Zero dependencies"
manifest — `min_by` + `total_cmp` = NaN loses the min, correct), `katgpt-tokenizer` (CDF
binary search), `katgpt-micro-belief` + `katgpt-sense` (katgpt-types only — bench/test
fixtures NaN-free by construction, commented at the site), and `katgpt-attn-match`'s
`highest_attn` (katgpt-core is OPTIONAL behind `maxsim` and must stay that way — production
descending key selection ships a local documented `desc_nan_last` specialization, the
ndb-sigmoid-keep precedent).

## Tasks

- [x] T1 production `src/` sweep — Group A prod + all of Group B (~60 sites)
- [x] T2 in-crate test modules + crate `tests/` (~40 sites)
- [x] T3 root `tests/` + `benches/` + `examples/` (~50 sites)
- [x] T4 validation — `cargo check --workspace --all-targets --keep-going` clean;
      clippy `-p katgpt-core --lib` clean; scoped lib suites all green:
      katgpt-core 1974/0, root 203/0, pruners 126/0, speculative 305/0, dec 225/0,
      forward 125/0, sparse 39/0, ruliology 93/0, percepta 40/0, band 24/0,
      sense 24/0, moka-wasm 17/0, micro-belief 54/0, tokenizer 10/0,
      integration 92/0, bench_manifold 12/0, and_or 5/0, types bench_578 10/0
- [x] T5 exclusions pinned — `float_order.rs` (9 grep hits: the deliberate
      `matches_partial_cmp_idiom_*` oracles + `legacy_max_by_can_select_nan_this_is_the_bug`
      documentation test — its own comment forbids "fixing"); the 8 loud-by-design
      `.expect("no NaN …")` timing-median guards (bench_578/582/583/656/680/stale_residual —
      the Issue-832 deliberate-leaf class); already-fixed sites whose COMMENTS mention the
      legacy idiom; `variance_minimizer`/`hull`/`game_state_02` `match`-with-`None`-arm sign
      checks (NaN falls to the neutral arm — safe by construction)

## Lessons

1. **Single-line grep census undercounts multi-line comparator chains.** The first census
   (~100 sites) missed ~50 sites whose `.unwrap_or(...)` sits on the NEXT line
   (`X\n.partial_cmp(&Y)\n.unwrap_or(Equal)`). The complete census needs the
   partial_cmp + 3-line-forward unwrap window (run here as a small Python scanner).
2. **Match-ergonomics deref table** (the compile gate taught it five times): destructured
   `|(_, a), (_, b)|` over `.iter().enumerate()` binds `&&f32` (`**a`); the same pattern
   over an owned-tuple iterator binds `&f32` (`*a`); non-destructured `|a, b|` with `a.1`
   field access binds the tuple ref (`*a.1`); `|&a, &b|` over an integer range binds values.
3. **Dep presence ≠ dep reachability**: katgpt-core as an OPTIONAL dep (attn-match `maxsim`)
   or absent (sense/micro-belief/tokenizer/dec) means the substrate terminal is unavailable
   at default features — a documented local specialization or a direction-safe total_cmp is
   the honest fallback, per-crate.
4. `cargo check -p A -p B …` feature-UNIFIES across the named crates; `cargo check
   --workspace` does not. A per-crate green pass is NOT a workspace claim (E0433/E0308s
   surfaced only under the workspace pass).

Cross-repo residual for the riir-ai lane — **RESOLVED same day** by riir-ai
`.issues/878_nan_comparator_tail_sweep.md` (riir-ai `1ee35da79`): the 2 recorded sites plus
~138 more the same catch-all grep found once the surface widened to tests/examples/benches/
cfg(test) mods — including PRODUCTION civ `map_tick` comparators (guard_ai, predator_fsm,
predator, eagle, crime, movement, leo_act) and riir-engine `fourier/tuning.rs` that 832's
production-only pass had also missed (the `std::cmp::Ordering::Equal` spelling + multi-line
comparator heads evade the single-line grep). That sweep also found and fixed HERE the deref
depth stragglers this issue's own f2c305dd shipped in feature-gated surfaces (7 sites + 11
dead imports, `649ce5fe`) — the closure-shape-dependent `&&f32` double-ref (iter-closure vs
owned-tuple map) is the systematic version of this issue's "5 sites fixed post-compile-gate"
note, now probed and documented in riir-ai 878.
