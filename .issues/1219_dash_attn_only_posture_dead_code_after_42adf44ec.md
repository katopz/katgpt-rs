# Issue 1219: katgpt-attn dash_attn-only posture warns dead_code on `from_ref_major` (i1210 C6's sparse_kl_bound feature regressed the diag-umbrella posture)

**Status:** OPEN — filed 2026-11-15 (amo housekeeping lane zed-amo-hk-1114b@amo); a compile-posture regression, NOT a semantic bug. Reproduced in a detached worktree at HEAD `7bbf56ed5` and bisected to `42adf44ec` (i1210 C6, 2026-11-14). Neither side of the bisection touched `from_ref_major`, `chunk_summary.rs`, or any katgpt-attn source — the regression is feature-graph-shaped, which is exactly what makes it worth a lane's attention.
**Claimed:** zed-amo-hk-1114b@amo, 1794721200, expires 2026-11-17T23:59:00+00:00

## The warning

`cargo clippy -p katgpt-attn --lib --features dash_attn` at HEAD:

```
warning: associated function `from_ref_major` is never used
   --> crates/katgpt-attn/src/dash_attn/chunk_summary.rs:116:19
```

`from_ref_major` is the `pub(crate)` sidecar constructor whose ONLY caller is
`chunk_summary_sidecar.rs:240` (behind the opt-in `zha_chunk_sidecar`
feature). Before `42adf44ec` this posture was clippy-clean — the landing's
own gate lines say "clippy -D warnings both postures" (917c06c83, 36 tests
at the feature posture + 165 at dash_attn-only).

## The bisect (worktree-verified, content-faithful)

| checkout of katgpt-core/{Cargo.toml,src,benches} | dash_attn-only clippy |
|---|---|
| `42adf44ec~1` (2ea2930f6 wip) | 0 warnings |
| `42adf44ec` (sparse_kl_bound lands) | the `from_ref_major` warning fires |
| `8c3349bfb` (the 635 flagless-default bake) | 0 warnings |
| `4a074958e` (635b umbrellas) | warning fires again |
| HEAD | warning fires |

Mechanism (verified, not guessed): `42adf44ec` added `sparse_kl_bound` to
katgpt-core's `diagnostics` umbrella (and the module behind the feature).
The `katgpt-attn` → `katgpt-core` dep edge builds katgpt-core's DEFAULT
feature set, and in katgpt-attn's own feature table the core-side default
posture changed shape between those commits (the `635a` umbrella era
resolved `default` differently than the current spellings). The dead-code
warn is the SYMPTOM — the visible `from_ref_major`-side effect of the
core-side feature resolution changing what the dash_attn-only posture
compiles. Enabling `katgpt-attn`'s dependency with
`katgpt-core/sparse_kl_bound` qualified, or any posture that pulls
`katgpt-core/full` / `diagnostics`, restores 0 warnings — the warning is
strictly a compile-POSTURE artifact, never a semantic drift (the same
function, same callers, same feature gates either way).

## Why this matters (the two standing gates it trips)

- The AGENTS.md full-gate line runs clippy `-D clippy::…` over
  `--workspace --all-targets --all-features` — `--all-features` is CLEAN
  (verified: `--all-features` posture = 0 warnings). The residue is the
  narrow `--features dash_attn` posture only, which standing lanes run
  (the 917c06c83 gate receipt itself pins "165 tests at dash_attn-only"
  + clippy -D warnings).
- The P43/required-features discipline and the `zha_chunk_sidecar`
  opt-in lane both compile this posture in CI-side gates; a `-D warnings`
  run at that posture now reds.

## Fix direction (bounded; do NOT blanket-allow dead_code)

1. Root cause first: re-spell the katgpt-attn → katgpt-core feature
   handoff so the dash_attn-only posture resolves the same core-side
   default set as before `42adf44ec` (the 635b umbrella migration left
   this seam; a one-row Cargo.toml spell may be the whole fix).
2. If (1) is genuinely out of reach, the bounded fallback is a
   file-local, comment-guarded `#[cfg_attr(not(feature =
   "zha_chunk_sidecar"), allow(dead_code))]` ON `from_ref_major` naming
   its one caller and the posture — never a crate-level allow, never a
   pin-row addition.
3. Regression fence: a gate row asserting the dash_attn-only posture is
   clippy-clean (the posture the 666-T2 landing promised), so the next
   core-side feature-graph edit cannot re-trip it silently.

## Non-goals

No semantic change to `from_ref_major` or the sidecar lane; no
`zha_chunk_sidecar` promotion talk (owner-gated separately); no
riir-train T3/T4 work here (that lane stays owner-gated spend in ITS
issue).
