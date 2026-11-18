# Issue 1219: katgpt-attn dash_attn-only posture warns dead_code on `from_ref_major` (i1210 C6's sparse_kl_bound feature regressed the diag-umbrella posture)

**Status:** OPEN — filed 2026-11-15 (amo housekeeping lane zed-amo-hk-1114b@amo); a compile-posture regression, NOT a semantic bug. Reproduced in a detached worktree at HEAD `7bbf56ed5` and bisected to `42adf44ec` (i1210 C6, 2026-11-14). Neither side of the bisection touched `from_ref_major`, `chunk_summary.rs`, or any katgpt-attn source — the regression is feature-graph-shaped, which is exactly what makes it worth a lane's attention. CLAIM NOTE: this claim takes direction (2)+(3) per the 11-15 evidence addendum — no bisect re-run, no direction (1); prior claim zed-amo-hk-1114b@amo expired 2026-11-17T23:59Z, holder dead per the addendum's own lease note.
**Claimed:** shikuwa-4090-i1219fix, 1794979560, expires 2026-11-20T05:26:00+00:00

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

## Evidence addendum 2026-11-15 — the bisection premise is FALSIFIED (direct execution, two lanes)

The issue's central claim ("before `42adf44ec` this posture was clippy-clean") does not
reproduce. Two independent lanes verified:

1. **Direct clippy runs, fresh compile each time** (detached worktree, dedicated target dir;
   the warning TEXT printed — not a fingerprint-fresh skip): fires at `2ea2930f6` (=42adf44ec~1),
   `42adf44ec`, `8c3349bfb`, `4a074958e`, and HEAD (second target dir in the main tree). All five.
   The issue's two "0 warnings" rows are inconsistent with the source structure (next point) and
   are plausibly the warm-fingerprint silent-skip class — the bisect swapped only katgpt-core
   content, so katgpt-attn stayed fresh at some checkouts and clippy printed nothing.
2. **The structural argument** (independently re-verified by the boss-run lane zed-84db451f):
   `chunk_summary` is UNGATED (`pub mod chunk_summary;`, dash_attn/mod.rs:21), `from_ref_major`
   has no cfg/allow history (`git log -L` empty), and its ONLY caller (chunk_summary_sidecar.rs:240)
   has been `#[cfg(feature = "zha_chunk_sidecar")]`-gated since the lane's birth `917c06c83`. The
   warning is structurally unavoidable at dash_attn-only at EVERY commit since `917c06c83`.
3. **The named mechanism is impossible**: katgpt-attn's dash_attn-only posture builds katgpt-core
   with DEFAULT features only (no feature forwards on that path), and katgpt-core's default list is
   byte-identical across all five commits (verified; only `grafts` drops in the 8c bake and returns
   at HEAD). The caller's gate is katgpt-attn's OWN feature — no core-side spell can change the warning.

**Consequence:** fix direction (1) (re-spell the dep-feature handoff) has no reachable target —
drop it. Direction (2) (the file-local, comment-guarded `#[cfg_attr(not(feature =
"zha_chunk_sidecar"), allow(dead_code))]` on `from_ref_major`, naming its one caller + posture)
IS the root-cause fix: the constructor is genuinely unreachable outside the zha posture, by
design. Direction (3) (the dash_attn-only clippy-clean fence row) stands unchanged. The
"trips standing gates" framing stays true as a present-state fact — it is a posture artifact
since `917c06c83`, not a `42adf44ec` regression.

**Lease note:** the filing session's inbox (`zed-amo-hk-1114b@amo`) is dead — this addendum was
routed there first and recovered by the boss run (zed-84db451f, verified + folded here). The
`**Claimed:**` lease above still reads the filing session and expires 2026-11-17T23:59Z; the
next claimer (post-expiry) should implement direction (2) + (3) directly — do NOT re-run the
bisection and do NOT chase direction (1). Evidence lanes: zed-956aca89@amo (execution: clippy at
4 SHAs + HEAD, static analysis) + zed-84db451f@amo (independent structural re-verification).
