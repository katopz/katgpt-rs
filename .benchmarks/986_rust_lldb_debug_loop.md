# Bench 986 — the lldb one-attach debug loop (rust-developer §8 adoption proof)

**Status:** PRE-REGISTERED + MEASURED (same session, same file) — the bars above are verbatim from before the first measurement; results below.
**Date:** 2026-11-17 · **Box:** m3 (owner laptop; `agent-mail rig status m3`: BUSY — load 28.4/16, contenders: mds_stores 170%, Zed Dev 88%/7 GB, sibling rustc + cargo in `merge-sprint040`; **all timings PROVISIONAL** — direction expected robust at the measured gaps, per the loaded-box law)
**Skill:** `.agents/skills/rust-developer` §8 (canonical home moved to this repo in the same session, from `seal-game-editor`)
**Origin:** JetBrains, "Debugging Rust in 2026" (2026-11-10) — the blog's terminal-lldb trick, adapted to the fleet's profile law.

## Question

Does the one-attach lldb lane beat the print loop for the residual bug class, and what does it cost under the fleet profile discipline (`[profile.dev] debug = "line-tables-only"`, deps `debug = false`, `[profile.dev-full]` escape hatch)?

## Pre-registered bars (written before measurement)

- **B1 mechanics (PASS/FAIL, required):** a `rust-lldb --batch` session on the seeded-bug test binary stops at the tie-comparison breakpoint and `frame variable` exposes the deciding state (candidate id, held id, distances) with ZERO source edits during diagnosis.
- **B2 break-even (≤ 3 questions):** `T_attach_lanes = (dev-full build delta) + (attach + breakpoint + one frame variable)` ≤ 3 × `T_print_question` (incremental edit→cargo test wall time, mean over the genuine diagnosis cycles). The debugger pays for itself within three inspection questions.
- **B3 dev-full tax (report, advisory):** `T_devfull_build − T_ltbl_build` on the demo crate; big-crate bounds cited from fleet records, not re-measured here.
- **Verdict rule:** land §8 in the skill iff B1 = PASS and B2 = PASS. B3 informational.

## Setup

Seeded comparator bug (own code, blog-class: a tie-break ordering law flipped), demo crate at `/tmp/lldbdemo` (source in the appendix; profile discipline mirrored: `debug = "line-tables-only"`, `dev-full` escape hatch). The failing test is the repro. The print lane is a GENUINE agent loop — real `println!` edits, real recompiles, per-cycle wall time; the lldb lane is one dev-full build + one attach. The fix + green re-run is common to both lanes and scored against neither.

## Results

| Metric | Value | How |
|---|---|---|
| `T_print_question` (incremental edit → `cargo test`) | 0.46 s / 0.67 s — mean **0.57 s** | two genuine `println!` cycles, `/usr/bin/time -p` real |
| `T_ltbl_build` (forced rebuild, line-tables) | **0.15 s** | `touch` + `cargo test --no-run` |
| `T_devfull_build` (forced rebuild) | **0.14 s** | `touch` + `cargo test --profile dev-full --no-run` |
| **B3** dev-full tax (toy crate) | **≈ 0 s** (0.14 vs 0.15 — noise) | the pair above |
| `T_attach` (rust-lldb batch: startup + breakpoint + run + `frame variable` + quit) | **0.91 s** | `/usr/bin/time -p` on the working session |
| **B1** mechanics | **PASS** | transcript below — zero source edits during diagnosis |
| **B2** break-even | **PASS** — 0.91 ≤ 1.71 (= 3 × 0.57) | attach lanes ≤ 3 print questions, already true on a 40-line crate |

**Diagnosis cost, toy crate:** print lane = 2 questions = **1.13 s** plus the edits; lldb lane = **one attach, 0.91 s**, zero edits, every further question ~free. The toy crate UNDER-PRICES the print lane by construction — a fleet-scale incremental test rebuild is seconds to minutes (fleet record: riir-infer Issue 1195 measured test compile+link 10.31 s → 6.77 s going TO line-tables-only, i.e. the dev-full tax on a real crate ≈ **+52%**; seal-remake Issue 124 landed the same cut at gate scale). At that scale the one-attach law pays for itself in ONE question.

### B1 transcript (verbatim, trimmed)

```
Process 36596 stopped
* thread #2, name = 'tests::tie_breaks_toward_lower_id', stop reason = breakpoint 1.1
       frame #0: lldbdemo::pick_nearest(seeker={id:0, x:0, y:0}, bodies=size=3) at lib.rs:20:20
-> 20                  if d < bd || (d == bd && b.id > bid) {
(lldb) frame variable
(core::option::Option<(f32, u32)>) best = Some((5, 7))
(lldbdemo::Body *) b = 0x… {id:2, x:4, y:3}
(float) d = 5
(float) bd = 5
(unsigned int) bid = 7
```

Candidate 2 ties held 7 at d = 5; the pinned law says lower wins; line 20 keeps 7 — the flipped comparator, named with zero source edits.

### Session lessons (folded into the skill text)

- `-f` takes the BASENAME (`-f lib.rs`); an absolute-path spelling stayed pending.
- The `--exact` filter needs the FULL test path (`tests::<name>`) — a bare name silently runs 0 tests (that failure also wasted the first attach attempt: 4 s). The one command in §8 carries the full-path spelling now.
- lldb warns "no plugin for the language rust"; the rustup shim's formatters still render Option/slices/POD structs.

## Box state

Session start (`agent-mail rig status m3`): **BUSY** — load 28.4/16, contenders mds_stores 170%, Zed Dev 88%/7 GB, sibling rustc+cargo in `merge-sprint040`. Measurement close (`riir-reflex/scripts/bench_preflight.sh`): **PASSED** — `PROVENANCE: power=AC Power load=4.52 swap=0.00M canary=skipped powermode=2(high)`. The battery straddled that load drop (first builds + print cycles under 28.4; the hi-res re-measures near 4.52). Direction is robust at the measured gaps; magnitudes are quoted for direction, not absolute calibration. High Power Mode was on (powermode=2) — disclosed.

## Verdict

B1 PASS ∧ B2 PASS → **§8 LANDED** in `.agents/skills/rust-developer` (this repo, same session). B3 ≈ 0 on the toy crate; fleet-scale tax cited (+52% class, riir-infer Issue 1195) — pay it once per attach, never per question. Honest limits recorded in the skill (async/macro stepping still weak per the JetBrains survey; basename `-f`; full-path `--exact`; formatter warning).

## Appendix — the demo crate

`Cargo.toml`:

```toml
[package]
name = "lldbdemo"
version = "0.1.0"
edition = "2021"

[profile.dev]
debug = "line-tables-only"

[profile.dev.package."*"]
debug = false

[profile.dev-full]
inherits = "dev"
debug = true
```

`src/lib.rs` — SEEDED BUG variant (the tie comparator is flipped: pinned law is lower-id-wins, code keeps the higher):

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub id: u32,
    pub x: f32,
    pub y: f32,
}

pub fn dist(a: &Body, b: &Body) -> f32 {
    (a.x - b.x).hypot(a.y - b.y)
}

/// Picks the body nearest `seeker`. PINNED LAW: an exact distance tie breaks toward the LOWER id.
pub fn pick_nearest(seeker: &Body, bodies: &[Body]) -> Option<u32> {
    let mut best: Option<(f32, u32)> = None;
    for b in bodies {
        let d = dist(seeker, b);
        match best {
            None => best = Some((d, b.id)),
            Some((bd, bid)) => {
                if d < bd || (d == bd && b.id > bid) {
                    best = Some((d, b.id));
                }
            }
        }
    }
    best.map(|(_, id)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tie_breaks_toward_lower_id() {
        let seeker = Body { id: 0, x: 0.0, y: 0.0 };
        let bodies = [
            Body { id: 7, x: 3.0, y: 4.0 }, // d = 5
            Body { id: 2, x: 4.0, y: 3.0 }, // d = 5 — pinned law: lower id (2) must win
            Body { id: 9, x: 0.0, y: 8.0 }, // d = 8
        ];
        assert_eq!(pick_nearest(&seeker, &bodies), Some(2));
    }
}
```

The FIXED variant differs in exactly one token: `b.id > bid` → `b.id < bid`.

Teardown: `/tmp/lldbdemo` removed at session close (cleanup law); the sources live on in this appendix.
