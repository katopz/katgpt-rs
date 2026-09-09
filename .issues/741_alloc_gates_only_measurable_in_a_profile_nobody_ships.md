# Issue 741 — every alloc gate was gated on `debug_assertions`, so the G4/G5 zero-alloc claims could only ever be measured in a profile nobody ships — and the auditor that should have said so read only the FIRST `#![cfg]` of each file

**Status:** RESOLVED 2026-09-09 for this repo's two targets (both now RUN and
PASS in release). Three sibling instances remain, filed in their own repos.
Both halves verified two-sided: the repair was measured in all three
configurations, and the pin that guards it was proved to FIRE by planting the
defect back and confirming exit 1 → exit 0 on restore.

## The finding, in the order it was actually found

### 1. The gate could not be run in the profile it is meant to be read in

`AGENTS.md` mandates `--release` for gates ("a latency gate in a debug build
measures an unoptimised binary"). Following that rule on this repo's G4 alloc
gate produced:

```
$ cargo test --features kimi_k3_loader --test kimi_k3_g4_alloc_free
test g4_block_state_push_uses_pool_no_alloc ... ok
test result: ok. 1 passed; 0 failed; 1 ignored

$ cargo test --release --features kimi_k3_loader --test kimi_k3_g4_alloc_free
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored     ← exit 0
```

`ok. 0 passed`, exit 0, byte-for-byte a pass. The file opened
`#![cfg(all(feature = "kimi_k3_loader", debug_assertions))]`, so in release it
compiled to an **empty binary**.

The cause was one axis choice, three layers down: the whole
`katgpt_core::alloc` module — `TrackingAllocator`, the per-thread counters,
`reset_alloc_stats`, `get_alloc_stats` — carried `#[cfg(debug_assertions)]` on
all nine of its items. The symbols do not exist in a release build, so the
gate genuinely could not compile there. **`debug_assertions` is not a knob:**
it is ON in dev and OFF in release and no flag the reader can type changes
that. So the gate was not merely unrun in release — it was *unrunnable*.

The file said so, honestly, and that is the part worth keeping:

> The zero-alloc claim in release builds is verified by **code review**: there
> are no `to_vec()`, `Vec::new()`, or `Vec::with_capacity()` calls inside the
> forward hot path.

That is an accurate statement and it is not a gate. It was true when written
(2026-08-02) and nothing re-checked it for 38 days. A code-review claim does
not survive the next refactor, which is the entire reason gates exist.

### 2. The auditor was blind to half the population

`scripts/cfg_gated_target_audit.py` returned the body of **the first**
whole-file `#![cfg(...)]` — and rustc **ANDs** every inner attribute. So

```rust
#![cfg(feature = "clr")]
#![cfg(debug_assertions)]
```

read as feature-gated with **no profile term at all**. That is
`tests/bench_284_clr_goat_g4.rs`, a second G4 alloc gate with exactly the same
defect, and it was invisible: `cfg_gated_floors.txt` pinned
`max_profile_debug_only = 1` and its own header stated katgpt-rs "has exactly
ONE: tests/kimi_k3_g4_alloc_free.rs". The header was not sloppy — it was the
faithful output of a blind instrument.

Population of the blind spot, measured before the fix: **56 of 1634**
`#![cfg]`-gated test/bench targets across the workspace carry 2+ whole-file
`#![cfg]` attributes (up to **5** in one file — `social_scenarios_headless.rs`),
spread over **8 repos**.

Both directions of that error are real and they differ in severity:

| non-first attribute contains | consequence |
|---|---|
| `debug_assertions` | the PROFILE dimension under-reports — the one class that is silent in the DEFAULT invocation |
| `feature = "x"` | the required-features check under-reports, so a row covering only the first cfg reads as `w/ req-f` (**protected**) while cargo silently SKIPS the target — the "row exists and is wrong" class, strictly worse than a missing row |

Fixing it moved the workspace numbers:

| | before | after |
|---|---|---|
| SILENT-NOW | 177 | **181** |
| load-bearing SILENT-NOW | **0 in every repo** | **2** |
| DEBUG-only profile-gated | 4 | **5** |

The `load-bear 0 → 2` is the sharp one: `max_load_bearing = 0` was green over
a population the classifier could not see. That is the **third** time this
exact shape is recorded in `cfg_gated_floors.txt` (Issue 713 T4c's vocabulary
gap, Issue 728's one-repo dialect, now this), and the two newly-visible rows
are in siblings — `riir-mmorpg-examples/tests/join_sync_goat.rs` (4 attributes,
no `[[test]]` row) and
`riir-train/crates/riir-train-gpu/tests/goat_300_tjs_lora_gpu.rs`.

⚠️ **Neither is a live CI launder** — checked rather than assumed.
`join_sync_goat` appears in `riir-mmorpg-examples/scripts/ci_feature_guard.sh`
only inside a **comment**; no script invokes either target. They are the
hand-typed exposure, same as riir-game-sdk `.issues/028`'s 31.

### 3. `load-bear` does not mean what `.issues/028` said it means

While tracing those two: riir-game-sdk `.issues/028` states "`load-bear = 0`
in the audit means no *script or workflow* names them by target name."
`is_load_bearing()` splits the **filename** into tokens and matches
`goat / gate / drill / proof / invariant / guard / conservation / safety /
security / audit / g<N>` — it never opens a script. The column means "the
target's NAME advertises that its green is evidence". Corrected there in the
same push as this issue.

## The fix

`debug_assertions` couples *"can I measure allocations?"* to *"am I
optimised?"*, and those are independent questions. A profile is not a
capability; a feature is.

- [x] **`alloc_tracking`**, a new opt-in feature, declared in `katgpt-core` and
      passed through from the root (`alloc_tracking =
      ["katgpt-core/alloc_tracking"]`). The `alloc` module is gated
      `any(debug_assertions, feature = "alloc_tracking")`.
- [x] **The gate is carried ONCE**, on the `pub mod alloc;` declaration in
      `katgpt-core/src/lib.rs`. The nine per-item `#[cfg(debug_assertions)]`
      attributes inside `alloc.rs` are gone — a per-item predicate could drift
      out of agreement with the module gate and with the two
      `#[global_allocator]` registrations that depend on it. One gate, not ten.
- [x] Re-gated the two targets, `tests/common/alloc_tracking.rs` (the shared
      per-binary allocator installer), and katgpt-core's own
      `TEST_GLOBAL_ALLOC`.
- [x] **Re-gated the in-body liveness sentinel too** — this is the step that
      matters most and it is easy to miss. `bench_284` has an Issue-682
      sentinel that probes the allocator and FAILS if it is not installed,
      plus a `#[cfg(not(debug_assertions))]` no-op twin. Widening only the
      file gate would have produced a release binary that runs the gate and
      asserts **nothing** — a vacuous pass, which is strictly worse than the
      empty binary it replaced. Both arms now follow the file's predicate, so
      the release run's green also proves the allocator is really installed.

### Three configurations, measured — not argued

|  | dev | release | release + `alloc_tracking` |
|---|---|---|---|
| `kimi_k3_g4_alloc_free` | 1 passed | `ok. 0 passed` ✗ | **1 passed** ✓ |
| `bench_284_clr_goat_g4` | 1 passed | `ok. 0 passed` ✗ | **1 passed** ✓ |
| `katgpt-core --lib alloc::` | 6 passed | (module absent) | **6 passed** ✓ |

The `katgpt-core` row includes `test_thread_isolation`, which is the property
*every* alloc gate relies on for parallel-safe measurement — now proven in the
optimised build rather than assumed from the debug one.

**The claims held.** These two gates were silently **UNVERIFIED, not silently
broken** — the same finding as `.issues/714` T3 and Issue 728 T2, and exactly
why nobody noticed. The value delivered is not a bug fix; it is that a
code-review claim became a gate.

Shipped release with the feature off is unchanged: the module does not exist,
`System` is the allocator, there is no TLS read on the allocation path.
Verified by a forced fresh `cargo clippy -p katgpt-core --release --lib` — 0
warnings, exit 0 — with and without the feature. (Forced with a `touch`: a
warm clippy on an up-to-date build emits nothing and proves nothing.)

`alloc_tracking` **must stay opt-in** and is **not a GOAT-promotion
candidate**: enabling it installs a `#[global_allocator]` doing a TLS read +
two adds per allocation. It is a measurement capability, not a perf primitive.

## The instrument, and the class boundary that IS the finding

- [x] `cfg_body()` returns the **conjunction** of every whole-file `#![cfg]`,
      wrapped as `all(...)` when there is more than one, so every downstream
      predicate (feature extraction, profile split, `any()` detection) sees one
      well-formed body and none of them needed to change. An **unbalanced**
      trailing attribute REFUSES the file rather than returning a partial
      conjunction, which would under-report exactly as the first-only bug did.
- [x] **A new class: ESCAPABLE vs UNFIXABLE profile gate.** After the repair
      both targets still contain the token `debug_assertions`, so a naive
      classifier still calls them DEBUG-only and the pin still reds — which
      would report the only available fix **as if it had changed nothing** and
      leave the pin raised forever. So the DEBUG-only half now splits:

      | class | meaning |
      |---|---|
      | **unfixable** | bare `debug_assertions`. No flag compiles it in release. `required-features` cannot express a profile, so a row moves it to `w/ req-f` (reading as protected) while changing nothing. A repair means rewriting the PREDICATE. **This is the pin worth having.** |
      | **escapable** | `any(debug_assertions, feature = …)`. Already repaired and verified in release. Still listed, because plain `--release` is still a green zero and the reader must know the feature exists — a documentation gap, not an unrunnable gate. |

      Matched **structurally**: the `any(` must contain both the profile term
      and a `feature =`. `any(debug_assertions, miri)` offers the reader no
      escape and stays unfixable.
- [x] Pinned from BOTH sides, because a bucket boundary is only testable
      against cases whose answer is known independently. A false ESCAPABLE
      retires a gate that genuinely cannot run in release; a false UNFIXABLE
      reports a completed repair as a no-op. Six cases: the repaired shape, the
      shape nested under an `all()`, the bare term, `any(debug_assertions,
      miri)`, a feature-only `any()`, `any(not(debug_assertions), feature=…)`
      (opposite direction), and the real two-attribute file shape fed through
      `cfg_body`.
- [x] `cfg_gated_floor_gate.py` reads `profile_gated_debug_only_unfixable`,
      **not** the pooled key, and its self-test fixture deliberately sets the
      pooled count to a value that would RED if the gate read the wrong key —
      the only assertion that can distinguish the two.
- [x] The gate's summary line prints the split. It printed the pooled 2, which
      no pin reads; a summary that prints a number no pin reads invites the
      next reader to treat it as the verdict.
- [x] **Proved the pin FIRES**, end to end, not just in the self-test: planted
      a bare `#![cfg(debug_assertions)]` back into `bench_284` → `UNFIXABLE 1 >
      pinned 0`, `✗ FAILED`, **exit 1**; restored → **exit 0**. Checked the
      real exit code without a pipe, since `| tail` returns tail's status and
      this repo has a documented class of gates that print FAILED and exit 0.

## Pins moved, each with its reason in the file that carries it

| file | key | 
|---|---|
| `cfg_gated_floors.txt` | `max_profile_debug_only` **1 → 0**, and re-pointed to the UNFIXABLE metric. A WALL — do not raise it; do what 741 did. |
| `cfg_row_implication_floors.txt` | `max_unresolved` **0 → 2**, both rows NAMED. `any(...)` containing a profile term is genuinely undecidable statically, and declining to rule is correct — the alternative is a confident answer that is wrong in one of the two profiles. The three-configuration measurement above is what licenses the pin; a third `any()` row needs its own. `max_empty_at_row` stays a WALL at 0. |
| `README.md` ×4, `examples/README.md` ×1 | total flag count **580 → 581**. |

## Not fixed here — filed where they are owned

Three UNFIXABLE debug-only alloc gates remain, all load-bearing, each fixable
the same way once its own allocator module takes a feature instead of a
profile:

| repo | target |
|---|---|
| `riir-neuron-db` | `tests/steering_g5_zero_alloc.rs` |
| `riir-train` | `crates/riir-train-gpu/tests/nora_phase2_g4_alloc_probe.rs` |
| `riir-train` | `crates/riir-train-gpu/tests/probe_684_topology_hotpath_alloc.rs` |

Plus the two newly-visible load-bearing SILENT-NOW rows (§2) and the +4
SILENT-NOW, in `riir-mmorpg-examples` and `riir-train`.

## The transferable lesson

Three distinct instruments in this repo said "clean" about the same two files,
and each was right about what it measured:

1. `cargo test --release` printed `ok. 0 passed`, **exit 0** — a green zero.
2. `cfg_gated_target_audit.py` reported no profile term — it read one of two
   attributes.
3. `max_profile_debug_only = 1` was green — it was a ceiling over instrument
   (2), and it carried a prose header confidently asserting the wrong count.

None of them lied. A ceiling is green over whatever its classifier can see,
and a classifier gap is **indistinguishable from a clean repo** — which is
what the closing paragraph of `cfg_gated_floors.txt` already said, before this
issue proved it again on the very pin that paragraph introduces.

And the axis lesson: **gating a MEASUREMENT on `debug_assertions` makes the
measurement impossible in the configuration that ships.** Ask of any
`debug_assertions` gate whether the thing behind it is a *capability* (feature)
or genuinely a *profile property* (an assertion about `debug_assert!`). Here it
was a capability the whole time — the counters work fine in release; they were
merely gated on the wrong axis, and "debug-only by design" had been written
into the pin's own header as if it were a constraint.

## Cross-refs

- riir-game-sdk `.issues/028` (31 targets, same class one axis over, resolved
  2026-09-09, `2380fc7`) · riir-clippy `.issues/083` · riir-dao `.issues/003`
- riir-ai `.issues/855` Class 2 (the PROFILE dimension's origin) and Class 3
  (an alloc+timing G4 measuring 3 ms/cycle in debug against 14.35 µs in
  release — a 209× artefact, the worked argument for why one file must not
  assert both)
- katgpt-rs `.issues/714` T3, Issue 728 T2 — "silently UNVERIFIED, not
  silently broken"
- `AGENTS.md` §"cfg-gated targets — the green-zero rule", §"The full gate"
  (the dev-vs-`--release` axis row)
