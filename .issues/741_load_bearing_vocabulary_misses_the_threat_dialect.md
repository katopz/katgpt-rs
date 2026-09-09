# Issue 741 — `is_load_bearing` cannot name a security gate that is named after its THREAT

**Status:** OPEN — filed 2026-09-09. Vocabulary finding on
`scripts/cfg_gated_target_audit.py`, measured over 2,328 test/bench target
names in 27 repos. **Adopting the fix changes today's counts by 0** (see
"Why the impact is zero" — that is the point, not a reason to skip it).

## The finding

`is_load_bearing()` decides whether a SILENT-NOW target's green zero is
*evidence*, and `scripts/cfg_gated_floors.txt` pins `max_load_bearing = 0` —
described there as "a WALL again, not a ratchet". The classifier reads target
NAMES against `LOAD_BEARING_TOKENS`: `goat`, `gate`, `drill`, `invariant`,
`guard`, `pin`, `proof`, `conservation`, `safety`, `security`, `audit`,
`alloc`, `correctness`, `determinism`, `equivalence`, `soundness`, `floor`,
`grad`, `regate`, `parity`, `monotonicity`, `roundtrip`, `integrity`,
`exactness`, `reachability`, plus `g<N>` ordinals and the `spec_match` bigram.

Every one of those names a **property the file asserts**. An entire naming
convention names the **threat the file defends against** instead, and the
classifier is blind to it. riir-game-sdk's `crates/riir-e2e` is written that
way — `prod_l<tier>_<threat>`:

| target | what it is | classified |
|---|---|---|
| `prod_l4_forgery_matrix` | a signature-forgery matrix | **not load-bearing** |
| `prod_l4_cpi_mitm` | a cross-program-invocation MITM probe | **not load-bearing** |
| `game_anticheat` | the anti-cheat suite | **not load-bearing** |
| `prod_l3_partition_heal` | a network-partition heal drill | **not load-bearing** |
| `prod_l2_crash_replay` | a crash-replay recovery gate | **not load-bearing** |
| `prod_l4_security_rejections` | security rejections | **load-bearing** ✓ |

The last row is the whole finding: same repo, same directory, same tier, same
purpose — and the opposite verdict, decided by whether the author happened to
write the word "security" in the filename. `prod_l3_sigkill_drills` is
load-bearing because it says "drills"; `prod_l3_partition_heal` is not,
because it does not.

Measured with the real classifier over riir-game-sdk's 31 gated targets:
**0 / 31 load-bearing**, and of the 14 that already carried rows, only 4
(`prod_l1_determinism`, `prod_l3_sigkill_drills`,
`prod_l4_security_rejections`, `prod_l5_g4_alloc`).

This is the third instance of the same class, and the doc already states the
lesson twice: "a ceiling of zero over a classifier is only as wide as the
classifier's vocabulary, and a vocabulary gap looks identical to a clean
repo" (`.docs/10_audits/cfg_gated_silent_zero_pass.md`; Issue 728's first
16-repo run found `silent_now_load_bearing = 0` in EVERY repo, which meant
"the classifier speaks one repo's dialect", not "nobody ships a silent
load-bearing gate").

## The measurement — 2,328 target names, 27 repos

The doc's discipline is measure-then-add, so every candidate was counted
across every `*tests/*.rs` and `*benches/*.rs` filename in the workspace
before being proposed. **ADMIT** — every hit is a thing the file exists to
FAIL on:

| token | hits | where | note |
|---|---|---|---|
| `forgery` | 2 | riir-chain `ledger_forgery`, riir-game-sdk `prod_l4_forgery_matrix` | both forgery-rejection |
| `mitm` | 1 | `prod_l4_cpi_mitm` | unambiguous by definition |
| `anticheat` | 1 | `game_anticheat` | unambiguous |
| `chaos` | 4 | riir-dapps ×2, riir-game-sdk ×2 | a chaos suite exists to fail on |
| `crash` | 4 | riir-chain ×2, riir-game-sdk ×2 | all crash-recovery |
| `agreement` | 4 | katgpt-rs, riir-ai, riir-clippy, riir-game-sdk | "two things must agree" — same family as the admitted `equivalence` / `parity` |
| `finiteness` | 2 | riir-chain `wire_finiteness_audit`, `prod_l4_finiteness_channels` | bounds assertions |
| `partition` | 1 | `prod_l3_partition_heal` | network sense; only hit |
| `sigkill` | 1 | `prod_l3_sigkill_drills` | already caught via `drills` |
| `overflow` | 1 | riir-chain `settlement_recipient_overflow` | arithmetic-overflow rejection |
| `fuzz` | 1 | riir-chain `ledger_conservation_fuzz` | already caught via `conservation` |

Bigrams (neither half admissible alone): `crash_replay`,
`divergence_injection`, `front_run`.

**REJECT — measured homonyms.** This half is the more useful half, because it
is the work the next person does not have to repeat:

| token | hits | why NOT |
|---|---|---|
| `replay` | 10 | includes **perf probes** — `bench_618_graph_replay_probe`, `probe_742_graph_replay`. Not gates |
| `divergence` | 9 | usually a **measured quantity**, not a gated one — `k3_pause_logit_divergence_bench`, `bench_337_phase4_g7_persona_divergence` |
| `injection` | 3 | also a bench **technique** — `triggered_injection_bench` |
| `rejection` | 1 | katgpt-rs `rejection_uniformity` is **rejection sampling** — a different word entirely |
| `watermark` | 2 | two senses: riir-chain `forensic_watermark` (steganographic) vs `prod_l2_watermark_agreement` (stream progress marker). `agreement` covers the second anyway |
| `tamper`, `spoof`, `dos`, `adversar`, `byzantine`, `exploit` | 0 | a zero-hit token cannot be validated against homonyms. Reserved, not admitted — add one when its first real target arrives |

## Why the impact is zero, and why that is the argument FOR landing it

Adopting the ADMIT set today moves `silent_now_load_bearing` from **0 to 0**
workspace-wide, and adds no findings in any other repo (measured by
monkey-patching the token set and re-running the audit over all 17 contract
repos). No homonym damage, no new backlog.

The reason is ordering, not health: riir-game-sdk `.issues/028` armed all 31
of those targets earlier the same day (`2380fc7`), so they left the
SILENT-NOW population before this measurement ran. **The counterfactual is
the finding** — run the widened classifier over those 31 names as they stood
before `2380fc7` and it flags **11**:

`prod_l2_crash_replay` · `prod_l2_program_state_crash` ·
`prod_l2_watermark_agreement` · `prod_l3_divergence_injection` ·
`prod_l3_partition_heal` · `prod_l4_cpi_mitm` ·
`prod_l4_finiteness_channels` · `prod_l4_forgery_matrix` ·
`prod_l4_mev_front_run` · `game_anticheat` · `game_settlement_chaos`

Eleven load-bearing SILENT-NOW targets would have red-ed `max_load_bearing =
0` — the workspace's sharpest pin — and it stayed green throughout. It is
green today for a good reason and was green yesterday for a bad one. The gap
reopens the moment somebody adds the next `prod_l4_<threat>` target, and the
pin will not say so.

## Fix

- [ ] Add the ADMIT tokens + 3 bigrams to `LOAD_BEARING_TOKENS` /
      `LOAD_BEARING_BIGRAMS`, each with its measured hit count in the comment,
      matching the existing convention for the 2026-09-03 and Issue 728
      additions
- [ ] Record the REJECT table in
      `.docs/10_audits/cfg_gated_silent_zero_pass.md`'s token table — the
      homonym evidence is the reusable artefact, and without it the next
      person re-measures `replay` and `divergence` and gets it wrong
- [ ] Re-run `scripts/cfg_gated_target_audit.py` (all 17) and confirm
      `silent_now_load_bearing` is still 0 in every repo, with `scanned` and
      `gated` unchanged. If a repo DOES light up, that is a real find and it
      is armed before the pin is touched, per Issue 728 T2's precedent
- [ ] Note in the floors file that the doc's "SILENT-NOW does not go to zero
      and should not — the residual are targets whose green zero nobody cites
      as evidence. Arming those is churn" now carries a caveat: whether a
      green zero is *cited as evidence* is decided by `is_load_bearing`, so
      "churn" and "the classifier cannot read the name" are indistinguishable
      from inside the report. riir-game-sdk's 31 read as churn and 11 were not

## Cross-refs

riir-game-sdk `.issues/028` (`2380fc7`) — the 31 targets, armed and verified
both directions, which is what makes the counterfactual measurable.
riir-clippy `.issues/083` (`2c7be78`) and riir-dao `.issues/003` (`f3a0578`)
— the same class in two more repos, same day.
`.docs/10_audits/cfg_gated_silent_zero_pass.md` — the doctrine, the token
table, and the two prior instances of this exact vocabulary failure.
`scripts/cfg_gated_floors.txt` — the `max_load_bearing = 0` wall and the
Issue 728 narrative of a classifier widening that temporarily red-ed it.
