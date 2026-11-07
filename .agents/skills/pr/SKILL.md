---
name: pr
description: Universal Do/Don't doctrine for opening and maintaining upstream pull requests as an agent — the trust rules distilled from real campaign rounds (slop tells, review-comment handling, CI triage, reply voice, force-push hygiene, rate ceilings). Use when opening a PR to any external repository, responding to a maintainer review, deciding whether a change is PR-worthy, writing or rewriting a PR/comment body, or triaging red CI on our PR; also when asked "how do we PR into X". Complements the campaign-specific skills (riir-refine's outbound-pr = the gated healer campaign lane; this one is the repo-agnostic etiquette + failure taxonomy).
---

# pr — upstream pull-request Do/Don't (agent doctrine)

Every rule below is something a real round paid for. Cite-the-round examples
are from mozilla/sccache PR #2977 (the worked example, 6 rounds, 3 owner
corrections, 2 CI breaks). When a campaign lane has its own skill (e.g.
`outbound-pr`), that skill's gates stack ON TOP of these — these are the
floor, not the ceiling.

## The one principle

**The PR is the fix + the verification story — nothing else.** Every other
rule is this principle applied to a surface: the diff, the body, the reply,
the cadence.

## DO

- **Do derive every factual claim from the primary source, in the same
  session.** Tool lists, config defaults, spec quotes, "their CI runs X" —
  fetch/grep the primary source before the claim is written. (Failure:
  "rsync with an exclude plugin" — rsync has no CACHEDIR.TAG support; the
  spec's own page lists it under "not directly sensitive". Caught by an
  external verdict; would have been caught by one curl.)
- **Do make the body agree with the diff.** After writing the body, re-read
  the diff and check every behavioral sentence against it. (Failure: body
  said "logs a warning", diff called `error!`.)
- **Do quote their exact numbers/defaults, not vibes.** ("defaults to 10
  GiB" not "can grow to tens of GB" — their config has the constant.)
- **Do verify at THEIR toolchain.** rustup their MSRV (rust-version field >
  rust-toolchain.toml > clippy.toml > ABSENT ⇒ refuse, never guess); run
  their CI's exact clippy flags; their fmt. The local bar is not the bar.
- **Do bench-then-PR for any perf claim.** Before/after at the target repo,
  pinned harness, box provenance (load/RAM/power beside every number). A
  number you have not measured does not ride.
- **Do answer reviews in short plain sentences.** Name the commit sha. Use
  their vocabulary. One idea per line. Open with the substance (the catch,
  the mechanism), not a pleasance ("Thanks for the quick review! Fixed all
  four:" is a ledger, not a reply).
- **Do own misreads in one clause.** "— sorry, I misread it as a main issue
  earlier" and move on. Over-apologizing reads worse.
- **Do patch existing comments (API PATCH) instead of posting duplicates.**
- **Do re-grep the touched FILE for the pattern a review comment flagged
  before pushing.** A review comment is a CLASS signal, not a line signal.
  (Failure: maintainer flagged the hardcoded literal + manual read in the
  product; we fixed the flagged lines but 4 more literals + the same dance
  survived in the test 20 lines below — one more round.)
- **Do acknowledge an existing claimant before sending.** Read the issue's
  comments for "I'd like to work on this" — a PR over an unacknowledged
  open claim reads as sniping. One plain sentence naming the earlier offer
  ("X offered to pick this up; if you've started, glad to fold this into
  yours"), or take the design-comment-first lane on contested issues:
  post the shape on the issue, open the PR after a reply. (Caught by the
  hard verdict on our second sccache PR — darshpatel1052 had claimed the
  work two days before; our body answered his design question but never
  acknowledged the claim.)
- **Do ship the docs row in the same PR.** A new env knob, flag, or config
  surface gets its entry in THEIR docs file in the same change, in the
  file's own voice. The docs file is a reader of your surface — an
  undocumented knob is a near-certain review round (and the FC10 class:
  the sibling knobs were documented; ours weren't).
- **Do sweep the composite layers that route through your seam.** Every
  wrapper/composite/dispatch that calls what you changed is a reader — a
  gate added at a storage seam skips the healthy local tier of a
  multilevel wrapper too (the M3 catch: `MultiLevelStorage`'s read loop,
  three files from the diff, no tree-walk grep finds a call-graph edge).
  Grep for CALLERS of the modified functions, not just walkers of trees.
- **Do name the workload shape beside every number.** "Sequential" vs the
  issue's parallel repro changes the result (in-flight probes multiply
  under -j); if the issue ships a repro script, run it before/after or
  state explicitly that its workload was not measured. A ratio that
carries over only in one shape is scoped to that shape IN THE BODY.
- **Do answer every requirement the issue states.** The issue's stated
  requirements are a checklist — each gets a body answer or an explicit
  "not addressed". (Miss: the issue said "avoid overreacting to transient
  slowness"; our body never said how correlated timeouts under parallelism
  don't trip the gate at once.)
- **Do verify a zero before claiming it.** A "clean / 0 findings" claim
  carries its completion marker — the tool's build-finished/success line,
  a non-empty capture, stderr collected (never /dev/null). A command that
  dies before compiling (wrong flag order: lint `-A` flags before `--`)
  produces an empty capture that reads as a green zero. The pre-existing
  baseline is adjudicated FIRST: same command on pristine main, failures
  named, then read your tree as a delta against it.
- **Do count the staleness.** Before reusing any earlier pin, `git
  rev-list --count <pin>..origin/main` — never eyeball. The readiness
  verdict caught "two commits" that were 21, three of them
  bench-load-bearing. Re-run the gates whose inputs moved.
- **Do grep EVERY walk/scan of any tree your new file lands in.** Adding a
  file to a repo means every recursive iteration over that tree now sees
  your file — find them all before pushing, not in someone's CI. (Failure:
  a CACHEDIR.TAG broke `--debug-preprocessor-cache`'s third walk; two CI
  runs and a freebsd log line to find it.)
- **Do rebase onto current main before every push** and re-check whether a
  maintainer fixed your issue upstream mid-review (the lockfile we shipped
  was superseded by the maintainer's own identical commit the same day).
- **Do force-push with the same branch name** (`refine/<topic>`) during an
  open review round — it keeps the PR thread one diff. New commits only for
  separable concerns a maintainer might cherry-pick or drop independently.
- **Do keep fixes the maintainer can drop independently in separate
  commits** (e.g. an incidental lockfile fix as its own `deps:` commit).
- **Do check repo house style for commit prefixes** (`deps:`, `fix:`,
  whatever their history shows) and match it.
- **Do stop and report "nothing to send" as a PASS.** A clean tree at their
  posture means the gate worked. A manufactured PR is the failure.
- **Do expect first-contribution CI to need a maintainer's "approve and
  run"** (`action_required` state) — say so in the reply rather than
  re-pushing to trigger it.

## DON'T

- **Don't add attribution/disclosure/self-promo** — no "Generated by X", no
  tool name, no site link, no logo, no repro-advertising, no aggregate
  stats. (Owner ban 2026-11-02: AI slop + shameless self-promo; public-goods
  posture keeps it out until the grant lands.) The proof is the fix + the
  verification story.
- **Don't send a PR whose purpose is "having sent a PR".** If the scan says
  nothing, the round is a pass. A hand-authored non-evidence PR under a
  campaign banner is the Goodhart failure. (Failure: a convenience-feature
  PR sent as "evidence" — the owner's verdict: goodwill, outside the ledger,
  and it evidences nothing.)
- **Don't invent a new lane mid-round to justify a send.** Campaign/marketing
  posture is the OWNER's call (what counts as evidence, what we send into).
  Surface the options and wait if the scan came back empty.
- **Don't fix only the flagged lines.** (See the re-grep DO.)
- **Don't trust your first root-cause read of a CI failure.**
  Reproduce
  locally, read the actual log lines (the failing stderr usually names the
  file — one freebsd log line contained the whole answer), and check whether
  the failure reproduces on a clean checkout of their main BEFORE writing
  "reproduces on main" in any reply. (Failure: we misattributed an
  `entry format 53` failure to their preprocessor rework; it was our file;
  the reply had to own it a round later.)
- **Don't expect their CI to be green, or to run at all.** First-contribution
  CI is maintainer-triggered (`action_required`) — zero runs for days is
  normal, and when runs appear, main's own pre-existing reds ride along
  (13 red jobs on our PR's first CI, all ONE upstream test broken by their
  own commit). Classify reds by grepping their logs for YOUR artifact
  names, never by job name; say so in the reply rather than re-pushing to
  trigger it.
- **Don't diagnose lockfile/dependency failures from the error text alone.**
  `--locked` failures: read the manifests of the involved crates in the
  registry (pinned reqs vs resolved versions) and check whether MAIN's lock
  is the broken thing. (Failure: "unpinned dep" — both versions pinned
  `= "0.12"`; main's lock was hand-merged inconsistent.)
- **Don't downgrade/upgrade deps inside a feature PR.** Minimal resolver
  delta, separate commit, or leave it to the maintainer.
- **Don't run repo-wide tools (fmt, tests, cargo commands) from the wrong
  working directory.** A `cargo fmt` with the wrong cwd reformats someone
  else's worktree — 52 files of sibling WIP in one incident; recovery was
  mtime triage + targeted checkout. Check `pwd` against the manifest you
  mean. Never `cargo fmt --all` in a repo with intentional drift or
  siblings.
- **Don't pile a competing PR onto another contributor's open PR** without
  an owner call, and don't argue in maintainer threads — rework or accept
  the close.
- **Don't exceed ≤3 PRs/day across the campaign, one PR per repo ever,
  never a re-PR after a close.** A repo block = permanent blacklist. Never
  CLA/DCO auto-sign — skip those repos.
- **Don't file issues on external repos** (the maintainer thread is the
  channel), don't auto-merge, don't ship dependency-bump PRs, don't send
  security findings as PRs (suggest-only, private disclosure path).
- **Don't keep persistent copies of their code.** Clones are ephemeral
  (`.raw/`, removed at close-out); only derived data (rule ids, shape
  classes, outcomes) survives in anything that ships.
- **Don't let a per-file edit tool stitch long ledger/record lines.** When
  updating round-based records, prefer appending whole rows or scripted
  replace with uniqueness asserts — three repair incidents in one record
  came from replace-instead-of-prepend edits.

## The pre-push checklist (60 seconds, every push)

1. `pwd` is the repo you intend; branch is yours; base is current main — and
   the staleness is COUNTED (`git rev-list --count <base-pin>..origin/main`),
   with the gates whose inputs moved re-run.
2. Every factual claim in body/reply has a primary-source check this session.
3. Body agrees with the diff; no attribution block; no unearned numbers; the
   workload shape is named beside every number; every requirement the issue
   states is answered or explicitly not addressed.
4. Re-grep the touched files for the patterns already flagged in review; the
   composite layers routing through your changed seams are adjudicated; new
   env knobs have their docs rows.
5. Their MSRV: tests green, their CI clippy flags clean, fmt clean — every
   zero carries its completion marker (build-finished/success + non-empty
   capture + stderr collected), and the pristine-main baseline was
   adjudicated first so reds read as a delta.
6. The issue's comments checked for claimants; any claim acknowledged in the
   body.
7. Diff contains only intended hunks (`git diff` read top to bottom).
8. If perf claim: bench exists, box provenance recorded.
9. If review response: every inline comment answered, reply is short/plain,
   commit shas in the reply match what you're pushing.
