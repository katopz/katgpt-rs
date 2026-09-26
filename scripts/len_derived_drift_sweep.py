#!/usr/bin/env python3
"""The `.len()`-derived binding verdict over every contract repo, PINNED.

`scripts/len_derived_binding_audit.py` was the LAST cross-repo instrument in
`scripts/` with no verdict half (Issue 786). It walks 8,694 tracked `.rs` over
16 repos, finds 52 `.len()`-deriving cube kernels and classifies 164 bind sites
into nine buckets — and nothing asserted any of it. No gate, no sweep, and no
`AGENTS.md` entry at all; its only mentions in this repo were three incidental
`HISTORY.md` lines.

Ninth instance of one shape (Issues 777, 778, 793, 782, 783, 784, 785): a rule
landed in one instrument and never generalised. This one was QUIETER than 784
and 785 — it never had a hand-typed standing figure to go stale, so there was
nothing to catch being wrong. An instrument nobody is told about does not drift
into error in public; it simply stops being run. That is the worse direction
and the reason to look for these by census rather than by symptom.

The class is not cosmetic: a kernel deriving a structural dimension from a
bound buffer's `.len()`, handed a buffer whose DECLARED size exceeds its live
range, silently derives the wrong shape — reads never-written memory, writes a
measured identically-zero result, no panic and no NaN (riir-ai `3e00c93e0`,
riir-train `.issues/511`).

Three pin shapes, each a MEASUREMENT rather than a copy of the last sweep
-------------------------------------------------------------------------

**1. The finding buckets are WALLS at 0.** CAPACITY, CAPACITY-UPSTREAM and
PERSISTENT are the joined defect — HALF A (in-kernel `.len()` derivation) met by
HALF B (a bind whose declared size can exceed the live range). All three are 0
today; a ratchet would let the first real one land silently.

**2. PERSISTENT-UPSTREAM is pinned by MEMBERSHIP, not by count.** It is the
EYES LIST — 4 rows, all riir-ai — and a count goes green the day one is
adjudicated and another regresses. The key is `(repo, file, kernel, handle)`,
deliberately LINE-FREE: a line number moves when anything above it is edited,
and a pin that reds on an unrelated edit is a pin somebody deletes. Because
that key is not unique in general (`attention_decode_q8kv`/`query_handle` binds
at two lines of one file), each row also carries its own row COUNT — within a
single address the substitution hazard that rules counts out ACROSS rows does
not apply, and without it a second bind site at a pinned address would be
invisible. Paths are normalised to `/` so the file is not Windows-shaped.

**3. UNRESOLVED (118 of 164, 72%) is deliberately UNPINNED.** Issue 785's rule:
a ratchet on a bucket whose meaning is *unanswered* is a backlog. wasm32 could
wall its UNRESOLVED at 0 only because Issue 738 T1 drove it there by ANSWERING
the rows (15 → 0); here the bucket is "a wrapper parameter whose provenance
lives one level up, whose caller is not a path-form associated fn", which HALF C
cannot reach at all. So it is reported, its reason is stated at the point it is
READ rather than only in this docstring, and it is never folded into a pass or
a fail. Same standing as `suite_membership_audit.py`'s 1,203 rows.

Why there is NO `min_rs_files` column, and why that is an assertion
-------------------------------------------------------------------
Every other sweep in this family carries one. A fourth copy here would add
exactly ZERO detection power: the walk is `tracked_walk.tracked_files(repo,
"*.rs")` — the identical call, over the identical population — and three sweeps
already floor it per repo (`orphaned_attr`, `platform_dead_code`, `percentile`).
If `git ls-files` goes blind in riir-chain, all three red.

They also already DISAGREE about that quantity — katgpt-rs is pinned 1500 /
1400 / 1500 and riir-ai 1500 / 1500 / 1800 across those three files, against
2415 and 2626 measured. Harmless, because each is an independently chosen slack
floor rather than an equality, but it is the shape this repo keeps rediscovering
and there is no reason to make it a fourth.

⚠ "Somebody else covers it" is an ASSUMPTION unless it is checked, so this sweep
checks it: every repo it pins must still carry a NON-ZERO `min_rs_files` row in
`orphaned_attr_drift_floors.txt`, and the sweep reds if that column is ever
dropped or zeroed. Delegation asserted, not trusted.

⛔ One MEASURED exception (Issue 902): a repo born md-only — no tracked `.rs`
at all (riir-instinct) — has a TRUTHFUL zero row, and reding on it forever is
the cries-wolf state Issue 793 forbids. The zero row is accepted only while
`tracked_walk` (the ONE walk, Issue 777) measures ZERO tracked `.rs` in that
repo, re-measured EVERY run; the first `.rs` to land reds exactly as a zeroed
row on a code repo does. The acceptance prints, it is never silent. (It also
catches the stale premise in the other direction: riir-reflexer was registered
md-only 2026-09-25, but its vessel workspace had landed 16 tracked `.rs` by the
time this shipped — the measurement refused the zero row and the rows were
re-pinned in the same commit.)

What this sweep owns that no other one can
-------------------------------------------
`min_kernels` and `min_binds`, the PARSE floors. They move when the classifier
breaks on an unchanged tree — which has happened: before the brace-matched body
fix (2026-09-13) the fixed 6000-char window bled and counted 74 kernels, 22 of
them host-side false positives.

⚠ They are VACUOUS in 14 of 16 repos, and that is a measurement, not an
oversight:

    riir-ai      43 kernels / 143 bind sites
    riir-train    9 kernels /  21 bind sites
    the other 14  0 / 0

This is Issue 783's population shape (where `min_calls` is 0 in 10 of 16), not
Issue 784's (where both floors bite everywhere). Do not carry one repo set's
warrant onto another. There is no reserved `TOTALS` row precisely BECAUSE of
this: a global kernel floor would red on a partial-clone box the moment riir-ai
or riir-train were absent, which is the case `population_verdict()` exists to
DEFER, and the two non-zero per-repo floors already catch a classifier that
breaks workspace-wide.

The one axis no other sweep in this family has: HALF C is cross-repo
---------------------------------------------------------------------
`trace_unresolved()` resolves a wrapper parameter's provenance through WORKSPACE
callers, so a partial checkout can corrupt the verdict of a row in a repo that
IS present. `DOCS_GATE_PARTIAL_CLONE`'s DEFERRED verdict does not cover that —
it addresses rows this run could not MEASURE, not rows it measured WRONG. Every
other sweep here classifies each repo independently, so DEFERRED is sufficient
there and is not sufficient here.

Measured 2026-09-14, both directions, rather than reasoned about:

- **7 of 251** cited caller references are cross-repo (riir-ai bind sites
  resolved through riir-train callers).
- **Leave-one-out over all 16 repos: 0 verdict flips.** Every one of those 7
  edges is an ADDITIONAL caller on a row already decided by a same-repo caller.

So per-repo pins are sound TODAY, and the day that stops being true is a day
this sweep must ANNOUNCE rather than defer. The check is TARGETED, not
exhaustive — 16 re-classifications at ~8s each is ~2 minutes, and only a repo
that actually SUPPLIES a cross-repo citation can flip anything. The supplier set
is derived from the run (today `{riir-ai, riir-train}`), so a new cross-repo
edge joins the check by EXISTING rather than by somebody remembering to add it.
`--no-stability` skips it when only the pins are being re-measured.

Why this is NOT in scripts/docs_gate.sh's CHECKS
------------------------------------------------
Identical to the other thirteen sweeps: CI has one checkout, the siblings are
private and simply absent, so this would either red on every run or derive an
EMPTY population and print a confident green over zero repos.

    this script                      workstation, on demand, every contract repo
    len_derived_binding_audit.py     the report half, exit 0 by design

Exit 0 clean, 1 on drift above the pins, **2 if the instrument itself is
untrustworthy** — an unreliable instrument is not the same finding as drift.

`--canary` runs the 12 adversary arms (see `canary()`); `--no-stability` skips
the leave-one-out.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# DRY: the classifier is the report half's, extracted by Issue 786 T1, so the
# sweep and the report can never disagree about what any bucket MEANS.
import len_derived_binding_audit as lda  # noqa: E402
from skill_repo_set_gate import derive_repos as derive_repo_names  # noqa: E402
from sweep_population import (  # noqa: E402
    population_verdict, pin_row_exempt, zero_walk_floor_accepted)
from worktree_state import (  # noqa: E402
    HeadDelta, deferral_line, delta_of, dirty_in_population, head_tree,
    sweep_advisory)

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "len_derived_drift_floors.txt"
EYES = HERE / "len_derived_eyes_expected.txt"
# The file whose `min_rs_files` column this sweep DELEGATES its walk floor to.
DELEGATED_WALK_PINS = HERE / "orphaned_attr_drift_floors.txt"

FIELDS = ("min_kernels", "min_binds", "max_findings")
# The joined defect. PERSISTENT-UPSTREAM is the EYES LIST and is pinned by
# membership instead; the remaining buckets are not findings.
FINDING_BUCKETS = ("CAPACITY", "CAPACITY-UPSTREAM", "PERSISTENT")
EYES_BUCKET = "PERSISTENT-UPSTREAM"


def parse_pins(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 1 + len(FIELDS):
            raise ValueError(
                f"malformed pin row (want {1 + len(FIELDS)} fields): {raw!r}")
        rows[parts[0]] = dict(zip(FIELDS, (int(v) for v in parts[1:])))
    return rows


def parse_eyes(path: Path) -> dict[tuple[str, str, str, str], int]:
    """(repo, file, kernel, handle) -> expected row count. See the docstring."""
    rows: dict[tuple[str, str, str, str], int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"malformed eyes row (want 5 fields): {raw!r}")
        rows[(parts[0], parts[1], parts[2], parts[3])] = int(parts[4])
    return rows


def delegated_walk_floors(path: Path) -> dict[str, int]:
    """`min_rs_files` per repo from the file this sweep delegates that axis to.

    Parsed positionally and defensively: this sweep does not own that file's
    schema, so a shape it cannot read must be reported as UNREADABLE rather
    than silently yielding an empty dict, which would turn the delegation
    assertion into a no-op — precisely the failure it exists to prevent.
    """
    floors: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"unreadable delegated row: {raw!r}")
        floors[parts[0]] = int(parts[1])
    return floors


def norm(rel: str) -> str:
    """Pin files are not Windows-shaped: the audit emits the host separator."""
    return rel.replace(chr(92), "/")


def eyes_key(b) -> tuple[str, str, str, str]:
    return (b.repo, norm(b.file), b.kernel, b.handle_expr)


# ONE list, read by the worktree advisory, by the HEAD trigger and by the
# archive pathspec. `classify_workspace` walks TRACKED `*.rs` (Issue 777) and
# reads nothing else, which is what makes both the narrowing and this constant
# sound — its inputs are ENUMERABLE.
SCOPE = ("*.rs",)


def detail_of(bk: dict[str, int]) -> str:
    return " ".join(f"{v}={n}" for v, n in sorted(bk.items())) or "—"


def bind_key(b) -> tuple:
    """A bind row's LINE-FREE, TREE-FREE identity.

    ⛔ **No LINE.** Any edit above a bind site shifts it, so a line-bearing key
    reports every row in an edited file as UNCOMMITTED *and* MASKED at once —
    the rule `line_free`/`ordinal_keys` exist for, three sweeps over.

    ⛔ **The VERDICT is in it.** Every ceiling here partitions by bucket
    (`FINDING_BUCKETS` against `max_findings`, `EYES_BUCKET` against the
    membership wall), so a key omitting it lets a worktree row silently
    override HEAD's — and the EYES direction is DESTRUCTIVE: a pinned row that
    looks "no longer reported" invites dropping the only record of a committed
    bind.

    ⚠ It is TREE-FREE for a reason worth stating rather than assuming: `b.file`
    is ALREADY repo-relative (measured — the classifier stores `src/k.rs`, not
    an absolute path), and `b.repo` is the DIRECTORY NAME, which is why this
    sweep depends on `head_tree` naming its checkout after the source repo.
    Nothing here strips a prefix; if either of those two facts changes, the two
    sides stop matching and every row reads as UNCOMMITTED *and* MASKED at
    once.

    ⛔ `b.repo` is load-bearing and its arm had to be built for it: two repos
    whose rows are otherwise identical — the same relative file, kernel and
    handle, which is what a kernel copied between siblings looks like — would
    otherwise let one repo's LIVE row match another's committed-and-hidden one,
    emptying MASKED. A committed defect reported clean because a different repo
    still has it.
    """
    return (b.repo, norm(b.file), b.kernel, b.handle_expr, b.verdict)


def adjudicate(repo_paths, rep, classify=None):
    """-> (`HeadDelta` over the bind rows, HEAD's report or None).

    Issue 822 T5j, and this is the one sweep in the family whose HEAD side is a
    WORKSPACE rather than a repo. `classify_workspace` resolves a wrapper
    parameter's provenance through workspace CALLERS (HALF C), so classifying
    one repo at HEAD against fifteen worktrees would mix the two states inside
    a single verdict — worse than either. Every repo is passed in one call,
    with the DIRTY ones swapped for materialised HEAD checkouts and the clean
    ones left exactly as they are, which costs nothing for them.

    ⚠ This depends on `head_tree` naming its checkout after the source repo:
    every row here is keyed on `b.repo`, which the classifier takes from the
    directory name. Two repos materialised at once would otherwise both be
    `head` — a collision AND a misattribution. Asserted in `worktree_state`'s
    own arms, not assumed here.

    `None` for the second element means no repo in the workspace is dirty in
    SCOPE, and the caller reads the worktree's own report — the conservative
    direction for a bucket the pins read.
    """
    dirty = [p for p in repo_paths if dirty_in_population(p, SCOPE)]
    if not dirty:
        return HeadDelta([(bind_key(b), b) for b in rep.binds], [], []), None
    classify = classify or lda.classify_workspace
    with contextlib.ExitStack() as stack:
        swapped = []
        for p in repo_paths:
            tree = stack.enter_context(head_tree(p, SCOPE, paths=SCOPE))
            swapped.append(tree if tree is not None else p)
        head = classify(swapped)
    return (delta_of([(bind_key(b), b) for b in rep.binds],
                     [(bind_key(b), b) for b in head.binds],
                     lambda kb: kb[0]), head)


_KERNEL = """#[cube(launch_unchecked)]
fn attention_decode_f32(query: &[f32], kv: &[f32]) {
    let kv_half = kv.len() as u32 / 2u32;
}

pub fn host() {
    let kv_handle = client.empty(cap);
    unsafe { attention_decode_f32::launch_unchecked::<R>(c, a,
        BufferArg::from_raw_parts(kv_handle, %s)) }
}
"""
# ⛔ The REAL classifier, captured at import. `canary()` below STUBS
# `lda.classify_workspace` to a fixed report — correctly, its arms are about pin
# arithmetic — and it reaches `main()`, which runs `selftest()`, which runs the
# provenance arms. Without this the arms measure the canary's stub and every one
# of them fails, taking all 12 canary arms down with it (measured, first run).
_REAL_CLASSIFY = lda.classify_workspace

CAPACITY_SRC = _KERNEL % "kv_handle.len()"      # a FINDING (declared size)
UNRESOLVED_SRC = _KERNEL % "n_positions"        # not a finding, same shape


def adjudicate_cases() -> list[str]:
    """`adjudicate` end to end against REAL git — Issue 822 T5j.

    A real workspace, a real kernel and a real bind site: the classifier is
    the thing under test, and `canary()` above deliberately STUBS it (its arms
    are about pin arithmetic), so a stub here would leave the seam this task
    landed covered by nothing at all — T5e's finding, which is that nothing
    automatic walls this join.
    """
    fails: list[str] = []

    def git(root, *args):
        subprocess.run(("git", "-C", str(root)) + args,
                       capture_output=True, check=True)

    def repo_at(ws: Path, name: str, committed: str, worktree: str | None):
        repo = ws / name
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "k.rs").write_text(committed, encoding="utf-8")
        (repo / "BOUNDARY.md").write_text("x", encoding="utf-8")
        git(ws, "init", "-q", name)
        git(repo, "config", "user.email", "arm@example.invalid")
        git(repo, "config", "user.name", "arm")
        git(repo, "add", "-A")
        git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "base")
        if worktree is not None:
            (repo / "src" / "k.rs").write_text(worktree, encoding="utf-8")
        return repo

    def run(paths, classify=None):
        classify = classify or _REAL_CLASSIFY
        rep = classify(paths)
        delta, head = adjudicate(paths, rep, classify=classify)
        return rep, delta, (rep if head is None else head)

    def caps(rep):
        return [b for b in rep.binds if b.verdict == "CAPACITY"]

    # a. ⛔ Issue 798's direction: a fix that is not COMMITTED is not landed.
    #    CAPACITY is a WALL at 0, so an uncommitted repair clearing it hands
    #    every other checkout a green over a live defect.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = repo_at(ws, "r", CAPACITY_SRC, UNRESOLVED_SRC)
        rep, delta, j = run([repo])
        if caps(rep):
            fails.append(f"arm a: the fixture is INERT — the worktree must "
                         f"read no CAPACITY ({[b.verdict for b in rep.binds]})")
        if len(caps(j)) != 1 or not delta.masked:
            fails.append(f"adjudicate: an UNCOMMITTED fix cleared the CAPACITY "
                         f"wall (judged {len(caps(j))}, masked "
                         f"{len(delta.masked)}) — the committed kernel still "
                         f"derives the wrong shape for everyone else")

    # b. And its mirror: a defect introduced in the worktree must not red a
    #    wall at 0 over a line no commit contains.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = repo_at(ws, "r", UNRESOLVED_SRC, CAPACITY_SRC)
        rep, delta, j = run([repo])
        if len(caps(rep)) != 1:
            fails.append("arm b: the fixture is INERT — the worktree must "
                         "read one CAPACITY")
        if caps(j) or len(delta.uncommitted) != 1:
            fails.append(f"adjudicate: an uncommitted CAPACITY reached the "
                         f"wall ({len(caps(j))}) instead of UNCOMMITTED "
                         f"({len(delta.uncommitted)})")

    # c. ⚠ The WORKSPACE axis, which no other sweep in the family has: HALF C
    #    resolves provenance through other repos, so the HEAD side must be a
    #    whole workspace — and each materialised repo must keep ITS OWN NAME,
    #    or two dirty repos both arrive as `head` and every row is
    #    misattributed. This is the arm that fails if `head_tree` stops naming
    #    its checkout after the source.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        a = repo_at(ws, "alpha", CAPACITY_SRC, UNRESOLVED_SRC)
        b = repo_at(ws, "beta", CAPACITY_SRC, UNRESOLVED_SRC)
        rep, delta, j = run([a, b])
        got = sorted({x.repo for x in j.binds})
        if got != ["alpha", "beta"]:
            fails.append(f"adjudicate: the HEAD workspace lost the repo names "
                         f"({got}) — two materialised checkouts collided, so "
                         f"every row is attributed to the wrong repo")
        if len(delta.masked) != 2:
            fails.append(f"adjudicate: {len(delta.masked)} MASKED row(s) over "
                         f"two repos that each hide one committed finding")

    # c2. ⛔ And the REPO must be in the key, which (c) cannot show because
    #     both its repos move together. Two repos whose rows are otherwise
    #     IDENTICAL — same relative file, kernel and handle, which is exactly
    #     what a copied kernel across siblings looks like — and only one of
    #     them hides its committed finding. Without `b.repo` the other repo's
    #     live row matches it and the MASKED bucket goes empty: a committed
    #     defect reported clean because a DIFFERENT repo still has it.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        a = repo_at(ws, "alpha", CAPACITY_SRC, UNRESOLVED_SRC)
        b = repo_at(ws, "beta", CAPACITY_SRC, None)
        rep, delta, j = run([a, b])
        if [k[0] for k, _b in delta.masked] != ["alpha"]:
            fails.append(f"row key: the repo is not in the key — alpha's "
                         f"committed finding was matched by beta's live one "
                         f"({[k[0] for k, _b in delta.masked]})")

    # d. The key must be stable under unrelated dirt: a repo whose only change
    #    is another file must report its own bind as moved NOWHERE.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = repo_at(ws, "r", CAPACITY_SRC, None)
        (repo / "src" / "other.rs").write_text("// dirt\n", encoding="utf-8")
        git(repo, "add", "-A")
        rep, delta, j = run([repo])
        if delta.uncommitted or delta.masked:
            fails.append(f"row key: an UNCHANGED bind moved on unrelated dirt "
                         f"(uncommitted={len(delta.uncommitted)}, "
                         f"masked={len(delta.masked)})")

    # d2. ⛔ And LINE-FREE specifically, which (d) cannot reach: the bind must
    #     survive an edit ABOVE it. A line-bearing key reports every row in an
    #     edited file as UNCOMMITTED *and* MASKED at once — the defect that
    #     `line_free`/`ordinal_keys` exist for, and the reason this arm plants
    #     a shift rather than a change.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = repo_at(ws, "r", CAPACITY_SRC,
                       "// a line nobody committed\n" + CAPACITY_SRC)
        rep, delta, j = run([repo])
        if delta.uncommitted or delta.masked:
            fails.append(f"row key: a bind whose LINE moved was reported as "
                         f"moved (uncommitted={len(delta.uncommitted)}, "
                         f"masked={len(delta.masked)}) — the key carries a "
                         f"line number, so any edit above a row reds it")

    # e. A CLEAN workspace must cost NOTHING: this instrument copies a tree per
    #    dirty repo, and the whole workspace is re-classified on top.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = repo_at(ws, "r", CAPACITY_SRC, None)
        calls = []
        rep, delta, j = run(
            [repo], classify=lambda ps: calls.append(ps) or _REAL_CLASSIFY(ps))
        if len(calls) != 1:
            fails.append(f"adjudicate: re-classified a CLEAN workspace "
                         f"({len(calls)} calls) — nothing is dirty in SCOPE "
                         f"and the caller must SKIP")
        if delta.uncommitted or delta.masked or not delta.committed:
            fails.append(f"adjudicate: a clean workspace's rows are not all "
                         f"COMMITTED ({delta})")
    return fails


def adjudicate_arms() -> list[str]:
    """The cases above, plus the STUB PROBE proving they sit on the seam.

    ⛔ Aimed at `delta_of` — the helper `adjudicate` ACTUALLY calls — because
    two of this family's first three probes reported a false all-clear by
    being aimed at a function the target never invokes.
    """
    fails = adjudicate_cases()
    real = globals()["delta_of"]
    globals()["delta_of"] = lambda wt, hd, key: HeadDelta(list(wt), [], [])
    try:
        probed = adjudicate_cases()
    finally:
        globals()["delta_of"] = real
    if len(probed) < 3:
        fails.append(f"STUB PROBE: a `delta_of` that files every row as "
                     f"COMMITTED red only {len(probed)} of the provenance "
                     f"arms — they are not sitting under the seam")
    return fails


def buckets_by_repo(rep) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for b in rep.binds:
        per = out.setdefault(b.repo, {})
        per[b.verdict] = per.get(b.verdict, 0) + 1
    return out


def cross_repo_suppliers(rep, names: list[str]) -> list[str]:
    """Repos that could change ANOTHER repo's verdict if they went missing.

    A repo qualifies by owning kernels or bind sites, or by being cited as a
    CALLER inside some other repo's resolution. Derived from the run rather
    than typed, so a new cross-repo edge joins the stability check by existing.
    The citation scan is anchored on the derived repo NAMES (never free-form
    prose parsing): `<name>/` at a token boundary, after separator normalisation.
    """
    suppliers = {b.repo for b in rep.binds} | {k.repo for k in rep.kernels}
    for b in rep.binds:
        for token in norm(b.reason).replace(";", " ").split():
            head = token.split("/", 1)[0]
            if head in names and head != b.repo:
                suppliers.add(head)
    return sorted(suppliers)


def verdict_map(rep) -> dict[tuple[str, str, int, str], str]:
    return {(b.repo, norm(b.file), b.line, b.handle_expr): b.verdict
            for b in rep.binds}


def selftest() -> list[str]:
    """Pin the parsers, the path normalisation, the supplier derivation and the
    delegation reader — each fails silently otherwise, and a silent failure here
    reports a clean workspace.

    The classifier itself is NOT re-tested: `len_derived_binding_audit.selftest`
    runs on every `main()` of the report half and covers HALF A/B/C and the
    guard-only rule. What this sweep adds is the pin arithmetic around it, and
    that is what these arms are aimed at.
    """
    fails: list[str] = []
    bs = chr(92)  # a literal backslash, the separator the audit emits on Windows

    class _B:
        def __init__(self, repo, file, line, kernel, handle, verdict, reason=""):
            self.repo, self.file, self.line = repo, file, line
            self.kernel, self.handle_expr, self.verdict = kernel, handle, verdict
            self.reason = reason
            self.length_expr = "n"

    class _K:
        def __init__(self, repo, name):
            self.repo, self.name = repo, name

    class _R:
        def __init__(self, binds, kernels=()):
            self.binds, self.kernels = binds, list(kernels)

    # 1. the host separator must not reach a pin key.
    win = "crates" + bs + "riir-gpu" + bs + "src" + bs + "x.rs"
    b = _B("riir-ai", win, 12, "k", "h", EYES_BUCKET)
    if eyes_key(b) != ("riir-ai", "crates/riir-gpu/src/x.rs", "k", "h"):
        fails.append(f"path normalisation broke: {eyes_key(b)}")

    # 2. the EYES key is line-free ON PURPOSE, and two bind sites at one
    #    address must therefore COLLAPSE to a count of 2, not to a silent 1.
    two = _R([b, _B("riir-ai", win, 90, "k", "h", EYES_BUCKET)])
    seen: dict = {}
    for row in two.binds:
        seen[eyes_key(row)] = seen.get(eyes_key(row), 0) + 1
    if seen != {("riir-ai", "crates/riir-gpu/src/x.rs", "k", "h"): 2}:
        fails.append(f"eyes collapse lost a row: {seen}")

    # 3. bucket tallying is per repo, and an unseen bucket is ABSENT, not 0 —
    #    a verdict this sweep does not know about must not be silently pooled.
    rep = _R([_B("riir-ai", "a.rs", 1, "k", "h", "CAPACITY"),
              _B("riir-ai", "a.rs", 2, "k", "j", "UNRESOLVED"),
              _B("riir-train", "b.rs", 3, "m", "h", "GUARDED")])
    got = buckets_by_repo(rep)
    if got != {"riir-ai": {"CAPACITY": 1, "UNRESOLVED": 1},
               "riir-train": {"GUARDED": 1}}:
        fails.append(f"bucket tally wrong: {got}")

    # 4. supplier derivation, BOTH directions: a same-repo citation must not
    #    make a repo a supplier, a cross-repo one must.
    names = ["riir-ai", "riir-train", "riir-chain"]
    same = _R([_B("riir-ai", "a.rs", 1, "k", "h", "GUARD-ONLY",
                  "riir-ai/crates" + bs + "x" + bs + "b.rs:7 UNRESOLVED")])
    if cross_repo_suppliers(same, names) != ["riir-ai"]:
        fails.append("same-repo citation promoted to supplier: "
                     f"{cross_repo_suppliers(same, names)}")
    cross = _R([_B("riir-ai", "a.rs", 1, "k", "h", "GUARD-ONLY",
                   "riir-train/crates" + bs + "x" + bs + "b.rs:7 PERSISTENT-UPSTREAM; "
                   "riir-ai/crates" + bs + "x" + bs + "c.rs:9 UNRESOLVED")])
    if cross_repo_suppliers(cross, names) != ["riir-ai", "riir-train"]:
        fails.append("cross-repo citation missed: "
                     f"{cross_repo_suppliers(cross, names)}")
    # ... and a repo with kernels but no binds is still a supplier: it can own
    # the wrapper every other repo resolves THROUGH.
    kern = _R([], [_K("riir-chain", "k")])
    if cross_repo_suppliers(kern, names) != ["riir-chain"]:
        fails.append("kernel-only repo not a supplier: "
                     f"{cross_repo_suppliers(kern, names)}")

    # 5. verdict_map is the leave-one-out comparison key; it must be
    #    separator-independent too, or off-Windows every row reads as a flip.
    vm = verdict_map(_R([b]))
    if list(vm) != [("riir-ai", "crates/riir-gpu/src/x.rs", 12, "h")]:
        fails.append(f"verdict_map key wrong: {list(vm)}")

    # 6. the report half still exposes the shared entry point and its floors.
    for attr in ("classify_workspace", "FLOOR_RS_FILES", "FLOOR_KERNELS",
                 "derive_repos", "Report"):
        if not hasattr(lda, attr):
            fails.append(f"report half lost `{attr}` — the sweep shares its "
                         f"classifier and must not fall back to a copy")

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        # 7. pin parser: arity ENFORCED, comments stripped.
        p = ws / "pins.txt"
        p.write_text("# c\nrepo-a 23 78 0  # trailing\n\n", encoding="utf-8")
        if parse_pins(p) != {"repo-a": dict(zip(FIELDS, (23, 78, 0)))}:
            fails.append("pin parse: 4-field row not read correctly")
        p.write_text("repo-a 1 2\n", encoding="utf-8")
        try:
            parse_pins(p)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass

        # 8. eyes parser: 5 fields, arity enforced.
        e = ws / "eyes.txt"
        e.write_text("# c\nrepo-a src/x.rs kern handle 1\n", encoding="utf-8")
        if parse_eyes(e) != {("repo-a", "src/x.rs", "kern", "handle"): 1}:
            fails.append("eyes parse: 5-field row not read correctly")
        e.write_text("repo-a src/x.rs kern handle\n", encoding="utf-8")
        try:
            parse_eyes(e)
            fails.append("eyes parse: short row accepted")
        except ValueError:
            pass

        # 9. the DELEGATION reader must refuse a shape it cannot read rather
        #    than return {} — an empty dict turns the assertion into a no-op,
        #    which is precisely the failure it exists to prevent.
        d = ws / "delegated.txt"
        d.write_text("# h\nrepo-a 1500 4000 0\n", encoding="utf-8")
        if delegated_walk_floors(d) != {"repo-a": 1500}:
            fails.append("delegated walk floor not read")
        d.write_text("repo-a\n", encoding="utf-8")
        try:
            delegated_walk_floors(d)
            fails.append("delegated reader accepted a 1-field row")
        except ValueError:
            pass

    # Issue 822 T5j. In `selftest` and not behind `--canary`: `canary()`
    # deliberately STUBS `classify_workspace` because its arms are about pin
    # arithmetic, so a provenance arm living there would run against a fixed
    # report and assert nothing. These drive a real workspace through real git
    # and cost ~4s.
    return fails + adjudicate_arms()


def canary() -> int:
    """`--canary`: perturb each pin axis and REQUIRE the sweep to red.

    `selftest()` covers the helpers; this covers the thing the helpers are for.
    A pin nobody has watched fail is a pin that certifies nothing, and the arms
    here caught exactly that on their first run: the UNPINNED arm matched on
    `11\\n` where the pinned row ends `11             0`, so it perturbed
    nothing and its green was real. Only the expectation that it would RED made
    that visible — the same failure as Issue 775's `vendor/` arm, which
    certified the code path it was not aimed at.

    Opt-in rather than default (contrast `platform_dead_code_drift_sweep`'s
    `--prove-fires`): the arms re-enter `main()`, and a verdict that runs its
    own adversary on every invocation is one more thing between a reader and
    the answer. Every arm monkeypatches MODULE state — never a tracked file —
    so a failed arm cannot leave the worktree dirty.
    """
    import contextlib
    import copy
    import io

    global PINS, EYES, DELEGATED_WALK_PINS

    td = Path(tempfile.mkdtemp())
    pins_src = PINS.read_text(encoding="utf-8")
    eyes_src = EYES.read_text(encoding="utf-8")
    delg_src = DELEGATED_WALK_PINS.read_text(encoding="utf-8")
    real = (PINS, EYES, DELEGATED_WALK_PINS)

    # One classification, reused: every arm is about the PIN ARITHMETIC, not
    # about re-measuring the tree — 12 walks at ~8s would make the adversary
    # the slow half.
    base = lda.classify_workspace(lda.derive_repos(WORKSPACE))
    real_classify = lda.classify_workspace

    def run(argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(list(argv))
        return rc, buf.getvalue()

    def sub(text, old, new):
        if old not in text:
            raise AssertionError(f"canary anchor missing: {old!r}")
        return text.replace(old, new, 1)

    results = []

    def arm(name, want_rc, want_text, pins=None, eyes=None, delg=None,
            classify=None, argv=("--no-stability",), zero_ok=None):
        global PINS, EYES, DELEGATED_WALK_PINS
        PINS, EYES, DELEGATED_WALK_PINS = td / "p.txt", td / "e.txt", td / "d.txt"
        PINS.write_text(pins if pins is not None else pins_src, encoding="utf-8")
        EYES.write_text(eyes if eyes is not None else eyes_src, encoding="utf-8")
        DELEGATED_WALK_PINS.write_text(
            delg if delg is not None else delg_src, encoding="utf-8")
        lda.classify_workspace = classify or (lambda repos: base)
        real_zero = globals()["zero_walk_floor_accepted"]
        if zero_ok is not None:
            # Issue-902 wiring arms only: no real repo on any box is md-only
            # forever, so the acceptance side is stubbed and the predicate's
            # own arms live in sweep_population.selftest. Arm 8 below stays on
            # the REAL predicate — a zeroed row on a code repo must still red
            # through the measurement, which is the direction that must never
            # loosen.
            globals()["zero_walk_floor_accepted"] = (
                lambda repo: (True, "stub: 0 tracked .rs (canary)"))
        # ⛔ The Issue-822 provenance seam is STUBBED here, and the cost is why
        # (measured: the first run of this canary took >120s and was killed).
        # Every arm re-enters `main()`, so without this each one materialises a
        # HEAD checkout per dirty repo (~20s each) and re-runs the six
        # provenance arms — 12x work that measures nothing new, because these
        # arms are about PIN ARITHMETIC and the seam has its own arms in
        # `selftest()`, which a direct invocation runs once. AGENTS.md's
        # "budget the canary cost" rule, reached the expensive way.
        real_adj, real_arms = globals()["adjudicate"], globals()["adjudicate_arms"]
        globals()["adjudicate"] = lambda paths, rep, classify=None: (
            HeadDelta([(bind_key(b), b) for b in rep.binds], [], []), None)
        globals()["adjudicate_arms"] = lambda: []
        try:
            rc, out = run(argv)
        finally:
            PINS, EYES, DELEGATED_WALK_PINS = real
            lda.classify_workspace = real_classify
            globals()["zero_walk_floor_accepted"] = real_zero
            globals()["adjudicate"] = real_adj
            globals()["adjudicate_arms"] = real_arms
        ok = rc == want_rc and want_text in out
        print(f"  {'✓' if ok else '✗'} {name}  (rc={rc}, want {want_rc})")
        if not ok:
            print(f"        wanted: {want_text}")
            for ln in [l for l in out.splitlines() if l.startswith(("✗", "⛔"))][:4]:
                print(f"        got: {ln}")
        results.append(ok)

    # 0. the unperturbed baseline must be GREEN, or every red below is vacuous.
    arm("baseline green", 0, "len-derived sweep PASSED")

    # 1-2. the two parse floors, one per non-vacuous repo. (Arm 1's anchor
    #      moved with the 2026-09-26 riir-ai re-pin: 6 measured kernels,
    #      floor typed above it.)
    arm("min_kernels floor reds", 1, "parse FLOOR breached",
        pins=sub(pins_src, "riir-ai                          6         10",
                 "riir-ai                         99         10"))
    arm("min_binds floor reds", 1, "join FLOOR breached",
        pins=sub(pins_src, "riir-train                       5         11",
                 "riir-train                       5         22"))

    # 3. a repo the walk finds with no row can never red.
    arm("missing row reds UNPINNED", 1, "UNPINNED",
        pins=sub(pins_src,
                 "riir-train                       5         11             0\n", ""))

    # 4. the WALL. Relabelling a real bind through the SHARED classifier is the
    #    only honest way to plant the joined defect without writing Rust into a
    #    sibling repo this one is upstream of.
    def with_capacity(repos):
        rep = copy.deepcopy(base)
        for b in rep.binds:
            if b.repo == "riir-ai" and b.verdict == "UNRESOLVED":
                b.verdict = "CAPACITY"
                break
        return rep

    arm("max_findings WALL reds", 1, "joined finding(s) > pinned",
        classify=with_capacity)

    # 5-7. the EYES pin: membership in BOTH directions, then the count WITHIN
    #      one address — the half a membership set cannot see. (Anchors moved
    #      to the riir-infer addresses with the carve re-pin, 2026-09-26.)
    arm("new EYES address reds", 1, f"NEW {EYES_BUCKET}",
        eyes=sub(eyes_src,
                 "riir-infer crates/riir-infer-gpu/src/gemv_geglu_f16_cubecl.rs "
                 "gemv_geglu_plane_f16 weight_gate_handle 1\n", ""))
    arm("stale EYES row reds", 1, "no longer reported",
        eyes=eyes_src + "riir-infer crates/riir-infer-gpu/src/ghost.rs k h 1\n")
    arm("EYES count move reds", 1, "count at a pinned address moved",
        eyes=sub(eyes_src,
                 "gemv_geglu_plane_f16 weight_gate_handle 1",
                 "gemv_geglu_plane_f16 weight_gate_handle 2"))

    # 8-9. the delegated walk floor, and the reader that must REFUSE rather
    #      than return {} — an empty dict turns arm 8 green forever. Arm 8
    #      runs the REAL Issue-902 predicate: riir-ai has tracked Rust, so the
    #      measurement must refuse the zero row — the "code repo with a zeroed
    #      row reds" direction, live.
    arm("delegation break reds", 1, "walk floor DELEGATION broken",
        delg=sub(delg_src, "riir-ai                  1500           5000",
                 "riir-ai                     0           5000"))
    arm("unreadable delegation refused", 2, "unreadable", delg="riir-ai\n")

    # 8b. the measured md-only exception (Issue 902): a ZERO row on a repo
    #     that measures no Rust is ACCEPTED, and the acceptance PRINTS. The
    #     predicate is stubbed (see arm()'s zero_ok note); the un-stubbed
    #     half is arm 8 above, and sweep_population.selftest arms the
    #     predicate itself against synthetic trees.
    arm("md-only zero row accepted", 0, "zero walk floor ACCEPTED",
        delg=sub(delg_src, "riir-ai                  1500           5000",
                 "riir-ai                     0           5000"),
        zero_ok=True)

    # 10. an empty pins file is refused, never read as "nothing to check".
    arm("empty pins refused", 2, "declares NO repos", pins="# nothing\n")

    # 11. the cross-repo axis. The only arm that runs WITH stability, so it
    #     also proves the supplier derivation reaches the dropped repo.
    def cross(repos):
        rep = copy.deepcopy(base)
        if not any(p.name == "riir-train" for p in repos):
            for b in rep.binds:
                if b.repo == "riir-ai" and b.verdict == "UNRESOLVED":
                    b.verdict = "EXACT-UPSTREAM"
                    break
        return rep

    arm("cross-repo flip reds", 1, "CROSS-REPO VERDICT", classify=cross, argv=())

    print(f"\n{sum(results)}/{len(results)} canary arm(s) PASSED")
    return 0 if all(results) else 2


def main(argv: list[str]) -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior

    stability = "--no-stability" not in argv

    fails = selftest()
    if fails:
        print("✗ len-derived sweep SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    for path, what in ((PINS, "pins"), (EYES, "expected-EYES"),
                       (DELEGATED_WALK_PINS, "delegated walk floors")):
        if not path.is_file():
            print(f"✗ {what} file missing: {path}")
            return 2
    try:
        pins = parse_pins(PINS)
        eyes = parse_eyes(EYES)
        delegated = delegated_walk_floors(DELEGATED_WALK_PINS)
    except ValueError as e:
        print(f"✗ a pins file is unreadable: {e}")
        return 2
    if not pins:
        print("✗ pins file declares NO repos — an empty expectation set is refused")
        return 2

    names = derive_repo_names(WORKSPACE)
    if not names:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    # Two derivations of one population, asserted rather than assumed: the
    # report half has its own `derive_repos` (BOUNDARY.md + .git, returning
    # PATHS) and the sweep family uses `skill_repo_set_gate`'s (returning
    # names). They must agree, or the pins describe a different set than the
    # classifier measured.
    repo_paths = lda.derive_repos(WORKSPACE)
    if sorted(p.name for p in repo_paths) != sorted(names):
        print("✗ population disagreement: len_derived_binding_audit.derive_repos "
              f"sees {sorted(p.name for p in repo_paths)}, skill_repo_set_gate "
              f"sees {sorted(names)} — the pins would describe a different set "
              f"than the classifier measures")
        return 2

    rep = lda.classify_workspace(repo_paths)
    repo_paths_by_name = {p.name: p for p in repo_paths}
    # Issue 822 T5j — the DISPLAY reads the worktree (it is what the files say
    # today); every CEILING, every FLOOR and the EYES membership wall read what
    # a commit of these checkouts would produce. `jrep` falls back to the
    # worktree's own report where there is nothing dirty to compare against.
    delta, head_rep = adjudicate(repo_paths, rep)
    jrep = rep if head_rep is None else head_rep
    held = {k for k, _b in delta.uncommitted}
    per = buckets_by_repo(jrep)
    kernels_by_repo: dict[str, int] = {}
    for k in jrep.kernels:
        kernels_by_repo[k.repo] = kernels_by_repo.get(k.repo, 0) + 1
    wt_per = buckets_by_repo(rep)

    bad = False
    tot_find = tot_unres = tot_binds = 0
    seen_eyes: dict[tuple[str, str, str, str], int] = {}

    for name in names:
        bk = per.get(name, {})
        nk = kernels_by_repo.get(name, 0)
        nb = sum(bk.values())
        findings = sum(bk.get(v, 0) for v in FINDING_BUCKETS)
        unres = bk.get("UNRESOLVED", 0)
        tot_find += findings
        tot_unres += unres
        tot_binds += nb
        notes: list[str] = []

        # ⛔ The EYES membership wall reads HEAD too, and it is the DESTRUCTIVE
        # direction that makes it matter: a pinned row the worktree happens to
        # hide prints "no longer reported — drop the row", and dropping the pin
        # for a bind that is still committed deletes the only record of it.
        for b in jrep.binds:
            if b.repo == name and b.verdict == EYES_BUCKET:
                seen_eyes[eyes_key(b)] = seen_eyes.get(eyes_key(b), 0) + 1

        row = pins.get(name)
        flags = []
        if row is None:
            # Issue 821: an acknowledged known-extra owes no pin row —
            # the marker reached population_verdict's FINAL line and not
            # this loop, so 8 of 9 sweeps red on repos they found
            # nothing in, hiding two live ratchet breaches.
            if not pin_row_exempt(name):
                flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if nk < row["min_kernels"]:
                flags.append(f"parse FLOOR breached: {nk} `.len()`-deriving "
                             f"kernel(s) < {row['min_kernels']} — the walk is "
                             f"intact but HALF A matched less than it did")
            if nb < row["min_binds"]:
                flags.append(f"join FLOOR breached: {nb} bind site(s) < "
                             f"{row['min_binds']} — HALF B stopped finding "
                             f"launch sites for kernels HALF A still sees")
            if findings > row["max_findings"]:
                flags.append(
                    f"{findings} joined finding(s) > pinned {row['max_findings']} — "
                    + "; ".join(f"{v}={bk[v]}" for v in FINDING_BUCKETS if bk.get(v)))
            # The delegated axis, asserted (see the docstring). A repo pinned
            # here with no non-zero `.rs` walk floor anywhere has NO blindness
            # detector at all once its kernel/bind floors are 0 — unless the
            # repo has no Rust at all, in which case the zero row is truthful
            # and MEASURED every run (Issue 902), never a standing amnesty.
            if delegated.get(name, 0) <= 0:
                ok, note = zero_walk_floor_accepted(repo_paths_by_name[name])
                if ok:
                    notes.append(
                        f"zero walk floor ACCEPTED — {note}")
                else:
                    flags.append(
                        f"walk floor DELEGATION broken: this sweep carries no "
                        f"min_rs_files column because {DELEGATED_WALK_PINS.name} "
                        f"floors it, and that file has no non-zero row for "
                        f"{name} ({note})")

        status = "✗" if flags else ("·" if (findings or bk.get(EYES_BUCKET)) else "✓")
        wbk = wt_per.get(name, {})
        detail = " ".join(f"{v}={n}" for v, n in sorted(wbk.items())) or "—"
        moved = ([1 for k in held if k[0] == name]
                 + [1 for k, _b in delta.masked if k[0] == name])
        split = ""
        if moved:
            split = (f"  [{sum(1 for k in held if k[0] == name)} uncommitted"
                     + (f", {sum(1 for k, _b in delta.masked if k[0] == name)}"
                        f" MASKED" if any(k[0] == name for k, _b in delta.masked)
                        else "")
                     + f"; HEAD {detail_of(bk)}]")
        print(f"{status} {name:22s} kernels={nk:<3d} binds={nb:<4d} {detail}"
              f"{split}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")
        for n_line in notes:
            print(f"      · {name}: {n_line}")
        for v in FINDING_BUCKETS:
            for b in rep.binds:
                if b.repo == name and b.verdict == v:
                    wip = (" [UNCOMMITTED — not adjudicated]"
                           if bind_key(b) in held else "")
                    print(f"      ⛔ {v} {norm(b.file)}:{b.line}  {b.kernel}  "
                          f"{b.handle_expr}  [len: {b.length_expr}]{wip}")
        # A MASKED row is NOT in the worktree report — that is what MASKED
        # means — so it prints from the HEAD side or it prints nowhere, and a
        # ceiling reds over a bind site nobody can see.
        for k, b in sorted(delta.masked, key=lambda kb: kb[0]):
            if k[0] == name:
                print(f"      ⛔ {b.verdict} {norm(b.file)}:{b.line}  "
                      f"{b.kernel}  {b.handle_expr}  [MASKED — committed, "
                      f"hidden by this worktree]")

    # ── the EYES LIST, by MEMBERSHIP, in BOTH directions ────────────────────
    for key in sorted(set(seen_eyes) - set(eyes)):
        bad = True
        print(f"⛔ NEW {EYES_BUCKET}: {' '.join(key)} x{seen_eyes[key]} — not in "
              f"{EYES.name}. A bind whose declared size is a struct FIELD's: "
              f"read the field's creation before pinning it.")
    for key in sorted(set(eyes) - set(seen_eyes)):
        if key[0] not in names:
            continue  # its repo is absent; population_verdict owns that verdict
        bad = True
        print(f"✗ pinned {EYES_BUCKET} no longer reported: {' '.join(key)} — it "
              f"was adjudicated (drop the row in that commit) or the classifier "
              f"stopped reaching it")
    for key in sorted(set(eyes) & set(seen_eyes)):
        if seen_eyes[key] != eyes[key]:
            bad = True
            print(f"✗ {EYES_BUCKET} count at a pinned address moved: "
                  f"{' '.join(key)} {eyes[key]} -> {seen_eyes[key]}")

    # ── the cross-repo stability axis (see the docstring) ───────────────────
    if stability:
        base = verdict_map(rep)
        suppliers = cross_repo_suppliers(rep, names)
        flips_total = 0
        for drop in suppliers:
            subset = [p for p in repo_paths if p.name != drop]
            alt = verdict_map(lda.classify_workspace(subset))
            flips = [(k, base[k], alt[k]) for k in alt
                     if k in base and base[k] != alt[k]]
            flips_total += len(flips)
            for k, was, now in flips:
                bad = True
                print(f"⛔ CROSS-REPO VERDICT: dropping {drop} flips "
                      f"{k[0]}/{k[1]}:{k[2]} {k[3]}  {was} -> {now}. A verdict "
                      f"in a PRESENT repo now depends on an ABSENT one — "
                      f"DEFERRED does not cover this, and a partial-clone run "
                      f"would report a WRONG bucket, not a missing one.")
        print(f"  stability: {len(suppliers)} supplier repo(s) "
              f"({', '.join(suppliers)}) · {flips_total} verdict flip(s)")
    else:
        print("  stability: SKIPPED (--no-stability) — the cross-repo axis was "
              "not measured on this run")

    # The population axis, shared (Issues 793 + 782).
    pop_lines, deferred, pop_fail = population_verdict(pins, names)

    # Issue 797 — the worktree is not the repo. This run reads files that
    # concurrent sessions are editing, so a finding may sit on a line no
    # commit contains. ADVISORY, never a failure: a sweep that hard-reds on
    # an ordinary dirty worktree is a sweep nobody runs. It rides the FINAL
    # line in BOTH directions (the `deferred` precedent) and is SILENT
    # unless the dirty set meets this sweep's own population — kernels and bind sites.
    deferred.extend(sweep_advisory(
        names, SCOPE, root=WORKSPACE,
        uncommitted_rows=len(delta.uncommitted),
        masked_rows=len(delta.masked)))
    for _line in pop_lines:
        print(_line)
    if pop_fail:
        bad = True

    print(f"\n{len(names)} contract repo(s) · {rep.files_scanned} tracked .rs · "
          f"{len(rep.kernels)} `.len()`-deriving kernel(s) · {tot_binds} bind "
          f"site(s) · {tot_find} joined finding(s) · {len(seen_eyes)} EYES "
          f"address(es)")
    print(f"  UNRESOLVED={tot_unres} is REPORTED AND UNPINNED on purpose: the "
          f"bucket means 'provenance lives one level up and HALF C could not "
          f"reach the caller', so a ratchet on it is a backlog (Issue 785). It "
          f"is never folded into a pass or a fail.")
    print(f"  walk floor: delegated to {DELEGATED_WALK_PINS.name} (identical "
          f"tracked walk, identical population) and ASSERTED above, not assumed.")

    if bad:
        print("✗ len-derived sweep FAILED — see the ✗ rows above")
        for _d in deferred:
            print(f"  {deferral_line(_d)}")
        print("    A joined finding is HALF A (in-kernel `.len()` derivation) "
              "met by HALF B (a bind whose DECLARED size can exceed the live "
              "range). Read the bind site's buffer creation, not the kernel.")
        return 1
    _line = "✓ len-derived sweep PASSED — every repo within its pins"
    if deferred:
        _line += "; DEFERRED: " + "; ".join(deferred)
    print(_line)
    return 0


if __name__ == "__main__":
    if "--canary" in sys.argv[1:]:
        fails = selftest()
        if fails:
            print("✗ len-derived sweep SELFTEST FAILED — instrument untrustworthy:")
            for _f in fails:
                print(f"    {_f}")
            sys.exit(2)
        sys.exit(canary())
    sys.exit(main(sys.argv[1:]))
