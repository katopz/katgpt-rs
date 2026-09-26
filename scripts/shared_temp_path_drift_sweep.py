#!/usr/bin/env python3
"""The cross-repo verdict half of `shared_temp_path_gate.py` — Issue 832 T3.

A test writing to a FIXED `std::env::temp_dir().join("literal")` path is safe
against its sibling tests inside one binary (each site has its own filename)
and is NOT safe against another PROCESS running the same test. `test_gate`,
`full_gate`, `x86_64_execution_matrix` and any hand-run `cargo test` each get
their own target dir and all share one `/tmp`. `create` truncates: A writes, B
truncates, A reads back zero bytes.

Why this exists at all
----------------------
The gate landed alone, and AGENTS.md records the cost of that shape **nine
times** (Issues 777, 778, 793, 782, 783, 789, 797, 820, 822): *a rule landed in
one instrument and never generalised*. The gate's own docstring named the
cross-repo axis as UNMEASURED and told the next reader not to carry
`check_validation_gate`'s "population of one, no sweep" answer across, because
`console_encoding_gate` assumed exactly that and was wrong by seven repos.

**Count first** was the instruction, and the count answers it: `scan()` already
took a repo path, so the question was answerable the whole time. Measured
2026-09-18 over the contract population on this box — 12 of 16 repos present —
**100 fixed-path sites, 98 of them outside this repo**, over 518
`env::temp_dir()` calls. Sampled for over-capture and it is not over-capture:
riir-ai's `go_bonsai_cache_test.bin` / `test_egl_roundtrip.bin` and friends are
`#[test]` bodies writing a fixed filename, which is byte-for-byte the shape that
produced this repo's own measured failures (five `katgpt-types::tests_types`
tests failing at once with *"File too small for header"* across five different
filenames).

Why the ceiling is a RATCHET and not a wall
-------------------------------------------
`max_fixed` is a ratchet on the derivative, per
`instrument_reachability_drift_sweep`'s answer and for its reason: 98 of the 100
rows are in ten repos this session does not own, and a wall would demand 98
repairs across ten trees in one change. AGENTS.md is explicit that a cross-repo
repair is not landed until it is COMMITTED in the sibling with a cited SHA
(Issue 798, measured: two tracked files recorded sibling repairs as landed and
green when two of five existed and both sweeps were red for six hours). A wall
here would make the next repo to join somebody's emergency, and a sweep that
always reds is a sweep nobody runs.

⚠ That is NOT a claim the 98 are acceptable. They are a real backlog, addressed
as Issue 832 T5 with the per-repo counts, and each repo's rows stay its owner's
to adjudicate. The ratchet's job is that the commit adding the NEXT one reds.

⛔ **katgpt-rs's own row does not restate its gate.** The gate walls this repo's
set at 0 UNPINNED by MEMBERSHIP, and *a pin that restates its own input cannot
fail*: a count comparison against a number this sweep derives from the same
`scan()` is true by construction. The sweep asserts the gate's **verdict** over
those rows instead (`numbering_drift_sweep`'s rule), so a stale membership pin
in `shared_temp_path_expected.txt` reds the sweep too.

Why there is NO `min_rs_files` column
-------------------------------------
`orphaned_attr`, `platform_dead_code` and `percentile` already floor this
identical `tracked_files(repo, "*.rs")` call over this identical population, and
a fourth copy is a fourth number to re-pin on ordinary churn. The delegation is
**asserted**, not assumed (`len_derived_drift_sweep`'s rule): every repo this
sweep pins must still carry a NON-ZERO `min_rs_files` row in the file it
delegates to, and a delegated file it cannot PARSE is UNREADABLE rather than an
empty dict — a silent empty dict turns the assertion into a no-op, which is
precisely the failure it exists to prevent.

⛔ One MEASURED exception (Issue 902): a repo born md-only — no tracked `.rs`
at all — has a TRUTHFUL zero row, and reding on it forever is the cries-wolf
state Issue 793 forbids. The zero row is accepted only while `tracked_walk`
(the ONE walk, Issue 777) measures ZERO tracked `.rs` in that repo, re-measured
EVERY run; the first `.rs` to land reds exactly as a zeroed row on a code repo
does. The acceptance prints, it is never silent. The predicate lives once, in
`sweep_population.zero_walk_floor_accepted`, so the two delegating sweeps can
never disagree about what a zero row MEANS.

The two floors that remain fail differently:

  min_temp_sites  the PREDICATE. A regex regression that stops matching
                  `env::temp_dir()` collapses the population to ~0 over an
                  unchanged walk, and then every ceiling passes vacuously.
  max_fixed       the RATCHET.

Head provenance (Issue 822)
---------------------------
The DISPLAY reads the worktree; the PINS read HEAD. Per-file row independence
holds here by construction — `sites()` is a pure function of ONE file's bytes —
so `head_delta` is the correct instrument and costs |dirty ∩ *.rs| `git show`
calls, zero on a clean run.

⛔ The row key is `(relpath, literal, ordinal)` and is LINE-FREE, so an edit
above a site does not report every row in the file as UNCOMMITTED *and* MASKED
at once. The ordinal is IN the key rather than a count beside it, because a
key-matched row is filed COMMITTED carrying the WORKTREE's object: any field the
key omits is one where the worktree silently overrides HEAD. A file that goes
from 2 sites to 3 therefore produces one UNCOMMITTED row, not a silent pass.

Usage
-----
    scripts/shared_temp_path_drift_sweep.py           # the verdict, every repo
    scripts/shared_temp_path_drift_sweep.py --canary  # the pin arithmetic arms
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import console_safe  # noqa: E402

console_safe.apply()

import shared_temp_path_gate as stp  # noqa: E402
from skill_repo_set_gate import derive_repos  # noqa: E402
from sweep_population import (  # noqa: E402
    open_repo, pin_row_exempt, population_verdict, zero_walk_floor_accepted)
from worktree_state import head_delta, sweep_advisory  # noqa: E402

# The repo that owns the membership pin — derived, never typed.
SELF = stp.REPO_ROOT.name

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "shared_temp_path_drift_floors.txt"
# The file whose `min_rs_files` column this sweep DELEGATES its walk floor to.
DELEGATED_WALK_PINS = HERE / "orphaned_attr_drift_floors.txt"

FIELDS = ("min_temp_sites", "max_fixed")

# This sweep's own population, as globs — passed to both `head_delta` and
# `sweep_advisory` from ONE constant so the two axes cannot disagree about what
# this sweep's population is (Issue 798's `_match_count` rule, one level up).
SCOPE = ("*.rs",)


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


def delegated_walk_floors(path: Path) -> dict[str, int]:
    """`min_rs_files` per repo from the file this sweep delegates that axis to.

    Parsed positionally and DEFENSIVELY: this sweep does not own that file's
    schema, so a shape it cannot read must be reported as UNREADABLE rather
    than silently yielding an empty dict — which would turn the delegation
    assertion into a no-op, precisely the failure it exists to prevent.
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


# ── the classifier ──────────────────────────────────────────────────────────


def expand(found: dict[str, dict[str, int]]) -> list[tuple[str, str, int]]:
    """`{rel: {lit: count}}` -> one LINE-FREE row per SITE.

    The ordinal is part of the address rather than a count carried beside it:
    see the module docstring on why a key must not omit a field the ceiling
    adjudicates.
    """
    rows: list[tuple[str, str, int]] = []
    for rel in sorted(found):
        for lit in sorted(found[rel]):
            for i in range(found[rel][lit]):
                rows.append((rel, lit, i))
    return rows


def classify(repo: Path) -> tuple[list[tuple[str, str, int]], int, int]:
    """(rows, env::temp_dir() sites, files carrying one) — the gate's closure."""
    found, n_temp, n_files = stp.scan(repo)
    return expand(found), n_temp, n_files


def head_rows(repo: Path):
    """`head_delta`'s per-file reclassifier for this sweep's ratchet.

    Issue 822. `classify` above reads the working tree, which in this workspace
    is shared with concurrent sessions, and the ratchet is a claim about the
    repo — whose state is its commits.
    """

    def rescan(rel: str, src: str | None) -> list[tuple[str, str, int]]:
        # None = absent from HEAD (a STAGED-but-never-committed file, Issue
        # 822's measured case): nothing committed to classify, so it yields no
        # row and the worktree's row is UNCOMMITTED.
        if src is None:
            return []
        counts = stp.sites(src)
        return [(rel, lit, i)
                for lit in sorted(counts)
                for i in range(counts[lit])]

    return rescan


def adjudicate(repo: Path, rows: list[tuple[str, str, int]]):
    """The worktree's rows, split COMMITTED / UNCOMMITTED / MASKED.

    A named seam rather than four lines inline, for the reason `arm_reach`
    keeps finding in this repo: verdict arithmetic that sits inside `main()`
    beside its own error messages is unreachable by construction.
    """
    return head_delta(
        repo, SCOPE, rows,
        lambda r: r[0],          # path_of — the repo-relative source file
        lambda r: r,             # key_of  — already line-free
        head_rows(repo))


# ── the arms ────────────────────────────────────────────────────────────────


def selftest() -> list[str]:
    """Arms on THIS module's own arithmetic.

    The classifier itself is armed in `shared_temp_path_gate.selftest()` and is
    not re-armed here — that is the delegation `check_validation_gate` credits.
    What no gate arm can reach is the pin arithmetic below.
    """
    import tempfile

    fails: list[str] = []

    def eq(label: str, got, want):
        if got != want:
            fails.append(f"{label}: got {got!r}, want {want!r}")

    # 1-3. expand(): one row per site, line-free, ordinal in the key.
    eq("expand single", expand({"a.rs": {"x": 1}}), [("a.rs", "x", 0)])
    eq("expand counts", expand({"a.rs": {"x": 2}}),
       [("a.rs", "x", 0), ("a.rs", "x", 1)])
    eq("expand sorted", expand({"b.rs": {"y": 1}, "a.rs": {"z": 1}}),
       [("a.rs", "z", 0), ("b.rs", "y", 0)])
    # A count CHANGE must move the key set, or the worktree silently overrides
    # HEAD on the one field the ceiling adjudicates.
    if set(expand({"a.rs": {"x": 2}})) == set(expand({"a.rs": {"x": 3}})):
        fails.append("expand: a count change did not move the key set")

    # 4-5. the HEAD reclassifier, including the staged-but-never-committed case.
    rescan = head_rows(Path("."))
    eq("rescan None -> no rows", rescan("a.rs", None), [])
    eq("rescan reads HEAD bytes",
       rescan("a.rs", 'let p = env::temp_dir().join("fixed");'),
       [("a.rs", "fixed", 0)])
    # A repaired site must produce NO row, or the sweep can never ratchet down.
    eq("rescan pid-suffixed is clean",
       rescan("a.rs",
              'env::temp_dir().join(format!("n_{}", std::process::id()))'),
       [])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 6-8. parse_pins: shape, comments, and a malformed row REFUSED.
        p = tmp / "pins.txt"
        p.write_text("# c\nrepo-a 10 3\n\nrepo-b 4 0\n", encoding="utf-8")
        eq("parse_pins", parse_pins(p),
           {"repo-a": {"min_temp_sites": 10, "max_fixed": 3},
            "repo-b": {"min_temp_sites": 4, "max_fixed": 0}})
        p.write_text("repo-a 10\n", encoding="utf-8")
        try:
            parse_pins(p)
            fails.append("parse_pins accepted a short row")
        except ValueError:
            pass

        # 9-10. the delegation reader must REFUSE rather than yield {} — a
        # silent empty dict turns the assertion into a no-op.
        d = tmp / "delegated.txt"
        d.write_text("# c\nrepo-a 1500 2 0\n", encoding="utf-8")
        eq("delegated walk floor read", delegated_walk_floors(d),
           {"repo-a": 1500})
        d.write_text("repo-a\n", encoding="utf-8")
        try:
            delegated_walk_floors(d)
            fails.append("delegated reader accepted a 1-field row")
        except ValueError:
            pass

    # 11. SELF is derived, never typed — the pin file is keyed on it.
    if SELF != REPO_ROOT.name:
        fails.append(f"SELF drifted: {SELF!r} != {REPO_ROOT.name!r}")

    # 12. SCOPE is a TUPLE, not a bare string. `("*.rs")` is not a tuple;
    # iterating it yields characters and `fnmatch(rel, "*")` matches
    # everything, so the advisory silently reports every dirty file in the
    # repo. Measured in worktree_state's own wiring commit: 8 of 15 call sites
    # had written it without the comma.
    if not isinstance(SCOPE, tuple) or not all(
            isinstance(s, str) and len(s) > 1 for s in SCOPE):
        fails.append(f"SCOPE is not a tuple of globs: {SCOPE!r}")

    return fails


def canary() -> int:
    """Arms over this sweep's PIN ARITHMETIC — the part no gate arm reaches.

    Each arm drives `main()` and requires a stated exit code and message.

    ⛔ **The derived population stays REAL and the CONTENT is stubbed**, and
    the split is not a convenience. `population_verdict` compares against the
    tracked `repo_set.txt` by design — *"a stubbed snapshot would test the
    stub"* — so an arm that invents repo names makes every real repo UNSEEN and
    the invented one UNREGISTERED, and every arm then reds for a reason that is
    not its subject. The first version of this canary did exactly that and all
    twelve arms failed on the same two lines.

    ⚠ The other half is AGENTS.md's budget rule: a canary that re-enters
    `main()` once per arm over every contract repo is how `len_derived`'s ran
    past 120s and was killed. `classify` here is a ~4-minute `*.rs` walk over
    16 repos, so it is stubbed; so is the provenance seam, which has its own
    arms in `worktree_state`. What is left is exactly the pin arithmetic.

    ⚠ It inherits this box's MARKERS exactly as the real run does
    (`DOCS_GATE_PARTIAL_CLONE`, `DOCS_GATE_KNOWN_EXTRA`) — AGENTS.md's stated
    rule for canaries that run the real workspace. A RED here on a known-subset
    box with no marker set is the environment, not the arithmetic.
    """
    import io
    import tempfile
    from contextlib import redirect_stdout

    g = globals()
    saved = {k: g[k] for k in
             ("classify", "adjudicate", "derive_repos", "self_verdict",
              "sweep_advisory", "PINS", "DELEGATED_WALK_PINS",
              "zero_walk_floor_accepted")}

    names = sorted(derive_repos(WORKSPACE))
    if not names:
        print(f"✗ canary cannot run — derived population is EMPTY under "
              f"{WORKSPACE}")
        return 2

    # The base files are generated FROM the derived population, so the base arm
    # is green for every repo the box actually carries and each arm perturbs
    # exactly one axis on top of it.
    base_pins = "".join(f"{n} 0 5\n" for n in names)
    base_delg = "".join(f"{n} 1500 2 0\n" for n in names)
    R = [("a.rs", "x", 0), ("a.rs", "x", 1)]

    fails: list[str] = []

    def arm(label: str, want_rc: int, want_txt: str, *,
            n_temp=99, pins=None, delg=None, repos=None, self_fails=None,
            zero_ok=None):
        g["classify"] = lambda repo: (list(R), n_temp, 9)
        # No git in the fixture: every row is COMMITTED, which is the arm's
        # subject. The three-way split has its own arms in `worktree_state`.
        g["adjudicate"] = lambda repo, rs: type(
            "D", (), {"head": list(rs), "uncommitted": [], "masked": []})()
        g["self_verdict"] = lambda repo: list(self_fails or [])
        g["sweep_advisory"] = lambda *a, **k: []
        if zero_ok is not None:
            # Issue-902 wiring arm only: the acceptance side is stubbed
            # because no real repo here is md-only. Arm 7 below stays on the
            # REAL predicate — a zeroed row on a code repo must still red
            # through the measurement — and the predicate's own arms live in
            # sweep_population.selftest.
            g["zero_walk_floor_accepted"] = (
                lambda repo: (True, "stub: 0 tracked .rs (canary)"))
        if repos is not None:
            g["derive_repos"] = lambda ws: list(repos)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pf, df = tmp / "pins.txt", tmp / "delg.txt"
            pf.write_text(base_pins if pins is None else pins,
                          encoding="utf-8")
            df.write_text(base_delg if delg is None else delg,
                          encoding="utf-8")
            g["PINS"], g["DELEGATED_WALK_PINS"] = pf, df
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main([], run_selftest=False)
            out = buf.getvalue()
        g["derive_repos"] = saved["derive_repos"]
        g["zero_walk_floor_accepted"] = saved["zero_walk_floor_accepted"]
        if rc != want_rc:
            fails.append(f"{label}: rc {rc} != {want_rc}\n{out}")
        elif want_txt not in out:
            fails.append(f"{label}: missing {want_txt!r}\n{out}")

    def repin(field: str, value: int, only: str | None = None) -> str:
        i = FIELDS.index(field)
        out = []
        for n in names:
            cols = ["0", "5"]
            if only is None or n == only:
                cols[i] = str(value)
            out.append(f"{n} {cols[0]} {cols[1]}\n")
        return "".join(out)

    n_arms = 0

    # 1. under the ratchet — a clean pass, and the base every arm perturbs.
    arm("under ratchet passes", 0, "PASSED"); n_arms += 1
    # 2. AT the ratchet is a pass; the ceiling is `>`, not `>=`.
    arm("at ratchet passes", 0, "PASSED", pins=repin("max_fixed", 2))
    n_arms += 1
    # 3. OVER the ratchet reds, and names the rows.
    arm("over ratchet reds", 1, "fixed 2 committed > pinned 1",
        pins=repin("max_fixed", 1)); n_arms += 1
    # 4. the PREDICATE floor — a regex regression over an unchanged walk.
    arm("predicate floor reds", 1, "population FLOOR breached",
        pins=repin("min_temp_sites", 999)); n_arms += 1
    # 5. an UNPINNED repo can never red, so its absence is itself the finding.
    arm("unpinned repo reds", 1, "UNPINNED",
        pins="".join(f"{n} 0 5\n" for n in names[1:])); n_arms += 1
    # 6. a pin for a repo the walk does not find. ⛔ The expected answer is a
    # PROPERTY OF THE BOX and the arm asks rather than assumes: without the
    # partial-clone marker an absent contract repo is UNSEEN and *never a
    # pass*; with it, the same absence rides the FINAL line as DEFERRED in
    # BOTH directions — which is the behaviour worth arming, because *a
    # deferral printed only on failure is one nobody reads on the run that
    # passes*. Hard-coding the unmarked answer made this the one failing arm
    # on this known-subset box, which is the environment, not the arithmetic.
    import os
    marked = os.environ.get("DOCS_GATE_PARTIAL_CLONE") == "1"
    arm("absent pin: DEFERRED with the marker, UNSEEN without",
        0 if marked else 1, "DEFERRED" if marked else "UNSEEN",
        pins=base_pins + "repo-zz 1 0\n")
    n_arms += 1
    # 7. the DELEGATION: a pinned repo with no non-zero walk floor there.
    #    Runs the REAL Issue-902 predicate — names[0] has tracked Rust, so
    #    the measurement must refuse the zero row ("code repo with a zeroed
    #    row reds", the direction that must never loosen).
    arm("delegation break reds", 1, "walk floor DELEGATION broken",
        delg=f"{names[0]} 0 2 0\n" + "".join(
            f"{n} 1500 2 0\n" for n in names[1:])); n_arms += 1
    # 7b. the measured md-only exception (Issue 902): a ZERO row on a repo
    #     that measures no Rust is ACCEPTED, and the acceptance PRINTS. The
    #     predicate is stubbed here (see arm()'s zero_ok note).
    arm("md-only zero row accepted", 0, "zero walk floor ACCEPTED",
        delg=f"{names[0]} 0 2 0\n" + "".join(
            f"{n} 1500 2 0\n" for n in names[1:]), zero_ok=True); n_arms += 1
    # 8. and a delegated file it cannot PARSE is refused, never an empty dict —
    # a silent {} turns arm 7's assertion into a no-op.
    arm("unreadable delegation refused", 2, "unreadable",
        delg=f"{names[0]}\n"); n_arms += 1
    # 9. an EMPTY pin file is refused — it would green every repo.
    arm("empty pins refused", 2, "declares NO repos", pins="# only\n")
    n_arms += 1
    # 10. an empty derived population is refused, never a green over zero.
    arm("empty population refused", 2, "derived population is EMPTY",
        repos=()); n_arms += 1
    # 11. SELF's row asserts the GATE's verdict, not a restated count — so a
    # stale membership row in shared_temp_path_expected.txt reds the sweep.
    arm("self gate verdict reds", 1, "own gate FAILS",
        self_fails=["✗ UNPINNED fixed temp path: q.rs::lit x1"]); n_arms += 1
    # 12. and is green when the gate is.
    arm("self gate verdict passes", 0, "PASSED"); n_arms += 1

    for k, v in saved.items():
        g[k] = v

    if fails:
        print("✗ canary FAILED — the pin arithmetic is not armed:")
        for f in fails:
            print(f"    {f}")
        return 1
    # COUNTED, never typed (Issue 798 T3: a hand-typed arm count was stale on
    # arrival in the module whose subject is records drifting from what they
    # describe).
    print(f"✓ canary — {n_arms} arm(s) over this sweep's own pin arithmetic, "
          f"over the {len(names)} derived repo(s)")
    return 0


# ── SELF's row: assert the gate's VERDICT, never a restated count ───────────


def self_verdict(repo: Path) -> list[str]:
    """`shared_temp_path_gate`'s own verdict over this repo.

    A count comparison would be true by construction — this sweep and that gate
    call the same `scan()` — and *a pin that restates its own input cannot
    fail*. Asserting the verdict instead means a stale membership row in
    `shared_temp_path_expected.txt` reds the sweep too.
    """
    pins = stp.parse_pins(stp.PINS)
    fails, _n_fixed, _n_temp, _n_files = stp.verdict(repo, pins)
    return fails


def main(argv: list[str], run_selftest: bool = True) -> int:
    if "--canary" in argv:
        return canary()

    fails = selftest() if run_selftest else []
    if fails:
        print("✗ shared-temp-path sweep SELFTEST FAILED — untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    if not PINS.is_file():
        print(f"✗ pins file missing: {PINS}")
        return 2
    try:
        pins = parse_pins(PINS)
    except ValueError as e:
        print(f"✗ pins file unreadable: {e}")
        return 2
    if not pins:
        print("✗ pins file declares NO repos — an empty expectation set is "
              "refused")
        return 2
    try:
        delegated = delegated_walk_floors(DELEGATED_WALK_PINS)
    except (OSError, ValueError) as e:
        print(f"✗ delegated walk-floor file unreadable: {e} — the delegation "
              f"assertion would silently become a no-op")
        return 2

    names = derive_repos(WORKSPACE)
    if not names:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    bad = False
    tot_fixed = tot_temp = tot_files = 0
    n_uncommitted = n_masked = 0

    for name in sorted(names):
        repo = open_repo(name, WORKSPACE)
        rows, n_temp, n_files = classify(repo)

        # ── Issue 822: the DISPLAY reads the worktree, the PINS read HEAD ───
        delta = adjudicate(repo, rows)
        n_uncommitted += len(delta.uncommitted)
        n_masked += len(delta.masked)
        tot_fixed += len(rows)
        tot_temp += n_temp
        tot_files += n_files

        row = pins.get(name)
        flags: list[str] = []
        notes: list[str] = []
        if row is None:
            # Issue 821: an acknowledged known-extra owes no pin row.
            if not pin_row_exempt(name):
                flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if n_temp < row["min_temp_sites"]:
                flags.append(
                    f"population FLOOR breached: {n_temp} env::temp_dir() "
                    f"site(s) < {row['min_temp_sites']} — the PREDICATE went "
                    f"blind over an unchanged walk, and then every ceiling "
                    f"passes vacuously")
            # The delegated axis, asserted (see the module docstring). The
            # zero row is MEASURED (Issue 902): accepted only while the repo
            # has no tracked Rust at all, never a standing amnesty.
            if delegated.get(name, 0) <= 0:
                ok, note = zero_walk_floor_accepted(repo)
                if ok:
                    notes.append(f"zero walk floor ACCEPTED — {note}")
                else:
                    flags.append(
                        f"walk floor DELEGATION broken: {name} has no non-zero "
                        f"min_rs_files row in {DELEGATED_WALK_PINS.name} — this "
                        f"sweep carries no walk floor of its own and now has "
                        f"none ({note})")
            if len(delta.head) > row["max_fixed"]:
                flags.append(
                    f"fixed {len(delta.head)} committed > pinned "
                    f"{row['max_fixed']} — a NEW test writing to a shared "
                    f"temp path; repair it with "
                    f"join(format!(\"...{{}}\", std::process::id()))")
                held = {r for r in delta.uncommitted}
                masked = {r for r in delta.masked}
                for r in sorted(set(delta.head) | set(rows)):
                    tag = ""
                    if r in held:
                        tag = "  [UNCOMMITTED — not adjudicated]"
                    elif r in masked:
                        tag = "  [MASKED — committed, and this worktree hides it]"
                    print(f"      ⛔ {name}/{r[0]}::{r[1]}{tag}")
            if name == SELF:
                # The membership wall, asserted through the gate itself.
                gf = self_verdict(repo)
                if gf:
                    flags.append(
                        f"own gate FAILS over {len(gf)} row(s) — "
                        f"shared_temp_path_gate.py disagrees with its pins")
                    for f in gf:
                        print(f"      ⛔ {SELF}: {f}")

        status = "✗" if flags else ("·" if rows else "✓")
        split = ""
        if delta.uncommitted or delta.masked:
            split = (f" ({len(delta.head)} committed"
                     + (f" + {len(delta.uncommitted)} uncommitted"
                        if delta.uncommitted else "")
                     + (f", {len(delta.masked)} MASKED"
                        if delta.masked else "") + ")")
        print(f"{status} {name:22s} temp_dir={n_temp:<4d} files={n_files:<4d} "
              f"fixed={len(rows)}{split}")
        for f in flags:
            bad = True
            print(f"    ⛔ {f}")
        for n_line in notes:
            print(f"    · {name}: {n_line}")

    lines, deferred, pv_fail = population_verdict(set(pins), set(names))
    for line in lines:
        print(line)
    if pv_fail:
        bad = True

    advisory = sweep_advisory(
        sorted(names), SCOPE, WORKSPACE,
        uncommitted_rows=n_uncommitted, masked_rows=n_masked)

    print(f"\nscope: fixed `env::temp_dir().join(\"literal\")` sites over "
          f"{len(names)} repo(s) — {tot_fixed} site(s), {tot_temp} "
          f"env::temp_dir() call(s), {tot_files} file(s)")
    print("       STATED blind spots (shared_temp_path_gate's own): a "
          "temp_dir() bound to a variable before the .join; other "
          "fixed-scratch spellings (PathBuf::from(\"/tmp/…\"))")

    tail = "".join(f"  [{d}]" for d in deferred + advisory)
    if bad:
        print(f"✗ shared-temp-path sweep FAILED{tail}")
        return 1
    print(f"✓ shared-temp-path sweep PASSED — every repo within its "
          f"pins{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
