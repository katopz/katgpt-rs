#!/usr/bin/env python3
"""A `path = "../X"` dependency on a repo NOBODY HAS — Issue 835.

riir-clippy `fbf698e6` added `riir-llm = { path = "../riir-llm", optional =
true }`. That repo existed on exactly one box. Every cargo command in
riir-clippy then died at manifest load:

    failed to load source for dependency `riir-llm`
      Caused by: failed to read `E:\\git\\riir-llm\\Cargo.toml`

⛔ **`optional = true` does not save you.** Cargo resolves path dependencies to
read their package metadata regardless of whether any feature enables them, so
this is not a feature you can switch off — the crate is simply unbuildable,
including `cargo check`, `cargo test` and `cargo fmt`.

## Why nothing caught it, which is the actual finding

The workspace's population checks partition repos two ways:

| state | who notices |
|---|---|
| on disk, absent from `repo_set.txt` | UNREGISTERED — reds in every posture |
| in `repo_set.txt`, absent from disk | UNSEEN, or DEFERRED under `DOCS_GATE_PARTIAL_CLONE` |
| **neither** | **nothing** |

Every predicate either derives from the WALK or compares walk-against-file, so
a repo in neither set is not a bucket anybody is lenient about — **it is not a
bucket at all.** Meanwhile it breaks the build of a repo that IS registered.

⚠ The REPAIR path was fully instrumented; only DISCOVERY was not. When the
repo was cloned and registered, `agents_repo_set_gate` immediately went to
26/27 on two independent signals (a `repo_set.txt` row AGENTS.md did not name,
and a count mismatch) — and said in its own message that a matching count
would prove nothing. This gate closes the other end.

## The verdicts

- **RESOLVED** — the target directory exists.
- **DEFERRED** — the target's repo is absent from this box but PRESENT in
  `repo_set.txt`, and `DOCS_GATE_PARTIAL_CLONE=1` is set. The seven canonical
  repos this workstation does not carry are exactly this case, and without the
  distinction the gate would be unrunnable on any partial box.
- ⛔ **ORPHAN** — the target's repo is absent from disk **and** absent from
  `repo_set.txt`. The class. Walled at **0**.
- **EXCLUDED** — an ORPHAN whose citing manifest sits under a directory its
  repo's ROOT `[workspace] exclude` lists (seal-remake Issue 129: the
  scenario-DAG view crate path-deps seal-game-editor, a repo the contract does
  not claim). Cargo never loads an excluded crate's manifest while resolving
  the workspace, so the CITING repo stays buildable — only that one crate,
  built by `--manifest-path`, needs the sibling. Reported and counted on the
  PASS line, never walled: the exclusion is explicit and reviewable in the
  root manifest, which is the opposite of the silent class this gate exists
  for.
- ⛔ **BROKEN-SUBPATH** — the repo is present but the crate directory under it
  is not. An ordinary broken dep, walled at 0 too; kept as its own verdict
  because the remedy is completely different (fix the path, not clone a repo).

`DOCS_GATE_CI=1` defers the whole cross-repo axis: a single-checkout CI job
would call every dep unresolvable and print a confident red.

## What it does NOT claim

That a resolvable dep is CORRECT — only that something is there. Version
skew, a stale sibling, and a dep pointing at the wrong crate all resolve.
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import console_safe  # noqa: E402
import repo_alias  # noqa: E402
import skill_repo_set_gate  # noqa: E402
from tracked_walk import tracked_files  # noqa: E402

console_safe.apply()

# Floors. Two, because they break differently: MIN_MANIFESTS catches the walk
# going blind, MIN_DEPS catches the regex going blind over an intact walk.
# Either one reports "0 ORPHAN", which is byte-identical to a clean workspace.
MIN_MANIFESTS = 40
MIN_DEPS = 60

# `path = "../x"` — at least one parent hop, so an intra-repo `path = "crates/x"`
# and a workspace-relative one are out of scope by construction.
PATH_DEP = re.compile(r'path\s*=\s*"((?:\.\./)+[^"]*)"')


def strip_toml_comments(text: str) -> str:
    """`text` with every TOML comment removed — a `#` outside a basic
    (`"…"`, backslash escapes) or literal (`'…'`) string ends its line.

    A commented-out dependency is not a dependency: cargo never resolves it.
    Measured 2026-10-23: riir-ai `crates/riir-games/Cargo.toml` carries
    `# seal-client-data = { path = "../../../seal-online-remaster/…" }`
    (TEMP-ISSUE-027, deliberately disabled) and this gate read it as an
    ORPHAN on every workstation — a red no repair could clear but deleting
    the record of why the line is off. Multi-line strings do not occur in
    the manifests' dependency tables and are not modelled.
    """
    out = []
    for line in text.split("\n"):
        quote = None
        i = 0
        while i < len(line):
            ch = line[i]
            if quote == '"' and ch == "\\":
                i += 2
                continue
            if quote is None and ch == "#":
                line = line[:i]
                break
            if ch in ('"', "'"):
                quote = None if quote == ch else (ch if quote is None else quote)
            i += 1
        out.append(line)
    return "\n".join(out)


WS_EXCLUDE = re.compile(r'^\s*exclude\s*=\s*\[(.*?)\]', re.S | re.M)


def workspace_excludes(root_manifest_text: str) -> tuple:
    """The `[workspace] exclude` entries of a repo's ROOT Cargo.toml, as
    normalized relative paths (no leading `./`, no trailing `/`).

    Only the `[workspace]` table is read — an `exclude` key under
    `[package]` is cargo's publish-file filter, a different thing entirely,
    and reading it would excuse ORPHANs nobody excluded from anything.
    """
    text = strip_toml_comments(root_manifest_text)
    out = []
    table = None
    body = []
    for line in text.split("\n"):
        head = line.strip()
        if head.startswith("[") and not head.startswith("[["):
            if table == "workspace":
                break
            table = head.strip("[] ")
            continue
        if table == "workspace":
            body.append(line)
    m = WS_EXCLUDE.search("\n".join(body))
    if m:
        for q in re.findall(r'"([^"]*)"', m.group(1)):
            q = q.strip().removeprefix("./").rstrip("/")
            if q:
                out.append(q)
    return tuple(out)


def is_excluded(rel_manifest: str, excludes: tuple) -> bool:
    """Does the manifest at repo-relative `rel_manifest` sit under an
    excluded directory? Component-wise, so `crates/a` never excludes
    `crates/ab/Cargo.toml`."""
    parent = rel_manifest.replace("\\", "/").rsplit("/", 1)[0] if "/" in rel_manifest else ""
    return any(parent == e or parent.startswith(e + "/") for e in excludes)


def repo_set(root: Path) -> set:
    """The canonical registry. Empty set when unreadable — the caller decides."""
    f = root / "scripts" / "repo_set.txt"
    try:
        text = f.read_text(encoding="utf-8")
    except OSError:
        return set()
    return {ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}


def workspace_repos(ws: Path) -> set:  # population-predicate: not a contract-repo walk
    """The contract repos on disk — DELEGATED, not re-derived.

    This started as an eleventh independent `BOUNDARY.md + .git` walk and
    `population_sync_gate` refused it, correctly. Delegating did not just
    satisfy a registry: the local copy tested `(p / ".git").exists()`, and the
    canonical predicate tests `.is_dir()` — because a `git worktree` has a
    `.git` **FILE**, so the local copy counted this box's `riir-chain.w152`
    worktree as a repo and would have attributed its manifests to a repo that
    does not exist. Two predicates that disagree about the population are the
    thing that gate exists to catch, and it caught this one.

    Names come back as CONTRACT spellings, so they are mapped back through
    `repo_alias.disk()` — this gate OPENS these directories, and a box whose
    on-disk names differ needs the on-disk half.
    """
    try:
        return {repo_alias.disk(n) for n in skill_repo_set_gate.derive_repos(ws)}
    except Exception:
        return set()


def target_repo_name(ws: Path, target: Path):
    """Which sibling repo does `target` land in? `None` if it escapes the workspace.

    Resolved through the WORKSPACE root rather than by counting `../` hops: a
    dep may point at a crate several levels down (`../katgpt-rs/crates/x`), and
    the repo is the first path component under the workspace, not the whole.
    """
    try:
        rel = target.relative_to(ws)
    except ValueError:
        return None
    parts = rel.parts
    return parts[0] if parts else None


def classify(ws: Path, own: Path, manifest: Path, raw: str, on_disk: set, registered: set,
             partial: bool, exists=None) -> tuple:
    """`(verdict, repo_name|None)` for one `path = "../…"` dependency.

    `exists` is the directory probe, INJECTABLE. It defaults to the live
    filesystem, which is what production wants — but a known-answer validation
    has to be able to rewind to a workspace state that no longer exists, and
    with a hard-coded `Path.exists` the probe silently overrides every injected
    set. Measured: rewinding `on_disk`/`registry` to the state before riir-llm
    was cloned reported **0 ORPHAN**, because the directory was there by then.
    An oracle that cannot be pointed at the known answer is not an oracle.
    """
    if exists is None:
        def exists(p):
            return p.exists()
    # Resolve BOTH sides. Mixing a resolved target with an unresolved
    # workspace root makes `relative_to` raise on Windows, where resolving is
    # what attaches the drive letter — and the failure reads as
    # OUT-OF-WORKSPACE, i.e. every finding silently disappears.
    ws = ws.resolve()
    target = (manifest.parent / raw).resolve()
    name = target_repo_name(ws, target)
    if name is None:
        # Points outside the workspace entirely. Not this gate's class, and
        # guessing would invent a verdict for a layout nobody here controls.
        return ("OUT-OF-WORKSPACE", None)
    try:
        if target.relative_to(own.resolve()) is not None:
            return ("INTRA-REPO", name)
    except ValueError:
        pass
    if exists(target):
        return ("RESOLVED", name)
    if name in on_disk:
        # The repo is here; the crate directory under it is not.
        return ("BROKEN-SUBPATH", name)
    if name in registered:
        return (("DEFERRED" if partial else "ORPHAN"), name)
    return ("ORPHAN", name)


def scan(root: str, ws=None, partial=None, repos_on_disk=None, registry=None,
         exists=None) -> dict:
    """Classify every cross-repo path dep reachable from `root`'s siblings.

    Every environment input is INJECTABLE, because the arms cannot assert a
    verdict that is read straight out of `os.environ` and the live filesystem.
    """
    root = Path(root).resolve()
    ws = Path(ws) if ws else root.parent
    if partial is None:
        partial = os.environ.get("DOCS_GATE_PARTIAL_CLONE") == "1"
    on_disk = workspace_repos(ws) if repos_on_disk is None else set(repos_on_disk)
    registered = repo_set(root) if registry is None else set(registry)

    rows = {}
    n_manifests = n_deps = 0
    for repo_name in sorted(on_disk):
        repo = ws / repo_name
        try:
            # `*.toml` then filter, NOT a bare "Cargo.toml" pathspec: git
            # matches that at the repo ROOT only, which silently reduced the
            # population to one manifest per repo — 18 instead of 700+, with
            # every crate-level dependency table unread. The floor caught it.
            candidates, _ = tracked_files(str(repo), "*.toml")
        except Exception:
            continue
        try:
            excludes = workspace_excludes(
                (repo / "Cargo.toml").read_text(encoding="utf-8", errors="replace"))
        except OSError:
            excludes = ()
        for m in candidates:
            m = Path(m)
            if m.name != "Cargo.toml":
                continue
            n_manifests += 1
            try:
                text = m.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for hit in PATH_DEP.finditer(strip_toml_comments(text)):
                raw = hit.group(1)
                verdict, name = classify(ws, repo, m, raw, on_disk, registered,
                                         partial, exists)
                if verdict in ("INTRA-REPO", "OUT-OF-WORKSPACE"):
                    continue
                n_deps += 1
                rel = str(m.relative_to(repo)).replace("\\", "/")
                if verdict == "ORPHAN" and is_excluded(rel, excludes):
                    verdict = "EXCLUDED"
                rows.setdefault(verdict, []).append((repo_name, rel, raw, name))
    return {"rows": rows, "manifests": n_manifests, "deps": n_deps,
            "on_disk": on_disk, "registered": registered, "partial": partial}


def floor_verdict(manifests: int, deps: int) -> tuple:
    """`(exit_code, message|None)` — EXTRACTED so an arm can reach it."""
    if manifests < MIN_MANIFESTS:
        return (2, f"⛔ WALK REGRESSION — {manifests} tracked Cargo.toml below floor "
                   f"{MIN_MANIFESTS}; a 0-ORPHAN verdict here is a green ZERO.")
    if deps < MIN_DEPS:
        return (2, f"⛔ PREDICATE REGRESSION — {deps} cross-repo path dep(s) below "
                   f"floor {MIN_DEPS}; the path regex is blind.")
    return (0, None)


def selftest() -> list:
    """Arms. Every environment input is injected, so these assert RULES."""
    fails = []
    ws = Path("/ws")
    own = Path("/ws/riir-clippy")
    man = Path("/ws/riir-clippy/Cargo.toml")

    def c(raw, on_disk, registered, partial=False, exists=()):
        real = set(str(Path(e).resolve()) for e in exists)
        orig = Path.exists

        def patched(self):
            return str(self) in real or Path(str(self)).as_posix() in exists
        Path.exists = patched
        try:
            return classify(ws, own, man, raw, set(on_disk), set(registered), partial)[0]
        finally:
            Path.exists = orig

    # 1. THE CLASS: absent from disk AND from the registry.
    if c("../riir-llm", {"riir-clippy"}, {"riir-clippy"}) != "ORPHAN":
        fails.append("a dep on a repo in neither set must be ORPHAN")

    # 2. Absent but REGISTERED, on a partial box -> DEFERRED. Without this the
    #    gate is unrunnable on the 7-repo-short workstation, which is the same
    #    as not existing.
    if c("../riir-dao", {"riir-clippy"}, {"riir-clippy", "riir-dao"}, partial=True) != "DEFERRED":
        fails.append("absent-but-registered under PARTIAL_CLONE must DEFER")

    # 3. ...and WITHOUT the marker it is still a finding. A deferral that
    #    applies unconditionally is not a deferral, it is an exemption.
    if c("../riir-dao", {"riir-clippy"}, {"riir-clippy", "riir-dao"}, partial=False) != "ORPHAN":
        fails.append("absent-but-registered with NO marker must not silently pass")

    # 4. The marker must not launder the real class. This is the arm that
    #    matters: a partial box must still red on a repo nobody has registered.
    if c("../riir-llm", {"riir-clippy"}, {"riir-clippy"}, partial=True) != "ORPHAN":
        fails.append("PARTIAL_CLONE must NOT excuse an unregistered target")

    # 5. Present repo, missing crate dir -> its own verdict, not ORPHAN. The
    #    remedy differs completely (fix the path vs clone a repo).
    if c("../riir-ai/crates/gone", {"riir-clippy", "riir-ai"}, {"riir-clippy", "riir-ai"}) \
            != "BROKEN-SUBPATH":
        fails.append("a missing subpath under a PRESENT repo must be BROKEN-SUBPATH")

    # 6. A dep that resolves is RESOLVED, and the repo name is taken from the
    #    workspace root rather than by counting `../` hops — a dep can point
    #    several levels into a sibling.
    if c("../katgpt-rs/crates/katgpt-core", {"riir-clippy", "katgpt-rs"},
         {"riir-clippy", "katgpt-rs"}, exists=("/ws/katgpt-rs/crates/katgpt-core",)) != "RESOLVED":
        fails.append("a deep but existing sibling path must be RESOLVED")
    if target_repo_name(ws, Path("/ws/katgpt-rs/crates/katgpt-core")) != "katgpt-rs":
        fails.append("target_repo_name must return the repo, not the crate")
    if target_repo_name(ws, Path("/elsewhere/x")) is not None:
        fails.append("a target outside the workspace must be None")

    # 7. The regex. `../` is required, so intra-repo and workspace-relative
    #    paths are out of scope by construction rather than by a skip list.
    got = PATH_DEP.findall('a = { path = "../x" }\nb = { path = "crates/y" }\n'
                           'c = { path = "../../z" }')
    if got != ["../x", "../../z"]:
        fails.append(f"PATH_DEP matched {got}")

    # 7b. Comments are not dependencies — but a `#` INSIDE a string is not a
    #     comment, and a trailing comment after a real dep keeps the dep.
    got = PATH_DEP.findall(strip_toml_comments(
        '# a = { path = "../off" }\n'
        'b = { path = "../on" }  # trailing note\n'
        "c = { path = \"../x#y\", note = 'q#r' }\n"
        '    #d = { path = "../indented_off" }\n'
        'e = { path = "../e", note = "a\\"#b" } # f = { path = "../after" }'))
    if got != ["../on", "../x#y", "../e"]:
        fails.append(f"comment stripping read {got}")

    # 8. Floors, both directions, distinguishable causes, and non-trivial.
    if floor_verdict(MIN_MANIFESTS, MIN_DEPS)[0] != 0:
        fails.append("floors must pass exactly at the boundary")
    if floor_verdict(MIN_MANIFESTS - 1, MIN_DEPS)[0] != 2:
        fails.append("a manifest walk below floor must exit 2")
    if floor_verdict(MIN_MANIFESTS, MIN_DEPS - 1)[0] != 2:
        fails.append("a dep count below floor must exit 2")
    if "WALK" not in (floor_verdict(0, MIN_DEPS)[1] or ""):
        fails.append("the walk floor must name itself")
    if "PREDICATE" not in (floor_verdict(MIN_MANIFESTS, 0)[1] or ""):
        fails.append("the predicate floor must name itself")
    if MIN_MANIFESTS <= 0 or MIN_DEPS <= 0:
        fails.append("a floor of 0 is not a floor")

    # 9. repo_set reader: comments and blanks are not repos. A `#` row counted
    #    as a repo would silently REGISTER whatever the comment mentions.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "scripts").mkdir()
        (Path(d) / "scripts" / "repo_set.txt").write_text(
            "# a comment\n\nriir-ai\n  riir-chain  \n", encoding="utf-8")
        got = repo_set(Path(d))
        if got != {"riir-ai", "riir-chain"}:
            fails.append(f"repo_set() read {got}")
    if repo_set(Path("/definitely/not/here")) != set():
        fails.append("an unreadable registry must read as empty, not raise")

    # 9b. The [workspace] exclude reader (seal-remake Issue 129). Only the
    #     WORKSPACE table counts — a [package] exclude is a publish filter —
    #     and the match is component-wise.
    got = workspace_excludes(
        '[package]\nexclude = ["pkg-only"]\n'
        '[workspace]\nmembers = ["crates/a"]\n'
        '# exclude = ["commented"]\n'
        'exclude = [\n  "./crates/view/",\n  "tools/x",  # note\n]\n'
        '[dependencies]\nexclude = ["dep-table"]\n')
    if got != ("crates/view", "tools/x"):
        fails.append(f"workspace_excludes read {got}")
    if workspace_excludes('[package]\nname = "x"\n') != ():
        fails.append("a manifest with no [workspace] must exclude nothing")
    if not is_excluded("crates/view/Cargo.toml", ("crates/view",)):
        fails.append("a manifest AT an excluded dir must be excluded")
    if not is_excluded("crates/view/sub/Cargo.toml", ("crates/view",)):
        fails.append("a manifest UNDER an excluded dir must be excluded")
    if is_excluded("crates/viewer/Cargo.toml", ("crates/view",)):
        fails.append("prefix match must be component-wise (crates/view vs crates/viewer)")
    if is_excluded("Cargo.toml", ("crates/view",)):
        fails.append("the ROOT manifest is never excluded")

    # 10. ⛔ `scan()` ITSELF, over a real fixture workspace (Issue 837).
    #     Arms 1-9 assert `classify()`, which is the classifier; `scan()` is
    #     the VERDICT layer above it and had NO arm at all, so its three
    #     decisions — the env default, the manifest-name filter and the
    #     not-a-finding filter — were unreached. That is the split
    #     `arm_reach_audit` records for every module it measured: the
    #     classifier well armed and the verdict not.
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        for name in ("alpha", "beta"):
            r = ws / name
            (r / "crates" / "inner").mkdir(parents=True)
            for a in (("init", "-q", "-b", "main"),
                      ("config", "user.email", "t@t"), ("config", "user.name", "t")):
                subprocess.run(["git", "-C", str(r), *a], check=True,
                               capture_output=True)
        a = ws / "alpha"
        # a cross-repo dep (counts), an intra-repo one (must NOT count), and a
        # non-manifest .toml carrying a path dep (must not be read at all).
        (a / "Cargo.toml").write_text(
            '[dependencies]\nbeta = { path = "../beta" }\n', encoding="utf-8")
        (a / "crates" / "inner" / "Cargo.toml").write_text(
            '[dependencies]\nsib = { path = "../../crates/inner" }\n', encoding="utf-8")
        (a / "crates" / "inner" / "Other.toml").write_text(
            '[dependencies]\nghost = { path = "../../../beta" }\n', encoding="utf-8")
        subprocess.run(["git", "-C", str(a), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(a), "commit", "-qm", "b"], check=True,
                       capture_output=True)

        # Issue 129 fixture: a SECOND repo whose root excludes crates/view,
        # and that excluded crate path-deps a repo in neither set; plus a
        # MEMBER crate with the same dep, which must stay a finding.
        g = ws / "gamma"
        (g / "crates" / "view").mkdir(parents=True)
        (g / "crates" / "member").mkdir(parents=True)
        for a2 in (("init", "-q", "-b", "main"),
                   ("config", "user.email", "t@t"), ("config", "user.name", "t")):
            subprocess.run(["git", "-C", str(g), *a2], check=True, capture_output=True)
        (g / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["crates/member"]\nexclude = ["crates/view"]\n',
            encoding="utf-8")
        (g / "crates" / "view" / "Cargo.toml").write_text(
            '[dependencies]\ned = { path = "../../../nobody/crates/ed" }\n', encoding="utf-8")
        (g / "crates" / "member" / "Cargo.toml").write_text(
            '[dependencies]\ned = { path = "../../../nobody/crates/ed" }\n', encoding="utf-8")
        subprocess.run(["git", "-C", str(g), "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(g), "commit", "-qm", "g"], check=True,
                       capture_output=True)
        gx = scan(str(a), ws=ws, partial=False,
                  repos_on_disk={"gamma"}, registry={"gamma"})
        exc = gx["rows"].get("EXCLUDED", [])
        orph = gx["rows"].get("ORPHAN", [])
        if [r[1] for r in exc] != ["crates/view/Cargo.toml"]:
            fails.append(f"an ORPHAN under a [workspace] exclude dir must be EXCLUDED: {gx['rows']}")
        if [r[1] for r in orph] != ["crates/member/Cargo.toml"]:
            fails.append(f"the same ORPHAN in a MEMBER crate must stay ORPHAN: {gx['rows']}")

        got = scan(str(a), ws=ws, partial=False,
                   repos_on_disk={"alpha", "beta"}, registry={"alpha", "beta"})
        if got["manifests"] != 2:
            fails.append(f"scan counted {got['manifests']} manifest(s), want 2 "
                         f"— `Other.toml` is not a manifest and a name filter "
                         f"that admits it reads dependency tables nobody declared")
        if got["deps"] != 1:
            fails.append(f"scan counted {got['deps']} dep(s), want 1 — the "
                         f"INTRA-REPO row must not enter the population, or "
                         f"the floor is satisfied by deps that cannot fail")
        # ⛔ UNCONDITIONAL. The first version of this guarded the check on
        # `if got["rows"].get("RESOLVED")`, and the mutant that inverts the
        # not-a-finding filter empties exactly that list — so the assertion
        # deleted itself on the input it existed to catch, and the mutant
        # survived with the count unchanged (1 INTRA-REPO instead of 1
        # RESOLVED). A guarded assertion is no assertion.
        resolved = got["rows"].get("RESOLVED", [])
        if len(resolved) != 1 or resolved[0][3] != "beta":
            fails.append(f"scan did not report exactly one RESOLVED dep on "
                         f"beta: {got['rows']}")
        if set(got["rows"]) - {"RESOLVED"}:
            fails.append(f"INTRA-REPO / OUT-OF-WORKSPACE reached the row set "
                         f"({sorted(got['rows'])}) — they are not findings, "
                         f"and counting them inflates every bucket")

        # the env DEFAULT, both arms — `partial=None` is the production path,
        # and it is read from os.environ, so only an injected environment can
        # assert it.
        prev = os.environ.get("DOCS_GATE_PARTIAL_CLONE")
        try:
            os.environ["DOCS_GATE_PARTIAL_CLONE"] = "1"
            if not scan(str(a), ws=ws, repos_on_disk={"alpha", "beta"},
                        registry={"alpha", "beta"})["partial"]:
                fails.append("DOCS_GATE_PARTIAL_CLONE=1 did not reach scan()")
            os.environ["DOCS_GATE_PARTIAL_CLONE"] = "0"
            if scan(str(a), ws=ws, repos_on_disk={"alpha", "beta"},
                    registry={"alpha", "beta"})["partial"]:
                fails.append("DOCS_GATE_PARTIAL_CLONE=0 read as SET — any "
                             "non-empty value would then excuse an ORPHAN")
        finally:
            if prev is None:
                os.environ.pop("DOCS_GATE_PARTIAL_CLONE", None)
            else:
                os.environ["DOCS_GATE_PARTIAL_CLONE"] = prev

    return fails


def main(argv) -> int:
    fails = selftest()
    if fails:
        print("⛔ cross-repo-path-dep gate SELFTEST FAILED:")
        for f in fails:
            print(f"    {f}")
        return 2

    if "--prove-fires" in argv:
        # Known-answer validation against the state this gate was written for:
        # riir-clippy depends on ../riir-llm and that repo is neither on disk
        # nor in repo_set.txt. Opt-in, because it rewinds the workspace rather
        # than reading it (the platform_dead_code_floor_gate precedent).
        root = Path(".").resolve()
        ws = root.parent
        gone = (ws / "riir-llm").resolve()
        r = scan(str(root), ws=ws, partial=True,
                 repos_on_disk=workspace_repos(ws) - {"riir-llm"},
                 registry=repo_set(root) - {"riir-llm"},
                 exists=lambda p: p.exists() and gone not in (p,) + tuple(p.parents))
        orphans = r["rows"].get("ORPHAN", [])
        hit = [o for o in orphans if o[3] == "riir-llm"]
        for o in orphans:
            print(f"    ⛔ ORPHAN  {o[0]}/{o[1]}  ->  {o[2]}")
        if not hit:
            print("✗ --prove-fires FAILED — the known ORPHAN was not reported. "
                  "A ceiling nobody has watched fail is not a ceiling.")
            return 1
        print(f"    ✓ --prove-fires PASSED — the riir-llm ORPHAN is reported "
              f"({len(hit)} row(s)) over {r['deps']} cross-repo dep(s)")
        return 0

    if os.environ.get("DOCS_GATE_CI") == "1":
        print("    ✓ cross-repo path-dep gate — CROSS-REPO AXIS DEFERRED (DOCS_GATE_CI=1): "
              "a single checkout has no siblings, so every dep would read unresolvable. "
              "Selftest green; this line is not an adjudication")
        return 0

    root = argv[1] if len(argv) > 1 else "."
    r = scan(root)
    rows = r["rows"]
    orphan = rows.get("ORPHAN", [])
    broken = rows.get("BROKEN-SUBPATH", [])
    deferred = rows.get("DEFERRED", [])
    excluded = rows.get("EXCLUDED", [])
    for repo, man, raw, name in excluded:
        print(f"    ⚠ EXCLUDED  {repo}/{man}  ->  {raw}   (ORPHAN in a [workspace] "
              f"exclude crate — the citing workspace loads; this crate alone needs the sibling)")

    for verdict, label in (("ORPHAN", "no repo on disk AND no repo_set.txt row"),
                           ("BROKEN-SUBPATH", "repo present, crate directory missing")):
        for repo, man, raw, name in rows.get(verdict, []):
            print(f"    ⛔ {verdict}  {repo}/{man}  ->  {raw}   ({label})")

    code, msg = floor_verdict(r["manifests"], r["deps"])
    if code:
        print("\n" + msg)
        return code

    if orphan or broken:
        print(f"\n✗ cross-repo path-dep gate FAILED — {len(orphan)} ORPHAN, "
              f"{len(broken)} BROKEN-SUBPATH over {r['deps']} cross-repo dep(s) "
              f"in {r['manifests']} tracked Cargo.toml. An ORPHAN target makes the "
              f"CITING repo unbuildable outright: cargo resolves path deps even "
              f"when `optional = true`.")
        return 1

    # The bucket census rides the PASS line in BOTH directions. A deferral
    # printed only when it is non-empty is one nobody reads on the run that
    # passes — this repo's rule for the sweep family's DEFERRED verdict — and
    # here the silent direction is worse than usual: with every dep RESOLVED,
    # `0 ORPHAN, 0 BROKEN-SUBPATH` says nothing whatever about the
    # absent-but-REGISTERED distinction the whole design rests on, and a reader
    # will assume otherwise. A bucket with no live cases is also the one a
    # later "simplification" deletes.
    if deferred:
        names = sorted({n for _, _, _, n in deferred})
        extra = (f"  [{len(deferred)} dep(s) on {len(names)} absent-but-REGISTERED "
                 f"repo(s) ({', '.join(names)}) — DEFERRED by "
                 f"DOCS_GATE_PARTIAL_CLONE=1, NOT measured by this run]")
    else:
        absent = sorted(set(r["registered"]) - set(r["on_disk"]))
        extra = (f"  [⚠ DEFERRED 0 — no dep on this box targets any of the "
                 f"{len(absent)} absent-but-REGISTERED repo(s), so the "
                 f"absent-registered vs absent-unregistered split this gate rests "
                 f"on is UNEXERCISED by live data and is asserted only by its arms. "
                 f"A green line here is not evidence that path works]")
    print(f"    ✓ cross-repo path-dep gate PASSED — 0 ORPHAN, 0 BROKEN-SUBPATH over "
          f"{r['deps']} cross-repo path dep(s) (floor {MIN_DEPS}) in {r['manifests']} "
          f"tracked Cargo.toml (floor {MIN_MANIFESTS}){extra}"
          f"  [{len(excluded)} EXCLUDED — ORPHAN deps of [workspace]-excluded crates, "
          f"reported, never walled]. "
          f"⚠ It does NOT claim a resolvable dep is CORRECT — version skew, a stale "
          f"sibling, and a dep pointing at the wrong crate all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
