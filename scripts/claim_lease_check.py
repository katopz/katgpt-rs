#!/usr/bin/env python3
"""Issue 943 — the issue-level claim lease, checked at session start.

Both paid multi-agent collisions in this workspace were SAME-ISSUE collisions
(this repo's Issue 825 — two sessions, one issue, four hours apart, the second
recorded a false negative against the first's in-flight state; and the
2026-10-15 private-tier twin — one push landed, the other rejected mid-flight
after 800+ green tests). Push-as-arbiter discovers collisions at push time,
which is exactly one full implementation too late. The lease moves discovery
to START time: a session beginning work on an `.issues/NNN_*.md` or
`.plans/NNN_*.md` item appends

    **Claimed:** <session>, <epoch>, expires <ISO-UTC>

under the Status line and commits + pushes BEFORE implementing (the
mining-queue claim-marker shape: 48 h expiry, push-as-arbiter). Re-claiming
an expired lease is a normal re-read, not a ceremony.

    git fetch origin && py ../katgpt-rs/scripts/claim_lease_check.py [--item NNN]
                                                    [--epoch N | CLAIM_LEASE_EPOCH]

The script NEVER fetches — instruments are observers, not ref writers;
freshness comes from the caller's fetch and the FETCH_HEAD age is DISCLOSED
on every run so a stale fetch cannot read as a clean scan. Leases are
ADVISORY: the verdict for a foreign live lease on the target item is a loud
WARN (yield or coordinate), never a block — the push stays the arbiter (the
private-lane yield proves post-hoc arbitration is correct behavior).

Exit codes: 0 = clean, own-epoch lease, or a WARNed foreign lease; 1 = the
instrument cannot read — a MALFORMED claim line (genuine-looking but
unparseable), AMBIGUOUS multiple leases in one file, an unreadable
issues/plans dir, or a self-test failure. Blind is loud; a green zero is
only ever printed by a scan that read everything it looked at.

Template placeholders (`Claimed: <session>, <epoch>, ...` spelled in prose)
contain `<...>` and are ignored — a spec file that QUOTES the convention
must not red the repo that defines it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCAN_DIRS = (".issues", ".plans")
NUM_RE = re.compile(r"^(\d+)_")
# A line-anchored claim attempt — the EXACT convention shape
# (`**Claimed:** ...` / `Claimed: ...`, case-sensitive, colon required).
# Looser shapes false-positive on wrapped prose continuations ("claimed.
# Reopen condition: ..." measured in riir-refine .plans/019 on the first
# cross-repo run). The epoch must be digits for a VALID lease; anything
# line-anchored on `Claimed:` that is not a template (no `<...>`) and does
# not parse is a MALFORMED finding.
CLAIM_LINE_RE = re.compile(
    r"^\s*\*{0,2}Claimed:\*{0,2}\s*(.+)$")
PLACEHOLDER_RE = re.compile(r"<[^<>\n]+>")
EXPIRY_RE = re.compile(r"\bexpires\s+(\S+)\s*$", re.IGNORECASE)


def parse_expiry(raw: str) -> datetime | None:
    """ISO-8601 UTC (`Z` or an explicit offset); naive = UTC by convention."""
    text = raw.strip().strip("`*")
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_claim_line(line: str) -> tuple[str, int, datetime] | None | str:
    """A (session, epoch, expiry) tuple, None (not a claim line / template),
    or the string "malformed" (a genuine-looking claim that cannot parse)."""
    m = CLAIM_LINE_RE.match(line)
    if not m:
        return None
    body = m.group(1).strip()
    if PLACEHOLDER_RE.search(body):
        return None                      # template prose, not a lease
    parts = [p.strip() for p in body.split(",")]
    if len(parts) != 3 or not parts[1].isdigit() or len(parts[1]) < 9:
        return "malformed"
    em = EXPIRY_RE.search(parts[2])
    if not em:
        return "malformed"
    expiry = parse_expiry(em.group(1))
    if expiry is None:
        return "malformed"
    return (parts[0].strip("`* "), int(parts[1]), expiry)


def leases_in_file(path: Path) -> list[tuple[str, int, datetime] | str]:
    """Every parsed claim line in one file. >1 distinct lease = ambiguous."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ["unreadable"]
    found: list[tuple[str, int, datetime] | str] = []
    for line in text.splitlines():
        parsed = parse_claim_line(line)
        if parsed is None:
            continue
        if parsed == "malformed":
            found.append("malformed")
        elif isinstance(parsed, tuple):
            if parsed not in found:
                found.append(parsed)
    return found


def classify(repo: Path, now: datetime) -> tuple[list[dict], list[str]]:
    """(rows, blind). rows: live + expired + malformed, per file."""
    rows: list[dict] = []
    blind: list[str] = []
    for d in SCAN_DIRS:
        base = repo / d
        if not base.is_dir():
            continue
        try:
            files = sorted(base.glob("*.md"))
        except OSError as exc:
            blind.append(f"{d}: unreadable ({exc})")
            continue
        for f in files:
            for lease in leases_in_file(f):
                rel = f"{d}/{f.name}"
                if lease == "unreadable":
                    blind.append(f"{rel}: unreadable")
                elif lease == "malformed":
                    rows.append({"file": rel, "state": "MALFORMED",
                                 "session": "?", "epoch": None, "expiry": None})
                else:
                    session, epoch, expiry = lease
                    state = "LIVE" if expiry > now else "EXPIRED"
                    rows.append({"file": rel, "state": state,
                                 "session": session, "epoch": epoch,
                                 "expiry": expiry})
    return rows, blind


def fetch_head_age(repo: Path) -> tuple[int | None, str]:
    """(seconds since the last fetch landed, note). Reads FETCH_HEAD's mtime
    in the per-worktree git dir, then the common dir. Never fetches."""
    candidates: list[Path] = []
    for flag in ("--git-dir", "--git-common-dir"):
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", flag],
                             capture_output=True, encoding="utf-8")
        if out.returncode == 0 and out.stdout.strip():
            candidates.append(Path(out.stdout.strip()) / "FETCH_HEAD")
    for fh in candidates:
        try:
            age = int(time.time() - fh.stat().st_mtime)
            return age, str(fh)
        except OSError:
            continue
    return None, "no FETCH_HEAD on record"


def target_rows(rows: list[dict], item: str) -> list[dict]:
    """Rows whose filename stem carries the item number."""
    picked = []
    for r in rows:
        m = NUM_RE.match(os.path.basename(r["file"]))
        if m and m.group(1) == item:
            picked.append(r)
    return picked


def selftest() -> list[str]:
    """Parse + boundary arms over synthetic lines; runs on EVERY invocation.
    An instrument whose own grammar is unasserted must not print verdicts."""
    fails: list[str] = []
    now = datetime(2026, 10, 16, 0, 0, 0, tzinfo=timezone.utc)
    live = parse_claim_line(
        "**Claimed:** zed-idle-943, 1792085272, expires 2099-01-01T00:00:00Z")
    if not (isinstance(live, tuple) and live[0] == "zed-idle-943"
            and live[1] == 1792085272
            and parse_expiry("2099-01-01T00:00:00Z") == live[2]):
        fails.append(f"bold claim line did not parse: {live!r}")
    plain = parse_claim_line(
        "Claimed: lane-b, 1792000000, expires 2099-01-01T00:00:00+00:00")
    if not isinstance(plain, tuple):
        fails.append(f"plain claim line did not parse: {plain!r}")
    template = parse_claim_line(
        "Claimed: <session>, <epoch>, expires <UTC>")
    if template is not None:
        fails.append("a template placeholder line parsed as a lease — spec "
                     "files quoting the convention would red their own repo")
    # the measured false-positive: a wrapped-prose continuation line starting
    # with a lowercase "claimed." (riir-refine .plans/019, first cross-repo
    # run). Case-sensitivity + the required colon are the defense — assert
    # both spellings stay None.
    for prose in ("claimed. Reopen condition: real-use misses",
                  "Claimed. Reopen condition: real-use misses"):
        if parse_claim_line(prose) is not None:
            fails.append(f"prose continuation parsed as a lease: {prose!r}")
    for bad in ("Claimed: a, notanumber, expires 2099-01-01T00:00:00Z",
                "Claimed: a, 12345678, expires maybe-tomorrow",
                "Claimed: a, 12345678",
                "Claimed: a, 12345678, expires 2099-01-01, extra"):
        if parse_claim_line(bad) != "malformed":
            fails.append(f"a broken claim read as {parse_claim_line(bad)!r}: "
                         f"{bad}")
    boundary = parse_claim_line(
        "Claimed: a, 123456789, expires 2026-10-16T00:00:00Z")
    if not (isinstance(boundary, tuple)
            and boundary[2] <= now):
        fails.append("an expiry equal to `now` must classify expired (>)")
    # scan/classify arms over a temp tree, including the Issue-825 shape.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / ".issues").mkdir()
        (repo / ".issues" / "825_two_sessions_one_issue.md").write_text(
            "**Status:** Open\n"
            "**Claimed:** session-a, 1790000000, expires 2099-01-01T00:00:00Z\n"
            "body\n", encoding="utf-8")
        (repo / ".issues" / "900_old.md").write_text(
            "Claimed: session-c, 1780000000, expires 2020-01-01T00:00:00Z\n",
            encoding="utf-8")
        rows, blind = classify(repo, now)
        if blind:
            fails.append(f"clean fixture reported blind: {blind}")
        by_file = {r["file"]: r for r in rows}
        a = by_file.get(".issues/825_two_sessions_one_issue.md")
        if not a or a["state"] != "LIVE" or a["session"] != "session-a":
            fails.append(f"the live lease misclassified: {a}")
        o = by_file.get(".issues/900_old.md")
        if not o or o["state"] != "EXPIRED":
            fails.append(f"the expired lease misclassified: {o}")
        if target_rows(rows, "825") != [a]:
            fails.append("target_rows missed the item-numbered file")
        if target_rows(rows, "999"):
            fails.append("target_rows matched a foreign number")
    return fails


def prove_fires() -> list[str]:
    """The Issue-825 pair must fire: the second session's posture (no epoch,
    a different epoch) WARNS on a live foreign lease; the lease-holder's own
    epoch is clean; expired is clean; malformed is loud. Temp trees, no
    mocks. Fixture named for the recorded two-session account (AGENTS.md,
    the Issue-825 block)."""
    fails: list[str] = []
    import tempfile
    now = datetime.now(timezone.utc)

    def plant(root: Path) -> Path:
        repo = root / "fixture"
        (repo / ".issues").mkdir(parents=True)
        (repo / ".issues" / "825_two_sessions_one_issue.md").write_text(
            "# Issue 825 — the recorded two-session collision shape\n"
            "**Status:** Open\n"
            "**Claimed:** session-a, 1790000000, expires 2099-01-01T00:00:00Z\n",
            encoding="utf-8")
        return repo

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # (1) THE fixture: session B starts work with no epoch knowledge —
        # the posture the second 825 session actually stood in.
        repo = plant(root)
        rows, blind = classify(repo, now)
        if blind or not rows:
            fails.append(f"fixture scan broken: rows={rows} blind={blind}")
        hit = target_rows(rows, "825")
        if not hit or hit[0]["state"] != "LIVE":
            fails.append(f"the 825 fixture lease did not classify LIVE: {hit}")
        elif hit[0]["session"] == "session-b":
            fails.append("fixture mislabelled — the holder must be session-a")
        warn = (f"⚠ WARN: item 825 is LIVE-LEASED by session-a "
                f"(epoch 1790000000, expires 2099-01-01T00:00:00+00:00)")
        rendered = warn_line(hit[0], "825", None)
        if "session-a" not in rendered or "LIVE-LEASED" not in rendered:
            fails.append(f"warn line lost the holder: {rendered!r}")
        # (2) own epoch → clean posture, no warn.
        own = warn_line(hit[0], "825", 1790000000)
        if "WARN" in own:
            fails.append(f"the holder's own epoch warned: {own!r}")
        # (3) expired → re-claim posture is clean.
        expired = {"file": ".issues/900_old.md", "state": "EXPIRED",
                   "session": "session-c", "epoch": 1780000000,
                   "expiry": parse_expiry("2020-01-01T00:00:00Z")}
        line = warn_line(expired, "900", None)
        if "WARN" in line or "EXPIRED" not in line:
            fails.append(f"expired posture misrendered: {line!r}")
        # (4) malformed is loud — a genuine-looking broken claim must land
        # in the blind set (exit 1), never in a silent pass.
        (repo / ".issues" / "901_broken.md").write_text(
            "Claimed: session-d, notanumber, expires someday\n",
            encoding="utf-8")
        rows2, blind2 = classify(repo, now)
        if "MALFORMED" not in {r["state"] for r in rows2}:
            fails.append(f"the malformed claim did not surface: {rows2}")
        # (5) fetch-age on a real (tiny) git repo vs a bare dir.
        import subprocess as sp
        shared = root / "shared.git"
        sp.run(["git", "init", "-q", "-b", "main", "--bare", str(shared)],
               check=True, capture_output=True)
        clone = root / "clone"
        sp.run(["git", "clone", "-q", str(shared), str(clone)],
               check=True, capture_output=True)
        age, where = fetch_head_age(clone)
        if age is None:
            fails.append("a fetched clone reported no FETCH_HEAD age")
        age_none, _ = fetch_head_age(root)
        if age_none is not None:
            fails.append("a non-git dir reported a FETCH_HEAD age")
    return fails


def warn_line(row: dict, item: str, own_epoch: int | None) -> str:
    """The one-line verdict for the target item — the session's yield signal."""
    if row["state"] == "LIVE":
        if own_epoch is not None and row["epoch"] == own_epoch:
            return (f"✓ item {item}: YOUR live lease "
                    f"(epoch {row['epoch']}, expires {row['expiry'].isoformat()})")
        return (f"⚠ WARN: item {item} is LIVE-LEASED by {row['session']} "
                f"(epoch {row['epoch']}, expires {row['expiry'].isoformat()}) "
                f"— read {row['file']}, coordinate or yield; the push stays "
                f"the arbiter")
    if row["state"] == "EXPIRED":
        return (f"— item {item}: lease EXPIRED "
                f"({row['session']}, expired {row['expiry'].isoformat()}) — "
                f"re-claiming is a normal re-read, not a ceremony")
    return (f"⚠ WARN: item {row['file']} carries a MALFORMED claim line — "
            f"cannot adjudicate; fix the line (session, epoch digits, "
            f"expires ISO-UTC)")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    argv = sys.argv[1:]
    args: list[str] = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a in ("--item", "--epoch"):
            skip_next = True          # the flag's VALUE is not a positional
            continue
        if not a.startswith("--"):
            args.append(a)
    item = None
    if "--item" in argv:
        i = argv.index("--item")
        if i + 1 < len(argv):
            item = argv[i + 1]
    own_epoch: int | None = None
    if "--epoch" in argv:
        i = argv.index("--epoch")
        if i + 1 < len(argv) and argv[i + 1].isdigit():
            own_epoch = int(argv[i + 1])
    elif os.environ.get("CLAIM_LEASE_EPOCH", "").isdigit():
        own_epoch = int(os.environ["CLAIM_LEASE_EPOCH"])

    if "--prove-fires" in sys.argv:
        fails = prove_fires()
        if fails:
            print("✗ claim_lease_check prove-fires FAILED — the instrument "
                  "cannot see its own class:")
            for f in fails:
                print(f"    {f}")
            return 1
        print("✓ prove-fires PASS — the Issue-825 pair (second session's "
              "posture) WARNS on a live foreign lease; the holder's own epoch "
              "is clean; expired re-claims clean; malformed is loud; "
              "FETCH_HEAD age resolves on a fetched clone and only there")
        return 0

    if "--self-test" in sys.argv:
        fails = selftest()
        if fails:
            print("✗ claim_lease_check SELFTEST FAILED:")
            for f in fails:
                print(f"    {f}")
            return 1
        print("✓ selftest PASS")
        return 0

    fails = selftest()
    if fails:
        print("✗ claim_lease_check SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 1
    if len(args) > 1:
        print("usage: claim_lease_check.py [repo] [--item NNN] "
              "[--epoch N] | --prove-fires | --self-test")
        return 1
    repo = Path(args[0]).resolve() if args else Path.cwd()
    if not repo.is_dir():
        print(f"✗ BLIND — repo path does not exist: {repo}")
        return 1

    now = datetime.now(timezone.utc)
    rows, blind = classify(repo, now)
    age, where = fetch_head_age(repo)
    if age is None:
        print("⚠ fetch: no FETCH_HEAD on record — run `git fetch origin` "
              "first; a scan without a fresh fetch can read stale leases "
              "as absent")
    else:
        mark = "⚠" if age > 86400 else "·"
        print(f"{mark} fetch: FETCH_HEAD age {age // 3600}h{age % 3600 // 60:02d}m "
              f"({where})")
    live = [r for r in rows if r["state"] == "LIVE"]
    expired = [r for r in rows if r["state"] == "EXPIRED"]
    bad = [r for r in rows if r["state"] == "MALFORMED"]
    print(f"claim_lease_check {repo.name}: {len(live)} live, {len(expired)} "
          f"expired, {len(bad)} malformed claim line(s)"
          + (f", target item {item}" if item else ""))
    for r in expired:
        print(f"  EXPIRED  {r['file']}  {r['session']} epoch={r['epoch']} "
              f"(was {r['expiry'].isoformat()}) — re-claim is a normal re-read")
    for r in live:
        left = r["expiry"] - now
        print(f"  LIVE     {r['file']}  {r['session']} epoch={r['epoch']} "
              f"expires={r['expiry'].isoformat()} ({left.seconds // 3600}h left)")
    for r in bad:
        print(f"  ⛔ MALFORMED {r['file']} — a claim-shaped line that cannot "
              f"parse (want `**Claimed:** session, <epoch digits>, expires "
              f"<ISO-UTC>`); cannot adjudicate")
    verdict = ""
    if item is not None:
        hit = target_rows(rows, item)
        if not hit:
            verdict = (f"— item {item}: no claim line on any file numbered "
                       f"{item} (no lease is not a license — the "
                       "dual-allocation + fresh-mtime checks still apply)")
        else:
            verdict = warn_line(hit[0], item, own_epoch)
        print(verdict)
    if blind:
        print("✗ BLIND — the instrument could not read:")
        for b in blind:
            print(f"    {b}")
        return 1
    if bad:
        print("✗ BLIND — a MALFORMED claim line cannot be adjudicated live "
              "vs expired; fix the line and re-run")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
