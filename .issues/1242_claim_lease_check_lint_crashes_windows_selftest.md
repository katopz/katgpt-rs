# Issue 1242 — claim_lease_check.py `--lint` crashes on Windows: the mandatory selftest's nested subprocess returns stdout=None (py3.14 reader-thread death), blinding every Windows lint call

**Status:** OPEN — filed 2026-11-23 (zed-989683dc@4090, from the live reflex 132 lint calls during the A3-republish build).
**Claimed:** zed-b175c253@m3 (m3), 1795460263, expires 2026-11-25T18:57:43Z
The FILE verdicts are unaffected (the file scans parse fine — verified: the 132 file's claim line lints clean through the file path when the selftest is bypassed); the crash is the SELFTEST half, which every `--lint` invocation runs FIRST, so **every `--lint` call on this box dies before reading any file**. The lint works on amo/m3 (py3.10/3.13) — this is the Windows-py3.14 class.

## The crash (reproduced every invocation)

```
py ../katgpt-rs/scripts/claim_lease_check.py --lint .issues/132_...
Exception in thread Thread-1 (_readerthread):
  File "...\Python\pythoncore-3.14-64\Lib\threading.py", line 1082, in _bootstrap_inner
  ...
  File "...\Lib\subprocess.py", line 1614, in _readerthread
    buffer.append(fh.read())
sys.exit(main())
  File "...\scripts\claim_lease_check.py", line 786, in main
    fails = selftest()
  File "...\scripts\claim_lease_check.py", line 609, in selftest
    if refused.returncode != 1 or "REFUSED" not in refused.stdout:
TypeError: argument of type 'NoneType' is not a container or iterable
```

- `selftest()` → `sp.run([sys.executable, script, ...], capture_output=True, encoding="utf-8")` — the NESTED process's stdout comes back `None` (the reader thread dies: the `_readerthread` traceback above is the nested run's own spawn), and `selftest()` then reads `refused.stdout` unguarded → TypeError. Note the nested process ITSELF crashed with an internal `_readerthread` exception — the thread death class, not a plain exit.
- Identical under py3.10 (the system launcher) — the selftest's nested `sp.run` dies the same way, so it is NOT a py3.14-only regression; the selftest has never passed on this box.
- Amo/m3 are unaffected (their py builds run the selftest clean — every fleet lint call elsewhere works).

## Why it matters

- Every Windows-box `--lint` call dies → the lease scanner is UNREACHABLE from this box's lanes (the file verdicts parse fine when the call is reconstructed, but no Windows lane can run the tool's own lint gate at all).
- The blind-the-scanner hazard (the katgpt-rs Issue-1226 class) is the tool's own reason to exist — a tool whose lint gate cannot run on a fleet rig is the same failure one layer up.

## The fix sketch

- Guard `refused.stdout` for None in `selftest()` (loud `fails.append` naming the platform) — the minimal unblock; OR
- Skip the nested-process selftest when `sys.platform == "win32"` and the nested run's own spawn is the crasher, with a loud "selftest unavailable on this platform" verdict (never a silent pass); OR
- The real fix: find why the nested `sp.run` dies on this box (py3.14's `_readerthread` + Windows console encoding — plausibly the cp874/cp1252 codec the console hands back, the same class the fleet's other Windows scripts hit).

## Repro

Any `--lint <file>` call on shikuwa (Windows 11, py3.10.11 system + py3.14.6 launcher, Git Bash sh). The file verdicts themselves are irrelevant — the crash is in the selftest preamble.
