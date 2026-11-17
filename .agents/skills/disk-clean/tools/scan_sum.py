import re, collections, sys

# Progressive `dir /s /-C` output summarizer: groups per-directory "N File(s) M bytes"
# lines under their "Directory of <prefix>\<rest>" heading, rolled up to `depth` path
# components below `prefix`. Lines seen before any heading are ignored (the root
# summary). Partial scans sum only the directories already listed.
#
# Usage: py scan_sum.py <scan.txt> [prefix] [depth]
#   prefix default: E:\git  — quote backslashes are literal in the dir output.

path = sys.argv[1] if len(sys.argv) > 1 else 'E:/git/scan_vendor.txt'
prefix = sys.argv[2] if len(sys.argv) > 2 else 'E:\\git'
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1

sizes = collections.OrderedDict()
cur = None
pat_dir = re.compile(re.escape('Directory of ' + prefix) + r'\\(.*)$')
pat_files = re.compile(r'^\s*\d+ File\(s\)\s+(\d+) bytes')
for line in open(path, encoding='utf-8', errors='replace'):
    m = pat_dir.search(line)
    if m:
        parts = m.group(1).split('\\')
        cur = '\\'.join(parts[:depth]) if parts[0] else '.'
        sizes.setdefault(cur, 0)
        continue
    m = pat_files.match(line)
    if m and cur:
        sizes[cur] += int(m.group(1))
total = 0
for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
    total += v
    print(f"{v/1073741824:8.2f} GB  {k}")
print(f"{total/1073741824:8.2f} GB  TOTAL")
