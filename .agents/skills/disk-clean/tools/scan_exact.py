import re, collections, sys

# Path-exact attribution for progressive `dir /s /-C` scans: every "N File(s) M bytes"
# line is credited to the EXACT heading path it follows (no bucketing), then rolled
# up as prefix sums to `depth` components under `root`. Immune to the
# first-component-bucketing inconsistency the quick scan_sum.py rollup can show.
#
# Usage: py scan_exact.py <scan.txt> <root> <depth>

path = sys.argv[1]
root = sys.argv[2]
depth = int(sys.argv[3])

pat_dir = re.compile(r'^\s*Directory of (.*)\r?$')
pat_files = re.compile(r'^\s*\d+ File\(s\)\s+(\d+) bytes\r?$')

sizes = collections.Counter()
cur = None
root_prefix = root.rstrip('\\') + '\\'
for line in open(path, encoding='utf-8', errors='replace'):
    m = pat_dir.match(line)
    if m:
        cur = m.group(1).rstrip()
        continue
    m = pat_files.match(line)
    if m and cur:
        sizes[cur] += int(m.group(1))

# roll up to depth components under root
roll = collections.Counter()
for p, v in sizes.items():
    if p == root:
        roll['.'] += v
        continue
    if not p.startswith(root_prefix):
        roll['<<outside:' + p] += v
        continue
    rest = p[len(root_prefix):]
    parts = rest.split('\\')
    key = parts[0] if parts[0] else '.'
    if depth > 1:
        key = '\\'.join(parts[:depth]) or '.'
    roll[key] += v

total = 0
for k, v in sorted(roll.items(), key=lambda kv: -kv[1]):
    total += v
    print(f"{v/1073741824:8.2f} GB  {k}")
print(f"{total/1073741824:8.2f} GB  TOTAL ({len(sizes)} exact dirs)")
