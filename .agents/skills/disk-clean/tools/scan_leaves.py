import re, collections, sys

# Roll up an exact-attribution dump (as produced by scan_exact-style parsing, but
# attribute each per-directory File(s) line to the heading's full path) into
# per-COMPONENT-prefix sums for one target directory, printing the biggest leaves.
# usage: py scan_leaves.py <scan.txt> <root> [min_gb]

path, root = sys.argv[1], sys.argv[2]
min_gb = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

pat_dir = re.compile(r'^\s*Directory of (.*)\r?$')
pat_files = re.compile(r'^\s*\d+ File\(s\)\s+(\d+) bytes\r?$')

sizes = collections.Counter()
cur = None
for line in open(path, encoding='utf-8', errors='replace'):
    m = pat_dir.match(line)
    if m:
        cur = m.group(1).rstrip()
        continue
    m = pat_files.match(line)
    if m and cur:
        sizes[cur] += int(m.group(1))

root_prefix = root.rstrip('\\') + '\\'
leaf = collections.Counter()
n = len(root_prefix)
for p, v in sizes.items():
    if not p.startswith(root_prefix):
        continue
    rest = p[n:]
    parts = rest.split('\\')
    # credit each ancestor prefix AND the leaf itself
    for i in range(1, len(parts) + 1):
        leaf['\\'.join(parts[:i])] += v

rows = [(k, v) for k, v in leaf.items() if v >= min_gb * 1073741824 and k]
for k, v in sorted(rows, key=lambda kv: -kv[1])[:25]:
    print(f"{v/1073741824:8.2f} GB  {k}")
