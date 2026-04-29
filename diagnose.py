"""
Run from project root: python3 fix_pw_refs.py
Removes every hardcoded pw-browsers / PLAYWRIGHT_BROWSERS_PATH / executable_path
reference from main.py and core/graph.py, and inserts the single correct setup.
"""
import re
import os
import shutil
from datetime import datetime

# ── Backup first ─────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
for f in ["main.py", "core/graph.py"]:
    if os.path.exists(f):
        shutil.copy(f, f + f".bak_{ts}")
        print(f"✅ Backed up {f} → {f}.bak_{ts}")

# ── Patterns to nuke ──────────────────────────────────────────────────────────
# Any line containing these strings gets removed entirely
KILL_PATTERNS = [
    "pw-browsers",
    "PLAYWRIGHT_BROWSERS_PATH",
    "base_pw_path",
    "get_executable_path",
    "executable_path=",
    "playwright_bins",
]

def purge_lines(path: str) -> int:
    with open(path) as f:
        lines = f.readlines()

    kept = []
    removed = 0
    for line in lines:
        if any(p in line for p in KILL_PATTERNS):
            print(f"  REMOVE [{path}]: {line.rstrip()}")
            removed += 1
        else:
            kept.append(line)

    with open(path, "w") as f:
        f.writelines(kept)

    return removed

print("\n── Purging main.py ──")
n = purge_lines("main.py")
print(f"   Removed {n} line(s)")

print("\n── Purging core/graph.py ──")
n = purge_lines("core/graph.py")
print(f"   Removed {n} line(s)")

# ── Now add the single correct setup to the TOP of main.py ───────────────────
CORRECT_SETUP = '''\
# ── Playwright browser setup (single source of truth) ────────────────────────
# Do NOT set PLAYWRIGHT_BROWSERS_PATH anywhere else in this codebase.
# sync_playwright will find the system-installed Chromium automatically.
import subprocess, sys as _sys
subprocess.run(
    [_sys.executable, "-m", "playwright", "install", "chromium"],
    capture_output=True,   # suppress install chatter in Streamlit logs
)
# ─────────────────────────────────────────────────────────────────────────────
'''

with open("main.py") as f:
    content = f.read()

if "playwright install" not in content:
    # Insert after the last import block (first non-import/non-comment line)
    lines = content.split("\n")
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "#", '"""', "'''")):
            insert_at = i + 1
        elif stripped == "":
            continue
        else:
            break  # first real code line
    lines.insert(insert_at, CORRECT_SETUP)
    with open("main.py", "w") as f:
        f.write("\n".join(lines))
    print("\n✅ Added single playwright install block to main.py")
else:
    print("\n⏭️  playwright install already in main.py — skipping insert")

# ── Verify graph.py launch block uses no executable_path ─────────────────────
print("\n── Verifying core/graph.py launch block ──")
with open("core/graph.py") as f:
    graph = f.read()

remaining = [p for p in KILL_PATTERNS if p in graph]
if remaining:
    print(f"⚠️  Still found in graph.py: {remaining}")
    print("   These are inside strings or comments — check manually.")
else:
    print("✅ core/graph.py is clean")

# ── Check the launch() call looks right ──────────────────────────────────────
if "p.chromium.launch(" in graph or "pw.chromium.launch(" in graph:
    print("✅ chromium.launch() call found")
    if "executable_path" not in graph:
        print("✅ No executable_path — Playwright will auto-locate binary")
    else:
        print("⚠️  executable_path still present — remove it manually")
else:
    print("ℹ️  No launch() call found — may be in universal_scraper")

print("\n── Done. Restart your Streamlit app. ──")
