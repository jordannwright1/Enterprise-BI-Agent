"""
python3 check_env.py — run on Streamlit Cloud via their terminal,
or add temporarily to main.py and read the Streamlit logs.
"""
import subprocess
import sys
import os

# 1. What playwright version is installed?
r = subprocess.run([sys.executable, "-m", "pip", "show", "playwright"],
                   capture_output=True, text=True)
print("=== playwright package ===")
print(r.stdout or r.stderr)

# 2. Is the binary actually there?
r2 = subprocess.run([sys.executable, "-m", "playwright", "install", "--dry-run"],
                    capture_output=True, text=True)
print("=== install dry-run ===")
print(r2.stdout or r2.stderr)

# 3. What does 'which chromium' say?
r3 = subprocess.run(["find", os.path.expanduser("~"), "-name",
                     "chrome-headless-shell", "-type", "f"],
                    capture_output=True, text=True, timeout=10)
print("=== binary locations ===")
print(r3.stdout or "(none found)")

# 4. Env vars that might be interfering
print("=== relevant env vars ===")
for k, v in os.environ.items():
    if any(x in k.upper() for x in ["PLAYWRIGHT", "CHROME", "BROWSER"]):
        print(f"  {k}={v}")
