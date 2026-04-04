"""Run all validation scripts in order, stopping on first failure."""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
SCRIPTS = sorted(SCRIPTS_DIR.glob("[0-9][0-9]_*.py"))

# Skip training/eval in the quick run — those take time
QUICK_STOP = 6  # run 01-06, skip 07-08

mode = "quick" if "--quick" in sys.argv else "full"
limit = QUICK_STOP if mode == "quick" else len(SCRIPTS)

print(f"=== Running validation ({mode} mode) ===\n")

for script in SCRIPTS[:limit]:
    print(f"--- {script.name} ---")
    result = subprocess.run([sys.executable, str(script)], cwd=str(SCRIPTS_DIR.parent))
    if result.returncode != 0:
        print(f"\nFAILED: {script.name}")
        sys.exit(1)
    print()

print("=== All checks passed ===")
