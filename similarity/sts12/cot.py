from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_runner import run_cli


if __name__ == "__main__":
    run_cli(default_baselines=["cot"])
