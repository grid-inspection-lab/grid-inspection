"""Run the experiment stages from the repository root."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
STAGES = {
    "train": "study.py",
    "repeat": "repeat.py",
    "audit": "audit.py",
    "pipeline": "pipeline.py",
    "physics": "physics-check.py",
    "plots": "plots.py",
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        subprocess.run([sys.executable, "learning/" + STAGES[stage]], check=True)
