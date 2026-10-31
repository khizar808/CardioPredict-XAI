"""Run the whole project in sequence:  EDA -> train -> predict + evaluate.

    python run_all.py               # everything (training takes a few minutes)
    python run_all.py --repeats 1   # faster training
    python run_all.py --skip-train  # EDA + evaluate the models already in models/
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Run the MedPredict pipeline.")
parser.add_argument("--repeats", type=int, default=3, help="outer-CV repeats for training (default 3)")
parser.add_argument("--skip-train", action="store_true", help="skip step 2 and reuse models/")
args = parser.parse_args()

steps = [["1_eda.py"]]
if not args.skip_train:
    steps.append(["2_train.py", "--repeats", str(args.repeats)])
steps.append(["3_predict_evaluate.py"])

for step in steps:
    print(f"\n{'#' * 70}\n# {' '.join(step)}\n{'#' * 70}", flush=True)
    result = subprocess.run([sys.executable, *step], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n{step[0]} failed (exit code {result.returncode}). Fix the error above and re-run.")

print("\nAll steps finished. Results are in the results/ folder.")
