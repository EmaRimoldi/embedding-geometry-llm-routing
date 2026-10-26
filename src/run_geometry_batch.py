"""Pack every geometry task into 20 disjoint sequential batches."""
import argparse, json, subprocess, sys
from common import ROOT
def assigned_tasks(worker, count, workers=20):
    if not 0 <= worker < workers:
        raise ValueError("worker must be in [0, workers)")
    return range(worker, count, workers)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", type=int)
    args = parser.parse_args()
    count = len(json.loads((ROOT / "configs/geometry_tasks.json").read_text()))
    for task in assigned_tasks(args.worker, count):
        subprocess.run([sys.executable, "-u", str(ROOT / "src/run_task.py"), str(task)], check=True)
if __name__ == "__main__":
    main()
