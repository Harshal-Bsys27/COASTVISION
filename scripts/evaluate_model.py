#!/usr/bin/env python3
r"""
Evaluate a YOLO model using the Ultralytics API and collect key metrics/artifacts.

Usage:
  .venv\Scripts\python scripts/evaluate_model.py --model models/best.pt --data dataset/data.yaml --device 0 --imgsz 640 --save-json

This script runs `model.val()` and then copies the latest `runs/val/exp*` outputs
into `runs/val_latest_evaluation/` for easy inspection. It prints a short summary
and lists key artifact files (confusion matrix, PR curve, predictions JSON).
"""
import argparse
import shutil
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    print("Error: could not import ultralytics. Make sure your venv has ultralytics installed.")
    raise


def find_latest_val_run(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        return None
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt", help="Path to model weights")
    p.add_argument("--data", default="dataset/data.yaml", help="Dataset YAML for validation")
    p.add_argument("--imgsz", type=int, default=640, help="Image size for validation")
    p.add_argument("--device", default="0", help="Device for inference (0 or 'cpu' or 'cuda:0')")
    p.add_argument("--save-json", action="store_true", help="Save predictions as COCO JSON if supported")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(2)

    print(f"Evaluating model {model_path} on data {args.data} (imgsz={args.imgsz}, device={args.device})")
    model = YOLO(str(model_path))

    # Run validation
    val_kwargs = dict(data=str(args.data), imgsz=args.imgsz, device=args.device)
    if args.save_json:
        # ultralytics uses `save_json` in some versions; include if available
        val_kwargs["save_json"] = True

    results = model.val(**val_kwargs)

    # Print concise summary
    print("\n=== Validation result object ===")
    try:
        print(results)
    except Exception:
        print(repr(results))

    # Locate latest runs/val directory and copy artifacts
    runs_val = Path("runs") / "val"
    latest = find_latest_val_run(runs_val)
    dest = Path("runs") / "val_latest_evaluation"
    if latest:
        print(f"\nFound latest validation run: {latest}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(latest, dest)
        print(f"Copied artifacts to {dest}")

        # List important files
        print("\nArtifacts:")
        for name in ["results.png", "confusion_matrix.png", "precision_recall_curve.png", "metrics.json", "metrics.csv", "predictions.json"]:
            pfile = dest / name
            if pfile.exists():
                print(f" - {name}: {pfile}")
        # Also print any images or json files in the directory
        for f in sorted(dest.iterdir()):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.json', '.csv'):
                print("  ", f.name)
    else:
        print("No validation run artifacts found under runs/val/. Check ultralytics output directory.")

    print("\nDone.")


if __name__ == '__main__':
    main()
