"""Train the YOLOv8 detector on the Kvasir-SEG train split with the recipe in configs/yolo_train.yaml (DL-31).

    python src/train_yolo.py                      # Colab, T4: about 30 minutes for 50 epochs of YOLOv8-m
    python src/train_yolo.py --epochs 1 --limit   # quick check that everything is wired

Outputs (runs/yolo/<name>/, git-ignored):
    weights/best.pt, weights/last.pt     best = highest val mAP50-95 (Ultralytics default)
    results.csv, results.png             training curves
    summary.json                         recipe + val and test mAP50 / mAP50-95 / precision / recall

The dataset YAML's `path` is rewritten to an absolute path in a temporary copy, because Ultralytics
resolves relative paths against its own datasets folder (DL-15). Needs ultralytics (Colab).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", type=Path, default=REPO_ROOT / "configs" / "yolo_train.yaml")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "configs" / "kvasir.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override the recipe (smoke tests)")
    parser.add_argument("--fraction", type=float, default=None, help="use only this fraction of the train split (smoke tests)")
    return parser.parse_args()


def absolute_data_yaml(data_yaml, out_dir):
    """Copy of the dataset YAML with `path` made absolute, so Ultralytics finds the images (DL-15)."""
    import yaml
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    data["path"] = str((REPO_ROOT / data["path"]).resolve())
    out = Path(out_dir) / f"{Path(data_yaml).stem}_abs.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def box_metrics(results):
    return {"map50": float(results.box.map50), "map50_95": float(results.box.map),
            "precision": float(results.box.mp), "recall": float(results.box.mr)}


def main():
    args = parse_args()
    import yaml
    from ultralytics import YOLO

    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    if args.epochs:
        recipe["epochs"] = args.epochs
    if args.fraction:
        recipe["fraction"] = args.fraction
    project = REPO_ROOT / recipe.pop("project", "runs/yolo")
    name = recipe.pop("name", "yolov8m_kvasir")
    data_yaml = absolute_data_yaml(args.data, project / name)

    model = YOLO(recipe.pop("model"))
    model.train(data=str(data_yaml), project=str(project), name=name, exist_ok=True, **recipe)

    best = project / name / "weights" / "best.pt"
    model = YOLO(str(best))
    summary = {
        "recipe": {**recipe, "model": str(best), "data": str(args.data)},
        "best_weights": str(best),
        "val": box_metrics(model.val(data=str(data_yaml), split="val", verbose=False)),
        "test": box_metrics(model.val(data=str(data_yaml), split="test", verbose=False)),
    }
    (project / name / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
