#!/usr/bin/env python3
# train.py — YOLO dice detector (6 classes: 1..6)
# Usage:
#   python train.py --data_root /path/to/dataset --do_split
# Dataset expected raw layout before split:
#   /dataset/images/*.jpg
#   /dataset/labels/*.txt   # YOLO txt, classes 0..5
# After split it creates images/{train,val,test} and labels/{train,val,test}

import argparse, random, shutil, sys, time
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = ["1", "2", "3", "4", "5", "6"]  # maps 0..5 → shown labels

def find_pairs(images_dir: Path, labels_dir: Path):
    pairs = []
    for img in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpeg")):
        lab = labels_dir / (img.stem + ".txt")
        if lab.exists():
            pairs.append((img, lab))
    return pairs

def split_dataset(root: Path, train=0.8, val=0.1, test=0.1, seed=42):
    assert abs(train + val + test - 1.0) < 1e-6
    images = root / "images"
    labels = root / "labels"
    assert images.exists() and labels.exists(), "Expected images/ and labels/ under data_root"

    pairs = find_pairs(images, labels)
    if not pairs:
        raise RuntimeError("No image/label pairs found")

    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_tr = int(n * train)
    n_val = int(n * val)
    n_te = n - n_tr - n_val

    # make dirs
    for split in ["train", "val", "test"]:
        (root / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (root / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    def move_subset(sub, split_name):
        for img, lab in sub:
            shutil.move(str(img), str(root / f"images/{split_name}/{img.name}"))
            shutil.move(str(lab), str(root / f"labels/{split_name}/{lab.name}"))

    move_subset(pairs[:n_tr], "train")
    move_subset(pairs[n_tr:n_tr + n_val], "val")
    move_subset(pairs[n_tr + n_val:], "test")

    return {"train": n_tr, "val": n_val, "test": n_te, "total": n}

def write_data_yaml(root: Path, yaml_path: Path):
    txt = (
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names: {CLASS_NAMES}\n"
    )
    yaml_path.write_text(txt)
    return yaml_path

def count_labels(label_dir: Path):
    import collections
    cnt = collections.Counter()
    for p in label_dir.rglob("*.txt"):
        for line in p.read_text().strip().splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0])
            cnt[cls] += 1
    return {CLASS_NAMES[k]: v for k, v in sorted(cnt.items())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True, help="Dataset root with images/ and labels/")
    ap.add_argument("--model", type=str, default="yolov8s.pt", help="Ultralytics model (yolov8n/s/m/l/x or path)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", type=str, default=None, help="e.g. 0, 0,1 or cpu")
    ap.add_argument("--do_split", action="store_true", help="Split images/labels into train/val/test")
    ap.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", type=Path, default=Path("runs/detect"))
    ap.add_argument("--name", type=str, default=None, help="Run name")
    args = ap.parse_args()

    root = args.data_root.resolve()
    if args.do_split:
        stats = split_dataset(root, *args.split, seed=args.seed)
        print(f"Split -> {stats}")

    # sanity check
    for sub in ["train", "val"]:
        if not (root / f"images/{sub}").exists():
            print(f"Missing images/{sub}. Provide split or use --do_split.", file=sys.stderr)
            sys.exit(1)

    yaml_path = write_data_yaml(root, root / "dice.yaml")
    print(f"Wrote {yaml_path}")

    # show label counts
    lbl_counts = count_labels(root / "labels")
    print(f"Label counts: {lbl_counts}")

    # train
    model = YOLO(args.model)
    name = args.name or f"dice_{Path(args.model).stem}_{int(time.time())}"
    print("Training...")
    model.train(
        data=str(yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=name,
        seed=args.seed,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=180, translate=0.20, scale=0.50, shear=2.0,
        flipud=0.0, fliplr=0.5,
        perspective=0.001, blur=0.2,
        mosaic=1.0,
        cos_lr=True,
        patience=30  # early stop
    )

    ckpt = args.project / name / "weights" / "best.pt"
    print(f"Best weights: {ckpt}")

    # validate on val and test
    print("Validate on val set...")
    model = YOLO(str(ckpt))
    val_metrics = model.val(data=str(yaml_path), imgsz=args.imgsz, device=args.device, split="val", plots=True, save_json=False)
    print(f"Val mAP50: {val_metrics.results_dict.get('metrics/mAP50', 'n/a'):.4f}  "
          f"mAP50-95: {val_metrics.results_dict.get('metrics/mAP50-95', 'n/a'):.4f}")

    if (root / "images/test").exists():
        print("Evaluate on test set...")
        test_metrics = model.val(data=str(yaml_path), imgsz=args.imgsz, device=args.device, split="test", plots=True, save_json=False)
        print(f"Test mAP50: {test_metrics.results_dict.get('metrics/mAP50', 'n/a'):.4f}  "
              f"mAP50-95: {test_metrics.results_dict.get('metrics/mAP50-95', 'n/a'):.4f}")

if __name__ == "__main__":
    main()
