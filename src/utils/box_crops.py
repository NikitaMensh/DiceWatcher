# detect_and_crop.py
import argparse, csv
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def pad_clip(x1, y1, x2, y2, w, h, pad=4):
    return (max(0, x1-pad), max(0, y1-pad), min(w-1, x2+pad), min(h-1, y2+pad))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder with source images")
    ap.add_argument("--out", default="crops", help="Output root")
    ap.add_argument("--weights", required=True, help="YOLO weights .pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min_side", type=int, default=16, help="Skip tiny boxes")
    args = ap.parse_args()

    imgdir = Path(args.images)
    out = Path(args.out)
    unlabeled = out / "unlabeled"
    unlabeled.mkdir(parents=True, exist_ok=True)
    meta_csv = out / "crops_meta.csv"

    model = YOLO(args.weights)

    with open(meta_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["crop_file","src_image","x1","y1","x2","y2","conf","pred_cls"])
        for imgp in sorted(imgdir.glob("*.*")):
            if imgp.suffix.lower() not in [".jpg",".jpeg",".png"]: 
                continue
            im = cv2.imread(str(imgp))
            if im is None: 
                continue
            H, W = im.shape[:2]
            r = model.predict(source=im, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)

            for k,(b,cf,cl) in enumerate(zip(xyxy, conf, cls)):
                x1,y1,x2,y2 = map(int, b)
                if (x2-x1)<args.min_side or (y2-y1)<args.min_side:
                    continue
                x1,y1,x2,y2 = pad_clip(x1,y1,x2,y2,W,H, pad=4)
                crop = im[y1:y2, x1:x2]
                fname = f"{imgp.stem}_k{k}_x{x1}_y{y1}_x{x2}_y{y2}_c{cf:.2f}.png"
                cv2.imwrite(str(unlabeled/fname), crop)
                wr.writerow([fname, imgp.name, x1,y1,x2,y2, f"{cf:.3f}", int(cl)])

if __name__ == "__main__":
    main()