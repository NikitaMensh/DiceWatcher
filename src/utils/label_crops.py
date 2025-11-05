# label_crops.py
import argparse, csv
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np

KEYMAP = {ord("1"):1, ord("2"):2, ord("3"):3, ord("4"):4, ord("5"):5, ord("6"):6}

def load_progress(csv_path):
    done = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done[row["crop_file"]] = int(row["label"])
    return done

def ensure_class_dirs(root):
    d = {i: root/str(i) for i in range(1,7)}
    for p in d.values(): p.mkdir(parents=True, exist_ok=True)
    return d

def counts(done):
    c = defaultdict(int)
    for v in done.values(): c[v] += 1
    return c

def save_log(done, path):
    tmp = path.with_suffix(".tmp.csv")
    with open(tmp, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["crop_file","label"])
        for k,v in sorted(done.items()):
            wr.writerow([k,v])
    tmp.replace(path)

def auto_gamma(im, alpha=1.08, beta=8):
    return cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

def render_canvas(img_bgr, title, stats, canvas_side=720, scale=3):
    Hc = Wc = canvas_side
    h, w = img_bgr.shape[:2]
    sf = max(1, min(scale, int(min(Wc / w, Hc / h))))
    show = cv2.resize(img_bgr, (w*sf, h*sf), interpolation=cv2.INTER_AREA)

    canvas = np.full((Hc, Wc, 3), 32, np.uint8)
    y0 = (Hc - show.shape[0]) // 2
    x0 = (Wc - show.shape[1]) // 2
    canvas[y0:y0+show.shape[0], x0:x0+show.shape[1]] = show

    # header
    cv2.rectangle(canvas, (0,0), (Wc, 38), (24,24,24), -1)
    cv2.putText(canvas, title, (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210,210,210), 2, cv2.LINE_AA)
    # footer
    cv2.rectangle(canvas, (0,Hc-30), (Wc,Hc), (24,24,24), -1)
    cv2.putText(canvas, stats, (10,Hc-9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops", help="Root crops folder")
    ap.add_argument("--window", default="Label crops", help="Window title")
    ap.add_argument("--canvas", type=int, default=720, help="Canvas size (px)")
    ap.add_argument("--scale", type=int, default=3, help="Upscale factor for crop view")
    ap.add_argument("--gamma", action="store_true", help="Apply mild brightness gain")
    ap.add_argument("--shuffle", action="store_true", help="Randomize unlabeled order")
    args = ap.parse_args()

    root = Path(args.crops)
    unlabeled = root/"unlabeled"
    log_csv = root/"labels.csv"
    done = load_progress(log_csv)
    class_dir = ensure_class_dirs(root)

    queue = [p for p in sorted(unlabeled.glob("*.png")) if p.name not in done]
    if args.shuffle:
        import random; random.shuffle(queue)

    if not queue and not done:
        print("No crops found in crops/unlabeled"); return

    # UI
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, args.canvas, args.canvas)

    history = deque()  # (filename, label)

    i = 0
    while i < len(queue):
        p = queue[i]
        im = cv2.imread(str(p))
        if im is None:
            i += 1; continue
        if args.gamma:
            im = auto_gamma(im)

        # HUD
        cnt = counts(done)
        stat = " ".join(f"{k}:{cnt.get(k,0)}" for k in range(1,7))
        helpbar = "  [1–6]=label  s=skip  b=back  u=undo  q=quit"
        title = f"{i+1}/{len(queue)}  {p.name}"
        canvas = render_canvas(im, title, stat + helpbar, args.canvas, args.scale)
        cv2.imshow(args.window, canvas)

        key = cv2.waitKey(0) & 0xFF
        if key in KEYMAP:
            lab = KEYMAP[key]
            dst = class_dir[lab] / p.name
            p.replace(dst)
            done[p.name] = lab
            history.append((p.name, lab))
            save_log(done, log_csv)
            i += 1
        elif key == ord("s"):   # skip
            i += 1
        elif key == ord("b"):   # view previous again
            i = max(0, i-1)
        elif key == ord("u"):   # undo last labeled
            if history:
                fname, lab = history.pop()
                src = class_dir[lab] / fname
                if src.exists(): src.replace(unlabeled / fname)
                done.pop(fname, None)
                save_log(done, log_csv)
                # jump back to that item if still in queue
                for idx, qf in enumerate(queue):
                    if qf.name == fname:
                        i = idx
                        break
                else:
                    i = max(0, i-1)
        elif key == ord("q"):
            break
        # else: ignore unknown keys

    cv2.destroyAllWindows()
    print(f"Finished. Labeled crops: {len(done)}. Log at {log_csv}")

if __name__ == "__main__":
    main()
