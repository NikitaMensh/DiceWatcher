from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import base64, io, numpy as np, cv2
from typing import List


from ultralytics import YOLO

# load once
MODEL_PATH = "models/best.pt"  # or "best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = ["1","2","3","4","5","6"]  # class 0->"1", ... 5->"6"

app = FastAPI(title="Dice OCR API", version="1.0.0")

def run_yolo(pil_img, conf, iou, imgsz):
    return model.predict(source=pil_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    label: str

class PredictResponse(BaseModel):
    boxes: List[Box]
    numbers: List[int]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0, le=1),
    iou: float = Query(0.6, ge=0, le=1),
    imgsz: int = Query(640),
    sort: str = Query("none", pattern="^(none|x|y)$")
):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    r = run_yolo(img, conf, iou, imgsz)

    # boxes + numbers
    boxes = []
    for b, c, s in zip(r.boxes.xyxy.cpu().tolist(),
                       r.boxes.cls.cpu().tolist(),
                       r.boxes.conf.cpu().tolist()):
        x1, y1, x2, y2 = b
        cls = int(c)
        boxes.append({
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "conf": float(s), "cls": cls, "label": CLASS_NAMES[cls]
        })

    if sort == "x":
        boxes.sort(key=lambda z: z["x1"])
    elif sort == "y":
        boxes.sort(key=lambda z: z["y1"])

    numbers = [b["cls"] + 1 for b in boxes]

    # annotated image → base64 PNG
    plotted = r.plot()  # numpy BGR
    ok, buf = cv2.imencode(".png", plotted)
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    return JSONResponse(content={
        "numbers": numbers,                     # e.g., [4,1,5,...]
        "boxes": boxes,                         # optional, keep if useful
        "image_b64": f"data:image/png;base64,{img_b64}"
    })