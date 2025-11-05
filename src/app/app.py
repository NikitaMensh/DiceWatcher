import time, io, base64, json
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

from mechanics import Weapon, UnitConfig
from mechanics import evaluate_attack_sequence

st.set_page_config(page_title="DiceWatcher Live + Mechanics", layout="wide")

# ---------- helpers ----------
def list_cameras(max_index=8):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        ok, _ = cap.read()
        if ok: found.append(i)
        cap.release()
    return found or [0]

def stabilize_score(prev_gray_small, gray_small):
    return float(np.mean(cv2.absdiff(prev_gray_small, gray_small)))

def encode_jpg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None

def decode_b64_image(data_url):
    if not data_url: return None
    b64 = data_url.split(",", 1)[1] if data_url.startswith("data:image") else data_url
    return base64.b64decode(b64)

def eval_counts(numbers):
    counts = Counter(numbers)
    idx = [1,2,3,4,5,6]
    df = pd.DataFrame({"count":[counts.get(i,0) for i in idx]}, index=[str(i) for i in idx])
    return df

def load_cfg(cfg):
    if cfg is not None:
        try:
            j = json.loads(cfg.getvalue().decode("utf-8"))
            ws = [Weapon(**w) for w in j["weapons"]]
            return UnitConfig(
                unit=j["unit"],
                models=int(j["models"]),
                bs=int(j["bs"]),
                weapons=ws,
                abilities=j.get("abilities", {})
            )
        except Exception as e:
            st.sidebar.error(f"Bad JSON: {e}")
            return None
    return None

def take_current_numbers():
    if not ss.last_numbers:
        st.warning("No numbers from detector yet."); return
    if phase == "Hit":
        if reroll_ones:
            # if reroll_ones is ticked, first click captures original hits; second click captures rerolled 1s
            if len(ss.hit_rolls) == 0:
                ss.hit_rolls = ss.last_numbers[:]
                st.success(f"Stored Hit rolls ({len(ss.hit_rolls)}). Now throw re-rolls of 1s and press 'Take numbers' again.")
            else:
                ss.hit_rerolls = ss.last_numbers[:]
                st.success(f"Stored re-rolls for 1s ({len(ss.hit_rerolls)}).")
        else:
            ss.hit_rolls = ss.last_numbers[:]
            st.success(f"Stored Hit rolls ({len(ss.hit_rolls)}).")
    elif phase == "Wound":
        ss.wound_rolls = ss.last_numbers[:]
        st.success(f"Stored Wound rolls ({len(ss.wound_rolls)}).")
    else:
        ss.save_rolls = ss.last_numbers[:]
        st.success(f"Stored Save rolls ({len(ss.save_rolls)}).")
    
# ---------- sidebar stuff ----------
st.sidebar.title("Settings")
api_url = st.sidebar.text_input("API endpoint", "http://dice_api:8000/predict")
conf = st.sidebar.slider("Conf", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU", 0.0, 1.0, 0.60, 0.01)
imgsz = st.sidebar.select_slider("Image size", [416,512,640,768,960], value=640)
sort = st.sidebar.selectbox("Sort detections", ["none","x","y"], index=0)

ss = st.session_state
if 'cams' not in ss:
    ss.cams = list_cameras()

cam_index = st.sidebar.selectbox("Camera", st.session_state.cams, index=0)
width = st.sidebar.select_slider("Capture width", [320,480,640,800,960,1280], value=640)
height = st.sidebar.select_slider("Capture height", [240,360,480,600,720], value=480)

st.sidebar.markdown("---")
st.sidebar.caption("Stabilization")
win = st.sidebar.slider("Window size (frames)", 3, 15, 7, 1)
diff_thr = st.sidebar.slider("Stillness threshold", 0.0, 8.0, 2.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.caption("Mechanics")

uploaded_unit = st.sidebar.file_uploader("Upload unit JSON", type=["json"], key="unit_file")
if 'cfg' not in ss:
    ss.cfg = uploaded_unit
if uploaded_unit != None:
    ss.cfg = uploaded_unit

target_toughness = st.sidebar.number_input("Target Toughness", 1, 10, 4)
target_save = st.sidebar.number_input("Target Save (2..6)", 2, 6, 3)
cover_mod = st.sidebar.number_input("Cover modifier", 0, 3, 0)
invuln = st.sidebar.number_input("Invuln (0=none)", 0, 6, 0)
use_invuln = None if invuln == 0 else int(invuln)

# ---------- top controls ----------
st.markdown("### DiceWatcher — Live + Phase Mechanics")

# phase controls
phase = st.radio("Phase", ["Hit", "Wound", "Save"], horizontal=True)
reroll_ones = st.checkbox("Re-roll hit 1s (aura/ability active)", value=False if phase != "Hit" else False)

# start/stop
col_btn = st.columns([1,1,6])
start = col_btn[0].button("Start camera", type="primary")
stop = col_btn[1].button("Stop")

# placeholders
col1, col2 = st.columns(2)
ph_live = col1.empty(); col1.caption("Live")
ph_ann = col2.empty();  col2.caption("Annotated")

ph_counts = st.empty()
ph_phase = st.empty()
ph_mech  = st.empty()
ph_dl1, ph_dl2 = st.columns(2)

# ---------- states ----------
if "run" not in ss: ss.run = False
if "last_numbers" not in ss: ss.last_numbers = []
if "hit_rolls" not in ss: ss.hit_rolls = []
if "hit_rerolls" not in ss: ss.hit_rerolls = []
if "wound_rolls" not in ss: ss.wound_rolls = []
if "save_rolls" not in ss: ss.save_rolls = []

if start: ss.run = True
if stop:  ss.run = False


col_take, col_clear = st.columns([1,1])
if col_take.button("Take numbers from detector", type="secondary"):
    take_current_numbers()
if col_clear.button("Clear phase data"):
    if phase == "Hit":
        ss.hit_rolls = []; ss.hit_rerolls = []
    elif phase == "Wound":
        ss.wound_rolls = []
    else:
        ss.save_rolls = []

# show phase data
with ph_phase.container():
    st.markdown("**Phase buffers**")
    st.code({
        "Hit": ss.hit_rolls,
        "Hit re-rolls 1s": ss.hit_rerolls,
        "Wound": ss.wound_rolls,
        "Save": ss.save_rolls
    }, language="json")

if 'cap' not in ss and 'cam_index' not in ss:
    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    ss.cap = cap
    ss.cam_index = cam_index

if ss.cam_index != cam_index:
    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    ss.cap = cap
    ss.cam_index = cam_index

# ---------- main loop ----------
if ss.run:
    if not ss.cap.isOpened():
        ss.cap.open(ss.cam_index)
    diffs = deque(maxlen=win)
    prev_small = None
    try:
        while ss.run:
            ok, frame = ss.cap.read()
            if not ok:
                st.error("Camera read failed."); break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ph_live.image(rgb, channels="RGB", width='stretch')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 120))
            if prev_small is not None:
                diffs.append(stabilize_score(prev_small, small))
            prev_small = small
            stable = (len(diffs) == diffs.maxlen) and (np.mean(diffs) < diff_thr)
            if stable:
                jpg = encode_jpg(frame)
                try:
                    r = requests.post(
                        api_url,
                        params={"conf": conf, "iou": iou, "imgsz": imgsz, "sort": sort},
                        files={"file": ("frame.jpg", jpg, "image/jpeg")},
                        timeout=15,
                    )
                    r.raise_for_status()
                    data = r.json()

                    ann_bytes = decode_b64_image(data.get("image_b64", ""))
                    if ann_bytes:
                        ph_ann.image(ann_bytes, width='stretch')

                    nums = data.get("numbers", [])
                    ss.last_numbers = nums[:]  # make available to “Take numbers” button

                    ph_counts.bar_chart(eval_counts(nums), height=220)

                    # downloads
                    with ph_dl1:
                        if ann_bytes:
                            st.download_button("Download annotated PNG", ann_bytes, "annotated.png",
                                               "image/png", width='stretch')
                    with ph_dl2:
                        st.download_button("Download JSON", json.dumps(data, indent=2),
                                           "prediction.json", "application/json", width='stretch')

                except Exception as e:
                    st.warning(f"Inference error: {e}")

                diffs.clear()

            time.sleep(0.01)
            ss.run = not stop and ss.run
    finally:
        pass
else:
    ss.cap.release()
    st.info("Start camera. When the scene stabilizes, detector runs and numbers appear. Use **Take numbers** to store per phase.")

# ---------- mechanics execution ----------
st.markdown("---")
run_mech = st.button("Evaluate Mechanics (from stored phases)", type="primary")
if run_mech:
    cfg = load_cfg(ss.cfg)
    if not cfg:
        st.error("No valid unit config.")
    else:
        res = evaluate_attack_sequence(
            cfg, weapon_idx=0,
            target_toughness=int(target_toughness),
            target_save=int(target_save),
            cover_mod=int(cover_mod),
            invuln=use_invuln,
            hit_rolls=st.session_state.hit_rolls,
            hit_rerolls_ones=(st.session_state.hit_rerolls if st.checkbox("Re-roll hit 1s", value=False, key="rr_use") else None),
            wound_rolls=st.session_state.wound_rolls,
            save_rolls=st.session_state.save_rolls
        )
        ph_mech.success(f"Mechanics result: {res}")
