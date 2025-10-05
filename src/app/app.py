# streamlit_app.py
import base64, io, json
from collections import Counter

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Dice Detector", layout="wide")

# ---- Sidebar ----
st.sidebar.title("Settings")
api_url = st.sidebar.text_input("API endpoint", "http://dice_api:8000/predict")
conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU", 0.0, 1.0, 0.60, 0.01)
imgsz = st.sidebar.select_slider("Image size", [416,512,640,768,960,1280], value=640)
sort = st.sidebar.selectbox("Sort detections", ["none","x","y"], index=0)

st.markdown("### Dice Face Detector")

# ---- Upload ----
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
img = Image.open(uploaded).convert("RGB") if uploaded else None

cctl = st.columns([1,1,1,1])
with cctl[0]:
    rule = st.selectbox("Rule", ["Sum of pips", "Successes (>= than threshold)", "Modifier hits"], index=0)
with cctl[1]:
    target = st.slider("Target", 2, 6, 5) if rule != "Sum of pips" else 5
with cctl[2]:
    modifier = st.number_input("Modifier", -3, 3, 0, 1) if rule == "Modifier hits" else 0
with cctl[3]:
    six_double = st.checkbox("6 counts as 2", value=False) if rule == "Modifier hits" else False

col_img, col_pred = st.columns(2)
with col_img:
    st.markdown("**Original**")
    if img:
        st.image(img, use_container_width=True)
    else:
        st.info("Choose an image.")

with col_pred:
    st.markdown("**Annotated**")
    ann_ph = st.empty()

# ---- Predict ----
if st.button("Predict", type="primary", disabled=(img is None)):
    try:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
        params = {"conf": conf, "iou": iou, "imgsz": imgsz, "sort": sort}
        with st.spinner("Running inference..."):
            r = requests.post(api_url, params=params, files=files, timeout=120)
        r.raise_for_status()
        data = r.json()

        # annotated image
        b64 = data.get("image_b64", "")
        if b64.startswith("data:image"):
            b64 = b64.split(",", 1)[1]
        annotated_bytes = base64.b64decode(b64) if b64 else None
        if annotated_bytes:
            ann_ph.image(annotated_bytes, use_container_width=True)

        # stats
        numbers = data.get("numbers", [])

        st.markdown("---")
        st.markdown("**Counts**")
        st.code(numbers, language="text")

        def eval_rule(nums):
            if rule == "Sum of pips":
                return sum(nums)
            if rule == "Successes (>= than threshold)":
                return sum(1 for d in nums if d >= target)
            # Wargame hits
            hits = 0
            for d in nums:
                val = d + modifier
                if val >= target:
                    hits += 2 if six_double and d == 6 else 1
            return hits

        result = eval_rule(numbers)
        st.success(f"Result ({rule}): {result}")

        # downloads
        d1, d2 = st.columns(2)
        with d1:
            if annotated_bytes:
                st.download_button("Download annotated PNG", annotated_bytes, "annotated.png", "image/png", use_container_width=True)
        with d2:
            st.download_button("Download JSON", json.dumps(data, indent=2), "prediction.json", "application/json", use_container_width=True)

    except Exception as e:
        st.error(f"{e}")
