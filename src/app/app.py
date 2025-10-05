import base64
import io
import json
from collections import Counter

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Dice Detector", layout="wide")

# ---- Sidebar controls ----
st.sidebar.title("Settings")
api_url = st.sidebar.text_input(
    "API endpoint",
    value="http://dice_api:8000/predict",
    help="FastAPI /predict URL"
)
conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU", 0.0, 1.0, 0.60, 0.01)
imgsz = st.sidebar.select_slider("Image size", options=[416, 512, 640, 768, 960, 1280], value=640)
sort = st.sidebar.selectbox("Sort detections", ["none", "x", "y"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: keep the FastAPI server running separately.")

# ---- Header ----
st.markdown(
    """
    <style>
      .big-title {font-size: 26px; font-weight: 700; margin-bottom: 0.2rem;}
      .subtle {color: #6b7280;}
      .boxed {border:1px solid #e5e7eb; border-radius: 12px; padding: 14px; background: #fafafa;}
      .pill {background:#eef2ff; color:#3730a3; border-radius:9999px; padding:4px 10px; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">Dice Face Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Upload a photo, run prediction, view boxes and counts.</div>', unsafe_allow_html=True)
st.markdown("")

# ---- Uploader ----
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

cols = st.columns([1, 1])
with cols[0]:
    st.markdown("**Input**")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
    else:
        st.info("Choose an image to get started.")

with cols[1]:
    st.markdown("**Prediction**")

# ---- Predict button ----
predict_clicked = st.button("Predict", type="primary", use_container_width=True, disabled=(uploaded is None))

# ---- Inference ----
if predict_clicked and uploaded:
    try:
        params = {"conf": conf, "iou": iou, "imgsz": imgsz, "sort": sort}
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
        with st.spinner("Running inference..."):
            resp = requests.post(api_url, params=params, files=files, timeout=120)

        if resp.status_code != 200:
            st.error(f"Request failed [{resp.status_code}]: {resp.text[:500]}")
        else:
            data = resp.json()
            numbers = data.get("numbers", [])
            boxes = data.get("boxes", [])
            b64 = data.get("image_b64", "")

            # Annotated image
            if b64.startswith("data:image"):
                b64 = b64.split(",", 1)[1]
            annotated_bytes = base64.b64decode(b64) if b64 else None

            c1, c2 = st.columns([1, 1])
            with c1:
                if annotated_bytes:
                    st.image(annotated_bytes, caption="Annotated", use_container_width=True)
                else:
                    st.warning("No annotated image returned.")

                # Downloads
                dcols = st.columns(2)
                with dcols[0]:
                    if annotated_bytes:
                        st.download_button(
                            "Download annotated PNG",
                            data=annotated_bytes,
                            file_name="annotated.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                with dcols[1]:
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(data, indent=2),
                        file_name="prediction.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            # Stats
            with c2:
                st.markdown('<div class="boxed">', unsafe_allow_html=True)
                st.markdown("**Detected numbers**")
                if numbers:
                    counts = Counter(numbers)
                    # Normalize to 1..6 for a consistent chart
                    idx = [1, 2, 3, 4, 5, 6]
                    df = pd.DataFrame(
                        {"count": [counts.get(i, 0) for i in idx]},
                        index=[str(i) for i in idx]
                    )

                    # KPI pills
                    kpi = st.columns(3)
                    total = sum(df["count"])
                    kpi[0].markdown(f'<span class="pill">Total dice: {total}</span>', unsafe_allow_html=True)
                    kpi[1].markdown(f'<span class="pill">Unique faces: {sum(df["count"]>0)}</span>', unsafe_allow_html=True)
                    kpi[2].markdown(f'<span class="pill">Max freq: {df["count"].max() if total>0 else 0}</span>', unsafe_allow_html=True)

                    st.bar_chart(df, height=260)
                    st.markdown("**List**")
                    st.code(numbers)
                else:
                    st.info("No detections.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Optional raw boxes
                with st.expander("Boxes (xyxy, class, conf)"):
                    st.json(boxes)

    except Exception as e:
        st.error(f"Error: {e}")