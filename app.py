import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(
    page_title="AI Digit Recognizer",
    page_icon="🔢",
    layout="wide"
)

# -------------------- HEADER DESIGN --------------------
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        text-align:center;
        font-weight:bold;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-title {
        text-align:center;
        font-size:18px;
        color:gray;
    }

    .card {
        background-color:#111;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🔢 AI Multi-Digit Recognizer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload handwritten digits and let AI predict the number</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def get_model():
    return tf.keras.models.load_model('models/digit_cnn_model.h5')

model = get_model()

# -------------------- SEGMENTATION --------------------
def segment_digits(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h > 20 and w > 10:
            roi = thresh[y:y+h, x:x+w]
            roi = cv2.copyMakeBorder(roi, 10,10,10,10, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(roi, (28, 28))
            roi = roi / 255.0

            boxes.append((x, roi))

    boxes = sorted(boxes, key=lambda x: x[0])

    return [b[1] for b in boxes]

# -------------------- LAYOUT --------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 📤 Upload Section")

    uploaded_file = st.file_uploader(
        "Upload handwritten number image",
        type=['png','jpg','jpeg']
    )

    st.info("💡 Tip: Use dark pen on white paper for best accuracy")

with col2:
    st.markdown("## 🔮 Prediction Panel")

    if st.button("🚀 PREDICT", use_container_width=True):

        if uploaded_file:

            image = Image.open(uploaded_file)
            img = np.array(image.convert('RGB'))

            st.image(img, caption="Input Image", use_container_width=True)

            img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            digits = segment_digits(img_cv)

            if len(digits) == 0:
                st.error("No digits detected!")
            else:
                result = ""

                st.markdown("### 📊 Predictions:")

                cols = st.columns(len(digits) if len(digits) < 6 else 6)

                for i, d in enumerate(digits):
                    pred = model.predict(d.reshape(1,28,28,1), verbose=0)
                    digit = np.argmax(pred)
                    conf = np.max(pred) * 100

                    result += str(digit)

                    with cols[i % len(cols)]:
                        st.markdown(f"**Digit {i+1}**")
                        st.image(d, width=70)
                        st.success(f"{digit} ({conf:.1f}%)")

                st.markdown("---")
                st.markdown(f"""
                    <div style='text-align:center; font-size:30px; color:#00ff99; font-weight:bold;'>
                        FINAL NUMBER: {result}
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("Please upload an image first!")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray;'>
AI Project | CNN + OpenCV | Multi-Digit Recognition 🤖
</div>
""", unsafe_allow_html=True)