import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# CONFIG
# ===============================
IMG_SIZE = (224, 224)
MODEL_PATH = "NASNetMobile_Sawi_Final.h5"

st.set_page_config(
    page_title="Deteksi Hama Daun Sawi",
    page_icon="🌿",
    layout="centered"
)

# ===============================
# LOAD MODEL (CACHE)
# ===============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ===============================
# UI
# ===============================
st.title("🌿 Deteksi Hama pada Daun Sawi")
st.write("Upload gambar daun sawi untuk mendeteksi **ada hama atau tidak**.")

uploaded_file = st.file_uploader(
    "Upload gambar daun sawi",
    type=["jpg", "jpeg", "png"]
)

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0   # WAJIB
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===============================
# PREDICTION
# ===============================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img_array = preprocess_image(image)

    pred = model.predict(img_array)[0][0]

    # ======================================
    # 🔥 LABEL FIX (PALING PENTING)
    # ======================================
    # Sesuai training:
    # {'Ada_Hama': 0, 'Tanpa_Hama': 1}

    if pred < 0.5:
        label = "🚨 ADA HAMA"
        confidence = (1 - pred) * 100
        st.error(f"{label} ({confidence:.2f}%)")
    else:
        label = "✅ TANPA HAMA"
        confidence = pred * 100
        st.success(f"{label} ({confidence:.2f}%)")

    # Detail probabilitas
    st.markdown("### 📊 Confidence per kelas:")
    st.write(f"- **Ada Hama** : {(1 - pred) * 100:.2f}%")
    st.write(f"- **Tanpa Hama** : {pred * 100:.2f}%")
