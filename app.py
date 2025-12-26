import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Deteksi Hama Sawi",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Deteksi Hama pada Daun Sawi")
st.write("Upload gambar daun sawi untuk mendeteksi **ADA HAMA** atau **TANPA HAMA**")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("NASNetMobile_Sawi_Final.h5'")

model = load_model()

CLASS_NAMES = ["with_pest", "without_pest"]

# =============================
# IMAGE PREPROCESS
# =============================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload Gambar Daun Sawi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # =============================
    # PREDICTION
    # =============================
    img_input = preprocess_image(image)
    preds = model.predict(img_input)[0]

    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.markdown("---")
    st.subheader("🧠 Hasil Prediksi")

    if predicted_class == "with_pest":
        st.error(f"🚨 **ADA HAMA** ({confidence:.2f}%)")
    else:
        st.success(f"✅ **TANPA HAMA** ({confidence:.2f}%)")

    st.write("📊 Confidence per kelas:")
    for cls, prob in zip(CLASS_NAMES, preds):
        st.write(f"- **{cls}** : {prob*100:.2f}%")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("Model: NASNetMobile | Deep Learning CNN")
