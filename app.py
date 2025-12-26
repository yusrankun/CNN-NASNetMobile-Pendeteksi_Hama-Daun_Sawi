import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ======================================
# CONFIG
# ======================================
st.set_page_config(
    page_title="CNN NASNetMobile - Deteksi Hama Daun Sawi",
    layout="centered"
)

MODEL_PATH = "NASNetMobile_Sawi_Final.h5"
IMG_SIZE = (224, 224)

CLASS_NAMES = {
    0: "WITH PEST",
    1: "WITHOUT PEST"
}

# ======================================
# LOAD MODEL (CACHE)
# ======================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ======================================
# IMAGE PREPROCESS
# ======================================
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ======================================
# GRAD-CAM FUNCTIONS
# ======================================
def make_gradcam_heatmap(img_array, model):
    # cari last conv layer otomatis
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def overlay_gradcam(image, heatmap, alpha=0.4):
    image = np.array(image)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ======================================
# UI
# ======================================
st.title("🌱 Deteksi Hama Daun Sawi")
st.markdown(
    "Model **CNN NASNetMobile** untuk mendeteksi **hama pada daun sawi** "
    "dengan **visualisasi Grad-CAM**."
)

uploaded_file = st.file_uploader(
    "Upload gambar daun sawi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        label = CLASS_NAMES[0]
        confidence = 1 - prediction
    else:
        label = CLASS_NAMES[1]
        confidence = prediction

    st.subheader("🧠 Prediction")
    st.write(f"**Class:** {label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # GRAD-CAM
    st.subheader("🔥 Grad-CAM Explanation")
    heatmap = make_gradcam_heatmap(img_array, model)
    gradcam_img = overlay_gradcam(image, heatmap)
    st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)

st.markdown("---")
st.caption("CNN NASNetMobile • Explainable AI (Grad-CAM)")
