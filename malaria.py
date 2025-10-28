import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 🧠 Load trained model
model = tf.keras.models.load_model("malaria.keras")

# 🔹 Get input shape dynamically
input_shape = model.input_shape[1:4]

st.set_page_config(page_title="Malaria Detection", page_icon="🩸", layout="centered")

st.title("🩸 Malaria Cell Detection App")
st.write("Upload a blood cell image to predict whether it's **Parasitized (Malaria Positive)** or **Uninfected (Normal)**.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif", "webp"])

if uploaded_file is not None:
    # ✅ Open and fix orientation
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    st.image(image, caption="🩺 Uploaded Image", use_container_width=True)

    # ✅ Preprocess image
    img = image.resize((input_shape[1], input_shape[0]))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, height, width, 3)

    # ✅ Predict
    prediction = model.predict(img_array)
    prob = prediction[0][0]

    # ✅ Interpret output
    if prob > 0.5:
        result = "🦠 Parasitized (Malaria Positive)"
        confidence = prob * 100
        st.error(f"**Prediction:** {result}")
    else:
        result = "✅ Uninfected (Normal)"
        confidence = (1 - prob) * 100
        st.success(f"**Prediction:** {result}")

    # ✅ Show confidence
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.caption(f"Raw model output: {prob:.4f}")
