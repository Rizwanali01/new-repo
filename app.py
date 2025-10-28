import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Malaria Detection", page_icon="🦠")

st.title("🦠 Malaria Detection Using CNN")
st.write("Upload a blood smear image to check if it's **Parasitized** or **Uninfected**.")

# Load model
try:
    model = tf.keras.models.load_model("malaria_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # 🟢 Convert to RGB (important fix)
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 🟢 Resize to match your model’s input shape
    img = image.resize((64, 64))  # or (128,128) if that’s your training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        prediction = model.predict(img_array)
        class_names = ['Parasitized', 'Uninfected']
        pred_class = class_names[np.argmax(prediction)]
        st.markdown(f"### 🧬 Prediction: **{pred_class}**")
        st.progress(float(np.max(prediction)))
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.info("Developed with ❤️ using Streamlit and TensorFlow")
