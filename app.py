import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
# ------------------------------
# Load your trained model
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("malaria_model.h5")
    return model

model = load_model()

# ------------------------------
# Streamlit App Interface
# ------------------------------
st.set_page_config(page_title="Malaria Detection", page_icon="🦠")

st.title("🦠 Malaria Detection Using CNN")
st.write("Upload a blood smear image to check if it's **Parasitized** or **Uninfected**.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image (update size if your model uses a different input shape)
    img = image.resize((64, 64))  # adjust to your model's input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result = "Parasitized" if prediction[0][0] > 0.5 else "Uninfected"

    st.markdown(f"### 🧬 Prediction: **{result}**")
    st.progress(float(prediction[0][0]))

st.markdown("---")
st.info("Developed with ❤️ using Streamlit and TensorFlow")

