import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\Home\Desktop\WASTE\waste_classifier_final.keras"
CLASS_NAMES_PATH = r"C:\Users\Home\Desktop\WASTE\class_names.txt"

IMAGE_SIZE = (224, 224)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)


# ============================================================
# LOAD THE ALREADY-TRAINED MODEL
# ============================================================

@st.cache_resource
def load_trained_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model not found:\n{MODEL_PATH}"
        )
        st.stop()

    # IMPORTANT:
    # This loads the existing trained model.
    # It DOES NOT train the model.
    model = tf.keras.models.load_model(MODEL_PATH)

    return model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )
        st.stop()

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return class_names


# ============================================================
# LOAD MODEL AND CLASSES
# ============================================================

model = load_trained_model()
class_names = load_class_names()


# ============================================================
# HEADER
# ============================================================

st.title("♻️ AI Waste Classification System")

st.write(
    "Upload an image of waste and the trained AI model "
    "will identify whether it is biodegradable, recyclable, or trash."
)


# ============================================================
# SHOW MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write("Model: MobileNetV2")
    st.write("Input size: 224 × 224 pixels")
    st.write("Number of classes:", len(class_names))

    st.write("Classes:")

    for class_name in class_names:
        st.write(f"• {class_name}")


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a waste image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ]
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def classify_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to the same size used during training
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Your trained model already contains
    # MobileNetV2 preprocessing.
    #
    # Therefore we DO NOT call preprocess_input here.

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# DISPLAY IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded waste image",
        width="stretch"
    )

    st.write("")


    # ========================================================
    # CLASSIFY BUTTON
    # ========================================================

    if st.button(
        "🔍 CLASSIFY WASTE",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analyzing the image..."
        ):

            (
                predicted_class,
                confidence,
                probabilities
            ) = classify_image(image)


        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.markdown("---")

        st.subheader("🤖 AI Result")

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        if confidence >= 0.90:

            st.success(
                f"✅ The model is highly confident "
                f"that this item is **{predicted_class}**."
            )

        elif confidence >= 0.70:

            st.warning(
                f"⚠️ The model predicts **{predicted_class}**, "
                f"but the confidence is moderate. "
                f"Try taking a clearer image."
            )

        else:

            st.error(
                f"⚠️ The model predicts **{predicted_class}**, "
                f"but confidence is low. "
                f"Please upload a clearer image."
            )


        # ====================================================
        # ALL PROBABILITIES
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📊 Classification Probabilities"
        )

        for i, class_name in enumerate(
            class_names
        ):

            probability = float(
                probabilities[i]
            )

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )


else:

    # ========================================================
    # START SCREEN
    # ========================================================

    st.info(
        "👆 Upload an image of a waste item "
        "to start classification."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Waste Classification System | "
    "Powered by TensorFlow and MobileNetV2"
)