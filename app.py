import os
import streamlit as st
from PIL import Image
import gdown
import torch
import json
from Script.prediction import load_model, predict_image  # assumes load_model reads model.pth from cwd

import os
BASE = os.path.dirname(__file__)            # the folder where app.py lives
CLASS_FILE = os.path.join(BASE, "meta", "classes.txt")
NUT_FILE   = os.path.join(BASE, "meta", "classes_nutrition.json")

# Cache downloads and model loading to speed up repeated inference
@st.cache(allow_output_mutation=True)
def fetch_weights(drive_id: str, dst: str = "model.pth"):
    """
    Download model weights from Google Drive using gdown if not already present.
    """
    if not os.path.exists(dst):
        url = f"https://drive.google.com/file/d/1Sh447_nPFg8WMIzX9k2ryQXZhbqEU42C/view?usp=sharing={drive_id}"
        gdown.download(url, dst, quiet=False)
    return dst

@st.cache(allow_output_mutation=True)
def load_model_and_data():
    # Download model weights
    drive_id = st.secrets.get("drive_model_id")
    if not drive_id:
        st.error("Model Drive ID not found in secrets. Please set 'drive_model_id'.")
        st.stop()
    fetch_weights(drive_id)

    # Load model (prediction.load_model should load from local 'model.pth')
    model = load_model(model_path="model.pth", classes_file=CLASS_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load class names
    with open(CLASS_FILE) as f:
     class_names = [line.strip() for line in f if line.strip()]

    # Load nutrition data from JSON
    with open(NUT_FILE) as f:
     nutrition_data = json.load(f)

    return model, device, class_names, nutrition_data


def main():
    st.title("🍽️ Food Classifier & Nutrition Facts")
    st.write(
        "Upload an image of a dish, and get its predicted label along with nutritional information."
    )

    # Confidence threshold slider
    threshold = st.sidebar.slider(
        "Confidence threshold (%) for detecting food",
        min_value=0,
        max_value=100,
        value=20
    )

    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model, device, class_names, nutrition_data = load_model_and_data()

        # Predict label and confidence from PIL Image
        label, confidence = predict_image(model, device, image, class_names)

        # Show warning if no food detected
        if confidence <= threshold or label.lower() == "nofood":
            st.warning("No food detected in the image. Please upload an appropriate dish image.")
            st.image(image, use_column_width=True)
        else:
            st.success(f"Prediction: {label} ({confidence:.2f}%)")

            # Display nutrition info
            if label in nutrition_data:
                st.subheader("Nutritional Information")
                info = nutrition_data[label]
                st.write(f"- Calories: {info.get('calories', 'N/A')} kcal")
                st.write(f"- Protein: {info.get('protein', 'N/A')} g")
                st.write(f"- Fat: {info.get('fat', 'N/A')} g")
                st.write(f"- Carbohydrates: {info.get('carbohydrate', 'N/A')} g")
                st.write(f"- Energy: {info.get('energy', 'N/A')} kJ")
            else:
                st.warning("No nutritional data available for this item.")

if __name__ == "__main__":
    main()
