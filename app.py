import os
import streamlit as st
from PIL import Image
import gdown
import torch
import json
from Script.prediction import load_model, predict_image

BASE = os.path.dirname(__file__)
CLASS_FILE = os.path.join(BASE, "meta", "classes.txt")
NUT_FILE = os.path.join(BASE, "meta", "classes_nutrition.json")
threshold = 20

@st.cache_data
def fetch_weights(drive_id: str, dst: str = "model.pth"):
    """
    Download model weights from Google Drive to `dst` if not already present.
    """
    if not os.path.exists(dst):
        if drive_id.startswith("http"):
            url = drive_id
            if "/view" in url:
                url = url.split("/view")[0] + "/uc?export=download"
        else:
            url = f"https://drive.google.com/uc?export=download&id={drive_id}"
        gdown.download(url, dst, quiet=False)
    return dst

@st.cache_resource
def load_model_and_data():
    # Download model weights
    drive_id = st.secrets.get("drive_model_id")
    if not drive_id:
        st.error("Model Drive ID not found in secrets. Please set 'drive_model_id'.")
        st.stop()
        
    fetch_weights(drive_id)

    # Ensure model file exists
    if not os.path.exists("model.pth"):
        st.error("Model weights not found! Check if the download was successful.")
        st.stop()

    # Load model
    model, _ = load_model(model_path="model.pth", classes_file=CLASS_FILE)
    if model is None:
        st.error("Model could not be loaded. Check the model path or loading logic.")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

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
        else:
            st.success(f"Prediction: {label} ({confidence:.2f}%)")

            # Display nutrition info
            if label in nutrition_data:
                st.subheader("Nutritional Information per 100 gm")
                info = nutrition_data[label]
                st.write(f"- Calories: {info.get('calories', 'N/A')}")
                st.write(f"- Protein: {info.get('protein', 'N/A')}")
                st.write(f"- Fat: {info.get('fat', 'N/A')}")
                st.write(f"- Carbohydrates: {info.get('carbohydrate', 'N/A')}")
                st.write(f"- Energy: {info.get('energy', 'N/A')}")
            else:
                st.warning("No nutritional data available for this item.")

if __name__ == "__main__":
    main()

