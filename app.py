import os
import base64
import streamlit as st
from PIL import Image
import gdown
import torch
import json
from Script.prediction import load_model, predict_image

# ─── 1) Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DishCovery",
    page_icon="logo.png",
    # layout="centered"
)

# ─── 1a) Force all text to black via global CSS override ──────────────────────
st.markdown(
    """
    <style>
      /* apply black color to every element */
      .stApp, .stApp * {
        color: #000000 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── 0) Background setup ──────────────────────────────────────────────────────
def set_background(png_file: str):
    """Inject a base64‐encoded background image via CSS."""
    with open(png_file, "rb") as img:
        b64 = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# call background injection *after* config, before any UI components
set_background("background.jpg")

BASE       = os.path.dirname(__file__)
CLASS_FILE = os.path.join(BASE, "meta", "classes.txt")
NUT_FILE   = os.path.join(BASE, "meta", "classes_nutrition.json")
threshold  = 60

@st.cache_data
def fetch_weights(drive_id: str, dst: str = "model.pth"):
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
    drive_id = st.secrets.get("drive_model_id")
    if not drive_id:
        st.error("Model Drive ID not found in secrets. Please set 'drive_model_id'.")
        st.stop()
    fetch_weights(drive_id)

    if not os.path.exists("model.pth"):
        st.error("Model weights not found! Check if the download was successful.")
        st.stop()

    model, _ = load_model(model_path="model.pth", classes_file=CLASS_FILE)
    if model is None:
        st.error("Model could not be loaded. Check the model path or loading logic.")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(CLASS_FILE) as f:
        class_names = [line.strip() for line in f if line.strip()]

    with open(NUT_FILE) as f:
        nutrition_data = json.load(f)

    return model, device, class_names, nutrition_data

def main():
    # ─── 2) Header with logo + title ────────────────────────────────────────────
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image("logo.png", width=60)
    with col2:
        st.title("DishCovery")

    st.write(
        "Upload an image of a dish, and get its predicted label along with nutritional information."
    )

    uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model, device, class_names, nutrition_data = load_model_and_data()

    label, confidence = predict_image(model, device, image, class_names)

    if confidence <= threshold or label.lower() == "nofood":
        st.warning("No food detected in the image. Please upload an appropriate dish image.")
    else:
        st.success(f"Prediction: {label} ({confidence:.2f}%)")
        if label in nutrition_data:
            st.subheader("Nutritional Information per 100 gm")
            info = nutrition_data[label]
            st.write(f"- **Calories:**       {info.get('calories', 'N/A')}")
            st.write(f"- **Protein:**        {info.get('protein', 'N/A')}")
            st.write(f"- **Fat:**            {info.get('fat', 'N/A')}")
            st.write(f"- **Carbohydrates:**  {info.get('carbohydrate', 'N/A')}")
            st.write(f"- **Energy:**         {info.get('energy', 'N/A')}")
        else:
            st.warning("No nutritional data available for this item.")

if __name__ == "__main__":
    main()
