import streamlit as st
from PIL import Image
import torch
import json
from prediction import load_model, predict_image  # assumes predict_image now accepts a PIL Image

# Cache model and data loading to speed up repeated inference
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load class names
    with open("classes.txt") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Load nutrition data from JSON
    with open("classes_nutrition.json") as f:
        nutrition_data = json.load(f)

    return model, device, class_names, nutrition_data


def main():
    st.title("🍽️ Food Classifier & Nutrition Facts")
    st.write(
        "Upload an image of a dish, and get its predicted label along with nutritional information."
    )

    # Allow user to adjust the confidence threshold for no-food detection
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

        # Predict label and confidence directly from PIL Image
        label, confidence = predict_image(model, device, image, class_names)

        # If confidence is below threshold or explicitly 'nofood', show warning
        if confidence <= threshold or label.lower() == "nofood":
            st.warning("No food detected in the image. Please upload an appropriate dish image.")
            st.image(image, use_column_width=True)
        else:
            st.success(f"Prediction: {label} ({confidence:.2f}%)")

            # Show nutrition info if available
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
