import argparse
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import Swin_B_Weights
from model_builder import get_model


def load_model():
    """
    Builds the model architecture based on the number of classes in classes.txt
    and loads weights from "model.pth" in the current working directory.
    """
    # Read class names to determine num_classes
    with open("classes.txt") as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)

    # Initialize and load model
    model = get_model(num_classes=num_classes)
    state = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state)
    return model


def predict_image(model, device, img: Image.Image):
    """
    Predicts the class index and confidence (0–1) from a PIL Image.
    """
    # Preprocessing from the pretrained weights
    weights = Swin_B_Weights.DEFAULT
    transform = weights.transforms()

    # Prepare input batch
    x = transform(img).unsqueeze(0).to(device)
    model.to(device).eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf_tensor, pred_idx = torch.max(probs, dim=1)

    return pred_idx.item(), conf_tensor.item()


def main():
    parser = argparse.ArgumentParser(
        description="Predict class for a single image and show nutrition info"
    )
    parser.add_argument(
        "--model-path", type=str, required=False,
        help="(unused) Path to saved model .pth file, defaults to 'model.pth'"
    )
    parser.add_argument(
        "--image-path", type=str, required=True,
        help="Path to input image file"
    )
    parser.add_argument(
        "--classes-file", type=str, required=True,
        help="Path to classes txt (one per line)"
    )
    parser.add_argument(
        "--nutrition-file", type=str, required=True,
        help="Path to classes_nutrition.json"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Confidence threshold (0-1) below which treat as no food"
    )
    args = parser.parse_args()

    # Load metadata files
    with open(args.classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    with open(args.nutrition_file) as j:
        nutrition_data = json.load(j)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (ignores --model-path, always uses 'model.pth')
    model = load_model().to(device)

    # Load and preprocess image
    img = Image.open(args.image_path).convert("RGB")

    # Predict
    pred_idx, conf = predict_image(model, device, img)
    pred_label = class_names[pred_idx].lower()

    # Plot result
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    ax.axis("off")
    if conf < args.threshold or pred_label == "nofood":
        title = "No Food"
    else:
        title = f"{pred_label} ({conf*100:.1f}%)"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # Console output
    print(title)
    if conf < args.threshold or pred_label == "nofood":
        print("No food found in this image, please upload an appropriate image.")
        return

    # Display nutrition
    info = nutrition_data.get(pred_label)
    if info:
        print("\nNutrition per 100 g:")
        for nutrient, val in info.items():
            print(f"  {nutrient.capitalize():12}: {val}")
    else:
        print("Nutrition info not found for:", pred_label)


if __name__ == "__main__":
    main()
