import os
import argparse
import json
import torch
import gdown
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import Swin_B_Weights
from Script.model_builder import get_model


def fetch_weights(drive_id: str, dst: str = "model.pth"):
    """
    Download model weights from Google Drive to `dst` if not already present.
    `drive_id` is the file ID from your Drive share link.
    """
    if not os.path.exists(dst):
        url = f"https://drive.google.com/file/d/1Sh447_nPFg8WMIzX9k2ryQXZhbqEU42C/view?usp=sharing={drive_id}"
        gdown.download(url, dst, quiet=False)
    return dst


def load_model(model_path: str = "model.pth", classes_file: str = None):
    """
    Builds and returns the model loaded with weights from `model_path`.
    Classifier head size is inferred from `classes_file`, defaulting to meta/classes.txt.
    """
    # Determine the classes file location
    if classes_file is None:
        base = os.path.dirname(__file__)
        classes_file = os.path.join(base, "meta", "classes.txt")

    # Read class labels
    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)

    # Initialize model architecture
    model = get_model(num_classes=num_classes)

    # Load weights (set weights_only=False to unpickle custom objects)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model, class_names


def predict_image(model, device, img: Image.Image):
    """
    Predicts the class index and confidence (0–1) from a PIL Image.
    Returns (pred_idx, confidence).
    """
    # Preprocessing from pretrained weights
    weights = Swin_B_Weights.DEFAULT
    transform = weights.transforms()

    x = transform(img).unsqueeze(0).to(device)
    model.to(device).eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf_tensor, pred_idx = torch.max(probs, dim=1)
    return pred_idx.item(), conf_tensor.item()


def main():
    parser = argparse.ArgumentParser(
        description="Predict class for an image (fetching model from Drive if needed)"
    )
    parser.add_argument(
        "--drive-id", type=str, default=None,
        help="Google Drive file ID for model weights (optional)"
    )
    parser.add_argument(
        "--model-path", type=str, default="model.pth",
        help="Local path to model weights (.pth)"
    )
    parser.add_argument(
        "--classes-file", type=str, default=None,
        help="Path to classes txt (one per line)"
    )
    parser.add_argument(
        "--nutrition-file", type=str, default=None,
        help="Path to classes_nutrition.json"
    )
    parser.add_argument(
        "--image-path", type=str, required=True,
        help="Path to input image file"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Confidence threshold (0-1) below which treat as no food"
    )
    args = parser.parse_args()

    # Fetch weights if drive_id provided
    if args.drive_id:
        fetch_weights(args.drive_id, dst=args.model_path)

    # Determine metadata paths
    base = os.path.dirname(__file__)
    classes_file = args.classes_file or os.path.join(base, "meta", "classes.txt")
    nutrition_file = args.nutrition_file or os.path.join(base, "meta", "classes_nutrition.json")

    # Load model and class names
    model, class_names = load_model(model_path=args.model_path, classes_file=classes_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load nutrition data
    with open(nutrition_file) as j:
        nutrition_data = json.load(j)

    # Load and preprocess image
    img = Image.open(args.image_path).convert("RGB")

    # Predict
    pred_idx, conf = predict_image(model, device, img)
    label = class_names[pred_idx]
    confidence = conf * 100

    # Display image with prediction
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    ax.axis("off")
    if conf < args.threshold or label.lower() == "nofood":
        title = "No Food"
    else:
        title = f"{label} ({confidence:.1f}%)"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # Console output
    print(title)
    if conf < args.threshold or label.lower() == "nofood":
        print("No food found in this image, please upload an appropriate image.")
        return

    # Display nutrition info
    info = nutrition_data.get(label.lower(), None)
    if info:
        print("\nNutrition per 100 g:")
        for nutrient, val in info.items():
            print(f"  {nutrient.capitalize():12}: {val}")
    else:
        print("Nutrition info not found for:", label)


if __name__ == "__main__":
    main()
