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
    `drive_id` can be either:
      - a file ID (the alphanumeric part from share link)
      - a full shareable URL (https://drive.google.com/file/d/.../view?usp=sharing)
    """
    if not os.path.exists(dst):
        # Determine download URL
        if drive_id.startswith("http"):
            # Convert share URL to direct download
            url = drive_id
            # If it ends with /view?usp=sharing, switch to export=download
            if "/view" in url:
                url = url.split("/view")[0] + "/uc?export=download"
        else:
            # drive_id is raw ID
            url = f"https://drive.google.com/uc?export=download&id={drive_id}"
        gdown.download(url, dst, quiet=False)
    return dst


def load_model(model_path: str = "model.pth", classes_file: str = None):
    """
    Builds and returns the model loaded with weights from `model_path`.
    Classifier head size is inferred from `classes_file`, defaulting to meta/classes.txt.
    """
    if classes_file is None:
        base = os.path.dirname(__file__)
        classes_file = os.path.join(base, "meta", "classes.txt")
    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)
    model = get_model(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model


def predict_image(model, device, img: Image.Image):
    """
    Predicts the class label and confidence percentage from a PIL Image.
    Returns (pred_label, confidence_percent).
    """
    weights = Swin_B_Weights.DEFAULT
    transform = weights.transforms()
    x = transform(img).unsqueeze(0).to(device)
    model.to(device).eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return idx.item(), conf.item()


def main():
    parser = argparse.ArgumentParser(
        description="Predict class for an image (fetching model from Drive if needed)"
    )
    parser.add_argument("--drive-id", type=str, default=None,
                        help="Google Drive file ID or shareable URL for model weights")
    parser.add_argument("--model-path", type=str, default="model.pth",
                        help="Local path to model weights (.pth)")
    parser.add_argument("--classes-file", type=str, default=None,
                        help="Path to classes txt (one per line)")
    parser.add_argument("--nutrition-file", type=str, default=None,
                        help="Path to classes_nutrition.json")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to input image file")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Confidence threshold (0-1) below which treat as no food")
    args = parser.parse_args()

    if args.drive_id:
        fetch_weights(args.drive_id, dst=args.model_path)
    base = os.path.dirname(__file__)
    classes_file = args.classes_file or os.path.join(base, "meta", "classes.txt")
    nutrition_file = args.nutrition_file or os.path.join(base, "meta", "classes_nutrition.json")

    model, class_names = load_model(model_path=args.model_path, classes_file=classes_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(nutrition_file) as j:
        nutrition_data = json.load(j)

    img = Image.open(args.image_path).convert("RGB")
    pred_idx, conf = predict_image(model, device, img)
    label = class_names[pred_idx]
    confidence = conf * 100

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

    print(title)
    if conf < args.threshold or label.lower() == "nofood":
        print("No food found in this image, please upload an appropriate image.")
        return

    info = nutrition_data.get(label.lower())
    if info:
        print("\nNutrition per 100 g:")
        for nutrient, val in info.items():
            print(f"  {nutrient.capitalize():12}: {val}")
    else:
        print("Nutrition info not found for:", label)

if __name__ == "__main__":
    main()
