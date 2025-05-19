import argparse
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import Swin_B_Weights
from model_builder import get_model


def predict_image(model, transform, img: Image.Image, device):
    """
    Predicts the class index and confidence from a PIL Image.
    """
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()


def main():
    parser = argparse.ArgumentParser(description="Predict class for a single image and show nutrition info")
    parser.add_argument("--model-path",    type=str,   required=True, help="Path to saved model .pth file")
    parser.add_argument("--image-path",    type=str,   required=True, help="Path to input image file or directory")
    parser.add_argument("--classes-file",  type=str,   required=True, help="Path to classes txt (one per line)")
    parser.add_argument("--nutrition-file",type=str,   required=True, help="Path to classes_nutrition.json")
    parser.add_argument("--threshold",     type=float, default=0.2,    help="Confidence threshold (0-1)")
    args = parser.parse_args()

    # Load class names
    with open(args.classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Load nutrition data
    with open(args.nutrition_file) as j:
        nutrition_data = json.load(j)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build & load model
    model = get_model(num_classes=len(class_names)).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    # Get transforms
    weights   = Swin_B_Weights.DEFAULT
    transform = weights.transforms()

    # Open image
    img = Image.open(args.image_path).convert("RGB")

    # Predict
    pred_idx, conf = predict_image(model, transform, img, device)
    pred_label = class_names[pred_idx].lower()

    # Display image with title
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

    info = nutrition_data.get(pred_label)
    if info:
        print("\nNutrition per 100 g:")
        for nutrient, val in info.items():
            print(f"  {nutrient.capitalize():12}: {val}")
    else:
        print("Nutrition info not found for:", pred_label)


if __name__ == "__main__":
    main()
