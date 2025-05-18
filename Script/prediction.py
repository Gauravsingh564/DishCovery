import argparse
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import Swin_B_Weights
from model_builder import get_model

def predict_image(model, transform, image_path, device):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return img, pred.item(), conf.item()

def main():
    parser = argparse.ArgumentParser(description="Predict class for a single image and show nutrition info")
    parser.add_argument("--model-path",    type=str,   required=True, help="Path to saved model .pth file")
    parser.add_argument("--image-path",    type=str,   required=True, help="Path to input image file")
    parser.add_argument("--classes-file",  type=str,   required=True, help="Path to classes txt (one per line)")
    parser.add_argument("--nutrition-file",type=str,   required=True, help="Path to classes_nutrition.json")
    parser.add_argument("--threshold",     type=float, default=0.2,    help="Confidence threshold")
    args = parser.parse_args()

    # load class names
    with open(args.classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

    # load nutrition data
    with open(args.nutrition_file) as j:
        nutrition_data = json.load(j)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build & load model
    model = get_model(num_classes=len(class_names)).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    # get transforms
    weights   = Swin_B_Weights.DEFAULT
    transform = weights.transforms()

    # predict
    img, pred_idx, conf = predict_image(model, transform, args.image_path, device)
    pred_label = class_names[pred_idx].lower()

    # show image with title only
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

    # console output
    print(title)
    if conf < args.threshold or pred_label == "nofood":
        print("No food found in this image, please upload an appropriate image.")
        return

    info = nutrition_data.get(pred_label)
    if info:
        print("\nNutrition per 100 g:")
        for nutrient, val in info.items():
            print(f"  {nutrient.capitalize():12s}: {val}")
    else:
        print("Nutrition info not found for:", pred_label)

if __name__ == "__main__":
    main()
