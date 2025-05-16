import argparse
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
    parser = argparse.ArgumentParser(description="Predict class for a single image")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model .pth file")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image file")
    parser.add_argument("--classes-file", type=str, required=True, help="Path to classes txt (one per line)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Confidence threshold")
    args = parser.parse_args()

    with open(args.classes_file) as f:
        class_names = [line.strip() for line in f.readlines()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=len(class_names)).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    weights = Swin_B_Weights.DEFAULT
    transform = weights.transforms()

    img, pred_idx, conf = predict_image(model, transform, args.image_path, device)
    pred_label = class_names[pred_idx]

    plt.imshow(img)
    title = f"{pred_label} ({conf*100:.2f}%)"
    if conf < args.threshold or pred_label.lower() == "nofood":
        title = "NoFood"
        plt.title(title)
        plt.axis('off')
        plt.show()
        print("No Food found in this image please upload appropriate image..")
    else:
        plt.title(title)
        plt.axis('off')
        plt.show()
        print(title)

if __name__ == "__main__":
    main()
