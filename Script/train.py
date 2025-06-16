import argparse
import os
import time
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import Swin_B_Weights
from model_builder import get_model
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_correct = 0.0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == y).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc

def main():
    parser = argparse.ArgumentParser(description="Train Swin-B on a 102-class dataset")
    parser.add_argument("--train-dir", type=str, required=True, help="Path to training data (ImageFolder)")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test data (ImageFolder)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for train/test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="DataLoader workers")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save model")
    parser.add_argument("--num-classes", type=int, default=102, help="Number of output classes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Swin_B_Weights.DEFAULT
    transform = weights.transforms()

    train_ds = datasets.ImageFolder(args.train_dir, transform=transform)
    test_ds  = datasets.ImageFolder(args.test_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(num_classes=args.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Training completed in {(time.time() - start)/60:.2f} mins")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "swin_base_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()
