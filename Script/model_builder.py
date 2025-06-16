import torch.nn as nn
from torchvision.models import swin_b, Swin_B_Weights
def get_model(num_classes: int = 102, dropout_p: float = 0.2) -> nn.Module:
    """
    Build and return a Swin-B model with a custom classification head.

    Args:
        num_classes (int): Number of classes for the classifier.
        dropout_p (float): Dropout probability for the head.

    Returns:
        nn.Module: Swin-B model ready for training.
    """
    # Load pretrained weights
    weights = Swin_B_Weights.DEFAULT
    model = swin_b(weights=weights)
    # Replace the classification head
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model
