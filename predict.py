"""
Inference script for image classification
"""
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from models.nn_models import get_model


def load_model(model_path, model_name="simplecnn", num_classes=10, device="cpu"):
    """Load trained model"""
    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=32, dataset="cifar10"):
    """Preprocess image for inference"""
    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:  # mnist
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, device="cpu"):
    """Predict image class"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence


def main():
    # Configuration
    model_path = "./outputs/best_model.pth"
    image_path = "./test_image.jpg"
    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(model_path, num_classes=num_classes, device=device)
    print(f"Model loaded on {device}")

    # Preprocess and predict
    image_tensor = preprocess_image(image_path)
    predicted_class, confidence = predict(model, image_tensor, device)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()