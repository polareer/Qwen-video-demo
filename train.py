"""
Training script for image classification
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import TRAIN_CONFIG
from models.nn_models import get_model
from utils.dataset import get_data_loader
from utils.training import train_epoch, evaluate, save_checkpoint


def main():
    # Setup
    device = torch.device(TRAIN_CONFIG["device"])
    print(f"Using device: {device}")

    # Data
    dataset_name = "cifar10"  # or "mnist"
    train_loader, test_loader = get_data_loader(
        dataset_name,
        TRAIN_CONFIG["batch_size"],
        TRAIN_CONFIG["num_workers"]
    )

    # Model
    num_classes = 10
    model = get_model("simplecnn", num_classes=num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    output_dir = "./outputs"
    best_acc = 0.0

    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{TRAIN_CONFIG['epochs']}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with accuracy: {best_acc:.2f}%")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, test_acc, output_dir)

    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()