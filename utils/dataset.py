"""
Dataset loading utilities
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImageDataLoader:
    def __init__(self, config):
        self.config = config
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_cifar10_loaders(self, batch_size, num_workers=4):
        """Load CIFAR10 dataset"""
        train_dataset = datasets.CIFAR10(
            root=self.config.get("data_dir", "./data"),
            train=True,
            download=True,
            transform=self.train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=self.config.get("data_dir", "./data"),
            train=False,
            download=True,
            transform=self.test_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, test_loader

    def get_mnist_loaders(self, batch_size, num_workers=4):
        """Load MNIST dataset"""
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.MNIST(
            root=self.config.get("data_dir", "./data"),
            train=True,
            download=True,
            transform=self.train_transform
        )
        test_dataset = datasets.MNIST(
            root=self.config.get("data_dir", "./data"),
            train=False,
            download=True,
            transform=self.test_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, test_loader

def get_data_loader(dataset_name, batch_size, num_workers=4):
    """Factory function to get data loader"""
    loader = ImageDataLoader({})
    if dataset_name.lower() == "cifar10":
        return loader.get_cifar10_loaders(batch_size, num_workers)
    elif dataset_name.lower() == "mnist":
        return loader.get_mnist_loaders(batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")