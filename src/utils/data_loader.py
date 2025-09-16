import random

import torch
from torch.utils.data import DataLoader


def create_data_loaders(dataset, train_split, batch_size, num_workers=4):
    """
    Create train and validation data loaders from a dataset.

    Args:
        dataset: The full dataset to split
        train_split: Fraction of data to use for training (e.g., 0.8)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    random.shuffle(dataset.samples)
    split = int(train_split * len(dataset))

    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
