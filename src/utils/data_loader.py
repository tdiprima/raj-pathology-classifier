import torch
from torch.utils.data import DataLoader


def create_data_loaders_from_separate_datasets(
    train_dataset, val_dataset, batch_size, num_workers=4
):
    """
    Create train and validation data loaders from separate pre-split datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
