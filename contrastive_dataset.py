"""
Contrastive dataset that wraps SatelliteImageDataset and provides
geographic labels for supervised contrastive learning.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple

from datasets import SatelliteImageDataset


class ContrastiveDataset(Dataset):
    """Wraps SatelliteImageDataset to return (image, coords) tuples.

    The contrastive trainer uses the coordinate labels to define
    positive/negative pairs via Haversine distance — the dataset
    itself just returns images with their known coordinates.
    """

    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
        random_seed: int = 42,
        image_size: int = 64,
        grayscale: bool = False,
    ):
        self.base_dataset = SatelliteImageDataset(
            image_dir=image_dir,
            metadata_dir=metadata_dir,
            split=split,
            train_split=train_split,
            val_split=val_split,
            random_seed=random_seed,
            image_size=image_size,
            grayscale=grayscale,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (image, coords) where coords = [lon, lat] in degrees."""
        return self.base_dataset[idx]


def create_contrastive_dataloaders(
    config,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Create train, val, test dataloaders for contrastive pre-training.

    Uses the same MPS-aware settings as the sister project.
    All three loaders return (images, coords) so the trainer can
    compute geographic pair labels on each batch.
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(config.training.device)

    if device.type == 'mps':
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    elif device.type == 'cuda':
        num_workers = min(num_workers, 4)
        pin_memory = True
        persistent_workers = True
    else:
        num_workers = min(num_workers, 2)
        pin_memory = False
        persistent_workers = num_workers > 0

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Contrastive dataloader settings for {device.type}: "
                f"workers={num_workers}, pin_memory={pin_memory}")

    # Create transforms
    from datasets import create_transforms
    transform = create_transforms(
        image_size=config.data.image_size,
        grayscale=config.data.grayscale,
    )

    # Create datasets — ContrastiveDataset directly, not wrapped SatelliteImageDataset
    common_kwargs = dict(
        image_dir=config.data.images_dir,
        metadata_dir=config.data.combined_dir,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        random_seed=config.training.random_seed,
        image_size=config.data.image_size,
        grayscale=config.data.grayscale,
    )

    # Override transforms on the base dataset since ContrastiveDataset delegates
    train_base = SatelliteImageDataset(transform=transform, split="train", **{
        k: v for k, v in common_kwargs.items() if k in SatelliteImageDataset.__init__.__code__.co_varnames
    })

    val_base = SatelliteImageDataset(transform=transform, split="val", **{
        k: v for k, v in common_kwargs.items() if k in SatelliteImageDataset.__init__.__code__.co_varnames
    })

    test_base = SatelliteImageDataset(transform=transform, split="test", **{
        k: v for k, v in common_kwargs.items() if k in SatelliteImageDataset.__init__.__code__.co_varnames
    })

    dl_kwargs_train = dict(
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    dl_kwargs_eval = dict(
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    train_loader = DataLoader(train_base, **dl_kwargs_train)
    val_loader = DataLoader(val_base, **dl_kwargs_eval)
    test_loader = DataLoader(test_base, **dl_kwargs_eval)

    return train_loader, val_loader, test_loader


def create_full_contrastive_loader(config, batch_size: int = 32, num_workers: int = 4):
    """Create a single DataLoader with all available data (no train/val/test split).

    Used for contrastive pre-training where we want to maximize the number of
    positive/negative geographic pairs.
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(config.training.device)

    if device.type == 'mps':
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    elif device.type == 'cuda':
        num_workers = min(num_workers, 4)
        pin_memory = True
        persistent_workers = True
    else:
        num_workers = min(num_workers, 2)
        pin_memory = False
        persistent_workers = num_workers > 0

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Full contrastive dataloader for {device.type}: "
                f"workers={num_workers}, pin_memory={pin_memory}")

    from datasets import create_transforms
    transform = create_transforms(
        image_size=config.data.image_size,
        grayscale=config.data.grayscale,
    )

    # Put all data into the train split
    full_dataset = SatelliteImageDataset(
        image_dir=config.data.images_dir,
        metadata_dir=config.data.combined_dir,
        transform=transform,
        split="train",
        train_split=1.0,
        val_split=0.0,
        random_seed=config.training.random_seed,
        image_size=config.data.image_size,
        grayscale=config.data.grayscale,
    )

    logger.info(f"Full dataset: {len(full_dataset)} samples")

    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return loader
