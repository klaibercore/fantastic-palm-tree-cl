"""
PyTorch datasets for satellite image coordinate prediction.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SatelliteImageDataset(Dataset):
    """Dataset for satellite images with coordinate labels."""
    
    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
        random_seed: int = 42,
        image_size: int = 64,
        grayscale: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.metadata_dir = Path(metadata_dir)
        self.transform = transform
        self.split = split
        self.image_size = image_size
        self.grayscale = grayscale
        
        # Load all metadata and create image-coordinate pairs
        self.samples = self._load_samples()
        
        # Split data
        self.samples = self._split_data(train_split, val_split, random_seed)
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> Dict[str, List[Tuple[str, Tuple[float, float]]]]:
        """Load image paths and their coordinates, grouped by date."""
        samples_by_date = {}
        date_dir_count = 0

        if not self.image_dir.exists():
            logger.error(f"Images directory not found: {self.image_dir}")
            return samples_by_date

        for date_dir in sorted(self.image_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            date_dir_count += 1

            date = date_dir.name
            metadata_file = self.metadata_dir / f"{date}.json"

            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)

                coord_map = {}
                for item in data:
                    image_name = item.get("image")
                    coords = item.get("centroid_coordinates", {})

                    if image_name and coords.get("lat") is not None and coords.get("lon") is not None:
                        coord_map[image_name] = (float(coords["lat"]), float(coords["lon"]))

                date_samples = []
                for image_file in sorted(date_dir.glob("*.png")):
                    image_name = image_file.stem

                    if image_name in coord_map:
                        lat, lon = coord_map[image_name]
                        date_samples.append((str(image_file), (lat, lon)))

                if date_samples:
                    samples_by_date[date] = date_samples

            except Exception as e:
                logger.warning(f"Error processing {date}: {e}")

        total = sum(len(s) for s in samples_by_date.values())
        logger.info(f"Found {total} image-coordinate pairs across {len(samples_by_date)} dates ({date_dir_count} date dirs scanned)")
        return samples_by_date

    def _split_data(
        self,
        train_split: float,
        val_split: float,
        random_seed: int
    ) -> List[Tuple[str, Tuple[float, float]]]:
        """Split data at the date level to prevent temporal leakage between splits."""
        import random

        random.seed(random_seed)
        dates = sorted(self.samples.keys())
        random.shuffle(dates)

        n_dates = len(dates)
        n_train = max(1, int(n_dates * train_split))
        n_val = max(1, int(n_dates * val_split))

        if self.split == "train":
            split_dates = set(dates[:n_train])
        elif self.split == "val":
            split_dates = set(dates[n_train:n_train + n_val])
        elif self.split == "test":
            split_dates = set(dates[n_train + n_val:])
        else:
            raise ValueError(f"Unknown split: {self.split}")

        return [s for date in dates if date in split_dates for s in self.samples[date]]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        """Get an image and its coordinates."""
        image_path, (lat, lon) = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback, respecting config settings
            mode = 'L' if self.grayscale else 'RGB'
            image = Image.new(mode, (self.image_size, self.image_size), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = transforms.ToTensor()(image)
        
        # Convert coordinates to tensor [lon, lat] for consistent processing
        coords = torch.tensor([lon, lat], dtype=torch.float32)
        
        return image, coords


def create_transforms(image_size: int = 64, grayscale: bool = True) -> transforms.Compose:
    """Create image transformation pipeline."""
    transform_list = []
    
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    if grayscale:
        transform_list.append(transforms.Grayscale())
    
    transform_list.append(transforms.ToTensor())
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    config,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    # Determine optimal dataloader settings based on device
    import torch
    device = torch.device(config.training.device)
    
    # Optimize dataloader settings for MPS vs CUDA vs CPU
    if device.type == 'mps':
        # MPS has issues with multiprocessing and pin_memory
        num_workers = 0  # Single-threaded for MPS
        pin_memory = False
        persistent_workers = False
    elif device.type == 'cuda':
        # CUDA benefits from multiprocessing and pin_memory
        num_workers = min(num_workers, 4)  # Cap workers to prevent oversubscription
        pin_memory = True
        persistent_workers = True
    else:
        # CPU: moderate workers, no pin_memory needed
        num_workers = min(num_workers, 2)
        pin_memory = False
        persistent_workers = num_workers > 0
    
    logger.info(f"Using dataloader settings for {device.type}: "
                f"workers={num_workers}, pin_memory={pin_memory}")
    
    # Create transforms
    transform = create_transforms(
        image_size=config.data.image_size,
        grayscale=config.data.grayscale
    )
    
    # Create datasets
    train_dataset = SatelliteImageDataset(
        image_dir=config.data.images_dir,
        metadata_dir=config.data.combined_dir,
        transform=transform,
        split="train",
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        image_size=config.data.image_size,
        grayscale=config.data.grayscale,
    )

    val_dataset = SatelliteImageDataset(
        image_dir=config.data.images_dir,
        metadata_dir=config.data.combined_dir,
        transform=transform,
        split="val",
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        image_size=config.data.image_size,
        grayscale=config.data.grayscale,
    )

    test_dataset = SatelliteImageDataset(
        image_dir=config.data.images_dir,
        metadata_dir=config.data.combined_dir,
        transform=transform,
        split="test",
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        image_size=config.data.image_size,
        grayscale=config.data.grayscale,
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


class CoordinateNormalizer:
    """Normalizes and denormalizes coordinate values."""
    
    def __init__(self, dataset_coords=None):
        # Use full coordinate ranges or dataset-specific ranges
        if dataset_coords is not None:
            coords_array = dataset_coords.clone().detach()
            self.lon_min = coords_array[:, 0].min().item()
            self.lon_max = coords_array[:, 0].max().item()
            self.lat_min = coords_array[:, 1].min().item()
            self.lat_max = coords_array[:, 1].max().item()
        else:
            # Global coordinate ranges
            self.lon_min = -180.0
            self.lon_max = 180.0
            self.lat_min = -90.0
            self.lat_max = 90.0
    
    def normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [0, 1] range."""
        # coords shape: [batch_size, 2] where [:, 0] = lon, [:, 1] = lat
        lon = coords[:, 0]
        lat = coords[:, 1]
        
        # Normalize longitude from [-180, 180] to [0, 1]
        lon_norm = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        
        # Normalize latitude from [-90, 90] to [0, 1]
        lat_norm = (lat - self.lat_min) / (self.lat_max - self.lat_min)
        
        # Clamp to [0, 1] range to handle any edge cases
        lon_norm = torch.clamp(lon_norm, 0.0, 1.0)
        lat_norm = torch.clamp(lat_norm, 0.0, 1.0)
        
        return torch.stack([lon_norm, lat_norm], dim=1)
    
    def denormalize(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize coordinates from [0, 1] to original range."""
        lon_norm = coords_norm[:, 0]
        lat_norm = coords_norm[:, 1]
        
        lon = lon_norm * (self.lon_max - self.lon_min) + self.lon_min
        lat = lat_norm * (self.lat_max - self.lat_min) + self.lat_min
        
        return torch.stack([lon, lat], dim=1)
    
    def compute_longitude_error(self, pred_lon: torch.Tensor, true_lon: torch.Tensor) -> torch.Tensor:
        """Compute longitude error accounting for wraparound at ±180°."""
        # Calculate direct difference
        diff = torch.abs(pred_lon - true_lon)
        
        # Account for wraparound: if difference > 180°, use the shorter path
        wrap_diff = 360.0 - diff
        
        # Take the minimum of direct and wraparound differences
        lon_error = torch.minimum(diff, wrap_diff)
        
        return lon_error
    
    def compute_coordinate_error_degrees(self, pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
        """Compute coordinate prediction error in degrees with proper longitude handling."""
        # pred_coords and true_coords are in real-world coordinates [lon, lat]
        pred_lon, pred_lat = pred_coords[:, 0], pred_coords[:, 1]
        true_lon, true_lat = true_coords[:, 0], true_coords[:, 1]
        
        # Longitude error with wraparound handling
        lon_error = self.compute_longitude_error(pred_lon, true_lon)
        
        # Latitude error (no wraparound needed)
        lat_error = torch.abs(pred_lat - true_lat)
        
        # Euclidean distance
        distance_error = torch.sqrt(lon_error**2 + lat_error**2)
        
        return distance_error
    
    def compute_haversine_distance(self, pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
        """Compute Haversine distance between coordinates in kilometers."""
        # Convert degrees to radians
        pred_lon, pred_lat = pred_coords[:, 0], pred_coords[:, 1]
        true_lon, true_lat = true_coords[:, 0], true_coords[:, 1]
        
        pred_lon_rad = torch.deg2rad(pred_lon)
        pred_lat_rad = torch.deg2rad(pred_lat)
        true_lon_rad = torch.deg2rad(true_lon)
        true_lat_rad = torch.deg2rad(true_lat)
        
        # Earth radius in kilometers
        R = 6371.0
        
        # Haversine formula
        dlat = true_lat_rad - pred_lat_rad
        dlon = true_lon_rad - pred_lon_rad
        
        # Handle longitude wraparound in radians
        dlon = torch.where(dlon > torch.pi, dlon - 2*torch.pi, dlon)
        dlon = torch.where(dlon < -torch.pi, dlon + 2*torch.pi, dlon)
        
        a = torch.sin(dlat/2)**2 + torch.cos(pred_lat_rad) * torch.cos(true_lat_rad) * torch.sin(dlon/2)**2
        a = torch.clamp(a, max=1.0)  # Prevent NaN from float rounding
        c = 2 * torch.asin(torch.sqrt(a))
        
        distance = R * c
        return distance