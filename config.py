"""
Configuration management for contrastive geographic image representation.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Configuration for data processing."""
    raw_data_dir: str = "raw_data"
    images_dir: str = "images"
    combined_dir: str = "combined"

    # Image processing
    image_size: int = 64
    grayscale: bool = False

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # NASA EPIC API
    api_base_url: str = "https://epic.gsfc.nasa.gov/api/natural"


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    input_channels: int = 3
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_size: int = 3
    pool_size: int = 4
    activation: str = "tanh"

    # Embedding
    hidden_dim: int = 128
    output_dim: int = 2


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive pre-training."""
    temperature: float = 0.07
    pos_threshold_km: float = 500.0
    neg_threshold_km: float = 5000.0
    projection_dim: int = 32
    embedding_log_every: int = 10
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    device: str = "auto"

    # Optimization
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    loss_function: str = "mse"
    scheduler: str = "step"
    step_size: int = 20
    gamma: float = 0.5
    max_epochs: int = 100
    gradient_clipping: float = 0.0

    # Logging
    log_dir: str = "logs/tensorboard"
    save_dir: str = "models"

    # TensorBoard
    launch_tensorboard: bool = True
    tensorboard_port: int = 6006
    open_browser: bool = False

    # Hardware
    num_threads: int = 16

    # Reproducibility
    random_seed: int = 42


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if self.training.device == "auto":
            self.training.device = self._get_best_device()

        if self.data.grayscale and self.model.input_channels != 1:
            raise ValueError(
                f"config.data.grayscale=True requires config.model.input_channels=1 "
                f"(got {self.model.input_channels})"
            )
        if not self.data.grayscale and self.model.input_channels != 3:
            raise ValueError(
                f"config.data.grayscale=False requires config.model.input_channels=3 "
                f"(got {self.model.input_channels})"
            )

    def _get_best_device(self) -> str:
        import torch
        import logging
        logger = logging.getLogger(__name__)

        if torch.cuda.is_available():
            logger.info("CUDA GPU detected - using CUDA acceleration")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon GPU detected - using MPS acceleration")
            return "mps"
        else:
            logger.info("No GPU detected - using CPU")
            return "cpu"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            contrastive=ContrastiveConfig(**config_dict.get("contrastive", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
        )
