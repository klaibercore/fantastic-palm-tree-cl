"""
Neural network models for contrastive geographic image representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class GeographicEncoder(nn.Module):
    """CNN backbone that encodes satellite images into a 128-d embedding.

    Architecture mirrors the sister project's LocationRegressor feature
    extractor so pre-trained weights are directly comparable.
    """

    def __init__(
        self,
        input_channels: int = 3,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        pool_size: int = 4,
        activation: str = "tanh",
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [64, 128, 256]

        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.activation_name = activation.lower()

        # Build conv blocks: Conv2d -> MaxPool -> Activation
        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        for out_ch in conv_channels:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=1),
                nn.MaxPool2d(pool_size),
                self._get_activation(),
            )
            self.conv_layers.append(block)
            in_ch = out_ch

        # Flattened size after 3 MaxPool4 ops on 64x64 input
        h = 64 // (pool_size ** len(conv_channels))
        w = 64 // (pool_size ** len(conv_channels))
        self.flattened_size = h * w * conv_channels[-1]

        # Embedding layer (before final projection)
        self.embedding = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self._initialize_weights()

    def _get_activation(self) -> nn.Module:
        if self.activation_name == "tanh":
            return nn.Tanh()
        elif self.activation_name == "relu":
            return nn.ReLU()
        elif self.activation_name in ("leaky_relu", "leakyrelu"):
            return nn.LeakyReLU(0.2)
        raise ValueError(f"Unknown activation: {self.activation_name}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_layers:
            x = conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return 128-d embeddings (same as forward, for compatibility)."""
        return self.forward(x)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Projects the 128-d encoder embedding to a lower-dimensional space
    where the contrastive loss is applied. L2-normalized output.
    Discarded after pre-training.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=1)


class RegressionHead(nn.Module):
    """Regression head for fine-tuning on (lon, lat) coordinates.

    Attached to the GeographicEncoder after contrastive pre-training.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 2,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveModel(nn.Module):
    """Full contrastive model: encoder + projection head."""

    def __init__(self, encoder: GeographicEncoder, projection: ProjectionHead):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        projections = self.projection(embeddings)
        return projections

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-d embeddings (before projection head)."""
        return self.encoder(x)


class FineTunedModel(nn.Module):
    """Fine-tuned model: encoder + regression head."""

    def __init__(self, encoder: GeographicEncoder, regression: RegressionHead):
        super().__init__()
        self.encoder = encoder
        self.regression = regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        return self.regression(embeddings)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def create_location_regressor(config):
    """Create a FineTunedModel (encoder + regression head).

    Compatibility wrapper matching the sister project's API — returns a model
    that takes images and outputs [lon, lat], with get_embeddings() support.
    """
    encoder = GeographicEncoder(
        input_channels=config.model.input_channels,
        conv_channels=config.model.conv_channels,
        kernel_size=config.model.kernel_size,
        pool_size=config.model.pool_size,
        activation=config.model.activation,
        hidden_dim=config.model.hidden_dim,
    )
    regression = RegressionHead(
        input_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
    )
    return FineTunedModel(encoder, regression)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_size: tuple = (1, 3, 64, 64)):
    try:
        from torchinfo import summary
        summary(model, input_size=input_size)
    except ImportError:
        print(f"Model: {model.__class__.__name__}")
        print(f"Trainable parameters: {count_parameters(model):,}")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                print(f"  {name}: {module}")
