# fantastic-palm-tree-cl

Contrastive geographic representation learning from NASA EPIC satellite imagery.

## Overview

Given a satellite image of Earth, can a model learn *where* it was taken without ever seeing explicit coordinates? This project learns an interpretable latent space where embedding distance corresponds to physical distance on Earth — purely from the visual structure of the images.

**Input**: 64x64 RGB satellite images from NASA's EPIC instrument at the L1 Lagrange point.
**Output**: A 128-d embedding where nearby embeddings = nearby locations on Earth.

## Approach

**Two-phase training**:

### Phase 1: Contrastive Pre-training

The encoder learns to structure its latent space geographically using **GeographicAlignmentLoss** — for every pair of images in a batch, the L2 distance between their embeddings is aligned with the Haversine (great-circle) distance between their capture locations. This preserves the full continuous geographic signal rather than thresholding into binary positive/negative pairs.

### Phase 2: Supervised Fine-tuning

A regression head is attached to the pre-trained encoder and fine-tuned on explicit (lon, lat) targets via MSE. The encoder can be frozen or fine-tuned (configurable).

## Architecture

```
Input (3x64x64)
  |
  v
GeographicEncoder
  Conv2d(3->64, 3x3) -> MaxPool4 -> Tanh
  Conv2d(64->128, 3x3) -> MaxPool4 -> Tanh
  Conv2d(128->256, 3x3) -> MaxPool4 -> Tanh
  Flatten -> FC(256->128) -> ReLU -> Dropout(0.2)
  |
  v
  128-d embedding (the interpretable latent space)
  |
  +---> ProjectionHead (contrastive only): FC(128->64)->ReLU->FC(64->32)->L2-norm
  |
  +---> RegressionHead (fine-tuning): FC(128->64)->ReLU->Dropout->FC(64->2)
```

## Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib pillow scipy scikit-learn cartopy tqdm
pip install tensorboard  # for training monitoring
```

### Download Data

```bash
# Download metadata + recent images from NASA EPIC API
python3 main.py setup

# Or download N days of images
python3 main.py download 7
```

### Train

```bash
# Phase 1: Contrastive pre-training (no coordinate labels needed)
python3 main.py pretrain

# Phase 2: Fine-tune on coordinates
python3 main.py finetune

# Or fine-tune with a specific encoder
python3 main.py finetune --encoder-path models/encoder_pretrained.pth

# Freeze the backbone during fine-tuning
python3 main.py finetune --freeze-backbone
```

### Evaluate

```bash
# Evaluate on test set
python3 main.py evaluate models/regressor_finetuned.pth

# Visualize predictions on world map
python3 main.py visualize models/regressor_finetuned.pth --n-clusters 12

# Test predictions with plots
python3 test_predictions.py --model_path models/regressor_finetuned.pth --num_samples 5
python3 test_predictions.py --model_path models/regressor_finetuned.pth --num_samples 50 --world_map
```

### TensorBoard

TensorBoard launches automatically during training (port 6006). Key metrics to watch:

| Tag | Description |
|-----|-------------|
| `pretrain/loss/train` | Geographic alignment loss |
| `pretrain/metrics/pos_dist` | Mean embedding distance for geographically close pairs |
| `pretrain/metrics/neg_dist` | Mean embedding distance for geographically distant pairs |
| `pretrain/metrics/pos_neg_ratio` | Ratio of negative to positive distances (higher = better separation) |
| `pretrain/metrics/geo_spearman_corr` | Spearman correlation between embedding and geographic distance |
| `finetune/loss/val` | MSE on normalized coordinates |
| `finetune/metrics/haversine_km` | Mean Haversine error in km |
| `latent_space/pretrain/latent_kmeans` | KMeans clusters projected onto world map |
| `Projector` tab | 128-d embeddings colored by true coordinates |

## Configuration

All defaults in `config.py`. Override via CLI or JSON config file:

```bash
python3 main.py pretrain --epochs 200 --batch-size 64 --lr 5e-4 --temperature 0.1
python3 main.py pretrain --config my_config.json
```

Key contrastive params:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.07 | Loss temperature (legacy, for SupCon) |
| `pos_threshold_km` | 500 | Haversine threshold for positive pairs (used in metrics) |
| `neg_threshold_km` | 5000 | Haversine threshold for negative pairs (used in metrics) |
| `projection_dim` | 32 | Projection head output dim |
| `freeze_backbone` | false | Freeze encoder during fine-tuning |

## Project Structure

```
fantastic-palm-tree-cl/
  main.py                    # CLI entry point
  config.py                  # Configuration dataclasses
  models.py                  # GeographicEncoder, ProjectionHead, RegressionHead
  losses.py                  # GeographicAlignmentLoss, Haversine utilities
  contrastive_dataset.py     # Dataset + dataloader creation
  contrastive_training.py    # Two-phase training orchestrator + TensorBoard
  data.py                    # NASA EPIC API downloader
  datasets.py                # SatelliteImageDataset, CoordinateNormalizer
  tensorboard_utils.py       # TensorBoard lifecycle management
  visualization.py           # Coordinate distribution plots
  visualize_representation.py # KMeans-on-world-map visualization
  evaluation_reporter.py     # Metric reporting utilities
  test_predictions.py        # Prediction testing + error maps
```

## Sister Project

[`fantastic-palm-tree`](https://github.com/klaibercore/fantastic-palm-tree) — end-to-end supervised coordinate regression on the same NASA EPIC data. This project asks: can self-supervised geographic pre-training improve on pure regression?
