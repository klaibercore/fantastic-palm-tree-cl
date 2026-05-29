"""
Visualize the model's learned geographic representation.

Extracts 128-dim embeddings from the test set, clusters them with KMeans in
feature space (not on coordinates), and plots each test sample on a world map
colored by its cluster id. If clusters form contiguous geographic regions, the
model has learned geography in feature space.
"""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datasets import create_dataloaders
from test_predictions import (
    HAS_BASEMAP,
    HAS_CARTOPY,
    _draw_miller_map,
)

logger = logging.getLogger(__name__)


def _load_model_for_embeddings(config, model_path: str):
    """Load any model that exposes get_embeddings() for feature extraction.

    Tries (in order): ContrastiveModel (encoder only), FineTunedModel, then
    LocationRegressor (sister project). All return 128-d embeddings.
    """
    device = torch.device(config.training.device)
    errors = []

    # Try ContrastiveModel / bare encoder (from contrastive pre-training)
    try:
        from models import GeographicEncoder
        encoder = GeographicEncoder(
            input_channels=config.model.input_channels,
            conv_channels=config.model.conv_channels,
            kernel_size=config.model.kernel_size,
            pool_size=config.model.pool_size,
            activation=config.model.activation,
            hidden_dim=config.model.hidden_dim,
            image_size=config.data.image_size,
        )
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            encoder.to(device)
            encoder.eval()
            logger.info("Loaded ContrastiveModel encoder for visualization")
            return encoder
        # model_state_dict might contain encoder. prefix — extract it
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            encoder_state = {k.replace('encoder.', ''): v
                           for k, v in state.items() if k.startswith('encoder.')}
            if encoder_state:
                encoder.load_state_dict(encoder_state)
                encoder.to(device)
                encoder.eval()
                logger.info("Loaded encoder from FineTunedModel checkpoint for visualization")
                return encoder
    except Exception as e:
        errors.append(f"ContrastiveModel/encoder: {e}")

    # Try FineTunedModel (contrastive project, phase 2 output)
    try:
        from models import GeographicEncoder, RegressionHead, FineTunedModel
        encoder = GeographicEncoder(
            input_channels=config.model.input_channels,
            conv_channels=config.model.conv_channels,
            kernel_size=config.model.kernel_size,
            pool_size=config.model.pool_size,
            activation=config.model.activation,
            hidden_dim=config.model.hidden_dim,
            image_size=config.data.image_size,
        )
        regression = RegressionHead(
            input_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
        )
        model = FineTunedModel(encoder, regression)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        logger.info("Loaded FineTunedModel for visualization")
        return model
    except Exception as e:
        errors.append(f"FineTunedModel: {e}")

    # Fallback: LocationRegressor from sister project
    try:
        from models import create_location_regressor
        model = create_location_regressor(config)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        logger.info("Loaded LocationRegressor for visualization")
        return model
    except Exception as e:
        errors.append(f"LocationRegressor: {e}")

    raise RuntimeError(
        "Failed to load model for visualization:\n  " + "\n  ".join(errors)
    )
    return model


def extract_test_embeddings(model, test_loader, device):
    """Run model.get_embeddings() across the test loader.

    Returns (embeddings [N, 128], coords [N, 2] as [lon, lat] in degrees).
    """
    model.eval()
    all_embeddings = []
    all_coords = []
    with torch.no_grad():
        for images, coords in test_loader:
            images = images.to(device)
            emb = model.get_embeddings(images)
            all_embeddings.append(emb.cpu().numpy())
            all_coords.append(coords.numpy())
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_coords, axis=0)


def cluster_and_order(embeddings, n_clusters, seed):
    """Standardize, KMeans-cluster the embeddings, then reorder cluster ids by
    each centroid's PC1 score so adjacent ids correspond to adjacent
    feature-space directions (gives the categorical colormap continuity)."""
    scaled = StandardScaler().fit_transform(embeddings)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    raw_ids = km.fit_predict(scaled)
    pca = PCA(n_components=1, random_state=seed)
    pca.fit(scaled)
    centroid_pc1 = pca.transform(km.cluster_centers_).ravel()
    order = np.argsort(centroid_pc1)
    remap = np.empty_like(order)
    remap[order] = np.arange(n_clusters)
    return remap[raw_ids]


def plot_clusters_on_world_map(coords, cluster_ids, n_clusters, output_path):
    """Headline panel: scatter (lon, lat) on a world map colored by cluster id.
    Auto-falls-back to a plain matplotlib axes if no map backend available."""
    cmap_name = 'tab20' if n_clusters > 10 else 'tab10'
    cmap = plt.get_cmap(cmap_name, n_clusters)
    has_map = HAS_CARTOPY or HAS_BASEMAP

    fig = plt.figure(figsize=(16, 10))

    if has_map:
        backend, is_cartopy = _draw_miller_map()
        if is_cartopy:
            import cartopy.crs as ccrs
            scatter = backend.scatter(
                coords[:, 0], coords[:, 1],
                c=cluster_ids, cmap=cmap,
                vmin=-0.5, vmax=n_clusters - 0.5,
                s=35, alpha=0.85, edgecolors='black', linewidth=0.3,
                zorder=5, transform=ccrs.PlateCarree(),
            )
            backend.gridlines(draw_labels=True)
        else:
            x, y = backend(coords[:, 0], coords[:, 1])
            scatter = backend.scatter(
                x, y, c=cluster_ids, cmap=cmap,
                vmin=-0.5, vmax=n_clusters - 0.5,
                s=35, alpha=0.85, edgecolors='black', linewidth=0.3, zorder=5,
            )
            backend.drawmeridians(np.arange(-180, 181, 60),
                                  labels=[0, 0, 0, 1], fontsize=10)
            backend.drawparallels(np.arange(-90, 91, 30),
                                  labels=[1, 0, 0, 0], fontsize=10)
    else:
        logger.warning("Neither cartopy nor basemap installed; "
                       "falling back to plain scatter.")
        ax = plt.gca()
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=cluster_ids, cmap=cmap,
            vmin=-0.5, vmax=n_clusters - 0.5,
            s=35, alpha=0.85, edgecolors='black', linewidth=0.3,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05,
                        shrink=0.7, ticks=np.arange(n_clusters))
    cbar.set_label(
        f'Feature-space cluster id (KMeans, k={n_clusters}, ordered by PC1)',
        fontsize=12,
    )

    plt.title(
        f'Learned Geographic Representation\n'
        f'{len(cluster_ids)} test samples colored by KMeans clusters of 128-d embeddings',
        fontsize=14, fontweight='bold',
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Representation plot saved to {output_path}")
    return output_path


def visualize_representation(config, model_path, n_clusters=8,
                             seed=42, output_dir='outputs'):
    """Load model, extract test embeddings, cluster, render the world-map figure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(config.training.device)

    logger.info(f"Loading model from {model_path}")
    model = _load_model_for_embeddings(config, model_path)

    _, _, test_loader = create_dataloaders(config, batch_size=32, num_workers=0)
    n_samples = len(test_loader.dataset)
    logger.info(f"Extracting embeddings on {n_samples} test samples...")
    embeddings, coords = extract_test_embeddings(model, test_loader, device)

    effective_k = max(2, min(n_clusters, n_samples // 5))
    if effective_k != n_clusters:
        logger.warning(
            f"Reducing n_clusters from {n_clusters} to {effective_k} "
            f"(only {n_samples} test samples available)"
        )

    logger.info(f"Running KMeans(k={effective_k}) on 128-d embeddings...")
    cluster_ids = cluster_and_order(embeddings, effective_k, seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(Path(output_dir) / f"representation_{timestamp}.png")
    return plot_clusters_on_world_map(coords, cluster_ids, effective_k, output_path)
