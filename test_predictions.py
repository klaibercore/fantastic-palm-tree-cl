#!/usr/bin/env python3
"""
Test model predictions on satellite images - single, multiple, or world map.
"""

import argparse
import math
import random
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

from PIL import Image

from config import Config
from datasets import create_dataloaders
from models import create_location_regressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map backend selection: cartopy (modern) > basemap (legacy) > none
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    from mpl_toolkits.basemap import Basemap
    HAS_BASEMAP = True
except ImportError:
    HAS_BASEMAP = False

if not HAS_CARTOPY and not HAS_BASEMAP:
    logger.warning("Neither cartopy nor basemap installed. Map plots will be unavailable.")
elif HAS_BASEMAP and not HAS_CARTOPY:
    logger.warning("Basemap is deprecated. Install cartopy for better map support.")


def load_model(config, model_path: str):
    """Load trained model from checkpoint or raw state dict."""
    model = create_location_regressor(config)
    checkpoint = torch.load(model_path, map_location=config.training.device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(torch.device(config.training.device))
    return model


def predict_single_image(model, image, true_coords, config):
    """Predict coordinates for a single image.

    The model outputs raw [lon, lat] coordinates (same space as training targets),
    so no denormalization is needed.
    """
    device = torch.device(config.training.device)

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    true_coords = true_coords.to(device)

    with torch.no_grad():
        prediction = model(image)

    pred_coords_np = prediction.squeeze().cpu().numpy()
    true_coords_np = true_coords.squeeze().cpu().numpy()

    return pred_coords_np, true_coords_np


def haversine_km(coord1, coord2):
    """Calculate Haversine distance between two [lon, lat] coordinates in km."""
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    if dlon > math.pi:
        dlon -= 2 * math.pi
    elif dlon < -math.pi:
        dlon += 2 * math.pi

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c


def load_original_image(image_path):
    """Load the original full-size RGB image from disk."""
    return np.array(Image.open(image_path).convert('RGB'))


def _draw_miller_map(ax=None):
    """Set up a Miller-projection world map. Returns (ax, is_cartopy)."""
    if not HAS_CARTOPY and not HAS_BASEMAP:
        raise RuntimeError("Install cartopy or basemap for map plots")

    if HAS_CARTOPY:
        if ax is None:
            ax = plt.gca()
        ax = plt.subplot(projection=ccrs.Miller())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        return ax, True
    else:
        m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=85,
                    llcrnrlon=-180, urcrnrlon=180, resolution='c')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.25)
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawmapboundary(fill_color='lightblue')
        return m, False


def _draw_map_point(backend, lon, lat, is_cartopy, **scatter_kwargs):
    """Plot a point on either cartopy or basemap backend."""
    if is_cartopy:
        backend.scatter(lon, lat, transform=ccrs.PlateCarree(), **scatter_kwargs)
    else:
        x, y = backend(lon, lat)
        backend.scatter(x, y, **scatter_kwargs)


def _draw_map_line(backend, lons, lats, is_cartopy, **plot_kwargs):
    """Draw a line on either cartopy or basemap backend."""
    if is_cartopy:
        backend.plot(lons, lats, transform=ccrs.PlateCarree(), **plot_kwargs)
    else:
        x, y = backend(lons, lats)
        backend.plot(x, y, **plot_kwargs)


def plot_single(image_np, true_coords, pred_coords, save_path=None, show_plot=True):
    """Plot image alongside prediction on world map."""
    img_np = image_np

    fig = plt.figure(figsize=(16, 8))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img_np)
    ax1.set_title('Satellite Image')
    ax1.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    backend, is_cartopy = _draw_miller_map(ax2)

    true_lon, true_lat = true_coords[0], true_coords[1]
    pred_lon, pred_lat = pred_coords[0], pred_coords[1]

    _draw_map_point(backend, true_lon, true_lat, is_cartopy,
                    marker='o', color='green', s=100, edgecolors='black',
                    linewidth=2, zorder=5,
                    label=f'True: ({true_lon:.2f}, {true_lat:.2f})')

    _draw_map_point(backend, pred_lon, pred_lat, is_cartopy,
                    marker='x', color='red', s=100, linewidths=3, zorder=5,
                    label=f'Pred: ({pred_lon:.2f}, {pred_lat:.2f})')

    error_km = haversine_km(true_coords, pred_coords)
    _draw_map_line(backend, [true_lon, pred_lon], [true_lat, pred_lat], is_cartopy,
                   'b--', alpha=0.7, linewidth=2, label=f'Error: {error_km:.1f} km')

    if not is_cartopy:
        backend.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1], fontsize=10)
        backend.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=10)
    else:
        backend.gridlines(draw_labels=True)

    ax2.set_title('Coordinate Prediction')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_multiple(images, true_list, pred_list, save_path=None, show_plot=True):
    """Plot multiple image predictions with world maps."""
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 4 * rows))

    for i in range(n):
        ax_img = plt.subplot(rows, cols * 2, i * 2 + 1)
        img_np = images[i]
        ax_img.imshow(img_np)
        ax_img.set_title(f'Image {i + 1}')
        ax_img.axis('off')

        ax_map = plt.subplot(rows, cols * 2, i * 2 + 2)
        backend, is_cartopy = _draw_miller_map(ax_map)

        true_lon, true_lat = true_list[i][0], true_list[i][1]
        pred_lon, pred_lat = pred_list[i][0], pred_list[i][1]

        _draw_map_point(backend, true_lon, true_lat, is_cartopy,
                        marker='o', color='green', s=80, edgecolors='black',
                        linewidth=1.5, zorder=5, label='True')

        _draw_map_point(backend, pred_lon, pred_lat, is_cartopy,
                        marker='x', color='red', s=80, linewidths=2, zorder=5,
                        label='Pred')

        error_km = haversine_km(true_list[i], pred_list[i])
        _draw_map_line(backend, [true_lon, pred_lon], [true_lat, pred_lat], is_cartopy,
                       'b--', alpha=0.7, linewidth=1.5)

        ax_map.text(0.02, 0.98, f'{error_km:.0f} km',
                    transform=ax_map.transAxes, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), va='top')
        ax_map.set_title(f'Map {i + 1}')
        if i == 0:
            ax_map.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_world_error_map(true_list, pred_list, save_path=None, show_plot=True):
    """Plot all predictions on a world map with dots colored by error magnitude."""
    errors = np.array([haversine_km(t, p) for t, p in zip(true_list, pred_list)])
    true_arr = np.array(true_list)  # [N, 2] with [lon, lat]

    fig = plt.figure(figsize=(16, 10))
    backend, is_cartopy = _draw_miller_map()

    if is_cartopy:
        scatter = backend.scatter(true_arr[:, 0], true_arr[:, 1],
                                   c=errors, cmap='RdYlGn_r', s=40, alpha=0.8,
                                   edgecolors='black', linewidth=0.3, zorder=5,
                                   transform=ccrs.PlateCarree())
        backend.gridlines(draw_labels=True)
    else:
        x, y = backend(true_arr[:, 0], true_arr[:, 1])
        scatter = backend.scatter(x, y, c=errors, cmap='RdYlGn_r', s=40, alpha=0.8,
                                  edgecolors='black', linewidth=0.3, zorder=5)
        backend.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1], fontsize=10)
        backend.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=10)

    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label('Prediction Error (km)', fontsize=12)

    plt.title(f'Prediction Errors ({len(errors)} samples)\n'
              f'Mean: {errors.mean():.0f} km | Median: {np.median(errors):.0f} km | '
              f'Min: {errors.min():.0f} km | Max: {errors.max():.0f} km',
              fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"World error map saved to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test model predictions on satellite images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of test samples (1 for single, >1 for multiple)")
    parser.add_argument("--world_map", action="store_true",
                        help="Generate world map with predictions colored by error")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    args = parser.parse_args()

    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = Config.from_dict(json.load(f))
    else:
        config = Config()

    model = load_model(config, args.model_path)
    _, _, test_loader = create_dataloaders(config, batch_size=1, num_workers=0)

    # Access the underlying dataset to get image file paths for full-size display
    test_dataset = test_loader.dataset
    indices = list(range(len(test_dataset)))
    random.shuffle(indices)
    indices = indices[:args.num_samples]

    images, true_list, pred_list = [], [], []

    for i, idx in enumerate(indices):
        image_path = test_dataset.samples[idx][0]
        image_tensor, true_coords = test_dataset[idx]
        pred, true_np = predict_single_image(model, image_tensor, true_coords, config)
        images.append(load_original_image(image_path))
        true_list.append(true_np)
        pred_list.append(pred)

        error_km = haversine_km(true_np, pred)
        error_deg = np.sqrt((true_np[0] - pred[0]) ** 2 + (true_np[1] - pred[1]) ** 2)
        print(f"Sample {i + 1}: True=({true_np[0]:.2f}, {true_np[1]:.2f})  "
              f"Pred=({pred[0]:.2f}, {pred[1]:.2f})  Error={error_km:.1f} km ({error_deg:.3f} deg)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.world_map:
        save_path = output_dir / f"world_error_map_{args.num_samples}.png"
        plot_world_error_map(true_list, pred_list,
                             save_path=str(save_path), show_plot=args.show)
    elif args.num_samples == 1:
        save_path = output_dir / "single_prediction_test.png"
        plot_single(images[0], true_list[0], pred_list[0],
                    save_path=str(save_path), show_plot=args.show)
    else:
        save_path = output_dir / f"multiple_predictions_{args.num_samples}.png"
        plot_multiple(images, true_list, pred_list,
                      save_path=str(save_path), show_plot=args.show)

    errors = [haversine_km(true_list[i], pred_list[i]) for i in range(len(pred_list))]
    print(f"\nMean={np.mean(errors):.1f} km  Median={np.median(errors):.1f} km  "
          f"Min={np.min(errors):.1f} km  Max={np.max(errors):.1f} km")


if __name__ == "__main__":
    main()
