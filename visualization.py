"""
Utility functions for visualization and evaluation.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Optional
try:
    import seaborn as sns
except ImportError:
    sns = None
from pathlib import Path
import logging

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



def plot_coordinate_distribution(
    lat_coords: List[float],
    lon_coords: List[float],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Plot distribution of latitude and longitude coordinates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Latitude distribution
    ax1.hist(lat_coords, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Latitude (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Latitude Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Longitude distribution
    ax2.hist(lon_coords, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Longitude (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Longitude Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Use bbox_inches to prevent layout warnings
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_world_map_with_coordinates(
    lat_coords: List[float],
    lon_coords: List[float],
    title: str = "Satellite Image Coordinates",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    highlight_points: Optional[List[Tuple[float, float]]] = None
) -> None:
    """Plot coordinates on world map using cartopy (preferred) or basemap (fallback)."""
    if not HAS_CARTOPY and not HAS_BASEMAP:
        logger.error("Cannot plot world map: install cartopy or basemap")
        return

    fig = plt.figure(figsize=(12, 8))

    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=10, central_latitude=30))
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.gridlines(draw_labels=True)

        if lat_coords and lon_coords:
            ax.scatter(lon_coords, lat_coords, marker='D', color='m', s=10, alpha=0.6,
                       transform=ccrs.PlateCarree(), label="Satellite coordinates")

        if highlight_points:
            for i, (lat, lon) in enumerate(highlight_points):
                ax.scatter(lon, lat, marker='o', color='red', s=100,
                           transform=ccrs.PlateCarree(), label=f"Point {i+1}")
    else:
        m = Basemap(projection='ortho', lat_0=30, lon_0=10, resolution='c')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.25)
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawmapboundary(fill_color='lightblue')
        m.drawmeridians(np.arange(0, 360, 30))
        m.drawparallels(np.arange(-90, 90, 30))

        if lat_coords and lon_coords:
            x, y = m(lon_coords, lat_coords)
            m.scatter(x, y, marker='D', color='m', s=10, alpha=0.6,
                      label="Satellite coordinates")

        if highlight_points:
            for i, (lat, lon) in enumerate(highlight_points):
                x, y = m(lon, lat)
                m.scatter(x, y, marker='o', color='red', s=100,
                          label=f"Point {i+1}")

    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Plot training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_coordinate_predictions(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Plot true vs predicted coordinates with error metrics."""
    # Coordinates are already in raw [lon, lat] degrees
    true_coords_np = true_coords.cpu().numpy()
    pred_coords_np = pred_coords.cpu().numpy()
    
    # Calculate longitude errors with wraparound handling
    lon_direct_diff = np.abs(true_coords_np[:, 0] - pred_coords_np[:, 0])
    lon_wrap_diff = 360.0 - lon_direct_diff
    lon_errors = np.minimum(lon_direct_diff, lon_wrap_diff)
    
    # Calculate latitude errors (no wraparound needed)
    lat_errors = np.abs(true_coords_np[:, 1] - pred_coords_np[:, 1])
    
    # Calculate Euclidean distance errors in degrees
    distance_errors = np.sqrt(lon_errors**2 + lat_errors**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Longitude comparison (index 0)
    ax1.scatter(true_coords_np[:, 0], pred_coords_np[:, 0], alpha=0.6, 
               c=distance_errors, cmap='viridis', s=30)
    ax1.plot([true_coords_np[:, 0].min(), true_coords_np[:, 0].max()],
             [true_coords_np[:, 0].min(), true_coords_np[:, 0].max()],
             'r--', label='Perfect Prediction', linewidth=2)
    ax1.set_xlabel('True Longitude (degrees)')
    ax1.set_ylabel('Predicted Longitude (degrees)')
    ax1.set_title(f'Longitude Predictions\nMAE: {lon_errors.mean():.3f}°')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Latitude comparison (index 1)
    ax2.scatter(true_coords_np[:, 1], pred_coords_np[:, 1], alpha=0.6,
               c=distance_errors, cmap='viridis', s=30)
    ax2.plot([true_coords_np[:, 1].min(), true_coords_np[:, 1].max()],
             [true_coords_np[:, 1].min(), true_coords_np[:, 1].max()],
             'r--', label='Perfect Prediction', linewidth=2)
    ax2.set_xlabel('True Latitude (degrees)')
    ax2.set_ylabel('Predicted Latitude (degrees)')
    ax2.set_title(f'Latitude Predictions\nMAE: {lat_errors.mean():.3f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Add colorbar for distance errors
    cbar = fig.colorbar(ax1.collections[0], ax=[ax1, ax2], 
                        label='Distance Error (degrees)', 
                        orientation='horizontal', pad=0.1, aspect=30)
    
    # Use constrained_layout instead of tight_layout to avoid warnings
    
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_error_distribution(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Plot distribution of coordinate prediction errors."""
    # Coordinates are already in raw [lon, lat] degrees
    if isinstance(true_coords, torch.Tensor):
        true_coords_np = true_coords.cpu().numpy()
    else:
        true_coords_np = true_coords

    if isinstance(pred_coords, torch.Tensor):
        pred_coords_np = pred_coords.cpu().numpy()
    else:
        pred_coords_np = pred_coords
    
    # Account for longitude wraparound
    lon_direct_diff = np.abs(true_coords_np[:, 0] - pred_coords_np[:, 0])
    lon_wrap_diff = 360.0 - lon_direct_diff
    lon_errors = np.minimum(lon_direct_diff, lon_wrap_diff)
    lat_errors = np.abs(true_coords_np[:, 1] - pred_coords_np[:, 1])
    
    # Calculate Euclidean distance errors in degrees
    distance_errors = np.sqrt(lon_errors**2 + lat_errors**2)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Longitude error distribution
    ax1.hist(lon_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Longitude Error (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Longitude Error\nMean: {lon_errors.mean():.3f}°, Median: {np.median(lon_errors):.3f}°')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(lon_errors.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
    ax1.axvline(np.median(lon_errors), color='orange', linestyle='--', alpha=0.8, label='Median')
    ax1.legend()
    
    # Latitude error distribution
    ax2.hist(lat_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Latitude Error (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Latitude Error\nMean: {lat_errors.mean():.3f}°, Median: {np.median(lat_errors):.3f}°')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(lat_errors.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
    ax2.axvline(np.median(lat_errors), color='orange', linestyle='--', alpha=0.8, label='Median')
    ax2.legend()
    
    # Distance error distribution
    ax3.hist(distance_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Distance Error (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Euclidean Distance Error\nMean: {distance_errors.mean():.3f}°, Median: {np.median(distance_errors):.3f}°')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(distance_errors.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
    ax3.axvline(np.median(distance_errors), color='orange', linestyle='--', alpha=0.8, label='Median')
    ax3.legend()
    
    
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_coordinate_statistics_table(
    lat_coords: List[float],
    lon_coords: List[float]
) -> pd.DataFrame:
    """Create a statistics table for coordinate data."""
    df_lat = pd.DataFrame(lat_coords)
    df_lon = pd.DataFrame(lon_coords)
    
    stats = pd.DataFrame({
        "Latitude": df_lat.describe()[0],
        "Longitude": df_lon.describe()[0]
    })
    
    return stats


def visualize_sample_images(
    images: torch.Tensor,
    coords: torch.Tensor,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Visualize sample satellite images with their coordinates."""
    # Coordinates are already in raw [lon, lat] degrees
    coords_np = coords.cpu().numpy()
    
    # Select random samples
    if len(images) > num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = images[indices]
        coords_np = coords_np[indices]
    
    # Create subplot grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.atleast_2d(axes)

    for i in range(num_samples):
        row = i // cols
        col = i % cols
        ax = axes[row][col]

        # Convert tensor to image
        img = images[i].squeeze().cpu().numpy()
        if img.ndim == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))

        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.axis('off')
        ax.set_title(f'Lon: {coords_np[i, 0]:.2f}°, Lat: {coords_np[i, 1]:.2f}°')

    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')
    
    
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_predictions_on_world_map(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    max_points: int = 1000
) -> None:
    """Plot prediction errors on world map using cartopy (preferred) or basemap (fallback)."""
    if not HAS_CARTOPY and not HAS_BASEMAP:
        logger.error("Cannot plot world map: install cartopy or basemap")
        return

    true_coords_np = true_coords.cpu().numpy()
    pred_coords_np = pred_coords.cpu().numpy()

    lon_errors = np.abs(true_coords_np[:, 0] - pred_coords_np[:, 0])
    lat_errors = np.abs(true_coords_np[:, 1] - pred_coords_np[:, 1])
    distance_errors = np.sqrt(lon_errors**2 + lat_errors**2)

    if len(true_coords_np) > max_points:
        indices = np.random.choice(len(true_coords_np), max_points, replace=False)
        true_coords_np = true_coords_np[indices]
        pred_coords_np = pred_coords_np[indices]
        distance_errors = distance_errors[indices]

    fig = plt.figure(figsize=(15, 10))

    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Miller())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.gridlines(draw_labels=True)

        scatter = ax.scatter(true_coords_np[:, 0], true_coords_np[:, 1],
                             c=distance_errors, s=20, alpha=0.7, cmap='Reds',
                             edgecolors='black', linewidth=0.5, transform=ccrs.PlateCarree())
    else:
        m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=85,
                    llcrnrlon=-180, urcrnrlon=180, resolution='c')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.25)
        m.fillcontinents(color='lightgray', lake_color='lightblue')
        m.drawmapboundary(fill_color='lightblue')
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], fontsize=10)
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0], fontsize=10)

        lon_true, lat_true = true_coords_np[:, 0], true_coords_np[:, 1]
        x, y = m(lon_true, lat_true)
        scatter = m.scatter(x, y, c=distance_errors, s=20, alpha=0.7,
                            cmap='Reds', edgecolors='black', linewidth=0.5)

    plt.colorbar(scatter, ax=plt.gca(), label='Prediction Error (degrees)',
                 orientation='vertical', pad=0.02, shrink=0.8)

    title = (f'Coordinate Prediction Errors Worldwide\n'
            f'Mean Error: {distance_errors.mean():.3f}°, '
            f'Median Error: {np.median(distance_errors):.3f}°, '
            f'Max Error: {distance_errors.max():.3f}°')
    plt.title(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()