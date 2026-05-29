"""
CLI entry point for contrastive geographic representation learning.

Commands:
  setup       — Download data pipeline (metadata + images)
  download N  — Download last N days of images
  pretrain    — Contrastive pre-training with geographic proximity
  finetune    — Fine-tune pre-trained encoder on (lon, lat) labels
  evaluate    — Evaluate a model on test set metrics
  visualize   — Visualize learned geographic representation on world map
"""

import logging
import argparse
import random
import os

import torch
import numpy as np
from pathlib import Path

from config import Config
from data import EPICDataDownloader, CoordinateExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_data_pipeline(config):
    """Download metadata and sample images."""
    logger.info("Setting up data pipeline...")

    downloader = EPICDataDownloader(config)

    if not Path(config.data.raw_data_dir).exists():
        logger.info("Downloading metadata...")
        downloader.download_metadata()

    if not Path(config.data.images_dir).exists() or not list(Path(config.data.images_dir).iterdir()):
        logger.info("Downloading recent images...")
        downloader.download_recent(7)

    logger.info("Data pipeline setup complete!")

    extractor = CoordinateExtractor(config)
    lat_coords, lon_coords = extractor.extract_coordinates()

    if lat_coords and lon_coords:
        logger.info(f"Coordinate range: Lat [{min(lat_coords):.3f}, {max(lat_coords):.3f}], "
                    f"Lon [{min(lon_coords):.3f}, {max(lon_coords):.3f}]")

        os.makedirs("outputs", exist_ok=True)
        logger.info("Creating coordinate distribution plots...")
        from visualization import (
            plot_coordinate_distribution,
            plot_world_map_with_coordinates,
            create_coordinate_statistics_table,
        )
        plot_coordinate_distribution(lat_coords, lon_coords,
                                     save_path="outputs/coordinate_distribution.png", show_plot=False)
        plot_world_map_with_coordinates(lat_coords, lon_coords,
                                        save_path="outputs/coordinate_world_map.png", show_plot=False)
        import pandas as pd
        stats_table = create_coordinate_statistics_table(lat_coords, lon_coords)
        stats_table.to_csv("outputs/coordinate_statistics.csv", index=False)
        logger.info("Coordinate statistics saved to outputs/coordinate_statistics.csv")


def download_data(config, num_days: int = 7):
    """Download recent satellite images."""
    logger.info(f"Downloading images from last {num_days} days...")
    downloader = EPICDataDownloader(config)
    downloader.download_recent(num_days)
    logger.info("Download complete!")


def pretrain_model(config):
    """Phase 1: Contrastive pre-training with geographic proximity."""
    from contrastive_dataset import create_full_contrastive_loader
    from contrastive_training import ContrastiveTrainer

    logger.info("=" * 60)
    logger.info("Phase 1: Contrastive Pre-training")
    logger.info("=" * 60)

    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)

    full_loader = create_full_contrastive_loader(
        config, batch_size=config.training.batch_size
    )

    trainer = ContrastiveTrainer(full_loader, val_loader=None, config=config)

    try:
        results = trainer.train_phase1()
        logger.info(f"Pre-training complete! Best val loss: {results['best_val_loss']:.4f}")

        # Evaluate the encoder's representation quality on the validation split
        logger.info("Evaluating encoder geographic representation...")
        trainer.evaluate_encoder_representation(trainer.val_loader)

        encoder_path = os.path.join(config.training.save_dir, "encoder_pretrained.pth")
        logger.info(f"Pre-trained encoder saved to: {encoder_path}")
        logger.info(f"Ready for fine-tuning: python main.py finetune {encoder_path}")

    finally:
        trainer.cleanup()


def finetune_model(config, encoder_path: str = None):
    """Phase 2: Fine-tune on (lon, lat) labels."""
    from contrastive_dataset import create_contrastive_dataloaders
    from contrastive_training import ContrastiveTrainer
    logger.info("=" * 60)
    logger.info("Phase 2: Supervised Fine-tuning")
    logger.info("=" * 60)

    if encoder_path is None:
        encoder_path = os.path.join(config.training.save_dir, "encoder_pretrained.pth")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Pre-trained encoder not found at {encoder_path}. "
                "Run 'python main.py pretrain' first or specify --encoder-path."
            )

    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)

    train_loader, val_loader, test_loader = create_contrastive_dataloaders(
        config, batch_size=config.training.batch_size
    )

    trainer = ContrastiveTrainer(train_loader, val_loader, config)

    try:
        results = trainer.train_phase2(encoder_path)
        logger.info(f"Fine-tuning complete! Best val loss: {results['best_val_loss']:.4f}")

        # Final evaluation on test set
        logger.info("Running test set evaluation...")
        metrics = trainer.evaluate(test_loader)
        logger.info("Test Results:")
        logger.info(f"  Mean coordinate error: {metrics['mean_coordinate_error_deg']:.4f} deg")
        logger.info(f"  Median coordinate error: {metrics['median_coordinate_error_deg']:.4f} deg")
        logger.info(f"  Mean Haversine distance: {metrics['mean_haversine_km']:.1f} km")
        logger.info(f"  Median Haversine distance: {metrics['median_haversine_km']:.1f} km")

    finally:
        trainer.cleanup()


def evaluate_model(config, model_path: str):
    """Evaluate a fine-tuned model on the test set."""
    logger.info(f"Evaluating model: {model_path}")

    from models import create_location_regressor
    from contrastive_dataset import create_contrastive_dataloaders
    from contrastive_training import ContrastiveTrainer

    model = create_location_regressor(config)

    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=config.training.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Model loaded successfully")

        # Move model to the correct device
        device = torch.device(config.training.device)
        model = model.to(device)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    _, _, test_loader = create_contrastive_dataloaders(
        config, batch_size=config.training.batch_size
    )

    # Create minimal dummy loaders for trainer init
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = TensorDataset(
        torch.zeros(1, config.model.input_channels, config.data.image_size, config.data.image_size),
        torch.zeros(1, 2),
    )
    dummy_loader = DataLoader(dummy_data, batch_size=1)

    trainer = ContrastiveTrainer(dummy_loader, test_loader, config)
    trainer.model = model
    trainer.phase = 'finetune'

    metrics = trainer.evaluate(test_loader)

    logger.info("Evaluation Results:")
    logger.info(f"  Mean coordinate error: {metrics['mean_coordinate_error_deg']:.4f} deg")
    logger.info(f"  Median coordinate error: {metrics['median_coordinate_error_deg']:.4f} deg")
    logger.info(f"  Mean Haversine distance: {metrics['mean_haversine_km']:.1f} km")
    logger.info(f"  Median Haversine distance: {metrics['median_haversine_km']:.1f} km")

    trainer.cleanup()
    return metrics


def visualize_model_representation(config, model_path: str, n_clusters: int = 8):
    """Visualize learned geographic representation on a world map."""
    from visualize_representation import visualize_representation

    os.makedirs("outputs", exist_ok=True)
    visualize_representation(
        config, model_path,
        n_clusters=n_clusters,
        seed=config.training.random_seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive Geographic Representation Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup                     # Download data pipeline
  %(prog)s download 7                # Download last 7 days of images
  %(prog)s pretrain                  # Phase 1: contrastive pre-training
  %(prog)s finetune                  # Phase 2: fine-tune on coordinates
  %(prog)s finetune --encoder-path models/encoder_pretrained.pth
  %(prog)s evaluate models/regressor_finetuned.pth
  %(prog)s visualize models/regressor_finetuned.pth
  %(prog)s visualize models/encoder_pretrained.pth --n-clusters 12
        """
    )

    parser.add_argument("command",
                        choices=["setup", "download", "pretrain", "finetune",
                                 "evaluate", "visualize"],
                        help="Command to execute")
    parser.add_argument("target", nargs="?",
                        help="Target (model path or number of days)")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"],
                        help="Training device")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard auto-launch")
    parser.add_argument("--encoder-path", type=str,
                        help="Path to pre-trained encoder for fine-tuning")
    parser.add_argument("--n-clusters", type=int, default=8,
                        help="KMeans clusters for visualize (default: 8)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze encoder weights during fine-tuning")
    parser.add_argument("--temperature", type=float,
                        help="Contrastive loss temperature")
    parser.add_argument("--pos-threshold", type=float,
                        help="Positive pair Haversine threshold (km)")
    parser.add_argument("--neg-threshold", type=float,
                        help="Negative pair Haversine threshold (km)")

    args = parser.parse_args()

    # Load config
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()

    # CLI overrides
    if args.epochs:
        config.training.max_epochs = args.epochs
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.training.device = args.device
    if args.no_tensorboard:
        config.training.launch_tensorboard = False
    if args.freeze_backbone:
        config.contrastive.freeze_backbone = True
    if args.temperature:
        config.contrastive.temperature = args.temperature
    if args.pos_threshold:
        config.contrastive.pos_threshold_km = args.pos_threshold
    if args.neg_threshold:
        config.contrastive.neg_threshold_km = args.neg_threshold

    # Set threads and seeds
    torch.set_num_interop_threads(config.training.num_threads)
    torch.set_num_threads(config.training.num_threads)
    _set_seed(config.training.random_seed)

    try:
        if args.command == "setup":
            setup_data_pipeline(config)

        elif args.command == "download":
            num_days = int(args.target) if args.target else 7
            download_data(config, num_days)

        elif args.command == "pretrain":
            pretrain_model(config)

        elif args.command == "finetune":
            encoder_path = args.encoder_path
            finetune_model(config, encoder_path)

        elif args.command == "evaluate":
            if not args.target:
                raise ValueError("evaluate requires a model path")
            evaluate_model(config, args.target)

        elif args.command == "visualize":
            if not args.target:
                raise ValueError("visualize requires a model path")
            visualize_model_representation(config, args.target, args.n_clusters)

    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
