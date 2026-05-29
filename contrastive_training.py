"""
Two-phase training orchestrator for contrastive geographic representation learning
with comprehensive TensorBoard monitoring.
"""

import os
import logging
import subprocess
import io
from datetime import datetime
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from models import (
    GeographicEncoder, ProjectionHead, RegressionHead,
    ContrastiveModel, FineTunedModel, count_parameters,
)
from losses import (
    GeographicAlignmentLoss, compute_embedding_distances,
    compute_geo_alignment_metrics,
)
from datasets import CoordinateNormalizer
from tensorboard_utils import start_tensorboard

logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    """Two-phase trainer: contrastive pre-training -> supervised fine-tuning.

    Phase 1 (pretrain): Train encoder + projection head with SupConLoss
        using geographic proximity to define positive/negative pairs.
    Phase 2 (finetune): Train encoder + regression head on (lon, lat) labels.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config = None,
        device: Optional[str] = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(config.training.device)

        # If no val_loader given, split off a portion of training data for validation
        self._val_split_frac = None
        if self.val_loader is None:
            self._val_split_frac = 0.1  # 10% of train data used as validation
            self._setup_train_val_split()

        # State
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.phase = None  # 'pretrain' or 'finetune'

        # TensorBoard
        self.tensorboard_process = None
        self.writer = None
        self.tensorboard_run_dir = None

        # Coordinate normalization (shared across phases)
        self._setup_coordinate_normalizer()

    # ── Setup ──

    def _setup_train_val_split(self):
        """Create a fixed validation subset from training data."""
        from torch.utils.data import Subset
        full_dataset = self.train_loader.dataset
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * self._val_split_frac))

        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(
            self.config.training.random_seed
        )).tolist()

        val_indices = set(indices[:n_val])
        train_indices = [i for i in range(n_total) if i not in val_indices]

        batch_size = self.train_loader.batch_size
        nw = self.train_loader.num_workers
        pm = self.train_loader.pin_memory
        pw = getattr(self.train_loader, 'persistent_workers', False)

        self._full_train_dataset = full_dataset
        self._val_dataset = Subset(full_dataset, list(val_indices))
        self.train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=pm,
            persistent_workers=pw,
        )
        self.val_loader = DataLoader(
            self._val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=pm,
        )
        logger.info(f"Split full dataset: {len(train_indices)} train, {len(val_indices)} val")

    def _setup_coordinate_normalizer(self):
        all_coords = []
        for _, coords in self.train_loader:
            all_coords.append(coords)
        if all_coords:
            self.coord_normalizer = CoordinateNormalizer(torch.cat(all_coords, dim=0))
        else:
            self.coord_normalizer = CoordinateNormalizer()

    def _init_tensorboard(self, phase_tag: str):
        """Initialize TensorBoard writer for a training phase."""
        try:
            from torch.utils.tensorboard.writer import SummaryWriter
        except ImportError:
            logger.warning("TensorBoard not available")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Finetune phase uses 'run_' prefix to match baseline UnifiedTrainer
        folder = f"run_{timestamp}" if phase_tag == "finetune" else f"{phase_tag}_{timestamp}"
        self.tensorboard_run_dir = os.path.join(
            self.config.training.log_dir, folder
        )
        os.makedirs(self.tensorboard_run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_run_dir)

        if self.config.training.launch_tensorboard:
            try:
                start_tensorboard(
                    self.tensorboard_run_dir,
                    self.config.training.tensorboard_port,
                    open_browser=self.config.training.open_browser,
                )
            except Exception as e:
                logger.warning(f"TensorBoard launch failed: {e}")

        logger.info(f"TensorBoard logs: {self.tensorboard_run_dir}")

    def _log_sample_images(self, loader: DataLoader, tag: str, n: int = 16):
        """Log a grid of input images to TensorBoard for visual inspection."""
        if self.writer is None:
            return
        try:
            images, _ = next(iter(loader))
            images = images[:n].cpu()
            grid = torch.cat([img for img in images], dim=2)  # horizontal strip
            # Pad to 3 channels for proper display
            if grid.shape[0] == 1:
                grid = grid.repeat(3, 1, 1)
            self.writer.add_image(f'images/{tag}', grid, self.current_epoch)
        except Exception as e:
            logger.warning(f"Failed to log sample images: {e}")

    def _log_histograms(self, prefix: str = ""):
        """Log weight and gradient histograms for all model parameters."""
        if self.writer is None:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(
                    f"params/{prefix}{name}", param.data, self.current_epoch
                )
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"gradients/{prefix}{name}", param.grad, self.current_epoch
                    )

    def _log_embeddings_tensorboard(self, loader: DataLoader, tag: str):
        """Log ALL 128-d embeddings to TensorBoard projector with image thumbnails
        and geographic metadata. TensorBoard's projector tab shows the evolving
        latent space — color by longitude/latitude to see geographic structure emerge."""
        if self.writer is None:
            return

        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_images = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                embeddings = self.model.get_embeddings(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(targets)
                all_images.append(images.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_images = torch.cat(all_images, dim=0)

        # TensorBoard needs images in [0, 1] range and 3 channels for thumbnails
        label_img = all_images
        if label_img.shape[1] == 1:
            label_img = label_img.repeat(1, 3, 1, 1)

        metadata = [[f"{lon:.1f}", f"{lat:.1f}"] for lon, lat in all_labels.tolist()]
        metadata_header = ["longitude", "latitude"]

        self.writer.add_embedding(
            all_embeddings,
            metadata=metadata,
            metadata_header=metadata_header,
            label_img=label_img,
            global_step=self.current_epoch,
            tag=tag,
        )
        logger.info(f"Logged {len(all_embeddings)} embeddings to projector ({tag})")

    def _log_kmeans_figure(self, loader: DataLoader, tag: str, n_clusters: int = 8):
        """Generate KMeans-on-world-map figure and log to TensorBoard.

        This provides a visual check that the latent space is developing
        geographic structure over the course of training.
        """
        if self.writer is None:
            return

        try:
            from visualize_representation import (
                extract_test_embeddings, cluster_and_order,
            )
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Extract embeddings and coords
            self.model.eval()
            all_embeddings = []
            all_coords = []
            with torch.no_grad():
                for images, coords in loader:
                    images = images.to(self.device)
                    emb = self.model.get_embeddings(images)
                    all_embeddings.append(emb.cpu().numpy())
                    all_coords.append(coords.numpy())
            embeddings = np.concatenate(all_embeddings, axis=0)
            coords = np.concatenate(all_coords, axis=0)

            n = len(embeddings)
            effective_k = max(2, min(n_clusters, n // 5))
            cluster_ids = cluster_and_order(embeddings, effective_k, self.config.training.random_seed)

            # Render to figure with map background
            cmap_name = 'tab20' if effective_k > 10 else 'tab10'
            cmap = plt.get_cmap(cmap_name, effective_k)

            from test_predictions import HAS_CARTOPY, HAS_BASEMAP, _draw_miller_map
            has_map = HAS_CARTOPY or HAS_BASEMAP

            fig = plt.figure(figsize=(14, 9))

            if has_map:
                backend, is_cartopy = _draw_miller_map()
                if is_cartopy:
                    import cartopy.crs as ccrs
                    scatter = backend.scatter(
                        coords[:, 0], coords[:, 1],
                        c=cluster_ids, cmap=cmap,
                        vmin=-0.5, vmax=effective_k - 0.5,
                        s=20, alpha=0.75, edgecolors='black', linewidth=0.2,
                        zorder=5, transform=ccrs.PlateCarree(),
                    )
                    backend.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
                else:
                    x, y = backend(coords[:, 0], coords[:, 1])
                    scatter = backend.scatter(
                        x, y,
                        c=cluster_ids, cmap=cmap,
                        vmin=-0.5, vmax=effective_k - 0.5,
                        s=20, alpha=0.75, edgecolors='black', linewidth=0.2,
                        zorder=5,
                    )
            else:
                ax = plt.gca()
                scatter = ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c=cluster_ids, cmap=cmap,
                    vmin=-0.5, vmax=effective_k - 0.5,
                    s=20, alpha=0.75, edgecolors='black', linewidth=0.2,
                )
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
                ax.set_xlabel('Longitude (deg)')
                ax.set_ylabel('Latitude (deg)')
                ax.grid(True, alpha=0.3)

            plt.colorbar(scatter, orientation='horizontal', pad=0.05, shrink=0.7,
                         ticks=np.arange(effective_k))
            plt.title(f'{tag} — Epoch {self.current_epoch} — KMeans(k={effective_k}) on 128-d Embeddings',
                      fontsize=12, fontweight='bold')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = plt.imread(buf)

            self.writer.add_image(
                f'latent_space/{tag}',
                img, self.current_epoch, dataformats='HWC',
            )
            plt.close(fig)
            logger.info(f"Logged KMeans figure to TensorBoard ({tag}, epoch {self.current_epoch})")

        except Exception as e:
            logger.warning(f"Failed to log KMeans figure: {e}")

    # ── Phase 1: Contrastive Pre-training ──

    def _setup_pretrain(self):
        """Build encoder + projection head, optimizer, criterion."""
        encoder = GeographicEncoder(
            input_channels=self.config.model.input_channels,
            conv_channels=self.config.model.conv_channels,
            kernel_size=self.config.model.kernel_size,
            pool_size=self.config.model.pool_size,
            activation=self.config.model.activation,
            hidden_dim=self.config.model.hidden_dim,
            image_size=self.config.data.image_size,
        )
        projection = ProjectionHead(
            input_dim=self.config.model.hidden_dim,
            output_dim=self.config.contrastive.projection_dim,
        )
        self.model = ContrastiveModel(encoder, projection).to(self.device)

        self.criterion = GeographicAlignmentLoss()

        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()
        self.phase = 'pretrain'

        logger.info(f"Phase 1 (pretrain): {count_parameters(self.model):,} parameters")
        logger.info(f"  pos_threshold: {self.config.contrastive.pos_threshold_km} km")
        logger.info(f"  neg_threshold: {self.config.contrastive.neg_threshold_km} km")
        logger.info(f"  temperature: {self.config.contrastive.temperature}")

    def _make_optimizer(self):
        name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay
        if name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        elif name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {name}")

    def _make_scheduler(self):
        name = self.config.training.scheduler.lower()
        if name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.training.step_size,
                gamma=self.config.training.gamma,
            )
        elif name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.max_epochs,
            )
        elif name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.config.training.gamma,
                patience=self.config.training.step_size,
            )
        elif name == 'none':
            return None
        raise ValueError(f"Unknown scheduler: {name}")

    def train_pretrain_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, targets in tqdm(self.train_loader, desc='Pretrain', leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            embeddings = self.model.get_embeddings(images)
            loss = self.criterion(embeddings, targets)

            if loss.item() > 0:
                loss.backward()
                if self.config.training.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clipping
                    )
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_pretrain_epoch(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Val-Pretrain', leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)

                embeddings = self.model.get_embeddings(images)
                loss = self.criterion(embeddings, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _compute_pair_metrics(self):
        """Compute geographic alignment metrics on a val batch."""
        self.model.eval()
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                embeddings = self.model.get_embeddings(images)
                pos_dist, neg_dist, ratio = compute_embedding_distances(
                    embeddings, targets,
                    self.config.contrastive.pos_threshold_km,
                    self.config.contrastive.neg_threshold_km,
                )
                geo_corr = compute_geo_alignment_metrics(embeddings, targets)
                return pos_dist, neg_dist, ratio, geo_corr

    def train_phase1(self):
        """Run contrastive pre-training loop."""
        self._setup_pretrain()
        self._init_tensorboard("pretrain")

        # Log model graph + config
        if self.writer is not None:
            self.writer.add_text('config', f"```\nPhase: pretrain\n"
                f"temperature={self.config.contrastive.temperature}\n"
                f"pos_threshold={self.config.contrastive.pos_threshold_km}km\n"
                f"neg_threshold={self.config.contrastive.neg_threshold_km}km\n"
                f"lr={self.config.training.learning_rate}\n"
                f"batch_size={self.config.training.batch_size}\n"
                f"epochs={self.config.training.max_epochs}\n```")
            try:
                dummy = torch.zeros(1, self.config.model.input_channels,
                                    self.config.data.image_size,
                                    self.config.data.image_size).to(self.device)
                self.writer.add_graph(self.model, dummy)
            except Exception as e:
                logger.warning(f"Failed to log graph: {e}")

        log_every = self.config.contrastive.embedding_log_every

        for epoch in tqdm(range(self.config.training.max_epochs), desc="Phase 1"):
            self.current_epoch = epoch

            train_loss = self.train_pretrain_epoch()
            val_loss = self.validate_pretrain_epoch()
            pos_dist, neg_dist, ratio, geo_corr = self._compute_pair_metrics()

            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ── TensorBoard: scalars (every epoch) ──
            if self.writer is not None:
                step = epoch
                self.writer.add_scalar('pretrain/loss/train', train_loss, step)
                self.writer.add_scalar('pretrain/loss/val', val_loss, step)
                self.writer.add_scalar('pretrain/metrics/pos_dist', pos_dist, step)
                self.writer.add_scalar('pretrain/metrics/neg_dist', neg_dist, step)
                self.writer.add_scalar('pretrain/metrics/pos_neg_ratio', ratio, step)
                self.writer.add_scalar('pretrain/metrics/geo_spearman_corr', geo_corr, step)
                self.writer.add_scalar('pretrain/lr', self.optimizer.param_groups[0]['lr'], step)

            # ── TensorBoard: embeddings projector (every epoch) ──
            if self.writer is not None:
                self._log_embeddings_tensorboard(self.val_loader, "pretrain")

            # ── TensorBoard: histograms + images (periodic) ──
            if self.writer is not None and epoch % 5 == 0:
                self._log_histograms(prefix="pretrain/")
                self._log_sample_images(self.train_loader, "pretrain/samples")

            # ── TensorBoard: KMeans world map (periodic) ──
            if self.writer is not None and epoch % log_every == 0:
                self._log_kmeans_figure(self.val_loader, "pretrain/latent_kmeans")

            # Save best
            best_val_loss = getattr(self, 'best_val_loss', float('inf'))
            if val_loss < best_val_loss:
                self.best_val_loss = val_loss
                self._save_encoder("encoder_best_pretrain.pth")

            logger.info(
                f'Phase1 Epoch {epoch+1}/{self.config.training.max_epochs}: '
                f'Train={train_loss:.4f} Val={val_loss:.4f} '
                f'PosDist={pos_dist:.3f} NegDist={neg_dist:.3f} Ratio={ratio:.2f}'
            )

        # Save final encoder
        self._save_encoder("encoder_pretrained.pth")

        # Log hparams
        if self.writer is not None:
            hparam_dict = {k: str(v) for k, v in {
                'phase': 'pretrain', 'temperature': self.config.contrastive.temperature,
                'pos_threshold': self.config.contrastive.pos_threshold_km,
                'neg_threshold': self.config.contrastive.neg_threshold_km,
                'lr': self.config.training.learning_rate,
                'batch_size': self.config.training.batch_size,
                'epochs': self.config.training.max_epochs,
            }.items()}
            self.writer.add_hparams(hparam_dict, {
                'hparam/best_val_loss': self.best_val_loss,
                'hparam/final_pos_neg_ratio': ratio,
            }, run_name='.')
            self.writer.flush()

        logger.info(f'Phase 1 complete! Best val loss: {self.best_val_loss:.4f}')
        return {'best_val_loss': self.best_val_loss}

    # ── Phase 2: Fine-tuning ──

    def _setup_finetune(self, encoder_path: str):
        """Load pre-trained encoder and attach regression head."""
        encoder = GeographicEncoder(
            input_channels=self.config.model.input_channels,
            conv_channels=self.config.model.conv_channels,
            kernel_size=self.config.model.kernel_size,
            pool_size=self.config.model.pool_size,
            activation=self.config.model.activation,
            hidden_dim=self.config.model.hidden_dim,
            image_size=self.config.data.image_size,
        )

        # Load pre-trained weights
        checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Try to extract encoder keys from full model state
            state = checkpoint['model_state_dict']
            encoder_state = {k.replace('encoder.', ''): v
                           for k, v in state.items() if k.startswith('encoder.')}
            if encoder_state:
                encoder.load_state_dict(encoder_state)
            else:
                encoder.load_state_dict(state)
        else:
            encoder.load_state_dict(checkpoint)
        logger.info(f"Loaded pre-trained encoder from {encoder_path}")

        if self.config.contrastive.freeze_backbone:
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen — only regression head will be trained")

        regression = RegressionHead(
            input_dim=self.config.model.hidden_dim,
            output_dim=self.config.model.output_dim,
        )
        self.model = FineTunedModel(encoder, regression).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()
        self.best_val_loss = float('inf')
        self.phase = 'finetune'

        # Recompute normalizer on the fine-tuning train split for comparability
        # with the baseline (which only sees the 80% train split)
        self._setup_coordinate_normalizer()

        logger.info(f"Phase 2 (finetune): {count_parameters(self.model):,} parameters")

    def train_finetune_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, targets in tqdm(self.train_loader, desc='Finetune', leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(
                self.coord_normalizer.normalize(outputs),
                self.coord_normalizer.normalize(targets),
            )
            loss.backward()

            if self.config.training.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clipping
                )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_finetune_epoch(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Val-Finetune', leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(
                    self.coord_normalizer.normalize(outputs),
                    self.coord_normalizer.normalize(targets),
                )
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _compute_regression_metrics(self, loader: DataLoader):
        """Compute coordinate error and Haversine distance on a loader."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                predictions = self.model(images)
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        coord_error = self.coord_normalizer.compute_coordinate_error_degrees(
            all_preds, all_targets
        ).mean().item()
        haversine = self.coord_normalizer.compute_haversine_distance(
            all_preds, all_targets
        ).mean().item()

        return coord_error, haversine

    def train_phase2(self, encoder_path: str):
        """Fine-tune the pre-trained encoder with a regression head."""
        self._setup_finetune(encoder_path)
        self._init_tensorboard("finetune")

        # Log config
        if self.writer is not None:
            self.writer.add_text('config', f"```\nPhase: finetune\n"
                f"encoder_path={encoder_path}\n"
                f"freeze_backbone={self.config.contrastive.freeze_backbone}\n"
                f"lr={self.config.training.learning_rate}\n"
                f"batch_size={self.config.training.batch_size}\n"
                f"epochs={self.config.training.max_epochs}\n```")

        log_every = self.config.contrastive.embedding_log_every

        for epoch in tqdm(range(self.config.training.max_epochs), desc="Phase 2"):
            self.current_epoch = epoch

            train_loss = self.train_finetune_epoch()
            val_loss = self.validate_finetune_epoch()
            coord_error, haversine = self._compute_regression_metrics(self.val_loader)

            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ── TensorBoard: scalars (every epoch) ──
            # Tag names mirror the baseline UnifiedTrainer for direct comparison
            if self.writer is not None:
                step = epoch
                self.writer.add_scalar('loss/train', train_loss, step)
                self.writer.add_scalar('loss/val', val_loss, step)
                self.writer.add_scalar('metrics/coord_error_deg', coord_error, step)
                self.writer.add_scalar('metrics/haversine_km', haversine, step)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], step)

            # ── TensorBoard: embeddings projector (every epoch) ──
            if self.writer is not None:
                self._log_embeddings_tensorboard(self.val_loader, "model_embeddings")

            # ── TensorBoard: histograms + images (periodic) ──
            if self.writer is not None and epoch % 5 == 0:
                self._log_histograms(prefix="finetune/")
                self._log_sample_images(self.train_loader, "finetune/samples")

            # ── TensorBoard: KMeans world map (periodic) ──
            if self.writer is not None and epoch % log_every == 0:
                self._log_kmeans_figure(self.val_loader, "finetune/latent_kmeans")

            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_full_model("regressor_best_finetune.pth")

            logger.info(
                f'Phase2 Epoch {epoch+1}/{self.config.training.max_epochs}: '
                f'Train={train_loss:.4f} Val={val_loss:.4f} '
                f'CoordErr={coord_error:.3f}deg Haversine={haversine:.1f}km'
            )

        # Save final model
        self._save_full_model("regressor_finetuned.pth")

        # Log hparams (aligned with baseline UnifiedTrainer for direct comparison)
        if self.writer is not None:
            hparam_dict = {
                'lr': self.config.training.learning_rate,
                'batch_size': self.config.training.batch_size,
                'epochs': self.config.training.max_epochs,
                'optimizer': self.config.training.optimizer,
                'loss_fn': self.config.training.loss_function,
                'scheduler': self.config.training.scheduler,
                'weight_decay': self.config.training.weight_decay,
                'device': str(self.device),
                'image_size': self.config.data.image_size,
                'conv_channels': str(self.config.model.conv_channels),
                'hidden_dim': self.config.model.hidden_dim,
                'params': sum(p.numel() for p in self.model.parameters()),
                'phase': 'finetune',
                'freeze_backbone': str(self.config.contrastive.freeze_backbone),
            }
            self.writer.add_hparams(hparam_dict, {
                'hparam/best_val_loss': self.best_val_loss,
                'hparam/final_train_loss': train_loss,
                'hparam/final_val_loss': val_loss,
                'hparam/final_coord_error_deg': coord_error,
                'hparam/final_haversine_km': haversine,
            }, run_name='.')
            self.writer.flush()

        logger.info(f'Phase 2 complete! Best val loss: {self.best_val_loss:.4f}')
        return {'best_val_loss': self.best_val_loss}

    # ── Checkpoint ──

    def _save_encoder(self, filename: str):
        filepath = os.path.join(self.config.training.save_dir, filename)
        os.makedirs(self.config.training.save_dir, exist_ok=True)
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
        }, filepath)
        logger.info(f"Encoder saved: {filepath}")

    def _save_full_model(self, filename: str):
        filepath = os.path.join(self.config.training.save_dir, filename)
        os.makedirs(self.config.training.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
        }, filepath)
        logger.info(f"Model saved: {filepath}")

    # ── Evaluation ──

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the fine-tuned model on a test set."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                predictions = self.model(images)
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        coord_errors = self.coord_normalizer.compute_coordinate_error_degrees(
            all_preds, all_targets
        )
        haversine = self.coord_normalizer.compute_haversine_distance(
            all_preds, all_targets
        )

        return {
            'mean_coordinate_error_deg': coord_errors.mean().item(),
            'median_coordinate_error_deg': coord_errors.median().item(),
            'mean_haversine_km': haversine.mean().item(),
            'median_haversine_km': haversine.median().item(),
        }

    # ── Evaluate pre-trained encoder (without fine-tuning) ──

    def evaluate_encoder_representation(self, test_loader: DataLoader,
                                         n_clusters: int = 8):
        """Evaluate how well the pre-trained encoder's latent space captures
        geographic structure by clustering embeddings and measuring
        spatial coherence."""
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        self.model.eval()
        all_embeddings = []
        all_coords = []

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                emb = self.model.get_embeddings(images)
                all_embeddings.append(emb.cpu().numpy())
                all_coords.append(targets.numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        coords = np.concatenate(all_coords, axis=0)

        # KMeans on embeddings
        scaled = StandardScaler().fit_transform(embeddings)
        km = KMeans(n_clusters=n_clusters, random_state=self.config.training.random_seed, n_init=10)
        cluster_ids = km.fit_predict(scaled)

        # Measure within-cluster geographic spread (Haversine)
        cluster_spreads = []
        for c in range(n_clusters):
            mask = cluster_ids == c
            if mask.sum() > 1:
                c_coords = torch.tensor(coords[mask])
                N = c_coords.shape[0]
                c_i = c_coords.unsqueeze(1).expand(-1, N, -1).reshape(-1, 2)
                c_j = c_coords.unsqueeze(0).expand(N, -1, -1).reshape(-1, 2)
                dists = self.coord_normalizer.compute_haversine_distance(c_i, c_j)
                cluster_spreads.append(dists.mean().item())

        mean_spread = np.mean(cluster_spreads) if cluster_spreads else 0.0

        logger.info(f"Encoder representation analysis (k={n_clusters}):")
        logger.info(f"  Mean within-cluster Haversine distance: {mean_spread:.1f} km")
        logger.info(f"  (Lower = more geographically coherent clusters)")

        return mean_spread

    def cleanup(self):
        if self.writer is not None:
            self.writer.close()
        if isinstance(self.tensorboard_process, subprocess.Popen):
            try:
                self.tensorboard_process.terminate()
                self.tensorboard_process.wait(timeout=5)
            except Exception:
                try:
                    self.tensorboard_process.kill()
                except Exception:
                    pass
        self.tensorboard_process = None
