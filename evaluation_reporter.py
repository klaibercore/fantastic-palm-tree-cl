"""
Comprehensive evaluation report generation utilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import pandas as pd
from datetime import datetime

from datasets import CoordinateNormalizer

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Comprehensive evaluation report generator."""
    
    def __init__(self, model_path: str, config):
        self.model_path = model_path
        self.config = config
        self.report_data = {}
        
    def generate_comprehensive_report(
        self,
        all_predictions: torch.Tensor,
        all_targets: torch.Tensor,
        mse_loss: float,
        output_dir: str = "outputs"
    ) -> str:
        """Generate comprehensive evaluation report."""
        
        # Both predictions and targets are in raw [lon, lat] coordinates
        normalizer = CoordinateNormalizer()

        pred_coords = all_predictions
        true_coords = all_targets

        # Calculate all metrics (inputs are raw coordinates in degrees)
        coord_errors = normalizer.compute_coordinate_error_degrees(pred_coords, true_coords)
        haversine_distances = normalizer.compute_haversine_distance(pred_coords, true_coords)
        lon_errors = normalizer.compute_longitude_error(pred_coords[:, 0], true_coords[:, 0])
        lat_errors = torch.abs(pred_coords[:, 1] - true_coords[:, 1])
        
        # Move to CPU for calculations (all values already non-negative)
        coord_errors_cpu = coord_errors.cpu()
        haversine_cpu = haversine_distances.cpu()
        lon_errors_cpu = lon_errors.cpu()
        lat_errors_cpu = lat_errors.cpu()
        
        # Compile comprehensive report data
        self.report_data = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "dataset_size": len(all_targets),
                "device": self.config.training.device
            },
            "model_performance": {
                "mse_loss": float(mse_loss),
                "coordinate_errors_degrees": {
                    "mean": float(coord_errors_cpu.mean()),
                    "median": float(coord_errors_cpu.median()),
                    "std": float(coord_errors_cpu.std()),
                    "min": float(coord_errors_cpu.min()),
                    "max": float(coord_errors_cpu.max()),
                    "p25": float(coord_errors_cpu.quantile(0.25)),
                    "p75": float(coord_errors_cpu.quantile(0.75)),
                    "p95": float(coord_errors_cpu.quantile(0.95)),
                    "p99": float(coord_errors_cpu.quantile(0.99))
                },
                "haversine_distances_km": {
                    "mean": float(haversine_cpu.mean()),
                    "median": float(haversine_cpu.median()),
                    "std": float(haversine_cpu.std()),
                    "min": float(haversine_cpu.min()),
                    "max": float(haversine_cpu.max()),
                    "p25": float(haversine_cpu.quantile(0.25)),
                    "p75": float(haversine_cpu.quantile(0.75)),
                    "p95": float(haversine_cpu.quantile(0.95)),
                    "p99": float(haversine_cpu.quantile(0.99))
                },
                "longitude_errors_degrees": {
                    "mean": float(lon_errors_cpu.mean()),
                    "median": float(lon_errors_cpu.median()),
                    "std": float(lon_errors_cpu.std()),
                    "min": float(lon_errors_cpu.min()),
                    "max": float(lon_errors_cpu.max())
                },
                "latitude_errors_degrees": {
                    "mean": float(lat_errors_cpu.mean()),
                    "median": float(lat_errors_cpu.median()),
                    "std": float(lat_errors_cpu.std()),
                    "min": float(lat_errors_cpu.min()),
                    "max": float(lat_errors_cpu.max())
                }
            },
            "accuracy_analysis": {
                "predictions_within_1km": int((haversine_cpu <= 1).sum()),
                "predictions_within_10km": int((haversine_cpu <= 10).sum()),
                "predictions_within_100km": int((haversine_cpu <= 100).sum()),
                "predictions_within_1000km": int((haversine_cpu <= 1000).sum()),
                "percentage_within_1km": float((haversine_cpu <= 1).float().mean() * 100),
                "percentage_within_10km": float((haversine_cpu <= 10).float().mean() * 100),
                "percentage_within_100km": float((haversine_cpu <= 100).float().mean() * 100),
                "percentage_within_1000km": float((haversine_cpu <= 1000).float().mean() * 100)
            }
        }
        
        # Save reports in different formats
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_path = output_path / f"evaluation_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        
        # Markdown report
        md_path = output_path / f"evaluation_report_{timestamp}.md"
        self._generate_markdown_report(md_path)
        
        # CSV summary
        csv_path = output_path / f"evaluation_summary_{timestamp}.csv"
        self._generate_csv_summary(csv_path)
        
        logger.info(f"Evaluation reports generated:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  Markdown: {md_path}")
        logger.info(f"  CSV: {csv_path}")
        
        return str(md_path)
    
    def _generate_markdown_report(self, output_path: Path) -> None:
        """Generate comprehensive markdown report."""
        
        md_content = f"""# Satellite Image Coordinate Prediction - Evaluation Report

## Evaluation Summary
- **Timestamp**: {self.report_data['evaluation_metadata']['timestamp']}
- **Model Path**: {self.report_data['evaluation_metadata']['model_path']}
- **Dataset Size**: {self.report_data['evaluation_metadata']['dataset_size']} samples
- **Device**: {self.report_data['evaluation_metadata']['device']}

## Model Performance

### Overall Metrics
- **MSE Loss**: {self.report_data['model_performance']['mse_loss']:.6f}

### Coordinate Errors (Degrees)
| Metric | Longitude | Latitude | Combined |
|--------|-----------|----------|----------|
| Mean | {self.report_data['model_performance']['longitude_errors_degrees']['mean']:.3f}° | {self.report_data['model_performance']['latitude_errors_degrees']['mean']:.3f}° | {self.report_data['model_performance']['coordinate_errors_degrees']['mean']:.3f}° |
| Median | {self.report_data['model_performance']['longitude_errors_degrees']['median']:.3f}° | {self.report_data['model_performance']['latitude_errors_degrees']['median']:.3f}° | {self.report_data['model_performance']['coordinate_errors_degrees']['median']:.3f}° |
| Std Dev | {self.report_data['model_performance']['longitude_errors_degrees']['std']:.3f}° | {self.report_data['model_performance']['latitude_errors_degrees']['std']:.3f}° | {self.report_data['model_performance']['coordinate_errors_degrees']['std']:.3f}° |
| Min | {self.report_data['model_performance']['longitude_errors_degrees']['min']:.3f}° | {self.report_data['model_performance']['latitude_errors_degrees']['min']:.3f}° | {self.report_data['model_performance']['coordinate_errors_degrees']['min']:.3f}° |
| Max | {self.report_data['model_performance']['longitude_errors_degrees']['max']:.3f}° | {self.report_data['model_performance']['latitude_errors_degrees']['max']:.3f}° | {self.report_data['model_performance']['coordinate_errors_degrees']['max']:.3f}° |
| 25th Percentile | - | - | {self.report_data['model_performance']['coordinate_errors_degrees']['p25']:.3f}° |
| 75th Percentile | - | - | {self.report_data['model_performance']['coordinate_errors_degrees']['p75']:.3f}° |
| 95th Percentile | - | - | {self.report_data['model_performance']['coordinate_errors_degrees']['p95']:.3f}° |
| 99th Percentile | - | - | {self.report_data['model_performance']['coordinate_errors_degrees']['p99']:.3f}° |

### Geographic Distance (Haversine - Kilometers)
| Metric | Distance |
|--------|----------|
| Mean | {self.report_data['model_performance']['haversine_distances_km']['mean']:.1f} km |
| Median | {self.report_data['model_performance']['haversine_distances_km']['median']:.1f} km |
| Std Dev | {self.report_data['model_performance']['haversine_distances_km']['std']:.1f} km |
| Min | {self.report_data['model_performance']['haversine_distances_km']['min']:.1f} km |
| Max | {self.report_data['model_performance']['haversine_distances_km']['max']:.1f} km |
| 25th Percentile | {self.report_data['model_performance']['haversine_distances_km']['p25']:.1f} km |
| 75th Percentile | {self.report_data['model_performance']['haversine_distances_km']['p75']:.1f} km |
| 95th Percentile | {self.report_data['model_performance']['haversine_distances_km']['p95']:.1f} km |
| 99th Percentile | {self.report_data['model_performance']['haversine_distances_km']['p99']:.1f} km |

## Accuracy Analysis

### Geographic Accuracy Thresholds
| Distance Threshold | Count | Percentage |
|-------------------|-------|------------|
| ≤ 1 km | {self.report_data['accuracy_analysis']['predictions_within_1km']} | {self.report_data['accuracy_analysis']['percentage_within_1km']:.2f}% |
| ≤ 10 km | {self.report_data['accuracy_analysis']['predictions_within_10km']} | {self.report_data['accuracy_analysis']['percentage_within_10km']:.2f}% |
| ≤ 100 km | {self.report_data['accuracy_analysis']['predictions_within_100km']} | {self.report_data['accuracy_analysis']['percentage_within_100km']:.2f}% |
| ≤ 1000 km | {self.report_data['accuracy_analysis']['predictions_within_1000km']} | {self.report_data['accuracy_analysis']['percentage_within_1000km']:.2f}% |

## Performance Assessment

### Interpretation
- **Excellent Performance**: Mean error < 50 km, > 50% predictions within 100 km
- **Good Performance**: Mean error < 200 km, > 30% predictions within 100 km  
- **Moderate Performance**: Mean error < 500 km, > 20% predictions within 100 km
- **Poor Performance**: Mean error > 500 km, < 20% predictions within 100 km

### Current Model Assessment
Mean error: {self.report_data['model_performance']['haversine_distances_km']['mean']:.1f} km
Accuracy within 100 km: {self.report_data['accuracy_analysis']['percentage_within_100km']:.2f}%

*This report was generated automatically by the evaluation system.*
"""
        
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def _generate_csv_summary(self, output_path: Path) -> None:
        """Generate CSV summary of key metrics."""
        
        summary_data = {
            "Metric": [
                "MSE Loss",
                "Mean Coordinate Error (degrees)",
                "Median Coordinate Error (degrees)", 
                "Mean Haversine Distance (km)",
                "Median Haversine Distance (km)",
                "Predictions within 1km (%)",
                "Predictions within 10km (%)",
                "Predictions within 100km (%)",
                "Predictions within 1000km (%)"
            ],
            "Value": [
                f"{self.report_data['model_performance']['mse_loss']:.6f}",
                f"{self.report_data['model_performance']['coordinate_errors_degrees']['mean']:.3f}",
                f"{self.report_data['model_performance']['coordinate_errors_degrees']['median']:.3f}",
                f"{self.report_data['model_performance']['haversine_distances_km']['mean']:.1f}",
                f"{self.report_data['model_performance']['haversine_distances_km']['median']:.1f}",
                f"{self.report_data['accuracy_analysis']['percentage_within_1km']:.2f}",
                f"{self.report_data['accuracy_analysis']['percentage_within_10km']:.2f}",
                f"{self.report_data['accuracy_analysis']['percentage_within_100km']:.2f}",
                f"{self.report_data['accuracy_analysis']['percentage_within_1000km']:.2f}"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
    
    def print_summary(self) -> None:
        """Print a concise summary of evaluation results."""
        if not self.report_data:
            logger.warning("No evaluation data available. Run generate_comprehensive_report() first.")
            return
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Dataset Size: {self.report_data['evaluation_metadata']['dataset_size']} samples")
        print(f"MSE Loss: {self.report_data['model_performance']['mse_loss']:.6f}")
        print(f"Mean Error: {self.report_data['model_performance']['haversine_distances_km']['mean']:.1f} km")
        print(f"Median Error: {self.report_data['model_performance']['haversine_distances_km']['median']:.1f} km")
        print(f"Accuracy within 100 km: {self.report_data['accuracy_analysis']['percentage_within_100km']:.2f}%")
        print(f"Accuracy within 10 km: {self.report_data['accuracy_analysis']['percentage_within_10km']:.2f}%")
        print("="*60)