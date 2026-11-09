"""
Trains UnifiedMLTrainer on the selected features and reports the ensemble's metrics.
"""
import logging
from typing import Dict, List, Optional
import pandas as pd

from src.ml.ml_trainer import UnifiedMLTrainer


class HippasusModelTrainer:

    def __init__(
            self,
            target_column: str,
            task_type: str = "binary",
            algorithms: List[str] = None,
            time_limit: Optional[int] = None,
            random_seed: int = 42
    ):
        self.target_column = target_column
        self.task_type = task_type.lower()
        self.algorithms = algorithms or ["RF", "GBM", "XT"]
        self.time_limit = time_limit
        self.random_seed = random_seed
        self.logger = logging.getLogger(__name__)

        self.logger.info(f" Hippasus ML Trainer initialized:")
        self.logger.info(f"   Target: {self.target_column}")
        self.logger.info(f"   Task: {self.task_type}")
        self.logger.info(f"   Algorithms: {self.algorithms}")
        self.logger.info(f"   Time limit: {self.time_limit}s")
        self.logger.info(f"   Random seed: {self.random_seed}")

    def evaluate_dataset(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        self.logger.info(f" Evaluating dataset - Train: {train_df.shape}, Test: {test_df.shape}")

        # Handle prefixed target column names from consolidator
        target_column = self.target_column
        if target_column not in train_df.columns:
            prefixed_candidates = [col for col in train_df.columns if col.endswith(f".{target_column}")]
            if prefixed_candidates:
                target_column = prefixed_candidates[0]
                self.logger.info(f" Using prefixed target column: {target_column}")

        unified_trainer = UnifiedMLTrainer(
            target_column=target_column,
            task_type=self.task_type,
            models=self.algorithms,  # ['RF', 'GBM', 'XT']
            time_limit=self.time_limit,
            random_seed=self.random_seed,
            verbosity=0
        )

        self.logger.info(" Training models with UnifiedMLTrainer (single call + ensemble)...")
        all_metrics = unified_trainer.evaluate_dataset(train_df, test_df)

        results = self._convert_to_per_algorithm_format(all_metrics)

        self._log_results(results)

        return results

    def _convert_to_per_algorithm_format(self, all_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        results = {}

        # Report metrics under 'ENSEMBLE' key
        # This represents the best model (usually WeightedEnsemble) from AutoGluon
        results['ENSEMBLE'] = all_metrics.copy()

        return results

    def _log_results(self, results: Dict[str, Dict[str, float]]):
        """Log results based on task type"""

        if 'ENSEMBLE' not in results:
            self.logger.warning("  No ensemble results found")
            return

        metrics = results['ENSEMBLE']

        if self.task_type == "binary":
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1', 0)
            auc = metrics.get('auc')
            self.logger.info(f"    ENSEMBLE: Accuracy = {acc:.4f}, F1 = {f1:.4f}" +
                             (f", AUC = {auc:.4f}" if auc else ""))

        elif self.task_type == "multiclass":
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_macro', 0)
            self.logger.info(f"    ENSEMBLE: Accuracy = {acc:.4f}, F1-macro = {f1:.4f}")

        else:
            rmse = metrics.get('rmse', 0)
            mae = metrics.get('mae', 0)
            self.logger.info(f"    ENSEMBLE: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
