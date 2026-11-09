"""
AutoGluon training and evaluation: random forests, gradient boosting, extra trees and their weighted ensemble.
"""
import logging
import tempfile
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from autogluon.core import TabularDataset
from autogluon.tabular import TabularPredictor

class UnifiedMLTrainer:

    def __init__(
            self,
            target_column: str,
            task_type: str,  # "binary", "multiclass", "regression"
            models: Optional[List[str]] = None,  # ['RF', 'GBM', 'XT'] or None
            time_limit: int = 30,
            random_seed: int = 42,
            verbosity: int = 0
    ):
        self.target_column = target_column
        self.task_type = task_type.lower()
        self.models = models
        self.time_limit = time_limit
        self.random_seed = random_seed
        self.verbosity = verbosity

        self.logger = logging.getLogger(__name__)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "models"
        self.model_path.mkdir(exist_ok=True, parents=True)

        self.logger.info(f" Unified ML Trainer initialized:")
        self.logger.info(f"   Target: {self.target_column}")
        self.logger.info(f"   Task: {self.task_type}")
        self.logger.info(f"   Models: {self.models or 'AutoGluon default'}")
        self.logger.info(f"   Time limit: {self.time_limit}s")
        self.logger.info(f"   Random seed: {self.random_seed}")

    def evaluate_dataset(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame
    ) -> Dict[str, float]:
        self.logger.info(f" Training - Train: {train_df.shape}, Test: {test_df.shape}")

        # Set reproducibility seeds
        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if self.target_column not in train_df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in training data")

        train_df = train_df.copy()
        test_df = test_df.copy()

        if self.task_type in ["binary", "multiclass"]:
            train_df[self.target_column] = train_df[self.target_column].astype(str)
            test_df[self.target_column] = test_df[self.target_column].astype(str)
        else:
            train_df[self.target_column] = pd.to_numeric(
                train_df[self.target_column], errors='coerce'
            )
            test_df[self.target_column] = pd.to_numeric(
                test_df[self.target_column], errors='coerce'
            )

        train_df = train_df.dropna(subset=[self.target_column])
        test_df = test_df.dropna(subset=[self.target_column])

        train_data = TabularDataset(train_df)
        test_data = TabularDataset(test_df)

        if self.task_type == "regression":
            eval_metric = 'root_mean_squared_error'
        else:
            eval_metric = 'accuracy'

        if self.models:
            # Use specific models + let AutoGluon create ensemble
            hyperparameters = {model: {} for model in self.models}
            self.logger.info(f"   Using {len(self.models)} models: {self.models}")
        else:
            # Let AutoGluon choose models
            hyperparameters = None
            self.logger.info(f"   Using AutoGluon default model selection")

        import time
        unique_path = self.model_path / f"ag_{int(time.time() * 1000000)}"

        try:
            predictor = TabularPredictor(
                label=self.target_column,
                eval_metric=eval_metric,
                verbosity=self.verbosity,
                path=str(unique_path),
                problem_type=self.task_type
            ).fit(
                train_data=train_data,
                hyperparameters=hyperparameters,
                time_limit=self.time_limit,
                num_bag_folds=0,  # Disable bagging for speed
                num_stack_levels=0,  # Disable stacking for speed
                auto_stack=False  # Disable auto-stacking
            )

            try:
                leaderboard = predictor.leaderboard(silent=True)
                print(f" Models trained:")
                for idx, row in leaderboard.iterrows():
                    print(f"   {row['model']}: score={row.get('score_val', 'N/A')}")

                best_model = predictor.model_best
                print(f" Best model (used for predictions): {best_model}")
            except Exception as e:
                print(f"  Could not log model details: {e}")

            predictions = predictor.predict(test_data)
            y_true = test_df[self.target_column]

            metrics = self._calculate_metrics(y_true, predictions, predictor, test_data)

            return metrics

        except Exception as e:
            self.logger.error(f" Training failed: {e}")
            import traceback
            traceback.print_exc()

            return self._get_empty_metrics()

    def _calculate_metrics(
            self,
            y_true: pd.Series,
            predictions: pd.Series,
            predictor: TabularPredictor,
            test_data: TabularDataset
    ) -> Dict[str, float]:
        """Calculate task-appropriate metrics"""

        metrics = {}

        if self.task_type == "binary":
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

            metrics['accuracy'] = accuracy_score(y_true, predictions)

            unique_labels = sorted(y_true.unique())
            if len(unique_labels) == 2:
                if set(unique_labels) == {0, 1} or set(unique_labels) == {'0', '1'}:
                    pos_label = 1 if 1 in unique_labels else '1'
                else:
                    pos_label = unique_labels[1]  # Second alphabetically
                metrics['f1'] = f1_score(y_true, predictions, pos_label=pos_label)
            else:
                metrics['f1'] = f1_score(y_true, predictions, average='binary')

            try:
                y_proba = predictor.predict_proba(test_data)
                if hasattr(y_proba, 'values'):
                    y_proba = y_proba.values
                if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC: {e}")
                metrics['auc'] = None

            self.logger.info(f"    Accuracy: {metrics['accuracy']:.4f}, "
                             f"F1: {metrics['f1']:.4f}" +
                             (f", AUC: {metrics['auc']:.4f}" if metrics['auc'] else ""))

        elif self.task_type == "multiclass":
            from sklearn.metrics import accuracy_score, f1_score

            metrics['accuracy'] = accuracy_score(y_true, predictions)
            metrics['f1_macro'] = f1_score(y_true, predictions, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, predictions, average='weighted')

            self.logger.info(f"    Accuracy: {metrics['accuracy']:.4f}, "
                             f"F1-macro: {metrics['f1_macro']:.4f}")

        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            mse = mean_squared_error(y_true, predictions)
            metrics['rmse'] = np.sqrt(mse)
            metrics['mse'] = mse
            metrics['mae'] = mean_absolute_error(y_true, predictions)

            try:
                metrics['r2'] = r2_score(y_true, predictions)
            except:
                metrics['r2'] = None

            # MAPE (with zero protection)
            try:
                mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
                metrics['mape'] = mape if np.isfinite(mape) else None
            except:
                metrics['mape'] = None

            self.logger.info(f"    RMSE: {metrics['rmse']:.4f}, "
                             f"MAE: {metrics['mae']:.4f}" +
                             (f", R²: {metrics['r2']:.4f}" if metrics['r2'] else ""))

        return metrics

    def _get_empty_metrics(self) -> Dict[str, Optional[float]]:
        """Return empty metrics structure for failed training"""
        if self.task_type == "binary":
            return {'accuracy': None, 'f1': None, 'auc': None}
        elif self.task_type == "multiclass":
            return {'accuracy': None, 'f1_macro': None, 'f1_weighted': None}
        else:
            return {'rmse': None, 'mse': None, 'mae': None, 'r2': None, 'mape': None}

    def __del__(self):
        """Cleanup temp directory"""
        try:
            self.temp_dir.cleanup()
        except:
            pass
