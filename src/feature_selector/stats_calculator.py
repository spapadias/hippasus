"""
Feature statistics on the training rows: mutual information, |Pearson| correlation and their Borda ranking.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder


class UnifiedStatsCalculator:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def compute_all_statistics(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            task_type: str,
    ) -> Dict[str, pd.Series]:
        X_encoded, categorical_mask = self._encode_mixed_types(X)

        stats_dict = {
            'pearson_corr': self._compute_pearson_correlation(X, y),
            'mutual_info': self._compute_mutual_info(X_encoded, y, task_type, categorical_mask),
        }

        return stats_dict

    def create_borda_ranking(self, stats_dict: Dict[str, pd.Series],
                             stats_to_use: List[str]) -> pd.Series:
        ranks = {}

        for stat_name in stats_to_use:
            if stat_name in stats_dict:
                stat_values = stats_dict[stat_name]
                ranks[stat_name] = stat_values.rank(ascending=False, method='average')

        # Borda: sum of ranks (lower total = better)
        borda_scores = sum(ranks.values())

        return borda_scores.rename('borda_score')

    def _compute_pearson_correlation(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Compute Pearson correlation (linear relationships).
        Time Complexity: O(n)
        """
        correlations = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(y):
                try:
                    corr = X[col].corr(y)
                    correlations.append(abs(corr) if not pd.isna(corr) else 0.0)
                except:
                    correlations.append(0.0)
            else:
                correlations.append(0.0)

        return pd.Series(correlations, index=X.columns, name='pearson_corr')

    def _compute_mutual_info(
            self,
            X_encoded: pd.DataFrame,
            y: pd.Series,
            task_type: str,
            categorical_mask: np.ndarray
    ) -> pd.Series:
        """
        Compute mutual information.
        Time Complexity: O(n log n)
        """
        # An unknown task name raises: a fallback score would hide the error.
        if task_type in ("binary", "multiclass"):
            scores = mutual_info_classif(
                X_encoded.values, y.values,
                discrete_features=categorical_mask,
                random_state=self.random_state
            )
        elif task_type == "regression":
            scores = mutual_info_regression(
                X_encoded.values, y.values,
                discrete_features=categorical_mask,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown task type '{task_type}': expected binary, multiclass or regression")

        return pd.Series(scores, index=X_encoded.columns, name='mutual_info')

    def _encode_mixed_types(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Encode mixed data types for ML algorithms.

        Returns:
            Tuple of (encoded DataFrame, categorical mask)
        """
        categorical_mask = np.array([
            pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col])
            for col in X.columns
        ])

        X_encoded = X.copy()

        if categorical_mask.any():
            cat_cols = X.columns[categorical_mask]

            # Convert pandas NA to numpy NaN for sklearn compatibility
            X_encoded[cat_cols] = X_encoded[cat_cols].fillna(np.nan)

            try:
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                X_encoded[cat_cols] = encoder.fit_transform(X_encoded[cat_cols])
            except:
                for col in cat_cols:
                    X_encoded[col] = pd.factorize(X_encoded[col])[0]

        X_encoded = X_encoded.fillna(-1)

        return X_encoded, categorical_mask
