"""
Feature Selector of Hippasus-Stats: the top k augmented features by mutual information.
"""
import pandas as pd
from typing import List, Optional
from .stats_calculator import UnifiedStatsCalculator


class StatsOnlySelector:

    def __init__(self, max_features: Optional[int] = None, top_percentage: Optional[float] = None):
        self.max_features = max_features
        self.top_percentage = top_percentage or 0.3
        self.stats_calculator = UnifiedStatsCalculator(random_state=42)

    def select_features(self,
                        train_df: pd.DataFrame,
                        target_col: str,
                        task_type: str,
                        base_features: List[str],
                        augmented_features: List[str]) -> List[str]:

        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in train_df")

        if not augmented_features:
            print("    StatsBorda: No augmented features to select from")
            return []

        print(f"\n    StatsBorda: Ranking {len(augmented_features)} augmented features")
        print(f"      (Base features: {len(base_features)} will be preserved by caller)")

        X_augmented = train_df[augmented_features]
        y = train_df[target_col]

        stats_dict = self.stats_calculator.compute_all_statistics(
            X_augmented,
            y,
            task_type,
        )

        # MI-only ranking: sort by mutual information descending (higher = better)
        ranking_df = pd.DataFrame({
            'feature': augmented_features,
            'mutual_info': stats_dict['mutual_info'],
            'pearson_corr': stats_dict['pearson_corr'],
        }).set_index('feature')
        ranking_df = ranking_df.sort_values('mutual_info', ascending=False)

        k = self._determine_k(len(augmented_features))
        selected_augmented = ranking_df.index.tolist()[:k]

        print(f"       Selected {len(selected_augmented)}/{len(augmented_features)} augmented features")
        print(f"       Top-5: {selected_augmented[:min(5, len(selected_augmented))]}")

        return selected_augmented

    def _determine_k(self, n_features: int) -> int:
        """Determine number of features to select."""
        if self.max_features is not None:
            return min(self.max_features, n_features)
        else:
            return max(1, int(n_features * self.top_percentage))
