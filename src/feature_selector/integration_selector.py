"""
Feature Selector: MI and |Pearson| on the training rows, a ranking by the LLM or the SLM, and the top k features.
"""
import pandas as pd
from typing import List, Optional, Dict
from .stats_calculator import UnifiedStatsCalculator
from .llm_rank_selector import LLMRankSelector


class IntegrationSelector:
    """
    Feature Selector of Hippasus, Hippasus-LLM and Hippasus-SLM: computes MI and |Pearson| on the training rows,
    ranks the augmented features with the LLM (which reranks the MI order) or with the SLM, and keeps the top k.
    Datasets without semantic signal keep the top k by MI.
    """

    def __init__(self, max_features: Optional[int] = None, top_percentage: Optional[float] = None,
                 dataset_name: str = None, use_feature_descriptions: bool = False,
                 pre_filter_K: Optional[int] = None,
                 use_slm: bool = False, slm_model_name: str = 'all-MiniLM-L6-v2',
                 use_mi_only: bool = False):
        self.max_features = max_features
        self.top_percentage = top_percentage or 0.3
        self.pre_filter_K = pre_filter_K
        self.dataset_name = dataset_name
        self.use_feature_descriptions = use_feature_descriptions
        self.use_mi_only = use_mi_only
        self.stats_calculator = UnifiedStatsCalculator(random_state=42)
        self.use_slm = use_slm
        self.slm_ranker = None
        if use_slm:
            from .slm_rank_selector import SLMRankSelector
            self.slm_ranker = SLMRankSelector(model_name=slm_model_name)
        self.llm_ranker = None
        if not use_slm:
            self.llm_ranker = LLMRankSelector()

    def select_features(self,
                        train_df: pd.DataFrame,
                        target_col: str,
                        task_type: str,
                        base_features: List[str],
                        augmented_features: List[str]) -> tuple[List[str], Dict]:

        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in train_df")

        if not augmented_features:
            print("    Integration: No augmented features to select from")
            return [], {'total': 0.0, 'stats_computation': 0.0,
                        'llm_ranking': {'total': 0.0, 'preprocessing': 0.0, 'llm_call': 0.0}}

        if self.max_features == 0:
            print(f"    Integration: max_features=0, bypassing LLM (returning 0 augmented features)")
            print(f"      This will train on base table only ({len(base_features)} base features)")
            return [], {'total': 0.0, 'stats_computation': 0.0,
                        'llm_ranking': {'total': 0.0, 'preprocessing': 0.0, 'llm_call': 0.0}}

        # A pool too large for one prompt: the prefilter (pre_filter_K) bounds what the LLM sees; without one, rank
        # by MI alone, as without semantic signal.
        too_many_for_llm = self.pre_filter_K is None and len(augmented_features) >= 700

        print(f"\n    Integration: Processing {len(augmented_features)} augmented features")
        print(f"      (Base features: {len(base_features)} will be preserved by caller)")

        X_augmented = train_df[augmented_features]
        y = train_df[target_col]

        import time
        selection_start = time.time()

        # 1) Statistics on the training rows
        print("       Computing statistics for LLM integration...")
        stats_start = time.time()
        stats_dict = self.stats_calculator.compute_all_statistics(
            X_augmented,
            y,
            task_type,
        )
        stats_time = time.time() - stats_start

        # 2) No semantic signal, or a pool too large without a prefilter: the top k by MI, without a language model
        from src.experiments.feature_descriptions import has_semantic_signal
        if too_many_for_llm or not has_semantic_signal(self.dataset_name):
            reason = (f"{len(augmented_features)} features and no prefilter" if too_many_for_llm
                      else f"dataset '{self.dataset_name}' has no semantic signal")
            print(f"    [FS] {reason} — using stats-only ranking (no LLM call)")
            k = self._determine_k(len(augmented_features))
            # MI alone for every variant, as for the statistics-only selector, so the variants coincide
            selected_augmented = stats_dict['mutual_info'].nlargest(k).index.tolist()
            total_time = time.time() - selection_start
            timings = {
                'total': total_time,
                'stats_computation': stats_time,
                'llm_ranking': {'total': 0.0, 'preprocessing': 0.0, 'llm_call': 0.0},
                'pre_filter_applied': False,
                'pre_filter_K': None,
                'features_before_prefilter': len(augmented_features),
                'features_after_prefilter': len(augmented_features),
            }
            print(f"       Selected {len(selected_augmented)}/{len(augmented_features)} "
                  f"augmented features (MI fallback)")
            return selected_augmented, timings

        # 3) Prefilter to the top pre_filter_K features: by MI for the LLM, by Borda over MI and |Pearson| for the SLM
        features_for_llm = augmented_features
        pre_filter_applied = False

        if self.pre_filter_K is not None and len(augmented_features) > self.pre_filter_K:
            print(f"       PRE-FILTERING: {len(augmented_features)} features > K={self.pre_filter_K}")

            if self.use_mi_only:
                print(f"         Applying MI-only ranking to select top-{self.pre_filter_K}")
                top_k_features = stats_dict['mutual_info'].nlargest(self.pre_filter_K).index.tolist()
            else:
                print(f"         Applying Borda ranking on 3 statistics to select top-{self.pre_filter_K}")
                borda_scores = self.stats_calculator.create_borda_ranking(
                    stats_dict,
                    stats_to_use=['mutual_info', 'pearson_corr']
                )
                top_k_features = borda_scores.nsmallest(self.pre_filter_K).index.tolist()

            features_for_llm = top_k_features
            pre_filter_applied = True

            print(f"          Pre-filtered to {len(features_for_llm)} features for LLM ranking")
            print(f"          Top-5 pre-filtered: {features_for_llm[:5]}")

            X_augmented = X_augmented[features_for_llm]

        stats_context = self._prepare_stats_context(features_for_llm, stats_dict)

        # 4) Rank: the SLM (cosine similarity, Borda-fused with MI and |Pearson|) or the LLM (reranks the MI order)
        ranker_label = 'SLM' if self.use_slm else 'LLM'
        print(f"       Getting {ranker_label} integrated ranking for {len(features_for_llm)} features...")

        if self.use_slm:
            feature_descs = None
            if self.use_feature_descriptions and self.dataset_name:
                from ..experiments.feature_descriptions import get_dataset_features
                feature_descs = get_dataset_features(self.dataset_name)
            llm_integrated_ranking = self.slm_ranker.rank_features(
                features_for_llm,
                target_col,
                feature_descs=feature_descs,
                stats_context=stats_context,
            )
            llm_timings = {'total': 0.0, 'preprocessing': 0.0, 'llm_call': 0.0}
        else:
            if task_type == "binary":
                task_desc = "Binary Classification (predict one of two classes)"
            elif task_type == "multiclass":
                task_desc = "Multiclass Classification (predict one of several categories)"
            else:
                task_desc = "Regression (predict continuous numeric value)"
            feature_descs = None
            if self.use_feature_descriptions and self.dataset_name:
                from ..experiments.feature_descriptions import get_dataset_features
                feature_descs = get_dataset_features(self.dataset_name)
            mi_scores_dict = {f: float(stats_dict['mutual_info'].get(f, 0.0)) for f in features_for_llm}
            llm_integrated_ranking, llm_timings = self.llm_ranker.rank_features_mi_only(
                features_for_llm, feature_descs, mi_scores_dict,
                target_col, task_desc,
            )

        # 5) Keep the top k
        k = self._determine_k(len(features_for_llm))
        selected_augmented = llm_integrated_ranking[:k]

        total_time = time.time() - selection_start
        timings = {
            'total': total_time,
            'stats_computation': stats_time,
            'llm_ranking': llm_timings,
            'pre_filter_applied': pre_filter_applied,
            'pre_filter_K': self.pre_filter_K if pre_filter_applied else None,
            'features_before_prefilter': len(augmented_features),
            'features_after_prefilter': len(features_for_llm)
        }

        if pre_filter_applied:
            print(f"       Selected {len(selected_augmented)}/{len(features_for_llm)} from pre-filtered set")
            print(
                f"         (Original pool: {len(augmented_features)} → Pre-filtered: {len(features_for_llm)} → Final: {len(selected_augmented)})")
        else:
            print(f"       Selected {len(selected_augmented)}/{len(augmented_features)} augmented features")

        print(f"       Final: {selected_augmented[:min(5, len(selected_augmented))]}")
        print(f"        Timing: stats={stats_time:.2f}s, llm={llm_timings['total']:.2f}s "
              f"(prep={llm_timings['preprocessing']:.2f}s, call={llm_timings['llm_call']:.2f}s)")

        return selected_augmented, timings

    def _prepare_stats_context(self, features: List[str], stats_dict: Dict[str, pd.Series]) -> Dict[
        str, Dict[str, float]]:
        stats_context = {}
        for feature in features:
            stats_context[feature] = {
                'mutual_info': float(stats_dict['mutual_info'].get(feature, 0.0)),
                'pearson_corr': float(stats_dict['pearson_corr'].get(feature, 0.0)),
            }
        return stats_context

    def _determine_k(self, n_features: int) -> int:
        """Determine number of features to select."""
        if self.max_features is not None:
            return min(self.max_features, n_features)
        else:
            return max(1, int(n_features * self.top_percentage))
