"""
SLM ranking for the Feature Selector of Hippasus-SLM: cosine similarity between each feature's text and the
target's (sentence-transformers), fused with the MI and |Pearson| rankings by Borda count.
"""

from typing import Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity
from src.experiments.feature_descriptions import describe
from src.path_explorer.slm_model_cache import get_or_load_model


class SLMRankSelector:
    """
    Ranks features by embedding similarity to the target, fused with the MI and |Pearson| rankings by Borda count.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = get_or_load_model(model_name)

    def rank_features(
        self,
        features: List[str],
        target_col: str,
        stats_context: Dict[str, Dict[str, float]],
        feature_descs: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Rank features by cosine similarity to the target column embedding,
        fused with the MI and |Pearson| rankings via Borda count.

        Parameters
        ----------
        features : list of str
            Augmented feature names to rank.
        target_col : str
            Name of the target column.
        stats_context : dict
            {feature_name: {'mutual_info': float, 'pearson_corr': float}}
        feature_descs : dict, optional
            {col_name: description} from get_dataset_features().

        Returns
        -------
        list of str
            Features ordered from most to least important.
        """
        if not features:
            return []

        target_desc = describe(feature_descs, target_col)
        if target_desc:
            target_text = f"Predict {target_col}: {target_desc}"
        else:
            target_text = f"Predict: {target_col.replace('_', ' ')}"

        feature_texts = []
        for feat in features:
            desc = describe(feature_descs, feat)
            if desc:
                feature_texts.append(f"{feat}: {desc}")
            else:
                feature_texts.append(feat.replace('_', ' '))

        all_texts = [target_text] + feature_texts
        embeddings = self.model.encode(all_texts)
        target_emb = embeddings[0].reshape(1, -1)
        feature_embs = embeddings[1:]

        sims = cosine_similarity(feature_embs, target_emb).flatten()
        semantic_scores = {feat: float(sims[i]) for i, feat in enumerate(features)}

        # Borda fusion: semantic + MI + |Pearson|
        n = len(features)

        def rank_descending(score_dict: Dict[str, float]) -> Dict[str, int]:
            """0-indexed rank: 0 = best (highest score)."""
            ordered = sorted(score_dict, key=score_dict.get, reverse=True)
            return {feat: i for i, feat in enumerate(ordered)}

        semantic_rank = rank_descending(semantic_scores)
        mi_rank = rank_descending(
            {f: stats_context[f].get('mutual_info', 0.0) for f in features}
        )

        pearson_rank = rank_descending(
            {f: abs(stats_context[f].get('pearson_corr', 0.0)) for f in features}
        )
        borda = {
            f: (n - 1 - semantic_rank[f])
             + (n - 1 - mi_rank[f])
             + (n - 1 - pearson_rank[f])
            for f in features
        }

        return sorted(features, key=lambda f: borda[f], reverse=True)
