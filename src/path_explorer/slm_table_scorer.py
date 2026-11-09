"""
SLM table scoring for the Path Explorer (the proxy of the cascade, and Hippasus-SLM): cosine similarity between
schema-level texts of the base table and of each candidate table.
"""

from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity
from ..experiments.feature_descriptions import describe
from .slm_model_cache import get_or_load_model


class SLMTableScorer:
    """
    Scores candidate tables in [0, 1] by embedding similarity to the base table's context, in the format of the
    LLM's table scores.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = get_or_load_model(model_name)

    def score_tables(self, base_table_name, base_table_meta, target_column,
                     candidate_tables, feature_descs=None, join_keys=None):
        base_text = self._build_base_text(
            base_table_name, base_table_meta, target_column, feature_descs, join_keys
        )

        table_names = list(candidate_tables.keys())
        cand_texts = [
            self._build_candidate_text(name, candidate_tables[name], feature_descs, join_keys)
            for name in table_names
        ]

        base_emb = self.model.encode([base_text])
        cand_embs = self.model.encode(cand_texts)

        scores = {}
        for i, table_name in enumerate(table_names):
            sim = cosine_similarity(base_emb, cand_embs[i].reshape(1, -1))[0][0]
            scores[table_name] = float(max(0.0, min(1.0, sim)))

        return scores

    def _build_base_text(
        self,
        base_table_name: str,
        meta: dict,
        target_column: str,
        feature_descs: Optional[dict],
        join_keys: Optional[set],
    ) -> str:
        parts = [f"Base table: {base_table_name}", f"Predict: {target_column}"]
        for col in meta['columns']:
            if join_keys is not None and col in join_keys:
                continue
            desc = describe(feature_descs, col)
            if desc:
                parts.append(f"{col}: {desc}")
            else:
                parts.append(col)
        return " | ".join(parts)

    def _build_candidate_text(
        self,
        table_name: str,
        meta: dict,
        feature_descs: Optional[dict],
        join_keys: Optional[set],
    ) -> str:
        parts = [f"Table: {table_name}"]
        for col in meta['columns']:
            if join_keys is not None and col in join_keys:
                continue
            desc = describe(feature_descs, col)
            if desc:
                parts.append(f"{col}: {desc}")
            else:
                parts.append(f"{col} ({meta['dtypes'][col]})")
        return " | ".join(parts)


def test_slm_table_scorer_basic():
    scorer = SLMTableScorer(model_name= 'all-MiniLM-L6-v2')

    base_meta = {
        'columns': ['Key_0', 'age', 'income'],
        'dtypes': {'Key_0': 'int64', 'age': 'int64', 'income': 'float64'},
    }
    candidate_tables = {
        'orders': {
            'columns': ['Key_0', 'Key_1', 'amount'],
            'dtypes': {'Key_0': 'int64', 'Key_1': 'int64', 'amount': 'float64'},
        },
        'products': {
            'columns': ['Key_1', 'category', 'price'],
            'dtypes': {'Key_1': 'int64', 'category': 'object', 'price': 'float64'},
        },
    }
    join_keys = {'Key_0', 'Key_1'}
    scores = scorer.score_tables(
        'customers', base_meta, 'income', candidate_tables, join_keys=join_keys
    )
    assert set(scores.keys()) == {'orders', 'products'}
    for v in scores.values():
        assert 0.0 <= v <= 1.0
    print("test_slm_table_scorer_basic PASSED:", scores)


if __name__ == '__main__':
    test_slm_table_scorer_basic()
