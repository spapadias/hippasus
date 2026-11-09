"""
LLM reranking of features presented in descending MI order.
"""
import json
from typing import List, Dict, Optional

from ..experiments.feature_descriptions import describe
from ..llm_client import HippasusLLMClient, LLMResponseParseError


class LLMRankSelector:

    # Prompts: the features arrive sorted by MI, and the LLM makes minimal adjustments to that order
    INTEGRATION_PROMPT_V2_SYSTEM = (
        "You are a data science expert performing feature selection for {task_description}.\n\n"
        "The features below are PRE-SORTED by mutual information with the target variable "
        "(strongest predictor first). Your task is to make MINIMAL, TARGETED adjustments "
        "to this ranking using domain knowledge.\n\n"
        "MUTUAL INFORMATION (MI) measures the statistical dependency between each feature "
        "and the target. Higher MI = stronger measured relationship on the actual training "
        "data. MI works for all feature types including categorical and numeric.\n\n"
        "ADJUSTMENT RULES:\n"
        "1. KEEP the statistical order by default. MI reflects actual measured predictive "
        "power on the training data.\n"
        "2. PROMOTE a feature only when its description reveals strong causal or domain "
        "relevance that MI likely underestimates — for example, a feature that is "
        "theoretically a primary driver of the target but has moderate MI due to noise "
        "or small sample size.\n"
        "3. DEMOTE a feature only when you have clear evidence its high MI is artifactual:\n"
        "   - Unique identifiers: addresses, names, phone numbers, postal codes, codes, "
        "IDs, or administrative fields are often unique per row. Unique features always "
        "have high MI with any target — even very high MI scores for such features are a "
        "statistical artifact, not predictive power. Do not be misled by a high MI value "
        "for these fields. Demote them unless their description explicitly confirms direct "
        "causal relevance to the prediction task.\n"
        "   - Leakage: features collected after the prediction point or that directly "
        "encode the target.\n"
        "4. Do NOT demote a feature merely because its name is technical or unfamiliar. "
        "High MI means it has demonstrated statistical dependency with the target.\n"
        "5. Make the MINIMUM changes necessary. If uncertain, preserve the MI ranking.\n\n"
        "Requirements:\n"
        "(1) Return ALL {n_features} features — no exceptions, no omissions\n"
        "(2) Do NOT add, remove, rename, or modify any feature name — use exact names as provided\n"
        "(3) Each feature must appear EXACTLY ONCE in the output\n"
        "(4) Return a JSON array ordered from most to least important: [\"feature_1\", ...]"
    )

    INTEGRATION_PROMPT_V2_USER = (
        "Task: {task_description}\n"
        "Target: {target_column}\n\n"
        "Features pre-sorted by mutual information (rank 1 = strongest predictor):\n\n"
        "{feature_list}"
    )

    def __init__(self, component: str = "feature_selector"):
        self._llm_component = component
        self.llm_client = None

    def rank_features_mi_only(self, features: List[str], descriptions: Optional[Dict[str, str]],
                               mi_scores, target_col: str, task_description: str) -> tuple[List[str], Dict]:
        """
        LLM reranking of the features, presented in descending MI order.
        Returns (ranking, timings).
        """
        import time

        prep_start = time.time()

        sorted_features = sorted(features, key=lambda f: mi_scores.get(f, 0.0), reverse=True)
        mi_values = {f: mi_scores.get(f, 0.0) for f in features}

        feature_list = json.dumps([
            {
                "rank": i + 1,
                "MI": round(float(mi_values[feat]), 4),
                "feature": feat,
                "description": describe(descriptions, feat) or feat,
            }
            for i, feat in enumerate(sorted_features)
        ], indent=2)

        system_msg = self.INTEGRATION_PROMPT_V2_SYSTEM.format(
            task_description=task_description,
            n_features=len(features),
        )
        user_msg = self.INTEGRATION_PROMPT_V2_USER.format(
            task_description=task_description,
            target_column=target_col,
            feature_list=feature_list,
        )
        prep_time = time.time() - prep_start

        print(f" Calling LLM for MI-only feature ranking (v2)...")
        if self.llm_client is None:
            self.llm_client = HippasusLLMClient(component=self._llm_component)
        llm_start = time.time()
        fallback = None
        try:
            raw_ranking = self.llm_client.call_for_json_array(
                system_prompt=system_msg,
                user_prompt=user_msg,
                timeout=60,
            )
        except LLMResponseParseError as e:  # unusable reply: keep the MI ranking the LLM was asked to rerank
            print(f" Unusable LLM reply ({e}); keeping the MI ranking")
            raw_ranking, fallback = sorted_features, str(e)
        llm_time = time.time() - llm_start

        timings = {
            'preprocessing': prep_time,
            'llm_call': llm_time,
            'total': prep_time + llm_time,
        }
        if fallback:
            timings['llm_fallback'] = fallback
        ranked = self._process_ranking_response(raw_ranking, features)
        return ranked, timings

    def _process_ranking_response(self, raw_ranking: List[str], expected_features: List[str]) -> List[str]:
        """Process and validate LLM ranking response."""

        clean_ranking = []
        seen_features = set()

        for feature in raw_ranking:
            feature_str = str(feature).strip()
            if feature_str in expected_features and feature_str not in seen_features:
                clean_ranking.append(feature_str)
                seen_features.add(feature_str)

        # Add any missing features at the end (LLM might have skipped some)
        missing_features = set(expected_features) - seen_features
        if missing_features:
            clean_ranking.extend(sorted(missing_features))
            print(f" LLM missed {len(missing_features)} features, added at end: {missing_features}")

        if len(clean_ranking) != len(expected_features):
            raise RuntimeError(
                f"After processing: expected {len(expected_features)} features, got {len(clean_ranking)}")

        if len(clean_ranking) != len(set(clean_ranking)):
            raise RuntimeError("Duplicate features found after processing")

        return clean_ranking
