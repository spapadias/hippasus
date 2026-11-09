"""
Path Explorer: loads the tables and their connections, scores the candidate tables (statistics, SLM, LLM, or
the SLM-LLM cascade) and searches join paths from the base table.
"""
import os
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set, Union
from collections import deque
import logging
import time

from ..experiments.feature_descriptions import describe
from ..llm_client import HippasusLLMClient, LLMResponseParseError


class NullNormalizer:
    """Centralized null normalization for consistent handling across all components"""

    @staticmethod
    def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize common null representations to pandas NA for object/string columns only"""
        df_clean = df.copy()

        null_sentinels = {
            "", " ", "NA", "N/A", "None", "null", "NULL", "\\N", "nan", "NaN"
        }

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                series = df_clean[col].astype(str).str.strip()
                df_clean[col] = series.replace(null_sentinels, pd.NA)

        return df_clean


class DatasetLoader:
    """Load and analyze CSV dataset with centralized null normalization"""

    def __init__(self, dataset_folder: str, base_table_name: str = None):
        self.folder = Path(dataset_folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

        self.tables = {}
        self.connections = []
        self.base_table = Path(base_table_name).stem if base_table_name else None
        self._load_dataset()

    def _load_dataset(self):
        """Load all CSV files with centralized null normalization"""
        logging.info(f"Loading dataset from: {self.folder}")

        # sorted: directory listing order differs between file systems
        csv_files = [f for f in sorted(self.folder.glob("*.csv")) if f.name != "connections.csv"]

        for csv_file in csv_files:
            table_name = csv_file.stem
            try:
                # Try multiple encodings to handle corrupted files
                for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(csv_file, dtype=str, low_memory=False, encoding=encoding, on_bad_lines='skip')
                        break
                    except Exception as e:
                        if encoding == 'cp1252':
                            raise e
                        continue

                df = NullNormalizer.normalize_nulls(df)

                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="ignore")

                self.tables[table_name] = {
                    'path': str(csv_file),
                    'columns': list(df.columns),
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'distinct_counts': {col: df[col].nunique() for col in df.columns},
                    'null_rates': {col: df[col].isnull().mean() for col in df.columns},
                    'dtypes': {col: str(df[col].dtype) for col in df.columns}
                }

                logging.debug(f"Loaded table {table_name}: {df.shape}")

            except Exception as e:
                logging.error(f"Failed to load {csv_file}: {e}")
                continue

        if not self.tables:
            raise ValueError(f"No valid CSV files found in {self.folder}")

        self._load_connections()
        logging.info(f"Dataset loaded: {len(self.tables)} tables, {len(self.connections)} connections")

    def _load_connections(self):
        """Load foreign key relationships from connections.csv"""
        connections_file = self.folder / "connections.csv"

        if not connections_file.exists():
            logging.warning("No connections.csv found")
            return

        try:
            conn_df = pd.read_csv(connections_file)
            required_cols = ['fk_table', 'pk_table', 'fk_column', 'pk_column']

            if not all(col in conn_df.columns for col in required_cols):
                raise ValueError(f"connections.csv must have columns: {required_cols}")

            for _, row in conn_df.iterrows():
                fk_table = Path(row['fk_table']).stem
                pk_table = Path(row['pk_table']).stem

                if fk_table in self.tables and pk_table in self.tables:
                    self.connections.append({
                        'table1': fk_table,
                        'table2': pk_table,
                        'key1': str(row['fk_column']),
                        'key2': str(row['pk_column'])
                    })

            logging.info(f"Loaded {len(self.connections)} valid connections")

        except Exception as e:
            logging.error(f"Failed to load connections.csv: {e}")
            self.connections = []

    def get_join_keys_info(self) -> Dict[str, Union[Set[str], List[Dict]]]:
        """Extract join key information for downstream components"""
        all_join_keys = set()

        for conn in self.connections:
            all_join_keys.add(conn['key1'])
            all_join_keys.add(conn['key2'])

        return {
            'all_join_keys': all_join_keys,
            'connections': self.connections.copy()
        }


class PathExplorer:
    """Scores the candidate tables (phase 1), then searches join paths from the base table (phase 2)"""

    def __init__(self,
                 base_df: pd.DataFrame,
                 dataset_folder: str,
                 target_column: str,
                 task_type: str,
                 max_depth: int = 3,
                 top_k: int = 15,
                 base_table_name: str = None,
                 use_feature_descriptions: bool = False,
                 use_slm: bool = False,
                 slm_model_name: str = 'all-MiniLM-L6-v2',
                 use_stats_only: bool = False,
                 use_adaptive: bool = False,
                 adaptive_tau: float = None):

        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if base_df.empty:
            raise ValueError("base_df cannot be empty")
        if target_column not in base_df.columns:
            raise ValueError(f"target_column '{target_column}' not found in base_df")

        self.base_df = NullNormalizer.normalize_nulls(base_df)
        self.target_column = target_column
        self.max_depth = max_depth
        self.top_k = top_k

        self.dataset = DatasetLoader(dataset_folder, base_table_name)
        self.task_type = task_type.lower()
        self.dataset_name = Path(dataset_folder).name

        self.use_slm = use_slm
        self.use_stats_only = use_stats_only
        self.use_adaptive = use_adaptive
        self.adaptive_tau = adaptive_tau
        if use_stats_only and use_slm:
            raise ValueError("use_stats_only and use_slm cannot both be True")
        if use_adaptive and use_stats_only:
            raise ValueError("use_adaptive and use_stats_only cannot both be True")
        if use_adaptive and use_slm:
            raise ValueError("use_adaptive and use_slm cannot both be True")

        self.slm_scorer = None
        if self.use_slm or self.use_adaptive:
            from .slm_table_scorer import SLMTableScorer
            self.slm_scorer = SLMTableScorer(model_name=slm_model_name)

        self.llm_client = None
        if not self.use_slm and not self.use_stats_only and not self.use_adaptive:
            self.llm_client = HippasusLLMClient(component="path_explorer")

        self.use_feature_descriptions = use_feature_descriptions or os.environ.get('HIPPASUS_PE_FEATURE_DESC',
                                                                                   'false').lower() == 'true'
        self.table_scores = {}
        self.connection_stats = {}
        self.top_k_paths = []
        self.stats = {
            'paths_evaluated': 0,
            'total_time': 0.0,
            'phase1_timings': {
                'total': 0.0,
                'preprocessing': 0.0,
                'llm_call': 0.0
            },
            'phase2_timings': {
                'total': 0.0,
                'path_generation': 0.0,
                'path_scoring': 0.0
            }
        }

        logging.info(f"PathExplorer initialized with centralized null normalization")

    def explore(self) -> Tuple[List[List[str]], Dict[str, Union[Set[str], List[Dict]]], Dict]:
        """Main exploration method - returns (paths, join_keys_info, timings)"""
        start_time = time.time()

        self._phase1_llm_table_scoring()
        self._phase2_guided_exploration()

        self.stats['total_time'] = time.time() - start_time
        final_paths = [path for path, score in self.top_k_paths]
        join_keys_info = self.dataset.get_join_keys_info()

        logging.info(f"Exploration completed in {self.stats['total_time']:.2f}s")
        logging.info(f"Final results: {len(final_paths)} paths discovered")
        logging.info(f"Join keys tracked: {len(join_keys_info['all_join_keys'])} unique keys")

        return final_paths, join_keys_info, self.stats

    def _assess_proxy_confidence(self, proxy_scores: dict, pi: int) -> dict:
        """
        Measure confidence of SLM proxy ranking using two complementary signals:

        1. boundary_gap: normalized score difference between the pi-th and
           (pi+1)-th ranked table. Large gap = confident separation at boundary.
           Range: [0, 1] after min-max normalization.

        2. score_entropy: Shannon entropy of the normalized score distribution.
           Low entropy = one table dominates = confident.
           High entropy = scores uniform = uncertain.
           Normalized to [0, 1] where 0 = maximally confident, 1 = maximally uncertain.

        The cascade decides on the boundary gap; the entropy is reported alongside it.
        """
        import math

        sorted_tables = sorted(proxy_scores, key=proxy_scores.get, reverse=True)
        raw_scores = [proxy_scores[t] for t in sorted_tables]
        n = len(raw_scores)

        # Min-max normalization
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        if score_range > 0:
            norm_scores = [(s - min_score) / score_range for s in raw_scores]
        else:
            norm_scores = [0.5] * n  # all identical → maximally uncertain

        # Signal 1: boundary gap
        if n > pi:
            boundary_gap = norm_scores[pi - 1] - norm_scores[pi]
        else:
            # No more tables than pi: no boundary gap, and the sentinel -1.0 flags this degenerate case
            boundary_gap = -1.0

        # Signal 2: normalized Shannon entropy of the scores (reported; the decision uses the boundary gap)
        # Convert normalized scores to a probability distribution via softmax
        # to avoid division by zero and ensure valid probability distribution
        temperature = 1.0
        exp_scores = [math.exp(s / temperature) for s in norm_scores]
        sum_exp = sum(exp_scores)
        probs = [e / sum_exp for e in exp_scores]

        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        max_entropy = math.log(n) if n > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Confidence from entropy: high entropy = uncertain = low confidence
        # We invert so that confidence_entropy behaves like boundary_gap:
        # large value = confident, small value = uncertain
        entropy_confidence = 1.0 - normalized_entropy

        return {
            'sorted_tables': sorted_tables,
            # Signal 1: boundary gap
            'boundary_gap': round(float(boundary_gap), 6),
            # Signal 2: entropy
            'normalized_entropy': round(float(normalized_entropy), 6),
            'entropy_confidence': round(float(entropy_confidence), 6),
            # Raw and normalized scores for diagnostics
            'raw_scores': {t: round(float(proxy_scores[t]), 6)
                           for t in sorted_tables},
            'normalized_scores': {t: round(float(norm_scores[i]), 6)
                                  for i, t in enumerate(sorted_tables)},
        }

    def _adaptive_table_scoring(self, tables_to_score: list, tau: float) -> tuple:
        """
        Adaptive scoring: SLM proxy first, LLM oracle if uncertain.

        Returns:
          - table_scores: dict {table_name: float} to feed into path scoring
          - cascade_info: dict with decision metadata
        """
        # Step 1: Always compute SLM proxy scores
        feature_descs = None
        if self.use_feature_descriptions:
            from ..experiments.feature_descriptions import get_dataset_features
            feature_descs = get_dataset_features(self.dataset_name)
        join_keys = self.dataset.get_join_keys_info()['all_join_keys']

        proxy_scores = self.slm_scorer.score_tables(
            base_table_name=self.dataset.base_table,
            base_table_meta=self.dataset.tables[self.dataset.base_table],
            target_column=self.target_column,
            candidate_tables={name: self.dataset.tables[name] for name in tables_to_score},
            feature_descs=feature_descs,
            join_keys=join_keys
        )

        # Step 2: Assess confidence at boundary pi = top_k
        confidence = self._assess_proxy_confidence(proxy_scores, self.top_k)

        confidence_value = confidence['boundary_gap']

        # Step 3: decision rule
        # tau=None  → always oracle (cascade disabled).
        # tau=float  → oracle when confidence_value < tau; tau=999 → proxy always.
        cascade_info_extra = {}
        if isinstance(tau, (int, float)) and tau > 1.0:
            # tau exceeds the maximum possible gap [0,1] → proxy always, never invoke oracle
            invoke_oracle = False
        elif confidence['boundary_gap'] == -1.0:
            # Degenerate case: n_tables <= pi, no meaningful boundary gap
            invoke_oracle = True
        elif tau is None:
            invoke_oracle = True
        else:
            # Fixed tau (0 < tau <= 1.0): small gap (< tau) → uncertain → oracle
            # Large gap (>= tau) → confident → proxy
            invoke_oracle = (confidence_value < tau)

        if invoke_oracle:
            # Uncertain or oracle-forced: invoke LLM (lazy-init client on first oracle call)
            if self.llm_client is None:
                self.llm_client = HippasusLLMClient(component="path_explorer")
            decision = "oracle_invoked"
            llm_start = time.time()
            batch_prompt = self._create_batch_scoring_prompt(tables_to_score)
            table_scores = self._llm_table_scores(batch_prompt, tables_to_score)
            if table_scores is None:  # unusable reply: keep the proxy's scores
                decision = "oracle_failed_proxy_used"
                table_scores = proxy_scores
                cascade_info_extra['llm_fallback'] = self.stats['llm_fallback']
            llm_time = time.time() - llm_start

            cascade_info = {
                'decision': decision,
                'confidence_metric': 'boundary_gap',
                'confidence_value': round(float(confidence_value), 6),
                'boundary_gap': confidence['boundary_gap'],
                'normalized_entropy': confidence['normalized_entropy'],
                'entropy_confidence': confidence['entropy_confidence'],
                'tau': tau,
                'proxy_scores': confidence['raw_scores'],
                'normalized_scores': confidence.get('normalized_scores', {}),
                'llm_invoked': True,
                'llm_time': round(llm_time, 3),
                **cascade_info_extra,
            }
        else:
            # Confident: use proxy scores, skip LLM
            decision = "proxy_only"
            table_scores = proxy_scores

            from src.cost_logger import cost_logger
            cost_logger.log_proxy_only("PEX", self.dataset_name)

            cascade_info = {
                'decision': decision,
                'confidence_metric': 'boundary_gap',
                'confidence_value': round(float(confidence_value), 6),
                'boundary_gap': confidence['boundary_gap'],
                'normalized_entropy': confidence['normalized_entropy'],
                'entropy_confidence': confidence['entropy_confidence'],
                'tau': tau,
                'proxy_scores': confidence['raw_scores'],
                'normalized_scores': confidence.get('normalized_scores', {}),
                'llm_invoked': False,
                'llm_time': 0.0,
                **cascade_info_extra,
            }

        return table_scores, cascade_info

    def _llm_table_scores(self, batch_prompt: str, tables_to_score: list):
        """LLM scores for the candidate tables, or None (recorded in stats) when the reply cannot be parsed."""
        try:
            # one score per table: the client's default of 1,200 tokens truncates replies for large schemas
            scores = self.llm_client.call_for_json_object(batch_prompt, max_tokens=max(1200, 40 * len(tables_to_score)))
        except LLMResponseParseError as e:
            logging.warning(f"Unusable LLM reply for table scoring: {e}")
            print(f"    [PEX] Unusable LLM reply ({e}); falling back to cheaper scores")
            self.stats['llm_fallback'] = str(e)
            return None
        scores = {k: v for k, v in scores.items() if k in tables_to_score}
        for table in tables_to_score:
            if table not in scores:
                scores[table] = 0.0
        return scores

    def _phase1_llm_table_scoring(self):
        """Phase 1: score every candidate table (statistics only, SLM, LLM, or the SLM-LLM cascade)"""
        phase1_start = time.time()

        tables_to_score = [name for name in self.dataset.tables.keys()
                           if name != self.dataset.base_table]

        logging.info(f"Phase 1: Batch scoring {len(tables_to_score)} tables with LLM")

        if not tables_to_score:
            logging.warning("No candidate tables to score")
            self.stats['phase1_timings'] = {
                'total': 0.0,
                'preprocessing': 0.0,
                'llm_call': 0.0
            }
            return

        from src.experiments.feature_descriptions import has_semantic_signal
        if not has_semantic_signal(self.dataset_name):
            # Both column names and descriptions are cryptic — semantic scoring
            # would introduce noise rather than signal. Use stats-only scoring.
            self.table_scores = {name: 0.0 for name in tables_to_score}
            self.stats['phase1_timings'] = {
                'total': 0.0, 'preprocessing': 0.0, 'llm_call': 0.0
            }
            print(f"    [PEX] Dataset '{self.dataset_name}' has no semantic signal "
                  f"— using stats-only path scoring (S_sem=0)")
            # In adaptive mode, write the cascade record, so run_hippasus.py never reads one left by an earlier run
            if self.use_adaptive:
                import json
                from pathlib import Path
                cascade_path = Path(f"src/experiments/results/cascade/"
                                    f"{self.dataset_name}_pex_decision.json")
                cascade_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cascade_path, 'w') as f:
                    json.dump({
                        'decision': 'stats_only_no_signal',
                        'reason': 'no_semantic_signal',
                        'boundary_gap': None,
                        'entropy_confidence': None,
                        'tau': self.adaptive_tau,
                        'llm_invoked': False,
                        'llm_time': 0.0,
                    }, f, indent=2)
            return

        if self.use_stats_only:
            # Stats-only mode: set S_sem = 0 for all tables
            # Path scoring will use only S_stat (connection quality)
            self.table_scores = {name: 0.0 for name in tables_to_score}
            self.stats['phase1_timings'] = {
                'total': 0.0,
                'preprocessing': 0.0,
                'llm_call': 0.0
            }
            return

        elif self.use_adaptive:
            # Adaptive mode: SLM proxy first, LLM oracle if uncertain
            self.table_scores, cascade_info = self._adaptive_table_scoring(
                tables_to_score, self.adaptive_tau
            )
            prep_time = 0.0
            llm_time = cascade_info['llm_time']

            import json
            from pathlib import Path
            cascade_path = Path(f"src/experiments/results/cascade/"
                                f"{self.dataset_name}_pex_decision.json")
            cascade_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cascade_path, 'w') as f:
                json.dump(cascade_info, f, indent=2)

        if not self.use_adaptive:
            if self.use_slm:
                feature_descs = None
                if self.use_feature_descriptions:
                    from ..experiments.feature_descriptions import get_dataset_features
                    feature_descs = get_dataset_features(self.dataset_name)
                join_keys = self.dataset.get_join_keys_info()['all_join_keys']
                self.table_scores = self.slm_scorer.score_tables(
                    base_table_name=self.dataset.base_table,
                    base_table_meta=self.dataset.tables[self.dataset.base_table],
                    target_column=self.target_column,
                    candidate_tables={name: self.dataset.tables[name] for name in tables_to_score},
                    feature_descs=feature_descs,
                    join_keys=join_keys
                )
                prep_time, llm_time = 0.0, 0.0
            else:
                prep_start = time.time()
                batch_prompt = self._create_batch_scoring_prompt(tables_to_score)
                prep_time = time.time() - prep_start

                llm_start = time.time()
                self.table_scores = self._llm_table_scores(batch_prompt, tables_to_score)
                if self.table_scores is None:  # unusable reply: statistics only, as without semantic signal
                    self.table_scores = {name: 0.0 for name in tables_to_score}
                llm_time = time.time() - llm_start

        if len(self.table_scores) != len(tables_to_score):
            missing_tables = set(tables_to_score) - set(self.table_scores.keys())
            raise ValueError(
                f"Expected {len(tables_to_score)} scores, got {len(self.table_scores)}. Missing: {missing_tables}")

        self.stats['phase1_timings'] = {
            'total': time.time() - phase1_start,
            'preprocessing': prep_time,
            'llm_call': llm_time
        }

        logging.info(f"Phase 1 complete: Scored {len(self.table_scores)} tables")
        logging.info(f"  Preprocessing: {prep_time:.2f}s, LLM call: {llm_time:.2f}s")

    def _phase2_guided_exploration(self):
        """Phase 2: BFS exploration with principled normalized scoring"""
        phase2_start = time.time()
        path_gen_time = 0.0
        path_score_time = 0.0

        logging.info(f"Phase 2: BFS exploration up to depth {self.max_depth}")

        explored_paths = set()
        exploration_queue = deque([[self.dataset.base_table]])

        while exploration_queue:
            current_path = exploration_queue.popleft()
            current_table = current_path[-1]

            if len(current_path) > 1:
                path_tuple = tuple(current_path)

                if path_tuple not in explored_paths:
                    explored_paths.add(path_tuple)
                    self.stats['paths_evaluated'] += 1

                    score_start = time.time()
                    path_score = self._calculate_principled_path_quality(current_path)
                    self._update_top_k_list(current_path[:], path_score)
                    path_score_time += time.time() - score_start

                    logging.debug(f"Added path: {' → '.join(current_path)} (score: {path_score:.3f})")

            if len(current_path) < self.max_depth:
                gen_start = time.time()
                connected_tables = self._get_connected_tables(current_table)

                for next_table in connected_tables:
                    if next_table not in current_path:
                        new_path = current_path + [next_table]
                        exploration_queue.append(new_path)

                path_gen_time += time.time() - gen_start

        self.stats['phase2_timings'] = {
            'total': time.time() - phase2_start,
            'path_generation': path_gen_time,
            'path_scoring': path_score_time
        }

        logging.info(
            f"Phase 2 complete: {self.stats['paths_evaluated']} paths evaluated, {len(self.top_k_paths)} top-k paths found")
        logging.info(f"  Path generation: {path_gen_time:.2f}s, Path scoring: {path_score_time:.2f}s")

    def _calculate_principled_path_quality(self, path: List[str]) -> float:
        """Calculate principled normalized quality score for a join path"""
        if len(path) <= 1:
            return 0.0

        semantic_utility = 0.0
        for table in path[1:]:
            semantic_utility += self.table_scores.get(table, 0.5)

        statistical_utility = 0.0
        for i in range(len(path) - 1):
            table_a, table_b = path[i], path[i + 1]
            statistical_utility += self._get_connection_quality(table_a, table_b)

        total_utility = semantic_utility + statistical_utility
        num_components = (len(path) - 1) * 2
        theoretical_maximum = num_components * 1.0

        if theoretical_maximum > 0:
            normalized_score = total_utility / theoretical_maximum
        else:
            normalized_score = 0.0

        return normalized_score

    def _create_batch_scoring_prompt(self, tables_to_score: List[str]) -> str:
        """Prompt that scores all candidate tables in one LLM call"""
        task_description = {
            "binary": "binary classification",
            "multiclass": "multiclass classification",
            "regression": "regression"
        }.get(self.task_type.lower(), self.task_type)

        prompt = f"""TASK: {task_description.capitalize()} to predict '{self.target_column}'
Dataset: {self.dataset_name}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE TABLE: {self.dataset.base_table}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{self._format_table_schema(self.dataset.base_table, is_base=True)}

TARGET: {self.target_column}
"""

        prompt += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        prompt += f"CANDIDATE TABLES ({len(tables_to_score)} tables)\n"
        prompt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        for table_name in tables_to_score:
            prompt += f"\nTABLE: {table_name}\n"
            prompt += self._format_table_schema(table_name, is_base=False)
            prompt += "\n"

        prompt += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        prompt += "FOREIGN KEY RELATIONSHIPS:\n"
        prompt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        prompt += self._format_connections()

        prompt += f"""

Rate each candidate table's relevance for predicting '{self.target_column}'.
Score range: [0.0 = irrelevant, 1.0 = highly relevant]

CRITICAL REQUIREMENTS:
• Score ALL {len(tables_to_score)} tables listed above
• Return ONLY a JSON object: {{"table_name": score}}
• No explanations, no code, no other text

JSON response for all {len(tables_to_score)} tables:"""

        return prompt

    def _format_table_schema(self, table_name: str, is_base: bool = False) -> str:
        """Table schema for the prompt, with feature descriptions when enabled"""
        meta = self.dataset.tables[table_name]
        schema_lines = []

        feature_descs = None
        if self.use_feature_descriptions:
            from ..experiments.feature_descriptions import get_dataset_features
            feature_descs = get_dataset_features(self.dataset_name)

        for col in meta['columns']:
            dtype = meta['dtypes'][col]
            distinct = meta['distinct_counts'][col]
            null_rate = meta['null_rates'][col]
            null_pct = int(null_rate * 100) if pd.notna(null_rate) else 0

            # idx.* columns are join keys: listed without a description
            if col.startswith('idx.'):
                schema_lines.append(f"  • {col}: {dtype} ({distinct} unique, {null_pct}% null)")
                continue

            desc = describe(feature_descs, col)

            if desc:
                schema_lines.append(f"  • {col}: {desc}")
                schema_lines.append(f"    Type: {dtype} | Distinct: {distinct} | Missing: {null_pct}%")
            else:
                schema_lines.append(f"  • {col}: {dtype} ({distinct} unique, {null_pct}% null)")

        return "\n".join(schema_lines)

    def _format_connections(self) -> str:
        """Format connections in readable structure"""
        if not self.dataset.connections:
            return "  No explicit connections defined"

        conn_lines = []
        for conn in self.dataset.connections:
            conn_lines.append(f"  {conn['table1']}.{conn['key1']} → {conn['table2']}.{conn['key2']}")

        return "\n".join(conn_lines)

    def _get_connected_tables(self, table_name: str) -> List[str]:
        """Get all tables directly connected to given table via foreign keys"""
        connected = []

        for conn in self.dataset.connections:
            if conn['table1'] == table_name and conn['table2'] in self.dataset.tables:
                connected.append(conn['table2'])
            elif conn['table2'] == table_name and conn['table1'] in self.dataset.tables:
                connected.append(conn['table1'])

        # Ties in the top-k list keep discovery order, so the visiting order decides them: random, seeded per run,
        # and independent of string hashing. Unset seed: sorted order.
        connected = sorted(set(connected))
        tie_seed = os.environ.get('HIPPASUS_TIE_SEED')
        if tie_seed is not None:
            random.Random(f"{tie_seed}:{table_name}").shuffle(connected)
        return connected

    def _get_connection_quality(self, table_a: str, table_b: str) -> float:
        """Calculate statistical quality of join between two tables"""
        conn_key = (table_a, table_b)

        if conn_key in self.connection_stats:
            return self.connection_stats[conn_key]

        fk_info = self._find_foreign_key(table_a, table_b)
        if not fk_info:
            self.connection_stats[conn_key] = 0.1
            return 0.1

        try:
            table_a_meta = self.dataset.tables[table_a]
            table_b_meta = self.dataset.tables[table_b]

            fk_null_rate = table_a_meta['null_rates'].get(fk_info['key_a'], 0.0)
            join_coverage = 1.0 - fk_null_rate

            pk_distinct = table_b_meta['distinct_counts'].get(fk_info['key_b'], 1)
            pk_total = table_b_meta['row_count']
            key_uniqueness = pk_distinct / max(pk_total, 1)

            size_ratio = min(table_a_meta['row_count'], table_b_meta['row_count']) / \
                         max(table_a_meta['row_count'], table_b_meta['row_count'], 1)

            overall_score = (join_coverage + key_uniqueness + size_ratio) / 3.0
            overall_score = max(0.0, min(1.0, overall_score))

            self.connection_stats[conn_key] = overall_score
            return overall_score

        except Exception as e:
            logging.warning(f"Error calculating connection quality {table_a}->{table_b}: {e}")
            self.connection_stats[conn_key] = 0.3
            return 0.3

    def _find_foreign_key(self, table_a: str, table_b: str):
        """Find foreign key relationship between two tables"""
        for conn in self.dataset.connections:
            if conn['table1'] == table_a and conn['table2'] == table_b:
                return {'key_a': conn['key1'], 'key_b': conn['key2']}
            elif conn['table2'] == table_a and conn['table1'] == table_b:
                return {'key_a': conn['key2'], 'key_b': conn['key1']}
        return None

    def _update_top_k_list(self, new_path: List[str], new_score: float):
        """Maintain sorted list of top-k scoring paths"""
        self.top_k_paths.append((new_path, new_score))
        self.top_k_paths.sort(key=lambda x: x[1], reverse=True)

        while len(self.top_k_paths) > self.top_k:
            self.top_k_paths.pop()
