"""
The Hippasus pipeline in seven steps: load the base table, explore join paths (Path Explorer), execute
the joins (Join Executor), consolidate them, split train/test, select features (Feature Selector), train and evaluate.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split

from src.path_explorer.path_explorer import PathExplorer
from src.join_executor.binary_join_executor import BinaryJoinExecutor
from src.join_executor.suffix_yannakakis_join_executor import SuffixYannakakisJoinExecutor
from src.join_executor.table_consolidator import TableConsolidator
from src.ml.model_trainer import HippasusModelTrainer

class Hippasus:

    def __init__(
            self,
            dataset_folder: str,
            base_table_name: str,
            target_column: str,
            task: str,
            join_algo: str = "binary",
            time_limit: int = None,
            dataset_name: str = None
    ):

        self.dataset_folder = dataset_folder
        self.base_table_name = Path(base_table_name).stem
        self.target_column = target_column
        self.task = task
        self.time_limit = time_limit
        self.dataset_name = dataset_name

        valid_algos = [
            'binary',
            'suffix-yannakakis',
        ]
        if join_algo not in valid_algos:
            raise ValueError(f"join_algo must be one of {valid_algos}, got: {join_algo}")
        self.join_algo = join_algo

        self.logger = logging.getLogger(__name__)

        self.full_base_df = None  # Full base table (before split)
        self.discovered_paths = None
        self.join_keys_info = None
        self.augmented_full_df = None  # After joins, before split
        self.consolidated_full_df = None  # After consolidation, before split
        self.consolidation_schema = None

        self.max_depth_requested = None
        self.max_depth_actual = None
        self.min_depth_actual = None
        self.avg_depth_actual = None
        self.path_depths = []
        self.selected_join_paths = None

        self.per_path_stats = []

        # After split
        self.train_df = None
        self.test_df = None

        # After feature selection
        self.selected_train_df = None
        self.selected_test_df = None
        self.feature_provenance = None

        self.ml_results = {}

        self.logger.info(f" Hippasus initialized:")
        self.logger.info(f"   Dataset: {dataset_folder}")
        self.logger.info(f"   Base table: {base_table_name}")
        self.logger.info(f"   Target: {target_column}")
        self.logger.info(f"   Task: {task}")
        self.logger.info(f"   Join algorithm: {join_algo}")

    def step1_load_base_table(self) -> bool:

        print(f"\n{'=' * 70}")
        print(f" STEP 1: LOAD FULL BASE TABLE")
        print(f"{'=' * 70}")

        base_csv_path = Path(self.dataset_folder) / f"{self.base_table_name}.csv"

        if not base_csv_path.exists():
            raise FileNotFoundError(f"Base table not found: {base_csv_path}")

        self.full_base_df = pd.read_csv(base_csv_path, dtype=str, low_memory=False)

        from src.path_explorer.path_explorer import NullNormalizer
        self.full_base_df = NullNormalizer.normalize_nulls(self.full_base_df)

        for col in self.full_base_df.columns:
            self.full_base_df[col] = pd.to_numeric(self.full_base_df[col], errors='ignore')

        if self.target_column not in self.full_base_df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in base table. "
                f"Available columns: {list(self.full_base_df.columns)}"
            )

        print(f"    Loaded base table: {self.full_base_df.shape}")
        print(f"    Target column: {self.target_column}")
        print(f"    Target distribution:")
        print(f"      {self.full_base_df[self.target_column].value_counts()}")
        print(f"    Features: {len(self.full_base_df.columns) - 1}")

        return True

    def step2_path_explorer(self, max_depth: int = 3, top_k: int = 10,
                            use_slm: bool = False,
                            use_stats_only: bool = False,
                            use_adaptive: bool = False,
                            adaptive_tau = None) -> bool:  # float or None

        print(f"\n{'=' * 70}")
        print(f" STEP 2: PATH EXPLORATION (LLM-Guided)")
        print(f"{'=' * 70}")

        if self.full_base_df is None:
            raise RuntimeError("Must run step1_load_base_table first")

        import time as _t
        _t0 = _t.time()
        print(f"[step2] About to instantiate PathExplorer")

        explorer = PathExplorer(
            base_df=self.full_base_df,
            dataset_folder=self.dataset_folder,
            target_column=self.target_column,
            task_type=self.task,
            max_depth=max_depth,
            top_k=top_k,
            base_table_name=self.base_table_name,
            use_slm=use_slm,
            use_stats_only=use_stats_only,
            use_adaptive=use_adaptive,
            adaptive_tau=adaptive_tau,
        )

        print(f"[step2] PathExplorer instantiated in {_t.time() - _t0:.3f}s")

        print(f"    Max depth: {max_depth}")
        print(f"    Top-K paths: {top_k}")
        print(f"    Scoring mode: "
              f"{'Adaptive (proxy+oracle, τ=' + str(adaptive_tau) + ')' if use_adaptive else 'Stats-only (no semantic)' if use_stats_only else 'SLM (embedding)' if use_slm else 'LLM (generative)'}")

        self.discovered_paths, self.join_keys_info, pe_timings = explorer.explore()

        if not hasattr(self, 'detailed_timings'):
            self.detailed_timings = {}
        self.detailed_timings['path_explorer'] = pe_timings

        if not self.discovered_paths:
            raise RuntimeError("Path explorer found no valid paths")

        self.max_depth_requested = max_depth
        self.selected_join_paths = self.discovered_paths[:top_k]
        self.path_depths = [len(path) for path in self.selected_join_paths]

        if self.path_depths:
            self.max_depth_actual = max(self.path_depths)
            self.min_depth_actual = min(self.path_depths)
            self.avg_depth_actual = round(sum(self.path_depths) / len(self.path_depths), 2)
        else:
            self.max_depth_actual = 0
            self.min_depth_actual = 0
            self.avg_depth_actual = 0.0

        print(f"\n     Timing breakdown:")
        print(f"      Total: {pe_timings['total_time']:.2f}s")
        print(f"      Phase 1 (LLM scoring): {pe_timings['phase1_timings']['total']:.2f}s "
              f"(prep: {pe_timings['phase1_timings']['preprocessing']:.2f}s, "
              f"llm: {pe_timings['phase1_timings']['llm_call']:.2f}s)")
        print(f"      Phase 2 (BFS): {pe_timings['phase2_timings']['total']:.2f}s "
              f"(gen: {pe_timings['phase2_timings']['path_generation']:.2f}s, "
              f"score: {pe_timings['phase2_timings']['path_scoring']:.2f}s)")

        print(f"\n    Discovered {len(self.discovered_paths)} paths:")

        for i, path in enumerate(self.discovered_paths[:top_k], 1):
            print(f"      {i}. {' → '.join(path)}")

        print(f"\n    Path depth statistics:")
        print(f"      Requested max_depth: {self.max_depth_requested}")
        print(f"      Actual max_depth: {self.max_depth_actual}")
        print(f"      Depth range: [{self.min_depth_actual}, {self.max_depth_actual}]")
        print(f"      Average depth: {self.avg_depth_actual:.1f}")

        return True

    def step3_join_executor(self, max_paths: int = 10) -> bool:

        print(f"\n{'=' * 70}")
        print(f" STEP 3: JOIN EXECUTION (Single, Full Data)")
        print(f"{'=' * 70}")

        if self.full_base_df is None:
            raise RuntimeError("Must run step1_load_base_table first")

        if not self.discovered_paths:
            raise RuntimeError("Must run step2_path_explorer first")

        if self.join_algo == 'binary':
            executor = BinaryJoinExecutor()
        elif self.join_algo == 'suffix-yannakakis':
            executor = SuffixYannakakisJoinExecutor()
        else:
            raise ValueError(f"Unknown join algorithm: {self.join_algo}")

        executor.set_dataset_folder(self.dataset_folder)
        executor.set_join_keys_info(self.join_keys_info)
        executor.add_row_id = False
        executor.log_quality = True

        paths_to_execute = self.discovered_paths[:max_paths]

        print(f"    Executor: {self.join_algo}")
        print(f"    Paths to execute: {len(paths_to_execute)}")
        print(f"    Base table shape: {self.full_base_df.shape}")

        executed_tables = []

        self.per_path_stats = []

        import time as time_module

        for i, path in enumerate(paths_to_execute, 1):
            path_str = ' → '.join(path)
            print(f"\n    Executing path {i}/{len(paths_to_execute)}: {path_str}")

            path_start_time = time_module.time()

            result_df = executor.execute_join_path(path, self.full_base_df, self.target_column)

            path_execution_time = time_module.time() - path_start_time
            features_added = result_df.shape[1] - self.full_base_df.shape[1]

            print(f"       Result shape: {result_df.shape}")
            print(f"       Features added: {features_added}")
            print(f"        Path execution time: {path_execution_time:.3f}s")

            self.per_path_stats.append({
                'path_id': i - 1,
                'path_length': len(path),
                'path_str': path_str,
                'execution_time': path_execution_time,
                'features_added': features_added,
                'result_shape': result_df.shape,
                'path_tables': [Path(p).stem for p in path]
            })

            executed_tables.append(result_df)

        print(f"\n    All paths executed successfully")
        print(f"    Total executed tables: {len(executed_tables)}")

        self._executed_tables = executed_tables
        self._executor = executor

        return True

    def step4_consolidate(self) -> bool:

        print(f"\n{'=' * 70}")
        print(f" STEP 4: CONSOLIDATION (Single Pass)")
        print(f"{'=' * 70}")

        if not hasattr(self, '_executed_tables'):
            raise RuntimeError("Must run step3_join_executor first")

        print(f"\n    DIAGNOSTIC: BEFORE CONSOLIDATION")
        joined_table = self._executed_tables[0]
        print(f"      Input shape: {joined_table.shape}")
        print(f"      Input columns ({len(joined_table.columns)}):")

        from collections import defaultdict
        cols_by_table = defaultdict(list)
        for col in joined_table.columns:
            if '.' in col:
                table_name = col.split('.')[0]
                cols_by_table[table_name].append(col)
            else:
                cols_by_table['NO_PREFIX'].append(col)

        for table_name in sorted(cols_by_table.keys()):
            print(f"         {table_name}: {len(cols_by_table[table_name])} columns")
            for col in sorted(cols_by_table[table_name]):
                print(f"            - {col}")

        consolidator = TableConsolidator()

        consolidator_join_keys = self._executor.get_join_keys_info()
        consolidator.set_join_keys_info(consolidator_join_keys)

        print(f"\n    Consolidating {len(self._executed_tables)} tables...")
        print(f"    Base table: {self.base_table_name}")

        self.consolidated_full_df, quality_report = consolidator.consolidate_tables(
            self._executed_tables,
            self.full_base_df,
            self.target_column
        )

        print(f"\n    DIAGNOSTIC: AFTER CONSOLIDATION")
        print(f"      Output shape: {self.consolidated_full_df.shape}")
        print(f"      Output columns ({len(self.consolidated_full_df.columns)}):")

        cols_by_table_after = defaultdict(list)
        for col in self.consolidated_full_df.columns:
            if '.' in col:
                table_name = col.split('.')[0]
                cols_by_table_after[table_name].append(col)
            else:
                cols_by_table_after['NO_PREFIX'].append(col)

        for table_name in sorted(cols_by_table_after.keys()):
            print(f"         {table_name}: {len(cols_by_table_after[table_name])} columns")
            for col in sorted(cols_by_table_after[table_name]):
                print(f"            - {col}")

        print(f"\n    DIAGNOSTIC: COLUMNS DROPPED")
        before_cols = set(joined_table.columns)
        after_cols = set(self.consolidated_full_df.columns)
        dropped_cols = before_cols - after_cols

        if dropped_cols:
            print(f"        {len(dropped_cols)} columns were dropped:")

            all_join_keys = consolidator_join_keys.get('all_join_keys', set())

            for col in sorted(dropped_cols):
                # Extract column name without table prefix (handle double-dot columns)
                # e.g., "base.idx.1" → table="base", col_name="idx.1"
                if '.' in col:
                    parts = col.split('.', 1)
                    table_prefix = parts[0]
                    col_name = parts[1] if len(parts) > 1 else col
                else:
                    table_prefix = 'NO_PREFIX'
                    col_name = col

                is_join_key = col_name in all_join_keys
                is_from_base = table_prefix == self.base_table_name

                if is_join_key and not is_from_base:
                    reason = " Foreign/Primary key from joined table (correctly dropped)"
                elif is_join_key and is_from_base:
                    reason = " Primary key from base table - SHOULD NOT BE DROPPED!"
                else:
                    reason = "  Duplicate or cleaning step (check consolidator logs)"

                print(f"         {col}")
                print(f"            └─ {reason}")
        else:
            print(f"       No columns were dropped")

        self.feature_provenance = quality_report.get('provenance', {})

        if not self.feature_provenance:
            raise RuntimeError("Consolidator did not return provenance information!")

        print(f"\n    Consolidation complete")
        print(f"    Consolidated shape: {self.consolidated_full_df.shape}")
        print(f"    Original features: {self.full_base_df.shape[1]}")
        print(f"    Features added: {self.consolidated_full_df.shape[1] - self.full_base_df.shape[1]}")

        if self.feature_provenance:
            base_count = self.feature_provenance['counts']['base']
            aug_count = self.feature_provenance['counts']['augmented']
            print(f"\n    FEATURE PROVENANCE:")
            print(f"      Base features: {base_count}")
            print(f"      Augmented features: {aug_count}")
            print(f"      Total features: {base_count + aug_count}")

        return True

    def step5_split(self, test_size: float = 0.2, random_state: int = 42) -> bool:

        print(f"\n{'=' * 70}")
        print(f"  STEP 5: SPLIT (Before Feature Selection - No Leakage)")
        print(f"{'=' * 70}")

        if self.consolidated_full_df is None:
            raise RuntimeError("Must run step4_consolidate first")

        target_col_prefixed = self.feature_provenance.get('target_column')

        if target_col_prefixed not in self.consolidated_full_df.columns:
            raise RuntimeError(f"Target column '{target_col_prefixed}' not found in consolidated table")

        # Stratified split
        self.train_df, self.test_df = train_test_split(
            self.consolidated_full_df,
            test_size=test_size,
            stratify=self.consolidated_full_df[target_col_prefixed] if self.task != "regression" else None,
            random_state=random_state
        )

        print(f"    Split ratio: {int((1 - test_size) * 100)}/{int(test_size * 100)}")
        print(f"    Train shape: {self.train_df.shape}")
        print(f"    Test shape: {self.test_df.shape}")
        print(f"    Schema consistency: {set(self.train_df.columns) == set(self.test_df.columns)}")

        if self.task != "regression":
            print(f"\n    Target distribution preservation:")
            train_dist = self.train_df[target_col_prefixed].value_counts(normalize=True).sort_index()
            test_dist = self.test_df[target_col_prefixed].value_counts(normalize=True).sort_index()
            print(f"      Train: {train_dist.to_dict()}")
            print(f"      Test:  {test_dist.to_dict()}")

        # Drop columns that identify rows instead of describing them: base-table foreign keys and ID-like columns
        print(f"\n    LEAKAGE DETECTION: Checking for potential leakage columns...")

        leakage_cols = []
        leakage_reasons = {}

        # 1) Base-table foreign keys, from connections.csv
        if hasattr(self, '_executor') and hasattr(self._executor, 'connections_df'):
            from src.join_executor.table_consolidator import JoinKeyClassifier

            connections_df = self._executor.connections_df
            if connections_df is not None and len(connections_df) > 0:
                classifier = JoinKeyClassifier(connections_df, self.base_table_name)

                print(f"       Using schema-based FK detection from connections.csv")

                for col in self.train_df.columns:
                    if col == target_col_prefixed:
                        continue

                    decision, reason = classifier.classify_column(col)

                    if decision == 'DROP' and 'Base table foreign key' in reason:
                        leakage_cols.append(col)
                        leakage_reasons[col] = reason

        cardinality_threshold = 0.95
        for col in self.train_df.columns:
            if col == target_col_prefixed or col in leakage_cols:
                continue

            unique_ratio = self.train_df[col].nunique() / len(self.train_df)

            # 2) High-cardinality columns with an ID-like name
            if unique_ratio <= cardinality_threshold:
                continue

            parts = col.split('.')
            if len(parts) >= 2 and parts[-2] == 'idx':
                # Special case: 'base.idx.5' → extract 'idx.5'
                col_name = '.'.join(parts[-2:])
            elif len(parts) >= 1:
                # Normal case: 'table_0_0.V4' → extract 'V4'
                col_name = parts[-1]
            else:
                col_name = col

            is_key_pattern = (
                    col_name.startswith('Key_') or  # Key_0_0, Key_2_6, etc.
                    col_name.startswith('key_') or  # key_0_0 (lowercase variant)
                    col_name.startswith('idx.') or  # idx.1, idx.5, etc.
                    col_name.startswith('id.') or  # id.123
                    col_name == 'id' or  # just 'id'
                    col_name == 'ID' or  # just 'ID'
                    col_name.endswith('_id') or  # user_id, patient_id, etc.
                    col_name.endswith('_ID')  # USER_ID, PATIENT_ID, etc.
            )

            if is_key_pattern:
                leakage_cols.append(col)
                leakage_reasons[col] = f"{unique_ratio:.1%} unique + KEY pattern '{col_name}' (likely ID column)"

        # 3) Integer columns with near-unique, nearly sequential values
        for col in self.train_df.columns:
            if col == target_col_prefixed or col in leakage_cols:
                continue
            if self.train_df[col].dtype in ['int64', 'int32']:
                unique_ratio = self.train_df[col].nunique() / len(self.train_df)
                if unique_ratio > 0.90:
                    sorted_vals = sorted(self.train_df[col].dropna().unique())
                    if len(sorted_vals) > 100:
                        gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
                        avg_gap = sum(gaps) / len(gaps) if gaps else 0
                        if avg_gap < 10:  # Small gaps = sequential
                            leakage_cols.append(col)
                            leakage_reasons[col] = f"{unique_ratio:.1%} unique, sequential integers (likely ID)"

        if leakage_cols:
            print(f"\n     DROPPING {len(leakage_cols)} potential leakage columns:")
            for col in sorted(leakage_cols):
                reason = leakage_reasons.get(col, "Unknown")
                print(f"       {col}")
                print(f"         └─ {reason}")

            self.train_df = self.train_df.drop(columns=leakage_cols, errors='ignore')
            self.test_df = self.test_df.drop(columns=leakage_cols, errors='ignore')

            base_before = len(self.feature_provenance['base_features'])
            aug_before = len(self.feature_provenance['augmented_features'])

            self.feature_provenance['base_features'] = [
                f for f in self.feature_provenance['base_features']
                if f not in leakage_cols
            ]

            # Some augmented features may be dropped as leakage (high cardinality, etc.)
            self.feature_provenance['augmented_features'] = [
                f for f in self.feature_provenance['augmented_features']
                if f not in leakage_cols
            ]

            base_after = len(self.feature_provenance['base_features'])
            aug_after = len(self.feature_provenance['augmented_features'])

            if base_before != base_after or aug_before != aug_after:
                print(f"\n    Updated feature provenance:")
                if base_before != base_after:
                    print(f"      Base features: {base_before} → {base_after} (-{base_before - base_after})")
                if aug_before != aug_after:
                    print(f"      Augmented features: {aug_before} → {aug_after} (-{aug_before - aug_after})")

            print(f"\n    After leakage removal:")
            print(f"      Train shape: {self.train_df.shape}")

            train_dups = [col for col in self.train_df.columns if list(self.train_df.columns).count(col) > 1]
            if train_dups:
                print(f"        WARNING: Train has duplicate columns: {set(train_dups)}")
            test_dups = [col for col in self.test_df.columns if list(self.test_df.columns).count(col) > 1]
            if test_dups:
                print(f"        WARNING: Test has duplicate columns: {set(test_dups)}")
        else:
            print(f"       No leakage columns detected")

            train_dups = [col for col in self.train_df.columns if list(self.train_df.columns).count(col) > 1]
            if train_dups:
                print(f"        WARNING: Train has duplicate columns: {set(train_dups)}")
            test_dups = [col for col in self.test_df.columns if list(self.test_df.columns).count(col) > 1]
            if test_dups:
                print(f"        WARNING: Test has duplicate columns: {set(test_dups)}")

        return True

    def step6_feature_selector(self, method: str = "hippasus-adaptive-v2", max_features: int = 10,
                                   pre_filter_K: int = None) -> bool:
        print(f"\n{'=' * 70}")
        print(f" STEP 6: FEATURE SELECTION (Train Only - No Leakage)")
        print(f"{'=' * 70}")

        if self.train_df is None or self.test_df is None:
            raise RuntimeError("Must run step5_split first")

        print(f"    Method: {method}")
        print(f"    Max augmented features: {max_features}")
        print(f"    Computing statistics on TRAIN ONLY (no test data used)")

        base_features = self.feature_provenance['base_features']
        augmented_features = self.feature_provenance['augmented_features']
        target_column = self.feature_provenance['target_column']

        print(f"\n    Feature breakdown:")
        print(f"      Base features: {len(base_features)} (always kept)")
        print(f"      Augmented features: {len(augmented_features)} (select from these)")
        print(f"      Target: {target_column}")

        # the feature selector's FDG switch, as the Path Explorer's (HIPPASUS_PE_FEATURE_DESC); on by default
        fs_descriptions = os.environ.get('HIPPASUS_FS_FEATURE_DESC', 'true').lower() == 'true'

        if method == "hippasus-slm":

            from src.feature_selector.integration_selector import IntegrationSelector

            selector = IntegrationSelector(
                max_features=max_features,
                dataset_name=self.dataset_name,
                use_feature_descriptions=fs_descriptions,
                pre_filter_K=pre_filter_K,
                use_slm=True
            )

        elif method == "hippasus-stats":

            from src.feature_selector.stats_only_selector import StatsOnlySelector

            selector = StatsOnlySelector(max_features=max_features)

        elif method in ("hippasus-llm-v2", "hippasus-adaptive-v2"):

            from src.feature_selector.integration_selector import IntegrationSelector

            selector = IntegrationSelector(
                max_features=max_features,
                dataset_name=self.dataset_name,
                use_feature_descriptions=fs_descriptions,
                pre_filter_K=pre_filter_K,
                use_mi_only=True,
            )

        else:

            raise ValueError(
                f"Unsupported feature selection method: {method}. "
                f"Valid: ['hippasus-stats', 'hippasus-slm', 'hippasus-llm-v2', 'hippasus-adaptive-v2']"
            )

        # Select features using TRAIN DATA ONLY

        if method in ("hippasus-slm", "hippasus-llm-v2", "hippasus-adaptive-v2"):
            # These methods return (selected_features, timings)
            result = selector.select_features(
                train_df=self.train_df,
                target_col=target_column,
                task_type=self.task,
                base_features=base_features,
                augmented_features=augmented_features
            )
            selected_augmented_features, extra_data = result

            if not hasattr(self, 'detailed_timings'):
                self.detailed_timings = {}
            self.detailed_timings['feature_selector'] = extra_data

        else:
            # hippasus-stats returns the list only
            selected_augmented_features = selector.select_features(
                train_df=self.train_df,
                target_col=target_column,
                task_type=self.task,
                base_features=base_features,
                augmented_features=augmented_features
            )

            if not hasattr(self, 'detailed_timings'):
                self.detailed_timings = {}
            self.detailed_timings['feature_selector'] = {
                'total': 0.0,
                'stats_computation': 0.0,
                'llm_ranking': {'total': 0.0, 'preprocessing': 0.0, 'llm_call': 0.0}
            }

        selected_features = base_features + selected_augmented_features + [target_column]

        if len(selected_features) != len(set(selected_features)):
            duplicates = [col for col in selected_features if selected_features.count(col) > 1]
            duplicates_unique = list(set(duplicates))

            print(f"\n     WARNING: Duplicate columns detected!")
            for dup in duplicates_unique:
                print(f"       '{dup}' appears {selected_features.count(dup)} times")

            print(f"   ℹ  Proceeding anyway (keeping all)")

        print(f"\n    Feature selection complete")
        print(f"    Selected augmented features: {len(selected_augmented_features)}/{len(augmented_features)}")
        print(f"    Total features (base + augmented + target): {len(selected_features)}")

        # Apply selection to BOTH train and test
        self.selected_train_df = self.train_df[selected_features].copy()
        self.selected_test_df = self.test_df[selected_features].copy()

        self.selected_features_metadata = {
            'method': method,
            'base_features': base_features,
            'selected_augmented': selected_augmented_features,
            'target': target_column,
            'total_features': len(selected_features)
        }

        print(f"\n    FEATURE BREAKDOWN - {method.upper()}:")
        print(f"      Base features: {len(base_features)}")
        for feat in base_features:
            print(f"         - {feat}")
        print(f"      Augmented features selected: {len(selected_augmented_features)}")
        for feat in selected_augmented_features:
            print(f"         - {feat}")

        print(f"\n{'=' * 70}")
        print(f" FINAL AUGMENTED TABLE SCHEMA (DETAILED)")
        print(f"{'=' * 70}")
        print(f"Shape: {self.selected_train_df.shape} (rows × columns)")
        print(f"Column Information:")
        print(f"{'-' * 70}")

        for idx, col in enumerate(self.selected_train_df.columns, 1):
            dtype = str(self.selected_train_df[col].dtype)
            null_count = self.selected_train_df[col].isnull().sum()
            null_pct = (null_count / len(self.selected_train_df)) * 100
            print(f"   {idx:2d}. {col:30s} | Type: {dtype:12s} | Nulls: {null_count:5d} ({null_pct:5.2f}%)")

        print(f"{'=' * 70}")
        print(f" FIRST 5 ROWS OF AUGMENTED TABLE")
        print(f"{'=' * 70}")
        print(self.selected_train_df.head(5).to_string(index=True))
        print(f"{'=' * 70}\n")

        return True

    def step7_ml_trainer(
            self,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            evaluation_name: str = "default"
    ) -> Dict[str, Dict[str, float]]:
        print(f"\n{'=' * 70}")
        print(f" STEP 7: ML EVALUATION")
        print(f"{'=' * 70}")

        target_col = self.feature_provenance['target_column']

        print(f"    Train shape: {train_data.shape}")
        print(f"    Test shape: {test_data.shape}")
        print(f"    Target: {target_col}")
        print(f"     Time limit: {self.time_limit}s")

        trainer = HippasusModelTrainer(
            target_column=target_col,
            task_type=self.task,
            algorithms=["RF", "GBM", "XT"],
            time_limit=self.time_limit,
            random_seed=42
        )

        import time
        start_time = time.time()

        detailed_scores = trainer.evaluate_dataset(train_data, test_data)

        elapsed_time = time.time() - start_time
        print(f"\n     ACTUAL ML TRAINING TIME: {elapsed_time:.2f}s (limit was {self.time_limit}s)")

        results_key = f"{evaluation_name}_detailed_scores"
        self.ml_results[results_key] = detailed_scores

        print(f"\n    ML evaluation complete")

        return detailed_scores
