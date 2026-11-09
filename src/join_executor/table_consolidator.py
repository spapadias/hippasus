"""
Consolidates the executed join paths into one table: keeps the base and augmented features, drops join keys,
and records the provenance of each feature.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set, Union
import logging


class JoinKeyClassifier:

    def __init__(self, connections_df: pd.DataFrame, base_table_name: str):
        self.base_table_name = base_table_name
        self.primary_keys = {}  # table_name -> set of PK column names
        self.foreign_keys = {}  # table_name -> set of FK column names
        self.logger = logging.getLogger(__name__)

        for _, row in connections_df.iterrows():
            # Format: connections.csv has 4 columns in order:
            # [FK table, FK column, PK table, PK column]
            # OR [left_table, pk_column, right_table, fk_column]

            if 'left_table' in connections_df.columns:
                left_table = row['left_table']
                pk_col = row['pk_column']
                right_table = row['right_table']
                fk_col = row['fk_column']
            elif 'fk_table' in connections_df.columns:
                # Format: fk_table,fk_column,pk_table,pk_column
                # Meaning: fk_table.fk_column → pk_table.pk_column (FK points to PK)
                fk_table_raw = row['fk_table']
                fk_col = row['fk_column']
                pk_table_raw = row['pk_table']
                pk_col = row['pk_column']

                fk_table = Path(str(fk_table_raw)).stem
                pk_table = Path(str(pk_table_raw)).stem

                # The table with PRIMARY KEY is "left" (source of truth)
                # The table with FOREIGN KEY is "right" (refers to left)
                left_table = pk_table  # PK table
                right_table = fk_table  # FK table
            else:
                fk_table = Path(str(row.iloc[0])).stem
                fk_col = str(row.iloc[1])
                pk_table = Path(str(row.iloc[2])).stem
                pk_col = str(row.iloc[3])

                left_table = pk_table  # PK table
                right_table = fk_table  # FK table

            if left_table not in self.primary_keys:
                self.primary_keys[left_table] = set()
            self.primary_keys[left_table].add(pk_col)

            if right_table not in self.foreign_keys:
                self.foreign_keys[right_table] = set()
            self.foreign_keys[right_table].add(fk_col)

        self.logger.info(f" JoinKeyClassifier initialized:")
        self.logger.info(f"   Base table: {base_table_name}")
        self.logger.info(f"   Primary keys tracked: {sum(len(v) for v in self.primary_keys.values())}")
        self.logger.info(f"   Foreign keys tracked: {sum(len(v) for v in self.foreign_keys.values())}")

    def classify_column(self, full_column_name: str) -> Tuple[str, str]:
        if '.' not in full_column_name:
            # No table prefix - shouldn't happen in consolidated table, but keep it
            return 'KEEP', 'No table prefix (unexpected)'

        table_name, col_name = full_column_name.split('.', 1)

        # RULE 1: DROP base table foreign keys (they're join infrastructure, not features!)
        if table_name == self.base_table_name:
            if col_name in self.foreign_keys.get(table_name, set()):
                return 'DROP', f'Base table foreign key (join infrastructure, not predictive)'
            # KEEP base table primary keys (they CAN be predictive, but often aren't)
            if col_name in self.primary_keys.get(table_name, set()):
                return 'KEEP', f'Base table primary key (may be predictive)'
            return 'KEEP', 'Base table feature column'

        # RULE 2: DROP foreign keys from ALL joined tables (they're duplicates)
        if col_name in self.foreign_keys.get(table_name, set()):
            return 'DROP', f'Foreign key from joined table (duplicate)'

        # RULE 3: DROP primary keys from joined tables (intermediate, not predictive)
        if col_name in self.primary_keys.get(table_name, set()):
            return 'DROP', f'Primary key from joined table (join artifact)'

        # RULE 4: KEEP everything else (actual features from joined tables)
        return 'KEEP', 'Feature column from joined table'

class TableConsolidator:

    def __init__(self):
        """Initialize consolidator (no state tracking needed)."""
        self.logger = logging.getLogger(__name__)
        self.join_keys_info = None  # Set via set_join_keys_info()
        self.base_table_name = None
        self.connections_df = None  # For JoinKeyClassifier

    def set_join_keys_info(self, join_keys_info: Dict[str, Union[Set[str], List[Dict]]]):
        if join_keys_info is None:
            raise ValueError("join_keys_info cannot be None")

        self.join_keys_info = join_keys_info
        self.base_table_name = join_keys_info.get('base_table_name')
        self.connections_df = join_keys_info.get('connections_df')

        if self.base_table_name is None:
            raise ValueError("base_table_name not provided in join_keys_info")

        if self.connections_df is None:
            raise ValueError("connections_df not provided in join_keys_info - required for role-based classification")

        self.logger.info(f" Consolidator initialized:")
        self.logger.info(f"   Base table: {self.base_table_name}")
        self.logger.info(f"   Join keys: {len(join_keys_info.get('all_join_keys', set()))}")

    def _normalize_base_table_columns(self, base_df: pd.DataFrame) -> pd.DataFrame:
        if self.base_table_name is None:
            raise ValueError("base_table_name not set")

        return base_df.rename(
            columns={col: f"{self.base_table_name}.{col}" for col in base_df.columns}
        )

    def _clean_features_for_ml(self,
                               consolidated_df: pd.DataFrame,
                               target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self.logger.info("=" * 70)
        self.logger.info(" MINIMAL CLEANING")
        self.logger.info("=" * 70)

        df = consolidated_df.copy()
        audit = {
            'row_id_columns': [],
            'duplicate_names': []
        }

        self.logger.info(f"   Input columns: {df.shape[1]}")
        self.logger.info(f"   Philosophy: Keep everything, trust ML algorithm")

        row_id_cols = [
            c for c in df.columns
            if c.endswith(".__row_id") or c == "__row_id"
        ]

        if row_id_cols:
            self.logger.info(f"\n    Removing {len(row_id_cols)} diagnostic __row_id columns:")
            for col in row_id_cols:
                self.logger.info(f"       {col}")
            df = df.drop(columns=row_id_cols, errors='ignore')
            audit['row_id_columns'] = row_id_cols
        else:
            self.logger.info(f"\n    No __row_id columns found")

        from collections import defaultdict

        column_counts = defaultdict(int)
        for col in df.columns:
            column_counts[col] += 1

        true_duplicates = {col: count for col, count in column_counts.items() if count > 1}

        if true_duplicates:
            self.logger.info(f"\n     Found {len(true_duplicates)} duplicate column names:")
            for col, count in true_duplicates.items():
                self.logger.info(f"       '{col}' appears {count} times")

            df = df.loc[:, ~df.columns.duplicated()]
            self.logger.info(f"    Removed duplicates (pandas limitation)")
            audit['duplicate_names'] = list(true_duplicates.keys())
        else:
            self.logger.info(f"\n    No duplicate column names found")

        self.logger.info(f"\n    Minimal cleaning complete:")
        self.logger.info(f"      Input columns:  {consolidated_df.shape[1]}")
        self.logger.info(f"      Output columns: {df.shape[1]}")
        self.logger.info(f"      Removed: {consolidated_df.shape[1] - df.shape[1]} (only __row_id and duplicate names)")
        self.logger.info("=" * 70)

        return df, audit

    def consolidate_tables(self,
                           executed_tables: List[pd.DataFrame],
                           original_base_df: pd.DataFrame,
                           target_column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.join_keys_info is None:
            raise ValueError("join_keys_info not set - call set_join_keys_info() first")

        self.logger.info("=" * 70)
        self.logger.info(" CONSOLIDATION: Single-call processing (full dataset)")
        self.logger.info("=" * 70)

        if original_base_df.empty:
            raise ValueError("original_base_df cannot be empty")
        if target_column not in original_base_df.columns:
            raise ValueError(f"target_column '{target_column}' not found in original_base_df")

        if not executed_tables:
            self.logger.warning("  No executed tables to consolidate")
            return self._normalize_base_table_columns(original_base_df.copy()), {}

        consolidated_df = executed_tables[0].copy()
        base_row_count = len(consolidated_df)

        self.logger.info(f" Using first executed table as base")
        self.logger.info(f"   Shape: {consolidated_df.shape}")
        self.logger.info(f"   Columns (first 10): {list(consolidated_df.columns)[:10]}")

        # Filter compatible tables (must have same row count)
        compatible_tables = [(0, executed_tables[0])]

        for i in range(1, len(executed_tables)):
            table = executed_tables[i]
            if not table.empty and len(table.columns) > 0 and len(table) == base_row_count:
                compatible_tables.append((i, table))
            else:
                self.logger.warning(
                    f"  Skipping table {i}: incompatible (rows={len(table)}, cols={len(table.columns)})")

        if len(compatible_tables) == 1:
            self.logger.info("ℹ  Only one executed table - no additional features to collect")
            consolidated_df, cleaning_audit = self._clean_features_for_ml(
                consolidated_df,
                target_column
            )

            provenance = self._build_feature_provenance(
                consolidated_df,
                original_base_df,
                target_column
            )

            quality_report = {
                'num_tables_consolidated': 1,
                'num_features_selected': 0,
                'provenance': provenance,
                'cleaning_audit': cleaning_audit
            }

            return consolidated_df, quality_report

        self.logger.info(f" Found {len(compatible_tables)} compatible executed tables")

        valid_tables = [table for _, table in compatible_tables]

        base_columns = set(original_base_df.columns)

        feature_candidates = self._collect_feature_candidates(valid_tables, base_columns)

        if not feature_candidates:
            self.logger.info("ℹ  No new features found in executed tables")
            consolidated_df, cleaning_audit = self._clean_features_for_ml(
                consolidated_df,
                target_column
            )

            provenance = self._build_feature_provenance(
                consolidated_df,
                original_base_df,
                target_column
            )

            quality_report = {
                'num_tables_consolidated': len(compatible_tables),
                'num_features_selected': 0,
                'provenance': provenance,
                'cleaning_audit': cleaning_audit
            }

            return consolidated_df, quality_report

        self.logger.info(f" Collected {len(feature_candidates)} NEW feature candidates")

        selected_features, quality_report = self._select_best_features(
            feature_candidates, valid_tables, consolidated_df
        )

        self.logger.info(f" Selected {len(selected_features)} best feature versions")

        for feature_name, selected_info in selected_features.items():
            if feature_name in consolidated_df.columns:
                self.logger.debug(f"   Skipping {feature_name} - already present")
                continue

            table_idx = selected_info['table_idx']
            source_table = valid_tables[table_idx]
            consolidated_df[feature_name] = source_table[feature_name].values

        self.logger.info(f" After feature addition: {consolidated_df.shape}")

        consolidated_df, cleaning_audit = self._clean_features_for_ml(
            consolidated_df,
            target_column
        )

        final_report = self._generate_quality_report(
            quality_report,
            selected_features,
            len(compatible_tables),
            cleaning_audit
        )

        provenance = self._build_feature_provenance(
            consolidated_df,
            original_base_df,
            target_column
        )

        final_report['provenance'] = provenance

        self.logger.info("=" * 70)
        self.logger.info(f" Consolidation complete:")
        self.logger.info(f"   Final shape: {consolidated_df.shape}")
        self.logger.info(f"   Base features: {provenance['counts']['base']}")
        self.logger.info(f"   Augmented features: {provenance['counts']['augmented']}")
        self.logger.info(f"   Target column: {provenance['target_column']}")
        self.logger.info("=" * 70)

        return consolidated_df, final_report

    def _collect_feature_candidates(self,
                                    valid_tables: List[pd.DataFrame],
                                    base_columns: Set[str]) -> Dict[str, List[Dict]]:
        from collections import defaultdict

        feature_candidates = defaultdict(list)

        for table_idx, table in enumerate(valid_tables):
            for col in table.columns:
                # Extract column name without prefix (handle double-dot columns)
                if '.' in col:
                    col_name = col.split('.', 1)[1]
                else:
                    col_name = col

                if col_name in base_columns:
                    continue

                null_ratio = table[col].isnull().mean()
                unique_ratio = table[col].nunique() / len(table) if len(table) > 0 else 0

                feature_candidates[col].append({
                    'table_idx': table_idx,
                    'null_ratio': null_ratio,
                    'unique_ratio': unique_ratio,
                    'column_name': col
                })

        return dict(feature_candidates)

    def _select_best_features(self,
                              feature_candidates: Dict[str, List[Dict]],
                              valid_tables: List[pd.DataFrame],
                              base_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        selected_features = {}
        quality_report = {}

        for feature_name, candidates in feature_candidates.items():
            if len(candidates) == 1:
                selected_features[feature_name] = candidates[0]
                quality_report[feature_name] = {
                    'versions': 1,
                    'selected': candidates[0],
                    'reason': 'Only version available'
                }
            else:
                # Sort by: lowest null ratio, then highest unique ratio
                candidates_sorted = sorted(
                    candidates,
                    key=lambda x: (x['null_ratio'], -x['unique_ratio'])
                )

                best = candidates_sorted[0]
                selected_features[feature_name] = best
                quality_report[feature_name] = {
                    'versions': len(candidates),
                    'selected': best,
                    'alternatives': candidates_sorted[1:],
                    'reason': f'Lowest null ratio ({best["null_ratio"]:.2%})'
                }

        return selected_features, quality_report

    def _generate_quality_report(self,
                                 quality_report: Dict,
                                 selected_features: Dict,
                                 num_tables: int,
                                 cleaning_audit: Dict = None) -> Dict[str, Any]:
        return {
            'num_tables_consolidated': num_tables,
            'num_features_selected': len(selected_features),
            'feature_quality': quality_report,
            'cleaning_audit': cleaning_audit or {}
        }

    def _build_feature_provenance(self,
                                  consolidated_df: pd.DataFrame,
                                  original_base_df: pd.DataFrame,
                                  target_column: str) -> Dict[str, Any]:
        all_join_keys = self.join_keys_info.get('all_join_keys', set())

        self.logger.info(f" Building feature provenance:")
        self.logger.info(f"   Total columns: {len(consolidated_df.columns)}")
        self.logger.info(f"   Known join keys: {len(all_join_keys)}")

        base_feature_names = set(original_base_df.columns) - {target_column}

        base_features = []
        augmented_features = []
        join_key_columns = []
        target_col = None

        for col in consolidated_df.columns:
            if '.' in col:
                parts = col.split('.', 1)
                table_prefix = parts[0]
                col_name = parts[1] if len(parts) > 1 else col
            else:
                col_name = col
                table_prefix = None

            if col_name == target_column:
                target_col = col
                continue

            # Join keys in connections.csv are unprefixed column names
            is_join_key = col_name in all_join_keys

            # Also check for idx.* pattern (common join key naming)
            is_idx_column = col_name.startswith('idx.')

            if is_join_key or is_idx_column:
                join_key_columns.append(col)
                self.logger.debug(f"    Join key: {col}")
                continue

            if col_name in base_feature_names or (table_prefix == self.base_table_name):
                base_features.append(col)
            else:
                augmented_features.append(col)

        self.logger.info(f"    Base features: {len(base_features)}")
        self.logger.info(f"    Augmented features: {len(augmented_features)}")
        self.logger.info(f"    Join key columns (excluded): {len(join_key_columns)}")

        if join_key_columns:
            self.logger.info(f"   Join keys excluded from feature selection:")
            for jk in join_key_columns[:10]:
                self.logger.info(f"      - {jk}")
            if len(join_key_columns) > 10:
                self.logger.info(f"      ... and {len(join_key_columns) - 10} more")

        return {
            'base_features': base_features,
            'augmented_features': augmented_features,
            'join_keys': join_key_columns,
            'target_column': target_col,
            'counts': {
                'base': len(base_features),
                'augmented': len(augmented_features),
                'join_keys': len(join_key_columns),
                'total': len(base_features) + len(augmented_features)
            }
        }
