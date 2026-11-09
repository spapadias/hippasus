"""
Join Executor that runs each join path as a chain of binary left outer joins (the baseline join plan).
"""
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Set, List, Union


class BinaryJoinExecutor:
    """Runs a join path as a chain of binary left outer joins (the baseline join plan)"""

    def __init__(self):
        self.dataset_folder = None
        self.connections_df = None
        self.join_keys_info = None  # MANDATORY - must be set before use
        self.base_table_name = None

        self.add_row_id = False
        self.row_id_col = "__row_id"
        self.log_quality = True
        self._last_join_stats = []

    def set_dataset_folder(self, folder_path):
        """Point to dataset folder containing CSV files and connections.csv"""
        self.dataset_folder = Path(folder_path)
        self.load_connections()

    def set_join_keys_info(self, join_keys_info: Dict[str, Union[Set[str], List[Dict]]]):
        """
        MANDATORY: Receive join key information from Path Explorer

        This method MUST be called before executing joins
        """
        if join_keys_info is None:
            raise ValueError("join_keys_info cannot be None - Path Explorer must provide this information")

        self.join_keys_info = join_keys_info
        logging.info(
            f"Join Executor received key info: {len(join_keys_info.get('all_join_keys', set()))} unique join keys")

    def get_join_keys_info(self) -> Dict[str, Union[Set[str], List[Dict]]]:
        """Forward join key information AND base table name to consolidator"""
        if self.join_keys_info is None:
            raise ValueError("join_keys_info not set - call set_join_keys_info() first")

        result = self.join_keys_info.copy()
        result['base_table_name'] = self.base_table_name
        result['connections_df'] = self.connections_df
        return result

    def load_connections(self):
        """Read connections.csv with FK→PK relationship definitions"""
        if self.dataset_folder is None:
            raise ValueError("Dataset folder not set")

        try:
            connections_path = self.dataset_folder / "connections.csv"
            if connections_path.exists():
                self.connections_df = pd.read_csv(connections_path)
                logging.info(f"Loaded {len(self.connections_df)} connections")
            else:
                raise FileNotFoundError("connections.csv not found")
        except Exception as e:
            raise RuntimeError(f"Failed to load connections: {e}")

    def find_join_keys(self, from_table, to_table):
        """Find join columns from connections.csv"""
        if self.connections_df is None:
            raise ValueError("connections.csv not loaded")

        for _, row in self.connections_df.iterrows():
            fk_table = Path(str(row.iloc[0])).stem
            fk_col = str(row.iloc[1])
            pk_table = Path(str(row.iloc[2])).stem
            pk_col = str(row.iloc[3])

            if fk_table == from_table and pk_table == to_table:
                return (fk_col, pk_col)
            elif fk_table == to_table and pk_table == from_table:
                return (pk_col, fk_col)

        raise ValueError(f"No connection found in connections.csv: {from_table} -> {to_table}")

    def add_table_prefixes(self, df, table_name):
        renamed_cols = {}
        for col in df.columns:
            if col.startswith(f"{table_name}."):
                renamed_cols[col] = col
            else:
                renamed_cols[col] = f"{table_name}.{col}"
        return df.rename(columns=renamed_cols)

    def remove_duplicates(self, right_df, join_column):
        """Remove duplicate keys to maintain left join cardinality"""
        return right_df.drop_duplicates(subset=[join_column])

    def _normalize_join_key(self, series: pd.Series) -> pd.Series:
        series = series.copy()

        if pd.api.types.is_numeric_dtype(series):
            # Numeric series: convert to integer representation
            # This handles both int64 and float64 uniformly

            # Fill NaN temporarily: NaN cannot be converted to int
            # Use a sentinel value that's unlikely in real data
            TEMP_NULL_SENTINEL = -999999999
            series_filled = series.fillna(TEMP_NULL_SENTINEL)

            # Convert to int (removes the .0 of floats)
            series_as_int = series_filled.astype(int)

            series_as_str = series_as_int.astype(str)

            # Restore the nulls
            series = series_as_str.replace(str(TEMP_NULL_SENTINEL), pd.NA)
        else:
            nulls = series.isna()  # astype(str) would turn a null into text that then matches other nulls
            series = series.astype(str).str.strip()
            series[nulls] = pd.NA

        # These are common ways nulls appear in CSV data
        null_sentinels = {
            "",  # Empty string
            "\\N",  # MySQL/PostgreSQL null export
            "nan",  # NumPy string representation
            "NaN",  # Pandas string representation
            "None",  # Python None as string
            "NA",  # R/statistical software
            "N/A",  # Human-readable null
            "null",  # JSON null (lowercase)
            "NULL"  # SQL NULL (uppercase)
        }
        series = series.replace(null_sentinels, pd.NA)

        return series

    def _apply_null_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same null normalization as Path Explorer for consistency"""
        df_clean = df.copy()

        null_sentinels = {
            "", " ", "NA", "N/A", "None", "null", "NULL", "\\N", "nan", "NaN"
        }

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                series = df_clean[col].astype(str).str.strip()
                df_clean[col] = series.replace(null_sentinels, pd.NA)

        return df_clean

    def execute_single_join(self, left_df, right_table_path, join_keys, left_table_name):
        """Perform ONE pandas left join with consistent null handling"""
        if join_keys is None:
            raise ValueError("Join keys cannot be None")

        from_col, to_col = join_keys
        to_table = Path(right_table_path).stem

        right_df = pd.read_csv(right_table_path, dtype=str, low_memory=False)

        right_df = self._apply_null_normalization(right_df)

        for col in right_df.columns:
            right_df[col] = pd.to_numeric(right_df[col], errors="ignore")

        right_df = self.add_table_prefixes(right_df, to_table)
        right_df = self.remove_duplicates(right_df, f"{to_table}.{to_col}")

        left_join_col = f"{left_table_name}.{from_col}"
        right_join_col = f"{to_table}.{to_col}"

        left_df[left_join_col] = self._normalize_join_key(left_df[left_join_col])
        right_df[right_join_col] = self._normalize_join_key(right_df[right_join_col])

        # Remove null RHS keys
        right_df = right_df.dropna(subset=[right_join_col])

        if left_join_col not in left_df.columns:
            raise KeyError(f"Join column missing in left table: {left_join_col}")
        if right_join_col not in right_df.columns:
            raise KeyError(f"Join column missing in right table: {right_join_col}")

        pre_cols = set(left_df.columns)
        result_df = pd.merge(
            left_df, right_df, left_on=left_join_col, right_on=right_join_col, how='left'
        )
        post_cols = set(result_df.columns)

        try:
            left_keys = left_df[left_join_col].nunique(dropna=True)
        except Exception:
            left_keys = None
        try:
            right_keys = right_df[right_join_col].nunique(dropna=True)
        except Exception:
            right_keys = None

        matched = result_df[right_join_col].notna().sum() if right_join_col in result_df.columns else 0
        coverage = matched / len(result_df) if len(result_df) else 0.0
        added_cols = [c for c in (post_cols - pre_cols) if not c.endswith(self.row_id_col)]

        if self.log_quality:
            print(
                f"Join {left_table_name} → {to_table}: "
                f"match={coverage:.1%} (matched {matched}/{len(result_df)}), "
                f"Lkeys={left_keys}, Rkeys={right_keys}, +cols={len(added_cols)}"
            )
            null_ratio = (
                result_df[right_join_col].isna().mean() if right_join_col in result_df.columns else 1.0
            )
            print(f"   ↳ nulls on {right_join_col}: {null_ratio:.3f} ({null_ratio * 100:.1f}% nulls)")

        self._last_join_stats.append({
            "from": left_table_name,
            "to": to_table,
            "left_rows": int(len(left_df)),
            "right_rows": int(len(right_df)),
            "left_keys": None if left_keys is None else int(left_keys),
            "right_keys": None if right_keys is None else int(right_keys),
            "matched_rows": int(matched),
            "coverage": float(coverage),
            "added_cols": sorted(list(added_cols)),
            "left_join_col": left_join_col,
            "right_join_col": right_join_col,
            "null_ratio": float(result_df[right_join_col].isna().mean()
                                if right_join_col in result_df.columns else 1.0),
        })

        return result_df

    def execute_join_path(self, path_list, base_df, target_column):
        """Execute FULL path using sequential binary joins with consistent normalization"""
        if len(path_list) <= 1:
            return base_df.copy()

        base_df_normalized = self._apply_null_normalization(base_df.copy())

        for col in base_df_normalized.columns:
            base_df_normalized[col] = pd.to_numeric(base_df_normalized[col], errors="ignore")

        base_table_name = Path(path_list[0]).stem
        self.base_table_name = base_table_name

        if self.add_row_id and self.row_id_col not in base_df_normalized.columns:
            base_df_normalized = base_df_normalized.copy()
            base_df_normalized[self.row_id_col] = range(len(base_df_normalized))

        current_df = self.add_table_prefixes(base_df_normalized, base_table_name)
        current_table_name = base_table_name

        self._last_join_stats = []

        print(f" Executing path: {' → '.join([Path(p).stem for p in path_list])}")

        for i in range(1, len(path_list)):
            from_table = current_table_name
            to_table = Path(path_list[i]).stem
            right_table_path = self.dataset_folder / f"{to_table}.csv"

            print(f"  Join {i}: {from_table} → {to_table}")

            join_keys = self.find_join_keys(from_table, to_table)
            print(f"    Using keys: {join_keys}")

            current_df = self.execute_single_join(current_df, right_table_path, join_keys, from_table)

            current_table_name = to_table

        total_nulls = current_df.isnull().sum().sum()
        total_cells = current_df.shape[0] * current_df.shape[1]
        final_null_ratio = total_nulls / total_cells if total_cells > 0 else 0

        print(f"Final path null ratio: {final_null_ratio:.3f}")

        if self.log_quality and self._last_join_stats:
            print("\n Join quality per-hop:")
            for j, s in enumerate(self._last_join_stats, 1):
                print(
                    f"  {j:02d}. {s['from']} → {s['to']} | "
                    f"match={s['coverage']:.1%} | nulls={s['null_ratio']:.1%} | +cols={len(s['added_cols'])}"
                )

        return current_df
