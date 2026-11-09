"""
Join Executor that runs each join path with suffix-Yannakakis (the default join plan).
"""
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List


class SuffixYannakakisJoinExecutor:
    """Materializes a join path suffix-first: reduce the suffix tables from the root down, left-join them, then
    left-join the result to the base table. Same rows and values as the binary left-join chain."""

    def __init__(self):
        self.dataset_folder = None
        self.connections_df = None
        self.join_keys_info = None
        self.base_table_name = None

        self.add_row_id = False
        self.row_id_col = "__row_id"
        self.log_quality = True
        self._last_join_stats = []

    def set_dataset_folder(self, folder_path):
        """Point to dataset folder containing CSV files and connections.csv"""
        self.dataset_folder = Path(folder_path)
        self.load_connections()

    def set_join_keys_info(self, join_keys_info: Dict):
        """MANDATORY: Receive join key information from Path Explorer"""
        if join_keys_info is None:
            raise ValueError("join_keys_info cannot be None")
        self.join_keys_info = join_keys_info
        logging.info("Suffix-Yannakakis Executor received key info")

    def get_join_keys_info(self) -> Dict:
        """Forward join key information to consolidator"""
        if self.join_keys_info is None:
            raise ValueError("join_keys_info not set")
        result = self.join_keys_info.copy()
        result['base_table_name'] = self.base_table_name
        result['connections_df'] = self.connections_df
        return result

    def load_connections(self):
        """Read connections.csv with FK→PK relationship definitions"""
        if self.dataset_folder is None:
            raise ValueError("Dataset folder not set")

        connections_path = self.dataset_folder / "connections.csv"
        if connections_path.exists():
            self.connections_df = pd.read_csv(connections_path)
            logging.info(f"Loaded {len(self.connections_df)} connections")
        else:
            raise FileNotFoundError("connections.csv not found")

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

        raise ValueError(f"No connection found: {from_table} -> {to_table}")

    def add_table_prefixes(self, df, table_name):
        """Rename columns to avoid collisions"""
        renamed_cols = {}
        for col in df.columns:
            if col.startswith(f"{table_name}."):
                renamed_cols[col] = col
            else:
                renamed_cols[col] = f"{table_name}.{col}"
        return df.rename(columns=renamed_cols)

    def _apply_null_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply null normalization"""
        df_clean = df.copy()
        null_sentinels = {
            "", " ", "NA", "N/A", "None", "null", "NULL", "\\N", "nan", "NaN"
        }

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                series = df_clean[col].astype(str).str.strip()
                df_clean[col] = series.replace(null_sentinels, pd.NA)

        return df_clean

    def _load_and_normalize_table(self, table_name: str, table_path: Path) -> pd.DataFrame:
        """Load table with consistent normalization"""
        df = pd.read_csv(table_path, dtype=str, low_memory=False)
        df = self._apply_null_normalization(df)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        return df

    def _normalize_join_key(self, series: pd.Series) -> pd.Series:
        """Apply consistent null normalization to join keys"""
        series = series.copy()

        if pd.api.types.is_numeric_dtype(series):
            TEMP_NULL_SENTINEL = -999999999
            series_filled = series.fillna(TEMP_NULL_SENTINEL)
            series_as_int = series_filled.astype(int)
            series_as_str = series_as_int.astype(str)
            series = series_as_str.replace(str(TEMP_NULL_SENTINEL), pd.NA)
        else:
            nulls = series.isna()  # astype(str) would turn a null into text that then matches other nulls
            series = series.astype(str).str.strip()
            series[nulls] = pd.NA

        null_sentinels = {
            "", "\\N", "nan", "NaN", "None", "NA", "N/A", "null", "NULL"
        }
        series = series.replace(null_sentinels, pd.NA)

        return series

    def _reduce_suffix(self, base_df: pd.DataFrame, base_table_name: str, names: List[str]) -> Dict[str, pd.DataFrame]:
        """Load each suffix table, keep the rows whose key its parent references, and keep one row per key.

        Only this direction is safe under left-join semantics: a pass that removed parent rows with no child match
        would lose values a left join must keep. Rows are filtered, never reordered, so the first row per key is the
        one the binary plan keeps.
        """
        parent_key, child_key = self.find_join_keys(base_table_name, names[0])
        keys = self._normalize_join_key(base_df[parent_key]).dropna().unique()
        reduced = {}
        for i, name in enumerate(names):
            df = self._load_and_normalize_table(name, self.dataset_folder / f"{name}.csv")
            df = self.add_table_prefixes(df, name)
            col = f"{name}.{child_key}"
            df[col] = self._normalize_join_key(df[col])
            df = df[df[col].isin(keys)].drop_duplicates(subset=[col], keep='first')
            reduced[name] = df
            if i + 1 < len(names):
                parent_key, child_key = self.find_join_keys(name, names[i + 1])
                keys = self._normalize_join_key(df[f"{name}.{parent_key}"]).dropna().unique()
        return reduced

    def execute_join_path(self, path_list: List[str], base_df: pd.DataFrame,
                          target_column: str) -> pd.DataFrame:
        if len(path_list) <= 1:
            return base_df.copy()

        base_df_normalized = self._apply_null_normalization(base_df.copy())
        for col in base_df_normalized.columns:
            base_df_normalized[col] = pd.to_numeric(base_df_normalized[col], errors="ignore")

        base_table_name = Path(path_list[0]).stem
        self.base_table_name = base_table_name
        names = [Path(p).stem for p in path_list[1:]]

        reduced = self._reduce_suffix(base_df_normalized, base_table_name, names)

        suffix_result = reduced[names[0]]
        for from_table, to_table in zip(names, names[1:]):
            from_col, to_col = self.find_join_keys(from_table, to_table)
            left_col = f"{from_table}.{from_col}"
            suffix_result = suffix_result.copy()
            suffix_result[left_col] = self._normalize_join_key(suffix_result[left_col])
            suffix_result = pd.merge(suffix_result, reduced[to_table], left_on=left_col,
                                     right_on=f"{to_table}.{to_col}", how='left')

        base_key, root_key = self.find_join_keys(base_table_name, names[0])
        base_df_prefixed = self.add_table_prefixes(base_df_normalized, base_table_name)
        base_join_col = f"{base_table_name}.{base_key}"
        base_df_prefixed[base_join_col] = self._normalize_join_key(base_df_prefixed[base_join_col])
        result = pd.merge(base_df_prefixed, suffix_result, left_on=base_join_col,
                          right_on=f"{names[0]}.{root_key}", how='left')

        if len(result) != len(base_df):
            raise RuntimeError(f"CARDINALITY VIOLATION: Base had {len(base_df)} rows, result has {len(result)} rows")
        return result
