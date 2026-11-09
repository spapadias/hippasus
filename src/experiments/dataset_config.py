"""
Dataset configuration for Hippasus experiments.
Maps dataset names to (folder, base_csv, target_column, task_type).
"""

DATASET_MAP = {
    "school": (
        "data/school",
        "base.csv",
        "class",
        "binary",
    ),
    "credit": (
        "data/credit",
        "table_0_0.csv",
        "class",
        "binary",
    ),
    "steel": (
        "data/steel",
        "table_0_0.csv",
        "Class",
        "binary",
    ),
    "eyemove": (
        "data/eyemove",
        "table_0_0.csv",
        "label",
        "binary",
    ),
    "jannis": (
        "data/jannis",
        "table_0_0.csv",
        "class",
        "binary",
    ),
    "miniboone": (
        "data/miniboone",
        "table_0_0.csv",
        "signal",
        "binary",
    ),
    "covertype": (
        "data/covertype",
        "table_0_0.csv",
        "class",
        "binary",
    ),
    "bioresponse": (
        "data/bioresponse",
        "table_0_0.csv",
        "target",
        "binary",
    ),
    "northwind": (
        "data/northwind",
        "order_details.csv",
        "Quantity",
        "regression",
    ),
    "fraud": (
        "data/fraud",
        "base.csv",
        "isFraud",
        "binary",
    ),
    "diabetes": (
        "data/diabetes",
        "base.csv",
        "Diabetes_binary",
        "binary",
    ),
    "air": (
        "data/air",
        "base.csv",
        "Temperature",
        "regression",
    ),
    "poverty": (
        "data/poverty",
        "base.csv",
        "POVALL_2016",
        "regression",
    ),
}
