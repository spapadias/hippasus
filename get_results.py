"""
Display results from Hippasus cache.

Usage:
    python get_results.py
    python get_results.py --cache path/to/cache.pkl
"""

import argparse
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cache',
        default='results/hippasus_cache.pkl',
        help='Path to results cache (default: results/hippasus_cache.pkl)',
    )
    return parser.parse_args()


PRIMARY = {
    'credit':    ('avg_accuracy', True,  'Acc'),
    'steel':     ('avg_accuracy', True,  'Acc'),
    'eyemove':   ('avg_accuracy', True,  'Acc'),
    'jannis':    ('avg_accuracy', True,  'Acc'),
    'miniboone': ('avg_accuracy', True,  'Acc'),
    'covertype': ('avg_accuracy', True,  'Acc'),
    'fraud':     ('avg_f1',       True,  'F1'),
    'diabetes':  ('avg_accuracy', True,  'Acc'),
    'poverty':   ('avg_mae',      False, 'MAE'),
    'air':       ('avg_rmse',     False, 'RMSE'),
    'northwind': ('avg_mae',      False, 'MAE'),
    'school':    ('avg_accuracy', True,  'Acc'),
}


def main():
    args = parse_args()
    cache_path = Path(args.cache)

    if not cache_path.exists():
        print(f"No results found at: {cache_path}")
        print("Run 'python run_hippasus.py' first.")
        return

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"Loaded {len(cache)} result(s) from {cache_path}")
    print()

    by_variant_ds = defaultdict(lambda: defaultdict(list))
    for v in cache.values():
        r   = v['result']
        var = r.get('variant', '?')
        ds  = r['dataset']
        field, _, _ = PRIMARY.get(ds, ('avg_accuracy', True, 'Acc'))
        val = r.get(field)
        if val is not None:
            by_variant_ds[var][ds].append(float(val))

    variants     = sorted(by_variant_ds.keys())
    all_datasets = sorted(set(ds for var in variants for ds in by_variant_ds[var]))

    col_w  = 12
    head   = f"{'Dataset':<{col_w}} | {'Metric':<6}"
    for var in variants:
        short = var.replace('hippasus_', '')
        head += f" | {short:>16}"
    print(head)
    print('-' * len(head))

    for ds in all_datasets:
        _, higher_better, metric = PRIMARY.get(ds, ('avg_accuracy', True, 'Acc'))
        row = f"{ds:<{col_w}} | {metric:<6}"
        for var in variants:
            vals = by_variant_ds[var][ds]
            if vals:
                mean = float(np.mean(vals))
                std  = float(np.std(vals)) if len(vals) > 1 else 0.0
                cell = f"{mean:.4f}" + (f" ±{std:.3f}" if std > 0 else "")
                row += f" | {cell:>16}"
            else:
                row += f" | {'—':>16}"
        print(row)

    print()
    seeds_found = sorted({v['result'].get('seed') for v in cache.values()})
    print(f"Seeds in results: {seeds_found}")
    total_tokens = sum(
        v['result'].get('llm_input_tokens', 0) or 0 for v in cache.values()
    )
    if total_tokens > 0:
        print(f"Total input tokens across all runs: {total_tokens:,}")
    print()
    print("To rerun:  python run_hippasus.py --no-cache")


if __name__ == '__main__':
    main()
