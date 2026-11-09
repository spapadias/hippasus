"""
Hippasus — main experiment runner.

Reads configuration from src/experiments/configs/hippasus.yaml
and runs the Hippasus pipeline for the specified datasets and seeds.

Usage:
    python run_hippasus.py                          # use hippasus.yaml
    python run_hippasus.py --no-cache               # ignore cache, rerun all
    python run_hippasus.py --config path/to/custom.yaml

Results are saved to results/hippasus_cache.pkl
Run: python get_results.py   to view a summary table.
"""

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Python randomizes string hashing per process, which changes the column order behind the mutual-information
# estimates and hence the selected features between identical runs: restart once with a fixed hash seed.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


@contextlib.contextmanager
def suppress_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# Pipeline settings of the four variants
VARIANT_CONFIG = {
    'hippasus_adaptive': {
        'pex_mode':       'adaptive',
        'fs_method':      'hippasus-adaptive-v2',
        'fdg':            True,
        'use_slm':        False,
        'use_stats_only': False,
        'use_adaptive':   True,
    },
    'hippasus_stats': {
        'pex_mode':       'stats_only',
        'fs_method':      'hippasus-stats',
        'fdg':            False,
        'use_slm':        False,
        'use_stats_only': True,
        'use_adaptive':   False,
    },
    'hippasus_slm': {
        'pex_mode':       'slm',
        'fs_method':      'hippasus-slm',
        'fdg':            True,
        'use_slm':        True,
        'use_stats_only': False,
        'use_adaptive':   False,
    },
    'hippasus_llm': {
        'pex_mode':       'llm',
        'fs_method':      'hippasus-llm-v2',
        'fdg':            True,
        'use_slm':        False,
        'use_stats_only': False,
        'use_adaptive':   False,
    },
}

PRIMARY_METRIC = {
    'credit':    ('avg_accuracy', 'Acc'),
    'steel':     ('avg_accuracy', 'Acc'),
    'eyemove':   ('avg_accuracy', 'Acc'),
    'jannis':    ('avg_accuracy', 'Acc'),
    'miniboone': ('avg_accuracy', 'Acc'),
    'covertype': ('avg_accuracy', 'Acc'),
    'fraud':     ('avg_f1',       'F1'),
    'diabetes':  ('avg_accuracy', 'Acc'),
    'poverty':   ('avg_mae',      'MAE'),
    'air':       ('avg_rmse',     'RMSE'),
    'northwind': ('avg_mae',      'MAE'),
    'school':    ('avg_accuracy', 'Acc'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run Hippasus pipeline')
    parser.add_argument(
        '--config',
        default='src/experiments/configs/hippasus.yaml',
        help='Path to YAML config (default: src/experiments/configs/hippasus.yaml)',
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Ignore existing cache and rerun all configurations from scratch',
    )
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_cache_key(variant, dataset, seed, cfg):
    key = {
        'variant':      variant,
        'dataset':      dataset,
        'seed':         seed,
        'max_depth':    cfg['pipeline']['max_depth'],
        'num_paths':    cfg['pipeline']['num_paths'],
        'adaptive_tau': cfg['pipeline'].get('adaptive_tau', 0.001),
        'k':            cfg['feature_selection']['k'],
        'pre_filter_K': cfg['feature_selection'].get('pre_filter_K', 100),
        'model':        cfg['llm']['model'],
    }
    import hashlib
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()


def run_single(variant, dataset, seed, cfg):
    from src.experiments.hippasus_pipeline import Hippasus
    from src.experiments.dataset_config import DATASET_MAP
    from src.cost_logger import cost_logger

    if variant not in VARIANT_CONFIG:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANT_CONFIG)}")
    if dataset not in DATASET_MAP:
        raise ValueError(f"Unknown dataset '{dataset}'. Check DATASET_MAP.")

    vc             = VARIANT_CONFIG[variant]
    fs_method      = vc['fs_method']
    fdg_enabled    = vc['fdg']
    use_slm        = vc['use_slm']
    use_stats_only = vc['use_stats_only']
    use_adaptive   = vc['use_adaptive']

    folder, base_csv, target, task = DATASET_MAP[dataset]

    join_algo    = cfg['pipeline'].get('join_algo', 'suffix-yannakakis')
    max_depth    = cfg['pipeline']['max_depth']
    num_paths    = cfg['pipeline']['num_paths']
    adaptive_tau = cfg['pipeline'].get('adaptive_tau', 0.001)
    test_size    = cfg['pipeline'].get('test_size', 0.2)
    k            = cfg['feature_selection']['k']
    pre_filter_K = cfg['feature_selection'].get('pre_filter_K', 100)
    llm_model    = cfg['llm']['model']
    time_limit   = cfg['evaluation'].get('time_limit')

    # Settings the pipeline components read from the environment
    os.environ['HIPPASUS_PE_FEATURE_DESC']      = 'true' if fdg_enabled else 'false'
    os.environ['HIPPASUS_FS_FEATURE_DESC']      = 'true' if fdg_enabled else 'false'
    os.environ['HIPPASUS_LLM_MODEL']            = llm_model
    os.environ['HIPPASUS_PATH_EXPLORER_LLM']    = llm_model
    os.environ['HIPPASUS_FEATURE_SELECTOR_LLM'] = llm_model
    os.environ['HIPPASUS_TIE_SEED']             = str(seed)  # path-score ties: a visiting order seeded per run

    cost_logger.set_context(dataset=dataset, seed=seed)

    timings = {}
    experiment_start = time.time()

    t = time.time()
    pipe = Hippasus(
        dataset_folder=folder,
        base_table_name=base_csv,
        target_column=target,
        task=task,
        join_algo=join_algo,
        time_limit=time_limit,
        dataset_name=dataset,
    )
    timings['initialization'] = time.time() - t

    t = time.time()
    pipe.step1_load_base_table()
    timings['step1_load_base'] = time.time() - t

    t = time.time()
    with suppress_stdout():
        pipe.step2_path_explorer(
            max_depth=max_depth,
            top_k=num_paths,
            use_slm=use_slm,
            use_stats_only=use_stats_only,
            use_adaptive=use_adaptive,
            adaptive_tau=adaptive_tau,
        )
    timings['step2_path_explorer'] = time.time() - t

    # Read cascade decision from JSON written by step2
    cascade_decision = None
    cascade_boundary_gap = None
    cascade_entropy_confidence = None
    if use_adaptive:
        cascade_path = Path(
            f"src/experiments/results/cascade/{dataset}_pex_decision.json"
        )
        if cascade_path.exists():
            with open(cascade_path) as f:
                ci = json.load(f)
            cascade_decision           = ci.get('decision')
            cascade_boundary_gap       = ci.get('boundary_gap')
            cascade_entropy_confidence = ci.get('entropy_confidence')

    t = time.time()
    with suppress_stdout():
        pipe.step3_join_executor(max_paths=num_paths)
    timings['step3_join_executor'] = time.time() - t

    t = time.time()
    with suppress_stdout():
        pipe.step4_consolidate()
    timings['step4_consolidate'] = time.time() - t

    t = time.time()
    with suppress_stdout():
        pipe.step5_split(test_size=test_size, random_state=seed)
    timings['step5_split'] = time.time() - t

    t = time.time()
    with suppress_stdout():
        pipe.step6_feature_selector(
            method=fs_method,
            max_features=k,
            pre_filter_K=pre_filter_K,
        )
    timings['step6_feature_selector'] = time.time() - t

    t = time.time()
    with suppress_stdout():
        detailed_scores = pipe.step7_ml_trainer(
            train_data=pipe.selected_train_df,
            test_data=pipe.selected_test_df,
            evaluation_name=f"{variant}_eval",
        )
    timings['step7_ml_trainer'] = time.time() - t

    timings['total_time'] = time.time() - experiment_start
    timings['feature_discovery_time'] = (
        timings['step1_load_base'] +
        timings['step2_path_explorer'] +
        timings['step3_join_executor'] +
        timings['step4_consolidate'] +
        timings['step5_split'] +
        timings['step6_feature_selector']
    )

    em = detailed_scores.get('ENSEMBLE', {})
    if task in ('binary', 'multiclass'):
        metrics = {
            'avg_accuracy': em.get('accuracy'),
            'avg_f1':       em.get('f1') or em.get('f1_macro'),
            'avg_auc':      em.get('auc'),
        }
    else:
        metrics = {
            'avg_rmse': em.get('rmse'),
            'avg_mae':  em.get('mae'),
        }

    selected_augmented_list = (
        pipe.selected_features_metadata['selected_augmented']
        if hasattr(pipe, 'selected_features_metadata') else []
    )
    prov = pipe.feature_provenance if hasattr(pipe, 'feature_provenance') else {}
    counts = prov.get('counts', {})

    summary        = cost_logger.summary_by_component()
    estimated_cost = cost_logger.compute_cost()
    cost_logger.reset()

    return {
        'variant':                    variant,
        'dataset':                    dataset,
        'seed':                       seed,
        'task_type':                  task,
        'pex_mode':                   vc['pex_mode'],
        'fs_method':                  fs_method,
        'fdg':                        fdg_enabled,
        'use_slm':                    use_slm,
        'use_stats_only':             use_stats_only,
        'use_adaptive':               use_adaptive,
        'adaptive_tau':               adaptive_tau if use_adaptive else None,
        'k':                          k,
        'max_depth':                  max_depth,
        'num_paths':                  num_paths,
        'join_algo':                  join_algo,
        'test_size':                  test_size,
        'llm_model':                  llm_model,
        'cascade_decision':           cascade_decision,
        'cascade_boundary_gap':       cascade_boundary_gap,
        'cascade_entropy_confidence': cascade_entropy_confidence,
        'base_features':              counts.get('base', 0),
        'augmented_features':         counts.get('augmented', 0),
        'selected_augmented_features': selected_augmented_list,
        'time_initialization':        timings['initialization'],
        'time_step1_load_base':       timings['step1_load_base'],
        'time_step2_path_explorer':   timings['step2_path_explorer'],
        'time_step3_join_executor':   timings['step3_join_executor'],
        'time_step4_consolidate':     timings['step4_consolidate'],
        'time_step5_split':           timings['step5_split'],
        'time_step6_feature_selector': timings['step6_feature_selector'],
        'time_step7_ml_trainer':      timings['step7_ml_trainer'],
        'time_feature_discovery':     timings['feature_discovery_time'],
        'time_total':                 timings['total_time'],
        'llm_input_tokens':           sum(s['input_tokens']  for s in summary.values()),
        'llm_output_tokens':          sum(s['output_tokens'] for s in summary.values()),
        'llm_total_tokens':           sum(s['total_tokens']  for s in summary.values()),
        'llm_estimated_cost_usd':     round(estimated_cost, 6),
        'llm_calls_pex':              summary.get('PEX', {}).get('calls', 0),
        'llm_calls_fs':               summary.get('FS',  {}).get('calls', 0),
        'llm_input_tokens_pex':       summary.get('PEX', {}).get('input_tokens', 0),
        'llm_input_tokens_fs':        summary.get('FS',  {}).get('input_tokens', 0),
        **metrics,
    }


def format_result(result):
    ds      = result['dataset']
    variant = result['variant']
    seed    = result['seed']
    elapsed = result.get('time_feature_discovery', 0) or 0
    tokens  = result.get('llm_input_tokens', 0) or 0
    cascade = result.get('cascade_decision', 'N/A')

    field, label = PRIMARY_METRIC.get(ds, ('avg_accuracy', 'Acc'))
    value = result.get(field) or float('nan')

    print(
        f"Dataset: {ds:<12} | Variant: {variant:<22} | Seed: {seed:<5} | "
        f"{label}: {value:.4f} | Time: {elapsed:.1f}s | "
        f"Tokens: {tokens:>6} | Cascade: {cascade}"
    )


def main():
    args = parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    cfg       = load_config(args.config)
    use_cache = (not args.no_cache) and cfg['output'].get('use_cache', True)

    results_dir = Path(cfg['output']['results_dir'])
    cache_path  = Path(cfg['output']['cache_file'])
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.no_cache and cache_path.exists():
        cache_path.unlink()
        print(f"Cache cleared: {cache_path}")

    cache = {}
    if use_cache and cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

    variant  = cfg['experiment']['variant']
    datasets = cfg['experiment']['datasets']
    seeds    = cfg['experiment']['seeds']
    total    = len(datasets) * len(seeds)

    print(f"Hippasus | variant: {variant} | datasets: {datasets} | seeds: {seeds}")
    print(f"Running {total} configuration(s)...")
    print()

    completed = skipped = failed = 0

    for dataset in datasets:
        for seed in seeds:
            cache_key = make_cache_key(variant, dataset, seed, cfg)

            if use_cache and cache_key in cache:
                result = cache[cache_key]['result']
                print('[cached] ', end='')
                format_result(result)
                skipped += 1
                continue

            try:
                result = run_single(variant, dataset, seed, cfg)
                cache[cache_key] = {'result': result}
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache, f)
                format_result(result)
                completed += 1
            except Exception as e:
                print(f"Dataset: {dataset:<12} | Seed: {seed} | ERROR: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print()
    summary = f"Done. Completed: {completed}/{total}"
    if skipped:
        summary += f" | Cached: {skipped}"
    if failed:
        summary += f" | Failed: {failed}"
    print(summary)
    if completed > 0 or skipped > 0:
        print(f"Results saved to: {cache_path}")
        print(f"Run 'python get_results.py' to view summary table.")
    print()
    print("To rerun from scratch:  python run_hippasus.py --no-cache")


if __name__ == '__main__':
    main()
