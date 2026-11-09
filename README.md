# Hippasus: Accurate and Cost-Efficient Feature Augmentation over Relational Tables

Code of the paper. The full version is on [arXiv](https://arxiv.org/abs/2602.02025).

## Setup

Python 3.12, and an [OpenRouter](https://openrouter.ai) API key for the LLM calls.

```bash
git clone https://github.com/spapadias/hippasus.git
cd hippasus
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the key in a file `.env` in the repository folder (or export it):

```bash
OPENROUTER_API_KEY=your-key
```

## Datasets

The 13 datasets of the paper, in the exact form used there:
[Google Drive](https://drive.google.com/drive/folders/1zzWgmhY-jUdIZTpyf-YfIdffETiItu2-?usp=sharing).
Download the 13 folders into `data/` (`data/school/`, `data/credit/`, ...).

| Dataset       | Task                  | Target            | Base table          | Metric printed |
|---------------|-----------------------|-------------------|---------------------|----------------|
| `school`      | binary classification | `class`           | `base.csv`          | accuracy       |
| `credit`      | binary classification | `class`           | `table_0_0.csv`     | accuracy       |
| `eyemove`     | binary classification | `label`           | `table_0_0.csv`     | accuracy       |
| `steel`       | binary classification | `Class`           | `table_0_0.csv`     | accuracy       |
| `jannis`      | binary classification | `class`           | `table_0_0.csv`     | accuracy       |
| `miniboone`   | binary classification | `signal`          | `table_0_0.csv`     | accuracy       |
| `covertype`   | binary classification | `class`           | `table_0_0.csv`     | accuracy       |
| `bioresponse` | binary classification | `target`          | `table_0_0.csv`     | accuracy       |
| `diabetes`    | binary classification | `Diabetes_binary` | `base.csv`          | accuracy       |
| `fraud`       | binary classification | `isFraud`         | `base.csv`          | F1             |
| `poverty`     | regression            | `POVALL_2016`     | `base.csv`          | MAE            |
| `air`         | regression            | `Temperature`     | `base.csv`          | RMSE           |
| `northwind`   | regression            | `Quantity`        | `order_details.csv` | MAE            |

These are the datasets released with AutoFeat and FeatPilot, with the corrections described in Appendix C of the paper.

## Running

1. Set the variant, the datasets and the seeds in `src/experiments/configs/hippasus.yaml`.
2. Run `python run_hippasus.py`. Each run prints one line with its metric.
3. Run `python get_results.py` for the mean and standard deviation over the seeds.

Results are cached in `results/hippasus_cache.pkl`; `python run_hippasus.py --no-cache` runs everything again.

| `variant`           | In the paper             | Needs the API key |
|---------------------|--------------------------|-------------------|
| `hippasus_adaptive` | Hippasus                 | yes               |
| `hippasus_llm`      | Hippasus<sub>LLM</sub>   | yes               |
| `hippasus_slm`      | Hippasus<sub>SLM</sub>   | no                |
| `hippasus_stats`    | Hippasus<sub>stats</sub> | no                |

The defaults in `hippasus.yaml` are the paper's. LLM replies can vary between runs, so results can differ slightly
from the paper's.
