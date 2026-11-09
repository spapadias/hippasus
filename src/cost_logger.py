"""
Cost logger for Hippasus LLM calls.
Tracks input/output tokens, wall-clock time, and estimated dollar cost
for every OpenRouter API call made during an experiment run.
"""
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List


# Per-model pricing: (input_per_1M, output_per_1M) in USD
MODEL_PRICING = {
    'openai/gpt-6-luna':                (0.10,  0.50),
    'openai/gpt-4o-mini':               (0.15,  0.60),
    'openai/gpt-4o':                    (2.50, 10.00),
    'meta-llama/llama-3.3-70b-instruct':(0.30,  0.30),
    'meta-llama/llama-3.1-8b-instruct': (0.10,  0.10),
    'mistralai/mistral-nemo':           (0.15,  0.15),
    'qwen/qwen-2.5-72b-instruct':       (0.30,  0.30),
    'anthropic/claude-3.5-sonnet':      (3.00, 15.00),
}


@dataclass
class LLMCallRecord:
    component: str           # "PEX" or "FS"
    dataset: str
    model: str
    input_tokens: int
    output_tokens: int
    call_type: str           # "always", "proxy", "oracle"
    wall_clock_seconds: float
    cascade_decision: Optional[str] = None  # "proxy_only" or "oracle_invoked"
    seed: Optional[int] = None


class CostLogger:

    def __init__(self):
        self._records: List[LLMCallRecord] = []
        self._timer_start: float = 0.0
        self._current_dataset: str = "unknown"
        self._current_seed: Optional[int] = None

    def set_context(self, dataset: str, seed: Optional[int] = None):
        """Set current dataset and seed context for all subsequent log_call calls."""
        self._current_dataset = dataset
        self._current_seed = seed

    def start_timer(self):
        self._timer_start = time.time()

    def log_call(self, component: str, model: str, usage: dict,
                 call_type: str = "always",
                 cascade_decision: Optional[str] = None,
                 dataset: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Log a completed LLM API call.
        usage must be the dict from response_json['usage']:
          {'prompt_tokens': int, 'completion_tokens': int, ...}
        """
        elapsed = time.time() - self._timer_start
        record = LLMCallRecord(
            component=component,
            dataset=dataset or self._current_dataset,
            model=model,
            input_tokens=int(usage.get('prompt_tokens', 0)),
            output_tokens=int(usage.get('completion_tokens', 0)),
            call_type=call_type,
            wall_clock_seconds=round(elapsed, 3),
            cascade_decision=cascade_decision,
            seed=seed if seed is not None else self._current_seed,
        )
        self._records.append(record)

    def log_proxy_only(self, component: str, dataset: Optional[str] = None):
        """Log a call where the SLM proxy was used — zero LLM tokens consumed."""
        record = LLMCallRecord(
            component=component,
            dataset=dataset or self._current_dataset,
            model="slm_proxy",
            input_tokens=0,
            output_tokens=0,
            call_type="proxy",
            wall_clock_seconds=0.0,
            cascade_decision="proxy_only",
            seed=self._current_seed,
        )
        self._records.append(record)

    def summary_by_component(self) -> Dict[str, dict]:
        """Aggregate tokens and cost per component."""
        summary = {}
        for r in self._records:
            if r.component not in summary:
                summary[r.component] = {
                    'calls': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'wall_clock_seconds': 0.0,
                }
            s = summary[r.component]
            s['calls'] += 1
            s['input_tokens'] += r.input_tokens
            s['output_tokens'] += r.output_tokens
            s['total_tokens'] += r.input_tokens + r.output_tokens
            s['wall_clock_seconds'] += r.wall_clock_seconds
        return summary

    def compute_cost(self, model: Optional[str] = None) -> float:
        """Compute estimated USD cost. If model is None, uses each record's model."""
        total = 0.0
        for r in self._records:
            m = model or r.model
            if m not in MODEL_PRICING:
                continue
            input_price, output_price = MODEL_PRICING[m]
            total += (r.input_tokens / 1_000_000) * input_price
            total += (r.output_tokens / 1_000_000) * output_price
        return total

    def dump(self, path: str):
        """Save all records to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump([asdict(r) for r in self._records], f, indent=2)
        print(f"[CostLogger] Saved {len(self._records)} records to {p}")

    def reset(self):
        """Clear all records for the next experiment run."""
        self._records.clear()
        self._current_dataset = "unknown"
        self._current_seed = None

    def __len__(self):
        return len(self._records)

cost_logger = CostLogger()
