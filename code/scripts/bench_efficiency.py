# -*- coding: utf-8 -*-
"""Figure 8 harness: inference cost per test epoch and LLM tokens per query.

The paper reports, for PhyICEWS14 and PhyICEWS-0515,

  * ``Avg. time``   -- inference time in **minutes per test epoch**;
  * ``Avg. tokens`` -- average LLM tokens **per query**.

This script measures both with the released code.  It evaluates the test split
(tail direction by default, ``--direction both`` for the full two-pass epoch),
records wall-clock time, agent-call counts and token usage from the judge
client, and writes a JSON row that can be dropped into the Figure 8 table.

Examples
--------
    # timed run with a trained checkpoint
    python scripts/bench_efficiency.py --dataset PhyICEWS14 \
        --data-dir ../datasets --checkpoint ../results/phyicews14/best.pt \
        --output ../results/efficiency/phyicews14.json

    # quick estimate on a 512-query sample (no checkpoint yet)
    python scripts/bench_efficiency.py --dataset PhyICEWS14 \
        --max-queries 512 --repeat 2 --output ../results/efficiency/smoke.json

The ``minutes_per_epoch_est`` field scales the measured per-query cost to the
full test split, so a sampled run still yields the Figure 8 quantity.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from phyworld.config import ExperimentConfig  # noqa: E402
from phyworld.data.layouts import normalise_layout  # noqa: E402
from phyworld.evaluate import evaluate  # noqa: E402
from phyworld.train import build_pipeline  # noqa: E402
from phyworld.utils import ensure_dir, save_json, setup_logger  # noqa: E402


def infer_setting(dataset: str) -> str:
    return "extrapolation" if str(dataset).lower().endswith("-ext") else "interpolation"


def peak_rss_mb() -> float:
    """Resident set size high-water mark in MiB (Linux reports KiB)."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def cuda_peak_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
    return 0.0


def usage_snapshot(model) -> Dict[str, int]:
    client = getattr(getattr(model, "reasoner", None), "judge", None)
    client = getattr(client, "client", None)
    if client is not None and hasattr(client, "cost"):
        return dict(client.cost())
    return {"calls": 0, "cached_calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0}


def diff_usage(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    keys = set(after) | set(before)
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in keys}


def main(argv=None) -> None:
    p = argparse.ArgumentParser("bench_efficiency")
    p.add_argument("--dataset", default="PhyICEWS14")
    p.add_argument("--data-dir", default="../datasets")
    p.add_argument("--rules-dir", default="rules")
    p.add_argument("--layout", default="auto", choices=["auto", "native", "ids"],
                   type=normalise_layout)
    p.add_argument("--checkpoint", default=None,
                   help="best.pt from phyworld.train (omit for a timing-only run)")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-queries", type=int, default=None,
                   help="evaluate a random sample instead of the full split")
    p.add_argument("--direction", default="tail", choices=["tail", "head", "both"])
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1,
                   help="warm-up passes excluded from the timing (default 1)")
    p.add_argument("--llm-backend", default="offline", choices=["offline", "openai"])
    p.add_argument("--llm-model", default="gpt-5.4-mini")
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--llm-cache", default=None)
    p.add_argument("--call-ratio", type=float, default=1.0)
    p.add_argument("--no-reasoner", action="store_true",
                   help="time Stage I+II only (Figure 8 'lightweight' row)")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    logger = setup_logger("bench_efficiency")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ExperimentConfig()
    cfg.data.data_dir = args.data_dir
    cfg.data.dataset = args.dataset
    cfg.data.setting = infer_setting(args.dataset)
    cfg.data.layout = args.layout
    cfg.data.rules_dir = args.rules_dir
    cfg.train.device = args.device
    cfg.train.batch_size = args.batch_size
    cfg.reasoner.backend = args.llm_backend
    cfg.reasoner.model = args.llm_model
    cfg.reasoner.api_base = args.api_base
    cfg.reasoner.api_key_env = args.api_key_env
    cfg.reasoner.cache_path = args.llm_cache
    cfg.reasoner.call_ratio = args.call_ratio
    cfg.ablation.ablate_reasoner = bool(args.no_reasoner)

    pipe = build_pipeline(cfg, device=args.device, logger=logger)
    model = pipe["model"]
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state["model"])
        logger.info("loaded checkpoint %s", args.checkpoint)
    else:
        logger.warning("no --checkpoint: timing a randomly initialised model; "
                       "latency is representative, accuracy is not")

    directions: List[str] = (["tail", "head"] if args.direction == "both"
                             else [args.direction])
    n_test = len(pipe["ds"].test.facts)
    max_q = (None if args.max_queries is None
             else max(1, int(args.max_queries) // len(directions)))

    def run_once() -> Dict[str, float]:
        out = {}
        for d in directions:
            t0 = time.perf_counter()
            res = evaluate(model, pipe["ds"], pipe["index"], pipe["batcher"], "test",
                           batch_size=args.batch_size, device=args.device,
                           gamma_table=pipe["gamma_table"], history=pipe["history"],
                           ablations=cfg.ablation, use_reasoner=not args.no_reasoner,
                           max_queries=max_q, direction=d)
            out[d] = {"seconds": time.perf_counter() - t0,
                      "queries": res.num_queries,
                      "MRR": res.mrr, "H@1": res.h1,
                      "H@3": res.h3, "H@10": res.h10}
        return out

    for _ in range(max(0, args.warmup)):
        run_once()

    before = usage_snapshot(model)
    reps: List[Dict[str, float]] = []
    first_pass = None
    for _ in range(max(1, args.repeat)):
        reps.append(run_once())
        if first_pass is None:
            # Tokens and agent calls are counted on the first pass only: later
            # repeats hit the on-disk judge cache, so amortising over all
            # repeats would understate the per-query cost reported by Fig. 8.
            first_pass = usage_snapshot(model)
    used = diff_usage(first_pass, before)

    # total measured cost and the per-query normalisation
    seconds = [sum(d["seconds"] for d in r.values()) for r in reps]
    queries = [sum(int(d["queries"]) for d in r.values()) for r in reps]
    best_i = int(np.argmin(seconds))
    total_sec = float(seconds[best_i])
    total_q = int(max(queries[best_i], 1))
    sec_per_query = total_sec / total_q
    tokens = int(used.get("prompt_tokens", 0) + used.get("completion_tokens", 0))
    tokens_per_query = tokens / float(max(queries[0], 1))
    calls = int(used.get("calls", 0) + used.get("cached_calls", 0))

    result = {
        "figure": "Fig. 8",
        "dataset": args.dataset,
        "setting": cfg.data.setting,
        "device": args.device,
        "checkpoint": args.checkpoint,
        "backend": args.llm_backend,
        "llm_model": args.llm_model,
        "direction": args.direction,
        "queries_evaluated": total_q,
        "test_split_size": n_test,
        "test_epoch_queries": n_test * len(directions),
        "inference_seconds": round(total_sec, 3),
        "inference_seconds_all_repeats": [round(s, 3) for s in seconds],
        "sec_per_query": round(sec_per_query, 5),
        "minutes_per_epoch_est": round(sec_per_query * n_test * len(directions) / 60.0, 3),
        "agent_calls": calls,
        "agent_calls_per_query": round(calls / float(max(sum(queries), 1)), 4),
        "prompt_tokens": int(used.get("prompt_tokens", 0)),
        "completion_tokens": int(used.get("completion_tokens", 0)),
        "tokens_per_query": round(tokens_per_query, 2),
        "tokens_per_agent_call": round(
            tokens / float(calls) if calls else 0.0, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "cuda_peak_mb": round(cuda_peak_mb(), 1),
        "max_queries": args.max_queries,
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "per_direction": reps[best_i],
    }
    path = args.output or os.path.join(
        "results", "efficiency",
        f"{args.dataset.lower()}_{args.llm_backend}.json")
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    save_json(result, path)
    logger.info(json.dumps({k: result[k] for k in
                            ("minutes_per_epoch_est", "sec_per_query",
                             "tokens_per_query", "agent_calls")}))
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()