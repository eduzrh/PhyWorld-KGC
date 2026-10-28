# -*- coding: utf-8 -*-
"""Driver for the 19 ablation variants of Figure 7.

Usage
-----
    python -m phyworld.ablations --dataset PhyICEWS14 --group A
    python -m phyworld.ablations --dataset PhyICEWS14 --names "w/o Gate" "w/o Evol."

Each run keeps the *paper* label in the result file, e.g.
``results/phyicews14/w_o_Gate/test.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import List, Optional

import torch

from .config import (ABLATION_GROUP_A, ABLATION_GROUP_B1, ABLATION_GROUP_B2,
                     ABLATION_GROUP_C, ABLATIONS_ALL, AblationConfig,
                     ExperimentConfig)
from .data.layouts import normalise_layout
from .evaluate import evaluate_both
from .train import build_pipeline, train
from .utils import ensure_dir, save_json, setup_logger

GROUPS = {"A": ABLATION_GROUP_A, "B1": ABLATION_GROUP_B1,
          "B2": ABLATION_GROUP_B2, "C": ABLATION_GROUP_C,
          "all": ABLATIONS_ALL}


def slug(name: str) -> str:
    return (name.replace("w/o ", "without_").replace(" ", "_")
            .replace(".", "").replace("/", "_").lower())


def run_one(cfg: ExperimentConfig, name: str, logger) -> dict:
    run_cfg = copy.deepcopy(cfg)
    run_cfg.ablation = AblationConfig.from_names([name])
    run_cfg.train.run_name = f"{cfg.train.run_name}_{slug(name)}"
    logger.info("=== ablation %s -> %s ===", name, run_cfg.train.run_name)
    best = train(run_cfg, device=cfg.train.device, logger=logger)

    pipe = build_pipeline(run_cfg, device=cfg.train.device, logger=logger)
    ckpt = os.path.join(run_cfg.train.log_dir, run_cfg.train.run_name, "best.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=cfg.train.device)
        pipe["model"].load_state_dict(state["model"])
    res = evaluate_both(pipe["model"], pipe["ds"], pipe["index"], pipe["batcher"],
                        "test", batch_size=64, device=cfg.train.device,
                        gamma_table=pipe["gamma_table"], history=pipe["history"],
                        ablations=run_cfg.ablation)
    out = {"ablation": name, "valid_best_mrr": best.get("MRR"),
           "tail": res["tail"].as_dict(), "head": res["head"].as_dict()}
    path = os.path.join(run_cfg.train.log_dir, run_cfg.train.run_name, "test.json")
    save_json(out, path)
    logger.info("ablation %s: tail MRR=%.4f head MRR=%.4f",
                name, res["tail"].mrr, res["head"].mrr)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser("phyworld.ablations")
    p.add_argument("--dataset", default="PhyICEWS14")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--rules-dir", default=None)
    p.add_argument("--group", default="all", choices=list(GROUPS))
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--layout", default="auto", choices=["auto", "native", "ids"],
                   type=normalise_layout)
    p.add_argument("--seed", type=int, default=42)
    # Stage III knobs (Table 7 backbone, Table 8 call budget)
    p.add_argument("--llm-backend", default="offline", choices=["offline", "openai"])
    p.add_argument("--llm-model", default="gpt-5.4-mini")
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--llm-cache", default=None)
    p.add_argument("--call-ratio", type=float, default=1.0)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    names: List[str] = args.names or GROUPS[args.group]
    cfg = ExperimentConfig()
    cfg.data.data_dir = args.data_dir
    cfg.data.dataset = args.dataset
    cfg.data.setting = ("extrapolation" if args.dataset.lower().endswith("-ext")
                        else "interpolation")
    cfg.data.layout = args.layout
    if args.rules_dir:
        cfg.data.rules_dir = args.rules_dir
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.device = args.device
    cfg.train.seed = args.seed
    cfg.train.run_name = args.dataset.lower()
    cfg.reasoner.backend = args.llm_backend
    cfg.reasoner.model = args.llm_model
    cfg.reasoner.api_base = args.api_base
    cfg.reasoner.api_key_env = args.api_key_env
    cfg.reasoner.cache_path = args.llm_cache
    cfg.reasoner.call_ratio = args.call_ratio

    logger = setup_logger("phyworld.ablations")
    results = [run_one(cfg, n, logger) for n in names]
    summary = {r["ablation"]: r["tail"]["MRR"] for r in results}
    path = args.output or os.path.join(cfg.train.log_dir, cfg.train.run_name,
                                       "ablations.json")
    ensure_dir(os.path.dirname(path))
    save_json({"dataset": args.dataset, "results": results, "summary": summary}, path)
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()