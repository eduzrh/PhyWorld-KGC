# -*- coding: utf-8 -*-
"""Mine spatio-temporal logic rules and cache them as ``rules/<dataset>.json``.

Pipeline (Appendix A.1, Steps 2-7):
  1. relation-adaptive walks over the training graph  -> candidate chains
  2. branch augmentation                              -> chain + branch rules
  3. quality estimation (supp / cov / conf / pca)     -> s(rule)
  4. filtering (pca >= c_min, supp >= n_min) and ranking
  5. optional LLM rewriting of the top-k chains       (Table 10)

The script mirrors ``phyworld.train``: it reads the same on-disk layout
(``--layout auto`` prefers ``datasets/ids/<dataset>``), infers the
interpolation/extrapolation setting from the ``*-Ext`` suffix, and uses the same
relation-id space, so a rule base mined here can be fed straight back with
``python -m phyworld.train --rules-dir rules``.

Large benchmarks
----------------
Quality estimation is the memory peak: it grounds ``--quality-sample`` head
facts for *every* candidate rule.  On a 7 GB CPU box the three largest
benchmarks (PhyICEWS18-Ext, PhyICEWS-0515, PhyGDELT) should be mined with

    python scripts/mine_rules.py --dataset PhyGDELT --data-dir ../datasets \
        --walks 10 --quality-sample 800

The un-scored candidate base is written to ``<dataset>.candidates.json`` first,
so a run that dies during scoring can be resumed with ``--relabel`` instead of
re-mining.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from phyworld.config import ExperimentConfig  # noqa: E402
from phyworld.data.layouts import normalise_layout  # noqa: E402
from phyworld.data.stkg import STKGIndex, load_stkg  # noqa: E402
from phyworld.rules.base import RuleBase  # noqa: E402
from phyworld.rules.grounding import CSRIndex, RuleGrounder  # noqa: E402
from phyworld.rules.llm import LLMClient, RuleInducer  # noqa: E402
from phyworld.rules.mining import SpatioTemporalRuleMiner  # noqa: E402
from phyworld.rules.quality import (estimate_rule_quality, filter_and_rank,  # noqa: E402
                                    quality_summary)
from phyworld.utils import ensure_dir, save_json, setup_logger  # noqa: E402


def infer_setting(dataset: str) -> str:
    """``PhyICEWS14-Ext`` / ``PhyICEWS18-Ext`` are extrapolation benchmarks."""
    return "extrapolation" if str(dataset).lower().endswith("-ext") else "interpolation"


def main(argv=None) -> None:
    p = argparse.ArgumentParser("mine_rules")
    p.add_argument("--dataset", default="PhyICEWS14")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--layout", default="auto", choices=["auto", "native", "ids"],
                   type=normalise_layout)
    p.add_argument("--out-dir", default="rules")
    p.add_argument("--max-length", type=int, default=3)
    p.add_argument("--walks", type=int, default=20)
    p.add_argument("--fact-sample", type=int, default=20000,
                   help="max training facts used for walk generation")
    p.add_argument("--min-support", type=int, default=5)
    p.add_argument("--pca-threshold", type=float, default=0.1)
    p.add_argument("--max-per-relation", type=int, default=40)
    p.add_argument("--quality-sample", type=int, default=800,
                   help="head facts sampled per candidate rule (800 fits 7 GB)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--llm-topk", type=int, default=20)
    p.add_argument("--llm-model", default="gpt-5.4-mini")
    p.add_argument("--relabel", action="store_true",
                   help="only recompute quality/thresholds of an existing rule base")
    args = p.parse_args(argv)

    logger = setup_logger("mine_rules")
    t0 = time.time()
    setting = infer_setting(args.dataset)
    ds = load_stkg(args.data_dir, args.dataset, setting=setting,
                   layout=args.layout)
    index = STKGIndex(ds)
    csr = CSRIndex(ds.train.facts, ds.train.loc_latlon, ds.num_entities,
                   num_relations_base=ds.num_relations_base)
    grounder = RuleGrounder(csr, setting=ds.setting, loc_fn=index.query_location)

    out_dir = ensure_dir(args.out_dir)
    path = os.path.join(out_dir, f"{args.dataset.lower()}.json")
    cand_path = os.path.join(out_dir, f"{args.dataset.lower()}.candidates.json")

    if args.relabel:
        # Always prefer the un-scored candidate base: re-scoring a filtered file
        # can only re-rank what survived the previous threshold.
        src = cand_path if os.path.exists(cand_path) else path
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"--relabel needs {path} or {cand_path}; mine once without it")
        rb = RuleBase.load(src)
        logger.info("re-scoring %d existing rules from %s", len(rb.rules), src)
    else:
        miner = SpatioTemporalRuleMiner(
            ds, index, max_length=args.max_length, walks_per_fact=args.walks,
            seed=args.seed, fact_sample=args.fact_sample,
        )
        rb = miner.mine()
        logger.info("mined %d candidate rules in %.1fs", len(rb.rules), time.time() - t0)

        if args.use_llm:
            client = LLMClient(backend="openai", model=args.llm_model)
            if client.available:
                inducer = RuleInducer(client, [str(i) for i in range(ds.num_relations)],
                                      topk=args.llm_topk)
                extra = []
                for r in range(ds.num_relations_base):
                    inst = [x for x in rb.for_relation(r)][:args.llm_topk]
                    if inst:
                        extra.extend(inducer.induce(r, inst))
                rb = RuleBase(rb.rules + extra)
                logger.info("LLM added %d rules", len(extra))
            else:
                logger.warning("LLM backend unavailable (no OPENAI_API_KEY); "
                               "skipping the supplementary rule step")
        rb.save(cand_path)
        logger.info("candidate rule base -> %s", cand_path)

    scored = [estimate_rule_quality(r, ds, grounder, sample_size=args.quality_sample,
                                    seed=args.seed)
              for r in rb.rules]
    logger.info("scored %d rules in %.1fs", len(scored), time.time() - t0)

    kept = filter_and_rank(scored, min_support=args.min_support,
                           min_pca=args.pca_threshold,
                           max_per_relation=args.max_per_relation)
    final = RuleBase(kept)
    final.save(path)
    stats = quality_summary(final)
    save_json({"dataset": args.dataset, "setting": setting, "stats": stats,
               "config": vars(args)}, os.path.join(out_dir, "summary.json"))
    logger.info("kept %d rules -> %s", len(kept), path)
    logger.info("stats: %s", stats)


if __name__ == "__main__":
    main()