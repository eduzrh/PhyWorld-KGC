#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-train static KG embeddings for the ``E_fix`` initialisation (Section 3.2).

The paper initialises the entity / relation embeddings "from static KG
embeddings pre-trained on the training graph".  This script performs that step
on any of the six benchmarks and writes a checkpoint that
``phyworld.train --init-emb`` consumes directly.

Example
-------
    python scripts/pretrain_embeddings.py --dataset PhyICEWS14 \
        --data-dir ../datasets --epochs 20 --dim 200 \
        --out checkpoints/static_kg_init_dim200.pt

    python -m phyworld.train --dataset PhyICEWS14 --data-dir ../datasets \
        --init-emb checkpoints/static_kg_init_dim200.pt

Only the training split is read, so the initialisation never sees validation or
test facts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from phyworld.data.layouts import normalise_layout  # noqa: E402
from phyworld.data.stkg import load_stkg  # noqa: E402
from phyworld.pretrain import pretrain_static_embeddings  # noqa: E402
from phyworld.utils import ensure_dir, setup_logger  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser("pretrain_embeddings")
    p.add_argument("--dataset", default="PhyICEWS14")
    p.add_argument("--data-dir", default="../datasets")
    p.add_argument("--layout", default="auto", choices=["auto", "native", "ids"],
                   type=normalise_layout)
    p.add_argument("--setting", default="auto",
                   choices=["auto", "interpolation", "extrapolation"])
    p.add_argument("--dim", type=int, default=200, help="embedding width")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--num-neighbours", type=int, default=20,
                   help="neighbours sampled for the aggregation")
    p.add_argument("--neg-ratio", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto: cuda when available")
    p.add_argument("--max-train-facts", type=int, default=None,
                   help="subsample the training split (smoke tests only)")
    p.add_argument("--out", default=None,
                   help="checkpoint path (default: checkpoints/static_kg_init_dim<dim>.pt)")
    args = p.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    setting = args.setting
    if setting == "auto":
        setting = "extrapolation" if args.dataset.endswith("-Ext") else "interpolation"

    ds = load_stkg(args.data_dir, args.dataset, setting=setting, layout=args.layout)
    facts = ds.train.facts
    if args.max_train_facts is not None and facts.shape[0] > args.max_train_facts:
        rs = __import__("numpy").random.RandomState(args.seed)
        sel = rs.choice(facts.shape[0], size=int(args.max_train_facts), replace=False)
        facts = facts[sel]

    out = args.out or os.path.join("checkpoints", f"static_kg_init_dim{args.dim}.pt")
    ensure_dir(os.path.dirname(out) or ".")
    logger = setup_logger("pretrain", log_file=os.path.join(
        os.path.dirname(out) or ".", f"pretrain_{args.dataset}.log"))

    logger.info("dataset=%s entities=%d relations=%d setting=%s layout=%s device=%s",
                args.dataset, ds.num_entities, ds.num_relations_base, setting,
                args.layout, device)
    logger.info("training facts=%d dim=%d epochs=%d", facts.shape[0], args.dim, args.epochs)

    t0 = time.time()
    payload = pretrain_static_embeddings(
        facts, num_entities=ds.num_entities,
        num_relations_base=ds.num_relations_base,
        embed_dim=args.dim, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        num_neighbours=args.num_neighbours, neg_ratio=args.neg_ratio,
        seed=args.seed, device=device, logger=logger,
    )
    payload["meta"].update({
        "dataset": args.dataset,
        "setting": setting,
        "layout": args.layout,
        "train_facts": int(facts.shape[0]),
        "device": device,
    })
    torch.save(payload, out)
    logger.info("saved %s  (entity_init %s, relation_init %s)  in %.1fs",
                out, tuple(payload["entity_init"].shape),
                tuple(payload["relation_init"].shape), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())