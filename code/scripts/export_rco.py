# -*- coding: utf-8 -*-
"""Export the co-occurrence relation set R_co of Section 3.3 to ``rules/``.

``rules/<dataset>.json`` holds the rule base Omega; this script materialises the
companion file ``rules/<dataset>_rco.json`` holding R_co (the relations whose
subject and object sit at the same place when the fact occurs), together with
the per-relation statistics used to annotate it.

Usage
-----
    python scripts/export_rco.py --dataset PhyICEWS14 --data-dir ../data
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phyworld.config import ExperimentConfig                     # noqa: E402
from phyworld.data.layouts import normalise_layout               # noqa: E402
from phyworld.data.stkg import load_stkg                         # noqa: E402
from phyworld.rules.cooccurrence import annotate_cooccurrence    # noqa: E402
from phyworld.utils import save_json                             # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser("export_rco")
    p.add_argument("--dataset", default="PhyICEWS14")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--layout", default="auto", choices=["auto", "native", "ids"],
                   type=normalise_layout)
    p.add_argument("--d-co", type=float, default=None,
                   help="distance threshold in km (default: rules.d_co)")
    p.add_argument("--pi-co", type=float, default=None,
                   help="required fraction (default: rules.pi_co)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = ExperimentConfig()
    d_co = float(args.d_co if args.d_co is not None else cfg.rules.coocc_distance_km)
    pi_co = float(args.pi_co if args.pi_co is not None else cfg.rules.coocc_fraction)

    ds = load_stkg(args.data_dir, args.dataset, setting=cfg.data.setting,
                   layout=args.layout)
    r_co, stats = annotate_cooccurrence(ds, d_co=d_co, pi_co=pi_co)

    names = None
    rel_file = os.path.join(args.data_dir, "ids", args.dataset.lower(),
                            "relations.txt")
    if os.path.exists(rel_file):
        with open(rel_file, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
    if names is None and getattr(ds, "relation_names", None) is not None:
        names = [str(x) for x in ds.relation_names[: ds.num_relations_base]]

    payload = {
        "dataset": args.dataset,
        "factorising": "co-occurrence type (Section 3.3)",
        "d_co": d_co,
        "pi_co": pi_co,
        "num_relations_base": int(ds.num_relations_base),
        "R_co": [int(r) for r in r_co],
        "R_co_names": ([names[r] for r in r_co] if names is not None else None),
        "stats": {str(k): v for k, v in stats.items()},
    }
    out = args.out or os.path.join(
        cfg.data.rules_dir, f"{args.dataset.lower()}_rco.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_json(payload, out)
    print(f"[rco] {len(r_co)}/{ds.num_relations_base} relations -> {out}")


if __name__ == "__main__":
    main()