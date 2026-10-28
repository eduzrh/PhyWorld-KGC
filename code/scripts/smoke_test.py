# -*- coding: utf-8 -*-
"""Smoke test: synthetic spatio-temporal KG -> full Stage I/II/III forward pass.

Run:  python scripts/smoke_test.py
It builds a tiny Phy* style dataset in a temp folder (train/valid/test.txt plus
locations/*.txt), mines a handful of rules, then executes one training step and
one evaluation query through the complete pipeline.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phyworld.config import AblationConfig, ExperimentConfig          # noqa: E402
from phyworld.evaluate import evaluate                                 # noqa: E402
from phyworld.models.losses import PhyWorldLoss                        # noqa: E402
from phyworld.train import build_pipeline, ring_centres                # noqa: E402


def make_dataset(root: str, n_ent: int = 60, n_rel: int = 4, n_facts: int = 400,
                 seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    ds_dir = os.path.join(root, "ids", "smoke")
    os.makedirs(os.path.join(ds_dir, "locations"), exist_ok=True)

    base_lat, base_lon = 35.0, 110.0
    ents_lat = base_lat + rng.uniform(-5, 5, size=n_ent)
    ents_lon = base_lon + rng.uniform(-5, 5, size=n_ent)

    rows = []
    for _ in range(n_facts):
        s = rng.randint(n_ent)
        o = rng.randint(n_ent)
        r = rng.randint(n_rel)
        t = rng.randint(60)
        rows.append((s, r, o, t))
    rows = np.asarray(sorted(set(rows)))
    rng.shuffle(rows)
    n = len(rows)
    cut1, cut2 = int(n * 0.7), int(n * 0.85)

    def dump(name, part):
        with open(os.path.join(ds_dir, f"{name}.txt"), "w") as f:
            for s, r, o, t in part:
                f.write(f"{s}\t{r}\t{o}\t{t}\n")
        with open(os.path.join(ds_dir, "locations", f"{name}.txt"), "w") as f:
            for s, r, o, t in part:
                f.write(f"{ents_lat[o]:.4f},{ents_lon[o]:.4f}\n")

    dump("train", rows[:cut1])
    dump("valid", rows[cut1:cut2])
    dump("test", rows[cut2:])


def main() -> None:
    root = tempfile.mkdtemp(prefix="phyworld_smoke_")
    try:
        make_dataset(root)
        cfg = ExperimentConfig()
        cfg.data.data_dir = root
        cfg.data.dataset = "smoke"
        cfg.data.num_neighbors = 8
        cfg.data.window_past = 20
        cfg.data.spatial_rings = 3
        cfg.train.batch_size = 4
        cfg.train.epochs = 1
        cfg.train.device = "cpu"
        cfg.reasoner.top_k = 5
        cfg.reasoner.backend = "offline"
        pipe = build_pipeline(cfg, device="cpu")

        model, ds = pipe["model"], pipe["ds"]
        batcher, gamma_table = pipe["batcher"], pipe["gamma_table"]
        facts = ds.train.facts[:4]
        queries = [(int(s), int(r), int(t)) for s, r, o, t in facts]
        gold = [int(o) for s, r, o, t in facts]
        batch = batcher.build(queries, gold=gold, setting="train")
        rs = pipe["grounder"].ground_batch(queries)

        out = model(batch, gamma_table=gamma_table, rule_scores=rs,
                    history=pipe["history"], ablations=cfg.ablation,
                    use_reasoner=True)
        assert out.scores.shape == (4, ds.num_entities), out.scores.shape
        assert torch_isfinite(out.scores), "non-finite scores"

        crit = PhyWorldLoss(cfg.model)
        loss = crit(out.scores, batch.gold, wm_cell=out.wm_cell, ent_emb=model.ent_emb,
                    h_l=out.state.fact_hl, loc=out.state.fact_loc,
                    loc_valid=out.state.fact_loc_valid)
        loss.total.backward()
        print(f"[ok] forward+backward  loss={float(loss.total):.4f} "
              f"(kgc={float(loss.l_kgc):.4f} wm={float(loss.l_wm):.4f} "
              f"geo={float(loss.l_geo):.4f})")

        res = evaluate(model, ds, pipe["index"], batcher, "test", batch_size=4,
                       device="cpu", gamma_table=gamma_table,
                       history=pipe["history"], ablations=cfg.ablation)
        print(f"[ok] evaluate  MRR={res.mrr:.3f} H@1={res.h1:.3f} n={res.num_queries}")

        # every ablation must run end-to-end
        from phyworld.config import ABLATIONS_ALL
        for name in ABLATIONS_ALL:
            abl = AblationConfig.from_names([name])
            o = model(batch, gamma_table=gamma_table, rule_scores=rs,
                      history=pipe["history"], ablations=abl, use_reasoner=True)
            assert o.scores.shape == (4, ds.num_entities)
            assert torch_isfinite(o.scores), name
        print(f"[ok] all {len(ABLATIONS_ALL)} ablations executed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def torch_isfinite(x) -> bool:
    import torch
    return bool(torch.isfinite(x).all())


if __name__ == "__main__":
    main()