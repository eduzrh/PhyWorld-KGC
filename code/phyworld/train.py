# -*- coding: utf-8 -*-
"""Training / evaluation driver for PhyWorld-KGC."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import AblationConfig, ExperimentConfig
from .data.batch import QueryBatch, QueryBatcher
from .data.layouts import normalise_layout
from .data.stkg import STKGIndex, compute_relation_scales, load_stkg
from .data.text import TextLocationEncoder
from .evaluate import EvalResult, evaluate, evaluate_both
from .models.astsc import GateTable
from .models.losses import PhyWorldLoss
from .models.phyworld import PhyWorldKGC
from .rules.base import RuleBase
from .rules.cooccurrence import annotate_cooccurrence
from .rules.grounding import CSRIndex, RuleGrounder
from .sampling.rtns_st import SpatioTemporalNeighborSampler
from .utils import ensure_dir, save_json, set_seed, setup_logger


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def ring_centres(cfg: ExperimentConfig) -> np.ndarray:
    """Representative distance d_bar_varsigma of each spatial ring (km)."""
    res = float(cfg.data.spatial_resolution_km)
    return np.asarray([(i + 0.5) * res for i in range(int(cfg.data.spatial_rings))],
                      dtype=np.float32)


def load_rules(cfg: ExperimentConfig, ds, logger=None) -> RuleBase:
    path = os.path.join(cfg.data.rules_dir, f"{cfg.data.dataset.lower()}.json")
    path = os.path.abspath(path)
    if os.path.exists(path):
        rb = RuleBase.load(path)
        if logger:
            logger.info("loaded %d rules from %s", len(rb.rules), path)
        return rb
    if logger:
        logger.warning("no rule file at %s -> empty rule base "
                       "(run scripts/mine_rules.py first)", path)
    return RuleBase([])


def build_pipeline(cfg: ExperimentConfig, device: str = "cpu", logger=None):
    """Data + index + sampler + rule engine + model, fully assembled."""
    set_seed(cfg.train.seed)
    ds = load_stkg(cfg.data.data_dir, cfg.data.dataset, setting=cfg.data.setting,
                   layout=cfg.data.layout)
    index = STKGIndex(ds, augment_inverse=True)
    rho, mu = compute_relation_scales(ds)
    alpha = 1.0 / np.clip(mu, 1e-3, None)
    zeta = 1.0 / np.clip(rho, 1e-3, None)

    sampler = SpatioTemporalNeighborSampler(
        ds, index, k=cfg.data.num_neighbors, seed=cfg.train.seed,
        time_mode="auto", ablate_s_sample=cfg.ablation.ablate_s_sample,
    )

    # ---- rules + grounding ----
    rule_base = load_rules(cfg, ds, logger)
    # Figure 7 rule-side ablations: they act on the cached rule base, so the
    # mining cost is paid once and every variant starts from the same Omega.
    if cfg.ablation.ablate_branch or cfg.ablation.ablate_llm_rule:
        rule_base = rule_base.filter(
            use_branch=not cfg.ablation.ablate_branch,
            use_llm=not cfg.ablation.ablate_llm_rule)
        if logger:
            logger.info("[ablation] rule base filtered -> %d rules", len(rule_base))
    if cfg.ablation.ablate_conf:
        for _r in rule_base.rules:
            _r.quality = float(_r.confidence)      # s(rule) = conf
        if logger:
            logger.info("[ablation] rule quality set to confidence only")
    if len(rule_base.rules) == 0 and logger:
        logger.warning("rule base is empty: Gamma == eps and the rule term is 0")
    csr = CSRIndex(ds.train.facts, ds.train.loc_latlon, ds.num_entities,
                   num_relations_base=ds.num_relations_base)
    grounder = RuleGrounder(csr, setting=cfg.data.setting,
                            ablate_s_const=cfg.ablation.ablate_s_const,
                            rule_base=rule_base, loc_fn=index.query_location)

    cooc_rels, _ = annotate_cooccurrence(
        ds, d_co=cfg.rules.coocc_distance_km, pi_co=cfg.rules.coocc_fraction)
    if cfg.ablation.ablate_coocc:
        cooc_rels = list(range(ds.num_relations_base))

    # ---- text-location table ----
    text_dim = 768
    text_matrix = None
    if len(index.text_vocab) > 0:
        enc = TextLocationEncoder(backend=cfg.data.text_encoder)
        text_matrix = enc.encode(index.text_vocab)
        text_dim = int(text_matrix.shape[1])

    # ---- model ----
    model = PhyWorldKGC(ds.num_entities, ds.num_relations, cfg,
                        num_texts=len(index.text_vocab), text_dim=text_dim,
                        ring_distance=ring_centres(cfg))
    # decay rates are defined per base relation; inverse relations share them
    model.astsc.register_decay_priors(np.concatenate([alpha, alpha]),
                                      np.concatenate([zeta, zeta]))
    if text_matrix is not None:
        with torch.no_grad():
            model.astsc.encoder.text_table.weight.copy_(
                torch.as_tensor(text_matrix, dtype=torch.float32))

    if cfg.model.init_emb:
        model.load_pretrained(cfg.model.init_emb, verbose=logger is not None)

    batcher = QueryBatcher(ds, index, sampler, fact_text_id=index.fact_text_id,
                           cooccurring=cooc_rels, grounder=grounder)
    gamma_table = GateTable(
        rule_base, ds.num_relations, cfg.data.spatial_rings + 1,
        cfg.data.time_buckets(), int(cfg.data.window_past // cfg.data.bin_width),
        cfg.data.bin_width, ring_centres(cfg), floor=cfg.model.gate_floor,
        ablate_rule_gate=cfg.ablation.ablate_rule_gate,
        ablate_s_const=cfg.ablation.ablate_s_const,
    )
    history = index.entity_history()
    model.to(device)
    return dict(ds=ds, index=index, sampler=sampler, rule_base=rule_base,
                grounder=grounder, model=model, batcher=batcher,
                gamma_table=gamma_table, history=history,
                alpha=alpha, zeta=zeta)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def rule_scores_for(grounder, queries, enabled: bool = True):
    return grounder.ground_batch(queries) if enabled else None


def train(cfg: ExperimentConfig, device: Optional[str] = None,
          logger=None) -> Dict[str, float]:
    logger = logger or setup_logger("phyworld")
    device = device or (cfg.train.device if torch.cuda.is_available() else "cpu")
    pipe = build_pipeline(cfg, device=device, logger=logger)
    ds, index, batcher = pipe["ds"], pipe["index"], pipe["batcher"]
    model, gamma_table, history = pipe["model"], pipe["gamma_table"], pipe["history"]

    criterion = PhyWorldLoss(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.train.epochs))

    facts = ds.train.facts
    if cfg.train.max_train_facts and len(facts) > cfg.train.max_train_facts:
        rng0 = np.random.RandomState(cfg.train.seed)
        facts = facts[np.sort(rng0.choice(len(facts), cfg.train.max_train_facts,
                                          replace=False))]
    rng = np.random.RandomState(cfg.train.seed)
    best = {"MRR": -1.0}
    wait = 0
    out_dir = ensure_dir(os.path.join(cfg.train.log_dir, cfg.train.run_name))
    metrics_path = os.path.join(out_dir, "metrics.json")
    history_log = []

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        order = rng.permutation(len(facts))
        t0 = time.time()
        total, n_batches = 0.0, 0
        for start in range(0, len(order), cfg.train.batch_size):
            sel = order[start:start + cfg.train.batch_size]
            rows = facts[sel]
            queries = [(int(s), int(r), int(t)) for s, r, o, t in rows]
            gold = [int(o) for s, r, o, t in rows]
            batch = batcher.build(queries, gold=gold, setting="train").to(device)
            rs = rule_scores_for(pipe["grounder"], queries,
                                 enabled=not cfg.ablation.ablate_rule_term)

            out = model(batch, gamma_table=gamma_table, rule_scores=rs,
                        history=history, ablations=cfg.ablation, use_reasoner=False)
            loss = criterion(
                out.scores, batch.gold, wm_cell=out.wm_cell,
                g_hat=out.sim.g_hat, ent_emb=model.ent_emb,
                h_l=out.state.fact_hl, loc=out.state.fact_loc,
                loc_valid=out.state.fact_loc_valid,
                emb_parts=[model.astsc.encoder.ent_emb.weight,
                           model.astsc.encoder.rel_emb.weight],
                h_reg=[out.h_sq, model.rel_emb(batch.rel.to(out.h_sq.device))],
                ablate_wm=cfg.ablation.ablate_st_conv, ablate_geo=cfg.ablation.ablate_l_geo,
            )
            opt.zero_grad()
            loss.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            total += float(loss.total)
            n_batches += 1
        sched.step()

        log = {"epoch": epoch, "loss": total / max(1, n_batches),
               "sec": round(time.time() - t0, 1)}
        if epoch % cfg.train.eval_every == 0:
            res = evaluate(model, ds, index, batcher, "valid", batch_size=64,
                           device=device, gamma_table=gamma_table, history=history,
                           ablations=cfg.ablation, use_reasoner=True)
            log.update({f"valid_{k}": v for k, v in res.as_dict().items()})
            if res.mrr > best["MRR"]:
                best = {"MRR": res.mrr, "epoch": epoch}
                wait = 0
                torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()},
                           os.path.join(out_dir, "best.pt"))
            else:
                wait += 1
        if logger:
            logger.info(json.dumps(log))
        history_log.append(log)
        save_json({"history": history_log, "best": best}, metrics_path)
        if wait >= cfg.train.patience:
            if logger:
                logger.info("early stop at epoch %d", epoch)
            break
    return best# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.data_dir = args.data_dir
    cfg.data.dataset = args.dataset
    setting = args.setting
    if setting == "auto":
        # PhyICEWS14-Ext / PhyICEWS18-Ext are the extrapolation benchmarks
        setting = ("extrapolation" if str(args.dataset).lower().endswith("-ext")
                   else "interpolation")
    cfg.data.setting = setting
    cfg.data.layout = getattr(args, "layout", "auto")
    cfg.data.num_neighbors = args.num_neighbors
    cfg.data.window_past = args.window
    cfg.data.bin_width = args.bin
    cfg.data.spatial_rings = args.spatial_rings
    cfg.data.spatial_resolution_km = args.spatial_resolution
    cfg.data.text_encoder = args.text_encoder
    cfg.model.init_emb = getattr(args, "init_emb", None)
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.device = args.device
    cfg.train.run_name = args.run_name or f"{args.dataset.lower()}"
    cfg.train.seed = args.seed
    cfg.train.max_train_facts = args.max_train_facts
    cfg.reasoner.entropy_threshold = args.entropy_threshold
    cfg.reasoner.fusion_weight = args.fusion_weight
    cfg.reasoner.call_ratio = getattr(args, "call_ratio", 1.0)
    cfg.reasoner.backend = args.llm_backend
    # Table 7: the judge backbone is a first-class knob.  ``api_base`` points at
    # any OpenAI-compatible endpoint (Azure/vLLM/Ollama proxy, ...), while
    # ``api_key_env`` keeps the credential out of the command line.
    cfg.reasoner.model = getattr(args, "llm_model", cfg.reasoner.model)
    cfg.reasoner.api_base = getattr(args, "api_base", None)
    cfg.reasoner.api_key_env = getattr(args, "api_key_env", "OPENAI_API_KEY")
    cfg.reasoner.cache_path = getattr(args, "llm_cache", None)
    cfg.rules.pca_threshold = args.pca_threshold
    cfg.ablation = AblationConfig.from_names(args.ablation or [])
    if args.ablation:
        cfg.train.run_name += "_" + args.ablation[0].replace(" ", "").replace("/", "")
        cfg.train.run_name = cfg.train.run_name.replace(".", "")
    return cfg


def main(argv=None) -> None:
    import argparse

    from .config import ABLATIONS_ALL

    p = argparse.ArgumentParser("phyworld")
    p.add_argument("--dataset", default="PhyICEWS14")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--setting", default="auto",
                   choices=["auto", "interpolation", "extrapolation"],
                   help="auto: extrapolation for *-Ext benchmarks")
    p.add_argument("--layout", default="auto",
                   choices=["auto", "native", "ids"], type=normalise_layout,
                   help="auto prefers datasets/ids/<dataset> when present")
    p.add_argument("--rules-dir", default=None, help="cached rule base directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto",
                   help="auto: cuda when available, otherwise cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-neighbors", type=int, default=50)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--bin", type=int, default=1)
    p.add_argument("--spatial-rings", type=int, default=4)
    p.add_argument("--spatial-resolution", type=float, default=200.0)
    p.add_argument("--text-encoder", default="auto", choices=["auto", "bert", "hash"])
    p.add_argument("--init-emb", default=None,
                   help="static KG embeddings produced by scripts/pretrain_embeddings.py")
    p.add_argument("--entropy-threshold", type=float, default=0.8)
    p.add_argument("--fusion-weight", type=float, default=0.8)
    p.add_argument("--pca-threshold", type=float, default=0.1)
    p.add_argument("--llm-backend", default="offline", choices=["offline", "openai"])
    p.add_argument("--llm-model", default="gpt-5.4-mini",
                   help="Table 7: judge backbone (gpt-5.4-mini, gpt-4.1-mini, "
                        "DeepSeek-R1, DeepSeek-V3, Llama3-70B, Llama3-8B, ...)")
    p.add_argument("--api-base", default=None,
                   help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="environment variable holding the API key")
    p.add_argument("--llm-cache", default=None,
                   help="JSON file caching judge responses (Table 8 / Fig. 8 reruns)")
    p.add_argument("--call-ratio", type=float, default=1.0,
                   help="Table 8: fraction of triggered Stage-III agent calls")
    p.add_argument("--ablation", nargs="*", choices=ABLATIONS_ALL, default=None,
                   help="paper-facing ablation names, e.g. --ablation \"w/o Gate\"")
    p.add_argument("--run-name", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max-train-facts", type=int, default=None)
    p.add_argument("--output", default=None, help="json file for test metrics")
    args = p.parse_args(argv)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_config(args)
    if args.rules_dir:
        cfg.data.rules_dir = args.rules_dir
    logger = setup_logger("phyworld")
    if args.eval_only:
        pipe = build_pipeline(cfg, device=args.device, logger=logger)
        model = pipe["model"]
        ckpt = args.checkpoint or os.path.join(
            cfg.train.log_dir, cfg.train.run_name, "best.pt")
        state = torch.load(ckpt, map_location=args.device)
        model.load_state_dict(state["model"])
        t_eval = time.time()
        res = evaluate_both(model, pipe["ds"], pipe["index"], pipe["batcher"],
                            "test", batch_size=64, device=args.device,
                            gamma_table=pipe["gamma_table"], history=pipe["history"],
                            ablations=cfg.ablation, use_reasoner=True)
        # Table 7/8 accounting: how many agent calls Stage III actually issued.
        judge = model.reasoner.judge
        usage = (judge.client.cost() if hasattr(judge.client, "cost")
                 else {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        n_q = 2 * len(pipe["ds"].test.facts)
        out = {"dataset": cfg.data.dataset, "setting": cfg.data.setting,
               "tail": res["tail"].as_dict(), "head": res["head"].as_dict(),
               "avg": {k: (res["tail"].as_dict()[k] + res["head"].as_dict()[k]) / 2
                       for k in ("MRR", "H@1", "H@3", "H@10")},
               "stage3": {"meta_calls": int(model.reasoner.meta_eval.calls),
                          "call_ratio": float(cfg.reasoner.call_ratio),
                          "backend": cfg.reasoner.backend,
                          "model": cfg.reasoner.model,
                          "judge_usage": usage,
                          "tokens_per_query": round(
                              (usage.get("prompt_tokens", 0)
                               + usage.get("completion_tokens", 0)) / max(n_q, 1), 2),
                          "eval_seconds": round(time.time() - t_eval, 1)},
               "config": cfg.to_dict()}
        path = args.output or os.path.join(
            cfg.train.log_dir, cfg.train.run_name, "test.json")
        save_json(out, path)
        logger.info(json.dumps(out["avg"]))
        return

    train(cfg, device=args.device, logger=logger)


if __name__ == "__main__":
    main()