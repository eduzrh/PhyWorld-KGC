# -*- coding: utf-8 -*-
"""Probe the Stage-III judge backend (Table 7 / Table 9 / Algorithm 1).

Two uses:

* ``--backend offline``  -- run the deterministic surrogate on a fixed,
  hand-built evidence card and print the per-candidate scores.  This is the
  default and needs no credentials; it is what ``results/reference`` used.
* ``--backend openai``   -- send the *same* Table 9 prompt to an
  OpenAI-compatible endpoint, print the parsed scores and the token bill.  Use
  this to validate the key / base URL / model name before launching a Table 7
  or Table 8 sweep.

Examples
--------
    python scripts/llm_probe.py --backend offline
    python scripts/llm_probe.py --backend openai --llm-model gpt-5.4-mini
    python scripts/llm_probe.py --backend openai --dry-run   # print prompt only

Environment
-----------
``OPENAI_API_KEY`` (or ``--api-key-env``) holds the credential;
``OPENAI_BASE_URL`` / ``--api-base`` point at the endpoint.  The key is never
printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from phyworld.config import ReasonerConfig  # noqa: E402
from phyworld.rules.llm import (JUDGE_SYSTEM_PROMPT, LLMJudge,  # noqa: E402
                                build_judge_prompt)


def demo_payload():
    """A three-candidate card with one clean, one impossible and one unknown."""
    query = {
        "subject": 7, "relation": 8, "coocc": True, "time": 132.0,
        "location": "(37.5683, 126.9780)",
        "rule_evidence": [
            {"rule": 3, "quality": 0.71, "candidate": 11,
             "facts": "(37.50, 127.00) @ t=130.0"},
        ],
    }
    candidates = [
        {"candidate": 11, "fact": "(37.50, 127.00) @ t=130.0",
         "v_impl": 4.2, "v_max": 900.0},
        {"candidate": 19, "fact": "(48.85, 2.35) @ t=131.0",
         "v_impl": 7200.0, "v_max": 900.0},
        {"candidate": 23, "fact": "unknown", "v_impl": 0.0, "v_max": 900.0},
    ]
    conflicts = [(0, 1)]
    return {"query": query, "candidates": candidates, "conflicts": conflicts}


def main(argv=None) -> int:
    p = argparse.ArgumentParser("llm_probe")
    p.add_argument("--backend", default="offline", choices=["offline", "openai"])
    p.add_argument("--llm-model", default="gpt-5.4-mini")
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--cache", default=None)
    p.add_argument("--n-meta", type=int, default=2)
    p.add_argument("--dry-run", action="store_true",
                   help="print the Table 9 prompt and exit without a request")
    args = p.parse_args(argv)

    cfg = ReasonerConfig(backend=args.backend, model=args.llm_model,
                         api_base=args.api_base, api_key_env=args.api_key_env,
                         cache_path=args.cache, n_meta=args.n_meta)
    judge = LLMJudge(cfg)
    payload = demo_payload()

    prompt = build_judge_prompt(payload["query"], payload["candidates"],
                                payload["conflicts"])
    if args.dry_run:
        print(f"[system]\n{JUDGE_SYSTEM_PROMPT}\n")
        print(f"[user]\n{prompt}")
        return 0

    has_key = bool(os.environ.get(args.api_key_env))
    if args.backend == "openai" and not has_key:
        print(json.dumps({
            "backend": args.backend, "model": args.llm_model,
            "available": False,
            "reason": f"${args.api_key_env} is empty; set it or use --backend offline",
            "prompt_chars": len(prompt),
        }, indent=2))
        return 2

    n = len(payload["candidates"])
    velocity_ok = torch.tensor([True, False, True])
    v_impl = torch.tensor([4.2, 7200.0, 0.0])
    with torch.no_grad():
        scores = judge.score(payload, n_candidates=n, rounds=cfg.n_meta,
                            velocity_ok=velocity_ok, v_impl=v_impl,
                            v_max=900.0)

    out = {
        "backend": args.backend,
        "model": args.llm_model,
        "available": judge.client.available,
        "prompt_chars": len(prompt),
        "candidates": [c["candidate"] for c in payload["candidates"]],
        "scores": [round(float(x), 4) for x in scores.tolist()],
        "usage": judge.client.cost(),
        "note": ("deterministic surrogate (no external call)"
                 if not judge.client.available else "LLM response parsed"),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())