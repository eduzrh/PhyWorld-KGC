# -*- coding: utf-8 -*-
"""LLM interfaces: (i) supplementary rule generation (Table 10) and
(ii) the spatio-temporal constraint judge (Table 9).

Two backends are provided:

* ``openai``  -- any OpenAI-compatible chat-completions endpoint.
* ``offline`` -- a deterministic surrogate that implements the *same* symbolic
  checks the prompt asks the LLM to perform (velocity bound, geographic
  exclusion, activity region, rule-grounding support).  It lets the whole
  pipeline run without an external service; every judgement is logged with
  ``backend="offline"`` so results are never mistaken for LLM output.

Both share one ``LLMClient``; responses can be cached on disk
(``--llm-cache``) so that a repeated evaluation -- e.g. the Table 8 sweep or a
second pass of Figure 8 -- does not re-bill the same prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .base import Atom, Rule


# ---------------------------------------------------------------------------
# low level client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, backend: str = "offline", model: str = "gpt-5.4-mini",
                 api_base: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.0, max_retries: int = 2, timeout: int = 60,
                 cache_path: Optional[str] = None):
        self.backend = (backend or "offline").lower()
        self.model = model
        self.api_base = (api_base or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.calls = 0
        self.cached_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_path = cache_path
        self._cache: Dict[str, str] = {}
        self._dirty = 0
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except Exception:
                self._cache = {}

    @property
    def available(self) -> bool:
        return self.backend == "openai" and bool(self.api_key)

    # ------------------------------------------------------------------
    def _key(self, system: str, user: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode("utf-8"))
        h.update(b"\x00")
        h.update(system.encode("utf-8"))
        h.update(b"\x00")
        h.update(user.encode("utf-8"))
        return h.hexdigest()

    def flush_cache(self) -> None:
        if not self.cache_path:
            return
        parent = os.path.dirname(os.path.abspath(self.cache_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{self.cache_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)
        os.replace(tmp, self.cache_path)
        self._dirty = 0

    def chat(self, system: str, user: str) -> Optional[str]:
        if not self.available:
            return None
        key = self._key(system, user)
        if key in self._cache:
            self.cached_calls += 1
            return self._cache[key]
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self.calls += 1
                usage = body.get("usage", {})
                self.prompt_tokens += int(usage.get("prompt_tokens", 0))
                self.completion_tokens += int(usage.get("completion_tokens", 0))
                content = body["choices"][0]["message"]["content"]
                if self.cache_path:
                    self._cache[key] = content
                    self._dirty += 1
                    if self._dirty >= 20:
                        self.flush_cache()
                return content
            except Exception:
                if attempt == self.max_retries:
                    return None
        return None

    def cost(self) -> Dict[str, int]:
        self.flush_cache()
        return {"calls": self.calls, "cached_calls": self.cached_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


# ---------------------------------------------------------------------------
# rule generation (Table 10)
# ---------------------------------------------------------------------------

RULE_SYSTEM_PROMPT = (
    "You are a spatio-temporal logical rule inducer. Given several rule instances "
    "observed in the graph, induce new rules that are structurally similar and "
    "likely to hold. Use only the given list of relation names. Do not invent "
    "relation names."
)


def rule_to_sentence(rule: Rule, rel_names: Sequence[str]) -> str:
    def nm(r: int) -> str:
        r = int(r)
        if 0 <= r < len(rel_names):
            return rel_names[r]
        if 0 <= r - len(rel_names) < len(rel_names):
            return rel_names[r - len(rel_names)] + "^inv"
        return f"r{r}"

    parts = []
    for i, a in enumerate(rule.atoms):
        s = "X" if i == 0 else f"Z{i}"
        o = "Y" if i == len(rule.atoms) - 1 else f"Z{i + 1}"
        parts.append(f"{s} {nm(a.rel)} {o} within {a.window:g} days and {a.radius:g} km")
    text = "If " + ", and ".join(parts)
    if rule.branch is not None:
        text += f", and X {nm(rule.branch.rel)} Z' within {rule.branch.window:g} days and {rule.branch.radius:g} km"
    return text + f", then X {nm(rule.head_rel)} Y."


class RuleInducer:
    """LLM-supplemented candidate rules; every candidate must pass the filter."""

    def __init__(self, client: LLMClient, rel_names: Sequence[str], topk: int = 20,
                 max_new: int = 20, seed: int = 42):
        self.client = client
        self.rel_names = list(rel_names)
        self.topk = int(topk)
        self.max_new = int(max_new)
        self.rng = np.random.RandomState(seed)

    def induce(self, head_rel: int, instances: Sequence[Rule]) -> List[Rule]:
        instances = list(instances)[: self.topk]
        if not instances:
            return []
        prompt = self._build_prompt(head_rel, instances)
        raw = self.client.chat(RULE_SYSTEM_PROMPT, prompt)
        if raw:
            parsed = self._parse(raw, head_rel)
            if parsed:
                return parsed
        return self._offline_surrogate(head_rel, instances)

    # ------------------------------------------------------------------
    def _build_prompt(self, head_rel: int, instances: Sequence[Rule]) -> str:
        rel_list = ", ".join(self.rel_names)
        inst = "\n".join(f"- {rule_to_sentence(r, self.rel_names)}" for r in instances)
        return (
            f"[Head relation] {self.rel_names[head_rel]}\n"
            f"[Available relations] {rel_list}\n"
            f"[Rule instances]\n{inst}\n"
            f"[Task] Generate at most {self.max_new} new candidate rules in the same format.\n"
            "[Structural constraints] Chain: each candidate rule must have a chain of "
            "length at most 3. Optional branch: append at most one branch condition in "
            "the form: \"and X <r'> Z' within w days and p km\".\n"
            "[Output (JSON)] Return the candidate rules as a JSON list of objects with "
            "keys: atoms (list of {rel, window, radius}), branch ({rel, window, radius} or null)."
        )

    def _parse(self, raw: str, head_rel: int) -> List[Rule]:
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except Exception:
            return []
        out: List[Rule] = []
        name2id = {n: i for i, n in enumerate(self.rel_names)}
        for it in items[: self.max_new]:
            try:
                atoms = []
                for a in it.get("atoms", []):
                    rel = a.get("rel")
                    rid = name2id.get(rel, rel)
                    if not isinstance(rid, int) or not (0 <= rid < len(self.rel_names)):
                        raise ValueError
                    atoms.append(Atom(rel=rid, window=float(a.get("window", 1)),
                                      radius=float(a.get("radius", 1))))
                if not (1 <= len(atoms) <= 3):
                    raise ValueError
                br = it.get("branch")
                branch = None
                if br:
                    rid = name2id.get(br.get("rel"), br.get("rel"))
                    if isinstance(rid, int) and 0 <= rid < len(self.rel_names):
                        branch = Atom(rel=rid, window=float(br.get("window", 1)),
                                      radius=float(br.get("radius", 1)))
                out.append(Rule(head_rel=head_rel, atoms=atoms, branch=branch, source="llm"))
            except Exception:
                continue
        return out

    def _offline_surrogate(self, head_rel: int, instances: Sequence[Rule]) -> List[Rule]:
        """Structural variants of the observed chains (no external LLM).

        Generates (i) re-orderings of the two leading atoms and (ii) branch-augmented
        variants.  They are candidates only and must pass the same quality filter.
        """
        out: List[Rule] = []
        for r in instances:
            if len(r.atoms) >= 2:
                atoms = list(r.atoms)
                swapped = [atoms[1], atoms[0]] + atoms[2:]
                out.append(Rule(head_rel=head_rel, atoms=swapped,
                                branch=r.branch, source="llm"))
            if r.branch is None and len(r.atoms) >= 2:
                first = r.atoms[0]
                out.append(Rule(head_rel=head_rel, atoms=list(r.atoms),
                                branch=Atom(rel=first.rel, window=first.window,
                                            radius=first.radius),
                                source="llm"))
            if len(out) >= self.max_new:
                break
        return out[: self.max_new]# ---------------------------------------------------------------------------
# constraint judge (Table 9)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a spatio-temporal constraint checker. Judge whether a candidate "
    "violates spatio-temporal constraints only on the basis of the given facts. "
    "Do not add facts from external knowledge or guess the answer directly."
)


def build_judge_prompt(query: Dict[str, Any], candidates: List[Dict[str, Any]],
                       conflicts: List[Tuple[int, int]],
                       previous: Optional[Dict[int, float]] = None,
                       contradictions: Optional[List[Tuple[int, int]]] = None) -> str:
    lines = [
        "[Query]",
        f"Subject: {query['subject']}; relation: {query['relation']}; "
        f"co-occurrence-type: {'yes' if query['coocc'] else 'no'}.",
        f"Time: {query['time']}; location: {query.get('location', 'unknown')}.",
        "[Rule grounding evidence]",
    ]
    for ev in query.get("rule_evidence", []):
        lines.append(f"- Rule {ev['rule']} (quality {ev['quality']:.3f}) grounded with "
                     f"candidate {ev['candidate']}: {ev['facts']}")
    lines.append("[Candidates and nearest location facts]")
    for c in candidates:
        lines.append(f"- {c['candidate']}: nearest fact ({c['fact']}). "
                     f"Implied velocity: {c['v_impl']:.1f} km/h; bound: {c['v_max']:.1f} km/h.")
    lines.append("[Pairwise conflicts F]")
    for a, b in conflicts:
        lines.append(f"- ({a}, {b}): the two candidates cannot both hold with the query fact.")
    if previous:
        lines.append("[Previous-round judgments, if any]")
        lines.append(str({k: round(v, 3) for k, v in previous.items()}))
        lines.append(f"Detected contradictions: {contradictions}")
    lines += [
        "[Task] Evaluate each candidate using the supplied evidence. If the given "
        "judgments contradict each other, correct them in this round.",
        "[Scoring criteria] The score must be a real number between -1 and 1. "
        "Negative: violation of the velocity bound, geographic exclusion, or "
        "inconsistency with the known activity region. Positive: compatibility with "
        "all constraints, together with rule grounding support. Zero: undecidable.",
        "[Output (JSON)] For each candidate, output: "
        '{"candidate": ..., "score": ..., "reason": ...}',
    ]
    return "\n".join(lines)


def parse_judge_output(raw: str, candidates: Sequence[int]) -> Dict[int, Tuple[float, str]]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return {}
    try:
        items = json.loads(m.group(0))
    except Exception:
        return {}
    out: Dict[int, Tuple[float, str]] = {}
    valid = set(int(c) for c in candidates)
    for it in items:
        try:
            c = int(it["candidate"])
            s = float(it["score"])
        except Exception:
            continue
        if c in valid:
            out[c] = (float(np.clip(s, -1.0, 1.0)), str(it.get("reason", "")))
    return out


class LLMJudge:
    """Candidate judge of Algorithm 1 (Table 9 prompt).

    Two backends:
      * ``openai`` -- the prompt of Table 9 is sent to ``cfg.model`` and the JSON
        answer is parsed back into per-candidate scores in ``[-1, 1]``.
      * ``offline`` -- a deterministic symbolic surrogate that applies the same
        velocity / geographic-exclusion criteria used to *build* the prompt, plus
        the meta-evaluation reconciliation over the conflict set ``F``.  It keeps
        the pipeline reproducible without an API key; the backend is recorded in
        every result file so surrogate scores are never mistaken for LLM output.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.client = LLMClient(
            backend=cfg.backend, model=cfg.model,
            api_base=getattr(cfg, "api_base", None),
            api_key_env=getattr(cfg, "api_key_env", "OPENAI_API_KEY"),
            temperature=getattr(cfg, "temperature_llm", 0.0),
            max_retries=getattr(cfg, "max_retries", 2),
            timeout=getattr(cfg, "timeout", 60),
            cache_path=getattr(cfg, "cache_path", None),
        )

    # ------------------------------------------------------------------
    def score(self, payload: Dict[str, Any], n_candidates: int,
              rounds: int = 1, ablate_meta: bool = False,
              velocity_ok: Optional[torch.Tensor] = None,
              v_impl: Optional[torch.Tensor] = None,
              v_max: float = 900.0) -> torch.Tensor:
        if self.client.available and str(self.cfg.backend) != "offline":
            return self._llm_score(payload, n_candidates, rounds)
        return self._surrogate(velocity_ok, v_impl, v_max,
                               payload.get("conflicts", []), rounds, ablate_meta)

    # ------------------------------------------------------------------
    def _surrogate(self, velocity_ok, v_impl, v_max, conflicts,
                   rounds: int, ablate_meta: bool) -> torch.Tensor:
        ratio = (v_impl / max(v_max, 1e-6)).clamp(0.0, 3.0)
        base = torch.sigmoid(3.0 * (1.0 - ratio))
        if velocity_ok is not None:
            base = torch.where(velocity_ok, base, torch.full_like(base, 0.02))

        if ablate_meta or rounds <= 1:            # w/o Meta: no reconciliation
            return base

        r = base.clone()
        for _ in range(int(rounds) - 1):          # Algorithm 1, lines 8-14
            changed = False
            for i, j in conflicts:
                if float(r[i]) > 0.5 and float(r[j]) > 0.5:
                    if float(r[i]) >= float(r[j]):
                        r[j] = r[j] * 0.5
                    else:
                        r[i] = r[i] * 0.5
                    changed = True
            if not changed:
                break
        return r

    # ------------------------------------------------------------------
    def _llm_score(self, payload: Dict[str, Any], n_candidates: int,
                   rounds: int) -> torch.Tensor:
        query = payload["query"]
        candidates = payload["candidates"]
        conflicts = payload.get("conflicts", [])
        previous: Optional[Dict[int, float]] = None
        scores = {c["candidate"]: 0.5 for c in candidates}
        for _ in range(max(1, int(rounds))):
            prompt = build_judge_prompt(query, candidates, conflicts, previous)
            raw = self.client.chat(JUDGE_SYSTEM_PROMPT, prompt)
            if raw is None:
                break
            parsed = parse_judge_output(raw, [c["candidate"] for c in candidates])
            if not parsed:
                break
            previous = {k: v[0] for k, v in parsed.items()}
            scores = {c["candidate"]: 0.5 * (parsed[c["candidate"]][0] + 1.0)
                      if c["candidate"] in parsed else scores[c["candidate"]]
                      for c in candidates}
        return torch.tensor([scores[c["candidate"]] for c in candidates],
                            dtype=torch.float32)