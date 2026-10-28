# -*- coding: utf-8 -*-
"""Frozen text encoder for administrative location descriptions.

The manuscript uses the [CLS] vector of a frozen pre-trained text encoder
(Section 3.2).  We support two backends:

* ``bert``  -- ``transformers`` + ``bert-base-uncased`` (the paper's setting);
* ``hash``  -- a deterministic character n-gram projection used when
  ``transformers`` / model weights are unavailable (offline environments).

Both produce a *frozen* (num_text, dim) table; only a linear projection on top is
trained, exactly as in the paper.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils import load_pickle, save_pickle


class TextLocationEncoder:
    def __init__(self, backend: str = "auto", dim: int = 768, cache_dir: str = "cache"):
        self.backend = backend
        self.dim = dim
        self.cache_dir = cache_dir
        self._model = None

    # ------------------------------------------------------------------
    def _load_bert(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception:
            return None
        name = "bert-base-uncased"
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name)
            mdl.eval()
            self._model = (tok, mdl)
            return self._model
        except Exception:
            return None

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        key = hashlib.md5(("\n".join(texts) + f"|{self.backend}").encode("utf-8")).hexdigest()[:16]
        cache = os.path.join(self.cache_dir, f"text_emb_{key}.pkl")
        if os.path.exists(cache):
            return load_pickle(cache)

        backend = self.backend
        if backend == "auto":
            backend = "bert" if self._load_bert() is not None else "hash"

        if backend == "bert" and self._model is None:
            self._load_bert()

        if backend == "bert" and self._model is not None:
            emb = self._encode_bert(texts, batch_size)
        else:
            emb = self._encode_hash(texts)

        os.makedirs(self.cache_dir, exist_ok=True)
        save_pickle(emb, cache)
        return emb

    def _encode_bert(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        import torch

        tok, mdl = self._model
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i:i + batch_size])
                enc = tok(batch, padding=True, truncation=True, max_length=32,
                          return_tensors="pt")
                h = mdl(**enc).last_hidden_state[:, 0, :]
                out.append(h.cpu().numpy())
        emb = np.concatenate(out, axis=0)
        return emb.astype(np.float32)

    def _encode_hash(self, texts: Sequence[str]) -> np.ndarray:
        """Deterministic hashed character/word n-gram embedding (offline fallback)."""
        emb = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = t.lower().replace("|", " ").split()
            feats: List[str] = []
            for w in toks:
                feats.append("w:" + w)
                for n in (3, 4):
                    for j in range(max(1, len(w) - n + 1)):
                        feats.append(f"c{n}:" + w[j:j + n])
            for f in feats:
                h = hashlib.md5(f.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                emb[i, idx] += sign
            n = np.linalg.norm(emb[i])
            if n > 0:
                emb[i] /= n
        return emb