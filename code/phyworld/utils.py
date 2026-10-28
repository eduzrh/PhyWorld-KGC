# -*- coding: utf-8 -*-
"""Shared utilities: seeding, IO, great-circle geometry, geocoding."""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0088
MISSING_TOKENS = {"", "∅", "none", "nan", "null", "-"}


# --------------------------------------------------------------------------
# seeding / logging
# --------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Global seed (the requirement document fixes seed=42)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch always present in practice
        pass


def setup_logger(name: str = "phyworld", log_file: Optional[str] = None, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# geography
# --------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance d^⊕(.,.) in kilometres (vectorised)."""
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def parse_location(token: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse one location field.

    Returns ``(lat, lon, text)``:
      * ``(lat, lon, None)``  for ``"(37.5,127.0)"``
      * ``(None, None, txt)`` for administrative text
      * ``(None, None, None)`` for the missing marker
    """
    if token is None:
        return None, None, None
    tok = token.strip()
    if tok.lower() in MISSING_TOKENS:
        return None, None, None
    body = tok[1:-1] if (tok.startswith("(") and tok.endswith(")")) else tok
    parts = [x.strip() for x in body.split(",")]
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1]), None
        except ValueError:
            pass
    return None, None, tok


# --------------------------------------------------------------------------
# gazetteer
# --------------------------------------------------------------------------

#: coarse country centroids (lat, lon) used as administrative-region centroids
#: when no finer gazetteer entry exists. Extend by dropping
#: ``data/gazetteer.json`` with {"name": [lat, lon], ...}.
COUNTRY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "united states": (39.8283, -98.5795), "china": (35.8617, 104.1954),
    "russia": (61.5240, 105.3188), "india": (20.5937, 78.9629),
    "iran": (32.4279, 53.6880), "iraq": (33.2232, 43.6793),
    "afghanistan": (33.9391, 67.7100), "pakistan": (30.3753, 69.3451),
    "turkey": (38.9637, 35.2433), "syria": (34.8021, 38.9968),
    "israel": (31.0461, 34.8516), "saudi arabia": (23.8859, 45.0792),
    "egypt": (26.8206, 30.8025), "libya": (26.3351, 17.2283),
    "nigeria": (9.0820, 8.6753), "south africa": (-30.5595, 22.9375),
    "kenya": (0.0236, 37.9062), "ethiopia": (9.1450, 40.4897),
    "somalia": (5.1521, 46.1996), "sudan": (12.8628, 30.2176),
    "france": (46.2276, 2.2137), "germany": (51.1657, 10.4515),
    "united kingdom": (55.3781, -3.4360), "italy": (41.8719, 12.5674),
    "spain": (40.4637, -3.7492), "ukraine": (48.3794, 31.1656),
    "poland": (51.9194, 19.1451), "netherlands": (52.1326, 5.2913),
    "belgium": (50.5039, 4.4699), "sweden": (60.1282, 18.6435),
    "norway": (60.4720, 8.4689), "finland": (61.9241, 25.7482),
    "greece": (39.0742, 21.8243), "portugal": (39.3999, -8.2245),
    "switzerland": (46.8182, 8.2275), "austria": (47.5162, 14.5501),
    "czech republic": (49.8175, 15.4730), "hungary": (47.1625, 19.5033),
    "romania": (45.9432, 24.9668), "bulgaria": (42.7339, 25.4858),
    "belarus": (53.7098, 27.9534), "serbia": (44.0165, 21.0059),
    "canada": (56.1304, -106.3468), "mexico": (23.6345, -102.5528),
    "brazil": (-14.2350, -51.9253), "argentina": (-38.4161, -63.6167),
    "chile": (-35.6751, -71.5430), "colombia": (4.5709, -74.2973),
    "venezuela": (6.4238, -66.5897), "peru": (-9.1900, -75.0152),
    "cuba": (21.5218, -77.7812), "japan": (36.2048, 138.2529),
    "south korea": (35.9078, 127.7669), "north korea": (40.3399, 127.5101),
    "indonesia": (-0.7893, 113.9213), "malaysia": (4.2105, 101.9758),
    "thailand": (15.8700, 100.9925), "vietnam": (14.0583, 108.2772),
    "philippines": (12.8797, 121.7740), "myanmar": (21.9162, 95.9560),
    "bangladesh": (23.6850, 90.3563), "sri lanka": (7.8731, 80.7718),
    "nepal": (28.3949, 84.1240), "australia": (-25.2744, 133.7751),
    "new zealand": (-40.9006, 174.8860), "kazakhstan": (48.0196, 66.9237),
    "uzbekistan": (41.3775, 64.5853), "tajikistan": (38.8610, 71.2761),
    "kyrgyzstan": (41.2044, 74.7661), "turkmenistan": (38.9697, 59.5563),
    "azerbaijan": (40.1431, 47.5769), "armenia": (40.0691, 45.0382),
    "georgia": (42.3154, 43.3569), "mongolia": (46.8625, 103.8467),
    "taiwan": (23.6978, 120.9605), "hong kong": (22.3193, 114.1694),
    "singapore": (1.3521, 103.8198), "qatar": (25.3548, 51.1839),
    "united arab emirates": (23.4241, 53.8478), "kuwait": (29.3117, 47.4818),
    "jordan": (30.5852, 36.2384), "lebanon": (33.8547, 35.8623),
    "yemen": (15.5527, 48.5164), "oman": (21.4735, 55.9754),
    "bahrain": (26.0667, 50.5577), "morocco": (31.7917, -7.0926),
    "algeria": (28.0339, 1.6596), "tunisia": (33.8869, 9.5375),
    "ghana": (7.9465, -1.0232), "senegal": (14.4974, -14.4524),
    "mali": (17.5707, -3.9962), "niger": (17.6078, 8.0817),
    "chad": (15.4542, 18.7322), "cameroon": (7.3697, 12.3547),
    "democratic republic of the congo": (-4.0383, 21.7587),
    "republic of the congo": (-0.2280, 15.8277), "angola": (-11.2027, 17.8739),
    "zimbabwe": (-19.0154, 29.1549), "zambia": (-13.1339, 27.8493),
    "mozambique": (-18.6657, 35.5296), "tanzania": (-6.3690, 34.8888),
    "uganda": (1.3733, 32.2903), "rwanda": (-1.9403, 29.8739),
    "burundi": (-3.3731, 29.9189), "central african republic": (6.6111, 20.9394),
    "south sudan": (6.8770, 31.3070), "eritrea": (15.1794, 39.7823),
    "djibouti": (11.8251, 42.5903), "mauritania": (21.0079, -10.9408),
    "burkina faso": (12.2383, -1.5616), "benin": (9.3077, 2.3158),
    "togo": (8.6195, 0.8248), "ivory coast": (7.5400, -5.5471),
    "guinea": (9.9456, -9.6966), "sierra leone": (8.4606, -11.7799),
    "liberia": (6.4281, -9.4295), "gambia": (13.4432, -15.3101),
    "guinea-bissau": (11.8037, -15.1804), "gabon": (-0.8037, 11.6094),
    "equatorial guinea": (1.6508, 10.2679), "botswana": (-22.3285, 24.6849),
    "namibia": (-22.9576, 18.4904), "lesotho": (-29.6100, 28.2336),
    "swaziland": (-26.5225, 31.4659), "madagascar": (-18.7669, 46.8691),
    "malawi": (-13.2543, 34.3015), "czechia": (49.8175, 15.4730),
    "slovakia": (48.6690, 19.6990), "slovenia": (46.1512, 14.9955),
    "croatia": (45.1000, 15.2000), "bosnia and herzegovina": (43.9159, 17.6791),
    "macedonia": (41.6086, 21.7453), "albania": (41.1533, 20.1683),
    "moldova": (47.4116, 28.3699), "lithuania": (55.1694, 23.8813),
    "latvia": (56.8796, 24.6032), "estonia": (58.5953, 25.0136),
    "iceland": (64.9631, -19.0208), "ireland": (53.1424, -7.6921),
    "denmark": (56.2639, 9.5018), "belize": (17.1899, -88.4976),
    "guatemala": (15.7835, -90.2308), "honduras": (15.2000, -86.2419),
    "el salvador": (13.7942, -88.8965), "nicaragua": (12.8654, -85.2072),
    "costa rica": (9.7489, -83.7534), "panama": (8.5380, -80.7821),
    "ecuador": (-1.8312, -78.1834), "bolivia": (-16.2902, -63.5887),
    "paraguay": (-23.4425, -58.4438), "uruguay": (-32.5228, -55.7658),
    "haiti": (18.9712, -72.2852), "dominican republic": (18.7357, -70.1627),
    "jamaica": (18.1096, -77.2975), "trinidad and tobago": (10.6918, -61.2225),
    "papua new guinea": (-6.3150, 143.9555), "fiji": (-17.7134, 178.0650),
    "brunei": (4.5353, 114.7277), "cambodia": (12.5657, 104.9910),
    "laos": (19.8563, 102.4955), "bhutan": (27.5142, 90.4336),
    "maldives": (3.2028, 73.2207), "timor-leste": (-8.8742, 125.7275),
}


class Geocoder:
    """Offline geocoder: administrative text -> centroid.

    Priority: explicit gazetteer -> exact country centroid -> last component of
    ``"City | Province | Country"`` -> first component -> unresolved (treated as
    missing, as required by Section 3.2).
    """

    def __init__(self, gazetteer_path: Optional[str] = None, mode: str = "builtin"):
        self.mode = (mode or "builtin").lower()
        self.table: Dict[str, Tuple[float, float]] = {}
        self.table.update(COUNTRY_CENTROIDS)
        self.cache: Dict[str, Optional[Tuple[float, float]]] = {}
        if gazetteer_path and os.path.exists(gazetteer_path):
            extra = load_json(gazetteer_path)
            for k, v in extra.items():
                if v is None:
                    continue
                self.table[self._norm(k)] = (float(v[0]), float(v[1]))

    @staticmethod
    def _norm(text: str) -> str:
        t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        return " ".join(t.lower().replace("_", " ").split())

    def resolve(self, text: Optional[str]) -> Optional[Tuple[float, float]]:
        if text is None:
            return None
        if text in self.cache:
            return self.cache[text]
        key = self._norm(text)
        result = None
        if key in self.table:
            result = self.table[key]
        else:
            # hierarchical administrative string, e.g. "Tehran | Tehran Province | Iran"
            parts = [self._norm(p) for p in text.split("|") if p.strip()]
            for cand in reversed(parts):          # country first, then province/city
                if cand in self.table:
                    result = self.table[cand]
                    break
            if result is None and parts:
                # try a city-level gazetteer entry on the most specific component
                for cand in parts:
                    if cand in self.table:
                        result = self.table[cand]
                        break
        self.cache[text] = result
        return result

    def stats(self) -> Dict[str, int]:
        hit = sum(1 for v in self.cache.values() if v is not None)
        return {"queried": len(self.cache), "resolved": hit,
                "unresolved": len(self.cache) - hit}


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def split_compound(name: str) -> List[str]:
    """Split a compound variant label such as ``"w/o Gate+w/o Roll."``."""
    return [p.strip() for p in name.split("+") if p.strip()]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path