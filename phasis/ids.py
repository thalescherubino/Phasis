from __future__ import annotations

"""
phasis.ids
----------
Lightweight, spawn-safe ID utilities.

Goals:
- Provide a single, unambiguous implementation of mergedClusterDict loading.
- Provide a reverse index (raw clusterID -> universalID) cached in both:
  - module globals (legacy compatibility)
  - phasis.runtime (forward-compatible single source of truth)

Important:
- The reverse map is process-local. On macOS (spawn), each worker may need to
  load/rebuild it; this module keeps it cheap by:
  - loading the persisted {phase}_mergedClusterDict.tab when present
  - caching results in-process
"""

import csv
import os
import re
import threading
from typing import Dict, List, Optional

import phasis.runtime as rt
from phasis.cache import phase2_basename

# ---------------------------------------------------------------------
# Process-local caches (safe for fork; rebuilt per-process for spawn)
# ---------------------------------------------------------------------

_MERGED_DICT_LOCK = threading.Lock()


def _load_simple_tab_dict(path: str) -> Dict[str, List[str]]:
    """Load a key \\t values... tab into {key: [values...]}; tolerate empty lines."""
    out: Dict[str, List[str]] = {}
    with open(path, "r") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            key = parts[0].strip()
            vals = [v for v in (p.strip() for p in parts[1:]) if v]
            out[key] = vals if vals else [key]
    return out


def _set_reverse_merged_map(mcd: Dict[str, List[str]]) -> None:
    """Cache clusterID -> universalID reverse map in both legacy globals and runtime."""
    rev: Dict[str, str] = {}
    for u, members in (mcd or {}).items():
        for cid in members or []:
            s = str(cid).strip()
            if s:
                rev[s] = str(u)

    # legacy globals (compat)
    globals()["MERGED_CLUSTER_REVERSE"] = rev
    globals()["mergedClusterReverse"] = rev
    globals()["mergedClusterDict"] = mcd

    # runtime (preferred going forward)
    rt.mergedClusterDict = mcd
    rt.mergedClusterReverse = rev


def ensure_mergedClusterDict(phase: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Light loader:
      1) return cached mergedClusterDict if available (either in globals or runtime),
      2) else load {phase}_mergedClusterDict.tab (phase2_basename),
      3) else set empty caches and return {}.

    NOTE: We intentionally do NOT build identity maps here (only universal IDs).
    """
    with _MERGED_DICT_LOCK:
        # 1) runtime cache
        mcd = getattr(rt, "mergedClusterDict", None)
        if isinstance(mcd, dict) and mcd:
            # ensure reverse exists too
            rev = getattr(rt, "mergedClusterReverse", None)
            if not isinstance(rev, dict) or not rev:
                _set_reverse_merged_map(mcd)
            return mcd

        # 1b) legacy global cache
        g = globals()
        if isinstance(g.get("mergedClusterDict"), dict) and g["mergedClusterDict"]:
            mcd = g["mergedClusterDict"]
            _set_reverse_merged_map(mcd)
            return mcd

        # 2) load from disk
        dict_tab = phase2_basename("mergedClusterDict.tab")
        if os.path.isfile(dict_tab):
            try:
                mcd = _load_simple_tab_dict(dict_tab)
                _set_reverse_merged_map(mcd)
                return mcd
            except Exception as e:
                print(f"[WARN] Failed to load {dict_tab}: {e}")

        # 3) empty fallback
        _set_reverse_merged_map({})
        return {}


def _strip_fileprefix_from_id(cid: str, phase: Optional[int] = None) -> str:
    """
    Strip the common PHAS candidate prefix from a clusterID string.

    Example:
      s.S3.1_Pre_1.24-PHAS.candidate.clusters1006_18  ->  1006_18
    """
    s = str(cid).strip()
    ph = None
    if phase is not None:
        try:
            ph = int(phase)
        except Exception:
            ph = None
    if ph is None:
        try:
            ph = int(getattr(rt, "phase", 0) or 0)
        except Exception:
            ph = 0

    if ph:
        pat = rf".*?\.{re.escape(str(ph))}-PHAS\.candidate\.clusters"
        s = re.sub(pat, "", s)
    return s.strip()


def normalize_cluster_id_for_lookup(cid: str, phase: Optional[int] = None) -> str:
    """
    Normalize a clusterID so it can be used as a key in caches/lookups.

    Rules:
    - If it's already coordinate-style ("chr:start..end"), return as-is.
    - Else, strip the PHAS candidate file prefix (when present).
    """
    s = str(cid).strip()
    if (":" in s) and (".." in s):
        return s
    return _strip_fileprefix_from_id(s, phase=phase)


def _ensure_reverse_index() -> Dict[str, str]:
    """Return the cached reverse index (clusterID -> universalID), building if needed."""
    rev = getattr(rt, "mergedClusterReverse", None)
    if isinstance(rev, dict) and rev:
        return rev

    g_rev = globals().get("MERGED_CLUSTER_REVERSE")
    if isinstance(g_rev, dict) and g_rev:
        rt.mergedClusterReverse = g_rev
        return g_rev

    # If neither exists, try loading dict and building
    ensure_mergedClusterDict()
    rev = getattr(rt, "mergedClusterReverse", None)
    if isinstance(rev, dict):
        return rev
    return {}


def getUniversalID(clusterID: str) -> Optional[str]:
    """
    Map a raw clusterID to its universal ID using mergedClusterDict.

    Strategy:
    1) Ensure mergedClusterDict + reverse map are available (light loader).
    2) Try exact key in mergedClusterDict (already universal).
    3) Try prefix-stripped key in mergedClusterDict (already universal).
    4) Try reverse map lookup on raw and stripped.
    5) Return None if not resolvable.
    """
    ensure_mergedClusterDict()
    rev = _ensure_reverse_index()

    cid_raw = str(clusterID).strip()
    cid_norm = normalize_cluster_id_for_lookup(cid_raw)

    mcd = getattr(rt, "mergedClusterDict", None)
    if not isinstance(mcd, dict) or not mcd:
        mcd = globals().get("mergedClusterDict", {}) or {}

    # Already universal?
    if cid_raw in mcd:
        return cid_raw
    if cid_norm in mcd:
        return cid_norm

    # Reverse map
    if cid_raw in rev:
        return rev[cid_raw]
    if cid_norm in rev:
        return rev[cid_norm]

    return None
