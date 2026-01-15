from __future__ import annotations
import phasis.runtime as rt
import configparser
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from hashlib import md5 as _md5
import hashlib
import os
import time

MEM_FILE_DEFAULT = "phasis.mem"


def phase2_basename(base_name:str)->str:
    try:
        is_concat = bool(rt.concat_libs)
    except Exception:
        is_concat = False
    prefix = "concat_" if is_concat else ""
    return f"{prefix}{rt.phase}_{base_name}"

EMPTY_MD5      = "d41d8cd98f00b204e9800998ecf8427e"
_CHUNK_SIZE    = 8 * 1024 * 1024  # 8 MiB
_MD5_RETRIES   = 3
_MD5_BACKOFF_S = 0.2

def _wait_size_stable(path, checks=3, interval=0.2, timeout=5.0):
    """Wait until file size stops changing for `checks` consecutive polls."""
    deadline = time.time() + timeout
    last = -1
    stable = 0
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(interval)
            continue
        if size == 0:
            # empty file is "stable" but handled by EMPTY_MD5 in getmd5()
            stable = 0
        if size == last and size > 0:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
            last = size
        time.sleep(interval)
    # best effort: if it exists at all, proceed
    return os.path.isfile(path)

def getmd5(afile):
    """
    Return (afile, md5hex_str). Robust against transient FS races.
    Strategy:
      - handle empty-file fast path,
      - wait for size stability,
      - try file_digest (3.11+),
      - fallback to streaming read (8 MiB chunks),
      - final fallback with smaller chunks.
    """
    p = afile
    try:
        # Empty file fast path
        try:
            if os.path.getsize(p) == 0:
                return (p, EMPTY_MD5)
        except Exception:
            # stat failed, keep going and let open() decide
            pass

        # Settle the file to avoid hashing during writes/flush on network FS
        _wait_size_stable(p, checks=3, interval=0.2, timeout=5.0)

        for attempt in range(_MD5_RETRIES):
            try:
                # 1) C-accelerated hash (Py 3.11+)
                if hasattr(hashlib, "file_digest"):
                    try:
                        with open(p, "rb", buffering=0) as f:
                            h = hashlib.file_digest(f, hashlib.md5())
                            return (p, h.hexdigest())
                    except Exception:
                        # fall through to streaming
                        pass

                # 2) Streaming in large chunks
                try:
                    md5 = hashlib.md5()
                    buf = bytearray(_CHUNK_SIZE)
                    mv = memoryview(buf)
                    with open(p, "rb", buffering=0) as f:
                        while True:
                            n = f.readinto(buf)
                            if not n:
                                break
                            md5.update(mv[:n])
                    return (p, md5.hexdigest())
                except Exception:
                    # 3) Final fallback: smaller chunks (1 MiB)
                    md5 = hashlib.md5()
                    small = 1024 * 1024
                    with open(p, "rb", buffering=0) as f:
                        while True:
                            chunk = f.read(small)
                            if not chunk:
                                break
                            md5.update(chunk)
                    return (p, md5.hexdigest())

            except Exception:
                time.sleep(_MD5_BACKOFF_S)

        # Persistent failure: last resort – do NOT crash; return empty string
        return (p, "")

    except Exception:
        return (p, "")

def compute_md5_str(path: str) -> str | None:
    """Chunked md5 -> hex string (or None on failure)."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None
    
def _md5_of_list_str(items):
    h = _md5()
    for s in items:
        h.update(str(s).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()

def md5_file_worker(path):
    """
    Return (ABS_REALPATH, md5hex or None).
    - Normalizes path to realpath for stable [CLUSTERED] keys.
    - Uses getmd5(), which now guarantees a string; maps "" -> None for failure.
    - No extra retries here (getmd5 already handles them).
    """
    try:
        p = os.path.realpath(path)
    except Exception:
        p = path

    _, md5hex = getmd5(p)          # always returns a string now
    if not md5hex:                  # "" => treat as failure
        md5hex = None
    return (p, md5hex)

def _chunk_id_from_name(fn):
    # 'ALL_LIBS-10.sRNA_21.cluster' -> 10 (int); robust to oddities
    try:
        after_dash = fn.rsplit("-", 1)[1]            # '10.sRNA_21.cluster'
        num = after_dash.split(".", 1)[0]            # '10'
        return int(num)
    except Exception:
        return 0
    
def discover_scored_prefixes(scored_dir, phase):
    # Returns sorted list of prefixes that have at least one *.cluster
    suffix = f".sRNA_{phase}.cluster"
    prefixes = set()
    if not os.path.isdir(scored_dir):
        return []
    for fn in os.listdir(scored_dir):
        if fn.endswith(suffix):
            prefixes.add(fn.rsplit("-", 1)[0])
    return sorted(prefixes)

def list_chunk_files_for_prefix(scored_dir, prefix, phase):
    # Returns sorted (by numeric chunk id) absolute paths for prefix
    suffix = f".sRNA_{phase}.cluster"
    out = []
    if not os.path.isdir(scored_dir):
        return out
    pref = f"{prefix}-"
    for fn in os.listdir(scored_dir):
        if fn.endswith(suffix) and fn.startswith(pref):
            out.append(os.path.join(scored_dir, fn))
    out.sort(key=lambda p: _chunk_id_from_name(os.path.basename(p)))
    return out

def assemble_candidate_from_chunks(scored_dir, lib_prefix, phase, out_path):
    """
    Stream-concatenate non-empty chunk files for a given lib_prefix into out_path.
    Returns (#chunks_used, total_bytes_written).
    """
    chunks = list_chunk_files_for_prefix(scored_dir, lib_prefix, phase)
    used = 0
    written = 0
    # fresh file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as outfh:
        for cp in chunks:
            try:
                if os.path.getsize(cp) == 0:
                    continue
            except Exception:
                # if stat fails, skip conservatively
                continue
            with open(cp, "rb") as infh:
                # copy in 1 MiB blocks
                while True:
                    blk = infh.read(1024 * 1024)
                    if not blk:
                        break
                    outfh.write(blk)
                    written += len(blk)
            used += 1
    return used, written

def _runtime_params_signature() -> str:
    """Hash of runtime knobs that affect window selection (no JSON dependency)."""
    phase_v = globals().get("phase")
    wl_v    = globals().get("window_len")
    sl_v    = globals().get("sliding")
    mcl_v   = globals().get("minClusterLength")
    sig_str = f"phase={phase_v};wl={wl_v};sl={sl_v};mcl={mcl_v};version=winselect.1"
    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()

# === Phase II cache and naming helpers (concat-aware, param-aware, ConfigParser-safe) ===

def _run_signature_keyprefix():
    # No '=' or ':' in the signature to keep ConfigParser option names safe.
    try:
        _ph = str(rt.phase)
    except Exception:
        _ph = "NA"
    try:
        _c = "1" if bool(rt.concat_libs) else "0"
    except Exception:
        _c = "0"
    try:
        _lib_names = [os.path.basename(x or "") for x in (rt.libs or [])]
    except Exception:
        _lib_names = []
    libs_md5   = _md5_of_list_str(sorted(_lib_names))[:8]
    params_md5 = _md5_of_list_str([rt.mindepth, rt.maxhits, rt.clustbuffer])[:8]
    return f"ph-{_ph}_c-{_c}_L-{libs_md5}_P-{params_md5}"

def _normalize_path_for_mem(path:str)->str:
    # Compose a ConfigParser-safe key using only [A-Za-z0-9_.-]
    import re as _re
    sig = _run_signature_keyprefix().replace('=', '-').replace('|', '_')
    abspath = os.path.abspath(path)
    from hashlib import md5 as _md5_local
    pmd5 = _md5_local(abspath.encode('utf-8')).hexdigest()[:10]
    base = os.path.basename(abspath)
    base_sanitized = _re.sub(r'[^A-Za-z0-9_.-]', '_', base)
    key = f"{sig}__path-{pmd5}__base-{base_sanitized}"
    key = _re.sub(r'[^A-Za-z0-9_.-]', '_', key)
    return key

def _mem_base_dir() -> str:
    """
    Where to anchor relative paths stored in the mem file.
    Prefer rt.outdir if defined; otherwise fall back to CWD.
    """
    base = getattr(rt, "outdir", None)
    if base:
        return os.path.abspath(base)
    return os.path.abspath(os.getcwd())


def _mem_ini_key_for_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Convert a file path into a stable INI key.
    - No expanduser("~") (per your constraint)
    - If path starts with "~", keep it literal
    - Otherwise normalize to an absolute path (anchored at base_dir for relative inputs)
    """
    p = (path or "").strip()
    if not p:
        return p

    # Do not expand "~" (leave literal)
    if p.startswith("~"):
        return p

    if base_dir is None:
        base_dir = _mem_base_dir()

    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)

    return os.path.normpath(os.path.abspath(p))


def sanitize_mem_md5s(config: configparser.ConfigParser,
                      sections: Iterable[str] = ("CLUSTERS", "CLUSTERED")) -> Dict[str, int]:
    """
    Remove empty-string / None MD5s that can poison cache decisions.
    Returns: dict {section: removed_count}
    """
    removed: Dict[str, int] = {}
    for sect in sections:
        cnt = 0
        if config.has_section(sect):
            for k in list(config[sect].keys()):
                v = config[sect].get(k)
                if v is None or not str(v).strip():
                    del config[sect][k]
                    cnt += 1
        removed[sect] = cnt
    return removed


def mem_get(cfg, section, path):
    if not cfg.has_section(section):
        return None

    base_dir = _mem_base_dir()
    k_new  = _mem_ini_key_for_path(path, base_dir=base_dir)  # stable + run-aware
    k_norm = _normalize_path_for_mem(path, base_dir=base_dir) # older style

    for k in (k_new, k_norm, path):
        v = cfg[section].get(k)
        if v is not None:
            return v
    return None


def mem_set(cfg: configparser.ConfigParser, section: str, path: str, value) -> None:
    """
    Store md5 for 'path' into cfg[section] using a stable normalized key.
    """
    if not cfg.has_section(section):
        cfg.add_section(section)

    base_dir = _mem_base_dir()
    key = _mem_ini_key_for_path(path, base_dir=base_dir)
    cfg[section][key] = str(value).strip()

@dataclass(frozen=True)
class MemBasic:
    ok: bool
    index: Optional[str]
    genomehash: Optional[str]
    indexhash: Optional[str]

def read_mem_basic(mem_file: str) -> MemBasic:
    """
    Pure read of the mem INI file.
    - No prints
    - No global writes
    - No dependency on legacy module state
    """
    config = configparser.ConfigParser()
    config.read(mem_file)

    if not config.has_section("BASIC"):
        return MemBasic(False, None, None, None)

    basic = config["BASIC"]
    genomehash = basic.get("genomehash")
    indexhash = basic.get("indexhash")
    index = basic.get("index")

    ok = bool(genomehash and indexhash and index)
    return MemBasic(ok=ok, index=index, genomehash=genomehash, indexhash=indexhash)

__all__ = [
    "MEM_FILE_DEFAULT",
    "phase2_basename",
    "getmd5",
    "compute_md5_str",
    "md5_file_worker",
    "list_chunk_files_for_prefix",
    "assemble_candidate_from_chunks",
    "sanitize_mem_md5s",
    "mem_get",
    "mem_set",
    "MemBasic",
    "read_mem_basic",
]