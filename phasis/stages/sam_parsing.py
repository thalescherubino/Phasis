# phasis/stages/sam_parsing.py
from __future__ import annotations

import os
import gc
import pickle
import configparser
from collections import defaultdict, OrderedDict, Counter
from typing import Iterable, List, Sequence, Tuple

import phasis.runtime as rt
from phasis.parallel import run_parallel_with_progress
from phasis.cache import getmd5, MEM_FILE_DEFAULT


def _resolve_mem_file() -> str:
    mem = getattr(rt, "memFile", None)
    return mem if mem else MEM_FILE_DEFAULT


def updatedsets(config: configparser.ConfigParser) -> List[str]:
    """
    Determine which settings changed since the last run by comparing the mem/settings
    file values to current runtime (rt.*).

    Defensive: if sections/keys are missing or malformed, we skip that check.
    """
    updated: List[str] = []

    # ADVANCED knobs
    try:
        if config.has_section("ADVANCED"):
            if "mismat" in config["ADVANCED"]:
                try:
                    if int(config["ADVANCED"]["mismat"]) != int(getattr(rt, "mismat", 0)):
                        updated.append("mismat")
                except Exception:
                    pass
            if "maxhits" in config["ADVANCED"]:
                try:
                    if int(config["ADVANCED"]["maxhits"]) != int(getattr(rt, "maxhits", 0)):
                        updated.append("maxhits")
                except Exception:
                    pass
            if "clustbuffer" in config["ADVANCED"]:
                try:
                    if int(config["ADVANCED"]["clustbuffer"]) != int(getattr(rt, "clustbuffer", 0)):
                        updated.append("clustbuffer")
                except Exception:
                    pass
    except Exception:
        pass

    # BASIC: phase length (optional; older configs may not have it)
    try:
        if config.has_section("BASIC") and "phaselen" in config["BASIC"]:
            try:
                if int(config["BASIC"]["phaselen"]) != int(getattr(rt, "phase", 0)):
                    updated.append("phaselen")
            except Exception:
                # Some historical configs stored non-int tokens here; ignore.
                pass
    except Exception:
        pass

    return updated


def libstoset(alist: Iterable[Tuple[str, str]], akey: str) -> None:
    """Write (path, md5) entries to the mem/settings file under section `akey`."""
    mem_file = _resolve_mem_file()

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(mem_file)

    if not config.has_section(akey):
        config.add_section(akey)

    for alib, ahash in alist:
        config[akey][str(alib)] = str(ahash)

    with open(mem_file, "w") as fh_out:
        config.write(fh_out)


def parserprocess(libs: Sequence[str], load_dicts: bool = False):
    """
    Parse mapped libraries (SAM files) in parallel.

    Default: return only file paths to avoid RAM blow-ups.
    If load_dicts=True, load them back SEQUENTIALLY (low peak RAM).

    Spawn-safe: reads runtime knobs from rt.* and does not rely on legacy globals.

    Note: includes a small correctness fix vs the historical legacy implementation:
    when 'mismat' changes, we reparse the corresponding *.sam files (not the *.fas paths).
    """
    print("#### Fn: Lib Parser ##########################")

    mem_file = _resolve_mem_file()
    phase = str(getattr(rt, "phase", ""))
    maxhits = int(getattr(rt, "maxhits", 0))
    mismat = int(getattr(rt, "mismat", 0))

    libs_to_parse: List[str] = []

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(mem_file)
    updatedsetL = updatedsets(config)

    if "mismat" in updatedsetL:
        print("Setting update detected for 'mismat' parameter")
        libs_to_parse = [f"{lib.rpartition('.')[0]}.sam" for lib in list(libs)]
    elif config.has_section("PARSED"):
        print("Subsequent run for parserprocess; parsing only remapped libraries")
        parsekeys = list(config["PARSED"].keys())
        for alib in libs:
            blib = "%s_%s.dict" % (alib.rpartition(".")[0], phase)
            if blib in parsekeys:
                xfiles = [k for k in parsekeys if k.rpartition(".")[0] == blib.rpartition(".")[0]]
                if xfiles:
                    _, bmd5 = getmd5(xfiles[0])
                    if bmd5 == config["PARSED"][xfiles[0]]:
                        print(f"MD5 matches for previously parsed library {alib.rpartition('.')[0]}")
                        continue
                    else:
                        toAppend = f"{alib.rpartition('.')[0]}.sam"
                        print(f"Added {toAppend} to libs_to_parse")
                        libs_to_parse.append(toAppend)
            else:
                toAppend = f"{alib.rpartition('.')[0]}.sam"
                print(f"Added {toAppend} to libs_to_parse")
                libs_to_parse.append(toAppend)
    else:
        libs_to_parse = [f"{lib.rpartition('.')[0]}.sam" for lib in list(libs)]

    dict_paths: List[str] = []
    count_paths: List[str] = []

    if libs_to_parse:
        print(f"Libraries to be parsed: {', '.join(libs_to_parse)}")
        rawinputs = [(alib, maxhits, mismat) for alib in libs_to_parse]

        out_pairs = run_parallel_with_progress(
            samparser_streaming, rawinputs, desc="Parsing SAM", unit="lib"
        )

        for dp, cp in out_pairs:
            dict_paths.append(dp)
            count_paths.append(cp)

        dicthashes = run_parallel_with_progress(getmd5, dict_paths, desc="MD5 dict")
        counterhashes = run_parallel_with_progress(getmd5, count_paths, desc="MD5 count")
        libstoset(dicthashes, "PARSED")
        libstoset(counterhashes, "COUNTERS")

        libs_nestdict = []
        libs_poscountdict = []
        if load_dicts:
            for dp, cp in zip(dict_paths, count_paths):
                with open(dp, "rb") as f1:
                    obj = pickle.load(f1)
                    if isinstance(obj, dict):
                        libs_nestdict.append(obj)
                with open(cp, "rb") as f2:
                    obj = pickle.load(f2)
                    if isinstance(obj, dict):
                        libs_poscountdict.append(obj)
                gc.collect()
        else:
            libs_nestdict, libs_poscountdict = dict_paths, count_paths
    else:
        dict_paths = [f"{alib.rpartition('.')[0]}_{phase}.dict" for alib in libs]
        count_paths = [f"{alib.rpartition('.')[0]}_{phase}.count" for alib in libs]
        if load_dicts:
            libs_nestdict = []
            libs_poscountdict = []
            for dp, cp in zip(dict_paths, count_paths):
                try:
                    with open(dp, "rb") as f1:
                        obj = pickle.load(f1)
                        if isinstance(obj, dict):
                            libs_nestdict.append(obj)
                    with open(cp, "rb") as f2:
                        obj = pickle.load(f2)
                        if isinstance(obj, dict):
                            libs_poscountdict.append(obj)
                except FileNotFoundError:
                    print(f"Warning: Missing parsed file for {dp}")
                gc.collect()
        else:
            libs_nestdict, libs_poscountdict = dict_paths, count_paths

    return libs_nestdict, libs_poscountdict


def samparser_streaming(aninput):
    """
    Parse one SAM -> write:
      - <lib>_<phase>.dict (pickle of nestdict)
      - <lib>_<phase>.count (pickle of poscountdict)
    Return only (outfile1, outfile2) to keep RAM low.
    """
    alib, maxhits, mismat = aninput

    phase = str(getattr(rt, "phase", ""))
    norm = bool(getattr(rt, "norm", False))
    norm_factor = float(getattr(rt, "norm_factor", 0.0))

    outfile1 = f"{alib.rpartition('.')[0]}_{phase}.dict"
    outfile2 = f"{alib.rpartition('.')[0]}_{phase}.count"
    asum = f"{alib.rpartition('.')[0]}.sum"

    total_abund = None
    if norm:
        total_abund = 0
        with open(alib, "r") as fh:
            for line in fh:
                if line.startswith("@"):
                    continue
                ent = line.rstrip("\n").split("\t")
                aflag = int(ent[1])
                if aflag not in {0, 256, 16, 272}:
                    continue
                aname = ent[0].strip()
                aabun = int(aname.split("|")[-1])
                total_abund += aabun

    tempdict1 = defaultdict(list)
    posdict = defaultdict(list)

    reads_passed = 0
    with open(alib, "r") as fh:
        for line in fh:
            if line.startswith("@"):
                continue
            ent = line.rstrip("\n").split("\t")
            aflag = int(ent[1])
            if aflag not in {0, 256, 16, 272}:
                continue

            aname = ent[0].strip()
            achr = ent[2]
            apos = int(ent[3])
            atag = ent[9].strip()
            alen = len(atag)
            aabun = int(aname.split("|")[-1])
            astrand = "w" if aflag in {0, 256} else "c"
            try:
                amismat = int(ent[-7].rpartition(":")[-1])
                ahits = int(ent[-1].rpartition(":")[-1])
            except Exception:
                continue

            if ahits < maxhits and amismat <= mismat:
                reads_passed += 1
                anid = make_akey(lib_stem(alib), achr)

                adj_abun = aabun
                if norm and total_abund and total_abund > 0:
                    adj_abun = max(round((aabun / total_abund) * norm_factor), 1)

                taginfo = [achr, astrand, ahits, atag, aname, apos, alen, adj_abun]
                tempdict1[anid].append((apos, taginfo))
                posdict[anid].append(apos)

    nestdict = defaultdict(list)
    for akey, aval in tempdict1.items():
        tmp = defaultdict(list)
        for p, tinfo in aval:
            tmp[p].append(tinfo)
        nestdict[akey].append(tmp)

    poscountdict = {
        akey: OrderedDict(sorted(Counter(aval).items(), key=lambda x: int(x[0])))
        for akey, aval in posdict.items()
    }

    with open(outfile1, "wb") as f1:
        pickle.dump(nestdict, f1, protocol=pickle.HIGHEST_PROTOCOL)
    with open(outfile2, "wb") as f2:
        pickle.dump(poscountdict, f2, protocol=pickle.HIGHEST_PROTOCOL)
    with open(asum, "a") as fsum:
        fsum.write(f"Reads passed filters for {alib}:\t{reads_passed}\n")

    del tempdict1, posdict, nestdict, poscountdict
    gc.collect()

    return outfile1, outfile2


def lib_stem(p: str) -> str:
    """'.../ALL_LIBS.sam' -> 'ALL_LIBS'; 'F_1.tag' -> 'F_1'."""
    return os.path.splitext(os.path.basename(p))[0]


def make_akey(lib_id: str, chr_id) -> str:
    """Consistent akey constructor."""
    return f"{lib_id}-{chr_id}"
