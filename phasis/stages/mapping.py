import configparser
import os
import subprocess
import sys
import time

from phasis import runtime as rt
from phasis.cache import MEM_FILE_DEFAULT, getmd5
from phasis.parallel import PPBalance, optimize, run_parallel_with_progress


# Stage-local globals (populated by sync_from_runtime)
mismat = None
maxhits = None
clustbuffer = None
phase = None
runtype = None
outdir = None
memFile = MEM_FILE_DEFAULT


def sync_from_runtime() -> None:
    """
    Populate mapping-stage globals from phasis.runtime.
    Keep this minimal: only values needed by updatedsets/mapper/mapprocess.
    """
    global mismat, maxhits, clustbuffer, phase, runtype, outdir, memFile

    mismat = rt.mismat
    maxhits = rt.maxhits
    clustbuffer = rt.clustbuffer
    phase = rt.phase
    runtype = rt.runtype
    outdir = rt.outdir

    # Ensure outdir exists even if stage is called directly
    if outdir:
        outdir_abs = os.path.abspath(os.path.expanduser(outdir))
        if outdir_abs != outdir:
            outdir = outdir_abs
            rt.outdir = outdir_abs
        os.makedirs(outdir, exist_ok=True)

    # Anchor memFile under outdir (same behavior as legacy)
    mem_override = getattr(rt, "memFile", None)
    if mem_override:
        memFile = mem_override
    else:
        if outdir:
            memFile = os.path.join(outdir, MEM_FILE_DEFAULT)
        else:
            memFile = MEM_FILE_DEFAULT
        rt.memFile = memFile


def updatedsets(config):
    """
    Checks which settings have been updated by comparing
    globals from settings file with entries in memfile.
    """
    updatedsetL = []

    if int(config["ADVANCED"]["mismat"]) != int(mismat):
        updatedsetL.append("mismat")
    if int(config["ADVANCED"]["maxhits"]) != int(maxhits):
        updatedsetL.append("maxhits")
    if int(config["ADVANCED"]["clustbuffer"]) != int(clustbuffer):
        updatedsetL.append("clustbuffer")

    # Settings that may not exist in early phases of analyses
    if config["BASIC"].getboolean("phaselen"):
        if int(config["BASIC"]["phaselen"]) != int(phase):
            updatedsetL.append("phaselen")

    return updatedsetL


def mapper(aninput):
    """
    Function to map individual files using HISAT2 and sort output with Samtools.
    Removes headers after sorting.
    """
    alib, genoIndex, nspread, maxhits_local, runtype_local = aninput

    asam_temp = f"{alib.rpartition('.')[0]}.temp.sam"
    asam_sorted = f"{alib.rpartition('.')[0]}.sorted.sam"
    asam_final = f"{alib.rpartition('.')[0]}.sam"
    asum = f"{alib.rpartition('.')[0]}.sum"
    nspread = str(nspread)

    if runtype_local == "G" or runtype_local == "S":
        retcode = subprocess.call(
            [
                "hisat2",
                "--no-softclip",
                "--no-spliced-alignment",
                "-k",
                str(maxhits_local),
                "-p",
                nspread,
                "-x",
                genoIndex,
                "-f",
                alib,
                "-S",
                asam_temp,
                "--summary-file",
                asum,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif runtype_local == "T":
        retcode = subprocess.call(
            [
                "hisat2",
                "--no-softclip",
                "--no-spliced-alignment",
                "-k",
                str(maxhits_local),
                "-p",
                nspread,
                "-x",
                genoIndex,
                "-f",
                alib,
                "-S",
                asam_temp,
                "--summary-file",
                asum,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print("Please input the correct setting for 'runtype' parameter in 'phasis.set' file")
        print("Script will exit for now\n")
        sys.exit()

    if retcode != 0:
        print(f"Error: HISAT2 mapping of '{alib}' to reference index failed.")
        sys.exit()

    retcode = subprocess.call(
        ["samtools", "sort", "-@", str(nspread), "-o", asam_sorted, asam_temp],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if retcode != 0:
        print(f"Error: Samtools sorting of '{asam_temp}' failed.")
        sys.exit()

    retcode = subprocess.call(
        ["samtools", "view", "-@", str(nspread), "-o", asam_final, asam_sorted],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if retcode != 0:
        print(f"Error: Removing headers from '{asam_sorted}' failed.")
        sys.exit()

    subprocess.call(["rm", asam_temp, asam_sorted])

    return asam_final


def mapprocess(
    libs,
    genoIndex,
    *,
    ncores_local,
):
    """
    Map the libs to reference index and update settings.

    INPUT: libs are paths (usually *.fas from libraryprocess; merged in --concat_libs)
    OUTPUT: list of mapped SAM files for the libs that required (re)mapping

    Guarantees:
      - [FASTAS] and [MAPS] end this step with non-empty MD5s when files exist.
      - Remaps if the .fas changed OR the .sam is missing/mismatched OR mem has blank MD5.
    """
    global maxhits, runtype

    sync_from_runtime()

    print("#### Fn: Lib Mapper ##########################")

    bases = [alib.rpartition(".")[0] for alib in libs]
    fas_inputs = [f"{b}.fas" for b in bases]
    sams_expected = [f"{b}.sam" for b in bases]

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)

    # Keep parity (even if currently unused in this function)
    updatedsetL = updatedsets(config)
    _ = updatedsetL

    for sect in ("MAPS", "FASTAS"):
        if not config.has_section(sect):
            config.add_section(sect)

    # Current FASTA md5s
    current_fas_md5 = {}
    for fas in fas_inputs:
        if os.path.isfile(fas):
            _, md5 = getmd5(fas)
            current_fas_md5[fas] = md5 or ""

    # Existing SAM md5s
    computed_sam_md5 = {}
    sams_to_check = [s for s in sams_expected if os.path.isfile(s)]
    if sams_to_check:
        md5_results = run_parallel_with_progress(
            getmd5, sams_to_check, desc="Checking existing SAM hashes", unit="lib"
        )
        for path, md5 in md5_results:
            computed_sam_md5[path] = md5 or ""

    libs_to_map = []
    for fas_path, sam_path in zip(fas_inputs, sams_expected):
        fas_prev = (config["FASTAS"].get(fas_path) or "").strip()
        fas_cur = current_fas_md5.get(fas_path, "")

        # best-effort cache stabilization
        if fas_cur and fas_prev != fas_cur:
            config["FASTAS"][fas_path] = fas_cur

        fas_changed = (not fas_prev) or (not fas_cur) or (fas_prev != fas_cur)
        sam_missing = not os.path.isfile(sam_path)
        sam_prev = (config["MAPS"].get(sam_path) or "").strip()
        sam_cur = computed_sam_md5.get(sam_path, "")
        sam_mismatch_or_blank = (not sam_prev) or (not sam_cur) or (sam_prev != sam_cur)

        if fas_changed or sam_missing or sam_mismatch_or_blank:
            libs_to_map.append(fas_path)

    if libs_to_map:
        print("Libraries to be mapped: %s" % (", ".join(libs_to_map)))

        nproc, nspread = optimize(ncores_local, len(libs_to_map))

        rawinputs = [(alib, genoIndex, nspread, maxhits, runtype) for alib in libs_to_map]
        PPBalance(mapper, rawinputs, n_workers=nproc)

        libs_mapped = [f"{alib.rpartition('.')[0]}.sam" for alib in libs_to_map]

        # Wait for filesystem stabilization before hashing
        for sam_path in libs_mapped:
            tries = 0
            last_size = -1
            while tries < 3:
                if os.path.isfile(sam_path):
                    try:
                        sz = os.path.getsize(sam_path)
                    except Exception:
                        sz = -1
                    if sz > 0 and sz == last_size:
                        break
                    last_size = sz
                time.sleep(0.5)
                tries += 1

        sam_md5s = run_parallel_with_progress(
            getmd5, libs_mapped, desc="Hashing mapped SAMs", unit="lib"
        )
        for sam_path, md5 in sam_md5s:
            if md5:
                config["MAPS"][sam_path] = md5
            else:
                print(f"[WARN] MD5 empty for {sam_path}; keeping blank (will force remap next run).")

        for fas in libs_to_map:
            _, fas_md5 = getmd5(fas)
            config["FASTAS"][fas] = fas_md5 or ""

        with open(memFile, "w") as fh:
            config.write(fh)

    else:
        libs_mapped = []
        print("\nNo new libraries to map this time")

        wrote_any = False
        for sam_path in sams_expected:
            if not os.path.isfile(sam_path):
                continue
            mem_md5 = (config["MAPS"].get(sam_path) or "").strip()
            if not mem_md5:
                _, cur_md5 = getmd5(sam_path)
                if cur_md5:
                    config["MAPS"][sam_path] = cur_md5
                    wrote_any = True

        for fas_path, cur in current_fas_md5.items():
            if cur and (config["FASTAS"].get(fas_path) or "").strip() != cur:
                config["FASTAS"][fas_path] = cur
                wrote_any = True

        if wrote_any:
            with open(memFile, "w") as fh:
                config.write(fh)

    return libs_mapped