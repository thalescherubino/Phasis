import os
import time

from phasis import runtime as rt
from phasis.cache import MEM_FILE_DEFAULT, read_mem_verbose


# Stage-local globals (only what getindex needs)
reference = None
outdir = None
memFile = MEM_FILE_DEFAULT
ncores = None


def sync_from_runtime() -> None:
    """
    Populate indexing-stage globals from phasis.runtime.
    Keep minimal and spawn-safe.
    """
    global reference, outdir, memFile, ncores

    reference = rt.reference
    outdir = rt.outdir
    ncores = rt.cores

    if outdir:
        outdir_abs = os.path.abspath(os.path.expanduser(outdir))
        if outdir_abs != outdir:
            outdir = outdir_abs
            rt.outdir = outdir_abs
        os.makedirs(outdir, exist_ok=True)

    mem_override = getattr(rt, "memFile", None)
    if mem_override:
        memFile = mem_override
    else:
        if outdir:
            memFile = os.path.join(outdir, MEM_FILE_DEFAULT)
        else:
            memFile = MEM_FILE_DEFAULT
        rt.memFile = memFile


def getindex(
    fh_run,
    *,
    indexIntegrityCheck_fn,
    indexBuilder_fn,
    compute_md5_str_fn,
):
    """
    Stage version of legacy.getindex(fh_run), preserving behavior/prints.

    Mem-file I/O now lives in cache.py; low-level index helpers remain injected.
    """
    global reference, memFile, ncores

    sync_from_runtime()

    # check if index exists
    if not os.path.isfile(memFile):
        print("This is first run - create index")
        indexflag = False  # index will be made on fly
    else:
        memflag, index, mem = read_mem_verbose(memFile)

        if memflag is False:
            print("Memory file is empty - seems like previous run crashed")
            print("Creating index")
            indexflag = False  # index will be made on fly
        else:
            exist_ref_hash = str(mem.genomehash) if mem.genomehash is not None else ""

            # valid memory file detected - use existing index if hashes/integrity match
            current_ref_hash = compute_md5_str_fn(reference) or ""

            if current_ref_hash != exist_ref_hash:
                print("Index status                     : Re-make")
                indexflag = False
                print("Existing index does not matches specified genome - It will be recreated")
            elif not index:
                print("Index status                     : Re-make")
                indexflag = False
                print("Existing index path missing from memory file - It will be recreated")
            else:
                indexIntegrity, indexExt = indexIntegrityCheck_fn(index)
                _ = indexExt  # parity / debug value, not otherwise used here

                if indexIntegrity:
                    print("Index status                     : Re-use")
                    genoIndex = index
                    indexflag = True
                    fh_run.write("Indexing Time: 0s\n")
                else:
                    print("Index status                     : Re-make")
                    indexflag = False  # index will be made on fly

    if indexflag is False:
        tstart = time.time()
        genoIndex = indexBuilder_fn(reference, ncores)
        tend = time.time()
        fh_run.write("Indexing Time:%ss\n" % (round(tend - tstart, 2)))

    print("Index to be used:%s" % (genoIndex))
    return genoIndex
