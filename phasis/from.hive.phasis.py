#!/usr/bin/env python3

version = 'v 5.2'

## updated      :   2025/07/22
##                  Authors : Atul Kakrana,Thales H Cherubino Ribeiro, Blake C. Meyers
##                  Affilation  : Meyers Lab (Donald Danforth Plant Science Center, St. Louis, MO)
##                  License copy: Included and found at https://opensource.org/licenses/Artistic-2.0
#### IMPORTS ##############################################
import os
import sys
import threading
import shutil
import subprocess
import multiprocessing
import time
import collections
import argparse
import re
import configparser
import pickle
import datetime
import hashlib
from multiprocessing import Process, Queue, Pool, cpu_count
from collections import defaultdict, OrderedDict, Counter
from scipy.stats import hypergeom, mannwhitneyu, combine_pvalues
from os.path import expanduser
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, Bbox
import joblib
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
import gc
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import csv
import traceback
from typing import List, Sequence, Dict, Tuple, Any

parser = argparse.ArgumentParser()
reqflags        = parser.add_argument_group("required arguments")
#optionalflags   = parser.add_argument_group("optional arguments")

reqflags.add_argument('-libs',help= 'Quality controlled libraries to process',required=False, nargs='*')
reqflags.add_argument('-reference',help= 'Genome or transcriptome reference FASTA',required=False)

parser.add_argument('-maxhits', default=25, type=int, help='Number of genome or transcriptome hits to be processed and passed as -k to hisat2 [default 25]', required=False)
parser.add_argument('-runtype',  default='G', type=str, help='G: Running on whole genome | T: running on transcriptome | S: running on scaffolded genome [default G]', required=False)
parser.add_argument('-mindepth',  default=2, type=int, help='Minimum depth of sRNA to be considered for p-value computation [default 2]', required=False)
parser.add_argument('-force', action='store_true', help='Allow Phasis to run with resource-intensive parameter combinations, which may cause system crashes', required=False)
parser.add_argument('-uniqueRatioCut',  default=0.2, type=float, help='The proportion of uniquely mapped reads to filter out a locus [default 0.2]', required=False)
parser.add_argument('-max_complexity',  default=0.3, type=float, help='The max complexity value of a locus [default 0.3]', required=False)
parser.add_argument('-mismat',  default=0, type=int, help='Number of mismatches allowed between sRNA and reference for mapping [default 0]', required=False)
parser.add_argument('-libformat',  default='F', type=str, help='Quality controlled format: FASTA (F) or Tag count (T) [default F]', required=False)
parser.add_argument('-phase', default=21, type=int, help="Desired phase length to use for prediction [default 21]", required=False)
parser.add_argument('-clustbuffer', default=300, type=int, help="Maximum distance between two clusters, otherwise they will be merged [default 300]", required=False)
parser.add_argument('-phasisScoreCutoff',default=50, type=int, help="Minimum score to report a PHAS locus [default for 21-PHAS: 50 | default for 24-PHAS: 250]", required=False)
parser.add_argument('-minClusterLength',default=350, type=int, help="Minimum length to score a PHAS locus [default 350]", required=False)
parser.add_argument('-cores', default=0, type=int, help="Number of cores to use for analysis. 0: for most of the free cores | INTEGER (>0): to specify exact cores to use [default 0]", required=False)
parser.add_argument('-norm', action='store_true', help='Allow Phasis to perform read count normalization (CP10M) [default False]', required=False)
parser.add_argument('-norm_factor', type=float, default=1e7, help='Optional: Set a specific normalization factor (default is 1e7 for CP10M)', required=False)
parser.add_argument('-classifier',  default='KNN', type=str, help='Use a pre-trained K Nearest Neighbors classifier (KNN) | Use an unsupervised Gaussian mixture model [default KNN]', required=False)
parser.add_argument('-cleanup', action='store_true', help='Delete intermediate directories and files (not recommended for successive runs)', required=False)
parser.add_argument('-steps', default='both', help='run the cluster detection and classifier [default] | run only cluster detection [cfind] | run only classification [class]', required=False)
parser.add_argument('-class_cluster_file', help='Cluster file name(s) for -steps class', nargs='*')
parser.add_argument('-version', action='version',help = "Print the version and then quit", version='%(prog)s ' + version)
parser.add_argument('-min_Howell_score', type=float, default=12.5, help='Minimum Phased (Howell) score', required=False)
parser.add_argument( '--concat_libs', dest='concat_libs', action='store_true', required=False, help='Concatenate all input libraries into a single virtual library before mapping (default: off).' )
parser.add_argument('--outdir', dest='outdir', metavar='DIR', type=str, required=False, default="{phase}_results", help='Directory to write all outputs; if omitted, defaults to {phase}_results (created if missing).')

args = parser.parse_args()

global libs
global reference
global norm
global norm_factor
global maxhits
global runtype
global mindepth
global uniqueRatioCut
global mismat
global libformat
global phase
global clustbuffer
global phasisScoreCutoff
global minClusterLength
global window_len
global sliding
global cores
global classifier
global steps
global class_cluster_file
global max_complexity
global ncores
global min_Howell_score
global concat_libs
global outdir
 
libs = args.libs
reference = args.reference
norm = args.norm
norm_factor = args.norm_factor
maxhits = args.maxhits
runtype = args.runtype
mindepth = args.mindepth
uniqueRatioCut = args.uniqueRatioCut
max_complexity = args.max_complexity
mismat = args.mismat
libformat = args.libformat
phase = args.phase
clustbuffer = args.clustbuffer
phasisScoreCutoff = args.phasisScoreCutoff
minClusterLength = args.minClusterLength
cores = args.cores
classifier = args.classifier
steps = args.steps
class_cluster_file = args.class_cluster_file
min_Howell_score = args.min_Howell_score
concat_libs = args.concat_libs
outdir = args.outdir

if not outdir or outdir == "{phase}_results":
    outdir = f"{phase}_results"

# normalize + create
outdir = os.path.abspath(os.path.expanduser(outdir))
os.makedirs(outdir, exist_ok=True)

if (classifier != "KNN") and (classifier != "GMM"):
    print("\nERROR: Wrong classifier option")
    sys.exit()
else: pass

if steps not in ['both', 'cfind', 'class']:
    print("Error: Invalid value for -steps. Possible values: 'both', 'cfind', 'class'")
    exit(1)

if phase > 21:
    window_len = 26
    sliding = 8

    # Clamp phasisScoreCutoff to [250, 300] for longer phasing
    _min_cutoff, _max_cutoff = 250, 300
    _orig = phasisScoreCutoff
    if phasisScoreCutoff < _min_cutoff:
        phasisScoreCutoff = _min_cutoff
        print(f"[INFO] phase={phase} > 21: phasisScoreCutoff raised from {_orig} to {phasisScoreCutoff} (min {_min_cutoff}).")
    elif phasisScoreCutoff > _max_cutoff:
        phasisScoreCutoff = _max_cutoff
        print(f"[INFO] phase={phase} > 21: phasisScoreCutoff lowered from {_orig} to {phasisScoreCutoff} (max {_max_cutoff}).")
    else:
        print(f"[INFO] phase={phase} > 21: phasisScoreCutoff kept at {phasisScoreCutoff} (within [{_min_cutoff}, {_max_cutoff}]).")

else:
    window_len = 23
    sliding = 5

# === Phase II cache and naming helpers (concat-aware, param-aware, ConfigParser-safe) ===
from hashlib import md5 as _md5

def _md5_of_list_str(items):
    h = _md5()
    for s in items:
        h.update(str(s).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()

def _run_signature_keyprefix():
    # No '=' or ':' in the signature to keep ConfigParser option names safe.
    try:
        _ph = str(phase)
    except Exception:
        _ph = "NA"
    try:
        _c = "1" if bool(concat_libs) else "0"
    except Exception:
        _c = "0"
    try:
        _lib_names = [os.path.basename(x or "") for x in (libs or [])]
    except Exception:
        _lib_names = []
    libs_md5   = _md5_of_list_str(sorted(_lib_names))[:8]
    params_md5 = _md5_of_list_str([mindepth, maxhits, clustbuffer])[:8]
    return f"ph-{_ph}_c-{_c}_L-{libs_md5}_P-{params_md5}"

def phase2_basename(base_name:str)->str:
    try:
        is_concat = bool(concat_libs)
    except Exception:
        is_concat = False
    prefix = "concat_" if is_concat else ""
    return f"{prefix}{phase}_{base_name}"

def _mem_key_for_path(path:str)->str:
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

def mem_get(cfg, section, path):
    key = _mem_key_for_path(path)
    return cfg[section].get(key)

def mem_set(cfg, section, path, value):
    key = _mem_key_for_path(path)
    cfg[section][key] = str(value)
memFile         = "phasis.mem"
home            = expanduser("~")
#uniqueRatioCut  = 0.2


def checkLibs():
    '''
    Read libs file
    '''
    ## Sanity check
    notfound = []
    for alibs in libs:
        if fileexists(alibs) == False:
            notfound.append(alibs)
    if notfound:
        print("\nERROR:These sRNA libraries not found   : %s" % (",".join(notfound)))
        print("------Please check file exists at specified location")
        sys.exit()

    if fileexists(reference) == False:
        print("\nERROR:Reference genome or transcriptome not found:%s" % (reference))
        print("------Please check file exists at specified location")
        sys.exit()
    return libs

## ADVANCED SETTINGS ######################################
UNIQRATIO_HIT   = 2             ## number of hits cutoff to consider sRNA as multihit for computing uniqness ratio of cluster
DOMSIZE_CUT     = 0.50          ## among all sRNAs, the user defined size should be more abundant than this cutoff
WINDOW_SIZE     = 15            ## Arbitrary window size;
                                ## alternatively you can compute max phases from start and end coords
###########################################################
def make_pool(nworkers: int):
    """
    Pool with safer defaults to limit RAM spikes.
    - maxtasksperchild=1 curbs per-worker memory growth
    - single-threaded BLAS via env vars
    - prefer 'fork' context on POSIX
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    ctx = multiprocessing.get_context("fork") if hasattr(multiprocessing, "get_context") else multiprocessing
    return ctx.Pool(processes=int(max(1, nworkers)), maxtasksperchild=1)

def safe_worker(args):
    """Run func(arg), catching exceptions; return RuntimeError sentinel on failure."""
    func, arg = args
    try:
        return func(arg)
    except Exception as e:
        import traceback  # allowed here (small, unavoidable for nice trace)
        return RuntimeError(f"Error in {func.__name__} with arg={arg}: {e}\n{traceback.format_exc()}")

def _compute_initial_chunk_size(n_data: int, ncores_local: int, unit: str, min_chunk: int, batch_factor: float):
    print(f"batch factor set to {batch_factor}")
    if unit == "lib":
        worker_cap_for_lib = 20  # conservative start for lib-level work
        return min(ncores_local, worker_cap_for_lib) or 1
    if n_data <= ncores_local:
        print("n_data <= ncores_local: return 1")
    #    return 1
    n_batches = int(ncores_local * batch_factor) or 1
    print(f"n_data is {n_data}")
    print(f"n_batches set to {n_batches}")
    chunk_size = max(min_chunk, int(n_data / n_batches), int(ncores_local))
    print(f" Initial chunk_size set to {chunk_size}")
    if n_data > 300:
        print("n_data > 300")
        max_chunk_size = max(min_chunk, int(ncores_local))
        chunk_size = max(chunk_size, max_chunk_size)
        print(f" Initial chunk_size set to {chunk_size}")
    return max(1, chunk_size)

def run_parallel_with_progress(
    func,
    data,
    desc=None,
    min_chunk=1,
    batch_factor=0.1,
    unit="lib",
    on_result=None,        # Optional: callable(result) -> None (avoid storing results)
    return_results=True    # If False and on_result provided, we won’t keep a results list
):
    """
    Parallel, streaming, and adaptive:
      - Streams results via imap_unordered (low peak memory).
      - On any pool failure, automatically retries the current slice with
        smaller (chunk_size, nworkers): [proposed] -> 10 -> 5 -> 1; workers n->8->4->2->1.
      - maxtasksperchild=1 to fight per-worker RSS growth.
      - BLAS single-threaded to avoid hidden fan-out.

    Tips:
      * If results are large, pass an `on_result` consumer and set return_results=False.
      * Keep `chunksize=1` to avoid big internal queues in the pool.
    """
    global ncores

    n_data = len(data)
    if n_data == 0:
        return []

    if ncores is None or ncores <= 0:
        ncores = multiprocessing.cpu_count()

    # Initial chunk size & workers
    chunk_size = _compute_initial_chunk_size(n_data, ncores, unit, min_chunk, batch_factor)
    print(f"initial chunk size set to {chunk_size}")
    nworkers = min(ncores, chunk_size) or 1

    # Decide whether to accumulate results or stream-only
    keep_results = (on_result is None) or return_results
    results = [] if keep_results else None

    i = 0
    with tqdm(total=n_data, desc=desc, unit=unit) as pbar:
        while i < n_data:
            start = i
            end = min(i + chunk_size, n_data)
            proposed = end - start if end > start else 1

            # Build retry ladder for sizes
            try_sizes = []
            for s in (proposed, 16, 12, 10, 8, 4, 2, 1):
                s = int(max(1, min(s, n_data - start)))
                if s not in try_sizes:
                    try_sizes.append(s)

            slice_completed = False
            last_exception = None

            for local_chunk_size in try_sizes:
                end = min(start + local_chunk_size, n_data)
                chunk = data[start:end]

                # Worker trials, decreasing
                worker_trials = []
                for w in (nworkers,16, 12, 10, 8, 4, 2, 1):
                    w = int(max(1, min(w, local_chunk_size, ncores)))
                    if w not in worker_trials:
                        worker_trials.append(w)

                for nw in worker_trials:
                    # Try streaming this chunk with nw workers
                    try:
                        with make_pool(nw) as pool:
                            # Stream results; avoid big intermediate lists
                            for res in pool.imap_unordered(safe_worker, ((func, arg) for arg in chunk), chunksize=1):
                                if isinstance(res, RuntimeError):
                                    # Retry failed item serially for deterministic logging
                                    idx = None  # only for clarity; we stream, so idx is not needed
                                    retry_res = safe_worker((func, chunk[0])) if False else res  # no-op placeholder
                                    # The safe pattern: rerun the actual arg serially
                                    # We don't have the arg here anymore; so re-run serially by index.
                                    # To keep memory low, do a small serial retry immediately:
                                    if hasattr(res, 'args') and res.args:
                                        # We encoded arg in the message, but parsing isn't robust; better to rerun by value
                                        pass
                                    # Safer: ignore this and do exact serial retry below with a tiny loop:
                                    if on_result is not None and return_results is False:
                                        # We'll handle retry after the pool closes below
                                        pass

                                # Normal path: consume result
                                if isinstance(res, RuntimeError):
                                    # Serial retry for the specific arg (exactly), one by one
                                    # Find the original arg by popping from the front—safe because chunksize=1 maps 1:1
                                    # Here we can't know which arg it was due to unordered mapping; do explicit serial pass:
                                    # Minimal overhead since failures should be rare.
                                    for arg in chunk:
                                        retry = safe_worker((func, arg))
                                        if not isinstance(retry, RuntimeError):
                                            if on_result: on_result(retry)
                                            if keep_results: results.append(retry)
                                        else:
                                            # Still failing — log and keep the sentinel
                                            print(f"[ERROR] Serial retry failed for arg: {arg}\n{retry}")
                                            if on_result: on_result(retry)
                                            if keep_results: results.append(retry)
                                    # Break out of this chunk; move to next slice
                                    break
                                else:
                                    if on_result:
                                        on_result(res)
                                    if keep_results:
                                        results.append(res)
                                    pbar.update(1)

                            # If we reached here without exceptions, the chunk is done
                            slice_completed = True

                        # Adopt smaller settings if they worked
                        if local_chunk_size < chunk_size:
                            chunk_size = local_chunk_size
                            print(f"[INFO] Lowering ongoing chunk size to {chunk_size}.")
                        if nw < nworkers:
                            nworkers = nw
                            print(f"[INFO] Lowering worker count to {nworkers}.")

                        break  # worker_trials loop
                    except MemoryError as e:
                        last_exception = e
                        print(f"\n[WARN] MemoryError on slice [{start}:{end}] size={local_chunk_size}, nworkers={nw}. Trying smaller.\n")
                    except Exception as e:
                        last_exception = e
                        print(f"\n[WARN] Pool error on slice [{start}:{end}] size={local_chunk_size}, nworkers={nw}: {e}\nTrying smaller.\n")

                if slice_completed:
                    # Mark progress for any remaining items in this slice (if failures were handled serially we already updated)
                    remaining = (end - start) - 0  # all streamed accounted for
                    if remaining > 0:
                        pbar.update(remaining)
                    break  # size loop

            # If pool attempts all failed for this slice, do serial for this slice
            if not slice_completed:
                print(f"[WARN] Running slice [{start}:{end}] serially after pool failures.")
                for arg in data[start:end]:
                    res = safe_worker((func, arg))
                    if on_result:
                        on_result(res)
                    if keep_results:
                        results.append(res)
                pbar.update(end - start)

            # Advance window and do some housekeeping
            i = end
            gc.collect()

    return results if keep_results else None


def checkDependency():
    '''Checks for required components on user system'''
    print("#### Fn: checkLibs ###########################")
    goSignal  = True ### Signal to process is set to true
    ### Check PYTHON version
    pythonver = sys.version_info[0]
    if int(pythonver) >= 3:
        print("--Python v3.0 or higher          : found")
        pass
    else:
        print("--Python v3.0 or higher          : missing")
        goSignal    = False
    ### Check hisat
    is_hisat = shutil.which("hisat2")
    if is_hisat:
        print("--Hisat (v2)                     : found")
        pass
    else:
        print("--Hisat (v2)                     : missing")
        goSignal    = False
    if goSignal == False:
        print("-------Please install the missing libraries and rerun analysis")
        sys.exit()
    return None

def fileexists(afile):
    '''
    test if file exists
    '''
    print("checking if file exists:%s" % (afile))
    if os.path.isfile(afile):
        abool = True
    else:
        abool = False
    print(f"File available:{abool}")
    return abool

def match_pattern(filename, patterns):
    for pattern in patterns:
        if filename.endswith(pattern):
            return True
    return False

def cleanup():
    cleanup_patterns = ['fas', 'sam', 'dict', 'count', 'runtime', 'sum', 'count', 'scoredClusters', 'candidate.clusters', 'clusters']
    for root, dirs, files in os.walk('.'):
        for filename in files + dirs:
            if match_pattern(filename, cleanup_patterns):
                path = os.path.join(root, filename)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

def refClean(filename):
    '''
    Cleans FASTA file - multi-line fasta to single line, header clean, empty lines removal
    '''
    ## Read seqeunce file
    fh_in       = open(filename, 'r')
    print ("Phasis uses FASTA header as key for identifying the phased loci")
    print ("Caching '%s' reference FASTA file" % (filename))
    ## Write file  
    fastaclean = ('%s/%s.clean.fa' % (os.getcwd(),filename.rpartition('/')[-1].rpartition('.')[0])) 
    ### Outfiles
    fh_out1     = open(fastaclean, 'w')
    fastasumm   = ('%s/%s.summ.txt' % (os.getcwd(),filename.rpartition('/')[-1].rpartition('.')[0]))
    fh_out2     = open(fastasumm, 'w')
    fh_out2.write("Name\tLen\n")
    ### Read files
    fasta       = fh_in.read()
    fasta_splt  = fasta.split('>')
    acount      = 0     ## count the number of entries
    empty_count = 0
    for i in fasta_splt[1:]:
        ent     = i.split('\n')
        aname   = ent[0].split()[0].strip()
        if runtype == 'G':
            ## To match with phasing-core script for
            ## genome version which remov non-numeric and preceding 0s
            name = re.sub("[^0-9]", "", aname).lstrip('0')
        else:
            name = aname
        seq     = ''.join(x.strip() for x in ent[1:]) ## Sequence in multiple lines
        alen    = len(seq)
        if alen > 200:
            fh_out1.write('>%s\n%s\n' % (name,seq))
            fh_out2.write('%s\t%s\n' % (name,alen))
            acount+=1
        else:
            empty_count+=1
            pass
    fh_in.close()
    fh_out1.close()
    fh_out2.close()
    print("Fasta file with reduced header: '%s' with total entries %s is prepared" % (fastaclean, acount))
    print("There were %s entries found with empty sequences and were removed\n" % (empty_count))
    return fastaclean,fastasumm

def indexBuilder(reference,ncores):
    '''
    Generic index building module
    '''
    print ("#### Fn: indexBuilder #######################")
    ### Clean reference ################
    fastaclean,fastasumm = refClean(reference)
    ### Prepare Index ##################
    print ("**Deleting old index")
    shutil.rmtree('./index', ignore_errors=True)
    os.mkdir('./index')
    genoIndex   = '%s/index/%s' % (os.getcwd(),fastaclean.rpartition('/')[-1].rpartition('.')[0]) ## Can be merged with genoIndex from earlier part if we use bowtie2 earlier
    print('Creating index of cDNA/genomic sequences:%s**\n' % (genoIndex))
    ncores      = str(ncores)
    retcode     = subprocess.call(["hisat2-build","-p",ncores,"-f", fastaclean, genoIndex])
    print(retcode)
    if retcode == 0: ## The hisat mapping exit with status 0, all is well
        pass
    else:
        print("There is some problem preparing index of reference '%s'" %  (reference))
        print("Is Hisat2 installed? And added to environment variable?")
        print("Script will exit now")
        sys.exit()

    ### Make a memory file ###################
    fh_out      = open(memFile,'w')
    refHash     = (hashlib.md5(open('%s' % (reference),'rb').read()).hexdigest()) ### reference hash used instead of cleaned FASTA because while comparing only the user input reference is available
    print("Generating MD5 hash for HiSat2 index")
    if os.path.isfile("%s.1.ht2l" % (genoIndex)):
        indexHash   = (hashlib.md5(open('%s.1.ht2l' % (genoIndex),'rb').read()).hexdigest())
    elif os.path.isfile("%s.1.ht2" % (genoIndex)):
        indexHash   = (hashlib.md5(open('%s.1.ht2' % (genoIndex),'rb').read()).hexdigest())
    else:
        print("File extension for index couldn't be determined properly")
        print("It could be an issue from 'HiSat2'")
        print("This needs to be reported to 'PHASIS' developer, report issue here\nhttps://github.com/atulkakrana/PHASIS/issues")
        print("Script will exit")
        sys.exit()
    ## write to memory file
    ## follow format: https://docs.python.org/3.4/library/configparser.html
    config = configparser.ConfigParser()
    config['BASIC']     = {'timestamp'  : datetime.datetime.now().strftime("%m_%d_%H_%M"),
                            'genomehash': refHash,
                            'index'     : genoIndex,
                            'indexhash' : indexHash}
    config['ADVANCED']  = {'mindepth'       : mindepth,
                            'clustbuffer'   : clustbuffer,
                            'maxhits'       : maxhits,
                            'mismat'        : mismat}
    config.write(fh_out)
    fh_out.close()
    print("Index prepared:%s\n" % (genoIndex))
    return genoIndex

def isfasta(afile):
    '''
    test if file is fasta format
    '''
    fh_in       = open(afile,'r')
    firstline   = fh_in.readline()
    fh_in.close()
    if not firstline.startswith('>') and len(firstline.split('\t')) > 1:
        print("\nERROR: File '%s' doesn't seems to be a FASTA" % (afile))
        print("------Please provide correct setting for 'libformat' in 'phasis.set'")
        abool = False
    else:
        abool = True
    return abool

def isfiletagcount(afile):
    '''
    test if file is tab seprated tag and counts file
    '''
    fh_in       = open(afile,'r')
    firstline   = fh_in.readline()
    fh_in.close()
    if firstline.startswith('>') or len(firstline.split('\t')) != 2 :
        print("\nERROR: File '%s' doesn't seems to be tab-seprated tag-count format" % (afile))
        print("------Please provide correct setting for 'libFormat' in 'phasis.set'")
        abool = False
    else:
        abool = True
    return abool

def getindex(fh_run):
    '''
    this is higer level function, which checks if index needs
    to be generated or reused, memory files are written in
    this process
    '''
    ### check if index exists
    if not os.path.isfile(memFile):
        print("This is first run - create index")
        indexflag = False       ## index will be made on fly
    else:
        memflag,index   = readMem(memFile)
        if memflag  == False:
            print("Memory file is empty - seems like previous run crashed")
            print("Creating index")
            indexflag = False   ## index will be made on fly
        elif memflag  == True:
            ## valid memory file detected - use existing index
            currentRefHash = hashlib.md5(open('%s' % (reference),'rb').read()).hexdigest()
            if currentRefHash == existRefHash:
                indexIntegrity,indexExt = indexIntegrityCheck(index)
                if indexIntegrity:          ### os.path.isdir(index.rpartition('/')[0]):
                    print("Index status                     : Re-use")
                    genoIndex   = index
                    indexflag   = True
                    fh_run.write("Indexing Time: 0s\n")
                else:
                    print("Index status                     : Re-make")
                    indexflag   = False   ## index will be made on fly
            else:
                ## Different reference file - index will be remade
                print("Index status                     : Re-make")
                indexflag       = False
                print("Existing index does not matches specified genome - It will be recreated")

    if indexflag    == False:
        ## index will be remade,
        ## mem file will be initiated
        tstart      = time.time()
        genoIndex   = indexBuilder(reference,ncores)
        tend        = time.time()
        fh_run.write("Indexing Time:%ss\n" % (round(tend-tstart,2)))
    else:
        pass
    print("Index to be used:%s" % (genoIndex))
    return genoIndex

def indexIntegrityCheck(index):
    '''
    Checks the integrity of index and the extension
    '''
    indexFolder     = index.rpartition("/")[0]
    if os.path.isfile("%s.1.ht2l" % (index)): ## Check if this extension exists in folder
        indexExt    = "ht2l"
        indexFiles  = [i for i in os.listdir('%s' % (indexFolder)) if i.endswith('.ht2l')]
        if len(indexFiles) >= 6:
            indexIntegrity = True
    elif os.path.isfile("%s.1.ht2" % (index)):
        indexExt    = "ht2"
        indexFiles  = [i for i in os.listdir('%s' % (indexFolder)) if i.endswith('.ht2')]
        if len(indexFiles) >= 6:
            indexIntegrity = True
    else:
        print("Existing index extension couldn't be determined")
        print("Genome index will be remade")
        indexExt        = False
        indexIntegrity  = False
    return indexIntegrity,indexExt

def readMem(memFile):
    '''
    Reads memory file and gives global variables
    '''
    print ("#### Fn: memReader ############################")
    ## Read Config File
    config  = configparser.ConfigParser()
    config.read(memFile)
    basickeys   = [key for key in config['BASIC']]
    ## iitial state variables
    memflag     = True
    varcount    = 0
    ## Extract Values
    if 'genomehash' in basickeys:
        global existRefHash
        varcount        +=1
        existRefHash    = str(config['BASIC']['genomehash'])
        print('Existing reference hash          :',existRefHash)
    if 'indexhash' in basickeys:
        global existIndexHash
        varcount        +=1
        existIndexHash  = str(config['BASIC']['indexhash'])
        print('Existing index hash              :',existIndexHash)
    if 'index' in basickeys:
        global index
        varcount        +=1
        index           = str(config['BASIC']['index'])
        print('Existing index location          :',index)
    else:
        pass
    if varcount == 3:
        memflag = True
    else:
        memflag = False
    return memflag,index

def filter_process(alib):
    '''
    filter tag count file for mindepth, and write
    to FASTA
    '''
    #print("Writing filtered FASTA for %s" % (alib))
    asum = "%s.sum" % alib.rpartition('.')[0]    # Summary file
    countFile   = "%s.fas" % alib.rpartition('.')[0]  ### Writing in de-duplicated FASTA format
    fh_out      = open(countFile,'w')
    fh_in       = open(alib,'r')
    aread       = fh_in.readlines()
    bcount      = 0 ## tags written
    ccount      = 0 ## tags excluded
    seqcount    = 1 ## To name seqeunces
    for aline in aread:
        atag,acount    = aline.strip("\n").split("\t")
        if int(acount) >= int(mindepth):
            fh_out.write(">seq_%s|%s\n%s\n" % (seqcount,acount,atag))
            bcount      += 1
            seqcount    += 1
        else:
            ccount+=1
    #print("Library %s - tag written:%s | tags filtered:%s" % (alib,bcount,ccount))
    with open(asum, 'a') as fh_sum:
        fh_sum.write("Library %s - tag written:%s | tags filtered:%s\n" % (alib, bcount, ccount))
    fh_in.close()
    fh_out.close()
    return countFile

def dedup_process(alib):
    '''
    To parallelize the process
    '''
    print("#### Fn: De-duplicater #######################")
    afastaL     = dedup_fastatolist(alib)         ## Read
    acounter    = deduplicate(afastaL )           ## De-duplicate
    fastafile   = dedup_writer(acounter,alib)     ## Write
    return fastafile

def dedup_fastatolist(alib):
    '''
    New FASTA reader
    '''
    ## Output
    fastaL      = [] ## List that holds FASTA tags
    ## input
    fh_in       = open(alib,'r')
    print("Reading FASTA file:%s" % (alib))
    read_start  = time.time()
    acount      = 0
    empty_count = 0
    for line in fh_in:
        if line.startswith('>'):
            seq = ''
            pass
        else:
          seq = line.rstrip('\n')
          fastaL.append(seq)
          acount += 1
    read_end    = time.time()
    print("Cached file: %s | Tags: %s | Empty headers: %ss" % (alib,acount,empty_count))
    fh_in.close()
    return fastaL

def deduplicate(afastaL):
    '''
    De-duplicates tags using multiple processes and libraries using multiple cores
    '''
    dedup_start = time.time()
    acounter    = collections.Counter(afastaL)
    dedup_end   = time.time()
    return acounter

def dedup_writer(acounter,alib):
    '''
    filter tag counts for 'mindepth' parameter, writes a dict
    pickle and filtered fasta file
    '''
    print("Writing filtered FASTA for %s" % (alib))
    sumFile = "%s.sum" % alib.rpartition('.')[0]    # Summary file

    countFile   = "%s.fas" % alib.rpartition('.')[0]  ### Writing in de-duplicated FASTA format as required for phaster-core
    fh_out      = open(countFile,'w')
    wcount      = 0 ## tags written
    bcount      = 0 ## tags excluded
    seqcount    = 1 ## To name seqeunces
    for atag,acount in acounter.items():
        if int(acount) >= int(mindepth):
            fh_out.write(">seq_%s|%s\n%s\n" % (seqcount,acount,atag))
            wcount      += 1
            seqcount    += 1
        else:
            bcount+=1
    with open(sumFile, 'w') as fh_sum:
        fh_sum.write("Library %s - tag written:%s | tags filtered:%s\n" % (alib, wcount, bcount))
    #print("Library %s - tag written:%s | tags filtered:%s" % (alib,wcount,bcount))
    fh_out.close()
    return countFile

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


def libstoset(alist,akey):
    '''
    write library info to settings file
    '''
    fh_out = open(memFile,'a')
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)
    if config.has_section(akey):
        ## subsequent run, add libraries
        for anent in alist:
            # print("Entry:",anent)
            alib,ahash = anent
            config[akey][alib] = ahash
    else:
        ## first run, make new section, and add libs
        config[akey] = {}
        for anent in alist:
            # print(anent)
            alib,ahash = anent
            config[akey][alib] = ahash
    ## write updated config
    fh_out = open(memFile,'w')
    config.write(fh_out)
    fh_out.close()
    return None
def fas_records(path):
    """
    Stream a .fas produced by your pipeline.
    Header: >seq_<n>|<count>
    Next line: <sequence>
    Yields (sequence, count:int).
    """
    with open(path, 'r') as fh:
        count_val = None
        for line in fh:
            if line.startswith('>'):
                # Example: >seq_123|45
                parts = line.split('|', 1)
                if len(parts) < 2:
                    raise ValueError(f"Malformed header in {path}: {line.strip()}")
                try:
                    count_val = int(parts[1].strip())
                except Exception:
                    raise ValueError(f"Non-integer count in {path}: {line.strip()}")
            else:
                seq = line.rstrip('\n')
                if not seq:
                    continue
                if count_val is None:
                    raise ValueError(f"Sequence without header in {path}")
                yield (seq, count_val)
                count_val = None

def write_merged_fas(seq_counter, out_path, mindepth):
    """
    Write a merged .fas applying mindepth to merged totals.
    Also writes a .sum sidecar like your per-lib writers.
    """
    wcount = 0
    bcount = 0
    seqnum = 1
    with open(out_path, 'w') as out_fh:
        for seq, total in seq_counter.items():
            if int(total) >= int(mindepth):
                out_fh.write(f">seq_{seqnum}|{total}\n{seq}\n")
                wcount += 1
                seqnum += 1
            else:
                bcount += 1
    with open(f"{out_path.rpartition('.')[0]}.sum", 'w') as fh_sum:
        fh_sum.write(
            f"Merged library {os.path.basename(out_path)} - tags written:{wcount} | tags filtered:{bcount}\n"
        )
    return out_path

def merge_processed_fastas(fas_paths, out_dir, out_basename, mindepth):
    """
    Merge multiple .fas by summing counts per identical sequence.
    Returns the path to the merged .fas.
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_basename}.fas")

    counter = collections.Counter()
    for p in fas_paths:
        for seq, cnt in fas_records(p):
            counter[seq] += cnt

    write_merged_fas(counter, out_path, mindepth)
    return out_path

def compute_md5_str(path: str) -> str | None:
    """Chunked md5 -> hex string (or None on failure)."""
    try:
        import hashlib
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def libraryprocess(libs):
    """
    Checks mem file to see if libs have been processed earlier.
    Only processes new libs, testing format, converting/cleaning, and updating MD5.
    Returns: list of processed libraries (*.fas). If --concat_libs, returns [merged.fas].

    Changes:
      - Ensure and write MD5s for produced .fas files into new [FASTAS] section.
      - Keep existing [LIBRARIES] behavior unchanged.
    """
    print("#### Fn: Lib Processor #######################")

    libs_to_process = []
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)

    # Ensure sections exist
    for sect in ('ADVANCED', 'LIBRARIES', 'FASTAS'):
        if not config.has_section(sect):
            config.add_section(sect)

    # Determine which original libs need processing (MD5-based)
    if str(mindepth) == str(config['ADVANCED'].get('mindepth', "")) and config.has_section('LIBRARIES'):
        libkeys = [key for key in config['LIBRARIES']]
        for alib in libs:
            if alib in libkeys:
                cur = compute_md5_str(alib)
                prev = (config['LIBRARIES'].get(alib) or "").strip()
                if cur and prev and (cur == prev):
                    print(f"MD5 matches for library: {alib}")
                else:
                    print(f"MD5 doesn't match (or missing) for library: {alib}")
                    libs_to_process.append(alib)
            else:
                libs_to_process.append(alib)
    else:
        libs_to_process = libs.copy()

    libs_processed = []

    if libs_to_process:
        print("\nLibraries to be processed: %s" % (', '.join(libs_to_process)))

        # Step 1: Check library format (FASTA vs tag-count)
        check_func = isfasta if libformat == "F" else isfiletagcount
        print("Checking format:")
        abools = run_parallel_with_progress(
            check_func, libs_to_process, desc="Checking format"
        )
        if not any(abools):
            sys.exit("No libraries passed format check.")

        # Step 2: Convert or filter to .fas
        proc_func = dedup_process if libformat == "F" else filter_process
        print("Processing libraries:")
        libs_processed = run_parallel_with_progress(
            proc_func, libs_to_process, desc="Filtering/Converting"
        )

        # Step 3: Record MD5 of original files to memory (direct write)
        print("Recording MD5:")
        for alib in libs_to_process:
            md5 = compute_md5_str(alib) or ""
            config['LIBRARIES'][alib] = md5
        with open(memFile, 'w') as fh:
            config.write(fh)

    else:
        print("\nNo new libraries to process this time.")
        # Try to reuse existing .fas files that correspond to requested libs
        expected_fas = [f"{alib.rpartition('.')[0]}.fas" for alib in libs]
        libs_processed = [p for p in expected_fas if os.path.exists(p)]

    # Guard: filter out any Nones or missing files (just in case)
    libs_processed = [p for p in libs_processed if p and os.path.exists(p)]

    # === Concatenation path (sums counts across libs) ===
    if concat_libs:
        if not libs_processed:
            sys.exit("No processed libraries available to concatenate.")
        merged_dir = os.path.dirname(libs_processed[0]) or os.getcwd()
        merged_basename = "ALL_LIBS"
        merged_path = merge_processed_fastas(
            fas_paths=libs_processed,
            out_dir=merged_dir,
            out_basename=merged_basename,
            mindepth=mindepth
        )
        print(f"[concat_libs] Created merged library: {merged_path}")

        # Record MD5 for merged .fas in [FASTAS]
        fas_md5 = compute_md5_str(merged_path) or ""
        config['FASTAS'][merged_path] = fas_md5
        with open(memFile, 'w') as fh:
            config.write(fh)

        return [merged_path]

    # Non-concat: record MD5 for each produced .fas in [FASTAS]
    updated = False
    for fas in libs_processed:
        if fas.endswith(".fas") and os.path.isfile(fas):
            fas_md5 = compute_md5_str(fas) or ""
            if (config['FASTAS'].get(fas) or "") != fas_md5:
                config['FASTAS'][fas] = fas_md5
                updated = True
    if updated:
        with open(memFile, 'w') as fh:
            config.write(fh)

    # Default: return per-library .fas list
    return libs_processed


def mapper(aninput):
    '''
    Function to map individual files using HISAT2 and sort output with Samtools.
    Removes headers after sorting.
    '''
    alib, genoIndex, nspread, maxhits = aninput
    ## Output file names
    asam_temp = f"{alib.rpartition('.')[0]}.temp.sam"  # Temporary file with headers
    asam_sorted = f"{alib.rpartition('.')[0]}.sorted.sam"  # Sorted file with headers
    asam_final = f"{alib.rpartition('.')[0]}.sam"  # Final sorted file without headers
    asum = f"{alib.rpartition('.')[0]}.sum"
    nspread = str(nspread)

    if runtype == "G" or runtype == "S":
        retcode = subprocess.call(
        ["hisat2", "--no-softclip", "--no-spliced-alignment", "-k", str(maxhits),
         "-p", nspread, "-x", genoIndex, "-f", alib, "-S", asam_temp, "--summary-file", asum],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif runtype == "T":
        retcode = subprocess.call(
        ["hisat2", "--no-softclip", "--no-spliced-alignment", "-k", str(maxhits),
         "-p", nspread, "-x", genoIndex, "-f", alib, "-S", asam_temp, "--summary-file", asum],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("Please input the correct setting for 'runtype' parameter in 'phasis.set' file")
        print("Script will exit for now\n")
        sys.exit()

    ## Check HISAT2 return code
    if retcode != 0:
        print(f"Error: HISAT2 mapping of '{alib}' to reference index failed.")
        sys.exit()

    #print(f"Mapping for {alib} complete. Sorting output...")

   # Run samtools sort (multicore support) -- suppress all output
    retcode = subprocess.call(
    ["samtools", "sort", "-@", str(nspread), "-o", asam_sorted, asam_temp],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)

    if retcode != 0:
        print(f"Error: Samtools sorting of '{asam_temp}' failed.")
        sys.exit()

    # Remove headers from sorted file -- suppress all output
    retcode = subprocess.call(
    ["samtools", "view", "-@", str(nspread), "-o", asam_final, asam_sorted],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)

    ## Check final step return code
    if retcode != 0:
        print(f"Error: Removing headers from '{asam_sorted}' failed.")
        sys.exit()

    ## Cleanup: Remove temporary files
    subprocess.call(["rm", asam_temp, asam_sorted])

    #print(f"Sorting complete for {alib}. Final output: {asam_final}")

    return asam_final


def mapprocess(libs, genoIndex):
    """
    Map the libs to reference index and update settings.
    INPUT: libs are paths (usually *.fas from libraryprocess; merged in --concat_libs)
    OUTPUT: list of mapped SAM files for the libs that required (re)mapping

    Guarantees:
      - [FASTAS] and [MAPS] end this step with non-empty MD5s when files exist.
      - Remaps if the .fas changed OR the .sam is missing/mismatched OR mem has blank MD5.
    """
    global nproc, nspread
    print("#### Fn: Lib Mapper ##########################")

    bases         = [alib.rpartition(".")[0] for alib in libs]
    fas_inputs    = [f"{b}.fas" for b in bases]
    sams_expected = [f"{b}.sam" for b in bases]

    # Load memo/config
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)
    updatedsetL = updatedsets(config)

    # Ensure sections
    for sect in ('MAPS', 'FASTAS'):
        if not config.has_section(sect):
            config.add_section(sect)

    # --- Current FASTA md5s (only for ones that exist) ---
    current_fas_md5 = {}
    for fas in fas_inputs:
        if os.path.isfile(fas):
            _, md5 = getmd5(fas)
            current_fas_md5[fas] = md5 or ""

    # --- Existing SAM md5s (only for tracked + on-disk) ---
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
        fas_prev = (config['FASTAS'].get(fas_path) or "").strip()
        fas_cur  = current_fas_md5.get(fas_path, "")

        # if FASTA md5 missing in mem, write it now (best-effort) so mem stays consistent
        if fas_cur and fas_prev != fas_cur:
            config['FASTAS'][fas_path] = fas_cur

        # Decide if we must (re)map
        fas_changed = (not fas_prev) or (not fas_cur) or (fas_prev != fas_cur)
        sam_missing = not os.path.isfile(sam_path)
        sam_prev    = (config['MAPS'].get(sam_path) or "").strip()
        sam_cur     = computed_sam_md5.get(sam_path, "")
        sam_mismatch_or_blank = (not sam_prev) or (not sam_cur) or (sam_prev != sam_cur)

        if fas_changed or sam_missing or sam_mismatch_or_blank:
            libs_to_map.append(fas_path)

    # --- Perform mapping if needed ---
    if libs_to_map:
        print("Libraries to be mapped: %s" % (', '.join(libs_to_map)))
        nproc, nspread = optimize(ncores, len(libs_to_map))
        rawinputs = [(alib, genoIndex, nspread, maxhits) for alib in libs_to_map]
        PPBalance(mapper, rawinputs)

        # Record MD5s: SAMs and FASTAs that were (re)mapped
        libs_mapped = [f"{alib.rpartition('.')[0]}.sam" for alib in libs_to_map]

        # Small wait loop for filesystem stability before hashing large SAMs
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

        # Hash SAMs and write to [MAPS]
        sam_md5s = run_parallel_with_progress(
            getmd5, libs_mapped, desc="Hashing mapped SAMs", unit="lib"
        )
        for sam_path, md5 in sam_md5s:
            if md5:
                config['MAPS'][sam_path] = md5
            else:
                print(f"[WARN] MD5 empty for {sam_path}; keeping blank (will force remap next run).")

        # Refresh [FASTAS] for the mapped set (in case they were new)
        for fas in libs_to_map:
            _, fas_md5 = getmd5(fas)
            config['FASTAS'][fas] = fas_md5 or ""

        with open(memFile, 'w') as fh:
            config.write(fh)

    else:
        libs_mapped = []
        print("\nNo new libraries to map this time")

        # Even if we didn't map, ensure existing SAMs have a non-empty MD5 in mem (stabilize cache)
        wrote_any = False
        for sam_path in sams_expected:
            if not os.path.isfile(sam_path):
                continue
            mem_md5 = (config['MAPS'].get(sam_path) or "").strip()
            if not mem_md5:
                _, cur_md5 = getmd5(sam_path)
                if cur_md5:
                    config['MAPS'][sam_path] = cur_md5
                    wrote_any = True
        # Also ensure FASTA MD5s exist in mem
        for fas_path, cur in current_fas_md5.items():
            if cur and (config['FASTAS'].get(fas_path) or "").strip() != cur:
                config['FASTAS'][fas_path] = cur
                wrote_any = True
        if wrote_any:
            with open(memFile, 'w') as fh:
                config.write(fh)

    return libs_mapped


def updatedsets(config):
    '''
    checks which settings have been updated by comparing
    globals from settings file with entries in memfile
    '''

    updatedsetL = []

    if int(config["ADVANCED"]["mismat"])     != int(mismat):
        updatedsetL.append("mismat")
    if int(config["ADVANCED"]["maxhits"])    != int(maxhits):
        updatedsetL.append("maxhits")
    if int(config["ADVANCED"]["clustbuffer"])!= int(clustbuffer):
        updatedsetL.append("clustbuffer")

    ## Settings that may not exist in ealry phase of
    ## analyses, generated error when this function
    ## is used by early functions
    if config['BASIC'].getboolean('phaselen'):
        if int(config["BASIC"]["phaselen"])!= int(phase):
            updatedsetL.append("phaselen")

    return updatedsetL


def dictcollector(libs, libs_to_parse, lib_maps):
    '''
    This loads the two main dicts into memory for clustering and scoring of
    libraries
    '''
    libs_nestdict = []  ## {libA-chr1:{pos:[taginfo1,taginfo2]}
    libs_poscountdict = []  ## {libA-chr1:(pos1,count),libA-chr2:(pos2,count)}

    total_libs = len(lib_maps)
    
    for nestdict_f, poscountdict_f, nestdict, poscountdict in lib_maps:
        libs_nestdict.append(nestdict)
        libs_poscountdict.append(poscountdict)
    return libs_nestdict, libs_poscountdict

def parserprocess(libs, load_dicts=False):
    """
    Parse mapped libraries (SAM files) in parallel.
    Default: return only file paths to avoid RAM blow-ups.
    If load_dicts=True, load them back SEQUENTIALLY (low peak RAM).
    """
    print("#### Fn: Lib Parser ##########################")
    libs_to_parse = []

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)
    updatedsetL = updatedsets(config)

    if 'mismat' in updatedsetL:
        print("Setting update detected for 'mismat' parameter")
        libs_to_parse = libs.copy()
    elif config.has_section('PARSED'):
        print("Subsequent run for parserprocess; parsing only remapped libraries")
        parsekeys = list(config['PARSED'].keys())
        for alib in libs:
            blib = '%s_%s.dict' % (alib.rpartition(".")[0], phase)
            if blib in parsekeys:
                xfiles = [k for k in parsekeys if k.rpartition(".")[0] == blib.rpartition(".")[0]]
                if xfiles:
                    _, bmd5 = getmd5(xfiles[0])
                    if bmd5 == config['PARSED'][xfiles[0]]:
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
        libs_to_parse = [f"{lib.rpartition('.')[0]}.sam" for lib in libs.copy()]

    dict_paths, count_paths = [], []

    if libs_to_parse:
        print(f"Libraries to be parsed: {', '.join(libs_to_parse)}")
        rawinputs = [(alib, maxhits, mismat) for alib in libs_to_parse]

        # Parse in parallel (results are just file paths)
        out_pairs = run_parallel_with_progress(
            samparser_streaming, rawinputs, desc="Parsing SAM", unit="lib"
        )  # returns [(dict_path, count_path), ...]

        # Split paths
        for dp, cp in out_pairs:
            dict_paths.append(dp)
            count_paths.append(cp)

        # MD5 in parallel on the produced files (paths only; low RAM)
        dicthashes    = run_parallel_with_progress(getmd5, dict_paths, desc="MD5 dict")
        counterhashes = run_parallel_with_progress(getmd5, count_paths, desc="MD5 count")
        libstoset(dicthashes, "PARSED")
        libstoset(counterhashes, "COUNTERS")

        # Only load dicts if explicitly requested (sequentially!)
        libs_nestdict, libs_poscountdict = [], []
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
            libs_nestdict, libs_poscountdict = dict_paths, count_paths  # return paths instead
    else:
        # No re-parse needed; use existing paths or sequential loads depending on flag
        dict_paths = [f"{alib.rpartition('.')[0]}_{phase}.dict" for alib in libs]
        count_paths = [f"{alib.rpartition('.')[0]}_{phase}.count" for alib in libs]
        if load_dicts:
            libs_nestdict, libs_poscountdict = [], []
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

def lib_stem(p: str) -> str:
    """'.../ALL_LIBS.sam' -> 'ALL_LIBS'; 'F_1.tag' -> 'F_1'."""
    return os.path.splitext(os.path.basename(p))[0]

def make_akey(lib_id: str, chr_id) -> str:
    """Consistent akey constructor."""
    return f"{lib_id}-{chr_id}"

def canonicalize_akey(s: str) -> str:
    """
    Return 'LIB-CHR' from any akey-like string.
    Examples:
      '/a/b/ALL_LIBS-1.lclust'              -> 'ALL_LIBS-1'
      '_quobyte_..._ALL_LIBS-1'             -> 'ALL_LIBS-1'
      'F_1-3.sclust'                        -> 'F_1-3'
    """
    base = os.path.basename(str(s))
    base = base.rsplit('.', 1)[0]  # drop .lclust/.sclust if present
    # Prefer the trailing "<lib>-<digits>" pattern if present
    m = re.search(r'([A-Za-z0-9._+-]+-\d+)$', base)
    if m:
        return m.group(1)
    return base

def samparser_streaming(aninput):
    """
    Parse one SAM -> write:
      - <lib>_<phase>.dict (pickle of nestdict)
      - <lib>_<phase>.count (pickle of poscountdict)
    Return only (outfile1, outfile2) to keep RAM low.
    """
    alib, maxhits, mismat = aninput
    outfile1 = f"{alib.rpartition('.')[0]}_{phase}.dict"
    outfile2 = f"{alib.rpartition('.')[0]}_{phase}.count"
    asum     = f"{alib.rpartition('.')[0]}.sum"

    # Optional first pass for normalization
    total_abund = None
    if norm:
        total_abund = 0
        with open(alib, 'r') as fh:
            for line in fh:
                if line.startswith("@"):
                    continue
                ent = line.rstrip("\n").split("\t")
                aflag = int(ent[1])
                if aflag not in {0,256,16,272}:
                    continue
                aname = ent[0].strip()
                aabun = int(aname.split("|")[-1])
                total_abund += aabun

    # Streamed build
    tempdict1 = defaultdict(list)   # {anid: [(pos, taginfo), ...]}
    posdict   = defaultdict(list)   # {anid: [pos, ...]}

    reads_passed = 0
    with open(alib, 'r') as fh:
        for line in fh:
            if line.startswith("@"):
                continue
            ent     = line.rstrip("\n").split("\t")
            aflag   = int(ent[1])
            if aflag not in {0,256,16,272}:
                continue

            aname   = ent[0].strip()
            achr    = ent[2]
            apos    = int(ent[3])
            amapq   = int(ent[4])
            atag    = ent[9].strip()
            alen    = len(atag)
            aabun   = int(aname.split("|")[-1])
            astrand = 'w' if aflag in {0,256} else 'c'
            try:
                amismat = int(ent[-7].rpartition(":")[-1])
                ahits   = int(ent[-1].rpartition(":")[-1])
            except Exception:
                # Guard against malformed optional fields
                continue

            if ahits < maxhits and amismat <= mismat:
                reads_passed += 1
                anid = make_akey(lib_stem(alib), achr)

                # normalize abundance if requested
                adj_abun = aabun
                if norm and total_abund and total_abund > 0:
                    adj_abun = max(round((aabun / total_abund) * norm_factor), 1)

                taginfo = [achr, astrand, ahits, atag, aname, apos, alen, adj_abun]
                tempdict1[anid].append((apos, taginfo))
                posdict[anid].append(apos)

    # Build final structures
    nestdict = defaultdict(list)  # {anid: [ {pos: [taginfo, ...]} ]}
    for akey, aval in tempdict1.items():
        tmp = defaultdict(list)
        for p, tinfo in aval:
            tmp[p].append(tinfo)
        nestdict[akey].append(tmp)

    poscountdict = {
        akey: OrderedDict(sorted(Counter(aval).items(), key=lambda x: int(x[0])))
        for akey, aval in posdict.items()
    }

    # Write outputs and a tiny summary line
    with open(outfile1, "wb") as f1:
        pickle.dump(nestdict, f1, protocol=pickle.HIGHEST_PROTOCOL)
    with open(outfile2, "wb") as f2:
        pickle.dump(poscountdict, f2, protocol=pickle.HIGHEST_PROTOCOL)
    with open(asum, "a") as fsum:
        fsum.write(f"Reads passed filters for {alib}:\t{reads_passed}\n")

    # Drop heavy objects before returning
    del tempdict1, posdict, nestdict, poscountdict
    gc.collect()

    return outfile1, outfile2

def clustmerge(clustlist_all):
    """
    Revised clustmerge: Merges clusters if the next cluster’s start is within
    the current cluster’s end plus the clustbuffer.
    """
    # Ensure clustlist_all is sorted by the starting coordinate of each cluster
    if any(int(clustlist_all[i][0]) > int(clustlist_all[i+1][0])
           for i in range(len(clustlist_all) - 1)):
        clustlist_all.sort(key=lambda x: int(x[0]))
        print("Sorting clusters before merging")

    merged_clusters = []
    # Start with the first cluster
    current_cluster = clustlist_all[0][:]  # make a copy

    for next_cluster in clustlist_all[1:]:
        # If next cluster starts before or within the clustbuffer after current cluster's end,
        # then merge them.
        current_end = int(current_cluster[-1])
        next_start = int(next_cluster[0])
        if next_start <= current_end + int(clustbuffer):
            # Merge and keep unique positions sorted
            current_cluster = sorted(set(current_cluster + next_cluster), key=int)
        else:
            merged_clusters.append(current_cluster)
            current_cluster = next_cluster[:]  # start a new cluster

    # Append the final cluster
    merged_clusters.append(current_cluster)
    return merged_clusters

def alt_parallel_process(func, data_chunks):
    
    max_workers = cores or cpu_count()
    with Pool(processes=max_workers) as pool:
        for result in pool.imap_unordered(func, data_chunks):
            yield result


def _flush_prev_cluster(prev_merged, clustid, clustlen_cutoff, clustdict_long, clustdict_short):
    if prev_merged:
        clustid += 1
        leftx  = prev_merged[0]
        rightx = prev_merged[-1]
        if (rightx - leftx) + 1 > clustlen_cutoff:
            clustdict_long[clustid] = prev_merged
        elif len(prev_merged) > 1:
            clustdict_short[clustid] = prev_merged
        prev_merged = []
    return prev_merged, clustid

# Refactored getclusters (no inner function)
def getclusters(args):
    """
    Compute clusters for one (akey, acounter) and write to disk immediately.
    Returns: (akey, lclust_file, sclust_file, lclust_md5)
    Memory-lean single pass; only small lists are kept in memory.
    """
    akey, acounter, clustfolder = args
    clustlen_cutoff = int(phase) * 4 + 3 + 1

    akey_safe = canonicalize_akey(akey)
    lclust_file = os.path.join(clustfolder, f"{akey_safe}.lclust")
    sclust_file = os.path.join(clustfolder, f"{akey_safe}.sclust")
    # Ensure ordered position keys with minimal overhead
    if isinstance(acounter, OrderedDict):
        keys_iter = acounter.keys()
    else:
        try:
            keys_iter = sorted(acounter.keys(), key=int)
        except Exception:
            keys_iter = sorted(acounter.keys())

    it = iter(keys_iter)
    try:
        first_key = next(it)
    except StopIteration:
        # No positions: write empty dicts
        with open(lclust_file, "wb") as f1, open(sclust_file, "wb") as f2:
            pickle.dump({}, f1, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump({}, f2, protocol=pickle.HIGHEST_PROTOCOL)
        _, lclust_md5 = getmd5(lclust_file)
        return (akey, lclust_file, sclust_file, lclust_md5)

    # First position (cast if needed)
    try:
        first_pos = int(first_key)
    except Exception:
        first_pos = first_key

    clustdict_long = {}
    clustdict_short = {}
    clustid = 0

    prev_merged = []          # last merged cluster (list of ints)
    curr_pre    = [first_pos] # current precluster
    last_pos    = first_pos

    for k in it:
        try:
            pos = int(k)
        except Exception:
            pos = k

        if pos - last_pos > CLUSTSPLIT:
            # boundary -> merge with prev if within clustbuffer, else flush
            if not prev_merged:
                prev_merged = curr_pre
            else:
                if curr_pre[0] <= (prev_merged[-1] + int(clustbuffer)):
                    prev_merged.extend(curr_pre)  # both sorted & disjoint
                else:
                    prev_merged, clustid = _flush_prev_cluster(
                        prev_merged, clustid, clustlen_cutoff, clustdict_long, clustdict_short
                    )
                    prev_merged = curr_pre
            curr_pre = [pos]
        else:
            curr_pre.append(pos)

        last_pos = pos

    # Tail handling
    if not prev_merged:
        prev_merged = curr_pre
    else:
        if curr_pre[0] <= (prev_merged[-1] + int(clustbuffer)):
            prev_merged.extend(curr_pre)
        else:
            prev_merged, clustid = _flush_prev_cluster(
                prev_merged, clustid, clustlen_cutoff, clustdict_long, clustdict_short
            )
            prev_merged = curr_pre

    prev_merged, clustid = _flush_prev_cluster(
        prev_merged, clustid, clustlen_cutoff, clustdict_long, clustdict_short
    )

    # Write to disk ASAP
    with open(lclust_file, "wb") as f1, open(sclust_file, "wb") as f2:
        pickle.dump(clustdict_long, f1, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(clustdict_short, f2, protocol=pickle.HIGHEST_PROTOCOL)

    _, lclust_md5 = getmd5(lclust_file)

    # Cleanup
    del clustdict_long, clustdict_short, prev_merged, curr_pre
    gc.collect()

    return (akey, lclust_file, sclust_file, lclust_md5)

def process_cluster_batch(batch, batch_id):
    """
    Run one clustering batch and return results.
    Stateless helper (no imports inside, no closures).
    """
    try:
        return run_parallel_with_progress(
            getclusters, batch, desc=f"Clustering batch {batch_id}", unit="lib-chr"
        )
    except Exception as e:
        print(f"[WARN] run_parallel_with_progress failed on batch {batch_id}: {e}")
        print("[INFO] Falling back to alternative chunked parallel_process...")
        try:
            return list(alt_parallel_process(getclusters, batch))
        except Exception as ee:
            print(f"[ERROR] Fallback failed for batch {batch_id}: {ee}")
            return []

def _safe_key(akey: str) -> str:
    """Normalize an akey to a filesystem-safe basename."""
    return os.path.basename(str(akey))

def _prune_old_clustered_entries(cfg: configparser.ConfigParser, basename: str, keep_abs: str) -> int:
    """
    Remove stale [CLUSTERED] entries that have the same basename but point
    to a different path than keep_abs. Returns the number of deletions.
    """
    if not cfg.has_section("CLUSTERED"):
        return 0
    to_delete = [k for k in cfg["CLUSTERED"].keys()
                 if os.path.basename(k) == basename and os.path.realpath(k) != os.path.realpath(keep_abs)]
    for k in to_delete:
        try:
            cfg.remove_option("CLUSTERED", k)
        except Exception:
            pass
    return len(to_delete)

def clusterprocess(libs_poscountdict, clustfolder):
    """
    RAM-reduced clustering that *only* writes to `clustfolder` and keeps the mem file clean:
      - Writes per-lib-chr clusters to <clustfolder>/<akey>.lclust|.sclust
      - Updates [CLUSTERED] with the ABS path under `clustfolder`
      - Prunes any stale [CLUSTERED] entries for the same basename elsewhere
      - Skips recompute when the on-disk hash matches the current [CLUSTERED] value
      - Returns: [(akey, lclust_file, sclust_file), ...] (paths in `clustfolder`)
    """
    print("#### Fn: Find Clusters #######################")
    global CLUSTSPLIT
    CLUSTSPLIT = int(phase) + 1 + 3

    os.makedirs(clustfolder, exist_ok=True)

    # Normalize sources
    if isinstance(libs_poscountdict, (dict, str)):
        sources = [libs_poscountdict]
    else:
        sources = list(libs_poscountdict)

    # Read memo
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(memFile)
    if not cfg.has_section("CLUSTERED"):
        cfg.add_section("CLUSTERED")

    # Fast lookup: existing md5s by path
    clustered_md5 = dict(cfg["CLUSTERED"])

    # Accumulators
    results: list[tuple[str, str, str]] = []
    new_hashes: dict[str, str] = {}
    processed_akeys: list[str] = []

    # Batching to cap memory
    CLUSTER_CHUNK_MAX = 10
    batch = []
    batch_index = 0

    # Helper to flush a batch via your existing worker
    def _flush_batch(batch_items, idx):
        nonlocal results, new_hashes, processed_akeys
        if not batch_items:
            return
        chunk_results = process_cluster_batch(batch_items, idx)
        for res in chunk_results:
            if len(res) == 4:
                a, lfile, sfile, lmd5 = res
            else:
                a, lfile, sfile = res
                _, lmd5 = getmd5(lfile)
            # Normalize to abs realpaths inside clustfolder
            a_safe = _safe_key(a)
            want_l = os.path.realpath(os.path.join(clustfolder, f"{a_safe}.lclust"))
            want_s = os.path.realpath(os.path.join(clustfolder, f"{a_safe}.sclust"))

            # If worker wrote elsewhere by mistake, move into clustfolder atomically
            if os.path.isfile(lfile) and os.path.realpath(lfile) != want_l:
                try:
                    os.replace(lfile, want_l)
                except Exception:
                    import shutil
                    shutil.copy2(lfile, want_l)
                    try:
                        os.remove(lfile)
                    except Exception:
                        pass
            if os.path.isfile(sfile) and os.path.realpath(sfile) != want_s:
                try:
                    os.replace(sfile, want_s)
                except Exception:
                    import shutil
                    shutil.copy2(sfile, want_s)
                    try:
                        os.remove(sfile)
                    except Exception:
                        pass

            # Refresh md5 from final location
            _, lmd5_final = getmd5(want_l)
            new_hashes[want_l] = lmd5_final

            # Prune any stale entries for same basename
            pruned = _prune_old_clustered_entries(cfg, os.path.basename(want_l), want_l)
            if pruned:
                # keep in-memory view in sync (will rewrite to disk later)
                for k in list(clustered_md5.keys()):
                    if os.path.basename(k) == os.path.basename(want_l) and os.path.realpath(k) != want_l:
                        clustered_md5.pop(k, None)

            clustered_md5[want_l] = lmd5_final
            results.append((a, want_l, want_s))
            processed_akeys.append(a)

    # Build batches
    for src in sources:
        if isinstance(src, str):
            try:
                with open(src, "rb") as fh:
                    libdict = pickle.load(fh)
            except Exception:
                print(f"[WARN] Failed to load count file {src}")
                continue
            loaded_from_path = True
        elif isinstance(src, dict):
            libdict = src
            loaded_from_path = False
        else:
            print(f"[WARN] Unexpected type in libs_poscountdict: {type(src)} (skipping)")
            continue

        for akey, positions in libdict.items():
            a_safe = _safe_key(akey)
            lclust_path = os.path.realpath(os.path.join(clustfolder, f"{a_safe}.lclust"))
            sclust_path = os.path.realpath(os.path.join(clustfolder, f"{a_safe}.sclust"))

            # Cache hit? compare on-disk md5 with current mem md5 for THIS clustfolder path
            if os.path.isfile(lclust_path):
                _, cur_md5 = getmd5(lclust_path)
                prev_md5 = cfg["CLUSTERED"].get(lclust_path, "")
                if prev_md5 and cur_md5 and prev_md5 == cur_md5:
                    # Also prune any stale duplicates for same basename
                    _prune_old_clustered_entries(cfg, os.path.basename(lclust_path), lclust_path)
                    results.append((akey, lclust_path, sclust_path))
                    processed_akeys.append(akey)
                    continue

            # enqueue work (worker writes; we’ll normalize on flush)
            batch.append((akey, positions, clustfolder))
            if len(batch) >= CLUSTER_CHUNK_MAX:
                batch_index += 1
                _flush_batch(batch, batch_index)
                batch = []
                gc.collect()

        if loaded_from_path:
            del libdict
            gc.collect()

    # Flush trailing work
    if batch:
        batch_index += 1
        _flush_batch(batch, batch_index)
        batch = []
        gc.collect()

    # Persist [CLUSTERED] updates (only our final clustfolder paths)
    if not cfg.has_section("CLUSTERED"):
        cfg.add_section("CLUSTERED")
    for lpath, md5 in new_hashes.items():
        cfg["CLUSTERED"][lpath] = md5
    with open(memFile, "w") as fh:
        cfg.write(fh)

    # Save the akeys list for downstream use
    try:
        with open(os.path.join(clustfolder, "libchr-keys.p"), "wb") as pf:
            pickle.dump(processed_akeys, pf)
    except Exception as e:
        print(f"[WARN] Could not write libchr-keys.p: {e}")

    return results


def build_libchrs_nestdict(sources, needed_keys=None):
    """
    Build {akey: value} from a mix of:
      - dict objects (each mapping akey -> value), or
      - '*.dict' pickle file paths containing such dicts.

    If needed_keys is provided (set of akeys), only those keys are loaded.
    """
    result = {}
    # Normalize input
    if isinstance(sources, (dict, str)):
        iterable = [sources]
    else:
        iterable = list(sources)

    for idx, src in enumerate(iterable):
        if isinstance(src, dict):
            # Direct merge, but only required keys if specified
            if needed_keys is None:
                result.update(src)
            else:
                for k, v in src.items():
                    if k in needed_keys:
                        result[k] = v
        elif isinstance(src, str):
            # Load from pickle file path
            try:
                with open(src, "rb") as fh:
                    loaded = pickle.load(fh)
            except Exception as e:
                print(f"[WARN] Could not load dict file '{src}': {e}")
                continue
            if not isinstance(loaded, dict):
                print(f"[WARN] Dict file '{src}' did not contain a dict; got {type(loaded)}")
                continue
            if needed_keys is None:
                result.update(loaded)
            else:
                # Only keep what we need
                for k in needed_keys:
                    if k in loaded:
                        result[k] = loaded[k]
            # free per-file object ASAP
            del loaded
            gc.collect()
        else:
            print(f"[WARN] Unexpected libs_nestdict element type at index {idx}: {type(src)}; skipping")

    return result

# Top-level helper (not nested, per your preference)
def iter_batches(seq, size):
    for i in range(0, len(seq), size):
        yield i // size + 1, (len(seq) + size - 1) // size, seq[i:i+size]

def load_lclust_for_scoring(arg):
    """
    arg = (akey_expected, lclust_path)
    Returns a tuple keyed by the file path (order-agnostic):
        (lclust_path, loaded_akey, ldict)
    - loaded_akey is derived from the filename stem, e.g. '<akey>.lclust' -> '<akey>'
    - on failure: (lclust_path, None, None)
    """
    akey_expected, lclust_path = arg
    try:
        fname = os.path.basename(lclust_path)
        loaded_akey = os.path.splitext(fname)[0]
        with open(lclust_path, "rb") as fh:
            ldict = pickle.load(fh)
        return (lclust_path, loaded_akey, ldict)
    except Exception as e:
        print(f"[WARN] Could not load {lclust_path}: {e}")
        return (lclust_path, None, None)


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


def sanitize_mem_md5s(config, sections=('CLUSTERS', 'CLUSTERED')):
    """
    Remove empty-string / None MD5s that can poison cache decisions.
    Returns: dict {section: removed_count}
    """
    removed = {}
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


def resolve_lclust_path(path_like, clustfolder):
    """
    Resolve .lclust path to an existing ABS REALPATH.
    Preference: given path → clustfolder/basename → CWD/basename fallback.
    """
    p = str(path_like)
    if os.path.isfile(p):
        return os.path.realpath(p)
    b = os.path.basename(p)
    cand = os.path.join(clustfolder, b) if clustfolder else None
    if cand and os.path.isfile(cand):
        return os.path.realpath(cand)
    # last-ditch: interpret relative to CWD
    cand2 = os.path.join(os.getcwd(), b)
    if os.path.isfile(cand2):
        return os.path.realpath(cand2)
    # return normalized original (may not exist; caller will check)
    return os.path.realpath(p)
# === Helpers (top-level; no imports here) ====================================

def _basename_no_ext(p):
    # '/a/b/ALL_LIBS.fas' -> 'ALL_LIBS'
    return os.path.basename(str(p)).rsplit('.', 1)[0]



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

# === Main =====================================================================
def scoringprocess(
    libs,
    libs_clustdicts,
    libs_nestdict,
    clustfolder,
    force_rescore=False,
    verify_outputs=True,
    scored_dir=None,
    purge_existing=False,
    # Back-compat: discover concat from global if not given
    concat_mode=None,
    merged_name="ALL_LIBS",
):
    """
    Cluster Scorer with concat-aware akey handling and robust mem/md5 bookkeeping.
    Produces *.cluster chunks (in <phase>_scoredClusters) and assembles
    <lib>.<phase>-PHAS.candidate.clusters outputs.

    Returns: list[str] of assembled cluster file paths.
    """
    print("#### Fn: Cluster Scorer ######################")

    # -------------------- Normalize inputs --------------------
    # Expect a list like [(akey, lclust_path), ...]
    libchrs_clust_toscore = []
    for tpl in (libs_clustdicts or []):
        if len(tpl) >= 2:
            akey, lclust_path = tpl[0], tpl[1]
            lclust_path = resolve_lclust_path(lclust_path, clustfolder)
            libchrs_clust_toscore.append((akey, lclust_path))
        else:
            print(f"[WARN] Unexpected libs_clustdicts element (len={len(tpl)}): {tpl}")

    if not libchrs_clust_toscore:
        print("[WARN] No .lclust inputs to score; returning empty list.")
        return []

    # -------------------- mem file & sections --------------------
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)
    _ = updatedsets(config)

    sect_clusters = "CLUSTERS"       # outputs: bare filename -> md5
    sect_lclust   = "CLUSTERED"      # inputs: absolute realpath -> md5
    sect_chunks   = "SCORED_CHUNKS"  # chunk *.cluster absolute paths -> md5
    for sect in (sect_clusters, sect_lclust, sect_chunks):
        if not config.has_section(sect):
            config.add_section(sect)

    removed = sanitize_mem_md5s(config, (sect_clusters, sect_lclust, sect_chunks))
    if any(removed.values()):
        with open(memFile, "w") as fh:
            config.write(fh)

    # -------------------- Collect nest dict (unfiltered; robust in concat) ----
    # Avoid losing entries by over-filtering here; we'll probe with canonical keys later.
    libchrs_nestdict = build_libchrs_nestdict(libs_nestdict)

    # -------------------- Filter: existing .lclust + available nest -----------
    filtered_inputs, missing_akeys, missing_files = [], [], []
    for akey, lclust_path in libchrs_clust_toscore:
        if not os.path.isfile(lclust_path):
            missing_files.append(lclust_path)
            continue
        if (akey in libchrs_nestdict) or (canonicalize_akey(akey) in libchrs_nestdict):
            filtered_inputs.append((akey, lclust_path))
        else:
            missing_akeys.append(akey)

    if missing_files:
        print(f"[WARN] Missing .lclust files for {len(missing_files)}; e.g.: {missing_files[:3]}")
    if missing_akeys:
        print(f"[WARN] Missing nestdict for {len(missing_akeys)} akeys; skipped. Example: {missing_akeys[:5]}")

    if not filtered_inputs:
        print("[WARN] No valid inputs after filtering; returning empty list.")
        return []

    # -------------------- Concat mode detection/target lib --------------------
    if concat_mode is None:
        # Concat if we detect the merged lib name as a basename, or if only one lib provided
        concat_mode = any(_basename_no_ext(alib) == merged_name for alib in (libs or [])) or (len(libs or []) == 1)

    concat_target_lib = None
    if concat_mode and libs:
        # Prefer the lib whose basename matches merged_name, fall back to first
        for alib in libs:
            if _basename_no_ext(alib) == merged_name:
                concat_target_lib = alib
                break
        if concat_target_lib is None:
            concat_target_lib = libs[0]

    # -------------------- Per-lib expected outfiles ---------------------------
    expected_outfiles = []
    if concat_mode and concat_target_lib:
        bname = _basename_no_ext(concat_target_lib)
        expected_outfiles.append((concat_target_lib, f"{bname}.{phase}-PHAS.candidate.clusters"))
    else:
        for alib in libs:
            bname = _basename_no_ext(alib)
            expected_outfiles.append((alib, f"{bname}.{phase}-PHAS.candidate.clusters"))

    # -------------------- Map akey (basename) -> lib path ---------------------
    akey_to_lib = {}
    if concat_mode and concat_target_lib:
        for akey, _ in filtered_inputs:
            akey_to_lib[os.path.basename(str(akey))] = concat_target_lib
    else:
        # Greedy longest-prefix match against per-lib basename
        lib_prefixes = [(_basename_no_ext(alib), alib) for alib in libs]
        for akey, _ in filtered_inputs:
            akey_base = os.path.basename(str(akey))
            best_lib, best_len = None, -1
            for pref, alib in lib_prefixes:
                if akey_base.startswith(pref) or (pref in akey_base):
                    if len(pref) > best_len:
                        best_lib, best_len = alib, len(pref)
            akey_to_lib[akey_base] = best_lib

    # -------------------- Scored chunks folder (global for clustwrite) --------
    currdir = os.getcwd()
    base_scored = os.path.join(currdir, f"{phase}_scoredClusters")
    os.makedirs(base_scored, exist_ok=True)

    global scoredClustFolder
    scoredClustFolder = scored_dir if scored_dir else base_scored

    if purge_existing and os.path.isdir(scoredClustFolder):
        # Purge only .cluster files; keep the folder
        for fn in os.listdir(scoredClustFolder):
            if fn.endswith(f".sRNA_{phase}.cluster"):
                try:
                    os.remove(os.path.join(scoredClustFolder, fn))
                except Exception:
                    pass

    # -------------------- Batch planning -------------------------------------
    n_data = len(filtered_inputs)
    print(f"n_data is {n_data}")
    batch_size = max(10, min(39, n_data))  # mirrors typical chunk sizes seen in logs
    batches = list(iter_batches(filtered_inputs, batch_size))
    print(f"n_batches set to {len(batches)}")

    # Precompute phased indexes
    sens, asens = getPhasedIndexes(WINDOW_SIZE)

    # Track md5 updates + libs that need re-assembly
    lclust_md5_updates = {}
    libs_marked_stale = set()
    if purge_existing or force_rescore:
        libs_marked_stale.update(alib for (alib, _) in expected_outfiles)

    # -------------------- Process batches ------------------------------------
    for b_idx, b_tot, batch in batches:
        print(f" Initial chunk_size set to {batch_size}")
        # 1) Determine affected libs for this batch
        batch_libs = set()
        for akey, _ in batch:
            alib = akey_to_lib.get(os.path.basename(str(akey)))
            if alib is not None:
                batch_libs.add(alib)

        # Batch outfiles (concat: single; normal: those touched)
        if concat_mode and concat_target_lib:
            batch_outfiles = [(concat_target_lib, expected_outfiles[0][1])]
        else:
            filtered = [(alib, outf) for (alib, outf) in expected_outfiles if alib in batch_libs]
            batch_outfiles = filtered or list(expected_outfiles)

        # 2) Output MD5 check (bare filenames in [CLUSTERS])
        batch_outputs_ok = False
        outfile_md5_current = {}
        outfile_realpaths = {}

        if verify_outputs and not force_rescore and batch_outfiles:
            existing_paths = []
            for _, outf in batch_outfiles:
                if os.path.isfile(outf):
                    p = os.path.realpath(outf)
                    outfile_realpaths[outf] = p
                    existing_paths.append(p)

            if existing_paths:
                md5_results_out = run_parallel_with_progress(
                    md5_file_worker,
                    existing_paths,
                    desc=f"Hashing existing outputs (batch {b_idx}/{b_tot})",
                    min_chunk=1,
                    unit="file",
                )
                for p_abs, md5 in md5_results_out:
                    if md5:
                        outfile_md5_current[p_abs] = md5

                # Compare bare name keys in [CLUSTERS]
                all_match = True
                for _, outf in batch_outfiles:
                    p_abs = outfile_realpaths.get(outf)
                    if not p_abs:
                        all_match = False
                        break
                    curr = outfile_md5_current.get(p_abs)
                    prev = config[sect_clusters].get(os.path.basename(outf))
                    if not curr or not prev or prev != curr:
                        all_match = False
                        break
                batch_outputs_ok = all_match

        # 3) Input MD5 check (absolute realpaths in [CLUSTERED])
        batch_pairs = []
        batch_paths = []
        for akey, p in batch:
            p_abs = os.path.realpath(p)
            batch_pairs.append((akey, p_abs))
            batch_paths.append(p_abs)

        changed_inputs = set()
        unknown_inputs = set()
        batch_md5_map = {}
        if not force_rescore:
            md5_results_in = run_parallel_with_progress(
                md5_file_worker,
                batch_paths,
                desc=f"Hashing .lclust (batch {b_idx}/{b_tot})",
                min_chunk=1,
                unit="file",
            )
            batch_md5_map = {p: md5 for (p, md5) in md5_results_in if md5 is not None}
            for akey, p_abs in batch_pairs:
                curr_in = batch_md5_map.get(p_abs)
                if curr_in is None:
                    changed_inputs.add(akey)
                    unknown_inputs.add(p_abs)
                    continue
                prev_in = config[sect_lclust].get(p_abs)
                if not prev_in or prev_in != curr_in:
                    changed_inputs.add(akey)
                else:
                    lclust_md5_updates[p_abs] = curr_in  # unchanged; refresh md5

        # If outputs are OK and there are no changed inputs, we can skip scoring
        if batch_outputs_ok and not changed_inputs and not force_rescore:
            print(f"[INFO] Batch {b_idx}/{b_tot}: outputs ok & inputs unchanged; skip scoring.")
            # still refresh any unknown inputs
            for p_abs in unknown_inputs:
                md5_now = batch_md5_map.get(p_abs)
                if md5_now:
                    lclust_md5_updates[p_abs] = md5_now
            continue

        # 4) Load lclust dicts for the batch
        to_process = [(akey, p_abs) for (akey, p_abs) in batch_pairs]
        loaded = run_parallel_with_progress(
            load_lclust_for_scoring,
            to_process,
            desc=f"Loading .lclust (batch {b_idx}/{b_tot}, {len(to_process)} akeys)",
            min_chunk=1,
            unit="lib-chr",
        )

        # Build lookup tables using canonical keys for robust matching
        by_path = {}
        by_akey = defaultdict(list)
        for path_abs, loaded_akey, ldict in loaded:
            loaded_akey_can = canonicalize_akey(loaded_akey)
            by_path[path_abs] = (loaded_akey_can, ldict)
            if loaded_akey_can is not None:
                by_akey[loaded_akey_can].append((path_abs, ldict))

        # 5) Build rawinputs for scoring (match loader/nest; handle mismatches)
        rawinputs = []
        mismatches = 0
        missing_in_loader = 0
        missing_nest = 0

        for akey_exp, lclust_path in to_process:
            akey_can = canonicalize_akey(akey_exp)
            entry = by_path.get(lclust_path)

            if entry is not None:
                loaded_akey, ldict = entry
                if ldict is None:
                    continue
                if loaded_akey != akey_can:
                    cand = by_akey.get(akey_can, [])
                    if len(cand) == 1:
                        _, ldict2 = cand[0]
                        ldict = ldict2
                        mismatches += 1
                    else:
                        print(f"[WARN] Key mismatch ({akey_can} vs {loaded_akey}); skipping.")
                        mismatches += 1
                        continue
            else:
                cand = by_akey.get(akey_can, [])
                if len(cand) == 1:
                    _, ldict = cand[0]
                    missing_in_loader += 1
                else:
                    print(f"[WARN] Loaded results missing path {lclust_path}")
                    missing_in_loader += 1
                    continue

            # Probe the nest dict using expected key, then canonical
            if akey_exp in libchrs_nestdict:
                nest_key = akey_exp
            elif akey_can in libchrs_nestdict:
                nest_key = akey_can
            else:
                missing_nest += 1
                continue

            rawinputs.append((akey_can, ldict, libchrs_nestdict[nest_key], sens, asens))

        if mismatches or missing_in_loader or missing_nest:
            print(
                f"[INFO] Batch {b_idx}/{b_tot}: fixups={mismatches}, "
                f"path_fallbacks={missing_in_loader}, missing_nest={missing_nest}"
            )
            # Any of these conditions means touched libs should be re-assembled
            libs_marked_stale.update(batch_libs)

        # 6) Score clusters -> *.cluster chunks
        if rawinputs:
            run_parallel_with_progress(
                clustassemble,
                rawinputs,
                desc=f"Scoring clusters (batch {b_idx}/{b_tot})",
                min_chunk=1,
                unit="lib-chr",
            )
            libs_marked_stale.update(batch_libs)
        else:
            print(f"[INFO] Batch {b_idx}/{b_tot}: nothing to score after loader/nest checks.")

        # 7) Refresh input MD5s for all batch paths
        if batch_paths:
            md5_results_in2 = run_parallel_with_progress(
                md5_file_worker,
                batch_paths,
                desc=f"Refreshing .lclust hashes (batch {b_idx}/{b_tot})",
                min_chunk=1,
                unit="file",
            )
            for p_abs, md5 in md5_results_in2:
                if md5 is not None:
                    lclust_md5_updates[p_abs] = md5

        # Free batch data
        del loaded, by_path, by_akey, rawinputs
        gc.collect()

    # -------------------- Assemble per-lib outputs ----------------------------
    clusterFiles = []
    regenerated = []

    def _needs_rebuild(path: str) -> bool:
        return (not os.path.isfile(path)) or (os.path.getsize(path) == 0)

    if libs:
        for alib, outfile in expected_outfiles:
            lib_prefix = _basename_no_ext(alib)
            must_rebuild = (
                force_rescore or purge_existing or
                (alib in libs_marked_stale) or
                _needs_rebuild(outfile)
            )

            if not must_rebuild:
                clusterFiles.append(outfile)
                continue

            # Rebuild from chunks
            if os.path.isfile(outfile):
                try:
                    os.remove(outfile)
                except Exception:
                    pass

            n_chunks, n_bytes = assemble_candidate_from_chunks(
                scoredClustFolder, lib_prefix, phase, outfile
            )
            if n_chunks > 0 and os.path.isfile(outfile) and os.path.getsize(outfile) > 0:
                clusterFiles.append(outfile)
                regenerated.append(outfile)
                print(f"[OK] Aggregated {n_chunks} chunk(s) ({n_bytes} bytes) -> {outfile}")
            else:
                print(f"[WARN] No non-empty chunks aggregated for {lib_prefix}; {os.path.basename(outfile)} is empty.")
                if os.path.isfile(outfile):
                    clusterFiles.append(outfile)
    else:
        print("[WARN] No libraries passed for output assembly step.")

    # -------------------- Hash regenerated outputs -> [CLUSTERS] --------------
    if regenerated:
        md5_after = run_parallel_with_progress(
            md5_file_worker,
            regenerated,
            desc="Hashing regenerated outputs",
            min_chunk=1,
            unit="file",
        )
        for p_abs, md5 in md5_after:
            if md5:
                config[sect_clusters][os.path.basename(p_abs)] = md5

    # -------------------- Hash chunk files -> [SCORED_CHUNKS] -----------------
    if os.path.isdir(scoredClustFolder):
        chunk_paths = [
            os.path.realpath(os.path.join(scoredClustFolder, f))
            for f in os.listdir(scoredClustFolder)
            if f.endswith(f".sRNA_{phase}.cluster")
        ]
        if chunk_paths:
            md5_chunks = run_parallel_with_progress(
                md5_file_worker,
                chunk_paths,
                desc="Hashing .cluster chunks",
                min_chunk=1,
                unit="file",
            )
            for p_abs, md5 in md5_chunks:
                if md5:
                    config[sect_chunks][p_abs] = md5

    # -------------------- Persist input md5 -> [CLUSTERED] --------------------
    for p_abs, md5 in lclust_md5_updates.items():
        if md5 is not None:
            config[sect_lclust][p_abs] = md5

    with open(memFile, "w") as fh:
        config.write(fh)

    print(f"cluster files are {clusterFiles}")
    return clusterFiles


def flatten_list_of_dict(alist):
    '''
    input: a list of dicts
    takes a list of dict and flattens to a dict
    output: a dict
    '''
    resdict = {}
    for i, adict in enumerate(alist):
        if not isinstance(adict, dict):
            print(f"Error: Element at index {i} is not a dictionary, it's a {type(adict)}: {adict}")
            continue  # Skip non-dictionary elements
        
        # Now proceed with normal execution if adict is a dictionary
        akeys = adict.keys()  # akeys are lib-chrs for this library
        for akey in akeys:
            # Fetch position-specific dict of tag infos
            bdict = adict[akey]
            resdict[akey] = bdict
    return resdict



def cacheclustdicts(libchrs_keys,libchr_clustered,clustfolder):
    '''
    reads long cluster and short cluster dicts
    to memory
    '''
    ## find which libchr elements are already in
    ## memory from current run and avoid reading
    ## these again. Those in memory must have been
    ## re-read due to some parameter change upstream
    libchr_to_read  = []
    for akey in libchrs_keys:
        if akey not in libchr_clustered:
            libchr_to_read.append(akey)
    ## read clust dicts for scoring
    libschrs_posdict_l  = []
    libschrs_nestdict_d = {}
    acount              = 0
    for akey in libchr_to_read:
        infile1         = "%s/%s.lclust"    % (clustfolder,akey)
        infile2         = "%s/%s.sclust"    % (clustfolder,akey)
        infile3         = "%s_%s.dict"         % (akey,phase)
        ldict           = pickle.load( open(infile1, "rb" ) )
        sdict           = pickle.load( open(infile2, "rb" ) )
        ndict           = pickle.load( open(infile3, "rb" ) )
        libschrs_posdict_l.append((akey,ldict,sdict))
        libschrs_nestdict_d[akey] = ndict
        acount          +=1
    return libschrs_posdict_l,libschrs_nestdict_d

def clustassemble(aninput):
    """
    gather full info for clusters i.e. all tags for cluster positions
    and write full clusters to a file
    """
    akey, lclustdict, nesteddict, sens, asens = aninput
    nesteddict = nesteddict[0]
    clustlist  = []
    phasedlist = []

    for aid, aclust in lclustdict.items():
        tempL1 = []   # full cluster tag records
        tempL2 = []   # tag lengths
        tempL3 = []   # (apos, strand, poscount, abun_all, abun_phase)

        tagcount = 0
        uniqcount = 0
        abuncount = 0
        abundict = defaultdict(list)

        for apos in aclust:
            tagslist = nesteddict[apos]
            poscount_w = poscount_c = 0
            posabun_a_w = posabun_p_w = 0
            posabun_a_c = posabun_p_c = 0

            for taginfo in tagslist:
                taghits  = int(taginfo[2])
                taglen   = int(taginfo[6])
                tagabun  = int(taginfo[7])
                tagstrand= taginfo[1]

                tempL1.append(taginfo)
                tempL2.append(taglen)

                abuncount += tagabun
                tagcount  += 1
                uniqcount += 1 if taghits <= UNIQRATIO_HIT else 0
                abundict[taglen].append(tagabun)

                if tagstrand == "w":
                    poscount_w  += 1
                    posabun_a_w += tagabun
                    posabun_p_w += tagabun if taglen == int(phase) else 0
                elif tagstrand == "c":
                    poscount_c  += 1
                    posabun_a_c += tagabun
                    posabun_p_c += tagabun if taglen == int(phase) else 0

            if poscount_w > 0:
                tempL3.append((apos, "w", poscount_w, posabun_a_w, posabun_p_w))
            if poscount_c > 0:
                tempL3.append((apos, "c", poscount_c, posabun_a_c, posabun_p_c))

        tempL1_s = sorted(tempL1, key=lambda x: x[5])
        tempL2_s = sorted(tempL2)
        tempL3_s = sorted(tempL3, key=lambda x: x[0])

        signal = clustfilter(tempL1_s, tempL2_s, abundict, uniqcount, tagcount, abuncount)
        if signal:
            # `asens` is accepted but not used by clustscore
            scoredclust = clustscore(tempL1_s, tempL3_s, sens, asens)
            clustlist.append((aid, scoredclust))

    clustwrite(akey, clustlist)
    return None

def clustfilter(tempL1_s, tempL2_s, abundict, uniqcount, tagcount, abuncount):
    """
    filters a lib-chr list of cluster
    Requires >= 10 tag entries in the cluster.
    """
    # Minimum evidence: at least 10 tag records in this cluster
    if len(tempL1_s) < 10:
        return False

    # Basic sanity checks to avoid division by zero
    if tagcount <= 0 or abuncount <= 0:
        return False

    ph = int(phase)

    # Compute dominant size metrics
    domsize_l, lencounter, DOMCOUNT_CUT = compute_domsize(tempL2_s, abundict, abuncount)

    # Uniqueness ratio and phase-length abundance ratio
    uniqratio = round(uniqcount / tagcount, 5)
    ph_abuns = abundict.get(ph, [])
    domsize_ratio = round((sum(ph_abuns) / abuncount), 5) if abuncount else 0.0

    # Count of reads at the phase length
    phaslen_counts = lencounter.get(ph, 0)

    # Final decision
    if (phaslen_counts > DOMCOUNT_CUT) and (uniqratio >= uniqueRatioCut) and (domsize_ratio >= DOMSIZE_CUT):
        return True
    else:
        # the cluster size class doesn't match the phase or it has lots of multihit sRNAs
        return False

def compute_domsize(lenlist,abundict,abuncount):
    '''
    compute the dominant sRNA size in clusters
    '''
    ## combine len, counts and abundance in one
    ## list for sorting on counts and abun
    lencounter   = Counter(lenlist)
    sizeinfo_l   = []
    for alen,acount in lencounter.items():
        aabun    = sum(abundict[alen])
        sizeinfo_l.append((alen,acount,aabun))
    sizeinfo_ls  = sorted(sizeinfo_l, key=lambda x: (-x[1],-x[2]))
    ## dominant size class based on distribution of counts
    if len(lencounter.keys()) <= 4:
        ## just four size classes
        DOMCOUNT_CUT = int(0.25*abuncount)
    else:
        DOMCOUNT_CUT = median(lencounter.values())
    ## dominant size class based on counts - very stringent
    domsize      = sizeinfo_ls[0][0] ## first element of list has most counts, store size
    domcount     = sizeinfo_ls[0][1] ## first element of list has most counts, store count
    domabun      = sizeinfo_ls[0][2] ## first element of list has most counts, store abun
    samecounts_l = [x for x in sizeinfo_ls     if  x[1] == domcount]  ## get size classes that have same counts as domsize
    domsize_l    = [x[0] for x in samecounts_l if  x[2] == domabun]   ## get size classes that have same abundance (alongwith counts) as domsize
    return domsize_l,lencounter,DOMCOUNT_CUT

def clustscore(aclust, poslist, sens, asens=None):
    """
    Scores an individual cluster and appends p-values.
    `asens` kept for signature compatibility but not used.
    """
    scoredclust = []
    cluster_pvals_f = []
    cluster_pvals_r = []
    pos_dict = {}

    for ind in range(0, len(poslist)):
        posinfo = poslist[ind]
        flist   = poslist[ind:]     # forward slice
        rlist   = poslist[:ind+1]   # reverse slice
        apos    = posinfo[0]

        # Fast labeling: set membership; no use of counts/abun for label
        flist_labeled = mapPhaseSites(posinfo, flist, "F", sens)
        rlist_labeled = mapPhaseSites(posinfo, rlist, "R", sens)

        test_values_list_f = collectstats(flist_labeled)
        test_values_list_r = collectstats(rlist_labeled)

        pval_h_f, N_f, X_f, pval_r_f, pval_corr_f, pval_h_r, N_r, X_r, pval_r_r, pval_corr_r = \
            compute_p_vals(test_values_list_f, test_values_list_r)

        pos_dict[int(apos)] = (pval_h_f, N_f, X_f, pval_r_f, pval_corr_f,
                               pval_h_r, N_r, X_r, pval_r_r, pval_corr_r)
        cluster_pvals_f.append(pval_corr_f)
        cluster_pvals_r.append(pval_corr_r)

    for taginfo in aclust:
        apos = int(taginfo[5])
        astats = pos_dict[apos]
        taginfo.extend(astats)
        scoredclust.append(taginfo)

    return scoredclust


def compute_p_vals(test_values_list_f,test_values_list_r):
    '''
    takes a list of values to compute p_values from forward and reverse direction
    of a cluster
    '''
    ## apply statistical test - hyp and ranksum
    ## to compute a pval for each position in cluster
    ## do this in both forward and reverse direction,
    ## for the latter used sens and asen in negative
    pval_h_f,N_f,X_f     = hypertest(test_values_list_f)
    pval_r_f             = ranksumtest(test_values_list_f)
    pval_corr_f          = round(stouffer([pval_h_f,pval_r_f]),15)
    pval_h_r,N_r,X_r     = hypertest(test_values_list_r)
    pval_r_r             = ranksumtest(test_values_list_r)
    pval_corr_r          = round(stouffer([pval_h_r,pval_r_r]),15)
    return pval_h_f,N_f,X_f,pval_r_f,pval_corr_f,pval_h_r,N_r,X_r,pval_r_r,pval_corr_r

def collectstats(labeled_list):
    '''
    takes a labelled list i.e. with 'p' and 'np'
    tags for a cluster and specific to a reference position;
    input list has follwing elements:
    apos, astrand, poscount(number of sRNAs at one positions),
    posabun_a_strand(abundance from all sRNAs),posabun_p_strand (abundance of sRNAs mathcing phase) and label (p and np)
    '''
    test_values_list = []
    count_p  = len([i for i in labeled_list if i[-1] == "p"])
    count_np = len([i for i in labeled_list if i[-1] == "np"])
    count_all = count_p + count_np
    test_values_list.extend((count_p, count_np, count_all))
    ## WHAT STATS YOU NEED FOR RANK SUM?
    ## Mann Whitney test requires two lists with values
    ## Here we choose all abundances from non-phased and
    ## all abundances from  phased positions; we could have
    ## also selected the abundnaces from phased positions
    ## correposnding to sRNAs that match phase length (since
    ## it's already included in the input list) but anyhow
    ## we are going to put a filter on the ratio of abundanced from
    ## phased position for phase len against abundances from non-phased
    ## postions.
    abun_p_all   =  [int(i[3]) for i in labeled_list if i[-1] == "p"] ## selecting abundances for all sizes at phased psoitions
    abun_np_all  =  [int(i[3]) for i in labeled_list if i[-1] == "np"] ## selecting abundances for all sizes from non-phased positions
    abun_p_phase =  [int(i[4]) for i in labeled_list if i[-1] == "p"]
    test_values_list.extend((abun_p_all,abun_np_all,abun_p_phase))
    # ## compute phased sRNAs (of phaselen) vs. all ratio i.e.
    # ## ph_size_prop using nesteddict fom clusterassemble
    # pos_ph_size_prop = round(sum(abun_p_phase)/(sum(abun_p_all)+sum(abun_np_all)),5)
    # test_values_list.extend(pos_ph_size_prop)
    return test_values_list

def getPhasedIndexes(WINDOW_SIZE):
    '''
    generates phased indexes for position labelling
    '''
    ## generate phase position maps for
    ## forward/reverse and for +/- strands
    regs    = list(range(0, WINDOW_SIZE))   ## a template for positions coefficiants - OK
    sens    = [i*int(phase) for i in regs]  ## forward direction to use with sRNAs on 'w' strand; for negative strand subtract - OK
    ## add dicer offsets
    dicer_off_left  = [x-1 for x in sens]
    dicer_off_right = [x+1 for x in sens]
    sens.extend(dicer_off_left)
    sens.extend(dicer_off_right)
    sens.sort()
    asens   = [i-3 for i in sens]           ## forward direction to use with sRNAs on 'c' strand; for negative strand subtract - OK
    return sens,asens

def mapPhaseSites(posinfo, poslist, direction, sens):
    """
    Label positions as phased ('p'), non-phased within window ('np'),
    or outside window ('na') relative to a reference position/strand.

    Notes:
      - Uses set membership for speed.
      - Ignores counts/abundances in poslist for labeling (only pos & strand).
      - Expects `sens` = list of integer offsets (includes wobble if desired).
      - Returns tuples: (apos, astrand, poscount, abun_all, abun_phase, label)
    """
    list_labeled = []
    refpos, refstrand = posinfo[0], posinfo[1]

    if direction == "F":
        if refstrand == "w":
            sens_p_set  = {refpos + o for o in sens}          # 'w' strand
            asens_p_set = {refpos + o - 3 for o in sens}      # 'c' strand (−3 shift)
            window_end  = max(sens_p_set)                     # 5'-most end
            for bent in poslist:
                bpos, bstrand = bent[0], bent[1]
                if (bstrand == "w" and bpos in sens_p_set) or (bstrand == "c" and bpos in asens_p_set):
                    label = "p"
                elif bpos <= window_end:
                    label = "np"
                else:
                    label = "na"
                list_labeled.append((*bent, label))
        else:  # refstrand == "c"
            sens_p_set  = {refpos + o for o in sens}          # 'c' strand
            asens_p_set = {refpos + o + 3 for o in sens}      # 'w' strand (+3 shift)
            window_end  = max(asens_p_set)                    # 5'-most end
            for bent in poslist:
                bpos, bstrand = bent[0], bent[1]
                if (bstrand == "w" and bpos in asens_p_set) or (bstrand == "c" and bpos in sens_p_set):
                    label = "p"
                elif bpos <= window_end:
                    label = "np"
                else:
                    label = "na"
                list_labeled.append((*bent, label))

    elif direction == "R":
        # Reverse direction (3' -> 5'): mirror offsets; order not needed for sets.
        if refstrand == "w":
            sens_p_set  = {refpos - o for o in sens}          # 'w' strand
            asens_p_set = {p - 3 for p in sens_p_set}         # 'c' strand (−3 shift)
            window_end  = min(asens_p_set)                    # 3'-most end
            for bent in poslist:
                bpos, bstrand = bent[0], bent[1]
                if (bstrand == "w" and bpos in sens_p_set) or (bstrand == "c" and bpos in asens_p_set):
                    label = "p"
                elif bpos >= window_end:
                    label = "np"
                else:
                    label = "na"
                list_labeled.append((*bent, label))
        else:  # refstrand == "c"
            sens_p_set  = {refpos - o for o in sens}          # 'c' strand
            asens_p_set = {p + 3 for p in sens_p_set}         # 'w' strand (+3 shift)
            window_end  = min(sens_p_set)                     # 3'-most end
            for bent in poslist:
                bpos, bstrand = bent[0], bent[1]
                if (bstrand == "w" and bpos in asens_p_set) or (bstrand == "c" and bpos in sens_p_set):
                    label = "p"
                elif bpos >= window_end:
                    label = "np"
                else:
                    label = "na"
                list_labeled.append((*bent, label))
    else:
        print(f"Unexpected input for scoring direction:{direction}")
        print("PHASIS will exit now, please contact authors")
        sys.exit()

    return list_labeled

def hypertest(test_values_list):
    '''
    input: N,K,n,k
    Computes probability from hypergeometric disribution
    https://alexlenail.medium.com/understanding-and-implementing-the-hypergeometric-test-in-python-a7db688a7458
    '''
    M       = WINDOW_SIZE*int(phase)*2  ## count of all the positions from both strands in a window i.e. 'N'
    n       = WINDOW_SIZE*2*3           ## count of expected phased position from both strands, including +1 and -1
                                        ## dicer offsets in a window i.e 'K' i.e. max successes
    N       = test_values_list[2]       ## count of all sRNAs filled positions (count_all) i.e. 'n'
    X       = test_values_list[0]       ## count of positions labelled as phased(count_p) i.e. 'k' the observed successes
    #print("Hyper pval variables - N:%s | K:%s | n:%s | k:%s" % (M,n,N,X))
    #For the hypergeomtric test the count of all sRNA filled positions (N) cannot be higher
    #than the possible number of positions from both strands in a window (M). It is biologically possible
    #because many smallRNAs fragments can overlap, but doing so they are not occuping more positions that the should
    #that why if N is bigger than N the value of N will be automatilly set to M
    hyperp  = round(hypergeom.sf(X-1, M, n, N),10)
    return hyperp,N,X

def ranksumtest(test_values_list):
    '''
    computes probability from ranksum test
    https://stats.stackexchange.com/questions/299733/how-to-interpret-wilcoxon-rank-sum-result
    Mann-Whitney with tie and continuity correction is better option but can be applied only when
    both populations have more than 20 postitiob-speciifc abundances
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
    https://www.youtube.com/watch?v=BT1FKd1Qzjw&t=0s
    Can' mix both tests for same cluster so Wilcoxon rank sum used
    '''
    abun_p_all  = test_values_list[3]
    abun_np_all = test_values_list[4]
    if not abun_p_all:
        abun_p_all = [0]
    if not abun_np_all:
        abun_np_all =  [0]
    ## Sanity check
    aset = set(abun_p_all+abun_np_all)
    if len(aset) == 1:
        rankp           = 1.0 
    else:
        mannu_res       = mannwhitneyu(abun_p_all, abun_np_all, alternative='greater') ## testing if x ranks are greater then y ranks
        rankp           = round(mannu_res[1],10)
    return rankp

def stouffer(pvals):
    '''
    combine pvals using stouffer's method
    '''
    apval = combine_pvalues(pvals, method='stouffer', weights=None)
    return apval[1]

def createfolders(currdir):
    '''
    create basic folders at the begining of process
    '''
    ## folder for storing cluster pickles
    clustfolder = "%s/%s_clusters" % (currdir,phase)
    if not os.path.isdir(clustfolder):
        os.mkdir("%s" % (clustfolder))
    else:
        pass
    return clustfolder

def FASTAclean(ent):
    '''
    Cleans one entry of FASTA file - multi-line fasta to single line, header clean, empty lines removal
    '''
    ent_splt    = ent.split('\n')
    aname       = ent_splt[0].split()[0].strip()
    if runtype == 'G':
        ## To match with phasing-core script for genome version which removed non-numeric and preceding 0s
        bname = re.sub("[^0-9]", "", aname).lstrip('0')
    else:
        bname = aname
    bseq     = ''.join(x.strip() for x in ent[1:]) ## Sequence in multiple lines
    return bname,bseq

def clustwrite(akey, clustlist):
    '''
    writes all clusters for a fragment
    '''
    #print("writting clusters")
    outfile = "%s/%s.sRNA_%s.cluster" % (scoredClustFolder,akey,phase)
    fh_out  = open(outfile,'a')
    for aclust in clustlist:
        #print(f"aclust is {aclust}")
        aid,taglist = aclust
        fh_out.write(">cluster = %s_%s_%s\n" % (akey,aid, taglist[0][0]))
        for taginfo in taglist:
            achr, astrand , ahits, atag, aname, apos, alen, abun, pval_h_f, N_f, X_f, pval_r_f, pval_corr_f, pval_h_r, N_r, X_r, pval_r_r,pval_corr_r = taginfo ## N and X stats are coming from hypergeometric test
            fh_out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (achr,astrand,apos,alen,ahits,abun,pval_h_f,N_f,X_f,pval_r_f,pval_corr_f,pval_h_r,N_r,X_r,pval_r_r,pval_corr_r,aname,atag))
    fh_out.close()
    return None

def median(lst):
    n = len(lst)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(lst)[n//2]
    else:
            return sum(sorted(lst)[n//2-1:n//2+1])/2.0

def coreReserve(cores):
    '''
    Decides the core pool for machine - written to make PHASIS comaptible with machines that
    '''
    totalcores = int(multiprocessing.cpu_count())
    if cores == 0:
        ## Automatic assignment of cores selected
        
        if totalcores   == 4: ## For quad core system
            ncores = 3
        elif totalcores == 6: ## For hexa core system
            ncores = 5
        elif totalcores > 6 and totalcores <= 10: ## For octa core system and those with less than 10 cores
            ncores = 7
        else:
            ncores = int(totalcores*0.95)
    else:
        ## Reserve user specifed cores
        if cores > totalcores:
            ncores = totalcores
        else: ncores = int(cores*0.95)

    return ncores

def optimize(ncores,nfiles):
    '''
    optimization of total processes and cores per process
    '''
    nspread     = int(ncores/nfiles) ### Number of parallel cores per process
    if nspread  < 3:
        ## enforce minimum nspread value
        ## and compute possible processes
        nspread = 3
        nproc   = int(ncores/3)
    else:
        ## nspread estimate is healthy
        ## compute possible processed
        nproc   = nfiles
    print("\n#### %s computing core(s) reserved for analysis" % (str(ncores)))
    print("#### %s computing core(s) assigned to each lib #\n" % (str(nspread)))
    return nproc,nspread

def PPBalance(module, alist):
    print("##    FN PPBalance   ######")
    n_workers = int(globals().get('nproc', 1))
    if n_workers < 1:
        n_workers = 1
    # Linux default context; no need to specify
    pool = multiprocessing.Pool(n_workers)
    errors = []
    try:
        results = []
        for i, res in enumerate(pool.imap_unordered(module, alist), 1):
            results.append(res)
        pool.close()
        pool.join()
        return results
    except Exception as e:
        print(f"[PPBalance] Error in parallel processing: {e}")
        traceback.print_exc()
        pool.terminate()
        pool.join()
        errors.append(e)
        raise
    finally:
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass
    if errors:
        print("[PPBalance] Some jobs failed, see logs above.")


def PPResults(module,alist):
    '''
    Parallelizes and stores result, uses raw size of cores
    '''
    npool   = multiprocessing.get_context("fork").Pool(int(ncores))
    res     = npool.map_async(module, alist)
    results = (res.get())
    npool.close()
    return results

#part II, cluster proccess

def process_single_lib_cluster(filename):

    clustlist = []

    # library name from file basename: AR_1_nocontam.21-PHAS.candidate.clusters -> AR_1_nocontam
    base = os.path.basename(filename)
    alib = re.sub(r'\.\d+-PHAS\.candidate\.clusters$', '', base)

    with open(filename) as fh:
        lines = fh.readlines()

    aid = None
    for line in lines:
        if line.startswith('>'):
            # header like: ">cluster = lobe_3_nocontam-1_3894_1"
            m = re.search(r'cluster\s*=\s*([^\s]+)', line)
            if not m:
                aid = None
                continue
            aid = m.group(1).strip()              # e.g. lobe_3_nocontam-1_3894_1
            continue

        # data lines belong to the most recent header (aid)
        if not aid:
            continue

        ent = line.rstrip('\n').split('\t')
        achr            = str(ent[0])
        astrand         = str(ent[1])
        apos            = int(ent[2])
        alen            = int(ent[3])
        ahits           = int(ent[4])
        abun            = int(ent[5])
        pval_h_f        = float(ent[6])
        N_f             = int(ent[7])
        X_f             = int(ent[8])
        pval_r_f        = float(ent[9])
        pval_corr_f     = float(ent[10])
        pval_h_r        = float(ent[11])
        N_r             = int(ent[12])
        X_r             = int(ent[13])
        pval_r_r        = float(ent[14])
        pval_corr_r     = float(ent[15])
        tag_id          = str(ent[16])
        tag_seq         = str(ent[17])

        # clusterID is the clean per-lib id (no filename glue)
        clustlist.append((
            alib, aid, achr, astrand, apos, alen, ahits, abun,
            pval_h_f, N_f, X_f, pval_r_f, pval_corr_f, pval_h_r, N_r, X_r,
            pval_r_r, pval_corr_r, tag_id, tag_seq
        ))

    return clustlist


def aggregate_and_write_processed_clusters(clusterFiles):
    print("### Aggregating and processing candidate cluster files per library ###")

    # --- Parallel processing ---
    # Use run_parallel_with_progress for multicore support (you must have it in your module!)
    all_clustlists = run_parallel_with_progress(
        process_single_lib_cluster,
        clusterFiles,
        desc="Aggregating cluster files",
        min_chunk=1,
        unit="lib"
    )
    # Flatten list of lists
    flat_clustlist = [item for sublist in all_clustlists for item in sublist]
    allClusters = pd.DataFrame(flat_clustlist, columns=[
        "alib", "clusterID", "chromosome", "strand", "pos", "len", "hits", "abun",
        "pval_h_f", "N_f", "X_f", "pval_r_f", "pval_corr_f", "pval_h_r", "N_r", "X_r",
        "pval_r_r", "pval_corr_r", "tag_id", "tag_seq"
    ])
    allClusters = allClusters.sort_values(by=["clusterID", "pos"])

    outfname = phase2_basename('processed_clusters.tab')
    allClusters.to_csv(outfname, sep="\t", index=False, header=True)

    # --- Update hash in memory file ---
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)
    section = 'PROCESSED'
    if not config.has_section(section):
        config.add_section(section)
    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        config[section][outfname] = out_md5
        print(f"Hash for {outfname}: {out_md5}")
    with open(memFile, 'w') as fh:
        config.write(fh)

    print(f"Processed clusters written to {outfname}")
    return allClusters

def _load_mergedClusterDict_from_tab(filename):
    """Load dict back from TSV: key \t values..."""
    out = defaultdict(list)
    if not os.path.isfile(filename):
        return {}
    with open(filename, 'r', newline='') as fh:
        rdr = csv.reader(fh, delimiter='\t')
        for row in rdr:
            if not row:
                continue
            key, vals = row[0], row[1:]
            out[key].extend(vals)
    return dict(out)

# ---- Grouping for parallel ----
def group_loci_by_chromosome_for_parallel(loci_by_chr):
    """Convert pandas groupby('chr') object into list-of-lists per chromosome."""
    return [group.values.tolist() for _, group in loci_by_chr]

# ---- Step 1: build pairs of overlapping loci per chromosome ----
def merge_loci_pairs_by_chromosome(loci_group):
    """
    loci_group: list of [name, pval, chr, start, end] (all same chr).
    Returns list of [A,B] pairs; adds [X,'singleLibOccurrence'] only for IDs
    that never paired with anyone.
    """
    # 1) normalize to compact records: [name, start, end]
    L = []
    for name, pval, chr_, start, end in loci_group:
        L.append([str(name).strip(), int(start), int(end)])

    n = len(L)
    if n <= 1:
        return [[L[0][0], 'singleLibOccurrence']] if n == 1 else []

    # 2) O(n) monotonicity check on (start, end)
    maybe_sorted = True
    ps, pe = L[0][1], L[0][2]
    for i in range(1, n):
        cs, ce = L[i][1], L[i][2]
        if cs < ps or (cs == ps and ce < pe):
            maybe_sorted = False
            break
        ps, pe = cs, ce

    # 3) Only sort if needed (by start, then end)
    if not maybe_sorted:
        L.sort(key=lambda r: (r[1], r[2]))

    # 4) pair generation with early break and true-singleton tracking
    pairs = []
    paired_ids = set()
    all_ids = {r[0] for r in L}  # set avoids duplicates

    for i in range(n):
        aname, astart, aend = L[i]
        for j in range(i + 1, n):
            bname, bstart, bend = L[j]

            # since L is sorted by start, once bstart is beyond a’s buffered end we can stop
            if bstart > aend + clustbuffer:
                break

            # buffered overlap test
            if (bend >= astart - clustbuffer) and (bstart <= aend + clustbuffer):
                pairs.append([aname, bname])
                paired_ids.add(aname)
                paired_ids.add(bname)

    # 5) emit singletons that never appeared in any pair
    for name in all_ids - paired_ids:
        pairs.append([name, 'singleLibOccurrence'])

    return pairs

# ---- Single-library preprocessing ----
def preprocess_single_library_clusters(mergedClusters):
    mergedClusterDict = defaultdict(list)
    for acount, apair in enumerate(mergedClusters):
        aPname = apair[0]
        mergedClusterDict[f"cluster_{acount}"].append(aPname)
    return mergedClusterDict

# ---- Step 2: assemble candidate sets per chromosome ----
def assemble_clusters_by_chromosome(mergedClusters_chr):
    """
    Build merged sets per chromosome from pair list produced by merge_loci_pairs_by_chromosome.
    Handles single-library case inside.
    """
    mergedClusterDict = defaultdict(list)
    assigned = set()
    acount = 0

    for apair in mergedClusters_chr:
        aPname, bPname = apair[0], apair[1]

        # Single library: just bucket everything 1-per-cluster (preProcess handles whole chr)
        if aPname == bPname:
            print("single-library chromosome: collapsing pairs")
            mergedClusterDict = preprocess_single_library_clusters(mergedClusters_chr)
            break

        # Pair merging mode
        if bPname != 'singleLibOccurrence':
            if acount == 0 or aPname not in assigned:
                key = f"cluster_{acount}"
                mergedClusterDict[key].extend([aPname, bPname])
                assigned.update([aPname, bPname])

                # bring in more neighbors that share the same anchor (aPname)
                for cPname, dPname in mergedClusters_chr:
                    if (aPname, bPname) == (cPname, dPname):
                        continue
                    if aPname == cPname and dPname not in assigned and dPname != 'singleLibOccurrence':
                        mergedClusterDict[key].append(dPname)
                        assigned.add(dPname)
                acount += 1

        # Handle 'singleLibOccurrence' residuals
        elif bPname == 'singleLibOccurrence':
            if aPname not in assigned:
                mergedClusterDict[f"cluster_{acount}"].append(aPname)
                assigned.add(aPname)
            acount += 1

    return mergedClusterDict

# ---- Step 3: assign final IDs per chromosome (genomic span keys) ----
def assign_final_ids_by_chromosome(chromosome_df):
    """
    Renames merged cluster IDs into genomic span keys (chrom:start..stop) per chromosome.
    Uses a global mapping prepared by assign_final_cluster_ids().
    """
    reNamesClusterDict = defaultdict(set)

    chromosome_df['pos'] = pd.to_numeric(chromosome_df['pos'], errors='coerce')
    chromosome_df['clusterID'] = chromosome_df['clusterID'].astype(str).str.strip()

    cluster_info = chromosome_df.groupby('clusterID').agg(
        start=('pos', 'min'),
        stop=('pos', 'max')
    ).to_dict('index')

    chromosome = chromosome_df['chromosome'].iloc[0]

    for entry in copy_mergedClusterDict_global:
        cluster_ids = copy_mergedClusterDict_global[entry]
        valid_cluster_ids = [cid for cid in cluster_ids if cid in cluster_info]
        if not valid_cluster_ids:
            continue

        if len(valid_cluster_ids) == 1:
            info = cluster_info[valid_cluster_ids[0]]
            start, stop = info['start'], info['stop']
        else:
            starts = [cluster_info[cid]['start'] for cid in valid_cluster_ids]
            stops  = [cluster_info[cid]['stop']  for cid in valid_cluster_ids]
            start, stop = min(starts), max(stops)

        reNamesClusterDict[f"{chromosome}:{start}..{stop}"].update(valid_cluster_ids)

    return {k: list(v) for k, v in reNamesClusterDict.items()}

def assign_final_cluster_ids(mergedClusterDict, allClusters):
    """
    Parallel per-chromosome remapping of mergedClusterDict -> genomic span IDs,
    then flattened back into one dict with unique clusterIDs per span.
    """
    chromosome_groups = [df for _, df in allClusters.groupby("chromosome")]

    # sanitize keys/vals
    copy_merged = {k.strip(): [cid.strip() for cid in v]
                   for k, v in mergedClusterDict.items()}

    global copy_mergedClusterDict_global
    copy_mergedClusterDict_global = copy_merged

    results = run_parallel_with_progress(
        assign_final_ids_by_chromosome,
        chromosome_groups,
        desc="Assign final cluster IDs",
        min_chunk=1,
        unit="lib-chr"
    )

    flattened = defaultdict(list)
    assigned = set()
    for rd in results:
        for k, vals in rd.items():
            new_vals = [v for v in vals if v not in assigned]
            assigned.update(new_vals)
            flattened[k].extend(new_vals)

    del copy_mergedClusterDict_global
    return dict(flattened)


def _load_mergedClusterDict_from_tab(path):
    d = defaultdict(list)
    if not os.path.isfile(path):
        return d
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            d[parts[0]].extend([p for p in parts[1:] if p])
    return d

def ensure_mergedClusterDict(phase: str):
    """
    Light loader only:
      1) return cached globals()['mergedClusterDict'] if available,
      2) else load {phase}_mergedClusterDict.tab,
      3) else identity from {phase}_merged_candidates.tab,
      4) else identity from {phase}_processed_clusters.tab,
      5) else {}.
    (No heavy cross-lib merging here.)
    """
    dict_tab   = phase2_basename('mergedClusterDict.tab')
    merged_tsv = phase2_basename('merged_candidates.tab')
    proc_tab   = phase2_basename('processed_clusters.tab')

    with _MERGED_DICT_LOCK:
        g = globals()
        if isinstance(g.get("mergedClusterDict"), dict) and g["mergedClusterDict"]:
            return g["mergedClusterDict"]

        # 2) dict tab
        if os.path.isfile(dict_tab):
            try:
                g["mergedClusterDict"] = _load_mergedClusterDict_from_tab(dict_tab)
                return g["mergedClusterDict"]
            except Exception as e:
                print(f"[WARN] Failed to load {dict_tab}: {e}")

        # 3) identity from merged candidates
        if os.path.isfile(merged_tsv):
            try:
                ident = _identity_dict_from_tsv_firstcol(merged_tsv)
                g["mergedClusterDict"] = ident
                # persist for next time
                with open(dict_tab, "w", newline="") as fh:
                    wr = csv.writer(fh, delimiter="\t")
                    for k, vs in ident.items():
                        wr.writerow([k] + vs)
                return g["mergedClusterDict"]
            except Exception as e:
                print(f"[WARN] Failed to derive identity dict from {merged_tsv}: {e}")

        # 4) identity from processed clusters
        if os.path.isfile(proc_tab):
            try:
                ident = _identity_dict_from_tsv_firstcol(proc_tab, id_col=("clusterID","Cluster","name","cID"))
                g["mergedClusterDict"] = ident
                with open(dict_tab, "w", newline="") as fh:
                    wr = csv.writer(fh, delimiter="\t")
                    for k, vs in ident.items():
                        wr.writerow([k] + vs)
                return g["mergedClusterDict"]
            except Exception as e:
                print(f"[WARN] Failed to derive identity dict from {proc_tab}: {e}")

        # 5) empty fallback
        g["mergedClusterDict"] = {}
        return g["mergedClusterDict"]


MERGED_REVERSE_BUILT = False

def _ensure_reverse_index() -> dict:
    mcd = globals().get("mergedClusterDict", {}) or {}
    rev = globals().get("MERGED_CLUSTER_REVERSE")
    if not rev:
        rev = {}
        for u, members in mcd.items():
            for cid in members:
                key = _normalize_cluster_id_for_lookup(cid)
                if key:
                    rev[key] = str(u)
                # (optional forgiving keys)
                raw = str(cid).strip()
                if raw:
                    rev[raw] = str(u)
        globals()["MERGED_CLUSTER_REVERSE"] = rev
        globals()["mergedClusterReverse"]  = rev  # compat alias
    return rev

def getUniversalID(clusterID):
    """Return universal coord ID (chr:start..end) for a per-lib clusterID."""
    ensure_mergedClusterDict(phase)
    _ensure_reverse_index()
    s = str(clusterID).strip()
    # If caller already passed a universal ID, return it
    if ":" in s and ".." in s:
        return s
    key = _normalize_cluster_id_for_lookup(s)
    rev = globals().get("mergedClusterReverse", {})
    return rev.get(key)


def _strip_fileprefix_from_id(cid: str,
                              lib: str | None = None,
                              phase: str | int | None = None) -> str:
    """
    Remove glued filename/prefix from cluster IDs like:
      ALL_LIBS.21-PHAS.candidate.clustersALL_LIBS-10_120402_10  ->  10_120402_10
    Works with/without lib and phase; returns input if no match.
    Leaves universal IDs (chr:start..end) untouched.
    """
    s = str(cid)
    if ":" in s and ".." in s:  # already universal
        return s

    pats = []
    if phase is not None:
        p = re.escape(str(phase))
        if lib:
            pats.append(rf"^{re.escape(lib)}\.{p}-PHAS\.candidate\.clusters{re.escape(lib)}-(.+)$")
        # generic with any lib after 'clusters'
        pats.append(rf"^[^.]+\.{p}-PHAS\.candidate\.clusters[^-]*-(.+)$")
    # broad fallback: anything up to '.PHAS.candidate.clusters' then a '-' then the real ID
    pats.append(r"^.+?\.PHAS\.candidate\.clusters[^-]*-(.+)$")

    for pat in pats:
        m = re.match(pat, s)
        if m:
            return m.group(1)
    return s

def _normalize_cluster_id_for_lookup(x: str) -> str:
    """Stringify, strip whitespace, and strip fileprefix based on `phase`.
    Leaves coordinate universal IDs (chr:start..end) untouched."""
    s = str(x).strip()
    # If already universal, keep as-is
    if ":" in s and ".." in s:
        return s
    return _strip_fileprefix_from_id(s, phase=phase)

def process_chromosome_data(loci_group):
    """
    Process data for a single chromosome-library group.
    STRICT: expects 20-column per-read/per-alignment rows with the schema below.
    This avoids accidental use of 6-col merged-candidates rows, which lack per-read detail.
    """
    expected_cols = [
        "alib","clusterID","chromosome","strand","pos","len","hits","abun",
        "pval_h_f","N_f","X_f","pval_r_f","pval_corr_f","pval_h_r","N_r",
        "X_r","pval_r_r","pval_corr_r","tag_id","tag_seq"
    ]

    if not loci_group:
        return pd.DataFrame(columns=expected_cols + ["identifier"])

    width = len(loci_group[0])
    if width != 20:
        print(f"[WARN] process_chromosome_data got {width} columns (expected 20). "
              "Did you pass merged-candidates instead of processed-clusters? Skipping group.")
        return pd.DataFrame(columns=expected_cols + ["identifier"])

    df = pd.DataFrame(loci_group, columns=expected_cols)

    # light dtype normalization used later downstream
    df["pos"]  = pd.to_numeric(df["pos"], errors="coerce")
    df["len"]  = pd.to_numeric(df["len"], errors="coerce")
    df["abun"] = pd.to_numeric(df["abun"], errors="coerce")
    df = df.dropna(subset=["pos","len"]).reset_index(drop=True)

    # Attach universal identifier (ensure mergedClusterDict has been prepared earlier)
    df["identifier"] = df["clusterID"].map(getUniversalID)

    # Drop rows we can't map
    df = df.dropna(subset=["identifier"]).reset_index(drop=True)
    return df

def assemble_candidate_clusters_parametric(
    mergedClusters: Sequence[Sequence[Any]],
    allClusters_df: pd.DataFrame,
    phase: str,
) -> Dict[str, List[str]]:

    def _norm(x: Any) -> str | None:
        """Normalize cluster id: stringify, strip, drop empty, strip prefix."""
        if x is None:
            return None
        s = str(x).strip()
        # NEW: drop sentinel rows entirely
        if not s or s == "singleLibOccurrence":
            return None
        return _strip_fileprefix_from_id(s, phase=phase)

    # --- NEW: build clusterID -> (chr, start, end) from loci table -----------
    cid2coord: Dict[str, tuple[str,int,int]] = {}
    try:
        loci_path = phase2_basename('candidate.loci_table.tab')
        ldf = pd.read_csv(loci_path, sep="\t")
        # normalize headers
        ldf = ldf.rename(columns={
            "Cluster":"name", "value1":"pval",
            "chromosome":"chr", "Start":"start", "End":"end"
        })
        # standardize types
        ldf["chr"] = ldf["chr"].astype(str)
        ldf["start"] = ldf["start"].astype(int)
        ldf["end"] = ldf["end"].astype(int)
        # keys need to match _norm(...)
        for _, row in ldf.iterrows():
            k = _norm(row["name"])
            if k is not None:
                cid2coord[k] = (row["chr"], int(row["start"]), int(row["end"]))
    except Exception as e:
        # stay permissive; we still produce groups, but keys may fallback
        print(f"[WARN] assemble(): failed to build coord map from loci table: {e}")

    # Union-Find / Disjoint Set (unchanged) -----------------------------------
    parent: Dict[str, str] = {}
    rank: Dict[str, int] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        ra_rank = rank.get(ra, 0)
        rb_rank = rank.get(rb, 0)
        if ra_rank < rb_rank:
            parent[ra] = rb
        elif ra_rank > rb_rank:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] = ra_rank + 1

    # 1) add edges from mergedClusters (unchanged except _norm handles sentinel)
    if mergedClusters:
        for pair in mergedClusters:
            if not pair:
                continue
            a = _norm(pair[0]) if len(pair) >= 1 else None
            b = _norm(pair[1]) if len(pair) >= 2 else None
            if a:
                parent.setdefault(a, a)
            if b:
                parent.setdefault(b, b)
            if a and b:
                union(a, b)

    # 2) add singletons from allClusters_df (unchanged)
    try:
        if allClusters_df is not None and 'clusterID' in allClusters_df.columns:
            for raw in allClusters_df['clusterID'].astype(str):
                cid = _norm(raw)
                if cid and cid not in parent:
                    parent[cid] = cid
    except Exception:
        pass

    if not parent:
        return {}

    # 3) connected components (unchanged)
    comps: Dict[str, set] = {}
    for node in list(parent.keys()):
        root = find(node)
        comps.setdefault(root, set()).add(node)

    # 4) produce universal IDs as COORDINATE KEYS (changed block) -------------
    mergedClusterDict: Dict[str, List[str]] = {}
    for members in comps.values():
        if not members:
            continue
        m_sorted = sorted(members)
        # NEW: compute coordinate span across members (same chr by construction)
        coords = [cid2coord.get(m) for m in m_sorted if m in cid2coord]
        coords = [c for c in coords if c is not None]
        if coords:
            achr = coords[0][0]
            smin = min(s for _, s, _ in coords)
            emax = max(e for _, _, e in coords)
            key = f"{achr}:{smin}..{emax}"
        else:
            # fallback to previous behavior if no coords available
            key = m_sorted[0]
        mergedClusterDict[key] = m_sorted

    return mergedClusterDict

def merge_candidate_clusters_parametric(loci_df: pd.DataFrame,
                                        allClusters_df: pd.DataFrame,
                                        phase: str,
                                        memFile: str):
    """
    Cross-library merge based on loci overlap, independent of concat mode.
    - loci_df columns: ['name','pval','chr','start','end'] (strings OK)
    - allClusters_df: DataFrame used to decide number of libraries (allClusters_df['alib'])
    Writes:
      {phase}_merged_clusters.tab
      {phase}_mergedClusterDict.tab
    Returns:
      mergedClusterDict (dict: universal_id -> [member clusterIDs])
    """
    print("### Merging candidate clusters across libraries (per chromosome) ###")
    # normalize loci_df columns just in case
    loci_df = loci_df.rename(columns={"Cluster":"name","value1":"pval","chromosome":"chr","Start":"start","End":"end"})

    merged_pairs_path = phase2_basename('merged_clusters.tab')
    merged_dict_path  = phase2_basename('mergedClusterDict.tab')

    # Hash-aware skip
    cfg = configparser.ConfigParser(); cfg.optionxform = str
    cfg.read(memFile)
    if not cfg.has_section('MERGED_CLUSTERS'): cfg.add_section('MERGED_CLUSTERS')
    if not cfg.has_section('MERGED_DICT'):     cfg.add_section('MERGED_DICT')

    if os.path.isfile(merged_pairs_path) and os.path.isfile(merged_dict_path):
        _, h_pairs = getmd5(merged_pairs_path)
        _, h_dict  = getmd5(merged_dict_path)
        if mem_get(cfg, 'MERGED_CLUSTERS', merged_pairs_path) == h_pairs and \
           mem_get(cfg, 'MERGED_DICT', merged_dict_path)     == h_dict:
            print("Outputs up-to-date (hash match). Skipping merge computation.")
            if '_load_mergedClusterDict_from_tab' in globals():
                return _load_mergedClusterDict_from_tab(merged_dict_path)
            # minimal loader fallback
            out = {}
            with open(merged_dict_path, 'r') as fh:
                for line in fh:
                    parts = line.rstrip('\n').split('\t')
                    if not parts: continue
                    key = parts[0].strip()
                    vals = [v for v in (p.strip() for p in parts[1:]) if v]
                    out[key] = vals if vals else [key]
            return out

    # ---- build overlap pairs per chromosome (parallel if multi-lib) ----
    mergedClusters = []
    try:
        nlibs = int(allClusters_df['alib'].nunique())
    except Exception:
        nlibs = 1

    loci_df_sorted = loci_df.sort_values(['chr','start','end'], ascending=True)
    if nlibs >= 2:
        groups_as_lists = group_loci_by_chromosome_for_parallel(loci_df_sorted.groupby('chr'))
        results = run_parallel_with_progress(
            merge_loci_pairs_by_chromosome,
            groups_as_lists,
            desc="Find overlapping loci across libs",
            min_chunk=1,
            unit="lib-chr"
        )
        mergedClusters = [pair for sub in results for pair in sub]
    else:
        # single-library pass-through (identity)
        for aname, apval, achr, astart, aend in loci_df_sorted[['name','pval','chr','start','end']].itertuples(index=False):
            mergedClusters.append([aname, aname])

    # write pairs (tab)
    with open(merged_pairs_path, 'w') as fh:
        for pair in mergedClusters:
            fh.write('\t'.join(map(str, [e for e in pair if str(e).strip()])) + '\n')

    # assemble dictionary and assign final IDs
    mcd = assemble_candidate_clusters_parametric(mergedClusters, allClusters_df, phase)

    # write dict (key \t values…)
    with open(merged_dict_path, 'w', newline='') as fh:
        wr = csv.writer(fh, delimiter='\t')
        for key, values in mcd.items():
            wr.writerow([key] + values)

    # update hashes
    if os.path.isfile(merged_pairs_path):
        _, hp = getmd5(merged_pairs_path); mem_set(cfg, 'MERGED_CLUSTERS', merged_pairs_path, hp)
    if os.path.isfile(merged_dict_path):
        _, hd = getmd5(merged_dict_path);  mem_set(cfg, 'MERGED_DICT', merged_dict_path, hd)
    with open(memFile, 'w') as fh: cfg.write(fh)

    return mcd

_MERGED_DICT_LOCK = threading.Lock()
_MERGED_REVERSE_BUILT = False

def _load_mergedClusterDict_from_tab(path: str):
    out = {}
    with open(path, "r") as fh:
        rd = csv.reader(fh, delimiter="\t")
        for row in rd:
            if not row:
                continue
            key = str(row[0]).strip()
            vals = [v for v in (str(x).strip() for x in row[1:]) if v]
            out[key] = vals if vals else [key]
    return out

def _identity_dict_from_tsv_firstcol(path: str, id_col=("Cluster","cluster","clusterID","name","cID")):
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
    except Exception:
        # very loose fallback: no header
        df = pd.read_csv(path, sep="\t", header=None, engine="python")
        if df.shape[1] >= 1:
            df.columns = ["Cluster"] + [f"col{i}" for i in range(2, df.shape[1]+1)]
    # pick a usable ID column
    col = next((c for c in id_col if c in df.columns), df.columns[0])
    ids = df[col].astype(str).tolist()
    return {cid: [cid] for cid in ids}

def ensure_mergedClusterDict(phase: str):
    """
    Light loader only:
      1) return cached globals()['mergedClusterDict'] if available,
      2) else load {phase}_mergedClusterDict.tab,
      3) else identity from {phase}_merged_candidates.tab,
      4) else identity from {phase}_processed_clusters.tab,
      5) else {}.
    (No heavy cross-lib merging here.)
    """
    dict_tab   = phase2_basename('mergedClusterDict.tab')
    merged_tsv = phase2_basename('merged_candidates.tab')
    proc_tab   = phase2_basename('processed_clusters.tab')

    with _MERGED_DICT_LOCK:
        g = globals()
        if isinstance(g.get("mergedClusterDict"), dict) and g["mergedClusterDict"]:
            return g["mergedClusterDict"]

        # 2) dict tab
        if os.path.isfile(dict_tab):
            try:
                g["mergedClusterDict"] = _load_mergedClusterDict_from_tab(dict_tab)
                return g["mergedClusterDict"]
            except Exception as e:
                print(f"[WARN] Failed to load {dict_tab}: {e}")

        # 3) identity from merged candidates
        if os.path.isfile(merged_tsv):
            try:
                ident = _identity_dict_from_tsv_firstcol(merged_tsv)
                g["mergedClusterDict"] = ident
                # persist for next time
                with open(dict_tab, "w", newline="") as fh:
                    wr = csv.writer(fh, delimiter="\t")
                    for k, vs in ident.items():
                        wr.writerow([k] + vs)
                return g["mergedClusterDict"]
            except Exception as e:
                print(f"[WARN] Failed to derive identity dict from {merged_tsv}: {e}")

        # 4) identity from processed clusters
        if os.path.isfile(proc_tab):
            try:
                ident = _identity_dict_from_tsv_firstcol(proc_tab, id_col=("clusterID","Cluster","name","cID"))
                g["mergedClusterDict"] = ident
                with open(dict_tab, "w", newline="") as fh:
                    wr = csv.writer(fh, delimiter="\t")
                    for k, vs in ident.items():
                        wr.writerow([k] + vs)
                return g["mergedClusterDict"]
            except Exception as e:
                print(f"[WARN] Failed to derive identity dict from {proc_tab}: {e}")

        # 5) empty fallback
        g["mergedClusterDict"] = {}
        return g["mergedClusterDict"]

# Optional: reverse index for O(1) member->universal lookups
def _ensure_reverse_index():
    global _MERGED_REVERSE_BUILT
    if _MERGED_REVERSE_BUILT:
        return
    mcd = globals().get("mergedClusterDict") or {}
    rev = {}
    for uid, members in mcd.items():
        uid_s = str(uid).strip()
        rev[uid_s] = uid_s  # universal as member too
        for m in members:
            rev[str(m).strip()] = uid_s
    globals()["mergedClusterReverse"] = rev
    _MERGED_REVERSE_BUILT = True

# If you use a glued prefix like "ALL_LIBS.21-PHAS.candidate.clusters<real_id>"
import re


# === Howell score safety: clamp cycles to [0, 10] ===
def _howell_clamp_cycles_10(num_cycles: int) -> int:
    try:
        n = int(num_cycles)
    except Exception:
        # Fall back to 0 if anything unexpected happens
        n = 0
    if n < 0:
        return 0
    if n > 10:
        return 10
    return n


def _strip_fileprefix_from_id(cid: str, alib: str = "", phase: str = "") -> str:
    s = str(cid)
    if phase:
        pat = rf".*?\.{re.escape(str(phase))}-PHAS\.candidate\.clusters"
        s = re.sub(pat, "", s)
    return s.strip()

def getUniversalID(clusterID):
    """
    Map a raw clusterID to its universal ID using mergedClusterDict.
    Tries exact, then prefix-stripped, then reverse index.
    """
    ensure_mergedClusterDict(phase)   # make sure dict is loaded
    _ensure_reverse_index()

    cid_raw = str(clusterID).strip()
    cid_stripped = _strip_fileprefix_from_id(cid_raw, phase=phase)

    mcd = globals().get("mergedClusterDict", {})
    if cid_raw in mcd:      # already universal
        return cid_raw
    if cid_stripped in mcd:
        return cid_stripped

    rev = globals().get("mergedClusterReverse", {})
    if cid_raw in rev:
        return rev[cid_raw]
    if cid_stripped in rev:
        return rev[cid_stripped]
    return None


# --- helpers ---------------------------------------------------------------

def _set_reverse_merged_map(mcd: dict) -> None:
    """Cache clusterID -> universalID reverse map in globals()."""
    rev = {}
    for u, members in mcd.items():
        for cid in members:
            rev[str(cid)] = str(u)
    globals()["MERGED_CLUSTER_REVERSE"] = rev


def _load_simple_tab_dict(path: str) -> dict:
    """Load a key \t values... tab into {key: [values...]}; tolerate empty lines."""
    out = {}
    with open(path, "r") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            key = parts[0].strip()
            vals = [v for v in (p.strip() for p in parts[1:]) if v]
            out[key] = vals if vals else [key]
    return out


# --- ensure mergedClusterDict (ALWAYS builds universal IDs) ----------------

def ensure_mergedClusterDict_always(concat_libs: bool,
                                    phase: str,
                                    merged_out_path: str,
                                    loci_table_df: pd.DataFrame,
                                    allClusters_df: pd.DataFrame,
                                    memFile: str) -> dict:
    """
    Always return a dict universal_id -> [member clusterIDs] and ensure the tab exists.
    In concat mode we derive universal IDs (chr:start..end) from {phase}_merged_candidates.tab.
    In non-concat mode we do the real cross-lib merge via the parametric path.
    Also caches a reverse map (clusterID -> universalID) for later lookups.
    """
    dict_tab = phase2_basename('mergedClusterDict.tab')

    # If a dict tab already exists, load and cache both directions.
    if os.path.isfile(dict_tab):
        try:
            mcd = _load_simple_tab_dict(dict_tab)
            globals()["mergedClusterDict"] = mcd
            _set_reverse_merged_map(mcd)
            return mcd
        except Exception as e:
            print(f"[WARN] Failed to load {dict_tab}: {e}. Recomputing…")

    if concat_libs:
        # Build universal IDs from the merged candidates TSV:
        #   expect columns like: Cluster, chromosome/chr, Start/start, End/end
        if not os.path.isfile(merged_out_path):
            raise FileNotFoundError(f"Missing merged candidates TSV: {merged_out_path}")

        try:
            mdf = pd.read_csv(merged_out_path, sep="\t", engine="python")
        except Exception:
            mdf = pd.read_csv(merged_out_path, sep="\t", header=None, engine="python")
            if mdf.shape[1] >= 1:
                mdf.columns = ["Cluster"] + [f"col{i}" for i in range(2, mdf.shape[1] + 1)]

        cid_col   = next((c for c in ("Cluster","clusterID","name","cID") if c in mdf.columns), None)
        chr_col   = next((c for c in ("chromosome","chr")               if c in mdf.columns), None)
        start_col = next((c for c in ("Start","start","begin")          if c in mdf.columns), None)
        end_col   = next((c for c in ("End","end","stop")               if c in mdf.columns), None)
        if not all([cid_col, chr_col, start_col, end_col]):
            raise ValueError(f"{merged_out_path} lacks required columns (have: {list(mdf.columns)})")

        from collections import defaultdict
        mcd = defaultdict(list)
        for cid, achr, s, e in mdf[[cid_col, chr_col, start_col, end_col]].itertuples(index=False):
            try:
                u = f"{str(achr)}:{int(s)}..{int(e)}"
            except Exception:
                # be forgiving if Start/End are strings already containing ints
                u = f"{str(achr)}:{str(s)}..{str(e)}"
            mcd[u].append(str(cid))

        # dedup + sort for stability
        mcd = {u: sorted(set(vs)) for u, vs in mcd.items()}

        # persist tab
        with open(dict_tab, "w", newline="") as fh:
            wr = csv.writer(fh, delimiter="\t")
            for key, values in mcd.items():
                wr.writerow([key] + values)

        # cache both directions
        globals()["mergedClusterDict"] = mcd
        _set_reverse_merged_map(mcd)
        return mcd

    # Non-concat: perform true cross-lib merge (already returns universal IDs)
    mcd = merge_candidate_clusters_parametric(loci_table_df, allClusters_df, phase, memFile)

    # persist (for fast reload next runs)
    with open(dict_tab, "w", newline="") as fh:
        wr = csv.writer(fh, delimiter="\t")
        for key, values in mcd.items():
            wr.writerow([key] + values)

    globals()["mergedClusterDict"] = mcd
    _set_reverse_merged_map(mcd)
    return mcd


# --- light loader used by getUniversalID() --------------------------------

def ensure_mergedClusterDict(phase: str) -> dict:
    """
    Light loader:
      1) return cached globals()['mergedClusterDict'] if available,
      2) else load {phase}_mergedClusterDict.tab,
      3) else empty dict.
    Never builds identity maps here (we want universal IDs only).
    """
    with _MERGED_DICT_LOCK:
        g = globals()
        if "mergedClusterDict" in g and isinstance(g["mergedClusterDict"], dict) and g["mergedClusterDict"]:
            return g["mergedClusterDict"]

        dict_tab = phase2_basename('mergedClusterDict.tab')
        if os.path.isfile(dict_tab):
            try:
                mcd = _load_simple_tab_dict(dict_tab)
                g["mergedClusterDict"] = mcd
                _set_reverse_merged_map(mcd)
                return mcd
            except Exception as e:
                print(f"[WARN] Failed to load {dict_tab}: {e}")

        g["mergedClusterDict"] = {}
        g["MERGED_CLUSTER_REVERSE"] = {}
        return g["mergedClusterDict"]


def merge_candidate_clusters_across_libs(loci_table_path: str, out_path: str) -> pd.DataFrame:
    """
    Produce the per-(chromosome, library) merged candidates and write `out_path`.
    On cache hit, LOAD + RETURN the cached file so callers always get a DataFrame.

    In --concat_libs mode (single logical library), the merge is effectively a
    pass-through of the loci table with alib="ALL_LIBS".
    """
    print("### Merging candidate clusters across libraries (per chromosome) ###")

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)

    section = "MERGED_CANDIDATES"
    if not config.has_section(section):
        config.add_section(section)

    # --- Cache check
    if os.path.isfile(out_path):
        _, curr_md5 = getmd5(out_path)
        prev_md5 = config[section].get(out_path)
        if prev_md5 and prev_md5 == curr_md5:
            print("Outputs up-to-date (hash match). Skipping merge computation.")
            df_cached = pd.read_csv(out_path, sep="\t")

            # Normalize required columns
            if "chromosome" not in df_cached.columns and "chr" in df_cached.columns:
                df_cached = df_cached.rename(columns={"chr": "chromosome"})
            if "alib" not in df_cached.columns and concat_libs:
                df_cached["alib"] = "ALL_LIBS"   # only in concat mode
            # If not concat and 'alib' is missing, warn but still return (failsafe)
            if "alib" not in df_cached.columns and not concat_libs:
                print("[WARN] Cached merged table lacks 'alib' in non-concat mode.")
            return df_cached

    # --- Compute (for concat_libs this is a pass-through of the loci table)
    if not os.path.isfile(loci_table_path):
        print(f"[WARN] Loci table not found: {loci_table_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    merged_df = pd.read_csv(loci_table_path, sep="\t")

    # Normalize required columns
    if "chromosome" not in merged_df.columns and "chr" in merged_df.columns:
        merged_df = merged_df.rename(columns={"chr": "chromosome"})
    if "alib" not in merged_df.columns:
        # In concat mode, set the single logical library id
        if concat_libs:
            merged_df["alib"] = "ALL_LIBS"
        else:
            # Non-concat: don't guess; warn and provide a safe default.
            # Downstream grouping will still work but may be degenerate; your aggregator should
            # ideally have written 'alib' per row already for non-concat runs.
            print("[WARN] 'alib' missing in loci table on non-concat run; setting 'alib'='UNKNOWN'.")
            merged_df["alib"] = "UNKNOWN"

    # Persist + hash
    merged_df.to_csv(out_path, sep="\t", index=False)
    _, new_md5 = getmd5(out_path)
    config[section][out_path] = new_md5
    with open(memFile, "w") as fh:
        config.write(fh)
    print(f"Hash for {os.path.basename(out_path)}:")

    return merged_df

def load_processed_clusters_fallback(phase: str) -> pd.DataFrame:
    """
    Module-level helper: load {phase}_processed_clusters.tab or return empty DF.
    """
    proc_path = phase2_basename('processed_clusters.tab')
    if os.path.isfile(proc_path):
        print(f"  - Detected non-20-col input; loading processed-clusters fallback: {proc_path}")
        return pd.read_csv(proc_path, sep="\t")
    print(f"[WARN] Processed-clusters fallback not found: {proc_path}")
    return pd.DataFrame()

def build_and_save_phas_clusters(allClusters: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-(chromosome, library) sRNA cluster features in parallel and write to TSV.
    Skips recomputation if the output file exists and matches the hash stored in memFile.
    Robust to accidentally receiving the 6-col merged-candidates frame: falls back to
    {phase}_processed_clusters.tab (20-col per-read/per-alignment schema).
    """
    print("### Step: Build PHAS clusters per (chromosome, library) — parallel ###")

    output_file = phase2_basename('PHAS_to_detect.tab')

    # --- Early hash check ---
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)
    section_name = "PHAS_TO_DETECT"
    if not config.has_section(section_name):
        config.add_section(section_name)

    if os.path.isfile(output_file):
        _, current_md5 = getmd5(output_file)
        previous_md5 = config[section_name].get(output_file)
        if previous_md5 and previous_md5 == current_md5:
            print(f"  - Output up-to-date (hash match). Skipping processing: {output_file}")
            df = pd.read_csv(output_file, sep="\t")
            numeric_allowlist = {
                "pos","len","hits","abun",
                "pval_h_f","N_f","X_f","pval_r_f","pval_corr_f",
                "pval_h_r","N_r","X_r","pval_r_r","pval_corr_r"
            }
            for col in numeric_allowlist.intersection(df.columns):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df

    # --- Accept only the 20-col per-read schema; else load the processed tab ---
    required_20 = {
        "alib","clusterID","chromosome","strand","pos","len","hits","abun",
        "pval_h_f","N_f","X_f","pval_r_f","pval_corr_f","pval_h_r","N_r",
        "X_r","pval_r_r","pval_corr_r","tag_id","tag_seq"
    }

    if not (isinstance(allClusters, pd.DataFrame) and required_20.issubset(set(allClusters.columns))):
        allClusters = load_processed_clusters_fallback(phase)

    # --- If still empty, bail cleanly ---
    if allClusters is None or getattr(allClusters, "empty", True):
        print("  - Found 0 (chromosome, library) groups (empty input). Returning empty DataFrame.")
        return pd.DataFrame(columns=list(required_20) + ["identifier"])

    # --- Ensure grouping columns exist / normalize ---
    if "chromosome" not in allClusters.columns and "chr" in allClusters.columns:
        allClusters = allClusters.rename(columns={"chr": "chromosome"})
    if "alib" not in allClusters.columns:
        try:
            if concat_libs:
                allClusters["alib"] = "ALL_LIBS"
            else:
                print("[WARN] 'alib' column missing and not in concat mode; returning empty DataFrame.")
                return pd.DataFrame(columns=list(required_20) + ["identifier"])
        except NameError:
            print("[WARN] 'alib' column missing; returning empty DataFrame.")
            return pd.DataFrame(columns=list(required_20) + ["identifier"])

    # --- Group → ((chromosome, library), loci_list) ---
    cluster_groups = [
        ((chromosome, alib), df.values.tolist())
        for (chromosome, alib), df in allClusters.groupby(['chromosome', 'alib'], sort=False)
    ]
    print(f"  - Found {len(cluster_groups)} (chromosome, library) groups")

    if not cluster_groups:
        print("  - No groups to process. Returning empty DataFrame.")
        return pd.DataFrame(columns=list(required_20) + ["identifier"])

    processed_results = run_parallel_with_progress(
        process_phas_cluster_group,
        cluster_groups,
        desc="Building PHAS cluster groups",
        min_chunk=1,
        unit="lib-chr"
    )

    if not processed_results:
        print("  - Worker returned no results. Returning empty DataFrame.")
        return pd.DataFrame(columns=list(required_20) + ["identifier"])

    processed_results = [r for r in processed_results if isinstance(r, pd.DataFrame) and not r.empty]
    if not processed_results:
        print("  - All worker results empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=list(required_20) + ["identifier"])

    clusters_data = pd.concat(processed_results, ignore_index=True)

    clusters_data.to_csv(output_file, sep="\t", encoding="utf-8", index=False)
    if os.path.isfile(output_file):
        _, new_md5 = getmd5(output_file)
        config[section_name][output_file] = new_md5
        with open(memFile, "w") as fh:
            config.write(fh)
        print(f"  - Wrote {output_file} (md5: {new_md5})")

    return clusters_data

def process_phas_cluster_group(group):
    """
    Process one group of loci: ((chromosome, alib), loci_group-as-list) -> DataFrame
    Adds 'chromosome' and 'alib' columns to the processed DataFrame.
    """
    (chromosome, alib), loci_group = group
    processed_df = process_chromosome_data(loci_group)  # existing worker
    processed_df['chromosome'] = chromosome
    processed_df['alib'] = alib
    return processed_df

def fishers(pvals):
    #print("#### Combine p-vals ######")
    '''
    combine pvals using fishers method
    '''

    apval = combine_pvalues(pvals, method='fisher', weights=None)
    #print("Fishers:",apval) ## return test statistics and pval

    return apval[1]

def _normalize_cluster_df(df: pd.DataFrame, is_concat: bool) -> pd.DataFrame:
    """
    Normalize column names/required cols for downstream grouping.
    - Ensures 'chromosome' column exists (renames 'chr' -> 'chromosome' if present).
    - Ensures 'alib' exists (sets to 'ALL_LIBS' in concat mode if missing).
    Returns the same DataFrame (mutated) for convenience.
    """
    if df is None:
        return pd.DataFrame()

    # Rename chr -> chromosome if needed
    if "chromosome" not in df.columns and "chr" in df.columns:
        df.rename(columns={"chr": "chromosome"}, inplace=True)

    # Ensure alib
    if "alib" not in df.columns:
        if is_concat:
            df["alib"] = "ALL_LIBS"
        else:
            # Don’t guess in multi-lib; keep empty but warn
            print("[WARN] DataFrame missing 'alib' in non-concat mode.")

    return df

def _safe_key(s: str) -> str:
    """Filesystem‑safe key for cache files (no regex)."""
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    return "".join(ch if ch in allowed else "_" for ch in str(s))


def _group_fingerprint(df: pd.DataFrame) -> str:
    """Stable, content‑based fingerprint of a group used to decide cache reuse.
    Only uses columns that affect window selection.
    """
    cols = [c for c in ("clusterID", "pos", "pval_corr_f", "pval_corr_r") if c in df.columns]
    if not cols:
        return ""
    ser = df[cols].sort_values([c for c in ("clusterID", "pos") if c in cols], kind="mergesort")
    payload = ser.to_csv(index=False).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _runtime_params_signature() -> str:
    """Hash of runtime knobs that affect window selection (no JSON dependency)."""
    phase_v = globals().get("phase")
    wl_v    = globals().get("window_len")
    sl_v    = globals().get("sliding")
    mcl_v   = globals().get("minClusterLength")
    sig_str = f"phase={phase_v};wl={wl_v};sl={sl_v};mcl={mcl_v};version=winselect.1"
    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()


# --- UPDATED: select_scoring_windows with per‑lib‑chr caching ---

def select_scoring_windows(clusters_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each chromosome (and library, if present), slide a fixed-length window across each
    cluster (>= minClusterLength) and record the best corrected p-values per window
    (forward/reverse) and their product.

    NEW: Resume‑safe chunked execution — each lib‑chr is written to {phase}_windows/<lib>__chr<id>.tsv
    and re-used on subsequent runs if the chunk file already exists (no md5/input checks for speed).

    Final merged output is written to {phase}_clusters_windows_to_score.tsv and cached in memFile.
    Safe on empty/partial input and cache hits.
    """
    print("### Step: select scoring windows per chromosome ###")

    outfname = phase2_basename('clusters_windows_to_score.tsv')
    # Encode key runtime params in the directory name to segregate caches across settings
    _sl = globals().get('sliding', 'NA')
    _wl = globals().get('window_len', 'NA')
    _mcl = globals().get('minClusterLength', 'NA')
    outdir = phase2_basename(f"windows_sl{_sl}_wl{_wl}_mcl{_mcl}")  # e.g., "24_windows_sl8_wl26_mcl200"
    os.makedirs(outdir, exist_ok=True)

    # --- memFile / config setup ---
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(memFile)
    sec_final = "WINDOWS_TO_SCORE"
    if not cfg.has_section(sec_final):
        cfg.add_section(sec_final)

    # Early return on final up-to-date file (hash match)
    if os.path.isfile(outfname):
        _, cur_md5 = getmd5(outfname)
        prev_md5   = cfg[sec_final].get(outfname)
        if prev_md5 and prev_md5 == cur_md5:
            print(f"  - Output up-to-date (hash match). Skipping computation: {outfname}")
            df = pd.read_csv(outfname, sep="\t")
            numeric_allowlist = {"window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"}
            for c in df.columns:
                if c in numeric_allowlist:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

    # --- Normalize/guard input ---
    required_in = ['clusterID', 'pos', 'pval_corr_f', 'pval_corr_r', 'chromosome']

    if clusters_data is None or getattr(clusters_data, "empty", True):
        print("[INFO] No clusters to select windows from; writing empty output.")
        empty_out = pd.DataFrame(columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value'])
        empty_out.to_csv(outfname, sep="\t", index=False)
        _, out_md5 = getmd5(outfname)
        cfg[sec_final][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        return empty_out

    if 'chromosome' not in clusters_data.columns and 'chr' in clusters_data.columns:
        clusters_data = clusters_data.rename(columns={'chr': 'chromosome'})

    missing = [c for c in required_in if c not in clusters_data.columns]
    if missing:
        print(f"[WARN] select_scoring_windows: missing columns {missing}; creating placeholders.")
        num_like = {'pos', 'pval_corr_f', 'pval_corr_r'}
        for c in missing:
            clusters_data[c] = pd.Series(dtype=("float64" if c in num_like else "object"))

    clusters_data = clusters_data[required_in + (["alib"] if "alib" in clusters_data.columns else [])].copy()
    if clusters_data.empty:
        print("[INFO] Required columns present but input empty after filtering; writing empty output.")
        empty_out = pd.DataFrame(columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value'])
        empty_out.to_csv(outfname, sep="\t", index=False)
        _, out_md5 = getmd5(outfname)
        cfg[sec_final][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        return empty_out

    # --- Build lib‑chr groups ---
    grouping = ['chromosome'] + (["alib"] if 'alib' in clusters_data.columns else [])
    groups = [(k, df) for k, df in clusters_data.groupby(grouping, sort=False)]
    print(f"  - Found {len(groups)} group(s) by {grouping}")
    # Resume policy: existence-only for chunk files (no md5/inputs to avoid overhead)

    # --- Plan tasks with cache checks ---
    tasks = []
    kept_paths = []  # paths we will merge (cached + newly written)
    for key_tuple, gdf in groups:
        # key normalization
        if isinstance(key_tuple, tuple):
            chrom = key_tuple[0]
            libid = key_tuple[1] if len(key_tuple) > 1 else 'concat'
        else:
            chrom = key_tuple
            libid = 'concat'
        key   = f"{libid}__chr{chrom}"
        outp  = os.path.join(outdir, f"{_safe_key(key)}.tsv")

        # Fast resume: if file exists and is non-empty, reuse it without hashing
        if os.path.isfile(outp) and os.path.getsize(outp) > 0:
            kept_paths.append(outp)
            continue

        tasks.append({
            'key': key,
            'df': gdf,
            'outpath': outp,
        })

    print(f"  - {len(kept_paths)} cached chunk(s) will be reused; {len(tasks)} chunk(s) to compute")

    results = []
    if tasks:
        # Parallel compute; each worker writes its own chunk then returns hashes
        results = run_parallel_with_progress(
            select_windows_task_worker,
            tasks,
            desc="Selecting windows (resume‑safe)",
            min_chunk=1,
            batch_factor=5,
            unit="lib-chr",
        ) or []

    # Update cache records for newly produced chunks
    for r in results:
        if not r:
            continue
        outp = r.get('outpath')
        if outp:
            kept_paths.append(outp)

    # Merge all chunk files (order by path for reproducibility)
    kept_paths = sorted(set(kept_paths))
    frames = []
    for p in kept_paths:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            try:
                frames.append(pd.read_csv(p, sep='\t'))
            except Exception as e:
                print(f"[WARN] Could not read chunk {p}: {e}")
    if frames:
        to_score = pd.concat(frames, ignore_index=True)
        # Optional: enforce dtypes
        for c in ("window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"):
            if c in to_score.columns:
                to_score[c] = pd.to_numeric(to_score[c], errors="coerce")
        # Optional, reproducible ordering
        sort_cols = [c for c in ("cluster_id", "window_n") if c in to_score.columns]
        if sort_cols:
            to_score = to_score.sort_values(sort_cols, kind="mergesort")
    else:
        to_score = pd.DataFrame(columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value'])

    # --- Write final + hash ---
    to_score.to_csv(outfname, sep="\t", index=False)
    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        cfg[sec_final][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        print(f"  - Wrote {outfname} (md5: {out_md5})\n    Cached chunks directory: {outdir}")

    return to_score


# --- NEW WORKER: write chunk to file and return hashes ---

def select_windows_task_worker(task: dict):
    """Worker wrapper that computes windows for a task and writes TSV to task['outpath'].
    Returns a dict with {'outpath'} for bookkeeping.
    """
    outpath = task['outpath']
    df = task['df']
    rows = select_windows_for_chromosome(df)  # existing logic

    # Ensure directory exists (defensive)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    if not rows:
        pd.DataFrame(columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value']).to_csv(outpath, sep='\t', index=False)
    else:
        pd.DataFrame(rows, columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value']).to_csv(outpath, sep='\t', index=False)

    # Hash the produced file
    return {'outpath': outpath, 'key': task.get('key')}



'''def select_scoring_windows(clusters_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each chromosome, slide a fixed-length window across each cluster (>= minClusterLength)
    and record the best corrected p-values per window (forward/reverse) and their product.
    Results are written to {phase}_clusters_windows_to_score.tsv and cached in memFile.
    Safe on empty/partial input and cache hits.
    """
    print("### Step: select scoring windows per chromosome ###")

    outfname = phase2_basename('clusters_windows_to_score.tsv')

    # --- Early hash check ---
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(memFile)
    section = "WINDOWS_TO_SCORE"
    if not cfg.has_section(section):
        cfg.add_section(section)

    if os.path.isfile(outfname):
        _, cur_md5 = getmd5(outfname)
        prev_md5 = cfg[section].get(outfname)
        if prev_md5 and prev_md5 == cur_md5:
            print(f"  - Output up-to-date (hash match). Skipping computation: {outfname}")
            df = pd.read_csv(outfname, sep="\t")
            # Only coerce known numeric columns from this TSV
            numeric_allowlist = {"window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"}
            for c in df.columns:
                if c in numeric_allowlist:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

    # --- Normalize/guard input ---
    # Expected input columns for this step
    required_in = ['clusterID', 'pos', 'pval_corr_f', 'pval_corr_r', 'chromosome']

    # Handle None/empty
    if clusters_data is None or getattr(clusters_data, "empty", True):
        print("[INFO] No clusters to select windows from; writing empty output.")
        empty_out = pd.DataFrame(columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value'])
        empty_out.to_csv(outfname, sep="\t", index=False)
        _, out_md5 = getmd5(outfname)
        cfg[section][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        return empty_out

    # Rename 'chr' -> 'chromosome' if needed
    if 'chromosome' not in clusters_data.columns and 'chr' in clusters_data.columns:
        clusters_data = clusters_data.rename(columns={'chr': 'chromosome'})

    # If any required columns are missing, add safe placeholders (keeps pipeline from crashing)
    missing = [c for c in required_in if c not in clusters_data.columns]
    if missing:
        print(f"[WARN] select_scoring_windows: missing columns {missing}; creating placeholders.")
        num_like = {'pos', 'pval_corr_f', 'pval_corr_r'}
        for c in missing:
            clusters_data[c] = pd.Series(dtype=("float64" if c in num_like else "object"))

    # Use only necessary columns (defensive copy) — avoids KeyError
    clusters_data = clusters_data[required_in].copy()

    # If after filtering nothing remains, emit empty file and return
    if clusters_data.empty:
        print("[INFO] Required columns present but input empty after filtering; writing empty output.")
        empty_out = pd.DataFrame(columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value'])
        empty_out.to_csv(outfname, sep="\t", index=False)
        _, out_md5 = getmd5(outfname)
        cfg[section][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        return empty_out

    # --- Split per chromosome (keep original order) ---
    chromosome_groups = [df for _, df in clusters_data.groupby('chromosome', sort=False)]
    print(f"  - Found {len(chromosome_groups)} chromosome groups")

    if not chromosome_groups:
        print("[INFO] No chromosome groups found; writing empty output.")
        empty_out = pd.DataFrame(columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value'])
        empty_out.to_csv(outfname, sep="\t", index=False)
        _, out_md5 = getmd5(outfname)
        cfg[section][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        return empty_out

    # --- Parallel worker ---
    results = run_parallel_with_progress(
        select_windows_for_chromosome,   # your existing worker
        chromosome_groups,
        desc="Selecting windows",
        min_chunk=1,
        batch_factor = 5,
        unit="lib-chr"
    )

    # Flatten and build output frame (robust to empty/None items)
    flat = []
    for r in results or []:
        if r:
            flat.extend(r)

    to_score = pd.DataFrame(
        flat,
        columns=['cluster_id', 'window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value']
    )

    # --- Write + hash (even if empty, to stabilize caching) ---
    to_score.to_csv(outfname, sep="\t", index=False)
    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        cfg[section][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        print(f"  - Wrote {outfname} (md5: {out_md5})")

    return to_score'''

def select_windows_for_chromosome(chromosome_df: pd.DataFrame):
    """
    Process window selection for a single chromosome DataFrame.
    Expected columns: clusterID, pos, pval_corr_f, pval_corr_r
    Returns list of [cluster_id, window_n, best_f, best_r, best_f*best_r]
    """
    # Keep only needed cols (defensive)
    df = chromosome_df[['clusterID', 'pos', 'pval_corr_f', 'pval_corr_r']].copy()

    # Ensure numeric types for computations (once per group)
    df['pos']         = pd.to_numeric(df['pos'], errors='coerce')
    df['pval_corr_f'] = pd.to_numeric(df['pval_corr_f'], errors='coerce')
    df['pval_corr_r'] = pd.to_numeric(df['pval_corr_r'], errors='coerce')

    # Drop rows with missing positions
    df = df.dropna(subset=['pos'])
    if df.empty:
        return []

    to_score = []
    append = to_score.append  # micro-opt

    for cID, aclust in df.groupby('clusterID', sort=False):
        if aclust.empty:
            continue

        # Check if positions are already sorted; sort only if needed
        pos = aclust['pos'].to_numpy()
        if pos.size == 0:
            continue
        # allow ties (<=). Using vectorized check avoids Python loops
        if not (pos[:-1] <= pos[1:]).all():
            aclust = aclust.sort_values('pos', kind='mergesort')
            pos = aclust['pos'].to_numpy()

        fw = aclust['pval_corr_f'].to_numpy()
        rv = aclust['pval_corr_r'].to_numpy()

        # Replace non-finite with +inf so they don't affect minima
        fw = np.where(np.isfinite(fw), fw, np.inf)
        rv = np.where(np.isfinite(rv), rv, np.inf)

        # Cluster span in genomic coordinates
        cluster_start = int(pos[0])
        cluster_end   = int(pos[-1])
        cluster_len   = cluster_end - cluster_start
        if cluster_len < minClusterLength or cluster_len < window_len:
            continue

        # Number of windows and starts/ends (half-open [start, end))
        nwin = 1 + (cluster_len - window_len) // sliding
        if nwin <= 0:
            continue
        w_starts = cluster_start + np.arange(0, nwin * sliding, sliding, dtype=np.int64)
        w_ends   = w_starts + window_len

        # Index bounds for each window in O(log N)
        left_idx  = np.searchsorted(pos, w_starts, side='left')
        right_idx = np.searchsorted(pos, w_ends,   side='left')  # half-open

        for w_i in range(nwin):
            li = left_idx[w_i]
            ri = right_idx[w_i]
            if li >= ri:
                continue

            best_f = fw[li:ri].min()
            best_r = rv[li:ri].min()

            if not np.isfinite(best_f) or not np.isfinite(best_r):
                continue

            append([cID, w_i, best_f, best_r, best_f * best_r])

    return to_score

'''def select_windows_for_chromosome(chromosome_df: pd.DataFrame):
    """
    Process window selection for a single chromosome DataFrame.
    Expected columns: clusterID, pos, pval_corr_f, pval_corr_r, (chromosome is not needed here)
    Returns list of [cluster_id, window_n, best_f, best_r, best_f*best_r]
    """
    # Keep only needed cols (defensive)
    chromosome_df = chromosome_df[['clusterID', 'pos', 'pval_corr_f', 'pval_corr_r']].copy()

    # Ensure numeric types for computations
    chromosome_df['pos'] = pd.to_numeric(chromosome_df['pos'], errors='coerce')
    chromosome_df['pval_corr_f'] = pd.to_numeric(chromosome_df['pval_corr_f'], errors='coerce')
    chromosome_df['pval_corr_r'] = pd.to_numeric(chromosome_df['pval_corr_r'], errors='coerce')

    # Drop rows with missing positions
    chromosome_df = chromosome_df.dropna(subset=['pos'])
    if chromosome_df.empty:
        return []

    to_score = []
    for cID, aclust in chromosome_df.groupby('clusterID', sort=False):
        if aclust.empty:
            continue

        # Cluster span
        cluster_start = int(aclust['pos'].min())
        cluster_end   = int(aclust['pos'].max())
        cluster_len   = cluster_end - cluster_start

        # Only consider sufficiently long clusters
        if cluster_len < minClusterLength:
            continue

        # Sliding windows
        window_n = 0
        # while window_start <= cluster_len - window_len ensures last window fits entirely
        window_start = 0
        while window_start <= (cluster_len - window_len):
            # Window coordinates in genomic space
            w_start = cluster_start + window_start
            w_end   = w_start + window_len  # half-open [w_start, w_end)

            # Boolean mask much faster than isin(range(...))
            sel = aclust['pos'].between(w_start, w_end - 1, inclusive='both')
            selected = aclust.loc[sel, ['pval_corr_f', 'pval_corr_r']]

            if not selected.empty:
                # min() is safe; if all NaN, result is NaN — we skip such windows
                best_f = selected['pval_corr_f'].min(skipna=True)
                best_r = selected['pval_corr_r'].min(skipna=True)

                if pd.notna(best_f) and pd.notna(best_r):
                    to_score.append([
                        cID,
                        window_n,
                        float(best_f),
                        float(best_r),
                        float(best_f) * float(best_r)
                    ])
            # advance
            window_start += sliding
            window_n += 1

    return to_score'''

def compute_scores_for_group(chromosome_data_group):
    """
    Worker: compute PHASIS scores for one (chromosome, library) group.

    Input: ((chromosome, library), rows-as-list) where each row has columns:
      ['cluster_id','window_n','fw_pval_corr','rv_pval_corr',
       'combined_window_p_value','chromosome','library']

    Output: list of [cID, phasis_score, combined_fishers]
    """
    (chromosome, lib), data_list = chromosome_data_group

    cols = ['cluster_id','window_n','fw_pval_corr','rv_pval_corr',
            'combined_window_p_value','chromosome','library']
    if not data_list:
        return []

    df = pd.DataFrame(data_list, columns=cols)

    # Re-assert group labels (defensive)
    df['chromosome'] = chromosome
    df['library'] = lib

    # Coerce numerics safely
    for c in ['window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    out = []
    for cID, aclust in df.groupby('cluster_id', sort=False):
        vals = aclust['combined_window_p_value'].dropna()
        if vals.empty:
            # ---- DEBUG LINE (requested) ----
            print(f"[DBG] no combined_window_p_value for {cID} in (chr={chromosome}, lib={lib}); rows={len(aclust)}")
            combined_fishers = 1.0
            phasis_score = 0.0
        else:
            combined_fishers = fishers(vals.tolist())
            if not np.isfinite(combined_fishers) or combined_fishers <= 0.0:
                combined_fishers = max(float(combined_fishers) if np.isfinite(combined_fishers) else 0.0, 1e-300)
            phasis_score = -np.log10(combined_fishers)
            if not np.isfinite(phasis_score):
                phasis_score = 300.0

        if phasis_score > 300.0:
            phasis_score = 300.0

        out.append([str(cID), float(phasis_score), float(combined_fishers)])

    return out

def compute_and_save_phasis_scores(clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PHASIS scores per (chromosome, library) group in parallel.
    Writes {phase}_clusters_scored.tsv and caches its hash in memFile.
    """
    print("### Step: Compute PHASIS scores per (chromosome, library) ###")
    outfname = phase2_basename('clusters_scored.tsv')

    # --- Early hash check ---
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(memFile)
    section = "CLUSTERS_SCORED"
    if not cfg.has_section(section):
        cfg.add_section(section)

    if os.path.isfile(outfname):
        _, cur_md5 = getmd5(outfname)
        prev_md5 = cfg[section].get(outfname)
        if prev_md5 and prev_md5 == cur_md5:
            print(f"  - Output up-to-date (hash match). Skipping computation: {outfname}")
            df_cached = pd.read_csv(outfname, sep="\t")
            for c in ("phasis_score", "combined_fishers"):
                if c in df_cached.columns:
                    df_cached[c] = pd.to_numeric(df_cached[c], errors="coerce").fillna(0.0)
            return df_cached

    # --- Guard input ---
    if clusters is None or getattr(clusters, "empty", True):
        print("[INFO] compute_and_save_phasis_scores: empty input; writing empty file.")
        pd.DataFrame(columns=["cID", "phasis_score", "combined_fishers"]).to_csv(outfname, sep="\t", index=False)
        if os.path.isfile(outfname):
            _, out_md5 = getmd5(outfname); cfg[section][outfname] = out_md5
            with open(memFile, 'w') as fh: cfg.write(fh)
        return pd.DataFrame(columns=["cID", "phasis_score", "combined_fishers"])

    # --- Derive chromosome & library from compact IDs (robust) ---
    clusters = clusters.copy()
    if 'cluster_id' not in clusters.columns:
        raise KeyError("compute_and_save_phasis_scores: input must contain 'cluster_id' column")

    clusters['cluster_id'] = clusters['cluster_id'].astype(str)

    # Chromosome = last underscore chunk
    clusters['chromosome'] = clusters['cluster_id'].str.rsplit('_', n=1).str[-1]

    # Library = everything before the last '-' (compact IDs), with fallbacks for old IDs
    swap_phase = (21 if phase == 24 else 24 if phase == 21 else phase)
    tag_main = f"{phase}-PHAS"
    tag_alt  = f"{swap_phase}-PHAS"

    def _infer_lib(cid: str) -> str:
        s = str(cid)
        if '-' in s:
            return s.rsplit('-', 1)[0]
        if tag_main in s:
            return s.split(tag_main)[0]
        if tag_alt in s:
            return s.split(tag_alt)[0]
        return "UNKNOWN"

    clusters['library'] = clusters['cluster_id'].map(_infer_lib)

    # Debug visibility on inference
    total_rows = len(clusters)
    unknown_libs = int((clusters['library'] == "UNKNOWN").sum())
    if unknown_libs:
        ex = clusters.loc[clusters['library'] == "UNKNOWN", 'cluster_id'].head(5).tolist()
        print(f"[WARN] {unknown_libs}/{total_rows} cluster_ids have UNKNOWN library. Examples: {ex}")

    # Ensure numeric columns (they might be strings if read from TSV)
    for c in ('window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value'):
        if c in clusters.columns:
            clusters[c] = pd.to_numeric(clusters[c], errors='coerce')

    # Group and ship to workers (dropna=False to avoid dropping UNKNOWN)
    groups = [
        ((chrom, lib), df[['cluster_id','window_n','fw_pval_corr','rv_pval_corr',
                           'combined_window_p_value','chromosome','library']].values.tolist())
        for (chrom, lib), df in clusters.groupby(['chromosome', 'library'], sort=False, dropna=False)
    ]
    print(f"  - Found {len(groups)} (chromosome, library) groups")

    if not groups:
        print(f"[ERR] No groups formed. Unique cluster_ids={clusters['cluster_id'].nunique()}, "
              f"chromosomes={clusters['chromosome'].nunique()}, libraries={clusters['library'].nunique()}")

    # --- Parallel compute ---
    results = run_parallel_with_progress(
        compute_scores_for_group,
        groups,
        desc="Scoring windows via Fisher's method",
        min_chunk=1,
        unit="lib-chr"
    )

    # Flatten and write
    flat = [item for sub in (results or []) for item in (sub or [])]
    win_phasis_score = pd.DataFrame(flat, columns=["cID", "phasis_score", "combined_fishers"])

    win_phasis_score.to_csv(outfname, sep="\t", index=False)
    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        cfg[section][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        print(f"  - Wrote {outfname} (md5: {out_md5})")

    return win_phasis_score


# Global variables
WINDOW_MULTIPLIER = 10  # 10 cycles
# Global cache used by workers (read-only)
WIN_SCORE_LOOKUP = {}

def set_win_score_lookup(win_df: pd.DataFrame) -> dict:
    """
    Build and set a compact lookup: cID -> (phasis_score, combined_fishers).
    Stores it in the module-global WIN_SCORE_LOOKUP and returns it.
    """
    mapping = {}
    if win_df is None or getattr(win_df, "empty", True):
        globals()["WIN_SCORE_LOOKUP"] = mapping
        return mapping

    df = win_df.copy()
    # Normalize expected columns
    if "cID" not in df.columns:
        # try common fallbacks; no-op if not found
        for alt in ("clusterID", "cid", "cluster_id"):
            if alt in df.columns:
                df = df.rename(columns={alt: "cID"})
                break

    # Coerce to numeric and drop missing IDs
    df["phasis_score"] = pd.to_numeric(df.get("phasis_score"), errors="coerce")
    df["combined_fishers"] = pd.to_numeric(df.get("combined_fishers"), errors="coerce")

    # Build mapping
    for cid, ps, cf in zip(df.get("cID", []), df.get("phasis_score", []), df.get("combined_fishers", [])):
        if pd.isna(cid):
            continue
        mapping[cid] = (None if pd.isna(ps) else float(ps),
                        None if pd.isna(cf) else float(cf))

    globals()["WIN_SCORE_LOOKUP"] = mapping
    return mapping


def features_to_detection(clusters_data: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble per-cluster feature set (parallel), write TSV, and memoize via md5.
    Uses legacy column names compatible with downstream KNN (no aliases).
    Expects process_chromosome_features() to return rows in FEATURE_COLS order.
    """
    print("### Step: assemble per-cluster features ###")

    outfname = phase2_basename('cluster_set_features.tsv')

    # ---- Legacy schema (KNN-compatible) ----
    FEATURE_COLS = [
        "identifier", "cID", "alib",
        "complexity", "strand_bias", "log_clust_len_norm_counts",
        "ratio_abund_len_phase", "phasis_score", "combined_fishers",
        "total_abund",
        # wobble-tolerant Howell (legacy names)
        "w_Howell_score", "w_window_start", "w_window_end",
        "c_Howell_score", "c_window_start", "c_window_end",
        "Peak_Howell_score",
        # classic Howell (strict; legacy-style names with _strict)
        "w_Howell_score_strict", "w_window_start_strict", "w_window_end_strict",
        "c_Howell_score_strict", "c_window_start_strict", "c_window_end_strict",
        "Peak_Howell_score_strict",
    ]

    # Numeric columns (all except id-like fields)
    NUMERIC_COLS = set(FEATURE_COLS) - {"identifier", "cID", "alib"}

    # ---------- Early hash check ----------
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(memFile)
    section = "CLUSTER_FEATURES"
    if not cfg.has_section(section):
        cfg.add_section(section)

    if os.path.isfile(outfname):
        _, cur_md5 = getmd5(outfname)
        prev_md5 = cfg[section].get(outfname)
        if prev_md5 and prev_md5 == cur_md5:
            print(f"  - Output up-to-date (hash match). Skipping assembly: {outfname}")
            df = pd.read_csv(outfname, sep="\t")

            # Coerce numerics by legacy names only
            for col in df.columns:
                if col in NUMERIC_COLS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Ensure exact column order if all present
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if not missing:
                df = df[FEATURE_COLS]
            else:
                print(f"[WARN] Existing file lacks expected columns: {missing}")

            return df

    # ---------- Build inputs ----------
    required_cols = ['clusterID', 'chromosome', 'strand', 'pos', 'len', 'abun', 'identifier', 'tag_seq', 'alib']
    missing_in_input = [c for c in required_cols if c not in clusters_data.columns]
    if missing_in_input:
        raise ValueError(f"clusters_data missing required columns: {missing_in_input}")

    clusters_data = clusters_data[required_cols].copy()

    # Split per chromosome for parallel processing
    chromosome_groups = [df for _, df in clusters_data.groupby('chromosome', sort=False)]
    print(f"  - Found {len(chromosome_groups)} chromosome groups")

    # ---------- Parallel processing ----------
    results = run_parallel_with_progress(
        process_chromosome_features,
        chromosome_groups,
        desc="Assemble features",
        min_chunk=1,
        unit="lib-chr"
    )

    # ---------- Flatten safely ----------
    flat = []
    bad_chunks = 0
    for sub in results:
        if isinstance(sub, list):
            for row in sub:
                if isinstance(row, (list, tuple)) and len(row) == len(FEATURE_COLS):
                    flat.append(list(row))
                else:
                    bad_chunks += 1
        else:
            bad_chunks += 1

    if bad_chunks:
        print(f"[WARN] Skipped {bad_chunks} malformed/failed rows or chunks during feature assembly.")

    if not flat:
        raise RuntimeError("No features assembled; all chunks failed or returned empty results.")

    # ---------- DataFrame materialization ----------
    collected_features = pd.DataFrame(flat, columns=FEATURE_COLS)

    # Numeric coercion on the newly built DF (legacy names)
    for col in collected_features.columns:
        if col in NUMERIC_COLS:
            collected_features[col] = pd.to_numeric(collected_features[col], errors="coerce")

    # ---------- Write + hash ----------
    collected_features.to_csv(outfname, sep="\t", index=False)
    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        cfg[section][outfname] = out_md5
        with open(memFile, 'w') as fh:
            cfg.write(fh)
        print(f"  - Wrote {outfname} (md5: {out_md5})")

    return collected_features

# --- helpers ---------------------------------------------------------------

DCL_OVERHANG = 3          # 2-nt 3' overhang in duplex -> 3-nt genomic offset
WINDOW_MULTIPLIER = 10    # 10 cycles per window
# ---------- Howell utilities (exact-phase only) ----------

def _strand_masks(df: pd.DataFrame):
    """Return boolean masks for W and C strands accepting several encodings."""
    s = df['strand'].astype(str).str.lower()
    w_mask = s.isin(['w', '+', 'watson', '1', 'true'])
    c_mask = s.isin(['c', '-', 'crick', '0', 'false'])
    return w_mask, c_mask


def _build_pos_abun_exact_phase(df: pd.DataFrame, seq_start: int, seq_end: int, phase: int):
    """
    Build {position -> abundance} using ONLY reads with length == phase
    and positions within [seq_start, seq_end].
    """
    ph = int(phase)
    d: dict[int, float] = {}
    # small speed-up: filter by length first
    df_ph = df.loc[pd.to_numeric(df['len'], errors='coerce') == ph]
    if df_ph.empty:
        return d
    for _, row in df_ph.iterrows():
        pos = int(row['pos'])
        if seq_start <= pos <= seq_end:
            d[pos] = d.get(pos, 0.0) + float(row['abun'])
    return d


# ---------- RELAXED Howell (positional ±1 wobble allowed) ----------
def _best_sliding_window_score_generic(pos_abun, phase, win_size, seq_start=None, seq_end=None, forward=True):
    """
    Generic window scan (relaxed Howell with ±1 wobble).
    Expects pos_abun already filtered to len == phase.
    """
    positions = sorted(pos_abun.keys())
    if not positions:
        return 0.0, None, None

    lower_bound = seq_start if seq_start is not None else positions[0]
    upper_bound = (seq_end - win_size + 1) if seq_end is not None else positions[-1] - win_size + 1
    if upper_bound < lower_bound:
        lower_bound = positions[0]
        upper_bound = lower_bound

    best_score = -float("inf")
    best_window = (None, None)

    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        if not window_positions:
            score = 0.0
        else:
            # Require at least 4 cycles possible in the window (n>3 guard)
            num_cycles = max(0, (win_end - win_start + 1) // int(phase))
            if num_cycles < 4:
                score = 0.0
            else:
                best_reg_sum = 0.0
                best_reg_total = 0.0
                best_reg_filled = 0

                for reg in range(int(phase)):
                    in_sum, eff_total, n_filled = _evaluate_register(
                        window_positions, pos_abun, win_start, win_end, int(phase), reg, forward=forward
                    )
                    if in_sum > best_reg_sum:
                        best_reg_sum = in_sum
                        best_reg_total = eff_total
                        best_reg_filled = n_filled

                out_of_phase = max(0.0, best_reg_total - best_reg_sum)  # U
                numerator = best_reg_sum
                denominator = 1.0 + out_of_phase
                if numerator <= 0.0 or not (best_reg_filled > 3):
                    score = 0.0
                else:
                    log_arg = 1.0 + 10.0 * (numerator / denominator)
                    if log_arg <= 0.0 or log_arg != log_arg:  # NaN guard
                        score = 0.0
                    else:
                        scale = max(min(best_reg_filled, num_cycles) - 2, 0)
                        score = scale * (0.0 if log_arg <= 0 else np.log(log_arg))

        if score > best_score:
            best_score = score
            best_window = (win_start, win_end)

    return best_score if best_score != -float("inf") else 0.0, best_window[0], best_window[1]



def best_sliding_window_score_forward(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=True
    )


def best_sliding_window_score_reverse(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=False
    )


def compute_phasing_score_Howell(aclust: pd.DataFrame):
    """
    Howell-like phasing WITH positional wobble (±1), but ONLY len == phase reads.
    Returns: (w_score,(w_start,w_end), c_score,(c_start,c_end))
    """
    win_size  = WINDOW_MULTIPLIER * int(phase)
    seq_start = int(aclust['pos'].min()); seq_end = int(aclust['pos'].max())
    w_mask, c_mask = _strand_masks(aclust)

    # Forward “w”
    if w_mask.any():
        w_pos_abun = _build_pos_abun_exact_phase(aclust.loc[w_mask], seq_start, seq_end, int(phase))
        w_score, w_s, w_e = (
            best_sliding_window_score_forward(w_pos_abun, int(phase), win_size, seq_start, seq_end)
            if w_pos_abun else (0.0, None, None)
        )
    else:
        w_score, w_s, w_e = None, None, None

    # Reverse “c”
    if c_mask.any():
        c_pos_abun = _build_pos_abun_exact_phase(aclust.loc[c_mask], seq_start, seq_end, int(phase))
        c_score, c_s, c_e = (
            best_sliding_window_score_reverse(c_pos_abun, int(phase), win_size, seq_start, seq_end)
            if c_pos_abun else (0.0, None, None)
        )
    else:
        c_score, c_s, c_e = None, None, None

    return (w_score, (w_s, w_e), c_score, (c_s, c_e))


# ---------- STRICT Howell (NO positional wobble; still ONLY len == phase) ----------
def _evaluate_register_strict_exact(window_positions, pos_abun, win_start, win_end, phase, reg, forward=True):
    """Count ONLY exact register hits (no ±1). Returns: (in_phase_sum, total_in_window, n_filled_cycles)"""
    positions_set = set(window_positions)
    num_cycles = max(0, (win_end - win_start + 1) // int(phase))

    in_phase_sum = 0.0
    n_filled = 0
    for c in range(num_cycles):
        expected_pos = (win_start + reg + c * int(phase)) if forward else (win_end - reg - c * int(phase))
        if expected_pos in positions_set:
            in_phase_sum += pos_abun[expected_pos]
            n_filled += 1

    total_in_window = sum(pos_abun[p] for p in window_positions)
    return in_phase_sum, total_in_window, n_filled

def _evaluate_register(window_positions, pos_abun, win_start, win_end, phase, reg, forward=True):
    """
    Wobble-tolerant register evaluation (±1 positional wobble).
    Returns: (in_phase_sum, effective_total, n_filled_cycles)
    Semantics:
      - If the exact expected position exists, count it for in-phase and quarantine its ±1 neighbors.
      - Else, pick the better of the ±1 neighbors (if any) and quarantine the sibling neighbor.
      - Each genomic position is counted at most once in-phase.
      - Effective total excludes quarantined neighbors so they don't inflate U.
    """
    ph = int(phase)
    positions_set = set(window_positions)
    num_cycles = max(0, (win_end - win_start + 1) // ph)

    used_positions = set()     # positions used as in-phase
    ignored_positions = set()  # neighbors to exclude from effective_total
    in_phase_sum = 0.0
    n_filled = 0

    for c in range(num_cycles):
        expected_pos = (win_start + reg + c * ph) if forward else (win_end - reg - c * ph)

        # Case 1: exact exists -> use and quarantine neighbors
        if expected_pos in positions_set and expected_pos not in used_positions:
            in_phase_sum += pos_abun[expected_pos]
            used_positions.add(expected_pos)
            n_filled += 1
            for off in (-1, 1):
                npos = expected_pos + off
                if npos in positions_set and npos not in used_positions:
                    ignored_positions.add(npos)
        else:
            # Case 2: consider ±1; choose best if present; quarantine sibling neighbor
            left = expected_pos - 1
            right = expected_pos + 1
            candidates = []
            if left in positions_set and left not in used_positions:
                candidates.append(left)
            if right in positions_set and right not in used_positions:
                candidates.append(right)
            if candidates:
                best = max(candidates, key=lambda p: pos_abun[p])
                in_phase_sum += pos_abun[best]
                used_positions.add(best)
                n_filled += 1
                sibling = right if best == left else left
                if sibling in positions_set and sibling not in used_positions:
                    ignored_positions.add(sibling)

    # Effective total excludes quarantined neighbors of exact/selected hits
    effective_positions = [p for p in window_positions if p not in ignored_positions]
    effective_total = sum(pos_abun[p] for p in effective_positions)
    return in_phase_sum, effective_total, n_filled
def _best_sliding_window_score_generic_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None, forward=True):
    positions = sorted(pos_abun.keys())
    if not positions:
        return 0.0, None, None

    lower_bound = seq_start if seq_start is not None else positions[0]
    upper_bound = (seq_end - win_size + 1) if seq_end is not None else positions[-1] - win_size + 1
    if upper_bound < lower_bound:
        lower_bound = positions[0]
        upper_bound = lower_bound

    best_score = -float("inf")
    best_window = (None, None)

    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        if not window_positions:
            score = 0.0
        else:
            num_cycles = max(0, (win_end - win_start + 1) // int(phase))
            if num_cycles < 4:
                score = 0.0
            else:
                best_reg_sum = 0.0
                best_reg_total = 0.0
                best_reg_filled = 0

                for reg in range(int(phase)):
                    in_sum, total, n_filled = _evaluate_register_strict_exact(
                        window_positions, pos_abun, win_start, win_end, int(phase), reg, forward=forward
                    )
                    if in_sum > best_reg_sum:
                        best_reg_sum = in_sum
                        best_reg_total = total
                        best_reg_filled = n_filled

                out_of_phase = max(0.0, best_reg_total - best_reg_sum)
                numerator = best_reg_sum
                denominator = 1.0 + out_of_phase
                if numerator <= 0.0 or not (best_reg_filled > 3):
                    score = 0.0
                else:
                    log_arg = 1.0 + 10.0 * (numerator / denominator)
                    if log_arg <= 0.0 or log_arg != log_arg:
                        score = 0.0
                    else:
                        scale = max(min(best_reg_filled, num_cycles) - 2, 0)
                        score = scale * (0.0 if log_arg <= 0 else np.log(log_arg))

        if score > best_score:
            best_score = score
            best_window = (win_start, win_end)

    return best_score if best_score != -float("inf") else 0.0, best_window[0], best_window[1]



def best_sliding_window_score_forward_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=True
    )


def best_sliding_window_score_reverse_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=False
    )


def compute_phasing_score_Howell_strict(aclust: pd.DataFrame):
    """
    Classic Howell phasing WITHOUT positional wobble.
    Uses ONLY len == phase reads.
    Returns: (w_score,(w_start,w_end), c_score,(c_start,c_end))
    """
    win_size  = WINDOW_MULTIPLIER * int(phase)
    seq_start = int(aclust['pos'].min()); seq_end = int(aclust['pos'].max())
    w_mask, c_mask = _strand_masks(aclust)

    # Forward “w”
    if w_mask.any():
        w_pos_abun = _build_pos_abun_exact_phase(aclust.loc[w_mask], seq_start, seq_end, int(phase))
        w_score, w_s, w_e = (
            best_sliding_window_score_forward_strict(w_pos_abun, int(phase), win_size, seq_start, seq_end)
            if w_pos_abun else (0.0, None, None)
        )
    else:
        w_score, w_s, w_e = None, None, None

    # Reverse “c”
    if c_mask.any():
        c_pos_abun = _build_pos_abun_exact_phase(aclust.loc[c_mask], seq_start, seq_end, int(phase))
        c_score, c_s, c_e = (
            best_sliding_window_score_reverse_strict(c_pos_abun, int(phase), win_size, seq_start, seq_end)
            if c_pos_abun else (0.0, None, None)
        )
    else:
        c_score, c_s, c_e = None, None, None

    return (w_score, (w_s, w_e), c_score, (c_s, c_e))


# =========================
# Feature assembly
# =========================
def process_chromosome_features(chromosome_df: pd.DataFrame):
    """
    Build per-cluster feature rows (wobble + strict Howell).
    Returns rows matching FEATURE_COLS order.
    Expected columns:
      ['clusterID','chromosome','strand','pos','len','abun','identifier','tag_seq','alib']
    """
    # --- local normalizer (uses robust prefix-stripper, preserves coord IDs) ---
    def _norm_id(x) -> str:
        s = str(x).strip()
        if ":" in s and ".." in s:   # already universal
            return s
        # prefer the robust variant of _strip_fileprefix_from_id
        return _strip_fileprefix_from_id(s, phase=phase)

    df = chromosome_df[['clusterID','chromosome','strand','pos','len','abun','identifier','tag_seq','alib']].copy()
    df['pos']  = pd.to_numeric(df['pos'], errors='coerce')
    df['len']  = pd.to_numeric(df['len'], errors='coerce')
    df['abun'] = pd.to_numeric(df['abun'], errors='coerce').fillna(0)
    df = df.dropna(subset=['pos', 'len'])

    # --- build a normalized view of the global score lookup (point #4) ---
    # Accept keys that might already be normalized or universal
    raw_lookup = (globals().get("WIN_SCORE_LOOKUP", {}) or {})
    score_lookup = {}
    for k, v in raw_lookup.items():
        try:
            score_lookup[_norm_id(k)] = v
        except Exception:
            # be forgiving; keep raw key too
            score_lookup[str(k).strip()] = v

    rows = []
    for cID, aclust in df.groupby('clusterID', sort=False):
        if aclust.empty:
            continue

        # --- universal identifier (point #5) ---
        cid_norm = _norm_id(cID)

        # derive genomic span from this cluster's rows (works even if lookup misses)
        achr  = str(aclust['chromosome'].iloc[0])
        start = int(aclust['pos'].min()); end = int(aclust['pos'].max())

        uid = None
        #getUniversalID already handles coord IDs; we pass normalized key
        uid = getUniversalID(cid_norm)

        if not uid or ":" not in uid or ".." not in uid:
            uid = f"{achr}:{start}..{end}"   # hard fallback: always coordinate-style

        identifier = uid
        alib = str(aclust['alib'].iloc[0])

        # Normalize strand labels
        s_norm = aclust['strand'].astype(str).str.lower()
        w_mask = s_norm.isin(['w', '+', 'watson', '1', 'true'])
        c_mask = s_norm.isin(['c', '-', 'crick', '0', 'false'])

        # Strand bias
        total_w = int(w_mask.sum())
        total_c = int(c_mask.sum())
        denom = total_w + total_c
        strand_bias = (total_w / denom) if denom > 0 else 1.0

        # Abundance ratios
        ph = int(phase)
        sum_abun_len_phase = aclust.loc[aclust['len'] == ph, 'abun'].sum()
        sum_abun_other_len = aclust.loc[aclust['len'] != ph, 'abun'].sum()
        ratio_abund_len_phase = (sum_abun_len_phase / sum_abun_other_len) if sum_abun_other_len > 0 else 1.0

        # Totals and cluster length
        total_abund = float(aclust['abun'].sum())
        aclust_len = max(end - start, 0)

        # CLNC for phase-length reads (log10(1 + CLNC))
        w_sum_abun_len_phase = aclust.loc[(aclust['len'] == ph) & w_mask, 'abun'].sum()
        c_sum_abun_len_phase = aclust.loc[(aclust['len'] == ph) & c_mask, 'abun'].sum()
        denom_len = max(aclust_len - ph, 0)
        CLNC = ((w_sum_abun_len_phase + c_sum_abun_len_phase) / denom_len) if denom_len > 0 else 0.0
        log_CLNC = float(np.log10(CLNC + 1.0))

        # Complexity
        distinct_alignments = aclust['tag_seq'].nunique(dropna=True)
        complexity = (distinct_alignments / total_abund) if total_abund > 0 else 0.0

        # --- look up PHASIS score using normalized key; be tolerant ---
        aclust_phasis_score = 0.0
        aclust_fishers_combined = 1.0
        tup = score_lookup.get(cid_norm)
        if tup is None:
            # Sometimes scores might be keyed by the universal ID already; try that too.
            tup = score_lookup.get(uid)
        if tup is not None:
            ps, cf = tup
            if ps is not None:
                try:   aclust_phasis_score = float(ps)
                except: pass
            if cf is not None:
                try:   aclust_fishers_combined = float(cf)
                except: pass

        # Howell (wobble-tolerant)
        (w_Howell, (w_s, w_e),
         c_Howell, (c_s, c_e)) = compute_phasing_score_Howell(aclust)

        # Howell strict
        (w_Howell_strict, (w_s_strict, w_e_strict),
         c_Howell_strict, (c_s_strict, c_e_strict)) = compute_phasing_score_Howell_strict(aclust)

        Peak_Howell = None if (w_Howell is None and c_Howell is None) \
                      else max([x for x in (w_Howell, c_Howell) if x is not None])

        Peak_Howell_strict = None if (w_Howell_strict is None and c_Howell_strict is None) \
                             else max([x for x in (w_Howell_strict, c_Howell_strict) if x is not None])

        rows.append([
            identifier, cID, alib,
            float(complexity), float(strand_bias), float(log_CLNC),
            float(ratio_abund_len_phase), aclust_phasis_score, aclust_fishers_combined,
            float(total_abund),
            # wobble-tolerant
            w_Howell, w_s, w_e, c_Howell, c_s, c_e, Peak_Howell,
            # classic (strict)
            w_Howell_strict, w_s_strict, w_e_strict, c_Howell_strict, c_s_strict, c_e_strict, Peak_Howell_strict
        ])

    return rows


def format_attributes(row):
    attributes = {
        'id':row['identifier'],
        'complexity': row['complexity'],
        'strand_bias': row['strand_bias'],
        'log_clust_len_norm_counts': row['log_clust_len_norm_counts'],
        'ratio_abund_len_phase': row['ratio_abund_len_phase']
    }
    attr_str = ';'.join([f"{key}={value}" for key, value in attributes.items()])
    return attr_str

def write_gff(phasis_result_df,gff_filename):
    unique_df = phasis_result_df.groupby('identifier').first().reset_index()
    # Write to GFF3 file
    
    with open(gff_filename, 'w') as gff_file:
        for index, row in unique_df.iterrows():
            seq_id = row['achr']
            source = 'Phasis'
            feature_type = f'{phase}-PHAS'
            start_pos = str(row['start'])
            end_pos = str(row['end'])
            score = str(row['phasis_score'])
            strand = '.'
            t_phase = '.'
            attributes = format_attributes(row)
            gff_line = f"{seq_id}\t{source}\t{feature_type}\t{start_pos}\t{end_pos}\t{score}\t{strand}\t{t_phase}\t{attributes}\n"
            gff_file.write(gff_line)

    return None

def plot_report_heat_map(phasis_result_df, plot_type):
    print("#### Plotting heatmap ######")

    # Create a DataFrame to store heatmap data
    #data = pd.DataFrame(data=0, columns=list(phasis_result_df["alib"].unique()), index=list(phasis_result_df["identifier"].unique()))

    data = pd.DataFrame(
    data=0.0, 
    columns=sorted(list(phasis_result_df["alib"].unique())),  # Sort columns alphanumerically
    index=list(phasis_result_df["identifier"].unique())
    )

    # Iterate over rows and columns to fill heatmap data
    for i in tqdm(data.index, desc="Processing Rows"):
        tempRows = phasis_result_df[phasis_result_df["identifier"] == i]
        for j in data.columns:
            k = 0.0
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty:
                if "non-PHAS" in subSetData["label"].values:
                    k = 1.0
                elif "PHAS" in subSetData["label"].values:
                    k = 2.0
            data.loc[i, j] = k

    # Extract chromosome data from the identifier
    chrom_data = [identifier.split(':')[0] for identifier in data.index]

    # Sort chromosomes numerically instead of lexicographically (i.e., 1, 2, 3, ... not 1, 10, 2, 3, ...)
    sorted_indices = sorted(range(len(chrom_data)), key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx])

    # Reorder the DataFrame based on sorted chromosomes
    data = data.iloc[sorted_indices]

    # Re-extract the sorted chromosome data
    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    # Create the heatmap figure with specified figure size
    f, ax = plt.subplots(figsize=(11, 11))

    # Define color map for the heatmap
    colors = ["#C3D8EA", "#3662A5", "#C24F4E"]
    cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

    # Create the actual heatmap for the PHAS data
    heat = sns.heatmap(data, square=False, cmap=cmap, cbar=False, xticklabels=True, yticklabels=False, ax=ax)

    # Add a new axis for the chromosome bar to the left
    cax = f.add_axes([0.17, 0.1, 0.02, 0.8])  # Adjust the position for the left side

    # Manually create a discrete grayscale colormap for the chromosomes
    base_cmap = plt.get_cmap('Greys')
    chrom_cmap = base_cmap(np.linspace(0.3, 0.9, len(unique_chromosomes)))  # Discretizing to avoid full black/white

    # Assign colors based on the order of unique chromosomes
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}

    # Map chromosomes to grayscale colors
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    # Plot the chromosome colorbar (no x-axis)
    cax.imshow(chrom_colors, cmap=LinearSegmentedColormap.from_list('CustomGreys', chrom_cmap, len(unique_chromosomes)), aspect="auto")
    cax.set_xticks([])  # Remove x-axis ticks
    cax.set_yticks([])  # Remove y-axis ticks

    # Remove the black border around the colorbar
    cax.spines['top'].set_visible(False)
    cax.spines['right'].set_visible(False)
    cax.spines['bottom'].set_visible(False)
    cax.spines['left'].set_visible(False)

    # Align chromosome bar with heatmap y-coordinates
    cax.set_ylim(len(data.index), 0)  # Set y-limits to invert the chromosome bar

    # Calculate the midpoints for each chromosome based on its range of rows
    chrom_ranges = {}  # Dictionary to store row ranges for each chromosome
    for chrom in unique_chromosomes:
        chrom_loci_indices = np.where(np.array(chrom_data) == chrom)[0]  # Get the row indices for this chromosome
        chrom_midpoint = (chrom_loci_indices[0] + chrom_loci_indices[-1]) / 2  # Calculate the midpoint of this range
        chrom_ranges[chrom] = chrom_midpoint

    # Set chromosome names in the middle of the chromosome color bar using calculated midpoints
    midpoints = [chrom_ranges[chrom] for chrom in unique_chromosomes]
    cax.set_yticks(midpoints)  # Set y-ticks at midpoints
    cax.set_yticklabels(unique_chromosomes, fontsize=16, rotation=0, ha='center')  # Add chromosome names

    # Adjust the y-axis tick parameters to move labels slightly to the left
    cax.yaxis.set_tick_params(labelsize=16, pad=11)  # Adjust pad to move the labels to the left

    # Add a custom color bar (legend) to the top right
    cax2 = inset_axes(ax,
                     width="2.5%",  # Set width for small size
                     height="9%",  # Set height for vertical orientation
                     loc='lower left',  # Position at the top left corner
                     bbox_to_anchor=(-0.25, 1.01, 1, 1),  # Adjust bbox_to_anchor for position
                     bbox_transform=ax.transAxes,
                     borderpad=0
                     )   
    cax2.spines['top'].set_visible(False)
    cax2.spines['right'].set_visible(False)
    cax2.spines['bottom'].set_visible(False)
    cax2.spines['left'].set_visible(False)

    # Create a custom color bar with specific text legends for each color group
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax2, orientation='vertical')
    cbar.set_ticklabels(["Not detected", r'non-$\it{PHAS}$ cluster', r'$\it{PHAS}$'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # Adjust space for better visualization - Increased left margin for y-tick labels
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)  # Adjust the margins

    # Save the plot
    fig = heat.get_figure()
    def _p(d, name):  # path join w/o imports
        return name if not d else (d + '/' + name if not d.endswith('/') else d + name)

    fig.savefig(_p(outdir, f'{phase}_{plot_type}_PHAS.pdf'), dpi=300)

    plt.close(fig)

    return

def plot_phasAbundance_heat_map(phasis_result_df, type):
    print("#### Plotting PHAS Abundance heatmap ######")

    # Create a DataFrame with identifiers as index and alib as columns, initialized with zeros
    #data = pd.DataFrame(data=0.0, columns=list(phasis_result_df["alib"].unique()), index=list(phasis_result_df["identifier"].unique()))

    data = pd.DataFrame(
    data=0.0, 
    columns=sorted(list(phasis_result_df["alib"].unique())),  # Sort columns alphanumerically
    index=list(phasis_result_df["identifier"].unique())
    )

    for i in tqdm(data.index, desc="Processing Rows"):
        tempRows = phasis_result_df[phasis_result_df["identifier"] == i]

        for j in data.columns:
            k = 0.0
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty:
                if "non-PHAS" in subSetData["label"].values:
                    k = 0.0
                elif "PHAS" in subSetData["label"].values:
                    # Ensure that k is a scalar value, not a Series
                    if len(subSetData['log_clust_len_norm_counts']) > 0:
                        k = float(subSetData['log_clust_len_norm_counts'].iloc[0])  # Get the first value and cast to float

            data.loc[i, j] = k

    # Sort chromosomes numerically instead of lexicographically (i.e., 1, 2, 3, ... not 1, 10, 2, 3, ...)
    chrom_data = [identifier.split(':')[0] for identifier in data.index]
    sorted_indices = sorted(range(len(chrom_data)), key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx])

    # Reorder the DataFrame based on sorted chromosomes
    data = data.iloc[sorted_indices]

    # Re-extract the sorted chromosome data
    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    # Get the maximum value for normalization
    max_value = data.max().max()

    f, ax = plt.subplots(figsize=(11, 11))

    # Define custom colors
    colors = ["#C3D8EA", "#3F2F13"]
    cmap = LinearSegmentedColormap.from_list('Custom', colors, 10)

    # Normalize the colormap to match the range of your data
    norm = Normalize(vmin=0, vmax=max_value)

    # Create a custom color bar
    cax = inset_axes(ax,
                     width="5%",  # Set width for small size
                     height="9%",  # Set height for vertical orientation
                     loc='lower left',  # Position at the top left corner
                     bbox_to_anchor=(-0.25, 1.01, 1, 1),  # Adjust bbox_to_anchor for position
                     bbox_transform=ax.transAxes,
                     borderpad=0
                     )

    # Create a custom color bar that reflects the normalized data range
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation='vertical')
    cbar.set_ticks([0, max_value/3, 2*max_value/3, max_value])  # Adjust ticks as needed
    cbar.set_ticklabels([f"{0:.1f}", f"{max_value/3:.1f}", f"{2*max_value/3:.1f}", f"{max_value:.1f}"])  # Set tick labels

    # Plot the heatmap with normalization
    heat = sns.heatmap(data, square=False, cmap=cmap, cbar=False, norm=norm, xticklabels=True, yticklabels=False, ax=ax)

    # Add a new axis for the chromosome bar to the left
    cax2 = f.add_axes([0.17, 0.1, 0.02, 0.8])  # Adjust the position for the left side

    # Manually create a discrete grayscale colormap for the chromosomes
    base_cmap = plt.get_cmap('Greys')
    chrom_cmap = base_cmap(np.linspace(0.3, 0.9, len(unique_chromosomes)))  # Discretizing to avoid full black/white

    # Assign colors based on the order of unique chromosomes
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}

    # Map chromosomes to grayscale colors
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    # Plot the chromosome colorbar (no x-axis)
    cax2.imshow(chrom_colors, cmap=LinearSegmentedColormap.from_list('CustomGreys', chrom_cmap, len(unique_chromosomes)), aspect="auto")
    cax2.set_xticks([])  # Remove x-axis ticks
    cax2.set_yticks([])  # Remove y-axis ticks

    # Remove the black border around the colorbar
    cax2.spines['top'].set_visible(False)
    cax2.spines['right'].set_visible(False)
    cax2.spines['bottom'].set_visible(False)
    cax2.spines['left'].set_visible(False)

    # Align chromosome bar with heatmap y-coordinates
    cax2.set_ylim(len(data.index), 0)  # Set y-limits to invert the chromosome bar

    # Calculate midpoints for chromosome labels
    chrom_positions = np.array([np.mean(np.where(np.array(chrom_data) == chrom)) for chrom in unique_chromosomes])
    cax2.set_yticks(chrom_positions)
    # Set chromosome names with specific fontsize and rotation
    cax2.set_yticklabels(unique_chromosomes, fontsize=16, rotation=0, ha='center')  # Set chromosome names

    # Adjust the y-axis tick parameters to move labels slightly to the left
    cax2.yaxis.set_tick_params(labelsize=16, pad=11)  # Adjust pad to move the labels to the left
    # Rotate column text 45 degrees clockwise and adjust font size
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # Adjust space for better visualization - Increased left margin for y-tick labels
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)  # Adjust the margins

    # Save the plot
    fig = heat.get_figure()
    def _p(d, name):  # path join w/o imports
        return name if not d else (d + '/' + name if not d.endswith('/') else d + name)
    
    fig.savefig(_p(outdir, f'{type}_{phase}_Abundance_PHAS.pdf'), dpi=300)
    plt.close(fig)

    return None

def plot_totalAbundance_heat_map(phasis_result_df, type):
    print("#### Plotting PHAS and non-PHAS Heatmaps ######")

    # Create DataFrames for PHAS and non-PHAS
    #data_phas = pd.DataFrame(data=0.0, columns=list(phasis_result_df["alib"].unique()), index=list(phasis_result_df["identifier"].unique()))
    data_phas = pd.DataFrame(
    data=0.0, 
    columns=sorted(list(phasis_result_df["alib"].unique())),  # Sort columns alphanumerically
    index=list(phasis_result_df["identifier"].unique())
    )
    data_non_phas = data_phas.copy()

    # Fill DataFrames with values from phasis_result_df for PHAS and non-PHAS
    for i in tqdm(data_phas.index, desc="Processing Rows"):
        tempRows = phasis_result_df[phasis_result_df["identifier"] == i]
        
        for j in data_phas.columns:
            subSetData = tempRows[tempRows["alib"] == j]
            if not subSetData.empty:
                if "PHAS" in subSetData["label"].values and len(subSetData['total_abund']) > 0:
                    data_phas.loc[i, j] = float(subSetData['total_abund'].iloc[0])
                elif "non-PHAS" in subSetData["label"].values and len(subSetData['total_abund']) > 0:
                    data_non_phas.loc[i, j] = float(subSetData['total_abund'].iloc[0])

    # Apply log10 normalization
    data_phas = np.log10(data_phas.replace(0, np.nan).fillna(1e-10))
    data_non_phas = np.log10(data_non_phas.replace(0, np.nan).fillna(1e-10))

    # Sort chromosomes numerically instead of lexicographically (i.e., 1, 2, 3, ... not 1, 10, 2, 3, ...)
    chrom_data = [identifier.split(':')[0] for identifier in data_phas.index]
    sorted_indices = sorted(range(len(chrom_data)), key=lambda idx: int(chrom_data[idx]) if chrom_data[idx].isdigit() else chrom_data[idx])

    # Reorder DataFrames based on sorted chromosomes
    data_phas = data_phas.iloc[sorted_indices]
    data_non_phas = data_non_phas.iloc[sorted_indices]

    # Re-extract the sorted chromosome data
    chrom_data = [chrom_data[idx] for idx in sorted_indices]
    unique_chromosomes = sorted(np.unique(chrom_data), key=lambda x: int(x) if x.isdigit() else x)

    # Max values for normalization
    max_value_phas = data_phas.max().max()
    max_value_non_phas = data_non_phas.max().max()

    f, ax = plt.subplots(figsize=(11, 11))

    # Define custom colors for PHAS and non-PHAS
    phas_colors = ["#D5E6D6", "#FF9999", "#FF0000"]  # Pastel red to red for PHAS
    non_phas_colors = ["#D5E6D6", "#9999FF", "#0000FF"]
    # Normalize the colormap for PHAS and non-PHAS
    norm_phas = Normalize(vmin=0, vmax=max_value_phas)
    norm_non_phas = Normalize(vmin=0, vmax=max_value_non_phas)

    # Plot PHAS heatmap first (non-transparent, priority layer)
    cmap_phas = LinearSegmentedColormap.from_list('PHAS', phas_colors, N=256)
    sns.heatmap(data_phas, square=False, cmap=cmap_phas, cbar=False, norm=norm_phas, xticklabels=True, yticklabels=False, ax=ax)

    # Overlay non-PHAS heatmap with alpha blending (transparency)
    cmap_non_phas = LinearSegmentedColormap.from_list('non-PHAS', non_phas_colors, N=256)
    sns.heatmap(data_non_phas, square=False, cmap=cmap_non_phas, cbar=False, norm=norm_non_phas, xticklabels=True, yticklabels=False, ax=ax, alpha=0.5)

    # Create color bars for PHAS and non-PHAS
    cax_phas = inset_axes(ax,
                          width="5%",  # Set width for small size
                          height="30%",  # Set height for vertical orientation
                          loc='lower left',
                          bbox_to_anchor=(-0.25, 0.6, 1, 1),  # Positioned closer
                          bbox_transform=ax.transAxes,
                          borderpad=0)
    cbar_phas = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_phas, norm=norm_phas), cax=cax_phas, orientation='vertical')
    cbar_phas.set_ticks([0, max_value_phas/3, 2*max_value_phas/3, max_value_phas])
    cbar_phas.set_ticklabels([f"{0:.1f}", f"{max_value_phas/3:.1f}", f"{2*max_value_phas/3:.1f}", f"{max_value_phas:.1f}"])
    cbar_phas.set_label(r'log of $\it{PHAS}$ abundance', rotation=90, labelpad=15)

    cax_non_phas = inset_axes(ax,
                              width="5%",  # Set width for small size
                              height="30%",  # Set height for vertical orientation
                              loc='lower left',
                              bbox_to_anchor=(-0.25, 0.0, 1, 1),  # Positioned below PHAS with vertical space
                              bbox_transform=ax.transAxes,
                              borderpad=0)
    cbar_non_phas = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_non_phas, norm=norm_non_phas), cax=cax_non_phas, orientation='vertical')
    cbar_non_phas.set_ticks([0, max_value_non_phas/3, 2*max_value_non_phas/3, max_value_non_phas])
    cbar_non_phas.set_ticklabels([f"{0:.1f}", f"{max_value_non_phas/3:.1f}", f"{2*max_value_non_phas/3:.1f}", f"{max_value_non_phas:.1f}"])
    cbar_non_phas.set_label(r'log of non-$\it{PHAS}$ abundance', rotation=90, labelpad=10)

    # Add chromosome bar with shades of grey
    cax2 = f.add_axes([0.17, 0.1, 0.02, 0.8])  # Adjust position for the left side
    base_cmap = plt.get_cmap('Greys')
    chrom_cmap = base_cmap(np.linspace(0.3, 0.9, len(unique_chromosomes)))  # Discretizing to avoid full black/white
    chrom_color_map = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
    chrom_colors = np.array([chrom_color_map[chrom] for chrom in chrom_data]).reshape(-1, 1)

    # Plot the chromosome colorbar
    cax2.imshow(chrom_colors, cmap=LinearSegmentedColormap.from_list('CustomGreys', chrom_cmap, len(unique_chromosomes)), aspect="auto")
    cax2.set_xticks([])  # Remove x-axis ticks
    cax2.set_yticks([])  # Remove y-axis ticks

    # Add chromosome numbers
    chrom_positions = np.array([np.mean(np.where(np.array(chrom_data) == chrom)) for chrom in unique_chromosomes])
    cax2.set_yticks(chrom_positions)
    cax2.set_yticklabels(unique_chromosomes, fontsize=12, rotation=0, ha='center')
    cax2.yaxis.set_tick_params(labelsize=16, pad=11)  # Adjust pad to move the labels to the left

    # Rotate column text 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # Adjust plot space for better visualization
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

    # Save the plot
    fig = ax.get_figure()
    def _p(d, name):  # path join w/o imports
        return name if not d else (d + '/' + name if not d.endswith('/') else d + name)
    fig.savefig(_p(outdir, f'{type}_{phase}_Abundance_PHAS_and_nonPHAS.pdf'), dpi=300)
    plt.close(fig)

    return None

def _parse_identifiers_and_alib(features: pd.DataFrame):
    """
    Return achr, start, end, cleaned alib arrays from features['identifier']/['alib'].
    If 'identifier' is not in 'chr:start..end' form, try to resolve via MERGED_CLUSTER_REVERSE
    using the row's 'cID'. Falls back to blanks to avoid IndexError.
    """
    achr, start, end = [], [], []
    rmap = globals().get("MERGED_CLUSTER_REVERSE", {}) or {}

    # iterate row-wise so we can look at both identifier and cID
    for id_str, cID in zip(features["identifier"].astype(str), features.get("cID", pd.Series([None]*len(features)))):
        u = None
        if ":" in id_str and ".." in id_str:
            u = id_str
        else:
            # try reverse map by cID first, then by the identifier string itself
            if pd.notna(cID) and str(cID) in rmap:
                u = rmap[str(cID)]
            elif id_str in rmap:
                u = rmap[id_str]

        if u and ":" in u and ".." in u:
            left, right = u.split(":", 1)
            achr.append(left)
            s_val, e_val = right.split("..", 1)
            start.append(s_val)
            end.append(e_val)
        else:
            achr.append("")
            start.append("")
            end.append("")

    # keep alib as-is unless it ends with ".{phase}-PHAS.candidate"
    alib_src = features["alib"].astype(str).tolist()
    alib_ids = [re.sub(rf"\.{re.escape(str(phase))}-PHAS\.candidate$", "", x) for x in alib_src]
    return achr, start, end, alib_ids

def _plot_wrapper(args):
    fn, df, mname = args
    return fn(df, mname)

def _finalize_and_write_results(method_name: str, features: pd.DataFrame):
    """
    Build result dataframe (all clusters + filtered PHAS),
    write standardized outputs, run 3 tree plots in parallel, and write GFF.
    """
    # Decompose identifiers and clean alib tags
    achr, start, end, alib_ids = _parse_identifiers_and_alib(features)

    nrows = len(features)
    def _fallback_series():
        return pd.Series([np.nan] * nrows)

    # ---- Full 'all clusters' table (unchanged, includes strict columns) ----
    all_df = pd.DataFrame({
        'identifier': features['identifier'],
        'phasis_score': features['phasis_score'],
        'achr': achr,
        'start': start,
        'end': end,
        'complexity': features['complexity'],
        'strand_bias': features['strand_bias'],
        'log_clust_len_norm_counts': features['log_clust_len_norm_counts'],
        'ratio_abund_len_phase': features['ratio_abund_len_phase'],
        'label': features['label'],
        'alib': alib_ids,
        'combined_fishers': features.get('combined_fishers', _fallback_series()),
        'total_abund': features.get('total_abund', _fallback_series()),
        'w_Howell_score': features.get('w_Howell_score', _fallback_series()),
        'w_window_start': features.get('w_window_start', _fallback_series()),
        'w_window_end': features.get('w_window_end', _fallback_series()),
        'c_Howell_score': features.get('c_Howell_score', _fallback_series()),
        'c_window_start': features.get('c_window_start', _fallback_series()),
        'c_window_end': features.get('c_window_end', _fallback_series()),
        'Peak_Howell_score': features.get('Peak_Howell_score', _fallback_series()),
        # strict (classic) Howell
        'w_Howell_score_strict': features.get('w_Howell_score_strict', _fallback_series()),
        'w_window_start_strict': features.get('w_window_start_strict', _fallback_series()),
        'w_window_end_strict': features.get('w_window_end_strict', _fallback_series()),
        'c_Howell_score_strict': features.get('c_Howell_score_strict', _fallback_series()),
        'c_window_start_strict': features.get('c_window_start_strict', _fallback_series()),
        'c_window_end_strict': features.get('c_window_end_strict', _fallback_series()),
        'Peak_Howell_score_strict': features.get('Peak_Howell_score_strict', _fallback_series()),
    })

    # ensure directory exists (assumes 'os' is already imported elsewhere)
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass

    def _p(dirpath, name):
        if not dirpath:
            return name
        return dirpath + name if dirpath.endswith('/') else dirpath + '/' + name

    # Standardized filenames
    all_out   = _p(outdir, f"{phase}_{method_name}_all_clusters.tsv")
    calls_out = _p(outdir, f"{phase}_{method_name}_calls.tsv")
    gff_out   = _p(outdir, f"{phase}_PHAS.gff")
 
    # Write all clusters with labels
    all_df.to_csv(all_out, sep="\t", index=False)

    # Keep only PHAS
    phas_df = all_df[all_df['label'] == 'PHAS'].copy()

    # Write GFF
    write_gff(phas_df,gff_out)

    # ---- Compact calls table (same as before + Peak_Howell_score_strict) ----
    calls_cols = [
        'identifier', 'phasis_score', 'achr', 'start', 'end', 'alib',
        'Peak_Howell_score', 'Peak_Howell_score_strict'
    ]
    # Ensure missing columns are created as NaN so write doesn't fail
    for col in calls_cols:
        if col not in phas_df.columns:
            phas_df[col] = np.nan
    compact_calls = phas_df[calls_cols].copy()
    compact_calls.to_csv(calls_out, sep="\t", index=False)

    # --- Run the 3 plots in parallel (each on a core) ---
    plot_jobs = [
        (plot_report_heat_map, all_df, method_name),
        (plot_phasAbundance_heat_map, all_df, method_name),
        (plot_totalAbundance_heat_map, all_df, method_name),
    ]
    with Pool(processes=min(3, cpu_count())) as p:
        p.map(_plot_wrapper, plot_jobs)

    print(f"  - Wrote: {all_out}, {calls_out}, and {gff_out}")

def KNN_phas_clustering(features: pd.DataFrame):
    """
    KNN classifier (reference implementation).
    Uses saved KNN model; applies post-filters and writes standardized outputs.
    """
    print("### KNN classifier ###")

    # Scale the same KNN feature set as before
    X = features[['complexity', 'strand_bias', 'log_clust_len_norm_counts', 'ratio_abund_len_phase', 'phasis_score']].copy()
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model and predict
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, 'knn_model.pkl')
    knn_clf = joblib.load(model_path)
    y_pred = knn_clf.predict(X_scaled)

    # Labels
    features = features.copy()
    features['label'] = np.where(y_pred == 1, 'PHAS', 'non-PHAS')

    # Post-filters to tighten calls (gold standard policy)
    features.loc[(features['phasis_score'] < phasisScoreCutoff) | (features['Peak_Howell_score'] < min_Howell_score), 'label'] = 'non-PHAS'
    features.loc[(features['complexity'] > max_complexity), 'label'] = 'non-PHAS'

    # Finalize outputs + plots (in parallel)
    _finalize_and_write_results("KNN", features)

def GMM_phas_clustering(features: pd.DataFrame, n_clusters: int = 2):
    """
    GMM clustering aligned with KNN feature scaling + post-filters.
    Chooses the GMM cluster with the highest mean phasis_score as PHAS.
    """
    print("### GMM classifier ###")

    # Use the same *core* feature philosophy as KNN; keep Peak_Howell for post-filter only
    cols_for_model = ['complexity', 'strand_bias', 'log_clust_len_norm_counts', 'ratio_abund_len_phase', 'phasis_score']
    X = features[cols_for_model].copy()
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM and label the cluster with highest mean phasis_score as PHAS
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    cluster_labels = gmm.fit_predict(X_scaled)
    tmp = pd.DataFrame({'cluster': cluster_labels, 'phasis_score': features['phasis_score'].values})
    phas_cluster = tmp.groupby('cluster')['phasis_score'].mean().idxmax()

    features = features.copy()
    features['label'] = np.where(cluster_labels == phas_cluster, 'PHAS', 'non-PHAS')

    # Apply the exact same post-filters as KNN to keep semantics consistent
    features.loc[(features['phasis_score'] < phasisScoreCutoff) | (features['Peak_Howell_score'] < min_Howell_score), 'label'] = 'non-PHAS'
    features.loc[(features['complexity'] > max_complexity), 'label'] = 'non-PHAS'

    # Finalize outputs + plots (in parallel)
    _finalize_and_write_results("GMM", features)

def chromosome_clusters_to_candidate_loci(chromosome_df):
    '''
    For a single chromosome, convert clusters to candidate loci based on minimum length.
    '''
    lociTablelist = []
    
    # Group by clusterID and calculate min and max positions
    cluster_positions = chromosome_df.groupby("clusterID")["pos"].agg(["min", "max"]).reset_index()

    # Merge to get the chromosome for each clusterID
    cluster_info = chromosome_df.merge(cluster_positions, on='clusterID')

    # Clean up the clusterID to remove extra tabs or whitespace
    cluster_info['clusterID'] = cluster_info['clusterID'].str.strip()

    # Create a mask for clusters longer than minClusterLength
    mask = (cluster_info['max'] - cluster_info['min']) >= minClusterLength

    # Apply mask and create lociTablelist with relevant columns, using .values.tolist()
    lociTablelist = cluster_info[mask].apply(lambda row: [
        row['clusterID'].replace('\t', '').strip(),
        0,
        int(row['chromosome']),
        int(row['min']),
        int(row['max'])
    ], axis=1).values.tolist()

    return lociTablelist
    
def loci_table_from_clusters(allClusters):
    print("### Building loci table from clusters per chromosome ###")
    
    outfname = phase2_basename('candidate.loci_table.tab')

    # Step 0: Check if output is up-to-date (file exists and hash matches memory file)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)
    section = 'LOCI_TABLE'
    if not config.has_section(section):
        config.add_section(section)

    if os.path.isfile(outfname):
        _, current_md5 = getmd5(outfname)
        prev_md5 = config[section].get(outfname)
        if prev_md5 and current_md5 == prev_md5:
            print(f"File {outfname} is up-to-date (hash match). Skipping recomputation.")
            print(f"Loci table written to {outfname}")
            with open(outfname, 'r') as file:
                lines = file.readlines()[1:]  # skip header
                lociTablelist_unique = [line.strip().split('\t') for line in lines]
            return pd.DataFrame(lociTablelist_unique, columns=["name","pval","chr","start","end"])

    # Step 1: Split data into chromosome groups
    chromosome_groups = [df for _, df in allClusters.groupby("chromosome")]
    
    # Step 2: Multicore processing (unit 'lib-chr')
    lociTablelist = run_parallel_with_progress(
        chromosome_clusters_to_candidate_loci,
        chromosome_groups,
        desc="LociTable chromosomes",
        min_chunk=1,
        unit="lib-chr"
    )

    # Step 3: Flatten the results
    lociTablelist = [item for sublist in lociTablelist for item in sublist]

    # Step 4: Remove duplicates
    seen = set()
    lociTablelist_unique = []
    for item in lociTablelist:
        row_tuple = tuple(item)
        if row_tuple not in seen:
            seen.add(row_tuple)
            lociTablelist_unique.append(item)

    # Step 5: Write results to file
    with open(outfname, 'w') as file:
        file.write("Cluster\tvalue1\tchromosome\tStart\tEnd\n")
        for row in lociTablelist_unique:
            file.write('\t'.join(map(str, row)) + '\n')

    # Step 6: Update memory file with output hash
    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        config[section][outfname] = out_md5
        print(f"Hash for {outfname}: {out_md5}")
    with open(memFile, 'w') as fh:
        config.write(fh)

    print(f"Loci table written to {outfname}")
    return pd.DataFrame(lociTablelist_unique, columns=["name","pval","chr","start","end"])

def main(libs):
    """
    Orchestrates the pipeline (concat- and non-concat-safe).

    Guarantees:
      - {phase}_candidate.loci_table.tab is written before any merging.
      - mergedClusterDict is ALWAYS available for getUniversalID(), in both modes.
      - WIN_SCORE_LOOKUP is set after window scoring so workers can read it.

    Notes:
      - Respects hash/caching behavior inside called steps.
      - Early exits if no clusters/windows.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    clusterFilePaths = None
    # make it globally available
    print(f"Output directory: {outdir}")

    # -------------------------- PART I: cfind --------------------------
    if (steps == 'both') or (steps == 'cfind'):
        print("######            Starting Phase I           #########")
        runLog = 'runtime_%s' % datetime.datetime.now().strftime("%m_%d_%H_%M")
        fh_run = open(runLog, 'w')
        try:
            # Create folders, build/reuse index
            clustfolder = createfolders(os.getcwd())
            genoIndex   = getindex(fh_run)

            # Preprocess → map → parse
            libs_processed                    = libraryprocess(libs)
            libs_mapped                       = mapprocess(libs_processed, genoIndex)
            libs_nestdict, libs_poscountdict  = parserprocess(libs_processed)

            # Find clusters
            libs_clustdicts = clusterprocess(libs_poscountdict, clustfolder)

            # Score (bucketed hashing will skip when cached)
            clusterFilePaths = scoringprocess(
                libs_processed, libs_clustdicts, libs_nestdict, clustfolder
            )
        finally:
            try: fh_run.close()
            except Exception: pass

    # -------------------------- PART II: class -------------------------
    if (steps == 'both') or (steps == 'class'):
        print("######            Starting Phase II          #########")

        # If running 'class' only, take precomputed cluster files
        if steps == 'class':
            clusterFilePaths = class_cluster_file

        # 1) Aggregate for side-effects (writes processed clusters)
        agg_df = aggregate_and_write_processed_clusters(clusterFilePaths)

        # Build "allClusters" baseline from aggregator result or file fallback
        if agg_df is not None:
            allClusters = agg_df
        else:
            proc_path = phase2_basename('processed_clusters.tab')
            allClusters = (
                pd.read_csv(proc_path, sep="\t") if os.path.isfile(proc_path) else pd.DataFrame()
            )

        # Normalize for downstream grouping
        allClusters = _normalize_cluster_df(allClusters, is_concat=concat_libs)

        # 2) ALWAYS emit loci table BEFORE merge (guarantees the input exists for merge)
        loci_table_df   = loci_table_from_clusters(allClusters)  # writes {phase}_candidate.loci_table.tab
        loci_table_path = phase2_basename('candidate.loci_table.tab')

        # 3) Build the DataFrame used downstream
        merged_out_path = phase2_basename('merged_candidates.tab')
        if concat_libs:
            # Cache-aware merge: returns a DataFrame (loads TSV on cache hit)
            allClustersMerged = merge_candidate_clusters_across_libs(
                loci_table_path, merged_out_path
            )
        else:
            # Non-concat: analysis continues with the pre-merge representation
            # (cross-lib overlap merging for universal IDs is handled right below)
            allClustersMerged = allClusters

        # 3.5) Always ensure universal-ID dict (used by getUniversalID)
        #      In concat mode: load/write identity dict from merged TSV if needed.
        #      In non-concat: perform cross-lib merging via parametric path.
        mcd = ensure_mergedClusterDict_always(
            concat_libs=concat_libs,
            phase=phase,
            merged_out_path=merged_out_path,
            loci_table_df=loci_table_df,
            allClusters_df=allClusters,
            memFile=memFile
        )
        globals()['mergedClusterDict'] = mcd
        print(f"[INFO] mergedClusterDict ready with {len(mcd)} universal IDs.")

        # 4) Build PHAS clusters (handles empty input)
        clusters_data = build_and_save_phas_clusters(allClusters)

        # 5) If there are no clusters, short-circuit cleanly
        if clusters_data is None or getattr(clusters_data, "empty", True):
            print("[INFO] No PHAS clusters to score; exiting classification early.")

        # 6) Select windows (cache-aware and robust)
        clusters = select_scoring_windows(clusters_data)
        if clusters is None or getattr(clusters, "empty", True):
            print("[INFO] No scoring windows found; exiting classification early.")

        # 7) Score windows, expose compact lookup to workers, extract features, classify
        win_phasis_score = compute_and_save_phasis_scores(clusters)

        # Make the compact, read-only lookup visible to workers (for process_chromosome_features)
        set_win_score_lookup(win_phasis_score)

        features = features_to_detection(clusters_data)
        if classifier == "GMM":
            GMM_phas_clustering(features)
        elif classifier == "KNN":
            KNN_phas_clustering(features)

        if args.cleanup:
            cleanup()
        return None
    
if __name__ == '__main__':
    #### Cores to use for analysis
    ncores          = coreReserve(cores)
    checkDependency()
    libs            = checkLibs()
    main(libs)
