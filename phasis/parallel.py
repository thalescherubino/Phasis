import os, multiprocessing, gc, traceback, sys
from tqdm import tqdm
import phasis.runtime as rt
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
    n_data = len(data)
    if n_data == 0:
        return []
    ncores = rt.ncores
    if rt.ncores is None or rt.ncores <= 0:
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


def _infer_runtime_snapshot_path():
    # Prefer an explicit snapshot path if runtime.py defines one
    p = getattr(rt, "runtime_snapshot", None)
    if p and os.path.isfile(p):
        return p

    # Fallback: look in run_dir, then CWD
    run_dir = getattr(rt, "run_dir", None) or os.getcwd()
    cand = os.path.join(run_dir, ".phasis.runtime.json")
    if os.path.isfile(cand):
        return cand

    cand = os.path.join(os.getcwd(), ".phasis.runtime.json")
    if os.path.isfile(cand):
        return cand

    return None


def _pool_initializer(snapshot_path, kind):
    # Keep BLAS single-threaded in workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Plot pools on macOS must avoid GUI backends
    if kind == "plot":
        os.environ.setdefault("MPLBACKEND", "Agg")

    # Load runtime snapshot if available (spawn-safe)
    try:
        if snapshot_path and hasattr(rt, "load_snapshot"):
            rt.load_snapshot(snapshot_path)
    except Exception:
        pass

    # Ensure workers operate from the run directory where intermediates live
    try:
        rd = getattr(rt, "run_dir", None)
        if rd:
            os.chdir(rd)
    except Exception:
        pass

    # Sync legacy globals from runtime if the function exists
    try:
        from . import legacy
        if hasattr(legacy, "sync_from_runtime"):
            legacy.sync_from_runtime()
    except Exception:
        pass


def make_pool(nworkers: int | None = None, *, processes: int | None = None, start_method: str | None = None,
             kind: str = "compute", snapshot_path: str | None = None):
    """
    Pool with safer defaults to limit RAM spikes.

    - BLAS threads set to 1.
    - maxtasksperchild=1.
    - macOS: spawn by default (safe for ObjC/matplotlib).
    - Linux: fork by default.

    NEW:
    - Supports `processes=` kwarg as alias for nworkers.
    - Loads runtime snapshot in workers (spawn-safe).
    - kind="plot" sets MPLBACKEND=Agg inside workers.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if processes is not None:
        nworkers = processes
    nworkers = int(max(1, nworkers or 1))

    if start_method is None:
        start_method = "spawn" if sys.platform == "darwin" else "fork"

    if snapshot_path is None:
        snapshot_path = _infer_runtime_snapshot_path()

    if hasattr(multiprocessing, "get_context"):
        try:
            ctx = multiprocessing.get_context(start_method)
        except ValueError:
            ctx = multiprocessing.get_context()
    else:
        ctx = multiprocessing

    return ctx.Pool(
        processes=nworkers,
        maxtasksperchild=1,
        initializer=_pool_initializer,
        initargs=(snapshot_path, kind),
    )

def safe_worker(args):
    """Run func(arg), catching exceptions; return RuntimeError sentinel on failure."""
    func, arg = args
    try:
        return func(arg)
    except Exception as e:
        import traceback  # allowed here (small, unavoidable for nice trace)
        return RuntimeError(f"Error in {func.__name__} with arg={arg}: {e}\n{traceback.format_exc()}")


