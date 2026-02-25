import os

from phasis import runtime as rt


# Stage-local globals (minimal)
phase = None


def sync_from_runtime() -> None:
    """
    Populate folder-setup stage globals from phasis.runtime.
    """
    global phase
    phase = rt.phase


def createfolders(currdir):
    """
    Create basic folders at the beginning of process.
    Legacy-compatible behavior:
      - creates {currdir}/{phase}_clusters if missing
      - returns cluster folder path
    """
    global phase

    sync_from_runtime()

    if phase is None:
        raise RuntimeError("rt.phase is not set before createfolders()")

    clustfolder = "%s/%s_clusters" % (currdir, phase)
    if not os.path.isdir(clustfolder):
        os.mkdir("%s" % (clustfolder))
    else:
        pass

    return clustfolder