import os


def compute_index_fingerprints(
    reference: str,
    genoIndex: str,
    compute_fingerprint_fn,
) -> tuple[str, str, str]:
    """
    Compute fast fingerprints for the reference FASTA and a HiSat2 index marker file.

    Returns:
        (ref_fingerprint, index_fingerprint, index_marker_path)

    Notes:
    - Intended for cache invalidation / memFile tracking, not cryptographic integrity.
    - Index marker file selection (in order):
        1) {genoIndex}.1.ht2l
        2) {genoIndex}.1.ht2

    Raises:
        FileNotFoundError: if neither marker exists.
    """
    ref_fp = compute_fingerprint_fn(reference) or ""

    marker_l = f"{genoIndex}.1.ht2l"
    marker_s = f"{genoIndex}.1.ht2"

    if os.path.isfile(marker_l):
        marker = marker_l
    elif os.path.isfile(marker_s):
        marker = marker_s
    else:
        raise FileNotFoundError(
            f"Could not determine HiSat2 index marker file for genoIndex='{genoIndex}'. "
            f"Tried '{marker_l}' and '{marker_s}'."
        )

    idx_fp = compute_fingerprint_fn(marker) or ""
    return (ref_fp, idx_fp, marker)

def indexIntegrityCheck(index: str) -> tuple[bool, str | bool]:
    """
    Check HiSat2 index integrity and infer extension.

    Returns:
        (indexIntegrity, indexExt)

    indexExt is "ht2l", "ht2", or False if undetermined.
    """
    indexFolder = index.rpartition("/")[0]

    if os.path.isfile(f"{index}.1.ht2l"):
        indexExt = "ht2l"
        indexFiles = [i for i in os.listdir(indexFolder) if i.endswith(".ht2l")]
        indexIntegrity = len(indexFiles) >= 6
    elif os.path.isfile(f"{index}.1.ht2"):
        indexExt = "ht2"
        indexFiles = [i for i in os.listdir(indexFolder) if i.endswith(".ht2")]
        indexIntegrity = len(indexFiles) >= 6
    else:
        print("Existing index extension couldn't be determined")
        print("Genome index will be remade")
        indexExt = False
        indexIntegrity = False

    return indexIntegrity, indexExt

