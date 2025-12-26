#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TagToFasta: convert a 2-column "tag" table into an expanded FASTA file.
Input format (whitespace- or tab-separated):
  <SEQUENCE> <COUNT>
Example:
  TTTGGATTGAAGGGAGCTCT    3824

Writes COUNT FASTA records per line with compact headers: >s1, >s2, ...
Supports .gz input/output by file extension; streams output.
"""

import sys, os, gzip, argparse

def _open(path, mode="rt"):
    """Open plain or gzipped files based on extension. '-' means stdin/stdout."""
    if path == "-":
        return sys.stdin if "r" in mode else sys.stdout
    if path.endswith(".gz"):
        return gzip.open(path, mode)  # text mode if 't' present
    return open(path, mode, newline="")

def wrap_seq(seq: str, width: int) -> str:
    if width <= 0:
        return seq
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def tag_to_fasta(in_path: str, out_path: str, prefix: str = "s", start_index: int = 1, wrap: int = 0) -> int:
    """
    Convert tag table to expanded FASTA.
    Returns total number of FASTA records written.
    """
    total = 0
    idx = start_index
    with _open(in_path, "rt") as inf, _open(out_path, "wt") as outf:
        for ln, line in enumerate(inf, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                sys.stderr.write(f"[WARN] Line {ln}: expected 2+ columns, got {len(parts)} -> skipped\n")
                continue
            seq, cnt_str = parts[0], parts[1]
            try:
                count = int(cnt_str)
            except ValueError:
                sys.stderr.write(f"[WARN] Line {ln}: non-integer count '{cnt_str}' -> skipped\n")
                continue
            if count <= 0:
                continue

            seq_wrapped = wrap_seq(seq, wrap)
            for _ in range(count):
                outf.write(f">{prefix}{idx}\n{seq_wrapped}\n")
                idx += 1
                total += 1
    return total

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Expand a <sequence><count> tag table into a multi-FASTA file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("input", help="Input tag file (whitespace-separated). Use '-' for STDIN. Supports .gz")
    ap.add_argument("-o", "--output", required=True, help="Output FASTA path. Use '-' for STDOUT. Supports .gz")
    ap.add_argument("--prefix", default="s", help="Header prefix (records will be >{prefix}{N})")
    ap.add_argument("--start-index", type=int, default=1, help="Starting index for headers")
    ap.add_argument("--wrap", type=int, default=0, help="Wrap sequence lines to this width (0 = no wrap)")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    try:
        total = tag_to_fasta(args.input, args.output, prefix=args.prefix, start_index=args.start_index, wrap=args.wrap)
    except BrokenPipeError:
        try: sys.stdout.close()
        except Exception: pass
        try: sys.stderr.close()
        except Exception: pass
        return 0

    sys.stderr.write(f"[OK] Wrote {total} FASTA records to {args.output}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())

