# Phasis — Phased sRNA Cluster Discovery and Annotation

**Version:** v2  
**Updated:** 2026-01-20

Phasis is a parallelized tool for large-scale analysis of small RNA (sRNA) libraries. It supports:

- *De novo* discovery of **PHAS loci** and precursor transcripts
- Summarization and comparison of PHAS loci across stages, tissues, and treatments
- Quantification and annotation of PHAS loci

---

## Installation

### 1) Create an environment

**Conda (recommended):**
```bash
conda create -n phasis python=3.12 -y
conda activate phasis
```

### 2) Install external tools

Phasis requires these executables on your `PATH`:

- **hisat2**
- **samtools**

Install via conda (example):
```bash
conda install -c bioconda hisat2 samtools -y
```

### 3) Install Phasis 

From the Phasis repository root:
```bash
python -m pip install -U pip
python -m pip install -e .
```

After this, you should have a `phasis` command available:
```bash
phasis -h
```

---

## How Phasis writes files (important)

Phasis uses **two locations**:

1) **Run directory** = your current working directory (the directory you run `phasis` from)  
   - Intermediate files (e.g., `21_candidate.loci_table.tab`, `21_processed_clusters.tab`, etc.)
   - Reusable caches such as:
     - `index/` (HISAT2 index)
     - `phasis.mem` (hash cache that decides what can be reused across runs/phases)

2) **Output directory (`--outdir`)**  
   - Final outputs for the selected phase (default: `{phase}_results`)

This design allows you to run **21-PHAS** and then **24-PHAS** from the same run directory while reusing safe intermediates (especially the HISAT2 index).

---

## Quick start

### 21-PHAS (default)
```bash
phasis -libs *.tag -libformat T -reference genome.fa -phase 21 -cores 0
```

### 24-PHAS
```bash
phasis -libs *.tag -libformat T -reference genome.fa -phase 24 -cores 0
```

You can keep the same run directory and let Phasis reuse `index/` and other intermediates via `phasis.mem`.

---

## Outputs

For each phase (e.g., `21`, `24`), Phasis writes the main outputs into `--outdir` (default: `{phase}_results`):

1. **`{phase}.phasis.result.tsv`**  
   Universal identifier per PHAS locus (across one or multiple libraries), plus score, chromosome, start/end, and library.

2. **`{phase}_PHAS.gff`**  
   GFF for downstream analyses (e.g., differential accumulation).

3. **`{phase}_Abundance_PHAS.pdf`**  
   Heatmap of phased-length read accumulation for each detected locus.

4. **`{phase}_PHAS.pdf`**  
   Heatmap distinguishing PHAS loci from non-PHAS sRNA clusters.

5. **`{phase}_all_clusters_result.tsv`**  
   Summary table for **all** evaluated sRNA clusters (not only PHAS), including features used for classification.

---

## Common options

- `-maxhits` (default: 25): `-k` passed to hisat2
- `-runtype` (default: G): `G` genome | `T` transcriptome | `S` scaffolded genome
- `-mindepth` (default: 2): minimum depth for p-value computation
- `-uniqueRatioCut` (default: 0.2): filter for uniquely mapped reads
- `-mismat` (default: 0): mismatches allowed in mapping
- `-phase` (default: 21): phasing length (21 or 24 common)
- `-clustbuffer` (default: 300): merging distance between clusters
- `-phasisScoreCutoff` (default: 50 for 21; internally clamped for 24 to 250–300)
- `-minClusterLength` (default: 350)
- `-cores` (default: 0): 0 uses most free cores; `>0` sets exact core count
- `-norm`: enable CP10M normalization (use `-norm_factor` to change factor; default `1e7`)
- `-classifier` (default: KNN): `KNN` or `GMM`
- `-steps` (default: both): `both` | `cfind` | `class`
- `-cleanup`: delete intermediates after run (**not recommended** if you plan to run multiple phases)

---

## High-sensitivity two-step workflow (no bash for-loops)

This approach is useful when you want to be permissive during clustering and then classify jointly.

### Step 1 — cluster detection
```bash
phasis -mindepth 1 -phase 24 -libformat T -classifier KNN \
  -libs *.tag -reference genome.fa -cores 0 -maxhits 2000 \
  -steps cfind -uniqueRatioCut 0.05
```

### Step 2 — classification
```bash
phasis -mindepth 1 -phase 24 -libformat T -classifier KNN \
  -libs *.tag -reference genome.fa -cores 0 -maxhits 2000 \
  -steps class -class_cluster_file *24-PHAS.candidate.clusters \
  -uniqueRatioCut 0.05
```

Notes:
- Setting `uniqueRatioCut` too low (e.g., 0.0) can dramatically increase runtime and memory in TE-rich genomes.
- Keeping the **same run directory** between steps lets Phasis reuse intermediates safely.

---

## Input library formats

Phasis accepts two library formats:

- **FASTA** (`-libformat F`)
- **Tag-count** (`-libformat T`) — recommended when FASTA is large or contains many ambiguous bases.

### Convert FASTQ → FASTA
Script: `support_scripts/fastq_to_fasta.sh`
```bash
bash support_scripts/fastq_to_fasta.sh sample.fq
```

### Convert FASTA → tag-count
Script: `support_scripts/fastaToTag.py`
```bash
python support_scripts/fastaToTag.py sample.fasta
```

Then run Phasis:
```bash
phasis -libs *.tag -reference genome.fa -libformat T
```

---

## FASTA headers: non-integer chromosome IDs

Phasis uses FASTA headers as keys. Very long or non-integer chromosome IDs can increase memory usage and may cause failures.

Use:
```bash
python support_scripts/replace_genome_headers.py genome.fa new_genome.fa equivalence.tsv
```

This writes:
- `new_genome.fa` with integer chromosome IDs
- `equivalence.tsv` mapping old → new IDs

---

## Comparing PHAS loci between runs (phasMatch)

Script:
```bash
python support_scripts/phasMatch.v03.py <phasis.result.tsv> <alternative_predictions.tsv>
```

The script matches loci using genomic overlap with a default ±300 nt flanking.

---

## Authors

- Atul Kakrana — kakrana@gmail.com  
- Thales Cherubino Ribeiro — thalescherubino@gmail.com
- Blake Meyers - bcmeyers@ucdavis.edu