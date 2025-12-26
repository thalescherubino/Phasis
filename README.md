# *Phasis* for Phased Clusters Discovery and Annotation

**Version**: v5  
**Updated**: 2024/09/16  
**Citation**: [Preprint Copy](http://www.biorxiv.org/content/early/2017/07/03/158832.full.pdf+html)  

**Phasis** is a parallelized tool designed for the large-scale survey of sRNA libraries. It supports:

- *De novo* discovery of *PHAS* loci and precursor transcripts
- Summarization and comparison of *PHAS* loci between groups of samples from different stages, tissues, and treatments
- Quantification and annotation of *PHAS* loci

The tool outputs results for downstream bioinformatics analysis and formatted files for immediate use.

## **Installation**

Ensure the following tools/packages are compiled and installed:

- **Python** >= 3.12.1
- **HISAT2** >= 2.2.1
- **SciPy** >= 1.11.4
- **Pandas** >= 2.1.4
- **NumPy** >= 1.26.3
- **scikit-learn** >= 1.3.0
- **Matplotlib** >= 3.8.0
- **Seaborn** >= 0.12.2
- **Joblib** >= 1.2.0
- **tqdm** >= 4.65.0

## **Running Phasis**

Once the dependencies are installed, you can run Phasis using the command:

```bash
python path_to_phasis/phasis.py -libs lib1.fasta lib2.fasta [...] -reference genome.fa
```

## **Usage**

```bash
python path_to_phasis/phasis.py -libs [LIBS ...] -reference genome.fa
```
### Required Arguments

- libs [LIBS ...]: Quality controlled libraries to process
- reference genome.fa: Genome or transcriptome reference FASTA

Please name the reference genome or transcriptome file with the `.fa` extension to prevent potential errors during processing.

### Phasis Output Files

Phasis will output five files for each selected phase length parameter:

1. **{phase}.phasis.result.tsv**: This table provides a universal identifier for each *PHAS* locus found in one or multiple sRNA libraries, followed by the `Phasis score`, chromosome, start and end coordinates, and the library in which it was detected.

2. **{phase}_PHAS.gff**: A General Feature Format (GFF) file for downstream analyses, such as differential accumulation analysis.

3. **{phase}_Abundance_PHAS.pdf**: A heatmap depicting the intensity of read accumulation at a specific phase length for each detected locus.

4. **{phase}_PHAS.pdf**: A heatmap distinguishing *PHAS* loci from non-*PHAS* sRNA clusters.

5. **{phase}_all_clusters_result.tsv**: A summary table describing the parameters used to categorize a locus as phasing or not. Unlike the `phasis.result.tsv` file, this file includes data for all evaluated sRNA clusters, including those dominated by a specific sRNA length.

### Optional Arguments

- maxhits: Max genome/transcript hits to process (default: 25)
- runtype: Genome (G), transcriptome (T), or scaffolded genome (S) (default: G)
- mindepth: Minimum sRNA depth for p-value computation (default: 2)
- uniqueRatioCut: Proportion of uniquely mapped reads to filter out loci (default: 0.2)
- mismat: Mismatches allowed between sRNA and reference for mapping (default: 0)
- libformat: Quality controlled format: FASTA (F) or Tag count (T) (default: F)
- phase: Phase length to predict (default: 21)
- clustbuffer: Minimum distance between clusters to avoid merging (default: 300)
- phasisScoreCutoff: Minimum score to report PHAS locus (default: 50)
- minClusterLength: Minimum length to score a PHAS locus (default: 350)
- cores: Number of cores for analysis. Set 0 for most free cores or specify an integer to set exact cores (default: 0)
-norm: Allow Phasis to perform read count normalization (CP10M) [default False]. If enabled, you can use -norm_factor: to set a custom normalization factor instead of the default 1e7.
- classifier: Choose K Nearest Neighbors (KNN) classifier or unsupervised Gaussian mixture model (default: KNN)
- cleanup: Delete intermediate directories and files after the run (not recommended for successive runs)
- steps: Run cluster detection and classifier [default] | Only cluster detection [cfind] | Only classification [class]
- class_cluster_file: Cluster file name(s) for classification steps
- version: Print the version and exit
- force: designed for advanced users who wish to override the safeguards built into Phasis when running potentially resource-intensive tasks. Under certain parameter combinations, Phasis may require significant computational resources and can even cause system crashes, especially when working with large datasets or settings that push the system’s limits (even on machines with up to 500 GB of RAM).


### Balancing Sensitivity with Computational Efficiency

The default parameters of *Phasis* are configured to provide a reasonable balance between precision and sensitivity, while also minimizing the use of computational resources. However, depending on the number of small RNA sequencing libraries processed, *Phasis* may require significant amounts of RAM memory.

For users who need increased sensitivity, *Phasis* can be run in two steps. In the first step, each small RNA library is processed separately using the cluster detector mode (`-steps cfind`). In the second step, all libraries are analyzed together in the classification mode (`-steps class`).

#### Example of a High-Sensitivity Strategy:

```bash
for file in $(ls *fasta); 
do 
python path_to_phasis/phasis.py -mindepth 1 -phase 24 -libformat F -classifier KNN -libs $file -reference genome.fa -cores 0 -maxhits 2000 -steps cfind -uniqueRatioCut 0.05;
done
```

Followed by:

```bash
python path_to_phasis/phasis.py -mindepth 1 -phase 24 -libformat F -classifier KNN -libs *fasta -reference genome.fa -cores 0 -maxhits 2000 -steps class -class_cluster_file *24-PHAS.candidate.clusters -uniqueRatioCut 0.05
```

With this strategy, Phasis will evaluate all small RNA clusters enriched for 24-nt small RNA reads across all processed libraries, providing increased sensitivity for PHAS loci detection. However, using a `uniqueRatioCut` of 0 will likely increase the computational resource usage to a prohibitive level and might not significantly enhance sensitivity in plant genomes with many transposable elements. It is advisable to use a `uniqueRatioCut` between 0.05 and 0.2 to balance computational efficiency while still filtering out non-unique mappings, ensuring optimal sensitivity and precision.

### -force Argument

This option should only be used with caution, as bypassing these safeguards can result in:

- Excessive memory usage
- Prolonged runtimes
- Potential system crashes or job termination on servers without sufficient resources
- By default, Phasis includes checks for combinations of parameters such as multiple libraries, high values for `-maxhits` (e.g., >100), or low `-mindepth` values (e.g., 1) with `-uniqueRatioCut < 0.05`, which could strain system resources. Without the `-force` option, Phasis will stop execution and alert the user when these potentially dangerous configurations are detected.

If you are confident your system can handle these intensive configurations and wish to proceed, you can invoke the `-force` flag to force Phasis to run.

#### Command Example:

```bash
Copy code
phasis.py -libs library1 library2 ... -maxhits 200 -uniqueRatioCut 0.01 -mindepth 1 -force 
```

#### Important:
Use this option only if you are aware of the computational demands of your parameter configuration. Always monitor system resources to ensure that the process doesn’t exhaust memory or CPU limits, especially when dealing with large sRNAseq datasets.

### Input Library Formats and Preprocessing for Phasis

Phasis accepts two input formats for small RNA libraries: FASTA and Tag Count format. It is necessary that small RNA libraries are quality controlled prior to running Phasis. Uncompressed FASTQ files can be converted to FASTA format using the script available at `path_to_phasis/support_scripts/fastq_to_fasta.sh`. This script takes a FASTQ file and outputs a corresponding FASTA file with the same name but with the .fasta extension.

Example for converting FASTQ to FASTA:

```bash
for file in $(ls *fq);
do
echo $file
path_to_phasis/support_scripts/fastq_to_fasta.sh $file &
done
```

If storing uncompressed FASTA files requires too much space or if the files contain abundant reads with "N" characters, it is recommended to further convert the FASTA files into Tag Count format. Tag Count is a lightweight, tab-delimited text format where identical read sequences are collapsed in the first column and their counts are recorded in the second column. Sequences containing "N" characters are also removed during this process.

To convert FASTA files to Tag Count format, you can run the following:

```bash
for file in $(ls *fasta);
do
python path_to_phasis/support_scripts/fastaToTag.py $file
done
```

Phasis can directly process Tag Count files by specifying the `-libformat T` argument in the command:

```bash
python path_to_phasis/phasis.py -libs [LIBS ...] -reference genome.fa -libformat T
```

### Handling Non-Integer Chromosome Headers in Phasis

Phasis uses FASTA headers as keys for identifying and mapping phased loci. If the chromosome headers are strings, such as `">chromosome_1A"`, it will significantly increase RAM usage and may lead to errors during execution. For Phasis to work properly, the chromosome headers must be named as incremental integers (e.g., `>1`, `>2`, `>3`). 

To resolve this issue, we provide the script `support_scripts/replace_genome_headers.py`, which generates a new genome FASTA file with integer-based chromosome headers. It also outputs an equivalence table between the original headers and the new integer-based ones. 

This script ensures that Phasis runs efficiently, avoiding excessive memory usage and preventing errors caused by non-integer chromosome IDs.

#### Example Usage:
```bash
python path_to_phasis/support_scripts/replace_genome_headers.py genome.fa new_genome.fa equivalence.tsv
```

This command will replace the chromosome headers in `genome.fa` with integers, save the new genome in `new_genome.fa`, and generate an equivalence table `equivalence.tsv`. The table contains two columns:
- `Old_ID`: Original chromosome header.
- `New_ID`: New integer-based header.

This allows you to use the modified genome in Phasis while maintaining a record of the changes.

### Instructions for *PHAS* Loci Comparison Using the phasMatch Script

If you need to compare *PHAS* loci between different runs, samples, or PHAS-prediction software, the following script helps match and summarize overlapping *PHAS* loci between datasets.

The phasMatch script is located at:
```bash
python path_to_phasis/support_scripts/phasMatch.v03.py
```

#### Steps to Run the phasMatch Script:

1. **Prepare Input Files**: 
   - **PHASIS Result File**: The first input is the PHASIS output file (with the extension `.phasis.result.tsv`).
   - **Alternative Prediction File**: The second input is a formated prediction file or a result from another PHAS-prediction software.

2. **Run the Script**: Use the following command in your terminal to run the script:
   ```bash
   python path_to_phasis/support_scripts/phasMatch.v03.py <phasis.result.tsv> <alternative_file>
   ```

3. **Output**:
   - A file named `matched_<timestamp>.txt` will contain details of overlapping *PHAS* loci between the two datasets.
   - A summary file `summary_<timestamp>.txt` will give an overview of how many loci were matched and unmatched.

#### Summary of phasMatch Script Behavior:

- **phasis.result.tsv**: This is the first argument, which must be the PHASIS output file.
- **alternative file**: The second argument is the alternative PHAS-prediction file for comparison.
- **Flanking region**: The script uses a default flanking size of 300 nt to account for small positional variations in loci detection.

Here’s a example of the expected format for the alternative predictions file (`alternative_predictions.tsv`) based on the column structure mentioned. Each line in the file should follow the structure:

```
ID    Chromosome    start    end    pval    trigger
```

#### Example `alternative_predictions.tsv` file:

```
PHASLocus_1    1    105000    106000    0.01    Trigger1
PHASLocus_2    3    50000    51000    0.05    Trigger2
PHASLocus_3    5    200000    201000    0.001    Trigger3
PHASLocus_4    7    75000    76000    0.02    Trigger4
```

#### Columns Explanation:
- **ID**: Unique identifier for each PHAS locus (e.g., `PHASLocus_1`, `PHASLocus_2`).
- **Chromosome**: Chromosome number or identifier (e.g., `1`, `3`, `5`, `7`).
- **start**: Start position of the PHAS locus (e.g., `105000`, `50000`).
- **end**: End position of the PHAS locus (e.g., `106000`, `51000`).
- **pval**: p-value or score indicating the statistical significance (e.g., `0.01`, `0.05`).
- **trigger**: miRNA triger associated feature (e.g., `Trigger1`, `Trigger2`). If unknown, set the value to NONE.

#### Format and Example for Comparison

- **phasis.result.tsv**: 
   Example:
   ```
   PHASLocus_1    1    105000    106000    300    Trigger1
   ```

- **alternative_predictions.tsv**:
   Example:
   ```
   PHASLocus_1    1    105000    106000    0.01    Trigger1
   ```

The script will match the entries between the two files based on the chromosome, start, and end positions, while accounting for a flanking region of ±300 nt to detect overlaps.

### Authors:

Atul kakrana
kakrana@gmail.com 

Thales Cherubino
thalescherubino@gmail.com