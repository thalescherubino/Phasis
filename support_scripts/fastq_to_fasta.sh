#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input_fastq_file"
    exit 1
fi

# Input FASTQ file
input_fastq=$1

# Check if the file has the correct extension (.fq or .fastq)
if [[ $input_fastq == *.fq ]]; then
    output_fasta="${input_fastq%.fq}.fasta"
elif [[ $input_fastq == *.fastq ]]; then
    output_fasta="${input_fastq%.fastq}.fasta"
else
    echo "Error: Input file must have a .fq or .fastq extension."
    exit 1
fi

# Convert FASTQ to FASTA
awk 'NR%4==1 {print ">"substr($0,2)} NR%4==2 {print $0}' $input_fastq > $output_fasta

echo "Conversion complete. Output written to $output_fasta"

