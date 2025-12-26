import sys

def replace_fasta_headers(fasta_file, output_fasta, equivalence_file):
    with open(fasta_file, 'r') as infile, open(output_fasta, 'w') as outfile, open(equivalence_file, 'w') as equiv_file:
        current_chr = 1  # Start numbering chromosomes from 1
        equiv_file.write("Old_ID\tNew_ID\n")  # Write header for the equivalence file
        
        for line in infile:
            if line.startswith('>'):
                old_header = line.strip()[1:]  # Remove '>' and get the header
                new_header = str(current_chr)  # Create new header with the integer
                outfile.write(f">{new_header}\n")  # Write the new header to the output FASTA
                equiv_file.write(f"{old_header}\t{new_header}\n")  # Save the equivalence
                current_chr += 1  # Increment the chromosome number
            else:
                outfile.write(line)  # Write the sequence as is

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python replace_fasta_headers.py <input_fasta> <output_fasta> <equivalence_file>")
        sys.exit(1)
    
    input_fasta = sys.argv[1]
    output_fasta = sys.argv[2]
    equivalence_file = sys.argv[3]

    replace_fasta_headers(input_fasta, output_fasta, equivalence_file)
