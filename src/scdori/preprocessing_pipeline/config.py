import logging
import numpy as np

"""
Global configuration module for the single-cell multiome data pre-processing used in scDoRI.

This module holds constants and parameters used across the pipeline, including:
1. Logging settings
2. Paths for data, genome references, and motif databases
3. Species and genome assembly details
4. HVG/peak selection parameters
5. Promoter logic
6. In-silico ChIP-seq correlation settings
7. Other hyperparameters or default values

Variables
---------
data_dir : str
    Directory where RNA and ATAC AnnData files are stored.
genome_dir : str
    Directory containing genome FASTA and chromosome size files.
motif_directory : str
    Directory containing .meme motif files for TF databases.
output_subdir_name : str
    Name of the subdirectory within data_dir to store intermediate outputs.
rna_adata_file_name : str
    Filename for the RNA AnnData file (H5AD).
atac_adata_file_name : str
    Filename for the ATAC AnnData file (H5AD).
species : str
    Species name, e.g., "mouse" or "human".
genome_assembly : str
    Genome assembly version, e.g., "mm10" or "hg38".
gtf_url : str or None
    URL to the GTF file override; if None, defaults are used based on species and assembly.
chrom_sizes_url : str or None
    URL to the chromosome sizes file override; if None, defaults are used if known.
fasta_url : str or None
    URL to the FASTA genome file override; if None, defaults are used if known.
chrom_sizes_file : str
    Local filename for the chromosome sizes, e.g., "mm10.chrom.sizes" or "hg38.chrom.sizes".
mitochondrial_prefix : str
    Prefix used to identify mitochondrial genes in the RNA AnnData.
genes_user : list of str
    User-provided genes always included in the final model, even if not Highly variable (HV).
tfs_user : list of str
    User-provided TFs always included in the final model, even if not HV.
motif_database : str
    Name of the motif database, e.g., "cisbp".
num_genes : int
    Number of genes to select for scDoRI training (via HVG + user overrides).
num_tfs : int
    Number of TFs to select for scDoRI training (via HV among potential TFs + user overrides).
min_cells_per_gene : int
    Minimum cell count threshold for gene detection (not enforced in current code).
window_size : int
    Genomic window (bp) around each gene for selecting peaks. Example: 80,000 => ±80 kb.
num_peaks : int
    Target number of peaks for training scDoRI; some are forced (promoters), the rest are HV.
peak_std_batch_key : str
    Key in `.obs` used to group cells and measure peak standard deviation for HV selection.
batch_key : str
    Key in `.obs` denoting experimental batch/covariate for integration or Harmony correction.
leiden_resolution : float
    Resolution parameter for the Leiden clustering used to create metacells.
keep_promoter_peaks : bool
    Whether to keep promoter peaks unconditionally in the final set of peaks.
promoter_col : str
    Column in ATAC `.var` indicating if a peak is a promoter peak (True/False).
motif_match_pvalue_threshold : float
    P-value threshold for motif hits in FIMO for in-silico ChIP-seq.
correlation_percentile : float
    Percentile cutoff for correlation significance with TF expression (e.g., 99 => p≈0.01).
n_bg_peaks_for_corr : int
    Number of peaks (lowest motif scores) used per TF as background in correlation tests.
peak_distance_scaling_factor : float
    Decay factor for exponential distance weighting in initial gene-peak links.
peak_distance_min_cutoff : float
    Minimum allowed scaled distance (exponential) threshold within the user-defined window.
"""

# Logging
logging_level = logging.INFO

# Directory structure
data_dir = "/data/saraswat/new_metacells/data_gastrulation_single_cell"
genome_dir = "/data/saraswat/new_metacells/mouse_genome_files"
motif_directory = "/data/saraswat/new_metacells/motif_database"
output_subdir_name = "generated"

# Input Filenames
rna_adata_file_name = "anndata.h5ad"
atac_adata_file_name = "PeakMatrix_anndata.h5ad"

# Species & references
species = "mouse"
genome_assembly = "mm10"

# Optional user-provided URLs for genome files (None => use defaults if known)
gtf_url = None
chrom_sizes_url = None
fasta_url = None

chrom_sizes_file = "mm10.chrom.sizes"

# Genes & TF selection
mitochondrial_prefix = "mt-"

genes_user = ["Myt1l"]
tfs_user = ["Tbx5", "Myt1l", "Cdx2", "Sox2"]
motif_database = "cisbp"

num_genes = 4000
num_tfs = 300
min_cells_per_gene = 4

# Genomic window
window_size = 80000

num_peaks = 90000
peak_std_batch_key = "leiden"

# Batch key & metacell parameters
batch_key = "sample"
leiden_resolution = 10

# Promoter logic
keep_promoter_peaks = True
promoter_col = "promoter_col"

# Correlation & in-silico ChIP-seq
motif_match_pvalue_threshold = 1e-3
correlation_percentile = 99
n_bg_peaks_for_corr = 5000

# Distance matrix parameters
peak_distance_scaling_factor = 20000
peak_distance_min_cutoff = np.e ** (-1 * (window_size / peak_distance_scaling_factor))
