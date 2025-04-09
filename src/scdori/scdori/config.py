#################################
# config.py
#################################
"""
Global configuration for the scDoRI modeling pipeline.

This file defines top-level constants and parameters controlling:

1. Logging
2. File paths for data and outputs
3. Model architecture details (numbers of topics, hidden dimensions)
4. Training phases and hyperparameters 
5. Loss weighting for different data modalities (ATAC, TF, RNA)
6. Regularization and early stopping settings
7. Significance testing cutoffs for TF-gene links
8. UMAP parameters for visualization

Attributes
----------
logging_level : int
    The Python logging level (e.g., logging.INFO).
data_dir : Path
    Base directory containing the processed anndata and other precomputed files.
output_subdir : str
    Subdirectory name within data_dir for storing or accessing outputs.
rna_metacell_file : str
    Name of the H5AD file containing RNA data.
atac_metacell_file : str
    Name of the H5AD file containing ATAC data.
batch_col : str
    Key for the batch column in the AnnData object.
gene_peak_distance_file : str
    Filename for the NumPy array containing gene-peak distances.
insilico_chipseq_act_file : str
    Filename for the in silico ChIP-seq activator embeddings.
insilico_chipseq_rep_file : str
    Filename for the in silico ChIP-seq repressor embeddings.
random_seed : int
    Random seed for reproducibility.
batch_size_cell : int
    Batch size for training.
dim_encoder1 : int
    Dimension of the first encoder layer.
dim_encoder2 : int
    Dimension of the second encoder layer.
num_topics : int
    Number of latent topics for scDoRI.
batch_size_cell_prediction : int
    Batch size when making predictions in eval mode (e.g., forward passes only).
epoch_warmup_1 : int
    Number of epochs to run "warmup_1" (ATAC+TF) before adding RNA.
max_scdori_epochs : int
    Maximum number of epochs for the scDoRI phase 1 training (module 1,2,3).
max_grn_epochs : int
    Maximum number of epochs for the GRN training phase (module 4).
update_encoder_in_grn : bool
    Whether to unfreeze the encoder during the GRN phase.
update_peak_gene_in_grn : bool
    Whether to unfreeze the peak-gene links in the GRN phase.
update_topic_peak_in_grn : bool
    Whether to unfreeze the topic-peak links in the GRN phase.
update_topic_tf_in_grn : bool
    Whether to unfreeze the topic-TF links in the GRN phase.
eval_frequency : int
    How often (in epochs) to evaluate validation loss.
phase1_patience : int
    Early stopping patience (in epochs) for phase 1 training (module 1,2 3).
grn_val_patience : int
    Early stopping patience (in epochs) for the GRN phase.
learning_rate_scdori : float
    Learning rate for scDoRI phase 1 training (module 1,2 3).
learning_rate_grn : float
    Learning rate for the GRN training phase.
weight_atac_phase1 : float
    Loss weight for ATAC reconstruction in Phase 1:warmup_1.
weight_tf_phase1 : float
    Loss weight for TF reconstruction in Phase 1:warmup_1.
weight_rna_phase1 : float
    Loss weight for RNA reconstruction in Phase 1:warmup_1. set to 0.
weight_rna_grn_phase1 : float
    Loss weight for GRN-based RNA reconstruction in Phase 1:warmup_1. set to 0.
weight_atac_phase2 : float
    Loss weight for ATAC reconstruction in Phase 1:warmup_2.
weight_tf_phase2 : float
    Loss weight for TF reconstruction in Phase 1:warmup_2.
weight_rna_phase2 : float
    Loss weight for RNA reconstruction in Phase 1:warmup_2.
weight_rna_grn_phase2 : float
    Loss weight for GRN-based RNA reconstruction in Phase 1:warmup_2. set to 0.
weight_atac_grn : float
    Loss weight for ATAC reconstruction in the GRN phase.
weight_tf_grn : float
    Loss weight for TF reconstruction in the GRN phase.
weight_rna_grn : float
    Loss weight for RNA reconstruction in the GRN phase.
weight_rna_from_grn : float
    Loss weight for the GRN-based RNA branch in the GRN phase.
l1_penalty_topic_tf : float
    L1 regularization coefficient on the topic_tf_decoder.
l2_penalty_topic_tf : float
    L2 regularization coefficient on the topic_tf_decoder.
l1_penalty_topic_peak : float
    L1 regularization coefficient on the topic_peak_decoder.
l2_penalty_topic_peak : float
    L2 regularization coefficient on the topic_peak_decoder.
l1_penalty_gene_peak : float
    L1 regularization coefficient on the gene_peak_factor_learnt.
l2_penalty_gene_peak : float
    L2 regularization coefficient on the gene_peak_factor_learnt.
l1_penalty_grn_activator : float
    L1 regularization on GRN activator parameters (tf_gene_topic_activator_grn).
l1_penalty_grn_repressor : float
    L1 regularization on GRN repressor parameters (tf_gene_topic_repressor_grn).
tf_expression_mode : str
    Either "True" (use actual TF expression) or "latent" (model's predicted TF expression).
tf_expression_clamp : float
    Clamping threshold for TF expression values in [0, 1].
cells_per_topic : int
    Number of cells sampled per topic to compute topic-level TF expression.
weights_folder_scdori : str
    Folder to save model weights after the scDoRI Phase 1.
weights_folder_grn : str
    Folder to save model weights after the GRN phase.
best_scdori_model_path : str
    Filename for saving the best scDoRI model (Phase 1).
best_grn_model_path : str
    Filename for saving the best GRN model.
umap_n_neighbors : int
    Number of neighbors for UMAP.
umap_min_dist : float
    Min dist parameter for UMAP.
umap_random_state : int
    Random seed for UMAP.
significance_cutoffs : list of float
    List of thresholds for empirical p-value cutoffs in TF-gene link permutation tests.
num_permutations : int
    Number of permutations used to compute TF-gene link significance.
"""

import logging
from pathlib import Path

# LOGGING
logging_level = logging.INFO

# DATA PATHS
data_dir = Path("/data/saraswat/new_metacells/data_gastrulation_single_cell")
output_subdir = "generated"

rna_metacell_file = "rna_processed.h5ad"
atac_metacell_file = "atac_processed.h5ad"
batch_col = "sample"
gene_peak_distance_file = "gene_peak_distance_exp.npy"
insilico_chipseq_act_file = "insilico_chipseq_act.npy"
insilico_chipseq_rep_file = "insilico_chipseq_rep.npy"

# RANDOM SEED
random_seed = 200

# BATCH / ARCHITECTURE
batch_size_cell = 128
dim_encoder1 = 500
dim_encoder2 = 200
num_topics = 40
batch_size_cell_prediction = 512

# PHASE1
epoch_warmup_1 = 5
max_scdori_epochs = 1000

# PHASE 2
max_grn_epochs = 1000
update_encoder_in_grn = False
update_peak_gene_in_grn = False
update_topic_peak_in_grn = False
update_topic_tf_in_grn = False

# early stopping and evaluation
eval_frequency = 1
phase1_patience = 50
grn_val_patience = 5

# LR / LOSSES
learning_rate_scdori = 0.005
learning_rate_grn = 0.001

# Phase 1 weights (warmup_1)
weight_atac_phase1 = 1.0
weight_tf_phase1   = 200.0
weight_rna_phase1  = 0.0
weight_rna_grn_phase1 = 0.0

# Phase 1 weights (warmup_2)
weight_atac_phase2 = 1.0
weight_tf_phase2   = 200.0
weight_rna_phase2  = 20.0
weight_rna_grn_phase2 = 0.0

# Phase 2 (GRN) weights
weight_atac_grn = 1.0
weight_tf_grn   = 200.0
weight_rna_grn  = 20.0
weight_rna_from_grn = 20.0

# REGULARIZATION
l1_penalty_topic_tf     = 0.001
l2_penalty_topic_tf     = 0.000
l1_penalty_topic_peak   = 0.001
l2_penalty_topic_peak   = 0.001
l1_penalty_gene_peak    = 0.001
l2_penalty_gene_peak    = 0.005
l1_penalty_grn_activator = 0.00005
l1_penalty_grn_repressor = 0.0000

# TF EXPRESSION SETTINGS
tf_expression_mode = "True"
tf_expression_clamp = 0.1
cells_per_topic = 200

# SAVE FOLDERS
weights_folder_scdori = "weights_directory_scdori"
weights_folder_grn    = "weights_directory_grn"
best_scdori_model_path = "best_scdori_final.pth"
best_grn_model_path    = "best_grn.pth"

# UMAP PARAMETERS
umap_n_neighbors   = 15
umap_min_dist      = 0.1
umap_random_state  = 42

# SIGNIFICANCE SETTINGS
significance_cutoffs = [0.001, 0.005, 0.01, 0.05]
num_permutations=1000
