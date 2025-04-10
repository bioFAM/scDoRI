import scanpy as sc
import torch
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_scdori_inputs(config_file):
    """
    Load RNA & ATAC data (.h5ad files), plus gene-peak distance and in silico chip-seq matrix.

    Parameters
    ----------
    config_file : object
        A configuration file containing the attributes:
        - data_dir : pathlib.Path
            The base directory for input data.
        - output_subdir : str
            The subdirectory where output files are located.
        - rna_metacell_file : str
            The filename for the RNA data (single cell or metacell) (H5AD).
        - atac_metacell_file : str
            The filename for the ATAC data (single cell or metacell) (H5AD).
        - gene_peak_distance_file : str
            The filename for the NumPy array with gene-peak distance matrix.
        - insilico_chipseq_act_file : str
            The filename for the in silico ChIP-seq activator matrix.
        - insilico_chipseq_rep_file : str
            The filename for the in silico ChIP-seq repressor matrix.

    Returns
    -------
    tuple
        A tuple containing:
        rna_metacell : anndata.AnnData
            RNA data loaded from H5AD.
        atac_metacell : anndata.AnnData
            ATAC data loaded from H5AD.
        gene_peak_dist : torch.Tensor
            A tensor of shape (num_genes, num_peaks) representing gene-peak distances.
        insilico_act : torch.Tensor
            A tensor of shape (num_peaks, num_motifs) for in silico ChIP-seq (activator) embeddings.
        insilico_rep : torch.Tensor
            A tensor of shape (num_peaks, num_motifs) for in silico ChIP-seq (repressor) embeddings.
    """
    out_dir = config_file.data_dir / config_file.output_subdir

    rna_path = out_dir / config_file.rna_metacell_file
    atac_path = out_dir / config_file.atac_metacell_file
    dist_path = out_dir / config_file.gene_peak_distance_file
    act_path = out_dir / config_file.insilico_chipseq_act_file
    rep_path = out_dir / config_file.insilico_chipseq_rep_file

    logger.info(f"Loading RNA from {rna_path}")
    rna_metacell = sc.read_h5ad(rna_path)

    logger.info(f"Loading ATAC from {atac_path}")
    atac_metacell = sc.read_h5ad(atac_path)

    logger.info(f"Loading gene-peak dist from {dist_path}")
    gene_peak_dist = torch.from_numpy(np.load(dist_path))

    logger.info(f"Loading insilico embeddings from {act_path} & {rep_path}")
    insilico_act = torch.from_numpy(np.load(act_path))
    insilico_rep = torch.from_numpy(np.load(rep_path))

    return rna_metacell, atac_metacell, gene_peak_dist, insilico_act, insilico_rep


def save_model_weights(model, path: Path, tag: str):
    """
    Save model weights to a specified path with a given tag.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model whose state_dict is to be saved.
    path : pathlib.Path
        The directory path where the weights file will be saved.
    tag : str
        An identifier to include in the saved filename (e.g., "best_eval").

    Returns
    -------
    None
    """
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"best_{tag}.pth"
    torch.save(model.state_dict(), file_path)
    logger.info(f"Saved model weights => {file_path}")
