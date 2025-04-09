#################################
# evaluation.py
#################################
import torch
import numpy as np
from tqdm import tqdm
from scdori.dataloader import create_minibatch

def get_latent_topics(
    model,
    device,
    data_loader,
    rna_anndata,
    atac_anndata,
    num_cells,
    tf_indices,
    encoding_batch_onehot
):
    """
    Extract the softmaxed topic activations (theta) for each cell in the dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The scDoRI model containing an encoder for generating topic distributions.
    device : torch.device
        The PyTorch device (e.g., 'cpu' or 'cuda') used for computations.
    data_loader : torch.utils.data.DataLoader
        A DataLoader that yields batches of cell indices.
    rna_anndata : anndata.AnnData
        RNA single-cell data in AnnData format.
    atac_anndata : anndata.AnnData
        ATAC single-cell data in AnnData format.
    num_cells : np.ndarray
        Number of cells in each row (e.g., if using metacells). set to ones for single-cell data.
    tf_indices : np.ndarray
        Indices of transcription factor genes in the RNA data.
    encoding_batch_onehot : np.ndarray
        One-hot encoding of batch information (cells x num_batches).

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_cells, n_topics) representing the softmaxed
        topic activations for each cell in the order given by the DataLoader.
    """
    model.eval()
    all_thetas = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Extracting latent topics"):
            cell_indices = batch_data[0].to(device)
            B = cell_indices.shape[0]

            (input_matrix, tf_exp, library_size_value, num_cells_value,
             input_batch) = create_minibatch(
                 device,
                 cell_indices,
                 rna_anndata,
                 atac_anndata,
                 num_cells,
                 tf_indices,
                 encoding_batch_onehot
             )

            # Split into rna_input, atac_input, tf_input
            rna_input = input_matrix[:, :model.num_genes]
            atac_input = input_matrix[:, model.num_genes:]
            tf_input = tf_exp

            log_lib_rna = library_size_value[:, 0].reshape(-1, 1)
            log_lib_atac = library_size_value[:, 1].reshape(-1, 1)
            batch_onehot = input_batch

            # We only need an encoder pass => e.g. use phase="warmup_1"
            out = model(
                rna_input,
                atac_input,
                tf_input,
                tf_input,
                log_lib_rna,
                log_lib_atac,
                num_cells_value,
                batch_onehot,
                phase="warmup_1"
            )

            # out["theta"] contains the softmaxed topic distribution
            theta = out["theta"].detach().cpu().numpy()
            all_thetas.append(theta)

    latent_topics = np.concatenate(all_thetas, axis=0)
    return latent_topics
