import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import anndata as ad
import scanpy
import scipy.sparse as sp

def create_minibatch(
    device,
    index_matrix,
    rna_anndata,
    atac_anndata,
    num_cells,
    tf_indices,
    encoding_batch_onehot
):
    """
    Create a minibatch of required input tensors using integer indices of cells.

    Parameters
    ----------
    device : torch.device
        The device (CPU or CUDA) to which the data should be moved.
    index_matrix : torch.Tensor
        A 1D tensor containing integer indices of the cells in the minibatch.
    rna_anndata : anndata.AnnData
        AnnData object for RNA data. The .X matrix should contain RNA counts or expression values.
    atac_anndata : anndata.AnnData
        AnnData object for ATAC data. The .X matrix should contain accessibility counts.
    num_cells : np.ndarray
        A NumPy array (N x 1) indicating the number of cells represented by each row (if using metacells).
        For single-cell level data, this may be an array of ones.
    tf_indices : np.ndarray
        Indices corresponding to transcription factors (TFs) in the RNA AnnData.
    encoding_batch_onehot : np.ndarray
        A one-hot encoded matrix representing batch information for each cell (cells x num_batches).

    Returns
    -------
    tuple
        A tuple containing:
        - input_matrix (torch.Tensor): Concatenated RNA and ATAC input of shape (B, g + p),
          where B is batch size, g is the number of genes, p is the number of peaks. 
          Values are floats on the given device.
        - tf_exp (torch.Tensor): RNA expression values for TFs, shape (B, num_tfs).
        - library_size_value (torch.Tensor): Log-scale library sizes for RNA and ATAC, shape (B, 2).
        - num_cells_value (torch.Tensor): Number of cells per row in the minibatch (B, 1).
        - input_batch (torch.Tensor): One-hot batch-encoding, shape (B, num_batches).

    Notes
    -----
    - This function converts sparse arrays to dense if necessary.
    - ATAC counts are converted from insertion counts to fragment counts by using (x + 1) // 2.
    """
    
    index_train = index_matrix.clone().detach().cpu().numpy()
    atac_input  = atac_anndata[index_train,:].X
    rna_input   = rna_anndata[index_train,:].X

    if sp.issparse(atac_input):
        atac_input = atac_input.toarray()
    if sp.issparse(rna_input):
        rna_input = rna_input.toarray()

    # Convert ATAC insertions to fragment counts
    atac_input = (np.array(atac_input) + 1) // 2
    rna_input  = np.array(rna_input)

    library_size_atac = atac_input.sum(axis=1).reshape(-1, 1) + 1e-8
    library_size_rna  = rna_input.sum(axis=1).reshape(-1, 1) + 1e-8
    library_size      = np.concatenate(
        (np.log(library_size_rna), np.log(library_size_atac)), axis=1
    )

    input_data   = np.concatenate((rna_input, atac_input), axis=1)
    input_matrix = torch.from_numpy(input_data).to(device, dtype=torch.float)

    input_batch = torch.from_numpy(
        encoding_batch_onehot[index_train, :]
    ).to(device, dtype=torch.float)

    tf_exp = torch.from_numpy(rna_input[:, tf_indices]).to(device, dtype=torch.float)

    library_size_value = torch.from_numpy(library_size).to(device, dtype=torch.float)
    num_cells_value    = torch.from_numpy(num_cells[index_train, :]).to(device, dtype=torch.float)

    return (
        input_matrix,
        tf_exp,
        library_size_value,
        num_cells_value,
        input_batch
    )
