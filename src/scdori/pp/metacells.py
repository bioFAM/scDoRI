import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import logging
import scipy.sparse as sp

logger = logging.getLogger(__name__)

def create_metacells(
    data_rna: ad.AnnData,
    data_atac: ad.AnnData,
    grouping_key: str = "leiden",
    resolution: float = 5.0,
    batch_key: str = "sample"
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Create metacell-level RNA and ATAC AnnData objects by clustering cells and computing
    mean values per cluster.
    This function:
    1. Normalizes and logs the RNA data, then runs PCA.
    2. Uses Harmony integration for batch correction on the PCA embeddings.
    3. Clusters the RNA data with Leiden at the specified resolution, storing the
        cluster labels in ``data_rna.obs[grouping_key]``.
    4. Summarizes RNA expression and ATAC accessibility for each cluster by taking 
        the mean of each feature across all cells in that cluster.

    Parameters
    ----------
    data_rna : anndata.AnnData
        RNA single-cell data. A layer "counts" is added and re-assigned later.
        The shape is (n_cells, n_genes).
    data_atac : anndata.AnnData
        ATAC single-cell data with the same set or superset of cell IDs in
        `data_rna.obs_names`. The shape is (n_cells, n_peaks).
    grouping_key : str, optional
        The key in `data_rna.obs` where the Leiden cluster labels will be stored.
        Default is "leiden".
    resolution : float, optional
        The resolution parameter for Leiden clustering. Higher values yield more clusters.
        Default is 5.0.
    batch_key : str, optional
        The column in `data_rna.obs` indicating batch information for Harmony integration.
        Default is "sample".

    Returns
    -------
    (rna_metacell, atac_metacell) : tuple of anndata.AnnData
        - rna_metacell : shape (#clusters, n_genes)
        - atac_metacell : shape (#clusters, n_peaks) 
        The `.obs` index is set to the cluster labels, and the `.var` is inherited from the original `data_rna`/`data_atac`.

    Notes
    -----
    - The function uses `scanpy.external.pp.harmony_integrate` for batch integration on the PCA representation stored in "X_pca_harmony".
    - The ATAC data is transformed by `(atac_vals + 1) // 2` to interpret insertions as fragment presence, following Martens et al. (2023).
    - Mean values are computed across cells in each cluster for both RNA and ATAC.
    - The original `data_rna.X` is restored to raw counts at the end.
    """
    import scanpy.external as sce

    logger.info(f"Creating metacells with resolution={resolution} (grouping key={grouping_key}).")
    # Keep original counts in a layer
    data_rna.layers["counts"] = data_rna.X.copy()

    # Normalize & run PCA
    sc.pp.normalize_total(data_rna)
    sc.pp.log1p(data_rna)
    sc.pp.pca(data_rna)

    # Harmony integration
    sce.pp.harmony_integrate(data_rna, batch_key)

    sc.pp.neighbors(data_rna, use_rep="X_pca_harmony")
    sc.tl.leiden(data_rna, resolution=resolution, key_added=grouping_key)

    # Summarize by cluster
    clusters = data_rna.obs[grouping_key].unique()
    cluster_groups = data_rna.obs.groupby(grouping_key)

    mean_rna_list = []
    mean_atac_list = []
    cluster_names = []

    for cluster_name in clusters:
        cell_idx = cluster_groups.get_group(cluster_name).index

        # RNA
        rna_vals = data_rna[cell_idx].X
        if sp.issparse(rna_vals):
            rna_vals = rna_vals.toarray()
        mean_rna = np.array(rna_vals.mean(axis=0)).ravel()
        mean_rna_list.append(mean_rna)

        # ATAC
        if len(set(cell_idx).intersection(data_atac.obs_names)) == 0:
            mean_atac_list.append(np.zeros(data_atac.shape[1]))
        else:
            atac_vals = data_atac[cell_idx].X
            if sp.issparse(atac_vals):
                atac_vals = atac_vals.toarray()
            # Convert insertions to fragments
            atac_bin = (atac_vals + 1) // 2
            mean_atac = np.array(atac_bin.mean(axis=0)).ravel()
            mean_atac_list.append(mean_atac)

        cluster_names.append(cluster_name)

    # Build new AnnData for metacells
    mean_rna_arr = np.vstack(mean_rna_list)
    mean_atac_arr = np.vstack(mean_atac_list)

    obs_df = pd.DataFrame({grouping_key: cluster_names}).set_index(grouping_key)

    rna_metacell = ad.AnnData(X=mean_rna_arr, obs=obs_df, var=data_rna.var)
    atac_metacell = ad.AnnData(X=mean_atac_arr, obs=obs_df, var=data_atac.var)

    # Restore original raw counts
    data_rna.X = data_rna.layers["counts"].copy()

    logger.info(f"Metacell shapes: RNA={rna_metacell.shape}, ATAC={atac_metacell.shape}")
    return rna_metacell, atac_metacell
