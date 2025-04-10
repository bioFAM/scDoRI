import numpy as np
import anndata as ad
import logging
import scipy.sparse as sp

logger = logging.getLogger(__name__)

def select_highly_variable_peaks_by_std(
    data_atac: ad.AnnData,
    n_top_peaks: int,
    cluster_key: str = "leiden"
) -> ad.AnnData:
    """
    Select highly variable peaks based on the standard deviation of peak accessibility
    across clusters.

    This function:
    1. Groups cells by `cluster_key` in `data_atac.obs`.
    2. Computes ATAC fragment counts and calculates mean accessibility
        per peak for each cluster.
    3. Computes the standard deviation of each peak's accessibility across clusters.
    4. Selects the top `n_top_peaks` peaks with the highest standard deviation.

    Parameters
    ----------
    data_atac : anndata.AnnData
        An AnnData containing ATAC data. Expects `.obs[cluster_key]` to exist.
    n_top_peaks : int
        Number of peaks to retain based on the highest standard deviation across clusters.
    cluster_key : str, optional
        The column in `data_atac.obs` specifying cluster labels. Defaults to "leiden".

    Returns
    -------
    anndata.AnnData
        A subset of `data_atac` containing only the selected peaks.
        If `n_top_peaks >= data_atac.shape[1]`, returns the original data.

    Notes
    -----
    - If `cluster_key` is missing, a warning is logged and the original AnnData is returned.
    - Transformation `(X + 1) // 2` to interpret insertions as fragment counts.
    """
    if cluster_key not in data_atac.obs.columns:
        logger.warning(f"{cluster_key} not found in data_atac.obs; skipping peak selection.")
        return data_atac

    clusters = data_atac.obs[cluster_key].unique()
    cluster_groups = data_atac.obs.groupby(cluster_key)
    mean_list = []

    for c_label in clusters:
        idx_cells = cluster_groups.get_group(c_label).index
        mat = data_atac[idx_cells].X
        if sp.issparse(mat):
            mat = mat.toarray()
        mat = (mat + 1) // 2  # get fragment presence
        mean_vec = mat.mean(axis=0).A1 if hasattr(mat, "A1") else mat.mean(axis=0)
        mean_list.append(mean_vec)

    # shape => (n_clusters, n_peaks)
    cluster_matrix = np.vstack(mean_list)
    stdev_peaks = cluster_matrix.std(axis=0)
    data_atac.var["std_cluster"] = stdev_peaks

    if n_top_peaks < data_atac.shape[1]:
        sorted_idx = np.argsort(stdev_peaks)[::-1]
        keep_idx = sorted_idx[:n_top_peaks]
        mask = np.zeros(data_atac.shape[1], dtype=bool)
        mask[keep_idx] = True
        data_atac_sub = data_atac[:, mask].copy()
        logger.info(f"Selected top {n_top_peaks} variable peaks (by std across {cluster_key}).")
        return data_atac_sub
    else:
        logger.info("n_top_peaks >= total peaks; no filtering applied.")
        return data_atac

def keep_promoters_and_select_hv_peaks(
    data_atac: ad.AnnData,
    total_n_peaks: int,
    cluster_key: str = "leiden",
    promoter_col: str = "is_promoter"
) -> ad.AnnData:
    """
    Retain all promoter peaks and then select highly variable (HV) peaks among non-promoters.

    Steps
    -----
    1. Identify peaks marked as promoters where `var[promoter_col] == True`.
    2. Keep all promoter peaks unconditionally.
    3. Among non-promoter peaks, select the top (total_n_peaks - #promoters) peaks by standard deviation across clusters.
    4. If the number of promoter peaks alone is >= `total_n_peaks`, keep all promoters.
    5. Combine the sets of promoter and HV non-promoter peaks.

    Parameters
    ----------
    data_atac : anndata.AnnData
        An AnnData containing ATAC data, with a boolean promoter column in `.var`.
    total_n_peaks : int
        The target total number of peaks to keep. May be exceeded if promoter peaks alone
        surpass this number.
    cluster_key : str, optional
        Column in `data_atac.obs` defining cluster labels for HV peak selection. Default is "leiden".
    promoter_col : str, optional
        Column in `data_atac.var` indicating which peaks are promoters. Default is "is_promoter".

    Returns
    -------
    anndata.AnnData
        A subset of `data_atac` containing all promoter peaks plus HV non-promoter peaks.

    Notes
    -----
    - If `promoter_col` is missing, falls back to standard HV peak selection (without promoter logic).
    - The standard deviation for HV peaks is computed by `select_highly_variable_peaks_by_std`.
    """
    if promoter_col not in data_atac.var.columns:
        logger.warning(f"Column {promoter_col} not found in data_atac.var; no special promoter logic.")
        return select_highly_variable_peaks_by_std(data_atac, total_n_peaks, cluster_key)

    # (A) Extract promoter vs non-promoter
    promoter_mask = data_atac.var[promoter_col].values == True
    promoter_peaks = data_atac.var_names[promoter_mask]
    n_promoters = len(promoter_peaks)

    logger.info(f"Found {n_promoters} promoter peaks. Target total is {total_n_peaks}.")

    if n_promoters >= total_n_peaks:
        logger.warning(
            f"Promoter peaks ({n_promoters}) exceed total_n_peaks={total_n_peaks}. "
            "Keeping all promoters, final set might exceed user target."
        )
        data_atac_sub = data_atac[:, promoter_peaks].copy()
        return data_atac_sub
    else:
        # (B) Keep all promoters, then select HV among non-promoters
        n_needed = total_n_peaks - n_promoters
        logger.info(f"Selecting HV among non-promoters => picking {n_needed} peaks.")

        # Subset to non-promoters
        non_promoter_mask = ~promoter_mask
        data_atac_nonprom = data_atac[:, non_promoter_mask].copy()

        # HV selection among non-promoters
        data_atac_nonprom_hv = select_highly_variable_peaks_by_std(data_atac_nonprom, n_needed, cluster_key)

        # Final union => promoter + HV(non-promoters)
        final_promoter_set = set(promoter_peaks)
        final_nonprom_set = set(data_atac_nonprom_hv.var_names)
        final_set = list(final_promoter_set.union(final_nonprom_set))

        data_atac_sub = data_atac[:, final_set].copy()
        logger.info(
            f"Final set => {len(promoter_peaks)} promoter + "
            f"{data_atac_nonprom_hv.shape[1]} HV => total {data_atac_sub.shape[1]} peaks."
        )
        return data_atac_sub
