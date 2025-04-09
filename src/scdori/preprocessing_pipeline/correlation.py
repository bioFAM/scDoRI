import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def compute_in_silico_chipseq(
    atac_matrix: np.ndarray,
    rna_matrix: np.ndarray,
    motif_scores: pd.DataFrame,
    percentile: float = 99.9,
    n_bg: int = 10000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation-based in-silico ChIP-seq embeddings, separating activator
    and repressor signals.

    Steps
    -----
    1. Compute the Pearson correlation for each (peak, TF) pair.
    2. Retain only the positive part as an "activator" correlation and the negative part as a "repressor" correlation.
    3. For each TF, estimate significance thresholds for correlations from background peaks (i.e., peaks with minimal motif scores for each TF).
    4. Multiply thresholded correlations by the motif scores to produce final activator and repressor embeddings.
    5. Apply MinMax scaling to each TF separately.

    Parameters
    ----------
    atac_matrix : np.ndarray
        Shape (n_metacells, n_peaks). Matrix of peak accessibility (or insertion counts)
        where each row is a metacell and each column is a peak.
    rna_matrix : np.ndarray
        Shape (n_metacells, n_tfs). Matrix of TF expression where each row is a metacell
        and each column is a TF.
    motif_scores : pd.DataFrame
        Shape (n_peaks, n_tfs). DataFrame containing motif scores for each (peak, TF) pair.
    percentile : float, optional
        The percentile threshold used to determine correlation significance (for activators
        and repressors). Default is 99.9.
    n_bg : int, optional
        Number of peaks per TF used to form the "background" distribution (lowest motif
        scores). Default is 10000.

    Returns
    -------
    insilico_chipseq_act_sig : np.ndarray
        Shape (n_peaks, n_tfs). Final activator scores after thresholding correlations
        and combining with motif scores. Scaled to [0, 1].
    insilico_chipseq_rep_sig : np.ndarray
        Shape (n_peaks, n_tfs). Final repressor scores after thresholding correlations
        and combining with motif scores. Scaled to [-1, 0].

    Notes
    -----
    - Positive correlation is interpreted as activator-like; negative correlation is repressor-like.
    - A background distribution is formed from peaks with the smallest motif scores, then used to find correlation thresholds at the requested percentile.
    - The final arrays are MinMax-scaled:
      * The repressor array is scaled into the range [-1, 0].
      * The activator array is scaled into the range [0, 1].
    """
    logger.info("Computing in-silico ChIP-seq correlation...")

    n_cells, n_peaks = atac_matrix.shape
    _, n_tfs = rna_matrix.shape
    if motif_scores.shape != (n_peaks, n_tfs):
        logger.warning("motif_scores dimension does not match (n_peaks x n_tfs).")

    # Z-score peaks & TF expression
    X = (atac_matrix - atac_matrix.mean(axis=0)) / (atac_matrix.std(axis=0) + 1e-8)
    Y = (rna_matrix - rna_matrix.mean(axis=0)) / (rna_matrix.std(axis=0) + 1e-8)

    # Pearson correlation => (n_peaks x n_tfs)
    pearson_r = (X.T @ Y) / n_cells
    pearson_r = np.nan_to_num(pearson_r)

    pearson_r_act = np.clip(pearson_r, 0, None)   # only positive
    pearson_r_rep = np.clip(pearson_r, None, 0)   # only negative

    pearson_r_act_sig = np.zeros_like(pearson_r_act)
    pearson_r_rep_sig = np.zeros_like(pearson_r_rep)

    tf_list = motif_scores.columns

    # Thresholding
    for t in tqdm(range(n_tfs), desc="Thresholding correlation"):
        tf_name = tf_list[t]
        # Find background peaks with smallest motif score
        scores_t = motif_scores[tf_name].values
        order = np.argsort(scores_t)
        bg_idx = order[:min(n_bg, n_peaks)]  # top n_bg smallest motif peaks

        # Activator significance
        bg_vals_act = pearson_r_act[bg_idx, t]
        cutoff_act = np.percentile(bg_vals_act, percentile)

        # Repressor significance
        bg_vals_rep = pearson_r_rep[bg_idx, t]
        cutoff_rep = np.percentile(bg_vals_rep, 100 - percentile)

        act_vec = pearson_r_act[:, t]
        rep_vec = pearson_r_rep[:, t]

        pearson_r_act_sig[:, t] = np.where(act_vec > cutoff_act, act_vec, 0)
        pearson_r_rep_sig[:, t] = np.where(rep_vec < cutoff_rep, rep_vec, 0)

    # Combine with motif
    insilico_chipseq_act_sig = motif_scores.values * pearson_r_act_sig
    insilico_chipseq_rep_sig = motif_scores.values * pearson_r_rep_sig

    # Scale repressor to [-1, 0]
    scaler = MinMaxScaler(feature_range=(-1, 0))
    insilico_chipseq_rep_sig = scaler.fit_transform(insilico_chipseq_rep_sig)

    # Scale activator to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    insilico_chipseq_act_sig = scaler.fit_transform(insilico_chipseq_act_sig)

    logger.info("Finished in-silico ChIP-seq computation.")
    return insilico_chipseq_act_sig, insilico_chipseq_rep_sig
