import logging

from ._core import (
    compute_activator_tf_activity_per_cell,
    compute_atac_grn_activator_with_significance,
    compute_atac_grn_repressor_with_significance,
    compute_neighbors_umap,
    compute_repressor_tf_activity_per_cell,
    compute_significant_grn,
    compute_topic_gene_matrix,
    compute_topic_peak_umap,
    get_latent_topics,
    get_tf_expression,
    get_top_activators_per_topic,
    get_top_repressor_per_topic,
    initialize_scdori_parameters,
    load_best_model,
    load_scdori_inputs,
    save_model_weights,
    save_regulons,
    scDoRI,
    set_seed,
    train_model_grn,
    train_scdori_phases,
    trainConfig,
)
from ._version import __version__, __version_tuple__
from .pl import plot_downstream_targets, plot_topic_activation_heatmap
from .pp import (
    compute_gene_peak_distance_matrix,
    compute_hvgs_and_tfs,
    compute_in_silico_chipseq,
    compute_motif_scores,
    create_dir_if_not_exists,
    create_extended_gene_bed,
    create_metacells,
    download_genome_references,
    filter_protein_coding_genes,
    intersect_cells,
    keep_promoters_and_select_hv_peaks,
    load_anndata,
    load_gtf,
    load_motif_database,
    ppConfig,
    remove_mitochondrial_genes,
    run_bedtools_intersect,
    save_processed_datasets,
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not logging.root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s\t%(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False
