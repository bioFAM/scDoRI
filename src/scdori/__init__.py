import logging

from ._core import (
    initialize_scdori_parameters,
    load_best_model,
    load_scdori_inputs,
    save_model_weights,
    scDoRI,
    set_seed,
    train_model_grn,
    train_scdori_phases,
    trainConfig,
)
from ._version import __version__, __version_tuple__
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
