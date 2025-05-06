.. _training_hyperparameters:

==================================================================
Hyperparameter Configuration Guide
==================================================================

This guide provides a comprehensive reference for configuring feature selection and training parameters when using scDoRI.

Input Feature Selection
--------------------------

In the preprocessing configuration file (`preprocessing_pipeline/config.py`), you can define the number of peaks, genes, and transcription factors (TFs) to be used in model training.

- Although you may increase the number of input features, your final selection should reflect the available GPU memory. The default configuration supports GPUs with approximately 12--15 GB of memory.
- If you are using GPUs with higher memory capacity, you may expand the feature set for finer granularity.

**Manual Inclusion of Specific Genes and TFs**

- You can force the inclusion of certain genes or TFs using the `genes_user` and `tfs_user` entries in the config file.
- This is particularly useful if you have known regulators or candidate markers from prior work or literature.
- Ensure that all TFs in the `tfs_user` list have corresponding entries in your motif database. TFs without motif matches will be excluded during motif scanning.

-------------------
Motif File Settings
-------------------

- The motif database must be provided in MEME format. The default setup includes cisBP files, but you may use alternative databases, provided they match the same structure.
- Motif scanning is executed using **FIMO** (Grant et al., https://meme-suite.org/meme/doc/fimo.html), via the **tangermeme** (https://tangermeme.readthedocs.io/en/latest/tutorials/Tutorial_D1_FIMO.html, author Jacob Schreiber).
- Use the `motif_match_pvalue_threshold` parameter to control the stringency of motif matching. A stricter cutoff (lower p-value) yields fewer but more confident matches.

--------------------------------------
TF - Peak Correlation: Empirical Filtering
--------------------------------------

- scDoRI computes TF expression –-peak acessibility Pearson correlations at the metacell level to enhance TF--peak specificity (insilico-ChIP-seq).
- Adjust the `correlation_percentile` parameter to prune weak associations.
- While not yet natively supported, advanced users can assign different correlation thresholds for activators (positive) and repressors (negative) in the source code for improved regulatory resolution.

----------------------
Metacell Construction
----------------------

- Metacells are formed via high-resolution Leiden clustering performed on Harmony-corrected PCA embeddings.
- For robust correlation analysis, aim for at least 50 metacell clusters.
- If fewer than 50 clusters are obtained, consider increasing the `leiden_resolution` parameter in the config.

------------------------------
Enhancer - Gene Distance Window
------------------------------

- scDoRI links enhancers to genes within a fixed genomic window. By default, this is set to 150 kb upstream and downstream of gene-body for the human genome.
- Expanding this window may increase sensitivity but also introduces a higher chance of spurious associations.
- Conversely, reducing the window will favor specificity but may exclude distal regulatory elements.

**Gene Annotation Management**

- Default GTF files for `hg38` (https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.primary_assembly.basic.annotation.gtf.gz) and `mm10` (https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M18/gencode.vM18.basic.annotation.gtf.gz) are included.
- Users can provide their own GTF annotation by setting a URL or local file path in the config file.

Model Training Settings (`scdori/config.py`)
--------------------------------------

**Choosing the Number of Topics**

- A recommended starting point is: `#topics = expected number of cell types + 10`.
- Topics with low activity will be suppressed due to softmax normalization.
- Adding more topics enables modeling of fine-grained programs, but increases memory usage as each topic has its own GRN.

**Regularisation Strategies**

- Regularisation is applied using L1/L2 penalties on the following matrices:

  - Topic - Peak
  - Gene - Peak
  - Topic - TF Expression
  - GRN: Topic - TF - Gene 3D matrices

- Higher regularisation promotes sparsity and easier biological interpretation.
- Consider different regularisation strengths for activator and repressor GRNs.

**Epoch Configuration and Training Length**

- Use the following heuristic based on number of training steps to set the number of epochs for Phase 1:

  .. code-block:: text

     epochs = 60000 training steps / (number_of_cells / batch_size)
     ### For example, 30,000 cells and batch_size = 128:
     ### epochs = 60000 / (30000 / 128) = 256 epochs

- Set the `patience` parameter to approximately 5% of the total epoch count, but minimum of 5. This is used to monitor the number of epochs to wait for before stopping training when validation loss doesn't improve (early stopping)

**Phase 2: GRN Inference and Fine-Tuning**

- For Phase 2, you can reduce the total training steps by half (i.e., ~15,000 updates), but minimum of 10 epochs.
- Due to added complexity in GRN logic (e.g., 3D tensors), wall time per epoch is higher than phase 1.

