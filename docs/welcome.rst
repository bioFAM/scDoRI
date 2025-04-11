.. _welcome:

=========================================================
Welcome & Executive Summary
=========================================================

.. image:: _static/scdori_schematic_main.png
   :align: center
   :width: 80%
   :alt: scDoRI Schematic

**scDoRI** jointly models single-cell RNA-seq and ATAC-seq multi-ome data to infer
**enhancer-mediated gene regulatory networks (eGRNs)**. It couples an
**Encoder-Decoder** neural architecture with mechanistic constraints
(enhancer-gene links, TF activators/repressors), yielding
**topics** of co-accessible peaks, co-expressed genes, TF regulators and their enhancer-mediated downstream targets.
By training in mini-batches, scDoRI handles large datasets while capturing
continuous, cell-specific changes in gene regulation.

Key Highlights
--------------

- **Unified** approach: single model for dimensionality reduction + eGRN inference
- **Biological insights**:  identifies lower dimensional topics, candidate enhancer-gene links, co-regulated gene programs, TF-gene networks per topic
- **Continuous eGRN Modelling without predefined clusters**: each cell is a mixture of regulatory topics, allowing assessment of fine-grained changes in regulatory programs
- **Scalable**: mini-batch training for large single-cell multiome datasets


Input Requirements
------------------

scDoRI expects **single-cell multiome data** with the following inputs:

- **RNA**: an AnnData `.h5ad` object with a **cells by genes** raw expression counts matrix
- **ATAC**: an AnnData `.h5ad` object with a **cells by peaks** raw Tn5 insertion counts matrix
  - Peaks must include genomic coordinates in `.var` with columns: `chr`, `start`, and `end`

These datasets must be paired i.e., RNA and ATAC should come from the **same cells**.

The example notebooks provided in this repository are built using the **mouse gastrulation dataset** from:

- Argelaguet et al., BioRxiv 2022: https://www.biorxiv.org/content/10.1101/2022.06.15.496239v1
- Dataset download link: https://www.dropbox.com/scl/fo/9inmw43pz2bygtqepxl82/ALeeNjuEqw4qp0L9Z9t71xo/data/processed?rlkey=5ihgkvafegkke9jnldlnhw1x6&subfolder_nav_tracking=1&st=cixvwynt&dl=0




Model Architecture and training
-------------

See the :doc:`method_overview` for descriptions on core features of the model including encoder--decoder design, reconstruction tasks and training scheme.


Project Layout
--------------

- **preprocessing_pipeline/**
  Scripts + a `config.py` for data filtering, highly variable peak/gene/TF selection. Also computes in-silico ChIP-seq matrix.

- **scdori/**
  Core scDoRI model code + another `config.py` for hyperparameters
  (number of topics, learning rate, sparsity, etc.).

- **notebooks/**
  - `preprocessing.ipynb`: Load & filter multi-ome data, obtain in-silico ChIP-seq matrix and other preprocessing steps.
  - `training.ipynb`: Train the scDoRI autoencoder with mini-batches, produce eGRN outputs.

- **environment.yml**
  Conda environment specifying dependencies (scanpy, pytorch, etc.).

- **cisbp_motif_file**
  Example motif DB for mouse/human. If you use a custom motif file,
  set the path in the config.

Installation and Usage
----------------------

1. **Clone** this repo + create the environment:

   .. code-block:: bash

      git clone https://github.com/saraswatmanu/scDoRI.git
      cd scDoRI
      conda env create -f environment.yml
      conda activate scdori_env

2. **Edit** config files:
   - `preprocessing_pipeline/config.py` to specify location of RNA and ATAC anndata .h5ad files, motif file, and set number of peaks/genes/TFs to train on.
   - `scdori/config.py` for scDoRI hyperparameters (number of topics, learning rate, epochs etc.)

3. **Run** notebooks in order:
   - `notebooks/preprocessing.ipynb`
   - `notebooks/training.ipynb`

.. caution::
   If using a mouse dataset, set ``species = "mouse"`` in config.
   For human, change accordingly and update your motif file path (cisbp or custom).
   Ensure consistent schema in motif meme file compared to the example cisbp file provided.

Tutorial Notebooks
------------------

.. grid:: 2
   :gutter: 2

   .. card:: Preprocessing (Notebook 1)
      :link: notebooks/preprocessing
      :link-type: doc

      - **Filter** to highly variable genes/peaks/TFs
      - **Compute** in-silico ChIP-seq from your motif DB, peak-gene distances
      - **Output** processed data, insilico-chipseq matrix, peak-gene distances

   .. card:: Training (Notebook 2)
      :link: notebooks/training
      :link-type: doc

      - **Train** model with mini-batches
      - **Infer** topics and TF–gene networks
      - **Downstream analysis** using inferred eGRNs and topic activities



Hyperparameter and feature selection guide
-------------

See the :doc:`training_guide` page for documentation for guidance on choosing number of features(peaks, genes, TFs) and hyperparameters(number of topics, regularisation etc)

API Reference
-------------

See the :doc:`api_reference` page for documentation on:

- **preprocessing_pipeline** scripts
- **scdori** model scripts

These detail function usage, parameters, and advanced features.

License & Citation
------------------

This project is under MIT License. If scDoRI aids your research, please cite our
upcoming publication. For questions, open a GitHub Issue or email the maintainers.
