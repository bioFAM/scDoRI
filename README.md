# scDORI: Single-cell Deep Multi-Omic Regulatory Inference
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21937078.svg)](https://doi.org/10.5281/zenodo.21937078)
[![Documentation Status](https://readthedocs.org/projects/scdori/badge/?version=latest)](https://scdori.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/bioFAM/scDoRI/graph/badge.svg?token=RK6G4LBUHL)](https://codecov.io/gh/bioFAM/scDoRI)

![graphical abstract](https://raw.githubusercontent.com/bioFAM/scDoRI/refs/heads/main/docs/_static/scdori_schematic_main.png)

**scDORI** is a deep learning model for single-cell **multiome** data (RNA + ATAC in same cell) that infers **enhancer-mediated gene regulatory networks (eGRNs)**. By combining an **encoder–decoder** approach with mechanistic constraints (enhancer–gene links, TF binding logic), scDoRI learns **topics** that group co-accessible peaks, their cis-linked genes and upstream activator and repressor TFs – all while scaling to large datasets via mini-batches.

## Highlights
- 🔄 **Unified** approach: a single model for dimensionality reduction + eGRN inference
- 🧠 Learns **topics** that represent cell-state-specific regulatory programs
- 🧬**Continuous eGRN modelling** : each cell is a mixture of topics, allowing the study of changes in GRNs. No need for predefined clusters
- 🧰 **Scalable** to large datasets via **mini-batch training**

## Input Requirements

scDORI expects **single-cell multiome data** with the following inputs:

- `RNA`: an AnnData `.h5ad` object with **cells × genes** expression matrix
- `ATAC`: an AnnData `.h5ad` object with **cells × peaks** accessibility matrix
  - Peaks must include genomic coordinates in `.var` (columns: `chr`, `start`, `end`)

These datasets must be paired — i.e., RNA and ATAC should come from the **same cells**. Make sure that **.X** of anndata objects contain raw counts.


## Installation

To install all dependencies for scDoRI, we recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). The installation should take a couple of minutes.


```bash
git clone https://github.com/bioFAM/scDoRI.git
cd scDoRI

# create a new environment
conda create -n scdori python=3.12 -y
conda activate scdori

# Install BEDTools dependencies, e.g., via conda (if not installed on your system)
conda install --channel conda-forge --channel bioconda bedtools htslib -y

# Install the scDoRI package
pip install -e .
```

Make sure the `-e` flag is used to install the package in editable mode.

The training process is GPU-accelerated and **highly recommended** to be run on a GPU-enabled machine. While preprocessing can run on CPU, training large datasets on CPU is not advised due to slow performance. If a GPU is available, the GPU-accelerated version will be installed automatically.
GPU memory scales with the number of selected features; as a guide, a GPU with 12–15 GB of memory supports ~4,000 genes, 70,000 peaks and 300 TFs. Larger feature sets require proportionally more memory. 

**Software dependencies:** Python 3.12; BEDTools and HTSlib (installable via conda/bioconda, see below); all Python dependencies and their pinned versions are listed in `pyproject.toml` and installed automatically by `pip install -e .`.


## Usage
You’ll work through three notebooks - Preprocessing, Training and Downstream analysis. This will require two separate config files to set parameters for your dataset preprocessing and training.
### Step 1: Preprocessing
#### Edit paths and parameters in:
```bash
src/scdori/pp/config.py
```
to specify the location of RNA and ATAC anndata .h5ad files, motif file, and set number of peaks/genes/TFs to train on.


#### Run preprocessing notebook
```bash
docs/notebooks/preprocessing.ipynb
```
This step can take between a few hours to a day depending on the dataset size and number of features selected.

**Expected output:** preprocessed RNA and ATAC AnnData objects, the in-silico ChIP-seq TF–peak matrices and gene-peak distance matrices, written to the output paths set in `config.py`.

### Step 2: Training

#### Edit paths and parameters in:
```bash
src/scdori/_core/config.py
```
for scDoRI hyperparameters (number of topics, learning rate, epochs etc.) and specify path for preprocessed anndata objects and insilico-chipseq files

#### Run training notebook
```bash
docs/notebooks/training.ipynb
```
This step again can take between a few hours to a day depending on the dataset size and number of features selected.

**Expected output:** a trained scDoRI model checkpoint and the learned latent Topic activities, topic–peak, topic–gene and topic–TF matrices.

### Step 3: Downstream analysis

#### Run downstream analysis notebook
```bash
docs/notebooks/downstream.ipynb
```
**Expected output:** topic-specific eGRNs, TF activity scores, and the downstream tables and visualisations described in the documentation.

## Dataset Demonstration

The provided notebooks use the **mouse gastrulation dataset** from:

📄 Paper: [Argelaguet et al., Bioarxiv 2022](https://www.biorxiv.org/content/10.1101/2022.06.15.496239v1)
📦 Download: [Dropbox link](https://www.dropbox.com/scl/fo/9inmw43pz2bygtqepxl82/ALeeNjuEqw4qp0L9Z9t71xo/data/processed?rlkey=5ihgkvafegkke9jnldlnhw1x6&subfolder_nav_tracking=1&st=cixvwynt&dl=0)

## Configuration Notes

`preprocessing_pipeline/config.py` provides flexible options:

- You can **set the number of peaks, genes, and TFs** to use for model training
  - 💡 Tip: Adjust based on your available **GPU memory**
- You can also **force inclusion of specific genes or TFs**, even if they aren’t highly variable
  - Useful for focusing on known regulators/ genes of interest

## Documentation
📖 Full documentation and API reference is hosted at: https://scdori.readthedocs.io/en/latest/

Includes:
- API reference (docstrings)
- In-depth method overview
- Preprocessing + training guides
- (upcoming) Customization tips

## Citation
If you use scDoRI in your work, please cite our [preprint](https://www.biorxiv.org/content/10.1101/2025.05.13.653733v1).
Feel free to open an issue or get in touch at manu.saraswat@dkfz.de
