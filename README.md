# scDoRI: Single-cell Deep Multi-Omic Regulatory Inference

![scDoRI Schematic](docs/_static/scdori_schematic_main.png)

**scDoRI** is a deep learning model for single-cell **multiome** data (RNA + ATAC in same cell) that infers **enhancer-mediated gene regulatory networks (eGRNs)**. By combining an **encoder–decoder** approach with mechanistic constraints (enhancer–gene links, TF binding logic), scDoRI learns **topics** that group co-accessible peaks, their cis-linked genes and upstream activator and repressor TFs – all while scaling to large datasets via mini-batches.

## 🚀 Highlights
- 🔄 **Unified** approach: single model for dimensionality reduction + eGRN inference
- 🧠 Learns **topics** that represent cell-state-specific regulatory programs
- 🧬**Continuous eGRN modelling** : each cell is a mixture of topics, allowing study of changes in GRNs. No need for predefined clusters  
- 🧰 **Scalable** to large datasets via **mini-batch training**

## 📥 Input Requirements

scDoRI expects **single-cell multiome data** with the following inputs:

- `RNA`: an AnnData `.h5ad` object with **cells × genes** expression matrix  
- `ATAC`: an AnnData `.h5ad` object with **cells × peaks** accessibility matrix  
  - Peaks must include genomic coordinates in `.var` (columns: `chr`, `start`, `end`)

These datasets must be paired — i.e., RNA and ATAC should come from the **same cells**.
  
## 📦 Installation 

To install all dependencies for scDoRI, we recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

```bash
git clone https://github.com/saraswatmanu/scDoRI.git
cd scDoRI
conda env create -f environment.yml
conda activate scdori_env
```
> ⚡ **Note**: The training process is GPU-accelerated and **highly recommended** to be run on a GPU-enabled machine.  
> While preprocessing can run on CPU, training large datasets on CPU is not advised due to slow performance.


## ⚙️ Usage
You’ll work through two notebooks, using two separate config files to set parameters for your dataset preprocessing and training.
### 🧹 Step 1: Preprocessing
#### Edit paths and parameters in:
```bash
preprocessing_pipeline/config.py
```
to specify location of RNA and ATAC anndata .h5ad files, motif file, and set number of peaks/genes/TFs to train on. 
#### Run preprocessing notebook
```bash
notebooks/preprocessing.ipynb
```
### 🧠 Step 2: Training and Downstream analysis

#### Edit paths and parameters in:
```bash
scdori/config.py
```
for scDoRI hyperparameters (number of topics, learning rate, epochs etc.) and specify path for preprocessed anndata objects and insilico-chipseq files
#### Run training and downstream analysis notebook
```bash
notebooks/training.ipynb
```
## 🧪 Dataset Demonstration

The provided notebooks use the **mouse gastrulation dataset** from:

📄 [Argelaguet et al., Bioarxiv 2022](https://www.biorxiv.org/content/10.1101/2022.06.15.496239v1)  
📦 Download: [Dropbox link](https://www.dropbox.com/scl/fo/9inmw43pz2bygtqepxl82/ALeeNjuEqw4qp0L9Z9t71xo/data/processed?rlkey=5ihgkvafegkke9jnldlnhw1x6&subfolder_nav_tracking=1&st=cixvwynt&dl=0)

## ⚙️ Configuration Notes

`preprocessing_pipeline/config.py` provide flexible options:

- You can **set the number of peaks, genes, and TFs** to use for model training  
  - 💡 Tip: Adjust based on your available **GPU memory**
- You can also **force inclusion of specific genes or TFs**, even if they aren’t highly variable  
  - Useful for focusing on known regulators/ genes of interest

## 📚 Documentation
📖 Full documentation and API reference is hosted at:

👉 https://saraswatmanu.github.io/scDoRI/

Includes:

✅ API reference (docstrings)

🛠️ In-depth method overview

🧪 Preprocessing + training guides

⚙️ (upcoming) Customization tips

## 📣 Citation
If you use scDoRI in your work, please cite our preprint/paper (coming soon).
Until then, feel free to open an issue or get in touch at manu.saraswat@dkfz.de
