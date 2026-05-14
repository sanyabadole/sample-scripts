# BCR repertoire + scRNA (CLL)

Two-notebook pipeline analyzing paired scRNA-seq and B-cell receptor (BCR) sequencing from a CLL cohort (GSE295491: HC-MBL, LC-MBL, CLL stages; 30 donors, ~250k cells).

## Notebooks

| # | Notebook | Steps |
|---|----------|-------|
| 01 | `01_qc_per_sample_concat.ipynb` | Parse series matrix metadata → per-sample MAD-based QC → write per-sample h5ad → concat to combined object |
| 02 | `02_scvi_integration.ipynb` | Load combined h5ad → scVI (2-layer, NB likelihood, batch=sample) → latent space → neighbors → UMAP → Leiden clustering |

## Key methods

- **Per-sample MAD filtering** (5 MADs on counts/genes, 3 MADs on MT%) rather than fixed global thresholds; adapts to library depth variation across donors
- **scVI** (`scvi-tools`) trained on raw counts with `layer="counts"`, `batch_key="gsm"` (30 RNA samples); `n_latent=20`, `batch_size=512`, early stopping
- Downstream: `sc.pp.neighbors(use_rep="X_scVI")` → `sc.tl.umap` → `sc.tl.leiden`

## Dependencies

`scanpy`, `anndata`, `scvi-tools`, `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## Data

GSE295491 — download from NCBI GEO; set `B_CELL_ATLAS_DATA` to point at your local copy.
