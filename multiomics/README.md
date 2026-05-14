# Multi-omics (snRNA + ATAC)

Joint analysis of matched single-nucleus RNA-seq and ATAC-seq data using `muon`, `scanpy`, and a custom Python package `gbm_tme`.

## Notebooks

| # | Notebook | Dataset | Focus |
|---|----------|---------|-------|
| 01 | `01_snrna_atac_preprocess.ipynb` | Per-modality QC, normalization, LSI for ATAC, UMAP, diffusion pseudotime |
| 02 | `02_snrna_atac_wnn_embedding.ipynb` | MuData construction, WNN joint UMAP, modality weight inspection |
| 03 | `03_peak2gene_correlation.ipynb` | Proximal peak discovery (pyranges), KNN smoothing, Spearman correlation, GC-matched background test, BH FDR |

## `gbm_tme/` package

Modular Python library for GBM tumor microenvironment analysis.

```
gbm_tme/
├── preprocessing.py   # QC metrics, violin + scatter plots, MAD filtering
├── annotation.py      # Marker gene dotplots, cell-type annotation scaffolding
├── integration.py     # Dimensionality reduction, embedding, batch correction
├── immune.py          # T/NK/myeloid subclustering helpers
├── de.py              # Differential expression wrappers (rank_genes_groups)
├── plotting.py        # Shared figure style, save_figure
├── io.py              # AnnData read/write utilities
└── utils.py           # repo_root, resolve_path, set_seed
```


## Dependencies

`scanpy`, `muon`, `mudata`, `anndata`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

