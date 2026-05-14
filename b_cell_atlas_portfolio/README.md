# Pan-pediatric B-cell atlas (single-cell)

Scanpy for QC, merge, and concat (including backed h5ad). Seurat v5 + BPCells after export; Harmony for batch correction. Raw data are not in this repo: set `B_CELL_ATLAS_DATA` to your local root (default `./data`).

## Layout

```text
b_cell_atlas_portfolio/
├── README.md
├── data/.gitkeep
├── figures/html/          # saved Plotly/htmlwidgets HTML
├── notebooks/             # 01–10 Scanpy / h5ad steps
└── scripts/
    ├── python/            # inventory, 10x features fix, SOLO doublet example
    └── r/                 # Seurat / Harmony / QC plots
```

Under `B_CELL_ATLAS_DATA` use folders like `geo/`, `intermediate/`, `raw/`, `scpca/`, `figures/` as in the notebooks and scripts.

## Dependencies

**Python:** scanpy, anndata, pandas, numpy, matplotlib, scipy, scvi-tools, seaborn; jupyter for notebooks.

**R:** Seurat, BPCells, harmony, ggplot2, future; SingleCellExperiment, scuttle for `09`.

Pin versions when you freeze an environment (`conda export`, `renv::snapshot()`, or `sessionInfo()`).

## Order

1. Notebooks `01`–`10` (Python).
2. `scripts/r/01` … `10` as needed: h5ad → Seurat → sketch or full PCA/UMAP → Harmony → QC figures → optional merge / SCE / small UMAP.

```bash
export B_CELL_ATLAS_DATA=/path/to/data
python scripts/python/01_inventory_scpca_rds_paths.py
Rscript scripts/r/01_seurat_from_scanpy_h5ad.R
```

HTML figures in `figures/html/` open in a browser. R scripts write PNGs under `$B_CELL_ATLAS_DATA/figures/` unless you change paths in the script.
