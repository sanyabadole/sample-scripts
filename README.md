# Computational biology portfolio

Single-cell and spatial genomics work across four projects. Each subfolder has its own README.

## Projects

| Folder | Modality | Summary |
|--------|----------|---------|
| [`b_cell_atlas_portfolio/`](b_cell_atlas_portfolio/README.md) | scRNA-seq | Pan-pediatric cancer B-cell atlas: Scanpy QC + concat (backed h5ad), Seurat v5 + BPCells + Harmony integration |
| [`spatial_transcriptomics/`](spatial_transcriptomics/README.md) | Spatial (Visium) | Three Visium pipelines: Squidpy neighborhood stats, GBM FFPE, breast cancer TME |
| [`multiomics/`](multiomics/README.md) | snRNA + ATAC | WNN joint embedding, peak–gene linkage, GBM MuData; includes `gbm_tme` Python package |
| [`bcr_cll_scrna/`](bcr_cll_scrna/README.md) | scRNA + BCR | CLL cohort (GSE295491, ~250k cells): MAD-based per-sample QC, scVI batch integration |

## Skills demonstrated

- Large-scale multi-cohort integration (backed AnnData, BPCells on-disk, sketched UMAP projection)
- Batch correction: Harmony, scVI (VAE-based), Seurat v5 WNN
- Spatial transcriptomics: Visium QC, clustering, spatial DE, neighborhood enrichment
- Multi-omics: joint snRNA + ATAC, LSI, WNN, peak-to-gene linkage
- Immune repertoire: BCR sequencing QC, paired RNA + BCR analysis
- Python: Scanpy, scvi-tools, muon, squidpy, anndata, pandas
- R: Seurat, BPCells, harmony, ggplot2, future
- Package engineering: `gbm_tme` (modular src layout, typed functions, tests)

## Setup

```bash
export B_CELL_ATLAS_DATA=/path/to/your/data
```

All notebooks and scripts read paths from `DATA_ROOT = Path(os.environ.get("B_CELL_ATLAS_DATA", "data"))` (Python) or `Sys.getenv("B_CELL_ATLAS_DATA", "data")` (R).
