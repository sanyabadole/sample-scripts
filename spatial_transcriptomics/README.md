# Spatial transcriptomics

1 end-to-end analyses using 10x Genomics Visium and Squidpy on publicly available datasets.

## Notebooks

| # | Notebook | Dataset | Focus |
|---|----------|---------|-------|
| 01 | `01_squidpy_neighborhood_analysis.ipynb` | Squidpy tutorial dataset | Cell-type neighborhood enrichment, spatial statistics |


1. Load SpaceRanger output (`sc.read_visium`)
2. QC on per-spot counts, genes, and mitochondrial fraction; FFPE-adjusted thresholds
3. Normalization (`sc.pp.normalize_total`, log1p), HVG selection
4. PCA → neighbors → Leiden clustering → UMAP
5. Spatial overlay of clusters on H&E image
6. Rank-genes DE (`sc.tl.rank_genes_groups`) → marker identification → cell-type annotation

## Dependencies

`scanpy`, `squidpy`, `anndata`, `numpy`, `pandas`, `matplotlib`, `seaborn`

## Data
