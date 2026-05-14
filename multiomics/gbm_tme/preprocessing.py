"""QC, normalization, and feature selection for scRNA-seq."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.figure import Figure

from gbm_tme import plotting
from gbm_tme.utils import repo_root, resolve_path, set_seed


def annotate_qc_metrics(
    adata: ad.AnnData,
    *,
    mito_prefix: str = "MT-",
) -> None:
    """Compute standard QC metrics in ``adata.obs`` (in-place)."""
    adata.var["mt"] = adata.var_names.str.startswith(mito_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)


def qc_violins(
    adata: ad.AnnData,
    *,
    keys: list[str] | None = None,
    save: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Violin plots for total counts, genes detected, and mito percent."""
    plotting.apply_plot_style()
    keys = keys or ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys, strict=False):
        sc.pl.violin(adata, keys=[k], ax=ax, show=False)
    fig.tight_layout()
    if save:
        plotting.save_figure(fig, save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def qc_scatter_counts_vs_genes(
    adata: ad.AnnData,
    *,
    color: str = "pct_counts_mt",
    save: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Scatter of total counts vs genes detected."""
    plotting.apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", color=color, ax=ax, show=False)
    fig.tight_layout()
    if save:
        plotting.save_figure(fig, save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def qc_hist_mito(
    adata: ad.AnnData,
    *,
    save: str | Path | None = None,
    show: bool = True,
    bins: int = 50,
) -> Figure:
    """Histogram of mitochondrial read percentage."""
    plotting.apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(adata.obs["pct_counts_mt"], bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("pct_counts_mt")
    ax.set_ylabel("cells")
    fig.tight_layout()
    if save:
        plotting.save_figure(fig, save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def apply_qc_filters(adata: ad.AnnData, qc_cfg: dict[str, Any]) -> ad.AnnData:
    """Subset cells/genes using thresholds from ``params.yaml`` ``qc`` block."""
    adata = adata.copy()
    min_genes = int(qc_cfg.get("min_genes", 200))
    min_counts = int(qc_cfg.get("min_counts", 500))
    max_mito = float(qc_cfg.get("max_mito_pct", 20))
    max_counts = qc_cfg.get("max_counts")

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    if max_counts is not None:
        adata = adata[adata.obs["total_counts"] <= float(max_counts)].copy()
    adata = adata[adata.obs["pct_counts_mt"] < max_mito].copy()

    min_cells = int(qc_cfg.get("min_cells", 3))
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


def normalize_and_hvgs(
    adata: ad.AnnData,
    params: dict[str, Any],
    *,
    target_sum: float | None = None,
) -> ad.AnnData:
    """Normalize to target sum, log1p, and select highly variable genes."""
    adata = adata.copy()
    tgt = target_sum if target_sum is not None else float(params.get("normalization", {}).get("target_sum", 1e4))
    sc.pp.normalize_total(adata, target_sum=tgt)
    sc.pp.log1p(adata)

    hvg = params.get("hvg", {})
    n_top = int(hvg.get("n_top_genes", 3000))
    flavor = str(hvg.get("flavor", "seurat"))
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=flavor, subset=bool(hvg.get("subset", True)))

    reg = params.get("regression", {}) or {}
    variables = list(reg.get("variables") or [])
    if variables:
        vars_present = [v for v in variables if v in adata.obs.columns]
        if vars_present:
            sc.pp.regress_out(adata, keys=vars_present, n_jobs=int(reg.get("n_jobs", 4)))

    sc.pp.scale(adata, max_value=10)
    return adata


def run_basic_pca_neighbors_umap(
    adata: ad.AnnData,
    params: dict[str, Any],
    *,
    random_seed: int,
    umap_key: str = "X_umap_pca",
    neighbors_key: str = "neighbors_pca",
) -> ad.AnnData:
    """PCA → neighbors → UMAP on HVG subset (pre-integration baseline)."""
    set_seed(random_seed)
    adata = adata.copy()
    if "highly_variable" in adata.var.columns:
        adata = adata[:, adata.var["highly_variable"]].copy()

    pca_cfg = params.get("pca", {})
    sc.tl.pca(adata, svd_solver=str(pca_cfg.get("svd_solver", "arpack")), n_comps=int(pca_cfg.get("n_comps", 50)))

    neigh = params.get("neighbors", {})
    sc.pp.neighbors(
        adata,
        n_neighbors=int(neigh.get("n_neighbors", 15)),
        n_pcs=int(neigh.get("n_pcs", 30)),
        key_added=neighbors_key,
    )
    umap_cfg = params.get("umap", {})
    sc.tl.umap(adata, min_dist=float(umap_cfg.get("min_dist", 0.3)), random_state=random_seed)
    adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
    return adata


def qc_and_preprocess_pipeline(
    adata: ad.AnnData,
    params: dict[str, Any],
    paths_cfg: dict[str, Any],
    *,
    figures_dir: str | Path | None = None,
    random_seed: int | None = None,
) -> ad.AnnData:
    """End-to-end QC + normalization + HVG + baseline PCA/UMAP; returns processed AnnData."""
    root = repo_root()
    seed = int(random_seed if random_seed is not None else params.get("random_seed", 0))
    set_seed(seed)

    mito_prefix = str(params.get("qc", {}).get("mito_prefix", "MT-"))
    annotate_qc_metrics(adata, mito_prefix=mito_prefix)

    fig_dir = resolve_path(str(figures_dir or paths_cfg.get("figures_dir", "reports/figures")), root=root)
    assert fig_dir is not None
    fig_dir.mkdir(parents=True, exist_ok=True)

    qc_violins(adata, save=fig_dir / "qc_violins.png", show=False)
    qc_scatter_counts_vs_genes(adata, save=fig_dir / "qc_counts_vs_genes.png", show=False)
    qc_hist_mito(adata, save=fig_dir / "qc_mito_hist.png", show=False)

    adata = apply_qc_filters(adata, params.get("qc", {}))
    adata = normalize_and_hvgs(adata, params)
    adata = run_basic_pca_neighbors_umap(adata, params, random_seed=seed)
    return adata
