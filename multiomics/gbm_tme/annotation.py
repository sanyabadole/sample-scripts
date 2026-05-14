"""Broad cell-type annotation scaffolding from marker dictionaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import scanpy as sc

from gbm_tme import plotting


def marker_genes_from_yaml(markers_cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Extract positive gene lists per compartment from ``markers.yaml`` structure."""
    out: dict[str, list[str]] = {}
    for key, block in markers_cfg.items():
        if key == "scores":
            continue
        if isinstance(block, dict) and "positive" in block:
            genes = [g for g in block["positive"] if isinstance(g, str)]
            if genes:
                out[key] = genes
    return out


def filter_markers_present(adata: ad.AnnData, markers: dict[str, list[str]]) -> dict[str, list[str]]:
    """Drop symbols absent from ``adata.var_names``."""
    var = set(adata.var_names)
    return {k: [g for g in v if g in var] for k, v in markers.items() if any(g in var for g in v)}


def dotplot_markers(
    adata: ad.AnnData,
    markers: dict[str, list[str]],
    groupby: str,
    *,
    save: str | Path | None = None,
    show: bool = True,
    dendrogram: bool = False,
):
    """Wrapper for ``sc.pl.dotplot`` with consistent styling."""
    plotting.apply_plot_style()
    genes_ordered = [g for genes in markers.values() for g in genes]
    if not genes_ordered:
        raise ValueError("No marker genes present in AnnData after filtering.")
    dp = sc.pl.dotplot(
        adata,
        var_names=genes_ordered,
        groupby=groupby,
        show=False,
        dendrogram=dendrogram,
        standard_scale="var",
    )
    fig = dp["fig"] if isinstance(dp, dict) else None
    if fig is not None:
        fig.tight_layout()
        if save:
            plotting.save_figure(fig, save)
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            import matplotlib.pyplot as plt

            plt.close(fig)
    return dp


def score_marker_sets(adata: ad.AnnData, markers: dict[str, list[str]], *, prefix: str = "score_") -> ad.AnnData:
    """Compute per-cell module scores for each marker set using ``sc.tl.score_genes``."""
    adata = adata.copy()
    for name, genes in markers.items():
        if len(genes) < 2:
            continue
        col = f"{prefix}{name}"
        sc.tl.score_genes(adata, gene_list=genes, score_name=col, use_raw=False)
    return adata


def suggest_broad_labels_from_cluster_scores(
    adata: ad.AnnData,
    cluster_key: str,
    score_prefix: str = "score_",
) -> pd.DataFrame:
    """Return a table of mean module scores per cluster for manual review."""
    score_cols = [c for c in adata.obs.columns if c.startswith(score_prefix)]
    if not score_cols:
        raise ValueError("No score_* columns found; run score_marker_sets first.")
    rows = []
    for clust in sorted(adata.obs[cluster_key].unique(), key=lambda x: str(x)):
        mask = adata.obs[cluster_key] == clust
        sub = adata.obs.loc[mask, score_cols].mean()
        rows.append({"cluster": clust, **sub.to_dict()})
    return pd.DataFrame(rows)


def assign_broad_celltype(
    adata: ad.AnnData,
    cluster_key: str,
    rules: dict[str, str],
    *,
    out_key: str = "celltype_broad",
) -> ad.AnnData:
    """Map Leiden (or other) clusters to broad labels using a user-curated dict.

    ``rules`` maps cluster id strings to labels, e.g. ``{"0": "Malignant-like", "1": "T/NK"}``.
    Unmapped clusters are labeled ``Unknown``.
    """
    adata = adata.copy()
    mapped = adata.obs[cluster_key].astype(str).map(lambda x: rules.get(x, "Unknown"))
    adata.obs[out_key] = pd.Categorical(mapped)
    return adata


def majority_vote_neighbors(
    adata: ad.AnnData,
    *,
    key: str,
    neighbors_key: str | None = None,
    new_key: str | None = None,
) -> ad.AnnData:
    """Optional smoothing: assign each cell the mode label among kNN (requires connectivities)."""
    neighbors_key = neighbors_key or "neighbors"
    nbrs_info = adata.uns.get(neighbors_key)
    if isinstance(nbrs_info, dict) and "connectivities_key" in nbrs_info:
        conn_mat = adata.obsp[nbrs_info["connectivities_key"]]
    elif "connectivities" in adata.obsp:
        conn_mat = adata.obsp["connectivities"]
    else:
        raise KeyError("No connectivities matrix found for majority vote.")

    labels = adata.obs[key].astype(str).values
    new_key = new_key or f"{key}_smoothed"
    smoothed = []
    conn = conn_mat.tocsr()
    for i in range(conn.shape[0]):
        start, end = conn.indptr[i], conn.indptr[i + 1]
        neigh_idx = conn.indices[start:end]
        neigh_w = conn.data[start:end]
        if len(neigh_idx) == 0:
            smoothed.append(labels[i])
            continue
        votes = labels[neigh_idx]
        # weighted mode
        df = pd.DataFrame({"v": votes, "w": neigh_w})
        top = df.groupby("v")["w"].sum().sort_values(ascending=False).index[0]
        smoothed.append(top)
    adata.obs[new_key] = pd.Categorical(smoothed)
    return adata
