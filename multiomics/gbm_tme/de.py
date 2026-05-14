"""Differential expression and optional pathway enrichment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import scanpy as sc

from gbm_tme.utils import resolve_path


def rank_genes_between_groups(
    adata: ad.AnnData,
    *,
    groupby: str,
    idents: tuple[str, str],
    params: dict[str, Any],
    key_added: str = "rank_genes_groups",
) -> ad.AnnData:
    """Run ``sc.tl.rank_genes_groups`` for two identities in ``groupby``."""
    g0, g1 = idents
    adata = adata.copy()
    de_cfg = params.get("de", {})
    method = str(de_cfg.get("method", "wilcoxon"))
    tie_correct = bool(de_cfg.get("tie_correct", True))
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        groups=[g1],
        reference=g0,
        method=method,
        key_added=key_added,
        rankby_abs=False,
        tie_correct=tie_correct,
    )
    return adata


def de_table_from_rank_genes(
    adata: ad.AnnData,
    *,
    key: str = "rank_genes_groups",
    group: str | None = None,
) -> pd.DataFrame:
    """Convert Scanpy ``rank_genes_groups`` output to a tidy DataFrame."""
    if key not in adata.uns:
        raise KeyError(f"{key!r} not found in adata.uns; run rank_genes_groups first.")
    if group is not None:
        return sc.get.rank_genes_groups_df(adata, key=key, group=group)
    group_names = list(adata.uns[key]["params"]["group_names"])
    dfs = [sc.get.rank_genes_groups_df(adata, key=key, group=g) for g in group_names]
    return pd.concat(dfs, ignore_index=True)


def save_de_csv(table: pd.DataFrame, path: str | Path, *, root: Path | None = None) -> Path:
    """Write DE table to CSV under ``reports/tables`` (or custom path)."""
    out = resolve_path(str(path), root=root)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return out


def run_preranked_gseapy(
    ranked_genes: pd.DataFrame,
    *,
    gene_col: str = "names",
    score_col: str = "scores",
    organism: str = "human",
    gene_sets: list[str] | None = None,
    outdir: str | Path | None = None,
) -> Any | None:
    """Run preranked GSEA if ``gseapy`` is installed; otherwise return None."""
    try:
        import gseapy as gp
    except ImportError:
        return None

    gene_sets = gene_sets or ["GO_Biological_Process_2023"]
    rnk = ranked_genes[[gene_col, score_col]].dropna().drop_duplicates(subset=[gene_col])
    rnk = rnk.sort_values(score_col, ascending=False)
    res = gp.prerank(
        rnk=rnk,
        gene_sets=gene_sets,
        organism=organism,
        outdir=str(outdir) if outdir else None,
        permutation_num=100,
        seed=0,
        verbose=False,
    )
    return res
