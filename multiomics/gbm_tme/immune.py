"""Immune compartment sub-clustering, scoring, and coarse subtype labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from gbm_tme import integration
from gbm_tme.utils import set_seed


def subset_immune_from_config(adata: ad.AnnData, params: dict[str, Any]) -> ad.AnnData:
    """Subset immune cells using ``params.immune.broad_labels_match`` substrings."""
    imm = params.get("immune", {})
    subs = list(imm.get("broad_labels_match", []))
    return subset_immune_cells(adata, label_substrings=subs)


def subset_immune_cells(
    adata: ad.AnnData,
    broad_key: str = "celltype_broad",
    *,
    label_substrings: list[str] | None = None,
    exact_labels: list[str] | None = None,
) -> ad.AnnData:
    """Subset to immune-lineage cells using broad annotations."""
    label_substrings = label_substrings or ["T", "NK", "Myeloid", "Immune", "Lymphoid", "Mono", "DC", "Macro", "Micro"]
    mask = np.zeros(adata.n_obs, dtype=bool)
    values = adata.obs[broad_key].astype(str)
    for pat in label_substrings:
        mask |= values.str.contains(pat, case=False, regex=False)
    if exact_labels:
        mask |= values.isin(exact_labels)
    if not mask.any():
        raise ValueError(
            "No cells matched immune subset rules. "
            f"Broad labels present: {sorted(values.unique().tolist())}"
        )
    return adata[mask].copy()


def recluster_immune_subset(
    adata: ad.AnnData,
    params: dict[str, Any],
    *,
    use_rep: str | None = None,
    neighbors_key: str = "neighbors_immune",
    umap_key: str = "X_umap_immune",
    leiden_key: str = "leiden_immune",
    random_seed: int | None = None,
) -> ad.AnnData:
    """Recompute neighbors / UMAP / Leiden within immune subset."""
    seed = int(random_seed if random_seed is not None else params.get("random_seed", 0))
    set_seed(seed)
    adata = adata.copy()

    rep = use_rep
    if rep is None:
        integ = params.get("integration", {})
        rep = str(integ.get("latent_key", "X_scVI")) if integ.get("latent_key") in adata.obsm else "X_pca"

    neigh = params.get("neighbors", {})
    if rep in adata.obsm:
        sc.pp.neighbors(
            adata,
            use_rep=rep,
            n_neighbors=int(neigh.get("n_neighbors", 15)),
            key_added=neighbors_key,
        )
    else:
        sc.pp.neighbors(
            adata,
            n_neighbors=int(neigh.get("n_neighbors", 15)),
            n_pcs=int(neigh.get("n_pcs", 30)),
            key_added=neighbors_key,
        )

    umap_cfg = params.get("umap", {})
    sc.tl.umap(adata, min_dist=float(umap_cfg.get("min_dist", 0.3)), random_state=seed, neighbors_key=neighbors_key)
    adata.obsm[umap_key] = adata.obsm["X_umap"].copy()

    res = float(params.get("immune", {}).get("leiden_resolution", 0.8))
    sc.tl.leiden(adata, resolution=res, key_added=leiden_key, neighbors_key=neighbors_key)
    return adata


def train_immune_scvi_optional(
    adata: ad.AnnData,
    params: dict[str, Any],
    *,
    model_dir: str | Path | None = None,
) -> tuple[Any | None, ad.AnnData]:
    """Optionally train a second scVI model on immune cells only."""
    cfg = params.get("immune_scvi", {})
    if not bool(cfg.get("enabled", False)):
        return None, adata
    # Temporarily override integration keys for latent storage
    params_ = dict(params)
    integ = dict(params_.get("integration", {}))
    integ["latent_key"] = str(cfg.get("latent_key", "X_scVI_immune"))
    integ["scvi_epochs"] = int(cfg.get("scvi_epochs", integ.get("scvi_epochs", 200)))
    params_["integration"] = integ
    model, ad_out = integration.train_scvi_and_latent(adata, params_, model_dir=model_dir)
    return model, ad_out


def add_immune_state_scores(adata: ad.AnnData, markers_cfg: dict[str, Any]) -> ad.AnnData:
    """Add module scores defined under ``markers.yaml`` → ``scores``."""
    adata = adata.copy()
    score_block = markers_cfg.get("scores") or {}
    if isinstance(score_block, dict):
        for name, genes in score_block.items():
            if not isinstance(genes, list) or len(genes) < 2:
                continue
            present = [g for g in genes if g in adata.var_names]
            if len(present) < 2:
                continue
            col = f"score_{name}"
            sc.tl.score_genes(adata, gene_list=present, score_name=col, use_raw=False)
    return adata


def annotate_immune_fine_types(
    adata: ad.AnnData,
    *,
    cd8_col: str = "score_cytotoxicity",
    exh_col: str = "score_exhaustion",
    treg_col: str = "score_Treg",  # may not exist unless scored from markers
    mg_col: str = "score_microglia",
    mphi_col: str = "score_macrophage",
    nk_col: str = "score_NK",
    cd4_col: str = "score_CD4_helper",
    quantile: float = 0.7,
) -> ad.AnnData:
    """Heuristic immune subtype labels (tunable thresholds; for exploration, not ground truth).

    Produces ``adata.obs['immune_celltype']`` with coarse categories. Curate after manual review.
    """
    adata = adata.copy()
    obs = adata.obs
    labels = pd.Series("other", index=obs.index, dtype="object")

    def hi(col: str) -> pd.Series:
        if col not in obs.columns:
            return pd.Series(False, index=obs.index)
        thr = obs[col].quantile(quantile)
        return obs[col] >= thr

    # Myeloid split (microglia-like vs macrophage-like)
    if mg_col in obs.columns and mphi_col in obs.columns:
        mg_hi = hi(mg_col) & (obs[mg_col] > obs[mphi_col])
        mac_hi = hi(mphi_col) & (obs[mphi_col] > obs[mg_col])
        labels.loc[mg_hi] = "microglia_like"
        labels.loc[mac_hi] = "macrophage_like"

    # Lymphoid: prioritize NK, CD8, CD4, Treg using marker scores when present
    if nk_col in obs.columns and hi(nk_col).any():
        labels.loc[hi(nk_col) & (labels == "other")] = "NK"

    if cd8_col in obs.columns and exh_col in obs.columns:
        cyto = hi(cd8_col) & (obs[cd8_col] > obs[exh_col])
        exh = hi(exh_col) & (obs[exh_col] >= obs[cd8_col])
        labels.loc[cyto & labels.eq("other")] = "CD8_cytotoxic"
        labels.loc[exh & labels.eq("other")] = "CD8_exhausted"

    if treg_col in obs.columns and hi(treg_col).any():
        labels.loc[hi(treg_col) & labels.eq("other")] = "Treg"

    if cd4_col in obs.columns and hi(cd4_col).any():
        labels.loc[hi(cd4_col) & labels.eq("other")] = "CD4"

    adata.obs["immune_celltype"] = pd.Categorical(labels)
    return adata


def score_lineage_markers_from_yaml(adata: ad.AnnData, markers_cfg: dict[str, Any]) -> ad.AnnData:
    """Score named YAML blocks (e.g. ``microglia``, ``Treg``) for use in ``annotate_immune_fine_types``."""
    from gbm_tme.annotation import filter_markers_present, marker_genes_from_yaml

    markers = marker_genes_from_yaml(markers_cfg)
    markers = filter_markers_present(adata, markers)
    adata = adata.copy()
    for name, genes in markers.items():
        if len(genes) < 2:
            continue
        col = f"score_{name}"
        sc.tl.score_genes(adata, gene_list=genes, score_name=col, use_raw=False)
    return adata
