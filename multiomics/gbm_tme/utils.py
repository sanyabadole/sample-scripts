"""Configuration loading, reproducibility helpers, and path utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


def repo_root(start: Path | None = None) -> Path:
    """Return repository root by searching parents for pyproject.toml."""
    path = (start or Path(__file__)).resolve()
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a nested dictionary."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def merge_paths_params(paths: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Shallow merge for convenience in notebooks (params override paths keys if duplicated)."""
    out = dict(paths)
    out.update(dict(params))
    return out


def resolve_path(path_str: str | None, root: Path | None = None) -> Path | None:
    """Resolve a path relative to repo root; return None if path_str is None/empty."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (root or repo_root()) / p


def set_seed(seed: int) -> None:
    """Best-effort deterministic seeding for NumPy and Python RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def export_obs_summary(
    adata,
    columns: list[str],
    out_path: str | Path,
    *,
    root: Path | None = None,
) -> Path:
    """Export value counts per column in ``adata.obs`` to a CSV summary table."""
    out = resolve_path(str(out_path), root=root)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in columns:
        if col not in adata.obs.columns:
            continue
        vc = adata.obs[col].value_counts(dropna=False)
        for val, count in vc.items():
            rows.append({"column": col, "value": val, "n_cells": int(count)})
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def cluster_composition_table(
    adata,
    cluster_key: str,
    sample_key: str,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Build a cells-by-cluster × sample contingency (optionally column-normalized per sample)."""
    if cluster_key not in adata.obs or sample_key not in adata.obs:
        raise KeyError(f"Missing {cluster_key=} or {sample_key=} in adata.obs")
    ct = pd.crosstab(adata.obs[cluster_key], adata.obs[sample_key])
    if normalize:
        ct = ct / ct.sum(axis=0)
    return ct


def export_cluster_composition(
    adata,
    cluster_key: str,
    sample_key: str,
    out_path: str | Path,
    *,
    root: Path | None = None,
    normalize: bool = True,
) -> Path:
    """Write cluster composition by sample to CSV."""
    out = resolve_path(str(out_path), root=root)
    assert out is not None
    tbl = cluster_composition_table(adata, cluster_key, sample_key, normalize=normalize)
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out)
    return out


def integration_umap_comparison_figure(
    adata,
    umap_pre_key: str,
    umap_post_key: str,
    color_key: str | None,
    *,
    title_pre: str = "Pre-integration UMAP",
    title_post: str = "Post-integration UMAP",
    save: str | Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (12, 5),
):
    """Plot side-by-side UMAPs from two ``obsm`` embeddings (e.g. PCA vs scVI).

    Parameters
    ----------
    adata
        AnnData with both UMAPs precomputed in ``obsm``.
    umap_pre_key, umap_post_key
        Keys in ``adata.obsm`` pointing to 2D coordinates.
    color_key
        Column in ``adata.obs`` used for coloring (same for both panels).
    save
        If set, path for figure export.
    show
        Whether to call ``plt.show()``.
    """
    import matplotlib.pyplot as plt

    from gbm_tme import plotting

    plotting.apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sc = None
    for ax, key, title in zip(
        axes,
        (umap_pre_key, umap_post_key),
        (title_pre, title_post),
        strict=True,
    ):
        coords = adata.obsm[key][:, :2]
        if color_key and color_key in adata.obs:
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=pd.Categorical(adata.obs[color_key]).codes,
                s=2,
                cmap="tab20",
                alpha=0.85,
                rasterized=True,
            )
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.85, rasterized=True)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if color_key and sc is not None:
        # Legend omitted for categorical scatter with tab20 — use dotplot for interpretability
        pass
    fig.tight_layout()
    if save:
        plotting.save_figure(fig, save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def write_json_summary(payload: Mapping[str, Any], out_path: str | Path, *, root: Path | None = None) -> Path:
    """Write a small JSON sidecar for pipeline provenance (parameters, versions)."""
    out = resolve_path(str(out_path), root=root)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return out
