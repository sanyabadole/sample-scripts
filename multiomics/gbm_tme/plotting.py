"""Publication-oriented plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from gbm_tme.utils import resolve_path


def apply_plot_style() -> None:
    """Set matplotlib defaults: white background, readable fonts, minimal clutter."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "text.color": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: mpl.figure.Figure, path: str | Path, *, root: Path | None = None) -> Path:
    """Save figure to disk (PNG + optional PDF if path ends with .pdf)."""
    out = resolve_path(str(path), root=root)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    return out


def umap_by_obs(
    adata,
    *,
    color: str,
    basis: str = "X_umap",
    neighbors_key: str | None = None,
    save: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
    palette: str | None = None,
    **kwargs: Any,
):
    """UMAP colored by an ``obs`` column (wraps ``sc.pl.embedding``)."""
    apply_plot_style()
    kwargs = {"show": False, "basis": basis, "color": color, **kwargs}
    if neighbors_key:
        kwargs["neighbors_key"] = neighbors_key
    if palette:
        kwargs["palette"] = palette
    ax = sc.pl.embedding(adata, **kwargs)
    fig = plt.gcf()
    if title:
        fig.axes[0].set_title(title)
    fig.tight_layout()
    if save:
        save_figure(fig, save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def qc_panel_grid(
    *,
    paths: list[Path],
    out_path: str | Path,
    root: Path | None = None,
    show: bool = False,
) -> Path:
    """Assemble pre-rendered QC PNGs into a single multi-panel figure."""
    from matplotlib import image as mpimg

    apply_plot_style()
    fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 4))
    axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax, p in zip(axes_list, paths, strict=False):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(p.stem.replace("_", " "))
    fig.tight_layout()
    save_figure(fig, out_path, root=root)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return resolve_path(str(out_path), root=root)  # type: ignore[return-value]


def barplot_composition(
    composition: pd.DataFrame,
    *,
    title: str = "Cell type composition by sample",
    save: str | Path | None = None,
    show: bool = True,
):
    """Stacked or grouped barplot of a composition table (clusters × samples)."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    composition.T.plot(kind="bar", stacked=True, ax=ax, cmap="tab20", legend=False)
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel("fraction of cells")
    fig.tight_layout()
    if save:
        save_figure(fig, save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def violin_scores(
    adata,
    score_cols: list[str],
    *,
    groupby: str,
    save: str | Path | None = None,
    show: bool = True,
):
    """Violin plots for module score columns."""
    apply_plot_style()
    fig, axes = plt.subplots(1, len(score_cols), figsize=(4 * len(score_cols), 4))
    axes_list = np.atleast_1d(axes)
    for ax, col in zip(axes_list, score_cols, strict=False):
        sc.pl.violin(adata, keys=[col], groupby=groupby, rotation=90, ax=ax, show=False)
    fig.suptitle("Immune state scores", y=1.02)
    fig.tight_layout()
    if save:
        save_figure(fig, save)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
