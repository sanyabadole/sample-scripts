"""Load and save single-cell data with configurable paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import anndata as ad
import pandas as pd

from gbm_tme.utils import repo_root, resolve_path


def load_metadata_table(path: str | Path, barcode_col: str = "cell_barcode") -> pd.DataFrame:
    """Load a metadata CSV/TSV; ensure barcodes are the index for merging."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    meta = pd.read_csv(p, sep=sep)
    if barcode_col in meta.columns:
        meta = meta.set_index(barcode_col)
    return meta


def merge_obs_metadata(adata: ad.AnnData, meta: pd.DataFrame, how: str = "inner") -> ad.AnnData:
    """Left-join metadata onto ``adata.obs`` on index (barcodes)."""
    overlap = meta.index.intersection(adata.obs_names)
    if len(overlap) == 0:
        raise ValueError("No overlapping barcodes between AnnData.obs_names and metadata index.")
    if how == "inner":
        adata = adata[overlap].copy()
        meta = meta.loc[overlap]
    for col in meta.columns:
        adata.obs[col] = meta.loc[adata.obs_names, col].values
    return adata


def load_raw_anndata(
    paths_cfg: dict[str, Any],
    *,
    root: Path | None = None,
) -> ad.AnnData:
    """Load counts from config ``paths.raw_counts`` block.

    Supported formats: ``h5ad``, ``10x_mtx``, ``10x_h5``.
    """
    raw = paths_cfg.get("raw_counts") or {}
    fmt: Literal["h5ad", "10x_mtx", "10x_h5"] = raw.get("format", "h5ad")
    path = resolve_path(raw.get("path"), root=root)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Raw data path not found: {path!s}. Update configs/paths.yaml after downloading data."
        )
    if fmt == "h5ad":
        return ad.read_h5ad(path)
    if fmt == "10x_mtx":
        import scanpy as sc

        return sc.read_10x_mtx(path.parent if path.is_file() else path)
    if fmt == "10x_h5":
        import scanpy as sc

        return sc.read_10x_h5(path)
    raise ValueError(f"Unsupported raw_counts.format: {fmt}")


def attach_metadata_if_configured(adata: ad.AnnData, paths_cfg: dict[str, Any], *, root: Path | None = None) -> ad.AnnData:
    """Merge optional metadata table from ``paths.metadata_table``."""
    meta_path = paths_cfg.get("metadata_table")
    if not meta_path:
        return adata
    mp = resolve_path(str(meta_path), root=root)
    if mp is None or not mp.exists():
        raise FileNotFoundError(f"metadata_table not found: {mp}")
    meta = load_metadata_table(mp)
    return merge_obs_metadata(adata, meta, how="inner")


def ensure_sparse_X(adata: ad.AnnData) -> None:
    """Ensure ``adata.X`` is sparse matrix for memory efficiency (in-place)."""
    import scipy.sparse as sp

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)


def write_h5ad(adata: ad.AnnData, path: str | Path, *, root: Path | None = None) -> Path:
    """Write AnnData to ``path`` (resolved relative to repo root if relative)."""
    out = resolve_path(str(path), root=root)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    return out


def processed_filepath(paths_cfg: dict[str, Any], key: str, *, root: Path | None = None) -> Path:
    """Resolve ``data/processed`` filename from ``paths.filenames``."""
    root = root or repo_root()
    processed = resolve_path(paths_cfg.get("processed_dir", "data/processed"), root=root)
    assert processed is not None
    names = paths_cfg.get("filenames") or {}
    fname = names.get(key)
    if not fname:
        raise KeyError(f"Unknown processed artifact key: {key}")
    return processed / fname
