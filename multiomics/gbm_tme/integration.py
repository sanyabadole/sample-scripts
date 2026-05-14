"""scVI integration: latent space, neighbors, UMAP, Leiden."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import scanpy as sc

from gbm_tme.utils import repo_root, resolve_path, set_seed


def _require_batch_key(adata: ad.AnnData, batch_key: str) -> None:
    if batch_key not in adata.obs.columns:
        raise KeyError(
            f"batch_key {batch_key!r} not in adata.obs. "
            "Set params.integration.batch_key to an existing metadata column (e.g. patient_id)."
        )


def train_scvi_and_latent(
    adata: ad.AnnData,
    params: dict[str, Any],
    *,
    model_dir: str | Path | None = None,
    random_seed: int | None = None,
) -> tuple[Any, ad.AnnData]:
    """Train scVI with patient/sample batch key; store latent in ``obsm``."""
    import scvi

    integ = params.get("integration", {})
    batch_key = str(integ.get("batch_key", "patient_id"))
    _require_batch_key(adata, batch_key)

    seed = int(random_seed if random_seed is not None else params.get("random_seed", 0))
    set_seed(seed)
    scvi.settings.seed = seed

    layer = integ.get("layer")
    kwargs = dict(batch_key=batch_key)
    if layer:
        scvi.model.SCVI.setup_anndata(adata, layer=str(layer), **kwargs)
    else:
        scvi.model.SCVI.setup_anndata(adata, **kwargs)

    model = scvi.model.SCVI(
        adata,
        n_latent=int(integ.get("scvi_latent_dim", 30)),
        n_layers=int(integ.get("scvi_n_layers", 2)),
        n_hidden=int(integ.get("scvi_n_hidden", 128)),
        dropout_rate=float(integ.get("scvi_dropout_rate", 0.1)),
        gene_likelihood=str(integ.get("scvi_gene_likelihood", "zinb")),
    )
    model.train(
        max_epochs=int(integ.get("scvi_epochs", 400)),
        train_size=float(integ.get("scvi_train_size", 0.9)),
        early_stopping=bool(integ.get("scvi_early_stopping", True)),
        accelerator="auto",
    )

    latent_key = str(integ.get("latent_key", "X_scVI"))
    adata.obsm[latent_key] = model.get_latent_representation()

    if model_dir:
        out = resolve_path(str(model_dir), root=repo_root())
        assert out is not None
        out.mkdir(parents=True, exist_ok=True)
        model.save(out, overwrite=True)

    return model, adata


def neighbors_umap_leiden_from_latent(
    adata: ad.AnnData,
    params: dict[str, Any],
    *,
    random_seed: int | None = None,
) -> ad.AnnData:
    """Compute neighbors / UMAP / Leiden using scVI latent coordinates."""
    integ = params.get("integration", {})
    latent_key = str(integ.get("latent_key", "X_scVI"))
    if latent_key not in adata.obsm:
        raise KeyError(f"Latent key {latent_key} missing; run train_scvi_and_latent first.")

    neigh_key = str(integ.get("neighbors_key", "neighbors_scVI"))
    umap_key = str(integ.get("umap_key", "X_umap_scVI"))
    seed = int(random_seed if random_seed is not None else params.get("random_seed", 0))
    set_seed(seed)

    neigh = params.get("neighbors", {})
    sc.pp.neighbors(
        adata,
        use_rep=latent_key,
        n_neighbors=int(neigh.get("n_neighbors", 15)),
        key_added=neigh_key,
    )
    umap_cfg = params.get("umap", {})
    sc.tl.umap(adata, min_dist=float(umap_cfg.get("min_dist", 0.3)), random_state=seed, neighbors_key=neigh_key)

    clust = params.get("clustering", {})
    resolution = float(clust.get("leiden_resolution", 0.6))
    key_added = str(clust.get("key_added", "leiden"))
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added, neighbors_key=neigh_key)

    adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
    return adata
