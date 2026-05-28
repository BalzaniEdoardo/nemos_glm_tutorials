import pathlib
from typing import Optional, Union

import pooch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

_NEMOS_ENV = "NEMOS_DATA_DIR"

# ── Dataset registry ──────────────────────────────────────────────────────────
# Add new datasets here; everything else is derived automatically.

DATASETS: dict[str, dict] = {
    "data_RGCs": {
        "files": {
            "SpTimes.mat": "aa0afcb6755fd61ed5dd26c4a8e5b8da91cc13bb3a3640d3294ac37b0193d640",    # replace None with sha256 once known
            "stimtimes.mat": "bdd5cb62a1b7500ebebb2d79beb1bffd97c6d010a2dcce148155fb26806a75e9",
            "Stim.mat": "e6d01592cd08a89740a018294a56d1c94b0254e34ae1a1cf56c468586e22e15e",
        },
        "base_url": (
            "https://raw.githubusercontent.com/"
            "pillowlab/GLMspiketraintutorial_python/main/data_RGCs/"
        ),
    },
}

# Flat registry and per-file URL map derived from DATASETS
REGISTRY_DATA: dict[str, Optional[str]] = {
    fname: fhash
    for ds in DATASETS.values()
    for fname, fhash in ds["files"].items()
}

FILE_URLS: dict[str, str] = {
    fname: ds["base_url"] + fname
    for ds in DATASETS.values()
    for fname in ds["files"]
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _create_retriever(path: Optional[pathlib.Path] = None) -> pooch.Pooch:
    """Create a pooch retriever for fetching datasets.

    Parameters
    ----------
    path :
        Directory where datasets will be stored. Defaults to
        ``pooch.os_cache('nemos_tutorials')``.

    Returns
    -------
    :
        A configured pooch retriever.
    """
    if path is None:
        path = pooch.os_cache("nemos_tutorials")

    return pooch.create(
        path=path,
        # base_url is unused when every file has an explicit entry in `urls`,
        # but pooch requires the argument, so pass an empty string.
        base_url="",
        urls=FILE_URLS,
        registry=REGISTRY_DATA,
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
        env=_NEMOS_ENV,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_data(
    dataset_name: str,
    path: Optional[Union[pathlib.Path, str]] = None,
) -> dict[str, str]:
    """Download all files belonging to a named dataset.

    Parameters
    ----------
    dataset_name :
        Key from ``DATASETS`` (e.g. ``"data_RGCs"``).
    path :
        Directory where files will be stored. Defaults to the pooch cache.

    Returns
    -------
    :
        Mapping of ``{filename: local_path}`` for every file in the dataset.

    Raises
    ------
    ValueError
        If ``dataset_name`` is not found in ``DATASETS``.
    """
    if dataset_name not in DATASETS:
        available = ", ".join(DATASETS)
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Available: {available}"
        )

    retriever = _create_retriever(pathlib.Path(path) if path else None)
    filenames = DATASETS[dataset_name]["files"]

    return {
        fname: retriever.fetch(fname, progressbar=tqdm is not None)
        for fname in filenames
    }