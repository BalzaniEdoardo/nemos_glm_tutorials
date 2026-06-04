"""Shared plotting helpers for the NeMoS tutorials."""

import matplotlib.pyplot as plt
import numpy as np

# Soft, qualitative palette reused across the whole tutorial series.
PALETTE = plt.cm.Pastel1.colors


def plot_counts(
    counts, ep, predictions=(), title="", ylabel="spike count", ylim=None, ax=None
):
    """Show binned spike counts over `ep`, optionally overlaying model predictions.

    The counts are drawn as a soft gray filled step (so zero-count bins still
    show a baseline), and each prediction as a line. Line colors default to a
    pastel palette, cycling in order; pass an explicit color to override.

    Parameters
    ----------
    counts:
        Tsd of binned spike counts.
    ep:
        ``(start, end)`` interval in seconds, forwarded to ``.get``.
    predictions:
        Optional list of ``(tsd, label)`` tuples. Optionally append a color (or
        ``None`` to keep cycling the palette) and a linestyle: ``(tsd, label,
        color, linestyle)``.
    title, ylabel, ylim:
        Standard axis cosmetics.
    ax:
        Axes to draw on. A new figure is created when ``None``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    c = counts.get(*ep)
    half = 0.5 / counts.rate
    edges = np.append(c.t - half, c.t[-1] + half)
    ax.stairs(
        c.d, edges, fill=True, facecolor="0.88", edgecolor="0.55",
        linewidth=0.8, label="spike count", zorder=1,
    )

    color_idx = 0
    for pred, label, *style in predictions:
        explicit = bool(style) and style[0] is not None
        color = style[0] if explicit else PALETTE[color_idx % len(PALETTE)]
        linestyle = style[1] if len(style) > 1 else "-"
        ax.plot(pred.get(*ep), color=color, linestyle=linestyle, linewidth=2,
                label=label, zorder=3)
        if not explicit:
            color_idx += 1

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (s)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, framealpha=0.6)
    return ax


def plot_design_matrix(split, counts=None, rows=None, title="design matrix"):
    """Display a design matrix with each feature's lags ordered oldest-first.

    NeMoS' ``HistoryConv`` returns each feature block with the *most recent* lag
    in its first column. That makes the raw image read right-to-left in time and,
    for the spike-history block, lines the most recent lag up against the response
    counts. For a readable picture we want the opposite — most lagged column
    first — so the image reads left (past) to right (present), matching the
    original tutorials.

    Reversing the whole matrix would mix the stimulus and spike-history blocks, so
    instead we flip each feature block on its own basis axis (operating on the
    ``split_by_feature`` dict the notebook already built) and stitch them back.

    Parameters
    ----------
    split:
        Mapping ``feature -> block`` from ``basis.split_by_feature(design, axis=1)``.
        Each block's last axis is the basis (lag) axis; extra axes (e.g. the
        per-neuron axis of a coupling block) are flattened into the columns.
    counts:
        Optional response counts to show as a thin panel on the right, sharing the
        color scale. When ``None`` only the design matrix is drawn.
    rows:
        Optional row ``slice`` applied to every block (and to ``counts``), e.g. to
        skip the NaN-padded burn-in and zoom into a short window.
    title:
        Title for the design-matrix panel.
    """
    if rows is None:
        rows = slice(None)

    # Flip the basis (last) axis of each feature block on its own, flatten any
    # extra axes into the column dimension, then stack the blocks side by side.
    blocks = []
    for block in split.values():
        block = block[rows]
        blocks.append(block[..., ::-1].reshape((block.shape[0], -1)))
    reordered = np.hstack(blocks)

    if counts is None:
        _, ax = plt.subplots(figsize=[12, 8])
        ax.imshow(reordered, aspect="auto", interpolation="nearest")
        ax.set_xlabel("regressor")
        ax.set_ylabel("time bin of response")
        ax.set_title(title)
        return ax

    counts = counts[rows]
    vmin = min(reordered.min(), counts.min())
    vmax = max(reordered.max(), counts.max())

    fig = plt.figure(figsize=[12, 8])
    ax_design = plt.subplot(1, 10, (1, 9))
    ax_design.imshow(
        reordered, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax
    )
    ax_design.set_xlabel("regressor")
    ax_design.set_ylabel("time bin of response")
    ax_design.set_title(title)

    ax_counts = plt.subplot(1, 10, 10)
    ax_counts.imshow(
        counts[:, None], aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax
    )
    ax_counts.set_yticks([])
    ax_counts.set_title("spike count")
    return fig
