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
