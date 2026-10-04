"""Shared chart style: one palette, used the same way in every chart.

Colour follows the entity, never the chart:
    Delhi's own pollution + weather  -> BLUE
    stubble-season smoke / fires      -> ORANGE
    Diwali                            -> VIOLET
"""
from pathlib import Path

import matplotlib.pyplot as plt

IMG = Path(__file__).resolve().parents[1] / "images"
IMG.mkdir(exist_ok=True)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, VIOLET, NEUTRAL = "#2a78d6", "#eb6834", "#4a3aa7", "#b9b8b2"
BLUE_LIGHT, ORANGE_LIGHT = "#cde2fb", "#f8c9b4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "Segoe UI", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.labelcolor": INK,
})


def titles(ax, title, subtitle):
    ax.set_title(title, pad=30)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=INK_2, fontsize=10.5, va="bottom")


def footnote(fig, text, y=0.005):
    fig.text(0.01, y, text, color=INK_2, fontsize=8.5)


def save(fig, name):
    fig.savefig(IMG / name, dpi=200, bbox_inches="tight")
