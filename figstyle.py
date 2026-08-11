"""Shared styling so both figures look like they belong to the same paper.

Figures are drawn at their true printed size and never rescaled by LaTeX, which
is what keeps label sizes honest.  Use ``\\includegraphics[width=\\linewidth]``
and pick the matching layout below.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"

# Printed widths of the IEEEconf two-column layout, in inches.  Check yours
# with \the\columnwidth and \the\textwidth if you change the document class.
COLUMN_WIDTH = 3.5   # \columnwidth  -> \begin{figure}
TEXT_WIDTH = 7.16    # \textwidth    -> \begin{figure*}

# Sequential map for activation time.  `YlGnBu` is a ColourBrewer sequential,
# monotone in lightness and colour-vision-deficiency safe, and sits in the same
# soft blue-green register as the `tab20c` node colours of Fig. 3.  Trimming the
# ends keeps the pale end clearly darker than the white inactive nodes and stops
# the dark end from going near-black.
TIME_CMAP_NAME = "YlGnBu"
TIME_CMAP_RANGE = (0.22, 0.80)

INACTIVE_FACE = "white"     # nodes that never activate
INACTIVE_EDGE = "#9a9a9a"
NODE_EDGE = "white"         # ring separating overlapping nodes
SEED_EDGE = "#000000"
EDGE_COLOR = "#8c8c8c"      # graph edges
GRID_COLOR = "#dcdcdc"


def use_paper_style(base_font=8.0):
    """Apply rcParams matching an IEEE two-column paper set in Times."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_font,
        "axes.labelsize": base_font,
        "axes.titlesize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,    # embed real TrueType, not Type 3
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep SVG labels as editable text, not outlines
    })


def time_cmap():
    """The truncated sequential colormap used for activation times."""
    lo, hi = TIME_CMAP_RANGE
    base = plt.get_cmap(TIME_CMAP_NAME)
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{TIME_CMAP_NAME}_trunc", base(np.linspace(lo, hi, 256)))


def node_palette(n):
    """`n` colours identifying individual nodes, in the order they are indexed.

    All of `tab20c`, then the two lighter shades of each `tab20b` family: 30
    distinct colours in one soft register, enough for the network used in the
    paper without the repeats that `tab20c` alone would give.
    """
    shades = [plt.get_cmap("tab20c")(i) for i in range(20)]
    shades += [plt.get_cmap("tab20b")(i) for i in range(20) if i % 4 >= 2]
    return np.array([shades[i % len(shades)] for i in range(n)])


def save(fig, name, dpi=300):
    """Write `name` as PDF for LaTeX, SVG for editing, and PNG for previewing.

    All three come out at the same printed size.  The SVG keeps its labels as
    live text so they can be restyled in a vector editor; see the README on
    fonts if your editor substitutes one.
    """
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.svg")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=dpi)
    print(f"  wrote figures/{name}.pdf, .svg and .png")
