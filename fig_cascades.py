"""Figure 2: one LTM cascade and two NOD relaxations of it.

Panel A is the discrete cascade; B and C are NOD relaxations of the same
instance with small and large social coupling.  B recovers the LTM cascade set
exactly (Theorem 2), C overshoots it to a full cascade.

Node colour is activation time *normalised by the last activation in that
panel*, so that a single colour bar can serve all three: the panels run on
wildly different clocks (a handful of discrete steps against hundreds of time
units of continuous flow), and only the ordering is comparable across them.
Each panel reports its own final time.

Run ``python fig_cascades.py`` to write all three layouts to ``figures/``.
"""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import figstyle as fs
import ltm_nod as ln

SEEDS = [0, 1, 2, 3]
K = 1.1                       # nonlinear self-reinforcement, k >= 1
GAMMA_FRACTIONS = (0.1, 0.8)  # social coupling, as a fraction of gamma_max
T_MAX = 1000.0                # integration horizon for the NOD runs
LAYOUT_SEED = 42              # spring layout, fixed so all panels share it

# One entry per printed size.  `panel_width` is the drawn width of a single
# network in inches and sets node and edge scaling; `cbar` is a figure-fraction
# rectangle.  Adjust these, not the drawing code, to retarget a new template.
LAYOUTS = {
    # \begin{figure} in one column, panels side by side.  The most compact
    # option, and the one the paper uses.
    "column": dict(
        figsize=(fs.COLUMN_WIDTH, 1.48), nrows=1, ncols=3, panel_width=1.03,
        font=6.0, inside=False, compact=True, headroom=0.0, rotate=False,
        grid=dict(left=0.004, right=0.875, top=0.862, bottom=0.095, wspace=0.02),
        cbar=(0.905, 0.115, 0.024, 0.72), orientation="vertical",
        legend=dict(bbox_to_anchor=(0.44, -0.01), ncol=2),
    ),
    # \begin{figure*} spanning both columns.  Roomiest, but a tall block.
    "wide": dict(
        figsize=(fs.TEXT_WIDTH, 2.99), nrows=1, ncols=3, panel_width=2.2,
        font=8.0, inside=False, compact=False, headroom=0.0, rotate=False,
        grid=dict(left=0.004, right=0.925, top=0.893, bottom=0.056, wspace=0.02),
        cbar=(0.945, 0.075, 0.010, 0.74), orientation="vertical",
        legend=dict(bbox_to_anchor=(0.46, -0.008), ncol=2),
    ),
    # \begin{figure} in one column, panels stacked.  Rotated to lie along the
    # layout's long axis, without which three networks would not fit a column.
    "stacked": dict(
        figsize=(fs.COLUMN_WIDTH, 6.5), nrows=3, ncols=1, panel_width=3.0,
        font=8.0, inside=True, compact=False, headroom=0.15, rotate=True,
        grid=dict(left=0.004, right=0.895, top=0.997, bottom=0.042, hspace=0.02),
        cbar=(0.925, 0.30, 0.020, 0.40), orientation="vertical",
        legend=dict(bbox_to_anchor=(0.45, -0.004), ncol=2),
    ),
}


def run():
    """Simulate the three panels and return ``(G, pos, panels)``."""
    A, tau = ln.load_instance()
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=LAYOUT_SEED)

    panels = [("LTM", None, ln.simulate_ltm(A, tau, SEEDS))]
    for fraction in GAMMA_FRACTIONS:
        gamma = fraction * ln.gamma_max(tau, K)
        mu = ln.nod_relaxation(tau, K, gamma)
        sol = ln.simulate_nod(A, mu, K, gamma, SEEDS, t_span=(0.0, T_MAX))
        panels.append(("NOD", gamma, ln.activation_times(sol, SEEDS)))

    for letter, (name, gamma, times) in zip("ABC", panels):
        detail = "" if gamma is None else f", gamma={gamma:.4f}"
        print(f"  {letter}) {name}{detail}: "
              f"{np.count_nonzero(~np.isnan(times))}/{len(times)} active, "
              f"t_final={np.nanmax(times):.4g}")
    return G, pos, panels


def principal_axis_layout(pos):
    """Rotate a layout so its long axis is horizontal.

    A graph drawing has no preferred orientation, and lying the network down is
    what lets the stacked layout fit three networks into one column.  The other
    layouts keep the upright drawing shared with Fig. 3.
    """
    P = np.array([pos[i] for i in sorted(pos)])
    P = P - P.mean(axis=0)
    _, _, axes = np.linalg.svd(P, full_matrices=False)
    return dict(zip(sorted(pos), P @ axes.T))


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def draw_panel(ax, G, pos, times, cmap, panel_width, headroom=0.0):
    """Draw one network, coloured by activation time normalised to [0, 1]."""
    scale = panel_width / 2.25                    # 2.25 in is the design width
    node_size = (9.7 * scale) ** 2                # marker area, points^2
    ring = 0.75 * scale

    active = ~np.isnan(times)
    face = np.where(active[:, None],
                    cmap(np.where(active, times / np.nanmax(times), 0.0)),
                    mcolors.to_rgba(fs.INACTIVE_FACE))
    edge = np.where(active[:, None], mcolors.to_rgba(fs.NODE_EDGE),
                    mcolors.to_rgba(fs.INACTIVE_EDGE))

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=fs.EDGE_COLOR,
                           width=0.6 * scale)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=face,
                           edgecolors=edge, linewidths=ring)
    # Seeds keep their t = 0 colour but gain a heavy black ring, so they read
    # as the input to the cascade rather than just its earliest nodes.
    nx.draw_networkx_nodes(G, pos, nodelist=SEEDS, ax=ax, node_size=node_size,
                           node_color=[cmap(0.0)] * len(SEEDS),
                           edgecolors=fs.SEED_EDGE, linewidths=2.4 * ring)

    ax.set_aspect("equal")
    ax.margins(0.04)
    ax.axis("off")
    if headroom:                                  # clear space for an in-axes label
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + headroom * (hi - lo))


def label_panel(ax, letter, name, gamma, times, inside=False, compact=False):
    """Panel letter, model, and the clock that panel ran on."""
    head = rf"$\bf{{{letter}}}$   {name}"
    if gamma is not None:
        head += rf"$,\ \gamma = {gamma:.4f}$"

    n_active = np.count_nonzero(~np.isnan(times))
    t_final = f"{np.nanmax(times):.4g}"
    tail = (rf"{n_active}/{len(times)},  $t_{{\mathrm{{f}}}} = {t_final}$" if compact
            else rf"{n_active}/{len(times)} active,  "
                 rf"$t_{{\mathrm{{final}}}} = {t_final}$")

    if inside:
        ax.text(0.0, 0.995, f"{head}\n{tail}", transform=ax.transAxes,
                ha="left", va="top", linespacing=1.45)
    else:
        ax.set_title(f"{head}\n{tail}", linespacing=1.45, pad=3)


def legend_handles():
    marker = dict(marker="o", linestyle="none", markersize=5.0)
    return [
        Line2D([], [], markerfacecolor=fs.INACTIVE_FACE, markeredgecolor=fs.INACTIVE_EDGE,
               markeredgewidth=0.7, label="never activates", **marker),
        Line2D([], [], markerfacecolor="white", markeredgecolor=fs.SEED_EDGE,
               markeredgewidth=1.4, label="seed set", **marker),
    ]


def make_figure(G, pos, panels, layout):
    """Assemble one named layout.  Returns the figure."""
    spec = LAYOUTS[layout]
    cmap = fs.time_cmap()
    if spec["rotate"]:
        pos = principal_axis_layout(pos)

    with mpl.rc_context({"font.size": spec["font"],
                         "axes.titlesize": spec["font"],
                         "axes.labelsize": spec["font"],
                         "legend.fontsize": spec["font"] - 0.5,
                         "xtick.labelsize": spec["font"] - 1,
                         "ytick.labelsize": spec["font"] - 1}):
        fig = plt.figure(figsize=spec["figsize"])
        gs = GridSpec(spec["nrows"], spec["ncols"], figure=fig, **spec["grid"])

        for i, (letter, (name, gamma, times)) in enumerate(zip("ABC", panels)):
            ax = fig.add_subplot(gs[np.unravel_index(i, (spec["nrows"], spec["ncols"]))])
            draw_panel(ax, G, pos, times, cmap, spec["panel_width"], spec["headroom"])
            label_panel(ax, letter, name, gamma, times, spec["inside"], spec["compact"])

        cax = fig.add_axes(spec["cbar"])
        bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax,
                           orientation=spec["orientation"], ticks=[0, 0.5, 1])
        bar.set_label(r"activation time  $t\,/\,t_{\mathrm{final}}$", labelpad=4,
                      rotation=90 if spec["orientation"] == "vertical" else 0)
        bar.outline.set_linewidth(0.5)
        bar.ax.tick_params(length=2, width=0.5, pad=1.5)

        fig.legend(handles=legend_handles(), loc="lower center", frameon=False,
                   handletextpad=0.3, columnspacing=1.6, borderpad=0.0,
                   **spec["legend"])
    return fig


if __name__ == "__main__":
    fs.use_paper_style()
    print("Simulating cascades...")
    G, pos, panels = run()
    print("Rendering...")
    for name in LAYOUTS:
        fs.save(make_figure(G, pos, panels, name), f"fig2_cascades_{name}")
