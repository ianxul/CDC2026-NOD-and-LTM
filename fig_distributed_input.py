"""Figure 3: cascades in the NOD model from distributed subthreshold inputs.

The same network as Fig. 2, but with uniform thresholds so every node shares
one activation threshold ``b*``.  A step input of half that threshold is applied
to all nodes at once (case b) and with staggered onsets (case c).  Simultaneous
subthreshold input cascades the group; the same input, spread out in time, does
not.  Nothing in the LTM distinguishes these two cases.

Run ``python fig_distributed_input.py`` to write both layouts to ``figures/``.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


import figstyle as fs
import ltm_nod as ln
from fig_cascades import LAYOUT_SEED

THRESHOLD = 0.5          # uniform LTM threshold, tau_i
K = 1.1                  # nonlinear self-reinforcement, k >= 1
GAMMA_FRACTION = 0.1     # social coupling, as a fraction of gamma_max
AMPLITUDE = 0.01         # input height; less than half of b*
ONSET, DURATION = 25.0, 50.0
MAX_DELAY = 50.0         # case c staggers onsets over [ONSET, ONSET + MAX_DELAY]
DELAY_SEED = 0           # the original run was unseeded; see README
T_MAX, T_SHOWN = 1000.0, 200.0

# Panel letters are left to be set by hand.  Their line is reserved either way,
# so switching this on changes what is drawn but not where anything sits.
PANEL_LABELS = False

# Network on the left, the two input cases stacked on the right.  Spacing is
# left to matplotlib's constrained layout, which measures the text: hand-tuned
# margins break as soon as a label changes length.
LAYOUTS = {
    # \begin{figure} in one column.
    "column": dict(figsize=(fs.COLUMN_WIDTH, 2.45), width_ratios=[1.55, 1.45],
                   font=7.0, net_width=1.78, case_gap=0.02,
                   xticks=[0, 100, 200], minor_xticks=[50, 150]),
    # \begin{figure*} spanning both columns, with room for wider time axes.
    "wide": dict(figsize=(fs.TEXT_WIDTH, 2.55), width_ratios=[1, 1.85],
                 font=8.0, net_width=2.4, case_gap=0.02,
                 xticks=[0, 50, 100, 150, 200], minor_xticks=[]),
}


def step_input(onsets, duration=DURATION, amplitude=AMPLITUDE):
    """A square pulse of `duration` on each node, starting at its own onset."""
    def b(t):
        return amplitude * ((t >= onsets) & (t < onsets + duration))
    return b


def run():
    """Simulate both cases and return everything the figure needs."""
    A, _ = ln.load_instance()
    n = A.shape[0]
    tau = np.full(n, THRESHOLD)

    gamma = GAMMA_FRACTION * ln.gamma_max(tau, K)
    mu = ln.nod_relaxation(tau, K, gamma)
    threshold = ln.b_star(mu, K)[0]          # uniform tau => uniform b*
    print(f"  gamma={gamma:.4f}, mu={mu[0]:.4f}, b*={threshold:.4f}, "
          f"input={AMPLITUDE} ({AMPLITUDE / threshold:.0%} of b*)")

    delays = np.random.default_rng(DELAY_SEED).uniform(0.0, MAX_DELAY, n)
    cases = []
    for name, onsets in [("simultaneous", np.full(n, ONSET)),
                         ("staggered", ONSET + delays)]:
        b = step_input(onsets)
        sol = ln.simulate_nod(A, mu, K, gamma, b=b, t_span=(0.0, T_MAX),
                              method="Radau", t_eval=np.linspace(0, T_MAX, 4001))
        cases.append((name, sol, b))
        print(f"  {name}: {np.count_nonzero(sol.y[:, -1] > 0.95)}/{n} "
              f"nodes reach z = 1")

    G = nx.from_numpy_array(A)
    return G, nx.spring_layout(G, seed=LAYOUT_SEED), cases, threshold


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def draw_network(ax, G, pos, colors, panel_width):
    scale = panel_width / 1.5
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=fs.EDGE_COLOR, width=0.15 * scale)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=(8.6 * scale) ** 2,
                           node_color=colors, edgecolors=fs.NODE_EDGE,
                           linewidths=0.5 * scale)
    ax.set_aspect("equal")
    ax.set_anchor("C")           # center the drawing vertically against the traces
    ax.margins(0.04)
    ax.axis("off")


def draw_case(ax_z, ax_b, sol, b, colors, threshold, line_width, ticks,
              minor_ticks):
    """Opinion trajectories over the input that produced them."""
    for z_i, color in zip(sol.y, colors):
        ax_z.plot(sol.t, z_i, color=color, lw=line_width)
    ax_z.set_ylim(-0.02, 1.1)
    ax_z.set_yticks([0.0, 0.5, 1.0])
    ax_z.set_ylabel(r"$z(t)$")

    inputs = np.array([b(t) for t in sol.t])          # (time, node)
    for input_i, color in zip(inputs.T, colors):
        ax_b.plot(sol.t, input_i, color=color, lw=line_width)
    ax_b.axhline(threshold, color="red", ls="--", lw=line_width * 0.8)
    # Hung off the dashed line by a fixed offset in points, so the superscript
    # star clears the line by the same amount in both layouts.
    ax_b.annotate(r"$b^*$", xy=(0.985 * T_SHOWN, threshold),
                  xytext=(0, -2.5), textcoords="offset points", color="red",
                  ha="right", va="top")
    ax_b.set_ylim(-0.0012, threshold * 1.14)
    ax_b.set_yticks([0.0, 0.01, 0.02])
    ax_b.set_ylabel(r"$b(t)$")
    # No per-case x label: one shared label sits under the column instead, so
    # neither case spends a text line on it and the two stay the same height.

    for ax in (ax_z, ax_b):
        ax.set_xlim(0, T_SHOWN)
        ax.set_xticks(ticks)
        ax.set_xticks(minor_ticks, minor=True)
    ax_z.tick_params(labelbottom=False)


def make_figure(G, pos, cases, threshold, layout, labels=True):
    """Assemble one named layout.  Returns the figure.

    With ``labels=False`` the panel letters are drawn as blanks rather than
    dropped, so the two versions have identical geometry and letters added by
    hand afterwards land exactly where these ones would.
    """
    spec = LAYOUTS[layout]
    # One consistent colour per node across the network and the z(t) traces.
    cmap = plt.get_cmap("tab20c")
    colors = [cmap(i % cmap.N) for i in range(len(pos))]
    titles = [f"{letter})" if labels else " " for letter in "abc"]
    line_width = 0.62 * spec["net_width"] / 1.5

    with mpl.rc_context({"font.size": spec["font"],
                         "axes.labelsize": spec["font"],
                         "axes.titlesize": spec["font"],
                         "xtick.labelsize": spec["font"] - 1,
                         "ytick.labelsize": spec["font"] - 1,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.edgecolor": "black"}):
        fig = plt.figure(figsize=spec["figsize"], layout="constrained")
        fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.0, hspace=0.0)
        # Subfigures rather than one grid, so the shared "Time" label can be
        # centred under the traces alone and not under the network as well.
        net_fig, cases_fig = fig.subfigures(1, 2, width_ratios=spec["width_ratios"],
                                            wspace=0.04)

        ax_net = net_fig.add_subplot()
        draw_network(ax_net, G, pos, colors, spec["net_width"])
        ax_net.set_title(titles[0], loc="left", fontweight="bold", pad=2)

        cases_grid = cases_fig.add_gridspec(2, 1, hspace=spec["case_gap"])
        trace_axes = []
        for i, ((name, sol, b), title) in enumerate(zip(cases, titles[1:])):
            block = cases_grid[i].subgridspec(2, 1, height_ratios=[1.7, 1], hspace=0.06)
            ax_z = cases_fig.add_subplot(block[0])
            ax_b = cases_fig.add_subplot(block[1], sharex=ax_z)
            draw_case(ax_z, ax_b, sol, b, colors, threshold, line_width,
                      spec["xticks"], spec["minor_xticks"])
            ax_z.set_title(title, loc="left", fontweight="bold", pad=2)
            trace_axes += [ax_z, ax_b]

        cases_fig.supxlabel("Time", fontsize=spec["font"])
        cases_fig.align_ylabels(trace_axes)

        # Constrained layout centres the shared label on the sub-figure, which
        # includes the y labels, leaving it left of the time axis.  Let the
        # layout settle, then re-centre it on the axes themselves.  Only the
        # vertical placement is recomputed on later draws, so this x sticks.
        fig.draw_without_rendering()
        box = trace_axes[-1].get_position()
        cases_fig.supxlabel("Time", x=0.5 * (box.x0 + box.x1),
                            fontsize=spec["font"])
    return fig


if __name__ == "__main__":
    fs.use_paper_style()
    print("Simulating distributed inputs...")
    G, pos, cases, threshold = run()
    print("Rendering...")
    for name in LAYOUTS:
        fs.save(make_figure(G, pos, cases, threshold, name, PANEL_LABELS),
                f"fig3_distributed_input_{name}")
