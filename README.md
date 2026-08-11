# A Continuous Relaxation of the Linear Threshold Model with Nonlinear Opinion Dynamics

Reference implementation and figure code for

> I. X. Belaustegui, H. Sinhmar, L.-W. Kong, A. M. Hein and N. E. Leonard,
> *A Continuous Time and State-Space Relaxation of the Linear Threshold Model
> with Nonlinear Opinion Dynamics*.

The paper replaces the step-function thresholds of the Linear Threshold Model (LTM) with the saddle-node bifurcations of Nonlinear Opinion Dynamics (NOD), turning a discrete cascade process into a continuous flow. This repository contains that model, the 30-node LTM instance used in the paper, and the scripts that generate the two simulation figures.

## What is here

| File | Contents |
| --- | --- |
| `ltm_nod.py` | The models: LTM cascades, the NOD vector field, and the map between them. Every other script builds on this one. |
| `fig_cascades.py` | Fig. 2: an LTM cascade beside two NOD relaxations of it, one recovering it exactly and one overshooting to a full cascade. |
| `fig_distributed_input.py` | Fig. 3: the same network driven by subthreshold input, which cascades when it arrives at once and does nothing when spread over time. |
| `figstyle.py` | Shared colours, fonts and printed sizes, so both figures match the paper and each other. |
| `test_ltm_nod.py` | Numerical checks of Theorems 1 and 2, and of the Fig. 3 claim. |
| `data/` | The instance `(A, tau)` used for every figure. |
| `figures/` | Generated output. |

## License

MIT, see `LICENSE`.
