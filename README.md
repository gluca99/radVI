# $\texttt{radVI}$: Variational inference via radial transport
$\texttt{radVI}$ is a variational inference framework that optimizes over radial transport maps to better capture the true radial profiles of high-dimensional distributions. It addresses the tail-behavior limitations of standard methods by serving as a simple add-on to Gaussian approximations like Laplace or Gaussian VI. By learning a non-linear radial transformation, $\texttt{radVI}$ enables accurate tail modeling and parameter estimation at minimal computational cost. This repository accompanies our AISTATS 2026 paper.

## Installation

Requires Python 3.10.

### Using Conda (recommended)
1) Create and activate the environment:
```bash
conda env create -f conda_env_requirements.yml
conda activate radvi
```

### Using pip
1) Create and activate a virtual environment with Python 3.10, then install requirements:
```bash
pip install -r pip_requirements.txt
```

## Running Examples

The `examples/` directory contains Python notebooks that reproduce the experiments and figures from the paper. You can run these scripts directly:

Each notebook demonstrates how to:
- Run the complete experiments with the same parameters as in the paper
- Generate and display the figures
- Optionally save plots to the `examples/plots/` directory

The notebooks are fully self-contained and will handle all the necessary setup.

## Repository Structure
```
radVI/
│
├── examples/        # Example notebooks and scripts to run radVI
│   ├── visualize_isotropic.ipynb             -> runs radVI for isotropic distributions
│   ├── visualize_anisotropic.ipynb           -> runs radVI for anisotropic settings using whitening
│   ├── visualize_funnel.ipynb                -> runs radVI for the Neal's Funnel example
│   ├── visualize_w2_convergence_alpha.ipynb  -> demonstrates robustness of radVI to α
│   └── visualize_w2_convergence.ipynb        -> visualize W2 error between analytical & learned maps
│
├── utils/           # Helper utilities
│   ├── basis_functions.py                    -> functions for constructing the radial basis dictionary Ψ(r)
│   ├── elliptical_distributions.py           -> elliptical distributions, with methods for
│   │                                            V, ∇V, ∇²V, and sampling
│   ├── importance_sampling.py                -> functions to run importance sampling on Gaussian VI and radVI
│   ├── integrals.py                          -> functions for known integrals of the chi_d distribution
│   ├── neals_funnel_helpers.py               -> functions specific for running examples/visualize_funnel.ipynb
│   ├── plotting.py                           -> functions for plotting the results of the example ipynb files
│   ├── transport_maps.py                     -> analytical optimal transport maps
│   └── wasserstein_distance.py               -> function to compute W2 distance between radial distributions
│
└── VI_solvers/      # Variational inference algorithms
    ├── rad_vi.py                             -> radVI algorithm
    └── other VI methods                      -> Laplace, Gaussian MFVI, Gaussian FB-GVI
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ghafourpour2026radvi,
  title={Variational inference via radial transport},
  author={Ghafourpour, Luca and Chewi, Sinho and Figalli, Alessio and Pooladian, Aram-Alexandre},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  organization={PMLR},
  year={2026}
}
```

## Contact

For questions and support, please contact Luca Ghafourpour (ldg34@cam.ac.uk).