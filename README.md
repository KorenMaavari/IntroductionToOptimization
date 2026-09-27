# Assignment 3 - Koren Ma'avari (207987314) & Sagi Roland (211548839)

## files
- `bfgs.py`: inverse-hessian BFGS and a weak-wolfe bracketing/zoom line search
- `rosenbrock.py`: rosenbrock objective, gradient, experiment and convergence plot
- `neural_network.py`: the 2-4-3-1 network, packing/unpacking, manual backpropagation,
  dataset routines and initialization
- `visualization.py`: the required surface/scatter visualization routine
- `neural_network_experiment.py`: all four tolerance experiments
- `gradient_checks.py`: centered finite-difference checks and comparison between
  the literal per-example average and the vectorized batch gradient
- `run_all.py`: reproduces every result and plot

## reproduction
```bash
python run_all.py
python gradient_checks.py
```