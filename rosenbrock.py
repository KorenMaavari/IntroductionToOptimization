"""rosenbrock objective, gradient, and assignment experiment"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from bfgs import BFGSResult, bfgs


def _validate_input(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Rosenbrock input must be a one-dimensional vector")
    if x.size < 2:
        raise ValueError("Rosenbrock input must contain at least two variables")
    return x


def rosenbrock(x: np.ndarray) -> float:
    x = _validate_input(x)
    return float(
        np.sum((1.0 - x[:-1]) ** 2 + 100.0 * (x[1:] - x[:-1] ** 2) ** 2)
    )


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    x = _validate_input(x)
    g = np.zeros_like(x)
    g[:-1] += (
        2.0 * (x[:-1] - 1.0)
        - 400.0 * x[:-1] * (x[1:] - x[:-1] ** 2)
    )
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


def run_rosenbrock_experiment(
    output_dir: str | Path,
) -> Tuple[BFGSResult, Dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x0 = np.zeros(10)
    result = bfgs(
        rosenbrock,
        rosenbrock_grad,
        x0,
        tol=1e-5,
        max_iter=3000,
        alpha0=1.0,
        beta=0.5,
        c1=0.25,
        c2=0.9,
    )

    values = np.asarray(result.f_history)
    plotted = np.maximum(values, np.finfo(float).tiny)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(np.arange(values.size), plotted)
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"$f(x_k)-f^*$")
    ax.set_title("BFGS convergence on the 10-dimensional Rosenbrock function")
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "rosenbrock_convergence.png", dpi=180)
    plt.close(fig)

    metrics = {
        "iterations": float(result.nit),
        "objective": float(result.fun),
        "gradient_norm": float(np.linalg.norm(result.grad)),
        "distance_to_ones": float(np.linalg.norm(result.x - np.ones(10))),
        "all_wolfe_steps": float(all(result.wolfe_history)),
    }
    return result, metrics
