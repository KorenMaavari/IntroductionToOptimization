"""numerical validation of all analytical gradients used"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from neural_network import (
    dataset_gradient,
    dataset_gradient_via_examples,
    dataset_loss,
    finite_difference_gradient,
    generate_training_dataset,
    initialize_parameters,
    pack_parameters,
    single_example_loss_and_gradient,
)
from rosenbrock import rosenbrock, rosenbrock_grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(a), np.linalg.norm(b)))


def main() -> None:
    metrics = {}

    rng = np.random.default_rng(123)
    x = rng.normal(size=10)
    g_fd = finite_difference_gradient(rosenbrock, x)
    g = rosenbrock_grad(x)
    metrics["rosenbrock_gradient_relative_error"] = relative_error(g, g_fd)

    w = pack_parameters(initialize_parameters(seed=12))
    xi = np.array([0.3, -0.7])
    yi = 0.12
    single_fun = lambda v: single_example_loss_and_gradient(xi, yi, v)[0]
    g_fd_single = finite_difference_gradient(single_fun, w, step=1e-6)
    _, g_single = single_example_loss_and_gradient(xi, yi, w)
    metrics["single_example_gradient_relative_error"] = relative_error(g_single, g_fd_single)

    X, y = generate_training_dataset(8, seed=11)
    full_fun = lambda v: dataset_loss(v, X, y)
    g_fd_full = finite_difference_gradient(full_fun, w, step=1e-6)
    g_full = dataset_gradient(w, X, y)
    metrics["full_dataset_gradient_relative_error"] = relative_error(g_full, g_fd_full)

    g_literal = dataset_gradient_via_examples(w, X, y)
    metrics["vectorized_vs_explicit_average_relative_error"] = relative_error(g_full, g_literal)

    out = Path(__file__).resolve().parent / "results" / "gradient_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")


if __name__ == "__main__":
    main()
