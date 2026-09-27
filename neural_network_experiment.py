"""run the neural-network portion"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from bfgs import BFGSResult, bfgs
from neural_network import (
    dataset_gradient,
    dataset_loss,
    forward_batch,
    generate_test_dataset,
    generate_training_dataset,
    initialize_parameters,
    model,
    pack_parameters,
    target_function,
    unpack_parameters,
)
from visualization import plot_surface_and_points


def run_neural_network_experiment(
    output_dir: str | Path,
    *,
    training_seed: int = 2022,
    test_seed: int = 2023,
    initialization_seed: int = 2024,
) -> Tuple[Dict[float, BFGSResult], List[Dict[str, float]]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = generate_training_dataset(500, seed=training_seed)
    X_test = generate_test_dataset(200, seed=test_seed)
    y_test = np.asarray(target_function(X_test), dtype=float)
    w0 = pack_parameters(initialize_parameters(seed=initialization_seed))

    # visualize the randomly sampled training dataset over the true surface
    plot_surface_and_points(
        target_function,
        X_train,
        y_train,
        True,
        output_dir / "training_dataset.png",
        title="Training data sampled from the target function",
    )

    results: Dict[float, BFGSResult] = {}
    metrics: List[Dict[str, float]] = []
    epsilons = (1e-1, 1e-2, 1e-3, 1e-4)

    objective = lambda w: dataset_loss(w, X_train, y_train)
    gradient = lambda w: dataset_gradient(w, X_train, y_train)

    for eps in epsilons:
        result = bfgs(
            objective,
            gradient,
            w0,
            tol=eps,
            max_iter=3000,
            alpha0=1.0,
            beta=0.5,
            c1=0.25,
            c2=0.9,
        )
        results[eps] = result
        params_star = unpack_parameters(result.x)
        test_predictions = forward_batch(X_test, params_star)
        test_mse = float(np.mean((test_predictions - y_test) ** 2))
        max_abs_error = float(np.max(np.abs(test_predictions - y_test)))

        tag = f"eps_{eps:.0e}".replace("-", "m")
        model_function = lambda points, w=result.x: np.asarray(model(points, w), dtype=float)

        plot_surface_and_points(
            model_function,
            None,
            None,
            False,
            output_dir / f"network_surface_{tag}.png",
            title=rf"Learned network surface, $\epsilon={eps:.0e}$",
        )
        plot_surface_and_points(
            target_function,
            X_test,
            test_predictions,
            True,
            output_dir / f"test_predictions_on_true_surface_{tag}.png",
            title=rf"Test predictions over true surface, $\epsilon={eps:.0e}$",
        )

        # loss curve is useful for interpreting the stopping thresholds, even though we're required for only the surface plots
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.semilogy(np.arange(len(result.f_history)), np.maximum(result.f_history, np.finfo(float).tiny))
        ax.set_xlabel("Iteration k")
        ax.set_ylabel("Training MSE")
        ax.set_title(rf"Neural-network BFGS training, $\epsilon={eps:.0e}$")
        ax.grid(True, which="both", alpha=0.35)
        fig.tight_layout()
        fig.savefig(output_dir / f"training_loss_{tag}.png", dpi=180)
        plt.close(fig)

        metrics.append(
            {
                "epsilon": eps,
                "iterations": float(result.nit),
                "success": float(result.success),
                "training_mse": float(result.fun),
                "gradient_norm": float(np.linalg.norm(result.grad)),
                "test_mse": test_mse,
                "max_abs_test_error": max_abs_error,
                "all_wolfe_steps": float(all(result.wolfe_history)),
            }
        )

    with (output_dir / "neural_network_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    np.savez(
        output_dir / "datasets_and_initialization.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        w0=w0,
    )
    for eps, result in results.items():
        tag = f"eps_{eps:.0e}".replace("-", "m")
        np.save(output_dir / f"w_star_{tag}.npy", result.x)

    return results, metrics
