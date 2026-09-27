"""run all experiments and write a compact json summary"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neural_network_experiment import run_neural_network_experiment
from rosenbrock import run_rosenbrock_experiment


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    rosen_result, rosen_metrics = run_rosenbrock_experiment(output_dir)
    _, nn_metrics = run_neural_network_experiment(output_dir)

    summary = {
        "rosenbrock": {
            **rosen_metrics,
            "success": rosen_result.success,
            "message": rosen_result.message,
            "x_star": rosen_result.x.tolist(),
        },
        "neural_network": nn_metrics,
        "assumptions": {
            "wolfe_c2": 0.9,
            "training_seed": 2022,
            "test_seed": 2023,
            "initialization_seed": 2024,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
