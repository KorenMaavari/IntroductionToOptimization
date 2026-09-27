"""visualization routines requested"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_surface_and_points(
    function: Callable[[np.ndarray], np.ndarray],
    X: Optional[np.ndarray],
    y: Optional[np.ndarray],
    scatter_points: bool,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    if not callable(function):
        raise TypeError("function must be callable")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = np.arange(-2.0, 2.0 + 1e-12, 0.2)
    x1, x2 = np.meshgrid(grid, grid)
    points = np.column_stack([x1.ravel(), x2.ravel()])
    values = np.asarray(function(points), dtype=float).reshape(-1)
    if values.size != points.shape[0]:
        raise ValueError(
            "function must return one scalar value for every input point"
        )
    z = values.reshape(x1.shape)

    if scatter_points:
        if X is None or y is None:
            raise ValueError("X and y are required when scatter_points=True")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (n, 2)")
        if X.shape[0] != y.size:
            raise ValueError("X and y must contain the same number of points")

    fig = plt.figure(figsize=(7.4, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x1, x2, z, alpha=0.78, linewidth=0, antialiased=True)
    if scatter_points:
        assert X is not None and y is not None
        ax.scatter(X[:, 0], X[:, 1], y, s=14, depthshade=True)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
