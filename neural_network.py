"""a hand-written 2-4-3-1 neural network and manual backpropagation"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class Parameters:
    W1: np.ndarray  # shape (2, 4)
    b1: np.ndarray  # shape (4,)
    W2: np.ndarray  # shape (4, 3)
    b2: np.ndarray  # shape (3,)
    W3: np.ndarray  # shape (3, 1)
    b3: np.ndarray  # shape (1,)


PARAMETER_SHAPES = {
    "W1": (2, 4),
    "b1": (4,),
    "W2": (4, 3),
    "b2": (3,),
    "W3": (3, 1),
    "b3": (1,),
}
PARAMETER_ORDER = ("W1", "b1", "W2", "b2", "W3", "b3")
PARAMETER_SIZE = sum(
    int(np.prod(PARAMETER_SHAPES[name])) for name in PARAMETER_ORDER
)


def _validate_points(x: np.ndarray, *, name: str = "x") -> np.ndarray:
    """validate one point of shape (2,) or a batch of shape (n, 2)"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        if x.shape != (2,):
            raise ValueError(f"{name} must have shape (2,), got {x.shape}")
    elif x.ndim == 2:
        if x.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2), got {x.shape}")
    else:
        raise ValueError(f"{name} must be one- or two-dimensional")
    return x


def _validate_dataset(
    X: np.ndarray, y: np.ndarray, *, allow_empty: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    X = _validate_points(X, name="X")
    if X.ndim != 2:
        raise ValueError("X must be a batch with shape (n, 2)")
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] != y.size:
        raise ValueError(
            f"X and y must contain the same number of examples: "
            f"{X.shape[0]} != {y.size}"
        )
    if not allow_empty and y.size == 0:
        raise ValueError("The dataset must contain at least one example")
    return X, y


def target_function(x: np.ndarray) -> float | np.ndarray:
    """evaluate f(x1,x2)=x1*exp(-(x1^2+x2^2))"""
    x = _validate_points(x)
    result = x[..., 0] * np.exp(-(x[..., 0] ** 2 + x[..., 1] ** 2))
    return float(result) if x.ndim == 1 else result


def activation(z: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(z, dtype=float))


def activation_derivative(z: np.ndarray) -> np.ndarray:
    # 1-tanh(z)^2 is numerically stable and vectorized
    a = np.tanh(np.asarray(z, dtype=float))
    return 1.0 - a * a


def single_loss_output_derivative(output: float, target: float) -> float:
    """derivative of (output-target)^2 with respect to output"""
    return 2.0 * (float(output) - float(target))


def initialize_parameters(seed: int = 42) -> Parameters:
    rng = np.random.default_rng(seed)
    # the assignment explicitly requests division by sqrt(n) for W in R^{m x n}
    W1 = rng.standard_normal((2, 4)) / np.sqrt(4.0)
    b1 = np.zeros(4)
    W2 = rng.standard_normal((4, 3)) / np.sqrt(3.0)
    b2 = np.zeros(3)
    W3 = rng.standard_normal((3, 1)) / np.sqrt(1.0)
    b3 = np.zeros(1)
    return Parameters(W1, b1, W2, b2, W3, b3)


def _validate_parameter_shapes(params: Parameters) -> None:
    for name in PARAMETER_ORDER:
        value = np.asarray(getattr(params, name), dtype=float)
        expected = PARAMETER_SHAPES[name]
        if value.shape != expected:
            raise ValueError(
                f"Parameter {name} must have shape {expected}, got {value.shape}"
            )


def pack_parameters(params: Parameters) -> np.ndarray:
    _validate_parameter_shapes(params)
    pieces = [
        np.asarray(getattr(params, name), dtype=float).reshape(-1)
        for name in PARAMETER_ORDER
    ]
    return np.concatenate(pieces)


def unpack_parameters(w: np.ndarray) -> Parameters:
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size != PARAMETER_SIZE:
        raise ValueError(f"Expected {PARAMETER_SIZE} parameters, got {w.size}")
    values: Dict[str, np.ndarray] = {}
    offset = 0
    for name in PARAMETER_ORDER:
        shape = PARAMETER_SHAPES[name]
        size = int(np.prod(shape))
        values[name] = w[offset : offset + size].reshape(shape).copy()
        offset += size
    return Parameters(**values)


def forward_single(
    x: np.ndarray, params: Parameters
) -> Tuple[float, Dict[str, np.ndarray]]:
    _validate_parameter_shapes(params)
    x = _validate_points(x).reshape(2)
    z1 = params.W1.T @ x + params.b1
    a1 = activation(z1)
    z2 = params.W2.T @ a1 + params.b2
    a2 = activation(z2)
    output = float((params.W3.T @ a2 + params.b3)[0])
    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return output, cache


def forward_batch(X: np.ndarray, params: Parameters) -> np.ndarray:
    _validate_parameter_shapes(params)
    X = _validate_points(X, name="X")
    if X.ndim != 2:
        raise ValueError("X must have shape (n, 2) for forward_batch")
    z1 = X @ params.W1 + params.b1
    a1 = activation(z1)
    z2 = a1 @ params.W2 + params.b2
    a2 = activation(z2)
    return (a2 @ params.W3 + params.b3).reshape(-1)


def model(x: np.ndarray, w: np.ndarray) -> float | np.ndarray:
    params = unpack_parameters(w)
    x = _validate_points(x)
    if x.ndim == 1:
        return forward_single(x, params)[0]
    return forward_batch(x, params)


def single_example_loss_and_gradient(
    x: np.ndarray, y: float, w: np.ndarray
) -> Tuple[float, np.ndarray]:
    params = unpack_parameters(w)
    output, cache = forward_single(x, params)
    residual = output - float(y)
    loss = residual * residual

    # backpropagation using the column-vector convention from the assignment
    delta3 = single_loss_output_derivative(output, y)
    grad_W3 = cache["a2"][:, None] * delta3
    grad_b3 = np.array([delta3])

    delta2 = (
        params.W3[:, 0] * delta3
    ) * activation_derivative(cache["z2"])
    grad_W2 = np.outer(cache["a1"], delta2)
    grad_b2 = delta2

    delta1 = (params.W2 @ delta2) * activation_derivative(cache["z1"])
    grad_W1 = np.outer(cache["x"], delta1)
    grad_b1 = delta1

    grads = Parameters(grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3)
    return float(loss), pack_parameters(grads)


def dataset_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    X, y = _validate_dataset(X, y)
    params = unpack_parameters(w)
    predictions = forward_batch(X, params)
    residuals = predictions - y
    return float(np.mean(residuals * residuals))


def dataset_gradient(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """gradient of the mean loss, implemented by vectorized backpropagation"""
    X, y = _validate_dataset(X, y)
    params = unpack_parameters(w)
    n = X.shape[0]

    z1 = X @ params.W1 + params.b1
    a1 = activation(z1)
    z2 = a1 @ params.W2 + params.b2
    a2 = activation(z2)
    predictions = (a2 @ params.W3 + params.b3).reshape(-1)

    delta3 = (2.0 / n) * (predictions - y)
    grad_W3 = a2.T @ delta3[:, None]
    grad_b3 = np.array([np.sum(delta3)])

    delta2 = (
        delta3[:, None] @ params.W3.T
    ) * activation_derivative(z2)
    grad_W2 = a1.T @ delta2
    grad_b2 = np.sum(delta2, axis=0)

    delta1 = (delta2 @ params.W2.T) * activation_derivative(z1)
    grad_W1 = X.T @ delta1
    grad_b1 = np.sum(delta1, axis=0)

    grads = Parameters(grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3)
    return pack_parameters(grads)


def dataset_gradient_via_examples(
    w: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """literal implementation of equation (5) by averaging examples"""
    X, y = _validate_dataset(X, y)
    total = np.zeros_like(np.asarray(w, dtype=float).reshape(-1))
    if total.size != PARAMETER_SIZE:
        raise ValueError(f"Expected {PARAMETER_SIZE} parameters, got {total.size}")
    for xi, yi in zip(X, y):
        _, grad_i = single_example_loss_and_gradient(xi, float(yi), w)
        total += grad_i
    return total / y.size


def _validate_dataset_size(n: int) -> int:
    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be positive")
    return int(n)


def generate_training_dataset(
    n: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    n = _validate_dataset_size(n)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = np.asarray(target_function(X), dtype=float)
    return X, y


def generate_test_dataset(n: int, seed: int = 1) -> np.ndarray:
    n = _validate_dataset_size(n)
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size=(n, 2))


def finite_difference_gradient(
    f, w: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    """central finite difference, used only for validation/tests"""
    if step <= 0.0:
        raise ValueError("step must be positive")
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("w must be a non-empty one-dimensional vector")
    g = np.zeros_like(w)
    for j in range(w.size):
        e = np.zeros_like(w)
        e[j] = step
        g[j] = (float(f(w + e)) - float(f(w - e))) / (2.0 * step)
    return g
