"""BFGS optimizer with a wolfe line search

the implementation follows the assignment settings
- inverse-hessian approximation H0 = I
- initial line-search step alpha0 = 1
- sufficient-decrease coefficient c1 = 0.25
- interval contraction factor beta = 0.5

the assignment does not prescribe the wolfe curvature coefficient, so c2 = 0.9 is used
the line search uses bracketing and a bisection-based zoom stage
bisection contracts an interval by beta = 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

Array = np.ndarray
Objective = Callable[[Array], float]
Gradient = Callable[[Array], Array]


@dataclass
class BFGSResult:
    x: Array
    fun: float
    grad: Array
    nit: int
    success: bool
    message: str
    x_history: List[Array]
    f_history: List[float]
    grad_norm_history: List[float]
    step_history: List[float]
    wolfe_history: List[bool]


def _evaluate(f: Objective, grad: Gradient, x: Array) -> Tuple[float, Array]:
    """evaluate an objective and its gradient with a shape sanity check"""
    x = np.asarray(x, dtype=float).reshape(-1)
    value = float(f(x))
    gradient = np.asarray(grad(x), dtype=float).reshape(-1)
    if gradient.size != x.size:
        raise ValueError(
            f"Gradient size mismatch: expected {x.size}, got {gradient.size}"
        )
    return value, gradient


def _zoom(
    f: Objective,
    grad: Gradient,
    x: Array,
    p: Array,
    phi0: float,
    dphi0: float,
    alo: float,
    ahi: float,
    phi_alo: float,
    *,
    c1: float,
    c2: float,
    beta: float,
    max_iter: int,
) -> Tuple[float, float, Array]:
    """zoom phase of a weak-wolfe line search

    with beta=0.5, the trial point is the interval midpoint
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0, 1)")

    for _ in range(max_iter):
        # for beta=0.5 this is ordinary bisection
        # writing it this way keeps the assignment's contraction parameter explicit
        aj = alo + beta * (ahi - alo)
        xj = x + aj * p
        phij, gj = _evaluate(f, grad, xj)

        if (not np.isfinite(phij)) or (
            phij > phi0 + c1 * aj * dphi0
        ) or (phij >= phi_alo):
            ahi = aj
        else:
            dphij = float(gj @ p)
            if dphij >= c2 * dphi0:
                return aj, phij, gj

            if dphij * (ahi - alo) >= 0.0:
                ahi = alo
            alo = aj
            phi_alo = phij

        if abs(ahi - alo) <= 1e-14 * max(1.0, abs(alo), abs(ahi)):
            break

    raise RuntimeError("Wolfe zoom failed to find an acceptable step")


def wolfe_line_search(
    f: Objective,
    grad: Gradient,
    x: Array,
    p: Array,
    f0: Optional[float] = None,
    g0: Optional[Array] = None,
    *,
    alpha0: float = 1.0,
    beta: float = 0.5,
    c1: float = 0.25,
    c2: float = 0.9,
    max_iter: int = 50,
    max_zoom_iter: int = 80,
    alpha_max: float = 1024.0,
) -> Tuple[float, float, Array]:
    """find a step satisfying the (weak) wolfe conditions

    the sufficient-decrease condition is
        f(x+alpha p) <= f(x) + c1*alpha*g(x)^T p
    the curvature condition is
        grad f(x+alpha p)^T p >= c2*g(x)^T p
    """
    if not (0.0 < c1 < c2 < 1.0):
        raise ValueError("Wolfe constants must satisfy 0 < c1 < c2 < 1")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0, 1)")
    if alpha0 <= 0.0:
        raise ValueError("alpha0 must be positive")
    if alpha_max <= 0.0:
        raise ValueError("alpha_max must be positive")
    if max_iter <= 0 or max_zoom_iter <= 0:
        raise ValueError("line-search iteration limits must be positive")

    if f0 is None or g0 is None:
        f0, g0 = _evaluate(f, grad, x)
    else:
        f0 = float(f0)
        g0 = np.asarray(g0, dtype=float).reshape(-1)

    dphi0 = float(g0 @ p)
    if dphi0 >= 0.0:
        raise ValueError("The search direction is not a descent direction")

    alpha_prev = 0.0
    phi_prev = f0
    alpha = min(alpha0, alpha_max)

    for i in range(max_iter):
        xa = x + alpha * p
        phia, ga = _evaluate(f, grad, xa)

        if (not np.isfinite(phia)) or (
            phia > f0 + c1 * alpha * dphi0
        ) or (i > 0 and phia >= phi_prev):
            return _zoom(
                f,
                grad,
                x,
                p,
                f0,
                dphi0,
                alpha_prev,
                alpha,
                phi_prev,
                c1=c1,
                c2=c2,
                beta=beta,
                max_iter=max_zoom_iter,
            )

        dphia = float(ga @ p)
        if dphia >= c2 * dphi0:
            return alpha, phia, ga

        if dphia >= 0.0:
            return _zoom(
                f,
                grad,
                x,
                p,
                f0,
                dphi0,
                alpha,
                alpha_prev,
                phia,
                c1=c1,
                c2=c2,
                beta=beta,
                max_iter=max_zoom_iter,
            )

        alpha_prev = alpha
        phi_prev = phia
        if alpha >= alpha_max:
            break
        # expanding by 1/beta (doubling for beta=0.5) is necessary when the curvature condition fails because the slope is still too negative
        alpha = min(alpha / beta, alpha_max)

    raise RuntimeError("Wolfe line search failed to bracket an acceptable step")


def check_wolfe(
    f0: float,
    g0: Array,
    p: Array,
    alpha: float,
    f_new: float,
    g_new: Array,
    *,
    c1: float = 0.25,
    c2: float = 0.9,
    atol: float = 1e-12,
) -> bool:
    dphi0 = float(g0 @ p)
    sufficient = f_new <= f0 + c1 * alpha * dphi0 + atol
    curvature = float(g_new @ p) >= c2 * dphi0 - atol
    return bool(sufficient and curvature)


def bfgs(
    f: Objective,
    grad: Gradient,
    x0: Array,
    *,
    tol: float = 1e-5,
    max_iter: int = 2000,
    alpha0: float = 1.0,
    beta: float = 0.5,
    c1: float = 0.25,
    c2: float = 0.9,
    verbose: bool = False,
) -> BFGSResult:
    """minimize a differentiable objective using inverse-form BFGS"""
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if alpha0 <= 0.0:
        raise ValueError("alpha0 must be positive")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0, 1)")
    if not (0.0 < c1 < c2 < 1.0):
        raise ValueError("Wolfe constants must satisfy 0 < c1 < c2 < 1")

    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    if x.size == 0:
        raise ValueError("x0 must contain at least one variable")
    n = x.size
    H = np.eye(n)
    fx, gx = _evaluate(f, grad, x)

    x_history: List[Array] = [x.copy()]
    f_history: List[float] = [fx]
    grad_norm_history: List[float] = [float(np.linalg.norm(gx))]
    step_history: List[float] = []
    wolfe_history: List[bool] = []

    if not np.isfinite(fx) or not np.all(np.isfinite(gx)):
        return BFGSResult(
            x=x,
            fun=fx,
            grad=gx,
            nit=0,
            success=False,
            message="Initial objective or gradient is not finite",
            x_history=x_history,
            f_history=f_history,
            grad_norm_history=grad_norm_history,
            step_history=step_history,
            wolfe_history=wolfe_history,
        )

    for k in range(max_iter):
        grad_norm = float(np.linalg.norm(gx))
        if grad_norm < tol:
            return BFGSResult(
                x=x,
                fun=fx,
                grad=gx,
                nit=k,
                success=True,
                message=f"Converged: ||grad f|| < {tol:g}",
                x_history=x_history,
                f_history=f_history,
                grad_norm_history=grad_norm_history,
                step_history=step_history,
                wolfe_history=wolfe_history,
            )

        p = -H @ gx
        if float(gx @ p) >= -1e-14 * max(1.0, np.linalg.norm(gx) * np.linalg.norm(p)):
            # numerical safeguard
            # it restore a guaranteed descent direction
            H = np.eye(n)
            p = -gx

        try:
            alpha, f_new, g_new = wolfe_line_search(
                f,
                grad,
                x,
                p,
                fx,
                gx,
                alpha0=alpha0,
                beta=beta,
                c1=c1,
                c2=c2,
            )
        except RuntimeError as exc:
            return BFGSResult(
                x=x,
                fun=fx,
                grad=gx,
                nit=k,
                success=False,
                message=str(exc),
                x_history=x_history,
                f_history=f_history,
                grad_norm_history=grad_norm_history,
                step_history=step_history,
                wolfe_history=wolfe_history,
            )

        wolfe_ok = check_wolfe(fx, gx, p, alpha, f_new, g_new, c1=c1, c2=c2)
        s = alpha * p
        y = g_new - gx
        ys = float(y @ s)

        # under the wolfe curvature condition ys>0
        # the tiny threshold only protects against floating-point loss of significance near convergence
        threshold = 1e-12 * max(1.0, np.linalg.norm(y) * np.linalg.norm(s))
        if ys > threshold:
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
            H = 0.5 * (H + H.T)  # remove roundoff asymmetry
        else:
            H = np.eye(n)

        x = x + s
        fx = f_new
        gx = g_new

        x_history.append(x.copy())
        f_history.append(fx)
        grad_norm_history.append(float(np.linalg.norm(gx)))
        step_history.append(alpha)
        wolfe_history.append(wolfe_ok)

        if verbose:
            print(
                f"iter={k+1:4d}  f={fx:.8e}  "
                f"||g||={np.linalg.norm(gx):.3e}  alpha={alpha:.3e}  "
                f"yTs={ys:.3e}"
            )

    return BFGSResult(
        x=x,
        fun=fx,
        grad=gx,
        nit=max_iter,
        success=False,
        message=f"Maximum iteration count ({max_iter}) reached",
        x_history=x_history,
        f_history=f_history,
        grad_norm_history=grad_norm_history,
        step_history=step_history,
        wolfe_history=wolfe_history,
    )
