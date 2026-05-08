from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks


class BaselineError(RuntimeError):
    pass


def _whittaker_smooth(y: np.ndarray, lam: float, weights: np.ndarray, diff_order: int = 2) -> np.ndarray:
    n = y.shape[0]
    if n <= diff_order + 1:
        return y.copy()

    eye = sparse.eye(n, format="csc")
    diff = eye[1:] - eye[:-1]
    for _ in range(diff_order - 1):
        diff = diff[1:] - diff[:-1]

    w = sparse.diags(weights, 0, shape=(n, n), format="csc")
    system = w + lam * (diff.T @ diff)
    rhs = weights * y
    return spsolve(system, rhs)


def airpls_single(y: np.ndarray, lam: float = 1e4, n_iter: int = 20) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y)

    for _ in range(n_iter):
        z = _whittaker_smooth(y, lam=lam, weights=w)
        residual = y - z
        neg = residual < 0

        if not np.any(neg):
            break

        scale = np.mean(np.abs(residual[neg]))
        if scale < 1e-12:
            break

        w = np.where(residual >= 0, 0.1, np.exp(np.abs(residual) / scale))
        w = np.clip(w, 1e-3, 1e6)

    baseline = _whittaker_smooth(y, lam=lam, weights=w)
    return y - baseline


def apply_airpls(X: np.ndarray, lam: float = 1e4, n_iter: int = 20) -> np.ndarray:
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i] = airpls_single(X[i], lam=lam, n_iter=n_iter)
    return out


def continuum_removal_single(wave: np.ndarray, y: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.size < 8:
        return y.copy()

    peaks, _ = find_peaks(y, distance=max(2, y.size // 40))
    anchors = np.unique(np.concatenate(([0], peaks, [y.size - 1])))

    if anchors.size < 2:
        return y.copy()

    interp = interp1d(
        wave[anchors],
        y[anchors],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    continuum = interp(wave)
    continuum = np.where(np.abs(continuum) < 1e-12, 1e-12, continuum)

    return y / continuum


def apply_continuum_removal(X: np.ndarray, wave: np.ndarray) -> np.ndarray:
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i] = continuum_removal_single(wave, X[i])
    return out


def apply_baseline_method(X: np.ndarray, wave: np.ndarray, method: str) -> np.ndarray:
    if method == "airpls":
        return apply_airpls(X)
    if method == "continuum":
        return apply_continuum_removal(X, wave)
    raise BaselineError(f"Unknown baseline method: {method}")
