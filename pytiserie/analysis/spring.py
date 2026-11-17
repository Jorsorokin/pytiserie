import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from typing import TypedDict


ExponentialParams = tuple[float, float, float]
DoubleExponentialParams = tuple[float, float, float, float]


class GoF(TypedDict):
    mse: float
    r_squared: float
    aic: float


def _fit_exponential(t, a, k):
    return a * np.exp(-k * t)


def _fit_exponential_double(t, y0, a1, k1, a2, k2):
    return y0 + _fit_exponential(t, a1, k1) + _fit_exponential(t, a2, k2)


def _calculate_gof(x, y, params, func):
    # Calculate the fitted values
    fitted_values = func(x, **params)
    # Calculate residuals
    residuals = y - fitted_values
    # Calculate goodness of fit metrics
    mse = np.mean(residuals ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (np.sum(residuals ** 2) / ss_total)
    n = len(y)
    k = len(params)
    aic = n * np.log(mse) + 2 * k

    return GoF(mse=mse, r_squared=r_squared, aic=aic)


def fit_exponential(x, y, init_params: ExponentialParams | DoubleExponentialParams = None, double: bool = False, **kwargs) -> tuple[ExponentialParams | DoubleExponentialParams, GoF]:
    """
    Fit an exponential function to the data using linear least squares.

    Parameters
    ----------
    x : array-like
        The independent variable.
    y : array-like
        The dependent variable.
    init_params : ExponentialParams or DoubleExponentialParams, optional
        Initial parameters for the fit. If not provided, default initial guesses will be used.
    double : bool, optional
        If True, perform a double exponential fit. If False, perform a single exponential fit.

    Returns
    -------
    params : ExponentialParams | DoubleExponentialParams
        The fitted parameters
    gof: GoF
        The goodness of fit metrics, including mean squared error (mse), R-squared (r_squared), and Akaike Information Criterion (aic).
    """
    func = _fit_exponential_double if double else _fit_exponential
    result = least_squares(lambda p: func(x, *p) - y, init_params, **kwargs)
    gof = _calculate_gof(x, y, result.x, func)
    return result.x, gof


def calculate_exponential_halflife(k: float) -> float:
    """
    Calculate the half-life of an exponential decay function.

    Parameters
    ----------
    k : float, the decay constant.

    Returns
    -------
    (float): the half-life of the exponential decay.
    """
    return -np.log(2) / k


def compute_spring_metrics(spring_data: np.ndarray, double: bool = False) -> dict[str, float]:
    """
    Computes various metrics on the spring structure of a croissant

    Parameters
    ----------
    spring_data: np.ndarray — a 1D signal of the height obtained from squishing a croissant
    double: bool, optional
        If True, compute metrics for a double exponential fit. If False, compute metrics for a single exponential fit.

    Returns
    -------
    dictionary with the following statistics:
        "spring_decay_halflife_" — the half-life of the exponential decay of the spring structure; if double, suffixed with the index of the component
        "spring_decay_mse" — the mean squared error of the exponential fit
        "spring_decay_r_squared" — the R-squared value of the exponential fit
        "spring_decay_aic" — the Akaike Information Criterion of the exponential fit

        these are 
    """

    x = np.arange(len(spring_data))

    # initial params
    init_guesses = {
        "single": [
            spring_data[0],  # A: amplitude
            0.3  # reasonable decay rate (30 % each squish)
        ],
        "double": [
            spring_data[0],  # y0: near baseline... doughy-lower-bound
            (spring_data[0] - np.min(spring_data)) * 0.6,  # A1: fast amplitude
            0.5,  # 50 % initial fast decay
            (spring_data[0] - np.min(spring_data)) * 0.4,  # A2: slow amplitude
            0.1  # 10% decay slow
        ]
    }       

    metrics = {}

    init_params = init_guesses["double"] if double else init_guesses["single"]
    params, gof = fit_exponential(x, spring_data, init_params=init_params, double=double)
    metrics.update({f"spring_decay_{key}": val for key, val in gof.items() if key})

    # compute half life; for double, need to compute per component
    if double:
        halflife = [
            calculate_exponential_halflife(params[i])  # only use the first two params for halflife
            for i in (1, 3)
        ]
        metrics.update({f"spring_decay_halflife_{i+1}": h for i, h in enumerate(halflife)})
    else:
        metrics[f"spring_decay_halflife"] = calculate_exponential_halflife(params[0])

    return metrics
    