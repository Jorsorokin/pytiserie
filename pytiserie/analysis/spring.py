import numpy as np
from scipy.optimize import least_squares
from typing import TypedDict


ExponentialParams = tuple[float, float]
DoubleExponentialParams = tuple[float, float, float, float]


class GoF(TypedDict):
    mse: float
    r_squared: float
    aic: float


def _fit_exponential(x, a, b):
    return a * np.exp(b * x)


def _fit_exponential_double(x, a1, b1, a2, b2):
    return _fit_exponential(x, a1, b1) + _fit_exponential(x, a2, b2)


def _fit(x, *params):
    if len(params) == 2:
        return _fit_exponential(x, *params)
    elif len(params) == 4:
        return _fit_exponential_double(x, *params)
    else:
        raise ValueError("Invalid number of parameters. Expected 2 or 4.")


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
    if double:
        # Initial guess for double exponential: [a1, b1, a2, b2]
        init_params = init_params or (1.0, -0.1, 0.5, -0.05)
        func = _fit_exponential_double
    else:
        # Initial guess for single exponential: [a, b]
        init_params = init_params or (1.0, -0.1)
        func = _fit_exponential

    result = least_squares(lambda p: func(x, *p) - y, init_params, **kwargs)
    gof = _calculate_gof(x, y, result.x, func)
    return result.x, gof

