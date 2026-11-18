import numpy as np
from scipy.optimize import least_squares
from typing import TypedDict
from functools import partial


ExponentialParams = tuple[float, float, float]
DoubleExponentialParams = tuple[float, float, float, float, float]

DECAY_CONSTANTS_GUESSES = {
    "single": 0.3,
    "double": (0.5, 0.1)
}
DOUGH_LOWER_BOUND = 1.0
ALPHA = 0.01  # L2 penalty term


class GoF(TypedDict):
    mse: float
    r_squared: float
    aic: float


def exponential(t, a, k):
    return a * np.exp(-k * t)


def double_exponential(t, y0, a1, k1, a2, k2):
    return y0 + exponential(t, a1, k1) + exponential(t, a2, k2)


def _calculate_gof(x, y, params, func):
    # Calculate the fitted values
    fitted_values = func(x, *params)
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
    def regularized_residuals(params, x, y, func, alpha):
        # Standard data residuals
        data_res = func(x, *params) - y
        # L2 penalty on the residuals
        reg_res = np.sqrt(alpha) * params
        return np.concatenate([data_res, reg_res])

    func = double_exponential if double else exponential
    loss = partial(regularized_residuals, x=x, y=y, func=func, alpha=ALPHA)

    if init_params is not None:
        result = least_squares(lambda p: loss(p), init_params, **kwargs)
    else:
        result = least_squares(lambda p: loss(p), **kwargs)

    gof = _calculate_gof(x, y, result.x, func)
    return result.x, gof


def _compute_init_param_guess(x, double: bool = False) -> ExponentialParams | DoubleExponentialParams:
    """
    Compute initial parameter guesses for exponential fitting.

    Parameters
    ----------
    x : array-like
        The independent variable.
    y : array-like
        The dependent variable.
    double : bool, optional
        If True, compute initial guesses for a double exponential fit. If False, compute for a single exponential fit.
    """
    if double:
        p0 = (
            DOUGH_LOWER_BOUND,  # y0: near baseline... doughy-lower-bound (cm)
            (x[0] - np.min(x)) * 0.6,  # a1: fast amplitude
            DECAY_CONSTANTS_GUESSES["double"][0],  # k1: initial fast decay
            (x[0] - np.min(x)) * 0.4,  # a2: slow amplitude
            DECAY_CONSTANTS_GUESSES["double"][1]  # k2: decay slow
        )
    else:
        p0 = (
            x[0],  # a: amplitude
            DECAY_CONSTANTS_GUESSES["single"]  # k: decay rate
        )

    return p0


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
        "params" — the fitted parameters of the exponential function; if double, includes both components
        "spring_decay_halflife" — the half-life of the exponential decay of the spring structure; if double, suffixed with the index of the component
        "spring_decay_mse" — the mean squared error of the exponential fit
        "spring_decay_r_squared" — the R-squared value of the exponential fit
        "spring_decay_aic" — the Akaike Information Criterion of the exponential fit

        these are 
    """

    x = np.arange(len(spring_data))

    metrics = {}

    init_params = _compute_init_param_guess(x, double=double)
    params, gof = fit_exponential(x, spring_data, init_params=init_params, double=double)
    metrics.update({f"spring_decay_{key}": val for key, val in gof.items() if key})
    metrics["params"] = params

    # compute half life; for double, need to compute per component
    if double:
        halflife = [np.log(2) / params[i] for i in (2, 4)]
        metrics[f"spring_decay_halflife"] = halflife[0]
        metrics[f"spring_decay_halflife_2"] = halflife[1]
    else:
        metrics[f"spring_decay_halflife"] = np.log(2) / params[1]

    return metrics
    