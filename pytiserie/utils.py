import numpy as np
import soundfile as sf
import pandas as pd

from scipy.signal import correlate
from typing import Optional


def make_gauss_kernel(t: int, t_sigma: float, sr: Optional[int] = 1) -> np.ndarray:
    """
    creates a 1D gaussian kernel to smooth a 1D signal w/ a convolution operator

    Parameters
    ----------
    t: int — the length of the kernel
    t_sigma: float — the scale parameter of the guassian (larger = wider gaussian)
    sr: Optional[int] — the sampling rate (defaults to 1)
    """
    def gauss(x):
        return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2))

    n = int(t * sr)
    sigma = t_sigma * sr
    r = range(-n // 2, n //2 + 1)
    kernel = np.array(list(map(gauss, r)))
    return kernel


def autocorr(x: np.ndarray, center: bool = False, symmetric: bool = False) -> np.ndarray:
    """
    Computes the autocorrelation of a 1D signal

    Parameters
    ----------
    x: np.ndarray — the input 1D signal
    center: bool (False) — if True, centers the signal x
    symmetric: bool (False) — if True, returns the full symmetric autocorrelation
    """
    if center:
        x = (x - x.mean()) / x.std()

    result = correlate(x, x, mode='full')
    result = result / result.max()  # this normalizes the scale so that lag=0 results in correlation of 1
    if symmetric:
        return result
    return result[len(x)-1:]


def read_wav_data(wav_file: str) -> tuple[pd.DataFrame, float]:
    """
    Reads in a .wav audio file and returns the data, relative time, and sampling rate
    in a pandas dataframe

    Parameters
    ---------
    wav_file: str — path to the audio file
    """
    data, sr = sf.read(wav_file)

    # create a time index
    T = 1 / sr
    time = np.arange(len(data)) * T
    wav_df = pd.DataFrame({'time': time, 'audio': data})
    return wav_df, sr
