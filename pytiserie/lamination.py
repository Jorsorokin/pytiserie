import numpy as np

from scipy.signal import fftconvolve, find_peaks
from typing import Optional
from pytiserie.utils import normalized_psd


def compute_lamination_peak_metrics(
    line_scan_data: np.ndarray,
    smoothing_kernel: Optional[np.ndarray] = None,
    width: int = 2,
    prominence: int = 10
) -> dict[str, float]:
    """
    Computes various metrics on the lamination structure of a pastry

    Parameters
    ----------
    line_scan_data: np.ndarray — a 1D signal of relative lamination pxl intensity obtained from a line scan
    smoothing_kernel: Optional[np.ndarray] — optional 1D kernel to smooth the line scan signal
    width: int (2) — the minimum width (in pxls) required for detecting lamination peaks
    prominence: int (10) — the minimum relative prominence required for detecting lamination peaks

    Returns
    -------
    dictionary with the following statistics:
        "rel_num_peaks" — the # of lamination peaks detected divided by the length of the line scan
        "peak_size_mean" — the mean width of the lamination peaks
        "peak_size_std" — the standard deviation of the lamination peak widths
        "cavity_size_mean" — the mean inter-peak distances (representing air holes)
        "cavity_size_std" — the standard deviation of cavity sizes
        "frequency_entropy" — the entropy of the fourier-decomposition of the line scan
    """
    # minmax scale the line-scan data so signals are on the same scale and bias-removed
    line_scan_data = (line_scan_data - line_scan_data.min()) / (line_scan_data.max() - line_scan_data.min())

    if smoothing_kernel is not None:
        line_scan_data = fftconvolve(line_scan_data, smoothing_kernel, mode='same')

    peaks, peak_stats = find_peaks(line_scan_data, width=width, prominence=prominence)
    delta_peak_positions = np.diff(peaks)
    freq, Pxx = normalized_psd(line_scan_data)
    fourier_entropy = -np.sum(Pxx * np.log(Pxx))

    metrics = {
        'rel_num_peaks':  len(peaks) / len(line_scan_data),
        'peak_size_mean': np.mean(peak_stats['widths']),
        'peak_size_std': np.std(peak_stats['widths']),
        'cavity_size_mean': delta_peak_positions.mean(),
        'cavity_size_std': delta_peak_positions.std(),
        "frequency_entropy": fourier_entropy,
    }

    diagnostics = {"freq": freq, "psd": Pxx, "peaks": peaks, "linescan": line_scan_data}

    return metrics, diagnostics
