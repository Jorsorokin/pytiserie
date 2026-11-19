import numpy as np
import pandas as pd

from scipy.signal import fftconvolve
from typing import Optional


def compute_crisp_db(
    crisp_data: pd.DataFrame,
    ref_time_block: tuple[float, float],
    smoothing_kernel: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    computes the DB of the pastry crisp audio waveform relative to a reference period in the audio data

    Parameters
    ----------
    crisp_data: pd.DataFrame — output from pytiserie.utils.read_wav_data
    ref_time_block: tuple[float, float] — the [min, max] time window in the audio data to use as a reference
    smoothing_kernel: Optional[np.ndarray] — optional gaussian kernel to smooth the audio datap

    Returns
    -------
    db: np.ndarray
    """
    power = crisp_data['audio'] ** 2

    if smoothing_kernel is not None:
        power = fftconvolve(power, smoothing_kernel, mode="same")

    ref = crisp_data.time.between(ref_time_block[0], ref_time_block[1])
    db = 10 * np.log10(power / power[ref].mean())
    return db


def locate_crisp_region_in_waveform(crisp_db: np.ndarray, time: np.ndarray, db_threshold: float, max_gap: float = 0.1) -> tuple[float, float]:
    """
    Locates the pastry crisp region within a lager audio recording
    """
    # threshold the DB and find the ∆t between each point above the threshold
    above_threshold = crisp_db >= db_threshold
    
    if not np.any(above_threshold):
        raise ValueError(
            f"no region of the waveform has a DB above threshold {db_threshold}"
        )

    T = time[above_threshold]
    dt = np.diff(T)

    breaks = np.where(dt > max_gap)[0]  # breaks in the thresholded waveform that surpass the allowed merging window
    blocks = np.split(T, breaks + 1)  # split each segment
    longest_block = max(blocks, key=len)  # choose the longest continuous waveform
    start, end = longest_block[0], longest_block[-1]
    return start, end


def compute_crisp_metrics(
    crisp_data: pd.DataFrame,
    ref_time_block: tuple[float, float],
    db_threshold: float,
    max_gap: float = 0.1,
    min_length: float = 0.0,
    smoothing_kernel: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Computes statistics on the pastry crisp audio waveform

    Parameters
    ----------
    crisp_data: pd.DataFrame — output from pytiserie.utils.read_audio_data()
    ref_time_block: tuple[float, float] — the [start, end] time (in seconds) of the reference time window
    db_threshold: float — the decible level threshold to isolate the crisp portion of the audio file
    max_gap: float (0.1) — the max gap (in seconds) between two isolated regions of the audio file to consider a merge
        - separated continuous segments of the audio file that are less than max_gap seconds apart will be merged to one
    smoothing_kernel: Optional — a 1D smoothing kernel to smooth the audio data

    Returns
    -------
    dictionary with the following:
        "avg_crisp_db": float — the average DB of the crisp waveform
        "max_crisp_db": float — the max DB of the crisp waveform
        "crisp_duration": float — the length of the crisp waveform (in seconds)
    """
    # compute the DB level relative to a reference region
    db = compute_crisp_db(crisp_data, ref_time_block=ref_time_block, smoothing_kernel=smoothing_kernel)

    # threshold the DB and find the lagest contiguous region above the threshold
    try:
        crisp_start, crisp_end = locate_crisp_region_in_waveform(
            crisp_db=db, time=crisp_data['time'].values, db_threshold=db_threshold, max_gap=max_gap
        )

        idx = crisp_data['time'].between(crisp_start, crisp_end)

        if idx.sum() > 0:
            crisp_length = crisp_end - crisp_start
            if crisp_length >= min_length:
                avg_db = np.nanmean(db[idx])
                max_db = db[idx].max()
            else:
                avg_db = max_db = crisp_length = 0
                crisp_start = crisp_end = None
        else:
            crisp_length = avg_db = max_db = 0
            crisp_start = crisp_end = None

        metrics = {'avg_crisp_db': avg_db, 'max_crisp_db': max_db, 'crisp_duration': crisp_length}

    except ValueError:
        crisp_start = None
        crisp_end = None
        metrics = {"avg_crisp_db": 0, "max_crisp_db": 0, "crisp_duration": 0}

    # extract the crisp portion of the audio & compute some stats
    diagnostics = {"crisp_start": crisp_start, "crisp_end": crisp_end, "db": db}
    return metrics, diagnostics

