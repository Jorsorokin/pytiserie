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

    if sum(above_threshold) == 0:
        raise ValueError(f"no region of the waveform has a DB above threshold {db_threshold}")

    T = time[above_threshold]
    dt = np.diff(T)

    # locate contiguous blocks where dt < max_gap
    gaps = np.where(dt > max_gap)[0]

    if len(gaps) == 0:  # only one continuous region of the audio file above the threshold
        block_start_t = T[0]
        block_end_t = T[-1]
    else:  # multiple continuous regions above the threshold — we'll assume the crisp region is the longest
        gaps = [*gaps, len(dt)]
        block_length = [gaps[0], *np.diff(gaps)]
        block_to_keep = np.argmax(block_length)
        block_start_idx = gaps[min(block_to_keep-1, 0)]
        block_end_idx = gaps[max(block_to_keep, 1)]

        # now map these block indices back to actual time
        block_start_t = T[block_start_idx + 1]
        block_end_t = T[block_end_idx]

    return block_start_t, block_end_t


def compute_crisp_metrics(
    crisp_data: pd.DataFrame,
    ref_time_block: tuple[float, float],
    db_threshold: float,
    max_gap: float = 0.1,
    smoothing_kernel: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Computes statistics on the pastry crisp audio waveform

    Parameters
    ----------
    crisp_data: pd.DataFrame — output from pytiserie.utils.read_wav_data()
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
    crisp_start, crisp_end = locate_crisp_region_in_waveform(
        crisp_db=db, time=crisp_data['time'].values, db_threshold=db_threshold, max_gap=max_gap
    )

    # extract the crisp portion of the audio & compute some stats
    idx = crisp_data['time'].between(crisp_start, crisp_end)
    avg_db = db[idx].mean()
    max_db = db[idx].max()
    crisp_length = crisp_end - crisp_start
    return {'avg_crisp_db': avg_db, 'max_crisp_db': max_db, 'crisp_duration': crisp_length}
