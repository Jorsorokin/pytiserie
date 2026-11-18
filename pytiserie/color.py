import pandas as pd


def fgd_bkgd_color_ratio(x: float, bkgd_white: float, bkgd_black: float) -> float:
    """
    calculate the relataive gray value of x given a white & black reference background

    ratio = 1 - (bkgd_white - x) / (bkgd_white - bkgd_black)

    as x -> white, ratio -> 1
    as x -> black, ratio -> 0

    The black background normalizes the total possible range, which may be affected by
    luminance, reflection, etc.
    """
    return 1 - (bkgd_white - x) / (bkgd_white - bkgd_black)


def compute_color_metrics(color_data: pd.DataFrame) -> dict[str, float]:
    means_ = color_data['metric'].str.contains("mean")
    std_ = color_data['metric'].str.contains("std")

    fgd_mean = color_data.loc[color_data['object'].eq("fgd") & means_]
    fgd_std = color_data.loc[color_data['object'].eq("fgd") & std_]
    bkgd_white = color_data.loc[color_data['object'].eq("bkgd_white") & means_]
    bkgd_black = color_data.loc[color_data['object'].eq("bkgd_black") & means_]

    # ensure we have samples
    if len(fgd_mean) == 0:
        raise ValueError("No foreground croissant color measurement found")
    if len(fgd_std) == 0:
        raise ValueError("No foreground stdev measurement found")
    if len(bkgd_white) == 0:
        raise ValueError("No white background color measurement found")
    if len(bkgd_black) == 0:
        raise ValueError("No black background color measurement found")

    # compute the mean + 5/95 CI ratios
    mu = fgd_mean['value'].iloc[0]
    sd = fgd_std['value'].iloc[0]
    bkgd_white = bkgd_white['value'].iloc[0]
    bkgd_black = bkgd_black['value'].iloc[0]

    ratio = fgd_bkgd_color_ratio(mu, bkgd_white, bkgd_black)
    ratio_low = fgd_bkgd_color_ratio(mu - sd, bkgd_white, bkgd_black)
    ratio_high = fgd_bkgd_color_ratio(mu + sd, bkgd_white, bkgd_black)

    color_metrics = {
        "color_ratio_mean": ratio,
        "color_ratio_low": ratio_low,
        "color_ratio_high": ratio_high,
    }
    return color_metrics
