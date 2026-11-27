"""Before/after anomaly scores for metric series.

Each series is compared with its own baseline, the part of the window before
the alert. The baseline gives a median and a robust scale (IQR / 1.349, falling
back to the standard deviation, and never below 5% of the median, so that a
near-constant series doesn't turn noise into huge scores). Every sample after
the alert becomes a robust z-score ``(x - median) / scale``. A series' score is
the 90th percentile of those z-scores taken in its dominant direction, so a
shift has to last for at least a tenth of the post-alert window to count,
whereas a single spike does not. A series that moved beyond the threshold both
ways (a spike, then a collapse) is reported as "mixed".

Times are seconds relative to the alert: the metrics frame passed in is indexed
that way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aiops.data.readers import split_metric_name

NOTABLE = 3.0  # |z| from which a series is reported as anomalous
Z_CAP = 1000.0  # scores of series whose baseline is exactly zero are capped here
MIN_BEFORE = 30  # samples needed for a baseline
MIN_AFTER = 10
ONSET_RUN = 3  # consecutive samples beyond NOTABLE that mark the onset
RELATIVE_FLOOR = 0.05

COLUMNS = ["series", "service", "metric", "score", "direction", "baseline", "after_median",
           "after_peak", "onset_s", "flat_baseline"]


def score_series(metrics: pd.DataFrame) -> pd.DataFrame:
    """Score every column of ``metrics`` (index: seconds relative to the alert).

    Returns one row per series that has enough samples on both sides of the
    alert, sorted by score (highest first), with the columns in :data:`COLUMNS`.
    """
    before = metrics.loc[metrics.index < 0]
    after = metrics.loc[metrics.index >= 0]
    valid = (before.count() >= MIN_BEFORE) & (after.count() >= MIN_AFTER)
    cols = valid.index[valid]
    if len(cols) == 0:
        return pd.DataFrame(columns=COLUMNS)
    before, after = before[cols], after[cols]

    median = before.median()
    scale = (before.quantile(0.75) - before.quantile(0.25)) / 1.349
    scale = scale.where(scale > 0, before.std()).fillna(0.0)
    scale = np.maximum(scale, RELATIVE_FLOOR * median.abs())
    flat = scale <= 0  # the baseline is constantly zero
    scale = scale.where(~flat, 1e-12)

    z = (after - median) / scale
    high, low = z.quantile(0.9), z.quantile(0.1)
    up = high.abs() >= low.abs()
    score = high.where(up, -low).clip(lower=0.0, upper=Z_CAP)
    peak = after.quantile(0.9).where(up, after.quantile(0.1))
    # a series that both rose and fell beyond NOTABLE (a spike, then a collapse) is "mixed"
    mixed = (high >= NOTABLE) & (low <= -NOTABLE)
    direction = np.where(mixed, "mixed", np.where(up, "up", "down"))

    beyond = (z.abs() >= NOTABLE).astype(float)  # NaN (not observed) counts as not beyond
    full_run = (beyond.rolling(ONSET_RUN, min_periods=ONSET_RUN).sum() >= ONSET_RUN).to_numpy()
    has_onset = full_run.any(axis=0)
    first_end = full_run.argmax(axis=0)
    onset = np.where(has_onset, after.index.to_numpy()[np.maximum(first_end - ONSET_RUN + 1, 0)], np.nan)

    names = [split_metric_name(c) for c in cols]
    table = pd.DataFrame({
        "series": cols,
        "service": [n[0] for n in names],
        "metric": [n[1] for n in names],
        "score": score.to_numpy(),
        "direction": direction,
        "baseline": median.to_numpy(),
        "after_median": after.median().to_numpy(),
        "after_peak": peak.to_numpy(),
        "onset_s": onset,
        "flat_baseline": flat.to_numpy(),
        # tie-break for capped scores: size of the change relative to the baseline level
        "_relative": ((peak - median).abs() / np.maximum(median.abs(), scale)).to_numpy(),
    })
    table = table.sort_values(["score", "_relative", "series"], ascending=[False, False, True],
                              kind="stable")
    return table.drop(columns="_relative").reset_index(drop=True)


def service_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Per service: strongest series score, number of anomalous series, earliest onset."""
    if scores.empty:
        return pd.DataFrame(columns=["service", "max_score", "anomalous_series", "top_metrics",
                                     "earliest_onset_s"])
    notable = scores[scores["score"] >= NOTABLE]
    rows = []
    for service, group in scores.groupby("service", sort=False):
        hits = notable[notable["service"] == service]
        rows.append({
            "service": service,
            "max_score": group["score"].max(),
            "anomalous_series": len(hits),
            "top_metrics": [f"{m} {d}" for m, d in
                            zip(hits["metric"].iloc[:3], hits["direction"].iloc[:3])],
            "earliest_onset_s": hits["onset_s"].min() if len(hits) else np.nan,
        })
    return (pd.DataFrame(rows).sort_values(["max_score", "service"], ascending=[False, True])
            .reset_index(drop=True))
