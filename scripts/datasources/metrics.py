import datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class strategy_metrics:
    strategy_name: str
    stock_ticker: str
    start_time: dt.datetime
    end_time: dt.datetime
    batting_average: float
    average_gain: float
    average_loss: float
    ratio: float
    maxR: float
    maxL: float
    annualised_return_rate: float


def evaluate_strategies(
    data_df: pd.DataFrame,
    stock_tocker_str: str,
    performance_outcome: performance_outcome,
) -> strategy_metrics:
    gains = 0
    ng = 0
    losses = 0
    nl = 0
    totalR = 1

    for pc in performance_outcome.pct_change:
        if pc > 0:
            gains += pc
            ng += 1
        else:
            losses += pc
            nl += 1
        totalR = totalR * ((pc / 100) + 1)

    totalR = round((totalR - 1) * 100, 2)

    if ng > 0:
        avgGains = gains / ng
        maxR = max(performance_outcome.pct_change)
    else:
        avgGains = 0
        maxR = "undefined"

    if nl > 0:
        avgLoss = losses / nl
        maxL = min(performance_outcome.pct_change)
        ratio = -avgGains / avgLoss
    else:
        avgLoss = 0
        maxL = np.inf
        ratio = np.inf

    if ng > 0 or nl > 0:
        battingAvg = ng / (ng + nl)
    else:
        battingAvg = 0

    total_years = (
        data_df["formatted_date"][data_df["formatted_date"].count() - 1]
        - data_df["formatted_date"][0]
    ).days / 365
    annualised_return_rate = round(
        ((1 + totalR / 100) ** (1 / total_years) - 1) * 100, 2
    )

    metrics = strategy_metrics(
        strategy_name=performance_outcome.strategy_name,
        stock_ticker=stock_tocker_str,
        start_time=data_df["formatted_date"][0],
        end_time=data_df["formatted_date"][data_df["formatted_date"].count() - 1],
        batting_average=battingAvg,
        average_gain=avgGains,
        average_loss=avgLoss,
        ratio=ratio,
        maxR=maxR,
        maxL=maxL,
        annualised_return_rate=annualised_return_rate,
    )

    return metrics
