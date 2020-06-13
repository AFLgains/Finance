import datetime as dt
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class stock_purchase:
    name: str
    price_bought: float
    date_bought: dt.datetime
    date_sold: dt.datetime
    price_sold: float
    status: str
    pct_change: float


@dataclass
class trade_class:
    stock_purchase_data: stock_purchase
    amount: float


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


@dataclass
class portfolio:
    current_purchases: List[stock_purchase] = None
    past_purchases: List[stock_purchase] = None
    date: dt.datetime = None


@dataclass
class stock:
    name: str
    price_history: pd.DataFrame
    date_start: dt.datetime
