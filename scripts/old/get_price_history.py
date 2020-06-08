import datetime as dt
import pickle
import random

import FundamentalAnalysis as fa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.data_classes import stock
from yahoofinancials import YahooFinancials as yf

n_stocks = 500
historical_sandp = pd.read_html(
    "https://github.com/leosmigel/analyzingalpha/blob/master/sp500-historical-components-and-changes/sp500_history.csv"
)[0]
historical_sandp.date = pd.to_datetime(historical_sandp.date)
historical_sandp_2010 = historical_sandp.loc[
    (
        (
            (historical_sandp.variable == "added_ticker")
            & (historical_sandp.date < "2010-01-01")
        )
        | (
            (historical_sandp.variable == "removed_ticker")
            & (historical_sandp.date > "2010-01-01")
        )
    ),
    "value",
]


historical_sandp_2010_list = list(set(historical_sandp_2010.tolist()))

start_str = "2008-01-01"
end_str = dt.datetime.now().strftime("%Y-%m-%d")

# Randomly sample a bunhch of stocks
yahoo_financials = YahooFinancials(historical_sandp_2010_list[0:n_stocks])
stock_data_dict = yahoo_financials.get_historical_price_data(
    start_date=start_str, end_date=end_str, time_interval="daily"
)

# Save them to the interim folder
with open("old/stock_history_price_data.pkl", "wb") as f:
    pickle.dump(stock_data_dict, f)
