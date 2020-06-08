import datetime as dt
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.metrics import evaluate_strategies
from data.performance import performance_outcome
from strategies import strategies as strat
from yahoofinancials import YahooFinancials

stock_tocker_str = "UPRO"

startYear = 1900
startMonth = 1
startDay = 1
start = dt.datetime(startYear, startMonth, startDay)
end = dt.datetime.now()

# Now lets use yahoo_financials
yahoo_financials = YahooFinancials(stock_tocker_str)
data = yahoo_financials.get_historical_price_data(
    start_date=start.strftime("%Y-%m-%d"),
    end_date=end.strftime("%Y-%m-%d"),
    time_interval="daily",
)
data_df = pd.DataFrame(data[stock_tocker_str]["prices"])
data_df["formatted_date"] = pd.to_datetime(data_df["formatted_date"])

# Define the strategies we wish to test
strat_to_test = [strat.red_white_blue, strat.buy_and_hold]

performance_outcome = [strat(data_df=data_df) for strat in strat_to_test]
metrics = [
    evaluate_strategies(
        data_df=data_df, stock_tocker_str=stock_tocker_str, performance_outcome=pc
    )
    for pc in performance_outcome
]
