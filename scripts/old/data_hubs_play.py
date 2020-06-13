import os

import pandas as pd
# Import the main functionality from the SimFin Python API.
import simfin as sf
from simfin.names import *

sf.set_api_key("AH4RWcm8001Mmz48dGzbZ13LGp1O5Yqw")

# Set the local directory where data-files are stored.
# The dir will be created if it does not already exist.
sf.set_data_dir(os.path.join("/", "raw_data"))

market = "us"
tickers = ["AAPL", "AMZN", "MSFT", "DIS", "dasdasd"]
offset = pd.DateOffset(days=0)
refresh_days = 30

# Refresh the dataset with shareprices every 10 days.
refresh_days_shareprices = 10

hub = sf.StockHub(
    market=market,
    tickers=tickers,
    offset=offset,
    refresh_days=refresh_days,
    refresh_days_shareprices=refresh_days_shareprices,
)


df_income_annual = hub.load_income(variant="ttm")
df_fin_signals_daily = hub.fin_signals(variant="daily")
df_growth_signals_daily = hub.growth_signals(variant="daily")
df_val_signals_latest = hub.val_signals(variant="daily")
df_income_annual = hub.load_income(variant="ttm")
