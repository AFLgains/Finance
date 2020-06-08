import datetime as dt
import pickle
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
from configuration.config import Configuration, load_configuration
from datasources.data_classes import stock
from datasources.performance import test_strategy
from strategies.strategies import (DCF, ROC_DCF, ROC_PE, buy_and_hold,
                                   red_white_blue)
from yahoofinancials import YahooFinancials

DOWNLOAD_DATA = False
# Define starting date for back test
STARTYEAR = 2000
STARTMONTH = 1
STARTDAY = 4


start = dt.datetime(STARTYEAR, STARTMONTH, STARTDAY)
end = dt.datetime.now()


print("Loading from predownloaded database....")
# Load if it's already been pickled
data_df = pickle.load(open("/data_cube.pkl", "rb"))
print("...done")

run_configs = load_configuration()

data_df_list = [
    data_df.loc[data_df.Ticker == x, :].rename(
        columns={"Date": "formatted_date", "Adj. Close": "adjclose"}
    )
    for x in set(data_df.Ticker)
    if len(data_df.loc[data_df.Ticker == x, "Ticker"]) > 200
]
data_df_list = [x.set_index("formatted_date", drop=False) for x in data_df_list]
stock_list = [
    stock(name=x.Ticker[0], price_history=x, date_start=x.index[0])
    for x in data_df_list
]


print(len(data_df))

#### EVALUATION STRATEGIES STARTS HERE ####
print("Evaluating strategies...")
print(
    f'{"Strategy":20}|{"Stock purchased":15}|{"E(return_annual)":16}|{"spread":10}|{"total_trades":15}|{"total_invested":15}|{"total_after":15}|{"overall_annualised_return":30}|{"batting_average":20}'
)
print(
    f"-----------------------------------------------------------------------------------------------------------------------------------------------------"
)

tic = time.perf_counter()
buy_and_hold_results = test_strategy(
    buy_and_hold, "buy_and_hold", stock_list, purchase_frequency=1
)
# dcf_results_10 = test_strategy(DCF,"dcf_10",data_df,purchase_frequency = 1,safety_factor_buy = 1.1,safety_factor_sell = 1)
# roce_pe_results = test_strategy(ROC_PE,"ROC_PE",data_df,purchase_frequency = 1,pe_limit = 30,roce_limit = 0.2)
# roce_dcf_results = test_strategy(ROC_DCF,"ROC_DCF",data_df,purchase_frequency = 1,safety_factor_buy = 1.1,roce_limit = 0.2)
red_white_blue_results = test_strategy(
    red_white_blue, "red_white_blue", stock_list, purchase_frequency=1
)

toc = time.perf_counter()
print(toc - tic)


plt.hist(
    [x.annualised_return_rate for x in buy_and_hold_results.evaluate()[0]],
    bins=20,
    alpha=0.3,
)
plt.hist(
    [x.annualised_return_rate for x in red_white_blue_results.evaluate()[0]],
    bins=20,
    alpha=0.3,
)
plt.xlim([-1, 0.4])
plt.show()
