import pandas as pd
import datetime as dt
from yahoofinancials import YahooFinancials
from datasources.data_classes import stock
import pickle
import simfin as sf
from simfin.names import *
import sys, os
from typing import Tuple
from configuration.config import Configuration, load_configuration
from functools import reduce
from typing import List
from datasources.performance import test_strategy, print_evaluation_header
from strategies.strategies import buy_and_hold, red_white_blue

cdir = os.path.dirname(os.path.realpath(__file__))

INCOME_COLUMNS = [REVENUE, NET_INCOME]
BALANCE_COLUMNS = [CASH_EQUIV_ST_INVEST, TOTAL_CUR_ASSETS]
CASH_COLUMNS = [NET_INCOME_START, NON_CASH_ITEMS]
PRICE_COLUMNS = [CLOSE, ADJ_CLOSE]


def _asfreq(df_grp):
    # Remove TICKER from the MultiIndex.
    df_grp = df_grp.reset_index(TICKER, drop=True)

    # Perform the operation on this group.
    df_result = df_grp.asfreq(freq='D', method='ffill')

    return df_result

def get_ticker_filter(ticker_string):
    return ticker_string.split(';')

def update_fundamental_data(run_configs: Configuration) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:

    sf.set_api_key(run_configs.apikey)
    sf.set_data_dir(os.path.join(cdir, "raw_data"))
    df_income = sf.load_income(variant='annual-full', market='us',refresh_days = run_configs.refresh_rate) # Perhpas use the publish date
    df_balance = sf.load_balance(variant='annual-full', market='us',refresh_days = run_configs.refresh_rate)
    df_cash = sf.load_cashflow(variant='annual-full', market='us',refresh_days = run_configs.refresh_rate)

    if run_configs.tickers is not None:
        ticker_filter =get_ticker_filter(run_configs.tickers)
        df_income = df_income.loc[ticker_filter,:]
        df_balance = df_balance.loc[ticker_filter,:]
        df_cash = df_cash.loc[ticker_filter,:]

    return df_income, df_balance, df_cash

def update_stock_price_data(run_configs: Configuration) -> Tuple[pd.DataFrame]:
    sf.set_api_key(run_configs.apikey)
    sf.set_data_dir(os.path.join(cdir, "raw_data"))
    df_prices = sf.load_shareprices(variant='daily', market='us',refresh_days = run_configs.refresh_rate)

    if run_configs.tickers is not None:
        ticker_filter =get_ticker_filter(run_configs.tickers)
        df_prices = df_prices.loc[ticker_filter,:]

    return df_prices

def join_stock_prices_fundamental_data(
        run_configs:  Configuration,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        cash: pd.DataFrame,
        prices: pd.DataFrame) -> pd.DataFrame:

    #select columns, fill forward, reset index and rename columns
    income = income.loc[:,INCOME_COLUMNS].groupby(TICKER).apply(_asfreq).reset_index().rename(columns={"Report Date": DATE})
    balance = balance.loc[:,BALANCE_COLUMNS].groupby(TICKER).apply(_asfreq).reset_index().rename(columns={"Report Date": DATE})
    cash = cash.loc[:,CASH_COLUMNS].groupby(TICKER).apply(_asfreq).reset_index().rename(columns={"Report Date": DATE})
    prices = prices.loc[:,PRICE_COLUMNS].reset_index()

    # Merge them all together
    data_frames = [prices, income, balance, cash]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on = [TICKER,DATE],
                                                    how='left'), data_frames)

    return df_merged.dropna() # <-- Probably dangerous.

def generate_metrics(data_no_metrics:pd.DataFrame) -> pd.DataFrame:
    return data_no_metrics

def build_data_cube() -> pd.DataFrame:
    run_configs = load_configuration()
    df_income, df_balance, df_cash = update_fundamental_data(run_configs)
    df_prices = update_stock_price_data(run_configs)
    df_no_metrics = join_stock_prices_fundamental_data(
        run_configs = Configuration,
        income = df_income,
        balance = df_balance,
        cash = df_cash,
        prices = df_prices)
    df_final = generate_metrics(df_no_metrics)
    if True:
        with open('data_cube.pkl', 'wb') as f:
            pickle.dump(df_final, f)

    return df_final

def build_stock_list(data_df: pd.DataFrame, min_records: int = 200) -> List[stock]:

    data_df_list = [data_df.loc[data_df.Ticker==x,:].rename(columns ={"Date":"formatted_date","Adj. Close":"adjclose"}) for x in set(data_df.Ticker) if len(data_df.loc[data_df.Ticker==x,'Ticker'])>min_records]
    data_df_list = [x.set_index("formatted_date",drop = False) for x in data_df_list]
    stock_list= [stock(name=x.Ticker[0],price_history=x,date_start=x.index[0]) for x in data_df_list]

    return stock_list

def main():
    # Build the data cube and stock lists
    df_final = build_data_cube()
    stock_list = build_stock_list(df_final)

    # Print out the results nicely
    print_evaluation_header()
    buy_and_hold_results = test_strategy(buy_and_hold,"buy_and_hold",stock_list,purchase_frequency = 1)
    red_white_blue_results = test_strategy(red_white_blue,"red_white_blue",stock_list,purchase_frequency = 1)

if __name__=="__main__":
    main()





