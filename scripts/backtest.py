import datetime as dt
import logging
import os
import pickle
import sys
from functools import reduce
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import simfin as sf
from configuration.config import Configuration, load_configuration
from datasources.data_classes import stock
from datasources.performance import print_evaluation_header, test_strategy
from log_helpers import log_runtime, setup_logging
from metrics.financial_metrics import (
    add_current_assets_to_liabilities, add_earnings_growth_n_periods,
    add_net_income_postive_history, add_net_income_to_book,
    add_net_income_to_book_g_limit_n_periods, add_pe_ratio, add_price_to_book,
    add_roa_ratio, add_roce_ratio)
from simfin.names import *
from strategies.strategies import (MOD_LIL_BOOK, MOD_WARI_B, ROC_PE,
                                   buy_and_hold, buy_and_hold_year,
                                   red_white_blue)
from yahoofinancials import YahooFinancials

cdir = os.path.dirname(os.path.realpath(__file__))

INCOME_COLUMNS = [
    FISCAL_YEAR,
    FISCAL_PERIOD,
    REVENUE,
    NET_INCOME,
    SHARES_DILUTED,
    GROSS_PROFIT,
    OPERATING_EXPENSES,
]
BALANCE_COLUMNS = [TOTAL_ASSETS, TOTAL_CUR_LIAB, TOTAL_LIABILITIES, TOTAL_CUR_ASSETS]
CASH_COLUMNS = [NET_INCOME_START, NON_CASH_ITEMS]
PRICE_COLUMNS = [CLOSE, ADJ_CLOSE, VOLUME]
FIN_COLUMNS = [CURRENT_RATIO, "Return on Assets", "Return on Equity"]
GROWTH_COLUMNS = [EARNINGS_GROWTH]
VAL_COLUMNS = [PE, P_NETNET, MARKET_CAP, P_BOOK]
FINAL_COLUMNS = [
    TICKER,
    DATE,
    ADJ_CLOSE,
    VOLUME,
    FISCAL_YEAR,
    FISCAL_PERIOD,
    PE,
    P_BOOK,
    CURRENT_RATIO,
    REVENUE,
    MARKET_CAP,
    NET_INCOME,
    "year_first_day",
    "net_income_positive_all_history",
    "mean_earnings_growth_n_periods",
    "net_income_to_book_g_limit_n_periods",
    "net_income_to_book",
    "roce_ratio",
    "Return on Assets",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)


def as_freq(df_grp):
    # Remove TICKER from the MultiIndex.
    df_grp = df_grp.reset_index(TICKER, drop=True)

    # Perform the operation on this group.
    df_result = df_grp.asfreq(freq="D", method="ffill")

    return df_result


def get_ticker_filter(ticker_string):
    return ticker_string.split(";")


def hub_setup(run_configs):
    sf.set_api_key(run_configs.apikey)
    sf.set_data_dir(os.path.join(cdir, "raw_data"))
    hub = sf.StockHub(
        market="us",
        tickers=run_configs.tickers.split(";"),
        offset=pd.DateOffset(days=run_configs.offset),
        refresh_days=run_configs.refresh_rate,
        refresh_days_shareprices=run_configs.refresh_rate,
    )

    return hub


@log_runtime
def update_fundamental_data(
    hub, run_configs: Configuration,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_income = hub.load_income(variant="ttm-full").copy()
    df_balance = hub.load_balance(variant="ttm-full").copy()
    df_cash = hub.load_cashflow(variant="ttm-full").copy()
    df_price = hub.load_shareprices(variant="daily").copy()
    df_fin_signals = hub.fin_signals(variant="daily").copy()
    df_growth_signals = hub.growth_signals(variant="daily").copy()
    df_val_signals = hub.val_signals(variant="daily").copy()

    return (
        df_income,
        df_balance,
        df_cash,
        df_price,
        df_fin_signals,
        df_growth_signals,
        df_val_signals,
    )


@log_runtime
def update_stock_price_data(run_configs: Configuration) -> Tuple[pd.DataFrame]:
    sf.set_api_key(run_configs.apikey)
    sf.set_data_dir(os.path.join(cdir, "raw_data"))
    df_prices = sf.load_shareprices(
        variant="daily", market="us", refresh_days=run_configs.refresh_rate
    )

    if run_configs.tickers is not None:
        ticker_filter = get_ticker_filter(run_configs.tickers)
        df_prices = df_prices.loc[ticker_filter, :]

    return df_prices


def save_data(
    run_configs: Configuration,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    # Save
    if run_configs.save_output:
        income.to_csv(os.path.join(cdir, "output", run_configs.name + "_income.csv"))
        balance.to_csv(os.path.join(cdir, "output", run_configs.name + "_balance.csv"))
        cash.to_csv(os.path.join(cdir, "output", run_configs.name + "_cash.csv"))
        prices.to_csv(os.path.join(cdir, "output", run_configs.name + "_prices.csv"))


@log_runtime
def select_data_columns(
    run_configs: Configuration,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    prices: pd.DataFrame,
    fin: pd.DataFrame,
    growth: pd.DataFrame,
    val: pd.DataFrame,
) -> pd.DataFrame:

    # select columns, fill forward, reset index and rename columns
    income = (
        income.loc[:, INCOME_COLUMNS]
        .groupby(TICKER)
        .apply(as_freq)
        .reset_index()
        .rename(columns={"Report Date": DATE})
    )
    balance = (
        balance.loc[:, BALANCE_COLUMNS]
        .groupby(TICKER)
        .apply(as_freq)
        .reset_index()
        .rename(columns={"Report Date": DATE})
    )
    cash = (
        cash.loc[:, CASH_COLUMNS]
        .groupby(TICKER)
        .apply(as_freq)
        .reset_index()
        .rename(columns={"Report Date": DATE})
    )
    prices = prices.loc[:, PRICE_COLUMNS].groupby(TICKER).apply(as_freq).reset_index()

    fin = fin.loc[:, FIN_COLUMNS].groupby(TICKER).apply(as_freq).reset_index()

    growth = growth.loc[:, GROWTH_COLUMNS].groupby(TICKER).apply(as_freq).reset_index()

    val = val.loc[:, VAL_COLUMNS].groupby(TICKER).apply(as_freq).reset_index()

    return income, balance, cash, prices, fin, growth, val


def generate_metrics_before_join(
    run_configs: Configuration,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:

    return income, balance, cash, prices


@log_runtime
def join_stock_prices_fundamental_data(
    run_configs: Configuration,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cash: pd.DataFrame,
    prices: pd.DataFrame,
    fin: pd.DataFrame,
    growth: pd.DataFrame,
    val: pd.DataFrame,
) -> pd.DataFrame:

    # Merge them all together
    data_frames = [prices, income, balance, cash, fin, growth, val]
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=[TICKER, DATE], how="left"),
        data_frames,
    )

    df_merged = (
        df_merged.set_index(TICKER, drop=True)
        .groupby(TICKER, as_index=False)
        .ffill()
        .reset_index()
    )
    return df_merged


@log_runtime
def generate_metrics_after_join(data_no_metrics: pd.DataFrame) -> pd.DataFrame:
    # Add alternative financial ratios
    data_no_metrics = add_roce_ratio(
        data_no_metrics, GROSS_PROFIT, OPERATING_EXPENSES, TOTAL_ASSETS, TOTAL_CUR_LIAB
    )
    data_no_metrics = add_net_income_to_book(
        data_no_metrics,
        net_income_col=NET_INCOME,
        Total_assets_col=TOTAL_ASSETS,
        Total_liabilities_col=TOTAL_LIABILITIES,
    )
    # Time based metrics
    data_no_metrics = add_net_income_to_book_g_limit_n_periods(
        data_no_metrics,
        net_income_to_book_col="net_income_to_book",
        limit=0.1,
        window=12,
    )
    data_no_metrics = add_earnings_growth_n_periods(
        data_no_metrics, earnings_col=NET_INCOME, window=8,
    )
    data_no_metrics = add_net_income_postive_history(
        data_no_metrics, net_income_col=NET_INCOME,
    )

    return data_no_metrics


@log_runtime
def generate_buy_time(df_final_no_buy_time: pd.DataFrame) -> pd.DataFrame:
    data_df = df_final_no_buy_time.copy()
    data_df["year"] = data_df["Date"].dt.to_period("Y")
    data_df["year_first_day"] = data_df.groupby("year")["Date"].transform(
        lambda x: min(x)
    )

    return data_df


@log_runtime
def build_data_cube(run_configs: Configuration) -> pd.DataFrame:
    hub = hub_setup(run_configs)

    (
        df_income,
        df_balance,
        df_cash,
        df_prices,
        df_fin_signals,
        df_growth_signals,
        df_val_signals,
    ) = update_fundamental_data(hub, run_configs)
    (
        df_income,
        df_balance,
        df_cash,
        df_prices,
        df_fin_signals,
        df_growth_signals,
        df_val_signals,
    ) = select_data_columns(
        run_configs,
        df_income,
        df_balance,
        df_cash,
        df_prices,
        df_fin_signals,
        df_growth_signals,
        df_val_signals,
    )
    save_data(run_configs, df_income, df_balance, df_cash, df_prices)

    df_no_metrics = join_stock_prices_fundamental_data(
        run_configs=run_configs,
        income=df_income,
        balance=df_balance,
        cash=df_cash,
        prices=df_prices,
        fin=df_fin_signals,
        growth=df_growth_signals,
        val=df_val_signals,
    )
    df_final_no_buy_time = generate_metrics_after_join(df_no_metrics)
    df_final = generate_buy_time(df_final_no_buy_time)
    df_final = df_final[FINAL_COLUMNS]
    if True:
        with open("data_cube.pkl", "wb") as f:
            pickle.dump(df_final, f)

    return df_final.dropna()


def validate(data_df: pd.DataFrame, stock_name: str, run_config: Configuration) -> bool:

    min_records = run_config.min_records
    min_price = run_config.min_price
    min_volume = run_config.min_volume

    if len(data_df.iloc[:, 0]) < min_records:
        logging.info(f"Removing {stock_name} because less than {min_records} records")
        return False
    elif min(data_df[ADJ_CLOSE]) < min_price:
        logging.info(
            f"Removing {stock_name} because min price less than {min_price} dollars"
        )
        return False
    elif min(data_df[VOLUME]) < min_volume:
        logging.info(
            f"Removing {stock_name} because min volume less than {min_volume} trades per day"
        )
        return False
    else:
        return True


@log_runtime
def build_stock_list(data_df: pd.DataFrame, run_config: Configuration) -> List[stock]:

    data_df_list = [
        data_df.loc[data_df.Ticker == x, :].rename(
            columns={"Date": "formatted_date", "Adj. Close": "adjclose"}
        )
        for x in set(data_df.Ticker)
        if validate(data_df.loc[data_df.Ticker == x, :], x, run_config)
    ]
    data_df_list = [x.set_index("formatted_date", drop=False) for x in data_df_list]
    stock_list = [
        stock(name=x.Ticker[0], price_history=x, date_start=x.index[0])
        for x in data_df_list
    ]
    print(f"Universe size: {len(stock_list)} stocks")

    return stock_list


def main():

    run_config = load_configuration()
    # Build the data cube and stock lists
    df_final = build_data_cube(run_config)
    # df_final.to_csv(os.path.join(cdir,"output","validations.csv"))
    # df_final = df_final.loc[df_final.year >= "2015"]
    stock_list = build_stock_list(df_final, run_config)

    # Print out the results nicely
    print_evaluation_header()
    buy_and_hold_year_results = test_strategy(
        buy_and_hold_year, "BH", stock_list, purchase_frequency=1, redistribute=True, opt_port = False
    )
    mod_wb_distribute = test_strategy(
        MOD_WARI_B, "MOD_WARI_B", stock_list, purchase_frequency=1, redistribute=True, opt_port = True
    )

    for rev_limit in [5, 10]:
        for roa_lower_limit in [0.15]:
            for stock_limit in [10, 20, 30]:
                MOD_LIL_BOOK_20 = test_strategy(
                    MOD_LIL_BOOK,
                    "MOD_LB_"
                    + str(stock_limit)
                    + "_"
                    + str(roa_lower_limit)
                    + "_"
                    + str(rev_limit),
                    stock_list,
                    purchase_frequency=1,
                    redistribute=True,
                    stock_limit=stock_limit,
                    roa_lower_limit=roa_lower_limit,
                    min_revenue=rev_limit * 1e8,
                )

    # red_white_blue_results = test_strategy(
    #    red_white_blue, "red_white_blue", stock_list, purchase_frequency=1
    # )


if __name__ == "__main__":
    main()
