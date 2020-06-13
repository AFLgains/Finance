from typing import List, Dict
from datasources.data_classes import (portfolio, stock, stock_purchase, trade_class, strategy_metrics)
from functools import reduce
import pandas as pd
from pypfopt import (EfficientFrontier, expected_returns, objective_functions,risk_models)


def check_stock_data(stock_data) -> bool:
    """
    """
    return True


def get_unique_dates(stock_data: List[stock], data_col: str):
    dates = set()
    for st in stock_data:
        dates = dates.union(st.price_history[data_col].tolist())
    dates_list = list(dates)
    dates_list.sort()
    return dates_list

def create_contiguous_buy_signals(stock: stock):
    # Take the buys and create groups of contiguous buy signals
    stock.price_history["buy_sell"] = (
            stock.price_history["buy_sell_signal"]
            * (
                    stock.price_history["buy_sell_signal"]
                    != stock.price_history["buy_sell_signal"].shift()
                    & stock.price_history["buy_sell_signal"]
            ).cumsum()
    )
    stock.price_history = stock.price_history.loc[
                          stock.price_history["buy_sell_signal"], :
                          ]

    # group by the buys and take the first price and date
    price_history_group = stock.price_history.groupby(["buy_sell"]).agg(
        {"adjclose": ["first", "last"], "formatted_date": ["first", "last"]}
    )
    price_history_group["pct_change"] = (price_history_group[("adjclose", "last")]- price_history_group[("adjclose", "first")]) / price_history_group[("adjclose", "first")]

    return price_history_group

def gen_price_history_df(stock_dictionary, tickers, buy_date):

    data_frames = [
        stock_dictionary[b]
        .price_history[["adjclose"]]
        .rename(columns={"adjclose": b})
        for b in tickers
    ]
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on="formatted_date", how="left"),
        data_frames,
    )
    return df_merged.loc[df_merged.index <= buy_date].dropna()

def get_no_distribution(portfolio_cash, buys) -> Dict:
    output_redistribute = {}
    output_portfolio_cash = portfolio_cash
    for b in buys:
        output_redistribute[b.name] = portfolio_cash[b.name]
        output_portfolio_cash[b.name] = 0
    return output_redistribute, output_portfolio_cash

def get_flat_distributions(portfolio_cash, buys) -> Dict:
    output_redistribute = {}
    output_portfolio_cash = {}
    total_cash = sum([v for x, v in portfolio_cash.items()])
    for b in buys:
        output_redistribute[b.name] = total_cash / len(buys)
    for x, v in portfolio_cash.items():
        output_portfolio_cash[x] = 0
    return output_redistribute, output_portfolio_cash


def get_optimal_distributions(stock_dictionary, portfolio_cash, buys, buy_date):
    output_redistribute = {}
    output_portfolio_cash = {}
    total_cash = sum([v for x, v in portfolio_cash.items()])
    price_histories = gen_price_history_df(stock_dictionary, [b.name for b in buys], buy_date)

    if not len(price_histories.index) > len(price_histories.columns):
        return get_flat_distributions(portfolio_cash, buys)

    # Obtain the historical return and sample cov.
    mu = expected_returns.mean_historical_return(price_histories)
    S = risk_models.sample_cov(price_histories)

    # Optimise for maximal Sharpe ratio using L2 regularization
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print("Cleaned_weights")

    # Add it to buys
    for b in buys:
        output_redistribute[b.name] = cleaned_weights[b.name] * total_cash

    for x, v in portfolio_cash.items():
        output_portfolio_cash[x] = 0

    return output_redistribute, output_portfolio_cash

def no_returns(name: str, total_years):
    return  {
            "strategy_name": name,
            "total_stocks_purchased": 0,
            "annualised_trades": 0,
            "total_trades": 0,
            "total_invested": 0,
            "total_after": 0,
            "overall_annualised_return": 0,
            "batting_average": 0,
            "weighted batting_average":0,
            "total_years": total_years,
        }