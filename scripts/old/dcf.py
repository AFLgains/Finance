import datetime as dt
from typing import Dict, List, Tuple

import FundamentalAnalysis as fa
import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials

stock_ticker_str = "DIS"

# Load the financials
yahoo_financials = YahooFinancials(stock_ticker_str)

# load financial statements
all_statement_data_annual = yahoo_financials.get_financial_stmts(
    "annual", ["income", "cash", "balance"]
)

# Obtain all financial statements
income = all_statement_data_annual["incomeStatementHistory"]
balance = all_statement_data_annual["balanceSheetHistory"]
cash = all_statement_data_annual["cashflowStatementHistory"]

# Get the dates of these financial series
dates = [list(a.keys())[0] for a in income[stock_ticker_str]]


def calc_time_series(
    stock_ticker_str: str, dates: List[str], statement: List[Dict], line_item: str
) -> Dict[str, float]:
    return {
        dates[d]: statement[stock_ticker_str][d][dates[d]][line_item]
        for d in range(len(dates))
    }


def calc_eff_tax_rate(stock_ticker_str: str) -> float:
    return 0.26


# Get the operating cashflow time series
cash_time_series = calc_time_series(
    stock_ticker_str, dates, cash, "totalCashFromOperatingActivities"
)
ebit_time_series = calc_time_series(stock_ticker_str, dates, income, "ebit")
DA_time_series = calc_time_series(stock_ticker_str, dates, cash, "depreciation")
capex_time_series = calc_time_series(
    stock_ticker_str, dates, cash, "capitalExpenditures"
)
