import pandas as pd
from simfin.names import *


def add_pe_ratio(
    data_df: pd.DataFrame, price_col: str, earnings_col: str, shares_diluted_col: str
) -> pd.DataFrame:
    """
    Price to earnings = price / (Net income)
    """
    data_df["pe_ratio"] = (
        data_df[price_col] / (data_df[earnings_col]) * data_df[shares_diluted_col]
    )
    return data_df


def add_roce_ratio(
    data_df: pd.DataFrame,
    gross_profit_col: str,
    operating_expenses_col: str,
    Total_assets_col: str,
    curr_liabilities_col: str,
) -> pd.DataFrame:
    """
    Return on capital employed = EBIT / (Total Assets – Current Liabilities)
    """
    data_df["roce_ratio"] = (
        data_df[gross_profit_col] - data_df[operating_expenses_col]
    ) / (data_df[Total_assets_col] - data_df[curr_liabilities_col])
    return data_df


def add_roa_ratio(
    data_df: pd.DataFrame, net_income_col: str, Total_assets_col: str
) -> pd.DataFrame:
    """
    Return on capital employed = EBIT / (Total Assets – Current Liabilities)
    """
    data_df["roa_ratio"] = (data_df[net_income_col]) / (data_df[Total_assets_col])
    return data_df


def add_price_to_book(
    data_df: pd.DataFrame,
    price_col: str,
    Total_assets_col: str,
    Total_liabilities_col: str,
    shares_diluted_col: str,
) -> pd.DataFrame:
    """
    Price-to-book = price / book value = Price / ((Total_assets_col - Total_liabilities_col) / Shares)
    """
    data_df["pb_ratio"] = (
        data_df[price_col]
        / (data_df[Total_assets_col] - data_df[Total_liabilities_col])
        * data_df[shares_diluted_col]
    )
    return data_df


def add_net_income_to_book(
    data_df: pd.DataFrame,
    net_income_col: str,
    Total_assets_col: str,
    Total_liabilities_col: str,
) -> pd.DataFrame:
    """
    net_income_to_book = net_income / book value = net_income / ((Total_assets_col - Total_liabilities_col))
    """
    data_df["net_income_to_book"] = data_df[net_income_col] / (
        data_df[Total_assets_col] - data_df[Total_liabilities_col]
    )
    return data_df


def add_current_assets_to_liabilities(
    data_df: pd.DataFrame, current_assets_col: str, curr_liabilities_col: str
) -> pd.DataFrame:
    """
    net_income_to_book = net_income / book value = net_income / ((Total_assets_col - Total_liabilities_col))
    """
    data_df["current_assets_to_liabilities"] = data_df[current_assets_col] / (
        data_df[curr_liabilities_col]
    )
    return data_df


def add_net_income_to_book_g_limit_n_periods(
    data_df: pd.DataFrame,
    net_income_to_book_col: str,
    limit: float = 0.1,
    window: float = 12,
) -> pd.DataFrame:
    """
    net_income_to_book = net_income / book value = net_income / ((Total_assets_col - Total_liabilities_col))
    """
    net_income_per_stock_date = data_df.groupby(
        ["Ticker", "Fiscal Year", "Fiscal Period"], as_index=False
    )[net_income_to_book_col].mean()
    net_income_per_stock_date = net_income_per_stock_date.sort_values(
        ["Ticker", "Fiscal Year", "Fiscal Period"], ascending=True
    )
    net_income_per_stock_date["net_income_book_value_g_limit"] = (
        net_income_per_stock_date[net_income_to_book_col] > limit
    )
    net_income_per_stock_date["net_income_to_book_g_limit_n_periods"] = (
        net_income_per_stock_date.groupby(["Ticker"], as_index=False)[
            "net_income_book_value_g_limit"
        ]
        .rolling(window)
        .sum()
        .reset_index(0, drop=True)
        == window
    )
    net_income_per_stock_date = net_income_per_stock_date[
        [
            "Ticker",
            "Fiscal Year",
            "Fiscal Period",
            "net_income_to_book_g_limit_n_periods",
        ]
    ]

    data_df = pd.merge(
        data_df,
        net_income_per_stock_date,
        on=["Ticker", "Fiscal Year", "Fiscal Period"],
        how="left",
    )

    return data_df


def add_earnings_growth_n_periods(
    data_df: pd.DataFrame, earnings_col: str, window: float = 16
) -> pd.DataFrame:
    """
    net_income_to_book = net_income / book value = net_income / ((Total_assets_col - Total_liabilities_col))
    """
    earnings_per_stock_per_date = data_df.groupby(
        ["Ticker", "Fiscal Year", "Fiscal Period"], as_index=False
    )[earnings_col].mean()
    earnings_per_stock_per_date = earnings_per_stock_per_date.sort_values(
        ["Ticker", "Fiscal Year", "Fiscal Period"], ascending=True
    )
    earnings_per_stock_per_date["earnings_growth"] = (
        earnings_per_stock_per_date[earnings_col].diff()
        / earnings_per_stock_per_date[earnings_col]
    )
    earnings_per_stock_per_date["mean_earnings_growth_n_periods"] = (
        earnings_per_stock_per_date.groupby(["Ticker"], as_index=False)[
            "earnings_growth"
        ]
        .rolling(window, min_periods=8)
        .mean()
        .reset_index(0, drop=True)
    )
    earnings_per_stock_per_date = earnings_per_stock_per_date[
        ["Ticker", "Fiscal Year", "Fiscal Period", "mean_earnings_growth_n_periods"]
    ]

    data_df = pd.merge(
        data_df,
        earnings_per_stock_per_date,
        on=["Ticker", "Fiscal Year", "Fiscal Period"],
        how="left",
    )
    return data_df


def add_net_income_postive_history(
    data_df: pd.DataFrame, net_income_col: str, window: int = 4
) -> pd.DataFrame:
    """
    net_income_to_book = net_income / book value = net_income / ((Total_assets_col - Total_liabilities_col))
    """
    net_income_per_stock_date = data_df.groupby(
        ["Ticker", "Fiscal Year"], as_index=False
    )[net_income_col].mean()
    net_income_per_stock_date = net_income_per_stock_date.sort_values(
        ["Ticker", "Fiscal Year"], ascending=True
    )
    net_income_per_stock_date["net_income_negative"] = (
        net_income_per_stock_date[net_income_col] < 0
    )
    net_income_per_stock_date["net_income_positive_all_history"] = (
        net_income_per_stock_date.groupby(["Ticker"], as_index=False)[
            "net_income_negative"
        ]
        .rolling(window=window, min_periods=2)
        .sum()
        .reset_index(0, drop=True)
        == 0
    )
    net_income_per_stock_date = net_income_per_stock_date[
        ["Ticker", "Fiscal Year", "net_income_positive_all_history"]
    ]
    data_df = pd.merge(
        data_df, net_income_per_stock_date, on=["Ticker", "Fiscal Year"], how="left"
    )

    return data_df
