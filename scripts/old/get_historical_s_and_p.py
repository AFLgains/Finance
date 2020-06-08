from datetime import datetime

import numpy as np
import pandas as pd

data = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")


# Get current S&P table and set header column
sp500 = data[0].iloc[1:, [0, 1, 6, 7]]
columns = ["added_ticker", "name", "date", "cik"]
sp500.columns = columns
sp500.loc[sp500["date"].isnull(), "date"] = "1957-01-01"

# One date is in the wrong format. Correcting it.
sp500.loc[sp500.added_ticker == "T", "date"] = "1983-11-30"
sp500.loc[:, "date"] = pd.to_datetime(sp500.loc[:, "date"])
sp500 = pd.melt(sp500, id_vars=["date", "name", "cik"], value_vars=["added_ticker"])
sp500.head()


# Get S&P500 adjustments table and set columns
sp500_adjustments = data[1]
sp500_adjustments = sp500_adjustments[2:].copy()
columns = [
    "date",
    "added_ticker",
    "added_name",
    "removed_ticker",
    "removed_name",
    "reason",
]
sp500_adjustments.columns = columns
updates = sp500_adjustments[~sp500_adjustments["date"].str.contains(",")].T.shift(1).T
sp500_adjustments["date"].loc[~sp500_adjustments["date"].str.contains(",")] = np.nan
sp500_adjustments[sp500_adjustments["added_ticker"].isnull()]
sp500_adjustments.update(updates)
sp500_adjustments["date"].loc[sp500_adjustments["date"].isnull()] = (
    sp500_adjustments["date"].T.shift(1).T
)
sp500_adjustments["date"].loc[sp500_adjustments["date"].isnull()] = (
    sp500_adjustments["date"].T.shift(1).T
)
sp500_adjustments["date"].loc[sp500_adjustments["date"].isnull()] = (
    sp500_adjustments["date"].T.shift(1).T
)
sp500_adjustments["date"].loc[sp500_adjustments["date"].isnull()] = (
    sp500_adjustments["date"].T.shift(1).T
)
sp500_adjustments["date"].loc[sp500_adjustments["date"].isnull()] = (
    sp500_adjustments["date"].T.shift(1).T
)
sp500_additions = sp500_adjustments[~sp500_adjustments["added_ticker"].isnull()]
sp500_additions = sp500_additions[["date", "added_ticker", "added_name"]]
sp500_additions.rename(columns={"added_name": "name"}, inplace=True)
sp500_additions = pd.melt(
    sp500_additions, id_vars=["date", "name"], value_vars=["added_ticker"]
)
sp500_deletions = sp500_adjustments[~sp500_adjustments["removed_ticker"].isnull()]
sp500_deletions = sp500_deletions[["date", "removed_ticker", "removed_name"]]
sp500_deletions.rename(columns={"removed_name": "name"}, inplace=True)
sp500_deletions = pd.melt(
    sp500_deletions, id_vars=["date", "name"], value_vars=["removed_ticker"]
)
sp500_history = pd.concat([sp500_deletions, sp500_additions])
sp500_history.head()


df = pd.concat([sp500, sp500_history], ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
df.sort_values(by="cik", ascending=False, inplace=True)
deduped_df = df[~df.duplicated(["date", "variable", "value"])].copy()
deduped_df.sort_values(by="date", inplace=True)
deduped_df.to_csv("sp500_history.csv")
print(deduped_df.head())


deduped_df.sort_values(by="cik", ascending=False, inplace=True)
deduped_df = deduped_df[~deduped_df.duplicated("value")]
# discovery has 2 share classes listed
deduped_df = deduped_df[~deduped_df.duplicated("cik")]
deduped_df.sort_values(by="value", inplace=True)
deduped_df.drop(["date", "variable"], axis=1, inplace=True)
deduped_df.rename(columns={"value": "ticker"}, inplace=True)
deduped_df.to_csv("sp500_constituents.csv")
deduped_df.head()

sp500_additions.loc[sp500_additions]
