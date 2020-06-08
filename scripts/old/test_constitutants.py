import numpy as np
import pandas as pd

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
";".join(historical_sandp_2010_list)


