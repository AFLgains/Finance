from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np

HEADER = [
    "Strategy",
    "Stock purchased",
    "E(return_annual)",
    "spread",
    "total_trades",
    "total_invested",
    "total_after",
    "annualised_return",
    "Batting_average",
    "Weight_batting_ave.",
    "Years",
]
MAX_LEN = max([len(x) for x in HEADER]) + 1
FORMATTED_STRING = " %-" + str(MAX_LEN) + "s|"


@dataclass
class performance_outcome:
    pct_change: List[float]
    strategy_name: str


def test_strategy(strat_class, name, data_df, purchase_frequency, **kwargs):
    strategy_instance = strat_class(
        stock_data=data_df, purchase_frequency=purchase_frequency, **kwargs
    )
    strategy_instance.run()
    total_portfolio_metrics = strategy_instance.evaluate()

    annualised_trades = [tr for tr in total_portfolio_metrics["annualised_trades"] if tr < 10]

    res = [
        name,
        total_portfolio_metrics["total_stocks_purchased"],
        np.mean(total_portfolio_metrics["annualised_trades"]),
        np.std(total_portfolio_metrics["annualised_trades"]),
        total_portfolio_metrics["total_trades"],
        total_portfolio_metrics["total_invested"],
        total_portfolio_metrics["total_after"],
        total_portfolio_metrics["overall_annualised_return"],
        total_portfolio_metrics["batting_average"],
        total_portfolio_metrics["weighted_batting_average"],
        total_portfolio_metrics["total_years"],
    ]
    assert len(res) == len(HEADER)
    print(FORMATTED_STRING * len(res) % tuple(res))

    plt.hist(annualised_trades, bins=max(15, round(len(annualised_trades) / 10)))
    plt.xlabel("Annualised returns per stock")
    plt.ylabel("Return (%)")
    plt.title(name)
    plt.xlim([-2, 2])
    plt.show()

    return strategy_instance


def print_evaluation_header():
    print("Evaluating strategies...")
    print(FORMATTED_STRING * len(HEADER) % tuple(HEADER))
    print("-" * (2 + MAX_LEN) * len(HEADER))
