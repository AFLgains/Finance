from dataclasses import dataclass
from typing import List
import numpy as np

HEADER =  ['Strategy', 'Stock purchased', 'E(return_annual)', 'spread', 'total_trades', 'total_invested','total_after', 'annualised_return', 'Batting_average']
MAX_LEN = max([len(x) for x in HEADER]) + 1
FORMATTED_STRING = " %-" + str(MAX_LEN) + "s|"


@dataclass
class performance_outcome:
    pct_change: List[float]
    strategy_name: str


def test_strategy(strat_class,name,data_df,purchase_frequency,**kwargs):
    strategy_instance = strat_class(stock_data = data_df,purchase_frequency = purchase_frequency,**kwargs)
    strategy_instance.run()
    results, total_portfolio_metrics = strategy_instance.evaluate()


    if results==[]:
        results = [name] + [0]*(len(HEADER)-1)
        print(FORMATTED_STRING*len(HEADER)%results)
        return strategy_instance
    else:
        expecte0_annualised_return = round(np.mean([st.annualised_return_rate for st in results]),2)
        spread = 2*round(np.std([st.annualised_return_rate for st in results]),2)
        error_in_expected_return =  2*round(spread/np.sqrt(len(results)),2)
        res = [name,
                    len(results),
                    expecte0_annualised_return,
                    spread,
                    total_portfolio_metrics["total_trades"],
                    total_portfolio_metrics["total_invested"],
                    total_portfolio_metrics["total_after"],
                    total_portfolio_metrics["overall_annualised_return"],
                    total_portfolio_metrics["batting_average"]
               ]
        assert len(res) == len(HEADER)
        print(FORMATTED_STRING*len(res) % tuple(res))

        return strategy_instance


def print_evaluation_header():
    print("Evaluating strategies...")
    print(FORMATTED_STRING*len(HEADER)%tuple(HEADER))
    print('-'*(2+MAX_LEN)*len(HEADER))
