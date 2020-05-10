import pandas as pd
import numpy as np
import datetime as dt
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt
from data.data_classes import stock_purchase, strategy_metrics, portfolio,stock

from typing import List, Dict
from data.performance import performance_outcome

class strategy:
    inflation = 1

    def __init__(self,stock_data: List[stock],name):
        self.strategy_name = name
        self.stock_data = self.check_stock_data(stock_data)
        self.portfolio = portfolio(past_purchases = [],current_purchases = [],date = stock_data[0].date_start)
        self.dates_considered = self.get_unique_dates(self.stock_data)
        self.date_lists = {st.name: st.price_history.formatted_date.tolist() for st in self.stock_data}
        self.min_dates = {st.name: min(st.price_history.formatted_date) for st in self.stock_data}
        self.max_dates = {st.name: max(st.price_history.formatted_date) for st in self.stock_data}

    def get_unique_dates(self,x):
        dates = set()
        for st in x:
            dates = dates.union(st.price_history.formatted_date.tolist())

        dates_list = list(dates)
        dates_list.sort()
        return dates_list


    def check_stock_data(self,stock_data):
        return stock_data

    def build_metrics(self):
        self.stock_data = self.stock_data

    def update_portfolio(self,date):

        past_purchases = self.portfolio.past_purchases
        current_purchases = self.portfolio.current_purchases
        date = date

        self.portfolio = portfolio(past_purchases= past_purchases,
                                   current_purchases=current_purchases,
                                   date = date)

    def run(self):
        # Build any metrics that need to be added to the stock price history
        self.build_metrics()

        # Loop through every trading day
        for d in self.dates_considered:
            self.update_portfolio(d)

        # Sell all remaining stocks
        for st in self.stock_data:
            if self.currently_hold_stock(st):
                self.sell(st,max(st.price_history.formatted_date))


    def buy(self,stock: stock,date):
        past_purchases = self.portfolio.past_purchases
        current_purchases = self.portfolio.current_purchases
        buy_price   = float(stock.price_history.adjclose[stock.price_history.formatted_date == date])
        date_bought = date
        current_purchases.append(
            stock_purchase(
                name=stock.name,
                price_bought=buy_price,
                date_bought =  date_bought,
                date_sold =  "",
                price_sold = np.nan,
                status = "open",
                pct_change = 0,
            )
        )

        self.portfolio = portfolio(
            current_purchases = current_purchases,
            past_purchases = past_purchases,
            date=date_bought,
        )

        return current_purchases, past_purchases

    def sell(self,stock: stock,date):
        past_purchases = self.portfolio.past_purchases
        current_purchases = self.portfolio.current_purchases
        sell_price   = float(stock.price_history.adjclose[stock.price_history.formatted_date == date])
        date_sold = date

        stock_to_sell = self.get_current_purchase(stock.name)

        # Add to the past purchases list
        past_purchases.append(
            stock_purchase(
                name=stock_to_sell.name,
                price_bought=stock_to_sell.price_bought,
                date_bought =  stock_to_sell.date_bought,
                date_sold =  date_sold,
                price_sold = sell_price,
                status = "closed",
                pct_change = (sell_price - stock_to_sell.price_bought) / stock_to_sell.price_bought
            )
        )

        # Remove from current purchases
        current_purchases = [stock_purchases for stock_purchases in current_purchases if stock_purchases.name != stock.name]

        self.portfolio = portfolio(
            current_purchases=current_purchases,
            past_purchases=past_purchases,
            date=date_sold,
        )

    def get_current_purchase(self, stock_name):
        current_purchase = [stock_purchase_item for stock_purchase_item in self.portfolio.current_purchases if stock_purchase_item.name == stock_name ]
        return current_purchase[0]


    def currently_hold_stock(self,stock):
        if len(self.portfolio.current_purchases)>0:
            return any([stock.name==cp.name for cp in self.portfolio.current_purchases])
        return False


    def evaluate(self):
        # Evaluation will be, for every company you invest in, put a dollar in, and then measure the % increase in
        # money at the end based on total invested (total ROI)

        start_date = self.stock_data[0].price_history.formatted_date.iloc[0]
        end_date = self.stock_data[0].price_history.formatted_date.iloc[-1]

        total_years = (end_date-start_date).days / 365

        purchased_stock_names= set([sp.name for sp in self.portfolio.past_purchases])

        gains= {st:0 for st in purchased_stock_names}
        ng = {st:0 for st in purchased_stock_names}
        losses= {st:0 for st in purchased_stock_names}
        nl = {st:0 for st in purchased_stock_names}
        totalR={st:1 for st in purchased_stock_names}
        batting_average={st:1 for st in purchased_stock_names}
        average_gain={st:1 for st in purchased_stock_names}
        average_loss={st:1 for st in purchased_stock_names}
        ratio={st:1 for st in purchased_stock_names}
        maxR={st:1 for st in purchased_stock_names}
        maxL={st:1 for st in purchased_stock_names}
        annualised_return_rate={st:1 for st in purchased_stock_names}

        metrics = []

        for st in purchased_stock_names:
            past_purchases_stock_history = [sp for sp in self.portfolio.past_purchases if sp.name == st]
            for trade in past_purchases_stock_history:
                pc = trade.pct_change
                if pc > 0:
                    gains[st] += pc
                    ng[st] += 1
                else:
                    losses[st] += pc
                    nl[st] += 1
                totalR[st] = totalR[st] * (pc + 1)
            totalR[st] = round((totalR[st] - 1), 2)

            if ng[st]>0:
                average_gain[st] = gains[st]/ ng[st]
            else:
                average_gain[st] = 0

            maxR[st] =  max([trade.pct_change for trade in past_purchases_stock_history] )

            if nl[st]>0:
                average_loss[st] = losses[st] / nl[st]
            else:
                average_gain[st] = 0

            maxL[st] =  min([trade.pct_change for trade in past_purchases_stock_history] )
            ratio[st] = -average_gain[st] / average_loss[st]
            if nl[st]+ng[st] > 0:
                batting_average[st] = ng[st] / (ng[st] + nl[st])
            else:
                batting_average[st] = 0

            annualised_return_rate[st] = round(((1 + totalR[st]) ** (1 / total_years) - 1), 2)

            metrics.append(strategy_metrics(
                strategy_name=self.strategy_name,
                stock_ticker=st,
                start_time=start_date,
                end_time=end_date,
                batting_average=batting_average[st],
                average_gain=average_gain[st],
                average_loss=average_loss[st],
                ratio=ratio[st],
                maxR=maxR[st],
                maxL=maxL[st],
                annualised_return_rate=annualised_return_rate[st],
                )
            )

        return metrics

class red_white_blue(strategy):
    def __init__(self,
                 stock_data: List[stock],
                 name = "Red_white_blue",
                 emas_st_used: List[int] = None,
                 emas_lt_used: List[int] = None):
        super().__init__(stock_data,name)
        if emas_st_used is None:
            self.emas_st_used = [3, 5, 8, 10, 12, 15]
        if emas_lt_used is None:
            self.emas_lt_used = [30, 35, 40, 45, 50, 60]


    def build_metrics(self):
        stock_data = []
        for st in self.stock_data:
            name = st.name
            date_start = st.date_start

            emas_used = self.emas_st_used + self.emas_lt_used
            stock_price_history = st.price_history.copy()
            for x in emas_used:
                ema = x
                stock_price_history["ema_" + str(ema)] = round(stock_price_history.loc[:, 'adjclose'].ewm(span=ema, adjust=False).mean(), 2)
            cmin = []
            cmax = []
            for i in stock_price_history.index:
                cmin.append(min([stock_price_history["ema_" + str(x)][i] for x in self.emas_st_used]))
                cmax.append(min([stock_price_history["ema_" + str(x)][i] for x in self.emas_lt_used]))
            stock_price_history["cmin"] = cmin
            stock_price_history["cmax"] = cmax

            stock_data.append(
                stock(name = name,
                      date_start = date_start,
                      price_history = stock_price_history)
            )

        self.stock_data = stock_data


    def update_portfolio(self,date):

        # Loop through all of the stocks on our stock market buy / sell if you can
        for st in self.stock_data:
            if date not in self.date_lists[st.name]:
                pass
            elif date == self.max_dates[st.name]:
                if self.currently_hold_stock(st):
                    self.sell(st, date)
            else:
                cmin = float(st.price_history.cmin[st.price_history.formatted_date==date])
                cmax = float(st.price_history.cmax[st.price_history.formatted_date==date])
                if cmin > cmax and not self.currently_hold_stock(st):
                    self.buy(st,date)
                elif cmin < cmax and self.currently_hold_stock(st):
                    self.sell(st,date)
                else:
                    pass

class buy_and_hold(strategy):
    def __init__(self,
                 stock_data: List[stock],
                 name = "buy_and_hold"):
        super().__init__(stock_data,name)

    def update_portfolio(self,date):

        # Loop through all of the stocks on our stock market buy / sell if you can
        for st in self.stock_data:
            if date == self.min_dates[st.name]:
                self.buy(st,date)
            elif date == self.max_dates[st.name]:
                self.sell(st, date)
            else:
                pass




def test_strategy(strat_class,name,data_df):
    strategy_instance = strat_class(stock_data = data_df)
    strategy_instance.run()
    results = strategy_instance.evaluate()
    expecte0_annualised_return = round(np.mean([st.annualised_return_rate for st in results]),2)
    spread = round(np.std([st.annualised_return_rate for st in results]),2)
    error_in_expected_return =  2*round(spread/np.sqrt(len(results)),2)
    print(name,": Out of ", len(results)," stocks, the average annualised ROI is: ",expecte0_annualised_return,"+/-",error_in_expected_return," with a spread of ",spread)
    return strategy_instance

table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df_stock_tickers = table[0]
df_stock_tickers = df_stock_tickers.loc[df_stock_tickers.Symbol !="T",:]
df_stock_tickers["Date first added"] = pd.to_datetime(df_stock_tickers["Date first added"])

# Define starting date for back test
startYear = 2000
startMonth = 1
startDay = 4
start = dt.datetime(startYear, startMonth, startDay)
end = dt.datetime.now()

#### ENTER IN YOUR STOCKS (this represents the pool of stocks for consideration)
stock_tocker_str = list(df_stock_tickers.Symbol)
#stock_tocker_str = ["DIS","TSLA"]

#yahoo_financials to obtain the stock history
print("Downloading price history....")
yahoo_financials= YahooFinancials(stock_tocker_str)
data = yahoo_financials.get_historical_price_data(start_date=start.strftime("%Y-%m-%d"),end_date=end.strftime("%Y-%m-%d"),time_interval='daily')
data_df = [stock(name = st,
          price_history=pd.DataFrame(data[st]['prices']),
          date_start = pd.to_datetime(pd.DataFrame(data[st]['prices'])['formatted_date'][0]),
          ) for st in stock_tocker_str if len(data[st])==6]
for dat in data_df:
    dat.price_history["formatted_date"]=pd.to_datetime(dat.price_history["formatted_date"])
print("...done")



#### EVALUATION STRATEGIES STARTS HERE ####
print("Evaluating strategies...")
buy_and_hold_results = test_strategy(buy_and_hold,"buy_and_hold",data_df)
red_white_blue_results = test_strategy(red_white_blue,"red_white_blue",data_df)





