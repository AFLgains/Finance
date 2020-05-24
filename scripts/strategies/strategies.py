import pandas as pd
import numpy as np
import datetime as dt
from yahoofinancials import YahooFinancials
from datasources.data_classes import stock_purchase, strategy_metrics, portfolio,stock
import time
import pickle
from typing import List
from datasources.performance import test_strategy


class strategy:
    """
    Super class
    """

    def __init__(self,stock_data: List[stock],name,purchase_frequency=1):
        self.strategy_name = name
        self.stock_data = self.check_stock_data(stock_data)
        self.purchase_frequency = purchase_frequency

        self.dates_considered = self.get_unique_dates(self.stock_data)
        self.date_lists = {st.name: st.price_history.formatted_date.tolist() for st in self.stock_data}
        self.min_dates = {st.name: min(st.price_history.formatted_date) for st in self.stock_data}
        self.max_dates = {st.name: max(st.price_history.formatted_date) for st in self.stock_data}
        self.portfolio = portfolio(past_purchases = [],current_purchases = [],date = min(self.dates_considered))


    def get_unique_dates(self,x):
        dates = set()
        for st in x:
            dates = dates.union(st.price_history.formatted_date.tolist())

        dates_list = list(dates)
        dates_list.sort()
        dates_list = dates_list[0::self.purchase_frequency]
        return dates_list

    def check_stock_data(self,stock_data):
        """
        Peform Checks to stock data
        :param stock_data:
        :return:
        """
        return stock_data

    def build_metrics(self):
        self.stock_data = self.stock_data

    def update_portfolio(self,stock):
        """
        Default behaviour is to not buy anything
        :param stock:
        :return:
        """
        return self.portfolio

    def update_portfolio_stock(self,stock):

        # Take the buys and create groups of contiguous buy signals
        stock.price_history['buy_sell'] = stock.price_history['buy_sell_signal']*(stock.price_history['buy_sell_signal']  != stock.price_history['buy_sell_signal'] .shift() & stock.price_history['buy_sell_signal']).cumsum()
        stock.price_history = stock.price_history.loc[stock.price_history['buy_sell_signal'],:]

        #group by the buys and take the first price and date
        price_history_group = stock.price_history.groupby(['buy_sell']).agg({'adjclose':['first','last'],'formatted_date':['first','last']})
        price_history_group['pct_change'] =(price_history_group[('adjclose','last')]-price_history_group[('adjclose','first')]) / price_history_group[('adjclose','first')]

        for ind in price_history_group.index:
            self.portfolio.past_purchases.append(
                stock_purchase(
                    name=stock.name,
                    price_bought=price_history_group[('adjclose','first')][ind],
                    date_bought =  price_history_group[('formatted_date','first')][ind],
                    date_sold =  price_history_group[('formatted_date','last')][ind],
                    price_sold = price_history_group[('adjclose','last')][ind],
                    status = "closed",
                    pct_change = price_history_group['pct_change'][ind]
                )
            )

    def run(self):
        # Build any metrics that need to be added to the stock price history
        self.build_metrics()

        # Loop through every stock and update portfolio with buy and sells
        for st in self.stock_data:
            self.update_portfolio(st)


    def evaluate(self):
        # Evaluation will be, for every company you invest in, put a dollar in, and then measure the % increase in
        # money at the end based on total invested (total ROI)

        start_date = min(self.dates_considered)
        end_date = max(self.dates_considered)

        total_years = (end_date-start_date).days / 365

        purchased_stock_names= set([sp.name for sp in self.portfolio.past_purchases])


        if purchased_stock_names==[]:
            return [strategy_metrics(
                strategy_name=self.strategy_name,
                stock_ticker="None",
                start_time=start_date,
                end_time=end_date,
                batting_average=1,
                average_gain=0,
                average_loss=0,
                ratio=1,
                maxR=0,
                maxL=0,
                annualised_return_rate=0,
                )]

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
        annualised_return_rate={st:0 for st in purchased_stock_names}


        funds_total_portfolio = 0
        costs_total_portfolio = 0
        total_trades = 0
        total_gains =0
        total_losses =0

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
                totalR[st] *= (pc + 1)
                total_trades += 1
            total_gains +=ng[st]
            total_losses += nl[st]
            funds_total_portfolio += totalR[st]
            costs_total_portfolio += 1


            totalR[st] = round((totalR[st] - 1), 2)

            if ng[st]>0:
                average_gain[st] = gains[st]/ ng[st]
            else:
                average_gain[st] = 0

            maxR[st] =  max([trade.pct_change for trade in past_purchases_stock_history] )
            if nl[st]>0:
                average_loss[st] = losses[st] / nl[st]
            else:
                average_loss[st] = 0

            maxL[st] =  min([trade.pct_change for trade in past_purchases_stock_history] )

            if average_loss[st]>0:
                ratio[st] = -average_gain[st] / average_loss[st]
            else:
                ratio[st] = 1

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

        total_portfolio_metrics ={"total_trades": round(total_trades,2),
                                 "total_invested":round(costs_total_portfolio,2),
                                  "total_after":round(funds_total_portfolio,2),
                                  "overall_annualised_return": round((funds_total_portfolio / max(costs_total_portfolio,1) )** (1 / total_years) - 1,2),
                                  "batting_average": round(total_gains/max(total_gains+total_losses,1),2),
                                  "total_years":total_years}

        return metrics, total_portfolio_metrics

class buy_and_hold(strategy):
    """
    Naive strategy to simply purchase the stock as soon as data becomes available and never sell it until the end.
    """

    def __init__(
            self,
            stock_data: List[stock],
            purchase_frequency: int,
            name: str = "buy_and_hold"):

        super().__init__(
            stock_data,
            name,
            purchase_frequency)

    def update_portfolio(self,stock):

        # Create one buy group signal on the first day and fill
        stock.price_history['buy_sell_signal'] = True

        # update portfolio for the stock
        self.update_portfolio_stock(stock)



class red_white_blue(strategy):
    def __init__(self,
                 stock_data: List[stock],
                 name = "Red_white_blue",
                 purchase_frequency = 1,
                 emas_st_used: List[int] = None,
                 emas_lt_used: List[int] = None):

        super().__init__(
            stock_data,
            name,
            purchase_frequency)

        if emas_st_used is None:
            self.emas_st_used = [3, 5, 8, 10, 12, 15]
        else:
            self.emas_st_used = emas_st_used

        if emas_lt_used is None:
            self.emas_lt_used = [30, 35, 40, 45, 50, 60]
        else:
            self.emas_lt_used = emas_lt_used


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
            stock_price_history["cmin"] = stock_price_history[["ema_" + str(x) for x in self.emas_st_used]].min(axis=1)
            stock_price_history["cmax"] = stock_price_history[["ema_" + str(x) for x in self.emas_lt_used]].max(axis=1)

            stock_data.append(
                stock(name = name,
                      date_start = date_start,
                      price_history = stock_price_history)
            )

        self.stock_data = stock_data


    def update_portfolio(self,stock):

        # Create labels by measuring groups of contiguous buy signals
        stock.price_history['buy_sell_signal'] = np.where(stock.price_history['cmin'] > stock.price_history['cmax'] , True, False)

        # update portfolio for the stock
        self.update_portfolio_stock(stock)





class DCF(strategy):
    def __init__(self,stock_data: List[stock],
                 purchase_frequency,
                 safety_factor_buy,
                 safety_factor_sell,
                 name="DCF"):

        super().__init__(stock_data, name,purchase_frequency)
        assert safety_factor_buy >=1
        assert safety_factor_sell >= 1
        self.safety_factor_buy = safety_factor_buy
        self.safety_factor_sell = safety_factor_sell

    def update_portfolio(self, date):

        # Loop through all of the stocks on our stock market buy / sell if you can
        for st in self.stock_data:
            if date not in self.date_lists[st.name]:
                pass
            elif date == self.max_dates[st.name]:
                if self.currently_hold_stock(st):
                    self.sell(st, date)
            else:
                dcf = st.price_history.DCF[date]
                price = st.price_history.adjclose[date]

                if dcf > price*self.safety_factor_buy and not self.currently_hold_stock(st):
                    self.buy(st, date)
                elif dcf < price/self.safety_factor_sell and self.currently_hold_stock(st):
                    self.sell(st, date)
                else:
                    pass

class ROC_PE(strategy):
    def __init__(self,stock_data: List[stock],
                 purchase_frequency,
                 pe_limit = 25,
                 roce_limit = 0.1,
                 name="DCF_PE"):

        super().__init__(stock_data, name,purchase_frequency)
        assert pe_limit >0
        assert roce_limit>0
        self.pe_limit = pe_limit
        self.roce_limit = roce_limit

    def update_portfolio(self, date):

        # Loop through all of the stocks on our stock market buy / sell if you can
        for st in self.stock_data:
            if date not in self.date_lists[st.name]:
                pass
            elif date == self.max_dates[st.name]:
                if self.currently_hold_stock(st):
                    self.sell(st, date)
            else:
                roce = st.price_history.ROCE[date]
                price = st.price_history.adjclose[date]
                pe_ratio = price / st.price_history["EPS"][date]


                if roce > self.roce_limit and pe_ratio<self.pe_limit and pe_ratio>0 and not self.currently_hold_stock(st):
                    self.buy(st, date)
                elif (roce < self.roce_limit or pe_ratio>self.pe_limit) and self.currently_hold_stock(st):
                    self.sell(st, date)
                else:
                    pass



class ROC_DCF(strategy):
    def __init__(self,stock_data: List[stock],
                 purchase_frequency,
                 safety_factor_buy = 1,
                 roce_limit = 0.1,
                 name="DCF_PE"):

        super().__init__(stock_data, name, purchase_frequency)
        assert safety_factor_buy >= 1
        assert roce_limit>0
        self.safety_factor_buy = safety_factor_buy
        self.roce_limit = roce_limit

    def update_portfolio(self, date):

        # Loop through all of the stocks on our stock market buy / sell if you can
        for st in self.stock_data:
            if date not in self.date_lists[st.name]:
                pass
            elif date == self.max_dates[st.name]:
                if self.currently_hold_stock(st):
                    self.sell(st, date)
            else:
                roce = st.price_history.ROCE[date]
                price = st.price_history.adjclose[date]
                dcf = st.price_history.DCF[date]


                if roce > self.roce_limit and dcf>price*self.safety_factor_buy and not self.currently_hold_stock(st):
                    self.buy(st, date)
                elif (roce < self.roce_limit or dcf<price/self.safety_factor_buy) and self.currently_hold_stock(st):
                    self.sell(st, date)
                else:
                    pass