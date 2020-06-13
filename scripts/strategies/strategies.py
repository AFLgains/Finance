import datetime as dt
import pickle
import time
from typing import List

import numpy as np
import pandas as pd
from datasources.data_classes import (portfolio, stock, stock_purchase, trade_class, strategy_metrics)
from datasources.performance import test_strategy
from yahoofinancials import YahooFinancials
from functools import reduce
from collections import defaultdict
from pypfopt import EfficientFrontier, objective_functions, risk_models, expected_returns


class strategy:
    """
    Super class
    """

    def __init__(self,stock_data: List[stock],name,purchase_frequency=1,redistribute = False):
        self.strategy_name = name
        self.stock_data = self.check_stock_data(stock_data)
        self.purchase_frequency = purchase_frequency
        self.dates_considered = self.get_unique_dates(self.stock_data,"formatted_date")
        self.yearly_purchase_dates = self.get_unique_dates(self.stock_data, "year_first_day")
        self.date_lists = {st.name: st.price_history.formatted_date.tolist() for st in self.stock_data}
        self.min_dates = {st.name: min(st.price_history.formatted_date) for st in self.stock_data}
        self.max_dates = {st.name: max(st.price_history.formatted_date) for st in self.stock_data}
        self.portfolio = portfolio(past_purchases = [],current_purchases = [],date = min(self.dates_considered))
        self.stock_dictionary ={stock.name:stock for stock in self.stock_data}
        self.redistribute = redistribute
        self.opt_port = True

    def get_unique_dates(self,x,data_col: str):
        dates = set()
        for st in x:
            dates = dates.union(st.price_history[data_col].tolist())

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

    def buy(self,stock_name: str, date):
        self.portfolio.current_purchases.append(
            stock_purchase(
                name=stock_name,
                price_bought=float(self.stock_dictionary[stock_name].price_history.adjclose[date]),
                date_bought =  date,
                date_sold =  "",
                price_sold = np.nan,
                status = "open",
                pct_change = 0,
            )
        )


    def sell(self,stock_name: str, date):

        stock_to_sell = self.get_current_purchase(stock_name)
        if date not in self.stock_dictionary[stock_name].price_history.index:
            #print("Selling at a date before true selling date")
            date_actually_sold = max(self.stock_dictionary[stock_name].price_history.index)
            sell_price = float(self.stock_dictionary[stock_name].price_history.adjclose[date_actually_sold])
        else:
            sell_price = float(self.stock_dictionary[stock_name].price_history.adjclose[date])

        # Add to the past purchases list
        self.portfolio.past_purchases.append(
            stock_purchase(
                name=stock_to_sell.name,
                price_bought=stock_to_sell.price_bought,
                date_bought =  stock_to_sell.date_bought,
                date_sold =  date,
                price_sold = sell_price,
                status = "closed",
                pct_change = (sell_price - stock_to_sell.price_bought) / stock_to_sell.price_bought
            )
        )

        # Remove from current purchases
        self.portfolio.current_purchases = [stock_purchases for stock_purchases in self.portfolio.current_purchases if stock_purchases.name != stock_name]


    def get_current_purchase(self, stock_name):
        current_purchase = [stock_purchase_item for stock_purchase_item in self.portfolio.current_purchases if stock_purchase_item.name == stock_name ]
        return current_purchase[0]


    def currently_hold_stock(self,stock_name):
        if len(self.portfolio.current_purchases)>0:
            return any([stock_name==cp.name for cp in self.portfolio.current_purchases])
        return False

    def get_flat_distriutions(self,portfolio_cash,buys):
        output = {}
        total_cash = sum([v for x, v in portfolio_cash.items()])
        for b in buys:
            output[b.name] = total_cash / len(buys)
        return output

    def gen_price_history_df(self, tickers):

        data_frames = [self.stock_dictionary[b].price_history[["adjclose"]].rename(columns={"adjclose": b}) for b in tickers]
        df_merged = reduce(
            lambda left, right: pd.merge(left, right, on="formatted_date", how="left"),
            data_frames,
        )

        return df_merged

    def get_optimal_distributions(self,portfolio_cash,buys,buy_date):
        output = {}
        total_cash = sum([v for x, v in portfolio_cash.items()])
        price_histories = self.gen_price_history_df([b.name for b in buys])

        # Obtain the historical return and sample cov.
        mu = expected_returns.mean_historical_return(price_histories)
        S = risk_models.sample_cov(price_histories)

        # Optimise for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=1)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        # Add it to buys
        for b in buys:
            output[b.name] = cleaned_weights[b.name]*total_cash
        return output


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

        # Portfolio level metrics
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
            total_gains += ng[st]
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

        dates_sold = set([x.date_sold for x in self.portfolio.past_purchases])
        dates_bought = set([x.date_bought for x in self.portfolio.past_purchases])
        transaction_dates = dates_sold.union(dates_bought)

        # Every stock you can start with a dollar
        stock_purchased_cash = {st:1 for st in purchased_stock_names}
        portfolio_cash = {st:0 for st in purchased_stock_names}
        trades = {}

        for purchase_date in sorted(list(transaction_dates)):
            # obtain sells and buys
            sells = [p for p in self.portfolio.past_purchases if p.date_sold == purchase_date]
            buys = [p for p in self.portfolio.past_purchases if p.date_bought == purchase_date]

            # first do sells
            if sells:
                for s in sells:
                    if trades[s.name].stock_purchase_data.date_sold == purchase_date:
                        # Find the trade that it refers to:
                        tr = trades[s.name]
                        portfolio_cash[s.name] = tr.amount*(1+tr.stock_purchase_data.pct_change)
                        trades[s.name].status = "closed"
            # then do buys
            if buys:
                if self.redistribute:
                    incr_redistribute = {}

                    if self.opt_port and len(buys)>1:
                        incr_redistribute = self.get_optimal_distributions(portfolio_cash =portfolio_cash, buys = buys, buy_date = purchase_date)
                    else:
                        incr_redistribute = self.get_flat_distriutions(portfolio_cash =portfolio_cash, buys = buys)

                    for b in buys:
                        trades[b.name] = trade_class(stock_purchase_data = b,
                                               amount = stock_purchased_cash[b.name] + incr_redistribute[b.name],
                                               status = "open")
                        stock_purchased_cash[b.name] = 0
                    for x in portfolio_cash.keys():
                        portfolio_cash[x] = 0
                else:
                    for b in buys:
                        incr = portfolio_cash[b.name]
                        trades[b.name] = trade_class(stock_purchase_data = b,
                                               amount = stock_purchased_cash[b.name] + incr,
                                               status = "open")
                        stock_purchased_cash[b.name] = 0
                        portfolio_cash[b.name] -= incr

        total_invested = len(purchased_stock_names)
        total_cash_end = sum([v for x,v in portfolio_cash.items()])
        total_portfolio_metrics = {"total_trades": round(len(self.portfolio.past_purchases), 2),
                                   "total_invested": round(total_invested, 2),
                                   "total_after": round(total_cash_end, 2),
                                   "overall_annualised_return": round(
                                       (total_cash_end / max(total_invested, 1)) ** (
                                               1 / total_years) - 1, 2),
                                   "batting_average": round(total_gains / max(total_gains + total_losses, 1), 2),
                                   "total_years": total_years}


        return metrics, total_portfolio_metrics

class buy_and_hold(strategy):
    """
    Naive strategy to simply purchase the stock as soon as data becomes available and never sell it until the end.
    """

    def __init__(
            self,
            stock_data: List[stock],
            purchase_frequency: int,
            redistribute = False,
            name: str = "buy_and_hold"):

        super().__init__(
            stock_data,
            name,
            purchase_frequency,
            redistribute)


    def update_portfolio(self,stock):

        # Create one buy group signal on the first day and fill
        stock.price_history['buy_sell_signal'] = True

        # update portfolio for the stock
        self.update_portfolio_stock(stock)


class buy_and_hold_year(strategy):
    """
    Naive strategy to simply purchase the stock as soon as data becomes available and never sell it until the end.
    """

    def __init__(
            self,
            stock_data: List[stock],
            purchase_frequency: int,
            redistribute=False,
            name: str = "buy_and_hold_yearly"):
        super().__init__(
            stock_data,
            name,
            purchase_frequency,
            redistribute)

    def run(self):
        # Build any metrics that need to be added to the stock price history
        self.build_metrics()

        # Loop the reevaluate dates

        #for date in portfolio_re_evaluate_dates:
        for date in self.yearly_purchase_dates:
            self.update_portfolio(date) # for every buy date update the porfoilio during this data

        # Sell everything at the end
        for st in self.stock_data:
            if self.currently_hold_stock(st.name):
                self.sell(st.name,max(st.price_history.formatted_date))

    def update_portfolio(self, buy_date):

        # Sell all stocks,
        if len(self.portfolio.current_purchases):
            for st in self.portfolio.current_purchases:
                self.sell(st.name, buy_date)

        # Apply the filtering criteria
        stock_to_purchase = {st for st in self.stock_dictionary.keys() if buy_date in self.stock_dictionary[st].price_history.index}

        if len(stock_to_purchase):
            for stock_name in stock_to_purchase:
                self.buy(stock_name, buy_date)

class red_white_blue(strategy):
    def __init__(self,
                 stock_data: List[stock],
                 name = "Red_white_blue",
                 purchase_frequency = 1,
                 redistribute = False,
                 emas_st_used: List[int] = None,
                 emas_lt_used: List[int] = None):

        super().__init__(
            stock_data,
            name,
            purchase_frequency,
            redistribute)


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

class ROC_PE(strategy):
    def __init__(self,stock_data: List[stock],
                 purchase_frequency,
                 redistribute = False,
                 stock_limit = 10,
                 pe_upper_limit = 25,
                 pe_lower_limit = 0,
                 roce_lower_limit = 0.5,
                 roce_upper_limit=100,
                 min_revenue = 100000000,
                 name="ROC_PE"):

        super().__init__(stock_data, name,purchase_frequency,redistribute)
        assert pe_upper_limit>=0
        assert pe_lower_limit>=0
        assert roce_lower_limit >0
        assert roce_upper_limit>0
        self.stock_limit = stock_limit
        self.pe_upper_limit = pe_upper_limit
        self.pe_lower_limit = pe_lower_limit
        self.roce_lower_limit = roce_lower_limit
        self.roce_upper_limit = roce_upper_limit
        self.min_revenue = min_revenue



    def run(self):
        # Build any metrics that need to be added to the stock price history
        self.build_metrics()

        # Loop the reevaluate dates
        portfolio_re_evaluate_dates = self.yearly_purchase_dates
        for date in portfolio_re_evaluate_dates:
            self.update_portfolio(date) # for every buy date update the porfoilio

        # Sell everything at the end
        for st in self.stock_data:
            if self.currently_hold_stock(st.name):
                self.sell(st.name,max(st.price_history.formatted_date))

    def rank_ratios(self, x, reverse: bool = True):
        output = {}
        for rank, key in enumerate(sorted(x, key=x.get, reverse=reverse), 1):
            output[key]= rank
        return output


    def update_portfolio(self, buy_date):

        # Sell all stocks,
        if len(self.portfolio.current_purchases):
            for st in self.portfolio.current_purchases:
                self.sell(st.name, buy_date)

        # Measure the ratio
        pe_ratio_unfiltered = {stock.name:stock.price_history['P/E'][buy_date] for stock in self.stock_data if buy_date in stock.price_history.index}
        roce_ratio_unfiltered = {stock.name: stock.price_history['roce_ratio'][buy_date] for stock in self.stock_data if buy_date in stock.price_history.index}
        revenue_unfiltered = defaultdict(int)
        for stock in self.stock_data:
            if buy_date in stock.price_history.index:
                revenue_unfiltered[stock.name] = stock.price_history["Revenue"][buy_date]

        # Apply the filtering criteria
        pe_ratio = {k:v for k,v in pe_ratio_unfiltered.items() if v <= self.pe_upper_limit and v >= self.pe_lower_limit and revenue_unfiltered[k] > self.min_revenue}
        roce_ratio = {k: v for k, v in roce_ratio_unfiltered.items() if v <= self.roce_upper_limit and v >= self.roce_lower_limit and revenue_unfiltered[k] > self.min_revenue }

        if len(pe_ratio) and len(roce_ratio):

            # Rank each and construct a score
            pe_ratio_rank = self.rank_ratios(pe_ratio, reverse = False)
            roce_ratio_rank = self.rank_ratios(roce_ratio)

            # score and take top n
            ranked_companies = pe_ratio_rank.keys() & roce_ratio_rank.keys()
            total_score = {s: pe_ratio_rank[s] + roce_ratio_rank[s] for s in ranked_companies}
            rank_score = self.rank_ratios(total_score, reverse = False)


            # Buy the top n
            for stock_name, rank in sorted(rank_score.items() ,  key=lambda x: x[1]):
                if rank<=self.stock_limit:
                    self.buy(stock_name, buy_date)

class MOD_LIL_BOOK(strategy):
    def __init__(self,stock_data: List[stock],
                 purchase_frequency,
                 redistribute=False,
                 stock_limit = 10,
                 pe_upper_limit = 40,
                 pe_lower_limit = 5,
                 roa_lower_limit = 0.25,
                 roa_upper_limit=100,
                 min_revenue = 100000000,
                 name="ROC_PE"):

        super().__init__(stock_data, name,purchase_frequency,redistribute)
        assert pe_upper_limit>=0
        assert pe_lower_limit>=0
        assert roa_lower_limit >0
        assert roa_upper_limit>0
        self.stock_limit = stock_limit
        self.pe_upper_limit = pe_upper_limit
        self.pe_lower_limit = pe_lower_limit
        self.roa_lower_limit = roa_lower_limit
        self.roa_upper_limit = roa_upper_limit
        self.min_revenue = min_revenue



    def run(self):
        # Build any metrics that need to be added to the stock price history
        self.build_metrics()

        # Loop the reevaluate dates
        portfolio_re_evaluate_dates = self.yearly_purchase_dates
        for date in portfolio_re_evaluate_dates:
            self.update_portfolio(date) # for every buy date update the porfoilio

        # Sell everything at the end
        for st in self.stock_data:
            if self.currently_hold_stock(st.name):
                self.sell(st.name,max(st.price_history.formatted_date))



    def rank_ratios(self, x, reverse: bool = True):
        output = {}
        for rank, key in enumerate(sorted(x, key=x.get, reverse=reverse), 1):
            output[key]= rank
        return output


    def update_portfolio(self, buy_date):

        # Sell all stocks,
        if len(self.portfolio.current_purchases):
            for st in self.portfolio.current_purchases:
                self.sell(st.name, buy_date)

        # Measure the ratio
        pe_ratio_unfiltered = {stock.name:stock.price_history['P/E'][buy_date] for stock in self.stock_data if buy_date in stock.price_history.index}
        roa_ratio_unfiltered = defaultdict(int)
        for stock in self.stock_data:
            if buy_date in stock.price_history.index:
                roa_ratio_unfiltered[stock.name] = stock.price_history["Return on Assets"][buy_date]

        revenue_unfiltered = defaultdict(int)
        for stock in self.stock_data:
            if buy_date in stock.price_history.index:
                revenue_unfiltered[stock.name] = stock.price_history["Revenue"][buy_date]

        # Apply the filtering criteria
        pe_ratio = {k:v for k,v in pe_ratio_unfiltered.items() if v <= self.pe_upper_limit and v >= self.pe_lower_limit and revenue_unfiltered[k] >= self.min_revenue and roa_ratio_unfiltered[k]>=self.roa_lower_limit}

        if len(pe_ratio):

            # Rank each and construct a score
            pe_ratio_rank = self.rank_ratios(pe_ratio, reverse = False)

            # score and take top n
            ranked_companies = pe_ratio_rank.keys()
            total_score = {s: pe_ratio_rank[s] for s in ranked_companies}
            rank_score = self.rank_ratios(total_score, reverse = False)

            # Buy the top n
            for stock_name, rank in sorted(rank_score.items() ,  key=lambda x: x[1]):
                if rank<=self.stock_limit:
                    self.buy(stock_name, buy_date)

class MOD_WARI_B(strategy):
    def __init__(self,stock_data: List[stock],
                 purchase_frequency,
                 redistribute=False,
                 stock_limit =10,
                 name="MOD_WARI_B"):
        super().__init__(stock_data, name,purchase_frequency,redistribute)
        self.stock_limit = stock_limit


    def run(self):
        # Build any metrics that need to be added to the stock price history
        self.build_metrics()

        # Loop the reevaluate dates

        #for date in portfolio_re_evaluate_dates:
        for date in self.yearly_purchase_dates:
            self.update_portfolio(date) # for every buy date update the porfoilio during this data

        # Sell everything at the end
        for st in self.stock_data:
            if self.currently_hold_stock(st.name):
                self.sell(st.name,max(st.price_history.formatted_date))

    def rank_ratios(self, x, reverse: bool = True):
        output = {}
        for rank, key in enumerate(sorted(x, key=x.get, reverse=reverse), 1):
            output[key]= rank
        return output

    def gen_metric_dict(self,buy_date,metric_str: str):
        metric_dict = {}
        for stock in self.stock_data:
            if buy_date in stock.price_history.index:
                metric_dict[stock.name] = stock.price_history[metric_str][buy_date]

        return metric_dict

    def update_portfolio(self, buy_date):

        # Measure the ratio
        pe_ratio = self.gen_metric_dict(buy_date, "P/E")
        assets_to_liability = self.gen_metric_dict(buy_date, "Current Ratio")
        revenue= self.gen_metric_dict(buy_date, "Revenue")
        price_to_book = self.gen_metric_dict(buy_date, "P/Book")
        ave_earnings_growth_n_periods = self.gen_metric_dict(buy_date, "mean_earnings_growth_n_periods")
        net_income = self.gen_metric_dict(buy_date, "Net Income")
        net_income_to_book_g_limit_n_years = self.gen_metric_dict(buy_date, "net_income_to_book_g_limit_n_periods")
        net_income_positive_all_history = self.gen_metric_dict(buy_date, "net_income_positive_all_history")

        #stock_pool = pe_ratio.keys()|assets_to_liability.keys()|revenue.keys()|price_to_book.keys()|ave_earnings_growth_n_years.keys()|net_income.keys()|net_income_to_book_g_limit_n_years.keys()


        # Apply the filtering criteria
        pe_ratio_list = {st:v for st,v in pe_ratio.items() if
                             pe_ratio[st] > 10 and
                             pe_ratio[st] < 40 and
                             assets_to_liability[st] > 1 and
                             revenue[st] > 1000000000 and
                             price_to_book[st] < 5 and
                             price_to_book[st] > 0 and
                             ave_earnings_growth_n_periods[st] > 0.1/4 and
                             net_income_to_book_g_limit_n_years[st] == True and
                             net_income_positive_all_history[st] == True
                             }

        #if len(stock_to_purchase):
        #    for stock_name in stock_to_purchase:
        #        self.buy(stock_name, buy_date)


        if len(pe_ratio_list):

            # Rank each and construct a score
            pe_ratio_rank = self.rank_ratios(pe_ratio_list, reverse = False)

            # score and take top n
            ranked_companies = pe_ratio_rank.keys()
            total_score = {s: pe_ratio_rank[s] for s in ranked_companies}
            rank_score = self.rank_ratios(total_score, reverse = False)

            # Sell all stocks,
            for st in self.portfolio.current_purchases:
                    self.sell(st.name, buy_date)

            # Buy the top n
            for stock_name, rank in sorted(rank_score.items() ,  key=lambda x: x[1]):
                if rank<=self.stock_limit:
                    self.buy(stock_name, buy_date)
