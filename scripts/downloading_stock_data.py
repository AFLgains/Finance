import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from yahoofinancials import YahooFinancials

stock_tocker_str = "DIS"


# Using yfinance to get the stock price over time
yf.pdr_override()
stock_ticker = yf.Ticker(stock_tocker_str)

startYear = 2000
startMonth = 1
startDay = 1
start = dt.datetime(startYear, startMonth, startDay)
end = dt.datetime.now()
# Download the stock price
df = yf.download(stock_tocker_str,start,end)


# Now lets use yahoo_financials
yahoo_financials= YahooFinancials(stock_tocker_str)

data = yahoo_financials.get_historical_price_data(start_date=start.strftime("%Y-%m-%d"),end_date=end.strftime("%Y-%m-%d"),time_interval='daily')
data_df = pd.DataFrame(data[stock_tocker_str]['prices'])
data_df = data_df.drop('date', axis=1).set_index('formatted_date')

# Lets also use yahoo financials to obtain company data
all_statement_data_qt =  yahoo_financials.get_financial_stmts('quarterly', ['income', 'cash', 'balance'])
all_statement_data_annual =  yahoo_financials.get_financial_stmts('quarterly', ['income', 'cash', 'balance'])



table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
#df.to_csv('S&P500-Info.csv')
#df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])