import FundamentalAnalysis as fa
import pandas as pd
import matplotlib.pyplot as plt

stock_ticker_str = "TQQQ"

table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df_stock_tickers = table[0]

current_valuation = {}
#for i in range(len(df_stock_tickers)):
for i in range(2):
    try:
        print("Stock " + str(i) + ":"+df_stock_tickers.iloc[i,1])
        ticker = df_stock_tickers.iloc[i,0]
        dcf_annual = fa.discounted_cash_flow(ticker, period="annual")
        current_valuation[ticker] = {"stock_price":dcf_annual.iloc[1,0],"DCF":dcf_annual.iloc[2,0],"Company_name":df_stock_tickers.iloc[i,1]}
    except:
        print("Stock ticker "+df_stock_tickers.iloc[i,1]+" not found")

results = pd.DataFrame.from_dict(current_valuation,orient='index')
plt.scatter(x = results.stock_price[0:500],y = results.DCF[0:500])
plt.xlabel("Current Stock Price")
plt.ylabel("DCF stock price")
plt.xlim([10,1000])
plt.ylim([10,1000])
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

results.to_csv('dcf_vs_stockprice.csv')


# Obtain the financial statements
#balance_sheet_annually = fa.balance_sheet_statement(stock_ticker_str, period="annual")
#income_statement_annually = fa.income_statement(stock_ticker_str, period="annual")
#cash_flow_statement_annually = fa.cash_flow_statement(stock_ticker_str, period="annual")


