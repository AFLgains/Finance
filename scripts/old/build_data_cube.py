APIKEY = a4dbc32f50b52a11f45e25117c6aa749

import pandas as pd
import datetime as dt
from yahoofinancials import YahooFinancials
from data.data_classes import stock
import pickle
import FundamentalAnalysis as fa
from yahoofinancials import YahooFinancials as yf
import numpy as np
import random
import matplotlib.pyplot as plt


# Define starting date for back test

# Use wikipedia to obtain a list of stocks
#table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
#df_stock_tickers = table[0]
#df_stock_tickers = df_stock_tickers.loc[df_stock_tickers.Symbol != "T", :]
#df_stock_tickers["Date first added"] = pd.to_datetime(df_stock_tickers["Date first added"])
#stock_tocker_str = list(df_stock_tickers.Symbol)
##stock_tocker_str = ["DIS","TSLA","V","MSFT","AMD","NVDA"]
#stock_tocker_str = random.sample(stock_tocker_str,100)



def validate_stock_data(price_history):
    return len(price_history.index)>200 and "adjclose" in price_history.columns

def transform_data(price_history):
    dat = price_history.copy()
    dat["formatted_date"] = pd.to_datetime(dat.index)
    dat = dat.loc[~np.isnan(dat.adjclose),]
    dat.set_index("formatted_date", drop=True, inplace=True)
    return dat

def add_dcf(stock_name,price_history,min_date,max_date):
    # Initially download all DCF data
    dcf_quarterly = fa.discounted_cash_flow(stock_name, period="quarter").transpose()
    dcf_quarterly.rename(columns={"date": "formatted_date"}, inplace=True)
    dcf_quarterly["formatted_date"] = pd.to_datetime(dcf_quarterly["formatted_date"])
    dcf_quarterly.sort_values("formatted_date", inplace=True)
    dcf_quarterly.set_index("formatted_date", drop=True, inplace=True)
    dcf_quarterly = dcf_quarterly[['DCF']]

    # expand to count for all days
    dcf_quarterly_all = pd.DataFrame({"formatted_date": pd.date_range(start=min_date, end=max_date)})
    dcf_quarterly_all = dcf_quarterly_all.merge(dcf_quarterly, on="formatted_date", how='left')
    dcf_quarterly_all["DCF"] = dcf_quarterly_all["DCF"].ffill()
    dcf_quarterly_all.set_index("formatted_date", drop=True, inplace=True)

    # Merge to the price history
    price_history = pd.merge(price_history, dcf_quarterly_all, on="formatted_date", how='left')

    return(price_history)



def add_key_metrics(stock_name,price_history,min_date,max_date):
    # Initially download all DCF data
    key_metrics_quarterly = fa.key_metrics(stock_name, period="quarter").transpose()
    key_metrics_quarterly["formatted_date"] = pd.to_datetime(key_metrics_quarterly.index)
    key_metrics_quarterly.sort_values("formatted_date", inplace=True)
    key_metrics_quarterly.set_index("formatted_date", drop=True, inplace=True)
    key_metrics_quarterly = key_metrics_quarterly[["PE ratio"]].astype(float)

    # expand to count for all days
    key_metrics_quarterly_all = pd.DataFrame(
        {"formatted_date": pd.date_range(start=min_date, end=max_date)})
    key_metrics_quarterly_all = key_metrics_quarterly_all.merge(key_metrics_quarterly, on="formatted_date", how='left')
    key_metrics_quarterly_all = key_metrics_quarterly_all.ffill()
    key_metrics_quarterly_all.set_index("formatted_date", drop=True, inplace=True)

    # Merge to the price history
    price_history = pd.merge(price_history, key_metrics_quarterly_all, on="formatted_date", how='left')

    return(price_history)

def add_financial_statement_data(stock_name,price_history,min_date,max_date):

    # Initially download all DCF data
    balance_sheet = fa.balance_sheet_statement(stock_name, period="quarter").transpose()
    balance_sheet["formatted_date"] = pd.to_datetime(balance_sheet.index)
    balance_sheet.sort_values("formatted_date", inplace=True)
    balance_sheet.set_index("formatted_date", drop=True, inplace=True)
    balance_sheet = balance_sheet[["Total assets","Total current liabilities","Goodwill and Intangible Assets"]].astype(float)

    # Income sheet
    income_sheet = fa.income_statement(stock_name, period="quarter").transpose()
    income_sheet["formatted_date"] = pd.to_datetime(income_sheet.index)
    income_sheet.sort_values("formatted_date", inplace=True)
    income_sheet.set_index("formatted_date", drop=True, inplace=True)

    income_sheet["EBIT"] = income_sheet["EBIT"].astype(float)
    income_sheet["Net Income"] = income_sheet["Net Income"].astype(float)
    income_sheet["Preferred Dividends"] = income_sheet["Preferred Dividends"].astype(float)
    income_sheet["Weighted Average Shs Out"] = income_sheet["Weighted Average Shs Out"].astype(float)

    income_sheet["earnings"] = income_sheet["Net Income"]- income_sheet["Preferred Dividends"]
    income_sheet["EBIT"] = income_sheet["EBIT"].rolling(4).sum() # Take rolling sum
    income_sheet["earnings"] = income_sheet["earnings"].rolling(4).sum() # Take rolling sum
    income_sheet["EPS"] =income_sheet["earnings"]/income_sheet["Weighted Average Shs Out"]
    income_sheet["earnings_growth"] = (income_sheet["earnings"] - income_sheet["earnings"].shift(1)) / income_sheet["earnings"].shift(1)
    income_sheet = income_sheet[['EPS','EBIT',"earnings_growth"]]

    # ROCE
    total_financials =  pd.merge(balance_sheet,income_sheet,on = "formatted_date", how='left')
    total_financials['ROCE'] = total_financials['EBIT']/(total_financials['Total assets'] - total_financials['Total current liabilities'] - 0*total_financials['Goodwill and Intangible Assets'])

    # Expand to count for all days
    total_financials_all = pd.DataFrame({"formatted_date": pd.date_range(start=min_date, end=max_date)})
    total_financials_all = total_financials_all.merge(total_financials, on="formatted_date", how='left')
    total_financials_all = total_financials_all.ffill()
    total_financials_all.set_index("formatted_date", drop=True, inplace=True)
    total_financials_all = total_financials_all[['ROCE','EPS','earnings_growth']]

    # Merge to the price history
    price_history = pd.merge(price_history, total_financials_all, on="formatted_date", how='left')

    return(price_history)

def main(n_stocks = 100):
    stock_data_dict = pickle.load(open('C:\Personalprojects\Finance\scripts\stock_history_price_data.pkl', 'rb'))
    stock_tocker_str = [ticker for ticker in stock_data_dict.keys()]
    stock_tocker_str = random.sample(stock_tocker_str,min(n_stocks,len(stock_tocker_str)) )

    data_df = []
    total_stocks_downloaded = 0
    print("Downloading stocks...")
    for stock_name in stock_tocker_str:
        try:

            assert len(stock_data_dict[stock_name]) == 6

            stock_data = pd.DataFrame(stock_data_dict[stock_name]['prices']).drop('date', axis=1).set_index('formatted_date')
            if validate_stock_data(stock_data):
                data_df.append(stock(name=stock_name,
                                 price_history=transform_data(stock_data),
                                 date_start=pd.to_datetime(stock_data.index[0])
                                 )
                               )
                total_stocks_downloaded+=1
                print("Total stocks downloaded: ",total_stocks_downloaded)
            else:
                pass
        except:
            pass

    print("...done")

    start_str = min([min(d.price_history.index) for d in data_df])
    end_str = max([max(d.price_history.index) for d in data_df])

    print("Appending on company financials")
    total_stocks_downloaded = 0
    final_data_df = []
    for i,st in enumerate(data_df):
        try:
            # Add DCF
            st.price_history = add_dcf(st.name, st.price_history,start_str,end_str)
            st.price_history = add_financial_statement_data(st.name,st.price_history,start_str,end_str)

            # Get rid of NANs in key fields #TODO: Make this better
            st.price_history = st.price_history.loc[~np.isnan(st.price_history.DCF),]
            st.price_history = st.price_history.loc[~np.isnan(st.price_history.adjclose),]
            st.price_history = st.price_history.loc[~np.isnan(st.price_history.ROCE),]
            st.price_history = st.price_history.loc[~np.isnan(st.price_history.EPS) ,]
            st.price_history = st.price_history.loc[st.price_history.EPS != float('inf'),]


            # Make formatted date an index
            st.price_history["formatted_date"]  = st.price_history.index
            if len(st.price_history.index)>10:
                final_data_df.append(st)
                total_stocks_downloaded+=1
                print("Total final stocks: ",total_stocks_downloaded,"/",i+1)
        except:
            pass


    with open('old/stock_history_data.pkl', 'wb') as f:
        pickle.dump(final_data_df, f)

if __name__=="__main__":
    main(n_stocks=500)


#plt.plot(see.index,see["adjclose"])
#plt.plot(see.index,see["DCF"],'r-')
#plt.show()
#ticker = stock_tocker_str[0]
#dcf_annual = fa.discounted_cash_flow(ticker, period="annual")
#current_valuation[ticker] = {"stock_price": dcf_annual.iloc[1, 0], "DCF": dcf_annual.iloc[2, 0],
                             #"Company_name": df_stock_tickers.iloc[i, 1]}