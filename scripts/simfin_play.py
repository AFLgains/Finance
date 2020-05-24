import simfin as sf
from simfin.names import *
import os
import random

# Set your API-key for downloading data.
# If the API-key is 'free' then you will get the free data,
# otherwise you will get the data you have paid for.
# See www.simfin.com for what data is free and how to buy more.
sf.set_api_key('AH4RWcm8001Mmz48dGzbZ13LGp1O5Yqw')
#sf.set_api_key('free')

# Set the local directory where data-files are stored.
# The dir will be created if it does not already exist.
sf.set_data_dir(os.path.join("C:\Personalprojects\Finance\scripts","raw_data"))

# Load the annual Income Statements for all companies in USA.
# The data is automatically downloaded if you don't have it already.
#df = sf.load_income(variant='quarterly-full', market='us')
df_income = sf.load_income(variant='ttm-full', market='us')
#df_balance = sf.load_balance(variant='annual-full', market='us')
#df_cash = sf.load_cashflow(variant='annual-full', market='us')

#df_income.to_csv("C:\Personalprojects\Finance\scripts\scripts"+"\statement_income_nice.csv")
#df_balance.to_csv("C:\Personalprojects\Finance\scripts\scripts"+"\statement_balance_nice.csv")
#df_cash.to_csv("C:\Personalprojects\Finance\scripts\scripts"+"\statement_cash_nice.csv")
#df = sf.load_balance(variant='annual-full', market='us')
#df = sf.load_cashflow(variant='annual-full', market='us')
#df = sf.load_companies(market='us')
# Print all Revenue and Net Income for Microsoft (ticker MSFT).
stock_tocker_str = random.sample(set(df_income.reset_index().Ticker),1000)
";".join(stock_tocker_str)

see = df_income.loc[['TSLA','DIS',"MSFT"], [REVENUE, COST_REVENUE, GROSS_PROFIT, INCOME_TAX]]
print(df_income.loc['TSLA', [CAPEX]])