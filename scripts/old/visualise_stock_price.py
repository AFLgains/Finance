import pandas as pd
import datetime as dt
from yahoofinancials import YahooFinancials
from data.data_classes import stock
import pickle
import FundamentalAnalysis as fa
import numpy as np
import random
import matplotlib.pyplot as plt



data_df = pickle.load(open('C:\Personalprojects\Finance\scripts\stock_history_data.pkl','rb'))

for d in data_df:
    plt.scatter(d.price_history.ROCE,d.price_history.adjclose/d.price_history.EPS)
    plt.xlim([0,1])
    plt.ylim([0, 300])


plt.title("ROCE vs (P/E)")
plt.show()