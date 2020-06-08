import datetime as dt
import pickle
import random

import FundamentalAnalysis as fa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.data_classes import stock
from yahoofinancials import YahooFinancials

data_df = pickle.load(
    open("C:\Personalprojects\Finance\scripts\stock_history_data.pkl", "rb")
)

for d in data_df:
    plt.scatter(d.price_history.ROCE, d.price_history.adjclose / d.price_history.EPS)
    plt.xlim([0, 1])
    plt.ylim([0, 300])


plt.title("ROCE vs (P/E)")
plt.show()
