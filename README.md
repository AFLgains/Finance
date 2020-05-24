# UAX2-Finance edition
 Repo used to assess stock picking strategies. Financial and stock price data for 1000's of stocks are downloaded and various fincial metrics are applied to create a picking strategy. The strategies are then tested on 10+ years worth of stock data.  
 
# Data Sources
Data is taken from SimFin (https://simfin.com/)
 
# How to use
You can implement the backtest through the main file [backtest.py](scripts/backtest.py) 
Use environment variables `NAMED_RUN` and `API_KEY` to specify a run:
```bash
NAMED_RUN=us API_KEY=fake123 python scripts/backtest.py
```
You can view and edit the runs [here](scripts/configuarion/scenarios.yml).
Please note you will also need a valid API KEY to obtain the full data. If you don't have one then simply use `API_KEY=free`



 
 
