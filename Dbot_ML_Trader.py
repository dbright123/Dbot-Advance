import MetaTrader5 as mt5
import numpy as np
import joblib
import time
from datetime import datetime,timezone

print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)


lot = 0.01

target_market = ["GBPUSD","USDCAD","AUDUSD","USDCHF","NZDUSD","EURUSD"]
models = []
sc_xs = []
sc_ys = []


print("Please wait while i learn from the data showing from your broker")

if not mt5.initialize():
    print('Initialization failed, check internet connection. You must have Meta Trader 5 installed.')
else:
    print(mt5.account_info()._asdict())
    print("\n")
    print(mt5.terminal_info()._asdict())