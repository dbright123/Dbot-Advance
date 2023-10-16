import MetaTrader5 as mt5
import numpy as np
import joblib

import time
from datetime import datetime, timezone


print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)

def r_squared(y_true, y_pred):
  mean_y_true = np.mean(y_true)
  ss_tot = np.sum((y_true - mean_y_true)**2)
  ss_res = np.sum((y_true - y_pred)**2)
  r_squared = 1 - ss_res / ss_tot
  return r_squared



target_market = ["GBPUSD","USDCAD","XAUUSD"]
models = []
sc_xs = []
sc_ys = []
for market in target_market:
    models.append(joblib.load(market+" regressor.joblib"))
    sc_xs.append(joblib.load(market+" sc_x.joblib"))
    sc_ys.append(joblib.load(market+" sc_y.joblib"))

n = 0

if not mt5.initialize():
    print('Initialization failed, check internet connection. You must have Meta Trader 5 installed.')

    
else:
    print(mt5.account_info()._asdict())
    print("\n")
    print(mt5.terminal_info()._asdict())
    while(True):
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        lot = 0.01
        print(account.equity)
        if(terminal.connected == True and terminal.trade_allowed == True):
            print("AI is functional loading "+target_market[n])
            model = models[n]
            sc_x = sc_xs[n]
            sc_y = sc_ys[n]
            
            rates = mt5.copy_rates_from_pos(target_market[n], mt5.TIMEFRAME_H4, 0, 500)
            print(rates[0][0])
            print(rates.shape)
            data = []
            close_price = []
            open_price = []

            for i in range(len(rates)):
                data.append([rates[i][0],rates[i][1],rates[i][5]])
                open_price.append(rates[i][1])
                close_price.append(rates[i][4])
            data = np.array(data)
            close_price = np.array(close_price)

            data = sc_x.transform(data)
            y_pred = model.predict(data)

            y_pred = sc_y.inverse_transform(y_pred.reshape((len(y_pred),1)))
            y_pred = y_pred.reshape(-1)

            r_squared = r_squared(close_price[-100:], y_pred[-100:])

            print(r_squared, " is the current prediction model performance")
            if(r_squared <= 85):
                print(target_market[n]+" will need re-training, please train the model again or check program for error, the prediction is too poor")
                time.sleep(5)
                print("checking other market")
                break

            data = sc_x.inverse_transform(data)

            # creating an assumption on the system
            print(close_price[-1:])
            print(data[-1:,:])
            y_pred = model.predict(sc_x.transform(data[-1:,:]))
            y_pred = sc_y.inverse_transform(y_pred.reshape((len(y_pred),1)))
            y_pred = y_pred.reshape(-1)
            print(y_pred)

            # set time zone to UTC
            # Convert Unix epoch time to UTC time
            utc_time = datetime.fromtimestamp(int(data[-1,0]))


            # Get the current datetime in UTC
            now_utc = datetime.utcnow()

            # Convert the UTC datetime to GMT +0
            now_gmt0 = now_utc.astimezone(timezone.utc)

            # Get the year, month, day, and hour from the GMT +0 datetime
            year = now_gmt0.year
            month = now_gmt0.month
            day = now_gmt0.day
            hour = now_gmt0.hour
            mins = now_gmt0.minute
            secs = now_gmt0.second
            # Print the results
            print("Year:", year)
            print("Month:", month)
            print("Day:", day)
            print("Hour:", hour)
            print("Minute:", mins)
            print("Seconds:", secs)

     
            symbol = target_market[n]
            price = mt5.symbol_info_tick(symbol).bid
            print("current price for ",target_market[n]," is ",price)


            permit_trade = False
            modify_trade = False
            o_price = 0
            c_price = 0
            profit = 0
            lot_size = 0
            order_type = 0
            sl = 0


            
            if(y_pred[-1] > price):
                price = mt5.symbol_info_tick(symbol).ask
                #buying a market
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": price,
                    "sl": 0.0,
                    "tp": y_pred[-1],
                    "deviation": 20,
                    "magic": 0,
                    "comment": "Dbot_ML",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                permit_trade = True
            elif(y_pred[-1] < price):
                price = mt5.symbol_info_tick(symbol).bid
                #Selling a market
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": 0.0,
                    "tp": y_pred[-1],
                    "deviation": 20,
                    "magic": 0,
                    "comment": "Dbot_ML",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                permit_trade = True




            if(permit_trade):
                print("Trade activation on "+target_market[n])

                if(mt5.positions_total() == 0):
                    #Ordering the trade
                    result=mt5.order_send(request)
                    print(result)

                else:
                    #checking if the trade exist so as to modified it 
                    order_symbols = mt5.positions_get()

                    for order_symbol in order_symbols:
                        #print(order_symbol)
                        if(target_market[n] == order_symbol.symbol):
                            print("seen")
                            target_order = order_symbol
                            print(target_order)
                            o_price = target_order.price_open
                            c_price = target_order.price_current
                            profit = target_order.profit
                            lot_size = target_order.volume
                            order_type = target_order.type

                            print("open price ", o_price)
                            print("close_price ",c_price)
                            print("profit ",profit)
                            print("lot size ",lot_size)
                            print("order type ",order_type)
                            modify_trade = True
                            break


                if(modify_trade == False):
                    result=mt5.order_send(request)
                    print(result)



            #current stage
            if(modify_trade):
                #modify the market
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": target_order.symbol,
                    "sl": 0.0,
                    "tp": y_pred[-1],
                    "position": target_order.ticket
                }
                price = mt5.symbol_info_tick(symbol).bid
                close_trade = False
                if(order_type == 0):
                    if(y_pred[-1] > price):
                        result=mt5.order_send(request)
                        print(result)
                    elif(y_pred[-1] < price):
                        close_trade = True
                elif(order_type == 1):
                    if(y_pred[-1] < price):
                        result=mt5.order_send(request)
                        print(result)
                    elif(y_pred[-1] > price):
                        close_trade = True
                    
            if(close_trade):
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": target_order.symbol,
                    "volume" : target_order.volume,
                    'ticket': target_order.ticket,
                    "type" : target_order.type,
                    "price" : target_order.price_current
                }
                result=mt5.order_send(request)
                print(result)



            if(target_order.profit >= target_order.volume * 100 and target_order.sl == 0):
                #modify the market
                sl = target_order.price_current + target_order.price_open
                sl = sl/2
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": target_order.symbol,
                    "sl": sl,
                    "tp": y_pred[-1],
                    "position": target_order.ticket
                }
                result=mt5.order_send(request)
                print(result)

            if(n < len(target_market)):
                n += 1
            else:
                n = 0



            
            print(mt5.last_error())
            break


        else:
            print("Please make sure metatrade 5 has internet and algo Trade is Turn On")
    

mt5.shutdown()
quit()
