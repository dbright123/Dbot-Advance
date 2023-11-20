import MetaTrader5 as mt5
import numpy as np
import joblib
import time
from datetime import datetime,timezone

print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)


lot = 0.01

target_market = ["GBPUSD","USDCAD","AUDUSD","USDCHF","NZDUSD","EURUSD","XAUUSD"] #list of market
#loading market model and standard scaler
models = []
sc_xs = []
sc_ys = []
for market in target_market:
    models.append(joblib.load(market+" regressor.joblib"))
    sc_xs.append(joblib.load(market+" sc_x.joblib"))
    sc_ys.append(joblib.load(market+" sc_y.joblib"))


n = 0 # counter variable

if not mt5.initialize():
    print('Initialization failed, check internet connection. You must have Meta Trader 5 installed.')

    
else:
    print(mt5.account_info()._asdict())
    print("\n")
    print(mt5.terminal_info()._asdict())
    while(True):
        terminal = mt5.terminal_info()
        if(terminal.connected == True and terminal.trade_allowed == True):
            account = mt5.account_info()
            print(account)
            print(account.equity)
            print("AI is functional loading "+target_market[n])
            model = models[n]
            sc_x = sc_xs[n]
            sc_y = sc_ys[n]
            
            rates = mt5.copy_rates_from_pos(target_market[n], mt5.TIMEFRAME_H4, 0, 500)
            print(rates[0][0])
            print(rates.shape)
            data = []
            close_price = []
           

            for i in range(len(rates)):
                data.append([rates[i][0],rates[i][1],rates[i][2],rates[i][3],rates[i][5]])
                close_price.append(rates[i][4])

            data = np.array(data)
            close_price = np.array(close_price)

            data = sc_x.transform(data)
            y_pred = model.predict(data)
  
            y_pred = sc_y.inverse_transform(y_pred.reshape((len(y_pred),1)))
            y_pred = y_pred.reshape(-1)
            #print(y_pred)

            r_squared = model.score(
                data,
                sc_y.transform(close_price.reshape((len(close_price),1))).reshape(-1)
            )
            print("stage 1")
            print(r_squared, " is the current prediction model performance")

            if(r_squared <= 0.95):
                print(target_market[n]+" will need re-training, please train the model again or check program for error, the prediction is too poor")
                print("checking other market")
                if(n < len(target_market)-1):
                    n += 1
                else:
                    n = 0

                time.sleep(20)

            else:
               # Get the current datetime in UTC
                now_utc = datetime.utcnow()

                # Convert the UTC datetime to GMT +0
                now_gmt0 = now_utc.astimezone(timezone.utc)

                # Get the year, month, day, and hour from the GMT +0 datetime
                year = now_gmt0.year
                month = now_gmt0.month
                day = now_gmt0.day
                hour = now_gmt0.hour + 1
                mins = now_gmt0.minute
                # Print the results
                print("Year:", year,"Month:", month,"Day:", day,"Minute:", mins)

                allow_trade = False
                #testing = True
                if(hour > 8 and hour < 18): allow_trade = True
                else: 
                    print("no new market will be able to get purchased due to late hour")
                    print("total profit is ",account.profit)
                    

                #Starting trade operation
        
                symbol = target_market[n]
                price = mt5.symbol_info_tick(symbol).bid
                print("current price for ",target_market[n]," is ",price, " predicted price is ",y_pred[0], " and close price on timeframe is ",close_price[0])


                permit_trade = False
                modify_trade = False
                o_price = 0
                c_price = 0
                profit = 0
                lot_size = 0
                order_type = 0
                sl = 0.0
                target_order = None

                
                if(y_pred[0] > price):
                    price = mt5.symbol_info_tick(symbol).ask
                    #buying a market
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": price,
                        "sl": 0.0,
                        "tp": y_pred[0],
                        "deviation": 20,
                        "magic": 0,
                        "comment": "Dbot_ML",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_RETURN,
                    }
                    permit_trade = True
                elif(y_pred[0] < price):
                    price = mt5.symbol_info_tick(symbol).bid
                    #Selling a market
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot,
                        "type": mt5.ORDER_TYPE_SELL,
                        "price": price,
                        "sl": 0.0,
                        "tp": y_pred[0],
                        "deviation": 20,
                        "magic": 0,
                        "comment": "Dbot_ML",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_RETURN,
                    }
                    permit_trade = True



                print("Stage 2")
                if(permit_trade):
                    price = mt5.symbol_info_tick(symbol).bid
                    if(n == len(target_market) - 1 and abs(y_pred[0] - price) > 2): #for Gold
                        permit_trade = True
                    elif(n < len(target_market) - 1 and abs(y_pred[0] - price) > 0.002): #for other 4 or 5 digit currency
                        permit_trade = True
                    else:
                        permit_trade = False


                if(permit_trade):
                    print("Trade activation on "+target_market[n])

                    if(mt5.positions_total() == 0):
                        #Ordering the trade
                        #here
                        if(allow_trade):
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
                                print("Stage 3")
                                ## Auto stoploss 
                                if(target_order.profit >= target_order.volume * 200  and target_order.sl == 0):
                                    #modify the market
                                    sl = target_order.price_current + target_order.price_open
                                    sl = sl/2
                                    request = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "symbol": target_order.symbol,
                                        "sl": sl,
                                        "tp": target_order.tp,
                                        "position": target_order.ticket
                                    }
                                    result=mt5.order_send(request)
                                    print(result)

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
                                if(order_type == 0 and target_order.tp < o_price):
                                    result = mt5.Close(target_order.symbol,ticket=target_order.ticket)
                                    print(result)
                                elif(order_type == 1 and target_order.tp > o_price):
                                    result = mt5.Close(target_order.symbol,ticket=target_order.ticket)
                                    print(result)
                                modify_trade = True
                                break


                    if(modify_trade == False):
                       # here
                        if(allow_trade):
                            result=mt5.order_send(request)
                            print(result)


                print("Stage 4")
                close_trade = False
                #current stage
                if(modify_trade):
                    #modify the market
                    if(target_order.sl != 0):
                        sl = target_order.sl
                    
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": target_order.symbol,
                        "sl": sl,
                        "tp": y_pred[0],
                        "position": target_order.ticket
                    }
                    price = mt5.symbol_info_tick(symbol).bid
                    close_trade = False
                    if(target_order.type == 0):
                        if(y_pred[0] > price and y_pred[0] != target_order.tp):
                            if(n == len(target_market) - 1 and abs(y_pred[0] - price) > 1): #for Gold
                                result=mt5.order_send(request)
                                print(result)
                            elif(n < len(target_market) - 1 and abs(y_pred[0] - price) > 0.001): #for other 4 or 5 digit currency
                                result=mt5.order_send(request)
                                print(result)
                        elif(y_pred[0] < price):
                            result = mt5.Close(target_order.symbol,ticket=target_order.ticket)
                            print(result)
                    elif(target_order.type == 1):
                        if(y_pred[0] < price and y_pred[0] != target_order.tp):
                            if(n == len(target_market) - 1 and abs(y_pred[0] - price) > 1): #for Gold
                                result=mt5.order_send(request)
                                print(result)
                            elif(n < len(target_market) - 1 and abs(y_pred[0] - price) > 0.001): #for other 4 or 5 digit currency
                                result=mt5.order_send(request)
                                print(result)
                        elif(y_pred[0] > price):
                            result = mt5.Close(target_order.symbol,ticket=target_order.ticket)
                            print(result)
                        
    
                
                if(n < len(target_market) - 1):
                    n += 1
                else:
                    n = 0

                print("Stage 5")

                
                print(mt5.last_error())
                
                
                time.sleep(20)
            
        else:
            print("Please make sure metatrade 5 has internet and algo Trade is Turn On")
            time.sleep(20)
    

mt5.shutdown()
quit()
