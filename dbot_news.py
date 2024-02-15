import requests
from bs4 import BeautifulSoup
import datetime
import time
import webbrowser


def initialize():
    url = f"https://ng.investing.com/economic-calendar/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    news = soup.find(id="economicCalendarData", class_="genTbl closedTbl ecoCalTbl persistArea js-economic-table")
    news = news.find("tbody")
    news = news.find_all("tr",class_="js-event-item")
    return(news)

while True:
    try: 
        news = initialize()
    except Exception as e:
        print(e)
        webbrowser.open("https://ng.investing.com/economic-calendar/", new = 1)
        print("please wait..., as i test website on browser")
        time.sleep(5)
        news = initialize()


    currency = ['USDCAD','USDJPY','EURUSD','GBPUSD','XAUUSD','AUDUSD']
    impact_color = ['blackFont','greenFont','redFont']
    order_operation = False
    strength = []# 0 is neutral 1 is strong and 2 is weak

    hour = 14
    min = 30
    # Get the current local time
    now = datetime.datetime.now()
    # Extract hour and minute
    #hour = now.hour
    #min = now.minute

    print("current time:",hour,":",min)
    
    if hour > 19: break

    print("Total number of news ",len(news))
    for i in range(len(news)):
        target_news = news[i].find_all("td")      
        t_currency = str(target_news[1]).split(' ')
        t_currency = t_currency[-1].replace("</td>","")
        
        if t_currency == "USD":#the currency i am concerned about
            t_impact = str(target_news[2]).split("High Volatility Expected")
            if len(t_impact) > 1 : 
                #print("News with high impact detected for ", t_currency)
                t_time = str(target_news[0]).replace('<td class="first left time js-time" title="">','')
                t_time = t_time.replace("</td>",'')
                d_time = t_time.split(":")
                print(d_time)
                if len(d_time) <= 1:
                    if len(t_time.split("min")) > 1:   
                        print(t_time, "Please wait as actual value will soon be released")
                else:
                    if hour < 10:
                        if "0"+str(hour) == d_time[0]:
                            if "0"+str(min) == d_time[1]:
                                for i in range(len(impact_color)):
                                    t_actual = str(target_news[4]).split(impact_color[i])
                                    if len(t_actual) <= 1 : 
                                        print("The news is still neutral at the moment") 
                                    else:
                                        if i == 1:
                                            print(t_currency, " is very strong and can be used for quick trading")
                                            order_operation = True
                                            strength.append(i)
                                            break
                                        elif i == 2:
                                            print(t_currency, " is very weak and can be used for quick trading")
                                            order_operation = True
                                            strength.append(i)
                                            break
                                        else:
                                            print(t_currency," is neutral, no point trading with it")
                                            break
                            elif str(min) == d_time[1]:
                                for i in range(len(impact_color)):
                                    t_actual = str(target_news[4]).split(impact_color[i])
                                    if len(t_actual) <= 1 : 
                                        print("The news is still neutral at the moment") 
                                    else:
                                        if i == 1:
                                            print(t_currency, " is very strong and can be used for quick trading")
                                            order_operation = True
                                            strength.append(i)
                                            break
                                        elif i == 2:
                                            print(t_currency, " is very weak and can be used for quick trading")
                                            order_operation = True
                                            strength.append(i)
                                            break
                                        else:
                                            print(t_currency," is neutral, no point trading with it")
                                            break
                            else:
                                print("Please wait on some minutes")
                                print(t_time)
                        
                    else:
                        if str(hour) == d_time[0]:
                            if "0"+str(min) == d_time[1]:
                                for i in range(len(impact_color)):
                                    t_actual = str(target_news[4]).split(impact_color[i])
                                    if len(t_actual) <= 1 : 
                                        print("The news is still neutral at the moment") 
                                    else:
                                        if i == 1:
                                            print(t_currency, " is very strong and can be used for quick trading")
                                            order_operation = True
                                            strength.append(i)
                                            break
                                        elif i == 2:
                                            print(t_currency, " is very weak and can be used for quick trading")
                                            order_operation = True
                                            strength.append(i)
                                            break
                                        else:
                                            print(t_currency," is neutral, no point trading with it")
                                            break
                            elif str(min) == d_time[1]:
                                for i in range(len(impact_color)):
                                    t_actual = str(target_news[4]).split(impact_color[i])
                                    if len(t_actual) <= 1 : 
                                        print("The news is still neutral at the moment") 
                                    else:
                                        if i == 1:
                                            print(t_currency, " is very strong and can be used for quick trading")
                                            order_operation = True
                                            strength.append(i)
                                            break
                                        elif i == 2:
                                            print(t_currency, " is very weak and can be used for quick trading")
                                            order_operation = True
                                            strength.append(i)
                                            break
                                        else:
                                            print(t_currency," is neutral, no point trading with it")
                                            break
                            else:
                                print("Please wait on some minutes")
                                print(t_time)         

        if(i == len(news) - 1 and order_operation):
            #Monitor and Calculate type of trade and also pull out after 5 mins count
            strong  = 0
            weak = 0
            for i in range(len(strength)):
                #Strength counting 
                if strength[i] == 1:
                    strong += 1
                elif strength[i] == 2:
                    weak += 1
            print("total number of strong news ",strong)   
            print("total number of weak news ",weak)  

            monitor = True

            if strong == len(strength):
                print("USD is strong, and ready to trade")
                for i in range(len(currency)):
                    if i <= 1:
                        print("Buy market", currency[i])
                    else:
                        print("Sell market", currency[i])
            elif weak == len(strength):
                print("USD is weak, and ready to trade")
                for i in range(len(currency)):
                    if i > 1:
                        print("Buy market ", currency[i])
                    else:
                        print("Sell market", currency[i])
            else:
                print("USD today news is a waste of time, or very risky")
                monitor = False

            if monitor:
                for i in range(3 * 60):
                    time.sleep(1)
                    print("Please wait..., while i make sure break even is given to market ", i)
                    #Edit the prices

                ## Closing all the market
                for market in currency:
                    print(market, " closed")
                
                print("All market closed for the day")
            order_operation = False
            strength = []
    time.sleep(10)



