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
    impact_color = ['greenFont','redFont','blackFont']

    try:
        print(len(news))
        for i in range(len(news)):
            target_news = news[i].find_all("td")
            
            t_currency = str(target_news[1]).split(' ')
            t_currency = t_currency[-1].replace("</td>","")
            print(t_currency)

            t_impact = str(target_news[2]).split("High Volatility Expected")
            if len(t_impact) <= 1 : 
                print("This news dooesn't have the impact i am looking for") 
            else: 
                print("News with high impact detected for ", t_currency)
                t_time = str(target_news[0]).replace('<td class="first left time js-time" title="">','')
                t_time = t_time.replace("</td>",'')

                t_time = t_time.split(":")
                print(t_time)
                t_hour = 0
                t_min = 0 

                print("currency time ",t_hour,":",t_min)

                # Get the current local time
                now = datetime.datetime.now()

                # Extract hour and minute
                hour = now.hour
                min = now.minute

                print("current time:",hour,":",min)
                '''
                for target in currency:
                    if target == t_currency:
                        print("currency is among target, please wait while i check for information on the currency")
                        if hour == t_hour:
                            if min >= t_min:
                                print("The news market is active")
                                for i in range(len(impact_color)):
                                    t_actual = str(target_news[4]).split(impact_color[i])
                                    if len(t_actual) <= 1 : 
                                        print("The news is still neutral at the moment") 
                                    else:
                                        if i == 0:
                                            print(t_currency, " is very strong and can be used for quick trading")
                                            break
                                        elif i == 1:
                                            print(t_currency, " is very weak and can be used for quick trading")
                                            break
                                        else:
                                            print(t_currency," is neutral, no point trading with it")
                                            break
                            else:
                                print("Please wait for just few minutes for the market actual value to be updated")
                        else:
                            print("The news market is not active")
                    break
                '''
    except Exception as e:
        print(e)
    time.sleep(20)

