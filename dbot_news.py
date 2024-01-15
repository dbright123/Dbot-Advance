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

try: 
    news = initialize()
except Exception as e:
    print(e)
    webbrowser.open("https://ng.investing.com/economic-calendar/")
    print("please wait..., as i test website on browser")
    time.sleep(5)
    news = initialize()
