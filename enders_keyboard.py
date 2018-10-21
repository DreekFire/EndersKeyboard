from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 

def search(n):
    browser = webdriver.Chrome(r"./chromedriver")

    if n==1:
        browser.get('http://www.google.com')
        search = browser.find_element_by_name("q")
        search.send_keys("fortnite")
        search.submit()

    elif  n==2:
        browser.get('http://www.youtube.com')
        search = browser.find_element_by_name("search_query")
        search.send_keys("fortnite")
        search.submit()

    elif  n==3:
        browser.get('https://github.com')
        search = browser.find_element_by_name("q")
        search.send_keys("fortnite")
        search.submit()

    elif  n==4:
        browser.get('https://www.wikipedia.org/')
        search = browser.find_element_by_name("search")
        search.send_keys("fortnite")
        search.submit() 

    elif  n==5:
        browser.get('https://www.twitch.tv/')
        search = browser.find_element_by_id("nav-search-input")
        search.send_keys("fortnite")
        search.submit() 