'''
Function to test scraping http://ee.princeton.edu/slider
'''
from bs4 import BeautifulSoup
import urllib2

# parse the html contents of http://ee.princeton.edu/slider,
# and retrieve the center div with the class slider-container
# replace all image srcs to link to the website, and return the BS object
def load_slidercontainer():

    # get the html from ee.princeton.edu/slider, load into bs
    url = "http://ee.princeton.edu/slider/"
    response = urllib2.urlopen(url)
    htmlContents = response.read()
    soup = BeautifulSoup(htmlContents, 'html.parser')

    # slide container class is the single div 
    rows = soup.find_all('div', class_='slide-container')
    for img in rows[0].find_all('img'):
        host = "http://ee.princeton.edu"
        # get the original image src, remove the first backslash
        orig = img['src']
        # new img src should be of the form:  "http://ee.princeton.edu/sites/default/files/Car Lab 2016.jpg"
        img['src'] = host + orig
    return rows[0]

