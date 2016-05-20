from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
import os, json, datetime, time, pytz
from bs4 import BeautifulSoup
import urllib2

# Helper method:
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

# View:
# display and load the default EE slider by scraping its main contents from the actual website
# plus our Javascript and CSS. We only update the slider-container into our copy of index.html
def index(request):
    currdir = os.path.dirname(os.path.dirname(__file__)) # curr directory
    currdir += '/templates/eeslides_app/'
    readfilepath = currdir + 'index.html'

    # TODO: fix this hacky soln. right now, bs messes up the formatting for django static files
    importsfilepath = currdir + 'imports.html'

    # open the current index.html page
    filedata = None
    if os.path.isfile(readfilepath):
        with open(readfilepath, 'r') as readfile:
            filedata = readfile.read()

    # replace the old slide-container with the new one from load_slidercontainer()
    if filedata is not None:
        soup = BeautifulSoup(filedata, "html.parser")
        try:
            newrow = load_slidercontainer()
            soup.find_all('div', class_='slide-container')[0].contents = newrow.contents
        except:
            print 'Could not load updated EE slider'
        else:

            html = soup.prettify('utf-8')

            # TODO: fix this hacky solution to imports, bs messes up formatting 
            with open(importsfilepath, 'r') as readfile:
                document= readfile.read()         
                parts = html.split('<!--IMPORTS-->')
                parts[1] = document
                html = ''.join(parts)

            # update our copy of the index.html
            with open(readfilepath, 'w+') as outfile:
               outfile.write(html)

    return render(request, 'eeslides_app/index.html')

# AJAX backend for polling for faces recognized 
def check_faces(request):
    if request.method == 'GET':
        names = [] # list of names to send to front-end

        # parse the JSON file updates.json
        currdir = os.path.dirname(os.path.dirname(__file__)) # curr directory
        currdir += '/eeslides/'

        readfilepath = currdir + 'static/updates_facerec.json'
        writefilepath = currdir + 'static/updates_django.json'
        if os.path.isfile(readfilepath):
            with open(readfilepath, 'r') as readfile:
                filedata = json.load(readfile)
        else:
            filedata = {}

        timefilepath = currdir + 'static/last_updates.json'
        if os.path.isfile(timefilepath):
            with open(timefilepath, 'r') as readfile:
                timedata = json.load(readfile)
        else:
            timedata = {}

        # convert since UTC is 4 hours ahead
        now = datetime.datetime.now() - datetime.timedelta(hours=4)
        currdate = now.strftime("%Y-%m-%d")

        # remove anything that took place over a week ago
        newfiledata = {}
        for i in filedata.keys():
            date_object = datetime.datetime.strptime(i, "%Y-%m-%d")
            if (now - date_object).total_seconds() < 60*60*24*7:
                newfiledata[i] = filedata[i]

        # look at all predictions today, find the ones that haven't been updated
        # for all in the current date
        if currdate in newfiledata:
            for p in newfiledata[currdate]:
                # only look at predictions within last 5 seconds
                if not p["parsed"] and time.time() - p["time"] < 5:
                    p["parsed"] = True
                    pred = p['prediction']
                    if pred in timedata:
                        lastpredtime = timedata[pred]
                        # wait at least 10 seconds before adding it again
                        if time.time() - lastpredtime > 10:
                            names.append(pred)
                            timedata[pred] = time.time()
                    else:
                        names.append(pred)
                        timedata[pred] = time.time()

        # save the updated file
        with open(writefilepath, 'w+') as outfile:
            json.dump(newfiledata,outfile, indent = 4, sort_keys = True)

         # save the updated file
        with open(timefilepath, 'w+') as outfile:
            json.dump(timedata,outfile, indent = 4, sort_keys = True)

        # take the unique names only
        names = list(set(names))
        print names
        return JsonResponse({"names": names})
    else:
        return JsonResponse({"names": []})
