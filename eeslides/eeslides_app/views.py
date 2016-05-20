from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
import os, json, datetime, time, pytz

# display and load the default EE homepage, with our modifications 
def index(request):
    #return HttpResponse("Hello World")
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
        print currdate

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
