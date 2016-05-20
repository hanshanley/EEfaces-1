'''
Author: Kevin Wang
Last updated: 2/10/16
Used with Python 2.7

Description: 
Depending on command-line argument, download HTML contents of specified URL(s),
then retrieves and save all images into a directory titled raw. 
Also generates a csv file with the labels (names) of each image 
Beware: may overwrite contents of '/raw'
'''
import urllib, urllib2, datetime, sys, csv, os
from bs4 import BeautifulSoup

# prints usage info to stdout and exits program
def printUsage():
    print "  usage: python scrapephotos.py [x]"
    print "  x is an optional integer to specify scraping the following:"
    print "\t1 faculty"
    print "\t2 admin"
    print "\t3 technical staff"
    print "\t4 grad students"
    print "\t5 research staff"
    print "\t6 undergrads"
    print "\t7 all but undergrads"
    print "  (if x is unspecified, all of the above)"
    sys.exit()

# handling command-line arguments
if len(sys.argv) == 1: 
    print "default - scraping all fields"
    urlselection = 0 
elif len(sys.argv) == 2:
    try:
        urlselection = int(sys.argv[1])
    except ValueError:
        printUsage()
    if urlselection < 1 or urlselection > 7:
        printUsage()
else:
    printUsage()


outputDir = 'raw'
curDir = os.path.dirname(os.path.realpath(__file__)) # current directory
saveDir = os.path.join(curDir, outputDir)            # dir to save to (raw)
# create it if it doesn't already exist
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
    
url1 = "http://www.ee.princeton.edu/people/faculty"        # faculty
url2 = "http://www.ee.princeton.edu/people/administrative" # admin
url3 = "http://www.ee.princeton.edu/people/technical"      # tech staff
url4 = "http://www.ee.princeton.edu/people/grad-students"  # grad students
url5 = "http://www.ee.princeton.edu/people/research-staff" # research staff
url6 = "https://www.princeton.edu/collegefacebook/search/?order" \
        "=last_name&sort=asc&view=photo&academics=Electrical%20Engineering"\
        "&page=1&limit=1000" # undergrads

urlls = [url1, url2, url3, url4, url5, url6]

if urlselection == 7:
    urlls = urlls[:-1]
elif urlselection != 0:
    urlls = [urlls[urlselection - 1]]


labelDict = {} # maps person name to filename of images

for url in urlls:
    # print where we are at 
    if url == url1:
        print "scraping faculty"
    elif url == url2:
        print "scraping admin"
    elif url == url3:
        print "scraping tech staff"
    elif url == url4:
        print "scraping grad students"
    elif url == url5:
        print "scraping research staff"
    elif url == url6:
        print "scraping undergrads"

    if url != url6: # first 5 urls are from EE department page
        response = urllib2.urlopen(url)
        htmlContents = response.read()

        soup = BeautifulSoup(htmlContents, 'html.parser')
        rows = soup.find_all('div', class_='views-row')
        for row in rows:
            # retrive the name
            namediv = row.find('div', class_='views-field-title')
            if not namediv: # sometimes namediv is stored under an h2 tag
                namediv = row.find('h2')
            name = namediv.find('a').string

            # replace the pesky unicode characters with a u
            name = ''.join([i if ord(i) < 128 else 'u' for i in name]) 

            # retrive the img URL
            imgurl = row.find('img')['src']

            if 'face_' in imgurl: # mark down with None when no picture 
                filename = "None"
            else:
                # save the image
                extension = imgurl.split(".")[-1][:3] # img extension
                filename = "raw_" + "_".join(name.split()) + "." + extension
                urllib.urlretrieve(imgurl, "raw/" + filename)

            # also store info into labelDict
            labelDict[name] = filename
    else: # last url is undergrads, for college facebook, needs authentication
        # create password manager
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

        # add username and password -- need a valid Princeton id
        username = "VALID_NETID"
        password = "VALID_PASSWORD"
        top_level_url = "https://www.princeton.edu/collegefacebook/"
        password_mgr.add_password(None, top_level_url, username, password)

        # create opener
        handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        opener = urllib2.build_opener(handler)
        opener.open(url)

        # install the opener. calls to urlopen will use opener
        urllib2.install_opener(opener)

        # open url and read it
        response = urllib2.urlopen(url)
        htmlContents = response.read()

        soup = BeautifulSoup(htmlContents, 'html.parser')
        # for some reason, can't directly get all img tags...
        divresult = soup.find('div', id='facebook-results')
        imgs = divresult.find_all('img')
        urlbase = 'https://www.princeton.edu'

        for imgtag in imgs:
            imgurl = urlbase + imgtag['src']
            name = imgtag['title']

            # replace any pesky unicode characters with a ?
            name = ''.join([i if ord(i) < 128 else '?' for i in name])     

            filename = "raw_" + "_".join(name.split()) + '.png'
            urllib.urlretrieve(imgurl, "raw/" + filename) # save the file

            # also store info into labelDict
            labelDict[name] = filename

# save the labels dictionary in a csv file 
csvOutpath = os.path.join(saveDir, 'raw_labels.csv') 
with open(csvOutpath, 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Name", "Filename"])
    for key, value in labelDict.items():
       writer.writerow([key, value])

print "done"