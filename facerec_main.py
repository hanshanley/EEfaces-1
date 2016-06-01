'''
Author: Kevin Wang
Last updated: 6/1/16    by Sanket Satpathy
Used with Python 2.7

Description: 
This is the backend program for a live face recognition system using a webcam for the Equad display. 
Command-line argument -t is the directory to search for training photos. 
Needs the train_labels.csv file to load photos and their labels, otherwise it will attempt
to infer photos/labels from that directory.
Trains the recognizer using the photos from the training directory. 
Starts the webcam capture. Runs a face detector on each frame to detect the face.
For each detected face, run the prediction from the recognizer to decide which face it is.
Save confident outputs into the JSON file that is parsed by the local website backend. 

This refreshes itself every 24 hours, to retrain from the same directory
'''
import numpy as np, cv2, argparse, os, sys, csv, dlib, time, operator, math, json, datetime
np.set_printoptions(precision=2)
from scipy.misc import imresize
from PIL import Image
from recognizer_util import get_images_and_labels
from recognizer_util import infer_images_and_labels
from openface_recognizer import OpenfaceRecognizer
from pprint import pprint

F_WIDTH = 640
F_HEIGHT = 480
display_feed = False#True

# flag for whether or not to use the VGG recognizer for verification
useVGG = False
if useVGG:
    from vgg_recognizer import VGGRecognizer

# the minimum proportion of the weight we need to be "confident"
# about a face and save it to a file
WEIGHT_CONFIDENCE = 40
NEAREST_NEIGHBORS = 2

# for tracking faces
class TrackFace:
    # x,y location of the center of the face
    x = None
    y = None

    # how many frames ago this face was updated
    lastUpdated = None

    # max distance (in pixels) to consider it to be the same face
    max_distance_tolerance = 55

    # fractional error within which we consider it to be the same face
    area_tolerance = 0.2

    # time within which we consider it to be the same face
    time_tolerance = 1

    # constructor. Pass in location of the center of the face
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.lastUpdated = time.time() 

    # check if a detected face should be considered the same face, using only location
    def checkSimilar(self, x, y, w, h):
        distance = math.sqrt((self.x - x)**2 + (self.y - y)**2)
        area_difference = abs(self.w * self.h - w * h) / float(self.w * self.h)
        time_difference = time.time() - self.lastUpdated
        if distance > self.max_distance_tolerance or \
            area_difference > self.area_tolerance or \
            time_difference > self.time_tolerance:
            self.lastUpdated = time.time()#+= 1   # only counts frames!!  these frames may be far apart in time
            return False
        else: 
            self.lastUpdated = time.time()#0 
            return True
            

# initializes an opencv webcam with specified parameters
def initializeWebcam(cam=0, width=640, height=480):
    cap = cv2.VideoCapture(cam)
    cap.set(3, width)  # 3 is width
    cap.set(4, height) # 4 is height
    return cap

# update the JSON file containing the predictions with the predicted name
# prediction, and the time of prediction
def updateJSON(prediction, predictiontime):
    readfilepath = './eeslides/eeslides/static/updates_django.json'
    writefilepath = './eeslides/eeslides/static/updates_facerec.json'

    try:
        if os.path.isfile(readfilepath):
            with open(readfilepath, 'r') as readfile:
                filedata = json.load(readfile)
        else:
            filedata = {}

        now = datetime.datetime.now()
        currdate = now.strftime("%Y-%m-%d")

        if currdate not in filedata:
            filedata[currdate] = []
        filedata[currdate].append({"prediction": prediction, "time": predictiontime, "parsed":False})

        with open(writefilepath, 'w+') as outfile:
            json.dump(filedata, outfile, indent = 4, sort_keys = True)
    except:
        print 'WARNING: Unable to read JSON file'
        pass

# retrieves training images and trains the passed recognizer on them
def train_routine(recognizer, trainingdir):
    print 'Retrieving training images.'
    now = time.time()
    images, labels = get_images_and_labels(trainingdir, 
        training=True, grayscale=False)
    
    print '\tTime : {0:.2f} s'.format(time.time() - now)

    print 'Training the recognizer.'
    now = time.time()
    ylabels = recognizer.train(images, labels)
    
    print '\tTime : {0:.2f} s'.format(time.time() - now)
    print 'People in the training set:',
    people = np.unique(ylabels)
    for i in range(len(people)):
        if (i % 4 == 0):
            print '\n\t {} : {}, '.format(i, people[i]),
        else:
            print '{}, '.format(people[i]),
    print ''
    return recognizer, ylabels

# returns the number of seconds until midnight
def secondsUntilMidnight():
    now = datetime.datetime.now()
    seconds_until_midnight = (now.replace(hour=23, minute=59, second=59) - now).total_seconds()
    return seconds_until_midnight

# the webcam capture loop -- it runs a continuous loop by collecting from the webcam
# until the specified timelimit has elapsed (in seconds)
def sub_routine(timelimit=86400):
    # construct argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--traindir", required=True, 
        help="directory for training faces")
    args = vars(ap.parse_args())

    # using dlib's histogram oriented gradients to detect the region of frame with the face
    dlib_detector = dlib.get_frontal_face_detector()

    # create a face recognizer 
    print 'Initializing recognizer.'
    now = time.time()
    if useVGG:
        recognizer = VGGRecognizer(knn=NEAREST_NEIGHBORS)
    else:
        recognizer = OpenfaceRecognizer(knn=NEAREST_NEIGHBORS)
    print '\tTime : {0:.2f} s'.format(time.time() - now)

    
    # train for the first time
    if useVGG:
        recognizer, ylabels_vgg = train_routine(recognizer, args['traindir'])
    else:
        recognizer, ylabels = train_routine(recognizer, args['traindir'])

    # set up webcam
    fwidth = F_WIDTH            # default frame dimensions
    fheight = F_HEIGHT
    cap = initializeWebcam(cam=0,width=fwidth,height=fheight)
    # for tracking fps
    prevtime = None
    currtime = None

    # for tracking faces
    trackedfaces = []
    lastJSONsave = None
    JSONsavePeriod = 5

    print 'Starting the capture.'
    timelimit_start = time.time()
    # Main running loop
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = frame[:-50,:,:]
        fheight, fwidth = frame.shape[:-1]

        # now = time.time()
        x_offset = 0
        y_offset = 0
        dlibfaces = dlib_detector(frame)    # ~0.5 s
        if not dlibfaces:   # zoom in if no faces found
            x_offset = 0
            x_span = fwidth - 2 * x_offset
            y_offset = 100
            y_span = 210
            cv2.rectangle(frame,(x_offset, y_offset),(x_offset+x_span, y_offset+y_span),(0,255,0),1)
            dlibfaces = dlib_detector(frame[y_offset:y_offset+y_span, x_offset:x_offset+x_span].astype('uint8'), 1)
            if not dlibfaces:   # zoom in if no faces found
                x_offset = 0
                x_span = fwidth - 2 * x_offset
                y_offset = 220
                y_span = 150
                cv2.rectangle(frame,(x_offset, y_offset),(x_offset+x_span, y_offset+y_span),(0,0,255),3)
                dlibfaces = dlib_detector(frame[y_offset:y_offset+y_span, x_offset:x_offset+x_span].astype('uint8'), 2)


        # print '\tDetecting faces took {0:.2f} seconds'.format(time.time() - now)

        faces = []
        for df in dlibfaces:
            # extract the x, y coordinates and width/height, 
            # then increase box size by 1.5 times in all directions
            centerx = (df.right() - df.left())/2 + df.left()
            centery = (df.bottom() - df.top())/2 + df.top()
            scalefactor = 2.
            x = x_offset + max(0, int(centerx - (centerx - df.left())*scalefactor))
            y = y_offset + max(0, int(centery - (centery - df.top())*scalefactor))
            w = min(fwidth - x, int((centerx - df.left())*scalefactor*2))
            h = min(fheight - y, int((centery - df.top())*scalefactor*2))
            faces.append((x, y, w, h))

        # prediction all faces in the frame
        for (x,y,w,h) in faces:
            # look for a matching face
            matchedface = None

            for oldface in trackedfaces:
                if oldface.checkSimilar(x, y, w, h):
                    matchedface = oldface

            # remove faces that haven't appeared in the last 3 seconds
            for oldface in trackedfaces:
                if oldface.lastUpdated > 3:
                    trackedfaces.remove(oldface)

            # if no matching face found, then create a new one
            if matchedface is None:
                matchedface = TrackFace(x, y, w, h)
                trackedfaces.append(matchedface)
                # display rectangle and text over face, in green
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            else:
                # display rectangle and text over face, in blue
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # run the prediction and update the matchedface accordingly
            roi = frame[y:y+h, x:x+w]

            # run prediction over the ROI
            prediction = recognizer.verbose_predict(roi)

            # if we have a valid prediction
            if prediction is not None:
                dist, ind = prediction

                name = ylabels[ind[0,0]]
                weight = ((1. - dist[0,0]/dist[0,1])*100)

                if weight > WEIGHT_CONFIDENCE:
                    print (name, weight) , datetime.datetime.now()
                if weight > WEIGHT_CONFIDENCE and \
                    (lastJSONsave is None or \
                    time.time() - lastJSONsave > JSONsavePeriod):
                    updateJSON(name, time.time())
                    lastJSONsave = time.time()
                    JSONsavePeriod = 5 + 20 * np.random.rand()  # random delay

                if display_feed:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, name, (x, y),font, 0.8, (255,255,255))
            else:
                print 'NO PREDICTION!!'

                
        if display_feed:
            # display the FPS and seconds per frame in the top left corner
            prevtime = currtime
            currtime = time.time()
            if currtime and prevtime:
                timeperframe = currtime - prevtime
                fps = "fps : {0}".format(1/timeperframe)
                #tpf = "tpf : {0}".format(timeperframe)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, fps, (20, 20), font, 0.5, (255,0,0))


            # Display the resulting frame
            cv2.imshow('frame',frame)
            # quit if the user presses q on the screen, 
            if (cv2.waitKey(1) & 0xFF == ord('q')) or time.time() - timelimit_start > timelimit:
                break

    # When everything done, release the capture    
    cap.release()
    cv2.destroyAllWindows()

# the main loop
# it runs the sub_routine loops infinitely, for the specified amount of time
def main():
    seconds_until_midnight = secondsUntilMidnight()
    print 'Will reset itself in {} sec'.format(seconds_until_midnight)
    sub_routine(timelimit=seconds_until_midnight)
    print 'Exit successful'


if __name__ == '__main__':
    main()