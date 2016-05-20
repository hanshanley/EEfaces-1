'''
Author: Kevin Wang
Last updated: 5/19/16
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


# flag for whether or not to use the VGG recognizer for verification
useVGG = 0
if useVGG == 1:
    from vgg_recognizer import VGGRecognizer

# the minimum proportion of the weight we need to be "confident"
# about a face and save it to a file
WEIGHT_CONFIDENCE = 0.6

# for tracking faces
class TrackFace:
    # x,y location of the center of the face
    x = None
    y = None

    # how many frames ago this face was updated
    lastUpdated = None

    # what was the last prediction for this face and their weights
    lastPredictions = None
    lastWeights = None

    # when the face was last predicted
    lastPredictionTime = None

    # how often to update face prediction, in seconds
    updatePeriod = 0.25

    # max distance to consider it to be the same face (in pixels)
    max_distance_tolerance = 55

    # max number of previous predictions to save and use for majority vote
    num_majority_vote = 10

    # time of last VGG prediction
    lastVGGTime = None
    # how often to update the VGG period
    VGGperiod = 10
    # how much extra the VGG weight is 
    VGGweight = 20

    # time of last JSON save
    lastJSONsave = None
    JSONsavePeriod = 5

    # constructor. Pass in location of the center of the face
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lastUpdated = 0 
        self.lastPredictions = []
        self.lastWeights = []
        self.lastVGGTime = None
        self.lastJSONsave = None

    # check if a detected face should be considered the same face, using only location
    def checkSimilar(self, x, y):
        distance = math.sqrt((self.x - x)**2 + (self.y - y)**2)
        if distance > self.max_distance_tolerance:
            self.lastUpdated += 1
            return False
        else: 
            self.lastUpdated = 0 
            return True

    # update the prediction with the given weight, and return the majority vote 
    # prediction of the last k predictions
    def updatePrediction(self, pred, weight=1.):
        if self.lastPredictionTime is None or time.time() - self.lastPredictionTime > self.updatePeriod:
            self.lastPredictionTime = time.time()
            self.lastPredictions.append(pred)
            self.lastWeights.append(weight)
            if len(self.lastPredictions) > 20:
                del self.lastPredictions[0]

        return self.getMajorityPrediction(print_=True)

        
    # return the majority vote prediction of the last k predictions
    def getMajorityPrediction(self, print_=False):
        if len(self.lastPredictions) == 0:
            return None
        # calculate weighted totals and return max weight
        scale = {}
        for i in range(0, len(self.lastPredictions)):
            predi = self.lastPredictions[i]
            weighti = self.lastWeights[i]
            if predi in scale:
                scale[predi] += weighti
            else:
                scale[predi] = weighti

        totalweight = float(sum(scale.values()))
        if print_:
            # print confidence measures, normaled from 0 to 1, sorted from high to low
            for name, weight in sorted(scale.iteritems(), key=operator.itemgetter(1))[::-1]:
                print '\t',
                print name,
                print "{0:.2f}".format(weight/totalweight)
        return max(scale.iteritems(), key=operator.itemgetter(1))[0]

    # return the majority vote prediction of the last k predictions and its proportion of weight
    def getMajorityPredictionAndWeight(self):
        if len(self.lastPredictions) == 0:
            return None
        # calculate weighted totals and return max weight
        scale = {}
        for i in range(0, len(self.lastPredictions)):
            predi = self.lastPredictions[i]
            weighti = self.lastWeights[i]
            if predi in scale:
                scale[predi] += weighti
            else:
                scale[predi] = weighti

        totalweight = float(sum(scale.values()))
        majorityprediction = max(scale.iteritems(), key=operator.itemgetter(1))[0]
        return majorityprediction, scale[majorityprediction]/totalweight

            

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
        json.dump(filedata,outfile, indent = 4, sort_keys = True)

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
    if useVGG == 1:
        vggrecognizer = VGGRecognizer(knn=3)
    recognizer = OpenfaceRecognizer(knn=3)
    print '\tTime : {0:.2f} s'.format(time.time() - now)

    print 'Retrieving training images.'
    now = time.time()
    images, labels = get_images_and_labels(args['traindir'], 
        training=True, grayscale=False)
    #images, labels = infer_images_and_labels(args['traindir'],grayscale=False)
    print '\tTime : {0:.2f} s'.format(time.time() - now)

    print 'Training the recognizer.'
    now = time.time()
    ylabels = recognizer.train(images, labels)
    if useVGG == 1:
        ylabels_vgg = vggrecognizer.train(images, labels)
    print '\tTime : {0:.2f} s'.format(time.time() - now)
    print 'People in the training set:',
    people = np.unique(ylabels)
    for i in range(len(people)):
        if (i % 4 == 0):
            print '\n\t {} : {}, '.format(i, people[i]),
        else:
            print '{}, '.format(people[i]),
    print ''
    # set up webcam
    fwidth = 640
    fheight = 480

    cap = initializeWebcam(cam=0,width=fwidth,height=fheight)
    # for tracking fps
    prevtime = None
    currtime = None

    # for tracking faces
    trackedfaces = []
    lastPredictionTime = None


    print 'Starting the capture.'
    timelimit_start = time.time()
    # Main running loop
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # create a copy for display and writing on
        cpframe = np.copy(frame)
        dlibfaces = dlib_detector(frame)
        faces = []
        for df in dlibfaces:
            # extract the x, y coordinates and width/height, 
            # then increase box size by 1.5 times in all directions
            centerx = (df.right() - df.left())/2 + df.left()
            centery = (df.bottom() - df.top())/2 + df.top()
            scalefactor = 2.
            x = max(0, int(centerx - (centerx - df.left())*scalefactor))
            y = max(0, int(centery - (centery - df.top())*scalefactor))
            w = min(fwidth - x, int((centerx - df.left())*scalefactor*2))
            h = min(fheight - y, int((centery - df.top())*scalefactor*2))
            faces.append((x, y, w, h))

        # prediction all faces in the frame
        for (x,y,w,h) in faces:
            # look for a matching face
            matchedface = None
            for oldface in trackedfaces:
                if oldface.checkSimilar(x, y):
                    matchedface = oldface

            # remove faces that haven't appeared in the last 5 frames
            for oldface in trackedfaces:
                if oldface.lastUpdated > 5:
                    trackedfaces.remove(oldface)

            # if no matching face found, then create a new one
            if matchedface is None:
                matchedface = TrackFace(x, y)
                trackedfaces.append(matchedface)
                # display rectangle and text over face, in green
                cv2.rectangle(cpframe,(x,y),(x+w,y+h),(0,255,0),2)
            else:
                # display rectangle and text over face, in blue
                cv2.rectangle(cpframe,(x,y),(x+w,y+h),(255,0,0),2)

            # run the prediction and update the matchedface accordingly
            roi = frame[y:y+h, x:x+w]

            # run prediction over the ROI
            prediction = recognizer.verbose_predict(roi)

            # if we have a valid prediction
            if prediction is not None:
                dist, ind = prediction
                print dist,
                print ind,

                # only update prediction if the two closest neighbors agree
                if ylabels[ind[0,0]] == ylabels[ind[0, 1]]:
                    name = ylabels[ind[0,0]]
                    # use 1/dist as a confidence weight
                    weight = 1./(max((dist[0,0] - 0.2)**2., 1e-10))
                    print name,
                    print weight

                    majpred = matchedface.updatePrediction(name, weight=weight)
                else:
                    print ""

                majpred = matchedface.getMajorityPrediction()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cpframe, majpred, (x, y),font, 0.8, (255,255,255))
        

                # when we have enough predictions for this face...
                if len(matchedface.lastPredictions) >= matchedface.num_majority_vote:
                    prediction, weight = matchedface.getMajorityPredictionAndWeight()


                    # perform VGG prediction, if this face has been seen a certain number of times
                    # and if a certain period of time has passed since the previous VGG prediction
                    if useVGG == 1  and (matchedface.lastVGGTime is None or time.time() - matchedface.lastVGGTime > matchedface.VGGperiod):
                        matchedface.lastVGGTime = time.time()
                        vggprediction = vggrecognizer.verbose_predict(roi)
                        # verify the VGG prediction with the majority prediction, and only update
                        # if they agree with each other
                        if vggprediction is not None:
                            dist, ind = prediction
                            # print 'VGG ',
                            # print dist,
                            # print ind,
                            name = ylabels_vgg[ind[0,0]]
                            # print name,
                            # print matchedface.VGGweight
                            if name == prediction and weight > WEIGHT_CONFIDENCE:
                                    updateJSON(prediction, time.time())

                    else: # VGG is not being used for verification
                        if weight > WEIGHT_CONFIDENCE and \
                            (matchedface.lastJSONsave is None or \
                            time.time() - matchedface.lastJSONsave > matchedface.JSONsavePeriod):
                            updateJSON(prediction, time.time())
                            matchedface.lastJSONsave = time.time()
                            # print 'update successful'

                

        # display the FPS and seconds per frame in the top left corner
        prevtime = currtime
        currtime = time.time()
        if currtime and prevtime:
            timeperframe = currtime - prevtime
            fps = "fps : {0}".format(1/timeperframe)
            #tpf = "tpf : {0}".format(timeperframe)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cpframe, fps, (20, 20), font, 0.5, (255,0,0))


        # Display the resulting frame
        cv2.imshow('frame',cpframe)
        # quit if the user presses q on the screen, or the elapsed amount of time has passed
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() - timelimit_start > timelimit):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return True

# the main loop
# it runs the sub_routine loops infinitely, for the specified amount of time
def main():
    looping = True
    while looping:
        print 'starting new subroutine'
        inroutine = True
        now = datetime.datetime.now()
        seconds_until_midnight = (now.replace(hour=23, minute=59, second=59) - now).total_seconds()
        print seconds_until_midnight
        looping = sub_routine(timelimit=15)


if __name__ == '__main__':
    main()