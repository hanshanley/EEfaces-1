'''
Author: Kevin Wang
Last updated: 6/6/16    by Sanket Satpathy
Used with Python 2.7

Description:
A general purpose face recognizer using the openface 0.2.0 implementation. Requires 
torch on the computer for running the neural net, as well as other requirements 
(see http://cmusatyalab.github.io/openface/). Supports training on a set of single
sample images, which are projected into a 128-dimension feature space, and efficient 
nearest neighbor search in that feature space for prediction of new faces. 
'''
import time
import argparse
import cv2
import os
import numpy as np
np.set_printoptions(precision=2)
import openface
from sklearn.neighbors import NearestNeighbors
from dlib import rectangle

# flag for saving npy array (1), or loading an old one (0)
saveon = 1
include_flip = True
include_truncation = True
include_rotation = True
include_blur = True

def rotate(img, angle):
    h, w = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

class OpenfaceRecognizer:
    # maps point to label in NN implementation
    ylabels = None
    # if the recognizer has been trained before
    initialized = False
    # openface object for aligning the faces
    align = None
    # reference to the neural net
    net = None
    # width/height to input to the neural net
    imgDim = 96
    # sklearn nearest neighbors object
    nbrs = None

    # need to specify where to find the dlib model and NN
    fileDir = os.path.dirname(os.path.realpath(__file__))
    modelDir = os.path.join(fileDir, 'openface', 'models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')

    # constructor -- use the default parameters for the best classifier 
    def __init__(self, dlibFacePredictor=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"), 
                 networkModel=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), knn=1):
        self.align = openface.AlignDlib(dlibFacePredictor)
        self.net = openface.TorchNeuralNet(networkModel, self.imgDim)
        self.initialized = False
        self.knn = knn

    # train the recognizer on the given set of images, with corresponding labels
    def train(self, images, labels):
        # Mark that we have trained now 
        if not self.initialized:
            self.initialized = True
        else:
            print 'Warning: Training again will override previous faces'

        self.ylabels = []
        if saveon == 1:
            features = np.array([]).reshape(0, 128)
            for i,l in zip(images,labels):
                print 'OpenFace: ', l
                rep = self.getRep(i)
                if rep is not None:
                    features = np.vstack((features, rep.reshape(1, -1)))
                    self.ylabels.append(l)
                    # also use the left-right flipped images
                    if include_flip:
                        feature = self.getRep(np.fliplr(i))
                        features = np.vstack((features, feature.reshape(1, -1)))
                        self.ylabels.append(l)
                    if include_truncation:
                        for j in xrange(1,6):
                            feature = self.getRep(i[j * i.shape[0]/10:,:,:])  # remove upper face
                            features = np.vstack((features, feature.reshape(1, -1)))
                            self.ylabels.append(l)
                            feature = self.getRep(i[:-j * i.shape[0]/10,:,:])  # remove lower face
                            features = np.vstack((features, feature.reshape(1, -1)))
                            self.ylabels.append(l)
                    if include_rotation:
                        for angle in xrange(-30, 35, 5):
                            feature = self.getRep(rotate(i, angle))  # rotate face
                            features = np.vstack((features, feature.reshape(1, -1)))
                            self.ylabels.append(l)
                    if include_blur:
                        for blur_radius in [3, 10]:
                            feature = vgg_feature.get_feature(cv2.blur(rgbi, (blur_radius, blur_radius), 0), self.net)  # blur face
                            tr_features = np.vstack((tr_features, feature))
                            self.ylabels.append(l)
            np.save('of_ylabels.npy', self.ylabels)
            np.save('of_features.npy', features)
        else:
            self.ylabels = np.load('of_ylabels.npy')
            features = np.load('of_features.npy')

        self.nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='ball_tree').fit(X=features)
        return np.copy(self.ylabels)


    # predict the class of the input image. returns None if it runs into errors
    # during the embedding, otherwise returns the label of the nearest point
    def predict(self, image):
        if not self.initialized:
            print 'Train before predicting!'
            return

        predrep = self.getRep(image)
        if predrep is None:
            return None
        dist, ind = self.nbrs.kneighbors(predrep.reshape(1, -1))
        return self.ylabels[ind[0,0]]

    # verbose prediction of the class of the input image. returns None if it runs 
    # into errors during the embedding, otherwise returns the distances 
    # and indices of the nearest neighbor search
    def verbose_predict(self, image):
        if not self.initialized:
            print 'Train before predicting!'
            return

        # now = time.time()
        predrep = self.getRep(image)
        # print '\tComputing OpenFace feature took {0:.2f} seconds'.format(time.time() - now)
        if predrep is None:
            print 'No feature'
            return None
        dist, ind = self.nbrs.kneighbors(predrep.reshape(1, -1))
        return dist, ind

    # returns the 128 dimension vector of the neural net feature embedding
    # or returns None if it runs into any errors during the embedding
    # use parameter rgb as False if the image is bgr.
    def getRep(self, img, rgb=True):
        if len(img.shape) < 3: # if grayscale, convert to 3 channel image
            rgbImg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        else: # otherwise convert to rgb 
            if not rgb:
                rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgbImg = img
        if rgbImg is None:
            print 'No image'
            return None

        # get bounding box of the face in the image
        bb = self.align.getLargestFaceBoundingBox(rgbImg)
        if bb is None:
            # print 'WARNING: Could not find largest bounding box, using ROI as is'
            bb = rectangle(0, 0, rgbImg.shape[1], rgbImg.shape[0])
            # return None
            
        # align the face within the bounding box
        alignedFace = self.align.align(self.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            print 'No aligned face'
            return None
            
        rep = self.net.forward(alignedFace)
        return rep
