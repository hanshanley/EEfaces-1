'''
Author: Kevin Wang
Last updated: 6/1/16    by Sanket Satpathy
Used with Python 2.7

Description:
A general purpose face recognizer using the VGG net for feature extraction. Requires 
Lasagne on the computer for running the neural net, as well as other requirements. 
Supports training on a set of single sample images, which are projected into a 
128-dimension feature space, and efficient nearest neighbor search in that feature 
space for prediction of new faces. 
'''
import time
import argparse
import cv2
import os
import numpy as np
np.set_printoptions(precision=2)
import sys
sys.path.insert(0, './vgg_feature')
import vgg_feature 
from sklearn.neighbors import NearestNeighbors

# flag for saving numpy array (1), or loading an old one (0)
saveon = 1

class VGGRecognizer:
    # maps point to label in NN implementation
    ylabels = None
    # if the recognizer has been trained before
    initialized = False
    # reference to the VGG neural net
    net = None
    # nearest neighbors sklearn object
    nbrs = None
    # number of nearest neighbors to use
    # knn = 1

    # constructor -- use the default parameters for the best classifier 
    # dire contains the location of the pickled weights
    def __init__(self, dire='./vgg_feature/', knn=1):
        self.initialized = False
        self.net = vgg_feature.load_weights(dire=dire)
        self.knn = knn

    # train the recognizer on the given set of images, with corresponding labels. 
    # return a copy of the labels
    def train(self, images, labels):
        # Mark that we have trained now 
        if not self.initialized:
            self.initialized = True
        else:
            print 'Warning: Training again will override previous faces'

        self.ylabels = []
        tr_features = None
        if saveon == 1:
            for i,l in zip(images,labels):
                print 'VGG: ', l
                if len(i.shape) == 2:
                    i = i[:, :, np.newaxis]
                    rgbi = np.repeat(i, 3, axis=2)
                else:
                    rgbi = i
                if rgbi is not None:
                    feature = vgg_feature.get_feature(rgbi, self.net)
                    if tr_features is None:
                        tr_features = feature
                    else:
                        tr_features = np.vstack((tr_features, feature))
                    self.ylabels.append(l)
                    # include the LR flip in training
                    flip = vgg_feature.get_feature(np.fliplr(rgbi), self.net)
                    if flip is not None:
                        tr_features = np.vstack((tr_features, flip))
                        self.ylabels.append(l)
            np.save('vgg_ylabels.npy', self.ylabels)
            np.save('vgg_features.npy', tr_features)
        else:
            self.ylabels = np.load('vgg_ylabels.npy')
            tr_features = np.load('vgg_features.npy')

        self.nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='ball_tree').fit(tr_features)

        return np.copy(self.ylabels)

    # predict the class of the input image. returns None if it runs into errors
    # during the feature extraction, otherwise returns the label of the nearest point
    def predict(self, image):
        if not self.initialized:
            print 'Train before predicting!'
            return

        pred_feature = vgg_feature.get_feature(image, self.net)
        dist, ind = self.nbrs.kneighbors(pred_feature)
        return self.ylabels[ind[0,0]]

    # verbose prediction of the class of the input image. returns None if it runs 
    # into errors during the embedding, otherwise returns the distances 
    # and indices of the nearest neighbor search
    def verbose_predict(self, image):
        if not self.initialized:
            print 'Train before predicting!'
            return

        pred_feature = vgg_feature.get_feature(image, self.net)
        dist, ind = self.nbrs.kneighbors(pred_feature)
        return dist, ind

