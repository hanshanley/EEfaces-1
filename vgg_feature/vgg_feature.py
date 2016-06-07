'''
Author: Sai Satpathy and Kevin Wang
Last updated: 5/2/16
Used with Python 2.7

Description:
Methods for constructing and using the VGG net as a facial feature extractor, 
with usage examples in main()
'''
import lasagne
from PIL import Image
from PIL import ImageFilter
import numpy as np
import pickle
import skimage.transform
import scipy
from sklearn.neighbors import NearestNeighbors

import cv2

import theano
import theano.tensor as T

from lasagne.utils import floatX

from lasagne.layers import InputLayer, DropoutLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import rectify

IMAGE_W = 224

# average pixel values (RGB) in VGG training set
MEAN_VALUES = np.array([129.1863,104.7624,93.5940]).reshape((3,1,1))	

LAYERS = ['pool6']  # features for classification


# image preprocessing routine, takes in a grayscale or rgb im 
def prep_image(im):
    if im.max() <= 1:		# pixel values must be float32 in [0,255]
        im *= 255.0
    
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)

    # height is span of head
    im = skimage.transform.resize(im, (IMAGE_W, w*IMAGE_W/h), preserve_range=True)
    # make square
    if im.shape[0] > im.shape[1]:
        im = np.hstack((np.tile(MEAN_VALUES, (IMAGE_W, (IMAGE_W-im.shape[1])/2, 1)), im, np.tile(MEAN_VALUES, (IMAGE_W, IMAGE_W - (IMAGE_W-im.shape[1])/2 - img.shape[1], 1))))
    else:
        im = im[:,:IMAGE_W,:]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    im = im - MEAN_VALUES
    im = im[::-1, :, :]

    return rawim, floatX(im[np.newaxis])

# Build the VGGnet
def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, IMAGE_W, IMAGE_W))
    net['conv1_1'] = ConvLayer(net['input'], num_filters=64, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['conv1_2'] = ConvLayer(net['conv1_1'], num_filters=64, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['pool1'] = PoolLayer(net['conv1_2'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['conv2_1'] = ConvLayer(net['pool1'], num_filters=128, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['conv2_2'] = ConvLayer(net['conv2_1'], num_filters=128, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['pool2'] = PoolLayer(net['conv2_2'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['conv3_1'] = ConvLayer(net['pool2'], num_filters=256, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['conv3_2'] = ConvLayer(net['conv3_1'], num_filters=256, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['conv3_3'] = ConvLayer(net['conv3_2'], num_filters=256, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['pool3'] = PoolLayer(net['conv3_3'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['conv4_1'] = ConvLayer(net['pool3'], num_filters=512, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['conv4_2'] = ConvLayer(net['conv4_1'], num_filters=512, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['conv4_3'] = ConvLayer(net['conv4_2'], num_filters=512, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['pool4'] = PoolLayer(net['conv4_3'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['conv5_1'] = ConvLayer(net['pool4'], num_filters=512, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['conv5_2'] = ConvLayer(net['conv5_1'], num_filters=512, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['conv5_3'] = ConvLayer(net['conv5_2'], num_filters=512, filter_size=3, pad=1, flip_filters=False, nonlinearity=rectify)
    net['pool5'] = PoolLayer(net['conv5_3'], pool_size=2, stride=2, mode='max', ignore_border=False)
    net['pool6'] = PoolLayer(net['pool5'], pool_size=5, stride=1, mode='max', ignore_border=False)  # additional max-pooling

    return net

# load weights
def load_weights(dire='./'):
    with open(dire + 'vgg_feature_vals.pickle', 'rb') as handle:
        vals = pickle.load(handle)

    # convert dict to list
    values = []
    for k in sorted(vals.keys()):
        for p in vals[k]:
            values.append(p)
    del vals
    net = build_model()
    lasagne.layers.set_all_param_values(net['pool6'], values)
    return net

# preprocessing the photo, then pass it through the net at the given LAYERS, and return the 
# feature vector
def get_feature(photo, net, layers=LAYERS):
    rawim, photo = prep_image(photo)	# preprocess to 3 x 224 x 224

    # input and output variables
    input_im_theano = T.tensor4()
    layers = {k: net[k] for k in layers}
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

    # get features from desired layers
    photo_features = {k: theano.shared(output.eval({input_im_theano: photo})) for k, output in zip(layers.keys(), outputs)}

    # concatenate them to get one feature vector
    feature = np.array([])
    for layer in layers:
        feature = np.concatenate((feature,photo_features[layer].get_value().flatten()))
    feature = feature.reshape(1,-1)

    return feature

# train using the label/photo list, the given net at the given labels, creating a num_nbrs
# nearest neighbor. dire is the directory containing the labels/photos. Returns the nearest
# neighbors object and the label list
def train_features(training, net, num_nbrs=1, layers=LAYERS, dire='../feret_faces/'):
    label = []
    train_set_features = None

    for l in training:
        name, loc = l
        label.append(name)
        photo = cv2.imread(dire + loc)
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

        feature = get_feature(photo, net, layers=LAYERS)
        
        if train_set_features is None:
            train_set_features = feature
        else:        
            train_set_features = np.vstack((train_set_features, feature))

    # return a nearest neighbors function on training features
    # num_nbrs can be tuned to vary the number of nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='ball_tree').fit(train_set_features)

    return nbrs, label

# run prediction on the photo, based on the feature extracted defined by net and layers, comparing 
# the L2 distance to the nearest neighbors in nbrs
def vgg_predict(photo, nbrs, net, layers=LAYERS):
    feature = get_feature(photo, net, layers=LAYERS)
    distances, indices = nbrs.kneighbors(feature)   # return the nearest neighbors along with L2 distances

    return distances, indices

# example main method
def main():
    # feature extraction
    net = load_weights()
    img = np.array(Image.open('ak.png'), dtype='float32')
    feature = get_feature(img, net=net)

    # one-shot learning
    training = np.recfromcsv('../feret_faces/90train_labels.csv')
    nbrs, label = train_features(training, net)
    
    testing = np.recfromcsv('../feret_faces/90test_labels.csv')
    numtotal = 0.
    numcorrect = 0.
    for l in testing:
        name, loc = l
        img = cv2.imread('../feret_faces/' + loc)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        distances, indices = vgg_predict(img, nbrs, net)
        numtotal += 1.
        print 'The correct answer is '+name
        print 'The nearest-neighbor prediction is ' + label[indices[0,0]]
        if name == label[indices[0,0]]:
            numcorrect += 1.
    print numcorrect/numtotal

    
if __name__ == "__main__":
    main()
