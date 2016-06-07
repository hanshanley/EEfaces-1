'''
Author: Kevin Wang
Last updated: 5/4/16
Used with Python 2.7

Contains common functions that are used in other files, i.e. for loading images and labels, 
normalizing brightness, and generating numerical labels from string labels
'''

import os, csv, sys, cv2
import numpy as np
from PIL import Image

# given a directory to search, return the images and labels as lists. training is a boolean to specify
# whether to get the images from the train_labels.csv (True) or test_labels.csv (False). 
# Grayscale is a parameter to specify whether to return 1 channel images or 3 channel BGR images
def get_images_and_labels(dire, training, grayscale=True):
    # check that dire exists and contains the labels.csv file
    if not os.path.isdir(dire):
        print dire + " is not found in current directory"
        sys.exit()
    if training:
        if not os.path.isfile(dire + '/train_labels.csv'):
            print "train_labels.csv is missing from " + dire,
            print "Attempting to infer directory contents"
            return infer_images_and_labels(dire, grayscale)
        else:
            csvfilename = '/train_labels.csv'
    else:
        if not os.path.isfile(dire + '/test_labels.csv'):
            print "test_labels.csv is missing from " + dire,
            print "Attempting to infer directory contents"
            return infer_images_and_labels(dire, grayscale)
        else:
            csvfilename = '/test_labels.csv'

    # load contents of the csv file
    column = {}
    with open(dire + csvfilename, 'rb') as csvfile:
        facereader = csv.reader(csvfile, delimiter=',')
        headers = facereader.next()
        for h in headers:
            column[h] = []
        for row in facereader:
            for h, v in zip(headers, row):
                column[h].append(v)

    images = [] # contains the face images 
    labels = [] # contains the string labels
    for i in range(len(column['Filename'])):
        filename = column['Filename'][i]
        label = column['Name'][i]
        image_path = dire + '/' + filename
        if grayscale: # read in as grayscale
            image_pil = Image.open(image_path).convert('L')
        else: # otherwise read in as BGR
            image_pil = cv2.imread(image_path)
        img = np.array(image_pil, 'float32') # dtype care
        if training:
            caption = 'Adding faces to training set'
        else:
            caption = 'Adding faces to testing set'
        
        #cv2.imshow(caption, img.astype(np.uint8))
        #cv2.waitKey(10)

        images.append(img.astype(np.uint8))
        labels.append(label)
        #cv2.destroyAllWindows()

    
    return images, labels

# given a directory to search, return the images and labels as lists.
# It matches with all files that are one of the listed extensions, and parses their filenames
# in the following format: [number/identifier]_[label].extension 
# Grayscale is a parameter to specify whether to return 1 channel images or 3 channel BGR images
def infer_images_and_labels(dire, grayscale=True):
    included_extensions = ['jpg', 'bmp', 'png', 'gif', 'ppm', 'PNG', 'JPG']
    file_names = []
    for fn in os.listdir(dire):
        for ext in included_extensions:
            if fn.endswith(ext):
                file_names.append(fn)
                break

    images = []
    labels = []
    for fn in file_names:
        try:
            filename = fn
            # extract label from: [number/identifier]_[label].extension 
            # label = (' '.join(fn.split('_')[1:])).split('.')[0]
            label = (' '.join(fn[:-4].split('_')[1:]))
            image_path = dire + '/' + filename
            if grayscale: # read in as grayscale
                image_pil = Image.open(image_path).convert('L')
            else: # otherwise read in as BGR
                image_pil = cv2.imread(image_path)
                image_pil = cv2.cvtColor(image_pil, cv2.COLOR_BGR2RGB) # convert to RGB
            img = np.array(image_pil, 'float32') # dtype care
            caption = 'Retrieving faces'
            
            #cv2.imshow(caption, img.astype(np.uint8))
            #cv2.waitKey(10)

            images.append(img.astype(np.uint8))
            labels.append(label)
            #cv2.destroyAllWindows()
        except: 
            continue

    return images, labels


# same as get images and labels, but returns extra information from the feret db filenames, 
# returns images, labels, orientations (str), sessions (str)
# Grayscale is a parameter to specify whether to return 1 channel images or 3 channel BGR images
def get_images_and_labels_feret(dire, training, grayscale=False):
    # check that dire exists and contains the labels.csv file
    if not os.path.isdir(dire):
        print dire + " is not found in current directory"
        sys.exit()
    if training:
        if not os.path.isfile(dire + '/train_labels.csv'):
            print "train_labels.csv is missing from " + dire
            sys.exit()
        csvfilename = '/train_labels.csv'
    else:
        if not os.path.isfile(dire + '/test_labels.csv'):
            print "test_labels.csv is missing from " + dire
            sys.exit()
        csvfilename = '/test_labels.csv'

    # load contents of the csv file
    column = {}
    with open(dire + csvfilename, 'rb') as csvfile:
        facereader = csv.reader(csvfile, delimiter=',')
        headers = facereader.next()
        for h in headers:
            column[h] = []
        for row in facereader:
            for h, v in zip(headers, row):
                column[h].append(v)

    images = [] # contains the face images 
    labels = [] # contains the string labels
    for i in range(len(column['Filename'])):
        filename = column['Filename'][i]
        label = column['Name'][i]
        image_path = dire + '/' + filename
        if grayscale: # read in as grayscale
            image_pil = Image.open(image_path).convert('L')
        else: # otherwise read in as BGR
            image_pil = cv2.imread(image_path)
        img = np.array(image_pil, 'float32') # dtype care
        if training:
            caption = 'Adding faces to training set'
        else:
            caption = 'Adding faces to testing set'
        
        #cv2.imshow(caption, img.astype(np.uint8))
        #cv2.waitKey(10)

        images.append(img.astype(np.uint8))
        labels.append(label)
        #cv2.destroyAllWindows()

    # find the orientation and session of the photo
    orientations = []
    sessions = []
    for file in column['Filename']:
        if 'fa' in file or 'fb' in file:
            orientations.append('f')
        elif 'ql' in file or 'qr' in file:
            orientations.append('q')
        elif 'rb' in file or 'rc' in file:
            orientations.append('r')
        else:
            orientations.append('other')
        if training:
            sessions.append(file.split('_')[2])
        else:
            sessions.append(file.split('_')[1])

    return images, labels, orientations, sessions


# converts the labels to a numerical labels, including a labelDict that maps the original 
# labels to the numbers. it will update a given labelDict if that is passed in with new labels,
# or it will return the given labelDict unchanged 
def convertToNumericalLabels(labels, labelDict=None):
    if labelDict == None:
        labelDict = {}
    j = len(labelDict.keys())
    numberlabels = []
    for i in range(len(labels)):
        if labels[i] not in labelDict.keys():
            labelDict[labels[i]] = j
            j += 1
        numberlabels.append(labelDict[labels[i]])
    return numberlabels, labelDict

# takes in an RGB 3 channel image, converts it to xyz then lms representation
# normalize in LMS space using whitepixel if given, or the 1,1 pixel location otherwise
# and returns the converted back to RGB space normalized image
def normalizeLMSwhite(image, whitepixel = None):
    conv1 = np.array([
                        [0.5767309, 0.1855540, 0.1881852],
                        [0.2973769, 0.623491, 0.0752741],
                        [0.0270343, 0.0706872, 0.9911085]
                    ])
    conv2 = np.array([
                        [0.4002, 0.7076, -0.0808],
                        [-.2263, 1.1653, 0.0457],
                        [0, 0, 0.9182]
                    ])
    convback2 = np.linalg.inv(conv2)
    convback1 = np.linalg.inv(conv1)
    #print convback2
    #print convback1
    outputimg = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            tmp = np.dot(conv1, image[i, j].T)
            tmp2 = np.dot(conv2, tmp)
            outputimg[i, j] = tmp2

    if not whitepixel:
        divideby = np.copy(outputimg[1,1])
    else:
        divideby = np.copy(outputimg[whitepixel[0],whitepixel[1]])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            normalPixel = outputimg[i, j]
            normalPixel = np.zeros(3)
            normalPixel[0] = outputimg[i,j][0]/divideby[0]
            normalPixel[1] = outputimg[i,j][1]/divideby[1]
            normalPixel[2] = outputimg[i,j][2]/divideby[2]
            tmp = np.dot(convback2, normalPixel)
            tmp2 = np.dot(convback1, tmp)
            for k in range(len(tmp2)):
                if tmp2[k] < 0:
                    tmp2[k] = 0
                elif tmp2[k] > 1:
                    tmp2[k] = 1
            outputimg[i,j] = tmp2 * 255.



    return outputimg

# testing various functions
if __name__ == "__main__":
    image_pil = cv2.imread('./raw_collectedfaces/raw_train/raw_Catherine_Hua.png')
    image_pil = cv2.imread('./raw_collectedfaces/raw_train/raw_Kevin_Wang.png')
    # image_pil = cv2.imread('./raw_collectedfaces/raw_train/raw_Mark_Petersen.png')
    img = np.array(image_pil, 'float32')
    print img.max()
    print img.min()
    cv2.imshow('original img', img.astype(np.uint8))
    gray1 = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cv2.imshow('original grayscale', gray1)
    # cv2.waitKey(0)

    normalized = normalizeLMSwhite(img, (100, 5))
    print normalized.max()
    print normalized.min()
    cv2.imshow('normalized img', normalized.astype(np.uint8))


    gray2 = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cv2.imshow('normalized grayscale', gray2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


