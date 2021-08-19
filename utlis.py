import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.image as npimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam


def getname(filepath):
    return filepath.split('\\')[-1]



def importDatainfo(path):
    columns=['center','left','right','steering','throttle','brake','speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    print(data.head())
    # # print(data['center'][0])
    # # print(getname(data['center'][0]))
    data['center']=data['center'].apply(getname)
    # #
    # # print((data.shape[0]))
    return data

# Visualize Distribution of Data
def balanceData(data,display=True):
    nBin = 13 #It is odd because 0 should be in middle of positive and negative values
    samplesPerBin = 300
    hist, bins = np.histogram(data['steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
        hist, _ = np.histogram(data['steering'], (nBin))
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['steering']), np.max(data['steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
    return data





#
#
def loaddata(path,data):
    imagespath=[]
    steering=[]

    for i in range(len(data)):
        indexdata=data.iloc[i]
        # print(indexdata)
        imagespath.append(os.path.join(path,'IMG',indexdata[0]))
        # print(os.path.join(path,'IMG',indexdata[0]))
        steering.append(float(indexdata[3]))
        # print(float(indexdata[3]))
    imagespath=np.asarray(imagespath)
    steering=np.asarray(steering)
    return imagespath,steering

def augmentimage(imgpath,steering):
    img = npimg.imread(imgpath)
    ##PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
        # RES,st=augmentimage('TEST.jpg')
        # plt.imshow(RES)
        # plt.show()
    ##ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    ##BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    ##STEERING
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def preprocess(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2Luv)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    return img



imgre=preprocess(mpimg.imread('TEST.jpg'))
plt.imshow(imgre)
plt.show()


def batchgen(imagepath,steeringlist,batchsize,trainflag):
    while True:
        imgbatch=[]
        steeringbatch=[]
        for i in range(batchsize):
            index=random.randint(0,len(imagepath)-1)
            if trainflag: #validation set we want it to not augment it
                img,steering=augmentimage(imagepath[index],steeringlist[index])
            else:
                img=mpimg.imread(imagepath[index])
                steering=steeringlist[index]
            img=preprocess(img)
            imgbatch.append(img)
            steeringbatch.append(steering)
        yield (np.asarray(imgbatch),np.asarray(steeringbatch))


def createmodel():
    model=Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model

#
#




#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#








