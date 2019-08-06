from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.models import *

from keras.utils import Sequence, to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import random
from copy import copy
import pudb
from datagen import *
from utils import *

def common_model(in_dim):
    input = Input(shape=in_dim)
    x = Dense(512, activation='relu')(input)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(input, x)

f = open("../VehicleReIDKeyPointData/keypoint_train.txt", "r")
files = {}
count = 0
miscount = 0
beyond_one = 0
while True:
    count += 1
    print(count)
    l = f.readline()
    if not l:
        f.close()
        break
    naam = l.split(" ")[0] 
    files[naam]=[float(int(x)) for x in l.split(" ")[1:]]
    try:
        img_here = image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+str(naam.split("/")[-1])))
    except:
        print("File not found!!")
        miscount+=1
        continue

    width = img_here.shape[1]
    height = img_here.shape[0]

    for i in xrange(len(files[naam]) - 1):
        if i % 2 == 0:
            files[naam][i]/=width
        else:
            files[naam][i]/=height
        if files[naam][i] > 1: beyond_one+=1
    files[naam][-1] = to_categorical(files[naam][-1], num_classes = 8)

all_lifters = []

for i in xrange(20):
    all_lifters.append(common_model((10,)))

for _ in xrange(EPOCHS):
    az_ang = random.uniform(-np.pi, np.pi)
    ev_ang = random.uniform(-np.pi/9, np.pi/9)
    rot_ang = random.uniform(0, 2*np.pi)
    x = np.cos(ev_ang)*np.cos(az_ang)
    y = np.cos(ev_ang)*np.sin(az_ang)
    z = np.sin(ev_ang)
    C = np.array([[0, -z, y],[z, 0, -x],[-y, x, 0]])

    R = np.identity(3) + np.sin(rot_ang) * C + (1 - np.cos(rot_ang)) * np.matmul(C, C)
    T = np.uniform(0, 1)
    
    for model in all_lifters:
        for naam in files:
            for i in xrange(20):
                X = files[naam][2*i:2*i+2]
                X.extend(files[naam][-8:])
                X.extend([0.,0.,0.])
                z = all_lifters[i].predict(X)

                x_ = X[0] * z
                y_ = X[1] * z
                z_ = z

                x_new, y_new, z_new = np.matmul(R, [[x_], [y_], [z_]])
                x_new, y_new, z_new = x_new[0], y_new[0], z_new[0]
                z_new += T

                x_new_2d = x_new / z_new
                y_new_2d = y_new / z_new

                X_new_2d = [x_new_2d, y_new_2d]
                X_new_2d.extend(files[naam][-8:])
                x_new_2d.extend([az_ang, ev_ang, rot_ang])

                all_lifters[i].fit([X_new_2d],[(X[0], X[1])])



pu.db



