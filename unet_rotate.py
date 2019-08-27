from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import pudb

import torch.nn as nn
import torch.functional as F
import torch
from datagen import *

class unet_torch(nn.Module):
    def __init__(self):
        super(unet_torch, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding = (0, 1, 1))
        self.conv2 = nn.Conv2d(64, 128, 3, padding = (0, 1, 1))
        self.conv3 = nn.Conv2d(128, 256, 3, padding = (0, 1, 1))
        self.conv4 = nn.Conv2d(256, 512, 3, padding = (0, 1, 1))
        self.conv5 = nn.Conv2d(512, 1024, 3, padding = (0, 1, 1))

        self.fc = nn.Linear(256 * 256 * 3, 200 * 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))

        x = self.fc(x)
        x = x.view(-1, 4, 200)

        R = torch.from_numpy(self.create_rot_matrix(1, 4))

        x = torch.matmul(R, x, requires_grad = True)

    def abs_angle(self, pos):
        if (pos == 0):
            return 0.
        elif (pos == 1):
            return np.pi
        elif (pos == 2):
            return np.pi / 2
        elif (pos == 3):
            return np.pi / 4
        elif (pos == 4):
            return 3 * np.pi / 4
        elif (pos == 5):
            return 3 * np.pi / 2
        elif (pos == 6):
            return 7 * np.pi / 4
        elif (pos == 7):
            return 5 * np.pi / 4

    def get_angle_diff(self, init_pos, final_pos):
        # pu.db
        ang = - (abs_angle(int(final_pos)) - abs_angle(int(init_pos)))
        # if (ang < 0):
        #     ang = 2 * np.pi - ang
        return ang

    def create_rot_matrix(self, init_pos, final_pos):
        ang = get_angle_diff(init_pos, final_pos)
        R = np.array([[np.cos(ang), -np.sin(ang), 0, 0],\
                      [np.sin(ang),  np.cos(ang), 0, 0],\
                      [0, 0, 1, 0],\
                      [0, 0, 0, 1]])

        return R

f = open("../VehicleReIDKeyPointData/keypoint_train.txt", "r")
files = []
while True:
    l = f.readline()
    if not l:
        f.close()
        break
    files.append(l.strip().split(" "))
f.close()

data = data_unet(files)

for each in data:
    pu.db

net = unet_torch()
pu.db



        
# def unet(pretrained_weights = None,input_size = (256,256,1)):
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

#     conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
#     flat = Flatten()(conv5)
#     centre_flat = Dense(200 * 4)(flat)
#     centre_flat_reshaped = Reshape((4, 200))(centre_flat)

#     R = Input(tensor = K.variable(create_rot_matrix(0, 1)))
#     centre_flat_rotated = K.dot(R, centre_flat_reshaped)

#     out = Softmax()(centre_flat_rotated)


#     # drop5 = Dropout(0.5)(conv5)

#     # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#     # merge6 = concatenate([drop4,up6], axis = 3)
#     # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#     # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

#     # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#     # merge7 = concatenate([conv3,up7], axis = 3)
#     # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#     # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

#     # up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
#     # merge8 = concatenate([conv2,up8], axis = 3)
#     # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#     # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

#     # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#     # merge9 = concatenate([conv1,up9], axis = 3)
#     # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
#     # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#     # conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

#     model = Model(input = inputs, output = out)

#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
#     return model

model = unet()
pu.db