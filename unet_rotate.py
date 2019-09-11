# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as K
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from datagen import *
import matplotlib.pyplot as plt
import pudb
import sys
import numpy as np

BATCH_SIZE = 200
EPOCHS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class unet_torch(nn.Module):
    def __init__(self):
        super(unet_torch, self).__init__()
        self.pool = nn.MaxPool2d(4, 4)
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding = 1)
        # self.conv4 = nn.Conv2d(256, 512, 3, padding = 1)
        # self.conv5 = nn.Conv2d(512, 1024, 3, padding = 1)

        self.fc_1 = nn.Linear(16 * 16 * 256, 200 * 4)
        self.fc_2 = nn.Linear(200 * 4, 16 * 16 * 256)

        # self.upsample_5 = nn.Upsample(scale_factor = 2.0)
        # self.conv5_up = nn.Conv2d(1024, 512, 3, padding = 1)
        # self.upsample_4 = nn.Upsample(scale_factor = 2.0)
        # self.conv4_up = nn.Conv2d(512, 256, 3, padding = 1)
        self.upsample_3 = nn.Upsample(scale_factor = 4.0)
        self.conv3_up = nn.Conv2d(256, 128, 3, padding = 1)
        self.upsample_2 = nn.Upsample(scale_factor = 4.0)
        self.conv2_up = nn.Conv2d(128, 64, 3, padding = 1)
        self.conv1_up = nn.Conv2d(64, 3, 3, padding = 1)


    def forward(self, x, R):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = F.relu(self.conv5(x))

        x = x.view(-1, 16 * 16 * 256)
        x = self.fc_1(x)
        x = x.view(-1, 4, 200)

        # improve this
        # for i in range(x.shape[0]):
        #     R.append(self.create_rot_matrix(view_1[i], view_2[i]))
        # R = torch.from_numpy(np.array(R))
        x = torch.matmul(R, x)
        x = x.view(-1, 200 * 4)
        x = self.fc_2(x)
        x = x.view(-1, 256, 16, 16)

        # x = F.relu(self.conv5_up(self.upsample_5(x)))
        # x = F.relu(self.conv4_up(self.upsample_4(x)))
        x = F.relu(self.conv3_up(self.upsample_3(x)))
        x = F.relu(self.conv2_up(self.upsample_2(x)))
        x = self.conv1_up(x)

        return x

f = open("../VehicleReIDKeyPointData/keypoint_train_corrected.txt", "r")
files = []
while True:
    l = f.readline()
    if not l:
        f.close()
        break
    files.append(l.strip().split(" "))
f.close()
print("Data loading starts")

# for i_batch, sample_batched in enumerate(dataloader):
#     pu.db
print("Data loading complete")
data = data_unet(files[:100])
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle = False, num_workers = 40, pin_memory=True)
net = unet_torch().to(device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)

criterion = nn.MSELoss()
optimiser = optim.Adam(net.parameters(), lr=0.01)
loss_min = np.inf
print("len: "+str(len(data)))
print()
# learning_rate = 0.01
for _ in range(EPOCHS):
    # print("_ is: "+str(_))
    loss_net = []
    print(str(_+1)+"/"+str(EPOCHS))
    i = 0
    for i, each in enumerate(dataloader):
        # print("len: "+str(len(data)))
        # i+=1
        print("%.2f"% ((i + 1) * 100.0 / len(dataloader))+"%", end="")
        for every in each[0]:
            if int(sum(sum(sum(every)))) == 0:
                continue
        # pu.db
        out = net(each[0].float().to(device), each[1].float().to(device))
        # pu.db
        loss = criterion(out, each[2].float().to(device))
        loss_net.append(loss)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(", loss: %.6f" %(np.sum(loss_net)/len(loss_net)/(256 * 256)), end="\r")
        # if (i == 2): break
        # for f in net.parameters():
        #     f.data.sub_(f.grad.data * learning_rate)


        # pu.db
        # plt.imshow(each[0][0])
        # plt.show()
    print()
    loss_now = np.sum(loss_net)/len(loss_net)/(256 * 256)
    # print("Final loss: "+str(loss_now))
    if (loss_now < loss_min):
        loss_min = loss_now
        torch.save(net.state_dict(), "./rotate_model.pth")
        print("Model saved")
    # del dataloader



        
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

# model = unet()
# pu.db