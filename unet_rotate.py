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
import matplotlib
from model import *
matplotlib.use('Agg')


BATCH_SIZE = 100
EPOCHS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def out_to_view(out):
    out = out[0]
    out = out.to("cpu")
    out = out.detach().numpy()
    out = np.transpose(out, (1,2,0))
    return out


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
data = data_unet(files[:1000])
# data[400]
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle = True, num_workers = 40, pin_memory=True)
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
        # for every in each[0]:
        #     if int(sum(sum(sum(every)))) == 0:
        #         continue
        # pu.db
        out = net(each[0].float().to(device), each[1].float().to(device))

        # out_here = out.cpu().detach().numpy()
        # org_here = each[0].numpy()

        # out_here = np.reshape(out_here, (out_here.shape[0], -1))
        # org_here = np.reshape(org_here, (org_here.shape[0], -1))

        # loss_avg = []

        # for l in range(out_here.shape[0]):
        #     out_arr = out_here[l, :]
        #     org_arr = org_here[l, :]
        #     loss_here = 0
        #     for m in range(len(out_arr)):
        #         loss_here += (out_arr[m] - org_arr[m])**2
        #     loss_avg.append(loss_here/len(out_arr))
        # loss = sum(loss_avg)/len(loss_avg)
        # pu.db
        # pu.db
        # if _ == 2 and i ==20:
        # pu.db
        loss = criterion(out, each[2].float().to(device))
        loss_net.append(loss / len(each))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(", loss: %.6f" %(np.sum(loss_net)/(len(loss_net))), end="\r")
        # out = out.to("cpu")[0]
        # out = out.detach().numpy()
        # out = np.reshape(out, (3, 256, 256))
        # plt.imshow(np.transpose(out, ((1,2,0))))
        # plt.savefig("res_img/img_"+str(_)+"_"+str(i)+".png")
        # pu.db
        # if (i == 2): break
        # for f in net.parameters():
        #     f.data.sub_(f.grad.data * learning_rate)


        # pu.db
        # plt.imshow(each[0][0])
        # plt.show()
    print()
    loss_now = np.sum(loss_net)/len(loss_net)
    # print("Final loss: "+str(loss_now))
    if (loss_now < loss_min):
        loss_min = loss_now
        torch.save(net.state_dict(), "./rotate_fresh.pth")
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