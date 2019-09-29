import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import datagen
import pudb
import sys

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
])

f = open("../VehicleReIDKeyPointData/keypoint_test_corrected.txt", "r")
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
data = datagen.data_unet(files[:1000])
dataloader = DataLoader(data, batch_size=batch_size, shuffle = False, num_workers = 120, pin_memory=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
            nn.Conv2d(32, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # b, 8, 2, 2
            )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor = 2.0),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(True),
            nn.Upsample(scale_factor = 2.0),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(64, affine=False),
            nn.Upsample(scale_factor = 2.0),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(3, affine=False),
            nn.Sigmoid()
            )
        self.fc =  nn.Linear(3528, 3528)

    def forward(self, x):
        x = self.encoder(x)
        # pu.db
        # print(x.shape)
        # x = x.view(x.shape[0], -1)
        # x = self.fc(x)
        # x = x.view(x.shape[0], 8, 21, 21)
        # print(x.shape)
        # sys.exit(0)
        x = self.decoder(x)
        return x

model = autoencoder().cuda()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    print()
    for i, data in enumerate(dataloader):
        # pu.db
        print(str(i + 1) +"/"+ str(len(dataloader)), end = "\r")
        img, _, _2 = data
        img = Variable(img).float().cuda()
        # ===================forward=====================
        # pu.db
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print()
    print('epoch ['+str(epoch + 1)+'/'+str(num_epochs)+'], loss: '+str(loss))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))
        torch.save(model.state_dict(), './conv_autoencoder'+str(epoch)+'.pth')
