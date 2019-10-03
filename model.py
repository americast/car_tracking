import torch.nn as nn
import torch.nn.functional as F
import torch

class unet_torch(nn.Module):
    def __init__(self):
        super(unet_torch, self).__init__()
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
        self.fc =  nn.Linear(16384, 16384)

    def forward(self, x, R):
        x = self.encoder(x)
        # pu.db
        # print(x.shape)
        org_shape = x.shape
        # x = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], 4, -1)
        x = torch.bmm(R, x)
        # print("x" + str(x.shape))
        # print("R" + str(R.shape))
        # x = self.fc(x)
        x = x.view(org_shape[0], org_shape[1], org_shape[2], org_shape[3])
        # print(x.shape)
        # sys.exit(0)
        x = self.decoder(x)
        return x


    # def forward(self, x, R):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = F.relu(self.conv3(x))
    #     # x = self.pool(F.relu(self.conv4(x)))
    #     # x = F.relu(self.conv5(x))

    #     # x = x.view(-1, 16 * 16 * 256)
    #     # x = self.fc_1(x)
    #     # x = x.view(-1, 4, 200)

    #     # improve this
    #     # for i in range(x.shape[0]):
    #     #     R.append(self.create_rot_matrix(view_1[i], view_2[i]))
    #     # R = torch.from_numpy(np.array(R))
    #     # x = torch.matmul(R, x)
    #     # x = x.view(-1, 200 * 4)
    #     # x = self.fc_2(x)
    #     # x = x.view(-1, 256, 16, 16)

    #     # x = F.relu(self.conv5_up(self.upsample_5(x)))
    #     # x = F.relu(self.conv4_up(self.upsample_4(x)))
    #     x = F.relu(self.conv3_up(self.upsample_3(x)))
    #     x = F.relu(self.conv2_up(self.upsample_2(x)))
    #     x = F.relu(self.conv1_up(x))

    #     return x

