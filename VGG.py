import torch
import torch.nn.functional as F
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,64,3)

        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,128,3)

        self.conv5 = nn.Conv2d(128,256,1)
        self.conv6 = nn.Conv2d(256,256,1)
        self.conv7 = nn.Conv2d(256,256,3)

        self.conv8 = nn.Conv2d(256,512,1)
        self.conv9 = nn.Conv2d(512,512,1)
        self.conv10 = nn.Conv2d(512,512,3)

        self.conv11 = nn.Conv2d(512,512,1)
        self.conv12 = nn.Conv2d(512,512,1)
        self.conv13 = nn.Conv2d(512,512,3)

        self.batch_size = args.train_batch_size

        self._to_linear = None

        x = torch.randn(3,50,50).view(-1,3,50,50)
        # self.convs(x)
        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def convs(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        return x

    def forward(self,x):
        x = self.convs(x)
        x = torch.flatten(x,1)

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        Output = F.softmax(x)
        return Output

class VGG19(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,64,3)

        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,128,3)

        self.conv5 = nn.Conv2d(128,256,3)
        self.conv6 = nn.Conv2d(256,256,3)
        self.conv7 = nn.Conv2d(256,256,3)
        self.conv8 = nn.Conv2d(256,256,3)

        self.conv9 = nn.Conv2d(256,512,3)
        self.conv10 = nn.Conv2d(512,512,3)
        self.conv11 = nn.Conv2d(512,512,3)
        self.conv12 = nn.Conv2d(512,512,3)

        self.conv13 = nn.Conv2d(512,512,3)
        self.conv14 = nn.Conv2d(512,512,3)
        self.conv15 = nn.Conv2d(512,512,3)
        self.conv16 = nn.Conv2d(512,512,3)

        self.batch_size = args.train_batch_size

        self._to_linear = None

        x = torch.randn(3,50,50).view(-1,3,50,50)
        self.convs(x)
        self.fc1 = nn.Linear(73728, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def convs(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        return x

    def forward(self,x):
        x = self.convs(x)
        x = torch.flatten(x,1)

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        Output = F.softmax(x)
        return Output
