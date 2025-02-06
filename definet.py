import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
                        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), 
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )


    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.pool1(x)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.pool2(x)
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        # x = self.pool3(x)
        # x = F.relu(self.conv8(x))
        # x = F.relu(self.conv9(x))
        # x = F.relu(self.conv10(x))
        # x = self.pool4(x)
        # x = F.relu(self.conv11(x))
        # x = F.relu(self.conv12(x))
        # x = F.relu(self.conv13(x))
        # x = self.pool5(x)

        x = self.model(x)

        return x


class decoder(nn.Module):
    def __init__(self, output):
        super(decoder, self).__init__()

        self.model = nn.Sequential(
                        # nn.Linear(512, 200),
                        # nn.Linear(200, 100),
                        nn.Linear(512, output),
                        nn.Tanh()
        )

    def forward(self, x):
        # x = torch.tanh(self.model(x))
        x = self.model(x)

        return x
    

class Autoencoder_only(nn.Module):
    def __init__(self, output):
        super(Autoencoder_only, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder(output)
        self.Global_Avg_pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0)

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = x1 - x2
        x = self.Global_Avg_pooling(x)
        x = x.squeeze()
        x = self.decoder(x)

        return x
    
class Autoencoder_solo(nn.Module):
    def __init__(self, output):
        super(Autoencoder_solo, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder(output)
        self.Global_Avg_pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.Global_Avg_pooling(x)
        x = x.squeeze()
        x = self.decoder(x)

        return x