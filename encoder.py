import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # x = F.relu(self.conv4(x))
        # x = self.pool4(x)
        
        # x = F.relu(self.conv5(x))
        # x = self.pool5(x)
        
        
        return x
    

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        # self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv7 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        # self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        # self.deconv7 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)



    def forward(self, x):
        # x = F.relu(self.deconv3(x))
        # x = F.relu(self.deconv4(x))
        # x = F.relu(self.deconv5(x))
        # x = F.relu(self.deconv6(x))
        # x = self.deconv7(x)

        # x = F.relu(self.deconv1(x))
        # x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        return x
    

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class encoder_color(nn.Module):
    def __init__(self):
        super(encoder_color, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        
        return x
    

class decoder_color(nn.Module):
    def __init__(self):
        super(decoder_color, self).__init__()

        # self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv7 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        # self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        # self.deconv7 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)



    def forward(self, x):
        # x = F.relu(self.deconv3(x))
        # x = F.relu(self.deconv4(x))
        # x = F.relu(self.deconv5(x))
        # x = F.relu(self.deconv6(x))
        # x = self.deconv7(x)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        return x
    

class Autoencoder_color(nn.Module):
    def __init__(self):
        super(Autoencoder_color, self).__init__()
        self.encoder = encoder_color()
        self.decoder = decoder_color()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class encoder_attension(nn.Module):
    def __init__(self):
        super(encoder_attension, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        # x = F.relu(self.conv5(x))
        # x = self.pool5(x)
        
        
        return x
    

class decoder_attension(nn.Module):
    def __init__(self):
        super(decoder_attension, self).__init__()

        # self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv7 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        # self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        # self.deconv7 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)



    def forward(self, x):
        # x = F.relu(self.deconv3(x))
        # x = F.relu(self.deconv4(x))
        # x = F.relu(self.deconv5(x))
        # x = F.relu(self.deconv6(x))
        # x = self.deconv7(x)

        # x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        return x
    

class Autoencoder_attension(nn.Module):
    def __init__(self):
        super(Autoencoder_attension, self).__init__()
        self.encoder = encoder_attension()
        self.decoder = decoder_attension()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class encoder_rgbd_attension(nn.Module):
    def __init__(self):
        super(encoder_rgbd_attension, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels=80, out_channels=160, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        
        return x
    

class decoder_rgbd_attension(nn.Module):
    def __init__(self):
        super(decoder_rgbd_attension, self).__init__()

        # self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.deconv7 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

        # self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        # self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        # self.deconv7 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.deconv1 = nn.Conv2d(in_channels=320, out_channels=160, kernel_size=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=160, out_channels=80, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=80, out_channels=40, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=40, out_channels=20, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=20, out_channels=5, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)



    def forward(self, x):
        # x = F.relu(self.deconv3(x))
        # x = F.relu(self.deconv4(x))
        # x = F.relu(self.deconv5(x))
        # x = F.relu(self.deconv6(x))
        # x = self.deconv7(x)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = self.deconv6(x)
        return x
    

class Autoencoder_rgbd_attension(nn.Module):
    def __init__(self):
        super(Autoencoder_rgbd_attension, self).__init__()
        self.encoder = encoder_rgbd_attension()
        self.decoder = decoder_rgbd_attension()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x