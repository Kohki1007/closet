import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from pix2pix_network import define_G, define_D
from gan_losses import GANLoss
from loss import Loss

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
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
        x = self.model(x)

        return x
    
class decoder(nn.Module):
    def __init__(self, output):
        super(decoder, self).__init__()

        self.model = nn.Sequential(
                        nn.Linear(512, output),
                        nn.Tanh()
        )

    def forward(self, x):
        # x = torch.tanh(self.model(x))
        x = self.model(x)

        return x
    
class pix2pix_encoder(nn.Module):
    def __init__(self, output):
        super(pix2pix_encoder, self).__init__()

        self.model_1 = nn.Sequential(
                        nn.Conv2d(5, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        )

        self.model_2 = nn.Sequential(
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_3 = nn.Sequential(
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_4 = nn.Sequential(
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_5 = nn.Sequential(
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_6 = nn.Sequential(
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_7 = nn.Sequential(
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_8 = nn.Sequential(
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        x1 = self.model_1(x)
        x2 = self.model_2(x1)
        x3 = self.model_3(x2)
        x4 = self.model_4(x3)
        x5 = self.model_5(x4)
        x6 = self.model_6(x5)
        x7 = self.model_7(x6)
        x8 = self.model_8(x7)

        return x1, x2, x3, x4, x5, x6, x7, x8


class pix2pix_decoder(nn.Module):
    def __init__(self, output):
        super(pix2pix_decoder, self).__init__()

        self.model_1 = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Dropout(p=0.5, inplace=False)
        )

        self.model_2 = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Dropout(p=0.5, inplace=False)
        )

        self.model_3 = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Dropout(p=0.5, inplace=False)
        )

        self.model_4 = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(1024, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_5 = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_6 = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.model_7 = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(128, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                        nn.Tanh()
        )

    def forward(self, encoder_output):
        x = torch.cat(self.model_1(encoder_output[7]), encoder_output[6], dim = 1)
        x = torch.cat(self.model_2(x), encoder_output[5], dim = 1)
        x = torch.cat(self.model_3(x), encoder_output[4], dim = 1)
        x = torch.cat(self.model_4(x), encoder_output[3], dim = 1)
        x = torch.cat(self.model_5(x), encoder_output[2], dim = 1)
        x = torch.cat(self.model_6(x), encoder_output[1], dim = 1)
        x = torch.cat(self.model_7(x), encoder_output[0], dim = 1)

        return x

class definet(nn.Module):
    def __init__(self, output = 6):
        super(definet, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder(output)
        self.Global_Avg_pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0)


    def forward(self,x1,x2):
        x1 = self.encoder(x1)
        # x2 = self.encoder(x2)
        # x = x1 - x2
        # x = self.Global_Avg_pooling(x)
        # x = x.squeeze()
        # x = self.decoder(x)

        return x1

class Autoencoder(nn.Module):
    def __init__(self, train = True):
        super(Autoencoder, self).__init__()
        self.generate_target = define_G(5, 2, 64, 'unet_256')
        self.delete_ur = define_G(4, 1, 64, 'unet_256')
        # self.decoder_generate_target_image = pix2pix_decoder()
        if train:
            self.target_discriminator = define_D(7)
            self.ur_discriminator = define_D(5)

        self.predict_action = definet()


        self.criterionGAN = GANLoss('vanilla').to("cuda")
        self.criterionL1 = torch.nn.L1Loss()
        self.action_loss = Loss()

    def forward(self, current_image, target_image):
        # target_image = self.encoder_generate_target_image(current_image)
        # target_image = self.pix2pix(current_image)
        feature = self.predict_action(current_image, target_image)

        return feature

    def forward_solo(self, init_image, current_image):
        # target_image = self.encoder_generate_target_image(current_image)
        self.target_image = self.generate_target(init_image)
        self.delete_ur_image = self.delete_ur(current_image)

        # image = self.delete_ur_image[0, 0]*255

        # depth_predict = image.to('cpu').detach().numpy().copy()
        # depth_predict = Image.fromarray((depth_predict).astype(np.uint8))
        # depth_predict.save(f"/home/engawa/py_ws/visual_servo/src/network/delete.png")
        # self.feature = self.predict_action(current_image, self.target_image)

        # return target_image
    
    def backward_D(self, init, target_init, current, delete_current):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        self.forward_solo(init, current)
        fake_target = torch.cat((init, self.target_image), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_delete = torch.cat((current, self.delete_ur_image), 1)
        pred_fake_target = self.target_discriminator(fake_target.detach())
        pred_fake_delete = self.ur_discriminator(fake_delete.detach())
        self.loss_D_fake_target = self.criterionGAN(pred_fake_target, False)
        self.loss_D_fake_delete = self.criterionGAN(pred_fake_delete, False)

        real_target = torch.cat((init, target_init), 1)
        real_delete = torch.cat((current, delete_current[:, 0].unsqueeze(1)), 1)############
        
        pred_real_target = self.target_discriminator(real_target)
        pred_real_delete = self.ur_discriminator(real_delete)
        self.loss_D_real_target = self.criterionGAN(pred_real_target, True)
        self.loss_D_real_delete = self.criterionGAN(pred_real_delete, True)
        # combine loss and calculate gradients
        self.loss_D_target = (self.loss_D_fake_target + self.loss_D_real_target) * 0.5
        self.loss_D_delete = (self.loss_D_fake_delete + self.loss_D_real_delete) * 0.5
        self.loss_D_target.backward()
        self.loss_D_delete.backward()

    def backward_other(self, init, target_init, current, delete_current, action, attension):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_target = torch.cat((init, self.target_image), 1)
        fake_delete = torch.cat((current, self.delete_ur_image), 1)
        # print(input.shape)
        pred_fake_target = self.target_discriminator(fake_target)
        pred_fake_delete = self.ur_discriminator(fake_delete)
        delete_ur_image= torch.cat([self.delete_ur_image, attension], 1)
        pred_action = self.forward(delete_ur_image.detach(), self.target_image.detach())
        self.loss_G_GAN_target = self.criterionGAN(pred_fake_target, True)
        self.loss_G_GAN_delete = self.criterionGAN(pred_fake_delete, True)
        # Second, G(A) = B
        self.loss_G_L1_target = self.criterionL1(self.target_image, target_init) * 100
        self.loss_G_L1_delete = self.criterionL1(self.delete_ur_image, delete_current) * 100
        # combine loss and calculate gradients
        self.loss_action = self.action_loss(pred_action, action)
        self.loss_G_target = self.loss_G_GAN_target + self.loss_G_L1_target
        self.loss_G_delete = self.loss_G_GAN_delete + self.loss_G_L1_delete
        self.loss_action.backward()
        self.loss_G_target.backward()
        self.loss_G_delete.backward()

        return pred_action

    def get_feature(self, init, current, attension):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.forward_solo(init, current)
        # print(input.shape)
        pred_action = self.forward(self.delete_ur_image.detach(), self.target_image.detach())

        return pred_action
    
    def predict_only(self, target, current):
        pred_action = self.forward(target, current)

        return pred_action

