import torch
import torch.nn as nn
from model.generator import Generator
from model.discriminator import Discriminator
import os


def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class DCGAN(nn.Module):
    def __init__(self, nz = 100, ngf = 64, ndf = 64, nc = 3):
        super(DCGAN, self).__init__()

        self.generator = Generator(nz, ngf, nc)
        self.discriminator = Discriminator(ndf, nc)


    def initialise(self):
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    
    def save_checkpoint(self, weights_filepath = 'weights/run.pt'):
        torch.save(self.state_dict(), weights_filepath)