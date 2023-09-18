import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, ndf = 64, nc = 3):
        """Discriminator: D(x) (where x is the input image) → outputs a scalar 
        probability that the image is real or fake (high when real, low when fake) 
        (real and fake image binary classifier). D(G(z)) will be used for fake images.

        :param ndf: Size of feature maps in discriminator, defaults to 64
        :type ndf: int, optional
        :param nc: Number of channels in the output images. For color images this is 3, defaults to 3
        :type nc: int, optional
        """
        super(Discriminator, self).__init__()

        self.nc = nc
        self.ndf = ndf

        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)


    def forward(self, x):

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.sigmoid(self.conv5(x))

        return x


    def get_shapes(self, x):
        
        print(x.shape)
        x = F.leaky_relu(self.conv1(x)); print(x.shape)
        x = F.leaky_relu(self.bn2(self.conv2(x))); print(x.shape)
        x = F.leaky_relu(self.bn3(self.conv3(x))); print(x.shape)
        x = F.leaky_relu(self.bn4(self.conv4(x))); print(x.shape)
        x = F.sigmoid(self.conv5(x)); print(x.shape)

        return x


def main():

    D = Discriminator()
    print(D)
    x = torch.randn(1, D.nc, D.ndf, D.ndf)
    D.get_shapes(x)
    out = D(x)
    print(out)

if __name__ == "__main__":
    main()