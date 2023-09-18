import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class Generator(nn.Module):
    def __init__(self, nz = 100, ngf = 64, nc = 3):
        """Generator: G(z) (where z is a vector sampledfrom a standard normal distribution, 
        mean (μ) of 0 and a standard deviation (σ) of 1) → maps $z$ to data-space. 
        The goal is to estimate the distribution that the training data 
        comes from so it can generate fake samples from that estimated distribution.

        :param nz: Size of z latent vector (i.e. size of generator input), defaults to 100
        :type nz: int, optional
        :param ngf: Size of feature maps in generator, defaults to 64
        :type ngf: int, optional
        :param nc: Number of channels in the output images. For color images this is 3, defaults to 3
        :type nc: int, optional
        """
        super(Generator, self).__init__()

        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.relu(self.bn2(self.conv2(x)), True)
        x = F.relu(self.bn3(self.conv3(x)), True)
        x = F.relu(self.bn4(self.conv4(x)), True)
        x = F.tanh(self.conv5(x))

        return x


    def generate_tensor(self, batch_size):
        device = next(self.parameters()).device
        z = torch.randn(batch_size, self.nz, 1, 1, device=device)
        out = self.forward(z)

        return out


    def generate_image(self, image_path = 'test/output_image.jpg'):

        device = next(self.parameters()).device
        z = torch.randn(1, self.nz, 1, 1).to(device)
        out = self.forward(z).clamp(0, 1)
        image = Image.fromarray((out[0].permute(1, 2, 0) * 255).byte().cpu().numpy())
        image.save(image_path) 


    def get_shapes(self):

        x = torch.randn(1, self.nz, 1, 1); print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)), True); print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)), True); print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)), True); print(x.shape)
        x = F.relu(self.bn4(self.conv4(x)), True); print(x.shape)
        x = F.tanh(self.conv5(x))

        return x
    

def main():

    G = Generator()
    print(G)
    G.generate_image()
    G.get_shapes()

if __name__ == "__main__":
    main()