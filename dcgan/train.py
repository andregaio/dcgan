import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import wandb
import argparse
from model.dcgan import DCGAN
from data import celeb_data_transforms


manualSeed = 37
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)


def train(args):

    if args.wandb:
        run = wandb.init(
            project='dcgan',            
            config={
                'dcgan': 'dcgan',
                'dataset': 'Celeb-A Faces',
                'epochs': args.epochs,
                'batch' : args.batch,
                'learning_rate': args.learning_rate,
                'beta1' : args.beta1,
            })

    dataroot = "/home/andre/repos/dcgan/dataset/"
    batch_size = 128
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5

    device = torch.device("cuda:0")

    dataset = dset.ImageFolder(root=dataroot, transform=celeb_data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = DCGAN()
    model.initialise()
    model.generator = model.generator.to(device)
    model.discriminator = model.discriminator.to(device)

    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    optimizer_discriminator = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_generator = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(num_epochs):
        
        for i, data in enumerate(dataloader, 0):

            # Update Discriminator
            # Real batch
            model.discriminator.zero_grad()
            x_real = data[0].to(device)
            y = torch.full((x_real.size(0),), real_label).to(device)
            y_hat = model.discriminator(x_real).view(-1)
            loss_discriminator_real = criterion(y_hat, y)
            loss_discriminator_real.backward()

            # Fake batch
            x_fake = model.generator.generate_tensor(x_real.size(0))
            y.fill_(fake_label)
            y_hat = model.discriminator(x_fake.detach()).view(-1)
            loss_discriminator_fake = criterion(y_hat, y)
            loss_discriminator_fake.backward()
            loss_discriminator = loss_discriminator_real + loss_discriminator_fake
            optimizer_discriminator.step()

            # Update Generator
            model.generator.zero_grad()
            y.fill_(real_label)
            y_hat = model.discriminator(x_fake).view(-1)
            loss_generator = criterion(y_hat, y)
            loss_generator.backward()
            optimizer_generator.step()
            
            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch + 1}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {loss_discriminator.item()}\tLoss_G: {loss_generator.item()}')

            if args.wandb:
                wandb.log({'loss_discriminator_real': loss_discriminator_real,
                            'loss_discriminator_fake': loss_discriminator_fake,
                            'loss_generator': loss_generator,
                            'loss_discriminator': loss_discriminator,
                            'epoch': epoch + 1,
                        })
            
            model.generator.generate_image()

        model.save_checkpoint('weights/last_weights.pt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--batch", type = int, default = 128)
    parser.add_argument("--epochs", type = int, default = 5)
    parser.add_argument("--learning_rate", type = float, default = .0002)
    parser.add_argument("--beta1", type = float, default = 0.5)
    parser.add_argument("--wandb", action="store_true", default = False)
    args = parser.parse_args()

    train(args)