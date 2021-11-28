from types import new_class
import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable, backward
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, ngpu, n_classes=10, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        
        
        self.image_net = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 16 x 16
        )
        
        '''
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.label_net = nn.Sequential(
            # input is one-hot embedding (n_classes)
            nn.Conv2d(in_channels=n_classes, out_channels=ndf, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 16 x 16
        )
        '''
        self.label_condition = nn.Sequential(
            nn.Embedding(n_classes, n_classes),
            nn.Linear(n_classes, 16 * 16 * ndf)
        )

        self.main = nn.Sequential(
            # input is (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        img, label = input
        label_output = self.label_condition(label)
        label_output = label_output.view(-1, self.ndf, 16, 16)
        concat = torch.cat((self.image_net(img), label_output), dim=1)
        return self.main(concat)


class Generator(nn.Module):
    def __init__(self, ngpu, n_classes=10, nz=100, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nc = nc
        self.ngf = ngf

        self.noise_net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
            # state size. (ngf*4) x 4 x 4
        )

        '''
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.label_net = nn.Sequential(
            # input is one-hot embedding (emedding_dim)
            nn.ConvTranspose2d(in_channels=n_classes, out_channels=ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
            # state size. (ngf*4) x 4 x 4
        )
        '''

        self.label_condition = nn.Sequential(
            nn.Embedding(n_classes, n_classes),
            nn.Linear(n_classes, 4 * 4 * 4 * ngf)
        )

        self.main = nn.Sequential(
            # input is (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        noise, label = input
        label_output = self.label_condition(label)
        label_output = label_output.view(-1, self.ngf * 4, 4, 4)
        concat = torch.cat((self.noise_net(noise), label_output), dim=1)
        return self.main(concat)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CGAN():
    def __init__(self, ngpu, device, n_classes=10, embedding_dim=10, lr=0.0002, nc=1, ndf=64, nz=100, ngf=64, beta1=0.5):
        self.ngpu = ngpu
        self.device = device

        self.n_classes = n_classes
        self.embeding_dim = embedding_dim
        self.lr = lr
        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.ngf = ngf

        # Create the generator
        self.netG = Generator(ngpu, n_classes, nz, ngf, nc).to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        self.netG.apply(weights_init)

        # Create the Discriminator
        self.netD = Discriminator(ngpu, n_classes, nc, ndf).to(device)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netD.apply(weights_init)


        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        self.fixed_label = torch.arange(64, device=device)
        self.fixed_label = torch.remainder(self.fixed_label, n_classes)
        self.fixed_label= self.fixed_label.view(-1, 1)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))


    def train(self, dataloader, num_epochs, plot=False):

        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, (real_images, labels) in enumerate(dataloader):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_images = real_images.to(self.device)
                b_size = real_images.size(0)
                labels = labels.to(self.device)
                labels = labels.unsqueeze(1).long()

                real_target = Variable(torch.ones(b_size, 1).to(self.device))
                fake_target = Variable(torch.zeros(b_size, 1).to(self.device))

                output = self.netD((real_images, labels)).view(-1, 1)
                errD_real = self.criterion(output, real_target)
                errD_real.backward()
                D_x = output.mean().item()

                noise_vector = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                noise_vector = noise_vector.to(self.device)

                generated_image = self.netG((noise_vector, labels))
                output = self.netD((generated_image.detach(), labels)).view(-1, 1)
                errD_fake = self.criterion(output, fake_target)
                errD_fake.backward()   
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake

                self.optimizerD.step()


                self.optimizerG.zero_grad()
                output = self.netD((generated_image, labels)).view(-1, 1)
                errG = self.criterion(output, real_target)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG((self.fixed_noise, self.fixed_label)).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1


        if plot:
            plt.figure(figsize=(10,5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses,label="G")
            plt.plot(D_losses,label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        return img_list