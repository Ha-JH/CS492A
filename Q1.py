import argparse
import logging
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch import cuda

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from DCGAN import DCGAN
from utils import get_data_loader, generate_images, save_gif


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGANS MNIST')
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs available. Use 0 for CPU mode')
    parser.add_argument('--ndf', type=int, default=128, help='Number of features to be used in Discriminator network')
    parser.add_argument('--ngf', type=int, default=128, help='Number of features to be used in Generator network')
    parser.add_argument('--nz', type=int, default=100, help='Size of the noise')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta for Adam optimizers')
    parser.add_argument('--nc', type=int, default=1, help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--image-size', type=int, default=64, help='Image size')
    parser.add_argument('--num-test-samples', type=int, default=64, help='Number of samples to visualize')
    parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
    parser.add_argument('--fps', type=int, default=5, help='frames-per-second value for the gif')
    parser.add_argument('--use-fixed', action='store_true', help='Boolean to use fixed noise or not')
    parser.add_argument('--plot', default=True, action='store_true', help='plot train loss')



    args = parser.parse_args()

    # Gather MNIST Dataset
    transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=0.5, std=0.5)
                            #    transforms.Normalize(mean=(0.1307, ), std=(0.3081, )),
                           ])    
    dataset = dset.MNIST(root='./mnist_data/',
                           transform=transform, download=True)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True)

    # Device configuration
    device = torch.device('cuda:0' if (torch.cuda.is_available() and args.ngpu > 0) else 'cpu')
    print("Using", cuda.get_device_name(0))


    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    dcgan = DCGAN(ngpu=args.ngpu, device=device, lr=args.lr, nc=args.nc, ndf=args.ndf, nz=args.nz, ngf=args.ngf, beta1=args.beta)
    
    # initialize other variables
    num_batches = len(dataloader)
    fixed_noise = torch.randn(args.num_test_samples, args.nz, 1, 1, device=device)



    img_list = dcgan.train(dataloader=dataloader, num_epochs=args.num_epochs, plot=args.plot)


    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    writergif = animation.PillowWriter(fps=30) 
    ani.save(args.output_path+"fake_dcgan.gif", writer=writergif)

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()