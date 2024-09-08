import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from dataset import get_dataloader
from gans import Generator, Discriminator

import os
import argparse
from time import time
from torch.utils.tensorboard import SummaryWriter


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
writer = SummaryWriter()

def train(args,
          dataloader, 
          generator, 
          discriminator, 
          optimizer_G,
          optimizer_D,
          loss_fn,
          runs_num):
    
    if not os.path.exists("./ckpts"):
        os.mkdir("./ckpts")
    ckpts_num = len(os.listdir("./ckpts"))
    os.mkdir("./ckpts/run-%d" % (ckpts_num + 1))
    os.mkdir("./ckpts/run-%d/gen" % (ckpts_num + 1))
    os.mkdir("./ckpts/run-%d/disc" % (ckpts_num + 1))
    time_total = 0

    for epoch in range(1, 1 + args.n_epochs):

        beginning = time()

        for i, (img, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(img.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(img.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(img.type(Tensor))

            ## Train Generator
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (img.shape[0], args.latent_dim))))
            # Generate a batch of images
            gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = loss_fn(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            ## Train Discriminator
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss_fn(discriminator(real_imgs), valid)
            fake_loss = loss_fn(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i

            if (i + 1) % 50 == 0 or i == len(dataloader) - 1:
                print("[Epoch %d/%d Batch %d/%d] D loss: %.4f G loss: %.4f"
                      % (epoch, args.n_epochs, 
                         i + 1, 
                         len(dataloader), 
                         d_loss.item(), 
                         g_loss.item()))
            writer.add_scalar('d_loss', d_loss.item(), len(dataloader) * epoch + i)
            writer.add_scalar('g_loss', g_loss.item(), len(dataloader) * epoch + i)
            
            
            # batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:args.nrow ** 2], "./images/run-%d/%d.png" % (runs_num + 1, batches_done), nrow=args.nrow, normalize=True)

        epoch_last_time = time() - beginning
        time_total += epoch_last_time
        minutes, seconds = divmod(time_total, 60)
        print("Epoch %d took %.2f seconds. %dm%.2fs in total.\n" % (epoch, epoch_last_time, minutes, seconds))

        if epoch % args.ckpts_interval == 0:
            assert os.path.exists('./ckpts/run-%d' % (ckpts_num + 1))
            torch.save(generator.state_dict(), "./ckpts/run-%d/gen/epoch-%d.pth" % (ckpts_num + 1, epoch))
            torch.save(discriminator.state_dict(), "./ckpts/run-%d/disc/epoch-%d.pth" % (ckpts_num + 1, epoch))


def check_load(args, net_type: str):
    if net_type == 'gen':
        check_path = f'./ckpts/run-{args.run_id}/' + net_type + f'/epoch-{args.epoch_id}.pth'
    elif net_type == 'disc':
        check_path = f'./ckpts/run-{args.run_id}/' + net_type + f'/epoch-{args.epoch_id}.pth'
    else:
        check_path = None
        raise ValueError("net type must be 'gen' or 'disc'")
    if os.path.exists(check_path):
        return check_path, True
    else:
        raise ValueError("checkpoint not found")


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(args.batch_size)

    loss_fn = nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    if args.warm_start:
        gen_path, _ = check_load(args, 'gen')
        generator.load_state_dict(torch.load(gen_path))
        disc_path, _ = check_load(args, 'disc')
        discriminator.load_state_dict(torch.load(disc_path))
        print('Pretrained models loaded.')

    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if os.path.exists("./images") is False:
        os.mkdir("./images")
    runs_num = len(os.listdir("./images"))
    assert not os.path.exists("./images/run-%d" % (runs_num + 1)), "Run folder %d already exists" % (runs_num + 1)
    os.mkdir("./images/run-%d" % (runs_num + 1))

    train(args, 
          dataloader, 
          generator, 
          discriminator, 
          optimizer_G, 
          optimizer_D, 
          loss_fn, 
          runs_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--lr', type=float, default=2e-4, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--sample_interval', type=int, default=500, help='interval betwen image samples')
    parser.add_argument('--ckpts_interval', type=int, default=20, help='interval betwen saved checkpoints')
    parser.add_argument('--nrow', type=int, default=6, help='number of rows in the sample images')
    parser.add_argument('--warm_start', type=bool, default=False, help='whether to load pretrained checkpoint')
    parser.add_argument('--run_id', type=int, help='run id to load the checkpoint')
    parser.add_argument('--epoch_id', type=int, help='epoch id to load the checkpoint')

    args = parser.parse_args()
    main(args)
