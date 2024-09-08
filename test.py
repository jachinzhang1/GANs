import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from gans import Generator, Discriminator
from train import Tensor

import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows', type=int, default=6, help='number of rows in the sample images')
    parser.add_argument('--ckpt_id', type=int, help='checkpoint id to load')
    parser.add_argument('--epoch_id', help='epoch id to load')
    parser.add_argument('--save_dir', type=str, default='./samples', help='save directory for the generated images')

    args = parser.parse_args()
    assert str(args.ckpt_id).isdigit(), "Checkpoint id must be a digit"
    assert str(args.epoch_id).isdigit(), "Epoch id must be a digit"
    
    args.ckpt_id = int(args.ckpt_id)
    args.epoch_id = int(args.epoch_id)
    load_path = './ckpts/run-%d/gen/epoch-%d.pth' % (args.ckpt_id, args.epoch_id)
    assert os.path.exists(load_path), "Checkpoint file does not exist"

    # Create the save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Load the trained generator model
    device = 'cpu'
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(load_path))
    generator.eval()

    # Generate a new image
    print('Generating...')
    noise = Variable(Tensor(np.random.normal(0, 1, (128, 100)))).to(device)
    generated_imgs = generator(noise)
    print('Done.')

    # Save the generated image
    sample_num = len(os.listdir(args.save_dir))
    sample_name = '%d-r%d-e%d.png' % (sample_num+1, args.ckpt_id, args.epoch_id)
    save_path = os.path.join(args.save_dir, sample_name)
    save_image(generated_imgs.data[:args.nrows**2], save_path, nrow=args.nrows, normalize=True)
    print('Sampled images have been saved successfully.')

