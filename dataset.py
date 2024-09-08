from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
import os

def get_dataloader(batchsize):
    dataset = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=Compose([
            Resize((28, 28)),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=True,
    )
    return dataloader
