import torch 
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
    ) # Data Extraction

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10) #Data Loading
