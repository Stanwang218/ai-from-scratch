import torch
import torchvision
from torchvision import transforms
from vae import VAE 
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

os.chdir(sys.path[0])


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='/Users/code/python/machine learning/dataset', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


test_dataset = torchvision.datasets.MNIST(root='/Users/code/python/machine learning/dataset', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


class celeba_dataset(Dataset):
    def __init__(self, path = "/Users/code/python/machine learning/dataset/celeba/pic") -> None:
        super().__init__()
        self.path = path
        self.pictures = os.listdir(self.path)
    
    def __getitem__(self, index):
        pic_path = os.path.join(self.path, self.pictures[index])
        img = Image.open(pic_path).convert('L')
        img = img.resize((64, 64))
        # img.show()
        return transform(img)
    
    def __len__(self):
        return len(self.pictures)


def train_script():
    # train_loader = DataLoader(celeba_dataset(), batch_size=128, shuffle=True)
    model = VAE()
    epoch = 10
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, 
                             patience=5, threshold=0.001, cooldown=0,
                             min_lr=0.0001, verbose=True)

    for ep in range(epoch):
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss_res = model.loss_function(output[0], output[1], output[2], output[3], 0.5)
            loss = loss_res[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            recon_loss += loss_res[1]
            kl_loss += loss_res[2]
            
        print(f"reconstruction loss: {recon_loss / len(train_loader)}")
        print(f"kl loss: {kl_loss / (len(train_loader))}")
        print(f"total loss : {total_loss / len(train_loader)}")
        scheduler.step(total_loss / len(train_loader))

    torch.save(model.state_dict(), './pth/cele_vae_model.pth')
    
def test_script():
    model = VAE()
    model.load_state_dict(torch.load('./pth/cele_vae_model.pth', map_location=torch.device("cpu")))
    # for x, y in test_loader:
    for x in DataLoader(celeba_dataset(), batch_size=128, shuffle=True):
        out = model(x)
        img = out[0].cpu().detach().numpy()
        origin_img = x.cpu().detach().numpy()
        origin_img = np.concatenate([np.concatenate([origin_img[8*j + i, 0, :, :] for i in range(8)], axis=1) for j in range(8)], axis=0)
        concat_img = np.concatenate([np.concatenate([img[8*j + i, 0, :, :] for i in range(8)], axis=1) for j in range(8)], axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(concat_img, cmap='gray')
        axes[0].set_title("Construction Picture")
        axes[1].imshow(origin_img, cmap='gray')
        axes[1].set_title("Original Picture")
        plt.show() 
        

        
train_script()

test_script()