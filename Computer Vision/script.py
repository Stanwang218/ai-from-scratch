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

os.chdir(sys.path[0])

def train_script():
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='/Users/code/python/machine learning/dataset', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


    test_dataset = torchvision.datasets.MNIST(root='/Users/code/python/machine learning/dataset', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    model = VAE()
    epoch = 40
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for ep in range(epoch):
        total_loss = 0
        for x, y in test_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = model.loss_function(output[0], output[1], output[2], output[3])[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss / len(train_loader))

    torch.save(model.state_dict(), './vae_model.pth')
    
def test_script():
    model = VAE()
    model.load_state_dict(torch.load('/Users/code/python/machine learning/AI_from_scratch/Computer Vision/vae_model_new.pth', map_location=torch.device("cpu")))
    num_samples = 90
    latent_variables = torch.randn(num_samples, model.latent_dim)
    x = model.first_decode_layer(latent_variables).view(-1, 512, 2, 2)
    newPic = model.final_layer(model.decoder(x))
    print(newPic.shape)
    for i in range(num_samples // 9):
        image = newPic[i : i + 9, 0, :, :].cpu().detach().numpy()
        concat_img = np.concatenate([np.concatenate([image[3*j + i, :, :] for i in range(3)], axis=1) for j in range(3)], axis=0)
        plt.imshow(concat_img, cmap='gray')
        plt.title('Sample Image')
        plt.show() 

test_script()