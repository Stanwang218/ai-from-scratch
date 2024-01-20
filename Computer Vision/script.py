import torch
import torchvision
from torchvision import transforms
from vae import VAE 
import torch.optim as optim
import sys
import os

os.chdir(sys.path[0])


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