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

print(len(train_loader))

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
    for ep in range(epoch):
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        times = 0
        for x, y in train_loader:
            times += 1
            optimizer.zero_grad()
            output = model(x)
            loss_res = model.loss_function(output[0], output[1], output[2], output[3], 0.5)
            loss = loss_res[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            recon_loss += loss_res[1]
            kl_loss += loss_res[2]
            if times > 15:
                break
            
        print(f"reconstruction loss: {recon_loss / (times)}")
        print(f"kl loss: {kl_loss / (times)}")
        print(f"total loss : {total_loss / (times)}")

    torch.save(model.state_dict(), './cele_vae_model.pth')
    
def test_script():
    model = VAE()
    model.load_state_dict(torch.load('/Users/code/python/machine learning/AI_from_scratch/Computer Vision/cele_vae_model.pth', map_location=torch.device("cpu")))
    num_samples = 90
    # test_loader = DataLoader(celeba_dataset(), batch_size=128, shuffle=True)
    # print(len(train_dataset))
    for x, y in test_loader:
        out = model(x)
        for i in range(64):
            img = out[0][0, 0, : ,].cpu().detach().numpy()
            img_resized = Image.fromarray((img * 255).astype('uint8'), 'L').resize((32, 32))
            img_resized_np = np.array(img_resized) / 255.0  # 转换为浮点数
            
            plt.imshow(img_resized_np, cmap='gray')
            plt.show()
        break
    # latent_variables = torch.randn(num_samples, model.latent_dim)
    # x = model.first_decode_layer(latent_variables).view(-1, 512, 2, 2)
    # newPic = model.final_layer(model.decoder(x))
    # print(newPic.shape)
    # for i in range(num_samples // 9):
    #     image = newPic[i : i + 9, 0, :, :].cpu().detach().numpy()
    #     concat_img = np.concatenate([np.concatenate([image[3*j + i, :, :] for i in range(3)], axis=1) for j in range(3)], axis=0)
    #     plt.imshow(concat_img, cmap='gray')
    #     plt.title('Sample Image')
    #     plt.show() 

        
train_script()

test_script()