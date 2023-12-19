import torch
import torch.nn as nn
import sys
import os
import cv2 as cv
import numpy as np
from torchinfo import summary

os.chdir(sys.path[0])

def splitPicture(patchSize = 16):
    img = cv.imread('./data/0000.jpg')
    img = cv.resize(img, (224, 224)).transpose(2, 0, 1)
    imgPatches = []
    h, w = img.shape[1], img.shape[1]
    for i in range(0,h,patchSize):
        for j in range(0,w,patchSize):
            imgPatches.append(img[:, i:i + patchSize, j : j + patchSize].reshape(-1))
    imgPatches = np.array(imgPatches)
    return torch.from_numpy(imgPatches) # 196, 768
    
class ViT(nn.Module):
    def __init__(self, picWidth = 224, picHeight = 224, patchSize = 16, channel = 3, dim = 768, num_class = 100):
        super().__init__()
        self.vecLength = patchSize * patchSize * channel
        assert picWidth % patchSize == 0 and picHeight % patchSize == 0
        self.pathNum = (picHeight // patchSize) * (picWidth // patchSize)
        self.embed = nn.Sequential(
            nn.LayerNorm(self.vecLength),
            nn.Linear(self.vecLength, dim),
            nn.LayerNorm(dim)
        )
        self.cls_token = nn.Parameter(torch.rand(1,1,dim))
        self.pos_encoder = nn.Parameter(torch.rand(1, self.pathNum + 1, dim))
        
        self.encoderLayer = nn.TransformerEncoderLayer(dim, 12, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(self.encoderLayer, 6)
        self.classify = nn.Sequential(
            nn.Linear(dim, num_class),
            nn.Dropout(),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_num = x.size(0)
        x = self.embed(x) # B, P, L -> B, P, D (batch, patches, length -> batch patches, dim)
        cls_token = self.cls_token.repeat(batch_num, 1, 1)
        x = torch.concat([x, cls_token], dim = 1)
        x += self.pos_encoder
        print(x.shape)
        x = self.encoder(x)
        x = self.classify(x)
        return x
    
splitPicture()

model = ViT()
summary(model, (10, 196, 768))
