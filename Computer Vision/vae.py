import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# encoder : learn a distribution q(z|x) minimize(D_KL(q, p(z|x)))
# decoder : learn a distribution p(x|z) ~ Gaussion(\miu, covariance)
# Assumption : q(z|x) ~ Gaussion(I, \sigma^2) latent variables are independent
# p(z) ~ N(0, I)
# how to get z ? Sample from q(z|x), but sampling can't propogate gradient
# Use reparameterization

class VAE(nn.Module):
    def __init__(self, in_channels = 1, latent_dim = 64, hidden_dims = None):
        # input img : 64 x 64
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # build decoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        modules.append(nn.Flatten())

        self.encoder = nn.Sequential(*modules)
        self.fc_miu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        
        # build decoder
        hidden_dims.reverse()
        self.first_decode_layer = nn.Linear(latent_dim, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.final_layer = nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1],
                                        hidden_dims[-1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                kernel_size= 3, padding= 1),
                    nn.Tanh())



        self.decoder = nn.Sequential(*modules)
        
    def reparameterization(self, miu, log_var):
        eps = torch.randn_like(log_var)
        std = torch.exp(log_var * 0.5) # log_var -> std
        return miu + eps * std # latent variable
    
    def forward(self, x):
        input_pic = x
        x = self.encoder(x)
        miu, log_var = self.fc_miu(x), self.fc_var(x)
        latent_varible = self.reparameterization(miu, log_var)
        x = self.first_decode_layer(latent_varible)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return [x, input_pic, miu, log_var]
    
    def loss_function(self, recon_pic, input_pic, miu, log_var):
        recon_loss = F.mse_loss(recon_pic, input_pic)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - miu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recon_loss + kld_loss
        return [loss, recon_loss.detach(), kld_loss.detach()]
    
if __name__ == '__main__':
    model = VAE()
    summary(model, (1, 1, 64, 64))
    
        