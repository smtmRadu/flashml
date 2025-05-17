import torch
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(input_dim, input_dim * 2 // 3)
        self.fc2 = nn.Linear(input_dim * 2 // 3, input_dim // 3)
        self.mu = nn.Linear(input_dim // 3, latent_dim)
        self.logvar = nn.Linear(input_dim // 3, latent_dim)

        self.fc_out1 = nn.Linear(latent_dim, input_dim // 3)
        self.fc_out2 = nn.Linear(input_dim // 3, input_dim * 2 // 3)
        self.fc_out3 = nn.Linear(input_dim * 2 // 3, input_dim)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.mu.weight)
        torch.nn.init.kaiming_normal_(self.logvar.weight)
        torch.nn.init.kaiming_normal_(self.fc_out1.weight)
        torch.nn.init.kaiming_normal_(self.fc_out2.weight)
        torch.nn.init.kaiming_normal_(self.fc_out3.weight)

    def encode(self, x):
        '''
        Returns the mean and log variance.
        '''
        h = F.silu(self.fc2(F.silu(self.fc1(x))))
        return self.mu(h), self.logvar(h)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.silu(self.fc_out2(F.silu(self.fc_out1(z))))
        return self.fc_out3(h)
    
    def forward(self, x):
        '''
        Returns: x_hat, mu, logvar'''
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, x, x_hat, mu, logvar, beta=1.0):
        BCE = F.mse_loss(x_hat, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta*KLD
