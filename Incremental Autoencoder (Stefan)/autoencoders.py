import torch
import torch.nn as nn
import numpy as np

from pyro.distributions.relaxed_straight_through import RelaxedBernoulliStraightThrough

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        binary_encoded = RelaxedBernoulliStraightThrough(1, logits=encoded).rsample()
        
        decoded = self.decoder(binary_encoded)
        return binary_encoded, decoded
    
    

class IncrementalAutoEncoder(nn.Module):
    def __init__(self):
        super(IncrementalAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        binary_encoded = RelaxedBernoulliStraightThrough(1, logits=encoded).rsample()
    
        # mask the binary data with random noise at the end of the encoded data
        if self.training:
            x,y = binary_encoded.shape
            k_values = [np.random.randint(0, y+1) for _ in range(x)]
            mask = np.where([[0 if i < k else 1 for i in range(y)] for k in k_values])
            binary_encoded[mask] = torch.distributions.Bernoulli(probs=(torch.ones(x,y)/2)).sample().to(device)[mask]
        
        decoded = self.decoder(binary_encoded)
        return binary_encoded, decoded
    
    
    
class RandomAutoEncoder(nn.Module):
    def __init__(self):
        super(RandomAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        binary_encoded = RelaxedBernoulliStraightThrough(1, logits=encoded).rsample()
    
        # mask the binary data with random noise anywhere in the encoded data
        if self.training:
            x,y = binary_encoded.shape
            k_values = [np.random.randint(0, y+1) for _ in range(x)]
            row_indices, col_indices = np.array([]), np.array([])
            for i, k in enumerate(k_values):
                row_indices = np.append(row_indices, np.full(k,i))
                col_indices = np.append(col_indices, np.random.choice(y, k, replace=False))
            mask = (row_indices, col_indices)
            binary_encoded[mask] = torch.distributions.Bernoulli(probs=(torch.ones(x,y)/2)).sample().to(device)[mask]
            
        decoded = self.decoder(binary_encoded)
        return binary_encoded, decoded