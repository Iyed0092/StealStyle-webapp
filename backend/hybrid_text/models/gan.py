
"""
Embedding-level GAN for style transformation.
Generator maps encoder aggregated embeddings -> style-conditioned embeddings,
Discriminator scores embeddings as real (modern->classical) or fake.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.norm(x + h)

class Generator(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, n_blocks=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        for _ in range(n_blocks):
            layers.append(ResidualMLP(hidden_dim, hidden_dim))
        layers += [nn.Linear(hidden_dim, input_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, spectral_norm=True):
        super().__init__()
        Linear = lambda l: nn.utils.spectral_norm(l) if spectral_norm else l
        self.net = nn.Sequential(
            Linear(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            Linear(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            Linear(nn.Linear(hidden_dim // 2, 1))
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  
