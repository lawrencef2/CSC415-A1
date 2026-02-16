"""
CURL: Contrastive Unsupervised Representations for Reinforcement Learning
Implementation of the contrastive learning component for visual RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Tuple, Optional
import kornia.augmentation as K


class RandomShiftsAug(nn.Module):
    """
    Random shift augmentation commonly used in CURL.
    Pads image and then randomly crops back to original size.
    """
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor [B, C, H, W]
        Returns:
            augmented tensor [B, C, H, W]
        """
        n, c, h, w = x.size()
        assert h == w, "Height and width must be equal"
        
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        
        # Random crop
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, 
                                device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), 
                              device=x.device, dtype=x.dtype)
        shift = shift * 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class Encoder(nn.Module):
    """
    Convolutional encoder for processing image observations.
    Architecture based on the CURL paper.
    """
    def __init__(self, obs_shape: Tuple[int, int, int], z_dim: int = 50):
        super().__init__()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.z_dim = z_dim
        
        # Convolutional layers
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            dummy_output = self.convnet(dummy_input)
            self.conv_out_size = int(np.prod(dummy_output.shape[1:]))
        
        # Projection head
        self.fc = nn.Linear(self.conv_out_size, z_dim)
        self.ln = nn.LayerNorm(z_dim)

    def forward(self, obs: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """
        Args:
            obs: observations [B, C, H, W]
            detach: whether to detach from computation graph
        Returns:
            latent representation [B, z_dim]
        """
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        z = self.fc(h)
        z = self.ln(z)
        
        if detach:
            z = z.detach()
        
        return z


class CURL:
    """
    CURL: Contrastive Unsupervised Representations for Reinforcement Learning
    """
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        z_dim: int = 50,
        lr: float = 1e-3,
        momentum: float = 0.95,
        temperature: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            obs_shape: shape of observations (C, H, W)
            z_dim: dimension of latent representation
            lr: learning rate
            momentum: momentum coefficient for target encoder update
            temperature: temperature parameter for contrastive loss
            device: device to run on
        """
        self.device = device
        self.z_dim = z_dim
        self.momentum = momentum
        self.temperature = temperature
        
        # Query encoder (online)
        self.encoder_q = Encoder(obs_shape, z_dim).to(device)
        
        # Key encoder (target/momentum)
        self.encoder_k = Encoder(obs_shape, z_dim).to(device)
        
        # Initialize key encoder with query encoder parameters
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        
        # Key encoder is not trained by gradient descent
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # Bilinear product matrix W
        self.W = nn.Parameter(torch.randn(z_dim, z_dim)).to(device)
        
        # Augmentation
        self.aug = RandomShiftsAug(pad=4).to(device)
        
        # Optimizer for query encoder and W
        self.optimizer = optim.Adam(
            list(self.encoder_q.parameters()) + [self.W],
            lr=lr
        )
        
        self.train()
    
    def train(self, training: bool = True):
        """Set training mode"""
        self.training = training
        self.encoder_q.train(training)
        self.encoder_k.train(False)  # Key encoder always in eval mode
    
    def compute_logits(self, z_q: torch.Tensor, z_k: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity logits using bilinear product.
        
        Args:
            z_q: query latents [B, z_dim]
            z_k: key latents [B, z_dim]
        Returns:
            logits [B, B]
        """
        # Normalize embeddings
        z_q = F.normalize(z_q, dim=1)
        z_k = F.normalize(z_k, dim=1)
        
        # Bilinear product: z_q @ W @ z_k^T
        proj_k = torch.matmul(self.W, z_k.T)  # [z_dim, B]
        logits = torch.matmul(z_q, proj_k)  # [B, B]
        
        # Apply temperature
        logits = logits / self.temperature
        
        return logits
    
    def contrastive_loss(self, z_q: torch.Tensor, z_k: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            z_q: query latents [B, z_dim]
            z_k: key latents [B, z_dim]
        Returns:
            loss scalar
        """
        logits = self.compute_logits(z_q, z_k)
        
        # Subtract max for numerical stability
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        
        # Labels are diagonal elements (positive pairs)
        labels = torch.arange(logits.shape[0], device=self.device, dtype=torch.long)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def update(self, obs: torch.Tensor) -> dict:
        """
        Update CURL with a batch of observations.
        
        Args:
            obs: batch of observations [B, C, H, W]
        Returns:
            dictionary with training metrics
        """
        obs = obs.to(self.device)
        
        # Apply two different random augmentations
        obs_q = self.aug(obs)
        obs_k = self.aug(obs)
        
        # Compute query and key embeddings
        z_q = self.encoder_q(obs_q)
        
        with torch.no_grad():
            z_k = self.encoder_k(obs_k)
        
        # Compute loss
        loss = self.contrastive_loss(z_q, z_k)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update key encoder with momentum
        self._momentum_update()
        
        # Compute accuracy (for monitoring)
        logits = self.compute_logits(z_q, z_k.detach())
        labels = torch.arange(logits.shape[0], device=self.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        
        return {
            'curl_loss': loss.item(),
            'curl_accuracy': accuracy.item()
        }
    
    def _momentum_update(self):
        """Update key encoder parameters using momentum."""
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data = self.momentum * param_k.data + \
                          (1.0 - self.momentum) * param_q.data
    
    def encode(self, obs: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Encode observations to latent representations.
        
        Args:
            obs: observations [B, C, H, W]
            detach: whether to detach from computation graph
        Returns:
            latent representations [B, z_dim]
        """
        obs = obs.to(self.device)
        return self.encoder_q(obs, detach=detach)


# Example usage and training loop
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    obs_shape = (9, 84, 84)  # 3 stacked frames of 3 channels (RGB)
    z_dim = 50
    num_iterations = 1000
    
    # Initialize CURL
    curl = CURL(
        obs_shape=obs_shape,
        z_dim=z_dim,
        lr=1e-3,
        momentum=0.95,
        temperature=0.1
    )
    
    print(f"CURL initialized on {curl.device}")
    print(f"Encoder architecture:")
    print(curl.encoder_q)
    
    # Simulate training loop
    print("\nTraining CURL...")
    for i in range(num_iterations):
        # Simulate batch of observations from replay buffer
        obs = torch.randn(batch_size, *obs_shape)
        
        # Update CURL
        metrics = curl.update(obs)
        
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{num_iterations}")
            print(f"  Loss: {metrics['curl_loss']:.4f}")
            print(f"  Accuracy: {metrics['curl_accuracy']:.4f}")
    
    # Test encoding
    print("\nTesting encoding...")
    test_obs = torch.randn(4, *obs_shape)
    with torch.no_grad():
        z = curl.encode(test_obs)
    print(f"Encoded shape: {z.shape}")
    print("CURL implementation complete!")
