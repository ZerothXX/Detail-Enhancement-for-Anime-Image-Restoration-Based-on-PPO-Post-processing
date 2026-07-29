'''
models.py -- All model definitions
  - LamaWrapper: frozen pre-trained LaMa model
  - DetailEnhancer: fixed (non-trainable) multi-scale detail extraction
  - StateEncoder: encodes (I_hat, mask) to state features
  - ActorCritic: PPO policy + value network
  - Discriminator: optional GAN discriminator (disabled by default)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# LamaWrapper: frozen pre-trained LaMa model
# ============================================================
class LamaWrapper(nn.Module):
    """Load TorchScript pre-trained LaMa, freeze all parameters."""
    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__()
        self.device = device
        self.model = torch.jit.load(checkpoint_path, map_location=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, image, mask):
        return self.model(image, mask)


# ============================================================
# DetailEnhancer: fixed multi-scale detail extraction (NO training)
# ============================================================
class DetailEnhancer(nn.Module):
    """Fixed detail enhancement module. No learnable parameters.
    Extracts multi-scale high-frequency details via unsharp masking.
    R(I_hat) = weighted sum of details at multiple scales.
    """
    def __init__(self, num_scales=3, base_sigma=1.0):
        super().__init__()
        self.num_scales = num_scales
        self.base_sigma = base_sigma
        # Pre-compute Gaussian kernels at multiple scales (non-parametrized)
        self.kernels = []
        for i in range(num_scales):
            sigma = base_sigma * (2 ** i)
            kernel_size = int(2 * math.ceil(3 * sigma) + 1)
            k = self._gaussian_kernel(kernel_size, sigma)
            self.register_buffer(f'kernel_{i}', k)
            self.kernels.append(f'kernel_{i}')
        # Scale weights (finer details weighted more)
        scale_weights = torch.tensor([0.6, 0.3, 0.1])
        self.register_buffer('scale_weights', scale_weights)
        # Final tanh to bound R in [-1, 1]

    @staticmethod
    def _gaussian_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        g = g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)
        return g.repeat(3, 1, 1, 1)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input image in [0, 1]
        Returns:
            R: [B, 3, H, W] detail residual in [-1, 1]
        """
        B, C, H, W = x.shape
        detail = torch.zeros_like(x)
        for i, kname in enumerate(self.kernels):
            kernel = getattr(self, kname)
            # Blur the image
            blurred = F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=C)
            # Unsharp mask: extract details
            d = x - blurred
            # Weight and accumulate
            detail = detail + self.scale_weights[i] * d
        # Normalize to reasonable range and apply tanh
        detail = detail / self.scale_weights.sum()
        return torch.tanh(detail * 3.0)  # scale before tanh for stronger effect


# ============================================================
# StateEncoder: encodes (I_hat, mask) -> state features
# ============================================================
class StateEncoder(nn.Module):
    """Encodes LaMa output + mask into PPO state features."""
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, i_hat, mask):
        x = torch.cat([i_hat, mask], dim=1)
        return self.fc(self.conv(x))


# ============================================================
# ActorCritic: PPO policy + value network
# ============================================================
class ActorCritic(nn.Module):
    """PPO policy: outputs w in [0,1] and V(s). Only this is trained."""
    def __init__(self, state_dim=256, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.actor_logstd = nn.Parameter(torch.tensor([0.0]))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        feat = self.shared(state)
        action_mean = self.actor_mean(feat)
        value = self.critic(feat)
        return action_mean, self.actor_logstd, value

    def get_action(self, state, deterministic=False):
        action_mean, logstd, value = self.forward(state)
        if deterministic:
            return action_mean, value
        std = logstd.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value, action_mean

    def evaluate(self, state, action):
        action_mean, logstd, value = self.forward(state)
        std = logstd.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


# ============================================================
# Discriminator: optional GAN discriminator (disabled by default)
# ============================================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Convenience builder
# ============================================================
def build_models(lama_path, device='cuda', state_dim=256):
    """Build all models. Only ActorCritic is trainable."""
    lama = LamaWrapper(lama_path, device).to(device)
    enhancer = DetailEnhancer().to(device)
    state_encoder = StateEncoder(feature_dim=state_dim).to(device)
    actor_critic = ActorCritic(state_dim=state_dim).to(device)
    return lama, enhancer, state_encoder, actor_critic