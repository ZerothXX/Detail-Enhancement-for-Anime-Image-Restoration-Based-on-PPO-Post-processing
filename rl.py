'''
rl.py - PPO algorithm, replay buffer, rollout, losses, reward computation
Device-safe implementation: all loss functions create kernels on input device.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features with robust .to() handling."""
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        slices = []
        sl = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16, 'relu4_3': 23}
        prev = 0
        for name, idx in sl.items():
            slices.append(vgg[prev:idx])
            prev = idx
        self.slices = nn.ModuleList(slices)
        self.to(device)
        for s in self.slices:
            for p in s.parameters():
                p.requires_grad_(False)
        self.eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x, y = pred, target
        for s in self.slices:
            x = s(x)
            y = s(y)
            loss += F.l1_loss(x, y)
        return loss / len(self.slices)


def gaussian_kernel(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g_1d = g.view(1, -1)
    g = g_1d.t() * g_1d
    return g.view(1, 1, size, size).repeat(channels, 1, 1, 1)


def ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    d = pred.device
    k = gaussian_kernel(window_size, 1.5, pred.shape[1], d)
    mu1 = F.conv2d(pred, k, padding=window_size//2, groups=pred.shape[1])
    mu2 = F.conv2d(target, k, padding=window_size//2, groups=target.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
    s1 = F.conv2d(pred*pred, k, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
    s2 = F.conv2d(target*target, k, padding=window_size//2, groups=target.shape[1]) - mu2_sq
    s12 = F.conv2d(pred*target, k, padding=window_size//2, groups=target.shape[1]) - mu1_mu2
    m = ((2*mu1_mu2+C1)*(2*s12+C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return m.mean()


def psnr(pred, target, max_val=1.0):
    return 20 * torch.log10(max_val / torch.sqrt(F.mse_loss(pred, target) + 1e-8))


def edge_loss(pred, target):
    d = pred.device
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=d).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=d).view(1,1,3,3)
    def sobel(x):
        gx = F.conv2d(x, kx.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
        gy = F.conv2d(x, ky.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
        return torch.sqrt(gx**2 + gy**2 + 1e-6)
    return F.l1_loss(sobel(pred), sobel(target))


def frequency_loss(pred, target):
    return F.l1_loss(torch.abs(torch.fft.fft2(pred, norm='ortho')),
                     torch.abs(torch.fft.fft2(target, norm='ortho')))


def sharpness_score(img):
    d = img.device
    k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=d).view(1,1,3,3)
    lap = F.conv2d(img, k.repeat(img.shape[1],1,1,1), padding=1, groups=img.shape[1])
    return lap.var(dim=[1,2,3]).mean()


class RewardComputer(nn.Module):
    def __init__(self, use_perceptual=True, device='cuda'):
        super().__init__()
        self.perceptual = PerceptualLoss(device) if use_perceptual else None

    @torch.no_grad()
    def forward(self, i_final, i_gt):
        r_ssim = ssim(i_final, i_gt)
        r_psnr = psnr(i_final, i_gt) / 40.0
        if self.perceptual is not None:
            r_lpips = self.perceptual(i_final, i_gt)
        else:
            r_lpips = torch.tensor(0.0, device=i_final.device)
        r_edge = edge_loss(i_final, i_gt)
        r_freq = frequency_loss(i_final, i_gt)
        r_sharp = sharpness_score(i_final) / 10.0
        reward = 1.0*r_ssim + 0.5*r_psnr - 1.2*r_lpips - 0.5*r_edge - 0.5*r_freq + 0.8*r_sharp
        return reward, {
            'ssim': r_ssim.item(), 'psnr': r_psnr.item(), 'lpips': r_lpips.item(),
            'edge': r_edge.item(), 'freq': r_freq.item(), 'sharp': r_sharp.item(),
        }


class ReplayBuffer:
    """Stores trajectories. Moves inputs to buffer device to prevent mismatch."""
    def __init__(self, buffer_size=2048, state_dim=256, device='cuda'):
        self.buffer_size = buffer_size
        self.device = device
        self.states = torch.zeros(buffer_size, state_dim, device=device)
        self.actions = torch.zeros(buffer_size, 1, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        self.ptr = 0
        self.full = False

    def add(self, state, action, log_prob, reward, value, done=False):
        idx = self.ptr
        # detach() ensures no grad_fn leaks; assignment broadcasts scalars
        self.states[idx] = state.to(self.device).detach()
        self.actions[idx] = action.to(self.device).detach()
        self.log_probs[idx] = log_prob.to(self.device).detach()
        self.rewards[idx] = reward.to(self.device).detach()
        self.dones[idx] = float(done)
        # value is guaranteed non-None and same device
        if value is not None:
            self.values[idx] = value.to(self.device).detach()
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def get(self):
        n = self.buffer_size if self.full else self.ptr
        if n == 0:
            return None
        return (self.states[:n], self.actions[:n], self.log_probs[:n],
                self.rewards[:n], self.values[:n], self.dones[:n])

    def ready(self):
        return self.full or self.ptr >= 256

    def reset(self):
        self.ptr = 0
        self.full = False


class PPO:
    def __init__(self, actor_critic, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, c1=0.5, c2=0.01, epochs=10, batch_size=64):
        self.actor_critic = actor_critic
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, dones):
        device = rewards.device
        adv = torch.zeros_like(rewards, device=device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nv = 0.0 if dones[t] else (values[t+1] if t+1 < len(values) else torch.tensor(0.0, device=device))
            delta = rewards[t] + self.gamma * nv - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            adv[t] = gae
        ret = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, buffer):
        data = buffer.get()
        if data is None:
            return {}
        s, a, olp, r, v, d = data
        adv, ret = self.compute_gae(r, v, d)
        pl, vl, en, nu = 0.0, 0.0, 0.0, 0
        n = len(s)
        idx = torch.randperm(n, device=s.device)
        for _ in range(self.epochs):
            for st in range(0, n, self.batch_size):
                bi = idx[st:st+self.batch_size]
                new_lp, ent, new_v = self.actor_critic.evaluate(s[bi], a[bi])
                ratio = torch.exp(new_lp - olp[bi])
                surr1 = ratio * adv[bi]
                surr2 = torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * adv[bi]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(new_v.squeeze(-1), ret[bi])
                loss = p_loss + self.c1*v_loss - self.c2*ent.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                pl += p_loss.item()
                vl += v_loss.item()
                en += ent.mean().item()
                nu += 1
        buffer.reset()
        return {'policy_loss': pl/max(nu,1), 'value_loss': vl/max(nu,1), 'entropy': en/max(nu,1)}


class Rollout:
    def __init__(self, lama, enhancer, state_encoder, actor_critic, reward_computer, buffer, device='cuda'):
        self.lama = lama
        self.enhancer = enhancer
        self.state_encoder = state_encoder
        self.actor_critic = actor_critic
        self.reward_computer = reward_computer
        self.buffer = buffer
        self.device = device

    def step(self, images, masks):
        B = images.shape[0]
        masked = images * (1.0 - masks)
        i_hat = self.lama(masked, masks)
        states = self.state_encoder(i_hat, masks)
        actions, log_probs, values, means = self.actor_critic.get_action(states)
        i_final = torch.clamp(i_hat + actions.view(B,1,1,1) * self.enhancer(i_hat), 0.0, 1.0)
        rl, det = [], {'ssim':[],'psnr':[],'lpips':[],'edge':[],'freq':[],'sharp':[]}
        for b in range(B):
            r, d = self.reward_computer(i_final[b:b+1], images[b:b+1])
            rl.append(r)
            for k in det:
                det[k].append(d[k])
        rewards = torch.stack(rl).to(self.device)
        for b in range(B):
            self.buffer.add(states[b], actions[b], log_probs[b], rewards[b], values[b], done=True)
        return i_final, rewards, {k: np.mean(v) for k, v in det.items()}
