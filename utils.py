'''
utils.py - Utility functions: metrics tracking, visualization, checkpointing
No TensorBoard. Per-step tracking. Mask/repaired image saving.
'''

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image


class MetricsTracker:
    def __init__(self):
        self.steps = []
        self.rewards = []
        self.ssims = []
        self.psnrs = []
        self.lpipses = []
        self.edge_losses = []
        self.freq_losses = []
        self.sharpnesses = []
        self.mean_ws = []
        self.ppo_steps = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []

    def update(self, step, reward, ppo_losses, reward_detail, mean_w):
        self.steps.append(step)
        self.rewards.append(reward)
        self.ssims.append(reward_detail.get('ssim', 0))
        self.psnrs.append(reward_detail.get('psnr', 0))
        lp_val = reward_detail.get('lpips', 0)
        self.lpipses.append(lp_val.item() if isinstance(lp_val, torch.Tensor) else lp_val)
        self.edge_losses.append(reward_detail.get('edge', 0))
        self.freq_losses.append(reward_detail.get('freq', 0))
        self.sharpnesses.append(reward_detail.get('sharp', 0))
        self.mean_ws.append(mean_w)
        if ppo_losses:
            self.ppo_steps.append(step)
            self.policy_losses.append(ppo_losses.get('policy_loss', 0))
            self.value_losses.append(ppo_losses.get('value_loss', 0))
            self.entropies.append(ppo_losses.get('entropy', 0))

    def save_curves(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        steps = self.steps
        if not steps:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        if self.ppo_steps:
            axes[0].plot(self.ppo_steps, self.policy_losses, 'b-', linewidth=0.8)
            axes[0].set_title('Policy Loss'); axes[0].set_xlabel('Step'); axes[0].grid(True, alpha=0.3)
            axes[1].plot(self.ppo_steps, self.value_losses, 'r-', linewidth=0.8)
            axes[1].set_title('Value Loss'); axes[1].set_xlabel('Step'); axes[1].grid(True, alpha=0.3)
            axes[2].plot(self.ppo_steps, self.entropies, 'm-', linewidth=0.8)
            axes[2].set_title('Policy Entropy'); axes[2].set_xlabel('Step'); axes[2].grid(True, alpha=0.3)
        else:
            for a in axes:
                a.text(0.5, 0.5, 'No PPO updates yet', ha='center', va='center', transform=a.transAxes)
                a.set_xlabel('Step')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(steps, self.rewards, 'b-', linewidth=1.0, alpha=0.6, label='Reward')
        if len(steps) > 1:
            window = max(1, len(steps) // 50)
            smoothed = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            sm_steps = steps[len(steps)-len(smoothed):]
            ax1.plot(sm_steps, smoothed, 'b-', linewidth=2.0, label='Reward (smoothed)')
        ax1.set_xlabel('Step'); ax1.set_ylabel('Reward', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(steps, self.mean_ws, 'orange', linewidth=1.0, alpha=0.6, label='Mean w')
        ax2.set_ylabel('Mean w', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        fig.savefig(os.path.join(output_dir, 'reward_convergence.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(steps, self.ssims, 'g-', linewidth=0.8)
        axes[0].set_title('SSIM (higher better)'); axes[0].set_xlabel('Step'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(steps, self.psnrs, 'b-', linewidth=0.8)
        axes[1].set_title('PSNR (higher better)'); axes[1].set_xlabel('Step'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(steps, self.lpipses, 'r-', linewidth=0.8)
        axes[2].set_title('LPIPS (lower better)'); axes[2].set_xlabel('Step'); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'quality_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(steps, self.sharpnesses, 'c-', linewidth=0.8)
        axes[0].set_title('Sharpness (higher better)'); axes[0].set_xlabel('Step'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(steps, self.edge_losses, 'm-', linewidth=0.8)
        axes[1].set_title('Edge Loss (lower better)'); axes[1].set_xlabel('Step'); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'detail_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        history = {
            'steps': self.steps, 'rewards': self.rewards,
            'ssim': self.ssims, 'psnr': self.psnrs, 'lpips': self.lpipses,
            'edge_loss': self.edge_losses, 'freq_loss': self.freq_losses,
            'sharpness': self.sharpnesses, 'mean_w': self.mean_ws,
            'ppo_steps': self.ppo_steps, 'policy_loss': self.policy_losses,
            'value_loss': self.value_losses, 'entropy': self.entropies,
        }
        with open(os.path.join(output_dir, 'metrics_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


def save_masks_individual(masks, save_dir, prefix='mask'):
    """Save each mask as an individual PNG file."""
    os.makedirs(save_dir, exist_ok=True)
    B = masks.shape[0]
    for i in range(B):
        fname = os.path.join(save_dir, f'{prefix}_{i:04d}.png')
        save_image(masks[i], fname, normalize=True)


def save_repaired_individual(i_final, images, masks, save_dir, prefix='repaired'):
    """Save a horizontal 3-panel composite (original | masked | repaired) per sample."""
    os.makedirs(save_dir, exist_ok=True)
    B = images.shape[0]
    masked = images * (1.0 - masks)
    for i in range(B):
        comp = torch.cat([images[i], masked[i], i_final[i]], dim=-1)
        fname = os.path.join(save_dir, f'{prefix}_{i:04d}.png')
        save_image(comp, fname, normalize=True)


def save_checkpoint(actor_critic, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'epoch': epoch, 'actor_critic': actor_critic.state_dict(), 'optimizer': optimizer.state_dict()}, path)


def load_checkpoint(actor_critic, optimizer, path, device='cuda'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor_critic.load_state_dict(ckpt['actor_critic'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0)
