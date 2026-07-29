'''
train.py - Main PPO training loop. Only ActorCritic trained.
Per-step metrics, per-epoch curves, mask/repaired image saving.
'''

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_models
from data_mask import InpaintingDataset, MaskGenerator
from rl import PPO, ReplayBuffer, Rollout, RewardComputer
from utils import MetricsTracker, save_checkpoint, save_masks_individual, save_repaired_individual


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Models: {args.model_dir}')
    print(f'Curves: {args.curves_dir}')
    print(f'Masks:  {args.masks_dir}')
    print(f'Repaired: {args.repaired_dir}')

    lama, enhancer, state_encoder, actor_critic = build_models(args.lama_path, device, args.state_dim)
    for m in [lama, enhancer]:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    for p in state_encoder.parameters():
        p.requires_grad_(False)
    state_encoder.eval()
    actor_critic.train()
    trainable = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
    frozen = sum(p.numel() for m in [lama, enhancer, state_encoder] for p in m.parameters())
    print(f'Trainable: {trainable:,}, Frozen: {frozen:,}')

    train_ds = InpaintingDataset(args.data_root, 'train', args.image_size)
    val_ds = InpaintingDataset(args.data_root, 'val', args.image_size)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    mask_gen = MaskGenerator(args.image_size)

    reward_comp = RewardComputer(use_perceptual=not args.no_perceptual, device=device).to(device)
    buffer = ReplayBuffer(args.buffer_size, args.state_dim, device)
    ppo = PPO(actor_critic, args.lr_ac, args.gamma, args.gae_lambda, args.clip_eps, 0.5, 0.01, args.ppo_epochs, args.ppo_batch)
    rollout = Rollout(lama, enhancer, state_encoder, actor_critic, reward_comp, buffer, device)

    for d in [args.model_dir, args.curves_dir, args.masks_dir, args.repaired_dir]:
        os.makedirs(d, exist_ok=True)

    tracker = MetricsTracker()
    best_reward = -float('inf')
    global_step = 0
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Epochs: {args.epochs}')

    for epoch in range(1, args.epochs + 1):
        actor_critic.train()
        epoch_rewards, epoch_ws = [], []
        rda = {'ssim': 0.0, 'psnr': 0.0, 'lpips': 0.0, 'edge': 0.0, 'freq': 0.0, 'sharp': 0.0}
        n_batches = 0

        bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        first_batch = True
        for images in bar:
            images = images.to(device)
            masks = mask_gen(images).to(device)
            with torch.no_grad():
                i_final, rewards, r_detail = rollout.step(images, masks)
            ppo_losses = {}
            if buffer.ready():
                ppo_losses = ppo.update(buffer)
            global_step += 1
            mean_w = rollout.actor_critic.get_action(
                rollout.state_encoder(rollout.lama(images*(1.0-masks), masks), masks),
                deterministic=True)[0].mean().item()
            tracker.update(global_step, rewards.mean().item(), ppo_losses, r_detail, mean_w)

            epoch_rewards.append(rewards.mean().item())
            epoch_ws.append(mean_w)
            for k in rda:
                rda[k] += r_detail.get(k, 0)
            n_batches += 1
            bar.set_postfix({'R': f'{rewards.mean().item():.3f}', 'w': f'{mean_w:.3f}'})

            if first_batch and epoch % args.save_image_every == 0:
                first_batch = False
                save_masks_individual(masks, args.masks_dir, prefix=f'epoch_{epoch}')
                save_repaired_individual(i_final, images, masks, args.repaired_dir, prefix=f'epoch_{epoch}')

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        avg_w = sum(epoch_ws) / len(epoch_ws)
        for k in rda:
            rda[k] /= max(n_batches, 1)
        print(f'Epoch {epoch:3d} | R={avg_reward:+.4f}  w={avg_w:.4f}  SSIM={rda["ssim"]:.4f}  PSNR={rda["psnr"]:.2f}')

        tracker.save_curves(args.curves_dir)

        if avg_reward > best_reward:
            best_reward = avg_reward
            save_checkpoint(actor_critic, ppo.optimizer, epoch, os.path.join(args.model_dir, 'checkpoint_best.pt'))
            print(f'  >> Best model (R={best_reward:+.4f})')

        if epoch % args.val_every == 0:
            val_r = validate(val_loader, lama, enhancer, state_encoder, actor_critic, reward_comp, mask_gen, device)
            print(f'  Val R={val_r:+.4f}')

    save_checkpoint(actor_critic, ppo.optimizer, args.epochs, os.path.join(args.model_dir, 'checkpoint_final.pt'))
    print(f'Done. Best R={best_reward:+.4f}')


def validate(loader, lama, enhancer, state_encoder, actor_critic, reward_comp, mask_gen, device):
    actor_critic.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            B = images.shape[0]
            masks = mask_gen(images).to(device)
            i_hat = lama(images * (1.0 - masks), masks)
            states = state_encoder(i_hat, masks)
            actions, _ = actor_critic.get_action(states, deterministic=True)
            i_final = torch.clamp(i_hat + actions.view(B,1,1,1) * enhancer(i_hat), 0.0, 1.0)
            for b in range(B):
                r, _ = reward_comp(i_final[b:b+1], images[b:b+1])
                total += r.item()
                n += 1
    actor_critic.train()
    return total / max(n, 1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=r'D:\Python All\Pytorch Study\image_inpainting')
    p.add_argument('--lama_path', default=r'D:\Python All\Pytorch Study\image_inpainting\anime-manga-big-lama.pt')
    p.add_argument('--model_dir', default=r'D:\Python All\Pytorch Study\image_inpainting\outputs\models')
    p.add_argument('--curves_dir', default=r'D:\Python All\Pytorch Study\image_inpainting\outputs\curves')
    p.add_argument('--masks_dir', default=r'D:\Python All\Pyt  orch Study\image_inpainting\outputs\masks')
    p.add_argument('--repaired_dir', default=r'D:\Python All\Pytorch Study\image_inpainting\outputs\repaired')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--state_dim', type=int, default=256)
    p.add_argument('--buffer_size', type=int, default=4096)
    p.add_argument('--lr_ac', type=float, default=2e-5)
    p.add_argument('--gamma', type=float, default=0.9)
    p.add_argument('--gae_lambda', type=float, default=0.95)
    p.add_argument('--clip_eps', type=float, default=0.15)
    p.add_argument('--ppo_epochs', type=int, default=5)
    p.add_argument('--ppo_batch', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--val_every', type=int, default=5)
    p.add_argument('--save_image_every', type=int, default=1)
    p.add_argument('--no_perceptual', action='store_true')
    args = p.parse_args()
    train(args)
