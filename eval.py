'''
eval.py - Inference and visualization on validation/test set
'''

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from models import LamaWrapper, DetailEnhancer, StateEncoder, ActorCritic
from data_mask import InpaintingDataset, MaskGenerator
from rl import RewardComputer
from utils import load_checkpoint


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    lama = LamaWrapper(args.lama_path, device).to(device)
    enhancer = DetailEnhancer().to(device)
    state_encoder = StateEncoder(feature_dim=256).to(device)
    actor_critic = ActorCritic(state_dim=256).to(device)

    load_checkpoint(actor_critic, None, args.checkpoint, device)
    lama.eval()
    enhancer.eval()
    state_encoder.eval()
    actor_critic.eval()

    # Dataset
    dataset = InpaintingDataset(args.data_root, split=args.split, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    mask_gen = MaskGenerator(args.image_size)
    reward_computer = RewardComputer(use_perceptual=not args.no_perceptual, device=device).to(device)

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    repaired_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(repaired_dir, exist_ok=True)

    all_rewards, all_ssim, all_psnr, all_lpips, all_w = [], [], [], [], []
    sample_count = 0

    with torch.no_grad():
        for images in tqdm(loader, desc=f'Evaluating ({args.split})'):
            images = images.to(device)
            B = images.shape[0]
            masks = mask_gen(images).to(device)
            i_hat = lama(images * (1.0 - masks), masks)
            states = state_encoder(i_hat, masks)
            actions, _ = actor_critic.get_action(states, deterministic=True)
            w = actions.view(B, 1, 1, 1)
            R = enhancer(i_hat)
            i_final = torch.clamp(i_hat + w * R, 0.0, 1.0)

            for b in range(B):
                r, detail = reward_computer(i_final[b:b+1], images[b:b+1])
                all_rewards.append(r.item())
                all_ssim.append(detail['ssim'])
                all_psnr.append(detail['psnr'])
                lp_val = detail['lpips']
                all_lpips.append(lp_val.item() if isinstance(lp_val, torch.Tensor) else lp_val)
                all_w.append(w[b].item())

                if sample_count < args.max_samples:
                    comp = torch.cat([images[b], images[b]*(1.0-masks[b]), i_final[b]], dim=-1)
                    fname = os.path.join(repaired_dir, f'sample_{sample_count:04d}.png')
                    save_image(comp, fname, normalize=True)
                    sample_count += 1

    # Summary
    print(f'\n===== Evaluation ({args.split}) =====')
    print(f'Samples: {len(all_rewards)}')
    print(f'Reward: {np.mean(all_rewards):.4f} +/- {np.std(all_rewards):.4f}')
    print(f'SSIM:   {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}')
    print(f'PSNR:   {np.mean(all_psnr):.4f} +/- {np.std(all_psnr):.4f}')
    print(f'LPIPS:  {np.mean(all_lpips):.4f} +/- {np.std(all_lpips):.4f}')
    print(f'w:      {np.mean(all_w):.4f} +/- {np.std(all_w):.4f}')

    # w distribution histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_w, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_xlabel('w (refinement strength)'); ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Refinement Strength w ({args.split} set)')
    ax.axvline(np.mean(all_w), color='red', linestyle='--', label=f'Mean = {np.mean(all_w):.3f}')
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, 'w_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Results: {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model')
    parser.add_argument('--data_root', type=str, default=r'D:\Python All\Pytorch Study\image_inpainting')
    parser.add_argument('--lama_path', type=str, default=r'D:\Python All\Pytorch Study\image_inpainting\anime-manga-big-lama.pt')
    parser.add_argument('--checkpoint', type=str, default=r'D:\Python All\Pytorch Study\image_inpainting\outputs\models\checkpoint_best.pt')
    parser.add_argument('--output_dir', type=str, default=r'D:\Python All\Pytorch Study\image_inpainting\outputs\repaired')
    parser.add_argument('--split', type=str, default='val', choices=['train','val','test'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument('--no_perceptual', action='store_true')
    args = parser.parse_args()
    evaluate(args)