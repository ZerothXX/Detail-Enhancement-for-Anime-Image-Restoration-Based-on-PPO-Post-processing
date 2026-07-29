'''
test.py - Single-image test; saved to project root folder by default.
'''

import os
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from models import LamaWrapper, DetailEnhancer, StateEncoder, ActorCritic
from utils import load_checkpoint


def test_single(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    lama = LamaWrapper(args.lama_path, device).to(device)
    enhancer = DetailEnhancer().to(device)
    state_encoder = StateEncoder(feature_dim=256).to(device)
    actor_critic = ActorCritic(state_dim=256).to(device)
    load_checkpoint(actor_critic, None, args.checkpoint, device)
    for m in [lama, enhancer, state_encoder, actor_critic]:
        m.eval()

    image = Image.open(args.input).convert('RGB')
    image = TF.resize(image, [args.image_size, args.image_size], interpolation=TF.InterpolationMode.BICUBIC)
    image_t = TF.to_tensor(image).unsqueeze(0).to(device)

    if args.mask_path:
        mask = Image.open(args.mask_path).convert('L')
        mask = TF.resize(mask, [args.image_size, args.image_size], interpolation=TF.InterpolationMode.NEAREST)
        mask = (TF.to_tensor(mask) > 0.5).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        B, C, H, W = image_t.shape
        mask = torch.zeros(1, 1, H, W, device=device)
        mh, mw = H // 5, W // 5
        mask[0, 0, (H-mh)//2:(H+mh)//2, (W-mw)//2:(W+mw)//2] = 1.0
        print(f'No mask, using default rectangle ({mw}x{mh}).')

    with torch.no_grad():
        masked = image_t * (1.0 - mask)
        i_hat = lama(masked, mask)
        state = state_encoder(i_hat, mask)
        action, _, _, _ = actor_critic.get_action(state, deterministic=True)
        i_final = torch.clamp(i_hat + action.view(1,1,1,1) * enhancer(i_hat), 0.0, 1.0)

    print(f'PPO w = {action.item():.4f}')
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    save_image(masked,  os.path.join(args.output_dir, f'{base}_masked.png'))
    save_image(i_hat,   os.path.join(args.output_dir, f'{base}_lama.png'))
    save_image(i_final, os.path.join(args.output_dir, f'{base}_repaired.png'))
    print(f'Outputs -> {args.output_dir}/')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--mask', dest='mask_path', default=None)
    p.add_argument('--lama_path', default=r'D:\Python All\Pytorch Study\image_inpainting\anime-manga-big-lama.pt')
    p.add_argument('--checkpoint', default=r'D:\Python All\Pytorch Study\image_inpainting\outputs\models\checkpoint_best.pt')
    p.add_argument('--output_dir', default=r'D:\Python All\Pytorch Study\image_inpainting')
    p.add_argument('--image_size', type=int, default=256)
    args = p.parse_args()
    test_single(args)
