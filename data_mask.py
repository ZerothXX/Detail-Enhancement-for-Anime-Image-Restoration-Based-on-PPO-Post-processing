'''
data_mask.py - Dataset loader + complexity-based mask generator
'''

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2


class InpaintingDataset(Dataset):
    """Danbooru-style image loader."""
    def __init__(self, root, split='train', image_size=256):
        self.root = root
        self.image_size = image_size
        list_file = os.path.join(root, 'dataset', f'{split}.txt')
        with open(list_file, 'r') as f:
            self.paths = [line.strip() for line in f if line.strip()]
        self.mask_generator = MaskGenerator(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'dataset', self.paths[idx])
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        # Resize keeping aspect ratio: longer side = image_size
        scale = self.image_size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC)
        image = TF.to_tensor(image)
        # Pad to square (image_size x image_size) for consistent batching
        pad_bottom = self.image_size - new_h
        pad_right = self.image_size - new_w
        if pad_bottom > 0 or pad_right > 0:
            image = TF.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
        return image


class MaskGenerator:
    """Complexity-based mask generator. Higher complexity -> smaller mask ratio."""
    def __init__(self, image_size=256):
        self.image_size = image_size

    def _compute_complexity(self, image_tensor):
        img = image_tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        complexity = mag.mean() / 255.0
        complexity = np.clip(complexity, 0.0, 1.0)
        return float(complexity)

    def _complexity_to_ratio(self, complexity):
        return 0.11 - complexity * 0.10

    def __call__(self, image_tensor):
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        B, C, H, W = image_tensor.shape
        masks = []
        for b in range(B):
            complexity = self._compute_complexity(image_tensor[b])
            ratio = self._complexity_to_ratio(complexity)
            mask = self._generate_mask(H, W, ratio)
            masks.append(torch.from_numpy(mask).float().unsqueeze(0))
        return torch.stack(masks, dim=0)

    def _generate_mask(self, H, W, ratio):
        mask_area = max(4, int(H * W * ratio))
        if ratio >= 0.05:
            return self._generate_strip_mask(H, W, mask_area)
        else:
            shape_type = random.choice(['rectangle', 'circle', 'ellipse'])
            if shape_type == 'rectangle':
                return self._generate_rect_mask(H, W, mask_area)
            elif shape_type == 'circle':
                return self._generate_circle_mask(H, W, mask_area)
            else:
                return self._generate_ellipse_mask(H, W, mask_area)

    def _generate_strip_mask(self, H, W, area):
        """Generate a long strip mask with aspect ratio 5:1 ~ 10:1 (for 7%%~11%% area)."""
        mask = np.zeros((H, W), dtype=np.uint8)
        aspect = random.uniform(5.0, 10.0)
        is_horizontal = random.random() < 0.5
        if is_horizontal:
            h = max(2, min(int(np.sqrt(area / aspect)), H - 1))
            w = max(2, min(int(area / h), W - 1))
        else:
            w = max(2, min(int(np.sqrt(area / aspect)), W - 1))
            h = max(2, min(int(area / w), H - 1))
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        mask[y:y + h, x:x + w] = 1
        return mask

    def _generate_rect_mask(self, H, W, area):
        mask = np.zeros((H, W), dtype=np.uint8)
        aspect = random.uniform(0.5, 2.0)
        h = min(int(np.sqrt(area * aspect)), H - 1)
        w = min(int(area / h), W - 1)
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        mask[y:y + h, x:x + w] = 1
        return mask

    def _generate_circle_mask(self, H, W, area):
        mask = np.zeros((H, W), dtype=np.uint8)
        radius = max(2, min(int(np.sqrt(area / np.pi)), H // 3, W // 3))
        cy = random.randint(radius, H - radius - 1)
        cx = random.randint(radius, W - radius - 1)
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask[dist <= radius] = 1
        return mask

    def _generate_ellipse_mask(self, H, W, area):
        mask = np.zeros((H, W), dtype=np.uint8)
        aspect = random.uniform(0.5, 2.0)
        b = max(2, min(int(np.sqrt(area / (np.pi * aspect))), H // 3))
        a = max(2, min(int(b * aspect), W // 3))
        cy = random.randint(b, H - b - 1)
        cx = random.randint(a, W - a - 1)
        yy, xx = np.ogrid[:H, :W]
        dist = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2
        mask[dist <= 1.0] = 1
        return mask
