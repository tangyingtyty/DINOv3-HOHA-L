import os
import cv2
import albumentations as A
import numpy as np
import random

import torch
from torch.utils.data import Dataset

def loveda_transform_mask(mask):
    mask[mask == 0] = 8
    mask = mask - 1
    return mask


class SegDataset(Dataset):
    def __init__(self, data_dir,  is_train=True, mosaic_prob=0.25):
        self.img_dir = os.path.join(data_dir, 'image')
        self.mask_dir = os.path.join(data_dir, 'label')
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
            # A.Rotate(limit=30, p=0.5),
            ])

        self.images = os.listdir(self.img_dir)
        self.is_train = is_train
        self.mosaic_prob = mosaic_prob

    def __len__(self):
        return len(self.images)

    def load_sample(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = cv2.imread(img_path)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        return image, mask
   
    def mosaic_augmentation(self, index):
        other_indices = random.choices([i for i in range(len(self)) if i != index], k=3)
        indexes = [index] + other_indices
        random.shuffle(indexes)

        samples = [self.load_sample(idx) for idx in indexes]
        images, masks = zip(*samples)

        h, w = images[0].shape[:2]
        
        split_x = random.randint(w // 4, w * 3 // 4)
        split_y = random.randint(h // 4, h * 3 // 4)

        regions = [
            {'crop': (split_x, split_y), 'pos': (0, split_y, 0, split_x)},           # 左上
            {'crop': (w - split_x, split_y), 'pos': (0, split_y, split_x, w)},       # 右上
            {'crop': (split_x, h - split_y), 'pos': (split_y, h, 0, split_x)},       # 左下
            {'crop': (w - split_x, h - split_y), 'pos': (split_y, h, split_x, w)}    # 右下
        ]
        
        mosaic_image = np.zeros((h, w, 3), dtype=np.uint8)
        mosaic_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i, region in enumerate(regions):
            crop_w, crop_h = region['crop']
            y1, y2, x1, x2 = region['pos']
            
            random_crop = A.RandomCrop(width=crop_w, height=crop_h)
            cropped = random_crop(image=images[i], mask=masks[i])
            
            mosaic_image[y1:y2, x1:x2] = cropped['image']
            mosaic_mask[y1:y2, x1:x2] = cropped['mask']

        # cv2.imwrite('mosaic_image.png', mosaic_image)
        # cv2.imwrite('mosaic_mask.png', mosaic_mask)

        return mosaic_image, mosaic_mask


    def __getitem__(self, index):
        if self.is_train and random.random() < self.mosaic_prob:
            image, mask = self.mosaic_augmentation(index)
        else:
            image, mask = self.load_sample(index)

        # mask = loveda_transform_mask(mask)    

        if self.is_train:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        image = torch.as_tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.as_tensor(mask)
        return {'name': self.images[index], 'image': image, 'label': mask}
    
if __name__== "__main__" :
    train_data = SegDataset('data/LoveDA/train')
    print(len(train_data))