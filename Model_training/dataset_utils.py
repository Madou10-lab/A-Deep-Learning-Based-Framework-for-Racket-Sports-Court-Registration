import os.path as osp
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from albumentations.core.transforms_interface import (
    ImageOnlyTransform,

)

def one_hot_encode(label, n_classes):
    semantic_map = []
    for i in range(n_classes):
        colour = np.full((3), i)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

class BuildingsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            n_classes=2,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.n_classes = n_classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.mask_paths[i],cv2.IMREAD_GRAYSCALE)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.n_classes).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image= sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def get_image_filename(self, i):
        return osp.basename(self.image_paths[i])

    def __len__(self):
        # return length of
        return len(self.image_paths)

class ToHSV(ImageOnlyTransform):
    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    def get_transform_init_args_names(self):
        return ()

class ToBGR(ImageOnlyTransform):
    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def get_transform_init_args_names(self):
        return ()
