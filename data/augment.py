from data.preprocess import CorrectionFilterEqualize
import numpy as np
from torchvision import transforms
from PIL import Image
import torch


# Augmentation for training set
#   (preprocessing) -> horizontal flip -> random crop (and resize) -> color jitter
def augmentation_train(crop_size, resize=0, preprocess=True, preprocess_arg=None):
    augment_stack = []
    if preprocess:
        if preprocess_arg:
            augment_stack.append(CorrectionFilterEqualize(**preprocess_arg))
        else:
            augment_stack.append(CorrectionFilterEqualize())
    augment_stack.append(transforms.RandomHorizontalFlip())
    augment_stack.append(transforms.RandomCrop(crop_size))
    if resize:
        augment_stack.append(transforms.Resize(resize))
    augment_stack.append(transforms.ColorJitter())
    augment_stack.append(transforms.ToTensor())

    return transforms.Compose(augment_stack)


# Augmentation for test/validation set
#   (preprocessing) -> FiveCrop (and resize)
def augmentation_test(crop_size, resize=0, preprocess=True, preprocess_arg=None):
    augment_stack = []
    if preprocess:
        if preprocess_arg:
            augment_stack.append(CorrectionFilterEqualize(**preprocess_arg))
        else:
            augment_stack.append(CorrectionFilterEqualize())
    augment_stack.append(transforms.FiveCrop(crop_size))
    if resize:
        class ResizeBatch(object):
            def __init__(self, size):
                self.trans_stack = transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor()
                ])

            def __call__(self, imgs):
                if isinstance(imgs, tuple) or isinstance(imgs, list):
                    return [self.trans_stack(img) for img in imgs]
                return self.trans_stack(imgs)

            def __repr__(self):
                return "Resizing a batch of images and converting them to tensor"
        augment_stack.append(ResizeBatch(resize))
    else:
        augment_stack.append(transforms.ToTensor())

    return transforms.Compose(augment_stack)