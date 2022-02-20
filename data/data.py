from data.augment import augmentation_train, augmentation_test
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import glob
import os
import re


# data locations
DATA_DIR = r'D:\Documents\datasets\AIST4010\face recognition'
TRAIN_FILES = os.path.join(DATA_DIR, 'train')
VAL_FILES = os.path.join(DATA_DIR, 'val')
TEST_FILES = os.path.join(DATA_DIR, 'test', '*')


# augmentation settings
ORIGIN_SIZE, CROP_SIZE = 64, 32


# test dataset class
# since the test dataset does not have any class folder, we need to define our own dataset class
class FaceDataset(Dataset):
    def __init__(self, data, img_id=None, transform=None):
        super(FaceDataset, self).__init__()
        self.data = data
        self.img_id = img_id if img_id else [None] * len(data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(self.data[idx])
        return sample, self.img_id[idx]

    def get_data(self, idx=None):
        if idx:
            return self.data[idx]
        return self.data


# load the test images one by one
def load_imgs(fp=TEST_FILES):
    fp = glob.glob(os.path.join(fp, '*'))
    rematch_pattern = r'^.*' + os.sep.replace('\\', '\\\\') + r'(\d+).jpg$'
    fp.sort(key=lambda fp: int(re.match(rematch_pattern, fp).group(1)))
    test_imgs = [None] * len(fp)
    test_img_ids = [None] * len(fp)
    for idx, img_fp in enumerate(fp):
        img_id = re.match(rematch_pattern, img_fp).group(1) + '.jpg'
        with Image.open(img_fp) as f:
            test_imgs[idx] = f.convert("RGB")
        test_img_ids[idx] = img_id
    return test_imgs, test_img_ids


test_imgs, test_img_ids = load_imgs(TEST_FILES)


# the below objects are expected to be imported in model
# train dataset
def get_ds(phase, transformation=None, test_imgs_fp=None):
    if transformation is None:
        transformation = augmentation_train if phase == "train" else augmentation_test
        transformation = transformation(CROP_SIZE, ORIGIN_SIZE)
    if phase == 'train':
        train_ds = ImageFolder(TRAIN_FILES,
                               transform=transformation,
                               target_transform=torch.tensor)
        class_to_idx_dict = train_ds.class_to_idx
        train_ds.class_to_idx = dict([(str(i), i) for i in range(len(class_to_idx_dict))])
        return train_ds
    if phase == 'val':
        val_ds = ImageFolder(VAL_FILES,
                             transform=transformation,
                             target_transform=torch.tensor)
        class_to_idx_dict = val_ds.class_to_idx
        val_ds.class_to_idx = dict([(str(i), i) for i in range(len(class_to_idx_dict))])
        return val_ds
    if phase == 'test':
        if test_imgs_fp:
            imgs, img_ids = load_imgs(test_imgs_fp)
        else:
            imgs = test_imgs
            img_ids = test_img_ids
        test_ds = FaceDataset(imgs,
                              img_id=img_ids,
                              transform=transformation)
        return test_ds
    raise NotImplementedError('Unknown phase!')


def get_loader(ds, batch_size, shuffle=True, **kwargs):
    if kwargs:
        kwargs['dataset'] = ds
        kwargs['batch_size'] = batch_size
        kwargs['shuffle'] = shuffle
        return DataLoader(**kwargs)
    return DataLoader(ds, batch_size, shuffle)
