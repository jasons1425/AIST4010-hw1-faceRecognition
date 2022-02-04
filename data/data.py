from augment import augmentation_train, augmentation_test
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
TEST_FILES = glob.glob(os.path.join(DATA_DIR, 'test', '*'))


# augmentation settings
ORIGIN_SIZE, CROP_SIZE = 64, 32


# test dataset class
# since the test dataset does not have any class folder, we need to define our own dataset class
class FaceDataset(Dataset):
    def __init__(self, data, label=None, img_id=None, transform=None):
        super(FaceDataset, self).__init__()
        self.data = data
        self.label = label if label else [None] * len(data)
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
        return sample, self.label[idx], self.img_id[idx]

    def get_data(self, idx=None):
        if idx:
            return self.data[idx]
        return self.data


test_imgs = [None] * len(TEST_FILES)
test_img_ids = [None] * len(TEST_FILES)
for idx, img_fp in enumerate(TEST_FILES):
    img_id = re.match(r'^.*\\(\d+).jpg$', img_fp).group(1) + '.jpg'
    with Image.open(img_fp) as f:
        test_imgs[idx] = f.convert("RGB")
    test_img_ids[idx] = img_id


# the below objects are expected to be imported in model
# train dataset
train_ds = ImageFolder(TRAIN_FILES,
                       transform=augmentation_train(CROP_SIZE, ORIGIN_SIZE),
                       target_transform=torch.tensor)
# validation dataset
val_ds = ImageFolder(VAL_FILES,
                     transform=augmentation_test(CROP_SIZE, ORIGIN_SIZE),
                     target_transform=torch.tensor)
# test dataset
test_ds = FaceDataset(test_imgs,
                      img_id=test_img_ids,
                      transform=augmentation_test)


def get_loader(ds, batch_size, shuffle=True, **kwargs):
    if kwargs:
        kwargs['dataset'] = ds
        kwargs['batch_size'] = batch_size
        kwargs['shuffle'] = shuffle
        return DataLoader(**kwargs)
    return DataLoader(ds, batch_size, shuffle)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    demo_sample, demo_label = train_ds[0]
    plt.imshow(demo_sample.permute(1, 2, 0))
    plt.show()