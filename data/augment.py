from data.preprocess import CorrectionFilterEqualize
from torchvision import transforms


# Augmentation for training set
#   (preprocessing) -> horizontal flip -> random crop (and resize) -> color jitter
def augmentation_train(crop_size=32, resize=0, preprocess=True,
                       preprocess_arg=None, no_augment=False, normalized=False):
    augment_stack = []
    if preprocess:
        if preprocess_arg:
            augment_stack.append(CorrectionFilterEqualize(**preprocess_arg))
        else:
            augment_stack.append(CorrectionFilterEqualize())
    if not no_augment:
        augment_stack.append(transforms.RandomHorizontalFlip())
        augment_stack.append(transforms.RandomCrop(crop_size))
        if resize:
            augment_stack.append(transforms.Resize(resize))
        augment_stack.append(transforms.ColorJitter())
    elif resize:
        augment_stack.append(transforms.Resize(resize))
    augment_stack.append(transforms.ToTensor())
    if normalized:
        augment_stack.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    augment_stack.append(transforms.RandomErasing(0.5, [0.25, 0.5], [0.5, 1.0]))

    return transforms.Compose(augment_stack)


# Resize function that takes in five-cropped image list
class ResizeBatch(object):
    def __init__(self, size, normalized=False):
        trans_stack = [transforms.Resize(size), transforms.ToTensor()]
        if normalized:
            trans_stack.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.trans_stack = transforms.Compose(trans_stack)

    def __call__(self, imgs):
        if isinstance(imgs, tuple) or isinstance(imgs, list):
            return [self.trans_stack(img) for img in imgs]
        return self.trans_stack(imgs)

    def __repr__(self):
        return "Resizing a batch of images and converting them to tensor"


# Augmentation for test/validation set
#   (preprocessing) -> FiveCrop (and resize)
def augmentation_test(crop_size, resize=0, preprocess=True,
                      preprocess_arg=None, no_augment=False, normalized=False):
    augment_stack = []
    if preprocess:
        if preprocess_arg:
            augment_stack.append(CorrectionFilterEqualize(**preprocess_arg))
        else:
            augment_stack.append(CorrectionFilterEqualize())
    if not no_augment:
        augment_stack.append(transforms.FiveCrop(crop_size))
        if resize:  # special transformation for five_cropped function
            augment_stack.append(ResizeBatch(resize, normalized=normalized))
        else:
            augment_stack.append(ResizeBatch(crop_size, normalized=normalized))
    else:
        if resize:
            augment_stack.append(transforms.Resize(resize))
        augment_stack.append(transforms.ToTensor())
        if normalized:
            augment_stack.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(augment_stack)
