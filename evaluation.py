from data.data import get_ds, get_loader
from data.augment import augmentation_test
from helper.eval import evaluation
from model.vgg_face import VGGFaceResNet
import pandas as pd
import torch


# Please define the test dataset path here!!! It should have the same format as the distributed dataset folder
# e.g. Directory - "D:Datasets/faceRecognition/test" ; Images - "D:Datasets/faceRecognition/test/1.jpg",...
DATA_PATH = r'D:\Documents\datasets\AIST4010\face recognition\test'
assert DATA_PATH is not None, "Please fill in the test dataset path!!!"


# load the dataset
BATCH_SIZE = 32
IMG_RESIZE = 224
CROP_SIZE = 48
test_ds = get_ds('test', transformation=augmentation_test(CROP_SIZE, IMG_RESIZE, preprocess=False),
                 test_imgs_fp=DATA_PATH)
test_loader = get_loader(test_ds, BATCH_SIZE, shuffle=False)

# prepare the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VGGFaceResNet(3, 1000).half().to(device)
model.load_state_dict(torch.load(r"trials/vggface-resnet-7610.pth"))

# make evaluation
preds = pd.DataFrame({'id': [], 'label': []})
for imgs, labels in test_loader:
    imgs = [img.half().to(device) for img in imgs]
    batch_preds = evaluation(model, imgs, weights_ratio=[0.1, 0.1, 0.1, 0.1, 0.6])
    preds = pd.concat([preds, pd.DataFrame({'id': labels, 'label': batch_preds})], ignore_index=True)


preds = preds.astype({'label': 'int32'})
preds.to_csv("preds.csv", index=False)





