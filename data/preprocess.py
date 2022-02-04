import numpy as np
from torchvision import transforms
from skimage.filters import difference_of_gaussians
from PIL import Image
import torch


# Gamma Correction, Difference Of Gaussian Filtering, Contrast Equalization
# ref: Preprocessing technique for face recognition applications under varying illumination conditions
class CorrectionFilterEqualize(object):
    def __init__(self, gamma=0.2, low_sigma=1, high_sigma=2):
        self.gamma = gamma
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma

    def __call__(self, img):
        # Gamma correction
        x = transforms.functional.adjust_gamma(img, self.gamma)
        # Difference of Gaussian (DOG) filtering
        y = ((difference_of_gaussians(x, self.low_sigma, self.high_sigma, channel_axis=2) + 1)*255 / 2)
        # Contrast equalization
        z = transforms.functional.equalize(Image.fromarray(y.astype(np.uint8), 'RGB'))
        return z

    def __repr__(self):
        return "Gamma Correction -> DOG Filtering -> Contrast Equalization "
