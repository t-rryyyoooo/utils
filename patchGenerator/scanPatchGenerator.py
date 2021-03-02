import numpy as np
import SimpleITK as sitk
import sys
sys.path.append("..")
from itertools import product
from utils.imageProcessing.clipping import clippingForNumpy, clipping

class ScanPatchGenerator:
    """ 
    This class is implemented by numpy and sitk.
    """
    def __init__(self, img_or_array, patch_size, slide):
        if isinstance(img_or_array, np.ndarray):
            shape = np.array(img_or_array.shape)
            ndim = img_or_array.ndim
            self.clip_function = clippingForNumpy

        elif isinstance(img_or_array, sitk.Image):
            shape = np.array(img_or_array.GetSize())
            ndim = img_or_array.GetDimension()
            self.clip_function = clipping

        self.img_or_array = img_or_array
        self.patch_size = np.array(patch_size)

        max_size = shape - self.patch_size + 1
        ranges = []
        for i in range(ndim):
            r = range(0, max_size[i], slide[i])
            ranges.append(r)

        self.indices = [i for i in product(*ranges)]

    def __len__(self):
        return len(self.indices)

    def __call__(self):
        for index in self.indices:
            lower_clip_size = np.array(index)
            upper_clip_size = lower_clip_size + self.patch_size

            patch_img_or_array = self.clip_function(self.img_or_array, lower_clip_size, upper_clip_size)

            yield index, patch_img_or_array


