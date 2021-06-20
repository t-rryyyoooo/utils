import numpy as np
import SimpleITK as sitk
import sys
sys.path.append("..")
from itertools import product
from utils.imageProcessing.clipping import clippingForNumpy, clipping
from utils.utils import isMasked

class ScanPatchGenerator:
    """ 
    This class is implemented by numpy and sitk.
    """
    def __init__(self, img_or_array, patch_size, slide, is_mask_array=False):
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

        self.is_mask_array = is_mask_array
        self.mask_bool_list = []
        if is_mask_array:
            for index in self.indices:
                lower_clip_size = np.array(index)
                upper_clip_size = lower_clip_size + self.patch_size

                patch_img_or_array = self.clip_function(self.img_or_array, lower_clip_size, upper_clip_size)

                if isinstance(patch_img_or_array, sitk.Image):
                    patch_array = sitk.GetArrayFromImage(patch_img_array_or_array)
                else:
                    patch_array = patch_img_or_array

                if isMasked(patch_array):
                    self.mask_bool_list.append(True)
                else:
                    self.mask_bool_list.append(False)

            del self.img_or_array

    def __len__(self):
        return len(self.indices)

    def __call__(self):
        if self.is_mask_array:
            for index, boolean in zip(self.indices, self.mask_bool_list):
                yield index, boolean
        else:
            for index in self.indices:
                lower_clip_size = np.array(index)
                upper_clip_size = lower_clip_size + self.patch_size

                patch_img_or_array = self.clip_function(self.img_or_array, lower_clip_size, upper_clip_size)

                yield index, patch_img_or_array


