import numpy as np
from scipy.signal import convolve

class PerimeterLabelArrayCreater():
    def __init__(self, radius=1, num_class=14):
        self.radius      = radius
        self.num_class   = num_class

    def __call__(self, label_array):
        onehot_array = self.makeOnehot(label_array, self.num_class)

        perimeter_array = None
        for c in range(self.num_class):
            pa = self.countAmbientLabel(onehot_array[..., c], radius=self.radius)
            pa = pa[np.newaxis, ...]

            if perimeter_array is None:
                perimeter_array = pa
            else:
                perimeter_array = np.concatenate([perimeter_array, pa], axis=0)
        counter_array = self.countAmbientLabel(np.ones_like(label_array), radius=self.radius)

        perimeter_array /= counter_array

        return perimeter_array

    def makeOnehot(self, array, num_class):
        return np.eye(num_class)[array]

    def countAmbientLabel(self, array, radius=1):
        ndim = array.ndim
        kernel = np.ones([1 + 2*radius] * ndim)

        count_array = convolve(array, kernel, mode="same")

        return count_array

if __name__ == "__main__":
    import SimpleITK as sitk
    import sys
    sys.path.append("..")
    from utils import getImageWithMeta
    img = sitk.ReadImage("/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_00/segmentation_resampled.nii.gz")
    img_array = sitk.GetArrayFromImage(img)
    plac = PerimeterLabelArrayCreater()
    new_array = plac(img_array) * 300
    print(new_array)
    new = getImageWithMeta.(sitk.GetImageFromArray(new_array), img)
    sitk.WriteImage(new, "/Users/tanimotoryou/Desktop/new.mha", True)

