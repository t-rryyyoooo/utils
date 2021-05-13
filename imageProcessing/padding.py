import SimpleITK as sitk
import numpy as np

"""
This file includes functions which add values to edges by simpleITK and numpy.

ex)
image shape: [300, 300, 400]
lower indices: [10, 20, 30]
upper indices: [30, 40, 50]

Then, padding returns the image which has [340, 360, 480] shape. returned image is padded edges with const_value or mirrored.

"""

def padding(image, lower_pad_size, upper_pad_size, mirroring = False, const_value=None):
    pad_filter = sitk.MirrorPadImageFilter() if mirroring else sitk.ConstantPadImageFilter()
    if not mirroring:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        if const_value is None:
            const_value = minmax.GetMinimum()

        pad_filter.SetConstant(const_value)

    pad_filter.SetPadLowerBound(lower_pad_size)
    pad_filter.SetPadUpperBound(upper_pad_size)
    padded_image = pad_filter.Execute(image)
    padded_image.SetOrigin(image.GetOrigin())

    return padded_image


def paddingForNumpy(image_array, lower_pad_size, upper_pad_size, mirroring=False, const_value=None):

    assert image_array.ndim == len(lower_pad_size) == len(upper_pad_size)
    
    pad_width = []
    for lower, upper in zip(lower_pad_size, upper_pad_size):
        pad_width.append((lower, upper))

    if mirroring:
        mode = "symmetric"
        padded_image_array = np.pad(
                image_array, 
                pad_width = pad_width,
                mode=mode, 
                )
    else:
        mode = "constant"
        if const_value is None:
            const_value = image_array.min()

        padded_image_array = np.pad(
                image_array, 
                pad_width = pad_width,
                mode=mode, 
                constant_values = const_value
                )


    return padded_image_array

# Test
if __name__ == "__main__":
    img_path = "/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_00/imaging_resampled.nii.gz"
    save_path = "/Users/tanimotoryou/Desktop"
    img = sitk.ReadImage(img_path)
    print("img shape: ", img.GetSize())

    img_array = sitk.GetArrayFromImage(img)
    print("img array shape: ", img_array.shape)

    lower = [50, 50, 50]
    upper = [100, 100, 70]
    print("lower indices: ", lower)
    print("upper indices: ", upper)

    clipped_img = padding(img, lower, upper)
    clipped_img_array = paddingForNumpy(img_array, lower[::-1], upper[::-1])
    print("clipped img shape: ", clipped_img.GetSize())
    print("clipped img array shape: ", clipped_img_array.shape)

    clipped_img_array = sitk.GetImageFromArray(clipped_img_array)
    clipped_img_array.SetDirection(clipped_img.GetDirection())
    clipped_img_array.SetOrigin(clipped_img.GetOrigin())
    clipped_img_array.SetSpacing(clipped_img.GetSpacing())

    sitk.WriteImage(clipped_img, save_path + "/sitk.mha", True)
    sitk.WriteImage(clipped_img_array, save_path + "/npy.mha", True)


