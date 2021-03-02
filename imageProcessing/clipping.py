"""
This file includes functions which dwaw out the image by simpleITK and numpy.

ex)
image shape : [500, 500, 100]
lower_clip_indices : [50, 60, 70]
upper_clip_indices : [100, 100, 100]

Then, clipping functions return the image in the range of (50 to 100, 50 to 100, 50 to 100) which has (50, 40, 30) shape. 

"""
def clipping(image, lower_clip_indices, upper_clip_indices):
    assert image.GetDimension() == len(lower_clip_indices) == len(upper_clip_indices)
    slices = []
    for lower, upper in zip(lower_clip_indices, upper_clip_indices):
        s = slice(lower, upper)
        slices.append(s)

    slices = tuple(slices)

    clipped_image = image[slices]
    clipped_image.SetOrigin(image.GetOrigin())

    return clipped_image


def clippingForNumpy(image_array, lower_clip_indices, upper_clip_indices):
    assert image_array.ndim == len(lower_clip_indices) == len(upper_clip_indices)
    slices = []
    for lower, upper in zip(lower_clip_indices, upper_clip_indices):
        s = slice(lower, upper)
        slices.append(s)

    slices = tuple(slices)

    clipped_image_array = image_array[slices]

    return clipped_image_array


# Test
if __name__ == "__main__":
    import SimpleITK as sitk
    img_path = "/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_00/imaging_resampled.nii.gz"
    save_path = "/Users/tanimotoryou/Desktop"
    img = sitk.ReadImage(img_path)
    print("img shape: ", img.GetSize())

    img_array = sitk.GetArrayFromImage(img)
    print("img array shape: ", img_array.shape)

    lower = [50, 50, 50]
    upper = [100, 100, 70]
    print("lower: ", lower)
    print("upper: ", upper)

    clipped_img = clipping(img, lower, upper)
    clipped_img_array = clippingForNumpy(img_array, lower[::-1], upper[::-1])
    print("clpped img shape: ", clipped_img.GetSize())
    print("clipped img array shape: ", clipped_img_array.shape)

    clipped_img_array = sitk.GetImageFromArray(clipped_img_array)
    clipped_img_array.SetDirection(clipped_img.GetDirection())
    clipped_img_array.SetOrigin(clipped_img.GetOrigin())
    clipped_img_array.SetSpacing(clipped_img.GetSpacing())


    sitk.WriteImage(clipped_img, save_path + "/sitk.mha", True)
    sitk.WriteImage(clipped_img_array, save_path + "/npy.mha", True)


