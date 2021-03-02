import SimpleITK as sitk

"""
This file includes functions which cut edges the image with numpy and simpleITK.

ex)
image shape : (200, 300, 300)
lower_indices : [20, 30, 40]
upper_indices : [50, 60, 70]

Then these cropping returns the image in the range of (20 to 150, 30 to 240, 40 to 230) which has (130, 210 190) shape.

"""

def croppingForNumpy(image_array, lower_crop_size, upper_crop_size):
    assert image_array.ndim == len(lower_crop_size) == len(upper_crop_size)
    
    slices = []
    for lower, upper, size in zip(lower_crop_size, upper_crop_size, image_array.shape):
        slices.append(slice(lower, size - upper))
        
    slices = tuple(slices)
    
    cropped_image_array = image_array[slices]
    
    return cropped_image_array

def cropping(image, lower_crop_size, upper_crop_size):
    assert image.GetDimension() == len(lower_crop_size) == len(upper_crop_size)
    crop_filter = sitk.CropImageFilter()
    crop_filter.SetLowerBoundaryCropSize(lower_crop_size)
    crop_filter.SetUpperBoundaryCropSize(upper_crop_size)
    cropped_image = crop_filter.Execute(image)
    cropped_image.SetOrigin(image.GetOrigin())

    return cropped_image

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

    clipped_img = cropping(img, lower, upper)
    clipped_img_array = croppingForNumpy(img_array, lower[::-1], upper[::-1])
    print("clipped img shape: ", clipped_img.GetSize())
    print("clipped img array shape: ", clipped_img_array.shape)

    clipped_img_array = sitk.GetImageFromArray(clipped_img_array)
    clipped_img_array.SetDirection(clipped_img.GetDirection())
    clipped_img_array.SetOrigin(clipped_img.GetOrigin())
    clipped_img_array.SetSpacing(clipped_img.GetSpacing())

    sitk.WriteImage(clipped_img, save_path + "/sitk.mha", True)
    sitk.WriteImage(clipped_img_array, save_path + "/npy.mha", True)


