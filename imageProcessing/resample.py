import simpleITK as sitk

"""
This file has functions which change image's size or spacing and so on(TODO).

"""

# 3D -> 3D or 2D -> 2D
def resampleSize(image, newSize, is_label = False):
    originalSpacing = image.GetSpacing()
    originalSize = image.GetSize()

    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None


    newSpacing = [osp * os / ns for osp, os, ns in zip(originalSpacing, originalSize, newSize)]
    newOrigin = image.GetOrigin()
    newDirection = image.GetDirection()

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(newSize)
    resampler.SetOutputOrigin(newOrigin)
    resampler.SetOutputDirection(newDirection)
    resampler.SetOutputSpacing(newSpacing)

    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(image)

    return resampled

def resampleSpacing(img, spacing, is_label=False):
      # original shape
      input_shape = img.GetSize()
      input_spacing = img.GetSpacing()
      new_shape = [int(ish * isp / osp) for ish, isp, osp in zip(input_shape, input_spacing, spacing)]

      if img.GetNumberOfComponentsPerPixel() == 1:
          minmax = sitk.MinimumMaximumImageFilter()
          minmax.Execute(img)
          minval = minmax.GetMinimum()
      else:
          minval = None

      resampler = sitk.ResampleImageFilter()
      resampler.SetSize(new_shape)
      resampler.SetOutputOrigin(img.GetOrigin())
      resampler.SetOutputDirection(img.GetDirection())
      resampler.SetOutputSpacing(spacing)

      if minval is not None:
          resampler.SetDefaultPixelValue(minval)
      if is_label:
          resampler.SetInterpolator(sitk.sitkNearestNeighbor)

      resampled = resampler.Execute(img)

      return resampled


