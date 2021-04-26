import numpy as np
from itertools import product

class Patch2DGenerator:
    def __init__(self, image_array, patch_size=None, slide=None, axis=0):
        """ Extract image array on specificed axis.

        Parameters: 
            image_array (np.array) -- This image array is padded to clip correctly. 
            [ex] patch_size = [256,256], axis = 0, image_array.shape = [126, 438, 438]
            You pad image array [[0,0],[37,37],[37,37]], which makes image array size from [126, 438, 438] to [126, 512, 512].
            patch_size (list)      -- Output image size. If None, overall size.
            slide (list)           -- Slide size when generating patches.
            axis (int)          -- image array is sliced perpendicularto this axis. [ex] axis = 1 -> [:, i, ...]
        """


        image_ndim = image_array.ndim
        plane_size = self.getPlaneSize(image_array.shape, axis)

        if patch_size is None:
            """ When patch size is not specified, cut out the image at the overall size"""
            patch_size = np.array(plane_size)

        else:
            patch_size = np.array(patch_size)

        if slide is None:
            """ When slide is not specified, no overlap. """
            slide = patch_size

        else:
            slide = slide

        assert (plane_size % patch_size == 0).all()
        assert 0 <= axis < image_ndim
        assert (image_ndim - 1) == len(patch_size)
        assert len(slide) == len(patch_size)


        max_size = plane_size - patch_size + 1
        ranges = []
        for i in range(image_ndim - 1): # The dimensionality of the plane.
            r = range(0, max_size[i], slide[i])
            ranges.append(r)

        self.image_array = image_array
        self.axis        = axis
        self.patch_size  = patch_size
        self.indices     = [i for i in product(*ranges)]

    def __len__(self):
        return self.image_array.shape[self.axis] * len(self.indices)

    def __call__(self):
        """ Slice image on axis and yield it. """
        for l in range(self.image_array.shape[self.axis]):
            for index in self.indices:
                slices = self.setSlices(l, index, self.patch_size, self.axis)
                slice_image_array = self.image_array[slices]

                yield slices, slice_image_array 

    def setSlices(self, l, index, patch_size, axis):
        """ Set slices to slice perpendicular to axis.
        
        Parameters: 
            l (int)               -- Location to slice along axis.
            index (np.array)      -- Start position to slice (clip) image in the plane perpendicular to axis.
            patch_size (np.array) -- patch size [ex] [100, 100]

        Returns:
            Slices. 
            [ex] index = [10, 30], patch_size = [100, 200, 200], axis = 1
            Then returns [[10,11,...,109], i, [30,31,...,229]]
        """
        slices = [slice(i, i + p) for i, p in zip(index, patch_size)]
        slices = np.insert(slices, axis, l)

        return tuple(slices)

    def getPlaneSize(self, image_size, axis):
        """ Output the plane size perpendicular to axis. 

        Parameters:
            image_size (list or np.array) -- image array size
            axis (int)

        Returns: 
            Plane size perpendicular to axis.
        """

        s = np.arange(len(image_size))
        s = np.delete(s, axis)
        plane_size = np.array(image_size)[s]

        return plane_size


# Test
if __name__ == "__main__":
    import SimpleITK as sitk

    image_array = sitk.GetArrayFromImage(sitk.ReadImage("/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_00/imaging_resampled.nii.gz"))
    image_array = np.pad(image_array, [[0,0],[37, 37],[37,37]])
    axis = 0
    patch_size = [256, 256]
    slide = None

    spg = SlicePatchGenerator(
            image_array, 
            patch_size = patch_size,
            slide      = slide,
            axis       = axis
            )
    print(image_array.shape)
    print(spg.__len__())
    prev = None
    for j, (i, p) in enumerate(spg()):
        img = sitk.GetImageFromArray(p)
        sitk.WriteImage(img, "test/image_{}.mha".format(j))


