import numpy as np

class SlicePatchGenerator:
    def __init__(self, image_array, on_axis):
        """ Slice image array on specificed axis.

        Parameters: 
            image_array (np.array) -- Image array (any dimension)
            on_axis (int)          -- image array is sliced perpendicularto this axis. [ex] on_axis = 1 -> [:, i, ...]
        """

        self.image_array = image_array
        assert 0 <= on_axis < image_array.ndim 
        self.on_axis     = on_axis

    def __len__(self):
        return self.image_array.shape[self.on_axis]

    def __call__(self):
        """ Slice image on on_axis and yield it. """

        for i in range(self.__len__()):
            slices = self.setSlices(i)
            patch_image_array = self.image_array[slices]
            print(patch_image_array.shape)

            yield i, patch_image_array 

    def setSlices(self, i):
        """ Set slices to slice on perpendicular to on_axis.
        
        Parameters: 
            i (int) -- which location to be sliced.

        Returns:
            slices. 
            [ex] image shape = [100, 200, 200]
            When on_axis = 1, then returns [[0,1,...,99], i, [0,1,...,199]]
        """
        slices = [slice(0, s) for s in self.image_array.shape]
        slices[on_axis] = i

        return tuple(slices)

# Test
if __name__ == "__main__":
    import SimpleITK as sitk

    image_array = sitk.GetArrayFromImage(sitk.ReadImage("/Users/tanimotoryou/Documents/lab/imageData/Abdomen/case_00/imaging_resampled.nii.gz"))
    on_axis = 1

    spg = SlicePatchGenerator(image_array, on_axis)
    print(image_array.shape)
    print(spg.__len__())
    print(spg.setSlices(10))
    for i, p in spg():
        print(i, p.shape)


