import numpy as np

def SlicePatchGenerator():
    def __init__(self, image_array=None, patch_width=1, slide=None, axis=0):
        """ Slice image array along axis.

        Parameters: 
            image_array (np.ndarray) -- The image array (3D).
            patch_width (int)        -- When axis equals to zero, the patch size is [patch_width, ...]
            slide (int)              -- The slide size along axis.
            axis (int)               -- Image array is sliced perpendicular to this axis.
            
        """
        
        self.image_array = image_array
        self.patch_width = patch_width
        self.slide       = slide
        self.axis        = axis
        self.plane_size  = self.getPlaneSize(image_array.shape, axis)

        assert (image_array.shape[axis] % patch_width) == 0

        max_size = image_array.shape[axis] - patch_width + 1
        self.indices = [ i for i in range(0, max_size, slide)]

    def __len__(self):
        return len(self.indices)

    def __call__(self):
        for index in self.indices:
            slices = tuple(np.insert(self.plane_size, self.axis, index))

            sliced_image_array = self.image_array[slices]

            yield slices, sliced_image_array

    def getPlaneSize(self, image_size, axis):
        """ Output the plane size perpendicular to axis.

        Parameters: 
            image_size (list or np.ndarray) -- image array size
            axis (int)

        Returns: 
            plane size perpendicular to axis.
        """
        s = np.arange(len(image_size))
        s = np.delete(s, axis)
        plane_size = np.array(image_size)[s]

        return plane_size
