import numpy as np

def calculatePaddingSize(image_size, image_patch, label_patch, slide):
    if not isinstance(image_size, np.ndarray):
        image_size = np.array(image_size)
    if not isinstance(image_size, np.ndarray):
        image_size = np.array(image_size)
    if not isinstance(image_size, np.ndarray):
        image_size = np.array(image_size)
    if not isinstance(image_size, np.ndarray):
        image_size = np.array(image_size)

    just = (image_size % label_patch) != 0
    label_pad_size = just * (label_patch - (image_size % label_patch)) + (label_patch - slide)
    image_pad_size = label_pad_size + (image_patch - label_patch)

    lower_pad_size_label = label_pad_size // 2
    upper_pad_size_label = (label_pad_size + 1) // 2
    lower_pad_size_image = image_pad_size // 2
    upper_pad_size_image = (image_pad_size + 1) // 2
    lower_pad_size = np.array([lower_pad_size_image, lower_pad_size_label])
    upper_pad_size = np.array([upper_pad_size_image, upper_pad_size_label])
    return lower_pad_size, upper_pad_size


