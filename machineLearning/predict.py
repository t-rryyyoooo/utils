import torch
import numpy as np
class Predictor():
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def __call__(self, input_array_list):
        """
        image_array = [input_array_1, input_array_2, ...]

        """
        input_array_list = [torch.from_numpy(input_array).to(self.device, dtype=torch.float) for input_array in input_array_list]

        segmented_array = self.model(*input_array_list)
        segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
        segmented_array = np.squeeze(segmented_array)

        return segmented_array

