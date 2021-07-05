import torch
import numpy as np
class Predictor():
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def __call__(self, input_array_or_list):
        """
        input_array_or_list -- np.ndarray or [np.ndarray, np.ndarray]

        """
        if isinstance(input_array_or_list, np.ndarray):
            input_array_list = [input_array_or_list]
        else:
            input_array_list = input_array_or_list

        input_array_list = [torch.from_numpy(input_array).to(self.device, dtype=torch.float) for input_array in input_array_list]

        segmented_array = self.model(*input_array_list)
        segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
        segmented_array = np.squeeze(segmented_array, axis=0)

        return segmented_array

