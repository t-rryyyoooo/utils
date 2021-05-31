import torch
import numpy as np
class Segmenter():
    def __init__(self, model, num_input_array=1, ndim=5, device="cpu"):
        """ Predict labels (images) with model.

        Parameters: 
            model (nn.Module)     -- trained model.
            num_input_array (int) -- The number of arrays (images) fed to model.
            ndim (list or int ex) -- The number of dimension of images fed to model. [ex] [5, 4, 4] or 5
            device (str)          -- calculation machine. (cpu, gpu)
        """
        self.model = model
        self.num_input_array = num_input_array
        self.device = device
        if isinstance(ndim, int):
            self.ndim = [ndim] * num_input_array
        else:
            assert len(self.ndim) == num_input_array

    def forward(self, input_array_list):
        """
        image_array = [input_array_1, input_array_2, ...]

        """
        assert len(input_array_list) == self.num_input_array
        for i in range(self.num_input_array):
            while input_array_list[i].ndim < self.ndim[i]:
                input_array_list[i] = input_array_list[i][np.newaxis, ...]

            input_array_list[i] = torch.from_numpy(input_array_list[i]).float()#.to(self.device, dtype=torch.float)
            for j in input_array_list:
                print(j)

        segmented_array = self.model(*input_array_list)
        segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
        segmented_array = np.squeeze(segmented_array)

        return segmented_array

