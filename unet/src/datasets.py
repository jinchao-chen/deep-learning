import torch
from skimage.io import imread
from torch.utils import data

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 notebook=False
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.notebook = notebook

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets, repeat(self.pre_transform)))


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(input_ID), imread(target_ID)

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y


    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(inp), imread(tar)
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar

# test the classes
if __name__ == "__main__":
    # example of input and target lists (here they only have 2 images)
    inputs = ['data_semantic_segmentation_baseline/imgs/train/bonirob_2016-05-23-10-37-10_0_frame36.png',
              'data_semantic_segmentation_baseline/imgs/train/bonirob_2016-05-23-10-37-10_0_frame38.png']
    targets = ['data_semantic_segmentation_baseline/masks/train/bonirob_2016-05-23-10-37-10_0_frame36.png',
               'data_semantic_segmentation_baseline/masks/train/bonirob_2016-05-23-10-37-10_0_frame38.png']

    training_dataset = SegmentationDataSet(inputs=inputs,
                                           targets=targets,
                                           transform=None)

    training_dataloader = data.DataLoader(dataset=training_dataset,
                                          batch_size=2,
                                          shuffle=True)
    x, y = next(iter(training_dataloader))

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

# The type of x and y is already correct.
# The x should have a shape of [N, C, H, W] .
# => So the channel dimension should be second instead of last.
# The y is supposed to have a shape [N, H, W]
# The target y has only 3 classes: 0,39, 78 (0 - 255).
# => But we want dense integer encoding, meaning 0, 1, 3.
# The input is in the range [0-255] — uint8,
# => but should be normalized or linearly scaled to [0, 1] or [-1, 1].
