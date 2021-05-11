import pathlib
import numpy as np


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    filenames.sort()  # sort alpabetically
    return filenames

class Metrics:
    """evaluate the output of the predictions
        example: 
        met = Metrics(output, targets_res)
        
        # dice score for each class
        met.dice_score
        # iou score for each class
        met.iou_score
        to do: add a weight factor of each class
    """
    def __init__(self, pre, tar): 
        self.pre = pre # inference from the model
        self.tar = tar # target value
    
    @staticmethod
    def one_hot(input, class_values=[0, 39, 78]): 
        """
        Args: 
        input: (np.ndarray): stacked image array in [image_index, width, height]
        class_values: the value assigned to each class in the image
        """
        return (np.isclose(class_values, input[...,None])).astype(int)
    
    @property
    def tar_onehot(self):
        return self.pre_process(self.tar)
    
    @property 
    def pre_onehot(self):
        return self.pre_process(self.pre)

    @staticmethod
    def stack_images(input):
        """stack the images"""
        return np.stack(input)

    def pre_process(self, input):
        """vectorize the image data,  """
        input_stack = self.stack_images(input) # dims: [image_index, width, height]
        # print(np.unique(output_stack))
        input_onehot= self.one_hot(input_stack, class_values = np.unique(input_stack)) # dims: [image_index, width, height, obj_class]
        return input_onehot

    @property
    def iou_score(self): 
        """returns iou for the objcet classed in the ach image 
        """
        intersection = (self.tar_onehot*self.pre_onehot).sum(axis=(1,2))  # sum over width and height 
        union = np.maximum(self.tar_onehot, self.pre_onehot).sum(axis=(1,2)) # sum over width and height

        iou_score = (intersection/union).mean(axis=0)  # average over img_idx 
        return iou_score  # dims:  [obj_class]

    @property
    def dice_score(self):
        """returns iou for the objcet classed in the ach image 
        """
        intersection = (self.tar_onehot*self.pre_onehot).sum(axis=(1,2))  # sum over width and height
        pixel_total = (self.tar_onehot + self.pre_onehot).sum(axis=(1,2))
        dice_score = (2*intersection/pixel_total).mean(axis=0)
        return dice_score  # dims: [img_idx, obj_class]