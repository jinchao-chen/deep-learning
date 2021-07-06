import math
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from torch import nn
import torch
import pandas as pd
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

        revisions: 
        23-05-2021: return a score for each figure
    """

    def __init__(self, pre, tar, return_all=False):
        self.pre = pre  # inference from the model
        self.tar = tar  # target value
        self.return_all =  return_all

    @staticmethod
    def one_hot(input, class_values=[0, 1, 2]):
        """
        Args: 
        input: (np.ndarray): stacked image array in [image_index, width, height]
        class_values: the value assigned to each class in the image
        """
        return (np.isclose(class_values, input[..., None])).astype(int)

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
        input_stack = self.stack_images(
            input)  # dims: [image_index, width, height]
        # print(np.unique(output_stack))
        input_onehot = self.one_hot(input_stack)  # dims: [image_index, width, height, obj_class]
        return input_onehot

    @property
    def iou_score(self):
        """returns iou for the objcet classed in the ach image 
        """
        intersection = (
            self.tar_onehot*self.pre_onehot).sum(axis=(1, 2))  # sum over width and height
        union = np.maximum(self.tar_onehot, self.pre_onehot).sum(
            axis=(1, 2))  # sum over width and height

        if self.return_all:
          return intersection/union
        else:
          iou_score = intersection.sum(axis=0)/union.sum(axis=0)  # average over img_idx
          return iou_score  # dims:  [obj_class]

    @property
    def dice_score(self):
        """returns iou for the objcet classed in the ach image 
        """
        intersection = (
            self.tar_onehot*self.pre_onehot).sum(axis=(1, 2))  # sum over width and height
        pixel_total = (self.tar_onehot + self.pre_onehot).sum(axis=(1, 2))

        if self.return_all:
          return 2*intersection/pixel_total
        else:
          dice_score = 2*intersection.sum(axis=0)/pixel_total.sum(axis=0)
          return dice_score  # dims: [obj_class]


class LearningRateFinder:
    """
    Train a model using different learning rates within a range to find the optimal learning rate.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 device
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_history = {}
        self._model_init = model.state_dict()
        self._opt_init = optimizer.state_dict()
        self.device = device

    def fit(self,
            data_loader: torch.utils.data.DataLoader,
            steps=100,
            min_lr=1e-7,
            max_lr=1,
            constant_increment=False
            ):
        """
        Trains the model for number of steps using varied learning rate and store the statistics
        """
        self.loss_history = {}
        self.model.train()
        current_lr = min_lr
        steps_counter = 0
        epochs = math.ceil(steps / len(data_loader))

        progressbar = trange(epochs, desc='Progress')
        for epoch in progressbar:
            batch_iter = tqdm(enumerate(data_loader), 'Training', total=len(data_loader),
                              leave=False)

            for i, (x, y) in batch_iter:
                x, y = x.to(self.device), y.to(self.device)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                self.loss_history[current_lr] = loss.item()

                steps_counter += 1
                if steps_counter > steps:
                    break

                if constant_increment:
                    current_lr += (max_lr - min_lr) / steps
                else:
                    current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)

    def plot(self,
             smoothing=True,
             clipping=True,
             smoothing_factor=0.1
             ):
        """
        Shows loss vs learning rate(log scale) in a matplotlib plot
        """
        loss_data = pd.Series(list(self.loss_history.values()))
        lr_list = list(self.loss_history.keys())
        if smoothing:
            loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
            loss_data = loss_data.divide(pd.Series(
                [1 - (1.0 - smoothing_factor) ** i for i in range(1, loss_data.shape[0] + 1)]))  # bias correction
        if clipping:
            loss_data = loss_data[10:-5]
            lr_list = lr_list[10:-5]
        plt.plot(lr_list, loss_data)
        plt.xscale('log')
        plt.title('Loss vs Learning rate')
        plt.xlabel('Learning rate (log scale)')
        plt.ylabel('Loss (exponential moving average)')
        plt.show()

    def reset(self):
        """
        Resets the model and optimizer to its initial state
        """
        self.model.load_state_dict(self._model_init)
        self.optimizer.load_state_dict(self._opt_init)
        print('Model and optimizer in initial state.')
