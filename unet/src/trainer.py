import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.utilities import Metrics
from src.predict_script import postprocess

writer = SummaryWriter()

class Trainer:
    """
    revision log: 
    22-May-2021: include weed dice score in the tensorboard.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 writer:torch.utils.tensorboard.SummaryWriter=SummaryWriter()
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.writer = writer

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    # learning rate scheduler step with validation loss
                    self.lr_scheduler.step(self.validation_loss[i])
                else:
                    self.lr_scheduler.step()  # learning rate scheduler step
        self.writer.flush()
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(
                self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(
                f'Training: (loss {loss_value:.4f})')  # update progressbar
        

        # write the weed score to tensorboard 
        output = postprocess(out)
        targets_res = target.cpu().numpy()
        met = Metrics(output, targets_res)
        self.writer.add_scalar("Weed Score/train", met.dice_score[2], self.epoch)
        self.writer.add_scalar("Crop Score/train", met.dice_score[1], self.epoch)


        # write loss to the tensorboard
        self.writer.add_scalar("Loss/train", loss_value, self.epoch)
        # self.writer.add_histogram('conv1.bias', self.model.conv_final.bias, self.epoch)
        # self.writer.add_histogram('conv1.weight', self.model.conv_final.weight, self.epoch)

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(
                self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()               
                valid_losses.append(loss_value)

                batch_iter.set_description(
                    f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.min(valid_losses))
        
        # write the weed score to tensorboard 
        output = postprocess(out)
        targets_res = target.cpu().numpy()
        met = Metrics(output, targets_res)

        self.writer.add_scalar("Loss/Valid", loss_value, self.epoch)
        self.writer.add_scalar("Weed Score/Valid", met.dice_score[2], self.epoch)
        self.writer.add_scalar("Crop Score/Valid", met.dice_score[1], self.epoch)

        batch_iter.close()