from transformations import Compose, AlbuSeg2d, DenseTarget, MoveAxis, Normalize01, Resize
from datasets import SegmentationDataSet
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib
from visual import Input_Target_Pair_Generator, show_input_target_pair_napari
from visual import plot_training
import albumentations
from unet import UNet
import torch
from trainer import Trainer
from PIL import ImageFile
from utilities import get_filenames_of_path

ImageFile.LOAD_TRUNCATED_IMAGES = True
# from torchsummary import summary

# root directory
# you must have this script in the same directory with the data folder
root = pathlib.Path('data_semantic_segmentation_baseline')
BATCH_SIZE = 2
NOTEBOOK = False

# input and target train files
inputs_train = get_filenames_of_path(root / 'imgs' / 'train')
targets_train = get_filenames_of_path(root / 'masks' / 'train')

# input and target valid files
inputs_valid = get_filenames_of_path(root / 'imgs' / 'validation')
targets_valid = get_filenames_of_path(root / 'masks' / 'validation')

# pre-transformations
pre_transforms = Compose([
    Resize(input_size=(256, 256, 3), target_size=(256, 256)),
])

# training transformations and augmentations
transforms_training = Compose([
    # Resize(input_size=(256, 256, 3), target_size=(256, 256)),
    # AlbuSeg2d(albu=albumentations.HorizontalFlip(p=0.5)),
    DenseTarget(),
    MoveAxis(),
    Normalize01()
])

# validation transformations
transforms_validation = Compose([
    # Resize(input_size=(256, 256, 3), target_size=(256, 256)),
    DenseTarget(),
    MoveAxis(),
    Normalize01()
])

# dataset training
dataset_train = SegmentationDataSet(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training,
                                    use_cache = True,
                                    pre_transform = pre_transforms,
                                    notebook = NOTEBOOK)

# dataset validation
dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation,
                                    use_cache = True,
                                    pre_transform = pre_transforms,
                                    notebook = NOTEBOOK)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)

# ---- uncomment to see the shape of the data ----
x, y = next(iter(dataloader_training))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')
# ----- end of comments -----

# ---- uncomment for visualization (uses napari) - press t to change image ----
# gen = Input_Target_Pair_Generator(dataloader_training, rgb=True)
# show_input_target_pair_napari(gen)
# ----- end of code for visualization -----

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model
model = UNet(in_channels=3,       # 3 because we have RGB images
             out_channels=3,      # 3 classes in segmentation
             n_blocks=4,          # 4 blocks of layers in each part of the unet
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)
# criterion
criterion = torch.nn.CrossEntropyLoss()
# optimizer
# we can make a function to find the optimal learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=None,
                  epochs=10,
                  epoch=0,
                  notebook=NOTEBOOK)

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=True,
                    sigma = 1, figsize = (10, 4))

# save the model
model_name =  'seg_unet_model.pt'
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)
