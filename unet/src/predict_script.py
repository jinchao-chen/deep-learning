import pathlib

import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt

from src.inference import predict
from src.transformations import normalize_01, re_normalize, create_dense_target
from src.unet import UNet
from src.utilities import get_filenames_of_path, Metrics


# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img

# postprocess function


def postprocess(img: torch.tensor):
    img = torch.topk(img, 1, dim=1)[1]  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    # img = re_normalize(img)  # scale it to the range [0-255]
    return img


def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    # out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    # result = postprocess(out_softmax)  # postprocess outputs
    result = postprocess(out)

    return result


class Predictions:
    """genralize the transformation functions to make it suitable for the other models"""

    def __init__(self, images_names, targets_names, model_path):
        self.images_names = images_names
        self.targets_names = targets_names
        self.model_path = model_path

    @property
    def images(self):
        return [imread(img_name) for img_name in self.images_names]

    @property
    def targets(self):
        return [create_dense_target(imread(tar_name)) for tar_name in self.targets_names]

    # def transform(self, img_w):
    #     self.images_res = [resize(img, (img_w, img_w, 3))
    #                        for img in self.images]  # reshaped image
    #     resize_kwargs = {'order': 0,
    #                      'anti_aliasing': False, 'preserve_range': True}
    #     self.targets_res = [resize(tar, (img_w, img_w), **resize_kwargs)
    #                         for tar in self.targets]  # reshaped targets

    def evaluate(self, model, return_all=True):
        model_weights = torch.load(self.model_path)
        model.load_state_dict(model_weights)
        # device

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # print("The divice running is: ", device)

        self.output = [predict(img, model, preprocess, postprocess, device)
                       for img in self.images]
        met = Metrics(self.output, self.targets, return_all=return_all)
        self._iou = met.iou_score
        self._dice = met.dice_score

        if not return_all:
            print(f'iou scores: {self.score_str(self.iou)}')
            print(f'dice scores: {self.score_str(self.dice)}')
        else:
            max_dice = np.max(self.dice, axis=0)
            print(f'best score for each class: {self.score_str(max_dice)}')

    @property
    def iou(self):
        return self._iou

    @property
    def dice(self):
        return self._dice

    @staticmethod
    def predict(img,
                model,
                preprocess,
                postprocess,
                device,
                ):
        model.eval()
        img = preprocess(img)  # preprocess image
        x = torch.from_numpy(img).to(device)  # to torch, send to device
        with torch.no_grad():
            out = model(x)  # send through model/network

        # out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
        # result = postprocess(out_softmax)  # postprocess outputs
        result = postprocess(out)
        return result

    @staticmethod
    def score_str(scores):
        return ' | '.join(['{:.2f}'.format(x) for x in scores])

    def plot(self, idxes):
        for idx in idxes:
            fig, axes = plt.subplots(1, 3, figsize=(16, 8))

            axes[0].imshow(self.images[idx])
            axes[1].imshow(self.output[idx],cmap=plt.cm.gray)
            axes[2].imshow(self.targets[idx], cmap=plt.cm.gray)
            axes[2].set_title(self.images_names[idx].stem)


if __name__ == '__main__':
    import napari

    # root directory
    root = pathlib.Path.cwd() / 'data_semantic_segmentation_baseline'
    # input and target files
    images_names = get_filenames_of_path(root / 'imgs' / 'validation')
    targets_names = get_filenames_of_path(root / 'masks' / 'validation')

    # read images and store them in memory
    images = [imread(img_name) for img_name in images_names]
    targets = [imread(tar_name) for tar_name in targets_names]

    # Resize images and targets
    images_res = [resize(img, (256, 256, 3)) for img in images]
    resize_kwargs = {'order': 0,
                     'anti_aliasing': False, 'preserve_range': True}
    targets_res = [resize(tar, (256, 256), **resize_kwargs) for tar in targets]

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # model
    model = UNet(in_channels=3,
                 out_channels=3,
                 n_blocks=4,
                 start_filters=32,
                 activation='relu',
                 normalization='batch',
                 conv_mode='same',
                 dim=2).to(device)

    model_name = 'seg_unet_model.pt'
    model_weights = torch.load(pathlib.Path.cwd() / model_name)

    model.load_state_dict(model_weights)
    # predict the segmentation maps
    output = [predict(img, model, preprocess, postprocess, device)
              for img in images_res]

    # view predictions with napari
    # the t key for next does not work yet as expected
    with napari.gui_qt():
        viewer = napari.Viewer()
        idx = 1
        img_nap = viewer.add_image(images_res[idx], name='Input')
        tar_nap = viewer.add_labels(targets_res[idx], name='Target')
        out_nap = viewer.add_labels(output[idx], name='Prediction')

        @viewer.bind_key('t')
        def next_batch_training(viewer):
            # idx = idx + 1
            img_nap.data = images_res[3]
            tar_nap.data = targets_res[3]
            out_nap.data = output[3]
            img_nap.name = 'Input'
            tar_nap.name = 'Target'
            out_nap.name = 'Prediction'
