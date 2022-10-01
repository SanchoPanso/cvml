import os
import math
import torch
import numpy as np
import cv2
import albumentations as albu
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss

from data_science_tools.core.segmenter import Segmenter


class UnetSegmentator(Segmenter):
    def __init__(self, path: str, device: str = 'cpu'):
        self.model = torch.load(path, map_location=device)
        self.size_image = 128
        self.device = device

    def __call__(self, img: np.ndarray, conf: float = 0.5) -> np.ndarray:

        x_tensor = self.prepare_to_inference(img)
        y_tensor = self.model.predict(x_tensor)
        predicted_mask = self.prepare_from_inference(y_tensor, img, conf)

        return predicted_mask

    def prepare_to_inference(self, img: np.ndarray):
        ENCODER = 'efficientnet-b0'
        ENCODER_WEIGHTS = 'imagenet'

        augmentation = self.get_validation_augmentation()

        params = smp.encoders.get_preprocessing_params(ENCODER, ENCODER_WEIGHTS)
        preprocessing_fn = smp.encoders.functools.partial(smp.encoders.preprocess_input, **params)
        preprocessing = self.get_preprocessing(preprocessing_fn)

        aug_img = augmentation(image=img)['image']
        prepr_img = preprocessing(image=aug_img)['image']

        x_tensor = torch.from_numpy(prepr_img).to(self.device).unsqueeze(0)
        return x_tensor

    def prepare_from_inference(self, tensor, orig_img: np.ndarray, conf: float):
        pr_mask = ((tensor.squeeze().cpu().numpy() + (0.5 - conf)).round())

        max_size = max(orig_img.shape[0:2])
        min_size = min(orig_img.shape[0:2])
        new_mask_size = (max_size, max_size)
        pr_mask = cv2.resize(pr_mask, new_mask_size)

        pr_mask = pr_mask[
                  (max_size - orig_img.shape[0]) // 2: (max_size + orig_img.shape[0]) // 2,
                  (max_size - orig_img.shape[1]) // 2: (max_size + orig_img.shape[1]) // 2,
                  ]
        assert pr_mask.shape[:2] == orig_img.shape[:2]

        pr_mask = pr_mask.round()
        pr_mask = pr_mask.astype('uint8')

        return pr_mask

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        size_image = self.size_image
        test_transform = [
            albu.LongestMaxSize(max_size=size_image, p=1.0),
            # albu.ToGray(p=1.0),
            albu.PadIfNeeded(size_image, size_image, border_mode=cv2.BORDER_CONSTANT)
        ]
        return albu.Compose(test_transform)

    def get_preprocessing(self, preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor),
        ]
        return albu.Compose(_transform)


if __name__ == '__main__':
    weights_dir = r'..\..\weights'
    weights_path = os.path.join(weights_dir, 'Unet_128_efficientnet-b0_best_model_26082022.pth')
    model = UnetSegmentator(weights_path)

    sample_img_path = '0.jpg'
    img = cv2.imread(sample_img_path)

    mask = model(img)

    print(mask.shape[0] * mask.shape[1])
    print(mask.sum())

    cv2.imshow('img', img)
    cv2.imshow('mask', mask * 255)
    cv2.waitKey()


