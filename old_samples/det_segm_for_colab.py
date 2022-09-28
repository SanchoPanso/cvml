import numpy as np
import cv2
import os
import time
import math
import torch
import albumentations as albu
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss


class Yolov5Detector:
    def __init__(self, weights_path: str):
        self.model = torch.hub.load('ultralytics/yolov5',
                                    'custom',
                                    path=weights_path,
                                    force_reload=True)
        self.duration = 0

    def __call__(self, img: np.ndarray,
                 count_part_x=1, count_part_y=1,
                 size=640, conf=0.25, iou=0.45):

        height, width = img.shape[0:2]
        height_crop = int(height / count_part_y)
        width_crop = int(width / count_part_x)
        det_grapes = []

        # cropping original image count_part_x*count_part_y parties and detection every parties
        for part_x in range(count_part_x):
            for part_y in range(count_part_y):
                x0_crop = part_x * width_crop
                y0_crop = part_y * height_crop
                img_crop = img[y0_crop: y0_crop + height_crop,
                           x0_crop: x0_crop + width_crop]

                dets = self.inference(img_crop, size, conf, iou)  # xywh
                dets[:, :4] = dets[:, :4].astype(int)

                dets[:, 0] += x0_crop
                dets[:, 1] += y0_crop

                if dets.shape[0] != 0 and len(dets.shape) > 1:
                    det_grapes.append(dets)
        if len(det_grapes) == 0:
            return np.zeros((0, 6))
        return np.concatenate(det_grapes, axis=0)

    def inference(self, img: np.ndarray, size=640, conf=0.25, iou=0.45) -> np.array:

        self.model.conf = conf  # NMS confidence threshold
        self.model.iou = iou  # NMS IoU threshold

        self.duration = time.time()
        results = self.model(img, size=size)
        self.duration = time.time() - self.duration

        df = results.pandas().xyxy[0]

        xmin = np.array(df['xmin'])
        xmax = np.array(df['xmax'])
        ymin = np.array(df['ymin'])
        ymax = np.array(df['ymax'])

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        bbox_xywh = np.stack([x, y, w, h]).T

        cls_conf = np.array([df['confidence']]).T
        cls_ids = np.array([df['class']]).T

        if bbox_xywh.shape[0] == 0:
            return np.zeros((0, 6))

        output = np.concatenate([bbox_xywh, cls_conf, cls_ids], axis=1)

        return output


class UnetSegmentator:
    def __init__(self, path: str):
        self.model = torch.load(path)
        self.SIZE_IMAGE = 128
        self.DEVICE = 'cuda'

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

        x_tensor = torch.from_numpy(prepr_img).to(self.DEVICE).unsqueeze(0)
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
        size_image = self.SIZE_IMAGE
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


class DetectorWithSegmentator:
    def __init__(self, detector_path: str, segmentator_path: str):
        self.detector = Yolov5Detector(detector_path)
        self.segmentator = UnetSegmentator(segmentator_path)

    def __call__(self, img, detector_conf=0.1, segmentator_conf=0.5, mask_abs_thresh=None, mask_rel_thresh=None):

        det_output = self.detector(img[:, :, ::-1], conf=detector_conf)
        filtered_bboxes = []

        for i in range(det_output.shape[0]):
            x, y, w, h, cls_conf, cls_id = det_output[i]
            if cls_id != 0:
                filtered_bboxes.append(np.array([[x, y, w, h, cls_conf, cls_id]]))
                continue

            crop_img = img[int(y):int(y + h), int(x):int(x + w)]
            crop_mask = self.segmentator(crop_img, conf=segmentator_conf)
            # cv2.imshow('crop', crop_mask * 255) #
            # cv2.waitKey()

            all_pixels_number = crop_mask.shape[0] * crop_mask.shape[1]
            obj_pixels_number = crop_mask.sum()

            if mask_abs_thresh is not None:
                if obj_pixels_number >= mask_abs_thresh:
                    filtered_bboxes.append(np.array([[x, y, w, h, cls_conf, cls_id]]))
            elif mask_rel_thresh is not None:
                # print(obj_pixels_number / all_pixels_number)    #
                if obj_pixels_number / all_pixels_number >= mask_rel_thresh:
                    filtered_bboxes.append(np.array([[x, y, w, h, cls_conf, cls_id]]))
            else:
                pass

        if len(filtered_bboxes) == 0:
            return np.zeros((0, 6))
        return np.concatenate(filtered_bboxes, axis=0)
