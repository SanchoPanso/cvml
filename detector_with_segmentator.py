import numpy as np
import cv2
import os

from detection.inference_tools.yolov5_detector import Yolov5Detector
from segmentation.inference_tools.unet_segmentator import UnetSegmentator


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


def show_with_bboxes(winname, img, det_output):
    vis_img = img.copy()
    for i in  range(det_output.shape[0]):
        x, y, w, h, cls_conf, cls_id = det_output[i]
        vis_img = cv2.rectangle(vis_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 5)
    vis_img = cv2.resize(vis_img, (400, 400))
    cv2.imshow(winname, vis_img)


if __name__ == '__main__':
    weights_dir = r'E:\PythonProjects\AnnotationConverter\weights'
    detector_path = os.path.join(weights_dir, 'yolov5l_100ep_24082022.pt')
    segmentator_path = os.path.join(weights_dir, 'Unet_128_efficientnet-b0_best_model_26082022.pth')

    sample_img_path = r'E:\PythonProjects\AnnotationConverter\datasets\defects_21082022\test\images\251_2.jpg'
    img = cv2.imread(sample_img_path)

    model = DetectorWithSegmentator(detector_path, segmentator_path)
    output_0 = model(img, mask_abs_thresh=0)
    output_1 = model(img, mask_abs_thresh=10*1000)
    output_2 = model(img, mask_abs_thresh=100*1000)

    print(output_0)
    print(output_1)
    print(output_2)

    show_with_bboxes('output_0', img, output_0)
    show_with_bboxes('output_1', img, output_1)
    show_with_bboxes('output_2', img, output_2)
    cv2.waitKey()



