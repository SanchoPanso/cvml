import torch
import cv2
import numpy as np
import glob
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cvml.detection.augmentation.sp_estimator import SPEstimator
from cvml.detection.dataset.image_transforming import expo


def normalize_min_max(data):
    data_min = data.min()
    data_max = data.max()
    norm_data = (data-data_min)/(data_max-data_min)
    return norm_data


def normalize_std(data: np.array):
    data_mean = data.mean()
    data_std = data.std()
    
    data_min = data_mean - data_std
    data_max = data_mean + data_std

    norm_data = (data-data_min) / (data_max-data_min)
    return norm_data


def convert_to_mixed(orig_img: np.ndarray) -> np.ndarray:

    height, width = orig_img.shape[0:2]
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    in_data = torch.from_numpy(img).float() #torch.frombuffer(orig_img.data, dtype=torch.uint8, count=img.size).float().detach_().reshape(height, width)
    
    estimator = SPEstimator()
    rho, phi = estimator.getAzimuthAndPolarization(in_data)
    
    normalized_rho = normalize_min_max(rho).numpy()
    normalized_phi = normalize_min_max(phi).numpy()


    rho_img = (normalized_rho * 255).astype('uint8')
    phi_img = (normalized_phi * 255).astype('uint8')

    #phi_img = cv2.GaussianBlur(phi_img, (5, 5), 1)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray_img = expo(img, 15)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    img = cv2.merge([phi_img, rho_img, gray_img])
    return img


image_dir = '/home/student2/datasets/raw/TMK_3010/csv1_comet_1/images'
image_paths = glob.glob(os.path.join(image_dir, '*.png'))

for path in image_paths:
    img = cv2.imread(path)
    
    img = convert_to_mixed(img)
    img = cv2.resize(img, (600, 600))

    phi, rho, gray = cv2.split(img)
    cv2.imshow('phi', phi)
    cv2.imshow('rho', rho)
    cv2.imshow('gray', gray)

    if cv2.waitKey() == 27:
        break