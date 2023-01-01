import os
import sys
import pytest
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cvml.detection.dataset.image_source import ImageSource, SingleImageReader, MultipleImageReader, DetectionImageSource


def zero_preprocessing(img: np.ndarray) -> np.ndarray:
    return img * 0 


def test_single_image_reader():
    image_dir = os.path.join(os.path.dirname(__file__), 'test_files')
    image_path = os.path.join(image_dir, '1.png')
    
    image_reader = SingleImageReader()
    
    assert image_reader.get_name([image_path]) == '1'
    assert np.array_equal(image_reader.read([image_path], [lambda x: x]), (cv2.imread(image_path, cv2.IMREAD_COLOR)))
    assert np.array_equal(image_reader.read([image_path], [zero_preprocessing]), cv2.imread(image_path, cv2.IMREAD_COLOR) * 0)


def test_multiple_image_reader():
    image_dir = os.path.join(os.path.dirname(__file__), 'test_files')
    
    image_path_1 = os.path.join(image_dir, '1.png')
    image_path_2 = os.path.join(image_dir, '2.png')
    image_path_3 = os.path.join(image_dir, '3.png')
    image_paths = [image_path_1, image_path_2, image_path_3]
    
    image_reader = MultipleImageReader(main_channel=0)
    
    assert image_reader.get_name(image_paths) == '1'
    
    expected_image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
    expected_image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
    expected_image_3 = cv2.imread(image_path_3, cv2.IMREAD_GRAYSCALE)
    expected_image = cv2.merge([expected_image_1, expected_image_2, expected_image_3 * 0])
    test_prepr = [lambda x: x, lambda x: x, zero_preprocessing]
    
    assert np.array_equal(image_reader.read(image_paths, test_prepr), expected_image)
    
    
    
    
