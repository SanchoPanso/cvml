import os
import sys
import glob
import logging

sys.path.append(os.path.dirname(__file__) + '/..')
from cvml.tools.create_dataset import create_tubes_detection_dataset
from cvml.detection.augmentation.golf_augmentation import TubeAugmentation


raw_datasets_dir = r'D:\Работа\СКЗ\datasets\SCV3_defects1'   # '/home/student2/datasets/raw/TMK_CVS3/*1'
raw_dirs = glob.glob(os.path.join(raw_datasets_dir))
raw_dirs.sort()

result_dir = r'D:\Работа\СКЗ\datasets\test_cvs3' #'/home/student2/datasets/prepared/TMK_CVS3_0701'

cls_names = ['other', 'tube', 'sink', 'riska']
split_proportions = {'train': 0.8, 'valid': 0.2, 'test': 0.0}

tube_augmentation = TubeAugmentation()

cvml_logger = logging.getLogger('cvml')

# Create handlers
s_handler = logging.StreamHandler(sys.stdout)
s_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
s_handler.setFormatter(s_format)

# Add handlers to the logger
cvml_logger.addHandler(s_handler)

create_tubes_detection_dataset(
    source_dirs=raw_dirs,
    save_dir=result_dir,
    classes=cls_names,
    sample_proportions=split_proportions,
    use_polar=True,
    install_images=True,
    install_labels=True,
    install_annotations=True,
    install_description=True,
    mask_mixup_augmentation=tube_augmentation,
    augmentation_samples = ['train'],
    crop_obj_dir=r'C:\Users\HP\Downloads\0210_defect_crops\0210_defect_crops',
    crop_class_names=cls_names,
    create_compressed_samples=False,
)
