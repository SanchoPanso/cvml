import os
import sys
import glob
import logging

sys.path.append(os.path.dirname(__file__) + '/..')
from cvml.tools.create_dataset import create_tubes_detection_dataset
from cvml.detection.augmentation.golf_augmentation import MandrelAugmentation, TubeAugmentation


raw_datasets_dir = '/home/student2/datasets/raw/TMK_3010'
raw_dirs = glob.glob(os.path.join(raw_datasets_dir, '*cvs1*'))
raw_dirs += glob.glob(os.path.join(raw_datasets_dir, '*csv1*'))
raw_dirs += glob.glob(os.path.join(raw_datasets_dir, '23_06_2021_номера_оправок_командир'))
raw_dirs += glob.glob(os.path.join(raw_datasets_dir, 'TMK_comet_november1'))
raw_dirs.sort()

template_dataset = '/home/student2/datasets/prepared/tmk_cvs1_yolov5_10012022'
result_dir = '/home/student2/datasets/prepared/tmk_cvs1_yolov5_10012022_gray' #'/home/student2/datasets/prepared/TMK_CVS3_0701'

cls_names = ['comet', 'joint', 'number']
split_proportions = {'train': 0.8, 'valid': 0.2, 'test': 0.0}

tube_augmentation = MandrelAugmentation()

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
    template_dataset=template_dataset,
    use_polar=False,
    install_images=True,
    install_labels=True,
    install_annotations=True,
    install_description=True,
    mask_mixup_augmentation=tube_augmentation,
    augmentation_samples = ['train'],
    crop_obj_dir=r'/home/student2/datasets/crops/0712_gray_comet_crops',
    crop_class_names=['comet'],
    create_compressed_samples=True,
)
