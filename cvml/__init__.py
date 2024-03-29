__version__ = '0.2.0'

import logging
import sys

from cvml.annotation.annotation import Annotation

from cvml.annotation.annotation_converting import read_coco, write_coco
from cvml.annotation.annotation_converting import read_yolo, write_yolo
from cvml.annotation.annotation_converting import write_yolo_seg

from cvml.annotation.annotation_edition import change_classes_by_id
from cvml.annotation.annotation_edition import change_classes_by_names
from cvml.annotation.annotation_edition import change_classes_by_new_classes

from cvml.dataset.image_source import ImageSource
from cvml.dataset.detection_dataset import DetectionDataset
from cvml.dataset.instance_segmentation_dataset import ISDataset

logging.getLogger('cvml').addHandler(logging.NullHandler())
# LOGGER_CONFIG = {
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "default": { 
#             "format": "%(asctime)s:%(name)s:%(process)d:%(lineno)d " "%(levelname)s %(message)s",
#             "datefmt": "%Y-%m-%d %H:%M:%S",
#         },    
#     },
#     "handlers": {
#         "verbose_output": {
#             "formatter": "default",
#             "level": "DEBUG",
#             "class": "logging.StreamHandler",
#             "stream": "ext://sys.stdout", 
#         },
#     },
#     "loggers": {
#         "cvml": {
#             "level": "DEBUG",
#             "handlers": [
#                 "verbose_output",
#             ],
#         },
#     },
#     "root": {},
# }


def get_default_logger() -> logging.Logger:
    cvml_logger = logging.getLogger('cvml')

    # Create handlers
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    s_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    s_handler.setFormatter(s_format)

    # Add handlers to the logger
    cvml_logger.addHandler(s_handler)

    return cvml_logger

