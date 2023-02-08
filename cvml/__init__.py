__version__ = '0.1.1'

import logging

from cvml.annotation.annotation import Annotation

from cvml.annotation.annotation_converting import read_coco, write_coco
from cvml.annotation.annotation_converting import read_yolo, write_yolo
from cvml.annotation.annotation_converting import write_yolo_seg

from cvml.annotation.annotation_edition import change_classes_by_id
from cvml.annotation.annotation_edition import change_classes_by_names
from cvml.annotation.annotation_edition import change_classes_by_new_classes

from cvml.dataset.detection_dataset import DetectionDataset
from cvml.dataset.instance_segmentation_dataset import ISDataset

logging.getLogger('cvml').addHandler(logging.NullHandler())
LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": { 
            "format": "%(asctime)s:%(name)s:%(process)d:%(lineno)d " "%(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },    
    },
    "handlers": {
        "verbose_output": {
            "formatter": "default",
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout", 
        },
    },
    "loggers": {
        "cvml": {
            "level": "DEBUG",
            "handlers": [
                "verbose_output",
            ],
        },
    },
    "root": {},
}
