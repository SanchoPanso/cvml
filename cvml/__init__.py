__version__ = '0.1.1'

import logging
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
