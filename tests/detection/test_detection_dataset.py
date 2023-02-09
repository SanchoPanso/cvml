import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from cvml import DetectionDataset, ImageSource


def test_magic_add():
    imsrc = ImageSource()

    d1 = DetectionDataset([imsrc, imsrc, imsrc], None, {'train': [0, 1], 'valid': [2], 'test': []})
    d2 = DetectionDataset([imsrc, imsrc, imsrc], None, {'train': [0], 'valid': [1, 2]})

    d3 = d1 + d2

    assert len(d3.image_sources) == 6
    assert d3.samples['train'] == [0, 1, 3]
    assert d3.samples['valid'] == [2, 4, 5]
    assert d3.samples['test'] == []

