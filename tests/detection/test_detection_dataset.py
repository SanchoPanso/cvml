import os
import sys
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
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


def test_add_with_proportions():
    imsrc = ImageSource()

    d1 = DetectionDataset([imsrc, imsrc, imsrc], None, {'train': [0, 1], 'valid': [2], 'test': []})
    d2 = DetectionDataset([imsrc], None, {})

    d3 = d1.add_with_proportion(d2, {'train': 0.5, 'valid': 0.4, 'test': 0.1})

    assert len(d3.image_sources) == 4
    assert d3.samples['train'] == [0, 1]
    assert d3.samples['valid'] == [2, 3]
    assert d3.samples['test'] == []
    
    
    d4 = DetectionDataset([], None, {'train': [], 'valid': []})
    d5 = DetectionDataset([imsrc], None, {})
    
    for i in range(10):
        d4 = d4.add_with_proportion(d5, {'train': 0.5, 'valid': 0.5})
    
    assert len(d4.samples['train']) == 5
    assert len(d4.samples['valid']) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))