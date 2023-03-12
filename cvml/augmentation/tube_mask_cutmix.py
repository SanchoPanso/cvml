import numpy as np
import cv2
from cvml.augmentation.mask_cutmix import MaskCutMix
from cvml.augmentation.mask_cutmix import get_tube_img


class TubeMaskCutMix(MaskCutMix):
    
    def __call__(self, im, labels, p=1.0, num_obj=10, **kwargs):
        tube_img, warp_mat = get_tube_img(im)
        tube_img, labels = super(self, TubeMaskCutMix).__call__(tube_img, labels, p, num_obj, **kwargs)

        inverse_warp_mat = np.invert(warp_mat)
        tube_img = cv2.warpPerspective(tube_img, inverse_warp_mat, (640, 640))
        
        return im, labels
    
