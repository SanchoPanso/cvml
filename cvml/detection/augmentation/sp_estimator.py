import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import time
from datetime import datetime

from re import A
import numpy as np
from array import array
from math import pi
import torch
from torch import tensor as Tens
import ctypes
import matplotlib.pyplot as plt


class SPEstimator:
    def __init__(self,):
        self._spec = None
        self._angles = array('d',[0,45,90,135])
        P = torch.zeros((1,4,3), dtype=torch.float64)
        P.select(2, 0).copy_(torch.ones((4), dtype=torch.float64))
        P.select(2, 1).copy_(torch.as_tensor(self._angles,dtype=torch.float64).mul_(2.0*pi/180.0).cos_().squeeze_())
        P.select(2, 2).copy_(torch.as_tensor(self._angles,dtype=torch.float64).mul_(2.0*pi/180.0).sin_().squeeze_())
        self._Pinv = torch.as_tensor( torch.linalg.pinv( P.squeeze(0).detach() ).detach() ,dtype=torch.float32).detach()
        # print("size {0} data {1}".format(self._Pinv.size(),self._Pinv))

    
    def getAzimuthAndPolarization(self, input:Tens)->(Tens,Tens,Tens):  # type: ignore

        height = input.shape[0]
        width = input.shape[1]

        in_arr = np.array((ctypes.c_float*input.numel()).from_address(input.data_ptr())).reshape(input.shape)


        a_arr = self.get_a_array(in_arr)
        b_arr = self.get_b_array(in_arr)
        c_arr = self.get_c_array(in_arr)
        d_arr = self.get_d_array(in_arr)

        #print("data copied")

        common_tensor=torch.zeros(4,input.shape[0],input.shape[1])
        common_tensor.select(0,0).copy_(torch.as_tensor(np.array(a_arr.T.reshape(width, height),  order='F').reshape(height, width)))
        common_tensor.select(0,1).copy_(torch.as_tensor(np.array(b_arr.T.reshape(width, height),  order='F').reshape(height, width)))
        common_tensor.select(0,2).copy_(torch.as_tensor(np.array(c_arr.T.reshape(width, height),  order='F').reshape(height, width)))
        common_tensor.select(0,3).copy_(torch.as_tensor(np.array(d_arr.T.reshape(width, height),  order='F').reshape(height, width)))

        common_tensor = common_tensor.reshape(1, 4, width * height).squeeze_(0)
        #print("common_tensor reshaped")
        O = self._Pinv.mm(common_tensor).cpu().detach_()
        #print("multiplicated")
        copy_one = O.select(0,2).clone().detach_()
        #multiplier=0.5*180.0/pi
        multiplier=0.5
        #azimuth
        phi=copy_one.atan2_(O.select(0,1)).mul_(multiplier).clone().detach_()
        #print("atan2 called")
        #polarization
        rho=O.select(0,1).square_().add_(O.select(0,2).square_()).sqrt_().divide_(O.select(0,0))

        # print("zenith obtained")
        return (rho.clone().detach_().reshape(width,height).transpose_(0,1).clone(),
                phi.clone().detach_().reshape(width,height).transpose_(0,1).clone())
    
    def to_numpy(self, input:Tens, transpose:bool=False) -> np.ndarray:
        height = input.shape[0]
        width = input.shape[1]
        if transpose:
            in_arr = np.array((ctypes.c_float*input.numel()).from_address(input.data_ptr())).reshape(width,height)
            in_arr = np.transpose(in_arr)
        else:
            in_arr = np.array((ctypes.c_float*input.numel()).from_address(input.data_ptr())).reshape(input.shape)    
        
        return in_arr
    
    def debugShow(self, img: Tens, rho: Tens, phi: Tens, theta: Tens):
        fig,axs=plt.subplots(2,2)
        axs[0,0].imshow(self.to_numpy(img) ,cmap=plt.cm.gray)
        axs[0,1].matshow(self.to_numpy(rho,True))
        axs[1,0].matshow(self.to_numpy(phi,True))
        axs[1,1].matshow(self.to_numpy(theta,True))
        plt.show()
    
    def get_a_array(self, input_array: np.ndarray):
        arr_center_part = input_array[2::2, 2::2]
        arr_center_part = np.repeat(arr_center_part, 2, axis=0)
        arr_center_part = np.repeat(arr_center_part, 2, axis=1)

        arr_first_row = np.repeat(input_array[0, 2::2], 2).reshape(1, -1)
        arr_first_col = np.repeat(input_array[2::2, 0], 2).reshape(-1, 1)
        arr_last_row = np.zeros((1, input_array.shape[1] - 2), dtype=input_array.dtype) 
        arr_last_col = np.zeros((input_array.shape[0] - 2, 1), dtype=input_array.dtype)

        arr_0_0 = np.array([[input_array[0, 0]]])
        arr_0_last = np.array([[0]], dtype=input_array.dtype)
        arr_last_0 = np.array([[0]], dtype=input_array.dtype)
        arr_last_last = np.array([[0]], dtype=input_array.dtype)

        arr_top_part = np.concatenate([arr_0_0, arr_first_row, arr_0_last], axis=1)
        arr_middle_part = np.concatenate([arr_first_col, arr_center_part, arr_last_col], axis=1)
        arr_bottom_part = np.concatenate([arr_last_0, arr_last_row, arr_last_last], axis=1)

        arr = np.concatenate([arr_top_part, arr_middle_part, arr_bottom_part], axis=0)

        return arr
    
    def get_b_array(self, input_array: np.ndarray):
        arr_center_part = input_array[1::2, 2::2]
        arr_center_part = np.repeat(arr_center_part, 2, axis=0)
        arr_center_part = np.repeat(arr_center_part, 2, axis=1)

        arr_left_part = np.repeat(input_array[1::2, 0], 2).reshape(-1, 1)
        arr_middle_part = arr_center_part
        arr_right_part =  np.zeros((input_array.shape[0], 1), dtype=input_array.dtype)

        arr = np.concatenate([arr_left_part, arr_middle_part, arr_right_part], axis=1)

        return arr
    
    def get_c_array(self, input_array: np.ndarray):
        arr_center_part = input_array[1::2, 1::2]
        arr_center_part = np.repeat(arr_center_part, 2, axis=0)
        arr_center_part = np.repeat(arr_center_part, 2, axis=1)

        arr = arr_center_part

        return arr
    
    def get_d_array(self, input_array: np.ndarray):
        arr_center_part = input_array[2::2, 1::2]
        arr_center_part = np.repeat(arr_center_part, 2, axis=0)
        arr_center_part = np.repeat(arr_center_part, 2, axis=1)

        arr_top_part = np.repeat(input_array[0, 1::2], 2).reshape(1, -1)
        arr_middle_part = arr_center_part
        arr_bottom_part = np.zeros((1, input_array.shape[1]), dtype=input_array.dtype)

        arr = np.concatenate([arr_top_part, arr_middle_part, arr_bottom_part], axis=0)

        return arr


        