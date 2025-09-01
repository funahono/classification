# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:49:55 2024

@author: Hono
"""

from scipy.io import loadmat
import os


folder = '/media/hdd1/Funatsuki/MATLAB/vbmeg_analysis/20241111_B92_right_index_20241112_ear_ref_car_standard_brain/tf_map_n_cycles_15_ref_-3_-1/194/194_0_01.mat'


file_path = os.path.join(folder)

data = loadmat(file_path)

erds_data = data['erds_data']

trial_freq = erds_data[:, 0, :, 0]
trial_time = erds_data[:, 0, 0, :]
freq_time = erds_data[0, 0, :, :]
