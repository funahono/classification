# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:36:00 2024

@author: Hono
"""
################################### import ####################################
#load
from scipy.io import loadmat
import os
#predict
import numpy as np
#from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from sklearn.model_selection import StratifiedKFold
#plot
import matplotlib.pyplot as plt
import seaborn as sns

######################### information ##########################################
# 作業ディレクトリを指定
day_sub = '20250313_B93'
f_range =[13,15]
t_range =[0,3]

######################### svm setting ##########################################

scaler = StandardScaler()

k_num = 5
kernel_type = 'linear'

############################# plot settineg #####################################

plt_labels = ['R index', 'R little']
plt.figure(figsize=(6, 5))

######################### dir & path ###########################################

# dir
read_dir = f'/media/hdd1/Funatsuki/Experiment/{day_sub}/mean'
save_dir = f'/media/hdd1/Funatsuki/Experiment/{day_sub}/svm/k={k_num}'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#path
read_file_name = f'mean_current_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s'
read_file_path = os.path.join(read_dir,read_file_name)

######################### data #################################################

#load data
data_dict = loadmat(read_file_path)
little_data = data_dict['right_little_mean']
index_data = data_dict['right_index_mean']

#transform to numpy 
little_data = np.array(little_data)
index_data = np.array(index_data)

#transpose data
little_data = np.transpose(little_data, (1,0))
index_data = np.transpose(index_data, (1,0))

#sort data in chronological order
data = np.zeros((120, 359))
for n in range(3):
    data[n*40:n*40+20, :] = index_data[n*20:n*20+20, :]
    data[n*40+20:n*40+40, :] = little_data[n*20:n*20+20, :]
data = scaler.fit_transform(data)

#set label Rindex:0 Rlittle:1
labels = np.ones(120,)
labels[0:20] = 0
labels[40:60] = 0
labels[80:100] = 0

#set k fold
skf = StratifiedKFold(k_num)
val_gen = skf.split(data, labels)

all_predkun = []
all_acc = []
all_fscore=[]
all_conf =[]

########################## calssification #######################################

for k in range(k_num):
    #fold_idx: [0]=traindata [1]=testdata
    fold_idx = val_gen.__next__() #for tekina
    train_idx = fold_idx[0]
    test_idx = fold_idx[1]
    
    #set train data & test data
    train_data = data[train_idx]
    train_labels = labels[train_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]
    
    # set modelkun
    modelkun = SVC(kernel=kernel_type)
    
    print("test_data shape:", test_data.shape)
    print("test_labels shape:", test_labels.shape)
    print("Number of NaN values in train_data:", np.isnan(train_data).sum())
    print("test_data sample:", test_data[5:10])  # 最初の5サンプルを表示


    