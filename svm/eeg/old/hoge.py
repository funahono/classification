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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
#plot
import matplotlib.pyplot as plt
import seaborn as sns

############################# information #####################################
# 作業ディレクトリを指定
day_sub = '20241111_B92'

trainkun = {'trial_num':50} #number of trial : training data
testkun = {'trial_num':10} #number of trial : test data

scaler = StandardScaler()
####################### svm setting ###########################################
k_num = 6
################################### dir #####################################

# dir
read_dir = f'/media/hdd1/Funatsuki/Experiment/{day_sub}/result'
save_dir = f'/media/hdd1/Funatsuki/Experiment/{day_sub}/result'

#path
read_file_path = os.path.join(read_dir, "mean.mat")
save_confusion_file_path = os.path.join(save_dir, f'{day_sub}_Rindex_vs_Rlittle.png')
save_report_file_path = os.path.join(save_dir, f'{day_sub}_Rindex_vs_Rlittle_classification_report.txt')

data_dict = loadmat(read_file_path)
little_data = data_dict['right_little_mean']
index_data = data_dict['right_index_mean']
little_data = np.array(little_data)
index_data = np.array(index_data)
little_data = np.transpose(little_data, (1,0))
index_data = np.transpose(index_data, (1,0))
data = np.zeros((120, 20))
for n in range(3):
    data[n*40:n*40+20, :] = index_data[n*20:n*20+20, :]
    data[n*40+20:n*40+40, :] = little_data[n*20:n*20+20, :]
data = scaler.fit_transform(data)

labels = np.ones(120,)
labels[0:20] = 0
labels[40:60] = 0
labels[80:100] = 0

skf = StratifiedKFold(k_num)
val_gen = skf.split(data, labels)

all_predkun = []
all_acc = []

for k in range(k_num):
    fold_idx = val_gen.__next__()
    train_idx = fold_idx[0]
    test_idx = fold_idx[1]
    
    train_data = data[train_idx]
    train_labels = labels[train_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]
    
    modelkun = SVC(kernel='linear')
    
    modelkun.fit(train_data, train_labels)
    predkun = modelkun.predict(test_data)
    all_predkun.append(predkun)
    
    conf_matrix = confusion_matrix(test_labels, predkun)
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) *100
    print(f'k={k}')
    print(conf_matrix)
    print('accuracy :',accuracy)
    print()
    all_acc.append(accuracy)
average_acc = sum(all_acc)/len(all_acc)
print(average_acc)
# # right_index_mean と right_little_mean を取り出す
# mean_right_index = data['right_index_mean']
# mean_right_little = data['right_little_mean']

# ####################### make testkun,trialkun #################################
# # trial range of trainkun,testkun  
# trainkun['trial_range'] = slice(0, trainkun['trial_num']) # (0,50)
# testkun['trial_range'] = slice(trainkun['trial_num'],trainkun['trial_num'] + testkun['trial_num']) #(50,60)

# # concatenate data of trainkun,testkun 
# trainkun['data']= np.concatenate([mean_right_index[:, trainkun['trial_range']], mean_right_little[:, trainkun['trial_range']]], axis=1).T #(0～49:right_index,50～99：right_little)×20(ch)
# testkun['data'] = np.concatenate([mean_right_index[:, testkun['trial_range']], mean_right_little[:, testkun['trial_range']]], axis=1).T #(0～9:right_index,10～19：right_little)×20(ch)

# # make label of trainkun,testkun 
# trainkun['label'] = np.concatenate([np.zeros(trainkun['trial_num']), np.ones(trainkun['trial_num'])])  # 0: right_index, 1: right_little
# testkun['label'] = np.concatenate([np.zeros(testkun['trial_num']), np.ones(testkun['trial_num'])])   # 0: right_index, 1: right_little

# # standard data of trainkun,testkun 
# scaler = StandardScaler() #average=0,variance=1
# trainkun['standard_data'] = scaler.fit_transform(trainkun['data']) #calculating average and variance #fitting standard scaler
# testkun['standard_data'] = scaler.transform(testkun['data'])

# ####################### make modelkun of svm #################################
# # make modelkun of svm
# modelkun = SVC(kernel='linear')  # 線形カーネルを使用

# # training modelkun
# modelkun.fit(trainkun['standard_data'], trainkun['label'])

# # predict modelkun
# predictkun = modelkun.predict(testkun['standard_data'])

# ######################## confusion matrix #####################################
# # confusion_matrixを計算
# conf_matrix = confusion_matrix(testkun['label'], predictkun)

# # 真値の合計と予測値の合計を計算
# #sum_row = conf_matrix.sum(axis=1)  # 真値（行）の合計
# #sum_col = conf_matrix.sum(axis=0)  # 予測値（列）の合計
# #total_sum = conf_matrix.sum()  # 全体の合計

# #change to 3×3 from 2×2  
# #conf_matrix_3x3 = np.vstack([np.hstack([conf_matrix, sum_row[:, np.newaxis]]), np.hstack([sum_col, total_sum])])

# # calculate accuracy(%)
# accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) *100
# ############################# plot result #####################################
# #plot label
# labels = ['R index', 'R little']

# #plot
# plt.figure(figsize=(6, 5))
# sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, linewidths=1, linecolor='black')

# # plot number
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         plt.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]),
#                  ha='center', va='center', color='black', fontsize=14)
        
# plt.title(f"Rindex vs Rlittle\nAccuracy: {accuracy:.2f} %")

# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")

# ########################### output result #####################################
# # 画像として保存
# plt.savefig(save_confusion_file_path, dpi=300, bbox_inches='tight')
# plt.close()  # 画像表示後に閉じて次の出力に影響しないようにする

# # Classification Reportを出力
# report = classification_report(testkun['label'], predictkun)
# with open(save_report_file_path, "w") as f:
#     f.write(report)

# # Classification Reportの表示
# print("Classification Report:")
# print(report)

# # Confusion Matrixの表示
# print("Confusion Matrix:")
# print(conf_matrix)