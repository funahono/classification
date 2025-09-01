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
#plot
import matplotlib.pyplot as plt
import seaborn as sns

############################# information #####################################
# 作業ディレクトリを指定
data_dir = 'C:/Users/Hono/Desktop/datadir'
day_sub = '20241126_B95'

trainkun = {'trial_num':50} #number of trial : training data
testkun = {'trial_num':10} #number of trial : test data

################################### dir #####################################

# dir
read_dir = f'C:/Users/Hono/Desktop/datadir/{day_sub}/result'
save_dir = f'C:/Users/Hono/Desktop/datadir/{day_sub}/result'

#path
read_file_path = os.path.join(read_dir, "mean.mat")
save_confusion_file_path = os.path.join(save_dir, f'{day_sub}_Rindex_vs_Lindex.png')
save_report_file_path = os.path.join(save_dir, f'{day_sub}_Rindex_vs_Lindex_classification_report.txt')

data = loadmat(read_file_path)

# right_index_mean と left_index_mean を取り出す
mean_right_index = data['right_index_mean']
mean_left_index = data['left_index_mean']

####################### make testkun,trialkun #################################
# trial range of trainkun,testkun  
trainkun['trial_range'] = slice(0, trainkun['trial_num']) # (0,50)
testkun['trial_range'] = slice(trainkun['trial_num'],trainkun['trial_num'] + testkun['trial_num']) #(50,60)

# concatenate data of trainkun,testkun 
trainkun['data']= np.concatenate([mean_right_index[:, trainkun['trial_range']], mean_left_index[:, trainkun['trial_range']]], axis=1).T #(0～49:right_index,50～99：left_index)×20(ch)
testkun['data'] = np.concatenate([mean_right_index[:, testkun['trial_range']], mean_left_index[:, testkun['trial_range']]], axis=1).T #(0～9:right_index,10～19：left_index)×20(ch)

# make label of trainkun,testkun 
trainkun['label'] = np.concatenate([np.zeros(trainkun['trial_num']), np.ones(trainkun['trial_num'])])  # 0: right_index, 1: left_index
testkun['label'] = np.concatenate([np.zeros(testkun['trial_num']), np.ones(testkun['trial_num'])])   # 0: right_index, 1: left_index

# standard data of trainkun,testkun 
scaler = StandardScaler() #average=0,variance=1
trainkun['standard_data'] = scaler.fit_transform(trainkun['data']) #calculating average and variance #fitting standard scaler
testkun['standard_data'] = scaler.transform(testkun['data'])

####################### make modelkun of svm #################################
# make modelkun of svm
modelkun = SVC(kernel='linear')  # 線形カーネルを使用

# training modelkun
modelkun.fit(trainkun['standard_data'], trainkun['label'])

# predict modelkun
predictkun = modelkun.predict(testkun['standard_data'])


######################## confusion matrix #####################################
# confusion_matrixを計算
conf_matrix = confusion_matrix(testkun['label'], predictkun)

# 真値の合計と予測値の合計を計算
#sum_row = conf_matrix.sum(axis=1)  # 真値（行）の合計
#sum_col = conf_matrix.sum(axis=0)  # 予測値（列）の合計
#total_sum = conf_matrix.sum()  # 全体の合計

#change to 3×3 from 2×2  
#conf_matrix_3x3 = np.vstack([np.hstack([conf_matrix, sum_row[:, np.newaxis]]), np.hstack([sum_col, total_sum])])

# calculate accuracy(%)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) *100
############################# plot result #####################################
#plot label
labels = ['R index', 'L index']

#plot
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, linewidths=1, linecolor='black')

plt.title(f"Rindex vs Lindex\nAccuracy: {accuracy:.2f} %")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

########################### output result #####################################
# 画像として保存
plt.savefig(save_confusion_file_path, dpi=300, bbox_inches='tight')
plt.close()  # 画像表示後に閉じて次の出力に影響しないようにする

# Classification Reportを出力
report = classification_report(testkun['label'], predictkun)
with open(save_report_file_path, "w") as f:
    f.write(report)

# Classification Reportの表示
print("Classification Report:")
print(report)

# Confusion Matrixの表示
print("Confusion Matrix:")
print(conf_matrix)