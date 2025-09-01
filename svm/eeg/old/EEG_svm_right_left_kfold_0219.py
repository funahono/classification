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
day_sub = '20250206_B92'
f_range =[12,14]
t_range =[0,3]
comment ='Only_C_CP'
######################### svm setting ##########################################
scaler = StandardScaler()
k_num = 5
karnel = 'linear'

############################# plot settineg #####################################

label = ['R index', 'L index']
plt.figure(figsize=(6, 5))

######################### dir & path ###########################################

# dir
read_dir = f'/media/hdd1/Funatsuki/Experiment/{day_sub}/mean'
save_dir = f'/media/hdd1/Funatsuki/Experiment/{day_sub}/svm/64ch/k={k_num}'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
#read path
read_file_name = f'mean_EEG_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s_{comment}'
read_file_path = os.path.join(read_dir,read_file_name)

######################### data #################################################

#load data
data_dict = loadmat(read_file_path)
Ldata = data_dict['Lindex_mean']
Rdata = data_dict['Rindex_mean']

#transform to numpy 
Ldata = np.array(Ldata)
Rdata = np.array(Rdata)

#transpose data
Ldata = np.transpose(Ldata, (1,0))
Rdata = np.transpose(Rdata, (1,0))

#sort data in chronological order
data = np.zeros((120, 20))
for n in range(3):
    data[n*40:n*40+20, :] = Rdata[n*20:n*20+20, :]
    data[n*40+20:n*40+40, :] = Ldata[n*20:n*20+20, :]
data = scaler.fit_transform(data)

#set label Rindex:0 Lindex:1
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
    modelkun = SVC(kernel='linear')
    
    #fit model and predict
    modelkun.fit(train_data, train_labels)
    predkun = modelkun.predict(test_data)
    print('test label : ',test_labels)
    print('pred result: ',predkun)
    all_predkun.append(predkun)
    
    conf_matrix = confusion_matrix(test_labels, predkun)
    all_conf.append(conf_matrix)
    
    #calculate accuracy
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) *100
    
    #calculate F1 score
    f1 = f1_score(test_labels, predkun, average='binary') # binary classification（2 class）の場合
    fscore = round(f1*100,2) #数第2位まで丸める
    
    #print result
    print(f'k={k+1}')
    print(conf_matrix)
    print('accuracy :',accuracy,'%')
    print('F score : ',fscore, '%')
    all_acc.append(accuracy)
    all_fscore.append(fscore)
    
    ########################### save png ##########################################
    #save paht
    save_confusion_png_path = os.path.join(save_dir, f'EEG_{day_sub}_Rindex_vs_Lindex_{karnel}_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s_{k+1}_in_{k_num}.png')
    save_report_file_path = os.path.join(save_dir, f'{day_sub}_Rindex_vs_Lindex_classification_report.txt')
    
    #plot
    sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", xticklabels=label, yticklabels=label, cbar=False, linewidths=1, linecolor='black')

    # plot number
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]),
                    ha='center', va='center', color='black', fontsize=14)
            
    plt.title(f"Rindex vs Lindex\nAccuracy: {accuracy:.2f} %\nF score: {fscore:.2f} %")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # save png
    plt.savefig(save_confusion_png_path, dpi=300, bbox_inches='tight')
    plt.close()  # 画像表示後に閉じて次の出力に影響しないようにする
 

############################# average all k ####################################

average_acc = sum(all_acc)/len(all_acc) #average of all accuracy
average_fscore=round(sum(all_fscore)/len(all_fscore),2) # average of all f score
#print(all_conf)
#sum_conf = sum(all_conf)
print('all accuracy:',average_acc,'%')
print('all f score :',average_fscore,'%')

# ################################ save result ####################################
# #Save data and parameters to JSON file
# save_results_path = os.path.join(save_dir, f'EEG_{day_sub}_results_Rindex_vs_Lindex_{karnel}_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s_.json')

# #choise and settibg result　data
# results_data = {
#     "f_range": f_range,
#     "t_range": t_range,
#     "comment": comment,
#     "kernel": karnel,
#     "k_fold": k_num,
#     "all_predkun": all_predkun,
#     "all_acc": all_acc,
#     "all_fscore":all_fscore,
#     "average_acc": average_acc,
#     "average_fscore":average_fscore,
#     "all_conf": [conf.tolist() for conf in all_conf]  # Convert numpy arrays to lists
#     }

# # Save results to JSON
# with open(save_results_path, 'w') as json_file:
#     json.dump(results_data, json_file, indent=)

# # print(f"Results saved to {save_results_path}")

#print(sum_conf)

# Classification Reportを出力
# report = classification_report(test_labels,sum_conf)
# with open(save_report_file_path, "w") as f:
#     f.write(report)

# # Classification Reportの表示
# # print("Classification Report:")
# # print(report)

# # # Confusion Matrixの表示
# # print("Confusion Matrix:")
# # print(conf_matrix)