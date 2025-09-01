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
import pandas as pd

######################### information ##########################################
# 作業ディレクトリを指定
# 作業ディレクトリを指定
day_sub = '20250107_B97'
f_range =[14,15]
t_range =[0,3]

######################### svm setting ##########################################

scaler = StandardScaler()
k_num = 5
kernel_type = 'linear'

############################# plot settineg #####################################

plt_labels = ['R index', 'L index']
plt.figure(figsize=(6, 5))

######################### dir & path ###########################################

# dir
read_dir = f'/media/honoka/HDD1/Funatsuki/Experiment/{day_sub}/mean'
save_dir = f'/media/honoka/HDD1/Funatsuki/Experiment/{day_sub}/svm/current'
# read_dir = f'/media/honoka/HDD2/Experiment/{day_sub}/mean'
# save_dir = f'/media/honoka/HDD2/Experiment/{day_sub}/svm/k={k_num}'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#path
read_file_name = f'mean_current_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s'
read_file_path = os.path.join(read_dir,read_file_name)

######################### data #################################################

#load data
data_dict = loadmat(read_file_path)
left_data = data_dict['left_index_mean']
right_data = data_dict['right_index_mean']

#transform to numpy 
left_data = np.array(left_data)
right_data = np.array(right_data)

#transpose data
left_data = np.transpose(left_data, (1,0))
right_data = np.transpose(right_data, (1,0))

#sort data in chronological order
data = np.zeros((120, 359))
for n in range(3):
    data[n*40:n*40+20, :] = right_data[n*20:n*20+20, :]
    data[n*40+20:n*40+40, :] = left_data[n*20:n*20+20, :]
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

all_precision = []
all_recall = []
all_specificity = []

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
    
    #fit model and predict
    modelkun.fit(train_data, train_labels)
    predkun = modelkun.predict(test_data)

    all_predkun.append(predkun)
    
    conf_matrix = confusion_matrix(test_labels, predkun)
    all_conf.append(conf_matrix)

    #TP,FP,FN,TP
    tn,fp,fn,tp = conf_matrix.flatten()
    
    #calculate accuracy
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) *100
    
    #calculate F1 score
    f1 = f1_score(test_labels, predkun, average='binary') # binary classification（2 class）の場合
    fscore = round(f1*100,2) #数第2位まで丸める

    #calculate recall,specificity,Precision
    precision = 100*tp / (tp + fp)
    recall = 100*tp / (tp+fn)
    specificity = 100*tn / (tn+fp)
    
    #print result
    # print(f'k={k}')

    # print('test label : ',test_labels)
    # print('pred result: ',predkun)

    # print(conf_matrix)
    # print('accuracy :',accuracy,'%')
    # print('F score : ',fscore, '%')

    all_acc.append(accuracy)
    all_fscore.append(fscore)

    all_precision.append(precision)
    all_recall.append(recall)
    all_specificity.append(specificity)
    
    ########################### save png ##########################################
    #save paht
    save_confusion_png_path = os.path.join(save_dir, f'current_{day_sub}_Rindex_vs_Lindex_{kernel_type}_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s_{k+1}.png')
    save_report_file_path = os.path.join(save_dir, f'current_{day_sub}_Rindex_vs_Lindex_classification_report.txt')
    save_report_csv_path = os.path.join(save_dir, f'current_{day_sub}_Rindex_vs_Lindex_classification_report.csv')
    
    
    #plot
    sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", xticklabels=plt_labels, yticklabels=plt_labels, cbar=False, linewidths=1, linecolor='black')

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

average_precision = sum(all_precision)/len(all_precision)
average_recall = sum(all_recall)/len(all_recall)
average_specificity = sum(all_specificity)/len(all_specificity)

print(f'---------------- {day_sub} ---------------------')
print('all accuracy:',average_acc,'%')
print('all f score :',average_fscore,'%')

average_result_text = f"Average Accuracy: {average_acc:.2f} %\n"
average_result_text += (
f"Average F-score: {average_fscore:.2f} %\n"
f"Average Precision: {average_precision:.2f}\n"
f"Average Recall: {average_recall:.2f}\n"
f"Average Specificity: {average_specificity:.2f}\n"
)

# full classification report
all_preds = np.concatenate(all_predkun)
all_true = np.concatenate([test_labels for _, test_labels in skf.split(data, labels)])

#save .txt
report = classification_report(all_true, all_preds, labels=[0, 1], target_names=plt_labels)
with open(save_report_file_path, "w") as f:
    f.write(report)
    f.write(average_result_text)

#save .csv
report_dict = classification_report(all_true, all_preds, labels=[0, 1], target_names=plt_labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df["specificity"] = np.nan

# 代入
report_df.loc["average_result", "precision"] = average_precision /100
report_df.loc["average_result", "recall"] = average_recall /100
report_df.loc["average_result", "f1-score"] = average_fscore / 100
report_df.loc["average_result", "support"] = average_acc / 100 * len(all_true)
report_df.loc["average_result", "specificity"] = average_specificity /100
report_df.loc["average_result", "accuracy"] = average_acc / 100

report_df.to_csv(save_report_csv_path,encoding='utf-8-sig', index=True)