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
#kansuu
import sys
sys.path.append('/home/honoka/programdir/kansuu')
from svc_ovo_tiebreak import predict_ovo_with_tiebreak
from plt_score_3class import visualize_decision_scores
#plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

######################### information ##########################################
# 作業ディレクトリを指定
day_sub_list = [
    '20241111_B92',
    '20241115_B94',
    '20241121_B99',
    '20241122_C02',
    '20241126_B95',
    '20241223_B90',
    '20250107_B97',
    '20250206_B92',
    '20250313_B93'
]

f_range_list = [
    [12,14],  # 20241111_B92
    [10,13],  # 20241115_B94
    [12,13],  # 20241121_B99
    [12,14],  # 20241122_C02
    [12,14],  # 20241126_B95
    [12,15],  # 20241223_B90
    [14,15],  # 20250107_B97
    [12,14],  # 20250206_B92
    [13,15]   # 20250313_B93
]

t_range_list = [
    [2,3],    # 20241111_B92
    [0,3],    # 20241115_B94
    [1,2],    # 20241121_B99
    [0,3],    # 20241122_C02
    [0,3],    # 20241126_B95
    [0,3],    # 20241223_B90
    [0,3],    # 20250107_B97
    [0,3],    # 20250206_B92
    [0,3]     # 20250313_B93
]

######################### svm setting ##########################################

scaler = StandardScaler()

k_num = 5
kernel_type = 'linear'
multi_classfication_type='ovo' #'ovr':one-vs-rest #'ovo':one-vs-one

############################# plot settineg #####################################

plt_labels = ['R index', 'L index','R little']
plt.figure(figsize=(6, 5))

raw_datasets = ['Rindex_data', 'Lindex_data', 'Rlittle_data']


for day_sub , f_range, t_range in zip(day_sub_list, f_range_list, t_range_list):

    ######################### dir & path ###########################################

    # dir
    read_dir = f'/media/honoka/HDD1/Funatsuki/Experiment/{day_sub}/mean'
    save_dir = f'/media/honoka/HDD1/Funatsuki/Experiment/{day_sub}/svm_3class/current_{kernel_type}_tiebreak'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #path
    read_file_name = f'mean_current_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s'
    read_file_path = os.path.join(read_dir,read_file_name)

    ######################### data #################################################

    #load data
    data_dict = loadmat(read_file_path)
    Rlittle_data = data_dict['right_little_mean']
    Rindex_data = data_dict['right_index_mean']
    Lindex_data = data_dict['left_index_mean']

    # transpose in loop
    class_datasets = [np.transpose(np.array(Rindex_data), (1, 0)),
                    np.transpose(np.array(Lindex_data), (1, 0)),
                    np.transpose(np.array(Rlittle_data), (1, 0))]
    class_num=len(class_datasets)

    #sort data in chronological order and set label( Rindex:0, Lindex;1, Rlittle:2)
    data = np.zeros((20*3*class_num, 359)) #(60 × 3 class, 359 vertexs)
    labels = np.zeros(20*3*class_num)

    for s in range(3):
        for c, class_data in enumerate(class_datasets):  # クラス番号とデータ
            start_idx = (s * class_num + c) * 20
            end_idx = start_idx + 20
            data[start_idx:end_idx, :] = class_data[s*20:(s+1)*20, :]
            labels[start_idx:end_idx] = c

    data = scaler.fit_transform(data)

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
        #modelkun = SVC(kernel=kernel_type,C=100, gamma='scale',decision_function_shape=multi_classfication_type)
        modelkun = SVC(kernel=kernel_type,decision_function_shape=multi_classfication_type)

        #fit model and predict
        modelkun.fit(train_data, train_labels)
        predkun , decision = predict_ovo_with_tiebreak(modelkun, test_data)
        all_predkun.append(predkun)

        conf_matrix = confusion_matrix(test_labels, predkun)
        all_conf.append(conf_matrix)

        #calculate accuracy
        accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) *100

        #calculate F1 score
        f1 = f1_score(test_labels, predkun, average='weighted')
        fscore = round(f1*100,2) #数第2位まで丸める

        #print result
        # print(f'k={k}')

        # print('test label : ',test_labels)
        # print('pred result: ',predkun)

        # print(conf_matrix)
        # print('accuracy :',accuracy,'%')
        # print('F score : ',fscore, '%')

        all_acc.append(accuracy)
        all_fscore.append(fscore)

        ########################### save png ##########################################
        #save paht
        save_confusion_png_path = os.path.join(save_dir, f'{day_sub}_3class_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s_{k+1}.png')
        save_report_file_path = os.path.join(save_dir, f'{day_sub}_3class_classification_report.txt')
        save_report_csv_path = os.path.join(save_dir, f'{day_sub}_3class_classification_report.csv')

        #plot
        sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", xticklabels=plt_labels, yticklabels=plt_labels, cbar=False, linewidths=1, linecolor='black')

        # plot number
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]),
                        ha='center', va='center', color='black', fontsize=14)

        plt.title(f"Rindex vs Lindex vs Rlittle\nAccuracy: {accuracy:.2f} %\nF score: {fscore:.2f} %")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # save png
        plt.savefig(save_confusion_png_path, dpi=300, bbox_inches='tight')
        plt.close()  # 画像表示後に閉じて次の出力に影響しないようにする

        # score plot
        # visualize_decision_scores(decision: np.ndarray, test_labels: np.ndarray)
        save_decision_png_path = os.path.join(save_dir, f'{day_sub}_3class_decision_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s_{k+1}.png')
        fig = visualize_decision_scores(decision=decision, test_labels = test_labels, label_list = plt_labels)
        fig.savefig(save_decision_png_path)
        plt.close()

    ############################# average all k ####################################

    average_acc = sum(all_acc)/len(all_acc) #average of all accuracy
    average_fscore=round(sum(all_fscore)/len(all_fscore),2) # average of all f score

    print(f'---------------- {day_sub} ---------------------')
    print('all accuracy:',average_acc,'%')
    print('all f score :',average_fscore,'%')

    average_result_text = f"Average Accuracy: {average_acc:.2f} %\n"
    average_result_text += (
        f"Average F-score: {average_fscore:.2f} %\n"
        )

    # full classification report
    all_preds = np.concatenate(all_predkun)
    all_true = np.concatenate([test_labels for _, test_labels in skf.split(data, labels)])

    #save .txt
    report = classification_report(all_true, all_preds, labels=[0, 1, 2], target_names=plt_labels)
    with open(save_report_file_path, "w") as f:
        f.write(report)
        f.write(average_result_text)

    #save .csv
    report_dict = classification_report(all_true, all_preds, labels=[0, 1, 2], target_names=plt_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    report_df["specificity"] = np.nan

    # 代入
    report_df.loc["average_result", "f1-score"] = average_fscore / 100
    report_df.loc["average_result", "support"] = average_acc / 100 * len(all_true)
    report_df.loc["average_result", "accuracy"] = average_acc / 100

    report_df.to_csv(save_report_csv_path,encoding='utf-8-sig', index=True)