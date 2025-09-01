## import ###############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## set up ################################################

#data
read_dir = f'/media/honoka/HDD1/Funatsuki/Experiment'
subject_list = [
                # '20241111_B92',
                '20241223_B90',
                '20241115_B94',
                # '20241121_B99',
                # '20241122_C02',
                '20241126_B95',
                '20250107_B97',
                '20250206_B92',
                # '20250313_B93'
                ]

key = 'f1-score'
comment = 'Only_C_CP'

#plot
colors = ['gainsboro','gold']
save_dir = f'/media/honoka/HDD1/Funatsuki/classification/202507_BMEiCON'

label_list =[
            'A',
            'B',
            'C',
            'D',
            'E',
            # 'F',
            # 'G',
            # 'H',
            # 'I',
            '',
            'avg.'
            ]

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 25


## read file & culculate avg ############################

cur_acc_list = []
eeg_acc_list = []

#read file of each subject
for subject in subject_list:
    
    #current data
    cur_read_file = f'{read_dir}/{subject}/svm/current/current_{subject}_Rindex_vs_Rlittle_classification_report.csv'
    cur_df = pd.read_csv(cur_read_file, index_col=0, encoding='utf-8-sig') * 100
    cur_acc = cur_df.loc['average_result', key]

    # print(cur_df)

    cur_acc_list.append(cur_acc)

    #eeg data
    eeg_read_file = f'{read_dir}/{subject}/svm/EEG_{comment}/EEG_{subject}_Rindex_vs_Rlittle_classification_report.csv'
    eeg_df = pd.read_csv(eeg_read_file, index_col=0, encoding='utf-8-sig') * 100
    eeg_acc = eeg_df.loc['average_result', key]

    eeg_acc_list.append(eeg_acc)

#calculate average
print(cur_acc_list)
cur_avg_acc = sum(cur_acc_list)/len(cur_acc_list)
eeg_avg_acc = sum(eeg_acc_list)/len(eeg_acc_list)
# print('==================== sub list ======================')
# print('current:',cur_acc_list)
# print('EEG:',eeg_acc_list)
print('==================== accuracy ======================')
print('current:',cur_avg_acc,'%')
print('EEG:',eeg_avg_acc,'%')
print('====================================================')

## plot result ##############################################

x = np.arange(len(subject_list)+2)
width = 0.3 #line width

# add blank and average
eeg_acc_list += [0, eeg_avg_acc]
cur_acc_list += [0, cur_avg_acc]

plt.figure(figsize=(12, 5))

for i in range(len(subject_list)+2):
    x_eeg = x[i] - width
    y_eeg = eeg_acc_list[i] - 0.5

    x_cur = x[i]
    y_cur = cur_acc_list[i] - 0.5

    # plt.plot([x_eeg, x_cur], [y_eeg, y_cur], color='black', linestyle='--', linewidth=0.5)

plt.bar(x - width/2, eeg_acc_list, width, color=colors[0],edgecolor = 'black')
plt.bar(x + width/2, cur_acc_list, width, color = colors[1],edgecolor = 'black')

plt.xticks(x, label_list)

if key == 'accuracy':
    plt.ylabel('Accuracy (%)')
    save_file_name = 'Rindex_vs_Rlittle_all_sub_acc.png'

elif key == 'f1-score':
    plt.ylabel('F1-Score (%)')
    save_file_name = 'Rindex_vs_Rlittle_all_sub_fscore.png'

plt.xlabel('Subject')
plt.ylim(0, 100)
plt.yticks(np.arange(0, 101, 10))

#plt.legend(loc = 'upper right',bbox_to_anchor = (1.15, 1))
plt.subplots_adjust(right = 0.8)

#ch level
plt.axhline(y = 50.00, color = 'black', linestyle = '--', linewidth = 1.5)

plt.savefig(os.path.join(save_dir,save_file_name),bbox_inches='tight')
