## import ###############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import sys
sys.path.append('/home/honoka/programdir/kansuu')
from plt_acc_2pattern_allsub import pltbar_average_allsub

## set up ################################################

# dir
read_dir = f'/media/honoka/HDD1/Funatsuki/Experiment'
save_dir = f'/media/honoka/HDD1/Funatsuki/classification/3class'

subject_list = [#'20241111_B92',
                '20241223_B90',
                '20241115_B94',
                '20241121_B99',
                '20241122_C02',
                '20241126_B95',
                '20250107_B97',
                '20250206_B92']

eeg_comment = 'Only_C_CP'

label_list = ['EEG','cortical current']
color_list = ['gainsboro','gold']
average = 'true'
legend = 'false'


## read file  ##################################################################################################################

cur_acc_list = []
cur_fscore_list = []

eeg_acc_list = []
eeg_fscore_list = []

for subject in subject_list:

    ## current data ########
    cur_read_file = f'{read_dir}/{subject}/svm_3class/current_linear_tiebreak/{subject}_3class_classification_report.csv'
    cur_df = pd.read_csv(cur_read_file, index_col=0, encoding='utf-8-sig') * 100

    # accuracy
    cur_acc = cur_df.loc['average_result', 'accuracy']
    cur_acc_list.append(cur_acc)

    # f1-score
    cur_fscore = cur_df.loc['average_result', 'f1-score']
    cur_fscore_list.append(cur_fscore)

    ## eeg data #############
    eeg_read_file = f'{read_dir}/{subject}/svm_3class/EEG_{eeg_comment}_linear_tiebreak/{subject}_3class_classification_report.csv'
    eeg_df = pd.read_csv(eeg_read_file, index_col=0, encoding='utf-8-sig') * 100

    # accuracy
    eeg_acc = eeg_df.loc['average_result', 'accuracy']
    eeg_acc_list.append(eeg_acc)

    # f1-score
    eeg_fscore = eeg_df.loc['average_result','f1-score']
    eeg_fscore_list.append(eeg_fscore)

## plot and save ################################################################################################################

# accuracy
print('\n[ accuracy ]')
fig_acc, eeg_acc_avg, cur_acc_avg , sub_label = pltbar_average_allsub(eeg_acc_list, cur_acc_list, label_list, chance_line = 50.0,
                                                                        colors = color_list, calculate_average = average, legend = legend)


# f1-score
print('\n[ f1-score ]')
fig_fscore, eeg_fscore_avg, cur_fscore_avg, sub_label = pltbar_average_allsub(eeg_fscore_list, cur_fscore_list, label_list, chance_line = 50.0,
                                                                                colors = color_list, calculate_average = average, legend = legend)

# *.png file
fig_acc.savefig(os.path.join(save_dir,'3class_all_sub_acc_linear_tiebreak.png'),bbox_inches='tight')
fig_fscore.savefig(os.path.join(save_dir,'3class_all_sub_fscore_linear_tiebreak.png'),bbox_inches='tight')

# *.txt file
save_file_name_txt = '3class_all_sub_linear_tiebreak.txt'
txt_file_path = os.path.join(save_dir, save_file_name_txt)
with open(txt_file_path, 'w', encoding='utf-8') as f:
    f.write('--- Classification Results --- \n\n')

    f.write('[ Subject ]\n')
    for i, sub in enumerate(subject_list):
        f.write(f'{sub_label[i]}: {sub}\n')

    f.write('\n[ Accuracy ]\n')
    f.write(f"{'Subject':<10} | {label_list[0]:<10} | {label_list[1]:<10}\n")
    for i in range(len(eeg_acc_list)):
        f.write(f"{sub_label[i]:<10} | {eeg_acc_list[i]:<10.2f} | {cur_acc_list[i]:<10.2f}\n")
    f.write(f"{sub_label[-1]:<10} | {eeg_acc_avg:<10.2f} | {cur_acc_avg:<10.2f}\n")

    f.write('\n[ F1-score ]\n')
    f.write(f"{'Subject':<10} | {label_list[0]:<10} | {label_list[1]:<10}\n")
    for i in range(len(eeg_fscore_list)):
        f.write(f"{sub_label[i]:<10} | {eeg_fscore_list[i]:<10.2f} | {cur_fscore_list[i]:<10.2f}\n")
    f.write(f"{sub_label[-1]:<10} | {eeg_fscore_avg:<10.2f} | {cur_fscore_avg:<10.2f}\n")

