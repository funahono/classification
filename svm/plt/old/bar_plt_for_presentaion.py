import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# matrix データ
#list = [subject, Rindex vs lindex EEG , R index vs Lindex current, Rindex vs Rlittel EEG, Rindex vs Rlittle current]
matrix = np.array([
    ['A', '75.83', '86.67','83.33','90.00'],#B90
    ['B','90.00','83.33','66.67','73.33'],#B94
    ['C','34.17','70.83','46.67','67.50'],#B99
    ['D','80.00','85.00','80.83','92.50'],#C02
    ['E', '78.33', '80.83','76.67','86.67'],#B95
    ['F', '59.17', '70.00','53.33','70.83'],#B97
    ['J','46.67','64.17','50.00','65.00']#B92
    ])

save_dir = '/media/honoka/HDD1/Funatsuki/Experiment'

# ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# データの抽出
subjects = matrix[:, 0]  # 被験者名
eeg_acc_right_left = matrix[:, 1].astype(float)  # 脳波データ R index vs Lindex
current_acc_right_left = matrix[:, 2].astype(float)  # 皮質電流データ R index vs Lindex
eeg_acc_index_little = matrix[:, 3].astype(float)  # 脳波データ R index vs Rlittle
current_acc_index_little = matrix[:, 4].astype(float)  # 皮質電流データ R index vs Rlittle


#calculate average
ave_eeg_acc_index_little =sum(eeg_acc_index_little)/len(eeg_acc_index_little)
ave_eeg_acc_right_left=sum(eeg_acc_right_left)/len(eeg_acc_right_left)
ave_current_acc_index_little=sum(current_acc_index_little)/len(current_acc_index_little)
ave_current_acc_right_left=sum(current_acc_right_left)/len(current_acc_right_left)

print('============Rindex vs Lindex==========')
print('EEG:',ave_eeg_acc_right_left,'%')
print('current:',ave_current_acc_right_left,'%')
print('============Rindex vs Rlittle==========')
print('EEG:',ave_eeg_acc_index_little,'%')
print('current:',ave_current_acc_index_little,'%')

#list include blank
eeg_acc_right_left = np.append(eeg_acc_right_left,'0')  # 脳波データ R index vs Lindex
current_acc_right_left = np.append(current_acc_right_left,'0')  # 皮質電流データ R index vs Lindex
eeg_acc_index_little = np.append(eeg_acc_index_little,'0')  # 脳波データ R index vs Rlittle
current_acc_index_little = np.append(current_acc_index_little,'0')  # 皮質電流データ R index vs Rlittle
subjects=np.append(subjects,' ')

#list include average
eeg_acc_right_left = np.append(eeg_acc_right_left,ave_eeg_acc_right_left).astype(float)  # 脳波データ R index vs Lindex
current_acc_right_left = np.append(current_acc_right_left,ave_current_acc_right_left).astype(float)  # 皮質電流データ R index vs Lindex
eeg_acc_index_little = np.append(eeg_acc_index_little,ave_eeg_acc_index_little).astype(float)  # 脳波データ R index vs Rlittle
current_acc_index_little = np.append(current_acc_index_little,ave_current_acc_index_little).astype(float)  # 皮質電流データ R index vs Rlittle
subjects=np.append(subjects,'avg.')



method=np.array(['EEG','cortical current'])

    
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 20
    
colors = [
    #each subject
    'navy', 
    # 脳波データ R index vs Lindex
    'forestgreen',  
    #  皮質電流データ R index vs Lindex
    'royalblue', 
    # 脳波データ R index vs Lindex
    'darkorange',
    # 皮質電流データ R index vs Lindex
    'crimson',
    #被験者
     'seagreen', 'steelblue', 'darkslateblue','orangered',
    'mediumblue', 'purple', 'darkgreen', 'darkgoldenrod', 'midnightblue',
    'tomato', 'slateblue', 'chocolate', 'firebrick', 'darkcyan',
    'blueviolet', 'green', 'darkturquoise', 'saddlebrown', 'maroon',
    'navy', 'olivedrab', 'darkkhaki', 'goldenrod', 'darkorchid', 'plum']

# all subject 
x = np.arange(len(subjects))  # 被験者ごとの位置
width = 0.3  # 棒の幅

#R index va Lindex
plt.figure(figsize=(12, 5))
plt.bar(x - width/2, eeg_acc_right_left, width, color=colors[1],edgecolor='black')
plt.bar(x + width/2, current_acc_right_left, width, color=colors[2],edgecolor='black')

plt.xticks(x, subjects)
#plt.title('all subject ( each subject ) ')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend(loc='upper right',bbox_to_anchor=(1.21, 1))
plt.subplots_adjust(right=0.8)
plt.axhline(y=50, color='black', linestyle='--', linewidth=1.5)

#plt.show()

plt.savefig(os.path.join(save_dir,'Rindex_Lindex_all_sub.png'),bbox_inches='tight')

#R index va R little

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, eeg_acc_index_little, width, color=colors[1],edgecolor='black')
plt.bar(x + width/2, current_acc_index_little, width, color=colors[2],edgecolor='black')


plt.xticks(x, subjects)

#plt.title('all subject ( each subject ) ')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend(loc='upper right',bbox_to_anchor=(1.21, 1))
plt.subplots_adjust(right=0.8)
plt.axhline(y=50, color='black', linestyle='--', linewidth=1.5)

#plt.show()

plt.savefig(os.path.join(save_dir,'Rindex_Rlittle_all_sub.png'),bbox_inches='tight')