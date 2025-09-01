import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# matrix データ
#list = [subject, Rindex vs lindex EEG , R index vs Lindex current, Rindex vs Rlittel EEG, Rindex vs Rlittle current]
matrix = np.array([ ['B96','79.17','93.33','0','0'],
                    ['B91','37.50','75.83','0','0'],
                    ['B93','65.00','80.00','0','0'],
                    ['B92_1','31.67','70.00','38.33','71.67'],
                    ['B94','90.00','83.33','66.67','73.33'],
                    ['B99','34.17','70.83','46.67','67.50'],
                    ['C02','80.00','85.00','80.83','92.50'],
                    ['B95', '78.33', '80.83','76.67','86.67'],
                    ['B90', '75.83', '86.67','83.33','90.00'],
                    ['B97', '59.17', '70.00','53.33','70.83'],
                    ['B92_2','46.67','64.17','50.00','65.00']])

save_dir = 'C:/Users/Hono/Desktop/datadir/svm/presentation/acc'

# データの抽出
subjects = matrix[:, 0]  # 被験者名
eeg_acc_right_left = matrix[:, 1].astype(float)  # 脳波データ R index vs Lindex
current_acc_right_left = matrix[:, 2].astype(float)  # 皮質電流データ R index vs Lindex
eeg_acc_index_little = matrix[:, 3].astype(float)  # 脳波データ R index vs Rlittle
current_acc_index_little = matrix[:, 4].astype(float)  # 皮質電流データ R index vs Rlittle
method=np.array(['EEG','cortical current'])

# ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 14
    
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



    
# 被験者ごとのグラフ
for i, subject in enumerate(subjects):
    
    plt.figure(figsize=(12, 5))
    
    if i <= 2:
        bars = plt.bar(['Rindex vs L index\n EEG','Rindex vs L index\n cortical current'],
                       [eeg_acc_right_left[i],current_acc_right_left[i]], 
                       color=colors[0],edgecolor='black')
    
    else:
        bars = plt.bar(['Rindex vs L index\n EEG', 'Rindex vs R little\n EEG','Rindex vs L index\n cortical current','Rindex vs Rlittle\n cortical current'], 
                       [eeg_acc_right_left[i],eeg_acc_index_little[i], current_acc_right_left[i], current_acc_index_little[i]], 
                       color=colors[0],edgecolor='black')
    
    #plt.title(f'{subject}')
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}%', ha='center', va='bottom',fontsize=12)
    plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5)
    plt.savefig(os.path.join(save_dir, f'{subject}_accuracy_2.png'), bbox_inches='tight')

    #plt.show()

# all subject 
x = np.arange(len(subjects))  # 被験者ごとの位置
width = 0.3  # 棒の幅

#R index va Lindex
plt.figure(figsize=(12, 5))
plt.bar(x - width/2, eeg_acc_right_left, width, label='EEG', color=colors[1],edgecolor='black')
plt.bar(x + width/2, current_acc_right_left, width, label='cortical\ncurrent', color=colors[2],edgecolor='black')

plt.xticks(x, subjects)
#plt.title('all subject ( each subject ) ')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend(loc='upper right',bbox_to_anchor=(1.21, 1))
plt.subplots_adjust(right=0.8)
plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5)

plt.show()

plt.savefig(os.path.join(save_dir,'Rindex_Lindex_all_sub_2.png'),bbox_inches='tight')

#R index va R little
eeg_acc_index_little=np.delete(eeg_acc_index_little, [0, 1, 2])
current_acc_index_little=np.delete(current_acc_index_little, [0, 1, 2])
subjects2=np.delete(subjects,[0,1,2])
x4 = np.arange(len(subjects2))

plt.figure(figsize=(12, 5))
plt.bar(x4 - width/2, eeg_acc_index_little, width, label='EEG', color=colors[3],edgecolor='black')
plt.bar(x4 + width/2, current_acc_index_little, width, label='cortical\ncurrent', color=colors[4],edgecolor='black')

plt.xticks(x4, subjects2)
#plt.title('all subject ( each subject ) ')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend(loc='upper right',bbox_to_anchor=(1.21, 1))
plt.subplots_adjust(right=0.8)
plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5)

plt.show()

plt.savefig(os.path.join(save_dir,'Rindex_Rlittle_all_sub_2.png'),bbox_inches='tight')
 
#all method
x2 = np.array([0,len(subjects)*width+0.5])

plt.figure(figsize=(12, 5))

#R index vs L index
for d , subject in enumerate(subjects):
    position = width*(d+0.5-len(subjects)*0.5)
    plt.bar(x2[0] + position, eeg_acc_right_left[d], width, label=subject, color=colors[d+5],edgecolor='black')
    plt.bar(x2[1] + position, current_acc_right_left[d], width, color=colors[d+5],edgecolor='black')
plt.xticks(x2, method)
#plt.title('all subject( each method )')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))
plt.subplots_adjust(right=0.85)
plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5)

plt.show()

plt.savefig(os.path.join(save_dir,'Rindex_Lindex_all_method_2.png'),bbox_inches='tight')

plt.figure(figsize=(12, 5))

# Rindex vs R little
x3 = np.array([0,len(eeg_acc_index_little)*width+0.5])
for d ,subject in enumerate(subjects2):
    position = width*(d+0.5-len(subjects2)*0.5)
    plt.bar(x3[0] + position, eeg_acc_index_little[d], width, label=subject, color=colors[d+5+3],edgecolor='black')
    plt.bar(x3[1] + position, current_acc_index_little[d], width, color=colors[d+5+3],edgecolor='black')
plt.xticks(x3, method)
#plt.title('all subject( each method )')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))
plt.subplots_adjust(right=0.85)
plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5)

plt.show()

plt.savefig(os.path.join(save_dir,'Rindex_Rlittle_all_method_2.png'),bbox_inches='tight')

#average all accuracy
ave_eeg_acc_index_little =sum(eeg_acc_index_little)/len(eeg_acc_index_little)
ave_eeg_acc_right_left=sum(eeg_acc_right_left)/len(eeg_acc_right_left)
ave_current_acc_index_little=sum(current_acc_index_little)/len(current_acc_index_little)
ave_current_acc_right_left=sum(current_acc_right_left)/len(current_acc_right_left)

plt.figure(figsize=(12, 5))
bars = plt.bar(['Rindex vs L index\n EEG', 'Rindex vs R little\n EEG','Rindex vs L index\n cortical current','Rindex vs Rlittle\n cortical current'], 
               [ave_eeg_acc_right_left,ave_eeg_acc_index_little, ave_current_acc_right_left, ave_current_acc_index_little], 
               color=colors[0],edgecolor='black')

plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}%', ha='center', va='bottom')

plt.axhline(y=50, color='red', linestyle='--', linewidth=1.5)

plt.savefig(os.path.join(save_dir, 'average_accuracy.png'), bbox_inches='tight')