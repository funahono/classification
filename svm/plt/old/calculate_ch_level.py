# -*- coding: utf-8 -*-
"""
Random Label Classification Experiment (1000 iterations)
Created on 2025/04/16
@author: Hono
"""

################################### import ####################################
from scipy.io import loadmat
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd

######################### 設定 ##########################################

# データ情報
day_sub = '20250206_B92'
f_range = [12, 14]
t_range = [0, 3]

# SVM設定
k_num = 5
kernel_type = 'linear'
multi_classfication_type = 'ovo'  # 'ovr' or 'ovo'

# ディレクトリ
read_dir = f'/media/honoka/HDD1/Funatsuki/Experiment/{day_sub}/mean'
save_dir = f'/media/honoka/HDD1/Funatsuki/Experiment/{day_sub}/svm_3class/random_label_k={k_num}_{kernel_type}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ファイルパス
read_file_name = f'mean_current_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s'
read_file_path = os.path.join(read_dir, read_file_name)

######################### データ読み込み ####################################

data_dict = loadmat(read_file_path)
Rlittle_data = data_dict['right_little_mean']
Rindex_data = data_dict['right_index_mean']
Lindex_data = data_dict['left_index_mean']

# 転置とまとめ
class_datasets = [
    np.transpose(np.array(Rindex_data), (1, 0)),
    np.transpose(np.array(Lindex_data), (1, 0)),
    np.transpose(np.array(Rlittle_data), (1, 0))
]
class_num = len(class_datasets)

data = np.zeros((20 * 3 * class_num, 359))  # (60 × 3 クラス, 359頂点)
labels = np.zeros(20 * 3 * class_num)

for s in range(3):
    for c, class_data in enumerate(class_datasets):
        start_idx = (s * class_num + c) * 20
        end_idx = start_idx + 20
        data[start_idx:end_idx, :] = class_data[s*20:(s+1)*20, :]
        labels[start_idx:end_idx] = c

# 標準化
scaler = StandardScaler()
data = scaler.fit_transform(data)

###################### ランダムラベル実験（1000回） #########################

num_iterations = 1000
random_accs = []
random_fscores = []

original_labels = labels.copy()  # 元のラベル保持

np.random.seed(40)  # 再現性のため

for i in range(num_iterations):
    # ラベルシャッフル
    shuffled_labels = original_labels.copy()
    np.random.shuffle(shuffled_labels)

    skf = StratifiedKFold(k_num)
    acc_list = []
    fscore_list = []

    for train_idx, test_idx in skf.split(data, shuffled_labels):
        train_data = data[train_idx]
        test_data = data[test_idx]
        train_labels = shuffled_labels[train_idx]
        test_labels = shuffled_labels[test_idx]

        model = SVC(kernel=kernel_type, decision_function_shape=multi_classfication_type)
        model.fit(train_data, train_labels)
        pred = model.predict(test_data)

        acc = np.mean(pred == test_labels) * 100
        f1 = f1_score(test_labels, pred, average='weighted') * 100

        acc_list.append(acc)
        fscore_list.append(f1)

    # 1回分の平均精度・Fスコア
    random_accs.append(np.mean(acc_list))
    random_fscores.append(np.mean(fscore_list))

    if i % 100 == 0:
        print(f"Random shuffle {i} / {num_iterations}")

# 平均を出力
mean_random_acc = np.mean(random_accs)
mean_random_f1 = np.mean(random_fscores)

print("\n========== ランダムラベル 1000回の結果 ==========")
print(f"Average Accuracy (Random): {mean_random_acc:.2f} %")
print(f"Average F1 Score (Random): {mean_random_f1:.2f} %")

# ###################### 結果保存（CSV） #####################################

# save_random_csv_path = os.path.join(save_dir, 'random_label_1000_results.csv')
# pd.DataFrame({
#     'accuracy': random_accs,
#     'f1_score': random_fscores
# }).to_csv(save_random_csv_path, index=False)

# # 平均も別ファイルで保存
# with open(os.path.join(save_dir, 'random_label_1000_summary.txt'), 'w') as f:
#     f.write(f"Average Accuracy (Random Labels): {mean_random_acc:.2f} %\n")
#     f.write(f"Average F1 Score (Random Labels): {mean_random_f1:.2f} %\n")
