import umap
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# select dir
day_sub = '20241223_B90'
f_range =[12,15]
t_range =[0,3]

scaler = StandardScaler()
metric_type = 'correlation'

colors = ['red', 'blue', 'green']
class_names = ['Rindex', 'Lindex', 'Rlittle']

# dir
read_dir = f'/media/honoka/HDD1/Funatsuki/Experiment/{day_sub}/mean'
# dir
save_dir = f'/media/honoka/HDD1/Funatsuki/Experiment/{day_sub}/svm_3class/umap_metric:{metric_type}'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#path
read_file_name = f'mean_current_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s'
read_file_path = os.path.join(read_dir,read_file_name)

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

#sort data in chronological order and set label(Rindex:0,Lindex;1,Rlittle:2)
data = np.zeros((20*3*class_num, 359)) #(60 × 3 class, 359 vertexs)
labels = np.zeros(20*3*class_num)

for s in range(3):
    for c, class_data in enumerate(class_datasets):  # クラス番号とデータ
        start_idx = (s * class_num + c) * 20
        end_idx = start_idx + 20
        data[start_idx:end_idx, :] = class_data[s*20:(s+1)*20, :]
        labels[start_idx:end_idx] = c

data = scaler.fit_transform(data)

for n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):

    reducer=umap.UMAP(random_state=45,
                        n_neighbors=n,
                        n_components=2,#圧縮したい次元数。2なら2次元、3なら3次元に圧縮される。
                        min_dist=0.8,
                        metric=metric_type,
                        learning_rate=1, #default:1
                        spread=1, #default:1
                        #set_op_mix_ratio=0.5,
                        init='spectral'
                        )

    embedding = reducer.fit_transform(data)

    score = silhouette_score(embedding, labels, metric='euclidean')
    print(f'n_neighbors={n}, silhouette_score={score:.3f}')

    combined = np.hstack((labels.reshape(-1, 1), embedding))
    # print(combined)

    plt.figure(figsize=(8, 6))

    for class_id in np.unique(combined[:, 0]):
        class_id = int(class_id)
        subset = combined[combined[:, 0] == class_id]
        plt.scatter(subset[:, 1], subset[:, 2],
                    c=colors[class_id],
                    label=class_names[class_id], alpha=0.7)

    plt.title(f'UMAP Projection (neighbor = {n}) : score = {score:.3f}')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f'3class_umap_n={n}.png'),bbox_inches='tight')