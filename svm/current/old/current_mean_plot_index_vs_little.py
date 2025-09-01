################################### import ####################################
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat

######################### information ##########################################
# 作業ディレクトリを指定
day_sub = '20241223_B90'
f_range =[12,15]
t_range =[0,3]
comment ='Only_C_CP'

fontsize=20
right_little_data_color = 'blue'
right_index_data_color = 'red'

labels_top = ["T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8"]
labels_bottom = ["TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10"]

######################### dir & path ###########################################

# dir
read_dir = f'/media/hdd1/Funatsuki/Experiment/{day_sub}/mean'

save_dir = f'/media/hdd1/Funatsuki/Experiment/{day_sub}/svm/right_index_vs_right_little'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
#path
read_file_name = f'mean_EEG_{f_range[0]}_{f_range[1]}Hz_{t_range[0]}_{t_range[1]}s_{comment}'
read_file_path = os.path.join(read_dir,read_file_name)

#load data
data_dict = loadmat(read_file_path)
right_little_data = data_dict['right_little_mean']
right_index_data = data_dict['right_index_mean']

#transform to numpy 
right_little_data = np.array(right_little_data)
right_index_data = np.array(right_index_data)

################################################### Find min and max values ##############################################################
# Find the minimum and maximum values for the y-axis
min_value = min(np.min(right_little_data), np.min(right_index_data))  # 最小値
max_value = max(np.max(right_little_data), np.max(right_index_data))  # 最大値

# Expand the range slightly to give some padding
padding = (max_value - min_value) * 0.05  # 5% padding
min_value -= padding
max_value += padding

##################################################right_little data and right_index data plot##########################################################
fig, axes = plt.subplots(2, 1, figsize=(20, 12)) 
# Plot top (right_little_data and right_index_data for labels_top)
x_positions_top = np.arange(len(labels_top))  # 上段のラベル数だけ
for i, x in enumerate(x_positions_top):
    y_values = right_little_data[i]  # 各ラベルの値 (60個)
    x_values = np.full_like(y_values, x)  # 各値に対応する横軸位置
    axes[0].scatter(x_values, y_values, color=right_little_data_color, alpha=0.3)

for i, x in enumerate(x_positions_top):
    y_values = right_index_data[i]  # 各ラベルの値 (60個)
    x_values = np.full_like(y_values, x)  # 各値に対応する横軸位置
    axes[0].scatter(x_values, y_values, color=right_index_data_color, alpha=0.3)

# Add horizontal dashed line at y=0 for the top plot
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)

# Customize top plot appearance
axes[0].set_xticks(x_positions_top)
axes[0].set_xticklabels(labels_top, rotation=0, fontsize=fontsize)
#axes[0].set_xlabel('Channel', fontsize=fontsize)
axes[0].set_ylabel('ERD/S', fontsize=fontsize)
axes[0].set_title('ERD/S Average Plot', fontsize=fontsize)
axes[0].tick_params(axis='y', labelsize=fontsize)
 
# Plot bottom (right_little_data and right_index_data for labels_bottom)
x_positions_bottom = np.arange(len(labels_bottom))  # 下段のラベル数だけ
for i, x in enumerate(x_positions_bottom):
    y_values = right_little_data[i + len(labels_top)]  # 下段のラベルの値 (60個)
    x_values = np.full_like(y_values, x)  # 各値に対応する横軸位置
    axes[1].scatter(x_values, y_values, color=right_little_data_color, alpha=0.3)

for i, x in enumerate(x_positions_bottom):
    y_values = right_index_data[i + len(labels_top)]  # 下段のラベルの値 (60個)
    x_values = np.full_like(y_values, x)  # 各値に対応する横軸位置
    axes[1].scatter(x_values, y_values, color=right_index_data_color, alpha=0.3)

# Add horizontal dashed line at y=0 for the bottom plot
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)

# Customize bottom plot appearance
axes[1].set_xticks(x_positions_bottom)
axes[1].set_xticklabels(labels_bottom, rotation=0, fontsize=fontsize)
#axes[1].set_xlabel('Channel', fontsize=fontsize)
axes[1].set_ylabel('ERD/S', fontsize=fontsize)
axes[1].tick_params(axis='y', labelsize=fontsize) 
#axes[1].set_title('ERD/S Average Plot (Bottom Channels)', fontsize=fontsize)

# Add legend to the top plot
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=right_little_data_color, markersize=10, label='right_little_data')
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=right_index_data_color, markersize=10, label='right_index_data')
axes[0].legend(handles=[blue_patch, red_patch], bbox_to_anchor=(0.5, 1.15),loc='center left', borderaxespad=0, ncol=2, fontsize=fontsize)

# Set fixed y-axis limits for both plots based on the calculated min and max
axes[0].set_ylim(-2,4)
axes[1].set_ylim(-2,4)
plt.tight_layout()  # レイアウトを調整
plt.show()

# 保存 (JPG形式)
file_path_jpg_1 = os.path.join(save_dir, "mean_plot_right_little_and_right_index.jpg")
plt.savefig(file_path_jpg_1, format='jpg', dpi=300)  # 解像度を300dpiに設定


################################################## only right_little data plot ##########################################################
fig, axes = plt.subplots(2, 1, figsize=(20,12)) 
# Plot top (right_little_data for labels_top)
x_positions_top = np.arange(len(labels_top))  # 上段のラベル数だけ
for i, x in enumerate(x_positions_top):
    y_values = right_little_data[i]  # 各ラベルの値 (60個)
    x_values = np.full_like(y_values, x)  # 各値に対応する横軸位置
    axes[0].scatter(x_values, y_values, color=right_little_data_color, alpha=0.3)

# Add horizontal dashed line at y=0 for the top plot
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)

# Customize top plot appearance
axes[0].set_xticks(x_positions_top)
axes[0].set_xticklabels(labels_top, rotation=0, fontsize=fontsize)
#axes[0].set_xlabel('Channel', fontsize=fontsize)
axes[0].set_ylabel('ERD/S', fontsize=fontsize)
axes[0].tick_params(axis='y', labelsize=fontsize) 
axes[0].set_title('ERD/S Average Plot', fontsize=fontsize)

# Plot bottom (right_little_data for labels_bottom)
x_positions_bottom = np.arange(len(labels_bottom))  # 下段のラベル数だけ
for i, x in enumerate(x_positions_bottom):
    y_values = right_little_data[i + len(labels_top)]  # 下段のラベルの値 (60個)
    x_values = np.full_like(y_values, x)  # 各値に対応する横軸位置
    axes[1].scatter(x_values, y_values, color=right_little_data_color, alpha=0.3)

# Add horizontal dashed line at y=0 for the bottom plot
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)

# Customize bottom plot appearance
axes[1].set_xticks(x_positions_bottom)
axes[1].set_xticklabels(labels_bottom, rotation=0, fontsize=fontsize)
#axes[1].set_xlabel('Channel', fontsize=fontsize)
axes[1].set_ylabel('ERD/S', fontsize=fontsize)
axes[1].tick_params(axis='y', labelsize=fontsize) 
#axes[1].set_title('ERD/S Average Plot (Bottom Channels)', fontsize=fontsize)

# Set fixed y-axis limits for both plots based on the calculated min and max
axes[0].set_ylim(-2,4)
axes[1].set_ylim(-2,4)
plt.tight_layout()  # レイアウトを調整
plt.show()

# 保存 (JPG形式)
file_path_jpg_2 = os.path.join(save_dir, "mean_plot_only_right_little.jpg")
plt.savefig(file_path_jpg_2, format='jpg', dpi=300)  # 解像度を300dpiに設定


################################################## only  right_index data plot ##########################################################
fig, axes = plt.subplots(2, 1, figsize=(20,12)) 
# Plot top (right_index_data for labels_top)
for i, x in enumerate(x_positions_top):
    y_values = right_index_data[i]  # 各ラベルの値 (60個)
    x_values = np.full_like(y_values, x)  # 各値に対応する横軸位置
    axes[0].scatter(x_values, y_values, color=right_index_data_color, alpha=0.3)

# Add horizontal dashed line at y=0 for the top plot
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)

# Customize top plot appearance
axes[0].set_xticks(x_positions_top)
axes[0].set_xticklabels(labels_top, rotation=0, fontsize=fontsize)
#axes[0].set_xlabel('Channel', fontsize=fontsize)
axes[0].set_ylabel('ERD/S', fontsize=fontsize)
axes[0].set_title('ERD/S Average Plot', fontsize=fontsize)

# Plot bottom (right_index_data for labels_bottom)
for i, x in enumerate(x_positions_bottom):
    y_values = right_index_data[i + len(labels_top)]  # 下段のラベルの値 (60個)
    x_values = np.full_like(y_values, x)  # 各値に対応する横軸位置
    axes[1].scatter(x_values, y_values, color=right_index_data_color, alpha=0.3)

# Add horizontal dashed line at y=0 for the bottom plot
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)

# Customize bottom plot appearance
axes[1].set_xticks(x_positions_bottom)
axes[1].set_xticklabels(labels_bottom, rotation=0, fontsize=fontsize)
#axes[1].set_xlabel('Channel', fontsize=fontsize)
axes[1].set_ylabel('ERD/S', fontsize=fontsize)
#axes[1].set_title('ERD/S Average Plot (Bottom Channels)', fontsize=fontsize)

# Set fixed y-axis limits for both plots based on the calculated min and max
axes[0].set_ylim(-2,4)
axes[1].set_ylim(-2,4)
plt.tight_layout()  # レイアウトを調整
plt.show()

# 保存 (JPG形式)
file_path_jpg_3 = os.path.join(save_dir, "mean_plot_only_right_index.jpg")
plt.savefig(file_path_jpg_3, format='jpg', dpi=300)  # 解像度を300dpiに設定


######################### Boxplot for right_little_data and right_index_data ###########################################
fig, axes = plt.subplots(2, 1, figsize=(20, 12))  # 2行1列のサブプロット

# Boxplot for top (right_little_data and right_index_data for labels_top)
# right_little_data: red, right_index_data: blue
axes[0].boxplot([right_little_data[i] for i in range(len(labels_top))], positions=np.arange(len(labels_top)), widths=0.3, showmeans=True,
                boxprops=dict(color=right_little_data_color), whiskerprops=dict(color=right_little_data_color), capprops=dict(color=right_little_data_color),
                flierprops=dict(markerfacecolor='none', marker='o', markersize=5), meanprops=dict(marker='+', color=right_index_data_color, markersize=10),
                showfliers=False)  # 外れ値を表示しない

axes[0].boxplot([right_index_data[i] for i in range(len(labels_top))], positions=np.arange(len(labels_top)) + 0.4, widths=0.3, showmeans=True,
                boxprops=dict(color=right_index_data_color), whiskerprops=dict(color=right_index_data_color), capprops=dict(color=right_index_data_color),
                flierprops=dict(markerfacecolor='none', marker='o', markersize=5), meanprops=dict(marker='+', color=right_index_data_color, markersize=10),
                showfliers=False)  # 外れ値を表示しない

# Customize top plot appearance
axes[0].set_xticks(np.arange(len(labels_top)) + 0.2)
axes[0].set_xticklabels(labels_top, rotation=0, fontsize=fontsize)  # x軸ラベルのフォントサイズ
axes[0].set_ylabel('ERD/S', fontsize=fontsize)  # y軸ラベルのフォントサイズ
axes[0].set_title('Boxplot of ERD/S average', fontsize=fontsize)  # タイトルのフォントサイズ
axes[0].tick_params(axis='y', labelsize=fontsize) 
# Add horizontal dashed line at y=0 for the bottom plot
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)

# Boxplot for bottom (right_little_data and right_index_data for labels_bottom)
axes[1].boxplot([right_little_data[i + len(labels_top)] for i in range(len(labels_bottom))], positions=np.arange(len(labels_bottom)), widths=0.3, showmeans=True,
                boxprops=dict(color=right_little_data_color), whiskerprops=dict(color=right_little_data_color), capprops=dict(color=right_little_data_color),
                flierprops=dict(markerfacecolor='none', marker='o', markersize=5), meanprops=dict(marker='+', color=right_little_data_color, markersize=10),
                showfliers=False)  # 外れ値を表示しない

axes[1].boxplot([right_index_data[i + len(labels_top)] for i in range(len(labels_bottom))], positions=np.arange(len(labels_bottom)) + 0.4, widths=0.3, showmeans=True,
                boxprops=dict(color=right_index_data_color), whiskerprops=dict(color=right_index_data_color), capprops=dict(color=right_index_data_color),
                flierprops=dict(markerfacecolor='none', marker='o', markersize=5), meanprops=dict(marker='+', color=right_index_data_color, markersize=10),
                showfliers=False)  # 外れ値を表示しない

# Customize bottom plot appearance
axes[1].set_xticks(np.arange(len(labels_bottom)) + 0.2)
axes[1].set_xticklabels(labels_bottom, rotation=0, fontsize=fontsize)  # x軸ラベルのフォントサイズ
axes[1].set_ylabel('ERD/S', fontsize=fontsize)  # y軸ラベルのフォントサイズ
#axes[1].set_title('Boxplot of ERD/S (Bottom Channels)', fontsize=fontsize)  # タイトルのフォントサイズ
axes[1].tick_params(axis='y', labelsize=fontsize) 
# Add horizontal dashed line at y=0 for the bottom plot
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)

# Add legend to the top plot
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=right_little_data_color, markersize=10, label='right_little_data')
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=right_index_data_color, markersize=10, label='right_index_data')
axes[0].legend(handles=[blue_patch, red_patch], bbox_to_anchor=(0.5, 1.15),loc='center left', borderaxespad=0, ncol=2, fontsize=fontsize)

axes[0].set_ylim(-2,4)
axes[1].set_ylim(-2,4)

plt.tight_layout()  # レイアウトを調整
plt.show()

# 保存 (JPG形式)
file_path_jpg_4 = os.path.join(save_dir, "mean_boxplot_right_little_and_right_index.jpg")
plt.savefig(file_path_jpg_4, format='jpg', dpi=300)  # 解像度を300dpiに設定
