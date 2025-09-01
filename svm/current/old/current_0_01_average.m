%paradigm
day_sub = '20250107_B97';
%current_days = {'20241110','20241110'};
current_day = '20250109';
tasks = {'right_index','left_index','right_little'};%'right_index', 'left_index','right_little'
t_range = [0 1];
f_range = [14 15];
t_paradigm = [-3 7]; 

load_base_dir = '/media/hdd1/Funatsuki/MATLAB/vbmeg_analysis/';
save_dir = '/media/hdd1/Funatsuki/Experiment';

for task_idx = 1:length(tasks)
    task = tasks{task_idx}; % 現在のタスクを取得
    %current_day=current_days{task_idx};
    disp(['----------------------[ ' task '  ]----------------------']);
    % load dir
    % load_dir = '/media/hdd1/Funatsuki/MATLAB/vbmeg_analysis/20241111_B92_right_index_20241112_ear_ref_car_standard_brain/tf_map_n_cycles_15_ref_-3_-1';
    load_dir = fullfile(load_base_dir, sprintf('%s_%s_%s_ear_ref_car_standard_brain', day_sub, task, current_day), 'tf_map_n_cycles_15');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check dir name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dir_kinds=dir(load_dir);
    dir_list = [];             % save dir name list

    for i = 1:length(dir_kinds)
        if dir_kinds(i).isdir % load only dir
            dir_name = dir_kinds(i).name; % dir name
            dir_list = [dir_list; str2double(dir_name)]; %add dir_name to dir_list
            
        end
    end
    dir_num = length(dir_list); %number of dir
    dir_list=sort(dir_list);
    % Remove NaN values from dir_list
    dir_list = dir_list(~isnan(dir_list));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% calculate average %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 結果を保存する行列
    all_erds_data_mean = []; 

    for i = 1:length(dir_list)
        now_dir = dir_list(i);
        fprintf('[%s] %d / %d：%d\n',task, i, dir_num,now_dir);
        now_file_name= sprintf('%d_0_01.mat',now_dir);
        % disp(now_dir);
        % disp(now_file_name);

        now_load_path = fullfile(load_dir,num2str(now_dir),now_file_name);
        
        % データ読み込み
        data = load(now_load_path);

        % % _0_01.mat を読み込む
        erds_data = data.erds_data; 
        [n_epochs, n_channels, n_freqs, n_times] = size(erds_data);
        % disp(n_epochs);
        % disp(n_channels);
        % disp(n_freqs);
        % disp(n_times);

        % 周波数（3次元目）
        f_index = f_range - 5; % 6Hz から時間周波数解析してるため -5
        erds_data_f_selected = erds_data(:, :, f_index(1):f_index(2), :);

        % 時間(4次元目)
        sampling_rate = (n_times - 1) / (t_paradigm(2) - t_paradigm(1)); %(sampling_rate) = [(num of all sumple)-1]/[7-(-3)]
        %disp(sampling_rate);
        t_index = round((t_range - t_paradigm(1)) * sampling_rate) + 1; % インデックス補正
        %disp(t_index)
        erds_data_ft_selected = erds_data_f_selected(:, :, :, t_index(1):t_index(2));

        % 平均を計算する
        erds_data_mean = squeeze(mean(mean(erds_data_ft_selected, 3), 4)); % サイズ (60, 1)
        
        % 結果を保存
        all_erds_data_mean = [all_erds_data_mean; erds_data_mean']; % 縦方向に追加

        % 特定のタスクに関連するデータを格納
        if strcmp(task, 'right_index')
           right_index_mean = all_erds_data_mean;
        elseif strcmp(task, 'left_index')
           left_index_mean = all_erds_data_mean;
        elseif strcmp(task, 'right_little')
            right_little_mean = all_erds_data_mean;
        end
    end
end

save_path = fullfile(save_dir,day_sub,'mean',sprintf('mean_current_%s_%sHz_%s_%ss.mat',num2str(f_range(1)),num2str(f_range(2)),num2str(t_range(1)),num2str(t_range(2))));
save(save_path, 'right_index_mean','left_index_mean','right_little_mean','n_epochs', 'n_channels', 'n_freqs', 'n_times','t_range','f_range','t_paradigm'); % 'right_index_mean','left_index_mean','right_little_mean'