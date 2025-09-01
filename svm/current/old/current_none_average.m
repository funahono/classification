%paradigm
tasks = {'right_index', 'left_index','right_little'};
t_range = [0 3];
f_range = [11 13];
t_paradigm = [-3 7]; 

%read filename
erds_data_dir_filename = "/media/hdd1/Funatsuki/MATLAB/vbmeg_analysis/20241111_B92_right_index_20241112_ear_ref_car_standard_brain/tf_map_n_cycles_15_ref_-3_-1";

%read file type
matfilename = "_0_01.mat";

for task_idx = 1:length(tasks)
    task = tasks{task_idx}; % 現在のタスクを取得
    disp(['----------------------[ ' task '  ]----------------------']);

    % 初期化
    all_erds_data_mean = []; % 結果を保存する行列
    
    for channel_idx = 1:length(eeg_channel)
        channel = eeg_channel(channel_idx);
        channel = channel';  % 20×1の列ベクトルに転置
        
        % none.mat のパスを指定
        file_path = fullfile(erds_data_dir_filename, task, channel, matfile);
        if exist(file_path, 'file') % ファイルが存在するかチェック
            % データ読み込み
            data = load(file_path);
            % none.mat を読み込む
            erds_data = data.erds_data;  % サイズ (60, 1, 34, 25001)
            [n_epochs, n_channels, n_freqs, n_times] = size(erds_data);
            disp(channel)
            
            % 周波数（3次元目）
            f_index = f_range - 5; % 6Hz から時間周波数解析してるため -5
            erds_data_f_selected = erds_data(:, :, f_index(1):f_index(2), :);
    
            % 時間(4次元目)
            sampling_rate=(n_times-1)/(t_paradiam(1)+t_paradgm(2))
            t_index = round((t_range - t_paradigm(1)) * sampling_rate) + 1; % インデックス補正
            erds_data_ft_selected = erds_data_f_selected(:, :, :, t_index(1):t_index(2));

            %平均を計算する
            erds_data_mean = squeeze(mean(mean(erds_data_ft_selected, 3), 4)); % サイズ (60, 1)
            
            % 結果を保存
            all_erds_data_mean = [all_erds_data_mean; erds_data_mean']; % 縦方向に追加

            if strcmp(task, 'right_index')
                right_index_mean = all_erds_data_mean;
            elseif strcmp(task, 'left_index')
                 left_index_mean = all_erds_data_mean;
            elseif strcmp(task, 'right_little')
                 right_little_mean = all_erds_data_mean;
            end
           
        else
            %disp(['File not found: ', file_path]);
        end
    end

end

save_path = fullfile(erds_data_dir_filename,'mean.mat');
save(save_path, 'right_index_mean','left_index_mean','right_little_mean','eeg_channel');