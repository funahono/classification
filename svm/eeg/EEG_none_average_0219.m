%eeg_channel
eeg_channel = ["T7","C5","C3","C1","Cz","C2","C4","C6","T8","TP9","TP7","CP5","CP3","CP1","CPz","CP2","CP4","CP6","TP8","TP10"];
%eeg_channel = ["Fp1","Fp2","AF7","AF3","AFz","AF4","AF8","F7","F5","F3","F1","Fz","F2","F4","F6","F8","FT9","FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8","FT10","T7","C5","C3","C1","Cz","C2","C4","C6","T8","TP9","TP7","CP5","CP3","CP1","CPz","CP2","CP4","CP6","TP8","TP10","P7","P5","P3","P1","Pz","P2","P4","P6","P8","PO7","PO3","POz","PO4","PO8","O1","Oz","O2"];

comment ='Only_C_CP';

%read filename
erds_data_dir_filename = "/media/honoka/HDD1/Funatsuki/Experiment/20250206_B92/result";
save_data_dir_filename = "/media/honoka/HDD1/Funatsuki/Experiment/20250206_B92/mean";
% erds_data_dir_filename = "/media/honoka/HDD2/Experiment/20250313_B93/result";
% save_data_dir_filename = "/media/honoka/HDD2/Experiment/20250313_B93/mean";

%paradigm
tasks = {'Rindex', 'Lindex','Rlittle'}; %'Rindex', 'Lindex','Rlittle'
t_range = [0 3];
f_range = [12 14];
t_paradigm = [-3 7]; 
sampling_rate = 2500;

%read file type
matfile = "none.mat";

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
            t_index = round((t_range - t_paradigm(1)) * sampling_rate) + 1; % インデックス補正
            erds_data_ft_selected = erds_data_f_selected(:, :, :, t_index(1):t_index(2));

            %平均を計算する
            erds_data_mean = squeeze(mean(mean(erds_data_ft_selected, 3), 4)); % サイズ (60, 1)
            
            % 結果を保存
            all_erds_data_mean = [all_erds_data_mean; erds_data_mean']; % 縦方向に追加

            if strcmp(task, 'Rindex')
                right_index_mean = all_erds_data_mean;
            elseif strcmp(task, 'Lindex')
                left_index_mean = all_erds_data_mean;
            elseif strcmp(task, 'Rlittle')
                right_little_mean = all_erds_data_mean;
            end
           
        else
            disp(['File not found: ', file_path]);
        end
    end

end

save_path = fullfile(save_data_dir_filename,sprintf('mean_EEG_%s_%sHz_%s_%ss_%s.mat',num2str(f_range(1)),num2str(f_range(2)),num2str(t_range(1)),num2str(t_range(2)),comment));
save(save_path, 'right_index_mean','left_index_mean','right_little_mean','eeg_channel','n_epochs', 'n_channels', 'n_freqs', 'n_times','t_range','f_range','t_paradigm'); %'Rindex_mean','Lindex_mean','Rlittle_mean'