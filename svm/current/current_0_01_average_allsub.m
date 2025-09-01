% 各被験者の日付・条件リスト
day_sub_list = {
    '20241111_B92'
    '20241115_B94'
    '20241121_B99'
    '20241122_C02'
    '20241126_B95'
    '20241223_B90'
    '20250107_B97'
    % '20250206_B92'
    % '20250313_B93'
};

% 各被験者に対応した周波数範囲（例：12〜14Hzなど）
f_range_list = [
    12,14;
    10,13;
    12,13;
    12,14;
    12,14;
    12,15;
    14,15
    % 12,14
    % 13,15
];

% 各被験者に対応した時間範囲（例：0〜3秒など）
t_range_list = [
    2,3;
    0,3;
    1,2;
    0,3;
    0,3;
    0,3;
    0,3
    % 0,3
    % 0,3
];

vbmeg_analysis_dates_list = {
    {'20241112','20241112','20241112'} % 20241111_B92
    {'20241119','20241119','20241119'} % 20241115_B94
    {'20241123','20241123','20241123'} % 20241121_B99
    {'20241124','20241124','20241124'} % 20241122_C02
    {'20241206','20241206','20241226'} % 20241126_B95
    {'20241226','20241226','20241226'} % 20241223_B90
    {'20250109','20250109','20250109'} % 20250107_B97
    % {'20250313','20250305','20250305'} % 20250206_B92
};

% 共通のパラメータ
tasks = {'right_index', 'left_index', 'right_little'};
t_paradigm = [-3 7];
load_base_dir = '/media/honoka/HDD1/MATLAB/vbmeg_analysis/';
save_dir = '/media/honoka/HDD1/Experiment';

% 各被験者に対して処理を繰り返す
for day_idx = 1:length(day_sub_list)
    day_sub = day_sub_list{day_idx};
    f_range = f_range_list(day_idx, :);
    t_range = t_range_list(day_idx, :);
    current_days = vbmeg_analysis_dates_list{day_idx};


    % 各タスク（右人差し指など）に対して処理
    for task_idx = 1:length(tasks)
        task = tasks{task_idx};
        current_day = current_days{task_idx};
        disp(['----------------------[ ' task '  ]----------------------']);

        % フォルダパスの設定
        load_dir = fullfile(load_base_dir, sprintf('%s_%s_%s_ear_ref_car_standard_brain', day_sub, task, current_day), 'tf_map_n_cycles_15');

        % フォルダ内のサブディレクトリを取得
        dir_kinds = dir(load_dir);
        dir_list = [];
        for i = 1:length(dir_kinds)
            if dir_kinds(i).isdir
                num = str2double(dir_kinds(i).name);
                if ~isnan(num)
                    dir_list = [dir_list; num];
                end
            end
        end
        dir_list = sort(dir_list);
        dir_num = length(dir_list);

        % 平均値を保存する変数
        all_erds_data_mean = [];

        % 各サブディレクトリ内のファイルを処理
        for i = 1:dir_num
            now_dir = dir_list(i);
            fprintf('[%s] %d / %d：%d\n', task, i, dir_num, now_dir);
            now_file_name = sprintf('%d_0_01.mat', now_dir);
            now_load_path = fullfile(load_dir, num2str(now_dir), now_file_name);

            data = load(now_load_path);
            erds_data = data.erds_data;
            [n_epochs, n_channels, n_freqs, n_times] = size(erds_data);

            % 周波数のインデックスを選ぶ（6Hz開始なので -5 してる）
            f_index = f_range - 5;
            erds_data_f_selected = erds_data(:, :, f_index(1):f_index(2), :);

            % 時間のインデックスを計算
            sampling_rate = (n_times - 1) / (t_paradigm(2) - t_paradigm(1));
            t_index = round((t_range - t_paradigm(1)) * sampling_rate) + 1;
            erds_data_ft_selected = erds_data_f_selected(:, :, :, t_index(1):t_index(2));

            % 平均値の計算（周波数と時間で平均）
            erds_data_mean = squeeze(mean(mean(erds_data_ft_selected, 3), 4)); % size: (n_epochs, n_channels)
            all_erds_data_mean = [all_erds_data_mean; erds_data_mean'];

            % タスク別に保存
            if strcmp(task, 'Rindex')
                Rindex_mean = all_erds_data_mean;
            elseif strcmp(task, 'Lindex')
                Lindex_mean = all_erds_data_mean;
            elseif strcmp(task, 'Rlittle')
                Rlittle_mean = all_erds_data_mean;
            end
        end
    end

    % 平均値を保存
    save_path = fullfile(save_dir, day_sub, 'mean', sprintf('mean_current_%s_%sHz_%s_%ss.mat',num2str(f_range(1)), num2str(f_range(2)),num2str(t_range(1)), num2str(t_range(2))));

    save(save_path, 'Rindex_mean', 'Lindex_mean', 'Rlittle_mean', 'n_epochs', 'n_channels', 'n_freqs', 'n_times', 't_range', 'f_range', 't_paradigm');
end
