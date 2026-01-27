%% 主程序 - 生成训练集和测试集
% 添加执行保护机制
train_save_dir = 'E:\xxbDigitalJamTrainset1000';
test_save_dir = 'E:\xxbDigitalJamTestset1000';

% 检查是否已存在数据集
if exist(train_save_dir, 'dir') && ~isempty(dir(fullfile(train_save_dir, '*.png')))
    answer = questdlg('训练集已存在，是否重新生成？', '确认', '是', '否', '否');
    if strcmp(answer, '否')
        error('用户取消执行');
    else
        fprintf('删除旧训练集...\n');
        rmdir(train_save_dir, 's');
    end
end

if exist(test_save_dir, 'dir') && ~isempty(dir(fullfile(test_save_dir, '*.png')))
    answer = questdlg('测试集已存在，是否重新生成？', '确认', '是', '否', '否');
    if strcmp(answer, '否')
        error('用户取消执行');
    else
        fprintf('删除旧测试集...\n');
        rmdir(test_save_dir, 's');
    end
end

% 训练集参数 (±20% 频移)
train_fc_range = [800e6, 1200e6]; % 800-1200 MHz (±20%)

% 测试集参数 (±20% 频移) - 修改为±20%
test_fc_range = [800e6, 1200e6]; % 800-1200 MHz (±20%)

% 生成训练集
fprintf('===== 正在生成训练集 (±20%% 频移) =====\n');
generate_dataset_fixed(train_fc_range, train_save_dir, 'train');

% 生成测试集
fprintf('\n===== 正在生成测试集 (±20%% 频移) =====\n');
generate_dataset_fixed(test_fc_range, test_save_dir, 'test');

fprintf('\n数据集生成完成！\n');
fprintf('训练集保存至: %s\n', train_save_dir);
fprintf('测试集保存至: %s\n', test_save_dir);

% 验证生成数量
train_files = dir(fullfile(train_save_dir, '*.png'));
test_files = dir(fullfile(test_save_dir, '*.png'));
fprintf('验证: 训练集%d个样本, 测试集%d个样本, 总计%d个样本\n', ...
        numel(train_files), numel(test_files), numel(train_files) + numel(test_files));

%% 验证前100个样本中干扰类型出现次数
fprintf('\n=== 验证前100个样本中干扰类型出现次数 ===\n');
validate_jam_type_counts(train_save_dir, 100);

%% ===== 修复后的数据集生成函数 =====
function generate_dataset_fixed(fc_range, save_dir, dataset_type)
% 全局参数设置
fs = 2500e6;        % 采样频率 2.5 GHz
duration = 100e-6;  % 信号持续时间 100μs
total_samples = 1000; % 总样本数 - 严格限制

% 高分辨率STFT参数
window_length = 4096; % 汉明窗长度
noverlap = 3072;     % 重叠点数 (75%重叠)
nfft = 8192;         % FFT点数

% LFM信号基准参数
f0_center = 1000e6; % 中心频率 1000 MHz
mu = 5e11;          % 调频斜率 5e11 Hz/s

% 数字调制干扰参数
symbol_rate = 500e3; % 符号率 500 kbps
sps = 8;             % 每符号样本数
rolloff = 0.5;       % 升余弦滚降系数

% 测试集配置
single_ratio = 0.3; % 单类干扰占比 (30%)
mixed_ratio = 0.7;  % 混合干扰占比 (70%)

% 信号池大小 - 修改：将所有干扰类型的池大小统一为50个
N_S = 500;          % 目标信号池
N_FSK = 50;         % FSK干扰池 - 从750改为50
N_BPSK = 50;        % BPSK干扰池
N_QPSK = 50;        % QPSK干扰池
N_16QAM = 50;       % 16QAM干扰池

% 创建共享频移池
N_fc = 100;         % 频移小样本池大小
fc_min = fc_range(1);
fc_max = fc_range(2);
fc_pool = fc_min + (fc_max-fc_min)*rand(1, N_fc); % 自定义频移范围

% 增强1：为训练集添加扩展频移样本（±30%）
if strcmp(dataset_type, 'train')
    extended_fc_min = 700e6;  % 扩展到700MHz
    extended_fc_max = 1300e6; % 扩展到1300MHz
    extended_N = round(N_fc * 0.2); % 20%的扩展样本
    extended_pool = extended_fc_min + (extended_fc_max - extended_fc_min)*rand(1, extended_N);
    fc_pool = [fc_pool, extended_pool];
    fprintf('训练集增强: 添加%d个扩展频移样本(%.1f-%.1f MHz)\n', ...
            extended_N, extended_fc_min/1e6, extended_fc_max/1e6);
end

% 新的混合干扰组合类型（11种）
MIX_TYPES = {
    [1,2],      % FSK+BPSK 
    [1,3],      % FSK+QPSK
    [1,4],      % FSK+16QAM
    [2,3],      % BPSK+QPSK
    [2,4],      % BPSK+16QAM
    [3,4],      % QPSK+16QAM
    [1,2,3],    % FSK+BPSK+QPSK
    [1,2,4],    % FSK+BPSK+16QAM
    [1,3,4],    % FSK+QPSK+16QAM
    [2,3,4],    % BPSK+QPSK+16QAM
    [1,2,3,4]   % FSK+BPSK+QPSK+16QAM
};

% 混合类型占比（均匀分布）
MIX_RATIOS = ones(1, 11) / 11; % 每种类型占比约9.09%

% JSR范围 (-10dB~10dB)
JSR_range = -10:1:20; 

% FSK参数
fdev_range = [0.5e6, 2e6]; % 频偏范围 0.5-2 MHz
fdev_span = fdev_range(2) - fdev_range(1); % 预先计算差值

% 创建保存目录
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% 计算正确的信号长度
num_samples = round(fs * duration); % 信号总长度
t = (0:num_samples-1)/fs;          % 时间向量

% 打印当前数据集信息
fprintf('频移范围: %.1f-%.1f MHz\n', min(fc_pool)/1e6, max(fc_pool)/1e6);
fprintf('保存路径: %s\n', save_dir);
fprintf('信号长度: %d 点 (%.1f μs)\n', num_samples, duration*1e6);
fprintf('严格样本总数限制: %d\n', total_samples);

%% 预生成信号池（使用共享频移池）
fprintf('正在生成信号池（使用共享频移池，大小=%d）...\n', length(fc_pool));

% 目标信号池 (LFM信号)
S_pool = cell(1, N_S);
for i = 1:N_S
    % 从共享频移池随机选择载频
    rand_fc = fc_pool(randi(length(fc_pool)));
    
    % 引入其他随机变化
    rand_phase = 2*pi*rand(); 
    rand_mu = mu * (1 + 0.05*(2*rand()-1)); % ±5%调频斜率变化
    
    % 生成信号
    phase = 2*pi*rand_fc*t + pi*rand_mu*t.^2 + rand_phase;
    S_pool{i} = cos(phase);
end

% FSK干扰池
FSK_pool = cell(1, N_FSK);
for i = 1:N_FSK
    % 从共享频移池随机选择载频
    rand_fc = fc_pool(randi(length(fc_pool)));
    
    % 随机频偏
    fdev = fdev_range(1) + fdev_span * rand();
    
    % 生成随机符号序列
    num_symbols = ceil(duration * symbol_rate);
    symbols = randi([0, 1], 1, num_symbols);
    
    % 创建FSK调制器
    fskMod = comm.FSKModulator(...
        'ModulationOrder', 2, ...
        'FrequencySeparation', 2*fdev, ...
        'SamplesPerSymbol', sps, ...
        'SymbolRate', symbol_rate);
    
    % 生成基带信号
    baseband = fskMod(symbols');
    
    % 裁剪到正确长度
    if length(baseband) > num_samples
        baseband = baseband(1:num_samples);
    else
        % 补零到正确长度
        baseband = [baseband; zeros(num_samples-length(baseband), 1)];
    end
    
    % 上变频
    carrier = exp(1j*2*pi*rand_fc*t);
    FSK_pool{i} = real(baseband(:).' .* real(carrier));
end

% BPSK干扰池
BPSK_pool = cell(1, N_BPSK);
for i = 1:N_BPSK
    % 从共享频移池随机选择载频
    rand_fc = fc_pool(randi(length(fc_pool)));
    
    % 生成随机符号序列
    num_symbols = ceil(duration * symbol_rate);
    symbols = randi([0, 1], 1, num_symbols);
    
    % 创建BPSK调制器
    bpskMod = comm.BPSKModulator();
    pulseShape = comm.RaisedCosineTransmitFilter(...
        'RolloffFactor', rolloff, ...
        'FilterSpanInSymbols', 10, ...
        'OutputSamplesPerSymbol', sps);
    
    % 生成基带信号
    modData = bpskMod(symbols');
    baseband = pulseShape(modData);
    
    % 裁剪到正确长度
    if length(baseband) > num_samples
        baseband = baseband(1:num_samples);
    else
        baseband = [baseband; zeros(num_samples-length(baseband), 1)];
    end
    
    % 上变频
    carrier = exp(1j*2*pi*rand_fc*t);
    BPSK_pool{i} = real(baseband(:).' .* real(carrier));
end

% QPSK干扰池
QPSK_pool = cell(1, N_QPSK);
for i = 1:N_QPSK
    % 从共享频移池随机选择载频
    rand_fc = fc_pool(randi(length(fc_pool)));
    
    % 生成随机符号序列
    num_symbols = ceil(duration * symbol_rate);
    symbols = randi([0, 3], 1, num_symbols);
    
    % 创建QPSK调制器
    qpskMod = comm.PSKModulator(...
        'ModulationOrder', 4, ...
        'PhaseOffset', pi/4, ...
        'BitInput', false);
    pulseShape = comm.RaisedCosineTransmitFilter(...
        'RolloffFactor', rolloff, ...
        'FilterSpanInSymbols', 10, ...
        'OutputSamplesPerSymbol', sps);
    
    % 生成基带信号
    modData = qpskMod(symbols');
    baseband = pulseShape(modData);
    
    % 裁剪到正确长度
    if length(baseband) > num_samples
        baseband = baseband(1:num_samples);
    else
        baseband = [baseband; zeros(num_samples-length(baseband), 1)];
    end
    
    % 上变频
    carrier = exp(1j*2*pi*rand_fc*t);
    QPSK_pool{i} = real(baseband(:).' .* real(carrier));
end

% 16QAM干扰池
QAM16_pool = cell(1, N_16QAM);
for i = 1:N_16QAM
    % 从共享频移池随机选择载频
    rand_fc = fc_pool(randi(length(fc_pool)));
    
    % 生成随机符号序列
    num_symbols = ceil(duration * symbol_rate);
    symbols = randi([0, 15], 1, num_symbols);
    
    % 创建16QAM调制器
    qam16Mod = comm.RectangularQAMModulator(...
        'ModulationOrder', 16, ...
        'NormalizationMethod', 'Average power');
    pulseShape = comm.RaisedCosineTransmitFilter(...
        'RolloffFactor', rolloff, ...
        'FilterSpanInSymbols', 10, ...
        'OutputSamplesPerSymbol', sps);
    
    % 生成基带信号
    modData = qam16Mod(symbols');
    baseband = pulseShape(modData);
    
    % 裁剪到正确长度
    if length(baseband) > num_samples
        baseband = baseband(1:num_samples);
    else
        baseband = [baseband; zeros(num_samples-length(baseband), 1)];
    end
    
    % 上变频
    carrier = exp(1j*2*pi*rand_fc*t);
    QAM16_pool{i} = real(baseband(:).' .* real(carrier));
end

fprintf('信号池生成完成! 干扰池: FSK(%d), BPSK(%d), QPSK(%d), 16QAM(%d)\n', ...
         N_FSK, N_BPSK, N_QPSK, N_16QAM);
fprintf('所有信号使用共享频移池，大小: %d (频率范围: %.1f-%.1f MHz)\n', ...
        length(fc_pool), min(fc_pool)/1e6, max(fc_pool)/1e6);

%% 生成测试集样本 - 修复样本计数逻辑
fprintf('开始生成测试样本（总样本数: %d）...\n', total_samples);
sample_counter = 0;

% ===== 1. 生成单类干扰样本 (30%) =====
single_samples = round(single_ratio * total_samples);
jam_types = [1, 2, 3, 4]; % 1:FSK, 2:BPSK, 3:QPSK, 4:16QAM

% 修改：四种干扰各25%
samples_per_type = floor(single_samples / 4);
extra_samples = single_samples - 4 * samples_per_type;

% 分配余数给前几个类型
fsk_samples = samples_per_type + (extra_samples >= 1);
bpsk_samples = samples_per_type + (extra_samples >= 2);
qpsk_samples = samples_per_type + (extra_samples >= 3);
qam_samples = samples_per_type + (extra_samples >= 4);

fprintf('单类样本分布: FSK(%d), BPSK(%d), QPSK(%d), 16QAM(%d), 总计(%d)\n', ...
        fsk_samples, bpsk_samples, qpsk_samples, qam_samples, ...
        fsk_samples + bpsk_samples + qpsk_samples + qam_samples);

% FSK样本
for j = 1:fsk_samples
    sample_counter = sample_counter + 1;
    
    % 选择目标和干扰
    s_idx = randi(N_S);
    jam_signal = FSK_pool{randi(N_FSK)};
    jsr_val = JSR_range(randi(length(JSR_range)));
    
    % 生成时频图像
    img_data = generate_MIXED_image_fixed(...
        S_pool{s_idx}, ...  % 目标信号
        jam_signal, ...    % 干扰信号
        jsr_val, ...        % JSR值
        1, ...             % 干扰类型标识 (FSK)
        fs, ...             % 传递采样频率
        num_samples, ...    % 传递信号长度
        window_length, ...  % STFT参数
        noverlap, ...       % STFT参数
        nfft, ...           % STFT参数
        fc_pool, ...        % 频移池信息
        dataset_type);      % 数据集类型
    
    % 保存样本
    type_name = get_jam_type_name(1);
    filename = sprintf('SINGLE_%s_JSR%+03d_%05d.png', type_name, jsr_val, sample_counter);
    imwrite(img_data, fullfile(save_dir, filename));
    
    % 进度显示
    if mod(sample_counter, 1000) == 0
        fprintf('  已生成 %d/%d 样本\n', sample_counter, total_samples);
    end
end

% BPSK样本
for j = 1:bpsk_samples
    sample_counter = sample_counter + 1;
    
    s_idx = randi(N_S);
    jam_signal = BPSK_pool{randi(N_BPSK)};
    jsr_val = JSR_range(randi(length(JSR_range)));
    
    img_data = generate_MIXED_image_fixed(...
        S_pool{s_idx}, jam_signal, jsr_val, 2, fs, num_samples, ...
        window_length, noverlap, nfft, fc_pool, dataset_type);
    
    filename = sprintf('SINGLE_%s_JSR%+03d_%05d.png', 'BPSK', jsr_val, sample_counter);
    imwrite(img_data, fullfile(save_dir, filename));
    
    if mod(sample_counter, 1000) == 0
        fprintf('  已生成 %d/%d 样本\n', sample_counter, total_samples);
    end
end

% QPSK样本
for j = 1:qpsk_samples
    sample_counter = sample_counter + 1;
    
    s_idx = randi(N_S);
    jam_signal = QPSK_pool{randi(N_QPSK)};
    jsr_val = JSR_range(randi(length(JSR_range)));
    
    img_data = generate_MIXED_image_fixed(...
        S_pool{s_idx}, jam_signal, jsr_val, 3, fs, num_samples, ...
        window_length, noverlap, nfft, fc_pool, dataset_type);
    
    filename = sprintf('SINGLE_%s_JSR%+03d_%05d.png', 'QPSK', jsr_val, sample_counter);
    imwrite(img_data, fullfile(save_dir, filename));
    
    if mod(sample_counter, 1000) == 0
        fprintf('  已生成 %d/%d 样本\n', sample_counter, total_samples);
    end
end

% 16QAM样本
for j = 1:qam_samples
    sample_counter = sample_counter + 1;
    
    s_idx = randi(N_S);
    jam_signal = QAM16_pool{randi(N_16QAM)};
    jsr_val = JSR_range(randi(length(JSR_range)));
    
    img_data = generate_MIXED_image_fixed(...
        S_pool{s_idx}, jam_signal, jsr_val, 4, fs, num_samples, ...
        window_length, noverlap, nfft, fc_pool, dataset_type);
    
    filename = sprintf('SINGLE_%s_JSR%+03d_%05d.png', '16QAM', jsr_val, sample_counter);
    imwrite(img_data, fullfile(save_dir, filename));
    
    if mod(sample_counter, 1000) == 0
        fprintf('  已生成 %d/%d 样本\n', sample_counter, total_samples);
    end
end

fprintf('单类干扰样本完成: %d个\n', sample_counter);

% ===== 2. 生成混合干扰样本 (70%) =====
mixed_samples = round(mixed_ratio * total_samples);

% 计算每种混合类型的样本量
mix_sample_counts = round(mixed_samples * MIX_RATIOS);

% 确保总数正确（由于四舍五入可能略有偏差）
total_mixed = sum(mix_sample_counts);
if total_mixed ~= mixed_samples
    % 调整最后一个类型的样本数以匹配总数
    diff_count = mixed_samples - total_mixed;
    mix_sample_counts(end) = mix_sample_counts(end) + diff_count;
end

fprintf('混合干扰样本分布: 11种类型，总计%d个样本\n', mixed_samples);
for mix_idx = 1:length(MIX_TYPES)
    fprintf('  类型%d: %d个样本\n', mix_idx, mix_sample_counts(mix_idx));
end

for mix_idx = 1:length(MIX_TYPES)
    jam_types = MIX_TYPES{mix_idx};
    num_jam_types = length(jam_types);
    
    fprintf('  正在生成混合类型%d (%d个样本)...\n', mix_idx, mix_sample_counts(mix_idx));
    
    for j = 1:mix_sample_counts(mix_idx)
        sample_counter = sample_counter + 1;
        
        % 初始化干扰信号和JSR
        jam_signals = cell(1, num_jam_types);
        jsr_values = zeros(1, num_jam_types);
        
        % 为每种干扰选择信号和JSR
        for k = 1:num_jam_types
            switch jam_types(k)
                case 1 % FSK
                    jam_signals{k} = FSK_pool{randi(N_FSK)};
                    jsr_values(k) = JSR_range(randi(length(JSR_range)));
                case 2 % BPSK
                    jam_signals{k} = BPSK_pool{randi(N_BPSK)};
                    jsr_values(k) = JSR_range(randi(length(JSR_range)));
                case 3 % QPSK
                    jam_signals{k} = QPSK_pool{randi(N_QPSK)};
                    jsr_values(k) = JSR_range(randi(length(JSR_range)));
                case 4 % 16QAM
                    jam_signals{k} = QAM16_pool{randi(N_16QAM)};
                    jsr_values(k) = JSR_range(randi(length(JSR_range)));
            end
        end
        
        % 生成时频图像
        img_data = generate_MIXED_image_mixed_fixed(...
            S_pool{randi(N_S)}, ...  % 随机选择目标信号
            jam_signals, ...         % 干扰信号集合
            jsr_values, ...          % JSR值集合
            jam_types, ...           % 干扰类型标识集合
            fs, ...                  % 传递采样频率
            num_samples, ...         % 传递信号长度
            window_length, ...       % STFT参数
            noverlap, ...            % STFT参数
            nfft, ...                % STFT参数
            fc_pool, ...             % 频移池信息
            dataset_type);           % 数据集类型
        
        % 生成混合类型标签字符串
        type_str = '';
        for k = 1:num_jam_types
            type_str = [type_str, num2str(jam_types(k))];
        end
        
        % 保存样本
        filename = sprintf('MIXED_Types%s_%05d.png', type_str, sample_counter);
        imwrite(img_data, fullfile(save_dir, filename));
        
        % 进度显示
        if mod(sample_counter, 1000) == 0
            fprintf('  已生成 %d/%d 样本\n', sample_counter, total_samples);
        end
    end
end

fprintf('样本生成完成! 总样本数: %d\n', sample_counter);

% 最终验证
if sample_counter ~= total_samples
    fprintf('警告: 实际生成样本数(%d)与目标数(%d)不符!\n', sample_counter, total_samples);
else
    fprintf('成功: 准确生成了%d个样本\n', total_samples);
end

% 打印样本分布统计
single_fsk_count = fsk_samples;
single_bpsk_count = bpsk_samples;
single_qpsk_count = qpsk_samples;
single_qam_count = qam_samples;

% 计算各类混合干扰样本数
mixed_counts = mix_sample_counts;

fprintf('\n=== 样本分布统计 ===\n');
fprintf('样本分布: 单类干扰(%d), 混合干扰(%d)\n', single_samples, mixed_samples);
fprintf('单类样本分布: FSK(%d), BPSK(%d), QPSK(%d), 16QAM(%d)\n', ...
        single_fsk_count, single_bpsk_count, single_qpsk_count, single_qam_count);
fprintf('混合样本分布 (11种类型):\n');
fprintf('  FSK+BPSK: %d, FSK+QPSK: %d, FSK+16QAM: %d\n', ...
        mixed_counts(1), mixed_counts(2), mixed_counts(3));
fprintf('  BPSK+QPSK: %d, BPSK+16QAM: %d, QPSK+16QAM: %d\n', ...
        mixed_counts(4), mixed_counts(5), mixed_counts(6));
fprintf('  FSK+BPSK+QPSK: %d, FSK+BPSK+16QAM: %d\n', ...
        mixed_counts(7), mixed_counts(8));
fprintf('  FSK+QPSK+16QAM: %d, BPSK+QPSK+16QAM: %d\n', ...
        mixed_counts(9), mixed_counts(10));
fprintf('  FSK+BPSK+QPSK+16QAM: %d\n', mixed_counts(11));

% 计算每种干扰的总样本量
total_fsk_samples = single_fsk_count + ...
                   mixed_counts(1) + mixed_counts(2) + mixed_counts(3) + ...
                   mixed_counts(7) + mixed_counts(8) + mixed_counts(9) + mixed_counts(11);
total_bpsk_samples = single_bpsk_count + ...
                    mixed_counts(1) + mixed_counts(4) + mixed_counts(5) + ...
                    mixed_counts(7) + mixed_counts(8) + mixed_counts(10) + mixed_counts(11);
total_qpsk_samples = single_qpsk_count + ...
                    mixed_counts(2) + mixed_counts(4) + mixed_counts(6) + ...
                    mixed_counts(7) + mixed_counts(9) + mixed_counts(10) + mixed_counts(11);
total_qam_samples = single_qam_count + ...
                   mixed_counts(3) + mixed_counts(5) + mixed_counts(6) + ...
                   mixed_counts(8) + mixed_counts(9) + mixed_counts(10) + mixed_counts(11);

fprintf('\n=== 每种干扰类型出现总次数 ===\n');
fprintf('FSK总出现次数: %d (占总样本%.1f%%)\n', total_fsk_samples, total_fsk_samples/total_samples*100);
fprintf('BPSK总出现次数: %d (占总样本%.1f%%)\n', total_bpsk_samples, total_bpsk_samples/total_samples*100);
fprintf('QPSK总出现次数: %d (占总样本%.1f%%)\n', total_qpsk_samples, total_qpsk_samples/total_samples*100);
fprintf('16QAM总出现次数: %d (占总样本%.1f%%)\n', total_qam_samples, total_qam_samples/total_samples*100);

fprintf('频移样本池大小: %d (频率范围: %.1f-%.1f MHz)\n', ...
        length(fc_pool), min(fc_pool)/1e6, max(fc_pool)/1e6);
fprintf('数字调制参数: 符号率=%.1fksps, 每符号样本数=%d\n', symbol_rate/1e3, sps);
fprintf('信号长度: %d 采样点 (%.1f μs)\n', num_samples, duration*1e6);
end

%% ========== 修复后的辅助函数定义 ==========

%% 辅助函数：获取干扰类型名称
function type_name = get_jam_type_name(type_id)
    switch type_id
        case 1
            type_name = 'FSK';
        case 2
            type_name = 'BPSK';
        case 3
            type_name = 'QPSK';
        case 4
            type_name = '16QAM';
        otherwise
            type_name = 'UNKNOWN';
    end
end

%% 验证函数：统计前N个样本中干扰类型出现次数
function validate_jam_type_counts(data_dir, num_samples)
    files = dir(fullfile(data_dir, '*.png'));
    
    % 只检查前num_samples个文件
    num_files_to_check = min(num_samples, length(files));
    
    % 初始化计数器
    jam_counts = zeros(1, 4); % FSK, BPSK, QPSK, 16QAM
    total_mentions = 0; % 总共提到的干扰次数（一个样本可能提到多个）
    
    fprintf('检查前%d个样本...\n', num_files_to_check);
    
    for i = 1:num_files_to_check
        filename = files(i).name;
        
        % 检查单类干扰
        if contains(filename, 'SINGLE_FSK')
            jam_counts(1) = jam_counts(1) + 1;
            total_mentions = total_mentions + 1;
        elseif contains(filename, 'SINGLE_BPSK')
            jam_counts(2) = jam_counts(2) + 1;
            total_mentions = total_mentions + 1;
        elseif contains(filename, 'SINGLE_QPSK')
            jam_counts(3) = jam_counts(3) + 1;
            total_mentions = total_mentions + 1;
        elseif contains(filename, 'SINGLE_16QAM')
            jam_counts(4) = jam_counts(4) + 1;
            total_mentions = total_mentions + 1;
        end
        
        % 检查混合干扰
        if contains(filename, 'MIXED_Types')
            % 提取类型数字串
            start_idx = strfind(filename, 'Types') + 5;
            end_idx = strfind(filename, '_') - 1;
            type_str = filename(start_idx:end_idx);
            
            % 统计每个数字（干扰类型）
            for k = 1:length(type_str)
                type_num = str2double(type_str(k));
                if type_num >= 1 && type_num <= 4
                    jam_counts(type_num) = jam_counts(type_num) + 1;
                    total_mentions = total_mentions + 1;
                end
            end
        end
    end
    
    fprintf('统计结果:\n');
    fprintf('FSK出现次数: %d\n', jam_counts(1));
    fprintf('BPSK出现次数: %d\n', jam_counts(2));
    fprintf('QPSK出现次数: %d\n', jam_counts(3));
    fprintf('16QAM出现次数: %d\n', jam_counts(4));
    fprintf('总提及次数: %d (平均每个样本%.2f种干扰)\n', total_mentions, total_mentions/num_files_to_check);
    
    % 计算百分比
    fprintf('出现频率百分比:\n');
    for i = 1:4
        percentage = jam_counts(i) / total_mentions * 100;
        type_name = get_jam_type_name(i);
        fprintf('  %s: %.1f%%\n', type_name, percentage);
    end
end

%% 修复的时频图像生成函数（单干扰）
function [img_data] = generate_MIXED_image_fixed(s_signal, jam_signal, jsr_val, jam_type, fs, num_samples, window_length, noverlap, nfft, fc_pool, dataset_type)
    % 1. 目标信号
    u_e = 0.7 * s_signal;
    
    % 2. 干扰信号处理
    A_j = sqrt(10^(jsr_val/10)) * 0.7;
    j_signal = A_j * jam_signal;
    
    % 3. 添加噪声（干噪比20dB）
    target_power = (0.7^2)/2;
    JNR = 20; 
    noise_power = target_power * 10^(-JNR/10);
    n = sqrt(noise_power) * randn(1, num_samples);
    
    % 4. 合成接收信号
    u = u_e + j_signal + n;
    
    % 5. 确保信号长度足够
    if length(u) < window_length
        % 补零到最小长度
        u = [u, zeros(1, window_length - length(u))];
    end
    
    % 6. 高分辨率时频分析
    [S, f, t] = spectrogram(u, hamming(window_length), noverlap, nfft, fs, 'yaxis');
    S_mag = abs(S);
    
    % 7. 动态确定截取频带（基于信号中心频率）
    % 估计信号中心频率
    [~, max_idx] = max(mean(S_mag, 2));
    center_freq = f(max_idx);
    
    % 计算自适应带宽（保留20%裕量）
    bandwidth = 400e6; % 基础带宽400MHz
    min_freq = max(0, center_freq - bandwidth/2);
    max_freq = min(fs/2, center_freq + bandwidth/2);
    
    % 确保覆盖训练集范围（针对测试集）
    if min_freq > min(fc_pool)
        min_freq = min(fc_pool);
    end
    if max_freq < max(fc_pool)
        max_freq = max(fc_pool);
    end
    
    % 8. 截取自适应的频段
    freq_idx = (f >= min_freq) & (f <= max_freq);
    cropped_freq = S_mag(freq_idx, :);
    
    % 9. 动态范围压缩（对数变换）
    S_log = 10*log10(cropped_freq + eps);
    
    % 10. 归一化处理
    min_val = min(S_log(:));
    max_val = max(S_log(:));
    if max_val > min_val
        normalized = (S_log - min_val) / (max_val - min_val);
    else
        normalized = zeros(size(S_log));
    end
    
    % 11. 频率信息编码（增强模型对频移的感知）
    freq_info = linspace(0, 1, size(normalized, 1))';
    freq_layer = repmat(freq_info, 1, size(normalized, 2));
    
    % 12. 双三次插值缩放
    target_size = [64, 64];
    normalized_resized = imresize(normalized, target_size, 'bicubic');
    freq_layer_resized = imresize(freq_layer, target_size, 'bicubic');
    
    % 13. 测试时随机缩放频率轴（增强泛化能力）
    if strcmp(dataset_type, 'test')
        scale_factor = 0.8 + 0.4*rand(); % 0.8-1.2随机缩放
        new_width = round(target_size(2) * scale_factor);
        normalized_resized = imresize(normalized_resized, [target_size(1), new_width], 'bicubic');
        freq_layer_resized = imresize(freq_layer_resized, [target_size(1), new_width], 'bicubic');
        
        % 填充或裁剪到标准尺寸
        if new_width > target_size(2)
            normalized_resized = normalized_resized(:, 1:target_size(2));
            freq_layer_resized = freq_layer_resized(:, 1:target_size(2));
        else
            pad_size = target_size(2) - new_width;
            normalized_resized = padarray(normalized_resized, [0, pad_size], 0, 'post');
            freq_layer_resized = padarray(freq_layer_resized, [0, pad_size], 0, 'post');
        end
    end
    
    % 14. 锐化增强特征
    sharp_kernel = fspecial('unsharp', 0.5);
    sharpened = imfilter(normalized_resized, sharp_kernel);
    
    % 15. 转换为三通道图像：时频图 + 频率信息 + 锐化图
    img_gray = im2uint8(sharpened);
    img_freq = im2uint8(freq_layer_resized);
    img_sharp = im2uint8(sharpened);
    
    img_data = cat(3, img_gray, img_freq, img_sharp);
    
    % 16. 数据增强：随机对比度调整
    if rand() > 0.7
        img_data = imadjust(img_data, [0.1 0.9], []);
    end
end

%% 修复的多干扰时频图像生成函数
function [img_data] = generate_MIXED_image_mixed_fixed(s_signal, jam_signals, jsr_values, jam_types, fs, num_samples, window_length, noverlap, nfft, fc_pool, dataset_type)
    % 1. 目标信号
    u_e = 0.7 * s_signal;
    
    % 2. 合成干扰信号
    j_total = zeros(1, num_samples);
    for k = 1:length(jam_signals)
        A_j = sqrt(10^(jsr_values(k)/10)) * 0.7;
        j_total = j_total + A_j * jam_signals{k};
    end
    
    % 3. 添加噪声（干噪比20dB）
    target_power = (0.7^2)/2;
    JNR = 20; 
    noise_power = target_power * 10^(-JNR/10);
    n = sqrt(noise_power) * randn(1, num_samples);
    
    % 4. 合成接收信号
    u = u_e + j_total + n;
    
    % 5. 确保信号长度足够
    if length(u) < window_length
        u = [u, zeros(1, window_length - length(u))];
    end
    
    % 6. 多尺度时频分析
    % 高分辨率分析
    [S_high, f, t] = spectrogram(u, hamming(window_length), noverlap, nfft, fs, 'yaxis');
    S_mag_high = abs(S_high);
    
    % 低分辨率分析
    win_low = round(window_length/2);
    nov_low = round(noverlap/2);
    nfft_low = round(nfft/2);
    [S_low, f_low, t_low] = spectrogram(u, hamming(win_low), nov_low, nfft_low, fs, 'yaxis');
    S_mag_low = abs(S_low);
    
    % 7. 动态确定截取频带（基于信号中心频率）
    [~, max_idx] = max(mean(S_mag_high, 2));
    center_freq = f(max_idx);
    
    bandwidth = 400e6; % 基础带宽400MHz
    min_freq = max(0, center_freq - bandwidth/2);
    max_freq = min(fs/2, center_freq + bandwidth/2);
    
    % 确保覆盖训练集范围
    if min_freq > min(fc_pool)
        min_freq = min(fc_pool);
    end
    if max_freq < max(fc_pool)
        max_freq = max(fc_pool);
    end
    
    % 8. 截取自适应的频段
    freq_idx = (f >= min_freq) & (f <= max_freq);
    cropped_high = S_mag_high(freq_idx, :);
    
    freq_idx_low = (f_low >= min_freq) & (f_low <= max_freq);
    cropped_low = S_mag_low(freq_idx_low, :);
    
    % 9. 动态范围压缩（对数变换）
    S_log_high = 10*log10(cropped_high + eps);
    S_log_low = 10*log10(cropped_low + eps);
    
    % 10. 归一化处理
    min_val_high = min(S_log_high(:));
    max_val_high = max(S_log_high(:));
    if max_val_high > min_val_high
        normalized_high = (S_log_high - min_val_high) / (max_val_high - min_val_high);
    else
        normalized_high = zeros(size(S_log_high));
    end
    
    min_val_low = min(S_log_low(:));
    max_val_low = max(S_log_low(:));
    if max_val_low > min_val_low
        normalized_low = (S_log_low - min_val_low) / (max_val_low - min_val_low);
    else
        normalized_low = zeros(size(S_log_low));
    end
    
    % 11. 频率信息编码
    freq_info_high = linspace(0, 1, size(normalized_high, 1))';
    freq_layer_high = repmat(freq_info_high, 1, size(normalized_high, 2));
    
    freq_info_low = linspace(0, 1, size(normalized_low, 1))';
    freq_layer_low = repmat(freq_info_low, 1, size(normalized_low, 2));
    
    % 12. 双三次插值缩放
    target_size = [64, 64];
    norm_high_resized = imresize(normalized_high, target_size, 'bicubic');
    freq_high_resized = imresize(freq_layer_high, target_size, 'bicubic');
    norm_low_resized = imresize(normalized_low, target_size, 'bicubic');
    freq_low_resized = imresize(freq_layer_low, target_size, 'bicubic');
    
    % 13. 测试时随机缩放频率轴
    if strcmp(dataset_type, 'test')
        scale_factor = 0.8 + 0.4*rand(); % 0.8-1.2随机缩放
        new_width = round(target_size(2) * scale_factor);
        
        norm_high_resized = imresize(norm_high_resized, [target_size(1), new_width], 'bicubic');
        freq_high_resized = imresize(freq_high_resized, [target_size(1), new_width], 'bicubic');
        norm_low_resized = imresize(norm_low_resized, [target_size(1), new_width], 'bicubic');
        freq_low_resized = imresize(freq_low_resized, [target_size(1), new_width], 'bicubic');
        
        % 填充或裁剪到标准尺寸
        if new_width > target_size(2)
            norm_high_resized = norm_high_resized(:, 1:target_size(2));
            freq_high_resized = freq_high_resized(:, 1:target_size(2));
            norm_low_resized = norm_low_resized(:, 1:target_size(2));
            freq_low_resized = freq_low_resized(:, 1:target_size(2));
        else
            pad_size = target_size(2) - new_width;
            norm_high_resized = padarray(norm_high_resized, [0, pad_size], 0, 'post');
            freq_high_resized = padarray(freq_high_resized, [0, pad_size], 0, 'post');
            norm_low_resized = padarray(norm_low_resized, [0, pad_size], 0, 'post');
            freq_low_resized = padarray(freq_low_resized, [0, pad_size], 0, 'post');
        end
    end
    
    % 14. 锐化增强特征
    sharp_kernel = fspecial('unsharp', 0.5);
    sharpened_high = imfilter(norm_high_resized, sharp_kernel);
    sharpened_low = imfilter(norm_low_resized, sharp_kernel);
    
    % 15. 融合多尺度特征
    img_high = im2uint8(sharpened_high);
    img_low = im2uint8(sharpened_low);
    img_freq = im2uint8(freq_high_resized);
    
    % 三通道：高分辨率时频图 + 低分辨率时频图 + 频率信息
    img_data = cat(3, img_high, img_low, img_freq);
    
    % 16. 数据增强：随机添加高斯噪声
    if rand() > 0.8
        noise_level = randi([5, 20]);
        img_data = imnoise(img_data, 'gaussian', 0, noise_level/255);
    end
end