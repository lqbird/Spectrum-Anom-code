% clc;
clear;
close all;

wavFiles = dir(fullfile('E:\添加异常段（单段and间歇）\未添加\间歇正常数据','*.wav'));

% ===== Output directory (automatically written to different subfolders by level) =====
outRoot = 'E:\添加异常段（单段and间歇）\间隔异常';

% ===== Level mode =====
% 'S'   : generate strong only
% 'M'   : generate mid only
% 'W'   : generate weak only
% 'SMW' : generate all three levels
%         (the same file will be output three times: S / M / W)
LEVEL_MODE = 'W';

% ===== Frequency range (MHz) =====
lower = 3;    % lower bound in MHz
upper = 37;   % upper bound in MHz

% ===== Original distribution parameters for "number of generated signals" (kept unchanged) =====
mu = 20;
sigma = 100;

% ===== (Optional) Random seed for reproducibility =====
rng(42);

for i = 1 : 1 : length(wavFiles)
    disp(['Processing file No. ', num2str(i)]);
    name = wavFiles(i).name;
    filePath = fullfile(wavFiles(i).folder,name);

    % Read data
    [data, fs] = audioread(filePath);
    t = (0:length(data)-1)'/fs;  % Time axis
    IQ_signal0 = data(:,1) + 1j*data(:,2);

    % Print SNR
    snr_dB = snr(data(:,1));
    fprintf('The SNR of this signal is %.2f dB.\n', snr_dB);

    %% Compute spectrum and find the main peak
    IQ_spect = fftshift(abs(fft(IQ_signal0)));
    IQ_spectrum = 20*log10(IQ_spect + eps);
    f_axis = linspace(-fs/2, fs/2, length(data));

    [max_power, max_index] = max(IQ_spectrum);
    A_IQ = sqrt(max_power)/1000; 
    f_peak = f_axis(max_index);
    fprintf('The strongest signal is located at %.2f MHz with amplitude %.4f\n', f_peak/1e6, A_IQ);

    % Quantity used in the amplitude formula
    % (your original code used mean(IQ_spectrum))
    meanSpec = mean(IQ_spectrum);

    % Determine which levels to run
    if strcmp(LEVEL_MODE,'SMW')
        level_list = {'strong','mid','weak'};
    elseif strcmp(LEVEL_MODE,'S')
        level_list = {'strong'};
    elseif strcmp(LEVEL_MODE,'M')
        level_list = {'mid'};
    elseif strcmp(LEVEL_MODE,'W')
        level_list = {'weak'};
    else
        error('LEVEL_MODE must be S/M/W/SMW');
    end

    for li = 1:numel(level_list)
        level = level_list{li};

        % ===== Parameters for each level:
        % Strong = keep your current parameters unchanged;
        % Mid/Weak are weakened based on Strong =====
        switch level
            case 'strong'
                % —— Strong: exactly the same as your current script strength (keep unchanged as much as possible) ——
                N = round(5 + 10*rand(1,1));      % 5~15
                L = length(data);
                min_dur = round(L/16);
                max_dur = round(L);              % can cover the whole segment
                BW_MAX  = 1e4;                   % bw_fm = 1e4*rand
                AMP_SCALE = 1.00;                % no amplitude scaling (keep strong)

            case 'mid'
                % —— Mid: significantly weaker
                % (smaller coverage + smaller bandwidth + smaller amplitude) ——
                N = round(3 + 6*rand(1,1));       % 3~9 (fewer segments)
                L = length(data);
                min_dur = round(L/24);            % shorter
                max_dur = round(L/2);             % at most half the segment (reduced coverage)
                BW_MAX  = 6e3;                    % reduced bandwidth
                AMP_SCALE = 0.45;                 % amplitude reduced to about 45% of strong

            case 'weak'
                % —— Weak: the weakest
                % (fewer segments, shorter duration, narrower bandwidth, smaller amplitude) ——
                N = round(1 + 4*rand(1,1));       % 1~5
                L = length(data);
                min_dur = round(L/32);
                max_dur = round(L/4);
                BW_MAX  = 3e3;
                AMP_SCALE = 0.20;                 % about 20% of strong
        end

        % ===== Superimpose FM signals onto the I/Q signal =====
        IQ_signal = IQ_signal0;  % restart from the original signal for each level
        FM_seg_last = complex(0,0); % used for the final plot

        for FM_num = 1 : 1 : N
            % 1) Duration
            this_len = randi([min_dur, max_dur]);

            % 2) Start position
            start_num = randi(L - this_len + 1);
            idx = start_num : (start_num + this_len - 1);

            % 3) Random carrier frequency, bandwidth, and amplitude
            fc = 1e6 * (lower + (upper - lower) * rand(1, 1)); % Hz

            bw_fm = BW_MAX * rand(1,1);                         % 0~BW_MAX
            kf = bw_fm / (2/2);                               

            % Amplitude term
            mean_IQ_spc = (meanSpec/fs) * bw_fm * 2 * rand(1, 1);
            mean_IQ_spc = mean_IQ_spc * AMP_SCALE;

            % 4) Time segment
            t_seg = t(idx);

            % 5) Modulation
            fm_mod_signal = cos(2 * pi * kf * (t_seg.^2));
            FM_seg = mean_IQ_spc * exp(1j * 2 * pi * fc .* t_seg + 1j * 2 * pi * fm_mod_signal);

            % 6) Superposition
            IQ_signal(idx) = IQ_signal(idx) + FM_seg;
            FM_seg_last = FM_seg; 
        end

        IQ_signal_modified = IQ_signal;

        %% ===== Save (recommended to enable for comparing the three levels) =====
        outDir = fullfile(outRoot, level);
        if ~exist(outDir,'dir'), mkdir(outDir); end
        out = [real(IQ_signal_modified), imag(IQ_signal_modified)];

        % Anti-clipping (optional, to avoid overflow during writing)
        mx = max(abs(out), [], 'all');
        if mx > 0.98
            out = out * (0.98/mx);
        end

        % Output filename includes the level
        [~, baseName, ext] = fileparts(name);
        outName = sprintf('%s_%s_N%d_amp%.2f_bw%.0f%s', baseName, level, N, AMP_SCALE, BW_MAX, ext);
        audiowrite(fullfile(outDir, outName), out, fs);

        fprintf('[%s] Written: %s\n', level, outName);

        % %% ===== (Optional) Plot only for strong, to avoid too many pop-up windows =====
        % if strcmp(level,'strong')
        %     % Time domain
        %     figure;
        %     subplot(2,1,1);
        %     plot(t(1:1000), real(IQ_signal_modified(1:1000))); hold on;
        %     plot(t(1:min(1000,numel(FM_seg_last))), real(FM_seg_last(1:min(1000,numel(FM_seg_last)))), 'r');
        %     legend('I (after superposition)', 'Last FM');
        %     xlabel('Time (s)'); ylabel('Amplitude'); title(['Time-domain I signal - ', level]);
        % 
        %     subplot(2,1,2);
        %     plot(t(1:1000), imag(IQ_signal_modified(1:1000))); hold on;
        %     plot(t(1:min(1000,numel(FM_seg_last))), imag(FM_seg_last(1:min(1000,numel(FM_seg_last)))), 'r');
        %     legend('Q (after superposition)', 'Last FM');
        %     xlabel('Time (s)'); ylabel('Amplitude'); title(['Time-domain Q signal - ', level]);
        % 
        %     % Spectrum
        %     figure;
        %     IQ_spectrum_mod = fftshift(abs(fft(IQ_signal_modified)));
        %     plot(f_axis/1e6, 20*log10(IQ_spectrum_mod + eps));
        %     xlabel('Frequency (MHz)'); ylabel('Power (dB)');
        %     title(['Power spectrum of I/Q signal (', level, ')']);
        %     grid on;
        % 
        %     figure;
        %     plot(f_axis/1e6, IQ_spectrum);
        %     xlabel('Frequency (MHz)'); ylabel('Power (dB)');
        %     title('Power spectrum of I/Q signal (normal)');
        %     grid on;
        % end

    end

    fprintf('Processing of this signal is complete.\n');
end