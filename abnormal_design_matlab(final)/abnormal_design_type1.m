%% A1 Anomaly Pattern 1: Spur-Interference Type (Multi-Carrier FM Superposition)
% Three forms: NB-FM + DRIFT + GATED
% Planned K count: 37 Gaussian-shaped K_plan values (fewer at both ends, more in the middle, with randomness)

clear; clc; close all;

%% ========== Path configuration ==========
inDir  = 'E:\单段未添加异常信号';
outDir = 'E:\异常FM\S';
if ~exist(outDir,'dir'), mkdir(outDir); end
wavFiles = dir(fullfile(inDir,'*.wav')); 

%% ========== Global control ==========
rng(42);                
WRITE_FILE = true;
SHOW_PLOT  = false;
SAFE_PEAK  = 0.98;

%% ========== Level selection mode ==========
% 'S'   : strong only
% 'SM'  : strong + medium
% 'SMW' : strong + medium + weak
LEVEL_MODE = 'S';

%% ========== Event window ==========
EVENT_MODE = 'random';    % 'fixed' or 'random'
EVENT_T0 = 0.50;
EVENT_T1 = 1.50;

RAND_T0_RANGE  = [0.20, 1.20];
RAND_LEN_RANGE = [0.15, 0.35];     % Still short (unchanged)

%% ========== Random carrier frequency within 40 MHz bandwidth ==========
BAND_40MHZ = 40e6;
HALF_BAND  = BAND_40MHZ/2;

%% ========== D) Main-peak guard band (avoid f_peak) ==========
% Force the carrier frequency to avoid the vicinity of the main peak
% (to prevent "stacking on the main carrier and looking like overall gain/shape change")
GUARD_HZ = 200e3;    % 200 kHz
MAX_TRY_FC = 30;     % Maximum number of resampling attempts

%% ========== Three forms ==========
NUM_TYPES = 3;
P_ON = 0.7;
GATE_BLOCK_MS = [5, 30];

%% ========== Strong / Medium / Weak (C: weak is slightly stronger but still short) ==========
db = @(x) 20*log10(x);

% strong: 0.60~0.90
% medium: 0.20~0.45
% weak  : 0.10~0.22  
LEVELS(1).name = 'strong'; LEVELS(1).gain_db_range = [db(0.60), db(0.90)];
LEVELS(2).name = 'medium'; LEVELS(2).gain_db_range = [db(0.20), db(0.45)];
LEVELS(3).name = 'weak';   LEVELS(3).gain_db_range = [db(0.10), db(0.22)];

% ===== Sampling probabilities for levels (used only in mixed modes) =====
pS = 0.25;
pM = 0.35;
% pW = 0.40 (implicit)

%% ========== K_plan (37 segments: fewer at both ends, more in the middle, Gaussian-shaped + random) ==========
NUM_SIG_TARGET = 37;
NUM_SIG = min(NUM_SIG_TARGET, length(wavFiles));

K_MIN  = 2;
K_PEAK = 8;
K_MAX  = 10;

sigma_i = NUM_SIG_TARGET/5;
sigma_K = 0.6;
center  = (NUM_SIG_TARGET+1)/2;

K_plan = zeros(NUM_SIG,1);
for ii = 1:NUM_SIG
    mu = K_MIN + (K_PEAK - K_MIN) * exp(-0.5*((ii - center)/sigma_i)^2);
    Kp = round(mu + sigma_K*randn);
    if Kp < K_MIN, Kp = K_MIN; end
    if Kp > K_MAX, Kp = K_MAX; end
    K_plan(ii) = Kp;
end

%% ========== Main loop ==========
for i = 1:NUM_SIG
    fprintf('\n==== [%d/%d] %s ====\n', i, NUM_SIG, wavFiles(i).name);

    filePath = fullfile(wavFiles(i).folder, wavFiles(i).name);
    [data, fs] = audioread(filePath);
    if size(data,2) < 2
        warning('File %s is not dual-channel (I/Q). Skipped.', wavFiles(i).name);
        continue;
    end

    IQ = data(:,1) + 1j*data(:,2);
    N  = length(IQ);
    T  = N/fs;

    % Reference amplitude (RMS)
    A_ref = sqrt(mean(abs(IQ).^2)) + eps;

    % Main peak estimation
    X0 = fft(IQ);
    mag0 = abs(fftshift(X0));
    [~, max_idx] = max(mag0);
    f_axis = linspace(-fs/2, fs/2, N).';
    f_peak = f_axis(max_idx);

    % Event window
    if strcmp(EVENT_MODE,'fixed')
        t0 = EVENT_T0; t1 = EVENT_T1;
    else
        t0  = RAND_T0_RANGE(1) + (RAND_T0_RANGE(2)-RAND_T0_RANGE(1))*rand;
        len = RAND_LEN_RANGE(1) + (RAND_LEN_RANGE(2)-RAND_LEN_RANGE(1))*rand;
        t1  = min(t0 + len, T-1e-6);
    end

    idx0 = max(1, floor(t0*fs)+1);
    idx1 = min(N, floor(t1*fs));
    if idx1 <= idx0 + 16
        warning('Event window is too short. Skipped this file.');
        continue;
    end

    dur = idx1 - idx0 + 1;
    tau = (0:dur-1).' / fs;

    spur = complex(zeros(dur,1), zeros(dur,1));
    K = K_plan(i);

    eff_half = min(HALF_BAND, 0.45*fs);
    ny = fs/2 * 0.98;

    cnt = [0 0 0]; % Count strong / medium / weak

    for k = 1:K
        %% ===== Select level_id (LEVEL_MODE controls available levels) =====
        switch LEVEL_MODE
            case 'S'
                level_id = 1;

            case 'SM'
                r = rand;
                if r < pS/(pS+pM)
                    level_id = 1;
                else
                    level_id = 2;
                end

            case 'SMW'
                r = rand;
                if r < pS
                    level_id = 1;
                elseif r < pS + pM
                    level_id = 2;
                else
                    level_id = 3;
                end

            otherwise
                error('LEVEL_MODE must be ''S'' / ''SM'' / ''SMW''.');
        end
        cnt(level_id) = cnt(level_id) + 1;

        %% ===== B) Amplitude: normalize according to K =====
        gr = LEVELS(level_id).gain_db_range;
        gain_db = gr(1) + (gr(2)-gr(1))*rand;

        % 
        A_fm = (A_ref * 10^(gain_db/20)) / sqrt(max(K,1));

        %% ===== D) Carrier frequency: avoid the main-peak guard band =====
        % Sample first, then ensure |fc0 - f_peak| > GUARD_HZ
        fc0 = f_peak; % init
        for tt = 1:MAX_TRY_FC
            fc_try = f_peak + (rand-0.5)*2*eff_half;
            if fc_try >  ny, fc_try =  ny; end
            if fc_try < -ny, fc_try = -ny; end
            if abs(fc_try - f_peak) > GUARD_HZ
                fc0 = fc_try;
                break;
            end
            fc0 = fc_try; % If repeated trials fail, still use the last one (rare)
        end

        %% ===== Three forms =====
        fm_type = randi(NUM_TYPES);

        % Narrower sidebands (unchanged)
        fdev_max = 1e3 + 4e3*rand;  % 1~5 kHz
        fmod     = 40  + 120*rand;  % 40~160 Hz

        if fm_type == 1
            % NB-FM
            inst_f = fc0 + fdev_max * sin(2*pi*fmod*tau);
            phi = 2*pi * cumsum(inst_f) / fs;
            s = A_fm .* exp(1j*phi);

        elseif fm_type == 2
            % DRIFT
            drift_bw = 5e2 + 2e3*rand; % 0.5~2.5 kHz
            drift = linspace(-drift_bw, drift_bw, dur).';
            inst_f = fc0 + drift;
            phi = 2*pi * cumsum(inst_f) / fs;
            s = A_fm .* exp(1j*phi);

        else
            % GATED
            inst_f = fc0 + fdev_max * sin(2*pi*fmod*tau);
            phi = 2*pi * cumsum(inst_f) / fs;
            s0 = exp(1j*phi);

            gate = zeros(dur,1);
            pos = 1;
            while pos <= dur
                blk_ms = GATE_BLOCK_MS(1) + (GATE_BLOCK_MS(2)-GATE_BLOCK_MS(1))*rand;
                blk = max(1, round(blk_ms*1e-3*fs));
                blk_end = min(dur, pos + blk - 1);
                if rand < P_ON
                    gate(pos:blk_end) = 1;
                end
                pos = blk_end + 1;
            end

            s = A_fm .* gate .* s0;
        end

        spur = spur + s;
    end

    fprintf('LEVEL_MODE=%s, level statistics: strong=%d, medium=%d, weak=%d (K=%d)\n', ...
        LEVEL_MODE, cnt(1), cnt(2), cnt(3), K);

    %% ===== A) Scale only the spur within the event window (do not modify the whole out) =====
    IQ_win = IQ(idx0:idx1);
    tmp_win = IQ_win + spur;
    mx_win = max(abs(tmp_win));

    if mx_win > SAFE_PEAK
        scale_spur = SAFE_PEAK / (mx_win + eps);
        spur = spur * scale_spur;

        % Recheck
        mx_win2 = max(abs(IQ_win + spur));
        fprintf('  [A:local spur scaling] mx_win=%.4f > %.2f, scale_spur=%.4f, mx_win2=%.4f\n', ...
            mx_win, SAFE_PEAK, scale_spur, mx_win2);
    else
        fprintf('  [A:no spur scaling needed] mx_win=%.4f <= %.2f\n', mx_win, SAFE_PEAK);
    end

    %% Inject the event window
    IQ_out = IQ;
    IQ_out(idx0:idx1) = IQ_out(idx0:idx1) + spur;

    %% Final pre-write check: prioritize "scaling only the event window", then fall back to global scaling if necessary
    out = [real(IQ_out), imag(IQ_out)];
    mx_all = max(abs(out), [], 'all');

    if mx_all > SAFE_PEAK
        % Try scaling only the event window again
        mx_win3 = max(abs(IQ_out(idx0:idx1)));
        scale_win = SAFE_PEAK / (mx_win3 + eps);
        IQ_out(idx0:idx1) = IQ_out(idx0:idx1) * scale_win;

        out = [real(IQ_out), imag(IQ_out)];
        mx_all2 = max(abs(out), [], 'all');

        if mx_all2 > SAFE_PEAK
            % Extreme fallback: global scaling (theoretically rare)
            out = out * (SAFE_PEAK / (mx_all2 + eps));
            fprintf('  [fallback:global scaling] mx_all2=%.4f > %.2f, global_scale=%.4f\n', ...
                mx_all2, SAFE_PEAK, SAFE_PEAK/(mx_all2+eps));
        else
            fprintf('  [window scaling fixed] mx_all=%.4f -> mx_all2=%.4f\n', mx_all, mx_all2);
        end
    end

    %% Visualization
    if SHOW_PLOT
        figure;
        subplot(2,1,1); plot(out(1:min(2000,N),1)); title('I (before writing)');
        subplot(2,1,2); plot(out(1:min(2000,N),2)); title('Q (before writing)');
        drawnow;
    end

    %% Save
    if WRITE_FILE
        [~, baseName, ext] = fileparts(wavFiles(i).name);
        outName = sprintf('%s_A1_%s_spurFM3types_K%02d_t%.3f-%.3f%s', ...
            baseName, LEVEL_MODE, K, t0, t1, ext);
        audiowrite(fullfile(outDir, outName), out, fs);
    end
end