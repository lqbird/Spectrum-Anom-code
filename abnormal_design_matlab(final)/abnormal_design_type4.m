%% A4 Peak Removal (Service-Missing Type) - V4_step1
% (Only enhance the "coverage area")

clear; close all; clc;

%% ===== Paths =====
inDir  = 'E:添加异常段（单段and间歇）\未添加\单段未添加异常数据';
outDir = 'E:\添加异常段（单段and间歇）\峰值去除';
if ~exist(outDir,'dir'), mkdir(outDir); end
wavFiles = dir(fullfile(inDir,'*.wav'));

%% ===== Global control =====
rng(42);

NUM_SIG_TARGET = 37;                 % Add anomalies only to the first 37 segments
NUM_SIG = min(NUM_SIG_TARGET, length(wavFiles));

% Level mode selection:
% "S"   strong only
% "M"   mid only
% "W"   weak only
% "SMW" all three levels allowed
%       (map Gaussian strength trend to strong/mid/weak)
LEVEL_MODE = "M";

SAVE_OTHERS_AS_NORMAL = true;        % Whether to save files after the first 37 unchanged

%% ===== Event window (service missing only within the window) =====
EVENT_MODE = "random";  % "fixed" / "random"
EVENT_T0 = 0.60;        % used for fixed mode
EVENT_T1 = 1.20;

RAND_T0_RANGE  = [0.15, 1.00];

%
RAND_LEN_RANGE = [0.90, 1.50];       % Original V4: [0.60, 1.30]
MIN_EVENT_LEN  = 0.60;

%% ===== Control of the "Gaussian envelope" within the event window =====
% ===== V4_step1 modification 2: flatter envelope (wider coverage) =====
ENV_SIGMA_RATIO = 0.30;              % Original V4: 0.22
ENV_FLOOR_EDGE  = 0.20;              % V4: edge decays to 20%

%% ===== 37-segment "Gaussian strength plan" g_plan =====
sigma_i = NUM_SIG_TARGET/5;
center  = (NUM_SIG_TARGET+1)/2;
sigma_g = 0.08;

g_plan = zeros(NUM_SIG,1);
for ii = 1:NUM_SIG
    mu = exp(-0.5*((ii - center)/sigma_i)^2); % 0~1
    g  = mu + sigma_g*randn;
    g  = min(max(g,0),1);
    g_plan(ii) = g;
end

%% ===== g -> three-level mapping
% (explicit separation among the three levels, not a continuous gradient) =====
THR_STRONG = 0.62;
THR_MID    = 0.30;

level_plan = strings(NUM_SIG,1);
for ii = 1:NUM_SIG
    g = g_plan(ii);
    if g >= THR_STRONG
        level_plan(ii) = "strong";
    elseif g >= THR_MID
        level_plan(ii) = "mid";
    else
        level_plan(ii) = "weak";
    end
end

%% ===== LEVEL_MODE restriction (strong only / mid only / weak only / all three) =====
for ii = 1:NUM_SIG
    if LEVEL_MODE == "S"
        level_plan(ii) = "strong";
    elseif LEVEL_MODE == "M"
        level_plan(ii) = "mid";
    elseif LEVEL_MODE == "W"
        level_plan(ii) = "weak";
    elseif LEVEL_MODE == "SMW"
        % keep
    else
        error('LEVEL_MODE must be "S"/"M"/"W"/"SMW".');
    end
end

%% ===== Base parameters for the three levels =====
% strong
K_S0      = 85;
beta_S0   = 0.25;
halfBW_S0 = 10;
timeA_S0  = 0.55;
mix_S0    = 0.75;

% mid
K_M0      = 120;
beta_M0   = 0.45;
halfBW_M0 = 6;
timeA_M0  = 0.75;
mix_M0    = 0.55;

% weak
K_W0      = 160;
beta_W0   = 0.70;
halfBW_W0 = 3;
timeA_W0  = 0.90;
mix_W0    = 0.35;

%% ===== Main loop =====
for i = 1:length(wavFiles)
    disp(['Processing file No. ', num2str(i)]);

    name = wavFiles(i).name;
    filePath = fullfile(wavFiles(i).folder, name);

    [data, fs] = audioread(filePath);
    if size(data,2) < 2
        warning('File %s is not dual-channel (I/Q). Skipped.', name);
        continue;
    end

    IQ = data(:,1) + 1j*data(:,2);
    Nall = length(IQ);
    Tall = Nall/fs;

    %% Save unchanged after the first 37
    if i > NUM_SIG
        if SAVE_OTHERS_AS_NORMAL
            audiowrite(fullfile(outDir, name), data, fs);
        end
        fprintf('Not in the first 37 segments: no anomaly added.\n');
        continue;
    end

    level = level_plan(i);
    g = g_plan(i);

    %% Event window
    if EVENT_MODE == "fixed"
        t0 = EVENT_T0; t1 = EVENT_T1;
    else
        t0  = RAND_T0_RANGE(1) + (RAND_T0_RANGE(2)-RAND_T0_RANGE(1))*rand;
        len = RAND_LEN_RANGE(1) + (RAND_LEN_RANGE(2)-RAND_LEN_RANGE(1))*rand;
        len = max(len, MIN_EVENT_LEN);
        t1  = min(t0 + len, Tall - 1e-6);
        if (t1 - t0) < MIN_EVENT_LEN
            t0 = max(0, t1 - MIN_EVENT_LEN);
        end
    end

    idx0 = max(1, floor(t0*fs)+1);
    idx1 = min(Nall, floor(t1*fs));
    if idx1 <= idx0 + round(0.10*fs)
        warning('Event interval is too short. Skipped this file.');
        continue;
    end

    seg = IQ(idx0:idx1);
    L = length(seg);

    %% Select base parameters for the level
    if level == "strong"
        K0 = K_S0; beta0 = beta_S0; halfBW0 = halfBW_S0; timeA0 = timeA_S0; mix0 = mix_S0;
        noiseGain0 = 0.020;
    elseif level == "mid"
        K0 = K_M0; beta0 = beta_M0; halfBW0 = halfBW_M0; timeA0 = timeA_M0; mix0 = mix_M0;
        noiseGain0 = 0.012;
    else
        K0 = K_W0; beta0 = beta_W0; halfBW0 = halfBW_W0; timeA0 = timeA_W0; mix0 = mix_W0;
        noiseGain0 = 0.007;
    end

    %% Intra-level strength variation (without crossing levels)
    s = 0.30 + 0.70*g; % 0.30~1.00
    K      = K0    * (1.18 - 0.45*s);
    beta   = beta0 * (1.30 - 0.55*s);
    halfBW = round(halfBW0 * (0.85 + 0.55*s));
    timeA  = timeA0 * (1.20 - 0.40*s);
    mixR   = mix0  * (0.88 + 0.12*s);
    noiseGain = noiseGain0 * (0.85 + 0.30*s);

    % Clamp values
    K = max(K, 5);
    beta = max(min(beta, 0.95), 1e-4);
    halfBW = max(halfBW, 2);
    timeA = max(min(timeA, 0.98), 0.02);
    mixR = max(min(mixR, 1.0), 0.05);

    fprintf('A4(V4_step1): i=%d level=%s g=%.3f | K=%.2f beta=%.4f halfBW=%d timeA=%.3f mix=%.2f | win=%.3f-%.3f\n', ...
        i, level, g, K, beta, halfBW, timeA, mixR, t0, t1);

    %% ========= Gaussian envelope within the event window
    % (no crossfade) =========
    n = (0:L-1)';
    mu_n = (L-1)/2;
    sig_n = max(8, round(ENV_SIGMA_RATIO * L));

    w_env = exp(-0.5 * ((n - mu_n) / (sig_n + eps)).^2);
    w_env = w_env / (max(w_env) + eps);                  % 0~1
    w_env = ENV_FLOOR_EDGE + (1-ENV_FLOOR_EDGE) * w_env; % edge=0.2, center=1

    %% ========= Original logic: threshold = K * mean(abs(X)) =========
    X = fft(seg);
    mag = abs(X);
    mabs = mean(mag) + eps;

    threshold = K * mabs;
    peak_idx = mag > threshold;

    if ~any(peak_idx)
        [~, imax] = max(mag);
        peak_idx(imax) = true;
    end

    % Expand neighborhood bandwidth
    win = 2*halfBW + 1;
    peak_idx = conv(double(peak_idx), ones(win,1), 'same') > 0;

    %% ========= Replace amplitude =========
    Xf = X;
    ph = angle(Xf(peak_idx));
    Xf(peak_idx) = (mabs * beta) .* exp(1j*ph);

    seg_f = ifft(Xf);

    %% ========= Use the Gaussian envelope to control
    % time-domain silence degree & mix weight =========
    timeA_t = 1 - w_env * (1 - timeA);
    seg_f = seg_f .* timeA_t;

    % Noise (stronger in the center, weaker at the edges)
    segStd = std(seg);
    if ~isfinite(segStd) || segStd <= 0, segStd = 1; end
    noise = (randn(L,1) + 1j*randn(L,1)) / sqrt(2);
    seg_f = seg_f + (noiseGain * segStd) .* noise .* w_env;

    %% ========= Mix (no crossfade) =========
    w = mixR * w_env;                 % edge=0.2*mixR, center=1*mixR
    seg_out = (1-w).*seg + w.*seg_f;

    %% ========= Refill =========
    IQ_out = IQ;
    IQ_out(idx0:idx1) = seg_out;

    out = [real(IQ_out), imag(IQ_out)];

    % Anti-clipping
    mx = max(abs(out), [], 'all');
    if mx > 0.98
        out = out * (0.98/mx);
    end

    %% ========= Save =========
    baseName = erase(name, '.wav');
    outName = sprintf('%s_A4refV4step1_%s_g%.3f_K%.1f_b%.4f_bw%d_att%.2f_m%.2f_edge%.2f_sig%.2f_t%.3f-%.3f.wav', ...
        baseName, level, g, K, beta, halfBW, timeA, mixR, ENV_FLOOR_EDGE, ENV_SIGMA_RATIO, t0, t1);

    audiowrite(fullfile(outDir, outName), out, fs);
    fprintf('Processing of this signal is complete: %s\n', outName);
end

fprintf('\nAll processing completed. Output directory: %s\n', outDir);