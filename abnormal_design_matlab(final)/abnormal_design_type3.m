%% A3 Anomaly Pattern 3: Background-Noise Suppression Type
% (Wavelet Denoising, steady version: MAD + Universal Threshold)
% Objective:
% (A) Threshold: replace max(C) with MAD noise estimation + universal threshold (most critical)
% (B) Separate and widen the alpha ranges for strong / medium / weak (make levels more distinct)
% (C) Increase the minimum event length to avoid being smoothed out by window averaging / post-processing
%
% Dependency: Wavelet Toolbox (wavedec / waverec / detcoef / wthresh)
% Description: apply wavelet soft-threshold denoising to I/Q only within the event window [t0,t1];
% all other time intervals remain unchanged

clear; clc; close all;

%% ========== Path configuration ==========
inDir  = 'E:\添加异常段（单段and间歇）\未添加\单段未添加异常数据';
outDir = 'E:\添加异常段（单段and间歇）\噪声减弱';

if ~exist(outDir,'dir'), mkdir(outDir); end
wavFiles = dir(fullfile(inDir,'*.wav'));

%% ========== Global control ==========
rng(42);                % Reproducible; comment out if fully random behavior is desired
WRITE_FILE = true;
SHOW_PLOT  = false;
SAFE_PEAK  = 0.98;

%% ========== Process only the first 37 segments
% (Gaussian strength plan) ==========
NUM_SIG_TARGET = 37;
NUM_SIG = min(NUM_SIG_TARGET, length(wavFiles));

%% ========== LEVEL mode ==========
% 'S'   : strong only
% 'SM'  : strong + medium
% 'SMW' : strong + medium + weak
LEVEL_MODE = 'SMW';

%% ========== Event window (steady: denoising persists throughout the window) ==========
EVENT_MODE = 'random';     % 'fixed' or 'random'
EVENT_T0 = 0.50;
EVENT_T1 = 1.50;

% (C) Slightly increase the lower bound of the event length
RAND_T0_RANGE      = [0.20, 1.10];   % Start-time range
RAND_LEN_RANGE     = [0.25, 0.45];   % (C) Duration range: lower bound changed from 0.15 -> 0.25
MIN_EVENT_LEN_SEC  = 0.25;           % (C) Hard minimum event length constraint (to avoid being suppressed)

%% ========== Wavelet denoising parameters (A) ==========
WNAME  = 'db4';
WLEVEL = 4;

% (A) Use MAD + universal threshold:
% thr = kappa * sigma_hat * sqrt(2*log(N))
% where sigma_hat is estimated from the MAD of the finest-scale detail coefficients d1:
% sigma_hat = median(|d1|)/0.6745
%
% Larger kappa => stronger denoising => more obvious "noise-floor reduction"

LEVELS(1).name = 'strong';
LEVELS(1).kappa_range = [1.60, 2.10];      % Strong: larger threshold
LEVELS(1).alpha_range = [0.94, 0.995];     % (B) Strong: nearly full replacement (no overlap)

LEVELS(2).name = 'medium';
LEVELS(2).kappa_range = [1.10, 1.45];      % Medium: moderate threshold
LEVELS(2).alpha_range = [0.78, 0.88];      % (B) Medium: clearly lower than strong, with no overlap

LEVELS(3).name = 'weak';
LEVELS(3).kappa_range = [0.70, 0.95];      % Weak: smaller threshold (but still visible)
LEVELS(3).alpha_range = [0.45, 0.62];      % (B) Weak: further reduced, with no overlap

%% ========== 37-segment Gaussian strength plan
% (weaker / medium at both ends, stronger in the middle, with slight randomness) ==========
sigma_i = NUM_SIG_TARGET/5;
center  = (NUM_SIG_TARGET+1)/2;
sigma_g = 0.08;

g_plan = zeros(NUM_SIG,1);
for ii = 1:NUM_SIG
    mu = exp(-0.5*((ii - center)/sigma_i)^2);  % 0~1
    g  = mu + sigma_g*randn;
    g  = min(max(g, 0), 1);
    g_plan(ii) = g;
end

THR_STRONG = 0.62;
THR_MEDIUM = 0.30;

level_plan = zeros(NUM_SIG,1);
for ii = 1:NUM_SIG
    g = g_plan(ii);
    if g >= THR_STRONG
        level_plan(ii) = 1; % strong
    elseif g >= THR_MEDIUM
        level_plan(ii) = 2; % medium
    else
        level_plan(ii) = 3; % weak
    end
end

% LEVEL_MODE constraint
for ii = 1:NUM_SIG
    if strcmp(LEVEL_MODE,'S')
        level_plan(ii) = 1;
    elseif strcmp(LEVEL_MODE,'SM')
        level_plan(ii) = min(level_plan(ii), 2);
    elseif strcmp(LEVEL_MODE,'SMW')
        % keep
    else
        error('LEVEL_MODE must be ''S'' / ''SM'' / ''SMW''.');
    end
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

    % Event window
    if strcmp(EVENT_MODE,'fixed')
        t0 = EVENT_T0; 
        t1 = EVENT_T1;
    else
        t0  = RAND_T0_RANGE(1) + (RAND_T0_RANGE(2)-RAND_T0_RANGE(1))*rand;

        len = RAND_LEN_RANGE(1) + (RAND_LEN_RANGE(2)-RAND_LEN_RANGE(1))*rand;
        len = max(len, MIN_EVENT_LEN_SEC);    % (C) Hard guarantee

        t1  = min(t0 + len, T-1e-6);

        % If t1 is truncated and causes len < MIN_EVENT_LEN_SEC, shift t0 backward
        if (t1 - t0) < MIN_EVENT_LEN_SEC
            t0 = max(0, t1 - MIN_EVENT_LEN_SEC);
        end
    end

    idx0 = max(1, floor(t0*fs)+1);
    idx1 = min(N, floor(t1*fs));
    if idx1 <= idx0 + 32
        warning('Event window is too short (samples=%d). Skipped this file.', idx1-idx0+1);
        continue;
    end

    % Select level
    level_id   = level_plan(i);
    level_name = LEVELS(level_id).name;

    % Sample kappa and alpha within the selected level
    kr = LEVELS(level_id).kappa_range;
    ar = LEVELS(level_id).alpha_range;
    kappa = kr(1) + (kr(2)-kr(1))*rand;
    alpha = ar(1) + (ar(2)-ar(1))*rand;

    fprintf('LEVEL_MODE=%s, planned level=%s, g=%.3f, kappa=%.3f, alpha=%.3f, window=%.3f-%.3f sec (len=%.3f)\n', ...
        LEVEL_MODE, level_name, g_plan(i), kappa, alpha, t0, t1, (t1-t0));

    % Apply denoising only within the event window
    seg = IQ(idx0:idx1);
    xI  = real(seg);
    xQ  = imag(seg);

    % (A) Wavelet soft-threshold denoising using MAD + universal threshold
    xI_dn = wavelet_denoise_mad_univ(xI, WNAME, WLEVEL, kappa);
    xQ_dn = wavelet_denoise_mad_univ(xQ, WNAME, WLEVEL, kappa);

    % (B) Mixing: the larger alpha is, the more "silent" the output becomes
    % (i.e., the more obvious the noise-floor reduction)
    xI_mix = (1-alpha)*xI + alpha*xI_dn;
    xQ_mix = (1-alpha)*xQ + alpha*xQ_dn;

    seg_out = xI_mix + 1j*xQ_mix;

    IQ_out = IQ;
    IQ_out(idx0:idx1) = seg_out;

    out = [real(IQ_out), imag(IQ_out)];
    mx = max(abs(out), [], 'all');
    if mx > SAFE_PEAK
        out = out * (SAFE_PEAK / mx);
    end

    if SHOW_PLOT
        t = (0:N-1).'/fs;

        figure; 
        subplot(2,1,1);
        plot(t(1:min(2000,N)), real(IQ(1:min(2000,N)))); hold on;
        plot(t(1:min(2000,N)), real(IQ_out(1:min(2000,N))), 'r');
        xline(t0,'k--'); xline(t1,'k--');
        legend('Original I','Processed I'); title('Time-Domain I');
        grid on;

        subplot(2,1,2);
        plot(t(1:min(2000,N)), imag(IQ(1:min(2000,N)))); hold on;
        plot(t(1:min(2000,N)), imag(IQ_out(1:min(2000,N))), 'r');
        xline(t0,'k--'); xline(t1,'k--');
        legend('Original Q','Processed Q'); title('Time-Domain Q');
        grid on;
    end

    if WRITE_FILE
        [~, baseName, ext] = fileparts(wavFiles(i).name);
        outName = sprintf('%s_A3_%s_steady_g%.3f_k%.3f_a%.3f_t%.3f-%.3f%s', ...
            baseName, upper(level_name(1)), g_plan(i), kappa, alpha, t0, t1, ext);
        audiowrite(fullfile(outDir, outName), out, fs);
    end

    fprintf('Completed: %s\n', wavFiles(i).name);
end

fprintf('\nAll processing finished. Output directory: %s\n', outDir);

%% ====================== Local function (fixed "index out of bounds" version) ======================
function x_dn = wavelet_denoise_mad_univ(x, wname, wlevel, kappa)
% MAD + Universal-threshold wavelet denoising (soft-threshold)
% thr = kappa * sigma_hat * sqrt(2*log(N))
%
% wavedec returns:
%   C: coefficient vector = [cA_wlevel, cD_wlevel, cD_wlevel-1, ..., cD1]
%   L: length vector      = [lenA_wlevel, lenD_wlevel, ..., lenD1, lenX]
% Note:
% L(end)=lenX is the original signal length. It is not a coefficient block in C,
% so it must not be used to index C.

    x = x(:);
    N = length(x);

    [C, L] = wavedec(x, wlevel, wname);

    % Use the finest-scale detail coefficients d1 for MAD noise estimation
    d1 = detcoef(C, L, 1);
    sigma_hat = median(abs(d1)) / 0.6745;

    if ~isfinite(sigma_hat) || sigma_hat <= 0
        thr = 0;
    else
        thr = kappa * sigma_hat * sqrt(2*log(N));
    end

    C2 = C;

    % Skip the approximation block: 1:L(1)
    pos = L(1) + 1;

    % Traverse detail blocks only: L(2) ... L(end-1)
    % L(end) is lenX (the original signal length), not a coefficient-block length
    for j = 2:(length(L)-1)
        len  = L(j);
        idx1 = pos + len - 1;

        if idx1 > length(C)
            error('Index exceeds C length: pos=%d len=%d idx1=%d | length(C)=%d. Check L format.', ...
                pos, len, idx1, length(C));
        end

        seg = C(pos:idx1);
        C2(pos:idx1) = wthresh(seg, 's', thr);

        pos = idx1 + 1;
    end

    x_dn = waverec(C2, L, wname);

    % waverec should theoretically return length N, but add a safety alignment
    if length(x_dn) > N
        x_dn = x_dn(1:N);
    elseif length(x_dn) < N
        x_dn = [x_dn; zeros(N-length(x_dn),1)];
    end
end