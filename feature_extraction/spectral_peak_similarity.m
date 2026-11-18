function finalSim = spectral_peak_similarity(sig1, sig2, Fs, N1, N2, P, sigma_f)
%SPECTRAL_PEAK_SIMILARITY Compare peaks in two unshifted FFT magnitude spectra.
%   sig1 and sig2 are spectra (not time-domain I/Q samples). N1 and N2 are
%   the original FFT lengths, even when only leading bins are compared.
%   Fs and sigma_f are in Hz. P is the minimum peak prominence.
%   The score averages the best matches in both directions. If either
%   spectrum has no qualifying peaks, the score is NaN.

    validateattributes(Fs, {'numeric'}, {'real', 'scalar', 'finite', 'positive'});
    validateattributes(N1, {'numeric'}, {'real', 'scalar', 'finite', 'integer', 'positive'});
    validateattributes(N2, {'numeric'}, {'real', 'scalar', 'finite', 'integer', 'positive'});
    validateattributes(P, {'numeric'}, {'real', 'scalar', 'finite', 'nonnegative'});
    validateattributes(sigma_f, {'numeric'}, {'real', 'scalar', 'finite', 'positive'});
    P = double(P);
    sigma_f = double(sigma_f);
    if isa(sig1, 'gpuArray')
        sig1 = gather(sig1);
    end
    if isa(sig2, 'gpuArray')
        sig2 = gather(sig2);
    end
    validateattributes(sig1, {'single', 'double'}, {'real', 'vector', 'finite', 'nonnegative'});
    validateattributes(sig2, {'single', 'double'}, {'real', 'vector', 'finite', 'nonnegative'});
    if numel(sig1) > N1 || numel(sig2) > N2
        error('spectral_peak_similarity:BadFFTLength', ...
            'Each spectrum must be a full FFT magnitude or a prefix of the given FFT length.');
    end
    if numel(sig1) < 3 || numel(sig2) < 3
        finalSim = NaN;
        return
    end

    % Use peak indices to avoid allocating a full frequency axis for each FFT.
    [pks1, locs1] = findpeaks(sig1(:), 'MinPeakProminence', P);
    [pks2, locs2] = findpeaks(sig2(:), 'MinPeakProminence', P);
    if isempty(pks1) || isempty(pks2)
        finalSim = NaN;
        return
    end
    freq1 = (double(locs1(:)) - 1) * (double(Fs) / double(N1));
    freq2 = (double(locs2(:)) - 1) * (double(Fs) / double(N2));
    pks1 = double(pks1(:));
    pks2 = double(pks2(:));

    % Same pairwise frequency and amplitude scores as the original, without
    % storing the full matrix of peak pairs or relying on input orientation.
    blockSize = 1000;
    best1 = zeros(size(pks1));
    best2 = zeros(size(pks2));
    for first1 = 1:blockSize:numel(pks1)
        idx1 = first1:min(first1 + blockSize - 1, numel(pks1));
        for first2 = 1:blockSize:numel(pks2)
            idx2 = first2:min(first2 + blockSize - 1, numel(pks2));
            freqDiff = freq1(idx1) - freq2(idx2).';
            freqSim = exp(-(freqDiff).^2 / (2 * sigma_f^2));
            amp1 = pks1(idx1);
            amp2 = pks2(idx2).';
            ampSim = 1 - abs(amp1 - amp2) ./ max(amp1, amp2);
            score = freqSim .* ampSim;
            best1(idx1) = max(best1(idx1), max(score, [], 2));
            best2(idx2) = max(best2(idx2), max(score, [], 1).');
        end
    end
    finalSim = (mean(best1) + mean(best2)) / 2;
end
