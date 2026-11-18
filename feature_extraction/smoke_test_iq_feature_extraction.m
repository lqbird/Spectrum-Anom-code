function smoke_test_iq_feature_extraction()
%SMOKE_TEST_IQ_FEATURE_EXTRACTION Check the packaged extraction on the CPU.
    workDir = tempname;
    mkdir(workDir);
    cleanup = onCleanup(@() rmdir(workDir, 's')); %#ok<NASGU>
    wavDir = fullfile(workDir, 'wav');
    mkdir(wavDir);
    outFile = fullfile(workDir, 'results', 'features.mat');

    fs = 8000;
    tokens = {'08140930', '08140940', '08141020'};
    lengths = [128, 192, 160];
    referenceLength = 64;
    wavPaths = cell(1, numel(lengths));
    for k = 1:numel(lengths)
        t = (0:lengths(k)-1).' / fs;
        i = 0.35 * cos(2*pi*(260 + 75*k)*t) + 0.2*cos(2*pi*600*t);
        q = 0.3 * sin(2*pi*(260 + 75*k)*t);
        wavPaths{k} = fullfile(wavDir, [repmat('a', 1, 24), tokens{k}, '.wav']);
        audiowrite(wavPaths{k}, [i, q], fs);
    end

    cfg = {'UseGPU', false, 'ComputePeakSimilarity', false, ...
        'InitialReferenceLength', referenceLength};
    features = iq_feature_extraction(wavDir, outFile, cfg{:});
    assert(isequal([features.time], [8140930, 8140940, 8141020]));
    assert(~isfield(features, 'peakSimilar_spectrum'));
    saved = load(outFile, 'feature');
    assert(isequaln(saved.feature, features));

    expected = zeros(numel(lengths), 9);
    previous = ones(referenceLength, 1);
    for k = 1:numel(lengths)
        x = audioread(wavPaths{k});
        z = complex(x(:,1), x(:,2));
        a = abs(z);
        s = abs(fft(z));
        q = s / (sum(s) + eps);
        h = -sum(q .* log2(q + eps));
        common = min(length(s), length(previous));
        u = s(1:common);
        v = previous(1:common);
        p = u / (sum(u) + eps);
        q = v / (sum(v) + eps);
        p(p == 0) = eps;
        q(q == 0) = eps;
        expected(k, :) = [mean(s), var(s), h, mean(a), var(a), ...
            sum(p .* log(eps + p ./ (q + eps))), abs(mean(u - v)), ...
            dot(u, v) / (norm(u)*norm(v)), sum(u > 3*mean(s))/common*100];
        previous = s;
    end
    actual = [[features.mean_spectrum].', [features.var_spectrum].', ...
        [features.enp_spectrum].', [features.mean_IQ].', [features.var_IQ].', ...
        [features.KL_spectrum].', [features.RMSE_spectrum].', ...
        [features.CS_spectrum].', [features.exceeding_count].'];
    relativeError = abs(actual - expected) ./ max(1, abs(expected));
    assert(all(relativeError(:) < 1e-10));

    try
        iq_feature_extraction(wavDir, outFile, cfg{:});
        error('smoke_test_iq_feature_extraction:OverwriteWasAllowed', ...
            'The extractor should reject an existing output by default.');
    catch cause
        assert(strcmp(cause.identifier, 'iq_feature_extraction:OutputExists'));
    end

    if exist('findpeaks', 'file') ~= 0
        testPeakSimilarity(fs);
    end
    fprintf('I/Q feature extraction smoke test passed.\n');
end

function testPeakSimilarity(fs)
    a = [0; 2; 0; 0; 3; 0];
    b = [0; 2; 0; 1; 0; 3; 0; 0];
    lenA = 8;
    lenB = 8;
    sigmaHz = 1000;
    fa = (0:numel(a)-1) * (fs/lenA);
    fb = (0:numel(b)-1) * (fs/lenB);
    [pa, xa] = findpeaks(a, fa, 'MinPeakProminence', 0.5);
    [pb, xb] = findpeaks(b, fb, 'MinPeakProminence', 0.5);
    dense = zeros(numel(pa), numel(pb));
    for i = 1:numel(pa)
        for j = 1:numel(pb)
            dense(i, j) = exp(-(xa(i) - xb(j))^2/(2*sigmaHz^2)) * ...
                (1 - abs(pa(i) - pb(j)) / max(pa(i), pb(j)));
        end
    end
    expected = (mean(max(dense, [], 2)) + mean(max(dense, [], 1))) / 2;
    actual = spectral_peak_similarity(a, b.', fs, lenA, lenB, 0.5, sigmaHz);
    assert(abs(actual - expected) < 1e-12);
    assert(isnan(spectral_peak_similarity(zeros(size(a)), b, fs, lenA, lenB, 0.5, sigmaHz)));
end
