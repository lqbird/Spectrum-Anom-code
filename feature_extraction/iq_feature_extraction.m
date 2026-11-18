function feature = iq_feature_extraction(inputDirectory, outputMatFile, varargin)
%IQ_FEATURE_EXTRACTION Extract TCCN case-study features from two-channel WAVs.
%   feature = iq_feature_extraction(inputDirectory, outputMatFile) reads the
%   WAV files in inputDirectory and saves the struct array feature to a MAT
%   file. WAV channel 1 is I and channel 2 is Q. By default, characters 25:32
%   of each file name encode the time as MMDDHHMM. The original directory
%   listing order and feature formulas are retained.
%
%   See README.md for name-value options and the output field definitions.

    isText = @(x) (ischar(x) && isrow(x) && ~isempty(x)) || ...
                 (isstring(x) && isscalar(x) && ~ismissing(x) && strlength(x) > 0);
    isFlag = @(x) islogical(x) && isscalar(x);
    isNonnegative = @(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x) && x >= 0;
    isPositive = @(x) isNonnegative(x) && x > 0;
    isPosition = @(x) isPositive(x) && x == fix(x);
    isTimeRange = @(x) isnumeric(x) && isreal(x) && numel(x) == 2 && ...
        all(isfinite(x(:))) && all(x(:) == fix(x(:))) && x(1) >= 1 && x(2) - x(1) == 7;

    p = inputParser;
    p.FunctionName = mfilename;
    p.PartialMatching = false;
    addRequired(p, 'inputDirectory', isText);
    addRequired(p, 'outputMatFile', isText);
    addParameter(p, 'TimestampRange', [25, 32], isTimeRange);
    addParameter(p, 'SortByTimestamp', false, isFlag);
    addParameter(p, 'UseGPU', true, isFlag);
    addParameter(p, 'ComputePeakSimilarity', true, isFlag);
    addParameter(p, 'PeakProminence', 10000, isNonnegative);
    addParameter(p, 'PeakSigmaHz', 5, isPositive);
    addParameter(p, 'ThresholdFactor', 3, isPositive);
    addParameter(p, 'InitialReferenceLength', 112000000, isPosition);
    addParameter(p, 'Overwrite', false, isFlag);
    parse(p, inputDirectory, outputMatFile, varargin{:});
    cfg = p.Results;
    cfg.PeakProminence = double(cfg.PeakProminence);
    cfg.ThresholdFactor = double(cfg.ThresholdFactor);
    cfg.InitialReferenceLength = double(cfg.InitialReferenceLength);
    inputDirectory = char(inputDirectory);
    outputMatFile = char(outputMatFile);

    if ~isfolder(inputDirectory)
        error('iq_feature_extraction:InputDirectory', ...
            'Input directory not found: %s', inputDirectory);
    end
    [outputDirectory, ~, extension] = fileparts(outputMatFile);
    if ~strcmpi(extension, '.mat') || isfolder(outputMatFile)
        error('iq_feature_extraction:OutputPath', ...
            'Provide an output file path ending in .mat.');
    end
    checkOutput(outputMatFile, cfg.Overwrite);

    wavFiles = dir(fullfile(inputDirectory, '*.wav'));
    wavFiles = wavFiles(~[wavFiles.isdir]);
    if isempty(wavFiles)
        error('iq_feature_extraction:NoWavFiles', ...
            'No .wav files found in %s.', inputDirectory);
    end

    times = zeros(1, numel(wavFiles));
    for k = 1:numel(wavFiles)
        times(k) = readTime(wavFiles(k).name, cfg.TimestampRange);
    end
    if cfg.SortByTimestamp
        [times, order] = sort(times);
        wavFiles = wavFiles(order);
    end

    includePeakSimilarity = cfg.ComputePeakSimilarity && numel(wavFiles) > 1;
    if includePeakSimilarity && exist('findpeaks', 'file') == 0
        error('iq_feature_extraction:MissingFindpeaks', ...
            ['Peak similarity requires findpeaks from Signal Processing Toolbox. ', ...
             'Use ''ComputePeakSimilarity'', false to extract only the nine statistical features.']);
    end
    if cfg.UseGPU
        try
            gpuDevice();
        catch cause
            error('iq_feature_extraction:GPUUnavailable', ...
                ['GPU mode requires Parallel Computing Toolbox and a compatible GPU. ', ...
                 'Use ''UseGPU'', false for a CPU run. Details: %s'], cause.message);
        end
    end

    fields = struct('time', [], 'mean_spectrum', [], 'var_spectrum', [], ...
        'enp_spectrum', [], 'mean_IQ', [], 'var_IQ', [], 'KL_spectrum', [], ...
        'RMSE_spectrum', [], 'CS_spectrum', [], 'exceeding_count', []);
    if includePeakSimilarity
        fields.peakSimilar_spectrum = [];
    end
    feature = repmat(fields, 1, numel(wavFiles));
    previousSpectrum = [];
    expectedFs = [];

    for k = 1:numel(wavFiles)
        name = wavFiles(k).name;
        fprintf('[%d/%d] %s (%08d)\n', k, numel(wavFiles), name, times(k));
        [item, previousSpectrum, expectedFs] = extractOne( ...
            fullfile(inputDirectory, name), times(k), previousSpectrum, expectedFs, fields, cfg);
        feature(k) = item;
    end

    if ~isempty(outputDirectory) && ~isfolder(outputDirectory)
        [ok, message] = mkdir(outputDirectory);
        if ~ok
            error('iq_feature_extraction:OutputDirectory', ...
                'Could not create the output directory: %s', message);
        end
    end
    checkOutput(outputMatFile, cfg.Overwrite);
    save(outputMatFile, 'feature');
end

function timestamp = readTime(fileName, positions)
    [~, stem] = fileparts(fileName);
    if numel(stem) < positions(2)
        error('iq_feature_extraction:BadTimestamp', ...
            'File name %s is too short for TimestampRange [%d %d].', ...
            fileName, positions(1), positions(2));
    end
    token = stem(positions(1):positions(2));
    if isempty(regexp(token, '^[0-9]{8}$', 'once'))
        error('iq_feature_extraction:BadTimestamp', ...
            'Expected eight digits at TimestampRange [%d %d] in %s.', ...
            positions(1), positions(2), fileName);
    end
    timestamp = str2double(token);
end

function checkOutput(path, overwrite)
    if isfile(path) && ~overwrite
        error('iq_feature_extraction:OutputExists', ...
            'Output already exists: %s. Use a new path or set ''Overwrite'', true.', path);
    end
end

function [item, spectrum, fs] = extractOne(path, timestamp, previousSpectrum, expectedFs, item, cfg)
    [data, fs] = audioread(path);
    if isempty(data) || size(data, 2) ~= 2
        error('iq_feature_extraction:BadWavChannels', ...
            'Expected a nonempty WAV with exactly two columns: I and Q. File: %s', path);
    end
    if ~isempty(expectedFs) && fs ~= expectedFs
        error('iq_feature_extraction:MixedSampleRates', ...
            'Expected %g Hz but found %g Hz in %s. Use one sampling rate per run.', ...
            expectedFs, fs, path);
    end
    if cfg.UseGPU
        data = gpuArray(data);
    end
    sig = data(:, 1) + 1i .* data(:, 2);
    clear data
    spectrum = abs(fft(sig));
    envelope = abs(sig);
    clear sig

    item.time = timestamp;
    item.mean_IQ = toHost(mean(envelope));
    item.var_IQ = toHost(var(envelope));
    clear envelope
    meanSpectrum = mean(spectrum);
    item.mean_spectrum = toHost(meanSpectrum);
    item.var_spectrum = toHost(var(spectrum));
    normalized = spectrum / (sum(spectrum) + eps);
    item.enp_spectrum = toHost(-sum(normalized .* log2(normalized + eps)));
    clear normalized

    % Align prefixes as in the original script. Retain the complete current
    % spectrum as the next reference, even when the current pair is cropped.
    lenNow = numel(spectrum);
    firstObservation = isempty(previousSpectrum);
    if firstObservation
        lenPrevious = cfg.InitialReferenceLength;
        previousAligned = ones(min(lenNow, lenPrevious), 1, 'like', spectrum);
    else
        lenPrevious = numel(previousSpectrum);
        if lenPrevious > lenNow
            previousAligned = previousSpectrum(1:lenNow);
        else
            previousAligned = previousSpectrum;
        end
    end
    if lenNow > lenPrevious
        currentAligned = spectrum(1:lenPrevious);
    else
        currentAligned = spectrum;
    end

    P = currentAligned / (sum(currentAligned) + eps);
    Q = previousAligned / (sum(previousAligned) + eps);
    P(P == 0) = eps;
    Q(Q == 0) = eps;
    item.KL_spectrum = toHost(sum(sum(P .* log(eps + P ./ (Q + eps)))));
    clear P Q

    % This historical field is abs(mean(difference)), not sqrt(mean(difference.^2)).
    item.RMSE_spectrum = toHost(sqrt(mean(currentAligned - previousAligned).^2));
    item.CS_spectrum = toHost(dot(currentAligned, previousAligned) / ...
        (norm(currentAligned) * norm(previousAligned)));
    exceeding = currentAligned > cfg.ThresholdFactor * meanSpectrum;
    item.exceeding_count = toHost(sum(exceeding) / numel(currentAligned) * 100);
    clear exceeding

    if ~firstObservation && isfield(item, 'peakSimilar_spectrum')
        item.peakSimilar_spectrum = spectral_peak_similarity( ...
            currentAligned, previousAligned, fs, lenNow, lenPrevious, ...
            cfg.PeakProminence, cfg.PeakSigmaHz);
    end
end

function value = toHost(value)
    if isa(value, 'gpuArray')
        value = gather(value);
    end
end
