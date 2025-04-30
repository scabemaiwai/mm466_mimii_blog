% Base Path only for 0dB
basePath = 'D:\Users\S11202884\Desktop\MIMII\0dB';  

% Where to save MAT batches
savePath = 'D:\Users\S11202884\Desktop\MIMII\0dB\mat_batches_0dB';
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% Settings
batchSize = 500;  % Number of audio files per batch
batchCounter = 1;
fileCounter = 0;

batchAudio = {};
batchLabels = [];
batchMeta = {};

machineTypes = {'fan', 'pump', 'valve', 'slide_rail'};
conditions = {'normal', 'abnormal'};

% Start conversion
for m = 1:length(machineTypes)
    machine = machineTypes{m};

    for id = 0:6
        idStr = sprintf('id_%02d', id);

        for c = 1:length(conditions)
            cond = conditions{c};
            label = double(strcmp(cond, 'abnormal'));  % 0 = normal, 1 = abnormal

            folderPath = fullfile(basePath, machine, idStr, cond);

            if ~exist(folderPath, 'dir')
                continue;
            end

            files = dir(fullfile(folderPath, '*.wav'));

            for f = 1:length(files)
                try
                    filePath = fullfile(folderPath, files(f).name);
                    [audio, fs] = audioread(filePath);

                    % Force mono
                    if size(audio,2) > 1
                        audio = mean(audio,2);
                    end

                    % Store in batch
                    batchAudio{end+1} = audio;
                    batchLabels(end+1) = label;
                    batchMeta{end+1} = struct('machine', machine, 'id', idStr, ...
                                              'condition', cond, 'filename', files(f).name);

                    fileCounter = fileCounter + 1;

                    % Save when batch is full
                    if fileCounter == batchSize
                        saveName = sprintf('batch_%03d.mat', batchCounter);
                        save(fullfile(savePath, saveName), ...
                            'batchAudio', 'batchLabels', 'batchMeta', '-v7.3');
                        fprintf('Saved %s with %d files\n', saveName, fileCounter);

                        % Reset batch
                        batchAudio = {};
                        batchLabels = [];
                        batchMeta = {};
                        fileCounter = 0;
                        batchCounter = batchCounter + 1;
                    end
                catch ME
                    warning('Skipping file: %s\nReason: %s', filePath, ME.message);
                end
            end
        end
    end
end

% Save any leftovers
if ~isempty(batchAudio)
    saveName = sprintf('batch_%03d.mat', batchCounter);
    save(fullfile(savePath, saveName), ...
        'batchAudio', 'batchLabels', 'batchMeta', '-v7.3');
    fprintf('Saved final %s with %d files\n', saveName, fileCounter);
end

disp('Finished converting 0dB WAV files into MAT batches!');

Converting 6dB to .mat files in batches of 500 files
%  Base Path only for 6dB
basePath = 'D:\Users\S11202884\Desktop\MIMII\6dB';  % <-- Update if your 6dB data is elsewhere

% Where to save MAT batches for 6dB
savePath = 'D:\Users\S11202884\Desktop\MIMII\6dB\mat_batches_6dB';
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

%  Settings
batchSize = 500;  % Number of audio files per batch
batchCounter = 1;
fileCounter = 0;

batchAudio = {};
batchLabels = [];
batchMeta = {};

machineTypes = {'fan', 'pump', 'valve', 'slide_rail'};
conditions = {'normal', 'abnormal'};

% Start conversion
for m = 1:length(machineTypes)
    machine = machineTypes{m};

    for id = 0:6
        idStr = sprintf('id_%02d', id);

        for c = 1:length(conditions)
            cond = conditions{c};
            label = double(strcmp(cond, 'abnormal'));  % 0 = normal, 1 = abnormal

            folderPath = fullfile(basePath, machine, idStr, cond);

            if ~exist(folderPath, 'dir')
                continue;
            end

            files = dir(fullfile(folderPath, '*.wav'));

            for f = 1:length(files)
                try
                    filePath = fullfile(folderPath, files(f).name);
                    [audio, fs] = audioread(filePath);

                    % Force mono if stereo
                    if size(audio,2) > 1
                        audio = mean(audio,2);
                    end

                    % Add to batch
                    batchAudio{end+1} = audio;
                    batchLabels(end+1) = label;
                    batchMeta{end+1} = struct('machine', machine, 'id', idStr, ...
                                              'condition', cond, 'filename', files(f).name);

                    fileCounter = fileCounter + 1;

                    %  Save when batch full
                    if fileCounter == batchSize
                        saveName = sprintf('batch_%03d.mat', batchCounter);
                        save(fullfile(savePath, saveName), ...
                            'batchAudio', 'batchLabels', 'batchMeta', '-v7.3');
                        fprintf(' Saved %s with %d files\n', saveName, fileCounter);

                        % Reset for next batch
                        batchAudio = {};
                        batchLabels = [];
                        batchMeta = {};
                        fileCounter = 0;
                        batchCounter = batchCounter + 1;
                    end
                catch ME
                    warning('Skipping file: %s\nReason: %s', filePath, ME.message);
                end
            end
        end
    end
end

% Save any remaining files
if ~isempty(batchAudio)
    saveName = sprintf('batch_%03d.mat', batchCounter);
    save(fullfile(savePath, saveName), 'batchAudio', 'batchLabels', 'batchMeta', '-v7.3');
    fprintf('Saved final %s with %d files\n', saveName, fileCounter);
end

disp('Finished converting 6dB WAV files into MAT batches!');

Converting -6dB to .mat files
%  Base Path only for -6dB
basePath = 'D:\Users\S11202884\Desktop\MIMII\-6dB';  % <-- your -6dB dataset folder

% Where to save MAT batches for -6dB
savePath = 'D:\Users\S11202884\Desktop\MIMII\-6dB\mat_batches_-6dB';
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% Settings
batchSize = 500;  % Number of audio files per batch
batchCounter = 1;
fileCounter = 0;

batchAudio = {};
batchLabels = [];
batchMeta = {};

machineTypes = {'fan', 'pump', 'valve', 'slide_rail'};
conditions = {'normal', 'abnormal'};

% Start conversion
for m = 1:length(machineTypes)
    machine = machineTypes{m};

    for id = 0:6
        idStr = sprintf('id_%02d', id);

        for c = 1:length(conditions)
            cond = conditions{c};
            label = double(strcmp(cond, 'abnormal'));  % 0 = normal, 1 = abnormal

            folderPath = fullfile(basePath, machine, idStr, cond);

            if ~exist(folderPath, 'dir')
                continue;
            end

            files = dir(fullfile(folderPath, '*.wav'));

            for f = 1:length(files)
                try
                    filePath = fullfile(folderPath, files(f).name);
                    [audio, fs] = audioread(filePath);

                    %  Force mono if stereo
                    if size(audio,2) > 1
                        audio = mean(audio,2);
                    end

                    %  Add to batch
                    batchAudio{end+1} = audio;
                    batchLabels(end+1) = label;
                    batchMeta{end+1} = struct('machine', machine, 'id', idStr, ...
                                              'condition', cond, 'filename', files(f).name);

                    fileCounter = fileCounter + 1;

                    %  Save when batch full
                    if fileCounter == batchSize
                        saveName = sprintf('batch_%03d.mat', batchCounter);
                        save(fullfile(savePath, saveName), 'batchAudio', 'batchLabels', 'batchMeta', '-v7.3');
                        fprintf('Saved %s with %d files\n', saveName, fileCounter);

                        % Reset for next batch
                        batchAudio = {};
                        batchLabels = [];
                        batchMeta = {};
                        fileCounter = 0;
                        batchCounter = batchCounter + 1;
                    end
                catch ME
                    warning('Skipping file: %s\nReason: %s', filePath, ME.message);
                end
            end
        end
    end
end

% Save any remaining files
if ~isempty(batchAudio)
    saveName = sprintf('batch_%03d.mat', batchCounter);
    save(fullfile(savePath, saveName), 'batchAudio', 'batchLabels', 'batchMeta', '-v7.3');
    fprintf(' Saved final %s with %d files\n', saveName, fileCounter);
end

disp(' Finished converting -6dB WAV files into MAT batches!');

fs = 16000;

For 0dB
% Change this for 0dB, -6dB, or 6dB
batchPath = 'D:\Users\S11202884\Desktop\MIMII\0dB\mat_batches_0dB';
saveFile = 'D:\Users\S11202884\Desktop\MIMII\0dB\features_0dB.mat';

% Sampling rate (MIMII standard)
fs = 16000;

% Get all batch file names
batchFiles = dir(fullfile(batchPath, 'batch_*.mat'));

% Initialize outputs
allFeatures = [];
allLabels = [];
allMeta = {};

% Loop through batches
for i = 1:length(batchFiles)
    fprintf("Processing %s (%d of %d)\n", batchFiles(i).name, i, length(batchFiles));
    
    data = load(fullfile(batchPath, batchFiles(i).name));
    batchAudio = data.batchAudio;
    batchLabels = data.batchLabels;
    batchMeta = data.batchMeta;
    
    for j = 1:length(batchAudio)
        x = batchAudio{j};
        
        % Feature 1: RMS Energy
        rmsEnergy = rms(x);
        
        % Feature 2: Zero Crossing Rate
        zcr = sum(abs(diff(x > 0))) / length(x);
        
        %  Feature 3: Spectral Centroid
        X = abs(fft(x));
        X = X(1:floor(length(X)/2));
        freqs = linspace(0, fs/2, length(X));
        centroid = sum(freqs .* X') / sum(X + eps);
        
        % Store features + labels + meta
        allFeatures(end+1, :) = [rmsEnergy, zcr, centroid];
        allLabels(end+1) = batchLabels(j);
        allMeta{end+1} = batchMeta{j};
    end
end

% Save extracted features
save(saveFile, 'allFeatures', 'allLabels', 'allMeta');
disp(" Feature extraction complete and saved.");

6dB
% Change this for 0dB, -6dB, or 6dB
batchPath = 'D:\Users\S11202884\Desktop\MIMII\6dB\mat_batches_6dB';
saveFile = 'D:\Users\S11202884\Desktop\MIMII\6dB\features_6dB.mat';

% Sampling rate (MIMII standard)
fs = 16000;

% Get all batch file names
batchFiles = dir(fullfile(batchPath, 'batch_*.mat'));

% Initialize outputs
allFeatures = [];
allLabels = [];
allMeta = {};

% Loop through batches
for i = 1:length(batchFiles)
    fprintf("Processing %s (%d of %d)\n", batchFiles(i).name, i, length(batchFiles));
    
    data = load(fullfile(batchPath, batchFiles(i).name));
    batchAudio = data.batchAudio;
    batchLabels = data.batchLabels;
    batchMeta = data.batchMeta;

    for j = 1:length(batchAudio)
        x = batchAudio{j};

        % Feature 1: RMS Energy
        rmsEnergy = rms(x);

        % Feature 2: Zero Crossing Rate
        zcr = sum(abs(diff(x > 0))) / length(x);

        %  Feature 3: Spectral Centroid
        X = abs(fft(x));
        X = X(1:floor(length(X)/2));
        freqs = linspace(0, fs/2, length(X));
        centroid = sum(freqs .* X') / sum(X + eps);

        % Store features + labels + meta
        allFeatures(end+1, :) = [rmsEnergy, zcr, centroid];
        allLabels(end+1) = batchLabels(j);
        allMeta{end+1} = batchMeta{j};
    end
end

% Save extracted features
save(saveFile, 'allFeatures', 'allLabels', 'allMeta');
disp(" Feature extraction complete and saved.");

-6dB
% Change this for 0dB, -6dB, or 6dB
batchPath = 'D:\Users\S11202884\Desktop\MIMII\-6dB\mat_batches_-6dB';
saveFile = 'D:\Users\S11202884\Desktop\MIMII\-6dB\features_-6dB.mat';

% Sampling rate (MIMII standard)
fs = 16000;

% Get all batch file names
batchFiles = dir(fullfile(batchPath, 'batch_*.mat'));

% Initialize outputs
allFeatures = [];
allLabels = [];
allMeta = {};

% Loop through batches
for i = 1:length(batchFiles)
    fprintf("Processing %s (%d of %d)\n", batchFiles(i).name, i, length(batchFiles));
    
    data = load(fullfile(batchPath, batchFiles(i).name));
    batchAudio = data.batchAudio;
    batchLabels = data.batchLabels;
    batchMeta = data.batchMeta;

    for j = 1:length(batchAudio)
        x = batchAudio{j};

        % Feature 1: RMS Energy
        rmsEnergy = rms(x);

        % Feature 2: Zero Crossing Rate
        zcr = sum(abs(diff(x > 0))) / length(x);

        %  Feature 3: Spectral Centroid
        X = abs(fft(x));
        X = X(1:floor(length(X)/2));
        freqs = linspace(0, fs/2, length(X));
        centroid = sum(freqs .* X') / sum(X + eps);

        % Store features + labels + meta
        allFeatures(end+1, :) = [rmsEnergy, zcr, centroid];
        allLabels(end+1) = batchLabels(j);
        allMeta{end+1} = batchMeta{j};
    end
end

% Save extracted features
save(saveFile, 'allFeatures', 'allLabels', 'allMeta');
disp(" Feature extraction complete and saved.");

disp(size(features0dB.allFeatures));
disp(size(featuresNeg6.allFeatures));
disp(size(features6dB.allFeatures));

size(features0dB.allLabels)
size(featuresNeg6.allLabels)
size(features6dB.allLabels)

% === Load individual feature sets ===
f0 = load('D:\Users\S11202884\Desktop\MIMII\0dB\features_0dB.mat');
fN6 = load('D:\Users\S11202884\Desktop\MIMII\-6dB\features_-6dB.mat');
f6 = load('D:\Users\S11202884\Desktop\MIMII\6dB\features_6dB.mat');

% === Fix shapes and types ===
f0.allFeatures  = double(f0.allFeatures);
fN6.allFeatures = double(fN6.allFeatures);
f6.allFeatures  = double(f6.allFeatures);

f0.allLabels  = double(f0.allLabels(:));
fN6.allLabels = double(fN6.allLabels(:));
f6.allLabels  = double(f6.allLabels(:));

f0.allMeta  = f0.allMeta(:);
fN6.allMeta = fN6.allMeta(:);
f6.allMeta  = f6.allMeta(:);

% === Sanity checks ===
assert(size(f0.allFeatures, 2) == size(fN6.allFeatures, 2) && size(f6.allFeatures, 2), ...
    'Feature columns do not match.');

assert(length(f0.allLabels) == size(f0.allFeatures,1), 'Mismatch in 0dB labels vs features');
assert(length(fN6.allLabels) == size(fN6.allFeatures,1), 'Mismatch in -6dB labels vs features');
assert(length(f6.allLabels) == size(f6.allFeatures,1), 'Mismatch in 6dB labels vs features');

% === Combine ===
combinedFeatures = [f0.allFeatures; fN6.allFeatures; f6.allFeatures];
combinedLabels   = [f0.allLabels;   fN6.allLabels;   f6.allLabels];
combinedMeta     = [f0.allMeta;     fN6.allMeta;     f6.allMeta];

snrLabels = [ ...
    repmat(0,  size(f0.allFeatures, 1), 1); 
    repmat(-6, size(fN6.allFeatures, 1), 1); 
    repmat(6,  size(f6.allFeatures, 1), 1)
];

% === Final sanity check ===
assert(size(combinedFeatures,1) == length(combinedLabels));
assert(size(combinedFeatures,1) == length(snrLabels));
assert(size(combinedFeatures,1) == length(combinedMeta));

% === Save to new file ===
save('D:\Users\S11202884\Desktop\MIMII\features_allSNR.mat', ...
     'combinedFeatures', 'combinedLabels', 'combinedMeta', 'snrLabels');

disp(" features_allSNR.mat created successfully!");

% === Load Combined Feature Set ===
load('D:\Users\S11202884\Desktop\MIMII\features_allSNR.mat');

% === Normalize Features ===
featuresNorm = normalize(combinedFeatures);

% === Apply PCA ===
[coeff, score, ~, ~, explained] = pca(featuresNorm);

% === Display Variance Explained ===
figure;
plot(cumsum(explained), '-o');
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('PCA Variance Explained (All SNR Levels)');
grid on;

% === Plot 1: PCA (Color by Label: Normal vs Abnormal) ===
figure;
gscatter(score(:,1), score(:,2), combinedLabels, 'br', 'ox');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PCA Projection: Normal vs Abnormal');
legend({'Normal','Abnormal'});
grid on;

% === Plot 2: PCA (Color by SNR) ===
figure;
snrColorMap = containers.Map([-6, 0, 6], {'g', 'b', 'r'});  % assign colors

% Get color for each point
colorCodes = arrayfun(@(x) snrColorMap(x), snrLabels, 'UniformOutput', false);
gscatter(score(:,1), score(:,2), snrLabels, ['g','b','r'], 'o+x');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PCA Projection: Colored by SNR Level');
legend({'-6 dB','0 dB','6 dB'});
grid on;


3D Feature Scatter Plot (Color by Label)
figure;
scatter3(combinedFeatures(:,1), combinedFeatures(:,2), combinedFeatures(:,3), 30, combinedLabels, 'filled');
xlabel('RMS'); ylabel('ZCR'); zlabel('Spectral Centroid');
title('3D Feature Space - Normal vs Abnormal');
grid on; view(30,30); colormap([0.2 0.6 1; 1 0.2 0.2]);  % blue = normal, red = abnormal


Boxplots for Feature Distributions (by Label)
figure;
subplot(3,1,1);
boxplot(combinedFeatures(:,1), combinedLabels); title('RMS by Label'); ylabel('RMS');

subplot(3,1,2);
boxplot(combinedFeatures(:,2), combinedLabels); title('ZCR by Label'); ylabel('ZCR');

subplot(3,1,3);
boxplot(combinedFeatures(:,3), combinedLabels); title('Centroid by Label'); ylabel('Centroid');

Histograms of Each Feature
figure;
subplot(3,1,1);
histogram(combinedFeatures(:,1), 30); title('Histogram of RMS');

subplot(3,1,2);
histogram(combinedFeatures(:,2), 30); title('Histogram of ZCR');

subplot(3,1,3);
histogram(combinedFeatures(:,3), 30); title('Histogram of Centroid');

Correlation Matrix Heatmap
corrMatrix = corr(combinedFeatures);
figure;
heatmap(corrMatrix, 'XData', {'RMS','ZCR','Centroid'}, ...
                    'YData', {'RMS','ZCR','Centroid'}, ...
                    'Title', 'Feature Correlation Matrix');

Class-wise Mean Feature Comparison
meanNormal = mean(combinedFeatures(combinedLabels == 0, :));
meanAbnormal = mean(combinedFeatures(combinedLabels == 1, :));

figure;
bar([meanNormal; meanAbnormal]');
set(gca, 'xticklabel', {'RMS','ZCR','Centroid'});
legend({'Normal','Abnormal'});
title('Mean Feature Values by Class');
ylabel('Mean Value');
