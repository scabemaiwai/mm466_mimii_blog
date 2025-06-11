**LATEST UPDATE**
**Sound-Based Anomaly Detection Using the MIMII Dataset**

This project focuses on developing a machine learning pipeline for acoustic-based fault detection in industrial machines using the MIMII dataset. The workflow includes data preprocessing, feature extraction, dimensionality reduction, visualization, and classification. This document outlines all completed work, along with explanations and justifications, to serve as a technical summary and GitHub blog reference.


**Tackling Class Imbalance: Manual Oversampling**

In real-world industrial datasets like MIMII, it’s common to face class imbalance—where the number of normal samples vastly outnumbers the abnormal (faulty) ones. This imbalance can bias models to favor the majority class, resulting in poor detection of actual faults.
To counter this, we applied manual oversampling, where we replicated abnormal class samples in our training data to create a more balanced class distribution. Instead of relying on automated SMOTE algorithms, we chose a straightforward and transparent method: simply duplicating the minority class until it matched the number of normal samples. This helped our model give equal attention to both classes during learning.


**Model Training in MATLAB**
With a balanced training dataset ready, we moved on to training multiple classification models using MATLAB’s Classification Learner App—a powerful GUI-based tool that simplifies model experimentation without writing extensive code.Training Setup
Before training, we split our preprocessed dataset into:
•	70% Training
•	15% Validation
•	15% Testing

This stratified split ensured both normal and abnormal classes were fairly represented in all subsets.
We enabled Principal Component Analysis (PCA) to reduce dimensionality, retaining 95% of the variance. This step helped simplify the learning task by compressing features without significant information loss.
We also configured Misclassification Costs to penalize false negatives more heavily—because missing an actual fault is more critical than flagging a healthy machine.

 
**Model Choices and Comparison**
We experimented with several models:
•	Fine KNN
•	Weighted KNN
•	Medium Tree
•	SVM (with adjusted misclassification cost)
•	Ensemble Bagged Trees

**Each model was evaluated using:**
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	ROC-AUC



**FIRST BLOG - WEEKS 6-10**
 Dataset and Project Scope

The MIMII dataset contains audio recordings from four industrial machines:

Fan
Pump
Valve
Slider

Each machine has samples under normal and abnormal conditions, recorded at three Signal-to-Noise Ratio (SNR) levels:

-6 dB (high noise)
0 dB (medium noise)
6 dB (low noise)

Each machine has multiple IDs to simulate different operational units.


** What Has Been Done So Far

 1. Data Conversion: WAV to MAT Format**

Action:

* All `.wav` files were converted to `.mat` format in batches of 500 samples.
* Separate folders were created for each SNR and machine type.
* `batchAudio`, `batchLabels`, and `batchMeta` were stored in each `.mat` file.

Justification:

* Efficiency: MATLAB reads `.mat` files significantly faster than `.wav` files.
* Scalability: Batch conversion reduces memory usage and simplifies downstream processing.
* Consistency: Having unified data structures allows easy looping through files for feature extraction and analysis.


 **2. Audio Preprocessing
**
Action:

* Stereo audio files were converted to mono by averaging channels.
* Basic checks ensured that each file is loadable, non-empty, and conforms to a consistent sample rate (`fs = 16000`).

Justification:

* Model Uniformity: Most signal processing and ML models expect single-channel input.
* Comparability: Normalizing input format ensures consistency across all machine types.


 
 **3. Feature Extraction (8 Features)
**
Action:
* The following eight features were extracted per audio sample:
  * RMS: Energy of signal
  * ZCR: Zero-crossing rate (temporal detail)
  * Spectral Centroid: Frequency balance
  * Spectral Bandwidth: Spread of frequencies
  * Spectral RollOff: High-frequency threshold
  * Spectral Flatness: Tonality vs. noisiness
  * Spectral Crest: Peak sharpness
  * Spectral Entropy: Signal randomness

Justification:
* Signal Insight: These features capture both **time-domain** and **frequency-domain** properties.
* Fault Sensitivity: Faulty machines often have distinct energy, harmonic, and spectral characteristics.
* Relevance: Widely used in anomaly detection literature.

 
 
** 4. Feature Dataset Assembly**

Action:
* Features from all batches and SNRs were merged into one file: `features_ALL_SNR.mat`
* Also includes:

  * `allFeatures`: \[Nx8] feature matrix
  * `allLabels`: 0 (normal), 1 (abnormal)
  * `snrLabels`: SNR value per sample
  * `allMeta`: machine, ID, condition, file path

Justification:
* Unified Analysis: Allows easy splitting/filtering by machine type, condition, or SNR.
* Compatibility: Ready to be used with PCA, classification, or clustering models.



**5. Principal Component Analysis (PCA)**

Action:
* Normalized features were passed into PCA.
* Generated:

  * `score`: projected features
  * `coeff`: feature loadings
  * `explained`: variance captured by each PC
* Saved to `pca_results.mat`

Justification:

* Dimensionality Reduction: Reduce 8D feature space to 2D/3D for visualization.
* Feature Evaluation: See which features contribute most to variance.
* Separation Testing: Evaluate whether normal and abnormal samples form separable clusters.


 
** 6. Data Visualization
**
A. PCA Scatter Plots

* 2D (PC1 vs PC2) and 3D (PC1-PC3) plots
* Colored by:
  * Normal vs Abnormal
  * SNR Levels
  * Machine Type

Justification:
* Visual Diagnosis: Highlights clustering, overlaps, and potential classification boundaries.


**B. Boxplots for All Features**

* Grouped by machine and SNR.
* Shows spread and skew of features under different noise levels.

Justification:

* tatistical Insight: Helps identify noise-sensitive or condition-sensitive features.

C. Spectrograms (Time-Frequency)

* Plotted from `batchAudio` for 0/1 labels.
* One normal + one abnormal sample per machine per SNR.

Justification:
* Human Interpretation: Helps visualize faults not easily captured by scalar features.


 
**WHAT IS TO BE DONE NEXT?**

1. Classification Model Training

   * Train classifiers (SVM, KNN, Decision Trees, etc.)
   * Cross-validate and tune hyperparameters

2. Performance Evaluation

   * Accuracy, Precision, Recall, F1-Score, Confusion Matrix

3. Final Report + Deployment Discussion

   * Interpret results
   * Discuss limitations
   * Propose future work and real-world integration ideas



File Summary

| File/Folder            | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| `mat_batches_*/`       | `.mat` batch files per SNR (slider handled separately) |
| `features_ALL_SNR.mat` | Merged feature matrix with labels and metadata         |
| `pca_results.mat`      | PCA scores, coefficients, explained variance           |
| `*.png`                | Visualizations: PCA plots, boxplots, spectrograms      |
| `README.md`            | Project summary and documentation (this file)          |



Purpose

This project demonstrates how audio signals can be leveraged to detect industrial machine anomalies using a structured machine learning pipeline. It combines practical engineering skills with signal processing and model-driven reasoning to support future applications in predictive maintenance.



MATLAB CODE

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


%%Progress two ->
isp(meta.machine)

snrFolders = {
    'D:\Users\S11202884\Desktop\MIMII\0dB\mat_batches_0dB', ...
    'D:\Users\S11202884\Desktop\MIMII\-6dB\mat_batches_-6dB', ...
    'D:\Users\S11202884\Desktop\MIMII\6dB\mat_batches_6dB'
};

allMachines = {};
allConditions = {};

for f = 1:length(snrFolders)
    batchFiles = dir(fullfile(snrFolders{f}, 'batch_*.mat'));
    for k = 1:length(batchFiles)
        data = load(fullfile(snrFolders{f}, batchFiles(k).name));
        if isfield(data, 'batchMeta')
            for i = 1:length(data.batchMeta)
                meta = data.batchMeta{i};
                if isfield(meta, 'machine')
                    allMachines{end+1} = meta.machine;
                    allConditions{end+1} = meta.condition;
                end
            end
        end
    end
end

disp("Unique machine types found:");
unique(allMachines)

disp("Unique conditions found:");
unique(allConditions)


% === SETTINGS ===
snrLevels = {'0dB', '-6dB', '6dB'};  % SNR folders
baseInputDir = 'D:\Users\S11202884\Desktop\MIMII';  % base folder containing the SNR folders
outputBaseDir = 'D:\Users\S11202884\Desktop\MIMII';  % where to save mat batches
batchSize = 500;

for s = 1:length(snrLevels)
    snr = snrLevels{s};
    fprintf('\nProcessing SNR: %s\n', snr);

    % Input and output paths
    inputDir = fullfile(baseInputDir, snr, 'slider');
    outputDir = fullfile(outputBaseDir, ['mat_batches_' snr]);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Collect all .wav files with metadata
    wavFiles = {};
    labels = [];
    metaData = [];

    machineFolders = dir(inputDir);
    machineFolders = machineFolders([machineFolders.isdir] & ~startsWith({machineFolders.name}, '.'));

    for m = 1:length(machineFolders)
        idFolder = fullfile(inputDir, machineFolders(m).name);
        for conditionType = ["normal", "abnormal"]
            conditionDir = fullfile(idFolder, conditionType);
            files = dir(fullfile(conditionDir, '*.wav'));

            for f = 1:length(files)
                wavFiles{end+1} = fullfile(conditionDir, files(f).name);
                labels(end+1) = strcmp(conditionType, 'abnormal');  % 1 = abnormal, 0 = normal

                meta.machine = 'slider';
                meta.id = machineFolders(m).name;
                meta.condition = conditionType;
                meta.filename = files(f).name;
                metaData{end+1} = meta;
            end
        end
    end

    % === Convert to .mat batches ===
    fprintf("Converting %d files to batches...\n", length(wavFiles));
    numFiles = length(wavFiles);
    batchIndex = 1;

    for i = 1:batchSize:numFiles
        endIdx = min(i+batchSize-1, numFiles);
        thisBatchSize = endIdx - i + 1;

        batchAudio = cell(thisBatchSize,1);
        batchLabels = zeros(thisBatchSize,1);
        batchMeta = cell(thisBatchSize,1);

        for j = 1:thisBatchSize
            idx = i + j - 1;
            [y, ~] = audioread(wavFiles{idx});
            if size(y,2) > 1  % Convert stereo to mono if needed
                y = mean(y,2);
            end
            batchAudio{j} = y;
            batchLabels(j) = labels(idx);
            batchMeta{j} = metaData{idx};
        end

        saveFileName = fullfile(outputDir, sprintf('batch_%03d.mat', batchIndex));
        save(saveFileName, 'batchAudio', 'batchLabels', 'batchMeta');
        fprintf("Saved: %s\n", saveFileName);
        batchIndex = batchIndex + 1;
    end
end

snrLevels = {'0dB', '-6dB', '6dB'};
basePath = 'D:\Users\S11202884\Desktop\MIMII';

fs = 16000;
windowLength = 256;
overlap = 128;
nfft = 512;

for s = 1:length(snrLevels)
    snr = snrLevels{s};
    batchFolder = fullfile(basePath, ['mat_batches_' snr]);
    batchFiles = dir(fullfile(batchFolder, 'batch_*.mat'));

    foundNormal = false;
    foundAbnormal = false;

    for k = 1:length(batchFiles)
        data = load(fullfile(batchFolder, batchFiles(k).name));
        for i = 1:length(data.batchMeta)
            meta = data.batchMeta{i};
            if contains(lower(meta.machine), 'slider')
                if strcmp(meta.condition, 'normal') && ~foundNormal
                    x = data.batchAudio{i};
                    [S, F, T, P] = spectrogram(x, windowLength, overlap, nfft, fs);
                    figure('Name', ['slider - normal - ', snr]);
                    surf(T, F, 10*log10(P), 'EdgeColor', 'none');
                    axis xy; axis tight;
                    xlabel('Time (s)'); ylabel('Frequency (Hz)');
                    title(['Spectrogram: slider - normal (' snr ')']);
                    colormap jet; colorbar; view(0, 90);
                    foundNormal = true;
                elseif strcmp(meta.condition, 'abnormal') && ~foundAbnormal
                    x = data.batchAudio{i};
                    [S, F, T, P] = spectrogram(x, windowLength, overlap, nfft, fs);
                    figure('Name', ['slider - abnormal - ', snr]);
                    surf(T, F, 10*log10(P), 'EdgeColor', 'none');
                    axis xy; axis tight;
                    xlabel('Time (s)'); ylabel('Frequency (Hz)');
                    title(['Spectrogram: slider - abnormal (' snr ')']);
                    colormap jet; colorbar; view(0, 90);
                    foundAbnormal = true;
                end
            end

            if foundNormal && foundAbnormal
                break;
            end
        end
        if foundNormal && foundAbnormal
            break;
        end
    end
end
% === SETTINGS ===
snrLevels = {'0dB', '-6dB', '6dB'};  % SNR folders
baseInputDir = 'D:\Users\S11202884\Desktop\MIMII';  % base folder containing the SNR folders
outputBaseDir = 'D:\Users\S11202884\Desktop\MIMII';  % where to save mat batches
batchSize = 500;

for s = 1:length(snrLevels)
    snr = snrLevels{s};
    fprintf('\nProcessing SNR: %s\n', snr);

    % Input and output paths
    inputDir = fullfile(baseInputDir, snr, 'slider');
    outputDir = fullfile(outputBaseDir, ['mat_batches_' snr]);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Collect all .wav files with metadata
    wavFiles = {};
    labels = [];
    metaData = [];

    machineFolders = dir(inputDir);
    machineFolders = machineFolders([machineFolders.isdir] & ~startsWith({machineFolders.name}, '.'));

    for m = 1:length(machineFolders)
        idFolder = fullfile(inputDir, machineFolders(m).name);
        for conditionType = ["normal", "abnormal"]
            conditionDir = fullfile(idFolder, conditionType);
            files = dir(fullfile(conditionDir, '*.wav'));

            for f = 1:length(files)
                wavFiles{end+1} = fullfile(conditionDir, files(f).name);
                labels(end+1) = strcmp(conditionType, 'abnormal');  % 1 = abnormal, 0 = normal

                meta.machine = 'slider';
                meta.id = machineFolders(m).name;
                meta.condition = conditionType;
                meta.filename = files(f).name;
                metaData{end+1} = meta;
            end
        end
    end

    % === Convert to .mat batches ===
    fprintf("Converting %d files to batches...\n", length(wavFiles));
    numFiles = length(wavFiles);
    batchIndex = 1;

    for i = 1:batchSize:numFiles
        endIdx = min(i+batchSize-1, numFiles);
        thisBatchSize = endIdx - i + 1;

        batchAudio = cell(thisBatchSize,1);
        batchLabels = zeros(thisBatchSize,1);
        batchMeta = cell(thisBatchSize,1);

        for j = 1:thisBatchSize
            idx = i + j - 1;
            [y, ~] = audioread(wavFiles{idx});
            if size(y,2) > 1  % Convert stereo to mono if needed
                y = mean(y,2);
            end
            batchAudio{j} = y;
            batchLabels(j) = labels(idx);
            batchMeta{j} = metaData{idx};
        end

        saveFileName = fullfile(outputDir, sprintf('batch_%03d.mat', batchIndex));
        save(saveFileName, 'batchAudio', 'batchLabels', 'batchMeta');
        fprintf("Saved: %s\n", saveFileName);
        batchIndex = batchIndex + 1;
    end
end

snrLevels = {'0dB', '-6dB', '6dB'};
basePath = 'D:\Users\S11202884\Desktop\MIMII';

fs = 16000;
windowLength = 256;
overlap = 128;
nfft = 512;

for s = 1:length(snrLevels)
    snr = snrLevels{s};
    batchFolder = fullfile(basePath, ['mat_batches_' snr]);
    batchFiles = dir(fullfile(batchFolder, 'batch_*.mat'));

    foundNormal = false;
    foundAbnormal = false;

    for k = 1:length(batchFiles)
        data = load(fullfile(batchFolder, batchFiles(k).name));
        for i = 1:length(data.batchMeta)
            meta = data.batchMeta{i};
            if contains(lower(meta.machine), 'slider')
                if strcmp(meta.condition, 'normal') && ~foundNormal
                    x = data.batchAudio{i};
                    [S, F, T, P] = spectrogram(x, windowLength, overlap, nfft, fs);
                    figure('Name', ['slider - normal - ', snr]);
                    surf(T, F, 10*log10(P), 'EdgeColor', 'none');
                    axis xy; axis tight;
                    xlabel('Time (s)'); ylabel('Frequency (Hz)');
                    title(['Spectrogram: slider - normal (' snr ')']);
                    colormap jet; colorbar; view(0, 90);
                    foundNormal = true;
                elseif strcmp(meta.condition, 'abnormal') && ~foundAbnormal
                    x = data.batchAudio{i};
                    [S, F, T, P] = spectrogram(x, windowLength, overlap, nfft, fs);
                    figure('Name', ['slider - abnormal - ', snr]);
                    surf(T, F, 10*log10(P), 'EdgeColor', 'none');
                    axis xy; axis tight;
                    xlabel('Time (s)'); ylabel('Frequency (Hz)');
                    title(['Spectrogram: slider - abnormal (' snr ')']);
                    colormap jet; colorbar; view(0, 90);
                    foundAbnormal = true;
                end
            end

            if foundNormal && foundAbnormal
                break;
            end
        end
        if foundNormal && foundAbnormal
            break;
        end
    end
end

f contains(lower(allMeta{i}.machine), 'slider')
        sliderIdx(i) = true;
    end
end

% === Extract slider PCA scores and SNR labels ===
sliderScores = score(sliderIdx, 1:3);         % PC1, PC2, PC3
sliderSNR = snrLabels(sliderIdx);            % Corresponding SNR values

% === Plot settings ===
colors = [0 0 1; 0 0.6 0; 1 0 0];             % Blue, Green, Red
snrLevels = [-6, 0, 6];
figure;
hold on;

for s = 1:length(snrLevels)
    idx = sliderSNR == snrLevels(s);
    scatter3(sliderScores(idx,1), sliderScores(idx,2), sliderScores(idx,3), ...
             40, colors(s,:), 'filled', 'DisplayName', sprintf('%ddB', snrLevels(s)));
end

xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('PCA Projection: Slider Machine Across SNR Levels');
legend;
grid on; view(45, 30);
Like the others, the -6 dB (blue) samples are most spread out and elevated in PC3, showing more variance due to noise.
The 6 dB (red) samples form a tighter, lower cluster — indicating more consistent behavior in cleaner recordings.
PCA again helps distinguish how background noise level alters the acoustic patterns for this machine type.

Boxplot for Fan over the 3 SNR's
% === Step 1: Filter fan data ===
fanIdx = false(length(allMeta), 1);
for i = 1:length(allMeta)
    if contains(lower(allMeta{i}.machine), 'fan')
        fanIdx(i) = true;
    end
end

% === Extract fan features and SNR labels ===
fanFeatures = allFeatures(fanIdx, :);
fanSNR = snrLabels(fanIdx);  % SNR: -6, 0, or 6
featureNames = {'RMS', 'ZCR', 'Centroid', 'Bandwidth', 'RollOff', ...
                'Flatness', 'Crest', 'Entropy'};

% === Step 2: Create boxplots for each feature grouped by SNR ===
figure;
for f = 1:size(fanFeatures, 2)
    subplot(4, 2, f);
    boxplot(fanFeatures(:, f), fanSNR, ...
        'Labels', {'-6dB', '0dB', '6dB'});
    title(featureNames{f});
    xlabel('SNR Level');
    ylabel('Value');
    grid on;
end

sgtitle('Fan Machine: Feature Distributions Across SNR Levels');
%The boxplots for all eight extracted features of the fan machine reveal clear patterns in how signal characteristics vary with different SNR levels. At -6 dB, features such as 
% RMS, Spectral Centroid, Bandwidth, and RollOff exhibit greater spread and higher values, indicating increased
% signal energy and variability caused by noise. ZCR and Crest also show higher variability under noise,
% reflecting the presence of more rapid waveform fluctuations and sharper transients. 
% Spectral Flatness increases with noise, a common indication of randomness in the signal.
% In contrast, at 6 dB, the features tend to stabilize and cluster more tightly, reflecting cleaner and more consistent acoustic patterns. 
% Spectral Entropy, while not as drastically affected, shows a subtle trend toward more uniform distribution as SNR improves. 
% Overall, the plots demonstrate that noise level significantly impacts acoustic features,
% and this insight will be important for model training and interpretation.

Boxplot for Pump data
% === Step 1: Filter pump data ===
pumpIdx = false(length(allMeta), 1);
for i = 1:length(allMeta)
    if contains(lower(allMeta{i}.machine), 'pump')
        pumpIdx(i) = true;
    end
end

% === Extract pump features and SNR labels ===
pumpFeatures = allFeatures(pumpIdx, :);
pumpSNR = snrLabels(pumpIdx);  % SNR: -6, 0, or 6
featureNames = {'RMS', 'ZCR', 'Centroid', 'Bandwidth', 'RollOff', ...
                'Flatness', 'Crest', 'Entropy'};

% === Step 2: Create boxplots for each feature grouped by SNR ===
figure;
for f = 1:size(pumpFeatures, 2)
    subplot(4, 2, f);
    boxplot(pumpFeatures(:, f), pumpSNR, ...
        'Labels', {'-6dB', '0dB', '6dB'});
    title(featureNames{f});
    xlabel('SNR Level');
    ylabel('Value');
    grid on;
end

sgtitle('Pump Machine: Feature Distributions Across SNR Levels');
%The boxplots for all eight features of the pump machine reveal how acoustic characteristics are affected by changes in noise levels (SNR). 
% At -6 dB, most features such as RMS, ZCR, Spectral Centroid, and Spectral Flatness show greater variability and higher median values compared to cleaner signals. 
% These patterns suggest the influence of background noise in boosting energy and increasing randomness in the signal.
% As the SNR improves to 6 dB, the distribution tightens, and feature values become more consistent and centered.
% Notably, Spectral Crest shows a significant reduction in outliers with increased SNR, while Spectral Entropy trends toward greater uniformity.
% Overall, these plots demonstrate that pump sound signals become more acoustically stable and predictable at higher SNRs, which is crucial for reliable anomaly detection.

Valve Machine: Boxplots Across SNR
% === Step 1: Filter valve data ===
valveIdx = false(length(allMeta), 1);
for i = 1:length(allMeta)
    if contains(lower(allMeta{i}.machine), 'valve')
        valveIdx(i) = true;
    end
end

% === Extract valve features and SNR labels ===
valveFeatures = allFeatures(valveIdx, :);
valveSNR = snrLabels(valveIdx);
featureNames = {'RMS', 'ZCR', 'Centroid', 'Bandwidth', 'RollOff', ...
                'Flatness', 'Crest', 'Entropy'};

% === Step 2: Boxplots by SNR ===
figure;
for f = 1:size(valveFeatures, 2)
    subplot(4, 2, f);
    boxplot(valveFeatures(:, f), valveSNR, ...
        'Labels', {'-6dB', '0dB', '6dB'});
    title(featureNames{f});
    xlabel('SNR Level');
    ylabel('Value');
    grid on;
end

sgtitle('Valve Machine: Feature Distributions Across SNR Levels');
%The boxplots for the valve machine across different SNR levels show significant changes in feature behavior influenced by noise.
% At -6 dB, features like RMS, Crest, and Flatness show increased variability and higher values, indicating the presence of more signal irregularities and energy fluctuations due to background noise. 
% As SNR improves to 6 dB, features such as Spectral Centroid, Bandwidth, and RollOff show a consistent upward trend, reflecting cleaner high-frequency content and more defined spectral boundaries. 
% Spectral Entropy also becomes more concentrated, suggesting a reduction in signal randomness at higher SNRs. 
% These trends reinforce that PCA and feature-based approaches are capable of capturing meaningful acoustic distinctions caused by changes in noise level in valve-type machinery.

Slider Machine: Boxplots Across SNR
% === Step 1: Filter slider data ===
sliderIdx = false(length(allMeta), 1);
for i = 1:length(allMeta)
    if contains(lower(allMeta{i}.machine), 'slider')
        sliderIdx(i) = true;
    end
end

% === Extract slider features and SNR labels ===
sliderFeatures = allFeatures(sliderIdx, :);
sliderSNR = snrLabels(sliderIdx);

% === Step 2: Boxplots by SNR ===
figure;
for f = 1:size(sliderFeatures, 2)
    subplot(4, 2, f);
    boxplot(sliderFeatures(:, f), sliderSNR, ...
        'Labels', {'-6dB', '0dB', '6dB'});
    title(featureNames{f});
    xlabel('SNR Level');
    ylabel('Value');
    grid on;
end

sgtitle('Slider Machine: Feature Distributions Across SNR Levels');
%For the slider machine, the boxplots reveal clear trends in how each feature responds to varying noise levels.
% At -6 dB, features such as RMS, ZCR, and Crest exhibit higher medians and wider spreads, indicating elevated signal energy and waveform instability due to ambient noise. 
% As the SNR improves to 6 dB, features like Centroid, Bandwidth, and RollOff gradually rise in value, suggesting clearer high-frequency content and a more pronounced spectral shape. 
% Spectral Flatness and Entropy also tighten in distribution, reflecting improved tonal structure and reduced randomness in cleaner recordings. 
% Overall, the boxplots demonstrate that the slider machine’s sound profile becomes significantly more stable and structured with increasing SNR — a trend consistent with the other machines.
