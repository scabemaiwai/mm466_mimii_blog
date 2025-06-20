# Acoustic Fault Detection using the MIMII Dataset - A Machine Learning Approach


## Introduction

This project tackles the classification of industrial machine conditions using audio data from the MIMII dataset. Our goal was to train a supervised machine learning model to detect whether a machine sound is normal or abnormal, and to further classify it by machine type. Eventually, this evolved into a multiclass model that identifies both the machine type and its condition.

---

## 1. Audio Conversion: `.wav` to `.mat`

### What We Did

We began with `.wav` audio recordings for different machines (fan, pump, valve, slider) under both normal and abnormal conditions. Each file also came from different machine IDs and SNR (Signal-to-Noise Ratio) levels. We created a MATLAB function `convert_wav_to_mat` that traversed through the dataset folders and converted each audio sample into MATLAB-readable `.mat` format.

Each `.mat` entry saved:

* The audio signal (converted to mono)
* A binary label (0 = normal, 1 = abnormal)
* Metadata: machine type, ID, condition, and filename

### Why We Did It

`.wav` files are heavy to process in MATLAB, especially in bulk. `.mat` files are much more efficient in terms of I/O and batch processing. It also allowed us to bundle audio signals with useful metadata for easy feature extraction later.

### How Users Can Do It

```matlab
convert_wav_to_mat('MIMII_root_folder', 'Processed_Mat_Folder', 500);
```

This processes the files in batches of 500 samples.

---

## 2. Feature Extraction

### What We Did

Using the `.mat` files, we extracted 8 audio features that represent both time-domain and frequency-domain characteristics:

* Root Mean Square (RMS)
* Zero Crossing Rate (ZCR)
* Spectral Centroid
* Spectral Bandwidth
* Spectral Roll-off
* Spectral Flatness
* Crest Factor
* Spectral Entropy

### Why These Features?

These features are standard in digital audio analysis. They capture:

* **Amplitude dynamics** (RMS, Crest Factor)
* **Frequency content** (Spectral Centroid, Bandwidth)
* **Noise characteristics** (Entropy, ZCR)

Normal machine sounds tend to be steady and harmonic, while abnormal sounds exhibit higher energy, more irregular patterns, and broader frequency components.

### How Users Can Do It

```matlab
extract_features_from_batch('Processed_Mat_Folder', 'features_output.mat');
```

This produces:

* `allFeatures`: numeric matrix (samples x 8 features)
* `allLabels`: 0 or 1
* `allMeta`: cell array of structs containing metadata
* `allSNRs`: numerical vector

---

## 3. Dataset Aggregation and Label Encoding

### What We Did

We combined all batches into a single dataset (`features_allSNR_combined.mat`) and generated a new label:

```matlab
multiLabel = machine_type + "_" + condition;
```

This gave us 8 unique classes like:

* fan\_normal
* fan\_abnormal
* pump\_normal
* slider\_abnormal

We converted this string label to categorical for multiclass training.

### Why?

The goal evolved from binary classification (normal vs abnormal) to a **multiclass setup**, which would identify both machine type and condition. This provides deeper fault localization and has more real-world value.

---

## 4. Data Cleaning and Preprocessing

### What We Did

* Skipped corrupted `.wav` files during conversion (implicitly cleaned).
* Dropped stereo audio channels (converted to mono).
* Normalized features using MATLAB's `normalize()` function.

### Why?

Clean and scaled data ensures:

* Faster model convergence
* Equal treatment of all features
* Avoidance of skew caused by high-magnitude features

### How?

```matlab
allFeatures = normalize(allFeatures);
```

---

## 5. Exploratory Data Analysis (EDA)

### What We Did

* **PCA Analysis**: Showed PC1 explained >97% variance.
* **Boxplots and Correlation Matrix**: Revealed which features were most informative.
* **Scatter plots**: Highlighted clusters of normal vs abnormal.

### Why?

EDA helped us:

* Understand class distribution
* Detect outliers
* Choose meaningful features

Picture







---

## 6. Principal Component Analysis (PCA)

### What We Did

Enabled PCA in the Classification Learner App to retain 95% explained variance.

### Why?

* To reduce feature dimensionality
* Combat noise and multicollinearity

### How?

In the Classification Learner App:

```text
Options > PCA > Enable > 95% Variance
```

---

## 7. Model Training

### What We Did

We trained multiple models using MATLAB’s **Classification Learner App**:

* Decision Trees (Coarse, Medium)
* KNN (Cosine, Fine, Weighted)
* Ensemble (Bagged Trees)
* SVM
* Neural Networks

Final best model for multiclass was **Model 2.1**, an ensemble method.

### Input: `allFeatures`

### Response: `allLabelsMulti`

### Validation: 5-fold cross-validation

### Why?

Ensemble methods are:

* Robust to noise
* Less prone to overfitting
* Good with imbalanced datasets

---

## 8. Evaluation and Confusion Matrix

### What We Did

* Plotted Confusion Matrix for 8 classes
* Observed highest misclassifications between slider and pump classes

### Why?

To identify:

* Where the model struggles (e.g., noisy or similar sounds)
* Which classes need better representation or features

---
##10. Function File Roles (Summary)

1. convert_wav_to_mat.m

This function recursively navigates the directory structure of the MIMII dataset, reads each .wav file, converts it to mono audio, and stores it in a .mat file along with essential metadata. Metadata includes the machine type (e.g., fan, pump), machine ID, operating condition (normal/abnormal), and filename. The data is saved in batches to improve processing efficiency and later used for feature extraction.

2. extract_features_from_batch.m

This function reads each batch .mat file, extracts 8 descriptive audio features per file (RMS, ZCR, Spectral Centroid, etc.), and stores the results in a structured format. It outputs allFeatures (feature matrix), allLabels (binary condition labels), allMeta (machine info), and allSNRs (signal noise ratio levels). This forms the core input for PCA and model training.

3. combine_all_batches.m

Used to merge feature outputs from multiple .mat batch files into a single dataset (features_allSNR_combined.mat). This simplifies analysis by ensuring all samples are in one array. The script concatenates feature matrices, labels, SNR values, and metadata into unified variables for downstream tasks.


## Conclusion

This project successfully demonstrated the full pipeline of an acoustic-based machine learning classifier using MATLAB:

* From audio preprocessing
* To multiclass labeling
* To model training and validation

This work now serves as a foundation to explore ensemble stacking, hybrid classifiers, or even double-stage architectures (first detect abnormal, then classify fault type).




---
**##FIRST BLOG - WEEKS 6-10**

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

**What Has Been Done So Far**

**1. Data Conversion: WAV to MAT Format**

Action:

* All `.wav` files were converted to `.mat` format in batches of 500 samples.
* Separate folders were created for each SNR and machine type.
* `batchAudio`, `batchLabels`, and `batchMeta` were stored in each `.mat` file.

Justification:
* Efficiency: MATLAB reads `.mat` files significantly faster than `.wav` files.
* Scalability: Batch conversion reduces memory usage and simplifies downstream processing.
* Consistency: Having unified data structures allows easy looping through files for feature extraction and analysis.


**2. Audio Preprocessing**

Action:

* Stereo audio files were converted to mono by averaging channels.
* Basic checks ensured that each file is loadable, non-empty, and conforms to a consistent sample rate (`fs = 16000`).

Justification:

* Model Uniformity: Most signal processing and ML models expect single-channel input.
* Comparability: Normalizing input format ensures consistency across all machine types.


 
 **3. Feature Extraction (8 Features)**
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

 
 
**4. Feature Dataset Assembly**

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


 
**6. Data Visualization**
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



---
**##LATEST UPDATE**
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


% Spectral Flatness and Entropy also tighten in distribution, reflecting improved tonal structure and reduced randomness in cleaner recordings. 
% Overall, the boxplots demonstrate that the slider machine’s sound profile becomes significantly more stable and structured with increasing SNR — a trend consistent with the other machines.
