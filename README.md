# Speech Emotion Analyzer

<img width="1134" height="1562" alt="Screenshot 2026-06-05 031917" src="https://github.com/user-attachments/assets/8029c91c-67d6-4f00-af56-b3e5c6789116" />

An advanced, end-to-end Speech Emotion Recognition (SER) system that leverages state-of-the-art Transformer-based deep learning architectures. By combining and standardizing four major emotional speech corpora, this analyzer effectively bridges human vocal physics and deep learning to classify spoken expressions into distinct emotional categories with an outstanding **94.04% accuracy**.

---

## Project Overview & Motivation

In spoken language, affective intent is deeply embedded within non-verbal, paralinguistic cues—such as pitch contours, vocal intensity, tone, and speech rhythm—which frequently reveal more about a speaker's true mental state than semantic text alone.

This project transitions SER from historical, hand-crafted machine learning frameworks into an automated, high-performing deep learning system capable of handling complex acoustic structures.

### Real-World Applications

* **Healthcare:** Detecting early vocal biomarkers for mental health conditions like depression or anxiety (e.g., observing compressed pitch variability).

* **Customer Service:** Powering live call center analytics to track agent-customer sentiment dynamically and flag escalating friction.

* **EdTech:** Driving adaptive learning platforms that gauge student engagement from voice responses and recalibrate task difficulties seamlessly.



---

## System Methodology & Pipeline

The system is engineered as a structured processing pipeline designed to handle raw acoustic signals seamlessly.

### 1. Data Acquisition

The pipeline ingests raw speech signals captured either via real-time microphone arrays or fetched directly from uncompressed audio data files.

### 2. Signal Pre-Processing

To guarantee consistent input tensors, raw waveforms undergo targeted audio engineering:

* Background noise suppression and silent interval trimming.


* Amplitude normalization.


* Standardization of sampling rates across different source formats.



### 3. Feature Extraction

While features like prosodic markers (pitch, energy, duration, zero-crossing rate) capture baseline arousal , this system fundamentally optimizes for **Mel-Frequency Cepstral Coefficients (MFCCs)**.
<img width="1400" height="458" alt="image" src="https://github.com/user-attachments/assets/ed60ea0d-9b88-43eb-b91a-b9e407b51adb" />
> 
> **Why MFCCs?** Alternative features like Log Mel-spectrograms or chromagrams preserve excess high-frequency noise or capture musical/harmonic structures irrelevant to human voice tracts, impeding model convergence. MFCCs closely map to the human ear's non-linear Mel scale, prioritizing lower-frequency vocal zones where the richest emotional information resides, while naturally compressing feature dimensionality.
> 
> 

### 4. Emotion Classification

The optimized MFCC feature vectors are directed into a deep sequence model. Moving past standard recurrent networks (RNNs/LSTMs), this model implements a **Transformer-based architecture** to capture long-range temporal interactions and complex acoustic correlations across utterance frames simultaneously.

### 5. Output and Analysis

The model computes a full probability distribution over the target emotional profiles, outputting a precise classification label alongside a definitive confidence score.

---

## Datasets & Target Classes

To eliminate geographical and speaker biases, the system was trained on a robust, integrated master corpus combining four premier datasets:

1. **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) 
2. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) 
3. **SAVEE** (Surrey Audio-Visual Expressed Emotion) 
4. **TESS** (Toronto Emotional Speech Set) 

The system classifies audio segments into **7 distinct emotional categories**:

* 🔴 Anger 
* 🟢 Disgust 
* 🟣 Fear 
* 🟡 Happiness 
* 🔵 Sadness 
* 🟠 Surprise 
* ⚪ Neutral



---

## Performance & Evaluation Results
Through iterative refinement, the final model achieved a test accuracy of **94.04%**, ascending drastically from the proof-of-concept demo phase (49.04%).
### Key Drivers of Performance
1. **Architectural Refinement:** Scaled up the depth and width of the Transformer multi-head attention layers to handle multi-corpus dataset complexities.
2. **Acoustic Augmentation:** Utilized active pitch shifting and time stretching to build generalization against unseen speakers.
3. **Feature Locking:** Standardizing on clean MFCC representations instead of raw spectrograms drastically boosted validation metrics.



### Final Per-Class Metrics

| Emotion Class | Precision (%) | Recall (%) | F1-Score (%) |
| --- | --- | --- | --- |
| **Angry** | 94.92 | 92.08 | 93.48 |
| **Disgust** | 97.22 | 98.59 | 97.90 |
| **Fear** | 98.58 | 97.47 | 98.02 |
| **Happy** | 97.79 | 98.88 | 98.33 |
| **Neutral** | 99.21 | 100.00 | 99.60 |
| **Sad** | 99.52 | 99.20 | 99.36 |
| **Surprise** | 100.00 | 100.00 | 100.00 |

(Source data collected from Final Model Evaluation metrics )

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/speech-emotion-analyzer.git
cd speech-emotion-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Demo video

https://github.com/user-attachments/assets/38aaf1c9-6970-45ec-aa06-fb88b05b9af7


