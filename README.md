# Artificial Neural Networks: Medical Image Classification
> This project was developed by **Yahya Ahmet Öğütcü** under the Artificial Neural Networks curriculum, focusing on the automated detection of Pneumonia from Chest X-Ray images using various deep learning architectures.
> 
> ⚠️ **Disclaimer:** Please note that comprehensive technical details, complete dataset preprocessing pipelines, source codes, and exhaustive ablation studies have been intentionally omitted from this public repository for academic confidentiality and intellectual property reasons.
---
## Project Overview & Data Preprocessing
The primary objective of this project is to perform binary classification (Normal vs. Pneumonia) on medical X-Ray images to assist in rapid medical diagnosis. To comprehensively evaluate performance, three fundamentally different network architectures were designed and optimized: **Feed Forward Neural Network (FFNN)**, **Long Short-Term Memory (LSTM)**, and **Simple Recurrent Neural Network (RNN)**. 
Prior to training, all images were converted to grayscale, aggressively resized to `28 x 28` pixels to optimize computational load, and their pixel values were normalized to a `[0, 1]` scale. A highly efficient data pipeline was constructed using `tf.data.Dataset` featuring multi-core asynchronous prefetching (`AUTOTUNE`) and data shuffling.
---
## 1. Feed Forward Neural Network (FFNN)
The FFNN treats the entire image as a single flat array of 784 pixels. Following systematic optimization, the architecture was deepened to include three hidden layers with a decreasing number of neurons. This architecture proved highly effective in mapping direct spatial relationships across the image.
**Optimized Hyperparameters:**
* **Architecture:** `Flatten()` -> `Dense(256, ReLU)` -> `Dense(128, ReLU)` -> `Dense(64, ReLU)` -> `Dropout(0.1)` -> `Dense(2, Softmax)`
* **Optimizer:** Adam
* **Learning Rate:** 0.0005 *(Reduced to ensure stable convergence)*
* **Batch Size:** 128
* **Epochs:** 90
* **Validation Accuracy:** 95.31%
<p align="center">
  <img src="Part1_Figures/Figure1_FFNN%20RESULTS.png" alt="FFNN Results">
  <br>
  <em><b>Figure 1:</b> Training and validation accuracy/loss graphs for the optimized FFNN model over 90 epochs.</em>
</p>
---
## 2. Long Short-Term Memory (LSTM)
In this approach, the `28 x 28` spatial image data was forcefully interpreted as a time-series sequence, where the model processes the image row by row (28 timesteps, 28 features per step). Because the data lacks true temporal dependency, LSTM required significant regularization to prevent severe overfitting.
**Optimized Hyperparameters:**
* **Architecture:** `LSTM(128)` -> `LSTM(64)` -> `LSTM(32)` -> `Dense(2, Softmax)`
* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Batch Size:** 128
* **Regularization:** Early Stopping (to prevent overfitting during sequential learning)
* **Validation Accuracy:** 90.92% *(Training halted at Epoch 16/60 via Early Stopping)*
<p align="center">
  <img src="Part1_Figures/Figure2_LSTM%20RESULTS.png" alt="LSTM Results">
  <br>
  <em><b>Figure 2:</b> Training and validation metrics for the LSTM network, demonstrating early stopping activation.</em>
</p>
---
## 3. Simple Recurrent Neural Network (RNN)
Similar to the LSTM, the SimpleRNN processes the image as a sequence. To stabilize the gradients and accelerate convergence, Tanh activations and Batch Normalization were heavily utilized between the recurrent layers.
**Optimized Hyperparameters:**
* **Architecture:** `SimpleRNN(64, Tanh, BatchNorm)` -> `SimpleRNN(32, Tanh, BatchNorm)` -> `Dense(2, Softmax)`
* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Batch Size:** 128
* **Regularization:** Early Stopping (Patience = 5) + Batch Normalization
* **Validation Accuracy:** 93.49% *(Training halted at Epoch 11/60 via Early Stopping)*
<p align="center">
  <img src="Part1_Figures/Figure3_RNN%20RESULTS.png" alt="RNN Results">
  <br>
  <em><b>Figure 3:</b> Training and validation metrics for the SimpleRNN architecture.</em>
</p>
---
## 4. Conclusion & Real-World Predictions
The comparative analysis decisively proved that **FFNN** vastly outperforms sequential models (LSTM, RNN) in both computational speed and classification accuracy for static spatial datasets like medical X-Rays. Sequential models suffered from slow training times and high overfitting risks due to the forced row-by-row temporal processing of non-temporal data.
To validate its robustness, the superior FFNN model was subjected to a blind test utilizing unseen, real-world chest X-ray images. The model successfully classified **4 out of 4 (100%)** of the previously unseen samples with high statistical confidence.
<p align="center">
  <img src="Part1_Figures/Figure4_FFNN%20PREDICTIONS.png" alt="FFNN Predictions">
  <br>
  <em><b>Figure 4:</b> Real-world inference results demonstrating 100% predictive accuracy on unseen data using the optimized FFNN model.</em>
</p>
