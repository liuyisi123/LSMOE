# LSM²oE: A Dual-Focus Cloud-Edge Framework for Multi-Task Hemodynamic Parameter Estimation

![Framework Overview](https://github.com/liuyisi123/LSMOE/blob/main/Figure/Fig.1.jpg)  

This repository contains the implementation of a Lightweight Sparse Multi-gate Mixture-of-Experts (LSM²oE) network for cross-scale analysis of multiple hemodynamic parameters using photoplethysmography (PPG) signals.

## 🌐 IoMT Framework for Hemodynamic Parameter Estimation

The working mechanism of the personalized hemodynamic parameter estimation framework is as follows:

- **🎯 Acquisition of High-Fidelity PPG Signals**: Ensures precise data collection for accurate analysis.
- **🔄 Integration of Cloud-Edge Collaboration**: Enables real-time, accurate hemodynamic parameter estimation.
- **🌍 Data Acquisition in Diverse Clinical Scenarios**: Supports a wide range of healthcare environments.
- **🩺 Deployment of CNAP Devices**: Continuous non-invasive monitoring of hemodynamic parameters.
- **⌚ Utilization of E4 Wearable Devices**: Advanced physiological data collection for personalized health monitoring.
- **📊 Comparative Analysis**: Evaluates the proposed algorithm against state-of-the-art models for blood pressure estimation.

## Framework Overview
![Framework Overview](https://github.com/liuyisi123/LSMOE/blob/main/Figure/Fig.2.jpg)  
This study introduces an IoMT-enabled framework for advanced hemodynamic parameter estimation, featuring:

- **💻 Edge Computing on a Linux-Based Embedded Development Board**: Real-time data processing for seamless performance.
- **🔧 Multitasking Framework with Multidimensional Compression Strategies**: Efficient PPG signal acquisition and processing.
- **🖥 Server Platform for Simultaneous Access**: Supports multiple IoMT devices, enabling real-time diagnostic reports and feedback to healthcare providers.

## 📋 Features

- **Multi-Scale Fusion Architecture**: Integrates data from multiple perspectives using MSFGM
- **Hybrid Attention Mechanism**: Combines channel and spatial attention for enhanced feature extraction
- **Uncertainty Regression Loss (UR-Loss)**: Handles heterogeneous hemodynamic parameters of different scales
- **Structured Pruning & Bidirectional Knowledge Distillation**: Reduces computational requirements while maintaining accuracy
- **Cloud-Edge Collaboration**: Enables deployment across various IoMT devices

## 📊 Model Architecture

### Sparse Multi-gate Mixture-of-Experts (SM²oE)
- Multiple expert networks specialized for different aspects of the input signal
- Noisy gating mechanism with Top-K expert selection for efficient computation
- Load balancing to prevent over-reliance on specific experts

### Multi-Scale Fusion Gated Module (MSFGM)
- Processes input signal at multiple scales
- Employs gating mechanisms to control information flow
- Enhances feature extraction from PPG signals

### Hybrid Attention Module
- Combines channel and spatial attention
- Groups features for more effective attention computation
- Enhances discriminative capabilities of learned features

### Uncertainty Regression Loss (UR-Loss)
- Learns to weight tasks based on homoscedastic uncertainty
- Automatically balances loss contributions from different tasks
- Improves overall performance in multi-task regression

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/liuyisi123/LSMOE.git
cd LSMOE

# Install dependencies
pip install -r requirements.txt

##🚀 Usage
Training
bash
python train.py \
--ppg_file path/to/ppg.npy \
--labels_file path/to/cnap.npy \
--num_experts 16 \
--hidden_channels 128 \
--noisy_gating \
--noise_type uniform \
--k 3 \
--use_msfgm \
--epochs 200 \
--batch_size 64 \
--lr 0.001 \
--output_dir ./output
