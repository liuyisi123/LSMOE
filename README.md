# LSM²oE: A Dual-Focus Cloud-Edge Framework for Multi-Task Hemodynamic Parameter Estimation

This repository contains the implementation of a **Lightweight Sparse Multi-gate Mixture-of-Experts (LSM²oE)** network for cross-scale analysis of multiple hemodynamic parameters using photoplethysmography (PPG) signals.

---

## 🌐 IoMT Framework for Hemodynamic Parameter Estimation

**The working mechanism of the personalized hemodynamic parameter estimation framework:**

- **🎯 Acquisition of High-Fidelity PPG Signals:** Ensures precise data collection for accurate analysis.
- **🔄 Integration of Cloud-Edge Collaboration:** Enables real-time, accurate hemodynamic parameter estimation.
- **🌍 Data Acquisition in Diverse Clinical Scenarios:** Supports a wide range of healthcare environments.
- **🩺 Deployment of CNAP Devices:** Continuous non-invasive monitoring of hemodynamic parameters.
- **⌚ Utilization of E4 Wearable Devices:** Advanced physiological data collection for personalized health monitoring.
- **📊 Comparative Analysis:** Evaluates the proposed algorithm against state-of-the-art models for blood pressure estimation.

---
![Framework Overview](https://github.com/liuyisi123/LSMOE/blob/main/Figure/Fig.1.jpg/)  
## 🏗 Framework Overview

This study introduces an **IoMT-enabled framework** for advanced hemodynamic parameter estimation, featuring:

- **💻 Edge Computing on a Linux-Based Embedded Development Board:** Real-time data processing for seamless performance.
- **🔧 Multitasking Framework with Multidimensional Compression Strategies:** Efficient PPG signal acquisition and processing.
- **🖥 Server Platform for Simultaneous Access:** Supports multiple IoMT devices, enabling real-time diagnostic reports and feedback to healthcare providers.

---
![Framework Overview](https://github.com/liuyisi123/LSMOE/blob/main/Figure/Fig.2.jpg)  
## 📋 Features

- **Multi-Scale Fusion Architecture:** Integrates data from multiple perspectives using MSFGM.
- **Hybrid Attention Mechanism:** Combines channel and spatial attention for enhanced feature extraction.
- **Uncertainty Regression Loss (UR-Loss):** Handles heterogeneous hemodynamic parameters of different scales.
- **Structured Pruning & Bidirectional Knowledge Distillation:** Reduces computational requirements while maintaining accuracy.
- **Cloud-Edge Collaboration:** Enables deployment across various IoMT devices.

---

## 📊 Model Architecture

The **LSM²oE** model consists of several key components:

### Sparse Multi-gate Mixture-of-Experts (SM²oE)
- Multiple expert networks specialized for different aspects of the input signal.
- Noisy gating mechanism with Top-K expert selection for efficient computation.
- Load balancing to prevent over-reliance on specific experts.

### Multi-Scale Fusion Gated Module (MSFGM)
- Processes input signal at multiple scales.
- Employs gating mechanisms to control information flow.
- Enhances feature extraction from PPG signals.

### Hybrid Attention Module
- Combines channel and spatial attention.
- Groups features for more effective attention computation.
- Enhances discriminative capabilities of learned features.

### Uncertainty Regression Loss (UR-Loss)
- Learns to weight tasks based on homoscedastic uncertainty.
- Automatically balances loss contributions from different tasks.
- Improves overall performance in multi-task regression.

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/liuyisi123/LSMOE.git
cd LSMOE

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training

```bash
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
```

### Model Compression

```bash
python lightweight.py \
  --ppg_file path/to/ppg.npy \
  --labels_file path/to/cnap.npy \
  --num_experts 16 \
  --hidden_channels 128 \
  --noisy_gating \
  --noise_type uniform \
  --k 3 \
  --use_msfgm \
  --target_sparsity 0.8 \
  --teacher_weights output/weights/best_model.pt \
  --output_dir ./compressed
```

### Evaluation

```bash
python eval.py \
  --ppg_file path/to/ppg.npy \
  --labels_file path/to/cnap.npy \
  --model_weights output/weights/best_model.pt \
  --output_dir ./evaluation
```

---

## 📁 Project Structure

```
LSM2oE-Hemodynamic/
│
├── models/
│   ├── moe.py                  # Main MoE implementation
│   ├── attention.py            # Hybrid attention module
│   ├── msfgm.py                # Multi-scale fusion gated module
│   ├── resnet.py               # ResNet expert backbone
│   └── losses.py               # UR-Loss implementation
│
├── utils/
│   ├── dataset.py              # Dataset handling
│   ├── metrics.py              # Evaluation metrics
│   ├── early_stopping.py       # Early stopping mechanism
│   └── pruning.py              # Model pruning utilities
│
├── train.py                    # Training script
├── eval.py                     # Evaluation script
├── lightweight.py              # Model compression utilities
└── requirements.txt            # Dependencies
```

---

## 🎥 Supplementary Videos

To provide a clearer understanding of the proposed framework and its practical implementation, the following supplementary videos are available:

- **🎬 Appendix I: Edge Estimation Process**  
  A video demonstrating the edge estimation process is available in `Appendix I`.

- **🎬 Appendix II: LAN Solution Workflow**  
  In addition to the edge computing solution, a LAN solution for multi-device access is also provided. The workflow of this solution is illustrated in `Appendix II`.

---

## 📈 Results

**Table V. Performance of the SM2oE algorithm on two external test sets.**

|        |        | MIMIC         | VitalDB        |
|--------|--------|---------------|----------------|
| SBP    | MAE    | 5.06±0.04     | 4.76±0.03      |
|        | ME     | 0.27±0.21     | -0.06±0.13     |
|        | SD     | 7.42±0.04     | 6.85±0.07      |
|        | R2     | 0.89±0.01     | 0.86±0.01      |
|        | CPE 5  | 63.97±0.59    | 66.15±0.18     |
|        | CPE 10 | 87.37±0.20    | 88.70±0.21     |
|        | CPE 15 | 95.09±0.10    | 95.73±0.07     |
| MAP    | MAE    | 3.98±0.02     | 3.26±0.03      |
|        | ME     | 0.15±0.08     | 0.01±0.03      |
|        | SD     | 4.63±0.02     | 4.71±0.07      |
|        | R2     | 0.89±0.01     | 0.87±0.01      |
|        | CPE 5  | 82.10±0.24    | 80.00±0.46     |
|        | CPE 10 | 96.30±0.09    | 95.39±0.23     |
|        | CPE 15 | 98.86±0.05    | 98.63±0.11     |
| DBP    | MAE    | 2.81±0.01     | 2.95±0.02      |
|        | ME     | 0.09±0.01     | 0.04±0.07      |
|        | SD     | 4.42±0.01     | 4.29±0.06      |
|        | R2     | 0.88±0.00     | 0.87±0.01      |
|        | CPE 5  | 85.07±0.20    | 83.10±0.29     |
|        | CPE 10 | 96.88±0.10    | 96.32±0.15     |
|        | CPE 15 | 98.93±0.03    | 99.07±0.08     |

**Table IV. The performance of the proposed multitasking framework for validation on a cross-database (CNAP dataset 2).**

|      | MAE         | RMSE        | MAPE        | SD          | R2         |
|------|-------------|-------------|-------------|-------------|------------|
| SBP  | 5.90±0.14   | 7.68±0.31   | 5.08±0.11   | 7.67±0.31   | 0.60±0.06  |
| MAP  | 3.97±0.13   | 5.2±0.24    | 4.56±0.14   | 5.18±0.23   | 0.57±0.08  |
| DBP  | 3.69±0.10   | 4.85±0.18   | 5.07±0.15   | 4.83±0.18   | 0.62±0.04  |
| HR   | 3.43±0.30   | 4.41±0.41   | 4.23±0.38   | 4.35±0.44   | 0.30±0.19  |
| CO   | 0.34±0.09   | 0.42±0.11   | 5.32±1.64   | 0.37±0.06   | 0.73±0.05  |
| CI   | 0.19±0.05   | 0.25±0.05   | 5.26±1.60   | 0.23±0.03   | 0.79±0.03  |
| SV   | 3.10±0.43   | 4.01±0.53   | 3.71±0.50   | 3.99±0.53   | 0.70±0.09  |
| SI   | 1.81±0.25   | 2.33±0.29   | 4.00±0.56   | 2.28±0.30   | 0.80±0.04  |
| SVR  | 58.17±2.66  | 76.59±4.21  | 6.12±0.23   | 76.46±4.34  | 0.64±0.03  |
| SVRI | 110.75±6.55 | 149.34±9.18 | 6.40±0.31   | 149.08±9.39 | 0.76±0.03  |

---

## 🏆 Advantages

- **Clinical Accuracy:** Meets IEEE/BHS/AAMI standards for blood pressure estimation.
- **Computational Efficiency:** Significant reduction in model size and computational requirements.
- **Real-time Processing:** Fast inference times suitable for IoMT applications.
- **Comprehensive Monitoring:** Simultaneous estimation of multiple hemodynamic parameters.
- **Deployment Flexibility:** Cloud-edge collaboration enables use across various device types.

---

## 📄 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{liu2025dualfocus,
  title={A Dual-Focus Cloud-Edge Collaborative Framework in Multi-Task Hemodynamic Parameter Cross-Scale Analysis: The Equilibrium of Clinical Performance and Efficiency},
  author={Liu, Jian and Hu, Shuaicong and Wang, Yanan and Xiang, Wei and Yang, Cuiwei},
  journal={IEEE Journal of Internet of Things},
  year={2025},
  publisher={IEEE}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

We would like to thank all participants who contributed to the data collection process. This research was conducted in accordance with the Declaration of Helsinki, with approval from the Ethics Committee of Xinghua People's Hospital, Jiangsu Province, China.
