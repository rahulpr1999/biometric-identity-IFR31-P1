# 🛡️ Project IFR31-P1
### **Marrty LLC: Secure Biometric Identity System**

[![Version](https://img.shields.io/badge/Version-V1.0_Master-blue.svg)](https://github.com/rahulpr1999/biometric-identity-IFR31-P1)
[![Phase](https://img.shields.io/badge/Phase-P1-green.svg)](https://github.com/rahulpr1999/biometric-identity-IFR31-P1)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](https://github.com/rahulpr1999/biometric-identity-IFR31-P1)

**IFR31-P1** is a professional-grade facial recognition solution developed by **Marrty LLC**. This system is designed for high-precision identity verification of 12 registered subjects, utilizing state-of-the-art deep learning architectures and geometric alignment protocols.

---

## 📖 Project Documentation

### **Technical Architecture**
* **Backbone:** ResNet-101 (Deep Residual Network) for robust feature extraction.
* **Loss Function:** ArcMarginProduct (ArcFace) for additive angular margin loss, maximizing inter-class separation.
* **Alignment Logic:** 5-point facial landmark warping (Pro-Alignment) to ensure geometric consistency across scans.
* **Data Strategy:** Weighted sampling to maintain high accuracy (98.41%) across all 12 unique identities.



### **Developer Credits**
* **Organization:** Marrty LLC
* **Co-Developed by:** Rhaul PR
* **Status:** Phase 1 Master Release (Production Ready).

---

## 🚀 Deployment Guide

### **1. GPU Cloud Instance (e.g., Ubuntu on AWS G5)**
*Best for high-speed testing, large-scale inference, and training.*
* **Prerequisites:** NVIDIA Drivers, CUDA Toolkit, and cuDNN.
* **Setup:**
    ```bash
    sudo apt update && sudo apt install python3-pip -y
    pip install -r requirements.txt
    ```
* **Run command:**
    ```bash
    python3 -m streamlit run app_vanguard_pro.py --server.port 8888
    ```
* **Performance:** Optimized for ~30-50ms latency using `CUDAExecutionProvider`.

### **2. CPU Desktop / Server (Windows or Ubuntu)**
*Best for 24/7 internal production with low hardware overhead.*
* **Prerequisites:** Python 3.10 or higher.
* **Setup:**
    ```bash
    # For Windows (PowerShell)
    pip install -r requirements.txt
    ```
* **Run command:**
    ```bash
    python -m streamlit run app_vanguard_pro.py
    ```
* **Performance:** Stable ~200-500ms latency using `CPUExecutionProvider`.

---

## 🛠️ Installation Requirements
The following libraries are required for both CPU and GPU environments:
* `torch` / `torchvision`: Core neural network framework.
* `streamlit`: Web portal interface.
* `insightface`: Face detection and landmark extraction.
* `opencv-python`: Image processing and warping.
* `onnxruntime-gpu`: Hardware-accelerated inference.

---

## ⚖️ License & Security
* **Proprietary:** This project is owned exclusively by **Marrty LLC**.
* **Usage:** Strictly restricted to **internal purposes only**. Unauthorized distribution is prohibited.
* **Asset Security:** Model weights (`IFR31_V1_MASTER.pth`) and identity mappings (`labels_v1.json`) are proprietary assets and should **never** be uploaded to public repositories.

---

## 📩 Contact & Support
For technical support or Phase 2 registration:
* **Primary Contact:** [rahul@marrty.com](mailto:rahul@marrty.com)
* **Corporate Office:** [info@marrty.com](mailto:info@marrty.com)
