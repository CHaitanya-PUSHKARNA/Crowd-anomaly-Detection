# Crowd Violence Detection (Sample Implementation)

This repository contains a **sample, refactored, and lightweight implementation** of a crowd violence detection system built using deep learning on video data.

⚠️ **Important Note**  
This codebase is intentionally designed as a **fast, minimal, and modular prototype** for demonstration, testing, and deployment purposes.

The **complete and detailed implementation**, including:
- full experimentation
- dataset exploration
- extensive training loops
- research-oriented analysis

is implemented in the **original Jupyter Notebook (`.ipynb`)**, which serves as the primary development and research artifact.

---

## 🎯 Purpose of This Repository

This repository focuses on:
- Converting notebook-based research into **production-style ML code**
- Demonstrating **clean project structure**
- Enabling **fast sanity testing** using a mini dataset
- Supporting **real-time inference (CNN + LSTM)**

It is **not intended** to replace the full notebook implementation, but to **complement it**.

---

## 📂 Project Structure

crowd_violence_detection/
│
├── data/
│ ├── raw/ # Original datasets (RLVS, RWF-2000)
│ └── mini/ # 10% sampled dataset for fast testing
│
├── src/
│ ├── dataset.py # Video loading & frame sampling
│ ├── model.py # CNN + LSTM architecture
│ ├── train.py # Training (sanity-check)
│ ├── inference.py# Real-time webcam/video inference
│ └── config.py # Centralized configuration
│
├── experiments/
│ └── sanity_test.ipynb
│
├── run.py
└── README.md


---

## 🧠 Model Overview

- **CNN** for spatial feature extraction from individual frames  
- **LSTM** for temporal modeling across video sequences  
- Designed for **crowd violence / fight detection** in surveillance footage  

This architecture mirrors the logic used in the original notebook, but in a **simplified and deployable form**.

---

## ⚡ Dataset Handling

- Original datasets: **RLVS** and **RWF-2000**
- A **10% class-balanced mini dataset** is created for:
  - fast debugging
  - sanity testing
  - CI-friendly execution

Full-scale training and evaluation are performed in the notebook.

---
##Link for the Dataset is below

https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset/data

## ▶️ How to Run (Sample)

### Train (sanity test)
```bash
python run.py

Real-time inference
python -m src.inference

