# Predicting energy consumption in electric vehicles: Hybridization of data-driven and physics-based models

## 📌 Overview

This repository contains the source code for the research paper **"Predicting energy consumption in electric vehicles: Hybridization of data-driven and physics-based models"**. The project implements a **Hybrid Model** that combines a physics-based energy consumption model with data-driven machine learning algorithms (XGBoost, Random Forest, Linear Regression) to enhance prediction accuracy, especially in scenarios with limited training data.

The system evaluates model performance across varying dataset sizes and compares the **Hybrid approach** against **ML-only approaches**.

## 🚀 Features

- **Hybrid Architecture**: Integrates physics-based residuals with ML predictions.
- **Multi-Model Support**: Includes XGBoost, Random Forest, and Linear Regression (MLR).
- **Automated Hyperparameter Tuning**: Utilizes **Optuna** for efficient optimization.
- **Data Efficiency Analysis**: Systematically evaluates performance across different sample sizes (from 10 trips to full dataset).
- **Reproducibility**: Includes configuration management and random seed control for consistent results.

## 📂 Project Structure

```bash
Hybrid_Model/
├── sample_trips/         # Anonymized sample trip data for demonstration
│   └── EV6/              # Vehicle-specific data
├── src/
│   ├── models/           # Model implementations (XGBoost, RF, MLR)
│   ├── data_loader.py    # Data loading and feature engineering
│   ├── experiment_manager.py # Experiment logging and artifact management
│   └── utils.py          # Utility functions (metrics, scaling)
├── results/              # Directory for experiment outputs
│   ├── optuna/           # Optuna study database
│   ├── logs/             # Detailed experiment logs (JSON)
│   ├── trained_models/   # Saved model artifacts (.joblib)
│   └── hyperparameters/  # Best hyperparameters found
├── config.py             # Global configuration file
├── main.py               # Main execution script
├── process_results.py    # Result aggregation and analysis script
└── requirements.txt      # Python dependencies
```

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- Recommended: Virtual environment (venv or conda)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ws-b/Hybrid_Model.git
   cd Hybrid_Model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration (`config.py`)

You can control the experiment behavior by modifying `config.py`:

- **`DATA_PATH`**: Path to the trip data (Default: `"sample_trips"`).
- **`SELECTED_VEHICLE`**: Vehicle identifier (Default: `"EV6"`).
- **`DEBUG_MODE`**:
    - `True`: Runs a fast verification test (small subset, 1 iteration).
    - `False`: Runs the full research experiment (comprehensive sampling, multiple iterations).
- **`MODELS_TO_RUN`**: Select which models to train (e.g., `["XGBoost", "RandomForest", "MLR"]`).

## 🏃 Usage

### 1. Run the Experiment
Execute the main script to start training and evaluation.
```bash
python main.py
```
> **Note:** The script will automatically perform hyperparameter tuning (using Optuna), run cross-validation on sampled datasets, and finally evaluate on the test set.

### 2. Analyze Results
After the experiment completes, process the logs to generate summary reports.
```bash
python process_results.py
```
Results will be saved in the `results/` directory.

---

# 전기차 에너지 소비 예측: 데이터 기반 모델과 물리 기반 모델의 하이브리드화

## 📌 개요

본 저장소는 **"전기차 에너지 소비 예측: 데이터 기반 모델과 물리 기반 모델의 하이브리드화"** 연구 논문의 소스 코드입니다. 이 프로젝트는 물리 기반(Physics-based) 모델과 데이터 기반(Data-driven) 머신러닝 알고리즘(XGBoost, Random Forest, Linear Regression)을 결합하여, 특히 학습 데이터가 부족한 환경에서도 예측 정확도를 향상시키는 **하이브리드 모델**을 구현합니다.

본 시스템은 다양한 데이터 크기에 따른 모델 성능을 체계적으로 평가하고, **하이브리드 접근 방식**과 **순수 ML 접근 방식**의 성능을 비교 분석합니다.

## 🚀 주요 기능

- **하이브리드 아키텍처**: 물리 모델의 잔차(Residual)를 ML 모델이 학습하여 보정하는 구조.
- **다양한 모델 지원**: XGBoost, Random Forest, 다중 선형 회귀(MLR) 지원.
- **자동 하이퍼파라미터 튜닝**: **Optuna**를 활용한 효율적인 파라미터 최적화.
- **데이터 효율성 분석**: 샘플 크기(Sample Size) 변화에 따른 성능 변화를 실험적으로 검증.
- **재현성(Reproducibility)**: 난수 시드(Seed) 고정 및 설정 관리를 통한 실험 결과의 일관성 보장.

## 📂 프로젝트 구조

```bash
Hybrid_Model/
├── sample_trips/         # 데모용 익명화된 주행 샘플 데이터
│   └── EV6/              # 차량별 데이터 폴더
├── src/
│   ├── models/           # 모델 구현체 (XGBoost, RF, MLR)
│   ├── data_loader.py    # 데이터 로드 및 피처 엔지니어링
│   ├── experiment_manager.py # 실험 로그 및 결과 저장 관리
│   └── utils.py          # 유틸리티 함수 (평가 지표, 스케일링 등)
├── results/              # 실험 결과 저장 디렉토리
│   ├── optuna/           # Optuna 튜닝 DB
│   ├── logs/             # 상세 실험 로그 (JSON)
│   ├── trained_models/   # 학습된 모델 아티팩트 (.joblib)
│   └── hyperparameters/  # 최적화된 하이퍼파라미터
├── config.py             # 전체 실험 설정 파일
├── main.py               # 메인 실행 스크립트
├── process_results.py    # 결과 집계 및 분석 스크립트
└── requirements.txt      # 파이썬 의존성 패키지 목록
```

## 🛠️ 시작하기

### 필수 사항
- Python 3.8 이상
- 권장: 가상 환경 (venv 또는 conda) 사용

### 설치 방법

1. **저장소 복제:**
   ```bash
   git clone https://github.com/ws-b/Hybrid_Model.git
   cd Hybrid_Model
   ```

2. **패키지 설치:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ 설정 (`config.py`)

`config.py` 파일을 수정하여 실험 환경을 제어할 수 있습니다:

- **`DATA_PATH`**: 데이터 파일 경로 (기본값: `"sample_trips"`).
- **`SELECTED_VEHICLE`**: 대상 차량명 (기본값: `"EV6"`).
- **`DEBUG_MODE`**:
    - `True`: 빠른 검증 모드 (적은 데이터, 1회 반복).
    - `False`: 전체 실험 모드 (논문용 전체 샘플링 및 반복 수행).
- **`MODELS_TO_RUN`**: 실행할 모델 선택 (예: `["XGBoost", "RandomForest", "MLR"]`).

## 🏃 사용 방법

### 1. 실험 실행
메인 스크립트를 실행하여 학습 및 평가를 시작합니다.
```bash
python main.py
```
> **참고:** 스크립트는 자동으로 하이퍼파라미터 튜닝(Optuna)을 수행하고, 다양한 샘플 크기에 대해 교차 검증을 진행한 뒤, 최종 테스트 셋에서 성능을 평가합니다.

### 2. 결과 분석
실험이 완료되면 결과 로그를 집계합니다.
```bash
python process_results.py
```
모든 결과물은 `results/` 디렉토리에 저장됩니다.

---
*This README was authored with the assistance of Gemini.*
