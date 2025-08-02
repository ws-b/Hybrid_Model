# Hybrid Machine Learning Model for Vehicle Energy Consumption Prediction

## Description

This project implements and evaluates a hybrid machine learning model for predicting vehicle energy consumption. The model combines a physics-based approach with machine learning to improve prediction accuracy. The project is designed to be a research tool for comparing the performance of different machine learning models and to study the effect of data size on model performance.

## Features

- **Hybrid Modeling**: Combines a physics-based model with machine learning models (XGBoost, RandomForest, MLP, Transformer, LinearRegression) to predict vehicle energy consumption.
- **Performance Evaluation**: Compares the performance of "Hybrid" models against "ML-only" models.
- **Hyperparameter Tuning**: Uses Optuna for hyperparameter tuning of the machine learning models.
- **Data Sampling**: a comprehensive analysis of the effect of data size on model performance.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- The required Python packages are listed in the `requirements.txt` file.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ws-b/Hybrid_Model.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Configure the experiment**:
   - Open the `config.py` file to set the vehicle, models, and other parameters.
2. **Run the experiment**:
   - Run the `main.py` script to start the experiment.
   ```bash
   python main.py
   ```
3. **Process the results**:
   - Run the `process_results.py` script to process the results.
   ```bash
   python process_results.py
   ```

## Project Structure

```
NEW_Hybrid_Model/
├── src/
│   ├── models/
│   │   ├── base_model.py
│   │   ├── linear_regression_model.py
│   │   ├── mlp_model.py
│   │   ├── random_forest_model.py
│   │   ├── transformer_model.py
│   │   └── xgboost_model.py
│   ├── data_loader.py
│   ├── experiment_manager.py
│   └── utils.py
├── results/
│   ├── optuna/
│   ├── logs/
│   ├── trained_models/
│   └── hyperparameters/
├── requirements.txt
├── config.py
├── main.py
└── process_results.py
```

## Models

The project implements the following machine learning models:

- **Linear Regression**: A simple linear regression model.
- **MLP**: A multi-layer perceptron (neural network) model.
- **Random Forest**: A random forest model.
- **Transformer**: A transformer-based model.
- **XGBoost**: An XGBoost model.

All models are implemented in the `src/models` directory.

---

# 하이브리드 머신러닝 모델을 이용한 차량 에너지 소비 예측

## 설명

이 프로젝트는 차량 에너지 소비 예측을 위한 하이브리드 머신러닝 모델을 구현하고 평가합니다. 이 모델은 물리 기반 접근 방식과 머신러닝을 결합하여 예측 정확도를 향상시킵니다. 이 프로젝트는 다양한 머신러닝 모델의 성능을 비교하고 데이터 크기가 모델 성능에 미치는 영향을 연구하기 위한 연구 도구로 설계되었습니다.

## 특징

- **하이브리드 모델링**: 물리 기반 모델과 머신러닝 모델(XGBoost, RandomForest, MLP, Transformer, LinearRegression)을 결합하여 차량 에너지 소비를 예측합니다.
- **성능 평가**: "하이브리드" 모델과 "ML 전용" 모델의 성능을 비교합니다.
- **하이퍼파라미터 튜닝**: Optuna를 사용하여 머신러닝 모델의 하이퍼파라미터를 튜닝합니다.
- **데이터 샘플링**: 데이터 크기가 모델 성능에 미치는 영향에 대한 포괄적인 분석.

## 시작하기

### 요구 사항

- Python 3.8 이상
- `requirements.txt` 파일에 명시된 필수 Python 패키지.

### 설치

1.  저장소 복제:
    ```bash
    git clone https://github.com/ws-b/Hybrid_Model.git
    ```
2.  필수 패키지 설치:
    ```bash
    pip install -r requirements.txt
    ```

## 사용법

1.  **실험 구성**:
    -   `config.py` 파일을 열어 차량, 모델 및 기타 매개변수를 설정합니다.
2.  **실험 실행**:
    -   `main.py` 스크립트를 실행하여 실험을 시작합니다.
        ```bash
        python main.py
        ```
3.  **결과 처리**:
    -   `process_results.py` 스크립트를 실행하여 결과를 처리합니다.
        ```bash
        python process_results.py
        ```

## 프로젝트 구조

```
NEW_Hybrid_Model/
├── src/
│   ├── models/
│   │   ├── base_model.py
│   │   ├── linear_regression_model.py
│   │   ├── mlp_model.py
│   │   ├── random_forest_model.py
│   │   ├── transformer_model.py
│   │   └── xgboost_model.py
│   ├── data_loader.py
│   ├── experiment_manager.py
│   └── utils.py
├── results/
│   ├── optuna/
│   ├── logs/
│   ├── trained_models/
│   └── hyperparameters/
├── requirements.txt
├── config.py
├── main.py
└── process_results.py
```

## 모델

이 프로젝트는 다음 머신러닝 모델을 구현합니다.

-   **선형 회귀**: 간단한 선형 회귀 모델.
-   **MLP**: 다층 퍼셉트론(신경망) 모델.
-   **랜덤 포레스트**: 랜덤 포레스트 모델.
-   **트랜스포머**: 트랜스포머 기반 모델.
-   **XGBoost**: XGBoost 모델.

모든 모델은 `src/models` 디렉토리에 구현되어 있습니다.