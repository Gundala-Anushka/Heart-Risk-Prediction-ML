# Heart Risk Prediction using Machine Learning

A machine learning classification pipeline built in Python to predict patient heart disease risk (`heart_risk`) based on 16 clinical, lifestyle, and demographic indicators.

---

## Project Overview

Early identification of cardiovascular risk helps enable timely preventive care. This project implements and compares multiple supervised machine learning algorithms—including **Scaled Logistic Regression**, **Random Forest**, **Gradient Boosting**, and **Support Vector Machines (SVM)**—on a 1,000-record patient health dataset (`heart_risk_dataset.csv`) to build a reliable binary classification system (`0 = Low Risk`, `1 = High Risk`).

---

## Dataset & Features

The dataset (`heart_risk_dataset.csv`) contains **1,000 patient records** with **16 input features** and **1 target variable**:

| Feature Name | Description |
| :--- | :--- |
| `age` | Age of the patient (in years) |
| `sex` | Biological sex (`0 = Female`, `1 = Male`) |
| `smoking` | Smoking status (`0 = No`, `1 = Yes`) |
| `diabetes` | History of diabetes (`0 = No`, `1 = Yes`) |
| `family_history` | Family history of heart disease (`0 = No`, `1 = Yes`) |
| `physical_activity` | Regular physical activity level |
| `bmi` | Body Mass Index (kg/m²) |
| `resting_bp` | Resting blood pressure (mm Hg) |
| `cholesterol` | Serum cholesterol level (mg/dL) |
| `fasting_blood_sugar` | Fasting blood sugar indicator |
| `resting_heart_rate` | Resting heart rate (bpm) |
| `chest_pain_type` | Type of chest pain experienced |
| `max_heart_rate` | Maximum heart rate achieved |
| `exercise_angina` | Exercise-induced angina (`0 = No`, `1 = Yes`) |
| `st_depression` | ST depression induced by exercise relative to rest |
| `resting_ecg` | Resting electrocardiographic results |
| **`heart_risk`** *(Target)* | **Heart disease risk diagnosis (`0 = Low Risk`, `1 = High Risk`)** |

---

## Model Benchmarking & Results

The dataset was split into **80% training data (800 samples)** and **20% testing data (200 samples)** using stratified sampling. Four models were evaluated:

| Model | Training Accuracy | Testing Accuracy | Observation |
| :--- | :--- | :--- | :--- |
| **Scaled Logistic Regression** | **74.75%** | **73.50%** | **Best generalization; minimal overfitting** |
| Gradient Boosting Classifier | 83.75% | 69.00% | Slight overfitting on training set |
| Support Vector Machine (SVM) | 83.13% | 67.50% | Slight overfitting on training set |
| Tuned Random Forest (`max_depth=5`) | 81.50% | 66.50% | Slight overfitting on training set |

**Selected Model:** The pipeline combining `StandardScaler` and `LogisticRegression` achieved the highest unseen test accuracy (**73.50%**) with virtually zero overfitting (`74.75%` train vs. `73.50%` test) and is exported as `heart_risk_model.pkl`.

---

## Project Structure

```text
├── heart_risk_dataset.csv                 # Custom 16-feature dataset
├── heart_risk_model.pkl                   # Saved trained model pipeline
├── Logistic-Heart-Disease-Prediction.ipynb # Jupyter Notebook (EDA, training & evaluation)
├── .gitignore                             # Ignored virtual environment files
├── LICENSE                                # MIT License
└── README.md                              # Project documentation
```
---
## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Gundala-Anushka/Heart-Risk-Prediction-ML.git](https://github.com/Gundala-Anushka/Heart-Risk-Prediction-ML.git)
   cd Heart-Risk-Prediction-ML
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter joblib
   ```

4. **Launch the Jupyter Notebook:**
   ```powershell
   jupyter notebook
   ```

---

## Authors

* **Anushka Gundala** — [@Gundala-Anushka](https://github.com/Gundala-Anushka)
* **Soumya M**
* **Prajna** -[@Prajna-dash](https://github.com/prajnadash73-netizen)
* **Soumyasmita**

---

## Acknowledgments & Inspiration

* Inspired by the baseline notebook structure from [NhanPhamThanh-IT/Logistic-Heart-Disease-Prediction](https://github.com/NhanPhamThanh-IT/Logistic-Heart-Disease-Prediction). Adapted and expanded with a custom 16-feature dataset, multi-model benchmarking, feature scaling pipelines, and model serialization.
