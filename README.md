# Financial Fraud Detection Using Machine Learning

## 1. Project Overview

Financial Fraud Detection Using Machine Learning is an end-to-end Data Science and Machine Learning project designed to identify potentially fraudulent financial transactions.

The project uses transaction-related and account-related features to classify transactions into legitimate and potentially fraudulent categories. It includes data validation, automated profiling, exploratory data analysis, feature engineering, machine learning model development, model evaluation, and a Streamlit prediction application.

## 2. Project Objectives

- Analyze financial transaction data.
- Identify patterns associated with fraudulent transactions.
- Perform data validation and data cleaning.
- Generate an automated dataset profiling report using YData Profiling.
- Perform exploratory data analysis and statistical analysis.
- Apply feature engineering and feature selection.
- Investigate and reduce potential data leakage.
- Handle class imbalance using SMOTENC.
- Develop and compare multiple machine learning models.
- Evaluate models using classification performance metrics.
- Deploy a fraud prediction prototype using Streamlit.

## 3. Dataset Information

- **Total Records:** 39,221
- **Original Features:** 8
- **Target Column:** `label`

### Target Variable

| Value | Description                        |
| ----- | ---------------------------------- |
| 0     | Legitimate Transaction             |
| 1     | Potentially Fraudulent Transaction |

### Dataset Features

| Feature              | Description                    |
| -------------------- | ------------------------------ |
| accountAgeDays       | Account age in days            |
| numItems             | Number of items                |
| localTime            | Transaction time-related value |
| paymentMethod        | Payment method                 |
| paymentMethodAgeDays | Payment method age             |
| isWeekend            | Weekend indicator              |
| Category             | Transaction category           |
| label                | Target variable                |

## 4. Advanced Data Profiling Using YData Profiling

An automated dataset profiling report was generated using **YData Profiling (formerly Pandas Profiling)**.

This report provides an overview of dataset quality, statistical distributions, missing values, duplicate records, correlations, and feature characteristics.

### Profiling Analysis Includes

- Dataset overview and structure.
- Variable-wise statistical summaries.
- Missing value analysis.
- Duplicate record analysis.
- Numeric and categorical feature analysis.
- Correlation analysis.
- Feature distribution analysis.
- Dataset quality observations.

**HTML Report:** `Pandas Profiling Html File/Financial_Fraud_Detection_Data_Profiling_Report.html`

The generated HTML report is included in the project to demonstrate automated data profiling and advanced exploratory data analysis skills.

## 5. Data Processing and Analysis

The project follows a structured data science workflow:

1. Dataset Understanding
2. Data Validation
3. Data Cleaning
4. Automated Data Profiling
5. Exploratory Data Analysis (EDA)
6. Univariate Analysis
7. Bivariate Analysis
8. Multivariate Analysis
9. Statistical Analysis
10. Target Variable Analysis
11. Feature Engineering
12. Feature Selection
13. Data Leakage Investigation
14. Train-Test Split
15. Data Processing and Encoding
16. Class Imbalance Handling

## 6. Machine Learning Algorithms

The following classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Random Forest with SMOTENC
5. XGBoost Classifier

A baseline model was also developed for performance comparison.

## 7. Data Processing Techniques

- Missing Value Handling
- Duplicate Record Analysis
- Data Validation
- Automated Data Profiling
- Exploratory Data Analysis
- Statistical Analysis
- Feature Engineering
- Feature Selection
- Data Leakage Investigation
- Standard Scaling
- One-Hot Encoding
- SMOTENC Oversampling

## 8. Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

### Final Test Results

The following results were obtained by Logistic Regression on the final test dataset:

| Metric                   | Result |
| ------------------------ | -----: |
| Accuracy                 | 97.18% |
| Precision                | 14.29% |
| Recall                   | 19.47% |
| F1-Score                 | 16.48% |
| ROC-AUC                  | 84.19% |
| PR-AUC                   |  9.92% |
| Classification Threshold |   0.88 |

Logistic Regression achieved the highest F1-Score, ROC-AUC, and PR-AUC among the five evaluated models on the final test dataset.

The relatively low recall indicates that further improvements are needed to detect a larger proportion of fraudulent transactions.

## 9. Streamlit Application

The project includes an interactive Streamlit application that:

- Accepts transaction details from users.
- Generates fraud probability.
- Applies the saved classification threshold.
- Displays the prediction result.

The application uses the saved machine learning model for prediction.

## 10. Project Structure

```text
Financial_Fraud_Detection/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── DS DATA.csv
│
├── notebooks/
│   └── Financial_Fraud_Detection.ipynb
│
├── reports/
│   └── DS_DATA_Report.html
│
├── images/
│   ├── EDA plots
│   ├── Confusion Matrix
│   ├── ROC Curve
│   └── Precision-Recall Curve
│
├── saved_models/
│   ├── fraud_detection_model.pkl
│   └── fraud_threshold.pkl
│
└── presentation/
    └── Financial_Fraud_Detection_Final_Presentation.pptx
```

## 11. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Directory

```bash
cd Financial_Fraud_Detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment (Windows)

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit Application

```bash
streamlit run app.py
```

## 12. Limitations

- The dataset contains a highly imbalanced target distribution.
- The final model has relatively low fraud detection recall.
- Independent external validation is required.
- Predictions are experimental and should not be treated as confirmed fraud decisions.
- Further optimization is needed before any real-world financial deployment.

## 13. Future Scope

- Real-time transaction monitoring.
- Improved fraud detection recall.
- Cost-sensitive learning.
- Advanced anomaly detection techniques.
- Model monitoring and drift detection.
- Secure API integration.
- Human verification and alert mechanisms.

## 14. Technologies Used

**Programming Language:** Python

**Data Analysis:** Pandas, NumPy

**Automated Data Profiling:** YData Profiling

**Data Visualization:** Matplotlib, Seaborn

**Machine Learning:** Scikit-learn, XGBoost

**Imbalanced Data Handling:** Imbalanced-learn, SMOTENC

**Model Evaluation:** Scikit-learn Metrics

**Model Persistence:** Joblib

**Application Development:** Streamlit

**Development Environment:** Jupyter Notebook, VS Code

## 15. Project Outcome

An end-to-end machine learning-based financial fraud detection prototype was developed with:

- Automated dataset profiling using YData Profiling.
- Comprehensive data validation and exploratory data analysis.
- Feature engineering and data leakage investigation.
- Five evaluated classification models.
- Final model evaluation using multiple performance metrics.
- Saved machine learning model and classification threshold.
- Interactive Streamlit prediction application.

This project demonstrates practical skills in Data Science, automated data profiling, exploratory data analysis, machine learning, model evaluation, and application development.

---

**Author**: Sagar Wagh
**Aspiring Data Scientist | Machine Learning Enthusiast**

**Project Domain:** Data Science | Machine Learning | Financial Fraud Detection
