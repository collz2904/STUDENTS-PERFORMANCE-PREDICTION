# Student Performance Prediction System

Data Science Project: Predicting Student Performance using classification models.
This project implements a comprehensive Machine Learning pipeline to identify factors influencing student academic success.

## Project Goal

To build a machine learning model that accurately predicts the final performance of students based on demographic, social, and academic features. The outcome of the project provides valuable insights into potential intervention points for students at risk.

## Dataset

* **Dataset:** Student Performance Prediction Dataset by Amr Maree.
* **Target Variable:** The final academic status (e.g., Pass/Fail or grade level) is defined as the target variable for prediction.

## Project Outline (5 Weeks)

### Week 1: Data Acquisition & Initial Review
* **Environment Setup:** Utilized **pandas** for data manipulation.
* **Data Loading:** Loaded the `student_performance_dataset.csv` file into a DataFrame.
* **Preliminary Inspection:** Used `df.head(3000)` to inspect the first 3000 rows, focusing on features like **Study_Hours_per_Week**, **Attendance_Rate**, and **Gender**.

### Week 2: Data Cleaning & Distribution Analysis
* **Integrity Checks:** Performed missing value identification with `df.isnull().sum()`.
* **Handling Nulls:** Implemented row-wise deletion for missing values using `df.dropna()`.
* **Type Validation:** Systematically checked and printed data types for both numeric and object-based columns.
* **Outlier Detection:** Leveraged **Seaborn** boxplots and histograms to visualize distributions and identify statistical outliers.

### Week 3: Advanced Feature Engineering & Selection
* **Correlation Mapping:** Generated a heatmap using `sns.heatmap` to identify linear relationships between numerical features.
* **Categorical Encoding:** Converted non-numeric data into machine-readable formats using `pd.get_dummies` with `drop_first=True`.
* **Statistical Filtering:** * Performed **Chi-Square tests** using `sklearn.feature_selection` to rank categorical features.
    * Calculated **Variance Inflation Factor (VIF)** using `statsmodels` to detect and remove multicollinearity.
* **Data Partitioning:** Split the dataset into training and testing sets using an **80/20 split** (`test_size=0.2`) with `random_state=84`.

### Week 4: Model Training
* **Algorithm Choice:** Implemented the **Random Forest Classifier** as the primary predictive engine.
* **Hyperparameter Configuration:** Configured the model with `n_estimators=2000` to ensure robust decision-making across the forest.
* **Implementation:** Fit the model on the training data (`X_train`, `y_train`) and generated predictions on the test set.

### Week 5: Evaluation & Performance Summary
* **Metric Assessment:** Evaluated the model using a **weighted average approach**.
* **Performance Metrics:** * Accuracy Score
    * F1 Score
    * Confusion Matrix
* **Final Output:** Delivered key findings and deployment recommendations based on model performance.