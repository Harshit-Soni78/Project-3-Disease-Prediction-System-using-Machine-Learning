# Project-3-Disease-Prediction-System-using-Machine-Learning

<hr><hr>

# 1. (UCI Heart Disease Dataset)_Disease-Prediction-System-using-ML

## Project Description

The **Disease Prediction System using Machine Learning** aims to develop an intelligent system that predicts the likelihood of a person having a particular disease based on various health-related features. The system utilizes machine learning algorithms to analyze historical health data and make predictions, contributing to early disease detection and proactive healthcare management.

## Project Objectives

1. **Data Collection**:
   - Gather a diverse dataset containing relevant health features, including but not limited to age, gender, BMI, blood pressure, cholesterol levels, and family medical history.

2. **Data Preprocessing**:
   - Perform thorough data cleaning and preprocessing to handle missing values, outliers, and ensure data quality.
   - Normalize or standardize features to bring them to a consistent scale.

3. **Feature Selection**:
   - Employ feature selection techniques to identify the most influential variables for disease prediction.
   - Ensure that selected features contribute significantly to the accuracy of the machine learning models.

4. **Model Development**:
   - Explore and implement various machine learning algorithms such as logistic regression, decision trees, random forests, and support vector machines for disease prediction.
   - Evaluate and compare the performance of different models using metrics like accuracy, precision, recall, and F1-score.

5. **Cross-Validation**:
   - Implement cross-validation techniques to assess the generalization performance of the models and mitigate overfitting.

6. **Hyperparameter Tuning**:
   - Fine-tune the hyperparameters of selected machine learning models to optimize their performance.

7. **Model Interpretability** (Optional):
   - Enhance the interpretability of the models to provide insights into the factors influencing the predictions.
   - Use techniques such as SHAP (SHapley Additive exPlanations) values or feature importance plots.

8. **User Interface** (Optional):
   - Develop a user-friendly interface that allows users to input their health-related data and receive predictions about the likelihood of having a particular disease.

9. **Integration with Electronic Health Records (EHR)** (Optional):
   - Explore the integration of the disease prediction system with electronic health records, facilitating seamless information flow between healthcare providers and the system.

10. **Documentation** (Optional):
    - Provide comprehensive documentation covering data sources, methodology, model architecture, and instructions for using the prediction system.

11. **Validation and Testing**:
    - Conduct extensive testing and validation to ensure the accuracy, reliability, and robustness of the disease prediction system.

## Dataset

The dataset used in this project is sourced from the [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). It contains various health-related features such as age, gender, BMI, blood pressure, cholesterol levels, and family medical history, which are crucial for predicting the likelihood of heart disease.

### Features in the Dataset

- **Age**: Age of the patient
- **Gender**: Gender of the patient
- **BMI**: Body Mass Index of the patient
- **Blood Pressure**: Blood pressure levels
- **Cholesterol Levels**: Cholesterol levels in mg/dl
- **Family History**: Family history of heart disease
- **Target**: Presence of heart disease (1) or absence (0)

## Project Workflow

1. **Data Collection and Loading**:
   - Download the dataset and load it into a pandas DataFrame.

2. **Data Preprocessing**:
   - Handle missing values and outliers.
   - Normalize/standardize features.

3. **Feature Selection**:
   - Use feature selection techniques to identify important features.

4. **Model Development**:
   - Implement and train various machine learning models.
   - Evaluate models using cross-validation.

5. **Hyperparameter Tuning**:
   - Optimize the models by tuning their hyperparameters.

6. **Model Evaluation**:
   - Evaluate the models on the test set.
   - Print metrics such as accuracy, precision, recall, F1-score, and ROC AUC.
   - Plot confusion matrix and ROC curve.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Harshit-Soni78/Project-3-Disease-Prediction-System-using-Machine-Learning
   cd Project-3-Disease-Prediction-System-using-Machine-Learning

<hr><hr>

# 2. (UCI Pima Indians Diabetes Dataset)_Disease-Prediction-System-using-ML

# Disease Prediction System using Machine Learning

## Project Overview
The "Disease Prediction System using Machine Learning" project aims to develop an intelligent system that predicts the likelihood of a person having a particular disease based on various health-related features. The system utilizes machine learning algorithms to analyze historical health data and make predictions, contributing to early disease detection and proactive healthcare management.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [Model Interpretability](#model-interpretability)
- [User Interface](#user-interface)
- [Integration with Electronic Health Records (EHR)](#integration-with-electronic-health-records-ehr)
- [Validation and Testing](#validation-and-testing)

## Dataset
The dataset used for this project is the Pima Indians Diabetes Database, sourced from the UCI Machine Learning Repository. The dataset contains 768 instances with 8 features and 1 target variable indicating the presence of diabetes.

- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target:**
  - Outcome (0 or 1, indicating non-diabetic or diabetic)

The dataset is available [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv).

## Project Structure
The project files are organized as follows:
```
  ├── data
  │ └── pima-indians-diabetes.data.csv
  ├── notebooks
  │ └── disease_prediction_system.ipynb
  ├── app.py
  ├── random_forest_model.pkl
  ├── README.md
  └── requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Harshit-Soni78/Project-3-Disease-Prediction-System-using-Machine-Learning.git
   cd Project-3-Disease-Prediction-System-using-Machine-Learning
   ```
   
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
## Model Development
### Data Preprocessing
<ul><li> Missing values were handled for features: Glucose, BloodPressure, SkinThickness, Insulin, and BMI.</li>
<li> StandardScaler was used to normalize the features.</li>
</ul>

### Feature Selection
<ul><li> Select KBest with ANOVA F-test was used to select the top 5 features for model training.</li></ul>

### Model Training
<ul><li> Various models were trained and evaluated:</li>
<ul><li>Logistic Regression</li>
<li>Decision Tree</li>
<li>Random Forest</li>
<li>Support Vector Machine</li></ul>

### Hyperparameter Tuning
<ul><li> GridSearchCV was used to find the best hyperparameters for the Random Forest model.</li></ul>

## Model Interpretability
SHAP (SHapley Additive exPlanations) was used to explain the model predictions and understand feature importance.

## User Interface
A simple user interface was built using Streamlit, allowing users to input their health-related data and receive predictions about the likelihood of having a particular disease.

## Integration with Electronic Health Records (EHR)
Example code is provided to demonstrate how the system can be integrated with EHR systems using API requests.

## Validation and Testing
The model was validated and tested using various metrics:
<ul>
<li>Accuracy</li>
<li>Precision</li>
<li>Recall</li>
<li>F1 Score</li>
<li>Cross-Validation Accuracy</li></ul>

## Contributing
*Contributions are welcome! Please create a pull request or open an issue to discuss any changes.*
