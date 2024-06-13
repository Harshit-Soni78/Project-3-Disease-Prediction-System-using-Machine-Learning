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
