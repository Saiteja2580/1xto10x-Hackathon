# Heart Disease Prediction Project

## Overview

This project aims to predict the likelihood of heart disease using machine learning models. It involves data exploration, preprocessing, model training, evaluation, and deployment.

## Libraries Used

The following libraries are essential for running this project:

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (including LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, SVC)
- joblib

To install these libraries, you can use pip:


## Process

1. **Data Loading and Exploration:** The dataset is loaded into a pandas DataFrame, and its structure, data types, descriptive statistics, missing values, and unique values are examined.
2. **Data Preprocessing:** This step may involve handling missing values, encoding categorical features, and scaling numerical features. However, in this specific project, the dataset is assumed to be preprocessed.
3. **Model Training and Evaluation:** Different machine learning models are trained on the data, and their performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC AUC. Hyperparameter tuning is performed using GridSearchCV.
4. **Model Saving:** The trained models are saved using joblib for later use.
5. **Prediction:** New data can be used to make predictions using the loaded models.


## How to Load the Models

The trained models are saved in the "models" folder as pickle files. You can load them using joblib:

Replace "Logistic Regression.pkl" with the name of the desired model file.

## How to Run the Scripts

1. **Install Libraries:** Install the necessary libraries as mentioned in the "Libraries Used" section.
2. **Download Dataset:** Download the dataset used for this project. Place it in the same directory or adjust the file path in the code accordingly.
3. **Execute Notebook:** Run the Jupyter Notebook or Python script containing the code.
4. **Make Predictions:** Load a model and use it to make predictions on new data as shown in the "How to Load the Models" section.

## Notes

- Ensure that the dataset file is in the correct location or specify the file path in the code.
- Models may be replaced or updated over time. Use the model name listed in `models` folder for prediction.
- Feel free to experiment with different models and hyperparameters for improving the results.
