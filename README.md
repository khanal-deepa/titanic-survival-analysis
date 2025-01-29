# titanic-survival-analysis

# Titanic Survival Prediction

## Project Overview

This project aims to predict the survival of Titanic passengers using supervised machine learning models. The dataset contains various features like passenger class, age, sex, and other personal details, and the goal is to predict whether a passenger survived or not based on these features.

The following machine learning models have been implemented and evaluated:

- *Logistic Regression*
- *Random Forest Classifier*
- *Neural Network* (using TensorFlow)

## Table of Contents

1. [Installation](#installation)
2. [Data](#data)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Run the Code](#run-the-code)
7. [Dependencies](#dependencies)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/khanal-deepa/titanic-survival-analysis.git
cd titanic-survival-prediction
pip install -r requirements.txt

## Model Evaluation

Below is the performance comparison of the different models used for predicting the survival of Titanic passengers. The metrics include accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix for each model.

| Model            | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix   |
|------------------|----------|-----------|--------|----------|---------|--------------------|
| *Logistic Regression* | 0.8101   | 0.7857    | 0.7432 | 0.7639   | 0.8002  | [[90 15] [19 55]] |
| *Random Forest*       | 0.8156   | 0.7971    | 0.7432 | 0.7692   | 0.8050  | [[91 14] [19 55]] |
| *Neural Network*      | 0.8324   | 0.8548    | 0.7162 | 0.7794   | 0.8153  | [[96  9] [21 53]] |

### Observations:
- *Neural Network* has the highest accuracy, precision, and ROC-AUC score, indicating better overall performance compared to the other models.
- *Random Forest* shows a balanced performance with a slight increase in precision compared to Logistic Regression.
- *Logistic Regression* maintains a solid performance but has the lowest F1-score among the three models.

These metrics provide a comprehensive comparison, highlighting the strengths and weaknesses of each model for this task.
