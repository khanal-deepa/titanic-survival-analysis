# titanic-survival-analysis

## Project Overview

This project aims to predict the survival of Titanic passengers using supervised machine learning models. The dataset contains various features like passenger class, age, sex, and other personal details, and the goal is to predict whether a passenger survived or not based on these features.

The following machine learning models have been implemented and evaluated:

- *Logistic Regression*
- *Random Forest Classifier*
- *Neural Network* (using TensorFlow)

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Model Evaluation](#model-evaluation)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Findings](#findings)


## Requirements

Make sure you have Python 3.8 installed before running the project.

```bash
python --version  # Should output Python 3.8.x
```
## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/khanal-deepa/titanic-survival-analysis.git
pip install -r requirements.txt
cd scripts
```
```bash
python main.py
```
When you run the script, you will be asked two questions:

1️⃣ "Do you want to preprocess new data? (yes/no)"

Type yes → The script will preprocess fresh data.

Type no → The script will use existing preprocessed data.

2️⃣ "Do you want to train a new model? (yes/no)"

Type yes → The script will train a new model from scratch.

Type no → The script will use an existing trained model.


## Model Evaluation

Below is the sample performance comparison of the different models used for predicting the survival of Titanic passengers. The metrics include accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix for each model.

| Model            | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix   |
|------------------|----------|-----------|--------|----------|---------|--------------------|
| *Logistic Regression* | 0.8100   | 0.7857    | 0.7432 | 0.7639   | 0.8001  | [[90 15] [19 55]] |
| *Random Forest*       | 0.8156   | 0.7971    | 0.7432 | 0.7692   | 0.8050  | [[91 14] [19 55]] |
| *Neural Network*      | 0.8324   | 0.8333    | 0.7432 | 0.7857   | 0.8192  | [[94  11] [19 55]] |

### Observations:
- *Neural Network* has the highest accuracy, precision,F1-score and ROC-AUC score, indicating better overall performance compared to the other models.
- *Neural Network* has the highest accuracy, precision,F1-score and ROC-AUC score, indicating better overall performance compared to the other models.
- *Random Forest* shows a balanced performance with a slight increase in precision compared to Logistic Regression.
- *Logistic Regression* maintains a solid performance but has the lowest accuracy among the three models.

These metrics provide a comprehensive comparison, highlighting the strengths and weaknesses of each model for this task.

## Key Terms:
- **Accuracy**: The percentage of correctly predicted samples out of the total samples. Higher values indicate better performance.
  
- **Precision**: The proportion of true positive predictions among all predicted positives. Measures how many predicted positives are actually correct.

- **Recall (Sensitivity)**: The proportion of actual positive cases that were correctly identified by the model. Measures how well the model identifies positive cases.

- **F1-Score**: The harmonic mean of Precision and Recall, providing a balance between both metrics.

- **ROC-AUC Score**: The Area Under the Receiver Operating Characteristic Curve. A higher value (closer to 1) indicates better classification performance.

- **Confusion Matrix**: A table that summarizes prediction outcomes:
  - **True Negative (TN)**: Correctly predicted negative cases.
  - **False Positive (FP)**: Incorrectly predicted positive cases.
  - **False Negative (FN)**: Incorrectly predicted negative cases.
  - **True Positive (TP)**: Correctly predicted positive cases.




