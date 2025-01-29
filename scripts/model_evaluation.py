import os
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from constants import PathConfig


class ModelEvaluator:
    """
    A class to evaluate machine learning models and save the results.

    Attributes:
        model_dir (Path): Directory where trained models are stored.
        data_dir (Path): Directory where test data is stored.
    """
    def __init__(self, model_dir=PathConfig.MODEL_DIR, data_dir=PathConfig.DATA_DIR):
        """
        Initializes the ModelEvaluator with directories for models and data.

        Args:
            model_dir (Path): Directory containing trained models. Defaults to PathConfig.MODEL_DIR.
            data_dir (Path): Directory containing test data. Defaults to PathConfig.DATA_DIR.
        """
            
        self.model_dir = model_dir
        self.data_dir = data_dir

    def evaluate_model(self, model, X_test, y_test, model_type):
        """
        Evaluates the performance of a given model.

        Args:
            model: The trained model to evaluate.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test data.
            model_type (str): Type of the model (e.g., 'Logistic Regression', 'Random Forest', 'Neural Network').

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        # Predict using the model
        if model_type == "Neural Network":
            y_pred = (model.predict(X_test) > 0.5).astype(int)  # Threshold predictions for binary classification
        else:
            y_pred = model.predict(X_test)  # Predict directly for non-NN models

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Return metrics as a dictionary
        return {
            "Model": model_type,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Confusion Matrix": conf_matrix,
        }

    def load_data(self):
        """
        Loads test data from the specified directory.

        Returns:
            tuple: A tuple containing X_test (features) and y_test (labels).
        """
        print("Loading test data........")
        X_test = pd.read_csv(os.path.join(self.data_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(self.data_dir, "y_test.csv")).values.ravel()
        return X_test, y_test

    def load_models(self):
        """
        Loads trained models from the specified directory.

        Returns:
            tuple: A tuple containing the loaded models (Logistic Regression, Random Forest, Neural Network).
        """
        lr_model = joblib.load(os.path.join(self.model_dir, "logistic_regression.pkl"))
        rf_model = joblib.load(os.path.join(self.model_dir, "random_forest.pkl"))
        nn_model = tf.keras.models.load_model(os.path.join(self.model_dir, "neural_network.h5"))
        return lr_model, rf_model, nn_model

    def save_results(self, results_df):
        """
        Saves evaluation results to a CSV file.

        Args:
            results_df (pd.DataFrame): Pandas DataFrame containing evaluation results.
        """
        results_df.to_csv(os.path.join(self.data_dir, "evaluation_results.csv"), index=False)
        print("Evaluation results saved to 'evaluation_results.csv'.")

    def run_evaluation(self):
        """
        Runs the evaluation process for all models and saves the results.
        """
        print("Model evaluation started....")
        # Load test data
        X_test, y_test = self.load_data()

        # Load trained models
        lr_model, rf_model, nn_model = self.load_models()

        # Evaluate models and store results in a list
        results = [
            self.evaluate_model(lr_model, X_test, y_test, "Logistic Regression"),
            self.evaluate_model(rf_model, X_test, y_test, "Random Forest"),
            self.evaluate_model(nn_model, X_test, y_test, "Neural Network"),
        ]

        # Convert the list of results to a Pandas DataFrame
        results_df = pd.DataFrame(results)

        # Display the DataFrame (table)
        print("Evaluation Results:")
        print(results_df)

        # Save the results to a CSV file
        self.save_results(results_df)

        print("Evaluation completed.")


if __name__ == "__main__":
    # Create an instance of ModelEvaluator and run the evaluation
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()