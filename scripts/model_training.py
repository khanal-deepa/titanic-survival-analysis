import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from constants import PathConfig, LogisticRegressionConfig, RandomForestConfig, NeuralNetworkConfig


class ModelTrainer:
    """
    A base class to define the structure for different model trainers.
    """
    
    def __init__(self):
        """
        Initializes the ModelTrainer class and ensures model directory exists.
        """
        os.makedirs(PathConfig.MODEL_DIR, exist_ok=True)

    def train(self, X_train, y_train):
        """
        Abstract train method to be implemented by subclasses.
        """
        raise NotImplementedError("Train method must be implemented by subclasses")


class LogisticRegressionTrainer(ModelTrainer):
    """
    A class to train and save the logistic regression model.
    """
    
    def train(self, X_train, y_train):
        model = LogisticRegression(max_iter=LogisticRegressionConfig.MAX_ITER)
        model.fit(X_train, y_train)
        joblib.dump(model, PathConfig.LOGISTIC_REGRESSION_MODEL)
        return model


class RandomForestTrainer(ModelTrainer):
    """
    A class to train and save the random forest model.
    """
    
    def train(self, X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=RandomForestConfig.N_ESTIMATORS,
            random_state=RandomForestConfig.RANDOM_STATE
        )
        model.fit(X_train, y_train)
        joblib.dump(model, PathConfig.RANDOM_FOREST_MODEL)
        return model


class NeuralNetworkTrainer(ModelTrainer):
    """
    A class to train and save the neural network model.
    """
    
    def train(self, X_train, y_train):
        model = Sequential([
            Dense(NeuralNetworkConfig.HIDDEN_LAYER_1_UNITS, activation=NeuralNetworkConfig.ACTIVATION_HIDDEN, input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(NeuralNetworkConfig.HIDDEN_LAYER_2_UNITS, activation=NeuralNetworkConfig.ACTIVATION_HIDDEN),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(NeuralNetworkConfig.OUTPUT_UNITS, activation=NeuralNetworkConfig.ACTIVATION_OUTPUT)
        ])
        
        optimizer = Adam(learning_rate=NeuralNetworkConfig.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=NeuralNetworkConfig.LOSS, metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=NeuralNetworkConfig.EPOCHS, batch_size=NeuralNetworkConfig.BATCH_SIZE, verbose=1)
        model.save(PathConfig.NEURAL_NETWORK_MODEL)
        return model


def model_train():
    """
    Main function to load data, train models, and save them.
    """
    # Load preprocessed training data
    X_train = pd.read_csv(PathConfig.DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(PathConfig.DATA_DIR / "y_train.csv")
    
    trainers = [
        LogisticRegressionTrainer(),
        RandomForestTrainer(),
        NeuralNetworkTrainer()
    ]
    
    for trainer in trainers:
        print(f"Training {trainer.__class__.__name__}...")
        trainer.train(X_train, y_train)
        print(f"{trainer.__class__.__name__} training completed.")

