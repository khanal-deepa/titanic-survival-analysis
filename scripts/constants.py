from pathlib import Path

TEST_DATA_SIZE = 0.2

class PathConfig:
    """Handles all directory and file paths."""

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data/processed"
    MODEL_DIR = BASE_DIR / "models/saved_models"

    # Ensure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Model filenames
    LOGISTIC_REGRESSION_MODEL = MODEL_DIR / "logistic_regression.pkl"
    RANDOM_FOREST_MODEL = MODEL_DIR / "random_forest.pkl"
    NEURAL_NETWORK_MODEL = MODEL_DIR / "neural_network.h5"


class LogisticRegressionConfig:
    """Configuration for Logistic Regression model."""
    MAX_ITER = 1000


class RandomForestConfig:
    """Configuration for Random Forest model."""
    N_ESTIMATORS = 100
    RANDOM_STATE = 42


class NeuralNetworkConfig:
    """Configuration for Neural Network model."""
    HIDDEN_LAYER_1_UNITS = 64
    HIDDEN_LAYER_2_UNITS = 32
    OUTPUT_UNITS = 1
    ACTIVATION_HIDDEN = "relu"
    ACTIVATION_OUTPUT = "sigmoid"
    LOSS = "binary_crossentropy"
    OPTIMIZER = "adam"
    EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 0.0001