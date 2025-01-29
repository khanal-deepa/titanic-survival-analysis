import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from pathlib import Path
from constants import PathConfig, TEST_DATA_SIZE, RandomForestConfig

class DataProcessor:
    """
    A single class responsible for loading, preprocessing, splitting, and saving data.
    It follows the SOLID principles for maintainability and scalability.
    
    Attributes:
        filepath (str): The path to the dataset CSV file.
        data (pd.DataFrame): The loaded dataset.
        X_train, X_test, y_train, y_test (pd.DataFrame): Split training and testing sets.
        label_encoder (LabelEncoder): Encoder for categorical features.
        scaler (StandardScaler): Scaler for numerical features.
    """
    
    def __init__(self, filepath):
        """Initializes the DataProcessor with the file path and necessary tools."""
        self.filepath = filepath
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def load_data(self):
        """Loads data from a CSV file into a Pandas DataFrame."""
        self.data = pd.read_csv(self.filepath)

    def preprocess_data(self):
        """
        Handles missing values, encodes categorical features, and scales numerical values.
        Steps:
        - Fills missing values for 'Age' and 'Embarked'.
        - Drops unnecessary columns ('Cabin', 'Ticket', 'Name', 'PassengerId').
        - Encodes categorical features ('Sex' and 'Embarked') using LabelEncoder.
        - Scales numerical features ('Age' and 'Fare') using StandardScaler.
        """
        self.data['Age'].fillna(self.data['Age'].median(), inplace=True)
        self.data['Embarked'].fillna(self.data['Embarked'].mode()[0], inplace=True)
        
        # Dropping columns that are not relevant for the model
        self.data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
        
        # Encoding categorical features into numerical values
        self.data['Sex'] = self.label_encoder.fit_transform(self.data['Sex'])
        self.data['Embarked'] = self.label_encoder.fit_transform(self.data['Embarked'])
        
        # Scaling numerical features to normalize the distribution
        self.data[['Age', 'Fare']] = self.scaler.fit_transform(self.data[['Age', 'Fare']])

    def split_data(self, test_size=TEST_DATA_SIZE , random_state= RandomForestConfig.RANDOM_STATE):
        """
        Splits the dataset into training and testing sets.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        """
        X = self.data.drop('Survived', axis=1)  # Features
        y = self.data['Survived']  # Target variable
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_data(self):
        """
        Saves processed datasets to CSV files in the designated directory.
        Ensures the target directory exists before saving.
        """
        os.makedirs(PathConfig.DATA_DIR, exist_ok=True)
        
        self.X_train.to_csv(PathConfig.DATA_DIR / "X_train.csv", index=False)
        self.X_test.to_csv(PathConfig.DATA_DIR / "X_test.csv", index=False)
        self.y_train.to_csv(PathConfig.DATA_DIR / "y_train.csv", index=False)
        self.y_test.to_csv(PathConfig.DATA_DIR / "y_test.csv", index=False)
        
        print("Data preprocessing and saving completed.")

    def process(self):
        """
        Executes the full pipeline: loading, preprocessing, splitting, and saving data.
        """
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.save_data()

if __name__ == "__main__":
    # Define processor instance with the dataset path and execute processing
    processor = DataProcessor(PathConfig.BASE_DIR / "data/titanic.csv")
    processor.process()
