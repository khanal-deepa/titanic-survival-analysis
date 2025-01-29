from preprocessing import DataProcessor
from model_training import model_train
from model_evaluation import ModelEvaluator
from constants import PathConfig

def get_model_evaluation(use_existing_data=True, use_existing_model=True):
    """
    Allows the user to choose between using existing preprocessed data and trained models
    or preprocessing and training from scratch before evaluation.

    Args:
        use_existing_data (bool): If True, uses preprocessed data; otherwise, preprocesses new data.
        use_existing_model (bool): If True, uses an existing trained model; otherwise, trains a new model.
    """
    
    if not use_existing_data:
        print("Preprocessing new data...")
        processor = DataProcessor(PathConfig.BASE_DIR / "data/titanic.csv")
        processor.process()
    else:
        print("Using existing preprocessed data...")
    
    if not use_existing_model:
        print("Training a new model...")
        model_train()
    else:
        print("Using existing trained model...")
    
    print("Running model evaluation...")
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    user_choice_data = input("Do you want to preprocess new data? (yes/no): ").strip().lower() == 'yes'
    user_choice_model = input("Do you want to train a new model? (yes/no): ").strip().lower() == 'yes'
    
    get_model_evaluation(use_existing_data=not user_choice_data, use_existing_model=not user_choice_model)