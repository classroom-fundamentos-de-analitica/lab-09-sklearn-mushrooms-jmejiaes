import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_datasets():
    """Load train and test datasets."""
    train_dataset = pd.read_csv("train_dataset.csv")
    test_dataset = pd.read_csv("test_dataset.csv")
    
    x_train = train_dataset.drop("type", axis=1)
    y_train = train_dataset["type"]
    
    x_test = test_dataset.drop("type", axis=1)
    y_test = test_dataset["type"]

    return x_train, x_test, y_train, y_test

def create_pipeline(x_train):
    """Create a pipeline for preprocessing and model training."""
    # Extract the categorical columns to apply one-hot encoding
    categorical_columns = x_train.select_dtypes(include=['object']).columns.tolist()
    
    # Define the ColumnTransformer to apply one-hot encoding to the categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_columns)
        ], remainder='passthrough'  # Leave other columns (like numeric ones) as they are
    )
    
    # Create a pipeline with the preprocessing step and the logistic regression model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    return pipeline

def train_model(pipeline, x_train, y_train):
    """Train the pipeline model."""
    pipeline.fit(x_train, y_train)
    return pipeline

def save_pipeline(pipeline):
    """Save the entire pipeline (including preprocessing) to a pickle file."""
    with open("model.pkl", "wb") as file:
        pickle.dump(pipeline, file)

def load_pipeline():
    """Load the pipeline from a pickle file."""
    with open("model.pkl", "rb") as file:
        pipeline = pickle.load(file)
    return pipeline

def evaluate_model(pipeline, x_train, x_test, y_train, y_test):
    """Evaluate the model performance."""
    # Predict on the training set
    y_pred_train = pipeline.predict(x_train)
    y_pred_test = pipeline.predict(x_test)
    
    # Calculate accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    print(f"Train accuracy: {accuracy_train:.2f}")
    print(f"Test accuracy: {accuracy_test:.2f}")

if __name__ == "__main__":
    # Load datasets
    x_train, x_test, y_train, y_test = load_datasets()
    
    # Create a pipeline with preprocessing and the logistic regression model
    pipeline = create_pipeline(x_train)
    
    # Train the model
    trained_pipeline = train_model(pipeline, x_train, y_train)
    
    # Save the entire pipeline (including preprocessing) to a .pkl file
    save_pipeline(trained_pipeline)
    
    # Load the pipeline from the .pkl file (for demonstration purposes)
    loaded_pipeline = load_pipeline()
    
    # Evaluate the model using the loaded pipeline
    evaluate_model(loaded_pipeline, x_train, x_test, y_train, y_test)
