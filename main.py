# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten

# Generating synthetic dataset
def create_synthetic_data(filename, num_samples=1000, num_features=10):
    """
    Function to generate synthetic data and save it to a CSV file.

    Args:
        filename (str): Name of the CSV file to save the data.
        num_samples (int): Number of samples to generate.
        num_features (int): Number of features for each sample.
    """
    data = np.random.randn(num_samples, num_features)
    target = np.random.randint(0, 2, size=(num_samples, 1))

    columns = [f'feature_{i}' for i in range(num_features)] + ['target']
    df = pd.DataFrame(np.hstack((data, target)), columns=columns)

    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

create_synthetic_data('data.csv')

# Class for loading data
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Method to load data from a CSV file.

        Returns:
            pandas.DataFrame: Loaded DataFrame.
        """
        data = pd.read_csv(self.file_path)
        data = data.dropna()
        return data

# Class for data preprocessing
class Preprocessor:
    @staticmethod
    def normalize_data(data):
        """
        Method to normalize data.

        Args:
            data (pandas.DataFrame): Input data.

        Returns:
            pandas.DataFrame: Normalized data.
        """
        return (data - data.mean()) / data.std()

    @staticmethod
    def split_data(data, target_column, test_size=0.2):
        """
        Method to split data into training and testing sets.

        Args:
            data (pandas.DataFrame): Input data.
            target_column (str): Name of the target column.
            test_size (float): Size of the test set.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

# Class for building models
class ModelBuilder:
    @staticmethod
    def build_lstm_model(input_shape):
        """
        Method to build an LSTM model.

        Args:
            input_shape (tuple): Input shape for the model.

        Returns:
            tensorflow.keras.Sequential: Compiled LSTM model.
        """
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=input_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def build_cnn_model(input_shape):
        """
        Method to build a CNN model.

        Args:
            input_shape (tuple): Input shape for the model.

        Returns:
            tensorflow.keras.Sequential: Compiled CNN model.
        """
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

# Class for training models
class Trainer:
    @staticmethod
    def train_model(model, X_train, y_train, epochs, batch_size):
        """
        Method to train the model.

        Args:
            model (tensorflow.keras.Sequential): Model to train.
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.

        Returns:
            tensorflow.keras.Sequential: Trained model.
        """
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()
        return model

# Class for making predictions
class Predictor:
    @staticmethod
    def make_predictions(model, X):
        """
        Method to make predictions using the trained model.

        Args:
            model (tensorflow.keras.Sequential): Trained model.
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted values.
        """
        predictions = model.predict(X)
        return predictions

    @staticmethod
    def save_predictions(predictions, file_path):
        """
        Method to save predictions to a file.

        Args:
            predictions (numpy.ndarray): Predicted values.
            file_path (str): Path to save the predictions.
        """
        np.savetxt(file_path, predictions, delimiter=",")

    @staticmethod
    def plot_predictions(predictions, y_true):
        """
        Method to plot the true values against the predicted values.

        Args:
            predictions (numpy.ndarray): Predicted values.
            y_true (numpy.ndarray): True values.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='True Values')
        plt.plot(predictions, label='Predictions')
        plt.legend()
        plt.show()

# Main function to orchestrate the entire process
def main(args):
    # Loading data
    data_loader = DataLoader(args.data_path)
    data = data_loader.load_data()

    # Preprocessing data
    preprocessor = Preprocessor()
    data = preprocessor.normalize_data(data)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = preprocessor.split_data(data, target_column=args.target_column)
    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = (X_train.shape[1], 1)

    # Building the specified model type
    if args.model_type == "LSTM":
        model = ModelBuilder.build_lstm_model(input_shape)
    elif args.model_type == "CNN":
        model = ModelBuilder.build_cnn_model(input_shape)
    else:
        raise ValueError("Invalid model type. Choose 'LSTM' or 'CNN'.")

    # Training the model if specified
    if args.train:
        trainer = Trainer()
        model = trainer.train_model(model, X_train, y_train, args.epochs, args.batch_size)
        model.save(args.model_save_path)

    # Loading the trained model
    model = tf.keras.models.load_model(args.model_save_path)

    # Making predictions
    predictor = Predictor()
    predictions = predictor.make_predictions(model, X_test)

    # Saving predictions to a file
    predictor.save_predictions(predictions, args.predictions_save_path)

    # Plotting predictions
    predictor.plot_predictions(predictions, y_test)


if __name__ == "__main__":
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--target_column', type=str, required=True, help='Name of the target column')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--predictions_save_path', type=str, required=True, help='Path to save the predictions')
    parser.add_argument('--model_type', type=str, choices=['LSTM', 'CNN'], required=True, help='Type of model to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    args = parser.parse_args()
    main(args)

